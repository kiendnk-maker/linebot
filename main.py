"""
Groq哥哥 LINE Bot — main.py
Spec version: 2026-03-16
"""

import os, base64, aiosqlite, httpx, re, json, asyncio, time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Taipei")

from groq import AsyncGroq
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from linebot.v3 import WebhookParser
from linebot.v3.messaging import (
    AsyncApiClient, AsyncMessagingApi, AsyncMessagingApiBlob, Configuration,
    ReplyMessageRequest, PushMessageRequest, TextMessage, ShowLoadingAnimationRequest,
    QuickReply, QuickReplyItem, MessageAction,
)
from linebot.v3.webhooks import (
    MessageEvent, TextMessageContent, ImageMessageContent,
    AudioMessageContent, FileMessageContent,
)
from prompts import get_system_prompt
import chromadb
import pdfplumber
import io
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
LINE_CHANNEL_SECRET       = os.environ["LINE_CHANNEL_SECRET"]
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
GROQ_API_KEY              = os.environ["GROQ_API_KEY"]
client = AsyncGroq(api_key=GROQ_API_KEY)  # Khởi tạo client dùng chung cho các Agent
CHROMA_PATH               = os.environ.get("CHROMA_PATH", "chroma")



WHISPER_MODEL   = "whisper-large-v3-turbo"
MAX_INPUT_CHARS = 8000
SUMMARY_TRIGGER = 20

# Trigger prefixes that activate LLM reply from voice messages
_REPLY_TRIGGERS = ("hello", "不好意思")

# ---------------------------------------------------------------------------
# MODEL REGISTRY
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict] = {
    "llama8b": {
        "model_id": "llama-3.1-8b-instant",
        "type":    "text",
        "tier":    "production",
        "display": "LLaMA 3.1 8B",
        "ctx":     131_072,
        "note":    "最快 ~900 t/s，超便宜 — 用於 classifier 和簡單對話",
    },
    "llama70b": {
        "model_id": "llama-3.3-70b-versatile",
        "type":    "text",
        "tier":    "production",
        "display": "LLaMA 3.3 70B",
        "ctx":     131_072,
        "note":    "均衡性能，寫作、翻譯、多語言",
    },
    "gpt20b": {
        "model_id": "openai/gpt-oss-20b",
        "type":    "reasoning",
        "tier":    "production",
        "display": "GPT-OSS 20B",
        "ctx":     131_072,
        "note":    "超快 ~1000 t/s，輕量推理",
    },
    "gpt120b": {
        "model_id": "openai/gpt-oss-120b",
        "type":    "reasoning",
        "tier":    "production",
        "display": "GPT-OSS 120B",
        "ctx":     131_072,
        "note":    "最強推理，~500 t/s",
    },
    "compound": {
        "model_id": "groq/compound",
        "type":    "text",
        "tier":    "production",
        "display": "Groq Compound（網路搜尋）",
        "ctx":     131_072,
        "note":    "內建網路搜尋 + 程式執行，最多10次工具呼叫",
    },
    "compound-mini": {
        "model_id": "groq/compound-mini",
        "type":    "text",
        "tier":    "production",
        "display": "Groq Compound Mini",
        "ctx":     131_072,
        "note":    "Compound 輕量版，單次工具呼叫，速度快3倍",
    },
    "scout": {
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "type":    "vision",
        "tier":    "preview",
        "display": "LLaMA 4 Scout（視覺）",
        "ctx":     131_072,
        "note":    "唯一視覺模型，低延遲",
    },
    "qwen": {
        "model_id": "qwen/qwen3-32b",
        "type":    "reasoning",
        "tier":    "preview",
        "display": "Qwen3 32B",
        "ctx":     131_072,
        "note":    "推理 + thinking mode，多語言強",
    },
    "kimi": {
        "model_id": "moonshotai/kimi-k2-instruct-0905",
        "type":    "text",
        "tier":    "preview",
        "display": "Kimi K2 0905",
        "ctx":     262_144,
        "note":    "最長 context 256K，agentic coding",
    },
}

DEFAULT_MODEL_KEY    = "llama70b"
VISION_MODEL_KEY     = "scout"
CLASSIFIER_MODEL_KEY = "llama8b"

# ---------------------------------------------------------------------------
# AUTO-ROUTER
# ---------------------------------------------------------------------------
ROUTE_MAP: dict[str, str] = {
    "simple":    "llama8b",
    "creative":  "llama70b",
    "reasoning": "qwen",
    "hard":      "gpt120b",
    "search":    "compound-mini",
}

_REALTIME_KEYWORDS = (
    "hôm nay", "bây giờ", "hiện tại", "mới nhất", "today", "now", "latest",
    "giá ", "price", "tỷ giá", "thời tiết", "weather", "tin tức", "news",
    "今天", "現在", "最新", "價格", "新聞", "天氣",
)

_CLASSIFIER_PROMPT = """Classify the user message into exactly one category.
Reply with ONLY one word from this list, nothing else.

Categories:
- simple    : greetings, chitchat, yes/no, very short factual (name, date)
- creative  : writing, translation, summarization, brainstorming, roleplay,
              long text analysis, summarize this, tóm tắt, phân tích đoạn văn
- reasoning : math, logic, code, step-by-step analysis, comparison, explanation
- hard      : ambiguous complex questions, multi-domain, requires deep thinking
- search    : current events, prices, news, weather, "latest", "now", "today"
              ONLY use this if the question requires real-time internet data

Message: {message}"""


def _needs_realtime(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in _REALTIME_KEYWORDS)


async def classify_query(user_text: str) -> str:
    async with httpx.AsyncClient() as http:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)
        try:
            resp = await client.chat.completions.create(
                model=MODEL_REGISTRY[CLASSIFIER_MODEL_KEY]["model_id"],
                messages=[{
                    "role": "user",
                    "content": _CLASSIFIER_PROMPT.format(message=user_text[:400]),
                }],
                temperature=0.0,
                max_tokens=3,
            )
            category = resp.choices[0].message.content.strip().lower()
            return ROUTE_MAP.get(category, DEFAULT_MODEL_KEY)
        except Exception:
            return DEFAULT_MODEL_KEY


async def resolve_model(user_id: str, user_text: str) -> tuple[str, str]:
    """
    Priority order (per SPEC §4 + flow.md):
      1. User manually set model (non-default) → use that
      2. _needs_realtime → compound-mini   (check BEFORE len>500 to avoid false positive)
      3. Text > 500 chars, no ? → llama70b (summarize)
      4. classify_query via llama8b → ROUTE_MAP
    """
    current_key = await get_user_model(user_id)
    if current_key != DEFAULT_MODEL_KEY:
        return current_key, MODEL_REGISTRY[current_key]["model_id"]

    # Priority 2: realtime check BEFORE len>500 (avoid "hôm nay" in reminder context — see SPEC §16.7)
    if _needs_realtime(user_text):
        return "compound-mini", MODEL_REGISTRY["compound-mini"]["model_id"]

    # Priority 3: long text without question → summarize
    if len(user_text) > 500 and "?" not in user_text and "？" not in user_text:
        return "llama70b", MODEL_REGISTRY["llama70b"]["model_id"]

    # Priority 4: classifier
    routed_key = await classify_query(user_text)
    return routed_key, MODEL_REGISTRY[routed_key]["model_id"]


# ---------------------------------------------------------------------------
# RAG — per-user ChromaDB lock registry
# ---------------------------------------------------------------------------
_chroma_locks: dict[str, asyncio.Lock] = {}
_chroma_locks_meta: asyncio.Lock | None = None   # initialized lazily inside event loop









# ---------------------------------------------------------------------------
# RAG — in-memory RAG-off toggle (per session, resets on restart)
# ---------------------------------------------------------------------------
_rag_disabled: set[str] = set()   # user_ids with RAG disabled


# ---------------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DATABASE — user settings helpers
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# DATABASE — user profile helpers
# ---------------------------------------------------------------------------




async def build_system_prompt(user_id: str, model_key: str) -> str:
    base    = get_system_prompt(model_key)
    profile = await get_user_profile(user_id)
    if not profile:
        return base
    lines: list[str] = []
    if profile.get("name"):       lines.append("用戶姓名：" + profile["name"])
    if profile.get("occupation"): lines.append("職業："     + profile["occupation"])
    if profile.get("learning"):   lines.append("正在學習：" + profile["learning"])
    if profile.get("notes"):      lines.append("備註："     + profile["notes"])
    return base + "\n\n【用戶資料】\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# DATABASE — history helpers
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# SUMMARY MEMORY
# ---------------------------------------------------------------------------




async def maybe_summarize(user_id: str) -> None:
    """Auto-summarize every SUMMARY_TRIGGER messages."""
    count = await count_history(user_id)
    if count % SUMMARY_TRIGGER != 0:
        return

    rows = await get_history_raw(user_id, limit=40)
    if not rows:
        return

    prev_summary = await get_summary(user_id)
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in rows)
    prompt = (
        f"Tóm tắt ngắn gọn (tối đa 150 từ) nội dung cuộc trò chuyện sau, "
        f"giữ lại thông tin quan trọng về người dùng và chủ đề chính.\n"
        f"Tóm tắt trước đó: {prev_summary}\n\n"
        f"Hội thoại mới:\n{history_text}"
    )

    async with httpx.AsyncClient() as http:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)
        try:
            resp = await client.chat.completions.create(
                model=MODEL_REGISTRY["llama8b"]["model_id"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            new_summary = resp.choices[0].message.content or ""
            await save_summary(user_id, new_summary)
        except Exception:
            pass




async def get_history_with_summary(user_id: str) -> list[dict]:
    """Return [summary_pair] + 5 most recent messages."""
    summary = await get_summary(user_id)
    recent  = await get_history_raw(user_id, limit=5)
    if summary:
        return [
            {"role": "user",      "content": f"[Tóm tắt hội thoại trước: {summary}]"},
            {"role": "assistant", "content": "Đã hiểu context."},
            *recent,
        ]
    return recent


# ---------------------------------------------------------------------------
# REMINDER SYSTEM
# ---------------------------------------------------------------------------






def _next_fire(fire_at: int, repeat: str) -> int:
    """Calculate next fire timestamp for repeating reminders — no dateutil dependency."""
    dt = datetime.fromtimestamp(fire_at, tz=TZ)
    if repeat == "daily":
        dt += timedelta(days=1)
    elif repeat == "weekly":
        dt += timedelta(weeks=1)
    elif repeat == "monthly":
        month = dt.month + 1
        year  = dt.year + (1 if month > 12 else 0)
        month = month if month <= 12 else 1
        dt    = dt.replace(year=year, month=month)
    return int(dt.timestamp())


# LLM returns HH:MM + DD/MM/YYYY → Python calculates fire_at to avoid TZ errors
_PARSE_REMINDER_PROMPT = """Extract reminder info from the user message.
Current datetime (UTC+8): {now_str}

Reply ONLY with JSON, no explanation:
{{"is_reminder": true/false, "message": "reminder content", "time": "HH:MM or null", "date": "DD/MM/YYYY", "repeat": null or "daily" or "weekly" or "monthly"}}

Rules:
- Return time as HH:MM 24h format. If no specific time return null.
- Convert Vietnamese/AM/PM: 7h toi/chieu/evening=19:00, 8h toi=20:00, 7h sang/morning=07:00, 12h trua/noon=12:00, 12h dem/midnight=00:00
- Return date as DD/MM/YYYY
- hom nay/tonight/today = {date_str}
- ngay mai/tomorrow = {tomorrow_str}
- moi ngay/every day/daily = repeat=daily
- moi tuan/every week/weekly = repeat=weekly
- moi thang/every month/monthly = repeat=monthly
- hom nay/tonight/today = {date_str}
- ngay mai/tomorrow = {tomorrow_str}
- moi ngay/every day/daily = repeat=daily
- moi tuan/every week/weekly = repeat=weekly
- moi thang/every month/monthly = repeat=monthly
- Set is_reminder=true if user mentions scheduled event with time even without explicit remind keyword.
- IMPORTANT - Examples of is_reminder=true:
    "cuoc hen luc 7h toi" -> {{"is_reminder": true, "time": "19:00", ...}}
    "toi co hop luc 14h" -> {{"is_reminder": true, "time": "14:00", ...}}
    "mai 8h sang di kham" -> {{"is_reminder": true, "time": "08:00", ...}}
    "I have a meeting at 2pm" -> {{"is_reminder": true, "time": "14:00", ...}}
- Key signals: cuoc hen, cuoc hop, thuyet trinh, gap, hen, meeting, appointment + time = is_reminder=true
- Do NOT return fire_at — only return time and date strings. Python will compute timestamp.
- If not a reminder: is_reminder=false

User message: {message}"""


async def parse_reminder_nlp(user_id: str, user_text: str) -> str | None:
    """
    Parse natural language reminder.
    Python calculates fire_at from llama8b's HH:MM + DD/MM/YYYY output
    to avoid timezone errors (SPEC §16.5).
    """
    now_dt      = datetime.now(TZ)
    now_str     = now_dt.strftime("%H:%M %d/%m/%Y %A")
    date_str    = now_dt.strftime("%d/%m/%Y")
    tomorrow_str = (now_dt + timedelta(days=1)).strftime("%d/%m/%Y")

    async with httpx.AsyncClient() as http:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)
        try:
            resp = await client.chat.completions.create(
                model=MODEL_REGISTRY["llama8b"]["model_id"],
                messages=[{
                    "role": "user",
                    "content": _PARSE_REMINDER_PROMPT.format(
                        now_str=now_str,
                        date_str=date_str,
                        tomorrow_str=tomorrow_str,
                        message=user_text[:300],
                    ),
                }],
                temperature=0.0,
                max_tokens=100,
            )
            raw  = resp.choices[0].message.content or ""
            raw  = re.sub(r"```[a-z]*\n?|```", "", raw).strip()
            data = json.loads(raw)
            logger.info(f"REMINDER JSON | {data}")

            if not data.get("is_reminder"):
                return None

            time_str: str | None = data.get("time")
            if not time_str:
                return "⏰ Bạn muốn đặt nhắc lúc mấy giờ?"

            date_val = data.get("date") or date_str
            repeat   = data.get("repeat")
            message  = data.get("message") or user_text[:80]

            # Python computes fire_at — llama8b never does this (SPEC §8 + §16.5)
            try:
                hh, mm    = map(int, time_str.split(":"))
                dd, mo, yy = map(int, date_val.split("/"))
                fire_dt   = datetime(yy, mo, dd, hh, mm, tzinfo=TZ)
            except Exception:
                return None

            if fire_dt <= now_dt:
                fire_dt += timedelta(days=1)

            rid        = await save_reminder(user_id, message, int(fire_dt.timestamp()), repeat)
            repeat_str = {
                "daily":   " (lặp hàng ngày)",
                "weekly":  " (lặp hàng tuần)",
                "monthly": " (lặp hàng tháng)",
            }.get(repeat or "", "")

            return (
                f"⏰ Đã đặt nhắc #{rid}{repeat_str}\n"
                f"Nội dung: {message}\n"
                f"Thời gian: {fire_dt.strftime('%H:%M %d/%m/%Y')}"
            )

        except Exception:
            return None


async def reminder_loop() -> None:
    """Background task — check reminders every 30 seconds."""
    while True:
        await asyncio.sleep(30)
        now = int(datetime.now(TZ).timestamp())

        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT id, user_id, message, repeat, fire_at FROM reminders "
                "WHERE fire_at <= ? AND done = 0",
                (now,),
            ) as cur:
                rows = await cur.fetchall()

            for rid, uid, message, repeat, fire_at_db in rows:
                try:
                    label = {
                        "daily":   " (hàng ngày)",
                        "weekly":  " (hàng tuần)",
                        "monthly": " (hàng tháng)",
                    }.get(repeat or "", "")
                    async with AsyncApiClient(line_config) as api_client:
                        line_api = AsyncMessagingApi(api_client)
                        await line_api.push_message(
                            PushMessageRequest(
                                to=uid,
                                messages=[TextMessage(text=f"⏰ Nhắc nhở{label}: {message}")],
                            )
                        )
                except Exception:
                    pass

                if repeat:
                    next_ts = _next_fire(fire_at_db, repeat)
                    await db.execute(
                        "UPDATE reminders SET fire_at = ? WHERE id = ?", (next_ts, rid)
                    )
                else:
                    await db.execute("UPDATE reminders SET done = 1 WHERE id = ?", (rid,))

            await db.commit()


# ---------------------------------------------------------------------------
# MARKDOWN STRIPPER
# ---------------------------------------------------------------------------
def strip_markdown(text: str) -> str:

    # Tách các khối code (```...```) ra để bảo vệ tuyệt đối
    parts_text = re.split(r'(```.*?```)', text, flags=re.DOTALL)
    for i in range(0, len(parts_text), 2):
        p = parts_text[i]
        p = re.sub(r"<think>.*?</think>", "", p, flags=re.DOTALL)
        p = re.sub(r"\*\*(.*?)\*\*", r"\1", p)                 # Xóa dấu in đậm
        p = re.sub(r"^#{1,6}\s+", "", p, flags=re.MULTILINE)     # Xóa dấu heading
        p = re.sub(r"^[\-\*]\s+", "• ", p, flags=re.MULTILINE)   # Đổi list thành dấu chấm tròn
        p = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", p)  # Rút gọn Link
        p = re.sub(r"\n{3,}", "\n\n", p)
        parts_text[i] = p
    return "".join(parts_text).strip()

# ---------------------------------------------------------------------------
# RAG — embedding
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# RAG — chunking
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RAG — ChromaDB upsert (thread-safe per user)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RAG — SQLite metadata helpers
# ---------------------------------------------------------------------------












# ---------------------------------------------------------------------------
# RAG — search
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RAG — PDF ingest pipeline
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GROQ CALLERS
# ---------------------------------------------------------------------------
async def call_groq_text(
    history: list[dict],
    model_id: str,
    model_key: str = DEFAULT_MODEL_KEY,
    user_id: str | None = None,
    rag_chunks: list[dict] | None = None,
) -> str:
    system = (
        await build_system_prompt(user_id, model_key)
        if user_id
        else get_system_prompt(model_key)
    )

    # Inject RAG context into system prompt if chunks present
    if rag_chunks:
        rag_context = "\n\n".join(
            f"[Nguồn: {c['filename']} chunk {c['chunk_index']}]\n{c['content']}"
            for c in rag_chunks
        )
        system = system + (
            "\n\n"
            "═══ QUAN TRỌNG: TÀI LIỆU THAM KHẢO ═══\n"
            "Dưới đây là nội dung từ tài liệu của người dùng. "
            "BẮT BUỘC tuân thủ các quy tắc sau:\n"
            "1. Ưu tiên trả lời DỰA TRÊN nội dung tài liệu bên dưới.\n"
            "2. Nếu câu trả lời CÓ trong tài liệu → trích dẫn và ghi rõ nguồn [Nguồn: tên_file].\n"
            "3. Nếu câu trả lời KHÔNG CÓ trong tài liệu → nói rõ 'Tài liệu không đề cập' rồi mới bổ sung từ kiến thức chung.\n"
            "4. KHÔNG BAO GIỜ bịa thông tin rồi gán cho tài liệu.\n"
            "═══════════════════════════════════════\n\n"
            f"{rag_context}"
        )

    # Compound models require last message role = "user" (SPEC §16.1)
    clean_history = list(history)
    while clean_history and clean_history[-1]["role"] == "assistant":
        clean_history.pop()

    # --- HOT-SWAP LANGUAGE FORCER ---
    if user_id and clean_history and clean_history[-1]["role"] == "user":
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                async with db.execute("SELECT language FROM user_settings WHERE user_id = ?", (user_id,)) as cur:
                    row = await cur.fetchone()
                    lang = row[0] if row else "vi"
            
            rule = "CRITICAL RULE: You MUST answer strictly in Vietnamese." if lang == "vi" else "CRITICAL RULE: You MUST answer strictly in Traditional Chinese (Taiwan)."
            clean_history[-1]["content"] = f"{clean_history[-1]['content']}\n\n[{rule}]"
        except Exception:
            pass
    # --------------------------------

    # reasoning_effort=low for gpt120b (SPEC §11)
    extra: dict = {}
    if model_id == MODEL_REGISTRY["gpt120b"]["model_id"]:
        extra["reasoning_effort"] = "low"

    # max_tokens: user override → model default (SPEC §11)
    user_max = await get_user_max_tokens(user_id) if user_id else 800
    max_tok  = user_max if user_max != 800 else (
        1500 if model_key in ("qwen", "gpt120b", "gpt20b") else 800
    )

    async with httpx.AsyncClient() as http:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": system}] + clean_history,
                temperature=0.6,
                max_tokens=max_tok,
                **extra,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            err = str(e)
            # Compound 400 → fallback to llama70b (SPEC §11)
            if "400" in err and model_id.startswith("groq/compound"):
                fallback_id = MODEL_REGISTRY["llama70b"]["model_id"]
                try:
                    resp = await client.chat.completions.create(
                        model=fallback_id,
                        messages=[{"role": "system", "content": system}] + clean_history,
                        temperature=0.6,
                        max_tokens=max_tok,
                    )
                    return (resp.choices[0].message.content or "").strip()
                except Exception as e2:
                    return f"⚠️ 錯誤 [{fallback_id}]: {str(e2)[:150]}"
            return f"⚠️ 錯誤 [{model_id}]: {err[:150]}"


async def call_groq_vision(image_b64: str) -> str:
    model_id = MODEL_REGISTRY[VISION_MODEL_KEY]["model_id"]
    system   = get_system_prompt(VISION_MODEL_KEY)
    async with httpx.AsyncClient() as http:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "請詳細分析並描述這張圖片。"
                                    "若有文字請完整擷取。"
                                    "若有中文請使用繁體中文。"
                                ),
                            },
                            {
                                "type":      "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                            },
                        ],
                    },
                ],
                max_tokens=800,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"⚠️ 視覺錯誤: {str(e)[:150]}"


async def call_groq_whisper(audio_bytes: bytes) -> str:
    async with httpx.AsyncClient() as http:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)
        try:
            result = await client.audio.transcriptions.create(
                file=("audio.m4a", audio_bytes),
                model=WHISPER_MODEL,
            )
            return result.text
        except Exception as e:
            return f"⚠️ Whisper 錯誤: {str(e)[:150]}"


async def clean_transcript(transcript: str) -> str:
    """Fix Whisper errors via gpt120b (temperature=0.0, max_tokens=300)."""
    async with httpx.AsyncClient() as http:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)
        try:
            resp = await client.chat.completions.create(
                model=MODEL_REGISTRY["gpt120b"]["model_id"],
                messages=[{
                    "role": "user",
                    "content": (
                        "Day la transcript tu nhan dang giong noi tu dong, co the co loi nghe nham, "
                        "sai chinh ta, hoac tu bi thay the sai nghia.\n"
                        "Nhiem vu: sua lai cho dung nghia nhat co the, giu nguyen ngon ngu goc.\n"
                        "Vi du loi thuong gap:\n"
                        "- 'cung mot' co the la '14h' hoac so gio khac\n"
                        "- 'thuc trinh' -> 'thuyet trinh'\n"
                        "Chi tra ve cau da sua, khong giai thich, khong them noi dung.\n\n"
                        f"Transcript: {transcript}"
                    ),
                }],
                temperature=0.0,
                max_tokens=300,
            )
            cleaned = resp.choices[0].message.content.strip()
            return cleaned if cleaned else transcript
        except Exception:
            return transcript


# ---------------------------------------------------------------------------
# COMMAND SYSTEM
# ---------------------------------------------------------------------------






import json

# 1. Định nghĩa các công cụ thực tế (Python Functions)


# Dictionary mapping tên công cụ với hàm Python

# 2. Định nghĩa Schema của Tools cho LLM hiểu

# 3. Vòng lặp Agentic (Core Logic)




# ---------------------------------------------------------------------------
# REPLY HELPER
# ---------------------------------------------------------------------------
def _split_reply(reply: str) -> list[str]:
    """Split reply into LINE-compatible chunks (max 4990 chars, max 5 messages)."""
    chunks: list[str] = []
    while len(reply) > 4990:
        cut = reply.rfind(" ", 0, 4990)
        if cut == -1:
            cut = 4990
        chunks.append(reply[:cut])
        reply = reply[cut:].strip()
    if reply:
        chunks.append(reply)
    return chunks[:5]


# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute("ALTER TABLE user_settings ADD COLUMN language TEXT DEFAULT 'vi'")
            await db.commit()
        except Exception:
            pass
    asyncio.create_task(reminder_loop())
    yield


app            = FastAPI(lifespan=lifespan)
line_config    = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
webhook_parser = WebhookParser(LINE_CHANNEL_SECRET)



from fastapi.responses import HTMLResponse

# --- MODULES TỰ VIẾT ---
from database import DB_PATH, init_db, save_message, save_reminder
from google_workspace import router as gw_router, handle_workspace_command
app.include_router(gw_router)
from rag_core import MAX_FILE_BYTES, SUPPORTED_RAG_EXTS, process_file_upload, has_rag_docs, rag_search, list_rag_docs, delete_rag_doc, clear_rag_docs
from database import get_user_model, set_user_model, get_user_max_tokens, set_user_max_tokens, get_user_profile, save_user_profile, get_history_raw, count_history, get_summary, save_summary, get_reminders, cancel_reminder
from tools_api import AVAILABLE_TOOLS, AGENT_TOOLS
from agents_workflow import run_multi_agent_workflow, run_pro_workflow, run_agentic_loop




from command_handler import handle_command
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    body      = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    try:
        events = webhook_parser.parse(body.decode("utf-8"), signature)
    except Exception:
        raise HTTPException(status_code=400)
    for event in events:
        if isinstance(event, MessageEvent):
            background_tasks.add_task(process_event, event)
    return JSONResponse({"status": "ok"})


async def process_event(event: MessageEvent) -> None:
    user_id = event.source.user_id

    async with AsyncApiClient(line_config) as api_client:
        line_api      = AsyncMessagingApi(api_client)
        line_blob_api = AsyncMessagingApiBlob(api_client)

        # Layer 2: show loading animation (best-effort)
        try:
            await line_api.show_loading_animation(
                ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=10)
            )
        except Exception:
            pass

        reply = ""

        # ── AUDIO PIPELINE ─────────────────────────────────────────────────
        if isinstance(event.message, AudioMessageContent):
            audio_bytes = await line_blob_api.get_message_content(event.message.id)
            transcript  = await call_groq_whisper(audio_bytes)

            if "⚠️" in transcript:
                reply = transcript
            else:
                transcript = await clean_transcript(transcript)
                logger.info(f"AUDIO cleaned | user={user_id} | text={transcript[:50]!r}")

                wants_reply = any(
                    transcript.strip().lower().startswith(t.lower())
                    for t in _REPLY_TRIGGERS
                )

                if wants_reply:
                    clean_text = transcript.strip()
                    for t in _REPLY_TRIGGERS:
                        if clean_text.lower().startswith(t.lower()):
                            clean_text = clean_text[len(t):]
                            break
                    clean_text = clean_text.lstrip(",.! ")

                    # Check reminder BEFORE LLM (deduplication guard)
                    reminder_reply = await parse_reminder_nlp(user_id, clean_text)
                    logger.info(f"REMINDER wants_reply | user={user_id} | found={reminder_reply is not None}")

                    if reminder_reply:
                        reply = f"🎤 {clean_text}\n\n{reminder_reply}"
                    else:
                        # RAG only in wants_reply branch (flow.md §rag_search)
                        model_key, model_id = await resolve_model(user_id, clean_text)
                        history = await get_history_with_summary(user_id)

                        rag_chunks: list[dict] = []
                        rag_enabled = user_id not in _rag_disabled
                        if rag_enabled and await has_rag_docs(user_id):
                            rag_chunks = await rag_search(user_id, clean_text)

                        await save_message(user_id, "user", clean_text)
                        answer = await call_groq_text(
                            history, model_id,
                            model_key=model_key,
                            user_id=user_id,
                            rag_chunks=rag_chunks or None,
                        )
                        await save_message(user_id, "assistant", answer)
                        await maybe_summarize(user_id)
                        reply = f"🎤 {clean_text}\n\n{answer}\n\n[{MODEL_REGISTRY[model_key]['display']}]"

                else:
                    # Transcribe-only branch: check reminder, no RAG
                    reminder_reply = await parse_reminder_nlp(user_id, transcript)
                    logger.info(f"REMINDER transcribe | user={user_id} | found={reminder_reply is not None}")
                    if reminder_reply:
                        reply = f"🎤 {transcript}\n\n{reminder_reply}"
                    else:
                        await save_message(user_id, "user", f"[Voice]: {transcript}")
                        reply = f"🎤 {transcript}"

        # ── IMAGE PIPELINE ─────────────────────────────────────────────────
        elif isinstance(event.message, ImageMessageContent):
            img_bytes = await line_blob_api.get_message_content(event.message.id)
            img_b64   = base64.b64encode(img_bytes).decode("utf-8")
            answer    = await call_groq_vision(img_b64)
            if "⚠️" in answer:
                reply = answer
            else:
                await save_message(user_id, "user", f"[Ảnh] {answer}")
                await save_message(user_id, "assistant", answer)
                await maybe_summarize(user_id)
                reply = answer

        # ── FILE PIPELINE ─────────────────────────────────────────────────
        elif isinstance(event.message, FileMessageContent):
            filename = event.message.file_name or ""
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            
            AUDIO_EXTS = {"mp3", "m4a", "wav", "ogg", "flac"}

            if f".{ext}" not in SUPPORTED_RAG_EXTS and ext not in AUDIO_EXTS:
                reply = f"⚠️ Chỉ hỗ trợ văn bản ({', '.join(sorted(SUPPORTED_RAG_EXTS))}) và âm thanh.\nNhận được: {filename}"
            else:
                file_bytes = await line_blob_api.get_message_content(event.message.id)
                if len(file_bytes) > MAX_FILE_BYTES:
                    reply = f"⚠️ File quá lớn ({len(file_bytes) // 1024 // 1024}MB)."
                else:
                    if ext in AUDIO_EXTS:
                        try:
                            await line_api.push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text="⏳ Đang bóc băng âm thanh...")]))
                        except Exception: pass
                        
                        transcript = await call_groq_whisper(file_bytes)
                        if "⚠️" in transcript:
                            reply = transcript
                        else:
                            transcript = await clean_transcript(transcript)
                            prompt_title = f"Tóm tắt đoạn văn sau thành tối đa 5 chữ để làm tên file, phân cách bằng dấu gạch ngang:\n{transcript[:1000]}"
                            title = await call_groq_text([{"role": "user", "content": prompt_title}], MODEL_REGISTRY["llama8b"]["model_id"], model_key="llama8b", user_id=user_id)
                            
                            import re, time
                            safe_title = re.sub(r'[^a-zA-Z0-9À-ɏḀ-ỿ]', '-', title).strip('-')
                            safe_title = re.sub(r'-+', '-', safe_title) or "audio"
                            new_filename = f"{safe_title[:30]}.txt"
                            
                            async with aiosqlite.connect(DB_PATH) as db:
                                await db.execute("CREATE TABLE IF NOT EXISTS audio_cache (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, transcript TEXT, filename TEXT, created_at INTEGER)")
                                cur = await db.execute("INSERT INTO audio_cache (user_id, transcript, filename, created_at) VALUES (?, ?, ?, ?)", (user_id, transcript, new_filename, int(time.time())))
                                audio_id = cur.lastrowid
                                await db.commit()
                                
                            reply = {
                                "text": f"🎤 Đã bóc băng xong! Tên file: {new_filename}\nBạn muốn xử lý như thế nào?",
                                "quickReply": {
                                    "items": [
                                        {"type": "action", "action": {"type": "message", "label": "1. Tóm tắt + TXT", "text": f"/audio {audio_id} 1"}},
                                        {"type": "action", "action": {"type": "message", "label": "2. Lưu RAG + TXT", "text": f"/audio {audio_id} 2"}},
                                        {"type": "action", "action": {"type": "message", "label": "3. Cả hai + TXT", "text": f"/audio {audio_id} 3"}}
                                    ]
                                }
                            }
                    else:
                        if not ext and file_bytes[:512].decode("utf-8", errors="ignore").isprintable():
                            filename = filename or "upload.txt"
                            ext = "txt"
                        reply = await process_file_upload(user_id, file_bytes, filename)

        # ── TEXT PIPELINE ──────────────────────────────────────────────────
        elif isinstance(event.message, TextMessageContent):
            user_text = event.message.text.strip()

            # UX Trick: Nếu người dùng chỉ gõ số (vd: "2"), tự động biên dịch thành lệnh "/mail 2"
            if user_text.isdigit() and len(user_text) <= 2:
                cmd_reply = await handle_command(user_id, f"/mail {user_text}")
            else:
                cmd_reply = await handle_command(user_id, user_text)
                
            # Layer 3a: command handler
            if cmd_reply is not None:
                reply = cmd_reply

            # Layer 3b: input length guard
            elif len(user_text) > MAX_INPUT_CHARS:
                reply = (
                    f"⚠️ Tin nhắn quá dài ({len(user_text)} ký tự).\n"
                    f"Vui lòng giới hạn dưới {MAX_INPUT_CHARS} ký tự."
                )

            else:
                # Layer 3c: natural language reminder (before model routing)
                reminder_reply = await parse_reminder_nlp(user_id, user_text)
                if reminder_reply:
                    reply = reminder_reply
                else:
                    # Layer 3d: model routing → history → RAG → LLM
                    model_key, model_id = await resolve_model(user_id, user_text)
                    logger.info(f"TEXT | user={user_id} | model={model_key} | text={user_text[:50]!r}")

                    # Summarize-mode vs normal history
                    is_summarize = (
                        len(user_text) > 500
                        and "?" not in user_text
                        and "？" not in user_text
                    )
                    await save_message(user_id, "user", user_text)

                    if is_summarize:
                        history = [{"role": "user", "content": f"Hãy tóm tắt nội dung sau:\n{user_text}"}]
                    else:
                        history = await get_history_with_summary(user_id)

                    # RAG search — only when: user has docs AND RAG not disabled AND not a command
                    rag_chunks: list[dict] = []
                    rag_enabled = user_id not in _rag_disabled
                    if rag_enabled and await has_rag_docs(user_id):
                        rag_chunks = await rag_search(user_id, user_text)

                    answer = await call_groq_text(
                        history, model_id,
                        model_key=model_key,
                        user_id=user_id,
                        rag_chunks=rag_chunks or None,
                    )
                    await save_message(user_id, "assistant", answer)
                    await maybe_summarize(user_id)
                    reply = answer

        # ── LAYER 4: REPLY ─────────────────────────────────────────────────
        if reply:
            if isinstance(reply, dict):
                qr_data = reply.get("quickReply")
                qr_items = []
                if qr_data:
                    for item in qr_data.get("items", []):
                        act = item.get("action", {})
                        qr_items.append(
                            QuickReplyItem(
                                action=MessageAction(
                                    label=act.get("label", ""),
                                    text=act.get("text", ""),
                                )
                            )
                        )
                quick_reply_obj = QuickReply(items=qr_items) if qr_items else None

                if reply.get("type") == "flex":
                    from linebot.v3.messaging import FlexMessage, FlexContainer
                    flex_msg = FlexMessage(
                        alt_text=reply.get("altText", "Hộp thư Flex Message"),
                        contents=FlexContainer.from_dict(reply.get("contents"))
                    )
                    if quick_reply_obj:
                        flex_msg.quick_reply = quick_reply_obj
                    messages = [flex_msg]
                else:
                    text_content = strip_markdown(reply.get("text", ""))
                    chunks = _split_reply(text_content)
                    messages = [TextMessage(text=c) for c in chunks]
                    if quick_reply_obj and messages:
                        messages[-1].quick_reply = quick_reply_obj
                
                await line_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=messages,
                    )
                )
            else:
                reply = strip_markdown(reply)
                chunks = _split_reply(reply)
                await line_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=c) for c in chunks],
                    )
                )
# Last Fix: Mon Mar 16 03:58:20 CST 2026
# Fix scope at Mon Mar 16 04:03:44 CST 2026
# Fix OAuth UX: Mon Mar 16 04:08:53 CST 2026
# Clean UI Fix: Mon Mar 16 04:19:18 CST 2026
# Clean UI Fix: Mon Mar 16 04:24:07 CST 2026
# Fix DB Connection: Mon Mar 16 04:29:38 CST 2026

# UI Update forced at: 2026-03-16
# Manual Trigger: Mon Mar 16 04:37:57 CST 2026
# Quick Reply Update: Mon Mar 16 04:41:56 CST 2026
# Fix Syntax & Finish QuickReply: Mon Mar 16 04:47:06 CST 2026
# Fix Syntax & Enable QR: Mon Mar 16 04:52:01 CST 2026
# Fix Syntax & Enable QR: Mon Mar 16 05:15:55 CST 2026
# Syntax fix for line 1380 - Mon Mar 16 05:21:46 CST 2026
# Update Limit to 200: Mon Mar 16 08:50:00 CST 2026
# Add Google Calendar Module: Mon Mar 16 13:05:59 CST 2026
# Fix OAuth Scope %20 - Mon Mar 16 13:09:16 CST 2026
