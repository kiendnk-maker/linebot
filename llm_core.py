import os
import re
import json
import httpx
import aiosqlite
import logging
import base64
from groq import AsyncGroq
from prompts import get_system_prompt
from database import DB_PATH, get_user_profile, get_user_model, get_user_max_tokens, count_history, get_history_raw, get_summary, save_summary

logger = logging.getLogger(__name__)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

WHISPER_MODEL   = "whisper-large-v3-turbo"

SUMMARY_TRIGGER = 20

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

