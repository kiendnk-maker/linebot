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
DB_PATH                   = os.environ.get("DB_PATH", "chat_history.db")
CHROMA_PATH               = os.environ.get("CHROMA_PATH", "chroma")


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_EMBED_MODEL = "gemini-embedding-001"

WHISPER_MODEL   = "whisper-large-v3-turbo"
MAX_INPUT_CHARS = 8000
SUMMARY_TRIGGER = 20
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
RAG_TOP_K       = 3
MAX_PDF_BYTES   = 10 * 1024 * 1024   # 10 MB
MAX_DOCS_PER_USER = 20

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


def _get_meta_lock() -> asyncio.Lock:
    """Return the global meta-lock, creating it on first call inside the event loop.
    asyncio.Lock() must NOT be instantiated at import time in Python 3.12+.
    """
    global _chroma_locks_meta
    if _chroma_locks_meta is None:
        _chroma_locks_meta = asyncio.Lock()
    return _chroma_locks_meta

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


async def _get_chroma_lock(user_id: str) -> asyncio.Lock:
    """Return (and lazily create) a per-user asyncio.Lock for ChromaDB writes."""
    async with _get_meta_lock():
        if user_id not in _chroma_locks:
            _chroma_locks[user_id] = asyncio.Lock()
        return _chroma_locks[user_id]


def get_user_collection(user_id: str):
    return chroma_client.get_or_create_collection(
        name=f"rag_{user_id}",
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# RAG — in-memory RAG-off toggle (per session, resets on restart)
# ---------------------------------------------------------------------------
_rag_disabled: set[str] = set()   # user_ids with RAG disabled


# ---------------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------------
async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "CREATE TABLE IF NOT EXISTS history "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
            " user_id TEXT, role TEXT, content TEXT)"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS user_settings "
            "(user_id TEXT PRIMARY KEY, model_key TEXT NOT NULL)"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS user_settings2 "
            "(user_id TEXT PRIMARY KEY, max_tokens INTEGER NOT NULL DEFAULT 800)"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS summary "
            "(user_id TEXT PRIMARY KEY, content TEXT, updated_at INTEGER)"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS reminders "
            "(id        INTEGER PRIMARY KEY AUTOINCREMENT, "
            " user_id   TEXT    NOT NULL, "
            " message   TEXT    NOT NULL, "
            " fire_at   INTEGER NOT NULL, "
            " repeat    TEXT    DEFAULT NULL, "
            " done      INTEGER DEFAULT 0)"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS user_profile "
            "(user_id TEXT PRIMARY KEY, "
            " name TEXT, occupation TEXT, learning TEXT, notes TEXT)"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS rag_docs "
            "(id          INTEGER PRIMARY KEY AUTOINCREMENT, "
            " user_id     TEXT    NOT NULL, "
            " filename    TEXT    NOT NULL, "
            " chunk_count INTEGER NOT NULL, "
            " uploaded_at INTEGER NOT NULL)"
        )
        await db.commit()


# ---------------------------------------------------------------------------
# DATABASE — user settings helpers
# ---------------------------------------------------------------------------
async def get_user_model(user_id: str) -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT model_key FROM user_settings WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
    key = row[0] if row else DEFAULT_MODEL_KEY
    return key if key in MODEL_REGISTRY else DEFAULT_MODEL_KEY


async def set_user_model(user_id: str, model_key: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO user_settings (user_id, model_key) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET model_key = excluded.model_key",
            (user_id, model_key),
        )
        await db.commit()


async def get_user_max_tokens(user_id: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT max_tokens FROM user_settings2 WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else 800


async def set_user_max_tokens(user_id: str, max_tokens: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO user_settings2 (user_id, max_tokens) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET max_tokens = excluded.max_tokens",
            (user_id, max_tokens),
        )
        await db.commit()


# ---------------------------------------------------------------------------
# DATABASE — user profile helpers
# ---------------------------------------------------------------------------
async def get_user_profile(user_id: str) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT name, occupation, learning, notes FROM user_profile WHERE user_id = ?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return {}
    return {k: v for k, v in zip(("name", "occupation", "learning", "notes"), row) if v}


async def save_user_profile(user_id: str, **kwargs) -> None:
    if not kwargs:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO user_profile (user_id) VALUES (?) "
            "ON CONFLICT(user_id) DO NOTHING",
            (user_id,),
        )
        for key, value in kwargs.items():
            if key in ("name", "occupation", "learning", "notes"):
                await db.execute(
                    f"UPDATE user_profile SET {key} = ? WHERE user_id = ?",
                    (value, user_id),
                )
        await db.commit()


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
async def save_message(user_id: str, role: str, content: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content),
        )
        await db.commit()


async def get_history_raw(user_id: str, limit: int = 30) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT role, content FROM history "
            "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ) as cur:
            rows = await cur.fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


async def count_history(user_id: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM history WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# SUMMARY MEMORY
# ---------------------------------------------------------------------------
async def get_summary(user_id: str) -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT content FROM summary WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else ""


async def save_summary(user_id: str, content: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO summary (user_id, content, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET "
            "content = excluded.content, updated_at = excluded.updated_at",
            (user_id, content, int(time.time())),
        )
        await db.commit()


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
async def save_reminder(
    user_id: str, message: str, fire_at: int, repeat: str | None = None
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO reminders (user_id, message, fire_at, repeat) VALUES (?, ?, ?, ?)",
            (user_id, message, fire_at, repeat),
        )
        await db.commit()
        return cur.lastrowid  # type: ignore[return-value]


async def get_reminders(user_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, message, fire_at, repeat FROM reminders "
            "WHERE user_id = ? AND done = 0 ORDER BY fire_at ASC",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [{"id": r[0], "message": r[1], "fire_at": r[2], "repeat": r[3]} for r in rows]


async def cancel_reminder(user_id: str, reminder_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "UPDATE reminders SET done = 1 WHERE id = ? AND user_id = ? AND done = 0",
            (reminder_id, user_id),
        )
        await db.commit()
        return cur.rowcount > 0


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
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"^[\-\*]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"^(-{3,}|\*{3,}|_{3,})\s*$", "─────", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# RAG — embedding
# ---------------------------------------------------------------------------
class EmbedError(RuntimeError):
    pass


async def embed_text(text: str) -> list[float]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBED_MODEL}:embedContent?key={GEMINI_API_KEY}"
    payload = {"model": f"models/{GEMINI_EMBED_MODEL}", "content": {"parts": [{"text": text}]}}
    for attempt in range(3):
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.post(url, headers={"Content-Type": "application/json"}, json=payload)
        if resp.status_code == 429:
            logger.warning(f"Gemini API rate limit 429, attempt {attempt+1}/3, sleeping 10s")
            await asyncio.sleep(10)
            continue
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]
    raise EmbedError("Gemini API unavailable after 3 retries")

async def embed_batch(texts: list[str]) -> list[list[float]]:
    result = []
    for text in texts:
        result.append(await embed_text(text))
        await asyncio.sleep(0.5)
    return result


# ---------------------------------------------------------------------------
# RAG — chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# RAG — ChromaDB upsert (thread-safe per user)
# ---------------------------------------------------------------------------
async def chroma_upsert(
    user_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    filename: str,
) -> None:
    lock = await _get_chroma_lock(user_id)
    async with lock:
        col = get_user_collection(user_id)
        ts  = int(time.time())
        col.upsert(
            ids=[f"{filename}_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[
                {"filename": filename, "chunk_index": i, "uploaded_at": ts}
                for i in range(len(chunks))
            ],
        )


# ---------------------------------------------------------------------------
# RAG — SQLite metadata helpers
# ---------------------------------------------------------------------------
async def save_rag_doc(user_id: str, filename: str, chunk_count: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO rag_docs (user_id, filename, chunk_count, uploaded_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, filename, chunk_count, int(time.time())),
        )
        await db.commit()


async def list_rag_docs(user_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, filename, chunk_count, uploaded_at FROM rag_docs "
            "WHERE user_id = ? ORDER BY uploaded_at DESC",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [{"id": r[0], "filename": r[1], "chunk_count": r[2], "uploaded_at": r[3]} for r in rows]


async def count_rag_docs(user_id: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM rag_docs WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else 0


async def has_rag_docs(user_id: str) -> bool:
    return await count_rag_docs(user_id) > 0


async def delete_rag_doc(user_id: str, filename: str) -> bool:
    """Delete one file from ChromaDB + SQLite metadata."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "DELETE FROM rag_docs WHERE user_id = ? AND filename = ?",
            (user_id, filename),
        )
        await db.commit()
        if cur.rowcount == 0:
            return False

    lock = await _get_chroma_lock(user_id)
    async with lock:
        try:
            col = get_user_collection(user_id)
            # Delete all chunks whose ID starts with filename_
            all_ids = col.get()["ids"]
            to_delete = [i for i in all_ids if i.startswith(f"{filename}_")]
            if to_delete:
                col.delete(ids=to_delete)
        except Exception as e:
            logger.warning(f"ChromaDB delete error for {filename}: {e}")
    return True


async def clear_rag_docs(user_id: str) -> int:
    """Clear all RAG docs for user. Returns number of files deleted."""
    docs = await list_rag_docs(user_id)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM rag_docs WHERE user_id = ?", (user_id,))
        await db.commit()

    lock = await _get_chroma_lock(user_id)
    async with lock:
        try:
            chroma_client.delete_collection(f"rag_{user_id}")
        except Exception:
            pass
    return len(docs)


# ---------------------------------------------------------------------------
# RAG — search
# ---------------------------------------------------------------------------
async def rag_search(
    user_id: str,
    query: str,
    top_k: int = RAG_TOP_K,
) -> list[dict]:
    """
    Guard conditions (flow.md §rag_search):
    - Only called when user_id has at least 1 doc (caller must check has_rag_docs)
    - Not called for commands, image, audio-transcribe-only, file handler, reminders
    Returns list of {content, filename, chunk_index}.
    """
    try:
        vec = await embed_text(query)
        col = get_user_collection(user_id)
        results = col.query(query_embeddings=[vec], n_results=top_k)
        docs      = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        return [
            {
                "content":     doc,
                "filename":    meta.get("filename", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
            }
            for doc, meta in zip(docs, metadatas)
        ]
    except Exception as e:
        logger.warning(f"rag_search error for {user_id}: {e}")
        return []


# ---------------------------------------------------------------------------
# RAG — PDF ingest pipeline
# ---------------------------------------------------------------------------
async def process_pdf_upload(user_id: str, file_bytes: bytes, filename: str) -> str:
    """
    PDF ingest guards (flow.md + SPEC §9.7):
    Guard 1: .pdf extension     (caller checks)
    Guard 2: <= 10MB            (caller checks)
    Guard 3: pdfplumber non-empty
    Guard 4: HuggingFace embed retry x3
    Guard 5: per-user doc limit = 20
    """
    # Guard 5: doc count limit
    doc_count = await count_rag_docs(user_id)
    if doc_count >= MAX_DOCS_PER_USER:
        return (
            f"⚠️ Bạn đã có {doc_count} tài liệu (tối đa {MAX_DOCS_PER_USER}).\n"
            "Dùng /rag delete <tên file> để xoá bớt."
        )

    # Guard 3: extract text with pdfplumber
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages_text = [page.extract_text() or "" for page in pdf.pages]
        full_text = "\n".join(pages_text).strip()
    except Exception as e:
        logger.warning(f"pdfplumber error for {filename}: {e}")
        return "⚠️ Không đọc được file PDF này. File có thể bị hỏng hoặc mã hoá."

    if not full_text:
        return "⚠️ PDF không có nội dung văn bản (có thể là ảnh scan, chưa hỗ trợ OCR)."

    # Chunk
    chunks = chunk_text(full_text)
    logger.info(f"PDF {filename} → {len(chunks)} chunks for user={user_id}")

    # Guard 4: embed with retry
    try:
        embeddings = await embed_batch(chunks)
    except EmbedError as e:
        return f"⚠️ Lỗi embedding: {e}"

    # Upsert to ChromaDB (per-user lock)
    await chroma_upsert(user_id, chunks, embeddings, filename)

    # Save metadata to SQLite
    await save_rag_doc(user_id, filename, len(chunks))

    return (
        f"✅ Đã lưu {len(chunks)} chunks từ {filename}\n"
        "Bạn có thể hỏi về nội dung file này."
    )


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
        system = system + f"\n\n【參考資料】\n{rag_context}"

    # Compound models require last message role = "user" (SPEC §16.1)
    clean_history = list(history)
    while clean_history and clean_history[-1]["role"] == "assistant":
        clean_history.pop()

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
            return strip_markdown(resp.choices[0].message.content or "")
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
                    return strip_markdown(resp.choices[0].message.content or "")
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
            return strip_markdown(resp.choices[0].message.content or "")
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
def _models_list_text() -> str:
    prod_lines: list[str] = []
    prev_lines: list[str] = []
    for key, cfg in MODEL_REGISTRY.items():
        icon = {"vision": "👁", "reasoning": "🧠", "text": "💬"}.get(cfg["type"], "💬")
        line = f"{icon} /{key} — {cfg['display']}\n   {cfg['note']}"
        if cfg["tier"] == "production":
            prod_lines.append(line)
        else:
            prev_lines.append(line)
    return "\n".join([
        "📋 可用模型列表\n",
        "── Production ──", *prod_lines,
        "\n── Preview ──",  *prev_lines,
        "\n─────────────────",
        "切換模型：/model <名稱>",
        "目前模型：/model",
        "自動模式：/auto",
        "清除紀錄：/clear",
        "提醒清單：/remind list",
    ])


async def handle_command(user_id: str, text: str) -> str | None:
    if not text.startswith("/"):
        return None

    parts = text[1:].strip().split(maxsplit=1)
    cmd   = parts[0].lower()
    arg   = parts[1].strip() if len(parts) > 1 else ""

    # ── CLEAR ──────────────────────────────────────────────────────────────
    if cmd == "clear":
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
            await db.execute("DELETE FROM summary WHERE user_id = ?", (user_id,))
            await db.commit()
        return "🗑 對話記錄已清除。"

    # ── MODELS ─────────────────────────────────────────────────────────────
    if cmd == "models":
        return _models_list_text()

    # ── AUTO ───────────────────────────────────────────────────────────────
    if cmd == "auto":
        await set_user_model(user_id, DEFAULT_MODEL_KEY)
        return "🤖 已切換至自動選擇模型模式。"

    # ── MODEL ──────────────────────────────────────────────────────────────
    if cmd == "model":
        if not arg:
            key  = await get_user_model(user_id)
            cfg  = MODEL_REGISTRY[key]
            mode = "自動" if key == DEFAULT_MODEL_KEY else "手動"
            return (
                f"🤖 目前模型：{cfg['display']}\n"
                f"   Tier：{cfg['tier']} | 模式：{mode}\n"
                f"   {cfg['note']}\n\n"
                "輸入 /models 查看全部。\n"
                "輸入 /auto 返回自動模式。"
            )
        target = arg.lower()
        if target not in MODEL_REGISTRY:
            return f"❌ /{target} 不存在。\n請輸入 /models 查看清單。"
        await set_user_model(user_id, target)
        return f"✅ 已切換至 {MODEL_REGISTRY[target]['display']}。\n輸入 /auto 返回自動模式。"

    # ── LONG / SHORT / TOKENS ──────────────────────────────────────────────
    if cmd == "long":
        val = int(arg) if arg.isdigit() else 3000
        val = min(val, 6000)
        await set_user_max_tokens(user_id, val)
        return f"Chế độ trả lời dài: tối đa {val} tokens (~{val*4} ký tự)"

    if cmd == "short":
        await set_user_max_tokens(user_id, 800)
        return "Chế độ trả lời ngắn: 800 tokens (mặc định)"

    if cmd == "tokens":
        val = await get_user_max_tokens(user_id)
        return f"Max tokens hiện tại: {val} (~{val*4} ký tự)"

    # ── PROFILE ────────────────────────────────────────────────────────────
    if cmd == "profile":
        if arg == "clear":
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("DELETE FROM user_profile WHERE user_id = ?", (user_id,))
                await db.commit()
            return "🗑 Đã xoá thông tin cá nhân."

        profile = await get_user_profile(user_id)
        if not arg:
            if not profile:
                return (
                    "Chưa có thông tin cá nhân.\n"
                    "Cập nhật:\n"
                    "/profile name Tên bạn\n"
                    "/profile job Nghề nghiệp\n"
                    "/profile learning Tiếng Trung B1\n"
                    "/profile note Ghi chú thêm"
                )
            lines = ["Thông tin của bạn:\n"]
            if profile.get("name"):       lines.append("Tên: "       + profile["name"])
            if profile.get("occupation"): lines.append("Nghề: "      + profile["occupation"])
            if profile.get("learning"):   lines.append("Đang học: "  + profile["learning"])
            if profile.get("notes"):      lines.append("Ghi chú: "   + profile["notes"])
            return "\n".join(lines)

        parts2 = arg.split(maxsplit=1)
        if len(parts2) < 2:
            return "Dùng: /profile name|job|learning|note <nội dung>"
        field, value = parts2[0].lower(), parts2[1]
        field_map = {"name": "name", "job": "occupation", "learning": "learning", "note": "notes"}
        if field not in field_map:
            return "Field hợp lệ: name, job, learning, note"
        await save_user_profile(user_id, **{field_map[field]: value})
        return f"Đã lưu {field}: {value}"

    # ── REMIND ─────────────────────────────────────────────────────────────
    if cmd == "remind":
        if arg == "list":
            reminders = await get_reminders(user_id)
            if not reminders:
                return "📭 Không có nhắc nhở nào đang chờ."
            lines = ["📋 Danh sách nhắc nhở:\n"]
            for r in reminders:
                dt = datetime.fromtimestamp(r["fire_at"], tz=TZ)
                label = {
                    "daily":   " 🔁 hàng ngày",
                    "weekly":  " 🔁 hàng tuần",
                    "monthly": " 🔁 hàng tháng",
                }.get(r["repeat"] or "", "")
                lines.append(
                    f"#{r['id']}{label}\n"
                    f"  {r['message']}\n"
                    f"  ⏰ {dt.strftime('%H:%M %d/%m/%Y')}"
                )
            return "\n".join(lines)

        # /remind <id> cancel
        parts2 = arg.split()
        if len(parts2) == 2 and parts2[1] == "cancel":
            try:
                rid = int(parts2[0])
                ok  = await cancel_reminder(user_id, rid)
                return f"✅ Đã huỷ nhắc #{rid}." if ok else f"❌ Không tìm thấy nhắc #{rid}."
            except ValueError:
                pass

        # /remind HH:MM [daily|weekly|monthly] [tối|chiều|sáng] [ngày DD/MM[/YYYY]] nội dung
        time_match = re.match(r"(\d{1,2}):(\d{2})\s+(.*)", arg)
        if time_match:
            hour   = int(time_match[1])
            minute = int(time_match[2])
            rest   = time_match[3].strip()

            # Repeat keyword
            repeat: str | None = None
            repeat_map = {
                "daily": "daily", "weekly": "weekly", "monthly": "monthly",
                "hàng ngày": "daily", "hàng tuần": "weekly", "hàng tháng": "monthly",
            }
            for kw, val in repeat_map.items():
                if rest.lower().startswith(kw):
                    repeat = val
                    rest   = rest[len(kw):].strip()
                    break

            # AM/PM detection from full arg string (SPEC §8)
            arg_lower = arg.lower()
            is_pm = any(kw in arg_lower for kw in ("tối", "chiều", "pm", "evening", "afternoon", "tonight"))
            is_am = any(kw in arg_lower for kw in ("sáng", "am", "morning", "trưa"))
            if is_pm and not is_am and hour < 12:
                hour += 12
            elif is_am and hour == 12:
                hour = 0

            # Date detection: regex DD/MM or DD-MM in rest string
            now_dt  = datetime.now(TZ)
            date_match = re.search(r"ngày\s+(\d{1,2})[/-](\d{1,2})(?:[/-](\d{4}))?", rest, re.IGNORECASE)
            if date_match:
                dd  = int(date_match[1])
                mo  = int(date_match[2])
                yy  = int(date_match[3]) if date_match[3] else now_dt.year
                # Remove date phrase from rest
                rest = rest[:date_match.start()].strip() + " " + rest[date_match.end():].strip()
                rest = rest.strip()
                fire_dt = datetime(yy, mo, dd, hour, minute, tzinfo=TZ)
            else:
                fire_dt = now_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if fire_dt <= now_dt:
                    fire_dt += timedelta(days=1)

            rid        = await save_reminder(user_id, rest, int(fire_dt.timestamp()), repeat)
            repeat_str = {
                "daily":   " (lặp hàng ngày)",
                "weekly":  " (lặp hàng tuần)",
                "monthly": " (lặp hàng tháng)",
            }.get(repeat or "", "")
            return (
                f"⏰ Đã đặt nhắc #{rid}{repeat_str}\n"
                f"Nội dung: {rest}\n"
                f"Thời gian: {fire_dt.strftime('%H:%M %d/%m/%Y')}"
            )

        return (
            "❓ Cách dùng /remind:\n"
            "/remind list\n"
            "/remind 2 cancel\n"
            "/remind 20:00 uống thuốc\n"
            "/remind 20:00 daily uống thuốc\n"
            "/remind 09:00 weekly họp team"
        )

    # ── RAG COMMANDS ───────────────────────────────────────────────────────
    if cmd == "rag":
        sub = arg.lower().split(maxsplit=1)
        sub_cmd = sub[0] if sub else ""
        sub_arg = sub[1].strip() if len(sub) > 1 else ""

        if sub_cmd == "list":
            docs = await list_rag_docs(user_id)
            if not docs:
                return "📂 Chưa có tài liệu nào. Gửi file PDF để thêm."
            lines = [f"📂 Tài liệu của bạn ({len(docs)} file):\n"]
            for d in docs:
                dt = datetime.fromtimestamp(d["uploaded_at"], tz=TZ).strftime("%d/%m/%Y %H:%M")
                lines.append(f"• {d['filename']}\n  {d['chunk_count']} chunks — {dt}")
            return "\n".join(lines)

        if sub_cmd == "delete" and sub_arg:
            ok = await delete_rag_doc(user_id, sub_arg)
            return (
                f"✅ Đã xoá {sub_arg} khỏi knowledge base."
                if ok
                else f"❌ Không tìm thấy file: {sub_arg}\nDùng /rag list để xem danh sách."
            )

        if sub_cmd == "clear":
            n = await clear_rag_docs(user_id)
            return f"🗑 Đã xoá {n} tài liệu khỏi knowledge base."

        if sub_cmd == "off":
            _rag_disabled.add(user_id)
            return "🔕 RAG đã tắt cho session này. Dùng /rag on để bật lại."

        if sub_cmd == "on":
            _rag_disabled.discard(user_id)
            return "🔔 RAG đã bật."

        return (
            "📚 Lệnh RAG:\n"
            "/rag list — xem danh sách file\n"
            "/rag delete <tên file> — xoá file\n"
            "/rag clear — xoá tất cả\n"
            "/rag off — tắt RAG\n"
            "/rag on — bật RAG"
        )

    # ── MODEL SHORTCUT ─────────────────────────────────────────────────────
    if cmd in MODEL_REGISTRY:
        await set_user_model(user_id, cmd)
        cfg = MODEL_REGISTRY[cmd]
        if arg:
            answer = await call_groq_text(
                [{"role": "user", "content": arg}],
                cfg["model_id"],
                model_key=cmd,
            )
            return f"[{cfg['display']}]\n{answer}"
        return f"✅ 已切換至 {cfg['display']}。\n輸入 /auto 返回自動模式。"

    return f"❓ 指令 /{cmd} 無效。請輸入 /models 查看。"


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
    asyncio.create_task(reminder_loop())
    yield


app            = FastAPI(lifespan=lifespan)
line_config    = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
webhook_parser = WebhookParser(LINE_CHANNEL_SECRET)


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

        # ── FILE PIPELINE (PDF INGEST) ─────────────────────────────────────
        elif isinstance(event.message, FileMessageContent):
            filename = event.message.file_name or ""

            # Guard 1: must be .pdf
            if not filename.lower().endswith(".pdf"):
                reply = "⚠️ Chỉ hỗ trợ file PDF. Vui lòng gửi file .pdf."
            else:
                file_size = getattr(event.message, "file_size", None)

                # Guard 2: size <= 10MB (check header if available, else download and check)
                if file_size is not None and file_size > MAX_PDF_BYTES:
                    reply = (
                        f"⚠️ File quá lớn ({file_size // 1024 // 1024}MB).\n"
                        f"Giới hạn tối đa {MAX_PDF_BYTES // 1024 // 1024}MB."
                    )
                else:
                    file_bytes = await line_blob_api.get_message_content(event.message.id)
                    if len(file_bytes) > MAX_PDF_BYTES:
                        reply = (
                            f"⚠️ File quá lớn ({len(file_bytes) // 1024 // 1024}MB).\n"
                            f"Giới hạn tối đa {MAX_PDF_BYTES // 1024 // 1024}MB."
                        )
                    else:
                        reply = await process_pdf_upload(user_id, file_bytes, filename)

        # ── TEXT PIPELINE ──────────────────────────────────────────────────
        elif isinstance(event.message, TextMessageContent):
            user_text = event.message.text.strip()

            # Layer 3a: command handler
            cmd_reply = await handle_command(user_id, user_text)
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
            chunks = _split_reply(reply)
            await line_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=c) for c in chunks],
                )
            )
