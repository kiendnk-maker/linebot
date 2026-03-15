import os, base64, aiosqlite, httpx, re, json, asyncio
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
    MessageEvent, TextMessageContent, ImageMessageContent, AudioMessageContent,
)
from prompts import get_system_prompt

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
LINE_CHANNEL_SECRET       = os.environ["LINE_CHANNEL_SECRET"]
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
GROQ_API_KEY              = os.environ["GROQ_API_KEY"]
DB_PATH                   = os.environ.get("DB_PATH", "chat_history.db")
WHISPER_MODEL             = "whisper-large-v3-turbo"
MAX_INPUT_CHARS           = 8000   # ~2000 tokens
SUMMARY_TRIGGER           = 20     # tóm tắt sau mỗi 20 tin

# Triggers kích hoạt LLM reply khi ghi âm
_REPLY_TRIGGERS = ("hello", "不好意思")

# ---------------------------------------------------------------------------
# MODEL REGISTRY
# Source: https://console.groq.com/docs/models  (checked 2026-03-14)
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict] = {
    "llama8b": {
        "model_id": "llama-3.1-8b-instant",
        "type":     "text",
        "tier":     "production",
        "display":  "LLaMA 3.1 8B",
        "ctx":      131_072,
        "note":     "最快 ~900 t/s，超便宜 — 用於 classifier 和簡單對話",
    },
    "llama70b": {
        "model_id": "llama-3.3-70b-versatile",
        "type":     "text",
        "tier":     "production",
        "display":  "LLaMA 3.3 70B",
        "ctx":      131_072,
        "note":     "均衡性能，寫作、翻譯、多語言",
    },
    "gpt20b": {
        "model_id": "openai/gpt-oss-20b",
        "type":     "reasoning",
        "tier":     "production",
        "display":  "GPT-OSS 20B",
        "ctx":      131_072,
        "note":     "超快 ~1000 t/s，輕量推理",
    },
    "gpt120b": {
        "model_id": "openai/gpt-oss-120b",
        "type":     "reasoning",
        "tier":     "production",
        "display":  "GPT-OSS 120B",
        "ctx":      131_072,
        "note":     "最強推理，~500 t/s",
    },
    "compound": {
        "model_id": "groq/compound",
        "type":     "text",
        "tier":     "production",
        "display":  "Groq Compound（網路搜尋）",
        "ctx":      131_072,
        "note":     "內建網路搜尋 + 程式執行，最多10次工具呼叫",
    },
    "compound-mini": {
        "model_id": "groq/compound-mini",
        "type":     "text",
        "tier":     "production",
        "display":  "Groq Compound Mini",
        "ctx":      131_072,
        "note":     "Compound 輕量版，單次工具呼叫，速度快3倍",
    },
    "scout": {
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "type":     "vision",
        "tier":     "preview",
        "display":  "LLaMA 4 Scout（視覺）",
        "ctx":      131_072,
        "note":     "唯一視覺模型，低延遲",
    },
    "qwen": {
        "model_id": "qwen/qwen3-32b",
        "type":     "reasoning",
        "tier":     "preview",
        "display":  "Qwen3 32B",
        "ctx":      131_072,
        "note":     "推理 + thinking mode，多語言強",
    },
    "kimi": {
        "model_id": "moonshotai/kimi-k2-instruct-0905",
        "type":     "text",
        "tier":     "preview",
        "display":  "Kimi K2 0905",
        "ctx":      262_144,
        "note":     "最長 context 256K，agentic coding",
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
    current_key = await get_user_model(user_id)

    if current_key != DEFAULT_MODEL_KEY:
        return current_key, MODEL_REGISTRY[current_key]["model_id"]

    if _needs_realtime(user_text):
        return "compound-mini", MODEL_REGISTRY["compound-mini"]["model_id"]

    # Text dài không có câu hỏi → tóm tắt/phân tích → llama70b
    if len(user_text) > 500 and "?" not in user_text and "？" not in user_text:
        return "llama70b", MODEL_REGISTRY["llama70b"]["model_id"]

    routed_key = await classify_query(user_text)
    return routed_key, MODEL_REGISTRY[routed_key]["model_id"]


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
        await db.commit()


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
    import time
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO summary (user_id, content, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET content = excluded.content, "
            "updated_at = excluded.updated_at",
            (user_id, content, int(time.time())),
        )
        await db.commit()


async def maybe_summarize(user_id: str) -> None:
    """Tóm tắt tự động sau mỗi SUMMARY_TRIGGER tin nhắn."""
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
    """
    Trả về summary + 5 tin gần nhất.
    Tiết kiệm ~60% token so với raw history 30 tin.
    """
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
    user_id: str,
    message: str,
    fire_at: int,
    repeat: str | None = None,
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO reminders (user_id, message, fire_at, repeat) VALUES (?, ?, ?, ?)",
            (user_id, message, fire_at, repeat),
        )
        await db.commit()
        return cur.lastrowid


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
    dt = datetime.fromtimestamp(fire_at, tz=TZ)
    if repeat == "daily":
        dt += timedelta(days=1)
    elif repeat == "weekly":
        dt += timedelta(weeks=1)
    elif repeat == "monthly":
        # Tăng tháng thủ công không cần dateutil
        month = dt.month + 1
        year  = dt.year + (1 if month > 12 else 0)
        month = month if month <= 12 else 1
        dt    = dt.replace(year=year, month=month)
    return int(dt.timestamp())


_PARSE_REMINDER_PROMPT = """Extract reminder info from the user message.
Current unix timestamp: {now}
Current datetime: {now_str}

Reply ONLY with JSON, no explanation:
{{"is_reminder": true/false, "message": "reminder content", "fire_at": unix_timestamp, "repeat": null or "daily" or "weekly" or "monthly"}}

Rules:
- "hôm nay/tonight/today" → same day
- "ngày mai/tomorrow" → next day
- "mỗi ngày/every day/daily" → repeat=daily
- "mỗi tuần/every week/weekly" → repeat=weekly
- "mỗi tháng/every month/monthly" → repeat=monthly
- If no date + no repeat → assume today, if time already passed → tomorrow
- If not a reminder → is_reminder: false

User message: {message}"""


async def parse_reminder_nlp(user_id: str, user_text: str) -> str | None:
    """
    Parse natural language reminder.
    Returns confirm string nếu tạo được, None nếu không phải reminder.
    """
    now     = int(datetime.now(TZ).timestamp())
    now_str = datetime.now(TZ).strftime("%H:%M %d/%m/%Y %A")

    async with httpx.AsyncClient() as http:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)
        try:
            resp = await client.chat.completions.create(
                model=MODEL_REGISTRY["llama8b"]["model_id"],
                messages=[{
                    "role": "user",
                    "content": _PARSE_REMINDER_PROMPT.format(
                        now=now, now_str=now_str, message=user_text[:300]
                    ),
                }],
                temperature=0.0,
                max_tokens=100,
            )
            text = resp.choices[0].message.content or ""
            text = re.sub(r"```[a-z]*\n?|```", "", text).strip()
            data = json.loads(text)

            if not data.get("is_reminder"):
                return None

            fire_at = int(data["fire_at"])
            message = data["message"]
            repeat  = data.get("repeat")

            rid = await save_reminder(user_id, message, fire_at, repeat)

            fire_dt    = datetime.fromtimestamp(fire_at, tz=TZ)
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
    """Background task — kiểm tra reminder mỗi 30 giây."""
    while True:
        await asyncio.sleep(30)
        now = int(datetime.now(TZ).timestamp())

        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT id, user_id, message, repeat FROM reminders "
                "WHERE fire_at <= ? AND done = 0",
                (now,),
            ) as cur:
                rows = await cur.fetchall()

            for rid, user_id, message, repeat in rows:
                try:
                    repeat_label = {
                        "daily":   " (hàng ngày)",
                        "weekly":  " (hàng tuần)",
                        "monthly": " (hàng tháng)",
                    }.get(repeat or "", "")
                    async with AsyncApiClient(line_config) as api_client:
                        line_api = AsyncMessagingApi(api_client)
                        await line_api.push_message(
                            PushMessageRequest(
                                to=user_id,
                                messages=[TextMessage(
                                    text=f"⏰ Nhắc nhở{repeat_label}: {message}"
                                )]
                            )
                        )
                except Exception:
                    pass

                if repeat:
                    next_fire = _next_fire(now, repeat)
                    await db.execute(
                        "UPDATE reminders SET fire_at = ? WHERE id = ?",
                        (next_fire, rid),
                    )
                else:
                    await db.execute(
                        "UPDATE reminders SET done = 1 WHERE id = ?", (rid,)
                    )

            await db.commit()


# ---------------------------------------------------------------------------
# MARKDOWN STRIPPER
# ---------------------------------------------------------------------------
def strip_markdown(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"^[\-\*]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"^(\-{3,}|\*{3,}|_{3,})\s*$", "─────", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# GROQ CALLERS
# ---------------------------------------------------------------------------
async def call_groq_text(
    history: list[dict],
    model_id: str,
    model_key: str = DEFAULT_MODEL_KEY,
) -> str:
    system = get_system_prompt(model_key)

    # Compound models require last message role to be "user"
    clean_history = list(history)
    while clean_history and clean_history[-1]["role"] == "assistant":
        clean_history.pop()

    async with httpx.AsyncClient() as http:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": system}] + clean_history,
                temperature=0.6,
                max_tokens=800,
            )
            return strip_markdown(resp.choices[0].message.content or "")
        except Exception as e:
            err = str(e)
            # Compound 400 error → fallback to llama70b
            if "400" in err and model_id.startswith("groq/compound"):
                fallback_id = MODEL_REGISTRY["llama70b"]["model_id"]
                try:
                    resp = await client.chat.completions.create(
                        model=fallback_id,
                        messages=[{"role": "system", "content": system}] + clean_history,
                        temperature=0.6,
                        max_tokens=800,
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
                                "type": "image_url",
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

    if cmd == "clear":
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
            await db.execute("DELETE FROM summary WHERE user_id = ?", (user_id,))
            await db.commit()
        return "🗑 對話記錄已清除。"

    if cmd == "models":
        return _models_list_text()

    if cmd == "auto":
        await set_user_model(user_id, DEFAULT_MODEL_KEY)
        return "🤖 已切換至自動選擇模型模式。"

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

    # ── REMIND COMMANDS ────────────────────────────────────────────────────
    if cmd == "remind":
        # /remind list
        if arg == "list":
            reminders = await get_reminders(user_id)
            if not reminders:
                return "📭 Không có nhắc nhở nào đang chờ."
            lines = ["📋 Danh sách nhắc nhở:\n"]
            for r in reminders:
                dt = datetime.fromtimestamp(r["fire_at"])
                repeat_label = {
                    "daily":   " 🔁 hàng ngày",
                    "weekly":  " 🔁 hàng tuần",
                    "monthly": " 🔁 hàng tháng",
                }.get(r["repeat"] or "", "")
                lines.append(
                    f"#{r['id']}{repeat_label}\n"
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

        # /remind HH:MM [daily|weekly|monthly] nội dung
        time_match = re.match(r"(\d{1,2}):(\d{2})\s+(.*)", arg)
        if time_match:
            hour   = int(time_match[1])
            minute = int(time_match[2])
            rest   = time_match[3].strip()

            repeat     = None
            repeat_map = {
                "daily": "daily", "weekly": "weekly", "monthly": "monthly",
                "hàng ngày": "daily", "hàng tuần": "weekly", "hàng tháng": "monthly",
            }
            for kw, val in repeat_map.items():
                if rest.lower().startswith(kw):
                    repeat = val
                    rest   = rest[len(kw):].strip()
                    break

            now_dt  = datetime.now(TZ)
            fire_dt = now_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if fire_dt <= now_dt:
                fire_dt += timedelta(days=1)

            rid        = await save_reminder(user_id, rest, int(fire_dt.timestamp()), repeat)
            repeat_str = {
                "daily": " (lặp hàng ngày)",
                "weekly": " (lặp hàng tuần)",
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

    # ── MODEL SHORTCUT ──────────────────────────────────────────────────────
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
# FASTAPI
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

        try:
            await line_api.show_loading_animation(
                ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=10)
            )
        except Exception:
            pass

        reply = ""

        # ── AUDIO ──────────────────────────────────────────────────────────
        if isinstance(event.message, AudioMessageContent):
            audio_bytes = await line_blob_api.get_message_content(event.message.id)
            transcript  = await call_groq_whisper(audio_bytes)

            if "⚠️" not in transcript:
                wants_reply = any(
                    transcript.strip().lower().startswith(t.lower())
                    for t in _REPLY_TRIGGERS
                )
                if wants_reply:
                    clean_text = transcript.strip()
                    for t in _REPLY_TRIGGERS:
                        if clean_text.lower().startswith(t.lower()):
                            clean_text = clean_text[len(t):].strip()
                            break
                    await save_message(user_id, "user", clean_text)
                    model_key, model_id = await resolve_model(user_id, clean_text)
                    history = await get_history_with_summary(user_id)
                    answer  = await call_groq_text(history, model_id, model_key=model_key)
                    await save_message(user_id, "assistant", answer)
                    await maybe_summarize(user_id)
                    reply = f"🎤 {clean_text}\n\n{answer}"
                else:
                    await save_message(user_id, "user", f"[Voice]: {transcript}")
                    reply = f"🎤 {transcript}"
            else:
                reply = transcript

        # ── IMAGE ──────────────────────────────────────────────────────────
        elif isinstance(event.message, ImageMessageContent):
            img_bytes = await line_blob_api.get_message_content(event.message.id)
            img_b64   = base64.b64encode(img_bytes).decode("utf-8")
            answer    = await call_groq_vision(img_b64)
            # Lưu nội dung ảnh vào history để text model sau đọc được context
            await save_message(user_id, "user", f"[Ảnh] {answer}")
            await save_message(user_id, "assistant", answer)
            await maybe_summarize(user_id)
            reply = answer

        # ── TEXT ───────────────────────────────────────────────────────────
        elif isinstance(event.message, TextMessageContent):
            user_text = event.message.text.strip()

            cmd_reply = await handle_command(user_id, user_text)
            if cmd_reply is not None:
                reply = cmd_reply
            else:
                # Block input quá dài
                if len(user_text) > MAX_INPUT_CHARS:
                    reply = (
                        f"⚠️ Tin nhắn quá dài ({len(user_text)} ký tự).\n"
                        f"Vui lòng giới hạn dưới {MAX_INPUT_CHARS} ký tự."
                    )
                else:
                    # Natural language reminder detection
                    reminder_reply = await parse_reminder_nlp(user_id, user_text)
                    if reminder_reply:
                        reply = reminder_reply
                    else:
                        model_key, model_id = await resolve_model(user_id, user_text)

                        # Text dài không có câu hỏi → inject tóm tắt instruction
                        if len(user_text) > 500 and "?" not in user_text and "？" not in user_text:
                            await save_message(user_id, "user", user_text)
                            history = [{"role": "user", "content": f"Hãy tóm tắt nội dung sau:\n{user_text}"}]
                        else:
                            await save_message(user_id, "user", user_text)
                            history = await get_history_with_summary(user_id)

                        answer = await call_groq_text(history, model_id, model_key=model_key)
                        await save_message(user_id, "assistant", answer)
                        await maybe_summarize(user_id)
                        reply  = answer

        # ── REPLY ──────────────────────────────────────────────────────────
        if reply:
            reply = reply[:4990] + "…" if len(reply) > 4990 else reply
            await line_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply)],
                )
            )
