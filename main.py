"""
LINE Bot + Groq API — single file, production-ready
Stack: FastAPI + line-bot-sdk v3 + aiosqlite + groq
"""

import os
import hmac
import hashlib
import base64
import asyncio
import aiosqlite
from contextlib import asynccontextmanager
from groq import Groq
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from linebot.v3 import WebhookParser
from linebot.v3.messaging import (
    AsyncApiClient,
    AsyncMessagingApi,
    Configuration,
    ReplyMessageRequest,
    TextMessage,
    ShowLoadingAnimationRequest,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

# ── Config from environment variables ──────────────────────────────────────────
LINE_CHANNEL_SECRET      = os.environ["LINE_CHANNEL_SECRET"]
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
GROQ_API_KEY             = os.environ["GROQ_API_KEY"]

DB_PATH          = "chat_history.db"
MAX_HISTORY      = 20       # số messages lưu per user
DEFAULT_MODEL    = "llama-3.3-70b-versatile"
SYSTEM_PROMPT    = "Bạn là trợ lý AI thân thiện, trả lời ngắn gọn bằng tiếng Việt."

# ── Available models (verified March 2026) ─────────────────────────────────────
AVAILABLE_MODELS: dict[str, str] = {
    # alias       : model_id
    "fast"        : "openai/gpt-oss-20b",
    "balanced"    : "llama-3.3-70b-versatile",
    "quality"     : "openai/gpt-oss-120b",
    "reasoning"   : "openai/gpt-oss-120b",
    "scout"       : "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen"        : "qwen/qwen3-32b",
    "kimi"        : "moonshotai/kimi-k2-instruct-0905",
    "8b"          : "llama-3.1-8b-instant",
    "70b"         : "llama-3.3-70b-versatile",
    "120b"        : "openai/gpt-oss-120b",
    "20b"         : "openai/gpt-oss-20b",
}

# ── DB helpers ─────────────────────────────────────────────────────────────────
async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   TEXT NOT NULL,
                role      TEXT NOT NULL,
                content   TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id TEXT PRIMARY KEY,
                model   TEXT NOT NULL DEFAULT 'llama-3.3-70b-versatile'
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_user ON history(user_id, id)")
        await db.commit()


async def get_history(user_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """SELECT role, content FROM history
               WHERE user_id = ?
               ORDER BY id DESC LIMIT ?""",
            (user_id, MAX_HISTORY),
        ) as cur:
            rows = await cur.fetchall()
    # reverse để đúng thứ tự cũ → mới
    return [{"role": r, "content": c} for r, c in reversed(rows)]


async def save_message(user_id: str, role: str, content: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content),
        )
        # Giữ MAX_HISTORY * 2 rows gần nhất, xoá cũ
        await db.execute(
            """DELETE FROM history WHERE user_id = ? AND id NOT IN (
                   SELECT id FROM history WHERE user_id = ?
                   ORDER BY id DESC LIMIT ?
               )""",
            (user_id, user_id, MAX_HISTORY * 2),
        )
        await db.commit()


async def clear_history(user_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
        await db.commit()


async def get_model(user_id: str) -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT model FROM user_settings WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else DEFAULT_MODEL


async def set_model(user_id: str, model: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO user_settings (user_id, model) VALUES (?, ?)
               ON CONFLICT(user_id) DO UPDATE SET model = excluded.model""",
            (user_id, model),
        )
        await db.commit()


# ── Groq call ──────────────────────────────────────────────────────────────────
def call_groq(history: list[dict], model: str) -> str:
    """Blocking Groq call — LINE không support streaming reply."""
    client = Groq(api_key=GROQ_API_KEY)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


# ── LINE signature verification ────────────────────────────────────────────────
def verify_signature(body: bytes, signature: str) -> bool:
    mac = hmac.new(
        LINE_CHANNEL_SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).digest()
    return hmac.compare_digest(
        base64.b64encode(mac).decode("utf-8"),
        signature,
    )


# ── App lifecycle ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    print("✅ DB initialized")
    yield


app = FastAPI(lifespan=lifespan)

# LINE SDK setup
parser = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
webhook_parser = WebhookParser(LINE_CHANNEL_SECRET)
line_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)


# ── Command handler ────────────────────────────────────────────────────────────
async def handle_command(user_id: str, text: str) -> str | None:
    """
    Trả về reply string nếu là command, None nếu không phải.
    """
    cmd = text.strip().lower()

    # /clear
    if cmd == "/clear":
        await clear_history(user_id)
        return "🗑 Đã xoá lịch sử chat."

    # /help
    if cmd == "/help":
        model_list = "\n".join(
            f"  • {alias} → {mid}" for alias, mid in AVAILABLE_MODELS.items()
        )
        return (
            "📖 Các lệnh hỗ trợ:\n\n"
            "/clear — Xoá lịch sử\n"
            "/model <alias> — Đổi model\n"
            "/model list — Xem tất cả model\n"
            "/currentmodel — Model đang dùng\n"
            "/help — Xem trợ giúp\n\n"
            f"Alias model:\n{model_list}"
        )

    # /currentmodel
    if cmd == "/currentmodel":
        m = await get_model(user_id)
        return f"🤖 Model hiện tại: {m}"

    # /model list
    if cmd == "/model list":
        lines = "\n".join(
            f"  {alias:12} → {mid}" for alias, mid in AVAILABLE_MODELS.items()
        )
        return f"📋 Danh sách model:\n\n{lines}"

    # /model <alias hoặc model_id>
    if cmd.startswith("/model "):
        arg = text.strip()[7:].strip()  # lấy phần sau "/model "

        # khớp alias
        target = AVAILABLE_MODELS.get(arg.lower())

        # hoặc nhập thẳng model_id
        if not target:
            all_ids = set(AVAILABLE_MODELS.values())
            if arg in all_ids:
                target = arg

        if target:
            await set_model(user_id, target)
            return f"✅ Đã đổi model sang:\n{target}"
        else:
            return (
                f"❌ Không tìm thấy model '{arg}'.\n"
                "Gõ /model list để xem danh sách."
            )

    return None  # không phải command


# ── Main webhook endpoint ──────────────────────────────────────────────────────
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    body      = await request.body()
    signature = request.headers.get("X-Line-Signature", "")

    # Verify chữ ký LINE
    if not verify_signature(body, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Parse events
    try:
        events = webhook_parser.parse(body.decode("utf-8"), signature)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    for event in events:
        if not isinstance(event, MessageEvent):
            continue
        if not isinstance(event.message, TextMessageContent):
            continue

        background_tasks.add_task(process_message, event)

    return JSONResponse({"status": "ok"})


async def process_message(event: MessageEvent) -> None:
    user_id     = event.source.user_id
    user_text   = event.message.text
    reply_token = event.reply_token

    async with AsyncApiClient(line_config) as api_client:
        line_api = AsyncMessagingApi(api_client)

        # Typing indicator (loading animation)
        try:
            await line_api.show_loading_animation(
                ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=10)
            )
        except Exception:
            pass  # không critical nếu fail

        # Kiểm tra command
        command_reply = await handle_command(user_id, user_text)
        if command_reply:
            await line_api.reply_message(
                ReplyMessageRequest(
                    reply_token=reply_token,
                    messages=[TextMessage(text=command_reply)],
                )
            )
            return

        # Chat bình thường
        await save_message(user_id, "user", user_text)
        history = await get_history(user_id)
        model   = await get_model(user_id)

        # Gọi Groq trong thread pool (blocking call)
        try:
            reply_text = await asyncio.get_event_loop().run_in_executor(
                None, call_groq, history, model
            )
        except Exception as e:
            reply_text = f"⚠️ Lỗi Groq: {str(e)[:200]}"

        await save_message(user_id, "assistant", reply_text)

        await line_api.reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[TextMessage(text=reply_text)],
            )
        )


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/")
async def health():
    return {"status": "ok", "bot": "LINE + Groq"}
