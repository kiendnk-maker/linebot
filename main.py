import os
import hmac
import hashlib
import base64
import asyncio
import aiosqlite
import httpx
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

# Import Character từ file riêng để dễ quản lý
try:
    from prompts import SYSTEM_PROMPT
except ImportError:
    SYSTEM_PROMPT = "Bạn là trợ lý AI thông minh, trả lời ngắn gọn bằng tiếng Việt."

# ── Config từ environment variables ──────────────────────────────────────────
LINE_CHANNEL_SECRET      = os.environ["LINE_CHANNEL_SECRET"]
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
GROQ_API_KEY             = os.environ["GROQ_API_KEY"]

DB_PATH          = "chat_history.db"
MAX_HISTORY      = 10       # Tối ưu Latency theo Groq Cookbook
DEFAULT_MODEL    = "llama-3.3-70b-versatile"

# ── Models chính thức và ổn định ──────────────────────────────────────────────
AVAILABLE_MODELS: dict[str, str] = {
    "fast"        : "llama-3.1-8b-instant",
    "balanced"    : "llama-3.3-70b-versatile",
    "70b"         : "llama-3.3-70b-versatile",
    "8b"          : "llama-3.1-8b-instant",
}

# ── DB Helpers ─────────────────────────────────────────────────────────────────
async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, role TEXT, content TEXT)")
        await db.execute("CREATE TABLE IF NOT EXISTS user_settings (user_id TEXT PRIMARY KEY, model TEXT DEFAULT 'llama-3.3-70b-versatile')")
        await db.commit()

async def get_history(user_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT role, content FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, MAX_HISTORY)) as cur:
            rows = await cur.fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]

async def save_message(user_id: str, role: str, content: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
        await db.commit()

async def get_model(user_id: str) -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT model FROM user_settings WHERE user_id = ?", (user_id,)) as cur:
            row = await cur.fetchone()
    return row[0] if row else DEFAULT_MODEL

async def set_model(user_id: str, model: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT INTO user_settings (user_id, model) VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET model=excluded.model", (user_id, model))
        await db.commit()

# ── Groq Call (Tối ưu theo Latency Cookbook) ──────────────────────────────────
def call_groq(history: list[dict], model: str) -> str:
    """Sửa lỗi Proxies và áp dụng tối ưu TTFT (Time To First Token)"""
    http_client = httpx.Client() # Fix lỗi Client.__init__ proxies trên Railway
    client = Groq(api_key=GROQ_API_KEY, http_client=http_client)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=800,  # Giảm tokens để tăng tốc độ phản hồi
            temperature=0.6, # Cân bằng giữa sáng tạo và tốc độ
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Lỗi Groq API: {str(e)[:100]}"
    finally:
        http_client.close()

# ── Webhook & Logic ────────────────────────────────────────────────────────────
def verify_signature(body: bytes, signature: str) -> bool:
    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    return hmac.compare_digest(base64.b64encode(mac).decode("utf-8"), signature)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(lifespan=lifespan)
line_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
webhook_parser = WebhookParser(LINE_CHANNEL_SECRET)

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    if not verify_signature(body, signature): raise HTTPException(status_code=400)
    
    try:
        events = webhook_parser.parse(body.decode("utf-8"), signature)
    except: raise HTTPException(status_code=400)

    for event in events:
        if isinstance(event, MessageEvent) and isinstance(event.message, TextMessageContent):
            background_tasks.add_task(process_message, event)
    return JSONResponse({"status": "ok"})

async def process_message(event: MessageEvent):
    user_id = event.source.user_id
    user_text = event.message.text
    
    async with AsyncApiClient(line_config) as api_client:
        line_api = AsyncMessagingApi(api_client)
        try:
            await line_api.show_loading_animation(ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=5))
        except: pass

        # Xử lý lệnh đơn giản
        if user_text.lower() == "/help":
            reply = "📖 Lệnh: /clear (Xoá nhớ), /model list (Xem model), /model <tên> (Đổi model)"
        elif user_text.lower() == "/model list":
            reply = f"📋 Model: {', '.join(AVAILABLE_MODELS.keys())}"
        elif user_text.lower().startswith("/model "):
            m_name = user_text[7:].strip().lower()
            if m_name in AVAILABLE_MODELS:
                await set_model(user_id, AVAILABLE_MODELS[m_name])
                reply = f"✅ Đã đổi sang {m_name}"
            else: reply = "❌ Model không tồn tại."
        elif user_text.lower() == "/clear":
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
                await db.commit()
            reply = "🗑 Đã xoá lịch sử chat."
        else:
            # Chat với AI
            await save_message(user_id, "user", user_text)
            history = await get_history(user_id)
            model = await get_model(user_id)
            reply = await asyncio.get_event_loop().run_in_executor(None, call_groq, history, model)
            await save_message(user_id, "assistant", reply)

        await line_api.reply_message(ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=reply)]
        ))

@app.get("/")
async def health(): return {"status": "optimized", "character": "Mentor Groq哥哥"}
