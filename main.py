import os, hmac, hashlib, base64, asyncio, aiosqlite, httpx
from contextlib import asynccontextmanager
from groq import Groq
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from linebot.v3 import WebhookParser
from linebot.v3.messaging import (
    AsyncApiClient, AsyncMessagingApi, Configuration, 
    ReplyMessageRequest, TextMessage, ShowLoadingAnimationRequest
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent

# --- CẤU HÌNH HỆ THỐNG ---
try:
    from prompts import SYSTEM_PROMPT
except ImportError:
    SYSTEM_PROMPT = "Bạn là trợ lý AI thông minh, phản hồi ngắn gọn bằng ngôn ngữ người dùng sử dụng."

LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
DB_PATH = "chat_history.db"
MAX_HISTORY = 10  # Tối ưu Latency theo Groq Cookbook
DEFAULT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "llama-3.2-11b-vision-preview"

AVAILABLE_MODELS = {
    "fast": "llama-3.1-8b-instant",
    "balanced": "llama-3.3-70b-versatile",
    "70b": "llama-3.3-70b-versatile",
    "8b": "llama-3.1-8b-instant",
    "vision": "llama-3.2-11b-vision-preview"
}

# --- DATABASE HELPERS ---
async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, role TEXT, content TEXT)")
        await db.execute("CREATE TABLE IF NOT EXISTS user_settings (user_id TEXT PRIMARY KEY, model TEXT DEFAULT 'llama-3.3-70b-versatile')")
        await db.commit()

async def get_history(user_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT role, content FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, MAX_HISTORY)) as cur:
            rows = await cur.fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]

async def save_message(user_id: str, role: str, content: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
        await db.commit()

async def get_user_model(user_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT model FROM user_settings WHERE user_id = ?", (user_id,)) as cur:
            row = await cur.fetchone()
    return row[0] if row else DEFAULT_MODEL

async def set_user_model(user_id: str, model: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT INTO user_settings (user_id, model) VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET model=excluded.model", (user_id, model))
        await db.commit()

# --- XỬ LÝ GROQ API ---
def call_groq(history, model, image_data=None):
    http_client = httpx.Client() # Fix lỗi proxy trên Railway
    client = Groq(api_key=GROQ_API_KEY, http_client=http_client)
    try:
        if image_data:
            # Chế độ Vision: Nhận diện ảnh
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Mô tả ảnh này theo phong cách nhân vật của bạn trong prompts."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            }]
            target_model = VISION_MODEL
        else:
            # Chế độ Text: Hội thoại thông thường
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
            target_model = model

        resp = client.chat.completions.create(
            model=target_model,
            messages=messages,
            max_tokens=800,
            temperature=0.6
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Lỗi Groq: {str(e)[:100]}"
    finally:
        http_client.close()

# --- FASTAPI & WEBHOOK ---
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
    try:
        events = webhook_parser.parse(body.decode("utf-8"), signature)
    except: raise HTTPException(status_code=400)
    for event in events:
        if isinstance(event, MessageEvent):
            background_tasks.add_task(process_event, event)
    return JSONResponse({"status": "ok"})

async def process_event(event):
    user_id = event.source.user_id
    async with AsyncApiClient(line_config) as api_client:
        line_api = AsyncMessagingApi(api_client)
        
        # Hiện hiệu ứng đang soạn tin
        try:
            await line_api.show_loading_animation(ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=5))
        except: pass
        
        reply_text = ""
        
        # 1. XỬ LÝ ẢNH
        if isinstance(event.message, ImageMessageContent):
            image_content = await line_api.get_message_content(event.message.id)
            image_base64 = base64.b64encode(image_content).decode('utf-8')
            reply_text = await asyncio.get_event_loop().run_in_executor(None, call_groq, None, None, image_base64)
        
        # 2. XỬ LÝ CHỮ & LỆNH
        elif isinstance(event.message, TextMessageContent):
            user_text = event.message.text
            
            if user_text.startswith("/"):
                if user_text == "/clear":
                    async with aiosqlite.connect(DB_PATH) as db:
                        await db.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
                        await db.commit()
                    reply_text = "🗑 Đã xoá lịch sử chat."
                elif user_text == "/model list":
                    reply_text = f"📋 Model: {', '.join(AVAILABLE_MODELS.keys())}"
                elif user_text.startswith("/model "):
                    m_name = user_text[7:].strip().lower()
                    if m_name in AVAILABLE_MODELS:
                        await set_user_model(user_id, AVAILABLE_MODELS[m_name])
                        reply_text = f"✅ Đã đổi sang {m_name}"
                    else: reply_text = "❌ Model không tồn tại."
                else:
                    reply_text = "📖 Lệnh: /clear, /model list, /model <tên>"
            else:
                await save_message(user_id, "user", user_text)
                history = await get_history(user_id)
                model = await get_user_model(user_id)
                reply_text = await asyncio.get_event_loop().run_in_executor(None, call_groq, history, model)
                await save_message(user_id, "assistant", reply_text)

        if reply_text:
            await line_api.reply_message(ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            ))

@app.get("/")
async def health(): return {"status": "running", "vision": "enabled", "memory": MAX_HISTORY}