import os, hmac, hashlib, base64, asyncio, aiosqlite, httpx
from contextlib import asynccontextmanager
from groq import AsyncGroq
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from linebot.v3 import WebhookParser
from linebot.v3.messaging import (
    AsyncApiClient, AsyncMessagingApi, AsyncMessagingApiBlob, Configuration, 
    ReplyMessageRequest, TextMessage, ShowLoadingAnimationRequest
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent, AudioMessageContent

# --- CONFIG ---
try:
    from prompts import SYSTEM_PROMPT
except ImportError:
    SYSTEM_PROMPT = "Bạn là trợ lý thông minh. Nếu dùng tiếng Trung hãy dùng Phồn thể."

LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
DB_PATH = "chat_history.db"

TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "llama-3.2-11b-vision-preview"
WHISPER_MODEL = "whisper-large-v3"

# --- DATABASE HELPERS ---
async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, role TEXT, content TEXT)")
        await db.commit()

async def get_history(user_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT role, content FROM history WHERE user_id = ? ORDER BY id DESC LIMIT 10") as cur:
            rows = await cur.fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]

async def save_message(user_id: str, role: str, content: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
        await db.commit()

# --- ASYNC GROQ HANDLERS ---
async def call_groq_text(history):
    async with httpx.AsyncClient() as http_client:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http_client)
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
            resp = await client.chat.completions.create(
                model=TEXT_MODEL, 
                messages=messages, 
                temperature=0.6,
                max_tokens=800
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"⚠️ Lỗi kết nối văn bản: {str(e)[:100]}"

async def call_groq_vision(image_base64):
    async with httpx.AsyncClient() as http_client:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http_client)
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Kiểm tra và giải thích nội dung ảnh này (ưu tiên tiếng Phồn thể nếu là tiếng Trung)."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }]
            resp = await client.chat.completions.create(model=VISION_MODEL, messages=messages, max_tokens=800)
            return resp.choices[0].message.content
        except Exception as e:
            return f"⚠️ Lỗi phân tích ảnh: {str(e)[:100]}"

async def call_groq_whisper(audio_bytes):
    async with httpx.AsyncClient() as http_client:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http_client)
        try:
            translation = await client.audio.transcriptions.create(
                file=("audio.m4a", audio_bytes), 
                model=WHISPER_MODEL
            )
            return translation.text
        except Exception as e:
            return f"⚠️ Lỗi nhận diện giọng nói: {str(e)[:100]}"

# --- FASTAPI APP ---
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
    except:
        raise HTTPException(status_code=400)
        
    for event in events:
        if isinstance(event, MessageEvent):
            background_tasks.add_task(process_event, event)
    return JSONResponse({"status": "ok"})

async def process_event(event):
    user_id = event.source.user_id
    async with AsyncApiClient(line_config) as api_client:
        line_api = AsyncMessagingApi(api_client)
        line_blob_api = AsyncMessagingApiBlob(api_client)  # <-- Đã thêm API tải file
        
        try:
            await line_api.show_loading_animation(ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=5))
        except: pass
        
        reply = ""
        
        # 1. XỬ LÝ GHI ÂM
        if isinstance(event.message, AudioMessageContent):
            # Sử dụng line_blob_api thay vì line_api
            audio_content = await line_blob_api.get_message_content(event.message.id)
            text_from_voice = await call_groq_whisper(audio_content)
            
            if "⚠️" not in text_from_voice:
                await save_message(user_id, "user", f"[Voice to Text]: {text_from_voice}")
                history = await get_history(user_id)
                reply_text = await call_groq_text(history)
                reply = f"🎤 Bạn nói: {text_from_voice}\n\n🤖 {reply_text}"
            else:
                reply = text_from_voice
        
        # 2. XỬ LÝ ẢNH
        elif isinstance(event.message, ImageMessageContent):
            # Sử dụng line_blob_api thay vì line_api
            img_content = await line_blob_api.get_message_content(event.message.id)
            img_b64 = base64.b64encode(img_content).decode('utf-8')
            reply = await call_groq_vision(img_b64)
        
        # 3. XỬ LÝ VĂN BẢN
        elif isinstance(event.message, TextMessageContent):
            user_text = event.message.text.strip()
            
            if user_text == "/clear":
                async with aiosqlite.connect(DB_PATH) as db:
                    await db.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
                    await db.commit()
                reply = "🗑 Đã xoá lịch sử."
            else:
                await save_message(user_id, "user", user_text)
                history = await get_history(user_id)
                reply = await call_groq_text(history)
                await save_message(user_id, "assistant", reply)

        if reply:
            await line_api.reply_message(ReplyMessageRequest(
                reply_token=event.reply_token, 
                messages=[TextMessage(text=reply)]
            ))