import os
import logging

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from linebot.v3 import WebhookParser
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, AsyncApiClient, AsyncMessagingApi, AsyncMessagingApiBlob,
    ReplyMessageRequest, TextMessage, ShowLoadingAnimationRequest,
    QuickReply, QuickReplyItem, MessageAction
)
from linebot.v3.webhooks import (
    MessageEvent, TextMessageContent, AudioMessageContent, ImageMessageContent
)

from database import init_db, DB_PATH, save_message
from llm_core import (
    resolve_model, call_mistral_text, call_mistral_vision,
    call_groq_whisper, clean_transcript, get_history_with_summary,
    strip_markdown
)
from command_handler import handle_command
from rag_core import rag_search, has_rag_docs
from reminder_system import parse_reminder_nlp

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- LINE CONFIG ---
line_config = Configuration(access_token=os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", ""))
parser = WebhookParser(os.environ.get("LINE_CHANNEL_SECRET", ""))

# --- APP ---
app = FastAPI()
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup():
    await init_db()

# --- STATE MANAGEMENT ---
_rag_disabled = set()
_pending_choice: dict[str, str] = {}
_REPLY_TRIGGERS = ["bot", "ai", "bolt", "trợ lý", "em ơi", "bạn ơi"]

async def _process_event_inner(event: MessageEvent) -> None:
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
        qr_items = None

        # ── AUDIO PIPELINE ──
        if isinstance(event.message, AudioMessageContent):
            audio_bytes = await line_blob_api.get_message_content(event.message.id)
            transcript  = await call_groq_whisper(audio_bytes)

            if "⚠️" in transcript:
                reply = transcript
            else:
                transcript = await clean_transcript(transcript)
                logger.info(f"AUDIO cleaned | user={user_id} | text={transcript[:50]!r}")

                wants_reply = any(transcript.strip().lower().startswith(t.lower()) for t in _REPLY_TRIGGERS)

                if wants_reply:
                    clean_text = transcript.strip()
                    for t in _REPLY_TRIGGERS:
                        if clean_text.lower().startswith(t.lower()):
                            clean_text = clean_text[len(t):]
                            break
                    clean_text = clean_text.lstrip(",.! ")

                    reminder_reply = await parse_reminder_nlp(user_id, clean_text)
                    if reminder_reply:
                        reply = f"🎤 {clean_text}\n\n{reminder_reply}"
                    else:
                        model_key, model_id = await resolve_model(user_id, clean_text)
                        history = await get_history_with_summary(user_id)

                        rag_chunks = []
                        if user_id not in _rag_disabled and await has_rag_docs(user_id):
                            rag_chunks = await rag_search(user_id, clean_text)

                        history.append({"role": "user", "content": clean_text})
                        if rag_chunks:
                            ctx_str = "\n".join(rag_chunks)
                            history[-1]["content"] = f"Context:\n{ctx_str}\n\nQuestion: {clean_text}"

                        answer = await call_mistral_text(history, model_id, model_key=model_key)
                        await save_message(user_id, "user", clean_text)
                        await save_message(user_id, "assistant", answer)
                        reply = f"🎤 {clean_text}\n\n🤖 {answer}"
                else:
                    import aiosqlite
                    async with aiosqlite.connect(DB_PATH) as db:
                        cur = await db.execute(
                            "INSERT INTO audio_cache (user_id, transcript, filename) VALUES (?, ?, ?) RETURNING id",
                            (user_id, transcript, f"audio_{event.message.id}.txt")
                        )
                        audio_id = (await cur.fetchone())[0]
                        await db.commit()

                    _pending_choice[user_id] = f"audio:{audio_id}"
                    reply = f"🎤 Đã bóc băng:\n{transcript}"
                    qr_items = [
                        QuickReplyItem(action=MessageAction(label="1️⃣ Tóm tắt", text="1")),
                        QuickReplyItem(action=MessageAction(label="2️⃣ Lưu RAG", text="2")),
                        QuickReplyItem(action=MessageAction(label="3️⃣ Cả hai", text="3"))
                    ]

        # ── IMAGE PIPELINE ──
        elif isinstance(event.message, ImageMessageContent):
            img_bytes = await line_blob_api.get_message_content(event.message.id)
            import base64
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            reply = await call_mistral_vision(img_b64)
            reply = f"👁️ Vision (Pixtral):\n{reply}"

        # ── TEXT PIPELINE ──
        elif isinstance(event.message, TextMessageContent):
            user_text = event.message.text.strip()

            pending = _pending_choice.get(user_id)
            cmd_reply = None

            # --- XỬ LÝ QUICK REPLY (1, 2, 3) DỰA THEO STATE ---
            if user_text.isdigit() and len(user_text) <= 2 and pending:
                if pending == "mail":
                    cmd_reply = await handle_command(user_id, f"/mail {user_text}")
                elif pending.startswith("audio:"):
                    audio_id = pending.split(":", 1)[1]
                    cmd_reply = await handle_command(user_id, f"/audio {audio_id} {user_text}")
                _pending_choice.pop(user_id, None)

            # --- XỬ LÝ LỆNH BÌNH THƯỜNG ---
            else:
                cmd_reply = await handle_command(user_id, user_text)

                cmd_check = user_text.strip().lower()
                if cmd_check in ["/mail", "/ls mail", "mail"] and cmd_reply and "1" in cmd_reply:
                    _pending_choice[user_id] = "mail"
                    qr_items = [QuickReplyItem(action=MessageAction(label=f"{i}️⃣ Đọc mail {i}", text=str(i))) for i in range(1, 6)]
                elif cmd_check.startswith("/audio "):
                    pass
                elif cmd_reply:
                    _pending_choice.pop(user_id, None)

            if cmd_reply is not None:
                reply = cmd_reply
            else:
                model_key, model_id = await resolve_model(user_id, user_text)
                history = await get_history_with_summary(user_id)

                rag_chunks = []
                if user_id not in _rag_disabled and await has_rag_docs(user_id):
                    rag_chunks = await rag_search(user_id, user_text)

                history.append({"role": "user", "content": user_text})
                if rag_chunks:
                    ctx_str = "\n".join(rag_chunks)
                    history[-1]["content"] = f"Context:\n{ctx_str}\n\nQuestion: {user_text}"

                answer = await call_mistral_text(history, model_id, model_key=model_key)
                await save_message(user_id, "user", user_text)
                await save_message(user_id, "assistant", answer)
                reply = answer

        # ── TRẢ LỜI NGƯỜI DÙNG ──
        if reply:
            try:
                # Strip markdown trước khi gửi (LINE không render Markdown)
                reply = strip_markdown(reply)
                chunks = [reply[i:i+5000] for i in range(0, len(reply), 5000)]
                messages = [TextMessage(text=c) for c in chunks]

                if qr_items and messages:
                    messages[-1] = TextMessage(text=chunks[-1], quick_reply=QuickReply(items=qr_items))

                await line_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=messages
                    )
                )
            except Exception as e:
                logger.error(f"Reply error: {e}")

@app.post("/webhook")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try:
        events = parser.parse(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        return JSONResponse(status_code=400, content={"message": "Invalid signature"})

    for event in events:
        if isinstance(event, MessageEvent):
            background_tasks.add_task(_process_event_inner, event)

    return "OK"
