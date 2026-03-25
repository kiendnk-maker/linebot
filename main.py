import os
import time
import base64
import logging
import asyncio
import aiosqlite
import datetime

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from linebot.v3 import WebhookParser
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, AsyncApiClient, AsyncMessagingApi, AsyncMessagingApiBlob,
    ReplyMessageRequest, PushMessageRequest, TextMessage, ShowLoadingAnimationRequest,
    QuickReply, QuickReplyItem, MessageAction
)
from linebot.v3.webhooks import (
    MessageEvent, TextMessageContent, AudioMessageContent, ImageMessageContent
)

from database import init_db, DB_PATH, save_message, get_pending_image, clear_pending_image
from llm_core import (
    resolve_model, call_mistral_text, call_mistral_vision,
    call_groq_whisper, clean_transcript, get_history_with_summary,
    strip_markdown
)
from command_handler import handle_command
from rag_core import rag_search, has_rag_docs, warmup_chromadb
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


# ── CLEANUP LOOP — xóa image_cache hết hạn mỗi 5 phút ──────────────────────
async def image_cache_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(300)
        cutoff = int(time.time()) - 600  # 10 phút
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                async with db.execute(
                    "SELECT id, user_id FROM image_cache WHERE created_at < ?", (cutoff,)
                ) as cur:
                    expired = await cur.fetchall()
                if expired:
                    ids      = [r[0] for r in expired]
                    user_ids = [r[1] for r in expired]
                    placeholders = ",".join("?" * len(ids))
                    await db.execute(
                        f"DELETE FROM image_cache WHERE id IN ({placeholders})", ids
                    )
                    await db.commit()
                    for uid in user_ids:
                        try:
                            async with AsyncApiClient(line_config) as api_client:
                                await AsyncMessagingApi(api_client).push_message(
                                    PushMessageRequest(
                                        to=uid,
                                        messages=[TextMessage(
                                            text="⏰ Ảnh đã hết hạn (10 phút). Vui lòng gửi lại."
                                        )]
                                    )
                                )
                        except Exception:
                            pass
        except Exception as e:
            logger.error(f"image_cache_cleanup_loop error: {e}")


@app.on_event("startup")
async def startup():
    await init_db()
    asyncio.create_task(image_cache_cleanup_loop())


@app.get("/warmup")
async def warmup():
    """Pre-warm ChromaDB so the first user request doesn't hit cold-start latency."""
    try:
        await warmup_chromadb()
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"warmup failed: {e}")
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})


# --- STATE MANAGEMENT ---
_rag_disabled   = set()
_pending_choice: dict[str, str] = {}
# _pending_image: dùng DB (image_cache) thay in-memory để survive restart
_VISION_PROMPTS = {
    "1": (
        "請詳細描述這張圖片的內容、色彩、構圖與所有可見元素。"
    ),
    "2": (
        "OCR任務：請將上方AI視覺描述中的所有文字原封不動地輸出。"
        "嚴禁改寫、摘要、翻譯或省略任何文字。"
        "保持原始語言、標點與排版結構。"
        "若同時有多種語言（如中文+越南文），全部照原樣輸出。"
    ),
    "3": (
        "請將圖片中所有文字翻譯成繁體中文。"
        "先輸出原文，再輸出對應的繁體中文翻譯。"
        "格式：[原文] → [繁體中文譯文]"
    ),
    "4": (
        "請深入分析這張圖片的內容、數據或訊息。"
        "若有圖表請解讀趨勢與數據；若有文件請總結重點；若是一般圖片請分析主題與含義。"
    ),
}
_REPLY_TRIGGERS = ["bot", "ai", "bolt", "trợ lý", "em ơi", "bạn ơi"]


async def _process_event_inner(event: MessageEvent) -> None:
    logger.info(f"[EVENT] user={event.source.user_id} msg_type={type(event.message).__name__}")
    user_id = event.source.user_id

    async with AsyncApiClient(line_config) as api_client:
        line_api      = AsyncMessagingApi(api_client)
        line_blob_api = AsyncMessagingApiBlob(api_client)

        try:
            await line_api.show_loading_animation(
                ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=30)
            )
        except Exception:
            pass

        reply    = ""
        qr_items = None

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
                    async with aiosqlite.connect(DB_PATH) as db:
                        cur = await db.execute(
                            "INSERT INTO audio_cache (user_id, transcript, filename)"
                            " VALUES (?, ?, ?) RETURNING id",
                            (user_id, transcript, f"audio_{event.message.id}.txt")
                        )
                        audio_id = (await cur.fetchone())[0]
                        await db.commit()

                    _pending_choice[user_id] = f"audio:{audio_id}"
                    reply = f"🎤 Đã bóc băng:\n{transcript}"
                    qr_items = [
                        QuickReplyItem(action=MessageAction(label="1️⃣ Tóm tắt", text="1")),
                        QuickReplyItem(action=MessageAction(label="2️⃣ Lưu RAG", text="2")),
                        QuickReplyItem(action=MessageAction(label="3️⃣ Cả hai",  text="3")),
                    ]

        # ── IMAGE PIPELINE — Pha 1: Hold & Wait ────────────────────────────
        elif isinstance(event.message, ImageMessageContent):
            img_bytes  = await line_blob_api.get_message_content(event.message.id)
            img_b64    = base64.b64encode(img_bytes).decode("utf-8")
            message_id = event.message.id

            # Nếu đang có ảnh cũ chờ → dọn dẹp trước
            await clear_pending_image(user_id)

            # Lưu ảnh mới vào DB
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "INSERT INTO image_cache (user_id, image_b64, message_id, created_at)"
                    " VALUES (?, ?, ?, ?)",
                    (user_id, img_b64, message_id, int(time.time()))
                )
                await db.commit()

            reply = "📸 Đã nhận ảnh! Bạn muốn làm gì với ảnh này?"
            qr_items = [
                QuickReplyItem(action=MessageAction(label="1️⃣ Mô tả",    text="1")),
                QuickReplyItem(action=MessageAction(label="2️⃣ OCR",       text="2")),
                QuickReplyItem(action=MessageAction(label="3️⃣ Dịch",      text="3")),
                QuickReplyItem(action=MessageAction(label="4️⃣ Phân tích", text="4")),
            ]

        # ── TEXT PIPELINE ───────────────────────────────────────────────────
        elif isinstance(event.message, TextMessageContent):
            user_text = event.message.text.strip()

            # ── VISION INTERCEPT — Pha 2: Activate ──────────────────────
            img_pending = await get_pending_image(user_id)
            if img_pending is not None:
                if user_text.lower() in ("/hủy", "/cancel", "hủy"):
                    await clear_pending_image(user_id)
                    reply = "🗑 Đã hủy. Ảnh đã được xóa."
                else:
                    vision_prompt = _VISION_PROMPTS.get(user_text) or user_text
                    async with aiosqlite.connect(DB_PATH) as db:
                        async with db.execute(
                            "SELECT image_b64 FROM image_cache WHERE id=? AND user_id=?",
                            (img_pending, user_id)
                        ) as cur:
                            row = await cur.fetchone()
                    if not row:
                        await clear_pending_image(user_id)
                        reply = "⚠️ Không tìm thấy ảnh trong bộ nhớ. Vui lòng gửi lại."
                    else:
                        img_b64 = row[0]
                        answer  = await call_mistral_vision(img_b64, user_prompt=vision_prompt)
                        await clear_pending_image(user_id)
                        await save_message(user_id, "user",      f"[Ảnh] {vision_prompt}")
                        await save_message(user_id, "assistant", answer)
                        reply = f"👁️ Vision:\n{answer}"

            # ── TEXT PIPELINE BÌNH THƯỜNG (không vướng Vision) ──────────
            else:
                pending   = _pending_choice.get(user_id)
                cmd_reply = None

                # Xử lý Quick Reply số (audio / mail) theo state
                if user_text.isdigit() and len(user_text) <= 2 and pending:
                    if pending == "mail":
                        cmd_reply = await handle_command(user_id, f"/mail {user_text}")
                    elif pending.startswith("audio:"):
                        audio_id  = pending.split(":", 1)[1]
                        cmd_reply = await handle_command(
                            user_id, f"/audio {audio_id} {user_text}"
                        )
                    _pending_choice.pop(user_id, None)

                # Xử lý lệnh bình thường
                else:
                    cmd_reply = await handle_command(user_id, user_text)

                    cmd_check = user_text.strip().lower()
                    if cmd_check in ["/mail", "/ls mail", "mail"] and cmd_reply and "1" in cmd_reply:
                        _pending_choice[user_id] = "mail"
                        qr_items = [
                            QuickReplyItem(action=MessageAction(
                                label=f"{i}️⃣ Đọc mail {i}", text=str(i)
                            )) for i in range(1, 6)
                        ]
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

                    logger.info(f"[LLM] calling model_key={model_key} model_id={model_id}")
                    answer = await call_mistral_text(history, model_id, model_key=model_key)
                    logger.info(f"[LLM] answer len={len(answer)} preview={answer[:80]!r}")
                    await save_message(user_id, "user",      user_text)
                    await save_message(user_id, "assistant", answer)
                    reply = answer

        # ── TRẢ LỜI NGƯỜI DÙNG ─────────────────────────────────────────────
        if reply:
            try:
                reply  = strip_markdown(reply)
                chunks = [reply[i:i+5000] for i in range(0, len(reply), 5000)]
                messages = [TextMessage(text=c) for c in chunks]

                # Auto-export to Google Drive if response is very long (>10k chars)
                if len(reply) > 10000:
                    try:
                        from commands.data import export_to_drive
                        import datetime
                        filename = f"response_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        drive_link = await export_to_drive(user_id, reply, filename)
                        # Add drive link as last message
                        messages.append(TextMessage(text=f"📤 Full response exported to Google Drive:\n🔗 {drive_link}"))
                    except Exception:
                        # If export fails, just continue without it
                        pass

                if qr_items and messages:
                    messages[-1] = TextMessage(
                        text=chunks[-1], quick_reply=QuickReply(items=qr_items)
                    )

                await line_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=messages
                    )
                )
            except Exception as e:
                logger.error(f"Reply error: {e}")



async def _run_event(event):
    try:
        await _process_event_inner(event)
    except Exception as e:
        logger.error(f"[CRASH] {type(e).__name__}: {e}", exc_info=True)

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
            background_tasks.add_task(_run_event, event)

    return "OK"
