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

from database import (
    init_db, DB_PATH, save_message, get_pending_image, clear_pending_image,
    get_pending_choice, set_pending_choice, clear_pending_choice,
    is_rag_disabled,
)
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
                    ids = [r[0] for r in expired]
                    placeholders = ",".join("?" * len(ids))
                    await db.execute(
                        f"DELETE FROM image_cache WHERE id IN ({placeholders})", ids
                    )
                    await db.commit()
                    logger.info(f"image_cache_cleanup: removed {len(ids)} expired entries")
        except Exception as e:
            logger.error(f"image_cache_cleanup_loop error: {e}")


_REQUIRED_ENV = [
    "LINE_CHANNEL_ACCESS_TOKEN",
    "LINE_CHANNEL_SECRET",
]

@app.on_event("startup")
async def startup():
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"FATAL: Missing required environment variables: {', '.join(missing)}. "
            "Bot will not accept webhooks without these."
        )
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
# _rag_disabled and _pending_choice are stored in DB (user_state table)
# so they work correctly under multi-worker deployment.
# _pending_image: uses DB (image_cache) to survive restarts
_VISION_PROMPTS = {
    "1": (
        "Hãy mô tả chi tiết nội dung, màu sắc, bố cục và tất cả các yếu tố nhìn thấy được trong ảnh này. "
        "Trả lời bằng tiếng Việt."
    ),
    "2": (
        "Nhiệm vụ OCR: Hãy trích xuất và xuất ra toàn bộ văn bản trong ảnh, nguyên văn không thay đổi. "
        "Giữ nguyên ngôn ngữ gốc, dấu câu và cấu trúc. "
        "Nếu có nhiều ngôn ngữ, xuất tất cả theo đúng nguyên bản."
    ),
    "3": (
        "Hãy dịch toàn bộ văn bản trong ảnh sang tiếng Việt. "
        "Định dạng: [Văn bản gốc] → [Bản dịch tiếng Việt]"
    ),
    "4": (
        "Hãy phân tích sâu nội dung, dữ liệu hoặc thông tin trong ảnh này. "
        "Nếu có biểu đồ, hãy diễn giải xu hướng và số liệu; nếu là tài liệu, hãy tóm tắt điểm chính; "
        "nếu là ảnh thông thường, hãy phân tích chủ đề và ý nghĩa. "
        "Trả lời bằng tiếng Việt."
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
                ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=60)
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
                        if not await is_rag_disabled(user_id) and await has_rag_docs(user_id):
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

                    await set_pending_choice(user_id, f"audio:{audio_id}")
                    reply = f"🎤 Đã bóc băng xong:\n\n{transcript}\n\nBạn muốn làm gì tiếp theo?"
                    qr_items = [
                        QuickReplyItem(action=MessageAction(label="1️⃣ Phân tích/Tóm tắt", text="1")),
                        QuickReplyItem(action=MessageAction(label="2️⃣ Lưu vào tài liệu",  text="2")),
                        QuickReplyItem(action=MessageAction(label="3️⃣ Cả hai",             text="3")),
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
                elif user_text not in _VISION_PROMPTS:
                    reply = "⚠️ Vui lòng chọn 1 trong 4 tùy chọn bên dưới, hoặc gõ /hủy để huỷ."
                    qr_items = [
                        QuickReplyItem(action=MessageAction(label="1️⃣ Mô tả",    text="1")),
                        QuickReplyItem(action=MessageAction(label="2️⃣ OCR",       text="2")),
                        QuickReplyItem(action=MessageAction(label="3️⃣ Dịch",      text="3")),
                        QuickReplyItem(action=MessageAction(label="4️⃣ Phân tích", text="4")),
                    ]
                else:
                    vision_prompt = _VISION_PROMPTS[user_text]
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
                # Check if this is a new user and send welcome message
                try:
                    from database import is_new_user, get_welcome_message, mark_user_onboarded
                    if await is_new_user(user_id):
                        welcome_msg = await get_welcome_message(user_id)
                        await line_api.push_message(
                            PushMessageRequest(
                                to=user_id,
                                messages=[TextMessage(text=welcome_msg)]
                            )
                        )
                        await mark_user_onboarded(user_id)
                except Exception as e:
                    logger.info(f"Welcome message check failed (not critical): {e}")

                pending   = await get_pending_choice(user_id)
                cmd_reply = None

                cmd_check = user_text.strip().lower()

                # Xử lý Quick Reply số (audio / mail) theo state
                if user_text.isdigit() and len(user_text) <= 2 and pending and pending != "clear_confirm":
                    if pending == "mail":
                        cmd_reply = await handle_command(user_id, f"/mail {user_text}")
                    elif pending.startswith("audio:"):
                        audio_id  = pending.split(":", 1)[1]
                        cmd_reply = await handle_command(
                            user_id, f"/audio {audio_id} {user_text}"
                        )
                    await clear_pending_choice(user_id)

                # /clear — yêu cầu xác nhận 2 bước
                elif cmd_check == "/clear":
                    if pending == "clear_confirm":
                        await clear_pending_choice(user_id)
                        cmd_reply = await handle_command(user_id, user_text)
                    else:
                        await set_pending_choice(user_id, "clear_confirm")
                        cmd_reply = (
                            "⚠️ Bạn có chắc muốn xoá toàn bộ lịch sử hội thoại không?\n"
                            "Gõ /clear lần nữa để xác nhận, hoặc nhắn bất kỳ nội dung khác để huỷ."
                        )

                # Xử lý lệnh bình thường
                else:
                    cmd_reply = await handle_command(user_id, user_text)

                    if cmd_check in ["/mail", "/ls mail", "mail"] and cmd_reply and "1" in cmd_reply:
                        await set_pending_choice(user_id, "mail")
                        qr_items = [
                            QuickReplyItem(action=MessageAction(
                                label=f"{i}️⃣ Đọc mail {i}", text=str(i)
                            )) for i in range(1, 6)
                        ]
                    elif cmd_check.startswith("/audio "):
                        pass
                    elif cmd_reply:
                        await clear_pending_choice(user_id)

                if cmd_reply is not None:
                    reply = cmd_reply
                else:
                    # User moved on — clear any stale confirmation state
                    if pending == "clear_confirm":
                        await clear_pending_choice(user_id)
                    model_key, model_id = await resolve_model(user_id, user_text)
                    history = await get_history_with_summary(user_id)

                    rag_chunks = []
                    if not await is_rag_disabled(user_id) and await has_rag_docs(user_id):
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

                # Auto-export to Google Drive if response is long (>5k chars)
                if len(reply) > 5000:
                    async with aiosqlite.connect(DB_PATH) as _db:
                        async with _db.execute(
                            "SELECT 1 FROM google_auth WHERE user_id = ?", (user_id,)
                        ) as _cur:
                            _has_drive = await _cur.fetchone() is not None
                    if _has_drive:
                        try:
                            from commands.data import export_to_drive
                            first_line = reply.split('\n')[0][:30].replace(' ', '_') if reply else "response"
                            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                            filename = f"{first_line}_{timestamp}.txt"
                            drive_link = await export_to_drive(user_id, reply, filename)
                            messages.append(TextMessage(
                                text=f"📤 Câu trả lời quá dài, đã lưu toàn bộ vào Google Drive:\n🔗 {drive_link}"
                            ))
                        except Exception as e:
                            logger.info(f"Auto-export failed: {e}")
                    else:
                        messages.append(TextMessage(
                            text="📝 Câu trả lời bị cắt bớt do quá dài (>5000 ký tự).\nDùng /login để kết nối Google Drive và nhận câu trả lời đầy đủ."
                        ))

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
