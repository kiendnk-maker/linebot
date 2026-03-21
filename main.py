"""
Groq哥哥 LINE Bot — Trạm Điều Phối (Main Router)
Spec version: 2026-03-16
"""

import os

_pending_choice: dict[str, str] = {}

# --- NUKE OLD DB (VOLUME SAFE) ---
import os, shutil
for db_dir in ['chroma_db', 'rag_db', 'vector_db']:
    if os.path.exists(db_dir):
        flag_file = os.path.join(db_dir, '.v2_wiped')
        # Nếu chưa có cờ đánh dấu -> Tiến hành dọn rác
        if not os.path.exists(flag_file):
            for filename in os.listdir(db_dir):
                filepath = os.path.join(db_dir, filename)
                try:
                    if os.path.isfile(filepath) or os.path.islink(filepath):
                        os.unlink(filepath)
                    elif os.path.isdir(filepath):
                        shutil.rmtree(filepath)
                except Exception:
                    pass
            # Cắm cờ để các lần khởi động sau không bị xóa nhầm
            try:
                with open(flag_file, 'w') as f: f.write('wiped')
                print(f"☢️ Đã dọn dẹp thành công nội dung Volume: {db_dir}")
            except Exception:
                pass
# ---------------------------------




import base64
import asyncio
import time
import re
import logging
import aiosqlite
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from linebot.v3 import WebhookParser
from linebot.v3.messaging import (
    Configuration, AsyncApiClient, AsyncMessagingApi, AsyncMessagingApiBlob,
    ReplyMessageRequest, PushMessageRequest, TextMessage, ShowLoadingAnimationRequest,
    QuickReply, QuickReplyItem, MessageAction, FlexMessage, FlexContainer
)
from linebot.v3.webhooks import (
    MessageEvent, TextMessageContent, ImageMessageContent,
    AudioMessageContent, FileMessageContent
)

# --- MODULES TỰ VIẾT ---
from database import DB_PATH, init_db, save_message
from llm_core import (
    MODEL_REGISTRY, DEFAULT_MODEL_KEY, resolve_model, get_history_with_summary, maybe_summarize,
    call_mistral_text, call_mistral_vision, call_groq_whisper, clean_transcript,
    strip_markdown, _split_reply
)
from reminder_system import parse_reminder_nlp, reminder_loop
from rag_core import MAX_FILE_BYTES, SUPPORTED_RAG_EXTS, process_file_upload, has_rag_docs, rag_search
from google_workspace import router as gw_router
from command_handler import handle_command

# --- CẤU HÌNH HỆ THỐNG ---
TZ = ZoneInfo("Asia/Taipei")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LINE_CHANNEL_SECRET       = os.environ["LINE_CHANNEL_SECRET"]
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]

MAX_INPUT_CHARS = 8000
_REPLY_TRIGGERS = ("hello", "不好意思")
_rag_disabled: set[str] = set()

line_config    = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
webhook_parser = WebhookParser(LINE_CHANNEL_SECRET)

# --- KHỞI TẠO FASTAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Auto-nuke old ChromaDB collections with wrong dimensions
    try:
        import chromadb
        chroma_path = os.environ.get("CHROMA_PATH", "chroma")
        _cc = chromadb.PersistentClient(path=chroma_path)
        for col_name in [c.name for c in _cc.list_collections()]:
            try:
                col = _cc.get_collection(col_name)
                peek = col.peek(1)
                if peek and peek.get("embeddings") and len(peek["embeddings"][0]) == 3072:
                    _cc.delete_collection(col_name)
                    logger.info(f"Deleted old 3072-dim collection: {col_name}")
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"ChromaDB cleanup: {e}")
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute("ALTER TABLE user_settings ADD COLUMN language TEXT DEFAULT 'vi'")
            await db.commit()
        except Exception:
            pass
    asyncio.create_task(reminder_loop())
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(gw_router)

# --- WEBHOOK ENDPOINT ---
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    try:
        events = webhook_parser.parse(body.decode("utf-8"), signature)
    except Exception:
        raise HTTPException(status_code=400)
    for event in events:
        if isinstance(event, MessageEvent):
            background_tasks.add_task(process_event, event)
    return JSONResponse({"status": "ok"})

# --- LUỒNG XỬ LÝ SỰ KIỆN ---
async def process_event(event: MessageEvent) -> None:
    try:
        await _process_event_inner(event)
    except Exception as e:
        logger.exception(f"CRASH in process_event | user={event.source.user_id}")
        try:
            async with AsyncApiClient(line_config) as api_client:
                line_api = AsyncMessagingApi(api_client)
                await line_api.push_message(PushMessageRequest(
                    to=event.source.user_id,
                    messages=[TextMessage(text=f"⚠️ Lỗi hệ thống: {str(e)[:150]}")]
                ))
        except Exception:
            pass


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

                        await save_message(user_id, "user", clean_text)
                        answer = await call_mistral_text(
                            history, model_id, model_key=model_key, user_id=user_id, rag_chunks=rag_chunks or None
                        )
                        await save_message(user_id, "assistant", answer)
                        await maybe_summarize(user_id)
                        reply = f"🎤 {clean_text}\n\n{answer}\n\n[{MODEL_REGISTRY[model_key]['display']}]"
                else:
                    reminder_reply = await parse_reminder_nlp(user_id, transcript)
                    if reminder_reply:
                        reply = f"🎤 {transcript}\n\n{reminder_reply}"
                    else:
                        await save_message(user_id, "user", f"[Voice]: {transcript}")
                        reply = f"🎤 {transcript}"

        # ── IMAGE PIPELINE ──
        elif isinstance(event.message, ImageMessageContent):
            img_bytes = await line_blob_api.get_message_content(event.message.id)
            img_b64   = base64.b64encode(img_bytes).decode("utf-8")
            answer    = await call_mistral_vision(img_b64)
            if "⚠️" in answer:
                reply = answer
            else:
                await save_message(user_id, "user", f"[Ảnh] {answer}")
                await save_message(user_id, "assistant", answer)
                await maybe_summarize(user_id)
                reply = answer

        # ── FILE PIPELINE ──
        elif isinstance(event.message, FileMessageContent):
            filename = event.message.file_name or ""
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            AUDIO_EXTS = {"mp3", "m4a", "wav", "ogg", "flac"}

            if f".{ext}" not in SUPPORTED_RAG_EXTS and ext not in AUDIO_EXTS:
                reply = f"⚠️ Chỉ hỗ trợ văn bản ({', '.join(sorted(SUPPORTED_RAG_EXTS))}) và âm thanh.\nNhận được: {filename}"
            else:
                file_bytes = await line_blob_api.get_message_content(event.message.id)
                if len(file_bytes) > MAX_FILE_BYTES:
                    reply = f"⚠️ File quá lớn ({len(file_bytes) // 1024 // 1024}MB)."
                else:
                    if ext in AUDIO_EXTS:
                        try:
                            await line_api.push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text="⏳ Đang bóc băng âm thanh...")]))
                        except Exception: pass
                        
                        transcript = await call_groq_whisper(file_bytes)
                        if "⚠️" in transcript:
                            reply = transcript
                        else:
                            transcript = await clean_transcript(transcript)
                            prompt_title = (
                                "Tạo tên file slug cho nội dung sau. "
                                "QUY TẮC BẮT BUỘC: Chỉ trả về ĐÚNG 3-5 từ tiếng Việt KHÔNG DẤU, "
                                "nối bằng gạch ngang. KHÔNG viết gì khác. "
                                "VD: phan-tich-du-lieu, hop-team-marketing, bao-cao-tai-chinh\n"
                                f"{transcript[:500]}"
                            )
                            title_raw = await call_mistral_text([{"role": "user", "content": prompt_title}], MODEL_REGISTRY["small"]["model_id"], model_key="small", user_id=user_id)
                            # Aggressive cleanup: only keep [a-z0-9-]
                            safe_title = re.sub(r'[^a-zA-Z0-9\-]', '', title_raw.strip().split('\n')[0].lower())
                            safe_title = re.sub(r'-+', '-', safe_title).strip('-')
                            # Validate: must be slug-like (has hyphens, reasonable length)
                            if not safe_title or len(safe_title) < 3 or '-' not in safe_title:
                                # Fallback: extract first few words from transcript
                                import unicodedata
                                fallback = transcript[:80].lower()
                                fallback = unicodedata.normalize('NFD', fallback)
                                fallback = ''.join(c for c in fallback if unicodedata.category(c) != 'Mn')
                                fallback = re.sub(r'[^a-z0-9]+', '-', fallback).strip('-')
                                words = fallback.split('-')[:4]
                                safe_title = '-'.join(w for w in words if w) or 'audio-transcript'
                            new_filename = f'{safe_title[:40]}.txt'
                            
                            async with aiosqlite.connect(DB_PATH) as db:
                                await db.execute("CREATE TABLE IF NOT EXISTS audio_cache (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, transcript TEXT, filename TEXT, created_at INTEGER)")
                                cur = await db.execute("INSERT INTO audio_cache (user_id, transcript, filename, created_at) VALUES (?, ?, ?, ?)", (user_id, transcript, new_filename, int(time.time())))
                                audio_id = cur.lastrowid
                                await db.commit()
                                
                            reply = {
                                "text": f"🎤 Đã bóc băng xong! Tên file: {new_filename}\nBạn muốn xử lý như thế nào?",
                                "quickReply": {
                                    "items": [
                                        {"type": "action", "action": {"type": "message", "label": "1. Tóm tắt + TXT", "text": f"/audio {audio_id} 1"}},
                                        {"type": "action", "action": {"type": "message", "label": "2. Lưu RAG + TXT", "text": f"/audio {audio_id} 2"}},
                                        {"type": "action", "action": {"type": "message", "label": "3. Cả hai + TXT", "text": f"/audio {audio_id} 3"}}
                                    ]
                                }
                            }
                    else:
                        if not ext and file_bytes[:512].decode("utf-8", errors="ignore").isprintable():
                            filename = filename or "upload.txt"
                            ext = "txt"
                        reply = await process_file_upload(user_id, file_bytes, filename)

        # ── TEXT PIPELINE ──
        elif isinstance(event.message, TextMessageContent):
            user_text = event.message.text.strip()

            pending = _pending_choice.get(user_id)
                if user_text.isdigit() and len(user_text) <= 2 and pending:
                    if pending == "mail":
                        cmd_reply = await handle_command(user_id, f"/mail {user_text}")
                    elif pending.startswith("audio:"):
                        audio_id = pending.split(":", 1)[1]
                        cmd_reply = await handle_command(user_id, f"/audio {audio_id} {user_text}")
                    _pending_choice.pop(user_id, None)  # Dùng xong xoá luôn state
                else:
                    cmd_reply = await handle_command(user_id, user_text)
                    
                    # Bắt state tự động: Nếu user vừa gọi lệnh check mail, nạp đạn chờ số
                    cmd_check = user_text.strip().lower()
                    if cmd_check in ["/mail", "/ls mail", "mail"] and cmd_reply and "1" in cmd_reply:
                        _pending_choice[user_id] = "mail"
                    elif cmd_check.startswith("/audio "):
                        pass # Bỏ qua để không xoá state của audio
                    elif cmd_reply:
                        _pending_choice.pop(user_id, None) # Các lệnh khác thì clear state
                
            if cmd_reply is not None:
                reply = cmd_reply
            elif len(user_text) > MAX_INPUT_CHARS:
                reply = f"⚠️ Tin nhắn quá dài ({len(user_text)} ký tự).\nVui lòng giới hạn dưới {MAX_INPUT_CHARS} ký tự."
            else:
                reminder_reply = await parse_reminder_nlp(user_id, user_text)
                if reminder_reply:
                    reply = reminder_reply
                else:
                    model_key, model_id = await resolve_model(user_id, user_text)
                    is_summarize = len(user_text) > 500 and "?" not in user_text and "？" not in user_text
                    
                    await save_message(user_id, "user", user_text)
                    history = [{"role": "user", "content": f"Hãy tóm tắt nội dung sau:\n{user_text}"}] if is_summarize else await get_history_with_summary(user_id)

                    rag_chunks = []
                    if user_id not in _rag_disabled and await has_rag_docs(user_id):
                        rag_chunks = await rag_search(user_id, user_text)

                    answer = await call_mistral_text(
                        history, model_id, model_key=model_key, user_id=user_id, rag_chunks=rag_chunks or None
                    )
                    await save_message(user_id, "assistant", answer)
                    await maybe_summarize(user_id)
                    reply = answer

        # ── TRẢ LỜI NGƯỜI DÙNG ──
        if reply:
            try:
                if isinstance(reply, dict):
                    qr_data = reply.get("quickReply")
                    qr_items = []
                    if qr_data:
                        for item in qr_data.get("items", []):
                            act = item.get("action", {})
                            qr_items.append(QuickReplyItem(action=MessageAction(label=act.get("label", ""), text=act.get("text", ""))))
                    quick_reply_obj = QuickReply(items=qr_items) if qr_items else None

                    if reply.get("type") == "flex":
                        flex_msg = FlexMessage(
                            alt_text=reply.get("altText", "Flex Message"),
                            contents=FlexContainer.from_dict(reply.get("contents"))
                        )
                        if quick_reply_obj:
                            flex_msg.quick_reply = quick_reply_obj
                        messages = [flex_msg]
                    else:
                        text_content = strip_markdown(reply.get("text", ""))
                        chunks = _split_reply(text_content)
                        messages = [TextMessage(text=c) for c in chunks]
                        if quick_reply_obj and messages:
                            messages[-1].quick_reply = quick_reply_obj

                    await line_api.reply_message(ReplyMessageRequest(reply_token=event.reply_token, messages=messages))
                else:
                    reply_str = strip_markdown(reply)
                    chunks = _split_reply(reply_str)
                    await line_api.reply_message(ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=c) for c in chunks]))
            except Exception:
                try:
                    fb = reply if isinstance(reply, str) else reply.get("text", str(reply))
                    fb = strip_markdown(fb)[:4990]
                    await line_api.push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text=fb)]))
                except Exception as pe:
                    logger.error(f"Push fallback failed | user={user_id} | {pe}")

# Force update to clear Mistral 422 cache

# Fixed Enum string 'none' for Mistral API validation
