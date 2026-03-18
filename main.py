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
    QuickReply, QuickReplyItem, MessageAction,
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
client = AsyncGroq(api_key=GROQ_API_KEY)  # Khởi tạo client dùng chung cho các Agent
CHROMA_PATH               = os.environ.get("CHROMA_PATH", "chroma")



MAX_INPUT_CHARS = 8000

# Trigger prefixes that activate LLM reply from voice messages
_REPLY_TRIGGERS = ("hello", "不好意思")

# ---------------------------------------------------------------------------
# MODEL REGISTRY
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# AUTO-ROUTER
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# RAG — per-user ChromaDB lock registry
# ---------------------------------------------------------------------------
_chroma_locks: dict[str, asyncio.Lock] = {}
_chroma_locks_meta: asyncio.Lock | None = None   # initialized lazily inside event loop









# ---------------------------------------------------------------------------
# RAG — in-memory RAG-off toggle (per session, resets on restart)
# ---------------------------------------------------------------------------
_rag_disabled: set[str] = set()   # user_ids with RAG disabled


# ---------------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DATABASE — user settings helpers
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# DATABASE — user profile helpers
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# DATABASE — history helpers
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# SUMMARY MEMORY
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# REMINDER SYSTEM
# ---------------------------------------------------------------------------








# LLM returns HH:MM + DD/MM/YYYY → Python calculates fire_at to avoid TZ errors






# ---------------------------------------------------------------------------
# MARKDOWN STRIPPER
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# RAG — embedding
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# RAG — chunking
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RAG — ChromaDB upsert (thread-safe per user)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RAG — SQLite metadata helpers
# ---------------------------------------------------------------------------












# ---------------------------------------------------------------------------
# RAG — search
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RAG — PDF ingest pipeline
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GROQ CALLERS
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# COMMAND SYSTEM
# ---------------------------------------------------------------------------






import json

# 1. Định nghĩa các công cụ thực tế (Python Functions)


# Dictionary mapping tên công cụ với hàm Python

# 2. Định nghĩa Schema của Tools cho LLM hiểu

# 3. Vòng lặp Agentic (Core Logic)




# ---------------------------------------------------------------------------
# REPLY HELPER
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute("ALTER TABLE user_settings ADD COLUMN language TEXT DEFAULT 'vi'")
            await db.commit()
        except Exception:
            pass
    asyncio.create_task(reminder_loop())
    yield


app            = FastAPI(lifespan=lifespan)
line_config    = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
webhook_parser = WebhookParser(LINE_CHANNEL_SECRET)



from fastapi.responses import HTMLResponse

# --- MODULES TỰ VIẾT ---
from llm_core import WHISPER_MODEL, SUMMARY_TRIGGER, MODEL_REGISTRY, DEFAULT_MODEL_KEY, VISION_MODEL_KEY, CLASSIFIER_MODEL_KEY, ROUTE_MAP, _REALTIME_KEYWORDS, _CLASSIFIER_PROMPT, _needs_realtime, classify_query, resolve_model, build_system_prompt, maybe_summarize, get_history_with_summary, strip_markdown, call_groq_text, call_groq_vision, call_groq_whisper, clean_transcript, _split_reply
from reminder_system import _next_fire, _PARSE_REMINDER_PROMPT, parse_reminder_nlp, reminder_loop
from database import DB_PATH, init_db, save_message, save_reminder
from google_workspace import router as gw_router, handle_workspace_command
app.include_router(gw_router)
from rag_core import MAX_FILE_BYTES, SUPPORTED_RAG_EXTS, process_file_upload, has_rag_docs, rag_search, list_rag_docs, delete_rag_doc, clear_rag_docs
from database import get_user_model, set_user_model, get_user_max_tokens, set_user_max_tokens, get_user_profile, save_user_profile, get_history_raw, count_history, get_summary, save_summary, get_reminders, cancel_reminder
from tools_api import AVAILABLE_TOOLS, AGENT_TOOLS
from agents_workflow import run_multi_agent_workflow, run_pro_workflow, run_agentic_loop




from command_handler import handle_command
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

        # ── FILE PIPELINE ─────────────────────────────────────────────────
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
                            prompt_title = f"Tóm tắt đoạn văn sau thành tối đa 5 chữ để làm tên file, phân cách bằng dấu gạch ngang:\n{transcript[:1000]}"
                            title = await call_groq_text([{"role": "user", "content": prompt_title}], MODEL_REGISTRY["llama8b"]["model_id"], model_key="llama8b", user_id=user_id)
                            
                            import re, time
                            safe_title = re.sub(r'[^a-zA-Z0-9À-ɏḀ-ỿ]', '-', title).strip('-')
                            safe_title = re.sub(r'-+', '-', safe_title) or "audio"
                            new_filename = f"{safe_title[:30]}.txt"
                            
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

        # ── TEXT PIPELINE ──────────────────────────────────────────────────
        elif isinstance(event.message, TextMessageContent):
            user_text = event.message.text.strip()

            # UX Trick: Nếu người dùng chỉ gõ số (vd: "2"), tự động biên dịch thành lệnh "/mail 2"
            if user_text.isdigit() and len(user_text) <= 2:
                cmd_reply = await handle_command(user_id, f"/mail {user_text}")
            else:
                cmd_reply = await handle_command(user_id, user_text)
                
            # Layer 3a: command handler
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
            if isinstance(reply, dict):
                qr_data = reply.get("quickReply")
                qr_items = []
                if qr_data:
                    for item in qr_data.get("items", []):
                        act = item.get("action", {})
                        qr_items.append(
                            QuickReplyItem(
                                action=MessageAction(
                                    label=act.get("label", ""),
                                    text=act.get("text", ""),
                                )
                            )
                        )
                quick_reply_obj = QuickReply(items=qr_items) if qr_items else None

                if reply.get("type") == "flex":
                    from linebot.v3.messaging import FlexMessage, FlexContainer
                    flex_msg = FlexMessage(
                        alt_text=reply.get("altText", "Hộp thư Flex Message"),
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
                
                await line_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=messages,
                    )
                )
            else:
                reply = strip_markdown(reply)
                chunks = _split_reply(reply)
                await line_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=c) for c in chunks],
                    )
                )
# Last Fix: Mon Mar 16 03:58:20 CST 2026
# Fix scope at Mon Mar 16 04:03:44 CST 2026
# Fix OAuth UX: Mon Mar 16 04:08:53 CST 2026
# Clean UI Fix: Mon Mar 16 04:19:18 CST 2026
# Clean UI Fix: Mon Mar 16 04:24:07 CST 2026
# Fix DB Connection: Mon Mar 16 04:29:38 CST 2026

# UI Update forced at: 2026-03-16
# Manual Trigger: Mon Mar 16 04:37:57 CST 2026
# Quick Reply Update: Mon Mar 16 04:41:56 CST 2026
# Fix Syntax & Finish QuickReply: Mon Mar 16 04:47:06 CST 2026
# Fix Syntax & Enable QR: Mon Mar 16 04:52:01 CST 2026
# Fix Syntax & Enable QR: Mon Mar 16 05:15:55 CST 2026
# Syntax fix for line 1380 - Mon Mar 16 05:21:46 CST 2026
# Update Limit to 200: Mon Mar 16 08:50:00 CST 2026
# Add Google Calendar Module: Mon Mar 16 13:05:59 CST 2026
# Fix OAuth Scope %20 - Mon Mar 16 13:09:16 CST 2026
