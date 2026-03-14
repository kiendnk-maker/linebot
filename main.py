import os, base64, aiosqlite, httpx
from contextlib import asynccontextmanager
from groq import AsyncGroq
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from linebot.v3 import WebhookParser
from linebot.v3.messaging import (
    AsyncApiClient, AsyncMessagingApi, AsyncMessagingApiBlob, Configuration,
    ReplyMessageRequest, TextMessage, ShowLoadingAnimationRequest,
)
from linebot.v3.webhooks import (
    MessageEvent, TextMessageContent, ImageMessageContent, AudioMessageContent,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
try:
    from prompts import SYSTEM_PROMPT
except ImportError:
    SYSTEM_PROMPT = (
        "Bạn là trợ lý thông minh. Nếu dùng tiếng Trung hãy dùng Phồn thể.\n"
        "QUAN TRỌNG: Đây là chat LINE, KHÔNG dùng Markdown. "
        "Không dùng **bold**, *italic*, # heading, ```code block```, bullet - hay bất kỳ ký tự định dạng nào. "
        "Chỉ dùng văn xuôi thuần, xuống dòng bình thường."
    )

# Append no-markdown rule vào bất kỳ SYSTEM_PROMPT nào (kể cả từ prompts.py)
_NO_MD_SUFFIX = (
    "\nQUAN TRỌNG: Đây là chat LINE, KHÔNG dùng Markdown. "
    "Không dùng **bold**, *italic*, # heading, ```code block```, - bullet, _ underscore. "
    "Chỉ trả lời bằng văn xuôi thuần túy."
)
if _NO_MD_SUFFIX.strip() not in SYSTEM_PROMPT:
    SYSTEM_PROMPT = SYSTEM_PROMPT + _NO_MD_SUFFIX

LINE_CHANNEL_SECRET       = os.environ["LINE_CHANNEL_SECRET"]
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
GROQ_API_KEY              = os.environ["GROQ_API_KEY"]
DB_PATH                   = "chat_history.db"

WHISPER_MODEL = "whisper-large-v3-turbo"   # production, cheapest whisper

# ---------------------------------------------------------------------------
# MODEL REGISTRY
# Source: https://console.groq.com/docs/models  (checked 2026-03-14)
#
# tier:  "production" | "preview"
# type:  "text" | "vision" | "reasoning"
#        reasoning  = chain-of-thought; may emit <think> tags
#        vision     = accepts image inputs
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict] = {
    # ── Production ──────────────────────────────────────────────────────────
    "llama70b": {
        "model_id": "llama-3.3-70b-versatile",
        "type":     "text",
        "tier":     "production",
        "display":  "LLaMA 3.3 70B Versatile",
        "ctx":      131_072,
        "note":     "Best all-round, 280 t/s",
    },
    "llama8b": {
        "model_id": "llama-3.1-8b-instant",
        "type":     "text",
        "tier":     "production",
        "display":  "LLaMA 3.1 8B Instant",
        "ctx":      131_072,
        "note":     "Fastest, 560 t/s, rất rẻ",
    },
    "gpt120b": {
        "model_id": "openai/gpt-oss-120b",
        "type":     "reasoning",
        "tier":     "production",
        "display":  "GPT-OSS 120B (OpenAI open-weight)",
        "ctx":      131_072,
        "note":     "Reasoning + browser search built-in, 500 t/s",
    },
    "gpt20b": {
        "model_id": "openai/gpt-oss-20b",
        "type":     "reasoning",
        "tier":     "production",
        "display":  "GPT-OSS 20B (OpenAI open-weight)",
        "ctx":      131_072,
        "note":     "Siêu nhanh 1000 t/s, reasoning nhẹ",
    },
    # ── Preview ─────────────────────────────────────────────────────────────
    "scout": {
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "type":     "vision",
        "tier":     "preview",
        "display":  "LLaMA 4 Scout 17B (Vision)",
        "ctx":      131_072,
        "note":     "Hỗ trợ ảnh, 750 t/s",
    },
    "qwen": {
        "model_id": "qwen/qwen3-32b",
        "type":     "reasoning",
        "tier":     "preview",
        "display":  "Qwen3 32B",
        "ctx":      131_072,
        "note":     "Reasoning tốt, 400 t/s",
    },
    "kimi": {
        "model_id": "moonshotai/kimi-k2-instruct-0905",
        "type":     "text",
        "tier":     "preview",
        "display":  "Kimi K2 0905 (Moonshot AI)",
        "ctx":      262_144,   # 256K — longest context on Groq
        "note":     "Context dài nhất 256K, agentic coding",
    },
    # ── Compound systems (built-in web search + code exec) ──────────────────
    "compound": {
        "model_id": "groq/compound",
        "type":     "text",
        "tier":     "production",
        "display":  "Groq Compound (web search + code)",
        "ctx":      131_072,
        "note":     "Tự tìm web, chạy code, ~450 t/s",
    },
    "compound-mini": {
        "model_id": "groq/compound-mini",
        "type":     "text",
        "tier":     "production",
        "display":  "Groq Compound Mini",
        "ctx":      131_072,
        "note":     "Phiên bản nhỏ của Compound",
    },
}

DEFAULT_MODEL_KEY  = "llama70b"
FALLBACK_VISION_KEY = "scout"   # used when current model != vision

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
        await db.commit()


async def get_history(user_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT role, content FROM history "
            "WHERE user_id = ? ORDER BY id DESC LIMIT 10",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


async def save_message(user_id: str, role: str, content: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content),
        )
        await db.commit()


async def get_user_model(user_id: str) -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT model_key FROM user_settings WHERE user_id = ?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
    key = row[0] if row else DEFAULT_MODEL_KEY
    # guard against stale keys after registry changes
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
# ---------------------------------------------------------------------------
# MARKDOWN STRIPPER  (lớp 2 — phòng thủ sau prompt)
# ---------------------------------------------------------------------------
import re

def strip_markdown(text: str) -> str:
    """
    Remove common Markdown symbols that look ugly in LINE chat.
    Preserves newlines and normal punctuation.
    """
    # Fenced code blocks  ```...```  → keep content, remove fences
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"```", "", text)

    # Inline code  `code`  → just the word
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Headings  ## Title  →  Title
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Bold/italic  **text**, *text*, __text__, _text_  → text
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)

    # Strikethrough  ~~text~~  → text
    text = re.sub(r"~~([^~]+)~~", r"\1", text)

    # Bullet lists  "- item" / "* item"  →  "• item"  (readable trên LINE)
    text = re.sub(r"^[\-\*]\s+", "• ", text, flags=re.MULTILINE)

    # Numbered lists keep as-is (1. 2. 3. đọc được bình thường)

    # Horizontal rules  ---  ***  ___
    text = re.sub(r"^(\-{3,}|\*{3,}|_{3,})\s*$", "─────", text, flags=re.MULTILINE)

    # Links  [text](url)  →  text (url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)

    # Blockquotes  > text  →  text
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)

    # Collapse 3+ blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# GROQ CALLERS
# ---------------------------------------------------------------------------
async def call_groq_text(history: list[dict], model_id: str) -> str:
    async with httpx.AsyncClient() as http:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
                temperature=0.6,
                max_tokens=800,
            )
            return strip_markdown(resp.choices[0].message.content or "")
        except Exception as e:
            return f"⚠️ Lỗi text [{model_id}]: {str(e)[:150]}"


async def call_groq_vision(image_b64: str, model_id: str) -> str:
    async with httpx.AsyncClient() as http:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Phân tích và mô tả ảnh này. "
                                    "Nếu có chữ Trung hãy dùng Phồn thể."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    },
                ],
                max_tokens=800,
            )
            return strip_markdown(resp.choices[0].message.content or "")
        except Exception as e:
            return f"⚠️ Lỗi vision [{model_id}]: {str(e)[:150]}"


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
            return f"⚠️ Lỗi Whisper: {str(e)[:150]}"

# ---------------------------------------------------------------------------
# COMMAND SYSTEM
# ---------------------------------------------------------------------------
def _models_list_text() -> str:
    """Build /models help text."""
    prod_lines: list[str] = []
    prev_lines: list[str] = []

    for key, cfg in MODEL_REGISTRY.items():
        icon = {"vision": "👁", "reasoning": "🧠", "text": "💬"}.get(cfg["type"], "💬")
        line = f"{icon} /{key} — {cfg['display']}\n   {cfg['note']}"
        if cfg["tier"] == "production":
            prod_lines.append(line)
        else:
            prev_lines.append(line)

    parts = [
        "📋 DANH SÁCH MODEL GROQ\n",
        "── Production ──",
        *prod_lines,
        "\n── Preview ──",
        *prev_lines,
        "\n─────────────────",
        "Chuyển model:  /model <tên>",
        "Hỏi 1 lần:     /<tên> <câu hỏi>",
        "Model hiện tại: /model",
        "Xoá lịch sử:   /clear",
    ]
    return "\n".join(parts)


async def handle_command(user_id: str, text: str) -> str | None:
    """
    Parse slash commands.
    Returns reply string if text is a command, else None.
    """
    if not text.startswith("/"):
        return None

    parts   = text[1:].strip().split(maxsplit=1)
    cmd     = parts[0].lower()
    arg     = parts[1].strip() if len(parts) > 1 else ""

    # /clear
    if cmd == "clear":
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
            await db.commit()
        return "🗑 Đã xoá lịch sử hội thoại."

    # /models
    if cmd == "models":
        return _models_list_text()

    # /model [key]
    if cmd == "model":
        if not arg:
            key = await get_user_model(user_id)
            cfg = MODEL_REGISTRY[key]
            return (
                f"🤖 Model hiện tại: {cfg['display']}\n"
                f"   Tier: {cfg['tier']} | Type: {cfg['type']}\n"
                f"   {cfg['note']}\n\n"
                "Dùng /models để xem tất cả."
            )
        target = arg.lower()
        if target not in MODEL_REGISTRY:
            return f"❌ /{target} không tồn tại.\nDùng /models để xem danh sách."
        await set_user_model(user_id, target)
        return f"✅ Đã chuyển sang {MODEL_REGISTRY[target]['display']}."

    # /<model_key> [câu hỏi tức thì]
    if cmd in MODEL_REGISTRY:
        await set_user_model(user_id, cmd)
        cfg = MODEL_REGISTRY[cmd]
        if arg:
            # One-shot: không lưu vào history, trả lời ngay
            answer = await call_groq_text(
                [{"role": "user", "content": arg}], cfg["model_id"]
            )
            return f"[{cfg['display']}]\n{answer}"
        return f"✅ Chuyển sang {cfg['display']}.\nTin nhắn tiếp theo dùng model này."

    return f"❓ Lệnh /{cmd} không hợp lệ. Dùng /models để xem."

# ---------------------------------------------------------------------------
# FASTAPI
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
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

        # Show loading indicator (best-effort)
        try:
            await line_api.show_loading_animation(
                ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=5)
            )
        except Exception:
            pass

        reply = ""

        # ── AUDIO ──────────────────────────────────────────────────────────
        if isinstance(event.message, AudioMessageContent):
            audio_bytes = await line_blob_api.get_message_content(event.message.id)
            transcript  = await call_groq_whisper(audio_bytes)

            if "⚠️" not in transcript:
                await save_message(user_id, "user", f"[Voice]: {transcript}")
                model_key = await get_user_model(user_id)
                model_id  = MODEL_REGISTRY[model_key]["model_id"]
                history   = await get_history(user_id)
                answer    = await call_groq_text(history, model_id)
                await save_message(user_id, "assistant", answer)
                reply = f"🎤 Bạn nói: {transcript}\n\n🤖 {answer}"
            else:
                reply = transcript

        # ── IMAGE ──────────────────────────────────────────────────────────
        elif isinstance(event.message, ImageMessageContent):
            img_bytes = await line_blob_api.get_message_content(event.message.id)
            img_b64   = base64.b64encode(img_bytes).decode("utf-8")

            model_key = await get_user_model(user_id)
            cfg       = MODEL_REGISTRY[model_key]

            if cfg["type"] == "vision":
                vision_id = cfg["model_id"]
                note      = ""
            else:
                vision_id = MODEL_REGISTRY[FALLBACK_VISION_KEY]["model_id"]
                note      = (
                    f"⚠️ /{model_key} không hỗ trợ ảnh "
                    f"→ tạm dùng /{FALLBACK_VISION_KEY}\n\n"
                )

            answer = await call_groq_vision(img_b64, vision_id)
            reply  = note + answer

        # ── TEXT ───────────────────────────────────────────────────────────
        elif isinstance(event.message, TextMessageContent):
            user_text = event.message.text.strip()

            cmd_reply = await handle_command(user_id, user_text)
            if cmd_reply is not None:
                reply = cmd_reply
            else:
                await save_message(user_id, "user", user_text)
                model_key = await get_user_model(user_id)
                model_id  = MODEL_REGISTRY[model_key]["model_id"]
                history   = await get_history(user_id)
                answer    = await call_groq_text(history, model_id)
                await save_message(user_id, "assistant", answer)
                reply = answer

        # ── REPLY ──────────────────────────────────────────────────────────
        if reply:
            # LINE hard-limit: 5000 chars per message
            reply = reply[:4990] + "…" if len(reply) > 4990 else reply
            await line_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply)],
                )
            )
