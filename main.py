import os, base64, aiosqlite, httpx, re
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
    )

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
WHISPER_MODEL             = "whisper-large-v3-turbo"

# ---------------------------------------------------------------------------
# MODEL REGISTRY
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict] = {
    "llama8b": {
        "model_id": "llama-3.1-8b-instant",
        "type":     "text",
        "tier":     "production",
        "display":  "LLaMA 3.1 8B",
        "ctx":      131_072,
        "note":     "Fastest ~900 t/s, rất rẻ — dùng cho chitchat & classifier",
    },
    "llama70b": {
        "model_id": "llama-3.3-70b-versatile",
        "type":     "text",
        "tier":     "production",
        "display":  "LLaMA 3.3 70B",
        "ctx":      131_072,
        "note":     "Balanced, tool-use tốt, viết lách, dịch thuật",
    },
    "gpt20b": {
        "model_id": "openai/gpt-oss-20b",
        "type":     "reasoning",
        "tier":     "production",
        "display":  "GPT-OSS 20B",
        "ctx":      131_072,
        "note":     "Siêu nhanh ~1000 t/s, reasoning nhẹ",
    },
    "gpt120b": {
        "model_id": "openai/gpt-oss-120b",
        "type":     "reasoning",
        "tier":     "production",
        "display":  "GPT-OSS 120B",
        "ctx":      131_072,
        "note":     "Reasoning mạnh nhất, ~500 t/s",
    },
    "scout": {
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "type":     "vision",
        "tier":     "preview",
        "display":  "LLaMA 4 Scout (Vision)",
        "ctx":      131_072,
        "note":     "Vision model duy nhất, latency thấp",
    },
    "qwen": {
        "model_id": "qwen/qwen3-32b",
        "type":     "reasoning",
        "tier":     "preview",
        "display":  "Qwen3 32B",
        "ctx":      131_072,
        "note":     "Reasoning + thinking mode, đa ngôn ngữ tốt",
    },
    "kimi": {
        "model_id": "moonshotai/kimi-k2-instruct-0905",
        "type":     "text",
        "tier":     "preview",
        "display":  "Kimi K2 0905",
        "ctx":      262_144,
        "note":     "Context 256K, agentic, coding",
    },
    "compound": {
        "model_id": "compound-beta",
        "type":     "text",
        "tier":     "production",
        "display":  "Groq Compound (web search)",
        "ctx":      131_072,
        "note":     "Web search + code execution built-in",
    },
    "compound-mini": {
        "model_id": "compound-beta-mini",
        "type":     "text",
        "tier":     "production",
        "display":  "Groq Compound Mini",
        "ctx":      131_072,
        "note":     "Compound nhanh hơn, 1 tool call",
    },
}

DEFAULT_MODEL_KEY   = "llama70b"
VISION_MODEL_KEY    = "scout"
CLASSIFIER_MODEL_KEY = "llama8b"

# ---------------------------------------------------------------------------
# AUTO-ROUTER
# ---------------------------------------------------------------------------

# Mapping category → model key
# Thứ tự ưu tiên: chất lượng > tốc độ (vì user chọn quality)
ROUTE_MAP: dict[str, str] = {
    "simple":    "llama8b",    # chitchat, chào hỏi → nhanh + rẻ
    "creative":  "llama70b",   # viết, dịch, tóm tắt → balanced
    "reasoning": "qwen",       # toán, logic, phân tích → thinking mode
    "hard":      "gpt120b",    # câu hỏi khó, mơ hồ, multi-step → mạnh nhất
    "search":    "compound",   # cần thông tin thực tế/mới → web search
}

# Classifier prompt: trả về đúng 1 từ, không giải thích
_CLASSIFIER_PROMPT = """Classify the user message into exactly one category.
Reply with ONLY one word from this list, nothing else.

Categories:
- simple    : greetings, chitchat, yes/no, very short factual (name, date)
- creative  : writing, translation, summarization, brainstorming, roleplay
- reasoning : math, logic, code, step-by-step analysis, comparison, explanation
- hard      : ambiguous complex questions, multi-domain, requires deep thinking
- search    : current events, prices, news, weather, "latest", "now", "today", real-time data

Message: {message}"""


async def classify_query(user_text: str) -> str:
    """
    Calls llama8b with max_tokens=3 to classify the query.
    Returns a key in ROUTE_MAP, defaults to DEFAULT_MODEL_KEY on error.
    Cost: negligible (~50 input tokens, 1 output token).
    """
    async with httpx.AsyncClient() as http:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)
        try:
            resp = await client.chat.completions.create(
                model=MODEL_REGISTRY[CLASSIFIER_MODEL_KEY]["model_id"],
                messages=[
                    {
                        "role": "user",
                        "content": _CLASSIFIER_PROMPT.format(
                            message=user_text[:400]  # cap để tránh prompt injection
                        ),
                    }
                ],
                temperature=0.0,  # deterministic — luôn cùng output cho cùng input
                max_tokens=3,
            )
            category = resp.choices[0].message.content.strip().lower()
            return ROUTE_MAP.get(category, DEFAULT_MODEL_KEY)
        except Exception:
            return DEFAULT_MODEL_KEY


async def resolve_model(user_id: str, user_text: str) -> tuple[str, str]:
    """
    Returns (model_key, model_id).

    Logic:
    - Nếu user đã manually override model → respect lựa chọn đó
    - Nếu user dùng default → auto-classify và route
    """
    current_key = await get_user_model(user_id)

    if current_key != DEFAULT_MODEL_KEY:
        # User đã chủ động chọn model — không override
        return current_key, MODEL_REGISTRY[current_key]["model_id"]

    routed_key = await classify_query(user_text)
    return routed_key, MODEL_REGISTRY[routed_key]["model_id"]


# ---------------------------------------------------------------------------
# DATABASE  (giữ nguyên logic, cải thiện history truncation)
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


async def get_history(user_id: str, max_chars: int = 4000) -> list[dict]:
    """
    Lấy lịch sử hội thoại, giới hạn theo ký tự thay vì số lượng cố định.
    Ưu tiên giữ tin nhắn gần nhất.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT role, content FROM history "
            "WHERE user_id = ? ORDER BY id DESC LIMIT 30",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()

    messages = [{"role": r, "content": c} for r, c in reversed(rows)]

    # Trim từ đầu nếu tổng vượt max_chars — giữ tin gần nhất
    total, trimmed = 0, []
    for msg in reversed(messages):
        total += len(msg["content"])
        if total > max_chars:
            break
        trimmed.insert(0, msg)

    return trimmed


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
# MARKDOWN STRIPPER
# ---------------------------------------------------------------------------
def strip_markdown(text: str) -> str:
    # Reasoning tags <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"^[\-\*]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"^(\-{3,}|\*{3,}|_{3,})\s*$", "─────", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
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
            return f"⚠️ Lỗi [{model_id}]: {str(e)[:150]}"


async def call_groq_vision(image_b64: str) -> str:
    """Luôn dùng scout — model vision duy nhất trên Groq hiện tại."""
    model_id = MODEL_REGISTRY[VISION_MODEL_KEY]["model_id"]
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
                                    "Phân tích và mô tả ảnh này chi tiết. "
                                    "Nếu có chữ hãy trích xuất toàn bộ. "
                                    "Nếu có chữ Trung hãy dùng Phồn thể."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                            },
                        ],
                    },
                ],
                max_tokens=800,
            )
            return strip_markdown(resp.choices[0].message.content or "")
        except Exception as e:
            return f"⚠️ Lỗi vision: {str(e)[:150]}"


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
    prod_lines, prev_lines = [], []
    for key, cfg in MODEL_REGISTRY.items():
        icon = {"vision": "👁", "reasoning": "🧠", "text": "💬"}.get(cfg["type"], "💬")
        line = f"{icon} /{key} — {cfg['display']}\n   {cfg['note']}"
        if cfg["tier"] == "production":
            prod_lines.append(line)
        else:
            prev_lines.append(line)

    return "\n".join([
        "📋 DANH SÁCH MODEL\n",
        "── Production ──",
        *prod_lines,
        "\n── Preview ──",
        *prev_lines,
        "\n─────────────────",
        "Chuyển model:  /model <tên>",
        "Model hiện tại: /model",
        "Xoá lịch sử:   /clear",
        "Chế độ tự động: /auto",
    ])


async def handle_command(user_id: str, text: str) -> str | None:
    if not text.startswith("/"):
        return None

    parts = text[1:].strip().split(maxsplit=1)
    cmd   = parts[0].lower()
    arg   = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "clear":
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
            await db.commit()
        return "🗑 Đã xoá lịch sử hội thoại."

    if cmd == "models":
        return _models_list_text()

    # /auto — reset về auto-routing
    if cmd == "auto":
        await set_user_model(user_id, DEFAULT_MODEL_KEY)
        return "🤖 Đã bật chế độ tự động chọn model."

    if cmd == "model":
        if not arg:
            key = await get_user_model(user_id)
            cfg = MODEL_REGISTRY[key]
            mode = "Tự động" if key == DEFAULT_MODEL_KEY else "Thủ công"
            return (
                f"🤖 Model hiện tại: {cfg['display']}\n"
                f"   Tier: {cfg['tier']} | Chế độ: {mode}\n"
                f"   {cfg['note']}\n\n"
                "Dùng /models để xem tất cả.\n"
                "Dùng /auto để về chế độ tự động."
            )
        target = arg.lower()
        if target not in MODEL_REGISTRY:
            return f"❌ /{target} không tồn tại.\nDùng /models để xem danh sách."
        await set_user_model(user_id, target)
        return f"✅ Đã chuyển sang {MODEL_REGISTRY[target]['display']}.\nDùng /auto để về tự động."

    if cmd in MODEL_REGISTRY:
        await set_user_model(user_id, cmd)
        cfg = MODEL_REGISTRY[cmd]
        if arg:
            answer = await call_groq_text(
                [{"role": "user", "content": arg}], cfg["model_id"]
            )
            return f"[{cfg['display']}]\n{answer}"
        return f"✅ Chuyển sang {cfg['display']}.\nDùng /auto để về tự động."

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

        try:
            await line_api.show_loading_animation(
                ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=10)
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
                # Audio cũng đi qua auto-router như text thường
                model_key, model_id = await resolve_model(user_id, transcript)
                history = await get_history(user_id)
                answer  = await call_groq_text(history, model_id)
                await save_message(user_id, "assistant", answer)
                reply = f"🎤 {transcript}\n\n{answer}"
            else:
                reply = transcript

        # ── IMAGE ──────────────────────────────────────────────────────────
        elif isinstance(event.message, ImageMessageContent):
            img_bytes = await line_blob_api.get_message_content(event.message.id)
            img_b64   = base64.b64encode(img_bytes).decode("utf-8")
            # Vision luôn dùng scout — không cần route
            reply = await call_groq_vision(img_b64)

        # ── TEXT ───────────────────────────────────────────────────────────
        elif isinstance(event.message, TextMessageContent):
            user_text = event.message.text.strip()

            cmd_reply = await handle_command(user_id, user_text)
            if cmd_reply is not None:
                reply = cmd_reply
            else:
                await save_message(user_id, "user", user_text)
                model_key, model_id = await resolve_model(user_id, user_text)
                history = await get_history(user_id)
                answer  = await call_groq_text(history, model_id)
                await save_message(user_id, "assistant", answer)
                reply = answer

        # ── REPLY ──────────────────────────────────────────────────────────
        if reply:
            reply = reply[:4990] + "…" if len(reply) > 4990 else reply
            await line_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply)],
                )
            )
