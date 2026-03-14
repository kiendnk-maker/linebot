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
from prompts import get_system_prompt

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
LINE_CHANNEL_SECRET       = os.environ["LINE_CHANNEL_SECRET"]
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
GROQ_API_KEY              = os.environ["GROQ_API_KEY"]
DB_PATH                   = "chat_history.db"
WHISPER_MODEL             = "whisper-large-v3-turbo"

# ---------------------------------------------------------------------------
# MODEL REGISTRY
# Source: https://console.groq.com/docs/models  (checked 2026-03-14)
#
# tier:  "production" | "preview"
# type:  "text" | "vision" | "reasoning"
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict] = {
    # ── Production ──────────────────────────────────────────────────────────
    "llama8b": {
        "model_id": "llama-3.1-8b-instant",
        "type":     "text",
        "tier":     "production",
        "display":  "LLaMA 3.1 8B",
        "ctx":      131_072,
        "note":     "最快 ~900 t/s，超便宜 — 用於 classifier 和簡單對話",
    },
    "llama70b": {
        "model_id": "llama-3.3-70b-versatile",
        "type":     "text",
        "tier":     "production",
        "display":  "LLaMA 3.3 70B",
        "ctx":      131_072,
        "note":     "均衡性能，寫作、翻譯、多語言",
    },
    "gpt20b": {
        "model_id": "openai/gpt-oss-20b",
        "type":     "reasoning",
        "tier":     "production",
        "display":  "GPT-OSS 20B",
        "ctx":      131_072,
        "note":     "超快 ~1000 t/s，輕量推理",
    },
    "gpt120b": {
        "model_id": "openai/gpt-oss-120b",
        "type":     "reasoning",
        "tier":     "production",
        "display":  "GPT-OSS 120B",
        "ctx":      131_072,
        "note":     "最強推理，~500 t/s",
    },
    "compound": {
        "model_id": "groq/compound",
        "type":     "text",
        "tier":     "production",
        "display":  "Groq Compound（網路搜尋）",
        "ctx":      131_072,
        "note":     "內建網路搜尋 + 程式執行，最多10次工具呼叫",
    },
    "compound-mini": {
        "model_id": "groq/compound-mini",
        "type":     "text",
        "tier":     "production",
        "display":  "Groq Compound Mini",
        "ctx":      131_072,
        "note":     "Compound 輕量版，單次工具呼叫，速度快3倍",
    },
    # ── Preview ─────────────────────────────────────────────────────────────
    "scout": {
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "type":     "vision",
        "tier":     "preview",
        "display":  "LLaMA 4 Scout（視覺）",
        "ctx":      131_072,
        "note":     "唯一視覺模型，低延遲",
    },
    "qwen": {
        "model_id": "qwen/qwen3-32b",
        "type":     "reasoning",
        "tier":     "preview",
        "display":  "Qwen3 32B",
        "ctx":      131_072,
        "note":     "推理 + thinking mode，多語言強",
    },
    "kimi": {
        "model_id": "moonshotai/kimi-k2-instruct-0905",
        "type":     "text",
        "tier":     "preview",
        "display":  "Kimi K2 0905",
        "ctx":      262_144,
        "note":     "最長 context 256K，agentic coding",
    },
}

DEFAULT_MODEL_KEY    = "llama70b"
VISION_MODEL_KEY     = "scout"
CLASSIFIER_MODEL_KEY = "llama8b"

# Triggers kích hoạt LLM reply khi ghi âm
_REPLY_TRIGGERS = ("hãy trả lời", "請回答我")

# ---------------------------------------------------------------------------
# AUTO-ROUTER
# ---------------------------------------------------------------------------
ROUTE_MAP: dict[str, str] = {
    "simple":    "llama8b",
    "creative":  "llama70b",
    "reasoning": "qwen",
    "hard":      "gpt120b",
    "search":    "compound-mini",
}

# Keywords rõ ràng cần real-time → bypass classifier, dùng compound-mini
_REALTIME_KEYWORDS = (
    "hôm nay", "bây giờ", "hiện tại", "mới nhất", "today", "now", "latest",
    "giá ", "price", "tỷ giá", "thời tiết", "weather", "tin tức", "news",
    "今天", "現在", "最新", "價格", "新聞", "天氣",
)

_CLASSIFIER_PROMPT = """Classify the user message into exactly one category.
Reply with ONLY one word from this list, nothing else.

Categories:
- simple    : greetings, chitchat, yes/no, very short factual (name, date)
- creative  : writing, translation, summarization, brainstorming, roleplay
- reasoning : math, logic, code, step-by-step analysis, comparison, explanation
- hard      : ambiguous complex questions, multi-domain, requires deep thinking
- search    : current events, prices, news, weather, "latest", "now", "today"

Message: {message}"""


def _needs_realtime(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in _REALTIME_KEYWORDS)


async def classify_query(user_text: str) -> str:
    """
    Calls llama8b with max_tokens=3 to classify the query.
    Returns a key in MODEL_REGISTRY. Falls back to DEFAULT_MODEL_KEY on error.
    Cost: ~50 input tokens + 1 output token per call — negligible.
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
                            message=user_text[:400]
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=3,
            )
            category = resp.choices[0].message.content.strip().lower()
            return ROUTE_MAP.get(category, DEFAULT_MODEL_KEY)
        except Exception:
            return DEFAULT_MODEL_KEY


async def resolve_model(user_id: str, user_text: str) -> tuple[str, str]:
    """
    Returns (model_key, model_id).
    Priority:
    1. User manually chose a model → respect that choice.
    2. Real-time keywords detected → compound-mini (no classifier call).
    3. Default → auto-classify and route.
    """
    current_key = await get_user_model(user_id)

    if current_key != DEFAULT_MODEL_KEY:
        return current_key, MODEL_REGISTRY[current_key]["model_id"]

    if _needs_realtime(user_text):
        return "compound-mini", MODEL_REGISTRY["compound-mini"]["model_id"]

    routed_key = await classify_query(user_text)
    return routed_key, MODEL_REGISTRY[routed_key]["model_id"]


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


async def get_history(user_id: str, max_chars: int = 4000) -> list[dict]:
    """
    Fetch conversation history, capped by total characters (not fixed count).
    Keeps the most recent messages when trimming.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT role, content FROM history "
            "WHERE user_id = ? ORDER BY id DESC LIMIT 30",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()

    messages = [{"role": r, "content": c} for r, c in reversed(rows)]

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
    """Remove <think> blocks and common Markdown symbols for LINE output."""
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
async def call_groq_text(
    history: list[dict],
    model_id: str,
    model_key: str = DEFAULT_MODEL_KEY,
) -> str:
    system = get_system_prompt(model_key)
    async with httpx.AsyncClient() as http:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": system}] + history,
                temperature=0.6,
                max_tokens=800,
            )
            return strip_markdown(resp.choices[0].message.content or "")
        except Exception as e:
            return f"⚠️ 錯誤 [{model_id}]: {str(e)[:150]}"


async def call_groq_vision(image_b64: str) -> str:
    """Always uses scout — the only vision model on Groq."""
    model_id = MODEL_REGISTRY[VISION_MODEL_KEY]["model_id"]
    system   = get_system_prompt(VISION_MODEL_KEY)
    async with httpx.AsyncClient() as http:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "請詳細分析並描述這張圖片。"
                                    "若有文字請完整擷取。"
                                    "若有中文請使用繁體中文。"
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
            return f"⚠️ 視覺錯誤: {str(e)[:150]}"


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
            return f"⚠️ Whisper 錯誤: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# COMMAND SYSTEM
# ---------------------------------------------------------------------------
def _models_list_text() -> str:
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
        "📋 可用模型列表\n",
        "── Production ──",
        *prod_lines,
        "\n── Preview ──",
        *prev_lines,
        "\n─────────────────",
        "切換模型：/model <名稱>",
        "目前模型：/model",
        "自動模式：/auto",
        "清除紀錄：/clear",
    ]
    return "\n".join(parts)


async def handle_command(user_id: str, text: str) -> str | None:
    """
    Parse slash commands.
    Returns reply string if text is a command, else None.
    """
    if not text.startswith("/"):
        return None

    parts = text[1:].strip().split(maxsplit=1)
    cmd   = parts[0].lower()
    arg   = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "clear":
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
            await db.commit()
        return "🗑 對話記錄已清除。"

    if cmd == "models":
        return _models_list_text()

    if cmd == "auto":
        await set_user_model(user_id, DEFAULT_MODEL_KEY)
        return "🤖 已切換至自動選擇模型模式。"

    if cmd == "model":
        if not arg:
            key  = await get_user_model(user_id)
            cfg  = MODEL_REGISTRY[key]
            mode = "自動" if key == DEFAULT_MODEL_KEY else "手動"
            return (
                f"🤖 目前模型：{cfg['display']}\n"
                f"   Tier：{cfg['tier']} | 模式：{mode}\n"
                f"   {cfg['note']}\n\n"
                "輸入 /models 查看全部。\n"
                "輸入 /auto 返回自動模式。"
            )
        target = arg.lower()
        if target not in MODEL_REGISTRY:
            return f"❌ /{target} 不存在。\n請輸入 /models 查看清單。"
        await set_user_model(user_id, target)
        return (
            f"✅ 已切換至 {MODEL_REGISTRY[target]['display']}。\n"
            "輸入 /auto 返回自動模式。"
        )

    if cmd in MODEL_REGISTRY:
        await set_user_model(user_id, cmd)
        cfg = MODEL_REGISTRY[cmd]
        if arg:
            answer = await call_groq_text(
                [{"role": "user", "content": arg}],
                cfg["model_id"],
                model_key=cmd,
            )
            return f"[{cfg['display']}]\n{answer}"
        return (
            f"✅ 已切換至 {cfg['display']}。\n"
            "輸入 /auto 返回自動模式。"
        )

    return f"❓ 指令 /{cmd} 無效。請輸入 /models 查看。"


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
                wants_reply = any(
                    transcript.strip().lower().startswith(t.lower())
                    for t in _REPLY_TRIGGERS
                )

                if wants_reply:
                    # Strip trigger prefix trước khi xử lý
                    clean_text = transcript.strip()
                    for t in _REPLY_TRIGGERS:
                        if clean_text.lower().startswith(t.lower()):
                            clean_text = clean_text[len(t):].strip()
                            break

                    await save_message(user_id, "user", clean_text)
                    model_key, model_id = await resolve_model(user_id, clean_text)
                    history = await get_history(user_id)
                    answer  = await call_groq_text(history, model_id, model_key=model_key)
                    await save_message(user_id, "assistant", answer)
                    reply = f"🎤 {clean_text}\n\n{answer}"
                else:
                    # Transcribe only — lưu history nhưng không gọi LLM
                    await save_message(user_id, "user", f"[Voice]: {transcript}")
                    reply = f"🎤 {transcript}"
            else:
                reply = transcript

        # ── IMAGE ──────────────────────────────────────────────────────────
        elif isinstance(event.message, ImageMessageContent):
            img_bytes = await line_blob_api.get_message_content(event.message.id)
            img_b64   = base64.b64encode(img_bytes).decode("utf-8")
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
                answer  = await call_groq_text(history, model_id, model_key=model_key)
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
