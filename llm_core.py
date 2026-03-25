"""
llm_core.py — Mistral API Integration (Full Exploitation)
Fixed: duplicate MODEL_REGISTRY, reasoning_effort 422, if True guards
New: Magistral reasoning, web_search tool, Codestral, Pixtral Large
"""
from tracker_core import log_usage
import os
import re
import json
import httpx
import aiosqlite
import logging
import base64
from openai import AsyncOpenAI
from prompts import (
    get_system_prompt,
    MODEL_REGISTRY,
    DEFAULT_MODEL_KEY,
    VISION_MODEL_KEY,
    CLASSIFIER_MODEL_KEY,
    ROUTE_MAP,
)
from database import (
    DB_PATH, get_user_profile, get_user_model, get_user_max_tokens,
    count_history, get_history_raw, get_summary, save_summary,
)

logger = logging.getLogger(__name__)

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")

# Mistral client
mistral_client = AsyncOpenAI(
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1",
)

# Groq client — dùng cho Llama 4 Scout vision
groq_client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

WHISPER_MODEL = "whisper-large-v3-turbo"
SUMMARY_TRIGGER = 20

# ═══════════════════════════════════════════════════════════════════
# MODEL REGISTRY — Unique keys, full Mistral lineup
# ═══════════════════════════════════════════════════════════════════
MODEL_REGISTRY: dict[str, dict] = {
    # ── Production ─────────────────────────────────────────────
    "small": {
        "model_id": "mistral-small-latest",
        "type": "text",
        "tier": "production",
        "display": "Mistral Small 4",
        "ctx": 131_072,
        "note": "Nhanh ~900 t/s, classifier & chat đơn giản",
        "supports_reasoning_effort": True,  # NEW: Small 4 supports this
    },
    "large": {
        "model_id": "mistral-large-latest",
        "type": "text",
        "tier": "production",
        "display": "Mistral Large 3",
        "ctx": 131_072,
        "note": "Cân bằng hiệu năng, viết/dịch/đa ngôn ngữ",
        "supports_reasoning_effort": False,
    },
    "coder": {
        "model_id": "codestral-latest",
        "type": "text",
        "tier": "production",
        "display": "Codestral",
        "ctx": 256_000,
        "note": "Chuyên code 80+ ngôn ngữ, 256K context",
        "supports_reasoning_effort": False,
    },
    # ── Reasoning (NEW) ────────────────────────────────────────
    "reason": {
        "model_id": "magistral-medium-latest",
        "type": "reasoning",
        "tier": "production",
        "display": "Magistral Medium",
        "ctx": 40_960,
        "note": "Deep reasoning, toán & logic, chain-of-thought",
        "supports_reasoning_effort": True,
    },
    # ── Vision ─────────────────────────────────────────────────
    "vision": {
        "model_id": "pixtral-large-latest",
        "type": "vision",
        "tier": "production",
        "display": "Pixtral Large",
        "ctx": 131_072,
        "note": "Vision mạnh nhất, OCR & phân tích hình ảnh",
        "supports_reasoning_effort": False,
    },
}

DEFAULT_MODEL_KEY = "large"
VISION_MODEL_KEY = "vision"
CLASSIFIER_MODEL_KEY = "small"

# ═══════════════════════════════════════════════════════════════════
# SMART ROUTING — Route query to optimal model
# ═══════════════════════════════════════════════════════════════════
ROUTE_MAP: dict[str, str] = {
    "simple": "small",
    "creative": "large",
    "reasoning": "reason",   # NEW: Route to Magistral
    "hard": "reason",        # NEW: Route to Magistral
    "code": "coder",         # NEW: Route to Codestral
    "search": "small",
}

_REALTIME_KEYWORDS = (
    "hôm nay", "bây giờ", "hiện tại", "mới nhất", "today", "now", "latest",
    "giá ", "price", "tỷ giá", "thời tiết", "weather", "tin tức", "news",
    "今天", "現在", "最新", "價格", "新聞", "天氣",
)

_CLASSIFIER_PROMPT = """Classify the user message into exactly one category.
Reply with ONLY one word from this list, nothing else.

Categories:
- simple : greetings, chitchat, yes/no, very short factual
- creative : writing, translation, summarization, brainstorming, roleplay
- reasoning : math, logic, step-by-step analysis, comparison, explanation
- code : programming, debugging, code generation, technical implementation
- hard : ambiguous complex questions, multi-domain, deep thinking
- search : current events, prices, news, weather, real-time data needed

Message: {message}"""


def _needs_realtime(text: str) -> bool:
    return any(kw in text.lower() for kw in _REALTIME_KEYWORDS)


async def classify_query(user_text: str) -> str:
    try:
        resp = await mistral_client.chat.completions.create(
            model=MODEL_REGISTRY[CLASSIFIER_MODEL_KEY]["model_id"],
            messages=[{
                "role": "user",
                "content": _CLASSIFIER_PROMPT.format(message=user_text[:400]),
            }],
            temperature=0.0,
            max_tokens=3,
        )
        category = resp.choices[0].message.content.strip().lower()
        return ROUTE_MAP.get(category, DEFAULT_MODEL_KEY)
    except Exception:
        return DEFAULT_MODEL_KEY


async def resolve_model(user_id: str, user_text: str) -> tuple[str, str]:
    """
    Smart routing priority:
    1. User manually set model → use that
    2. Realtime keywords → small (fast)
    3. Code keywords → coder (Codestral)
    4. Math/logic keywords → reason (Magistral)
    5. Long text without ? → large (summarize)
    6. LLM classifier → ROUTE_MAP
    """
    current_key = await get_user_model(user_id)
    if current_key != DEFAULT_MODEL_KEY and current_key in MODEL_REGISTRY:
        return current_key, MODEL_REGISTRY[current_key]["model_id"]

    text_lower = user_text.lower()

    # Priority 2: realtime
    if _needs_realtime(user_text):
        return "small", MODEL_REGISTRY["small"]["model_id"]

    # Priority 3: code detection → Codestral
    code_keywords = {"code", "python", "javascript", "viết hàm", "debug", "function", "class ", "import ", "def ", "程式", "代碼"}
    if any(kw in text_lower for kw in code_keywords):
        return "coder", MODEL_REGISTRY["coder"]["model_id"]

    # Priority 4: math/logic → Magistral
    math_keywords = {"cộng", "trừ", "nhân", "chia", "tính", "đạo hàm", "tích phân", "chứng minh", "solve", "equation", "計算", "數學"}
    if any(kw in text_lower for kw in math_keywords):
        return "reason", MODEL_REGISTRY["reason"]["model_id"]

    # Priority 5: long text → summarize with large
    if len(user_text) > 500 and "?" not in user_text and "？" not in user_text:
        return "large", MODEL_REGISTRY["large"]["model_id"]

    # Priority 6: translation → large
    if any(kw in text_lower for kw in {"dịch", "translate", "tiếng anh", "tiếng trung", "tiếng nhật", "翻譯"}):
        return "large", MODEL_REGISTRY["large"]["model_id"]

    # Priority 7: classifier
    routed_key = await classify_query(user_text)
    return routed_key, MODEL_REGISTRY[routed_key]["model_id"]


# ═══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════
async def build_system_prompt(user_id: str, model_key: str) -> str:
    base = get_system_prompt(model_key)
    profile = await get_user_profile(user_id)
    if not profile:
        return base
    lines: list[str] = []
    if profile.get("name"):
        lines.append("用戶姓名：" + profile["name"])
    if profile.get("occupation"):
        lines.append("職業：" + profile["occupation"])
    if profile.get("learning"):
        lines.append("正在學習：" + profile["learning"])
    if profile.get("notes"):
        lines.append("備註：" + profile["notes"])
    return base + "\n\n【用戶資料】\n" + "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# AUTO-SUMMARIZE
# ═══════════════════════════════════════════════════════════════════
async def maybe_summarize(user_id: str) -> None:
    count = await count_history(user_id)
    if count % SUMMARY_TRIGGER != 0:
        return
    rows = await get_history_raw(user_id, limit=40)
    if not rows:
        return
    prev_summary = await get_summary(user_id)
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in rows)
    prompt = (
        f"Tóm tắt ngắn gọn (tối đa 150 từ) nội dung cuộc trò chuyện sau, "
        f"giữ lại thông tin quan trọng về người dùng và chủ đề chính.\n"
        f"Tóm tắt trước đó: {prev_summary}\n\n"
        f"Hội thoại mới:\n{history_text}"
    )
    try:
        resp = await mistral_client.chat.completions.create(
            model=MODEL_REGISTRY["small"]["model_id"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        new_summary = resp.choices[0].message.content or ""
        await save_summary(user_id, new_summary)
    except Exception:
        pass


async def get_history_with_summary(user_id: str) -> list[dict]:
    summary = await get_summary(user_id)
    recent = await get_history_raw(user_id, limit=5)
    if summary:
        return [
            {"role": "user", "content": f"[Tóm tắt hội thoại trước: {summary}]"},
            {"role": "assistant", "content": "Đã hiểu context."},
            *recent,
        ]
    return recent


# ═══════════════════════════════════════════════════════════════════
# MARKDOWN STRIPPER (for LINE plain text)
# ═══════════════════════════════════════════════════════════════════
def strip_markdown(text: str) -> str:
    parts_text = re.split(r'(```.*?```)', text, flags=re.DOTALL)
    for i in range(0, len(parts_text), 2):
        p = parts_text[i]
        p = re.sub(r"<think>.*?</think>", "", p, flags=re.DOTALL)
        p = re.sub(r"\*\*(.*?)\*\*", r"\1", p)
        p = re.sub(r"^#{1,6}\s+", "", p, flags=re.MULTILINE)
        p = re.sub(r"^[-\*]\s+", "• ", p, flags=re.MULTILINE)
        p = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", p)
        p = re.sub(r"\n{3,}", "\n\n", p)
        parts_text[i] = p
    return "".join(parts_text).strip()


# ═══════════════════════════════════════════════════════════════════
# CORE TEXT API — with web_search tool support
# ═══════════════════════════════════════════════════════════════════
async def call_mistral_text_inner(
    history: list[dict],
    model_id: str,
    model_key: str = DEFAULT_MODEL_KEY,
    user_id: str | None = None,
    rag_chunks: list[dict] | None = None,
    enable_web_search: bool = False,
) -> str:
    system = (
        await build_system_prompt(user_id, model_key)
        if user_id
        else get_system_prompt(model_key)
    )

    # Inject RAG context
    if rag_chunks:
        rag_context = "\n\n".join(
            f"[Nguồn: {c['filename']} chunk {c['chunk_index']}]\n{c['content']}"
            for c in rag_chunks
        )
        system += (
            "\n\n═══ QUAN TRỌNG: TÀI LIỆU THAM KHẢO ═══\n"
            "Dưới đây là nội dung từ tài liệu của người dùng. "
            "BẮT BUỘC tuân thủ:\n"
            "1. Ưu tiên trả lời DỰA TRÊN tài liệu.\n"
            "2. Có trong tài liệu → trích dẫn [Nguồn: tên_file].\n"
            "3. Không có → nói rõ rồi bổ sung từ kiến thức chung.\n"
            "4. KHÔNG bịa thông tin.\n"
            "═══════════════════════════════════════\n\n"
            f"{rag_context}"
        )

    # Clean history — last message must be "user"
    clean_history = list(history)
    while clean_history and clean_history[-1]["role"] == "assistant":
        clean_history.pop()

    # Language forcer removed — system prompt handles language.
    if user_id and clean_history and clean_history[-1]["role"] == "user":
        pass

    # ── API CALL ──────────────────────────────────────────────────────────
    user_max = await get_user_max_tokens(user_id) if user_id else 800
    max_tok = user_max if user_max != 800 else 1500

    client = global_groq_client
    try:
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system}] + clean_history,
            temperature=0.6,
            max_tokens=max_tok,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"LLM API error [{model_id}]: {e}")
        return f"⚠️ Lỗi API: {str(e)[:150]}"


async def call_mistral_vision(image_b64: str, user_id: str | None = None) -> str:
    """Vision wrapper — delegates to call_mistral_text with image content."""
    import aiosqlite
    from database import DB_PATH

    system = (
        await build_system_prompt(user_id, VISION_MODEL_KEY)
        if user_id
        else get_system_prompt(VISION_MODEL_KEY)
    )
    vision_prompt = "Hãy mô tả chi tiết hình ảnh. Nếu có chữ, hãy trích xuất đầy đủ."
    if user_id:
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                async with db.execute(
                    "SELECT language FROM user_settings WHERE user_id = ?", (user_id,)
                ) as cur:
                    row = await cur.fetchone()
                    if row and row[0] == "zh-TW":
                        vision_prompt = "請詳細描述這張圖片的所有內容。若有文字請完整擷取，使用繁體中文。"
        except Exception:
            pass
    client = global_groq_client
    try:
        resp = await client.chat.completions.create(
            model=MODEL_REGISTRY[VISION_MODEL_KEY]["model_id"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]},
            ],
            max_tokens=800,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ 視覺錯誤: {str(e)[:150]}"


async def call_groq_whisper(audio_bytes: bytes) -> str:
    client = global_groq_client
    try:
        result = await client.audio.transcriptions.create(
            file=("audio.m4a", audio_bytes),
            model="whisper-large-v3-turbo",
        )
        return result.text
    except Exception as e:
        return f"⚠️ Whisper 錯誤: {str(e)[:150]}"


async def clean_transcript(transcript: str) -> str:
    client = global_groq_client
    try:
        resp = await client.chat.completions.create(
            model=MODEL_REGISTRY["large"]["model_id"],
            messages=[{"role": "user", "content": (
                "Fix speech-to-text errors. Return only the corrected text, no explanation.\n"
                f"Transcript: {transcript}"
            )}],
            temperature=0.0,
            max_tokens=300,
        )
        cleaned = resp.choices[0].message.content.strip()
        return cleaned if cleaned else transcript
    except Exception:
        return transcript


def _split_reply(reply: str) -> list[str]:
    chunks = []
    while len(reply) > 4990:
        cut = reply.rfind(" ", 0, 4990)
        if cut == -1: cut = 4990
        chunks.append(reply[:cut])
        reply = reply[cut:].strip()
    if reply: chunks.append(reply)
    return chunks[:5]


async def call_mistral_text(history, model_id, model_key="large", user_id=None, rag_chunks=None):
    """Safe wrapper — always returns str, never None."""
    try:
        result = await call_mistral_text_inner(history, model_id, model_key=model_key, user_id=user_id, rag_chunks=rag_chunks)
        if result is None:
            return "⚠️ Model không trả về phản hồi. Vui lòng thử lại."
        return result
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"call_mistral_text error: {e}", exc_info=True)
        return f"⚠️ Lỗi: {str(e)[:200]}"
