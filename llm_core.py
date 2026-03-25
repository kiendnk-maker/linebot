"""
llm_core.py - LINE bot LLM core
Clean rewrite - no patch artifacts
"""

import os
import re
import logging
import aiosqlite

from openai import AsyncOpenAI
from prompts import get_system_prompt, MODEL_REGISTRY, DEFAULT_MODEL_KEY, VISION_MODEL_KEY
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

# Groq client
groq_client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

WHISPER_MODEL   = "whisper-large-v3-turbo"
SUMMARY_TRIGGER = 20

CLASSIFIER_MODEL_KEY = "small"

ROUTE_MAP = {
    "simple":    "small",
    "creative":  "large",
    "reasoning": "small",
    "hard":      "large",
    "search":    "large",
}

_REALTIME_KEYWORDS = (
    "hôm nay", "bây giờ", "hiện tại", "mới nhất", "today", "now", "latest",
    "giá ", "price", "tỷ giá", "thời tiết", "weather", "tin tức", "news",
    "今天", "現在", "最新", "價格", "新聞", "天氣",
)

_CLASSIFIER_PROMPT = """Classify the user message into exactly one category.
Reply with ONLY one word from: simple, creative, reasoning, hard, search

- simple   : greetings, chitchat, yes/no, very short factual
- creative : writing, translation, summarization, brainstorming
- reasoning: math, logic, code, step-by-step analysis
- hard     : ambiguous complex, multi-domain, deep thinking
- search   : current events, prices, news, weather, "latest", "now", "today"

Message: {message}"""


def _needs_realtime(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in _REALTIME_KEYWORDS)


async def classify_query(user_text: str) -> str:
    try:
        resp = await mistral_client.chat.completions.create(
            model=MODEL_REGISTRY[CLASSIFIER_MODEL_KEY]["model_id"],
            messages=[{"role": "user", "content": _CLASSIFIER_PROMPT.format(message=user_text[:400])}],
            temperature=0.0,
            max_tokens=5,
        )
        cat = resp.choices[0].message.content.strip().lower()
        return ROUTE_MAP.get(cat, DEFAULT_MODEL_KEY)
    except Exception:
        return DEFAULT_MODEL_KEY


async def resolve_model(user_id: str, user_text: str) -> tuple[str, str]:
    current_key = await get_user_model(user_id)
    if current_key != DEFAULT_MODEL_KEY:
        return current_key, MODEL_REGISTRY[current_key]["model_id"]

    if _needs_realtime(user_text):
        return "small", MODEL_REGISTRY["small"]["model_id"]

    if len(user_text) > 500 and "?" not in user_text and "？" not in user_text:
        return "large", MODEL_REGISTRY["large"]["model_id"]

    t = user_text.lower()
    if any(kw in t for kw in {"cộng", "trừ", "nhân", "chia", "tính", "đạo hàm", "tích phân"}):
        return "small", MODEL_REGISTRY["small"]["model_id"]
    if any(kw in t for kw in {"dịch", "translate", "tiếng anh", "tiếng trung", "tiếng nhật"}):
        return "large", MODEL_REGISTRY["large"]["model_id"]

    routed = await classify_query(user_text)
    return routed, MODEL_REGISTRY[routed]["model_id"]


async def build_system_prompt(user_id: str, model_key: str) -> str:
    base = get_system_prompt(model_key)
    profile = await get_user_profile(user_id)
    if not profile:
        return base
    lines = []
    if profile.get("name"):       lines.append("用戶姓名：" + profile["name"])
    if profile.get("occupation"): lines.append("職業：" + profile["occupation"])
    if profile.get("learning"):   lines.append("正在學習：" + profile["learning"])
    if profile.get("notes"):      lines.append("備註：" + profile["notes"])
    return base + "\n\n【用戶資料】\n" + "\n".join(lines)


async def get_history_with_summary(user_id: str) -> list[dict]:
    summary = await get_summary(user_id)
    recent  = await get_history_raw(user_id, limit=5)
    if summary:
        return [
            {"role": "user",      "content": f"[Tóm tắt hội thoại trước: {summary}]"},
            {"role": "assistant", "content": "Đã hiểu context."},
            *recent,
        ]
    return recent


async def maybe_summarize(user_id: str) -> None:
    count = await count_history(user_id)
    if count % SUMMARY_TRIGGER != 0:
        return
    rows = await get_history_raw(user_id, limit=40)
    if not rows:
        return
    prev = await get_summary(user_id)
    text = "\n".join(f"{m['role']}: {m['content']}" for m in rows)
    prompt = (
        f"Tóm tắt ngắn gọn (tối đa 150 từ):\n"
        f"Tóm tắt trước đó: {prev}\n\nHội thoại:\n{text}"
    )
    try:
        resp = await mistral_client.chat.completions.create(
            model=MODEL_REGISTRY["small"]["model_id"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        await save_summary(user_id, resp.choices[0].message.content or "")
    except Exception:
        pass


def _enforce_max_bullets(text: str, max_b: int = 3) -> str:
    """If response has more than max_b bullets, merge extras into prose."""
    lines = text.splitlines()
    bullet_indices = [i for i, l in enumerate(lines) if l.strip().startswith("•")]
    if len(bullet_indices) <= max_b:
        return text

    # Collect all bullet texts and non-bullet lines
    result = []
    bullets_collected = []
    for line in lines:
        if line.strip().startswith("•"):
            bullets_collected.append(line.strip()[1:].strip())
        else:
            if bullets_collected:
                # Flush bullets: first 3 as bullets, rest as prose
                for j, b in enumerate(bullets_collected):
                    if j < max_b:
                        result.append(f"• {b}")
                    else:
                        # Append to last bullet as prose continuation
                        if result and result[-1].startswith("•"):
                            result.append(b + ".")
                        else:
                            result[-1] += " " + b + "."
                bullets_collected = []
            result.append(line)

    if bullets_collected:
        for j, b in enumerate(bullets_collected):
            if j < max_b:
                result.append(f"• {b}")
            else:
                result.append(b + ".")

    return "\n".join(result)


def strip_markdown(text: str) -> str:
    parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    for i in range(0, len(parts), 2):
        p = parts[i]
        p = re.sub(r"<think>.*?</think>", "", p, flags=re.DOTALL)
        p = re.sub(r"\*\*(.*?)\*\*", r"\1", p)
        p = re.sub(r"^#{1,6}\s+", "", p, flags=re.MULTILINE)
        p = re.sub(r"^[-*]\s+", "• ", p, flags=re.MULTILINE)
        p = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", p)
        p = re.sub(r"\n{3,}", "\n\n", p)
        parts[i] = p
    result = "".join(parts).strip()
    return _enforce_max_bullets(result)


def _split_reply(reply: str) -> list[str]:
    chunks = []
    while len(reply) > 4990:
        cut = reply.rfind(" ", 0, 4990)
        if cut == -1:
            cut = 4990
        chunks.append(reply[:cut])
        reply = reply[cut:].strip()
    if reply:
        chunks.append(reply)
    return chunks[:5]


async def call_mistral_text(
    history: list[dict],
    model_id: str,
    model_key: str = DEFAULT_MODEL_KEY,
    user_id: str | None = None,
    rag_chunks: list[dict] | None = None,
) -> str:
    """Call Mistral text model. Always returns str."""
    # Check if user wants detailed response (skip bullet limit)
    _detail_keywords = ("chi tiết", "detailed", "detail", "list all", "liệt kê", "列出", "詳細")
    _last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
    _want_detail = any(kw in _last_user.lower() for kw in _detail_keywords)
    # Check if user wants detailed response (skip bullet limit)
    _detail_keywords = ("chi tiết", "detailed", "detail", "list all", "liệt kê", "列出", "詳細")
    _last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
    _want_detail = any(kw in _last_user.lower() for kw in _detail_keywords)
    try:
        system = (
            await build_system_prompt(user_id, model_key)
            if user_id
            else get_system_prompt(model_key)
        )

        if rag_chunks:
            rag_ctx = "\n\n".join(
                f"[{c.get('filename','?')} chunk {c.get('chunk_index',0)}]\n{c.get('content','')}"
                for c in rag_chunks
            )
            system += (
                "\n\n═══ TÀI LIỆU THAM KHẢO ═══\n"
                "Ưu tiên trả lời DỰA TRÊN tài liệu. Trích dẫn nguồn nếu có.\n"
                "═══════════════════════════\n\n" + rag_ctx
            )

        clean_history = list(history)
        while clean_history and clean_history[-1]["role"] == "assistant":
            clean_history.pop()

        user_max = await get_user_max_tokens(user_id) if user_id else 800
        max_tok  = user_max if user_max != 800 else 1500

        resp = await mistral_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system}] + clean_history,
            temperature=0.6,
            max_tokens=max_tok,
        )
        result = (resp.choices[0].message.content or "").strip()
        if _want_detail:
            if _want_detail:
            return result if result else "⚠️ Model không trả về phản hồi. Vui lòng thử lại."
        return strip_markdown(result) if result else "⚠️ Model không trả về phản hồi. Vui lòng thử lại."
        return strip_markdown(result) if result else "⚠️ Model không trả về phản hồi. Vui lòng thử lại."

    except Exception as e:
        logger.error(f"call_mistral_text error [{model_id}]: {e}")
        return f"⚠️ Lỗi: {str(e)[:200]}"


async def call_mistral_vision(
    image_b64: str,
    user_id: str | None = None,
    user_prompt: str = "",
) -> str:
    """Vision via Groq Llama-4-Scout. Falls back to Vietnamese prompt if user_prompt empty."""
    vision_model = MODEL_REGISTRY.get(VISION_MODEL_KEY, {}).get(
        "model_id", "meta-llama/llama-4-scout-17b-16e-instruct"
    )

    # Detect language from DB
    lang = "vi"
    if user_id:
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                async with db.execute(
                    "SELECT language FROM user_settings WHERE user_id=?", (user_id,)
                ) as cur:
                    row = await cur.fetchone()
                    if row:
                        lang = row[0]
        except Exception:
            pass

    if not user_prompt:
        if lang == "zh-TW":
            user_prompt = "請詳細描述這張圖片的所有內容。若有文字請完整擷取，使用繁體中文。"
        else:
            user_prompt = "Hãy mô tả chi tiết hình ảnh. Nếu có chữ, hãy trích xuất đầy đủ."

    system = await build_system_prompt(user_id, VISION_MODEL_KEY) if user_id else get_system_prompt(VISION_MODEL_KEY)

    try:
        resp = await groq_client.chat.completions.create(
            model=vision_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]},
            ],
            max_tokens=800,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ 視覺錯誤: {str(e)[:150]}"


async def call_groq_whisper(audio_bytes: bytes) -> str:
    try:
        result = await groq_client.audio.transcriptions.create(
            file=("audio.m4a", audio_bytes),
            model=WHISPER_MODEL,
        )
        return result.text
    except Exception as e:
        return f"⚠️ Whisper 錯誤: {str(e)[:150]}"


async def clean_transcript(transcript: str) -> str:
    try:
        resp = await mistral_client.chat.completions.create(
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
