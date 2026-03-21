"""
llm_core.py — Dual Provider: Mistral + Groq
Mistral Small 4 (2603) = unified: chat + reasoning + vision + code
Mistral Large 3 = heavy writing/translation
Groq = FREE Qwen3 reasoning + Whisper STT + LLaMA 4 vision backup
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
from prompts import get_system_prompt
from database import (
    DB_PATH, get_user_profile, get_user_model, get_user_max_tokens,
    count_history, get_history_raw, get_summary, save_summary,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# DUAL CLIENT
# ═══════════════════════════════════════════════════════════════
mistral_client = AsyncOpenAI(
    api_key=os.environ.get("MISTRAL_API_KEY", ""),
    base_url="https://api.mistral.ai/v1",
)
groq_client = AsyncOpenAI(
    api_key=os.environ.get("GROQ_API_KEY", ""),
    base_url="https://api.groq.com/openai/v1",
)

WHISPER_MODEL = "whisper-large-v3-turbo"
SUMMARY_TRIGGER = 20

# ═══════════════════════════════════════════════════════════════
# MODEL REGISTRY
# Small 4 (2603) = Magistral + Pixtral + Devstral unified
# reasoning_effort: "none" = fast chat, "high" = deep thinking
# ═══════════════════════════════════════════════════════════════
MODEL_REGISTRY: dict[str, dict] = {
    # ── Mistral (paid but cheap) ───────────────────────────────
    "small": {
        "model_id": "mistral-small-latest",
        "provider": "mistral",
        "type": "text",
        "tier": "production",
        "display": "Small 4 (chat)",
        "ctx": 262_144,
        "note": "$0.15/1M — chat nhanh, classifier",
        "reasoning_effort": "none",
    },
    "large": {
        "model_id": "mistral-large-latest",
        "provider": "mistral",
        "type": "text",
        "tier": "production",
        "display": "Large 3 (viet/dich)",
        "ctx": 131_072,
        "note": "$2/1M — viet dai, dich thuat, da ngon ngu",
        "reasoning_effort": None,
    },
    "coder": {
        "model_id": "mistral-small-latest",
        "provider": "mistral",
        "type": "text",
        "tier": "production",
        "display": "Small 4 (code)",
        "ctx": 262_144,
        "note": "$0.15/1M — Small 4 = Devstral mode",
        "reasoning_effort": "high",
    },
    "vision": {
        "model_id": "mistral-small-latest",
        "provider": "mistral",
        "type": "vision",
        "tier": "production",
        "display": "Small 4 (vision)",
        "ctx": 262_144,
        "note": "$0.15/1M — Small 4 = Pixtral mode",
        "reasoning_effort": "none",
    },
    "think": {
        "model_id": "mistral-small-latest",
        "provider": "mistral",
        "type": "reasoning",
        "tier": "production",
        "display": "Small 4 (reasoning)",
        "ctx": 262_144,
        "note": "$0.15/1M — Small 4 = Magistral mode",
        "reasoning_effort": "high",
    },
    # ── Groq (FREE) ───────────────────────────────────────────
    "reason": {
        "model_id": "qwen/qwen3-32b",
        "provider": "groq",
        "type": "reasoning",
        "tier": "production",
        "display": "Qwen3 32B (Groq)",
        "ctx": 131_072,
        "note": "FREE — reasoning, toan, logic",
        "reasoning_effort": None,
    },
    "llama": {
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "provider": "groq",
        "type": "vision",
        "tier": "preview",
        "display": "LLaMA 4 Scout (Groq)",
        "ctx": 131_072,
        "note": "FREE — vision backup",
        "reasoning_effort": None,
    },
}

DEFAULT_MODEL_KEY = "large"
VISION_MODEL_KEY = "vision"
CLASSIFIER_MODEL_KEY = "small"


def get_client(model_key: str) -> AsyncOpenAI:
    cfg = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY[DEFAULT_MODEL_KEY])
    return groq_client if cfg.get("provider") == "groq" else mistral_client


def get_client_by_model_id(model_id: str, model_key: str = "") -> AsyncOpenAI:
    if model_key:
        cfg = MODEL_REGISTRY.get(model_key)
        if cfg:
            return groq_client if cfg.get("provider") == "groq" else mistral_client
    for cfg in MODEL_REGISTRY.values():
        if cfg["model_id"] == model_id:
            return groq_client if cfg.get("provider") == "groq" else mistral_client
    return mistral_client


# ═══════════════════════════════════════════════════════════════
# SMART ROUTING
# ═══════════════════════════════════════════════════════════════
ROUTE_MAP: dict[str, str] = {
    "simple": "small",        # Small 4 none
    "creative": "large",      # Large 3
    "reasoning": "reason",    # Qwen3 FREE
    "hard": "think",          # Small 4 high
    "code": "coder",          # Small 4 high
    "search": "small",        # Small 4 none
}

_REALTIME_KEYWORDS = (
    "hom nay", "bay gio", "hien tai", "moi nhat", "today", "now", "latest",
    "gia ", "price", "ty gia", "thoi tiet", "weather", "tin tuc", "news",
    "今天", "現在", "最新", "價格", "新聞", "天氣",
)

_CLASSIFIER_PROMPT = """Classify into exactly one category. Reply ONLY one word.
- simple : greetings, chitchat, short factual
- creative : writing, translation, summarization, brainstorming
- reasoning : math, logic, step-by-step, comparison
- code : programming, debugging, code generation
- hard : complex multi-domain, deep thinking
- search : current events, prices, news, weather
Message: {message}"""


def _needs_realtime(text: str) -> bool:
    return any(kw in text.lower() for kw in _REALTIME_KEYWORDS)


async def classify_query(user_text: str) -> str:
    try:
        resp = await mistral_client.chat.completions.create(
            model=MODEL_REGISTRY["small"]["model_id"],
            messages=[{"role": "user", "content": _CLASSIFIER_PROMPT.format(message=user_text[:400])}],
            temperature=0.0,
            max_tokens=3,
        )
        cat = resp.choices[0].message.content.strip().lower()
        return ROUTE_MAP.get(cat, DEFAULT_MODEL_KEY)
    except Exception:
        return DEFAULT_MODEL_KEY


async def resolve_model(user_id: str, user_text: str) -> tuple[str, str]:
    current_key = await get_user_model(user_id)
    if current_key != DEFAULT_MODEL_KEY and current_key in MODEL_REGISTRY:
        return current_key, MODEL_REGISTRY[current_key]["model_id"]

    tl = user_text.lower()

    if _needs_realtime(user_text):
        return "small", MODEL_REGISTRY["small"]["model_id"]

    if any(kw in tl for kw in {"code", "python", "javascript", "viet ham", "debug", "function", "def ", "class ", "程式"}):
        return "coder", MODEL_REGISTRY["coder"]["model_id"]

    if any(kw in tl for kw in {"tinh", "cong", "tru", "nhan", "chia", "dao ham", "chung minh", "solve", "計算"}):
        return "reason", MODEL_REGISTRY["reason"]["model_id"]

    if len(user_text) > 500 and "?" not in user_text and "？" not in user_text:
        return "large", MODEL_REGISTRY["large"]["model_id"]

    if any(kw in tl for kw in {"dich", "translate", "tieng anh", "tieng trung", "翻譯"}):
        return "large", MODEL_REGISTRY["large"]["model_id"]

    routed_key = await classify_query(user_text)
    return routed_key, MODEL_REGISTRY[routed_key]["model_id"]


# ═══════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════
async def build_system_prompt(user_id: str, model_key: str) -> str:
    base = get_system_prompt(model_key)
    profile = await get_user_profile(user_id)
    if not profile:
        return base
    lines: list[str] = []
    if profile.get("name"): lines.append("用戶姓名：" + profile["name"])
    if profile.get("occupation"): lines.append("職業：" + profile["occupation"])
    if profile.get("learning"): lines.append("正在學習：" + profile["learning"])
    if profile.get("notes"): lines.append("備註：" + profile["notes"])
    return base + "\n\n【用戶資料】\n" + "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# AUTO SUMMARIZE
# ═══════════════════════════════════════════════════════════════
async def maybe_summarize(user_id: str) -> None:
    count = await count_history(user_id)
    if count % SUMMARY_TRIGGER != 0:
        return
    rows = await get_history_raw(user_id, limit=40)
    if not rows:
        return
    prev = await get_summary(user_id)
    hist = "\n".join(f"{m['role']}: {m['content']}" for m in rows)
    try:
        resp = await mistral_client.chat.completions.create(
            model=MODEL_REGISTRY["small"]["model_id"],
            messages=[{"role": "user", "content": f"Tom tat (150 tu):\nTruoc: {prev}\n\n{hist}"}],
            temperature=0.3,
            max_tokens=200,
        )
        await save_summary(user_id, resp.choices[0].message.content or "")
    except Exception:
        pass


async def get_history_with_summary(user_id: str) -> list[dict]:
    summary = await get_summary(user_id)
    recent = await get_history_raw(user_id, limit=5)
    if summary:
        return [
            {"role": "user", "content": f"[Tom tat truoc: {summary}]"},
            {"role": "assistant", "content": "Da hieu."},
            *recent,
        ]
    return recent


# ═══════════════════════════════════════════════════════════════
# STRIP MARKDOWN (LINE = plain text)
# ═══════════════════════════════════════════════════════════════
def strip_markdown(text: str) -> str:
    parts = re.split(r'(```.*?```)', text, flags=re.DOTALL)
    for i in range(0, len(parts), 2):
        p = parts[i]
        p = re.sub(r"<think>.*?</think>", "", p, flags=re.DOTALL)
        p = re.sub(r"\*\*(.*?)\*\*", r"\1", p)
        p = re.sub(r"^#{1,6}\s+", "", p, flags=re.MULTILINE)
        p = re.sub(r"^[-\*]\s+", "• ", p, flags=re.MULTILINE)
        p = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", p)
        p = re.sub(r"\n{3,}", "\n\n", p)
        parts[i] = p
    return "".join(parts).strip()


# ═══════════════════════════════════════════════════════════════
# CORE TEXT API
# Auto-selects client + injects reasoning_effort for Small 4
# ═══════════════════════════════════════════════════════════════
async def call_mistral_text(
    history: list[dict],
    model_id: str,
    model_key: str = DEFAULT_MODEL_KEY,
    user_id: str | None = None,
    rag_chunks: list[dict] | None = None,
) -> str:
    system = await build_system_prompt(user_id, model_key) if user_id else get_system_prompt(model_key)

    # RAG injection
    if rag_chunks:
        ctx = "\n\n".join(f"[Nguon: {c['filename']} #{c['chunk_index']}]\n{c['content']}" for c in rag_chunks)
        system += f"\n\n═══ TAI LIEU ═══\nUu tien tra loi tu tai lieu. Trich dan [Nguon: file]. Khong bia.\n═══════════════\n\n{ctx}"

    # Clean history
    clean = list(history)
    while clean and clean[-1]["role"] == "assistant":
        clean.pop()

    # Language forcer
    if user_id and clean and clean[-1]["role"] == "user":
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                async with db.execute("SELECT language FROM user_settings WHERE user_id = ?", (user_id,)) as cur:
                    row = await cur.fetchone()
                    lang = row[0] if row else "vi"
                    rule = "CRITICAL: Answer in Vietnamese." if lang == "vi" else "CRITICAL: Answer in Traditional Chinese (Taiwan)."
                    clean[-1]["content"] += f"\n\n[{rule}]"
        except Exception:
            pass

    # Select client
    client = get_client_by_model_id(model_id, model_key)

    # Build extra kwargs — reasoning_effort for Small 4
    extra: dict = {}
    cfg = MODEL_REGISTRY.get(model_key, {})
    re_val = cfg.get("reasoning_effort")
    if re_val and cfg.get("provider") == "mistral":
        extra["extra_body"] = {"reasoning_effort": re_val}

    # max_tokens
    user_max = await get_user_max_tokens(user_id) if user_id else 800
    max_tok = user_max if user_max != 800 else {
        "small": 1500, "large": 1500, "coder": 2000,
        "think": 2000, "reason": 2000, "vision": 800, "llama": 800,
    }.get(model_key, 800)

    try:
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system}] + clean,
            temperature=0.6,
            max_tokens=max_tok,
            **extra,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        err = str(e)
        # Fallback chain: Groq fail → Mistral Small, Mistral fail → Mistral Large
        fallback_id = MODEL_REGISTRY["large"]["model_id"] if client == mistral_client else MODEL_REGISTRY["small"]["model_id"]
        fallback_client = mistral_client
        try:
            resp = await fallback_client.chat.completions.create(
                model=fallback_id,
                messages=[{"role": "system", "content": system}] + clean,
                temperature=0.6,
                max_tokens=max_tok,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e2:
            return f"⚠️ Error: {str(e2)[:120]}"


# ═══════════════════════════════════════════════════════════════
# VISION — Small 4 (Pixtral mode) → Groq LLaMA 4 fallback
# ═══════════════════════════════════════════════════════════════
async def call_mistral_vision(image_b64: str) -> str:
    system = get_system_prompt(VISION_MODEL_KEY)
    msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "text", "text": "請詳細分析並描述這張圖片。若有文字請完整擷取。若有中文請使用繁體中文。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]},
    ]
    try:
        resp = await mistral_client.chat.completions.create(
            model="mistral-small-latest", messages=msg, max_tokens=1200,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        try:
            resp = await groq_client.chat.completions.create(
                model=MODEL_REGISTRY["llama"]["model_id"], messages=msg, max_tokens=800,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"⚠️ Vision error: {str(e)[:150]}"


# ═══════════════════════════════════════════════════════════════
# WHISPER — Groq FREE → Mistral fallback
# ═══════════════════════════════════════════════════════════════
async def call_groq_whisper(audio_bytes: bytes) -> str:
    try:
        r = await groq_client.audio.transcriptions.create(file=("audio.m4a", audio_bytes), model=WHISPER_MODEL)
        return r.text
    except Exception:
        try:
            r = await mistral_client.audio.transcriptions.create(file=("audio.m4a", audio_bytes), model=WHISPER_MODEL)
            return r.text
        except Exception as e:
            return f"⚠️ Whisper error: {str(e)[:150]}"


async def clean_transcript(transcript: str) -> str:
    try:
        resp = await mistral_client.chat.completions.create(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": f"Sua loi nghe nham, giu ngon ngu goc. Chi tra ve cau da sua.\n\nTranscript: {transcript}"}],
            temperature=0.0, max_tokens=300,
        )
        c = resp.choices[0].message.content.strip()
        return c if c else transcript
    except Exception:
        return transcript


# ═══════════════════════════════════════════════════════════════
# LINE SPLITTER
# ═══════════════════════════════════════════════════════════════
def _split_reply(reply: str) -> list[str]:
    chunks: list[str] = []
    while len(reply) > 4990:
        cut = reply.rfind(" ", 0, 4990)
        if cut == -1: cut = 4990
        chunks.append(reply[:cut])
        reply = reply[cut:].strip()
    if reply:
        chunks.append(reply)
    return chunks[:5]
