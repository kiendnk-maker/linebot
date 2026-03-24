"""
tests/test_model_routing.py — Test LLM model routing logic in llm_core.py.

Tests resolve_model() which returns (model_key, model_id) based on query
content and user preferences — without calling any API.

NOTE: llm_core.DEFAULT_MODEL_KEY = "large". Routing only kicks in when the
user is on the default model. Explicit non-default choice is returned as-is.
"""
import pytest
from unittest.mock import AsyncMock, patch


async def _resolve(text: str, user_model: str = "large") -> tuple[str, str]:
    """Call resolve_model with mocked user settings and classifier."""
    with (
        patch("llm_core.get_user_model",     new=AsyncMock(return_value=user_model)),
        patch("llm_core.classify_query",     new=AsyncMock(return_value="small")),
    ):
        from llm_core import resolve_model
        return await resolve_model("test_user", text)


# ── User preference overrides routing ────────────────────────────────────────

@pytest.mark.asyncio
async def test_explicit_small_model_is_respected():
    """If user explicitly set 'small' (non-default), routing is skipped."""
    key, model_id = await _resolve("hello", user_model="small")
    assert key == "small"
    assert model_id == "mistral-small-latest"


@pytest.mark.asyncio
async def test_explicit_coder_model_is_respected():
    """If user explicitly set 'coder', routing is skipped."""
    key, _ = await _resolve("hello world", user_model="coder")
    assert key == "coder"


# ── Content-based routing (user is on default 'large') ───────────────────────

@pytest.mark.asyncio
async def test_code_query_routes_to_coder():
    """'def fibonacci():' should route to coder (Codestral)."""
    key, model_id = await _resolve("def fibonacci(): viết hàm này cho tôi", user_model="large")
    assert key == "coder"
    assert model_id == "codestral-latest"


@pytest.mark.asyncio
async def test_math_keyword_routes_to_reason():
    """'tính toán xác suất' should route to reason (Magistral)."""
    key, model_id = await _resolve("tính toán xác suất của 3 sự kiện độc lập", user_model="large")
    assert key == "reason"
    assert model_id == "magistral-medium-latest"


@pytest.mark.asyncio
async def test_long_text_routes_to_large():
    """Text >500 chars without '?' routes to 'large' for summarization."""
    long_text = "đây là một đoạn văn rất dài " * 30  # >500 chars, no ?
    key, _ = await _resolve(long_text, user_model="large")
    assert key == "large"


@pytest.mark.asyncio
async def test_simple_greeting_falls_to_classifier():
    """Simple greeting with no signals uses classifier result."""
    key, _ = await _resolve("xin chào, bạn khỏe không?", user_model="large")
    # Classifier mocked to return "small"
    assert key == "small"


@pytest.mark.asyncio
async def test_python_keyword_routes_to_coder():
    key, _ = await _resolve("python code to sort a list", user_model="large")
    assert key == "coder"


# ── Model registry completeness ───────────────────────────────────────────────

def test_llm_core_registry_has_required_keys():
    """llm_core.MODEL_REGISTRY must have all required fields."""
    from llm_core import MODEL_REGISTRY
    for key, cfg in MODEL_REGISTRY.items():
        assert "model_id" in cfg, f"Missing model_id for {key}"
        assert "display"  in cfg, f"Missing display for {key}"
        assert "tier"     in cfg, f"Missing tier for {key}"
        assert "note"     in cfg, f"Missing note for {key}"


def test_default_model_key_exists_in_llm_core_registry():
    from llm_core import MODEL_REGISTRY, DEFAULT_MODEL_KEY
    assert DEFAULT_MODEL_KEY in MODEL_REGISTRY


def test_llama8b_removed_from_llm_core_registry():
    """llama8b was retired — must not appear in llm_core.MODEL_REGISTRY."""
    from llm_core import MODEL_REGISTRY
    assert "llama8b" not in MODEL_REGISTRY


def test_prompts_registry_keys_valid():
    """prompts.py MODEL_REGISTRY — required fields present."""
    from prompts import MODEL_REGISTRY
    for key, cfg in MODEL_REGISTRY.items():
        assert "model_id" in cfg, f"Missing model_id for {key}"
        assert "display"  in cfg, f"Missing display for {key}"


def test_prompts_llama8b_removed():
    from prompts import MODEL_REGISTRY
    assert "llama8b" not in MODEL_REGISTRY


def test_prompts_classifier_key_exists():
    from prompts import MODEL_REGISTRY, CLASSIFIER_MODEL_KEY
    assert CLASSIFIER_MODEL_KEY in MODEL_REGISTRY
