"""
tests/test_command_routing.py — Verify the dispatcher routes each /command
to the correct handler and returns a non-None string.

All external calls (LLM APIs, LINE API, DB) are mocked.
"""
import pytest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch, MagicMock


# ── helpers ──────────────────────────────────────────────────────────────────

_PATCHES = [
    ("command_handler.handle_workspace_command", AsyncMock(return_value=None)),
    ("commands.ai.run_pro_workflow",             AsyncMock(return_value="pro_ok")),
    ("commands.ai.run_agentic_loop",             AsyncMock(return_value="agent_ok")),
    ("commands.ai.run_multi_agent_workflow",     AsyncMock(return_value="coder_ok")),
    ("commands.ai.run_debate",                   AsyncMock(return_value="debate_ok")),
    ("commands.data.handle_money_command",       AsyncMock(return_value="mn_ok")),
    ("commands.data.list_rag_docs",              AsyncMock(return_value=[])),
    ("commands.data.delete_rag_doc",             AsyncMock(return_value=True)),
    ("commands.data.clear_rag_docs",             AsyncMock(return_value=0)),
    ("commands.settings.get_usage_report",       AsyncMock(return_value="usage_ok")),
    ("commands.settings.get_user_model",         AsyncMock(return_value="small")),
    ("commands.settings.set_user_model",         AsyncMock()),
    ("commands.settings.get_user_max_tokens",    AsyncMock(return_value=1000)),
    ("commands.settings.set_user_max_tokens",    AsyncMock()),
    ("commands.settings.get_user_profile",       AsyncMock(return_value=None)),
    ("commands.settings.save_user_profile",      AsyncMock()),
    ("commands.settings.get_reminders",          AsyncMock(return_value=[])),
    ("commands.settings.cancel_reminder",        AsyncMock(return_value=True)),
    ("commands.settings.save_reminder",          AsyncMock(return_value=1)),
    ("aiosqlite.connect",                        MagicMock()),
]


async def _dispatch(text: str, user_id: str = "test_user") -> str | None:
    """Call handle_command with a fully mocked environment.

    Uses ExitStack to apply many patches without deep nesting — deeply nested
    `with (p1, p2, ..., pN)` blocks trigger a CPython 3.12 compiler segfault.
    """
    with ExitStack() as stack:
        for target, mock in _PATCHES:
            stack.enter_context(patch(target, new=mock))
        from command_handler import handle_command
        return await handle_command(user_id, text)


# ── AI workflow commands ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pro_routes_to_ai():
    result = await _dispatch("/pro Phân tích học thạc sĩ")
    assert result == "pro_ok"


@pytest.mark.asyncio
async def test_pro_without_arg_returns_help():
    result = await _dispatch("/pro")
    assert result is not None and "Vui lòng" in result


@pytest.mark.asyncio
async def test_agent_routes_to_ai():
    result = await _dispatch("/agent tính 2+2")
    assert result == "agent_ok"


@pytest.mark.asyncio
async def test_coder_routes_to_ai():
    result = await _dispatch("/coder viết hàm fibonacci")
    assert result == "coder_ok"


@pytest.mark.asyncio
async def test_debate_routes_to_ai():
    result = await _dispatch("/debate AI có nguy hiểm không?")
    assert result == "debate_ok"


@pytest.mark.asyncio
async def test_debate_without_arg_returns_help():
    result = await _dispatch("/debate")
    assert result is not None and "debate" in result.lower()


# ── Data commands ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mn_routes_to_data():
    result = await _dispatch("/mn Ăn trưa, -50")
    assert result == "mn_ok"


@pytest.mark.asyncio
async def test_rag_list_returns_empty_message():
    result = await _dispatch("/rag list")
    assert result is not None and "Chưa có" in result


@pytest.mark.asyncio
async def test_rag_clear():
    result = await _dispatch("/rag clear")
    assert result is not None and "xoá" in result.lower()


# ── Settings commands ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_usage_routes_to_settings():
    result = await _dispatch("/usage")
    assert result == "usage_ok"


@pytest.mark.asyncio
async def test_vi_language_switch():
    result = await _dispatch("/vi")
    assert result is not None and "Tiếng Việt" in result


@pytest.mark.asyncio
async def test_tw_language_switch():
    result = await _dispatch("/tw")
    assert result is not None and "繁體中文" in result


@pytest.mark.asyncio
async def test_model_no_arg_shows_current():
    result = await _dispatch("/model")
    assert result is not None and "small" in result.lower() or "model" in result.lower()


@pytest.mark.asyncio
async def test_long_sets_token_limit():
    result = await _dispatch("/long 4000")
    assert result is not None and "4000" in result


@pytest.mark.asyncio
async def test_short_sets_800_tokens():
    result = await _dispatch("/short")
    assert result is not None and "800" in result


@pytest.mark.asyncio
async def test_tokens_shows_current():
    result = await _dispatch("/tokens")
    assert result is not None and "1000" in result


@pytest.mark.asyncio
async def test_profile_no_arg_shows_empty():
    result = await _dispatch("/profile")
    assert result is not None and "Chưa có" in result


@pytest.mark.asyncio
async def test_remind_list_empty():
    result = await _dispatch("/remind list")
    assert result is not None and "Không có" in result


# ── Unknown command ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unknown_command_returns_error():
    result = await _dispatch("/xyzunknown")
    assert result is not None and "無效" in result


# ── Non-command returns None ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_plain_text_returns_none():
    result = await _dispatch("hello world")
    assert result is None
