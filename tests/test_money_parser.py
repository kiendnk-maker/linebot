"""
tests/test_money_parser.py — Test money tracker parsing and command routing.

Uses mock DB so no real SQLite needed.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager


def _mock_db():
    """Return an async context manager yielding a mock DB.

    Supports both:
      cur = await db.execute(...)        — for INSERT/DELETE/UPDATE
      async with db.execute(...) as cur: — for SELECT queries
    """
    mock_cursor = MagicMock()
    mock_cursor.lastrowid = 42
    mock_cursor.rowcount = 1
    mock_cursor.fetchall = AsyncMock(return_value=[])
    mock_cursor.fetchone = AsyncMock(return_value=None)

    class _AsyncCursorCtx:
        """Supports both `await` (returns cursor) and `async with` (yields cursor)."""
        def __init__(self): pass
        def __await__(self):
            async def _inner(): return mock_cursor
            return _inner().__await__()
        async def __aenter__(self): return mock_cursor
        async def __aexit__(self, *_): pass

    mock_db = MagicMock()
    mock_db.execute = MagicMock(return_value=_AsyncCursorCtx())
    mock_db.commit = AsyncMock()

    @asynccontextmanager
    async def _ctx(*a, **kw):
        yield mock_db

    return _ctx


async def _call_mn(arg: str) -> str:
    with patch("aiosqlite.connect", _mock_db()):
        from money_tracker import handle_money_command
        return await handle_money_command("test_user", arg)


# ── Add transaction parsing ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_add_expense():
    result = await _call_mn("Ăn trưa, -50")
    assert "42" in result          # new_id = 42
    assert "Ăn trưa" in result
    assert "-50" in result


@pytest.mark.asyncio
async def test_add_income():
    result = await _call_mn("Lương tháng, +5000000")
    assert "Lương tháng" in result
    assert "+5,000,000" in result or "5000000" in result


@pytest.mark.asyncio
async def test_add_bare_positive_is_income():
    result = await _call_mn("Thưởng, 2000")
    assert "Thưởng" in result
    assert "+2,000" in result or "2000" in result


@pytest.mark.asyncio
async def test_add_missing_comma_returns_error():
    result = await _call_mn("Ăn trưa -50")
    assert "dấu phẩy" in result or "⚠️" in result


@pytest.mark.asyncio
async def test_add_invalid_amount_returns_error():
    result = await _call_mn("Cà phê, abc")
    assert "⚠️" in result or "không hợp lệ" in result.lower()


@pytest.mark.asyncio
async def test_empty_arg_returns_help():
    result = await _call_mn("")
    assert "HDSD" in result or "/mn" in result


# ── List command ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_today_empty():
    result = await _call_mn("list today")
    assert "Không có" in result or "📭" in result


@pytest.mark.asyncio
async def test_list_week_empty():
    result = await _call_mn("list week")
    assert "Không có" in result or "📭" in result


@pytest.mark.asyncio
async def test_list_all_empty():
    result = await _call_mn("list all")
    assert "Không có" in result or "📭" in result


# ── Delete command ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rm_existing_record():
    result = await _call_mn("rm 5")
    # rowcount=1 means deleted
    assert "xoá" in result.lower() or "✅" in result


@pytest.mark.asyncio
async def test_rm_missing_id_returns_error():
    result = await _call_mn("rm")
    assert "⚠️" in result


# ── Edit command ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_edit_basic():
    result = await _call_mn("edit 3, Cà phê, -25")
    assert result is not None


@pytest.mark.asyncio
async def test_edit_missing_fields_returns_error():
    result = await _call_mn("edit 3, Chỉ một field")
    assert "⚠️" in result
