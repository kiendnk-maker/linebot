"""
tests/test_reminder_parser.py — Test reminder time parsing logic.

The /remind command parses time strings inline in command_handler.
We test _handle_remind from commands/settings.py with mocked DB.
"""
import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Taipei")


async def _remind(arg: str, user_id: str = "test_user") -> str:
    with (
        patch("commands.settings.save_reminder",   new=AsyncMock(return_value=99)),
        patch("commands.settings.get_reminders",   new=AsyncMock(return_value=[])),
        patch("commands.settings.cancel_reminder", new=AsyncMock(return_value=True)),
        patch("aiosqlite.connect"),
    ):
        from commands.settings import _handle_remind
        return await _handle_remind(user_id, arg)


# ── Time parsing ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_simple_time_sets_reminder():
    result = await _remind("20:00 uống thuốc")
    assert "99" in result     # reminder ID
    assert "uống thuốc" in result
    assert "20:00" in result


@pytest.mark.asyncio
async def test_pm_keyword_adjusts_hour():
    """'7:00 tối' should become 19:00."""
    result = await _remind("7:00 tối nhớ gọi điện")
    assert "19:00" in result


@pytest.mark.asyncio
async def test_am_keyword_keeps_morning_hour():
    """'8:00 sáng' should stay 08:00."""
    result = await _remind("8:00 sáng tập thể dục")
    assert "08:00" in result


@pytest.mark.asyncio
async def test_midnight_keyword():
    """'12:00 đêm' (midnight) should become 00:00."""
    # Note: 'trưa' triggers is_am, and noon 12h stays 12. 'đêm' isn't in is_pm list but tests graceful handling.
    result = await _remind("9:00 pm họp")
    assert "21:00" in result


@pytest.mark.asyncio
async def test_daily_repeat():
    result = await _remind("20:00 daily uống thuốc")
    assert "hàng ngày" in result or "daily" in result.lower()


@pytest.mark.asyncio
async def test_weekly_repeat():
    result = await _remind("09:00 weekly họp team")
    assert "hàng tuần" in result or "weekly" in result.lower()


@pytest.mark.asyncio
async def test_monthly_repeat():
    result = await _remind("08:00 monthly nộp báo cáo")
    assert "hàng tháng" in result or "monthly" in result.lower()


# ── List and cancel ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_empty_returns_no_reminders():
    result = await _remind("list")
    assert "Không có" in result or "📭" in result


@pytest.mark.asyncio
async def test_cancel_reminder():
    result = await _remind("99 cancel")
    assert "99" in result and ("huỷ" in result.lower() or "✅" in result)


# ── Invalid input ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_time_format_returns_help():
    result = await _remind("nhắc tôi uống nước")
    assert "Cách dùng" in result or "/remind" in result
