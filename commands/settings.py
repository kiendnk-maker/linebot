"""
commands/settings.py — User settings & system commands:
/vi, /tw, /usage, /nuke, /clear, /auto, /model, /models,
/long, /short, /tokens, /profile, /remind
"""
import re
import sqlite3
import aiosqlite
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from tracker_core import get_usage_report
from llm_core import MODEL_REGISTRY, DEFAULT_MODEL_KEY
from database import (
    DB_PATH, get_user_model, set_user_model,
    get_user_max_tokens, set_user_max_tokens,
    get_user_profile, save_user_profile,
    get_reminders, cancel_reminder, save_reminder,
)

TZ = ZoneInfo("Asia/Taipei")


async def handle_settings_command(user_id: str, cmd: str, arg: str) -> str | None:
    """Return response string or None if command not handled here."""

    if cmd == "vi":
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO user_settings (user_id, model_key, language) VALUES (?, 'large', ?) "
                "ON CONFLICT(user_id) DO UPDATE SET language=?",
                (user_id, "vi", "vi"),
            )
            await db.commit()
        return "🇻🇳 Đã chuyển đổi ngôn ngữ sang Tiếng Việt. Từ giờ tôi sẽ trả lời bạn 100% bằng Tiếng Việt."

    if cmd == "tw":
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO user_settings (user_id, model_key, language) VALUES (?, 'large', ?) "
                "ON CONFLICT(user_id) DO UPDATE SET language=?",
                (user_id, "tw", "tw"),
            )
            await db.commit()
        return "🇹🇼 已將語言切換為繁體中文。從現在起，我將只使用繁體中文回答。"

    if cmd == "usage":
        return await get_usage_report()

    if cmd == "nuke":
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cur.fetchall()
            dropped = 0
            for (t_name,) in tables:
                if any(x in t_name.lower() for x in ["rag", "chunk", "vector", "embed", "collection"]):
                    cur.execute(f"DROP TABLE IF EXISTS {t_name}")
                    dropped += 1
            conn.commit()
            conn.close()
            return f"💥 BOOM! Đã nổ tung {dropped} bảng Vector DB cũ! Hệ thống RAG đã sạch sẽ, hãy gửi file mới để test."
        except Exception as e:
            return f"⚠️ Lỗi kích nổ DB: {e}"

    if cmd == "clear":
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
            await db.execute("DELETE FROM summary WHERE user_id = ?", (user_id,))
            await db.commit()
        return "🗑 對話記錄已清除。"

    # Short commands
    if cmd == "a":
        await set_user_model(user_id, DEFAULT_MODEL_KEY)
        return "🤖 Đã chuyển sang tự động chọn model"
    
    if cmd == "auto":
        await set_user_model(user_id, DEFAULT_MODEL_KEY)
        return "🤖 已切換至自動選擇模型模式。"

    if cmd == "models":
        return _models_list_text()

    if cmd == "model":
        if not arg:
            key = await get_user_model(user_id)
            cfg = MODEL_REGISTRY[key]
            mode = "自動" if key == DEFAULT_MODEL_KEY else "手動"
            return (
                f"🤖 目前模型：{cfg['display']}\n"
                f"   模式：{mode}\n"
                f"   {cfg['note']}\n\n"
                "輸入 /models 查看全部。\n"
                "輸入 /auto 返回自動模式。"
            )
        target = arg.lower()
        if target not in MODEL_REGISTRY:
            return f"❌ /{target} 不存在。\n請輸入 /models 查看清單。"
        await set_user_model(user_id, target)
        return f"✅ 已切換至 {MODEL_REGISTRY[target]['display']}。\n輸入 /auto 返回自動模式。"

    # Short model commands: /m <number>
    if cmd == "m":
        if not arg:
            return "🔢 Gõ /m <số> (1-8) để chọn model nhanh"
        try:
            choice = int(arg)
            model_map = {
                1: "mistral_small",
                2: "mistral_medium", 
                3: "mistral_large",
                4: "codestral",
                5: "pixtral",
                6: "small",
                7: "large",
                8: "qwen3",
            }
            if choice in model_map:
                model_key = model_map[choice]
                await set_user_model(user_id, model_key)
                return f"✅ Đã chuyển sang {MODEL_REGISTRY[model_key]['display']}"
            else:
                return "❌ Chọn số 1-8 thôi nhé!"
        except ValueError:
            return "❌ Gõ /m <số> (1-8)"
    
    # Mistral AI model shortcuts
    if cmd == "mistral_small":
        await set_user_model(user_id, "mistral_small")
        return f"✅ 已切換至 {MODEL_REGISTRY['mistral_small']['display']}。"
    
    if cmd == "mistral_medium":
        await set_user_model(user_id, "mistral_medium")
        return f"✅ 已切換至 {MODEL_REGISTRY['mistral_medium']['display']}。"
    
    if cmd == "mistral_large":
        await set_user_model(user_id, "mistral_large")
        return f"✅ 已切換至 {MODEL_REGISTRY['mistral_large']['display']}。"
    
    if cmd == "codestral":
        await set_user_model(user_id, "codestral")
        return f"✅ 已切換至 {MODEL_REGISTRY['codestral']['display']}。"
    
    if cmd == "pixtral":
        await set_user_model(user_id, "pixtral")
        return f"✅ 已切換至 {MODEL_REGISTRY['pixtral']['display']}。"

    if cmd == "long":
        val = int(arg) if arg.isdigit() else 3000
        val = min(val, 6000)
        await set_user_max_tokens(user_id, val)
        return f"Chế độ trả lời dài: tối đa {val} tokens (~{val*4} ký tự)"

    if cmd == "short":
        await set_user_max_tokens(user_id, 800)
        return "Chế độ trả lời ngắn: 800 tokens (mặc định)"

    if cmd == "tokens":
        val = await get_user_max_tokens(user_id)
        return f"Max tokens hiện tại: {val} (~{val*4} ký tự)"

    if cmd == "profile":
        return await _handle_profile(user_id, arg)

    if cmd == "remind":
        return await _handle_remind(user_id, arg)

    return None


async def _handle_profile(user_id: str, arg: str) -> str:
    if arg == "clear":
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM user_profile WHERE user_id = ?", (user_id,))
            await db.commit()
        return "🗑 Đã xoá thông tin cá nhân."

    profile = await get_user_profile(user_id)
    if not arg:
        if not profile:
            return (
                "Chưa có thông tin cá nhân.\n"
                "Cập nhật:\n"
                "/profile name Tên bạn\n"
                "/profile job Nghề nghiệp\n"
                "/profile learning Tiếng Trung B1\n"
                "/profile note Ghi chú thêm"
            )
        lines = ["Thông tin của bạn:\n"]
        if profile.get("name"):       lines.append("Tên: "       + profile["name"])
        if profile.get("occupation"): lines.append("Nghề: "      + profile["occupation"])
        if profile.get("learning"):   lines.append("Đang học: "  + profile["learning"])
        if profile.get("notes"):      lines.append("Ghi chú: "   + profile["notes"])
        return "\n".join(lines)

    parts2 = arg.split(maxsplit=1)
    if len(parts2) < 2:
        return "Dùng: /profile name|job|learning|note <nội dung>"
    field, value = parts2[0].lower(), parts2[1]
    field_map = {"name": "name", "job": "occupation", "learning": "learning", "note": "notes"}
    if field not in field_map:
        return "Field hợp lệ: name, job, learning, note"
    await save_user_profile(user_id, **{field_map[field]: value})
    return f"Đã lưu {field}: {value}"


async def _handle_remind(user_id: str, arg: str) -> str:
    if arg == "list":
        reminders = await get_reminders(user_id)
        if not reminders:
            return "📭 Không có nhắc nhở nào đang chờ."
        lines = ["📋 Danh sách nhắc nhở:\n"]
        for r in reminders:
            dt = datetime.fromtimestamp(r["fire_at"], tz=TZ)
            label = {
                "daily":   " 🔁 hàng ngày",
                "weekly":  " 🔁 hàng tuần",
                "monthly": " 🔁 hàng tháng",
            }.get(r["repeat"] or "", "")
            lines.append(
                f"#{r['id']}{label}\n"
                f"  {r['message']}\n"
                f"  ⏰ {dt.strftime('%H:%M %d/%m/%Y')}"
            )
        return "\n".join(lines)

    # /remind <id> cancel
    parts2 = arg.split()
    if len(parts2) == 2 and parts2[1] == "cancel":
        try:
            rid = int(parts2[0])
            ok = await cancel_reminder(user_id, rid)
            return f"✅ Đã huỷ nhắc #{rid}." if ok else f"❌ Không tìm thấy nhắc #{rid}."
        except ValueError:
            pass

    # /remind HH:MM [repeat] [AM/PM keywords] [ngày DD/MM[/YYYY]] nội dung
    time_match = re.match(r"(\d{1,2}):(\d{2})\s+(.*)", arg)
    if time_match:
        hour   = int(time_match[1])
        minute = int(time_match[2])
        rest   = time_match[3].strip()

        repeat: str | None = None
        repeat_map = {
            "daily": "daily", "weekly": "weekly", "monthly": "monthly",
            "hàng ngày": "daily", "hàng tuần": "weekly", "hàng tháng": "monthly",
        }
        for kw, val in repeat_map.items():
            if rest.lower().startswith(kw):
                repeat = val
                rest   = rest[len(kw):].strip()
                break

        arg_lower = arg.lower()
        is_pm = any(kw in arg_lower for kw in ("tối", "chiều", "pm", "evening", "afternoon", "tonight"))
        is_am = any(kw in arg_lower for kw in ("sáng", "am", "morning", "trưa"))
        if is_pm and not is_am and hour < 12:
            hour += 12
        elif is_am and hour == 12:
            hour = 0

        now_dt = datetime.now(TZ)
        date_match = re.search(r"ngày\s+(\d{1,2})[/-](\d{1,2})(?:[/-](\d{4}))?", rest, re.IGNORECASE)
        if date_match:
            dd  = int(date_match[1])
            mo  = int(date_match[2])
            yy  = int(date_match[3]) if date_match[3] else now_dt.year
            rest = (rest[:date_match.start()] + " " + rest[date_match.end():]).strip()
            fire_dt = datetime(yy, mo, dd, hour, minute, tzinfo=TZ)
        else:
            fire_dt = now_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if fire_dt <= now_dt:
                fire_dt += timedelta(days=1)

        rid = await save_reminder(user_id, rest, int(fire_dt.timestamp()), repeat)
        repeat_str = {
            "daily":   " (lặp hàng ngày)",
            "weekly":  " (lặp hàng tuần)",
            "monthly": " (lặp hàng tháng)",
        }.get(repeat or "", "")
        return (
            f"⏰ Đã đặt nhắc #{rid}{repeat_str}\n"
            f"Nội dung: {rest}\n"
            f"Thời gian: {fire_dt.strftime('%H:%M %d/%m/%Y')}"
        )

    return (
        "❓ Cách dùng /remind:\n"
        "/remind list\n"
        "/remind 2 cancel\n"
        "/remind 20:00 uống thuốc\n"
        "/remind 20:00 daily uống thuốc\n"
        "/remind 09:00 weekly họp team"
    )


def _models_list_text() -> str:
    return (
        "📋 Con mèo ngốc 🐱 — LỆNH NHANH\n"
        "\n"
        "🔥 MODELS (gõ /m <số>)\n"
        "1🐿 Mistral Small | 2🦊 Medium | 3🦁 Large\n"
        "4💻 Codestral | 5👁 Pixtral\n"
        "6⚡ Llama 8B | 7🦙 Llama 70B | 8🧠 Qwen3\n"
        "\n"
        "🤖 AI MODES\n"
        "/a — Auto model | /p <câu hỏi> — Pro mode\n"
        "/d <câu hỏi> — Debate | /ag <task> — Agent\n"
        "/co <req> — Coder mode\n"
        "\n"
        "📧 GOOGLE\n"
        "/login | /ls | /mail <số>\n"
        "/cal | /block | /unblock\n"
        "\n"
        "📚 RAG\n"
        "Gửi file PDF/TXT → auto KB\n"
        "/rag list|clear|off|on\n"
        "\n"
        "👤 CÁ NHÂN\n"
        "/pro | /remind | /mn\n"
        "\n"
        "⚙ CÀI ĐẶT\n"
        "/vi | /tw | /long | /short\n"
        "/clear | /usage | /models"
    )
