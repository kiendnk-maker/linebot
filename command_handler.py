import re
import aiosqlite
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Taipei")

import main
from llm_core import MODEL_REGISTRY, DEFAULT_MODEL_KEY
from database import DB_PATH, get_user_model, set_user_model, get_user_max_tokens, set_user_max_tokens, get_user_profile, save_user_profile, get_reminders, cancel_reminder, save_reminder
from google_workspace import handle_workspace_command
from rag_core import process_file_upload, list_rag_docs, delete_rag_doc, clear_rag_docs
from agents_workflow import run_pro_workflow, run_agentic_loop, run_multi_agent_workflow

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
    return "\n".join([
        "📋 可用模型列表\n",
        "── Production ──", *prod_lines,
        "\n── Preview ──",  *prev_lines,
        "\n─────────────────",
        "切換模型：/model <名稱>",
        "目前模型：/model",
        "自動模式：/auto",
        "清除紀錄：/clear",
        "提醒清單：/remind list",
    ])

async def handle_command(user_id: str, text: str) -> str | None:
    if not text.startswith("/"):
        return None

    parts = text[1:].strip().split(maxsplit=1)
    cmd   = parts[0].lower()
    arg   = parts[1].strip() if len(parts) > 1 else ""
    # --- CHUYỂN HƯỚNG SANG GOOGLE WORKSPACE ---
    ws_reply = await handle_workspace_command(cmd, arg, user_id)
    if ws_reply:
        return ws_reply

    # ── AI AGENTS & WORKFLOWS ──────────────────────────────────────────────
    if cmd == "pro":
        if not arg:
            return "⚠️ Vui lòng nhập yêu cầu phức tạp. Ví dụ: /pro Phân tích ưu nhược điểm của việc học Thạc sĩ tại Đài Loan"
        return await run_pro_workflow(user_id, arg)


    if cmd == "agent":
        if not arg:
            return "⚠️ Vui lòng nhập nhiệm vụ. Ví dụ: /agent Bây giờ là mấy giờ? Tính giúp tôi 12345 * 6789"
        return await run_agentic_loop(user_id, arg)

    if cmd == "dev":
        if not arg:
            return "⚠️ Vui lòng nhập yêu cầu. Ví dụ: /dev Viết hàm Python tính dãy Fibonacci"
        return await run_multi_agent_workflow(user_id, arg)

    # ── PREFERENCES & SYSTEM ───────────────────────────────────────────────
    if cmd == "vi":
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("INSERT INTO user_settings (user_id, model_key, language) VALUES (?, 'llama70b', ?) ON CONFLICT(user_id) DO UPDATE SET language=?", (user_id, "vi", "vi"))
            await db.commit()
        return "🇻🇳 Đã chuyển đổi ngôn ngữ sang Tiếng Việt. Từ giờ tôi sẽ trả lời bạn 100% bằng Tiếng Việt."

    if cmd == "tw":
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("INSERT INTO user_settings (user_id, model_key, language) VALUES (?, 'llama70b', ?) ON CONFLICT(user_id) DO UPDATE SET language=?", (user_id, "tw", "tw"))
            await db.commit()
        return "🇹🇼 已將語言切換為繁體中文。從現在起，我將只使用繁體中文回答。"

    if cmd == "clear":
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
            await db.execute("DELETE FROM summary WHERE user_id = ?", (user_id,))
            await db.commit()
        return "🗑 對話記錄已清除。"

    # ── MONEY TRACKER ──────────────────────────────────────────────────────
    if cmd == "mn":
        from money_tracker import handle_money_command
        return await handle_money_command(user_id, arg)

    # ── MODELS ─────────────────────────────────────────────────────────────
    if cmd == "models":
        return _models_list_text()

    # ── AUTO ───────────────────────────────────────────────────────────────
    if cmd == "auto":
        await set_user_model(user_id, DEFAULT_MODEL_KEY)
        return "🤖 已切換至自動選擇模型模式。"

    # ── MODEL ──────────────────────────────────────────────────────────────
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
        return f"✅ 已切換至 {MODEL_REGISTRY[target]['display']}。\n輸入 /auto 返回自動模式。"

    # ── LONG / SHORT / TOKENS ──────────────────────────────────────────────
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

    # ── PROFILE ────────────────────────────────────────────────────────────
    if cmd == "profile":
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

    # ── REMIND ─────────────────────────────────────────────────────────────
    if cmd == "remind":
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
                ok  = await cancel_reminder(user_id, rid)
                return f"✅ Đã huỷ nhắc #{rid}." if ok else f"❌ Không tìm thấy nhắc #{rid}."
            except ValueError:
                pass

        # /remind HH:MM [daily|weekly|monthly] [tối|chiều|sáng] [ngày DD/MM[/YYYY]] nội dung
        time_match = re.match(r"(\d{1,2}):(\d{2})\s+(.*)", arg)
        if time_match:
            hour   = int(time_match[1])
            minute = int(time_match[2])
            rest   = time_match[3].strip()

            # Repeat keyword
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

            # AM/PM detection from full arg string (SPEC §8)
            arg_lower = arg.lower()
            is_pm = any(kw in arg_lower for kw in ("tối", "chiều", "pm", "evening", "afternoon", "tonight"))
            is_am = any(kw in arg_lower for kw in ("sáng", "am", "morning", "trưa"))
            if is_pm and not is_am and hour < 12:
                hour += 12
            elif is_am and hour == 12:
                hour = 0

            # Date detection: regex DD/MM or DD-MM in rest string
            now_dt  = datetime.now(TZ)
            date_match = re.search(r"ngày\s+(\d{1,2})[/-](\d{1,2})(?:[/-](\d{4}))?", rest, re.IGNORECASE)
            if date_match:
                dd  = int(date_match[1])
                mo  = int(date_match[2])
                yy  = int(date_match[3]) if date_match[3] else now_dt.year
                # Remove date phrase from rest
                rest = rest[:date_match.start()].strip() + " " + rest[date_match.end():].strip()
                rest = rest.strip()
                fire_dt = datetime(yy, mo, dd, hour, minute, tzinfo=TZ)
            else:
                fire_dt = now_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if fire_dt <= now_dt:
                    fire_dt += timedelta(days=1)

            rid        = await save_reminder(user_id, rest, int(fire_dt.timestamp()), repeat)
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

    # ── RAG COMMANDS ───────────────────────────────────────────────────────
    if cmd == "audio":
        parts = arg.split()
        if len(parts) != 2: return "⚠️ Sai cú pháp. Dùng: /audio <id> <1|2|3>"
        audio_id, choice = parts[0], parts[1]
        if choice not in ("1", "2", "3"): return "⚠️ Chọn 1, 2 hoặc 3."

        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute("SELECT transcript, filename FROM audio_cache WHERE id = ? AND user_id = ?", (int(audio_id), user_id)) as cur:
                row = await cur.fetchone()
        if not row: return "❌ Không tìm thấy bản ghi âm thanh."

        transcript, filename = row
        summary = ""
        rag_msg = ""

        # --- Progress feedback ---
        try:
            from linebot.v3.messaging import (
                Configuration, AsyncApiClient, AsyncMessagingApi,
                PushMessageRequest, TextMessage
            )
            import os
            _cfg = Configuration(access_token=os.environ["LINE_CHANNEL_ACCESS_TOKEN"])
            async with AsyncApiClient(_cfg) as _api_client:
                _line = AsyncMessagingApi(_api_client)
                label = {"1": "phân tích", "2": "lưu RAG", "3": "phân tích + lưu RAG"}[choice]
                await _line.push_message(PushMessageRequest(
                    to=user_id, messages=[TextMessage(text=f"⏳ Đang {label}...")]
                ))
        except Exception:
            pass

        # --- Summarize (choice 1 or 3) ---
        if choice in ["1", "3"]:
            agent_prompt = f"""Bạn là một trợ lý phân tích dữ liệu và thư ký chuyên nghiệp, có khả năng đọc hiểu, tổng hợp thông tin xuất sắc.

Nhiệm vụ: Đọc bản ghi âm (transcript) được cung cấp ở phần cuối và tạo ra một bản tóm tắt toàn diện, cấu trúc rõ ràng. Bản ghi âm này có thể chứa các từ ngữ lặp lại, câu nói ngập ngừng hoặc lỗi diễn đạt do văn phong nói. Hãy bỏ qua những phần "nhiễu" này và chỉ tập trung trích xuất nội dung cốt lõi.

Yêu cầu định dạng đầu ra:
Vui lòng trình bày bản tóm tắt bằng Markdown với các mục sau:
- Tổng quan (Executive Summary): 2-3 câu khái quát toàn bộ chủ đề và mục đích chính của bản ghi âm.
- Các điểm nhấn chính (Key Takeaways): Liệt kê 3-5 luận điểm hoặc ý tưởng quan trọng nhất được thảo luận (sử dụng gạch đầu dòng).
- Thông tin chi tiết (Important Details): Ghi chú lại các số liệu cụ thể, định nghĩa khái niệm, hoặc các lập luận quan trọng hỗ trợ cho các điểm nhấn chính ở trên.
- Hành động tiếp theo/Kết luận (Action Items/Conclusion): Liệt kê các công việc cần làm tiếp theo (ai làm gì, bao giờ xong - nếu có) hoặc kết luận cuối cùng của cuộc trò chuyện.

Nguyên tắc nghiêm ngặt:
1. Giữ giọng văn khách quan, học thuật và chuyên nghiệp.
2. Tuyệt đối không tự suy diễn hoặc bịa đặt thêm thông tin không có trong bản ghi.
3. Trình bày ngắn gọn, tránh dài dòng.

Nội dung bản ghi âm cần tóm tắt:
{transcript[:15000]}"""

            # Đẩy thẳng qua hệ thống Agent Tự Trị
            summary = await run_agentic_loop(user_id, agent_prompt)

        # --- RAG (choice 2 or 3) ---
        if choice in ["2", "3"]:
            rag_msg = await process_file_upload(user_id, transcript.encode('utf-8'), filename)

        # --- Build TXT content with metadata ---
        from datetime import datetime
        from zoneinfo import ZoneInfo
        _now = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M")
        _word_count = len(transcript.split())

        sections = [f"📄 {filename}", f"📅 {_now} | ~{_word_count} từ", ""]
        if summary:
            sections.append("═══ PHÂN TÍCH ═══")
            sections.append(summary)
            sections.append("")
        sections.append("═══ NỘI DUNG GỐC ═══")
        sections.append(transcript)
        content_to_save = "\n".join(sections)

        # --- Upload to Google Drive ---
        link = ""
        try:
            import httpx, json
            from google_workspace import get_google_access_token
            token = await get_google_access_token(user_id)
            if token:
                metadata = {"name": filename, "mimeType": "text/plain"}
                files = {
                    "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
                    "file": (filename, content_to_save.encode("utf-8"), "text/plain")
                }
                async with httpx.AsyncClient() as client:
                    headers = {"Authorization": f"Bearer {token}"}
                    resp = await client.post("https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart", headers=headers, files=files, timeout=20.0)
                    if resp.status_code == 200:
                        file_id = resp.json().get("id")
                        link = f"https://drive.google.com/file/d/{file_id}/view"
                    else:
                        link = f"(Lỗi upload: HTTP {resp.status_code})"
            else:
                link = "(Gửi /login để liên kết Google Drive)"
        except Exception as e:
            link = f"(Lỗi: {str(e)[:40]})"

        # --- Format output message ---
        out = f"✅ {filename}\n"
        if summary: out += f"\n📝 PHÂN TÍCH:\n{summary}\n"
        if rag_msg: out += f"\n📚 {rag_msg}\n"
        if link: out += f"\n🔗 {link}"

        return out

    if cmd == "rag":
        sub = arg.lower().split(maxsplit=1)
        sub_cmd = sub[0] if sub else ""
        sub_arg = sub[1].strip() if len(sub) > 1 else ""

        if sub_cmd == "list":
            docs = await list_rag_docs(user_id)
            if not docs:
                return "📂 Chưa có tài liệu nào. Gửi file .pdf .txt .docx để thêm."
            lines = [f"📂 Tài liệu của bạn ({len(docs)} file):\n"]
            for d in docs:
                dt = datetime.fromtimestamp(d["uploaded_at"], tz=TZ).strftime("%d/%m/%Y %H:%M")
                lines.append(f"• {d['filename']}\n  {d['chunk_count']} chunks — {dt}")
            return "\n".join(lines)

        if sub_cmd == "delete" and sub_arg:
            ok = await delete_rag_doc(user_id, sub_arg)
            return (
                f"✅ Đã xoá {sub_arg} khỏi knowledge base."
                if ok
                else f"❌ Không tìm thấy file: {sub_arg}\nDùng /rag list để xem danh sách."
            )

        if sub_cmd == "clear":
            n = await clear_rag_docs(user_id)
            return f"🗑 Đã xoá {n} tài liệu khỏi knowledge base."

        if sub_cmd == "off":
            main._rag_disabled.add(user_id)
            return "🔕 RAG đã tắt cho session này. Dùng /rag on để bật lại."

        if sub_cmd == "on":
            main._rag_disabled.discard(user_id)
            return "🔔 RAG đã bật."

        return (
            "📚 Lệnh RAG:\n"
            "/rag list — xem danh sách file\n"
            "/rag delete <tên file> — xoá file\n"
            "/rag clear — xoá tất cả\n"
            "/rag off — tắt RAG\n"
            "/rag on — bật RAG"
        )

    # ── MODEL SHORTCUT ─────────────────────────────────────────────────────
    if cmd in MODEL_REGISTRY:
        await set_user_model(user_id, cmd)
        cfg = MODEL_REGISTRY[cmd]
        if arg:
            answer = await main.call_groq_text(
                [{"role": "user", "content": arg}],
                cfg["model_id"],
                model_key=cmd,
            )
            return f"[{cfg['display']}]\n{answer}"
        return f"✅ 已切換至 {cfg['display']}。\n輸入 /auto 返回自動模式。"

    return f"❓ 指令 /{cmd} 無效。請輸入 /models 查看。"
