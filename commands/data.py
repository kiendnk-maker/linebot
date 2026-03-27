"""
commands/data.py — Data management commands: /mn, /rag, /audio
"""
import os
import aiosqlite
from datetime import datetime
from zoneinfo import ZoneInfo

from database import DB_PATH, set_rag_disabled
from money_tracker import handle_money_command
from rag_core import process_file_upload, list_rag_docs, delete_rag_doc, clear_rag_docs
from agents_workflow import run_agentic_loop

TZ = ZoneInfo("Asia/Taipei")


async def handle_data_command(user_id: str, cmd: str, arg: str) -> str | None:
    """Return response string or None if command not handled here."""

    if cmd == "mn":
        return await handle_money_command(user_id, arg)

    if cmd == "export":
        if not arg:
            return "⚠️ Gõ /export <nội dung> để lưu vào Google Drive"
        # Extract filename if provided, otherwise use timestamp
        parts = arg.split("|", 1)
        if len(parts) == 2:
            content, filename = parts[0].strip(), parts[1].strip()
        else:
            content, filename = arg.strip(), f"export_{datetime.now(TZ).strftime('%Y%m%d_%H%M%S')}.txt"
        
        if not filename.endswith('.txt'):
            filename += '.txt'
            
        link = await export_to_drive(user_id, content, filename)
        return f"📤 Đã export:\n🔗 {link}"

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
            await set_rag_disabled(user_id, True)
            return "🔕 Đã tắt tìm kiếm tài liệu. Bot sẽ không dùng tài liệu của bạn cho đến khi gõ /rag on."

        if sub_cmd == "on":
            await set_rag_disabled(user_id, False)
            return "🔔 Đã bật tìm kiếm tài liệu."

        return (
            "📚 Lệnh RAG:\n"
            "/rag list — xem danh sách file\n"
            "/rag delete <tên file> — xoá file\n"
            "/rag clear — xoá tất cả\n"
            "/rag off — tắt RAG\n"
            "/rag on — bật RAG"
        )

    if cmd == "audio":
        return await _handle_audio(user_id, arg)

    return None


async def _handle_audio(user_id: str, arg: str) -> str:
    parts = arg.split()
    if len(parts) != 2:
        return "⚠️ Sai cú pháp. Dùng: /audio <id> <1|2|3>"
    audio_id, choice = parts[0], parts[1]
    if choice not in ("1", "2", "3"):
        return "⚠️ Chọn 1, 2 hoặc 3."

    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT transcript, filename FROM audio_cache WHERE id = ? AND user_id = ?",
            (int(audio_id), user_id),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return "❌ Không tìm thấy bản ghi âm thanh."

    transcript, filename = row
    summary = ""
    rag_msg = ""

    # Progress feedback
    try:
        from linebot.v3.messaging import (
            Configuration, AsyncApiClient, AsyncMessagingApi,
            PushMessageRequest, TextMessage,
        )
        _cfg = Configuration(access_token=os.environ["LINE_CHANNEL_ACCESS_TOKEN"])
        async with AsyncApiClient(_cfg) as _api_client:
            _line = AsyncMessagingApi(_api_client)
            label = {"1": "phân tích", "2": "lưu RAG", "3": "phân tích + lưu RAG"}[choice]
            await _line.push_message(PushMessageRequest(
                to=user_id, messages=[TextMessage(text=f"⏳ Đang {label}...")]
            ))
    except Exception:
        pass

    # Summarize (choice 1 or 3)
    if choice in ["1", "3"]:
        agent_prompt = (
            "Bạn là một trợ lý phân tích dữ liệu và thư ký chuyên nghiệp, "
            "có khả năng đọc hiểu, tổng hợp thông tin xuất sắc.\n\n"
            "Nhiệm vụ: Đọc bản ghi âm (transcript) được cung cấp ở phần cuối và "
            "tạo ra một bản tóm tắt toàn diện, cấu trúc rõ ràng.\n\n"
            "Yêu cầu định dạng đầu ra:\n"
            "- Tổng quan (Executive Summary): 2-3 câu\n"
            "- Các điểm nhấn chính (Key Takeaways): 3-5 luận điểm\n"
            "- Thông tin chi tiết (Important Details)\n"
            "- Hành động tiếp theo/Kết luận\n\n"
            f"Bản ghi âm:\n{transcript[:15000]}"
        )
        summary = await run_agentic_loop(user_id, agent_prompt)

    # RAG (choice 2 or 3)
    if choice in ["2", "3"]:
        rag_msg = await process_file_upload(user_id, transcript.encode("utf-8"), filename)

    # Build content for Drive upload
    _now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    _word_count = len(transcript.split())
    sections = [f"📄 {filename}", f"📅 {_now} | ~{_word_count} từ", ""]
    if summary:
        sections.extend(["═══ PHÂN TÍCH ═══", summary, ""])
    sections.extend(["═══ NỘI DUNG GỐC ═══", transcript])
    content_to_save = "\n".join(sections)

    # Upload to Google Drive
    link = ""
    try:
        import httpx, json
        from google_workspace import get_google_access_token
        token = await get_google_access_token(user_id)
        if token:
            metadata = {"name": filename, "mimeType": "text/plain"}
            files = {
                "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
                "file": (filename, content_to_save.encode("utf-8"), "text/plain"),
            }
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {token}"}
                resp = await client.post(
                    "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
                    headers=headers, files=files, timeout=20.0,
                )
                if resp.status_code == 200:
                    file_id = resp.json().get("id")
                    link = f"https://drive.google.com/file/d/{file_id}/view"
                else:
                    link = f"(Lỗi upload: HTTP {resp.status_code})"
        else:
            link = "(Gửi /login để liên kết Google Drive)"
    except Exception as e:
        link = f"(Lỗi: {str(e)[:40]})"

    out = f"✅ {filename}\n"
    if summary:
        out += f"\n📝 PHÂN TÍCH:\n{summary}\n"
    if rag_msg:
        out += f"\n📚 {rag_msg}\n"
    if link:
        out += f"\n🔗 {link}"
    return out


# Export long text to Google Drive
async def export_to_drive(user_id: str, content: str, filename: str) -> str:
    """Export text content to Google Drive and return shareable link."""
    try:
        import httpx, json
        from google_workspace import get_google_access_token
        token = await get_google_access_token(user_id)
        if not token:
            return "(Gửi /login để liên kết Google Drive)"
        
        metadata = {"name": filename, "mimeType": "text/plain"}
        files = {
            "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
            "file": (filename, content.encode("utf-8"), "text/plain"),
        }
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {token}"}
            resp = await client.post(
                "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
                headers=headers, files=files, timeout=20.0,
            )
            if resp.status_code == 200:
                file_id = resp.json().get("id")
                return f"https://drive.google.com/file/d/{file_id}/view"
            else:
                return f"(Lỗi upload: HTTP {resp.status_code})"
    except Exception as e:
        return f"(Lỗi: {str(e)[:40]})"
