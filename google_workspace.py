import os
import re
import base64
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import httpx
import asyncio
import aiosqlite
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from database import DB_PATH

TZ = ZoneInfo('Asia/Taipei')
router = APIRouter()

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")

GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")

RAILWAY_URL = "https://line-groq-bot-production-b699.up.railway.app"

async def get_google_access_token(user_id: str) -> str | None:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT refresh_token FROM google_auth WHERE user_id = ?", (user_id,)) as cur:
            row = await cur.fetchone()
    if not row: return None
    refresh_token = row[0]
    
    async with httpx.AsyncClient() as client:
        resp = await client.post("https://oauth2.googleapis.com/token", data={
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        })
        if resp.status_code == 200:
            return resp.json().get("access_token")
    return None

@router.get("/google/callback")
async def google_callback(code: str, state: str):
    user_id = state
    async with httpx.AsyncClient() as client:
        resp = await client.post("https://oauth2.googleapis.com/token", data={
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": f"{RAILWAY_URL}/google/callback"
        })
        data = resp.json()
        if "refresh_token" in data:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "INSERT INTO google_auth (user_id, refresh_token) VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET refresh_token = excluded.refresh_token",
                    (user_id, data["refresh_token"])
                )
                await db.commit()
            return HTMLResponse("<h1>✅ Cấp quyền thành công!</h1><p>Hệ thống đã kết nối với tài khoản Google. Bạn có thể đóng trang này và quay lại LINE chat với bot.</p>")
        return HTMLResponse(f"<h1>❌ Lỗi xác thực:</h1><p>{data}</p>")


async def handle_workspace_command(cmd: str, arg: str, user_id: str):
    try:
        from main import call_mistral_text, strip_markdown
    except ImportError:
        pass

    if cmd == "login":
        if not GOOGLE_CLIENT_ID: return "⚠️ Thiếu GOOGLE_CLIENT_ID trên Railway."
        redirect_uri = f"{RAILWAY_URL}/google/callback"
        scope = "https://www.googleapis.com/auth/gmail.readonly%20https://www.googleapis.com/auth/drive.file%20https://www.googleapis.com/auth/calendar.events"
        url = f"https://accounts.google.com/o/oauth2/v2/auth?client_id={GOOGLE_CLIENT_ID}&redirect_uri={redirect_uri}&response_type=code&scope={scope}&access_type=offline&prompt=consent&state={user_id}&openExternalBrowser=1"
        return f"🔐 Bấm vào link sau để cấp quyền cho Bot đọc Mail & Lịch của bạn (Chỉ cần làm 1 lần duy nhất):\n\n{url}"

    if cmd == "wedding":
        access_token = await get_google_access_token(user_id)
        if not access_token: return "⚠️ Bạn chưa đăng nhập. Hãy gõ lệnh /login"
        
        async with httpx.AsyncClient() as http:
            headers = {"Authorization": f"Bearer {access_token}"}
            now_iso = datetime.now(TZ).isoformat()
            resp = await http.get(f"https://www.googleapis.com/calendar/v3/calendars/primary/events?timeMin={now_iso}&maxResults=10&orderBy=startTime&singleEvents=true", headers=headers)
            events = resp.json().get("items", [])
            
        if not events:
            return "📅 Không có lịch trình nào sắp tới."
            
        calendar_data = ""
        for e in events:
            start = e.get("start", {}).get("dateTime", e.get("start", {}).get("date"))
            calendar_data += f"- {start}: {e.get('summary', 'No title')}\n"
            
        prompt = f"Dưới đây là lịch trình thực tế lấy từ Google Calendar. Hãy kiểm tra xem có lịch chuẩn bị đám cưới nào vào 29/03 không và tóm tắt lại thật thân thiện:\n\n{calendar_data}"
        reply = await call_mistral_text([{"role": "user", "content": prompt}], "mistral-large-latest", user_id=user_id)
        return strip_markdown(reply)

    if cmd == "block":
        if not arg or arg.strip() == "":
            return "⚠️ Cú pháp:\n/block ls — xem danh sách\n/block Shopee — thêm từ khoá\n/block Shopee_GitHub — thêm nhiều"

        if arg.strip().lower() == "ls":
            async with aiosqlite.connect(DB_PATH) as db:
                async with db.execute("SELECT keyword FROM mail_block WHERE user_id = ? ORDER BY keyword", (user_id,)) as cur:
                    rows = await cur.fetchall()
            if not rows:
                return "📋 Chưa block từ khoá nào.\n/block Shopee — để thêm"
            kw_list = "\n".join(f"  ✕ {r[0]}" for r in rows)
            return f"🚫 Đang block ({len(rows)}):\n{kw_list}\n\n/unblock Shopee — để mở lại"

        # Add keywords (split by _ or space)
        raw_keywords = re.split(r'[_\s]+', arg.strip())
        keywords = [k.strip() for k in raw_keywords if k.strip()]
        if not keywords:
            return "⚠️ Không có từ khoá hợp lệ."

        added: list[str] = []
        async with aiosqlite.connect(DB_PATH) as db:
            for kw in keywords:
                try:
                    await db.execute("INSERT INTO mail_block (user_id, keyword) VALUES (?, ?)", (user_id, kw.lower()))
                    added.append(kw)
                except Exception:
                    pass  # duplicate — skip
            await db.commit()

        if not added:
            return f"ℹ️ Tất cả từ khoá đã có trong danh sách block."
        return f"🚫 Đã block: {', '.join(added)}\n/block ls — xem danh sách"

    if cmd == "unblock":
        if not arg or arg.strip() == "":
            return "⚠️ Cú pháp: /unblock Shopee hoặc /unblock Shopee_GitHub"

        raw_keywords = re.split(r'[_\s]+', arg.strip())
        keywords = [k.strip() for k in raw_keywords if k.strip()]

        removed: list[str] = []
        async with aiosqlite.connect(DB_PATH) as db:
            for kw in keywords:
                cur = await db.execute("DELETE FROM mail_block WHERE user_id = ? AND keyword = ?", (user_id, kw.lower()))
                if cur.rowcount > 0:
                    removed.append(kw)
            await db.commit()

        if not removed:
            return f"ℹ️ Không tìm thấy từ khoá nào để mở block."
        return f"✅ Đã mở block: {', '.join(removed)}"

    if cmd == "cal":
        access_token = await get_google_access_token(user_id)
        if not access_token: return "⚠️ Bạn chưa đăng nhập. Hãy gõ lệnh /login"

        async with httpx.AsyncClient() as http:
            headers = {"Authorization": f"Bearer {access_token}"}

            # /cal hoặc /cal ls -> Hiển thị lịch
            if not arg or arg == "ls":
                now = datetime.now(TZ).isoformat()
                params = {"timeMin": now, "maxResults": 5, "singleEvents": "true", "orderBy": "startTime"}
                resp = await http.get("https://www.googleapis.com/calendar/v3/calendars/primary/events", headers=headers, params=params)
                events = resp.json().get("items", [])

                if not events: return "📅 Không có lịch trình sắp tới."

                out = "📅 LỊCH TRÌNH SẮP TỚI\n"
                for e in events:
                    start = e["start"].get("dateTime", e["start"].get("date", ""))
                    start = start.replace("T", " ")[:16]
                    summary = e.get("summary", "(Không tiêu đề)")
                    out += f"🔹 {start} — {summary}\n"

                out += "\n➕ Thêm: /cal add Họp nhóm 3pm thứ 6"
                return out

            # /cal add [nội dung] -> Thêm lịch nhanh
            elif arg.startswith("add "):
                event_text = arg[4:].strip()
                resp = await http.post("https://www.googleapis.com/calendar/v3/calendars/primary/events/quickAdd", headers=headers, params={"text": event_text})

                if resp.status_code == 200:
                    e = resp.json()
                    start = e["start"].get("dateTime", e["start"].get("date", "")).replace("T", " ")[:16]
                    return f"✅ Đã thêm lịch!\n📝 {e.get('summary')}\n⏰ {start}"
                else:
                    return f"❌ Lỗi tạo lịch: {resp.text}"

    if cmd == "ls" and arg.startswith("mail"):
        access_token = await get_google_access_token(user_id)
        if not access_token: return "⚠️ Bạn chưa đăng nhập. Hãy gõ lệnh /login"

        # Parse: /ls mail [page] [Nd]  e.g. /ls mail 2 7d
        page = 1
        days = 3
        tokens = arg.replace("mail", "").strip().split()
        for t in tokens:
            if re.fullmatch(r'\d+d', t):
                days = int(t[:-1])
            elif t.isdigit():
                page = int(t)

        async with httpx.AsyncClient() as http:
            headers = {"Authorization": f"Bearer {access_token}"}
            params = {
                "q": f"in:inbox newer_than:{days}d -category:promotions -category:updates -category:social -category:forums",
                "maxResults": 200,
            }
            resp = await http.get("https://gmail.googleapis.com/gmail/v1/users/me/messages",
                                  headers=headers, params=params)
            mail_list = resp.json().get("messages", [])

            if not mail_list: return "📬 Hộp thư sạch — không có email quan trọng nào."

            # Fetch metadata for all messages, then deduplicate by sender
            fetched: list[dict] = []
            sem = asyncio.Semaphore(15)  # Giới hạn 15 luồng đồng thời để chống Rate Limit

            async def fetch_single_mail(m: dict) -> dict:
                m_id = m["id"]
                async with sem:
                    try:
                        det_resp = await http.get(
                            f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{m_id}?format=metadata&metadataHeaders=Subject&metadataHeaders=From",
                            headers=headers,
                            timeout=15.0
                        )
                        h_data = det_resp.json().get("payload", {}).get("headers", [])
                        subj = next((h["value"] for h in h_data if h["name"] == "Subject"), "(No Subject)")
                        frm = next((h["value"] for h in h_data if h["name"] == "From"), "Unknown")
                        sender_email = re.search(r'<(.+?)>', frm)
                        sender_key = sender_email.group(1).lower() if sender_email else frm.lower().strip()
                        sender_name = re.sub(r'<.*?>', '', frm).strip()[:20]
                        return {"id": m_id, "subject": subj, "sender_name": sender_name, "sender_key": sender_key}
                    except Exception:
                        return None

            # Phóng tất cả các luồng cùng một lúc và gom kết quả
            tasks = [fetch_single_mail(m) for m in mail_list]
            results = await asyncio.gather(*tasks)
            fetched = [res for res in results if res is not None]

            # Deduplicate: keep only the latest email per sender
            seen_senders: set[str] = set()
            unique_mails: list[dict] = []
            for mail in fetched:
                if mail["sender_key"] not in seen_senders:
                    seen_senders.add(mail["sender_key"])
                    unique_mails.append(mail)

            # Apply user's block list
            async with aiosqlite.connect(DB_PATH) as db:
                async with db.execute("SELECT keyword FROM mail_block WHERE user_id = ?", (user_id,)) as cur:
                    block_rows = await cur.fetchall()
            blocked_keywords = [r[0] for r in block_rows]

            if blocked_keywords:
                def _is_blocked(mail: dict) -> bool:
                    haystack = f"{mail['sender_name']} {mail['sender_key']} {mail['subject']}".lower()
                    return any(kw in haystack for kw in blocked_keywords)
                unique_mails = [m for m in unique_mails if not _is_blocked(m)]

            if not unique_mails: return "📬 Hộp thư sạch — không có email quan trọng nào."

            start_idx = (page - 1) * 20
            end_idx = start_idx + 20
            display_list = unique_mails[start_idx:end_idx]

            if not display_list: return f"⚠️ Trang {page} không có email nào."

            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("DELETE FROM mail_cache WHERE user_id = ?", (user_id,))
                total_unique = len(unique_mails)
                total_pages = (total_unique + 19) // 20
                day_label = f" · {days}d" if days != 3 else ""
                
                bubbles = []
                for chunk_idx in range(0, len(display_list), 5):
                    chunk = display_list[chunk_idx:chunk_idx+5]
                    box_contents = []
                    for i, mail in enumerate(chunk):
                        idx = start_idx + chunk_idx + i + 1
                        await db.execute("INSERT INTO mail_cache (user_id, idx, mail_id) VALUES (?, ?, ?)", (user_id, idx, mail["id"]))
                        
                        item = {
                            "type": "box",
                            "layout": "vertical",
                            "margin": "md",
                            "spacing": "sm",
                            "action": {
                                "type": "message",
                                "label": "Read",
                                "text": f"/mail {idx}"
                            },
                            "contents": [
                                {
                                    "type": "text",
                                    "text": f"{idx}. {mail['sender_name'][:25]}",
                                    "weight": "bold",
                                    "size": "sm",
                                    "color": "#1DB446",
                                    "wrap": True
                                },
                                {
                                    "type": "text",
                                    "text": mail['subject'],
                                    "size": "xs",
                                    "color": "#888888",
                                    "wrap": True,
                                    "maxLines": 2
                                }
                            ]
                        }
                        box_contents.append(item)
                        if i < len(chunk) - 1:
                            box_contents.append({"type": "separator", "margin": "md", "color": "#eeeeee"})
                    
                    bubble = {
                        "type": "bubble",
                        "size": "mega",
                        "header": {
                            "type": "box",
                            "layout": "vertical",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": f"HỘP THƯ ({page}/{total_pages}){day_label}",
                                    "weight": "bold",
                                    "size": "md",
                                    "color": "#ffffff"
                                }
                            ],
                            "backgroundColor": "#2c3e50"
                        },
                        "body": {
                            "type": "box",
                            "layout": "vertical",
                            "contents": box_contents
                        }
                    }
                    bubbles.append(bubble)
                await db.commit()

        day_suffix = f" {days}d" if days != 3 else ""
        qr_items = []
        if page > 1:
            qr_items.append({"type": "action", "action": {"type": "message", "label": "⬅️ Trước", "text": f"/ls mail {page-1}{day_suffix}"}})
        if len(unique_mails) > end_idx:
            qr_items.append({"type": "action", "action": {"type": "message", "label": "Sau ➡️", "text": f"/ls mail {page+1}{day_suffix}"}})
        qr_items.append({"type": "action", "action": {"type": "message", "label": "🔄 Mới", "text": f"/ls mail 1{day_suffix}"}})
        qr_items.append({"type": "action", "action": {"type": "message", "label": "🚫 Block Nhanh", "text": "/block ls"}})

        return {
            "type": "flex",
            "altText": f"Hộp thư: {total_unique} email",
            "contents": {
                "type": "carousel",
                "contents": bubbles
            },
            "quickReply": {"items": qr_items}
        }

    if cmd == "mail":
        access_token = await get_google_access_token(user_id)
        if not access_token: return "⚠️ Bạn chưa đăng nhập. Hãy gõ lệnh /login"
        
        if not arg or not arg.isdigit():
            return "⚠️ Vui lòng nhập số thứ tự email. Ví dụ: /mail 1"

        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute("SELECT mail_id FROM mail_cache WHERE user_id = ? AND idx = ?", (user_id, int(arg))) as cur:
                row = await cur.fetchone()
        
        if not row:
            return f"⚠️ Không tìm thấy email số {arg} trong trang hiện hành. Hãy gõ /ls mail để tải lại."
        m_id = row[0]

        async with httpx.AsyncClient() as http:
            headers = {"Authorization": f"Bearer {access_token}"}
            det_resp = await http.get(f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{m_id}?format=full", headers=headers)
            payload = det_resp.json().get("payload", {})
            
            def get_text(p):
                mime = p.get("mimeType", "")
                if mime == "text/plain":
                    return p.get("body", {}).get("data", "")
                for sub in p.get("parts", []):
                    res = get_text(sub)
                    if res: return res
                if mime == "text/html":
                    return p.get("body", {}).get("data", "")
                return ""
                
            body_data = get_text(payload)
            if body_data:
                body_data = body_data.replace("-", "+").replace("_", "/")
                body_data += "=" * ((4 - len(body_data) % 4) % 4)
                try:
                    content_str = base64.b64decode(body_data).decode('utf-8')
                    content_str = re.sub(r'<style.*?>.*?</style>', '', content_str, flags=re.DOTALL|re.IGNORECASE)
                    content_str = re.sub(r'<script.*?>.*?</script>', '', content_str, flags=re.DOTALL|re.IGNORECASE)
                    content_str = re.sub(r'<[^>]+>', ' ', content_str)
                    content_str = re.sub(r'\s+', ' ', content_str).strip()
                except Exception:
                    content_str = det_resp.json().get("snippet", "Lỗi giải mã nội dung.")
            else:
                content_str = det_resp.json().get("snippet", "Chỉ đọc được ảnh hoặc định dạng ẩn.")
                
        prompt = f"Dưới đây là nội dung email thực tế tôi vừa nhận. Hãy tóm tắt lại gọn gàng, liệt kê các ý chính:\n\n{content_str[:3000]}"
        reply = await call_mistral_text([{"role": "user", "content": prompt}], "mistral-large-latest", user_id=user_id)
        return strip_markdown(reply)

    return None

# FIXED_422_MISTRAL
