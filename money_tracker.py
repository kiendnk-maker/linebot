import aiosqlite
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from database import DB_PATH
from collections import defaultdict

TZ = ZoneInfo("Asia/Taipei")

async def init_money_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS money_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                record_date TEXT NOT NULL,
                description TEXT NOT NULL,
                amount REAL NOT NULL,
                created_at INTEGER
            )
        ''')
        await db.commit()

def parse_date(date_str, now_dt):
    if not date_str or str(date_str).lower() in ["date", "today", "nay", "hôm nay"]:
        return now_dt.strftime("%Y-%m-%d")
    
    match = re.search(r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{4}))?", str(date_str))
    if match:
        d = int(match.group(1))
        m = int(match.group(2))
        y = int(match.group(3)) if match.group(3) else now_dt.year
        try:
            return datetime(y, m, d).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return now_dt.strftime("%Y-%m-%d")

async def handle_money_command(user_id: str, arg: str) -> str:
    await init_money_db()
    now_dt = datetime.now(TZ)
    arg = arg.strip()
    
    if not arg:
        return (
            "⚠️ HDSD Quản lý chi tiêu:\n\n"
            "➕ THÊM GIAO DỊCH:\n"
            "/mn Tên, Số tiền, [Ngày]\n"
            "VD: /mn Ăn trưa, -50\n\n"
            "📊 XEM THỐNG KÊ:\n"
            "/mn list [today/week/month/all]\n\n"
            "✏️ SỬA GIAO DỊCH:\n"
            "/mn edit ID, Tên mới, Tiền mới, [Ngày]\n"
            "VD: /mn edit 5, Ăn tối, -100\n\n"
            "❌ XOÁ GIAO DỊCH:\n"
            "/mn rm ID\n"
            "VD: /mn rm 5"
        )

    cmd_lower = arg.lower()
    
    # ─── 1. XOÁ (RM / DEL) ───
    if cmd_lower.startswith("rm ") or cmd_lower.startswith("del "):
        try:
            target_id = int(arg.split()[1])
        except (IndexError, ValueError):
            return "⚠️ Cú pháp xoá: /mn rm <ID giao dịch>"
            
        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute("DELETE FROM money_tracking WHERE id = ? AND user_id = ?", (target_id, user_id))
            await db.commit()
            if cur.rowcount > 0:
                return f"✅ Đã xoá giao dịch #{target_id}."
            return f"❌ Không tìm thấy giao dịch #{target_id} của bạn."

    # ─── 2. SỬA (EDIT) ───
    if cmd_lower.startswith("edit "):
        edit_data = arg[5:].strip() # Cắt bỏ chữ "edit "
        parts = [p.strip() for p in edit_data.split(",")]
        if len(parts) < 3:
            return "⚠️ Cú pháp sửa: /mn edit ID, Tên mới, Số tiền mới, [Ngày]\nVD: /mn edit 12, Đổi tên, -60, 18/3"
            
        try:
            target_id = int(parts[0])
            new_amt = float(parts[1]) if parts[1].replace('.','',1).replace('-','',1).isdigit() else float(parts[2])
            # Hỗ trợ người dùng nhập lộn thứ tự Tiền và Tên
            if parts[1].replace('.','',1).replace('-','',1).isdigit():
                new_desc = parts[2]
                new_amt = float(parts[1])
            else:
                new_desc = parts[1]
                new_amt = float(parts[2])
        except ValueError:
            return "⚠️ ID hoặc Số tiền không hợp lệ."
            
        new_date_str = parts[3] if len(parts) >= 4 else "today"
        final_date = parse_date(new_date_str, now_dt)
        
        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute(
                "UPDATE money_tracking SET record_date=?, description=?, amount=? WHERE id=? AND user_id=?",
                (final_date, new_desc, new_amt, target_id, user_id)
            )
            await db.commit()
            if cur.rowcount > 0:
                d_obj = datetime.strptime(final_date, "%Y-%m-%d")
                sign = "+" if new_amt > 0 else ""
                return f"✅ Đã cập nhật giao dịch #{target_id}!\n📅 {d_obj.strftime('%d/%m/%Y')}\n📝 {new_desc}: {sign}{new_amt:,.0f}"
            return f"❌ Không tìm thấy giao dịch #{target_id}."

    # ─── 3. THỐNG KÊ (LIST) ───
    if cmd_lower.startswith("list"):
        parts = arg.split()
        range_type = parts[1].lower() if len(parts) > 1 else "today"
        
        if range_type == "today":
            start_date = now_dt.strftime("%Y-%m-%d")
            title = "HÔM NAY"
        elif range_type == "week":
            start_date = (now_dt - timedelta(days=7)).strftime("%Y-%m-%d")
            title = "7 NGÀY QUA"
        elif range_type == "month":
            start_date = (now_dt - timedelta(days=30)).strftime("%Y-%m-%d")
            title = "30 NGÀY QUA"
        elif range_type == "all":
            start_date = "2000-01-01"
            title = "TẤT CẢ"
        else:
            start_date = now_dt.strftime("%Y-%m-%d")
            title = "HÔM NAY"

        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT id, record_date, description, amount FROM money_tracking "
                "WHERE user_id = ? AND record_date >= ? "
                "ORDER BY record_date DESC, id DESC", 
                (user_id, start_date)
            ) as cur:
                rows = await cur.fetchall()

        if not rows:
            return f"📭 Không có giao dịch nào trong khoảng thời gian: {title}"

        grouped = defaultdict(list)
        total_thu = 0
        total_chi = 0

        for t_id, r_date, desc, amt in rows:
            grouped[r_date].append((t_id, desc, amt))
            if amt > 0: total_thu += amt
            else: total_chi += amt

        out = [f"📊 THỐNG KÊ ({title})", "────────────────"]
        
        for r_date, items in grouped.items():
            d_obj = datetime.strptime(r_date, "%Y-%m-%d")
            d_str = d_obj.strftime("%d/%m")
            if r_date == now_dt.strftime("%Y-%m-%d"):
                d_str += " (Hôm nay)"
            
            day_thu = sum(amt for _, _, amt in items if amt > 0)
            day_chi = sum(amt for _, _, amt in items if amt < 0)
            
            out.append(f"📅 {d_str} [Thu: {day_thu:,.0f} | Chi: {day_chi:,.0f}]")
            for t_id, desc, amt in items:
                sign = "+" if amt > 0 else ""
                out.append(f" [#{t_id}] {desc}: {sign}{amt:,.0f}")
            out.append("")

        out.append("────────────────")
        out.append(f"💰 TỔNG THU: +{total_thu:,.0f}")
        out.append(f"💸 TỔNG CHI: {total_chi:,.0f}")
        out.append(f"⚖️ CÂN ĐỐI: {(total_thu + total_chi):,.0f}")
        
        return "\n".join(out).strip()

    # ─── 4. THÊM MỚI (Mặc định) ───
    parts = [p.strip() for p in arg.split(",")]
    if len(parts) < 2:
        return "⚠️ Sai cú pháp. Dùng dấu phẩy ngăn cách:\n/mn Mua cơm, -55"
        
    desc = parts[0]
    try:
        amt = float(parts[1])
    except ValueError:
        return f"⚠️ Số tiền không hợp lệ: {parts[1]}"
        
    date_str = parts[2] if len(parts) >= 3 else "today"
    final_date = parse_date(date_str, now_dt)

    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO money_tracking (user_id, record_date, description, amount, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, final_date, desc, amt, int(now_dt.timestamp()))
        )
        new_id = cur.lastrowid
        await db.commit()

    d_obj = datetime.strptime(final_date, "%Y-%m-%d")
    d_display = d_obj.strftime("%d/%m/%Y")
    if final_date == now_dt.strftime("%Y-%m-%d"):
        d_display += " (Hôm nay)"

    sign = "+" if amt > 0 else ""
    return f"✅ Đã ghi sổ [#{new_id}]!\n📅 {d_display}\n📝 {desc}: {sign}{amt:,.0f}"
