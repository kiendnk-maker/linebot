import aiosqlite
import datetime
from database import DB_PATH

# Bảng giá Mistral (USD / 1.000.000 tokens)
PRICING = {
    "mistral-large-latest": {"in": 2.0, "out": 6.0},
    "mistral-small-latest": {"in": 0.2, "out": 0.6},
    "codestral-latest": {"in": 0.2, "out": 0.6},
    "pixtral-12b-2409": {"in": 0.2, "out": 0.6},
    "voxtral-mini-latest": {"in": 0.0, "out": 0.0} # Chưa có giá chính thức, tạm tính 0
}

async def init_tracker_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('''CREATE TABLE IF NOT EXISTS token_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_id TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER
        )''')
        await db.commit()

async def log_usage(model_id: str, prompt_tokens: int, completion_tokens: int):
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO token_usage (model_id, prompt_tokens, completion_tokens) VALUES (?, ?, ?)",
                (model_id, prompt_tokens, completion_tokens)
            )
            await db.commit()
    except Exception as e:
        print(f"Lỗi log token: {e}")

async def get_usage_report() -> str:
    current_month = datetime.datetime.now().strftime('%Y-%m')
    total_cost = 0.0
    report_lines = [f"📊 BÁO CÁO CHI PHÍ MISTRAL API ({current_month})", "-"*30]
    
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT model_id, SUM(prompt_tokens), SUM(completion_tokens) FROM token_usage WHERE strftime('%Y-%m', timestamp) = ? GROUP BY model_id", 
                (current_month,)
            ) as cursor:
                rows = await cursor.fetchall()
                
                if not rows:
                    return "Tháng này bạn chưa tiêu tốn đồng nào cho API cả! 💸"
                
                for row in rows:
                    model, p_tokens, c_tokens = row
                    rates = PRICING.get(model, {"in": 0.0, "out": 0.0})
                    
                    cost_in = (p_tokens / 1_000_000) * rates["in"]
                    cost_out = (c_tokens / 1_000_000) * rates["out"]
                    cost_total = cost_in + cost_out
                    total_cost += cost_total
                    
                    if cost_total > 0.0001:
                        report_lines.append(f"🤖 {model}")
                        report_lines.append(f"   ↳ In: {p_tokens:,} tks (${cost_in:.4f})")
                        report_lines.append(f"   ↳ Out: {c_tokens:,} tks (${cost_out:.4f})")
                        report_lines.append(f"   ↳ Tổng: ${cost_total:.4f}")
                        
        report_lines.append("-" * 30)
        report_lines.append(f"💰 TỔNG CHI PHÍ THÁNG: ${total_cost:.4f}")
        report_lines.append("📌 Lưu ý: Đây là chi phí ước tính dựa trên Token đếm được. Vui lòng check Mistral Console để đối chiếu chính xác.")
        return "\n".join(report_lines)
    except Exception as e:
        return f"Lỗi trích xuất báo cáo: {str(e)}"
