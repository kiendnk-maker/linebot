import os
import re
import json
import httpx
import logging
import asyncio
import aiosqlite
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from groq import AsyncGroq
from linebot.v3.messaging import AsyncApiClient, AsyncMessagingApi, PushMessageRequest, TextMessage

from database import DB_PATH, save_reminder
from llm_core import MODEL_REGISTRY
import main

TZ = ZoneInfo("Asia/Taipei")
logger = logging.getLogger(__name__)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def _next_fire(fire_at: int, repeat: str) -> int:
    """Calculate next fire timestamp for repeating reminders — no dateutil dependency."""
    dt = datetime.fromtimestamp(fire_at, tz=TZ)
    if repeat == "daily":
        dt += timedelta(days=1)
    elif repeat == "weekly":
        dt += timedelta(weeks=1)
    elif repeat == "monthly":
        month = dt.month + 1
        year  = dt.year + (1 if month > 12 else 0)
        month = month if month <= 12 else 1
        dt    = dt.replace(year=year, month=month)
    return int(dt.timestamp())

_PARSE_REMINDER_PROMPT = """Extract reminder info from the user message.
Current datetime (UTC+8): {now_str}

Reply ONLY with JSON, no explanation:
{{"is_reminder": true/false, "message": "reminder content", "time": "HH:MM or null", "date": "DD/MM/YYYY", "repeat": null or "daily" or "weekly" or "monthly"}}

Rules:
- Return time as HH:MM 24h format. If no specific time return null.
- Convert Vietnamese/AM/PM: 7h toi/chieu/evening=19:00, 8h toi=20:00, 7h sang/morning=07:00, 12h trua/noon=12:00, 12h dem/midnight=00:00
- Return date as DD/MM/YYYY
- hom nay/tonight/today = {date_str}
- ngay mai/tomorrow = {tomorrow_str}
- moi ngay/every day/daily = repeat=daily
- moi tuan/every week/weekly = repeat=weekly
- moi thang/every month/monthly = repeat=monthly
- hom nay/tonight/today = {date_str}
- ngay mai/tomorrow = {tomorrow_str}
- moi ngay/every day/daily = repeat=daily
- moi tuan/every week/weekly = repeat=weekly
- moi thang/every month/monthly = repeat=monthly
- Set is_reminder=true if user mentions scheduled event with time even without explicit remind keyword.
- IMPORTANT - Examples of is_reminder=true:
    "cuoc hen luc 7h toi" -> {{"is_reminder": true, "time": "19:00", ...}}
    "toi co hop luc 14h" -> {{"is_reminder": true, "time": "14:00", ...}}
    "mai 8h sang di kham" -> {{"is_reminder": true, "time": "08:00", ...}}
    "I have a meeting at 2pm" -> {{"is_reminder": true, "time": "14:00", ...}}
- Key signals: cuoc hen, cuoc hop, thuyet trinh, gap, hen, meeting, appointment + time = is_reminder=true
- Do NOT return fire_at — only return time and date strings. Python will compute timestamp.
- If not a reminder: is_reminder=false

User message: {message}"""

async def parse_reminder_nlp(user_id: str, user_text: str) -> str | None:
    """
    Parse natural language reminder.
    Python calculates fire_at from llama8b's HH:MM + DD/MM/YYYY output
    to avoid timezone errors (SPEC §16.5).
    """
    now_dt      = datetime.now(TZ)
    now_str     = now_dt.strftime("%H:%M %d/%m/%Y %A")
    date_str    = now_dt.strftime("%d/%m/%Y")
    tomorrow_str = (now_dt + timedelta(days=1)).strftime("%d/%m/%Y")

    async with httpx.AsyncClient() as http:
        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)
        try:
            resp = await client.chat.completions.create(
                model=MODEL_REGISTRY["llama8b"]["model_id"],
                messages=[{
                    "role": "user",
                    "content": _PARSE_REMINDER_PROMPT.format(
                        now_str=now_str,
                        date_str=date_str,
                        tomorrow_str=tomorrow_str,
                        message=user_text[:300],
                    ),
                }],
                temperature=0.0,
                max_tokens=100,
            )
            raw  = resp.choices[0].message.content or ""
            raw  = re.sub(r"```[a-z]*\n?|```", "", raw).strip()
            data = json.loads(raw)
            logger.info(f"REMINDER JSON | {data}")

            if not data.get("is_reminder"):
                return None

            time_str: str | None = data.get("time")
            if not time_str:
                return "⏰ Bạn muốn đặt nhắc lúc mấy giờ?"

            date_val = data.get("date") or date_str
            repeat   = data.get("repeat")
            message  = data.get("message") or user_text[:80]

            # Python computes fire_at — llama8b never does this (SPEC §8 + §16.5)
            try:
                hh, mm    = map(int, time_str.split(":"))
                dd, mo, yy = map(int, date_val.split("/"))
                fire_dt   = datetime(yy, mo, dd, hh, mm, tzinfo=TZ)
            except Exception:
                return None

            if fire_dt <= now_dt:
                fire_dt += timedelta(days=1)

            rid        = await save_reminder(user_id, message, int(fire_dt.timestamp()), repeat)
            repeat_str = {
                "daily":   " (lặp hàng ngày)",
                "weekly":  " (lặp hàng tuần)",
                "monthly": " (lặp hàng tháng)",
            }.get(repeat or "", "")

            return (
                f"⏰ Đã đặt nhắc #{rid}{repeat_str}\n"
                f"Nội dung: {message}\n"
                f"Thời gian: {fire_dt.strftime('%H:%M %d/%m/%Y')}"
            )

        except Exception:
            return None

async def reminder_loop() -> None:
    """Background task — check reminders every 30 seconds."""
    while True:
        await asyncio.sleep(30)
        now = int(datetime.now(TZ).timestamp())

        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT id, user_id, message, repeat, fire_at FROM reminders "
                "WHERE fire_at <= ? AND done = 0",
                (now,),
            ) as cur:
                rows = await cur.fetchall()

            for rid, uid, message, repeat, fire_at_db in rows:
                try:
                    label = {
                        "daily":   " (hàng ngày)",
                        "weekly":  " (hàng tuần)",
                        "monthly": " (hàng tháng)",
                    }.get(repeat or "", "")
                    async with AsyncApiClient(main.line_config) as api_client:
                        line_api = AsyncMessagingApi(api_client)
                        await line_api.push_message(
                            PushMessageRequest(
                                to=uid,
                                messages=[TextMessage(text=f"⏰ Nhắc nhở{label}: {message}")],
                            )
                        )
                except Exception:
                    pass

                if repeat:
                    next_ts = _next_fire(fire_at_db, repeat)
                    await db.execute(
                        "UPDATE reminders SET fire_at = ? WHERE id = ?", (next_ts, rid)
                    )
                else:
                    await db.execute("UPDATE reminders SET done = 1 WHERE id = ?", (rid,))

            await db.commit()

