"""
database.py — Unified DB layer
Fixed: duplicate init_db(), init_tracker_db() spam, audio_cache outside context
"""
import os
import aiosqlite
import time
import logging
from cachetools import TTLCache

user_model_cache = TTLCache(maxsize=1000, ttl=600)
tokens_cache = TTLCache(maxsize=1000, ttl=600)

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("DB_PATH", "chat_history.db")


# ═══════════════════════════════════════════════════════════════
# SINGLE init_db() — merges both old definitions
# ═══════════════════════════════════════════════════════════════
async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        # Core tables
        await db.execute(
            "CREATE TABLE IF NOT EXISTS history "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
            " user_id TEXT, role TEXT, content TEXT)"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS user_settings "
            "(user_id TEXT PRIMARY KEY, model_key TEXT NOT NULL, language TEXT DEFAULT 'vi')"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS user_settings2 "
            "(user_id TEXT PRIMARY KEY, max_tokens INTEGER NOT NULL DEFAULT 800)"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS summary "
            "(user_id TEXT PRIMARY KEY, content TEXT, updated_at INTEGER)"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS reminders "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
            " user_id TEXT NOT NULL, "
            " message TEXT NOT NULL, "
            " fire_at INTEGER NOT NULL, "
            " repeat TEXT DEFAULT NULL, "
            " done INTEGER DEFAULT 0)"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS user_profile "
            "(user_id TEXT PRIMARY KEY, "
            " name TEXT, occupation TEXT, learning TEXT, notes TEXT)"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS rag_docs "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
            " user_id TEXT NOT NULL, "
            " filename TEXT NOT NULL, "
            " chunk_count INTEGER NOT NULL, "
            " uploaded_at INTEGER NOT NULL)"
        )
        # Audio cache
        await db.execute(
            "CREATE TABLE IF NOT EXISTS audio_cache "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
            " user_id TEXT, transcript TEXT, filename TEXT, created_at INTEGER)"
        )
        # Google workspace tables
        await db.execute(
            "CREATE TABLE IF NOT EXISTS google_auth "
            "(user_id TEXT PRIMARY KEY, refresh_token TEXT)"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS mail_cache "
            "(user_id TEXT, idx INTEGER, mail_id TEXT, PRIMARY KEY (user_id, idx))"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS mail_block "
            "(user_id TEXT, keyword TEXT, PRIMARY KEY (user_id, keyword))"
        )
        await db.commit()

    # Tracker tables (once at startup, not on every save)
    from tracker_core import init_tracker_db
    await init_tracker_db()


# ═══════════════════════════════════════════════════════════════
# MESSAGE CRUD
# ═══════════════════════════════════════════════════════════════
async def save_message(user_id: str, role: str, content: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content),
        )
        await db.commit()


async def count_history(user_id: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM history WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else 0


async def get_history_raw(user_id: str, limit: int = 5) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT role, content FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ) as cur:
            rows = await cur.fetchall()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
async def get_summary(user_id: str) -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT content FROM summary WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else ""


async def save_summary(user_id: str, content: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO summary (user_id, content, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET content=?, updated_at=?",
            (user_id, content, int(time.time()), content, int(time.time())),
        )
        await db.commit()


# ═══════════════════════════════════════════════════════════════
# USER SETTINGS (model, tokens)
# ═══════════════════════════════════════════════════════════════
async def get_user_model(user_id: str) -> str:
    if user_id in user_model_cache:
        return user_model_cache[user_id]
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT model_key FROM user_settings WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
            val = row[0] if row else "large"
    user_model_cache[user_id] = val
    return val


async def set_user_model(user_id: str, model_key: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO user_settings (user_id, model_key) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET model_key=?",
            (user_id, model_key, model_key),
        )
        await db.commit()
    user_model_cache[user_id] = model_key


async def get_user_max_tokens(user_id: str) -> int:
    if user_id in tokens_cache:
        return tokens_cache[user_id]
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT max_tokens FROM user_settings2 WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
            val = row[0] if row else 800
    tokens_cache[user_id] = val
    return val


async def set_user_max_tokens(user_id: str, val: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO user_settings2 (user_id, max_tokens) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET max_tokens=?",
            (user_id, val, val),
        )
        await db.commit()
    tokens_cache[user_id] = val


# ═══════════════════════════════════════════════════════════════
# USER PROFILE
# ═══════════════════════════════════════════════════════════════
async def get_user_profile(user_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT name, occupation, learning, notes FROM user_profile WHERE user_id = ?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            return {
                "name": row[0],
                "occupation": row[1],
                "learning": row[2],
                "notes": row[3],
            }


async def save_user_profile(user_id: str, **kwargs) -> None:
    profile = await get_user_profile(user_id) or {}
    profile.update({k: v for k, v in kwargs.items() if v is not None})
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO user_profile (user_id, name, occupation, learning, notes) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET name=?, occupation=?, learning=?, notes=?",
            (
                user_id,
                profile.get("name", ""),
                profile.get("occupation", ""),
                profile.get("learning", ""),
                profile.get("notes", ""),
                profile.get("name", ""),
                profile.get("occupation", ""),
                profile.get("learning", ""),
                profile.get("notes", ""),
            ),
        )
        await db.commit()


# ═══════════════════════════════════════════════════════════════
# REMINDERS
# ═══════════════════════════════════════════════════════════════
async def save_reminder(
    user_id: str, message: str, fire_at: int, repeat: str | None = None
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO reminders (user_id, message, fire_at, repeat) VALUES (?, ?, ?, ?)",
            (user_id, message, fire_at, repeat),
        )
        await db.commit()
        return cur.lastrowid


async def get_reminders(user_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, message, fire_at, repeat FROM reminders "
            "WHERE user_id = ? AND done = 0 ORDER BY fire_at",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [
        {"id": r[0], "message": r[1], "fire_at": r[2], "repeat": r[3]}
        for r in rows
    ]


async def cancel_reminder(user_id: str, reminder_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "DELETE FROM reminders WHERE id = ? AND user_id = ?",
            (reminder_id, user_id),
        )
        await db.commit()
        return cur.rowcount > 0
