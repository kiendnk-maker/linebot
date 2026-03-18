import os
import aiosqlite
import time
import logging

logger = logging.getLogger(__name__)

DB_PATH                   = os.environ.get("DB_PATH", "chat_history.db")


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "CREATE TABLE IF NOT EXISTS history "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
            " user_id TEXT, role TEXT, content TEXT)"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS user_settings "
            "(user_id TEXT PRIMARY KEY, model_key TEXT NOT NULL)"
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
            "(id        INTEGER PRIMARY KEY AUTOINCREMENT, "
            " user_id   TEXT    NOT NULL, "
            " message   TEXT    NOT NULL, "
            " fire_at   INTEGER NOT NULL, "
            " repeat    TEXT    DEFAULT NULL, "
            " done      INTEGER DEFAULT 0)"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS user_profile "
            "(user_id TEXT PRIMARY KEY, "
            " name TEXT, occupation TEXT, learning TEXT, notes TEXT)"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS rag_docs "
            "(id          INTEGER PRIMARY KEY AUTOINCREMENT, "
            " user_id     TEXT    NOT NULL, "
            " filename    TEXT    NOT NULL, "
            " chunk_count INTEGER NOT NULL, "
            " uploaded_at INTEGER NOT NULL)"
        )
        await db.commit()
        await db.execute("CREATE TABLE IF NOT EXISTS audio_cache (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, transcript TEXT, filename TEXT, created_at INTEGER)")
        await db.commit()


async def save_message(user_id: str, role: str, content: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content),
        )
        await db.commit()


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('CREATE TABLE IF NOT EXISTS google_auth (user_id TEXT PRIMARY KEY, refresh_token TEXT)')
        await db.execute('CREATE TABLE IF NOT EXISTS mail_cache (user_id TEXT, idx INTEGER, mail_id TEXT, PRIMARY KEY (user_id, idx))')
        await db.execute('CREATE TABLE IF NOT EXISTS mail_block (user_id TEXT, keyword TEXT, PRIMARY KEY (user_id, keyword))')
        await db.commit()


async def save_reminder(
    user_id: str, message: str, fire_at: int, repeat: str | None = None
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO reminders (user_id, message, fire_at, repeat) VALUES (?, ?, ?, ?)",
            (user_id, message, fire_at, repeat),
        )
        await db.commit()
        return cur.lastrowid  # type: ignore[return-value]


