import os
import aiosqlite
import time
import logging
from cachetools import TTLCache

user_model_cache = TTLCache(maxsize=1000, ttl=600)
tokens_cache = TTLCache(maxsize=1000, ttl=600)

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




# --- CÁC HÀM ĐƯỢC BÓC TÁCH TỪ MAIN.PY ---
async def get_user_model(user_id: str) -> str:
    if user_id in user_model_cache: return user_model_cache[user_id]
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT model_key FROM user_settings WHERE user_id = ?", (user_id,)) as cur:
            row = await cur.fetchone()
    key = row[0] if row else "large"
    try:
        from llm_core import MODEL_REGISTRY, DEFAULT_MODEL_KEY
        ans = key if key in MODEL_REGISTRY else DEFAULT_MODEL_KEY
    except ImportError:
        ans = key
    user_model_cache[user_id] = ans
    return ans

async def set_user_model(user_id: str, model_key: str) -> None:
    user_model_cache.pop(user_id, None)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO user_settings (user_id, model_key) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET model_key = excluded.model_key",
            (user_id, model_key),
        )
        await db.commit()

async def get_user_max_tokens(user_id: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT max_tokens FROM user_settings2 WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else 800

async def set_user_max_tokens(user_id: str, max_tokens: int) -> None:
    tokens_cache.pop(user_id, None)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO user_settings2 (user_id, max_tokens) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET max_tokens = excluded.max_tokens",
            (user_id, max_tokens),
        )
        await db.commit()

async def get_user_profile(user_id: str) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT name, occupation, learning, notes FROM user_profile WHERE user_id = ?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return {}
    return {k: v for k, v in zip(("name", "occupation", "learning", "notes"), row) if v}

async def save_user_profile(user_id: str, **kwargs) -> None:
    if not kwargs:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO user_profile (user_id) VALUES (?) "
            "ON CONFLICT(user_id) DO NOTHING",
            (user_id,),
        )
        for key, value in kwargs.items():
            if key in ("name", "occupation", "learning", "notes"):
                await db.execute(
                    f"UPDATE user_profile SET {key} = ? WHERE user_id = ?",
                    (value, user_id),
                )
        await db.commit()

async def get_history_raw(user_id: str, limit: int = 30) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT role, content FROM history "
            "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ) as cur:
            rows = await cur.fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]

async def count_history(user_id: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM history WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else 0

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
            "ON CONFLICT(user_id) DO UPDATE SET "
            "content = excluded.content, updated_at = excluded.updated_at",
            (user_id, content, int(time.time())),
        )
        await db.commit()

async def get_reminders(user_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, message, fire_at, repeat FROM reminders "
            "WHERE user_id = ? AND done = 0 ORDER BY fire_at ASC",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [{"id": r[0], "message": r[1], "fire_at": r[2], "repeat": r[3]} for r in rows]

async def cancel_reminder(user_id: str, reminder_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "UPDATE reminders SET done = 1 WHERE id = ? AND user_id = ? AND done = 0",
            (reminder_id, user_id),
        )
        await db.commit()
        return cur.rowcount > 0

