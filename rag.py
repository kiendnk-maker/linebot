"""
rag.py — RAG system for Groq哥哥 LINE Bot
Per-user knowledge base: ChromaDB + HuggingFace embedding
"""

import os
import re
import asyncio
import logging
import time
from typing import Any

import httpx
import aiosqlite
import pdfplumber
import chromadb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
HF_API_KEY   = os.environ.get("HF_API_KEY", "")
HF_EMBED_URL = (
    "https://api-inference.huggingface.co/models/"
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
CHROMA_PATH   = os.environ.get("CHROMA_PATH", "chroma")
DB_PATH       = os.environ.get("DB_PATH", "chat_history.db")

CHUNK_SIZE    = 500   # chars per chunk
CHUNK_OVERLAP = 50    # overlap between chunks
RAG_TOP_K     = 3     # similarity search top-k
MAX_PDF_BYTES = 10 * 1024 * 1024   # 10 MB guard
MAX_DOCS_PER_USER = 20             # abuse prevention

# ---------------------------------------------------------------------------
# CHROMA CLIENT — singleton, initialized once at startup
# ---------------------------------------------------------------------------
_chroma_client: chromadb.PersistentClient | None = None
_chroma_locks: dict[str, asyncio.Lock] = {}


def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        logger.info(f"ChromaDB initialized at {CHROMA_PATH}")
    return _chroma_client


def _get_user_lock(user_id: str) -> asyncio.Lock:
    """Per-user asyncio.Lock to prevent concurrent ChromaDB writes."""
    if user_id not in _chroma_locks:
        _chroma_locks[user_id] = asyncio.Lock()
    return _chroma_locks[user_id]


def _collection_name(user_id: str) -> str:
    # ChromaDB collection names: alphanumeric + underscore, max 63 chars
    safe = re.sub(r"[^a-zA-Z0-9]", "_", user_id)[:50]
    return f"rag_{safe}"


def _get_user_collection(user_id: str) -> chromadb.Collection:
    return get_chroma_client().get_or_create_collection(
        name=_collection_name(user_id),
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# EMBEDDING — HuggingFace Inference API with cold-start retry
# ---------------------------------------------------------------------------
class EmbedError(Exception):
    pass


async def embed_text(text: str) -> list[float]:
    """
    Embed a single text string via HuggingFace Inference API.
    Retries up to 3 times on 503 (model cold start).
    """
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    async with httpx.AsyncClient(timeout=60) as http:
        for attempt in range(3):
            try:
                resp = await http.post(
                    HF_EMBED_URL,
                    headers=headers,
                    json={"inputs": text},
                )
                if resp.status_code == 503:
                    logger.warning(f"HF cold start (attempt {attempt+1}/3), sleeping 20s")
                    await asyncio.sleep(20)
                    continue
                resp.raise_for_status()
                result = resp.json()
                # HF returns list[list[float]] or list[float]
                if isinstance(result, list) and isinstance(result[0], list):
                    return result[0]
                return result
            except httpx.TimeoutException:
                logger.warning(f"HF timeout (attempt {attempt+1}/3)")
                await asyncio.sleep(5)
    raise EmbedError("HuggingFace embedding unavailable after 3 retries")


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts sequentially.
    HF free tier does not support true batch embedding.
    """
    results = []
    for text in texts:
        vec = await embed_text(text)
        results.append(vec)
    return results


# ---------------------------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------------------------
def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks of CHUNK_SIZE chars."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------------------
# PDF EXTRACTION
# ---------------------------------------------------------------------------
def extract_pdf_text(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """
    Extract text from PDF bytes.
    Returns list of (page_number, text) tuples.
    Raises ValueError if PDF is empty or unreadable.
    """
    import io
    pages: list[tuple[int, str]] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    pages.append((i, text))
    except Exception as e:
        raise ValueError(f"Cannot read PDF: {e}") from e

    if not pages:
        raise ValueError("PDF has no extractable text (may be scanned image)")
    return pages


# ---------------------------------------------------------------------------
# SQLITE — rag_docs metadata
# ---------------------------------------------------------------------------
async def count_user_docs(user_id: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM rag_docs WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else 0


async def save_rag_doc(user_id: str, filename: str, chunk_count: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO rag_docs (user_id, filename, chunk_count, uploaded_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, filename, chunk_count, int(time.time())),
        )
        await db.commit()


async def list_rag_docs(user_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, filename, chunk_count, uploaded_at FROM rag_docs "
            "WHERE user_id = ? ORDER BY uploaded_at DESC",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [
        {"id": r[0], "filename": r[1], "chunk_count": r[2], "uploaded_at": r[3]}
        for r in rows
    ]


async def delete_rag_doc(user_id: str, filename: str) -> bool:
    """Delete doc metadata from SQLite. Returns True if found."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "DELETE FROM rag_docs WHERE user_id = ? AND filename = ?",
            (user_id, filename),
        )
        await db.commit()
        return cur.rowcount > 0


async def clear_rag_docs(user_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM rag_docs WHERE user_id = ?", (user_id,))
        await db.commit()


async def has_rag_docs(user_id: str) -> bool:
    return await count_user_docs(user_id) > 0


# ---------------------------------------------------------------------------
# INGEST — full PDF → ChromaDB pipeline
# ---------------------------------------------------------------------------
async def process_pdf_upload(
    user_id: str,
    pdf_bytes: bytes,
    filename: str,
) -> str:
    """
    Full ingest pipeline per flow.md guards:
      Guard 1: extension .pdf (checked in main.py before calling)
      Guard 2: file size <= 10MB
      Guard 3: pdfplumber extract not empty
      Guard 4: HF embed retry x3
      Guard 5: per-user doc limit 20 files
    Returns reply string.
    """
    # Guard 2: file size
    if len(pdf_bytes) > MAX_PDF_BYTES:
        return f"⚠️ File quá lớn ({len(pdf_bytes)//1024//1024}MB). Giới hạn 10MB."

    # Guard 5: doc limit
    doc_count = await count_user_docs(user_id)
    if doc_count >= MAX_DOCS_PER_USER:
        return (
            f"⚠️ Đã đạt giới hạn {MAX_DOCS_PER_USER} files.\n"
            "Dùng /rag delete <tên file> để xoá bớt."
        )

    # Guard 3: extract text
    try:
        pages = extract_pdf_text(pdf_bytes)
    except ValueError as e:
        return f"⚠️ {e}"

    # Chunk all pages
    all_chunks: list[str] = []
    chunk_meta: list[dict[str, Any]] = []
    for page_num, page_text in pages:
        for i, chunk in enumerate(chunk_text(page_text)):
            all_chunks.append(chunk)
            chunk_meta.append({
                "filename":    filename,
                "page":        page_num,
                "chunk_index": i,
                "uploaded_at": int(time.time()),
            })

    if not all_chunks:
        return "⚠️ Không trích xuất được nội dung từ PDF này."

    # Guard 4: embed with retry
    try:
        vectors = await embed_batch(all_chunks)
    except EmbedError as e:
        return f"⚠️ Lỗi embedding: {e}"

    # ChromaDB upsert with per-user lock
    ids = [f"{filename}_{i}" for i in range(len(all_chunks))]
    async with _get_user_lock(user_id):
        col = _get_user_collection(user_id)
        col.upsert(
            ids=ids,
            embeddings=vectors,
            documents=all_chunks,
            metadatas=chunk_meta,
        )

    # Save metadata to SQLite
    await save_rag_doc(user_id, filename, len(all_chunks))

    logger.info(f"RAG ingest | user={user_id} | file={filename} | chunks={len(all_chunks)}")
    return (
        f"✅ Đã lưu {len(all_chunks)} chunks từ {filename}\n"
        "Bạn có thể hỏi về nội dung file này."
    )


# ---------------------------------------------------------------------------
# SEARCH
# ---------------------------------------------------------------------------
async def rag_search(
    user_id: str,
    query: str,
    top_k: int = RAG_TOP_K,
) -> list[dict]:
    """
    Semantic search in user's ChromaDB collection.
    Returns list of {content, filename, page} or [] if no docs / no results.

    Guards (per flow.md):
    - Only called when has_rag_docs() is True
    - RAG not disabled (/rag off) — checked in main.py
    """
    try:
        query_vec = await embed_text(query)
    except EmbedError:
        logger.warning(f"RAG search embed failed for user={user_id}")
        return []

    try:
        col = _get_user_collection(user_id)
        results = col.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, col.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.warning(f"RAG search chroma failed user={user_id}: {e}")
        return []

    if not results["documents"] or not results["documents"][0]:
        return []

    chunks: list[dict] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # cosine distance < 0.5 = relevant (> 0.5 = likely irrelevant)
        if dist < 0.5:
            chunks.append({
                "content":  doc,
                "filename": meta.get("filename", ""),
                "page":     meta.get("page", 0),
                "distance": dist,
            })

    return chunks


# ---------------------------------------------------------------------------
# DELETE helpers (called from handle_command in main.py)
# ---------------------------------------------------------------------------
async def rag_delete_file(user_id: str, filename: str) -> str:
    """Delete all chunks for a filename from ChromaDB + SQLite metadata."""
    # Delete from ChromaDB
    try:
        col = _get_user_collection(user_id)
        # Get all IDs for this filename
        results = col.get(where={"filename": filename}, include=["documents"])
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            async with _get_user_lock(user_id):
                col.delete(ids=ids_to_delete)
    except Exception as e:
        logger.warning(f"RAG delete chroma error: {e}")

    # Delete from SQLite
    found = await delete_rag_doc(user_id, filename)
    if not found:
        return f"❌ Không tìm thấy file: {filename}"

    return f"✅ Đã xoá {filename} khỏi knowledge base."


async def rag_clear_all(user_id: str) -> str:
    """Delete entire knowledge base for a user."""
    # Delete ChromaDB collection
    try:
        async with _get_user_lock(user_id):
            get_chroma_client().delete_collection(_collection_name(user_id))
    except Exception as e:
        logger.warning(f"RAG clear chroma error: {e}")

    # Delete SQLite metadata
    await clear_rag_docs(user_id)
    return "✅ Đã xoá toàn bộ knowledge base."


async def rag_list_files(user_id: str) -> str:
    """Return formatted list of uploaded files."""
    docs = await list_rag_docs(user_id)
    if not docs:
        return "📭 Chưa có file nào trong knowledge base.\nGửi file PDF để bắt đầu."

    from datetime import datetime
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("Asia/Taipei")

    lines = [f"📚 Knowledge base ({len(docs)} files):\n"]
    for d in docs:
        dt = datetime.fromtimestamp(d["uploaded_at"], tz=TZ).strftime("%d/%m %H:%M")
        lines.append(f"• {d['filename']} ({d['chunk_count']} chunks) — {dt}")
    lines.append("\n/rag delete <tên file> để xoá")
    return "\n".join(lines)
