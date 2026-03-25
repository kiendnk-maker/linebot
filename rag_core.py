import os
import time
import logging
import asyncio
import httpx
import chromadb
import pdfplumber
import io
import aiosqlite
from database import DB_PATH

logger = logging.getLogger(__name__)
CHROMA_PATH = os.environ.get('CHROMA_PATH', 'chroma')

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")

GEMINI_EMBED_MODEL = "gemini-embedding-001"

CHUNK_SIZE      = 500

CHUNK_OVERLAP   = 50

RAG_TOP_K       = 3

MAX_FILE_BYTES  = 10 * 1024 * 1024   # 10 MB

MAX_DOCS_PER_USER = 20

SUPPORTED_RAG_EXTS = {".pdf", ".txt", ".docx"}

def _get_meta_lock() -> asyncio.Lock:
    """Return the global meta-lock, creating it on first call inside the event loop.
    asyncio.Lock() must NOT be instantiated at import time in Python 3.12+.
    """
    global _chroma_locks_meta
    if _chroma_locks_meta is None:
        _chroma_locks_meta = asyncio.Lock()
    return _chroma_locks_meta

_chroma_locks_meta = None
_chroma_locks: dict[str, asyncio.Lock] = {}

_chroma_client = None  # Lazy singleton — initialized on first RAG use

def _get_chroma_client() -> chromadb.PersistentClient:
    """Return ChromaDB client, initializing it on first call (lazy init)."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client

async def warmup_chromadb() -> None:
    """Pre-warm ChromaDB. Call from /warmup endpoint to avoid cold-start on first user request."""
    _get_chroma_client()

async def _get_chroma_lock(user_id: str) -> asyncio.Lock:
    """Return (and lazily create) a per-user asyncio.Lock for ChromaDB writes."""
    async with _get_meta_lock():
        if user_id not in _chroma_locks:
            _chroma_locks[user_id] = asyncio.Lock()
        return _chroma_locks[user_id]

def get_user_collection(user_id: str):
    return _get_chroma_client().get_or_create_collection(
        name=f"rag_{user_id}",
        metadata={"hnsw:space": "cosine"},
    )

class EmbedError(RuntimeError):
    pass

async def embed_text(text: str) -> list[float]:
    import httpx, os
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                "https://api.mistral.ai/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "mistral-embed", "input": [text]}
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"Lỗi Embeddings: {e}")
        return []
async def embed_batch(texts: list[str]) -> list[list[float]]:
    sem = asyncio.Semaphore(5)  # Giới hạn 5 luồng đồng thời gọi API Mistral
    
    async def embed_with_sem(text: str) -> list[float]:
        async with sem:
            return await embed_text(text)
            
    # Xử lý đồng thời tất cả các chunk
    tasks = [embed_with_sem(t) for t in texts]
    return await asyncio.gather(*tasks)

def chunk_text(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]

async def chroma_upsert(
    user_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    filename: str,
) -> None:
    lock = await _get_chroma_lock(user_id)
    async with lock:
        col = get_user_collection(user_id)
        ts  = int(time.time())
        col.upsert(
            ids=[f"{filename}_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[
                {"filename": filename, "chunk_index": i, "uploaded_at": ts}
                for i in range(len(chunks))
            ],
        )

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
    return [{"id": r[0], "filename": r[1], "chunk_count": r[2], "uploaded_at": r[3]} for r in rows]

async def count_rag_docs(user_id: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM rag_docs WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else 0

async def has_rag_docs(user_id: str) -> bool:
    return await count_rag_docs(user_id) > 0

async def delete_rag_doc(user_id: str, filename: str) -> bool:
    """Delete one file from ChromaDB + SQLite metadata."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "DELETE FROM rag_docs WHERE user_id = ? AND filename = ?",
            (user_id, filename),
        )
        await db.commit()
        if cur.rowcount == 0:
            return False

    lock = await _get_chroma_lock(user_id)
    async with lock:
        try:
            col = get_user_collection(user_id)
            # Delete all chunks whose ID starts with filename_
            all_ids = col.get()["ids"]
            to_delete = [i for i in all_ids if i.startswith(f"{filename}_")]
            if to_delete:
                col.delete(ids=to_delete)
        except Exception as e:
            logger.warning(f"ChromaDB delete error for {filename}: {e}")
    return True

async def clear_rag_docs(user_id: str) -> int:
    """Clear all RAG docs for user. Returns number of files deleted."""
    docs = await list_rag_docs(user_id)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM rag_docs WHERE user_id = ?", (user_id,))
        await db.commit()

    lock = await _get_chroma_lock(user_id)
    async with lock:
        try:
            chroma_client.delete_collection(f"rag_{user_id}")
        except Exception:
            pass
    return len(docs)

async def rag_search(
    user_id: str,
    query: str,
    top_k: int = RAG_TOP_K,
) -> list[dict]:
    """
    Guard conditions (flow.md §rag_search):
    - Only called when user_id has at least 1 doc (caller must check has_rag_docs)
    - Not called for commands, image, audio-transcribe-only, file handler, reminders
    Returns list of {content, filename, chunk_index}.
    """
    try:
        vec = await embed_text(query)
        if not vec:
            return []
        col = get_user_collection(user_id)
        try:
            results = col.query(query_embeddings=[vec], n_results=top_k)
        except Exception as dim_err:
            if "dimension" in str(dim_err).lower():
                logger.warning(f"Dimension mismatch for {user_id}, deleting old collection")
                try:
                    chroma_client.delete_collection(f"rag_{user_id}")
                except Exception:
                    pass
                return []
            raise
        docs      = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        return [
            {
                "content":     doc,
                "filename":    meta.get("filename", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
            }
            for doc, meta in zip(docs, metadatas)
        ]
    except Exception as e:
        logger.warning(f"rag_search error for {user_id}: {e}")
        return []

async def process_file_upload(user_id: str, file_bytes: bytes, filename: str) -> str:
    """
    File ingest for RAG — supports .pdf, .txt, .docx
    Guard 1: supported extension  (caller checks)
    Guard 2: <= 10MB              (caller checks)
    Guard 3: text extraction non-empty
    Guard 4: embed retry x3
    Guard 5: per-user doc limit = 20
    """
    # Guard 5: doc count limit
    doc_count = await count_rag_docs(user_id)
    if doc_count >= MAX_DOCS_PER_USER:
        return (
            f"⚠️ Bạn đã có {doc_count} tài liệu (tối đa {MAX_DOCS_PER_USER}).\n"
            "Dùng /rag delete <tên file> để xoá bớt."
        )

    # Guard 3: extract text based on file type
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    full_text = ""

    try:
        if ext == "pdf":
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages_text = [page.extract_text() or "" for page in pdf.pages]
            full_text = "\n".join(pages_text).strip()

        elif ext == "txt":
            # Try UTF-8 first, fallback to latin-1
            try:
                full_text = file_bytes.decode("utf-8").strip()
            except UnicodeDecodeError:
                full_text = file_bytes.decode("latin-1").strip()

        elif ext == "docx":
            try:
                from docx import Document
            except ImportError:
                return "⚠️ Server chưa cài python-docx. Hãy thêm python-docx vào requirements.txt."
            doc = Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = "\n".join(paragraphs).strip()

        else:
            return f"⚠️ Định dạng .{ext} chưa được hỗ trợ."

    except Exception as e:
        logger.warning(f"File extract error for {filename}: {e}")
        return f"⚠️ Không đọc được file {filename}. File có thể bị hỏng hoặc mã hoá."

    if not full_text:
        return f"⚠️ File {filename} không có nội dung văn bản."

    # Chunk
    chunks = chunk_text(full_text)
    logger.info(f"FILE {filename} → {len(chunks)} chunks for user={user_id}")

    # Guard 4: embed with retry
    try:
        embeddings = await embed_batch(chunks)
    except EmbedError as e:
        return f"⚠️ Lỗi embedding: {e}"

    # Upsert to ChromaDB (per-user lock)
    await chroma_upsert(user_id, chunks, embeddings, filename)

    # Save metadata to SQLite
    await save_rag_doc(user_id, filename, len(chunks))

    return (
        f"✅ Đã lưu {len(chunks)} chunks từ {filename}\n"
        "Bạn có thể hỏi về nội dung file này."
    )

