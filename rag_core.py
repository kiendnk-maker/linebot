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

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

async def _get_chroma_lock(user_id: str) -> asyncio.Lock:
    """Return (and lazily create) a per-user asyncio.Lock for ChromaDB writes."""
    async with _get_meta_lock():
        if user_id not in _chroma_locks:
            _chroma_locks[user_id] = asyncio.Lock()
        return _chroma_locks[user_id]

def get_user_collection(user_id: str):
    return chroma_client.get_or_create_collection(
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

