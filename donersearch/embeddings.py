"""Semantic embedding module for DonerSearch.

Provides vector embeddings for documents and queries using sentence-transformers
or fallback to API-based embeddings (Gemini/OpenAI).

Performance: Embeddings are cached in a numpy matrix in memory after first load.
Similarity search uses vectorized numpy dot-product (~1000x faster than pure Python).
"""
from __future__ import annotations

import os
import struct
import threading
import time
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Lazy-loaded model reference
_MODEL = None
_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
_EMBEDDING_DIM = 768

# ── In-memory embedding index (vectorized) ──────────────────────────────────
_cache_lock = threading.Lock()
_cache_doc_ids: Optional[np.ndarray] = None      # shape (N,) int64
_cache_matrix: Optional[np.ndarray] = None        # shape (N, 768) float32, L2-normalized
_cache_id_to_idx: Optional[Dict[int, int]] = None # doc_id → row index
_cache_ts: float = 0.0                            # last refresh timestamp
_CACHE_TTL = 300.0                                # refresh every 5 min


def _get_model():
    """Lazy-load the sentence-transformers model."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.environ.get("YOX_EMBEDDING_MODEL", _MODEL_NAME)
        _MODEL = SentenceTransformer(model_name)
        return _MODEL
    except ImportError:
        return None


def is_available() -> bool:
    """Check if sentence-transformers is available."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def embed_text(text: str, max_length: int = 512) -> np.ndarray:
    """Generate embedding for a single text.

    Returns:
        numpy array of shape (768,)
    """
    if not text or not text.strip():
        return np.zeros(_EMBEDDING_DIM, dtype=np.float32)

    words = text.split()[:max_length]
    truncated = " ".join(words)

    model = _get_model()
    if model is None:
        vec = _api_embed_text(truncated)
        return np.array(vec, dtype=np.float32)

    try:
        return model.encode(truncated, convert_to_numpy=True).astype(np.float32)
    except Exception:
        return np.zeros(_EMBEDDING_DIM, dtype=np.float32)


def embed_batch(texts: List[str], max_length: int = 512) -> List[np.ndarray]:
    """Generate embeddings for multiple texts efficiently."""
    if not texts:
        return []

    truncated = []
    for t in texts:
        if not t or not t.strip():
            truncated.append("")
        else:
            words = t.split()[:max_length]
            truncated.append(" ".join(words))

    model = _get_model()
    if model is None:
        return [np.array(_api_embed_text(t), dtype=np.float32) for t in truncated]

    try:
        embeddings = model.encode(truncated, convert_to_numpy=True, show_progress_bar=True)
        return [e.astype(np.float32) for e in embeddings]
    except Exception:
        return [np.zeros(_EMBEDDING_DIM, dtype=np.float32) for _ in texts]


def _api_embed_text(text: str) -> List[float]:
    """Fallback: Use Gemini API for embeddings."""
    try:
        import requests
    except ImportError:
        return [0.0] * _EMBEDDING_DIM

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={api_key}"
        payload = {
            "model": "models/embedding-001",
            "content": {"parts": [{"text": text[:2000]}]}
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                values = resp.json().get("embedding", {}).get("values", [])
                if values:
                    return values
        except Exception:
            pass

    return [0.0] * _EMBEDDING_DIM


# ── Vectorized similarity (numpy) ───────────────────────────────────────────

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector or matrix of vectors."""
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-9 else v
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return v / norms


def cosine_similarity(v1, v2) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.asarray(v1, dtype=np.float32)
    b = np.asarray(v2, dtype=np.float32)
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(dot / (na * nb))


# ── Blob serialization ──────────────────────────────────────────────────────

def vector_to_blob(vector) -> bytes:
    """Compress vector to binary blob for storage."""
    arr = np.asarray(vector, dtype=np.float32)
    return zlib.compress(arr.tobytes(), level=6)


def blob_to_vector(blob: bytes) -> np.ndarray:
    """Decompress binary blob back to numpy array."""
    if not blob:
        return np.zeros(_EMBEDDING_DIM, dtype=np.float32)
    try:
        raw = zlib.decompress(blob)
        return np.frombuffer(raw, dtype=np.float32).copy()
    except Exception:
        return np.zeros(_EMBEDDING_DIM, dtype=np.float32)


# ── In-memory index management ──────────────────────────────────────────────

def _load_index(conn) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """Load all embeddings into a single numpy matrix (cached).

    Returns (doc_ids, matrix, id_to_idx).
    Matrix rows are L2-normalized for fast cosine via dot-product.
    """
    global _cache_doc_ids, _cache_matrix, _cache_id_to_idx, _cache_ts

    now = time.monotonic()
    with _cache_lock:
        if _cache_matrix is not None and (now - _cache_ts) < _CACHE_TTL:
            return _cache_doc_ids, _cache_matrix, _cache_id_to_idx

    # Build fresh index
    cur = conn.execute("SELECT doc_id, vector FROM embeddings")
    rows = cur.fetchall()

    if not rows:
        empty_ids = np.array([], dtype=np.int64)
        empty_mat = np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)
        with _cache_lock:
            _cache_doc_ids = empty_ids
            _cache_matrix = empty_mat
            _cache_id_to_idx = {}
            _cache_ts = now
        return empty_ids, empty_mat, {}

    doc_ids = []
    vectors = []
    for doc_id, blob in rows:
        if blob:
            vec = blob_to_vector(blob)
            if vec.shape == (_EMBEDDING_DIM,):
                doc_ids.append(doc_id)
                vectors.append(vec)

    if not doc_ids:
        empty_ids = np.array([], dtype=np.int64)
        empty_mat = np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)
        with _cache_lock:
            _cache_doc_ids = empty_ids
            _cache_matrix = empty_mat
            _cache_id_to_idx = {}
            _cache_ts = now
        return empty_ids, empty_mat, {}

    ids_arr = np.array(doc_ids, dtype=np.int64)
    mat = np.vstack(vectors).astype(np.float32)
    mat = _l2_normalize(mat)  # pre-normalize for fast dot-product cosine
    id_to_idx = {int(d): i for i, d in enumerate(doc_ids)}

    with _cache_lock:
        _cache_doc_ids = ids_arr
        _cache_matrix = mat
        _cache_id_to_idx = id_to_idx
        _cache_ts = now

    return ids_arr, mat, id_to_idx


def invalidate_cache():
    """Force cache refresh on next query."""
    global _cache_ts
    with _cache_lock:
        _cache_ts = 0.0


# ── Database operations ─────────────────────────────────────────────────────

def save_embedding(conn, doc_id: int, vector) -> None:
    """Save embedding to database."""
    blob = vector_to_blob(vector)
    conn.execute(
        """
        INSERT INTO embeddings(doc_id, vector) VALUES(?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET vector=excluded.vector
        """,
        (doc_id, blob),
    )
    conn.commit()


def get_embedding(conn, doc_id: int) -> Optional[np.ndarray]:
    """Retrieve embedding for a document."""
    cur = conn.execute("SELECT vector FROM embeddings WHERE doc_id=?", (doc_id,))
    row = cur.fetchone()
    if not row or not row[0]:
        return None
    return blob_to_vector(row[0])


def get_all_embeddings(conn) -> Dict[int, np.ndarray]:
    """Retrieve all embeddings (uses cache)."""
    ids, mat, id_to_idx = _load_index(conn)
    return {int(ids[i]): mat[i] for i in range(len(ids))}


def find_similar(
    conn,
    query_vector,
    top_k: int = 10,
    exclude_ids: Optional[Sequence[int]] = None,
) -> List[Tuple[int, float]]:
    """Find documents most similar to query vector using vectorized numpy.

    ~1000x faster than pure-Python loop for 21K documents.
    """
    ids_arr, mat, id_to_idx = _load_index(conn)

    if mat.shape[0] == 0:
        return []

    qvec = np.asarray(query_vector, dtype=np.float32)
    norm = np.linalg.norm(qvec)
    if norm < 1e-9:
        return []
    qvec = qvec / norm  # normalize query

    # Single matrix-vector multiply → all cosine scores at once
    scores = mat @ qvec  # shape (N,)

    # Mask excluded IDs
    if exclude_ids:
        for eid in exclude_ids:
            idx = id_to_idx.get(eid)
            if idx is not None:
                scores[idx] = -2.0

    # Top-k via argpartition (faster than full sort for large N)
    k = min(top_k, len(scores))
    if k <= 0:
        return []
    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [(int(ids_arr[i]), float(scores[i])) for i in top_indices]


def find_similar_to_doc(
    conn,
    doc_id: int,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """Find documents similar to a given document."""
    vec = get_embedding(conn, doc_id)
    if vec is None:
        return []
    return find_similar(conn, vec, top_k=top_k, exclude_ids=[doc_id])


def semantic_search(
    conn,
    query_text: str,
    top_k: int = 10,
    candidate_ids: Optional[Sequence[int]] = None,
) -> Dict[int, float]:
    """Fast semantic search: embed query → score all docs.

    If candidate_ids is given, only score those documents (even faster).
    Returns dict of doc_id → similarity score.
    """
    ids_arr, mat, id_to_idx = _load_index(conn)

    if mat.shape[0] == 0:
        return {}

    qvec = embed_text(query_text)
    norm = np.linalg.norm(qvec)
    if norm < 1e-9:
        return {}
    qvec = qvec / norm

    if candidate_ids is not None:
        # Only score the candidate subset
        result = {}
        for cid in candidate_ids:
            idx = id_to_idx.get(cid)
            if idx is not None:
                result[cid] = float(np.dot(mat[idx], qvec))
        return result
    else:
        scores = mat @ qvec
        return {int(ids_arr[i]): float(scores[i]) for i in range(len(ids_arr))}


def docs_without_embeddings(conn, limit: int = 500) -> List[Tuple[int, str]]:
    """Get documents that don't have embeddings yet."""
    cur = conn.execute(
        """
        SELECT d.id, d.content
        FROM documents d
        LEFT JOIN embeddings e ON d.id = e.doc_id
        WHERE e.doc_id IS NULL
        LIMIT ?
        """,
        (limit,),
    )
    return [(row[0], row[1] or "") for row in cur.fetchall()]


def embedding_count(conn) -> int:
    """Count documents with embeddings."""
    cur = conn.execute("SELECT COUNT(*) FROM embeddings")
    return cur.fetchone()[0]
