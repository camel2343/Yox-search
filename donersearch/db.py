from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import time
import random

_KEEP = object()


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    # Improve concurrency tolerance (especially under multiprocess crawling)
    conn.execute("PRAGMA busy_timeout=15000;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _execute_retry(conn: sqlite3.Connection, sql: str, params: Sequence[object] = (), *,
                   tries: int = 8, base_sleep: float = 0.05) -> sqlite3.Cursor:
    """Execute with retries on SQLITE_BUSY/locked errors.

    Under multi-process writes, short lock conflicts can occur. This helper
    retries with exponential backoff and small jitter.
    """
    last_err: Optional[Exception] = None
    for i in range(max(1, tries)):
        try:
            return conn.execute(sql, params)
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "locked" in msg or "busy" in msg:
                sleep_for = min(1.0, base_sleep * (2 ** i) + random.uniform(0, base_sleep))
                time.sleep(sleep_for)
                last_err = e
                continue
            raise
    assert last_err is not None
    raise last_err


def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            url TEXT NOT NULL UNIQUE,
            title TEXT,
            content TEXT,
            length INTEGER NOT NULL DEFAULT 0,
            last_crawled TEXT,
            language TEXT,
            content_hash TEXT,
            next_crawl_at TEXT
        );

        CREATE TABLE IF NOT EXISTS terms (
            term TEXT PRIMARY KEY,
            df INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS postings (
            term TEXT NOT NULL,
            doc_id INTEGER NOT NULL,
            tf INTEGER NOT NULL,
            PRIMARY KEY (term, doc_id),
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            url TEXT NOT NULL,
            alt TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS discovered_hosts (
            host TEXT PRIMARY KEY,
            first_seen TEXT NOT NULL,
            status TEXT NOT NULL,
            source_url TEXT,
            last_checked TEXT,
            retry_after TEXT,
            failure_count INTEGER NOT NULL DEFAULT 0
        );

        -- Speed up lookups
        CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term);
        CREATE INDEX IF NOT EXISTS idx_postings_doc ON postings(doc_id);
        CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);
        CREATE INDEX IF NOT EXISTS idx_images_doc ON images(doc_id);

        CREATE TABLE IF NOT EXISTS crawl_queue (
            url TEXT PRIMARY KEY,
            depth INTEGER NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            host TEXT,
            added_at TEXT NOT NULL,
            last_attempt TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_crawl_queue_priority ON crawl_queue(priority DESC, added_at ASC);

        CREATE TABLE IF NOT EXISTS crawl_runs (
            id INTEGER PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL DEFAULT 'running',
            seed_urls TEXT NOT NULL DEFAULT '[]',
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS raw_snapshots (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER,
            url TEXT NOT NULL,
            crawl_run_id INTEGER,
            fetched_at TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            raw_html TEXT,
            content_hash TEXT NOT NULL,
            document_version INTEGER NOT NULL DEFAULT 1,
            is_current INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE SET NULL,
            FOREIGN KEY (crawl_run_id) REFERENCES crawl_runs(id) ON DELETE SET NULL,
            UNIQUE(url, content_hash)
        );

        CREATE INDEX IF NOT EXISTS idx_raw_snapshots_doc ON raw_snapshots(doc_id, is_current);
        CREATE INDEX IF NOT EXISTS idx_raw_snapshots_url ON raw_snapshots(url, is_current);

        CREATE TABLE IF NOT EXISTS parsed_documents (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            raw_snapshot_id INTEGER,
            url TEXT NOT NULL,
            crawl_run_id INTEGER,
            title TEXT,
            content TEXT,
            language TEXT,
            content_hash TEXT NOT NULL,
            document_version INTEGER NOT NULL DEFAULT 1,
            license_status TEXT NOT NULL DEFAULT 'unknown',
            quality_score REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            is_current INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
            FOREIGN KEY (raw_snapshot_id) REFERENCES raw_snapshots(id) ON DELETE SET NULL,
            FOREIGN KEY (crawl_run_id) REFERENCES crawl_runs(id) ON DELETE SET NULL,
            UNIQUE(doc_id, document_version)
        );

        CREATE INDEX IF NOT EXISTS idx_parsed_documents_doc ON parsed_documents(doc_id, is_current);
        CREATE INDEX IF NOT EXISTS idx_parsed_documents_hash ON parsed_documents(content_hash);

        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY,
            parsed_document_id INTEGER NOT NULL,
            doc_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL DEFAULT 0,
            content_hash TEXT NOT NULL,
            quality_score REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (parsed_document_id) REFERENCES parsed_documents(id) ON DELETE CASCADE,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
            UNIQUE(parsed_document_id, chunk_index)
        );

        CREATE INDEX IF NOT EXISTS idx_document_chunks_doc ON document_chunks(doc_id, active);
        CREATE INDEX IF NOT EXISTS idx_document_chunks_parsed ON document_chunks(parsed_document_id, active);

        CREATE TABLE IF NOT EXISTS embeddings (
            doc_id INTEGER PRIMARY KEY,
            vector BLOB NOT NULL,
            model_name TEXT,
            updated_at TEXT,
            source_hash TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS distill_samples (
            id INTEGER PRIMARY KEY,
            dataset_version TEXT NOT NULL,
            task_type TEXT NOT NULL DEFAULT 'grounded_qa',
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            supporting_chunk_ids TEXT NOT NULL,
            source_url_ids TEXT NOT NULL,
            teacher_model TEXT NOT NULL,
            generation_prompt_version TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ready',
            sample_hash TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_distill_samples_dataset ON distill_samples(dataset_version, task_type);

        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY,
            run_type TEXT NOT NULL,
            dataset_version TEXT,
            model_name TEXT,
            model_version TEXT,
            status TEXT NOT NULL,
            config_json TEXT,
            artifact_path TEXT,
            metrics_json TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_training_runs_lookup ON training_runs(run_type, dataset_version, model_name, model_version);

        CREATE TABLE IF NOT EXISTS model_registry (
            id INTEGER PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            provider TEXT NOT NULL DEFAULT 'local',
            artifact_path TEXT,
            config_json TEXT,
            is_active INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            UNIQUE(model_name, model_version)
        );

        CREATE INDEX IF NOT EXISTS idx_model_registry_active ON model_registry(model_name, is_active);

        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            dataset_version TEXT,
            benchmark_name TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );        """
    )
    conn.commit()

    # Migration: ensure language/content_hash columns exist on documents
    cur = conn.execute("PRAGMA table_info(documents)")
    cols = {row[1] for row in cur.fetchall()}
    if "language" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN language TEXT")
        conn.commit()
        cols.add("language")
    if "content_hash" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT")
        conn.commit()
    if "next_crawl_at" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN next_crawl_at TEXT")
        conn.commit()

    # Migration: ensure images table has metadata columns
    cur = conn.execute("PRAGMA table_info(images)")
    image_cols = {row[1] for row in cur.fetchall()}
    extra_columns = {
        "width": "INTEGER",
        "height": "INTEGER",
        "format": "TEXT",
        "size_bytes": "INTEGER",
        "aspect_ratio": "REAL",
        "hash": "TEXT",
        "file_path": "TEXT",
        "thumbnail_path": "TEXT",
    }
    for col, decl in extra_columns.items():
        if col not in image_cols:
            conn.execute(f"ALTER TABLE images ADD COLUMN {col} {decl}")
    conn.commit()

    cur = conn.execute("PRAGMA table_info(embeddings)")
    embedding_cols = {row[1] for row in cur.fetchall()}
    embedding_extra = {
        "model_name": "TEXT",
        "updated_at": "TEXT",
        "source_hash": "TEXT",
    }
    for col, decl in embedding_extra.items():
        if col not in embedding_cols:
            conn.execute(f"ALTER TABLE embeddings ADD COLUMN {col} {decl}")
    conn.commit()

    cur = conn.execute("PRAGMA table_info(discovered_hosts)")
    host_cols = {row[1] for row in cur.fetchall()}
    host_extra = {
        "last_checked": "TEXT",
        "retry_after": "TEXT",
        "failure_count": "INTEGER NOT NULL DEFAULT 0",
    }
    for col, decl in host_extra.items():
        if col not in host_cols:
            conn.execute(f"ALTER TABLE discovered_hosts ADD COLUMN {col} {decl}")
    conn.commit()


def get_setting(conn: sqlite3.Connection, key: str, default: Optional[str] = None) -> Optional[str]:
    cur = conn.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = cur.fetchone()
    if not row:
        return default
    return row[0]


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    _execute_retry(
        conn,
        "INSERT INTO settings(key, value) VALUES(?, ?)\n         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def get_doc_id_by_url(conn: sqlite3.Connection, url: str) -> Optional[int]:
    cur = conn.execute("SELECT id FROM documents WHERE url=?", (url,))
    row = cur.fetchone()
    return row[0] if row else None


def upsert_document(
    conn: sqlite3.Connection,
    url: str,
    title: str,
    content: str,
    length: int,
    last_crawled: str,
    language: Optional[str],
    content_hash: str,
    next_crawl_at: Optional[str] = None,
) -> Tuple[int, str, int, bool]:
    """Insert or update a document record.

    Returns (doc_id, previous_hash, previous_length, created)
    """

    lang = (language or "").strip().lower()
    hash_value = content_hash or ""

    cur = conn.execute(
        "SELECT id, length, IFNULL(content_hash, ''), IFNULL(next_crawl_at, '') FROM documents WHERE url=?",
        (url,),
    )
    row = cur.fetchone()
    if row:
        doc_id = int(row[0])
        prev_length = int(row[1])
        prev_hash = row[2] or ""
        prev_next = row[3] or ""
        next_value = next_crawl_at if next_crawl_at is not None else prev_next or None
        conn.execute(
            "UPDATE documents SET title=?, content=?, length=?, last_crawled=?, language=?, content_hash=?, next_crawl_at=? WHERE id=?",
            (title, content, length, last_crawled, lang, hash_value, next_value, doc_id),
        )
        delta_len = length - prev_length
        if delta_len:
            total_len = int(get_setting(conn, "total_doc_len", "0") or 0) + delta_len
            set_setting(conn, "total_doc_len", str(max(total_len, 0)))
        return doc_id, prev_hash, prev_length, False

    cur = conn.execute(
        "INSERT INTO documents(url, title, content, length, last_crawled, language, content_hash, next_crawl_at) VALUES(?,?,?,?,?,?,?,?)",
        (url, title, content, length, last_crawled, lang, hash_value, next_crawl_at),
    )
    doc_id = int(cur.lastrowid)
    total_docs = int(get_setting(conn, "total_docs", "0") or 0) + 1
    total_len = int(get_setting(conn, "total_doc_len", "0") or 0) + length
    set_setting(conn, "total_docs", str(total_docs))
    set_setting(conn, "total_doc_len", str(total_len))
    return doc_id, "", length, True


def update_term_df(conn: sqlite3.Connection, term: str, delta: int) -> None:
    if delta == 0:
        return
    if delta > 0:
        _execute_retry(
            conn,
            "INSERT INTO terms(term, df) VALUES(?, ?)\n             ON CONFLICT(term) DO UPDATE SET df=df+?",
            (term, delta, delta),
        )
        return

    cur = conn.execute("SELECT df FROM terms WHERE term=?", (term,))
    row = cur.fetchone()
    if not row:
        return
    new_df = row[0] + delta
    if new_df > 0:
        _execute_retry(conn, "UPDATE terms SET df=? WHERE term=?", (new_df, term))
    else:
        _execute_retry(conn, "DELETE FROM terms WHERE term=?", (term,))


def upsert_posting(conn: sqlite3.Connection, term: str, doc_id: int, tf: int) -> None:
    _execute_retry(
        conn,
        "INSERT INTO postings(term, doc_id, tf) VALUES(?,?,?)\n         ON CONFLICT(term, doc_id) DO UPDATE SET tf=excluded.tf",
        (term, doc_id, tf),
    )


def get_existing_doc_terms(conn: sqlite3.Connection, doc_id: int) -> set[str]:
    cur = conn.execute("SELECT term FROM postings WHERE doc_id=?", (doc_id,))
    return {row[0] for row in cur.fetchall()}


def get_doc_postings(conn: sqlite3.Connection, doc_id: int) -> List[Tuple[str, int]]:
    cur = conn.execute("SELECT term, tf FROM postings WHERE doc_id=?", (doc_id,))
    return [(row[0], int(row[1])) for row in cur.fetchall()]


def delete_doc_postings(conn: sqlite3.Connection, doc_id: int) -> None:
    _execute_retry(conn, "DELETE FROM postings WHERE doc_id=?", (doc_id,))


def doc_count(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM documents")
    return int(cur.fetchone()[0])


def term_count(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM terms")
    return int(cur.fetchone()[0])


def get_doc(conn: sqlite3.Connection, doc_id: int) -> Tuple[int, str, str, str, int, str]:
    cur = conn.execute(
        "SELECT id, url, title, content, length, IFNULL(language, '') FROM documents WHERE id=?",
        (doc_id,),
    )
    row = cur.fetchone()
    if not row:
        raise KeyError(doc_id)
    return int(row[0]), row[1], row[2] or "", row[3] or "", int(row[4]), row[5] or ""


def get_docs_last_crawled(conn: sqlite3.Connection, doc_ids: Sequence[int]) -> Dict[int, str]:
    if not doc_ids:
        return {}
    qmarks = ",".join(["?\n"] * len(doc_ids))
    cur = conn.execute(
        f"SELECT id, IFNULL(last_crawled, '') FROM documents WHERE id IN ({qmarks})",
        tuple(int(i) for i in doc_ids),
    )
    return {int(row[0]): (row[1] or "") for row in cur.fetchall()}


def replace_doc_images(conn: sqlite3.Connection, doc_id: int, images: List[Dict[str, object]]) -> None:
    _execute_retry(conn, "DELETE FROM images WHERE doc_id=?", (doc_id,))
    if not images:
        return
    for record in images:
        record = dict(record)
        record.pop("data", None)
        _execute_retry(
            conn,
            "INSERT INTO images(doc_id, url, alt, width, height, format, size_bytes, aspect_ratio, hash, file_path, thumbnail_path)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                doc_id,
                record.get("url", ""),
                record.get("alt", "") or "",
                int(record.get("width") or 0),
                int(record.get("height") or 0),
                record.get("format") or "unknown",
                int(record.get("size_bytes") or 0),
                float(record.get("aspect_ratio") or 0.0),
                record.get("hash") or "",
                record.get("file_path") or "",
                record.get("thumbnail_path") or "",
            ),
        )




def image_hash_exists(conn: sqlite3.Connection, hash_value: str) -> bool:
    if not hash_value:
        return False
    cur = conn.execute("SELECT 1 FROM images WHERE hash=?", (hash_value,))
    return cur.fetchone() is not None

def update_document_language(conn: sqlite3.Connection, doc_id: int, language: str) -> None:
    lang = (language or "").strip().lower()
    _execute_retry(conn, "UPDATE documents SET language=? WHERE id=?", (lang, doc_id))


def get_images_for_docs(conn: sqlite3.Connection, doc_ids: Sequence[int]) -> Dict[int, List[Dict[str, object]]]:
    if not doc_ids:
        return {}
    qmarks = ",".join(["?"] * len(doc_ids))
    cur = conn.execute(
        f"""
        SELECT doc_id,
               url,
               IFNULL(alt, ''),
               IFNULL(width, 0),
               IFNULL(height, 0),
               IFNULL(format, ''),
               IFNULL(size_bytes, 0),
               IFNULL(aspect_ratio, 0.0),
               IFNULL(hash, ''),
               IFNULL(file_path, ''),
               IFNULL(thumbnail_path, '')
        FROM images
        WHERE doc_id IN ({qmarks})
        """,
        tuple(int(i) for i in doc_ids),
    )
    res: Dict[int, List[Dict[str, object]]] = {int(i): [] for i in doc_ids}
    for row in cur.fetchall():
        doc_id = int(row[0])
        res.setdefault(doc_id, []).append(
            {
                "url": row[1],
                "alt": row[2],
                "width": int(row[3] or 0),
                "height": int(row[4] or 0),
                "format": row[5] or "unknown",
                "size_bytes": int(row[6] or 0),
                "aspect_ratio": float(row[7] or 0.0),
                "hash": row[8] or "",
                "file_path": row[9] or "",
                "thumbnail_path": row[10] or "",
            }
        )
    return res


def record_discovered_host(
    conn: sqlite3.Connection,
    host: str,
    source_url: Optional[str],
    status: str = "pending",
) -> bool:
    host = host.lower()
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    cur = _execute_retry(
        conn,
        "INSERT INTO discovered_hosts(host, first_seen, status, source_url, last_checked, retry_after, failure_count) VALUES(?,?,?,?,?,?,0)"
        " ON CONFLICT(host) DO NOTHING",
        (host, ts, status, source_url, None, None),
    )
    return cur.rowcount > 0


def update_discovered_host_status(
    conn: sqlite3.Connection,
    host: str,
    status: str,
    *,
    retry_after: object = _KEEP,
    failure_increment: bool = False,
) -> None:
    host = host.lower()
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    set_parts = ["status=?", "last_checked=?"]
    params: List[object] = [status, ts]

    if retry_after is not _KEEP:
        set_parts.append("retry_after=?")
        params.append(retry_after)

    if failure_increment:
        set_parts.append("failure_count=failure_count+1")
    elif status in {"queued", "crawled", "seed", "pending"}:
        set_parts.append("failure_count=0")

    sql = f"UPDATE discovered_hosts SET {', '.join(set_parts)} WHERE host=?"
    params.append(host)
    _execute_retry(conn, sql, tuple(params))


def get_discovered_hosts(conn: sqlite3.Connection) -> Dict[str, str]:
    cur = conn.execute("SELECT host, status FROM discovered_hosts")
    return {row[0].lower(): row[1] for row in cur.fetchall()}


def get_hosts_for_retry(conn: sqlite3.Connection, before: str, limit: int = 50) -> List[str]:
    cur = conn.execute(
        """
        SELECT host
        FROM discovered_hosts
        WHERE status IN ('failed', 'blocked')
          AND (retry_after IS NULL OR retry_after <= ?)
        ORDER BY COALESCE(retry_after, first_seen)
        LIMIT ?
        """,
        (before, limit),
    )
    return [row[0].lower() for row in cur.fetchall()]


def get_due_documents(conn: sqlite3.Connection, before: str, limit: int = 200) -> List[Tuple[int, str]]:
    cur = conn.execute(
        """
        SELECT id, url
        FROM documents
        WHERE next_crawl_at IS NULL OR next_crawl_at <= ?
        ORDER BY COALESCE(next_crawl_at, last_crawled)
        LIMIT ?
        """,
        (before, limit),
    )
    return [(int(row[0]), row[1]) for row in cur.fetchall()]


def get_next_due_after(conn: sqlite3.Connection, after: str) -> Optional[str]:
    cur = conn.execute(
        """
        SELECT MIN(next_crawl_at)
        FROM documents
        WHERE next_crawl_at IS NOT NULL AND next_crawl_at > ?
        """,
        (after,),
    )
    row = cur.fetchone()
    if not row:
        return None
    val = row[0]
    return val if val else None


def enqueue_crawl_url(
    conn: sqlite3.Connection,
    url: str,
    depth: int,
    *,
    priority: int = 0,
    host: Optional[str] = None,
) -> None:
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    _execute_retry(
        """
        INSERT INTO crawl_queue(url, depth, priority, host, added_at)
        VALUES(?,?,?,?,?)
        ON CONFLICT(url) DO UPDATE SET
            depth = MIN(depth, excluded.depth),
            priority = MAX(priority, excluded.priority),
            host = COALESCE(excluded.host, host)
        """,
        (url, int(depth), int(priority), host, ts),
    )
    conn.commit()


def dequeue_crawl_url(conn: sqlite3.Connection, url: str) -> None:
    _execute_retry(conn, "DELETE FROM crawl_queue WHERE url=?", (url,))
    conn.commit()


def load_crawl_queue(conn: sqlite3.Connection, limit: Optional[int] = None) -> List[Tuple[str, int, int]]:
    sql = "SELECT url, depth, priority FROM crawl_queue ORDER BY priority DESC, added_at ASC"
    params: Tuple[object, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (int(limit),)
    cur = conn.execute(sql, params)
    return [(row[0], int(row[1]), int(row[2])) for row in cur.fetchall()]


def get_postings_for_terms(conn: sqlite3.Connection, terms: Sequence[str]) -> Dict[str, List[Tuple[int, int]]]:
    if not terms:
        return {}
    qmarks = ",".join(["?"] * len(terms))
    cur = conn.execute(
        f"SELECT term, doc_id, tf FROM postings WHERE term IN ({qmarks})",
        tuple(terms),
    )
    result: Dict[str, List[Tuple[int, int]]] = {t: [] for t in terms}
    for term, doc_id, tf in cur.fetchall():
        result[term].append((int(doc_id), int(tf)))
    return result


def get_dfs(conn: sqlite3.Connection, terms: Sequence[str]) -> Dict[str, int]:
    if not terms:
        return {}
    qmarks = ",".join(["?"] * len(terms))
    cur = conn.execute(
        f"SELECT term, df FROM terms WHERE term IN ({qmarks})",
        tuple(terms),
    )
    return {row[0]: int(row[1]) for row in cur.fetchall()}


def get_doc_lens(conn: sqlite3.Connection, doc_ids: Sequence[int]) -> Dict[int, int]:
    if not doc_ids:
        return {}
    qmarks = ",".join(["?"] * len(doc_ids))
    cur = conn.execute(
        f"SELECT id, length FROM documents WHERE id IN ({qmarks})",
        tuple(int(i) for i in doc_ids),
    )
    return {int(row[0]): int(row[1]) for row in cur.fetchall()}


def get_doc_languages(conn: sqlite3.Connection, doc_ids: Sequence[int]) -> Dict[int, str]:
    if not doc_ids:
        return {}
    qmarks = ",".join(["?"] * len(doc_ids))
    cur = conn.execute(
        f"SELECT id, IFNULL(language, '') FROM documents WHERE id IN ({qmarks})",
        tuple(int(i) for i in doc_ids),
    )
    return {int(row[0]): (row[1] or "") for row in cur.fetchall()}


def get_total_docs(conn: sqlite3.Connection) -> int:
    return int(get_setting(conn, "total_docs", "0") or 0)


def get_avg_doc_len(conn: sqlite3.Connection) -> float:
    total_docs = int(get_setting(conn, "total_docs", "0") or 0)
    total_len = int(get_setting(conn, "total_doc_len", "0") or 0)
    if total_docs == 0:
        return 0.0
    return float(total_len) / float(total_docs)


def delete_document(conn: sqlite3.Connection, doc_id: int) -> bool:
    cur = conn.execute("SELECT length FROM documents WHERE id=?", (doc_id,))
    row = cur.fetchone()
    if not row:
        return False
    length = int(row[0] or 0)
    for term, _ in get_doc_postings(conn, doc_id):
        update_term_df(conn, term, -1)
    conn.execute("DELETE FROM postings WHERE doc_id=?", (doc_id,))
    conn.execute("DELETE FROM images WHERE doc_id=?", (doc_id,))
    conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    total_docs = max(int(get_setting(conn, "total_docs", "0") or 0) - 1, 0)
    set_setting(conn, "total_docs", str(total_docs))
    total_len = max(int(get_setting(conn, "total_doc_len", "0") or 0) - length, 0)
    set_setting(conn, "total_doc_len", str(total_len))
    return True


def delete_document_by_url(conn: sqlite3.Connection, url: str) -> bool:
    cur = conn.execute("SELECT id FROM documents WHERE url=?", (url,))
    row = cur.fetchone()
    if not row:
        return False
    return delete_document(conn, int(row[0]))


