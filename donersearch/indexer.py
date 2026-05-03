from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import random
from typing import Dict, Tuple

from . import ai_platform as aimod
from . import db as dbmod
from .tokenize import tokenize


def index_document(
    conn,
    url: str,
    title: str,
    content_text: str,
    *,
    language: str | None = None,
    title_boost: int = 3,
    force: bool = False,
    next_crawl_after_hours: float | None = None,
    raw_html: str = "",
    crawl_run_id: int | None = None,
    license_status: str = "unknown",
) -> Tuple[int, bool]:
    base_text = content_text or ""
    tokens = tokenize(base_text)
    length = len(tokens)
    last_crawled = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    content_hash = hashlib.sha256(base_text.encode("utf-8", errors="ignore")).hexdigest()

    next_crawl_at = None
    if next_crawl_after_hours and next_crawl_after_hours > 0:
        jitter_fraction = random.uniform(-0.15, 0.15)
        total_hours = max(0.25, next_crawl_after_hours * (1 + jitter_fraction))
        next_ts = datetime.utcnow() + timedelta(hours=total_hours)
        next_crawl_at = next_ts.isoformat(timespec="seconds") + "Z"

    doc_id, previous_hash, previous_length, created = dbmod.upsert_document(
        conn,
        url,
        title,
        base_text,
        length,
        last_crawled,
        language,
        content_hash,
        next_crawl_at,
    )

    changed = force or created or (previous_hash != content_hash)
    if changed and not created:
        for term, _ in dbmod.get_doc_postings(conn, doc_id):
            dbmod.update_term_df(conn, term, -1)
        dbmod.delete_doc_postings(conn, doc_id)

    if changed:
        tf: Dict[str, int] = {}
        for term in tokens:
            tf[term] = tf.get(term, 0) + 1
        if title:
            for term in tokenize(title or ""):
                tf[term] = tf.get(term, 0) + max(1, int(title_boost))
        for term, freq in tf.items():
            dbmod.update_term_df(conn, term, 1)
            dbmod.upsert_posting(conn, term, doc_id, freq)

    conn.commit()
    aimod.sync_document_pipeline(
        conn,
        doc_id=doc_id,
        url=url,
        title=title,
        raw_html=raw_html,
        content=base_text,
        language=language or "",
        content_hash=content_hash,
        crawl_run_id=crawl_run_id,
        license_status=license_status,
    )
    return doc_id, bool(changed)


def reindex_missing(conn) -> int:
    cur = conn.execute(
        """
        SELECT d.id, d.url, d.title, d.content, IFNULL(d.language, '')
        FROM documents d
        LEFT JOIN postings p ON p.doc_id = d.id
        GROUP BY d.id
        HAVING COUNT(p.term)=0
        """
    )
    rows = cur.fetchall()
    fixed = 0
    for doc_id, url, title, content, lang in rows:
        index_document(conn, url, title or "", content or "", language=lang or None, force=True)
        fixed += 1
    return fixed
