from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import db as dbmod
from . import embeddings as embmod
from .search import search_with_fuzzy
from .tokenize import tokenize

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

DATA_ROOT = Path("data")
DATASETS_ROOT = DATA_ROOT / "datasets"
MODELS_ROOT = DATA_ROOT / "models"
DEFAULT_TEACHER_MODEL = os.environ.get("YOX_TEACHER_MODEL", "openai/gpt-oss-120b:exacto")
DEFAULT_GENERATION_PROMPT_VERSION = "v1"


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ensure_ai_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)


def quality_score(title: str, content: str) -> float:
    text = (content or "").strip()
    if not text:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    length_score = min(len(words) / 250.0, 1.0)
    uniq_ratio = len(set(w.casefold() for w in words)) / max(1.0, float(len(words)))
    uniq_score = min(max((uniq_ratio - 0.18) / 0.45, 0.0), 1.0)
    punct = sum(1 for ch in text if ch in ".,!?:;") / max(1.0, float(len(words)))
    punct_score = min(punct * 18.0, 1.0)
    title_bonus = 0.15 if (title or "").strip() else 0.0
    score = 0.45 * length_score + 0.35 * uniq_score + 0.20 * punct_score + title_bonus
    return round(min(max(score, 0.0), 1.0), 4)


def chunk_text(text: str, max_tokens: int = 160, overlap: int = 30) -> List[str]:
    words = (text or "").split()
    if not words:
        return []
    if len(words) <= max_tokens:
        return [" ".join(words)]
    step = max(1, max_tokens - max(0, overlap))
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_tokens)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += step
    return chunks


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _json_dump(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def start_crawl_run(conn, seed_urls: Sequence[str], notes: str = "") -> int:
    ts = utc_now()
    cur = conn.execute(
        "INSERT INTO crawl_runs(started_at, status, seed_urls, notes) VALUES(?,?,?,?)",
        (ts, "running", _json_dump(list(seed_urls)), notes or ""),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_crawl_run(conn, crawl_run_id: Optional[int], status: str = "completed", notes: str = "") -> None:
    if not crawl_run_id:
        return
    conn.execute(
        "UPDATE crawl_runs SET finished_at=?, status=?, notes=COALESCE(NULLIF(?, ''), notes) WHERE id=?",
        (utc_now(), status, notes or "", int(crawl_run_id)),
    )
    conn.commit()


def _latest_snapshot(conn, doc_id: int, content_hash: str) -> Optional[Tuple[int, int]]:
    cur = conn.execute(
        "SELECT id, document_version FROM raw_snapshots WHERE doc_id=? AND content_hash=? ORDER BY id DESC LIMIT 1",
        (int(doc_id), content_hash),
    )
    row = cur.fetchone()
    if not row:
        return None
    return int(row[0]), int(row[1])


def _next_document_version(conn, doc_id: int) -> int:
    cur = conn.execute("SELECT COALESCE(MAX(document_version), 0) FROM raw_snapshots WHERE doc_id=?", (int(doc_id),))
    row = cur.fetchone()
    return int(row[0] or 0) + 1


def _current_parsed_document(conn, doc_id: int) -> Optional[Tuple[int, str]]:
    cur = conn.execute(
        "SELECT id, content_hash FROM parsed_documents WHERE doc_id=? AND is_current=1 ORDER BY id DESC LIMIT 1",
        (int(doc_id),),
    )
    row = cur.fetchone()
    if not row:
        return None
    return int(row[0]), row[1] or ""


def sync_document_pipeline(
    conn,
    *,
    doc_id: int,
    url: str,
    title: str,
    raw_html: str,
    content: str,
    language: str,
    content_hash: str,
    crawl_run_id: Optional[int] = None,
    license_status: str = "unknown",
    max_chunk_tokens: int = 160,
    chunk_overlap: int = 30,
) -> Dict[str, object]:
    now = utc_now()
    clean_title = _normalize_whitespace(title)
    clean_content = _normalize_whitespace(content)
    lang = (language or "").strip().lower()
    score = quality_score(clean_title, clean_content)
    existing = _latest_snapshot(conn, int(doc_id), content_hash)
    created = False

    if existing:
        snapshot_id, version = existing
        conn.execute(
            "UPDATE raw_snapshots SET last_seen_at=?, fetched_at=?, crawl_run_id=?, is_current=1 WHERE id=?",
            (now, now, crawl_run_id, snapshot_id),
        )
        conn.execute(
            "UPDATE raw_snapshots SET is_current=0 WHERE doc_id=? AND id<>?",
            (int(doc_id), snapshot_id),
        )
        conn.execute(
            "UPDATE parsed_documents SET last_seen_at=?, crawl_run_id=?, is_current=CASE WHEN content_hash=? THEN 1 ELSE 0 END WHERE doc_id=?",
            (now, crawl_run_id, content_hash, int(doc_id)),
        )
        cur = conn.execute(
            "SELECT id FROM parsed_documents WHERE doc_id=? AND content_hash=? ORDER BY id DESC LIMIT 1",
            (int(doc_id), content_hash),
        )
        parsed_row = cur.fetchone()
        parsed_document_id = int(parsed_row[0]) if parsed_row else 0
        changed = False
    else:
        version = _next_document_version(conn, int(doc_id))
        conn.execute("UPDATE raw_snapshots SET is_current=0 WHERE doc_id=?", (int(doc_id),))
        conn.execute("UPDATE parsed_documents SET is_current=0 WHERE doc_id=?", (int(doc_id),))
        cur = conn.execute(
            "INSERT INTO raw_snapshots(doc_id, url, crawl_run_id, fetched_at, first_seen_at, last_seen_at, raw_html, content_hash, document_version, is_current) VALUES(?,?,?,?,?,?,?,?,?,1)",
            (int(doc_id), url, crawl_run_id, now, now, now, raw_html or "", content_hash, version),
        )
        snapshot_id = int(cur.lastrowid)
        cur = conn.execute(
            "INSERT INTO parsed_documents(doc_id, raw_snapshot_id, url, crawl_run_id, title, content, language, content_hash, document_version, license_status, quality_score, created_at, first_seen_at, last_seen_at, is_current) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
            (int(doc_id), snapshot_id, url, crawl_run_id, clean_title, clean_content, lang, content_hash, version, license_status, score, now, now, now),
        )
        parsed_document_id = int(cur.lastrowid)
        conn.execute("UPDATE document_chunks SET active=0 WHERE doc_id=?", (int(doc_id),))
        for idx, chunk in enumerate(chunk_text(clean_content, max_chunk_tokens, chunk_overlap)):
            conn.execute(
                "INSERT INTO document_chunks(parsed_document_id, doc_id, chunk_index, content, token_count, content_hash, quality_score, created_at, active) VALUES(?,?,?,?,?,?,?,?,1)",
                (
                    parsed_document_id,
                    int(doc_id),
                    idx,
                    chunk,
                    len(tokenize(chunk, remove_stopwords=False)),
                    _sha(chunk),
                    score,
                    now,
                ),
            )
        changed = True
        created = True

    conn.commit()
    return {
        "snapshot_id": int(snapshot_id),
        "parsed_document_id": int(parsed_document_id),
        "document_version": int(version),
        "changed": bool(changed),
        "created": bool(created),
        "quality_score": float(score),
    }


def backfill_pipeline(conn, *, limit: Optional[int] = None, chunk_tokens: int = 160, chunk_overlap: int = 30) -> int:
    sql = (
        "SELECT d.id, d.url, IFNULL(d.title,''), IFNULL(d.content,''), IFNULL(d.language,''), IFNULL(d.content_hash,'') "
        "FROM documents d LEFT JOIN parsed_documents p ON p.doc_id=d.id AND p.is_current=1 "
        "WHERE p.id IS NULL ORDER BY d.id"
    )
    params: Tuple[object, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (int(limit),)
    cur = conn.execute(sql, params)
    count = 0
    for doc_id, url, title, content, language, content_hash in cur.fetchall():
        effective_hash = (content_hash or _sha(content or ""))
        sync_document_pipeline(
            conn,
            doc_id=int(doc_id),
            url=url,
            title=title or "",
            raw_html="",
            content=content or "",
            language=language or "",
            content_hash=effective_hash,
            chunk_overlap=chunk_overlap,
            max_chunk_tokens=chunk_tokens,
        )
        count += 1
    return count


def chunk_documents(conn, *, limit: Optional[int] = None, max_chunk_tokens: int = 160, chunk_overlap: int = 30) -> int:
    cur = conn.execute(
        "SELECT id, url, IFNULL(title,''), IFNULL(content,''), IFNULL(language,''), IFNULL(content_hash,'') FROM documents ORDER BY id"
        + (" LIMIT ?" if limit is not None else ""),
        ((int(limit),) if limit is not None else ()),
    )
    count = 0
    for doc_id, url, title, content, language, content_hash in cur.fetchall():
        sync_document_pipeline(
            conn,
            doc_id=int(doc_id),
            url=url,
            title=title or "",
            raw_html="",
            content=content or "",
            language=language or "",
            content_hash=content_hash or _sha(content or ""),
            max_chunk_tokens=max_chunk_tokens,
            chunk_overlap=chunk_overlap,
        )
        count += 1
    return count


def _dataset_dir(dataset_version: str) -> Path:
    ensure_ai_dirs()
    path = DATASETS_ROOT / dataset_version
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_dataset(conn, *, dataset_version: str, min_quality: float = 0.15) -> Dict[str, object]:
    ensure_ai_dirs()
    out_dir = _dataset_dir(dataset_version)
    plain_path = out_dir / "plain_text.jsonl"
    source_path = out_dir / "grounded_sources.jsonl"
    manifest_path = out_dir / "manifest.json"

    docs = conn.execute(
        "SELECT p.doc_id, p.url, IFNULL(p.title,''), IFNULL(p.content,''), IFNULL(p.language,''), p.content_hash, p.document_version, p.quality_score "
        "FROM parsed_documents p WHERE p.is_current=1 AND p.quality_score>=? ORDER BY p.doc_id",
        (float(min_quality),),
    ).fetchall()
    chunks = conn.execute(
        "SELECT c.id, c.doc_id, c.chunk_index, c.content, c.token_count, c.content_hash, p.url, IFNULL(p.title,''), IFNULL(p.language,''), p.document_version "
        "FROM document_chunks c JOIN parsed_documents p ON p.id=c.parsed_document_id "
        "WHERE c.active=1 AND p.is_current=1 AND p.quality_score>=? ORDER BY c.doc_id, c.chunk_index",
        (float(min_quality),),
    ).fetchall()

    with plain_path.open("w", encoding="utf-8") as fp:
        for row in docs:
            fp.write(_json_dump({
                "doc_id": int(row[0]),
                "url": row[1],
                "title": row[2] or "",
                "text": row[3] or "",
                "language": row[4] or "",
                "content_hash": row[5] or "",
                "document_version": int(row[6] or 0),
                "quality_score": float(row[7] or 0.0),
            }) + "\n")

    with source_path.open("w", encoding="utf-8") as fp:
        for row in chunks:
            fp.write(_json_dump({
                "chunk_id": int(row[0]),
                "doc_id": int(row[1]),
                "chunk_index": int(row[2]),
                "text": row[3] or "",
                "token_count": int(row[4] or 0),
                "content_hash": row[5] or "",
                "url": row[6],
                "title": row[7] or "",
                "language": row[8] or "",
                "document_version": int(row[9] or 0),
            }) + "\n")

    manifest = {
        "dataset_version": dataset_version,
        "created_at": utc_now(),
        "document_count": len(docs),
        "chunk_count": len(chunks),
        "min_quality": min_quality,
        "files": {
            "plain_text": str(plain_path),
            "grounded_sources": str(source_path),
        },
    }
    manifest_path.write_text(_json_dump(manifest), encoding="utf-8")
    return manifest


def _documents_for_embedding(conn, *, only_missing: bool = True, limit: Optional[int] = None) -> List[Tuple[int, str, str, str]]:
    sql = (
        "SELECT d.id, IFNULL(d.title,''), IFNULL(d.content,''), IFNULL(d.content_hash,'') "
        "FROM documents d LEFT JOIN embeddings e ON e.doc_id=d.id "
    )
    params: List[object] = []
    if only_missing:
        sql += "WHERE e.doc_id IS NULL OR IFNULL(e.source_hash,'') <> IFNULL(d.content_hash,'') "
    sql += "ORDER BY d.id"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    cur = conn.execute(sql, tuple(params))
    return [(int(row[0]), row[1] or "", row[2] or "", row[3] or "") for row in cur.fetchall()]


def build_embeddings(conn, *, limit: Optional[int] = None, only_missing: bool = True, model_name: Optional[str] = None) -> Dict[str, object]:
    docs = _documents_for_embedding(conn, only_missing=only_missing, limit=limit)
    if not docs:
        return {"processed": 0, "model_name": model_name or os.environ.get("YOX_EMBEDDING_MODEL", "default")}
    texts = [((title + "\n\n") if title else "") + content for _, title, content, _ in docs]
    vectors = embmod.embed_batch(texts)
    now = utc_now()
    used_model = model_name or os.environ.get("YOX_EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
    for (doc_id, _, _, content_hash), vector in zip(docs, vectors):
        blob = embmod.vector_to_blob(vector)
        conn.execute(
            "INSERT INTO embeddings(doc_id, vector, model_name, updated_at, source_hash) VALUES(?,?,?,?,?) "
            "ON CONFLICT(doc_id) DO UPDATE SET vector=excluded.vector, model_name=excluded.model_name, updated_at=excluded.updated_at, source_hash=excluded.source_hash",
            (int(doc_id), blob, used_model, now, content_hash or ""),
        )
    conn.commit()
    embmod.invalidate_cache()
    return {"processed": len(docs), "model_name": used_model}


def active_chunks_for_docs(conn, doc_ids: Sequence[int], per_doc: int = 3) -> Dict[int, List[Dict[str, object]]]:
    if not doc_ids:
        return {}
    qmarks = ",".join(["?"] * len(doc_ids))
    cur = conn.execute(
        f"SELECT c.id, c.doc_id, c.chunk_index, c.content, c.token_count, p.url, IFNULL(p.title,''), IFNULL(p.language,'') "
        f"FROM document_chunks c JOIN parsed_documents p ON p.id=c.parsed_document_id "
        f"WHERE c.active=1 AND p.is_current=1 AND c.doc_id IN ({qmarks}) ORDER BY c.doc_id, c.chunk_index",
        tuple(int(doc_id) for doc_id in doc_ids),
    )
    result: Dict[int, List[Dict[str, object]]] = {int(doc_id): [] for doc_id in doc_ids}
    for row in cur.fetchall():
        doc_id = int(row[1])
        if len(result.setdefault(doc_id, [])) >= per_doc:
            continue
        result[doc_id].append({
            "chunk_id": int(row[0]),
            "doc_id": doc_id,
            "chunk_index": int(row[2]),
            "content": row[3] or "",
            "token_count": int(row[4] or 0),
            "url": row[5],
            "title": row[6] or "",
            "language": row[7] or "",
        })
    return result


def source_hits(conn, query: str, *, top_docs: int = 5, chunks_per_doc: int = 2, preferred_lang: Optional[str] = None) -> List[Dict[str, object]]:
    items, _, used_terms = search_with_fuzzy(conn, query, top_k=top_docs, fuzzy=True, preferred_lang=preferred_lang)
    if not items:
        return []
    chunk_map = active_chunks_for_docs(conn, [doc_id for doc_id, _ in items], per_doc=max(1, chunks_per_doc * 2))
    q_terms = set(tokenize(query, remove_stopwords=False))
    hits: List[Dict[str, object]] = []
    for rank, (doc_id, score) in enumerate(items, 1):
        for chunk in chunk_map.get(doc_id, []):
            c_terms = set(tokenize(chunk["content"], remove_stopwords=False))
            overlap = len(q_terms & c_terms) + len(set(used_terms) & c_terms)
            hit_score = float(score) + overlap + (1.0 / rank)
            hit = dict(chunk)
            hit["score"] = hit_score
            hits.append(hit)
    hits.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return hits[: max(1, top_docs * chunks_per_doc)]


def _openrouter_chat(messages: Sequence[Dict[str, str]], *, model: Optional[str] = None, max_tokens: int = 500, temperature: float = 0.2) -> Tuple[str, str, str]:
    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        return "", "OPENROUTER_API_KEY not configured", model or DEFAULT_TEACHER_MODEL
    if requests is None:
        return "", "requests not available", model or DEFAULT_TEACHER_MODEL
    used_model = model or DEFAULT_TEACHER_MODEL
    base = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    url = base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = (os.environ.get("OPENROUTER_HTTP_REFERER") or "").strip()
    title = (os.environ.get("OPENROUTER_X_TITLE") or "DonerSearch").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    payload = {
        "model": used_model,
        "messages": list(messages),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=45)
        if resp.status_code != 200:
            return "", f"OpenRouter HTTP {resp.status_code}", used_model
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return "", "OpenRouter returned no choices", used_model
        content = choices[0].get("message", {}).get("content", "")
        return (content or "").strip(), "", used_model
    except Exception as exc:  # pragma: no cover
        return "", f"OpenRouter request failed: {exc}", used_model


def grounded_answer(conn, query: str, *, preferred_lang: Optional[str] = None, top_docs: int = 5, chunks_per_doc: int = 2) -> Dict[str, object]:
    hits = source_hits(conn, query, top_docs=top_docs, chunks_per_doc=chunks_per_doc, preferred_lang=preferred_lang)
    if not hits:
        return {"answer": "Kaynak bulunamadi.", "sources": [], "provider": "none", "error": "no_sources"}

    context_lines = []
    sources = []
    for hit in hits:
        snippet = (hit["content"] or "")[:900]
        context_lines.append(f"[chunk:{hit['chunk_id']}] {hit['title'] or hit['url']}\nURL: {hit['url']}\n{snippet}")
        sources.append({
            "chunk_id": hit["chunk_id"],
            "doc_id": hit["doc_id"],
            "chunk_index": hit["chunk_index"],
            "title": hit["title"],
            "url": hit["url"],
            "score": hit["score"],
        })

    system = "Yalnizca verilen baglama dayanarak cevap ver. Bilinmeyen bir sey varsa bilmiyorum de. Kaynak uydurma. Turkce ve kisa cevap ver."
    user = "Soru: " + query.strip() + "\n\nBaglam:\n" + "\n\n".join(context_lines)
    answer, err, provider_model = _openrouter_chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], max_tokens=500, temperature=0.1)
    if not answer:
        fallback = []
        for hit in hits[: min(3, len(hits))]:
            snippet = (hit["content"] or "").strip()
            if snippet:
                fallback.append(snippet[:280])
        answer = "\n\n".join(fallback) if fallback else "Kaynak bulundu ama cevap uretilemedi."
        return {"answer": answer, "sources": sources, "provider": "extractive_fallback", "error": err}
    return {"answer": answer, "sources": sources, "provider": provider_model, "error": ""}


def _heuristic_distill_sample(chunk: Dict[str, object]) -> Dict[str, object]:
    title = (chunk.get("title") or "Bu icerik")
    text = (chunk.get("content") or "").strip()
    question = f"{title} hakkinda ne anlatiliyor?"
    answer = text[:700].strip()
    return {"question": question, "answer": answer}


def _teacher_distill_sample(
    chunk: Dict[str, object],
    *,
    teacher_model: str,
    prompt_version: str,
) -> Tuple[Dict[str, str], str, bool]:
    prompt = (
        "Asagidaki tek kaynak parcadan, yalnizca bu parcaya dayanan bir JSON uret. "
        "JSON anahtarlari: question, answer. Soru Turkce olsun ve cevap kaynaga sadik kalsin.\n\n"
        f"Baslik: {chunk.get('title') or ''}\n"
        f"URL: {chunk.get('url') or ''}\n"
        f"Icerik: {(chunk.get('content') or '')[:1600]}"
    )
    answer, err, used_model = _openrouter_chat(
        [
            {"role": "system", "content": "Sadece gecerli JSON dondur."},
            {"role": "user", "content": prompt},
        ],
        model=teacher_model,
        max_tokens=350,
        temperature=0.2,
    )
    if not answer:
        return _heuristic_distill_sample(chunk), err, True
    try:
        data = json.loads(answer)
        question = (data.get("question") or "").strip()
        final_answer = (data.get("answer") or "").strip()
        if question and final_answer:
            return {"question": question, "answer": final_answer}, used_model, False
    except Exception:
        pass
    return _heuristic_distill_sample(chunk), err or used_model, True


def generate_distill_samples(
    conn,
    *,
    dataset_version: str,
    limit: int = 50,
    task_type: str = "grounded_qa",
    teacher_model: Optional[str] = None,
    prompt_version: str = DEFAULT_GENERATION_PROMPT_VERSION,
) -> Dict[str, object]:
    ensure_ai_dirs()
    out_dir = _dataset_dir(dataset_version)
    sample_path = out_dir / "distill_samples.jsonl"
    teacher = teacher_model or DEFAULT_TEACHER_MODEL
    cur = conn.execute(
        "SELECT c.id, c.doc_id, c.chunk_index, c.content, p.url, IFNULL(p.title,''), IFNULL(p.language,'') "
        "FROM document_chunks c JOIN parsed_documents p ON p.id=c.parsed_document_id "
        "WHERE c.active=1 AND p.is_current=1 ORDER BY p.quality_score DESC, c.doc_id, c.chunk_index LIMIT ?",
        (int(limit),),
    )
    created = 0
    skipped = 0
    fallback_count = 0
    teacher_success_count = 0
    with sample_path.open("w", encoding="utf-8") as fp:
        for row in cur.fetchall():
            chunk = {
                "chunk_id": int(row[0]),
                "doc_id": int(row[1]),
                "chunk_index": int(row[2]),
                "content": row[3] or "",
                "url": row[4],
                "title": row[5] or "",
                "language": row[6] or "",
            }
            data, used_teacher, used_fallback = _teacher_distill_sample(
                chunk,
                teacher_model=teacher,
                prompt_version=prompt_version,
            )
            supporting_chunk_ids = [chunk["chunk_id"]]
            source_urls = [chunk["url"]]
            sample_hash = _sha(_json_dump({
                "dataset_version": dataset_version,
                "task_type": task_type,
                "question": data["question"],
                "answer": data["answer"],
                "supporting_chunk_ids": supporting_chunk_ids,
                "source_url_ids": source_urls,
            }))
            conn.execute(
                "INSERT OR IGNORE INTO distill_samples(dataset_version, task_type, question, answer, supporting_chunk_ids, source_url_ids, teacher_model, generation_prompt_version, status, sample_hash, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    dataset_version,
                    task_type,
                    data["question"],
                    data["answer"],
                    _json_dump(supporting_chunk_ids),
                    _json_dump(source_urls),
                    used_teacher or teacher,
                    prompt_version,
                    "ready",
                    sample_hash,
                    utc_now(),
                ),
            )
            inserted = int(conn.execute("SELECT changes()").fetchone()[0] or 0)
            if inserted:
                created += 1
            else:
                skipped += 1
            if used_fallback:
                fallback_count += 1
            else:
                teacher_success_count += 1
            fp.write(_json_dump({
                "question": data["question"],
                "answer": data["answer"],
                "supporting_chunk_ids": supporting_chunk_ids,
                "source_url_ids": source_urls,
                "teacher_model": used_teacher or teacher,
                "generation_prompt_version": prompt_version,
                "used_fallback": used_fallback,
            }) + "\n")
    conn.commit()
    return {
        "dataset_version": dataset_version,
        "created": created,
        "skipped": skipped,
        "fallback_count": fallback_count,
        "teacher_success_count": teacher_success_count,
        "path": str(sample_path),
        "teacher_model": teacher,
    }


def start_training_run(conn, *, run_type: str, dataset_version: str, model_name: str, model_version: str, status: str, config: Dict[str, object], artifact_path: str = "") -> int:
    cur = conn.execute(
        "INSERT INTO training_runs(run_type, dataset_version, model_name, model_version, status, config_json, artifact_path, metrics_json, started_at, finished_at) VALUES(?,?,?,?,?,?,?,?,?,NULL)",
        (run_type, dataset_version, model_name, model_version, status, _json_dump(config), artifact_path, "{}", utc_now()),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_training_run(conn, run_id: int, *, status: str, metrics: Optional[Dict[str, object]] = None) -> None:
    conn.execute(
        "UPDATE training_runs SET status=?, metrics_json=?, finished_at=? WHERE id=?",
        (status, _json_dump(metrics or {}), utc_now(), int(run_id)),
    )
    conn.commit()


def train_student(
    conn,
    *,
    dataset_version: str,
    model_name: str,
    model_version: str,
    base_params: int = 80000000,
    context_window: int = 512,
    precision: str = "fp16",
) -> Dict[str, object]:
    ensure_ai_dirs()
    model_dir = MODELS_ROOT / model_name / model_version
    model_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_version": dataset_version,
        "model_name": model_name,
        "model_version": model_version,
        "base_params": int(base_params),
        "context_window": int(context_window),
        "precision": precision,
        "recommended_runtime": "single_gpu_rtx_3060",
        "phases": ["pretrain", "distill", "grounded_tune"],
    }
    run_id = start_training_run(
        conn,
        run_type="train_student",
        dataset_version=dataset_version,
        model_name=model_name,
        model_version=model_version,
        status="prepared",
        config=config,
        artifact_path=str(model_dir),
    )
    (model_dir / "training_manifest.json").write_text(_json_dump({"run_id": run_id, **config}), encoding="utf-8")
    return {"run_id": run_id, "artifact_path": str(model_dir), "status": "prepared"}


def evaluate_model(conn, *, model_name: str, model_version: str, dataset_version: str, benchmark_name: str = "grounded_qa_smoke") -> Dict[str, object]:
    cur = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT sample_hash) FROM distill_samples WHERE dataset_version=?",
        (dataset_version,),
    )
    row = cur.fetchone() or (0, 0)
    metrics = {
        "sample_count": int(row[0] or 0),
        "unique_sample_count": int(row[1] or 0),
        "benchmark_name": benchmark_name,
    }
    conn.execute(
        "INSERT INTO eval_runs(model_name, model_version, dataset_version, benchmark_name, metrics_json, created_at) VALUES(?,?,?,?,?,?)",
        (model_name, model_version, dataset_version, benchmark_name, _json_dump(metrics), utc_now()),
    )
    conn.commit()
    return metrics


def publish_model(conn, *, model_name: str, model_version: str, provider: str = "local", artifact_path: str = "", config: Optional[Dict[str, object]] = None, activate: bool = True) -> Dict[str, object]:
    now = utc_now()
    if activate:
        conn.execute("UPDATE model_registry SET is_active=0 WHERE model_name=?", (model_name,))
    conn.execute(
        "INSERT INTO model_registry(model_name, model_version, provider, artifact_path, config_json, is_active, created_at) VALUES(?,?,?,?,?,?,?) "
        "ON CONFLICT(model_name, model_version) DO UPDATE SET provider=excluded.provider, artifact_path=excluded.artifact_path, config_json=excluded.config_json, is_active=excluded.is_active",
        (model_name, model_version, provider, artifact_path, _json_dump(config or {}), 1 if activate else 0, now),
    )
    conn.commit()
    return {"model_name": model_name, "model_version": model_version, "provider": provider, "is_active": bool(activate)}


def list_models(conn) -> List[Dict[str, object]]:
    cur = conn.execute(
        "SELECT model_name, model_version, provider, IFNULL(artifact_path,''), IFNULL(config_json,'{}'), is_active, created_at FROM model_registry ORDER BY model_name, created_at DESC"
    )
    items: List[Dict[str, object]] = []
    for row in cur.fetchall():
        try:
            config = json.loads(row[4] or "{}")
        except Exception:
            config = {}
        items.append({
            "model_name": row[0],
            "model_version": row[1],
            "provider": row[2],
            "artifact_path": row[3] or "",
            "config": config,
            "is_active": bool(row[5]),
            "created_at": row[6],
        })
    return items
