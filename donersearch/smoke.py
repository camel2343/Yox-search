from __future__ import annotations

import io
import json
import os
import tempfile
from collections.abc import Iterable
from datetime import datetime
from typing import Dict, Optional, Tuple
from urllib.parse import urlencode
from wsgiref.util import setup_testing_defaults

from . import ai_platform as aimod
from . import db as dbmod
from .crawler import USER_AGENT as DEFAULT_USER_AGENT, crawl
from .web import app_factory


def _default_version(prefix: str) -> str:
    return f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"


def _normalize_seed(seed_url: str) -> str:
    seed = (seed_url or "").strip()
    if not seed:
        raise ValueError("seed_url is required")
    return seed if "://" in seed else ("https://" + seed)


def _counts(conn) -> Dict[str, int]:
    return {
        "documents": int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] or 0),
        "raw_snapshots": int(conn.execute("SELECT COUNT(*) FROM raw_snapshots").fetchone()[0] or 0),
        "current_parsed_documents": int(
            conn.execute("SELECT COUNT(*) FROM parsed_documents WHERE is_current=1").fetchone()[0] or 0
        ),
        "active_chunks": int(conn.execute("SELECT COUNT(*) FROM document_chunks WHERE active=1").fetchone()[0] or 0),
        "embeddings": int(conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] or 0),
        "distill_samples": int(conn.execute("SELECT COUNT(*) FROM distill_samples").fetchone()[0] or 0),
        "training_runs": int(conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0] or 0),
        "eval_runs": int(conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0] or 0),
        "model_registry": int(conn.execute("SELECT COUNT(*) FROM model_registry").fetchone()[0] or 0),
    }


def _current_doc_payload(conn) -> Tuple[int, str, str, str, str, str]:
    row = conn.execute(
        "SELECT d.id, p.url, IFNULL(p.title,''), IFNULL(p.content,''), IFNULL(p.language,''), p.content_hash "
        "FROM documents d JOIN parsed_documents p ON p.doc_id=d.id "
        "WHERE p.is_current=1 ORDER BY d.id LIMIT 1"
    ).fetchone()
    if not row:
        raise RuntimeError("No current parsed document available after crawl")
    return int(row[0]), row[1], row[2] or "", row[3] or "", row[4] or "", row[5] or ""


def _wsgi_json(app, path: str, params: Optional[Dict[str, object]] = None) -> Tuple[int, Dict[str, object]]:
    query_string = urlencode({k: v for k, v in (params or {}).items() if v is not None}, doseq=True)
    environ: Dict[str, object] = {}
    setup_testing_defaults(environ)
    environ["REQUEST_METHOD"] = "GET"
    environ["PATH_INFO"] = path
    environ["QUERY_STRING"] = query_string
    environ["wsgi.input"] = io.BytesIO(b"")
    captured: Dict[str, object] = {"status": "500 Internal Server Error", "headers": []}

    def start_response(status, headers, exc_info=None):
        captured["status"] = status
        captured["headers"] = headers

    body = b"".join(app(environ, start_response))
    status_code = int(str(captured["status"]).split()[0])
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"{path} did not return JSON: {exc}") from exc
    return status_code, payload


def run_smoke_e2e(
    *,
    seed_url: str,
    query: str,
    db_path: Optional[str] = None,
    teacher_model: Optional[str] = None,
    max_pages: int = 5,
    max_depth: int = 1,
    distill_limit: int = 5,
    min_quality: float = 0.0,
    user_agent: str = DEFAULT_USER_AGENT,
    allow_answer_fallback: bool = False,
    allow_distill_fallback: bool = False,
) -> Dict[str, object]:
    if not (query or "").strip():
        raise ValueError("query is required")
    if not allow_answer_fallback or not allow_distill_fallback:
        if not (os.environ.get("OPENROUTER_API_KEY") or "").strip():
            raise RuntimeError("OPENROUTER_API_KEY is required for smoke-e2e live validation")
        if aimod.requests is None:
            raise RuntimeError("requests module is required for smoke-e2e live validation")

    temporary_db = False
    if not db_path:
        fd, temp_path = tempfile.mkstemp(prefix="donersearch-smoke-", suffix=".db")
        os.close(fd)
        db_path = temp_path
        temporary_db = True

    seed = _normalize_seed(seed_url)
    dataset_version = _default_version("smoke_dataset")
    distill_version = _default_version("smoke_distill")
    model_name = "smoke-student"
    model_version = _default_version("smoke_model")

    result: Dict[str, object] = {
        "db_path": db_path,
        "temporary_db": temporary_db,
        "seed_url": seed,
        "query": query,
        "dataset_version": dataset_version,
        "distill_version": distill_version,
        "model_name": model_name,
        "model_version": model_version,
        "steps": {},
    }

    conn = dbmod.open_db(db_path)
    dbmod.ensure_schema(conn)
    crawl_run_id = aimod.start_crawl_run(conn, [seed], notes="smoke-e2e")
    conn.close()

    try:
        pages_indexed = crawl(
            db_path=db_path,
            seed_urls=[seed],
            allowed_domains=None,
            max_pages=max_pages,
            max_depth=max_depth,
            delay_seconds=0.5,
            auto_discover=False,
            user_agent=user_agent,
            render=False,
            revisit_interval_hours=24.0,
            failed_host_retry_hours=12.0,
            workers=1,
            crawl_run_id=crawl_run_id,
        )
    except Exception as exc:
        conn = dbmod.open_db(db_path)
        dbmod.ensure_schema(conn)
        aimod.finish_crawl_run(conn, crawl_run_id, status="failed", notes=str(exc))
        conn.close()
        raise

    conn = dbmod.open_db(db_path)
    dbmod.ensure_schema(conn)
    aimod.finish_crawl_run(conn, crawl_run_id, status="completed", notes=f"pages_indexed={pages_indexed}")
    counts_after_crawl = _counts(conn)
    if pages_indexed < 1 or counts_after_crawl["documents"] < 1:
        raise RuntimeError("Crawl completed without indexing any documents")
    if counts_after_crawl["raw_snapshots"] < 1 or counts_after_crawl["active_chunks"] < 1:
        raise RuntimeError("Crawl did not populate the AI pipeline tables")
    result["steps"]["crawl"] = {
        "crawl_run_id": crawl_run_id,
        "pages_indexed": pages_indexed,
        "counts": counts_after_crawl,
    }

    doc_id, url, title, content, language, content_hash = _current_doc_payload(conn)
    before_resync = _counts(conn)
    resync = aimod.sync_document_pipeline(
        conn,
        doc_id=doc_id,
        url=url,
        title=title,
        raw_html="",
        content=content,
        language=language,
        content_hash=content_hash,
        crawl_run_id=crawl_run_id,
    )
    after_resync = _counts(conn)
    if resync["changed"]:
        raise RuntimeError("Pipeline re-sync changed records for identical content")
    for key in ("raw_snapshots", "current_parsed_documents", "active_chunks"):
        if before_resync[key] != after_resync[key]:
            raise RuntimeError(f"Pipeline dedupe failed for {key}")
    result["steps"]["dedupe"] = {
        "doc_id": doc_id,
        "resync": resync,
        "counts_before": before_resync,
        "counts_after": after_resync,
    }

    dataset_manifest = aimod.prepare_dataset(conn, dataset_version=dataset_version, min_quality=min_quality)
    if int(dataset_manifest["document_count"] or 0) < 1 or int(dataset_manifest["chunk_count"] or 0) < 1:
        raise RuntimeError("prepare_dataset did not produce documents and chunks")
    result["steps"]["dataset"] = dataset_manifest

    embedding_first = aimod.build_embeddings(conn, only_missing=True)
    embedding_second = aimod.build_embeddings(conn, only_missing=True)
    if int(embedding_first["processed"] or 0) < 1:
        raise RuntimeError("build_embeddings did not process any documents")
    if int(embedding_second["processed"] or 0) != 0:
        raise RuntimeError("build_embeddings reprocessed unchanged documents")
    stale_embeddings = int(
        conn.execute(
            "SELECT COUNT(*) FROM embeddings e JOIN documents d ON d.id=e.doc_id "
            "WHERE IFNULL(e.source_hash,'') <> IFNULL(d.content_hash,'')"
        ).fetchone()[0]
        or 0
    )
    if stale_embeddings != 0:
        raise RuntimeError("Embedding source hashes are stale")
    result["steps"]["embeddings"] = {
        "first_run": embedding_first,
        "second_run": embedding_second,
        "stale_embeddings": stale_embeddings,
    }

    distill_first = aimod.generate_distill_samples(
        conn,
        dataset_version=distill_version,
        limit=distill_limit,
        teacher_model=teacher_model,
    )
    distill_count_after_first = int(
        conn.execute("SELECT COUNT(*) FROM distill_samples WHERE dataset_version=?", (distill_version,)).fetchone()[0] or 0
    )
    if distill_count_after_first < 1:
        raise RuntimeError("generate_distill_samples did not create any rows")
    if not allow_distill_fallback and int(distill_first["teacher_success_count"] or 0) < 1:
        raise RuntimeError("Distill sample generation fell back instead of using the teacher")
    distill_second = aimod.generate_distill_samples(
        conn,
        dataset_version=distill_version,
        limit=distill_limit,
        teacher_model=teacher_model,
    )
    distill_count_after_second = int(
        conn.execute("SELECT COUNT(*) FROM distill_samples WHERE dataset_version=?", (distill_version,)).fetchone()[0] or 0
    )
    if distill_count_after_second != distill_count_after_first:
        raise RuntimeError("Distill sample dedupe failed across repeated generation")
    result["steps"]["distill"] = {
        "first_run": distill_first,
        "second_run": distill_second,
        "db_count": distill_count_after_second,
    }

    train_result = aimod.train_student(
        conn,
        dataset_version=distill_version,
        model_name=model_name,
        model_version=model_version,
    )
    eval_result = aimod.evaluate_model(
        conn,
        model_name=model_name,
        model_version=model_version,
        dataset_version=distill_version,
    )
    publish_result = aimod.publish_model(
        conn,
        model_name=model_name,
        model_version=model_version,
        provider="local",
        artifact_path=str(train_result["artifact_path"]),
        config={"smoke": True, "dataset_version": distill_version},
        activate=True,
    )
    result["steps"]["train"] = train_result
    result["steps"]["eval"] = eval_result
    result["steps"]["publish"] = publish_result

    conn.close()

    app = app_factory(db_path)
    api_checks: Dict[str, object] = {}
    for endpoint, params in (
        ("/api/search", {"q": query}),
        ("/api/sources", {"q": query}),
        ("/api/answer", {"q": query}),
        ("/api/models", None),
    ):
        status_code, payload = _wsgi_json(app, endpoint, params)
        if status_code != 200:
            raise RuntimeError(f"{endpoint} returned HTTP {status_code}")
        api_checks[endpoint] = payload

    search_payload = api_checks["/api/search"]
    if not isinstance(search_payload, dict) or not search_payload.get("results"):
        raise RuntimeError("/api/search returned no results")
    sources_payload = api_checks["/api/sources"]
    if not isinstance(sources_payload, dict) or not sources_payload.get("sources"):
        raise RuntimeError("/api/sources returned no sources")
    answer_payload = api_checks["/api/answer"]
    provider = ""
    if isinstance(answer_payload, dict):
        provider = str(answer_payload.get("provider") or "")
    if not isinstance(answer_payload, dict) or not (answer_payload.get("answer") or "").strip():
        raise RuntimeError("/api/answer returned an empty answer")
    if not allow_answer_fallback and provider in ("", "extractive_fallback", "none"):
        raise RuntimeError("/api/answer fell back instead of using the teacher")
    models_payload = api_checks["/api/models"]
    models = models_payload.get("models") if isinstance(models_payload, dict) else None
    if not isinstance(models, Iterable):
        raise RuntimeError("/api/models returned an invalid payload")
    models_list = list(models)
    if not any(
        isinstance(item, dict)
        and item.get("model_name") == model_name
        and item.get("model_version") == model_version
        and item.get("is_active")
        for item in models_list
    ):
        raise RuntimeError("Published model was not visible via /api/models")

    result["steps"]["api_checks"] = {
        "search_result_count": len(search_payload["results"]),
        "source_count": len(sources_payload["sources"]),
        "answer_provider": provider,
        "model_count": len(models_list),
    }
    result["status"] = "ok"
    return result
