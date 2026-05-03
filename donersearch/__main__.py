from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

from . import ai_platform as aimod
from . import db as dbmod
from . import smoke as smokemod
from .crawler import crawl, crawl_loop, USER_AGENT as DEFAULT_USER_AGENT, MAX_AUTO_DISCOVER_PER_HOST
from .indexer import reindex_missing
from .search import search_with_fuzzy
from .web import serve


def _default_version(prefix: str) -> str:
    return f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"


def cmd_crawl(args: argparse.Namespace) -> int:
    seeds = [s if "://" in s else ("https://" + s) for s in args.seeds]
    extra_seeds = []
    if getattr(args, "auto_seeds", None):
        for host in args.auto_seeds:
            if host:
                extra_seeds.append(host if "://" in host else ("https://" + host))
    if extra_seeds:
        seeds.extend(extra_seeds)
    if not seeds:
        print("En az bir seed URL gerekli.")
        return 2
    if getattr(args, "daily", False):
        args.loop = True
        args.revisit_hours = 24.0
    if getattr(args, "preset", None):
        preset = args.preset
        if preset == "default":
            args.auto_discover = True
            args.delay = 0.7
            args.max_depth = 2
            args.max_pages = 200
            args.revisit_hours = 24.0
            args.retry_hours = 12.0
            args.max_auto_discover_per_host = 5
            args.headless = True
            args.render = False
            args.workers = max(getattr(args, "workers", 1) or 1, 2)
            if getattr(args, "max_host_pages", None) is None:
                args.max_host_pages = 50
        elif preset == "fast":
            args.auto_discover = False
            args.delay = 0.3
            args.max_depth = 1
            args.max_pages = 80
            args.render = False
            args.headless = True
            args.workers = max(getattr(args, "workers", 1) or 1, 4)
        elif preset == "deep":
            args.auto_discover = True
            args.delay = 1.0
            args.max_depth = 3
            args.max_pages = 500
            args.revisit_hours = 12.0
            args.retry_hours = 6.0
            args.max_auto_discover_per_host = 10
            args.render = False
            args.headless = True
            args.workers = max(getattr(args, "workers", 1) or 1, 4)
    common_kwargs = dict(
        allowed_domains=args.allowed_domains,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        delay_seconds=args.delay,
        auto_discover=args.auto_discover,
        max_auto_discover_per_host=args.max_auto_discover_per_host,
        manual_sitemaps=args.sitemap,
        user_agent=args.user_agent,
        render=args.render,
        revisit_interval_hours=args.revisit_hours,
        render_pool_size=args.render_pool_size,
        render_headless=args.headless,
        failed_host_retry_hours=args.retry_hours,
        max_pages_per_host=args.max_host_pages,
        workers=getattr(args, "workers", 1),
    )
    if args.loop:
        crawl_loop(
            args.db,
            seeds,
            cycle_sleep=args.cycle_sleep,
            max_cycles=args.max_cycles,
            **common_kwargs,
        )
        return 0

    conn = dbmod.open_db(args.db)
    dbmod.ensure_schema(conn)
    crawl_run_id = aimod.start_crawl_run(conn, seeds, notes="cli crawl")
    conn.close()
    try:
        count = crawl(db_path=args.db, seed_urls=seeds, crawl_run_id=crawl_run_id, **common_kwargs)
    except Exception as exc:
        conn = dbmod.open_db(args.db)
        dbmod.ensure_schema(conn)
        aimod.finish_crawl_run(conn, crawl_run_id, status="failed", notes=str(exc))
        conn.close()
        raise
    conn = dbmod.open_db(args.db)
    dbmod.ensure_schema(conn)
    aimod.finish_crawl_run(conn, crawl_run_id, status="completed", notes=f"pages_indexed={count}")
    conn.close()
    print(f"Indekslenen sayfa: {count}")
    print(f"Crawl run id: {crawl_run_id}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    conn = dbmod.open_db(args.db)
    dbmod.ensure_schema(conn)
    items, corrections, _used_terms = search_with_fuzzy(
        conn,
        args.query,
        top_k=args.top_k,
        fuzzy=(not args.no_fuzzy),
        preferred_lang=args.lang,
    )
    if corrections:
        pairs = ", ".join(f"{k}->{v}" for k, v in corrections.items())
        print(f"Duzeltme: {pairs}")
    images_map = {}
    if getattr(args, "images", False) and items:
        doc_ids = [doc_id for doc_id, _ in items]
        images_map = dbmod.get_images_for_docs(conn, doc_ids)
    for i, (doc_id, score) in enumerate(items, 1):
        _, url, title, _content, _, doc_lang = dbmod.get_doc(conn, doc_id)
        title = title or url
        print(f"{i:2d}. {title}  ({score:.3f})\n    {url}")
        if getattr(args, "images", False):
            img_entries = []
            for info in images_map.get(doc_id) or []:
                img_entries.append({
                    "url": info.get("url"),
                    "alt": info.get("alt"),
                    "format": info.get("format"),
                    "width": info.get("width"),
                    "height": info.get("height"),
                    "size_bytes": info.get("size_bytes"),
                    "aspect_ratio": info.get("aspect_ratio"),
                    "file_path": info.get("file_path"),
                    "thumbnail_path": info.get("thumbnail_path"),
                    "source": url,
                })
            if img_entries:
                print(json.dumps(img_entries, ensure_ascii=False, indent=2))
        if args.lang:
            print(f"    Dil: {doc_lang or 'bilinmiyor'}")
    conn.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    if getattr(args, "mistral_key", None):
        os.environ["MISTRAL_API_KEY"] = args.mistral_key
    if getattr(args, "ai_model", None):
        os.environ["MISTRAL_MODEL"] = args.ai_model
    if getattr(args, "ai_base", None):
        os.environ["MISTRAL_API_BASE"] = args.ai_base
    serve(args.db, host=args.host, port=args.port)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    conn = dbmod.open_db(args.db)
    dbmod.ensure_schema(conn)
    docs = dbmod.doc_count(conn)
    terms = dbmod.term_count(conn)
    avg = dbmod.get_avg_doc_len(conn)
    snapshots = conn.execute("SELECT COUNT(*) FROM raw_snapshots").fetchone()[0]
    parsed = conn.execute("SELECT COUNT(*) FROM parsed_documents WHERE is_current=1").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM document_chunks WHERE active=1").fetchone()[0]
    embeddings = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    print(f"Documents: {docs}")
    print(f"Terms: {terms}")
    print(f"Avg doc len: {avg:.2f}")
    print(f"Raw snapshots: {snapshots}")
    print(f"Current parsed docs: {parsed}")
    print(f"Active chunks: {chunks}")
    print(f"Embeddings: {embeddings}")
    conn.close()
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    conn = dbmod.open_db(args.db)
    dbmod.ensure_schema(conn)
    n = reindex_missing(conn)
    print(f"Yeniden indekslenen belge: {n}")
    conn.close()
    return 0


def cmd_repair(args: argparse.Namespace) -> int:
    conn = dbmod.open_db(args.db)
    try:
        conn.execute("DROP TABLE IF EXISTS postings")
        conn.execute("DROP TABLE IF EXISTS terms")
        conn.execute("DROP TABLE IF EXISTS images")
        conn.commit()
        dbmod.ensure_schema(conn)
        conn.execute("PRAGMA foreign_keys=OFF")
        fixed = reindex_missing(conn)
        conn.commit()
        conn.execute("DELETE FROM postings WHERE doc_id NOT IN (SELECT id FROM documents)")
        conn.execute("DELETE FROM images WHERE doc_id NOT IN (SELECT id FROM documents)")
        conn.commit()
        conn.execute("PRAGMA foreign_keys=ON")
        print(f"Onarim tamamlandi. Yeniden indekslenen belge: {fixed}")
        return 0
    finally:
        conn.close()


def cmd_delete(args: argparse.Namespace) -> int:
    conn = dbmod.open_db(args.db)
    dbmod.ensure_schema(conn)
    deleted = 0
    try:
        for url in args.urls:
            if dbmod.delete_document_by_url(conn, url):
                deleted += 1
        conn.commit()
    finally:
        conn.close()
    print(f"Silinen belge: {deleted}")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    conn = dbmod.open_db(args.db)
    try:
        conn.execute("ANALYZE")
        conn.execute("PRAGMA optimize")
        conn.commit()
        if getattr(args, "vacuum", False):
            conn.execute("VACUUM")
        print("Optimize tamamlandi.")
        return 0
    finally:
        conn.close()


def cmd_prepare_dataset(args: argparse.Namespace) -> int:
    conn = dbmod.open_db(args.db)
    dbmod.ensure_schema(conn)
    backfilled = aimod.backfill_pipeline(conn, limit=args.limit)
    manifest = aimod.prepare_dataset(conn, dataset_version=args.dataset_version, min_quality=args.min_quality)
    conn.close()
    print(f"Backfilled docs: {backfilled}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def cmd_chunk_documents(args: argparse.Namespace) -> int:
    conn = dbmod.open_db(args.db)
    dbmod.ensure_schema(conn)
    count = aimod.chunk_documents(
        conn,
        limit=args.limit,
        max_chunk_tokens=args.max_chunk_tokens,
        chunk_overlap=args.chunk_overlap,
    )
    conn.close()
    print(f"Chunked docs: {count}")
    return 0


def cmd_build_embeddings(args: argparse.Namespace) -> int:
    conn = dbmod.open_db(args.db)
    dbmod.ensure_schema(conn)
    result = aimod.build_embeddings(conn, limit=args.limit, only_missing=(not args.force), model_name=args.model_name)
    conn.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_generate_distill_samples(args: argparse.Namespace) -> int:
    conn = dbmod.open_db(args.db)
    dbmod.ensure_schema(conn)
    result = aimod.generate_distill_samples(
        conn,
        dataset_version=args.dataset_version,
        limit=args.limit,
        task_type=args.task_type,
        teacher_model=args.teacher_model,
        prompt_version=args.prompt_version,
    )
    conn.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_train_student(args: argparse.Namespace) -> int:
    conn = dbmod.open_db(args.db)
    dbmod.ensure_schema(conn)
    result = aimod.train_student(
        conn,
        dataset_version=args.dataset_version,
        model_name=args.model_name,
        model_version=args.model_version,
        base_params=args.base_params,
        context_window=args.context_window,
        precision=args.precision,
    )
    conn.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_eval_model(args: argparse.Namespace) -> int:
    conn = dbmod.open_db(args.db)
    dbmod.ensure_schema(conn)
    result = aimod.evaluate_model(
        conn,
        model_name=args.model_name,
        model_version=args.model_version,
        dataset_version=args.dataset_version,
        benchmark_name=args.benchmark_name,
    )
    conn.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_publish_model(args: argparse.Namespace) -> int:
    conn = dbmod.open_db(args.db)
    dbmod.ensure_schema(conn)
    config = {}
    if args.config_json:
        config = json.loads(args.config_json)
    result = aimod.publish_model(
        conn,
        model_name=args.model_name,
        model_version=args.model_version,
        provider=args.provider,
        artifact_path=args.artifact_path,
        config=config,
        activate=(not args.no_activate),
    )
    conn.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_smoke_e2e(args: argparse.Namespace) -> int:
    result = smokemod.run_smoke_e2e(
        seed_url=args.seed_url,
        query=args.query,
        db_path=args.db,
        teacher_model=args.teacher_model,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        distill_limit=args.distill_limit,
        min_quality=args.min_quality,
        user_agent=args.user_agent,
        allow_answer_fallback=args.allow_answer_fallback,
        allow_distill_fallback=args.allow_distill_fallback,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="donersearch", description="Basit crawler'li arama motoru")
    p.add_argument("--db", default="donersearch.db", help="SQLite veritabani yolu")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("crawl", help="Seed URL'lerden tarama yap ve indeksle")
    pc.add_argument("--db", default="donersearch.db", help="SQLite veritabani yolu")
    pc.add_argument("seeds", nargs="+", help="Seed URL(ler)")
    pc.add_argument("--allowed-domains", nargs="*", default=None, help="Sadece bu domain(ler)")
    pc.add_argument("--max-pages", type=int, default=100, help="Maksimum sayfa sayisi")
    pc.add_argument("--max-depth", type=int, default=2, help="Maksimum derinlik")
    pc.add_argument("--delay", type=float, default=0.5, help="Istekler arasi bekleme (sn)")
    pc.add_argument("--auto-discover", action="store_true", help="Dis domainleri otomatik kesfet ve kuyruga al")
    pc.add_argument("--sitemap", nargs="*", default=None, help="Manuel sitemap URL(leri)")
    pc.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="HTTP isteklerinde kullanilacak User-Agent")
    pc.add_argument("--render", action="store_true", help="JS ile icerik yukleyen sayfalari Playwright ile render et")
    pc.add_argument("--max-auto-discover-per-host", type=int, default=MAX_AUTO_DISCOVER_PER_HOST)
    pc.add_argument("--render-pool-size", type=int, default=5)
    pc.add_argument("--headless", action="store_true")
    pc.add_argument("--revisit-hours", type=float, default=24.0)
    pc.add_argument("--retry-hours", type=float, default=12.0)
    pc.add_argument("--max-host-pages", type=int)
    pc.add_argument("--loop", action="store_true")
    pc.add_argument("--cycle-sleep", type=float, default=900.0)
    pc.add_argument("--max-cycles", type=int)
    pc.add_argument("--workers", type=int, default=1)
    pc.add_argument("--daily", action="store_true")
    pc.add_argument("--auto-seeds", nargs="*", default=None)
    pc.add_argument("--preset", choices=["default", "fast", "deep"])
    pc.set_defaults(func=cmd_crawl)

    ps = sub.add_parser("search", help="Komut satirindan ara")
    ps.add_argument("--db", default="donersearch.db", help="SQLite veritabani yolu")
    ps.add_argument("query", help="Sorgu")
    ps.add_argument("--top-k", type=int, default=10)
    ps.add_argument("--no-fuzzy", action="store_true")
    ps.add_argument("--lang", help="Tercih edilen dil (or. tr)")
    ps.add_argument("--images", action="store_true")
    ps.set_defaults(func=cmd_search)

    pv = sub.add_parser("serve", help="Web arayuzunu baslat")
    pv.add_argument("--db", default="donersearch.db", help="SQLite veritabani yolu")
    pv.add_argument("--host", default="127.0.0.1")
    pv.add_argument("--port", type=int, default=8000)
    pv.add_argument("--mistral-key")
    pv.add_argument("--ai-model", default="mistral-small-latest")
    pv.add_argument("--ai-base")
    pv.set_defaults(func=cmd_serve)

    pt = sub.add_parser("status", help="Indeks durumu")
    pt.add_argument("--db", default="donersearch.db", help="SQLite veritabani yolu")
    pt.set_defaults(func=cmd_status)

    pr = sub.add_parser("reindex", help="Postings eksikse yeniden indeksle")
    pr.add_argument("--db", default="donersearch.db", help="SQLite veritabani yolu")
    pr.set_defaults(func=cmd_reindex)

    pm = sub.add_parser("repair", help="Eski DB'de index tablolarini sifirla ve yeniden kur")
    pm.add_argument("--db", default="donersearch.db", help="SQLite veritabani yolu")
    pm.set_defaults(func=cmd_repair)

    pd = sub.add_parser("delete", help="Belirtilen URL(leri) indeksden sil")
    pd.add_argument("urls", nargs="+")
    pd.add_argument("--db", default="donersearch.db", help="SQLite veritabani yolu")
    pd.set_defaults(func=cmd_delete)

    po = sub.add_parser("optimize", help="ANALYZE/PRAGMA optimize uygula")
    po.add_argument("--db", default="donersearch.db", help="SQLite veritabani yolu")
    po.add_argument("--vacuum", action="store_true")
    po.set_defaults(func=cmd_optimize)

    ppd = sub.add_parser("prepare-dataset", help="Current parsed docs ve chunks'tan dataset olustur")
    ppd.add_argument("--db", default="donersearch.db", help="SQLite veritabani yolu")
    ppd.add_argument("--dataset-version", default=_default_version("dataset"))
    ppd.add_argument("--min-quality", type=float, default=0.15)
    ppd.add_argument("--limit", type=int)
    ppd.set_defaults(func=cmd_prepare_dataset)

    pcd = sub.add_parser("chunk-documents", help="Belgeleri parsed/chunk zincirine backfill et")
    pcd.add_argument("--db", default="donersearch.db", help="SQLite veritabani yolu")
    pcd.add_argument("--limit", type=int)
    pcd.add_argument("--max-chunk-tokens", type=int, default=160)
    pcd.add_argument("--chunk-overlap", type=int, default=30)
    pcd.set_defaults(func=cmd_chunk_documents)

    pbe = sub.add_parser("build-embeddings", help="Eksik veya stale document embeddings uret")
    pbe.add_argument("--db", default="donersearch.db", help="SQLite veritabani yolu")
    pbe.add_argument("--limit", type=int)
    pbe.add_argument("--force", action="store_true", help="Tum belgeleri yeniden embed et")
    pbe.add_argument("--model-name", help="Embeddings model etiketi")
    pbe.set_defaults(func=cmd_build_embeddings)

    pds = sub.add_parser("generate-distill-samples", help="Teacher veya fallback ile grounded distillation dataset uret")
    pds.add_argument("--db", default="donersearch.db", help="SQLite veritabani yolu")
    pds.add_argument("--dataset-version", default=_default_version("distill"))
    pds.add_argument("--limit", type=int, default=50)
    pds.add_argument("--task-type", default="grounded_qa")
    pds.add_argument("--teacher-model")
    pds.add_argument("--prompt-version", default=aimod.DEFAULT_GENERATION_PROMPT_VERSION)
    pds.set_defaults(func=cmd_generate_distill_samples)

    pts = sub.add_parser("train-student", help="Student model icin training manifest ve run kaydi olustur")
    pts.add_argument("--db", default="donersearch.db", help="SQLite veritabani yolu")
    pts.add_argument("--dataset-version", required=True)
    pts.add_argument("--model-name", default="student")
    pts.add_argument("--model-version", default=_default_version("model"))
    pts.add_argument("--base-params", type=int, default=80000000)
    pts.add_argument("--context-window", type=int, default=512)
    pts.add_argument("--precision", default="fp16")
    pts.set_defaults(func=cmd_train_student)

    pem = sub.add_parser("eval-model", help="Eval kaydi olustur")
    pem.add_argument("--db", default="donersearch.db", help="SQLite veritabani yolu")
    pem.add_argument("--dataset-version", required=True)
    pem.add_argument("--model-name", required=True)
    pem.add_argument("--model-version", required=True)
    pem.add_argument("--benchmark-name", default="grounded_qa_smoke")
    pem.set_defaults(func=cmd_eval_model)

    ppm = sub.add_parser("publish-model", help="Model registry'ye model kaydet")
    ppm.add_argument("--db", default="donersearch.db", help="SQLite veritabani yolu")
    ppm.add_argument("--model-name", required=True)
    ppm.add_argument("--model-version", required=True)
    ppm.add_argument("--provider", default="local")
    ppm.add_argument("--artifact-path", default="")
    ppm.add_argument("--config-json")
    ppm.add_argument("--no-activate", action="store_true")
    ppm.set_defaults(func=cmd_publish_model)

    pse = sub.add_parser("smoke-e2e", help="Canli crawl + teacher + API zincirini tek akista dogrula")
    pse.add_argument("--db", help="SQLite veritabani yolu; verilmezse gecici DB kullanilir")
    pse.add_argument("--seed-url", required=True, help="Canli crawl icin seed URL")
    pse.add_argument("--query", required=True, help="Arama ve answer smoke kontrolunde kullanilacak sorgu")
    pse.add_argument("--teacher-model", help="OpenRouter teacher modeli")
    pse.add_argument("--max-pages", type=int, default=5)
    pse.add_argument("--max-depth", type=int, default=1)
    pse.add_argument("--distill-limit", type=int, default=5)
    pse.add_argument("--min-quality", type=float, default=0.0)
    pse.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    pse.add_argument("--allow-answer-fallback", action="store_true", help="Teacher cevap vermezse fallback'i hata sayma")
    pse.add_argument("--allow-distill-fallback", action="store_true", help="Teacher distill fallback'ini hata sayma")
    pse.set_defaults(func=cmd_smoke_e2e)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = build_parser()
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
