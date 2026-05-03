from __future__ import annotations

import math
import difflib
import os
from datetime import datetime, timezone
import time
from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence, Tuple, Mapping, Set, Dict as TDict

from . import db as dbmod
from .tokenize import tokenize

try:
    import requests  # type: ignore
except Exception:  # optional dependency
    requests = None

_VOCAB_CACHE = {"count": -1, "vocab": []}

# Small in-process LRU cache for document lengths to avoid repeated DB hits
_DOC_LEN_CACHE: "OrderedDict[int, int]" = OrderedDict()
_DOC_LEN_CACHE_MAX = 50000

# Query expansion cache
_QE_CACHE: Dict[Tuple[str, str], List[str]] = {}
# Negative cache with TTL to avoid hammering APIs on failures (e.g., 429)
_QE_NEG_CACHE: Dict[Tuple[str, str], float] = {}

# Cached k-gram index for faster fuzzy matching over large vocabularies.
_KGRAM_DATA_CACHE = {
    "count": -1,  # term table row count snapshot
    "k": 3,
    "index": {},      # gram -> set(terms)
    "termgrams": {},  # term -> set(grams)
}

_DEFAULT_K = 3


def bm25_scores_terms(
    conn,
    terms: Sequence[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    top_k: int = 10,
    max_df_ratio: float | None = 0.6,
    preferred_lang: str | None = None,
    lang_boost: float = 1.3,
    other_lang_penalty: float = 0.9,
    seed_with_rare_terms: bool = True,
    seed_terms: int = 2,
) -> List[Tuple[int, float]]:
    if not terms:
        return []

    N = dbmod.get_total_docs(conn)
    if N == 0:
        return []
    avgdl = dbmod.get_avg_doc_len(conn) or 1.0

    dfs = dbmod.get_dfs(conn, terms)
    # Drop terms with df==0 early
    terms = [t for t in terms if dfs.get(t, 0) > 0]
    if not terms:
        return []
    # Optionally drop overly common terms entirely (aggressive)
    threshold = None
    if max_df_ratio is not None and N > 0:
        threshold = max_df_ratio * N
        if threshold < 1:
            threshold = 1
    postings = dbmod.get_postings_for_terms(conn, terms)

    scores: Dict[int, float] = {}
    doc_lens_cache: Dict[int, int] = {}

    # sort terms by rarity so that we seed candidate set with rare terms
    ordered_terms = sorted(terms, key=lambda t: dfs.get(t, 0))
    seed_set: Set[str] = set()
    if seed_with_rare_terms and ordered_terms:
        for t in ordered_terms:
            if len(seed_set) >= max(1, int(seed_terms)):
                break
            seed_set.add(t)

    candidates: Set[int] = set()

    def _preload_lengths(ids: List[int]):
        # Use small LRU to reuse across queries
        missing = [i for i in ids if (i not in doc_lens_cache and i not in _DOC_LEN_CACHE)]
        if missing:
            fetched = dbmod.get_doc_lens(conn, missing)
            for k, v in fetched.items():
                _DOC_LEN_CACHE[k] = int(v)
                doc_lens_cache[k] = int(v)
                # LRU eviction
                _DOC_LEN_CACHE.move_to_end(k)
                if len(_DOC_LEN_CACHE) > _DOC_LEN_CACHE_MAX:
                    _DOC_LEN_CACHE.popitem(last=False)
        # Pull cached values
        for i in ids:
            if i in _DOC_LEN_CACHE and i not in doc_lens_cache:
                doc_lens_cache[i] = _DOC_LEN_CACHE[i]

    for t in ordered_terms:
        df = dfs.get(t, 0)
        if df == 0:
            continue
        idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
        plist = postings.get(t, [])
        if not plist:
            continue
        _preload_lengths([doc_id for doc_id, _ in plist])
        high_df = threshold is not None and df > threshold
        allow_expand = (t in seed_set) or (not high_df)
        for doc_id, tf in plist:
            if (doc_id not in candidates) and (not allow_expand):
                # do not expand candidate set via very frequent term
                continue
            # ensure candidate marked
            candidates.add(doc_id)
            dl = doc_lens_cache.get(doc_id)
            if dl is None:
                # last-resort fetch (rare)
                _, _, _, _, dl, _ = dbmod.get_doc(conn, doc_id)
                dl = int(dl)
                doc_lens_cache[doc_id] = dl
                _DOC_LEN_CACHE[doc_id] = dl
                _DOC_LEN_CACHE.move_to_end(doc_id)
                if len(_DOC_LEN_CACHE) > _DOC_LEN_CACHE_MAX:
                    _DOC_LEN_CACHE.popitem(last=False)
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
            score = idf * ((tf * (k1 + 1.0)) / (denom + 1e-9))
            scores[doc_id] = scores.get(doc_id, 0.0) + score

    if preferred_lang:
        pref = preferred_lang.strip().lower()
        langs = dbmod.get_doc_languages(conn, list(scores.keys()))
        filtered = {doc_id: score for doc_id, score in scores.items() if (langs.get(doc_id) or "").lower().startswith(pref)}
        if filtered:
            scores = filtered
        else:
            for doc_id, score in list(scores.items()):
                lang = (langs.get(doc_id) or "").lower()
                if lang.startswith(pref):
                    scores[doc_id] = score * lang_boost
                elif lang:
                    scores[doc_id] = score * other_lang_penalty

    # Deterministic tie-break by doc_id ascending for consistent ordering
    ranked = sorted(scores.items(), key=lambda x: (x[1], -x[0]), reverse=True)
    return ranked[:top_k]


def _domain_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _rerank_results(
    conn,
    candidates: List[Tuple[int, float]],
    terms: Sequence[str],
    *,
    top_k: int = 10,
    preferred_lang: str | None = None,
) -> List[Tuple[int, float]]:
    if not candidates:
        return []
    # Fetch docs
    doc_ids = [doc_id for doc_id, _ in candidates]
    meta = {doc_id: dbmod.get_doc(conn, doc_id) for doc_id in doc_ids}
    last = dbmod.get_docs_last_crawled(conn, doc_ids)

    # Weights (env-tunable)
    w_title = float(os.environ.get("YOX_TITLE_WEIGHT", "0.15"))
    w_prox = float(os.environ.get("YOX_PROX_WEIGHT", "0.10"))
    w_fresh = float(os.environ.get("YOX_FRESH_WEIGHT", "0.08"))
    fresh_days = float(os.environ.get("YOX_FRESH_DAYS", "14"))
    max_per_domain = int(os.environ.get("YOX_MAX_PER_DOMAIN", "2"))

    now = datetime.now(timezone.utc)
    used_terms = list(terms)

    # Quality filters (env-tunable)
    min_tokens = int(os.environ.get("YOX_MIN_DOC_TOKENS", "20") or 20)
    min_unique_ratio = float(os.environ.get("YOX_MIN_UNIQUE_RATIO", "0.2") or 0.2)
    strict_lang = (os.environ.get("YOX_STRICT_LANG", "0").strip() in ("1", "true", "True"))
    exclude_list = os.environ.get(
        "YOX_EXCLUDE_URL",
        "/privacy,/gizlilik,/kvkk,/terms,/cookies,/login,/register,/signup,/cart,/account,/checkout,/404,/error,/rss",
    )
    exclude_subs = [s.strip().lower() for s in exclude_list.split(",") if s.strip()]

    def bad_url(u: str) -> bool:
        ul = (u or "").lower()
        return any(x in ul for x in exclude_subs)

    def proximity_boost(text: str) -> float:
        if len(used_terms) < 2 or not text:
            return 0.0
        lc = text.casefold()
        positions = []
        for t in used_terms:
            p = lc.find(t)
            if p >= 0:
                positions.append(p)
        if len(positions) < 2:
            return 0.0
        positions.sort()
        # smaller gap -> larger boost, bounded
        min_gap = min(positions[i+1] - positions[i] for i in range(len(positions)-1))
        return 1.0 / (1.0 + (min_gap / 80.0))

    rescored: List[Tuple[int, float, str]] = []  # (doc_id, score, domain)
    seen_titles: Set[str] = set()
    for doc_id, base in candidates:
        _, url, title, content, _, doc_lang = meta.get(doc_id) or (None, "", "", "", 0, "")
        if bad_url(url):
            continue
        # Language gate (optional strict)
        if strict_lang and preferred_lang:
            if not (doc_lang or "").lower().startswith(preferred_lang.strip().lower()):
                continue
        # Basic content quality filter
        toks = tokenize(content or "", remove_stopwords=False)
        if len(toks) < min_tokens:
            continue
        uniq_ratio = (len(set(toks)) / float(len(toks))) if toks else 0.0
        if uniq_ratio < min_unique_ratio:
            continue
        # De-duplicate by normalized title
        norm_title = (title or "").strip().lower()
        if norm_title and norm_title in seen_titles:
            continue
        if norm_title:
            seen_titles.add(norm_title)
        dom = _domain_from_url(url)
        lc_title = (title or "").casefold()
        title_hits = sum(1 for t in used_terms if t and t in lc_title)
        title_boost = (title_hits / max(1, len(used_terms)))
        prox = proximity_boost(content or "")
        # Freshness
        fresh_bonus = 0.0
        iso = last.get(doc_id) or ""
        if iso:
            try:
                dt = datetime.strptime(iso.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S%z")
                age_days = (now - dt).total_seconds() / 86400.0
                if age_days <= fresh_days:
                    fresh_bonus = 1.0 - (age_days / max(1e-6, fresh_days))
            except Exception:
                pass
        score = base * (1.0 + w_title * title_boost + w_prox * prox + w_fresh * fresh_bonus)
        rescored.append((doc_id, score, dom))

    rescored.sort(key=lambda x: x[1], reverse=True)

    # Domain diversity: cap per domain in final top_k
    if max_per_domain > 0:
        taken: Dict[str, int] = {}
        final: List[Tuple[int, float]] = []
        for doc_id, sc, dom in rescored:
            cnt = taken.get(dom, 0)
            if cnt >= max_per_domain and dom:
                continue
            final.append((doc_id, sc))
            if dom:
                taken[dom] = cnt + 1
            if len(final) >= top_k:
                break
        return final
    return [(doc_id, sc) for doc_id, sc, _ in rescored][:top_k]


def _ai_expand_terms(query: str, *, lang: str = "tr", max_terms: int = 6) -> List[str]:
    """Use Mistral (if available) to expand query with semantically related terms.

    Returns a list of short keywords/phrases. If API/key not available,
    returns an empty list.
    """
    # Allow enable/disable via env (default on if API key exists)
    enable_env = os.environ.get("YOX_QUERY_EXPAND")
    if enable_env is not None and enable_env.strip() in ("0", "false", "False"):
        return []
    key = (query.strip().lower(), (lang or "").strip().lower())
    # Negative cache: if within TTL, skip calling remote APIs
    neg_until = _QE_NEG_CACHE.get(key)
    if neg_until and time.time() < neg_until:
        return []

    if key in _QE_CACHE:
        return _QE_CACHE[key]

    def _parse_terms(text: str) -> List[str]:
        parts = [p.strip() for p in (text or "").replace("\n", " ").split(",")]
        return [p for p in parts if p]

    # 1) Try Mistral first (if configured)
    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if api_key and requests is not None:
        base = os.environ.get("MISTRAL_API_BASE", "https://api.mistral.ai")
        model = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
        url = base.rstrip("/") + "/v1/chat/completions"
        prompt = (
            "Kullanıcının arama ifadesi: '" + query.strip() + "'.\n"
            "Bu ifadeye anlamca yakın 3-8 kısa anahtar kelime/ifade üret."
            " Yalnızca virgülle ayrılmış bir liste döndür. Alakasız, çok genel"
            " veya markasız tekil kelimeler üretme. Dil: Türkçe."
        )
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Kısa, virgülle ayrılmış anahtar kelimeler üret."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 200,
        }
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=12)
            if resp.status_code == 200:
                js = resp.json()
                choices = js.get("choices") or []
                if choices:
                    content = (choices[0].get("message", {}) or {}).get("content", "")
                    terms = _parse_terms(content)
                    if max_terms > 0:
                        terms = terms[:max_terms]
                    _QE_CACHE[key] = terms
                    return terms
            elif resp.status_code == 429:
                # Honor Retry-After (seconds) if present, else backoff 60s
                try:
                    ra = int(resp.headers.get("Retry-After", "0") or "0")
                except Exception:
                    ra = 0
                ttl = max(5, min(120, ra or 60))
                _QE_NEG_CACHE[key] = time.time() + ttl
            # For other non-200, fall through to Gemini
        except Exception:
            # Fall through to Gemini
            pass

    # 2) Fallback to Gemini if configured
    g_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if g_key and requests is not None:
        g_model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        g_url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent?key={g_key}"
        prompt = (
            "Sorgu: '" + query.strip() + "'.\n"
            "Bu sorguya anlamca yakın 3-8 kısa Türkçe anahtar kelime/ifade üret."
            " Yalnızca virgülle ayrılmış bir liste döndür."
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ]
        }
        try:
            resp = requests.post(g_url, json=payload, timeout=12)
            if resp.status_code == 200:
                js = resp.json()
                candidates = js.get("candidates") or []
                text = ""
                if candidates:
                    parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
                    if parts:
                        text = (parts[0] or {}).get("text", "")
                terms = _parse_terms(text)
                if max_terms > 0:
                    terms = terms[:max_terms]
                _QE_CACHE[key] = terms
                return terms
            elif resp.status_code == 429:
                try:
                    ra = int(resp.headers.get("Retry-After", "0") or "0")
                except Exception:
                    ra = 0
                ttl = max(5, min(120, ra or 45))
                _QE_NEG_CACHE[key] = time.time() + ttl
        except Exception:
            pass

    # Negative cache short TTL to avoid immediate retries
    _QE_NEG_CACHE[key] = time.time() + 30
    return []


def bm25_scores(
    conn,
    query: str,
    *,
    k1: float = 1.5,
    b: float = 0.75,
    top_k: int = 10,
    preferred_lang: str | None = None,
) -> List[Tuple[int, float]]:
    terms = tokenize(query)
    return bm25_scores_terms(conn, terms, k1=k1, b=b, top_k=top_k, preferred_lang=preferred_lang)


def snippets(content: str, terms: Sequence[str], radius: int = 120) -> str:
    if not content:
        return ""
    lc = content.casefold()
    pos = -1
    for t in terms:
        p = lc.find(t)
        if p != -1:
            pos = p
            break
    if pos == -1:
        return (content[:radius * 2] + "...") if len(content) > radius * 2 else content
    start = max(0, pos - radius)
    end = min(len(content), pos + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""
    return prefix + content[start:end] + suffix


def _vocab(conn) -> List[str]:
    try:
        cnt = dbmod.term_count(conn)
    except Exception:
        cnt = -1
    if _VOCAB_CACHE["count"] == cnt and _VOCAB_CACHE["vocab"]:
        return _VOCAB_CACHE["vocab"]
    cur = conn.execute("SELECT term FROM terms")
    vocab = [row[0] for row in cur.fetchall()]
    _VOCAB_CACHE["count"] = cnt
    _VOCAB_CACHE["vocab"] = vocab
    return vocab


def _kgrams(term: str, k: int = _DEFAULT_K) -> Set[str]:
    if k <= 0:
        k = 3
    # Use boundary markers to improve matching for prefixes/suffixes
    padded = f"^{term}$"
    return {padded[i : i + k] for i in range(max(0, len(padded) - k + 1))}


def _ensure_kgram_data(conn, k: int = _DEFAULT_K):
    try:
        cnt = dbmod.term_count(conn)
    except Exception:
        cnt = -1
    # Reuse if up-to-date and present
    if (
        _KGRAM_DATA_CACHE.get("count") == cnt
        and _KGRAM_DATA_CACHE.get("k") == k
        and _KGRAM_DATA_CACHE.get("index")
        and _KGRAM_DATA_CACHE.get("termgrams")
    ):
        return _KGRAM_DATA_CACHE

    vocab = _vocab(conn)
    index: TDict[str, Set[str]] = {}
    termgrams: TDict[str, Set[str]] = {}
    for t in vocab:
        gs = _kgrams(t, k)
        termgrams[t] = gs
        for g in gs:
            s = index.get(g)
            if s is None:
                s = set()
                index[g] = s
            s.add(t)

    _KGRAM_DATA_CACHE["count"] = cnt
    _KGRAM_DATA_CACHE["k"] = k
    _KGRAM_DATA_CACHE["index"] = index
    _KGRAM_DATA_CACHE["termgrams"] = termgrams
    return _KGRAM_DATA_CACHE


def _kgram_candidates(
    query: str,
    *,
    conn,
    k: int = _DEFAULT_K,
    jaccard_min: float = 0.2,
    max_candidates: int = 200,
) -> List[Tuple[str, float]]:
    data = _ensure_kgram_data(conn, k)
    index: Dict[str, Set[str]] = data["index"]  # type: ignore[assignment]
    termgrams: Dict[str, Set[str]] = data["termgrams"]  # type: ignore[assignment]
    qg = _kgrams(query, k)
    if not qg:
        return []
    # Gather union of candidates that share any gram
    cands: Set[str] = set()
    for g in qg:
        cands.update(index.get(g, ()))
        if len(cands) >= max_candidates * 4:
            break
    if not cands:
        return []
    scored: List[Tuple[str, float]] = []
    for t in cands:
        tg = termgrams.get(t)
        if not tg:
            continue
        inter = len(qg & tg)
        if inter == 0:
            continue
        union = len(qg | tg) or 1
        j = inter / union
        if j >= jaccard_min:
            scored.append((t, j))
        if len(scored) > max_candidates * 2:
            break
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_candidates]


def fuzzy_correct_terms(
    conn,
    terms: Sequence[str],
    cutoff: float = 0.72,
    *,
    k: int = _DEFAULT_K,
    jaccard_min: float = 0.2,
    max_candidates: int = 200,
) -> Tuple[List[str], Dict[str, str]]:
    """Return corrected terms and a mapping of original->corrected.

    Uses a k-gram index to shortlist candidates via Jaccard similarity,
    then refines with difflib ratio. This avoids scanning the entire
    vocabulary for each query term and scales better as vocab grows.
    """
    if not terms:
        return [], {}
    dfs = dbmod.get_dfs(conn, terms)
    corrected: List[str] = []
    mapping: Dict[str, str] = {}
    for t in terms:
        if dfs.get(t, 0) > 0:
            corrected.append(t)
            continue

        # k-gram shortlist
        cands = _kgram_candidates(
            t, conn=conn, k=k, jaccard_min=jaccard_min, max_candidates=max_candidates
        )
        best = None
        best_score = 0.0
        for cand, j in cands:
            # Use difflib ratio as final metric; jaccard is coarse filter
            r = difflib.SequenceMatcher(a=t, b=cand).ratio()
            # Combine scores mildly to favor high jaccard too
            score = 0.7 * r + 0.3 * j
            if score > best_score:
                best_score = score
                best = cand
        if best is None:
            # Fallback to difflib across full vocab in rare cases
            vocab = _vocab(conn)
            fallback = difflib.get_close_matches(t, vocab, n=1, cutoff=cutoff)
            if fallback:
                best = fallback[0]
                best_score = difflib.SequenceMatcher(a=t, b=best).ratio()

        if best is not None and best_score >= cutoff:
            mapping[t] = best
            corrected.append(best)
        else:
            corrected.append(t)
    return corrected, mapping


def search_with_fuzzy(
    conn,
    query: str,
    *,
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
    fuzzy: bool = True,
    preferred_lang: str | None = None,
    expand: bool | None = None,
) -> Tuple[List[Tuple[int, float]], Dict[str, str], List[str]]:
    # Optional AI-based query expansion (Mistral API)
    expand_terms: List[str] = []
    try:
        do_expand = expand
        if do_expand is None:
            # env-driven default: expand if API key present and not disabled
            do_expand = bool(os.environ.get("MISTRAL_API_KEY")) and (
                os.environ.get("YOX_QUERY_EXPAND", "1").strip() not in ("0", "false", "False")
            )
        if do_expand:
            expand_terms = _ai_expand_terms(query, lang=(preferred_lang or "tr"))
    except Exception:
        expand_terms = []
    extra_text = ", ".join(expand_terms) if expand_terms else ""
    if extra_text:
        combined = f"{query} {extra_text}".strip()
    else:
        combined = query
    terms = tokenize(combined)
    corrections: Dict[str, str] = {}
    used_terms = terms
    if fuzzy:
        used_terms, corrections = fuzzy_correct_terms(conn, terms)
    # Get a wider initial set then rerank for quality and diversity
    initial_k = max(50, top_k * 4)
    results = bm25_scores_terms(
        conn,
        used_terms,
        k1=k1,
        b=b,
        top_k=initial_k,
        preferred_lang=preferred_lang,
    )
    reranked = _rerank_results(
        conn,
        results,
        used_terms,
        top_k=top_k,
        preferred_lang=(preferred_lang or None),
    )
    return reranked, corrections, used_terms
