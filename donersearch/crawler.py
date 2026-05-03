from __future__ import annotations

import asyncio
import os
import gzip
import hashlib
import time
import urllib.request
import random
import threading
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse, urldefrag
from xml.etree import ElementTree
from datetime import datetime, timedelta


try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image
except ImportError:
    Image = None

from .htmlparse import extract_text_and_links
from .indexer import index_document
from . import db as dbmod


# VarsayÄ±lan bot kimliÄŸi (robots.txt ve HTTP User-Agent iÃ§in)
USER_AGENT = "yoxbot/1.0 (+https://local)"
MAX_SITEMAP_URLS = 10000
MAX_AUTO_DISCOVER_PER_HOST = 10

IMAGES_DIR = Path("data/images")
THUMBNAILS_DIR = IMAGES_DIR / "thumbnails"
HTTP_WARNING_ISSUED = False
PIL_WARNING_ISSUED = False
PLAYWRIGHT_WARNING_SHOWN = False

# Resource limits for stability (suitable for VDS/server)
MAX_HTML_BYTES = 2 * 1024 * 1024  # 2 MB
MAX_XML_BYTES = 2 * 1024 * 1024   # 2 MB
MAX_TXT_BYTES = 512 * 1024        # 512 KB (robots.txt)
MAX_IMAGE_BYTES = 8 * 1024 * 1024 # 8 MB per image


def _build_opener(user_agent: str, accept: str = "text/html,application/xhtml+xml") -> urllib.request.OpenerDirector:
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", user_agent), ("Accept", accept)]
    return opener


class RenderPool:
    def __init__(self, size: int, user_agent: str, headless: bool = True):
        from playwright.sync_api import sync_playwright

        self.size = max(1, size)
        self.user_agent = user_agent
        self._lock = threading.Lock()
        self._next = 0
        self._contexts: List[Optional[object]] = []
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=headless)
        self._context_args = {}
        if user_agent:
            self._context_args["user_agent"] = user_agent
        for _ in range(self.size):
            self._contexts.append(self._browser.new_context(**self._context_args))

    def acquire(self) -> Tuple[int, object]:
        with self._lock:
            idx = self._next
            self._next = (self._next + 1) % len(self._contexts)
            context = self._contexts[idx]
            if context is None:
                context = self._browser.new_context(**self._context_args)
                self._contexts[idx] = context
        return idx, context

    def mark_unhealthy(self, idx: int) -> None:
        with self._lock:
            context = self._contexts[idx]
            if context is not None:
                try:
                    context.close()
                except Exception:
                    pass
            try:
                self._contexts[idx] = self._browser.new_context(**self._context_args)
            except Exception:
                self._contexts[idx] = None

    def close(self) -> None:
        with self._lock:
            contexts = list(self._contexts)
            self._contexts = []
        for ctx in contexts:
            if ctx is None:
                continue
            try:
                ctx.close()
            except Exception:
                pass
        if hasattr(self, "_browser") and self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
        if hasattr(self, "_playwright") and self._playwright:
            try:
                self._playwright.stop()
            except Exception:
                pass


@dataclass
class RobotRules:
    rules: List[Tuple[str, bool]]
    crawl_delay: Optional[float]
    sitemaps: List[str]

    def allows(self, path: str) -> bool:
        if not self.rules:
            return True
        best_len = -1
        allowed = True
        for pattern, is_allow in self.rules:
            if pattern == "":
                best_len = max(best_len, 0)
                allowed = is_allow
            elif path.startswith(pattern) and len(pattern) > best_len:
                best_len = len(pattern)
                allowed = is_allow
        return allowed


def _fetch_bytes(
    url: str,
    *,
    timeout: float = 10.0,
    accept: str = "*/*",
    user_agent: str = USER_AGENT,
    max_bytes: Optional[int] = None,
) -> Tuple[Optional[bytes], str]:
    opener = _build_opener(user_agent, accept)
    try:
        with opener.open(url, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "").lower()
            try:
                cl = int(resp.headers.get("Content-Length") or "0")
            except Exception:
                cl = 0
            if max_bytes and cl and cl > max_bytes:
                return None, ctype
            if not max_bytes:
                return resp.read(), ctype
            # Read with cap
            buf = BytesIO()
            chunk = 64 * 1024
            remaining = max_bytes
            while remaining > 0:
                data = resp.read(min(chunk, remaining))
                if not data:
                    break
                buf.write(data)
                remaining -= len(data)
            return buf.getvalue(), ctype
    except Exception:
        return None, ""


def _fetch_html(
    url: str,
    timeout: float = 10.0,
    user_agent: str = USER_AGENT,
    render_pool: Optional[RenderPool] = None,
) -> Optional[bytes]:
    if render_pool is not None:
        page = None
        idx = -1
        try:
            idx, context = render_pool.acquire()
            page = context.new_page()
            goto_timeout = 20000
            page.set_default_timeout(goto_timeout)
            page.goto(url, timeout=goto_timeout, wait_until="networkidle")
            html = page.content()
            return html.encode("utf-8")
        except Exception:
            if idx >= 0:
                try:
                    render_pool.mark_unhealthy(idx)
                except Exception:
                    pass
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass
    data, ctype = _fetch_bytes(
        url,
        timeout=timeout,
        accept="text/html,application/xhtml+xml",
        user_agent=user_agent,
        max_bytes=MAX_HTML_BYTES,
    )
    if data is None:
        return None
    if "text/html" in ctype or "application/xhtml" in ctype or ctype == "":
        return data
    return None


def _read_text(bs: Optional[bytes]) -> Optional[str]:
    if bs is None:
        return None
    try:
        if bs.startswith(b"\x1f\x8b"):
            try:
                bs = gzip.decompress(bs)
            except Exception:
                pass
        return bs.decode("utf-8", errors="ignore")
    except Exception:
        try:
            return bs.decode("latin-1", errors="ignore")
        except Exception:
            return None


class RobotsCache:
    def __init__(self, user_agent: str = USER_AGENT):
        self.cache: Dict[str, RobotRules] = {}
        self.processed_sitemaps: Set[str] = set()
        self.user_agent = user_agent.strip() or USER_AGENT

    def get_rules(self, parsed) -> RobotRules:
        base = f"{parsed.scheme}://{parsed.netloc}"
        rules = self.cache.get(base)
        if rules:
            return rules

        robots_url = base + "/robots.txt"
        data, _ = _fetch_bytes(
            robots_url,
            timeout=5.0,
            accept="text/plain,text/*;q=0.9,*/*;q=0.1",
            max_bytes=MAX_TXT_BYTES,
        )
        text = _read_text(data) or ""

        allow_rules: List[Tuple[str, bool]] = []
        crawl_delay: Optional[float] = None
        sitemaps: List[str] = []

        current_relevant = False
        last_key = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            key, value = parts[0].strip().lower(), parts[1].strip()
            if key == "user-agent":
                if last_key != "user-agent":
                    current_relevant = False
                agent = value.lower()
                if agent == self.user_agent.lower() or agent == "*":
                    current_relevant = True
                last_key = "user-agent"
            elif key == "allow":
                if current_relevant:
                    allow_rules.append((value or "", True))
                last_key = key
            elif key == "disallow":
                if current_relevant:
                    allow_rules.append((value or "", False))
                last_key = key
            elif key == "crawl-delay":
                if current_relevant:
                    try:
                        crawl_delay = float(value)
                    except ValueError:
                        pass
                last_key = key
            elif key == "sitemap":
                if value:
                    sitemaps.append(value)
                last_key = key
            else:
                last_key = key

        rules = RobotRules(allow_rules, crawl_delay, sitemaps)
        self.cache[base] = rules
        return rules

    def is_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        rules = self.get_rules(parsed)
        path = parsed.path or "/"
        return rules.allows(path)

    def delay_for_host(self, parsed) -> Optional[float]:
        return self.get_rules(parsed).crawl_delay

    def sitemap_urls(self, parsed) -> List[str]:
        base = f"{parsed.scheme}://{parsed.netloc}"
        rules = self.get_rules(parsed)
        urls: List[str] = []
        queue = [s for s in rules.sitemaps if s not in self.processed_sitemaps]
        while queue and len(urls) < MAX_SITEMAP_URLS:
            sitemap_url = queue.pop(0)
            self.processed_sitemaps.add(sitemap_url)
            data, _ = _fetch_bytes(
                sitemap_url,
                timeout=10.0,
                accept="application/xml,text/xml,text/plain",
                user_agent=self.user_agent,
                max_bytes=MAX_XML_BYTES,
            )
            if not data:
                continue
            xml_text = _read_text(data)
            if not xml_text:
                continue
            try:
                root = ElementTree.fromstring(xml_text)
            except Exception:
                continue
            tag = root.tag.lower()
            if tag.endswith("sitemapindex"):
                for loc in root.findall(".//{*}loc"):
                    loc_text = (loc.text or "").strip()
                    if loc_text and loc_text not in self.processed_sitemaps:
                        queue.append(loc_text)
            elif tag.endswith("urlset"):
                for loc in root.findall(".//{*}loc"):
                    loc_text = (loc.text or "").strip()
                    if loc_text:
                        urls.append(loc_text)
                        if len(urls) >= MAX_SITEMAP_URLS:
                            break
        return urls[:MAX_SITEMAP_URLS]


def _ensure_image_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)


def _choose_extension(format_name: str, url: str) -> str:
    ext = ""
    if format_name and format_name.lower() != "unknown":
        ext = format_name.lower()
    if not ext:
        path = urlparse(url).path
        if "." in path:
            ext = path.rsplit(".", 1)[-1].split("?", 1)[0].lower()
    if not ext:
        ext = "bin"
    return ext


def _build_image_record(url: str, alt: str, data: Optional[bytes]) -> Dict[str, object]:
    size_bytes = len(data) if data else 0
    hash_value = hashlib.sha256(data).hexdigest() if data else ""
    width = height = 0
    fmt = "unknown"
    aspect = 0.0
    if data and Image:
        try:
            with Image.open(BytesIO(data)) as img:
                img.load()
                width, height = img.size
                fmt = (img.format or "unknown").lower()
                if height:
                    aspect = round(width / height, 4)
        except Exception:
            pass
    return {
        "url": url,
        "alt": alt or "",
        "width": width,
        "height": height,
        "format": fmt,
        "size_bytes": size_bytes,
        "aspect_ratio": aspect,
        "hash": hash_value,
        "file_path": "",
        "thumbnail_path": "",
        "data": data,
    }


async def _download_images_async(entries: List[Tuple[str, str]], user_agent: str) -> List[Tuple[str, str, Optional[bytes]]]:
    results: List[Tuple[str, str, Optional[bytes]]] = []
    if not aiohttp:
        return results
    timeout = aiohttp.ClientTimeout(total=20)

    async def fetch(session: aiohttp.ClientSession, url: str, alt: str, sem: asyncio.Semaphore):
        async with sem:
            try:
                async with session.get(url, timeout=15) as resp:
                    if resp.status != 200:
                        return url, alt, None
                    try:
                        cl = int(resp.headers.get("Content-Length") or "0")
                    except Exception:
                        cl = 0
                    if cl and cl > MAX_IMAGE_BYTES:
                        return url, alt, None
                    # Read with cap
                    buf = bytearray()
                    async for chunk in resp.content.iter_chunked(64 * 1024):
                        buf.extend(chunk)
                        if len(buf) > MAX_IMAGE_BYTES:
                            return url, alt, None
                    data = bytes(buf)
                    return url, alt, data
            except Exception:
                return url, alt, None

    connector = aiohttp.TCPConnector(limit=10)
    sem = asyncio.Semaphore(5)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, headers={"User-Agent": user_agent}
    ) as session:
        tasks = [fetch(session, url, alt, sem) for url, alt in entries]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
    for item in gathered:
        if isinstance(item, tuple):
            results.append(item)
    return results


def _download_images_sync(entries: List[Tuple[str, str]], user_agent: str) -> List[Tuple[str, str, Optional[bytes]]]:
    if not requests:
        return [(url, alt, None) for url, alt in entries]
    results: List[Tuple[str, str, Optional[bytes]]] = []
    session = requests.Session()
    for url, alt in entries:
        try:
            resp = session.get(url, headers={"User-Agent": user_agent}, timeout=15, stream=True)
            if resp.status_code != 200:
                results.append((url, alt, None))
                continue
            try:
                cl = int(resp.headers.get("Content-Length") or "0")
            except Exception:
                cl = 0
            if cl and cl > MAX_IMAGE_BYTES:
                results.append((url, alt, None))
                continue
            # Stream with cap
            buf = BytesIO()
            remaining = MAX_IMAGE_BYTES
            for chunk in resp.iter_content(64 * 1024):
                if not chunk:
                    break
                if remaining <= 0:
                    buf = None
                    break
                take = chunk if len(chunk) <= remaining else chunk[:remaining]
                buf.write(take)
                remaining -= len(take)
                if remaining <= 0:
                    buf = None
                    break
            if buf is None:
                results.append((url, alt, None))
            else:
                results.append((url, alt, buf.getvalue()))
        except Exception:
            results.append((url, alt, None))
    session.close()
    return results


def _save_image_assets(record: Dict[str, object]) -> None:
    data = record.pop("data", None)
    if not data:
        record["file_path"] = record.get("file_path") or ""
        record["thumbnail_path"] = record.get("thumbnail_path") or ""
        return
    hash_value = record.get("hash") or hashlib.sha256(data).hexdigest()
    record["hash"] = hash_value
    _ensure_image_dirs()
    ext = _choose_extension(record.get("format", ""), record.get("url", ""))
    file_path = IMAGES_DIR / f"{hash_value}.{ext}"
    if not file_path.exists():
        file_path.write_bytes(data)
    record["file_path"] = str(file_path)

    if Image and record.get("width") and record.get("height"):
        thumb_path = THUMBNAILS_DIR / f"{hash_value}.jpg"
        if not thumb_path.exists():
            try:
                with Image.open(BytesIO(data)) as img:
                    img.load()
                    thumb = img.copy()
                    thumb.thumbnail((200, 200))
                    if thumb.mode not in ("RGB", "L"):
                        thumb = thumb.convert("RGB")
                    thumb.save(thumb_path, format="JPEG", quality=85)
            except Exception:
                thumb_path = None
        record["thumbnail_path"] = str(thumb_path) if thumb_path else ""
    else:
        record["thumbnail_path"] = record.get("thumbnail_path") or ""


def _process_image_entries(entries: List[Tuple[str, str]], user_agent: str) -> List[Dict[str, object]]:
    if not entries:
        return []
    filtered: List[Tuple[str, str]] = []
    for url, alt in entries:
        if not url or url.lower().startswith("data:"):
            continue
        filtered.append((url, alt))
    if not filtered:
        return []

    global HTTP_WARNING_ISSUED, PIL_WARNING_ISSUED

    if not (aiohttp or requests):
        if not HTTP_WARNING_ISSUED:
            print("[images] HTTP client libraries unavailable; skipping image downloads")
            HTTP_WARNING_ISSUED = True
        return [_build_image_record(url, alt, None) for url, alt in filtered]

    results: List[Tuple[str, str, Optional[bytes]]] = []
    if aiohttp:
        try:
            results = asyncio.run(_download_images_async(filtered, user_agent))
        except Exception:
            results = []
    if not results:
        results = _download_images_sync(filtered, user_agent)

    if not results:
        return [_build_image_record(url, alt, None) for url, alt in filtered]

    if Image is None and not PIL_WARNING_ISSUED:
        print("[images] Pillow not available; image metadata will be limited")
        PIL_WARNING_ISSUED = True

    return [_build_image_record(url, alt, data) for url, alt, data in results]

def _same_domain(url: str, domains: Set[str]) -> bool:
    if not domains:
        return True
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in domains)


def _canonicalize(url: str) -> str:
    url, _ = urldefrag(url)
    p = urlparse(url)
    scheme = p.scheme.lower()
    netloc = p.netloc.lower()
    # Remove default ports
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]
    path = p.path or "/"
    return urlunparse((scheme, netloc, path, p.params, p.query, ""))


_TURKISH_CHARS = {chr(cp) for cp in (0x00E7, 0x011F, 0x0131, 0x00F6, 0x015F, 0x00FC, 0x00C7, 0x011E, 0x0130, 0x00D6, 0x015E, 0x00DC)}


def _normalize_lang(lang: str | None) -> str:
    if not lang:
        return ""
    lang = lang.strip().lower()
    for sep in ("-", "_", ","):
        if sep in lang:
            lang = lang.split(sep, 1)[0]
    return lang[:5]


def _is_turkish_host(host: str, path: str = "", query: str = "") -> bool:
    host = (host or "").lower()
    if host.endswith(".tr") or host.endswith(".com.tr") or host.endswith(".net.tr"):
        return True
    parts = host.split(".")
    if parts and parts[0] not in ("www", "m") and len(parts[0]) == 2 and parts[0].isalpha():
        if parts[0] == "tr":
            return True
    path = (path or "").lower()
    if "/tr/" in path or path.startswith("/tr"):
        return True
    query = (query or "").lower()
    if "lang=tr" in query:
        return True
    return False


def _guess_language(url: str, lang_hint: str, text: str) -> str:
    lang_hint = _normalize_lang(lang_hint)
    if lang_hint:
        return lang_hint
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if _is_turkish_host(host, path, parsed.query):
        return "tr"
    query = parsed.query.lower()
    if "lang=" in query:
        for item in query.split('&'):
            if item.startswith('lang='):
                value = item.split('=', 1)[1] if '=' in item else ''
                value = value.split(';', 1)[0]
                if value:
                    return _normalize_lang(value)
    sample = text[:400]
    if sample:
        count = sum(1 for ch in sample if ch in _TURKISH_CHARS)
        if count >= 3:
            return "tr"
    return ""



def crawl(
    db_path: str,
    seed_urls: List[str],
    *,
    allowed_domains: Optional[List[str]] = None,
    max_pages: int = 100,
    max_depth: int = 2,
    delay_seconds: float = 0.5,
    auto_discover: bool = False,
    max_auto_discover_per_host: int = MAX_AUTO_DISCOVER_PER_HOST,
    manual_sitemaps: Optional[List[str]] = None,
    user_agent: str = USER_AGENT,
    render: bool = False,
    revisit_interval_hours: float = 24.0,
    render_pool_size: int = 5,
    render_headless: bool = False,
    failed_host_retry_hours: float = 12.0,
    max_pages_per_host: Optional[int] = None,
    workers: int = 1,
    crawl_run_id: Optional[int] = None,
) -> int:
    global PLAYWRIGHT_WARNING_SHOWN
    if not seed_urls and not auto_discover:
        return 0

    conn = dbmod.open_db(db_path)
    dbmod.ensure_schema(conn)

    robots = RobotsCache(user_agent=user_agent)
    seen: Set[str] = set()
    domains = {d.lower() for d in (allowed_domains or [])}
    q: Deque[Tuple[str, int]] = deque()
    queued: Set[str] = set()
    known_hosts: Dict[str, str] = {}
    auto_counts: Dict[str, int] = defaultdict(int)

    def push_url(canon_url: str, depth: int, *, front: bool = False) -> None:
        if canon_url in queued:
            return
        if front:
            q.appendleft((canon_url, depth))
        else:
            q.append((canon_url, depth))
        queued.add(canon_url)

    def add_seed(url: str) -> None:
        canon = _canonicalize(url)
        push_url(canon, 0)
        host = urlparse(canon).netloc.lower()
        if host:
            domains.add(host)

    for seed in seed_urls:
        add_seed(seed)

    # Helper to parse a sitemap URL into page URLs (manual sitemaps)
    def _parse_sitemap_urls(sm_url: str) -> List[str]:
        data, _ = _fetch_bytes(
            sm_url,
            timeout=10.0,
            accept="application/xml,text/xml,text/plain",
            user_agent=user_agent,
        )
        xml_text = _read_text(data)
        if not xml_text:
            return []
        try:
            root = ElementTree.fromstring(xml_text)
        except Exception:
            # Fallback: treat as newline-separated plain text list of URLs
            lines = [ln.strip() for ln in xml_text.splitlines()]
            return [ln for ln in lines if ln and "://" in ln]

        urls: List[str] = []
        tag = root.tag.lower()
        if tag.endswith("sitemapindex"):
            for loc in root.findall(".//{*}loc"):
                loc_text = (loc.text or "").strip()
                if loc_text:
                    urls.extend(_parse_sitemap_urls(loc_text))
        elif tag.endswith("urlset"):
            for loc in root.findall(".//{*}loc"):
                loc_text = (loc.text or "").strip()
                if loc_text:
                    urls.append(loc_text)
        return urls

    if manual_sitemaps:
        for sm in manual_sitemaps:
            if not sm:
                continue
            canon_sm = _canonicalize(sm)
            sm_host = urlparse(canon_sm).netloc.lower()
            if sm_host:
                domains.add(sm_host)
            try:
                print(f"[sitemap] fetching manual sitemap: {canon_sm}", flush=True)
                for u in _parse_sitemap_urls(canon_sm):
                    cu = _canonicalize(u)
                    push_url(cu, 0)
            except Exception:
                # Sessizce geÃ§; manuel sitemap baÅŸarÄ±sÄ±z ise normal crawling devam etsin
                pass

    if auto_discover:
        known_hosts = dbmod.get_discovered_hosts(conn)
        for seed in seed_urls:
            host = urlparse(seed).netloc.lower()
            if host:
                mark = known_hosts.get(host)
                if not mark:
                    dbmod.record_discovered_host(conn, host, seed, status="seed")
                    dbmod.update_discovered_host_status(conn, host, "seed")
                    known_hosts[host] = "seed"
        conn.commit()

    def _retry_timestamp(base_hours: float) -> Optional[str]:
        if base_hours <= 0:
            return None
        jitter = random.uniform(0.1, 0.35) * max(base_hours, 0.1)
        delta_hours = max(0.1, base_hours + jitter)
        ts = datetime.utcnow() + timedelta(hours=delta_hours)
        return ts.isoformat(timespec="seconds") + "Z"

    def mark_host(
        host: str,
        status: str,
        source_url: Optional[str],
        *,
        retry_after: Optional[str] = None,
        set_retry: bool = False,
        failure: bool = False,
    ) -> None:
        if not auto_discover:
            return
        dbmod.record_discovered_host(conn, host, source_url, status=status)
        kwargs = {"failure_increment": failure}
        if set_retry:
            kwargs["retry_after"] = retry_after
        dbmod.update_discovered_host_status(conn, host, status, **kwargs)
        known_hosts[host] = status

    def enqueue_discovered_url(
        canon_url: str,
        depth_value: int,
        origin_host: str,
        source_url: str,
        *,
        prefer_front: bool = False,
    ) -> None:
        nonlocal auto_counts
        link_parsed = urlparse(canon_url)
        if link_parsed.scheme not in ("http", "https"):
            return
        link_host = link_parsed.netloc.lower()
        if not link_host:
            return

        if link_host in domains or link_host == origin_host:
            push_url(canon_url, depth_value, front=prefer_front)
            return

        if not auto_discover:
            return

        auto_counts.setdefault(origin_host, 0)
        if max_auto_discover_per_host >= 0 and auto_counts[origin_host] >= max_auto_discover_per_host:
            return

        status = known_hosts.get(link_host)
        if status in {"queued", "seed", "crawled"}:
            return
        if status in {"failed", "blocked"}:
            return

        try:
            new_rules = robots.get_rules(link_parsed)
        except Exception:
            new_rules = None

        if not new_rules:
            retry_ts = _retry_timestamp(failed_host_retry_hours)
            mark_host(link_host, "failed", source_url, retry_after=retry_ts, set_retry=True, failure=True)
            conn.commit()
            print(f"[auto-discover] host={link_host} status=failed source={source_url}")
            return

        if not new_rules.allows(link_parsed.path or "/"):
            retry_ts = _retry_timestamp(failed_host_retry_hours * 2 if failed_host_retry_hours > 0 else failed_host_retry_hours)
            mark_host(link_host, "blocked", source_url, retry_after=retry_ts, set_retry=True, failure=True)
            conn.commit()
            print(f"[auto-discover] host={link_host} status=blocked source={source_url}")
            return

        mark_host(link_host, "queued", source_url, retry_after=None, set_retry=True)
        domains.add(link_host)
        auto_counts[origin_host] += 1
        prefer = prefer_front or _is_turkish_host(link_host, link_parsed.path, link_parsed.query)
        push_url(canon_url, 0, front=prefer)
        conn.commit()
        print(f"[auto-discover] host={link_host} status=queued source={source_url}")

    now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    due_docs = dbmod.get_due_documents(conn, now_iso, limit=max_pages * 2)
    for _, due_url in due_docs:
        canon_due = _canonicalize(due_url)
        host = urlparse(canon_due).netloc.lower()
        if host:
            domains.add(host)
        push_url(canon_due, 0)

    if auto_discover:
        retry_hosts = dbmod.get_hosts_for_retry(conn, now_iso, limit=max_pages * 2)
        for host in retry_hosts:
            if domains and not any(host == d or host.endswith("." + d) for d in domains):
                continue
            seed_url = f"https://{host}"
            domains.add(host)
            push_url(_canonicalize(seed_url), 0)
            mark_host(host, "queued", seed_url, retry_after=None, set_retry=True)
        if retry_hosts:
            conn.commit()

    render_pool: Optional[RenderPool] = None
    if render:
        try:
            render_pool = RenderPool(render_pool_size, user_agent, headless=render_headless)
        except ImportError:
            if not PLAYWRIGHT_WARNING_SHOWN:
                print("[render] Playwright not available; skipping JS rendering")
                PLAYWRIGHT_WARNING_SHOWN = True
            render_pool = None
        except Exception as exc:
            print(f"[render] Failed to initialize Playwright: {exc}")
            render_pool = None
            if not PLAYWRIGHT_WARNING_SHOWN:
                PLAYWRIGHT_WARNING_SHOWN = True

    pages_indexed = 0
    sitemap_processed: Set[str] = set()
    last_fetch_time: Dict[str, float] = {}
    host_processed: Dict[str, int] = defaultdict(int)

    try:
        # If multiple workers are requested, use a parallel scheduling path
        if workers and int(workers) > 1:
            return _crawl_parallel(
                db_path=db_path,
                seed_state=(q, queued, seen, domains),
                robots=robots,
                allowed_domains=domains,
                max_pages=max_pages,
                max_depth=max_depth,
                delay_seconds=delay_seconds,
                auto_discover=auto_discover,
                max_auto_discover_per_host=max_auto_discover_per_host,
                manual_sitemaps=manual_sitemaps,
                user_agent=user_agent,
                revisit_interval_hours=revisit_interval_hours,
                render=render,
                render_pool_size=render_pool_size,
                render_headless=render_headless,
                failed_host_retry_hours=failed_host_retry_hours,
                max_pages_per_host=max_pages_per_host,
                sitemap_processed=sitemap_processed,
                workers=workers,
                crawl_run_id=crawl_run_id,
            )

        while q and pages_indexed < max_pages:
            url, depth = q.popleft()
            queued.discard(url)
            canon_url = _canonicalize(url)
            if canon_url in seen:
                continue
            seen.add(canon_url)

            if not _same_domain(canon_url, domains):
                continue

            parsed = urlparse(canon_url)
            origin_host = parsed.netloc.lower()
            auto_counts.setdefault(origin_host, 0)

            # Host baÅŸÄ±na sayfa limiti
            if max_pages_per_host is not None and host_processed[origin_host] >= int(max_pages_per_host):
                continue

            rules = robots.get_rules(parsed)
            if not rules.allows(parsed.path or "/"):
                if auto_discover:
                    retry_ts = _retry_timestamp(failed_host_retry_hours * 2 if failed_host_retry_hours > 0 else failed_host_retry_hours)
                    mark_host(origin_host, "blocked", canon_url, retry_after=retry_ts, set_retry=True, failure=True)
                    conn.commit()
                print(f"[robots] blocked: {canon_url}", flush=True)
                continue

            host_base = f"{parsed.scheme}://{parsed.netloc}"
            if host_base not in sitemap_processed:
                for sitemap_url in robots.sitemap_urls(parsed):
                    sm = _canonicalize(sitemap_url)
                    sm_host = urlparse(sm).netloc.lower()
                    if sm_host == origin_host:
                        push_url(sm, 0)
                    else:
                        enqueue_discovered_url(sm, 0, origin_host, canon_url)
                sitemap_processed.add(host_base)

            host_delay = rules.crawl_delay if rules.crawl_delay is not None else 0.0
            wait_target = max(delay_seconds, host_delay)
            now = time.time()
            last = last_fetch_time.get(origin_host, 0.0)
            wait = wait_target - (now - last)
            if wait > 0:
                time.sleep(wait)

            # Bu host iÃ§in sayaÃ§ artÄ±r (denenen/iÅŸlenen sayfa)
            host_processed[origin_host] += 1
            print(f"[fetch] GET {canon_url} depth={depth}", flush=True)
            raw = _fetch_html(
                canon_url,
                timeout=10,
                user_agent=user_agent,
                render_pool=render_pool,
            )
            last_fetch_time[origin_host] = time.time()
            if not raw:
                if auto_discover:
                    retry_ts = _retry_timestamp(failed_host_retry_hours)
                    mark_host(origin_host, "failed", canon_url, retry_after=retry_ts, set_retry=True, failure=True)
                    conn.commit()
                print(f"[fetch] FAILED {canon_url}", flush=True)
                continue

            html = _read_text(raw)
            if not html:
                if auto_discover:
                    retry_ts = _retry_timestamp(failed_host_retry_hours)
                    mark_host(origin_host, "failed", canon_url, retry_after=retry_ts, set_retry=True, failure=True)
                    conn.commit()
                print(f"[fetch] EMPTY/UNREADABLE {canon_url}", flush=True)
                continue

            title, text, links, images, lang_hint = extract_text_and_links(html, canon_url)
            language = _guess_language(canon_url, lang_hint, text)
            image_alts = [alt for _, alt in images if alt]
            if image_alts:
                text = f"{text} {' '.join(image_alts)}".strip()

            try:
                doc_id, changed = index_document(
                    conn,
                    canon_url,
                    title,
                    text,
                    language=language,
                    next_crawl_after_hours=revisit_interval_hours,
                raw_html=html,
                crawl_run_id=crawl_run_id,
                )
                if changed:
                    pages_indexed += 1
                print(
                    f"[index] url={canon_url} title='{(title or canon_url)[:80]}' lang={language or '-'} changed={changed}",
                    flush=True,
                )
            except Exception:
                doc_id, changed = (None, False)
                print(f"[index] ERROR while indexing {canon_url}", flush=True)

            if doc_id is not None and changed:
                image_entries = images[:12]
                try:
                    records = _process_image_entries(image_entries, user_agent)
                    processed: List[Dict[str, object]] = []
                    for record in records:
                        hash_value = record.get("hash") or ""
                        if hash_value and dbmod.image_hash_exists(conn, hash_value):
                            record.pop("data", None)
                            continue
                        _save_image_assets(record)
                        processed.append(record)
                    dbmod.replace_doc_images(conn, doc_id, processed)
                    conn.commit()
                except Exception:
                    pass

            if auto_discover:
                mark_host(origin_host, "crawled", canon_url, retry_after=None, set_retry=True)
                conn.commit()

            if depth + 1 <= max_depth:
                before_q = len(queued)
                for link in links:
                    canon_link = _canonicalize(link)
                    prefer_front = _is_turkish_host(
                        urlparse(canon_link).netloc.lower(),
                        urlparse(canon_link).path,
                        urlparse(canon_link).query,
                    )
                    enqueue_discovered_url(
                        canon_link,
                        depth + 1,
                        origin_host,
                        canon_url,
                        prefer_front=prefer_front,
                    )
                added = max(0, len(queued) - before_q)
                if added or links:
                    print(f"[links] from={canon_url} found={len(links)} enqueued~={added}", flush=True)
        return pages_indexed
    finally:
        conn.close()
        if render_pool:
            render_pool.close()


def _worker_fetch_index(
    db_path: str,
    url: str,
    user_agent: str,
    revisit_interval_hours: float,
    render: bool = False,
    render_pool_size: int = 1,
    render_headless: bool = True,
    crawl_run_id: Optional[int] = None,
) -> Dict[str, object]:
    """Worker: fetch HTML, parse, index, process images. Returns links and status.
    Rendering in workers is disabled by default due to overhead.
    """
    result: Dict[str, object] = {"url": url, "ok": False, "links": [], "changed": False}
    try:
        conn = dbmod.open_db(db_path)
        dbmod.ensure_schema(conn)
    except Exception:
        return result

    try:
        print(f"[worker] GET {url}", flush=True)
        # For parallel mode, skip heavy render by default
        rp = None
        raw = _fetch_html(url, timeout=10, user_agent=user_agent, render_pool=rp if render else None)
        if not raw:
            return result
        html = _read_text(raw)
        if not html:
            return result
        title, text, links, images, lang_hint = extract_text_and_links(html, url)
        language = _guess_language(url, lang_hint, text)
        image_alts = [alt for _, alt in images if alt]
        if image_alts:
            text = f"{text} {' '.join(image_alts)}".strip()

        try:
            doc_id, changed = index_document(
                conn,
                url,
                title,
                text,
                language=language,
                next_crawl_after_hours=revisit_interval_hours,
                raw_html=html,
                crawl_run_id=crawl_run_id,
            )
        except Exception:
            doc_id, changed = (None, False)

        if doc_id is not None and changed:
            image_entries = images[:12]
            try:
                records = _process_image_entries(image_entries, user_agent)
                processed: List[Dict[str, object]] = []
                for record in records:
                    hash_value = record.get("hash") or ""
                    if hash_value and dbmod.image_hash_exists(conn, hash_value):
                        record.pop("data", None)
                        continue
                    _save_image_assets(record)
                    processed.append(record)
                dbmod.replace_doc_images(conn, doc_id, processed)
                conn.commit()
            except Exception:
                pass

        result.update({
            "ok": True,
            "links": links,
            "changed": bool(changed),
        })
        print(f"[worker] indexed url={url} changed={bool(changed)}", flush=True)
        return result
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _crawl_parallel(
    *,
    db_path: str,
    seed_state: Tuple[Deque[Tuple[str, int]], Set[str], Set[str], Set[str]],
    robots: RobotsCache,
    allowed_domains: Set[str],
    max_pages: int,
    max_depth: int,
    delay_seconds: float,
    auto_discover: bool,
    max_auto_discover_per_host: int,
    manual_sitemaps: Optional[List[str]],
    user_agent: str,
    revisit_interval_hours: float,
    render: bool,
    render_pool_size: int,
    render_headless: bool,
    failed_host_retry_hours: float,
    max_pages_per_host: Optional[int],
    sitemap_processed: Set[str],
    workers: int = 1,
    crawl_run_id: Optional[int] = None,
) -> int:
    q, queued, seen, domains = seed_state

    conn = dbmod.open_db(db_path)
    dbmod.ensure_schema(conn)

    pages_indexed = 0
    last_fetch_time: Dict[str, float] = {}
    host_processed: Dict[str, int] = defaultdict(int)

    def _retry_timestamp(base_hours: float) -> Optional[str]:
        if base_hours <= 0:
            return None
        jitter = random.uniform(0.1, 0.35) * max(base_hours, 0.1)
        delta_hours = max(0.1, base_hours + jitter)
        ts = datetime.utcnow() + timedelta(hours=delta_hours)
        return ts.isoformat(timespec="seconds") + "Z"

    def mark_host(host: str, status: str, source_url: Optional[str], *, retry_after: Optional[str] = None, set_retry: bool = False, failure: bool = False) -> None:
        if not auto_discover:
            return
        dbmod.record_discovered_host(conn, host, source_url, status=status)
        kwargs = {"failure_increment": failure}
        if set_retry:
            kwargs["retry_after"] = retry_after
        dbmod.update_discovered_host_status(conn, host, status, **kwargs)

    # Prepare initial due_docs and retry hosts like in single-thread path
    now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    due_docs = dbmod.get_due_documents(conn, now_iso, limit=max_pages * 2)
    for _, due_url in due_docs:
        canon_due = _canonicalize(due_url)
        host = urlparse(canon_due).netloc.lower()
        if host:
            domains.add(host)
        q.append((canon_due, 0))
        queued.add(canon_due)

    if auto_discover:
        retry_hosts = dbmod.get_hosts_for_retry(conn, now_iso, limit=max_pages * 2)
        for host in retry_hosts:
            if domains and not any(host == d or host.endswith("." + d) for d in domains):
                continue
            seed_url = f"https://{host}"
            domains.add(host)
            canon = _canonicalize(seed_url)
            if canon not in queued:
                q.append((canon, 0))
                queued.add(canon)
            mark_host(host, "queued", seed_url, retry_after=None, set_retry=True)
        if retry_hosts:
            conn.commit()

    # Process pool size driven by requested workers (bounded 1..16)
    pool_size = max(1, min(16, int(workers) if workers else 1))

    with ProcessPoolExecutor(max_workers=pool_size) as ex:
        inflight: Dict[object, Tuple[str, int, str]] = {}
        while pages_indexed < max_pages and (q or inflight):
            # Submit new tasks if capacity and ready URL exists
            while len(inflight) < pool_size and q:
                # Try to find a ready item w.r.t host delay and limits
                picked = None
                picked_idx = -1
                min_wait = None
                # scan up to current queue length
                q_len = len(q)
                for i in range(q_len):
                    cand_url, cand_depth = q[i]
                    if cand_url in seen:
                        continue
                    if not _same_domain(cand_url, allowed_domains):
                        continue
                    parsed = urlparse(cand_url)
                    origin_host = parsed.netloc.lower()
                    if max_pages_per_host is not None and host_processed[origin_host] >= int(max_pages_per_host):
                        continue

                    rules = robots.get_rules(parsed)
                    if not rules.allows(parsed.path or "/"):
                        if auto_discover:
                            retry_ts = _retry_timestamp(failed_host_retry_hours * 2 if failed_host_retry_hours > 0 else failed_host_retry_hours)
                            mark_host(origin_host, "blocked", cand_url, retry_after=retry_ts, set_retry=True, failure=True)
                            conn.commit()
                        seen.add(cand_url)
                        continue

                    # sitemap discovery (once per host)
                    host_base = f"{parsed.scheme}://{parsed.netloc}"
                    if host_base not in sitemap_processed:
                        for sitemap_url in robots.sitemap_urls(parsed):
                            sm = _canonicalize(sitemap_url)
                            sm_host = urlparse(sm).netloc.lower()
                            if sm_host == origin_host:
                                if sm not in queued:
                                    q.append((sm, 0))
                                    queued.add(sm)
                            else:
                                # enqueue discovered by domain logic below
                                pass
                        sitemap_processed.add(host_base)

                    host_delay = rules.crawl_delay if rules.crawl_delay is not None else 0.0
                    wait_target = max(delay_seconds, host_delay)
                    now = time.time()
                    last = last_fetch_time.get(origin_host, 0.0)
                    wait = wait_target - (now - last)
                    if wait <= 0:
                        picked = (cand_url, cand_depth, origin_host)
                        picked_idx = i
                        break
                    else:
                        if min_wait is None or wait < min_wait:
                            min_wait = wait

                if picked is None:
                    if min_wait is not None and min_wait > 0:
                        time.sleep(min_wait)
                    else:
                        break
                else:
                    # Remove picked from queue
                    q.rotate(-picked_idx)
                    item = q.popleft()
                    q.rotate(picked_idx)
                    cand_url, cand_depth = item
                    queued.discard(cand_url)
                    seen.add(cand_url)

                    # Submit to pool
                    host_processed[picked[2]] += 1
                    last_fetch_time[picked[2]] = time.time()
                    try:
                        print(f"[worker] GET {cand_url} depth={cand_depth}", flush=True)
                    except Exception:
                        pass
                    fut = ex.submit(
                        _worker_fetch_index,
                        db_path,
                        cand_url,
                        user_agent,
                        revisit_interval_hours,
                        False,  # render off in parallel workers by default
                        1,
                        True,
                        crawl_run_id,
                    )
                    inflight[fut] = (cand_url, cand_depth, picked[2])

            # Collect completions
            if inflight:
                done = []
                for fut in list(inflight.keys()):
                    if fut.done():
                        done.append(fut)
                if not done:
                    # Small sleep to avoid busy loop
                    time.sleep(0.01)
                    continue
                for fut in done:
                    cand_url, cand_depth, origin_host = inflight.pop(fut)
                    try:
                        res = fut.result()
                    except Exception:
                        res = {"ok": False, "links": []}
                    ok = bool(res.get("ok"))
                    changed = bool(res.get("changed"))
                    links = list(res.get("links") or [])
                    if not ok:
                        if auto_discover:
                            retry_ts = _retry_timestamp(failed_host_retry_hours)
                            mark_host(origin_host, "failed", cand_url, retry_after=retry_ts, set_retry=True, failure=True)
                            conn.commit()
                        try:
                            print(f"[worker] FAILED {cand_url}", flush=True)
                        except Exception:
                            pass
                    else:
                        if auto_discover:
                            mark_host(origin_host, "crawled", cand_url, retry_after=None, set_retry=True)
                            conn.commit()
                        if changed:
                            pages_indexed += 1
                        try:
                            print(f"[worker] indexed url={cand_url} changed={changed} links={len(links)}", flush=True)
                        except Exception:
                            pass
                        # enqueue discovered links
                        if cand_depth + 1 <= max_depth:
                            for link in links:
                                canon_link = _canonicalize(link)
                                prefer_front = _is_turkish_host(
                                    urlparse(canon_link).netloc.lower(),
                                    urlparse(canon_link).path,
                                    urlparse(canon_link).query,
                                )
                                # Internal enqueue: mimic single-thread behavior
                                link_parsed = urlparse(canon_link)
                                if link_parsed.scheme not in ("http", "https"):
                                    continue
                                link_host = link_parsed.netloc.lower()
                                if not link_host:
                                    continue
                                if link_host in allowed_domains or link_host == origin_host:
                                    if canon_link not in queued:
                                        if prefer_front:
                                            q.appendleft((canon_link, cand_depth + 1))
                                        else:
                                            q.append((canon_link, cand_depth + 1))
                                        queued.add(canon_link)
                                    continue
                                if not auto_discover:
                                    continue
                                # Auto-discovery limits
                                # For parallel path, we skip cross-domain robots pre-check here; single-thread
                                # branch will handle via normal flow when these URLs are popped.
                                if canon_link not in queued:
                                    q.append((canon_link, 0))
                                    queued.add(canon_link)

            # Stop if reached limit
            if pages_indexed >= max_pages:
                break

    conn.close()
    return pages_indexed


def crawl_loop(
    db_path: str,
    seed_urls: List[str],
    *,
    allowed_domains: Optional[List[str]] = None,
    max_pages: int = 100,
    max_depth: int = 2,
    delay_seconds: float = 0.5,
    auto_discover: bool = False,
    max_auto_discover_per_host: int = MAX_AUTO_DISCOVER_PER_HOST,
    manual_sitemaps: Optional[List[str]] = None,
    user_agent: str = USER_AGENT,
    render: bool = False,
    revisit_interval_hours: float = 24.0,
    render_pool_size: int = 5,
    render_headless: bool = False,
    failed_host_retry_hours: float = 12.0,
    cycle_sleep: float = 900.0,
    max_cycles: Optional[int] = None,
    max_pages_per_host: Optional[int] = None,
    workers: int = 1,
    crawl_run_id: Optional[int] = None,
) -> None:
    cycle = 0
    while True:
        pages = crawl(
            db_path,
            seed_urls,
            allowed_domains=allowed_domains,
            max_pages=max_pages,
            max_depth=max_depth,
            delay_seconds=delay_seconds,
            auto_discover=auto_discover,
            max_auto_discover_per_host=max_auto_discover_per_host,
            manual_sitemaps=manual_sitemaps,
            user_agent=user_agent,
            render=render,
            revisit_interval_hours=revisit_interval_hours,
            render_pool_size=render_pool_size,
            render_headless=render_headless,
            failed_host_retry_hours=failed_host_retry_hours,
            max_pages_per_host=max_pages_per_host,
            workers=workers,
            crawl_run_id=crawl_run_id,
        )
        cycle += 1
        print(f"[scheduler] cycle={cycle} pages_indexed={pages}")
        if max_cycles is not None and cycle >= max_cycles:
            break
        if cycle_sleep > 0:
            # Dinamik bekleme: bir sonraki due belgeye kadar bekleme sÃ¼resini sÄ±nÄ±rlÄ± tut
            try:
                from datetime import datetime
                now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"
                conn = dbmod.open_db(db_path)
                dbmod.ensure_schema(conn)
                next_due = dbmod.get_next_due_after(conn, now_iso)
                try:
                    conn.close()
                except Exception:
                    pass
                sleep_for = cycle_sleep
                if next_due:
                    try:
                        # next_due format: YYYY-MM-DDTHH:MM:SSZ
                        dt_next = datetime.strptime(next_due, "%Y-%m-%dT%H:%M:%SZ")
                        dt_now = datetime.strptime(now_iso, "%Y-%m-%dT%H:%M:%SZ")
                        diff = (dt_next - dt_now).total_seconds()
                        if diff > 0:
                            sleep_for = min(cycle_sleep, diff)
                    except Exception:
                        pass
                time.sleep(max(0.5, sleep_for))
            except Exception:
                time.sleep(cycle_sleep)


