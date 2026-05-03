from __future__ import annotations

from html.parser import HTMLParser
from typing import List, Tuple
from urllib.parse import urljoin


class _TextAndLinkParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self._in_ignored = 0  # script/style/noscript
        self._title = []
        self._in_title = False
        self._texts: List[str] = []
        self._links: List[str] = []
        self._images: List[Tuple[str, str]] = []  # (url, alt)
        self._lang_hint: str = ""

    def handle_starttag(self, tag: str, attrs):
        attr_map = {k.lower(): v for k, v in attrs if v is not None}
        if tag in ("script", "style", "noscript"):
            self._in_ignored += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "html":
            lang = attr_map.get("lang") or attr_map.get("xml:lang")
            if lang:
                self._lang_hint = lang.strip()
        elif tag == "meta":
            http_equiv = (attr_map.get("http-equiv") or "").lower()
            name = (attr_map.get("name") or "").lower()
            content = attr_map.get("content") or ""
            if http_equiv == "content-language" and content:
                self._lang_hint = content.split(",")[0].strip()
            elif name in ("language", "lang") and content:
                self._lang_hint = content.split(",")[0].strip()
        elif tag == "a":
            href = None
            for k, v in attrs:
                if k.lower() == "href":
                    href = v
                    break
            if href:
                href = href.strip()
                if any(href.startswith(s) for s in ("javascript:", "mailto:", "tel:", "data:")):
                    return
                try:
                    abs_url = urljoin(self.base_url, href)
                    self._links.append(abs_url)
                except Exception:
                    pass
        elif tag == "img":
            src = None
            alt = ""
            for k, v in attrs:
                kl = k.lower()
                if kl == "src":
                    src = v
                elif kl == "data-src" and not src:
                    src = v
                elif kl == "alt" and v:
                    alt = v
            if src:
                src = src.strip()
                if src and not src.startswith("data:"):
                    try:
                        abs_url = urljoin(self.base_url, src)
                        self._images.append((abs_url, alt.strip()))
                    except Exception:
                        pass

    def handle_endtag(self, tag: str):
        if tag in ("script", "style", "noscript") and self._in_ignored > 0:
            self._in_ignored -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str):
        if self._in_title:
            self._title.append(data)
        if self._in_ignored == 0 and data and not data.isspace():
            self._texts.append(data)

    def error(self, message):
        pass

    def result(self) -> Tuple[str, str, List[str], List[Tuple[str, str]], str]:
        title = " ".join(self._title).strip()
        text = " ".join(self._texts)
        return title, text, self._links, self._images, (self._lang_hint or "")


def extract_text_and_links(
    html: str, base_url: str
) -> Tuple[str, str, List[str], List[Tuple[str, str]], str]:
    parser = _TextAndLinkParser(base_url)
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.result()
