from __future__ import annotations

import html
import os
import json
from collections import OrderedDict
import random
import time
import hmac
import hashlib
import base64
import secrets
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

from . import ai_platform as aimod
from . import db as dbmod
from .search import search_with_fuzzy, snippets
from .tokenize import tokenize

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None


HTML_HEAD = """
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>yox</title>
  <style>
    /* Logo renk paleti: kÄ±rmÄ±zÄ± ve koyu turuncu tonlarÄ± */
    :root { --c1:#E53935; --c2:#D84315; --c3:#FF6F00; }
    body { font-family: Arial, Helvetica, sans-serif; margin:0; color:#202124; }
    a { color:#1a73e8; text-decoration:none; }
    a:hover { text-decoration:underline; }

    /* Home */
    .home { display:flex; flex-direction:column; align-items:center; min-height:100vh; padding:0 16px; }
    .logo { font-size: 84px; font-weight: 700; letter-spacing:3px; margin-top: 120px; }
    .logo .y{ color: var(--c1);} .logo .o{ color: var(--c2);} .logo .x{ color: var(--c3);} 
    .search-form { display:flex; flex-direction:column; align-items:center; gap:12px; width:100%; max-width:640px; margin-top:28px; }
    .search-form .search-row { width:100%; }
    .search-form input[type=text] { width:100%; padding:12px 18px; border:1px solid #dadce0; border-radius:999px; font-size:16px; outline:none; }
    .search-form input[type=text]:focus { box-shadow:0 1px 6px rgba(32,33,36,0.28); border-color:transparent; }
    .search-form .control-row { display:flex; gap:12px; justify-content:center; align-items:center; flex-wrap:wrap; width:100%; }
    .search-form select { padding:10px 14px; border:1px solid #dadce0; border-radius:999px; background:#fff; font-size:14px; min-width:120px; }
    .search-form .actions { display:flex; gap:10px; }
    .btn { background:#f8f9fa; border:1px solid #dadce0; border-radius:999px; padding:8px 18px; font-size:14px; cursor:pointer; }
    .btn:hover { background:#f1f3f4; }
    .muted { color:#70757a; font-size:12px; }

    .quick-menu { margin-top:16px; text-align:center; }
    .quick-menu summary { list-style:none; cursor:pointer; padding:8px 16px; border:1px solid #dadce0; border-radius:999px; background:#f8f9fa; display:inline-flex; align-items:center; gap:6px; font-size:14px; }
    .quick-menu summary::-webkit-details-marker { display:none; }
    .quick-menu[open] summary { background:#e8f0fe; border-color:#d2e3fc; color:#1a73e8; }
    .quick-menu nav { margin-top:12px; display:flex; gap:12px; flex-wrap:wrap; justify-content:center; }
    .quick-menu nav a { color:#3c4043; text-decoration:none; padding:6px 12px; border-radius:999px; border:1px solid transparent; }
    .quick-menu nav a:hover { border-color:#dadce0; background:#f1f3f4; }

    /* Results */
    .topbar { display:flex; align-items:center; gap:12px; padding:12px 16px; border-bottom:1px solid #eee; flex-wrap:wrap; }
    .topbar .logo { font-size:28px; margin:0; }
    .top-search { display:flex; gap:12px; align-items:center; flex-wrap:wrap; flex:1 1 320px; }
    .top-search .input-wrap { display:flex; gap:8px; align-items:center; flex:1 1 220px; }
    .top-search input[type=text] { flex:1; padding:10px 14px; border:1px solid #dadce0; border-radius:24px; font-size:16px; }
    .top-search select { padding:8px 12px; border:1px solid #dadce0; border-radius:999px; background:#fff; font-size:14px; }
    .top-search .actions { display:flex; gap:6px; }
    .navlinks { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    .navlinks a { color:#5f6368; font-size:14px; padding:6px 12px; border-radius:999px; border:1px solid transparent; }
    .navlinks a.active { color:#1a73e8; border-color:#d2e3fc; background:#e8f0fe; }
    .navlinks details.quick-menu { margin-left:4px; }
    .navlinks details.quick-menu summary { padding:6px 10px; font-size:13px; }

    .container { max-width: 720px; margin: 16px auto; padding: 0 16px; }
    .result { display:flex; gap:14px; padding: 12px 0; border-bottom: 1px solid #eee; }
    .thumb { flex:0 0 140px; }
    .thumb img { width:140px; height:90px; object-fit:cover; border-radius:12px; background:#f1f3f4; }
    .rbody { flex:1; }
    .title { font-weight: 600; font-size: 18px; }
    .snippet { color: #4d5156; margin-top:4px; }
    .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(160px,1fr)); gap:12px; padding:16px; }
    .grid-item { background:#fff; border:1px solid #dadce0; border-radius:12px; overflow:hidden; display:flex; flex-direction:column; }
    .grid-item img { width:100%; height:140px; object-fit:cover; background:#f1f3f4; }
    .grid-item .meta { padding:8px 10px; font-size:12px; color:#5f6368; }
    .badge { position:fixed; bottom:16px; right:16px; background:rgba(32,33,36,0.75); color:#fff; padding:6px 14px; border-radius:999px; font-size:12px; letter-spacing:1.5px; text-transform:uppercase; }
    /* AI Answer */
    .ai-wrap { border-radius:16px; padding:2px; background: linear-gradient(135deg, #FFD166, #FCA311, #FB8500); }
    .ai-card { background:#fff; border-radius:14px; padding:16px 18px; }
    .ai-head { display:flex; align-items:center; gap:8px; font-weight:600; margin-bottom:8px; }
    .ai-body { white-space:pre-wrap; line-height:1.5; color:#202124; }
    .ai-muted { color:#5f6368; font-size:12px; margin-top:6px; }

    /* Theme 2 â€” glass/blur with blue/teal gradient */
    body.t2 { background: radial-gradient(1200px 700px at 20% 0%, #c7f1ff 0%, #8ec5ff 40%, #2d6cdf 100%) fixed; }
    .t2 .topbar, .t2 .home .search-form, .t2 .container, .t2 .result, .t2 .grid-item, .t2 .ai-card {
      background: rgba(255,255,255,0.22);
      border: 1px solid rgba(255,255,255,0.35);
      box-shadow: 0 6px 24px rgba(0,0,0,0.15);
      backdrop-filter: blur(12px) saturate(120%);
      -webkit-backdrop-filter: blur(12px) saturate(120%);
      border-radius: 16px;
    }
    .t2 .search-form input[type=text], .t2 .top-search input[type=text] {
      background: rgba(255,255,255,0.35);
      border-color: rgba(255,255,255,0.55);
    }
    .t2 .btn { background: rgba(255,255,255,0.55); border-color: rgba(255,255,255,0.6); }
    .t2 .navlinks a.active { background: rgba(255,255,255,0.35); border-color: rgba(255,255,255,0.55); }
    .t2 .thumb img, .t2 .grid-item img { border-radius: 14px; }
  </style>
  </head>
<body>
"""

HTML_FOOT = """
  <div class="badge">v2</div>
  </body>
</html>
"""


def _logo_html(size_class: str = "logo") -> str:
    return f"<div class='{size_class}'><span class='y'>y</span><span class='o'>o</span><span class='x'>x</span></div>"

def _head_with_theme(theme_cls: str) -> str:
    if theme_cls:
        return HTML_HEAD.replace("<body>", f"<body class='{theme_cls}'>")
    return HTML_HEAD


LANG_OPTIONS = [("", "Tum"), ("tr", "Turkce"), ("en", "Ingilizce")]


def _lang_options_html(selected: str) -> str:
    selected = (selected or "").strip().lower()
    options = []
    for code, label in LANG_OPTIONS:
        code_html = html.escape(code, quote=True)
        sel = " selected" if code == selected else ""
        options.append(f"<option value='{code_html}'{sel}>{html.escape(label)}</option>")
    return "".join(options)


def _mistral_answer(conn, q: str, lang: str = "") -> Tuple[str, str]:
    """Call Mistral API to get an AI answer. Returns (text, error)."""
    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not api_key:
        return "", "MISTRAL_API_KEY tanÄ±mlÄ± deÄŸil. LÃ¼tfen bir API anahtarÄ± saÄŸlayÄ±n."
    if requests is None:
        return "", "requests modÃ¼lÃ¼ yÃ¼klÃ¼ deÄŸil. `pip install requests` yapÄ±n."
    base = os.environ.get("MISTRAL_API_BASE", "https://api.mistral.ai")
    model = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
    url = base.rstrip("/") + "/v1/chat/completions"
    # Use top results as lightweight context
    items, _, used = search_with_fuzzy(conn, q, top_k=5, fuzzy=True, preferred_lang=(lang or None))
    refs = []
    for doc_id, _ in items:
        try:
            _, urlx, title, content, _, _ = dbmod.get_doc(conn, doc_id)
            snip = snippets(content, used)
            refs.append(f"- {title or urlx}: {urlx}\n  {snip}")
        except Exception:
            pass
    ref_text = "\n".join(refs)
    prompt = (
        "KullanÄ±cÄ± sorusu: " + q.strip() + "\n\n"
        "AÅŸaÄŸÄ±daki referanslara dayanarak kÄ±sa ve doÄŸru bir yanÄ±t ver. "
        "MÃ¼mkÃ¼nse maddeler halinde Ã¶zetle. BilmediÄŸin kÄ±sÄ±mlarda tahmin yÃ¼rÃ¼tme.\n\n"
        + ("Referanslar:\n" + ref_text if ref_text else "")
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "YanÄ±t dili: TÃ¼rkÃ§e. KÄ±sa ve Ã¶z yanÄ±t ver."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 600,
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=20)
        if resp.status_code != 200:
            return "", f"Mistral API hatasÄ±: HTTP {resp.status_code}"
        js = resp.json()
        # OpenAI-compatible style
        choices = js.get("choices") or []
        if choices:
            content = choices[0].get("message", {}).get("content", "").strip()
        else:
            content = (js.get("output") or "").strip()
        return content, ""
    except Exception as e:
        return "", f"Mistral API Ã§aÄŸrÄ±sÄ± baÅŸarÄ±sÄ±z: {e}"


def _render_ai_results(conn, q: str, lang: str = "", expand: bool = False, theme: str = "") -> bytes:
    qh = html.escape(q)
    qh_attr = html.escape(q, quote=True)
    lang_val = (lang or "").strip().lower()
    lang_html = html.escape(lang_val, quote=True)
    options_html = _lang_options_html(lang_val)
    theme_val = "t2" if theme else ""
    theme_options = (
        "<option value=''" + (" selected" if not theme_val else "") + ">Klasik</option>"
        + "<option value='t2'" + (" selected" if theme_val == "t2" else "") + ">Cam (t2)</option>"
    )
    expand_checked = " checked" if expand else ""
    lang_link_suffix = f"&amp;lang={lang_html}" if lang_val else ""
    lang_param = f"?lang={lang_html}" if lang_val else ""
    params = [f"q={qh_attr}"]
    if lang_val:
        params.append(f"lang={lang_html}")
    if expand:
        params.append("expand=1")
    if theme:
        params.append("theme=t2")
    image_href = "/images?" + "&amp;".join(params)
    all_href = f"/?q={qh_attr}{lang_link_suffix}{('&amp;expand=1' if expand else '')}{('&amp;theme=t2' if theme else '')}"
    menu_detail = (
        "<details class='quick-menu mini'>"
        "<summary>â˜°</summary>"
        "<nav>"
        f"<a href='/about{lang_param}'>HakkÄ±nda</a>"
        f"<a href='/help{lang_param}'>YardÄ±m</a>"
        f"<a href='/status{lang_param}'>Durum</a>"
        "</nav>"
        "</details>"
    )
    body: List[str] = [
        _head_with_theme(theme),
        "<div class='topbar'>",
        _logo_html("logo"),
        "<form method='get' action='/ai' class='top-search'>",
        "<div class='input-wrap'>",
        f"<input type='text' name='q' value='{qh}' placeholder='Soru sor' autofocus/>",
        f"<select name='lang'>{options_html}</select>",
        f"<select name='theme' style='min-width:140px'><option value=''>Tema</option>" + theme_options + "</select>",
        f"<label><input type='checkbox' name='expand' value='1'{expand_checked}/> Anlamca geniÅŸlet</label>",
        "</div>",
        "<div class='actions'>",
        "<button class='btn' type='submit'>Sor</button>",
        "</div>",
        "</form>",
        f"<div class='navlinks'><a href='{all_href}'>TÃ¼mÃ¼</a><a href='{image_href}'>GÃ¶rseller</a><a class='active' href='/ai?q={qh_attr}{lang_link_suffix}'>AI</a>{menu_detail}</div>",
        "</div>",
        "<div class='container'>",
    ]

    content, err = _mistral_answer(conn, q, lang_val)
    if err:
        content_html = f"<div class='ai-muted'>{html.escape(err)}</div>"
    else:
        content_html = html.escape(content)
    body.extend([
        "<div class='ai-wrap'>",
        "<div class='ai-card'>",
        "<div class='ai-head'>Yapay Zeka YanÄ±tÄ±</div>",
        f"<div class='ai-body'>{content_html}</div>",
        "</div>",
        "</div>",
        "</div>",
        HTML_FOOT,
    ])
    return "".join(body).encode("utf-8")


def _ai_card_probability() -> float:
    try:
        envv = os.environ.get("AI_CARD_PROB")
        if envv is not None:
            p = float(envv)
            if 0.0 <= p <= 1.0:
                return p
    except Exception:
        pass
    return 0.20


def _ai_card_html(text: str) -> str:
    content_html = html.escape(text)
    return (
        "<div class='ai-wrap'>"
        "<div class='ai-card'>"
        "<div class='ai-head'>Yapay Zeka YanÄ±tÄ±</div>"
        f"<div class='ai-body'>{content_html}</div>"
        "</div>"
        "</div>"
    )
def _render_home(q: str, lang: str = "", expand: bool = False, theme: str = "") -> bytes:
    qh = html.escape(q or "")
    qh_attr = html.escape(q or "", quote=True)
    lang_val = (lang or "").strip().lower()
    lang_html = html.escape(lang_val, quote=True)
    options_html = _lang_options_html(lang_val)
    image_params = []
    if q:
        image_params.append(f"q={qh_attr}")
    if lang_val:
        image_params.append(f"lang={lang_html}")
    if theme:
        image_params.append("theme=t2")
    image_href = "/images"
    if image_params:
        image_href += "?" + "&amp;".join(image_params)
    lang_param = f"?lang={lang_html}" if lang_val else ""
    menu_html = (
        "<details class='quick-menu home-menu'>"
        "<summary>â˜° MenÃ¼</summary>"
        "<nav>"
        f"<a href='/about{lang_param}'>HakkÄ±nda</a>"
        f"<a href='/help{lang_param}'>YardÄ±m</a>"
        f"<a href='{image_href}'>GÃ¶rseller</a>"
        f"<a href='/status{lang_param}'>Durum</a>"
        "</nav>"
        "</details>"
    )
    body = [
        _head_with_theme(theme),
        "<div class='home'>",
        _logo_html("logo"),
        "<form method='get' action='/' class='search-form'>",
        f"<div class='search-row'><input type='text' name='q' value='{qh}' placeholder='Ara' autofocus/></div>",
        "<div class='control-row'>",
        f"<select name='lang'>{options_html}</select>",
        "<div class='actions'>",
        "<button class='btn' type='submit'>Ara</button>",
        "<button class='btn' type='submit' name='lucky' value='1'>ÅansÄ±mÄ± dene</button>",
        "</div>",
        "</div>",
        "</form>",
        menu_html,
        "<div style='height:48px'></div>",
        "</div>",
        HTML_FOOT,
    ]
    return "".join(body).encode("utf-8")


def _render_tanitim() -> bytes:
    parts = [
        HTML_HEAD,
        "<div class='container'>",
        _logo_html("logo"),
        "<h1>Yox â€” Hafif ve HÄ±zlÄ± Arama Motoru</h1>",
        "<p>Yox; Python + SQLite ile yazÄ±lmÄ±ÅŸ, crawler (tarayÄ±cÄ±) tabanlÄ±, yerel/Ã¶zel kullanÄ±m iÃ§in pratik bir arama motorudur. TÃ¼rkÃ§e odaklÄ±dÄ±r, Ä°ngilizceyi de destekler.</p>",
        "<h2>Ã–ne Ã‡Ä±kanlar</h2>",
        "<ul>"
        "<li>BM25 skorlama ile ilgili ve dengeli sonuÃ§lar</li>"
        "<li>TÃ¼rkÃ§e ve Ä°ngilizce iÃ§in temel durak kelime (stopwords) desteÄŸi</li>"
        "<li>GÃ¶rsellerin otomatik tespiti, meta bilgiler ve kÃ¼Ã§Ã¼k Ã¶nizleme (thumbnail)</li>"
        "<li>Robots.txt ve sitemap desteÄŸi; &ouml;zenli gecikme (crawl-delay)</li>"
        "<li>G&uuml;nl&uuml;k yeniden tarama i&ccedil;in kolay mod: <code>--daily</code></li>"
        "<li>&Ccedil;ok &ccedil;ekirdekli tarama: <code>--workers N</code></li>"
        "<li>AI yanÄ±t kartÄ± (Mistral API) â€” sonu&ccedil;larda %20 olasÄ±lÄ±kla &uuml;stte</li>"
        "</ul>",
        "<h2>HÄ±zlÄ± BaÅŸlangÄ±Ã§</h2>",
        "<pre><code>python -m donersearch --db donersearch.db crawl example.com \\\n+  --allowed-domains example.com --max-pages 100 --workers 4\n\n"
        "python -m donersearch --db donersearch.db serve --host 0.0.0.0 --port 8000\n"
        "# AI i&ccedil;in ekleyin:\n"
        "python -m donersearch --db donersearch.db serve --host 0.0.0.0 --port 8000 \\\n+  --mistral-key &quot;API_ANAHTARINIZ&quot; --ai-model mistral-small-latest</code></pre>",
        "<h2>KullanÄ±m Ä°pu&ccedil;larÄ±</h2>",
        "<ul>"
        "<li>Alan i&ccedil;inde kalmak i&ccedil;in <code>--allowed-domains</code> kullanÄ±n.</li>"
        "<li>AÄŸÄ±r sitelerde <code>--delay</code> ve <code>--max-depth</code> ile taramayÄ± nazik ve kontroll&uuml; tutun.</li>"
        "<li>G&ouml;rsel ve HTML i&ccedil;erikleri VDS istikrarÄ± i&ccedil;in boyutla sÄ±nÄ±rlandÄ±rÄ±lÄ±r.</li>"
        "</ul>",
        "<h2>Gizlilik</h2>",
        "<p>T&uuml;m i&ccedil;erik yerel SQLite veritabanÄ±nda saklanÄ±r. &Ccedil;evrimdÄ±ÅŸÄ± &ccedil;alÄ±ÅŸtÄ±rÄ±labilir.</p>",
        "<div style='height:24px'></div>",
        "</div>",
        HTML_FOOT,
    ]
    return "".join(parts).encode("utf-8")

def _render_results(conn, q: str, lang: str = "", expand: bool = False, theme: str = "") -> bytes:
    qh = html.escape(q)
    qh_attr = html.escape(q, quote=True)
    lang_val = (lang or "").strip().lower()
    lang_html = html.escape(lang_val, quote=True)
    options_html = _lang_options_html(lang_val)
    items, corrections, used_terms = search_with_fuzzy(
        conn,
        q,
        top_k=20,
        fuzzy=True,
        preferred_lang=lang_val or None,
        expand=expand,
    )
    lang_link_suffix = f"&amp;lang={lang_html}" if lang_val else ""
    lang_param = f"?lang={lang_html}" if lang_val else ""
    image_params = [f"q={qh_attr}"]
    if lang_val:
        image_params.append(f"lang={lang_html}")
    if expand:
        image_params.append("expand=1")
    if theme:
        image_params.append("theme=t2")
    image_href = "/images?" + "&amp;".join(image_params)
    ai_href = "/ai?" + "&amp;".join(image_params)
    menu_detail = (
        "<details class='quick-menu mini'>"
        "<summary>â˜°</summary>"
        "<nav>"
        f"<a href='/about{lang_param}'>HakkÄ±nda</a>"
        f"<a href='/help{lang_param}'>YardÄ±m</a>"
        f"<a href='/status{lang_param}'>Durum</a>"
        "</nav>"
        "</details>"
    )
    html_parts = [
        _head_with_theme(theme),
        "<div class='topbar'>",
        _logo_html("logo"),
        "<form method='get' action='/' class='top-search'>",
        "<div class='input-wrap'>",
        f"<input type='text' name='q' value='{qh}' placeholder='Ara' autofocus/>",
        f"<select name='lang'>{options_html}</select>",
        f"<label><input type='checkbox' name='expand' value='1'{(' checked' if 'expand' in locals() and expand else '')}/> Anlamca geniÅŸlet</label>",
        "</div>",
        "<div class='actions'>",
        "<button class='btn' type='submit'>Ara</button>",
        "<button class='btn' type='submit' name='lucky' value='1'>ÅansÄ±mÄ± dene</button>",
        "</div>",
        "</form>",
        f"<div class='navlinks'><a class='active' href='/?q={qh_attr}{lang_link_suffix}{('&amp;expand=1' if expand else '')}{('&amp;theme=t2' if theme else '')}'>TÃ¼mÃ¼</a><a href='{image_href}'>GÃ¶rseller</a><a href='{ai_href}'>AI</a>{menu_detail}</div>",
        "</div>",
        "<div class='container'>",
        f"<p class='muted'>{len(items)} sonuÃ§</p>",
    ]
    # 20% olasÄ±lÄ±kla (AI_CARD_PROB ile deÄŸiÅŸtirilebilir) AI kartÄ±nÄ± en Ã¼stte gÃ¶ster
    try:
        if (os.environ.get("MISTRAL_API_KEY") or "") and requests is not None and q.strip():
            if random.random() < _ai_card_probability():
                content, err = _mistral_answer(conn, q, lang_val)
                if content and not err:
                    html_parts.append(_ai_card_html(content))
    except Exception:
        pass

    if corrections:
        sug = []
        for src, dst in corrections.items():
            sug.append(f"{html.escape(src)} â†’ <strong>{html.escape(dst)}</strong>")
        html_parts.append(f"<p class='muted'>DÃ¼zeltildi: {' , '.join(sug)}</p>")
    doc_ids = [doc_id for doc_id, _ in items]
    images_map = dbmod.get_images_for_docs(conn, doc_ids)
    for doc_id, score in items:
        _, url, title, content, _, doc_lang = dbmod.get_doc(conn, doc_id)
        title = title or url
        snip = snippets(content, used_terms)
        image_infos = images_map.get(doc_id) or []
        seen_local: set[str] = set()
        thumb_info: Optional[Dict[str, object]] = None
        for info in image_infos:
            img_url = ""
            if isinstance(info, dict):
                img_url = str(info.get("url") or "")
            else:
                img_url = str(info[0] if len(info) > 0 else "")
            if not img_url or img_url in seen_local:
                continue
            seen_local.add(img_url)
            if thumb_info is None:
                thumb_info = info if isinstance(info, dict) else {"url": img_url, "alt": info[1] if len(info) > 1 else ""}
        img_html = ""
        if thumb_info:
            display_src = ""
            alt_text = ""
            if isinstance(thumb_info, dict):
                display_src = str(thumb_info.get("thumbnail_path") or thumb_info.get("url") or "")
                alt_text = str(thumb_info.get("alt") or title or "")
            else:
                display_src = str(thumb_info[0])
                alt_text = str(thumb_info[1] or title or "")
            if display_src and not display_src.startswith("http"):
                display_src = thumb_info.get("url") if isinstance(thumb_info, dict) else display_src
            img_src = html.escape(display_src or "")
            img_alt = html.escape(alt_text)
            img_html = f"<div class='thumb'><img src='{img_src}' alt='{img_alt}' loading='lazy'/></div>"
        lang_display = f" | Dil: {html.escape(doc_lang)}" if doc_lang else ""
        html_parts.append(
            """
            <div class="result">
              {thumb}
              <div class="rbody">
                <div class="title"><a class="url" href="{url}" target="_blank" rel="noreferrer">{title}</a></div>
                <div class="snippet">{snippet}</div>
                <div class="muted">Skor: {score:.3f}{lang_display}</div>
              </div>
            </div>
            """.format(
                thumb=img_html or "",
                url=html.escape(url),
                title=html.escape(title),
                snippet=html.escape(snip),
                score=score,
                lang_display=lang_display,
            )
        )
    html_parts.append("</div>")
    html_parts.append(HTML_FOOT)
    return "".join(html_parts).encode("utf-8")


def _render_image_results(conn, q: str, lang: str = "", expand: bool = False, theme: str = "") -> bytes:
    qh = html.escape(q)
    qh_attr = html.escape(q, quote=True)
    lang_val = (lang or "").strip().lower()
    lang_html = html.escape(lang_val, quote=True)
    options_html = _lang_options_html(lang_val)
    expand_checked = " checked" if expand else ""
    items, corrections, used_terms = search_with_fuzzy(
        conn,
        q,
        top_k=40,
        fuzzy=True,
        preferred_lang=lang_val or None,
        expand=expand,
    )
    lang_link_suffix = f"&amp;lang={lang_html}" if lang_val else ""
    lang_param = f"?lang={lang_html}" if lang_val else ""
    image_params = [f"q={qh_attr}"]
    if lang_val:
        image_params.append(f"lang={lang_html}")
    if expand:
        image_params.append("expand=1")
    if theme:
        image_params.append("theme=t2")
    image_href = "/images?" + "&amp;".join(image_params)
    ai_href = "/ai?" + "&amp;".join(image_params)
    menu_detail = (
        "<details class='quick-menu mini'>"
        "<summary>â˜°</summary>"
        "<nav>"
        f"<a href='/about{lang_param}'>HakkÄ±nda</a>"
        f"<a href='/help{lang_param}'>YardÄ±m</a>"
        f"<a href='/status{lang_param}'>Durum</a>"
        "</nav>"
        "</details>"
    )
    html_parts = [
        _head_with_theme(theme),
        "<div class='topbar'>",
        _logo_html("logo"),
        "<form method='get' action='/images' class='top-search'>",
        "<div class='input-wrap'>",
        f"<input type='text' name='q' value='{qh}' placeholder='Ara' autofocus/>",
        f"<select name='lang'>{options_html}</select>",
        f"<label><input type='checkbox' name='expand' value='1'{expand_checked}/> Anlamca geniÅŸlet</label>",
        "</div>",
        "<div class='actions'>",
        "<button class='btn' type='submit'>Ara</button>",
        "<button class='btn' type='submit' name='lucky' value='1'>ÅansÄ±mÄ± dene</button>",
        "</div>",
        "</form>",
        f"<div class='navlinks'><a href='/?q={qh_attr}{lang_link_suffix}{('&amp;expand=1' if expand else '')}{('&amp;theme=t2' if theme else '')}'>TÃ¼mÃ¼</a><a class='active' href='{image_href}'>GÃ¶rseller</a><a href='{ai_href}'>AI</a>{menu_detail}</div>",
        "</div>",
    ]
    if corrections:
        sug = []
        for src, dst in corrections.items():
            sug.append(f"{html.escape(src)} â†’ <strong>{html.escape(dst)}</strong>")
        html_parts.append(f"<div class='container'><p class='muted'>DÃ¼zeltildi: {' , '.join(sug)}</p></div>")
    doc_ids = [doc_id for doc_id, _ in items]
    images_map = dbmod.get_images_for_docs(conn, doc_ids)
    seen_urls: set[str] = set()
    cards = []
    for doc_id, score in items:
        page_images = images_map.get(doc_id) or []
        if not page_images:
            continue
        _, page_url, title, _, _, doc_lang = dbmod.get_doc(conn, doc_id)
        count = 0
        for info in page_images:
            img_url = info.get("url") if isinstance(info, dict) else None
            if not img_url or img_url in seen_urls:
                continue
            seen_urls.add(img_url)
            cards.append({
                "image": img_url,
                "alt": (info.get("alt") if isinstance(info, dict) else None) or title or page_url,
                "page": page_url,
                "doc_lang": doc_lang,
                "width": info.get("width") if isinstance(info, dict) else 0,
                "height": info.get("height") if isinstance(info, dict) else 0,
                "format": info.get("format") if isinstance(info, dict) else "",
                "size_bytes": info.get("size_bytes") if isinstance(info, dict) else 0,
                "thumbnail": info.get("thumbnail_path") if isinstance(info, dict) else "",
            })
            count += 1
            if len(cards) >= 60 or count >= 3:
                break
        if len(cards) >= 60:
            break
    if cards:
        html_parts.append("<div class='grid'>")
        for info in cards:
            img_src = info.get("thumbnail") or info.get("image") or ""
            if img_src and not str(img_src).startswith("http"):
                img_src = info.get("image") or ""
            alt_text = info.get("alt") or ""
            meta_bits: List[str] = []
            width = info.get("width") or 0
            height = info.get("height") or 0
            fmt = info.get("format") or ""
            size_bytes = info.get("size_bytes") or 0
            if width and height:
                meta_bits.append(f"{width}x{height}px")
            if fmt:
                meta_bits.append(fmt.upper())
            if size_bytes:
                meta_bits.append(f"{size_bytes/1024:.1f} KB")
            meta_text = " | ".join(meta_bits)
            extra_meta = f"<div class='meta muted'>{html.escape(meta_text)}</div>" if meta_text else ""
            lang_meta = info.get("doc_lang") or ""
            lang_html = f"<div class='meta' style='font-weight:600'>{html.escape(lang_meta)}</div>" if lang_meta else ""
            html_parts.append(
                """
                <a class="grid-item" href="{page}" target="_blank" rel="noreferrer">
                  <img src="{img}" alt="{alt}" loading="lazy" />
                  <div class="meta">{alt}</div>
                  {extra}
                  {lang}
                </a>
                """.format(
                    page=html.escape(info.get("page") or ""),
                    img=html.escape(str(img_src or "")),
                    alt=html.escape(alt_text),
                    extra=extra_meta,
                    lang=lang_html,
                )
            )
        html_parts.append("</div>")
    else:
        html_parts.append(f"<div class='container'><p class='muted'>'{qh}' iÃ§in gÃ¶rsel bulunamadÄ±.</p></div>")
    html_parts.append(HTML_FOOT)
    return "".join(html_parts).encode("utf-8")


def _render_recent_sites(conn, q: str, lang: str = "", expand: bool = False, theme: str = "") -> bytes:
    # Show recently indexed sites (unique hosts) using last_crawled ordering
    from urllib.parse import urlparse
    qh = html.escape(q)
    qh_attr = html.escape(q, quote=True)
    lang_val = (lang or "").strip().lower()
    lang_html = html.escape(lang_val, quote=True)
    options_html = _lang_options_html(lang_val)
    expand_checked = " checked" if expand else ""
    image_params = [f"q={qh_attr}"]
    if lang_val:
        image_params.append(f"lang={lang_html}")
    if expand:
        image_params.append("expand=1")
    if theme:
        image_params.append("theme=t2")
    image_href = "/images?" + "&amp;".join(image_params)
    ai_href = "/ai?" + "&amp;".join(image_params)
    lang_link_suffix = f"&amp;lang={lang_html}" if lang_val else ""
    menu_detail = (
        "<details class='quick-menu mini'>"
        "<summary>â˜°</summary>"
        "<nav>"
        f"<a href='/about'>HakkÄ±nda</a>"
        f"<a href='/help'>YardÄ±m</a>"
        f"<a href='/status'>Durum</a>"
        "</nav>"
        "</details>"
    )
    html_parts = [
        _head_with_theme(theme),
        "<div class='topbar'>",
        _logo_html("logo"),
        "<form method='get' action='/' class='top-search'>",
        "<div class='input-wrap'>",
        f"<input type='text' name='q' value='{qh}' placeholder='Ara' autofocus/>",
        f"<select name='lang'>{options_html}</select>",
        f"<label><input type='checkbox' name='expand' value='1'{expand_checked}/> Anlamca geniÅŸlet</label>",
        "</div>",
        "<div class='actions'>",
        "<button class='btn' type='submit'>Ara</button>",
        "</div>",
        "</form>",
        f"<div class='navlinks'><a href='/?q={qh_attr}{lang_link_suffix}{('&amp;expand=1' if expand else '')}{('&amp;theme=t2' if theme else '')}'>TÃ¼mÃ¼</a><a href='{image_href}'>GÃ¶rseller</a><a href='{ai_href}'>AI</a>{menu_detail}</div>",
        "</div>",
        "<div class='container'>",
        "<h2>Son Ä°ndekslenen Siteler</h2>",
    ]
    cur = conn.execute(
        "SELECT url, IFNULL(last_crawled,''), IFNULL(title,'') FROM documents WHERE last_crawled IS NOT NULL ORDER BY last_crawled DESC LIMIT 400"
    )
    rows = cur.fetchall()
    seen: Dict[str, Tuple[str, str, str]] = {}
    counts: Dict[str, int] = {}
    for url, last_ts, title in rows:
        try:
            p = urlparse(url)
            host = (p.netloc or "").lower()
            base = f"{p.scheme}://{p.netloc}" if p.scheme and p.netloc else url
        except Exception:
            host = ""
            base = url
        if not host:
            continue
        if host not in seen:
            seen[host] = (host, last_ts or "", base)
            counts[host] = 1
        else:
            counts[host] = counts.get(host, 1) + 1
    items: List[Tuple[str, str, str, int]] = []  # (host,last,base,count)
    for host, (h, last, base) in seen.items():
        items.append((h, last, base, counts.get(host, 1)))
    # last timestamps are ISO; seen preserves descending order from query
    if items:
        html_parts.append("<table style='width:100%; border-collapse:collapse'>")
        html_parts.append("<tr><th style='text-align:left;padding:6px 8px'>Site</th><th style='text-align:left;padding:6px 8px'>Son Tarama</th><th style='text-align:left;padding:6px 8px'>Belge</th></tr>")
        take = min(80, len(items))
        for host, last, base, cnt in items[:take]:
            host_html = html.escape(host)
            last_html = html.escape(last or "")
            base_html = html.escape(base)
            html_parts.append(
                f"<tr><td style='border-bottom:1px solid #eee;padding:6px 8px'><a href='{base_html}' target='_blank' rel='noreferrer'>{host_html}</a></td>"
                f"<td style='border-bottom:1px solid #eee;padding:6px 8px'>{last_html}</td>"
                f"<td style='border-bottom:1px solid #eee;padding:6px 8px'>{cnt}</td></tr>"
            )
        html_parts.append("</table>")
    else:
        html_parts.append("<p class='muted'>HenÃ¼z veri yok.</p>")
    html_parts.append("</div>")
    html_parts.append(HTML_FOOT)
    return "".join(html_parts).encode("utf-8")


def _page_shell(inner_html: str) -> bytes:
    return (HTML_HEAD + inner_html + HTML_FOOT).encode("utf-8")


def _render_about() -> bytes:
    inner = (
        "<div class='home' style='align-items:stretch'>"
        + "<div class='topbar'>" + _logo_html("logo") + "</div>"
        + "<div class='container'>"
        + "<h2>HakkÄ±nda</h2><p>yox, yerel ve basit bir arama arayÃ¼zÃ¼dÃ¼r. Veriler cihazÄ±nÄ±zdaki SQLite veritabanÄ±nda saklanÄ±r.</p>"
        + "</div></div>"
    )
    return _page_shell(inner)


def _render_help() -> bytes:
    inner = (
        "<div class='home' style='align-items:stretch'>"
        + "<div class='topbar'>" + _logo_html("logo") + "</div>"
        + "<div class='container'>"
        + "<h2>YardÄ±m</h2>"
        + "<p>Aramak iÃ§in kutuya yazÄ±p Enter'a basÄ±n. YazÄ±m hatalarÄ±nda otomatik dÃ¼zeltme uygulanÄ±r (Ã¶r. pyhton â†’ python).</p>"
        + "</div></div>"
    )
    return _page_shell(inner)


def _render_status(conn) -> bytes:
    docs = dbmod.doc_count(conn)
    terms = dbmod.term_count(conn)
    avg = dbmod.get_avg_doc_len(conn)
    inner = (
        "<div class='home' style='align-items:stretch'>"
        + "<div class='topbar'>" + _logo_html("logo") + "</div>"
        + "<div class='container'>"
        + f"<h2>Durum</h2><p>Belgeler: {docs} &nbsp; | &nbsp; Terimler: {terms} &nbsp; | &nbsp; Ortalama uzunluk: {avg:.1f}</p>"
        + "</div></div>"
    )
    return _page_shell(inner)


def app_factory(db_path: str):
    conn = dbmod.open_db(db_path)
    dbmod.ensure_schema(conn)

    # Small LRU cache for repeated queries
    class _LRU:
        def __init__(self, cap=128):
            self.cap = cap
            self.d = OrderedDict()
        def get(self, k):
            if k in self.d:
                v = self.d.pop(k)
                self.d[k] = v
                return v
            return None
        def put(self, k, v):
            if k in self.d:
                self.d.pop(k)
            elif len(self.d) >= self.cap:
                self.d.popitem(last=False)
            self.d[k] = v

    qcache = _LRU(256)

    def _ip_allowed(environ) -> bool:
        ip = environ.get("REMOTE_ADDR") or ""
        allow = os.environ.get("YOX_ADMIN_ALLOW", "127.0.0.1,::1")
        allowed = [s.strip() for s in allow.split(",") if s.strip()]
        if not allowed:
            return True
        return any(ip == a or ip.endswith(a) for a in allowed)

    def _admin_secret() -> str:
        return os.environ.get("YOX_ADMIN_SECRET", "")

    def _admin_password() -> str:
        return os.environ.get("YOX_ADMIN_PASSWORD", "")

    def _sign(data: str) -> str:
        sec = _admin_secret().encode("utf-8")
        return hmac.new(sec, data.encode("utf-8"), hashlib.sha256).hexdigest()

    def _issue_session(user: str = "admin", ttl: int = 3600) -> str:
        exp = int(time.time()) + int(ttl)
        payload = f"{user}|{exp}"
        sig = _sign(payload)
        raw = f"{payload}|{sig}".encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii")

    def _verify_session(token: str) -> bool:
        try:
            raw = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
            user, exp, sig = raw.split("|", 2)
            if not hmac.compare_digest(sig, _sign(f"{user}|{int(exp)}")):
                return False
            if int(exp) < int(time.time()):
                return False
            return True
        except Exception:
            return False

    def _admin_authed(environ) -> bool:
        if not _admin_secret():
            return False
        # Cookie auth
        cookie = environ.get("HTTP_COOKIE") or ""
        token = ""
        for part in cookie.split(";"):
            p = part.strip()
            if p.startswith("yox_admin="):
                token = p.split("=", 1)[1]
                break
        if token and _verify_session(token):
            return True
        # Bearer token (optional)
        auth = environ.get("HTTP_AUTHORIZATION") or ""
        if auth.startswith("Bearer "):
            bearer = auth.split(" ", 1)[1]
            if hmac.compare_digest(bearer, _admin_secret()):
                return True
        return False

    def _set_cookie(headers: list[tuple[str, str]], name: str, value: str, max_age: int = 3600):
        cookie = f"{name}={value}; Path=/; HttpOnly; SameSite=Strict; Max-Age={int(max_age)}"
        if os.environ.get("YOX_ADMIN_SECURE", "0") in ("1", "true", "True"):
            cookie += "; Secure"
        headers.append(("Set-Cookie", cookie))

    def _clear_cookie(headers: list[tuple[str, str]], name: str):
        headers.append(("Set-Cookie", f"{name}=; Path=/; HttpOnly; Max-Age=0; SameSite=Strict"))

    def _render_admin_login(msg: str = "") -> bytes:
        warn = (
            "<p class='muted'>YÃ¶netim paneli iÃ§in parola gerekli. "
            "Sadece yerel IP'lerden eriÅŸime izin verilir (YOX_ADMIN_ALLOW ile deÄŸiÅŸtirilebilir).</p>"
        )
        err = f"<p class='muted' style='color:#d93025'>{html.escape(msg)}</p>" if msg else ""
        body = [
            HTML_HEAD,
            "<div class='container'>",
            _logo_html("logo"),
            "<h1>YÃ¶netim GiriÅŸi</h1>",
            warn,
            err,
            "<form method='post' action='/admin/login' class='search-form' style='max-width:420px'>",
            "<div class='search-row'><input type='password' name='p' placeholder='Parola' autofocus/></div>",
            "<div class='actions'><button class='btn' type='submit'>GiriÅŸ</button></div>",
            "</form>",
            "</div>",
            HTML_FOOT,
        ]
        return "".join(body).encode("utf-8")

    def _parse_post(environ) -> Dict[str, str]:
        try:
            ln = int(environ.get("CONTENT_LENGTH") or "0")
        except Exception:
            ln = 0
        data = (environ.get("wsgi.input").read(ln) if ln > 0 else b"")
        try:
            from urllib.parse import parse_qs as _pqs
            d = {k: (v[0] if v else "") for k, v in _pqs(data.decode("utf-8", "ignore")).items()}
            return d
        except Exception:
            return {}

    def _render_admin(conn) -> bytes:
        # Stats
        cur = conn.execute("SELECT COUNT(*) FROM documents")
        doc_cnt = int(cur.fetchone()[0])
        cur = conn.execute("SELECT COUNT(*) FROM terms")
        term_cnt = int(cur.fetchone()[0])
        cur = conn.execute("SELECT COUNT(*) FROM images")
        img_cnt = int(cur.fetchone()[0])
        cur = conn.execute("SELECT COUNT(*) FROM discovered_hosts")
        host_cnt = int(cur.fetchone()[0])
        # Recent docs
        cur = conn.execute(
            "SELECT url, IFNULL(title,''), IFNULL(last_crawled,''), IFNULL(language,''), length"
            " FROM documents ORDER BY COALESCE(last_crawled, url) DESC LIMIT 30"
        )
        recents = cur.fetchall()
        # Host stats
        cur = conn.execute("SELECT status, COUNT(*) FROM discovered_hosts GROUP BY status")
        host_stats = cur.fetchall()
        # Due soon
        cur = conn.execute(
            "SELECT url, IFNULL(next_crawl_at,'') FROM documents WHERE next_crawl_at IS NOT NULL"
            " ORDER BY next_crawl_at LIMIT 20"
        )
        due_rows = cur.fetchall()

        def _rows(table_rows: list[tuple]) -> str:
            parts = ["<table style='width:100%; border-collapse:collapse'>"]
            for row in table_rows:
                cols = [f"<td style='border-bottom:1px solid #eee; padding:6px 8px'>{html.escape(str(c))}</td>" for c in row]
                parts.append("<tr>" + "".join(cols) + "</tr>")
            parts.append("</table>")
            return "".join(parts)

        body = [
            HTML_HEAD,
            "<div class='container'>",
            _logo_html("logo"),
            "<h1>YÃ¶netim Paneli</h1>",
            "<p class='muted'>Genel istatistikler</p>",
            f"<p>DokÃ¼manlar: <b>{doc_cnt}</b> | Terimler: <b>{term_cnt}</b> | GÃ¶rseller: <b>{img_cnt}</b> | Hostlar: <b>{host_cnt}</b></p>",
            "<h2>Son Ä°ndekslenenler</h2>",
            _rows(recents),
            "<h2>Host DurumlarÄ±</h2>",
            _rows(host_stats),
            "<h2>YakÄ±nda Yeniden Tarama</h2>",
            _rows(due_rows),
            "<p><a class='btn' href='/admin/logout'>Ã‡Ä±kÄ±ÅŸ</a></p>",
            "</div>",
            HTML_FOOT,
        ]
        return "".join(body).encode("utf-8")

    def app(environ, start_response):
        try:
            if environ.get("REQUEST_METHOD") != "GET":
                # Allow POST only for admin login
                if environ.get("PATH_INFO") == "/admin/login" and environ.get("REQUEST_METHOD") == "POST":
                    if not _admin_secret() or not _ip_allowed(environ):
                        start_response("403 Forbidden", [("Content-Type", "text/plain; charset=utf-8")])
                        return [b"admin disabled or ip not allowed"]
                    form = _parse_post(environ)
                    pw = (form.get("p") or "").strip()
                    if not pw or not hmac.compare_digest(pw, _admin_password()):
                        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                        return [_render_admin_login("Hatali parola")] 
                    token = _issue_session("admin", 3600)
                    headers = [("Content-Type", "text/html; charset=utf-8"), ("Location", "/admin")]
                    _set_cookie(headers, "yox_admin", token, 3600)
                    start_response("302 Found", headers)
                    return [b""]
                start_response("405 Method Not Allowed", [("Content-Type", "text/plain; charset=utf-8")])
                return [b"Only GET is supported"]
            path = environ.get("PATH_INFO", "/") or "/"
            qs = parse_qs(environ.get("QUERY_STRING", ""))
            q = (qs.get("q", [""]) or [""])[0]
            lang = (qs.get("lang", [""]) or [""])[0].strip().lower()[:5]
            lucky = (qs.get("lucky", ["0"]) or ["0"])[0] in ("1", "true", "True")
            expand = (qs.get("expand", ["0"]) or ["0"])[0] in ("1", "true", "True")
            theme_raw = (qs.get("theme", [""]) or [""])[0].strip().lower()
            theme_cls = "t2" if theme_raw in ("2", "t2", "glass", "ios") else ""

            # Route handling
            if path.startswith("/api/"):
                status = "200 OK"
                payload: Dict[str, object]
                if path == "/api/models":
                    payload = {"models": aimod.list_models(conn)}
                elif path == "/api/search":
                    if not q.strip():
                        status = "400 Bad Request"
                        payload = {"error": "missing_query"}
                    else:
                        items, corrections, _used = search_with_fuzzy(
                            conn,
                            q,
                            top_k=min(20, int((qs.get("top_k", ["10"]) or ["10"])[0] or "10")),
                            fuzzy=True,
                            preferred_lang=lang or None,
                        )
                        results = []
                        for doc_id, score in items:
                            _, url, title, content, length, doc_lang = dbmod.get_doc(conn, doc_id)
                            results.append({
                                "doc_id": doc_id,
                                "url": url,
                                "title": title or url,
                                "score": score,
                                "language": doc_lang or "",
                                "length": length,
                                "snippet": snippets(content, tokenize(q, remove_stopwords=False)),
                            })
                        payload = {"query": q, "corrections": corrections, "results": results}
                elif path == "/api/sources":
                    if not q.strip():
                        status = "400 Bad Request"
                        payload = {"error": "missing_query"}
                    else:
                        payload = {"query": q, "sources": aimod.source_hits(conn, q, preferred_lang=lang or None)}
                elif path == "/api/answer":
                    if not q.strip():
                        status = "400 Bad Request"
                        payload = {"error": "missing_query"}
                    else:
                        payload = {"query": q, **aimod.grounded_answer(conn, q, preferred_lang=lang or None)}
                else:
                    status = "404 Not Found"
                    payload = {"error": "not_found"}
                start_response(status, [("Content-Type", "application/json; charset=utf-8")])
                return [json.dumps(payload, ensure_ascii=False).encode("utf-8")]
            if path.startswith("/admin"):
                if not _admin_secret() or not _ip_allowed(environ):
                    start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
                    return [b"not found"]
                if path == "/admin/login":
                    start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                    return [_render_admin_login()]
                if path == "/admin/logout":
                    headers = [("Content-Type", "text/html; charset=utf-8"), ("Location", "/")] 
                    _clear_cookie(headers, "yox_admin")
                    start_response("302 Found", headers)
                    return [b""]
                if not _admin_authed(environ):
                    headers = [("Content-Type", "text/html; charset=utf-8"), ("Location", "/admin/login")] 
                    start_response("302 Found", headers)
                    return [b""]
                start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                return [_render_admin(conn)]
            if path == "/about":
                start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                return [_render_about()]
            if path == "/help":
                start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                return [_render_help()]
            if path == "/status":
                start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                return [_render_status(conn)]
            if path == "/ai":
                if not q.strip():
                    start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                    return [_render_home(q, lang, expand, theme_cls)]
                cache_key = ("ai", q, lang, expand, theme_cls)
                cached = qcache.get(cache_key)
                if cached is None:
                    body = _render_ai_results(conn, q, lang, expand, theme_cls)
                    qcache.put(cache_key, body)
                    start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                    return [body]
                start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                return [cached]
            if path == "/images":
                if not q.strip():
                    start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                    return [_render_home(q, lang, expand, theme_cls)]
                cache_key = ("images", q, lang, expand, theme_cls)
                cached = qcache.get(cache_key)
                if cached is None:
                    body = _render_image_results(conn, q, lang, expand, theme_cls)
                    qcache.put(cache_key, body)
                    start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                    return [body]
                start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                return [cached]

            if not q.strip():
                start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                return [_render_home(q, lang, expand, theme_cls)]

            # Special command: show recently indexed sites when query is '.so'
            if q.strip().lower() == ".so":
                cache_key = ("recent_sites", lang, expand, theme_cls)
                cached = qcache.get(cache_key)
                if cached is None:
                    body = _render_recent_sites(conn, q, lang, expand, theme_cls)
                    qcache.put(cache_key, body)
                    start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                    return [body]
                start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                return [cached]

            if lucky:
                items, _, _ = search_with_fuzzy(
                    conn,
                    q,
                    top_k=1,
                    fuzzy=True,
                    preferred_lang=lang or None,
                )
                if items:
                    doc_id, _ = items[0]
                    _, url, _, _, _, _ = dbmod.get_doc(conn, doc_id)
                    start_response("302 Found", [("Location", url)])
                    return [b""]

            cache_key = ("search", q, lang, expand, theme_cls)
            cached = qcache.get(cache_key)
            if cached is None:
                body = _render_results(conn, q, lang, expand, theme_cls)
                qcache.put(cache_key, body)
                start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                return [body]
            start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
            return [cached]
        except Exception as e:
            start_response("500 Internal Server Error", [("Content-Type", "text/plain; charset=utf-8")])
            return [str(e).encode("utf-8")]

    return app


def serve(db_path: str, host: str = "127.0.0.1", port: int = 8000):
    httpd = make_server(host, port, app_factory(db_path))
    print(f"Serving on http://{host}:{port} ...")
    httpd.serve_forever()


