import os
import re
import time
import random
import requests
import sys
from typing import Optional, Any
from urllib.parse import urlparse, unquote

from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


# ----- LICENSE/HOST FILTERS (DDG ONLY) -----
CC_PATTERNS = [
    r'creativecommons\.org/licenses/[^"\s<]+',
    r'creativecommons\.org/publicdomain/[^"\s<]+',
    r'\bCC\s?BY(?:-SA|-NC|-ND)?\s?(?:\d\.\d)?\b',
    r'\bCC0\b',
    r'\bCreative\s+Commons\b.*?(?:Attribution|ShareAlike|NonCommercial|NoDerivatives|Zero|Public\s+Domain)',
    r'\bPublic\s+Domain\b',
]
_CC_RX_FALLBACK = [re.compile(p, re.I) for p in CC_PATTERNS]

_BLOCKED_HOSTS_FALLBACK = {
    "twitter.com", "x.com", "facebook.com", "instagram.com", "tiktok.com",
    "pinterest.com", "pinimg.com", "reddit.com", "imgur.com", "tumblr.com"
}

_UAS_FALLBACK = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; en-US) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Edge/127.0",
]


def _resolve_main_deps():
    """
    Pull required globals from the running main module (your imageresearcher.py),
    without importing it (avoids circular imports).
    """
    main = sys.modules.get("__main__")

    IMAGES_PATH = getattr(main, "IMAGES_PATH", os.path.join(os.getcwd(), "ResearchImages"))
    dbg = getattr(main, "dbg", lambda *a, **k: None)
    register_meta = getattr(main, "_register_image_metadata", lambda path, meta: None)
    normalize_ctx_meta = getattr(main, "_normalize_ctx_meta", lambda d: d or {})

    download_to = getattr(main, "_download_to", None)

    _UAS = getattr(main, "_UAS", _UAS_FALLBACK)
    _BLOCKED_HOSTS = getattr(main, "_BLOCKED_HOSTS", _BLOCKED_HOSTS_FALLBACK)
    _CC_RX = getattr(main, "_CC_RX", _CC_RX_FALLBACK)

    return {
        "IMAGES_PATH": IMAGES_PATH,
        "dbg": dbg,
        "register_meta": register_meta,
        "normalize_ctx_meta": normalize_ctx_meta,
        "download_to": download_to,
        "_UAS": _UAS,
        "_BLOCKED_HOSTS": _BLOCKED_HOSTS,
        "_CC_RX": _CC_RX,
    }


def _is_blocked_host(u: str, blocked_hosts: set[str]) -> bool:
    try:
        host = (urlparse(u).hostname or "").lower()
        return any(host.endswith(b) for b in blocked_hosts)
    except Exception:
        return True


def _cc_evidence_in_html(html: str, cc_rx: list[re.Pattern]) -> bool:
    if not html:
        return False

    for rx in cc_rx:
        if rx.search(html):
            return True

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return False

    for ln in soup.find_all("link", rel=True, href=True):
        relv = " ".join(ln.get("rel") if isinstance(ln.get("rel"), list) else [ln.get("rel")]).lower()
        if "license" in relv and "creativecommons.org" in (ln.get("href") or "").lower():
            return True

    for a in soup.find_all("a", href=True):
        if "creativecommons.org" in (a.get("href") or "").lower():
            return True

    for m in soup.find_all("meta"):
        for attr in ("content", "value", "href", "property", "name"):
            v = (m.get(attr) or "")
            if isinstance(v, str) and "creativecommons.org" in v.lower():
                return True

    return False


def _download_to_fallback(u: str, dest_dir: str, idx: int, dbg=print) -> str | None:
    """
    Fallback downloader if your main script doesn't expose _download_to for some reason.
    Tries to behave the same as your original _download_to.
    """
    try:
        os.makedirs(dest_dir, exist_ok=True)
        parsed = urlparse(u)
        fname = unquote(os.path.basename(parsed.path)) or f"file_{idx}"
        fname = re.sub(r"[?#].*$", "", fname)
        if not os.path.splitext(fname)[1]:
            fname += ".img"
        fname = re.sub(r"[^A-Za-z0-9._\- ]+", "_", fname)[:100]
        path = os.path.join(dest_dir, fname)

        if os.path.exists(path):
            dbg(f"[IMG][skip] already exists {path}")
            return path

        headers = {"User-Agent": "diag-scrape/0.1"}
        with requests.get(u, headers=headers, stream=True, timeout=30) as r:
            r.raise_for_status()
            ctype = (r.headers.get("Content-Type") or "").lower()

            if not (ctype.startswith("image/") or re.search(r"\.(png|jpe?g|gif|webp|svg|bmp|tiff?)($|\?)", u, re.I)):
                dbg(f"[IMG][skip] not image ct={ctype} url={u}")
                return None

            cl = r.headers.get("Content-Length")
            if cl and cl.isdigit() and int(cl) < 2048:
                dbg(f"[IMG][skip] too small ({cl} B) {u}")
                return None

            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)

        dbg(f"[IMG][ok] saved {path}")
        return path

    except Exception as e:
        dbg(f"[IMG][err] {u} -> {e}")
        return None


def ddg_cc_image_harvest(
    query: str,
    target_count: int = 10,
    ddg_cap: int = 60,               # soft cap on how many DDG items we *scan*
    sleep_between: float = 1.0,      # seconds between item fetches
    backoff_base: float = 4.0,       # starting backoff on 403
    backoff_max: float = 45.0,       # max backoff cap
    safesearch: str = "moderate",
    region: str = "wt-wt",
    proxies: dict | None = None,
    lemma_obj: Optional[dict] = None,
    encoder: Optional[Any] = None,
    query_embedding: Optional[Any] = None,
    base_ctx_embedding: Optional[Any] = None
) -> list[str]:
    """
    Rate-limit-aware DDG image flow:
      - streams results gently
      - retries on 403 with backoff + jitter
      - downloads ONLY if the result page shows CC evidence
      - attaches per-image context from ddg_image_context (filename/alt/title + nearby text)
      - optionally attaches context embedding/semantic score
      - stops early if 20 consecutive pages have no CC evidence
    """
    deps = _resolve_main_deps()
    IMAGES_PATH = deps["IMAGES_PATH"]
    dbg = deps["dbg"]
    register_meta = deps["register_meta"]
    normalize_ctx_meta = deps["normalize_ctx_meta"]
    download_to = deps["download_to"]
    _UAS = deps["_UAS"]
    _BLOCKED_HOSTS = deps["_BLOCKED_HOSTS"]
    _CC_RX = deps["_CC_RX"]

    dest_dir = os.path.join(IMAGES_PATH, "ddg")
    os.makedirs(dest_dir, exist_ok=True)

    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(_UAS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    if proxies:
        s.proxies.update(proxies)

    saved: list[str] = []
    seen_pages: set[str] = set()
    seen_imgs: set[str] = set()

    no_cc_streak = 0
    NO_CC_STREAK_LIMIT = 20

    def _yield_ddg_images(q: str, cap: int):
        total = 0
        delay = backoff_base
        while total < cap:
            try:
                headers = {
                    "User-Agent": random.choice(_UAS),
                    "Accept-Language": "en-US,en;q=0.9",
                }
                with DDGS(headers=headers, proxies=proxies) as ddgs:
                    for item in ddgs.images(q, max_results=cap - total, safesearch=safesearch, region=region):
                        yield item
                        total += 1
                        if total >= cap:
                            break
                        time.sleep(sleep_between)
                break
            except Exception as e:
                msg = str(e).lower()
                if "403" in msg or "ratelimit" in msg or "too many requests" in msg:
                    jitter = random.uniform(0.4, 0.9)
                    wait = min(delay * (1.6 + jitter), backoff_max)
                    dbg(f"[DDG] rate-limited; backing off for {wait:.1f}s")
                    time.sleep(wait)
                    delay = min(delay * 2.0, backoff_max)
                    continue
                dbg(f"[DDG] unexpected error: {e}")
                break

    last_saved_path = None

    for r in _yield_ddg_images(query, ddg_cap):
        if len(saved) >= target_count:
            break

        if no_cc_streak >= NO_CC_STREAK_LIMIT:
            dbg(f"[DDG] stopping: {no_cc_streak} consecutive pages without CC evidence")
            break

        page_url = r.get("url") or r.get("source") or ""
        img_url = r.get("image") or ""
        if not page_url or not img_url:
            continue

        if _is_blocked_host(page_url, _BLOCKED_HOSTS) or _is_blocked_host(img_url, _BLOCKED_HOSTS):
            continue

        if page_url in seen_pages or img_url in seen_imgs:
            continue

        seen_pages.add(page_url)
        seen_imgs.add(img_url)

        # fetch the *page* and check for CC evidence
        try:
            pr = s.get(page_url, timeout=25)
            ct = (pr.headers.get("Content-Type") or "").lower()
            if "html" not in ct and "xml" not in ct:
                dbg(f"[DDG] non-HTML page ct={ct} -> skip  {page_url}")
                no_cc_streak += 1
                continue
            html = pr.text
        except Exception:
            no_cc_streak += 1
            continue

        if not _cc_evidence_in_html(html, _CC_RX):
            dbg(f"[DDG] no CC evidence -> skip  {page_url}")
            no_cc_streak += 1
            continue

        no_cc_streak = 0

        # --- per-image context (filename/alt/title + nearby text) ---
        ctx_meta: dict = {}
        try:
            soup = BeautifulSoup(html, "html.parser")
            try:
                from ddg_image_context import ddg_extract_image_context
                ctx_meta = ddg_extract_image_context(
                    soup,
                    page_url=page_url,
                    image_url=img_url,
                    base_query=query,
                    lemma_obj=lemma_obj or {},
                    encoder=encoder,
                    query_embedding=query_embedding,
                    return_embedding=True,
                ) or {}
                ctx_meta = normalize_ctx_meta(ctx_meta)
            except ImportError as e:
                dbg(f"[DDG] ddg_image_context import error: {e}")
            except Exception as e:
                dbg(f"[DDG] ddg_extract_image_context error: {e}")
        except Exception as e:
            dbg(f"[DDG] context soup error: {e}")

        # ok to download the image
        if download_to is None:
            p = _download_to_fallback(img_url, dest_dir, len(saved), dbg=dbg)
        else:
            p = download_to(img_url, dest_dir, len(saved), dbg=dbg)

        if p:
            meta = {
                "source_kind": "ddg",
                "page_url": page_url,
                "image_url": img_url,
                "base_context": query,
            }

            if base_ctx_embedding is not None:
                meta["prompt_embedding"] = base_ctx_embedding

            if ctx_meta:
                meta.update(ctx_meta)

            if "ctx_score" in meta or "ctx_sem_score" in meta:
                meta["ctx_confidence"] = meta.get("ctx_sem_score", meta.get("ctx_score"))
                dbg(
                    f"[DDG][CTX] saved {p} ctx_score={meta.get('ctx_score')} "
                    f"ctx_sem={meta.get('ctx_sem_score')} conf={meta.get('ctx_confidence')}"
                )
            else:
                dbg(f"[DDG][CTX] saved {p} ctx=n/a")

            register_meta(p, meta)
            saved.append(p)
            last_saved_path = p

    dbg(f"[DDG] saved ({len(saved)}/{target_count}) {last_saved_path}")
    return saved
