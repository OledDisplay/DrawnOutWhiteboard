import os, json, copy, requests, re, hashlib, datetime, time, threading, random, shutil
from urllib.parse import urlparse, unquote, urljoin, urlunparse
from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup ,NavigableString, Tag
import tldextract
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path
from PIL import Image

from duckduckgo_search import DDGS  # (left intact for API section)

from smart_hits import smart_find_hits_in_soup

from parsers import parse_wikimedia, parse_openverse, parse_plos, parse_usgs

try:
    from sentence_transformers import SentenceTransformer
    _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as _e:
    _EMBED_MODEL = None
    print(f"[EMBED] sentence-transformers not available or model load failed: {_e}", flush=True)

from ImageRanker import SiglipBackend, rerank_image_candidates_siglip


ROOT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = os.path.join(ROOT_DIR, "source_urls")
IMAGES_PATH = os.path.join(ROOT_DIR, "ResearchImages")
os.makedirs(SOURCE_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)

# global image metadata: path -> list of metadata dicts
IMAGE_METADATA: Dict[str, List[dict]] = {}



def _register_image_metadata(path: str, meta: dict) -> None:
    if not path:
        return
    lst = IMAGE_METADATA.get(path)
    if lst is None:
        IMAGE_METADATA[path] = [meta]
    else:
        lst.append(meta)


DEBUG = True
def dbg(*args):
    if DEBUG:
        print(*args, flush=True)

# ----- LICENSE/HOST FILTERS (left intact for API sections and potential DDG) -----
CC_PATTERNS = [
    r'creativecommons\.org/licenses/[^"\s<]+',
    r'creativecommons\.org/publicdomain/[^"\s<]+',
    r'\bCC\s?BY(?:-SA|-NC|-ND)?\s?(?:\d\.\d)?\b',
    r'\bCC0\b',
    r'\bCreative\s+Commons\b.*?(?:Attribution|ShareAlike|NonCommercial|NoDerivatives|Zero|Public\s+Domain)',
    r'\bPublic\s+Domain\b',
]
_CC_RX = [re.compile(p, re.I) for p in CC_PATTERNS]

_BLOCKED_HOSTS = {
    "twitter.com", "x.com", "facebook.com", "instagram.com", "tiktok.com",
    "pinterest.com", "pinimg.com", "reddit.com", "imgur.com", "tumblr.com"
}

_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; en-US) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Edge/127.0",
]


IMG_EXT_RE = re.compile(r"\.(png|jpe?g|gif|webp|bmp|tiff?|svg)$", re.I)


UA = {"User-Agent": "diag-scrape/0.2 (+research/cc-check; contact: your-email@example.com)"}

# =========================
# URL BLOCKS / NORMALIZATION
# =========================
BLOCKED_URL_WORDS = [
    "blog", "press", "partners", "privacy", "license", "tos",
    "accessibility", "accounts/login", "facebook", "twitter",
    "instagram", "youtube", "linkedin", "help.openstax.org",
    "rice.edu", "gatesfoundation.org", "google.com", "business.safety.google",
    "mailto:", "tel:", "pdf", "drive.google.com",
]

NON_HTML_EXT = (".pdf", ".doc", ".docx", ".zip", ".rar", ".7z",
                ".ppt", ".pptx", ".xls", ".xlsx",
                ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
                ".mp3", ".wav", ".mp4", ".webm", ".avi")

SCHEME_BLOCK = ("mailto:", "javascript:", "data:")

# Skip obvious UI assets (kept small & safe)
UI_IMG_SKIP_RE = re.compile(
    r"(?:sprite|favicon|logo|text[-_]?size|icon|loader|brand|wordmark)\.(?:svg|png|webp|gif|jpg|jpeg)$",
    re.IGNORECASE,
)
DATA_URL_RE = re.compile(r"^\s*data:", re.IGNORECASE)

EXCLUDE_REGION_CLASS_WORDS = (
    "header", "footer", "navbar", "breadcrumb", "toolbar", "sidebar", "aside",
    "toc", "table-of-contents", "menu", "pager", "nav", "masthead", "brand", "banner"
)

def _is_excluded_region(tag: Tag) -> bool:
    """
    Return True if a tag is likely part of a page shell (header, footer, nav, etc.)
    rather than main content.
    Used to prevent false hits and UI icons from being scraped.
    """
    try:
        # Basic tag name filters
        name = (tag.name or "").lower()
        if name in ("header", "footer", "nav", "aside"):
            return True

        # Class-based filters
        cls_list = tag.get("class") or []
        for c in cls_list:
            c_low = (c or "").lower()
            if any(word in c_low for word in EXCLUDE_REGION_CLASS_WORDS):
                return True

        # ID-based filters
        id_val = (tag.get("id") or "").lower()
        if any(word in id_val for word in EXCLUDE_REGION_CLASS_WORDS):
            return True

        # Role-based (some sites mark <div role="banner"> etc.)
        role = (tag.get("role") or "").lower()
        if role in ("banner", "navigation", "complementary", "contentinfo"):
            return True

        return False

    except Exception:
        re


def clean_url(u: str) -> str:
    if not u:
        return ""
    u = unquote(u)
    if any(u.lower().startswith(s) for s in SCHEME_BLOCK):
        return ""
    u = re.sub(r"#.*$", "", u)
    u = re.sub(r"[?&](utm_[^=&]+|fbclid|gclid)=[^&]*", "", u)
    p = urlparse(u)
    host = (p.hostname or "").lower()
    path = p.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return urlunparse((p.scheme or "https", host, path, "", p.query, ""))

def same_registrable(host_a: str, host_b: str) -> bool:
    if not host_a or not host_b:
        return False
    t1 = tldextract.extract(host_a)
    t2 = tldextract.extract(host_b)
    r1 = ".".join([t1.domain, t1.suffix]) if t1.suffix else t1.domain
    r2 = ".".join([t2.domain, t2.suffix]) if t2.suffix else t2.domain
    return r1 == r2

def domain_and_phrase_lock(root: str, candidate: str) -> bool:
    """Strict dynamic lock: same registrable domain + shared path token."""
    pr = urlparse(root); pc = urlparse(candidate)
    if not same_registrable((pr.hostname or "").lower(), (pc.hostname or "").lower()):
        return False
    rp = {s.strip().lower() for s in (pr.path or "/").split("/") if s.strip()}
    cp = {s.strip().lower() for s in (pc.path or "/").split("/") if s.strip()}
    return bool(rp and cp and (rp & cp))

def parent_part(u: str) -> str:
    """Return the 'parent' part: scheme://host/<path up to last '/'>/"""
    cu = clean_url(u)
    p = urlparse(cu)
    path = p.path or "/"
    if path == "/":
        return urlunparse((p.scheme or "https", (p.hostname or "").lower(), "/", "", "", ""))
    base = path.rsplit("/", 1)[0] or "/"
    if not base.endswith("/"):
        base = base + "/"
    return urlunparse((p.scheme or "https", (p.hostname or "").lower(), base, "", "", ""))

# =========================
# JS / Playwright helpers
# =========================

def expand_all_disclosures_sync(
    page,
    max_rounds: int = 8,
    per_round_click_cap: int = 80,
    quiet_ms: int = 350,
    stall_round_limit: int = 2,   # ← NEW: stop after N consecutive no-growth rounds
):
    """
    Generic, dynamic expansion of hidden TOCs/accordions/dropdowns.
    - Blocks 'document' navigations while expanding to avoid leaving the page.
    - Opens <details>, toggles [aria-expanded], [data-*] collapses, and common accordion/dropdown classes.
    - Repeats until growth stalls (no new anchors for `stall_round_limit` rounds) or max_rounds reached.
    """

    # --- block full-page navigations during expansion ---
    def _route_handler(route):
        req = route.request
        # Block only 'document' navigations to other pages; let subresources through.
        if req.resource_type == "document" and req.url != page.url:
            return route.abort()
        return route.continue_()

    page.route("**/*", _route_handler)
    try:
        def _anchors_count():
            try:
                return page.eval_on_selector_all("a[href]", "els => els.length")
            except Exception:
                return 0

        prev = _anchors_count()
        no_growth_streak = 0  # ← NEW: consecutive rounds with no anchor growth

        # expand <details> everywhere (very cheap + reliable)
        page.evaluate("""
(() => {
  document.querySelectorAll('details:not([open])').forEach(d => { try { d.open = true; } catch(e) {} });
})();
""")

        for round_i in range(1, max_rounds + 1):
            clicks = 0

            # 1) Expand elements with aria-expanded/aria-controls and common collapsed classes
            candidates = page.query_selector_all(
                ",".join([
                    # ARIA/semantic toggles
                    "[aria-controls]", "[aria-expanded='false']",
                    # Common accordion/dropdown controls
                    ".accordion-button", ".accordion-toggle", ".accordion__button",
                    ".dropdown-toggle", ".dropdown__toggle", ".collapsible", ".collapse-toggle",
                    # TOCs & treeviews
                    "[role='treeitem'] > [role='button']",
                    "[role='button']",
                    # Generic clickable headings
                    "summary", "button"
                ])
            )

            def _looks_collapsed_and_visible(el):
                try:
                    # visible?
                    box = el.bounding_box()
                    if not box or box["width"] < 2 or box["height"] < 2:
                        return False
                    # collapsed hints
                    ar = (el.get_attribute("aria-expanded") or "").strip().lower()
                    cls = (el.get_attribute("class") or "").lower()
                    name = (el.inner_text() or "").strip().lower()
                    collapsed_by_aria = (ar == "false")
                    collapsed_by_class = any(c in cls for c in ("collapsed", "is-collapsed", "closed"))
                    looks_like_toggle = any(t in name for t in ("expand", "show", "open", "more", "sections", "contents"))
                    has_controls = (el.get_attribute("aria-controls") is not None) or \
                                   (el.get_attribute("data-toggle") in ("collapse", "dropdown")) or \
                                   (el.get_attribute("data-target") is not None) or \
                                   (el.get_attribute("data-open") is not None) or \
                                   (el.evaluate("e => !!e.closest('.accordion, .dropdown, .toc, .toc-container, .toc__list')"))
                    return collapsed_by_aria or collapsed_by_class or (has_controls and looks_like_toggle)
                except Exception:
                    return False

            # Click a batch of candidates (no_wait_after to avoid waiting for nav)
            for el in candidates:
                if clicks >= per_round_click_cap:
                    break
                try:
                    if _looks_collapsed_and_visible(el):
                        el.click(force=True, no_wait_after=True)
                        clicks += 1
                except Exception:
                    continue

            # 2) Programmatically open any remaining <details> created dynamically
            page.evaluate("""
(() => {
  document.querySelectorAll('details:not([open])').forEach(d => { try { d.open = true; } catch(e) {} });
})();
""")

            # Wait a bit for DOM to settle / lazy chunks to attach anchors
            try:
                page.wait_for_load_state("networkidle", timeout=quiet_ms)
            except Exception:
                pass
            page.wait_for_timeout(quiet_ms)

            cur = _anchors_count()

            # round summary
            print(f"[EXPAND] round={round_i} clicks={clicks} anchors={cur} (prev={prev})")

            # growth check (NEW)
            if cur > prev:
                no_growth_streak = 0
            else:
                no_growth_streak += 1

            # early exits
            if clicks == 0 and cur <= prev:
                # If literally nothing changed this round, bail immediately
                print("[EXPAND] no clicks + no growth → stop")
                break

            if no_growth_streak >= stall_round_limit:
                print(f"[EXPAND] stalled for {no_growth_streak} rounds (limit={stall_round_limit}) → stop")
                break

            prev = cur

    finally:
        try:
            page.unroute("**/*", _route_handler)
        except Exception:
            pass


def _collect_basic_anchors_sync(page):
    return page.eval_on_selector_all("a[href]", "els => els.map(e => e.href).filter(Boolean)")

def _collect_data_links_sync(page):
    script = """
(() => {
  const urls = new Set();
  const pick = (v) => {
    if (!v || typeof v !== 'string') return;
    try { urls.add(new URL(v, document.baseURI).href); } catch(e) {}
  };

  // Expand all <details>…</details>
  document.querySelectorAll('details').forEach(d => { try { d.open = true; } catch(e){} });

  // Expand common dropdowns (aria-expanded toggles)
  document.querySelectorAll('[aria-expanded="false"]').forEach(el => {
    try { el.setAttribute('aria-expanded','true'); } catch(e){}
  });

  // data-* url hints
  document.querySelectorAll('[data-href],[data-url]').forEach(el => {
    pick(el.getAttribute('data-href'));
    pick(el.getAttribute('data-url'));
  });

  // onclick patterns
  document.querySelectorAll('[onclick]').forEach(el => {
    const t = el.getAttribute('onclick') || '';
    const m1 = t.match(/location\\.(?:href|assign|replace)\\s*=\\s*['"]([^'"]+)['"]/i);
    if (m1) pick(m1[1]);
    const m2 = t.match(/window\\.open\\(\\s*['"]([^'"]+)['"]/i);
    if (m2) pick(m2[1]);
  });

  return Array.from(urls);
})()
"""
    return page.evaluate(script)

def _collect_shadow_anchors_sync(page):
    script = """
(() => {
  const urls = new Set();
  const getAnchors = (root) =>
    Array.from(root.querySelectorAll("a[href]")).map(a => a.href).filter(Boolean);

  const seen = new Set();
  const stack = [document];

  while (stack.length) {
    const root = stack.pop();
    if (!root || seen.has(root)) continue;
    seen.add(root);
    getAnchors(root).forEach(u => urls.add(u));
    const all = root.querySelectorAll("*");
    for (const el of all) {
      if (el.shadowRoot) stack.push(el.shadowRoot);
    }
  }
  return Array.from(urls);
})()
"""
    return page.evaluate(script)

MODAL_TEXT_PATTERNS = (
    r"\b(read|view|open|launch|access|start|continue|student resources|instructor resources|resources|contents|table of contents|toc)\b"
)

def _wait_for_modal_sync(page, timeout_ms):
    selectors = [
        "[role='dialog']",
        "[aria-modal='true']",
        "dialog[open]",
        ".modal.show",
        ".modal[aria-hidden='false']",
        ".MuiModal-root", ".mantine-Modal-root", ".ant-modal",
    ]
    for sel in selectors:
        try:
            el = page.wait_for_selector(sel, state="visible", timeout=timeout_ms)
            if el:
                return el
        except Exception:
            pass
    return None

def _find_modal_triggers_sync(page):
    triggers = set()
    query = (
        "a[href='#'], a[role='button'], button, [data-toggle='modal'], "
        "[data-target], [aria-controls], [data-dialog-target], [data-modal], [data-open]"
    )
    candidates = page.query_selector_all(query)
    for el in candidates:
        try:
            txt = (el.inner_text() or "").strip().lower()
        except Exception:
            txt = ""
        try:
            tag = (el.evaluate("e => e.tagName") or "").lower()
        except Exception:
            tag = ""
        try:
            href = (el.get_attribute("href") or "").strip()
        except Exception:
            href = ""

        looks_like_trigger = (
            tag == "button"
            or href in ("", "#")
            or "modal" in (el.get_attribute("class") or "").lower()
            or "dialog" in (el.get_attribute("class") or "").lower()
            or re.search(MODAL_TEXT_PATTERNS, txt, re.I) is not None
            or any(
                attr in (el.get_attribute(attr) or "").lower()
                for attr in ("data-toggle", "data-target", "data-modal", "data-open", "aria-controls")
            )
        )
        if looks_like_trigger:
            triggers.add(el)

    try:
        btns = page.get_by_role("button", name=re.compile(MODAL_TEXT_PATTERNS, re.I)).all()
        for b in btns:
            triggers.add(b)
    except Exception:
        pass

    return list(triggers)

def _harvest_popup_urls_on_click_sync(page, el, popup_timeout_ms=1500):
    anchors = set()
    pop_urls = set()

    with page.expect_popup(timeout=popup_timeout_ms) as maybe_popup:
        try:
            el.click(force=True, no_wait_after=True)
        except Exception:
            pass
        try:
            popup = maybe_popup.value
            try:
                popup.wait_for_load_state("domcontentloaded", timeout=popup_timeout_ms)
            except Exception:
                pass
            try:
                pop_urls.add(popup.url)
            except Exception:
                pass
            try:
                a2 = popup.eval_on_selector_all("a[href]", "els => els.map(e => e.href).filter(Boolean)")
                for u in a2 or []:
                    pop_urls.add(u)
            except Exception:
                pass
            try:
                popup.close()
            except Exception:
                pass
        except Exception:
            pass

    modal = _wait_for_modal_sync(page, timeout_ms=1200)
    if modal:
        try:
            in_modal = page.eval_on_selector_all(
                "[role='dialog'], [aria-modal='true'], dialog[open], .modal.show, .modal[aria-hidden='false']",
                """els => {
                    const urls = new Set();
                    const add = (u) => { try{ urls.add(new URL(u, document.baseURI).href); } catch(e){} };
                    for (const root of els) {
                      root.querySelectorAll('a[href]').forEach(a => add(a.href));
                      root.querySelectorAll('[data-href],[data-url]').forEach(el => {
                        const dh = el.getAttribute('data-href');
                        const du = el.getAttribute('data-url');
                        if (dh) add(dh);
                        if (du) add(du);
                      });
                      root.querySelectorAll('[onclick]').forEach(el => {
                        const t = el.getAttribute('onclick') || '';
                        const m1 = t.match(/location\\.(?:href|assign|replace)\\s*=\\s*['"]([^'"]+)['"]/i);
                        if (m1) add(m1[1]);
                        const m2 = t.match(/window\\.open\\(\\s*['"]([^'"]+)['"]/i);
                        if (m2) add(m2[1]);
                      });
                    }
                    return Array.from(urls);
                }"""
            )
            for u in in_modal or []:
                anchors.add(u)
        except Exception:
            pass
        try:
            page.keyboard.press("Escape")
        except Exception:
            pass

    return anchors, pop_urls

SCHEME_BLOCK = ("mailto:", "javascript:", "data:")
NON_HTML_EXT = (".pdf", ".doc", ".docx", ".zip", ".rar", ".7z",
                ".ppt", ".pptx", ".xls", ".xlsx",
                ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
                ".mp3", ".wav", ".mp4", ".webm", ".avi")


def ordered_dedupe(iterable):
    seen = set()
    out = []
    for x in iterable:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def collect_links_with_modals_sync(page, expand=True, max_modal_clicks=6, **kwargs):
    """
    Returns a deduped list of links discovered on:
      - the main DOM (after expanding disclosures)   [only if expand=True]
      - shadow DOM                                    [only if expand=True]
      - data-* / onclick                              [only if expand=True]
      - links revealed after clicking modal triggers  [only if expand=True]
    If expand=False, we ONLY read the currently visible <a href> anchors (fast).
    Preserves DOM order as much as possible.
    """
    urls = []

    if expand:
        # fully expand accordions/TOC/dropdowns generically
        expand_all_disclosures_sync(page, max_rounds=3, per_round_click_cap=120, quiet_ms=350)

        # main anchors first (DOM order)
        for u in _collect_basic_anchors_sync(page):
            urls.append(u)
        # shadow DOM next
        for u in _collect_shadow_anchors_sync(page):
            urls.append(u)
        # data-/onclick-derived last
        for u in _collect_data_links_sync(page):
            urls.append(u)

        # modal/popup harvest (append in the order we encounter them)
        triggers = _find_modal_triggers_sync(page)[:max_modal_clicks]
        for el in triggers:
            try:
                modal_links, popup_links = _harvest_popup_urls_on_click_sync(page, el)
                for u in modal_links:
                    urls.append(u)
                for u in popup_links:
                    urls.append(u)
            except Exception:
                continue
    else:
        # FAST PATH: visible anchors only; no expansion, no modals, no shadow scan
        for u in _collect_basic_anchors_sync(page):
            urls.append(u)

    return ordered_dedupe([u for u in urls if u])





# =========================
# NEW: Playwright reuse (Stage 2)
# =========================
class _PlaywrightReuse:
    """
    One Playwright browser/context reused across many fetches.
    Used by Stage 2 only (unless you pass it elsewhere).
    """
    def __init__(self, headless: bool = True):
        try:
            from playwright.sync_api import sync_playwright
        except Exception:
            self._ok = False
            self._p = None
            self.browser = None
            self.ctx = None
            return

        self._ok = True
        self._p = sync_playwright().start()
        self.browser = self._p.chromium.launch(headless=headless)
        self.ctx = self.browser.new_context()

    def ok(self) -> bool:
        return bool(self._ok and self.browser and self.ctx)

    def new_page(self, timeout_ms: int):
        page = self.ctx.new_page()
        page.set_default_timeout(timeout_ms)
        return page

    def close(self):
        if not self._ok:
            return
        try:
            if self.ctx:
                self.ctx.close()
        except Exception:
            pass
        try:
            if self.browser:
                self.browser.close()
        except Exception:
            pass
        try:
            if self._p:
                self._p.stop()
        except Exception:
            pass


# =========================
# NEW: Terminal index helpers (json + embeddings)
# =========================
def _short_url(u: str, max_len: int = 90) -> str:
    try:
        p = urlparse(u)
        tail = (p.path or "/").rstrip("/").split("/")[-1] or "/"
        host = p.hostname or ""
        s = f"{host}/{tail}"
    except Exception:
        s = (u or "")
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."

def _last_path_part(u: str) -> str:
    try:
        p = urlparse(u)
        segs = [s for s in (p.path or "").split("/") if s]
        return segs[-1] if segs else ""
    except Exception:
        return ""

def _normalize_for_embed(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _url_key_for_embed(u: str) -> str:
    return _normalize_for_embed(_last_path_part(u))

def _norm_vec(v):
    import numpy as np
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    if n <= 0:
        return a
    return a / n

def _registrable_of(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
        t = tldextract.extract(host)
        return ".".join([t.domain, t.suffix]) if t.suffix else t.domain
    except Exception:
        return ""

def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def _save_json_atomic(path: str, data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        try:
            os.replace(tmp, path)
        except Exception:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        dbg(f"[IDX][SAVE][ERR] {path} -> {e}")

def _load_all_terminal_indexes_for_subject(subj: str) -> list[dict]:
    """
    Read ALL source json files and return all terminal indexes stored under subj.
    Each returned dict is enriched with:
      - _source_name
      - _json_path
      - _source_url (if present)
    """
    out = []
    try:
        for filename in os.listdir(SOURCE_PATH):
            if not filename.endswith(".json"):
                continue
            jpath = os.path.join(SOURCE_PATH, filename)
            data = _load_json(jpath)
            ti = (data.get("TerminalIndexes") or {})
            arr = ti.get(subj) or []
            if not isinstance(arr, list) or not arr:
                continue
            src_name = Path(filename).stem
            src_url = data.get("url")
            for idx in arr:
                if not isinstance(idx, dict):
                    continue
                d = dict(idx)
                d["_source_name"] = src_name
                d["_json_path"] = jpath
                d["_source_url"] = src_url
                out.append(d)
    except Exception as e:
        dbg(f"[IDX][LOAD_ALL][ERR] subj={subj} -> {e}")
    return out

def _persist_terminal_indexes_to_source(source_name: str, subj: str, new_indexes: list[dict]) -> None:
    """
    Save indexes into THIS source's json under TerminalIndexes[subj].
    If entry_url already exists, update if terminal_count improved or centroid exists.
    """
    if not new_indexes:
        return

    json_path = os.path.join(SOURCE_PATH, f"{source_name}.json")
    data = _load_json(json_path)
    data.setdefault("TerminalIndexes", {})
    data["TerminalIndexes"].setdefault(subj, [])

    existing = data["TerminalIndexes"][subj]
    if not isinstance(existing, list):
        existing = []
        data["TerminalIndexes"][subj] = existing

    # index by entry_url
    by_entry = {}
    for i, it in enumerate(existing):
        if isinstance(it, dict) and it.get("entry_url"):
            by_entry[clean_url(it["entry_url"])] = (i, it)

    added = 0
    updated = 0

    for idx in new_indexes:
        if not isinstance(idx, dict):
            continue
        entry = clean_url(idx.get("entry_url") or "")
        terms = idx.get("terminal_urls") or []
        cnt = int(idx.get("terminal_count") or len(terms) or 0)

        if not entry or cnt <= 0:
            continue

        if entry in by_entry:
            pos, cur = by_entry[entry]
            cur_cnt = int(cur.get("terminal_count") or 0)
            # update if better
            if cnt > cur_cnt or (idx.get("centroid_embedding") and not cur.get("centroid_embedding")):
                existing[pos] = idx
                updated += 1
        else:
            existing.append(idx)
            added += 1

    # keep it bounded (avoid unbounded growth)
    MAX_PER_SUBJ = 20
    if len(existing) > MAX_PER_SUBJ:
        # keep highest terminal_count first
        existing.sort(key=lambda d: int((d or {}).get("terminal_count") or 0), reverse=True)
        data["TerminalIndexes"][subj] = existing[:MAX_PER_SUBJ]

    _save_json_atomic(json_path, data)
    dbg(f"[IDX][SAVE] source={source_name} subj='{subj}' added={added} updated={updated} total_now={len(data['TerminalIndexes'][subj])}")


# =========================
# MODIFIED: js_capable_fetch (Playwright reuse)
# =========================
def js_capable_fetch(
    url: str,
    timeout_ms: int = 15000,
    wait_selector: str | None = "a[href]",
    js_mode: str = "full",          # "full" | "light" | "none" | "smart-light"
    max_modal_clicks: int = 3,
    # smart-light thresholds (tunable; ignored unless js_mode="smart-light")
    min_text_chars: int = 1200,
    min_p_tags: int = 3,
    min_anchors_hint: int = 8,
    # NEW: reuse one Playwright browser/context
    pw: "_PlaywrightReuse | None" = None,
) -> Tuple[str | None, List[str], str]:
    """
    Fetch a page and return (html, anchors, via).

    js_mode:
      - "full": Playwright render + DOM expansion + modal/shadow/data-* harvesting.
      - "light": Playwright render only; NO expansion/modals/shadow scan (visible anchors fast path).
      - "none": Plain requests + BeautifulSoup; no JS, no expansion.
      - "smart-light": Try requests first (cheap text + anchors); if text is too thin, fallback to Playwright light.

    Notes:
      - Non-HTML resources and blocklisted URLs are skipped early.
      - Anchors are absolutized against the input URL.
      - 'via' indicates which path was used: "playwright:full", "playwright:light", "requests", or "requests:light".
    """
    from typing import Tuple, List, Optional
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    # quick skips
    low = (url or "").lower()
    if low.endswith((".pdf", ".docx", ".zip", ".pptx", ".xls", ".xlsx",
                     ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")):
        dbg(f"[SKIP] non-HTML resource: {url}")
        return None, [], "none"
    if any(bad in low for bad in BLOCKED_URL_WORDS):
        dbg(f"[SKIP][BLOCKLIST] {url}")
        return None, [], "blocked"

    # ---- helper: requests-only fetch (shared by 'none' and 'smart-light' first pass) ----
    def _requests_fetch(u: str) -> Tuple[Optional[str], List[str], str]:
        try:
            r = requests.get(u, headers=UA, timeout=timeout_ms / 1000)
            r.raise_for_status()
            ct = (r.headers.get("Content-Type") or "").lower()
            if "html" not in ct and "xml" not in ct:
                return None, [], "requests"
            html = r.text
            soup = BeautifulSoup(html, "html.parser")
            anchors = []
            for a in soup.find_all("a", href=True):
                try:
                    anchors.append(urljoin(u, a.get("href")))
                except Exception:
                    continue
            return html, anchors, "requests"
        except Exception as e:
            dbg(f"[JSFETCH][requests] failed for {u}: {e}")
            return None, [], "requests"

    # ---- js_mode: none (requests + BS4) ----
    if js_mode == "none":
        html, anchors, _ = _requests_fetch(url)
        dbg(f"[JSFETCH][requests] {_short_url(url)} anchors={len(anchors)} html_len={len(html or '')}")
        return html, anchors, "requests"

    # ---- js_mode: smart-light (requests-first; fallback to playwright:light) ----
    if js_mode == "smart-light":
        html, anchors, _ = _requests_fetch(url)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            text_len = len(soup.get_text(" ", strip=True))
            p_tags = len(soup.find_all("p"))
            if (text_len >= min_text_chars and p_tags >= min_p_tags) or len(anchors) >= min_anchors_hint:
                dbg(f"[JSFETCH][requests:light] {_short_url(url)} anchors={len(anchors)} (text={text_len}, p={p_tags})")
                return html, anchors, "requests:light"
            else:
                dbg(f"[JSFETCH][requests:light→pw] {_short_url(url)} (text={text_len}, p={p_tags}, a={len(anchors)})")

        # fallback to Playwright light (reused if pw provided)
        try:
            from urllib.parse import urljoin

            if pw is not None and pw.ok():
                page = pw.new_page(timeout_ms)
                try:
                    page.goto(url, wait_until="domcontentloaded")
                    try:
                        page.wait_for_load_state("networkidle", timeout=timeout_ms // 2)
                    except Exception:
                        pass
                    if wait_selector:
                        try:
                            page.wait_for_selector(wait_selector, state="attached", timeout=timeout_ms // 2)
                        except Exception:
                            pass

                    anchors_pw = collect_links_with_modals_sync(page, expand=False, max_modal_clicks=0)
                    anchors_pw = [urljoin(url, a) for a in anchors_pw if a]
                    html_pw = page.content()
                    dbg(f"[JSFETCH][playwright:light] {_short_url(url)} anchors={len(anchors_pw)} html_len={len(html_pw)}")
                    return html_pw, anchors_pw, "playwright:light"
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass

            # non-reuse path
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                ctx = browser.new_context()
                page = ctx.new_page()
                page.set_default_timeout(timeout_ms)

                page.goto(url, wait_until="domcontentloaded")
                try:
                    page.wait_for_load_state("networkidle", timeout=timeout_ms // 2)
                except Exception:
                    pass
                if wait_selector:
                    try:
                        page.wait_for_selector(wait_selector, state="attached", timeout=timeout_ms // 2)
                    except Exception:
                        pass

                anchors_pw = collect_links_with_modals_sync(page, expand=False, max_modal_clicks=0)
                anchors_pw = [urljoin(url, a) for a in anchors_pw if a]
                html_pw = page.content()

                try:
                    ctx.close()
                except Exception:
                    pass
                browser.close()

                dbg(f"[JSFETCH][playwright:light] {_short_url(url)} anchors={len(anchors_pw)} html_len={len(html_pw)}")
                return html_pw, anchors_pw, "playwright:light"

        except Exception as e:
            dbg(f"[JSFETCH][playwright][smart-light] failed for {_short_url(url)}: {e}")

        return html, anchors, "requests:light" if html else (None, [], "none")

    # ---- js_mode: full/light (Playwright) ----
    expand = (js_mode == "full")
    try:
        from urllib.parse import urljoin

        if pw is not None and pw.ok():
            page = pw.new_page(timeout_ms)
            try:
                page.goto(url, wait_until="domcontentloaded")
                try:
                    page.wait_for_load_state("networkidle", timeout=timeout_ms // 2)
                except Exception:
                    pass
                if wait_selector:
                    try:
                        page.wait_for_selector(wait_selector, state="attached", timeout=timeout_ms // 2)
                    except Exception:
                        pass

                anchors = collect_links_with_modals_sync(
                    page,
                    expand=expand,
                    max_modal_clicks=(max_modal_clicks if expand else 0)
                )
                anchors = [urljoin(url, a) for a in anchors if a]
                html = page.content()

                mode_tag = f"playwright:{'full' if expand else 'light'}"
                dbg(f"[JSFETCH][{mode_tag}] {_short_url(url)} anchors={len(anchors)} html_len={len(html)}")
                return html, anchors, mode_tag
            finally:
                try:
                    page.close()
                except Exception:
                    pass

        # non-reuse path
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context()
            page = ctx.new_page()
            page.set_default_timeout(timeout_ms)

            page.goto(url, wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("networkidle", timeout=timeout_ms // 2)
            except Exception:
                pass
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, state="attached", timeout=timeout_ms // 2)
                except Exception:
                    pass

            anchors = collect_links_with_modals_sync(
                page,
                expand=expand,
                max_modal_clicks=(max_modal_clicks if expand else 0)
            )
            anchors = [urljoin(url, a) for a in anchors if a]
            html = page.content()

            try:
                ctx.close()
            except Exception:
                pass
            browser.close()

            mode_tag = f"playwright:{'full' if expand else 'light'}"
            dbg(f"[JSFETCH][{mode_tag}] {_short_url(url)} anchors={len(anchors)} html_len={len(html)}")
            return html, anchors, mode_tag

    except Exception as e:
        dbg(f"[JSFETCH][playwright] failed for {_short_url(url)}: {e}")

    return None, [], "none"






# =========================
# WordNet
# =========================
def get_limited_lemmas(word, per_synset_limit=4, pos_filter=None, debug=True):
    if isinstance(pos_filter, str):
        pos_filter = [pos_filter]

    try:
        synsets = wn.synsets(word)
    except Exception:
        synsets = []
    if pos_filter:
        synsets = [s for s in synsets if s.pos() in pos_filter]

    if debug:
        print(f"\n[LEMMA] word='{word}'  synsets_found={len(synsets)}  pos_filter={pos_filter or 'all'}")

    results = {}
    for i, syn in enumerate(synsets, start=1):
        lemmas = syn.lemmas()
        limited = [lemma.name().replace('_', ' ') for lemma in lemmas[:per_synset_limit]]
        results[syn.name()] = {"definition": syn.definition(), "lemmas": limited}
        if debug:
            print(f"  [{i}] synset='{syn.name()}' ({syn.pos()}) → def='{syn.definition()[:70]}...'")
            print(f"      lemmas: {', '.join(limited)}")

    if debug and not synsets:
        print(f"  [!] No synsets found for '{word}'")

    return results

# =========================
# Text Hit + Image (Stage 2.5 + Stage 3)
# =========================

BG_URL_RE = re.compile(r'url\((["\']?)(.*?)\1\)', re.I)

def _urls_from_style_attr(tag: Tag, page_url: str) -> list[str]:
    urls = []
    try:
        style = tag.get("style") or ""
        for _, u in BG_URL_RE.findall(style):
            try:
                urls.append(urljoin(page_url, u))
            except Exception:
                continue
    except Exception:
        pass
    return urls

def save_inline_svg(tag: Tag, dest_dir: str, source_url: str) -> Optional[str]:
    """Dump inline <svg> (closest figure) to a .svg file."""
    try:
        svg = tag if tag.name == "svg" else tag.find("svg")
        if not svg:
            return None
        os.makedirs(dest_dir, exist_ok=True)
        raw = str(svg)
        # name hint from page + hash of content
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        base = os.path.basename(urlparse(source_url).path) or "inline"
        base = re.sub(r"[^\w.-]+","_", base)
        out = os.path.join(dest_dir, f"{base}.{h}.svg")
        with open(out, "w", encoding="utf-8") as f:
            f.write(raw)
        dbg(f"[IMG_SAVE][inline-svg] {out} ({len(raw)} chars)")
        return out
    except Exception as e:
        dbg(f"[IMG_SAVE][inline-svg][err] {e}")
        return None

# --- robust image sniffers / naming ---

def _sniff_image_type(head: bytes, ctype: str | None, as_text: str | None) -> str | None:
    """
    Return a lowercase image type extension without dot (png, jpg, gif, webp, svg, tiff, bmp)
    by checking (1) content-type, (2) magic bytes, (3) svg text.
    None if not confidently an image.
    """
    ctype = (ctype or "").lower()

    # 1) Header hints
    if ctype.startswith("image/"):
        mapping = {
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "image/png": "png",
            "image/gif": "gif",
            "image/webp": "webp",
            "image/svg+xml": "svg",
            "image/tiff": "tiff",
            "image/bmp": "bmp",
        }
        ext = mapping.get(ctype.split(";")[0].strip())
        if ext:
            return ext

    # 2) Magic bytes
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head[:3] == b"\xff\xd8\xff":
        return "jpg"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:4] in (b"MM\x00*", b"II*\x00"):
        return "tiff"
    if head[:2] == b"BM":
        return "bmp"

    # 3) SVG (text)
    if as_text:
        t = as_text.lstrip().lower()
        if t.startswith("<svg") or ("<svg" in t[:2048] and "</svg>" in t):
            return "svg"

    return None


def _unique_path_with_ext(dest_dir: str, stem: str, ext: str) -> str:
    """
    Build a unique path under dest_dir with given stem and extension.
    """
    os.makedirs(dest_dir, exist_ok=True)
    base = f"{stem}.{ext}"
    path = os.path.join(dest_dir, base)
    if not os.path.exists(path):
        return path
    i = 1
    while True:
        cand = os.path.join(dest_dir, f"{stem}_{i}.{ext}")
        if not os.path.exists(cand):
            return cand
        i += 1


def compile_phrase_regexes(base_query: str, lemma_obj: dict) -> List[re.Pattern]:
    """
    Build *only* exact-phrase regexes for the query + its lemmas (case-insensitive).
    No tokens, no proximity.
    """
    phrases = []
    seen = set()

    def add(p: str):
        k = (p or "").strip().lower()
        if k and k not in seen:
            seen.add(k); phrases.append(p)

    add(base_query)
    for entry in (lemma_obj or {}).values():
        for w in entry.get("lemmas", []):
            if isinstance(w, str) and len(w.strip()) > 1:
                add(w)

    def _compile_exact(phrase: str) -> re.Pattern:
        # word-boundary, whitespace-normalized, case-insensitive
        esc = r"\b" + re.sub(r"\s+", " ", re.escape(phrase)).replace(r"\ ", " ") + r"\b"
        return re.compile(esc, re.IGNORECASE)

    rxs = [_compile_exact(p) for p in phrases]
    dbg(f"[PHRASES] base='{base_query}' → phrases ({len(phrases)}): {phrases[:12]}{' ...' if len(phrases)>12 else ''}")
    return rxs

import re

def _compile_phrase_regexes_adapter(base_query: str, lemma_obj: dict | None = None):
    """
    Build regex patterns and token helpers for the given phrase or lemma set.
    Used in smart_find_hits_in_soup() and stage2_drill_search_and_images().
    Returns:
        (phrase_rxs, proximity_tokens)
        - phrase_rxs: list of compiled regex objects (ordered by strictness)
        - proximity_tokens: set of key tokens to allow loose matching
    """

    def _escape_ph(s: str) -> str:
        return re.escape(s.strip())

    # Build base phrases list from the query and lemma expansions (if any)
    phrases = []
    if lemma_obj:
        for k, vs in lemma_obj.items():
            phrases.extend(vs)
    phrases.append(base_query)
    phrases = [p for p in phrases if p.strip()]
    phrases = sorted(set(phrases), key=len, reverse=True)  # longest first

    phrase_rxs = []
    for p in phrases:
        esc = _escape_ph(p)
        # exact match with word boundaries, case-insensitive
        strict = re.compile(rf"\b{esc}\b", re.IGNORECASE)
        phrase_rxs.append(strict)

        # flexible variant: allows small gaps/punctuation between words
        parts = [re.escape(tok) for tok in re.split(r"\s+", p.strip()) if tok]
        if len(parts) > 1:
            fuzzy = re.compile(r"\b" + r"\s*(?:[^\w\s]{0,3}\s*)?".join(parts) + r"\b", re.IGNORECASE)
            phrase_rxs.append(fuzzy)

    # token set for proximity scoring (smart_find_hits_in_soup)
    proximity_tokens = set()
    for p in phrases:
        for t in re.split(r"\W+", p.lower()):
            if len(t) > 2:
                proximity_tokens.add(t)

    return phrase_rxs, proximity_tokens



def nearest_image_src_for_hit(node: NavigableString | Tag, soup: BeautifulSoup, page_url: str):
    def absu(x):
        try: return urljoin(page_url, x) if x else None
        except Exception: return None

    def is_imgish_url(u: str) -> bool:
        if not u: return False
        if DATA_URL_RE.search(u): return False
        if UI_IMG_SKIP_RE.search(u): return False
        if IMG_EXT_RE.search(u): return True
        # OpenStax & typical CDNs
        return ("image-cdn" in u) or ("/resources/" in u) or ("/media/" in u)

    def pick_img_src_from_img(img: Tag) -> Optional[str]:
        for k in ("srcset","data-srcset"):
            sv = img.get(k)
            if sv:
                best_u, best_w = None, -1
                for part in [p.strip() for p in sv.split(",") if p.strip()]:
                    bits = part.split()
                    cu = absu(bits[0]) if bits else None
                    if not (cu and is_imgish_url(cu)): continue
                    w = -1
                    for b in bits[1:]:
                        m = re.match(r"(\d+)w$", b)
                        if m: w = int(m.group(1)); break
                    if w > best_w: best_w, best_u = w, cu
                if best_u: return best_u
        for k in ("src","data-src","data-original"):
            cu = absu(img.get(k))
            if cu and is_imgish_url(cu): return cu
        if img.parent and isinstance(img.parent, Tag) and img.parent.name == "picture":
            for src in img.parent.find_all("source"):
                for k in ("srcset","src"):
                    cu = absu(src.get(k))
                    if cu and is_imgish_url(cu): return cu
        return None

    def pick_from_anchor(a: Tag) -> Optional[str]:
        cu = absu(a.get("href"))
        if cu and is_imgish_url(cu): return cu
        img = a.find("img")
        if img: return pick_img_src_from_img(img)
        return None

    def pick_from_object(obj: Tag) -> Optional[str]:
        t = (obj.get("type") or "").lower()
        if t in ("image/svg+xml","image/png","image/jpeg","image/webp","image/gif"):
            cu = absu(obj.get("data") or obj.get("src"))
            if cu and is_imgish_url(cu): return cu
        return None

    def css_bg_urls(tag: Tag) -> list[str]:
        urls = []
        urls.extend(_urls_from_style_attr(tag, page_url))
        p = tag.parent; hops = 0
        while p and hops < 2:
            urls.extend(_urls_from_style_attr(p, page_url))
            p = p.parent; hops += 1
        return [u for u in urls if is_imgish_url(u)]

    start = node.parent if isinstance(node, NavigableString) else node
    if not start: return None

    best = (None, -1)
    q, seen = [start], {id(start)}
    steps = 0
    MAX_NODES = 700

    while q and steps < MAX_NODES:
        cur = q.pop(0); steps += 1
        if not isinstance(cur, Tag): continue
        if _is_excluded_region(cur):  # skip header/footer/menus
            continue

        # Prefer figure-like areas first
        looks_figure = cur.name == "figure" or "figure" in (cur.get("class") or []) \
                       or cur.has_attr("data-type") and cur.get("data-type") == "figure"
        if looks_figure:
            # 1) inline <svg> (return the Tag directly)
            svg = cur.find("svg")
            if svg and len(str(svg)) > 2000:
                return svg

            # 2) <object>/<embed>
            for obj in cur.find_all(["object","embed"]):
                u = pick_from_object(obj)
                if u and not UI_IMG_SKIP_RE.search(u):
                    score = 3800
                    if score > best[1]: best = (u, score)

            # 3) plain <img>
            for img in cur.find_all("img"):
                u = pick_img_src_from_img(img)
                if u and not UI_IMG_SKIP_RE.search(u):
                    score = 3600
                    if score > best[1]: best = (u, score)

            # 4) CSS background
            for u in css_bg_urls(cur):
                score = 3400
                if score > best[1]: best = (u, score)

        # Nearby anchors/images (limited)
        for a in cur.find_all("a", href=True, limit=4):
            u = pick_from_anchor(a)
            if u and not UI_IMG_SKIP_RE.search(u):
                score = 1400
                if score > best[1]: best = (u, score)

        for img in cur.find_all("img", limit=4):
            u = pick_img_src_from_img(img)
            if u and not UI_IMG_SKIP_RE.search(u):
                score = 1200
                if score > best[1]: best = (u, score)

        # expand neighborhood
        for ch in cur.children:
            if isinstance(ch, Tag) and id(ch) not in seen:
                seen.add(id(ch)); q.append(ch)
        for sib in (cur.previous_sibling, cur.next_sibling, cur.parent):
            if isinstance(sib, Tag) and id(sib) not in seen:
                seen.add(id(sib)); q.append(sib)

    return best[0]

def save_image_candidate(
    session: requests.Session,
    candidate,                 # str URL OR <svg> Tag
    soup_for_inline: BeautifulSoup,
    page_url: str,
    dest_dir: str,
    referer: Optional[str] = None,
    dedupe_set: Optional[set] = None,
) -> Optional[str]:
    # inline SVG case
    if isinstance(candidate, Tag) and candidate.name == "svg":
        return save_inline_svg(candidate, dest_dir, page_url)

    # URL case
    if isinstance(candidate, str):
        if dedupe_set is not None and candidate in dedupe_set:
            return None
        d = download_image(session, candidate, dest_dir)
        if d and dedupe_set is not None:
            dedupe_set.add(candidate)
        return d

    return None




# ---------- downloader (stricter + clearer debug) ----------

def _is_diagram_like(data: bytes,
                     white_min_frac: float = 0.22,    # >= ~22% whitespace
                     edge_majority: float = 0.2,     # >= 15% common edge color - normal images dont (usually) hold that
                     resize_max: int = 900) -> bool:
    """
    Heuristic 'diagram-like' check for raster images:
      1) Whitespace (white OR transparent) area >= white_min_frac.
      2) Among edges that TOUCH whitespace, one ink-like color dominates
         (mostly dark) OR one hue bin dominates.

    Strong logging so you can see decisions in your debug stream.
    """
    import io
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        dbg("[DIAGRAM] PIL/numpy not available -> ALLOW")
        return True

    # Decode with alpha preserved if present
    try:
        im0 = Image.open(io.BytesIO(data))
        # Keep alpha if present; otherwise convert to RGB
        im = im0.convert("RGBA") if im0.mode in ("RGBA", "LA", "P") else im0.convert("RGB")
    except Exception:
        dbg("[DIAGRAM] decode failed -> REJECT")
        return False

    # Resize for speed
    w, h = im.size
    if max(w, h) > resize_max:
        scale = resize_max / float(max(w, h))
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        w, h = im.size

    arr = np.asarray(im, dtype=np.uint8)
    has_alpha = arr.shape[2] == 4
    if has_alpha:
        rgb = arr[..., :3]
        alpha = arr[..., 3].astype(np.float32) / 255.0
    else:
        rgb = arr[..., :3]
        alpha = np.ones((h, w), dtype=np.float32)

    # ---------- WHITESPACE (white OR transparent) ----------
    # Treat low-alpha pixels as whitespace straight up
    trans_mask = alpha <= 0.06  # ~<= 6% opacity counts as white
    # Near-white RGB (tolerate light gray)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    near_white_rgb = (r > 235) & (g > 235) & (b > 235)

    # Low saturation + bright also counts (handles compressed WEBP/PNG)
    rgb_f = rgb.astype(np.float32) / 255.0
    cmax = rgb_f.max(axis=-1)
    cmin = rgb_f.min(axis=-1)
    delta = cmax - cmin
    with np.errstate(divide='ignore', invalid='ignore'):
        sat = np.where(cmax > 0, delta / (cmax + 1e-8), 0.0)
    bright = cmax >= 0.93
    near_white_hsv = bright & (sat <= 0.18)

    white_mask = trans_mask | near_white_rgb | near_white_hsv
    white_frac = float(white_mask.mean())

    if white_frac < white_min_frac:
        dbg(f"[DIAGRAM] white_frac={white_frac:.3f} < {white_min_frac:.3f} -> REJECT")
        return False

    # ---------- EDGES touching whitespace ----------
    # Grayscale for edges
    gray = (0.2989 * rgb_f[..., 0] + 0.5870 * rgb_f[..., 1] + 0.1140 * rgb_f[..., 2])
    try:
        import cv2
        edges = cv2.Canny((gray * 255).astype(np.uint8), 80, 180) > 0
    except Exception:
        # Sobel fallback
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
        gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
        mag = np.hypot(gx, gy)
        thr = max(0.2, float(mag.mean() + 1.5 * mag.std()))
        edges = mag > thr

    # Dilate whitespace a bit so “touch” is robust
    wm = white_mask
    neigh = wm.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neigh |= np.roll(np.roll(wm, dx, axis=0), dy, axis=1)

    touch_white = edges & neigh
    idx = np.where(touch_white)
    count = int(idx[0].size)
    if count < 80:
        dbg(f"[DIAGRAM] edge_touch_count={count} < 80 -> REJECT")
        return False

    # Colors at edge positions (avoid sampling transparent: replace with nearest opaque)
    rr = rgb[idx[0], idx[1], 0].astype(np.int16)
    gg = rgb[idx[0], idx[1], 1].astype(np.int16)
    bb = rgb[idx[0], idx[1], 2].astype(np.int16)
    aa = alpha[idx[0], idx[1]] if has_alpha else np.ones_like(rr, dtype=np.float32)

    # If most edge-touch samples are actually transparent, reject (icons/logos often)
    trans_edge_frac = float((aa <= 0.06).mean())
    if trans_edge_frac > 0.95:
        dbg(f"[DIAGRAM] edges mostly transparent ({trans_edge_frac:.3f}) -> REJECT")
        return False
    else:
        dbg(f"[DIAGRAM] edges transparent frac={trans_edge_frac:.3f} (no reject)")

    # “Ink-like dark” dominance
    dark = (rr < 70) & (gg < 70) & (bb < 70) & (aa > 0.2)
    dark_frac = float(dark.mean())

    # Hue dominance on sufficiently saturated edge pixels
    rf = rr / 255.0
    gf = gg / 255.0
    bf = bb / 255.0
    cmax = np.maximum.reduce([rf, gf, bf])
    cmin = np.minimum.reduce([rf, gf, bf])
    delta = cmax - cmin + 1e-8
    sat_edges = np.where(cmax > 0, delta / (cmax + 1e-8), 0.0)
    h = np.zeros_like(cmax)
    r_is_max = (cmax == rf)
    g_is_max = (cmax == gf) & ~r_is_max
    b_is_max = ~r_is_max & ~g_is_max
    h[r_is_max] = ((gf[r_is_max] - bf[r_is_max]) / delta[r_is_max]) % 6.0
    h[g_is_max] = ((bf[g_is_max] - rf[g_is_max]) / delta[g_is_max]) + 2.0
    h[b_is_max] = ((rf[b_is_max] - gf[b_is_max]) / delta[b_is_max]) + 4.0
    h = (h / 6.0) % 1.0

    hue_bins = 24
    mask_hue = (sat_edges >= 0.35) & (aa > 0.2) & ~dark  # look at colored (non-dark) edges
    if mask_hue.any():
        bins = np.floor(h[mask_hue] * hue_bins).astype(int)
        bins = np.clip(bins, 0, hue_bins - 1)
        counts = np.bincount(bins, minlength=hue_bins).astype(np.float32)
        hue_dom_frac = float(counts.max() / max(1.0, counts.sum()))
    else:
        hue_dom_frac = 0.0

    ok = white_frac >= white_min_frac and ((dark_frac >= edge_majority) or (hue_dom_frac >= edge_majority))
    dbg(f"[DIAGRAM] white={white_frac:.3f} dark_edge={dark_frac:.3f} hue_dom={hue_dom_frac:.3f} -> {'ALLOW' if ok else 'REJECT'}")
    return ok

class PlaywrightReuseFetcher:
    """
    One browser/context/page reused for ALL Stage2 fetches.
    """
    def __init__(self, timeout_ms: int = 15000):
        from playwright.sync_api import sync_playwright
        self.timeout_ms = timeout_ms
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=True)
        self.ctx = self.browser.new_context()
        self.page = self.ctx.new_page()
        self.page.set_default_timeout(timeout_ms)

    def close(self):
        try:
            self.ctx.close()
        except Exception:
            pass
        try:
            self.browser.close()
        except Exception:
            pass
        try:
            self._pw.stop()
        except Exception:
            pass


def js_capable_fetch_reuse(
    url: str,
    *,
    fetcher: PlaywrightReuseFetcher,
    timeout_ms: int = 15000,
    js_mode: str = "smart-light",   # "full"|"light"|"smart-light"|"none"
    max_modal_clicks: int = 3,
    wait_selector: str | None = "a[href]",
    min_text_chars: int = 1200,
    min_p_tags: int = 3,
    min_anchors_hint: int = 8,
) -> tuple[str | None, list[str], str]:
    """
    Same semantics as js_capable_fetch, but reuses ONE Playwright page.
    """
    low = (url or "").lower()
    if low.endswith((".pdf", ".docx", ".zip", ".pptx", ".xls", ".xlsx",
                     ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")):
        return None, [], "none"

    if any(bad in low for bad in BLOCKED_URL_WORDS):
        return None, [], "blocked"

    # ---- requests path ----
    def _requests_fetch(u: str):
        try:
            r = requests.get(u, headers=UA, timeout=timeout_ms / 1000)
            r.raise_for_status()
            ct = (r.headers.get("Content-Type") or "").lower()
            if "html" not in ct and "xml" not in ct:
                return None, [], "requests"
            html = r.text
            soup = BeautifulSoup(html, "html.parser")
            anchors = []
            for a in soup.find_all("a", href=True):
                anchors.append(urljoin(u, a.get("href")))
            return html, anchors, "requests"
        except Exception:
            return None, [], "requests"

    if js_mode == "none":
        return _requests_fetch(url)

    if js_mode == "smart-light":
        html, anchors, via = _requests_fetch(url)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            text_len = len(soup.get_text(" ", strip=True))
            p_tags = len(soup.find_all("p"))
            if (text_len >= min_text_chars and p_tags >= min_p_tags) or len(anchors) >= min_anchors_hint:
                return html, anchors, "requests:light"
            dbg(f"[JSFETCH][requests:light→pw] {urlparse(url).netloc}{urlparse(url).path} (text={text_len}, p={p_tags}, a={len(anchors)})")

        # fallback to playwright:light
        js_mode = "light"

    # ---- playwright path ----
    page = fetcher.page
    try:
        page.goto(url, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=timeout_ms // 2)
        except Exception:
            pass
        if wait_selector:
            try:
                page.wait_for_selector(wait_selector, state="attached", timeout=timeout_ms // 2)
            except Exception:
                pass

        expand = (js_mode == "full")
        anchors_pw = collect_links_with_modals_sync(
            page,
            expand=expand,
            max_modal_clicks=(max_modal_clicks if expand else 0)
        )
        anchors_pw = [urljoin(url, a) for a in anchors_pw if a]
        html_pw = page.content()
        return html_pw, anchors_pw, f"playwright:{'full' if expand else 'light'}"

    except Exception as e:
        dbg(f"[JSFETCH][reuse][err] {url} -> {e}")
        return None, [], "none"



# === REPLACE: download_image (runs filter & logs decision) ===================
def download_image(session: requests.Session, img_url: str, dest_dir: str) -> Optional[str]:
    """
    Robust downloader with diagram filter:
      - supports data: URLs (base64)
      - follows redirects
      - rejects HTML pages
      - infers extension from content-type/magic bytes/SVG text
      - assigns deterministic name from SHA1(content) to avoid accidental overwrite
      - skips tiny UI assets (< 6 KB) unless SVG
      - RUNS '_is_diagram_like' on raster images; rejects if it fails
    """
    import re, os, hashlib, base64
    from urllib.parse import urlparse

    def _save_bytes(raw: bytes, ext_hint: str, ctype_hint: str, source_label: str) -> Optional[str]:
        # Try to detect real ext
        as_text = None
        try:
            as_text = raw.decode("utf-8", errors="ignore")
        except Exception:
            pass
        ext = _sniff_image_type(raw[:64], ctype_hint, as_text) or ext_hint or "img"

        # Tiny asset guard (except SVG)
        if ext != "svg" and len(raw) < 6 * 1024:
            dbg(f"[IMG_SAVE][tiny/ui-skip] {source_label} ({len(raw)} B)")
            return None

        # Diagram filter for raster (skip SVG)
        if ext != "svg":
            if not _is_diagram_like(raw):
                dbg(f"[IMG_SAVE][filter] {source_label} REJECTED by diagram filter")
                return None

        h = hashlib.sha1(raw).hexdigest()
        out = _unique_path_with_ext(dest_dir, h, ext)
        with open(out, "wb") as f:
            f.write(raw)
        dbg(f"[IMG_SAVE] {source_label} -> {out}")
        return out

    try:
        # --- data: URLs ---
        if img_url.startswith("data:image/"):
            m = re.match(r"^data:(image/[\w\+\-\.]+);base64,(.+)$", img_url, re.I | re.S)
            if not m:
                return None
            ctype = m.group(1)
            payload = m.group(2)
            raw = base64.b64decode(payload, validate=True)
            return _save_bytes(raw, ext_hint="", ctype_hint=ctype, source_label="data-url")

        # --- http(s) URLs ---
        headers = {"Referer": img_url, **UA}
        with session.get(img_url, headers=headers, stream=True, timeout=30, allow_redirects=True) as r:
            r.raise_for_status()

            ctype = r.headers.get("Content-Type", "")
            # Peek to detect HTML and type
            buf = bytearray()
            max_peek = 4096
            for chunk in r.iter_content(4096):
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) >= max_peek:
                    break

            head_lower = bytes(buf[:64]).lower()
            if b"<html" in head_lower or b"<!doctype html" in head_lower:
                dbg(f"[IMG_SAVE][html-skip] {img_url}")
                return None

            # Content-Length-based tiny guard (pre-stream)
            cl_header = r.headers.get("Content-Length")
            maybe_len = int(cl_header) if (cl_header and cl_header.isdigit()) else None
            if maybe_len is not None and maybe_len < 6 * 1024:
                dbg(f"[IMG_SAVE][tiny/ui-skip] {img_url} ({maybe_len} B)")
                return None

            # Stream the rest
            data = bytes(buf)
            for chunk in r.iter_content(8192):
                if not chunk:
                    break
                data += chunk

            # Detect extension (URL as hint if needed)
            path_ext = os.path.splitext(urlparse(img_url).path)[1].lower().lstrip(".")
            ext_hint = "jpg" if path_ext == "jpeg" else (path_ext if path_ext in {"png","jpg","jpeg","gif","webp","svg","tiff","bmp"} else "")

            return _save_bytes(data, ext_hint=ext_hint, ctype_hint=ctype, source_label=img_url)

    except Exception as e:
        dbg(f"[IMG_SAVE][err] {img_url} -> {e}")
        return None




def _extract_img_candidates_near_hit(node, soup, page_url, *, per_hit_cap: Optional[int] = None):
    """
    Return image URLs near a hit node (optionally capped by per_hit_cap).
    Handles <img src>, data-* sources, srcset (chooses largest), <picture><source>,
    and <a href="...img.ext">. Searches nearest <figure>/<section>/<article> ancestors,
    then a limited sibling/parent scope, then (last) a small global fallback.
    """
    from urllib.parse import urljoin

    cand = []
    seen = set()

    def _add(u, why):
        if not u:
            return
        try:
            u_abs = urljoin(page_url, u)
        except Exception:
            return
        key = (u_abs, why)
        if key in seen:
            return
        if per_hit_cap is not None and len(cand) >= per_hit_cap:
            return
        seen.add(key)
        cand.append((u_abs, why))

    def _img_url_from_tag(img):
        # Prefer srcset/data-srcset largest width, then src / data-* fallbacks
        srcset = (img.get("srcset") or img.get("data-srcset") or "").strip()
        if srcset:
            items = []
            for part in srcset.split(","):
                p = part.strip().split()
                if not p:
                    continue
                url = p[0]
                w = 0
                if len(p) >= 2 and p[1].endswith("w"):
                    try:
                        w = int(p[1][:-1])
                    except Exception:
                        w = 0
                elif len(p) >= 2 and p[1].endswith("x"):
                    try:
                        w = int(float(p[1][:-1]) * 1000)
                    except Exception:
                        w = 0
                items.append((w, url))
            if items:
                items.sort(reverse=True)
                return items[0][1]

        for attr in ("src", "data-src", "data-original", "data-lazy", "data-image"):
            v = (img.get(attr) or "").strip()
            if v:
                return v
        return None

    # Prefer closest semantic ancestor first
    cur = node
    steps_up = 5
    hit_container = None
    while cur is not None and steps_up > 0:
        name = getattr(cur, "name", "").lower() if hasattr(cur, "name") else ""
        if name in ("figure", "section", "article", "main"):
            hit_container = cur
            break
        cur = getattr(cur, "parent", None)
        steps_up -= 1

    scopes = []
    if hit_container:
        scopes.append(hit_container)
    if getattr(node, "parent", None):
        scopes.append(node.parent)
    # small global fallback
    scopes.append(soup)

    for sc in scopes:
        # <picture><source> first
        for pic in sc.find_all("picture"):
            for src in pic.find_all("source"):
                srcset = (src.get("srcset") or "").strip()
                if srcset:
                    url = srcset.split(",")[-1].strip().split()[0]
                    _add(url, "picture/srcset")
                    if per_hit_cap is not None and len(cand) >= per_hit_cap:
                        return cand
            img = pic.find("img")
            if img:
                url = _img_url_from_tag(img)
                if url:
                    _add(url, "picture/img")
                    if per_hit_cap is not None and len(cand) >= per_hit_cap:
                        return cand

        # plain <img>
        for img in sc.find_all("img"):
            url = _img_url_from_tag(img)
            if url:
                _add(url, "img")
                if per_hit_cap is not None and len(cand) >= per_hit_cap:
                    return cand

        # <a href="...img.ext">
        for a in sc.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if href and re.search(r"\.(png|jpe?g|gif|webp|svg|tiff?)($|\?)", href, re.I):
                _add(href, "a->image")
                if per_hit_cap is not None and len(cand) >= per_hit_cap:
                    return cand

        if per_hit_cap is not None and len(cand) >= per_hit_cap:
            break

    if per_hit_cap is not None:
        return cand[:per_hit_cap]
    return cand

    
def partition_anchors(cur: str, anchors: list[str], allowed_fn):
    """
    Split anchor URLs from the current page into:
      - terminal: same parent path as current URL (ex: same section/page level)
      - deeper: deeper subpaths for recursive exploration
    Only keeps URLs that pass allowed_fn(a).
    """
    def _parent_part(u: str) -> str:
        p = urlparse(u)
        path = (p.path or "/").rstrip("/")
        if path and path != "/":
            path = path.rsplit("/", 1)[0] or "/"
        return urlunparse((p.scheme or "https", (p.hostname or "").lower(), path, "", "", ""))

    cur_parent = _parent_part(cur)
    terms, deeper = [], []
    for a in anchors:
        if not a or a == cur:
            continue
        if not allowed_fn(a):
            continue
        # same parent path → terminal
        if _parent_part(a) == cur_parent:
            terms.append(a)
        else:
            deeper.append(a)

    return ordered_dedupe(terms), ordered_dedupe(deeper)

def _url_last_segment_key(u: str) -> str:
    """
    Same idea as Stage 1: embed only the last path part of the URL.
    """
    try:
        p = urlparse(u)
        segs = [s for s in (p.path or "").split("/") if s]
        last = segs[-1] if segs else ""
        last = (last or "").lower().replace("-", " ").replace("_", " ")
        last = re.sub(r"\s+", " ", last).strip()
        return last
    except Exception:
        return ""


def _norm_vec(v):
    try:
        import numpy as np
        arr = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(arr))
        if n > 0:
            arr = arr / n
        return arr
    except Exception:
        return v


def _dot_sim(a, b) -> float:
    try:
        import numpy as np
        aa = np.asarray(a, dtype=np.float32)
        bb = np.asarray(b, dtype=np.float32)
        return float(np.dot(aa, bb))
    except Exception:
        return 0.0


def rank_urls_by_query_similarity(
    urls: list[str],
    *,
    encoder,
    query_embedding,
    top_k: int = 75,
    debug_label: str = "URL_SORT",
    dbg_top: int = 8,
) -> list[str]:
    """
    Reorder URLs so highest-similar are first, using only last segment key.
    Does NOT hard-drop the rest; it sorts them all, but we only print top K.
    """
    if not urls:
        return []

    if encoder is None or query_embedding is None:
        return urls

    keys = [_url_last_segment_key(u) for u in urls]
    try:
        embs = encoder.encode(keys, normalize_embeddings=True)
        q = _norm_vec(query_embedding)
        scores = [_dot_sim(q, e) for e in embs]
    except Exception as e:
        dbg(f"[S2][{debug_label}] embed fail: {e}")
        return urls

    ranked = sorted(zip(scores, urls, keys), key=lambda x: x[0], reverse=True)

    # Debug print
    dbg(f"[S2][{debug_label}] top {min(top_k, len(ranked))}/{len(ranked)}:")
    for s, u, k in ranked[:dbg_top]:
        dbg(f"   ↳ score={s:.3f} key='{k[:48]}' url={(urlparse(u).netloc + urlparse(u).path)}")

    return [u for _, u, _ in ranked]


def build_terminal_index_candidate(
    entry_url: str,
    terminal_urls: list[str],
    *,
    encoder,
    query_embedding,
    max_store: int = 600,
) -> dict | None:
    """
    Build an index object for a page that fans out into many terminal URLs.
    Centroid is built ONLY from terminal URLs (your requirement).
    """
    if encoder is None or query_embedding is None:
        return None
    if not terminal_urls:
        return None

    # limit storage size (big TOCs can be >1000)
    store_list = terminal_urls[:max_store]

    keys = [_url_last_segment_key(u) for u in store_list]
    try:
        import numpy as np
        embs = encoder.encode(keys, normalize_embeddings=True)
        centroid = np.mean(np.asarray(embs, dtype=np.float32), axis=0)
        centroid = centroid / max(1e-9, float(np.linalg.norm(centroid)))
        centroid_list = centroid.tolist()
        q = _norm_vec(query_embedding)
        sim = _dot_sim(q, centroid)
    except Exception:
        return None

    return {
        "entry_url": clean_url(entry_url),
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "terminal_count": int(len(terminal_urls)),
        "terminals": [clean_url(u) for u in store_list],
        "centroid_embedding": centroid_list,
        "centroid_sim_to_query": float(sim),
    }


def pick_best_terminal_index(
    indexes: list[dict],
    *,
    encoder,
    query_embedding,
    min_sim: float = 0.22,
    root_hint: str | None = None,
) -> tuple[dict | None, float]:
    """
    Read ALL terminal indexes for subject and pick the closest centroid to current prompt.
    """
    if not indexes or encoder is None or query_embedding is None:
        return None, 0.0

    q = _norm_vec(query_embedding)

    best = None
    best_sim = -1.0

    for idx in indexes:
        try:
            cent = idx.get("centroid_embedding")
            if not cent:
                continue

            # optional domain match hint
            if root_hint:
                pr = urlparse(root_hint)
                pi = urlparse(idx.get("entry_url") or "")
                if pr.hostname and pi.hostname:
                    if not same_registrable((pr.hostname or "").lower(), (pi.hostname or "").lower()):
                        continue

            sim = _dot_sim(q, cent)
            if sim > best_sim:
                best_sim = sim
                best = idx
        except Exception:
            continue

    if best is None or best_sim < min_sim:
        return None, float(best_sim)

    return best, float(best_sim)

def _json_safe(x):
    # keep it simple; your embeddings are already python lists
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    return x


def load_terminal_indexes_for_subject(source_name: str, subject: str) -> list[dict]:
    """
    Reads SOURCE_PATH/<source>.json and returns TerminalIndexes[subject] list.
    """
    try:
        path = os.path.join(SOURCE_PATH, f"{source_name}.json")
        if not os.path.exists(path):
            dbg(f"[IDX][LOAD] none (missing json) path={path}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        idxs = (data.get("TerminalIndexes", {}) or {}).get(subject, []) or []
        dbg(f"[IDX][LOAD] subject='{subject}' indexes={len(idxs)} path={path}")
        return idxs
    except Exception as e:
        dbg(f"[IDX][LOAD][ERR] {e}")
        return []


def persist_terminal_index_for_subject(source_name: str, subject: str, idx_obj: dict) -> None:
    """
    Writes idx_obj into SOURCE_PATH/<source>.json under TerminalIndexes[subject].
    Dedupes by entry_url.
    """
    path = os.path.join(SOURCE_PATH, f"{source_name}.json")
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}

        data.setdefault("TerminalIndexes", {})
        data["TerminalIndexes"].setdefault(subject, [])

        entry = clean_url(idx_obj.get("entry_url") or "")
        if not entry:
            dbg("[IDX][WRITE][REJECT] no entry_url")
            return

        # dedupe
        for ex in data["TerminalIndexes"][subject]:
            if clean_url(ex.get("entry_url") or "") == entry:
                dbg(f"[IDX][WRITE][SKIP] already exists entry={entry}")
                return

        data["TerminalIndexes"][subject].append(_json_safe(idx_obj))

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

        dbg(f"[IDX][WRITE] ✅ saved entry={entry}  total={len(data['TerminalIndexes'][subject])}  path={path}")
    except Exception as e:
        dbg(f"[IDX][WRITE][ERR] {e}")


def pick_best_terminal_index_for_prompt(indexes: list[dict], prompt_vec, min_sim: float = 0.22):
    """
    Returns (best_index, sim) using dot(prompt_vec, centroid_embedding).
    Accepts both formats:
      - "centroid_embedding" + "terminal_urls"
      - "centroid_embedding" + "terminals"
    """
    if not indexes or prompt_vec is None:
        return None, 0.0

    best = None
    best_sim = -1.0

    for idx in indexes:
        try:
            cent = idx.get("centroid_embedding")
            if not cent:
                continue
            sim = float(_dot_sim(prompt_vec, cent))
            if sim > best_sim:
                best_sim = sim
                best = idx
        except Exception:
            continue

    if best is None or best_sim < min_sim:
        return None, float(best_sim)

    return best, float(best_sim)






def _dbg_hit_reason(url, node, rx):
    try:
        s = node if isinstance(node, str) else (node.get_text(" ", strip=True) or "")
    except Exception:
        s = ""
    snippet = s[:140].replace("\n", " ")
    print(f"[HIT] url={url} rx=/{rx.pattern}/ -> '{snippet}...'")

# =========================
# NEW STAGE 2: Drill-by-parent with inline text search + image pick
# =========================
# =========================
# MODIFIED: Stage 2 drill + search + images
# =========================
def stage2_drill_search_and_images(
    roots: list[str] | dict,
    base_query: str,
    lemma_obj: dict,
    hard_image_cap: int,
    session: requests.Session,
    html_cache: dict,
    anchors_cache: dict,
    global_image_cap: int | None = None,
    per_hit_cap: Optional[int] = None,
    fatigue_limit: int = 25,
    encoder: Optional[Any] = None,
    query_embedding: Optional[Any] = None,
    prompt_embedding: Optional[Any] = None,

    url_sort_k: int = 75,

    # NEW: fully working terminal index system
    terminal_indexes: Optional[list[dict]] = None,
    terminal_index_min_terminals: int = 25,
    terminal_index_use_threshold: float = 0.22,
    terminal_index_store_cap: int = 600,
    terminal_index_centroid_sample: int = 60,

    subject: Optional[str] = None,
    source_name: Optional[str] = None,

    on_terminal_index_promoted: Optional[Any] = None,

    *_, **compat_kwargs,
) -> list[str]:

    import numpy as np

    # compatibility: some callers pass base_ctx_embedding
    if prompt_embedding is None:
        prompt_embedding = (
            compat_kwargs.pop("base_ctx_embedding", None) or
            compat_kwargs.pop("base_context_embedding", None) or
            compat_kwargs.pop("prompt_vec", None)
        )

    phrase_rxs, _ = _compile_phrase_regexes_adapter(base_query, lemma_obj)

    class _NextRoot(BaseException): pass
    class _StopAll(BaseException): pass

    saved_paths: list[str] = []

    # normalize roots input
    if isinstance(roots, dict):
        roots_list = list(roots.keys())
    else:
        roots_list = list(roots or [])

    # one playwright reuse
    pw = _PlaywrightReuse(headless=True)

    # normalized prompt vec
    prompt_vec = _norm_vec(prompt_embedding if prompt_embedding is not None else query_embedding)

    # url-key embedding cache
    key_emb_cache: dict[str, np.ndarray] = {}

    def _global_cap_hit() -> bool:
        return global_image_cap is not None and len(saved_paths) >= global_image_cap

    def _embed_keys_batch(keys: list[str]) -> None:
        if encoder is None:
            return
        missing = [k for k in keys if k and k not in key_emb_cache]
        if not missing:
            return
        try:
            embs = encoder.encode(missing, normalize_embeddings=True)
            for k, e in zip(missing, embs):
                key_emb_cache[k] = np.asarray(e, dtype=np.float32)
        except Exception as e:
            dbg(f"[S2][URL_EMBED][ERR] batch={len(missing)} -> {e}")

    def _score_and_sort_urls(urls: list[str], *, k: int) -> list[str]:
        urls = ordered_dedupe([u for u in urls if u])
        if not urls:
            return []
        if encoder is None or prompt_vec is None:
            return urls[:k] if k and len(urls) > k else urls

        keys = [_url_key_for_embed(u) for u in urls]
        _embed_keys_batch([kk for kk in keys if kk])

        scored = []
        for u, kk in zip(urls, keys):
            e = key_emb_cache.get(kk)
            s = float(np.dot(prompt_vec, e)) if e is not None else 0.0
            scored.append((s, u, kk))

        scored.sort(key=lambda x: x[0], reverse=True)

        # debug top few
        dbg("[S2][URL_SORT] top:")
        for s, u, kk in scored[:8]:
            dbg(f"   ↳ score={s:.3f} key='{kk}' url={_short_url(u)}")

        top = scored[:k] if k and len(scored) > k else scored
        return [u for _, u, _ in top]

    def _allowed(root: str, child_url: str) -> bool:
        try:
            return domain_and_phrase_lock(root, child_url)
        except Exception:
            return False

    def _subsample_terminals(urls: list[str], cap: int) -> list[str]:
        if not urls:
            return []
        if len(urls) <= cap:
            return urls
        step = max(1, len(urls) // cap)
        out = urls[::step][:cap]
        return out

    def _centroid_for_terminal_urls(term_urls: list[str]) -> Optional[list]:
        """
        centroid ONLY from terminal urls (last segment embeddings)
        """
        if encoder is None or not term_urls:
            return None

        sampled = _subsample_terminals(term_urls, terminal_index_centroid_sample)
        keys = [_url_key_for_embed(u) for u in sampled]
        keys = [k for k in keys if k]
        if not keys:
            return None

        _embed_keys_batch(keys)
        vecs = [key_emb_cache.get(k) for k in keys]
        vecs = [v for v in vecs if v is not None]
        if not vecs:
            return None

        v = np.mean(np.stack(vecs, axis=0), axis=0)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        return v.astype(np.float32).tolist()

    # -------------------------
    # TERMINAL INDEX STATE
    # -------------------------
    stored_entries = set()
    if terminal_indexes:
        for it in terminal_indexes:
            stored_entries.add(clean_url(it.get("entry_url") or ""))

    pending_indexes: dict[str, dict] = {}
    # structure:
    # pending_indexes[entry_url] = {
    #   "entry_url": ...,
    #   "terminal_set": set([...]),
    #   "idx_obj": {...},
    #   "saved_from_bucket": 0
    # }

    def _create_pending_index(entry_url: str, terminal_urls: list[str]):
        entry = clean_url(entry_url)
        if not entry:
            return
        if entry in stored_entries:
            return
        if entry in pending_indexes:
            return
        if len(terminal_urls) < terminal_index_min_terminals:
            return

        # store list cap
        store_terms = ordered_dedupe([clean_url(u) for u in terminal_urls if u])[:terminal_index_store_cap]
        cent = _centroid_for_terminal_urls(store_terms)
        if not cent:
            dbg(f"[IDX][PENDING][REJECT] entry={_short_url(entry)} terminals={len(terminal_urls)} reason=no-centroid")
            return

        idx_obj = {
            "id": hashlib.sha1(f"{subject}|{entry}|{len(store_terms)}".encode("utf-8")).hexdigest()[:10],
            "subject": subject,
            "entry_url": entry,
            "terminal_urls": store_terms,
            "terminal_count": int(len(store_terms)),
            "centroid_embedding": cent,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        }

        pending_indexes[entry] = {
            "entry_url": entry,
            "terminal_set": set(store_terms),
            "idx_obj": idx_obj,
            "saved_from_bucket": 0,
        }

        dbg(f"[IDX][PENDING] ★ entry={_short_url(entry)} terminals={len(store_terms)} sample={min(len(store_terms), terminal_index_centroid_sample)}")

    def _promote_index_if_hit(page_url: str):
        """
        If we saved an image from a terminal page that belongs to a pending index bucket:
        promote + persist immediately.
        """
        pu = clean_url(page_url)
        if not pu:
            return

        for entry, obj in list(pending_indexes.items()):
            if pu in obj["terminal_set"]:
                obj["saved_from_bucket"] += 1

                dbg(f"[IDX][BUCKET_HIT] ✅ entry={_short_url(entry)} saved_from_bucket={obj['saved_from_bucket']} (page={_short_url(pu)})")

                # Promote on first real success
                idx_obj = obj["idx_obj"]

                dbg(f"[IDX][PROMOTE] ✅ entry={_short_url(entry)} terminals={idx_obj.get('terminal_count')} -> WRITE")

                stored_entries.add(entry)
                pending_indexes.pop(entry, None)

                if terminal_indexes is not None:
                    terminal_indexes.append(idx_obj)

                if on_terminal_index_promoted:
                    try:
                        on_terminal_index_promoted(idx_obj)
                    except Exception as e:
                        dbg(f"[IDX][WRITE][ERR] {e}")
                return

    # -------------------------
    # RENDER
    # -------------------------
    def render(url: str, *, expand: bool, force_js_light: bool = False):
        if not force_js_light and url in html_cache:
            soup = BeautifulSoup(html_cache[url], "html.parser") if html_cache[url] else None
            a = anchors_cache.get(url, [])
            return soup, a, False

        mode = "full" if expand else ("light" if force_js_light else "smart-light")
        html, anchors, via = js_capable_fetch(url, js_mode=mode, pw=pw)

        if not html and not anchors:
            if not force_js_light:
                html_cache[url] = None
                anchors_cache[url] = []
            return None, [], False

        full = ordered_dedupe([urljoin(url, a) for a in anchors if a])
        html_cache[url] = html
        anchors_cache[url] = full
        dbg(f"[S2] via={via:<16} url={_short_url(url)} anchors={len(full)} expand={expand} force_js_light={force_js_light}")
        soup = BeautifulSoup(html, "html.parser") if html else None
        return soup, full, True

    # -------------------------
    # IMAGE SAVE WRAPPER
    # -------------------------
    def _save_one(candidate, tsoup, page_url, hit_meta: Optional[dict] = None) -> bool:
        dest = save_image_candidate(
            session=session,
            candidate=candidate,
            soup_for_inline=tsoup,
            page_url=page_url,
            dest_dir=os.path.join(IMAGES_PATH, "domain_search"),
            referer=page_url,
            dedupe_set=globals().setdefault("_saved_img_urls", set()),
        )
        if not dest:
            return False

        meta = {
            "source_kind": "domain_search",
            "page_url": page_url,
            "base_context": base_query,
        }
        if isinstance(candidate, str):
            meta["image_url"] = candidate
        if prompt_embedding is not None:
            meta["prompt_embedding"] = prompt_embedding
        if hit_meta:
            meta.update(hit_meta)

        conf = meta.get("ctx_sem_score", meta.get("ctx_score"))
        if conf is not None:
            meta["ctx_confidence"] = conf

        _register_image_metadata(dest, meta)
        saved_paths.append(dest)

        dbg(f"🖼️  [SAVE] {os.path.basename(dest)}  conf={meta.get('ctx_confidence','n/a')}  from={_short_url(page_url)}")

        # THIS IS THE KEY: promotion happens only after REAL SAVE
        _promote_index_if_hit(page_url)
        return True

    def _save_from_hits(page_url: str, tsoup, ranked_hits: list, *, per_root_counter: list[int]):
        for node, score, details in ranked_hits:
            if _global_cap_hit():
                raise _StopAll()

            ctx_score = details.get("ctx_score", score)
            ctx_sem = details.get("ctx_sem_score", details.get("sem_score"))
            hit_meta = _normalize_ctx_meta({
                "ctx_text": details.get("ctx_text") or details.get("snippet") or details.get("heading_text"),
                "ctx_score": ctx_score,
                "ctx_sem_score": ctx_sem,
                "ctx_confidence": ctx_sem if ctx_sem is not None else ctx_score,
            })

            cands = _extract_img_candidates_near_hit(node, tsoup, page_url, per_hit_cap=per_hit_cap) or []
            if not cands:
                continue

            for img_url, why in cands:
                if _global_cap_hit():
                    raise _StopAll()
                if _save_one(img_url, tsoup, page_url, hit_meta=hit_meta):
                    per_root_counter[0] += 1
                    dbg(f"·--[COUNT] {per_root_counter[0]}/{hard_image_cap}  why={why}")
                    if per_root_counter[0] >= hard_image_cap:
                        raise _NextRoot()

    def _process_terminal_url(tu: str, *, per_root_counter: list[int], fatigue_counter: list[int], idx: int, total: int):
        before = per_root_counter[0]

        tsoup, _, _ = render(tu, expand=False)
        if not tsoup:
            if per_root_counter[0] > 0:
                fatigue_counter[0] += 1
            dbg(f"[TERM {idx:02d}/{total:02d}] ❌ {_short_url(tu)}  soup=None  fatigue={fatigue_counter[0]}")
            return

        ranked = smart_find_hits_in_soup(
            tsoup,
            base_query,
            lemma_obj,
            min_score=0.7,
            top_k=None,
            encoder=encoder,
            query_embedding=query_embedding,
            sem_min_score=0.5,
            return_embedding=True,
        )

        if not ranked:
            if per_root_counter[0] > 0:
                fatigue_counter[0] += 1
            dbg(f"[TERM {idx:02d}/{total:02d}] ❌ {_short_url(tu)}  hits=0  fatigue={fatigue_counter[0]}")
            return

        dbg(f"[TERM {idx:02d}/{total:02d}] ✅ {_short_url(tu)}  hits={len(ranked)}  imgs={per_root_counter[0]}")

        any_cand = False
        for n, _, _ in ranked[:3]:
            if _extract_img_candidates_near_hit(n, tsoup, tu, per_hit_cap=1):
                any_cand = True
                break
        if not any_cand:
            tsoup, _, _ = render(tu, expand=False, force_js_light=True)

        _save_from_hits(tu, tsoup, ranked_hits=ranked, per_root_counter=per_root_counter)

        if per_root_counter[0] > before:
            fatigue_counter[0] = 0
        elif per_root_counter[0] > 0:
            fatigue_counter[0] += 1

        if per_root_counter[0] > 0 and fatigue_counter[0] >= fatigue_limit:
            dbg(f"[S2] FATIGUE LIMIT ({fatigue_limit}) after first success → next root")
            raise _NextRoot()

    # -------------------------
    # 1) USE BEST STORED INDEX FIRST (if available)
    # -------------------------
    chosen_index = None
    chosen_sim = 0.0
    if terminal_indexes:
        chosen_index, chosen_sim = pick_best_terminal_index_for_prompt(
            terminal_indexes, prompt_vec, min_sim=terminal_index_use_threshold
        )
        if chosen_index:
            dbg(f"[IDX][USE] ⭐ sim={chosen_sim:.3f} entry={_short_url(chosen_index.get('entry_url'))} terminals={len(chosen_index.get('terminal_urls') or [])}")

    # -------------------------
    # MAIN LOOP
    # -------------------------
    seen_urls: set[str] = set()

    try:
        for rdx, root in enumerate(roots_list, start=1):
            if _global_cap_hit():
                raise _StopAll()

            dbg(f"\n[S2] ROOT[{rdx}/{len(roots_list)}] {root}")

            per_root_saved = [0]
            fatigue_counter = [0]

            try:
                # if we have a chosen index and it's same domain, run its terminals first
                if chosen_index:
                    terms = chosen_index.get("terminal_urls") or chosen_index.get("terminals") or []
                    terms = [clean_url(u) for u in terms if u]
                    terms = [u for u in terms if _allowed(root, u)]
                    terms = _score_and_sort_urls(terms, k=max(75, url_sort_k))

                    dbg(f"[IDX][RUN] ⭐ processing terminals={len(terms)} from entry={_short_url(chosen_index.get('entry_url'))}")

                    for i, tu in enumerate(terms, start=1):
                        if _global_cap_hit():
                            raise _StopAll()
                        if per_root_saved[0] >= hard_image_cap:
                            raise _NextRoot()
                        if tu in seen_urls:
                            continue
                        seen_urls.add(tu)
                        _process_terminal_url(tu, per_root_counter=per_root_saved, fatigue_counter=fatigue_counter, idx=i, total=len(terms))

                    # disable after use so we don't keep spamming
                    chosen_index = None

                # normal drill
                soup0, anchors0, _ = render(root, expand=True)
                if not anchors0:
                    dbg(f"[S2] ROOT DONE {_short_url(root)} (no anchors)")
                    continue

                term0_full, deeper0_full = partition_anchors(root, anchors0, lambda u: _allowed(root, u))
                dbg(f"[S2] root partition: terminal={len(term0_full)} deeper={len(deeper0_full)}")

                # create pending index (NO gating, NO threshold)
                _create_pending_index(root, term0_full)

                term0 = _score_and_sort_urls(term0_full, k=url_sort_k)
                deeper0 = _score_and_sort_urls(deeper0_full, k=url_sort_k)

                for i, tu in enumerate(term0, start=1):
                    if _global_cap_hit():
                        raise _StopAll()
                    if per_root_saved[0] >= hard_image_cap:
                        raise _NextRoot()
                    if tu in seen_urls:
                        continue
                    seen_urls.add(tu)
                    _process_terminal_url(tu, per_root_counter=per_root_saved, fatigue_counter=fatigue_counter, idx=i, total=len(term0))

                queue = deque(deeper0)
                while queue:
                    if _global_cap_hit():
                        raise _StopAll()
                    if per_root_saved[0] >= hard_image_cap:
                        raise _NextRoot()

                    cur = queue.popleft()
                    if cur in seen_urls:
                        continue
                    seen_urls.add(cur)

                    csoup, canchors, _ = render(cur, expand=True)
                    if not canchors:
                        if per_root_saved[0] > 0:
                            fatigue_counter[0] += 1
                            if fatigue_counter[0] >= fatigue_limit:
                                raise _NextRoot()
                        continue

                    term_full, deeper_full = partition_anchors(cur, canchors, lambda u: _allowed(root, u))
                    dbg(f"[S2] cur={_short_url(cur)} term={len(term_full)} deeper={len(deeper_full)} q={len(queue)} fatigue={fatigue_counter[0]}")

                    _create_pending_index(cur, term_full)

                    term = _score_and_sort_urls(term_full, k=url_sort_k)
                    deeper = _score_and_sort_urls(deeper_full, k=url_sort_k)

                    for i, tu in enumerate(term, start=1):
                        if _global_cap_hit():
                            raise _StopAll()
                        if per_root_saved[0] >= hard_image_cap:
                            raise _NextRoot()
                        if tu in seen_urls:
                            continue
                        seen_urls.add(tu)
                        _process_terminal_url(tu, per_root_counter=per_root_saved, fatigue_counter=fatigue_counter, idx=i, total=len(term))

                    for du in deeper:
                        if du not in seen_urls:
                            queue.append(du)

            except _NextRoot:
                dbg(f"[S2] NEXT ROOT (saved {per_root_saved[0]} for {_short_url(root)})")

            dbg(f"[S2] ROOT DONE {_short_url(root)} images_for_root={per_root_saved[0]} total_images={len(saved_paths)}")

    except _StopAll:
        dbg(f"[S2] GLOBAL STOP total_images={len(saved_paths)} (global_cap={global_image_cap})")

    finally:
        try:
            pw.close()
        except Exception:
            pass

    return saved_paths











def find_valid_roots(
    base_url: str,
    lemma_obj: dict,
    max_pages_stage1: int = 100000,
    max_pages_stage2_per_root: int = 220,
    timeout: int = 8,
    known_roots: list[str] | None = None,
    js_mode: str = "light",
):
    """
    Modified Stage 1: BFS discovery using MiniLM semantic scoring.
    Keeps only URLs semantically relevant to the query or 'menu' keyword.
    Reorders kept URLs by prompt similarity.

    NEW BEHAVIOR (your request):
      1) Run normal BFS until we hit at least one high-prompt ROOT candidate in a layer.
      2) Pick the BEST ROOT candidate (highest prompt score) as the "accepted candidate".
      3) From that accepted candidate, run a "high-sim tree crawl":
         - expand ONLY through URLs that pass threshold_prompt_root
         - stop as soon as a layer produces ZERO URLs that pass threshold_prompt_root
      4) Return FINAL roots as the deepest prompt_root URLs found under the accepted candidate.
         If none found under it, return the accepted candidate itself.
      5) Values contain a tiny parent/depth structure for later Stage 2 integration.
    """
    import re
    import numpy as np
    from collections import deque
    from urllib.parse import urlparse, urlunparse, unquote

    # ----------------------------
    # Helpers
    # ----------------------------
    def clean_url(u: str) -> str:
        if not u:
            return ""
        u = unquote(u)
        if any(u.lower().startswith(s) for s in ("mailto:", "javascript:", "data:")):
            return ""
        u = re.sub(r"#.*$", "", u)
        u = re.sub(r"[?&](utm_[^=&]+|fbclid|gclid)=[^&]*", "", u)
        p = urlparse(u)
        scheme = p.scheme or "https"
        host = (p.hostname or "").lower()
        path = (p.path or "/").rstrip("/")
        if not path:
            path = "/"
        return urlunparse((scheme, host, path, "", "", ""))

    def _last_path_part(u: str) -> str:
        p = urlparse(u)
        segs = [s for s in (p.path or "").split("/") if s]
        return segs[-1] if segs else ""

    def _normalize_for_embed(s: str) -> str:
        s = (s or "").lower()
        s = s.replace("-", " ").replace("_", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def prompt_key(u: str) -> str:
        """
        PROMPT embedding input:
        - ONLY the final path segment
        - slug-normalized into words
        - NO domain
        - NO parent structure
        """
        return _normalize_for_embed(_last_path_part(u))

    def menu_key(u: str) -> str:
        """
        MENU embedding input:
        - ALSO only final segment
        - same normalization
        - menu similarity is handled by thresholding, not structure depth
        """
        return _normalize_for_embed(_last_path_part(u))

    def same_reg(u: str, reg: str) -> bool:
        import tldextract
        t = tldextract.extract(urlparse(u).hostname or "")
        r = ".".join([t.domain, t.suffix]) if t.suffix else t.domain
        return r == reg

    def prefer_shortest_by_bucket(urls):
        seen = {}
        order = []
        for u in urls:
            p = urlparse(u)
            key = (p.hostname, p.path.split("/")[-1])
            segs = len([s for s in p.path.split("/") if s])
            if key not in seen:
                seen[key] = (u, segs)
                order.append(key)
            else:
                if segs < seen[key][1]:
                    seen[key] = (u, segs)
        return [seen[k][0] for k in order]

    def _lemma_phrases_from_obj(lo: dict) -> list[str]:
        out = []
        seen = set()
        for entry in (lo or {}).values():
            for w in (entry.get("lemmas") or []):
                if not isinstance(w, str):
                    continue
                s = w.strip()
                if not s:
                    continue
                k = s.lower()
                if k in seen:
                    continue
                seen.add(k)
                out.append(s)
        return out

    def _embed_centroid(texts: list[str]):
        vecs = encoder.encode(texts, normalize_embeddings=True)
        v = np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
        n = float(np.linalg.norm(v))
        return (v / n) if n > 0 else v

    def _dot(a, b) -> float:
        return float(np.dot(a, b))

    # ----------------------------
    # Initialize embeddings
    # ----------------------------
    encoder = _EMBED_MODEL
    if encoder is None:
        print("[S1] ⚠ MiniLM encoder unavailable — returning empty roots")
        return {}

    lemma_phrases = _lemma_phrases_from_obj(lemma_obj)
    if not lemma_phrases:
        inferred = prompt_key(clean_url(base_url))
        lemma_phrases = [inferred or "query"]

    menu_keyword = "site navigation menu toc section list"
    prompt_emb = _embed_centroid(lemma_phrases)
    menu_emb = encoder.encode(menu_keyword, normalize_embeddings=True)

    # ----------------------------
    # Thresholds (tune here)
    # ----------------------------
    threshold_prompt_expand = 0.2    # keep in BFS expansion if prompt OR menu passes
    threshold_menu_expand = 0.2
    threshold_prompt_root = 0.5     # HIGH prompt threshold for "true" roots

    # ----------------------------
    # Setup BFS
    # ----------------------------
    base_clean = clean_url(base_url)
    if not base_clean:
        print("[S1] Invalid base URL.")
        return {}

    host = (urlparse(base_clean).hostname or "").lower()
    import tldextract
    t = tldextract.extract(host)
    base_reg = ".".join([t.domain, t.suffix]) if t.suffix else t.domain

    # known_roots passthrough
    if known_roots:
        roots0 = [clean_url(r) for r in known_roots if r]
        roots0 = [r for r in roots0 if r and same_reg(r, base_reg) and not any(r.lower().endswith(ext) for ext in NON_HTML_EXT)]
        roots0 = prefer_shortest_by_bucket(roots0)
        print(f"[S1] known_roots supplied; skipping discovery. roots={len(roots0)}")
        return {r: {"parent": None, "depth": 0, "prompt": None, "menu": None} for r in roots0}

    visited = {base_clean}
    frontier = deque([base_clean])
    layer_idx = 0
    total_fetches = 0

    print(f"[S1] Starting discovery on {base_clean} (domain={base_reg})")
    print(f"[S1][EMBED] prompt_phrases={len(lemma_phrases)} example={lemma_phrases[:6]} menu='{menu_keyword}'")
    print(f"[S1][THRESH] expand: prompt>={threshold_prompt_expand} OR menu>={threshold_menu_expand} | root: prompt>={threshold_prompt_root}")

    # we will pick the BEST root candidate from the first layer that produces any
    accepted_root = None
    accepted_root_scores = None  # (prompt, menu, combo, key)

    # ----------------------------
    # Phase 1: normal BFS until first layer with any ROOT candidates
    # ----------------------------
    while frontier and layer_idx < 10 and total_fetches < max_pages_stage1:
        layer_idx += 1
        layer = list(frontier)
        frontier.clear()

        print(f"\n[S1][LAYER {layer_idx}] Frontier={len(layer)} Visited={len(visited)}")

        # ---- Step 1: Scan layer
        candidates = []
        for u in layer:
            if not same_reg(u, base_reg):
                continue
            if any(u.lower().endswith(ext) for ext in NON_HTML_EXT):
                continue
            candidates.append(u)

        if not candidates:
            print("[S1] no candidates in this layer")
            continue

        # ---- Step 2: Embed candidates and rank
        prompt_texts = [prompt_key(u) for u in candidates]
        menu_texts = [menu_key(u) for u in candidates]

        prompt_url_embs = encoder.encode(prompt_texts, normalize_embeddings=True)
        menu_url_embs = encoder.encode(menu_texts, normalize_embeddings=True)

        prompt_scores = [_dot(prompt_emb, e) for e in prompt_url_embs]
        menu_scores = [_dot(menu_emb, e) for e in menu_url_embs]
        combined = [0.8 * p + 0.2 * m for p, m in zip(prompt_scores, menu_scores)]

        kept_expand = []   # (u, ps, ms, cs, key)
        kept_roots = []    # prompt-only high
        dropped = []

        for u, k, ps, ms, cs in zip(candidates, prompt_texts, prompt_scores, menu_scores, combined):
            if ps >= threshold_prompt_expand or ms >= threshold_menu_expand:
                kept_expand.append((u, ps, ms, cs, k))
                if ps >= threshold_prompt_root:
                    kept_roots.append((u, ps, ms, cs, k))
            else:
                dropped.append((u, ps, ms, cs, k))

        kept_expand_sorted = sorted(kept_expand, key=lambda x: x[1], reverse=True)
        kept_roots_sorted = sorted(kept_roots, key=lambda x: x[1], reverse=True)
        dropped_sorted = sorted(dropped, key=lambda x: x[1], reverse=True)

        # Debug (layer-based)
        print(f"[S1][LAYER {layer_idx}][STATS] kept_expand={len(kept_expand)} kept_roots={len(kept_roots)} dropped={len(dropped)}")
        if kept_expand_sorted:
            print("[S1][LAYER RANK] Kept-for-expand URLs (prompt-sorted):")
            for u, ps, ms, cs, k in kept_expand_sorted[:15]:
                mark = "ROOT" if ps >= threshold_prompt_root else "keep"
                print(f"  + [{mark}] {u}\n    key='{k}' prompt={ps:.3f} menu={ms:.3f} combo={cs:.3f}")
        else:
            print("[S1][LAYER RANK] Kept-for-expand URLs: (none)")

        if dropped_sorted:
            print("[S1][LAYER DROP] Dropped URLs (top 20 by prompt):")
            for u, ps, ms, cs, k in dropped_sorted[:20]:
                print(f"  - {u}\n    key='{k}' prompt={ps:.3f} menu={ms:.3f} combo={cs:.3f}")

        # ---- If this layer contains any ROOT candidates, pick the BEST and stop phase 1.
        if kept_roots_sorted:
            best = kept_roots_sorted[0]
            accepted_root = best[0]
            accepted_root_scores = (best[1], best[2], best[3], best[4])
            print(f"[S1][LAYER {layer_idx}] Accepted BEST ROOT → {accepted_root}")
            break

        # ---- Step 3: Expand if no roots yet
        expand_from = [u for (u, _, _, _, _) in kept_expand_sorted] if kept_expand_sorted else list(candidates)

        next_frontier = []
        for u in expand_from:
            try:
                html, anchors, via = js_capable_fetch(u, js_mode=js_mode)
            except Exception:
                anchors = []
            total_fetches += 1
            if not anchors:
                continue
            for a in anchors:
                cu = clean_url(a)
                if not cu or cu in visited:
                    continue
                if not same_reg(cu, base_reg):
                    continue
                if any(cu.lower().endswith(ext) for ext in NON_HTML_EXT):
                    continue
                visited.add(cu)
                next_frontier.append(cu)

        # reorder next frontier by prompt for faster convergence
        if next_frontier:
            nf_keys = [prompt_key(u) for u in next_frontier]
            nf_embs = encoder.encode(nf_keys, normalize_embeddings=True)
            nf_scores = [_dot(prompt_emb, e) for e in nf_embs]
            next_frontier = [u for _, u in sorted(zip(nf_scores, next_frontier), reverse=True)]

        for n in prefer_shortest_by_bucket(next_frontier):
            frontier.append(n)

        print(f"[S1][LAYER {layer_idx}] Expanded → next frontier size={len(frontier)}")

    if not accepted_root:
        print("\n[S1] Discovery complete. Found 0 candidate roots.")
        return {}

    # ----------------------------
    # Phase 2: "High-sim tree" crawl under accepted_root
    # Expand ONLY through nodes that pass threshold_prompt_root.
    # Stop when a layer yields zero passers.
    # ----------------------------
    print("\n[S1][TREE] Starting high-sim crawl under accepted_root")
    print(f"[S1][TREE] seed={accepted_root}")

    tree_nodes = {}  # url -> meta
    seed_ps, seed_ms, seed_cs, seed_key = accepted_root_scores
    tree_nodes[accepted_root] = {
        "parent": None,
        "depth": 0,
        "prompt": float(seed_ps),
        "menu": float(seed_ms),
        "combo": float(seed_cs),
        "key": seed_key,
    }

    tree_frontier = [accepted_root]
    tree_visited = set([accepted_root])
    tree_depth = 0
    max_tree_depth = 8
    max_tree_nodes = 120

    while tree_frontier and tree_depth < max_tree_depth and len(tree_nodes) < max_tree_nodes:
        tree_depth += 1

        # collect candidate children from this frontier
        child_candidates = []
        parent_for_child = {}  # first parent wins

        for parent_url in tree_frontier:
            try:
                html, anchors, via = js_capable_fetch(parent_url, js_mode=js_mode)
            except Exception:
                anchors = []

            if not anchors:
                continue

            for a in anchors:
                cu = clean_url(a)
                if not cu or cu in tree_visited:
                    continue
                if not same_reg(cu, base_reg):
                    continue
                if any(cu.lower().endswith(ext) for ext in NON_HTML_EXT):
                    continue

                tree_visited.add(cu)
                child_candidates.append(cu)
                if cu not in parent_for_child:
                    parent_for_child[cu] = parent_url

        child_candidates = prefer_shortest_by_bucket(child_candidates)

        if not child_candidates:
            print(f"[S1][TREE][DEPTH {tree_depth}] candidates=0 → STOP")
            break

        # score children
        p_texts = [prompt_key(u) for u in child_candidates]
        m_texts = [menu_key(u) for u in child_candidates]

        p_embs = encoder.encode(p_texts, normalize_embeddings=True)
        m_embs = encoder.encode(m_texts, normalize_embeddings=True)

        p_scores = [_dot(prompt_emb, e) for e in p_embs]
        m_scores = [_dot(menu_emb, e) for e in m_embs]
        c_scores = [0.8 * p + 0.2 * m for p, m in zip(p_scores, m_scores)]

        pass_high = []
        for u, k, ps, ms, cs in zip(child_candidates, p_texts, p_scores, m_scores, c_scores):
            if ps >= threshold_prompt_root:
                pass_high.append((u, ps, ms, cs, k))

        pass_high_sorted = sorted(pass_high, key=lambda x: x[1], reverse=True)

        print(f"[S1][TREE][DEPTH {tree_depth}] scanned={len(child_candidates)} pass_high={len(pass_high_sorted)} (prompt>={threshold_prompt_root})")

        # full detail but not spammy: show top 20 passers + top 20 failures by prompt
        if pass_high_sorted:
            print("[S1][TREE][PASS] top passers:")
            for u, ps, ms, cs, k in pass_high_sorted[:20]:
                print(f"  + {u}\n    key='{k}' prompt={ps:.3f} menu={ms:.3f} combo={cs:.3f}")

        if not pass_high_sorted:
            print(f"[S1][TREE][DEPTH {tree_depth}] no high-sim nodes in this layer → STOP")
            break

        # record nodes + build next frontier (ONLY passers)
        next_frontier = []
        for u, ps, ms, cs, k in pass_high_sorted:
            if u in tree_nodes:
                continue
            tree_nodes[u] = {
                "parent": parent_for_child.get(u),
                "depth": tree_depth,
                "prompt": float(ps),
                "menu": float(ms),
                "combo": float(cs),
                "key": k,
            }
            next_frontier.append(u)

        tree_frontier = next_frontier

    # ----------------------------
    # Final root set selection:
    # - if we found any depth>=1 high nodes, return those (NOT the seed)
    # - else return the seed
    # ----------------------------
    depth1_plus = [u for u, meta in tree_nodes.items() if meta.get("depth", 0) >= 1]
    depth1_plus = prefer_shortest_by_bucket(depth1_plus)

    if depth1_plus:
        print(f"\n[S1] Final roots: returning depth>=1 high-sim nodes count={len(depth1_plus)} (excluding seed)")
        out = {u: tree_nodes[u] for u in depth1_plus if u in tree_nodes}
    else:
        print("\n[S1] Final roots: no high-sim descendants found; returning seed only")
        out = {accepted_root: tree_nodes[accepted_root]}

    # Summary
    keys = list(out.keys())
    print(f"[S1] Discovery complete. Found {len(keys)} candidate roots:")
    for r in keys[:12]:
        m = out[r]
        print(f"  → {r}  (depth={m.get('depth')} prompt={m.get('prompt'):.3f})")

    return out







# =========================
# Sources / API (kept)
# =========================
class Source():
    def __init__(self, json_obj, name):
        self.name = name
        self.url = json_obj.get("url")
        if json_obj.get("HasApi") == "Y":
            self.TemplateEQ = json_obj.get("TemplateEquivalent")
            self.Template = json_obj.get("FullTemplate")
            self.type = "API"
        else:
            self.type = "NORMAL"
            self.subjecturls = json_obj.get("SubjectUrls")
            self.subjecturlsfound = json_obj.get("SubjectUrlsFound")
        self.img_paths = None

def read_sources():
    sources = []
    for filename in os.listdir(SOURCE_PATH):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(SOURCE_PATH, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            sources.append(Source(data, Path(filename).stem))
    dbg(f"[MAIN] sources loaded={len(sources)}")
    return sources

#tried to make it flexible and have one builder for all sources, but it was too wierd tro sync sop resort to hardcode for now
def build_params_from_settings(source: Source, settings: dict) -> dict:
    params = copy.deepcopy(source.Template)

    # generic mapping
    for k, v in settings.items():
        if k == "pagination_field":
            api_field = source.TemplateEQ.get(k)
            if api_field and v and isinstance(v, str) and v.strip() and v.lower() != "continue":
                params[api_field] = v
            continue
        api_field = source.TemplateEQ.get(k)
        if api_field:
            params[api_field] = v

    fmt_field = source.TemplateEQ.get("format_field")
    fmt_value = settings.get("format_value") or settings.get("format_field")
    if fmt_field and fmt_value:
        params[fmt_field] = fmt_value

    name = (source.name or "").lower()

    if name == "wikimedia":
        params.setdefault("action", "query")
        params.setdefault("generator", "search")
        params.setdefault("prop", "imageinfo|info")
        params.setdefault("iiprop", "url|mime|extmetadata|size|sha1")
        params["gsrnamespace"] = 6
        q = params.get("gsrsearch", "") or ""
        if "filetype:" not in q.lower():
            params["gsrsearch"] = q + " filetype:bitmap|drawing|svg"
        try:
            params["gsrlimit"] = min(int(params.get("gsrlimit", 5)), 10)
        except Exception:
            params["gsrlimit"] = 5

    elif name == "openverse":
        try:
            params["page_size"] = min(int(params.get("page_size", 5)), 10)
        except Exception:
            params["page_size"] = 5
        params.setdefault("page", 1)

    elif name == "usgs":
        # ScienceBase won’t include files/attachments unless you ask
        params.setdefault("fields", "files,attachments,link,linkUrl,id")
        try:
            params["max"] = min(int(params.get("max", 5)), 10)
        except Exception:
            params["max"] = 5
        params.setdefault("offset", 0)
        params.setdefault("format", "json")

    elif name == "plos":
        try:
            params["rows"] = min(int(params.get("rows", 5)), 10)
        except Exception:
            params["rows"] = 5
        params["wt"] = "json"

    return params


def send_request(source: Source, settings: dict):
    built = build_params_from_settings(source, settings)
    if source.type != "API":
        return None, None, built
    headers = {"User-Agent": "diag-scrape/0.1"}
    try:
        dbg(f"[API][{source.name}] GET {source.url} params={built}")
        resp = requests.get(source.url, params=built, headers=headers, timeout=20)
        ct = resp.headers.get("Content-Type", "")
        dbg(f"[API][{source.name}] status={resp.status_code} len={len(resp.content)} ct={ct}")
        try:
            data = resp.json()
        except ValueError:
            data = resp.text
        return resp.status_code, data, built
    except requests.RequestException as e:
        dbg(f"[API][{source.name}] REQUEST_ERROR: {e}")
        return None, f"REQUEST_ERROR: {e}", built



PARSERS = {
    "wikimedia": parse_wikimedia,
    "openverse": parse_openverse,
    "plos": parse_plos,
    "usgs": parse_usgs
}

# =========================
# HANDLE NON-API (NEW PIPELINE)
# =========================
# =========================
# MODIFIED: handle_result_no_api (reads/uses/saves terminal indexes)
# =========================
def handle_result_no_api(source: Source, query: str, subj: str, hard_image_cap: int = 5,
                         encoder: Optional[Any] = None,
                         query_embedding: Optional[Any] = None,
                         base_ctx_embedding: Optional[Any] = None):
    def uniq_keep_order(items):
        seen = set()
        out = []
        for x in items:
            if x and x not in seen:
                seen.add(x); out.append(x)
        return out

    lemma_obj_subj = get_limited_lemmas(subj, 4)
    lemma_obj_query = get_limited_lemmas(query, 5)

    valid_endpoints: List[str] = []

    if getattr(source, "subjecturls", None):
        dbg("[NOAPI] subject urls exist")
        for current_subj, url_or_list in source.subjecturls.items():
            if current_subj != subj or not url_or_list:
                continue
            if isinstance(url_or_list, (list, tuple, set)):
                valid_endpoints.extend(url_or_list)
            else:
                valid_endpoints.append(url_or_list)

    if getattr(source, "subjecturlsfound", None):
        dbg("[NOAPI] previously found urls loaded")
        for current_subj, url_or_list in source.subjecturlsfound.items():
            if current_subj != subj or not url_or_list:
                continue
            if isinstance(url_or_list, (list, tuple, set)):
                valid_endpoints.extend(url_or_list)
            else:
                valid_endpoints.append(url_or_list)

    valid_endpoints = uniq_keep_order(clean_url(u) for u in valid_endpoints if u)

    # Stage 1 discovery (or normalize known)
    if valid_endpoints:
        dbg(f"[NOAPI] using known endpoints count={len(valid_endpoints)} for subj={subj}")
        roots = find_valid_roots(base_url=source.url, lemma_obj=lemma_obj_subj, known_roots=valid_endpoints)
    else:
        dbg("[NOAPI] no valid endpoints; starting Stage 1 discovery")
        roots = find_valid_roots(base_url=source.url, lemma_obj=lemma_obj_subj,)

    if not roots:
        dbg(f"[NOAPI] no roots after Stage 1 for {source.name}")
        source.img_paths = []
        return []

    # Persist roots we used (so next run can reuse)
    try:
        if roots:
            json_path = os.path.join(SOURCE_PATH, f"{source.name}.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {}
            data.setdefault("SubjectUrlsFound", {})
            data["SubjectUrlsFound"].setdefault(subj, [])
            prev = set(map(clean_url, data["SubjectUrlsFound"][subj]))
            new_roots = [r for r in roots if r and r not in prev]
            if new_roots:
                data["SubjectUrlsFound"][subj].extend(new_roots)
                dedup = []
                s = set()
                for u in data["SubjectUrlsFound"][subj]:
                    if u and u not in s:
                        s.add(u); dedup.append(u)
                data["SubjectUrlsFound"][subj] = dedup

                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                tmp = json_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                try:
                    os.replace(tmp, json_path)
                except Exception:
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                dbg(f"[NOAPI] persisted {len(new_roots)} roots under SubjectUrlsFound[{subj}]")
    except Exception as e:
        dbg(f"[NOAPI] persist warning: {e}")

    # =========================
    # NEW: Load ALL terminal indexes for this subject, pick best for THIS domain
    # =========================
    terminal_seed = None
    try:
        all_idx = _load_all_terminal_indexes_for_subject(subj)
        if all_idx and base_ctx_embedding is not None and encoder is not None:
            import numpy as np

            prompt_vec = _norm_vec(base_ctx_embedding)
            my_reg = _registrable_of(source.url)

            # filter to matching registrable domain + valid centroid
            candidates = []
            for idx in all_idx:
                entry = idx.get("entry_url") or ""
                reg = _registrable_of(entry) if entry else _registrable_of(idx.get("_source_url") or "")
                if reg != my_reg:
                    continue
                cent = idx.get("centroid_embedding")
                if not isinstance(cent, list) or len(cent) < 32:
                    continue
                score = float(np.dot(prompt_vec, np.asarray(cent, dtype=np.float32)))
                candidates.append((score, idx))

            candidates.sort(key=lambda x: x[0], reverse=True)

            dbg(f"[IDX][SELECT] subj='{subj}' domain='{my_reg}' indexes_found_total={len(all_idx)} domain_matched={len(candidates)}")
            for s, idx in candidates[:10]:
                dbg(f"  ↳ score={s:.3f} terminals={int(idx.get('terminal_count') or 0):>3} entry={_short_url(idx.get('entry_url') or '')}  (src={idx.get('_source_name')})")

            if candidates:
                best_score, best = candidates[0]
                if best_score >= 0.35:  # keep consistent with Stage2 default use threshold
                    terminal_seed = {
                        "entry_url": best.get("entry_url"),
                        "terminal_urls": best.get("terminal_urls") or [],
                        "centroid_embedding": best.get("centroid_embedding"),
                        "score": float(best_score),
                        "source": best.get("_source_name"),
                        "id": best.get("id"),
                    }
                    dbg(f"[IDX][SELECT][USE] id={terminal_seed.get('id')} score={best_score:.3f} entry={_short_url(terminal_seed.get('entry_url') or '')}")
                else:
                    dbg(f"[IDX][SELECT][DROP] best_score={best_score:.3f} < 0.35 threshold")
    except Exception as e:
        dbg(f"[IDX][SELECT][ERR] {e}")

    # Stage 2
    session = requests.Session()
    session.headers.update({"User-Agent": "diag-scrape/0.1"})
    html_cache = {}
    anchors_cache = {}

    terminal_indexes_out: list[dict] = []

    dbg(f"[NOAPI] Stage 2 starting roots={len(roots)} hard_image_cap={hard_image_cap}")
    loaded_idxs = load_terminal_indexes_for_subject(source.name, subj)

    saved = stage2_drill_search_and_images(
        roots=roots,
        base_query=query,
        lemma_obj=lemma_obj_query,
        hard_image_cap=hard_image_cap,
        session=session,
        html_cache=html_cache,
        anchors_cache=anchors_cache,
        encoder=encoder,
        query_embedding=query_embedding,
        prompt_embedding=base_ctx_embedding,
        per_hit_cap=1,

        url_sort_k=75,

        terminal_indexes=loaded_idxs,
        terminal_index_min_terminals=25,
        terminal_index_use_threshold=0.22,
        terminal_index_store_cap=600,
        terminal_index_centroid_sample=60,

        subject=subj,
        source_name=source.name,
        on_terminal_index_promoted=lambda idx: persist_terminal_index_for_subject(source.name, subj, idx),
    )


    source.img_paths = saved

    # NEW: Persist terminal indexes discovered during Stage2 into THIS source json
    if terminal_indexes_out:
        dbg(f"[IDX][RUN] produced_indexes={len(terminal_indexes_out)} (will persist)")
        _persist_terminal_indexes_to_source(source.name, subj, terminal_indexes_out)

    return saved



def _download_to(u: str, dest_dir: str, idx: int, dbg=print) -> str | None:
    """
    Downloads an image from URL `u` into `dest_dir` with a safe filename.
    Returns the absolute path if saved, or None if skipped or failed.
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

        # Skip if already downloaded
        if os.path.exists(path):
            dbg(f"[IMG][skip] already exists {path}")
            return path

        headers = {"User-Agent": "diag-scrape/0.1"}
        with requests.get(u, headers=headers, stream=True, timeout=30) as r:
            r.raise_for_status()
            ctype = (r.headers.get("Content-Type") or "").lower()
            # Filter only image-like responses
            if not (ctype.startswith("image/") or re.search(r"\.(png|jpe?g|gif|webp|svg|bmp|tiff?)($|\?)", u, re.I)):
                dbg(f"[IMG][skip] not image ct={ctype} url={u}")
                return None
            # Avoid empty/tiny assets (<2 KB)
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


def _is_blocked_host(u: str) -> bool:
    try:
        host = (urlparse(u).hostname or "").lower()
        return any(host.endswith(b) for b in _BLOCKED_HOSTS)
    except Exception:
        return True

def _cc_evidence_in_html(html: str) -> bool:
    if not html:
        return False
    for rx in _CC_RX:
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

# expects your existing helpers:
# - IMAGES_PATH (folder path)
# - dbg(*args) (debug printer)
# - _download_to(url, dest_dir, idx) -> str|None  (your downloader)

def ddg_cc_image_harvest(query: str,
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
                         base_ctx_embedding: Optional[Any] = None) -> list[str]:
    """
    Rate-limit-aware DDG image flow:
      - streams results gently
      - retries on 403 with backoff + jitter
      - downloads ONLY if the result page shows CC evidence
      - attaches per-image context from ddg_image_context (filename/alt/title + nearby text)
      - optionally attaches context embedding/semantic score
      - stops early if 20 consecutive pages have no CC evidence
    """
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

    # consecutive pages with no CC evidence
    no_cc_streak = 0
    NO_CC_STREAK_LIMIT = 20

    def _yield_ddg_images(q: str, cap: int):
        """
        Generator wrapper that handles 403/ratelimit with backoff+jitter,
        recreating the DDGS client as needed.
        """
        total = 0
        delay = backoff_base
        while total < cap:
            try:
                headers = {
                    "User-Agent": random.choice(_UAS),
                    "Accept-Language": "en-US,en;q=0.9",
                }
                with DDGS(headers=headers, proxies=proxies) as ddgs:
                    # Avoid huge grabs; stream and sleep ourselves.
                    for item in ddgs.images(q, max_results=cap - total, safesearch=safesearch, region=region):
                        yield item
                        total += 1
                        if total >= cap:
                            break
                        time.sleep(sleep_between)  # <- gentle pacing to avoid 403s
                break  # normal exit
            except Exception as e:
                msg = str(e).lower()
                if "403" in msg or "ratelimit" in msg or "too many requests" in msg:
                    # exponential backoff + jitter
                    jitter = random.uniform(0.4, 0.9)
                    wait = min(delay * (1.6 + jitter), backoff_max)
                    dbg(f"[DDG] rate-limited; backing off for {wait:.1f}s")
                    time.sleep(wait)
                    delay = min(delay * 2.0, backoff_max)
                    continue
                dbg(f"[DDG] unexpected error: {e}")
                break

    for r in _yield_ddg_images(query, ddg_cap):
        if len(saved) >= target_count:
            break

        # early stop if we've seen too many CC-miss pages in a row
        if no_cc_streak >= NO_CC_STREAK_LIMIT:
            dbg(f"[DDG] stopping: {no_cc_streak} consecutive pages without CC evidence")
            break

        page_url = r.get("url") or r.get("source") or ""
        img_url  = r.get("image") or ""
        if not page_url or not img_url:
            continue
        if _is_blocked_host(page_url) or _is_blocked_host(img_url):
            continue
        if page_url in seen_pages or img_url in seen_imgs:
            continue
        seen_pages.add(page_url); seen_imgs.add(img_url)

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

        if not _cc_evidence_in_html(html):
            dbg(f"[DDG] no CC evidence -> skip  {page_url}")
            no_cc_streak += 1
            continue

        # we *did* find CC evidence → reset streak
        no_cc_streak = 0

        # --- per-image context (filename/alt/title + nearby text) ---
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
                # normalize so ddg + domain use exactly the same ctx_* fields
                ctx_meta = _normalize_ctx_meta(ctx_meta)
            except ImportError as e:
                dbg(f"[DDG] ddg_image_context import error: {e}")
            except Exception as e:
                dbg(f"[DDG] ddg_extract_image_context error: {e}")
        except Exception as e:
            dbg(f"[DDG] context soup error: {e}")

        # ok to download the image
        p = _download_to(img_url, dest_dir, len(saved), dbg=dbg)
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
                dbg(f"[DDG][CTX] saved {p} ctx_score={meta.get('ctx_score')} "
                    f"ctx_sem={meta.get('ctx_sem_score')} conf={meta.get('ctx_confidence')}")
            else:
                dbg(f"[DDG][CTX] saved {p} ctx=n/a")

            _register_image_metadata(p, meta)
            saved.append(p)

    dbg(f"[DDG] saved ({len(saved)}/{target_count}) {p}")
    return saved




def _file_sha1(path: str, chunk=1024*1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def _ahash64(path: str, size: int = 8) -> str | None:
    """
    64-bit average hash (scale-invariant; good for spotting resized duplicates).
    Returns hex string or None if image can't be decoded (e.g., SVG).
    """
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((size, size), Image.LANCZOS)
            px = list(im.getdata())
            avg = sum(px) / len(px)
            bits = 0
            for i, p in enumerate(px):
                if p >= avg:
                    bits |= (1 << i)
            return f"{bits:016x}"
    except Exception:
        return None  # skip non-decodable rasters (SVG etc.)
    
import numpy as np

def _json_safe(obj):
    """
    Recursively convert objects into something json.dumps can handle:
    - numpy arrays -> lists
    - numpy scalars -> float/int
    - sets -> lists
    - dict/list/tuple -> walk recursively
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, set):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj

def _normalize_ctx_meta(ctx_meta: dict) -> dict:
    """
    Keep only unified context fields so DDG and domain_search share the same schema.
    Allowed keys:
      - ctx_text
      - ctx_embedding
      - ctx_score
      - ctx_sem_score
      - ctx_confidence
    """
    if not ctx_meta:
        return {}
    allowed = {"ctx_text", "ctx_embedding", "ctx_score", "ctx_sem_score", "ctx_confidence"}
    out = {}
    for k in allowed:
        if k in ctx_meta:
            out[k] = ctx_meta[k]
    return out



def collect_unique_images(images_root: str, base_ctx_embedding=None, prompt_text: str | None = None, dbg=print):
    """
    Walks `images_root`, dedupes images, ranks them with SigLIP + context,
    and keeps ONLY the top 10 in <images_root>/UniqueImages.

    Pipeline:
      1) Deduplicate by SHA1 and perceptual hash, aggregate IMAGE_METADATA per canonical path.
      2) For each image, pick the best context (highest ctx_confidence) as its representative.
      3) Build candidates and run SigLIP-based reranking (top_n=30, final_k=10).
      4) Copy ONLY the 10 winners into UniqueImages.
      5) Write:
         - image_metadata_core.json     (source/page/url/etc., no heavy vectors or scores)
         - image_metadata_context.json  (prompt embedding + SigLIP embedding + scores + per-context ctx_*)

    SigLIP outputs added to context JSON per image:
      - clip_embedding   (SigLIP image embedding)
      - clip_score       (SigLIP image score)
      - confidence_score (context centroid+prompt score)
      - final_score      (65% confidence, 35% clip_score)
    """
    unique_dir = os.path.join(images_root, "UniqueImages")
    os.makedirs(unique_dir, exist_ok=True)

    # Clear previous run outputs (files only)
    for fname in os.listdir(unique_dir):
        fpath = os.path.join(unique_dir, fname)
        if os.path.isfile(fpath):
            try:
                os.remove(fpath)
            except Exception as e:
                dbg(f"[collect][warn][cleanup] {fpath} -> {e}")

    # 1) Dedupe by SHA1 + perceptual hash, aggregate metadata by canonical path
    sha1_to_canonical: Dict[str, str] = {}
    ahash_to_canonical: Dict[str, str] = {}
    canonical_metas: Dict[str, List[dict]] = {}

    for dirpath, _, filenames in os.walk(images_root):
        # skip copying UniqueImages into itself
        if os.path.abspath(dirpath) == os.path.abspath(unique_dir):
            continue

        for fname in filenames:
            if not IMG_EXT_RE.search(fname):
                continue
            path = os.path.join(dirpath, fname)

            # 1) exact byte hash
            try:
                sha1 = _file_sha1(path)
            except Exception as e:
                dbg(f"[collect][skip][read] {path} -> {e}")
                continue

            canonical = sha1_to_canonical.get(sha1)

            # 2) perceptual hash if needed
            if canonical is None:
                ah = _ahash64(path)
                if ah is not None and ah in ahash_to_canonical:
                    canonical = ahash_to_canonical[ah]
                else:
                    canonical = path
                    sha1_to_canonical[sha1] = canonical
                    if ah is not None:
                        ahash_to_canonical[ah] = canonical

            # 3) merge metadata for this path into the canonical path
            metas = IMAGE_METADATA.get(path)
            if metas:
                lst = canonical_metas.get(canonical)
                if lst is None:
                    canonical_metas[canonical] = list(metas)
                else:
                    lst.extend(metas)

    if not canonical_metas:
        dbg("[collect] no images with metadata found; nothing to rank")
        return []

    # 2) Build ranking candidates from best context per image
    candidates: List[dict] = []
    for img_path, metas in canonical_metas.items():
        # pick prompt embedding for this image (if any meta has it), else global base_ctx_embedding
        img_prompt_emb = None
        for m in metas:
            pe = m.get("prompt_embedding") or m.get("base_ctx_embedding")
            if pe is not None:
                img_prompt_emb = pe
                break
        if img_prompt_emb is None:
            img_prompt_emb = base_ctx_embedding

        # best context for ranking
        best_ctx = None
        for m in metas:
            ctx_emb = m.get("ctx_embedding")
            if ctx_emb is None:
                continue
            ctx_conf = m.get("ctx_confidence", m.get("ctx_sem_score", m.get("ctx_score", 0.0)))
            if best_ctx is None or ctx_conf > best_ctx["ctx_confidence"]:
                best_ctx = {
                    "ctx_embedding": ctx_emb,
                    "ctx_text": m.get("ctx_text"),
                    "ctx_score": m.get("ctx_score"),
                    "ctx_sem_score": m.get("ctx_sem_score"),
                    "ctx_confidence": float(ctx_conf),
                }

        if best_ctx is None:
            # no usable context embedding → can't rank this image
            continue

        candidates.append(
            {
                "image_path": img_path,
                "ctx_embedding": best_ctx["ctx_embedding"],
                "prompt_embedding": img_prompt_emb,
                "ctx_confidence": best_ctx["ctx_confidence"],
                "ctx_score": best_ctx["ctx_score"],
                "ctx_sem_score": best_ctx["ctx_sem_score"],
                "ctx_text": best_ctx["ctx_text"] or "",
            }
        )

    if not candidates:
        dbg("[collect] no candidates with ctx_embedding; skipping SigLIP ranking")
        return []

    # 3) Run SigLIP ranking (top 30 → final 10)
    prompt_text = prompt_text or ""
    backend = SiglipBackend()
    ranked = rerank_image_candidates_siglip(
        candidates=candidates,
        prompt_text=prompt_text,
        backend=backend,
        top_n=30,
        final_k=10,
    )

    if not ranked:
        dbg("[collect] SigLIP ranking returned no winners")
        return []

    winners_by_path = {w["image_path"]: w for w in ranked}

    # 4) Copy ONLY winners to UniqueImages and build metadata
    unique_paths: List[str] = []
    core_meta: Dict[str, List[dict]] = {}
    ctx_meta: Dict[str, dict] = {}

    heavy_fields = {
        "ctx_embedding",
        "prompt_embedding",
        "base_ctx_embedding",
        "ctx_score",
        "ctx_sem_score",
        "sem_score",
        "ctx_confidence",
        "clip_embedding",
        "clip_score",
        "confidence_score",
        "final_score",
    }

    for img_path, metas in canonical_metas.items():
        winner = winners_by_path.get(img_path)
        if winner is None:
            continue  # not in top 10

        # copy image into UniqueImages (avoid name collisions)
        dest_path = os.path.join(unique_dir, os.path.basename(img_path))
        base, ext = os.path.splitext(dest_path)
        n = 1
        while os.path.exists(dest_path):
            dest_path = f"{base}_{n}{ext}"
            n += 1
        try:
            shutil.copy2(img_path, dest_path)
            unique_paths.append(dest_path)
            dbg(f"[collect][copy] {img_path} -> {dest_path}")
        except Exception as e:
            dbg(f"[collect][err][copy] {img_path} -> {e}")
            continue

        # per-image prompt embedding (reuse if present on any meta)
        img_prompt_emb = None
        for m in metas:
            pe = m.get("prompt_embedding") or m.get("base_ctx_embedding")
            if pe is not None:
                img_prompt_emb = pe
                break
        if img_prompt_emb is None:
            img_prompt_emb = base_ctx_embedding

        # core metadata + per-context blocks
        core_list: List[dict] = []
        ctx_contexts: List[dict] = []

        for m in metas:
            # core metadata = everything except heavy vector + scoring fields
            m_core = {}
            for k, v in m.items():
                if k in heavy_fields:
                    continue
                m_core[k] = v
            if m_core:
                core_list.append(m_core)

            # context block if we have any context-ish info
            has_ctx = any(
                k in m
                for k in ("ctx_embedding", "ctx_text", "ctx_score", "ctx_sem_score", "ctx_confidence")
            )
            if has_ctx:
                ctx_conf = m.get("ctx_confidence", m.get("ctx_sem_score", m.get("ctx_score")))
                ctx_entry = {
                    "source_kind": m.get("source_kind"),
                    "source_name": m.get("source_name"),
                    "page_url": m.get("page_url"),
                    "image_url": m.get("image_url"),
                    "ctx_text": m.get("ctx_text"),
                    "ctx_score": m.get("ctx_score"),
                    "ctx_sem_score": m.get("ctx_sem_score"),
                    "ctx_confidence": ctx_conf,
                    "ctx_embedding": m.get("ctx_embedding"),
                }
                ctx_contexts.append(ctx_entry)

        if core_list:
            core_meta[dest_path] = core_list

        ctx_meta[dest_path] = {
            "prompt_embedding": img_prompt_emb,
            "clip_embedding": winner.get("clip_embedding"),
            "clip_score": winner.get("clip_score"),
            "confidence_score": winner.get("confidence_score"),
            "final_score": winner.get("final_score"),
            "contexts": ctx_contexts,
        }

    # 5) write metadata JSONs (SigLIP vectors & scores go into the context JSON)
    try:
        core_path = os.path.join(unique_dir, "image_metadata_core.json")
        with open(core_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(core_meta), f, indent=2, ensure_ascii=False)
        dbg(f"[collect] core metadata saved to {core_path}")

        ctx_path = os.path.join(unique_dir, "image_metadata_context.json")
        with open(ctx_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(ctx_meta), f, indent=2, ensure_ascii=False)
        dbg(f"[collect] context metadata saved to {ctx_path}")
    except Exception as e:
        dbg(f"[collect][err][meta] {e}")

    dbg(f"[collect] winners={len(unique_paths)} saved to {unique_dir}")
    return unique_paths



def _embed_prompt_to_list(text: str):
    if _EMBED_MODEL is None:
        return None
    v = _EMBED_MODEL.encode(text)  # or however you embed text
    # make sure it’s JSON-serializable
    if hasattr(v, "tolist"):
        v = v.tolist()
    else:
        v = list(v)
    return v


# =========================
# MAIN
# =========================
def research(query: str, subj: str):

    base_ctx_embedding = _embed_prompt_to_list(query)

    settings = {
        "query_field": query,
        "limit_field": 10,
        "pagination_field": 1,
        "format_field": "json",
    }

    encoder = _EMBED_MODEL
    query_embedding = None
    if encoder is not None:
        try:
            query_embedding = encoder.encode(query)
            dbg(f"[EMBED] query embedding dim={len(query_embedding)}")
        except Exception as e:
            dbg(f"[EMBED] failed to encode query: {e}")
            query_embedding = None

    lemma_obj_query = get_limited_lemmas(query, 5)

    sources = read_sources()
    web_sources = []
    for src in sources:
        if src.type == "API":
            dbg(f"opened source {src.name} as api (parser='{src.name}'), sending request..")
            # status, data, built = send_request(src, settings)
            # dbg(f"sent request, responded {status}")
            # if status == 200 and data is not None:
            #     try:
            #         _ = PARSERS.get(src.name)
            #         if _:
            #             _. __call__  # noqa: touch
            #     except Exception:
            #         pass
            # if data is not None:
            #     parse_fn = PARSERS.get(src.name)
            #     if parse_fn:
            #         parse_fn(src, data)
        else:
            web_sources.append(src)

    # DDG: CC-backed images with per-image context
    #ddg_cc_image_harvest(
    #    query=query,
    #    target_count=20,              # higher cap so the new behaviour actually matters
    #    lemma_obj=lemma_obj_query,
    #    encoder=encoder,
    #    query_embedding=query_embedding,
    #    base_ctx_embedding=base_ctx_embedding,
    #)

    for src in web_sources:
        dbg(f"opened source {src.name}, has no api, starting process..")
        handle_result_no_api(src, query, subj, hard_image_cap=25,
                             encoder=encoder,
                             query_embedding=query_embedding,
                             base_ctx_embedding=base_ctx_embedding)

    # attach base_context + source info for any images parsed from APIs (and also reinforce for no-api)
    for src in sources:
        if not getattr(src, "img_paths", None):
            continue
        for p in src.img_paths:
            if not p:
                continue
            meta = {
                "source_kind": "api" if src.type == "API" else "noapi",
                "source_name": src.name,
                "base_context": query,
            }
            if base_ctx_embedding is not None:
                meta["prompt_embedding"] = base_ctx_embedding
            _register_image_metadata(p, meta)
    
    unique = collect_unique_images(IMAGES_PATH, base_ctx_embedding, prompt_text=query, dbg=dbg)
    dbg(f"Unique images:{unique}")
    return unique

if "__main__":
    research("Eukaryotic cell", "Biology")