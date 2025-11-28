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
    import re
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
            if "html" not in ct and "xml" not in ct:  # allow a few servers that mislabel
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
        dbg(f"[JSFETCH][requests] {url} ok anchors={len(anchors)} html_len={len(html or '')}")
        return html, anchors, "requests"

    # ---- js_mode: smart-light (requests-first; fallback to playwright:light) ----
    if js_mode == "smart-light":
        html, anchors, _ = _requests_fetch(url)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            text_len = len(soup.get_text(" ", strip=True))
            p_tags = len(soup.find_all("p"))
            # If page already looks textful enough, keep requests result
            if (text_len >= min_text_chars and p_tags >= min_p_tags) or len(anchors) >= min_anchors_hint:
                dbg(f"[JSFETCH][requests:light] {url} ok anchors={len(anchors)} html_len={len(html)} "
                    f"(text_len={text_len}, p_tags={p_tags})")
                return html, anchors, "requests:light"
            else:
                dbg(f"[JSFETCH][requests:light][fallback] {url} text_len={text_len} p_tags={p_tags} anchors={len(anchors)}")

        # Fallback to Playwright light
        try:
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

                anchors_pw = collect_links_with_modals_sync(
                    page,
                    expand=False,
                    max_modal_clicks=0
                )
                anchors_pw = [urljoin(url, a) for a in anchors_pw if a]
                html_pw = page.content()

                try:
                    ctx.close()
                except Exception:
                    pass
                browser.close()

                dbg(f"[JSFETCH][playwright:light] {url} ok anchors={len(anchors_pw)} html_len={len(html_pw)}")
                return html_pw, anchors_pw, "playwright:light"

        except ImportError:
            dbg("[JSFETCH] Playwright not installed (smart-light fallback).")
        except Exception as e:
            dbg(f"[JSFETCH][playwright][smart-light] failed for {url}: {e}")

        # If everything fails:
        return html, anchors, "requests:light" if html else (None, [], "none")

    # ---- js_mode: full/light (Playwright) ----
    expand = (js_mode == "full")
    try:
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
            anchors = [urljoin(url, a) for a in anchors if a]  # absolutize
            html = page.content()

            try:
                ctx.close()
            except Exception:
                pass
            browser.close()

            mode_tag = f"playwright:{'full' if expand else 'light'}"
            dbg(f"[JSFETCH][{mode_tag}] {url} ok anchors={len(anchors)} html_len={len(html)}")
            return html, anchors, mode_tag

    except ImportError:
        dbg("[JSFETCH] Playwright not installed.")
    except Exception as e:
        dbg(f"[JSFETCH][playwright] failed for {url}: {e}")

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



    


def _extract_img_candidates_near_hit(node, soup, page_url, *, per_hit_cap=2):
    """
    Return up to per_hit_cap best image URLs near a hit node.
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
                    if len(cand) >= per_hit_cap:
                        return cand
            img = pic.find("img")
            if img:
                url = _img_url_from_tag(img)
                if url:
                    _add(url, "picture/img")
                    if len(cand) >= per_hit_cap:
                        return cand

        # plain <img>
        for img in sc.find_all("img"):
            url = _img_url_from_tag(img)
            if url:
                _add(url, "img")
                if len(cand) >= per_hit_cap:
                    return cand

        # <a href="...img.ext">
        for a in sc.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if href and re.search(r"\.(png|jpe?g|gif|webp|svg|tiff?)($|\?)", href, re.I):
                _add(href, "a->image")
                if len(cand) >= per_hit_cap:
                    return cand

        if len(cand) >= per_hit_cap:
            break

    return cand[:per_hit_cap]
    
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
def stage2_drill_search_and_images(
    roots: list[str],
    base_query: str,
    lemma_obj: dict,
    hard_image_cap: int,
    session: requests.Session,
    html_cache: dict,
    anchors_cache: dict,
    global_image_cap: int | None = None,
    per_hit_cap: int = 2,
    fatigue_limit: int = 25,   # trigger only *after first image*, after 25 dry pages
    encoder: Optional[Any] = None,
    query_embedding: Optional[Any] = None,
) -> list[str]:
    phrase_rxs, _ = _compile_phrase_regexes_adapter(base_query, lemma_obj)

    class _NextRoot(BaseException): pass
    class _StopAll(BaseException): pass

    saved_paths: list[str] = []

    def _global_cap_hit() -> bool:
        return global_image_cap is not None and len(saved_paths) >= global_image_cap

    # === RENDER ===
    def render(url: str, *, expand: bool, force_js_light: bool = False):
        if not force_js_light and url in html_cache:
            soup = BeautifulSoup(html_cache[url], "html.parser") if html_cache[url] else None
            a = anchors_cache.get(url, [])
            dbg(f"[S2][cache] url={url} anchors={len(a)} soup_ok={bool(html_cache[url])}")
            return soup, a, False

        mode = "full" if expand else ("light" if force_js_light else "smart-light")
        html, anchors, via = js_capable_fetch(url, js_mode=mode)
        if not html and not anchors:
            if not force_js_light:
                html_cache[url] = None
                anchors_cache[url] = []
            return None, [], False

        full = ordered_dedupe([urljoin(url, a) for a in anchors if a])
        html_cache[url] = html
        anchors_cache[url] = full
        dbg(f"[S2] rendered via={via} url={url} anchors={len(full)} soup_ok={bool(html)} expand={expand} force_js_light={force_js_light}")
        soup = BeautifulSoup(html, "html.parser") if html else None
        return soup, full, True

    # === HELPERS ===
    def _allowed(root: str, child_url: str) -> bool:
        try:
            return domain_and_phrase_lock(root, child_url)
        except Exception:
            return False

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
        if dest:
            dbg(f"[IMG] saved {dest}")
            meta = {
                "source_kind": "domain_search",
                "page_url": page_url,
                "base_context": base_query,
            }
            if hit_meta:
                meta.update(hit_meta)
            _register_image_metadata(dest, meta)
            saved_paths.append(dest)
            return True
        return False

    def _save_from_hits(page_url: str, tsoup, ranked_hits: list, *, per_root_counter: list[int]):
        for node, score, details in ranked_hits:
            if _global_cap_hit():
                raise _StopAll()
            _dbg_hit_reason(page_url, node, phrase_rxs[0])
            hit_meta = {
                "hit_score": score,
                "hit_heading": details.get("heading_text"),
                "hit_snippet": details.get("snippet"),
                "ctx_text": details.get("ctx_text"),
                "ctx_sem_score": details.get("sem_score"),
            }
            if "ctx_embedding" in details:
                hit_meta["ctx_embedding"] = details["ctx_embedding"]
            cands = _extract_img_candidates_near_hit(node, tsoup, page_url, per_hit_cap=per_hit_cap) or []
            for img_url, why in cands:
                if _global_cap_hit():
                    raise _StopAll()
                if _save_one(img_url, tsoup, page_url, hit_meta=hit_meta):
                    per_root_counter[0] += 1
                    dbg(f"[COUNT][per-root] {per_root_counter[0]}/{hard_image_cap} (why={why})")
                    if per_root_counter[0] >= hard_image_cap:
                        dbg(f"[S2] PER-ROOT CAP {hard_image_cap} reached → next root")
                        raise _NextRoot()

    # === TERMINAL PROCESSOR ===
    def _process_terminal_url(tu: str, *, per_root_counter: list[int], fatigue_counter: list[int]):
        """
        Handles per-page logic, counting fatigue *only after first success*.
        """
        before_saves = per_root_counter[0]
        tsoup, _, _ = render(tu, expand=False)
        if not tsoup:
            dbg(f"[S2][TERM] url={tu} soup=None")
            if per_root_counter[0] > 0:
                fatigue_counter[0] += 1
            return

        ranked = smart_find_hits_in_soup(
            tsoup,
            base_query,
            lemma_obj,
            min_score=0.60,
            top_k=None,
            encoder=encoder,
            query_embedding=query_embedding,
            sem_min_score=0.45,
            return_embedding=True,
        )
        dbg(f"[S2][TERM] url={tu} hits={len(ranked)}")

        if not ranked:
            if per_root_counter[0] > 0:
                fatigue_counter[0] += 1
            return

        any_cand = False
        for n, _, _ in ranked[:3]:
            if _extract_img_candidates_near_hit(n, tsoup, tu, per_hit_cap=1):
                any_cand = True
                break
        if not any_cand:
            tsoup, _, _ = render(tu, expand=False, force_js_light=True)

        _save_from_hits(tu, tsoup, ranked_hits=ranked, per_root_counter=per_root_counter)

        # update fatigue only if we've already had a success
        if per_root_counter[0] > before_saves:
            fatigue_counter[0] = 0  # reset after any new image
        elif per_root_counter[0] > 0:
            fatigue_counter[0] += 1

        if per_root_counter[0] > 0 and fatigue_counter[0] >= fatigue_limit:
            dbg(f"[S2] FATIGUE LIMIT ({fatigue_limit}) after first success → next root")
            raise _NextRoot()

    # === MAIN LOOP ===
    seen_urls: set[str] = set()
    try:
        for rdx, root in enumerate(roots, start=1):
            if _global_cap_hit():
                raise _StopAll()

            dbg(f"[S2] ROOT[{rdx}/{len(roots)}]: {root}")
            per_root_saved = [0]
            fatigue_counter = [0]

            try:
                soup0, anchors0, _ = render(root, expand=True)
                if not anchors0:
                    dbg(f"[S2] ROOT DONE {root} (no anchors)")
                    continue

                term0, deeper0 = partition_anchors(root, anchors0, lambda u: _allowed(root, u))
                dbg(f"[S2] root partition: terminal={len(term0)} deeper={len(deeper0)}")

                for tu in term0:
                    if _global_cap_hit(): raise _StopAll()
                    if per_root_saved[0] >= hard_image_cap: raise _NextRoot()
                    if tu in seen_urls: continue
                    seen_urls.add(tu)
                    _process_terminal_url(tu, per_root_counter=per_root_saved, fatigue_counter=fatigue_counter)

                queue = deque(deeper0)
                while queue:
                    if _global_cap_hit(): raise _StopAll()
                    if per_root_saved[0] >= hard_image_cap: raise _NextRoot()
                    cur = queue.popleft()
                    if cur in seen_urls: continue
                    seen_urls.add(cur)

                    csoup, canchors, _ = render(cur, expand=True)
                    if not canchors:
                        if per_root_saved[0] > 0:
                            fatigue_counter[0] += 1
                            if fatigue_counter[0] >= fatigue_limit:
                                dbg(f"[S2] FATIGUE LIMIT ({fatigue_limit}) after first success → next root")
                                raise _NextRoot()
                        continue

                    term, deeper = partition_anchors(cur, canchors, lambda u: _allowed(root, u))
                    dbg(f"[S2] cur={cur} terminal={len(term)} deeper={len(deeper)} queue_left={len(queue)} fatigue={fatigue_counter[0]}")

                    for tu in term:
                        if _global_cap_hit(): raise _StopAll()
                        if per_root_saved[0] >= hard_image_cap: raise _NextRoot()
                        if tu in seen_urls: continue
                        seen_urls.add(tu)
                        _process_terminal_url(tu, per_root_counter=per_root_saved, fatigue_counter=fatigue_counter)

                    for du in deeper:
                        if du not in seen_urls:
                            queue.append(du)

            except _NextRoot:
                dbg(f"[S2] NEXT ROOT (saved {per_root_saved[0]} for {root})")

            dbg(f"[S2] ROOT DONE {root} images_for_root={per_root_saved[0]} total_images={len(saved_paths)}")

    except _StopAll:
        dbg(f"[S2] GLOBAL STOP total_images={len(saved_paths)} (global_cap={global_image_cap})")

    return saved_paths








def find_valid_roots(
    base_url: str,
    lemma_obj: dict,
    max_pages_stage1: int = 100000,          # large ceiling; stopping is layer-based now
    max_pages_stage2_per_root: int = 220,    # untouched (Stage 2)
    timeout: int = 8,
    known_roots: list[str] | None = None,
    js_mode: str = "light",
):
    """
    Stage 1: strict BFS with 'scan then expand'.
    If ANY roots are found in a layer, finish that whole layer, then STOP
    (do NOT expand to the next layer).

    Per layer:
      1) SCAN: check all URLs in the current frontier against lemma phrases → collect roots.
      2) EXPAND: fetch each frontier URL (JS-light), gather anchors → next_frontier (but ONLY
                 if no roots were found this layer).
    - Global 'visited' prevents loops.
    - Non-HTML / scheme-block filters on both frontier URLs and extracted anchors.
    - known_roots passthrough preserved and normalized; returns {root: []}.
    """
    import re
    from collections import deque
    from urllib.parse import urlparse, urlunparse, unquote

    def dbg(*a, **k): print(*a, **k)

    NON_HTML_EXT = (
        ".pdf", ".doc", ".docx", ".zip", ".rar", ".7z",
        ".ppt", ".pptx", ".xls", ".xlsx",
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
        ".mp3", ".wav", ".mp4", ".webm", ".avi"
    )
    SCHEME_BLOCK = ("mailto:", "javascript:", "data:")

    # --- registrable helper (tldextract preferred, safe fallback) ---
    try:
        import tldextract
        def registrable(host: str) -> str:
            t = tldextract.extract(host or "")
            return ".".join([t.domain, t.suffix]) if t.suffix else (t.domain or "")
    except Exception:
        def registrable(host: str) -> str:
            parts = (host or "").split(".")
            return ".".join(parts[-2:]) if len(parts) >= 2 else (host or "")

    # --- URL utils ---
    def clean_url(u: str) -> str:
        if not u:
            return ""
        u = unquote(u)
        if u.lower().startswith(SCHEME_BLOCK):
            return ""
        u = re.sub(r"#.*$", "", u)
        u = re.sub(r"[?&](utm_[^=&]+|fbclid|gclid)=[^&]*", "", u)
        p = urlparse(u)
        scheme = p.scheme or "https"
        host = (p.hostname or "").lower()
        path = p.path or "/"
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        return urlunparse((scheme, host, path, "", p.query, ""))

    def same_reg(u: str, base_reg: str) -> bool:
        return registrable((urlparse(u).hostname or "").lower()) == base_reg

    def is_non_html(u: str) -> bool:
        lu = (u or "").lower()
        return any(lu.endswith(ext) for ext in NON_HTML_EXT)

    def path_segments(u: str):
        p = urlparse(u)
        return [s for s in (p.path or "/").split("/") if s]

    # Bucket by (host, last path segment); preserve first-seen order,
    # but if a *shorter* path for same bucket appears in the same pass, prefer that.
    def prefer_shortest_by_bucket_preserve(urls_iterable):
        best = {}        # bucket -> (url, seg_count, first_index)
        ordered = []
        for idx, u in enumerate(urls_iterable):
            p = urlparse(u)
            host = (p.hostname or "").lower()
            segs = path_segments(u)
            primary = segs[-1] if segs else ""
            bucket = (host, primary)
            seg_count = len(segs)
            if bucket not in best:
                best[bucket] = (u, seg_count, idx)
                ordered.append(bucket)
            else:
                cur_u, cur_seg_count, cur_idx = best[bucket]
                if seg_count < cur_seg_count:
                    best[bucket] = (u, seg_count, cur_idx)  # keep original first_idx
        return [best[b][0] for b in ordered]

    # Use ONLY provided lemma_obj
    lemma_phrases = {
        (w or "").strip().lower()
        for entry in (lemma_obj or {}).values()
        for w in entry.get("lemmas", [])
        if isinstance(w, str) and w.strip()
    }

    def matches_phrase(u: str) -> bool:
        if not lemma_phrases:
            return True
        lu = (u or "").lower()
        return any(p in lu for p in lemma_phrases)

    # --- Normalize inputs ---
    base_c = clean_url(base_url)
    base_host = (urlparse(base_c).hostname or "").lower()
    base_reg = registrable(base_host)

    # known_roots passthrough
    if known_roots:
        roots = [clean_url(r) for r in known_roots if r]
        roots = [r for r in roots if r and same_reg(r, base_reg) and not is_non_html(r)]
        roots = prefer_shortest_by_bucket_preserve(roots)
        dbg(f"[FIND][S1] skipping Stage 1 via known_roots count={len(roots)}")
        return {r: [] for r in roots}

    if not base_host:
        dbg("[FIND][S1] no base_host")
        return {}

    # --- BFS state ---
    visited = set([base_c])
    frontier = deque([base_c])
    roots = []
    breaths = 0
    total_fetches = 0

    while frontier and total_fetches < max_pages_stage1:
        breaths += 1
        layer = list(frontier)
        frontier.clear()
        dbg(f"[FIND][S1][LAYER {breaths} START] frontier={len(layer)} visited={len(visited)} roots={len(roots)}")

        # 1) SCAN: check this layer's URLs for matches (no fetching)
        layer_scan = [u for u in layer if same_reg(u, base_reg) and not is_non_html(u)]
        layer_matches = [u for u in layer_scan if matches_phrase(u)]
        layer_found_any = False

        if layer_matches:
            for m in prefer_shortest_by_bucket_preserve(layer_matches):
                if m not in roots:
                    roots.append(m)
                    layer_found_any = True
                    dbg(f"[FIND][S1][ROOT] {m}")

        # If we found any roots in this layer, STOP after finishing the layer (no next expansion)
        if layer_found_any:
            dbg(f"[FIND][S1][LAYER {breaths} STOP] roots_found={len(roots)} — not expanding further")
            break

        # 2) EXPAND: fetch to collect anchors for NEXT layer
        next_candidates = []
        for u in layer_scan:
            try:
                html, anchors, via = js_capable_fetch(u, js_mode=js_mode)
            except Exception as e:
                dbg(f"[JSFETCH][ERR] {u} -> {e}")
                anchors, via = [], "error"

            total_fetches += 1
            dbg(f"[JSFETCH][{via}] {u} ok anchors={len(anchors) if anchors else 0}")

            if not anchors:
                continue

            for a in anchors:
                cu = clean_url(a)
                if not cu:
                    continue
                if cu in visited:
                    continue
                if not same_reg(cu, base_reg):
                    continue
                if is_non_html(cu):
                    continue
                visited.add(cu)
                next_candidates.append(cu)

        # Keep order; bucket-dedupe while preserving first-seen
        next_frontier = prefer_shortest_by_bucket_preserve(next_candidates)
        for cu in next_frontier:
            frontier.append(cu)

        dbg(f"[FIND][S1][LAYER {breaths} END] next_frontier={len(next_frontier)} roots={len(roots)}")

    # Final normalized roots map
    return {r: [] for r in prefer_shortest_by_bucket_preserve(roots)}





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
def handle_result_no_api(source: Source, query: str, subj: str, hard_image_cap: int = 5,
                         encoder: Optional[Any] = None,
                         query_embedding: Optional[Any] = None):
    def uniq_keep_order(items):
        seen = set()
        out = []
        for x in items:
            if x and x not in seen:
                seen.add(x); out.append(x)
        return out

    # Build subject lemmas for Stage 1 discovery filtering
    lemma_obj_subj = get_limited_lemmas(subj, 4)
    # Build query + synonyms for text matching (Stage 2)
    lemma_obj_query = get_limited_lemmas(query, 5)

    valid_endpoints: List[str] = []

    # Predefined subject urls
    if getattr(source, "subjecturls", None):
        dbg("[NOAPI] subject urls exist")
        for current_subj, url_or_list in source.subjecturls.items():
            if current_subj != subj or not url_or_list:
                continue
            if isinstance(url_or_list, (list, tuple, set)):
                valid_endpoints.extend(url_or_list)
            else:
                valid_endpoints.append(url_or_list)

    # Previously found urls
    if getattr(source, "subjecturlsfound", None):
        dbg("[NOAPI] previously found urls loaded")
        for current_subj, url_or_list in source.subjecturlsfound.items():
            if current_subj != subj or not url_or_list:
                continue
            if isinstance(url_or_list, (list, tuple, set)):
                valid_endpoints.extend(url_or_list)
            else:
                valid_endpoints.append(url_or_list)

    # Normalize + dedupe
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
                # de-dupe
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


    #ALL STAGE 2

    session = requests.Session()
    session.headers.update({"User-Agent": "diag-scrape/0.1"})
    html_cache = {}
    anchors_cache = {}

    # NEW STAGE 2 (drill + text search + image download) — pass the *roots* actually selected
    dbg(f"[NOAPI] Stage 2 starting with roots={len(roots)} hard_image_cap={hard_image_cap}")
    saved = stage2_drill_search_and_images(
        roots=roots,                      # ← FIX: use roots, not valid_endpoints
        base_query=query,
        lemma_obj=lemma_obj_query,        # ← FIX: use query lemmas for text matching
        hard_image_cap=hard_image_cap,    # ← honor caller's cap
        session=session,
        html_cache=html_cache,
        anchors_cache=anchors_cache,
        encoder=encoder,
        query_embedding=query_embedding,
    )


    source.img_paths = saved
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
      - optionally runs smart hits on the CC page to attach context/embedding
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

        ctx_meta = None
        if lemma_obj and encoder is not None and query_embedding is not None:
            try:
                soup = BeautifulSoup(html, "html.parser")
                ranked = smart_find_hits_in_soup(
                    soup,
                    query,
                    lemma_obj,
                    min_score=0.55,
                    top_k=3,
                    encoder=encoder,
                    query_embedding=query_embedding,
                    sem_min_score=0.45,
                    return_embedding=True,
                )
                if ranked:
                    node, score, details = ranked[0]
                    ctx_meta = {
                        "hit_score": score,
                        "hit_heading": details.get("heading_text"),
                        "hit_snippet": details.get("snippet"),
                        "ctx_text": details.get("ctx_text"),
                        "ctx_sem_score": details.get("sem_score"),
                    }
                    if "ctx_embedding" in details:
                        ctx_meta["ctx_embedding"] = details["ctx_embedding"]
            except Exception as e:
                dbg(f"[DDG] smart_hits error: {e}")

        # ok to download the image
        p = _download_to(img_url, dest_dir, len(saved), dbg=dbg)
        if p:
            meta = {
                "source_kind": "ddg",
                "page_url": page_url,
                "image_url": img_url,
            }
            
            if base_ctx_embedding is not None:
                if hasattr(base_ctx_embedding, "tolist"):
                    meta["base_ctx_embedding"] = base_ctx_embedding.tolist()
                else:
                    meta["base_ctx_embedding"] = list(base_ctx_embedding)

            if ctx_meta:
                meta.update(ctx_meta)
            if ctx_meta:
                meta.update(ctx_meta)
            _register_image_metadata(p, meta)
            saved.append(p)
            dbg(f"[DDG] saved ({len(saved)}/{target_count}) {p}")

    dbg(f"[DDG] done saved={len(saved)}")
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



def collect_unique_images(images_root: str, dbg=print):
    """
    Walks `images_root`, finds unique images, and copies them into
    <images_root>/UniqueImages.
    Dedup rules:
      1) Exact file content (SHA1) → drop duplicates.
      2) Same perceptual hash (aHash64) → drop later ones (handles 1:1 resized copies).
    Keeps the first encountered path for each group.
    Also aggregates IMAGE_METADATA into a per-unique-image map and writes
    <UniqueImages>/image_metadata.json.
    """
    unique_paths: list[str] = []
    unique_dir = os.path.join(images_root, "UniqueImages")
    os.makedirs(unique_dir, exist_ok=True)

    sha1_to_dest: Dict[str, str] = {}
    ahash_to_dest: Dict[str, str] = {}
    unique_meta: Dict[str, List[dict]] = {}

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

            dest_path = sha1_to_dest.get(sha1)

            # 2) perceptual hash if needed
            if dest_path is None:
                ah = _ahash64(path)
                if ah is not None and ah in ahash_to_dest:
                    dest_path = ahash_to_dest[ah]

            # 3) new unique -> copy
            if dest_path is None:
                try:
                    dest_path = os.path.join(unique_dir, os.path.basename(path))
                    base, ext = os.path.splitext(dest_path)
                    n = 1
                    while os.path.exists(dest_path):
                        dest_path = f"{base}_{n}{ext}"
                        n += 1
                    shutil.copy2(path, dest_path)
                    unique_paths.append(dest_path)
                    dbg(f"[collect][copy] {path} -> {dest_path}")
                except Exception as e:
                    dbg(f"[collect][err][copy] {path} -> {e}")
                    continue

                sha1_to_dest[sha1] = dest_path
                ah_local = _ahash64(path)
                if ah_local is not None:
                    ahash_to_dest[ah_local] = dest_path

            # 4) merge metadata for this path into the unique dest
            metas = IMAGE_METADATA.get(path)
            if metas:
                lst = unique_meta.get(dest_path)
                if lst is None:
                    unique_meta[dest_path] = list(metas)
                else:
                    lst.extend(metas)

    # write metadata JSON
    try:
        meta_path = os.path.join(unique_dir, "image_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(unique_meta), f, indent=2, ensure_ascii=False)
        dbg(f"[collect] metadata saved to {meta_path}")
    except Exception as e:
        dbg(f"[collect][err][meta] {e}")

    dbg(f"[collect] unique={len(unique_paths)} saved to {unique_dir}")
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
def research(query : str, subj : str ):

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
    #        status, data, built = send_request(src, settings)
    #        dbg(f"sent request, responded {status}")
    #        if status == 200 and data is not None:
    #            try:
    #                _ = PARSERS.get(src.name)
    #                if _:
    #                   _. __call__  # noqa: touch
    #            except Exception:
    #               pass
    #        # parse regardless; the parser guards internally
    #        if data is not None:
    #            parse_fn = PARSERS.get(src.name)
    #            if parse_fn:
    #                parse_fn(src, data)
        else:
            web_sources.append(src)


    ddg_cc_image_harvest(
        query=query,
        target_count=7,
        lemma_obj=lemma_obj_query,
        encoder=encoder,
        query_embedding=query_embedding,
        base_ctx_embedding=base_ctx_embedding
    )
    
    for src in web_sources:
        dbg(f"opened source {src.name}, has no api, starting process..")
        handle_result_no_api(src, query, subj, hard_image_cap=5,
                             encoder=encoder,
                             query_embedding=query_embedding)

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
            _register_image_metadata(p, meta)
    
    unique = collect_unique_images(IMAGES_PATH, dbg)
    dbg(f"Unique images:{unique}")
    return unique
