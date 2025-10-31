import os, json, copy, requests, re, hashlib, datetime, time, threading
from urllib.parse import urlparse, unquote, urljoin, urlunparse
from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup ,NavigableString, Tag
import tldextract
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

from duckduckgo_search import DDGS  # (left intact for API section)

# ========= CONFIG & DEBUG =========
DEBUG = True
def dbg(*args):
    if DEBUG:
        print(*args, flush=True)

SOURCE_PATH = r'C:\Users\marti\Code\DrawnOutWhiteboard\whiteboard\source_urls'
IMAGES_PATH = r'C:\Users\marti\Code\DrawnOutWhiteboard\whiteboard\ResearchImages'

SUBJECTS = [
    "Maths",
    "Physics",
    "Biology",
    "Chemistry",
    "Geography",
]

# ----- LICENSE/HOST FILTERS (left intact for API sections and potential DDG) -----
CC_PATTERNS = [
    r"creativecommons\.org/licenses/([a-z\-0-9/\.]+)",
    r"creativecommons\.org/publicdomain/([a-z\-0-9/\.]+)",
    r"\bCC\s?BY(?:-SA|-NC|-ND)?\s?(?:\d\.\d)?\b",
    r"\bCC0\b",
    r"\bCreative\s+Commons\b.*?(?:Attribution|ShareAlike|NonCommercial|NoDerivatives|Zero|Public\s+Domain)",
    r"\bPublic\s+Domain\b",
]

BLOCKED_HOSTS = {
    "twitter.com", "x.com", "facebook.com", "instagram.com", "tiktok.com",
    "pinterest.com", "pinimg.com", "reddit.com", "imgur.com", "tumblr.com"
}

IMG_EXT_RE = re.compile(r"\.(png|jpe?g|gif|webp|svg|tiff?)($|\?)", re.I)

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

def expand_all_disclosures_sync(page, max_rounds: int = 8, per_round_click_cap: int = 80, quiet_ms: int = 350):
    """
    Generic, dynamic expansion of hidden TOCs/accordions/dropdowns.
    - Blocks 'document' navigations while expanding to avoid leaving the page.
    - Opens <details>, toggles [aria-expanded], [data-*] collapses, and common accordion/dropdown classes.
    - Repeats until no new anchors appear or max_rounds reached.
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

        # expand <details> everywhere (very cheap + reliable)
        page.evaluate("""
(() => {
  document.querySelectorAll('details:not([open])').forEach(d => { try { d.open = true; } catch(e) {} });
})();
""")

        for round_i in range(1, max_rounds + 1):
            clicks = 0

            # 1) Expand elements with aria-expanded/aria-controls and common collapsed classes
            # We filter to *visible* and *collapsed-looking* candidates in JS for speed.
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

            # Helper inside loop to decide if an element looks collapsed and visible
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
            # print round summary
            print(f"[EXPAND] round={round_i} clicks={clicks} anchors={cur} (prev={prev})")
            if clicks == 0 and cur <= prev:
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
    expand: bool = True,
    max_modal_clicks: int = 6,
) -> Tuple[str | None, List[str], str]:
    if url.lower().endswith((".pdf", ".docx", ".zip", ".pptx", ".xls", ".xlsx", ".png", ".jpg", ".jpeg")):
        dbg(f"[SKIP] non-HTML resource: {url}")
        return None, [], "none"
    if any(bad in url.lower() for bad in BLOCKED_URL_WORDS):
        dbg(f"[SKIP][BLOCKLIST] {url}")
        return None, [], "blocked"

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

            # IMPORTANT: pass expand flag; if False, also zero out modal clicks
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

            dbg(f"[JSFETCH][playwright] {url} ok anchors={len(anchors)} html_len={len(html)} expand={expand}")
            return html, anchors, "playwright"
    except ImportError:
        dbg("[JSFETCH] Playwright not installed (required).")
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

def download_image(session: requests.Session, img_url: str, dest_dir: str) -> Optional[str]:
    """
    Robust downloader:
      - supports data: URLs (base64)
      - follows redirects
      - rejects HTML pages
      - infers extension from content-type/magic bytes/SVG text
      - assigns deterministic name from SHA1(content) to avoid accidental overwrite
      - skips tiny UI assets (< 6 KB) unless SVG
    Returns local file path or None.
    """
    try:
        # --- data: URLs ---
        if img_url.startswith("data:image/"):
            # parse: data:image/<subtype>;base64,<payload>
            m = re.match(r"^data:(image/[\w\+\-\.]+);base64,(.+)$", img_url, re.I | re.S)
            if not m:
                return None
            ctype = m.group(1)
            payload = m.group(2)
            raw = base64.b64decode(payload, validate=True)
            # try text decode for SVG test; ignore errors
            as_text = None
            try:
                as_text = raw.decode("utf-8", errors="ignore")
            except Exception:
                pass
            ext = _sniff_image_type(raw[:64], ctype, as_text) or "img"
            # tiny asset guard (keep SVG even if small)
            if ext != "svg" and len(raw) < 6 * 1024:
                dbg(f"[IMG_SAVE][tiny/ui-skip] data-url ({len(raw)} B)")
                return None
            h = hashlib.sha1(raw).hexdigest()
            out = _unique_path_with_ext(dest_dir, h, ext)
            with open(out, "wb") as f:
                f.write(raw)
            return out

        # --- http(s) URLs ---
        headers = {"Referer": img_url, **UA}
        with session.get(img_url, headers=headers, stream=True, timeout=30, allow_redirects=True) as r:
            r.raise_for_status()

            ctype = r.headers.get("Content-Type", "")
            # pre-read head (and a small text sample) to sniff type and catch HTML
            buf = bytearray()
            max_peek = 4096
            for chunk in r.iter_content(4096):
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) >= max_peek:
                    break

            # If looks like HTML, bail
            head_lower = bytes(buf[:64]).lower()
            if b"<html" in head_lower or b"<!doctype html" in head_lower:
                dbg(f"[IMG_SAVE][html-skip] {img_url}")
                return None

            # For SVG we need some text to confirm; try decode a small slice
            as_text = None
            try:
                as_text = bytes(buf).decode("utf-8", errors="ignore")
            except Exception:
                pass

            ext = _sniff_image_type(bytes(buf[:64]), ctype, as_text) or ""
            # tiny asset guard (skip logos etc.) unless it's clearly SVG
            cl_header = r.headers.get("Content-Length")
            maybe_len = int(cl_header) if (cl_header and cl_header.isdigit()) else None
            if ext != "svg":
                # consider both peeked + reported length if available
                if (maybe_len is not None and maybe_len < 6 * 1024) or (maybe_len is None and len(buf) < 6 * 1024):
                    dbg(f"[IMG_SAVE][tiny/ui-skip] {img_url} ({maybe_len or len(buf)} B)")
                    return None

            # stream to bytes (we already have 'buf')
            data = bytes(buf)
            for chunk in r.iter_content(8192):
                if not chunk:
                    break
                data += chunk

            # If we still couldn't infer type, last-ditch: derive from URL path
            if not ext:
                path_ext = os.path.splitext(urlparse(img_url).path)[1].lower().lstrip(".")
                if path_ext in {"png", "jpg", "jpeg", "gif", "webp", "svg", "tiff", "bmp"}:
                    ext = "jpg" if path_ext == "jpeg" else path_ext
                else:
                    # final attempt: simple magic again on full data
                    ext = _sniff_image_type(data[:64], ctype, as_text) or "img"

            # name by content hash (prevents overwrite collisions)
            h = hashlib.sha1(data).hexdigest()
            out = _unique_path_with_ext(dest_dir, h, ext)
            with open(out, "wb") as f:
                f.write(data)
            return out

    except Exception as e:
        dbg(f"[IMG_SAVE][err] {img_url} -> {e}")
        return None



    

def _compile_phrase_regexes_adapter(base_query: str, lemma_obj: dict):
    """
    Normalize to (phrase_rxs, proximity_tokens). We *always* return an empty token list
    so nothing downstream can do proximity.
    """
    try:
        phrase_rxs = compile_phrase_regexes(base_query, lemma_obj)
        return phrase_rxs, []  # ← empty tokens = no proximity anywhere
    except Exception:
        esc = r"\b" + re.sub(r"\s+", " ", re.escape(base_query)).replace(r"\ ", " ") + r"\b"
        return [re.compile(esc, re.IGNORECASE)], []

    

_URL_LIKE_RE = re.compile(r"""(?ix)
    \b(?:https?://|www\.)\S+   # URLs
  | \b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b  # emails
""")
_WS_RE = re.compile(r"\s+")

def _is_hidden_tag(tag: Tag) -> bool:
    # structural skip
    if tag.name in ("script", "style", "noscript", "template", "code", "pre"):
        return True
    # attribute-based hiding
    try:
        if tag.has_attr("hidden"):
            return True
        if tag.get("aria-hidden", "").strip().lower() == "true":
            return True
        style = (tag.get("style") or "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            return True
    except Exception:
        pass
    # common “noise” containers/classes
    cls = (tag.get("class") or [])
    cls = " ".join(cls).lower() if isinstance(cls, (list, tuple)) else str(cls).lower()
    if any(x in cls for x in ("sr-only", "visually-hidden", "skip-link", "breadcrumb", "nav", "footer", "sidebar")):
        return True
    return False

def _clean_search_text(s: str) -> str:
    # strip urls/emails and collapse whitespace
    s = _URL_LIKE_RE.sub(" ", s or "")
    s = _WS_RE.sub(" ", s).strip()
    return s

def find_hits_in_soup(
    soup: BeautifulSoup,
    phrase_rxs: List[re.Pattern],
    max_hits_per_page: int = 10,
):
    """
    Exact-phrase hits over *visible textual content only*.

    - Ignores script/style/noscript/template/code/pre and hidden elements.
    - Removes URLs/emails from text before matching.
    - Preserves DOM order; caps results by max_hits_per_page.
    """
    if not soup or not phrase_rxs:
        return []

    hits = []
    seen = set()

    # iterate paragraph-like & texty containers in DOM order
    for node in soup.find_all(["main", "article", "section", "div", "p", "li", "td", "th", "figcaption", "h1","h2","h3","h4","h5","h6"]):
        if _is_hidden_tag(node):
            continue
        try:
            # fast skip: if the subtree contains any obviously hidden inner tag, BeautifulSoup won't compute CSS;
            # but we already filtered the container; it's enough for our purpose here.
            txt = node.get_text(" ", strip=True)
        except Exception:
            continue

        text = _clean_search_text(txt)
        if not text:
            continue

        matched = False
        for rx in phrase_rxs:
            try:
                if rx.search(text):
                    matched = True
                    break
            except re.error:
                continue

        if matched:
            nid = id(node)
            if nid not in seen:
                seen.add(nid)
                hits.append(node)
                # DEBUG: show a safe snippet so you can verify it’s real text, not JSON
                if DEBUG:
                    snip = (text[:160] + "…") if len(text) > 160 else text
                    dbg(f"[HIT][text-only] snippet='{snip}'")
                if len(hits) >= max_hits_per_page:
                    break

    return hits


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
) -> list[str]:
    phrase_rxs, proximity_tokens = _compile_phrase_regexes_adapter(base_query, lemma_obj)

    def render(url: str, *, expand: bool) -> tuple[Optional[BeautifulSoup], list[str], bool]:
        if url in html_cache:
            soup = BeautifulSoup(html_cache[url], "html.parser") if html_cache[url] else None
            a = anchors_cache.get(url, [])
            dbg(f"[S2][cache] url={url} anchors={len(a)} soup_ok={bool(html_cache[url])}")
            return soup, a, False

        html, anchors, via = js_capable_fetch(url, expand=expand)
        if not html and not anchors:
            html_cache[url] = None
            anchors_cache[url] = []
            return None, [], False

        full = ordered_dedupe([urljoin(url, a) for a in anchors if a])
        html_cache[url] = html
        anchors_cache[url] = full
        dbg(f"[S2] rendered via={via} url={url} anchors={len(full)} soup_ok={bool(html)} expand={expand}")
        soup = BeautifulSoup(html, "html.parser") if html else None
        return soup, full, True

    def _parent_part(u: str) -> str:
        p = urlparse(u)
        path = p.path or "/"
        if path.endswith("/") and len(path) > 1:
            path = path[:-1]
        if "/" in path:
            path = path.rsplit("/", 1)[0]
        else:
            path = "/"
        return urlunparse((p.scheme or "https", (p.hostname or "").lower(), path, "", "", ""))

    def partition_anchors(cur: str, anchors: list[str], allowed_fn) -> tuple[list[str], list[str]]:
        cur_parent = _parent_part(cur)
        terms, deeper = [], []
        for a in anchors:
            if not a or a == cur:
                continue
            if not allowed_fn(a):
                continue
            if _parent_part(a) == cur_parent:
                terms.append(a)
            else:
                deeper.append(a)
        return ordered_dedupe(terms), ordered_dedupe(deeper)

    saved_paths: list[str] = []
    seen_urls: set[str] = set()

    for rdx, root in enumerate(roots, start=1):
        dbg(f"[S2] ROOT[{rdx}/{len(roots)}]: {root}")

        def _allowed(child_url: str) -> bool:
            try:
                return domain_and_phrase_lock(root, child_url)
            except Exception:
                return False

        # ROOT: expand to expose TOC / nested links
        soup0, anchors0, _ = render(root, expand=True)
        if not anchors0:
            dbg(f"[S2] ROOT DONE {root} images_so_far={len(saved_paths)}"); 
            continue

        term0, deeper0 = partition_anchors(root, anchors0, _allowed)
        dbg(f"[S2] root partition: terminal={len(term0)} deeper={len(deeper0)}")

        # TERMINALS: fetch FAST (expand=False)
        for tu in term0:
            if tu in seen_urls:
                continue
            seen_urls.add(tu)
            tsoup, _, _ = render(tu, expand=False)      # ← no expand on terminals
            if not tsoup:
                dbg(f"[S2][TERM] url={tu} soup=None"); 
                continue
            hits = find_hits_in_soup(tsoup, phrase_rxs=phrase_rxs)
            dbg(f"[S2][TERM] url={tu} hits={len(hits)}")
            for node in hits:
                _dbg_hit_reason(tu, node, phrase_rxs[0])
                if len(saved_paths) >= hard_image_cap:
                    dbg(f"[S2] HARD CAP reached ({hard_image_cap})"); 
                    return saved_paths
                candidate = nearest_image_src_for_hit(node, tsoup, tu)
                if not candidate:
                    continue
                dest = save_image_candidate(
                    session=session,
                    candidate=candidate,
                    soup_for_inline=tsoup,
                    page_url=tu,
                    dest_dir=os.path.join(IMAGES_PATH, "domain_search"),
                    referer=tu,
                    dedupe_set=globals().setdefault("_saved_img_urls", set()),
                )
                if dest:
                    dbg(f"[IMG] saved {dest} (from {tu})")
                    saved_paths.append(dest)
                    if len(saved_paths) >= hard_image_cap:
                        dbg(f"[S2] HARD CAP reached ({hard_image_cap})"); return saved_paths


        # DEEPER: expand to harvest more children
        queue = deque(deeper0)
        visited_parents = set()
        while queue and len(saved_paths) < hard_image_cap:
            cur = queue.popleft()
            if cur in seen_urls:
                continue
            seen_urls.add(cur)

            csoup, canchors, _ = render(cur, expand=True)   # ← expand on deeper pages
            if not canchors:
                dbg(f"[S2] cur={cur} terminal=0 deeper=0 queue_left={len(queue)}"); 
                continue

            term, deeper = partition_anchors(cur, canchors, _allowed)
            dbg(f"[S2] cur={cur} terminal={len(term)} deeper={len(deeper)} queue_left={len(queue)}")

            for tu in term:
                if tu in seen_urls:
                    continue
                seen_urls.add(tu)
                tsoup, _, _ = render(tu, expand=False)      # ← no expand on terminals
                if not tsoup:
                    dbg(f"[S2][TERM] url={tu} soup=None"); 
                    continue
                hits = find_hits_in_soup(tsoup, phrase_rxs=phrase_rxs,)
                dbg(f"[S2][TERM] url={tu} hits={len(hits)}")
                for node in hits:
                    _dbg_hit_reason(tu, node, phrase_rxs[0])
                    if len(saved_paths) >= hard_image_cap:
                        dbg(f"[S2] HARD CAP reached ({hard_image_cap})"); 
                        return saved_paths
                    candidate = nearest_image_src_for_hit(node, tsoup, tu)
                    if not candidate:
                        continue
                    dest = save_image_candidate(
                        session=session,
                        candidate=candidate,
                        soup_for_inline=tsoup,
                        page_url=tu,
                        dest_dir=os.path.join(IMAGES_PATH, "domain_search"),
                        referer=tu,
                        dedupe_set=globals().setdefault("_saved_img_urls", set()),
                    )
                    if dest:
                        dbg(f"[IMG] saved {dest} (from {tu})")
                        saved_paths.append(dest)
                        if len(saved_paths) >= hard_image_cap:
                            dbg(f"[S2] HARD CAP reached ({hard_image_cap})"); return saved_paths


            visited_parents.add(_parent_part(cur))
            for du in deeper:
                if du not in seen_urls:
                    queue.append(du)

        dbg(f"[S2] ROOT DONE {root} images_so_far={len(saved_paths)}")

    return saved_paths




# =========================
# STAGE 1 (kept as-is; returns roots)
# =========================
def find_valid_roots(
    base_url: str,
    lemma_obj: dict,
    max_pages_stage1: int = 80,
    timeout: int = 8,
    known_roots: List[str] | None = None,
) -> List[str]:
    """
    Stage 1 (light): Use JS fetch on base_url; collect same-site anchors.
    Pick URLs whose string contains any lemma-phrase (subject synonyms).
    If known_roots are provided, just normalize and return them.
    """
    try:
        def registrable(host: str) -> str:
            if not host:
                return ""
            t = tldextract.extract(host)
            return ".".join([t.domain, t.suffix]) if t.suffix else t.domain
    except Exception:
        def registrable(host: str) -> str:
            if not host:
                return ""
            parts = host.split(".")
            return ".".join(parts[-2:]) if len(parts) >= 2 else host

    parsed_base = urlparse(base_url)
    base_host = (parsed_base.hostname or "").lower()
    base_reg = registrable(base_host)

    lemma_phrases = {
        (w or "").strip().lower()
        for entry in (lemma_obj or {}).values()
        for w in entry.get("lemmas", [])
        if isinstance(w, str) and len(w.strip()) > 1
    }

    def url_matches_any_phrase(u: str) -> bool:
        lu = (u or "").lower()
        return any(p in lu for p in lemma_phrases) if lemma_phrases else True

    if known_roots:
        roots = []
        for u in known_roots:
            cu = clean_url(u)
            if not cu:
                continue
            h = (urlparse(cu).hostname or "").lower()
            if not same_registrable(base_host, h):
                continue
            if cu.lower().endswith(NON_HTML_EXT):
                continue
            roots.append(cu)
        dbg(f"[FIND][S1] skipping discovery via known_roots count={len(roots)}")
        return sorted(set(roots))

    # Discovery from base
    html0, anchors0, via0 = js_capable_fetch(clean_url(base_url))
    same_site = set()
    if not anchors0 and not html0:
        dbg(f"[FIND][S1] base fetch failed: {base_url}")
        return []

    for a in anchors0:
        cu = clean_url(urljoin(base_url, a))
        if not cu:
            continue
        lcu = cu.lower()
        if lcu.endswith(NON_HTML_EXT) or any(b in lcu for b in BLOCKED_URL_WORDS):
            continue
        if not same_registrable((urlparse(base_url).hostname or "").lower(), (urlparse(cu).hostname or "").lower()):
            continue
        if url_matches_any_phrase(cu):
            same_site.add(cu)

    roots = list(same_site)[:max_pages_stage1]  # mild cap
    dbg(f"[FIND][S1] roots_found={len(roots)} via={via0}")
    return roots

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

def build_params_from_settings(source: Source, settings: dict) -> dict:
    params = copy.deepcopy(source.Template)
    for k, v in settings.items():
        if k == "pagination_field":
            api_field_name = source.TemplateEQ.get(k)
            if api_field_name and v and isinstance(v, str) and v.strip() and v.lower() != "continue":
                params[api_field_name] = v
            continue
        api_field_name = source.TemplateEQ.get(k)
        if api_field_name:
            params[api_field_name] = v
    fmt_field = source.TemplateEQ.get("format_field")
    fmt_value = settings.get("format_value") or settings.get("format_field")
    if fmt_field and fmt_value:
        params[fmt_field] = fmt_value
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

# ---------------------------
# API Parsers (kept)
# ---------------------------
def parse_wikimedia(src, data):
    """
    Robust Wikimedia Commons parser.

    Fixes & improvements:
    - Handles multiple pages and multiple imageinfo entries per page.
    - Falls back to 'thumburl' when 'url' is missing or download fails.
    - Filters to actual images via mime/ext and deduplicates URLs.
    - Builds safe, non-colliding filenames; adds extension from mime when missing.
    - Captures lightweight license evidence from extmetadata (for debugging) via dbg().
    - Keeps your original directory layout and return type.
    """
    # ---- collect candidate image URLs from the API response ----
    img_urls = []
    if isinstance(data, dict):
        pages = (data.get("query") or {}).get("pages") or {}
        for _, page in pages.items():
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue
            for ii in info_list:
                # Prefer original 'url', fall back to 'thumburl'
                u = ii.get("url") or ii.get("thumburl") or ""
                if not u:
                    continue

                # Basic type gate: mime or extension should look like an image
                mime = (ii.get("mime") or "").lower()
                has_image_mime = mime.startswith("image/")
                has_image_ext = bool(re.search(r"\.(png|jpe?g|gif|webp|svg|tiff?)($|\?)", u, re.I))
                if not (has_image_mime or has_image_ext):
                    continue

                # (Optional) License hints for debugging (Wikimedia usually CC/PD)
                em = ii.get("extmetadata") or {}
                lic_short = (em.get("LicenseShortName", {}) or {}).get("value")
                lic_url   = (em.get("LicenseUrl", {}) or {}).get("value")
                if lic_short or lic_url:
                    try:
                        dbg(f"[PARSER][wikimedia] license={lic_short or '?'} {lic_url or ''}")
                    except Exception:
                        pass

                img_urls.append(u)

    # Dedup while preserving order
    seen = set()
    img_urls = [u for u in img_urls if not (u in seen or seen.add(u))]

    if not img_urls:
        src.img_paths = []
        return []

    # ---- destination dir ----
    base_name = os.path.splitext(src.name)[0] if getattr(src, "name", None) else "wikimedia_batch"
    dest_dir = os.path.join(IMAGES_PATH, "wikimedia", base_name)
    os.makedirs(dest_dir, exist_ok=True)

    # ---- filename builder ----
    def _choose_ext_from_mime(mime: str) -> str:
        mime = (mime or "").lower()
        if mime.startswith("image/"):
            ext = mime.split("/", 1)[1].split(";")[0].strip()
            if ext == "jpeg": return ".jpg"
            if ext: return f".{ext}"
        return ".img"

    def build_filename(u: str, idx: int, mime_hint: str = "") -> str:
        parsed = urlparse(u)
        fname = os.path.basename(parsed.path) or f"file_{idx}"
        fname = unquote(fname)
        fname = re.sub(r"[?#].*$", "", fname)
        root, ext = os.path.splitext(fname)

        if not ext:
            ext = _choose_ext_from_mime(mime_hint)

        # Make it filesystem-safe but readable
        safe_root = re.sub(r"[^A-Za-z0-9._\- ]+", "_", root)[:80] or f"file_{idx}"
        candidate = os.path.join(dest_dir, f"{safe_root}{ext}")
        if not os.path.exists(candidate):
            return candidate

        # Avoid collisions
        n = 1
        while True:
            alt = os.path.join(dest_dir, f"{safe_root}_{n}{ext}")
            if not os.path.exists(alt):
                return alt
            n += 1

    # ---- downloader with fallback to thumburl if original fails ----
    headers = {"User-Agent": "diag-scrape/0.1"}

    def _try_download(url: str, local_path: str) -> bool:
        try:
            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                r.raise_for_status()
                # Guard: ensure it's an image
                ctype = (r.headers.get("Content-Type") or "").lower()
                if not ctype.startswith("image/") and not re.search(r"\.(png|jpe?g|gif|webp|svg|tiff?)($|\?)", url, re.I):
                    return False
                # Optional small-file guard (skip likely icons)
                clen = r.headers.get("Content-Length")
                if clen is not None:
                    try:
                        if int(clen) < 2048:  # <2KB? probably a spacer
                            return False
                    except Exception:
                        pass
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception:
            return False

    saved_paths = []
    for i, u in enumerate(img_urls):
        # We may not have mime here—only the URL. We'll infer extension from URL;
        # server response will be validated.
        local_path = build_filename(u, i)

        ok = _try_download(u, local_path)
        if not ok:
            # Fallback: if this was an original likely failing due to size/permission,
            # try to coerce a thumbnail URL if we can (Wikimedia often provides 'thumb' paths).
            # Quick heuristic: replace '/commons/' original path with '/commons/thumb/.../WIDTHpx-<name>'
            # Only do this if the URL looks like a Commons upload path.
            fallback = None
            try:
                p = urlparse(u)
                # Typical pattern: https://upload.wikimedia.org/wikipedia/commons/<a>/<ab>/FileName.ext
                if "upload.wikimedia.org" in (p.netloc or "") and "/commons/" in (p.path or ""):
                    base = p.path
                    if "/thumb/" not in base:
                        fname = os.path.basename(base)
                        # crude: insert '/thumb' and append width variant
                        head = base.rsplit("/", 1)[0]
                        fallback = f"{p.scheme}://{p.netloc}{head}/thumb/{fname}/800px-{fname}"
            except Exception:
                fallback = None

            if fallback:
                local_path = build_filename(fallback, i)
                ok = _try_download(fallback, local_path)

        if ok:
            saved_paths.append(local_path)
            try:
                dbg(f"[PARSER][wikimedia] saved {local_path}")
            except Exception:
                pass

    src.img_paths = saved_paths
    return saved_paths

def parse_openverse(src, data):
    img_urls = []
    if isinstance(data, dict):
        for item in data.get("results", []):
            u = item.get("url")
            if u:
                img_urls.append(u)
    if not img_urls:
        src.img_paths = []
        return []

    base_name = os.path.splitext(src.name)[0] if getattr(src, "name", None) else "openverse_batch"
    dest_dir = os.path.join(IMAGES_PATH, "openverse", base_name)
    os.makedirs(dest_dir, exist_ok=True)

    saved_paths = []
    def build_filename(u, idx):
        parsed = urlparse(u)
        fname = os.path.basename(parsed.path) or f"file_{idx}"
        fname = unquote(fname)
        fname = re.sub(r"[?#].*$", "", fname)
        if not os.path.splitext(fname)[1]:
            fname = f"{fname}.img"
        candidate = os.path.join(dest_dir, fname)
        if not os.path.exists(candidate):
            return candidate
        root, ext = os.path.splitext(candidate)
        n = 1
        while True:
            alt = f"{root}_{n}{ext}"
            if not os.path.exists(alt):
                return alt
            n += 1

    headers = {"User-Agent": "diag-scrape/0.1"}
    for i, u in enumerate(img_urls):
        local_path = build_filename(u, i)
        try:
            with requests.get(u, headers=headers, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
            saved_paths.append(local_path)
            dbg(f"[PARSER][openverse] saved {local_path}")
        except Exception:
            continue

    src.img_paths = saved_paths
    return saved_paths

def parse_plos(src, data):
    import re
    from urllib.parse import urljoin

    dois = []
    if isinstance(data, dict):
        docs = data.get("response", {}).get("docs", [])
        for doc in docs[:5]:
            doi = doc.get("doi") or doc.get("id")
            if doi:
                dois.append(doi)

    if not dois:
        src.img_paths = []
        return []

    base_name = os.path.splitext(src.name)[0] if getattr(src, "name", None) else "plos_batch"
    batch_dir = os.path.join(IMAGES_PATH, "plos", base_name)
    os.makedirs(batch_dir, exist_ok=True)

    headers = {"User-Agent": "diag-scrape/0.1"}
    saved_paths = []

    def safe_filename_from_url(u: str, idx: int) -> str:
        parsed = urlparse(u)
        fname = os.path.basename(parsed.path) or f"file_{idx}"
        fname = unquote(fname)
        fname = re.sub(r"[?#].*$", "", fname)
        if not os.path.splitext(fname)[1]:
            fname = f"{fname}.img"
        return fname

    for a_idx, doi in enumerate(dois, start=1):
        try:
            resp = requests.get(f"https://doi.org/{doi}", headers=headers, timeout=30, allow_redirects=True)
            resp.raise_for_status()
            article_url = resp.url
        except Exception:
            continue

        try:
            art = requests.get(article_url, headers=headers, timeout=30)
            art.raise_for_status()
        except Exception:
            continue

        soup = BeautifulSoup(art.text, "html.parser")
        img_urls = set()

        for img in soup.select("figure img"):
            src_attr = img.get("src") or img.get("data-src")
            if not src_attr:
                continue
            img_urls.add(urljoin(article_url, src_attr))

        for a in soup.select("figure a[href]"):
            href = a.get("href")
            if href and re.search(r"\.(png|jpe?g|gif|svg)(\?|$)", href, re.IGNORECASE):
                img_urls.add(urljoin(article_url, href))

        if not img_urls:
            continue

        article_dir = os.path.join(batch_dir, f"article_{a_idx}")
        os.makedirs(article_dir, exist_ok=True)

        for i, u in enumerate(sorted(img_urls)):
            local_name = safe_filename_from_url(u, i)
            local_path = os.path.join(article_dir, local_name)
            if os.path.exists(local_path):
                saved_paths.append(local_path)
                continue
            try:
                with requests.get(u, headers=headers, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r.iter_content(8192):
                            if chunk:
                                f.write(chunk)
                saved_paths.append(local_path)
                dbg(f"[PARSER][plos] saved {local_path}")
            except Exception:
                continue

    src.img_paths = saved_paths
    return saved_paths

def parse_usgs(src, data):
    import re
    img_tasks = []
    if isinstance(data, dict):
        items = data.get("items", [])
        for idx, item in enumerate(items, start=1):
            file_entries = item.get("files") or item.get("attachments") or []
            base_url = item.get("link") or item.get("self", {}).get("linkUrl") or ""
            for f in file_entries:
                u = f.get("url") or f.get("downloadUri") or ""
                if not u:
                    href = f.get("href") or f.get("uri") or ""
                    if href:
                        u = urljoin(base_url, href)
                if not u:
                    continue
                ctype = (f.get("contentType") or "").lower()
                is_image = any(mt in ctype for mt in ("image/png","image/jpeg","image/jpg","image/gif","image/svg","image/webp","image/tiff"))
                if not is_image and re.search(r"\.(png|jpe?g|gif|svg|webp|tiff?)($|\?)", u, re.I):
                    is_image = True
                if is_image:
                    img_tasks.append((idx, u))

    if not img_tasks:
        src.img_paths = []
        return []

    base_name = os.path.splitext(src.name)[0] if getattr(src, "name", None) else "usgs_batch"
    root_dir = os.path.join(IMAGES_PATH, "usgs", base_name)
    os.makedirs(root_dir, exist_ok=True)

    saved_paths = []
    def build_filename(u: str, idx_in_item: int, item_idx: int) -> str:
        parsed = urlparse(u)
        fname = os.path.basename(parsed.path) or f"file_{idx_in_item}"
        fname = unquote(fname)
        fname = re.sub(r"[?#].*$", "", fname)
        if not os.path.splitext(fname)[1]:
            fname = f"{fname}.img"
        item_dir = os.path.join(root_dir, f"item_{item_idx}")
        os.makedirs(item_dir, exist_ok=True)
        candidate = os.path.join(item_dir, fname)
        if not os.path.exists(candidate):
            return candidate
        root, ext = os.path.splitext(candidate)
        n = 1
        while True:
            alt = f"{root}_{n}{ext}"
            if not os.path.exists(alt):
                return alt
            n += 1

    headers = {"User-Agent": "diag-scrape/0.1"}
    for seq, (item_idx, url) in enumerate(img_tasks):
        local_path = build_filename(url, seq, item_idx)
        try:
            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
            saved_paths.append(local_path)
            dbg(f"[PARSER][usgs] saved {local_path}")
        except Exception:
            continue

    src.img_paths = saved_paths
    return saved_paths

PARSERS = {
    "wikimedia": parse_wikimedia,
    "openverse": parse_openverse,
    "plos": parse_plos,
    "usgs": parse_usgs
}

# =========================
# HANDLE NON-API (NEW PIPELINE)
# =========================
def handle_result_no_api(source: Source, query: str, subj: str, hard_image_cap: int = 20):
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
        roots = find_valid_roots(base_url=source.url, lemma_obj=lemma_obj_subj, max_pages_stage1=80)

    if not roots:
        dbg(f"[NOAPI] no roots after Stage 1 for {source.name}")
        source.img_paths = []
        return []

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
    )

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

    source.img_paths = saved
    return saved


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    subj = "Biology"
    query = "Prokaryotic Cells"

    settings = {
        "query_field": query,
        "limit_field": 10,
        "pagination_field": 1,
        "format_field": "json",
    }

    sources = read_sources()
    for src in sources:
        if src.type == "API":
            dbg(f"opened source {src.name} as api (parser='{src.name}'), sending request..")
            status, data, built = send_request(src, settings)
            dbg(f"sent request, responded {status}")
            if status == 200 and data is not None:
                try:
                    _ = PARSERS.get(src.name)
                    if _:
                        _. __call__  # noqa: touch
                except Exception:
                    pass
            # parse regardless; the parser guards internally
            if data is not None:
                parse_fn = PARSERS.get(src.name)
                if parse_fn:
                    parse_fn(src, data)
        else:
            dbg(f"opened source {src.name}, has no api, starting process..")
            handle_result_no_api(src, query, subj, hard_image_cap=20)

    # (Optional) DDG kept as-is but may rate-limit; you can comment out if not needed now
    # try:
    #     dbg(f"[DDG] query='{subj} {query} diagram' requesting up to {100} results")
    #     with DDGS() as ddgs:
    #         results = ddgs.images(f"{subj} {query} diagram", max_results=100, safesearch="moderate")
    #     # your DDG CC evidence flow would go here if you want it in this run
    # except Exception as e:
    #     dbg(f"[DDG] skipped due to: {e}")
