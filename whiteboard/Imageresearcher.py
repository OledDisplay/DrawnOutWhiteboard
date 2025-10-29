import os, json, copy, requests, re, hashlib, datetime
from urllib.parse import urlparse, unquote, urljoin
from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup ,NavigableString, Tag
import tldextract
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple
from pathlib import Path
from typing import Optional
import time

from duckduckgo_search import DDGS

# ========= CONFIG & DEBUG =========
DEBUG = True
def dbg(*args):
    if DEBUG:
        print(*args, flush=True)

# FIXED: use a raw string for Windows path to avoid escape issues
SOURCE_PATH = r'C:\Users\marti\Code\DrawnOutWhiteboard\whiteboard\source_urls'
IMAGES_PATH = r'C:\Users\marti\Code\DrawnOutWhiteboard\whiteboard\ResearchImages'

SUBJECTS = [
    "Maths",
    "Physics",
    "Biology",  #Includes anatomy
    "Chemistry",
    "Geography",
]

# ----- FOR DUCKDUCKGO SEARCH -----

# Creative Commons / Public Domain evidence patterns
CC_PATTERNS = [
    r"creativecommons\.org/licenses/([a-z\-0-9/\.]+)",
    r"creativecommons\.org/publicdomain/([a-z\-0-9/\.]+)",
    r"\bCC\s?BY(?:-SA|-NC|-ND)?\s?(?:\d\.\d)?\b",
    r"\bCC0\b",
    r"\bCreative\s+Commons\b.*?(?:Attribution|ShareAlike|NonCommercial|NoDerivatives|Zero|Public\s+Domain)",
    r"\bPublic\s+Domain\b",
]

# Social/media hosts we want to exclude
BLOCKED_HOSTS = {
    "twitter.com", "x.com", "facebook.com", "instagram.com", "tiktok.com",
    "pinterest.com", "pinimg.com", "reddit.com", "imgur.com", "tumblr.com"
}

# Image extension heuristic
IMG_EXT_RE = re.compile(r"\.(png|jpe?g|gif|webp|svg|tiff?)($|\?)", re.I)

# User-Agent for polite crawling
UA = {"User-Agent": "diag-scrape/0.2 (+research/cc-check; contact: your-email@example.com)"}

# ======= Sources =======

class Source():
    def __init__(self, json_obj, name):
        self.name = name  # e.g., "openverse" from openverse.json
        self.url = json_obj.get("url")
        if json_obj.get("HasApi") == "Y":
            self.TemplateEQ = json_obj.get("TemplateEquivalent")
            self.Template = json_obj.get("FullTemplate")
            self.type = "API"
        else:
            self.type = "NORMAL"
            self.subjecturls = json_obj.get("SubjectUrls")
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
    """
    Clone the source FullTemplate and overwrite the API-specific parameter names
    using TemplateEquivalent mappings fed with values from 'settings'.
    """
    params = copy.deepcopy(source.Template)

    for k, v in settings.items():
        # Skip pagination_field unless we actually have a token value (non-empty, non-sentinel)
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
        return None, None, built  # nothing to send; you still get the built params

    headers = {"User-Agent": "diag-scrape/0.1"}
    try:
        dbg(f"[API][{source.name}] GET {source.url} params={built}")
        resp = requests.get(source.url, params=built, headers=headers, timeout=20)
        ct = resp.headers.get("Content-Type", "")
        dbg(f"[API][{source.name}] status={resp.status_code} len={len(resp.content)} ct={ct}")
        # Try JSON first; fall back to text
        try:
            data = resp.json()
        except ValueError:
            data = resp.text
        return resp.status_code, data, built
    except requests.RequestException as e:
        dbg(f"[API][{source.name}] REQUEST_ERROR: {e}")
        return None, f"REQUEST_ERROR: {e}", built

def handle_result_api(source, data):
    parser = PARSERS.get(source.name)
    if not parser:
        print(f"No parser found for {source.name}")
        return []
    dbg(f"[API] parsing data with parser='{source.name}' for source='{source.name}'")
    return parser(source, data)

# --------------- Synonym finder ---------------
def get_limited_lemmas(word, per_synset_limit=4, pos_filter=None):
    """
    Get all lemma names (synonyms) for a given word, limiting the number per synset.
    """
    if isinstance(pos_filter, str):
        pos_filter = [pos_filter]

    synsets = wn.synsets(word)
    if pos_filter:
        synsets = [s for s in synsets if s.pos() in pos_filter]

    results = {}
    for syn in synsets:
        lemmas = syn.lemmas()
        limited = [lemma.name().replace('_', ' ') for lemma in lemmas[:per_synset_limit]]
        results[syn.name()] = {
            "definition": syn.definition(),
            "lemmas": limited
        }
    return results

# --------------- SearchFor helpers ---------------
# =========================
# STAGE 1: build phrase list (query + synonyms)
# =========================
def _stage1_phrases(query: str) -> List[str]:
    def _flatten_syns(synmap: dict) -> List[str]:
        out, seen = [], set()
        for payload in synmap.values():
            for lem in payload.get("lemmas", []):
                s = lem.strip()
                if s:
                    k = s.lower()
                    if k not in seen:
                        seen.add(k); out.append(s)
        return out

    synmap = {}
    try:
        synmap = get_limited_lemmas(query, per_synset_limit=5, pos_filter=None)
    except NameError:
        pass  # if wordnet not available

    synonyms = _flatten_syns(synmap)
    phrases, seen = [], set()

    def add(p: str):
        k = p.lower().strip()
        if k and k not in seen:
            seen.add(k); phrases.append(p)

    add(query)
    for s in synonyms:
        add(s)
    dbg(f"[PHRASES] query='{query}' → phrases ({len(phrases)}): {phrases[:12]}{' ...' if len(phrases)>12 else ''}")
    return phrases

# =========================
# STAGE 2: fetch pages + find exact phrase hits
# =========================
def _stage2_hits(valid_map: Dict[str, List[str]], phrases: List[str], session: requests.Session):
    def _compile_exact(phrase: str) -> re.Pattern:
        esc = r"\b" + re.sub(r"\s+", " ", re.escape(phrase)).replace(r"\ ", " ") + r"\b"
        return re.compile(esc, re.IGNORECASE)

    phrase_rx = [_compile_exact(p) for p in phrases]

    def _fetch(url: str):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            return soup
        except requests.RequestException:
            return None

    # flatten/dedup URL list
    all_urls, seen = [], set()
    for root, kids in (valid_map or {}).items():
        for u in [root, *(kids or [])]:
            if u not in seen:
                seen.add(u); all_urls.append(u)

    dbg(f"[HITS] total URLs to scan={len(all_urls)}")
    results = []
    pages_with_hits = 0
    for url in all_urls:
        soup = _fetch(url)
        if not soup:
            continue
        hits = []
        for rx in phrase_rx:
            for node in soup.find_all(string=rx):
                if isinstance(node, NavigableString) and node.strip():
                    hits.append(node)
        if hits:
            pages_with_hits += 1
            dbg(f"[HITS] url={url} hits={len(hits)}")
            results.append({"page_url": url, "soup": soup, "hits": hits})
    dbg(f"[HITS] pages with hits={pages_with_hits}")
    return results

# =========================
# STAGE 3: find nearest image for each hit + download
# =========================
def _stage3_download_images(hit_pages, session: requests.Session) -> List[str]:
    dest_dir = os.path.join(IMAGES_PATH, "domain_search")
    os.makedirs(dest_dir, exist_ok=True)

    saved_paths: List[str] = []
    seen_img_urls: Set[str] = set()

    def _pick_img_src(img: Tag, page_url: str) -> str | None:
        def absu(x): return urljoin(page_url, x) if x else None

        for k in ("src", "data-src", "data-original"):
            v = img.get(k)
            if v: return absu(v)

        for k in ("srcset", "data-srcset"):
            sv = img.get(k)
            if not sv: continue
            best_u, best_w = None, -1
            for part in [p.strip() for p in sv.split(",") if p.strip()]:
                chunks = part.split()
                if not chunks: continue
                cu = absu(chunks[0])
                w = -1
                for c in chunks[1:]:
                    m = re.match(r"(\d+)w$", c)
                    if m: w = int(m.group(1)); break
                if w > best_w:
                    best_w, best_u = w, cu
            if best_u: return best_u

        if img.parent and isinstance(img.parent, Tag) and img.parent.name == "picture":
            src = img.parent.find("source")
            if src:
                return absu(src.get("srcset") or src.get("src"))
        return None

    def _nearest_img(node: NavigableString | Tag, page_url: str) -> str | None:
        start = node.parent if isinstance(node, NavigableString) else node
        if not start: return None
        q, visited = [start], {id(start)}
        max_nodes = 600

        while q and len(visited) <= max_nodes:
            cur = q.pop(0)
            if getattr(cur, "name", None) == "img":
                src = _pick_img_src(cur, page_url)
                if src: return src
            img = cur.find("img") if isinstance(cur, Tag) else None
            if img:
                src = _pick_img_src(img, page_url)
                if src: return src

            if isinstance(cur, Tag):
                for ch in cur.children:
                    if isinstance(ch, Tag) and id(ch) not in visited:
                        visited.add(id(ch)); q.append(ch)
                for sib in (cur.previous_sibling, cur.next_sibling, cur.parent):
                    if isinstance(sib, Tag) and id(sib) not in visited:
                        visited.add(id(sib)); q.append(sib)
        return None

    def _safe_name(u: str) -> str:
        parsed = urlparse(u)
        base = unquote(os.path.basename(parsed.path)) or "image"
        base = re.sub(r"[?#].*$", "", base)
        root, ext = os.path.splitext(base)
        if not ext: ext = ".img"
        h = hashlib.sha1(u.encode("utf-8")).hexdigest()[:10]
        return f"{root}_{h}{ext}"

    def _download(u: str) -> str | None:
        try:
            h = session.head(u, allow_redirects=True, timeout=12)
            if not h or h.status_code >= 400 or not (h.headers.get("Content-Type","").lower().startswith("image/")):
                g = session.get(u, stream=True, timeout=30)
                g.raise_for_status()
                ct = g.headers.get("Content-Type","").lower()
                if not ct.startswith("image/"): return None
                fname = _safe_name(u); path = os.path.join(dest_dir, fname)
                with open(path, "wb") as f:
                    for chunk in g.iter_content(8192):
                        if not chunk: break
                        f.write(chunk)
                return path
            else:
                g = session.get(u, stream=True, timeout=30)
                g.raise_for_status()
                fname = _safe_name(u); path = os.path.join(dest_dir, fname)
                with open(path, "wb") as f:
                    for chunk in g.iter_content(8192):
                        if not chunk: break
                        f.write(chunk)
                return path
        except requests.RequestException:
            return None

    total_hits = sum(len(item["hits"]) for item in hit_pages)
    dbg(f"[IMG] total hits scanned={total_hits}")
    for item in hit_pages:
        page_url, hits = item["page_url"], item["hits"]
        for node in hits:
            img_url = _nearest_img(node, page_url)
            if not img_url or img_url in seen_img_urls:
                continue
            local = _download(img_url)
            if local:
                seen_img_urls.add(img_url)
                saved_paths.append(local)
                dbg(f"[IMG] saved {local} from {page_url}")
    dbg(f"[IMG] images saved={len(saved_paths)}")
    return saved_paths

# =========================
# ORCHESTRATOR
# =========================
def search_for(valid_map: Dict[str, List[str]], query: str) -> List[str]:
    dbg(f"[SEARCH] starting search_for; urls={sum(1+len(v) for v in (valid_map or {}).values())} query='{query}'")
    session = requests.Session()
    session.headers.update({"User-Agent": "diag-scrape/0.1"})
    phrases   = _stage1_phrases(query)  # 1) synonyms -> phrases
    hit_pages = _stage2_hits(valid_map, phrases, session)  # 2) pages -> text hits
    paths     = _stage3_download_images(hit_pages, session) # 3) hits -> nearest images (download)
    dbg(f"[SEARCH] finished search_for; images={len(paths)}")
    return paths  # 4) return literal paths

# ------- FIND ALL THE URLS TO USE IN SEARCH ---

# --- JS rendering / robust HTML fetch (Playwright first, fallbacks) ---
def js_capable_fetch(url: str, timeout_ms: int = 15000, wait_selector: str | None = "a[href]"):
    """
    Fetch HTML and (if possible) the anchor hrefs from a page that may require JS.
    Prefer Playwright to render. Falls back to plain requests on failure.

    Returns: (html_text: str|None, anchor_urls: list[str], dbg_reason: str)
    """
    # 1) Try Playwright (preferred)
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context()
            page = ctx.new_page()
            page.set_default_timeout(timeout_ms)
            page.goto(url, wait_until="domcontentloaded")  # fast first paint

            # settle network; don't hang forever
            try:
                page.wait_for_load_state("networkidle", timeout=timeout_ms // 2)
            except Exception:
                pass

            # optionally wait for link nodes to appear
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, state="attached", timeout=timeout_ms // 2)
                except Exception:
                    pass

            html = page.content()
            anchors = []
            try:
                els = page.query_selector_all("a[href]")
                for el in els:
                    href = el.get_attribute("href")
                    if href:
                        anchors.append(urljoin(url, href))
            except Exception:
                pass

            browser.close()
            print(f"[JSFETCH][playwright] {url} ok anchors={len(anchors)} html_len={len(html)}")
            return html, anchors, "playwright"
    except ImportError:
        print("[JSFETCH] Playwright not installed; falling back to requests")
    except Exception as e:
        print(f"[JSFETCH][playwright] failed for {url}: {e}")

    # 2) Plain requests fallback (no JS)
    try:
        r = requests.get(url, headers={"User-Agent": "subfinder/2.1"}, timeout=15, allow_redirects=True)
        if r.ok and "text/html" in (r.headers.get("Content-Type") or "").lower():
            soup = BeautifulSoup(r.text, "html.parser")
            anchors = []
            for a in soup.find_all("a", href=True):
                anchors.append(urljoin(url, a["href"]))
            print(f"[JSFETCH][requests] {url} ok anchors={len(anchors)} html_len={len(r.text)}")
            return r.text, anchors, "requests"
    except Exception as e:
        print(f"[JSFETCH][requests] failed for {url}: {e}")

    # 3) total failure

    return None, [], "none"

def find_valid_urls_and_trees(
    base_url: str,
    lemma_obj: dict,
    max_pages_stage1: int = 80,
    max_pages_stage2_per_root: int = 220,
    timeout: int = 8,
    known_roots: List[str] | None = None,
):
    """
    Two-stage crawl (JS-aware via js_capable_fetch).

    Stage 1 (same registrable domain) – runs ONLY when known_roots is empty:
      • Start from base_url and BFS only within the same registrable domain.
      • For the FIRST page (base), force JS render (js_capable_fetch) to expose SPA menus.
      • For subsequent same-site pages: try fast requests; if 0 links, fallback to js_capable_fetch.
      • Build valid_roots by checking if any lemma phrase is a substring of the URL.

    Stage 2 (per valid root, any domain):
      • BFS outward from each valid root (no domain restriction).
      • For each page: try fast requests; if 0 links found, fallback to js_capable_fetch.
      • Build valid_map[root] = sorted(outward links)

    Returns:
      Dict[str, List[str]]  # root -> outward URLs
    """
    if known_roots is None:
        known_roots = []

    # ---------- tiny local helpers ----------
    def dbg(msg: str):
        print(msg)

    def clean_url(u: str) -> str:
        if not u:
            return u
        u = unquote(u)
        u = re.sub(r"#.*$", "", u)  # strip fragments
        u = re.sub(r"[?&](utm_[^=&]+|fbclid|gclid)=[^&]*", "", u)  # strip trackers
        return u

    def is_html_response(resp) -> bool:
        if not resp or not getattr(resp, "ok", False):
            return False
        ct = (resp.headers.get("Content-Type") or "").lower()
        return "text/html" in ct

    headers = {"User-Agent": "subfinder/2.1"}

    # ---------- build lemma phrase set ----------
    lemma_phrases = {
        (w or "").strip().lower()
        for entry in (lemma_obj or {}).values()
        for w in (entry.get("lemmas") or [])
        if isinstance(w, str) and len(w.strip()) > 1
    }

    # ---------- base host / registrable ----------
    parsed_base = urlparse(base_url)
    base_host = parsed_base.hostname or ""
    if base_host:
        b = tldextract.extract(base_host)
        registrable = f"{b.domain}.{b.suffix}" if b.suffix else b.domain
    else:
        registrable = ""

    if not base_host and not known_roots:
        dbg("[FIND] no base_host and no known_roots → empty map")
        return {}

    dbg(f"[FIND] base_url={base_url} registrable={registrable} "
        f"known_roots={len(known_roots)} lemma_phrases={len(lemma_phrases)}")

    # ====================================================
    # STAGE 1 — same-site BFS to discover candidate roots
    # ====================================================
    valid_roots: List[str] = []
    if not known_roots:
        if not registrable:
            dbg("[FIND][S1] no registrable domain → empty map")
            return {}

        queue = deque([clean_url(base_url)])
        visited_stage1: Set[str] = set()
        same_site_urls: Set[str] = set()

        dbg(f"[FIND][S1] START queue={[base_url]} max_pages={max_pages_stage1}")
        while queue and len(visited_stage1) < max_pages_stage1:
            cur = clean_url(queue.popleft())
            if cur in visited_stage1:
                continue
            visited_stage1.add(cur)

            # For the base (first) page, use JS fetch to expose nav links in SPAs.
            if len(visited_stage1) == 1:
                html, anchors_js, mode = js_capable_fetch(cur, timeout_ms=timeout * 1000)
                if not html:
                    dbg(f"[FIND][S1] JS fetch failed for base page: {cur}")
                    continue
                candidate_links = anchors_js[:]  # already absolute from js_capable_fetch
                if not candidate_links:
                    # Fallback: parse anchors from HTML just in case
                    soup = BeautifulSoup(html, "html.parser")
                    candidate_links = [urljoin(cur, a["href"]) for a in soup.find_all("a", href=True)]
                dbg(f"[FIND][S1][BASE] mode={mode} links={len(candidate_links)} cur={cur}")
            else:
                # Fast path: plain requests first
                candidate_links = []
                try:
                    r = requests.get(cur, headers=headers, timeout=timeout, allow_redirects=True)
                    if is_html_response(r):
                        soup = BeautifulSoup(r.text, "html.parser")
                        candidate_links = [urljoin(cur, a["href"]) for a in soup.find_all("a", href=True)]
                except requests.RequestException:
                    candidate_links = []

                # If none discovered, fallback to JS render
                if not candidate_links:
                    html2, anchors_js2, mode2 = js_capable_fetch(cur, timeout_ms=timeout * 1000)
                    if anchors_js2:
                        candidate_links = anchors_js2
                        dbg(f"[FIND][S1][JS] cur={cur} gained_links={len(candidate_links)} mode={mode2}")

            # Keep only same-registrable-domain links
            added_links = 0
            for child in candidate_links:
                child = clean_url(child)
                host = urlparse(child).hostname or ""
                if not host:
                    continue
                ext = tldextract.extract(host)
                child_reg = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
                if child_reg != registrable:
                    continue
                if child not in same_site_urls:
                    same_site_urls.add(child)
                    added_links += 1
                if child not in visited_stage1:
                    queue.append(child)

            dbg(f"[FIND][S1] visited={len(visited_stage1)} added_links={added_links} cur={cur}")

        # Choose valid roots: URL contains any lemma phrase
        for url in (same_site_urls or {clean_url(base_url)}):
            ul = url.lower()
            if any(p in ul for p in lemma_phrases):
                valid_roots.append(url)

        valid_roots = sorted(set(valid_roots))
        dbg(f"[FIND][S1] same_site_urls={len(same_site_urls)} valid_roots={len(valid_roots)}")

        if not valid_roots:
            dbg("[FIND][S1] no valid roots → using base_url as a root (fallback)")
            valid_roots = [clean_url(base_url)]
    else:
        valid_roots = sorted({clean_url(u) for u in known_roots if u})
        dbg(f"[FIND][S1] bypassed via known_roots={len(valid_roots)}")

    # ===================================================
    # STAGE 2 — per-root outward exploration (any domain)
    # ===================================================
    valid_map: Dict[str, List[str]] = {}
    fetched_html_cache: Dict[str, Optional[str]] = {}

    def fetch_fast_then_js(url: str) -> List[str]:
        """Return absolute anchors from url. Try requests → fallback to js_capable_fetch if 0 links."""
        # Cache: only cache HTML; anchors depend on parsing anyway
        # 1) fast requests
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            if is_html_response(r):
                soup = BeautifulSoup(r.text, "html.parser")
                links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
                if links:
                    return links
        except requests.RequestException:
            pass

        # 2) JS fallback
        html_js, anchors_js, mode_js = js_capable_fetch(url, timeout_ms=timeout * 1000)
        if anchors_js:
            dbg(f"[FIND][S2][JS] {url} links={len(anchors_js)} mode={mode_js}")
            return anchors_js
        return []

    for idx, root in enumerate(valid_roots, start=1):
        frontier = deque([root])
        visited_root: Set[str] = set()
        outward: Set[str] = set()

        dbg(f"[FIND][S2] root[{idx}/{len(valid_roots)}]={root} START frontier=1 limit={max_pages_stage2_per_root}")
        while frontier and len(visited_root) < max_pages_stage2_per_root:
            cur = clean_url(frontier.popleft())
            if cur in visited_root:
                continue
            visited_root.add(cur)

            links = fetch_fast_then_js(cur)
            push = 0
            for nxt in links:
                nxt = clean_url(nxt)
                if not nxt or nxt in visited_root:
                    continue
                if nxt not in outward:
                    outward.add(nxt)
                    push += 1
                frontier.append(nxt)

            if len(visited_root) % 20 == 0 or push > 40:
                dbg(f"[FIND][S2] root={root} visited={len(visited_root)} outward={len(outward)} "
                    f"frontier={len(frontier)} last_added={push}")

        valid_map[root] = sorted(outward)
        dbg(f"[FIND][S2] root={root} DONE visited={len(visited_root)} outward={len(outward)}")

    dbg(f"[FIND] DONE roots={len(valid_roots)} total_maps={len(valid_map)}")
    return valid_map



# ------- DUCKDUCKGO RESEARCHER -------
def research_cc_images(subject: str, query: str, max_images: int = 10, ddg_max: int = 100) -> List[str]:
    """
    DDG images → open origin page → verify CC/PD evidence → download chosen image.
    """
    def _slugify(s: str, max_len: int = 60) -> str:
        s = "".join(ch if ch.isalnum() or ch in "-_ " else " " for ch in s.lower())
        s = "-".join(filter(None, s.split()))
        return s[:max_len] if s else "query"

    def _download_and_save(img_url: str, base_dir: Path, size_cap_mb: int = 15) -> Optional[str]:
        try:
            with requests.get(img_url, headers=UA, stream=True, timeout=20) as r:
                r.raise_for_status()
                ctype = (r.headers.get("Content-Type") or "").lower()
                if not ctype.startswith("image/") and not IMG_EXT_RE.search(img_url):
                    return None
                limit = size_cap_mb * 1024 * 1024
                sha1 = hashlib.sha1()
                chunks, total = [], 0
                for ch in r.iter_content(8192):
                    if not ch:
                        break
                    chunks.append(ch)
                    total += len(ch)
                    if total > limit:
                        return None
                    sha1.update(ch)
                data = b"".join(chunks)
                ext = None
                if ctype.startswith("image/"):
                    ext = "." + ctype.split("/", 1)[1].split(";")[0].strip()
                    if ext == ".jpeg": ext = ".jpg"
                if not ext:
                    m = IMG_EXT_RE.search(img_url)
                    ext = "." + (m.group(1).lower() if m else "img")
                digest = sha1.hexdigest()
                subdir = base_dir / digest[:2] / digest[2:4]
                subdir.mkdir(parents=True, exist_ok=True)
                out = subdir / f"{digest}{ext}"
                if not out.exists():
                    with open(out, "wb") as f:
                        f.write(data)
                return str(out)
        except Exception:
            return None

    full_q = f"{subject} {query} diagram".strip()
    slug = _slugify(f"{subject}-{query}-diagram")
    stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    batch_root = Path(IMAGES_PATH) / f"research_{slug}_{stamp}"
    batch_root.mkdir(parents=True, exist_ok=True)

    kept_paths: List[str] = []
    seen_pages = set()

    dbg(f"[DDG] query='{full_q}' requesting up to {ddg_max} results")
    with DDGS() as ddgs:
        results = ddgs.images(full_q, max_results=ddg_max, safesearch="moderate")

    for idx, res in enumerate(results, start=1):
        if len(kept_paths) >= max_images:
            break

        page_url = (res.get("url") or "").strip()
        expected_serp_img = (res.get("image") or "").strip()

        if not page_url:
            continue

        try:
            host = (urlparse(page_url).hostname or "").lower()
            if any(host.endswith(b) for b in BLOCKED_HOSTS):
                dbg(f"[DDG] skip blocked host {host}")
                continue
        except Exception:
            continue

        if page_url in seen_pages:
            continue
        seen_pages.add(page_url)

        time.sleep(0.6)

        try:
            r = requests.get(page_url, headers=UA, timeout=15)
            if r.status_code != 200 or "text/html" not in (r.headers.get("Content-Type") or ""):
                continue
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception:
            continue

        candidates: List[Tuple[Tag, str, int]] = []
        exp_name = os.path.basename(urlparse(expected_serp_img).path).split("?")[0].lower() if expected_serp_img else ""

        for img in soup.select("img"):
            src = img.get("src") or img.get("data-src") or ""
            if not src:
                continue
            full = urljoin(page_url, src)
            try:
                ih = (urlparse(full).hostname or "").lower()
                if any(ih.endswith(b) for b in BLOCKED_HOSTS):
                    continue
            except Exception:
                continue
            try:
                w = int(img.get("width") or 0); h = int(img.get("height") or 0)
                area = w * h
            except Exception:
                area = 0
            candidates.append((img, full, area))

        if not candidates:
            continue

        chosen_tag, chosen_url = None, None
        if exp_name:
            for (tag, full, _) in candidates:
                cand = os.path.basename(urlparse(full).path).split("?")[0].lower()
                if cand and cand == exp_name:
                    chosen_tag, chosen_url = tag, full
                    break
        if not chosen_url:
            chosen_tag, chosen_url = max(candidates, key=lambda t: t[2])[:2]

        node = chosen_tag
        for _ in range(4):
            if node and node.parent:
                node = node.parent
        local_soup = BeautifulSoup(str(node), "html.parser") if node else soup

        def _extract_cc_evidence(_soup: BeautifulSoup) -> List[Tuple[str, str]]:
            evidences = []
            for a in _soup.select('a[rel~="license"]'):
                href = a.get("href", ""); text = (a.get_text(" ", strip=True) or "").strip()
                evidences.append((text or "rel=license", href))
            for a in _soup.select("a[href]"):
                href = a.get("href") or ""
                if "creativecommons.org" in href:
                    text = (a.get_text(" ", strip=True) or "").strip()
                    evidences.append((text or "cc-link", href))
            for meta in _soup.select('meta[name="license"], meta[property="license"]'):
                content = meta.get("content") or meta.get("value") or ""
                if content:
                    evidences.append(("meta:license", content))
            for script in _soup.select('script[type="application/ld+json"]'):
                try:
                    data = json.loads(script.string or "")
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if isinstance(item, dict) and item.get("license"):
                            lic = item["license"]
                            evidences.append(("jsonld:license", lic if isinstance(lic, str) else json.dumps(lic)))
                except Exception:
                    pass
            text = _soup.get_text(" ", strip=True)
            for pat in CC_PATTERNS:
                m = re.search(pat, text, re.I)
                if m:
                    evidences.append(("text:cc", m.group(0)))
            return evidences

        evidence_found = None
        near_evs = _extract_cc_evidence(local_soup)
        if near_evs:
            near_evs.sort(key=lambda e: ("creativecommons.org" not in (e[1] if len(e) > 1 else ""), 0))
            ev = near_evs[0]
            evidence_found = f"{ev[0]} -> {ev[1]}" if len(ev) > 1 else ev[0]
        if not evidence_found:
            page_evs = _extract_cc_evidence(soup)
            for t, href in page_evs:
                blob = f"{t} {href}"
                if re.search(r"creativecommons\.org|CC\s?BY|CC0|Public\s+Domain", blob, re.I):
                    evidence_found = f"{t} -> {href}" if href else t
                    break

        if not evidence_found or not re.search(r"creativecommons\.org|CC\s?BY|CC0|Public\s+Domain", evidence_found, re.I):
            dbg(f"[DDG] no CC/PD evidence on {page_url}")
            continue

        saved = _download_and_save(chosen_url, batch_root)
        if saved:
            kept_paths.append(saved)
            dbg(f"[DDG] saved {saved} from {page_url}")

    dbg(f"[DDG] total saved={len(kept_paths)}")
    return kept_paths

def handle_result_no_api(source, query, subj):
    all_img_paths = []
    valid_endpoints = []
    if getattr(source, "subjecturls", None):  # predefined subject urls
        dbg("subject urls exist")
        for current_subj, url in source.subjecturls.items():
            if current_subj == subj and url:
                valid_endpoints.append(url)

        if valid_endpoints:
            dbg(f"[NOAPI] using known endpoints count={len(valid_endpoints)} for subj={subj}")
            # base_url can be the site root; Stage 1 will be skipped because known_roots is provided
            valid_map = find_valid_urls_and_trees(
                base_url=source.url,
                lemma_obj=get_limited_lemmas(subj, 4),
                max_pages_stage1=60,
                max_pages_stage2_per_root=160,
                timeout=8,
                known_roots=valid_endpoints
            )
            all_img_paths = search_for(valid_map, query)
            source.img_paths = all_img_paths
            return all_img_paths
        else:
            dbg("no valid endpoints")

    dbg(f"starting url finding process for {source.name}")
    to_search = find_valid_urls_and_trees(source.url, get_limited_lemmas(subj, 4), 80, 220, 8)
    dbg("got urls, proceeding to search ")
    all_img_paths = search_for(to_search, query)
    dbg("search finished, saving.. ")
    source.img_paths = all_img_paths
    return all_img_paths

# ---------------------------
# vibe coded parsers (REAL)
# ---------------------------
def parse_wikimedia(src, data):
    img_urls = []
    if isinstance(data, dict):
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue
            url = info_list[0].get("url")
            if url:
                img_urls.append(url)
    if not img_urls:
        src.img_paths = []
        return []

    base_name = os.path.splitext(src.name)[0] if getattr(src, "name", None) else "wikimedia_batch"
    dest_dir = os.path.join(IMAGES_PATH, "wikimedia", base_name)
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
            dbg(f"[PARSER][wikimedia] saved {local_path}")
        except Exception:
            continue

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
                is_image = any(mt in ctype for mt in (
                    "image/png", "image/jpeg", "image/jpg", "image/gif", "image/svg", "image/webp", "image/tiff"
                ))
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

# Build PARSERS after real functions exist (prevents stub-call errors)
PARSERS = {
    "wikimedia": parse_wikimedia,
    "openverse": parse_openverse,
    "plos": parse_plos,
    "usgs": parse_usgs
}

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    subj = "Biology"
    query = "Cell structure"

    # IMPORTANT: Provide sane defaults; per-source TemplateEquivalent controls the mapping.
    # For Openverse (page), use an integer (1). For PLOS (start), use an integer offset (0).
    settings = {
        "query_field": query,
        "limit_field": 10,
        "pagination_field": 1,   # will become 'page'=1 for Openverse, 'start'=1 for PLOS, etc.
        "format_field": "json",  # ignored for Openverse (no mapping), used by APIs that map 'format'
    }

    sources = read_sources()
    for src in sources:
        if src.type == "API":
            dbg(f"opened source {src.name} as api (parser='{src.name}'), sending request..")
            status, data, built = send_request(src, settings)
            dbg(f"sent request, responded {status}")
            handle_result_api(src, data)
        else:
            dbg(f"opened source {src.name}, has no api, starting process..")
            handle_result_no_api(src, query, subj)

    # generic web search as a sort of fallback AND because results are good
    paths_duck = research_cc_images(subj, query, 10, 100)



         



#TO DO: ADD HARDCAP ON ADDED LINKS IN SUBDOMAINS -> MORE THAN ONE OR TWO, AS RANDOM MATCHES HAPPEN AND SLOW US DOWN
# REDO FILTERED_SUBDOMAINS - NOT ENTIRELY BY HAND, BOOTSRATP
# TRY!!! TO REDO VIBE CODED SEARCH - ITS HELLA BIG. 



#####################
#vibe coded parsers - NOT worth coding on hand

def parse_wikimedia(src, data):
    """
    Parse Wikimedia Commons API JSON, download all images to IMAGES_PATH,
    and store their local file paths in src.img_paths (list of strings).
    """
    img_urls = []

    # 1) Collect image URLs from the API payload
    if isinstance(data, dict):
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue
            url = info_list[0].get("url")
            if url:
                img_urls.append(url)

    # If nothing to do, set empty list and return
    if not img_urls:
        src.img_paths = []
        return []

    # 2) Prepare destination folder: IMAGES_PATH/wikimedia/<source_name_without_ext>/
    base_name = os.path.splitext(src.name)[0] if getattr(src, "name", None) else "wikimedia_batch"
    dest_dir = os.path.join(IMAGES_PATH, "wikimedia", base_name)
    os.makedirs(dest_dir, exist_ok=True)

    saved_paths = []

    # 3) Helper to build a safe local filename from URL
    def build_filename(u, idx):
        parsed = urlparse(u)
        # Wikimedia file names often in the last path segment
        fname = os.path.basename(parsed.path) or f"file_{idx}"
        fname = unquote(fname)
        # Strip query fragments if any leaked into name
        fname = re.sub(r"[?#].*$", "", fname)
        if not os.path.splitext(fname)[1]:
            # Default to .svg if MIME unknown (Commons often uses SVG/PNG)
            fname = f"{fname}.img"
        # De-duplicate if exists
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

    # 4) Download each image
    headers = {"User-Agent": "diag-scrape/0.1"}
    for i, u in enumerate(img_urls):
        local_path = build_filename(u, i)
        try:
            with requests.get(u, headers=headers, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            saved_paths.append(local_path)
            print(f"[WIKI] saved -> {local_path}")
        except Exception as e:
            print(f"[WIKI] download fail url={u} err={e}")
            continue

    # 5) Attach to the Source object and return
    src.img_paths = saved_paths
    return saved_paths


def parse_openverse(src, data):
    """
    Parse Openverse API JSON, download all images to IMAGES_PATH,
    and store their local file paths in src.img_paths (list of strings).
    """

    img_urls = []

    # 1) Collect image URLs from the API payload
    if isinstance(data, dict):
        for item in data.get("results", []):
            u = item.get("url")
            if u:
                img_urls.append(u)

    # If nothing to do, set empty list and return
    if not img_urls:
        src.img_paths = []
        return []

    # 2) Prepare destination folder: IMAGES_PATH/openverse/<source_name_without_ext>/
    base_name = os.path.splitext(src.name)[0] if getattr(src, "name", None) else "openverse_batch"
    dest_dir = os.path.join(IMAGES_PATH, "openverse", base_name)
    os.makedirs(dest_dir, exist_ok=True)

    saved_paths = []

    # 3) Helper to build a safe local filename from URL
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

    # 4) Download each image
    headers = {"User-Agent": "diag-scrape/0.1"}
    for i, u in enumerate(img_urls):
        local_path = build_filename(u, i)
        try:
            with requests.get(u, headers=headers, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            saved_paths.append(local_path)
            print(f"[OPENVERSE] saved -> {local_path}")
        except Exception as e:
            print(f"[OPENVERSE] download fail url={u} err={e}")
            continue

    # 5) Attach to the Source object and return
    src.img_paths = saved_paths
    return saved_paths

def parse_plos(src, data):
    """
    PLOS parser:
      1) Read the PLOS search JSON (article-level metadata).
      2) Take top 5 docs (by returned order).
      3) For each DOI, resolve via https://doi.org/<doi> (follow redirects to the correct PLOS journal URL).
      4) Parse the article HTML for figure <img> elements and collect their image URLs.
      5) Download images to IMAGES_PATH/plos/<source_name_without_ext>/<article_idx>/
      6) Save all local file paths to src.img_paths (list of strings), and return the same list.
    """
    import os
    import re
    from urllib.parse import urlparse, unquote, urljoin
    import requests
    from bs4 import BeautifulSoup

    # 1) Collect top-5 DOIs from the PLOS search payload
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

    # 2) Prepare base destination folder: IMAGES_PATH/plos/<batch_name>/
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

    # 3) For each DOI → resolve article URL → parse figures → download images
    for a_idx, doi in enumerate(dois, start=1):
        # Resolve DOI to actual article URL (handles journal subdomain automatically)
        try:
            resp = requests.get(f"https://doi.org/{doi}", headers=headers, timeout=30, allow_redirects=True)
            resp.raise_for_status()
            article_url = resp.url  # final resolved URL at journals.plos.org/...
        except Exception as e:
            print(f"[PLOS] DOI resolve fail doi={doi} err={e}")
            continue

        # Fetch the article HTML
        try:
            art = requests.get(article_url, headers=headers, timeout=30)
            art.raise_for_status()
        except Exception as e:
            print(f"[PLOS] article fetch fail url={article_url} err={e}")
            continue

        soup = BeautifulSoup(art.text, "html.parser")

        # Collect figure image URLs; PLOS places images inside <figure> blocks.
        img_urls = set()

        # Primary: any <figure> contains <img>
        for img in soup.select("figure img"):
            src_attr = img.get("src") or img.get("data-src")
            if not src_attr:
                continue
            # Some src may be relative; join with article_url
            img_urls.add(urljoin(article_url, src_attr))

        # Fallback: some pages may have figure thumbnails linking to full-size via <a href>
        for a in soup.select("figure a[href]"):
            href = a.get("href")
            if href and re.search(r"\.(png|jpe?g|gif|svg)(\?|$)", href, re.IGNORECASE):
                img_urls.add(urljoin(article_url, href))

        if not img_urls:
            print(f"[PLOS] no figure images url={article_url}")
            continue

        # Per-article directory
        article_dir = os.path.join(batch_dir, f"article_{a_idx}")
        os.makedirs(article_dir, exist_ok=True)

        # Download each image
        for i, u in enumerate(sorted(img_urls)):
            local_name = safe_filename_from_url(u, i)
            local_path = os.path.join(article_dir, local_name)
            # De-duplicate if exists
            if os.path.exists(local_path):
                saved_paths.append(local_path)
                continue
            try:
                with requests.get(u, headers=headers, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                saved_paths.append(local_path)
                print(f"[PLOS] saved -> {local_path}")
            except Exception as e:
                print(f"[PLOS] download fail url={u} err={e}")
                continue

    src.img_paths = saved_paths
    return saved_paths


def parse_usgs(src, data):
    """
    USGS ScienceBase parser:
      - Walk 'items' -> ('files' OR 'attachments') and collect image URLs.
      - Download to IMAGES_PATH/usgs/<source_name_without_ext>/item_<n>/
      - Save local file paths in src.img_paths and return the list.
    """
    import os
    import re
    from urllib.parse import urlparse, unquote, urljoin
    import requests

    # 1) Collect candidate image URLs from the payload
    img_tasks = []  # list of (item_index, url)

    if isinstance(data, dict):
        items = data.get("items", [])
        for idx, item in enumerate(items, start=1):
            # Prefer 'files', fall back to 'attachments'
            file_entries = item.get("files") or item.get("attachments") or []
            base_url = item.get("link") or item.get("self", {}).get("linkUrl") or ""

            for f in file_entries:
                # USGS ScienceBase uses 'url', sometimes 'downloadUri'
                u = f.get("url") or f.get("downloadUri") or ""
                if not u:
                    # try to resolve relative links against item link if present
                    href = f.get("href") or f.get("uri") or ""
                    if href:
                        u = urljoin(base_url, href)

                if not u:
                    continue

                ctype = (f.get("contentType") or "").lower()
                # Accept common image MIME types; if missing, guess from extension
                is_image = any(mt in ctype for mt in (
                    "image/png", "image/jpeg", "image/jpg", "image/gif", "image/svg", "image/webp", "image/tiff"
                ))
                if not is_image:
                    # try extension heuristic if contentType missing/unknown
                    if re.search(r"\.(png|jpe?g|gif|svg|webp|tiff?)($|\?)", u, re.I):
                        is_image = True

                if is_image:
                    img_tasks.append((idx, u))

    # If nothing found, record empty list and return
    if not img_tasks:
        src.img_paths = []
        return []

    # 2) Prepare destination root: IMAGES_PATH/usgs/<batch_name>/
    base_name = os.path.splitext(src.name)[0] if getattr(src, "name", None) else "usgs_batch"
    root_dir = os.path.join(IMAGES_PATH, "usgs", base_name)
    os.makedirs(root_dir, exist_ok=True)

    saved_paths = []

    # 3) Helper to build safe local filenames
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

    # 4) Download
    headers = {"User-Agent": "diag-scrape/0.1"}
    for seq, (item_idx, url) in enumerate(img_tasks):
        local_path = build_filename(url, seq, item_idx)
        try:
            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            saved_paths.append(local_path)
            print(f"[USGS] saved -> {local_path}")
        except Exception as e:
            print(f"[USGS] download fail url={url} err={e}")
            continue

    src.img_paths = saved_paths
    return saved_paths

# Rebuild PARSERS at the very end to bind real implementations
PARSERS = {
    "wikimedia": parse_wikimedia,
    "openverse": parse_openverse,
    "plos": parse_plos,
    "usgs": parse_usgs
}
