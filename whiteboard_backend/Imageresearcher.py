import os, json, copy, requests, re, hashlib, datetime, time, random, shutil
from urllib.parse import urlparse, unquote, urljoin, urlunparse
from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup, NavigableString, Tag
import tldextract
from collections import deque
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from PIL import Image

import threading
import queue
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import base64, gzip


try:
    from ddgs import DDGS  # new name
except Exception:
    from duckduckgo_search import DDGS  # old name (still works)

from ddg_research import ddg_cc_image_harvest


from smart_hits import smart_find_hits_in_soup

@dataclass
class _EmbedJob:
    texts: list[str]
    normalize: bool
    is_single: bool
    done: threading.Event
    result: object = None
    error: Exception | None = None


class _EmbeddingService:
    """
    Single-thread owner of SentenceTransformer to keep it hot and prevent concurrent encode() interference.
    Also batches small requests to cut overhead.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_batch_texts: int = 256, batch_timeout_ms: int = 6):
        self._ok = False
        self._stop = False
        self._q: "queue.Queue[_EmbedJob]" = queue.Queue()
        self._max_batch_texts = int(max_batch_texts)
        self._batch_timeout_ms = int(batch_timeout_ms)

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._ok = True
        except Exception as e:
            self._model = None
            self._ok = False
            print(f"[EMBED] sentence-transformers not available or model load failed: {e}", flush=True)
            return

        self._thr = threading.Thread(target=self._loop, name="MiniLM-Encode-Worker", daemon=True)
        self._thr.start()
        print("[EMBED] MiniLM worker thread started (model is hot).", flush=True)

    def ok(self) -> bool:
        return bool(self._ok and self._model is not None)

    def shutdown(self):
        self._stop = True
        try:
            self._q.put_nowait(_EmbedJob(texts=[""], normalize=False, is_single=True, done=threading.Event()))
        except Exception:
            pass

    def encode(self, sentences, normalize_embeddings: bool = False):
        if not self.ok():
            raise RuntimeError("Embedding service not available")

        # - input is str -> return 1D vector
        # - input is list/tuple -> return 2D array
        is_single = isinstance(sentences, str)
        if is_single:
            texts = [sentences]
        else:
            texts = list(sentences or [])

        ev = threading.Event()
        job = _EmbedJob(texts=texts, normalize=bool(normalize_embeddings), is_single=is_single, done=ev)
        self._q.put(job)
        ev.wait()

        if job.error:
            raise job.error
        return job.result

    def _loop(self):
        import time

        while not self._stop:
            job0 = self._q.get()
            if self._stop:
                break

            jobs = [job0]

            # batch window
            deadline = time.time() + (self._batch_timeout_ms / 1000.0)

            while time.time() < deadline:
                try:
                    j = self._q.get_nowait()
                    jobs.append(j)
                except Exception:
                    break

            for normalize_flag in (False, True):
                group = [j for j in jobs if j.normalize == normalize_flag]
                if not group:
                    continue

                gi = 0
                while gi < len(group):
                    flat = []
                    slices = []
                    # build one batch up to cap
                    while gi < len(group) and len(flat) < self._max_batch_texts:
                        j = group[gi]
                        start = len(flat)
                        flat.extend(j.texts)
                        end = len(flat)
                        slices.append((j, start, end))
                        gi += 1

                        if len(flat) >= self._max_batch_texts:
                            break

                    try:
                        embs = self._model.encode(flat, normalize_embeddings=normalize_flag)
                        for j, s, e in slices:
                            chunk = embs[s:e]
                            j.result = chunk[0] if j.is_single else chunk
                            j.done.set()
                    except Exception as e:
                        for j, _, _ in slices:
                            j.error = e
                            j.done.set()



class _QueuedEncoder:
    """
    Minimal shim so your existing code can keep calling encoder.encode(...).
    """
    def __init__(self, service: _EmbeddingService):
        self._svc = service

    def encode(self, sentences, normalize_embeddings: bool = False):
        return self._svc.encode(sentences, normalize_embeddings=normalize_embeddings)


# Global hot MiniLM
_EMBED_SVC = _EmbeddingService("all-MiniLM-L6-v2", max_batch_texts=256, batch_timeout_ms=6)
_EMBED_MODEL = _QueuedEncoder(_EMBED_SVC) if _EMBED_SVC.ok() else None


from dataclasses import dataclass

@dataclass
class CrawlExhaustion:
    pages_seen: int = 0

    miss_streak: int = 0              # consecutive failures anywhere
    terminal_miss_streak: int = 0     # consecutive failures on terminal pages

    successes: int = 0

    def register(self, accepted: bool, is_terminal: bool) -> None:
        self.pages_seen += 1

        if accepted:
            self.successes += 1
            self.miss_streak = 0
            if is_terminal:
                self.terminal_miss_streak = 0
            return

        # miss
        self.miss_streak += 1
        if is_terminal:
            self.terminal_miss_streak += 1



from ImageRanker import SiglipBackend, rerank_image_candidates_siglip

# -------------------------
# Global SigLIP reuse (lazy)
# -------------------------
_SIGLIP_BACKEND: SiglipBackend | None = None
_SIGLIP_BACKEND_LOCK = threading.Lock()

# If SigLIP inference isn't thread-safe / you want to avoid GPU contention:
_SIGLIP_INFER_LOCK = threading.Lock()

def get_siglip_backend() -> SiglipBackend | None:
    global _SIGLIP_BACKEND
    if _SIGLIP_BACKEND is not None:
        return _SIGLIP_BACKEND

    with _SIGLIP_BACKEND_LOCK:
        if _SIGLIP_BACKEND is not None:
            return _SIGLIP_BACKEND
        try:
            _SIGLIP_BACKEND = SiglipBackend()
            dbg("[SIGLIP] backend initialized (hot).")
        except Exception as e:
            dbg(f"[SIGLIP] backend init failed: {e}")
            _SIGLIP_BACKEND = None

    return _SIGLIP_BACKEND

def trace(*_a, **_k): 
    return

def trace_set_ctx(**_k): 
    return

def trace_clear_ctx(*_a): 
    return

ROOT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = os.path.join(ROOT_DIR, "source_urls")
IMAGES_PATH = os.path.join(ROOT_DIR, "ResearchImages")
os.makedirs(SOURCE_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)

# global image metadata: path -> list of metadata dicts
IMAGE_METADATA: Dict[str, List[dict]] = {}


# =========================
# Terminal bucket files (ROOT storage)
# =========================
TERMINAL_BUCKETS_DIR = os.path.join(ROOT_DIR, "terminal_buckets")
os.makedirs(TERMINAL_BUCKETS_DIR, exist_ok=True)

def _fs_safe(s: str, max_len: int = 80) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    s = s.strip("._-")
    return (s[:max_len] or "x")

def _terminal_bucket_file_abs(registrable: str, subject: str, idx_id: str) -> str:
    reg = _fs_safe(registrable or "site", 90)
    subj = _fs_safe(subject or "default", 90)
    iid = _fs_safe(idx_id or "bucket", 64)
    d = os.path.join(TERMINAL_BUCKETS_DIR, reg, subj)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{iid}.json")

def save_terminal_bucket_file(idx_obj: dict, registrable: str, subject: str) -> str:
    """
    Saves the FULL terminal list to disk in ROOT/terminal_buckets/<reg>/<subject>/<id>.json
    Returns relative path (from ROOT_DIR) to store into Pinecone metadata.
    """
    try:
        entry = clean_url(idx_obj.get("entry_url") or "")
        terminals = idx_obj.get("terminal_urls") or idx_obj.get("terminals") or []
        terminals = [clean_url(u) for u in terminals if u]
        idx_id = (idx_obj.get("id") or "").strip() or hashlib.sha1(
            f"{subject}|{registrable}|{entry}".encode("utf-8")
        ).hexdigest()[:12]

        abs_path = _terminal_bucket_file_abs(registrable, subject, idx_id)

        payload = {
            "id": idx_id,
            "subject": subject,
            "registrable": registrable,
            "entry_url": entry,
            "created_at": idx_obj.get("created_at") or (datetime.datetime.utcnow().isoformat() + "Z"),
            "terminal_count": int(idx_obj.get("terminal_count") or len(terminals)),
            # store url + key (your “last part” embed key)
            "terminals": [{"url": u, "key": _url_key_for_embed(u)} for u in terminals],
        }

        _save_json_atomic(abs_path, payload)
        return os.path.relpath(abs_path, ROOT_DIR).replace("\\", "/")
    except Exception as e:
        dbg(f"[IDX][FILE_SAVE][ERR] {e}")
        return ""

def load_terminal_bucket_urls(rel_path: str) -> list[str]:
    """
    Loads terminal urls from a bucket json (relative to ROOT_DIR).
    Returns list[str].
    """
    try:
        if not rel_path:
            return []
        abs_path = os.path.join(ROOT_DIR, rel_path)
        if not os.path.exists(abs_path):
            return []
        data = _load_json(abs_path) or {}
        terms = data.get("terminals") or []
        out = []
        for it in terms:
            if isinstance(it, dict):
                u = it.get("url")
            else:
                u = it
            u = clean_url(u or "")
            if u:
                out.append(u)
        return ordered_dedupe(out)
    except Exception:
        return []




_IMAGE_META_LOCK = threading.Lock()

def _register_image_metadata(path: str, meta: dict) -> None:
    if not path:
        return
    with _IMAGE_META_LOCK:
        lst = IMAGE_METADATA.get(path)
        if lst is None:
            IMAGE_METADATA[path] = [meta]
        else:
            lst.append(meta)

_JSON_LOCKS_LOCK = threading.Lock()
_JSON_LOCKS: dict[str, threading.Lock] = {}

def _json_lock_for(path: str) -> threading.Lock:
    ap = os.path.abspath(path)
    with _JSON_LOCKS_LOCK:
        lk = _JSON_LOCKS.get(ap)
        if lk is None:
            lk = threading.Lock()
            _JSON_LOCKS[ap] = lk
        return lk

MAX_STAGE2_IMAGES_PER_PROMPT = 10
MAX_WEB_SOURCES_PER_PROMPT = 2

def _prompt_image_paths(query: str, prompt_id: str) -> set[str]:
    """
    Best-effort: return image paths that look like they belong to this prompt.
    Uses metadata. Prefers prompt_id match, falls back to base_context match.
    """
    qnorm = (query or "").strip().casefold()
    out = set()
    try:
        with _IMAGE_META_LOCK:
            for path, metas in (IMAGE_METADATA or {}).items():
                if not metas:
                    continue
                for m in metas:
                    try:
                        if m.get("prompt_id") == prompt_id:
                            out.add(path); break
                        bc = str(m.get("base_context", "") or "").strip().casefold()
                        if qnorm and bc == qnorm:
                            out.add(path); break
                    except Exception:
                        continue
    except Exception:
        pass
    return out




DEBUG = True
def dbg(*args):
    if DEBUG:
        print(*args, flush=True)



WMF_UA = "https://github.com/OledDisplay/DrawnOutWhiteboard"

BASE_HEADERS = {
    "User-Agent": WMF_UA,
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}



# =========================
# LICENSE/HOST FILTERS
# =========================
CC_PATTERNS = [
    r'creativecommons\.org/licenses/[^"\s<]+',
    r'creativecommons\.org/publicdomain/[^"\s<]+',
    r'\bCC\s?BY(?:-SA|-NC|-ND)?\s?(?:\d\.\d)?\b',
    r'\bCC0\b',
    r'\bCreative\s+Commons\b.*?(?:Attribution|ShareAlike|NonCommercial|NoDerivatives|Zero|Public\s+Domain)',
    r'\bPublic\s+Domain\b',
]

IMG_EXT_RE = re.compile(r"\.(png|jpe?g|gif|webp|bmp|tiff?|svg)$", re.I)

UA = {"User-Agent": "diag-scrape/0.2 (+research/cc-check; contact: your-email@example.com)"}

# =========================
# URL BLOCKS / NORMALIZATION
# =========================
BLOCKED_URL_WORDS = [
    "/blog", "/press", "/partners", "/privacy", "/license", "/tos",
    "/accessibility", "/accounts/login", "/facebook", "/twitter",
    "/instagram", "/youtube", "/linkedin", "/help",
    "/rice.edu", "/gatesfoundation.org", "google.com", "business.safety.google",
    "/mailto:", "/tel:", "/pdf", "drive.google.com",
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


def _to_list_embedding(v):
    if v is None:
        return None
    try:
        if hasattr(v, "tolist"):
            return v.tolist()
    except Exception:
        pass
    try:
        return list(v)
    except Exception:
        return None

def _ctx_confidence_from_embeddings(query_emb, ctx_emb) -> float:
    try:
        qv = _norm_vec(query_emb)
        cv = _norm_vec(ctx_emb)
        if qv is None or cv is None:
            return 0.0
        return float(_dot_sim(qv, cv))
    except Exception:
        return 0.0

def _ccish(text: str) -> bool:
    if not text:
        return False
    for pat in CC_PATTERNS:
        try:
            if re.search(pat, text, re.I):
                return True
        except Exception:
            continue
    return False

def parse_openverse(src, data, query=None, prompt_id=None, encoder=None, query_embedding=None, base_ctx_embedding=None, **_):
    results = []
    try:
        if isinstance(data, dict):
            results = data.get("results") or []
    except Exception:
        results = []

    prompt_id = prompt_id or _prompt_key(query or "")
    dest_dir = os.path.join(IMAGES_PATH, "api", "openverse", prompt_id)
    os.makedirs(dest_dir, exist_ok=True)

    session = requests.Session()
    saved = []

    for r in results[:20]:
        try:
            img_url = (r.get("url") or r.get("thumbnail") or r.get("thumbnail_url") or "").strip()
            if not img_url:
                continue

            title = (r.get("title") or "").strip()
            creator = (r.get("creator") or "").strip()
            landing = (r.get("foreign_landing_url") or r.get("detail_url") or "").strip()
            license_url = (r.get("license_url") or "").strip()
            license_code = (r.get("license") or "").strip()

            # If you want CC-only, enforce it here. If not, remove this.
            lic_blob = " ".join([license_code, license_url, title, creator])
            if license_code or license_url:
                if not _ccish(lic_blob) and "public domain" not in lic_blob.lower():
                    continue

            ctx_text = " ".join(x for x in [title, creator, license_code, query] if x).strip()
            ctx_emb = None
            if encoder is not None and ctx_text:
                try:
                    ctx_emb = encoder.encode(ctx_text)
                except Exception:
                    ctx_emb = None

            path = download_image(session, img_url, dest_dir, apply_diagram_filter=False)
            if not path:
                continue

            conf = _ctx_confidence_from_embeddings(query_embedding, ctx_emb)
            meta = {
                "source_kind": "api",
                "source_name": "openverse",
                "base_context": query or "",
                "prompt_id": prompt_id,
                "page_url": landing or "",
                "image_url": img_url,
                "ctx_text": ctx_text,
                "ctx_embedding": _to_list_embedding(ctx_emb),
                "ctx_sem_score": conf,
                "ctx_confidence": conf,
            }
            if base_ctx_embedding is not None:
                meta["prompt_embedding"] = base_ctx_embedding

            _register_image_metadata(path, meta)
            saved.append(path)

            if len(saved) >= 10:
                break
        except Exception:
            continue

    src.img_paths = saved
    return saved

def parse_wikimedia(src, data, query=None, prompt_id=None, encoder=None, query_embedding=None, base_ctx_embedding=None, **_):
    prompt_id = prompt_id or _prompt_key(query or "")
    dest_dir = os.path.join(IMAGES_PATH, "api", "wikimedia", prompt_id)
    os.makedirs(dest_dir, exist_ok=True)

    session = requests.Session()
    saved = []

    pages = {}
    try:
        if isinstance(data, dict):
            pages = (data.get("query") or {}).get("pages") or {}
    except Exception:
        pages = {}

    for _, p in (pages or {}).items():
        try:
            infos = p.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            img_url = (info.get("url") or "").strip()
            if not img_url:
                continue

            ext = info.get("extmetadata") or {}
            # extmetadata entries are often dicts with {"value": "..."}
            def _ext_val(k):
                v = ext.get(k)
                if isinstance(v, dict):
                    return (v.get("value") or "").strip()
                if isinstance(v, str):
                    return v.strip()
                return ""

            lic_short = _ext_val("LicenseShortName")
            lic_url = _ext_val("LicenseUrl")
            desc = _ext_val("ImageDescription")
            objname = _ext_val("ObjectName")
            artist = _ext_val("Artist")
            credit = _ext_val("Credit")
            usage = _ext_val("UsageTerms")

            lic_blob = " ".join([lic_short, lic_url, usage, credit, artist, objname, desc])
            if (lic_short or lic_url or usage) and not _ccish(lic_blob) and "public domain" not in lic_blob.lower():
                continue

            ctx_text = " ".join(x for x in [objname, desc, artist, lic_short, query] if x).strip()
            ctx_emb = None
            if encoder is not None and ctx_text:
                try:
                    ctx_emb = encoder.encode(ctx_text)
                except Exception:
                    ctx_emb = None

            path = download_image(session, img_url, dest_dir, apply_diagram_filter=False)
            if not path:
                continue

            conf = _ctx_confidence_from_embeddings(query_embedding, ctx_emb)
            meta = {
                "source_kind": "api",
                "source_name": "wikimedia",
                "base_context": query or "",
                "prompt_id": prompt_id,
                "page_url": "",  # Wikimedia pages are derivable, but not required
                "image_url": img_url,
                "ctx_text": ctx_text,
                "ctx_embedding": _to_list_embedding(ctx_emb),
                "ctx_sem_score": conf,
                "ctx_confidence": conf,
            }
            if base_ctx_embedding is not None:
                meta["prompt_embedding"] = base_ctx_embedding

            _register_image_metadata(path, meta)
            saved.append(path)

            if len(saved) >= 10:
                break
        except Exception:
            continue

    src.img_paths = saved
    return saved

PARSERS = {
    "wikimedia": parse_wikimedia,
    "openverse": parse_openverse
}

# =========================
# Per-source prompt blocklist 
# =========================

def _source_json_path(source_name: str) -> str:
    return os.path.join(SOURCE_PATH, f"{source_name}.json")

def _is_prompt_blocklisted(source_name: str, prompt_id: str) -> bool:
    path = _source_json_path(source_name)
    data = _load_json(path)
    bl = (data.get("NoImagePromptBlocklist") or {})
    return bool(prompt_id and str(prompt_id) in bl)

def _blocklist_prompt(source_name: str, prompt_id: str, query: str, reason: str) -> None:
    if not source_name or not prompt_id:
        return
    path = _source_json_path(source_name)
    lock = _json_lock_for(path)
    with lock:
        data = _load_json(path)
        data.setdefault("NoImagePromptBlocklist", {})
        bl = data["NoImagePromptBlocklist"]

        entry = bl.get(prompt_id) or {}
        try:
            cnt = int(entry.get("count") or 0) + 1
        except Exception:
            cnt = 1

        bl[prompt_id] = {
            "query": str(query or entry.get("query") or ""),
            "count": cnt,
            "reason": str(reason or entry.get("reason") or "no_images"),
            "last_seen": datetime.datetime.utcnow().isoformat() + "Z",
        }

        _save_json_atomic(path, data)
        dbg(f"[BL] blocklisted prompt_id={prompt_id} source={source_name} count={cnt} reason={reason}")



import re
import hashlib

def _prompt_key(prompt: str) -> str:
    """
    Stable folder-safe key for a prompt.
    Example: "Eukaryotic cell" -> "eukaryotic_cell_a1b2c3d4"
    """
    p = (prompt or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", p).strip("_")
    if not slug:
        slug = "prompt"
    slug = slug[:40]
    h = hashlib.sha1((prompt or "").encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{h}"


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
        name = (tag.name or "").lower()
        if name in ("header", "footer", "nav", "aside"):
            return True

        cls_list = tag.get("class") or []
        for c in cls_list:
            c_low = (c or "").lower()
            if any(word in c_low for word in EXCLUDE_REGION_CLASS_WORDS):
                return True

        id_val = (tag.get("id") or "").lower()
        if any(word in id_val for word in EXCLUDE_REGION_CLASS_WORDS):
            return True

        role = (tag.get("role") or "").lower()
        if role in ("banner", "navigation", "complementary", "contentinfo"):
            return True

        return False
    except Exception:
        return False


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
    """
    Dynamic lock:
      - must be same registrable domain
      - allows shared path tokens OR shallow cross-links (prevents OpenStax-style slugs from being dropped)
    """
    pr = urlparse(root); pc = urlparse(candidate)
    ha = (pr.hostname or "").lower()
    hb = (pc.hostname or "").lower()
    if not same_registrable(ha, hb):
        return False

    cu = clean_url(candidate)
    if not cu:
        return False
    low = cu.lower()
    if any(bad in low for bad in BLOCKED_URL_WORDS):
        return False
    if any(low.endswith(ext) for ext in NON_HTML_EXT):
        return False

    # token overlap check
    rp = [s for s in (pr.path or "/").split("/") if s.strip()]
    cp = [s for s in (pc.path or "/").split("/") if s.strip()]
    rset = {s.strip().lower() for s in rp}
    cset = {s.strip().lower() for s in cp}
    if rset & cset:
        return True

    # allow shallow links (generic) so content routed via short slugs isn't discarded
    if len(cp) <= 2 or len(rp) <= 2:
        return True

    return False



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


def ordered_dedupe(iterable):
    seen = set()
    out = []
    for x in iterable:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def ddg_site_seed_urls(
    query: str,
    domain: str,
    max_results: int = 25,
    region: str = "wt-wt",
    safesearch: str = "moderate",
    proxies: dict | None = None,
) -> list[str]:
    """
    Generic prompt-driven site seeding. NO hardcoding to any topic.
    Pulls URLs via DDG text search: site:<domain> <query> (diagram/svg/figure).
    """
    seeds: list[str] = []
    if not query or not domain:
        return seeds

    q = query.strip()
    d = domain.strip().lower()
    ddg_queries = [
        f'site:{d} "{q}"',
        f"site:{d} {q} diagram",
        f"site:{d} {q} svg",
        f"site:{d} {q} figure",
    ]

    def _ok(u: str) -> bool:
        if not u:
            return False
        low = u.lower()
        if any(low.startswith(s) for s in SCHEME_BLOCK):
            return False
        if any(low.endswith(ext) for ext in NON_HTML_EXT):
            return False
        if any(bad in low for bad in BLOCKED_URL_WORDS):
            return False
        return True

    try:
        headers = {"User-Agent": UA.get("User-Agent", "diag-scrape/0.2")}
        with DDGS(headers=headers, proxies=proxies) as ddgs:
            for qq in ddg_queries:
                try:
                    # ddgs vs duckduckgo_search differ; handle both shapes
                    gen = None
                    if hasattr(ddgs, "text"):
                        gen = ddgs.text(qq, max_results=max_results, region=region, safesearch=safesearch)
                    elif hasattr(ddgs, "search"):
                        gen = ddgs.search(qq, max_results=max_results, region=region, safesearch=safesearch)
                    else:
                        gen = []

                    for item in gen:
                        u = (item.get("href") or item.get("url") or item.get("link") or "").strip()
                        u = clean_url(u)
                        if _ok(u):
                            seeds.append(u)
                        if len(seeds) >= max_results:
                            break
                except Exception:
                    continue
                if len(seeds) >= max_results:
                    break
    except Exception:
        return []

    return ordered_dedupe(seeds)[:max_results]



# =========================
# JS / Playwright helpers
# =========================
def expand_all_disclosures_sync(
    page,
    max_rounds: int = 6,
    per_round_click_cap: int = 600,
    quiet_ms: int = 250,
    stall_round_limit: int = 2,
):
    """
    Expand everything expandable ONCE.
    - Clicks only elements that are currently collapsed.
    - Marks clicked toggles with data-wmf-expanded="1" to prevent repeats.
    - Opens <details> directly.
    Stops when no more clicks AND anchors stop growing for stall_round_limit rounds.
    """

    def _anchors_count():
        try:
            return page.eval_on_selector_all("a[href]", "els => els.length")
        except Exception:
            return 0

    prev = _anchors_count()
    no_growth = 0

    for round_i in range(1, max_rounds + 1):
        try:
            clicks = page.evaluate(
                """(cap) => {
  const isVisible = (el) => {
    if (!el) return false;
    const r = el.getBoundingClientRect();
    if (!r || r.width < 2 || r.height < 2) return false;
    const st = getComputedStyle(el);
    return st && st.display !== 'none' && st.visibility !== 'hidden' && st.opacity !== '0';
  };

  const isCollapsed = (el) => {
    const aria = (el.getAttribute('aria-expanded') || '').trim().toLowerCase();
    if (aria === 'false') return true;
    if (aria === 'true') return false;

    const cls = (el.getAttribute('class') || '').toLowerCase();
    if (cls.includes('collapsed') || cls.includes('is-collapsed') || cls.includes('closed')) return true;

    const controls = el.getAttribute('aria-controls');
    if (controls) {
      const t = document.getElementById(controls);
      if (t) {
        const st = getComputedStyle(t);
        if (t.hidden) return true;
        if (t.getAttribute('aria-hidden') === 'true') return true;
        if (st && (st.display === 'none' || st.visibility === 'hidden')) return true;
      }
    }
    return false;
  };

  // Always open <details>
  document.querySelectorAll('details').forEach(d => { try { d.open = true; } catch(e){} });

  // Candidate toggles (NO plain "button" catch-all)
  const sel = [
    "summary",
    "button[aria-expanded='false']",
    "[role='button'][aria-expanded='false']",
    "[aria-controls][aria-expanded='false']",
    "[data-toggle='collapse']",
    "[data-bs-toggle='collapse']",
    ".accordion-button.collapsed",
    ".accordion-toggle",
    ".dropdown-toggle",
    ".collapsible",
    ".collapse-toggle",
    "[role='treeitem'][aria-expanded='false']",
    ".toc button[aria-expanded='false']",
    ".toc__toggle",
    ".toc-toggle"
  ].join(",");

  const els = Array.from(document.querySelectorAll(sel))
    .filter(el => el && !el.dataset.wmfExpanded)
    .filter(isVisible)
    .filter(isCollapsed)
    // don't click anchors
    .filter(el => (el.tagName || '').toLowerCase() !== 'a');

  let n = 0;
  for (const el of els) {
    if (n >= cap) break;
    try { el.dataset.wmfExpanded = "1"; } catch(e) {}
    try { el.click(); n++; } catch(e) {}
    try { el.setAttribute('aria-expanded', 'true'); } catch(e) {}
  }

  // Re-open details after clicks
  document.querySelectorAll('details').forEach(d => { try { d.open = true; } catch(e){} });

  // Scroll to trigger lazy/virtualized TOC loads
  try { window.scrollTo(0, document.body.scrollHeight); } catch(e) {}
  const scrollables = Array.from(document.querySelectorAll("[role='tree'], nav, aside, .toc, .table-of-contents"))
    .filter(el => el && el.scrollHeight > el.clientHeight + 50);
  for (const s of scrollables) {
    try { s.scrollTop = s.scrollHeight; } catch(e) {}
  }

  return n;
}""",
                per_round_click_cap,
            )
        except Exception:
            clicks = 0

        try:
            page.wait_for_load_state("networkidle", timeout=quiet_ms)
        except Exception:
            pass
        page.wait_for_timeout(quiet_ms)

        cur = _anchors_count()
        print(f"[EXPAND] round={round_i} clicks={clicks} anchors={cur} (prev={prev})")

        if cur > prev:
            no_growth = 0
        else:
            no_growth += 1

        if clicks == 0 and no_growth >= stall_round_limit:
            print(f"[EXPAND] no clicks + stalled {no_growth}/{stall_round_limit} → stop")
            break

        prev = cur



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

  document.querySelectorAll('details').forEach(d => { try { d.open = true; } catch(e){} });

  document.querySelectorAll('[aria-expanded="false"]').forEach(el => {
    try { el.setAttribute('aria-expanded','true'); } catch(e){}
  });

  document.querySelectorAll('[data-href],[data-url]').forEach(el => {
    pick(el.getAttribute('data-href'));
    pick(el.getAttribute('data-url'));
  });

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


def collect_links_with_modals_sync(page, expand=True, max_modal_clicks=6, **kwargs):
    """
    Returns deduped list of links discovered on:
      - main DOM
      - shadow DOM
      - data-* / onclick
      - modal/popup links
    If expand=False: ONLY visible <a href> anchors.
    """
    urls = []
    if expand:
        expand_all_disclosures_sync(page, max_rounds=10, per_round_click_cap=160, quiet_ms=350)

        basic = []
        shadow = []
        data_links = []
        modal_links_all = []
        popup_links_all = []


        basic = list(_collect_basic_anchors_sync(page) or [])
        for u in basic:
            urls.append(u)

        shadow = list(_collect_shadow_anchors_sync(page) or [])
        for u in shadow:
            urls.append(u)

        data_links = list(_collect_data_links_sync(page) or [])
        for u in data_links:
            urls.append(u)

        triggers = _find_modal_triggers_sync(page)[:max_modal_clicks]

        for el in triggers:
            try:
                modal_links, popup_links = _harvest_popup_urls_on_click_sync(page, el)
                modal_links = list(modal_links or [])
                popup_links = list(popup_links or [])

                modal_links_all.extend(modal_links)
                popup_links_all.extend(popup_links)

                for u in modal_links:
                    urls.append(u)
                for u in popup_links:
                    urls.append(u)
            except Exception:
                continue


    else:
        basic = []
        basic = list(_collect_basic_anchors_sync(page) or [])
        for u in basic:
            urls.append(u)

    raw = [u for u in urls if u]

    deduped = ordered_dedupe(raw)
    return deduped



# =========================
# Playwright reuse (Stage 2)
# =========================
class _PlaywrightReuse:
    """
    One Playwright browser/context reused across many fetches.
    Used by Stage 2 only.
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
# Shared math helpers
# =========================
def _norm_vec(v):
    try:
        import numpy as np
        if v is None:
            return None
        a = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(a))
        if n <= 0:
            return a
        return a / n
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


# =========================
#Terminal index helpers (json + embeddings)
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


# =========================
#Pinecone terminal-index storage
# =========================

_PINECONE_INDEX = None
_PINECONE_DIM = None

def _pinecone_index_name() -> str:
    pref = (os.getenv("PINECONE_RESEARCH_PREFIX") or "research").strip()
    return f"{pref}-terminal-index"

def _pinecone_dim_fallback() -> int:
    # all-MiniLM-L6-v2 is 384. Allow override.
    try:
        v = int(os.getenv("PINECONE_DIM") or "0")
        if v > 0:
            return v
    except Exception:
        pass
    return 384

def _get_pinecone_index(dim: int):
    """
    Supports pinecone client v3+ (Pinecone class) and older pinecone.init().
    Creates the index if missing (best-effort).
    """
    global _PINECONE_INDEX, _PINECONE_DIM
    if _PINECONE_INDEX is not None and _PINECONE_DIM == dim:
        return _PINECONE_INDEX

    api_key = os.getenv("PINECONE_API_KEY")
    cloud = os.getenv("PINECONE_CLOUD") or "aws"
    region = os.getenv("PINECONE_REGION") or "us-east-1"
    name = _pinecone_index_name()

    if not api_key:
        dbg("[PINECONE] missing PINECONE_API_KEY -> disabled")
        return None

    # --- try new client ---
    try:
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=api_key)

        # create if missing
        try:
            existing = {x["name"] for x in (pc.list_indexes() or [])}
        except Exception:
            try:
                existing = set(pc.list_indexes().names())
            except Exception:
                existing = set()

        if name not in existing:
            try:
                pc.create_index(
                    name=name,
                    dimension=int(dim),
                    metric="cosine",
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )
                dbg(f"[PINECONE] created index='{name}' dim={dim} metric=cosine spec={cloud}/{region}")
            except Exception as e:
                dbg(f"[PINECONE] create_index failed (maybe already exists / permissions): {e}")

        idx = pc.Index(name)
        _PINECONE_INDEX = idx
        _PINECONE_DIM = dim
        dbg(f"[PINECONE] ready index='{name}' dim={dim}")
        return idx
    except Exception:
        pass

    # --- try old client ---
    try:
        import pinecone
        pinecone.init(api_key=api_key, environment=f"{cloud}-{region}")
        try:
            existing = set([x["name"] for x in pinecone.list_indexes()])  # sometimes list[str]
        except Exception:
            try:
                existing = set(pinecone.list_indexes() or [])
            except Exception:
                existing = set()

        if name not in existing:
            try:
                pinecone.create_index(name=name, dimension=int(dim), metric="cosine")
                dbg(f"[PINECONE] created index='{name}' dim={dim} metric=cosine")
            except Exception as e:
                dbg(f"[PINECONE] create_index failed: {e}")

        idx = pinecone.Index(name)
        _PINECONE_INDEX = idx
        _PINECONE_DIM = dim
        dbg(f"[PINECONE] ready index='{name}' dim={dim} (old client)")
        return idx
    except Exception as e:
        dbg(f"[PINECONE] client import/init failed -> disabled: {e}")
        return None


def _pack_terminal_urls(urls: list[str]) -> str:
    """
    Pack terminal urls into a gzip+base64 string for Pinecone metadata.
    Avoids huge list metadata payloads.
    """
    try:
        raw = "\n".join([u for u in (urls or []) if u]).encode("utf-8", errors="ignore")
        gz = gzip.compress(raw, compresslevel=6)
        return base64.b64encode(gz).decode("ascii")
    except Exception:
        return ""

def _unpack_terminal_urls(packed: str) -> list[str]:
    try:
        if not packed:
            return []
        gz = base64.b64decode(packed.encode("ascii"))
        raw = gzip.decompress(gz).decode("utf-8", errors="ignore")
        return [line.strip() for line in raw.split("\n") if line.strip()]
    except Exception:
        return []

def _extract_page_signals(soup: BeautifulSoup | None) -> dict:
    """
    Basic page details you asked for: title / meta / H1-H2.
    Keep it small to stay inside Pinecone metadata limits.
    """
    if soup is None:
        return {}

    def _txt(node):
        try:
            return (node.get_text(" ", strip=True) or "").strip()
        except Exception:
            return ""

    def _cap(s: str, n: int = 240) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[: n - 3] + "..."

    out = {}

    try:
        title = soup.find("title")
        t = _cap(_txt(title), 240)
        if t:
            out["page_title"] = t
    except Exception:
        pass

    try:
        h1 = soup.find("h1")
        h1t = _cap(_txt(h1), 240)
        if h1t:
            out["h1"] = h1t
    except Exception:
        pass

    try:
        h2s = []
        for h2 in soup.find_all("h2")[:3]:
            ht = _cap(_txt(h2), 180)
            if ht:
                h2s.append(ht)
        if h2s:
            out["h2"] = h2s
    except Exception:
        pass

    try:
        md = soup.find("meta", attrs={"name": "description"})
        if md and md.get("content"):
            out["meta_description"] = _cap(md.get("content"), 320)
    except Exception:
        pass

    try:
        ogt = soup.find("meta", attrs={"property": "og:title"})
        if ogt and ogt.get("content"):
            out["og_title"] = _cap(ogt.get("content"), 240)
    except Exception:
        pass

    try:
        ogd = soup.find("meta", attrs={"property": "og:description"})
        if ogd and ogd.get("content"):
            out["og_description"] = _cap(ogd.get("content"), 320)
    except Exception:
        pass

    return out


def pinecone_query_terminal_indexes(
    *,
    subject: str,
    query_vec,
    registrable: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Returns a list of idx dicts shaped like your old JSON terminal indexes:
      {"entry_url","terminal_urls","terminal_count","centroid_embedding",...}
    But fetched from Pinecone.
    """
    if query_vec is None:
        return []

    # vector -> list[float]
    v = query_vec
    try:
        if hasattr(v, "tolist"):
            v = v.tolist()
    except Exception:
        pass

    dim = len(v) if isinstance(v, list) and v else _pinecone_dim_fallback()
    idx = _get_pinecone_index(dim)
    if idx is None:
        return []

    ns = (subject or "").strip() or "default"

    # metadata filter to keep indexes on same site (useful, not “desired source”)
    flt = None
    if registrable:
        flt = {"registrable": {"$eq": registrable}}

    try:
        res = idx.query(
            vector=v,
            top_k=int(top_k),
            include_metadata=True,
            include_values=True,
            namespace=ns,
            filter=flt,
        )
    except Exception as e:
        dbg(f"[PINECONE][QUERY] filter query failed -> retry without filter: {e}")
        try:
            res = idx.query(
                vector=v,
                top_k=int(top_k),
                include_metadata=True,
                include_values=True,
                namespace=ns,
            )
        except Exception as e2:
            dbg(f"[PINECONE][QUERY] failed: {e2}")
            return []

    matches = []
    try:
        matches = res.get("matches") or res.matches or []
    except Exception:
        matches = []

    out = []
    for m in matches:
        try:
            md = (m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", None)) or {}
            vals = (m.get("values") if isinstance(m, dict) else getattr(m, "values", None)) or None

            # NEW: prefer local bucket file if present
            terms = []
            term_file = md.get("terminals_file") or ""
            if term_file:
                terms = load_terminal_bucket_urls(term_file)

            if not terms:
                packed = md.get("terminals_packed") or ""
                terms = _unpack_terminal_urls(packed)


            out.append({
                "id": (m.get("id") if isinstance(m, dict) else getattr(m, "id", "")) or md.get("id"),
                "subject": md.get("subject") or subject,
                "entry_url": md.get("entry_url") or "",
                "terminal_urls": terms,
                "terminal_count": int(md.get("terminal_count") or len(terms)),
                "centroid_embedding": vals,  # keep your old field name
                "created_at": md.get("created_at") or "",
                # extra signals
                "registrable": md.get("registrable") or "",
                "entry_tail": md.get("entry_tail") or "",
                "page_title": md.get("page_title"),
                "h1": md.get("h1"),
                "h2": md.get("h2"),
                "meta_description": md.get("meta_description"),
                "og_title": md.get("og_title"),
                "og_description": md.get("og_description"),
                "source_name": md.get("source_name"),
            })
        except Exception:
            continue

    return out


def pinecone_upsert_terminal_index(
    *,
    idx_obj: dict,
    subject: str,
    registrable: str,
    source_name: str | None = None,
) -> None:
    """
    Upsert one terminal index bucket into Pinecone.
    Vector = centroid_embedding
    Metadata holds packed terminals + page header signals + entry info.
    """
    if not idx_obj:
        return

    cent = idx_obj.get("centroid_embedding")
    if cent is None:
        return

    v = cent
    try:
        if hasattr(v, "tolist"):
            v = v.tolist()
    except Exception:
        pass

    if not isinstance(v, list) or not v:
        return

    dim = len(v)
    idx = _get_pinecone_index(dim)
    if idx is None:
        return

    ns = (subject or "").strip() or "default"

    entry = clean_url(idx_obj.get("entry_url") or "")
    if not entry:
        return

    terminals = idx_obj.get("terminal_urls") or idx_obj.get("terminals") or []
    terminals = [clean_url(u) for u in terminals if u]
    packed = _pack_terminal_urls(terminals)

    term_file_rel = save_terminal_bucket_file(idx_obj, registrable=registrable, subject=subject)

    md = {
        "id": idx_obj.get("id") or "",
        "subject": subject,
        "registrable": registrable,
        "source_name": source_name or "",
        "entry_url": entry,
        "entry_tail": _last_path_part(entry),
        "terminal_count": int(idx_obj.get("terminal_count") or len(terminals)),
        "terminals_packed": packed,
        "terminals_file": term_file_rel,
        # small preview only
        "terminals_head": terminals[:40],
        "created_at": idx_obj.get("created_at") or (datetime.datetime.utcnow().isoformat() + "Z"),
    }

    # carry your page header signals if present
    for k in ("page_title","h1","h2","meta_description","og_title","og_description"):
        if k in idx_obj and idx_obj.get(k):
            md[k] = idx_obj.get(k)

    rec_id = (idx_obj.get("id") or hashlib.sha1(f"{subject}|{registrable}|{entry}".encode("utf-8")).hexdigest()[:12])

    try:
        idx.upsert(vectors=[(rec_id, v, md)], namespace=ns)
        dbg(f"[PINECONE][UPSERT] ✅ ns={ns} id={rec_id} entry={_short_url(entry)} terms={md['terminal_count']}")
    except Exception as e:
        dbg(f"[PINECONE][UPSERT][ERR] {e}")



# =========================
#js_capable_fetch 
# =========================
def js_capable_fetch(
    url: str,
    timeout_ms: int = 15000,
    wait_selector: str | None = "a[href]",
    js_mode: str = "full",          # "full" | "light" | "none" | "smart-light"
    max_modal_clicks: int = 3,
    min_text_chars: int = 1200,
    min_p_tags: int = 3,
    min_anchors_hint: int = 8,
    pw: "_PlaywrightReuse | None" = None,
    collect_anchors: bool = True,
) -> Tuple[str | None, List[str], str]:
    """
    Fetch a page and return (html, anchors, via).
    """

    low = (url or "").lower()
    if low.endswith((".pdf", ".docx", ".zip", ".pptx", ".xls", ".xlsx",
                     ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")):
        dbg(f"[SKIP] non-HTML resource: {url}")
        return None, [], "none"
    if any(bad in low for bad in BLOCKED_URL_WORDS):
        dbg(f"[SKIP][BLOCKLIST] {url}")
        return None, [], "blocked"

    def _requests_fetch(u: str) -> Tuple[Optional[str], List[str], str]:
        try:
            r = requests.get(u, headers=UA, timeout=timeout_ms / 1000)
            r.raise_for_status()
            ct = (r.headers.get("Content-Type") or "").lower()
            if "html" not in ct and "xml" not in ct:
                return None, [], "requests"

            html = r.text or ""
            if not collect_anchors:
                trace("fetch.requests", url=u, html_len=len(html), anchors=[], content_type=(r.headers.get("Content-Type") or ""))
                return html, [], "requests"

            soup = BeautifulSoup(html, "html.parser")
            anchors = []
            for a in soup.find_all("a", href=True):
                try:
                    anchors.append(urljoin(u, a.get("href")))
                except Exception:
                    continue

            trace("fetch.requests", url=u, html_len=len(html), anchors=anchors, content_type=(r.headers.get("Content-Type") or ""))
            return html, anchors, "requests"
        except Exception as e:
            dbg(f"[JSFETCH][requests] failed for {u}: {e}")
            return None, [], "requests"

    def _pw_fetch(pw_obj: _PlaywrightReuse) -> Tuple[str | None, list[str], str]:
        expand = (js_mode == "full")
        page = pw_obj.new_page(timeout_ms)
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

            if collect_anchors:
                anchors = collect_links_with_modals_sync(
                    page,
                    expand=expand,
                    max_modal_clicks=(max_modal_clicks if expand else 0),
                )
                anchors = [urljoin(url, a) for a in (anchors or []) if a]
            else:
                anchors = []

            html = page.content() or ""
            via = f"playwright:{'full' if expand else 'light'}"
            dbg(f"[JSFETCH][{via}] {_short_url(url)} anchors={len(anchors)} html_len={len(html)}")
            return html, anchors, via
        finally:
            try:
                page.close()
            except Exception:
                pass

    # --- mode routing ---
    if js_mode == "none":
        html, anchors, _ = _requests_fetch(url)
        dbg(f"[JSFETCH][requests] {_short_url(url)} anchors={len(anchors)} html_len={len(html or '')}")
        return html, anchors, "requests"

    if js_mode == "smart-light":
        html, anchors, _ = _requests_fetch(url)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            text_len = len(soup.get_text(" ", strip=True))
            p_tags = len(soup.find_all("p"))
            if (text_len >= min_text_chars and p_tags >= min_p_tags) or len(anchors) >= min_anchors_hint:
                dbg(f"[JSFETCH][requests:light] {_short_url(url)} anchors={len(anchors)} (text={text_len}, p={p_tags})")
                return html, anchors, "requests:light"
            dbg(f"[JSFETCH][requests:light→pw] {_short_url(url)} (text={text_len}, p={p_tags}, a={len(anchors)})")
        js_mode = "light"

    # --- Playwright path (reuse if provided, else one-shot) ---
    try:
        if pw is not None and pw.ok():
            return _pw_fetch(pw)

        temp = _PlaywrightReuse(headless=True)
        if not temp.ok():
            return None, [], "none"
        try:
            return _pw_fetch(temp)
        finally:
            temp.close()

    except Exception as e:
        dbg(f"[JSFETCH][playwright] failed for {_short_url(url)}: {e}")

    return None, [], "none"



# =========================
# WordNet
# =========================
def get_limited_lemmas(word, per_synset_limit=4, pos_filter=None, debug=True):
    if isinstance(pos_filter, str):
        pos_filter = [pos_filter]

    def _emit_none():
        if debug:
            print(f"\n[LEMMA] word='{word}'  synsets_found=0  pos_filter={pos_filter or 'all'}")
            print(f"  [!] No synsets found for '{word}'")

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

    # Fallback: if WordNet has nothing for a phrase, build a generic lemma set
    if not results:
        _emit_none()

        phrase = (word or "").strip()
        toks = [t for t in re.split(r"[\s\-_\/]+", phrase) if t]
        toks_norm = []
        for t in toks:
            t2 = t.strip()
            if t2:
                toks_norm.append(t2)

        # Try synsets per token (generic; not hardcoding)
        collected = []
        for t in toks_norm:
            try:
                ss = wn.synsets(t)
            except Exception:
                ss = []
            for s in ss[:2]:
                for lm in (s.lemmas() or [])[:per_synset_limit]:
                    nm = lm.name().replace("_", " ").strip()
                    if nm:
                        collected.append(nm)

        # Always include the phrase and tokens
        base = []
        if phrase:
            base.append(phrase)
        base.extend(toks_norm)
        base.extend(collected)

        # Dedupe while preserving order
        seen = set()
        lemmas = []
        for x in base:
            k = x.strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            lemmas.append(x)

        if lemmas:
            results["fallback.phrase"] = {"definition": "", "lemmas": lemmas}

    return results



# =========================
# Text Hit + Image
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


def _sniff_image_type(head: bytes, ctype: str | None, as_text: str | None) -> str | None:
    """
    Return a lowercase image type extension without dot (png, jpg, gif, webp, svg, tiff, bmp)
    by checking (1) content-type, (2) magic bytes, (3) svg text.
    None if not confidently an image.
    """
    ctype = (ctype or "").lower()

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

    if as_text:
        t = as_text.lstrip().lower()
        if t.startswith("<svg") or ("<svg" in t[:2048] and "</svg>" in t):
            return "svg"

    return None


def _unique_path_with_ext(dest_dir: str, stem: str, ext: str) -> str:
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


def _compile_phrase_regexes_adapter(base_query: str, lemma_obj: dict | None = None):
    """
    Build regex patterns and token helpers for the given phrase or lemma set.
    Used in smart_find_hits_in_soup() and stage2_drill_search_and_images().
    Returns:
        (phrase_rxs, proximity_tokens)
    """
    def _escape_ph(s: str) -> str:
        return re.escape(s.strip())

    phrases = []
    if lemma_obj:
        for k, vs in lemma_obj.items():
            phrases.extend(vs)
    phrases.append(base_query)
    phrases = [p for p in phrases if p.strip()]
    phrases = sorted(set(phrases), key=len, reverse=True)

    phrase_rxs = []
    for p in phrases:
        esc = _escape_ph(p)
        strict = re.compile(rf"\b{esc}\b", re.IGNORECASE)
        phrase_rxs.append(strict)

        parts = [re.escape(tok) for tok in re.split(r"\s+", p.strip()) if tok]
        if len(parts) > 1:
            fuzzy = re.compile(r"\b" + r"\s*(?:[^\w\s]{0,3}\s*)?".join(parts) + r"\b", re.IGNORECASE)
            phrase_rxs.append(fuzzy)

    proximity_tokens = set()
    for p in phrases:
        for t in re.split(r"\W+", p.lower()):
            if len(t) > 2:
                proximity_tokens.add(t)

    return phrase_rxs, proximity_tokens

def _dominant_light_background(rgb_f, *, border_frac=0.05):
    """
    rgb_f: float32 RGB in [0,1], already composited to white if original had alpha.
    Returns: (bg_frac, bg_luma, bg_sat)
    """
    import numpy as np

    h, w = rgb_f.shape[:2]
    b = max(1, int(min(h, w) * border_frac))

    border = np.zeros((h, w), bool)
    border[:b, :] = True
    border[-b:, :] = True
    border[:, :b] = True
    border[:, -b:] = True

    # Estimate background from border median (robust against corner labels, etc.)
    bg_rgb = np.median(rgb_f[border], axis=0)  # shape (3,)

    # Luma + saturation (HSV-ish)
    cmax = rgb_f.max(axis=-1)
    cmin = rgb_f.min(axis=-1)
    delta = cmax - cmin
    sat = np.where(cmax > 1e-6, delta / (cmax + 1e-6), 0.0)
    luma = 0.2126*rgb_f[...,0] + 0.7152*rgb_f[...,1] + 0.0722*rgb_f[...,2]

    bg_luma = float(0.2126*bg_rgb[0] + 0.7152*bg_rgb[1] + 0.0722*bg_rgb[2])
    bg_sat  = float((bg_rgb.max() - bg_rgb.min()) / (bg_rgb.max() + 1e-6))

    # Distance to background color (cheap + works well enough)
    dist = np.max(np.abs(rgb_f - bg_rgb[None, None, :]), axis=-1)

    # "Background-like" pixels:
    # - close to bg color
    # - not too saturated (paper can be slightly warm/blue but not vivid)
    bg_mask = (dist <= 0.10) & (sat <= 0.35) & (luma >= 0.35)

    bg_frac = float(bg_mask.mean())
    return bg_frac, bg_luma, bg_sat



# ---------- downloader (stricter + clearer debug) ----------
def _is_diagram_like(data: bytes,
                     bg_min_frac: float = 0.4,     # must be mostly one background
                     bg_min_luma: float = 0.4,     # background must be light (paper/white)
                     resize_max: int = 900) -> bool:
    import io
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        dbg("[DIAGRAM] PIL/numpy not available -> ALLOW")
        return True

    try:
        im0 = Image.open(io.BytesIO(data))
        im = im0.convert("RGBA") if im0.mode in ("RGBA", "LA", "P") else im0.convert("RGB")
    except Exception:
        dbg("[DIAGRAM] decode failed -> REJECT")
        return False

    w, h = im.size
    if max(w, h) > resize_max:
        scale = resize_max / float(max(w, h))
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)

    arr = np.asarray(im, dtype=np.float32) / 255.0

    if arr.shape[2] == 4:
        rgb = arr[..., :3]
        a   = arr[..., 3]
        # Composite to white paper
        rgb_f = rgb * a[..., None] + 1.0 * (1.0 - a[..., None])
    else:
        rgb_f = arr[..., :3]

    # --- dominant background test ---
    bg_frac, bg_luma, bg_sat = _dominant_light_background(rgb_f, border_frac=0.05)

    if bg_luma < bg_min_luma:
        dbg(f"[DIAGRAM] bg_luma={bg_luma:.3f} < {bg_min_luma:.3f} -> REJECT")
        return False

    if bg_frac < bg_min_frac:
        dbg(f"[DIAGRAM] bg_frac={bg_frac:.3f} < {bg_min_frac:.3f} -> REJECT")
        return False

    # If you want: also reject obvious photos by "colorfulness" when bg_frac is borderline
    rg = rgb_f[..., 0] - rgb_f[..., 1]
    yb = 0.5*(rgb_f[..., 0] + rgb_f[..., 1]) - rgb_f[..., 2]
    colorfulness = float(np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3*np.sqrt(np.mean(np.abs(rg))**2 + np.mean(np.abs(yb))**2))

    # Allow high bg dominance even if colorful (your ideal diagrams include colored parts),
    # but reject low-bg colorful stuff (photos / paintings).
    if bg_frac < 0.70 and colorfulness > 0.50:
        dbg(f"[DIAGRAM] bg_frac={bg_frac:.3f} colorfulness={colorfulness:.3f} -> REJECT(photo-ish)")
        return False

    dbg(f"[DIAGRAM] bg_frac={bg_frac:.3f} bg_luma={bg_luma:.3f} bg_sat={bg_sat:.3f} colorfulness={colorfulness:.3f} -> ALLOW")
    return True




# === download_image (runs filter & logs decision) ===================
def download_image(
    session: requests.Session,
    img_url: str,
    dest_dir: str,
    referer: str | None = None,
    *,
    apply_diagram_filter: bool = True,
) -> Optional[str]:
    """
    Robust downloader with diagram filter.
    Fixes Wikimedia 403 issues by:
      - using a correct Referer (page url), never the file url
      - retrying 403 without referer and with commons referer
      - adding sane Accept headers
    """
    import base64

    def _save_bytes(raw: bytes, ext_hint: str, ctype_hint: str, source_label: str) -> Optional[str]:
        try:
            as_text = raw.decode("utf-8", errors="ignore")
        except Exception:
            as_text = None

        ext = _sniff_image_type(raw[:64], ctype_hint, as_text) or ext_hint or "img"

        if ext != "svg" and apply_diagram_filter:
            if not _is_diagram_like(raw):
                dbg(f"[IMG_SAVE][filter] {source_label} REJECTED by diagram filter")
                return None

        if ext == "svg":
            dbg(f"[IMG_SAVE][svg-skip] {source_label}")
            return None

        h = hashlib.sha1(raw).hexdigest()
        out = _unique_path_with_ext(dest_dir, h, ext)
        with open(out, "wb") as f:
            f.write(raw)
        dbg(f"[IMG_SAVE] {source_label} -> {out}")
        return out

    try:
        # ---- data URL handling ----
        if img_url.startswith("data:image/"):
            m = re.match(r"^data:(image/[\w\+\-\.]+);base64,(.+)$", img_url, re.I | re.S)
            if not m:
                return None
            ctype = m.group(1)
            raw = base64.b64decode(m.group(2), validate=True)
            return _save_bytes(raw, ext_hint="", ctype_hint=ctype, source_label="data-url")

        # ---- request headers ----
        hdr = dict(BASE_HEADERS)
        hdr.update({
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        })
        if referer:
            hdr["Referer"] = referer  # IMPORTANT: never file-url referer; caller passes page url.

        # ---- 403 retry strategy ----
        attempts = [
            dict(hdr),
            {k: v for k, v in hdr.items() if k.lower() != "referer"},
            {**dict(hdr), "Referer": "https://commons.wikimedia.org/"},
        ]

        r = None
        for h in attempts:
            try:
                r = session.get(img_url, headers=h, stream=True, timeout=30, allow_redirects=True)
            except Exception:
                r = None
            if r is None:
                continue
            if r.status_code != 403:
                break
            try:
                r.close()
            except Exception:
                pass

        if r is None:
            return None

        if r.status_code == 403:
            try:
                dbg(f"[IMG_SAVE][403 body] {r.text[:300]}")
            except Exception:
                pass

        with r:
            r.raise_for_status()

            ctype = r.headers.get("Content-Type", "") or ""

            first = r.raw.read(4096)
            head_lower = first[:64].lower()
            if b"<html" in head_lower or b"<!doctype html" in head_lower:
                dbg(f"[IMG_SAVE][html-skip] {img_url}")
                return None

            cl_header = r.headers.get("Content-Length")
            maybe_len = int(cl_header) if (cl_header and cl_header.isdigit()) else None
            if maybe_len is not None and maybe_len < 1000:
                dbg(f"[IMG_SAVE][tiny/ui-skip] {img_url} ({maybe_len} B)")
                return None

            data = first + r.raw.read()

            path_ext = os.path.splitext(urlparse(img_url).path)[1].lower().lstrip(".")
            ext_hint = "jpg" if path_ext == "jpeg" else (
                path_ext if path_ext in {"png", "jpg", "jpeg", "gif", "webp", "svg", "tiff", "bmp"} else ""
            )

            return _save_bytes(data, ext_hint=ext_hint, ctype_hint=ctype, source_label=img_url)

    except Exception as e:
        dbg(f"[IMG_SAVE][err] {img_url} -> {type(e).__name__}: {e!r}")
    return None





def save_image_candidate(
    session: requests.Session,
    candidate,                 # str URL OR <svg> Tag
    soup_for_inline: BeautifulSoup,
    page_url: str,
    dest_dir: str,
    referer: Optional[str] = None,
    dedupe_set: Optional[set] = None,
) -> Optional[str]:
    if isinstance(candidate, Tag) and candidate.name == "svg":
        return None

    if isinstance(candidate, str):
        if dedupe_set is not None and candidate in dedupe_set:
            return None
        d = download_image(session, candidate, dest_dir, referer=referer)
        if d and dedupe_set is not None:
            dedupe_set.add(candidate)
        return d

    return None


def _extract_img_candidates_near_hit(node, soup, page_url, *, per_hit_cap: Optional[int] = None):
    """
    Return image URLs near a hit node (optionally capped by per_hit_cap).
    """
    cand = []
    seen = set()

    def _add_url(u, why):
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

    def _add_svg(svg_tag, why):
        if not svg_tag:
            return
        key = (id(svg_tag), why)
        if key in seen:
            return
        if per_hit_cap is not None and len(cand) >= per_hit_cap:
            return
        seen.add(key)
        cand.append((svg_tag, why))

    def _img_url_from_tag(img):
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

    def _svg_is_probably_ui(svg: Tag) -> bool:
        try:
            if not isinstance(svg, Tag) or svg.name != "svg":
                return True
            cls = " ".join(svg.get("class") or []).lower()
            if any(x in cls for x in ("icon", "sprite", "logo", "wordmark", "btn", "button")):
                return True
            aria_hidden = (svg.get("aria-hidden") or "").strip().lower()
            if aria_hidden == "true":
                w = svg.get("width")
                h = svg.get("height")
                try:
                    wi = float(str(w).replace("px", "")) if w else 0.0
                    hi = float(str(h).replace("px", "")) if h else 0.0
                    if wi and hi and (wi * hi) < 6000:
                        return True
                except Exception:
                    pass
            # If it's inside obvious nav/header/footer, treat as UI.
            if _is_excluded_region(svg) or _is_excluded_region(svg.parent if isinstance(svg.parent, Tag) else svg):
                return True
            return False
        except Exception:
            return True

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
    scopes.append(soup)

    for sc in scopes:

        for pic in sc.find_all("picture"):
            for src in pic.find_all("source"):
                srcset = (src.get("srcset") or "").strip()
                if srcset:
                    url = srcset.split(",")[-1].strip().split()[0]
                    _add_url(url, "picture/srcset")
                    if per_hit_cap is not None and len(cand) >= per_hit_cap:
                        return cand
            img = pic.find("img")
            if img:
                url = _img_url_from_tag(img)
                if url:
                    _add_url(url, "picture/img")
                    if per_hit_cap is not None and len(cand) >= per_hit_cap:
                        return cand

        for img in sc.find_all("img"):
            url = _img_url_from_tag(img)
            if url:
                _add_url(url, "img")
                if per_hit_cap is not None and len(cand) >= per_hit_cap:
                    return cand

        for a in sc.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if href and re.search(r"\.(png|jpe?g|gif|webp|svg|tiff?)($|\?)", href, re.I):
                _add_url(href, "a->image")
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
      - terminal: same parent path as current URL
      - deeper: deeper subpaths for recursive exploration
    allowed_fn may return bool or (bool, reason_str). Reason is ignored.
    """
    def _parent_part(u: str) -> str:
        p = urlparse(u)
        path = (p.path or "/").rstrip("/")
        if path and path != "/":
            path = path.rsplit("/", 1)[0] or "/"
        return urlunparse((p.scheme or "https", (p.hostname or "").lower(), path, "", "", ""))

    cur_parent = _parent_part(cur)
    terms, deeper = [], []

    for a in anchors or []:
        if not a or a == cur:
            continue

        try:
            res = allowed_fn(a)
            ok = bool(res[0]) if isinstance(res, tuple) else bool(res)
        except Exception:
            ok = False

        if not ok:
            continue

        (terms if _parent_part(a) == cur_parent else deeper).append(a)

    return ordered_dedupe(terms), ordered_dedupe(deeper)



# =========================
# Terminal index IO (per-source)
# =========================
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

def _coerce_terminal_urls(raw) -> list[str]:
    """
    Accepts:
      - ["https://..", ...]
      - [{"url":"https://..","key":"..."}, ...]   (legacy)
      - [("https://..","key"), ...]              (legacy)
    Returns: clean_url deduped list[str]
    """
    out: list[str] = []
    for t in (raw or []):
        u = ""
        if isinstance(t, str):
            u = t
        elif isinstance(t, dict):
            u = t.get("url") or t.get("href") or ""
        elif isinstance(t, (list, tuple)) and t:
            u = t[0]
        u = clean_url(u)
        if u:
            out.append(u)
    return ordered_dedupe(out)



def persist_terminal_index_for_subject(source_name: str, subject: str, idx_obj: dict) -> None:
    """
    Writes idx_obj into SOURCE_PATH/<source>.json under TerminalIndexes[subject].
    Dedupes by entry_url.

    Persistence format:
      - terminals: [url, url, ...]    (URL-only)
      - terminal_count: stored count (len(terminals))
      - terminal_count_total: optional, if provided by caller
    """
    path = os.path.join(SOURCE_PATH, f"{source_name}.json")
    lock = _json_lock_for(path)
    with lock:
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {}

            data.setdefault("TerminalIndexes", {})
            data["TerminalIndexes"].setdefault(subject, [])

            entry = clean_url((idx_obj or {}).get("entry_url") or "")
            if not entry:
                dbg("[IDX][WRITE][REJECT] no entry_url")
                return

            for ex in data["TerminalIndexes"][subject]:
                if clean_url((ex or {}).get("entry_url") or "") == entry:
                    dbg(f"[IDX][WRITE][SKIP] already exists entry={entry}")
                    return

            # --- normalize terminals to URL-only ---
            store_obj = dict(idx_obj or {})
            raw_terms = store_obj.get("terminals") or store_obj.get("terminal_urls") or []
            urls = _coerce_terminal_urls(raw_terms)

            store_obj.pop("terminal_urls", None)
            store_obj["terminals"] = urls
            store_obj["terminal_count"] = int(store_obj.get("terminal_count") or len(urls))

            data["TerminalIndexes"][subject].append(store_obj)

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


def _normalize_ctx_meta(ctx_meta: dict) -> dict:
    if not ctx_meta:
        return {}
    allowed = {"ctx_text", "ctx_embedding", "ctx_score", "ctx_sem_score", "ctx_confidence"}
    out = {}
    for k in allowed:
        if k in ctx_meta:
            out[k] = ctx_meta[k]
    return out

from dataclasses import dataclass

@dataclass
class TerminalFatigue:
    """
    Tracks consecutive "no gain" terminal attempts across the entire root traversal.

    - misses: consecutive terminal attempts that produced 0 new images (includes soup=None / hits=0 / no candidates)
    - grace: number of consecutive misses that are free (do not count towards fatigue)
    - limit: once (misses - grace) >= limit => root is considered exhausted
    """
    grace: int = 30
    limit: int = 25
    misses: int = 0

    def register(self, gained: int) -> None:
        # gained = number of new images saved from this terminal page
        if gained > 0:
            self.misses = 0
        else:
            self.misses += 1

    @property
    def fatigue(self) -> int:
        # what you print as "fatigue" in logs
        return max(0, self.misses - self.grace)

    def hit_limit(self) -> bool:
        return self.fatigue >= self.limit

# =========================
# Per-root prompt blocklist (stored WITH the root entry in <source>.json)
# =========================

def _roots_key_candidates():
    # tolerate existing schemas
    return ("Roots", "roots", "root_urls", "RootUrls")

def _get_roots_container(data: dict) -> tuple[str, list]:
    """
    Returns (key, list_obj). If none exist, creates data["Roots"] = [].
    """
    for k in _roots_key_candidates():
        v = data.get(k)
        if isinstance(v, list):
            return k, v
    data["Roots"] = []
    return "Roots", data["Roots"]

def _root_url_from_entry(entry):
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("root") or entry.get("url") or entry.get("entry_url") or entry.get("Root") or ""
    return ""

def _ensure_root_dict(entry):
    """
    If roots are stored as strings, we must convert to dict so we can attach NoImagePromptBlocklist to it.
    """
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, str) and entry.strip():
        return {"root": entry.strip()}
    return None

def _find_root_entry_index(data: dict, root_url: str) -> int:
    """
    Find the root entry index by comparing clean_url(root).
    Returns -1 if not found.
    """
    if not root_url:
        return -1
    root_cu = clean_url(root_url)
    k, roots_list = _get_roots_container(data)
    for i, e in enumerate(roots_list):
        ru = _root_url_from_entry(e)
        if ru and clean_url(ru) == root_cu:
            return i
    return -1

def _is_prompt_blocklisted_for_root(source_name: str, root_url: str, prompt_id: str) -> bool:
    """
    READ PATH: checks <source>.json -> Roots[*].NoImagePromptBlocklist[prompt_id]
    """
    if not source_name or not root_url or not prompt_id:
        return False

    path = _source_json_path(source_name)
    data = _load_json(path)

    idx = _find_root_entry_index(data, root_url)
    if idx < 0:
        return False

    k, roots_list = _get_roots_container(data)
    entry = roots_list[idx]

    # tolerate old string roots that haven't been converted yet
    if not isinstance(entry, dict):
        return False

    bl = (entry.get("NoImagePromptBlocklist") or {})
    return bool(str(prompt_id) in bl)

def _blocklist_prompt_for_root(source_name: str, root_url: str, prompt_id: str, query: str, reason: str) -> None:
    """
    WRITE PATH: stores record under the specific root object in <source>.json
      Roots: [
        {"root":"https://...", "NoImagePromptBlocklist": { "<prompt_id>": {...} } }
      ]
    If roots were strings, converts that element to dict to attach the blocklist.
    """
    if not source_name or not root_url or not prompt_id:
        return

    path = _source_json_path(source_name)
    lock = _json_lock_for(path)

    with lock:
        data = _load_json(path)
        k, roots_list = _get_roots_container(data)

        idx = _find_root_entry_index(data, root_url)
        if idx < 0:
            # root not present -> append a dict entry
            roots_list.append({"root": clean_url(root_url)})
            idx = len(roots_list) - 1

        # ensure dict entry (convert from string if necessary)
        cur_entry = roots_list[idx]
        root_obj = _ensure_root_dict(cur_entry)
        if root_obj is None:
            return

        # if we converted, write it back
        roots_list[idx] = root_obj
        root_obj["root"] = clean_url(_root_url_from_entry(root_obj) or root_url)

        root_obj.setdefault("NoImagePromptBlocklist", {})
        bl = root_obj["NoImagePromptBlocklist"]

        old = bl.get(str(prompt_id)) or {}
        try:
            cnt = int(old.get("count") or 0) + 1
        except Exception:
            cnt = 1

        bl[str(prompt_id)] = {
            "query": str(query or old.get("query") or ""),
            "count": cnt,
            "reason": str(reason or old.get("reason") or "no_images"),
            "last_seen": datetime.datetime.utcnow().isoformat() + "Z",
        }

        _save_json_atomic(path, data)
        dbg(f"[BL][ROOT] blocklisted prompt_id={prompt_id} root={_short_url(root_url)} source={source_name} count={cnt} reason={reason}")


# =========================
# STAGE 2: Drill-by-parent with inline text search + image pick
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

    # root exhaustion on terminal no-accept streak
    terminal_no_accept_limit: int = 15,
    encoder: Optional[Any] = None,
    query_embedding: Optional[Any] = None,
    prompt_embedding: Optional[Any] = None,
    url_sort_k: int = 75,
    fatigue_grace: int = 5,

    terminal_indexes: Optional[list[dict]] = None,
    terminal_index_min_terminals: int = 25,
    terminal_index_use_threshold: float = 0.22,
    terminal_index_store_cap: int = 600,
    terminal_index_centroid_sample: int = 60,
    terminal_js_mode: str = "none",          # "none" | "smart-light" | "light"
    terminal_collect_anchors: bool = False,  # terminals never need anchors
    terminal_js_fallback: bool = False,      # if True, re-render terminal with js_mode="light" when no img candidates


    subject: Optional[str] = None,
    source_name: Optional[str] = None,

    on_terminal_index_promoted: Optional[Any] = None,

    img_url_dedupe_set: Optional[set] = None,

    *_, **compat_kwargs,
) -> list[str]:

    import numpy as np

    if prompt_embedding is None:
        prompt_embedding = (
            compat_kwargs.pop("base_ctx_embedding", None) or
            compat_kwargs.pop("base_context_embedding", None) or
            compat_kwargs.pop("prompt_vec", None)
        )

    # Enforce your requirement: max 5 images per anchor(root)
    try:
        hard_image_cap = int(hard_image_cap)
    except Exception:
        hard_image_cap = 5
    hard_image_cap = max(1, min(hard_image_cap, 5))

    class _NextRoot(Exception): pass
    class _StopAll(BaseException): pass
    class _SkipTerminals(BaseException): pass  # NEW: don't kill root; just stop terminal loop & continue BFS
    class _FatigueState:
        """
        One fatigue state per ROOT traversal.
        - starts only after first success (first saved image)
        - grace applies ONCE (not per BFS layer, not per later successes)
        - misses reset on success, grace does NOT re-arm
        """
        def __init__(self, limit: int, grace: int):
            self.limit = max(1, int(limit))
            self.grace_total = max(0, int(grace))
            self.started = False
            self.grace_left = 0
            self.misses = 0

        def note_success(self):
            if not self.started:
                self.started = True
                self.grace_left = self.grace_total
            self.misses = 0  # success resets misses

        def note_failure(self):
            if not self.started:
                return  # no fatigue before first success

            if self.grace_left > 0:
                self.grace_left -= 1
                return

            self.misses += 1

        def should_bail(self) -> bool:
            return self.started and self.misses >= self.limit


    saved_paths: list[str] = []

    if isinstance(roots, dict):
        roots_list = list(roots.keys())
    else:
        roots_list = list(roots or [])

    prompt_id = _prompt_key(base_query or "")

    pw = _PlaywrightReuse(headless=True)
    prompt_vec = _norm_vec(prompt_embedding if prompt_embedding is not None else query_embedding)
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

    # NEW: dedupe on clean_url BEFORE scoring so you don't hammer the same page 8 times
    def _score_and_sort_urls(urls: list[str], *, k: int) -> list[str]:
        normed = []
        seen = set()
        for u in (urls or []):
            cu = clean_url(u)
            if not cu or cu in seen:
                continue
            seen.add(cu)
            normed.append(cu)

        if not normed:
            return []
        if encoder is None or prompt_vec is None:
            return normed[:k] if k and len(normed) > k else normed

        keys = [_url_key_for_embed(u) for u in normed]
        _embed_keys_batch([kk for kk in keys if kk])

        scored = []
        for u, kk in zip(normed, keys):
            e = key_emb_cache.get(kk)
            s = float(np.dot(prompt_vec, e)) if e is not None else 0.0
            scored.append((s, u))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k] if k and len(scored) > k else scored
        return [u for _, u in top]

    def _path_depth(u: str) -> int:
        try:
            p = urlparse(clean_url(u))
            return len([s for s in (p.path or "").split("/") if s])
        except Exception:
            return 0

    def _url_sim_to_prompt(u: str) -> float:
        if encoder is None or prompt_vec is None:
            return 0.0
        kk = _url_key_for_embed(u)
        if not kk:
            return 0.0
        if kk not in key_emb_cache:
            _embed_keys_batch([kk])
        e = key_emb_cache.get(kk)
        return float(np.dot(prompt_vec, e)) if e is not None else 0.0


    # gate knobs (tune once; default values work well to stop /math /science style hops)
    MIN_SIM_NORMAL = 0.20        # normal pages
    MIN_SIM_SHALLOW = 0.50       # very shallow pages like /math
    MIN_SIM_UPWARD = 0.65        # "going up" from the chosen root depth
    SHALLOW_DEPTH_MAX = 2        # <=2 path segments is considered "site nav shallow"

    root_depth_cache: dict[str, int] = {}

    def _allowed(root: str, child_url: str):
        cu = clean_url(child_url)
        if not cu:
            return (False, "clean_url_empty")

        low = cu.lower()
        if any(bad in low for bad in BLOCKED_URL_WORDS):
            return (False, "blocked_word")
        if any(low.endswith(ext) for ext in NON_HTML_EXT):
            return (False, "non_html_ext")

        pr = urlparse(clean_url(root))
        pc = urlparse(cu)

        ha = (pr.hostname or "").lower()
        hb = (pc.hostname or "").lower()
        if not same_registrable(ha, hb):
            return (False, "different_registrable")

        rd = root_depth_cache.get(root)
        if rd is None:
            rd = _path_depth(root)
            root_depth_cache[root] = rd

        cd = _path_depth(cu)
        sim = _url_sim_to_prompt(cu)

        # 1) Hard stop: don't climb up unless extremely on-prompt
        if cd < rd:
            ok = sim >= MIN_SIM_UPWARD
            return (ok, f"upward cd<{rd} sim={sim:.3f} thr={MIN_SIM_UPWARD:.3f}")

        # 2) Shallow nav pages need high similarity
        if cd <= SHALLOW_DEPTH_MAX:
            ok = sim >= MIN_SIM_SHALLOW
            return (ok, f"shallow cd<={SHALLOW_DEPTH_MAX} sim={sim:.3f} thr={MIN_SIM_SHALLOW:.3f}")

        # 3) Normal pages: allow if prompt similarity decent OR token overlap lock passes
        if sim >= MIN_SIM_NORMAL:
            return (True, f"normal sim={sim:.3f} thr={MIN_SIM_NORMAL:.3f}")

        try:
            ok = domain_and_phrase_lock(root, cu)
            return (ok, f"domain_and_phrase_lock ok={ok}")
        except Exception as e:
            return (False, f"domain_and_phrase_lock_exc:{e}")
        
    def _allowed_ok(root_url: str, child_url: str) -> bool:
        try:
            res = _allowed(root_url, child_url)
            if isinstance(res, tuple):
                return bool(res[0])
            return bool(res)
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

    stored_entries = set()
    if terminal_indexes:
        for it in terminal_indexes:
            stored_entries.add(clean_url(it.get("entry_url") or ""))

    pending_indexes: dict[str, dict] = {}

    def _index_terminals_for_entry(term_full: list[str], deeper_full: list[str]) -> list[str]:
        # if term_full alone is large enough, use it
        if term_full and len(term_full) >= terminal_index_min_terminals:
            return term_full

        # hub page: deeper is the universe, but KEEP term_full too
        if deeper_full and len(deeper_full) >= terminal_index_min_terminals:
            return ordered_dedupe((term_full or []) + (deeper_full or []))

        return ordered_dedupe((term_full or []) + (deeper_full or []))


    def _create_pending_index(entry_url: str, terminal_urls: list[str], entry_signals: dict | None = None):
        entry = clean_url(entry_url)
        if not entry:
            return
        if entry in stored_entries:
            return
        if entry in pending_indexes:
            return

        # prompt-agnostic: build full deduped + stable ordered terminal set
        full_terms = _stable_terminal_set(terminal_urls)

        # IMPORTANT: this is the REAL check (cleaned+deduped), not "how many we walked"
        if len(full_terms) < terminal_index_min_terminals:
            dbg(
                f"[IDX][PENDING][REJECT] entry={_short_url(entry)} "
                f"clean_terms={len(full_terms)} < min={terminal_index_min_terminals}"
            )
            return

        cent = _centroid_for_terminal_urls(full_terms)
        if not cent:
            dbg(f"[IDX][PENDING][REJECT] entry={_short_url(entry)} terminals={len(full_terms)} reason=no-centroid")
            return

        # Persisted list can be capped to save space
        store_terms = full_terms[:terminal_index_store_cap]

        idx_obj = {
            "id": hashlib.sha1(f"{subject}|{entry}|{len(store_terms)}".encode("utf-8")).hexdigest()[:10],
            "subject": subject,
            "entry_url": entry,

            # URL-only persistence format
            "terminals": store_terms,

            # keep both counts: stored vs total discovered
            "terminal_count": int(len(store_terms)),
            "terminal_count_total": int(len(full_terms)),

            "centroid_embedding": cent,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        }

        try:
            for k, v in (entry_signals or {}).items():
                if v:
                    idx_obj[k] = v
        except Exception:
            pass

        pending_indexes[entry] = {
            "entry_url": entry,

            # CRITICAL FIX:
            # membership set must be FULL, otherwise promotion never fires for terminals outside the cap
            "terminal_set": set(full_terms),

            "idx_obj": idx_obj,
            "saved_from_bucket": 0,
        }

        dbg(
            f"[IDX][PENDING] ★ entry={_short_url(entry)} "
            f"stored={len(store_terms)} total={len(full_terms)} sample={min(len(full_terms), terminal_index_centroid_sample)}"
        )


    def _promote_index_if_hit(page_url: str):
        pu = clean_url(page_url)
        if not pu:
            return

        for entry, obj in list(pending_indexes.items()):
            if pu in obj["terminal_set"]:
                obj["saved_from_bucket"] += 1

                dbg(f"[IDX][BUCKET_HIT] ✅ entry={_short_url(entry)} saved_from_bucket={obj['saved_from_bucket']} (page={_short_url(pu)})")

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

    render_expand_state: dict[str, bool] = {}

    def render(
        url: str,
        *,
        expand: bool,
        force_js_light: bool = False,
        collect_anchors: bool = True,
        js_mode_override: str | None = None,
    ):
        if not force_js_light and url in html_cache:
            was_expanded = bool(render_expand_state.get(url, False))
            if expand and not was_expanded:
                pass  # re-fetch in full mode
            else:
                soup = BeautifulSoup(html_cache[url], "html.parser") if html_cache[url] else None
                a = anchors_cache.get(url, [])
                if not collect_anchors:
                    a = []
                return soup, a, False

        if js_mode_override is not None:
            mode = js_mode_override
        else:
            mode = "full" if expand else ("light" if force_js_light else "smart-light")

        html, anchors, via = js_capable_fetch(url, js_mode=mode, pw=pw, collect_anchors=collect_anchors)

        render_expand_state[url] = bool(expand)

        if not html and not anchors:
            if not force_js_light:
                html_cache[url] = None
                anchors_cache[url] = []
            return None, [], False

        if collect_anchors:
            full = ordered_dedupe([urljoin(url, a) for a in anchors if a])
        else:
            full = []

        html_cache[url] = html
        anchors_cache[url] = full
        dbg(f"[S2] via={via:<16} url={_short_url(url)} anchors={len(full)} expand={expand} force_js_light={force_js_light}")
        soup = BeautifulSoup(html, "html.parser") if html else None
        return soup, full, True


    def _save_one(candidate, tsoup, page_url, hit_meta: Optional[dict] = None) -> bool:
        dest = save_image_candidate(
            session=session,
            candidate=candidate,
            soup_for_inline=tsoup,
            page_url=page_url,
            dest_dir=os.path.join(IMAGES_PATH, "domain_search"),
            referer=page_url,
            dedupe_set=(img_url_dedupe_set if img_url_dedupe_set is not None else set()),
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

        dbg(f"---[SAVE] {os.path.basename(dest)}  conf={meta.get('ctx_confidence','n/a')}  from={_short_url(page_url)}")

        _promote_index_if_hit(page_url)
        return True

    def _save_from_hits(page_url: str, tsoup, ranked_hits: list, *, per_root_counter: list[int]):
        for node, score, details in ranked_hits:
            if _global_cap_hit():
                raise _StopAll()

            ctx_score = details.get("ctx_score", score)
            ctx_sem = details.get("ctx_sem_score", details.get("sem_score"))
            ctx_text = details.get("ctx_text") or details.get("snippet") or details.get("heading_text") or ""

            ctx_emb = None
            for k in ("ctx_embedding", "embedding", "sem_embedding", "context_embedding"):
                if k in details and details[k] is not None:
                    ctx_emb = details[k]
                    break

            if ctx_emb is not None and hasattr(ctx_emb, "tolist"):
                try:
                    ctx_emb = ctx_emb.tolist()
                except Exception:
                    pass

            if ctx_emb is None and encoder is not None and ctx_text:
                try:
                    v = encoder.encode(ctx_text)
                    if hasattr(v, "tolist"):
                        ctx_emb = v.tolist()
                    else:
                        ctx_emb = list(v)
                except Exception as e:
                    dbg(f"[S2][CTX_EMB][ERR] encode failed: {e}")
                    ctx_emb = None

            hit_meta = _normalize_ctx_meta({
                "ctx_text": ctx_text,
                "ctx_embedding": ctx_emb,
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

    def _process_terminal_url(tu: str, *, per_root_counter: list[int], fatigue_state: Optional[_FatigueState], idx: int, total: int) -> int:
        before = per_root_counter[0]

        tsoup, _, _ = render(
            tu,
            expand=False,
            collect_anchors=terminal_collect_anchors,
            js_mode_override=terminal_js_mode,
        )

        if not tsoup:
            dbg(f"[TERM {idx:02d}/{total:02d}]  {_short_url(tu)}  soup=None")
            if fatigue_state is not None and per_root_counter[0] > 0:
                fatigue_state.note_failure()
                if fatigue_state.should_bail():
                    dbg(f"[S2] FATIGUE LIMIT (limit={fatigue_state.limit}, grace={fatigue_state.grace_total}) → next root")
                    raise _NextRoot()
            return 0

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
            dbg(f"[TERM {idx:02d}/{total:02d}]  {_short_url(tu)}  hits=0")
            if fatigue_state is not None and per_root_counter[0] > 0:
                fatigue_state.note_failure()
                if fatigue_state.should_bail():
                    dbg(f"[S2] FATIGUE LIMIT (limit={fatigue_state.limit}, grace={fatigue_state.grace_total}) → next root")
                    raise _NextRoot()
            return 0

        dbg(f"[TERM {idx:02d}/{total:02d}]{_short_url(tu)}  hits={len(ranked)}  imgs={per_root_counter[0]}")

        any_cand = False
        for n, _, _ in ranked[:3]:
            if _extract_img_candidates_near_hit(n, tsoup, tu, per_hit_cap=1):
                any_cand = True
                break

        if (not any_cand) and terminal_js_fallback:
            tsoup, _, _ = render(
                tu,
                expand=False,
                collect_anchors=terminal_collect_anchors,
                js_mode_override="light",
            )


        _save_from_hits(tu, tsoup, ranked_hits=ranked, per_root_counter=per_root_counter)

        gained = max(0, per_root_counter[0] - before)

        if fatigue_state is not None:
            if gained > 0:
                fatigue_state.note_success()
            elif per_root_counter[0] > 0:
                fatigue_state.note_failure()

            if fatigue_state.should_bail():
                dbg(f"[S2] FATIGUE LIMIT (limit={fatigue_state.limit}, grace={fatigue_state.grace_total}) → next root")
                raise _NextRoot()

        return gained


    def _stable_terminal_set(term_urls: list[str]) -> list[str]:
        """
        Build a prompt-agnostic terminal list:
        - clean_url
        - dedupe
        - stable deterministic ordering (lexicographic) so storage is consistent
        """
        cleaned = [clean_url(u) for u in (term_urls or []) if u]
        cleaned = [u for u in cleaned if u]
        cleaned = ordered_dedupe(cleaned)
        cleaned.sort()  # IMPORTANT: removes prompt-order bias
        return cleaned
    
    chosen_index = None
    chosen_sim = 0.0
    if terminal_indexes:
        chosen_index, chosen_sim = pick_best_terminal_index_for_prompt(
            terminal_indexes, prompt_vec, min_sim=terminal_index_use_threshold
        )
        if chosen_index:
           dbg(
                f"[IDX][USE] ⭐ sim={chosen_sim:.3f} entry={_short_url(chosen_index.get('entry_url'))} "
                f"terminals={len(chosen_index.get('terminals') or chosen_index.get('terminal_urls') or [])}"
            )


    def _extract_index_terminals(idx_obj: dict) -> list[str]:
        raw = (idx_obj or {}).get("terminal_urls") or (idx_obj or {}).get("terminals") or []
        return _coerce_terminal_urls(raw)



    # Terminal processing is pooled across BFS discovery.
    processed_terminals: set[str] = set()   # dedupe for terminal-walk
    expanded_pages: set[str] = set()        # dedupe for BFS expands

    terminal_pool: list[str] = []
    terminal_pool_set: set[str] = set()

    def _pool_add(urls: list[str]):
        for u in (urls or []):
            cu = clean_url(u)
            if not cu:
                continue
            if cu in processed_terminals:
                continue
            if cu in terminal_pool_set:
                continue
            terminal_pool_set.add(cu)
            terminal_pool.append(cu)

    def _pool_drain_batch(
        *,
        batch_k: int,
        per_root_saved: list[int],
        fatigue_state: _FatigueState,
        no_accept_streak: list[int],
    ):
        if not terminal_pool:
            return

        k = max(75, int(batch_k))  # enforce >= top 75
        batch = _score_and_sort_urls(terminal_pool, k=k)
        if not batch:
            return

        batch_set = set(batch)
        terminal_pool[:] = [u for u in terminal_pool if u not in batch_set]
        terminal_pool_set.difference_update(batch_set)

        dbg(f"[S2][POOL] drain batch={len(batch)} remaining_pool={len(terminal_pool)}")

        for i, tu in enumerate(batch, start=1):
            if _global_cap_hit():
                raise _StopAll()
            if per_root_saved[0] >= hard_image_cap:
                raise _NextRoot()

            if tu in processed_terminals:
                continue
            processed_terminals.add(tu)

            gained = _process_terminal_url(
                tu,
                per_root_counter=per_root_saved,
                fatigue_state=fatigue_state,
                idx=i,
                total=len(batch),
            )

            if gained <= 0:
                no_accept_streak[0] += 1
            else:
                no_accept_streak[0] = 0

            if no_accept_streak[0] >= terminal_no_accept_limit:
                dbg(f"[S2] TERMINAL STREAK {no_accept_streak[0]} with 0 accepts → NEXT ROOT")
                raise _NextRoot()


    try:
        for rdx, root in enumerate(roots_list, start=1):
            index_routed_once = False
            if _global_cap_hit():
                raise _StopAll()

            # --- ROOT prompt blocklist (READ) ---
            if source_name and prompt_id and root:
                try:
                    if _is_prompt_blocklisted_for_root(source_name, root, prompt_id):
                        dbg(f"[S2][ROOT][SKIP][BL] prompt_id={prompt_id} root={_short_url(root)} source={source_name}")
                        continue
                except Exception:
                    pass

            dbg(f"\n[S2] ROOT[{rdx}/{len(roots_list)}] {root}")

            per_root_saved = [0]
            no_accept_streak = [0]
            fatigue_state = _FatigueState(limit=fatigue_limit, grace=fatigue_grace)

            # BFS discovery queue starts with root
            queue = deque([clean_url(root)])

            # clear pool for this root
            terminal_pool.clear()
            terminal_pool_set.clear()

            try:
                # ----------------------------
                # Terminal-index routing FIRST
                # ----------------------------
                if (not index_routed_once) and chosen_index:
                    idx_terms = _extract_index_terminals(chosen_index)
                    dbg(f"[IDX][ROUTE] ▶ entry={_short_url(chosen_index.get('entry_url') or '')} terms={len(idx_terms)}")

                    if idx_terms:
                        # push index terminals into pool and drain WITHOUT scoring
                        _pool_add(idx_terms)

                        while terminal_pool and per_root_saved[0] < hard_image_cap:
                            _pool_drain_batch(
                                batch_k=max(75, url_sort_k),
                                per_root_saved=per_root_saved,
                                fatigue_state=fatigue_state,
                                no_accept_streak=no_accept_streak,
                            )

                    index_routed_once = True

                    #if temrinal index for source is good
                    if per_root_saved[0] > 0:
                        dbg(f"[IDX][ROUTE] produced {per_root_saved[0]} images → skip BFS for this source")
                        raise _NextRoot()
                    else:
                        dbg("[IDX][ROUTE] produced 0 images → fallback to BFS normal routing")

           
                while queue:
                    if _global_cap_hit():
                        raise _StopAll()
                    if per_root_saved[0] >= hard_image_cap:
                        raise _NextRoot()

                    cur = queue.popleft()
                    if not cur:
                        continue
                    if cur in expanded_pages:
                        continue
                    expanded_pages.add(cur)

                    csoup, canchors, _ = render(cur, expand=True)
                    if not canchors:
                        continue

                    term_full, deeper_full = partition_anchors(cur, canchors, lambda u: _allowed_ok(root, u))

                    dbg(f"[S2] cur={_short_url(cur)} term={len(term_full)} deeper={len(deeper_full)} q={len(queue)}")

                    idx_terms = _index_terminals_for_entry(term_full, deeper_full)
                    _create_pending_index(cur, idx_terms, entry_signals=_extract_page_signals(csoup))


                    # Pool terminals from this page
                    _pool_add(term_full)
                    if len(term_full) < 10 and len(deeper_full) > 50:
                        _pool_add(deeper_full)

                    deeper = _score_and_sort_urls(deeper_full, k=max(75, url_sort_k))
                    for du in deeper:
                        cu = clean_url(du)
                        if cu and cu not in expanded_pages:
                            queue.append(cu)

                    # Drain terminals in big batches (not per BFS layer)
                    try:
                        if len(terminal_pool) >= max(75, url_sort_k):
                            _pool_drain_batch(
                                batch_k=max(75, url_sort_k),
                                per_root_saved=per_root_saved,
                                fatigue_state=fatigue_state,
                                no_accept_streak=no_accept_streak,
                            )
                    except _SkipTerminals:
                        pass

                # after discovery ends, drain whatever remains
                while terminal_pool and per_root_saved[0] < hard_image_cap:
                    try:
                        _pool_drain_batch(
                            batch_k=max(75, url_sort_k),
                            per_root_saved=per_root_saved,
                            fatigue_state=fatigue_state,
                            no_accept_streak=no_accept_streak,
                        )
                    except _SkipTerminals:
                        break

            except _NextRoot:
                dbg(f"[S2] NEXT ROOT (saved {per_root_saved[0]} for {_short_url(root)})")


            dbg(f"[S2] ROOT DONE {_short_url(root)} images_for_root={per_root_saved[0]} total_images={len(saved_paths)}")
            if source_name and prompt_id and root and per_root_saved[0] == 0:
                try:
                    _blocklist_prompt_for_root(
                        source_name=source_name,
                        root_url=root,
                        prompt_id=prompt_id,
                        query=base_query,
                        reason="no_images",
                    )
                except Exception:
                    pass

    except _StopAll:
        dbg(f"[S2] GLOBAL STOP total_images={len(saved_paths)} (global_cap={global_image_cap})")

    finally:
        try:
            pw.close()
        except Exception:
            pass

    return saved_paths





# =========================
# STAGE 1: Valid roots finder
# =========================

def find_valid_roots(
    base_url: str,
    lemma_obj: dict,
    max_pages_stage1: int = 100000,
    max_pages_stage2_per_root: int = 220,   # kept for compatibility (unused)
    timeout: int = 8,                       # kept for compatibility (unused)
    known_roots: list[str] | None = None,
    js_mode: str = "light",
    prompt_text: str | None = None,
    max_roots: int = 2,
    *,
    verbose: bool = False,               
):
    """
    Stage 1: BFS discovery using MiniLM semantic scoring.
    Same behavior, but removes the heavy ranking/log output by default.
    """
    import numpy as np
    from collections import deque

    log = print if verbose else (lambda *_a, **_k: None)

    # ----------------------------
    # Helpers
    # ----------------------------
    def _clean_url_s1(u: str) -> str:
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

    def _last_path_part_s1(u: str) -> str:
        p = urlparse(u)
        segs = [s for s in (p.path or "").split("/") if s]
        return segs[-1] if segs else ""

    def _normalize_for_embed_s1(s: str) -> str:
        s = (s or "").lower()
        s = s.replace("-", " ").replace("_", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def key_for_url(u: str) -> str:
        return _normalize_for_embed_s1(_last_path_part_s1(u))

    def same_reg(u: str, reg: str) -> bool:
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

    def _prompt_rerank_and_cap(out: dict) -> dict:
        # Collection behavior unchanged; only reorder + cap final returned roots.
        if not (prompt_text and encoder is not None and out):
            if max_roots and len(out) > max_roots:
                return {k: out[k] for k in list(out.keys())[:max_roots]}
            return out

        try:
            pvec = encoder.encode(str(prompt_text), normalize_embeddings=True)
            urls = list(out.keys())
            keys = [key_for_url(u) for u in urls]
            embs = encoder.encode(keys, normalize_embeddings=True)

            scored = []
            for u, k, e in zip(urls, keys, embs):
                s = _dot(pvec, e)
                scored.append((s, u, k))

            scored.sort(key=lambda x: x[0], reverse=True)
            keep = scored[:max_roots] if max_roots else scored

            out2 = {}
            for s, u, k in keep:
                meta = dict(out[u])
                meta["prompt_rank_score"] = float(s)
                meta["prompt_rank_key"] = k
                out2[u] = meta
            return out2
        except Exception:
            # exact fallback behavior: return original (then cap)
            if max_roots and len(out) > max_roots:
                return {k: out[k] for k in list(out.keys())[:max_roots]}
            return out

    encoder = _EMBED_MODEL
    if encoder is None:
        print("[S1] ⚠ MiniLM encoder unavailable — returning empty roots")
        return {}

    lemma_phrases = _lemma_phrases_from_obj(lemma_obj)
    if not lemma_phrases:
        inferred = key_for_url(_clean_url_s1(base_url))
        lemma_phrases = [inferred or "query"]

    menu_keyword = "site navigation menu toc section list"
    prompt_emb = _embed_centroid(lemma_phrases)
    menu_emb = encoder.encode(menu_keyword, normalize_embeddings=True)

    threshold_prompt_expand = 0.2
    threshold_menu_expand = 0.2
    threshold_prompt_root = 0.5

    base_clean = _clean_url_s1(base_url)
    if not base_clean:
        print("[S1] Invalid base URL.")
        return {}

    host = (urlparse(base_clean).hostname or "").lower()
    t = tldextract.extract(host)
    base_reg = ".".join([t.domain, t.suffix]) if t.suffix else t.domain

    # ----------------------------
    # If known_roots provided: skip discovery, then rerank+cap at the end.
    # ----------------------------
    if known_roots:
        roots0 = [_clean_url_s1(r) for r in known_roots if r]
        roots0 = [
            r for r in roots0
            if r and same_reg(r, base_reg) and not any(r.lower().endswith(ext) for ext in NON_HTML_EXT)
        ]
        roots0 = prefer_shortest_by_bucket(roots0)
        out = {r: {"parent": None, "depth": 0, "prompt": None, "menu": None} for r in roots0}
        return _prompt_rerank_and_cap(out)

    visited = {base_clean}
    frontier = deque([base_clean])
    layer_idx = 0
    total_fetches = 0

    log(f"[S1] Starting discovery on {base_clean} (domain={base_reg})")

    accepted_root = None
    accepted_root_scores = None

    while frontier and layer_idx < 10 and total_fetches < max_pages_stage1:
        layer_idx += 1
        layer = list(frontier)
        frontier.clear()

        candidates = []
        for u in layer:
            if not same_reg(u, base_reg):
                continue
            if any(u.lower().endswith(ext) for ext in NON_HTML_EXT):
                continue
            candidates.append(u)

        if not candidates:
            continue

        keys = [key_for_url(u) for u in candidates]

        # Note: menu_texts == prompt_texts (menu_key was identical)
        prompt_url_embs = encoder.encode(keys, normalize_embeddings=True)
        menu_url_embs = encoder.encode(keys, normalize_embeddings=True)

        prompt_scores = [_dot(prompt_emb, e) for e in prompt_url_embs]
        menu_scores = [_dot(menu_emb, e) for e in menu_url_embs]
        combined = [0.8 * p + 0.2 * m for p, m in zip(prompt_scores, menu_scores)]

        kept_expand = []
        kept_roots = []

        for u, k, ps, ms, cs in zip(candidates, keys, prompt_scores, menu_scores, combined):
            if ps >= threshold_prompt_expand or ms >= threshold_menu_expand:
                kept_expand.append((u, ps, ms, cs, k))
                if ps >= threshold_prompt_root:
                    kept_roots.append((u, ps, ms, cs, k))

        kept_expand_sorted = sorted(kept_expand, key=lambda x: x[1], reverse=True)
        kept_roots_sorted = sorted(kept_roots, key=lambda x: x[1], reverse=True)

        if kept_roots_sorted:
            best = kept_roots_sorted[0]
            accepted_root = best[0]
            accepted_root_scores = (best[1], best[2], best[3], best[4])
            break

        expand_from = [u for (u, _, _, _, _) in kept_expand_sorted] if kept_expand_sorted else list(candidates)

        next_frontier = []
        for u in expand_from:
            try:
                _, anchors, _via = js_capable_fetch(u, js_mode=js_mode)
            except Exception:
                anchors = []
            total_fetches += 1
            if not anchors:
                continue
            for a in anchors:
                cu = _clean_url_s1(a)
                if not cu or cu in visited:
                    continue
                if not same_reg(cu, base_reg):
                    continue
                if any(cu.lower().endswith(ext) for ext in NON_HTML_EXT):
                    continue
                visited.add(cu)
                next_frontier.append(cu)

        if next_frontier:
            nf_keys = [key_for_url(u) for u in next_frontier]
            nf_embs = encoder.encode(nf_keys, normalize_embeddings=True)
            nf_scores = [_dot(prompt_emb, e) for e in nf_embs]
            next_frontier = [u for _, u in sorted(zip(nf_scores, next_frontier), reverse=True)]

        for n in prefer_shortest_by_bucket(next_frontier):
            frontier.append(n)

    if not accepted_root:
        return {}

    tree_nodes = {}
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

        child_candidates = []
        parent_for_child = {}

        for parent_url in tree_frontier:
            try:
                _, anchors, _via = js_capable_fetch(parent_url, js_mode=js_mode)
            except Exception:
                anchors = []

            if not anchors:
                continue

            for a in anchors:
                cu = _clean_url_s1(a)
                if not cu or cu in tree_visited:
                    continue
                if not same_reg(cu, base_reg):
                    continue
                if any(cu.lower().endswith(ext) for ext in NON_HTML_EXT):
                    continue

                tree_visited.add(cu)
                child_candidates.append(cu)
                parent_for_child.setdefault(cu, parent_url)

        child_candidates = prefer_shortest_by_bucket(child_candidates)
        if not child_candidates:
            break

        c_keys = [key_for_url(u) for u in child_candidates]

        p_embs = encoder.encode(c_keys, normalize_embeddings=True)
        m_embs = encoder.encode(c_keys, normalize_embeddings=True)

        p_scores = [_dot(prompt_emb, e) for e in p_embs]
        m_scores = [_dot(menu_emb, e) for e in m_embs]
        c_scores = [0.8 * p + 0.2 * m for p, m in zip(p_scores, m_scores)]

        pass_high = []
        for u, k, ps, ms, cs in zip(child_candidates, c_keys, p_scores, m_scores, c_scores):
            if ps >= threshold_prompt_root:
                pass_high.append((u, ps, ms, cs, k))

        pass_high_sorted = sorted(pass_high, key=lambda x: x[1], reverse=True)
        if not pass_high_sorted:
            break

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

    depth1_plus = [u for u, meta in tree_nodes.items() if meta.get("depth", 0) >= 1]
    depth1_plus = prefer_shortest_by_bucket(depth1_plus)

    if depth1_plus:
        out = {u: tree_nodes[u] for u in depth1_plus if u in tree_nodes}
    else:
        out = {accepted_root: tree_nodes[accepted_root]}

    return _prompt_rerank_and_cap(out)


# =========================
# Sources / API
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
    try:
        dbg(f"[API][{source.name}] GET {source.url} params={built}")
        resp = requests.get(source.url, params=built, headers=BASE_HEADERS, timeout=20)
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

# =========================
# HANDLE NON-API
# =========================
def handle_result_no_api(source: Source, query: str, subj: str, hard_image_cap: int = 5,
                         encoder: Optional[Any] = None,
                         query_embedding: Optional[Any] = None,
                         base_ctx_embedding: Optional[Any] = None,
                         global_image_cap: int | None = None):

    def uniq_keep_order(items):
        seen = set()
        out = []
        for x in items:
            if x and x not in seen:
                seen.add(x); out.append(x)
        return out

    # Keep subject-driven Stage 1 discovery (you said it's intentional)
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

    if valid_endpoints:
        dbg(f"[NOAPI] using known endpoints count={len(valid_endpoints)} for subj={subj}")
        roots = find_valid_roots(
            base_url=source.url,
            lemma_obj=lemma_obj_subj,      # subject-driven discovery stays
            known_roots=valid_endpoints,
            js_mode="light",
            prompt_text=query,             # prompt rerank on final roots only
            max_roots=2, 
        )
    else:
        dbg("[NOAPI] no valid endpoints; starting Stage 1 discovery")
        roots = find_valid_roots(
            base_url=source.url,
            lemma_obj=lemma_obj_subj,      #
            js_mode="light",
            prompt_text=query,           
            max_roots=2,                  
        )

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

    session = requests.Session()
    session.headers.update({"User-Agent": "diag-scrape/0.1"})
    html_cache = {}
    anchors_cache = {}

    # Enforce your requirement here too (caller safety)
    try:
        hard_image_cap = int(hard_image_cap)
    except Exception:
        hard_image_cap = 5
    hard_image_cap = max(1, min(hard_image_cap, 5))

    dbg(f"[NOAPI] Stage 2 starting roots={len(roots)} hard_image_cap={hard_image_cap}")
    reg = _registrable_of(source.url or "")
    loaded_idxs = pinecone_query_terminal_indexes(
        subject=subj,
        query_vec=(base_ctx_embedding if base_ctx_embedding is not None else query_embedding),
        registrable=reg,
        top_k=5,
    )
    dbg(f"[PINECONE] loaded terminal indexes candidates={len(loaded_idxs)} subj={subj} reg={reg}")


    img_url_dedupe = set()


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
        on_terminal_index_promoted=lambda idx: pinecone_upsert_terminal_index(
            idx_obj=idx,
            subject=subj,
            registrable=reg,
            source_name=source.name,
        ),


        img_url_dedupe_set=img_url_dedupe,

        terminal_no_accept_limit=20 ,
        global_image_cap=global_image_cap,

    )

    source.img_paths = saved
    return saved



# =========================
# Unique collection + SigLIP rerank
# =========================
def _file_sha1(path: str, chunk=1024*1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _ahash64(path: str, size: int = 8) -> str | None:
    """
    64-bit average hash.
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
        return None


def _json_safe(obj):
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
    except Exception:
        pass

    if isinstance(obj, set):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


def collect_unique_images(
    images_root: str,
    base_ctx_embedding=None,
    prompt_text: str | None = None,
    prompt_filter_text: str | None = None,
    unique_subdir: str | None = None,
    dbg=print,
    backend: SiglipBackend | None = None,
):
    """
    Walks `images_root`, dedupes images, ranks them with SigLIP + context,
    and keeps ONLY the top 10 in <images_root>/UniqueImages/<unique_subdir>.
    Only considers images whose metadata base_context matches prompt_filter_text (if provided).
    """
    unique_dir = os.path.join(images_root, "UniqueImages")
    if unique_subdir:
        unique_dir = os.path.join(unique_dir, unique_subdir)

    os.makedirs(unique_dir, exist_ok=True)

    # cleanup only THIS prompt folder
    for fname in os.listdir(unique_dir):
        fpath = os.path.join(unique_dir, fname)
        if os.path.isfile(fpath):
            try:
                os.remove(fpath)
            except Exception as e:
                dbg(f"[collect][warn][cleanup] {fpath} -> {e}")

    sha1_to_canonical: Dict[str, str] = {}
    ahash_to_canonical: Dict[str, str] = {}
    canonical_metas: Dict[str, List[dict]] = {}

    pf_norm = None
    if prompt_filter_text:
        pf_norm = str(prompt_filter_text).strip().casefold()

    for dirpath, _, filenames in os.walk(images_root):
        if os.path.abspath(dirpath) == os.path.abspath(unique_dir):
            continue

        for fname in filenames:
            if not IMG_EXT_RE.search(fname):
                continue
            path = os.path.join(dirpath, fname)

            try:
                sha1 = _file_sha1(path)
            except Exception as e:
                dbg(f"[collect][skip][read] {path} -> {e}")
                continue

            canonical = sha1_to_canonical.get(sha1)

            if canonical is None:
                ah = _ahash64(path)
                if ah is not None and ah in ahash_to_canonical:
                    canonical = ahash_to_canonical[ah]
                else:
                    canonical = path
                    sha1_to_canonical[sha1] = canonical
                    if ah is not None:
                        ahash_to_canonical[ah] = canonical

            metas = IMAGE_METADATA.get(path)

            if metas and pf_norm is not None:
                metas = [
                    m for m in metas
                    if str(m.get("base_context", "") or "").strip().casefold() == pf_norm
                ]

            if metas:
                lst = canonical_metas.get(canonical)
                if lst is None:
                    canonical_metas[canonical] = list(metas)
                else:
                    lst.extend(metas)

    if not canonical_metas:
        dbg("[collect] no images with metadata found; nothing to rank")
        return []

    candidates: List[dict] = []
    for img_path, metas in canonical_metas.items():
        img_prompt_emb = None
        for m in metas:
            pe = m.get("prompt_embedding") or m.get("base_ctx_embedding")
            if pe is not None:
                img_prompt_emb = pe
                break
        if img_prompt_emb is None:
            img_prompt_emb = base_ctx_embedding

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
            continue

        # pick best available source tags (don’t assume metas[0])
        sk = None
        sn = None
        for m in metas:
            if sk is None and m.get("source_kind"):
                sk = m.get("source_kind")
            if sn is None and m.get("source_name"):
                sn = m.get("source_name")
            if sk is not None and sn is not None:
                break
        sk = sk or "unknown"
        sn = sn or ""

        candidates.append(
            {
                "image_path": img_path,
                "ctx_embedding": best_ctx["ctx_embedding"],
                "prompt_embedding": img_prompt_emb,
                "ctx_confidence": best_ctx["ctx_confidence"],
                "ctx_score": best_ctx["ctx_score"],
                "ctx_sem_score": best_ctx["ctx_sem_score"],
                "ctx_text": best_ctx["ctx_text"] or "",
                "source_kind": sk,
                "source_name": sn,
            }
        )


    if not candidates:
        dbg("[collect] no candidates with ctx_embedding; skipping SigLIP ranking")
        return []

    prompt_text = prompt_text or ""
    backend = backend or get_siglip_backend()
    desired_final_k = 5
    if backend is None:
        dbg("[collect] SigLIP backend unavailable; skipping ranking")
        return []

    with _SIGLIP_INFER_LOCK:
        raw = rerank_image_candidates_siglip(
            candidates=candidates,
            prompt_text=prompt_text,
            backend=backend,
            top_n=60,
            final_k=max(25, desired_final_k * 6),
        )


    if not raw:
        dbg("[collect] SigLIP ranking returned no winners")
        return []

    cand_by_path = {c["image_path"]: c for c in candidates}

    has_non_api = any((c.get("source_kind") or "") != "api" for c in candidates)

    SOURCE_KIND_WEIGHT = {
        # “verified” in your system: content-found pages with context + strict filtering
        "domain_search": 1.0,
        "noapi": 1.0,
        # DDG is weaker verification than your site-harvest
        "ddg": 0.70,
        "ddg_cc": 0.70,
        # APIs are kept as fallback: they should lose when non-api exists
        "api": 0.35 if has_non_api else 1.0,
        "generated": 1.0,
        "unknown": 0.85,
    }

    for w in raw:
        c = cand_by_path.get(w.get("image_path") or "", {})
        sk = (c.get("source_kind") or "unknown").lower()
        base = float(w.get("final_score") or 0.0)
        weight = float(SOURCE_KIND_WEIGHT.get(sk, 0.85))
        w["weighted_score"] = base * weight

    raw.sort(key=lambda x: float(x.get("weighted_score") or 0.0), reverse=True)
    ranked = raw[:desired_final_k]


    if not ranked:
        dbg("[collect] SigLIP ranking returned no winners")
        return []

    winners_by_path = {w["image_path"]: w for w in ranked}

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
            continue

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

        img_prompt_emb = None
        for m in metas:
            pe = m.get("prompt_embedding") or m.get("base_ctx_embedding")
            if pe is not None:
                img_prompt_emb = pe
                break
        if img_prompt_emb is None:
            img_prompt_emb = base_ctx_embedding

        core_list: List[dict] = []
        ctx_contexts: List[dict] = []

        for m in metas:
            m_core = {}
            for k, v in m.items():
                if k in heavy_fields:
                    continue
                m_core[k] = v
            if m_core:
                core_list.append(m_core)

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

        base_ctx_text = ""
        for m in metas:
            bc = m.get("base_context")
            if isinstance(bc, str) and bc.strip():
                base_ctx_text = bc.strip()
                break
        if not base_ctx_text:
            base_ctx_text = (prompt_filter_text or prompt_text or "").strip()

        ctx_meta[dest_path] = {
            "base_context": base_ctx_text,
            "prompt_id": str(unique_subdir or ""),
            "prompt_embedding": img_prompt_emb,
            "clip_embedding": winner.get("clip_embedding"),
            "clip_score": winner.get("clip_score"),
            "confidence_score": winner.get("confidence_score"),
            "final_score": winner.get("final_score"),
            "contexts": ctx_contexts,
        }

    try:
        core_path = os.path.join(unique_dir, "image_metadata_core.json")
        with open(core_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(core_meta), f, indent=2, ensure_ascii=False)
        dbg(f"[collect] core metadata saved to {core_path}")

        ctx_path = os.path.join(unique_dir, "image_metadata_context.json")
        with open(ctx_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(ctx_meta), f, indent=2, ensure_ascii=False)
        dbg(f"[collect] context metadata saved to {ctx_path}")

        # ALSO maintain global metadata files at <images_root>/UniqueImages/
        global_unique = os.path.join(images_root, "UniqueImages")
        os.makedirs(global_unique, exist_ok=True)

        def _merge_json(path: str, new_obj: dict):
            old = {}
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as ff:
                        old = json.load(ff) or {}
            except Exception:
                old = {}
            old.update(new_obj)
            with open(path, "w", encoding="utf-8") as ff:
                json.dump(_json_safe(old), ff, indent=2, ensure_ascii=False)

        _merge_json(os.path.join(global_unique, "image_metadata_core.json"), core_meta)
        _merge_json(os.path.join(global_unique, "image_metadata_context.json"), ctx_meta)

    except Exception as e:
        dbg(f"[collect][err][meta] {e}")

    dbg(f"[collect] winners={len(unique_paths)} saved to {unique_dir}")
    return unique_paths



def _embed_prompt_to_list(text: str):
    if _EMBED_MODEL is None:
        return None
    v = _EMBED_MODEL.encode(text)
    if hasattr(v, "tolist"):
        v = v.tolist()
    else:
        v = list(v)
    return v


def research_many(prompt_to_topic: Dict[str, str], max_workers: int = 3) -> Dict[str, List[str]]:
    """
    Parallel prompt research:
      - Phase A (parallel): crawl/download/metadata for each prompt (NO SigLIP)
      - Phase B (serial): SigLIP rerank per prompt in order, using ONE loaded backend
    """
    out: Dict[str, List[str]] = {}

    items = [(p, t) for p, t in (prompt_to_topic or {}).items()
             if isinstance(p, str) and p.strip() and isinstance(t, str) and t.strip()]

    if not items:
        return out

    # Phase A: parallel gather
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        for prompt, topic in items:
            pid = _prompt_key(prompt)
            dbg(f"[MULTI][SUBMIT] prompt='{prompt}' topic='{topic}' prompt_id={pid}")
            futures.append(ex.submit(research, prompt, topic, pid, False, None))  # rank_images=False

        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                dbg(f"[MULTI][ERR] worker failed: {e}")

    # Phase B: serial rank, SigLIP loaded once
    backend = SiglipBackend()

    for prompt, topic in items:
        pid = _prompt_key(prompt)
        dbg(f"[MULTI][RANK] prompt='{prompt}' prompt_id={pid}")

        base_ctx_embedding = _embed_prompt_to_list(prompt)

        winners = collect_unique_images(
            IMAGES_PATH,
            base_ctx_embedding=base_ctx_embedding,
            prompt_text=prompt,
            prompt_filter_text=prompt,
            unique_subdir=pid,
            dbg=dbg,
            backend=backend,
        )
        out[prompt] = winners

    return out




# =========================
# MAIN
# =========================
def research(
    query: str,
    subj: str,
    prompt_id: str | None = None,
    rank_images: bool = True,
    siglip_backend: SiglipBackend | None = None,
    api_fallback_only: bool = True,   # NEW: APIs run only if DDG + sources yield 0
):
    prompt_id = prompt_id or _prompt_key(query)
    base_ctx_embedding = _embed_prompt_to_list(query)

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
    api_sources = []
    web_sources = []
    for src in sources:
        if src.type == "API":
            api_sources.append(src)
        else:
            web_sources.append(src)

    # Snapshot what we already have for this prompt (pre-run)
    before = _prompt_image_paths(query, prompt_id)

    # ---- 1) DDG in background (so Stage2 overlaps DDG wait times) ----
    ddg_done = threading.Event()
    ddg_err = {"exc": None}

    def _run_ddg():
        try:
            ddg_cc_image_harvest(
                query=query,
                target_count=5,
                lemma_obj=lemma_obj_query,
                encoder=encoder,
                query_embedding=query_embedding,
                base_ctx_embedding=base_ctx_embedding,
            )
        except Exception as e:
            ddg_err["exc"] = e
        finally:
            ddg_done.set()

    ddg_thr = threading.Thread(target=_run_ddg, name=f"DDG-{prompt_id}", daemon=True)
    ddg_thr.start()

    # ---- 2) Stage2 runs on main thread (Playwright stays safe) ----
    stage2_total = 0
    sources_used = 0

    for src in web_sources:
        if sources_used >= MAX_WEB_SOURCES_PER_PROMPT:
            dbg(f"[NOAPI] stop: reached MAX_WEB_SOURCES_PER_PROMPT={MAX_WEB_SOURCES_PER_PROMPT}")
            break

        remaining = MAX_STAGE2_IMAGES_PER_PROMPT - stage2_total
        if remaining <= 0:
            dbg(f"[NOAPI] stop: reached MAX_STAGE2_IMAGES_PER_PROMPT={MAX_STAGE2_IMAGES_PER_PROMPT}")
            break

        dbg(f"opened source {src.name}, has no api, starting process.. (remaining_stage2_cap={remaining})")

        saved = handle_result_no_api(
            src, query, subj,
            hard_image_cap=5,
            encoder=encoder,
            query_embedding=query_embedding,
            base_ctx_embedding=base_ctx_embedding,
            global_image_cap=remaining,
        )

        stage2_total += len(saved or [])
        sources_used += 1

    # ---- wait for DDG to finish before deciding API fallback ----
    ddg_done.wait()
    if ddg_err["exc"] is not None:
        dbg(f"[DDG][ERR] {ddg_err['exc']}")


    # Check if DDG + Stage2 produced anything new for this prompt
    after_non_api = _prompt_image_paths(query, prompt_id)
    new_non_api = len(after_non_api - before)
    dbg(f"[NONAPI] new_images={new_non_api} (ddg + stage2) prompt_id={prompt_id}")

    # ---- 3) APIs: run only if fallback condition met (or api_fallback_only=False) ----
    run_apis = (not api_fallback_only) or (new_non_api == 0)

    if run_apis:
        dbg("[API] running APIs (fallback condition met)")
        for src in api_sources:
            dbg(f"opened source {src.name} as api (parser='{src.name}'), sending request..")

            parser = PARSERS.get((src.name or "").lower())
            if not parser:
                dbg(f"[API][{src.name}] no parser registered; skipping")
                continue

            settings = {
                "query_field": query,
                "limit_field": 5,
                "pagination_field": 1,
            }

            status, data, built = send_request(src, settings)
            if not status or int(status) != 200:
                dbg(f"[API][{src.name}] failed status={status}")
                src.img_paths = []
                continue

            try:
                parser(
                    src,
                    data,
                    query=query,
                    prompt_id=prompt_id,
                    encoder=encoder,
                    query_embedding=query_embedding,
                    base_ctx_embedding=base_ctx_embedding,
                )
            except Exception as e:
                dbg(f"[API][{src.name}] parse failed: {e}")
                src.img_paths = []
    else:
        dbg("[API] skipped (non-api research produced images)")

    # ---- 4) Ensure per-source images get prompt metadata (unchanged behavior) ----
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
                "prompt_id": prompt_id,
            }
            if base_ctx_embedding is not None:
                meta["prompt_embedding"] = base_ctx_embedding
            _register_image_metadata(p, meta)

    if not rank_images:
        dbg("[RANK] skipped (rank_images=False)")
        return []

    unique = collect_unique_images(
        IMAGES_PATH,
        base_ctx_embedding,
        prompt_text=query,
        prompt_filter_text=query,
        unique_subdir=prompt_id,
        dbg=dbg,
        backend=siglip_backend,
    )

    dbg(f"Unique images:{unique}")
    return unique
