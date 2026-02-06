import os, re, requests
from urllib.parse import urlparse, unquote, urljoin
from bs4 import BeautifulSoup
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = os.path.join(ROOT_DIR, "source_urls")
IMAGES_PATH = os.path.join(ROOT_DIR, "ResearchImages")
os.makedirs(SOURCE_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)

DEBUG = True
def dbg(*a): 
    if DEBUG: print(*a, flush=True)

# --- tiny shared helpers (minimal, not bloated) -----------------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _filename_from_url(u: str, idx: int) -> str:
    base = unquote(os.path.basename(urlparse(u).path)) or f"file_{idx}"
    # ensure an extension (lightweight)
    if not os.path.splitext(base)[1]:
        base += ".img"
    # trim junk after ? or #
    base = re.sub(r"[?#].*$", "", base)
    # filesystem-safe-ish
    base = re.sub(r"[^A-Za-z0-9._\- ]+", "_", base)[:120] or f"file_{idx}.img"
    return base

def _download_to(u: str, dest_dir: str, idx: int) -> str | None:
    try:
        _ensure_dir(dest_dir)
        path = os.path.join(dest_dir, _filename_from_url(u, idx))
        if os.path.exists(path):
            return path

        from urllib.parse import urlparse
        pu = urlparse(u)
        referer = f"{pu.scheme}://{pu.netloc}" if pu.scheme and pu.netloc else None
        headers = {
            "User-Agent": "diag-scrape/0.3 (+edu-diagrams)",
        }
        if referer:
            headers["Referer"] = referer

        with requests.get(u, headers=headers, stream=True, timeout=30, allow_redirects=True) as r:
            r.raise_for_status()
            ct = (r.headers.get("Content-Type") or "").lower()
            # image gate: content-type OR url extension
            if not (ct.startswith("image/") or re.search(r"\.(png|jpe?g|gif|webp|svg|tiff?)($|\?)", u, re.I)):
                return None
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
        return path
    except Exception:
        return None


# =========================
# Wikimedia Commons (query API) — cap 5
# =========================
def _strip_html_text(s: str) -> str:
    try:
        return BeautifulSoup(s or "", "html.parser").get_text(" ", strip=True)
    except Exception:
        return (s or "").strip()


def parse_wikimedia(src, data, **kwargs):
    """
    Wikimedia Commons (query API) — cap 5
    Registers ctx_text/ctx_embedding so API images participate in SigLIP ranking.
    Applies CC/PD check from extmetadata (hard filter).
    """
    query = (kwargs.get("query") or "").strip()
    prompt_id = kwargs.get("prompt_id")
    encoder = kwargs.get("encoder")
    base_ctx_embedding = kwargs.get("base_ctx_embedding")

    if not isinstance(data, dict):
        src.img_paths = []
        return []

    pages = (data.get("query") or {}).get("pages") or {}
    items = []

    # collect up to 8 raw, we’ll hard-cap after filtering/downloading
    for _, page in pages.items():
        title = page.get("title") or ""
        pageid = page.get("pageid")
        file_page_url = f"https://commons.wikimedia.org/?curid={pageid}" if pageid else ""

        for ii in (page.get("imageinfo") or []):
            u = ii.get("url") or ii.get("thumburl")
            if not u:
                continue

            extm = (ii.get("extmetadata") or {}) if isinstance(ii, dict) else {}
            def _m(k):
                v = extm.get(k)
                if isinstance(v, dict):
                    return _strip_html_text(v.get("value") or "")
                if isinstance(v, str):
                    return _strip_html_text(v)
                return ""

            license_url = _m("LicenseUrl")
            license_short = _m("LicenseShortName")
            usage_terms = _m("UsageTerms")
            permissions = _m("Permissions")
            credit = _m("Credit")
            artist = _m("Artist")
            desc = _m("ImageDescription") or _m("ObjectName")

            lic_blob = " ".join([license_url, license_short, usage_terms, permissions]).strip()
            is_cc = False
            for pat in CC_PATTERNS:
                try:
                    if re.search(pat, lic_blob, re.I):
                        is_cc = True
                        break
                except Exception:
                    continue

            # hard CC/PD requirement for Commons too
            if not is_cc:
                continue

            ctx_text = " ".join(x for x in [
                query,
                title.replace("File:", "").replace("_", " "),
                desc,
                artist,
                credit,
                license_short,
            ] if x).strip()

            items.append({
                "image_url": u,
                "page_url": file_page_url or "",
                "ctx_text": ctx_text,
                "license": license_short,
                "license_url": license_url,
            })

            if len(items) >= 8:
                break
        if len(items) >= 8:
            break

    if not items:
        src.img_paths = []
        return []

    session = requests.Session()
    session.headers.update({"User-Agent": "diag-scrape/0.2 (+research/cc-check)"})
    dest_dir = os.path.join(IMAGES_PATH, src.name)
    saved = []

    for i, it in enumerate(items):
        if len(saved) >= 5:
            break
        u = it["image_url"]
        p = download_image(session, u, dest_dir)
        if not p:
            continue
        saved.append(p)

        ctx_embedding = None
        if encoder is not None and it["ctx_text"]:
            try:
                v = encoder.encode(it["ctx_text"])
                ctx_embedding = v.tolist() if hasattr(v, "tolist") else list(v)
            except Exception:
                ctx_embedding = None

        meta = {
            "source_kind": "api",
            "source_name": src.name,
            "base_context": query,
            "prompt_id": prompt_id,
            "page_url": it.get("page_url") or "",
            "image_url": u,
            "ctx_text": it.get("ctx_text") or "",
            "ctx_embedding": ctx_embedding,
            # keep it low: API is fallback-y in your system
            "ctx_confidence": 0.12,
            "license": it.get("license") or "",
            "license_url": it.get("license_url") or "",
        }
        if base_ctx_embedding is not None:
            meta["prompt_embedding"] = base_ctx_embedding

        _register_image_metadata(p, meta)
        dbg(f"[wikimedia][api] saved={p}")

    src.img_paths = saved
    return saved


def parse_openverse(src, data, **kwargs):
    """
    Openverse images API — cap 5
    Registers ctx_text/ctx_embedding so API images participate in SigLIP ranking.
    """
    query = (kwargs.get("query") or "").strip()
    prompt_id = kwargs.get("prompt_id")
    encoder = kwargs.get("encoder")
    base_ctx_embedding = kwargs.get("base_ctx_embedding")

    if not isinstance(data, dict):
        src.img_paths = []
        return []

    results = data.get("results", []) or []
    items = []

    for r in results:
        u = (r.get("url") or "").strip()
        if not u:
            continue

        title = (r.get("title") or "").strip()
        creator = (r.get("creator") or "").strip()
        license_ = (r.get("license") or "").strip()
        lic_ver = (r.get("license_version") or "").strip()
        prov = (r.get("provider") or r.get("source") or "").strip()
        landing = (r.get("foreign_landing_url") or r.get("detail_url") or "").strip()

        tags = []
        for t in (r.get("tags") or []):
            if isinstance(t, dict):
                nm = (t.get("name") or "").strip()
                if nm:
                    tags.append(nm)
            elif isinstance(t, str):
                t2 = t.strip()
                if t2:
                    tags.append(t2)

        ctx_text = " ".join(x for x in [
            query,
            title,
            creator,
            " ".join(tags[:10]),
            f"{license_} {lic_ver}".strip(),
            prov,
        ] if x).strip()

        items.append({
            "image_url": u,
            "page_url": landing,
            "ctx_text": ctx_text,
            "license": f"{license_} {lic_ver}".strip(),
            "provider": prov,
        })

        if len(items) >= 8:
            break

    if not items:
        src.img_paths = []
        return []

    session = requests.Session()
    session.headers.update({"User-Agent": "diag-scrape/0.2 (+research/cc-check)"})
    dest_dir = os.path.join(IMAGES_PATH, src.name)
    saved = []

    for i, it in enumerate(items):
        if len(saved) >= 5:
            break
        u = it["image_url"]
        p = download_image(session, u, dest_dir)
        if not p:
            continue
        saved.append(p)

        ctx_embedding = None
        if encoder is not None and it["ctx_text"]:
            try:
                v = encoder.encode(it["ctx_text"])
                ctx_embedding = v.tolist() if hasattr(v, "tolist") else list(v)
            except Exception:
                ctx_embedding = None

        meta = {
            "source_kind": "api",
            "source_name": src.name,
            "base_context": query,
            "prompt_id": prompt_id,
            "page_url": it.get("page_url") or "",
            "image_url": u,
            "ctx_text": it.get("ctx_text") or "",
            "ctx_embedding": ctx_embedding,
            "ctx_confidence": 0.10,
            "license": it.get("license") or "",
            "provider": it.get("provider") or "",
        }
        if base_ctx_embedding is not None:
            meta["prompt_embedding"] = base_ctx_embedding

        _register_image_metadata(p, meta)
        dbg(f"[openverse][api] saved={p}")

    src.img_paths = saved
    return saved



# =========================
# USGS / ScienceBase (items API) — cap 5
# =========================
def parse_usgs(src, data):
    """
    ScienceBase:
      - /catalog/items?q=... returns items[]
      - files[] or attachments[] hold image links
      - item['link'] is a list; pick rel=='self' or fall back to /catalog/item/<id>
    """
    if not isinstance(data, dict):
        src.img_paths = []
        return []

    items = data.get("items", []) or []
    all_saved = []

    def _item_self_url(item):
        # 1) link array with rel == 'self'
        link = item.get("link")
        if isinstance(link, list):
            for l in link:
                if isinstance(l, dict) and l.get("rel") == "self" and l.get("url"):
                    return l["url"]
        # 2) explicit self.linkUrl
        self_obj = item.get("self") or {}
        if isinstance(self_obj, dict) and self_obj.get("linkUrl"):
            return self_obj["linkUrl"]
        # 3) fallback by id
        iid = item.get("id")
        if iid:
            return f"https://www.sciencebase.gov/catalog/item/{iid}"
        return "https://www.sciencebase.gov/catalog/"

    for item_idx, item in enumerate(items, start=1):
        files = item.get("files") or item.get("attachments") or []
        if not files:
            continue

        base_url = _item_self_url(item)
        dest_dir = os.path.join(IMAGES_PATH, src.name, f"item_{item_idx}")
        saved = []

        for j, f in enumerate(files):
            u = f.get("url") or f.get("downloadUri")
            if not u:
                href = f.get("href") or f.get("uri")
                if href:
                    u = urljoin(base_url, href)
            if not u:
                continue

            ctype = (f.get("contentType") or "").lower()
            looks_img = ctype.startswith("image/") or re.search(r"\.(png|jpe?g|gif|webp|svg|tiff?)($|\?)", u, re.I)
            if not looks_img:
                continue

            p = _download_to(u, dest_dir, j)
            if p:
                saved.append(p)

        if saved:
            dbg(f"[usgs] item {item_idx} saved={len(saved)}")
            all_saved.extend(saved)

    src.img_paths = all_saved
    return all_saved



# =========================
# PLOS Search API — cap 5 (total images)
# =========================
def parse_plos(src, data):
    if not isinstance(data, dict):
        src.img_paths = []
        return []

    docs = (data.get("response") or {}).get("docs", []) or []
    dois = []
    for d in docs[:5]:
        doi = d.get("doi") or d.get("id")
        if doi:
            dois.append(doi)
    if not dois:
        src.img_paths = []
        return []

    headers = {"User-Agent": "diag-scrape/0.1"}
    batch_root = os.path.join(IMAGES_PATH, "plos", src.name)
    _ensure_dir(batch_root)
    saved_all = []

    for a_idx, doi in enumerate(dois, start=1):
        if len(saved_all) >= 5:
            break
        try:
            r = requests.get(f"https://doi.org/{doi}", headers=headers, timeout=30, allow_redirects=True)
            r.raise_for_status()
            article_url = r.url
        except Exception:
            continue

        try:
            art = requests.get(article_url, headers=headers, timeout=30)
            art.raise_for_status()
        except Exception:
            continue

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(art.text, "html.parser")
        img_urls = []

        # gather, but stop early if we already have enough
        for img in soup.select("figure img"):
            if len(img_urls) >= 5:
                break
            src_attr = img.get("src") or img.get("data-src")
            if src_attr:
                img_urls.append(urljoin(article_url, src_attr))

        if len(img_urls) < 5:
            for a in soup.select("figure a[href]"):
                if len(img_urls) >= 5:
                    break
                href = a.get("href") or ""
                if re.search(r"\.(png|jpe?g|gif|webp|svg|tiff?)($|\?)", href, re.I):
                    img_urls.append(urljoin(article_url, href))

        # dedupe + cap 5
        img_urls = list(dict.fromkeys(img_urls))[:5]
        if not img_urls:
            continue

        dest_dir = os.path.join(batch_root, f"article_{a_idx}")
        for j, u in enumerate(img_urls):
            if len(saved_all) >= 5:
                break
            p = _download_to(u, dest_dir, j)
            if p:
                saved_all.append(p)

    src.img_paths = saved_all
    return saved_all

