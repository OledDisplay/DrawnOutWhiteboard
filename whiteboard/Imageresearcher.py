import os, json, copy, requests, re
from urllib.parse import urlparse, unquote, urljoin
from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup
import tldextract
from collections import deque

def parse_wikimedia(): pass
def parse_openverse(): pass
def parse_plos(): pass
def parse_usgs(): pass
    
    
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

PARSERS = {
    "wikimedia": parse_wikimedia,
    "openverse": parse_openverse,
    "plos": parse_plos,
    "usgs": parse_usgs
}

#SOURCE STRUCTURE. NORMAL SEARCH ENGINE RESULTS ARE ALSO TAKEN SO SOURCES ARENT NECESAARRY
# { 
#  "url":
#  "Hasapi":                                                                            
#  "TemplateEquivalent": {  //PUT NAMES OF EQUIVALENT FIELDS IN YOUR API REQUEST FORMAT (IF API)
#    "query_field": 
#    "limit_field": 
#    "pagination_field": 
#    "format_field": 
#  },
#  "FullTemplate": {   //FULL TEMPLATE FOR SUBMITTING A REQUEST TO THE API              (IF API)
#  }
#   "SubjectUrls":{    //PUT URLS TO SCRAPE FOR SUBJECTS YOU ARE USING - OPTIONAL       (IF NO API)
#    "Maths" : "",
#    "Physics" : "",
#    "Biology" : "" , 
#    "Chemistry" : "",
#    "Geography" : ""
#  }
# }

#IF MULTIPLE URLS FOR SCRAPE - CLONE TOPIC, EX: "Biology" : "" , "Biology" : "" , 

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
          self.subjurls = json_obj.get("SubjectUrls")
        self.img_paths = None

def read_sources():
    sources = []
    for filename in os.listdir(SOURCE_PATH):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(SOURCE_PATH, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            sources.append(Source(data, filename))
    return sources

def build_params_from_settings(source: Source, settings: dict) -> dict:
    """
    Clone the source FullTemplate and overwrite the API-specific parameter names
    using TemplateEquivalent mappings fed with values from 'settings'.
    Rules:
      - For each key in 'settings' (e.g., 'query_field'), find its API-specific
        parameter name in source.TemplateEQ[key]. If present and not None,
        set template[ api_param_name ] = settings[key].
      - 'format_field' + 'format_value' are coupled:
          if TemplateEquivalent['format_field'] exists AND settings['format_value'] exists,
          set template[ format_field_name ] = settings['format_value'].
      - If an equivalent field name does not exist or is None, skip it.
    """
    params = copy.deepcopy(source.Template)

    # Generic field mappings (except the special-cased format pair)
    for k, v in settings.items():
        api_field_name = source.TemplateEQ.get(k)
        if api_field_name:  # ignore None or missing
            params[api_field_name] = v

    return params

def send_request(source: Source, settings: dict):
    built = build_params_from_settings(source, settings)

    if source.type != "API":
        return None, None, built  # nothing to send; you still get the built params

    headers = {"User-Agent": "diag-scrape/0.1"}
    try:
        resp = requests.get(source.url, params=built, headers=headers, timeout=20)
        # Try JSON first; fall back to text
        try:
            data = resp.json()
        except ValueError:
            data = resp.text
        return resp.status_code, data
    except requests.RequestException as e:
        return None, f"REQUEST_ERROR: {e}", built


def handle_result_api(source, data):
    parser = PARSERS.get(source.get("name"))
    if not parser:
        print(f"No parser found for {source.get("name")}")
        return []
    return parser(source, data)

def search_for(url, subj, query): #return paths to locally saved images. Take all the text -> search for hits on phrases first, then trully scrape
    
def find_filtered_subdomains(base_url, lemma_obj, max_pages=120, timeout=8):
    ext = tldextract.extract(urlparse(base_url).hostname or "")
    registrable = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    if not registrable: return [], set()

    phrases = {s.strip().lower() for v in lemma_obj.values() for s in v.get("lemmas", []) if len(s.strip())>1}
    clean = lambda u: unquote(re.sub(r"[#].*$", "", re.sub(r"[?&](utm_[^=&]+|fbclid|gclid)=[^&]*", "", u)))
    same_reg = lambda h: (lambda e: f"{e.domain}.{e.suffix}" if e.suffix else e.domain)(tldextract.extract(h)) == registrable

    visited, q = set(), deque([clean(base_url)])
    same_site, subs, valid = set(), set(), set()
    headers = {"User-Agent": "subfinder/1.0"}

    while q and len(visited) < max_pages:
        cur = q.popleft()
        if cur in visited: continue
        visited.add(cur)
        try:
            r = requests.get(cur, headers=headers, timeout=timeout)
            if not r.ok or "text/html" not in (r.headers.get("Content-Type","")): continue
        except requests.RequestException:
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = clean(urljoin(cur, a["href"]))
            host = urlparse(href).hostname or ""
            if not host or not same_reg(host): continue
            same_site.add(href)
            sub = tldextract.extract(host).subdomain
            if sub: subs.add(f"{sub}.{registrable}")
            if href not in visited: q.append(href)

    for u in same_site:
        if any(p in u.lower() for p in phrases):
            valid.add(u)

    return sorted(valid), subs


def get_limited_lemmas(word, per_synset_limit=4, pos_filter=None):
    """
    Get all lemma names (synonyms) for a given word,
    limiting the number of lemmas per synset.

    Args:
        word (str): the word to search
        per_synset_limit (int): how many lemmas to take from each synset
        pos_filter (str or list): optional, restrict by part of speech ('n', 'v', 'a', 'r')

    Returns:
        dict: mapping from synset name -> list of lemma names
    """
    # Normalize POS filter
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



def handle_result_no_api(source, query, subj):
    if source.subjecturls:
      for current_subj, url in source.subjecturls:
        if(current_subj == subj and url):
            search_for(url, query, subj) # append local image paths, so that we can have multiple scrape endpoints for a subj. Finally add the full thing to image paths

    find_filtered_subdomains(source.url,get_limited_lemmas(subj, 4), 60, 8)
            
        
        



if __name__ == "__main__":
    
    subj = "Biology" #we are getting these with llm
    query = "Cell structure"
    # Example generic settings you will swap at runtime
    settings = {
        "query_field": query,
        "limit_field": "10",
        "pagination_field": "temp_pagination_token",
        "format_field": "ignored_here",   # ignored value; only the field name matters
        "format_value": "json"
    }
    
    subject = "Maths"

    sources = read_sources()
    for src in sources:
        if src.type == "API":
         status, data = send_request(src, settings)
         print(f"sent request, responded {status}")
         handle_result_api(src, data)
        else:
         handle_result_no_api(src, query, subj)
    
    #generic web search as a sort of fallback
         









#####################
#vibe coded parsers

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
        except Exception:
            # Skip failed downloads; continue with the rest
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
        except Exception:
            # Skip failed downloads; continue with the rest
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
        except Exception:
            continue

        # Fetch the article HTML
        try:
            art = requests.get(article_url, headers=headers, timeout=30)
            art.raise_for_status()
        except Exception:
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
            except Exception:
                # Skip failed downloads
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
        except Exception:
            # Skip failed downloads; continue with others
            continue

    src.img_paths = saved_paths
    return saved_paths

