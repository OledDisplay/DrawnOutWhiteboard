from __future__ import annotations
import re
from typing import List, Tuple, Optional, Iterable, Set, Any
from bs4 import BeautifulSoup, Tag
import numpy as np
from urllib.parse import urlparse, unquote, urljoin

# ---------- lightweight helpers ----------

def dbg(*args):
        print(*args, flush=True)

_STOP = {
    "a","an","the","and","or","but","if","while","with","of","for","to","in","on",
    "at","by","from","as","is","are","was","were","be","been","being","that","this",
    "these","those","it","its","their","his","her","your","our","not","no","do","does",
    "did","can","could","may","might","should","would","will"
}

_TOKEN_RX = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)?")  # keeps hyphenated like "prokaryotic-like"

def _tok(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RX.findall(text or "")]

def _normalize_token(t: str) -> str:
    """
    Very small, safe normalizer:
      - lowercases
      - strips simple possessives (’s, 's)
      - collapses basic English plurals: cells -> cell, lysosomes -> lysosome
    This is enough to unify 'cell', 'cells', 'cell’s' etc.
    """
    t = t.lower()

    # strip common possessives
    if t.endswith("’s") or t.endswith("'s"):
        t = t[:-2]

    # crude plural → singular (only if reasonably long)
    if len(t) > 3:
        if t.endswith("ies"):
            # bodies -> body
            t = t[:-3] + "y"
        elif t.endswith("ses"):
            # processes -> process
            t = t[:-2]
        elif t.endswith("s") and not t.endswith("ss"):
            # cells -> cell, organelles -> organelle
            t = t[:-1]

    return t

def _content_tokens(text: str) -> List[str]:
    """
    Tokens used for scoring:
      - normalized via _normalize_token
      - stopwords removed
      - length > 1
    """
    out: List[str] = []
    for raw in _tok(text):
        t = _normalize_token(raw)
        if t in _STOP:
            continue
        if len(t) <= 1:
            continue
        out.append(t)
    return out



def _bigrams(tokens: List[str]) -> Set[Tuple[str,str]]:
    return set(zip(tokens, tokens[1:]))

def _unique(seq: Iterable[str]) -> List[str]:
    s, out = set(), []
    for x in seq:
        if x not in s:
            s.add(x); out.append(x)
    return out

def _text_of(node: Optional[Tag]) -> str:
    return node.get_text(" ", strip=True) if (node and isinstance(node, Tag)) else ""

def _nearest_heading(node: Tag) -> Optional[Tag]:
    # walk up; grab first heading in ancestors or previous siblings
    cur = node
    while cur and isinstance(cur, Tag):
        # self or previous siblings
        for sib in [cur] + list(cur.previous_siblings):
            if isinstance(sib, Tag) and sib.name and sib.name.lower() in ("h1","h2","h3","h4"):
                return sib
        cur = cur.parent
    return None

def _split_sentences(text: str) -> List[str]:
    """
    Very lightweight sentence splitter: split on punctuation followed by whitespace.
    Good enough for short HTML paragraphs.
    """
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def _find_context_window(text: str, base_query: str, profile: "QueryProfile") -> str:
    """
    Take current, previous and next sentence around the first hit of the base_query
    (or any query token); fall back to the middle sentence if nothing matches.
    """
    sents = _split_sentences(text)
    if not sents:
        return text or ""

    q = (base_query or "").strip().lower()
    hit_idx: Optional[int] = None

    # First try exact phrase
    if q:
        for i, sent in enumerate(sents):
            if q in sent.lower():
                hit_idx = i
                break

    # Fallback: any query token
    if hit_idx is None and profile.query_tokens:
        q_set = set(profile.query_tokens)
        for i, sent in enumerate(sents):
            toks = set(_content_tokens(sent))
            if q_set & toks:
                hit_idx = i
                break

    # Fallback: middle sentence
    if hit_idx is None:
        hit_idx = len(sents) // 2

    start = max(0, hit_idx - 1)
    end = min(len(sents), hit_idx + 2)  # [start, end)
    return " ".join(sents[start:end])

def _to_vec(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr

def _cosine_sim(a: Any, b: Any) -> float:
    va = _to_vec(a)
    vb = _to_vec(b)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(va.dot(vb) / (na * nb))

def _build_fuzzy_phrase_rx(phrase: str) -> Optional[re.Pattern]:
    """
    Build a regex for the base phrase that:
      - normalizes tokens
      - allows the *last* word to be plural/possessive:
        'eukaryotic cell' -> matches 'eukaryotic cell', 'eukaryotic cells', 'eukaryotic cell’s'
    """
    phrase = (phrase or "").strip()
    if not phrase:
        return None

    toks = _tok(phrase)
    if not toks:
        return None

    norm = [_normalize_token(t) for t in toks]
    last = norm[-1]

    # last token: allow plural/possessive variants
    last_pat = rf"{re.escape(last)}(?:['’]s|s)?"

    if len(norm) == 1:
        pat = rf"\b{last_pat}\b"
    else:
        mid = r"\s+".join(re.escape(t) for t in norm[:-1])
        pat = rf"\b{mid}\s+{last_pat}\b"

    return re.compile(pat, re.I)


# ---------- query profile ----------

class QueryProfile:
    def __init__(self, base_query: str, lemma_obj: dict):
        self.base_query = (base_query or "").strip()

        # Phrase match that tolerates plural/possessive on the last word.
        # 'eukaryotic cell' will match 'eukaryotic cells', 'eukaryotic cell’s', etc.
        self.exact_rx = _build_fuzzy_phrase_rx(self.base_query)

        # ---- split core tokens vs lemma tokens ----
        lemma_tokens: List[str] = []
        for entry in (lemma_obj or {}).values():
            for w in entry.get("lemmas", []) or []:
                w = (w or "").strip()
                if w:
                    lemma_tokens.extend(_content_tokens(w))

        # tokens coming ONLY from the user query
        self.core_tokens: List[str] = _unique(_content_tokens(self.base_query))

        # extra lemma tokens (don’t let them dilute coverage)
        all_lemma = _unique(lemma_tokens)
        self.syn_tokens: List[str] = [t for t in all_lemma if t not in self.core_tokens]

        # backwards-compat: anything that uses query_tokens keeps seeing just core tokens
        self.query_tokens: List[str] = self.core_tokens

        # order-agnostic bigrams for mild reordering tolerance, from the core tokens only
        self.query_bigrams = _bigrams(self.core_tokens)

    def any_tokens(self) -> bool:
        return bool(self.query_tokens)


# ---------- scoring ----------

def _coverage(query_tokens: List[str], cand_tokens: List[str]) -> float:
    if not query_tokens:
        return 0.0
    ct = set(cand_tokens)
    hit = sum(1 for t in set(query_tokens) if t in ct)
    return hit / float(len(set(query_tokens)))

def _density(query_tokens: List[str], cand_tokens: List[str]) -> float:
    if not cand_tokens:
        return 0.0
    ct = set(cand_tokens)
    hits = sum(1 for t in cand_tokens if t in ct and t in query_tokens)
    # normalize to per 50 tokens, capped
    return min(1.0, (hits / max(1, len(cand_tokens))) * 50.0)

def _proximity_span(query_tokens: List[str], cand_tokens: List[str]) -> float:
    # 1.0 = tight cluster, 0.0 = very dispersed
    pos = [i for i, t in enumerate(cand_tokens) if t in query_tokens]
    if len(pos) < 2:
        return 0.5 if pos else 0.0
    span = pos[-1] - pos[0] + 1
    return max(0.0, min(1.0, (10.0 / span)))  # span <=10 tokens -> near 1.0

def _bigram_overlap(q_bi: Set[Tuple[str,str]], cand_tokens: List[str]) -> float:
    if not q_bi:
        return 0.0
    cb = _bigrams(cand_tokens)
    inter = len(q_bi & cb)
    return inter / float(len(q_bi))

def _heading_affinity(profile: QueryProfile, heading_text: str) -> float:
    if not heading_text:
        return 0.0
    htoks = _content_tokens(heading_text)
    cov = _coverage(profile.query_tokens, htoks)
    # small boost if the exact phrase appears in the heading
    exact = 1.0 if (profile.exact_rx and profile.exact_rx.search(heading_text)) else 0.0
    return min(1.0, cov * 0.7 + exact * 0.3)

def _score_block(profile: QueryProfile, text: str, heading_text: str) -> Tuple[float, dict]:
    toks = _content_tokens(text)
    if not toks:
        return 0.0, {"reason":"empty"}

    exact = 1.0 if (profile.exact_rx and profile.exact_rx.search(text)) else 0.0
    cov   = _coverage(profile.query_tokens, toks)
    dens  = _density(profile.query_tokens, toks)
    prox  = _proximity_span(profile.query_tokens, toks)
    bigr  = _bigram_overlap(profile.query_bigrams, toks)
    head  = _heading_affinity(profile, heading_text)

    # Weighted sum; exact phrase is a big lever, coverage+density next, others fine-tune
    score = (
        exact * 0.45 +
        cov   * 0.22 +
        dens  * 0.15 +
        prox  * 0.08 +
        bigr  * 0.05 +
        head  * 0.05
    )

    # Minimal guardrails: require at least some semantic presence
    if exact < 1.0 and cov < 0.34:
        score *= 0.5  # downweight weak mentions without exact phrase

    details = dict(exact=exact, coverage=cov, density=dens, proximity=prox, bigram=bigr, heading=head)
    return score, details

# ---------- candidate enumeration (text blocks) ----------

_CAND_SELECTORS = [
    "h1","h2","h3","h4",
    "p","li",
    "figure","figcaption",
    ".caption",".figure"
]

def _candidate_nodes(soup: BeautifulSoup) -> List[Tag]:
    seen: Set[Tag] = set()
    out: List[Tag] = []
    for sel in _CAND_SELECTORS:
        for n in soup.select(sel):
            if isinstance(n, Tag) and n not in seen:
                seen.add(n); out.append(n)
    return out

# ---------- image helpers (attrs + local context) ----------

def _image_attr_text_and_tokens(img: Tag) -> Tuple[str, List[str]]:
    """
    Build a text blob from img alt/title/src (filename + tail path),
    then tokenize it for matching.
    """
    parts: List[str] = []

    alt = img.get("alt")
    if isinstance(alt, str) and alt.strip():
        parts.append(alt)

    title = img.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(title)

    src = img.get("src") or img.get("data-src") or img.get("data-original")
    if isinstance(src, str) and src.strip():
        try:
            p = urlparse(src)
            path = unquote(p.path or "")
        except Exception:
            path = src
        # keep only the tail of the path; split on separators to expose keywords
        segs = [s for s in path.split("/") if s]
        tail = "/".join(segs[-3:]) if segs else ""
        tail = tail.replace("_", " ").replace("-", " ")
        if tail:
            parts.append(tail)

    text = " ".join(parts).strip()
    tokens = _content_tokens(text)
    return text, tokens

def _image_attr_match(profile: QueryProfile, img: Tag, min_cov: float = 0.34) -> Tuple[bool, dict]:
    """
    Decide if an <img> is worth considering based only on alt/title/filename/path.
    Uses coverage against query tokens + optional exact phrase in those attrs.
    """
    attr_text, attr_tokens = _image_attr_text_and_tokens(img)

    if not attr_text or not attr_tokens or not profile.any_tokens():
        return False, {
            "attr_cov": 0.0,
            "attr_exact": 0.0,
            "attr_text": attr_text[:200],
        }

    cov = _coverage(profile.query_tokens, attr_tokens)
    exact = 1.0 if (profile.exact_rx and profile.exact_rx.search(attr_text)) else 0.0
    ok = (exact >= 1.0) or (cov >= min_cov)

    details = {
        "attr_cov": cov,
        "attr_exact": exact,
        "attr_text": attr_text[:200],
    }
    return ok, details

def _image_dom_context(img: Tag) -> str:
    """
    Gather local HTML text around an image:
      - parent block (figure/div/p/li)
      - figcaption if inside <figure>
      - a couple of previous/next block siblings
      - fallback to nearest texty ancestor
    """
    blocks: List[str] = []

    parent = img.parent if isinstance(img.parent, Tag) else None
    if parent is not None:
        blocks.append(_text_of(parent))

        if parent.name and parent.name.lower() == "figure":
            for cap in parent.find_all("figcaption"):
                blocks.append(_text_of(cap))

        # up to 2 previous block siblings
        prev = parent.previous_sibling
        count = 0
        while prev is not None and count < 2:
            if isinstance(prev, Tag):
                blocks.append(_text_of(prev))
                count += 1
            prev = prev.previous_sibling

        # up to 2 next block siblings
        nxt = parent.next_sibling
        count = 0
        while nxt is not None and count < 2:
            if isinstance(nxt, Tag):
                blocks.append(_text_of(nxt))
                count += 1
            nxt = nxt.next_sibling

    # Fallback: nearest texty ancestor
    if not any(b.strip() for b in blocks):
        cur = img
        hops = 0
        while cur and isinstance(cur, Tag) and hops < 4:
            cur = cur.parent
            hops += 1
            if cur is not None and cur.name and cur.name.lower() in ("p", "li", "div", "section", "article"):
                blocks.append(_text_of(cur))
                break

    text = " ".join(b for b in blocks if b).strip()
    return text


# ---------- hit→image selection (integrated + minimal) ----------

_UI_IMG_SKIP_RX = re.compile(r"(sprite|icon|logo|avatar|spinner|loading|button|badge|emoji)", re.I)

def _img_url(img: Tag) -> Optional[str]:
    # small + robust: srcset last entry (often biggest) else src/data-src/data-original
    srcset = (img.get("srcset") or img.get("data-srcset") or "").strip()
    if srcset:
        last = srcset.split(",")[-1].strip().split(" ")[0].strip()
        if last:
            return last
    for a in ("src", "data-src", "data-original"):
        v = img.get(a)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _hit_container(node: Tag) -> Tag:
    # keep it simple: nearest block-ish container
    BLOCKS = {"p","li","figure","div","section","article","h1","h2","h3","h4"}
    cur = node
    for _ in range(6):
        if not isinstance(cur, Tag):
            break
        if (cur.name or "").lower() in BLOCKS:
            return cur
        cur = cur.parent
    return node

def _neighbors(block: Tag, n: int = 2) -> List[Tag]:
    out = [block]
    p = block.previous_sibling
    c = 0
    while p is not None and c < n:
        if isinstance(p, Tag):
            out.append(p); c += 1
        p = p.previous_sibling
    q = block.next_sibling
    c = 0
    while q is not None and c < n:
        if isinstance(q, Tag):
            out.append(q); c += 1
        q = q.next_sibling
    return out

def _cand_cov(cand_tokens: List[str], ref_set: Set[str]) -> float:
    if not cand_tokens:
        return 0.0
    s = set(cand_tokens)
    if not s:
        return 0.0
    return len(s & ref_set) / float(len(s))

def pick_best_image_for_hit(
    soup: BeautifulSoup,
    hit_node: Tag,
    hit_details: dict,
    page_url: str,
    base_query: str,
    lemma_obj: dict,
    *,
    encoder: Optional[Any] = None,
    query_embedding: Optional[Any] = None,
    window_blocks: int = 2,
    min_score: float = 0.14,
    debug: bool = True,
) -> Optional[Tuple[str, float, dict]]:
    profile = QueryProfile(base_query, lemma_obj)

    hit_ctx = (hit_details.get("ctx_text") or hit_details.get("snippet") or "") if isinstance(hit_details, dict) else ""
    hit_head = (hit_details.get("heading_text") or "") if isinstance(hit_details, dict) else ""
    hit_ref = set(_content_tokens((hit_head + " " + hit_ctx).strip()))
    if not hit_ref:
        hit_ref = set(_content_tokens(_text_of(hit_node)))

    block = _hit_container(hit_node)
    scopes = _neighbors(block, n=window_blocks)

    seen: Set[str] = set()
    cands: List[dict] = []

    for sc in scopes:
        if not isinstance(sc, Tag):
            continue

        for img in sc.find_all("img"):
            if not isinstance(img, Tag):
                continue

            raw = _img_url(img)
            if not raw or raw.lower().startswith("data:"):
                continue
            if _UI_IMG_SKIP_RX.search(raw):
                continue

            # skip nav/header/footer/sidebar images
            if img.find_parent(["nav", "header", "footer", "aside"]):
                continue

            # tiny pixel trackers
            try:
                w = int(img.get("width") or 0)
                h = int(img.get("height") or 0)
                if w > 0 and h > 0 and (w <= 32 and h <= 32):
                    continue
            except Exception:
                pass

            abs_u = urljoin(page_url or "", raw)
            if abs_u in seen:
                continue
            seen.add(abs_u)

            attr_text, attr_tokens = _image_attr_text_and_tokens(img)
            dom = _image_dom_context(img)
            dom_ctx = _find_context_window(dom, base_query, profile) if dom else ""
            dom_tokens = _content_tokens(dom_ctx)

            hit_loc = _cand_cov(dom_tokens, hit_ref)
            hit_attr = _cand_cov(attr_tokens, hit_ref)

            ok_attr, _ = _image_attr_match(profile, img, min_cov=0.34)

            # hard gate: if nothing matches the hit AND no attr match → drop
            if (hit_loc < 0.12 and hit_attr < 0.12) and not ok_attr:
                continue

            lex = 0.70 * hit_loc + 0.30 * hit_attr

            cands.append({
                "url": abs_u,
                "lex": float(lex),
                "hit_loc": float(hit_loc),
                "hit_attr": float(hit_attr),
                "ctx_text": (dom_ctx or "")[:200],
                "attr_text": (attr_text or "")[:120],
            })

    if not cands:
        if debug:
            dbg("[HIT→IMG] ❌ none")
        return None

    cands.sort(key=lambda d: d["lex"], reverse=True)

    # optional semantic bump only for top few
    if encoder is not None and query_embedding is not None:
        topm = cands[:3]
        for d in topm:
            try:
                t = (d["ctx_text"] or "").strip()
                if not t:
                    d["score"] = d["lex"]
                else:
                    sem = _cosine_sim(query_embedding, encoder.encode(t))
                    d["score"] = 0.75 * d["lex"] + 0.25 * float(sem)
            except Exception:
                d["score"] = d["lex"]
        for d in cands[3:]:
            d["score"] = d["lex"]
    else:
        for d in cands:
            d["score"] = d["lex"]

    cands.sort(key=lambda d: d["score"], reverse=True)
    best = cands[0]

    if best["score"] < float(min_score):
        if debug:
            dbg(f"[HIT→IMG] ❌ best={best['score']:.3f} < {min_score:.3f}")
        return None

    if debug:
        dbg(f"[HIT→IMG] ✅ score={best['score']:.3f} loc={best['hit_loc']:.2f} attr={best['hit_attr']:.2f}  {best['url']}")

    meta = {
        "best_img_hitLoc": best["hit_loc"],
        "best_img_hitAttr": best["hit_attr"],
        "best_img_attr_text": best["attr_text"],
        "best_img_ctx_text": best["ctx_text"],
        "best_img_score": best["score"],
    }
    return best["url"], best["score"], meta


# ---------- main entry: text hits ----------

def smart_find_hits_in_soup(
    soup: BeautifulSoup,
    base_query: str,
    lemma_obj: dict,
    *,
    page_url: str = "",
    min_score: float = 0.60,
    top_k: Optional[int] = None,
    encoder: Optional[Any] = None,
    query_embedding: Optional[Any] = None,
    sem_min_score: float = 0.45,
    return_embedding: bool = True
) -> List[Tuple[Tag, float, dict]]:
    """
    Returns a ranked list of (node, score, details). Filter by min_score and optional top_k.
    """
    profile = QueryProfile(base_query, lemma_obj)
    if not profile.any_tokens() and not profile.exact_rx:
        return []

    use_semantic = encoder is not None and query_embedding is not None

    cands = _candidate_nodes(soup)
    scored: List[Tuple[Tag,float,dict]] = []

    # Precompute nearest heading texts to avoid repeated DOM walk
    heading_cache = {}
    for n in cands:
        if n in heading_cache:
            htxt = heading_cache[n]
        else:
            h = _nearest_heading(n)
            htxt = _text_of(h)
            heading_cache[n] = htxt

        text = _text_of(n)
        s, details = _score_block(profile, text, htxt)
        if s < min_score:
            continue

        # Build context window around the hit
        ctx_text = _find_context_window(text, base_query, profile)
        details["heading_text"] = htxt[:120]
        details["snippet"] = text[:160]
        details["ctx_text"] = ctx_text[:240]

        # Optional semantic filtering with MiniLM (or any sentence encoder)
        if use_semantic:
            ctx_emb = encoder.encode(ctx_text)
            sem_score = _cosine_sim(query_embedding, ctx_emb)
            details["sem_score"] = sem_score
            if return_embedding:
                details["ctx_embedding"] = ctx_emb

            # --- DEBUG: log every rejection by semantic context ---
            if sem_score < sem_min_score:
                try:
                    snip = ctx_text.replace("\n", " ")
                    if len(snip) > 160:
                        snip = snip[:160] + "..."
                except Exception:
                    snip = ""
                print(
                    f"[HIT-REJECT][CTX] sem_score={sem_score:.3f} < threshold={sem_min_score:.3f}  "
                    f"snippet='{snip}'"
                )
                continue
            # --- end debug block ---

        # attach ONE best image for the hit (above/below only)
        picked = pick_best_image_for_hit(
            soup=soup,
            hit_node=n,
            hit_details=details,
            page_url=page_url,
            base_query=base_query,
            lemma_obj=lemma_obj,
            encoder=encoder,
            query_embedding=query_embedding,
            window_blocks=2,
            min_score=0.14,
            debug=True,
        )
        if picked:
            img_url, img_score, img_meta = picked
            details["best_img_url"] = img_url
            details["best_img_score"] = img_score
            details.update(img_meta)

        scored.append((n, s, details))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k] if top_k else scored

# ---------- second pass: image-driven hits ----------

def smart_find_image_hits_in_soup(
    soup: BeautifulSoup,
    base_query: str,
    lemma_obj: dict,
    *,
    min_attr_cov: float = 0.34,        # gating on filename/alt/title coverage
    min_score: float = 0.55,           # reuse same main score threshold for context
    encoder: Optional[Any] = None,
    query_embedding: Optional[Any] = None,
    sem_min_score: float = 0.42,
    return_embedding: bool = True
) -> List[Tuple[Tag, float, dict]]:
    """
    Second-pass image hit finder.

    For *every* <img>:
      1) Check filename/path + alt/title coverage vs query/lemmas.
         - If coverage is low and no exact phrase -> skip.
      2) If it passes, collect local DOM text around the image
         (parent, figcaption, nearby siblings) and build a context window.
      3) Run the same scoring + optional semantic threshold on that context.

    Returns a ranked list of (img_tag, score, details), with:
      - attr_cov / attr_exact / attr_text
      - heading_text / snippet / ctx_text
      - sem_score / ctx_embedding (if semantic enabled)
    """
    profile = QueryProfile(base_query, lemma_obj)
    if not profile.any_tokens() and not profile.exact_rx:
        return []

    use_semantic = encoder is not None and query_embedding is not None
    results: List[Tuple[Tag, float, dict]] = []

    for img in soup.find_all("img"):
        if not isinstance(img, Tag):
            continue

        # 1) attribute-level gate (filename/path + alt + title)
        ok_attr, attr_details = _image_attr_match(profile, img, min_cov=min_attr_cov)
        if not ok_attr:
            continue

        # 2) local DOM context around the image
        raw_ctx = _image_dom_context(img)
        if not raw_ctx:
            continue

        heading_tag = _nearest_heading(img)
        heading_text = _text_of(heading_tag)

        ctx_text = _find_context_window(raw_ctx, base_query, profile)
        score, score_details = _score_block(profile, ctx_text, heading_text)
        if score < min_score:
            continue

        details = dict(attr_details)
        details.update(score_details)

        # STANDARDIZED CONTEXT METADATA
        details["source"] = "image_context"
        details["heading_text"] = heading_text[:120]
        details["snippet"] = ctx_text[:160]
        details["ctx_text"] = ctx_text[:240]
        details["ctx_score"] = score  # unified lexical context score

        # Optional semantic filtering on the context text
        if use_semantic and ctx_text:
            ctx_emb = encoder.encode(ctx_text)
            sem_score = _cosine_sim(query_embedding, ctx_emb)
            details["sem_score"] = sem_score
            details["ctx_sem_score"] = sem_score  # alias
            if return_embedding:
                details["ctx_embedding"] = ctx_emb
            if sem_score < sem_min_score:
                continue

        results.append((img, score, details))


    results.sort(key=lambda x: x[1], reverse=True)
    return results
