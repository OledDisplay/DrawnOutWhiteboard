# smart_hits.py
from __future__ import annotations
import re
from typing import List, Tuple, Optional, Iterable, Set
from bs4 import BeautifulSoup, Tag

# ---------- lightweight helpers ----------

_STOP = {
    "a","an","the","and","or","but","if","while","with","of","for","to","in","on",
    "at","by","from","as","is","are","was","were","be","been","being","that","this",
    "these","those","it","its","their","his","her","your","our","not","no","do","does",
    "did","can","could","may","might","should","would","will"
}

_TOKEN_RX = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)?")  # keeps hyphenated like "prokaryotic-like"

def _tok(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RX.findall(text or "")]

def _content_tokens(text: str) -> List[str]:
    return [t for t in _tok(text) if t not in _STOP and len(t) > 1]

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

# ---------- query profile ----------

class QueryProfile:
    def __init__(self, base_query: str, lemma_obj: dict):
        self.base_query = (base_query or "").strip()
        self.exact_rx = re.compile(r"\b" + re.escape(self.base_query) + r"\b", re.I) if self.base_query else None

        lemmas = []
        for entry in (lemma_obj or {}).values():
            for w in entry.get("lemmas", []) or []:
                w = (w or "").strip()
                if w:
                    lemmas.append(w)
        # canonical token set from base + lemmas
        toks = _content_tokens(" ".join([self.base_query] + lemmas))
        self.query_tokens = _unique(toks)

        # order-agnostic bigrams for mild reordering tolerance
        self.query_bigrams = _bigrams(self.query_tokens)

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

# ---------- candidate enumeration ----------

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

# ---------- main entry ----------

def smart_find_hits_in_soup(
    soup: BeautifulSoup,
    base_query: str,
    lemma_obj: dict,
    *,
    min_score: float = 0.60,
    top_k: Optional[int] = None
) -> List[Tuple[Tag, float, dict]]:
    """
    Returns a ranked list of (node, score, details). Filter by min_score and optional top_k.
    """
    profile = QueryProfile(base_query, lemma_obj)
    if not profile.any_tokens() and not profile.exact_rx:
        return []

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
        if s >= min_score:
            details["heading_text"] = htxt[:120]
            details["snippet"] = text[:160]
            scored.append((n, s, details))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k] if top_k else scored
