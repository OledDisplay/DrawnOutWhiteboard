# ddg_image_context.py

from __future__ import annotations

from typing import Optional, Any, Dict
from bs4 import BeautifulSoup, Tag
from urllib.parse import urlparse, unquote
import re

from smart_hits import (
    QueryProfile,
    _image_dom_context,
    _nearest_heading,
    _find_context_window,
    _score_block,
    _cosine_sim,
    _image_attr_text_and_tokens,
    _coverage,
    _text_of,
)


def _best_img_for_ddg_url(soup: BeautifulSoup, image_url: str) -> Optional[Tag]:
    """
    Try to locate the <img> on the page that corresponds to the DDG image_url.
    Best-effort: match by filename / path tail, with some token overlap.
    If nothing matches, fall back to the first <img> on the page.
    """
    try:
        target_path = urlparse(image_url).path or ""
    except Exception:
        target_path = image_url or ""
    target_path = unquote(target_path)
    target_base = (target_path.rsplit("/", 1)[-1] if target_path else "").lower()

    target_tokens = [
        t for t in re.split(r"[^A-Za-z0-9]+", target_base) if t
    ]

    best_img: Optional[Tag] = None
    best_score: float = 0.0

    for img in soup.find_all("img"):
        if not isinstance(img, Tag):
            continue

        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if not isinstance(src, str) or not src.strip():
            continue

        try:
            cand_path = urlparse(src).path or ""
        except Exception:
            cand_path = src
        cand_path = unquote(cand_path)
        cand_base = (cand_path.rsplit("/", 1)[-1] if cand_path else "").lower()
        if not cand_base:
            continue

        score = 0.0

        # Strong: exact basename match
        if target_base and cand_base == target_base:
            score = 3.0
        # Medium: one basename contains the other
        elif target_base and (target_base in cand_base or cand_base in target_base):
            score = 2.0
        # Weak: overlapping alphanumeric tokens in basename
        elif target_tokens:
            cand_tokens = [
                t for t in re.split(r"[^A-Za-z0-9]+", cand_base) if t
            ]
            if cand_tokens:
                overlap = len(set(target_tokens) & set(cand_tokens))
                if overlap:
                    score = 1.0 + 0.1 * overlap

        if score > best_score:
            best_score = score
            best_img = img

    # Fallback: just take the first image on the page if we have nothing
    if best_img is None:
        first = soup.find("img")
        if isinstance(first, Tag):
            best_img = first

    return best_img


def ddg_extract_image_context(
    soup: BeautifulSoup,
    page_url: str,
    image_url: str,
    base_query: str,
    lemma_obj: dict,
    *,
    encoder: Optional[Any] = None,
    query_embedding: Optional[Any] = None,
    return_embedding: bool = True,
) -> Dict[str, Any]:
    """
    For a single DDG image result (page_url + image_url),
    find the corresponding <img> on the page, gather local HTML context,
    score it (same scoring as smart_hits), and return a standardized
    context metadata dict.

    IMPORTANT:
    - No thresholding: we ALWAYS return a context dict if we can find an <img>.
    - Structure matches smart_hits' details: ctx_text, ctx_score,
      ctx_sem_score, ctx_embedding, snippet, heading_text, plus image attrs.
    """
    profile = QueryProfile(base_query, lemma_obj or {})

    img = _best_img_for_ddg_url(soup, image_url)
    if img is None:
        # No image found at all; return empty and let caller handle it.
        return {}

    # Local DOM context around the image
    raw_ctx = _image_dom_context(img)
    heading_tag = _nearest_heading(img)
    heading_text = _text_of(heading_tag)

    if raw_ctx:
        ctx_text = _find_context_window(raw_ctx, base_query, profile)
    else:
        ctx_text = ""

    # Score the context (same lexical scoring as smart_hits)
    score, score_details = _score_block(profile, ctx_text, heading_text)

    details: Dict[str, Any] = dict(score_details)
    details["source"] = "ddg_image_context"
    details["page_url"] = page_url
    details["image_url"] = image_url

    # Canonical context fields
    details["ctx_text"] = (ctx_text or "")[:240]
    details["snippet"] = (ctx_text or "")[:160]
    details["heading_text"] = (heading_text or "")[:120]
    details["ctx_score"] = score  # unified lexical context score

    # Image attribute metadata
    src = img.get("src") or img.get("data-src") or img.get("data-original")
    details["img_src"] = src
    details["img_alt"] = img.get("alt")
    details["img_title"] = img.get("title")

    attr_text, attr_tokens = _image_attr_text_and_tokens(img)
    details["img_attr_text"] = (attr_text or "")[:200]

    if profile.any_tokens() and attr_tokens:
        details["img_attr_cov"] = _coverage(profile.query_tokens, attr_tokens)
    else:
        details["img_attr_cov"] = 0.0

    if profile.exact_rx and attr_text:
        details["img_attr_exact"] = 1.0 if profile.exact_rx.search(attr_text) else 0.0
    else:
        details["img_attr_exact"] = 0.0

    # Optional semantic info
    if encoder is not None and query_embedding is not None and ctx_text:
        ctx_emb = encoder.encode(ctx_text)
        sem_score = _cosine_sim(query_embedding, ctx_emb)
        details["sem_score"] = sem_score
        details["ctx_sem_score"] = sem_score
        if return_embedding:
            details["ctx_embedding"] = ctx_emb
    else:
        details["sem_score"] = None
        details["ctx_sem_score"] = None

    return details
