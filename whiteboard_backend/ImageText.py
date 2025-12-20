#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image

import inspect


# PaddleOCR 3.x pipeline API
from paddleocr import PaddleOCR

# ---- Stop PaddleOCR from doing the modelscope/host connectivity check (often drags in torch) ----
# Your log literally tells you you to set this.
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")


# ============================
# PATHS (NO ARGS)
# ============================
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "ResearchImages" / "UniqueImages"
OUT_DIR = BASE_DIR / "ProccessedImages"

# ============================
# CONFIG
# ============================
OCR_LANG = "en"
OCR_VERSION = "PP-OCRv5"

# IMPORTANT: PaddleOCR 3.x uses `device`, not `use_gpu`
DEVICE = "gpu:0"   # change to "cpu" if needed

# Keep only usable text (JSON only; masking uses ALL detected boxes)
MIN_TEXT_LEN = 3
MIN_CONF = 0

# Fill settings
BBOX_PAD_PX = 2          # expand each text bbox a bit
INPAINT_RADIUS = 8        # cv2.inpaint radius

# EXTRA: after smart-fill, blend the region toward white so text is definitely gone
WHITE_BLEND_ALPHA = 0.92  # 0=no whitening, 1=pure white
INPAINT_CONTEXT_PAD = 18  # extra pixels around bbox so inpaint has context

# OCR fallback scaling (helps when detector misses “obvious” text at native scale)
# NOTE: for scattered diagram labels, an UPSCALE pass usually helps more than downscale.
OCR_SCALES = (1.0, 1.5)

# Add a border before OCR so text near edges doesn't get clipped by internal resize/pad
OCR_BORDER_PX = 30

# Tiled OCR (huge for scattered labels)
USE_TILED_OCR = True
TILE_SIZE = 1024
TILE_OVERLAP = 160

# Detector tuning (recall > precision for your masking use-case)
DET_LIMIT_SIDE_LEN = 2560
DET_LIMIT_TYPE = "max"  # common setting; keeps aspect, limits max side
TEXT_DET_THRESH = 0.10
TEXT_DET_BOX_THRESH = 0.15
TEXT_DET_UNCLIP_RATIO = 2.2

# JSON acceptance (strict)
KEEP_MIN_REC_SCORE = 0.85  # only these go into labels json

# Pixel-mask behavior (masking uses pixel-threshold inside OCR region)
PIX_MASK_MIN_CC_AREA = 8
PIX_MASK_MAX_ASPECT = 18.0        # drop very long thin CCs (arrow lines)
PIX_MASK_LINE_AREA_MAX = 1400     # long-thin + small area => probably line
PIX_MASK_MIN_FILL_RATIO = 0.002   # if region has almost no dark pixels, skip masking it
PIX_MASK_DILATE = 2               # expand letters a bit so edges are fully covered
PIX_MASK_LINE_MIN_LEN = 40        # NEW: don't kill thin letters; only treat "line" if it's actually long

# --- NEW: bbox refinement (pixel-tight word bbox, do NOT rely on PaddleOCR bbox) ---
REFINE_SEARCH_PAD_PX = 18         # how far beyond Paddle bbox we search for actual ink
REFINE_BBOX_PAD_PX = 2            # final pad around detected ink bbox
REFINE_MIN_INK_PIXELS = 12        # if fewer ink pixels, keep original bbox
REFINE_MAX_EXPAND_FACTOR = 3.5    # safety: don't let refined bbox blow up vs original bbox
REFINE_ADAPT_BLOCK = 31           # adaptive threshold blockSize (odd)
REFINE_ADAPT_C = 7                # adaptive threshold C


# Some builds expose these; if your PaddleOCR throws on them, comment them out.
DET_DB_SCORE_MODE = "slow"   # "fast" vs "slow" scoring
USE_DILATION = True          # helps connect characters in clean printed text

# output naming
OUT_PREFIX = "processed_"


# ============================
# NEW SYSTEM CONFIG (scout + word-guided tracing)
# ============================
TRACE_SEARCH_PAD_PX = 28          # base search pad around Paddle bbox (hint only)
TRACE_SEARCH_PAD_MULTS = (1, 2, 3)  # scout windows: 1x, 2x, 3x
TRACE_FINAL_PAD_PX = 2            # final pad around traced bbox
TRACE_MIN_INK_PIXELS = 18         # if fewer ink px, that attempt is treated as failed
TRACE_MAX_EXPAND_FACTOR = 3.5     # safety vs approx bbox area

ATLAS_GLYPH_SIZE = 32             # normalized glyph size for matching
ATLAS_MIN_WORD_LEN = 2
ATLAS_MIN_SCORE = 0.90            # only very confident words seed templates
ATLAS_MAX_SAMPLES_PER_CHAR = 48

TRACE_VALLEY_RATIO = 0.25         # projection valley threshold for character splitting
TRACE_MIN_GAP_FACTOR = 0.35       # min spacing between cuts relative to avg char width
TRACE_SIM_WEIGHT = 6.0            # similarity bonus influence
TRACE_SIM_MIN_COVERAGE = 0.55     # require enough chars with templates


def _predict_kwargs(ocr: PaddleOCR) -> Dict[str, Any]:
    """
    PaddleOCR v3 applies many tuning params on predict(). We only pass
    params that exist in this build to avoid crashes.
    """
    sig = inspect.signature(ocr.predict)
    kw: Dict[str, Any] = {}

    def put(name: str, val: Any) -> None:
        if name in sig.parameters:
            kw[name] = val

    # Detection recall knobs (these names vary across builds)
    put("det_limit_side_len", DET_LIMIT_SIDE_LEN)
    put("det_limit_type", DET_LIMIT_TYPE)
    put("text_det_limit_type", DET_LIMIT_TYPE)

    put("text_det_thresh", TEXT_DET_THRESH)
    put("text_det_box_thresh", TEXT_DET_BOX_THRESH)
    put("text_det_unclip_ratio", TEXT_DET_UNCLIP_RATIO)

    put("use_dilation", USE_DILATION)

    # Keep recognition threshold low here (don’t kill boxes); filter later for JSON
    put("text_rec_score_thresh", 0.0)

    return kw


# ============================
# IO helpers
# ============================
def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def _list_images_topdown(folder: Path) -> List[Path]:
    # stable order: name sort
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name.lower())
    return files

def _load_image_cv_unchanged(path: Path) -> np.ndarray:
    """
    Old approach (better webp/alpha handling):
    - Read with IMREAD_UNCHANGED
    - If alpha exists: only fully transparent pixels become white
    - Return BGR uint8
    """
    img0 = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img0 is None:
        raise RuntimeError(f"cannot read image: {path}")

    if img0.ndim == 3 and img0.shape[2] == 4:  # BGRA
        b, g, r, a = cv2.split(img0)
        rgb = cv2.merge([b, g, r])  # still BGR ordering in OpenCV var naming
        mask = (a == 0)
        if np.any(mask):
            rgb[mask] = [255, 255, 255]
        return rgb

    if img0.ndim == 2:
        return cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)

    if img0.ndim == 3 and img0.shape[2] == 3:
        return img0

    # fallback
    img1 = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img1 is None:
        raise RuntimeError(f"cannot read image (fallback): {path}")
    return img1

def _save_png(path: Path, img_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img_bgr)

def _prep_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Diagram-friendly OCR view:
    - grayscale normalize
    - CLAHE for contrast
    - mild unsharp
    Return 3-channel BGR (PaddleOCR expects 3-ch images typically).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gray = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# ============================
# PaddleOCR result parsing (handles dict + result-objects + numpy arrays)
# ============================
def _try_json_load(x: Any) -> Optional[Dict[str, Any]]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None

def _result_obj_to_dict(r: Any) -> Dict[str, Any]:
    """
    PaddleOCR 3.x returns Result objects; docs say it can print/save json.
    We try common ways to get a dict.
    """
    if isinstance(r, dict):
        return r

    # Some Result objects have `.res` (already the inner payload)
    if hasattr(r, "res"):
        rr = getattr(r, "res")
        if isinstance(rr, dict):
            return {"res": rr}

    # Some Result objects have `.json` (string or dict-like)
    if hasattr(r, "json"):
        j = getattr(r, "json")
        d = _try_json_load(j) if not isinstance(j, dict) else j
        if isinstance(d, dict):
            return d

    # Some have a method
    for meth in ("to_json", "to_dict", "dict"):
        if hasattr(r, meth):
            fn = getattr(r, meth)
            if callable(fn):
                try:
                    out = fn()
                except TypeError:
                    continue
                d = _try_json_load(out) if not isinstance(out, dict) else out
                if isinstance(d, dict):
                    return d

    # Last resort: __dict__ (not guaranteed useful)
    try:
        dd = dict(getattr(r, "__dict__", {}))
        if dd:
            return dd
    except Exception:
        pass

    return {}

def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    # scalar / string etc.
    return [x]

def _extract_ocr_items(res_any: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Returns:
      items_all: ALL detected boxes (even if rec text is empty/short) -> used for masking
      items_kept: filtered items (MIN_TEXT_LEN / MIN_CONF) -> used for your labels json
      raw_count: how many DET boxes existed BEFORE filtering (useful for debugging)

    IMPORTANT FIX:
      Do NOT use min(len(polys), len(texts), len(scores)).
      Detection can produce polys even when recognition returns fewer/empty fields.
      For masking you want ALL polys.
    """
    if res_any is None:
        return [], [], 0

    # predict() returns an iterable of per-image Result objects
    if isinstance(res_any, (list, tuple)):
        results = list(res_any)
    else:
        try:
            results = list(res_any)
            if not results:
                results = [res_any]
        except TypeError:
            results = [res_any]

    items_all: List[Dict[str, Any]] = []
    items_kept: List[Dict[str, Any]] = []
    raw_count = 0

    for r in results:
        d = _result_obj_to_dict(r)
        if not d:
            continue

        # PaddleOCR often nests fields under "res"
        payload = d.get("res", d)
        if not isinstance(payload, dict):
            continue

        # Prefer detection polys for coverage; rec polys can be absent.
        polys = payload.get("dt_polys") or payload.get("rec_polys") or payload.get("polys") or payload.get("boxes")
        texts = payload.get("rec_texts") or payload.get("rec_text") or payload.get("texts") or payload.get("text")
        scores = payload.get("rec_scores") or payload.get("rec_score") or payload.get("scores") or payload.get("score")

        polys_l = _as_list(polys)
        texts_l = _as_list(texts)
        scores_l = _as_list(scores)

        # We count by detection polys, not by min()
        n_polys = len(polys_l)
        raw_count += n_polys

        for i in range(n_polys):
            poly = polys_l[i]
            if isinstance(poly, np.ndarray):
                poly = poly.tolist()

            # poly should be 4 points [[x,y]...]
            pts: List[List[float]] = []
            if isinstance(poly, list) and len(poly) >= 4 and isinstance(poly[0], (list, tuple)):
                ok = True
                for p in poly[:4]:
                    if not (isinstance(p, (list, tuple)) and len(p) >= 2):
                        ok = False
                        break
                    pts.append([float(p[0]), float(p[1])])
                if not ok:
                    continue
            else:
                continue

            # Text/score may be missing -> default empty/0
            text = texts_l[i] if i < len(texts_l) else ""
            t = text.strip() if isinstance(text, str) else ""
            try:
                sc = float(scores_l[i]) if i < len(scores_l) else 0.0
            except Exception:
                sc = 0.0

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

            item = {
                "poly": pts,
                "text": t,
                "score": sc,
                "bbox_xyxy": [x0, y0, x1, y1],
            }
            items_all.append(item)

            if len(t) >= MIN_TEXT_LEN and sc >= KEEP_MIN_REC_SCORE:
                 items_kept.append(item)

    return items_all, items_kept, raw_count

def _sort_topdown(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda it: (float(it["bbox_xyxy"][1]), float(it["bbox_xyxy"][0])))

def _rescale_items(items: List[Dict[str, Any]], inv_scale: float) -> List[Dict[str, Any]]:
    if abs(inv_scale - 1.0) < 1e-9:
        return items
    out: List[Dict[str, Any]] = []
    for it in items:
        bb = it["bbox_xyxy"]
        out.append({
            "poly": [[p[0] * inv_scale, p[1] * inv_scale] for p in it["poly"]],
            "text": it["text"],
            "score": it["score"],
            "bbox_xyxy": [bb[0] * inv_scale, bb[1] * inv_scale, bb[2] * inv_scale, bb[3] * inv_scale],
        })
    return out

# ============================
# Box merging / de-dup
# ============================
def _iou_xyxy(a: List[float], b: List[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, (ax1 - ax0)) * max(0.0, (ay1 - ay0))
    b_area = max(0.0, (bx1 - bx0)) * max(0.0, (by1 - by0))
    den = a_area + b_area - inter
    return float(inter / den) if den > 0 else 0.0

def _dedup_items(items: List[Dict[str, Any]], iou_thr: float = 0.65) -> List[Dict[str, Any]]:
    if not items:
        return []
    # Keep higher score first; ties -> bigger box first
    def _key(it):
        bb = it["bbox_xyxy"]
        area = max(0.0, (bb[2]-bb[0])) * max(0.0, (bb[3]-bb[1]))
        return (float(it.get("score", 0.0)), area)
    items_sorted = sorted(items, key=_key, reverse=True)

    kept: List[Dict[str, Any]] = []
    for it in items_sorted:
        bb = it["bbox_xyxy"]
        ok = True
        for kt in kept:
            if _iou_xyxy(bb, kt["bbox_xyxy"]) >= iou_thr:
                ok = False
                break
        if ok:
            kept.append(it)
    return kept

def _dedup_items_prefer_text(items: List[Dict[str, Any]], iou_thr: float = 0.65) -> List[Dict[str, Any]]:
    if not items:
        return []
    def _key(it):
        bb = it["bbox_xyxy"]
        area = max(0.0, (bb[2]-bb[0])) * max(0.0, (bb[3]-bb[1]))
        has_text = 1 if (isinstance(it.get("text", ""), str) and len(it["text"]) > 0) else 0
        return (has_text, float(it.get("score", 0.0)), area)
    items_sorted = sorted(items, key=_key, reverse=True)

    kept: List[Dict[str, Any]] = []
    for it in items_sorted:
        bb = it["bbox_xyxy"]
        ok = True
        for kt in kept:
            if _iou_xyxy(bb, kt["bbox_xyxy"]) >= iou_thr:
                ok = False
                break
        if ok:
            kept.append(it)
    return kept


# ============================
# NEW: Word scout + trace bboxes (Paddle bbox is only a hint)
# ============================
def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))

def _rect_poly_from_bbox(bb: List[float]) -> List[List[float]]:
    x0, y0, x1, y1 = bb
    return [[float(x0), float(y0)], [float(x1), float(y0)], [float(x1), float(y1)], [float(x0), float(y1)]]

def _smooth1d(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return a
    k = int(k)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    ap = np.pad(a.astype(np.float32), (pad, pad), mode="edge")
    ker = np.ones((k,), dtype=np.float32) / float(k)
    return np.convolve(ap, ker, mode="valid")

def _robust_ink_mask(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Otsu threshold -> dark becomes 255 (binary inverse)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    block = int(REFINE_ADAPT_BLOCK)
    if block < 3: block = 3
    if block % 2 == 0: block += 1
    try:
        adapt = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block, int(REFINE_ADAPT_C)
        )
        ink = cv2.bitwise_or(otsu, adapt)
    except Exception:
        ink = otsu

    # denoise a bit
    ink = cv2.medianBlur(ink, 3)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k3, iterations=1)
    return ink

def _filter_textish_components(ink: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int,int,int,int,int,float,float]]]:
    """
    Returns:
      filtered mask (0/255)
      list of CC tuples: (x, y, w, h, area, cx, cy) in local ROI coords
    """
    num, lab, stats, cents = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if num <= 1:
        return np.zeros_like(ink), []

    out = np.zeros_like(ink)
    ccs: List[Tuple[int,int,int,int,int,float,float]] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < PIX_MASK_MIN_CC_AREA:
            continue
        ar = max(w / max(1, h), h / max(1, w))

        # IMPORTANT: don't delete thin letters.
        # Only treat it as a "line" if it's both very thin AND actually long.
        if ar > PIX_MASK_MAX_ASPECT and area < PIX_MASK_LINE_AREA_MAX and max(w, h) >= PIX_MASK_LINE_MIN_LEN:
            continue

        out[lab == i] = 255
        cx, cy = float(cents[i][0]), float(cents[i][1])
        ccs.append((int(x), int(y), int(w), int(h), int(area), cx, cy))
    return out, ccs

def _cluster_1d_sorted(xs: List[float], max_gap: float) -> int:
    if not xs:
        return 0
    xs = sorted(xs)
    clusters = 1
    prev = xs[0]
    for v in xs[1:]:
        if (v - prev) > max_gap:
            clusters += 1
        prev = v
    return clusters

def _segment_valleys(mask255: np.ndarray, expected_len: int) -> List[Tuple[int, int]]:
    """
    Split a word mask into expected_len x-ranges using vertical projection valleys.
    Returns list of (x0, x1) inclusive.
    """
    h, w = mask255.shape[:2]
    if expected_len <= 1 or w <= 2:
        return [(0, w - 1)]

    proj = (mask255 > 0).sum(axis=0).astype(np.float32)
    maxp = float(np.max(proj)) if proj.size else 0.0
    if maxp <= 0.0:
        return [(0, w - 1)]

    # smooth projection
    avg_char_w = max(2.0, float(w) / float(expected_len))
    k = int(max(7, min(31, round(avg_char_w))))
    proj_s = _smooth1d(proj, k=k)

    # find local minima below threshold
    thr = float(maxp) * float(TRACE_VALLEY_RATIO)
    cands: List[Tuple[float, int]] = []
    for i in range(1, w - 1):
        if proj_s[i] <= proj_s[i - 1] and proj_s[i] <= proj_s[i + 1] and proj_s[i] <= thr:
            cands.append((float(proj_s[i]), i))

    need = expected_len - 1
    if need <= 0:
        return [(0, w - 1)]

    # pick cuts greedily from deepest valleys, with spacing
    min_gap = int(max(2, round(avg_char_w * TRACE_MIN_GAP_FACTOR)))
    cands.sort(key=lambda t: t[0])  # smallest proj first
    cuts: List[int] = []
    for _, x in cands:
        if all(abs(x - c) >= min_gap for c in cuts):
            cuts.append(int(x))
            if len(cuts) >= need:
                break
    cuts.sort()

    # fallback: uniform cuts if not enough valleys
    if len(cuts) < need:
        cuts = []
        for k_i in range(1, expected_len):
            cuts.append(int(round((w * k_i) / expected_len)))
        cuts = sorted(list(dict.fromkeys([min(w - 2, max(1, c)) for c in cuts])))

    # build segments from cuts
    segs: List[Tuple[int, int]] = []
    x0 = 0
    for c in cuts[:need]:
        x1 = max(x0, min(w - 1, c))
        segs.append((x0, x1))
        x0 = min(w - 1, c + 1)
    segs.append((x0, w - 1))

    # ensure exactly expected_len segments
    if len(segs) > expected_len:
        segs = segs[:expected_len]
    while len(segs) < expected_len:
        segs.append((w - 1, w - 1))

    return segs

def _normalize_glyph(mask255: np.ndarray, size: int = ATLAS_GLYPH_SIZE) -> Optional[np.ndarray]:
    ys, xs = np.where(mask255 > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    crop = mask255[y0:y1 + 1, x0:x1 + 1]
    if crop.size == 0:
        return None

    # pad to square
    h, w = crop.shape[:2]
    side = max(h, w)
    pad_y = side - h
    pad_x = side - w
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left
    sq = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # resize & convert to float [0,1]
    rs = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
    g = (rs > 0).astype(np.float32)
    return g

def _glyphs_from_word_mask(mask255: np.ndarray, word: str) -> Optional[List[np.ndarray]]:
    if not word:
        return None
    segs = _segment_valleys(mask255, len(word))
    glyphs: List[np.ndarray] = []
    for (x0, x1) in segs:
        sub = mask255[:, x0:x1 + 1]
        ng = _normalize_glyph(sub)
        if ng is None:
            return None
        glyphs.append(ng)
    if len(glyphs) != len(word):
        return None
    return glyphs

def _corr01(a: np.ndarray, b: np.ndarray) -> float:
    # normalized dot for 0/1 glyphs
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na <= 1e-6 or nb <= 1e-6:
        return 0.0
    return float(np.dot(aa, bb) / (na * nb))

def _build_char_atlas_templates(img_bgr: np.ndarray, items: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Build per-image glyph templates from VERY confident words.
    This is NOT OCR. It’s just harvesting the image’s own font shapes.
    """
    h, w = img_bgr.shape[:2]
    samples: Dict[str, List[np.ndarray]] = {}

    for it in items:
        word = it.get("text", "")
        sc = float(it.get("score", 0.0))
        if not isinstance(word, str) or len(word) < ATLAS_MIN_WORD_LEN or sc < ATLAS_MIN_SCORE:
            continue

        bb = it.get("bbox_xyxy")
        if not isinstance(bb, list) or len(bb) != 4:
            continue

        ax0, ay0, ax1, ay1 = [float(v) for v in bb]
        sx0 = _clamp_int(math.floor(ax0) - TRACE_SEARCH_PAD_PX, 0, w - 1)
        sy0 = _clamp_int(math.floor(ay0) - TRACE_SEARCH_PAD_PX, 0, h - 1)
        sx1 = _clamp_int(math.ceil(ax1) + TRACE_SEARCH_PAD_PX, 0, w - 1)
        sy1 = _clamp_int(math.ceil(ay1) + TRACE_SEARCH_PAD_PX, 0, h - 1)
        if sx1 <= sx0 or sy1 <= sy0:
            continue

        roi = img_bgr[sy0:sy1 + 1, sx0:sx1 + 1]
        ink = _robust_ink_mask(roi)
        filt, _ = _filter_textish_components(ink)

        if int(np.count_nonzero(filt)) < TRACE_MIN_INK_PIXELS:
            continue

        # Tighten to ink to avoid pulling neighbor junk into glyph split
        ys, xs = np.where(filt > 0)
        if xs.size == 0 or ys.size == 0:
            continue
        tx0, ty0, tx1, ty1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        wordmask = filt[ty0:ty1 + 1, tx0:tx1 + 1]

        glyphs = _glyphs_from_word_mask(wordmask, word)
        if glyphs is None:
            continue

        for ch, g in zip(word, glyphs):
            lst = samples.setdefault(ch, [])
            if len(lst) < ATLAS_MAX_SAMPLES_PER_CHAR:
                lst.append(g)

    templates: Dict[str, np.ndarray] = {}
    for ch, lst in samples.items():
        if not lst:
            continue
        stack = np.stack(lst, axis=0).astype(np.float32)
        templates[ch] = np.mean(stack, axis=0)
    return templates

def _trace_word_bbox_once(
    img_bgr: np.ndarray,
    approx_bb: List[float],
    word: str,
    templates: Dict[str, np.ndarray],
    pad_px: int,
    min_ink_pixels: int,
) -> Tuple[Optional[List[float]], Optional[float]]:
    """
    One attempt: search pad_px around approx bbox, find best word blob.
    Returns (bbox_xyxy or None, quality_score or None). Lower score is better.
    """
    h, w = img_bgr.shape[:2]
    ax0, ay0, ax1, ay1 = [float(v) for v in approx_bb]

    ax0 = float(max(0, min(w - 1, ax0)))
    ay0 = float(max(0, min(h - 1, ay0)))
    ax1 = float(max(0, min(w - 1, ax1)))
    ay1 = float(max(0, min(h - 1, ay1)))
    if ax1 <= ax0 or ay1 <= ay0:
        return None, None
    if not isinstance(word, str) or len(word) == 0:
        return None, None

    approx_area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    cx = 0.5 * (ax0 + ax1)
    cy = 0.5 * (ay0 + ay1)

    sx0 = _clamp_int(math.floor(ax0) - pad_px, 0, w - 1)
    sy0 = _clamp_int(math.floor(ay0) - pad_px, 0, h - 1)
    sx1 = _clamp_int(math.ceil(ax1) + pad_px, 0, w - 1)
    sy1 = _clamp_int(math.ceil(ay1) + pad_px, 0, h - 1)
    if sx1 <= sx0 or sy1 <= sy0:
        return None, None

    roi = img_bgr[sy0:sy1 + 1, sx0:sx1 + 1]
    if roi.size == 0:
        return None, None

    ink = _robust_ink_mask(roi)
    filt, ccs = _filter_textish_components(ink)

    if int(np.count_nonzero(filt)) < int(min_ink_pixels) or not ccs:
        return None, None

    # estimate a "text height" from CCs
    hs = [cc[3] for cc in ccs]
    med_h = int(np.median(np.array(hs, dtype=np.int32))) if hs else max(8, int(round(ay1 - ay0)))
    med_h = max(6, med_h)

    # connect letters into word blobs (horizontal bias)
    kx = max(2, int(round(med_h * 0.65)))
    ky = max(1, int(round(med_h * 0.22)))
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * kx + 1, 2 * ky + 1))
    connected = cv2.morphologyEx(filt, cv2.MORPH_CLOSE, kern, iterations=1)

    num2, lab2, stats2, cents2 = cv2.connectedComponentsWithStats(connected, connectivity=8)
    if num2 <= 1:
        return None, None

    cx_l = float(cx - sx0)
    cy_l = float(cy - sy0)

    templ_cov = sum(1 for ch in word if ch in templates) / float(len(word))
    use_sim = templ_cov >= TRACE_SIM_MIN_COVERAGE

    cc_centers = [(cc[5], cc[6]) for cc in ccs]  # (cx, cy)

    best_i = 0
    best_score = None

    for i in range(1, num2):
        x, y, ww, hh, area = stats2[i]
        if area < int(min_ink_pixels):
            continue

        # distance to approx center (normalized by text height)
        ccx, ccy = float(cents2[i][0]), float(cents2[i][1])
        d2 = ((ccx - cx_l) ** 2 + (ccy - cy_l) ** 2) / float((med_h * med_h) + 1)

        # estimate character columns inside this blob
        xs_in = [px for (px, py) in cc_centers if (px >= x and px <= (x + ww) and py >= y and py <= (y + hh))]
        max_gap = max(2.0, float(med_h) * 0.60)
        cols = _cluster_1d_sorted(xs_in, max_gap=max_gap)
        if cols <= 0:
            continue

        len_pen = float(abs(cols - len(word))) * 2.2 if len(word) >= 2 else 0.0

        bb_area = float(ww * hh)
        area_pen = 0.0
        if bb_area > approx_area * (TRACE_MAX_EXPAND_FACTOR * 1.8):
            area_pen = (bb_area / approx_area) * 0.5

        sim_bonus = 0.0
        if use_sim and ww >= 4 and hh >= 4:
            sub = filt[y:y + hh, x:x + ww]
            glyphs = _glyphs_from_word_mask(sub, word)
            if glyphs is not None:
                sims = []
                for ch, g in zip(word, glyphs):
                    t = templates.get(ch)
                    if t is None:
                        continue
                    sims.append(_corr01(g, t))
                if sims:
                    sim_avg = float(np.mean(np.array(sims, dtype=np.float32)))
                    sim_bonus = sim_avg * TRACE_SIM_WEIGHT

        score = d2 + len_pen + area_pen - sim_bonus
        if best_score is None or score < best_score:
            best_score = score
            best_i = i

    if best_i == 0 or best_score is None:
        return None, None

    # Tight bbox from filtered ink restricted to chosen blob
    sel = (lab2 == best_i) & (filt > 0)
    ys, xs = np.where(sel)
    if xs.size == 0 or ys.size == 0:
        return None, None

    tx0, ty0, tx1, ty1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    fx0 = _clamp_int(sx0 + tx0 - TRACE_FINAL_PAD_PX, 0, w - 1)
    fy0 = _clamp_int(sy0 + ty0 - TRACE_FINAL_PAD_PX, 0, h - 1)
    fx1 = _clamp_int(sx0 + tx1 + TRACE_FINAL_PAD_PX, 0, w - 1)
    fy1 = _clamp_int(sy0 + ty1 + TRACE_FINAL_PAD_PX, 0, h - 1)

    if fx1 <= fx0 or fy1 <= fy0:
        return None, None

    new_area = float((fx1 - fx0) * (fy1 - fy0))
    if new_area > approx_area * TRACE_MAX_EXPAND_FACTOR:
        return None, None

    return [float(fx0), float(fy0), float(fx1), float(fy1)], float(best_score)


def _trace_word_bbox(img_bgr: np.ndarray, approx_bb: List[float], word: str, templates: Dict[str, np.ndarray]) -> List[float]:
    """
    Scout & search:
    - try multiple expanding windows around approx_bb
    - pick the best scoring blob
    - if everything fails, fall back to approx_bb (still gets masked)
    """
    h, w = img_bgr.shape[:2]
    ax0, ay0, ax1, ay1 = [float(v) for v in approx_bb]
    ax0 = float(max(0, min(w - 1, ax0)))
    ay0 = float(max(0, min(h - 1, ay0)))
    ax1 = float(max(0, min(w - 1, ax1)))
    ay1 = float(max(0, min(h - 1, ay1)))
    if ax1 <= ax0 or ay1 <= ay0:
        return [ax0, ay0, ax1, ay1]

    best_bb: Optional[List[float]] = None
    best_score: Optional[float] = None

    for m in TRACE_SEARCH_PAD_MULTS:
        pad = int(TRACE_SEARCH_PAD_PX * int(m))

        # strict attempt
        bb1, sc1 = _trace_word_bbox_once(
            img_bgr, [ax0, ay0, ax1, ay1], word, templates,
            pad_px=pad,
            min_ink_pixels=TRACE_MIN_INK_PIXELS
        )
        if bb1 is not None and sc1 is not None:
            # small bias to prefer smaller pads if scores tie
            sc_adj = float(sc1) + (0.015 * float(m))
            if best_score is None or sc_adj < best_score:
                best_score = sc_adj
                best_bb = bb1

        # relaxed attempt (helps faint text / thin strokes)
        bb2, sc2 = _trace_word_bbox_once(
            img_bgr, [ax0, ay0, ax1, ay1], word, templates,
            pad_px=pad,
            min_ink_pixels=max(6, TRACE_MIN_INK_PIXELS // 2)
        )
        if bb2 is not None and sc2 is not None:
            sc_adj = float(sc2) + (0.020 * float(m)) + 0.10
            if best_score is None or sc_adj < best_score:
                best_score = sc_adj
                best_bb = bb2

    if best_bb is not None:
        return best_bb

    # total failure => keep approx (still masks something instead of missing)
    return [ax0, ay0, ax1, ay1]


def _refine_items_using_letter_trace(img_bgr: np.ndarray, items: List[Dict[str, Any]], templates: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """
    For each OCR item, replace bbox/poly with traced bbox/poly around actual letters.
    Paddle bbox is only a search hint.
    """
    out: List[Dict[str, Any]] = []
    for it in items:
        bb = it.get("bbox_xyxy")
        word = it.get("text", "")
        if not isinstance(bb, list) or len(bb) != 4:
            out.append(it)
            continue

        if isinstance(word, str) and len(word) > 0:
            bb2 = _trace_word_bbox(img_bgr, bb, word, templates)
        else:
            bb2 = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]

        it2 = dict(it)
        it2["bbox_xyxy"] = bb2
        it2["poly"] = _rect_poly_from_bbox(bb2)
        out.append(it2)

    return _sort_topdown(out)


def _white_fill_each_bbox(img_bgr: np.ndarray, items: List[Dict[str, Any]]) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    for it in items:
        bb = it.get("bbox_xyxy")
        if not isinstance(bb, list) or len(bb) != 4:
            continue
        x0, y0, x1, y1 = bb

        x0i = _clamp_int(math.floor(x0) - BBOX_PAD_PX, 0, w - 1)
        y0i = _clamp_int(math.floor(y0) - BBOX_PAD_PX, 0, h - 1)
        x1i = _clamp_int(math.ceil(x1) + BBOX_PAD_PX, 0, w - 1)
        y1i = _clamp_int(math.ceil(y1) + BBOX_PAD_PX, 0, h - 1)

        if x1i <= x0i or y1i <= y0i:
            continue

        cv2.rectangle(img_bgr, (x0i, y0i), (x1i, y1i), (255, 255, 255), thickness=-1)
    return img_bgr


# ============================
# OCR runners
# ============================
def _run_predict_once(ocr: PaddleOCR, img_bgr: np.ndarray) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    kw = _predict_kwargs(ocr)
    res = ocr.predict(input=img_bgr, **kw)
    return _extract_ocr_items(res)


def _run_ocr_with_scales(
    ocr: PaddleOCR,
    img_bgr: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, float]:
    """
    Returns (items_all, items_kept, raw_count, used_scale)
    Picks the scale that produces the most DET boxes.
    """
    h0, w0 = img_bgr.shape[:2]
    best_all: List[Dict[str, Any]] = []
    best_kept: List[Dict[str, Any]] = []
    best_raw = 0
    best_s = 1.0

    for s in OCR_SCALES:
        if abs(s - 1.0) < 1e-9:
            img_in = img_bgr
            inv = 1.0
        else:
            nw = max(16, int(round(w0 * s)))
            nh = max(16, int(round(h0 * s)))
            interp = cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA
            img_in = cv2.resize(img_bgr, (nw, nh), interpolation=interp)
            inv = 1.0 / s

        items_all, items_kept, raw_count = _run_predict_once(ocr, img_in)

        if items_all and abs(inv - 1.0) > 1e-9:
            items_all = _rescale_items(items_all, inv)
            items_kept = _rescale_items(items_kept, inv)

        if len(items_all) > len(best_all):
            best_all, best_kept, best_raw, best_s = items_all, items_kept, raw_count, s

    return _sort_topdown(best_all), _sort_topdown(best_kept), int(best_raw), float(best_s)

def _run_ocr_tiled(
    ocr: PaddleOCR,
    img_bgr: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Run OCR on overlapping tiles; merge results.
    Returns (items_all, items_kept, raw_count_sum)
    """
    h, w = img_bgr.shape[:2]
    all_items: List[Dict[str, Any]] = []
    kept_items: List[Dict[str, Any]] = []
    raw_sum = 0

    stride = max(64, TILE_SIZE - TILE_OVERLAP)

    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            y1 = min(h, y0 + TILE_SIZE)
            x1 = min(w, x0 + TILE_SIZE)
            tile = img_bgr[y0:y1, x0:x1]
            if tile.size == 0:
                continue

            items_all, items_kept, raw_count = _run_predict_once(ocr, tile)
            raw_sum += int(raw_count)

            # offset coordinates back to full image
            for it in items_all:
                it2 = dict(it)
                it2["poly"] = [[p[0] + x0, p[1] + y0] for p in it["poly"]]
                bb = it["bbox_xyxy"]
                it2["bbox_xyxy"] = [bb[0] + x0, bb[1] + y0, bb[2] + x0, bb[3] + y0]
                all_items.append(it2)
            for it in items_kept:
                it2 = dict(it)
                it2["poly"] = [[p[0] + x0, p[1] + y0] for p in it["poly"]]
                bb = it["bbox_xyxy"]
                it2["bbox_xyxy"] = [bb[0] + x0, bb[1] + y0, bb[2] + x0, bb[3] + y0]
                kept_items.append(it2)

    return _sort_topdown(all_items), _sort_topdown(kept_items), int(raw_sum)

# ============================
# Main
# ============================
def main() -> None:

    _ensure_dirs()

    files = _list_images_topdown(SRC_DIR)
    if not files:
        print(f"[error] no images found in {SRC_DIR}")
        return

    # PaddleOCR 3.x: device controls GPU/CPU selection.
    # Tuning args are documented in PaddleOCR 3.x pipeline usage; these increase recall for scattered labels.
    ocr_kwargs = dict(
        lang=OCR_LANG,
        ocr_version=OCR_VERSION,
        device=DEVICE,

        # If you don't NEED rotated text, turning this off removes a whole failure mode.
        use_textline_orientation=False,

        det_limit_side_len=DET_LIMIT_SIDE_LEN,
        det_limit_type=DET_LIMIT_TYPE,

        text_det_thresh=TEXT_DET_THRESH,
        text_det_box_thresh=TEXT_DET_BOX_THRESH,
        text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,

        text_rec_score_thresh=0.0,
    )

    ocr = PaddleOCR(**ocr_kwargs)

    for n, src_path in enumerate(files):
        img_bgr = _load_image_cv_unchanged(src_path)
        h, w = img_bgr.shape[:2]

        # OCR view + border
        ocr_view = _prep_for_ocr(img_bgr)
        ocr_view = cv2.copyMakeBorder(
            ocr_view,
            OCR_BORDER_PX, OCR_BORDER_PX, OCR_BORDER_PX, OCR_BORDER_PX,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

        # Full-image OCR (multi-scale)
        items_all_full, items_kept_full, raw_full, used_scale = _run_ocr_with_scales(ocr, ocr_view)

        # Tiled OCR (optional)
        raw_tiled = 0
        items_all_tile: List[Dict[str, Any]] = []
        items_kept_tile: List[Dict[str, Any]] = []
        if USE_TILED_OCR:
            items_all_tile, items_kept_tile, raw_tiled = _run_ocr_tiled(ocr, ocr_view)

        # Merge + dedup
        items_all = _dedup_items(items_all_full + items_all_tile, iou_thr=0.65)
        items_kept = _dedup_items(items_kept_full + items_kept_tile, iou_thr=0.65)

        # Remove border offset and clamp into original image coords
        def _unborder(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for it in items:
                poly = [[p[0] - OCR_BORDER_PX, p[1] - OCR_BORDER_PX] for p in it["poly"]]
                bb = it["bbox_xyxy"]
                bb2 = [bb[0] - OCR_BORDER_PX, bb[1] - OCR_BORDER_PX, bb[2] - OCR_BORDER_PX, bb[3] - OCR_BORDER_PX]
                # clamp
                bb2[0] = float(max(0, min(w - 1, bb2[0])))
                bb2[1] = float(max(0, min(h - 1, bb2[1])))
                bb2[2] = float(max(0, min(w - 1, bb2[2])))
                bb2[3] = float(max(0, min(h - 1, bb2[3])))

                poly2 = []
                for x, y in poly:
                    poly2.append([float(max(0, min(w - 1, x))), float(max(0, min(h - 1, y)))])
                out.append({**it, "poly": poly2, "bbox_xyxy": bb2})
            return out

        items_all = _sort_topdown(_unborder(items_all))
        items_kept = _sort_topdown(_unborder(items_kept))

        # --- NEW SYSTEM: templates + scout tracing ---
        templates = _build_char_atlas_templates(img_bgr, items_kept)

        if items_all:
            items_all = _refine_items_using_letter_trace(img_bgr, items_all, templates)
        if items_kept:
            items_kept = _refine_items_using_letter_trace(img_bgr, items_kept, templates)

        # CRITICAL FIX:
        # Some labels end up in items_kept but not in items_all (dedup differences).
        # Mask must include BOTH, preferring entries that actually have text.
        items_mask_src = _dedup_items_prefer_text(items_all + items_kept, iou_thr=0.65)

        # Mask only recognized words/strings (prevents random garbage boxes)
        items_mask = [it for it in items_mask_src if isinstance(it.get("text", ""), str) and len(it["text"]) > 0]

        out_img = img_bgr.copy()
        if items_mask:
            out_img = _white_fill_each_bbox(out_img, items_mask)

        out_img_name = f"{OUT_PREFIX}{n}.png"
        out_json_name = f"{OUT_PREFIX}{n}_lables.json"

        _save_png(OUT_DIR / out_img_name, out_img)

        out_payload = {
            "source_file": src_path.name,
            "processed_file": out_img_name,
            "image_size": [int(w), int(h)],

            "raw_labels_full": int(raw_full),
            "raw_labels_tiled": int(raw_tiled),
            "boxes_masked": int(len(items_mask)),
            "kept_labels": int(len(items_kept)),
            "used_scale_full": float(used_scale),

            "det_limit_side_len": int(DET_LIMIT_SIDE_LEN),
            "text_det_thresh": float(TEXT_DET_THRESH),
            "text_det_box_thresh": float(TEXT_DET_BOX_THRESH),
            "text_det_unclip_ratio": float(TEXT_DET_UNCLIP_RATIO),
            "tiled": bool(USE_TILED_OCR),
            "tile_size": int(TILE_SIZE),
            "tile_overlap": int(TILE_OVERLAP),

            "labels": [
                {
                    "text": it["text"],
                    "score": float(round(it["score"], 6)),
                    "poly": it["poly"],
                    "bbox_xyxy": [int(round(v)) for v in it["bbox_xyxy"]],
                }
                for it in items_kept
            ]
        }
        (OUT_DIR / out_json_name).write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        print(
            f"[ok] {src_path.name} -> {out_img_name} | "
            f"raw_full={raw_full} raw_tiled={raw_tiled} masked={len(items_mask)} kept={len(items_kept)} "
            f"det_side={DET_LIMIT_SIDE_LEN} det_th={TEXT_DET_THRESH}/{TEXT_DET_BOX_THRESH} unclip={TEXT_DET_UNCLIP_RATIO}"
        )

if __name__ == "__main__":
    main()
