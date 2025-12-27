# process_images.py
from __future__ import annotations

import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"


import json
import math
import re
import warnings
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher

import cv2
import numpy as np
from PIL import Image


import nltk
from nltk.corpus import words as nltk_words
from nltk.corpus import wordnet as wn

# Ensure resources exist (run once)
# nltk.download("words")
# nltk.download("wordnet")

WORD_SET = set(w.lower() for w in nltk_words.words())


# -----------------------------
# USER SETTINGS (as provided)
# -----------------------------
OCR_LANG = "en"
OCR_VERSION = "PP-OCRv5"
DEVICE = "gpu:0"
DET_LIMIT_SIDE_LEN = 2560
DET_LIMIT_TYPE = "max"
TEXT_DET_THRESH = 0.10
TEXT_DET_BOX_THRESH = 0.15
TEXT_DET_UNCLIP_RATIO = 2.2
USE_DILATION = True
use_textline_orientation = False

OCR_BORDER_PAD = 30
OCR_SCALE = 1.5

# -----------------------------
# NEW LAYER-2 (TESSERACT) SETTINGS
# -----------------------------
USE_TESSERACT_LAYER2 = True

# If Tesseract is not on PATH (Windows), set this, e.g.:
# TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD: str | None = None

# How far around Paddle anchor we search (processed-image coords)
TESS_SEARCH_PAD_PX = 60
TESS_MIN_TILE_PAD_PX = 18

# Text matching tolerance (>=0.85 ~ "15% give")
TESS_MIN_SIM = 0.85

# Accept weaker text sim only if location match is very strong
TESS_MIN_SIM_WEAK = 0.70
TESS_MIN_LOC_STRONG = 0.78

# Tesseract page segmentation mode for small tiles
# 6 = Assume a block of text; 7 = single text line; 11 = sparse text
TESS_PSM = 11

# -----------------------------
# FALLBACK INK-TIGHT SETTINGS (if Tesseract fails)
# -----------------------------
REFINE_EXPAND_FACTOR = 1.25
REFINE_MIN_PAD_PX = 10
CC_MIN_AREA_FRAC = 0.0008
CC_MAX_AREA_FRAC = 0.40




def is_valid_dictionary_word(
    text: str,
    *,
    min_len: int = 2,
    allow_plural: bool = True,
    allow_capitalized: bool = True,
    allow_numbers: bool = False,
) -> bool:
    """
    HARD FILTER.
    Returns True if word is accepted, False if it should be dropped.
    """

    if not text:
        return False

    raw = text.strip()
    if len(raw) < min_len:
        return False

    # Reject pure punctuation
    if not any(c.isalnum() for c in raw):
        return False

    # Numbers
    if raw.isdigit():
        return allow_numbers

    # Normalize
    w = raw.lower()

    # Direct dictionary hit
    if w in WORD_SET:
        return True

    # Simple plural handling
    if allow_plural:
        if w.endswith("s") and w[:-1] in WORD_SET:
            return True
        if w.endswith("es") and w[:-2] in WORD_SET:
            return True
        if w.endswith("ies") and w[:-3] + "y" in WORD_SET:
            return True

    # WordNet fallback (very strict)
    if wn.synsets(w):
        return True

    return False


# -----------------------------
# MASKING
# -----------------------------
# User request: remove "very smart masking" => no inpaint, just white fill
WHITE_PAD_PX = 2  # expand bbox slightly before filling

SAVE_DEBUG = False

# -----------------------------
# LOADING LOGIC (DO NOT MODIFY)
# -----------------------------
def load_image_cv_unchanged(path: str | Path) -> np.ndarray:
    """
    Load an image robustly into a NumPy BGR uint8 array.

    Supported:
    - PNG (with or without alpha)
    - WEBP (with or without alpha)
    - JPG / JPEG
    - TIFF
    - BMP
    - Grayscale images

    Behavior:
    - Alpha channel is respected:
        * fully transparent pixels are replaced with white
    - Output is ALWAYS:
        np.ndarray of shape (H, W, 3), dtype=uint8, BGR order
    """

    path = str(path)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")

    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        bgr = cv2.merge([b, g, r])

        transparent = (a == 0)
        if np.any(transparent):
            bgr[transparent] = (255, 255, 255)

        return bgr.astype(np.uint8)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 3:
        return img.astype(np.uint8)

    img_fallback = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_fallback is None:
        raise RuntimeError(f"Cannot read image (fallback): {path}")

    return img_fallback.astype(np.uint8)


# -----------------------------
# DATA STRUCTURES
# -----------------------------
@dataclass
class WordDet:
    text: str
    quad: np.ndarray  # (4,2) float32 in processed-image coords
    score: float | None = None


# -----------------------------
# PATHS
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "ResearchImages" / "UniqueImages"
OUTPUT_DIR = SCRIPT_DIR / "ProccessedImages"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


# -----------------------------
# IMAGE PREP
# -----------------------------
def pad_and_scale_for_ocr(img_bgr: np.ndarray, pad: int, scale: float) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = img_bgr.shape[:2]
    padded = cv2.copyMakeBorder(
        img_bgr, pad, pad, pad, pad,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    new_w = int(round(padded.shape[1] * scale))
    new_h = int(round(padded.shape[0] * scale))
    ocr_img = cv2.resize(padded, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    meta = {
        "orig_w": w,
        "orig_h": h,
        "pad": pad,
        "scale": scale,
        "proc_w": new_w,
        "proc_h": new_h,
    }
    return ocr_img, meta


def proc_to_orig_bbox_xyxy(x1: float, y1: float, x2: float, y2: float, meta: dict[str, Any]) -> tuple[int, int, int, int]:
    pad = float(meta["pad"])
    scale = float(meta["scale"])
    ow = int(meta["orig_w"])
    oh = int(meta["orig_h"])

    x1 /= scale; y1 /= scale; x2 /= scale; y2 /= scale
    x1 -= pad; y1 -= pad; x2 -= pad; y2 -= pad

    x1i = max(0, min(ow, int(math.floor(x1))))
    y1i = max(0, min(oh, int(math.floor(y1))))
    x2i = max(0, min(ow, int(math.ceil(x2))))
    y2i = max(0, min(oh, int(math.ceil(y2))))
    if x2i < x1i:
        x1i, x2i = x2i, x1i
    if y2i < y1i:
        y1i, y2i = y2i, y1i
    return x1i, y1i, x2i, y2i


# -----------------------------
# PADDLE OCR INIT
# -----------------------------
def _safe_ctor_kwargs(PaddleOCR_cls, preferred: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(PaddleOCR_cls.__init__)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return preferred
        return {k: v for k, v in preferred.items() if k in sig.parameters}
    except Exception:
        return preferred


def create_paddle_ocr():
    """
    Minimal + stable ctor args only.
    Your build rejects unknown args (like use_dilation) inside parse_common_args.
    """
    from paddleocr import PaddleOCR  # type: ignore
    common = dict(
        lang=OCR_LANG,
        ocr_version=OCR_VERSION,
        device=DEVICE,
        use_textline_orientation=use_textline_orientation,
        text_rec_score_thresh=0.0,
    )

    # Try "text_det_*" style first (newer API)
    try:
        return PaddleOCR(
            **common,
            text_det_limit_side_len=DET_LIMIT_SIDE_LEN,
            text_det_limit_type=DET_LIMIT_TYPE,
            text_det_thresh=TEXT_DET_THRESH,
            text_det_box_thresh=TEXT_DET_BOX_THRESH,
            text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,
        )
    except Exception:
        # Fallback to older names
        return PaddleOCR(
            **common,
            det_limit_side_len=DET_LIMIT_SIDE_LEN,
            det_limit_type=DET_LIMIT_TYPE,
            text_det_thresh=TEXT_DET_THRESH,
            text_det_box_thresh=TEXT_DET_BOX_THRESH,
            text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,
        )



def _predict_kwargs(ocr) -> dict[str, Any]:
    kw: dict[str, Any] = {}
    try:
        sig = inspect.signature(ocr.predict)
    except Exception:
        sig = None

    def put(name: str, val: Any) -> None:
        if sig is None or name in sig.parameters:
            kw[name] = val

    put("use_doc_orientation_classify", False)
    put("use_doc_unwarping", False)
    put("use_textline_orientation", use_textline_orientation)
    put("return_word_box", True)
    put("text_rec_score_thresh", 0.0)
    put("use_dilation", USE_DILATION)

    if sig is not None and "text_det_limit_side_len" in sig.parameters:
        put("text_det_limit_side_len", DET_LIMIT_SIDE_LEN)
        put("text_det_limit_type", DET_LIMIT_TYPE)
        put("text_det_thresh", TEXT_DET_THRESH)
        put("text_det_box_thresh", TEXT_DET_BOX_THRESH)
        put("text_det_unclip_ratio", TEXT_DET_UNCLIP_RATIO)
    else:
        put("det_limit_side_len", DET_LIMIT_SIDE_LEN)
        put("det_limit_type", DET_LIMIT_TYPE)
        put("det_db_thresh", TEXT_DET_THRESH)
        put("det_db_box_thresh", TEXT_DET_BOX_THRESH)
        put("det_db_unclip_ratio", TEXT_DET_UNCLIP_RATIO)

    return kw


def ocr_predict(ocr, image_bgr: np.ndarray):
    return ocr.predict(input=image_bgr, **_predict_kwargs(ocr))


# -----------------------------
# PARSING
# -----------------------------
def _as_quad(points: Any) -> np.ndarray:
    arr = np.array(points, dtype=np.float32).reshape(-1, 2)
    if arr.shape[0] >= 4:
        return arr[:4]
    if arr.shape[0] == 2:
        x1, y1 = arr[0]
        x2, y2 = arr[1]
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    xs = arr[:, 0]
    ys = arr[:, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _split_words_keep_basic(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", (text or "").strip()) if t]


def parse_paddle_result_to_words(result: Any) -> list[WordDet]:
    words_out: list[WordDet] = []
    if result is None:
        return words_out

    if isinstance(result, list) and result and isinstance(result[0], dict):
        for page in result:
            payload = page.get("res", page)
            if not isinstance(payload, dict):
                continue
            rec_texts = payload.get("rec_texts") or payload.get("texts") or payload.get("rec_text") or []
            polys = payload.get("dt_polys") or payload.get("dt_boxes") or payload.get("det_polys") or payload.get("polys") or []

            n = min(len(rec_texts), len(polys))
            for i in range(n):
                line_text = str(rec_texts[i] or "")
                quad = _as_quad(polys[i])
                toks = _split_words_keep_basic(line_text)
                if len(toks) <= 1:
                    if line_text.strip():
                        words_out.append(WordDet(text=line_text.strip(), quad=quad))
                else:
                    # keep it simple: add each word anchored to same quad; layer2 will localize with tesseract anyway
                    for t in toks:
                        words_out.append(WordDet(text=t.strip(), quad=quad))
        return words_out

    # v2-ish fallback (rare in your setup)
    if isinstance(result, list):
        for item in result:
            try:
                quad = _as_quad(item[0])
                text = item[1][0] if isinstance(item[1], (list, tuple)) else ""
                if isinstance(text, str) and text.strip():
                    toks = _split_words_keep_basic(text)
                    for t in toks:
                        words_out.append(WordDet(text=t.strip(), quad=quad))
            except Exception:
                continue
    return words_out


# -----------------------------
# GEOMETRY UTILS
# -----------------------------
def _bbox_from_quad_xyxy(quad: np.ndarray) -> tuple[float, float, float, float]:
    x1 = float(np.min(quad[:, 0]))
    y1 = float(np.min(quad[:, 1]))
    x2 = float(np.max(quad[:, 0]))
    y2 = float(np.max(quad[:, 1]))
    return x1, y1, x2, y2


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter))


def _dist_score(cx: float, cy: float, tx: float, ty: float, diag: float) -> float:
    d = math.hypot(cx - tx, cy - ty)
    if diag <= 1e-6:
        return 0.0
    return max(0.0, 1.0 - min(1.0, d / diag))


def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    # remove spaces and punctuation for matching
    return re.sub(r"[^a-z0-9]+", "", s)


def _sim(a: str, b: str) -> float:
    a2 = _norm_text(a)
    b2 = _norm_text(b)
    if not a2 or not b2:
        return 0.0
    return float(SequenceMatcher(None, a2, b2).ratio())


# -----------------------------
# TESSERACT LAYER-2
# -----------------------------
def _tess_import():
    import pytesseract  # type: ignore
    from pytesseract import Output  # type: ignore
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    return pytesseract, Output


def _prep_for_tesseract(tile_bgr: np.ndarray) -> np.ndarray:
    """
    Make a tile that Tesseract is less likely to screw up on.
    """
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Otsu binarization, choose polarity by background
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If background ended up mostly black, invert
    if float(np.mean(bw)) < 127.0:
        bw = 255 - bw

    # Mild cleanup
    bw = cv2.medianBlur(bw, 3)

    # Tesseract expects RGB if you pass a 3ch image, but it works fine on 1ch too.
    return bw


def _tesseract_words(tile_bgr: np.ndarray) -> list[dict[str, Any]]:
    pytesseract, Output = _tess_import()

    img = _prep_for_tesseract(tile_bgr)

    config = f"--oem 1 --psm {int(TESS_PSM)}"
    data = pytesseract.image_to_data(img, output_type=Output.DICT, config=config)

    out: list[dict[str, Any]] = []
    n = len(data.get("text", []))
    for i in range(n):
        t = str(data["text"][i] or "").strip()
        if not t:
            continue
        try:
            conf = float(data.get("conf", ["-1"])[i])
        except Exception:
            conf = -1.0

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        if w <= 0 or h <= 0:
            continue

        out.append({
            "text": t,
            "conf": conf,
            "bbox": (float(x), float(y), float(x + w), float(y + h)),
            "line_id": (
                int(data.get("block_num", [0])[i]),
                int(data.get("par_num", [0])[i]),
                int(data.get("line_num", [0])[i]),
            ),
            "word_num": int(data.get("word_num", [0])[i]),
        })

    return out


def _merge_adjacent_words(words: list[dict[str, Any]], max_join: int = 3) -> list[dict[str, Any]]:
    """
    Tesseract splits stuff a lot. Build merged candidates (2-3 consecutive words on the same line).
    """
    by_line: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for w in words:
        by_line.setdefault(w["line_id"], []).append(w)

    merged: list[dict[str, Any]] = []
    for _, ws in by_line.items():
        ws.sort(key=lambda x: x["bbox"][0])  # left-to-right
        for i in range(len(ws)):
            x1, y1, x2, y2 = ws[i]["bbox"]
            t = ws[i]["text"]
            conf = ws[i]["conf"]
            merged.append({"text": t, "conf": conf, "bbox": (x1, y1, x2, y2)})

            for k in range(2, max_join + 1):
                if i + k - 1 >= len(ws):
                    break
                t2 = " ".join(w["text"] for w in ws[i:i+k])
                x1m = min(w["bbox"][0] for w in ws[i:i+k])
                y1m = min(w["bbox"][1] for w in ws[i:i+k])
                x2m = max(w["bbox"][2] for w in ws[i:i+k])
                y2m = max(w["bbox"][3] for w in ws[i:i+k])
                confm = float(np.mean([w["conf"] for w in ws[i:i+k]]))
                merged.append({"text": t2, "conf": confm, "bbox": (x1m, y1m, x2m, y2m)})

                # also a no-space variant because "CELL-1" may split
                t3 = "".join(_norm_text(w["text"]) for w in ws[i:i+k])
                merged.append({"text": t3, "conf": confm, "bbox": (x1m, y1m, x2m, y2m)})

    return merged


def tesseract_refine_bbox(
    ocr_img_bgr: np.ndarray,
    expected_word: str,
    anchor_bbox_proc: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    """
    Crop a search tile around anchor, run Tesseract, pick the word box that best matches:
      - expected text (with tolerance)
      - location (IoU + center distance)
    Returns bbox in processed-image coords, or None if no good match.
    """
    h, w = ocr_img_bgr.shape[:2]
    ax1, ay1, ax2, ay2 = anchor_bbox_proc
    acx = 0.5 * (ax1 + ax2)
    acy = 0.5 * (ay1 + ay2)

    aw = max(1.0, ax2 - ax1)
    ah = max(1.0, ay2 - ay1)
    pad = max(float(TESS_MIN_TILE_PAD_PX), float(TESS_SEARCH_PAD_PX), 0.6 * max(aw, ah))

    tx1 = int(max(0, math.floor(ax1 - pad)))
    ty1 = int(max(0, math.floor(ay1 - pad)))
    tx2 = int(min(w, math.ceil(ax2 + pad)))
    ty2 = int(min(h, math.ceil(ay2 + pad)))

    if tx2 <= tx1 or ty2 <= ty1:
        return None

    tile = ocr_img_bgr[ty1:ty2, tx1:tx2]
    if tile.size == 0:
        return None

    # anchor bbox in tile coords
    anchor_tile = (ax1 - tx1, ay1 - ty1, ax2 - tx1, ay2 - ty1)
    diag = math.hypot(float(tx2 - tx1), float(ty2 - ty1))

    try:
        words = _tesseract_words(tile)
    except Exception:
        return None

    candidates = _merge_adjacent_words(words, max_join=3)
    if not candidates:
        return None

    best = None
    best_score = -1e9

    for c in candidates:
        ctext = str(c["text"] or "")
        bb = c["bbox"]
        sim = _sim(expected_word, ctext)

        # location score: IoU + distance
        iou = _iou(anchor_tile, bb)
        ccx = 0.5 * (bb[0] + bb[2])
        ccy = 0.5 * (bb[1] + bb[3])
        dist = _dist_score(ccx, ccy, acx - tx1, acy - ty1, diag)

        loc = 0.65 * iou + 0.35 * dist

        # accept rule: strong text OR text ok + very strong location
        if sim < TESS_MIN_SIM:
            if not (sim >= TESS_MIN_SIM_WEAK and loc >= TESS_MIN_LOC_STRONG):
                continue

        score = 0.72 * sim + 0.28 * loc

        # reject absurdly huge boxes vs anchor unless sim is basically perfect
        area_c = max(1.0, (bb[2] - bb[0]) * (bb[3] - bb[1]))
        area_a = max(1.0, (anchor_tile[2] - anchor_tile[0]) * (anchor_tile[3] - anchor_tile[1]))
        if area_c > area_a * 8.0 and sim < 0.95:
            continue

        if score > best_score:
            best_score = score
            best = bb

    if best is None:
        return None

    bx1, by1, bx2, by2 = best
    # back to processed coords
    return (float(tx1) + float(bx1), float(ty1) + float(by1), float(tx1) + float(bx2), float(ty1) + float(by2))


# -----------------------------
# FALLBACK INK-TIGHT (NO OCR, JUST PIXELS)
# -----------------------------
def _expand_bbox_xyxy(bb: tuple[float, float, float, float], factor: float, min_pad: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bb
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(1.0, (x2 - x1) * factor)
    h = max(1.0, (y2 - y1) * factor)
    w = max(w, 2.0 * min_pad)
    h = max(h, 2.0 * min_pad)
    return (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)


def ink_tight_bbox(ocr_img_bgr: np.ndarray, anchor_bbox_proc: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    h, w = ocr_img_bgr.shape[:2]
    x1, y1, x2, y2 = _expand_bbox_xyxy(anchor_bbox_proc, REFINE_EXPAND_FACTOR, float(REFINE_MIN_PAD_PX))
    x1i = int(max(0, math.floor(x1))); y1i = int(max(0, math.floor(y1)))
    x2i = int(min(w, math.ceil(x2)));  y2i = int(min(h, math.ceil(y2)))
    if x2i <= x1i or y2i <= y1i:
        return anchor_bbox_proc

    roi = ocr_img_bgr[y1i:y2i, x1i:x2i]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num, lab, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), 8)
    if num <= 1:
        return anchor_bbox_proc

    H, W = bw.shape[:2]
    patch_area = float(H * W)
    min_area = max(5.0, CC_MIN_AREA_FRAC * patch_area)
    max_area = max(20.0, CC_MAX_AREA_FRAC * patch_area)

    xs = []
    ys = []
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < min_area or area > max_area:
            continue
        xs.extend([x, x + ww])
        ys.extend([y, y + hh])

    if not xs or not ys:
        return anchor_bbox_proc

    rx1 = float(min(xs)); ry1 = float(min(ys))
    rx2 = float(max(xs)); ry2 = float(max(ys))
    return (float(x1i) + rx1, float(y1i) + ry1, float(x1i) + rx2, float(y1i) + ry2)


# -----------------------------
# MASKING (WHITE)
# -----------------------------
def mask_with_white_fill(img_bgr: np.ndarray, boxes_xyxy: list[tuple[int, int, int, int]]) -> np.ndarray:
    if not boxes_xyxy:
        return img_bgr
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in boxes_xyxy:
        x1 = max(0, x1 - WHITE_PAD_PX)
        y1 = max(0, y1 - WHITE_PAD_PX)
        x2 = min(w, x2 + WHITE_PAD_PX)
        y2 = min(h, y2 + WHITE_PAD_PX)
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
    return out


# -----------------------------
# NMS
# -----------------------------
def rect_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter) / float(area_a + area_b - inter)


def nms_rectangles(
    boxes: list[tuple[int, int, int, int]],
    words: list[dict[str, Any]],
    iou_thresh: float = 0.8,
) -> tuple[list[tuple[int, int, int, int]], list[dict[str, Any]]]:
    if not boxes:
        return boxes, words
    keep_idx: list[int] = []
    used = [False] * len(boxes)

    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1]), reverse=True)
    for i in order:
        if used[i]:
            continue
        keep_idx.append(i)
        for j in order:
            if i == j or used[j]:
                continue
            if rect_iou(boxes[i], boxes[j]) >= iou_thresh:
                used[j] = True

    kept_boxes = [boxes[i] for i in keep_idx]
    kept_words = [words[i] for i in keep_idx]
    return kept_boxes, kept_words


# -----------------------------
# MAIN PROCESS
# -----------------------------
def process_all_images():
    ocr = create_paddle_ocr()

    if USE_TESSERACT_LAYER2:
        # Force import early so you fail fast if it's missing
        _tess_import()

    paths = [p for p in INPUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
    paths.sort()
    if not paths:
        raise RuntimeError(f"No images found in: {INPUT_DIR}")

    index = 0
    for p in paths:
        try:
            img_bgr = load_image_cv_unchanged(p)
            ocr_img_bgr, meta = pad_and_scale_for_ocr(img_bgr, OCR_BORDER_PAD, OCR_SCALE)

            raw = ocr_predict(ocr, ocr_img_bgr)
            word_dets = parse_paddle_result_to_words(raw)

            refined_words: list[dict[str, Any]] = []
            refined_boxes_orig: list[tuple[int, int, int, int]] = []

            for wd in word_dets:
                t = (wd.text or "").strip()
                if not t:
                    continue

                # anchor bbox (processed coords)
                ax1p, ay1p, ax2p, ay2p = _bbox_from_quad_xyxy(wd.quad)
                anchor_proc = (ax1p, ay1p, ax2p, ay2p)

                # Try Tesseract layer-2 bbox
                mask_proc: tuple[float, float, float, float] | None = None
                if USE_TESSERACT_LAYER2:
                    mask_proc = tesseract_refine_bbox(ocr_img_bgr, t, anchor_proc)

                # Fallback to ink-tight pixels if Tesseract doesn't match
                if mask_proc is None:
                    mask_proc = ink_tight_bbox(ocr_img_bgr, anchor_proc)

                mx1p, my1p, mx2p, my2p = mask_proc

                # convert both to original coords
                ax1o, ay1o, ax2o, ay2o = proc_to_orig_bbox_xyxy(ax1p, ay1p, ax2p, ay2p, meta)
                mx1o, my1o, mx2o, my2o = proc_to_orig_bbox_xyxy(mx1p, my1p, mx2p, my2p, meta)

                if mx2o <= mx1o or my2o <= my1o:
                    continue

                refined_boxes_orig.append((mx1o, my1o, mx2o, my2o))
                refined_words.append({
                    "text": t,
                    "bbox_anchor": [ax1o, ay1o, ax2o, ay2o],
                    "bbox_mask": [mx1o, my1o, mx2o, my2o],
                })

            refined_boxes_orig, refined_words = nms_rectangles(refined_boxes_orig, refined_words, iou_thresh=0.80)

            masked_bgr = mask_with_white_fill(img_bgr, refined_boxes_orig)

            out_img = OUTPUT_DIR / f"processed_{index}.png"
            out_json = OUTPUT_DIR / f"processed_{index}.json"
            cv2.imwrite(str(out_img), masked_bgr)

            payload = {
                "source_path": str(p),
                "image_size": [int(img_bgr.shape[1]), int(img_bgr.shape[0])],
                "ocr_preprocess": {"border_pad": OCR_BORDER_PAD, "scale": OCR_SCALE},
                "layer2": {
                    "mode": "tesseract_local" if USE_TESSERACT_LAYER2 else "ink_tight_only",
                    "tess_search_pad_px": int(TESS_SEARCH_PAD_PX),
                    "tess_min_sim": float(TESS_MIN_SIM),
                    "tess_psm": int(TESS_PSM),
                },
                "words": refined_words,
            }
            out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

            if SAVE_DEBUG:
                dbg = img_bgr.copy()
                for (x1, y1, x2, y2) in refined_boxes_orig:
                    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.imwrite(str(OUTPUT_DIR / f"debug_boxes_{index}.png"), dbg)

            print(f"[OK] {p.name} -> processed_{index}.png | words={len(refined_words)} masks={len(refined_boxes_orig)}")
            index += 1

        except Exception as e:
            print(f"[SKIP] {p} -> {e}")


if __name__ == "__main__":
    process_all_images()
