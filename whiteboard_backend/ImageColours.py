#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image
import cv2


# ============================
# PATHS (NO ARGS)
# ============================
BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "ProccessedImages"
VECTORS_DIR = BASE_DIR / "StrokeVectors"


# ============================
# 11-PASS COLOR ORDER
# ============================
COLOR_ORDER_LIGHT_TO_DARK = [
    "white",
    "yellow",
    "orange",
    "cyan",
    "green",
    "magenta",
    "red",
    "blue",
    "purple",
    "gray",
    "black",
]


def color_rank(name: str) -> int:
    n = (name or "").strip().lower()
    try:
        return COLOR_ORDER_LIGHT_TO_DARK.index(n)
    except ValueError:
        return 999


# ============================
# CONFIG (simple + blunt)
# ============================
# If False: skips the expensive per-stroke color sampling and does NOT rewrite vectors json.
ENABLE_STROKE_COLOR_SAMPLING = True


# ============================
# COLOR ASSIGNING
# ============================
SAMPLE_RADIUS_PX_SMALL = 1
SAMPLE_RADIUS_PX_LARGE = 2
MIN_CONTRIB_PIXELS_PER_STROKE = 60

# Color bit-shift quantization (RESTORED)
# (r>>4, g>>4, b>>4) => 16x16x16 bins (4096 total)
RGB_Q_SHIFT = 10


# ============================
# SPEED KNOBS (NO ARGS)
# ============================
PARALLEL = True
MAX_WORKERS = None  # None => cpu_count()-1

# Stroke sampling density (pure speed lever)
BEZIER_STEPS = 10
POLYLINE_TARGET_POINTS = 50


# ============================
# IO helpers
# ============================
_PROCESSED_PLAIN_RE = re.compile(r"^(?:proccessed|processed)_(\d+)\.png$", re.IGNORECASE)


def _glob_processed_images() -> List[Path]:
    imgs: List[Path] = []
    for p in PROCESSED_DIR.rglob("*.png"):
        if _PROCESSED_PLAIN_RE.match(p.name):
            imgs.append(p)
    imgs.sort(key=lambda x: str(x).lower())
    return imgs


def _extract_index_from_processed_name(name: str) -> Optional[int]:
    m = _PROCESSED_PLAIN_RE.match(name)
    if not m:
        return None
    return int(m.group(1))


def _vector_json_path_for_index(n: int) -> Path:
    # vectorizer writes processed_<idx>.json
    return VECTORS_DIR / f"processed_{n}.json"


def _load_rgb_image(path: Path) -> np.ndarray:
    # Prefer OpenCV decode for speed; fallback to PIL if needed.
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is not None:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


# ============================
# Stroke parsing (color_id normalization / naming)
# ============================
def _stroke_color_id_raw(stroke: Dict[str, Any]) -> Optional[int]:
    # be tolerant about key naming
    for k in ("color_id", "colour_id", "color_group_id", "pass_id", "group_id"):
        v = stroke.get(k)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
    return None


def _normalize_color_ids(ids: List[Optional[int]]) -> List[Optional[int]]:
    # Support both 0..10 and 1..11 without guessing per-stroke.
    vals = [v for v in ids if isinstance(v, int)]
    if not vals:
        return ids

    has_zero = any(v == 0 for v in vals)
    if has_zero:
        # assume 0-based
        out: List[Optional[int]] = []
        for v in ids:
            out.append((v + 1) if isinstance(v, int) else None)
        return out

    # assume already 1-based
    return ids


def _color_name_from_id(cid_1based: int) -> str:
    if 1 <= cid_1based <= len(COLOR_ORDER_LIGHT_TO_DARK):
        return COLOR_ORDER_LIGHT_TO_DARK[cid_1based - 1]
    return "unknown"


# ============================
# Color helpers
# ============================
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _estimate_background_rgb(img_rgb: np.ndarray) -> Tuple[int, int, int]:
    h, w, _ = img_rgb.shape
    patch = max(10, min(40, min(h, w) // 20))
    corners = np.concatenate(
        [
            img_rgb[0:patch, 0:patch].reshape(-1, 3),
            img_rgb[0:patch, w - patch : w].reshape(-1, 3),
            img_rgb[h - patch : h, 0:patch].reshape(-1, 3),
            img_rgb[h - patch : h, w - patch : w].reshape(-1, 3),
        ],
        axis=0,
    )
    med = np.median(corners, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _rgb_to_lab_u8(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)


def _lab_of_rgb(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    arr = np.array([[list(rgb)]], dtype=np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0]
    return (float(lab[0]), float(lab[1]), float(lab[2]))


def _delta_e_lab(lab_img_u8: np.ndarray, lab_bg: Tuple[float, float, float]) -> np.ndarray:
    dl = lab_img_u8[..., 0].astype(np.float32) - lab_bg[0]
    da = lab_img_u8[..., 1].astype(np.float32) - lab_bg[1]
    db = lab_img_u8[..., 2].astype(np.float32) - lab_bg[2]
    return np.sqrt(dl * dl + da * da + db * db)


def _otsu_threshold_deltae(deltae: np.ndarray) -> float:
    d = np.clip(deltae, 0.0, 60.0)
    u8 = (d * (255.0 / 60.0)).astype(np.uint8)
    thr_u8, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_de = thr_u8 * (60.0 / 255.0)
    thr_de = max(6.0, min(26.0, thr_de * 0.85))
    return float(thr_de)


def _quant_key(rgb: np.ndarray) -> np.ndarray:
    # Works for both (H,W,3) and (N,3)
    rgb = np.asarray(rgb, dtype=np.uint8)
    r = (rgb[..., 0] >> RGB_Q_SHIFT).astype(np.int32)
    g = (rgb[..., 1] >> RGB_Q_SHIFT).astype(np.int32)
    b = (rgb[..., 2] >> RGB_Q_SHIFT).astype(np.int32)
    return (r << 8) | (g << 4) | b  # 12-bit key, 0..4095


# ============================
# Stroke sampling
# ============================
def _eval_cubic(p0, c1, c2, p1, t: float) -> Tuple[float, float]:
    mt = 1.0 - t
    mt2 = mt * mt
    t2 = t * t
    x = (mt2 * mt) * p0[0] + 3 * (mt2 * t) * c1[0] + 3 * (mt * t2) * c2[0] + (t2 * t) * p1[0]
    y = (mt2 * mt) * p0[1] + 3 * (mt2 * t) * c1[1] + 3 * (mt * t2) * c2[1] + (t2 * t) * p1[1]
    return x, y


def _stroke_samples_json_from_beziers(stroke: Dict[str, Any]) -> List[Tuple[float, float]]:
    segs = stroke.get("segments") or []
    if not isinstance(segs, list) or not segs:
        return []

    pts: List[Tuple[float, float]] = []
    steps = int(BEZIER_STEPS)

    last = None
    for seg in segs:
        if not isinstance(seg, list) or len(seg) < 8:
            continue
        p0 = (float(seg[0]), float(seg[1]))
        c1 = (float(seg[2]), float(seg[3]))
        c2 = (float(seg[4]), float(seg[5]))
        p1 = (float(seg[6]), float(seg[7]))

        for i in range(steps + 1):
            t = i / steps
            x, y = _eval_cubic(p0, c1, c2, p1, t)
            if last is not None and (abs(x - last[0]) + abs(y - last[1]) < 0.05):
                continue
            pts.append((x, y))
            last = (x, y)

    return pts


def _stroke_samples_json_from_polyline(stroke: Dict[str, Any]) -> List[Tuple[float, float]]:
    pts = stroke.get("points") or []
    if not isinstance(pts, list) or len(pts) < 2:
        return []
    out: List[Tuple[float, float]] = []
    for p in pts:
        if isinstance(p, list) and len(p) >= 2:
            out.append((float(p[0]), float(p[1])))
    if len(out) < 2:
        return []
    target = int(POLYLINE_TARGET_POINTS)
    step = max(1, len(out) // max(1, target))
    return out[::step]


# ============================
# Color estimation per stroke (OPTIMIZED)
# ============================
def _estimate_stroke_color(
    img_rgb: np.ndarray,
    fg_mask: np.ndarray,
    stroke_samples_json: List[Tuple[float, float]],
    scale_x: float,
    scale_y: float,
    radius: int,
) -> Tuple[Tuple[int, int, int], float, int]:
    h, w, _ = img_rgb.shape
    if not stroke_samples_json:
        return (0, 0, 0), 0.0, 0

    pts = np.asarray(stroke_samples_json, dtype=np.float32)
    if pts.size == 0:
        return (0, 0, 0), 0.0, 0

    xs = np.rint(pts[:, 0] * float(scale_x)).astype(np.int32)
    ys = np.rint(pts[:, 1] * float(scale_y)).astype(np.int32)

    inside = (xs >= 0) & (ys >= 0) & (xs < w) & (ys < h)
    if not np.any(inside):
        return (0, 0, 0), 0.0, 0

    xs = xs[inside]
    ys = ys[inside]
    sampled = int(xs.size)

    # Build offset grid once per call
    r = int(radius)
    d = np.arange(-r, r + 1, dtype=np.int32)
    dx, dy = np.meshgrid(d, d, indexing="xy")
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    # Broadcast sample centers + offsets -> flattened gather coords
    gx = (xs[:, None] + dx[None, :]).reshape(-1)
    gy = (ys[:, None] + dy[None, :]).reshape(-1)

    inb = (gx >= 0) & (gy >= 0) & (gx < w) & (gy < h)
    if not np.any(inb):
        return (0, 0, 0), 0.0, 0

    gx = gx[inb]
    gy = gy[inb]

    m = fg_mask[gy, gx]
    if not np.any(m):
        return (0, 0, 0), 0.0, 0

    gx = gx[m]
    gy = gy[m]

    pix = img_rgb[gy, gx]  # (n,3)
    if pix.size == 0:
        return (0, 0, 0), 0.0, 0

    contrib = int(pix.shape[0])

    # 16x16x16 bins => 4096
    BIN_COUNT = 4096

    keys = _quant_key(pix)
    counts = np.bincount(keys, minlength=BIN_COUNT)

    if counts.max() == 0:
        return (0, 0, 0), 0.0, 0

    sum_r = np.bincount(keys, weights=pix[:, 0], minlength=BIN_COUNT).astype(np.int64)
    sum_g = np.bincount(keys, weights=pix[:, 1], minlength=BIN_COUNT).astype(np.int64)
    sum_b = np.bincount(keys, weights=pix[:, 2], minlength=BIN_COUNT).astype(np.int64)

    best_k = int(np.argmax(counts))
    c_best = int(counts[best_k])

    rgb = (
        int(sum_r[best_k] / max(1, c_best)),
        int(sum_g[best_k] / max(1, c_best)),
        int(sum_b[best_k] / max(1, c_best)),
    )

    denom = max(1, sampled * ((2 * r + 1) * (2 * r + 1)))
    conf = _clamp(contrib / denom, 0.0, 1.0)

    return rgb, float(conf), int(contrib)
    
def apply_colours_in_memory(
    preproc_by_idx: Dict[int, Dict[str, Any]],
    vectors_by_idx: Dict[int, Dict[str, Any]],
    *,
    save_outputs: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    In-memory version of process_one().

    Inputs:
      preproc_by_idx[idx]["cleaned_bgr"] : np.ndarray (H,W,3) BGR uint8
      vectors_by_idx[idx] : dict loaded/produced by vectorizer (same schema as processed_<idx>.json)

    Behavior:
      - Mutates vectors_by_idx[idx] IN PLACE (adds stroke colors + normalized ids + names)
      - Optionally writes VECTORS_DIR/processed_<idx>.json if save_outputs=True
      - Returns vectors_by_idx (same dict, mutated)
    """
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    for idx, prep in preproc_by_idx.items():
        data = vectors_by_idx.get(idx)
        if data is None:
            # nothing to colourize
            continue

        strokes = data.get("strokes") or []
        if not isinstance(strokes, list) or not strokes:
            continue

        img_bgr = prep.get("cleaned_bgr")
        if img_bgr is None:
            continue

        # Ensure RGB for sampling (matches original process_one)
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = img_rgb.shape

        fmt = (data.get("vector_format") or "bezier_cubic").lower()
        src_w = float(data.get("width") or w_img)
        src_h = float(data.get("height") or h_img)

        scale_x = (w_img / src_w) if src_w > 0 else 1.0
        scale_y = (h_img / src_h) if src_h > 0 else 1.0

        did_change = False

        # Precompute fg mask once per image (exactly like your process_one)
        if ENABLE_STROKE_COLOR_SAMPLING:
            bg_rgb = _estimate_background_rgb(img_rgb)
            img_lab = _rgb_to_lab_u8(img_rgb)
            bg_lab = _lab_of_rgb(bg_rgb)

            deltae = _delta_e_lab(img_lab, bg_lab)
            deltae_thr = _otsu_threshold_deltae(deltae)
            fg_mask = deltae >= float(deltae_thr)

            radius = SAMPLE_RADIUS_PX_LARGE if max(w_img, h_img) >= 1400 else SAMPLE_RADIUS_PX_SMALL

        color_ids_raw: List[Optional[int]] = []

        for s in strokes:
            # Samples along stroke geometry in JSON coords
            if fmt == "bezier_cubic":
                samples_json = _stroke_samples_json_from_beziers(s)
            else:
                samples_json = _stroke_samples_json_from_polyline(s)

            if ENABLE_STROKE_COLOR_SAMPLING:
                rgb, conf, contrib = _estimate_stroke_color(
                    img_rgb=img_rgb,
                    fg_mask=fg_mask,
                    stroke_samples_json=samples_json,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    radius=int(radius),
                )
                s["color_rgb"] = [int(rgb[0]), int(rgb[1]), int(rgb[2])]
                s["color_hex"] = "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
                s["color_conf"] = float(round(float(conf), 4))
                s["color_contrib_pixels"] = int(contrib)
                did_change = True

            cid = _stroke_color_id_raw(s)
            color_ids_raw.append(cid)

        # Normalize ids (0..10 => 1..11)
        color_ids = _normalize_color_ids(color_ids_raw)

        # Ensure normalized 'color_id' and a derived 'color_name' exist
        for s, cid in zip(strokes, color_ids):
            if isinstance(cid, int):
                if s.get("color_id") != int(cid):
                    s["color_id"] = int(cid)
                    did_change = True
                if not isinstance(s.get("color_name"), str) or not s.get("color_name").strip():
                    s["color_name"] = _color_name_from_id(int(cid))
                    did_change = True

        if ENABLE_STROKE_COLOR_SAMPLING:
            data["stroke_color_sampling"] = {
                "bg_rgb": list(bg_rgb),
                "deltae_threshold": float(round(float(deltae_thr), 3)),
                "radius_px": int(radius),
                "rgb_quant_shift": int(RGB_Q_SHIFT),
            }
            did_change = True

        # Persist back into dict and optionally to disk
        if did_change:
            data["strokes"] = strokes
            vectors_by_idx[idx] = data

            if save_outputs:
                out_path = VECTORS_DIR / f"processed_{idx}.json"
                out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

            print(f"[ok][mem] processed_{idx}: colours applied" + (f" -> saved" if save_outputs else ""))
        else:
            print(f"[ok][mem] processed_{idx}: no changes")

    return vectors_by_idx

    
    

