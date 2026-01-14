#!/usr/bin/env python3
from __future__ import annotations

import json
import os, shutil
import math
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
CLUSTER_RENDER_DIR = BASE_DIR / "ClusterRenders"
CLUSTER_MAP_DIR = BASE_DIR / "ClusterMaps"


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
# “blanket threshold” in image pixels (bbox-to-bbox distance)
GROUP_RADIUS_PX = 30.0

# spatial hash cell size; should be >= radius so neighbors are local
GRID_CELL_PX = 48.0

# crop padding (image pixels)
CLUSTER_CROP_PAD = 8

# stroke keep-color padding around skeleton line (pixels)
STROKE_SKELETON_PAD_PX = 3

# --- SPEED KNOBS ---
# If False: skips the expensive per-stroke color sampling and does NOT rewrite vectors json.
ENABLE_STROKE_COLOR_SAMPLING = False

# Faster PNG writes (lower compression = much faster I/O)
PNG_COMPRESS_LEVEL = 1


# ============================
# COLOR ASSIGNING
# ============================
SAMPLE_RADIUS_PX_SMALL = 1
SAMPLE_RADIUS_PX_LARGE = 2
MIN_CONTRIB_PIXELS_PER_STROKE = 60

# Color bit-shift quantization (RESTORED)
# (r>>4,g>>4,b>>4) => 16x16x16 bins
RGB_Q_SHIFT = 10


# ============================
# Union-Find
# ============================
class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


# ============================
# IO helpers
# ============================
def _ensure_dirs() -> None:
    if os.path.isdir(CLUSTER_RENDER_DIR):
        shutil.rmtree(CLUSTER_RENDER_DIR)
    if os.path.isdir(CLUSTER_MAP_DIR):
        shutil.rmtree(CLUSTER_MAP_DIR)
    CLUSTER_RENDER_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTER_MAP_DIR.mkdir(parents=True, exist_ok=True)

# ONLY accept processed_<n>.png (ignore processed_<n>_<colour>.png)
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
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def _save_png_rgb(arr_rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(arr_rgb).save(path, compress_level=int(PNG_COMPRESS_LEVEL))

# ============================
# Full-image render helper
# ============================
RECT_THICKNESS_PX = 2  # 1 or 2 as requested

def _render_full_with_red_bbox(img_rgb: np.ndarray, bbox_xyxy: List[int], thickness: int = RECT_THICKNESS_PX) -> np.ndarray:
    """
    Returns FULL image (same size as img_rgb) with a red rectangle drawn around bbox_xyxy.
    bbox_xyxy is treated as [x0, y0, x1, y1] where x1/y1 are exclusive (crop-style).
    """
    out = img_rgb.copy()
    h, w, _ = out.shape

    if not (isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4):
        return out

    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]

    # Clamp
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))

    # Convert exclusive end to inclusive pixel corner for drawing
    x2 = max(0, min(w - 1, x1 - 1))
    y2 = max(0, min(h - 1, y1 - 1))

    if x2 <= x0 or y2 <= y0:
        return out

    # img_rgb is RGB; cv2.rectangle will write values directly into channels.
    # (255,0,0) => red in RGB arrays.
    cv2.rectangle(out, (x0, y0), (x2, y2), (255, 0, 0), thickness=int(thickness), lineType=cv2.LINE_8)
    return out



# ============================
# Geometry
# ============================
def _bbox_distance(a: Tuple[float, float, float, float],
                   b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = 0.0
    if ax1 < bx0:
        dx = bx0 - ax1
    elif bx1 < ax0:
        dx = ax0 - bx1
    dy = 0.0
    if ay1 < by0:
        dy = by0 - ay1
    elif by1 < ay0:
        dy = ay0 - by1
    return math.hypot(dx, dy)

def _cluster_bbox(bboxes: List[Tuple[float,float,float,float]], idxs: List[int]) -> Tuple[float,float,float,float]:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for i in idxs:
        x0,y0,x1,y1 = bboxes[i]
        xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


# ============================
# Stroke parsing
# ============================
def _stroke_bbox_from_beziers(stroke: Dict[str, Any]) -> Optional[Tuple[float,float,float,float]]:
    segs = stroke.get("segments") or []
    if not isinstance(segs, list) or not segs:
        return None

    xs: List[float] = []
    ys: List[float] = []
    for seg in segs:
        if not isinstance(seg, list) or len(seg) < 8:
            continue
        # p0,c1,c2,p1 control points
        xs.extend([float(seg[0]), float(seg[2]), float(seg[4]), float(seg[6])])
        ys.extend([float(seg[1]), float(seg[3]), float(seg[5]), float(seg[7])])

    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))

def _stroke_bbox_from_polyline(stroke: Dict[str, Any]) -> Optional[Tuple[float,float,float,float]]:
    pts = stroke.get("points") or []
    if not isinstance(pts, list) or len(pts) < 2:
        return None
    xs = []
    ys = []
    for p in pts:
        if isinstance(p, list) and len(p) >= 2:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
    if len(xs) < 2:
        return None
    return (min(xs), min(ys), max(xs), max(ys))

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
# Color helpers (RESTORED)
# ============================
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _estimate_background_rgb(img_rgb: np.ndarray) -> Tuple[int, int, int]:
    h, w, _ = img_rgb.shape
    patch = max(10, min(40, min(h, w) // 20))
    corners = np.concatenate([
        img_rgb[0:patch, 0:patch].reshape(-1, 3),
        img_rgb[0:patch, w-patch:w].reshape(-1, 3),
        img_rgb[h-patch:h, 0:patch].reshape(-1, 3),
        img_rgb[h-patch:h, w-patch:w].reshape(-1, 3),
    ], axis=0)
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
    return np.sqrt(dl*dl + da*da + db*db)

def _otsu_threshold_deltae(deltae: np.ndarray) -> float:
    d = np.clip(deltae, 0.0, 60.0)
    u8 = (d * (255.0 / 60.0)).astype(np.uint8)
    thr_u8, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_de = thr_u8 * (60.0 / 255.0)
    thr_de = max(6.0, min(26.0, thr_de * 0.85))
    return float(thr_de)

def _quant_key(rgb: np.ndarray) -> np.ndarray:
    r = (rgb[..., 0] >> RGB_Q_SHIFT).astype(np.int32)
    g = (rgb[..., 1] >> RGB_Q_SHIFT).astype(np.int32)
    b = (rgb[..., 2] >> RGB_Q_SHIFT).astype(np.int32)
    return (r << 8) | (g << 4) | b  # 12-bit key

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
    steps = 18

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
    step = max(1, len(out) // 80)
    return out[::step]

def _estimate_stroke_color(
    img_rgb: np.ndarray,
    img_lab: np.ndarray,
    bg_lab: Tuple[float, float, float],
    deltae_thr: float,
    stroke_samples_json: List[Tuple[float, float]],
    scale_x: float,
    scale_y: float,
    radius: int,
) -> Tuple[Tuple[int, int, int], float, int]:
    h, w, _ = img_rgb.shape
    if not stroke_samples_json:
        return (0, 0, 0), 0.0, 0

    counts: Dict[int, int] = {}
    sums: Dict[int, np.ndarray] = {}
    contrib = 0
    sampled = 0

    for (xj, yj) in stroke_samples_json:
        x = int(round(xj * scale_x))
        y = int(round(yj * scale_y))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        sampled += 1

        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)

        patch_rgb = img_rgb[y0:y1, x0:x1]
        patch_lab = img_lab[y0:y1, x0:x1]

        de = _delta_e_lab(patch_lab, bg_lab)
        mask = de >= float(deltae_thr)
        if not np.any(mask):
            continue

        pix = patch_rgb[mask]
        if pix.size == 0:
            continue

        keys = _quant_key(pix.reshape(-1, 3))
        for k, rgbpix in zip(keys.tolist(), pix.reshape(-1, 3)):
            counts[k] = counts.get(k, 0) + 1
            if k not in sums:
                sums[k] = rgbpix.astype(np.int64).copy()
            else:
                sums[k] += rgbpix.astype(np.int64)
        contrib += int(pix.shape[0])

    if not counts:
        return (0, 0, 0), 0.0, 0

    best_k = max(counts.keys(), key=lambda k: counts[k])
    c = counts[best_k]
    mean_rgb = (sums[best_k] / max(1, c)).astype(np.int64)
    rgb = (int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2]))

    denom = max(1, sampled * ((2 * radius + 1) * (2 * radius + 1)))
    conf = _clamp(contrib / denom, 0.0, 1.0)

    return rgb, float(conf), int(contrib)


# ============================
# Clustering per color (simple)
# ============================
def _cluster_indices_by_bbox_proximity(
    idxs: List[int],
    bboxes: List[Tuple[float,float,float,float]],
    radius_px: float,
    grid_cell_px: float,
) -> List[List[int]]:
    """
    Clusters strokes by bbox distance <= radius_px, but ONLY among idxs list.
    Uses spatial hashing on bbox centroid to reduce comparisons.

    Returns clusters as lists of stroke indices (original stroke index space).
    """
    if not idxs:
        return []

    local_to_global = idxs
    n = len(local_to_global)

    cent = np.zeros((n, 2), dtype=np.float32)
    for li, gi in enumerate(local_to_global):
        x0,y0,x1,y1 = bboxes[gi]
        cent[li, 0] = (x0 + x1) * 0.5
        cent[li, 1] = (y0 + y1) * 0.5

    cell = float(max(8.0, grid_cell_px))
    grid: Dict[Tuple[int,int], List[int]] = {}

    def cell_key(x: float, y: float) -> Tuple[int,int]:
        return (int(math.floor(x / cell)), int(math.floor(y / cell)))

    for li in range(n):
        k = cell_key(float(cent[li,0]), float(cent[li,1]))
        grid.setdefault(k, []).append(li)

    uf = UnionFind(n)

    for li in range(n):
        cx, cy = float(cent[li,0]), float(cent[li,1])
        gx, gy = cell_key(cx, cy)

        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                bucket = grid.get((gx+ox, gy+oy))
                if not bucket:
                    continue
                for lj in bucket:
                    if lj <= li:
                        continue
                    gi = local_to_global[li]
                    gj = local_to_global[lj]
                    d = _bbox_distance(bboxes[gi], bboxes[gj])
                    if d <= radius_px:
                        uf.union(li, lj)

    groups: Dict[int, List[int]] = {}
    for li in range(n):
        r = uf.find(li)
        groups.setdefault(r, []).append(local_to_global[li])

    out = list(groups.values())
    out.sort(key=lambda g: (min(bboxes[i][0] for i in g), min(bboxes[i][1] for i in g)))
    return out

def _render_cluster_crop_keep_strokes_color(
    img_rgb: np.ndarray,
    x0p: int,
    y0p: int,
    x1p: int,
    y1p: int,
    stroke_indices: List[int],
    stroke_samples_px_cache: List[Optional[np.ndarray]],
    pad_px: int,
) -> np.ndarray:
    """
    Returns an RGB crop where pixels within (skeleton line dilated by pad_px)
    remain in original color, and everything else is grayscaled.
    """
    crop_rgb = img_rgb[y0p:y1p, x0p:x1p]
    ch, cw, _ = crop_rgb.shape
    if ch <= 0 or cw <= 0:
        return crop_rgb

    mask = np.zeros((ch, cw), dtype=np.uint8)

    # Draw skeletons using precomputed pixel coords; use polylines to reduce overhead
    extra = int(pad_px) + 2
    min_x = x0p - extra
    max_x = x1p + extra
    min_y = y0p - extra
    max_y = y1p + extra

    for si in stroke_indices:
        if si < 0 or si >= len(stroke_samples_px_cache):
            continue
        pts_img = stroke_samples_px_cache[si]
        if pts_img is None or pts_img.shape[0] < 2:
            continue

        # Keep points that are near this crop
        x = pts_img[:, 0]
        y = pts_img[:, 1]
        keep = (x >= min_x) & (x < max_x) & (y >= min_y) & (y < max_y)
        if not np.any(keep):
            continue

        pts = pts_img[keep].copy()
        pts[:, 0] -= int(x0p)
        pts[:, 1] -= int(y0p)

        # Clip to crop bounds to avoid cv2 issues
        pts[:, 0] = np.clip(pts[:, 0], 0, cw - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, ch - 1)

        cv2.polylines(mask, [pts.reshape(-1, 1, 2)], isClosed=False, color=255, thickness=1, lineType=cv2.LINE_8)

    if not np.any(mask):
        # If something went wrong, keep original crop rather than outputting all gray.
        return crop_rgb

    # Expand outward from skeleton by pad_px
    if pad_px > 0:
        k = 2 * int(pad_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)

    # Grayscale background
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray3 = np.repeat(gray[..., None], 3, axis=2)

    out = np.where(mask[..., None] > 0, crop_rgb, gray3).astype(np.uint8)
    return out


# ============================
# Main per-image pipeline
# ============================
def process_one(processed_img_path: Path) -> None:
    idx = _extract_index_from_processed_name(processed_img_path.name)
    if idx is None:
        print(f"[skip] can't parse index from {processed_img_path.name}")
        return

    vectors_path = _vector_json_path_for_index(idx)
    if not vectors_path.exists():
        print(f"[skip] missing vectors json: {vectors_path.name}")
        return

    # OUTPUT: one folder per JSON/index
    out_render_dir = CLUSTER_RENDER_DIR / f"processed_{idx}"
    out_map_dir = CLUSTER_MAP_DIR / f"processed_{idx}"
    out_render_dir.mkdir(parents=True, exist_ok=True)
    out_map_dir.mkdir(parents=True, exist_ok=True)

    img_rgb = _load_rgb_image(processed_img_path)
    h_img, w_img, _ = img_rgb.shape

    data = json.loads(vectors_path.read_text(encoding="utf-8"))
    strokes = data.get("strokes") or []
    if not isinstance(strokes, list) or not strokes:
        print(f"[skip] no strokes in {vectors_path.name}")
        return

    fmt = (data.get("vector_format") or "bezier_cubic").lower()
    src_w = float(data.get("width") or w_img)
    src_h = float(data.get("height") or h_img)

    scale_x = (w_img / src_w) if src_w > 0 else 1.0
    scale_y = (h_img / src_h) if src_h > 0 else 1.0

    # --------- RESTORED: assign per-stroke color_rgb/hex/conf/contrib (not used for grouping) ---------
    # This is the slow part. Default OFF via ENABLE_STROKE_COLOR_SAMPLING.
    if ENABLE_STROKE_COLOR_SAMPLING:
        bg_rgb = _estimate_background_rgb(img_rgb)
        img_lab = _rgb_to_lab_u8(img_rgb)
        bg_lab = _lab_of_rgb(bg_rgb)
        deltae_thr = _otsu_threshold_deltae(_delta_e_lab(img_lab, bg_lab))
        radius = SAMPLE_RADIUS_PX_LARGE if max(w_img, h_img) >= 1400 else SAMPLE_RADIUS_PX_SMALL

    # Precompute bbox + color_id per stroke
    bboxes_img: List[Tuple[float,float,float,float]] = []
    color_ids_raw: List[Optional[int]] = []
    stroke_samples_px_cache: List[Optional[np.ndarray]] = []

    for s in strokes:
        if fmt == "bezier_cubic":
            bb = _stroke_bbox_from_beziers(s)
            samples_json = _stroke_samples_json_from_beziers(s)
        else:
            bb = _stroke_bbox_from_polyline(s)
            samples_json = _stroke_samples_json_from_polyline(s)

        # Precompute stroke samples in IMAGE PIXELS (int) once for fast mask rendering
        if samples_json and len(samples_json) >= 2:
            pts = np.array(
                [(int(round(x * scale_x)), int(round(y * scale_y))) for (x, y) in samples_json],
                dtype=np.int32
            )
            # remove consecutive duplicates cheaply
            if pts.shape[0] >= 2:
                d = np.abs(np.diff(pts, axis=0)).sum(axis=1)
                keep = np.concatenate(([True], d > 0))
                pts = pts[keep]
            stroke_samples_px_cache.append(pts if pts.shape[0] >= 2 else None)
        else:
            stroke_samples_px_cache.append(None)

        # assign stored color (sampling)
        if ENABLE_STROKE_COLOR_SAMPLING:
            rgb, conf, contrib = _estimate_stroke_color(
                img_rgb=img_rgb,
                img_lab=img_lab,
                bg_lab=bg_lab,
                deltae_thr=deltae_thr,
                stroke_samples_json=samples_json,
                scale_x=scale_x,
                scale_y=scale_y,
                radius=int(radius),
            )
            s["color_rgb"] = [int(rgb[0]), int(rgb[1]), int(rgb[2])]
            s["color_hex"] = "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            s["color_conf"] = float(round(float(conf), 4))
            s["color_contrib_pixels"] = int(contrib)

        cid = _stroke_color_id_raw(s)
        color_ids_raw.append(cid)

        if bb is None:
            bboxes_img.append((0.0, 0.0, 0.0, 0.0))
        else:
            x0, y0, x1, y1 = bb
            bboxes_img.append((x0*scale_x, y0*scale_y, x1*scale_x, y1*scale_y))

    # normalize ids (0-based -> 1-based if needed)
    color_ids = _normalize_color_ids(color_ids_raw)

    # also ensure color_name exists (derived from id only, no reassigning)
    for s, cid in zip(strokes, color_ids):
        if isinstance(cid, int):
            s["color_id"] = int(cid)
            if not isinstance(s.get("color_name"), str) or not s.get("color_name").strip():
                s["color_name"] = _color_name_from_id(int(cid))

    # persist stroke updates
    if ENABLE_STROKE_COLOR_SAMPLING:
        # write color sampling params back (kept simple)
        data["stroke_color_sampling"] = {
            "bg_rgb": list(bg_rgb),
            "deltae_threshold": float(round(float(deltae_thr), 3)),
            "radius_px": int(radius),
            "rgb_quant_shift": int(RGB_Q_SHIFT),
        }
        data["strokes"] = strokes
        vectors_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # --------- grouping strictly by color_id (no reassigning) ---------
    cluster_entries: List[Dict[str, Any]] = []

    # Build idx lists once (avoid scanning color_ids 11 times)
    by_cid: Dict[int, List[int]] = {}
    for i, v in enumerate(color_ids):
        if isinstance(v, int):
            by_cid.setdefault(int(v), []).append(i)

    for cid in range(1, 12):
        cname = _color_name_from_id(cid)
        idxs = by_cid.get(cid)
        if not idxs:
            continue

        groups = _cluster_indices_by_bbox_proximity(
            idxs=idxs,
            bboxes=bboxes_img,
            radius_px=float(GROUP_RADIUS_PX),
            grid_cell_px=float(GRID_CELL_PX),
        )

        # group index WITHIN THIS COLOUR
        for gi, g in enumerate(groups):
            bb = _cluster_bbox(bboxes_img, g)
            x0, y0, x1, y1 = bb

            x0p = max(0, int(math.floor(x0 - CLUSTER_CROP_PAD)))
            y0p = max(0, int(math.floor(y0 - CLUSTER_CROP_PAD)))
            x1p = min(w_img, int(math.ceil(x1 + CLUSTER_CROP_PAD)))
            y1p = min(h_img, int(math.ceil(y1 + CLUSTER_CROP_PAD)))
            if x1p <= x0p or y1p <= y0p:
                continue

            # DEAD SIMPLE names: {colour}_{k}_{bbox/mask}.png
            bbox_name = f"{cname}_{gi}_bbox.png"
            mask_name = f"{cname}_{gi}_mask.png"

            bbox_path = out_render_dir / bbox_name
            mask_path = out_render_dir / mask_name

            # bbox output is now FULL IMAGE with red rectangle around bbox
            full_with_box = _render_full_with_red_bbox(img_rgb, [x0p, y0p, x1p, y1p], thickness=RECT_THICKNESS_PX)
            Image.fromarray(full_with_box).save(bbox_path, compress_level=int(PNG_COMPRESS_LEVEL))

            # masked crop (new behavior)
            crop_arr = _render_cluster_crop_keep_strokes_color(
                img_rgb=img_rgb,
                x0p=int(x0p), y0p=int(y0p), x1p=int(x1p), y1p=int(y1p),
                stroke_indices=[int(i) for i in g],
                stroke_samples_px_cache=stroke_samples_px_cache,
                pad_px=int(STROKE_SKELETON_PAD_PX),
            )
            Image.fromarray(crop_arr).save(mask_path, compress_level=int(PNG_COMPRESS_LEVEL))

            cluster_entries.append({
                "color_id": int(cid),
                "color_name": cname,
                "group_index_in_color": int(gi),
                "stroke_indexes": [int(i) for i in g],
                "bbox_xyxy": [int(x0p), int(y0p), int(x1p), int(y1p)],
                "crop_file_bbox": bbox_name,
                "crop_file_mask": mask_name,
            })

    cluster_map = {
        "processed_image": processed_img_path.name,
        "vectors_json": vectors_path.name,
        "image_index": int(idx),
        "image_size": [int(w_img), int(h_img)],
        "group_radius_px": float(GROUP_RADIUS_PX),
        "colors_order": COLOR_ORDER_LIGHT_TO_DARK,
        "clusters": cluster_entries,
    }

    map_path = out_map_dir / "clusters.json"
    map_path.write_text(json.dumps(cluster_map, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] {processed_img_path.name}: strokes={len(strokes)} clusters={len(cluster_entries)} -> {map_path}")


def main() -> None:
    _ensure_dirs()
    imgs = _glob_processed_images()
    if not imgs:
        print(f"[error] no processed_<n>.png images found in {PROCESSED_DIR}")
        return

    for p in imgs:
        try:
            process_one(p)
        except Exception as e:
            print(f"[fail] {p.name}: {e}")


if __name__ == "__main__":
    main()