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
GROUP_RADIUS_PX = 20

# spatial hash cell size; should be >= radius so neighbors are local
GRID_CELL_PX = 48.0

# crop padding (image pixels)
CLUSTER_CROP_PAD = 8

# stroke keep-color padding around skeleton line (pixels)
STROKE_SKELETON_PAD_PX = 3

# Faster PNG writes (lower compression = much faster I/O)
PNG_COMPRESS_LEVEL = 1

# ============================
# THICKENED / PADDED BBOXES (per-stroke -> per-cluster)
# ============================
BBOX_PAD_FRAC = 0.06   # 6% of bbox size
BBOX_PAD_MIN_PX = 5
BBOX_PAD_MAX_PX = 15


# ============================
# CROSS-COLOUR CLUSTER MERGE
# ============================
ENABLE_CROSS_COLOR_CLUSTER_MERGE = True

# overlap ratio = intersection_area / area_of_smaller_cluster_bbox
CROSS_COLOR_OVERLAP_RATIO = 0.70  # "high %" overlap


# ============================
# FILLED WRAP MASK FOR OUTPUT CROPS
# ============================
# This is ONLY for the grayscale+color crop output (helps when strokes are outlines).
ENABLE_FILLED_WRAP_MASK = True

# How aggressively to bridge gaps between nearby stroke segments inside the crop.
WRAP_BRIDGE_PX = 6

# How much to "tighten" after filling (erosion).
WRAP_TIGHTEN_PX = 2

# Pixel thickness used when rasterizing stroke lines into the initial mask
WRAP_STROKE_THICKNESS = 1


# ============================
# OVERLAP MASK FOR MERGING (EXACT PIXEL OVERLAP OF STROKES)
# ============================
# This is used ONLY for cross-colour merge decisions, NOT for output crops.
MERGE_MASK_STROKE_THICKNESS = 1
MERGE_MASK_DILATE_PX = 1  # small dilation so pixel overlap is robust but not over-permissive
MERGE_GRID_CELL_PX = 96.0


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

#============================
#MERGE OVERLAPPING CLUSTERS FROM DIFFERENT COLOURS
# (STILL NEEDED EVEN WITH OPTIMIZED STROKES)
#============================
def _bbox_area(bb: Tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bb
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


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


#POLYLINE COMP LOGIC
def _min_pointset_d2(a: np.ndarray, b: np.ndarray) -> float:
    """
    Minimum squared distance between two point sets.
    a: (Na,2), b: (Nb,2)
    """
    if a is None or b is None:
        return float("inf")
    if a.size == 0 or b.size == 0:
        return float("inf")
    A = a.astype(np.float32)
    B = b.astype(np.float32)
    diff = A[:, None, :] - B[None, :, :]
    d2 = diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1]
    return float(d2.min())


def _min_points_to_polyline_d2(points: np.ndarray, poly: np.ndarray) -> float:
    """
    Minimum squared distance from a set of points to a polyline (piecewise linear).
    points: (N,2)
    poly:   (M,2), M>=2
    """
    if points is None or poly is None:
        return float("inf")
    if points.size == 0 or poly.size == 0:
        return float("inf")
    if poly.shape[0] < 2:
        return _min_pointset_d2(points, poly)

    P = points.astype(np.float32)               # (N,2)
    A = poly[:-1].astype(np.float32)            # (S,2)
    B = poly[1:].astype(np.float32)             # (S,2)
    V = B - A                                   # (S,2)

    VV = V[:, 0] * V[:, 0] + V[:, 1] * V[:, 1]  # (S,)
    VV = np.maximum(VV, 1e-6)                   # avoid div by 0

    # Broadcast:
    # P -> (N,1,2), A -> (1,S,2), V -> (1,S,2)
    W = P[:, None, :] - A[None, :, :]           # (N,S,2)

    t = (W[..., 0] * V[None, :, 0] + W[..., 1] * V[None, :, 1]) / VV[None, :]  # (N,S)
    t = np.clip(t, 0.0, 1.0)

    proj = A[None, :, :] + V[None, :, :] * t[..., None]  # (N,S,2)
    D = P[:, None, :] - proj                               # (N,S,2)

    d2 = D[..., 0] * D[..., 0] + D[..., 1] * D[..., 1]    # (N,S)
    return float(d2.min())


def _polyline_min_distance_d2(a: np.ndarray, b: np.ndarray) -> float:
    """
    Symmetric min squared distance between 2 polylines.
    Works well with your sampled stroke points.
    """
    if a is None or b is None:
        return float("inf")
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("inf")

    # If either is just one point -> pointset distance
    if a.shape[0] < 2 or b.shape[0] < 2:
        return _min_pointset_d2(a, b)

    d2_1 = _min_points_to_polyline_d2(a, b)
    d2_2 = _min_points_to_polyline_d2(b, a)
    return float(min(d2_1, d2_2))


# ============================
# Clustering per color (simple)
# ============================
def _cluster_indices_by_polyline_proximity(
    idxs: List[int],
    bboxes: List[Tuple[float, float, float, float]],
    stroke_pts: List[Optional[np.ndarray]],
    radius_px: float,
    grid_cell_px: float,
) -> List[List[int]]:
    """
    Cluster strokes by TRUE geometric proximity:
    min distance between sampled polylines <= radius_px.

    Uses bbox+grid ONLY for candidate generation (speed).
    """
    if not idxs:
        return []

    local_to_global = idxs
    n = len(local_to_global)

    cell = float(max(8.0, grid_cell_px))
    r = float(max(0.0, radius_px))
    r2 = r * r

    grid: Dict[Tuple[int, int], List[int]] = {}

    def cell_range(x0: float, y0: float, x1: float, y1: float) -> Tuple[int, int, int, int]:
        ex0 = x0 - r
        ey0 = y0 - r
        ex1 = x1 + r
        ey1 = y1 + r
        gx0 = int(math.floor(ex0 / cell))
        gy0 = int(math.floor(ey0 / cell))
        gx1 = int(math.floor(ex1 / cell))
        gy1 = int(math.floor(ey1 / cell))
        return gx0, gy0, gx1, gy1

    # Index by expanded bbox cells (candidate generation)
    for li, gi in enumerate(local_to_global):
        x0, y0, x1, y1 = bboxes[gi]
        gx0, gy0, gx1, gy1 = cell_range(float(x0), float(y0), float(x1), float(y1))
        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                grid.setdefault((gx, gy), []).append(li)

    uf = UnionFind(n)

    # Compare only candidates, but merge using polyline distance
    for li, gi in enumerate(local_to_global):
        x0, y0, x1, y1 = bboxes[gi]
        gx0, gy0, gx1, gy1 = cell_range(float(x0), float(y0), float(x1), float(y1))

        candidates = set()
        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                bucket = grid.get((gx, gy))
                if not bucket:
                    continue
                for lj in bucket:
                    if lj > li:
                        candidates.add(lj)

        pts_i = stroke_pts[gi]

        for lj in candidates:
            gj = local_to_global[lj]
            pts_j = stroke_pts[gj]

            # If either stroke has no samples, fallback to bbox distance
            if pts_i is None or pts_j is None:
                d = _bbox_distance(bboxes[gi], bboxes[gj])
                if d <= r:
                    uf.union(li, lj)
                continue

            d2 = _polyline_min_distance_d2(pts_i, pts_j)
            if d2 <= r2:
                uf.union(li, lj)

    groups: Dict[int, List[int]] = {}
    for li in range(n):
        root = uf.find(li)
        groups.setdefault(root, []).append(local_to_global[li])

    out = list(groups.values())
    out.sort(key=lambda g: (min(bboxes[i][0] for i in g), min(bboxes[i][1] for i in g)))
    return out


def _render_cluster_crop_keep_strokes_color_filled(
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
    Same output idea as _render_cluster_crop_keep_strokes_color, BUT:
    it attempts to "wrap" the stroke set and color-fill the inside region
    (helps when your skeleton strokes are outlines of the real object).
    """
    crop_rgb = img_rgb[y0p:y1p, x0p:x1p]
    ch, cw, _ = crop_rgb.shape
    if ch <= 0 or cw <= 0:
        return crop_rgb

    skel = np.zeros((ch, cw), dtype=np.uint8)

    extra = int(max(pad_px, WRAP_BRIDGE_PX)) + 2
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

        x = pts_img[:, 0]
        y = pts_img[:, 1]
        keep = (x >= min_x) & (x < max_x) & (y >= min_y) & (y < max_y)
        if not np.any(keep):
            continue

        pts = pts_img[keep].copy()
        pts[:, 0] -= int(x0p)
        pts[:, 1] -= int(y0p)

        pts[:, 0] = np.clip(pts[:, 0], 0, cw - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, ch - 1)

        cv2.polylines(
            skel,
            [pts.reshape(-1, 1, 2)],
            isClosed=False,
            color=255,
            thickness=int(max(1, WRAP_STROKE_THICKNESS)),
            lineType=cv2.LINE_8,
        )

    if not np.any(skel):
        return crop_rgb

    # baseline stroke thickness so strokes themselves stay colored
    if pad_px > 0:
        k = 2 * int(pad_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        thick = cv2.dilate(skel, kernel, iterations=1)
    else:
        thick = skel.copy()

    if not ENABLE_FILLED_WRAP_MASK:
        # fall back to skeleton behavior
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        gray3 = np.repeat(gray[..., None], 3, axis=2)
        out = np.where(thick[..., None] > 0, crop_rgb, gray3).astype(np.uint8)
        return out

    # "wrap" behavior: bridge nearby strokes, then fill the resulting shape
    wrap = thick.copy()

    if WRAP_BRIDGE_PX > 0:
        k = 2 * int(WRAP_BRIDGE_PX) + 1
        kernel_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        wrap = cv2.morphologyEx(wrap, cv2.MORPH_CLOSE, kernel_b, iterations=1)

    # Fill outer contour(s) to get a non-rect region
    contours, _ = cv2.findContours(wrap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(wrap)
    if contours:
        cv2.drawContours(filled, contours, contourIdx=-1, color=255, thickness=-1)

    # Tighten (optional)
    if WRAP_TIGHTEN_PX > 0 and np.any(filled):
        k = 2 * int(WRAP_TIGHTEN_PX) + 1
        kernel_t = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        filled = cv2.erode(filled, kernel_t, iterations=1)

    # Always keep the thick stroke pixels even if filling fails
    keep_mask = np.maximum(filled, thick)

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray3 = np.repeat(gray[..., None], 3, axis=2)

    out = np.where(keep_mask[..., None] > 0, crop_rgb, gray3).astype(np.uint8)
    return out


# ============================
# FAST EXACT-PIXEL OVERLAP MERGE (CROSS-COLOUR)
# ============================
def _cluster_tight_bbox_from_strokes(
    stroke_indices: List[int],
    stroke_pts: List[Optional[np.ndarray]],
) -> Tuple[int, int, int, int]:
    """
    Returns a tight integer bbox [x0,y0,x1,y1) around sampled points of the strokes.
    """
    xs = []
    ys = []
    for si in stroke_indices:
        if si < 0 or si >= len(stroke_pts):
            continue
        p = stroke_pts[si]
        if p is None or p.shape[0] < 1:
            continue
        xs.append(int(np.min(p[:, 0])))
        xs.append(int(np.max(p[:, 0])))
        ys.append(int(np.min(p[:, 1])))
        ys.append(int(np.max(p[:, 1])))

    if not xs or not ys:
        return (0, 0, 0, 0)

    x0 = int(min(xs))
    y0 = int(min(ys))
    x1 = int(max(xs)) + 1
    y1 = int(max(ys)) + 1
    return (x0, y0, x1, y1)

def _cluster_stroke_mask_local(
    stroke_indices: List[int],
    stroke_pts: List[Optional[np.ndarray]],
    bbox_xyxy: Tuple[int, int, int, int],
    dilate_px: int,
    thickness: int,
) -> Tuple[np.ndarray, int]:
    """
    Rasterize cluster strokes into a local mask (tight bbox coords), then (optional) dilate.
    Returns (mask01, area_pixels) where mask01 is uint8 {0,1}.
    """
    x0, y0, x1, y1 = bbox_xyxy
    w = int(max(0, x1 - x0))
    h = int(max(0, y1 - y0))
    if w <= 0 or h <= 0:
        return np.zeros((0, 0), dtype=np.uint8), 0

    m = np.zeros((h, w), dtype=np.uint8)

    for si in stroke_indices:
        if si < 0 or si >= len(stroke_pts):
            continue
        pts_img = stroke_pts[si]
        if pts_img is None or pts_img.shape[0] < 2:
            continue

        pts = pts_img.copy()
        pts[:, 0] -= int(x0)
        pts[:, 1] -= int(y0)

        # Clip to mask bounds
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        cv2.polylines(
            m,
            [pts.reshape(-1, 1, 2)],
            isClosed=False,
            color=255,
            thickness=int(max(1, thickness)),
            lineType=cv2.LINE_8,
        )

    if not np.any(m):
        return np.zeros((h, w), dtype=np.uint8), 0

    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.dilate(m, ker, iterations=1)

    m01 = (m > 0).astype(np.uint8)
    area = int(np.count_nonzero(m01))
    return m01, area

def _mask_overlap_ratio_small_in_big(
    small_bbox: Tuple[int, int, int, int],
    small_mask01: np.ndarray,
    small_area: int,
    big_bbox: Tuple[int, int, int, int],
    big_mask01: np.ndarray,
) -> float:
    """
    overlap ratio = intersection_pixels / small_area  (exact pixels)
    """
    if small_area <= 0:
        return 0.0
    if small_mask01.size == 0 or big_mask01.size == 0:
        return 0.0

    sx0, sy0, sx1, sy1 = small_bbox
    bx0, by0, bx1, by1 = big_bbox

    ix0 = max(sx0, bx0)
    iy0 = max(sy0, by0)
    ix1 = min(sx1, bx1)
    iy1 = min(sy1, by1)

    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    # slices into small
    s_x0 = ix0 - sx0
    s_y0 = iy0 - sy0
    s_x1 = ix1 - sx0
    s_y1 = iy1 - sy0

    # slices into big
    b_x0 = ix0 - bx0
    b_y0 = iy0 - by0
    b_x1 = ix1 - bx0
    b_y1 = iy1 - by0

    s_sub = small_mask01[s_y0:s_y1, s_x0:s_x1]
    b_sub = big_mask01[b_y0:b_y1, b_x0:b_x1]

    if s_sub.size == 0 or b_sub.size == 0:
        return 0.0

    inter = int(np.count_nonzero((s_sub & b_sub) > 0))
    return float(inter / float(max(1, small_area)))

def _merge_overlapping_clusters_across_colors_fast(
    clusters: List[Dict[str, Any]],
    stroke_pts: List[Optional[np.ndarray]],
    overlap_ratio_thr: float,
) -> List[Dict[str, Any]]:
    """
    Cross-colour merging (FAST + robust):
    - Build an exact pixel mask of STROKES for each cluster (tight bbox local).
    - For each small cluster, find a bigger cluster of different color such that:
        intersection_pixels / small_pixels >= overlap_ratio_thr
      Then merge strokes small -> big and delete small.

    This avoids square bbox overlap (your old issue), and removes the insane nested polyline loops.
    """
    if not clusters:
        return clusters

    work = []
    for c in clusters:
        cc = dict(c)
        cc["_alive"] = True
        cc["stroke_indexes"] = [int(x) for x in (cc.get("stroke_indexes") or [])]
        work.append(cc)

    # Precompute tight bbox + local stroke masks once
    for c in work:
        st = c["stroke_indexes"]
        bb = _cluster_tight_bbox_from_strokes(st, stroke_pts)
        c["_mask_bbox"] = bb
        m01, area = _cluster_stroke_mask_local(
            stroke_indices=st,
            stroke_pts=stroke_pts,
            bbox_xyxy=bb,
            dilate_px=int(MERGE_MASK_DILATE_PX),
            thickness=int(MERGE_MASK_STROKE_THICKNESS),
        )
        c["_mask01"] = m01
        c["_mask_area"] = int(area)

    # Spatial grid for candidate lookup (bbox-based ONLY for speed)
    cell = float(max(16.0, MERGE_GRID_CELL_PX))
    grid: Dict[Tuple[int, int], List[int]] = {}

    def cell_range(bb: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = bb
        gx0 = int(math.floor(float(x0) / cell))
        gy0 = int(math.floor(float(y0) / cell))
        gx1 = int(math.floor(float(x1) / cell))
        gy1 = int(math.floor(float(y1) / cell))
        return gx0, gy0, gx1, gy1

    for i, c in enumerate(work):
        bb = c["_mask_bbox"]
        gx0, gy0, gx1, gy1 = cell_range(bb)
        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                grid.setdefault((gx, gy), []).append(i)

    # Process small -> big
    idxs = list(range(len(work)))
    idxs.sort(key=lambda i: work[i].get("_mask_area", 0))  # smallest first

    for si in idxs:
        small = work[si]
        if not small["_alive"]:
            continue
        a_small = int(small.get("_mask_area", 0))
        if a_small <= 0:
            continue

        sbb = small["_mask_bbox"]
        gx0, gy0, gx1, gy1 = cell_range(sbb)

        candidates = set()
        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                bucket = grid.get((gx, gy))
                if bucket:
                    for bi in bucket:
                        if bi != si:
                            candidates.add(bi)

        best_big = None
        best_ratio = 0.0
        best_big_area = None

        for bi in candidates:
            big = work[bi]
            if not big["_alive"]:
                continue
            # Different colours only (your rule)
            if int(big["color_id"]) == int(small["color_id"]):
                continue

            a_big = int(big.get("_mask_area", 0))
            if a_big < a_small:
                continue

            ratio = _mask_overlap_ratio_small_in_big(
                small_bbox=small["_mask_bbox"],
                small_mask01=small["_mask01"],
                small_area=a_small,
                big_bbox=big["_mask_bbox"],
                big_mask01=big["_mask01"],
            )

            if ratio < float(overlap_ratio_thr):
                continue

            # choose best by ratio, tie break by smallest big area (tighter containment)
            if ratio > best_ratio:
                best_ratio = ratio
                best_big = big
                best_big_area = a_big
            elif ratio == best_ratio and best_big is not None:
                if best_big_area is not None and a_big < best_big_area:
                    best_big = big
                    best_big_area = a_big

        if best_big is None:
            continue

        # MERGE: small -> best_big
        set_big = set(best_big["stroke_indexes"])
        for sidx in small["stroke_indexes"]:
            set_big.add(int(sidx))
        best_big["stroke_indexes"] = sorted(set_big)

        small["_alive"] = False

    out = [c for c in work if c["_alive"]]
    for c in out:
        c.pop("_alive", None)
        c.pop("_mask_bbox", None)
        c.pop("_mask01", None)
        c.pop("_mask_area", None)
    return out

def cluster_in_memory(
    preproc_by_idx: dict[int, dict],
    vectors_by_idx: dict[int, dict],
    *,
    save_outputs: bool = False,
) -> dict[int, dict]:
    """
    Returns:
      clusters_by_idx[idx] = {
        "clusters": cluster_entries (same structure as clusters.json),
        "renders_mask_rgb": {mask_filename: np.ndarray RGB uint8}
      }
    """
    out: dict[int, dict] = {}

    for idx, prep in preproc_by_idx.items():
        vec = vectors_by_idx.get(idx)
        if vec is None:
            continue

        cleaned_bgr = prep["cleaned_bgr"]
        img_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)

        h_img, w_img, _ = img_rgb.shape
        data = vec
        strokes = data.get("strokes") or []
        if not isinstance(strokes, list) or not strokes:
            continue

        fmt = (data.get("vector_format") or "bezier_cubic").lower()
        src_w = float(data.get("width") or w_img)
        src_h = float(data.get("height") or h_img)

        scale_x = (w_img / src_w) if src_w > 0 else 1.0
        scale_y = (h_img / src_h) if src_h > 0 else 1.0

        color_ids_raw: List[Optional[int]] = []
        stroke_samples_px_cache: List[Optional[np.ndarray]] = []
        bboxes_img_raw: List[Tuple[float, float, float, float]] = []
        bboxes_img_pad: List[Tuple[float, float, float, float]] = []

        for s in strokes:
            if fmt == "bezier_cubic":
                bb = _stroke_bbox_from_beziers(s)
                samples_json = _stroke_samples_json_from_beziers(s)
            else:
                bb = _stroke_bbox_from_polyline(s)
                samples_json = _stroke_samples_json_from_polyline(s)

            pts: Optional[np.ndarray] = None
            if samples_json and len(samples_json) >= 2:
                pts = np.array(
                    [(int(round(x * scale_x)), int(round(y * scale_y))) for (x, y) in samples_json],
                    dtype=np.int32
                )
                if pts.shape[0] >= 2:
                    d = np.abs(np.diff(pts, axis=0)).sum(axis=1)
                    keep = np.concatenate(([True], d > 0))
                    pts = pts[keep]
                stroke_samples_px_cache.append(pts if pts.shape[0] >= 2 else None)
            else:
                stroke_samples_px_cache.append(None)

            cid = _stroke_color_id_raw(s)
            color_ids_raw.append(cid)

            if pts is not None and pts.shape[0] >= 2:
                x0i = float(np.min(pts[:, 0]))
                y0i = float(np.min(pts[:, 1]))
                x1i = float(np.max(pts[:, 0]))
                y1i = float(np.max(pts[:, 1]))
                raw = (x0i, y0i, x1i, y1i)
            else:
                if bb is None:
                    raw = (0.0, 0.0, 0.0, 0.0)
                else:
                    x0, y0, x1, y1 = bb
                    raw = (float(x0 * scale_x), float(y0 * scale_y), float(x1 * scale_x), float(y1 * scale_y))

            bboxes_img_raw.append(raw)

            x0i, y0i, x1i, y1i = raw
            w = max(1.0, x1i - x0i)
            h = max(1.0, y1i - y0i)
            pad = max(BBOX_PAD_MIN_PX, min(BBOX_PAD_MAX_PX, BBOX_PAD_FRAC * max(w, h)))
            padbb = (x0i - pad, y0i - pad, x1i + pad, y1i + pad)
            bboxes_img_pad.append(padbb)

        color_ids = _normalize_color_ids(color_ids_raw)

        for s, cid in zip(strokes, color_ids):
            if isinstance(cid, int):
                s["color_id"] = int(cid)
                if not isinstance(s.get("color_name"), str) or not s.get("color_name").strip():
                    s["color_name"] = _color_name_from_id(int(cid))

        cluster_entries: List[Dict[str, Any]] = []
        renders_mask_rgb: Dict[str, np.ndarray] = {}

        by_cid: Dict[int, List[int]] = {}
        for i, v in enumerate(color_ids):
            if isinstance(v, int):
                by_cid.setdefault(int(v), []).append(i)

        clusters_raw: List[Dict[str, Any]] = []
        for cid in range(1, 12):
            cname = _color_name_from_id(cid)
            idxs = by_cid.get(cid)
            if not idxs:
                continue

            groups = _cluster_indices_by_polyline_proximity(
                idxs=idxs,
                bboxes=bboxes_img_raw,
                stroke_pts=stroke_samples_px_cache,
                radius_px=float(GROUP_RADIUS_PX),
                grid_cell_px=float(GRID_CELL_PX),
            )

            for g in groups:
                bb = _cluster_bbox(bboxes_img_raw, g)
                clusters_raw.append({
                    "color_id": int(cid),
                    "color_name": cname,
                    "stroke_indexes": [int(i) for i in g],
                    "bbox_raw": bb,
                })

        if "ENABLE_CROSS_COLOR_CLUSTER_MERGE" in globals() and ENABLE_CROSS_COLOR_CLUSTER_MERGE:
            overlap_thr = float(globals().get("CROSS_COLOR_OVERLAP_RATIO", 0.75))
            clusters_raw = _merge_overlapping_clusters_across_colors_fast(
                clusters=clusters_raw,
                stroke_pts=stroke_samples_px_cache,
                overlap_ratio_thr=overlap_thr,
            )

        clusters_by_color: Dict[int, List[Dict[str, Any]]] = {}
        for c in clusters_raw:
            clusters_by_color.setdefault(int(c["color_id"]), []).append(c)

        for cid in range(1, 12):
            cname = _color_name_from_id(cid)
            col_clusters = clusters_by_color.get(cid)
            if not col_clusters:
                continue

            col_clusters.sort(key=lambda c: (c["bbox_raw"][0], c["bbox_raw"][1]))

            for gi, c in enumerate(col_clusters):
                g = [int(i) for i in (c.get("stroke_indexes") or [])]
                if not g:
                    continue

                bb_pad = _cluster_bbox(bboxes_img_pad, g)
                x0, y0, x1, y1 = bb_pad

                x0p = max(0, int(math.floor(x0 - CLUSTER_CROP_PAD)))
                y0p = max(0, int(math.floor(y0 - CLUSTER_CROP_PAD)))
                x1p = min(w_img, int(math.ceil(x1 + CLUSTER_CROP_PAD)))
                y1p = min(h_img, int(math.ceil(y1 + CLUSTER_CROP_PAD)))
                if x1p <= x0p or y1p <= y0p:
                    continue

                mask_name = f"{cname}_{gi}_mask.png"

                crop_arr = _render_cluster_crop_keep_strokes_color_filled(
                    img_rgb=img_rgb,
                    x0p=int(x0p), y0p=int(y0p), x1p=int(x1p), y1p=int(y1p),
                    stroke_indices=[int(i) for i in g],
                    stroke_samples_px_cache=stroke_samples_px_cache,
                    pad_px=int(STROKE_SKELETON_PAD_PX),
                )
                renders_mask_rgb[mask_name] = crop_arr

                if save_outputs:
                    out_render_dir = CLUSTER_RENDER_DIR / f"processed_{idx}"
                    out_map_dir = CLUSTER_MAP_DIR / f"processed_{idx}"
                    out_render_dir.mkdir(parents=True, exist_ok=True)
                    out_map_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(crop_arr).save(out_render_dir / mask_name, compress_level=int(PNG_COMPRESS_LEVEL))

                cluster_entries.append({
                    "color_id": int(cid),
                    "color_name": cname,
                    "group_index_in_color": int(gi),
                    "stroke_indexes": [int(i) for i in g],
                    "bbox_xyxy": [int(x0p), int(y0p), int(x1p), int(y1p)],
                    "crop_file_mask": mask_name,
                })

        if save_outputs:
            out_map_dir = CLUSTER_MAP_DIR / f"processed_{idx}"
            out_map_dir.mkdir(parents=True, exist_ok=True)
            cluster_map = {
                "image_index": int(idx),
                "image_size": [int(w_img), int(h_img)],
                "clusters": cluster_entries,
            }
            (out_map_dir / "clusters.json").write_text(
                json.dumps(cluster_map, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        out[idx] = {"clusters": cluster_entries, "renders_mask_rgb": renders_mask_rgb}

    return out



