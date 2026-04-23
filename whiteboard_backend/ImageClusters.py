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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


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
GROUP_RADIUS_PX = _env_float("IMAGE_CLUSTERS_GROUP_RADIUS_PX", 28.0)

# spatial hash cell size; should be >= radius so neighbors are local
GRID_CELL_PX = _env_float("IMAGE_CLUSTERS_GRID_CELL_PX", 48.0)

# dynamic post-pass split for very spread-out same-colour clusters
ENABLE_DYNAMIC_INTERNAL_CLUSTER_SPLIT = _env_bool("IMAGE_CLUSTERS_ENABLE_DYNAMIC_INTERNAL_CLUSTER_SPLIT", False)

# We only attempt the post-pass when there is enough internal structure to read.
DYNAMIC_SPLIT_MIN_STROKES_FOR_CHECK = _env_int("IMAGE_CLUSTERS_DYNAMIC_SPLIT_MIN_STROKES_FOR_CHECK", 8)

# "Huge" is treated relatively: cluster span vs its own typical nearest-neighbour spacing.
DYNAMIC_SPLIT_MIN_SPREAD_RATIO = _env_float("IMAGE_CLUSTERS_DYNAMIC_SPLIT_MIN_SPREAD_RATIO", 7.0)

# Split only when candidate bridge edges are clearly weaker than local internal ties.
DYNAMIC_SPLIT_BRIDGE_RATIO = _env_float("IMAGE_CLUSTERS_DYNAMIC_SPLIT_BRIDGE_RATIO", 1.8)
DYNAMIC_SPLIT_MST_GAP_IQR_SCALE = _env_float("IMAGE_CLUSTERS_DYNAMIC_SPLIT_MST_GAP_IQR_SCALE", 1.75)
DYNAMIC_SPLIT_MAX_GROUPS = _env_int("IMAGE_CLUSTERS_DYNAMIC_SPLIT_MAX_GROUPS", 3)
DYNAMIC_SPLIT_MIN_COMPONENT_STROKES = _env_int("IMAGE_CLUSTERS_DYNAMIC_SPLIT_MIN_COMPONENT_STROKES", 3)

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
ENABLE_CROSS_COLOR_CLUSTER_MERGE = _env_bool("IMAGE_CLUSTERS_ENABLE_CROSS_COLOR_CLUSTER_MERGE", True)

# shared bbox growth =
#   (shared_bbox_area - area_of_smaller_cluster_bbox) / area_of_smaller_cluster_bbox
CROSS_COLOR_SHARED_BBOX_GROWTH_MAX = _env_float(
    "IMAGE_CLUSTERS_CROSS_COLOR_SHARED_BBOX_GROWTH_MAX",
    0.50,
)  # allow up to 20% growth vs the smaller cluster


# ============================
# FILLED WRAP MASK FOR OUTPUT CROPS
# ============================
# This is ONLY for the grayscale+color crop output (helps when strokes are outlines).
ENABLE_FILLED_WRAP_MASK = _env_bool("IMAGE_CLUSTERS_ENABLE_FILLED_WRAP_MASK", True)

# How aggressively to bridge gaps between nearby stroke segments inside the crop.
WRAP_BRIDGE_PX = _env_int("IMAGE_CLUSTERS_WRAP_BRIDGE_PX", 6)

# How much to "tighten" after filling (erosion).
WRAP_TIGHTEN_PX = _env_int("IMAGE_CLUSTERS_WRAP_TIGHTEN_PX", 2)

# Pixel thickness used when rasterizing stroke lines into the initial mask
WRAP_STROKE_THICKNESS = _env_int("IMAGE_CLUSTERS_WRAP_STROKE_THICKNESS", 1)


# ============================
# MERGE SEARCH
# ============================
MERGE_GRID_CELL_PX = _env_float("IMAGE_CLUSTERS_MERGE_GRID_CELL_PX", 96.0)


def get_runtime_cluster_settings() -> Dict[str, Any]:
    return {
        "GROUP_RADIUS_PX": float(GROUP_RADIUS_PX),
        "GRID_CELL_PX": float(GRID_CELL_PX),
        "ENABLE_DYNAMIC_INTERNAL_CLUSTER_SPLIT": bool(ENABLE_DYNAMIC_INTERNAL_CLUSTER_SPLIT),
        "DYNAMIC_SPLIT_MIN_STROKES_FOR_CHECK": int(DYNAMIC_SPLIT_MIN_STROKES_FOR_CHECK),
        "DYNAMIC_SPLIT_MIN_SPREAD_RATIO": float(DYNAMIC_SPLIT_MIN_SPREAD_RATIO),
        "DYNAMIC_SPLIT_BRIDGE_RATIO": float(DYNAMIC_SPLIT_BRIDGE_RATIO),
        "DYNAMIC_SPLIT_MST_GAP_IQR_SCALE": float(DYNAMIC_SPLIT_MST_GAP_IQR_SCALE),
        "DYNAMIC_SPLIT_MAX_GROUPS": int(DYNAMIC_SPLIT_MAX_GROUPS),
        "DYNAMIC_SPLIT_MIN_COMPONENT_STROKES": int(DYNAMIC_SPLIT_MIN_COMPONENT_STROKES),
        "ENABLE_CROSS_COLOR_CLUSTER_MERGE": bool(ENABLE_CROSS_COLOR_CLUSTER_MERGE),
        "CROSS_COLOR_SHARED_BBOX_GROWTH_MAX": float(CROSS_COLOR_SHARED_BBOX_GROWTH_MAX),
        "ENABLE_FILLED_WRAP_MASK": bool(ENABLE_FILLED_WRAP_MASK),
        "WRAP_BRIDGE_PX": int(WRAP_BRIDGE_PX),
        "WRAP_TIGHTEN_PX": int(WRAP_TIGHTEN_PX),
        "WRAP_STROKE_THICKNESS": int(WRAP_STROKE_THICKNESS),
        "MERGE_GRID_CELL_PX": float(MERGE_GRID_CELL_PX),
    }


def apply_runtime_cluster_settings(**overrides: Any) -> Dict[str, Any]:
    previous = get_runtime_cluster_settings()
    for key, value in overrides.items():
        if value is None or key not in previous:
            continue
        globals()[key] = value
    return previous


def restore_runtime_cluster_settings(settings: Dict[str, Any]) -> None:
    if not isinstance(settings, dict):
        return
    for key, value in settings.items():
        if key in globals():
            globals()[key] = value


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


def _bbox_union_xyxy(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    return (
        int(min(a[0], b[0])),
        int(min(a[1], b[1])),
        int(max(a[2], b[2])),
        int(max(a[3], b[3])),
    )


def _shared_bbox_growth_vs_smaller(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    """
    How much larger the shared/union bbox is versus the smaller input bbox.
    0.0 means one bbox is fully contained in the other (or identical).
    0.20 means the shared bbox is 20% larger than the smaller bbox.
    """
    a_area = _bbox_area((float(a[0]), float(a[1]), float(a[2]), float(a[3])))
    b_area = _bbox_area((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
    smaller = min(a_area, b_area)
    if smaller <= 0.0:
        return float("inf")

    shared = _bbox_union_xyxy(a, b)
    shared_area = _bbox_area((float(shared[0]), float(shared[1]), float(shared[2]), float(shared[3])))
    return float((shared_area - smaller) / smaller)


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


def _stroke_distance_px(
    i: int,
    j: int,
    bboxes: List[Tuple[float, float, float, float]],
    stroke_pts: List[Optional[np.ndarray]],
    polyline_cutoff_px: float,
) -> float:
    bb_dist = float(_bbox_distance(bboxes[i], bboxes[j]))
    pts_i = stroke_pts[i]
    pts_j = stroke_pts[j]
    if pts_i is None or pts_j is None:
        return bb_dist
    if bb_dist > float(polyline_cutoff_px):
        return bb_dist

    d2 = _polyline_min_distance_d2(pts_i, pts_j)
    if not math.isfinite(d2):
        return bb_dist
    return float(math.sqrt(max(0.0, d2)))


def _pairwise_cluster_distance_matrix(
    idxs: List[int],
    bboxes: List[Tuple[float, float, float, float]],
    stroke_pts: List[Optional[np.ndarray]],
    radius_px: float,
) -> np.ndarray:
    n = len(idxs)
    dist = np.full((n, n), np.inf, dtype=np.float32)
    if n <= 0:
        return dist

    polyline_cutoff_px = float(max(radius_px * 2.0, 16.0))
    for i in range(n):
        dist[i, i] = 0.0
        gi = idxs[i]
        for j in range(i + 1, n):
            gj = idxs[j]
            d = _stroke_distance_px(
                gi,
                gj,
                bboxes=bboxes,
                stroke_pts=stroke_pts,
                polyline_cutoff_px=polyline_cutoff_px,
            )
            dist[i, j] = np.float32(d)
            dist[j, i] = np.float32(d)
    return dist


def _mst_edges_from_dense_distances(dist: np.ndarray) -> List[Tuple[int, int, float]]:
    n = int(dist.shape[0])
    if n <= 1:
        return []

    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf, dtype=np.float32)
    parent = np.full(n, -1, dtype=np.int32)
    best[0] = 0.0

    edges: List[Tuple[int, int, float]] = []
    for _ in range(n):
        masked = np.where(in_tree, np.inf, best)
        u = int(np.argmin(masked))
        if not math.isfinite(float(masked[u])):
            break

        in_tree[u] = True
        if parent[u] >= 0:
            edges.append((int(parent[u]), u, float(best[u])))

        row = dist[u]
        for v in range(n):
            if in_tree[v]:
                continue
            w = float(row[v])
            if w < float(best[v]):
                best[v] = np.float32(w)
                parent[v] = np.int32(u)

    return edges


def _components_from_tree_edges(
    n: int,
    tree_edges: List[Tuple[int, int, float]],
    cut_edge_ids: set[int],
) -> List[List[int]]:
    uf = UnionFind(n)
    for ei, (u, v, _w) in enumerate(tree_edges):
        if ei in cut_edge_ids:
            continue
        uf.union(int(u), int(v))

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values())


def _maybe_split_large_cluster_by_internal_proximity(
    idxs: List[int],
    bboxes: List[Tuple[float, float, float, float]],
    stroke_pts: List[Optional[np.ndarray]],
    radius_px: float,
) -> List[List[int]]:
    """
    Conservative post-pass:
    If a same-colour cluster became large via transitive threshold chaining,
    look for a few unusually long bridge links inside its MST and split only
    when that separation is clearly stronger than the local internal spacing.
    """
    n = len(idxs)
    if n < int(DYNAMIC_SPLIT_MIN_STROKES_FOR_CHECK):
        return [sorted(int(i) for i in idxs)]

    dist = _pairwise_cluster_distance_matrix(
        idxs=idxs,
        bboxes=bboxes,
        stroke_pts=stroke_pts,
        radius_px=radius_px,
    )
    if dist.size == 0:
        return [sorted(int(i) for i in idxs)]

    nn_vals: List[float] = []
    for i in range(n):
        row = dist[i]
        finite = row[np.isfinite(row)]
        finite = finite[finite > 1e-3]
        if finite.size > 0:
            nn_vals.append(float(np.min(finite)))

    if len(nn_vals) < 3:
        return [sorted(int(i) for i in idxs)]

    nn_arr = np.asarray(nn_vals, dtype=np.float32)
    local_scale = float(np.median(nn_arr))
    if not math.isfinite(local_scale) or local_scale <= 0.0:
        return [sorted(int(i) for i in idxs)]

    bb = _cluster_bbox(bboxes, idxs)
    span_w = max(0.0, float(bb[2] - bb[0]))
    span_h = max(0.0, float(bb[3] - bb[1]))
    spread_ratio = float(math.hypot(span_w, span_h) / max(local_scale, 1.0))
    if spread_ratio < float(DYNAMIC_SPLIT_MIN_SPREAD_RATIO):
        return [sorted(int(i) for i in idxs)]

    mst_edges = _mst_edges_from_dense_distances(dist)
    if len(mst_edges) < n - 1:
        return [sorted(int(i) for i in idxs)]

    mst_weights = np.asarray(
        [w for (_u, _v, w) in mst_edges if math.isfinite(w) and w > 1e-3],
        dtype=np.float32,
    )
    if mst_weights.size < 2:
        return [sorted(int(i) for i in idxs)]

    q1 = float(np.percentile(mst_weights, 25))
    q3 = float(np.percentile(mst_weights, 75))
    iqr = max(0.0, q3 - q1)
    bridge_floor = max(
        float(local_scale) * float(DYNAMIC_SPLIT_BRIDGE_RATIO),
        float(np.median(mst_weights)) + float(DYNAMIC_SPLIT_MST_GAP_IQR_SCALE) * iqr,
    )

    candidate_edges = [
        (ei, u, v, w)
        for ei, (u, v, w) in enumerate(mst_edges)
        if math.isfinite(w) and w >= bridge_floor
    ]
    candidate_edges.sort(key=lambda it: it[3], reverse=True)
    if not candidate_edges:
        return [sorted(int(i) for i in idxs)]

    max_cuts = min(
        int(DYNAMIC_SPLIT_MAX_GROUPS) - 1,
        len(candidate_edges),
    )
    best_components: Optional[List[List[int]]] = None
    best_score: Optional[Tuple[float, float]] = None

    for k in range(1, max_cuts + 1):
        cut_ids = {int(ei) for (ei, _u, _v, _w) in candidate_edges[:k]}
        components = _components_from_tree_edges(n, mst_edges, cut_ids)
        if not (2 <= len(components) <= int(DYNAMIC_SPLIT_MAX_GROUPS)):
            continue

        comp_sizes = sorted((len(c) for c in components), reverse=True)
        if comp_sizes and min(comp_sizes) < int(DYNAMIC_SPLIT_MIN_COMPONENT_STROKES):
            continue

        kept_weights = [
            float(w)
            for ei, (_u, _v, w) in enumerate(mst_edges)
            if ei not in cut_ids and math.isfinite(w) and w > 1e-3
        ]
        if not kept_weights:
            continue

        removed_weights = [float(w) for (_ei, _u, _v, w) in candidate_edges[:k]]
        kept_scale = float(np.median(np.asarray(kept_weights, dtype=np.float32)))
        if not math.isfinite(kept_scale) or kept_scale <= 0.0:
            continue

        separation_ratio = float(min(removed_weights) / max(kept_scale, 1e-3))
        if separation_ratio < float(DYNAMIC_SPLIT_BRIDGE_RATIO):
            continue

        score = (separation_ratio, -float(k))
        if best_score is None or score > best_score:
            best_score = score
            best_components = components

    if not best_components:
        return [sorted(int(i) for i in idxs)]

    out_groups: List[List[int]] = []
    for comp in best_components:
        out_groups.append(sorted(int(idxs[li]) for li in comp))
    out_groups.sort(key=lambda g: (min(bboxes[i][0] for i in g), min(bboxes[i][1] for i in g)))
    return out_groups


def _split_large_clusters_by_internal_proximity(
    groups: List[List[int]],
    bboxes: List[Tuple[float, float, float, float]],
    stroke_pts: List[Optional[np.ndarray]],
    radius_px: float,
) -> List[List[int]]:
    if not groups:
        return []

    out: List[List[int]] = []
    for g in groups:
        pieces = _maybe_split_large_cluster_by_internal_proximity(
            idxs=[int(i) for i in g],
            bboxes=bboxes,
            stroke_pts=stroke_pts,
            radius_px=radius_px,
        )
        out.extend(pieces)

    out.sort(key=lambda grp: (min(bboxes[i][0] for i in grp), min(bboxes[i][1] for i in grp)))
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


def _refresh_cluster_merge_geom(
    cluster: Dict[str, Any],
    stroke_pts: List[Optional[np.ndarray]],
) -> None:
    bb = _cluster_tight_bbox_from_strokes(
        [int(x) for x in (cluster.get("stroke_indexes") or [])],
        stroke_pts,
    )
    cluster["_merge_bbox"] = bb
    cluster["_merge_area"] = float(
        _bbox_area((float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])))
    )
    cluster["bbox_raw"] = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))

def _merge_overlapping_clusters_across_colors_fast(
    clusters: List[Dict[str, Any]],
    stroke_pts: List[Optional[np.ndarray]],
    shared_bbox_growth_max: float,
) -> List[Dict[str, Any]]:
    """
    Cross-colour merging using the cluster shared bbox:
    - Build a tight bbox for each cluster from its strokes.
    - First do a small -> big pass.
    - Then do one follow-up pass over the remaining bigger clusters.
    - In both passes, only merge when the shared bbox is no more than
      `shared_bbox_growth_max` larger than the smaller cluster bbox.
    """
    if not clusters:
        return clusters

    work = []
    for c in clusters:
        cc = dict(c)
        cc["_alive"] = True
        cc["stroke_indexes"] = [int(x) for x in (cc.get("stroke_indexes") or [])]
        work.append(cc)

    # Precompute tight merge bbox once
    for c in work:
        _refresh_cluster_merge_geom(c, stroke_pts)

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

    def add_cluster_to_grid(cluster_idx: int) -> None:
        bb = work[cluster_idx]["_merge_bbox"]
        gx0, gy0, gx1, gy1 = cell_range(bb)
        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                grid.setdefault((gx, gy), []).append(cluster_idx)

    for i, _ in enumerate(work):
        add_cluster_to_grid(i)

    def run_merge_pass(*, smallest_first: bool) -> None:
        idxs = [i for i, c in enumerate(work) if c["_alive"]]
        idxs.sort(
            key=lambda i: work[i].get("_merge_area", 0.0),
            reverse=not smallest_first,
        )

        for si in idxs:
            small = work[si]
            if not small["_alive"]:
                continue
            a_small = float(small.get("_merge_area", 0.0))
            if a_small <= 0.0:
                continue

            sbb = small["_merge_bbox"]
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
            best_big_idx = None
            best_growth = None
            best_big_area = None

            for bi in candidates:
                big = work[bi]
                if not big["_alive"]:
                    continue
                # Different colours only (your rule)
                if int(big["color_id"]) == int(small["color_id"]):
                    continue

                a_big = float(big.get("_merge_area", 0.0))
                if a_big < a_small:
                    continue

                growth = _shared_bbox_growth_vs_smaller(
                    small["_merge_bbox"],
                    big["_merge_bbox"],
                )
                if growth > float(shared_bbox_growth_max):
                    continue

                # choose the tightest shared bbox; tie break by the smaller big cluster
                if best_growth is None or growth < best_growth:
                    best_growth = growth
                    best_big = big
                    best_big_idx = bi
                    best_big_area = a_big
                elif growth == best_growth and best_big is not None:
                    if best_big_area is not None and a_big < best_big_area:
                        best_big = big
                        best_big_idx = bi
                        best_big_area = a_big

            if best_big is None or best_big_idx is None:
                continue

            # MERGE: small -> best_big
            set_big = set(best_big["stroke_indexes"])
            for sidx in small["stroke_indexes"]:
                set_big.add(int(sidx))
            best_big["stroke_indexes"] = sorted(set_big)
            _refresh_cluster_merge_geom(best_big, stroke_pts)
            add_cluster_to_grid(best_big_idx)

            small["_alive"] = False

    run_merge_pass(smallest_first=True)
    run_merge_pass(smallest_first=False)

    out = [c for c in work if c["_alive"]]
    for c in out:
        c.pop("_alive", None)
        c.pop("_merge_bbox", None)
        c.pop("_merge_area", None)
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
            if "ENABLE_DYNAMIC_INTERNAL_CLUSTER_SPLIT" in globals() and ENABLE_DYNAMIC_INTERNAL_CLUSTER_SPLIT:
                groups = _split_large_clusters_by_internal_proximity(
                    groups=groups,
                    bboxes=bboxes_img_raw,
                    stroke_pts=stroke_samples_px_cache,
                    radius_px=float(GROUP_RADIUS_PX),
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
            shared_bbox_growth_max = float(globals().get("CROSS_COLOR_SHARED_BBOX_GROWTH_MAX", 0.20))
            clusters_raw = _merge_overlapping_clusters_across_colors_fast(
                clusters=clusters_raw,
                stroke_pts=stroke_samples_px_cache,
                shared_bbox_growth_max=shared_bbox_growth_max,
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



