#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image


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
    CLUSTER_RENDER_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTER_MAP_DIR.mkdir(parents=True, exist_ok=True)

def _glob_processed_images() -> List[Path]:
    # supports both processed_ and proccessed_ spelling
    imgs = sorted(PROCESSED_DIR.glob("proccessed_*.png")) + sorted(PROCESSED_DIR.glob("processed_*.png"))
    uniq = {}
    for p in imgs:
        uniq[p.name.lower()] = p
    return [uniq[k] for k in sorted(uniq.keys())]

def _extract_index_from_processed_name(name: str) -> Optional[int]:
    m = re.search(r"(?:proccessed|processed)_(\d+)\.png$", name.lower())
    if not m:
        return None
    return int(m.group(1))

def _vector_json_path_for_index(n: int) -> Path:
    # vectorizer writes processed_<idx>.json
    return VECTORS_DIR / f"processed_{n}.json"

def _load_rgb_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


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

def _stroke_color_name(stroke: Dict[str, Any]) -> str:
    """
    Prefer explicit 'color_name' if present.
    Otherwise fall back to approximate mapping from color_hex / color_rgb.
    """
    cn = stroke.get("color_name")
    if isinstance(cn, str) and cn.strip():
        return cn.strip().lower()

    # If you already store per-stroke hex/rgb, use it.
    hx = stroke.get("color_hex")
    rgb = stroke.get("color_rgb")

    # Very dumb fallback: map by dominant channel + brightness.
    # (This is only used if you didn't store color_name, so keep it simple.)
    if isinstance(rgb, list) and len(rgb) >= 3:
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    elif isinstance(hx, str) and re.match(r"^#?[0-9a-fA-F]{6}$", hx.strip()):
        h = hx.strip().lstrip("#")
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
    else:
        return "unknown"

    # brightness
    v = (r + g + b) / 3.0
    mx = max(r, g, b)
    mn = min(r, g, b)

    # grayscale
    if mx - mn < 18:
        if v > 220: return "white"
        if v < 50:  return "black"
        return "gray"

    # crude hue-ish logic
    if r > 200 and g > 200 and b < 120: return "yellow"
    if r > 200 and g > 120 and b < 100: return "orange"
    if r > 200 and g < 120 and b > 200: return "magenta"
    if r > 200 and g < 110 and b < 110: return "red"
    if g > 170 and b > 170 and r < 140: return "cyan"
    if g > 170 and r < 140 and b < 140: return "green"
    if b > 170 and r < 140 and g < 160: return "blue"
    if b > 140 and r > 140 and g < 140: return "purple"

    return "unknown"


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

    # local remap: local index -> global stroke index
    local_to_global = idxs
    n = len(local_to_global)

    # centroids for spatial hashing
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

    # stable ordering: left-most bbox then top-most
    out = list(groups.values())
    out.sort(key=lambda g: (min(bboxes[i][0] for i in g), min(bboxes[i][1] for i in g)))
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

    img_rgb = _load_rgb_image(processed_img_path)
    h_img, w_img, _ = img_rgb.shape
    img_pil = Image.fromarray(img_rgb)

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

    # Precompute bbox per stroke in IMAGE coords
    bboxes_img: List[Tuple[float,float,float,float]] = []
    colors: List[str] = []

    for s in strokes:
        if fmt == "bezier_cubic":
            bb = _stroke_bbox_from_beziers(s)
        else:
            bb = _stroke_bbox_from_polyline(s)

        if bb is None:
            bboxes_img.append((0.0,0.0,0.0,0.0))
            colors.append(_stroke_color_name(s))
            continue

        x0,y0,x1,y1 = bb
        bboxes_img.append((x0*scale_x, y0*scale_y, x1*scale_x, y1*scale_y))
        colors.append(_stroke_color_name(s))

    # Build clusters per color, no merging
    cluster_entries: List[Dict[str, Any]] = []
    cluster_id = 0

    for cname in COLOR_ORDER_LIGHT_TO_DARK:
        idxs = [i for i, c in enumerate(colors) if c == cname]
        if not idxs:
            continue

        groups = _cluster_indices_by_bbox_proximity(
            idxs=idxs,
            bboxes=bboxes_img,
            radius_px=float(GROUP_RADIUS_PX),
            grid_cell_px=float(GRID_CELL_PX),
        )

        for g in groups:
            bb = _cluster_bbox(bboxes_img, g)
            x0,y0,x1,y1 = bb

            x0p = max(0, int(math.floor(x0 - CLUSTER_CROP_PAD)))
            y0p = max(0, int(math.floor(y0 - CLUSTER_CROP_PAD)))
            x1p = min(w_img, int(math.ceil(x1 + CLUSTER_CROP_PAD)))
            y1p = min(h_img, int(math.ceil(y1 + CLUSTER_CROP_PAD)))
            if x1p <= x0p or y1p <= y0p:
                continue

            out_name = f"edges_{idx}_color_{cname}_cluster_{cluster_id:04d}.png"
            crop_path = CLUSTER_RENDER_DIR / out_name
            img_pil.crop((x0p, y0p, x1p, y1p)).save(crop_path)

            cluster_entries.append({
                "cluster_id": cluster_id,
                "color_name": cname,
                "source_processed_image": processed_img_path.name,
                "source_vectors_json": vectors_path.name,
                "stroke_indexes": [int(i) for i in g],
                "bbox_xyxy": [int(x0p), int(y0p), int(x1p), int(y1p)],
                "crop_file": out_name,
            })
            cluster_id += 1

    cluster_map = {
        "image_index": idx,
        "processed_image": processed_img_path.name,
        "vectors_json": vectors_path.name,
        "image_size": [int(w_img), int(h_img)],
        "group_radius_px": float(GROUP_RADIUS_PX),
        "colors_order": COLOR_ORDER_LIGHT_TO_DARK,
        "clusters": cluster_entries,
    }

    map_path = CLUSTER_MAP_DIR / f"edges_{idx}_clusters.json"
    map_path.write_text(json.dumps(cluster_map, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] {processed_img_path.name}: strokes={len(strokes)} clusters={len(cluster_entries)} -> {map_path.name}")


def main() -> None:
    _ensure_dirs()
    imgs = _glob_processed_images()
    if not imgs:
        print(f"[error] no processed images found in {PROCESSED_DIR}")
        return

    for p in imgs:
        try:
            process_one(p)
        except Exception as e:
            print(f"[fail] {p.name}: {e}")


if __name__ == "__main__":
    main()
