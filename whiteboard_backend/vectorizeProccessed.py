#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ------------------- PATHS -------------------
BASE = Path(__file__).resolve().parent
IN_DIR  = BASE / "Skeletonized"   # skeleton PNGs from previous step
OUT_DIR = BASE / "StrokeVectors"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------- GLOBAL KNOBS -------------------

# RDP simplification strength (fraction of image diagonal)
RDP_EPS_FRAC = 0.003

# Chaikin smoothing iterations (0..2 is reasonable)
CHAIKIN_ITERS = 1

# Minimum stroke length (fraction of diagonal) – drop super tiny crap
MIN_STROKE_LEN_FRAC = 0.01

# Angle logic
ANGLE_BREAK_DEG   = 30.0   # sharp turn above this → break stroke
DIR_WINDOW_BACK   = 3      # how many pixels back to estimate direction

# Safe checks
def finite2(x: float) -> bool:
    return math.isfinite(x)

def finite_seq(seq) -> bool:
    return all(math.isfinite(float(v)) for v in seq)

def finite_nd(a: np.ndarray) -> bool:
    return np.isfinite(a).all()

def nan_to_num_nd(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, copy=False)

# ------------------- LOAD SKELETON -------------------

def _load_skeleton_binary(path: Path) -> np.ndarray:
    """
    Read skeleton PNG, return mask 0/1 (1 = skeleton pixel).
    Assumes skeleton is white-ish on black background.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    # Just threshold near mid
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    skel = (bw > 0).astype(np.uint8)
    return skel

# ------------------- GRAPH / STROKE TRACING -------------------

# 8-neighbourhood offsets
NEIGH_OFFS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

def _neighbors(mask: np.ndarray, y: int, x: int) -> List[Tuple[int, int]]:
    h, w = mask.shape
    out = []
    for dy, dx in NEIGH_OFFS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
            out.append((ny, nx))
    return out

def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    # u, v: 2D vectors
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    u /= nu
    v /= nv
    c = float(np.dot(u, v))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))

def _trace_stroke_from(
    start_y: int,
    start_x: int,
    skel: np.ndarray,
    deg: np.ndarray,
    visited_edges: set,
    w: int,
) -> List[Tuple[float, float]]:
    """
    Walk along skeleton edges, starting at (start_y, start_x),
    using an angular continuity criterion and visited_edges to avoid loops.
    Returns list of (x, y) floats forming one stroke.
    """
    path: List[Tuple[float, float]] = []
    h, _ = skel.shape

    curr_y, curr_x = start_y, start_x
    prev_y, prev_x = None, None
    prev_dir = None  # direction vector

    # helper: edge id
    def edge_id(y1, x1, y2, x2):
        a = y1 * w + x1
        b = y2 * w + x2
        return (a, b) if a < b else (b, a)

    while True:
        path.append((float(curr_x), float(curr_y)))

        # Collect candidates (neighbors with unused edges)
        neighs = _neighbors(skel, curr_y, curr_x)
        candidates = []
        for ny, nx in neighs:
            eid = edge_id(curr_y, curr_x, ny, nx)
            if eid in visited_edges:
                continue
            # don't immediately backtrack if we have alternatives
            if prev_y is not None and ny == prev_y and nx == prev_x and len(neighs) > 1:
                continue
            candidates.append((ny, nx))

        if not candidates:
            break

        # For the first step, we don't have a direction yet
        if prev_dir is None:
            # At junction, prefer direction that leads out (lower degree), but it's not critical
            candidates.sort(key=lambda p: deg[p[0], p[1]])
            next_y, next_x = candidates[0]
            dvec = np.array([next_x - curr_x, next_y - curr_y], dtype=np.float32)
            prev_dir = dvec
        else:
            # choose candidate with smallest angle to prev_dir
            best = None
            best_ang = 1e9
            base_dir = prev_dir

            # Optionally use multiple previous points to smooth direction
            if len(path) > DIR_WINDOW_BACK:
                x0, y0 = path[max(0, len(path)-1-DIR_WINDOW_BACK)]
                base_dir = np.array([curr_x - x0, curr_y - y0], dtype=np.float32)

            for ny, nx in candidates:
                dvec = np.array([nx - curr_x, ny - curr_y], dtype=np.float32)
                ang = _angle_between(base_dir, dvec)
                if ang < best_ang:
                    best_ang = ang
                    best = (ny, nx, dvec)

            if best is None:
                break

            # sharp corner → break stroke here, don't consume edge
            if best_ang > ANGLE_BREAK_DEG:
                break

            next_y, next_x, dvec = best
            prev_dir = dvec

        # mark edge as used
        eid = edge_id(curr_y, curr_x, next_y, next_x)
        visited_edges.add(eid)

        # move
        prev_y, prev_x = curr_y, curr_x
        curr_y, curr_x = next_y, next_x

    # make sure it's not degenerate
    if len(path) < 2:
        return []
    return path

def _extract_strokes_from_skeleton(skel01: np.ndarray) -> List[np.ndarray]:
    """
    Main stroke extraction:
      - compute degree for each skeleton pixel
      - trace from endpoints first (deg == 1)
      - then trace from any remaining unused edges
    Returns list of polylines as N×2 float arrays (x,y).
    """
    skel = (skel01 > 0).astype(np.uint8)
    h, w = skel.shape
    if skel.max() == 0:
        return []

    # degree map
    deg = np.zeros_like(skel, dtype=np.uint8)
    ys, xs = np.where(skel > 0)
    for y, x in zip(ys, xs):
        deg[y, x] = len(_neighbors(skel, y, x))

    # set of used edges (by id)
    visited_edges = set()

    strokes: List[np.ndarray] = []

    # Endpoint seeds first (deg == 1)
    endpoints = [(int(y), int(x)) for y, x in zip(ys, xs) if deg[y, x] == 1]

    for sy, sx in endpoints:
        # Check if there is any unused edge from here
        neighs = _neighbors(skel, sy, sx)
        has_free_edge = False
        for ny, nx in neighs:
            a = sy * w + sx
            b = ny * w + nx
            eid = (a, b) if a < b else (b, a)
            if eid not in visited_edges:
                has_free_edge = True
                break
        if not has_free_edge:
            continue

        path = _trace_stroke_from(sy, sx, skel, deg, visited_edges, w)
        if path:
            arr = np.asarray(path, dtype=np.float32)
            nan_to_num_nd(arr)
            if finite_nd(arr) and arr.shape[0] >= 2:
                strokes.append(arr)

    # Now any remaining unused edges (loops / junction leftovers)
    for sy, sx in zip(ys, xs):
        # for each pixel, see if it has an unused edge
        neighs = _neighbors(skel, sy, sx)
        start_needed = False
        for ny, nx in neighs:
            a = sy * w + sx
            b = ny * w + nx
            eid = (a, b) if a < b else (b, a)
            if eid not in visited_edges:
                start_needed = True
                break
        if not start_needed:
            continue

        path = _trace_stroke_from(sy, sx, skel, deg, visited_edges, w)
        if path:
            arr = np.asarray(path, dtype=np.float32)
            nan_to_num_nd(arr)
            if finite_nd(arr) and arr.shape[0] >= 2:
                strokes.append(arr)

    return strokes

# ------------------- RDP / CHAIKIN / BÉZIER -------------------

def _rdp(points: np.ndarray, eps: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] <= 2:
        return points
    start, end = points[0], points[-1]
    seg = end - start
    seglen2 = float(seg[0]*seg[0] + seg[1]*seg[1])
    if seglen2 <= 1e-12:
        return np.vstack((start, end))

    dmax, idx = 0.0, -1
    for i in range(1, points.shape[0]-1):
        v = points[i] - start
        proj = (v[0]*seg[0] + v[1]*seg[1]) / seglen2
        proj = max(0.0, min(1.0, proj))
        closest = start + proj * seg
        d = float(np.linalg.norm(points[i] - closest))
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps and idx > 0:
        left = _rdp(points[:idx+1], eps)
        right = _rdp(points[idx:], eps)
        out = np.vstack((left[:-1], right))
        nan_to_num_nd(out)
        return out
    else:
        out = np.vstack((start, end))
        nan_to_num_nd(out)
        return out

def _chaikin(points: np.ndarray, iters: int) -> np.ndarray:
    if iters <= 0 or points.shape[0] < 3:
        return points
    P = points.astype(np.float32, copy=True)
    for _ in range(iters):
        Q = [P[0]]
        for i in range(P.shape[0] - 1):
            p = P[i]
            r = P[i+1]
            q = 0.75 * p + 0.25 * r
            s = 0.25 * p + 0.75 * r
            Q.extend([q, s])
        Q.append(P[-1])
        P = np.asarray(Q, dtype=np.float32)
        nan_to_num_nd(P)
    return P

def _catmull_rom_to_beziers(pts: np.ndarray, alpha: float = 0.5) -> List[List[float]]:
    """
    Convert open polyline to list of cubic Béziers:
      [x0,y0, cx1,cy1, cx2,cy2, x1,y1]
    """
    P = pts.astype(np.float32)
    nan_to_num_nd(P)
    n = P.shape[0]
    if n < 2:
        return []

    def tj(ti, Pi, Pj):
        d = float(np.linalg.norm(Pj - Pi))
        if not math.isfinite(d):
            d = 0.0
        return ti + (d ** alpha)

    beziers: List[List[float]] = []

    P_ext = np.vstack((P[0], P, P[-1]))
    t = [0.0]
    for i in range(1, P_ext.shape[0]):
        t.append(tj(t[-1], P_ext[i-1], P_ext[i]))

    for k in range(1, n+0):
        i1 = k
        if i1 >= n:
            break

        Pm = P_ext[k-1].astype(np.float32)
        P0 = P_ext[k].astype(np.float32)
        P1 = P_ext[k+1].astype(np.float32)
        Pp = P_ext[k+2].astype(np.float32)

        tm, t0, t1, tp = t[k-1], t[k], t[k+1], t[k+2]
        dt0 = (t1 - tm)
        dt1 = (tp - t0)
        if abs(dt0) < 1e-12 or abs(dt1) < 1e-12:
            m0 = (P1 - P0)
            m1 = (P1 - P0)
        else:
            m0 = ((P1 - Pm) / (t1 - tm) - (P0 - Pm) / (t0 - tm)) * (t1 - t0)
            m1 = ((Pp - P0) / (tp - t0) - (P1 - P0) / (t1 - t0)) * (t1 - t0)

        c1 = P0 + m0 / 3.0
        c2 = P1 - m1 / 3.0

        seg = [
            float(P0[0]), float(P0[1]),
            float(c1[0]), float(c1[1]),
            float(c2[0]), float(c2[1]),
            float(P1[0]), float(P1[1]),
        ]
        if finite_seq(seg):
            beziers.append(seg)

    return beziers

# ------------------- PROCESS SINGLE -------------------

def _poly_length(pp: np.ndarray) -> float:
    if pp.shape[0] < 2:
        return 0.0
    d = np.linalg.norm(np.diff(pp, axis=0), axis=1).sum()
    if not math.isfinite(d):
        return 0.0
    return float(d)

def _process_single(path: Path) -> Tuple[str, dict]:
    skel = _load_skeleton_binary(path)
    H, W = skel.shape
    diag = math.hypot(W, H)

    t0 = time.time()

    # 1) extract raw polylines from skeleton
    polys = _extract_strokes_from_skeleton(skel)

    # 2) simplify + smooth (light)
    rdp_eps = RDP_EPS_FRAC * diag
    polys = [_rdp(p, rdp_eps) for p in polys if p.shape[0] >= 2]
    polys = [_chaikin(p, CHAIKIN_ITERS) for p in polys if p.shape[0] >= 2]
    polys = [nan_to_num_nd(p) for p in polys]
    polys = [p for p in polys if finite_nd(p) and p.shape[0] >= 2]

    # 3) drop very tiny strokes
    min_len = MIN_STROKE_LEN_FRAC * diag
    polys = [p for p in polys if _poly_length(p) >= min_len]

    # 4) convert to cubic Béziers
    strokes = []
    for p in polys:
        if p.shape[0] < 2:
            continue
        beziers = _catmull_rom_to_beziers(p, alpha=0.5)
        if not beziers:
            # fallback: straight segments
            q = p.astype(float)
            segs = []
            for i in range(q.shape[0]-1):
                a = q[i]
                b = q[i+1]
                c1 = a + (b - a) / 3.0
                c2 = a + 2.0 * (b - a) / 3.0
                seg = [
                    float(a[0]), float(a[1]),
                    float(c1[0]), float(c1[1]),
                    float(c2[0]), float(c2[1]),
                    float(b[0]), float(b[1]),
                ]
                if finite_seq(seg):
                    segs.append(seg)
            beziers = segs
        if beziers:
            strokes.append({"segments": beziers})

    t1 = time.time()

    data = {
        "version": 2,
        "source_image": str(path),
        "width": W,
        "height": H,
        "vector_format": "bezier_cubic",
        "strokes": strokes,
        "stats": {
            "curves": len(strokes),
            "time_sec": round(t1 - t0, 3),
            "rdp_eps": round(rdp_eps, 4),
            "chaikin_iters": CHAIKIN_ITERS,
            "angle_break_deg": ANGLE_BREAK_DEG,
        },
    }

    out_json = OUT_DIR / f"{path.stem}.json"
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path), data

# ------------------- MAIN -------------------

def main():
    imgs = sorted(
        [
            p for p in IN_DIR.glob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        ],
        key=lambda p: p.name.lower(),
    )
    print(f"[INFO] IN={IN_DIR}  OUT={OUT_DIR}  found={len(imgs)} image(s)")
    if not imgs:
        return

    for p in imgs:
        src, meta = _process_single(p)
        print(f"[OK] {Path(src).name}: strokes={len(meta['strokes'])}, "
              f"time={meta['stats']['time_sec']}s")

if __name__ == "__main__":
    main()
