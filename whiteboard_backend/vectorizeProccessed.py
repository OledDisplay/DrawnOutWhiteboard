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
IN_DIR = BASE / "ProccessedImages"
OUT_DIR = BASE / "StrokeVectors"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------- GLOBAL KNOBS -------------------
MIN_COMPONENT_AREA = 9         # remove tiny specks
BLUR_BEFORE_BIN = 0            # 0 off, else odd (e.g., 3)

USE_SKIMAGE = True             # skeletonize speed

RDP_EPS_FRAC = 0.003           # simplification strength
CHAIKIN_ITERS = 1              # 0..2

LINK_DIST_FRAC = 0.004         # endpoint gap (fraction of diagonal)
LINK_ANG_DEG   = 25.0          # tangent mismatch
MAX_LINK_PASSES = 6

MIN_STROKE_LEN_FRAC = 0.01     # drop short strokes (< this * diag)

CATMULL_ALPHA = 0.5            # centripetal Catmull–Rom
SEG_STRIDE = 1                 # subsample before fitting

# ------------------- FINITE CHECKS -------------------

def finite2(x: float) -> bool:
    return math.isfinite(x)

def finite_seq(seq) -> bool:
    return all(math.isfinite(float(v)) for v in seq)

def finite_nd(a: np.ndarray) -> bool:
    return np.isfinite(a).all()

def nan_to_num_nd(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, copy=False)

# ------------------- IO / BINARIZE -------------------

def _load_edges_binary(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))

    if BLUR_BEFORE_BIN and BLUR_BEFORE_BIN % 2 == 1:
        img = cv2.GaussianBlur(img, (BLUR_BEFORE_BIN, BLUR_BEFORE_BIN), 0)

    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = (bw == 255).mean()
    fg = (bw == 0).astype(np.uint8) if white_ratio > 0.5 else (bw == 255).astype(np.uint8)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = np.zeros_like(fg)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            out[lab == i] = 1
    return out

# ------------------- SKELETON -------------------

def _skeletonize(mask01: np.ndarray) -> np.ndarray:
    if USE_SKIMAGE:
        try:
            from skimage.morphology import skeletonize
            sk = skeletonize(mask01 > 0).astype(np.uint8)
            return sk
        except Exception:
            pass
    # Zhang–Suen fallback
    img = (mask01 > 0).astype(np.uint8).copy()
    h, w = img.shape

    def nbrs(y, x):
        return [img[y-1, x], img[y-1, x+1], img[y, x+1], img[y+1, x+1],
                img[y+1, x], img[y+1, x-1], img[y, x-1], img[y-1, x-1]]

    def trans(seq):
        c = 0
        for i in range(8):
            if seq[i] == 0 and seq[(i+1) % 8] == 1:
                c += 1
        return c

    changed = True
    while changed:
        changed = False
        rem = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                if img[y, x] != 1: continue
                p = nbrs(y, x); bp = sum(p)
                if bp < 2 or bp > 6: continue
                if trans(p) != 1: continue
                if p[0]*p[2]*p[4] != 0: continue
                if p[2]*p[4]*p[6] != 0: continue
                rem.append((y, x))
        for y, x in rem: img[y, x] = 0
        changed = changed or bool(rem)

        rem = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                if img[y, x] != 1: continue
                p = nbrs(y, x); bp = sum(p)
                if bp < 2 or bp > 6: continue
                if trans(p) != 1: continue
                if p[0]*p[2]*p[6] != 0: continue
                if p[0]*p[4]*p[6] != 0: continue
                rem.append((y, x))
        for y, x in rem: img[y, x] = 0
        changed = changed or bool(rem)
    return img.astype(np.uint8)

# ------------------- POLYLINES -------------------

def _trace_polylines_from_skeleton(skel01: np.ndarray) -> List[np.ndarray]:
    sk255 = (skel01 * 255).astype(np.uint8)
    contours, _ = cv2.findContours(sk255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in contours:
        if c is None or len(c) < 3:
            continue
        p = c.reshape(-1, 2).astype(np.float32)
        nan_to_num_nd(p)
        if p.shape[0] >= 2 and finite_nd(p):
            polys.append(p)
    return polys

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
        for i in range(P.shape[0]-1):
            p = P[i]; r = P[i+1]
            q = 0.75 * p + 0.25 * r
            s = 0.25 * p + 0.75 * r
            Q.extend([q, s])
        Q.append(P[-1])
        P = np.asarray(Q, dtype=np.float32)
        nan_to_num_nd(P)
    return P

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)

def _end_tangent(poly: np.ndarray, at_start: bool) -> np.ndarray:
    if poly.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=np.float32)
    if at_start:
        a, b = poly[0], poly[min(1, poly.shape[0]-1)]
    else:
        a, b = poly[-1], poly[max(poly.shape[0]-2, 0)]
    t = _unit((b - a).astype(np.float32))
    nan_to_num_nd(t)
    return t

def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    cu = _unit(u); cv = _unit(v)
    c = float(np.dot(cu, cv))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))

def _link_fragments(polys: List[np.ndarray], diag: float, max_passes: int) -> List[np.ndarray]:
    if not polys:
        return polys

    max_dist = LINK_DIST_FRAC * diag
    max_ang = LINK_ANG_DEG

    work = [p.astype(np.float32) for p in polys]
    for p in work:
        nan_to_num_nd(p)

    changed = True
    passes = 0

    while changed and passes < max_passes:
        passes += 1
        changed = False

        endpoints = []
        for idx, p in enumerate(work):
            if p.shape[0] == 0: continue
            s, e = p[0], p[-1]
            endpoints.append((idx, True,  s.copy()))
            endpoints.append((idx, False, e.copy()))

        used = set()
        for i in range(len(endpoints)):
            if i in used: continue
            idx_i, is_start_i, pt_i = endpoints[i]
            tan_i = _end_tangent(work[idx_i], at_start=is_start_i)

            best = (-1, 1e18, 1e9)
            for j in range(len(endpoints)):
                if j == i or j in used: continue
                idx_j, is_start_j, pt_j = endpoints[j]
                if idx_i == idx_j: continue
                d = float(np.linalg.norm(pt_j - pt_i))
                if not math.isfinite(d) or d > max_dist: continue

                tan_j = _end_tangent(work[idx_j], at_start=is_start_j)
                tj = -tan_j if is_start_j else tan_j
                ti = tan_i if is_start_i else -tan_i

                ang = _angle_between(ti, tj)
                if not math.isfinite(ang) or ang > max_ang: continue

                if d < best[1] or (abs(d - best[1]) < 1e-6 and ang < best[2]):
                    best = (j, d, ang)

            j = best[0]
            if j == -1:
                continue

            used.add(i); used.add(j)
            idx_j, is_start_j, _ = endpoints[j]

            A = work[idx_i]
            B = work[idx_j]

            if is_start_i and is_start_j:
                A = A[::-1]
                merged = np.vstack((A, B))
            elif is_start_i and not is_start_j:
                merged = np.vstack((B, A))
            elif not is_start_i and is_start_j:
                merged = np.vstack((A, B))
            else:
                B = B[::-1]
                merged = np.vstack((A, B))

            nan_to_num_nd(merged)
            if not finite_nd(merged) or merged.shape[0] < 2:
                continue

            work[idx_i] = merged
            work[idx_j] = np.zeros((0, 2), dtype=np.float32)
            changed = True

        work = [p for p in work if p.shape[0] > 0]

    return work

# ------------------- CURVES (Catmull–Rom → Bézier) -------------------

def _catmull_rom_to_beziers(pts: np.ndarray, alpha=0.5) -> List[List[float]]:
    """
    Convert open polyline to list of cubic Beziers:
    [x0,y0, cx1,cy1, cx2,cy2, x1,y1]
    All numbers guaranteed finite.
    """
    P = pts.astype(np.float32)
    nan_to_num_nd(P)
    n = P.shape[0]
    if n < 2:
        return []

    def tj(ti, Pi, Pj):
        d = float(np.linalg.norm(Pj - Pi))
        if not math.isfinite(d): d = 0.0
        return ti + (d ** alpha)

    beziers: List[List[float]] = []

    P_ext = np.vstack((P[0], P, P[-1]))
    t = [0.0]
    for i in range(1, P_ext.shape[0]):
        t.append(tj(t[-1], P_ext[i-1], P_ext[i]))

    for k in range(1, n+0):
        i1 = k
        if i1 >= n: break

        Pm = P_ext[k-1].astype(np.float32)
        P0 = P_ext[k].astype(np.float32)
        P1 = P_ext[k+1].astype(np.float32)
        Pp = P_ext[k+2].astype(np.float32)

        tm, t0, t1, tp = t[k-1], t[k], t[k+1], t[k+2]
        dt0 = (t1 - tm); dt1 = (tp - t0)
        if abs(dt0) < 1e-12 or abs(dt1) < 1e-12:
            # fallback straight tangent
            m0 = (P1 - P0)
            m1 = (P1 - P0)
        else:
            m0 = ((P1 - Pm) / (t1 - tm) - (P0 - Pm) / (t0 - tm)) * (t1 - t0)
            m1 = ((Pp - P0) / (tp - t0) - (P1 - P0) / (t1 - t0)) * (t1 - t0)

        c1 = P0 + m0 / 3.0
        c2 = P1 - m1 / 3.0

        seg = [float(P0[0]), float(P0[1]),
               float(c1[0]), float(c1[1]),
               float(c2[0]), float(c2[1]),
               float(P1[0]), float(P1[1])]
        if finite_seq(seg):
            beziers.append(seg)

    return beziers

# ------------------- PROCESS SINGLE -------------------

def _process_single(path: Path) -> Tuple[str, dict]:
    fg = _load_edges_binary(path)
    H, W = fg.shape
    diag = math.hypot(W, H)

    t0 = time.time()

    sk = _skeletonize(fg)

    polys = _trace_polylines_from_skeleton(sk)

    rdp_eps = RDP_EPS_FRAC * diag
    polys = [ _rdp(p, rdp_eps) for p in polys if p.shape[0] >= 2 ]
    polys = [ _chaikin(p, CHAIKIN_ITERS) for p in polys if p.shape[0] >= 2 ]
    # --- FIXED: sanitize arrays without boolean 'or' on ndarray ---
    polys = [ nan_to_num_nd(p) for p in polys ]
    polys = [ p for p in polys if finite_nd(p) and p.shape[0] >= 2 ]

    polys = _link_fragments(polys, diag, max_passes=MAX_LINK_PASSES)
    polys = [ p for p in polys if finite_nd(p) and p.shape[0] >= 2 ]

    # remove tiny strokes
    min_len = MIN_STROKE_LEN_FRAC * diag
    def plen(pp: np.ndarray) -> float:
        if pp.shape[0] < 2: return 0.0
        d = np.linalg.norm(np.diff(pp, axis=0), axis=1).sum()
        if not math.isfinite(d): return 0.0
        return float(d)
    polys = [ p for p in polys if plen(p) >= min_len ]

    if SEG_STRIDE > 1:
        polys = [ p[::SEG_STRIDE] if p.shape[0] > SEG_STRIDE else p for p in polys ]

    strokes = []
    for p in polys:
        if p.shape[0] < 2:
            continue
        beziers = _catmull_rom_to_beziers(p, alpha=CATMULL_ALPHA)
        if not beziers:
            # fallback straight pieces with safe controls
            q = p.astype(float)
            segs = []
            for i in range(q.shape[0]-1):
                a = q[i]; b = q[i+1]
                c1 = a + (b - a) / 3.0
                c2 = a + 2.0 * (b - a) / 3.0
                seg = [float(a[0]), float(a[1]),
                       float(c1[0]), float(c1[1]),
                       float(c2[0]), float(c2[1]),
                       float(b[0]), float(b[1])]
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
            "linked_passes": MAX_LINK_PASSES,
            "chaikin_iters": CHAIKIN_ITERS,
        }
    }
    out_json = OUT_DIR / f"{path.stem}.json"
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path), data

# ------------------- MAIN -------------------

def main():
    imgs = sorted([p for p in IN_DIR.glob("*")
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")])
    print(f"[INFO] IN={IN_DIR}  OUT={OUT_DIR}  found={len(imgs)} image(s)")
    if not imgs:
        return
    for p in imgs:
        src, meta = _process_single(p)
        print(f"[OK] {Path(src).name}: strokes={len(meta['strokes'])}, time={meta['stats']['time_sec']}s")

if __name__ == "__main__":
    main()
