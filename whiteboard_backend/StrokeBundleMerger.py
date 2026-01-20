#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

#ACCURACY SETTINGS
#INTERSECT_RATIO_MAX = 0.8
#CUT_RATIO_MIN = 0.8
#RAISE DELETE RATIO

# ------------------- MERGE THRESHOLDS -------------------
INTERSECT_ABS_MAX = 20    # pixels
INTERSECT_RATIO_MAX = 0.25     # overlap / smaller_pixels

DELETE_RATIO = 0.45  # overlap / smaller_pixels
DELETE_REMAIN_MAX_PIX = 5  # allow tiny poke-out, still delete

CUT_RATIO_MIN = 0.20      # overlap / smaller_pixels

MIN_STROKE_PIXELS = 5

TILE_SIZE = 64


# ------------------- FUZZY OVERLAP SETTINGS -------------------
# IMPORTANT:
# We split tolerance into:
#  - MATCH: used to MEASURE overlap (detect duplicates even if offset by few px)
#  - CUT:   used when SUBTRACTING pixels (to avoid shaving fine details)
OVERLAP_TOL_MATCH = 4
OVERLAP_TOL_CUT   = 0

# bbox pad should cover the bigger of the two tolerances
BBOX_PAD = max(OVERLAP_TOL_MATCH, OVERLAP_TOL_CUT) + 2

_DILATE_KERNEL_MATCH = None
if OVERLAP_TOL_MATCH > 0:
    _DILATE_KERNEL_MATCH = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * OVERLAP_TOL_MATCH + 1, 2 * OVERLAP_TOL_MATCH + 1)
    )

_DILATE_KERNEL_CUT = None
if OVERLAP_TOL_CUT > 0:
    _DILATE_KERNEL_CUT = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * OVERLAP_TOL_CUT + 1, 2 * OVERLAP_TOL_CUT + 1)
    )


# ------------------- JOIN / ENDPOINT MERGE SETTINGS -------------------
# (kept from the previous drop-in)
JOIN_ENABLE = True

JOIN_MAX_PASSES = 6
JOIN_DIST_MAX = 8.0
JOIN_DIST_MIN = 0.0

JOIN_ANGLE_MAX = 75.0
JOIN_ANGLE_WEIGHT = 0.15

JOIN_TANGENT_K = 6
JOIN_GLOBAL_DIR_BLEND = 0.30

JOIN_CROSS_GROUP = False
JOIN_KEEP_DARKER_GROUP = True

JOIN_SNAP_ENDPOINTS = True
JOIN_SNAP_MAX_DIST = 3.0

JOIN_INSERT_BRIDGE_POINT = True
JOIN_BRIDGE_MAX_DIST = 8.0

JOIN_SKIP_IF_DUP_OVERLAP_RATIO = 0.75


# ------------------- COLOR ORDER -------------------
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

def color_group_id(name: str) -> int:
    r = color_rank(name)
    if 0 <= r < len(COLOR_ORDER_LIGHT_TO_DARK):
        return int(r + 1)  # 1..11
    return 0


# ------------------- SIMPLE TRACE FOR CUT RESULTS -------------------
NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

def _edge_key(p, q):
    (y1, x1) = p
    (y2, x2) = q
    if (y1, x1) <= (y2, x2):
        return (int(y1), int(x1), int(y2), int(x2))
    else:
        return (int(y2), int(x2), int(y1), int(x1))

def _build_degree_map(mask01: np.ndarray) -> np.ndarray:
    h, w = mask01.shape
    deg = np.zeros_like(mask01, dtype=np.uint8)
    ys, xs = np.where(mask01 > 0)
    for y, x in zip(ys, xs):
        c = 0
        for dy, dx in NEIGHBORS_8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask01[ny, nx] > 0:
                c += 1
        deg[y, x] = c
    return deg

def _neighbors_fg(y: int, x: int, mask01: np.ndarray):
    h, w = mask01.shape
    out = []
    for dy, dx in NEIGHBORS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and mask01[ny, nx] > 0:
            out.append((ny, nx))
    return out

def _trace_simple(mask01: np.ndarray) -> List[np.ndarray]:
    """
    Trace 1px mask into polylines. Used only after CUT.
    Returns list of Nx2 float32 [x,y].
    """
    sk = (mask01 > 0).astype(np.uint8)
    h, w = sk.shape
    if int(np.count_nonzero(sk)) < MIN_STROKE_PIXELS:
        return []

    deg = _build_degree_map(sk)
    visited_edges = set()
    polys = []

    ys, xs = np.where(sk > 0)

    def walk(start, first):
        sy, sx = start
        cy, cx = first
        poly = [[float(sx), float(sy)], [float(cx), float(cy)]]
        visited_edges.add(_edge_key(start, first))
        prev = start
        curr = first

        while True:
            y, x = curr
            nbrs = []
            for ny, nx in _neighbors_fg(y, x, sk):
                ek = _edge_key((y, x), (ny, nx))
                if ek in visited_edges:
                    continue
                if (ny, nx) == prev:
                    continue
                nbrs.append((ny, nx))

            if not nbrs:
                break

            nxt = nbrs[0]
            visited_edges.add(_edge_key(curr, nxt))
            poly.append([float(nxt[1]), float(nxt[0])])
            prev = curr
            curr = nxt

            if int(deg[curr[0], curr[1]]) != 2:
                break

        arr = np.asarray(poly, dtype=np.float32)
        if arr.shape[0] >= 2:
            polys.append(arr)

    # pass 1: from non-degree-2
    for y, x in zip(ys, xs):
        d = int(deg[y, x])
        if d == 0 or d == 2:
            continue
        for ny, nx in _neighbors_fg(y, x, sk):
            ek = _edge_key((y, x), (ny, nx))
            if ek in visited_edges:
                continue
            walk((y, x), (ny, nx))

    # pass 2: loops
    for y, x in zip(ys, xs):
        if int(deg[y, x]) != 2:
            continue
        for ny, nx in _neighbors_fg(y, x, sk):
            ek = _edge_key((y, x), (ny, nx))
            if ek in visited_edges:
                continue
            walk((y, x), (ny, nx))

    return polys


# ------------------- GEOM HELPERS -------------------

def _unit_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)

def _angle_between_vecs(u: np.ndarray, v: np.ndarray) -> float:
    cu = _unit_vec(u)
    cv = _unit_vec(v)
    c = float(np.dot(cu, cv))
    c = max(-1.0, min(1.0, c))
    return float(math.degrees(math.acos(c)))

def _endpoint_direction(poly: np.ndarray, at_start: bool) -> np.ndarray:
    if poly is None or poly.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=np.float32)

    pts = poly.astype(np.float32, copy=False)
    n = int(pts.shape[0])
    k = max(1, min(int(JOIN_TANGENT_K), n - 1))

    if at_start:
        local = pts[k] - pts[0]
    else:
        local = pts[-1] - pts[-1 - k]

    global_dir = pts[-1] - pts[0]

    a = float(JOIN_GLOBAL_DIR_BLEND)
    d = (1.0 - a) * _unit_vec(local) + a * _unit_vec(global_dir)
    return _unit_vec(d)

def _poly_clean_consecutive_duplicates(poly: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    if poly is None or poly.shape[0] < 2:
        return poly
    pts = poly.astype(np.float32, copy=False)
    out = [pts[0]]
    for i in range(1, pts.shape[0]):
        if float(np.linalg.norm(pts[i] - out[-1])) > eps:
            out.append(pts[i])
    if len(out) < 2:
        return pts
    return np.asarray(out, dtype=np.float32)


# ------------------- ROI / OVERLAP HELPERS -------------------

def _poly_bbox(poly: np.ndarray, W: int, H: int) -> Tuple[int,int,int,int]:
    xs = poly[:, 0]
    ys = poly[:, 1]
    x0 = max(0, int(math.floor(float(xs.min()))) - BBOX_PAD)
    y0 = max(0, int(math.floor(float(ys.min()))) - BBOX_PAD)
    x1 = min(W - 1, int(math.ceil(float(xs.max()))) + BBOX_PAD)
    y1 = min(H - 1, int(math.ceil(float(ys.max()))) + BBOX_PAD)
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    return (x0, y0, x1, y1)

def _poly_to_roi_mask(poly: np.ndarray, bbox: Tuple[int,int,int,int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    rw = max(1, x1 - x0 + 1)
    rh = max(1, y1 - y0 + 1)
    roi = np.zeros((rh, rw), dtype=np.uint8)

    if poly.shape[0] < 2:
        return roi

    pts = np.round(poly).astype(np.int32)
    pts[:, 0] -= x0
    pts[:, 1] -= y0
    pts[:, 0] = np.clip(pts[:, 0], 0, rw - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, rh - 1)

    cv2.polylines(roi, [pts.reshape(-1, 1, 2)], isClosed=False, color=1, thickness=1)
    return roi

def _fuzzy_overlap_small_to_big(s_bbox, s_roi, b_bbox, b_roi_dil) -> int:
    """
    Count how many pixels of SMALL are inside BIG (typically dilated).
    """
    sx0, sy0, sx1, sy1 = s_bbox
    bx0, by0, bx1, by1 = b_bbox

    ix0 = max(sx0, bx0)
    iy0 = max(sy0, by0)
    ix1 = min(sx1, bx1)
    iy1 = min(sy1, by1)
    if ix1 < ix0 or iy1 < iy0:
        return 0

    s_x0 = ix0 - sx0
    s_y0 = iy0 - sy0
    s_x1 = ix1 - sx0
    s_y1 = iy1 - sy0

    b_x0 = ix0 - bx0
    b_y0 = iy0 - by0
    b_x1 = ix1 - bx0
    b_y1 = iy1 - by0

    sub_s = s_roi[s_y0:s_y1+1, s_x0:s_x1+1]
    sub_b = b_roi_dil[b_y0:b_y1+1, b_x0:b_x1+1]
    if sub_s.size == 0 or sub_b.size == 0:
        return 0

    return int(np.count_nonzero((sub_s > 0) & (sub_b > 0)))

def _subtract_roi(small_bbox, small_roi, big_bbox, big_roi_dil_for_cut) -> np.ndarray:
    """
    Subtract BIG (cut mask) from SMALL.
    IMPORTANT: this uses CUT dilation (or none), NOT match dilation.
    """
    sm = small_roi.copy()

    sx0, sy0, sx1, sy1 = small_bbox
    bx0, by0, bx1, by1 = big_bbox

    ix0 = max(sx0, bx0)
    iy0 = max(sy0, by0)
    ix1 = min(sx1, bx1)
    iy1 = min(sy1, by1)
    if ix1 < ix0 or iy1 < iy0:
        return sm

    s_x0 = ix0 - sx0
    s_y0 = iy0 - sy0
    s_x1 = ix1 - sx0
    s_y1 = iy1 - sy0

    b_x0 = ix0 - bx0
    b_y0 = iy0 - by0
    b_x1 = ix1 - bx0
    b_y1 = iy1 - by0

    subb = big_roi_dil_for_cut[b_y0:b_y1+1, b_x0:b_x1+1]
    sub = sm[s_y0:s_y1+1, s_x0:s_x1+1]
    sub[subb > 0] = 0
    sm[s_y0:s_y1+1, s_x0:s_x1+1] = sub
    return sm

def _roi_to_polys(roi01: np.ndarray, bbox: Tuple[int,int,int,int]) -> List[np.ndarray]:
    x0, y0, _, _ = bbox
    polys = _trace_simple(roi01)
    out = []
    for p in polys:
        if p.shape[0] < 2:
            continue
        q = p.copy()
        q[:, 0] += float(x0)
        q[:, 1] += float(y0)
        out.append(q)
    return out


# ------------------- TILE INDEX -------------------

class _TileIndex:
    def __init__(self, tile_size: int):
        self.ts = int(tile_size)
        self.map: Dict[Tuple[int,int], set] = {}

    def _tiles_for_bbox(self, bbox: Tuple[int,int,int,int]) -> List[Tuple[int,int]]:
        x0, y0, x1, y1 = bbox
        tx0 = x0 // self.ts
        ty0 = y0 // self.ts
        tx1 = x1 // self.ts
        ty1 = y1 // self.ts
        out = []
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                out.append((tx, ty))
        return out

    def add(self, sid: int, bbox: Tuple[int,int,int,int]):
        for t in self._tiles_for_bbox(bbox):
            s = self.map.get(t)
            if s is None:
                s = set()
                self.map[t] = s
            s.add(sid)

    def remove(self, sid: int, bbox: Tuple[int,int,int,int]):
        for t in self._tiles_for_bbox(bbox):
            s = self.map.get(t)
            if s is None:
                continue
            s.discard(sid)
            if not s:
                self.map.pop(t, None)

    def query(self, bbox: Tuple[int,int,int,int]) -> List[int]:
        out = set()
        for t in self._tiles_for_bbox(bbox):
            s = self.map.get(t)
            if s:
                out |= s
        return list(out)


# ------------------- RECORD -------------------

@dataclass
class PolyRec:
    poly: np.ndarray
    rank: int
    group_id: int
    bbox: Tuple[int,int,int,int]
    roi: np.ndarray
    roi_dil_match: np.ndarray
    roi_match: np.ndarray
    roi_dil_cut: np.ndarray
    pix: int
    p0: np.ndarray
    p1: np.ndarray
    t0: np.ndarray
    t1: np.ndarray


def _build_poly_rec(poly: np.ndarray, rank: int, group_id: int, W: int, H: int) -> Optional[PolyRec]:
    if poly is None or poly.shape[0] < 2:
        return None

    poly = poly.astype(np.float32, copy=False)
    poly = _poly_clean_consecutive_duplicates(poly)

    if poly is None or poly.shape[0] < 2:
        return None

    bbox = _poly_bbox(poly, W, H)
    roi = _poly_to_roi_mask(poly, bbox)
    roi_match = roi
    pix = int(np.count_nonzero(roi))
    if pix < MIN_STROKE_PIXELS:
        return None

    # dilated mask for MATCH
    if OVERLAP_TOL_MATCH > 0 and _DILATE_KERNEL_MATCH is not None:
        roi_dil_match = cv2.dilate(roi_match, _DILATE_KERNEL_MATCH, iterations=1)
    else:
        roi_dil_match = roi_match

    # dilated mask for CUT
    if OVERLAP_TOL_CUT > 0 and _DILATE_KERNEL_CUT is not None:
        roi_dil_cut = cv2.dilate(roi, _DILATE_KERNEL_CUT, iterations=1)
    else:
        roi_dil_cut = roi

    p0 = poly[0].copy()
    p1 = poly[-1].copy()
    t0 = _endpoint_direction(poly, at_start=True)
    t1 = _endpoint_direction(poly, at_start=False)

    return PolyRec(
        poly=poly,
        rank=int(rank),
        group_id=int(group_id),
        bbox=bbox,
        roi=roi,
        roi_match = roi_match,
        roi_dil_match=roi_dil_match,
        roi_dil_cut=roi_dil_cut,
        pix=pix,
        p0=p0,
        p1=p1,
        t0=t0,
        t1=t1
    )


# ------------------- JOIN LOGIC -------------------

def _concat_polys_for_join(a: np.ndarray, b: np.ndarray, rev_a: bool, rev_b: bool) -> np.ndarray:
    A = a[::-1].copy() if rev_a else a.copy()
    B = b[::-1].copy() if rev_b else b.copy()

    if A.shape[0] < 2 or B.shape[0] < 2:
        return None

    pa = A[-1].astype(np.float32)
    pb = B[0].astype(np.float32)
    dist = float(np.linalg.norm(pb - pa))

    if JOIN_SNAP_ENDPOINTS and dist <= float(JOIN_SNAP_MAX_DIST):
        mid = (pa + pb) * 0.5
        A[-1] = mid
        B[0] = mid
        pa = A[-1]
        pb = B[0]
        dist = float(np.linalg.norm(pb - pa))

    parts = [A]

    if JOIN_INSERT_BRIDGE_POINT and dist > 1e-3 and dist <= float(JOIN_BRIDGE_MAX_DIST):
        mid = (pa + pb) * 0.5
        parts.append(mid.reshape(1, 2))

    if float(np.linalg.norm(B[0] - parts[-1][-1])) <= 1e-3:
        parts.append(B[1:])
    else:
        parts.append(B)

    merged = np.vstack(parts).astype(np.float32, copy=False)
    merged = _poly_clean_consecutive_duplicates(merged)
    return merged


def _resolve_join_rank_gid(a: PolyRec, b: PolyRec) -> Tuple[int, int]:
    if a.group_id == b.group_id and a.rank == b.rank:
        return int(a.rank), int(a.group_id)

    if not JOIN_CROSS_GROUP:
        return int(a.rank), int(a.group_id)

    if JOIN_KEEP_DARKER_GROUP:
        if b.rank > a.rank:
            return int(b.rank), int(b.group_id)
        return int(a.rank), int(a.group_id)

    return int(a.rank), int(a.group_id)

def _apply_match_guard(piece: PolyRec, blocker: Optional[PolyRec]):
    """
    Fake CUT margin: stop `piece` from matching `blocker` again,
    WITHOUT changing the visual stroke.
    We only modify piece.roi_match (used by overlap logic).
    """
    if piece is None or blocker is None:
        return

    # Remove pixels from match-mask that fall inside blocker MATCH dilation
    guarded = _subtract_roi(
        piece.bbox, piece.roi_match,
        blocker.bbox, blocker.roi_dil_match
    )

    if int(np.count_nonzero(guarded)) < MIN_STROKE_PIXELS:
        # If it nukes everything, just keep original match mask
        return

    piece.roi_match = guarded

    # Rebuild MATCH dilation from the guarded match mask
    if OVERLAP_TOL_MATCH > 0 and _DILATE_KERNEL_MATCH is not None:
        piece.roi_dil_match = cv2.dilate(piece.roi_match, _DILATE_KERNEL_MATCH, iterations=1)
    else:
        piece.roi_dil_match = piece.roi_match




def _find_best_join_candidate(active: Dict[int, PolyRec], order: List[int]) -> Optional[Tuple[int,int,bool,bool,float,float]]:
    if not JOIN_ENABLE or len(order) < 2:
        return None

    cell = int(max(4.0, float(JOIN_DIST_MAX) * 2.0))

    def cell_for_pt(p: np.ndarray) -> Tuple[int, int]:
        return (int(p[0] // cell), int(p[1] // cell))

    grid: Dict[Tuple[int,int], List[int]] = {}
    for sid in order:
        rec = active.get(sid)
        if rec is None:
            continue
        for pt in (rec.p0, rec.p1):
            key = cell_for_pt(pt)
            grid.setdefault(key, []).append(sid)

    pos = {sid: i for i, sid in enumerate(order)}

    best = None
    best_score = None

    dist_max = float(JOIN_DIST_MAX)
    dist_min = float(JOIN_DIST_MIN)
    angle_max = float(JOIN_ANGLE_MAX)

    cfgs = [
        (False, False),
        (False, True),
        (True,  False),
        (True,  True),
    ]

    def end_dir_after_rev(rec: PolyRec, rev: bool) -> np.ndarray:
        return (-rec.t0) if rev else rec.t1

    def start_dir_after_rev(rec: PolyRec, rev: bool) -> np.ndarray:
        return (-rec.t1) if rev else rec.t0

    for keep_id in order:
        a = active.get(keep_id)
        if a is None:
            continue

        cand_ids = set()
        for pt in (a.p0, a.p1):
            tx, ty = cell_for_pt(pt)
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    for sid in grid.get((tx + ox, ty + oy), []):
                        if sid != keep_id:
                            cand_ids.add(sid)

        if not cand_ids:
            continue

        for drop_id in cand_ids:
            b = active.get(drop_id)
            if b is None:
                continue

            if pos.get(keep_id, 0) > pos.get(drop_id, 0):
                continue

            if (not JOIN_CROSS_GROUP) and (a.group_id != b.group_id):
                continue

            # duplication check uses MATCH dilation
            small = a if a.pix <= b.pix else b
            big = b if small is a else a
            ov = _fuzzy_overlap_small_to_big(small.bbox, small.roi_match, big.bbox, big.roi_dil_match)
            dup_ratio = float(ov) / float(max(1, small.pix))
            if dup_ratio >= float(JOIN_SKIP_IF_DUP_OVERLAP_RATIO):
                continue

            best_local = None
            best_local_score = None

            for rev_a, rev_b in cfgs:
                pa = a.p0 if rev_a else a.p1
                pb = b.p1 if rev_b else b.p0

                d = float(np.linalg.norm(pb - pa))
                if d < dist_min or d > dist_max:
                    continue

                ta = end_dir_after_rev(a, rev_a)
                tb = start_dir_after_rev(b, rev_b)
                ang = _angle_between_vecs(ta, tb)
                if ang > angle_max:
                    continue

                score = float(d + JOIN_ANGLE_WEIGHT * ang)

                if best_local_score is None or score < best_local_score:
                    best_local_score = score
                    best_local = (rev_a, rev_b, d, ang)

            if best_local is None:
                continue

            rev_a, rev_b, d, ang = best_local
            if best_score is None or best_local_score < best_score:
                best_score = best_local_score
                best = (keep_id, drop_id, rev_a, rev_b, d, ang)

    return best


def _run_endpoint_join_pass(active: Dict[int, PolyRec], order: List[int], idx: _TileIndex, W: int, H: int):
    if not JOIN_ENABLE:
        return

    for _ in range(int(JOIN_MAX_PASSES)):
        merged_any = False

        while True:
            cand = _find_best_join_candidate(active, order)
            if cand is None:
                break

            keep_id, drop_id, rev_keep, rev_drop, dist, ang = cand
            a = active.get(keep_id)
            b = active.get(drop_id)
            if a is None or b is None:
                break

            merged_poly = _concat_polys_for_join(a.poly, b.poly, rev_keep, rev_drop)
            if merged_poly is None or merged_poly.shape[0] < 2:
                break

            new_rank, new_gid = _resolve_join_rank_gid(a, b)
            new_rec = _build_poly_rec(merged_poly, new_rank, new_gid, W, H)
            if new_rec is None:
                break

            idx.remove(keep_id, a.bbox)
            idx.remove(drop_id, b.bbox)

            active[keep_id] = new_rec
            idx.add(keep_id, new_rec.bbox)

            active.pop(drop_id, None)
            try:
                order.remove(drop_id)
            except ValueError:
                pass

            merged_any = True

        if not merged_any:
            break


# ------------------- PUBLIC API -------------------

def merge_polyline_collections(
    collections: List[Tuple[str, List[np.ndarray]]],
    width: int,
    height: int,
) -> List[Tuple[np.ndarray, int]]:
    """
    Input:  list of (color_name, polylines)
    Output: merged polylines + their originating color_group_id (1..11)
    """
    W = int(width)
    H = int(height)

    # Flatten and order by color rank (light->dark)
    pending: List[Tuple[int, int, np.ndarray]] = []
    for cname, polys in collections:
        r = color_rank(cname)
        gid = color_group_id(cname)
        for p in polys or []:
            pending.append((r, gid, p))
    pending.sort(key=lambda x: x[0])

    active: Dict[int, PolyRec] = {}
    order: List[int] = []
    idx = _TileIndex(TILE_SIZE)
    next_id = 1

    def add_rec(rec: PolyRec, insert_at: Optional[int] = None) -> int:
        nonlocal next_id
        sid = next_id
        next_id += 1
        active[sid] = rec
        idx.add(sid, rec.bbox)
        if insert_at is None:
            order.append(sid)
        else:
            order.insert(insert_at, sid)
        return sid

    def remove_id(sid: int):
        rec = active.get(sid)
        if rec is None:
            return
        idx.remove(sid, rec.bbox)
        active.pop(sid, None)

    def replace_id_with_polys(
    sid: int,
    polys: List[np.ndarray],
    rank: int,
    group_id: int,
    match_guard_blocker: Optional[PolyRec] = None
    ):
        try:
            pos = order.index(sid)
        except ValueError:
            pos = None

        old = active.get(sid)
        if old is None:
            return

        remove_id(sid)

        new_recs: List[PolyRec] = []
        for p in polys:
            pr = _build_poly_rec(p, rank, group_id, W, H)
            if pr is None:
                continue
            _apply_match_guard(pr, match_guard_blocker)
            new_recs.append(pr)

        if not new_recs:
            return

        if pos is None:
            for pr in new_recs:
                add_rec(pr)
            return

        order.pop(pos)
        insert_pos = pos
        for pr in new_recs:
            add_rec(pr, insert_at=insert_pos)
            insert_pos += 1


    def best_overlap_candidate(curr: PolyRec) -> Tuple[int, int, float]:
        """
        Pick candidate with best "duplicate-likeness" (MATCH dilation):
        maximize overlap ratio relative to the smaller stroke.
        """
        cands = idx.query(curr.bbox)
        best_id = 0
        best_ov = 0
        best_ratio = 0.0

        for sid in cands:
            other = active.get(sid)
            if other is None:
                continue

            if curr.pix <= other.pix:
                ov = _fuzzy_overlap_small_to_big(curr.bbox, curr.roi_match, other.bbox, other.roi_dil_match)
                ratio = float(ov) / float(max(1, curr.pix))
            else:
                ov = _fuzzy_overlap_small_to_big(other.bbox, other.roi_match, curr.bbox, curr.roi_dil_match)
                ratio = float(ov) / float(max(1, other.pix))

            if (ratio > best_ratio) or (ratio == best_ratio and ov > best_ov):
                best_ratio = ratio
                best_ov = ov
                best_id = sid

        return best_id, best_ov, best_ratio

    for r, gid, poly in pending:
        rec = _build_poly_rec(poly, r, gid, W, H)
        if rec is None:
            continue

        while True:
            sid, ov_guess, ratio_guess = best_overlap_candidate(rec)
            if sid == 0 or ov_guess <= 0:
                break

            other = active.get(sid)
            if other is None:
                break

            if rec.pix <= other.pix:
                small_is_new = True
                small = rec
                big = other
            else:
                small_is_new = False
                small = other
                big = rec

            small_pix = int(small.pix)
            if small_pix <= 0:
                break

            # recompute overlap using MATCH dilation (duplicate detection / ratios)
            ov = _fuzzy_overlap_small_to_big(small.bbox, small.roi_match, big.bbox, big.roi_dil_match)
            if ov <= 0:
                break

            ratio = float(ov) / float(small_pix + 10)  #Buff up small strokes so they dont get sucked in and ruined

            # Case 1: tiny overlap -> treat as intersection, do nothing
            if (ov <= INTERSECT_ABS_MAX) and (ratio <= INTERSECT_RATIO_MAX):
                break

            # Case 2: mostly contained -> delete smaller (still based on MATCH overlap)
            if (ratio >= DELETE_RATIO) or ((small_pix - ov) <= DELETE_REMAIN_MAX_PIX):
                if small_is_new:
                    rec = None
                    break
                else:
                    remove_id(sid)
                    continue

            # Case 3: partial overlap -> cut smaller (IMPORTANT: CUT uses CUT dilation)
            # Case 3: partial overlap -> cut smaller
            if ratio >= CUT_RATIO_MIN:

                # IMPORTANT:
                # Only CUT when there is REAL pixel overlap (not just fuzzy tolerance).
                # Otherwise CUT does nothing, but fuzzy overlap keeps re-triggering forever.
                ov_real = _fuzzy_overlap_small_to_big(small.bbox, small.roi, big.bbox, big.roi)
                if ov_real <= 0:
                    break

                if small_is_new:
                    # VISUAL CUT = hairline: subtract BIG raw roi (no dilation) from SMALL raw roi
                    new_roi = _subtract_roi(rec.bbox, rec.roi, other.bbox, other.roi)

                    remain_pix = int(np.count_nonzero(new_roi))
                    if remain_pix < MIN_STROKE_PIXELS:
                        rec = None
                        break

                    pieces = _roi_to_polys(new_roi, rec.bbox)
                    rec = None

                    # Fake CUT tolerance: prevent re-matching against the blocker,
                    # without changing visuals.
                    for p in pieces:
                        pr = _build_poly_rec(p, r, gid, W, H)
                        if pr is not None:
                            _apply_match_guard(pr, other)  # blocker is the existing stroke
                            add_rec(pr)

                    break

                else:
                    # Existing stroke is smaller -> cut that existing stroke against the new stroke (rec)
                    new_roi = _subtract_roi(other.bbox, other.roi, rec.bbox, rec.roi)

                    remain_pix = int(np.count_nonzero(new_roi))
                    if remain_pix < MIN_STROKE_PIXELS:
                        remove_id(sid)
                        continue

                    pieces = _roi_to_polys(new_roi, other.bbox)
                    if not pieces:
                        remove_id(sid)
                        continue

                    # Replace old stroke with pieces + apply match guard against the new blocker (rec)
                    replace_id_with_polys(
                        sid,
                        pieces,
                        other.rank,
                        other.group_id,
                        match_guard_blocker=rec
                    )
                    continue
            break

        if rec is not None:
            add_rec(rec)

    # Endpoint join pass (optional)
    _run_endpoint_join_pass(active, order, idx, W, H)

    out: List[Tuple[np.ndarray, int]] = []
    for sid in order:
        rr = active.get(sid)
        if rr is None:
            continue
        out.append((rr.poly, int(rr.group_id)))

    return out