#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


# ------------------- MERGE THRESHOLDS -------------------
INTERSECT_ABS_MAX = 10          # pixels
INTERSECT_RATIO_MAX = 0.8      # overlap / smaller_pixels

DELETE_RATIO = 0.22            # overlap / smaller_pixels
DELETE_REMAIN_MAX_PIX = 5       # allow tiny poke-out, still delete

CUT_RATIO_MIN = 0.8            # overlap / smaller_pixels

MIN_STROKE_PIXELS = 3

TILE_SIZE = 64


# ------------------- FUZZY OVERLAP SETTINGS -------------------
# If two strokes are "the same line" but offset by 1px, raw overlap is ~0.
# We fix this by counting small pixels that fall within OVERLAP_TOL of big.
OVERLAP_TOL = 3  # try 1 first; if still weak, try 2
BBOX_PAD = OVERLAP_TOL + 2

_DILATE_KERNEL = None
if OVERLAP_TOL > 0:
    _DILATE_KERNEL = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * OVERLAP_TOL + 1, 2 * OVERLAP_TOL + 1)
    )


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
    Count how many pixels of SMALL are within OVERLAP_TOL of BIG.
    BIG is pre-dilated (b_roi_dil).
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

def _subtract_roi(small_bbox, small_roi, big_bbox, big_roi_dil) -> np.ndarray:
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

    subb = big_roi_dil[b_y0:b_y1+1, b_x0:b_x1+1]
    sub = sm[s_y0:s_y1+1, s_x0:s_x1+1]
    sub[subb > 0] = 0
    sm[s_y0:s_y1+1, s_x0:s_x1+1] = sub
    return sm

def _roi_to_polys(roi01: np.ndarray, bbox: Tuple[int,int,int,int]) -> List[np.ndarray]:
    """
    ROI mask -> polylines in full image coords.
    """
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
    bbox: Tuple[int,int,int,int]
    roi: np.ndarray
    roi_dil: np.ndarray
    pix: int


def _build_poly_rec(poly: np.ndarray, rank: int, W: int, H: int) -> Optional[PolyRec]:
    if poly is None or poly.shape[0] < 2:
        return None

    bbox = _poly_bbox(poly, W, H)
    roi = _poly_to_roi_mask(poly, bbox)
    pix = int(np.count_nonzero(roi))
    if pix < MIN_STROKE_PIXELS:
        return None

    if OVERLAP_TOL > 0 and _DILATE_KERNEL is not None:
        roi_dil = cv2.dilate(roi, _DILATE_KERNEL, iterations=1)
    else:
        roi_dil = roi

    return PolyRec(
        poly=poly.astype(np.float32, copy=False),
        rank=int(rank),
        bbox=bbox,
        roi=roi,
        roi_dil=roi_dil,
        pix=pix
    )


# ------------------- PUBLIC API -------------------

def merge_polyline_collections(
    collections: List[Tuple[str, List[np.ndarray]]],
    width: int,
    height: int,
) -> List[np.ndarray]:
    """
    Input:  list of (color_name, polylines)
    Output: merged polylines (kept order light->dark, minus contained/cut duplicates)
    """
    W = int(width)
    H = int(height)

    # Flatten and order by color rank (light->dark)
    pending: List[Tuple[int, np.ndarray]] = []
    for cname, polys in collections:
        r = color_rank(cname)
        for p in polys or []:
            pending.append((r, p))
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

    def replace_id_with_polys(sid: int, polys: List[np.ndarray], rank: int):
        try:
            pos = order.index(sid)
        except ValueError:
            pos = None

        old = active.get(sid)
        if old is None:
            return

        remove_id(sid)
        if pos is None:
            for p in polys:
                pr = _build_poly_rec(p, rank, W, H)
                if pr is not None:
                    add_rec(pr)
            return

        order.pop(pos)
        insert_pos = pos
        for p in polys:
            pr = _build_poly_rec(p, rank, W, H)
            if pr is not None:
                add_rec(pr, insert_at=insert_pos)
                insert_pos += 1

    def best_overlap_candidate(curr: PolyRec) -> Tuple[int, int, float]:
        """
        Pick candidate with best "duplicate-likeness":
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

            # compute overlap w.r.t the smaller stroke
            if curr.pix <= other.pix:
                ov = _fuzzy_overlap_small_to_big(curr.bbox, curr.roi, other.bbox, other.roi_dil)
                ratio = float(ov) / float(max(1, curr.pix))
            else:
                ov = _fuzzy_overlap_small_to_big(other.bbox, other.roi, curr.bbox, curr.roi_dil)
                ratio = float(ov) / float(max(1, other.pix))

            if (ratio > best_ratio) or (ratio == best_ratio and ov > best_ov):
                best_ratio = ratio
                best_ov = ov
                best_id = sid

        return best_id, best_ov, best_ratio

    for r, poly in pending:
        rec = _build_poly_rec(poly, r, W, H)
        if rec is None:
            continue

        while True:
            sid, ov_guess, ratio_guess = best_overlap_candidate(rec)
            if sid == 0 or ov_guess <= 0:
                break

            other = active.get(sid)
            if other is None:
                break

            # Decide smaller/bigger by pixel count
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

            # recompute consistent fuzzy overlap: SMALL vs dilated BIG
            ov = _fuzzy_overlap_small_to_big(small.bbox, small.roi, big.bbox, big.roi_dil)
            if ov <= 0:
                break

            ratio = float(ov) / float(small_pix)

            # Case 1: tiny overlap -> treat as intersection, do nothing
            if ov <= INTERSECT_ABS_MAX or ratio <= INTERSECT_RATIO_MAX:
                break

            # Case 2: mostly contained -> delete smaller
            if ratio >= DELETE_RATIO or (small_pix - ov) <= DELETE_REMAIN_MAX_PIX:
                if small_is_new:
                    rec = None
                    break
                else:
                    remove_id(sid)
                    continue

            # Case 3: partial overlap -> cut smaller
            if ratio >= CUT_RATIO_MIN:
                if small_is_new:
                    new_roi = _subtract_roi(rec.bbox, rec.roi, other.bbox, other.roi_dil)
                    remain_pix = int(np.count_nonzero(new_roi))
                    if remain_pix < MIN_STROKE_PIXELS:
                        rec = None
                        break
                    pieces = _roi_to_polys(new_roi, rec.bbox)
                    rec = None
                    for p in pieces:
                        pr = _build_poly_rec(p, r, W, H)
                        if pr is not None:
                            add_rec(pr)
                    break
                else:
                    new_roi = _subtract_roi(other.bbox, other.roi, rec.bbox, rec.roi_dil)
                    remain_pix = int(np.count_nonzero(new_roi))
                    if remain_pix < MIN_STROKE_PIXELS:
                        remove_id(sid)
                        continue
                    pieces = _roi_to_polys(new_roi, other.bbox)
                    if not pieces:
                        remove_id(sid)
                        continue
                    replace_id_with_polys(sid, pieces, other.rank)
                    continue

            break

        if rec is not None:
            add_rec(rec)

    out: List[np.ndarray] = []
    for sid in order:
        rr = active.get(sid)
        if rr is None:
            continue
        out.append(rr.poly)

    return out
