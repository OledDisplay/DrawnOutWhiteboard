#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from StrokeBundleMerger import merge_polyline_collections



# ------------------- PARALLELISM -------------------
BUNDLE_PARALLEL = True
BUNDLE_MAX_WORKERS = 0  # 0 = auto
CV2_THREADS_PER_PROCESS = 1  # prevent OpenCV thread oversubscription when using many processes


# ------------------- PATHS -------------------
BASE = Path(__file__).resolve().parent

# skeleton PNGs (0/255) from skeletonizer
IN_DIR = BASE / "Skeletonized"
OUT_DIR = BASE / "StrokeVectors"

# ------------------- GLOBAL KNOBS -------------------
LEFTOVER_MIN_CC_PIXELS = 0
BLUR_BEFORE_BIN = 0            # 0 off, else odd (e.g., 3)

CATMULL_ALPHA = 0.5            # centripetal Catmull–Rom
SEG_STRIDE = 1                 # subsample before fitting (keep 1 for full detail)

# stroke extraction / junction look-ahead knobs
LOOKAHEAD_STEPS_MAX        = 20   # max steps to simulate along a candidate branch

# branch scoring weights (lower score is better)
# stronger preference for straight & long branches, weaker for curvature
BRANCH_W_ANG               = 1.0  # weight for deflection angle
BRANCH_W_CURV              = 0.2  # weight for max local curvature
BRANCH_W_LEN               = 0.5  # weight for length (subtracted)

# junction continuation gating:
#ONLY angle is used to decide if we stop at a junction;
# curvature is NOT a hard gate here (corner splitting happens later).
JUNC_CONTINUE_ANG_MAX      = 60   # deg – how much deflection we still accept

# per-stroke segmentation (corner detection) knobs
CURV_WIN_RADIUS_SMALL      = 4      # half-window for local orientation smoothing
CURV_WARN_ANGLE            = 22.0   # deg – below this we don't consider a corner
CURV_CORNER_TOTAL_MIN      = 80.0   # deg – total turn threshold for a corner
CURV_CORNER_MAX_MIN        = 50.0   # deg – max local turn threshold for a corner
CURV_CORNER_REGION_MAXLEN  = 5      # max allowed region length for a concentrated corner
CURV_LOOKAHEAD_MAX         = 10     # max indices ahead for aggregation
CURV_MIN_SEG_POINTS        = 2      # minimum points in a segment to keep

# stroke extraction generic knobs
MIN_STROKE_POINTS          = 2      # drop only single-pixel "strokes"

#SET MERGE ANGLE < 90 FOR ACTUAL ACCURATE STROKE SPLITS - RIGHT NOW I HAVE IT SET MORE AGGRESIVE FOR MORE RENDER ENGINE WORK

# merge-small-strokes knobs
MERGE_SMALL_LEN_PERCENTILE = 75.0   # strokes below this length percentile are "small"
MERGE_SMALL_LEN_MIN        = 65.0   # absolute minimum length (px) to consider "not tiny"
MERGE_DIST_FRAC_DIAG       = 0.004  # relative distance gate vs image diagonal
MERGE_DIST_MIN             = 2   # px
MERGE_DIST_MAX             = 10.0    # px
MERGE_ANGLE_MAX            = 90.0  # deg – max angle at join to allow merge

#POINT SIMPLIFICATION
SIMPLIFY_EPS_PX = 0.4

# 8-connected neighbor offsets
NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

# ------------------- FAST 8-NBR LUTS (bitmasks) -------------------

_DIRS_8 = np.array(NEIGHBORS_8, dtype=np.int8)   # (8,2) dy,dx in the same order as NEIGHBORS_8
_DIR_DY = _DIRS_8[:, 0]
_DIR_DX = _DIRS_8[:, 1]

_DIR_TO_IDX = {(int(dy), int(dx)): i for i, (dy, dx) in enumerate(NEIGHBORS_8)}
_DIR_OPP = np.array([_DIR_TO_IDX[(-int(dy), -int(dx))] for dy, dx in NEIGHBORS_8], dtype=np.uint8)

_DIR_VEC = np.stack([_DIR_DX.astype(np.float32), _DIR_DY.astype(np.float32)], axis=1)  # (dx,dy)
_DIR_NRM = np.linalg.norm(_DIR_VEC, axis=1, keepdims=True)
_DIR_UNIT = _DIR_VEC / np.maximum(_DIR_NRM, 1e-12)

_DOT8 = (_DIR_UNIT @ _DIR_UNIT.T).astype(np.float32)
_DOT8 = np.clip(_DOT8, -1.0, 1.0)
_ANG8 = np.degrees(np.arccos(_DOT8)).astype(np.float32)

_BITCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _build_neighbor_mask_and_deg(skel01: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build:
      sk      : contiguous uint8 0/1
      nei_mask: uint8 bitmask per pixel (bit i => that 8-neighbor exists and is FG)
      deg     : uint8 degree per pixel (popcount of nei_mask), 0 on background
    """
    sk = (skel01 > 0).astype(np.uint8)
    if sk.size == 0:
        return sk, np.zeros_like(sk, dtype=np.uint8), np.zeros_like(sk, dtype=np.uint8)

    sk = np.ascontiguousarray(sk)
    h, w = sk.shape

    sp = np.pad(sk, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    center = sp[1:-1, 1:-1]

    nei = np.zeros((h, w), dtype=np.uint8)
    for i, (dy, dx) in enumerate(NEIGHBORS_8):
        nb = sp[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]
        nei |= ((nb > 0).astype(np.uint8) << i)

    nei[center == 0] = 0
    deg = _BITCOUNT_LUT[nei]
    return sk, nei, deg


def _angle_between_dir_and_chord_deg(dir_idx: int, chord_dx: float, chord_dy: float) -> float:
    """
    Angle (deg) between a step direction (one of the 8 dirs) and a chord vector.
    """
    nd = math.hypot(chord_dx, chord_dy)
    if nd < 1e-6:
        return 0.0

    dx = float(_DIR_DX[dir_idx])
    dy = float(_DIR_DY[dir_idx])
    nr = math.hypot(dx, dy)
    if nr < 1e-6:
        return 0.0

    dot = dx * chord_dx + dy * chord_dy
    c = dot / (nr * nd)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


# ------------------- COLOR GROUP ID -------------------
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

def _color_group_id_from_color_name(name: str) -> int:
    n = (name or "").strip().lower()
    try:
        return int(COLOR_ORDER_LIGHT_TO_DARK.index(n) + 1)  # 1..11
    except ValueError:
        return 0


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

    # skeleton output is 0 background, >0 foreground
    return (img > 0).astype(np.uint8)




# ------------------- GEOM HELPERS -------------------

def _unit_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)

def _angle_between_vecs(u: np.ndarray, v: np.ndarray) -> float:
    cu = _unit_vec(u)
    cv = _unit_vec(v)
    c = float(np.dot(cu, cv))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))

def _angle_diff_deg(a: float, b: float) -> float:
    """
    Smallest absolute difference between two angles (inputs in radians),
    returned in degrees.
    """
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return abs(d) * 180.0 / math.pi

def _stroke_length(poly: np.ndarray) -> float:
    """
    Arc-length of a polyline in pixels.
    """
    if poly is None or poly.shape[0] < 2:
        return 0.0
    diff = np.diff(poly.astype(np.float32), axis=0)
    seg_len = np.linalg.norm(diff, axis=1)
    return float(np.sum(seg_len))

# ------------------- GRAPH-BASED STROKE TRACER -------------------

def _edge_key(p, q):
    """
    Canonical undirected edge key so (p,q) and (q,p) are the same.
    p, q are (y, x) integer tuples.
    """
    (y1, x1) = p
    (y2, x2) = q
    if (y1, x1) <= (y2, x2):
        return (int(y1), int(x1), int(y2), int(x2))
    else:
        return (int(y2), int(x2), int(y1), int(x1))

def _build_degree_map(skel01: np.ndarray) -> np.ndarray:
    """
    For each foreground pixel (==1), count how many 8-neighbors are also foreground.
    Returns a uint8 degree image.
    """
    sk = (skel01 > 0).astype(np.uint8)
    if sk.size == 0:
        return np.zeros_like(sk, dtype=np.uint8)

    k = np.array([[1,1,1],
                  [1,0,1],
                  [1,1,1]], dtype=np.uint8)

    deg = cv2.filter2D(sk, ddepth=cv2.CV_8U, kernel=k, borderType=cv2.BORDER_CONSTANT)
    deg[sk == 0] = 0
    return deg


def _neighbors_fg(y: int, x: int, skel01: np.ndarray):
    """
    Return all foreground neighbors (8-connected) of (y,x).
    """
    h, w = skel01.shape
    out = []
    for dy, dx in NEIGHBORS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and skel01[ny, nx] > 0:
            out.append((ny, nx))
    return out

def _simulate_branch_from(curr, first_nbr, prev_dir_ref, skel01: np.ndarray, deg: np.ndarray,
                          max_steps: int = LOOKAHEAD_STEPS_MAX):
    """
    Simulate a short walk along a branch starting from 'curr' going to 'first_nbr'.
    Does NOT touch global visited_edges. Used only for scoring branches at junctions.

    Returns a dict with:
      - steps: number of steps taken (>=1 if valid)
      - length: same as steps (since pixels are 1 apart on graph)
      - ang_deflect: angle (deg) between prev_dir_ref and chord from curr to end
      - curv_max: maximum local turn angle (deg) along the simulated path
    """
    h, w = skel01.shape
    cy, cx = curr
    ny, nx = first_nbr

    path_pts = [[float(cx), float(cy)], [float(nx), float(ny)]]
    prev = (cy, cx)
    curr2 = (ny, nx)
    prev_step = np.array([nx - cx, ny - cy], dtype=np.float32)
    if np.linalg.norm(prev_step) < 1e-6:
        return None

    if prev_dir_ref is None:
        prev_dir_ref = prev_step.copy()

    steps = 1
    curv_max = 0.0
    local_seen = {(cy, cx), (ny, nx)}

    while steps < max_steps:
        cy2, cx2 = curr2
        nbrs = []
        for dy, dx in NEIGHBORS_8:
            yy, xx = cy2 + dy, cx2 + dx
            if 0 <= yy < h and 0 <= xx < w and skel01[yy, xx] > 0:
                if (yy, xx) == prev:
                    continue
                nbrs.append((yy, xx))
        if not nbrs:
            break

        best = None
        best_dot = -1e9
        for yy, xx in nbrs:
            v = np.array([xx - cx2, yy - cy2], dtype=np.float32)
            nrm = float(np.linalg.norm(v))
            if nrm < 1e-6:
                continue
            dot = float(np.dot(_unit_vec(prev_step), v / nrm))
            if dot > best_dot:
                best_dot = dot
                best = (yy, xx)
        if best is None:
            break

        yy, xx = best
        if (yy, xx) in local_seen:
            break
        local_seen.add((yy, xx))

        step_vec = np.array([xx - cx2, yy - cy2], dtype=np.float32)
        if np.linalg.norm(step_vec) < 1e-6:
            break

        turn = _angle_between_vecs(prev_step, step_vec)
        if turn > curv_max:
            curv_max = turn

        path_pts.append([float(xx), float(yy)])
        prev = (cy2, cx2)
        curr2 = (yy, xx)
        prev_step = step_vec
        steps += 1

        # stop sim at junction / endpoint
        if int(deg[yy, xx]) != 2:
            break

    end_x, end_y = path_pts[-1][0], path_pts[-1][1]
    chord = np.array([end_x - float(cx), end_y - float(cy)], dtype=np.float32)
    if np.linalg.norm(chord) < 1e-6:
        ang_deflect = 0.0
    else:
        ang_deflect = _angle_between_vecs(prev_dir_ref, chord)

    return {
        "steps": steps,
        "length": float(steps),
        "ang_deflect": float(ang_deflect),
        "curv_max": float(curv_max),
    }

def _trace_one_stroke(start, first_nbr, skel01: np.ndarray,
                      deg: np.ndarray,
                      visited_edges: set):
    """
    Walk a single stroke starting from 'start' going first to 'first_nbr',
    following the skeleton as a graph:

      - continue along the straightest branch through degree-2 chains,
      - at junctions, run a look-ahead to score branches,
      - ONLY terminate at junctions when the best branch deflects too much;
      - non-chosen branches at a junction are NOT marked visited here,
        so they can form independent strokes later.
    """
    h, w = skel01.shape
    sy, sx = start
    cy, cx = first_nbr

    stroke = [
        [float(sx), float(sy)],
        [float(cx), float(cy)],
    ]
    visited_edges.add(_edge_key(start, first_nbr))

    prev = (sy, sx)
    curr = (cy, cx)

    prev_dir = np.array([cx - sx, cy - sy], dtype=np.float32)
    if np.linalg.norm(prev_dir) < 1e-6:
        prev_dir = np.array([1.0, 0.0], dtype=np.float32)

    while True:
        cy, cx = curr
        py, px = prev

        # collect neighbors via unvisited edges only
        nbrs = []
        for dy, dx in NEIGHBORS_8:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and skel01[ny, nx] > 0:
                ek = _edge_key((cy, cx), (ny, nx))
                if ek in visited_edges:
                    continue
                nbrs.append((ny, nx))

        if not nbrs:
            break

        d_here = int(deg[cy, cx])

        # pure interior node: deg==2, usually one forward neighbor
        if d_here == 2:
            best = None
            best_dot = -1e9
            for ny, nx in nbrs:
                v = np.array([nx - cx, ny - cy], dtype=np.float32)
                if np.linalg.norm(v) < 1e-6:
                    continue
                dot = float(np.dot(_unit_vec(prev_dir), _unit_vec(v)))
                if dot > best_dot:
                    best_dot = dot
                    best = (ny, nx)
            if best is None:
                break

            ny, nx = best
            visited_edges.add(_edge_key((cy, cx), (ny, nx)))
            stroke.append([float(nx), float(ny)])
            prev = curr
            curr = (ny, nx)
            prev_dir = np.array([nx - cx, ny - cy], dtype=np.float32)
            if np.linalg.norm(prev_dir) < 1e-6:
                prev_dir = np.array([1.0, 0.0], dtype=np.float32)
            continue

        # junction / endpoint: score branches, pick best, but apply angle gating only
        best_candidate = None
        best_score = None
        best_metrics = None

        for ny, nx in nbrs:
            metrics = _simulate_branch_from(curr, (ny, nx), prev_dir, skel01, deg,
                                            max_steps=LOOKAHEAD_STEPS_MAX)
            if metrics is None:
                continue
            ang_def = metrics["ang_deflect"]
            curv_max = metrics["curv_max"]
            length = metrics["length"]

            # lower is better: small angle, low curvature, long branch
            score = (BRANCH_W_ANG * ang_def +
                     BRANCH_W_CURV * curv_max -
                     BRANCH_W_LEN * length)

            if (best_score is None) or (score < best_score):
                best_score = score
                best_candidate = (ny, nx)
                best_metrics = metrics

        # no usable branch: terminate stroke
        if best_candidate is None or best_metrics is None:
            break

        # junction continuation gating: only angle matters now
        if best_metrics["ang_deflect"] > JUNC_CONTINUE_ANG_MAX:
            # stop here; other edges remain unvisited for future strokes
            break

        # otherwise, accept this as the main continuation and record ONLY that edge
        ny, nx = best_candidate
        visited_edges.add(_edge_key((cy, cx), (ny, nx)))
        stroke.append([float(nx), float(ny)])
        prev = curr
        curr = (ny, nx)
        prev_dir = np.array([nx - cx, ny - cy], dtype=np.float32)
        if np.linalg.norm(prev_dir) < 1e-6:
            prev_dir = np.array([1.0, 0.0], dtype=np.float32)

    if len(stroke) < MIN_STROKE_POINTS:
        return None

    arr = np.asarray(stroke, dtype=np.float32)
    nan_to_num_nd(arr)
    return arr

def _trace_strokes_graph(skel01: np.ndarray) -> List[np.ndarray]:
    """
    Main graph-based stroke extractor.

    Input: skel01 – 0/1 skeleton mask (1px wide lines).
    Output: list of polylines, each Nx2 [x,y] float32.

    Same logic as before (degree-2 chains, junction lookahead scoring, angle gating),
    but using per-pixel 8-neighbor bitmasks + per-pixel visited-direction bitmasks
    instead of a Python set of edge tuples.
    """
    skel, nei_mask, deg = _build_neighbor_mask_and_deg(skel01)
    h, w = skel.shape
    if skel.size == 0:
        return []

    visited = np.zeros((h, w), dtype=np.uint8)
    strokes: List[np.ndarray] = []

    ys, xs = np.nonzero(skel)

    def _simulate_branch_from_fast(curr_yx, first_dir_idx, prev_dir_idx, max_steps: int = LOOKAHEAD_STEPS_MAX):
        """
        Simulate a short walk along a branch (NO global visited usage) for scoring at junctions.
        Mirrors _simulate_branch_from behavior but runs on masks and direction LUTs.
        """
        cy, cx = curr_yx
        dy = int(_DIR_DY[first_dir_idx]); dx = int(_DIR_DX[first_dir_idx])
        ny = cy + dy; nx = cx + dx
        if not (0 <= ny < h and 0 <= nx < w) or skel[ny, nx] == 0:
            return None

        if prev_dir_idx is None:
            prev_dir_idx = first_dir_idx

        steps = 1
        curv_max = 0.0

        prev = (cy, cx)
        curr2 = (ny, nx)
        prev_step_dir = int(first_dir_idx)

        local_seen = {(cy, cx), (ny, nx)}

        while steps < max_steps:
            cy2, cx2 = curr2

            avail = int(nei_mask[cy2, cx2])
            back_dir = int(_DIR_OPP[prev_step_dir])
            avail &= ~(1 << back_dir)
            if avail == 0:
                break

            best_dir = None
            best_dot = -1e9

            m = avail
            while m:
                lsb = m & -m
                d_idx = (lsb.bit_length() - 1)
                dot = float(_DOT8[prev_step_dir, d_idx])
                if dot > best_dot:
                    best_dot = dot
                    best_dir = d_idx
                m ^= lsb

            if best_dir is None:
                break

            yy = cy2 + int(_DIR_DY[best_dir])
            xx = cx2 + int(_DIR_DX[best_dir])

            if (yy, xx) in local_seen:
                break
            local_seen.add((yy, xx))

            turn = float(_ANG8[prev_step_dir, best_dir])
            if turn > curv_max:
                curv_max = turn

            prev = curr2
            curr2 = (yy, xx)
            prev_step_dir = int(best_dir)
            steps += 1

            if int(deg[yy, xx]) != 2:
                break

        end_y, end_x = curr2
        chord_dx = float(end_x - cx)
        chord_dy = float(end_y - cy)
        ang_deflect = _angle_between_dir_and_chord_deg(int(prev_dir_idx), chord_dx, chord_dy)

        return {
            "steps": steps,
            "length": float(steps),
            "ang_deflect": float(ang_deflect),
            "curv_max": float(curv_max),
        }

    def _trace_one_stroke_fast(start_yx, first_dir_idx):
        """
        Trace one stroke starting at start_yx and taking first_dir_idx as the first edge.
        Uses visited-direction bitmasks instead of visited edge tuple set.
        """
        sy, sx = start_yx

        dy = int(_DIR_DY[first_dir_idx]); dx = int(_DIR_DX[first_dir_idx])
        cy = sy + dy; cx = sx + dx
        if not (0 <= cy < h and 0 <= cx < w) or skel[cy, cx] == 0:
            return None

        stroke = [
            [float(sx), float(sy)],
            [float(cx), float(cy)],
        ]

        visited[sy, sx] |= (1 << int(first_dir_idx))
        visited[cy, cx] |= (1 << int(_DIR_OPP[int(first_dir_idx)]))

        prev = (sy, sx)
        curr = (cy, cx)
        prev_dir_idx = int(first_dir_idx)

        while True:
            cy, cx = curr

            avail = int(nei_mask[cy, cx]) & (~int(visited[cy, cx]) & 0xFF)
            if avail == 0:
                break

            d_here = int(deg[cy, cx])

            if d_here == 2:
                best_dir = None
                best_dot = -1e9
                m = avail
                while m:
                    lsb = m & -m
                    d_idx = (lsb.bit_length() - 1)
                    dot = float(_DOT8[prev_dir_idx, d_idx])
                    if dot > best_dot:
                        best_dot = dot
                        best_dir = d_idx
                    m ^= lsb

                if best_dir is None:
                    break

                ny = cy + int(_DIR_DY[best_dir])
                nx = cx + int(_DIR_DX[best_dir])

                visited[cy, cx] |= (1 << int(best_dir))
                visited[ny, nx] |= (1 << int(_DIR_OPP[int(best_dir)]))

                stroke.append([float(nx), float(ny)])

                prev = curr
                curr = (ny, nx)
                prev_dir_idx = int(best_dir)
                continue

            best_dir = None
            best_score = None
            best_metrics = None

            m = avail
            while m:
                lsb = m & -m
                d_idx = (lsb.bit_length() - 1)
                metrics = _simulate_branch_from_fast(curr, d_idx, prev_dir_idx, max_steps=LOOKAHEAD_STEPS_MAX)
                if metrics is not None:
                    ang_def = metrics["ang_deflect"]
                    curv_max = metrics["curv_max"]
                    length = metrics["length"]

                    score = (BRANCH_W_ANG * ang_def +
                             BRANCH_W_CURV * curv_max -
                             BRANCH_W_LEN * length)

                    if (best_score is None) or (score < best_score):
                        best_score = score
                        best_dir = d_idx
                        best_metrics = metrics
                m ^= lsb

            if best_dir is None or best_metrics is None:
                break

            if best_metrics["ang_deflect"] > JUNC_CONTINUE_ANG_MAX:
                break

            ny = cy + int(_DIR_DY[best_dir])
            nx = cx + int(_DIR_DX[best_dir])

            visited[cy, cx] |= (1 << int(best_dir))
            visited[ny, nx] |= (1 << int(_DIR_OPP[int(best_dir)]))

            stroke.append([float(nx), float(ny)])

            prev = curr
            curr = (ny, nx)
            prev_dir_idx = int(best_dir)

        if len(stroke) < MIN_STROKE_POINTS:
            return None

        arr = np.asarray(stroke, dtype=np.float32)
        nan_to_num_nd(arr)
        return arr

    # pass 1: start from nodes that are NOT pure degree-2 (endpoints, junctions)
    for y, x in zip(ys, xs):
        d = int(deg[y, x])
        if d == 0 or d == 2:
            continue

        m = int(nei_mask[y, x])
        while m:
            lsb = m & -m
            d_idx = (lsb.bit_length() - 1)
            if (int(visited[y, x]) & (1 << d_idx)) == 0:
                stroke = _trace_one_stroke_fast((int(y), int(x)), int(d_idx))
                if stroke is not None:
                    strokes.append(stroke)
            m ^= lsb

    # pass 2: pick up loops made only of degree-2 nodes (circles etc.)
    for y, x in zip(ys, xs):
        if int(deg[y, x]) != 2:
            continue

        m = int(nei_mask[y, x])
        while m:
            lsb = m & -m
            d_idx = (lsb.bit_length() - 1)
            if (int(visited[y, x]) & (1 << d_idx)) == 0:
                stroke = _trace_one_stroke_fast((int(y), int(x)), int(d_idx))
                if stroke is not None:
                    strokes.append(stroke)
            m ^= lsb

    return strokes


# ------------------- PER-STROKE SEGMENTATION (CORNERS) -------------------

def _compute_orientations(pts: np.ndarray, win_radius: int) -> np.ndarray:
    """
    Compute smoothed orientation angle (in radians) along a polyline using
    chord directions in a local window.
    pts: Nx2 array [x, y]
    """
    n = pts.shape[0]
    phi = np.zeros(n, dtype=np.float32)
    if n < 2:
        return phi

    P = pts.astype(np.float32, copy=False)
    nan_to_num_nd(P)

    idx = np.arange(n, dtype=np.int32)
    a = np.clip(idx - int(win_radius), 0, n - 1)
    b = np.clip(idx + int(win_radius), 0, n - 1)

    v = P[b] - P[a]

    eq = (a == b)
    if np.any(eq):
        # only really happens for win_radius==0 or n==1; match your original behavior
        ii = np.where(eq)[0]
        for i in ii:
            if i < n - 1:
                v[i] = P[i + 1] - P[i]
            else:
                v[i] = P[i] - P[i - 1]

    dx = v[:, 0]
    dy = v[:, 1]
    small = (np.abs(dx) < 1e-6) & (np.abs(dy) < 1e-6)

    phi[~small] = np.arctan2(dy[~small], dx[~small]).astype(np.float32)
    phi[small] = 0.0
    return phi


def _segment_polyline_on_corners(poly: np.ndarray) -> List[np.ndarray]:
    """
    Split a polyline into segments at concentrated corner regions, using
    multi-step curvature:

      - compute smoothed orientation φ[i] over a local window
      - local curvature dφ[i] = |φ[i] - φ[i-1]|
      - scan along the stroke; when dφ[i] exceeds CURV_WARN_ANGLE,
        look ahead up to CURV_LOOKAHEAD_MAX indices and aggregate:

          Σ|dφ|, max|dφ|, region length

        We only cut when all of these are true:

          - Σ|dφ| >= CURV_CORNER_TOTAL_MIN
          - max|dφ| >= CURV_CORNER_MAX_MIN
          - region length <= CURV_CORNER_REGION_MAXLEN
          - max|dφ| / Σ|dφ| is high (corner energy concentrated)
    """
    n = poly.shape[0]
    if n <= CURV_MIN_SEG_POINTS:
        return [poly]

    pts = poly.astype(np.float32, copy=True)
    nan_to_num_nd(pts)

    phi = _compute_orientations(pts, CURV_WIN_RADIUS_SMALL)
    dphi_deg = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        dphi_deg[i] = _angle_diff_deg(phi[i], phi[i-1])

    split_indices = []
    i = 1
    while i < n - 1:
        if dphi_deg[i] < CURV_WARN_ANGLE:
            i += 1
            continue

        j = min(n - 1, i + CURV_LOOKAHEAD_MAX)
        region = dphi_deg[i:j+1]
        total_turn = float(np.sum(np.abs(region)))
        max_turn = float(np.max(np.abs(region)))
        region_len = int(j - i + 1)

        if total_turn <= 1e-6:
            i += 1
            continue

        concentration = max_turn / total_turn

        if (total_turn >= CURV_CORNER_TOTAL_MIN and
            max_turn >= CURV_CORNER_MAX_MIN and
            region_len <= CURV_CORNER_REGION_MAXLEN and
            concentration >= 0.7):
            # concentrated sharp corner → choose index of max_turn
            rel_idx = int(np.argmax(np.abs(region)))
            corner_idx = i + rel_idx
            if 0 < corner_idx < n - 1:
                split_indices.append(corner_idx)
            i = corner_idx + 2
        else:
            i += 1

    if not split_indices:
        return [poly]

    split_indices = sorted(set(split_indices))
    segments: List[np.ndarray] = []
    start = 0
    for s in split_indices:
        end = s
        if end - start + 1 >= CURV_MIN_SEG_POINTS:
            seg = pts[start:end+1].copy()
            segments.append(seg)
        start = s
    if n - start >= CURV_MIN_SEG_POINTS:
        segments.append(pts[start:].copy())

    if not segments:
        return [poly]
    return segments

# ------------------- CONSERVATIVE POLY SIMPLIFIER -------------------

def _simplify_poly_conservative(poly: np.ndarray) -> np.ndarray:
    """
    Extremely conservative simplifier:
    - remove exact duplicate / near-duplicate points
    - remove middle points where the step direction prev→curr and curr→next
      is EXACTLY the same (integer direction), i.e. pure straight runs
      like horizontal / vertical / 45° lines.
    - never touches curves where direction changes at any step.
    """
    n = poly.shape[0]
    if n <= 2:
        return poly

    pts = poly.astype(np.float32, copy=True)
    nan_to_num_nd(pts)

    # 1) drop exact / near-duplicate points
    dedup = [pts[0]]
    for i in range(1, n):
        if np.linalg.norm(pts[i] - dedup[-1]) > 1e-3:
            dedup.append(pts[i])
    if len(dedup) <= 2:
        return np.asarray(dedup, dtype=np.float32)
    pts = np.asarray(dedup, dtype=np.float32)
    n = pts.shape[0]

    # 2) drop middle points with identical step direction left/right
    out = [pts[0]]
    for i in range(1, n - 1):
        p_prev = pts[i - 1]
        p_curr = pts[i]
        p_next = pts[i + 1]

        v1 = p_curr - p_prev
        v2 = p_next - p_curr

        v1i = np.round(v1).astype(int)
        v2i = np.round(v2).astype(int)

        if v1i[0] == v2i[0] and v1i[1] == v2i[1]:
            continue

        out.append(p_curr)

    out.append(pts[-1])
    if len(out) < 2:
        return pts
    return np.asarray(out, dtype=np.float32)


# ------------------- MERGE SMALL STROKES -------------------

def _merge_small_strokes(polys: List[np.ndarray], diag: float) -> List[np.ndarray]:
    """
    Optimized version of your merge-small-strokes pass.

    Old version was O(n^2). This uses spatial hashing on stroke endpoints:
    - index endpoints into grid tiles
    - for each small stroke, only consider candidates in neighboring tiles
    """
    n = len(polys)
    if n == 0:
        return polys

    merged = [p.astype(np.float32, copy=True) for p in polys]
    for p in merged:
        nan_to_num_nd(p)

    lengths = [_stroke_length(p) for p in merged]
    len_arr = np.array(lengths, dtype=np.float32)
    if not np.any(len_arr > 0):
        return merged

    if n >= 10:
        perc = float(np.percentile(len_arr, MERGE_SMALL_LEN_PERCENTILE))
        small_len_thresh = max(MERGE_SMALL_LEN_MIN, perc)
    else:
        small_len_thresh = max(MERGE_SMALL_LEN_MIN, float(np.min(len_arr)))

    small_indices = [i for i, L in enumerate(lengths) if 0.0 < L <= small_len_thresh]
    if not small_indices:
        return merged

    base = MERGE_DIST_FRAC_DIAG * float(diag)
    dist_thresh = max(MERGE_DIST_MIN, min(MERGE_DIST_MAX, base))
    dist2_thresh = float(dist_thresh * dist_thresh)
    angle_thresh = MERGE_ANGLE_MAX

    tile = int(max(4.0, dist_thresh * 2.0))
    grid = {}  # (tx,ty) -> set(stroke_index)
    active = [True] * n

    s_start = [None] * n
    s_end   = [None] * n
    t_start = [None] * n
    t_end   = [None] * n

    stroke_cells: List[set] = [set() for _ in range(n)]

    def _cell_for_pt(pt):
        return (int(pt[0] // tile), int(pt[1] // tile))

    def _deindex(idx):
        cells = stroke_cells[idx]
        if not cells:
            return
        for key in cells:
            s = grid.get(key)
            if s is not None:
                s.discard(idx)
                if not s:
                    grid.pop(key, None)
        cells.clear()

    def _index_endpoint(idx, pt):
        key = _cell_for_pt(pt)
        grid.setdefault(key, set()).add(idx)
        stroke_cells[idx].add(key)

    def recache(idx):
        _deindex(idx)

        p = merged[idx]
        if p is None or p.shape[0] < 2:
            s_start[idx] = s_end[idx] = None
            t_start[idx] = t_end[idx] = None
            return

        a = p[0]
        b = p[-1]
        s_start[idx] = a
        s_end[idx] = b

        if p.shape[0] >= 2:
            t_start[idx] = _unit_vec(p[1] - p[0])
            t_end[idx]   = _unit_vec(p[-1] - p[-2])
        else:
            t_start[idx] = np.array([1.0, 0.0], np.float32)
            t_end[idx]   = np.array([1.0, 0.0], np.float32)

        _index_endpoint(idx, a)
        _index_endpoint(idx, b)

    for i in range(n):
        if merged[i] is None or merged[i].shape[0] < 2:
            active[i] = False
            continue
        recache(i)

    def candidates_near(pt):
        tx, ty = _cell_for_pt(pt)
        out = set()
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                out |= grid.get((tx + ox, ty + oy), set())
        return out

    small_indices_sorted = sorted(small_indices, key=lambda i: lengths[i])

    for si in small_indices_sorted:
        if not active[si]:
            continue
        p_small = merged[si]
        if p_small is None or p_small.shape[0] < 2:
            active[si] = False
            _deindex(si)
            continue

        s0 = s_start[si]
        s1 = s_end[si]
        if s0 is None or s1 is None:
            active[si] = False
            _deindex(si)
            continue

        st0 = t_start[si]
        st1 = t_end[si]

        cand = candidates_near(s0) | candidates_near(s1)
        cand.discard(si)

        best_target_j = None
        best_new_poly = None
        best_score = None

        for j in cand:
            if not active[j]:
                continue
            p_big = merged[j]
            if p_big is None or p_big.shape[0] < 2:
                continue

            b0 = s_start[j]
            b1 = s_end[j]
            if b0 is None or b1 is None:
                continue

            bt0 = t_start[j]
            bt1 = t_end[j]

            dx = float(b1[0] - s0[0]); dy = float(b1[1] - s0[1])
            d2 = dx*dx + dy*dy
            if d2 <= dist2_thresh:
                d = math.sqrt(d2)
                ang = _angle_between_vecs(bt1, st0)
                if ang <= angle_thresh:
                    score = float(d + 0.1 * ang)
                    if best_score is None or score < best_score:
                        new = np.vstack([p_big, p_small[1:]]) if d < 1e-3 else np.vstack([p_big, p_small])
                        nan_to_num_nd(new)
                        if finite_nd(new) and new.shape[0] >= 2:
                            best_score = score
                            best_target_j = j
                            best_new_poly = new

            dx = float(b1[0] - s1[0]); dy = float(b1[1] - s1[1])
            d2 = dx*dx + dy*dy
            if d2 <= dist2_thresh:
                d = math.sqrt(d2)
                ang = _angle_between_vecs(bt1, -st1)
                if ang <= angle_thresh:
                    score = float(d + 0.1 * ang)
                    if best_score is None or score < best_score:
                        s_rev = p_small[::-1]
                        new = np.vstack([p_big, s_rev[1:]]) if d < 1e-3 else np.vstack([p_big, s_rev])
                        nan_to_num_nd(new)
                        if finite_nd(new) and new.shape[0] >= 2:
                            best_score = score
                            best_target_j = j
                            best_new_poly = new

            dx = float(s1[0] - b0[0]); dy = float(s1[1] - b0[1])
            d2 = dx*dx + dy*dy
            if d2 <= dist2_thresh:
                d = math.sqrt(d2)
                ang = _angle_between_vecs(st1, bt0)
                if ang <= angle_thresh:
                    score = float(d + 0.1 * ang)
                    if best_score is None or score < best_score:
                        new = np.vstack([p_small, p_big[1:]]) if d < 1e-3 else np.vstack([p_small, p_big])
                        nan_to_num_nd(new)
                        if finite_nd(new) and new.shape[0] >= 2:
                            best_score = score
                            best_target_j = j
                            best_new_poly = new

            dx = float(s0[0] - b0[0]); dy = float(s0[1] - b0[1])
            d2 = dx*dx + dy*dy
            if d2 <= dist2_thresh:
                d = math.sqrt(d2)
                ang = _angle_between_vecs(-st0, bt0)
                if ang <= angle_thresh:
                    score = float(d + 0.1 * ang)
                    if best_score is None or score < best_score:
                        s_rev = p_small[::-1]
                        new = np.vstack([s_rev, p_big[1:]]) if d < 1e-3 else np.vstack([s_rev, p_big])
                        nan_to_num_nd(new)
                        if finite_nd(new) and new.shape[0] >= 2:
                            best_score = score
                            best_target_j = j
                            best_new_poly = new

        if best_target_j is not None and best_new_poly is not None:
            merged[best_target_j] = best_new_poly
            lengths[best_target_j] = _stroke_length(best_new_poly)
            recache(best_target_j)

            active[si] = False
            _deindex(si)

    out = [
        merged[i]
        for i in range(n)
        if active[i] and merged[i] is not None and merged[i].shape[0] >= 2
    ]
    return out



def _coverage_safety_pass(skel01: np.ndarray,
                          polys: List[np.ndarray]) -> List[np.ndarray]:
    """
    Final safety pass:

    We rasterize all current polylines into a coverage mask and compare
    that to the original skeleton. Any skeleton pixels that are still
    uncovered and form a connected component with >= LEFTOVER_MIN_CC_PIXELS
    pixels are turned into simple polylines and appended.

    This catches cases where a branch was visited by the smart logic,
    but the corresponding stroke got thrown away later.
    """
    sk = (skel01 > 0).astype(np.uint8)
    h, w = sk.shape

    # draw all existing strokes into a mask
    coverage = np.zeros_like(sk, dtype=np.uint8)
    for p in polys:
        if p is None or p.shape[0] < 2:
            continue
        pts = np.round(p).astype(np.int32)
        # clip to image bounds to be safe
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        cv2.polylines(
            coverage,
            [pts.reshape(-1, 1, 2)],
            isClosed=False,
            color=255,
            thickness=1,
        )

    # skeleton pixels not covered by any current stroke
    uncovered = (sk > 0) & (coverage == 0)
    if not np.any(uncovered):
        return polys

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        uncovered.astype(np.uint8), connectivity=8
    )

    extra: List[np.ndarray] = []

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < LEFTOVER_MIN_CC_PIXELS:
            continue

        comp_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        for c in contours:
            if c is None or len(c) < 2:
                continue
            pts = c.reshape(-1, 2).astype(np.float32)  # (x,y)
            nan_to_num_nd(pts)
            if pts.shape[0] >= 2 and finite_nd(pts):
                extra.append(pts)

    if extra:
        polys = polys + extra
    return polys


# ------------------- CURVES (Catmull–Rom → Bézier) -------------------

def _catmull_rom_to_beziers(pts: np.ndarray, alpha=0.5) -> List[List[float]]:
    """
    Safe Catmull–Rom (centripetal) → cubic Beziers.
    Fixes NaNs caused by duplicated endpoints / zero-length parameter intervals.
    """
    P = pts.astype(np.float32)
    nan_to_num_nd(P)

    n = P.shape[0]
    if n < 2:
        return []

    # 1) drop consecutive duplicates (very common in pixel polylines)
    keep = [0]
    for i in range(1, n):
        if float(np.linalg.norm(P[i] - P[i - 1])) > 1e-6:
            keep.append(i)
    P = P[keep]
    n = P.shape[0]
    if n < 2:
        return []

    # 2) endpoint extrapolation instead of duplicating endpoints
    # avoids t0==tm and similar zero denominators on first/last segment
    if n >= 3:
        Pm = P[0] + (P[0] - P[1])
        Pp = P[-1] + (P[-1] - P[-2])
    else:
        # only 2 points: just extrapolate linearly
        Pm = P[0] + (P[0] - P[1])
        Pp = P[1] + (P[1] - P[0])

    P_ext = np.vstack((Pm, P, Pp)).astype(np.float32)

    # centripetal parameterization
    def tj(ti, Pi, Pj):
        d = float(np.linalg.norm(Pj - Pi))
        if not math.isfinite(d):
            d = 0.0
        d = max(d, 1e-4)  # enforce strictly increasing t
        return ti + (d ** alpha)

    t = [0.0]
    for i in range(1, P_ext.shape[0]):
        t.append(tj(t[-1], P_ext[i - 1], P_ext[i]))

    eps = 1e-6
    beziers: List[List[float]] = []

    # segments between P[i] and P[i+1] in original polyline
    # P_ext index shift: original P[0] is at P_ext[1]
    for i in range(1, n):
        k = i  # P0 at P_ext[k], P1 at P_ext[k+1]
        Pm = P_ext[k - 1]
        P0 = P_ext[k]
        P1 = P_ext[k + 1]
        Pp = P_ext[k + 2]

        tm, t0, t1, tp = t[k - 1], t[k], t[k + 1], t[k + 2]

        # If any interval is degenerate, fall back to straight tangents
        if (abs(t0 - tm) < eps or abs(t1 - t0) < eps or abs(tp - t1) < eps):
            m0 = (P1 - P0)
            m1 = (P1 - P0)
        else:
            # safe denominators
            a1 = (P1 - Pm) / max(eps, float(t1 - tm))
            a2 = (P0 - Pm) / max(eps, float(t0 - tm))
            m0 = (a1 - a2) * float(t1 - t0)

            b1 = (Pp - P0) / max(eps, float(tp - t0))
            b2 = (P1 - P0) / max(eps, float(t1 - t0))
            m1 = (b1 - b2) * float(t1 - t0)

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
        else:
            # hard fallback: straight segment
            a = P0.astype(np.float32)
            b = P1.astype(np.float32)
            c1s = a + (b - a) / 3.0
            c2s = a + 2.0 * (b - a) / 3.0
            seg2 = [
                float(a[0]), float(a[1]),
                float(c1s[0]), float(c1s[1]),
                float(c2s[0]), float(c2s[1]),
                float(b[0]), float(b[1]),
            ]
            if finite_seq(seg2):
                beziers.append(seg2)

    return beziers


# ------------------- PROCESS SINGLE -------------------

def _process_single_to_data(path: Path, color_group_id: int = 0) -> dict:
    fg = _load_edges_binary(path)
    H, W = fg.shape
    diag = math.hypot(W, H)

    t0 = time.time()

    sk = (fg > 0).astype(np.uint8)

    polys_raw = _trace_strokes_graph(sk)

    polys_seg: List[np.ndarray] = []
    for p in polys_raw:
        if p.shape[0] < 2:
            continue
        segments = _segment_polyline_on_corners(p)
        for seg in segments:
            if seg.shape[0] >= CURV_MIN_SEG_POINTS:
                polys_seg.append(seg)

    polys_simpl: List[np.ndarray] = []
    for p in polys_seg:
        if p.shape[0] < 2:
            continue
        simp = _simplify_poly_conservative(p)
        if simp.shape[0] >= 2:
            polys_simpl.append(simp)
        else:
            polys_simpl.append(p)

    polys_merged = _merge_small_strokes(polys_simpl, diag)

    if SEG_STRIDE > 1:
        polys_final = [
            p[::SEG_STRIDE] if p.shape[0] > SEG_STRIDE else p
            for p in polys_merged
        ]
    else:
        polys_final = polys_merged

    polys_final = _coverage_safety_pass(sk, polys_final)

    strokes = []
    for p in polys_final:
        if p.shape[0] < 2:
            continue
        beziers = _catmull_rom_to_beziers(p, alpha=CATMULL_ALPHA)
        if not beziers:
            q = p.astype(float)
            segs = []
            for i in range(q.shape[0] - 1):
                a = q[i]
                b = q[i + 1]
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
            strokes.append({"segments": beziers, "color_group_id": int(color_group_id)})

    t1 = time.time()

    data = {
        "version": 14,
        "source_image": str(path),
        "width": W,
        "height": H,
        "vector_format": "bezier_cubic",
        "strokes": strokes,
        "stats": {
            "curves": len(strokes),
            "time_sec": round(t1 - t0, 3),
            "seg_stride": SEG_STRIDE,
            "min_stroke_points": MIN_STROKE_POINTS,
            "curv_win_radius_small": CURV_WIN_RADIUS_SMALL,
            "curv_warn_angle": CURV_WARN_ANGLE,
            "curv_corner_total_min": CURV_CORNER_TOTAL_MIN,
            "curv_corner_max_min": CURV_CORNER_MAX_MIN,
            "curv_lookahead_max": CURV_LOOKAHEAD_MAX,
            "junc_continue_ang_max": JUNC_CONTINUE_ANG_MAX,
            "merge_small_len_percentile": MERGE_SMALL_LEN_PERCENTILE,
        },
    }
    return data

def _process_single_to_polylines(path: Path) -> Tuple[int, int, List[np.ndarray]]:
    fg = _load_edges_binary(path)
    H, W = fg.shape
    diag = math.hypot(W, H)

    sk = (fg > 0).astype(np.uint8)

    polys_raw = _trace_strokes_graph(sk)

    polys_simpl: List[np.ndarray] = []
    for p in polys_raw:
        if p.shape[0] < 2:
            continue
        segments = _segment_polyline_on_corners(p)
        for seg in segments:
            if seg.shape[0] < CURV_MIN_SEG_POINTS:
                continue
            simp = _simplify_poly_conservative(seg)
            polys_simpl.append(simp if simp.shape[0] >= 2 else seg)

    polys_merged = _merge_small_strokes(polys_simpl, diag)

    if SEG_STRIDE > 1:
        polys_final = [
            p[::SEG_STRIDE] if p.shape[0] > SEG_STRIDE else p
            for p in polys_merged
        ]
    else:
        polys_final = polys_merged

    polys_final = _coverage_safety_pass(sk, polys_final)

    return W, H, polys_final




def _process_single(path: Path, output: Path) -> Tuple[str, dict]:
    cname = _parse_color_name_from_stem(path.stem)
    gid = _color_group_id_from_color_name(cname)
    data = _process_single_to_data(path, color_group_id=gid)
    output.mkdir(parents=True, exist_ok=True)
    out_json = output / f"{path.stem}.json"
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path), data

def _process_one_bundle_dir(bdir_str: str, out_dir_str: str) -> Tuple[str, dict]:
    """
    Worker: process ONE bundle folder (11-pass images), merge polylines, convert to bezier, write JSON.
    Returns (bundle_name, stats_dict). Safe for Windows ProcessPool (top-level, picklable).
    """
    # keep OpenCV from spawning extra threads in each process
    try:
        cv2.setNumThreads(int(CV2_THREADS_PER_PROCESS))
    except Exception:
        pass

    bdir = Path(bdir_str)
    out_dir = Path(out_dir_str)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    imgs = sorted(
        [p for p in bdir.glob("*") if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: p.name.lower(),
    )

    if not imgs:
        return (bdir.name, {"skipped": True, "reason": "no images"})

    t0 = time.time()

    collections = []
    base_W = None
    base_H = None
    pass_total_polys = 0

    for p in imgs:
        W, H, polys = _process_single_to_polylines(p)
        cname = _parse_color_name_from_stem(p.stem)
        collections.append((cname, polys))

        pass_total_polys += len(polys)
        if base_W is None:
            base_W = W
            base_H = H

    if base_W is None or base_H is None:
        return (bdir.name, {"skipped": True, "reason": "no valid passes"})

    merged_items = merge_polyline_collections(collections, width=base_W, height=base_H)

    # Convert merged polylines -> bezier strokes once (keep color_group_id from merger)
    strokes = []
    for poly, gid in merged_items:
        if poly is None or poly.shape[0] < 2:
            continue
        beziers = _catmull_rom_to_beziers(poly, alpha=CATMULL_ALPHA)
        if not beziers:
            continue
        strokes.append({"segments": beziers, "color_group_id": int(gid)})

    out = {
        "version": 15,
        "source_image": str(bdir),
        "width": int(base_W),
        "height": int(base_H),
        "vector_format": "bezier_cubic",
        "strokes": strokes,
        "stats": {
            "curves": int(len(strokes)),
            "passes": int(len(collections)),
            "pass_total_polys": int(pass_total_polys),
            "time_sec": round(time.time() - t0, 3),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{bdir.name}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    return (bdir.name, out["stats"])



# ------------------- MAIN -------------------

def _parse_color_name_from_stem(stem: str) -> str:
    s = stem.lower()
    # expected: edges_<idx>_<color>
    parts = s.split("_")
    if len(parts) >= 3:
        return parts[-1]
    return "unknown"


def vectorize_images():

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _out_exists(stem: str) -> bool:
        return (OUT_DIR / f"{stem}.json").exists()

    # bundle folders (new format)
    bundle_dirs = sorted(
        [p for p in IN_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower()
    )

    # legacy flat files in root (old format)
    flat_imgs = sorted([
        p for p in IN_DIR.glob("*")
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    ], key=lambda p: p.name.lower())

    bundle_dirs = [b for b in bundle_dirs if not _out_exists(b.name)]
    flat_imgs = [p for p in flat_imgs if not _out_exists(p.stem)]


    print(f"[INFO] IN={IN_DIR} OUT={OUT_DIR} bundles={len(bundle_dirs)} flat={len(flat_imgs)}")

    # ---- 1) bundled (11-pass) images in parallel ----
    if bundle_dirs:
        if BUNDLE_PARALLEL:
            cpu = os.cpu_count() or 1
            if BUNDLE_MAX_WORKERS and BUNDLE_MAX_WORKERS > 0:
                workers = min(int(BUNDLE_MAX_WORKERS), len(bundle_dirs))
            else:
                workers = min(max(1, cpu - 1), len(bundle_dirs))

            print(f"[INFO] bundle parallel: workers={workers}")

            futures = []
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for bdir in bundle_dirs:
                    futures.append(ex.submit(_process_one_bundle_dir, str(bdir), str(OUT_DIR)))

                for f in as_completed(futures):
                    bname, stats = f.result()
                    if stats.get("skipped"):
                        print(f"[SKIP] {bname}: {stats.get('reason')}")
                    else:
                        print(f"[OK] {bname}: curves={stats['curves']} passes={stats['passes']} "
                              f"pass_polys={stats['pass_total_polys']} time={stats['time_sec']}s")
        else:
            for bdir in bundle_dirs:
                bname, stats = _process_one_bundle_dir(str(bdir), str(OUT_DIR))
                if stats.get("skipped"):
                    print(f"[SKIP] {bname}: {stats.get('reason')}")
                else:
                    print(f"[OK] {bname}: curves={stats['curves']} passes={stats['passes']} "
                          f"pass_polys={stats['pass_total_polys']} time={stats['time_sec']}s")

    # ---- 2) legacy flat images exactly as before ----
    for p in flat_imgs:
        src, meta = _process_single(p, OUT_DIR)
        print(f"[OK] {Path(src).name}: strokes={len(meta['strokes'])}, "
              f"time={meta['stats']['time_sec']}s")

def _process_edges_mask_to_polylines(edges_u8: np.ndarray) -> Tuple[int, int, List[np.ndarray]]:
    """
    edges_u8: 0/255 (or 0/1) skeleton/edges mask.
    Returns (W, H, polys_final) with the SAME pipeline as _process_single_to_polylines().
    """
    m = (edges_u8 > 0).astype(np.uint8)
    H, W = m.shape[:2]
    diag = math.hypot(W, H)

    polys_raw = _trace_strokes_graph(m)

    polys_simpl: List[np.ndarray] = []
    for p in polys_raw:
        if p.shape[0] < 2:
            continue
        segments = _segment_polyline_on_corners(p)
        for seg in segments:
            if seg.shape[0] < CURV_MIN_SEG_POINTS:
                continue
            simp = _simplify_poly_conservative(seg)
            polys_simpl.append(simp if simp.shape[0] >= 2 else seg)

    polys_merged = _merge_small_strokes(polys_simpl, diag)

    if SEG_STRIDE > 1:
        polys_final = [
            p[::SEG_STRIDE] if p.shape[0] > SEG_STRIDE else p
            for p in polys_merged
        ]
    else:
        polys_final = polys_merged

    polys_final = _coverage_safety_pass(m, polys_final)

    return W, H, polys_final


def _vectorize_worker_payload(payload: tuple[int, dict, bool]) -> tuple[int, dict]:
    idx, item, save_outputs = payload
    try:
        cv2.setNumThreads(int(CV2_THREADS_PER_PROCESS))
    except Exception:
        pass

    skeletons = item["skeletons"]
    H, W = item["size"]

    collections: list[tuple[str, List[np.ndarray]]] = []
    base_W = None
    base_H = None

    for gname, skel_u8 in skeletons.items():
        w2, h2, polys = _process_edges_mask_to_polylines(skel_u8)
        collections.append((str(gname), polys))
        if base_W is None:
            base_W = w2
            base_H = h2

    use_W = int(base_W if base_W is not None else W)
    use_H = int(base_H if base_H is not None else H)

    merged_items = merge_polyline_collections(collections, width=use_W, height=use_H)

    strokes: list[dict] = []
    for poly, gid in merged_items:
        if poly is None or getattr(poly, "shape", None) is None or poly.shape[0] < 2:
            continue

        beziers = _catmull_rom_to_beziers(poly, alpha=CATMULL_ALPHA)
        if not beziers:
            q = poly.astype(float)
            segs = []
            for i in range(q.shape[0] - 1):
                a = q[i]
                b = q[i + 1]
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
            strokes.append({"segments": beziers, "color_group_id": int(gid)})

    meta = {
        "version": 15,
        "source_image": f"in_memory:{idx}",
        "width": use_W,
        "height": use_H,
        "vector_format": "bezier_cubic",
        "strokes": strokes,
    }

    if save_outputs:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        p = OUT_DIR / f"processed_{idx}.json"
        if not p.exists():
            p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


    return idx, meta



def vectorize_in_memory(
    skel_by_idx: dict[int, dict],
    *,
    save_outputs: bool = False,
    parallel: bool = False,
) -> dict[int, dict]:
    """
    Input: skel_by_idx[idx]["skeletons"][gname] is 0/255 mask
           skel_by_idx[idx]["size"] is (H, W)
    Output: same shape as your JSON style: strokes with flat cubic segments.
    """
    out: dict[int, dict] = {}

    def _one(idx: int, item: dict) -> tuple[int, dict]:
        skeletons = item["skeletons"]
        H, W = item["size"]

        collections: list[tuple[str, List[np.ndarray]]] = []

        base_W = None
        base_H = None

        for gname, skel_u8 in skeletons.items():
            w2, h2, polys = _process_edges_mask_to_polylines(skel_u8)
            collections.append((str(gname), polys))

            if base_W is None:
                base_W = w2
                base_H = h2

        # If size metadata disagrees with mask size, trust the mask.
        use_W = int(base_W if base_W is not None else W)
        use_H = int(base_H if base_H is not None else H)

        merged_items = merge_polyline_collections(collections, width=use_W, height=use_H)

        strokes: list[dict] = []
        for poly, gid in merged_items:
            if poly is None or getattr(poly, "shape", None) is None or poly.shape[0] < 2:
                continue

            beziers = _catmull_rom_to_beziers(poly, alpha=CATMULL_ALPHA)
            if not beziers:
                # keep the exact fallback behavior you use in _process_single_to_data
                q = poly.astype(float)
                segs = []
                for i in range(q.shape[0] - 1):
                    a = q[i]
                    b = q[i + 1]
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
                strokes.append({"segments": beziers, "color_group_id": int(gid)})

        meta = {
            "version": 15,
            "source_image": f"in_memory:{idx}",
            "width": use_W,
            "height": use_H,
            "vector_format": "bezier_cubic",
            "strokes": strokes,
        }

        if save_outputs:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            p = OUT_DIR / f"processed_{idx}.json"
            p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        return idx, meta

    if parallel:
        cpu = os.cpu_count() or 2
        workers = max(1, cpu - 1)
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = []
            for idx, item in skel_by_idx.items():
                futs.append(ex.submit(_vectorize_worker_payload, (int(idx), item, bool(save_outputs))))
            for f in as_completed(futs):
                idx, meta = f.result()
                out[idx] = meta
    else:
        for idx, item in skel_by_idx.items():
            idx, meta = _one(idx, item)
            out[idx] = meta

    return out




if __name__ == "__main__":
    mp.freeze_support()  # Windows-safe for ProcessPool
    try:
        cv2.setNumThreads(int(CV2_THREADS_PER_PROCESS))
    except Exception:
        pass
    vectorize_images()