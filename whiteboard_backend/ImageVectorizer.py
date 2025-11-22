#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


#Step 3 of printing - take our skeletonized image and split it into vectors to draw

#Input  - 1px skeleton PNGs (0 / 255) coming out of our pen-width-aware skeletonizer
#Output - JSON files with clean, cubic Bezier strokes, ready for drawing / printing

#This is the main image → vector step. Here we take the thin skeleton lines and turn them into strokes that actually make sense to draw

#Step 1 – load + binarize
#We treat the skeleton PNG as a simple 0/1 mask, and dont do any extra thinning -> thats been pushed out to the skeletonizer :

#Step 2 – graph-based stroke tracing
#We see the skeleton as a graph:
#- every foreground pixel is a node,
#- 8-connected neighbors are edges,
#- degree map tells us if a pixel is an endpoint of a line, a plain interior point - 2 node, or a junction (intersecting point of lines)

#Tracing logic:
#As a baseline we  want to break off at SHARP turns to have simpler strokes. But when 2 lines intersect we dont want a false breakoff -> we wanna keep going
#So we start strokes from all non-degree-2 nodes first (endpoints / junctions),
#- follow chains of degree-2 pixels straight through (simple interior lines) -> simply walking a line,
#- at junctions (intersections) we simulate short “look-ahead” walks on all branches and scores them by:
#   * how straight they are (deflection angle),
#   * how smooth they are (local curvature),
#   * how long they run.
#This allows us to continue on a valid path, and leave the other sharp degreee breakoffs behind, not causing trouble

#Second pass:
#- after endpoints/junctions, we run again starting from pure degree-2 loops to pick up closed shapes like circles that never hit an endpoint.

#Step 3 – corner-aware segmentation 
#Here we have ready traced lines and we ACTUALLY start splitting their inner sharp turns into seperate strokes
#We do this by:
#- computing a smoothed direction angle along the polyline in a small window,
#- measuring how much the direction changes between samples (curvature),
#- scanning forward and grouping “turn regions” over several indices.

#Step 4 – Douglas–Peucker simplification
#Now each stroke is still a polyline with potentially too many points ->
#We want to give more opurtunity to the rendering engine, and not just have many pixels to draw
#We run a conservative
#We never touch indices where the direction changes, so curves stay intact;

#Step 5 – merge small strokes into neighbours
#The vectorizer tends to create lots of tiny strokes around bigger ones and we end up with a crazy amount of strokes
#*optimizing the walk and breakoff logic to prevent this is just not worth it, considering how difficult i'd be 
#To make drawing cleaner and reduce overhead we try to merge small strokes together, and with their natural bigger neighbors

#Step 5b – coverage safety pass
#All of the earlier logic sometimes drops some strokes, so we keep a hard pixel record.
#If any of the pixels have been left unasigned we walk them to turn them into strokes

#Step 6 – Catmull–Rom → cubic Bezier conversion
#At this point we have clean polylines (list of [x, y] points). We convert them into
#cubic Bezier segments, which are nicer to render and animate:
#- use centripetal Catmull–Rom splines (alpha = 0.5) to estimate a smooth curve
#  passing through the sample points,
#- for each segment between Pi and Pi+1 we compute control points from local tangents
#  and emit [x0, y0, cx1, cy1, cx2, cy2, x1, y1].



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
JUNC_CONTINUE_ANG_MAX      = 45.0   # deg – how much deflection we still accept

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
MERGE_SMALL_LEN_MIN        = 65.0    # absolute minimum length (px) to consider "not tiny"
MERGE_DIST_FRAC_DIAG       = 0.004  # relative distance gate vs image diagonal
MERGE_DIST_MIN             = 2.0    # px
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
    """
    Load skeleton / edge image and return foreground mask 0/1.

    We assume the input is your skeletonized PNG (0/255). We:
      - Otsu-threshold to be safe,
      - pick minority color as foreground (line).
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))

    if BLUR_BEFORE_BIN and BLUR_BEFORE_BIN % 2 == 1:
        img = cv2.GaussianBlur(img, (BLUR_BEFORE_BIN, BLUR_BEFORE_BIN), 0)

    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = (bw == 255).mean()

    # minority color = foreground
    if white_ratio < 0.5:
        fg = (bw == 255).astype(np.uint8)
    else:
        fg = (bw == 0).astype(np.uint8)

    return fg

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
    h, w = skel01.shape
    deg = np.zeros_like(skel01, dtype=np.uint8)
    ys, xs = np.where(skel01 > 0)
    for y, x in zip(ys, xs):
        c = 0
        for dy, dx in NEIGHBORS_8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel01[ny, nx] > 0:
                c += 1
        deg[y, x] = c
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
    """
    skel = (skel01 > 0).astype(np.uint8)
    h, w = skel.shape
    deg = _build_degree_map(skel)

    visited_edges: set = set()
    strokes: List[np.ndarray] = []

    ys, xs = np.where(skel > 0)

    # pass 1: start from nodes that are NOT pure degree-2 (endpoints, junctions)
    for y, x in zip(ys, xs):
        d = int(deg[y, x])
        if d == 0 or d == 2:
            continue
        for ny, nx in _neighbors_fg(y, x, skel):
            ek = _edge_key((y, x), (ny, nx))
            if ek in visited_edges:
                continue
            stroke = _trace_one_stroke((y, x), (ny, nx), skel, deg, visited_edges)
            if stroke is not None:
                strokes.append(stroke)

    # pass 2: pick up loops made only of degree-2 nodes (circles etc.)
    for y, x in zip(ys, xs):
        if int(deg[y, x]) != 2:
            continue
        for ny, nx in _neighbors_fg(y, x, skel):
            ek = _edge_key((y, x), (ny, nx))
            if ek in visited_edges:
                continue
            stroke = _trace_one_stroke((y, x), (ny, nx), skel, deg, visited_edges)
            if stroke is not None:
                strokes.append(stroke)

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

    for i in range(n):
        a = max(0, i - win_radius)
        b = min(n - 1, i + win_radius)
        if b == a:
            if i < n - 1:
                v = pts[i + 1] - pts[i]
            else:
                v = pts[i] - pts[i - 1]
        else:
            v = pts[b] - pts[a]
        dx, dy = float(v[0]), float(v[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            phi[i] = 0.0
        else:
            phi[i] = math.atan2(dy, dx)
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
    Post-pass: merge small strokes into neighboring strokes to reduce stroke count.

    - define "small" strokes by length percentile (lower 25%) with an absolute floor;
    - only those small strokes try to merge into others;
    - require:
        * endpoints very close in Euclidean distance,
        * direction at the join compatible (angle <= MERGE_ANGLE_MAX);
    - non-merged strokes remain unchanged.
    """
    n = len(polys)
    if n == 0:
        return polys

    # copy to avoid aliasing
    merged = [p.astype(np.float32, copy=True) for p in polys]
    for p in merged:
        nan_to_num_nd(p)

    lengths = [_stroke_length(p) for p in merged]
    len_arr = np.array(lengths, dtype=np.float32)
    if not np.any(len_arr > 0):
        return merged

    # define "small" threshold from distribution
    if n >= 10:
        perc = float(np.percentile(len_arr, MERGE_SMALL_LEN_PERCENTILE))
        small_len_thresh = max(MERGE_SMALL_LEN_MIN, perc)
    else:
        small_len_thresh = max(MERGE_SMALL_LEN_MIN, float(np.min(len_arr)))

    # which indices are small
    small_indices = [i for i, L in enumerate(lengths) if 0.0 < L <= small_len_thresh]
    if not small_indices:
        return merged

    # tight distance gate
    base = MERGE_DIST_FRAC_DIAG * float(diag)
    dist_thresh = max(MERGE_DIST_MIN, min(MERGE_DIST_MAX, base))
    angle_thresh = MERGE_ANGLE_MAX

    active = [True] * n

    # process small strokes from shortest to longest
    small_indices_sorted = sorted(small_indices, key=lambda i: lengths[i])

    for si in small_indices_sorted:
        if not active[si]:
            continue
        p_small = merged[si]
        if p_small is None or p_small.shape[0] < 2:
            active[si] = False
            continue

        # endpoints and tangents for small
        s_start = p_small[0]
        s_end = p_small[-1]
        if p_small.shape[0] >= 2:
            s_tan_start = _unit_vec(p_small[1] - p_small[0])
            s_tan_end = _unit_vec(p_small[-1] - p_small[-2])
        else:
            s_tan_start = np.array([1.0, 0.0], dtype=np.float32)
            s_tan_end = np.array([1.0, 0.0], dtype=np.float32)

        best_target_j = None
        best_new_poly = None
        best_score = None

        # search all other active strokes as potential targets
        for j in range(n):
            if j == si or not active[j]:
                continue
            p_big = merged[j]
            if p_big is None or p_big.shape[0] < 2:
                continue

            b_start = p_big[0]
            b_end = p_big[-1]
            b_tan_start = _unit_vec(p_big[1] - p_big[0])
            b_tan_end = _unit_vec(p_big[-1] - p_big[-2])

            combos = []

            # 1) attach small at end of big: big_end -> small_start
            d = float(np.linalg.norm(b_end - s_start))
            if d <= dist_thresh:
                ang = _angle_between_vecs(b_tan_end, s_tan_start)
                combos.append(("b_end_s_start", d, ang))

            # 2) big_end -> small_end (reverse small)
            d = float(np.linalg.norm(b_end - s_end))
            if d <= dist_thresh:
                ang = _angle_between_vecs(b_tan_end, -s_tan_end)
                combos.append(("b_end_s_end", d, ang))

            # 3) attach small at start of big: small_end -> big_start
            d = float(np.linalg.norm(s_end - b_start))
            if d <= dist_thresh:
                ang = _angle_between_vecs(s_tan_end, b_tan_start)
                combos.append(("s_end_b_start", d, ang))

            # 4) small_start -> big_start (reverse small)
            d = float(np.linalg.norm(s_start - b_start))
            if d <= dist_thresh:
                ang = _angle_between_vecs(-s_tan_start, b_tan_start)
                combos.append(("s_start_b_start", d, ang))

            for tag, d, ang in combos:
                if ang > angle_thresh:
                    continue
                # score: prioritize close and reasonably aligned
                score = float(d + 0.1 * ang)

                if best_score is not None and score >= best_score:
                    continue

                # build candidate merged polyline for this combo
                if tag == "b_end_s_start":
                    if d < 1e-3:
                        new = np.vstack([p_big, p_small[1:]])
                    else:
                        new = np.vstack([p_big, p_small])
                elif tag == "b_end_s_end":
                    s_rev = p_small[::-1]
                    if d < 1e-3:
                        new = np.vstack([p_big, s_rev[1:]])
                    else:
                        new = np.vstack([p_big, s_rev])
                elif tag == "s_end_b_start":
                    if d < 1e-3:
                        new = np.vstack([p_small, p_big[1:]])
                    else:
                        new = np.vstack([p_small, p_big])
                elif tag == "s_start_b_start":
                    s_rev = p_small[::-1]
                    if d < 1e-3:
                        new = np.vstack([s_rev, p_big[1:]])
                    else:
                        new = np.vstack([s_rev, p_big])
                else:
                    continue

                nan_to_num_nd(new)
                if not finite_nd(new) or new.shape[0] < 2:
                    continue

                best_score = score
                best_target_j = j
                best_new_poly = new

        # merge small stroke into best target if found
        if best_target_j is not None and best_new_poly is not None:
            merged[best_target_j] = best_new_poly
            lengths[best_target_j] = _stroke_length(best_new_poly)
            active[si] = False

    # keep only active, non-degenerate polylines
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
        if not math.isfinite(d):
            d = 0.0
        return ti + (d ** alpha)

    beziers: List[List[float]] = []

    P_ext = np.vstack((P[0], P, P[-1]))
    t = [0.0]
    for i in range(1, P_ext.shape[0]):
        t.append(tj(t[-1], P_ext[i-1], P_ext[i]))

    for k in range(1, n + 0):
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

def _process_single(path: Path, output : Path) -> Tuple[str, dict]:
    fg = _load_edges_binary(path)
    H, W = fg.shape
    diag = math.hypot(W, H)

    t0 = time.time()

    sk = (fg > 0).astype(np.uint8)

    # 1) graph-based strokes directly from skeleton
    polys_raw = _trace_strokes_graph(sk)

    # 2) per-stroke corner-aware segmentation
    polys_seg: List[np.ndarray] = []
    for p in polys_raw:
        if p.shape[0] < 2:
            continue
        segments = _segment_polyline_on_corners(p)
        for seg in segments:
            if seg.shape[0] >= CURV_MIN_SEG_POINTS:
                polys_seg.append(seg)

    # 3) conservative simplification
    polys_simpl: List[np.ndarray] = []
    for p in polys_seg:
        if p.shape[0] < 2:
            continue
        simp = _simplify_poly_conservative(p)
        if simp.shape[0] >= 2:
            polys_simpl.append(simp)
        else:
            polys_simpl.append(p)

    # 4) merge small strokes into neighbours (big merge wave)
    polys_merged = _merge_small_strokes(polys_simpl, diag)

    # 5) optional stride
    if SEG_STRIDE > 1:
        polys_final = [
            p[::SEG_STRIDE] if p.shape[0] > SEG_STRIDE else p
            for p in polys_merged
        ]
    else:
        polys_final = polys_merged

     # 5b) coverage-based safety pass: ensure every skeleton pixel
    #      is represented by at least one stroke
    polys_final = _coverage_safety_pass(sk, polys_final)

    # 6) Catmull–Rom → cubic Beziers
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
            strokes.append({"segments": beziers})

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
    output.mkdir(parents=True, exist_ok=True)
    out_json = output / f"{path.stem}.json"
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path), data

# ------------------- MAIN -------------------

def main():
    imgs = sorted([
        p for p in IN_DIR.glob("*")
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    ])
    print(f"[INFO] IN={IN_DIR}  OUT={OUT_DIR}  found={len(imgs)} image(s)")
    if not imgs:
        return
    for p in imgs:
        src, meta = _process_single(p, OUT_DIR)
        print(f"[OK] {Path(src).name}: strokes={len(meta['strokes'])}, "
              f"time={meta['stats']['time_sec']}s")

if __name__ == "__main__":
    main()
