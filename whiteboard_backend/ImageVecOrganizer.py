#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

BASE = Path(__file__).resolve().parent
JSON_DIR = BASE / "StrokeVectors"   # same folder your vectorizer writes to


#Take our ready vectors (strokes) and organizes them in groups for drawing - like a human would
#We try to link hotspots of vectors together, and big objects with small ones around them
#We just order the drawing around a couple of centralized areas, instead of just random blobs popping up
#We also try to go from the center a little bit
#Right now we have some pretty tight grouping settings.
#Set to these values for "default" grouping:
#       GROUP_NEAR_DIST_FRAC = 0.03
#       GROUP_FAR_DIST_FRAC  = 0.06



# ----------------- TUNABLE KNOBS -----------------
# All distances are relative to image diagonal so they scale with resolution.

# How far a stroke center can be from a cluster center to be absorbed
# near radius is for normal / big strokes, far radius is for small strokes.

GROUP_NEAR_DIST_FRAC = 0.04    # ~3% of diagonal
GROUP_FAR_DIST_FRAC  = 0.05    # ~6% of diagonal

# Percentile to define "small" stroke by length (used to allow bigger attach radius)
SMALL_LEN_PERCENTILE = 35.0    # strokes below this length percentile are "small"

# Safety: if a stroke is extremely tiny (almost a dot) we just leave it where it is
ABS_MIN_STROKE_LEN   = 1.0     # px

# -----------------------------------------------


def _finite_nd(a: np.ndarray) -> bool:
    return np.isfinite(a).all()


def _stroke_points(stroke: Dict[str, Any]) -> np.ndarray:
    """
    Extract a polyline of points from a stroke made of cubic Bezier segments.
    We only care about the endpoints for clustering: [x0,y0] of first segment,
    and [x1,y1] of each segment.
    """
    segs = stroke.get("segments", [])
    if not segs:
        return np.zeros((0, 2), dtype=np.float32)

    pts = []
    for idx, seg in enumerate(segs):
        if not seg or len(seg) < 8:
            continue
        x0, y0, _, _, _, _, x1, y1 = seg
        if idx == 0:
            pts.append([float(x0), float(y0)])
        pts.append([float(x1), float(y1)])

    if not pts:
        return np.zeros((0, 2), dtype=np.float32)

    arr = np.asarray(pts, dtype=np.float32)
    arr = np.nan_to_num(arr, copy=False)
    return arr


def _stroke_length_from_points(pts: np.ndarray) -> float:
    """
    Approximate stroke length from a polyline of points.
    """
    if pts.shape[0] < 2:
        return 0.0
    diff = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(diff, axis=1)
    return float(np.sum(seg_len))


def _compute_stroke_features(
    strokes: List[Dict[str, Any]],
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each stroke, compute:
      - length[i]  : approximate length in pixels
      - centers[i] : (cx, cy)
      - dist_ctr[i]: distance from image center
    All returned as numpy arrays of shape:
      length:   (N,)
      centers:  (N, 2)
      dist_ctr: (N,)
    """
    N = len(strokes)
    length = np.zeros(N, dtype=np.float32)
    centers = np.zeros((N, 2), dtype=np.float32)

    cx_img = width * 0.5
    cy_img = height * 0.5

    for i, s in enumerate(strokes):
        pts = _stroke_points(s)
        if pts.shape[0] == 0:
            length[i] = 0.0
            centers[i] = [cx_img, cy_img]
            continue

        L = _stroke_length_from_points(pts)
        length[i] = float(L)

        c = np.mean(pts, axis=0)
        centers[i, 0] = float(c[0])
        centers[i, 1] = float(c[1])

    dx = centers[:, 0] - cx_img
    dy = centers[:, 1] - cy_img
    dist_ctr = np.sqrt(dx * dx + dy * dy).astype(np.float32)

    return length, centers, dist_ctr


def _build_clusters(
    length: np.ndarray,
    centers: np.ndarray,
    dist_ctr: np.ndarray,
    width: int,
    height: int,
) -> List[List[int]]:
    """
    Greedy spatial clustering:

    - strokes are seeded in order of:
         1) longer first
         2) if tie, closer to image center first
    - each seed creates a cluster
    - then we pull in unassigned strokes whose centers are close enough
      to the cluster center:
        * "near" radius for normal/big strokes
        * "far" radius for small strokes only
    - cluster center is updated as length-weighted average of member centers

    Returns:
       clusters: list of lists of stroke indices
    """
    N = length.shape[0]
    if N == 0:
        return []

    diag = math.hypot(width, height)
    near_r = float(GROUP_NEAR_DIST_FRAC * diag)
    far_r = float(GROUP_FAR_DIST_FRAC * diag)

    # define "small stroke" by length percentile
    len_pos = length[length > ABS_MIN_STROKE_LEN]
    if len_pos.size > 0:
        small_thresh = float(np.percentile(len_pos, SMALL_LEN_PERCENTILE))
    else:
        small_thresh = float(ABS_MIN_STROKE_LEN)

    # indices of strokes in seed order: big & central first
    indices = np.arange(N, dtype=int)
    seed_order = sorted(
        indices,
        key=lambda i: (-float(length[i]), float(dist_ctr[i])),
    )

    unassigned = set(indices.tolist())
    clusters: List[List[int]] = []

    for seed in seed_order:
        if seed not in unassigned:
            continue

        # start new cluster with this seed
        cluster_inds = [seed]
        unassigned.remove(seed)

        # cluster center = length-weighted center
        total_L = float(max(length[seed], ABS_MIN_STROKE_LEN))
        c_x = float(centers[seed, 0])
        c_y = float(centers[seed, 1])

        changed = True
        while changed:
            changed = False
            # snapshot so we can iterate while mutating unassigned
            candidates = list(unassigned)
            for j in candidates:
                # distance from stroke center to current cluster center
                dx = float(centers[j, 0] - c_x)
                dy = float(centers[j, 1] - c_y)
                d = math.hypot(dx, dy)

                Lj = float(length[j])
                if Lj < ABS_MIN_STROKE_LEN:
                    # degenerate / dot-like stroke, skip; it will form its own tiny cluster
                    continue

                # decide allowable radius
                if Lj <= small_thresh:
                    # small strokes get more generosity in distance
                    allowed_r = far_r
                else:
                    allowed_r = near_r

                if d <= allowed_r:
                    # absorb stroke j into this cluster
                    cluster_inds.append(j)
                    unassigned.remove(j)

                    old_total = total_L
                    total_L = old_total + Lj

                    # update length-weighted cluster center
                    c_x = (c_x * old_total + centers[j, 0] * Lj) / total_L
                    c_y = (c_y * old_total + centers[j, 1] * Lj) / total_L

                    changed = True

        clusters.append(cluster_inds)

    return clusters


def _order_clusters_and_strokes(
    clusters: List[List[int]],
    length: np.ndarray,
    centers: np.ndarray,
    dist_ctr: np.ndarray,
    width: int,
    height: int,
) -> List[int]:
    """
    Given clusters (as lists of stroke indices) and per-stroke features,
    return a single global ordering of stroke indices.

    - clusters are ordered primarily by TOTAL stroke length (biggest groups first),
      then by distance of their centroid to image center (center-outwards).
    - inside each cluster, strokes are ordered:
        1) longer first (big structure first),
        2) if tie, closer to image center.
    """
    if not clusters:
        return []

    cx_img = width * 0.5
    cy_img = height * 0.5

    cluster_info = []
    for ci, idxs in enumerate(clusters):
        if not idxs:
            continue

        # cluster centroid
        pts = centers[idxs]
        c = np.mean(pts, axis=0)
        dx = float(c[0] - cx_img)
        dy = float(c[1] - cy_img)
        d_center = math.hypot(dx, dy)

        # total stroke length in this cluster
        total_len = float(np.sum(length[idxs]))

        # store: cluster index, -total_len (for descending), distance to center
        cluster_info.append((ci, -total_len, d_center))

    # big clusters first, then nearer to center
    cluster_info.sort(key=lambda t: (t[1], t[2]))

    ordered_indices: List[int] = []

    for ci, _, _ in cluster_info:
        idxs = clusters[ci]
        if not idxs:
            continue
        # inside cluster: big, central strokes first
        idxs_sorted = sorted(
            idxs,
            key=lambda i: (-float(length[i]), float(dist_ctr[i])),
        )
        ordered_indices.extend(idxs_sorted)

    return ordered_indices


def reorder_strokes_in_file(path: Path) -> None:
    """
    Load one JSON from `path`, reorder `data["strokes"]` in place,
    and save back to the same file.
    """
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)

    strokes = data.get("strokes")
    if not isinstance(strokes, list) or len(strokes) == 0:
        # nothing useful to do
        return

    # width/height must be in the JSON; if missing, we bail out.
    W = int(data.get("width", 0))
    H = int(data.get("height", 0))
    if W <= 0 or H <= 0:
        return

    # compute per-stroke features
    length, centers, dist_ctr = _compute_stroke_features(strokes, W, H)

    # build clusters
    clusters = _build_clusters(length, centers, dist_ctr, W, H)

    # get global ordering of stroke indices
    new_order = _order_clusters_and_strokes(
        clusters,
        length,
        centers,
        dist_ctr,
        W,
        H,
    )

    if not new_order:
        # if for some reason clustering failed, don't touch file
        return

    # apply reordering to strokes
    reordered = [strokes[i] for i in new_order]
    data["strokes"] = reordered

    # keep stats.curves consistent, if stats exists
    stats = data.get("stats")
    if isinstance(stats, dict):
        stats["curves"] = len(reordered)

    # write back in-place
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[OK] reordered {path.name}: strokes={len(reordered)}, clusters={len(clusters)}")


def main():
    files = sorted(
        [p for p in JSON_DIR.glob("*.json")],
        key=lambda p: p.name.lower(),
    )
    print(f"[INFO] JSON_DIR={JSON_DIR}  found={len(files)} json file(s)")
    if not files:
        return

    for p in files:
        reorder_strokes_in_file(p)


if __name__ == "__main__":
    main()
