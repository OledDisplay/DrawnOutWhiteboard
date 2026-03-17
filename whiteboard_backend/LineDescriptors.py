#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


BASE = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE / "StrokeVectors"
DEFAULT_OUTPUT_DIR = BASE / "StrokeDescriptions"

# ------------------- BASIC KNOBS -------------------
CUBIC_SAMPLES_PER_SEGMENT = 12
STRAIGHT_ANGLE_DELTA_DEG = 7.5
MERGE_ADJACENT_STRAIGHT_ANGLE_DEG = 8.0
MERGE_ADJACENT_CURVE_TURN_DEG = 18.0
CURVE_MIN_TOTAL_TURN_DEG = 12.0
CHUNK_MIN_POINTS = 3

# conservative endpoint-touch grouping
ENDPOINT_TOUCH_MAX_PX = 8.0
ENDPOINT_TOUCH_MIN_PX = 2.0
ENDPOINT_TOUCH_FRAC_DIAG = 0.006
GROUP_MAX_MEMBERS = 8

# neighbor reading
NEIGHBOR_LAYER1_COUNT = 2
NEIGHBOR_LAYER2_COUNT = 2

# angle reading
ANGLE_RAY_LOOKAHEAD_POINTS = 3

# output compactness
# Set these to an integer to re-enable truncation limits.
TRACE_SENTENCE_LIMIT: Optional[int] = None
MAX_LINE_DESCRIPTION_CHARS: Optional[int] = None
MAX_GROUP_DESCRIPTION_CHARS: Optional[int] = None


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def angle_diff_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def signed_angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def norm_angle_deg(a: float) -> float:
    out = a % 360.0
    if out < 0.0:
        out += 360.0
    return out


def euclid(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.hypot(dx, dy)


def round2(v: float) -> float:
    return round(float(v), 2)


def pt_to_list(pt: Sequence[float]) -> List[float]:
    return [round2(pt[0]), round2(pt[1])]


def clamp_text(text: str, max_chars: Optional[int]) -> str:
    s = " ".join(str(text or "").split()).strip()
    if max_chars is None or max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def bbox_of_points(pts: np.ndarray) -> Tuple[float, float, float, float]:
    min_x = float(np.min(pts[:, 0]))
    min_y = float(np.min(pts[:, 1]))
    max_x = float(np.max(pts[:, 0]))
    max_y = float(np.max(pts[:, 1]))
    return min_x, min_y, max_x, max_y


def bbox_gap_distance(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0.0, bx1 - ax2, ax1 - bx2)
    dy = max(0.0, by1 - ay2, ay1 - by2)
    return math.hypot(dx, dy)


def polyline_length(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 0.0
    diff = np.diff(pts.astype(np.float64), axis=0)
    seg_lengths = np.linalg.norm(diff, axis=1)
    return float(np.sum(seg_lengths))


def dedupe_points(pts: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    if pts.shape[0] <= 1:
        return pts.astype(np.float32)
    out = [pts[0].astype(np.float32)]
    for i in range(1, pts.shape[0]):
        if euclid(pts[i], out[-1]) > eps:
            out.append(pts[i].astype(np.float32))
    if len(out) == 1:
        out.append(pts[-1].astype(np.float32))
    return np.asarray(out, dtype=np.float32)


def resample_polyline(pts: np.ndarray, sample_count: int) -> np.ndarray:
    pts = dedupe_points(pts)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if pts.shape[0] == 1:
        return np.repeat(pts, sample_count, axis=0)

    seg = np.linalg.norm(np.diff(pts.astype(np.float64), axis=0), axis=1)
    total = float(np.sum(seg))
    if total <= 1e-8:
        return np.repeat(pts[:1], sample_count, axis=0)

    target = np.linspace(0.0, total, sample_count)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    j = 0
    for t in target:
        while j < len(seg) - 1 and cum[j + 1] < t:
            j += 1
        span = cum[j + 1] - cum[j]
        if span <= 1e-8:
            out.append(pts[j].astype(np.float32))
            continue
        r = (t - cum[j]) / span
        p = pts[j] + (pts[j + 1] - pts[j]) * float(r)
        out.append(p.astype(np.float32))
    return np.asarray(out, dtype=np.float32)


def heading_deg_between(a: Sequence[float], b: Sequence[float]) -> float:
    return norm_angle_deg(math.degrees(math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0]))))


def cubic_point(p0: np.ndarray, c1: np.ndarray, c2: np.ndarray, p1: np.ndarray, t: float) -> np.ndarray:
    omt = 1.0 - t
    return (
        (omt ** 3) * p0
        + 3.0 * (omt ** 2) * t * c1
        + 3.0 * omt * (t ** 2) * c2
        + (t ** 3) * p1
    )


def cubic_to_points(seg: Sequence[float], steps: int = CUBIC_SAMPLES_PER_SEGMENT) -> np.ndarray:
    vals = np.asarray(seg, dtype=np.float32)
    if vals.shape[0] != 8:
        raise ValueError(f"Expected 8 values for cubic BÃ©zier, got {vals.shape[0]}")
    p0 = vals[0:2]
    c1 = vals[2:4]
    c2 = vals[4:6]
    p1 = vals[6:8]
    ts = np.linspace(0.0, 1.0, steps)
    pts = [cubic_point(p0, c1, c2, p1, float(t)) for t in ts]
    return dedupe_points(np.asarray(pts, dtype=np.float32))


def stroke_to_polyline(stroke: Dict[str, Any]) -> np.ndarray:
    segments = stroke.get("segments", [])
    all_pts: List[np.ndarray] = []
    for idx, seg in enumerate(segments):
        pts = cubic_to_points(seg, CUBIC_SAMPLES_PER_SEGMENT)
        if pts.shape[0] == 0:
            continue
        if idx > 0 and len(all_pts) > 0 and euclid(all_pts[-1], pts[0]) <= 1e-4:
            pts = pts[1:]
        for p in pts:
            all_pts.append(p.astype(np.float32))
    if not all_pts:
        return np.zeros((0, 2), dtype=np.float32)
    return dedupe_points(np.asarray(all_pts, dtype=np.float32))


def headings_of_polyline(pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32)
    diff = np.diff(pts.astype(np.float64), axis=0)
    out: List[float] = []
    for dx, dy in diff:
        ang = norm_angle_deg(math.degrees(math.atan2(float(dy), float(dx))))
        out.append(float(ang))
    return np.asarray(out, dtype=np.float32)


def direction_phrase(angle_deg: float) -> str:
    a = norm_angle_deg(angle_deg)
    if a < 22.5 or a >= 337.5:
        return "rightward"
    if a < 67.5:
        return "down-right"
    if a < 112.5:
        return "downward"
    if a < 157.5:
        return "down-left"
    if a < 202.5:
        return "leftward"
    if a < 247.5:
        return "up-left"
    if a < 292.5:
        return "upward"
    return "up-right"


def slope_class_phrase(angle_deg: float) -> str:
    a = norm_angle_deg(angle_deg) % 180.0
    if a <= 10.0 or a >= 170.0:
        return "nearly horizontal"
    if 80.0 <= a <= 100.0:
        return "nearly vertical"
    if 25.0 <= a <= 65.0:
        return "moderate diagonal"
    if 115.0 <= a <= 155.0:
        return "moderate diagonal"
    if a < 25.0 or a > 155.0:
        return "shallow diagonal"
    return "steep diagonal"


def tangent_heading(pts: np.ndarray, endpoint: str, outward: bool) -> float:
    if pts.shape[0] < 2:
        return 0.0
    look = min(max(1, ANGLE_RAY_LOOKAHEAD_POINTS), pts.shape[0] - 1)
    if endpoint == "start":
        if outward:
            return heading_deg_between(pts[0], pts[look])
        return heading_deg_between(pts[look], pts[0])
    if outward:
        return heading_deg_between(pts[-1], pts[-1 - look])
    return heading_deg_between(pts[-1 - look], pts[-1])


@dataclass
class TraceChunk:
    kind: str
    start_idx: int
    end_idx: int
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    length: float
    heading_deg: float
    total_turn_deg: float
    signed_turn_deg: float
    phrase: str
    details: Dict[str, Any]


@dataclass
class PreparedStroke:
    line_index: int
    stroke_index: int
    stroke: Dict[str, Any]
    polyline: np.ndarray
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    start: Tuple[float, float]
    end: Tuple[float, float]
    length: float
    chord_heading_deg: float
    start_heading_deg: float
    end_heading_deg: float
    start_out_heading_deg: float
    end_out_heading_deg: float


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int, max_size: Optional[int] = None) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        if max_size is not None and self.size[ra] + self.size[rb] > max_size:
            return False
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True



def straight_phrase(start_pt: np.ndarray, end_pt: np.ndarray, angle_deg: float, length: float, diag: float) -> str:
    slope = slope_class_phrase(angle_deg)
    direction = direction_phrase(angle_deg)
    rel_len = length / max(diag, 1e-6)
    if rel_len < 0.04:
        size = "short"
    elif rel_len < 0.12:
        size = "medium-length"
    else:
        size = "long"
    return f"{size} {slope} straight run heading {direction}" 


def curve_style_phrase(abs_turn: float, monotonicity: float, rel_len: float, signed_turn: float) -> str:
    rot = "clockwise" if signed_turn > 0 else "counterclockwise"
    if abs_turn < 22.0:
        base = f"slight {rot} bend"
    elif abs_turn < 45.0:
        base = f"gentle {rot} arc"
    elif abs_turn < 80.0:
        base = f"balanced {rot} curve"
    elif abs_turn < 135.0:
        base = f"strong {rot} sweep"
    else:
        base = f"tight {rot} hook"

    if monotonicity >= 0.88:
        balance = "clean"
    elif monotonicity >= 0.68:
        balance = "slightly uneven"
    elif monotonicity >= 0.50:
        balance = "uneven"
    else:
        balance = "mixed"

    if rel_len < 0.04:
        span = "short"
    elif rel_len < 0.12:
        span = "medium"
    else:
        span = "long"

    return f"{span} {balance} {base}"



def chunk_polyline_trace(pts: np.ndarray, image_diag: float) -> List[TraceChunk]:
    pts = dedupe_points(pts)
    if pts.shape[0] < 2:
        return []
    if pts.shape[0] == 2:
        heading = float(headings_of_polyline(pts)[0])
        length = euclid(pts[0], pts[1])
        phrase = straight_phrase(pts[0], pts[1], heading, length, image_diag)
        return [
            TraceChunk(
                kind="straight",
                start_idx=0,
                end_idx=1,
                start_point=(float(pts[0, 0]), float(pts[0, 1])),
                end_point=(float(pts[1, 0]), float(pts[1, 1])),
                length=float(length),
                heading_deg=round2(heading),
                total_turn_deg=0.0,
                signed_turn_deg=0.0,
                phrase=phrase,
                details={"slope_class": slope_class_phrase(heading), "direction": direction_phrase(heading)},
            )
        ]

    headings = headings_of_polyline(pts)
    delta = np.zeros((headings.shape[0],), dtype=np.float32)
    for i in range(1, headings.shape[0]):
        delta[i] = float(signed_angle_diff_deg(float(headings[i]), float(headings[i - 1])))

    step_labels: List[str] = []
    for i in range(headings.shape[0]):
        if i == 0:
            local = abs(float(delta[1])) if headings.shape[0] > 1 else 0.0
        else:
            local = abs(float(delta[i]))
        step_labels.append("straight" if local <= STRAIGHT_ANGLE_DELTA_DEG else "curve")

    spans: List[Tuple[int, int, str]] = []
    start = 0
    current = step_labels[0]
    for i in range(1, len(step_labels)):
        if step_labels[i] != current:
            spans.append((start, i, current))
            start = i
            current = step_labels[i]
    spans.append((start, len(step_labels), current))

    cleaned: List[Tuple[int, int, str]] = []
    for span in spans:
        s, e, kind = span
        point_count = e - s + 1
        if cleaned and point_count < CHUNK_MIN_POINTS:
            ps, pe, pk = cleaned[-1]
            cleaned[-1] = (ps, e, pk)
        else:
            cleaned.append(span)
    spans = cleaned

    def _build_chunk_from_slice(point_slice: np.ndarray, kind_hint: str, start_idx: int, end_idx: int) -> TraceChunk:
        local_head = headings_of_polyline(point_slice)
        if local_head.shape[0] > 0:
            heading = float(local_head[-1])
        else:
            heading = 0.0
        length = polyline_length(point_slice)

        if point_slice.shape[0] >= 3 and local_head.shape[0] >= 2:
            signed_turns = [
                float(signed_angle_diff_deg(float(local_head[j]), float(local_head[j - 1])))
                for j in range(1, local_head.shape[0])
            ]
            signed_total = float(np.sum(signed_turns)) if signed_turns else 0.0
            abs_total = float(np.sum(np.abs(signed_turns))) if signed_turns else 0.0
            monotonicity = abs(signed_total) / abs_total if abs_total > 1e-6 else 1.0
        else:
            signed_total = 0.0
            abs_total = 0.0
            monotonicity = 1.0

        kind = kind_hint
        if kind == "curve" and abs_total < CURVE_MIN_TOTAL_TURN_DEG:
            kind = "straight"

        if kind == "straight":
            phrase = straight_phrase(point_slice[0], point_slice[-1], heading, length, image_diag)
            details = {
                "slope_class": slope_class_phrase(heading),
                "direction": direction_phrase(heading),
            }
            signed_total = 0.0
            abs_total = 0.0
        else:
            rel_len = length / max(image_diag, 1e-6)
            phrase = curve_style_phrase(abs_total, monotonicity, rel_len, signed_total)
            details = {
                "rotation": "clockwise" if signed_total > 0 else "counterclockwise",
                "curve_balance": (
                    "clean" if monotonicity >= 0.88 else
                    "slightly uneven" if monotonicity >= 0.68 else
                    "uneven" if monotonicity >= 0.50 else
                    "mixed"
                ),
                "dominant_heading": direction_phrase(heading),
                "monotonicity": round2(monotonicity),
            }

        return TraceChunk(
            kind=kind,
            start_idx=start_idx,
            end_idx=end_idx,
            start_point=(float(point_slice[0, 0]), float(point_slice[0, 1])),
            end_point=(float(point_slice[-1, 0]), float(point_slice[-1, 1])),
            length=round2(length),
            heading_deg=round2(heading),
            total_turn_deg=round2(abs_total),
            signed_turn_deg=round2(signed_total),
            phrase=phrase,
            details=details,
        )

    chunks: List[TraceChunk] = []
    for s, e, kind in spans:
        point_slice = pts[s:e + 1]
        if point_slice.shape[0] < 2:
            continue
        chunks.append(_build_chunk_from_slice(point_slice, kind, s, e))

    if not chunks:
        heading = float(headings[0])
        length = polyline_length(pts)
        phrase = straight_phrase(pts[0], pts[-1], heading, length, image_diag)
        return [
            TraceChunk(
                kind="straight",
                start_idx=0,
                end_idx=pts.shape[0] - 1,
                start_point=(float(pts[0, 0]), float(pts[0, 1])),
                end_point=(float(pts[-1, 0]), float(pts[-1, 1])),
                length=round2(length),
                heading_deg=round2(heading),
                total_turn_deg=0.0,
                signed_turn_deg=0.0,
                phrase=phrase,
                details={"slope_class": slope_class_phrase(heading), "direction": direction_phrase(heading)},
            )
        ]

    merged: List[TraceChunk] = [chunks[0]]
    for ch in chunks[1:]:
        prev = merged[-1]
        if prev.kind != ch.kind:
            merged.append(ch)
            continue

        can_merge = False
        if ch.kind == "straight":
            if angle_diff_deg(prev.heading_deg, ch.heading_deg) <= MERGE_ADJACENT_STRAIGHT_ANGLE_DEG:
                can_merge = True
        else:
            if (
                (prev.signed_turn_deg == 0.0 or ch.signed_turn_deg == 0.0 or np.sign(prev.signed_turn_deg) == np.sign(ch.signed_turn_deg))
                and abs(prev.total_turn_deg - ch.total_turn_deg) <= MERGE_ADJACENT_CURVE_TURN_DEG
            ):
                can_merge = True

        if not can_merge:
            merged.append(ch)
            continue

        start_idx = prev.start_idx
        end_idx = ch.end_idx
        merged_slice = pts[start_idx:end_idx + 1]
        if merged_slice.shape[0] >= 2:
            merged[-1] = _build_chunk_from_slice(merged_slice, prev.kind, start_idx, end_idx)
        else:
            merged.append(ch)

    return merged



def overall_turn_metrics(pts: np.ndarray) -> Tuple[float, float]:
    headings = headings_of_polyline(pts)
    if headings.shape[0] < 2:
        return 0.0, 0.0
    signed_turns = [float(signed_angle_diff_deg(float(headings[i]), float(headings[i - 1]))) for i in range(1, headings.shape[0])]
    signed_total = float(np.sum(signed_turns)) if signed_turns else 0.0
    abs_total = float(np.sum(np.abs(signed_turns))) if signed_turns else 0.0
    return abs_total, signed_total



def prepare_strokes(data: Dict[str, Any]) -> List[PreparedStroke]:
    prepared: List[PreparedStroke] = []
    for idx, stroke in enumerate(data.get("strokes", [])):
        poly = stroke_to_polyline(stroke)
        if poly.shape[0] < 2:
            continue
        bbox = bbox_of_points(poly)
        centroid_np = np.mean(poly.astype(np.float64), axis=0)
        start = (float(poly[0, 0]), float(poly[0, 1]))
        end = (float(poly[-1, 0]), float(poly[-1, 1]))
        length = polyline_length(poly)
        chord_heading = heading_deg_between(poly[0], poly[-1])
        prepared.append(
            PreparedStroke(
                line_index=len(prepared),
                stroke_index=idx,
                stroke=stroke,
                polyline=poly,
                bbox=bbox,
                centroid=(float(centroid_np[0]), float(centroid_np[1])),
                start=start,
                end=end,
                length=float(length),
                chord_heading_deg=chord_heading,
                start_heading_deg=tangent_heading(poly, "start", outward=True),
                end_heading_deg=tangent_heading(poly, "end", outward=False),
                start_out_heading_deg=tangent_heading(poly, "start", outward=True),
                end_out_heading_deg=tangent_heading(poly, "end", outward=True),
            )
        )
    return prepared



def touch_threshold(width: int, height: int) -> float:
    diag = math.hypot(width, height)
    return clamp(diag * ENDPOINT_TOUCH_FRAC_DIAG, ENDPOINT_TOUCH_MIN_PX, ENDPOINT_TOUCH_MAX_PX)



def line_endpoint_records(strokes: List[PreparedStroke]) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for s in strokes:
        recs.append({
            "record_index": len(recs),
            "line_index": s.line_index,
            "source_stroke_index": s.stroke_index,
            "endpoint_name": "start",
            "point": s.start,
            "out_heading_deg": s.start_out_heading_deg,
            "travel_heading_deg": s.start_heading_deg,
        })
        recs.append({
            "record_index": len(recs),
            "line_index": s.line_index,
            "source_stroke_index": s.stroke_index,
            "endpoint_name": "end",
            "point": s.end,
            "out_heading_deg": s.end_out_heading_deg,
            "travel_heading_deg": s.end_heading_deg,
        })
    return recs



def pairwise_touch_edges(strokes: List[PreparedStroke], width: int, height: int) -> List[Dict[str, Any]]:
    thresh = touch_threshold(width, height)
    edges: List[Dict[str, Any]] = []
    for i in range(len(strokes)):
        a = strokes[i]
        for j in range(i + 1, len(strokes)):
            b = strokes[j]
            candidates = [
                ("start", "start", a.start, b.start, a.start_out_heading_deg, b.start_out_heading_deg),
                ("start", "end", a.start, b.end, a.start_out_heading_deg, b.end_out_heading_deg),
                ("end", "start", a.end, b.start, a.end_out_heading_deg, b.start_out_heading_deg),
                ("end", "end", a.end, b.end, a.end_out_heading_deg, b.end_out_heading_deg),
            ]
            best = None
            for a_ep, b_ep, pa, pb, ha, hb in candidates:
                d = euclid(pa, pb)
                if d > thresh:
                    continue
                meet = ((float(pa[0]) + float(pb[0])) * 0.5, (float(pa[1]) + float(pb[1])) * 0.5)
                inter_angle = angle_diff_deg(ha, hb)
                cand = {
                    "a": i,
                    "b": j,
                    "a_endpoint": a_ep,
                    "b_endpoint": b_ep,
                    "distance": round2(d),
                    "junction_point": pt_to_list(meet),
                    "angle_between_deg": round2(inter_angle),
                    "a_out_heading_deg": round2(ha),
                    "b_out_heading_deg": round2(hb),
                }
                if best is None or d < best["distance"] or (d == best["distance"] and inter_angle < best["angle_between_deg"]):
                    best = cand
            if best is not None:
                edges.append(best)
    edges.sort(key=lambda e: (e["distance"], e["angle_between_deg"], e["a"], e["b"]))
    return edges



def group_strokes_by_touch(strokes: List[PreparedStroke], width: int, height: int) -> Tuple[List[Dict[str, Any]], Dict[int, int], List[Dict[str, Any]]]:
    n = len(strokes)
    if n == 0:
        return [], {}, []

    edges = pairwise_touch_edges(strokes, width, height)
    dsu = DSU(n)
    accepted_edges: List[Dict[str, Any]] = []
    for edge in edges:
        if dsu.union(edge["a"], edge["b"], max_size=GROUP_MAX_MEMBERS):
            accepted_edges.append(edge)

    members_by_root: Dict[int, List[int]] = {}
    for i in range(n):
        root = dsu.find(i)
        members_by_root.setdefault(root, []).append(i)

    groups: List[Dict[str, Any]] = []
    line_to_group: Dict[int, int] = {}
    for gid, member_ids in enumerate(sorted(members_by_root.values(), key=lambda arr: arr[0])):
        member_ids = sorted(member_ids)
        member_strokes = [strokes[i] for i in member_ids]
        group_pts = np.vstack([s.polyline for s in member_strokes])
        bbox = bbox_of_points(group_pts)
        centroid_np = np.mean(group_pts.astype(np.float64), axis=0)
        group_edges = [e for e in accepted_edges if e["a"] in member_ids and e["b"] in member_ids]
        groups.append({
            "group_index": gid,
            "member_line_indices": member_ids,
            "source_stroke_indices": [strokes[i].stroke_index for i in member_ids],
            "bbox": bbox,
            "centroid": (float(centroid_np[0]), float(centroid_np[1])),
            "edge_connections": group_edges,
            "max_group_members_limit": GROUP_MAX_MEMBERS,
        })
        for i in member_ids:
            line_to_group[i] = gid

    return groups, line_to_group, accepted_edges



def centered_coords(pt: Tuple[float, float], width: int, height: int) -> Tuple[float, float]:
    x = float(pt[0]) - width / 2.0
    y = height / 2.0 - float(pt[1])
    return x, y



def quadrant_name(centered: Tuple[float, float], width: int, height: int) -> str:
    x, y = centered
    tol_x = width * 0.04
    tol_y = height * 0.04
    if abs(x) <= tol_x and abs(y) <= tol_y:
        return "center-cross"
    if abs(x) <= tol_x:
        return "on vertical axis band"
    if abs(y) <= tol_y:
        return "on horizontal axis band"
    if x > 0 and y > 0:
        return "quadrant I (top-right)"
    if x < 0 and y > 0:
        return "quadrant II (top-left)"
    if x < 0 and y < 0:
        return "quadrant III (bottom-left)"
    return "quadrant IV (bottom-right)"



def quadrant_center_for(centered: Tuple[float, float], width: int, height: int) -> Tuple[float, float]:
    x, y = centered
    qx = width / 4.0 if x >= 0 else -width / 4.0
    qy = height / 4.0 if y >= 0 else -height / 4.0
    return qx, qy



def distance_band(distance: float, diag: float) -> str:
    r = distance / max(diag, 1e-6)
    if r < 0.035:
        return "very close"
    if r < 0.08:
        return "close"
    if r < 0.18:
        return "moderately separated"
    if r < 0.32:
        return "far"
    return "very far"



def edge_relation(bbox: Tuple[float, float, float, float], width: int, height: int) -> Dict[str, Any]:
    min_x, min_y, max_x, max_y = bbox
    distances = {
        "left": float(min_x),
        "top": float(min_y),
        "right": float(width - max_x),
        "bottom": float(height - max_y),
    }
    nearest = min(distances, key=distances.get)
    edge_band = "near edge" if distances[nearest] <= 0.08 * max(width, height) else "interior"
    return {
        "nearest_edge": nearest,
        "edge_distances": {k: round2(v) for k, v in distances.items()},
        "edge_band": edge_band,
        "phrase": f"{edge_band}, nearest to the {nearest} boundary",
    }



def axis_relation(centered_centroid: Tuple[float, float], bbox: Tuple[float, float, float, float], width: int, height: int) -> Dict[str, Any]:
    cx, cy = centered_centroid
    min_x, min_y, max_x, max_y = bbox
    crosses_vertical = min_x <= width / 2.0 <= max_x
    crosses_horizontal = min_y <= height / 2.0 <= max_y
    dist_y_axis = abs(cx)
    dist_x_axis = abs(cy)

    if crosses_vertical and crosses_horizontal:
        phrase = "crosses both central axes"
    elif crosses_vertical:
        phrase = "crosses the vertical center axis"
    elif crosses_horizontal:
        phrase = "crosses the horizontal center axis"
    else:
        x_band = "near" if dist_x_axis <= 0.08 * height else "away from"
        y_band = "near" if dist_y_axis <= 0.08 * width else "away from"
        phrase = f"{x_band} the horizontal axis and {y_band} the vertical axis"

    return {
        "crosses_vertical_axis": crosses_vertical,
        "crosses_horizontal_axis": crosses_horizontal,
        "distance_to_vertical_axis": round2(dist_y_axis),
        "distance_to_horizontal_axis": round2(dist_x_axis),
        "phrase": phrase,
    }



def location_summary_for_line(poly: np.ndarray, bbox: Tuple[float, float, float, float], width: int, height: int) -> Dict[str, Any]:
    centroid_np = np.mean(poly.astype(np.float64), axis=0)
    centroid = (float(centroid_np[0]), float(centroid_np[1]))
    centered = centered_coords(centroid, width, height)
    quad = quadrant_name(centered, width, height)
    q_center = quadrant_center_for(centered, width, height)
    q_dist = euclid(centered, q_center)
    diag = math.hypot(width, height)

    axis = axis_relation(centered, bbox, width, height)
    edge = edge_relation(bbox, width, height)

    qdx = centered[0] - q_center[0]
    qdy = centered[1] - q_center[1]
    side_x = "right of" if qdx > 0 else "left of"
    side_y = "above" if qdy > 0 else "below"
    if abs(qdx) <= width * 0.03:
        side_x = "aligned with"
    if abs(qdy) <= height * 0.03:
        side_y = "aligned with"

    location_phrase = (
        f"Centroid sits in {quad}, with centered coordinates {pt_to_list(centered)}; "
        f"{axis['phrase']}; {edge['phrase']}; "
        f"relative to its quadrant center it is {distance_band(q_dist, diag)} and {side_x} that center, {side_y} it vertically."
    )

    return {
        "centroid": pt_to_list(centroid),
        "centroid_centered": pt_to_list(centered),
        "quadrant": quad,
        "axis_relation": axis,
        "quadrant_center": {
            "point_centered": pt_to_list(q_center),
            "distance": round2(q_dist),
            "distance_band": distance_band(q_dist, diag),
            "horizontal_relation": side_x,
            "vertical_relation": side_y,
        },
        "edge_relation": edge,
        "phrase": location_phrase,
    }



def line_size_summary(poly: np.ndarray, bbox: Tuple[float, float, float, float], width: int, height: int, peer_lengths: List[float]) -> Dict[str, Any]:
    length = polyline_length(poly)
    diag = math.hypot(width, height)
    rel = length / max(diag, 1e-6)
    min_x, min_y, max_x, max_y = bbox
    span_w = max_x - min_x
    span_h = max_y - min_y
    area = span_w * span_h

    if rel < 0.04:
        size_band = "very short"
    elif rel < 0.09:
        size_band = "short"
    elif rel < 0.18:
        size_band = "medium"
    elif rel < 0.32:
        size_band = "long"
    else:
        size_band = "very long"

    if span_w > span_h * 1.45:
        span_type = "horizontally dominant"
    elif span_h > span_w * 1.45:
        span_type = "vertically dominant"
    else:
        span_type = "balanced span"

    if peer_lengths:
        longer_than = sum(1 for x in peer_lengths if length > x)
        percentile = 100.0 * longer_than / max(len(peer_lengths), 1)
    else:
        percentile = 100.0

    if percentile >= 85.0:
        peer_rank = "among the longest lines"
    elif percentile >= 60.0:
        peer_rank = "longer than most nearby lines"
    elif percentile >= 40.0:
        peer_rank = "middle-sized compared with peers"
    elif percentile >= 15.0:
        peer_rank = "shorter than most peers"
    else:
        peer_rank = "among the smallest lines"

    phrase = (
        f"Overall size is {size_band}; arc length {round2(length)} px, bounding span {round2(span_w)}Ã—{round2(span_h)} px, "
        f"{span_type}, {peer_rank}."
    )

    return {
        "length_px": round2(length),
        "length_vs_image_diag": round2(rel),
        "bbox_width": round2(span_w),
        "bbox_height": round2(span_h),
        "bbox_area": round2(area),
        "size_band": size_band,
        "span_type": span_type,
        "peer_percentile_estimate": round2(percentile),
        "peer_rank_phrase": peer_rank,
        "phrase": phrase,
    }



def relative_direction_phrase(source: Tuple[float, float], target: Tuple[float, float]) -> str:
    dx = target[0] - source[0]
    dy = target[1] - source[1]
    horiz = ""
    vert = ""
    if abs(dx) > 1e-6:
        horiz = "right" if dx > 0 else "left"
    if abs(dy) > 1e-6:
        vert = "below" if dy > 0 else "above"

    if horiz and vert:
        return f"{horiz} and {vert}"
    if horiz:
        return horiz
    if vert:
        return vert
    return "coincident"



def tracing_summary(line: PreparedStroke, width: int, height: int) -> Dict[str, Any]:
    diag = math.hypot(width, height)
    chunks = chunk_polyline_trace(line.polyline, diag)
    abs_turn, signed_turn = overall_turn_metrics(line.polyline)

    chunk_phrases = [str(ch.phrase) for ch in chunks if str(ch.phrase).strip()]
    if TRACE_SENTENCE_LIMIT is None or TRACE_SENTENCE_LIMIT <= 0:
        preview = list(chunk_phrases)
    else:
        preview = chunk_phrases[:TRACE_SENTENCE_LIMIT]
    omitted = max(0, len(chunk_phrases) - len(preview))
    if preview:
        trace_profile = "; ".join(preview)
        if omitted > 0:
            trace_profile += f"; plus {omitted} additional trace sections"
    else:
        trace_profile = "no stable trace sections"

    start_ray = {
        "heading_deg": round2(line.start_heading_deg),
        "direction": direction_phrase(line.start_heading_deg),
    }
    end_ray = {
        "heading_deg": round2(line.end_heading_deg),
        "direction": direction_phrase(line.end_heading_deg),
    }
    overall_angle_phrase = (
        f"Starts {direction_phrase(line.start_heading_deg)} ({round2(line.start_heading_deg)} deg) and ends "
        f"{direction_phrase(line.end_heading_deg)} ({round2(line.end_heading_deg)} deg); chord heading "
        f"{round2(line.chord_heading_deg)} deg, accumulated turn {round2(abs_turn)} deg."
    )

    trace_phrase_cap: Optional[int] = None
    if MAX_LINE_DESCRIPTION_CHARS is not None and MAX_LINE_DESCRIPTION_CHARS > 0:
        trace_phrase_cap = MAX_LINE_DESCRIPTION_CHARS // 2
    phrase = clamp_text(
        f"Trace profile: {trace_profile}. {overall_angle_phrase}".strip(),
        trace_phrase_cap,
    )

    return {
        "chunk_count": len(chunk_phrases),
        "trace_signature": preview,
        "trace_signature_omitted": omitted,
        "overall_angles": {
            "start_heading_deg": round2(line.start_heading_deg),
            "end_heading_deg": round2(line.end_heading_deg),
            "chord_heading_deg": round2(line.chord_heading_deg),
            "start_ray": start_ray,
            "end_ray": end_ray,
            "absolute_total_turn_deg": round2(abs_turn),
            "signed_total_turn_deg": round2(signed_turn),
            "phrase": overall_angle_phrase,
        },
        "phrase": phrase,
    }


def line_neighbor_relation(source: PreparedStroke, target: PreparedStroke, width: int, height: int) -> Dict[str, Any]:
    src_centroid = source.centroid
    trg_centroid = target.centroid
    diag = math.hypot(width, height)
    center_distance = euclid(src_centroid, trg_centroid)
    bbox_distance = bbox_gap_distance(source.bbox, target.bbox)
    direction = relative_direction_phrase(src_centroid, trg_centroid)

    size_a = source.length
    size_b = target.length
    if size_b > size_a * 1.15:
        size_relation = "larger"
    elif size_b < size_a * 0.85:
        size_relation = "smaller"
    else:
        size_relation = "similar in size"

    angle_gap = angle_diff_deg(source.chord_heading_deg, target.chord_heading_deg)
    if angle_gap <= 12.0:
        orientation_relation = "roughly parallel"
    elif 78.0 <= angle_gap <= 102.0:
        orientation_relation = "roughly perpendicular"
    else:
        orientation_relation = "differently oriented"

    phrase = (
        f"{direction}, {distance_band(center_distance, diag)} by centroid and {distance_band(bbox_distance, diag)} by box gap; "
        f"the neighbor is {size_relation} and {orientation_relation}."
    )

    return {
        "target_line_index": target.line_index,
        "target_source_stroke_index": target.stroke_index,
        "relative_direction": direction,
        "centroid_distance": round2(center_distance),
        "bbox_gap_distance": round2(bbox_distance),
        "centroid_distance_band": distance_band(center_distance, diag),
        "bbox_gap_band": distance_band(bbox_distance, diag),
        "size_relation": size_relation,
        "orientation_relation": orientation_relation,
        "phrase": phrase,
    }



def build_full_description(trace: Dict[str, Any], size: Dict[str, Any], location: Dict[str, Any], neighbors: Dict[str, Any]) -> str:
    parts = [trace["phrase"], size["phrase"], location["phrase"]]
    layer1 = neighbors.get("layer1", [])
    layer2 = neighbors.get("layer2", [])
    if layer1:
        parts.append("Nearest neighbors: " + " ".join(n["phrase"] for n in layer1))
    if layer2:
        parts.append("Second-layer neighbors: " + " ".join(n["phrase"] for n in layer2))
    return clamp_text(" ".join(p for p in parts if p).strip(), MAX_LINE_DESCRIPTION_CHARS)



def endpoint_clusters_for_group(group: Dict[str, Any], strokes: List[PreparedStroke], width: int, height: int) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, str], int]]:
    member_ids = group["member_line_indices"]
    thresh = touch_threshold(width, height)
    recs: List[Dict[str, Any]] = []
    for line_idx in member_ids:
        s = strokes[line_idx]
        recs.append({
            "record_index": len(recs),
            "line_index": line_idx,
            "endpoint_name": "start",
            "point": s.start,
            "out_heading_deg": s.start_out_heading_deg,
        })
        recs.append({
            "record_index": len(recs),
            "line_index": line_idx,
            "endpoint_name": "end",
            "point": s.end,
            "out_heading_deg": s.end_out_heading_deg,
        })

    dsu = DSU(len(recs))
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            if euclid(recs[i]["point"], recs[j]["point"]) <= thresh:
                dsu.union(i, j)

    by_root: Dict[int, List[Dict[str, Any]]] = {}
    for rec in recs:
        by_root.setdefault(dsu.find(rec["record_index"]), []).append(rec)

    clusters: List[Dict[str, Any]] = []
    rec_to_cluster: Dict[Tuple[int, str], int] = {}
    for cid, members in enumerate(by_root.values()):
        xs = [m["point"][0] for m in members]
        ys = [m["point"][1] for m in members]
        center = (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))
        incident_lines = sorted(set(m["line_index"] for m in members))
        clusters.append({
            "cluster_index": cid,
            "center": center,
            "member_count": len(members),
            "incident_line_indices": incident_lines,
            "members": members,
        })
        for m in members:
            rec_to_cluster[(m["line_index"], m["endpoint_name"])] = cid

    return clusters, rec_to_cluster



def describe_two_way_junction(cluster: Dict[str, Any]) -> Dict[str, Any]:
    members = cluster["members"]
    m1, m2 = members[0], members[1]
    ang = angle_diff_deg(m1["out_heading_deg"], m2["out_heading_deg"])
    if ang <= 20.0:
        shape_name = "narrow merge"
        desc = f"Two-line near-parallel merge with only {round2(ang)}Â° between the rays."
    elif ang <= 70.0:
        shape_name = "acute corner"
        desc = f"Two-line acute turn with a junction angle of {round2(ang)}Â°."
    elif ang <= 110.0:
        shape_name = "right-angle corner"
        desc = f"Two-line right-angle style turn with a junction angle of {round2(ang)}Â°."
    elif ang <= 160.0:
        shape_name = "obtuse corner"
        desc = f"Two-line obtuse corner with a junction angle of {round2(ang)}Â°."
    else:
        shape_name = "straight continuation"
        desc = f"Two-line near-straight continuation with rays separated by {round2(ang)}Â°."
    return {
        "shape_type": "shared_endpoint_junction",
        "shape_name": shape_name,
        "line_indices": cluster["incident_line_indices"],
        "junction_point": pt_to_list(cluster["center"]),
        "junction_member_count": 2,
        "pairwise_angles_deg": [round2(ang)],
        "description": desc,
    }



def describe_multiway_junction(cluster: Dict[str, Any]) -> Dict[str, Any]:
    members = cluster["members"]
    angles = sorted([norm_angle_deg(m["out_heading_deg"]) for m in members])
    diffs = []
    for i in range(len(angles)):
        nxt = angles[(i + 1) % len(angles)]
        cur = angles[i]
        diff = (nxt - cur) % 360.0
        diffs.append(diff)

    n = len(members)
    if n == 3:
        if any(abs(d - 180.0) <= 25.0 for d in diffs):
            shape_name = "T-junction"
            desc = "Three-line T-style junction with one near-opposite pair and one branch entering from the side."
        elif max(abs(d - 120.0) for d in diffs) <= 25.0:
            shape_name = "Y-junction"
            desc = "Three-line balanced Y-style junction with roughly even angular spacing."
        else:
            shape_name = "3-way junction"
            desc = "Three-line junction with uneven branch spacing."
    elif n == 4:
        rightish = sum(1 for d in diffs if 65.0 <= d <= 115.0)
        oppositeish = sum(1 for d in diffs if abs(d - 180.0) <= 20.0)
        if rightish >= 3:
            shape_name = "cross junction"
            desc = "Four-line cross-style junction with roughly orthogonal branch spacing."
        elif oppositeish >= 2:
            shape_name = "4-way aligned junction"
            desc = "Four-line junction with opposite branch pairs and aligned rays."
        else:
            shape_name = "4-way junction"
            desc = "Four-line junction with irregular branch spacing."
    else:
        shape_name = f"{n}-way junction"
        desc = f"Multi-line junction with {n} incident lines."

    pairwise = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            pairwise.append(round2(angle_diff_deg(members[i]["out_heading_deg"], members[j]["out_heading_deg"])))

    return {
        "shape_type": "shared_endpoint_junction",
        "shape_name": shape_name,
        "line_indices": cluster["incident_line_indices"],
        "junction_point": pt_to_list(cluster["center"]),
        "junction_member_count": n,
        "pairwise_angles_deg": sorted(pairwise),
        "description": desc,
    }



def polygon_like_name(node_count: int, aspect_ratio: float) -> str:
    if node_count == 3:
        return "triangle"
    if node_count == 4:
        if 0.8 <= aspect_ratio <= 1.25:
            return "square-like loop"
        return "quadrilateral loop"
    if node_count == 5:
        return "pentagon-like loop"
    if node_count == 6:
        return "hexagon-like loop"
    return f"{node_count}-sided polygon-like loop"



def infer_group_shapes(group: Dict[str, Any], strokes: List[PreparedStroke], width: int, height: int) -> Dict[str, Any]:
    clusters, rec_to_cluster = endpoint_clusters_for_group(group, strokes, width, height)
    member_ids = group["member_line_indices"]
    bbox = group["bbox"]
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    aspect_ratio = bbox_w / max(bbox_h, 1e-6)

    shapes: List[Dict[str, Any]] = []

    for cl in clusters:
        incident = cl["incident_line_indices"]
        if len(incident) < 2:
            continue
        if len(incident) == 2:
            shapes.append(describe_two_way_junction(cl))
        else:
            shapes.append(describe_multiway_junction(cl))

    node_adjacency: Dict[int, List[Tuple[int, int]]] = {cl["cluster_index"]: [] for cl in clusters}
    component_line_ids: List[int] = []
    for line_idx in member_ids:
        a = rec_to_cluster.get((line_idx, "start"))
        b = rec_to_cluster.get((line_idx, "end"))
        if a is None or b is None:
            continue
        component_line_ids.append(line_idx)
        if a != b:
            node_adjacency.setdefault(a, []).append((b, line_idx))
            node_adjacency.setdefault(b, []).append((a, line_idx))

    visited_nodes = set()
    for node in node_adjacency.keys():
        if node in visited_nodes:
            continue
        stack = [node]
        comp_nodes = []
        comp_lines = set()
        while stack:
            cur = stack.pop()
            if cur in visited_nodes:
                continue
            visited_nodes.add(cur)
            comp_nodes.append(cur)
            for nxt, lid in node_adjacency.get(cur, []):
                comp_lines.add(lid)
                if nxt not in visited_nodes:
                    stack.append(nxt)
        if len(comp_nodes) < 3 or len(comp_lines) < 3:
            continue
        degrees = [len(node_adjacency[n]) for n in comp_nodes]
        if all(d == 2 for d in degrees) and len(comp_lines) == len(comp_nodes):
            name = polygon_like_name(len(comp_nodes), aspect_ratio)
            shapes.append({
                "shape_type": "far_endpoint_loop",
                "shape_name": name,
                "line_indices": sorted(comp_lines),
                "node_cluster_indices": sorted(comp_nodes),
                "description": f"Far-endpoint closed loop consistent with a {name}, spanning the group bounding box.",
            })

    endpoint_cluster_json = []
    for cl in clusters:
        endpoint_cluster_json.append({
            "cluster_index": cl["cluster_index"],
            "center": pt_to_list(cl["center"]),
            "member_count": cl["member_count"],
            "incident_line_indices": cl["incident_line_indices"],
            "members": [
                {
                    "line_index": m["line_index"],
                    "endpoint_name": m["endpoint_name"],
                    "point": pt_to_list(m["point"]),
                    "out_heading_deg": round2(m["out_heading_deg"]),
                }
                for m in cl["members"]
            ],
        })

    shape_summary = " ".join(shape["description"] for shape in shapes) if shapes else "No stable group-level shape inference was found beyond the loose endpoint-touch grouping."
    return {
        "bbox": {
            "min_x": round2(bbox[0]),
            "min_y": round2(bbox[1]),
            "max_x": round2(bbox[2]),
            "max_y": round2(bbox[3]),
            "width": round2(bbox_w),
            "height": round2(bbox_h),
        },
        "endpoint_clusters": endpoint_cluster_json,
        "shapes": shapes,
        "shape_summary": shape_summary,
    }



def describe_lines_and_groups(strokes: List[PreparedStroke], groups: List[Dict[str, Any]], line_to_group: Dict[int, int], width: int, height: int, source_json_name: str) -> Dict[str, Any]:
    if not strokes:
        return {
            "schema_version": 2,
            "schema_name": "stroke_descriptions_compact_v2",
            "source_json": source_json_name,
            "width": width,
            "height": height,
            "described_lines": [],
            "groups": [],
            "index_maps": {
                "line_to_stroke_index": {},
                "stroke_to_line_index": {},
            },
            "stats": {
                "line_count": 0,
                "group_count": 0,
                "touch_threshold_px": round2(touch_threshold(width, height)),
                "max_group_members_limit": GROUP_MAX_MEMBERS,
            },
        }

    peer_lengths = [float(s.length) for s in strokes]
    described_lines: List[Dict[str, Any]] = []

    for idx, line in enumerate(strokes):
        trace = tracing_summary(line, width, height)
        size = line_size_summary(line.polyline, line.bbox, width, height, peer_lengths)
        location = location_summary_for_line(line.polyline, line.bbox, width, height)

        other_indices = [j for j in range(len(strokes)) if j != idx]
        other_indices.sort(key=lambda j: euclid(line.centroid, strokes[j].centroid))
        layer1_ids = other_indices[:NEIGHBOR_LAYER1_COUNT]
        layer2_ids = other_indices[NEIGHBOR_LAYER1_COUNT: NEIGHBOR_LAYER1_COUNT + NEIGHBOR_LAYER2_COUNT]
        neighbors = {
            "layer1": [line_neighbor_relation(line, strokes[j], width, height) for j in layer1_ids],
            "layer2": [line_neighbor_relation(line, strokes[j], width, height) for j in layer2_ids],
        }

        full_text = build_full_description(trace, size, location, neighbors)
        group_idx = line_to_group.get(idx)
        described_lines.append({
            "described_line_index": int(idx),
            "source_stroke_index": int(line.stroke_index),
            "group_index": int(group_idx) if group_idx is not None else None,
            "centroid": pt_to_list(line.centroid),
            "description": full_text,
        })

    described_lines.sort(key=lambda r: (float(r["centroid"][0]), float(r["centroid"][1]), int(r["described_line_index"])))
    for rank, row in enumerate(described_lines):
        row["left_to_right_rank"] = int(rank)

    described_groups: List[Dict[str, Any]] = []
    for group in groups:
        shape_info = infer_group_shapes(group, strokes, width, height)
        described_groups.append({
            "group_index": int(group["group_index"]),
            "member_line_indices": [int(i) for i in group["member_line_indices"]],
            "source_stroke_indices": [int(i) for i in group["source_stroke_indices"]],
            "centroid": pt_to_list(group["centroid"]),
            "description": clamp_text(str(shape_info["shape_summary"]), MAX_GROUP_DESCRIPTION_CHARS),
        })

    described_groups.sort(key=lambda g: (min(g["member_line_indices"]) if g["member_line_indices"] else 10**9, int(g["group_index"])))

    line_to_stroke_index = {
        str(int(r["described_line_index"])): int(r["source_stroke_index"])
        for r in described_lines
    }
    stroke_to_line_index = {
        str(int(r["source_stroke_index"])): int(r["described_line_index"])
        for r in described_lines
    }

    return {
        "schema_version": 2,
        "schema_name": "stroke_descriptions_compact_v2",
        "source_json": source_json_name,
        "width": width,
        "height": height,
        "described_lines": described_lines,
        "groups": described_groups,
        "index_maps": {
            "line_to_stroke_index": line_to_stroke_index,
            "stroke_to_line_index": stroke_to_line_index,
        },
        "stats": {
            "line_count": len(described_lines),
            "group_count": len(described_groups),
            "touch_threshold_px": round2(touch_threshold(width, height)),
            "max_group_members_limit": GROUP_MAX_MEMBERS,
        },
    }


def describe_vectorizer_json(json_path: Path) -> Dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    width = int(data.get("width", 0))
    height = int(data.get("height", 0))
    strokes = prepare_strokes(data)
    groups, line_to_group, _ = group_strokes_by_touch(strokes, width, height)
    return describe_lines_and_groups(strokes, groups, line_to_group, width, height, json_path.name)



def iter_input_jsons(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.json"), key=lambda p: p.name.lower())


def describe_vectorizer_jsons_batch(
    json_paths: List[Path],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_workers: int = 8,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [Path(p) for p in (json_paths or []) if Path(p).is_file()]
    if not files:
        return {"ok": False, "error": "no_input_json_files", "written": [], "errors": []}

    workers = max(1, min(int(max_workers or 1), len(files)))
    written: List[str] = []
    errors: List[Dict[str, str]] = []

    def _describe_one(src: Path) -> Tuple[Path, Path, Dict[str, Any]]:
        described = describe_vectorizer_json(src)
        out_path = output_dir / f"{src.stem}_described.json"
        out_path.write_text(json.dumps(described, ensure_ascii=False, indent=2), encoding="utf-8")
        return src, out_path, described

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="line_desc") as ex:
        futs = {ex.submit(_describe_one, src): src for src in files}
        for fut in as_completed(futs):
            src = futs[fut]
            try:
                src_path, out_path, described = fut.result()
                written.append(str(out_path))
                print(
                    f"[OK] {src_path.name} -> {out_path.name} "
                    f"lines={described['stats']['line_count']} groups={described['stats']['group_count']}"
                )
            except Exception as e:
                errors.append({"source": str(src), "error": f"{type(e).__name__}: {e}"})
                print(f"[ERR] {src.name}: {type(e).__name__}: {e}")

    written.sort()
    return {
        "ok": len(written) > 0 and not errors,
        "written": written,
        "errors": errors,
        "count": len(written),
    }



def run(input_path: Path = DEFAULT_INPUT_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    json_files = iter_input_jsons(input_path)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {input_path}")

    summary = describe_vectorizer_jsons_batch(
        json_files,
        output_dir=output_dir,
        max_workers=8,
    )
    if summary.get("errors"):
        raise RuntimeError(f"Line descriptor generation had errors: {summary['errors'][:3]}")


if __name__ == "__main__":
    run()


