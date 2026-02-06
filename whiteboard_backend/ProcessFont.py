#!/usr/bin/env python3
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from PIL import Image, ImageDraw, ImageFont

import shutil
import numpy as np
import cv2
import json

import ImageVectorizer as IV  # using your new vectorizer internals


FONT_PATH = Path("./StartStory.otf")
PNG_PATH = Path("./FontPngs")
VECTORS_PATH = Path("./Font")
METRICS_PATH = VECTORS_PATH / "font_metrics.json"


# -----------------------------
# Configuration
# -----------------------------

# 2K image size (square)
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 2048
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

# Colors: black background, white glyph
BG_COLOR = 0          # for mode "L" (grayscale): 0 = black
FG_COLOR = 255        # 255 = white

# Font size as a fraction of image height
FONT_SCALE = 0.25     # 25% of height -> 2048 * 0.25 = 512
FONT_SIZE = int(IMAGE_HEIGHT * FONT_SCALE)


# -----------------------------
# Accepted characters list
# -----------------------------

LETTERS_UPPER: List[str] = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
LETTERS_LOWER: List[str] = [chr(c) for c in range(ord('a'), ord('z') + 1)]
DIGITS: List[str] = [str(d) for d in range(10)]
SPECIALS: List[str] = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

ACCEPTED_CHARS: List[str] = LETTERS_UPPER + LETTERS_LOWER + DIGITS + SPECIALS


# -----------------------------
# Core logic
# -----------------------------

def load_font(font_path: Path) -> ImageFont.FreeTypeFont:
    """Load the TTF/OTF font with the configured FONT_SIZE."""
    if not font_path.is_file():
        raise FileNotFoundError(f"Font file not found: {font_path}")
    return ImageFont.truetype(str(font_path), FONT_SIZE)


def compute_text_bbox(draw: ImageDraw.ImageDraw, char: str, font: ImageFont.FreeTypeFont):
    """
    Compute bounding box of the character.

    Returns (x0, y0, x1, y1). Uses textbbox when available; falls back to textsize.
    """
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
    except AttributeError:
        w, h = draw.textsize(char, font=font)
        bbox = (0, 0, w, h)
    return bbox


def render_char_image(char: str, font: ImageFont.FreeTypeFont,
                      ascent: int, descent: int) -> Image.Image:
    """
    Create a 2K grayscale image with black background and a glyph drawn
    using a *fixed baseline* for all characters.
    """
    img = Image.new("L", IMAGE_SIZE, BG_COLOR)
    draw = ImageDraw.Draw(img)

    bbox = compute_text_bbox(draw, char, font)
    x0, y0, x1, y1 = bbox
    text_width = x1 - x0

    baseline_y = IMAGE_HEIGHT // 2
    y = baseline_y - ascent
    x = (IMAGE_WIDTH - text_width) // 2 - x0

    draw.text((x, y), char, font=font, fill=FG_COLOR)
    return img


def skeletonize_pil_image(img: Image.Image) -> Image.Image:
    """
    Skeletonize, but DO NOT crop. We keep the full 2048x2048 canvas so that
    all glyphs live in the same coordinate system.
    """
    arr = np.array(img)

    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    binary = np.where(arr > 127, 255, 0).astype(np.uint8)

    try:
        from cv2.ximgproc import thinning, THINNING_ZHANGSUEN
        skel = thinning(binary, thinningType=THINNING_ZHANGSUEN)
        skel_img = Image.fromarray(skel, mode="L")
    except Exception:
        try:
            from skimage.morphology import thin as sk_thin
        except Exception as e:
            raise RuntimeError(
                "No available skeletonizer: cv2.ximgproc.thinning and skimage.morphology.thin both unavailable."
            ) from e

        mask = binary > 0
        skel_bool = sk_thin(mask)
        skel_arr = (skel_bool.astype(np.uint8)) * 255
        skel_img = Image.fromarray(skel_arr, mode="L")

    return skel_img


def safe_filename(index: int, char: str) -> str:
    codepoint = ord(char)
    return f"{codepoint:04x}.png"


def render_all_chars(font_path: Path, output_dir: Path):
    """
    - Load font.
    - Compute global metrics.
    - Render each glyph using fixed baseline.
    - Skeletonize.
    - Save hex PNG names.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(font_path)

    ascent, descent = font.getmetrics()
    print(f"[font] ascent={ascent}, descent={descent}, total={ascent+descent}")

    for idx, ch in enumerate(ACCEPTED_CHARS):
        img = render_char_image(ch, font, ascent, descent)
        skel_img = skeletonize_pil_image(img)

        filename = safe_filename(idx, ch)
        out_path = output_dir / filename
        skel_img.save(out_path, format="PNG")
        print(f"[png] '{ch}' -> {out_path}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "image_width": IMAGE_WIDTH,
                "image_height": IMAGE_HEIGHT,
                "font_size_px": FONT_SIZE,
                "ascent_px": ascent,
                "descent_px": descent,
                "line_height_px": ascent + descent,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[font] wrote metrics to {METRICS_PATH}")


def collect_path(inpath: Path):
    imgs = sorted(
        [
            p for p in inpath.glob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        ],
        key=lambda p: p.name.lower(),
    )
    return imgs


# -----------------------------
# Endpoint join logic (LIFTED)
# -----------------------------
# This is the "extra step at the end" from StrokeBundleMerger,
# stripped down to ONLY endpoint joining (no overlap/cut system).

JOIN_ENABLE = True

JOIN_MAX_PASSES = 6
JOIN_DIST_MAX = 8.0
JOIN_DIST_MIN = 0.0

JOIN_ANGLE_MAX = 75.0
JOIN_ANGLE_WEIGHT = 0.15

JOIN_TANGENT_K = 6
JOIN_GLOBAL_DIR_BLEND = 0.30

JOIN_SNAP_ENDPOINTS = True
JOIN_SNAP_MAX_DIST = 3.0

JOIN_INSERT_BRIDGE_POINT = True
JOIN_BRIDGE_MAX_DIST = 8.0


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
    return float(np.degrees(np.arccos(c)))


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


def _concat_polys_for_join(a: np.ndarray, b: np.ndarray, rev_a: bool, rev_b: bool) -> Optional[np.ndarray]:
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


def _endpoint_join_polylines(polys: List[np.ndarray]) -> List[np.ndarray]:
    """
    Endpoint join pass:
    - builds a spatial grid of endpoints
    - repeatedly merges the best match by (distance + angle)
    """
    if not JOIN_ENABLE or len(polys) < 2:
        return polys

    # Clean + drop garbage
    active: Dict[int, np.ndarray] = {}
    next_id = 1
    order: List[int] = []
    for p in polys:
        if p is None or p.shape[0] < 2:
            continue
        q = _poly_clean_consecutive_duplicates(p)
        if q is None or q.shape[0] < 2:
            continue
        active[next_id] = q
        order.append(next_id)
        next_id += 1

    if len(order) < 2:
        return [active[i] for i in order]

    cell = int(max(4.0, float(JOIN_DIST_MAX) * 2.0))

    def cell_for_pt(pt: np.ndarray) -> Tuple[int, int]:
        return (int(pt[0] // cell), int(pt[1] // cell))

    def endpoints(rec: np.ndarray):
        p0 = rec[0].astype(np.float32)
        p1 = rec[-1].astype(np.float32)
        t0 = _endpoint_direction(rec, at_start=True)
        t1 = _endpoint_direction(rec, at_start=False)
        return p0, p1, t0, t1

    def build_grid() -> Dict[Tuple[int, int], List[int]]:
        g: Dict[Tuple[int, int], List[int]] = {}
        for sid in order:
            rec = active.get(sid)
            if rec is None:
                continue
            p0, p1, t0, t1 = endpoints(rec)
            g.setdefault(cell_for_pt(p0), []).append(sid)
            g.setdefault(cell_for_pt(p1), []).append(sid)
        return g

    def best_candidate() -> Optional[Tuple[int, int, bool, bool, float, float, float]]:
        """
        Returns:
        (keep_id, drop_id, rev_keep, rev_drop, dist, ang, score)
        """
        grid = build_grid()

        dist_max = float(JOIN_DIST_MAX)
        dist_min = float(JOIN_DIST_MIN)
        angle_max = float(JOIN_ANGLE_MAX)

        cfgs = [
            (False, False),
            (False, True),
            (True,  False),
            (True,  True),
        ]

        best = None
        best_score = None

        for keep_id in order:
            a = active.get(keep_id)
            if a is None:
                continue

            a_p0, a_p1, a_t0, a_t1 = endpoints(a)

            cand_ids = set()
            for pt in (a_p0, a_p1):
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

                b_p0, b_p1, b_t0, b_t1 = endpoints(b)

                best_local = None
                best_local_score = None

                # helper to pick the correct endpoint and tangent after reversing
                def end_dir_after_rev(t0, t1, rev: bool) -> np.ndarray:
                    return (-t0) if rev else t1

                def start_dir_after_rev(t0, t1, rev: bool) -> np.ndarray:
                    return (-t1) if rev else t0

                for rev_a, rev_b in cfgs:
                    pa = a_p0 if rev_a else a_p1
                    pb = b_p1 if rev_b else b_p0

                    d = float(np.linalg.norm(pb - pa))
                    if d < dist_min or d > dist_max:
                        continue

                    ta = end_dir_after_rev(a_t0, a_t1, rev_a)
                    tb = start_dir_after_rev(b_t0, b_t1, rev_b)

                    ang = _angle_between_vecs(ta, tb)
                    if ang > angle_max:
                        continue

                    score = float(d + JOIN_ANGLE_WEIGHT * ang)

                    if best_local_score is None or score < best_local_score:
                        best_local_score = score
                        best_local = (rev_a, rev_b, d, ang, score)

                if best_local is None:
                    continue

                rev_a, rev_b, d, ang, score = best_local
                if best_score is None or score < best_score:
                    best_score = score
                    best = (keep_id, drop_id, rev_a, rev_b, d, ang, score)

        return best

    for _ in range(int(JOIN_MAX_PASSES)):
        merged_any = False

        while True:
            cand = best_candidate()
            if cand is None:
                break

            keep_id, drop_id, rev_keep, rev_drop, dist, ang, score = cand
            a = active.get(keep_id)
            b = active.get(drop_id)
            if a is None or b is None:
                break

            merged = _concat_polys_for_join(a, b, rev_keep, rev_drop)
            if merged is None or merged.shape[0] < 2:
                break

            active[keep_id] = merged
            active.pop(drop_id, None)

            try:
                order.remove(drop_id)
            except ValueError:
                pass

            merged_any = True

        if not merged_any:
            break

    return [active[i] for i in order if i in active]


# -----------------------------
# Stroke ordering (left -> right)
# -----------------------------

def _stroke_min_x_polyline(stroke: dict) -> float:
    pts = stroke.get("points") or []
    xs = [float(p[0]) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
    return min(xs) if xs else 0.0


def _stroke_min_x_cubic(stroke: dict) -> float:
    segs = stroke.get("segments") or []
    xs: List[float] = []
    for seg in segs:
        if not isinstance(seg, (list, tuple)):
            continue
        if len(seg) >= 2:
            xs.append(float(seg[0]))
        if len(seg) >= 4:
            xs.append(float(seg[2]))
        if len(seg) >= 6:
            xs.append(float(seg[4]))
        if len(seg) >= 8:
            xs.append(float(seg[6]))
    return min(xs) if xs else 0.0


def order_strokes_in_vector_jsons(vectors_dir: Path):
    if not vectors_dir.is_dir():
        print(f"[order] No vector dir: {vectors_dir}")
        return

    for json_path in sorted(vectors_dir.glob("*.json"), key=lambda p: p.name.lower()):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[order] Failed to load {json_path}: {e}")
            continue

        strokes = data.get("strokes")
        if not isinstance(strokes, list) or not strokes:
            continue

        first = strokes[0]
        is_poly = isinstance(first, dict) and "points" in first
        is_cubic = isinstance(first, dict) and "segments" in first

        if not (is_poly or is_cubic):
            print(f"[order] Unknown stroke format in {json_path.name}, skipping")
            continue

        if is_poly:
            def key_fn(s: dict) -> float:
                return _stroke_min_x_polyline(s)
        else:
            def key_fn(s: dict) -> float:
                return _stroke_min_x_cubic(s)

        try:
            strokes_sorted = sorted(strokes, key=key_fn)
        except Exception as e:
            print(f"[order] Failed to sort strokes in {json_path.name}: {e}")
            continue

        data["strokes"] = strokes_sorted

        try:
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            print(f"[order] Failed to save {json_path.name}: {e}")
            continue

        print(f"[order] Reordered strokes left->right in {json_path.name}")


# -----------------------------
# Vectorize ONE glyph PNG using new vectorizer internals,
# but add endpoint-join in between.
# -----------------------------

def _vectorize_single_glyph_png(png_path: Path, out_dir: Path):
    """
    This keeps the exact naming scheme:
      0041.png -> 0041.json

    Pipeline:
      skeleton PNG -> polylines (new tracer) -> endpoint join -> bezier -> json
    """
    W, H, polys = IV._process_single_to_polylines(png_path)

    # endpoint join (lifted)
    polys = _endpoint_join_polylines(polys)

    strokes = []
    for poly in polys:
        if poly is None or poly.shape[0] < 2:
            continue
        beziers = IV._catmull_rom_to_beziers(poly, alpha=IV.CATMULL_ALPHA)
        if not beziers:
            continue
        strokes.append({"segments": beziers, "color_group_id": 0})

    data = {
        "version": 15,
        "source_image": str(png_path),
        "width": int(W),
        "height": int(H),
        "vector_format": "bezier_cubic",
        "strokes": strokes,
        "stats": {
            "curves": int(len(strokes)),
            "endpoint_join_enabled": bool(JOIN_ENABLE),
            "endpoint_join_max_passes": int(JOIN_MAX_PASSES),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{png_path.stem}.json"
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[vec] {png_path.name} -> {out_json.name} (strokes={len(strokes)})")


# -----------------------------
# Main
# -----------------------------

def main():
    if PNG_PATH.exists():
        shutil.rmtree(PNG_PATH)
    if VECTORS_PATH.exists():
        shutil.rmtree(VECTORS_PATH)

    PNG_PATH.mkdir(parents=True, exist_ok=True)
    VECTORS_PATH.mkdir(parents=True, exist_ok=True)

    # 1) render + skeletonize using a fixed baseline (no cropping)
    render_all_chars(FONT_PATH, PNG_PATH)

    # 2) vectorize each skeleton PNG independently
    for p in collect_path(PNG_PATH):
        _vectorize_single_glyph_png(p, VECTORS_PATH)

    # 3) reorder strokes in all generated JSONs so leftmost strokes come first
    order_strokes_in_vector_jsons(VECTORS_PATH)


if __name__ == "__main__":
    main()
