#!/usr/bin/env python3
import os
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont
from ImageVectorizer import _process_single

import shutil
import numpy as np
import cv2
import json


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
# You can edit this section to add / remove characters as needed.

# Uppercase A-Z
LETTERS_UPPER: List[str] = [chr(c) for c in range(ord('A'), ord('Z') + 1)]

# Lowercase a-z
LETTERS_LOWER: List[str] = [chr(c) for c in range(ord('a'), ord('z') + 1)]

# Digits 0-9
DIGITS: List[str] = [str(d) for d in range(10)]

# Common special symbols (no space)
SPECIALS: List[str] = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

# Final accepted characters list
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

    - Same canvas size for all (IMAGE_WIDTH x IMAGE_HEIGHT)
    - Baseline is fixed in the image (using ascent/descent)
    - No per-glyph scaling, no bbox-centering bullshit
    """
    img = Image.new("L", IMAGE_SIZE, BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Get bbox to center horizontally (only)
    bbox = compute_text_bbox(draw, char, font)
    x0, y0, x1, y1 = bbox
    text_width = x1 - x0

    # Baseline in the middle-ish of the image
    baseline_y = IMAGE_HEIGHT // 2

    # In PIL, y passed to text() is the *top* of the text box.
    # Baseline is at y + ascent.
    # So to place baseline at baseline_y:
    y = baseline_y - ascent

    # Horizontal center: adjust for x0 offset in bbox
    x = (IMAGE_WIDTH - text_width) // 2 - x0

    draw.text((x, y), char, font=font, fill=FG_COLOR)
    return img


def skeletonize_pil_image(img: Image.Image) -> Image.Image:
    """
    Skeletonize, but DO NOT crop. We keep the full 2048x2048 canvas so that
    all glyphs live in the same coordinate system; only their strokes differ.
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

    # IMPORTANT: we do NOT crop here anymore.
    # Thin glyphs like 'I' keep the same global canvas/baseline.
    return skel_img


def safe_filename(index: int, char: str) -> str:
    codepoint = ord(char)
    # 4-digit lowercase hex; matches the lookup logic in main.dart
    return f"{codepoint:04x}.png"


def render_all_chars(font_path: Path, output_dir: Path):
    """
    Main routine:
    - Load font.
    - Compute global metrics (ascent, descent).
    - Iterate over ACCEPTED_CHARS.
    - Render each glyph using the *same baseline*.
    - Skeletonize it.
    - Save the skeletonized glyph as PNG with hex-based name.
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

    # dump metrics so the Dart side can use a global scale
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
        # seg expected: [x0, y0, c1x, c1y, c2x, c2y, x1, y1]
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
    """
    For each JSON in vectors_dir:
      - load
      - compute leftmost x for each stroke
      - sort strokes so that leftmost strokes come first (left -> right)
      - save back
    """
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

    # 2) vectorize skeletonized PNGs
    for p in collect_path(PNG_PATH):
        # _process_single keeps basename; so 0041.png -> 0041.json
        _process_single(p, VECTORS_PATH)

    # 3) reorder strokes in all generated JSONs so leftmost strokes come first
    order_strokes_in_vector_jsons(VECTORS_PATH)


if __name__ == "__main__":
    main()
