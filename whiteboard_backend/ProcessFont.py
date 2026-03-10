#!/usr/bin/env python3
import json
import shutil
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

import ImageVectorizer as IV


FONT_PATH = Path("./Luculine.otf")
PNG_PATH = Path("./FontPngs")
VECTORS_PATH = Path("./Font")
METRICS_PATH = VECTORS_PATH / "font_metrics.json"


# -----------------------------
# Configuration
# -----------------------------

IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 2048
FONT_SCALE = 0.25
FONT_SIZE = int(IMAGE_HEIGHT * FONT_SCALE)


# -----------------------------
# Accepted characters list
# -----------------------------

LETTERS_UPPER: List[str] = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
LETTERS_LOWER: List[str] = [chr(c) for c in range(ord("a"), ord("z") + 1)]
DIGITS: List[str] = [str(d) for d in range(10)]
SPECIALS: List[str] = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

ACCEPTED_CHARS: List[str] = LETTERS_UPPER + LETTERS_LOWER + DIGITS + SPECIALS


def load_font(font_path: Path) -> ImageFont.FreeTypeFont:
    if not font_path.is_file():
        raise FileNotFoundError(f"Font file not found: {font_path}")
    return ImageFont.truetype(str(font_path), FONT_SIZE)


def safe_stem(char: str) -> str:
    return f"{ord(char):04x}"


def safe_png_name(char: str) -> str:
    return f"{safe_stem(char)}.png"


def safe_json_name(char: str) -> str:
    return f"{safe_stem(char)}.json"


def _render_filled_glyph(char: str, font: ImageFont.FreeTypeFont, ascent: int) -> np.ndarray:
    img = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
    draw = ImageDraw.Draw(img)

    try:
        x0, y0, x1, y1 = draw.textbbox((0, 0), char, font=font)
    except AttributeError:
        w, h = draw.textsize(char, font=font)
        x0, y0, x1, y1 = 0, 0, w, h

    text_width = x1 - x0
    baseline_y = IMAGE_HEIGHT // 2
    x = (IMAGE_WIDTH - text_width) // 2 - x0
    y = baseline_y - int(ascent)

    # Step 1: render filled OTF outline to grayscale canvas.
    draw.text((x, y), char, font=font, fill=255)

    arr = np.asarray(img, dtype=np.uint8)
    return np.where(arr > 127, 255, 0).astype(np.uint8)


def _skeletonize_mask(mask_u8: np.ndarray) -> np.ndarray:
    # Step 2: skeletonize with scikit-image.
    sk = skeletonize(mask_u8 > 0)
    return (sk.astype(np.uint8) * 255)


def _stroke_min_x_cubic(stroke: dict) -> float:
    segs = stroke.get("segments") or []
    xs = []
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


def _vectorize_skeleton_png(png_path: Path, char: str) -> dict:
    # Step 3: mirror ImageVectorizer main flow for a single image.
    width, height, polys = IV._process_single_to_polylines(png_path)

    strokes = []
    for poly in polys:
        if poly is None or getattr(poly, "shape", None) is None or poly.shape[0] < 2:
            continue

        beziers = IV._catmull_rom_to_beziers(poly, alpha=IV.CATMULL_ALPHA)
        if not beziers:
            q = poly.astype(float)
            segs = []
            for i in range(q.shape[0] - 1):
                a = q[i]
                b = q[i + 1]
                c1 = a + (b - a) / 3.0
                c2 = a + 2.0 * (b - a) / 3.0
                seg = [
                    float(a[0]),
                    float(a[1]),
                    float(c1[0]),
                    float(c1[1]),
                    float(c2[0]),
                    float(c2[1]),
                    float(b[0]),
                    float(b[1]),
                ]
                if IV.finite_seq(seg):
                    segs.append(seg)
            beziers = segs

        if beziers:
            strokes.append({"segments": beziers, "color_group_id": 0})

    strokes = sorted(strokes, key=_stroke_min_x_cubic)
    total_segments = sum(len(s.get("segments") or []) for s in strokes)

    return {
        "version": 15,
        "source_image": str(FONT_PATH),
        "width": int(width),
        "height": int(height),
        "vector_format": "bezier_cubic",
        "strokes": strokes,
        "stats": {
            "curves": int(len(strokes)),
            "segments": int(total_segments),
            "char": char,
            "converter": "pillow_fill_skimage_skeleton_imagevectorizer",
        },
    }


def write_font_metrics(font: ImageFont.FreeTypeFont) -> None:
    ascent_px, descent_px = font.getmetrics()
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "image_width": int(IMAGE_WIDTH),
                "image_height": int(IMAGE_HEIGHT),
                "font_size_px": int(FONT_SIZE),
                "ascent_px": int(ascent_px),
                "descent_px": int(descent_px),
                "line_height_px": int(ascent_px + descent_px),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[font] wrote metrics to {METRICS_PATH}")


def main() -> None:
    if PNG_PATH.exists():
        shutil.rmtree(PNG_PATH)
    if VECTORS_PATH.exists():
        shutil.rmtree(VECTORS_PATH)
    PNG_PATH.mkdir(parents=True, exist_ok=True)
    VECTORS_PATH.mkdir(parents=True, exist_ok=True)

    font = load_font(FONT_PATH)
    write_font_metrics(font)
    ascent, _ = font.getmetrics()

    for ch in ACCEPTED_CHARS:
        filled = _render_filled_glyph(ch, font, ascent)
        skel = _skeletonize_mask(filled)

        skel_path = PNG_PATH / safe_png_name(ch)
        Image.fromarray(skel, mode="L").save(skel_path)

        data = _vectorize_skeleton_png(skel_path, ch)
        out_path = VECTORS_PATH / safe_json_name(ch)
        out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"[vec] '{ch}' -> {out_path.name} (strokes={len(data['strokes'])})")


if __name__ == "__main__":
    main()
