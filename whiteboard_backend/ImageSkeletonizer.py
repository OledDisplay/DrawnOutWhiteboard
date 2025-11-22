#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from pathlib import Path

import cv2
import numpy as np

#Currently pushing pen width above 3 breaks stuff, but im happy to work with it as "pen width" only really matters for customization of the output, not the core functionality.

#Step 2 of printing - thin images to simple 1px wide lines, so they are easier to split up in strokes

#Input - edge canny images that have had some filling between edges

#This is the second step of our image preproccess / vector conversion - Here we turn the images big filled lines and objects into small thin ones that are easier to proccess.
#We are using guo hall style thinning but we need some extra help to get good results
#The problem is we want both single lines, when proccessing just a filled straight line AND outlines on big filled objects
#We solve this by taking the width with which our future drawing pen is going to pad thin vectors -> and applying it to mapping here
#When meeting a filled area of pixels instead of mapping the edge we go inside from each pixel, by "pen width" pixels (pad pixel size)
#We create our outline inside the filled object, and when we pad later -> it turns out 1 : 1
#With thinner outlines, the thinning line gets put close to the center of the line and if the distance from the center is <= the width of the pen -> we get only one line to trace!
#This happens because we record "built" lines and when we go from one edge of a thin object, mapping from there, and flip over to the other edge and start looking inside -> we find a ready line
#We use the same "hald width" as the preproccessor -> if half width is <= pen width we only get a line in the dead center of the object, if >, we trace "two lines" like a ring from both edges
#This way we practically get a canny with bigger filled objects
#The whole is our practical alternative to raw canny or raw thinning
#This allows us to then easily measure, decide on line directions and record lines with our vectorizer (look at it next)



# ===================== PATHS =====================
BASE = Path(__file__).resolve().parent
IN_DIR  = BASE / "ProccessedImages"   # input: your already processed images
OUT_DIR = BASE / "Skeletonized"       # output: skeleton PNGs

# ===================== KNOBS =====================

# "Pen thickness" in pixels for the *final drawing*.
# This is the radius we go inside from the border for fat blobs.
PEN_WIDTH = 3.0              # try 2.0 or 3.0 like you said

# Small noise kill before everything
MIN_COMPONENT_AREA = 5      # drop tiny blobs before skeletonizing

# Half-width classification tolerance:
# if max distance in a component <= PEN_WIDTH + WIDTH_EPS → treat as thin stroke
WIDTH_EPS = 0.5

# Band for offset lines in fat blobs:
# we keep pixels with distance in [PEN_WIDTH - BAND_HALF, PEN_WIDTH + BAND_HALF]
BAND_HALF = 0.5

# Thinning passes for strokes / bands (only to make them 1px wide, nothing fancy)
USE_SKIMAGE = True           # try skimage.thin if available
MAX_THIN_PASSES = 1          # Zhang–Suen fallback will loop until stable anyway


# ===================== BASIC HELPERS =====================

def _finite_nd(a: np.ndarray) -> bool:
    return np.isfinite(a).all()


# ---------- Binarize from processed image ----------
def load_foreground(path: Path) -> np.ndarray:
    """
    Input: grayscale line-art like your processed PNGs (white on black).
    Output: fg 0/1 mask (1 = foreground strokes/shapes).
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))

    # Otsu to get a clear 0/255 separation
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Decide which is foreground: minority color is fg
    white_ratio = float((bw == 255).mean())
    if white_ratio < 0.5:
        fg = (bw == 255).astype(np.uint8)
    else:
        fg = (bw == 0).astype(np.uint8)

    # Drop tiny connected components
    num, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = np.zeros_like(fg, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            out[lab == i] = 1

    return out


# ---------- Thinning (for thin strokes + offset bands) ----------

def thin_mask(mask01: np.ndarray) -> np.ndarray:
    """
    0/1 mask → 0/1 mask, thinned to ~1px while preserving topology.
    MUST use skimage.morphology.thin. If not available → hard error.
    """
    m = (mask01 > 0).astype(np.uint8)

    if USE_SKIMAGE:
        try:
            from skimage.morphology import thin
        except Exception as e:
            raise RuntimeError("ERROR: skimage.morphology.thin is required but not available.") from e

        try:
            th = thin(m > 0, max_num_iter=MAX_THIN_PASSES).astype(np.uint8)
            return th
        except Exception as e:
            raise RuntimeError("ERROR: skimage thinning failed during execution.") from e

    # If USE_SKIMAGE is False, stop immediately.
    raise RuntimeError("ERROR: USE_SKIMAGE is False but no fallback thinning is allowed.")



# ===================== PEN-WIDTH LOGIC =====================

def pen_width_skeleton(fg01: np.ndarray, pen_width: float) -> np.ndarray:
    """
    Core logic:

    - compute distance-to-border for each FG pixel
    - find connected components
    - per component:
        if max_dist <= pen_width + WIDTH_EPS:
            → treat as 'thin stroke': thin it to single center line
        else:
            → treat as 'fat shape':
               keep an iso-distance ring at depth ~= pen_width from border
    """
    fg = (fg01 > 0).astype(np.uint8)
    if fg.max() == 0:
        return fg.copy()

    # distance to nearest background (outside)
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3).astype(np.float32)

    # connected components over fg
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

    # A: thin all foreground once (for "thin stroke" behaviour)
    thin_all = thin_mask(fg)

    # B: build iso-distance band for fat blobs
    band_lo = max(0.0, pen_width - BAND_HALF)
    band_hi = pen_width + BAND_HALF
    band_mask = ((dist >= band_lo) & (dist <= band_hi) & (fg > 0)).astype(np.uint8)

    out = np.zeros_like(fg, dtype=np.uint8)

    for comp_idx in range(1, num):
        if stats[comp_idx, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
            continue

        comp_mask = (labels == comp_idx)
        # local max distance (half-width of thickest section in this component)
        max_d = float(dist[comp_mask].max()) if np.any(comp_mask) else 0.0

        if max_d <= pen_width + WIDTH_EPS:
            # -------- thin stroke: centerline only --------
            # use thinned version, but *only* inside this component
            out[comp_mask & (thin_all > 0)] = 1
        else:
            # -------- fat blob: canny-style offset lines --------
            # we keep a ring at depth ~= pen_width from the border
            out[comp_mask & (band_mask > 0)] = 1

    # one more very light thin pass, just to ensure 1px thickness everywhere
    out = thin_mask(out)

    return out


# ===================== MAIN PIPELINE =====================

def process_one(path: Path, output : Path):

    output.mkdir(parents=True, exist_ok=True)

    print(f"[START] {path.name}")
    fg = load_foreground(path)

    skel = pen_width_skeleton(fg, PEN_WIDTH)

    # 0/1 → 0/255 for saving
    img_out = (skel * 255).astype(np.uint8)
    out_path = output / f"{path.stem}_skeleton.png"
    cv2.imwrite(str(out_path), img_out)
    print(f"[OK]  wrote {out_path}")


def main():
    imgs = sorted(
        [
            p for p in IN_DIR.glob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        ],
        key=lambda p: p.name.lower(),
    )
    print(f"[INFO] IN={IN_DIR}  OUT={OUT_DIR}  found={len(imgs)} image(s)")

    if not imgs:
        print("[!] No images.")
        return

    for p in imgs:
        process_one(p, OUT_DIR)


if __name__ == "__main__":
    main()