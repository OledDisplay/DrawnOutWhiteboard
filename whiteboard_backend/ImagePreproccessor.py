#!/usr/bin/env python3
# preprocess_and_edges.pY

import json
import cv2
import numpy as np
import sys
import traceback
import re
import os
from pathlib import Path
import pytesseract
from pytesseract import Output


#Step 1 of printing - prepare the images

#Takes ready images of diagrams from research. 
#We first find all "text lables" in diagrams and extract and save them with ocr, then paint them over so they dont get ruined in all the proccessing.
#After that we run grayscale and canny
#We want to represent thin lines with actual thin lines when drawing on the board, but canny outlines everythin -> we get two lines for one is in some cases
#To combat this we merge close lines under a threshhold of 2.4 pixels (we do float operations, and in a lot of diagrams you have curved / not straight stuff -> 2.4 is valid)
#To get clean result we do our calculations in small 32 x 32 grids in the image. In this area we find the "mean gap" between two lines
#We only merge the lines that are very close to the mean gap -> that way we preserve fine small detail, but dont have false positives on big gaps
#To calculate our gaps we go in between two lines and "cast rays" (explore in directions) from a single point with (0, 45, 90, 135) for the direction to find the true width
#The true results are in the fine canny and other settings tweaks!


# ---------- Tesseract path (Windows) ----------
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# ===== SETTINGS =====
BASE = Path(__file__).resolve().parent
IN_DIR = BASE / r"ResearchImages/ddg"
OUT_DIR = BASE / r"ProccessedImages"

MIN_CONF = 10
PAD = 2
INPAINT_RADIUS = 2.7
CLAHE_CLIP = 2.5
MIN_SIZE = 5
SHOW_PROGRESS = True

# Preprocess tweaks (basic)
BORDER_PAD = 3
UPSCALE_TRIGGER = 600
UPSCALE_FACTOR = 1.3
UNSHARP_AMOUNT = 0.6
BILATERAL_D = 5
BILATERAL_SC = 35
BILATERAL_SS = 35

# ===== Fine Canny params =====
CANNY_SIGMA = 1.0
CANNY_APERTURE = 3
CANNY_K_LOW = 0.35
CANNY_K_HIGH = 1.5
CLOSE_R = 1
SAVE_DEBUG_GRAY = False

# ===== FILL TUNING (For accepting and merging gaps in tiles) =====
MERGE_LEVEL = ""  # "", "light", "medium", "aggressive"

FILL_HALF_WIDTH = 2.4         # <— allow non-integer half-width for decisions
FILL_BRIDGE_ITERS = 1
FILL_MIN_CC_AREA = 6
FILL_DISTANCE_GATE = True
FILL_ORIENTATIONS = (0, 45, 90, 135)
FILL_DIST_MULT = 1.0

# --- box-based merge knobs ---
TILE_SIZE         = 32
MAX_GAP           = 15     
GAP_TOLERANCE     = 15.0   # percent ± around center (median after IQR), keep float
MIN_CANDIDATES    = 2  #at least two gaps between lines in a box to be valid
MIN_EDGES_IN_TILE = 3  #at least 3 lines in total in a box 

# Ridge gating knobs
BAND_MAX_PIX         = 2
BAND_MAX_SHARE       = 0.40
RUN_MIN              = 6
RUN_INLIER_FRAC      = 0.70
MAX_NORMAL_DRIFT_DEG = 20.0
SIDE_CONFIRM         = 2

def _resolve_merge_params():
    if MERGE_LEVEL == "light":
        return dict(half_width=2.0, bridge_iters=1, clean_area=6,
                    do_distance_gate=True, orientations=(0, 90))
    if MERGE_LEVEL == "medium":
        return dict(half_width=3.0, bridge_iters=1, clean_area=6,
                    do_distance_gate=True, orientations=(0, 45, 90, 135))
    if MERGE_LEVEL == "aggressive":
        return dict(half_width=4.0, bridge_iters=2, clean_area=8,
                    do_distance_gate=True, orientations=(0, 30, 45, 60, 90, 120, 135, 150))
    return dict(
        half_width=float(FILL_HALF_WIDTH),
        bridge_iters=FILL_BRIDGE_ITERS,
        clean_area=FILL_MIN_CC_AREA,
        do_distance_gate=FILL_DISTANCE_GATE,
        orientations=FILL_ORIENTATIONS,
    )

# =========================================================
# PREPROCESS + OCR
# =========================================================
def preprocess(img_bgr):
    h, w = img_bgr.shape[:2]
    short = min(h, w)
    if short < UPSCALE_TRIGGER:
        new_w, new_h = int(w * UPSCALE_FACTOR), int(h * UPSCALE_FACTOR)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = img_bgr.shape[:2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    if BILATERAL_D > 0:
        gray = cv2.bilateralFilter(gray, d=BILATERAL_D,
                                   sigmaColor=BILATERAL_SC,
                                   sigmaSpace=BILATERAL_SS)

    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gray = cv2.addWeighted(gray, 1.0 + UNSHARP_AMOUNT, blurred, -UNSHARP_AMOUNT, 0)

    gray = cv2.copyMakeBorder(
        gray, BORDER_PAD, BORDER_PAD, BORDER_PAD, BORDER_PAD,
        borderType=cv2.BORDER_CONSTANT, value=255
    )
    return img_bgr, gray

def ocr_tesseract(img_gray):
    data = pytesseract.image_to_data(img_gray, output_type=Output.DICT)
    n = len(data["text"])
    boxes = []
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            continue
        if conf < MIN_CONF:
            continue
        x = int(data["left"][i]); y = int(data["top"][i])
        w = int(data["width"][i]); h = int(data["height"][i])
        if w < MIN_SIZE or h < MIN_SIZE:
            continue
        boxes.append((txt, conf, x, y, w, h))
    return boxes

# =========================================================
# CANNY
# =========================================================
def edge_canny_fine(img_bgr,
                    sigma=CANNY_SIGMA,
                    k_low=CANNY_K_LOW,
                    k_high=CANNY_K_HIGH,
                    aperture=CANNY_APERTURE,
                    close_r=CLOSE_R):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if sigma and sigma > 0:
        gray = cv2.GaussianBlur(gray, (3, 3), sigma)

    med = float(np.median(gray))
    low  = int(np.clip(k_low  * med,  0, 255))
    high = int(np.clip(k_high * med,  0, 255))
    if high <= low:
        high = min(255, low + 20)

    edges = cv2.Canny(gray, low, high, apertureSize=aperture, L2gradient=True)

    if close_r and close_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)

    return edges, gray

# =========================================================
# CLEAN NOISE (kept; separate from ridge output)
# =========================================================
def prune_small_dots(edges_bin: np.ndarray,
                     area_max: int = 40,
                     circ_min: float = 0.65,
                     skel_len_min: int = 10,
                     use_skeleton: bool = False) -> np.ndarray:
    src = (edges_bin > 0).astype(np.uint8) * 255
    num, lab, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    if num <= 1:
        return src

    out = np.zeros_like(src)
    k3 = np.ones((3, 3), np.uint8)

    for i in range(1, num):
        _, _, _, _, area = stats[i]
        comp_mask = (lab == i).astype(np.uint8) * 255

        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        perim = float(cv2.arcLength(cnts[0], True))
        if perim <= 0:
            perim = 1.0

        circularity = float((4.0 * np.pi * area) / (perim * perim))
        is_small = area <= area_max
        is_dotty = circularity >= circ_min

        if use_skeleton:
            comp = cv2.morphologyEx(comp_mask, cv2.MORPH_OPEN, k3, iterations=1)
            skel = np.zeros_like(comp)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            while True:
                eroded = cv2.erode(comp, element)
                opened = cv2.dilate(eroded, element)
                sub = cv2.subtract(comp, opened)
                skel = cv2.bitwise_or(skel, sub)
                comp = eroded
                if cv2.countNonZero(comp) == 0:
                    break
            skel_len = int(cv2.countNonZero(skel))
            too_short = skel_len < skel_len_min
        else:
            too_short = False

        if is_small and (is_dotty or too_short):
            continue
        out[lab == i] = 255
    return out

def bridge_gaps_1px(edges: np.ndarray, iters: int = 1) -> np.ndarray:
    E = (edges > 0).astype(np.uint8) * 255
    k_h  = np.array([[0,0,0],[1,0,1],[0,0,0]], np.uint8)
    k_v  = np.array([[0,1,0],[0,0,0],[0,1,0]], np.uint8)
    k_d1 = np.array([[1,0,0],[0,0,0],[0,0,1]], np.uint8)
    k_d2 = np.array([[0,0,1],[0,0,0],[1,0,0]], np.uint8)

    def _bridge_once(img, k):
        conv = cv2.filter2D((img > 0).astype(np.uint8), -1, k)
        add  = ((conv == 2) & (img == 0)).astype(np.uint8) * 255
        return cv2.bitwise_or(img, add)

    for _ in range(iters):
        for k in (k_h, k_v, k_d1, k_d2):
            E = _bridge_once(E, k)
    return E

# =========================================================
# TILE-BASED GAP-AVERAGE FILL (RIDGE RUNS ONLY)
# =========================================================
def _scan_corridor(E_bin: np.ndarray, x: int, y: int, dx: int, dy: int, max_gap: int) -> int:
    """Return integer corridor width between nearest edges in ±(dx,dy), else 0."""
    h, w = E_bin.shape
    s1 = 0
    for s in range(1, max_gap + 1):
        nx, ny = x + dx * s, y + dy * s
        if nx < 0 or nx >= w or ny < 0 or ny >= h: break
        if E_bin[ny, nx]: s1 = s; break
    if s1 == 0: return 0
    s2 = 0
    for s in range(1, max_gap + 1):
        nx, ny = x - dx * s, y - dy * s
        if nx < 0 or nx >= w or ny < 0 or ny >= h: break
        if E_bin[ny, nx]: s2 = s; break
    if s2 == 0: return 0
    width = s1 + s2 - 1
    if width < 2 or width > max_gap: return 0
    return width

def _scan_corridor_width_and_normal(E_bin: np.ndarray, x: int, y: int, dirs: list, max_gap: int):
    """Return (best_width [int], nx, ny [float unit normal])."""
    best_w = 0
    best_n = (0.0, 0.0)
    for (dx, dy) in dirs:
        wgap = _scan_corridor(E_bin, x, y, dx, dy, max_gap)
        if wgap > 0 and (best_w == 0 or wgap < best_w):
            nlen = float(np.hypot(dx, dy))
            best_w = wgap
            best_n = (dx / nlen, dy / nlen) if nlen > 0 else (0.0, 0.0)
    return float(best_w), best_n  # <- width as float for decision path

def _ridge_mask(tile_inv: np.ndarray) -> np.ndarray:
    """Ridge = local maxima of distance transform in 3x3 neighborhood."""
    dist = cv2.distanceTransform(tile_inv, cv2.DIST_L2, 3).astype(np.float32)
    mx = cv2.dilate(dist, np.ones((3, 3), np.uint8))
    ridge = (dist >= (mx - 1e-6)) & (dist > 0)
    return ridge

def _trace_ridge_runs(ridge_bool: np.ndarray):
    """Return list of runs (list of (y,x)) over ridge pixels by greedy chaining."""
    H, W = ridge_bool.shape
    visited = np.zeros_like(ridge_bool, dtype=np.uint8)
    runs = []
    nbrs8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    ys, xs = np.where(ridge_bool)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]: continue
        run = [(y0, x0)]
        visited[y0, x0] = 1
        for direction in (1, -1):
            curr = (y0, x0)
            prev = None
            while True:
                best = None
                best_dot = -1e9
                vx, vy = 0, 0
                if prev is not None:
                    vx = curr[1] - prev[1]
                    vy = curr[0] - prev[0]
                for dy, dx in nbrs8:
                    ny, nx = curr[0] + dy, curr[1] + dx
                    if ny < 0 or ny >= H or nx < 0 or nx >= W: continue
                    if not ridge_bool[ny, nx] or visited[ny, nx]: continue
                    if prev is None:
                        best = (ny, nx); break
                    dot = vx * dx + vy * dy
                    if dot > best_dot:
                        best_dot = dot
                        best = (ny, nx)
                if best is None: break
                visited[best[0], best[1]] = 1
                if direction == 1:
                    run.append(best)
                else:
                    run.insert(0, best)
                prev, curr = curr, best
        runs.append(run)
    return runs

def _angle_between(n1, n2):
    dot = float(n1[0]*n2[0] + n1[1]*n2[1])
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(np.arccos(dot)))

def fill_between_outlines_tilewise(
    E_edges: np.ndarray,
    half_width: float,
    tile_size: int = TILE_SIZE,
    max_gap: int = MAX_GAP,
    gap_tol: float = GAP_TOLERANCE,
) -> np.ndarray:
    """
    Ridge run logic with float decision path:
      - widths measured (still pixel grid) but carried as float
      - robust center via IQR+median (float)
      - inlier check uses float band (no rounding)
      - only final pixel write is integer
    """
    if half_width <= 0.0:
        return (E_edges > 0).astype(np.uint8) * 255

    E_bin = (E_edges > 0).astype(np.uint8)
    h, w = E_bin.shape
    fill = np.zeros_like(E_bin, dtype=np.uint8)

    # corridor directions
    dirs = []
    for a in (0, 45, 90, 135):
        rad = np.deg2rad(a)
        dx = int(round(np.cos(rad)))
        dy = int(round(np.sin(rad)))
        if dx == 0 and dy == 0: continue
        dirs.append((dx, dy))

    tol_frac = float(gap_tol) / 100.0
    step = int(tile_size)

    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1 = min(h, y0 + step)
            x1 = min(w, x0 + step)

            tile = E_bin[y0:y1, x0:x1]
            if tile.size == 0: 
                continue
            if int(tile.sum()) < MIN_EDGES_IN_TILE: 
                continue

            ridge = _ridge_mask((tile == 0).astype(np.uint8))
            runs = _trace_ridge_runs(ridge)
            if not runs: 
                continue

            all_widths = []
            per_run_samples = []

            for run in runs:
                samples = []
                for (ry, rx) in run:
                    yy, xx = y0 + ry, x0 + rx
                    wgap, nvec = _scan_corridor_width_and_normal(E_bin, xx, yy, dirs, max_gap)
                    if wgap > 0.0:
                        all_widths.append(wgap)                # float
                        samples.append((yy, xx, wgap, nvec))   # keep float
                per_run_samples.append(samples)

            if len(all_widths) < MIN_CANDIDATES:
                continue

            gaps = np.array(all_widths, dtype=np.float32)
            q1, q3 = np.percentile(gaps, [25.0, 75.0])
            iqr = max(1e-6, q3 - q1)
            keep = (gaps >= (q1 - 1.5 * iqr)) & (gaps <= (q3 + 1.5 * iqr))
            gaps_f = gaps[keep]
            if gaps_f.size < MIN_CANDIDATES:
                continue

            center = float(np.median(gaps_f))
            if center <= 0.0:
                continue

            # float decision band
            low_f  = center * (1.0 - tol_frac)
            high_f = center * (1.0 + tol_frac)

            # integer band only for anti-flood stats; decision uses floats
            low_i  = max(2, int(np.floor(low_f)))
            high_i = int(np.ceil(high_f))
            if (high_i - low_i + 1) > BAND_MAX_PIX:
                while (high_i - low_i + 1) > BAND_MAX_PIX:
                    if (high_i - center) > (center - low_i):
                        high_i -= 1
                    else:
                        low_i += 1
            if high_i < low_i:
                continue

            # anti-flood using INT histogram (stats only)
            hist = {}
            for wv in gaps_f:
                k = int(round(wv))
                hist[k] = hist.get(k, 0) + 1
            band_count = sum(cnt for wv, cnt in hist.items() if low_i <= wv <= high_i)
            if band_count / max(1, gaps_f.size) > BAND_MAX_SHARE:
                continue

            # evaluate runs with FLOAT inlier check
            for samples in per_run_samples:
                if len(samples) < RUN_MIN:
                    continue

                inlier_flags = [ (low_f <= s[2] <= high_f) for s in samples ]
                inliers = [s for s, ok in zip(samples, inlier_flags) if ok]
                if len(inliers) < int(RUN_MIN * RUN_INLIER_FRAC):
                    continue

                # normal drift across inliers
                normals = [nv for (_, _, _, nv) in inliers]
                base = normals[0]
                if any(_angle_between(base, nv) > MAX_NORMAL_DRIFT_DEG for nv in normals[1:]):
                    continue

                # side confirmation (integer walk, but decision remains float band)
                def _side_ok(yy, xx, nv):
                    best = None; best_ang = 1e9
                    for (dx, dy) in dirs:
                        nlen = float(np.hypot(dx, dy))
                        if nlen == 0: continue
                        cand = (dx/nlen, dy/nlen)
                        ang = _angle_between(nv, cand)
                        if ang < best_ang:
                            best_ang = ang; best = (dx, dy)
                    if best is None: 
                        return False
                    dx, dy = best
                    pos_ok = any(
                        (0 <= xx + dx*s < w) and (0 <= yy + dy*s < h) and E_bin[yy + dy*s, xx + dx*s]
                        for s in range(1, SIDE_CONFIRM + 1)
                    )
                    neg_ok = any(
                        (0 <= xx - dx*s < w) and (0 <= yy - dy*s < h) and E_bin[yy - dy*s, xx - dx*s]
                        for s in range(1, SIDE_CONFIRM + 1)
                    )
                    return pos_ok and neg_ok

                side_ok_flags = [ _side_ok(yy, xx, nv) for (yy, xx, _, nv) in inliers ]
                if sum(side_ok_flags) < int(len(inliers) * RUN_INLIER_FRAC):
                    continue

                # write only those samples that are float-inliers (rounding only at write)
                for idx, ok in enumerate(inlier_flags):
                    if not ok:
                        continue
                    yy, xx, _, _ = samples[idx]
                    fill[int(yy), int(xx)] = 1  # final integer pixel write

    out = np.clip(E_bin + fill, 0, 1).astype(np.uint8) * 255
    return out

# =========================================================
# SMALL HOLE (uses float half_width)
# =========================================================
def small_hole_fill_near_edges(E: np.ndarray, half_width: float) -> np.ndarray:
    if half_width <= 0.0:
        return np.zeros_like(E)
    src = (E > 0).astype(np.uint8)
    contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(E)

    filled = np.zeros_like(E)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    inv  = (E == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    gate = (dist <= float(half_width) + 1e-6).astype(np.uint8)

    res = (filled & (~src) & (gate * 255).astype(np.uint8))
    return res

# =========================================================
# PIPELINE
# =========================================================
def fill_outlines_pipeline(edge_like_img: np.ndarray,
                           half_width: float,
                           bridge_iters: int,
                           clean_area: int,
                           do_distance_gate: bool,
                           orientations: tuple) -> np.ndarray:
    if edge_like_img.ndim == 3:
        gray = cv2.cvtColor(edge_like_img, cv2.COLOR_BGR2GRAY)
        E0 = (gray > 0).astype(np.uint8) * 255
    else:
        E0 = (edge_like_img > 0).astype(np.uint8) * 255

    E1 = bridge_gaps_1px(E0, iters=bridge_iters)

    E2 = fill_between_outlines_tilewise(
        E1,
        half_width=float(half_width),
        tile_size=TILE_SIZE,
        max_gap=MAX_GAP,
        gap_tol=float(GAP_TOLERANCE),
    )

    near = small_hole_fill_near_edges(E2, half_width=float(half_width))
    E3 = cv2.bitwise_or(E2, near)
    return E3

# ============== OUTPUT INDEXING + PIPELINE ==============
def _get_next_index() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r'^processed_(\d+)$')
    used = set()
    for p in OUT_DIR.glob("processed_*.png"):
        m = pat.match(p.stem)
        if m:
            try:
                used.add(int(m.group(1)))
            except:
                pass
    n = 0
    while n in used:
        n += 1
    return n

def process_one(path: Path):
    try:
        if SHOW_PROGRESS:
            print(f"[START] {path}")
        img0 = cv2.imread(str(path))
        if img0 is None:
            print(f"[WARN] cannot read {path}")
            return
        index = _get_next_index()

        img, gray = preprocess(img0)
        H, W = img.shape[:2]

        boxes_bordered = ocr_tesseract(gray)
        if SHOW_PROGRESS:
            print(f"[{path.name}] OCR boxes: {len(boxes_bordered)}")

        mask = np.zeros((H, W), np.uint8)
        labels = []
        for text, conf, x, y, w, h in boxes_bordered:
            x -= BORDER_PAD; y -= BORDER_PAD
            x1 = max(0, x - PAD); y1 = max(0, y - PAD)
            x2 = min(W, x + w + PAD); y2 = min(H, y + h + PAD)
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            labels.append({
                "text": text,
                "confidence": round(conf / 100.0, 3),
                "x": int(x1), "y": int(y1),
                "width": int(x2 - x1), "height": int(y2 - y1)
            })

        cleaned = cv2.inpaint(img, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA) if np.any(mask) else img

        edges, gray_used = edge_canny_fine(cleaned)
        edges = prune_small_dots(edges, area_max=30, circ_min=0.6, skel_len_min=10, use_skeleton=False)

        params = _resolve_merge_params()
        edges = fill_outlines_pipeline(
            edges,
            half_width=float(params["half_width"]),
            bridge_iters=params["bridge_iters"],
            clean_area=params["clean_area"],
            do_distance_gate=params["do_distance_gate"],
            orientations=params["orientations"],
        )

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_img = OUT_DIR / f"processed_{index}.png"
        out_edges = OUT_DIR / f"edges_{index}.png"
        out_json = OUT_DIR / f"processed_{index}_labels.json"

        cv2.imwrite(str(out_img), cleaned)
        cv2.imwrite(str(out_edges), edges)
        if SAVE_DEBUG_GRAY and gray_used is not None:
            cv2.imwrite(str(OUT_DIR / f"gray_{index}.png"), gray_used)

        meta = {
            "image_index": index,
            "original_name": path.name,
            "resolution": {"width": W, "height": H},
            "labels": labels,
        }
        out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if SHOW_PROGRESS:
            print(f"[OK] wrote:\n  {out_img}\n  {out_edges}\n  {out_json}")

    except Exception:
        print(f"[ERR] crashed on {path}:\n{traceback.format_exc()}")

def main():
    print(f"[INFO] IN_DIR={IN_DIR}")
    print(f"[INFO] OUT_DIR={OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    imgs = sorted(
        [p for p in IN_DIR.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")],
        key=lambda p: p.name.lower(),
    )
    print(f"[INFO] found {len(imgs)} image(s).")
    if not imgs:
        print(f"[!] No images found in {IN_DIR}. Exiting.")
        return

    for p in imgs:
        process_one(p)

if __name__ == "__main__":
    main()
