#!/usr/bin/env python3
# preprocess_and_edges.py
import json, cv2, numpy as np, sys, traceback, re, math, os
from pathlib import Path
import pytesseract
from pytesseract import Output

# ---------- Tesseract path (Windows) ----------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ===== SETTINGS =====
BASE = Path(__file__).resolve().parent
IN_DIR  = BASE / r"ResearchImages/ddg"
OUT_DIR = BASE / r"ProccessedImages"

MIN_CONF = 10            # OCR min confidence (Tesseract 0..100)
PAD = 2                  # pad around text boxes before inpaint
INPAINT_RADIUS = 2.7     # Telea inpaint radius
CLAHE_CLIP = 2.5         # CLAHE clip limit
MIN_SIZE = 5             # minimum OCR box size (px)
SHOW_PROGRESS = True

# Preprocess tweaks
BORDER_PAD = 2           # tiny white border to catch edge-hugging text
UPSCALE_TRIGGER = 600    # if min(H,W) < this, upscale once
UPSCALE_FACTOR  = 1.3
UNSHARP_AMOUNT  = 0.6
BILATERAL_D     = 5
BILATERAL_SC    = 35
BILATERAL_SS    = 35

# ===== Fine Canny params =====
CANNY_SIGMA        = 1.0      # Gaussian sigma before Canny
CANNY_APERTURE     = 3        # 3 for fine detail
CANNY_K_LOW        = 0.35     # auto low  = k_low  * median(gray)
CANNY_K_HIGH       = 1.5      # auto high = k_high * median(gray)
CLOSE_R            = 1        # small morph close radius to bridge 1px gaps
SAVE_DEBUG_GRAY    = False    # set True to save the gray used for thresholds

# ===== FILL TUNING (intensity knobs) =====
MERGE_LEVEL = ""  # "", "light", "medium", "aggressive"

FILL_HALF_WIDTH      = 1
FILL_BRIDGE_ITERS    = 2
FILL_MIN_CC_AREA     = 6
FILL_DISTANCE_GATE   = True
FILL_ORIENTATIONS    = (0,11,22,34,45,56,67,78,90,101,116,126,140,152)
FILL_DIST_MULT       = 1.3

# =========================================================
def _resolve_merge_params():
    if MERGE_LEVEL == "light":
        return dict(half_width=2, bridge_iters=1, clean_area=6, do_distance_gate=True, orientations=(0,90))
    if MERGE_LEVEL == "medium":
        return dict(half_width=3, bridge_iters=1, clean_area=6, do_distance_gate=True, orientations=(0,45,90,135))
    if MERGE_LEVEL == "aggressive":
        return dict(half_width=4, bridge_iters=2, clean_area=8, do_distance_gate=True, orientations=(0,30,45,60,90,120,135,150))
    return dict(
        half_width=FILL_HALF_WIDTH,
        bridge_iters=FILL_BRIDGE_ITERS,
        clean_area=FILL_MIN_CC_AREA,
        do_distance_gate=FILL_DISTANCE_GATE,
        orientations=FILL_ORIENTATIONS,
    )

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

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    if BILATERAL_D > 0:
        gray = cv2.bilateralFilter(gray, d=BILATERAL_D, sigmaColor=BILATERAL_SC, sigmaSpace=BILATERAL_SS)

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
def edge_canny_fine(img_bgr,
                    sigma=CANNY_SIGMA,
                    k_low=CANNY_K_LOW,
                    k_high=CANNY_K_HIGH,
                    aperture=CANNY_APERTURE,
                    close_r=CLOSE_R):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if sigma and sigma > 0:
        gray = cv2.GaussianBlur(gray, (3,3), sigma)

    med = float(np.median(gray))
    low  = int(np.clip(k_low  * med,  0, 255))
    high = int(np.clip(k_high * med,  0, 255))
    if high <= low:
        high = min(255, low + 20)

    edges = cv2.Canny(gray, low, high, apertureSize=aperture, L2gradient=True)

    if close_r and close_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*close_r+1, 2*close_r+1))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)

    return edges, gray

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
    k3 = np.ones((3,3), np.uint8)
    for i in range(1, num):
        _, _, _, _, area = stats[i]
        comp_mask = (lab == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: continue
        perim = float(cv2.arcLength(cnts[0], True))
        if perim <= 0: perim = 1.0
        circularity = float((4.0 * np.pi * area) / (perim * perim))
        is_small = area <= area_max
        is_dotty = circularity >= circ_min
        if use_skeleton:
            comp = cv2.morphologyEx(comp_mask, cv2.MORPH_OPEN, k3, iterations=1)
            skel = np.zeros_like(comp)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            while True:
                eroded = cv2.erode(comp, element)
                opened = cv2.dilate(eroded, element)
                sub = cv2.subtract(comp, opened)
                skel = cv2.bitwise_or(skel, sub)
                comp = eroded
                if cv2.countNonZero(comp) == 0: break
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

def _line_kernel(length: int, angle_deg: int) -> np.ndarray:
    if angle_deg % 180 == 0:   return cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    if angle_deg % 180 == 90:  return cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    k = np.zeros((length, length), np.uint8)
    if angle_deg % 180 == 45:  np.fill_diagonal(k, 1); return k
    if angle_deg % 180 == 135: return np.fliplr(np.eye(length, dtype=np.uint8))
    return cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))

def curvature_mask(E: np.ndarray,
                   curve_thresh: float = 0.0035,
                   smooth_size: int = 5) -> np.ndarray:
    E = (E > 0).astype(np.uint8) * 255
    E_blur = cv2.GaussianBlur(E, (smooth_size, smooth_size), 0)
    gx = cv2.Sobel(E_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(E_blur, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.Sobel(gx, cv2.CV_32F, 1, 0, ksize=3)
    gyy = cv2.Sobel(gy, cv2.CV_32F, 0, 1, ksize=3)
    gxy = cv2.Sobel(gx, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2) + 1e-6
    curvature = (np.abs(gxx * gy**2 - 2 * gx * gy * gxy + gyy * gx**2)) / (grad_mag**3 + 1e-6)
    curvature = cv2.GaussianBlur(curvature, (3,3), 0)
    curvature = cv2.normalize(curvature, None, 0, 1, cv2.NORM_MINMAX)
    mask = (curvature < curve_thresh).astype(np.uint8)
    return mask

def fill_between_outlines_fixed(E: np.ndarray,
                                half_width: float = 2.0,
                                orientations: tuple = (0, 22, 45, 67, 90, 117, 135, 152),
                                distance_gate: bool = True,
                                softness: float = 0.65,
                                curve_thresh: float = 0.0035) -> np.ndarray:
    E = (E > 0).astype(np.uint8)
    if half_width < 0.5:
        return (E * 255).astype(np.uint8)
    hw_int = max(1, int(round(half_width)))
    length = max(1, 2 * hw_int + 1)
    curve_m = curvature_mask(E, curve_thresh=curve_thresh)
    ribbon_union = np.zeros_like(E)
    for a in orientations:
        k = _line_kernel(length, int(a))
        closed = cv2.morphologyEx(E, cv2.MORPH_CLOSE, k, iterations=1)
        dil    = cv2.dilate(E, k, iterations=1)
        ribbon = np.clip(closed - dil, 0, 1)
        ribbon_union = np.bitwise_or(ribbon_union, ribbon)
    ribbon_union = cv2.bitwise_and(ribbon_union, curve_m)
    if distance_gate:
        inv  = (E == 0).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        gate = (dist <= float(half_width * 1.25) + 1e-6).astype(np.uint8)
        ribbon_union *= gate
    out = E.astype(np.float32) + softness * ribbon_union.astype(np.float32)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)

def small_hole_fill_near_edges(E: np.ndarray, half_width: int) -> np.ndarray:
    contours, _ = cv2.findContours(E, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(E)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    inv  = (E == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    gate = (dist <= float(half_width * FILL_DIST_MULT) + 1e-6).astype(np.uint8) * 255
    return cv2.bitwise_and(filled, gate)

def remove_small_cc(bin_img: np.ndarray, min_area: int) -> np.ndarray:
    num, lab, stats, _ = cv2.connectedComponentsWithStats((bin_img > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(bin_img, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 255
    return out

def pixelwise_smoother_no_pad(E: np.ndarray,
                              strength: float = 0.35,
                              window: int = 3,
                              iters: int = 2) -> np.ndarray:
    src = (E.astype(np.float32) / 255.0).copy()
    for _ in range(iters):
        mean = cv2.blur(src, (window, window))
        src = cv2.addWeighted(src, 1.0 - strength, mean, strength, 0)
    out = (src > 0.5).astype(np.uint8) * 255
    return out

def fill_outlines_pipeline(edge_like_img: np.ndarray,
                           half_width: int,
                           bridge_iters: int,
                           clean_area: int,
                           do_distance_gate: bool,
                           orientations: tuple) -> np.ndarray:
    if edge_like_img.ndim == 3:
        gray = cv2.cvtColor(edge_like_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = edge_like_img.copy()
    _, E0 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    E1 = bridge_gaps_1px(E0, iters=bridge_iters)
    E2 = fill_between_outlines_fixed(E1, half_width=half_width, orientations=orientations, distance_gate=do_distance_gate)
    near = small_hole_fill_near_edges(E2, half_width=half_width)
    E3 = cv2.bitwise_or(E2, near)
    out = remove_small_cc(E3, min_area=clean_area)
    return out

# ============== OUTPUT INDEXING + PIPELINE ==============
def _get_next_index() -> int:
    """
    Return the smallest non-negative integer N such that
    processed_N.png does not exist in OUT_DIR.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r'^processed_(\d+)$')
    used = set()
    for p in OUT_DIR.glob("processed_*.png"):
        m = pat.match(p.stem)
        if m:
            try: used.add(int(m.group(1)))
            except: pass
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
            half_width=params["half_width"],
            bridge_iters=params["bridge_iters"],
            clean_area=params["clean_area"],
            do_distance_gate=params["do_distance_gate"],
            orientations=params["orientations"],
        )

        edges = pixelwise_smoother_no_pad(edges, strength=0.35, window=3, iters=2)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_img   = OUT_DIR / f"processed_{index}.png"
        out_edges = OUT_DIR / f"edges_{index}.png"
        out_json  = OUT_DIR / f"processed_{index}_labels.json"

        cv2.imwrite(str(out_img), cleaned)
        cv2.imwrite(str(out_edges), edges)
        if SAVE_DEBUG_GRAY and gray_used is not None:
            cv2.imwrite(str(OUT_DIR / f"gray_{index}.png"), gray_used)

        meta = {
            "image_index": index,
            "original_name": path.name,
            "resolution": {"width": W, "height": H},
            "labels": labels
        }
        out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if SHOW_PROGRESS:
            print(f"[OK] wrote:\n  {out_img}\n  {out_edges}\n  {out_json}")

        return 
    except Exception:
        print(f"[ERR] crashed on {path}:\n{traceback.format_exc()}")


def process_paths(input_imgs, output_dir):
    processed = []

    # accepted file extensions
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    for fname in input_imgs:

        # call your existing image processor
        process_one(fname)

        # expected output file (optional)
        out_path = os.path.join(output_dir, fname)
        if os.path.exists(out_path):
            processed.append(out_path)

    return processed

def main():
    print(f"[INFO] IN_DIR={IN_DIR}")
    print(f"[INFO] OUT_DIR={OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    imgs = sorted(
        [p for p in IN_DIR.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")],
        key=lambda p: p.name.lower()
    )
    print(f"[INFO] found {len(imgs)} image(s).")
    if not imgs:
        print(f"[!] No images found in {IN_DIR}. Exiting.")

        return
    process_paths(imgs, OUT_DIR)

if __name__ == "__main__":
    main()
