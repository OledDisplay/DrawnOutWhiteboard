#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration



# bitsandbytes optional
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False


# ============================
# PATHS (NO ARGS)
# ============================
BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "ProccessedImages"
CLUSTER_RENDER_DIR = BASE_DIR / "ClusterRenders"
CLUSTER_MAP_DIR = BASE_DIR / "ClusterMaps"

OUT_DIR = BASE_DIR / "SmolVLM2Labels"
CACHE_DIR = BASE_DIR / "_path_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
IMG_CACHE_PATH = CACHE_DIR / "processed_png_index.json"
JSON_CACHE_PATH = CACHE_DIR / "processed_json_index.json"


# ============================
# MODEL CONFIG (VRAM CAP)
# ============================
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

QWEN_MIN_PIXELS = 256 * 28 * 28
QWEN_MAX_PIXELS = 512 * 28 * 28



GPU_INDEX = 0
FORCE_CPU = False

QUANT_MODE = "4bit" # "4bit" | "8bit" | "none"
INT8_CPU_OFFLOAD = False  # only relevant if QUANT_MODE=="8bit"


# ============================
# BATCHING (NO AUTO BACKOFF)
# ============================
# Set to 1 => strict cluster-by-cluster (lowest VRAM).
# Set >1 => batching (faster, more VRAM).
BATCH_SIZE = 8


# ============================
# SPEED / MEMORY LEVERS
# ============================
# ONE composite image = two square panels side-by-side.
# PANEL_EDGE = PROC_LONGEST_EDGE//2
PROC_LONGEST_EDGE = 768 # composite max side; panels ~384 each

SUGGESTION_LIMIT = 20
MAX_NEW_TOKENS = 70
DO_SAMPLE = False

RECT_THICKNESS_PX = 3
DTYPE = torch.float16
MIN_ACCEPT_CONF = 0.65

# Prints for every completed cluster (even in batching).
PRINT_EVERY_CLUSTER = True

# For speed: don't empty_cache each cluster.
EMPTY_CACHE_EVERY = 0  # 0 disables

# Optional cheap reject to skip obvious garbage (OFF by default)
ENABLE_FAST_REJECT = False
MIN_COLORED_PIXELS = 25


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ============================
# LABEL CLEANING
# ============================
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "from", "by", "with", "without",
    "is", "are", "was", "were", "be", "been", "being", "as", "that", "this", "these", "those",
    "it", "its", "into", "over", "under", "up", "down", "left", "right", "top", "bottom",
    "label", "labels", "figure", "fig", "diagram", "image", "picture", "illustration",
}

def _clean_word(w: str) -> Optional[str]:
    if not w:
        return None
    s = w.strip()
    s = re.sub(r"^[^\w]+|[^\w]+$", "", s)
    if not s or len(s) == 1:
        return None
    low = s.lower()
    if low in STOPWORDS:
        return None
    if re.fullmatch(r"\d+", s):
        return None
    return s


# ============================
# DISCOVERY / INDEXING
# ============================
_FULL_IMG_RE = re.compile(r"^(?:proccessed|processed)_(\d+)\.png$", re.IGNORECASE)
_FULL_JSON_RE = re.compile(r"^(?:proccessed|processed)_(\d+)\.json$", re.IGNORECASE)
_CLUSTER_DIR_RE = re.compile(r"^processed_(\d+)$", re.IGNORECASE)

def find_indices_from_cluster_maps() -> List[int]:
    out: List[int] = []
    if not CLUSTER_MAP_DIR.exists():
        return out
    for d in CLUSTER_MAP_DIR.iterdir():
        if not d.is_dir():
            continue
        m = _CLUSTER_DIR_RE.match(d.name)
        if not m:
            continue
        if (d / "clusters.json").exists():
            out.append(int(m.group(1)))
    out.sort()
    return out

def cluster_map_path_for_idx(idx: int) -> Path:
    return CLUSTER_MAP_DIR / f"processed_{idx}" / "clusters.json"

def cluster_renders_dir_for_idx(idx: int) -> Path:
    return CLUSTER_RENDER_DIR / f"processed_{idx}"

def parse_colour_hint_from_filename(name: str) -> Optional[str]:
    n = (name or "").lower()
    for c in ["white","yellow","orange","cyan","green","magenta","red","blue","purple","gray","black"]:
        if c in n:
            return c
    return None

def _load_cache(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("map"), dict):
            return {str(k): str(v) for k, v in obj["map"].items()}
    except Exception:
        pass
    return {}

def _save_cache(path: Path, mapping: Dict[str, str]) -> None:
    payload = {"created_utc": time.time(), "root": str(PROCESSED_DIR), "map": mapping}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _build_processed_index(regex: re.Pattern, suffix_hint: str) -> Dict[str, str]:
    t0 = time.time()
    mapping: Dict[str, str] = {}
    seen = 0
    matched = 0

    for root, _, files in os.walk(PROCESSED_DIR):
        for fn in files:
            seen += 1
            if (seen % 20000) == 0:
                print(f"[scan] {suffix_hint} files seen={seen} matched={matched} ({time.time()-t0:.1f}s)")
            m = regex.match(fn)
            if not m:
                continue
            idx = str(int(m.group(1)))
            p = str(Path(root) / fn)

            if idx in mapping:
                prev = Path(mapping[idx]).name.lower()
                cur = fn.lower()
                if prev.startswith("proccessed_") and cur.startswith("processed_"):
                    mapping[idx] = p
            else:
                mapping[idx] = p
            matched += 1

    print(f"[scan] {suffix_hint} index built: entries={len(mapping)} files_seen={seen} ({time.time()-t0:.1f}s)")
    return mapping

def ensure_indexes() -> Tuple[Dict[str, str], Dict[str, str]]:
    img_map = _load_cache(IMG_CACHE_PATH)
    json_map = _load_cache(JSON_CACHE_PATH)

    if not img_map:
        print("[scan] building PNG index cache (first run only)...")
        img_map = _build_processed_index(_FULL_IMG_RE, "png")
        _save_cache(IMG_CACHE_PATH, img_map)

    if not json_map:
        print("[scan] building JSON index cache (first run only)...")
        json_map = _build_processed_index(_FULL_JSON_RE, "json")
        _save_cache(JSON_CACHE_PATH, json_map)

    return img_map, json_map

def load_candidate_labels(idx: int, json_index: Dict[str, str]) -> List[str]:
    p = json_index.get(str(idx))
    if not p:
        return []
    path = Path(p)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    words = data.get("words") or []
    out: List[str] = []
    seen = set()
    for item in words:
        t = item.get("text") if isinstance(item, dict) else None
        if not isinstance(t, str):
            continue
        c = _clean_word(t)
        if not c:
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# ============================
# Image helpers (COMPOSITE)
# ============================
def _fit_into_square(img: Image.Image, side: int) -> Tuple[Image.Image, float, int, int, int, int]:
    img = img.convert("RGB")
    ow, oh = img.size
    if ow <= 0 or oh <= 0:
        square = Image.new("RGB", (side, side), (0, 0, 0))
        return square, 1.0, side, side, 0, 0

    scale = min(side / ow, side / oh)
    rw = max(1, int(round(ow * scale)))
    rh = max(1, int(round(oh * scale)))
    resized = img.resize((rw, rh), resample=Image.BICUBIC)

    square = Image.new("RGB", (side, side), (0, 0, 0))
    pad_x = (side - rw) // 2
    pad_y = (side - rh) // 2
    square.paste(resized, (pad_x, pad_y))

    return square, float(scale), rw, rh, pad_x, pad_y

def _draw_red_rect_pil(img: Image.Image, bbox_xyxy: List[int], thickness: int) -> None:
    if thickness <= 0:
        return
    w, h = img.size
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]

    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))

    x2 = max(0, min(w - 1, x1 - 1))
    y2 = max(0, min(h - 1, y1 - 1))
    if x2 <= x0 or y2 <= y0:
        return

    draw = ImageDraw.Draw(img)
    for t in range(int(thickness)):
        draw.rectangle([x0 - t, y0 - t, x2 + t, y2 + t], outline=(255, 0, 0))

def _make_composite(left_sq: Image.Image, right_sq: Image.Image) -> Image.Image:
    side = left_sq.size[0]
    canvas = Image.new("RGB", (side * 2, side), (0, 0, 0))
    canvas.paste(left_sq, (0, 0))
    canvas.paste(right_sq, (side, 0))

    draw = ImageDraw.Draw(canvas)
    x = side
    draw.line([(x - 1, 0), (x - 1, side)], fill=(80, 80, 80), width=1)
    draw.line([(x, 0), (x, side)], fill=(80, 80, 80), width=1)
    return canvas

def _bbox_norm_xyxy(bbox_xyxy: List[int], w: int, h: int) -> List[float]:
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    return [x0 / max(1, w), y0 / max(1, h), x1 / max(1, w), y1 / max(1, h)]

def _scale_bbox_xyxy(bbox_xyxy: List[int], scale: float) -> List[int]:
    return [int(round(v * scale)) for v in bbox_xyxy]

def _clamp_bbox_xyxy(bbox_xyxy: List[int], w: int, h: int) -> List[int]:
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    return [x0, y0, x1, y1]


# ============================
# Prompt + parse (ONE IMAGE)
# ============================
def build_prompt(
    colour_hint: Optional[str],
    suggestions: List[str],
) -> str:
    colour_txt = colour_hint if colour_hint else "unknown"
    sug = suggestions[:SUGGESTION_LIMIT]

    # Keep suggestions readable but short
    if sug:
        sug_text = ",".join(sug)
    else:
        sug_text = "(none)"

    return (
        "Here is an image with LEFT and RIGHT panels.\n"
        "- LEFT: a cropped target region (only the main color is active; other colors are grayscaled.\n"
        "- RIGHT: the full image with a red rectangle marking where LEFT came (coloured)\n\n"
        "Identify, count and describe pure shapes building up LEFT - big / small circles, lines, squares\n"
        "Analyze RIGHT + coloured crop for semantic context.\n"
        "Based on shapes and visual characteristics pick the closest label:\n\n"
        f"{sug_text}\n\n"
        "Return ONLY JSON with REQUIRED keys:\n"
        "{"
        "\"label\":string,"
        "\"shapes\": Found shapes listed,"
        "\"evidence\": What identifies it as that label,"
        "\"confidence\": 0 - 1"
        "}\n"
    )

def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    if not isinstance(text, str):
        return []

    s = text.strip()
    out: List[Dict[str, Any]] = []

    i = 0
    n = len(s)
    while i < n:
        start = s.find("{", i)
        if start < 0:
            break

        depth = 0
        end = -1
        for j in range(start, n):
            ch = s[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break

        if end < 0:
            break

        chunk = s[start:end + 1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            pass

        i = end + 1

    return out

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    objs = extract_json_objects(text)
    return objs[0] if objs else None

def _ensure_cluster_containers(results: Dict[str, Any]) -> None:
    if "clusters" not in results or not isinstance(results.get("clusters"), dict):
        results["clusters"] = {}
    if "cluster_order" not in results or not isinstance(results.get("cluster_order"), list):
        results["cluster_order"] = []

def _store_cluster_result(results: Dict[str, Any], cluster_key: str, payload: Dict[str, Any]) -> None:
    _ensure_cluster_containers(results)
    clusters = results["clusters"]
    order = results["cluster_order"]

    if cluster_key not in clusters:
        order.append(cluster_key)
    clusters[cluster_key] = payload


# ============================
# VRAM debug
# ============================
def _vram_str() -> str:
    if not torch.cuda.is_available():
        return "cuda=off"
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    reserv = torch.cuda.memory_reserved() / (1024 ** 2)
    return f"alloc={alloc:.0f}MiB reserved={reserv:.0f}MiB"


# ============================
# Optional fast reject (noise)
# ============================
def _count_colored_pixels(mask_rgb: Image.Image) -> int:
    arr = np.asarray(mask_rgb.convert("RGB"), dtype=np.uint8)
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    colored = (np.abs(r - g) + np.abs(r - b) + np.abs(g - b)) > 20
    return int(colored.sum())


# ============================
# Model loading
# ============================
def load_model_and_processor():
    have_cuda = torch.cuda.is_available() and not FORCE_CPU
    device = torch.device(f"cuda:{GPU_INDEX}" if have_cuda else "cpu")

    # Processor
    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            min_pixels=QWEN_MIN_PIXELS,
            max_pixels=QWEN_MAX_PIXELS,
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(MODEL_ID)

    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.padding_side = "left"
        tok.truncation_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

    used_quant = False

    # If you're judging QUALITY: do NOT quantize. Use the base BF16/FP16 path.
    # Qwen3-VL-2B weights are BF16 on HF. :contentReference[oaicite:6]{index=6}
    if have_cuda:
        # Qwen3 model card uses dtype="auto" and device_map="auto". :contentReference[oaicite:7]{index=7}
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map={"": 0},  
            low_cpu_mem_usage=True,
        ).eval()
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            low_cpu_mem_usage=True,
        ).eval().to(device)

    if tok is not None and hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id

    try:
        model.config.use_cache = True
    except Exception:
        pass

    return model, processor, device, used_quant



# ============================
# MAIN
# ============================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[env] torch: {torch.__version__}")
    print(f"[env] cuda_available: {torch.cuda.is_available()} count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [env] cuda {i} : {torch.cuda.get_device_name(i)}")

    print("[load] model...")
    model, processor, device, used_quant = load_model_and_processor()
    print(f"[load] used_quant={used_quant} QUANT_MODE={QUANT_MODE} _HAS_BNB={_HAS_BNB} INT8_CPU_OFFLOAD={INT8_CPU_OFFLOAD}")
    print(f"[cfg] device={device} dtype={DTYPE} PROC_LONGEST_EDGE={PROC_LONGEST_EDGE} panel_edge={PROC_LONGEST_EDGE//2}")
    print(f"[cfg] BATCH_SIZE={BATCH_SIZE} max_new_tokens={MAX_NEW_TOKENS} suggestion_limit={SUGGESTION_LIMIT}")
    print(f"[vram] after load: {_vram_str()}")

    idxs = find_indices_from_cluster_maps()
    if not idxs:
        print(f"[error] no ClusterMaps/processed_<idx>/clusters.json found in {CLUSTER_MAP_DIR}")
        return

    img_index, json_index = ensure_indexes()

    gen_kwargs = dict(
        max_new_tokens=int(MAX_NEW_TOKENS),
        do_sample=bool(DO_SAMPLE),
        num_beams=1,
        use_cache=True,
    )

    PANEL_EDGE = max(64, int(PROC_LONGEST_EDGE) // 2)

    for idx in idxs:
        t0 = time.time()

        full_img_path_s = img_index.get(str(idx))
        if not full_img_path_s:
            print(f"[skip] processed_{idx}: full image path not found in cache")
            continue
        full_img_path = Path(full_img_path_s)
        if not full_img_path.exists():
            print(f"[skip] processed_{idx}: cached full image missing: {full_img_path}")
            continue

        cmap_path = cluster_map_path_for_idx(idx)
        renders_dir = cluster_renders_dir_for_idx(idx)
        if not cmap_path.exists() or not renders_dir.exists():
            print(f"[skip] processed_{idx}: missing clusters.json or renders dir")
            continue

        try:
            cmap = json.loads(cmap_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[skip] processed_{idx}: bad clusters.json: {e}")
            continue

        entries = cmap.get("clusters") or []
        if not isinstance(entries, list) or not entries:
            print(f"[skip] processed_{idx}: no clusters")
            continue

        suggestions = load_candidate_labels(idx, json_index)[: int(SUGGESTION_LIMIT)]

        # Prepare full image base once
        try:
            full_img0 = Image.open(full_img_path).convert("RGB")
            full_w0, full_h0 = full_img0.size

            right_panel_base, full_scale, full_rw, full_rh, full_pad_x, full_pad_y = _fit_into_square(full_img0, PANEL_EDGE)
            resized_full_base = full_img0.resize((full_rw, full_rh), resample=Image.BICUBIC)
        except Exception as e:
            print(f"[skip] processed_{idx}: cannot open/prepare full: {e}")
            continue

        out_folder = OUT_DIR / f"processed_{idx}"
        out_folder.mkdir(parents=True, exist_ok=True)
        out_json_path = out_folder / "labels.json"

        results: Dict[str, Any] = {
            "image_index": idx,
            "full_image_source": str(full_img_path),
            "clusters_json": str(cmap_path),
            "candidate_labels": suggestions,
            "model_id": MODEL_ID,
            "quant_mode": QUANT_MODE,
            "used_quant": bool(used_quant),
            "device": str(device),
            "proc_longest_edge": int(PROC_LONGEST_EDGE),
            "panel_edge": int(PANEL_EDGE),
            "batch_size": int(BATCH_SIZE),
            "clusters": {},
            "cluster_order": [],
        }

        print(f"[run] processed_{idx}: clusters={len(entries)} full={full_img_path.name} vram={_vram_str()}")

        # Build work list (entry + computed static fields)
        work: List[Tuple[int, Any, Path, str, str, Optional[str], List[int], List[float]]] = []
        for ci, entry in enumerate(entries, start=1):
            bbox_xyxy = entry.get("bbox_xyxy")
            cluster_key = f"cluster_{ci:04d}"
            mask_name_for_key = entry.get("crop_file_mask")
            if isinstance(mask_name_for_key, str) and mask_name_for_key.strip():
                cluster_key = f"{ci:04d}_{mask_name_for_key.strip()}"

            if not (isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4):
                _store_cluster_result(results, cluster_key, {"entry": entry, "error": "bad_bbox"})
                if PRINT_EVERY_CLUSTER:
                    print(f"  [cluster {ci}/{len(entries)}] bad_bbox vram={_vram_str()}")
                continue

            bbox_xyxy = [int(x) for x in bbox_xyxy]
            bbox_norm = _bbox_norm_xyxy(bbox_xyxy, full_w0, full_h0)

            mask_name = entry.get("crop_file_mask")
            if not isinstance(mask_name, str):
                _store_cluster_result(results, cluster_key, {"entry": entry, "error": "missing_mask_name"})
                if PRINT_EVERY_CLUSTER:
                    print(f"  [cluster {ci}/{len(entries)}] missing_mask_name vram={_vram_str()}")
                continue

            mask_name = mask_name.strip()
            if not mask_name:
                _store_cluster_result(results, cluster_key, {"entry": entry, "error": "missing_mask_name"})
                if PRINT_EVERY_CLUSTER:
                    print(f"  [cluster {ci}/{len(entries)}] missing_mask_name vram={_vram_str()}")
                continue

            mask_path = renders_dir / mask_name
            if not mask_path.exists():
                _store_cluster_result(results, cluster_key, {"entry": entry, "mask_path": str(mask_path), "error": "mask_missing"})
                if PRINT_EVERY_CLUSTER:
                    print(f"  [cluster {ci}/{len(entries)}] mask_missing vram={_vram_str()}")
                continue

            colour_hint = (
                (entry.get("color_name").strip().lower()
                 if isinstance(entry.get("color_name"), str) and entry.get("color_name").strip()
                 else None)
                or parse_colour_hint_from_filename(mask_name)
            )

            work.append((ci, entry, mask_path, cluster_key, mask_name, colour_hint, bbox_xyxy, bbox_norm))

        # Run in fixed batches
        bs = max(1, int(BATCH_SIZE))
        for start in range(0, len(work), bs):
            chunk = work[start:start + bs]
            if not chunk:
                continue

            texts_batch: List[str] = []
            images_batch: List[Image.Image] = []

            meta_batch: List[Dict[str, Any]] = []

            # Build batch inputs
            for (ci, entry, mask_path, cluster_key, mask_file, colour_hint, bbox_xyxy, bbox_norm) in chunk:
                try:
                    img_mask0 = Image.open(mask_path).convert("RGB")
                except Exception as e:
                    _store_cluster_result(results, cluster_key, {"entry": entry, "mask_path": str(mask_path), "error": f"mask_load_failed: {e}"})
                    if PRINT_EVERY_CLUSTER:
                        print(f"  [cluster {ci}/{len(entries)}] mask_load_failed vram={_vram_str()}")
                    continue

                if ENABLE_FAST_REJECT:
                    try:
                        colored = _count_colored_pixels(img_mask0)
                        if colored < int(MIN_COLORED_PIXELS):
                            _store_cluster_result(results, cluster_key, {
                                "entry": entry,
                                "mask_path": str(mask_path),
                                "bbox_xyxy": bbox_xyxy,
                                "fast_reject": True,
                                "colored_pixels": int(colored),
                                "final": {"decision": "reject", "label": None, "from_suggestions": False, "confidence": 0.0, "reason": "too few colored pixels"},
                            })
                            if PRINT_EVERY_CLUSTER:
                                print(f"  [cluster {ci}/{len(entries)}] fast_reject colored={colored} vram={_vram_str()}")
                            continue
                    except Exception:
                        pass

                left_panel_sq, _, _, _, _, _ = _fit_into_square(img_mask0, PANEL_EDGE)

                bbox_resized = _scale_bbox_xyxy(bbox_xyxy, full_scale)
                bbox_resized = _clamp_bbox_xyxy(bbox_resized, full_rw, full_rh)

                bbox_panel = [
                    bbox_resized[0] + full_pad_x,
                    bbox_resized[1] + full_pad_y,
                    bbox_resized[2] + full_pad_x,
                    bbox_resized[3] + full_pad_y,
                ]

                # right_panel_base is already a square with the resized full image pasted at pad_x/pad_y
                right_panel_sq = right_panel_base.copy()
                _draw_red_rect_pil(right_panel_sq, bbox_panel, thickness=RECT_THICKNESS_PX)

                composite = _make_composite(left_panel_sq, right_panel_sq)

                prompt_text = build_prompt(
                    colour_hint=colour_hint,
                    suggestions=suggestions,
                )
                SYSTEM_MSG = (
                    "You are a visual object labeling engine\n"
                    "Assosiate objects only to provided materials as opposed to applying general knowledge\n"
                    "If the image is too ambigous reject\n"
                )

                # SmolVLM2 expects chat-template formatting (role/content with image+text).
                # One "conversation" per sample; batching is a list of conversations.
                conv = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                prompt_str = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)


                texts_batch.append(prompt_str)
                images_batch.append(composite)

                meta_batch.append({
                    "ci": ci,
                    "cluster_key": cluster_key,
                    "mask_file": str(mask_file),
                    "entry": entry,
                    "mask_path": str(mask_path),
                    "colour_hint": colour_hint,
                    "bbox_xyxy": bbox_xyxy,
                    "bbox_resized_xyxy": bbox_resized,
                    "bbox_right_panel_xyxy": bbox_panel,
                    "bbox_norm_xyxy": bbox_norm,
                })

            if not texts_batch:
                continue

            # Batch encode -> move -> generate
            t1 = time.time()
            try:
                inputs = processor(
                    text=texts_batch,
                    images=images_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )

                for k, v in list(inputs.items()):
                    if torch.is_tensor(v):
                        if k == "pixel_values":
                            inputs[k] = v.to(device=device, dtype=DTYPE, non_blocking=True)
                        else:
                            inputs[k] = v.to(device=device, non_blocking=True)

                with torch.inference_mode():
                    out_ids = model.generate(**inputs, **gen_kwargs)

                # Decode ONLY the generated continuation (not the prompt).
                # generate() returns [prompt + new_tokens] by default.
                if "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
                    prompt_len = int(inputs["input_ids"].shape[1])
                    gen_only_ids = out_ids[:, prompt_len:]
                else:
                    gen_only_ids = out_ids

                texts_out = processor.batch_decode(gen_only_ids, skip_special_tokens=True)
                dt_batch = time.time() - t1

            except Exception as e:
                if torch.cuda.is_available() and not FORCE_CPU:
                    torch.cuda.empty_cache()
                # Record failure for each element of this batch
                for meta in meta_batch:
                    meta_err = dict(meta)
                    meta_err["error"] = f"batch_inference_failed: {e}"
                    _store_cluster_result(results, str(meta.get("cluster_key", f"cluster_{meta.get('ci','?')}")), meta_err)
                    if PRINT_EVERY_CLUSTER:
                        ci = meta.get("ci", "?")
                        print(f"  [cluster {ci}/{len(entries)}] batch_inference_failed ({e}) vram={_vram_str()}")
                continue

            # Unpack per sample
            # Per-cluster dt is not truly measurable inside a batch without CUDA events per sample,
            # so we print batch dt and also dt/len as an average.
            avg_dt = dt_batch / max(1, len(texts_out))

            n_out = min(len(texts_out), len(meta_batch))
            for j in range(n_out):
                text_out = texts_out[j]
                meta = meta_batch[j]
                ci = int(meta["ci"])
                cluster_key = str(meta.get("cluster_key", f"cluster_{ci:04d}"))

                json_objects = extract_json_objects(text_out)
                parsed = json_objects[0] if json_objects else None
                final = parsed
                capped = False
                if isinstance(final, dict):
                    conf = final.get("confidence")
                    dec = final.get("decision")
                    if isinstance(conf, (int, float)) and isinstance(dec, str):
                        if dec.lower() == "accept" and float(conf) < float(MIN_ACCEPT_CONF):
                            final = dict(final)
                            final["decision"] = "reject"
                            final["label"] = None
                            final["reason"] = (final.get("reason") or "") + " | auto-reject"
                            capped = True

                _store_cluster_result(results, cluster_key, {
                    "entry": meta["entry"],
                    "mask_file": meta.get("mask_file"),
                    "mask_path": meta["mask_path"],
                    "colour_hint": meta["colour_hint"],
                    "bbox_xyxy": meta["bbox_xyxy"],
                    "bbox_resized_xyxy": meta["bbox_resized_xyxy"],
                    "bbox_right_panel_xyxy": meta["bbox_right_panel_xyxy"],
                    "bbox_norm_xyxy": meta["bbox_norm_xyxy"],
                    "batch_size": len(texts_out),
                    "batch_timing_s": float(f"{dt_batch:.4f}"),
                    "avg_per_item_s": float(f"{avg_dt:.4f}"),
                    "model_raw": text_out,
                    "json_objects": json_objects,
                    "parsed": parsed,
                    "final": final,
                    "auto_capped": capped,
                })

                if PRINT_EVERY_CLUSTER:
                    print(f"  [cluster {ci}/{len(entries)}] ok batch_dt={dt_batch:.2f}s avg={avg_dt:.2f}s vram={_vram_str()}")

            if EMPTY_CACHE_EVERY and ((start // bs + 1) % int(EMPTY_CACHE_EVERY) == 0):
                if torch.cuda.is_available() and not FORCE_CPU:
                    torch.cuda.empty_cache()

        out_json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] processed_{idx}: wrote {out_json_path} ({time.time()-t0:.2f}s) vram={_vram_str()}")

        if torch.cuda.is_available() and not FORCE_CPU:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
