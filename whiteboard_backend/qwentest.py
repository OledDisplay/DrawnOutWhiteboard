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
# PATHS
# ============================
BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "ProccessedImages"
CLUSTER_RENDER_DIR = BASE_DIR / "ClusterRenders"
CLUSTER_MAP_DIR = BASE_DIR / "ClusterMaps"

OUT_DIR = BASE_DIR / "ClustersLabeled"
CACHE_DIR = BASE_DIR / "_path_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
IMG_CACHE_PATH = CACHE_DIR / "processed_png_index.json"
JSON_CACHE_PATH = CACHE_DIR / "processed_json_index.json"

SKIP_EXISTING_LABELS = True



# ============================
# MODEL CONFIG 
# ============================
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

QWEN_MIN_PIXELS = 256 * 28 * 28
QWEN_MAX_PIXELS = 512 * 28 * 28



GPU_INDEX = 0
FORCE_CPU = False

QUANT_MODE = None # "4bit" | "8bit" | "none"
INT8_CPU_OFFLOAD = False 


# ============================
# BATCHING 
# ============================
# Set to 1 => strict cluster-by-cluster (lowest VRAM).
# Set >1 => batching (faster, more VRAM).
BATCH_SIZE = 8

# ============================
# SPEED / MEMORY LEVERS
# ============================
# ONE composite image = two square panels side-by-side.
# PANEL_EDGE = PROC_LONGEST_EDGE//2
PROC_LONGEST_EDGE = 600 # composite max side;

SUGGESTION_LIMIT = 40
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
_FULL_IMG_RE = re.compile(r"processed_(\d+)\.png$", re.IGNORECASE)
_FULL_JSON_RE = re.compile(r"processed_(\d+)\.json$", re.IGNORECASE)
_CLUSTER_DIR_RE = re.compile(r"processed_(\d+)$", re.IGNORECASE)

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

def _load_cache(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """
    Returns (map, meta). meta may contain 'root', 'created_utc', etc.
    """
    if not path.exists():
        return {}, {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("map"), dict):
            m = {str(k): str(v) for k, v in obj["map"].items()}
            return m, obj
    except Exception:
        pass
    return {}, {}


def _prune_missing_files(mapping: dict[str, str], *, must_be_file: bool = True) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in mapping.items():
        p = Path(v)
        try:
            if must_be_file:
                if p.is_file():
                    out[k] = v
            else:
                if p.exists():
                    out[k] = v
        except Exception:
            # Path() can throw on weird Windows path edge cases; treat as invalid.
            pass
    return out


def ensure_indexes() -> tuple[dict[str, str], dict[str, str]]:
    # Load (map, meta)
    img_map, img_meta = _load_cache(IMG_CACHE_PATH)
    json_map, json_meta = _load_cache(JSON_CACHE_PATH)

    # If cache was built for a different root, rebuild (prevents stale-project issues)
    cur_root = str(PROCESSED_DIR.resolve())
    img_root = str(img_meta.get("root", ""))
    json_root = str(json_meta.get("root", ""))

    # Prune missing entries first (cheap)
    if img_map:
        img_map = _prune_missing_files(img_map, must_be_file=True)
    if json_map:
        json_map = _prune_missing_files(json_map, must_be_file=True)

    # Rebuild conditions
    def need_rebuild(map_now: dict[str, str], root_now: str) -> bool:
        if not map_now:
            return True
        if root_now and (Path(root_now).resolve().as_posix() != Path(cur_root).resolve().as_posix()):
            return True
        return False

    if need_rebuild(img_map, img_root):
        print("[scan] rebuilding PNG index cache...")
        img_map = _build_processed_index(_FULL_IMG_RE, "png")
        _save_cache(IMG_CACHE_PATH, img_map)

    if need_rebuild(json_map, json_root):
        print("[scan] rebuilding JSON index cache...")
        json_map = _build_processed_index(_FULL_JSON_RE, "json")
        _save_cache(JSON_CACHE_PATH, json_map)

    return img_map, json_map


def load_candidate_labels(idx: int, json_index: Dict[str, str]) -> Tuple[List[str], str]:
    p = json_index.get(str(idx))
    if not p:
        return [], ""
    path = Path(p)
    if not path.exists():
        return [], ""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [], ""

    # NEW: base_context comes from processed_<idx>.json (added by your other script)
    base_context = data.get("base_context")
    if not isinstance(base_context, str):
        base_context = ""
    base_context = base_context.strip()

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

    return out, base_context



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

def build_label_refine_prompt(base_context: str, raw_candidates: List[str]) -> str:
    bc = (base_context or "").strip()
    if not bc:
        bc = "unknown"

    # keep it short to avoid garbage / token spam
    cand = raw_candidates[:80]
    cand_text = ", ".join(cand) if cand else "(none)"

    return (
        f"You are given a noisy list of candidate words, describing parts of a(n) {base_context} - MAIN .\n\n"
        f"Return a NEW list of OBJECT COMPONENTS / PARTS that the {base_context} would be built from.\n"
        "Fill in gaps in the pruned list so that it becomes a set of ALL objects that build up MAIN.\n\n"
        "Fill the new list with as many unique objects as possible"
        "Examples:\n"
        "- face -> f each face part\n"
        "- eukaryotic cell -> all organels\n"
        "- car -> wheel, tire, rim, door...n\n"
        f"RAW CANDIDATES: {cand_text}\n\n"
        "Include a very consise description of how each object is represented and differentiated VISUALLY"
        "RULES:\n"
        "- Output ONLY JSON\n"
        "- Example JSON in format:{"
        "{\"labels\": [{label : description, label : description}]}\n"
        "- labels must be short strings (1-4 words max)\n"
        "- no duplicates\n"
    )


def _parse_refined_labels(text_out: str) -> List[str]:
    obj = extract_json_object(text_out)

    if isinstance(obj, dict):
        labels = obj.get("labels")

        if isinstance(labels, list):
            out: List[str] = []
            seen = set()

            for item in labels:
                name = None

                if isinstance(item, str):
                    name = item
                elif isinstance(item, dict):
                    n = item.get("name")
                    if isinstance(n, str):
                        name = n

                if not isinstance(name, str):
                    continue

                c = _clean_word(name)
                if not c:
                    continue
                k = c.lower()
                if k in seen:
                    continue
                seen.add(k)
                out.append(c)

            if out:
                return out

    # fallback: try comma/newline split if model didn't follow JSON
    out: List[str] = []
    seen = set()
    for part in re.split(r"[,;\n]+", (text_out or "").strip()):
        c = _clean_word(part)
        if not c:
            continue
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out



def refine_candidate_labels_with_qwen(
    model,
    processor,
    device,
    base_context: str,
    raw_candidates: List[str],
) -> List[str]:
    SYSTEM_MSG = (
        "You are a label-refinement engine.\n"
        "You produce component/part labels, not generic category names.\n"
        "Return only JSON.\n"
    )

    prompt_text = build_label_refine_prompt(base_context, raw_candidates)

    conv = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]
    prompt_str = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    # slightly more tokens than the main run because it must output a list
    gen_kwargs_refine = dict(
        max_new_tokens=140,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

    try:
        inputs = processor(
            text=[prompt_str],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device=device, non_blocking=True)

        with torch.inference_mode():
            out_ids = model.generate(**inputs, **gen_kwargs_refine)

        if "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
            prompt_len = int(inputs["input_ids"].shape[1])
            gen_only_ids = out_ids[:, prompt_len:]
        else:
            gen_only_ids = out_ids

        text_out = processor.batch_decode(gen_only_ids, skip_special_tokens=True)[0]
        refined = _parse_refined_labels(text_out)

        # if model produced nothing useful -> fallback to original candidates
        if not refined:
            return raw_candidates[:SUGGESTION_LIMIT]

        return refined

    except Exception as e:
        print(f"[warn] refine_candidate_labels_with_qwen failed: {e}")
        return raw_candidates[:SUGGESTION_LIMIT]



# ============================
# Prompt + parse (ONE IMAGE)
# ============================
def build_prompt(
    colour_hint: Optional[str],
    suggestions: List[str],
    base_context: str,
) -> str:

    colour_txt = colour_hint if colour_hint else "unknown"
    sug = suggestions[:SUGGESTION_LIMIT]

    # Keep suggestions readable but short
    if sug:
        sug_text = ",".join(sug)
    else:
        sug_text = "(none)"

    return (
        "Input : Two images merged horizontally\n"
        f"- LEFT: Cropped box with with coloured {colour_txt} main object and grayscaled surroundings\n"
        f"- RIGHT: A(n) {base_context}, LEFT crop from there\n\n"
        f"Visually charecterize the features of LEFT's object and its placement in the {base_context} based on foreground.\n"
        "Look at RIGHT and LEFTS FOREGROUND and link the crop to WIDER surroundings (their shapes, colours) for RELATIVE context \n"
        "Based on LEFT's full semantic profile label it, matching it's characteristics with a candidate  label:\n\n"
        f"{sug_text}\n\n"
        "Return ONLY JSON with REQUIRED keys:\n"
        "{"
        "\"label\":string|null,"
        "\"full_visual_LEFT\": string,"
        "\"LEFT_SURROUNDINGS\": string,"
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

    # Use disk offload to avoid CPU RAM spikes during loading
    offload_dir = str((BASE_DIR / "_hf_offload").resolve())
    os.makedirs(offload_dir, exist_ok=True)

    # Quant config (actually used now)
    quant_config = None
    if have_cuda and QUANT_MODE in ("4bit", "8bit") and _HAS_BNB:
        if QUANT_MODE == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=DTYPE,
            )
            used_quant = True
        elif QUANT_MODE == "8bit":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=bool(INT8_CPU_OFFLOAD),
            )
            used_quant = True

    if have_cuda:
        # Prefer auto placement + offload knobs (reduces CPU peak)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype="auto" if not used_quant else None,
            device_map="auto",
            low_cpu_mem_usage=True,
            offload_state_dict=True,
            offload_folder=offload_dir,
            quantization_config=quant_config,
        ).eval()
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            low_cpu_mem_usage=True,
            offload_state_dict=True,
            offload_folder=offload_dir,
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
# Preload + warmup (for orchestrator)
# ============================
def _make_dummy_vl_sample() -> tuple[str, Image.Image]:
    # Small image keeps warmup cheap
    img = Image.new("RGB", (32, 32), (0, 0, 0))
    SYSTEM_MSG = "You are a test harness. Reply with JSON only."
    prompt_text = 'Return ONLY JSON: {"ok": true}'
    conv = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
    ]
    return conv, img

def _img_tensor_dtype(device: torch.device) -> torch.dtype:
    # CPU fp16 is a common source of pain; keep CPU on fp32.
    return DTYPE if device.type == "cuda" else torch.float32


def _as_rgb_uint8(arr: Any) -> Optional[np.ndarray]:
    """
    Accepts: np arrays in RGB/BGR, PIL Image, etc.
    Returns: np.ndarray RGB uint8 (H,W,3) or None.
    """
    if arr is None:
        return None

    if isinstance(arr, Image.Image):
        arr = np.asarray(arr.convert("RGB"), dtype=np.uint8)

    if not isinstance(arr, np.ndarray):
        return None

    if arr.dtype != np.uint8:
        try:
            arr = arr.astype(np.uint8, copy=False)
        except Exception:
            return None

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim != 3 or arr.shape[2] < 3:
        return None

    # If 4ch, drop alpha.
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]

    return arr


def _extract_mask_map(pack: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Tries multiple key shapes that ImageClusters might use.
    Returns dict: mask_filename -> RGB uint8 array.
    """
    # Common dict-style keys
    dict_keys_rgb = [
        "renders_mask_rgb",
        "mask_renders_rgb",
        "mask_crops_rgb",
        "render_masks_rgb",
    ]
    dict_keys_bgr = [
        "renders_mask_bgr",
        "mask_renders_bgr",
        "mask_crops_bgr",
        "render_masks_bgr",
    ]

    for k in dict_keys_rgb:
        mm = pack.get(k)
        if isinstance(mm, dict):
            out: Dict[str, np.ndarray] = {}
            for name, img in mm.items():
                rgb = _as_rgb_uint8(img)
                if rgb is not None:
                    out[str(name)] = rgb
            if out:
                return out

    for k in dict_keys_bgr:
        mm = pack.get(k)
        if isinstance(mm, dict):
            out: Dict[str, np.ndarray] = {}
            for name, img in mm.items():
                bgr = _as_rgb_uint8(img)
                if bgr is None:
                    continue
                # treat as BGR -> convert to RGB
                rgb = bgr[:, :, ::-1].copy()
                out[str(name)] = rgb
            if out:
                return out

    # List-style payloads (less common)
    list_keys = [
        "mask_crops",
        "mask_renders",
        "renders_masks",
        "mask_images",
    ]
    for k in list_keys:
        lst = pack.get(k)
        if isinstance(lst, list):
            out: Dict[str, np.ndarray] = {}
            for it in lst:
                if not isinstance(it, dict):
                    continue
                name = it.get("name") or it.get("file") or it.get("filename")
                img = it.get("rgb") or it.get("img_rgb") or it.get("image_rgb")
                if img is None:
                    img = it.get("bgr") or it.get("img_bgr") or it.get("image_bgr")
                    bgr = _as_rgb_uint8(img)
                    if bgr is not None:
                        out[str(name)] = bgr[:, :, ::-1].copy()
                    continue
                rgb = _as_rgb_uint8(img)
                if rgb is not None:
                    out[str(name)] = rgb
            if out:
                return out

    return {}



def warmup_qwen_once(model, processor, device: torch.device) -> None:
    conv, img = _make_dummy_vl_sample()
    prompt_str = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[prompt_str],
        images=[img],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    img_dtype = _img_tensor_dtype(device)

    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            if k == "pixel_values":
                inputs[k] = v.to(device=device, dtype=img_dtype, non_blocking=True)
            else:
                inputs[k] = v.to(device=device, non_blocking=True)

    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=8, do_sample=False, num_beams=1, use_cache=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

def _clean_short_label(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    t = re.sub(r"\s+", " ", s.strip())
    if not t:
        return None

    # keep short
    parts = t.split(" ")
    if len(parts) > 4:
        t = " ".join(parts[:4]).strip()

    if len(t) > 60:
        t = t[:60].strip()

    return t or None


def _resize_longest_side(img: Image.Image, max_side: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (max_side, max_side), (0, 0, 0))

    longest = max(w, h)
    if longest <= max_side:
        return img

    scale = float(max_side) / float(longest)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), resample=Image.BICUBIC)




def preload_qwen_cpu_only(
    model_id: str = MODEL_ID,
) -> tuple[Any, Any, torch.device]:
    """
    Loads Qwen on CPU and runs a dummy prompt.
    Safe to do while Paddle/other GPU work is running.
    """
    device = torch.device("cpu")

    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=QWEN_MIN_PIXELS,
            max_pixels=QWEN_MAX_PIXELS,
            trust_remote_code=True,
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.padding_side = "left"
        tok.truncation_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

    # CPU: use fp32 (much less likely to break)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval().to(device)

    if tok is not None and hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id

    try:
        model.config.use_cache = True
    except Exception:
        pass

    warmup_qwen_once(model, processor, device)
    return model, processor, device



def move_qwen_to_cuda_if_available(
    model,
    *,
    gpu_index: int = GPU_INDEX,
) -> tuple[Any, torch.device]:
    if torch.cuda.is_available() and not FORCE_CPU:
        device = torch.device(f"cuda:{gpu_index}")
        model = model.to(device)
        model.eval()

        # Try to reduce VRAM by casting after moving (safe-ish, but guard it)
        try:
            if hasattr(model, "to"):
                model = model.to(dtype=DTYPE)
        except Exception:
            pass

        return model, device

    return model, torch.device("cpu")



# ============================
# In-memory entrypoint (called by Orchestrator)
# ============================
def label_clusters_transformers(
    clusters_state: Dict[int, Dict[str, Any]],
    model,
    processor,
    device: torch.device,
    *,
    save_outputs: bool = False,
) -> Dict[int, Dict[str, Any]]:

    out_dir = (Path(__file__).resolve().parent / "ClustersLabeled")
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_edge = max(64, int(PROC_LONGEST_EDGE) // 2)

    gen_kwargs = dict(
        max_new_tokens=int(MAX_NEW_TOKENS),
        do_sample=bool(DO_SAMPLE),
        num_beams=1,
        use_cache=True,
    )

    img_dtype = _img_tensor_dtype(device)

    results_by_idx: Dict[int, Dict[str, Any]] = {}

    

    for idx in sorted(clusters_state.keys()):

        if save_outputs and SKIP_EXISTING_LABELS:
            folder = out_dir / f"processed_{idx}"
            labels_path = folder / "labels.json"
            if labels_path.exists():
                try:
                    results_by_idx[idx] = json.loads(labels_path.read_text(encoding="utf-8"))
                    print(f"[skip] idx={idx}: existing labels.json loaded")
                    continue
                except Exception:
                    pass

        pack = clusters_state[idx]
        t0 = time.time()

        base_context = str(pack.get("base_context", "") or "")
        suggestions_all = list(pack.get("candidate_labels_raw", []) or [])

        refined = refine_candidate_labels_with_qwen(
            model=model,
            processor=processor,
            device=device,
            base_context=base_context,
            raw_candidates=suggestions_all,
        ) if suggestions_all else []

        suggestions = (refined or suggestions_all)[: int(SUGGESTION_LIMIT)]

        # full_img_rgb is required (your pipeline provides it)
        full_img_rgb = _as_rgb_uint8(pack.get("full_img_rgb"))
        if full_img_rgb is None:
            print(f"[skip] idx={idx}: missing full_img_rgb")
            continue

        full_img0 = Image.fromarray(full_img_rgb, mode="RGB")
        right_panel_base, full_scale, full_rw, full_rh, full_pad_x, full_pad_y = _fit_into_square(full_img0, panel_edge)

        results: Dict[str, Any] = {
            "image_index": int(idx),
            "candidate_labels_raw": suggestions_all[: int(SUGGESTION_LIMIT)],
            "candidate_labels_refined": suggestions,
            "base_context": base_context,
            "model_id": str(getattr(model, "name_or_path", MODEL_ID)),
            "proc_longest_edge": int(PROC_LONGEST_EDGE),
            "panel_edge": int(panel_edge),
            "batch_size": int(BATCH_SIZE),
            "clusters": {},
            "cluster_order": [],
        }
        results_by_idx[idx] = results

        entries = pack.get("clusters") or []
        if not isinstance(entries, list) or not entries:
            print(f"[skip] idx={idx}: pack has no clusters list")
            continue

        # Get masks from pack (robust), else fallback to disk renders
        mask_map = _extract_mask_map(pack)

        renders_dir = CLUSTER_RENDER_DIR / f"processed_{idx}"

        def _load_mask_from_disk(mask_name: str) -> Optional[np.ndarray]:
            p = renders_dir / mask_name
            if not p.exists():
                return None
            try:
                img = Image.open(p).convert("RGB")
                return np.asarray(img, dtype=np.uint8)
            except Exception:
                return None

        def _store(cluster_key: str, payload: Dict[str, Any]) -> None:
            if cluster_key not in results["clusters"]:
                results["cluster_order"].append(cluster_key)
            results["clusters"][cluster_key] = payload

        # Build tasks
        tasks: List[Dict[str, Any]] = []
        for ci, entry in enumerate(entries, start=1):
            bbox_xyxy = entry.get("bbox_xyxy")
            mask_name = entry.get("crop_file_mask")

            if not (isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4):
                continue
            if not isinstance(mask_name, str) or not mask_name.strip():
                continue

            mask_name = mask_name.strip()

            mask_rgb = mask_map.get(mask_name)
            if mask_rgb is None:
                # fallback to disk if available
                mask_rgb = _load_mask_from_disk(mask_name)

            if mask_rgb is None:
                continue

            colour_hint = None
            if isinstance(entry.get("color_name"), str) and entry.get("color_name").strip():
                colour_hint = entry["color_name"].strip().lower()
            else:
                colour_hint = parse_colour_hint_from_filename(mask_name)

            tasks.append({
                "ci": ci,
                "entry": entry,
                "bbox_xyxy": [int(x) for x in bbox_xyxy],
                "mask_name": mask_name,
                "mask_rgb": mask_rgb,
                "colour_hint": colour_hint,
            })

        # Batch inference
        bs = max(1, int(BATCH_SIZE))
        for start in range(0, len(tasks), bs):
            chunk = tasks[start:start + bs]
            if not chunk:
                continue

            texts_batch: List[str] = []
            images_batch: List[Image.Image] = []
            meta_batch: List[Dict[str, Any]] = []

            for task in chunk:
                ci = int(task["ci"])
                entry = task["entry"]
                bbox_xyxy = task["bbox_xyxy"]
                mask_rgb = task["mask_rgb"]
                colour_hint = task.get("colour_hint")

                cluster_key = f"{ci:04d}_{task['mask_name']}"

                left_img = Image.fromarray(_as_rgb_uint8(mask_rgb), mode="RGB")
                left_sq, _, _, _, _, _ = _fit_into_square(left_img, panel_edge)

                bbox_resized = _scale_bbox_xyxy(bbox_xyxy, full_scale)
                bbox_resized = _clamp_bbox_xyxy(bbox_resized, full_rw, full_rh)
                bbox_panel = [
                    bbox_resized[0] + full_pad_x,
                    bbox_resized[1] + full_pad_y,
                    bbox_resized[2] + full_pad_x,
                    bbox_resized[3] + full_pad_y,
                ]

                right_sq = right_panel_base.copy()
                _draw_red_rect_pil(right_sq, bbox_panel, thickness=RECT_THICKNESS_PX)

                composite = _make_composite(left_sq, right_sq)

                prompt_text = build_prompt(colour_hint, suggestions, base_context)

                SYSTEM_MSG = (
                    "You are a visual object labeling engine.\n"
                    "Use ONLY the provided candidate labels.\n"
                    "Return ONLY JSON.\n"
                )
                conv = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
                ]
                prompt_str = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

                texts_batch.append(prompt_str)
                images_batch.append(composite)

                meta_batch.append({
                    "ci": ci,
                    "cluster_key": cluster_key,
                    "entry": entry,
                    "colour_hint": colour_hint,
                    "bbox_xyxy": bbox_xyxy,
                    "bbox_right_panel_xyxy": bbox_panel,
                })

            if not texts_batch:
                continue

            t_req0 = time.time()
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
                            inputs[k] = v.to(device=device, dtype=img_dtype, non_blocking=True)
                        else:
                            inputs[k] = v.to(device=device, non_blocking=True)

                with torch.inference_mode():
                    out_ids = model.generate(**inputs, **gen_kwargs)

                if "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
                    prompt_len = int(inputs["input_ids"].shape[1])
                    gen_only_ids = out_ids[:, prompt_len:]
                else:
                    gen_only_ids = out_ids

                texts_out = processor.batch_decode(gen_only_ids, skip_special_tokens=True)
                dt = time.time() - t_req0

            except Exception as e:
                for meta in meta_batch:
                    _store(meta["cluster_key"], {"entry": meta["entry"], "error": f"batch_inference_failed: {e}"})
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            n_out = min(len(texts_out), len(meta_batch))
            for j in range(n_out):
                text_out = texts_out[j]
                meta = meta_batch[j]

                json_objects = extract_json_objects(text_out)
                parsed = json_objects[0] if json_objects else None

                _store(meta["cluster_key"], {
                    "entry": meta["entry"],
                    "colour_hint": meta["colour_hint"],
                    "bbox_xyxy": meta["bbox_xyxy"],
                    "bbox_right_panel_xyxy": meta["bbox_right_panel_xyxy"],
                    "timing_s": float(f"{dt:.4f}"),
                    "model_raw": text_out,
                    "json_objects": json_objects,
                    "parsed": parsed,
                })

        if save_outputs and SKIP_EXISTING_LABELS:
            folder = out_dir / f"processed_{idx}"
            labels_path = folder / "labels.json"
            if labels_path.exists():
                try:
                    results_by_idx[idx] = json.loads(labels_path.read_text(encoding="utf-8"))
                    print(f"[skip] idx={idx}: existing labels.json loaded")
                    continue
                except Exception:
                    pass


        print(f"[ok] idx={idx}: clusters={len(tasks)} dt={time.time()-t0:.2f}s")

    return results_by_idx

def build_stage1_visual_probe_prompt() -> str:
    # Strict schema; no semantic inference; small + consistent keys for stage2.
    return ( 
        "Make a full VISUAL characteristic of the image with WITH NO regard to semantic meaning\n"
          "You have to describe all the shapes and objects spread out through it from an analytical standpoint" 
          "Talk figures, clusters, membranes, blobs, ex." 
          "Describe how the object(s) are spread out and what area they cover in the image"
          "Do NOT infer meaning or identity.\n"
          "Output JSON ONLY with:\n" "{" "\"objects\":[\"shape\", ...],"
          "\"structure\":\"1-4 words\"," "\"dominant\"" "}\n"
          "Rules:\n" 
           "- objects: 3-8 items, each 1-4 words\n"
           "- structure: how the objects are located (center, spread out, ..) (1-4 words)\n" 
           "- description : a litteral full visual description of everything" "- no extra keys, no prose\n" 
        )


def _as_meta_score(v: Any) -> Optional[float]:
    """
    Normalize external score (final_score) into a float if possible.
    Keep it as-is (0..1 expected), clamp to sane bounds.
    """
    if v is None:
        return None
    try:
        s = float(v)
    except Exception:
        return None
    # clamp to avoid prompt garbage if something went wrong upstream
    if s != s:  # NaN
        return None
    return max(0.0, min(1.0, s))


def _parse_stage1_visual_probe(text_out: str) -> Dict[str, Any]:
    """
    New stage-1 structure:
      { "objects":[...], "structure":"...", "dominant":"...", "simplicity":1..5 }

    Parser is tolerant to minor key drift (labels->objects, layout->structure, etc).
    """
    obj = extract_json_object(text_out or "")
    if not isinstance(obj, dict):
        return {"objects": [], "structure": "", "dominant": "", "simplicity": 5}

    raw_objects = (
        obj.get("objects")
        if obj.get("objects") is not None else
        obj.get("labels")  # tolerate old key
    )
    raw_structure = (
        obj.get("structure")
        if obj.get("structure") is not None else
        obj.get("layout")  # tolerate variants
    )
    raw_dominant = (
        obj.get("dominant")
        if obj.get("dominant") is not None else
        obj.get("main_object")
    )
    raw_simplicity = (
        obj.get("simplicity")
        if obj.get("simplicity") is not None else
        obj.get("complexity")
    )

    out_objects: List[str] = []
    if isinstance(raw_objects, list):
        seen = set()
        for it in raw_objects:
            if not isinstance(it, str):
                continue
            c = _clean_short_label(it)
            if not c:
                continue
            k = c.lower()
            if k in seen:
                continue
            seen.add(k)
            out_objects.append(c)
            if len(out_objects) >= 8:
                break

    structure = ""
    if isinstance(raw_structure, str):
        structure = _clean_short_label(raw_structure) or ""

    dominant = ""
    if isinstance(raw_dominant, str):
        dominant = _clean_short_label(raw_dominant) or ""

    simp_i = 5
    try:
        simp_i = int(raw_simplicity)
    except Exception:
        simp_i = 5
    simp_i = max(1, min(5, simp_i))

    # Ensure minimum objects count
    if len(out_objects) < 3:
        # salvage from raw text (comma/newline split) if model didn't follow schema
        for part in re.split(r"[,;\n]+", (text_out or "").strip()):
            c = _clean_short_label(part)
            if not c:
                continue
            out_objects.append(c)
            if len(out_objects) >= 3:
                break

    return {
        "objects": out_objects[:8],
        "structure": structure,
        "dominant": dominant,
        "simplicity": simp_i,
    }


def probe_processed_images_stage1(
    *,
    model,
    processor,
    device: torch.device,
    items: List[Dict[str, Any]],
    batch_size: int = 8,
    longest_side: int = 600,
) -> Dict[str, Dict[str, Any]]:
    """
    items: [
      {
        "processed_id": "processed_12",
        "image_pil": PIL.Image,
        "base_context": "...",
        # NEW (optional): "final_score": 0.83  (or "score": 0.83)
      }, ...
    ]

    Returns:
      {
        processed_id: {
          "objects":[...],
          "structure":"...",
          "dominant":"...",
          "simplicity":1..5,
          "score": Optional[float]   # passthrough external meta score
        },
        ...
      }
    """
    SYSTEM_MSG = (
        "You are a visual transcription engine.\n"
        "Describe only visible shapes/layout.\n"
        "No guessing. Output JSON only.\n"
    )

    prompt_text = build_stage1_visual_probe_prompt()
    conv = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
    ]
    prompt_str = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    gen_kwargs = dict(
        max_new_tokens=160,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

    img_dtype = _img_tensor_dtype(device)
    out: Dict[str, Dict[str, Any]] = {}

    bs = max(1, int(batch_size))
    for start in range(0, len(items), bs):
        chunk = items[start:start + bs]
        if not chunk:
            continue

        texts_batch: List[str] = []
        images_batch: List[Image.Image] = []
        ids_batch: List[str] = []
        scores_batch: List[Optional[float]] = []

        for it in chunk:
            pid = str(it.get("processed_id", "") or "").strip()
            img = it.get("image_pil")
            if not pid or not isinstance(img, Image.Image):
                continue

            # NEW: passthrough meta score (final_score) so stage2 can use it
            score = _as_meta_score(it.get("final_score", None))
            if score is None:
                score = _as_meta_score(it.get("score", None))

            img2 = _resize_longest_side(img, int(longest_side))

            texts_batch.append(prompt_str)
            images_batch.append(img2)
            ids_batch.append(pid)
            scores_batch.append(score)

        if not texts_batch:
            continue

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
                        inputs[k] = v.to(device=device, dtype=img_dtype, non_blocking=True)
                    else:
                        inputs[k] = v.to(device=device, non_blocking=True)

            with torch.inference_mode():
                out_ids = model.generate(**inputs, **gen_kwargs)

            if "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
                prompt_len = int(inputs["input_ids"].shape[1])
                gen_only_ids = out_ids[:, prompt_len:]
            else:
                gen_only_ids = out_ids

            texts_out = processor.batch_decode(gen_only_ids, skip_special_tokens=True)

        except Exception as e:
            for pid, sc in zip(ids_batch, scores_batch):
                out[pid] = {
                    "objects": [],
                    "structure": "",
                    "dominant": "",
                    "simplicity": 5,
                    "score": sc,
                    "error": f"stage1_failed: {e}",
                }
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        n_out = min(len(texts_out), len(ids_batch))
        for j in range(n_out):
            pid = ids_batch[j]
            parsed = _parse_stage1_visual_probe(texts_out[j])
            parsed["score"] = scores_batch[j]  # NEW: attach external meta score
            out[pid] = parsed

    return out


def build_stage2_pick_prompt(
    base_context: str,
    stage1_map: Dict[str, Dict[str, Any]],
    processed_to_unique: Optional[Dict[str, str]] = None,
) -> str:
    bc = (base_context or "").strip() or "unknown"

    lines: List[str] = []
    for pid, info in stage1_map.items():
        objects = info.get("objects") or []
        structure = info.get("structure") or ""
        dominant = info.get("dominant") or ""
        simplicity = info.get("simplicity", 5)
        score = info.get("score", None)

        try:
            simp_i = int(simplicity)
        except Exception:
            simp_i = 5
        simp_i = max(1, min(5, simp_i))

        # keep objects short
        if isinstance(objects, list):
            obj_s = ",".join(str(x) for x in objects[:8])
        else:
            obj_s = ""

        # format score compactly
        sc_s = "NA"
        try:
            if isinstance(score, (int, float)):
                sc_s = f"{float(score):.3f}"
        except Exception:
            sc_s = "NA"

        up = ""
        if processed_to_unique and pid in processed_to_unique:
            up = processed_to_unique[pid]

        if up:
            lines.append(
                f"{pid} | score={sc_s} | simp={simp_i} | struct={structure} | dom={dominant} | objs={obj_s} | path={up}"
            )
        else:
            lines.append(
                f"{pid} | score={sc_s} | simp={simp_i} | struct={structure} | dom={dominant} | objs={obj_s}"
            )

    joined = "\n".join(lines)

    return ( 
        f"DESIRED IMAGE CONTENT: {bc}\n\n" 
        "You must pick EXACTLY 1 best image to keep.\n" 
        "Goal: a SIMPLE visual OBJECT - one full thing identified in the center of the screen.\n" 
        "Seperated complex compact detail is wanted. Example : Eukaryotic cell > nucleus + organells" 
        "Each candidate comes with an external score. When between two good candidates, pick based on score" 
        "REJECT:\n" "- Images filled with many things spread out\n" 
        "- flowcharts, clusters linked with arrouws, any dense text\n"
        "Candidates:\n" f"{joined}\n\n" 
        "Return JSON ONLY with this exact schema:\n" 
        "{" "\"candidate\":[" 
        "{\"processed_id\":\"...\",\"reason\":\"...\",\"main_object\":\"...\",\"simplicity\":1}" "]" "}\n" 
        "Rules:\n" "- exactly 1 candidates\n" 
        "- processed_id must match one of the listed ids\n" 
        "- reason and main_object must be 1-8 words (short)\n"
        "- simplicity integer 1..5\n" "- no extra keys\n" 
        )


def _parse_stage2_candidates(text_out: str) -> List[Dict[str, Any]]:
    obj = extract_json_object(text_out or "")
    if not isinstance(obj, dict):
        return []

    # Prefer the schema your prompt actually requests: "candidate": [...]
    cands = obj.get("candidate")
    if not isinstance(cands, list):
        # tolerate old key if the model drifts
        cands = obj.get("candidates")
    if not isinstance(cands, list):
        return []

    out: List[Dict[str, Any]] = []
    seen = set()

    for it in cands:
        if not isinstance(it, dict):
            continue

        pid = it.get("processed_id")
        if not isinstance(pid, str) or not pid.strip():
            continue
        pid = pid.strip()
        if pid in seen:
            continue
        seen.add(pid)

        reason = it.get("reason")
        if not isinstance(reason, str):
            reason = ""

        main_obj = it.get("main_object")
        if not isinstance(main_obj, str):
            main_obj = ""

        simp = it.get("simplicity", 5)
        try:
            simp_i = int(simp)
        except Exception:
            simp_i = 5
        simp_i = max(1, min(5, simp_i))

        out.append({
            "processed_id": pid,
            "reason": _clean_short_label(reason) or "",
            "main_object": _clean_short_label(main_obj) or "",
            "simplicity": simp_i,
        })

        # EXACTLY 1 candidate
        break

    return out


def pick_two_processed_candidates_transformers(
    *,
    model,
    processor,
    device: torch.device,
    processed_items: List[Dict[str, Any]],
    processed_to_unique: Dict[str, str],
    base_context: str,
    batch_size: int = 8,
    longest_side: int = 600,
) -> Dict[str, Any]:
    """
    Stage2 returns/keeps EXACTLY 1 candidate (as your prompt/schema expects).
    Prompts are untouched; we enforce 1 in parsing + fallback + packaging.
    """
    # Stage 1 (vision)
    stage1 = probe_processed_images_stage1(
        model=model,
        processor=processor,
        device=device,
        items=processed_items,
        batch_size=batch_size,
        longest_side=longest_side,
    )

    # Stage 2 (text-only decision)  <-- PROMPT TEXT IS UNCHANGED (build_stage2_pick_prompt)
    SYSTEM_MSG = (
        "You are a strict JSON decision engine.\n"
        "Pick exactly 2 candidates.\n"
        "Bias heavily toward simple single-object visuals.\n"
        "Use score to break ties.\n"
        "Output JSON only.\n"
    )
    prompt_text = build_stage2_pick_prompt(base_context, stage1, processed_to_unique)

    conv = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]
    prompt_str = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    gen_kwargs = dict(
        max_new_tokens=260,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

    chosen: List[Dict[str, Any]] = []
    try:
        inputs = processor(
            text=[prompt_str],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device=device, non_blocking=True)

        with torch.inference_mode():
            out_ids = model.generate(**inputs, **gen_kwargs)

        if "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
            prompt_len = int(inputs["input_ids"].shape[1])
            gen_only_ids = out_ids[:, prompt_len:]
        else:
            gen_only_ids = out_ids

        text_out = processor.batch_decode(gen_only_ids, skip_special_tokens=True)[0]
        chosen = _parse_stage2_candidates(text_out)

    except Exception as e:
        print(f"[warn] stage2 selection failed: {e}")
        chosen = []

    # Validate + keep ONLY 1
    valid_ids = set(stage1.keys())
    chosen = [c for c in chosen if c.get("processed_id") in valid_ids][:1]

    # Fallback if model didn't return 1 valid candidate
    if len(chosen) < 1:
        ranked = []
        for pid, info in stage1.items():
            simp = info.get("simplicity", 5)
            try:
                simp_i = int(simp)
            except Exception:
                simp_i = 5
            simp_i = max(1, min(5, simp_i))

            objs = info.get("objects") or []
            nobj = len(objs) if isinstance(objs, list) else 999

            sc = info.get("score", None)
            try:
                sc_f = float(sc) if isinstance(sc, (int, float)) else -1.0
            except Exception:
                sc_f = -1.0

            # lower simplicity better, higher score better, fewer objects better
            ranked.append((simp_i, -sc_f, nobj, pid))

        ranked.sort()
        fallback_pid = ranked[0][3] if ranked else None

        chosen = []
        if fallback_pid:
            chosen.append({
                "processed_id": fallback_pid,
                "reason": "fallback rank",
                "main_object": stage1.get(fallback_pid, {}).get("dominant", "") or "",
                "simplicity": int(stage1.get(fallback_pid, {}).get("simplicity", 5)),
            })

    # Attach unique path + score (authoritative) and keep ONLY 1
    final_candidates: List[Dict[str, Any]] = []
    if chosen:
        c = chosen[0]
        pid = c.get("processed_id", "")
        info = stage1.get(pid, {}) if isinstance(stage1.get(pid, {}), dict) else {}

        sc = info.get("score", None)
        sc_out = None
        try:
            if isinstance(sc, (int, float)):
                sc_out = float(sc)
        except Exception:
            sc_out = None

        final_candidates.append({
            "processed_id": pid,
            "unique_path": str(processed_to_unique.get(pid, "")),
            "reason": str(c.get("reason", "")),
            "main_object": str(c.get("main_object", "")),
            "simplicity": int(c.get("simplicity", 5)),
            "score": sc_out,
        })

    return {
        "base_context": (base_context or "").strip(),
        "stage1": stage1,
        "candidates": final_candidates,  # list with exactly 1 dict (or empty if stage1 empty)
    }




# ============================
# MAIN - kept for testing
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

        suggestions_all, base_context = load_candidate_labels(idx, json_index)

        # NEW: first Qwen run to refine candidate labels into component/part labels
        refined_labels = refine_candidate_labels_with_qwen(
            model=model,
            processor=processor,
            device=device,
            base_context=base_context,
            raw_candidates=suggestions_all,
        )

        suggestions = refined_labels[: int(SUGGESTION_LIMIT)]



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

        if SKIP_EXISTING_LABELS and out_json_path.exists():
            print(f"[skip] processed_{idx}: labels already exist: {out_json_path}")
            continue


        results: Dict[str, Any] = {
            "image_index": idx,
            "full_image_source": str(full_img_path),
            "clusters_json": str(cmap_path),
            "candidate_labels_raw": suggestions_all[: int(SUGGESTION_LIMIT)],
            "candidate_labels_refined": suggestions,
            "base_context": base_context,
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
                    base_context=base_context
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