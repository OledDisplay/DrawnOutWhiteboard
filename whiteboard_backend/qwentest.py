#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
import gc
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

# Planner prompt/memory guardrails (env-tunable).
def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw not in ("0", "false", "no", "off", "")


PLAN_PROMPT_CHAR_CAP = max(512, int(os.getenv("QWEN_PLAN_PROMPT_CHAR_CAP", "12000") or 12000))
PLAN_BATCH_CHAR_BUDGET = max(1024, int(os.getenv("QWEN_PLAN_BATCH_CHAR_BUDGET", "18000") or 18000))
PLAN_DO_SAMPLE = _env_flag("QWEN_PLAN_DO_SAMPLE", False)
PLAN_USE_CACHE = _env_flag("QWEN_PLAN_USE_CACHE", True)

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


# ----------------------------
# FAST INFERENCE CONFIG
# ----------------------------
USE_SDPA = True
USE_TORCH_COMPILE = True          # keep on only if your hot path is shape-stable
TRY_STATIC_CACHE = False          # transformers StaticLayer cache has been unstable here
COMPILE_MODE = "reduce-overhead"
PAD_TO_MULTIPLE_OF = 8            # reduces shape churn a bit without wasting too much padding


def _qwen_attn_impl() -> Optional[str]:
    if FORCE_CPU or not torch.cuda.is_available() or not USE_SDPA:
        return None
    return "sdpa"


def _processor_batch(
    processor,
    *,
    text=None,
    images=None,
):
    kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
    }

    if text is not None:
        kwargs["text"] = text
        if PAD_TO_MULTIPLE_OF:
            kwargs["pad_to_multiple_of"] = int(PAD_TO_MULTIPLE_OF)

    if images is not None:
        kwargs["images"] = images

    try:
        return processor(**kwargs)
    except TypeError:
        kwargs.pop("pad_to_multiple_of", None)
        return processor(**kwargs)


def _move_inputs_to_device(inputs, device: torch.device):
    img_dtype = _img_tensor_dtype(device)

    for k, v in list(inputs.items()):
        if not torch.is_tensor(v):
            continue
        if k == "pixel_values":
            inputs[k] = v.to(device=device, dtype=img_dtype, non_blocking=True)
        else:
            inputs[k] = v.to(device=device, non_blocking=True)

    return inputs


def _maybe_enable_qwen_fast_inference(
    model,
    device: torch.device,
    *,
    allow_compile: bool = True,
):
    flags = {
        "attn_implementation": None,
        "static_cache": False,
        "compiled": False,
        "compile_fullgraph": None,
    }

    if device.type != "cuda":
        setattr(model, "_qwen_fast_flags", flags)
        return model

    try:
        if USE_SDPA and hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("sdpa")
            flags["attn_implementation"] = "sdpa"
    except Exception as e:
        print(f"[warn] set_attn_implementation(sdpa) failed: {e}")

    if TRY_STATIC_CACHE and allow_compile:
        try:
            model.generation_config.cache_implementation = "static"
            flags["static_cache"] = True
        except Exception as e:
            print(f"[warn] static cache unavailable: {e}")

    if USE_TORCH_COMPILE and allow_compile and flags["static_cache"]:
        try:
            model.forward = torch.compile(
                model.forward,
                mode=COMPILE_MODE,
                fullgraph=True,
            )
            flags["compiled"] = True
            flags["compile_fullgraph"] = True
        except Exception:
            try:
                model.forward = torch.compile(
                    model.forward,
                    mode=COMPILE_MODE,
                    fullgraph=False,
                )
                flags["compiled"] = True
                flags["compile_fullgraph"] = False
            except Exception as e:
                print(f"[warn] torch.compile disabled: {e}")
                try:
                    model.generation_config.cache_implementation = None
                except Exception:
                    pass
                flags["static_cache"] = False

    setattr(model, "_qwen_fast_flags", flags)
    return model


def _is_static_cache_batch_error(err: BaseException) -> bool:
    msg = f"{type(err).__name__}: {err}".lower()
    return ("staticlayer" in msg) and ("max_batch_size" in msg)


def _downgrade_cache_mode_for_runtime(model) -> None:
    changed: List[str] = []

    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        try:
            if getattr(gen_cfg, "cache_implementation", None) != "dynamic":
                gen_cfg.cache_implementation = "dynamic"
                changed.append("cache_implementation=dynamic")
        except Exception:
            try:
                gen_cfg.cache_implementation = None
                changed.append("cache_implementation=None")
            except Exception:
                pass

    try:
        if hasattr(model, "config"):
            model.config.use_cache = False
            changed.append("model.config.use_cache=False")
    except Exception:
        pass

    for attr in ("_cache", "_past_key_values", "past_key_values"):
        try:
            if hasattr(model, attr):
                setattr(model, attr, None)
        except Exception:
            pass

    flags = getattr(model, "_qwen_fast_flags", None)
    if isinstance(flags, dict):
        flags["static_cache"] = False
        flags["compiled"] = False
        flags["compile_fullgraph"] = None

    if changed:
        print(f"[warn] qwen cache fallback applied ({', '.join(changed)})")


def _safe_generate(
    model,
    *,
    inputs: Dict[str, Any],
    gen_kwargs: Dict[str, Any],
):
    try:
        return model.generate(**inputs, **gen_kwargs)
    except Exception as e:
        if not _is_static_cache_batch_error(e):
            raise
        print(f"[warn] qwen static cache mismatch, retrying without cache: {e}")
        _downgrade_cache_mode_for_runtime(model)
        retry_kwargs = dict(gen_kwargs or {})
        retry_kwargs["use_cache"] = False
        return model.generate(**inputs, **retry_kwargs)


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
        inputs = _processor_batch(
            processor,
            text=[prompt_str],
        )
        inputs = _move_inputs_to_device(inputs, device)

        with torch.inference_mode():
            out_ids = _safe_generate(model, inputs=inputs, gen_kwargs=gen_kwargs_refine)

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
        "Based on LEFT's full semantic profile label it - what it is\n\n"
        "Return ONLY JSON with REQUIRED keys:\n"
        "{"
        "\"label\":string|null,"
        "\"full_visual_LEFT\": string,"
        "\"LEFT_SURROUNDINGS\": string,"
        "\"geometry_keywords\": strings"
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

def build_cluster_visual_prompt(
    colour_hint: Optional[str],
    base_context: str,
) -> str:
    colour_txt = colour_hint if colour_hint else "unknown"
    bc = (base_context or "").strip() or "unknown"

    return (
        "Input : Two images merged horizontally\n"
        f"- LEFT: Cropped box with with coloured {colour_txt} main object and grayscaled surroundings\n"
        f"- RIGHT: A(n) {base_context}, LEFT crop from there\n\n"
        f"Visually charecterize the features of LEFT's object and its placement in the {base_context} based on foreground.\n"
        "Look at RIGHT and LEFTS FOREGROUND and link the crop to WIDER surroundings (their shapes, colours) for RELATIVE context \n"
        "Based on LEFT's full semantic profile label it - what it is\n\n"
        "Return ONLY JSON with REQUIRED keys:\n"
        "{"
        "\"label\":string|null,"
        "\"full_visual_LEFT\": string,"
        "\"LEFT_SURROUNDINGS\": string,"
        "\"geometry_keywords\": strings"
        "}\n"
    )


def _parse_cluster_visual(text_out: str) -> Dict[str, Any]:
    obj = extract_json_object(text_out or "")
    if not isinstance(obj, dict):
        return {
            "full_visual_LEFT": "",
            "LEFT_SURROUNDINGS": "",
            "geometry_keywords": [],
        }

    fv = obj.get("full_visual_LEFT")
    ls = obj.get("LEFT_SURROUNDINGS")
    gk = obj.get("geometry_keywords")

    out = {
        "full_visual_LEFT": fv if isinstance(fv, str) else "",
        "LEFT_SURROUNDINGS": ls if isinstance(ls, str) else "",
        "geometry_keywords": [],
    }

    if isinstance(gk, list):
        kk = []
        for it in gk:
            if isinstance(it, str):
                s = _clean_short_label(it)
                if s:
                    kk.append(s)
        out["geometry_keywords"] = kk[:12]

    return out


def build_postfacto_label_match_prompt(
    base_context: str,
    refined_labels: List[str],
    cluster_visual_map: Dict[str, Dict[str, Any]],
) -> str:
    bc = (base_context or "").strip() or "unknown"

    # compact the payload
    labels = [str(x).strip() for x in (refined_labels or []) if str(x).strip()]
    labels = labels[:120]

    clusters_compact = {}
    for k, v in (cluster_visual_map or {}).items():
        if not isinstance(v, dict):
            continue
        clusters_compact[str(k)] = {
            "geometry_keywords": v.get("geometry_keywords", []) if isinstance(v.get("geometry_keywords"), list) else [],
            "full_visual_LEFT": str(v.get("full_visual_LEFT", "") or "")[:600],
            "LEFT_SURROUNDINGS": str(v.get("LEFT_SURROUNDINGS", "") or "")[:600],
        }

    return (
        f"BASE CONTEXT: {bc}\n\n"
        "You are given:\n"
        "1) A list of refined component labels (things that buuld up something - cell, face, ex).\n"
        "2) A map unlabeled objects, described ONLY by visual characteristics.\n\n"
        "Your job:\n"
        "- For EACH object, choose the single BEST label from the refined label list.\n"
        "- Match by how that label would LOOK visually (its known characteristics).\n"
        "- If none fit, output null.\n\n"
        "Output JSON ONLY with schema:\n"
        "{"
        "\"matches\":["
        "{\"object_key\":\"...\",\"label\":string|null,\"confidence\":0.0,\"reason\":\"...\"}"
        "]"
        "}\n\n"
        "Rules:\n"
        "- label must be exactly one of refined_labels or null\n"
        "- confidence is 0..1\n"
        "- reason is 2-10 words\n"
        "REFINED LABELS:\n"
        f"{json.dumps(labels, ensure_ascii=False)}\n\n"
        "OBJECT VISUAL MAP, INDEXED:\n"
        f"{json.dumps(clusters_compact, ensure_ascii=False)}\n"
    )


def _parse_postfacto_matches(text_out: str, refined_labels: List[str]) -> Dict[str, Dict[str, Any]]:
    obj = extract_json_object(text_out or "")
    if not isinstance(obj, dict):
        return {}

    matches = obj.get("matches")
    if not isinstance(matches, list):
        return {}

    allowed = {str(x) for x in (refined_labels or []) if isinstance(x, str) and x.strip()}
    out: Dict[str, Dict[str, Any]] = {}

    for it in matches:
        if not isinstance(it, dict):
            continue
        ck = it.get("object_key")
        if not isinstance(ck, str) or not ck.strip():
            continue
        ck = ck.strip()

        lab = it.get("label")
        if lab is not None and (not isinstance(lab, str) or lab not in allowed):
            lab = None

        conf = it.get("confidence", 0.0)
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        conf_f = max(0.0, min(1.0, conf_f))

        reason = it.get("reason")
        reason_s = str(reason or "").strip()

        out[ck] = {
            "label": lab,
            "confidence": conf_f,
            "reason": reason_s,
        }

    return out


def postfacto_match_labels_with_qwen(
    model,
    processor,
    device: torch.device,
    *,
    base_context: str,
    refined_labels: List[str],
    cluster_visual_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if not refined_labels or not cluster_visual_map:
        return {}

    SYSTEM_MSG = (
        "You are a strict visual to semantic matcher.\n"
        "You output only JSON.\n"
    )

    prompt_text = build_postfacto_label_match_prompt(base_context, refined_labels, cluster_visual_map)

    conv = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]
    prompt_str = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    gen_kwargs = dict(
        max_new_tokens=500,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

    try:
        inputs = _processor_batch(
            processor,
            text=[prompt_str],
        )
        inputs = _move_inputs_to_device(inputs, device)

        with torch.inference_mode():
            out_ids = _safe_generate(model, inputs=inputs, gen_kwargs=gen_kwargs)

        if "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
            prompt_len = int(inputs["input_ids"].shape[1])
            gen_only_ids = out_ids[:, prompt_len:]
        else:
            gen_only_ids = out_ids

        text_out = processor.batch_decode(gen_only_ids, skip_special_tokens=True)[0]
        return _parse_postfacto_matches(text_out, refined_labels)

    except Exception as e:
        print(f"[warn] postfacto_match_labels_with_qwen failed: {e}")
        return {}




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

    load_kwargs = {
        "low_cpu_mem_usage": True,
    }

    attn_impl = _qwen_attn_impl()
    if attn_impl is not None:
        load_kwargs["attn_implementation"] = attn_impl

    if have_cuda:
        if used_quant:
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto" if INT8_CPU_OFFLOAD else {"": GPU_INDEX}
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                **load_kwargs,
            ).eval()
        else:
            load_kwargs["torch_dtype"] = DTYPE
            load_kwargs["device_map"] = {"": GPU_INDEX}
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                **load_kwargs,
            ).eval()
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            **load_kwargs,
        ).eval().to(device)

    if tok is not None and hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id

    try:
        model.config.use_cache = True
    except Exception:
        pass

    if have_cuda:
        model = _maybe_enable_qwen_fast_inference(
            model,
            device,
            allow_compile=not used_quant,
        )

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

    inputs = _processor_batch(
        processor,
        text=[prompt_str],
        images=[img],
    )
    inputs = _move_inputs_to_device(inputs, device)

    fast_flags = getattr(model, "_qwen_fast_flags", {})
    warmup_iters = 2 if fast_flags.get("compiled") else 1

    with torch.inference_mode():
        warmup_kwargs = {
            "max_new_tokens": 8,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
        }
        for _ in range(warmup_iters):
            _ = _safe_generate(model, inputs=inputs, gen_kwargs=warmup_kwargs)

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
    *,
    warmup: bool = False,
) -> tuple[Any, Any, torch.device]:
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

    if warmup:
        warmup_qwen_once(model, processor, device)

    return model, processor, device



def move_qwen_to_cuda_if_available(
    model,
    *,
    gpu_index: int = GPU_INDEX,
) -> tuple[Any, torch.device]:
    if torch.cuda.is_available() and not FORCE_CPU:
        device = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(gpu_index)

        model = model.to(device)
        model.eval()

        try:
            model = model.to(dtype=DTYPE)
        except Exception:
            pass

        model = _maybe_enable_qwen_fast_inference(
            model,
            device,
            allow_compile=True,
        )
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

    results_by_idx: Dict[int, Dict[str, Any]] = {}

    for idx in sorted(clusters_state.keys()):

        folder = out_dir / f"processed_{idx}"
        folder.mkdir(parents=True, exist_ok=True)
        labels_path = folder / "labels.json"

        if save_outputs and SKIP_EXISTING_LABELS and labels_path.exists():
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
        refined_upstream = list(pack.get("candidate_labels_refined", []) or [])

        # NOTE: refined_upstream now used ONLY in post-facto stage2 matching
        refined_labels = refined_upstream or suggestions_all
        refined_labels = [str(x).strip() for x in refined_labels if str(x).strip()]

        full_img_rgb = _as_rgb_uint8(pack.get("full_img_rgb"))
        if full_img_rgb is None:
            print(f"[skip] idx={idx}: missing full_img_rgb")
            continue

        full_img0 = Image.fromarray(full_img_rgb, mode="RGB")
        right_panel_base, full_scale, full_rw, full_rh, full_pad_x, full_pad_y = _fit_into_square(full_img0, panel_edge)

        results: Dict[str, Any] = {
            "image_index": int(idx),
            "candidate_labels_raw": suggestions_all[: int(SUGGESTION_LIMIT)],
            "candidate_labels_refined": refined_labels[: int(SUGGESTION_LIMIT)],
            "base_context": base_context,
            "model_id": str(getattr(model, "name_or_path", MODEL_ID)),
            "proc_longest_edge": int(PROC_LONGEST_EDGE),
            "panel_edge": int(panel_edge),
            "batch_size": int(BATCH_SIZE),
            "clusters": {},
            "cluster_order": [],
            "postfacto_matches": {},
        }
        results_by_idx[idx] = results

        entries = pack.get("clusters") or []
        if not isinstance(entries, list) or not entries:
            print(f"[skip] idx={idx}: pack has no clusters list")
            continue

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

                prompt_text = build_cluster_visual_prompt(colour_hint, base_context)

                SYSTEM_MSG = (
                    "You are a strict visual transcription engine.\n"
                    "No guessing identity.\n"
                    "Return JSON only.\n"
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
                inputs = _processor_batch(
                    processor,
                    text=texts_batch,
                    images=images_batch,
                )
                inputs = _move_inputs_to_device(inputs, device)

                with torch.inference_mode():
                    out_ids = _safe_generate(model, inputs=inputs, gen_kwargs=gen_kwargs)

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

                parsed_visual = _parse_cluster_visual(text_out)

                _store(meta["cluster_key"], {
                    "entry": meta["entry"],
                    "colour_hint": meta["colour_hint"],
                    "bbox_xyxy": meta["bbox_xyxy"],
                    "bbox_right_panel_xyxy": meta["bbox_right_panel_xyxy"],
                    "timing_s": float(f"{dt:.4f}"),
                    "model_raw": text_out,
                    "visual": parsed_visual,
                    "matched_label": None,
                    "match_confidence": None,
                    "match_reason": "",
                })

        # ---- Stage 2: post-facto label matching across all clusters (text-only)
        cluster_visual_map: Dict[str, Dict[str, Any]] = {}
        for ck in results.get("cluster_order", []):
            rec = results["clusters"].get(ck, {})
            vis = rec.get("visual")
            if isinstance(vis, dict):
                cluster_visual_map[ck] = vis

        matches = postfacto_match_labels_with_qwen(
            model=model,
            processor=processor,
            device=device,
            base_context=base_context,
            refined_labels=refined_labels,
            cluster_visual_map=cluster_visual_map,
        )

        results["postfacto_matches"] = matches

        for ck, m in matches.items():
            if ck in results["clusters"] and isinstance(m, dict):
                results["clusters"][ck]["matched_label"] = m.get("label")
                results["clusters"][ck]["match_confidence"] = m.get("confidence")
                results["clusters"][ck]["match_reason"] = m.get("reason", "")

        if save_outputs:
            labels_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

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
            inputs = _processor_batch(
                processor,
                text=texts_batch,
                images=images_batch,
            )
            inputs = _move_inputs_to_device(inputs, device)

            with torch.inference_mode():
                out_ids = _safe_generate(model, inputs=inputs, gen_kwargs=gen_kwargs)

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
        inputs = _processor_batch(
            processor,
            text=[prompt_str],
        )
        inputs = _move_inputs_to_device(inputs, device)

        with torch.inference_mode():
            out_ids = _safe_generate(model, inputs=inputs, gen_kwargs=gen_kwargs)

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


import json
import re
from typing import Any, Dict, List, Optional

import torch


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Pull the first {...} JSON object out of model text, tolerant of pre/post junk.
    """
    if not isinstance(text, str):
        return None

    # fast path
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    # find first balanced {...}
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1]
                try:
                    return json.loads(chunk)
                except Exception:
                    return None
    return None


def _clamp_i(v: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(v), int(hi)))


def _bbox_overlap_xywh(a: Dict[str, int], b: Dict[str, int]) -> bool:
    ax0 = int(a.get("x", 0))
    ay0 = int(a.get("y", 0))
    ax1 = ax0 + int(a.get("w", 0))
    ay1 = ay0 + int(a.get("h", 0))
    bx0 = int(b.get("x", 0))
    by0 = int(b.get("y", 0))
    bx1 = bx0 + int(b.get("w", 0))
    by1 = by0 + int(b.get("h", 0))
    if ax1 <= bx0 or bx1 <= ax0:
        return False
    if ay1 <= by0 or by1 <= ay0:
        return False
    return True


def _normalize_print_bbox(
    pb: Optional[Dict[str, Any]],
    *,
    default_w: int,
    default_h: int,
    board_w: int,
    board_h: int,
) -> Dict[str, int]:
    if not isinstance(pb, dict):
        pb = {}
    w = int(pb.get("w", default_w) or default_w)
    h = int(pb.get("h", default_h) or default_h)
    # keep space boxes large, but cap around quadrant-ish scale so multiple entries can coexist
    max_w = max(80, int(board_w) // 2)
    max_h = max(60, int(board_h) // 2)
    w = max(80, min(w, max_w))
    h = max(60, min(h, max_h))

    x = int(pb.get("x", 20) or 20)
    y = int(pb.get("y", 20) or 20)
    max_x = max(0, int(board_w) - w)
    max_y = max(0, int(board_h) - h)
    x = _clamp_i(x, 0, max_x)
    y = _clamp_i(y, 0, max_y)
    return {"x": x, "y": y, "w": int(w), "h": int(h)}


def _find_non_overlapping_box(
    desired: Dict[str, int],
    *,
    placed: List[Dict[str, int]],
    board_w: int,
    board_h: int,
    step: int = 40,
) -> Dict[str, int]:
    cand = _normalize_print_bbox(
        desired,
        default_w=int(desired.get("w", 240) or 240),
        default_h=int(desired.get("h", 180) or 180),
        board_w=board_w,
        board_h=board_h,
    )
    if all(not _bbox_overlap_xywh(cand, p) for p in placed):
        return cand

    max_x = max(0, int(board_w) - int(cand["w"]))
    max_y = max(0, int(board_h) - int(cand["h"]))

    def _scan(cur: Dict[str, int]) -> Optional[Dict[str, int]]:
        mx = max(0, int(board_w) - int(cur["w"]))
        my = max(0, int(board_h) - int(cur["h"]))
        yy = 0
        while yy <= my:
            xx = 0
            while xx <= mx:
                test = {"x": int(xx), "y": int(yy), "w": int(cur["w"]), "h": int(cur["h"])}
                if all(not _bbox_overlap_xywh(test, p) for p in placed):
                    return test
                xx += int(step)
            yy += int(step)
        return None

    found = _scan(cand)
    if found is not None:
        return found

    # If no space with current size, shrink progressively to force non-overlap placement.
    cur = dict(cand)
    for _ in range(8):
        cur["w"] = max(80, int(round(cur["w"] * 0.88)))
        cur["h"] = max(60, int(round(cur["h"] * 0.88)))
        found = _scan(cur)
        if found is not None:
            return found

    return cand


def _render_chat_text(processor, prompt: str) -> str:
    try:
        tok = getattr(processor, "tokenizer", None)
        if tok is not None and hasattr(tok, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass
    return prompt


def _clip_str(v: Any, max_chars: int) -> str:
    s = str(v or "").strip()
    if max_chars > 0 and len(s) > max_chars:
        return s[:max_chars].rstrip()
    return s


def _clip_str_list(values: Any, *, max_items: int, max_chars: int) -> List[str]:
    out: List[str] = []
    if not isinstance(values, list):
        return out
    for it in values:
        s = _clip_str(it, max_chars=max_chars)
        if not s:
            continue
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def _compact_space_chunk_for_prompt(chunk: Dict[str, Any], *, board_width: int, board_height: int) -> Dict[str, Any]:
    steps_src = chunk.get("steps") if isinstance(chunk, dict) else None
    entries_src = chunk.get("entries") if isinstance(chunk, dict) else None

    steps_out: List[Dict[str, Any]] = []
    for s in (steps_src or []):
        if not isinstance(s, dict):
            continue
        steps_out.append(
            {
                "key": _clip_str(s.get("key", ""), 24),
                "timeline_text": _clip_str(s.get("timeline_text", ""), 180),
                "start_word_index": int(s.get("start_word_index", 0) or 0),
                "end_word_index": int(s.get("end_word_index", 0) or 0),
            }
        )

    entries_out: List[Dict[str, Any]] = []
    for e in (entries_src or []):
        if not isinstance(e, dict):
            continue
        row: Dict[str, Any] = {
            "entry_index": int(e.get("entry_index", 0) or 0),
            "type": _clip_str(e.get("type", ""), 16),
            "name": _clip_str(e.get("name", ""), 80),
            "step_key": _clip_str(e.get("step_key", ""), 24),
            "range_start": int(e.get("range_start", 0) or 0),
            "range_end": int(e.get("range_end", 0) or 0),
            "duration_sec": float(e.get("duration_sec", 0.0) or 0.0),
            "diagram": int(e.get("diagram", 0) or 0),
            "text_tag": int(e.get("text_tag", 0) or 0),
            "write_text": _clip_str(e.get("write_text", ""), 180),
            "bbox_px": e.get("bbox_px") if isinstance(e.get("bbox_px"), dict) else None,
            "delete_all": bool(e.get("delete_all", False)),
            "speech_text_in_range": _clip_str(e.get("speech_text_in_range", ""), 220),
        }
        if row["type"] != "silence":
            row["content"] = _clip_str(e.get("content", ""), 160)
            row["objects_that_comprise_image"] = _clip_str_list(
                e.get("objects_that_comprise_image"),
                max_items=32,
                max_chars=64,
            )
        entries_out.append(row)

    return {
        "chapter_index": int(chunk.get("chapter_index", 0) or 0),
        "board": {"w": int(board_width), "h": int(board_height)},
        "steps": steps_out,
        "entries": entries_out,
    }


def _compact_visual_event_for_prompt(ev: Dict[str, Any]) -> Dict[str, Any]:
    ctx_out: List[Dict[str, Any]] = []
    for c in (ev.get("static_plan_context") or []):
        if not isinstance(c, dict):
            continue
        ctx_out.append(
            {
                "name": _clip_str(c.get("name", ""), 80),
                "type": _clip_str(c.get("type", ""), 16),
                "print_bbox": c.get("print_bbox") if isinstance(c.get("print_bbox"), dict) else None,
                "range_start": int(c.get("range_start", 0) or 0),
                "range_end": int(c.get("range_end", 0) or 0),
                "diagram": int(c.get("diagram", 0) or 0),
                "text_tag": int(c.get("text_tag", 0) or 0),
                "speech_text_in_range": _clip_str(c.get("speech_text_in_range", ""), 180),
            }
        )
        if len(ctx_out) >= 12:
            break

    return {
        "name": _clip_str(ev.get("name", ""), 80),
        "type": _clip_str(ev.get("type", ""), 16),
        "content": _clip_str(ev.get("content", ""), 180),
        "write_text": _clip_str(ev.get("write_text", ""), 180),
        "diagram": int(ev.get("diagram", 0) or 0),
        "text_tag": int(ev.get("text_tag", 0) or 0),
        "range_start": int(ev.get("range_start", 0) or 0),
        "range_end": int(ev.get("range_end", 0) or 0),
        "speech_text_in_range": _clip_str(ev.get("speech_text_in_range", ""), 260),
        "print_bbox": ev.get("print_bbox") if isinstance(ev.get("print_bbox"), dict) else None,
        "bbox_px": ev.get("bbox_px") if isinstance(ev.get("bbox_px"), dict) else None,
        "objects_that_comprise_image": _clip_str_list(
            ev.get("objects_that_comprise_image"),
            max_items=48,
            max_chars=64,
        ),
        "static_plan_context": ctx_out,
    }


def _compact_active_objects_for_prompt(ev: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    src = ev.get("active_objects")
    if not isinstance(src, list):
        return out
    for idx, row in enumerate(src):
        if not isinstance(row, dict):
            continue
        bb0 = row.get("bbox")
        bb = bb0 if isinstance(bb0, dict) else {}
        one = {
            "name": _clip_str(row.get("name", ""), 96),
            "kind": _clip_str(row.get("kind", ""), 20),
            "bbox": {
                "x": int(bb.get("x", row.get("x", 0)) or 0),
                "y": int(bb.get("y", row.get("y", 0)) or 0),
                "w": int(bb.get("w", row.get("w", 0)) or 0),
                "h": int(bb.get("h", row.get("h", 0)) or 0),
            },
            "source_type": _clip_str(row.get("source_type", ""), 16),
            "created_order": int(row.get("created_order", idx) or idx),
        }
        if one["name"]:
            out.append(one)
        if len(out) >= 96:
            break
    return out


def _compact_silence_event_for_prompt(ev: Dict[str, Any]) -> Dict[str, Any]:
    active_objs = _compact_active_objects_for_prompt(ev)
    active_names = [str(x.get("name", "")) for x in active_objs if str(x.get("name", ""))]
    return {
        "chapter_index": int(ev.get("chapter_index", 0) or 0),
        "chunk_index": int(ev.get("chunk_index", 0) or 0),
        "segment_index": int(ev.get("segment_index", 0) or 0),
        "name": _clip_str(ev.get("name", ""), 80),
        "start_word_index": int(ev.get("start_word_index", 0) or 0),
        "end_word_index": int(ev.get("end_word_index", 0) or 0),
        "duration_sec": float(ev.get("duration_sec", 0.0) or 0.0),
        "active_names": _clip_str_list(active_names, max_items=128, max_chars=96),
        "active_objects": active_objs,
    }


def _generate_json_objects_from_prompts(
    *,
    model,
    processor,
    device,
    prompts: List[str],
    temperature: float,
    max_new_tokens: int,
    batch_size: int = 8,
    debug_label: str = "qwen_json",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not prompts:
        return [], []

    capped_prompts: List[str] = []
    for p in prompts:
        one = str(p or "")
        if PLAN_PROMPT_CHAR_CAP > 0 and len(one) > PLAN_PROMPT_CHAR_CAP:
            one = one[:PLAN_PROMPT_CHAR_CAP].rstrip() + "\n\n[TRUNCATED_FOR_VRAM]"
        capped_prompts.append(one)

    chat_texts = [_render_chat_text(processor, p) for p in capped_prompts]
    bs = max(1, int(batch_size or 1))
    all_raws: List[str] = []
    empty_cache_each_batch = str(os.getenv("QWEN_PLAN_EMPTY_CACHE_EACH_BATCH", "1")).strip().lower() not in ("0", "false", "no")
    batch_char_budget = max(1024, int(os.getenv("QWEN_PLAN_BATCH_CHAR_BUDGET", str(PLAN_BATCH_CHAR_BUDGET)) or PLAN_BATCH_CHAR_BUDGET))

    def _cleanup_cuda_cache() -> None:
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

    def _run_generate(texts_batch: List[str]) -> List[str]:
        inputs = None
        out_ids = None
        gen_only_ids = None
        try:
            with torch.no_grad():
                inputs = _processor_batch(processor, text=texts_batch)
                inputs = _move_inputs_to_device(inputs, device)

                gen_kwargs = {
                    "max_new_tokens": int(max_new_tokens),
                    "do_sample": bool(PLAN_DO_SAMPLE),
                    "temperature": float(temperature),
                    "top_p": 0.9,
                    "num_beams": 1,
                    "use_cache": bool(PLAN_USE_CACHE),
                }
                out_ids = _safe_generate(model, inputs=inputs, gen_kwargs=gen_kwargs)

                if "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
                    prompt_len = int(inputs["input_ids"].shape[1])
                    gen_only_ids = out_ids[:, prompt_len:]
                else:
                    gen_only_ids = out_ids

                decoded = processor.batch_decode(gen_only_ids, skip_special_tokens=True)
                return decoded
        finally:
            try:
                del gen_only_ids
            except Exception:
                pass
            try:
                del out_ids
            except Exception:
                pass
            try:
                del inputs
            except Exception:
                pass

    def _run_with_backoff(texts_batch: List[str], *, global_start: int) -> List[str]:
        if not texts_batch:
            return []
        try:
            return _run_generate(texts_batch)
        except RuntimeError as e:
            msg = str(e).lower()
            is_oom = ("out of memory" in msg) or ("cuda error: out of memory" in msg)
            if is_oom and len(texts_batch) > 1:
                print(
                    f"[qwen][DBG] {debug_label} split_after_oom start={global_start} "
                    f"size={len(texts_batch)} -> {len(texts_batch) // 2}+{len(texts_batch) - (len(texts_batch) // 2)}"
                )
                _cleanup_cuda_cache()
                mid = max(1, len(texts_batch) // 2)
                left = _run_with_backoff(texts_batch[:mid], global_start=global_start)
                right = _run_with_backoff(texts_batch[mid:], global_start=global_start + mid)
                return left + right

            print(f"[qwen][DBG] {debug_label} single_fail idx={global_start} err={type(e).__name__}: {e}")
            _cleanup_cuda_cache()
            return [""] * len(texts_batch)
        except Exception as e:
            print(f"[qwen][DBG] {debug_label} batch_fail start={global_start} size={len(texts_batch)} err={type(e).__name__}: {e}")
            _cleanup_cuda_cache()
            return [""] * len(texts_batch)

    start = 0
    while start < len(chat_texts):
        cur_bs = min(bs, len(chat_texts) - start)
        if batch_char_budget > 0:
            while cur_bs > 1:
                cand = chat_texts[start:start + cur_bs]
                if sum(len(x) for x in cand) <= batch_char_budget:
                    break
                cur_bs = max(1, cur_bs // 2)

        batch = chat_texts[start:start + cur_bs]
        if not batch:
            continue

        t0 = time.perf_counter()
        batch_chars = sum(len(x) for x in batch)
        max_chars = max((len(x) for x in batch), default=0)
        print(
            f"[qwen][DBG] {debug_label} batch_start start={start} size={len(batch)} "
            f"chars_total={batch_chars} chars_max={max_chars} vram={_vram_str()}"
        )
        try:
            batch_raws = _run_with_backoff(batch, global_start=start)
        finally:
            print(
                f"[qwen][DBG] {debug_label} batch_end start={start} size={len(batch)} "
                f"elapsed_ms={round((time.perf_counter() - t0) * 1000.0, 2)} vram={_vram_str()}"
            )
            if empty_cache_each_batch:
                _cleanup_cuda_cache()

        if len(batch_raws) < len(batch):
            batch_raws = list(batch_raws) + [""] * (len(batch) - len(batch_raws))
        all_raws.extend(batch_raws[:len(batch)])
        start += len(batch)

    objs: List[Dict[str, Any]] = []
    for raw in all_raws:
        parsed = _extract_first_json_object(raw)
        if not isinstance(parsed, dict):
            parsed = {}
        parsed["raw_text"] = raw
        objs.append(parsed)
    return objs, all_raws

def plan_whiteboard_actions_transformers(
    *,
    model,
    processor,
    device,
    event: Dict[str, Any],
    whiteboard_state: Dict[str, Any],
    base_actions: List[str],
    diagram_actions: List[str],
    temperature: float = 0.2,
    max_new_tokens: int = 700,
) -> Dict[str, Any]:
    """
    Qwen simple call:
      - event: image or silence packet (includes fine speech text + bbox + objects for diagrams)
      - whiteboard_state: current objects and bboxes on the board
      - base_actions: list of allowed base actions
      - diagram_actions: extra actions allowed only if diagram=1 (operate on objects_that_comprise_image)

    Returns dict:
      {
        "actions": [ ... ],   # list of dicts OR strings
        "raw_text": "...",
        "notes": "...",
      }
    """

    # Build instruction prompt (NO loops/state management here)
    kind = str(event.get("kind", "") or "")
    is_diagram = int(event.get("diagram", 0) or 0) == 1

    # NEW: text tag (0/1). When 1, this "image event" represents board text.
    text_tag = int(event.get("text", 0) or 0)
    write_text = str(event.get("write_text", "") or "")

    # Strong constraints: produce JSON only, avoid code fences.
    system_rules = (
        "You are planning actions with images to build a synced animation under some narration.\n"
        "You have to fill as much different actions for a rich result, using your base actions.\n"
        "Images will come sometimes come with a DIAGRAM tag.\n"
        "For these use the special diagram actions with each of their objects, pushing for text.\n\n"

        "You are provided with a white state with all present objects.\n"
        "Draw in empty spaces not already taken up by an object and manage object layout based on their relations.\n"
        "If there's no space either shift the board to make space (center shifts, you can draw at further coords)"
        "or delete an irrelevant object.\n"
        "Try and find an empty space first, if you erase plan your actions - > first erase then draw\n"

        "After you finish with an image dont erase it by default if it wasnt very \"temporary\""
        "During SILENCE: prefer erasing many objects and/or shifting the board to reset space.\n"

        "Try to keep something happening most of the time: output many actions when appropriate.\n"

        # --- ONLY ADDITION: TEXT TAG RULES ---
        "TEXT TAG RULES:\n"
        "- If CURRENT EVENT includes text_tag=1, this is NOT an image draw request.\n"
        "- For text_tag=1 you should primarily output a 'write' action to place the provided write_text.\n"
        "- Avoid complex layout/multi-step work for text_tag=1; only erase/shift if needed to make space.\n"
        "- Assume one letter width = 15 px (times scale). If you must clear space, erase first, then write.\n"
        "- Do NOT output diagram-only actions for text_tag=1.\n\n"
        # ------------------------------------

        "You MUST output ONLY a single JSON object, no markdown, no code fences.\n"
        "Do not invent new action types.\n"
        "All actions MUST be from the allowed action sets.\n"
        "Do not invent new action types.\n"
        "Whiteboard default scale is 4000 x 2000 px"
        "Coordinates are in whiteboard coords - center 0, 0.\n"
    )

    allowed = {
        "base_actions": base_actions,
        "diagram_actions": diagram_actions if is_diagram else [],
    }

    # Event payload
    event_desc = {
        "kind": kind,
        "duration_sec": float(event.get("duration_sec", 0.0) or 0.0),
        "chapter_index": int(event.get("chapter_index", 0) or 0),
    }

    if kind == "image":
        event_desc.update(
            {
                "image_name": event.get("image_name"),
                "image_text": event.get("image_text", ""),
                "bbox_px": event.get("bbox_px", {}),
                "diagram": 1 if is_diagram else 0,
                "objects_that_comprise_image": event.get("objects_that_comprise_image", []) if is_diagram else [],
                # NEW: propagate text tag & the text to write (if any)
                "text_tag": int(text_tag),
                "write_text": write_text if int(text_tag) == 1 else "",
            }
        )
    else:
        event_desc.update(
            {
                "context_before": event.get("context_before", ""),
                "context_after": event.get("context_after", ""),
            }
        )

    # Required output schema
    schema = {
        "actions": [
            # base:
            # {"type":"draw","target":"<image name>","x":123,"y":456}
            # {"type":"erase","target":"<image name or text object name>"}
            # {"type":"link","a":"<image name>","b":"<image name>"}
            # {"type":"write","text":"...","x":100,"y":200,"scale":1.0}
            # {"type":"move","target":"<image name>","x":400,"y":200}
            #
            # diagram-only (do not affect board geometry, still output them):
            # {"type":"highlight","image":"<image name>","object":"<object name>","time":2.0}
            # {"type":"zoom","image":"<image name>","object":"<object name>","time":3.0}
            # {"type":"refine_detail","image":"<image name>","object":"<object name>"}
        ],
        "notes": "short optional notes",
    }

    prompt = (
        system_rules
        + "\n\nALLOWED ACTIONS:\n"
        + json.dumps(allowed, ensure_ascii=False, indent=2)
        + "\n\nCURRENT WHITEBOARD STATE:\n"
        + json.dumps(whiteboard_state, ensure_ascii=False, indent=2)
        + "\n\nCURRENT EVENT:\n"
        + json.dumps(event_desc, ensure_ascii=False, indent=2)
        + "\n\nOUTPUT JSON SCHEMA (follow this shape):\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + "\n\nNow output the JSON object only."
    )

    # Try to use chat template if available
    chat_text = None
    try:
        tok = getattr(processor, "tokenizer", None)
        if tok is not None and hasattr(tok, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            chat_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        chat_text = None

    if chat_text is None:
        chat_text = prompt

    with torch.no_grad():
        inputs = _processor_batch(
            processor,
            text=[chat_text],
        )
        inputs = _move_inputs_to_device(inputs, device)

        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": True,
            "temperature": float(temperature),
            "top_p": 0.9,
        }

        out_ids = _safe_generate(model, inputs=inputs, gen_kwargs=gen_kwargs)
        if "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
            prompt_len = int(inputs["input_ids"].shape[1])
            gen_only_ids = out_ids[:, prompt_len:]
        else:
            gen_only_ids = out_ids
        try:
            raw = processor.batch_decode(gen_only_ids, skip_special_tokens=True)[0]
        except Exception:
            # some processors expose tokenizer.decode
            raw = processor.tokenizer.decode(gen_only_ids[0], skip_special_tokens=True)

    parsed = _extract_first_json_object(raw)
    if not isinstance(parsed, dict):
        return {
            "actions": [],
            "raw_text": raw,
            "notes": "failed_to_parse_json",
        }

    # Ensure keys exist
    if "actions" not in parsed or not isinstance(parsed["actions"], list):
        parsed["actions"] = []
    if "notes" not in parsed:
        parsed["notes"] = ""

    parsed["raw_text"] = raw
    return parsed


def _repair_space_plan_chunk(
    *,
    chunk: Dict[str, Any],
    planned_entries: Optional[List[Dict[str, Any]]],
    board_w: int,
    board_h: int,
) -> Dict[str, Any]:
    entries_in = chunk.get("entries") or []
    if not isinstance(entries_in, list):
        entries_in = []
    plan_rows = planned_entries if isinstance(planned_entries, list) else []
    plan_by_idx: Dict[int, Dict[str, Any]] = {}
    for row in plan_rows:
        if not isinstance(row, dict):
            continue
        try:
            k = int(row.get("entry_index", -1))
        except Exception:
            k = -1
        if k >= 0:
            plan_by_idx[k] = row

    merged: List[Dict[str, Any]] = []
    for e in entries_in:
        if not isinstance(e, dict):
            continue
        out = dict(e)
        idx = int(out.get("entry_index", 0) or 0)
        p = plan_by_idx.get(idx) or {}
        t = str(out.get("type", "") or "").lower()
        if t in ("image", "text"):
            bbox_px = out.get("bbox_px") or {}
            bw = int(bbox_px.get("w", 400) or 400)
            bh = int(bbox_px.get("h", 300) or 300)
            # "very big bbox" policy: image/text occupancy can approach quadrant scale.
            if t == "text":
                target_w = max(bw + 80, board_w // 8)
                target_h = max(bh + 60, board_h // 10)
            else:
                target_w = max(bw + 140, board_w // 5)
                target_h = max(bh + 120, board_h // 5)
            target_w = min(target_w, board_w)
            target_h = min(target_h, board_h)
            pb = p.get("print_bbox")
            out["print_bbox"] = _normalize_print_bbox(
                pb if isinstance(pb, dict) else None,
                default_w=int(target_w),
                default_h=int(target_h),
                board_w=board_w,
                board_h=board_h,
            )
        elif t == "silence":
            default_delete = bool(out.get("chunk_boundary_silence", False))
            out["delete_all"] = bool((p.get("delete_all") if isinstance(p, dict) else default_delete) if isinstance(p, dict) else default_delete)
        merged.append(out)

    # deterministic no-overlap repair for print boxes
    placed: List[Dict[str, int]] = []
    for row in sorted(
        [x for x in merged if str(x.get("type", "") or "") in ("image", "text")],
        key=lambda x: (int(x.get("range_start", 0) or 0), int(x.get("entry_index", 0) or 0)),
    ):
        pb = row.get("print_bbox") if isinstance(row.get("print_bbox"), dict) else None
        bbox_px = row.get("bbox_px") or {}
        dw = int(bbox_px.get("w", 400) or 400)
        dh = int(bbox_px.get("h", 300) or 300)
        desired = _normalize_print_bbox(
            pb,
            default_w=max(dw + 120, board_w // 6),
            default_h=max(dh + 100, board_h // 6),
            board_w=board_w,
            board_h=board_h,
        )
        fixed = _find_non_overlapping_box(desired, placed=placed, board_w=board_w, board_h=board_h, step=40)
        row["print_bbox"] = fixed
        placed.append(fixed)

    merged.sort(key=lambda x: (int(x.get("range_start", 0) or 0), 0 if str(x.get("type", "") or "") == "silence" else 1))
    for i, row in enumerate(merged):
        row["entry_index"] = int(i)

    return {
        "chapter_index": int(chunk.get("chapter_index", 0) or 0),
        "board": {"w": int(board_w), "h": int(board_h)},
        "step_order": list(chunk.get("step_order") or []),
        "steps": list(chunk.get("steps") or []),
        "entries": merged,
    }


def plan_space_timeline_batch_transformers(
    *,
    model,
    processor,
    device,
    chunk_maps: List[Dict[str, Any]],
    board_width: int = 4000,
    board_height: int = 4000,
    batch_size: int = 8,
    temperature: float = 0.15,
    max_new_tokens: int = 900,
) -> Dict[str, Any]:
    """
    Plans static print space per chunk:
      - assigns print_bbox for image/text entries
      - marks silence entries with delete_all true/false
    Always validates + repairs output to ensure valid non-overlapping bboxes.
    """
    chunks = [c for c in (chunk_maps or []) if isinstance(c, dict)]
    if not chunks:
        return {"chunks": []}

    prompts: List[str] = []
    for ch in chunks:
        payload = _compact_space_chunk_for_prompt(
            ch,
            board_width=int(board_width),
            board_height=int(board_height),
        )
        prompt = (
            "You are managing where images will be printed on a whiteboard.\n"
            "Goal: assign each image/text entry to a non-overlapping print_bbox and choose which silence rows should be cleanup points.\n"
            "You receive chronologically synced entries with range_start/range_end and speech_text_in_range.\n"
            "For each image/text entry you MUST output print_bbox.\n"
            "For each silence you MUST output delete_all true/false.\n"
            "Set delete_all=true only for pauses where board cleanup is useful before the next cluster of visuals.\n"
            "Keep nearby chronology conceptually close where possible.\n"
            "Use large boxes, avoid overlap, stay inside board bounds.\n"
            "Output JSON only.\n\n"
            "Input chunk:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n\nOutput schema:\n"
            + json.dumps(
                {
                    "entries": [
                        {"entry_index": 0, "type": "image", "print_bbox": {"x": 0, "y": 0, "w": 800, "h": 800}},
                        {"entry_index": 1, "type": "text", "print_bbox": {"x": 900, "y": 100, "w": 500, "h": 300}},
                        {"entry_index": 2, "type": "silence", "delete_all": False},
                    ],
                    "notes": "short optional note",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        prompts.append(prompt)

    parsed_objs, _ = _generate_json_objects_from_prompts(
        model=model,
        processor=processor,
        device=device,
        prompts=prompts,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        batch_size=int(batch_size),
        debug_label="plan_space",
    )

    out_chunks: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        obj = parsed_objs[i] if i < len(parsed_objs) else {}
        plan_entries = obj.get("entries") if isinstance(obj, dict) else None
        repaired = _repair_space_plan_chunk(
            chunk=ch,
            planned_entries=plan_entries if isinstance(plan_entries, list) else None,
            board_w=int(board_width),
            board_h=int(board_height),
        )
        out_chunks.append(repaired)

    return {"chunks": out_chunks}


def _default_visual_actions_for_event(ev: Dict[str, Any]) -> List[Dict[str, Any]]:
    s = int(ev.get("range_start", 0) or 0)
    e = int(ev.get("range_end", s) or s)
    if e < s:
        e = s
    span = max(0, e - s)
    pb = ev.get("print_bbox") or {"x": 0, "y": 0, "w": 400, "h": 300}
    x = int(pb.get("x", 0) or 0)
    y = int(pb.get("y", 0) or 0)

    acts: List[Dict[str, Any]] = []
    if int(ev.get("text_tag", 0) or 0) == 1:
        acts.append(
            {
                "type": "write_text",
                "target": str(ev.get("name", "") or ""),
                "text": str(ev.get("write_text", "") or ev.get("content", "")),
                "x": x,
                "y": y,
                "sync_local": {"start_word_offset": 0, "end_word_offset": span},
            }
        )
        return acts

    acts.append(
        {
            "type": "draw_image",
            "target": str(ev.get("name", "") or ""),
            "x": x,
            "y": y,
            "sync_local": {"start_word_offset": 0, "end_word_offset": min(span, 2)},
        }
    )
    if int(ev.get("diagram", 0) or 0) == 1:
        objs = ev.get("objects_that_comprise_image") or []
        first_obj = str(objs[0] if isinstance(objs, list) and objs else "").strip()
        acts.append(
            {
                "type": "highlight_cluster",
                "cluster_name": first_obj,
                "sync_local": {"start_word_offset": min(span, 1), "end_word_offset": min(span, max(1, span // 2))},
            }
        )
    return acts


_VISUAL_ACTION_ALIASES = {
    "draw_image": "draw_image",
    "draw": "draw_image",
    "write_text": "write_text",
    "write": "write_text",
    "annotate": "write_text",
    "move_inside_bbox": "move_inside_bbox",
    "move_inside_print_bbox": "move_inside_bbox",
    "move": "move_inside_bbox",
    "link_to_image": "link_to_image",
    "link_to_previous": "link_to_image",
    "link": "link_to_image",
    "delete_self": "delete_self",
    "highlight_cluster": "highlight_cluster",
    "unhighlight_cluster": "unhighlight_cluster",
    "zoom_cluster": "zoom_cluster",
    "unzoom_cluster": "unzoom_cluster",
    "write_label": "write_label",
}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return int(default)


def _normalize_sync_local(sync: Any, *, span: int) -> Dict[str, int]:
    if not isinstance(sync, dict):
        sync = {}
    s = _safe_int(sync.get("start_word_offset", sync.get("start", 0)), 0)
    e = _safe_int(sync.get("end_word_offset", sync.get("end", s)), s)
    s = _clamp_i(s, 0, max(0, span))
    e = _clamp_i(e, s, max(0, span))
    return {
        "start_word_offset": int(s),
        "end_word_offset": int(e),
    }


def _extract_xy(raw: Dict[str, Any], *, default_x: int, default_y: int, x_key: str = "x", y_key: str = "y") -> Tuple[int, int]:
    loc = raw.get("location")
    xv = raw.get(x_key, None)
    yv = raw.get(y_key, None)
    if isinstance(loc, dict):
        if xv is None:
            xv = loc.get("x")
        if yv is None:
            yv = loc.get("y")
    x = _safe_int(xv, default_x)
    y = _safe_int(yv, default_y)
    return x, y


def _clean_visual_action_for_event(ev: Dict[str, Any], a: Dict[str, Any], *, action_index: int) -> Optional[Dict[str, Any]]:
    raw_t = str(a.get("type", "") or "").strip().lower()
    if not raw_t:
        return None
    t = _VISUAL_ACTION_ALIASES.get(raw_t)
    if not t:
        return None

    is_diagram = int(ev.get("diagram", 0) or 0) == 1
    diagram_types = ("highlight_cluster", "unhighlight_cluster", "zoom_cluster", "unzoom_cluster", "write_label")
    if t in diagram_types and not is_diagram:
        return None

    s = int(ev.get("range_start", 0) or 0)
    e = int(ev.get("range_end", s) or s)
    if e < s:
        e = s
    span = max(0, e - s)

    pb = ev.get("print_bbox") if isinstance(ev.get("print_bbox"), dict) else {"x": 0, "y": 0, "w": 400, "h": 300}
    default_x = _safe_int(pb.get("x", 0), 0)
    default_y = _safe_int(pb.get("y", 0), 0)
    default_name = str(ev.get("name", "") or "").strip()

    out: Dict[str, Any] = {"type": t}

    if t == "draw_image":
        x, y = _extract_xy(a, default_x=default_x, default_y=default_y, x_key="x", y_key="y")
        out["target"] = str(a.get("target", "") or default_name).strip()
        out["x"] = int(x)
        out["y"] = int(y)
    elif t == "write_text":
        x, y = _extract_xy(a, default_x=default_x, default_y=default_y, x_key="x", y_key="y")
        text_fallback = str(ev.get("write_text", "") or ev.get("content", "")).strip()
        out["text"] = str(a.get("text", "") or text_fallback)
        out["x"] = int(x)
        out["y"] = int(y)
        out["scale"] = float(_safe_float(a.get("scale", 1.0), 1.0))
        tgt = str(a.get("target", "") or "").strip()
        if not tgt:
            tgt = default_name if int(ev.get("text_tag", 0) or 0) == 1 else f"{default_name}__text_{action_index + 1}"
        out["target"] = tgt
    elif t == "move_inside_bbox":
        nx, ny = _extract_xy(a, default_x=default_x, default_y=default_y, x_key="new_x", y_key="new_y")
        if "new_x" not in a and "new_y" not in a:
            nx, ny = _extract_xy(a, default_x=default_x, default_y=default_y, x_key="x", y_key="y")
        out["target"] = str(a.get("target", "") or default_name).strip()
        out["new_x"] = int(nx)
        out["new_y"] = int(ny)
    elif t == "link_to_image":
        other = (
            str(a.get("image_name", "") or a.get("target_image", "") or a.get("link_to_image", "") or a.get("to", "") or "")
            .strip()
        )
        if not other:
            return None
        out["target"] = default_name
        out["image_name"] = other
    elif t == "delete_self":
        pass
    elif t in diagram_types:
        objs = ev.get("objects_that_comprise_image") if isinstance(ev.get("objects_that_comprise_image"), list) else []
        first_obj = str(objs[0] if objs else "").strip()
        cluster_name = str(
            a.get("cluster_name", "")
            or a.get("cluster", "")
            or a.get("object", "")
            or a.get("name", "")
            or first_obj
        ).strip()
        if not cluster_name:
            return None
        out["cluster_name"] = cluster_name
        if t == "write_label":
            out["text"] = str(a.get("text", "") or cluster_name)

    out["sync_local"] = _normalize_sync_local(a.get("sync_local"), span=span)
    return out


def plan_visual_actions_batch_transformers(
    *,
    model,
    processor,
    device,
    events: List[Dict[str, Any]],
    batch_size: int = 8,
    temperature: float = 0.2,
    max_new_tokens: int = 700,
) -> Dict[str, Any]:
    """
    Plans actions for image/text/diagram events.
    Uses static timeline context (only items before current event in current batch).
    """
    rows = [e for e in (events or []) if isinstance(e, dict)]
    if not rows:
        return {"items": []}

    prompts: List[str] = []
    for ev in rows:
        payload = {
            "event": _compact_visual_event_for_prompt(ev),
            "allowed_actions": {
                "base": [
                    "draw_image",
                    "write_text",
                    "move_inside_bbox",
                    "link_to_image",
                    "delete_self",
                ],
                "diagram_extra": [
                    "highlight_cluster",
                    "unhighlight_cluster",
                    "zoom_cluster",
                    "unzoom_cluster",
                    "write_label",
                ],
            },
            "rules": {
                "print_bbox_is_fixed_context": True,
                "all_actions_inside_print_bbox": True,
                "generic_delete_not_allowed": True,
                "delete_self_only_for_temporary_non_diagram": True,
                "must_emit_sync_local_per_action": True,
                "sync_local_is_relative_to_event_range": True,
            },
        }
        prompt = (
            "You plan actions for one visual event on a whiteboard timeline.\n"
            "Use ONLY the allowed actions and exactly their field names.\n"
            "Every action MUST include sync_local {start_word_offset,end_word_offset} inside the event range.\n"
            "Coordinates must stay inside the event print_bbox.\n"
            "For diagram events, clusters in objects_that_comprise_image are interactable.\n"
            "Use diagram actions only when diagram=1.\n"
            "For text_tag=1 prioritize write_text.\n"
            "Base actions schema:\n"
            "- draw_image: {type,target,x,y,sync_local}\n"
            "- write_text: {type,target,text,x,y,scale,sync_local}\n"
            "- move_inside_bbox: {type,target,new_x,new_y,sync_local}\n"
            "- link_to_image: {type,target,image_name,sync_local}\n"
            "- delete_self: {type,sync_local}\n"
            "Diagram actions schema:\n"
            "- highlight_cluster/unhighlight_cluster/zoom_cluster/unzoom_cluster: {type,cluster_name,sync_local}\n"
            "- write_label: {type,cluster_name,text,sync_local}\n"
            "Return JSON only.\n"
            "Input includes static_plan_context with all prior entries from this batch.\n\n"
            "Input:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n\nOutput schema:\n"
            + json.dumps(
                {
                    "actions": [
                        {"type": "draw_image", "target": "img_name", "x": 100, "y": 180, "sync_local": {"start_word_offset": 0, "end_word_offset": 2}},
                        {"type": "write_text", "target": "img_name__text_1", "text": "label", "x": 120, "y": 260, "scale": 1.0, "sync_local": {"start_word_offset": 2, "end_word_offset": 4}},
                        {"type": "move_inside_bbox", "target": "img_name", "new_x": 180, "new_y": 220, "sync_local": {"start_word_offset": 4, "end_word_offset": 6}},
                        {"type": "link_to_image", "target": "img_name", "image_name": "previous_img", "sync_local": {"start_word_offset": 6, "end_word_offset": 8}},
                        {"type": "highlight_cluster", "cluster_name": "nucleus", "sync_local": {"start_word_offset": 7, "end_word_offset": 9}},
                        {"type": "write_label", "cluster_name": "nucleus", "text": "Nucleus", "sync_local": {"start_word_offset": 8, "end_word_offset": 10}},
                    ],
                    "notes": "short note",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        prompts.append(prompt)

    parsed_objs, raws = _generate_json_objects_from_prompts(
        model=model,
        processor=processor,
        device=device,
        prompts=prompts,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        batch_size=int(batch_size),
        debug_label="plan_visual",
    )

    out_items: List[Dict[str, Any]] = []
    for i, ev in enumerate(rows):
        obj = parsed_objs[i] if i < len(parsed_objs) else {}
        acts = obj.get("actions") if isinstance(obj, dict) else None
        if not isinstance(acts, list):
            acts = []

        cleaned: List[Dict[str, Any]] = []
        for ai, a in enumerate(acts):
            if not isinstance(a, dict):
                continue
            one = _clean_visual_action_for_event(ev, a, action_index=ai)
            if one is None:
                continue
            cleaned.append(one)

        if not cleaned:
            cleaned = _default_visual_actions_for_event(ev)

        out_items.append(
            {
                "actions": cleaned,
                "notes": str(obj.get("notes", "") or ""),
                "raw_text": raws[i] if i < len(raws) else "",
            }
        )

    return {"items": out_items}


def _ordered_active_names_for_silence(ev: Dict[str, Any]) -> List[str]:
    src = ev.get("active_objects")
    rows: List[Dict[str, Any]] = src if isinstance(src, list) else []
    sortable: List[Tuple[int, int, int, str]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "") or "").strip()
        if not name:
            continue
        bb = row.get("bbox") if isinstance(row.get("bbox"), dict) else {}
        y = _safe_int(bb.get("y", row.get("y", 10_000_000)), 10_000_000)
        x = _safe_int(bb.get("x", row.get("x", 10_000_000)), 10_000_000)
        created = _safe_int(row.get("created_order", i), i)
        sortable.append((y, x, created, name))
    if not sortable:
        names = ev.get("active_names")
        if isinstance(names, list):
            return [str(x).strip() for x in names if str(x).strip()]
        return []
    sortable.sort(key=lambda t: (t[0], t[1], t[2], t[3].lower()))
    out: List[str] = []
    seen = set()
    for _, _, _, name in sortable:
        k = name.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(name)
    return out


def _default_silence_delete_actions(ordered_names: List[str], *, span: int) -> List[Dict[str, Any]]:
    if not ordered_names:
        return []
    out: List[Dict[str, Any]] = []
    den = max(1, len(ordered_names))
    for i, nm in enumerate(ordered_names):
        start = 0 if span <= 0 else int(round((float(i) / float(den)) * float(span)))
        end = max(start, min(span, start))
        out.append(
            {
                "type": "delete_by_name",
                "target": str(nm),
                "sync_local": {"start_word_offset": int(start), "end_word_offset": int(end)},
            }
        )
    return out


def plan_silence_actions_batch_transformers(
    *,
    model,
    processor,
    device,
    events: List[Dict[str, Any]],
    batch_size: int = 8,
    temperature: float = 0.1,
    max_new_tokens: int = 500,
) -> Dict[str, Any]:
    """
    Plans deletion-focused actions for silences marked for cleanup.
    """
    rows = [e for e in (events or []) if isinstance(e, dict)]
    if not rows:
        return {"items": []}

    prompts: List[str] = []
    for ev in rows:
        ordered_names = _ordered_active_names_for_silence(ev)
        payload = {
            "silence_event": _compact_silence_event_for_prompt(ev),
            "allowed_actions": [
                "delete_by_name",
            ],
            "rules": {
                "goal": "delete all active objects",
                "order": "top_to_bottom_then_left_to_right",
                "must_emit_sync_local_per_action": True,
            },
            "ordered_names_for_cleanup": ordered_names,
        }
        prompt = (
            "You plan silence cleanup actions.\n"
            "This stage only deletes active objects from the board.\n"
            "Use delete_by_name only.\n"
            "Delete every object in ordered_names_for_cleanup from first to last.\n"
            "If active_objects include bbox, follow top-to-bottom order.\n"
            "Return JSON only.\n"
            "Every action must include sync_local {start_word_offset,end_word_offset}.\n\n"
            "Input:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n\nOutput schema:\n"
            + json.dumps(
                {
                    "actions": [
                        {"type": "delete_by_name", "target": "name", "sync_local": {"start_word_offset": 0, "end_word_offset": 0}},
                    ],
                    "notes": "short note",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        prompts.append(prompt)

    parsed_objs, raws = _generate_json_objects_from_prompts(
        model=model,
        processor=processor,
        device=device,
        prompts=prompts,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        batch_size=int(batch_size),
        debug_label="plan_silence",
    )

    out_items: List[Dict[str, Any]] = []
    for i, ev in enumerate(rows):
        obj = parsed_objs[i] if i < len(parsed_objs) else {}
        acts = obj.get("actions") if isinstance(obj, dict) else None
        if not isinstance(acts, list):
            acts = []

        s = int(ev.get("start_word_index", 0) or 0)
        e = int(ev.get("end_word_index", s) or s)
        if e < s:
            e = s
        span = max(0, e - s)
        ordered_names = _ordered_active_names_for_silence(ev)

        cleaned: List[Dict[str, Any]] = []
        seen_targets = set()
        for a in acts:
            if not isinstance(a, dict):
                continue
            t = str(a.get("type", "") or "").strip().lower()
            if t != "delete_by_name":
                continue
            target = str(a.get("target", "") or "").strip()
            if not target:
                continue
            if ordered_names:
                allowed = {x.lower() for x in ordered_names}
                if target.lower() not in allowed:
                    continue
            one = dict(a)
            one["target"] = target
            one["sync_local"] = _normalize_sync_local(one.get("sync_local"), span=span)
            cleaned.append(one)
            seen_targets.add(target.lower())

        missing = [nm for nm in ordered_names if nm.lower() not in seen_targets]
        if missing:
            cleaned.extend(_default_silence_delete_actions(missing, span=span))
        if not cleaned:
            cleaned = _default_silence_delete_actions(ordered_names, span=span)

        out_items.append(
            {
                "actions": cleaned,
                "notes": str(obj.get("notes", "") or ""),
                "raw_text": raws[i] if i < len(raws) else "",
            }
        )

    return {"items": out_items}





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
                inputs = _processor_batch(
                    processor,
                    text=texts_batch,
                    images=images_batch,
                )
                inputs = _move_inputs_to_device(inputs, device)

                with torch.inference_mode():
                    out_ids = _safe_generate(model, inputs=inputs, gen_kwargs=gen_kwargs)

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
