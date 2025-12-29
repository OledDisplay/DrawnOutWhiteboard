#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


# ============================
# PATHS (NO ARGS)
# ============================
BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "ProccessedImages"
CLUSTER_RENDER_DIR = BASE_DIR / "ClusterRenders"
CLUSTER_MAP_DIR = BASE_DIR / "ClusterMaps"

OUT_DIR = BASE_DIR / "InternLabels"   # <--- new output root


# ============================
# MODEL CONFIG
# ============================
MODEL_ID = "OpenGVLab/InternVL2_5-2B"

# Tiles (speed vs detail)
MAX_TILES_MASK = 4
MAX_TILES_BBOX = 4
MAX_TILES_FULL = 12

# Generation
MAX_NEW_TOKENS = 256
DO_SAMPLE = False

# If your GPU doesn't support bf16, set to torch.float16
DTYPE = torch.bfloat16

# If flash-attn isn't installed / errors, set False.
USE_FLASH_ATTN = True

# Confidence cutoff (you can tune). Final decision still comes from the model.
MIN_ACCEPT_CONF = 0.65


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
    s = re.sub(r"^[^\w]+|[^\w]+$", "", s)  # trim punctuation edges
    if not s:
        return None
    # kill single-letter garbage hard
    if len(s) == 1:
        return None
    low = s.lower()
    if low in STOPWORDS:
        return None
    # kill purely numeric
    if re.fullmatch(r"\d+", s):
        return None
    return s

def load_candidate_labels(idx: int) -> List[str]:
    # prefer processed_<n>.json
    p1 = PROCESSED_DIR / f"processed_{idx}.json"
    p2 = PROCESSED_DIR / f"proccessed_{idx}.json"
    path = p1 if p1.exists() else p2 if p2.exists() else None
    if path is None:
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
# CLUSTER DISCOVERY
# ============================
_FULL_IMG_RE = re.compile(r"^(?:proccessed|processed)_(\d+)\.png$", re.IGNORECASE)

def find_full_images() -> List[Tuple[int, Path]]:
    imgs: List[Tuple[int, Path]] = []
    for p in PROCESSED_DIR.glob("*.png"):
        m = _FULL_IMG_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        imgs.append((idx, p))
    imgs.sort(key=lambda x: x[0])
    return imgs

def find_cluster_folders_for_idx(idx: int) -> List[Path]:
    """
    Any folder in ClusterMaps that ends with _<idx> and contains clusters.json.
    Example: black_12, cyan_12, whatever_12.
    """
    out: List[Path] = []
    suffix = f"_{idx}"
    for d in CLUSTER_MAP_DIR.iterdir():
        if not d.is_dir():
            continue
        if not d.name.lower().endswith(suffix):
            continue
        cj = d / "clusters.json"
        if cj.exists():
            out.append(d)
    out.sort(key=lambda p: p.name.lower())
    return out

def parse_colour_hint_from_filename(name: str) -> Optional[str]:
    n = name.lower()
    for c in ["white","yellow","orange","cyan","green","magenta","red","blue","purple","gray","black"]:
        if c in n:
            return c
    return None


# ============================
# INTERNVL IMAGE PREPROCESS (official tiling approach)
# ============================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / max(1, orig_height)

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images

def load_image_tiles(image_path: Path, input_size=448, max_num=12) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size)
    imgs = dynamic_preprocess(image, image_size=input_size, max_num=max_num, use_thumbnail=True)
    pixel_values = torch.stack([transform(im) for im in imgs])
    return pixel_values


# ============================
# PROMPT + PARSE
# ============================
def build_prompt(
    colour_hint: Optional[str],
    bbox_xyxy: List[int],
    suggestions: List[str],
) -> str:
    colour_txt = f"{colour_hint}" if colour_hint else "unknown"

    # Keep suggestions short to avoid blowing context
    sug = suggestions[:60]

    return (
        "You will be given 3 images.\n"
        "Image-1 is the MOST IMPORTANT: a masked crop. The colored pixels are the object; grayscale pixels are surrounding context and should be treated as background.\n"
        "Image-2 is a plain crop of the same bbox for local context.\n"
        "Image-3 is the full diagram for global context; the object appears inside bbox xyxy = "
        f"{bbox_xyxy} in the full image.\n\n"
        f"Extra hint: the crop file name contains a colour hint: '{colour_txt}'. Use it only as a weak hint.\n\n"
        "Suggested labels (strong suggestions, but you may create a better label if needed):\n"
        f"{sug}\n\n"
        "Task:\n"
        "1) Decide if the object is recognizable and labelable (ACCEPT) or if it is trash/too ambiguous (REJECT).\n"
        "2) If ACCEPT, output a short label (prefer one from suggestions if it matches; otherwise create a new one).\n"
        "3) Be HARSH: only accept if you are genuinely confident from the images + context.\n\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "decision": "accept" | "reject",\n'
        '  "label": string | null,\n'
        '  "from_suggestions": boolean,\n'
        '  "confidence": number,  // 0..1\n'
        '  "reason": string\n'
        "}\n"
    )

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    i = s.find("{")
    j = s.rfind("}")
    if i < 0 or j <= i:
        return None
    candidate = s[i:j+1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


# ============================
# MAIN LOOP
# ============================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] model...")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        use_flash_attn=USE_FLASH_ATTN,
        trust_remote_code=True,
    ).eval()

    if torch.cuda.is_available():
        model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)

    generation_config = dict(
        max_new_tokens=int(MAX_NEW_TOKENS),
        do_sample=bool(DO_SAMPLE),
    )

    full_images = find_full_images()
    if not full_images:
        print(f"[error] no processed_<n>.png found in {PROCESSED_DIR}")
        return

    for idx, full_img_path in full_images:
        # gather all cluster folders for this idx
        cluster_folders = find_cluster_folders_for_idx(idx)
        if not cluster_folders:
            print(f"[skip] processed_{idx}: no ClusterMaps/*_{idx}/clusters.json folders")
            continue

        suggestions = load_candidate_labels(idx)

        out_folder = OUT_DIR / f"processed_{idx}"
        out_folder.mkdir(parents=True, exist_ok=True)
        out_json_path = out_folder / "labels.json"

        results: Dict[str, Any] = {
            "image_index": idx,
            "full_image": full_img_path.name,
            "candidate_labels": suggestions,
            "clusters": [],
        }

        # Preload full-image tiles once (big speed win)
        try:
            pv_full = load_image_tiles(full_img_path, max_num=MAX_TILES_FULL)
            pv_full = pv_full.to(dtype=DTYPE)
            if torch.cuda.is_available():
                pv_full = pv_full.cuda()
        except Exception as e:
            print(f"[fail] processed_{idx}: cannot load full image tiles: {e}")
            continue

        for folder in cluster_folders:
            clusters_path = folder / "clusters.json"
            try:
                cmap = json.loads(clusters_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[skip] {folder.name}: bad clusters.json: {e}")
                continue

            # crops live in ClusterRenders/<same folder name>/
            renders_dir = CLUSTER_RENDER_DIR / folder.name
            if not renders_dir.exists():
                print(f"[skip] {folder.name}: missing renders dir {renders_dir}")
                continue

            entries = cmap.get("clusters") or []
            if not isinstance(entries, list) or not entries:
                continue

            # Each cluster folder corresponds to a group colour, but you also wanted the colour inside crop name.
            group_colour = cmap.get("group_colour")
            if isinstance(group_colour, str):
                group_colour = group_colour.lower()
            else:
                group_colour = None

            for entry in entries:
                bbox_xyxy = entry.get("bbox_xyxy")
                if not (isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4):
                    bbox_xyxy = [0, 0, 0, 0]

                # Your cluster script may name these fields differently; handle the common ones.
                bbox_name = entry.get("crop_file_bbox") or entry.get("bbox") or entry.get("bbox_file")
                mask_name = entry.get("crop_file_mask") or entry.get("mask") or entry.get("mask_file")

                if not (isinstance(bbox_name, str) and isinstance(mask_name, str)):
                    continue

                bbox_path = renders_dir / bbox_name
                mask_path = renders_dir / mask_name
                if not (bbox_path.exists() and mask_path.exists()):
                    continue

                colour_hint = parse_colour_hint_from_filename(mask_name) or parse_colour_hint_from_filename(bbox_name) or group_colour

                # Load tiles for mask + bbox
                try:
                    pv_mask = load_image_tiles(mask_path, max_num=MAX_TILES_MASK).to(dtype=DTYPE)
                    pv_bbox = load_image_tiles(bbox_path, max_num=MAX_TILES_BBOX).to(dtype=DTYPE)
                    if torch.cuda.is_available():
                        pv_mask = pv_mask.cuda()
                        pv_bbox = pv_bbox.cuda()
                except Exception as e:
                    results["clusters"].append({
                        "source_folder": folder.name,
                        "bbox_crop": str(bbox_path),
                        "mask_crop": str(mask_path),
                        "bbox_xyxy": bbox_xyxy,
                        "model_raw": None,
                        "parsed": None,
                        "error": f"tile_load_failed: {e}",
                    })
                    continue

                # Combine 3 images as separate images
                pixel_values = torch.cat((pv_mask, pv_bbox, pv_full), dim=0)
                num_patches_list = [pv_mask.size(0), pv_bbox.size(0), pv_full.size(0)]

                prompt = (
                    "Image-1: <image>\n"
                    "Image-2: <image>\n"
                    "Image-3: <image>\n\n"
                    + build_prompt(colour_hint=colour_hint, bbox_xyxy=[int(x) for x in bbox_xyxy], suggestions=suggestions)
                )

                try:
                    resp = model.chat(
                        tokenizer,
                        pixel_values,
                        prompt,
                        generation_config,
                        num_patches_list=num_patches_list,
                    )
                except Exception as e:
                    results["clusters"].append({
                        "source_folder": folder.name,
                        "bbox_crop": str(bbox_path),
                        "mask_crop": str(mask_path),
                        "bbox_xyxy": bbox_xyxy,
                        "model_raw": None,
                        "parsed": None,
                        "error": f"inference_failed: {e}",
                    })
                    continue

                parsed = extract_json_object(resp)
                final = parsed

                # Apply a hard “cap” rule on top (you asked for this kind of harsh filtering)
                capped = False
                if isinstance(final, dict):
                    conf = final.get("confidence")
                    dec = final.get("decision")
                    if isinstance(conf, (int, float)) and isinstance(dec, str):
                        if dec.lower() == "accept" and float(conf) < float(MIN_ACCEPT_CONF):
                            final = dict(final)
                            final["decision"] = "reject"
                            final["label"] = None
                            final["reason"] = (final.get("reason") or "") + " | auto-reject: confidence below cutoff"
                            capped = True

                results["clusters"].append({
                    "source_folder": folder.name,
                    "bbox_crop": str(bbox_path),
                    "mask_crop": str(mask_path),
                    "bbox_xyxy": bbox_xyxy,
                    "colour_hint": colour_hint,
                    "model_raw": resp,
                    "parsed": parsed,
                    "final": final,
                    "auto_capped": capped,
                })

        out_json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] processed_{idx}: wrote {out_json_path}")

if __name__ == "__main__":
    main()
