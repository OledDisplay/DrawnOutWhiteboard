import argparse
import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration


LABELS_DEFAULT = [
    "cell membrane", "cytoplasm", "nucleus", "nucleolus", "nuclear envelope",
    "rough endoplasmic reticulum", "smooth endoplasmic reticulum", "ribosomes",
    "golgi apparatus", "mitochondrion", "lysosome", "vesicle",
    "centrosome/centriole", "cytoskeleton", "other"
]


def clamp(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def bbox_to_loc_tokens(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> str:
    """
    Florence uses <loc_0>...<loc_999> as quantized coordinates.
    Map pixel coords -> 0..999 bins.
    """
    W = max(W, 1)
    H = max(H, 1)

    def qx(x: int) -> int:
        return clamp(int(round((x / W) * 999)), 0, 999)

    def qy(y: int) -> int:
        return clamp(int(round((y / H) * 999)), 0, 999)

    return f"<loc_{qx(x1)}><loc_{qy(y1)}><loc_{qx(x2)}><loc_{qy(y2)}>"


def make_choice_prompt(
    label_list: List[str],
    *,
    topic: Optional[str],
    color_name: Optional[str],
) -> Tuple[str, Dict[int, str]]:
    """
    Return prompt text + mapping from number -> label.
    Keep it short; Florence is small.
    """
    mapping: Dict[int, str] = {i + 1: lab for i, lab in enumerate(label_list)}
    options = "\n".join([f"{i}) {lab}" for i, lab in mapping.items()])

    parts = []
    if topic:
        parts.append(f"Topic: {topic}.")
    if color_name:
        parts.append(f"Color hint: {color_name}.")
    parts.append(
        "Task: choose exactly ONE option for this object.\n"
        "Return ONLY the number (1..N). No words."
    )
    parts.append(options)

    return "\n".join(parts), mapping


def parse_choice_number(text: str, n: int) -> Optional[int]:
    """
    Extract first integer 1..n from model output.
    """
    if not text:
        return None
    m = re.search(r"\b([1-9]\d*)\b", text)
    if not m:
        return None
    k = int(m.group(1))
    if 1 <= k <= n:
        return k
    return None


def load_florence(model_id: str, device: str, dtype_name: str):
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    dev = torch.device(device if use_cuda else "cpu")

    if dev.type == "cuda":
        if dtype_name == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    model = Florence2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(dev)

    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor, dev, dtype


@torch.inference_mode()
def florence_generate_text(
    model,
    processor,
    dev: torch.device,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    num_beams: int = 1,
) -> str:
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    # Critical: match pixel_values dtype to model weights dtype
    model_dtype = next(model.parameters()).dtype
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)

    out_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=num_beams,
    )

    # skip_special_tokens=True gives cleaner text for parsing numbers/labels
    txt = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    return (txt or "").strip()


def resolve_full_image_path(json_path: str, data: Dict[str, Any], image_override: str) -> str:
    if image_override:
        return image_override

    base_dir = os.path.dirname(json_path)
    rel = data.get("processed_image") or data.get("source_processed_image")
    if rel and isinstance(rel, str):
        p = rel
        if not os.path.isabs(p):
            p = os.path.join(base_dir, p)
        return p

    raise RuntimeError("No --image provided and json has no 'processed_image' field.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to ClusterMaps/<image>.json")
    ap.add_argument("--image", default="", help="Optional full image path. If omitted, uses json.processed_image")
    ap.add_argument("--crops_root", default="", help="Optional root folder for crop_file paths")
    ap.add_argument("--out", default="", help="Output json path (default: alongside input with _labeled)")
    ap.add_argument("--model", default="florence-community/Florence-2-base")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--labels", default="", help="Comma-separated override label list")
    ap.add_argument("--topic", default="", help="One short label for the whole image (context)")
    ap.add_argument("--prefer_crops", action="store_true", help="Use crop_file if available (recommended)")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    clusters: List[Dict[str, Any]] = data.get("clusters", [])
    if not clusters:
        raise RuntimeError("No clusters in json")

    label_list = LABELS_DEFAULT
    if args.labels.strip():
        label_list = [x.strip() for x in args.labels.split(",") if x.strip()]
    if len(label_list) < 2:
        raise RuntimeError("Label list must have at least 2 entries.")

    model, processor, dev, _dtype = load_florence(args.model, args.device, args.dtype)
    print("Device:", dev, "dtype:", next(model.parameters()).dtype)

    base_dir = os.path.dirname(args.json)
    full_image_path = resolve_full_image_path(args.json, data, args.image)
    if not os.path.exists(full_image_path):
        raise FileNotFoundError(f"Full image not found: {full_image_path}")

    full_img = Image.open(full_image_path).convert("RGB")
    W, H = full_img.size

    n = len(label_list)

    for i, c in enumerate(clusters, 1):
        color_name = c.get("color_name") if isinstance(c.get("color_name"), str) else None
        topic = args.topic.strip() or (data.get("topic") if isinstance(data.get("topic"), str) else "")
        prompt_text, mapping = make_choice_prompt(label_list, topic=topic or None, color_name=color_name)

        used_mode = ""
        raw_out = ""
        chosen_label = ""

        # ---- Mode 1: prefer crop_file (fast + stable)
        crop_img = None
        crop_file = c.get("crop_file")
        if args.prefer_crops and isinstance(crop_file, str) and crop_file:
            crop_path = crop_file
            if args.crops_root:
                crop_path = os.path.join(args.crops_root, crop_file)
            elif not os.path.isabs(crop_path):
                crop_path = os.path.join(base_dir, crop_file)

            if os.path.exists(crop_path):
                crop_img = Image.open(crop_path).convert("RGB")

        if crop_img is not None:
            # Use CAPTION task + constrained multiple-choice instruction
            used_mode = "crop_caption_choice"
            prompt = "<CAPTION>\n" + prompt_text
            raw_out = florence_generate_text(
                model, processor, dev, crop_img, prompt,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.beams,
            )
            k = parse_choice_number(raw_out, n)
            if k is not None:
                chosen_label = mapping[k]

        # ---- Mode 2: fallback to region prompt on full image using bbox tokens
        if not chosen_label:
            box = c.get("bbox_xyxy")
            if isinstance(box, list) and len(box) == 4:
                x1, y1, x2, y2 = [int(v) for v in box]
                x1 = clamp(x1, 0, W - 1)
                y1 = clamp(y1, 0, H - 1)
                x2 = clamp(x2, 0, W - 1)
                y2 = clamp(y2, 0, H - 1)
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1

                loc = bbox_to_loc_tokens(x1, y1, x2, y2, W, H)
                used_mode = "full_region_choice"
                prompt = "<REGION_TO_DESCRIPTION>" + loc + "\n" + prompt_text
                raw_out = florence_generate_text(
                    model, processor, dev, full_img, prompt,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.beams,
                )
                k = parse_choice_number(raw_out, n)
                if k is not None:
                    chosen_label = mapping[k]

        # Final fallback: if Florence refuses to output a number, try direct label match
        if not chosen_label and raw_out:
            out_low = raw_out.lower()
            for lab in label_list:
                if lab.lower() in out_low:
                    chosen_label = lab
                    used_mode = used_mode + "_labelmatch"
                    break

        c["local_vlm_label"] = chosen_label
        c["local_vlm_raw"] = raw_out
        c["local_vlm_mode"] = used_mode or "failed"

        if i % 10 == 0 or i == len(clusters):
            print(f"[{i}/{len(clusters)}] last={chosen_label} mode={used_mode}")

    out_path = args.out.strip()
    if not out_path:
        root, ext = os.path.splitext(args.json)
        out_path = root + "_labeled" + ext

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
