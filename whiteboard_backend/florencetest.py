import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration



def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tok(s: str) -> List[str]:
    return [t for t in _norm(s).split() if t]

def pick_label_from_text(desc: str, labels: List[str]) -> Tuple[str, float]:
    """
    Cheap scoring:
      - token overlap (Jaccard-ish)
      - substring bonus (label appears in desc)
      - shorter labels slightly preferred if tied
    Returns: (best_label, score). If score is 0 -> no confident pick.
    """
    dnorm = _norm(desc)
    dtoks = set(_tok(desc))
    if not labels:
        return ("", 0.0)

    best_label = ""
    best_score = 0.0

    for lab in labels:
        lnorm = _norm(lab)
        ltoks = set(_tok(lab))
        if not ltoks:
            continue

        inter = len(dtoks & ltoks)
        union = len(dtoks | ltoks)
        overlap = inter / union if union else 0.0

        sub_bonus = 0.35 if (lnorm and lnorm in dnorm) else 0.0

        score = overlap + sub_bonus

        # tiny bias to prefer slightly more specific labels (2-3 tokens)
        if 2 <= len(ltoks) <= 3:
            score += 0.05

        if score > best_score or (abs(score - best_score) < 1e-9 and len(lab) < len(best_label)):
            best_score = score
            best_label = lab

    if best_score <= 0.0:
        return ("", 0.0)
    return (best_label, best_score)


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



def labels_from_processed_ocr(
    full_image_path: str,
    *,
    processed_dir: str = "",
    max_labels: int = 20,
) -> List[str]:
    """
    Reads ProccessedImages/<image_basename>.json (or overrides) and returns <= max_labels
    candidate labels. Strongly filters to avoid prose.

    Keeps:
      - "Heading:" style OCR (colon present -> take left side)
      - short TitleCase phrases (e.g. "Cytoplasm", "Nuclear envelope")
    """

    # --- locate matching OCR json
    base = os.path.splitext(os.path.basename(full_image_path))[0]
    want = base + ".json"

    candidates = [
        os.path.join(os.path.dirname(full_image_path), want),
    ]
    if processed_dir:
        candidates.append(os.path.join(processed_dir, want))

    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "ProccessedImages", want))
    candidates.append(os.path.join(here, "ProcessedImages", want))

    ocr_json = next((p for p in candidates if os.path.exists(p)), "")
    if not ocr_json:
        raise FileNotFoundError("Processed OCR JSON not found. Tried:\n  " + "\n  ".join(candidates))

    with open(ocr_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = data.get("words", [])
    if not isinstance(words, list) or not words:
        return []

    STOP = {
        "a","an","the","and","or","but","to","of","in","on","for","with","from","into","out","up","down",
        "is","are","was","were","be","been","being",
        "this","that","these","those","there","here","where","when","what","how","why","which","who",
        "as","at","by","than","then","so","if","it","its","they","them","their","we","our","you","your",
        "can","could","may","might","will","would","should",
        "plus","also",
    }

    def bbox_of(w: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
        bb = w.get("bbox_mask")
        if isinstance(bb, list) and len(bb) == 4:
            return float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        bb = w.get("bbox_anchor")
        if isinstance(bb, list) and len(bb) == 4:
            return float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        return None

    def clean_raw(s: str) -> str:
        s = (s or "").strip()
        s = s.replace("•", " ").replace("·", " ").replace("●", " ").replace("▪", " ").replace("—", " ").replace("–", " ")
        s = s.strip("\"'“”‘’`")
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def norm_key(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def tokenize(s: str) -> List[str]:
        return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]

    def collapse_repeats(toks: List[str]) -> List[str]:
        out = []
        prev = None
        for t in toks:
            if t != prev:
                out.append(t)
            prev = t
        return out

    def titleish(phrase: str) -> bool:
        # Accept things like "Cytoplasm", "Nuclear envelope", "Golgi apparatus"
        ws = [w for w in re.split(r"\s+", phrase.strip()) if w]
        if not (1 <= len(ws) <= 4):
            return False
        good = 0
        for w in ws:
            w2 = re.sub(r"[^A-Za-z0-9]+", "", w)
            if not w2:
                continue
            if w2[0].isupper():
                good += 1
        return good >= max(1, len(ws) - 1)

    # --- collect OCR tokens with simple geometry
    items = []
    for w in words:
        if not isinstance(w, dict):
            continue
        raw = clean_raw(str(w.get("text", "") or ""))
        if not raw:
            continue
        bb = bbox_of(w)
        if not bb:
            continue
        x1, y1, x2, y2 = bb
        cy = 0.5 * (y1 + y2)
        items.append((cy, x1, x2, raw))

    if not items:
        return []

    # --- group into lines by Y, then merge nearby X into phrases
    items.sort(key=lambda t: t[0])
    y_tol = 12.0
    x_gap = 26.0

    lines: List[List[Tuple[float, float, float, str]]] = []
    for it in items:
        if not lines or abs(it[0] - lines[-1][0][0]) > y_tol:
            lines.append([it])
        else:
            lines[-1].append(it)

    phrases: List[str] = []
    for line in lines:
        line.sort(key=lambda t: t[1])  # x1
        cur: List[str] = []
        cur_x2: Optional[float] = None

        def flush():
            nonlocal cur, cur_x2
            if cur:
                phrases.append(clean_raw(" ".join(cur)))
            cur = []
            cur_x2 = None

        for _, x1, x2, txt in line:
            if cur_x2 is None:
                cur = [txt]
                cur_x2 = x2
                continue
            if (x1 - float(cur_x2)) <= x_gap:
                cur.append(txt)
                cur_x2 = max(float(cur_x2), x2)
            else:
                flush()
                cur = [txt]
                cur_x2 = x2
        flush()

    # --- filter + score candidates (hard filter to avoid prose)
    scored: Dict[str, float] = {}
    best_disp: Dict[str, str] = {}

    for ph in phrases:
        if not ph:
            continue

        has_colon = ":" in ph
        cand = ph.split(":", 1)[0].strip() if has_colon else ph.strip()
        cand = cand.strip(" .,:;()[]{}<>/\\|+-_=~!@#$%^&*")
        if not cand:
            continue

        # token-clean, remove stopwords, collapse repeats, limit length
        toks = collapse_repeats(tokenize(cand))
        toks = [t for t in toks if t not in STOP]
        if not toks:
            continue
        toks = toks[:4]
        cleaned = " ".join(toks).strip()
        key = norm_key(cleaned)
        if not key:
            continue

        # HARD filter:
        # - keep colon headings always
        # - otherwise only keep Title-ish short labels
        if not has_colon and not titleish(cand):
            continue

        # Score: colon headings dominate; also prefer shorter labels
        score = (100.0 if has_colon else 20.0) + (6.0 - min(6.0, float(len(toks))))
        prev = scored.get(key, -1e9)
        if score > prev:
            scored[key] = score
            best_disp[key] = cleaned

    if not scored:
        return []

    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], len(kv[0])))
    out: List[str] = []
    for k, _ in ranked:
        out.append(best_disp[k])
        if len(out) >= max_labels:
            break

    return out



def make_choice_prompt(
    label_list: List[str],
    *,
    topic: Optional[str],
    color_name: Optional[str],
) -> Tuple[str, Dict[int, str]]:
    mapping: Dict[int, str] = {i + 1: lab for i, lab in enumerate(label_list)}
    options = "\n".join([f"{i}) {lab}" for i, lab in mapping.items()])

    parts = []
    if topic:
        parts.append(f"Topic: {topic}.")
    if color_name:
        parts.append(f"Color hint: {color_name}.")
    parts.append("Choose exactly one label from the list. Return ONLY the index number.")
    parts.append(options)
    return "\n".join(parts), mapping



def parse_choice_number(text: str, n: int) -> Optional[int]:
    """
    STRICT: accept ONLY a bare integer response, e.g. "7".
    Reject anything that contains option lists / extra words (prompt echo).
    """
    if not text:
        return None

    t = text.strip()

    # If it echoed your options list, reject
    if re.search(r"\b1\)\s|\b2\)\s|\b3\)\s", t):
        return None

    m = re.fullmatch(r"([1-9]\d*)", t)
    if not m:
        return None

    k = int(m.group(1))
    return k if 1 <= k <= n else None



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
    ap.add_argument("--crops_root", default="", help="Optional root folder for crop_file paths (e.g. ClusterRenders)")
    ap.add_argument("--out", default="", help="Output json path (default: alongside input with _labeled)")
    ap.add_argument("--model", default="florence-community/Florence-2-base")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--topic", default="", help="One short label for the whole image (context)")
    ap.add_argument("--prefer_crops", action="store_true", help="Use crop_file if available (recommended)")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=64)

    # labels come from processed OCR jsons (same basename as --image, .json)
    ap.add_argument("--processed_dir", default="", help="Folder containing processed OCR jsons (optional)")
    ap.add_argument("--max_labels", type=int, default=20)

    # matcher controls
    ap.add_argument("--match_min", type=float, default=0.18, help="Min similarity to accept a label (else empty)")
    ap.add_argument("--always_pick", action="store_true", help="Always pick best label even if score < match_min")

    args = ap.parse_args()

    # ----------------- helpers (no deps) -----------------
    import difflib

    STOP = {
        "a","an","the","and","or","but","to","of","in","on","for","with","from","into","out","up","down",
        "is","are","was","were","be","been","being",
        "this","that","these","those","there","here","where","when","what","how","why","which","who",
        "as","at","by","than","then","so","if","it","its","they","them","their","we","our","you","your",
        "can","could","may","might","will","would","should","plus","also",
    }

    def _norm(s: str) -> str:
        s = (s or "").lower()
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _tokens(s: str) -> List[str]:
        parts = [x for x in _norm(s).split(" ") if x and x not in STOP]
        out: List[str] = []
        prev = None
        for x in parts:
            # tiny plural trim helps OCR noise a bit (ribosomes/ribosome)
            if len(x) > 3 and x.endswith("s"):
                x2 = x[:-1]
                if x2 and x2 != prev:
                    out.append(x2)
                    prev = x2
                    continue
            if x != prev:
                out.append(x)
                prev = x
        return out

    def _score(desc: str, label: str) -> float:
        d = _norm(desc)
        l = _norm(label)
        if not d or not l:
            return 0.0

        dt = set(_tokens(d))
        lt = set(_tokens(l))
        jacc = (len(dt & lt) / len(dt | lt)) if (dt or lt) else 0.0
        seq = difflib.SequenceMatcher(a=d, b=l).ratio()

        sub = 0.0
        if l and l in d:
            sub = 0.20
        elif d and d in l:
            sub = 0.10

        return 0.55 * jacc + 0.45 * seq + sub

    def best_label(desc: str, labels: List[str]) -> Tuple[str, float]:
        best_lab = ""
        best_sc = 0.0
        for lab in labels:
            sc = _score(desc, lab)
            if sc > best_sc:
                best_lab, best_sc = lab, sc
        return best_lab, best_sc

    def _resolve_crop_path(crop_file: str, base_dir: str, crops_root: str) -> str:
        # priority: crops_root/<file>, base_dir/crops_root/<file>, base_dir/<file>, absolute
        if os.path.isabs(crop_file):
            return crop_file

        cand: List[str] = []
        if crops_root:
            cand.append(os.path.join(crops_root, crop_file))
            cand.append(os.path.join(base_dir, crops_root, crop_file))

        cand.append(os.path.join(base_dir, crop_file))
        cand.append(crop_file)

        for p in cand:
            if os.path.exists(p):
                return p
        return cand[0] if cand else crop_file

    # ----------------- load cluster map json -----------------
    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    clusters: List[Dict[str, Any]] = data.get("clusters", [])
    if not clusters:
        raise RuntimeError("No clusters in json")

    base_dir = os.path.dirname(args.json)

    full_image_path = resolve_full_image_path(args.json, data, args.image)
    if not os.path.exists(full_image_path):
        raise FileNotFoundError(f"Full image not found: {full_image_path}")

    # ----------------- labels from processed OCR json -----------------
    label_list = labels_from_processed_ocr(
        full_image_path,
        processed_dir=args.processed_dir.strip(),
        max_labels=int(args.max_labels),
    )

    if len(label_list) < 2:
        raise RuntimeError(
            "Could not build a usable label list from processed OCR json.\n"
            f"Image: {full_image_path}\n"
            f"processed_dir: {args.processed_dir or '(auto)'}\n"
            f"got: {label_list}"
        )

    data["ocr_label_candidates"] = label_list
    data["processed_dir_used"] = args.processed_dir.strip() or "(auto)"
    if args.topic.strip():
        data["topic"] = args.topic.strip()

    print("OCR label candidates:", label_list)

    # ----------------- model + image -----------------
    model, processor, dev, _dtype = load_florence(args.model, args.device, args.dtype)
    print("Device:", dev, "dtype:", next(model.parameters()).dtype)

    full_img = Image.open(full_image_path).convert("RGB")
    W, H = full_img.size

    # ----------------- infer -----------------
    for i, c in enumerate(clusters, 1):
        used_mode = "failed"
        raw_out = ""
        chosen = ""
        sc = 0.0

        crop_img = None
        crop_file = c.get("crop_file")

        # ---- Mode 1: caption crop (ONLY "<CAPTION>")
        if args.prefer_crops and isinstance(crop_file, str) and crop_file:
            crop_path = _resolve_crop_path(crop_file, base_dir, args.crops_root.strip())
            if os.path.exists(crop_path):
                crop_img = Image.open(crop_path).convert("RGB")

        try:
            if crop_img is not None:
                used_mode = "crop_caption"
                raw_out = florence_generate_text(
                    model, processor, dev, crop_img, "<CAPTION>",
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.beams,
                )
            else:
                # ---- Mode 2: describe region on full image (ONLY "<REGION_TO_DESCRIPTION><loc...>")
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
                    used_mode = "full_region_desc"
                    raw_out = florence_generate_text(
                        model, processor, dev, full_img, "<REGION_TO_DESCRIPTION>" + loc,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.beams,
                    )
                else:
                    used_mode = "full_caption"
                    raw_out = florence_generate_text(
                        model, processor, dev, full_img, "<CAPTION>",
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.beams,
                    )

            # map description -> one OCR label
            if raw_out and "no object detected" not in raw_out.lower():
                best_lab, best_sc = best_label(raw_out, label_list)
                if args.always_pick or best_sc >= float(args.match_min):
                    chosen, sc = best_lab, best_sc

        except Exception as e:
            c["local_vlm_label"] = ""
            c["local_vlm_raw"] = raw_out or ""
            c["local_vlm_mode"] = used_mode
            c["local_vlm_score"] = 0.0
            c["local_vlm_error"] = f"{type(e).__name__}: {e}"
            if i % 10 == 0 or i == len(clusters):
                print(f"[{i}/{len(clusters)}] last= ERROR mode={used_mode}")
            continue

        c["local_vlm_label"] = chosen
        c["local_vlm_raw"] = raw_out
        c["local_vlm_mode"] = used_mode
        c["local_vlm_score"] = float(sc)
        c.pop("local_vlm_error", None)

        if i % 10 == 0 or i == len(clusters):
            print(f"[{i}/{len(clusters)}] last={chosen} score={sc:.3f} mode={used_mode}")

    out_path = args.out.strip()
    if not out_path:
        root, ext = os.path.splitext(args.json)
        out_path = root + "_labeled" + ext

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
