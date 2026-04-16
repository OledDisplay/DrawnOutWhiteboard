#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
try:
    import cv2
except Exception:
    cv2 = None


# ----------------------------
# Bundle (worker)
# ----------------------------

@dataclass
class EfficientSAM3Bundle:
    """
    Worker wrapper:
      - model: EfficientSAM3 model built by sam3.model_builder.build_efficientsam3_image_model
      - processor: sam3.model.sam3_image_processor.Sam3Processor(model)
      - device: string ("cuda"/"cpu") or torch.device
    """
    model: Any
    processor: Any
    device: Any
    model_id: str = "efficientsam3"


# ----------------------------
# Helpers
# ----------------------------

def _mask_to_bbox_xyxy(mask: np.ndarray) -> Optional[List[int]]:
    """
    mask: HxW bool/0-1 array
    returns [x1,y1,x2,y2] inclusive-ish pixel bbox (xyxy), or None if empty
    """
    if mask is None:
        return None
    m = _normalize_mask_2d(mask)
    if m is None:
        return None
    if not np.any(m):
        return None
    ys, xs = np.where(m)
    y1 = int(ys.min())
    y2 = int(ys.max())
    x1 = int(xs.min())
    x2 = int(xs.max())
    return [x1, y1, x2, y2]


def _find_processed_png(processed_id: str, *, search_roots: List[Path]) -> Optional[Path]:
    """
    You have inconsistent naming in messages (ProcessedImages vs ProccessedImages).
    This searches multiple roots.
    """
    fname = f"{processed_id}.png" if processed_id.endswith(".png") else f"{processed_id}.png"
    for root in search_roots:
        p = root / fname
        if p.is_file():
            return p
        # also allow "processed_n.png" stored under subfolders
        try:
            hits = list(root.rglob(fname))
            if hits:
                return hits[0]
        except Exception:
            pass
    return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _clean_prompt_list(labels: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in labels or []:
        sx = str(x or "").strip()
        if not sx:
            continue
        key = sx.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(sx)
    return out


def _normalize_mask_2d(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Normalize model masks to 2THRD bool arrays.
    EfficientSAM can emit [N,1,H,W], [1,H,W], [H,W,1], or [H,W].
    """
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.size <= 0:
        return None

    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr.astype(bool)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return np.asarray(arr[0]).astype(bool)
        if arr.shape[-1] == 1:
            return np.asarray(arr[..., 0]).astype(bool)
        # Fallback: union across extra channel-like axis.
        return np.any(arr, axis=0).astype(bool)

    if arr.ndim > 3:
        lead_axes = tuple(range(0, arr.ndim - 2))
        arr2 = np.any(arr, axis=lead_axes)
        arr2 = np.squeeze(arr2)
        if arr2.ndim == 2:
            return arr2.astype(bool)
        if arr2.ndim == 3:
            if arr2.shape[0] == 1:
                return np.asarray(arr2[0]).astype(bool)
            if arr2.shape[-1] == 1:
                return np.asarray(arr2[..., 0]).astype(bool)
            return np.any(arr2, axis=0).astype(bool)

    return None


def _mask_metrics(mask: np.ndarray) -> Dict[str, Any]:
    m = _normalize_mask_2d(mask)
    if m is None:
        return {
            "bbox_xyxy": None,
            "mask_area_px": 0,
            "mask_area_frac": 0.0,
            "component_count": 0,
            "largest_component_px": 0,
            "largest_component_frac": 0.0,
            "bbox_fill_frac": 0.0,
        }
    h = int(m.shape[0]) if m.ndim >= 2 else 0
    w = int(m.shape[1]) if m.ndim >= 2 else 0
    img_area = max(1, h * w)
    area = int(np.count_nonzero(m))
    if area <= 0 or h <= 0 or w <= 0:
        return {
            "bbox_xyxy": None,
            "mask_area_px": 0,
            "mask_area_frac": 0.0,
            "component_count": 0,
            "largest_component_px": 0,
            "largest_component_frac": 0.0,
            "bbox_fill_frac": 0.0,
        }

    ys, xs = np.where(m)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    bbox_area = max(1, bw * bh)

    component_count = 1
    largest_component = area
    if cv2 is not None:
        try:
            num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
            comp_areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, int(num_labels))]
            if comp_areas:
                component_count = len(comp_areas)
                largest_component = max(comp_areas)
        except Exception:
            pass

    return {
        "bbox_xyxy": [x0, y0, x1, y1],
        "mask_area_px": int(area),
        "mask_area_frac": float(area / float(img_area)),
        "component_count": int(component_count),
        "largest_component_px": int(largest_component),
        "largest_component_frac": float(largest_component / float(max(1, area))),
        "bbox_fill_frac": float(area / float(bbox_area)),
    }


def _estimate_render_foreground_hint(image_rgb: np.ndarray, color_name: Optional[str] = None) -> Optional[np.ndarray]:
    arr = np.asarray(image_rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None

    rgb = arr[:, :, :3].astype(np.int16)
    gray = np.round(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.int16)
    dev = np.max(np.abs(rgb - gray[..., None]), axis=2)

    hint = dev >= 14

    cname = str(color_name or "").strip().lower()
    if cname:
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]
        if cname == "red":
            hint = hint | ((r - np.maximum(g, b)) >= 18)
        elif cname == "green":
            hint = hint | ((g - np.maximum(r, b)) >= 18)
        elif cname == "blue":
            hint = hint | ((b - np.maximum(r, g)) >= 18)
        elif cname == "cyan":
            hint = hint | (((g + b) // 2 - r) >= 16)
        elif cname == "magenta":
            hint = hint | (((r + b) // 2 - g) >= 16)
        elif cname == "yellow":
            hint = hint | ((np.minimum(r, g) - b) >= 16)
        elif cname == "orange":
            hint = hint | ((r - b >= 24) & (r >= g) & (g >= b - 4))

    coverage = float(np.count_nonzero(hint) / float(max(1, hint.size)))
    if coverage <= 0.0025:
        return None
    return hint.astype(bool)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _build_render_probe_prompt(label: str, color_name: Optional[str] = None) -> str:
    lab = str(label or "").strip()
    color_txt = str(color_name or "").strip().lower() or "foreground"
    return (
        f"{lab}. "
        f"Focus on the coherent {color_txt} foreground object only. "
        "Ignore gray or noisy background context. "
        "Prefer a substantial structured region with real shape, repeated detail, blobs, branches, membranes, or grouped islands. "
        "Do not lock onto a tiny speck, stray fragment, or accidental thin noise."
    )


def _score_prompt_mask(
    *,
    score: float,
    metrics: Dict[str, Any],
    foreground_hint: Optional[np.ndarray],
    mask: np.ndarray,
) -> Dict[str, Any]:
    mask_bool = np.asarray(mask).astype(bool)
    mask_area = int(metrics.get("mask_area_px", 0) or 0)
    area_frac = float(metrics.get("mask_area_frac", 0.0) or 0.0)
    component_count = int(metrics.get("component_count", 0) or 0)
    largest_component_px = int(metrics.get("largest_component_px", 0) or 0)
    largest_component_frac = float(metrics.get("largest_component_frac", 0.0) or 0.0)
    bbox_fill_frac = float(metrics.get("bbox_fill_frac", 0.0) or 0.0)
    img_area = max(1, int(mask_bool.size))

    fg_overlap = None
    fg_precision = None
    if foreground_hint is not None and foreground_hint.shape == mask_bool.shape:
        inter = int(np.count_nonzero(mask_bool & foreground_hint))
        hint_area = int(np.count_nonzero(foreground_hint))
        fg_overlap = float(inter / float(max(1, hint_area)))
        fg_precision = float(inter / float(max(1, mask_area)))

    tiny_thr = max(48, int(round(img_area * 0.0035)))
    small_thr = max(110, int(round(img_area * 0.0080)))
    largest_thr = max(36, int(round(img_area * 0.0025)))

    tiny_mask = mask_area < tiny_thr or area_frac < 0.0035
    small_mask = mask_area < small_thr or area_frac < 0.0080
    # EfficientSAM3 prompt confidences are often low-magnitude after presence scaling.
    # Calibrate for practical ranges seen on real cluster renders.
    low_score = float(score) < 0.02
    very_low_score = float(score) < 0.008
    fragmented = component_count >= 10 and largest_component_frac < 0.40
    sparse = bbox_fill_frac < 0.05
    poor_foreground_alignment = (
        fg_overlap is not None
        and fg_precision is not None
        and fg_overlap < 0.18
        and fg_precision < 0.18
    )
    speckish = tiny_mask or largest_component_px < largest_thr

    score_term = _clamp01(score)
    area_term = _clamp01((area_frac - 0.006) / 0.080)
    comp_term = _clamp01((largest_component_frac - 0.35) / 0.65)
    fill_term = _clamp01((bbox_fill_frac - 0.03) / 0.35)
    if fg_overlap is None or fg_precision is None:
        fg_term = 0.50
    else:
        fg_term = 0.5 * _clamp01(fg_overlap / 0.35) + 0.5 * _clamp01(fg_precision / 0.35)

    quality = (
        0.38 * score_term
        + 0.24 * area_term
        + 0.18 * comp_term
        + 0.10 * fill_term
        + 0.10 * fg_term
    )
    quality = _clamp01(quality)

    reasons: List[str] = []
    if tiny_mask:
        reasons.append("tiny_mask")
    elif small_mask:
        reasons.append("small_mask")
    if very_low_score:
        reasons.append("very_low_score")
    elif low_score:
        reasons.append("low_score")
    if fragmented:
        reasons.append("fragmented")
    if sparse:
        reasons.append("sparse_fill")
    if poor_foreground_alignment:
        reasons.append("weak_foreground_alignment")
    if speckish and "tiny_mask" not in reasons:
        reasons.append("speckish")

    signal = "bad"
    if (
        quality >= 0.52
        and not tiny_mask
        and float(score) >= 0.012
        and largest_component_frac >= 0.34
        and not sparse
        and not poor_foreground_alignment
    ):
        signal = "good"
    elif (
        quality >= 0.42
        and not tiny_mask
        and float(score) >= 0.010
        and largest_component_px >= largest_thr
    ):
        signal = "marginal"

    return {
        "score": float(score),
        "quality": float(round(quality, 4)),
        "keep_signal": signal,
        "foreground_hint_overlap": None if fg_overlap is None else float(round(fg_overlap, 4)),
        "foreground_hint_precision": None if fg_precision is None else float(round(fg_precision, 4)),
        "flags": {
            "tiny_mask": bool(tiny_mask),
            "small_mask": bool(small_mask),
            "low_score": bool(low_score),
            "very_low_score": bool(very_low_score),
            "fragmented": bool(fragmented),
            "sparse": bool(sparse),
            "poor_foreground_alignment": bool(poor_foreground_alignment),
            "speckish": bool(speckish),
        },
        "reasons": reasons,
    }


def analyze_cluster_render_with_prompts(
    *,
    bundle: EfficientSAM3Bundle,
    image_pil: Image.Image,
    prompts: List[str],
    color_name: Optional[str] = None,
    top_k: int = 3,
    min_score: float = 0.0,
) -> Dict[str, Any]:
    processor = bundle.processor
    cleaned_prompts = _clean_prompt_list(prompts)
    image_rgb = np.asarray(image_pil.convert("RGB"), dtype=np.uint8)
    foreground_hint = _estimate_render_foreground_hint(image_rgb, color_name=color_name)
    base_state = processor.set_image(image_pil)

    prompt_results: Dict[str, Dict[str, Any]] = {}
    ranked_results: List[Dict[str, Any]] = []

    for label in cleaned_prompts:
        prompt = _build_render_probe_prompt(label, color_name=color_name)
        try:
            state_in = dict(base_state)
            state_out = processor.set_text_prompt(prompt=prompt, state=state_in)
            masks = state_out.get("masks", None)
            scores = state_out.get("scores", None)
        except Exception as e:
            prompt_results[label] = {
                "prompt": prompt,
                "bbox_xyxy": None,
                "mask_area_px": 0,
                "mask_area_frac": 0.0,
                "component_count": 0,
                "largest_component_px": 0,
                "largest_component_frac": 0.0,
                "bbox_fill_frac": 0.0,
                "score": 0.0,
                "quality": 0.0,
                "keep_signal": "bad",
                "foreground_hint_overlap": None,
                "foreground_hint_precision": None,
                "flags": {
                    "tiny_mask": True,
                    "small_mask": True,
                    "low_score": True,
                    "very_low_score": True,
                    "fragmented": False,
                    "sparse": True,
                    "poor_foreground_alignment": False,
                    "speckish": True,
                },
                "reasons": [f"prompt_failed:{type(e).__name__}"],
            }
            ranked_results.append({"label": label, **prompt_results[label]})
            continue

        if masks is None:
            prompt_results[label] = {
                "prompt": prompt,
                "bbox_xyxy": None,
                "mask_area_px": 0,
                "mask_area_frac": 0.0,
                "component_count": 0,
                "largest_component_px": 0,
                "largest_component_frac": 0.0,
                "bbox_fill_frac": 0.0,
                "score": 0.0,
                "quality": 0.0,
                "keep_signal": "bad",
                "foreground_hint_overlap": None,
                "foreground_hint_precision": None,
                "flags": {
                    "tiny_mask": True,
                    "small_mask": True,
                    "low_score": True,
                    "very_low_score": True,
                    "fragmented": False,
                    "sparse": True,
                    "poor_foreground_alignment": False,
                    "speckish": True,
                },
                "reasons": ["no_mask"],
            }
            ranked_results.append({"label": label, **prompt_results[label]})
            continue

        try:
            import torch  # optional
            if isinstance(masks, torch.Tensor):
                masks_np = masks.detach().cpu().numpy()
            else:
                masks_np = np.asarray(masks)
        except Exception:
            masks_np = np.asarray(masks)

        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0, :, :]
        elif masks_np.ndim == 4 and masks_np.shape[-1] == 1:
            masks_np = masks_np[..., 0]
        elif masks_np.ndim == 2:
            masks_np = masks_np[None, :, :]

        if scores is None:
            scores_np = np.ones((masks_np.shape[0],), dtype=np.float32)
        else:
            try:
                import torch
                if isinstance(scores, torch.Tensor):
                    scores_np = scores.detach().cpu().numpy().astype(np.float32)
                else:
                    scores_np = np.asarray(scores).astype(np.float32)
            except Exception:
                scores_np = np.asarray(scores).astype(np.float32)
        scores_np = np.asarray(scores_np, dtype=np.float32).reshape(-1)

        best_row: Optional[Dict[str, Any]] = None
        max_masks = max(1, int(top_k))
        for i in range(min(int(masks_np.shape[0]), max_masks)):
            sc = float(scores_np[i]) if i < len(scores_np) else 1.0
            if sc < float(min_score):
                continue

            mask_bool = _normalize_mask_2d(masks_np[i])
            if mask_bool is None:
                continue
            metrics = _mask_metrics(mask_bool)
            if metrics.get("bbox_xyxy") is None:
                continue

            scored = _score_prompt_mask(
                score=sc,
                metrics=metrics,
                foreground_hint=foreground_hint,
                mask=mask_bool,
            )
            candidate = {
                "label": label,
                "prompt": prompt,
                **metrics,
                **scored,
            }

            if (
                best_row is None
                or float(candidate.get("quality", 0.0)) > float(best_row.get("quality", 0.0))
                or (
                    float(candidate.get("quality", 0.0)) == float(best_row.get("quality", 0.0))
                    and float(candidate.get("score", 0.0)) > float(best_row.get("score", 0.0))
                )
            ):
                best_row = candidate

        if best_row is None:
            best_row = {
                "label": label,
                "prompt": prompt,
                "bbox_xyxy": None,
                "mask_area_px": 0,
                "mask_area_frac": 0.0,
                "component_count": 0,
                "largest_component_px": 0,
                "largest_component_frac": 0.0,
                "bbox_fill_frac": 0.0,
                "score": 0.0,
                "quality": 0.0,
                "keep_signal": "bad",
                "foreground_hint_overlap": None,
                "foreground_hint_precision": None,
                "flags": {
                    "tiny_mask": True,
                    "small_mask": True,
                    "low_score": True,
                    "very_low_score": True,
                    "fragmented": False,
                    "sparse": True,
                    "poor_foreground_alignment": False,
                    "speckish": True,
                },
                "reasons": ["empty_after_thresholding"],
            }

        prompt_results[label] = {k: v for k, v in best_row.items() if k != "label"}
        ranked_results.append(best_row)

    ranked_results.sort(
        key=lambda r: (
            float(r.get("quality", 0.0) or 0.0),
            float(r.get("score", 0.0) or 0.0),
            float(r.get("mask_area_frac", 0.0) or 0.0),
        ),
        reverse=True,
    )

    prompt_count = len(ranked_results)
    good_count = sum(1 for r in ranked_results if str(r.get("keep_signal", "")) == "good")
    marginal_count = sum(1 for r in ranked_results if str(r.get("keep_signal", "")) == "marginal")
    bad_count = max(0, prompt_count - good_count - marginal_count)
    small_count = sum(1 for r in ranked_results if bool(((r.get("flags") or {}).get("small_mask"))))
    low_conf_count = sum(1 for r in ranked_results if bool(((r.get("flags") or {}).get("low_score"))))

    best = ranked_results[0] if ranked_results else {}
    second = ranked_results[1] if len(ranked_results) > 1 else {}
    best_score = float(best.get("score", 0.0) or 0.0)
    best_quality = float(best.get("quality", 0.0) or 0.0)
    second_quality = float(second.get("quality", 0.0) or 0.0)

    keep = False
    decision_reason = "no_prompt_results"
    if prompt_count <= 0:
        keep = False
        decision_reason = "no_prompt_results"
    elif best_score < 0.013:
        keep = False
        decision_reason = "best_prompt_score_below_min_keep"
    elif best_quality > 0.38 and best_score > 0.025:
        keep = True
        decision_reason = "auto_keep_quality_and_score_gate"
    elif small_count >= prompt_count and best_quality < 0.58:
        keep = False
        decision_reason = "all_prompt_masks_small"
    elif low_conf_count >= prompt_count and best_quality < 0.30:
        keep = False
        decision_reason = "all_prompt_scores_low"
    elif good_count >= 1 and best_quality >= 0.52:
        keep = True
        decision_reason = "strong_prompt_mask"
    elif marginal_count >= 2 and best_quality >= 0.42 and second_quality >= 0.36:
        keep = True
        decision_reason = "multiple_marginal_masks"
    elif marginal_count >= 1 and best_quality >= 0.56:
        keep = True
        decision_reason = "single_strong_mask"
    elif best_quality >= 0.60 and small_count < prompt_count:
        keep = True
        decision_reason = "high_quality_despite_low_conf"
    elif best_quality < 0.40:
        keep = False
        decision_reason = "best_prompt_quality_too_low"
    else:
        keep = False
        decision_reason = "prompts_consistently_bad"

    return {
        "keep": bool(keep),
        "decision_reason": decision_reason,
        "best_prompt": str(best.get("label", "") or "") or None,
        "best_score": float(best_score),
        "best_quality": float(round(best_quality, 4)),
        "aggregate": {
            "prompt_count": int(prompt_count),
            "good_prompt_count": int(good_count),
            "marginal_prompt_count": int(marginal_count),
            "bad_prompt_count": int(bad_count),
            "small_prompt_count": int(small_count),
            "low_conf_prompt_count": int(low_conf_count),
            "top_two_quality_gap": float(round(best_quality - second_quality, 4)),
        },
        "prompt_results": prompt_results,
    }


def prune_cluster_renders_for_processed_images(
    *,
    bundle: EfficientSAM3Bundle,
    processed_id_to_labels: Dict[str, List[str]],
    cluster_maps_root: Optional[str] = None,
    cluster_renders_root: Optional[str] = None,
    top_k: int = 3,
    min_score: float = 0.0,
    save_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    maps_root = Path(cluster_maps_root) if cluster_maps_root else (Path(__file__).resolve().parent / "ClusterMaps")
    renders_root = Path(cluster_renders_root) if cluster_renders_root else (Path(__file__).resolve().parent / "ClusterRenders")

    result: Dict[str, Any] = {"by_processed_id": {}}

    for pid, labels in (processed_id_to_labels or {}).items():
        pid_s = str(pid or "").strip()
        cleaned_prompts = _clean_prompt_list(labels)
        row: Dict[str, Any] = {
            "processed_id": pid_s,
            "labels": cleaned_prompts,
            "prompts": cleaned_prompts,
            "clusters_total": 0,
            "kept_mask_names": [],
            "dropped_mask_names": [],
            "by_mask": {},
        }

        if not pid_s:
            continue

        cmap_path = maps_root / pid_s / "clusters.json"
        renders_dir = renders_root / pid_s
        if not cmap_path.is_file() or not renders_dir.is_dir():
            row["error"] = "missing_cluster_maps_or_renders"
            result["by_processed_id"][pid_s] = row
            continue

        try:
            cmap = json.loads(cmap_path.read_text(encoding="utf-8"))
        except Exception as e:
            row["error"] = f"bad_clusters_json:{e}"
            result["by_processed_id"][pid_s] = row
            continue

        clusters = cmap.get("clusters") or []
        if not isinstance(clusters, list):
            clusters = []
        row["clusters_total"] = int(len(clusters))

        for idx, entry in enumerate(clusters, start=1):
            if not isinstance(entry, dict):
                continue

            mask_name = str(entry.get("crop_file_mask", "") or "").strip()
            if not mask_name:
                continue

            mask_path = renders_dir / mask_name
            if not mask_path.is_file():
                row["by_mask"][mask_name] = {
                    "keep": False,
                    "decision_reason": "render_missing",
                    "cluster_index": int(idx),
                    "color_name": str(entry.get("color_name", "") or "").strip() or None,
                }
                row["dropped_mask_names"].append(mask_name)
                continue

            try:
                with Image.open(str(mask_path)) as im:
                    image_pil = im.convert("RGB")
            except Exception as e:
                row["by_mask"][mask_name] = {
                    "keep": False,
                    "decision_reason": f"render_open_failed:{e}",
                    "cluster_index": int(idx),
                    "color_name": str(entry.get("color_name", "") or "").strip() or None,
                }
                row["dropped_mask_names"].append(mask_name)
                continue

            try:
                analysis = analyze_cluster_render_with_prompts(
                    bundle=bundle,
                    image_pil=image_pil,
                    prompts=cleaned_prompts,
                    color_name=str(entry.get("color_name", "") or "").strip() or None,
                    top_k=top_k,
                    min_score=min_score,
                )
            except Exception as e:
                row["by_mask"][mask_name] = {
                    "keep": False,
                    "decision_reason": f"analysis_failed:{type(e).__name__}",
                    "cluster_index": int(idx),
                    "color_name": str(entry.get("color_name", "") or "").strip() or None,
                }
                row["dropped_mask_names"].append(mask_name)
                continue
            analysis["cluster_index"] = int(idx)
            analysis["color_name"] = str(entry.get("color_name", "") or "").strip() or None
            analysis["bbox_xyxy"] = [int(x) for x in entry.get("bbox_xyxy", [])] if isinstance(entry.get("bbox_xyxy"), list) and len(entry.get("bbox_xyxy")) == 4 else None
            row["by_mask"][mask_name] = analysis

            try:
                print(
                    "[sam-cutdown] pid=%s mask=%s keep=%s best_prompt=%s best_score=%.4f best_quality=%.4f reason=%s"
                    % (
                        pid_s,
                        mask_name,
                        bool(analysis.get("keep", False)),
                        str(analysis.get("best_prompt", "") or ""),
                        float(analysis.get("best_score", 0.0) or 0.0),
                        float(analysis.get("best_quality", 0.0) or 0.0),
                        str(analysis.get("decision_reason", "") or ""),
                    )
                )
                prompt_results = analysis.get("prompt_results") if isinstance(analysis.get("prompt_results"), dict) else {}
                for prompt_label, prompt_row in prompt_results.items():
                    if not isinstance(prompt_row, dict):
                        continue
                    print(
                        "  [sam-cutdown][prompt] label=%s keep_signal=%s score=%.4f quality=%.4f area_frac=%.4f reasons=%s"
                        % (
                            str(prompt_label or ""),
                            str(prompt_row.get("keep_signal", "") or ""),
                            float(prompt_row.get("score", 0.0) or 0.0),
                            float(prompt_row.get("quality", 0.0) or 0.0),
                            float(prompt_row.get("mask_area_frac", 0.0) or 0.0),
                            ",".join(str(x) for x in (prompt_row.get("reasons") or []) if str(x)),
                        )
                    )
            except Exception:
                pass

            if bool(analysis.get("keep", False)):
                row["kept_mask_names"].append(mask_name)
            else:
                row["dropped_mask_names"].append(mask_name)

        result["by_processed_id"][pid_s] = row

    if save_json_path:
        p = Path(save_json_path)
        _ensure_dir(p.parent)
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


# ----------------------------
# Core API
# ----------------------------

def compute_text_prompt_bboxes_for_image(
    *,
    bundle: EfficientSAM3Bundle,
    image_pil: Image.Image,
    prompts: List[str],
    top_k: int = 3,
    min_score: float = 0.0,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    For one image, run EfficientSAM3 text prompting for each prompt string.

    Returns:
      {
        "label": [
           {"bbox_xyxy": [x1,y1,x2,y2], "score": float},
           ...
        ],
        ...
      }
    """
    out: Dict[str, List[Dict[str, Any]]] = {}

    # EfficientSAM3 API from their README:
    #   inference_state = processor.set_image(image)
    #   inference_state = processor.set_text_prompt(prompt="shoe", state=inference_state)
    #   masks = inference_state["masks"]
    #   scores = inference_state["scores"]
    processor = bundle.processor

    # base per-image state
    base_state = processor.set_image(image_pil)

    for label in prompts:
        lbl = (label or "").strip()
        if not lbl:
            continue

        # Avoid mutating base_state if their processor mutates in-place:
        state_in = dict(base_state)

        state_out = processor.set_text_prompt(prompt=lbl, state=state_in)
        masks = state_out.get("masks", None)
        scores = state_out.get("scores", None)

        if masks is None:
            out[lbl] = []
            continue

        # masks may be torch tensors or numpy arrays
        try:
            import torch  # optional
            if isinstance(masks, torch.Tensor):
                masks_np = masks.detach().cpu().numpy()
            else:
                masks_np = np.asarray(masks)
        except Exception:
            masks_np = np.asarray(masks)

        if scores is None:
            scores_np = np.ones((masks_np.shape[0],), dtype=np.float32)
        else:
            try:
                import torch
                if isinstance(scores, torch.Tensor):
                    scores_np = scores.detach().cpu().numpy().astype(np.float32)
                else:
                    scores_np = np.asarray(scores).astype(np.float32)
            except Exception:
                scores_np = np.asarray(scores).astype(np.float32)
        scores_np = np.asarray(scores_np, dtype=np.float32).reshape(-1)

        # normalize shapes: [N,H,W]
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0, :, :]
        elif masks_np.ndim == 4 and masks_np.shape[-1] == 1:
            masks_np = masks_np[..., 0]
        elif masks_np.ndim == 2:
            masks_np = masks_np[None, :, :]

        picks: List[Tuple[float, List[int]]] = []
        for i in range(int(masks_np.shape[0])):
            sc = float(scores_np[i]) if i < len(scores_np) else 1.0
            if sc < float(min_score):
                continue
            bbox = _mask_to_bbox_xyxy(masks_np[i])
            if bbox is None:
                continue
            picks.append((sc, bbox))

        picks.sort(key=lambda x: x[0], reverse=True)
        picks = picks[: max(1, int(top_k))]

        out[lbl] = [{"bbox_xyxy": bb, "score": float(sc)} for sc, bb in picks]

    return out


def compute_label_bboxes_for_processed_images(
    *,
    bundle: EfficientSAM3Bundle,
    processed_id_to_labels: Dict[str, List[str]],
    processed_images_roots: Optional[List[str]] = None,
    top_k: int = 3,
    min_score: float = 0.0,
    save_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entry:
      processed_id_to_labels: { "processed_12": ["nucleus", "membrane", ...], ... }

    Returns:
      {
        "by_processed_id": {
          "processed_12": {
             "image_path": "...",
             "labels": { "nucleus": [{"bbox_xyxy":[...],"score":...}, ...], ... }
          },
          ...
        }
      }
    """
    roots: List[Path] = []
    if processed_images_roots:
        roots = [Path(r) for r in processed_images_roots]
    else:
        # common roots (covering your naming inconsistency + pipeline output folder)
        roots = [
            Path("ProcessedImages"),
            Path("ProccessedImages"),
            Path(__file__).resolve().parent / "ProcessedImages",
            Path(__file__).resolve().parent / "ProccessedImages",
            Path(__file__).resolve().parent / "PipelineOutputs" / "ProcessedImages",
            Path(__file__).resolve().parent / "PipelineOutputs" / "ProccessedImages",
            Path("PipelineOutputs") / "ProcessedImages",
            Path("PipelineOutputs") / "ProccessedImages",
        ]

    result: Dict[str, Any] = {"by_processed_id": {}}

    for pid, labels in (processed_id_to_labels or {}).items():
        pid_s = str(pid or "").strip()
        if not pid_s:
            continue

        img_path = _find_processed_png(pid_s, search_roots=roots)
        if img_path is None:
            result["by_processed_id"][pid_s] = {"image_path": None, "labels": {}, "error": "processed_png_not_found"}
            continue

        try:
            im = Image.open(str(img_path)).convert("RGB")
        except Exception as e:
            result["by_processed_id"][pid_s] = {"image_path": str(img_path), "labels": {}, "error": f"open_failed:{e!r}"}
            continue

        label_map = compute_text_prompt_bboxes_for_image(
            bundle=bundle,
            image_pil=im,
            prompts=list(labels or []),
            top_k=top_k,
            min_score=min_score,
        )

        result["by_processed_id"][pid_s] = {
            "image_path": str(img_path),
            "labels": label_map,
        }

    if save_json_path:
        p = Path(save_json_path)
        _ensure_dir(p.parent)
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result
