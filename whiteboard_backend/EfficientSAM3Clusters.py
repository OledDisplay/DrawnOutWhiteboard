#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


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
    m = mask.astype(bool)
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

        # normalize shapes: [N,H,W]
        if masks_np.ndim == 2:
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