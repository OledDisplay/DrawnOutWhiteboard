#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "ProccessedImages"

_LATEST_IDX_PATH = OUTPUT_DIR / "latest_processed_index.json"
_LATEST_PROCESSED_IDX = -1


def _discover_latest_processed_index() -> int:
    mx = -1
    try:
        if OUTPUT_DIR.exists():
            for p in OUTPUT_DIR.iterdir():
                if not p.is_file():
                    continue
                m = re.match(r"processed_(\d+)\.(?:png|json)$", p.name, re.IGNORECASE)
                if not m:
                    continue
                try:
                    mx = max(mx, int(m.group(1)))
                except Exception:
                    pass
    except Exception:
        pass

    try:
        if _LATEST_IDX_PATH.exists():
            obj = json.loads(_LATEST_IDX_PATH.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                v = obj.get("latest_index")
                if isinstance(v, int):
                    mx = max(mx, v)
                elif isinstance(v, str) and v.isdigit():
                    mx = max(mx, int(v))
    except Exception:
        pass

    return mx


def _sync_latest_processed_index() -> int:
    global _LATEST_PROCESSED_IDX
    disk_mx = _discover_latest_processed_index()
    if _LATEST_PROCESSED_IDX < 0:
        _LATEST_PROCESSED_IDX = disk_mx
    else:
        _LATEST_PROCESSED_IDX = max(_LATEST_PROCESSED_IDX, disk_mx)
    return _LATEST_PROCESSED_IDX


def _bump_latest_processed_index(idx: int) -> None:
    global _LATEST_PROCESSED_IDX
    try:
        idx = int(idx)
    except Exception:
        return
    if idx <= _LATEST_PROCESSED_IDX:
        return

    _LATEST_PROCESSED_IDX = idx
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _LATEST_IDX_PATH.write_text(
            json.dumps({"latest_index": int(_LATEST_PROCESSED_IDX)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _normalize_bgr3(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        b, g, r, a = cv2.split(img_bgr)
        bgr = cv2.merge([b, g, r])
        mask = (a == 0)
        if np.any(mask):
            bgr[mask] = [255, 255, 255]
        return bgr
    if img_bgr.ndim == 2:
        return cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)


def translate_images_in_memory(
    image_items: List[Dict[str, Any]],
    *,
    start_index: int = 0,
    save_outputs: bool = False,
    return_path_map: bool = True,
) -> Any:
    """
    Lightweight "diagram=0" translator:
    - no OCR / no masking / no label extraction
    - assigns stable processed_<n> ids
    - optionally writes ProccessedImages artifacts
    """
    out: List[Dict[str, Any]] = []
    unique_to_processed: Dict[str, str] = {}

    _sync_latest_processed_index()
    index = int(max(int(start_index), _LATEST_PROCESSED_IDX + 1))

    for it in image_items:
        try:
            src_path = str(it.get("source_path", "") or "")
            img_bgr = it.get("img_bgr", None)
            if img_bgr is None:
                continue

            norm_bgr = _normalize_bgr3(img_bgr)
            base_context = str(it.get("base_context", "") or "")

            processed_id = f"processed_{index}"
            _bump_latest_processed_index(index)
            unique_to_processed[src_path] = processed_id

            payload = {
                "source_path": src_path,
                "base_context": base_context,
                "image_size": [int(norm_bgr.shape[1]), int(norm_bgr.shape[0])],
                "words": [],
                "translated_without_ocr": True,
                "processed_id": processed_id,
            }

            if save_outputs:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                out_img = OUTPUT_DIR / f"{processed_id}.png"
                out_json = OUTPUT_DIR / f"{processed_id}.json"
                cv2.imwrite(str(out_img), norm_bgr)
                out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

            out.append(
                {
                    "idx": int(index),
                    "source_path": src_path,
                    "base_context": base_context,
                    "masked_bgr": norm_bgr,
                    "payload_json": payload,
                    "processed_id": processed_id,
                }
            )
            index += 1
        except Exception:
            continue

    if return_path_map:
        return out, unique_to_processed
    return out

