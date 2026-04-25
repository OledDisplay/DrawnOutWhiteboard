#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import math
import os
import shutil
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cosine
from transformers import AutoImageProcessor, Dinov2Model, Sam2Model, Sam2Processor


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "ProccessedImages"
CLUSTER_RENDER_DIR = BASE_DIR / "ClusterRenders"
CLUSTER_MAP_DIR = BASE_DIR / "DiagramMaskClusterMaps"
DEBUG_ROOT = BASE_DIR / "PipelineOutputs" / "diagram_mask_clusters"
VECTORS_DIR = BASE_DIR / "StrokeVectors"
SAM_DINO_RENDER_DIR = BASE_DIR / "sam_dino_renders"

_PROCESSED_PLAIN_RE = __import__("re").compile(r"^(?:proccessed|processed)_(\d+)\.png$", __import__("re").IGNORECASE)

_SAM2_BUNDLE: Optional[Dict[str, Any]] = None
_DINO_BUNDLE: Optional[Dict[str, Any]] = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def get_diagram_cluster_backend() -> str:
    text = str(os.environ.get("DIAGRAM_CLUSTER_BACKEND", "stroke") or "stroke").strip().lower()
    return text if text in {"stroke", "sam2_dinov2"} else "stroke"


def _torch_device() -> torch.device:
    force_cpu = _env_bool("DIAGRAM_CLUSTER_FORCE_CPU", False)
    if torch.cuda.is_available() and not force_cpu:
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class DiagramMaskClusterConfig:
    sam_model_id: str = (
        os.environ.get("DIAGRAM_CLUSTER_SAM_MODEL_ID")
        or os.environ.get("DIAGRAM_CLUSTER_SAM2_MODEL_ID")
        or "facebook/sam2.1-hiera-large"
    )
    dino_model_id: str = os.environ.get("DIAGRAM_CLUSTER_DINO_MODEL_ID", "facebook/dinov2-base")
    refiner_model_id: str = os.environ.get("DIAGRAM_CLUSTER_REFINER_MODEL_ID", "").strip()
    device: str = str(_torch_device())

    # pre-clean
    dark_threshold: int = _env_int("DIAGRAM_CLUSTER_DARK_THRESHOLD", 58)
    dark_rgb_ceiling: int = _env_int("DIAGRAM_CLUSTER_DARK_RGB_CEILING", 92)
    thin_component_max_width: int = _env_int("DIAGRAM_CLUSTER_THIN_COMPONENT_MAX_WIDTH", 11)
    thin_component_min_aspect: float = _env_float("DIAGRAM_CLUSTER_THIN_COMPONENT_MIN_ASPECT", 4.0)
    dot_component_max_area: int = _env_int("DIAGRAM_CLUSTER_DOT_COMPONENT_MAX_AREA", 180)
    line_component_max_area: int = _env_int("DIAGRAM_CLUSTER_LINE_COMPONENT_MAX_AREA", 1800)
    line_hough_threshold: int = _env_int("DIAGRAM_CLUSTER_LINE_HOUGH_THRESHOLD", 18)
    line_min_length_px: int = _env_int("DIAGRAM_CLUSTER_LINE_MIN_LENGTH_PX", 36)
    line_max_gap_px: int = _env_int("DIAGRAM_CLUSTER_LINE_MAX_GAP_PX", 8)
    line_hough_thickness_px: int = _env_int("DIAGRAM_CLUSTER_LINE_HOUGH_THICKNESS_PX", 5)
    removal_mask_expand_px: int = _env_int("DIAGRAM_CLUSTER_REMOVAL_MASK_EXPAND_PX", 2)
    inpaint_radius: int = _env_int("DIAGRAM_CLUSTER_INPAINT_RADIUS", 3)

    # sam2 proposal generation
    point_grid_step: int = _env_int("DIAGRAM_CLUSTER_POINT_GRID_STEP", 64)
    point_batch_size: int = _env_int("DIAGRAM_CLUSTER_POINT_BATCH_SIZE", 32)
    sam_pred_iou_thresh: float = _env_float("DIAGRAM_CLUSTER_SAM_PRED_IOU_THRESH", 0.84)
    sam_stability_score_thresh: float = _env_float("DIAGRAM_CLUSTER_SAM_STABILITY_SCORE_THRESH", 0.82)
    sam_mask_threshold: float = _env_float("DIAGRAM_CLUSTER_SAM_MASK_THRESHOLD", 0.0)
    sam_stability_offset: float = _env_float("DIAGRAM_CLUSTER_SAM_STABILITY_OFFSET", 0.05)
    min_region_area_px: int = _env_int("DIAGRAM_CLUSTER_MIN_REGION_AREA_PX", 320)
    max_points: int = _env_int("DIAGRAM_CLUSTER_MAX_POINTS", 240)

    # prompt-level mini-mask consolidation
    prompt_keep_max_per_point: int = _env_int("DIAGRAM_CLUSTER_PROMPT_KEEP_MAX_PER_POINT", 3)
    prompt_same_scale_area_ratio_max: float = _env_float("DIAGRAM_CLUSTER_PROMPT_SAME_SCALE_AREA_RATIO_MAX", 1.55)
    prompt_same_scale_iou_thresh: float = _env_float("DIAGRAM_CLUSTER_PROMPT_SAME_SCALE_IOU_THRESH", 0.52)
    prompt_parent_containment_thresh: float = _env_float("DIAGRAM_CLUSTER_PROMPT_PARENT_CONTAINMENT_THRESH", 0.86)
    prompt_parent_max_area_ratio: float = _env_float("DIAGRAM_CLUSTER_PROMPT_PARENT_MAX_AREA_RATIO", 5.0)

    # cleanup / suppression
    annotation_overlap_drop_ratio: float = _env_float("DIAGRAM_CLUSTER_ANNOTATION_OVERLAP_DROP_RATIO", 0.45)
    proposal_nms_iou_thresh: float = _env_float("DIAGRAM_CLUSTER_PROPOSAL_NMS_IOU_THRESH", 0.72)
    proposal_containment_thresh: float = _env_float("DIAGRAM_CLUSTER_PROPOSAL_CONTAINMENT_THRESH", 0.92)

    # feature extraction
    dino_image_size: int = _env_int("DIAGRAM_CLUSTER_DINO_IMAGE_SIZE", 224)
    histogram_bins: int = _env_int("DIAGRAM_CLUSTER_HISTOGRAM_BINS", 8)
    crop_pad_px: int = _env_int("DIAGRAM_CLUSTER_CROP_PAD_PX", 10)

    # merge
    merge_min_dino_cosine: float = _env_float("DIAGRAM_CLUSTER_MERGE_MIN_DINO_COSINE", 0.92)
    merge_min_color_similarity: float = _env_float("DIAGRAM_CLUSTER_MERGE_MIN_COLOR_SIM", 0.78)
    merge_min_shape_similarity: float = _env_float("DIAGRAM_CLUSTER_MERGE_MIN_SHAPE_SIM", 0.74)
    merge_min_contrast_color_similarity: float = _env_float("DIAGRAM_CLUSTER_MERGE_MIN_CONTRAST_COLOR_SIM", 0.82)
    merge_combined_feature_score: float = _env_float("DIAGRAM_CLUSTER_MERGE_COMBINED_FEATURE_SCORE", 0.85)
    merge_soft_min_dino_cosine: float = _env_float("DIAGRAM_CLUSTER_MERGE_SOFT_MIN_DINO_COSINE", 0.89)
    merge_soft_min_shape_similarity: float = _env_float("DIAGRAM_CLUSTER_MERGE_SOFT_MIN_SHAPE_SIM", 0.68)
    merge_similarity_only_min_dino_cosine: float = _env_float("DIAGRAM_CLUSTER_MERGE_SIMILARITY_ONLY_MIN_DINO_COSINE", 0.94)
    merge_similarity_only_min_color_similarity: float = _env_float("DIAGRAM_CLUSTER_MERGE_SIMILARITY_ONLY_MIN_COLOR_SIM", 0.88)
    merge_similarity_only_min_shape_similarity: float = _env_float("DIAGRAM_CLUSTER_MERGE_SIMILARITY_ONLY_MIN_SHAPE_SIM", 0.84)
    merge_similarity_only_min_score: float = _env_float("DIAGRAM_CLUSTER_MERGE_SIMILARITY_ONLY_MIN_SCORE", 0.90)
    merge_similarity_only_bbox_gap_px: float = _env_float("DIAGRAM_CLUSTER_MERGE_SIMILARITY_ONLY_BBOX_GAP_PX", 42.0)
    merge_similarity_only_bbox_growth_max: float = _env_float("DIAGRAM_CLUSTER_MERGE_SIMILARITY_ONLY_BBOX_GROWTH_MAX", 0.30)
    merge_iou_thresh: float = _env_float("DIAGRAM_CLUSTER_MERGE_IOU_THRESH", 0.18)
    merge_containment_thresh: float = _env_float("DIAGRAM_CLUSTER_MERGE_CONTAINMENT_THRESH", 0.84)
    merge_bbox_gap_px: float = _env_float("DIAGRAM_CLUSTER_MERGE_BBOX_GAP_PX", 22.0)
    merge_bbox_growth_max: float = _env_float("DIAGRAM_CLUSTER_MERGE_BBOX_GROWTH_MAX", 0.40)

    # samrefiner-style post-filter
    refiner_enabled: bool = _env_bool("DIAGRAM_CLUSTER_REFINER_ENABLED", False)
    refiner_prompt_pad_px: int = _env_int("DIAGRAM_CLUSTER_REFINER_PROMPT_PAD_PX", 18)
    refiner_positive_points: int = _env_int("DIAGRAM_CLUSTER_REFINER_POSITIVE_POINTS", 3)
    refiner_negative_points: int = _env_int("DIAGRAM_CLUSTER_REFINER_NEGATIVE_POINTS", 4)
    refiner_candidate_bbox_gap_px: float = _env_float("DIAGRAM_CLUSTER_REFINER_CANDIDATE_BBOX_GAP_PX", 180.0)
    refiner_candidate_min_dino_cosine: float = _env_float("DIAGRAM_CLUSTER_REFINER_CANDIDATE_MIN_DINO_COSINE", 0.76)
    refiner_candidate_min_color_similarity: float = _env_float("DIAGRAM_CLUSTER_REFINER_CANDIDATE_MIN_COLOR_SIM", 0.52)
    refiner_candidate_min_shape_similarity: float = _env_float("DIAGRAM_CLUSTER_REFINER_CANDIDATE_MIN_SHAPE_SIM", 0.50)
    refiner_candidate_min_combined_score: float = _env_float("DIAGRAM_CLUSTER_REFINER_CANDIDATE_MIN_COMBINED_SCORE", 0.66)
    refiner_min_member_coverage: float = _env_float("DIAGRAM_CLUSTER_REFINER_MIN_MEMBER_COVERAGE", 0.72)
    refiner_min_coarse_coverage: float = _env_float("DIAGRAM_CLUSTER_REFINER_MIN_COARSE_COVERAGE", 0.68)
    refiner_max_extra_area_ratio: float = _env_float("DIAGRAM_CLUSTER_REFINER_MAX_EXTRA_AREA_RATIO", 0.60)
    refiner_max_annotation_overlap: float = _env_float("DIAGRAM_CLUSTER_REFINER_MAX_ANNOTATION_OVERLAP", 0.18)
    refiner_duplicate_iou_thresh: float = _env_float("DIAGRAM_CLUSTER_REFINER_DUPLICATE_IOU_THRESH", 0.72)
    refiner_duplicate_containment_thresh: float = _env_float("DIAGRAM_CLUSTER_REFINER_DUPLICATE_CONTAINMENT_THRESH", 0.92)

    # local island grouping over mini masks
    local_group_max_neighbors: int = _env_int("DIAGRAM_CLUSTER_LOCAL_GROUP_MAX_NEIGHBORS", 6)
    local_group_mutual_top_k: int = _env_int("DIAGRAM_CLUSTER_LOCAL_GROUP_MUTUAL_TOP_K", 4)
    local_group_base_gap_px: float = _env_float("DIAGRAM_CLUSTER_LOCAL_GROUP_BASE_GAP_PX", 14.0)
    local_group_gap_diag_scale: float = _env_float("DIAGRAM_CLUSTER_LOCAL_GROUP_GAP_DIAG_SCALE", 0.35)
    local_group_max_center_factor: float = _env_float("DIAGRAM_CLUSTER_LOCAL_GROUP_MAX_CENTER_FACTOR", 2.3)
    local_group_area_ratio_max: float = _env_float("DIAGRAM_CLUSTER_LOCAL_GROUP_AREA_RATIO_MAX", 2.8)
    local_group_min_dino_cosine: float = _env_float("DIAGRAM_CLUSTER_LOCAL_GROUP_MIN_DINO_COSINE", 0.955)
    local_group_min_color_similarity: float = _env_float("DIAGRAM_CLUSTER_LOCAL_GROUP_MIN_COLOR_SIM", 0.86)
    local_group_min_shape_similarity: float = _env_float("DIAGRAM_CLUSTER_LOCAL_GROUP_MIN_SHAPE_SIM", 0.80)
    local_group_min_combined_score: float = _env_float("DIAGRAM_CLUSTER_LOCAL_GROUP_MIN_COMBINED_SCORE", 0.90)
    local_group_min_shared_neighbors: int = _env_int("DIAGRAM_CLUSTER_LOCAL_GROUP_MIN_SHARED_NEIGHBORS", 1)
    local_group_border_radius_scale: float = _env_float("DIAGRAM_CLUSTER_LOCAL_GROUP_BORDER_RADIUS_SCALE", 1.6)
    local_group_max_members: int = _env_int("DIAGRAM_CLUSTER_LOCAL_GROUP_MAX_MEMBERS", 6)
    local_group_min_inner_consistency_score: float = _env_float("DIAGRAM_CLUSTER_LOCAL_GROUP_MIN_INNER_CONSISTENCY_SCORE", 0.88)
    local_group_max_score_drop: float = _env_float("DIAGRAM_CLUSTER_LOCAL_GROUP_MAX_SCORE_DROP", 0.045)

    # output
    png_compress_level: int = _env_int("DIAGRAM_CLUSTER_PNG_COMPRESS_LEVEL", 1)
    clear_existing_outputs: bool = _env_bool("DIAGRAM_CLUSTER_CLEAR_EXISTING_OUTPUTS", False)
    log_progress: bool = _env_bool("DIAGRAM_CLUSTER_LOG_PROGRESS", False)
    stroke_coverage_min: float = _env_float("DIAGRAM_CLUSTER_STROKE_COVERAGE_MIN", 0.68)
    stroke_sample_radius_px: int = _env_int("DIAGRAM_CLUSTER_STROKE_SAMPLE_RADIUS_PX", 1)
    sam_dino_render_crop_pad_px: int = _env_int("DIAGRAM_CLUSTER_SAM_DINO_RENDER_CROP_PAD_PX", 12)
    max_candidate_variants_per_island: int = _env_int("DIAGRAM_CLUSTER_MAX_CANDIDATE_VARIANTS_PER_ISLAND", 8)

    def to_public_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_config(**overrides: Any) -> DiagramMaskClusterConfig:
    data = asdict(DiagramMaskClusterConfig())
    for key, value in overrides.items():
        if value is None or key not in data:
            continue
        data[key] = value
    return DiagramMaskClusterConfig(**data)


def _glob_processed_images() -> List[Path]:
    out: List[Path] = []
    for p in PROCESSED_DIR.rglob("*.png"):
        if _PROCESSED_PLAIN_RE.match(p.name):
            out.append(p)
    out.sort(key=lambda p: str(p).lower())
    return out


def _extract_index_from_processed_name(name: str) -> Optional[int]:
    m = _PROCESSED_PLAIN_RE.match(name)
    if not m:
        return None
    return int(m.group(1))


def _resolve_processed_png_path(processed_id_or_index: Any) -> Path:
    text = str(processed_id_or_index or "").strip()
    if not text:
        raise FileNotFoundError("empty processed id")
    if text.lower().startswith("processed_"):
        want = f"{text}.png"
    elif text.isdigit():
        want = f"processed_{int(text)}.png"
    else:
        want = text
    for p in _glob_processed_images():
        if p.name.lower() == want.lower():
            return p
    raise FileNotFoundError(f"Processed image not found: {want}")


def _resolve_vector_json_path(processed_id_or_index: Any) -> Optional[Path]:
    text = str(processed_id_or_index or "").strip()
    if not text:
        return None
    if text.lower().startswith("processed_"):
        stem = text[:-5] if text.lower().endswith(".json") else text
    elif text.isdigit():
        stem = f"processed_{int(text)}"
    else:
        stem = Path(text).stem
    candidates = [
        VECTORS_DIR / f"{stem}.json",
        PROCESSED_DIR / f"{stem}.json",
        BASE_DIR / "ProcessedImages" / f"{stem}.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_rgb_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _image_to_data_uri(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(buf, format="PNG")
    import base64
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _progress_log(cfg: DiagramMaskClusterConfig, message: str) -> None:
    if not bool(getattr(cfg, "log_progress", False)):
        return
    print(f"[diagram-cluster] {message}", flush=True)


def _mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _mask_bbox(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _bbox_area_xyxy(bbox: Sequence[int]) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_union(a: Sequence[int], b: Sequence[int]) -> List[int]:
    return [
        int(min(a[0], b[0])),
        int(min(a[1], b[1])),
        int(max(a[2], b[2])),
        int(max(a[3], b[3])),
    ]


def _bbox_gap_px(a: Sequence[int], b: Sequence[int]) -> float:
    ax0, ay0, ax1, ay1 = [float(x) for x in a]
    bx0, by0, bx1, by1 = [float(x) for x in b]
    dx = max(0.0, max(ax0 - bx1, bx0 - ax1))
    dy = max(0.0, max(ay0 - by1, by0 - ay1))
    if dx == 0.0:
        return dy
    if dy == 0.0:
        return dx
    return math.hypot(dx, dy)


def _bbox_intersection_xyxy(a: Sequence[int], b: Sequence[int]) -> Optional[List[int]]:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return None
    x0 = int(max(a[0], b[0]))
    y0 = int(max(a[1], b[1]))
    x1 = int(min(a[2], b[2]))
    y1 = int(min(a[3], b[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _mask_iou(a: np.ndarray, b: np.ndarray, bbox_a: Optional[Sequence[int]] = None, bbox_b: Optional[Sequence[int]] = None) -> float:
    if bbox_a is not None and bbox_b is not None:
        shared_bbox = _bbox_union(bbox_a, bbox_b)
        x0, y0, x1, y1 = [int(v) for v in shared_bbox]
        a = np.asarray(a[y0:y1, x0:x1], dtype=bool)
        b = np.asarray(b[y0:y1, x0:x1], dtype=bool)
    else:
        a = np.asarray(a, dtype=bool)
        b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    if inter <= 0:
        return 0.0
    union = np.logical_or(a, b).sum(dtype=np.int64)
    return float(inter) / float(max(1, union))


def _mask_containment(a: np.ndarray, b: np.ndarray, bbox_a: Optional[Sequence[int]] = None, bbox_b: Optional[Sequence[int]] = None) -> float:
    if bbox_a is not None and bbox_b is not None:
        shared_bbox = _bbox_union(bbox_a, bbox_b)
        x0, y0, x1, y1 = [int(v) for v in shared_bbox]
        a = np.asarray(a[y0:y1, x0:x1], dtype=bool)
        b = np.asarray(b[y0:y1, x0:x1], dtype=bool)
    else:
        a = np.asarray(a, dtype=bool)
        b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    if inter <= 0:
        return 0.0
    aa = np.count_nonzero(a)
    bb = np.count_nonzero(b)
    return max(float(inter) / float(max(1, aa)), float(inter) / float(max(1, bb)))


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr
    return arr / norm


def _histogram_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    den = float(np.abs(aa).sum() + np.abs(bb).sum())
    if den <= 1e-8:
        return 1.0
    return max(0.0, 1.0 - float(np.abs(aa - bb).sum()) / den)


def _shape_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = ("aspect_ratio", "solidity", "fill_ratio", "eccentricity")
    diffs: List[float] = []
    for key in keys:
        av = float(a.get(key, 0.0) or 0.0)
        bv = float(b.get(key, 0.0) or 0.0)
        scale = max(1.0, abs(av), abs(bv))
        diffs.append(min(1.0, abs(av - bv) / scale))
    return max(0.0, 1.0 - (sum(diffs) / float(max(1, len(diffs)))))


def _preclean_diagram_copy(image_rgb: np.ndarray, cfg: DiagramMaskClusterConfig) -> Dict[str, np.ndarray]:
    rgb = np.asarray(image_rgb, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark_mask = (gray <= int(cfg.dark_threshold)) & (np.max(rgb, axis=2) <= int(cfg.dark_rgb_ceiling))
    dark_u8 = (dark_mask.astype(np.uint8) * 255)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_u8, connectivity=8)
    remove_mask = np.zeros_like(dark_u8)

    for label_id in range(1, int(num_labels)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if area <= 0 or w <= 0 or h <= 0:
            continue
        aspect = max(float(w) / max(1.0, float(h)), float(h) / max(1.0, float(w)))
        thickness = float(area) / float(max(w, h))
        fill_ratio = float(area) / float(max(1, w * h))
        is_dot = area <= int(cfg.dot_component_max_area)
        is_thin_line = (
            max(w, h) >= int(cfg.line_min_length_px)
            and aspect >= float(cfg.thin_component_min_aspect)
            and thickness <= float(cfg.thin_component_max_width)
            and (
                area <= int(cfg.line_component_max_area)
                or fill_ratio <= (1.0 / max(1.0, float(cfg.thin_component_min_aspect)))
            )
        )
        if is_dot or is_thin_line:
            remove_mask[labels == label_id] = 255

    lines = cv2.HoughLinesP(
        dark_u8,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(1, int(cfg.line_hough_threshold)),
        minLineLength=max(8, int(cfg.line_min_length_px)),
        maxLineGap=max(0, int(cfg.line_max_gap_px)),
    )
    if lines is not None and len(lines) > 0:
        hough_mask = np.zeros_like(dark_u8)
        thickness_px = max(1, int(cfg.line_hough_thickness_px))
        for row in lines:
            seg = np.asarray(row).reshape(-1)
            if seg.size < 4:
                continue
            x0, y0, x1, y1 = [int(v) for v in seg[:4]]
            cv2.line(hough_mask, (x0, y0), (x1, y1), 255, thickness=thickness_px, lineType=cv2.LINE_AA)
        remove_mask = np.maximum(remove_mask, cv2.bitwise_and(hough_mask, dark_u8))

    expanded_remove_mask = remove_mask
    if np.count_nonzero(remove_mask) > 0 and int(cfg.removal_mask_expand_px) > 0:
        k = 2 * int(cfg.removal_mask_expand_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        expanded_remove_mask = cv2.dilate(remove_mask, kernel, iterations=1)

    if np.count_nonzero(expanded_remove_mask) > 0:
        prefilled_bgr = bgr.copy()
        prefilled_bgr[expanded_remove_mask > 0] = 255
    else:
        prefilled_bgr = bgr

    if int(cfg.inpaint_radius) > 0 and np.count_nonzero(expanded_remove_mask) > 0:
        cleaned_bgr = cv2.inpaint(prefilled_bgr, expanded_remove_mask, int(cfg.inpaint_radius), cv2.INPAINT_TELEA)
    else:
        cleaned_bgr = prefilled_bgr.copy()
    overlay = rgb.copy()
    overlay[expanded_remove_mask > 0] = np.array([255, 64, 64], dtype=np.uint8)
    cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
    return {
        "cleaned_rgb": cleaned_rgb,
        "removed_mask": expanded_remove_mask,
        "removed_overlay_rgb": overlay,
    }


def _get_sam2_bundle_by_model(model_id: str, device_name: str) -> Dict[str, Any]:
    global _SAM2_BUNDLE
    if _SAM2_BUNDLE and _SAM2_BUNDLE.get("model_id") == model_id and str(_SAM2_BUNDLE.get("device")) == device_name:
        return _SAM2_BUNDLE
    device = torch.device(device_name)
    processor = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(model_id)
    model.to(device)
    model.eval()
    _SAM2_BUNDLE = {
        "processor": processor,
        "model": model,
        "device": device,
        "model_id": model_id,
    }
    return _SAM2_BUNDLE


def _get_sam2_bundle(cfg: DiagramMaskClusterConfig) -> Dict[str, Any]:
    return _get_sam2_bundle_by_model(cfg.sam_model_id, cfg.device)


def _get_refiner_sam2_bundle(cfg: DiagramMaskClusterConfig) -> Dict[str, Any]:
    model_id = str(cfg.refiner_model_id or cfg.sam_model_id).strip() or cfg.sam_model_id
    return _get_sam2_bundle_by_model(model_id, cfg.device)


def _get_dino_bundle(cfg: DiagramMaskClusterConfig) -> Dict[str, Any]:
    global _DINO_BUNDLE
    if _DINO_BUNDLE and _DINO_BUNDLE.get("model_id") == cfg.dino_model_id and str(_DINO_BUNDLE.get("device")) == cfg.device:
        return _DINO_BUNDLE
    device = torch.device(cfg.device)
    processor = AutoImageProcessor.from_pretrained(cfg.dino_model_id)
    model = Dinov2Model.from_pretrained(cfg.dino_model_id)
    model.to(device)
    model.eval()
    _DINO_BUNDLE = {
        "processor": processor,
        "model": model,
        "device": device,
        "model_id": cfg.dino_model_id,
    }
    return _DINO_BUNDLE


def unload_hot_models() -> None:
    global _SAM2_BUNDLE, _DINO_BUNDLE
    for bundle_name in ("_SAM2_BUNDLE", "_DINO_BUNDLE"):
        bundle = _SAM2_BUNDLE if bundle_name == "_SAM2_BUNDLE" else _DINO_BUNDLE
        if not isinstance(bundle, dict):
            continue
        model = bundle.get("model")
        processor = bundle.get("processor")
        try:
            if model is not None:
                model.to("cpu")
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        try:
            del processor
        except Exception:
            pass
    _SAM2_BUNDLE = None
    _DINO_BUNDLE = None
    try:
        gc = getattr(torch, "cuda", None)
        if gc is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _build_point_grid(width: int, height: int, step: int, max_points: int) -> List[Tuple[float, float]]:
    step = max(32, int(step))
    xs = list(range(step // 2, max(step // 2 + 1, width), step))
    ys = list(range(step // 2, max(step // 2 + 1, height), step))
    if not xs:
        xs = [max(0, width // 2)]
    if not ys:
        ys = [max(0, height // 2)]
    pts = [(float(x), float(y)) for y in ys for x in xs]
    if len(pts) > int(max_points):
        stride = max(1, int(math.ceil(len(pts) / float(max_points))))
        pts = pts[::stride]
    return pts[: int(max_points)]


def _coerce_mask_batch(mask_batch: Any) -> np.ndarray:
    arr = mask_batch
    if isinstance(arr, list):
        arr = arr[0]
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    while arr.ndim > 4:
        arr = arr[0]
    return arr


def _coerce_scores_batch(score_batch: Any) -> np.ndarray:
    arr = score_batch
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def _compute_stability_from_logits(
    processor: Sam2Processor,
    pred_masks: torch.Tensor,
    original_sizes: torch.Tensor,
    base_threshold: float,
    offset: float,
) -> np.ndarray:
    masks_a = processor.post_process_masks(
        pred_masks.detach().cpu(),
        original_sizes.detach().cpu(),
        mask_threshold=float(base_threshold),
        binarize=True,
    )
    masks_b = processor.post_process_masks(
        pred_masks.detach().cpu(),
        original_sizes.detach().cpu(),
        mask_threshold=float(base_threshold + offset),
        binarize=True,
    )
    arr_a = _coerce_mask_batch(masks_a)
    arr_b = _coerce_mask_batch(masks_b)
    if arr_a.shape != arr_b.shape:
        return np.ones(arr_a.shape[:2], dtype=np.float32)
    out = np.ones(arr_a.shape[:2], dtype=np.float32)
    for i in range(arr_a.shape[0]):
        for j in range(arr_a.shape[1]):
            ma = np.asarray(arr_a[i, j], dtype=bool)
            mb = np.asarray(arr_b[i, j], dtype=bool)
            inter = np.logical_and(ma, mb).sum(dtype=np.int64)
            union = np.logical_or(ma, mb).sum(dtype=np.int64)
            out[i, j] = float(inter) / float(max(1, union))
    return out


def _generate_sam2_candidate_masks(image_rgb: np.ndarray, cfg: DiagramMaskClusterConfig) -> List[Dict[str, Any]]:
    t0 = time.perf_counter()
    bundle = _get_sam2_bundle(cfg)
    processor: Sam2Processor = bundle["processor"]
    model: Sam2Model = bundle["model"]
    device: torch.device = bundle["device"]

    pil_image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))
    width, height = pil_image.size
    points = _build_point_grid(width, height, int(cfg.point_grid_step), int(cfg.max_points))
    if not points:
        return []
    total_batches = int(math.ceil(len(points) / float(max(1, int(cfg.point_batch_size)))))
    _progress_log(
        cfg,
        f"sam2 proposals start size={width}x{height} points={len(points)} batch_size={int(cfg.point_batch_size)} batches={total_batches}",
    )

    image_inputs = processor(images=pil_image, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device)
    with torch.inference_mode():
        image_embeddings = model.get_image_embeddings(pixel_values=pixel_values)

    proposals: List[Dict[str, Any]] = []
    batch_size = max(1, int(cfg.point_batch_size))
    for start in range(0, len(points), batch_size):
        batch_points = points[start:start + batch_size]
        batch_index = start // batch_size + 1
        input_points = [[[list(map(float, pt))] for pt in batch_points]]
        input_labels = [[[1] for _ in batch_points]]
        inputs = processor(
            images=pil_image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        )
        original_sizes = inputs["original_sizes"]
        point_tensor = inputs["input_points"].to(device)
        label_tensor = inputs["input_labels"].to(device)
        with torch.inference_mode():
            outputs = model(
                input_points=point_tensor,
                input_labels=label_tensor,
                image_embeddings=image_embeddings,
                multimask_output=True,
            )
        mask_batch = processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            original_sizes.detach().cpu(),
            mask_threshold=float(cfg.sam_mask_threshold),
            binarize=True,
        )
        mask_arr = _coerce_mask_batch(mask_batch)
        score_arr = _coerce_scores_batch(outputs.iou_scores)
        stability_arr = _compute_stability_from_logits(
            processor,
            outputs.pred_masks,
            original_sizes,
            base_threshold=float(cfg.sam_mask_threshold),
            offset=float(cfg.sam_stability_offset),
        )

        if mask_arr.ndim != 4:
            continue

        for prompt_index in range(mask_arr.shape[0]):
            prompt_point = batch_points[prompt_index] if prompt_index < len(batch_points) else None
            global_prompt_index = start + prompt_index
            prompt_key = ""
            if prompt_point is not None:
                prompt_key = f"{int(round(float(prompt_point[0])))}_{int(round(float(prompt_point[1])))}"
            for mask_index in range(mask_arr.shape[1]):
                mask = np.asarray(mask_arr[prompt_index, mask_index], dtype=bool)
                area_px = _mask_area(mask)
                if area_px < int(cfg.min_region_area_px):
                    continue
                pred_iou = float(score_arr[prompt_index, mask_index]) if score_arr.ndim == 2 else 0.0
                stability = float(stability_arr[prompt_index, mask_index]) if stability_arr.ndim == 2 else pred_iou
                if pred_iou < float(cfg.sam_pred_iou_thresh):
                    continue
                if stability < float(cfg.sam_stability_score_thresh):
                    continue
                bbox = _mask_bbox(mask)
                if not bbox:
                    continue
                proposals.append(
                    {
                        "prompt_point": list(prompt_point) if prompt_point is not None else None,
                        "prompt_key": prompt_key,
                        "prompt_index": int(global_prompt_index),
                        "prompt_mask_index": int(mask_index),
                        "proposal_id": f"p{len(proposals):04d}",
                        "mask": mask,
                        "bbox_xyxy": bbox,
                        "mask_area_px": area_px,
                        "sam_pred_iou": pred_iou,
                        "sam_stability_score": stability,
                    }
                )
        if batch_index == 1 or batch_index == total_batches or batch_index % 5 == 0:
            _progress_log(
                cfg,
                f"sam2 proposals batch={batch_index}/{total_batches} kept={len(proposals)} elapsed_s={time.perf_counter() - t0:.1f}",
            )
    _progress_log(cfg, f"sam2 proposals done kept={len(proposals)} elapsed_s={time.perf_counter() - t0:.1f}")
    return proposals


def _extract_mask_crop_rgba(image_rgb: np.ndarray, mask: np.ndarray, pad: int) -> Tuple[np.ndarray, List[int], np.ndarray]:
    bbox = _mask_bbox(mask)
    if not bbox:
        raise ValueError("mask has no bbox")
    x0, y0, x1, y1 = bbox
    h, w = mask.shape
    x0 = max(0, int(x0) - int(pad))
    y0 = max(0, int(y0) - int(pad))
    x1 = min(int(w), int(x1) + int(pad))
    y1 = min(int(h), int(y1) + int(pad))
    crop_rgb = np.asarray(image_rgb[y0:y1, x0:x1], dtype=np.uint8).copy()
    crop_mask = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
    rgba = np.zeros((crop_rgb.shape[0], crop_rgb.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = crop_rgb
    rgba[..., 3] = (crop_mask.astype(np.uint8) * 255)
    rgba[~crop_mask, :3] = 255
    return rgba, [int(x0), int(y0), int(x1), int(y1)], crop_mask


def _compute_shape_features(mask: np.ndarray, bbox: Sequence[int]) -> Dict[str, float]:
    area_px = float(max(1, _mask_area(mask)))
    bw = max(1.0, float(bbox[2]) - float(bbox[0]))
    bh = max(1.0, float(bbox[3]) - float(bbox[1]))
    aspect_ratio = bw / bh
    fill_ratio = area_px / max(1.0, bw * bh)
    mask_u8 = (np.asarray(mask, dtype=np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 1.0
    eccentricity = 0.0
    contour_complexity = 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        hull_area = float(max(1.0, cv2.contourArea(hull)))
        solidity = float(cv2.contourArea(contour)) / hull_area
        perimeter = float(max(1.0, cv2.arcLength(contour, True)))
        contour_complexity = (perimeter * perimeter) / float(max(1.0, 4.0 * math.pi * area_px))
        pts = np.column_stack(np.where(mask_u8 > 0)).astype(np.float32)
        if pts.shape[0] >= 5:
            cov = np.cov(pts.T)
            vals = np.linalg.eigvalsh(cov)
            vals = np.clip(np.sort(vals), 1e-6, None)
            eccentricity = float(math.sqrt(max(0.0, 1.0 - (vals[0] / vals[-1]))))
    return {
        "area_px": area_px,
        "aspect_ratio": aspect_ratio,
        "fill_ratio": fill_ratio,
        "solidity": float(solidity),
        "eccentricity": float(eccentricity),
        "contour_complexity": float(contour_complexity),
    }


def _compute_color_histogram(image_rgb: np.ndarray, mask: np.ndarray, bins: int) -> np.ndarray:
    bins = max(4, int(bins))
    pixels = np.asarray(image_rgb, dtype=np.uint8)[np.asarray(mask, dtype=bool)]
    if pixels.size == 0:
        return np.zeros((bins * 3,), dtype=np.float32)
    feats: List[np.ndarray] = []
    for ch in range(3):
        hist, _ = np.histogram(pixels[:, ch], bins=bins, range=(0, 256), density=True)
        feats.append(hist.astype(np.float32))
    return np.concatenate(feats, axis=0)


def _compute_contrast_color_histogram(image_rgb: np.ndarray, mask: np.ndarray, bins: int) -> np.ndarray:
    bins = max(4, int(bins))
    pixels = np.asarray(image_rgb, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    if pixels.size == 0:
        return np.zeros((bins * 3,), dtype=np.float32)
    denom = np.clip(pixels.sum(axis=1, keepdims=True), 1.0, None)
    chroma = pixels / denom
    feats: List[np.ndarray] = []
    for ch in range(3):
        hist, _ = np.histogram(chroma[:, ch], bins=bins, range=(0.0, 1.0), density=True)
        feats.append(hist.astype(np.float32))
    return np.concatenate(feats, axis=0)


def _attach_dino_and_shape_features(
    image_rgb: np.ndarray,
    proposals: List[Dict[str, Any]],
    cfg: DiagramMaskClusterConfig,
) -> List[Dict[str, Any]]:
    if not proposals:
        return []
    bundle = _get_dino_bundle(cfg)
    processor = bundle["processor"]
    model: Dinov2Model = bundle["model"]
    device: torch.device = bundle["device"]

    crops: List[Image.Image] = []
    crop_masks: List[np.ndarray] = []
    crop_bboxes: List[List[int]] = []
    for row in proposals:
        rgba, crop_bbox, crop_mask = _extract_mask_crop_rgba(image_rgb, row["mask"], int(cfg.crop_pad_px))
        rgb_pil = Image.fromarray(rgba[..., :3])
        crops.append(rgb_pil)
        crop_masks.append(crop_mask)
        crop_bboxes.append(crop_bbox)
        row["crop_bbox_xyxy"] = crop_bbox

    inputs = processor(images=crops, return_tensors="pt")
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

    for idx, row in enumerate(proposals):
        crop_bbox = crop_bboxes[idx]
        crop_mask = crop_masks[idx]
        x0, y0, x1, y1 = crop_bbox
        crop_rgb = image_rgb[y0:y1, x0:x1]
        row["dino_embedding"] = _normalize_vec(emb[idx]).tolist()
        row["color_histogram"] = _compute_color_histogram(crop_rgb, crop_mask, int(cfg.histogram_bins)).tolist()
        row["contrast_color_histogram"] = _compute_contrast_color_histogram(crop_rgb, crop_mask, int(cfg.histogram_bins)).tolist()
        row["shape_features"] = _compute_shape_features(row["mask"], row["bbox_xyxy"])
    return proposals


def _bbox_center_xyxy(bbox: Sequence[int]) -> List[float]:
    return [
        (float(bbox[0]) + float(bbox[2])) / 2.0,
        (float(bbox[1]) + float(bbox[3])) / 2.0,
    ]


def _bbox_diag_xyxy(bbox: Sequence[int]) -> float:
    return math.hypot(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))


def _area_ratio(a: float, b: float) -> float:
    aa = max(1.0, float(a))
    bb = max(1.0, float(b))
    return max(aa, bb) / min(aa, bb)


def _proposal_rank_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(row.get("sam_pred_iou", 0.0) or 0.0),
        float(row.get("sam_stability_score", 0.0) or 0.0),
        -float(row.get("mask_area_px", 0.0) or 0.0),
    )


def _annotate_basic_proposal_geometry(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for row in rows:
        bbox = row.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        row["bbox_center_xy"] = _bbox_center_xyxy(bbox)
        row["bbox_diag_px"] = float(_bbox_diag_xyxy(bbox))
    return rows


def _consolidate_prompt_level_proposals(
    proposals: List[Dict[str, Any]],
    cfg: DiagramMaskClusterConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not proposals:
        return [], []

    by_prompt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in proposals:
        prompt_key = str(row.get("prompt_key", "") or "")
        if not prompt_key:
            pt = row.get("prompt_point") if isinstance(row.get("prompt_point"), list) else [0.0, 0.0]
            prompt_key = f"{int(round(float(pt[0])))}_{int(round(float(pt[1])))}"
            row["prompt_key"] = prompt_key
        by_prompt[prompt_key].append(row)

    kept_all: List[Dict[str, Any]] = []
    prompt_debug: List[Dict[str, Any]] = []
    for prompt_key, rows in by_prompt.items():
        rows_sorted = sorted(
            [dict(row) for row in rows],
            key=lambda row: (
                float(row.get("mask_area_px", 0.0) or 0.0),
                -float(row.get("sam_pred_iou", 0.0) or 0.0),
                -float(row.get("sam_stability_score", 0.0) or 0.0),
            ),
        )
        prompt_kept: List[Dict[str, Any]] = []
        suppressed_ids: List[str] = []
        parent_links: List[Dict[str, Any]] = []

        for row in rows_sorted:
            area_px = float(row.get("mask_area_px", 0.0) or 0.0)
            duplicate_idx: Optional[int] = None
            for idx, prev in enumerate(prompt_kept):
                shared = _bbox_intersection_xyxy(row.get("bbox_xyxy"), prev.get("bbox_xyxy"))
                if shared is None:
                    continue
                iou = _mask_iou(row["mask"], prev["mask"], row["bbox_xyxy"], prev["bbox_xyxy"])
                containment = _mask_containment(row["mask"], prev["mask"], row["bbox_xyxy"], prev["bbox_xyxy"])
                area_ratio = _area_ratio(area_px, float(prev.get("mask_area_px", 0.0) or 0.0))
                if (
                    area_ratio <= float(cfg.prompt_same_scale_area_ratio_max)
                    and (iou >= float(cfg.prompt_same_scale_iou_thresh) or containment >= float(cfg.prompt_parent_containment_thresh))
                ):
                    if _proposal_rank_key(row) > _proposal_rank_key(prev):
                        suppressed_ids.append(str(prev.get("proposal_id", "") or ""))
                        prompt_kept[idx] = row
                    else:
                        suppressed_ids.append(str(row.get("proposal_id", "") or ""))
                    duplicate_idx = idx
                    break
            if duplicate_idx is not None:
                continue

            child_ids: List[str] = []
            for prev in prompt_kept:
                prev_area = float(prev.get("mask_area_px", 0.0) or 0.0)
                if prev_area >= area_px:
                    continue
                shared = _bbox_intersection_xyxy(row.get("bbox_xyxy"), prev.get("bbox_xyxy"))
                if shared is None:
                    continue
                containment = _mask_containment(row["mask"], prev["mask"], row["bbox_xyxy"], prev["bbox_xyxy"])
                if (
                    containment >= float(cfg.prompt_parent_containment_thresh)
                    and area_px <= prev_area * float(cfg.prompt_parent_max_area_ratio)
                ):
                    child_ids.append(str(prev.get("proposal_id", "") or ""))
            if child_ids:
                row["prompt_child_mask_ids"] = child_ids
                for child_id in child_ids:
                    parent_links.append({"parent_id": str(row.get("proposal_id", "") or ""), "child_id": child_id})

            prompt_kept.append(row)

        prompt_kept.sort(
            key=lambda row: (
                float(row.get("mask_area_px", 0.0) or 0.0),
                -float(row.get("sam_pred_iou", 0.0) or 0.0),
            )
        )
        if len(prompt_kept) > int(cfg.prompt_keep_max_per_point):
            prompt_kept = prompt_kept[: int(cfg.prompt_keep_max_per_point)]
        kept_all.extend(prompt_kept)
        prompt_debug.append(
            {
                "prompt_key": prompt_key,
                "raw_count": len(rows),
                "kept_count": len(prompt_kept),
                "suppressed_ids": [x for x in suppressed_ids if x],
                "parent_links": list(parent_links),
            }
        )

    kept_all = _annotate_basic_proposal_geometry(kept_all)
    return kept_all, prompt_debug


def _build_local_group_edge_candidates(
    proposals: List[Dict[str, Any]],
    cfg: DiagramMaskClusterConfig,
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    neighbors: Dict[int, List[Dict[str, Any]]] = {idx: [] for idx in range(len(proposals))}
    edge_debug: List[Dict[str, Any]] = []
    for i in range(len(proposals)):
        a = proposals[i]
        for j in range(i + 1, len(proposals)):
            b = proposals[j]
            area_ratio = _area_ratio(a.get("mask_area_px", 1.0), b.get("mask_area_px", 1.0))
            if area_ratio > float(cfg.local_group_area_ratio_max):
                continue
            metrics = _compute_pair_metrics(a, b)
            center_a = a.get("bbox_center_xy") if isinstance(a.get("bbox_center_xy"), list) else _bbox_center_xyxy(a["bbox_xyxy"])
            center_b = b.get("bbox_center_xy") if isinstance(b.get("bbox_center_xy"), list) else _bbox_center_xyxy(b["bbox_xyxy"])
            center_dist = math.hypot(float(center_a[0]) - float(center_b[0]), float(center_a[1]) - float(center_b[1]))
            local_gap = float(cfg.local_group_base_gap_px) + float(cfg.local_group_gap_diag_scale) * min(
                float(a.get("bbox_diag_px", 0.0) or 0.0),
                float(b.get("bbox_diag_px", 0.0) or 0.0),
            )
            if metrics["bbox_gap_px"] > local_gap:
                continue
            if center_dist > local_gap * float(cfg.local_group_max_center_factor):
                continue
            if metrics["dino_cosine"] < float(cfg.local_group_min_dino_cosine):
                continue
            if metrics["color_similarity"] < float(cfg.local_group_min_color_similarity):
                continue
            if metrics["shape_similarity"] < float(cfg.local_group_min_shape_similarity):
                continue
            if metrics["combined_feature_score"] < float(cfg.local_group_min_combined_score):
                continue
            edge = {
                "other": int(j),
                "bbox_gap_px": float(metrics["bbox_gap_px"]),
                "center_dist_px": float(center_dist),
                "score": float(metrics["combined_feature_score"]),
                "dino_cosine": float(metrics["dino_cosine"]),
                "color_similarity": float(metrics["color_similarity"]),
                "shape_similarity": float(metrics["shape_similarity"]),
                "area_ratio": float(area_ratio),
            }
            rev = dict(edge)
            rev["other"] = int(i)
            neighbors[i].append(edge)
            neighbors[j].append(rev)
            edge_debug.append(
                {
                    "a": str(a.get("proposal_id", "") or ""),
                    "b": str(b.get("proposal_id", "") or ""),
                    **{k: v for k, v in edge.items() if k != "other"},
                }
            )

    for idx in range(len(proposals)):
        neighbors[idx].sort(key=lambda row: (-float(row["score"]), float(row["bbox_gap_px"]), float(row["center_dist_px"])))
        neighbors[idx] = neighbors[idx][: max(1, int(cfg.local_group_max_neighbors))]
    return neighbors, edge_debug


def _build_local_island_groups(
    proposals: List[Dict[str, Any]],
    cfg: DiagramMaskClusterConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not proposals:
        return [], []

    neighbors, edge_debug = _build_local_group_edge_candidates(proposals, cfg)
    edge_lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}
    strong_sets: Dict[int, set[int]] = {}
    top_k = max(1, int(cfg.local_group_mutual_top_k))
    for idx, items in neighbors.items():
        strong_sets[idx] = {int(item["other"]) for item in items[:top_k]}
        for item in items:
            edge_lookup[(int(idx), int(item["other"]))] = item

    assigned: Dict[int, str] = {}
    groups: List[Dict[str, Any]] = []
    seed_order = sorted(
        range(len(proposals)),
        key=lambda idx: (
            float(proposals[idx].get("sam_pred_iou", 0.0) or 0.0),
            float(proposals[idx].get("sam_stability_score", 0.0) or 0.0),
            -float(proposals[idx].get("mask_area_px", 0.0) or 0.0),
        ),
        reverse=True,
    )

    for seed_idx in seed_order:
        if seed_idx in assigned:
            continue
        members: List[int] = [int(seed_idx)]
        member_set = {int(seed_idx)}
        seed_id = str(proposals[seed_idx].get("proposal_id", "") or "")
        while True:
            if len(members) >= max(1, int(cfg.local_group_max_members)):
                break
            candidate_scores: List[Tuple[int, int, float, float]] = []
            frontier = set()
            for midx in members:
                for item in neighbors.get(midx, []):
                    other = int(item["other"])
                    if other in assigned or other in member_set:
                        continue
                    frontier.add(other)
            for cand_idx in frontier:
                support_edges: List[Dict[str, Any]] = []
                early_hits = 0
                early_members = members[: min(3, len(members))]
                shared_neighbor_hits = 0
                for midx in members:
                    edge = edge_lookup.get((midx, cand_idx))
                    if edge is None:
                        continue
                    support_edges.append(edge)
                    if midx in early_members:
                        early_hits += 1
                    shared_neighbors = strong_sets.get(midx, set()).intersection(strong_sets.get(cand_idx, set()))
                    if shared_neighbors:
                        shared_neighbor_hits += 1
                if not support_edges:
                    continue
                support_edges.sort(key=lambda row: (-float(row["score"]), float(row["bbox_gap_px"])))
                required_support = 1 if len(members) == 1 else min(2, len(members))
                if len(support_edges) < required_support:
                    continue
                if len(members) > 1 and early_hits < required_support:
                    continue
                if len(members) > 1 and shared_neighbor_hits < int(cfg.local_group_min_shared_neighbors):
                    continue
                support_slice = support_edges[:required_support]
                avg_score = float(sum(float(row["score"]) for row in support_slice) / float(len(support_slice)))
                avg_gap = float(sum(float(row["bbox_gap_px"]) for row in support_slice) / float(len(support_slice)))

                if len(members) > 1:
                    inner_scores = []
                    candidate_to_all_scores = []
                    for pos, left_idx in enumerate(members):
                        for right_idx in members[pos + 1:]:
                            edge = edge_lookup.get((left_idx, right_idx)) or edge_lookup.get((right_idx, left_idx))
                            if edge is not None:
                                inner_scores.append(float(edge.get("score", 0.0) or 0.0))
                        edge_to_candidate = edge_lookup.get((left_idx, cand_idx)) or edge_lookup.get((cand_idx, left_idx))
                        if edge_to_candidate is not None:
                            candidate_to_all_scores.append(float(edge_to_candidate.get("score", 0.0) or 0.0))
                    if candidate_to_all_scores:
                        avg_all = float(sum(candidate_to_all_scores) / float(len(candidate_to_all_scores)))
                        worst_all = float(min(candidate_to_all_scores))
                        if worst_all < float(cfg.local_group_min_inner_consistency_score):
                            continue
                        if inner_scores:
                            inner_avg = float(sum(inner_scores) / float(len(inner_scores)))
                            if avg_all < inner_avg - float(cfg.local_group_max_score_drop):
                                continue
                candidate_scores.append((cand_idx, len(support_edges), avg_score, avg_gap))
            if not candidate_scores:
                break
            candidate_scores.sort(key=lambda row: (row[1], row[2], -row[3]), reverse=True)
            best_idx = int(candidate_scores[0][0])
            members.append(best_idx)
            member_set.add(best_idx)

        group_id = f"group_{len(groups):04d}"
        for midx in members:
            assigned[midx] = group_id
        border_candidates: List[Tuple[str, float]] = []
        member_mask_ids = [str(proposals[midx].get("proposal_id", "") or "") for midx in members]
        local_radius = max(
            float(cfg.local_group_base_gap_px),
            float(np.median([float(proposals[midx].get("bbox_diag_px", 0.0) or 0.0) for midx in members])) * float(cfg.local_group_border_radius_scale),
        )
        for other_idx, row in enumerate(proposals):
            if other_idx in member_set:
                continue
            best_gap: Optional[float] = None
            best_score: float = 0.0
            for midx in members:
                metrics = _compute_pair_metrics(proposals[midx], row)
                if metrics["bbox_gap_px"] <= local_radius:
                    if best_gap is None or float(metrics["bbox_gap_px"]) < best_gap:
                        best_gap = float(metrics["bbox_gap_px"])
                    best_score = max(best_score, float(metrics["combined_feature_score"]))
            if best_gap is None:
                continue
            if best_score >= float(cfg.local_group_min_combined_score):
                continue
            border_candidates.append((str(row.get("proposal_id", "") or ""), float(best_gap)))
        border_candidates.sort(key=lambda row: row[1])
        groups.append(
            {
                "group_id": group_id,
                "seed_mask_id": seed_id,
                "member_mask_ids": member_mask_ids,
                "member_count": len(member_mask_ids),
                "border_mask_ids": [mask_id for mask_id, _gap in border_candidates[:12] if mask_id],
                "group_quality": float(np.mean([
                    float(proposals[midx].get("sam_pred_iou", 0.0) or 0.0)
                    for midx in members
                ])) if members else 0.0,
            }
        )

    for idx, row in enumerate(proposals):
        row["group_id"] = assigned.get(idx)
    return groups, edge_debug


def _mask_centroid(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def _expand_bbox_xyxy(
    bbox: Sequence[int],
    pad: int,
    width: int,
    height: int,
) -> List[int]:
    if not bbox or len(bbox) != 4:
        return [0, 0, int(width), int(height)]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    return [
        max(0, x0 - int(pad)),
        max(0, y0 - int(pad)),
        min(int(width), x1 + int(pad)),
        min(int(height), y1 + int(pad)),
    ]


def _eval_cubic(p0: Tuple[float, float], c1: Tuple[float, float], c2: Tuple[float, float], p1: Tuple[float, float], t: float) -> Tuple[float, float]:
    mt = 1.0 - float(t)
    mt2 = mt * mt
    t2 = float(t) * float(t)
    x = (mt2 * mt) * p0[0] + 3.0 * (mt2 * float(t)) * c1[0] + 3.0 * (mt * t2) * c2[0] + (t2 * float(t)) * p1[0]
    y = (mt2 * mt) * p0[1] + 3.0 * (mt2 * float(t)) * c1[1] + 3.0 * (mt * t2) * c2[1] + (t2 * float(t)) * p1[1]
    return x, y


def _stroke_samples_json_from_beziers(stroke: Dict[str, Any]) -> List[Tuple[float, float]]:
    segs = stroke.get("segments") or []
    if not isinstance(segs, list) or not segs:
        return []
    pts: List[Tuple[float, float]] = []
    last: Optional[Tuple[float, float]] = None
    for seg in segs:
        if not isinstance(seg, list) or len(seg) < 8:
            continue
        p0 = (float(seg[0]), float(seg[1]))
        c1 = (float(seg[2]), float(seg[3]))
        c2 = (float(seg[4]), float(seg[5]))
        p1 = (float(seg[6]), float(seg[7]))
        for i in range(19):
            x, y = _eval_cubic(p0, c1, c2, p1, i / 18.0)
            if last is not None and (abs(x - last[0]) + abs(y - last[1]) < 0.05):
                continue
            pts.append((x, y))
            last = (x, y)
    return pts


def _stroke_samples_json_from_polyline(stroke: Dict[str, Any]) -> List[Tuple[float, float]]:
    pts = stroke.get("points") or []
    if not isinstance(pts, list) or len(pts) < 2:
        return []
    out: List[Tuple[float, float]] = []
    for p in pts:
        if isinstance(p, list) and len(p) >= 2:
            out.append((float(p[0]), float(p[1])))
    if len(out) < 2:
        return []
    step = max(1, len(out) // 120)
    return out[::step]


def _load_vector_stroke_samples(
    processed_id: str,
    *,
    image_width: int,
    image_height: int,
) -> Tuple[List[Optional[np.ndarray]], Optional[str], Dict[str, Any]]:
    vector_path = _resolve_vector_json_path(processed_id)
    if vector_path is None:
        return [], None, {"error": f"vector json not found for {processed_id}"}
    try:
        data = json.loads(vector_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], str(vector_path), {"error": f"{type(exc).__name__}: {exc}"}

    strokes = data.get("strokes") if isinstance(data, dict) else []
    if not isinstance(strokes, list):
        return [], str(vector_path), {"error": "vector json has no strokes list"}
    fmt = str((data or {}).get("vector_format") or "bezier_cubic").lower()
    src_w = float((data or {}).get("width") or image_width or 1)
    src_h = float((data or {}).get("height") or image_height or 1)
    scale_x = float(image_width) / src_w if src_w > 0 else 1.0
    scale_y = float(image_height) / src_h if src_h > 0 else 1.0

    samples: List[Optional[np.ndarray]] = []
    for stroke in strokes:
        if not isinstance(stroke, dict):
            samples.append(None)
            continue
        if fmt == "bezier_cubic":
            raw_pts = _stroke_samples_json_from_beziers(stroke)
        else:
            raw_pts = _stroke_samples_json_from_polyline(stroke)
        if len(raw_pts) < 2:
            samples.append(None)
            continue
        pts = np.asarray(
            [(int(round(x * scale_x)), int(round(y * scale_y))) for x, y in raw_pts],
            dtype=np.int32,
        )
        if pts.shape[0] >= 2:
            d = np.abs(np.diff(pts, axis=0)).sum(axis=1)
            keep = np.concatenate(([True], d > 0))
            pts = pts[keep]
        samples.append(pts if pts.shape[0] >= 2 else None)
    return samples, str(vector_path), {"stroke_count": len(strokes), "vector_format": fmt}


def _mask_with_sample_radius(mask: np.ndarray, radius_px: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    radius = max(0, int(radius_px))
    if radius <= 0:
        return mask_bool
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1) > 0


def _stroke_indexes_inside_mask(
    union_mask: np.ndarray,
    stroke_samples: Sequence[Optional[np.ndarray]],
    *,
    coverage_min: float,
    sample_radius_px: int,
) -> Tuple[List[int], Dict[str, float]]:
    if not stroke_samples:
        return [], {}
    h, w = union_mask.shape[:2]
    test_mask = _mask_with_sample_radius(union_mask, int(sample_radius_px))
    threshold = float(coverage_min)
    stroke_indexes: List[int] = []
    coverage_by_stroke: Dict[str, float] = {}
    for idx, pts in enumerate(stroke_samples):
        if pts is None or pts.size <= 0:
            continue
        xs = np.clip(pts[:, 0].astype(np.int32), 0, max(0, w - 1))
        ys = np.clip(pts[:, 1].astype(np.int32), 0, max(0, h - 1))
        coverage = float(np.count_nonzero(test_mask[ys, xs])) / float(max(1, len(xs)))
        if coverage >= threshold:
            stroke_indexes.append(int(idx))
            coverage_by_stroke[str(int(idx))] = round(float(coverage), 4)
    return stroke_indexes, coverage_by_stroke


def _union_masks_for_ids(mask_ids: Sequence[str], proposals_by_id: Dict[str, Dict[str, Any]], shape: Tuple[int, int]) -> np.ndarray:
    union_mask = np.zeros(shape, dtype=bool)
    for mask_id in mask_ids:
        row = proposals_by_id.get(str(mask_id))
        if not isinstance(row, dict):
            continue
        union_mask |= np.asarray(row.get("mask"), dtype=bool)
    return union_mask


def _build_mask_candidate_specs(
    proposals: List[Dict[str, Any]],
    island_groups: List[Dict[str, Any]],
    cfg: DiagramMaskClusterConfig,
) -> List[Dict[str, Any]]:
    proposals_by_id = {str(row.get("proposal_id", "") or ""): row for row in proposals if str(row.get("proposal_id", "") or "")}
    if not proposals_by_id:
        return []
    specs: List[Dict[str, Any]] = []
    seen: set[Tuple[str, ...]] = set()

    def push(group_id: str, source_kind: str, mask_ids: Sequence[str], quality: float) -> None:
        clean_ids = [str(mask_id) for mask_id in mask_ids if str(mask_id) in proposals_by_id]
        key = tuple(sorted(set(clean_ids)))
        if not key or key in seen:
            return
        seen.add(key)
        specs.append(
            {
                "source_group_id": str(group_id or ""),
                "source_kind": str(source_kind or "island"),
                "mask_ids": list(key),
                "group_quality": float(quality),
            }
        )

    max_variants = max(1, int(cfg.max_candidate_variants_per_island))
    for group in island_groups:
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("group_id", "") or f"group_{len(specs):04d}")
        quality = float(group.get("group_quality", 0.0) or 0.0)
        member_ids = [str(x) for x in (group.get("member_mask_ids") or []) if str(x) in proposals_by_id]
        border_ids = [str(x) for x in (group.get("border_mask_ids") or []) if str(x) in proposals_by_id]
        before = len(specs)
        push(group_id, "island", member_ids, quality)

        # Ambiguous junctions get alternative local slices, so downstream matching can pick
        # the right object even when the big island is too fused or too coarse.
        if len(member_ids) > 1:
            seed_id = str(group.get("seed_mask_id", "") or member_ids[0])
            for mask_id in member_ids:
                if len(specs) - before >= max_variants:
                    break
                push(group_id, "single_mask_variant", [mask_id], quality)
            for mask_id in member_ids:
                if len(specs) - before >= max_variants:
                    break
                if mask_id != seed_id:
                    push(group_id, "seed_pair_variant", [seed_id, mask_id], quality)
        for mask_id in border_ids[:3]:
            if len(specs) - before >= max_variants:
                break
            push(group_id, "island_plus_boundary_variant", member_ids + [mask_id], quality)

    if not specs:
        for row in proposals:
            mask_id = str(row.get("proposal_id", "") or "")
            if mask_id:
                push("ungrouped", "single_mask_fallback", [mask_id], float(row.get("sam_pred_iou", 0.0) or 0.0))
    for idx, spec in enumerate(specs):
        spec["candidate_id"] = f"sam_dino_{idx:04d}"
    return specs


def _save_sam_dino_candidate_render(
    *,
    image_rgb: np.ndarray,
    union_mask: np.ndarray,
    processed_id: str,
    candidate_id: str,
    cfg: DiagramMaskClusterConfig,
) -> Tuple[str, str, List[int]]:
    bbox = _mask_bbox(union_mask)
    if not bbox:
        return "", "", [0, 0, 0, 0]
    h, w = union_mask.shape[:2]
    x0, y0, x1, y1 = _expand_bbox_xyxy(bbox, int(cfg.sam_dino_render_crop_pad_px), w, h)
    crop_rgb = np.asarray(image_rgb[y0:y1, x0:x1], dtype=np.uint8)
    crop_mask = np.asarray(union_mask[y0:y1, x0:x1], dtype=bool)
    gray = cv2.cvtColor(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    out = gray.copy()
    out[crop_mask] = crop_rgb[crop_mask]
    render_dir = SAM_DINO_RENDER_DIR / str(processed_id)
    render_dir.mkdir(parents=True, exist_ok=True)
    render_name = f"{candidate_id}.png"
    render_path = render_dir / render_name
    Image.fromarray(out).save(render_path, compress_level=int(cfg.png_compress_level))
    return render_name, str(render_path), [int(x0), int(y0), int(x1), int(y1)]


def _build_stroke_candidate_outputs(
    *,
    image_rgb: np.ndarray,
    processed_id: str,
    proposals: List[Dict[str, Any]],
    island_groups: List[Dict[str, Any]],
    cfg: DiagramMaskClusterConfig,
) -> Dict[str, Any]:
    h, w = image_rgb.shape[:2]
    stroke_samples, vector_path, vector_debug = _load_vector_stroke_samples(processed_id, image_width=w, image_height=h)
    proposals_by_id = {str(row.get("proposal_id", "") or ""): row for row in proposals if str(row.get("proposal_id", "") or "")}
    specs = _build_mask_candidate_specs(proposals, island_groups, cfg)
    candidates: List[Dict[str, Any]] = []
    seen_stroke_sets: set[Tuple[int, ...]] = set()
    for spec in specs:
        union_mask = _union_masks_for_ids(spec.get("mask_ids") or [], proposals_by_id, (h, w))
        if _mask_area(union_mask) <= 0:
            continue
        stroke_indexes, coverage = _stroke_indexes_inside_mask(
            union_mask,
            stroke_samples,
            coverage_min=float(cfg.stroke_coverage_min),
            sample_radius_px=int(cfg.stroke_sample_radius_px),
        )
        stroke_key = tuple(stroke_indexes)
        if not stroke_key:
            continue
        if stroke_key in seen_stroke_sets and str(spec.get("source_kind", "") or "") != "island":
            continue
        seen_stroke_sets.add(stroke_key)
        render_name, render_path, bbox = _save_sam_dino_candidate_render(
            image_rgb=image_rgb,
            union_mask=union_mask,
            processed_id=processed_id,
            candidate_id=str(spec.get("candidate_id") or f"sam_dino_{len(candidates):04d}"),
            cfg=cfg,
        )
        candidates.append(
            {
                "candidate_id": str(spec.get("candidate_id") or f"sam_dino_{len(candidates):04d}"),
                "processed_id": str(processed_id),
                "source": "sam_dino_mask_group",
                "source_kind": str(spec.get("source_kind", "") or "island"),
                "source_group_id": str(spec.get("source_group_id", "") or ""),
                "mask_ids": [str(x) for x in (spec.get("mask_ids") or [])],
                "stroke_indexes": [int(x) for x in stroke_indexes],
                "stroke_coverage": coverage,
                "bbox_xyxy": bbox,
                "render_file": render_name,
                "render_path": render_path,
                "group_quality": float(spec.get("group_quality", 0.0) or 0.0),
                "mask_area_px": int(_mask_area(union_mask)),
            }
        )
    candidates.sort(
        key=lambda row: (
            len(row.get("stroke_indexes") or []),
            float(row.get("group_quality", 0.0) or 0.0),
            int(row.get("mask_area_px", 0) or 0),
        ),
        reverse=True,
    )
    return {
        "processed_id": str(processed_id),
        "vector_path": vector_path or "",
        "vector_debug": vector_debug,
        "coverage_min": float(cfg.stroke_coverage_min),
        "sample_radius_px": int(cfg.stroke_sample_radius_px),
        "render_dir": str(SAM_DINO_RENDER_DIR / str(processed_id)),
        "candidates": candidates,
    }


def _extract_distance_peak_points(mask: np.ndarray, count: int) -> List[List[float]]:
    work = np.asarray(mask, dtype=np.uint8)
    if work.ndim != 2 or np.count_nonzero(work) <= 0:
        return []
    dist = cv2.distanceTransform(work, cv2.DIST_L2, 5)
    points: List[List[float]] = []
    suppress_radius = 6
    for _ in range(max(0, int(count))):
        idx = np.unravel_index(int(np.argmax(dist)), dist.shape)
        y, x = int(idx[0]), int(idx[1])
        if float(dist[y, x]) <= 0.0:
            break
        points.append([float(x), float(y)])
        y0 = max(0, y - suppress_radius)
        y1 = min(dist.shape[0], y + suppress_radius + 1)
        x0 = max(0, x - suppress_radius)
        x1 = min(dist.shape[1], x + suppress_radius + 1)
        dist[y0:y1, x0:x1] = 0.0
    return points


def _extract_negative_ring_points(mask: np.ndarray, count: int) -> List[List[float]]:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2 or np.count_nonzero(mask_bool) <= 0:
        return []
    h, w = mask_bool.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(mask_bool.astype(np.uint8) * 255, kernel, iterations=2) > 0
    ring = np.logical_and(dilated, ~mask_bool)
    ys, xs = np.where(ring)
    if xs.size == 0 or ys.size == 0:
        return []
    picks: List[List[float]] = []
    sample_indexes = np.linspace(0, xs.size - 1, num=min(int(count), int(xs.size)), dtype=int)
    for idx in sample_indexes.tolist():
        picks.append([float(xs[idx]), float(ys[idx])])
    # Keep points deterministic and spatially spread.
    deduped: List[List[float]] = []
    seen = set()
    for x, y in picks:
        key = (int(round(x / 4.0)), int(round(y / 4.0)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append([float(x), float(y)])
    if deduped:
        return deduped[: int(count)]
    return [[0.0, 0.0], [float(max(0, w - 1)), 0.0], [0.0, float(max(0, h - 1))]][: int(count)]


def _compute_pair_metrics(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    bbox_gap = _bbox_gap_px(a["bbox_xyxy"], b["bbox_xyxy"])
    shared_bbox = _bbox_union(a["bbox_xyxy"], b["bbox_xyxy"])
    smaller_bbox_area = min(_bbox_area_xyxy(a["bbox_xyxy"]), _bbox_area_xyxy(b["bbox_xyxy"]))
    shared_growth = (
        max(0.0, _bbox_area_xyxy(shared_bbox) - smaller_bbox_area) / float(max(1.0, smaller_bbox_area))
        if smaller_bbox_area > 0
        else 999.0
    )
    if _bbox_intersection_xyxy(a["bbox_xyxy"], b["bbox_xyxy"]) is None:
        iou = 0.0
        containment = 0.0
    else:
        iou = _mask_iou(a["mask"], b["mask"], a["bbox_xyxy"], b["bbox_xyxy"])
        containment = _mask_containment(a["mask"], b["mask"], a["bbox_xyxy"], b["bbox_xyxy"])
    dino_cos = 1.0 - float(cosine(_normalize_vec(a.get("dino_embedding", [])), _normalize_vec(b.get("dino_embedding", []))))
    if not math.isfinite(dino_cos):
        dino_cos = 0.0
    rgb_color_sim = _histogram_similarity(a["color_histogram"], b["color_histogram"])
    contrast_color_sim = _histogram_similarity(
        a.get("contrast_color_histogram", a["color_histogram"]),
        b.get("contrast_color_histogram", b["color_histogram"]),
    )
    color_sim = max(rgb_color_sim, contrast_color_sim)
    shape_sim = _shape_similarity(a["shape_features"], b["shape_features"])
    combined_feature_score = (
        0.46 * dino_cos
        + 0.24 * color_sim
        + 0.30 * shape_sim
    )
    return {
        "iou": iou,
        "containment": containment,
        "bbox_gap_px": bbox_gap,
        "bbox_growth": shared_growth,
        "dino_cosine": dino_cos,
        "rgb_color_similarity": rgb_color_sim,
        "contrast_color_similarity": contrast_color_sim,
        "color_similarity": color_sim,
        "shape_similarity": shape_sim,
        "combined_feature_score": combined_feature_score,
    }


def _cluster_quality_score(row: Dict[str, Any]) -> float:
    sam_pred = float(row.get("sam_pred_iou", 0.0) or 0.0)
    stability = float(row.get("sam_stability_score", 0.0) or 0.0)
    refine_score = float(row.get("refine_score", 0.0) or 0.0)
    area_term = min(1.0, math.log1p(float(row.get("mask_area_px", 0.0) or 0.0)) / 10.0)
    return (
        0.32 * sam_pred
        + 0.28 * stability
        + 0.25 * refine_score
        + 0.15 * area_term
    )


def _build_cluster_row_from_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    source_rows: Sequence[Dict[str, Any]],
    cfg: DiagramMaskClusterConfig,
    *,
    proposal_id: str,
    feature_cluster_id: str,
    refine_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    union_mask = np.asarray(mask, dtype=bool)
    bbox = _mask_bbox(union_mask)
    if not bbox:
        return None
    rgba, crop_bbox, crop_mask = _extract_mask_crop_rgba(image_rgb, union_mask, int(cfg.crop_pad_px))
    source_rows = list(source_rows)
    best = max(source_rows, key=_cluster_quality_score) if source_rows else {}
    stroke_indexes = sorted(
        {
            int(v)
            for row in source_rows
            for v in (row.get("stroke_indexes") or [])
            if isinstance(v, (int, np.integer))
        }
    )
    merged_from_mask_ids = sorted(
        {
            str(v)
            for row in source_rows
            for v in (row.get("merged_from_mask_ids") or [row.get("proposal_id")])
            if str(v or "").strip()
        }
    )
    refined_from_cluster_ids = sorted(
        {
            str(row.get("feature_cluster_id", "") or row.get("proposal_id", "") or "").strip()
            for row in source_rows
            if str(row.get("feature_cluster_id", "") or row.get("proposal_id", "") or "").strip()
        }
    )
    crop_rgb = image_rgb[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]
    row = {
        "proposal_id": proposal_id,
        "pipeline": "sam2_dinov2",
        "mask": union_mask,
        "crop_rgba": rgba,
        "crop_bbox_xyxy": crop_bbox,
        "bbox_xyxy": bbox,
        "mask_area_px": _mask_area(union_mask),
        "sam_pred_iou": float(best.get("sam_pred_iou", 0.0) or 0.0),
        "sam_stability_score": float(best.get("sam_stability_score", 0.0) or 0.0),
        "merged_from_mask_ids": merged_from_mask_ids,
        "merge_count": len(merged_from_mask_ids),
        "feature_cluster_id": feature_cluster_id,
        "color_id": 0,
        "color_name": "mask",
        "group_index_in_color": 0,
        "stroke_indexes": stroke_indexes,
        "dino_embedding": list(best.get("dino_embedding", []) or []),
        "shape_features": _compute_shape_features(union_mask, bbox),
        "color_histogram": _compute_color_histogram(crop_rgb, crop_mask, int(cfg.histogram_bins)).tolist(),
        "contrast_color_histogram": _compute_contrast_color_histogram(crop_rgb, crop_mask, int(cfg.histogram_bins)).tolist(),
        "refined_from_cluster_ids": refined_from_cluster_ids,
    }
    if refine_meta:
        row.update(refine_meta)
    return row


def _suppress_redundant_proposals(
    proposals: List[Dict[str, Any]],
    removed_mask: np.ndarray,
    cfg: DiagramMaskClusterConfig,
) -> List[Dict[str, Any]]:
    work = sorted(
        list(proposals),
        key=lambda row: (
            float(row.get("sam_pred_iou", 0.0) or 0.0),
            float(row.get("sam_stability_score", 0.0) or 0.0),
            float(row.get("mask_area_px", 0.0) or 0.0),
        ),
        reverse=True,
    )
    kept: List[Dict[str, Any]] = []
    ann_mask = np.asarray(removed_mask > 0, dtype=bool)
    for row in work:
        mask = np.asarray(row.get("mask"), dtype=bool)
        area_px = _mask_area(mask)
        if area_px < int(cfg.min_region_area_px):
            continue
        annotation_overlap = float(np.logical_and(mask, ann_mask).sum(dtype=np.int64)) / float(max(1, area_px))
        row["annotation_overlap_ratio"] = annotation_overlap
        if annotation_overlap > float(cfg.annotation_overlap_drop_ratio):
            continue
        drop = False
        for prev in kept:
            bbox = row.get("bbox_xyxy")
            prev_bbox = prev.get("bbox_xyxy")
            if _bbox_intersection_xyxy(bbox, prev_bbox) is None:
                iou = 0.0
                containment = 0.0
            else:
                iou = _mask_iou(mask, prev["mask"], bbox, prev_bbox)
                containment = _mask_containment(mask, prev["mask"], bbox, prev_bbox)
            if iou >= float(cfg.proposal_nms_iou_thresh) or containment >= float(cfg.proposal_containment_thresh):
                drop = True
                break
        if not drop:
            kept.append(row)
    return kept


def _should_merge_pair(a: Dict[str, Any], b: Dict[str, Any], cfg: DiagramMaskClusterConfig) -> Tuple[bool, Dict[str, float]]:
    metrics = _compute_pair_metrics(a, b)

    spatial_ok = (
        metrics["iou"] >= float(cfg.merge_iou_thresh)
        or (
            metrics["containment"] >= float(cfg.merge_containment_thresh)
            and metrics["bbox_growth"] <= float(cfg.merge_bbox_growth_max)
        )
        or (metrics["bbox_gap_px"] <= float(cfg.merge_bbox_gap_px) and metrics["bbox_growth"] <= float(cfg.merge_bbox_growth_max))
    )
    strict_feature_ok = (
        metrics["dino_cosine"] >= float(cfg.merge_min_dino_cosine)
        and metrics["rgb_color_similarity"] >= float(cfg.merge_min_color_similarity)
        and metrics["shape_similarity"] >= float(cfg.merge_min_shape_similarity)
    )
    soft_feature_ok = (
        metrics["dino_cosine"] >= float(cfg.merge_soft_min_dino_cosine)
        and metrics["shape_similarity"] >= float(cfg.merge_soft_min_shape_similarity)
        and metrics["contrast_color_similarity"] >= float(cfg.merge_min_contrast_color_similarity)
        and metrics["combined_feature_score"] >= float(cfg.merge_combined_feature_score)
    )
    # Similar-looking islands were previously allowed to chain-merge across the
    # diagram too easily. Keep this path much tighter than the main spatial merge.
    similarity_only_ok = (
        metrics["dino_cosine"] >= float(cfg.merge_similarity_only_min_dino_cosine)
        and metrics["color_similarity"] >= float(cfg.merge_similarity_only_min_color_similarity)
        and metrics["shape_similarity"] >= float(cfg.merge_similarity_only_min_shape_similarity)
        and metrics["combined_feature_score"] >= float(cfg.merge_similarity_only_min_score)
        and metrics["bbox_gap_px"] <= float(cfg.merge_similarity_only_bbox_gap_px)
        and metrics["bbox_growth"] <= float(cfg.merge_similarity_only_bbox_growth_max)
    )
    feature_ok = strict_feature_ok or soft_feature_ok
    merge_ok = (spatial_ok and feature_ok) or similarity_only_ok
    return bool(merge_ok), {
        **metrics,
        "strict_feature_ok": float(1.0 if strict_feature_ok else 0.0),
        "soft_feature_ok": float(1.0 if soft_feature_ok else 0.0),
        "similarity_only_ok": float(1.0 if similarity_only_ok else 0.0),
    }


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _merge_component_proposals(
    image_rgb: np.ndarray,
    proposals: List[Dict[str, Any]],
    cfg: DiagramMaskClusterConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not proposals:
        return [], []
    t0 = time.perf_counter()
    uf = _UnionFind(len(proposals))
    pair_debug: List[Dict[str, Any]] = []
    total_pairs = (len(proposals) * max(0, len(proposals) - 1)) // 2
    checked_pairs = 0
    _progress_log(cfg, f"merge start proposals={len(proposals)} pairs={total_pairs}")
    for i in range(len(proposals)):
        for j in range(i + 1, len(proposals)):
            ok, metrics = _should_merge_pair(proposals[i], proposals[j], cfg)
            checked_pairs += 1
            if ok:
                uf.union(i, j)
                pair_debug.append({"a": proposals[i]["proposal_id"], "b": proposals[j]["proposal_id"], **metrics})
        if (i + 1) == len(proposals) or (i + 1) % 25 == 0:
            _progress_log(
                cfg,
                f"merge progress anchors={i + 1}/{len(proposals)} checked_pairs={checked_pairs} matched_pairs={len(pair_debug)} elapsed_s={time.perf_counter() - t0:.1f}",
            )
    groups: Dict[int, List[int]] = {}
    for idx in range(len(proposals)):
        groups.setdefault(uf.find(idx), []).append(idx)

    merged: List[Dict[str, Any]] = []
    for merged_index, indexes in enumerate(groups.values()):
        masks = [np.asarray(proposals[i]["mask"], dtype=bool) for i in indexes]
        union_mask = np.logical_or.reduce(masks)
        bbox = _mask_bbox(union_mask)
        if not bbox:
            continue
        rgba, crop_bbox, crop_mask = _extract_mask_crop_rgba(image_rgb, union_mask, int(cfg.crop_pad_px))
        best = max(
            (proposals[i] for i in indexes),
            key=lambda row: (
                float(row.get("sam_pred_iou", 0.0) or 0.0),
                float(row.get("sam_stability_score", 0.0) or 0.0),
                float(row.get("mask_area_px", 0.0) or 0.0),
            ),
        )
        merged.append(
            {
                "proposal_id": f"m{merged_index:04d}",
                "pipeline": "sam2_dinov2",
                "mask": union_mask,
                "crop_rgba": rgba,
                "crop_bbox_xyxy": crop_bbox,
                "bbox_xyxy": bbox,
                "mask_area_px": _mask_area(union_mask),
                "sam_pred_iou": float(best.get("sam_pred_iou", 0.0) or 0.0),
                "sam_stability_score": float(best.get("sam_stability_score", 0.0) or 0.0),
                "merged_from_mask_ids": [str(proposals[i]["proposal_id"]) for i in indexes],
                "merge_count": len(indexes),
                "feature_cluster_id": f"fc_{merged_index:04d}",
                "color_id": 0,
                "color_name": "mask",
                "group_index_in_color": merged_index,
                "stroke_indexes": [],
                "dino_embedding": list(best.get("dino_embedding", []) or []),
                "shape_features": _compute_shape_features(union_mask, bbox),
                "color_histogram": _compute_color_histogram(image_rgb[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]], crop_mask, int(cfg.histogram_bins)).tolist(),
                "contrast_color_histogram": _compute_contrast_color_histogram(image_rgb[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]], crop_mask, int(cfg.histogram_bins)).tolist(),
            }
        )
    merged.sort(key=lambda row: (int(row["bbox_xyxy"][0]), int(row["bbox_xyxy"][1])))
    _progress_log(
        cfg,
        f"merge done coarse_clusters={len(merged)} matched_pairs={len(pair_debug)} elapsed_s={time.perf_counter() - t0:.1f}",
    )
    return merged, pair_debug


def _build_refiner_prompt_sets(
    coarse_mask: np.ndarray,
    member_rows: Sequence[Dict[str, Any]],
    cfg: DiagramMaskClusterConfig,
) -> List[Tuple[List[List[float]], List[int]]]:
    def _dedupe_points(points: Sequence[List[float]], labels: Sequence[int]) -> Tuple[List[List[float]], List[int]]:
        out_points: List[List[float]] = []
        out_labels: List[int] = []
        seen = set()
        for pt, lab in zip(points, labels):
            key = (int(round(float(pt[0]))), int(round(float(pt[1]))), int(lab))
            if key in seen:
                continue
            seen.add(key)
            out_points.append([float(pt[0]), float(pt[1])])
            out_labels.append(int(lab))
        return out_points, out_labels

    positive_points: List[List[float]] = []
    for row in member_rows:
        centroid = _mask_centroid(row.get("mask"))
        if centroid is not None:
            positive_points.append(centroid)
    positive_points.extend(_extract_distance_peak_points(coarse_mask, int(cfg.refiner_positive_points)))
    negative_points = _extract_negative_ring_points(coarse_mask, int(cfg.refiner_negative_points))

    prompt_sets: List[Tuple[List[List[float]], List[int]]] = []
    if positive_points:
        pts, labs = _dedupe_points(positive_points[: max(1, int(cfg.refiner_positive_points))], [1] * min(len(positive_points), max(1, int(cfg.refiner_positive_points))))
        if pts:
            prompt_sets.append((pts, labs))
    peak_only = _extract_distance_peak_points(coarse_mask, max(1, int(cfg.refiner_positive_points)))
    if peak_only:
        pts, labs = _dedupe_points(peak_only, [1] * len(peak_only))
        if pts:
            prompt_sets.append((pts, labs))
    combo_points = list(peak_only or positive_points[: max(1, int(cfg.refiner_positive_points))]) + list(negative_points)
    combo_labels = [1] * len((peak_only or positive_points[: max(1, int(cfg.refiner_positive_points))])) + [-1] * len(negative_points)
    if combo_points:
        pts, labs = _dedupe_points(combo_points, combo_labels)
        if pts:
            prompt_sets.append((pts, labs))
    if not prompt_sets and positive_points:
        prompt_sets.append(([positive_points[0]], [1]))
    return prompt_sets


def _run_sam2_prompt_refinement(
    image_rgb: np.ndarray,
    prompt_sets: Sequence[Tuple[List[List[float]], List[int]]],
    cfg: DiagramMaskClusterConfig,
) -> List[Dict[str, Any]]:
    if not prompt_sets:
        return []
    bundle = _get_refiner_sam2_bundle(cfg)
    processor: Sam2Processor = bundle["processor"]
    model: Sam2Model = bundle["model"]
    device: torch.device = bundle["device"]

    pil_image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))
    image_inputs = processor(images=pil_image, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device)
    with torch.inference_mode():
        image_embeddings = model.get_image_embeddings(pixel_values=pixel_values)

    out: List[Dict[str, Any]] = []
    for prompt_points, prompt_labels in prompt_sets:
        if not prompt_points or not prompt_labels or len(prompt_points) != len(prompt_labels):
            continue
        inputs = processor(
            images=pil_image,
            input_points=[[[list(map(float, pt)) for pt in prompt_points]]],
            input_labels=[[[int(v) for v in prompt_labels]]],
            return_tensors="pt",
        )
        original_sizes = inputs["original_sizes"]
        point_tensor = inputs["input_points"].to(device)
        label_tensor = inputs["input_labels"].to(device)
        with torch.inference_mode():
            outputs = model(
                input_points=point_tensor,
                input_labels=label_tensor,
                image_embeddings=image_embeddings,
                multimask_output=True,
            )
        mask_batch = processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            original_sizes.detach().cpu(),
            mask_threshold=float(cfg.sam_mask_threshold),
            binarize=True,
        )
        mask_arr = _coerce_mask_batch(mask_batch)
        score_arr = _coerce_scores_batch(outputs.iou_scores)
        stability_arr = _compute_stability_from_logits(
            processor,
            outputs.pred_masks,
            original_sizes,
            base_threshold=float(cfg.sam_mask_threshold),
            offset=float(cfg.sam_stability_offset),
        )
        if mask_arr.ndim != 4:
            continue
        for prompt_index in range(mask_arr.shape[0]):
            for mask_index in range(mask_arr.shape[1]):
                out.append(
                    {
                        "mask": np.asarray(mask_arr[prompt_index, mask_index], dtype=bool),
                        "sam_pred_iou": float(score_arr[prompt_index, mask_index]) if score_arr.ndim == 2 else 0.0,
                        "sam_stability_score": float(stability_arr[prompt_index, mask_index]) if stability_arr.ndim == 2 else 0.0,
                        "prompt_points": [list(map(float, pt)) for pt in prompt_points],
                        "prompt_labels": [int(v) for v in prompt_labels],
                    }
                )
    return out


def _score_refiner_candidate(
    candidate_mask: np.ndarray,
    coarse_mask: np.ndarray,
    member_masks: Sequence[np.ndarray],
    removed_mask: np.ndarray,
    candidate: Dict[str, Any],
    cfg: DiagramMaskClusterConfig,
) -> Dict[str, float]:
    candidate_bool = np.asarray(candidate_mask, dtype=bool)
    candidate_area = _mask_area(candidate_bool)
    if candidate_area <= 0:
        return {
            "keep": 0.0,
            "refine_score": 0.0,
            "member_coverage_mean": 0.0,
            "member_coverage_min": 0.0,
            "coarse_coverage": 0.0,
            "extra_area_ratio": 1.0,
            "annotation_overlap_ratio": 1.0,
        }
    coarse_bool = np.asarray(coarse_mask, dtype=bool)
    coarse_inter = np.logical_and(candidate_bool, coarse_bool).sum(dtype=np.int64)
    coarse_coverage = float(coarse_inter) / float(max(1, _mask_area(coarse_bool)))
    member_coverages: List[float] = []
    for member_mask in member_masks:
        member_bool = np.asarray(member_mask, dtype=bool)
        member_coverages.append(
            float(np.logical_and(candidate_bool, member_bool).sum(dtype=np.int64)) / float(max(1, _mask_area(member_bool)))
        )
    member_coverage_mean = float(sum(member_coverages) / float(max(1, len(member_coverages))))
    member_coverage_min = float(min(member_coverages)) if member_coverages else 0.0
    extra_area = max(0, candidate_area - int(coarse_inter))
    extra_area_ratio = float(extra_area) / float(max(1, candidate_area))
    annotation_overlap = float(np.logical_and(candidate_bool, np.asarray(removed_mask, dtype=bool)).sum(dtype=np.int64)) / float(max(1, candidate_area))
    refine_score = (
        0.30 * member_coverage_mean
        + 0.22 * member_coverage_min
        + 0.18 * coarse_coverage
        + 0.14 * float(candidate.get("sam_pred_iou", 0.0) or 0.0)
        + 0.10 * float(candidate.get("sam_stability_score", 0.0) or 0.0)
        + 0.06 * max(0.0, 1.0 - extra_area_ratio)
    )
    keep = (
        member_coverage_min >= float(cfg.refiner_min_member_coverage)
        and coarse_coverage >= float(cfg.refiner_min_coarse_coverage)
        and extra_area_ratio <= float(cfg.refiner_max_extra_area_ratio)
        and annotation_overlap <= float(cfg.refiner_max_annotation_overlap)
    )
    return {
        "keep": float(1.0 if keep else 0.0),
        "refine_score": float(refine_score),
        "member_coverage_mean": float(member_coverage_mean),
        "member_coverage_min": float(member_coverage_min),
        "coarse_coverage": float(coarse_coverage),
        "extra_area_ratio": float(extra_area_ratio),
        "annotation_overlap_ratio": float(annotation_overlap),
    }


def _should_consider_refiner_pair(a: Dict[str, Any], b: Dict[str, Any], cfg: DiagramMaskClusterConfig) -> Tuple[bool, Dict[str, float]]:
    metrics = _compute_pair_metrics(a, b)
    ok = (
        metrics["bbox_gap_px"] <= float(cfg.refiner_candidate_bbox_gap_px)
        and metrics["dino_cosine"] >= float(cfg.refiner_candidate_min_dino_cosine)
        and metrics["color_similarity"] >= float(cfg.refiner_candidate_min_color_similarity)
        and metrics["shape_similarity"] >= float(cfg.refiner_candidate_min_shape_similarity)
        and metrics["combined_feature_score"] >= float(cfg.refiner_candidate_min_combined_score)
    )
    return bool(ok), metrics


def _run_samrefiner_group_refinement(
    image_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    removed_mask: np.ndarray,
    member_rows: Sequence[Dict[str, Any]],
    cfg: DiagramMaskClusterConfig,
) -> Optional[Dict[str, Any]]:
    source_rows = list(member_rows)
    if len(source_rows) <= 1:
        return None
    union_mask = np.logical_or.reduce([np.asarray(row["mask"], dtype=bool) for row in source_rows])
    bbox = _mask_bbox(union_mask)
    if not bbox:
        return None
    h, w = union_mask.shape
    crop_bbox = _expand_bbox_xyxy(bbox, int(cfg.refiner_prompt_pad_px), width=w, height=h)
    x0, y0, x1, y1 = crop_bbox
    local_cleaned = np.asarray(cleaned_rgb[y0:y1, x0:x1], dtype=np.uint8)
    local_removed = np.asarray(removed_mask[y0:y1, x0:x1] > 0, dtype=bool)
    local_union = np.asarray(union_mask[y0:y1, x0:x1], dtype=bool)
    local_members = [np.asarray(row["mask"][y0:y1, x0:x1], dtype=bool) for row in source_rows]
    prompt_sets = _build_refiner_prompt_sets(local_union, [{"mask": m} for m in local_members], cfg)
    candidates = _run_sam2_prompt_refinement(local_cleaned, prompt_sets, cfg)
    best_row: Optional[Dict[str, Any]] = None
    for candidate in candidates:
        candidate_mask = np.asarray(candidate.get("mask"), dtype=bool)
        score = _score_refiner_candidate(candidate_mask, local_union, local_members, local_removed, candidate, cfg)
        if score["keep"] <= 0.0:
            continue
        full_mask = np.zeros_like(union_mask, dtype=bool)
        full_mask[y0:y1, x0:x1] = candidate_mask
        refine_meta = {
            "refine_stage": "samrefiner_style",
            "refine_score": float(score["refine_score"]),
            "regroup_score": float(score["member_coverage_mean"]),
            "dedupe_score": float(max(score["member_coverage_min"], 1.0 - score["extra_area_ratio"])),
            "refine_prompt_count": len(prompt_sets),
            "refine_candidate_member_coverage": float(score["member_coverage_mean"]),
            "refine_candidate_coarse_coverage": float(score["coarse_coverage"]),
        }
        built = _build_cluster_row_from_mask(
            image_rgb,
            full_mask,
            source_rows,
            cfg,
            proposal_id="refined_pending",
            feature_cluster_id="refined_pending",
            refine_meta=refine_meta,
        )
        if built is None:
            continue
        if best_row is None or _cluster_quality_score(built) > _cluster_quality_score(best_row):
            best_row = built
    return best_row


def _dedupe_cluster_rows(
    rows: Sequence[Dict[str, Any]],
    cfg: DiagramMaskClusterConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ordered = sorted(list(rows), key=_cluster_quality_score, reverse=True)
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for row in ordered:
        drop = False
        for prev in kept:
            bbox = row.get("bbox_xyxy")
            prev_bbox = prev.get("bbox_xyxy")
            if _bbox_intersection_xyxy(bbox, prev_bbox) is None:
                iou = 0.0
                containment = 0.0
            else:
                iou = _mask_iou(np.asarray(row["mask"], dtype=bool), np.asarray(prev["mask"], dtype=bool), bbox, prev_bbox)
                containment = _mask_containment(np.asarray(row["mask"], dtype=bool), np.asarray(prev["mask"], dtype=bool), bbox, prev_bbox)
            if iou >= float(cfg.refiner_duplicate_iou_thresh) or containment >= float(cfg.refiner_duplicate_containment_thresh):
                drop = True
                dropped.append(
                    {
                        "dropped_feature_cluster_id": str(row.get("feature_cluster_id", "") or row.get("proposal_id", "") or ""),
                        "kept_feature_cluster_id": str(prev.get("feature_cluster_id", "") or prev.get("proposal_id", "") or ""),
                        "iou": float(iou),
                        "containment": float(containment),
                    }
                )
                break
        if not drop:
            kept.append(row)
    return kept, dropped


def _refine_merged_clusters_samrefiner_style(
    image_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    removed_mask: np.ndarray,
    merged: List[Dict[str, Any]],
    cfg: DiagramMaskClusterConfig,
    refinement_runner: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not bool(cfg.refiner_enabled) or not merged:
        return list(merged), []
    t0 = time.perf_counter()
    runner = refinement_runner or _run_samrefiner_group_refinement
    coarse_rows = [dict(row) for row in merged]
    coarse_rows, dedupe_debug = _dedupe_cluster_rows(coarse_rows, cfg)
    if not coarse_rows:
        return [], dedupe_debug
    _progress_log(cfg, f"refiner start coarse_clusters={len(coarse_rows)}")

    uf = _UnionFind(len(coarse_rows))
    refine_debug: List[Dict[str, Any]] = list(dedupe_debug)
    for i in range(len(coarse_rows)):
        for j in range(i + 1, len(coarse_rows)):
            ok, metrics = _should_consider_refiner_pair(coarse_rows[i], coarse_rows[j], cfg)
            if ok:
                uf.union(i, j)
                refine_debug.append(
                    {
                        "candidate_a": str(coarse_rows[i].get("feature_cluster_id", "") or coarse_rows[i].get("proposal_id", "") or ""),
                        "candidate_b": str(coarse_rows[j].get("feature_cluster_id", "") or coarse_rows[j].get("proposal_id", "") or ""),
                        "candidate_pair": True,
                        **metrics,
                    }
                )
        if (i + 1) == len(coarse_rows) or (i + 1) % 25 == 0:
            _progress_log(
                cfg,
                f"refiner candidate scan anchors={i + 1}/{len(coarse_rows)} debug_events={len(refine_debug)} elapsed_s={time.perf_counter() - t0:.1f}",
            )
    groups: Dict[int, List[int]] = {}
    for idx in range(len(coarse_rows)):
        groups.setdefault(uf.find(idx), []).append(idx)

    final_rows: List[Dict[str, Any]] = []
    refined_index = 0
    for indexes in groups.values():
        members = [coarse_rows[i] for i in indexes]
        if len(members) <= 1:
            final_rows.extend(members)
            continue
        refined = runner(image_rgb, cleaned_rgb, removed_mask, members, cfg)
        if refined is not None:
            refined["proposal_id"] = f"r{refined_index:04d}"
            refined["feature_cluster_id"] = f"rf_{refined_index:04d}"
            refined_index += 1
            final_rows.append(refined)
            refine_debug.append(
                {
                    "refined_group_ids": list(refined.get("refined_from_cluster_ids", []) or []),
                    "feature_cluster_id": str(refined.get("feature_cluster_id", "") or ""),
                    "refine_score": float(refined.get("refine_score", 0.0) or 0.0),
                    "coarse_cluster_count": len(members),
                }
            )
        else:
            final_rows.extend(members)

    final_rows, final_dedupe_debug = _dedupe_cluster_rows(final_rows, cfg)
    refine_debug.extend(final_dedupe_debug)
    final_rows.sort(key=lambda row: (int(row["bbox_xyxy"][0]), int(row["bbox_xyxy"][1])))
    for idx, row in enumerate(final_rows):
        row["proposal_id"] = f"r{idx:04d}"
        row["feature_cluster_id"] = str(row.get("feature_cluster_id", "") or f"rf_{idx:04d}")
        row["group_index_in_color"] = int(idx)
    _progress_log(
        cfg,
        f"refiner done final_clusters={len(final_rows)} debug_events={len(refine_debug)} elapsed_s={time.perf_counter() - t0:.1f}",
    )
    return final_rows, refine_debug


def _cluster_entry_from_row(row: Dict[str, Any], cluster_index: int, mask_name: str) -> Dict[str, Any]:
    entry = {
        "color_id": 0,
        "color_name": "mask",
        "group_index_in_color": int(cluster_index),
        "stroke_indexes": [int(v) for v in (row.get("stroke_indexes") or [])],
        "bbox_xyxy": [int(v) for v in row.get("crop_bbox_xyxy", row.get("bbox_xyxy", [0, 0, 0, 0]))],
        "crop_file_mask": mask_name,
        "pipeline": "sam2_dinov2_mini_mask_groups_v1",
        "mask_area_px": int(row["mask_area_px"]),
        "sam_stability_score": float(row["sam_stability_score"]),
        "sam_pred_iou": float(row["sam_pred_iou"]),
        "merged_from_mask_ids": list(row.get("merged_from_mask_ids", []) or []),
        "feature_cluster_id": str(row.get("feature_cluster_id", "") or ""),
        "proposal_id": str(row.get("proposal_id", "") or f"mask_{cluster_index:04d}"),
        "prompt_key": str(row.get("prompt_key", "") or ""),
        "prompt_index": row.get("prompt_index"),
        "prompt_mask_index": row.get("prompt_mask_index"),
        "prompt_child_mask_ids": list(row.get("prompt_child_mask_ids", []) or []),
        "group_id": row.get("group_id"),
        "bbox_center_xy": [float(v) for v in (row.get("bbox_center_xy") or _bbox_center_xyxy(row.get("bbox_xyxy", [0, 0, 0, 0])))],
        "bbox_diag_px": float(row.get("bbox_diag_px", 0.0) or 0.0),
    }
    optional_keys = [
        "refined_from_cluster_ids",
        "refine_stage",
        "refine_score",
        "dedupe_score",
        "regroup_score",
        "refine_prompt_count",
        "refine_candidate_member_coverage",
        "refine_candidate_coarse_coverage",
    ]
    for key in optional_keys:
        if key in row and row.get(key) is not None:
            entry[key] = row.get(key)
    return entry


def _write_outputs_for_processed(
    *,
    idx: int,
    image_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    removed_overlay_rgb: np.ndarray,
    removed_mask: np.ndarray,
    proposals: List[Dict[str, Any]],
    prompt_debug: List[Dict[str, Any]],
    island_groups: List[Dict[str, Any]],
    island_edge_debug: List[Dict[str, Any]],
    stroke_candidates: Optional[Dict[str, Any]],
    cfg: DiagramMaskClusterConfig,
) -> Dict[str, Any]:
    render_dir = CLUSTER_RENDER_DIR / f"processed_{int(idx)}"
    map_dir = CLUSTER_MAP_DIR / f"processed_{int(idx)}"
    debug_dir = DEBUG_ROOT / f"processed_{int(idx)}"
    if bool(cfg.clear_existing_outputs):
        for path in (render_dir, map_dir, debug_dir):
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    cluster_entries: List[Dict[str, Any]] = []
    renders_mask_rgb: Dict[str, np.ndarray] = {}
    for cluster_index, row in enumerate(proposals):
        mask_name = f"mini_{cluster_index:04d}.png"
        rgba, crop_bbox, _crop_mask = _extract_mask_crop_rgba(image_rgb, np.asarray(row["mask"], dtype=bool), int(cfg.crop_pad_px))
        row["crop_bbox_xyxy"] = crop_bbox
        renders_mask_rgb[mask_name] = rgba[..., :3]
        Image.fromarray(rgba, mode="RGBA").save(render_dir / mask_name, compress_level=int(cfg.png_compress_level))
        cluster_entries.append(_cluster_entry_from_row(row, cluster_index, mask_name))

    cluster_map = {
        "image_index": int(idx),
        "image_size": [int(image_rgb.shape[1]), int(image_rgb.shape[0])],
        "pipeline": "sam2_dinov2_mini_mask_groups_v1",
        "clusters": cluster_entries,
        "mini_masks": cluster_entries,
        "groups": list(island_groups or []),
        "stroke_candidates": (stroke_candidates or {}).get("candidates", []),
        "stroke_candidate_meta": {
            key: value
            for key, value in dict(stroke_candidates or {}).items()
            if key != "candidates"
        },
    }
    (map_dir / "clusters.json").write_text(json.dumps(cluster_map, ensure_ascii=False, indent=2), encoding="utf-8")

    Image.fromarray(cleaned_rgb).save(debug_dir / "cleaned.png", compress_level=int(cfg.png_compress_level))
    Image.fromarray(removed_overlay_rgb).save(debug_dir / "cleanup_overlay.png", compress_level=int(cfg.png_compress_level))
    Image.fromarray((removed_mask > 0).astype(np.uint8) * 255).save(debug_dir / "cleanup_removed_mask.png", compress_level=int(cfg.png_compress_level))
    manifest = {
        "processed_id": f"processed_{int(idx)}",
        "config": cfg.to_public_dict(),
        "proposals": [
            {
                k: v for k, v in row.items()
                if k not in {"mask", "dino_embedding", "crop_rgba"}
            }
            for row in proposals
        ],
        "prompt_consolidation": list(prompt_debug or []),
        "island_groups": list(island_groups or []),
        "island_edge_debug": list(island_edge_debug or []),
        "stroke_candidates": stroke_candidates or {},
    }
    (debug_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "cluster_entries": cluster_entries,
        "renders_mask_rgb": renders_mask_rgb,
        "debug_dir": debug_dir,
        "cluster_map_path": map_dir / "clusters.json",
        "render_dir": render_dir,
        "groups": list(island_groups or []),
    }


def _build_in_memory_cluster_contract(
    image_rgb_or_rows: Any,
    proposals: Optional[List[Dict[str, Any]]] = None,
    island_groups: Optional[List[Dict[str, Any]]] = None,
    cfg: Optional[DiagramMaskClusterConfig] = None,
) -> Dict[str, Any]:
    if proposals is None and isinstance(image_rgb_or_rows, list):
        rows = list(image_rgb_or_rows)
        cluster_entries: List[Dict[str, Any]] = []
        renders_mask_rgb: Dict[str, np.ndarray] = {}
        for cluster_index, row in enumerate(rows):
            mask_name = f"mask_{cluster_index:04d}.png"
            rgba = np.asarray(row["crop_rgba"], dtype=np.uint8)
            renders_mask_rgb[mask_name] = rgba[..., :3]
            cluster_entries.append(_cluster_entry_from_row(row, cluster_index, mask_name))
        return {
            "cluster_entries": cluster_entries,
            "renders_mask_rgb": renders_mask_rgb,
            "groups": list(island_groups or []),
        }

    image_rgb = np.asarray(image_rgb_or_rows, dtype=np.uint8)
    cfg = cfg or build_config()
    proposals = list(proposals or [])
    cluster_entries: List[Dict[str, Any]] = []
    renders_mask_rgb: Dict[str, np.ndarray] = {}
    for cluster_index, row in enumerate(proposals):
        mask_name = f"mini_{cluster_index:04d}.png"
        rgba, crop_bbox, _crop_mask = _extract_mask_crop_rgba(image_rgb, np.asarray(row["mask"], dtype=bool), int(cfg.crop_pad_px))
        row["crop_bbox_xyxy"] = crop_bbox
        renders_mask_rgb[mask_name] = rgba[..., :3]
        cluster_entries.append(_cluster_entry_from_row(row, cluster_index, mask_name))
    return {
        "cluster_entries": cluster_entries,
        "renders_mask_rgb": renders_mask_rgb,
        "groups": list(island_groups or []),
    }


def cluster_image_rgb(
    image_rgb: np.ndarray,
    *,
    processed_id: str,
    save_outputs: bool = False,
    config: Optional[DiagramMaskClusterConfig] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = config or build_config()
    _progress_log(cfg, f"cluster start processed_id={processed_id} size={image_rgb.shape[1]}x{image_rgb.shape[0]}")
    cleanup = _preclean_diagram_copy(image_rgb, cfg)
    cleaned_rgb = cleanup["cleaned_rgb"]
    removed_mask = cleanup["removed_mask"]
    _progress_log(cfg, f"preclean done removed_px={int(np.count_nonzero(removed_mask))} elapsed_s={time.perf_counter() - t0:.1f}")
    raw_proposals = _generate_sam2_candidate_masks(cleaned_rgb, cfg)
    _progress_log(cfg, f"proposal generation done raw_proposals={len(raw_proposals)} elapsed_s={time.perf_counter() - t0:.1f}")
    raw_proposals = _suppress_redundant_proposals(raw_proposals, removed_mask, cfg)
    _progress_log(cfg, f"proposal suppression done raw_proposals={len(raw_proposals)} elapsed_s={time.perf_counter() - t0:.1f}")
    proposals, prompt_debug = _consolidate_prompt_level_proposals(raw_proposals, cfg)
    _progress_log(cfg, f"prompt consolidation done mini_masks={len(proposals)} elapsed_s={time.perf_counter() - t0:.1f}")
    proposals = _attach_dino_and_shape_features(cleaned_rgb, proposals, cfg)
    proposals = _annotate_basic_proposal_geometry(proposals)
    _progress_log(cfg, f"feature attach done mini_masks={len(proposals)} elapsed_s={time.perf_counter() - t0:.1f}")
    island_groups, island_edge_debug = _build_local_island_groups(proposals, cfg)
    _progress_log(cfg, f"island grouping done groups={len(island_groups)} mini_masks={len(proposals)} elapsed_s={time.perf_counter() - t0:.1f}")
    stroke_candidates = _build_stroke_candidate_outputs(
        image_rgb=image_rgb,
        processed_id=str(processed_id),
        proposals=proposals,
        island_groups=island_groups,
        cfg=cfg,
    )
    _progress_log(
        cfg,
        f"stroke candidates done candidates={len(stroke_candidates.get('candidates') or [])} elapsed_s={time.perf_counter() - t0:.1f}",
    )

    idx = _extract_index_from_processed_name(f"{processed_id}.png")
    out: Dict[str, Any] = {
        "processed_id": str(processed_id),
        "config": cfg.to_public_dict(),
        "cleaned_rgb": cleaned_rgb,
        "removed_mask": removed_mask,
        "removed_overlay_rgb": cleanup["removed_overlay_rgb"],
        "raw_proposals": raw_proposals,
        "proposals": proposals,
        "mini_mask_proposals": proposals,
        "prompt_consolidation": prompt_debug,
        "groups": island_groups,
        "island_groups": island_groups,
        "island_edge_debug": island_edge_debug,
        "coarse_merged_clusters": [],
        "merged_clusters": [],
        "merge_pairs": [],
        "refine_debug": [],
        "stroke_candidates": stroke_candidates,
    }
    out.update(_build_in_memory_cluster_contract(image_rgb, proposals, island_groups, cfg))
    if save_outputs and idx is not None:
        out.update(
            _write_outputs_for_processed(
                idx=idx,
                image_rgb=image_rgb,
                cleaned_rgb=cleaned_rgb,
                removed_overlay_rgb=cleanup["removed_overlay_rgb"],
                removed_mask=removed_mask,
                proposals=proposals,
                prompt_debug=prompt_debug,
                island_groups=island_groups,
                island_edge_debug=island_edge_debug,
                stroke_candidates=stroke_candidates,
                cfg=cfg,
            )
        )
    _progress_log(cfg, f"cluster done processed_id={processed_id} mini_masks={len(proposals)} groups={len(island_groups)} total_elapsed_s={time.perf_counter() - t0:.1f}")
    return out


def ensure_processed_clusters(
    processed_id_or_index: Any,
    *,
    save_outputs: bool = True,
    config: Optional[DiagramMaskClusterConfig] = None,
) -> Dict[str, Any]:
    cfg = config or build_config()
    image_path = _resolve_processed_png_path(processed_id_or_index)
    idx = _extract_index_from_processed_name(image_path.name)
    if idx is None:
        raise RuntimeError(f"Could not parse processed index from {image_path.name}")
    return cluster_image_rgb(
        _load_rgb_image(image_path),
        processed_id=f"processed_{idx}",
        save_outputs=save_outputs,
        config=cfg,
    )


def cluster_diagrams_in_memory(
    preproc_by_idx: Dict[int, Dict[str, Any]],
    *,
    save_outputs: bool = False,
    config: Optional[DiagramMaskClusterConfig] = None,
) -> Dict[int, Dict[str, Any]]:
    cfg = config or build_config()
    out: Dict[int, Dict[str, Any]] = {}
    for idx, prep in (preproc_by_idx or {}).items():
        if not isinstance(prep, dict):
            continue
        cleaned_bgr = prep.get("cleaned_bgr")
        if cleaned_bgr is None:
            continue
        image_rgb = cv2.cvtColor(np.asarray(cleaned_bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        row = cluster_image_rgb(
            image_rgb,
            processed_id=f"processed_{int(idx)}",
            save_outputs=save_outputs,
            config=cfg,
        )
        out[int(idx)] = {
            "clusters": row.get("cluster_entries", []),
            "mini_masks": row.get("cluster_entries", []),
            "groups": row.get("groups", []),
            "stroke_candidates": row.get("stroke_candidates", {}),
            "renders_mask_rgb": row.get("renders_mask_rgb", {}),
            "debug_dir": str(row.get("debug_dir")) if row.get("debug_dir") else None,
            "cluster_map_path": str(row.get("cluster_map_path")) if row.get("cluster_map_path") else None,
        }
    return out


def build_overlay_rgb(image_rgb: np.ndarray, clusters: Sequence[Dict[str, Any]]) -> np.ndarray:
    out = np.asarray(image_rgb, dtype=np.uint8).copy()
    palette = [
        (255, 0, 0),
        (0, 180, 255),
        (0, 200, 120),
        (255, 180, 0),
        (180, 0, 255),
        (255, 80, 140),
        (80, 255, 220),
    ]
    for idx, row in enumerate(clusters, start=1):
        bbox = row.get("bbox_xyxy") if isinstance(row, dict) else None
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox]
        color = palette[(idx - 1) % len(palette)]
        cv2.rectangle(out, (x0, y0), (max(x0 + 1, x1 - 1), max(y0 + 1, y1 - 1)), color, 2)
        cv2.putText(out, str(idx), (max(0, x0 + 3), max(14, y0 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


__all__ = [
    "DEBUG_ROOT",
    "CLUSTER_MAP_DIR",
    "CLUSTER_RENDER_DIR",
    "SAM_DINO_RENDER_DIR",
    "DiagramMaskClusterConfig",
    "build_config",
    "build_overlay_rgb",
    "cluster_diagrams_in_memory",
    "cluster_image_rgb",
    "ensure_processed_clusters",
    "get_diagram_cluster_backend",
    "unload_hot_models",
    "_preclean_diagram_copy",
    "_suppress_redundant_proposals",
    "_merge_component_proposals",
]
