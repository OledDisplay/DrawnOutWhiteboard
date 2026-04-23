#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import math
import os
import shutil
import time
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
CLUSTER_MAP_DIR = BASE_DIR / "ClusterMaps"
DEBUG_ROOT = BASE_DIR / "PipelineOutputs" / "diagram_mask_clusters"

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
    sam_model_id: str = os.environ.get("DIAGRAM_CLUSTER_SAM2_MODEL_ID", "facebook/sam2.1-hiera-small")
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
    point_grid_step: int = _env_int("DIAGRAM_CLUSTER_POINT_GRID_STEP", 112)
    point_batch_size: int = _env_int("DIAGRAM_CLUSTER_POINT_BATCH_SIZE", 24)
    sam_pred_iou_thresh: float = _env_float("DIAGRAM_CLUSTER_SAM_PRED_IOU_THRESH", 0.88)
    sam_stability_score_thresh: float = _env_float("DIAGRAM_CLUSTER_SAM_STABILITY_SCORE_THRESH", 0.86)
    sam_mask_threshold: float = _env_float("DIAGRAM_CLUSTER_SAM_MASK_THRESHOLD", 0.0)
    sam_stability_offset: float = _env_float("DIAGRAM_CLUSTER_SAM_STABILITY_OFFSET", 0.05)
    min_region_area_px: int = _env_int("DIAGRAM_CLUSTER_MIN_REGION_AREA_PX", 1800)
    max_points: int = _env_int("DIAGRAM_CLUSTER_MAX_POINTS", 220)

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
    merge_similarity_only_bbox_gap_px: float = _env_float("DIAGRAM_CLUSTER_MERGE_SIMILARITY_ONLY_BBOX_GAP_PX", 160.0)
    merge_iou_thresh: float = _env_float("DIAGRAM_CLUSTER_MERGE_IOU_THRESH", 0.18)
    merge_containment_thresh: float = _env_float("DIAGRAM_CLUSTER_MERGE_CONTAINMENT_THRESH", 0.84)
    merge_bbox_gap_px: float = _env_float("DIAGRAM_CLUSTER_MERGE_BBOX_GAP_PX", 26.0)
    merge_bbox_growth_max: float = _env_float("DIAGRAM_CLUSTER_MERGE_BBOX_GROWTH_MAX", 0.55)

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

    # output
    png_compress_level: int = _env_int("DIAGRAM_CLUSTER_PNG_COMPRESS_LEVEL", 1)
    clear_existing_outputs: bool = _env_bool("DIAGRAM_CLUSTER_CLEAR_EXISTING_OUTPUTS", False)
    log_progress: bool = _env_bool("DIAGRAM_CLUSTER_LOG_PROGRESS", False)

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
        or metrics["containment"] >= float(cfg.merge_containment_thresh)
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
    similarity_only_ok = (
        metrics["dino_cosine"] >= float(cfg.merge_similarity_only_min_dino_cosine)
        and metrics["color_similarity"] >= float(cfg.merge_similarity_only_min_color_similarity)
        and metrics["shape_similarity"] >= float(cfg.merge_similarity_only_min_shape_similarity)
        and metrics["combined_feature_score"] >= float(cfg.merge_similarity_only_min_score)
        and metrics["bbox_gap_px"] <= float(cfg.merge_similarity_only_bbox_gap_px)
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
        "bbox_xyxy": [int(v) for v in row["crop_bbox_xyxy"]],
        "crop_file_mask": mask_name,
        "pipeline": "sam2_dinov2",
        "mask_area_px": int(row["mask_area_px"]),
        "sam_stability_score": float(row["sam_stability_score"]),
        "sam_pred_iou": float(row["sam_pred_iou"]),
        "merged_from_mask_ids": list(row.get("merged_from_mask_ids", []) or []),
        "feature_cluster_id": str(row.get("feature_cluster_id", "") or ""),
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
    pair_debug: List[Dict[str, Any]],
    coarse_merged: List[Dict[str, Any]],
    merged: List[Dict[str, Any]],
    refine_debug: List[Dict[str, Any]],
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
    for cluster_index, row in enumerate(merged):
        mask_name = f"mask_{cluster_index:04d}.png"
        rgba = np.asarray(row["crop_rgba"], dtype=np.uint8)
        renders_mask_rgb[mask_name] = rgba[..., :3]
        Image.fromarray(rgba, mode="RGBA").save(render_dir / mask_name, compress_level=int(cfg.png_compress_level))
        cluster_entries.append(_cluster_entry_from_row(row, cluster_index, mask_name))

    cluster_map = {
        "image_index": int(idx),
        "image_size": [int(image_rgb.shape[1]), int(image_rgb.shape[0])],
        "pipeline": "sam2_dinov2",
        "clusters": cluster_entries,
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
        "merge_pairs": pair_debug,
        "coarse_merged_clusters": [
            {
                k: v for k, v in row.items()
                if k not in {"mask", "crop_rgba"}
            }
            for row in coarse_merged
        ],
        "merged_clusters": [
            {
                k: v for k, v in row.items()
                if k not in {"mask", "crop_rgba"}
            }
            for row in merged
        ],
        "refine_debug": list(refine_debug or []),
    }
    (debug_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "cluster_entries": cluster_entries,
        "renders_mask_rgb": renders_mask_rgb,
        "debug_dir": debug_dir,
        "cluster_map_path": map_dir / "clusters.json",
        "render_dir": render_dir,
    }


def _build_in_memory_cluster_contract(merged: List[Dict[str, Any]]) -> Dict[str, Any]:
    cluster_entries: List[Dict[str, Any]] = []
    renders_mask_rgb: Dict[str, np.ndarray] = {}
    for cluster_index, row in enumerate(merged):
        mask_name = f"mask_{cluster_index:04d}.png"
        rgba = np.asarray(row["crop_rgba"], dtype=np.uint8)
        renders_mask_rgb[mask_name] = rgba[..., :3]
        cluster_entries.append(_cluster_entry_from_row(row, cluster_index, mask_name))
    return {
        "cluster_entries": cluster_entries,
        "renders_mask_rgb": renders_mask_rgb,
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
    proposals = _generate_sam2_candidate_masks(cleaned_rgb, cfg)
    _progress_log(cfg, f"proposal generation done proposals={len(proposals)} elapsed_s={time.perf_counter() - t0:.1f}")
    proposals = _suppress_redundant_proposals(proposals, removed_mask, cfg)
    _progress_log(cfg, f"proposal suppression done proposals={len(proposals)} elapsed_s={time.perf_counter() - t0:.1f}")
    proposals = _attach_dino_and_shape_features(cleaned_rgb, proposals, cfg)
    _progress_log(cfg, f"feature attach done proposals={len(proposals)} elapsed_s={time.perf_counter() - t0:.1f}")
    coarse_merged, pair_debug = _merge_component_proposals(image_rgb, proposals, cfg)
    _progress_log(cfg, f"coarse merge done coarse_clusters={len(coarse_merged)} elapsed_s={time.perf_counter() - t0:.1f}")
    merged, refine_debug = _refine_merged_clusters_samrefiner_style(
        image_rgb,
        cleaned_rgb,
        removed_mask,
        coarse_merged,
        cfg,
    )
    _progress_log(cfg, f"refiner stage done final_clusters={len(merged)} elapsed_s={time.perf_counter() - t0:.1f}")

    idx = _extract_index_from_processed_name(f"{processed_id}.png")
    out: Dict[str, Any] = {
        "processed_id": str(processed_id),
        "config": cfg.to_public_dict(),
        "cleaned_rgb": cleaned_rgb,
        "removed_mask": removed_mask,
        "removed_overlay_rgb": cleanup["removed_overlay_rgb"],
        "proposals": proposals,
        "coarse_merged_clusters": coarse_merged,
        "merged_clusters": merged,
        "merge_pairs": pair_debug,
        "refine_debug": refine_debug,
    }
    out.update(_build_in_memory_cluster_contract(merged))
    if save_outputs and idx is not None:
        out.update(
            _write_outputs_for_processed(
                idx=idx,
                image_rgb=image_rgb,
                cleaned_rgb=cleaned_rgb,
                removed_overlay_rgb=cleanup["removed_overlay_rgb"],
                removed_mask=removed_mask,
                proposals=proposals,
                pair_debug=pair_debug,
                coarse_merged=coarse_merged,
                merged=merged,
                refine_debug=refine_debug,
                cfg=cfg,
            )
        )
    _progress_log(cfg, f"cluster done processed_id={processed_id} final_clusters={len(merged)} total_elapsed_s={time.perf_counter() - t0:.1f}")
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
    "DiagramMaskClusterConfig",
    "build_config",
    "build_overlay_rgb",
    "cluster_diagrams_in_memory",
    "cluster_image_rgb",
    "ensure_processed_clusters",
    "get_diagram_cluster_backend",
    "_preclean_diagram_copy",
    "_suppress_redundant_proposals",
    "_merge_component_proposals",
]
