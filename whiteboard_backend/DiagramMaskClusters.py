#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import math
import os
import shutil
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cosine
from transformers import AutoImageProcessor, Dinov2Model


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "ProccessedImages"
CLUSTER_RENDER_DIR = BASE_DIR / "ClusterRenders"
CLUSTER_MAP_DIR = BASE_DIR / "ClusterMaps"
DEBUG_ROOT = BASE_DIR / "PipelineOutputs" / "diagram_mask_clusters"

_PROCESSED_PLAIN_RE = __import__("re").compile(r"^(?:proccessed|processed)_(\d+)\.png$", __import__("re").IGNORECASE)

_SAM3_BUNDLE: Optional[Dict[str, Any]] = None
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


def _env_optional_str(name: str, default: Optional[str] = None) -> Optional[str]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = str(raw).strip()
    return text or default


def get_diagram_cluster_backend() -> str:
    text = str(os.environ.get("DIAGRAM_CLUSTER_BACKEND", "stroke") or "stroke").strip().lower()
    if text == "sam2_dinov2":
        return "sam3_dinov2"
    return text if text in {"stroke", "sam3_dinov2"} else "stroke"


def _torch_device() -> torch.device:
    force_cpu = _env_bool("DIAGRAM_CLUSTER_FORCE_CPU", False)
    if torch.cuda.is_available() and not force_cpu:
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class DiagramMaskClusterConfig:
    sam_model_id: str = os.environ.get(
        "DIAGRAM_CLUSTER_SAM3_MODEL_ID",
        os.environ.get("DIAGRAM_CLUSTER_SAM2_MODEL_ID", "sam3"),
    )
    sam_checkpoint_path: Optional[str] = _env_optional_str("DIAGRAM_CLUSTER_SAM3_CKPT", None)
    sam_backbone_type: str = os.environ.get("DIAGRAM_CLUSTER_SAM3_BACKBONE_TYPE", "tinyvit")
    sam_model_name: str = os.environ.get("DIAGRAM_CLUSTER_SAM3_MODEL_NAME", "11m")
    dino_model_id: str = os.environ.get("DIAGRAM_CLUSTER_DINO_MODEL_ID", "facebook/dinov2-base")
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

    # sam3 proposal generation
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
    proposal_canvas_max_mask_area_ratio: float = _env_float("DIAGRAM_CLUSTER_PROPOSAL_CANVAS_MAX_MASK_AREA_RATIO", 0.78)
    proposal_canvas_min_border_touches: int = _env_int("DIAGRAM_CLUSTER_PROPOSAL_CANVAS_MIN_BORDER_TOUCHES", 3)
    proposal_border_touch_margin_px: int = _env_int("DIAGRAM_CLUSTER_PROPOSAL_BORDER_TOUCH_MARGIN_PX", 8)
    proposal_background_prompt_max_mask_area_ratio: float = _env_float("DIAGRAM_CLUSTER_PROPOSAL_BACKGROUND_PROMPT_MAX_MASK_AREA_RATIO", 0.45)
    proposal_background_prompt_rgb_floor: int = _env_int("DIAGRAM_CLUSTER_PROPOSAL_BACKGROUND_PROMPT_RGB_FLOOR", 245)
    proposal_large_mask_near_white_drop_ratio: float = _env_float("DIAGRAM_CLUSTER_PROPOSAL_LARGE_MASK_NEAR_WHITE_DROP_RATIO", 0.18)

    # feature extraction
    dino_image_size: int = _env_int("DIAGRAM_CLUSTER_DINO_IMAGE_SIZE", 224)
    histogram_bins: int = _env_int("DIAGRAM_CLUSTER_HISTOGRAM_BINS", 8)
    crop_pad_px: int = _env_int("DIAGRAM_CLUSTER_CROP_PAD_PX", 10)

    # merge
    merge_min_dino_cosine: float = _env_float("DIAGRAM_CLUSTER_MERGE_MIN_DINO_COSINE", 0.94)
    merge_min_color_similarity: float = _env_float("DIAGRAM_CLUSTER_MERGE_MIN_COLOR_SIM", 0.82)
    merge_min_shape_similarity: float = _env_float("DIAGRAM_CLUSTER_MERGE_MIN_SHAPE_SIM", 0.80)
    merge_min_contrast_color_similarity: float = _env_float("DIAGRAM_CLUSTER_MERGE_MIN_CONTRAST_COLOR_SIM", 0.84)
    merge_combined_feature_score: float = _env_float("DIAGRAM_CLUSTER_MERGE_COMBINED_FEATURE_SCORE", 0.88)
    merge_soft_min_dino_cosine: float = _env_float("DIAGRAM_CLUSTER_MERGE_SOFT_MIN_DINO_COSINE", 0.92)
    merge_soft_min_shape_similarity: float = _env_float("DIAGRAM_CLUSTER_MERGE_SOFT_MIN_SHAPE_SIM", 0.76)
    merge_similarity_only_min_dino_cosine: float = _env_float("DIAGRAM_CLUSTER_MERGE_SIMILARITY_ONLY_MIN_DINO_COSINE", 0.97)
    merge_similarity_only_min_color_similarity: float = _env_float("DIAGRAM_CLUSTER_MERGE_SIMILARITY_ONLY_MIN_COLOR_SIM", 0.92)
    merge_similarity_only_min_shape_similarity: float = _env_float("DIAGRAM_CLUSTER_MERGE_SIMILARITY_ONLY_MIN_SHAPE_SIM", 0.88)
    merge_similarity_only_min_score: float = _env_float("DIAGRAM_CLUSTER_MERGE_SIMILARITY_ONLY_MIN_SCORE", 0.94)
    merge_similarity_only_bbox_gap_px: float = _env_float("DIAGRAM_CLUSTER_MERGE_SIMILARITY_ONLY_BBOX_GAP_PX", 32.0)
    merge_iou_thresh: float = _env_float("DIAGRAM_CLUSTER_MERGE_IOU_THRESH", 0.24)
    merge_containment_thresh: float = _env_float("DIAGRAM_CLUSTER_MERGE_CONTAINMENT_THRESH", 0.90)
    merge_bbox_gap_px: float = _env_float("DIAGRAM_CLUSTER_MERGE_BBOX_GAP_PX", 14.0)
    merge_bbox_growth_max: float = _env_float("DIAGRAM_CLUSTER_MERGE_BBOX_GROWTH_MAX", 0.35)

    # output
    png_compress_level: int = _env_int("DIAGRAM_CLUSTER_PNG_COMPRESS_LEVEL", 1)
    clear_existing_outputs: bool = _env_bool("DIAGRAM_CLUSTER_CLEAR_EXISTING_OUTPUTS", False)

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


def _bbox_border_touch_count(bbox: Sequence[int], width: int, height: int, margin_px: int) -> int:
    if not bbox or len(bbox) != 4:
        return 0
    x0, y0, x1, y1 = [int(v) for v in bbox]
    margin = max(0, int(margin_px))
    touches = 0
    if x0 <= margin:
        touches += 1
    if y0 <= margin:
        touches += 1
    if x1 >= max(0, int(width) - margin):
        touches += 1
    if y1 >= max(0, int(height) - margin):
        touches += 1
    return touches


def _prompt_is_near_white(image_rgb: np.ndarray, prompt_point: Any, rgb_floor: int) -> bool:
    if prompt_point is None or len(prompt_point) < 2:
        return False
    x = int(round(float(prompt_point[0])))
    y = int(round(float(prompt_point[1])))
    h, w = image_rgb.shape[:2]
    x = min(max(0, x), max(0, w - 1))
    y = min(max(0, y), max(0, h - 1))
    px = np.asarray(image_rgb[y, x], dtype=np.uint8).reshape(-1)
    return bool(px.size >= 3 and int(px.min()) >= int(rgb_floor))


def _mask_near_white_ratio(image_rgb: np.ndarray, mask: np.ndarray, rgb_floor: int) -> float:
    pixels = np.asarray(image_rgb, dtype=np.uint8)[np.asarray(mask, dtype=bool)]
    if pixels.size == 0:
        return 0.0
    near_white = np.min(pixels, axis=1) >= int(rgb_floor)
    return float(np.count_nonzero(near_white)) / float(max(1, near_white.size))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    if inter <= 0:
        return 0.0
    union = np.logical_or(a, b).sum(dtype=np.int64)
    return float(inter) / float(max(1, union))


def _mask_containment(a: np.ndarray, b: np.ndarray) -> float:
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


def _get_sam3_bundle(cfg: DiagramMaskClusterConfig) -> Dict[str, Any]:
    global _SAM3_BUNDLE
    device = torch.device(cfg.device)
    from shared_models import _resolve_efficientsam3_bpe_path

    def _resolve_sam3_checkpoint_path() -> str:
        explicit = str(cfg.sam_checkpoint_path or "").strip()
        if explicit:
            ckpt_path = Path(explicit)
            if not ckpt_path.is_file():
                raise RuntimeError(f"SAM3 checkpoint not found: {explicit}")
            return str(ckpt_path)

        local_sam3_candidates = [
            BASE_DIR / "checkpoints" / "sam3.pt",
            BASE_DIR / "checkpoints" / "sam3" / "sam3.pt",
        ]
        for ckpt_path in local_sam3_candidates:
            if ckpt_path.is_file():
                return str(ckpt_path)

        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
        except Exception as e:
            raise RuntimeError(
                "huggingface_hub is required to auto-download the SAM3 checkpoint. "
                "Install it or set DIAGRAM_CLUSTER_SAM3_CKPT to a local sam3.pt."
            ) from e

        repo_id = str(os.environ.get("DIAGRAM_CLUSTER_SAM3_HF_REPO_ID", "facebook/sam3") or "").strip()
        filename = str(os.environ.get("DIAGRAM_CLUSTER_SAM3_HF_FILENAME", "sam3.pt") or "").strip()
        local_dir = str(
            os.environ.get("DIAGRAM_CLUSTER_SAM3_HF_LOCAL_DIR", str(BASE_DIR / "checkpoints")) or ""
        ).strip()
        local_files_only = _env_bool("DIAGRAM_CLUSTER_SAM3_HF_LOCAL_FILES_ONLY", False)
        token = str(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or "").strip() or None

        if not repo_id or not filename:
            raise RuntimeError(
                "Missing SAM3 Hugging Face configuration. "
                "Set DIAGRAM_CLUSTER_SAM3_HF_REPO_ID and DIAGRAM_CLUSTER_SAM3_HF_FILENAME."
            )

        Path(local_dir).mkdir(parents=True, exist_ok=True)
        try:
            _ = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                local_dir=local_dir,
                local_files_only=bool(local_files_only),
                token=token,
            )
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_files_only=bool(local_files_only),
                token=token,
            )
        except GatedRepoError as e:
            raise RuntimeError(
                "SAM3 checkpoint download is gated by Hugging Face access. "
                "Request access to facebook/sam3, then set HF_TOKEN (or run hf auth login) and retry."
            ) from e
        except HfHubHTTPError as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code in {401, 403}:
                raise RuntimeError(
                    "SAM3 checkpoint download was denied by Hugging Face. "
                    "Request access to facebook/sam3, then set HF_TOKEN (or run hf auth login) and retry."
                ) from e
            raise RuntimeError(f"SAM3 checkpoint download failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"SAM3 checkpoint download failed: {e}") from e

        resolved = Path(str(ckpt_path))
        if not resolved.is_file():
            raise RuntimeError(f"SAM3 checkpoint download did not produce a file: {ckpt_path}")
        return str(resolved)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Importing from timm\.models\.layers is deprecated.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*No module named 'triton'.*",
            category=UserWarning,
        )
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

    bpe_path = _resolve_efficientsam3_bpe_path()
    resolved_checkpoint_path = _resolve_sam3_checkpoint_path()
    cache_key = (
        cfg.sam_model_id,
        resolved_checkpoint_path,
        bpe_path,
        str(device),
    )
    if _SAM3_BUNDLE and _SAM3_BUNDLE.get("cache_key") == cache_key:
        return _SAM3_BUNDLE

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Importing from timm\.models\.layers is deprecated.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*No module named 'triton'.*",
            category=UserWarning,
        )
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=resolved_checkpoint_path,
            load_from_HF=False,
            device=str(device),
            enable_inst_interactivity=True,
            text_encoder_type=None,
        )
    try:
        model.to(str(device))
    except Exception:
        pass
    try:
        model.eval()
    except Exception:
        pass
    processor = Sam3Processor(
        model,
        device=str(device),
        confidence_threshold=0.0,
    )
    _SAM3_BUNDLE = {
        "processor": processor,
        "model": model,
        "device": device,
        "model_id": cfg.sam_model_id,
        "resolved_checkpoint_path": resolved_checkpoint_path,
        "cache_key": cache_key,
    }
    return _SAM3_BUNDLE


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


def _compute_stability_from_low_res_logits(
    low_res_logits: Any,
    *,
    base_threshold: float,
    offset: float,
) -> np.ndarray:
    arr = np.asarray(low_res_logits)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        return np.zeros((0,), dtype=np.float32)
    masks_a = arr > float(base_threshold)
    masks_b = arr > float(base_threshold + offset)
    inter = np.logical_and(masks_a, masks_b).sum(axis=(1, 2), dtype=np.int64)
    union = np.logical_or(masks_a, masks_b).sum(axis=(1, 2), dtype=np.int64)
    return inter.astype(np.float32) / np.maximum(1, union).astype(np.float32)


def _generate_sam3_candidate_masks(image_rgb: np.ndarray, cfg: DiagramMaskClusterConfig) -> List[Dict[str, Any]]:
    bundle = _get_sam3_bundle(cfg)
    processor = bundle["processor"]
    model = bundle["model"]

    pil_image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))
    width, height = pil_image.size
    points = _build_point_grid(width, height, int(cfg.point_grid_step), int(cfg.max_points))
    if not points:
        return []

    inference_state = processor.set_image(pil_image)

    proposals: List[Dict[str, Any]] = []
    for prompt_index, prompt_point in enumerate(points):
        mask_logits, score_arr, low_res_logits = model.predict_inst(
            inference_state,
            point_coords=np.asarray([prompt_point], dtype=np.float32),
            point_labels=np.asarray([1], dtype=np.int32),
            multimask_output=True,
            return_logits=True,
            normalize_coords=False,
        )
        mask_arr = np.asarray(mask_logits)
        if mask_arr.ndim == 2:
            mask_arr = mask_arr[None, ...]
        score_arr = np.asarray(score_arr, dtype=np.float32).reshape(-1)
        stability_arr = _compute_stability_from_low_res_logits(
            low_res_logits,
            base_threshold=float(cfg.sam_mask_threshold),
            offset=float(cfg.sam_stability_offset),
        )
        if mask_arr.ndim != 3:
            continue

        for mask_index in range(mask_arr.shape[0]):
            mask = np.asarray(mask_arr[mask_index] > float(cfg.sam_mask_threshold), dtype=bool)
            area_px = _mask_area(mask)
            if area_px < int(cfg.min_region_area_px):
                continue
            pred_iou = float(score_arr[mask_index]) if mask_index < score_arr.shape[0] else 0.0
            stability = float(stability_arr[mask_index]) if mask_index < stability_arr.shape[0] else pred_iou
            if pred_iou < float(cfg.sam_pred_iou_thresh):
                continue
            if stability < float(cfg.sam_stability_score_thresh):
                continue
            bbox = _mask_bbox(mask)
            if not bbox:
                continue
            proposals.append(
                {
                    "prompt_point": list(prompt_point),
                    "proposal_id": f"p{len(proposals):04d}",
                    "mask": mask,
                    "bbox_xyxy": bbox,
                    "mask_area_px": area_px,
                    "sam_pred_iou": pred_iou,
                    "sam_stability_score": stability,
                    "sam_prompt_index": int(prompt_index),
                }
            )
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


def _suppress_redundant_proposals(
    proposals: List[Dict[str, Any]],
    removed_mask: np.ndarray,
    image_rgb: np.ndarray,
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
    image_h, image_w = ann_mask.shape[:2]
    image_area = float(max(1, image_h * image_w))
    for row in work:
        mask = np.asarray(row.get("mask"), dtype=bool)
        area_px = _mask_area(mask)
        if area_px < int(cfg.min_region_area_px):
            continue
        bbox = row.get("bbox_xyxy") or _mask_bbox(mask)
        border_touches = _bbox_border_touch_count(
            bbox,
            image_w,
            image_h,
            int(cfg.proposal_border_touch_margin_px),
        )
        mask_area_ratio = float(area_px) / image_area
        near_white_ratio = _mask_near_white_ratio(
            image_rgb,
            mask,
            int(cfg.proposal_background_prompt_rgb_floor),
        )
        row["mask_area_ratio"] = mask_area_ratio
        row["bbox_border_touches"] = int(border_touches)
        row["near_white_ratio"] = near_white_ratio
        if (
            mask_area_ratio >= float(cfg.proposal_canvas_max_mask_area_ratio)
            and border_touches >= int(cfg.proposal_canvas_min_border_touches)
        ):
            continue
        if (
            mask_area_ratio >= float(cfg.proposal_background_prompt_max_mask_area_ratio)
            and (
                _prompt_is_near_white(
                    image_rgb,
                    row.get("prompt_point"),
                    int(cfg.proposal_background_prompt_rgb_floor),
                )
                or near_white_ratio >= float(cfg.proposal_large_mask_near_white_drop_ratio)
            )
        ):
            continue
        annotation_overlap = float(np.logical_and(mask, ann_mask).sum(dtype=np.int64)) / float(max(1, area_px))
        row["annotation_overlap_ratio"] = annotation_overlap
        if annotation_overlap > float(cfg.annotation_overlap_drop_ratio):
            continue
        drop = False
        for prev in kept:
            iou = _mask_iou(mask, prev["mask"])
            containment = _mask_containment(mask, prev["mask"])
            if iou >= float(cfg.proposal_nms_iou_thresh) or containment >= float(cfg.proposal_containment_thresh):
                drop = True
                break
        if not drop:
            kept.append(row)
    return kept


def _should_merge_pair(a: Dict[str, Any], b: Dict[str, Any], cfg: DiagramMaskClusterConfig) -> Tuple[bool, Dict[str, float]]:
    iou = _mask_iou(a["mask"], b["mask"])
    containment = _mask_containment(a["mask"], b["mask"])
    bbox_gap = _bbox_gap_px(a["bbox_xyxy"], b["bbox_xyxy"])
    shared_bbox = _bbox_union(a["bbox_xyxy"], b["bbox_xyxy"])
    smaller_bbox_area = min(_bbox_area_xyxy(a["bbox_xyxy"]), _bbox_area_xyxy(b["bbox_xyxy"]))
    smaller_bbox_span = math.sqrt(max(1.0, smaller_bbox_area))
    shared_growth = (
        max(0.0, _bbox_area_xyxy(shared_bbox) - smaller_bbox_area) / float(max(1.0, smaller_bbox_area))
        if smaller_bbox_area > 0
        else 999.0
    )
    dino_cos = 1.0 - float(cosine(_normalize_vec(a["dino_embedding"]), _normalize_vec(b["dino_embedding"])))
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

    # SAM3 proposals are denser than the old SAM2 path and can chain-merge
    # large parts of the canvas when gap-based thresholds are too loose.
    # Clamp the effective merge window by object scale so permissive old
    # tuning values do not collapse many unrelated masks into one cluster.
    effective_merge_iou_thresh = max(float(cfg.merge_iou_thresh), 0.08)
    effective_merge_containment_thresh = max(float(cfg.merge_containment_thresh), 0.90)
    effective_merge_bbox_gap_px = min(
        float(cfg.merge_bbox_gap_px),
        max(12.0, min(32.0, 0.60 * smaller_bbox_span)),
    )
    effective_similarity_only_bbox_gap_px = min(
        float(cfg.merge_similarity_only_bbox_gap_px),
        max(24.0, min(64.0, 1.80 * smaller_bbox_span)),
    )
    effective_merge_bbox_growth_max = min(float(cfg.merge_bbox_growth_max), 1.20)

    effective_merge_min_dino_cosine = max(float(cfg.merge_min_dino_cosine), 0.90)
    effective_merge_min_color_similarity = max(float(cfg.merge_min_color_similarity), 0.55)
    effective_merge_min_shape_similarity = max(float(cfg.merge_min_shape_similarity), 0.60)
    effective_merge_min_contrast_color_similarity = max(float(cfg.merge_min_contrast_color_similarity), 0.60)
    effective_merge_combined_feature_score = max(float(cfg.merge_combined_feature_score), 0.72)
    effective_merge_soft_min_dino_cosine = max(float(cfg.merge_soft_min_dino_cosine), 0.86)
    effective_merge_soft_min_shape_similarity = max(float(cfg.merge_soft_min_shape_similarity), 0.55)
    effective_similarity_only_min_dino_cosine = max(float(cfg.merge_similarity_only_min_dino_cosine), 0.93)
    effective_similarity_only_min_color_similarity = max(float(cfg.merge_similarity_only_min_color_similarity), 0.70)
    effective_similarity_only_min_shape_similarity = max(float(cfg.merge_similarity_only_min_shape_similarity), 0.72)
    effective_similarity_only_min_score = max(float(cfg.merge_similarity_only_min_score), 0.82)

    spatial_ok = (
        iou >= effective_merge_iou_thresh
        or (
            containment >= effective_merge_containment_thresh
            and shared_growth <= effective_merge_bbox_growth_max
        )
        or (bbox_gap <= effective_merge_bbox_gap_px and shared_growth <= effective_merge_bbox_growth_max)
    )
    strict_feature_ok = (
        dino_cos >= effective_merge_min_dino_cosine
        and rgb_color_sim >= effective_merge_min_color_similarity
        and shape_sim >= effective_merge_min_shape_similarity
    )
    soft_feature_ok = (
        dino_cos >= effective_merge_soft_min_dino_cosine
        and shape_sim >= effective_merge_soft_min_shape_similarity
        and contrast_color_sim >= effective_merge_min_contrast_color_similarity
        and combined_feature_score >= effective_merge_combined_feature_score
    )
    similarity_only_ok = (
        dino_cos >= effective_similarity_only_min_dino_cosine
        and color_sim >= effective_similarity_only_min_color_similarity
        and shape_sim >= effective_similarity_only_min_shape_similarity
        and combined_feature_score >= effective_similarity_only_min_score
        and bbox_gap <= effective_similarity_only_bbox_gap_px
    )
    feature_ok = strict_feature_ok or soft_feature_ok
    merge_ok = (spatial_ok and feature_ok) or similarity_only_ok
    return bool(merge_ok), {
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
        "effective_merge_iou_thresh": effective_merge_iou_thresh,
        "effective_merge_containment_thresh": effective_merge_containment_thresh,
        "effective_merge_bbox_gap_px": effective_merge_bbox_gap_px,
        "effective_similarity_only_bbox_gap_px": effective_similarity_only_bbox_gap_px,
        "effective_merge_bbox_growth_max": effective_merge_bbox_growth_max,
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
    uf = _UnionFind(len(proposals))
    pair_debug: List[Dict[str, Any]] = []
    for i in range(len(proposals)):
        for j in range(i + 1, len(proposals)):
            ok, metrics = _should_merge_pair(proposals[i], proposals[j], cfg)
            if ok:
                uf.union(i, j)
                pair_debug.append({"a": proposals[i]["proposal_id"], "b": proposals[j]["proposal_id"], **metrics})
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
                "pipeline": "sam3_dinov2",
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
                "shape_features": _compute_shape_features(union_mask, bbox),
                "color_histogram": _compute_color_histogram(image_rgb[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]], crop_mask, int(cfg.histogram_bins)).tolist(),
                "contrast_color_histogram": _compute_contrast_color_histogram(image_rgb[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]], crop_mask, int(cfg.histogram_bins)).tolist(),
            }
        )
    merged.sort(key=lambda row: (int(row["bbox_xyxy"][0]), int(row["bbox_xyxy"][1])))
    return merged, pair_debug


def _write_outputs_for_processed(
    *,
    idx: int,
    image_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    removed_overlay_rgb: np.ndarray,
    removed_mask: np.ndarray,
    proposals: List[Dict[str, Any]],
    pair_debug: List[Dict[str, Any]],
    merged: List[Dict[str, Any]],
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
        cluster_entries.append(
            {
                "color_id": 0,
                "color_name": "mask",
                "group_index_in_color": int(cluster_index),
                "stroke_indexes": [],
                "bbox_xyxy": [int(v) for v in row["crop_bbox_xyxy"]],
                "crop_file_mask": mask_name,
                "pipeline": "sam3_dinov2",
                "mask_area_px": int(row["mask_area_px"]),
                "sam_stability_score": float(row["sam_stability_score"]),
                "sam_pred_iou": float(row["sam_pred_iou"]),
                "merged_from_mask_ids": list(row["merged_from_mask_ids"]),
                "feature_cluster_id": str(row["feature_cluster_id"]),
            }
        )

    cluster_map = {
        "image_index": int(idx),
        "image_size": [int(image_rgb.shape[1]), int(image_rgb.shape[0])],
        "pipeline": "sam3_dinov2",
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
        "merged_clusters": [
            {
                k: v for k, v in row.items()
                if k not in {"mask", "crop_rgba"}
            }
            for row in merged
        ],
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
        cluster_entries.append(
            {
                "color_id": 0,
                "color_name": "mask",
                "group_index_in_color": int(cluster_index),
                "stroke_indexes": [],
                "bbox_xyxy": [int(v) for v in row["crop_bbox_xyxy"]],
                "crop_file_mask": mask_name,
                "pipeline": "sam3_dinov2",
                "mask_area_px": int(row["mask_area_px"]),
                "sam_stability_score": float(row["sam_stability_score"]),
                "sam_pred_iou": float(row["sam_pred_iou"]),
                "merged_from_mask_ids": list(row["merged_from_mask_ids"]),
                "feature_cluster_id": str(row["feature_cluster_id"]),
            }
        )
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
    cfg = config or build_config()
    cleanup = _preclean_diagram_copy(image_rgb, cfg)
    cleaned_rgb = cleanup["cleaned_rgb"]
    removed_mask = cleanup["removed_mask"]
    proposals = _generate_sam3_candidate_masks(cleaned_rgb, cfg)
    proposals = _suppress_redundant_proposals(proposals, removed_mask, cleaned_rgb, cfg)
    proposals = _attach_dino_and_shape_features(cleaned_rgb, proposals, cfg)
    merged, pair_debug = _merge_component_proposals(image_rgb, proposals, cfg)

    idx = _extract_index_from_processed_name(f"{processed_id}.png")
    out: Dict[str, Any] = {
        "processed_id": str(processed_id),
        "config": cfg.to_public_dict(),
        "cleaned_rgb": cleaned_rgb,
        "removed_mask": removed_mask,
        "removed_overlay_rgb": cleanup["removed_overlay_rgb"],
        "proposals": proposals,
        "merged_clusters": merged,
        "merge_pairs": pair_debug,
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
                merged=merged,
                cfg=cfg,
            )
        )
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
