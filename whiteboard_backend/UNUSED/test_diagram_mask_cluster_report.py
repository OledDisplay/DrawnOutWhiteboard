from __future__ import annotations

import argparse
import html as html_lib
import os
import random
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

import DiagramMaskClusters


BACKEND_DIR = Path(__file__).resolve().parent
REPORT_ROOT = BACKEND_DIR / "ClusterSplitReports"


def _choose_processed_id(requested: str) -> str:
    text = str(requested or "").strip()
    if text:
        if text.lower().startswith("processed_"):
            return text
        return f"processed_{int(text)}"
    choices: List[str] = []
    for path in DiagramMaskClusters._glob_processed_images():
        idx = DiagramMaskClusters._extract_index_from_processed_name(path.name)
        if idx is not None:
            choices.append(f"processed_{int(idx)}")
    if not choices:
        raise RuntimeError("No processed images found under ProccessedImages.")
    return random.choice(choices)


def _open_report(path: Path) -> bool:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
            return True
        return bool(webbrowser.open(path.resolve().as_uri()))
    except Exception:
        return False


def _build_config(args: argparse.Namespace) -> DiagramMaskClusters.DiagramMaskClusterConfig:
    return DiagramMaskClusters.build_config(
        sam_model_id=args.sam_model_id,
        dino_model_id=args.dino_model_id,
        point_grid_step=args.point_grid_step,
        point_batch_size=args.point_batch_size,
        max_points=args.max_points,
        thin_component_max_width=args.thin_component_max_width,
        thin_component_min_aspect=args.thin_component_min_aspect,
        line_component_max_area=args.line_component_max_area,
        line_hough_threshold=args.line_hough_threshold,
        line_min_length_px=args.line_min_length_px,
        line_max_gap_px=args.line_max_gap_px,
        line_hough_thickness_px=args.line_hough_thickness_px,
        sam_pred_iou_thresh=args.sam_pred_iou_thresh,
        sam_stability_score_thresh=args.sam_stability_score_thresh,
        min_region_area_px=args.min_region_area_px,
        annotation_overlap_drop_ratio=args.annotation_overlap_drop_ratio,
        proposal_nms_iou_thresh=args.proposal_nms_iou_thresh,
        proposal_containment_thresh=args.proposal_containment_thresh,
        proposal_canvas_max_mask_area_ratio=args.proposal_canvas_max_mask_area_ratio,
        proposal_canvas_min_border_touches=args.proposal_canvas_min_border_touches,
        proposal_border_touch_margin_px=args.proposal_border_touch_margin_px,
        proposal_background_prompt_max_mask_area_ratio=args.proposal_background_prompt_max_mask_area_ratio,
        proposal_background_prompt_rgb_floor=args.proposal_background_prompt_rgb_floor,
        proposal_large_mask_near_white_drop_ratio=args.proposal_large_mask_near_white_drop_ratio,
        merge_min_dino_cosine=args.merge_min_dino_cosine,
        merge_min_color_similarity=args.merge_min_color_similarity,
        merge_min_shape_similarity=args.merge_min_shape_similarity,
        merge_min_contrast_color_similarity=args.merge_min_contrast_color_similarity,
        merge_combined_feature_score=args.merge_combined_feature_score,
        merge_soft_min_dino_cosine=args.merge_soft_min_dino_cosine,
        merge_soft_min_shape_similarity=args.merge_soft_min_shape_similarity,
        merge_similarity_only_min_dino_cosine=args.merge_similarity_only_min_dino_cosine,
        merge_similarity_only_min_color_similarity=args.merge_similarity_only_min_color_similarity,
        merge_similarity_only_min_shape_similarity=args.merge_similarity_only_min_shape_similarity,
        merge_similarity_only_min_score=args.merge_similarity_only_min_score,
        merge_similarity_only_bbox_gap_px=args.merge_similarity_only_bbox_gap_px,
        merge_iou_thresh=args.merge_iou_thresh,
        merge_containment_thresh=args.merge_containment_thresh,
        merge_bbox_gap_px=args.merge_bbox_gap_px,
        merge_bbox_growth_max=args.merge_bbox_growth_max,
        clear_existing_outputs=True,
    )


def _image_data_uri(arr: np.ndarray) -> str:
    return DiagramMaskClusters._image_to_data_uri(arr)


def _write_report(
    *,
    report_path: Path,
    title: str,
    source_rgb: np.ndarray,
    result: Dict[str, Any],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_rgb = np.asarray(result["cleaned_rgb"], dtype=np.uint8)
    overlay_rgb = DiagramMaskClusters.build_overlay_rgb(source_rgb, result.get("cluster_entries") or [])
    removed_overlay_rgb = np.asarray(result["removed_overlay_rgb"], dtype=np.uint8)
    cfg_pretty = html_lib.escape(str(result.get("config")))
    debug_dir = html_lib.escape(str(result.get("debug_dir") or "(none)"))

    cards: List[str] = []
    clusters = result.get("cluster_entries") or []
    merged_clusters = result.get("merged_clusters") or []
    by_feature = {str(row.get("feature_cluster_id")): row for row in merged_clusters if isinstance(row, dict)}
    render_dir = Path(result.get("render_dir")) if result.get("render_dir") else None
    in_memory_renders = result.get("renders_mask_rgb") if isinstance(result.get("renders_mask_rgb"), dict) else {}

    for pos, row in enumerate(clusters, start=1):
        if not isinstance(row, dict):
            continue
        mask_name = str(row.get("crop_file_mask", "") or "").strip()
        if not mask_name:
            continue
        rgba: Optional[np.ndarray] = None
        if render_dir:
            path = render_dir / mask_name
            if path.is_file():
                rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
        if rgba is None:
            rgb = in_memory_renders.get(mask_name)
            if rgb is None:
                continue
            rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
            rgba[..., :3] = np.asarray(rgb, dtype=np.uint8)
            rgba[..., 3] = 255
        feature_id = str(row.get("feature_cluster_id", "") or "")
        rich = by_feature.get(feature_id, {})
        bbox = row.get("bbox_xyxy") if isinstance(row.get("bbox_xyxy"), list) else []
        bbox_text = ", ".join(str(int(v)) for v in bbox) if bbox else "(none)"
        cards.append(
            "<section class=\"cluster-card\">"
            f"<div class=\"cluster-image\"><img src=\"{_image_data_uri(rgba)}\" alt=\"cluster {pos}\"></div>"
            "<div class=\"cluster-meta\">"
            f"<h2>#{pos} {html_lib.escape(mask_name)}</h2>"
            f"<p><strong>BBox:</strong> {html_lib.escape(bbox_text)}</p>"
            f"<p><strong>Mask area:</strong> {int(row.get('mask_area_px', 0) or 0)}</p>"
            f"<p><strong>SAM pred IoU:</strong> {float(row.get('sam_pred_iou', 0.0) or 0.0):.3f}</p>"
            f"<p><strong>SAM stability:</strong> {float(row.get('sam_stability_score', 0.0) or 0.0):.3f}</p>"
            f"<p><strong>Merged from:</strong> {html_lib.escape(', '.join(rich.get('merged_from_mask_ids', []) or [])) or '(none)'}</p>"
            f"<p><strong>Merge count:</strong> {int(rich.get('merge_count', 1) or 1)}</p>"
            "</div>"
            "</section>"
        )

    html = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{html_lib.escape(title)}</title>"
        "<style>"
        "body{font-family:Segoe UI,Arial,sans-serif;background:#111827;color:#f3f4f6;margin:0;padding:24px;}"
        "h1,h2{margin:0 0 10px 0;} p{margin:6px 0;} .muted{color:#cbd5e1;}"
        ".band{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin:18px 0 28px;}"
        ".panel{background:#1f2937;border:1px solid #334155;border-radius:8px;padding:14px;}"
        ".panel img{width:100%;height:auto;border-radius:6px;background:#fff;}"
        ".clusters{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px;}"
        ".cluster-card{display:grid;grid-template-columns:minmax(160px,220px) 1fr;gap:14px;background:#1f2937;border:1px solid #334155;border-radius:8px;padding:14px;align-items:start;}"
        ".cluster-image img{width:100%;height:auto;border-radius:6px;background:#fff;}"
        "pre{white-space:pre-wrap;background:#0f172a;border-radius:8px;padding:12px;overflow:auto;}"
        "</style></head><body>"
        f"<h1>{html_lib.escape(title)}</h1>"
        f"<p class=\"muted\">Clusters: {len(clusters)} | Raw proposals: {len(result.get('proposals') or [])} | Debug bundle: {debug_dir}</p>"
        "<div class=\"band\">"
        f"<section class=\"panel\"><h2>Original</h2><img src=\"{_image_data_uri(source_rgb)}\"></section>"
        f"<section class=\"panel\"><h2>Cleaned</h2><img src=\"{_image_data_uri(cleaned_rgb)}\"></section>"
        f"<section class=\"panel\"><h2>Cleanup Overlay</h2><img src=\"{_image_data_uri(removed_overlay_rgb)}\"></section>"
        f"<section class=\"panel\"><h2>Final Overlay</h2><img src=\"{_image_data_uri(overlay_rgb)}\"></section>"
        "</div>"
        "<section class=\"panel\"><h2>Config</h2>"
        f"<pre>{cfg_pretty}</pre></section>"
        "<h2 style=\"margin-top:28px;\">Final Clusters</h2>"
        f"<div class=\"clusters\">{''.join(cards) or '<p>No clusters produced.</p>'}</div>"
        "</body></html>"
    )
    report_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SAM3 + DINOv2 diagram clustering backend and write an HTML split report.")
    parser.add_argument("--processed-id", default="", help="processed_N or N")
    parser.add_argument("--image-path", default="", help="Optional arbitrary image path instead of processed_N.")
    parser.add_argument("--sam-model-id", default=None)
    parser.add_argument("--dino-model-id", default=None)
    parser.add_argument("--point-grid-step", type=int, default=None)
    parser.add_argument("--point-batch-size", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--thin-component-max-width", type=int, default=None)
    parser.add_argument("--thin-component-min-aspect", type=float, default=None)
    parser.add_argument("--line-component-max-area", type=int, default=None)
    parser.add_argument("--line-hough-threshold", type=int, default=None)
    parser.add_argument("--line-min-length-px", type=int, default=None)
    parser.add_argument("--line-max-gap-px", type=int, default=None)
    parser.add_argument("--line-hough-thickness-px", type=int, default=None)
    parser.add_argument("--sam-pred-iou-thresh", type=float, default=None)
    parser.add_argument("--sam-stability-score-thresh", type=float, default=None)
    parser.add_argument("--min-region-area-px", type=int, default=None)
    parser.add_argument("--annotation-overlap-drop-ratio", type=float, default=None)
    parser.add_argument("--proposal-nms-iou-thresh", type=float, default=None)
    parser.add_argument("--proposal-containment-thresh", type=float, default=None)
    parser.add_argument("--proposal-canvas-max-mask-area-ratio", type=float, default=None)
    parser.add_argument("--proposal-canvas-min-border-touches", type=int, default=None)
    parser.add_argument("--proposal-border-touch-margin-px", type=int, default=None)
    parser.add_argument("--proposal-background-prompt-max-mask-area-ratio", type=float, default=None)
    parser.add_argument("--proposal-background-prompt-rgb-floor", type=int, default=None)
    parser.add_argument("--proposal-large-mask-near-white-drop-ratio", type=float, default=None)
    parser.add_argument("--merge-min-dino-cosine", type=float, default=None)
    parser.add_argument("--merge-min-color-similarity", type=float, default=None)
    parser.add_argument("--merge-min-shape-similarity", type=float, default=None)
    parser.add_argument("--merge-min-contrast-color-similarity", type=float, default=None)
    parser.add_argument("--merge-combined-feature-score", type=float, default=None)
    parser.add_argument("--merge-soft-min-dino-cosine", type=float, default=None)
    parser.add_argument("--merge-soft-min-shape-similarity", type=float, default=None)
    parser.add_argument("--merge-similarity-only-min-dino-cosine", type=float, default=None)
    parser.add_argument("--merge-similarity-only-min-color-similarity", type=float, default=None)
    parser.add_argument("--merge-similarity-only-min-shape-similarity", type=float, default=None)
    parser.add_argument("--merge-similarity-only-min-score", type=float, default=None)
    parser.add_argument("--merge-similarity-only-bbox-gap-px", type=float, default=None)
    parser.add_argument("--merge-iou-thresh", type=float, default=None)
    parser.add_argument("--merge-containment-thresh", type=float, default=None)
    parser.add_argument("--merge-bbox-gap-px", type=float, default=None)
    parser.add_argument("--merge-bbox-growth-max", type=float, default=None)
    parser.add_argument("--open", action="store_true", help="Open the HTML report when done.")
    args = parser.parse_args()

    cfg = _build_config(args)

    if str(args.image_path or "").strip():
        image_path = Path(str(args.image_path)).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        processed_id = image_path.stem
        result = DiagramMaskClusters.cluster_image_rgb(
            image_rgb,
            processed_id=processed_id,
            save_outputs=False,
            config=cfg,
        )
        report_dir = REPORT_ROOT / processed_id
        report_path = report_dir / "diagram_mask_cluster_report.html"
    else:
        processed_id = _choose_processed_id(args.processed_id)
        image_path = DiagramMaskClusters._resolve_processed_png_path(processed_id)
        image_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        result = DiagramMaskClusters.ensure_processed_clusters(
            processed_id,
            save_outputs=True,
            config=cfg,
        )
        report_dir = REPORT_ROOT / processed_id
        report_path = report_dir / "diagram_mask_cluster_report.html"

    _write_report(
        report_path=report_path,
        title=f"Diagram Mask Cluster Report - {processed_id}",
        source_rgb=image_rgb,
        result=result,
    )

    print(f"[done] report: {report_path}")
    if bool(args.open):
        opened = _open_report(report_path)
        print(f"[done] opened={opened}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
