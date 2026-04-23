from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import ImageClusters


BACKEND_DIR = Path(__file__).resolve().parent
REPORT_ROOT = BACKEND_DIR / "ClusterSplitReports"


def _parse_processed_index(value: str) -> int:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty processed id")
    if text.lower().startswith("processed_"):
        return int(text.split("_", 1)[1])
    return int(text)


def _choose_processed_index(requested: str) -> int:
    if str(requested or "").strip():
        return _parse_processed_index(requested)
    choices: List[int] = []
    for path in ImageClusters._glob_processed_images():
        idx = ImageClusters._extract_index_from_processed_name(path.name)
        if idx is not None:
            choices.append(int(idx))
    if not choices:
        raise RuntimeError("No processed images found under ProccessedImages.")
    return int(random.choice(choices))


def _processed_png_path(idx: int) -> Path:
    want = f"processed_{int(idx)}.png"
    for path in ImageClusters._glob_processed_images():
        if path.name.lower() == want.lower():
            return path
    raise FileNotFoundError(f"Processed image not found: {want}")


def _vector_json_path(idx: int) -> Path:
    path = ImageClusters._vector_json_path_for_index(int(idx))
    if not path.is_file():
        raise FileNotFoundError(f"Stroke vector JSON not found: {path}")
    return path


def _rgb_array_to_png_data_uri(rgb: np.ndarray) -> str:
    arr = np.asarray(rgb, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_path_to_data_uri(path: Path) -> str:
    with Image.open(path).convert("RGB") as img:
        return _rgb_array_to_png_data_uri(np.asarray(img, dtype=np.uint8))


def _cluster_overlay_rgb(img_rgb: np.ndarray, clusters: List[Dict[str, Any]]) -> np.ndarray:
    out = np.asarray(img_rgb, dtype=np.uint8).copy()
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
        cv2.putText(
            out,
            str(idx),
            (max(0, x0 + 3), max(14, y0 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def _open_report(path: Path) -> bool:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
            return True
        return bool(webbrowser.open(path.resolve().as_uri()))
    except Exception:
        return False


def _build_cluster_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if bool(args.very_coarse):
        if args.group_radius is None:
            overrides["GROUP_RADIUS_PX"] = 44.0
        if args.grid_cell is None:
            overrides["GRID_CELL_PX"] = 96.0
        if not bool(args.enable_dynamic_split):
            overrides["ENABLE_DYNAMIC_INTERNAL_CLUSTER_SPLIT"] = False
        if args.cross_color_growth_max is None:
            overrides["CROSS_COLOR_SHARED_BBOX_GROWTH_MAX"] = 1.0
        if args.dynamic_split_min_spread_ratio is None:
            overrides["DYNAMIC_SPLIT_MIN_SPREAD_RATIO"] = 9.0
        if args.dynamic_split_bridge_ratio is None:
            overrides["DYNAMIC_SPLIT_BRIDGE_RATIO"] = 2.0
    if bool(args.coarse):
        if args.group_radius is None:
            overrides["GROUP_RADIUS_PX"] = 36.0
        if args.grid_cell is None:
            overrides["GRID_CELL_PX"] = 96.0
        if not bool(args.enable_dynamic_split):
            overrides["ENABLE_DYNAMIC_INTERNAL_CLUSTER_SPLIT"] = False
        if args.cross_color_growth_max is None:
            overrides["CROSS_COLOR_SHARED_BBOX_GROWTH_MAX"] = 0.70
        if args.dynamic_split_min_spread_ratio is None:
            overrides["DYNAMIC_SPLIT_MIN_SPREAD_RATIO"] = 8.0
        if args.dynamic_split_bridge_ratio is None:
            overrides["DYNAMIC_SPLIT_BRIDGE_RATIO"] = 1.9
    if args.group_radius is not None:
        overrides["GROUP_RADIUS_PX"] = float(args.group_radius)
    if args.grid_cell is not None:
        overrides["GRID_CELL_PX"] = float(args.grid_cell)
    if bool(args.enable_dynamic_split):
        overrides["ENABLE_DYNAMIC_INTERNAL_CLUSTER_SPLIT"] = True
    if bool(args.disable_dynamic_split):
        overrides["ENABLE_DYNAMIC_INTERNAL_CLUSTER_SPLIT"] = False
    if args.dynamic_split_min_spread_ratio is not None:
        overrides["DYNAMIC_SPLIT_MIN_SPREAD_RATIO"] = float(args.dynamic_split_min_spread_ratio)
    if args.dynamic_split_bridge_ratio is not None:
        overrides["DYNAMIC_SPLIT_BRIDGE_RATIO"] = float(args.dynamic_split_bridge_ratio)
    if args.dynamic_split_min_strokes is not None:
        overrides["DYNAMIC_SPLIT_MIN_STROKES_FOR_CHECK"] = int(args.dynamic_split_min_strokes)
    if args.dynamic_split_min_component_strokes is not None:
        overrides["DYNAMIC_SPLIT_MIN_COMPONENT_STROKES"] = int(args.dynamic_split_min_component_strokes)
    if bool(args.disable_cross_color_merge):
        overrides["ENABLE_CROSS_COLOR_CLUSTER_MERGE"] = False
    if bool(args.enable_cross_color_merge):
        overrides["ENABLE_CROSS_COLOR_CLUSTER_MERGE"] = True
    if args.cross_color_growth_max is not None:
        overrides["CROSS_COLOR_SHARED_BBOX_GROWTH_MAX"] = float(args.cross_color_growth_max)
    return overrides


def _load_existing_clusters(idx: int) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    map_path = ImageClusters.CLUSTER_MAP_DIR / f"processed_{int(idx)}" / "clusters.json"
    if not map_path.is_file():
        raise FileNotFoundError(f"Existing cluster map not found: {map_path}")
    obj = json.loads(map_path.read_text(encoding="utf-8"))
    clusters = obj.get("clusters") if isinstance(obj, dict) else []
    clusters = clusters if isinstance(clusters, list) else []
    render_dir = ImageClusters.CLUSTER_RENDER_DIR / f"processed_{int(idx)}"
    renders: Dict[str, np.ndarray] = {}
    for row in clusters:
        if not isinstance(row, dict):
            continue
        name = str(row.get("crop_file_mask", "") or "").strip()
        if not name:
            continue
        path = render_dir / name
        if path.is_file():
            with Image.open(path).convert("RGB") as img:
                renders[name] = np.asarray(img, dtype=np.uint8)
    return clusters, renders


def _recluster_processed(idx: int, overrides: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], Dict[str, Any]]:
    img_path = _processed_png_path(int(idx))
    vec_path = _vector_json_path(int(idx))
    img_rgb = ImageClusters._load_rgb_image(img_path)
    vec = json.loads(vec_path.read_text(encoding="utf-8"))
    preproc_by_idx = {
        int(idx): {
            "cleaned_bgr": cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        }
    }
    vectors_by_idx = {int(idx): vec}
    prev = ImageClusters.apply_runtime_cluster_settings(**overrides)
    try:
        settings_used = ImageClusters.get_runtime_cluster_settings()
        result = ImageClusters.cluster_in_memory(preproc_by_idx, vectors_by_idx, save_outputs=False)
    finally:
        ImageClusters.restore_runtime_cluster_settings(prev)
    row = result.get(int(idx)) if isinstance(result, dict) else {}
    if not isinstance(row, dict):
        raise RuntimeError(f"Clustering produced no result for processed_{int(idx)}")
    clusters = row.get("clusters") if isinstance(row.get("clusters"), list) else []
    renders = row.get("renders_mask_rgb") if isinstance(row.get("renders_mask_rgb"), dict) else {}
    return clusters, renders, settings_used


def _write_report(
    *,
    report_path: Path,
    processed_id: str,
    source_image_path: Path,
    image_rgb: np.ndarray,
    clusters: List[Dict[str, Any]],
    renders_mask_rgb: Dict[str, np.ndarray],
    settings_used: Dict[str, Any],
    reclustered: bool,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_rgb = _cluster_overlay_rgb(image_rgb, clusters)
    src_uri = _image_path_to_data_uri(source_image_path)
    overlay_uri = _rgb_array_to_png_data_uri(overlay_rgb)
    settings_pretty = json.dumps(settings_used, ensure_ascii=False, indent=2)

    cards: List[str] = []
    for pos, row in enumerate(clusters, start=1):
        if not isinstance(row, dict):
            continue
        mask_name = str(row.get("crop_file_mask", "") or "").strip()
        arr = renders_mask_rgb.get(mask_name)
        if arr is None:
            continue
        bbox = row.get("bbox_xyxy") if isinstance(row.get("bbox_xyxy"), list) else []
        bbox_text = ", ".join(str(int(v)) for v in bbox) if bbox else "(none)"
        stroke_count = len(row.get("stroke_indexes") or []) if isinstance(row.get("stroke_indexes"), list) else 0
        data_uri = _rgb_array_to_png_data_uri(arr)
        cards.append(
            "<section class=\"cluster-card\">"
            f"<div class=\"cluster-image\"><img src=\"{data_uri}\" alt=\"cluster {pos}\"></div>"
            "<div class=\"cluster-meta\">"
            f"<h2>#{pos} {row.get('color_name', 'cluster')}</h2>"
            f"<p><strong>Mask:</strong> {mask_name}</p>"
            f"<p><strong>Color:</strong> {row.get('color_name', '(none)')}</p>"
            f"<p><strong>Stroke count:</strong> {stroke_count}</p>"
            f"<p><strong>BBox:</strong> {bbox_text}</p>"
            f"<p><strong>Group index:</strong> {row.get('group_index_in_color', '(none)')}</p>"
            "</div>"
            "</section>"
        )

    html = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>Cluster Split Report - {processed_id}</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:0;background:#f4f4f6;color:#111;}"
        ".page{max-width:1400px;margin:0 auto;padding:24px;}"
        ".hero{display:grid;grid-template-columns:1fr;gap:18px;margin-bottom:24px;}"
        ".hero-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px;}"
        ".panel{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;}"
        ".panel img{width:100%;height:auto;display:block;border-radius:6px;background:#fff;}"
        ".panel pre{white-space:pre-wrap;word-break:break-word;background:#f8f8f8;border:1px solid #e5e5e5;padding:12px;border-radius:6px;}"
        ".clusters{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;}"
        ".cluster-card{background:#fff;border:1px solid #ddd;border-radius:8px;overflow:hidden;display:grid;grid-template-columns:1fr;}"
        ".cluster-image{background:#fff;padding:12px;border-bottom:1px solid #eee;}"
        ".cluster-image img{width:100%;height:auto;display:block;}"
        ".cluster-meta{padding:14px;}"
        ".cluster-meta h2{font-size:18px;margin:0 0 10px 0;}"
        ".cluster-meta p{margin:6px 0;word-break:break-word;}"
        ".muted{color:#555;}"
        "</style></head><body>"
        "<div class=\"page\">"
        f"<h1>Cluster Split Report - {processed_id}</h1>"
        f"<p class=\"muted\">Mode: {'reclustered in memory' if reclustered else 'existing ClusterMaps/ClusterRenders'} | Clusters: {len(clusters)}</p>"
        "<div class=\"hero\">"
        "<div class=\"hero-grid\">"
        f"<section class=\"panel\"><h2>Source image</h2><img src=\"{src_uri}\" alt=\"source image\"></section>"
        f"<section class=\"panel\"><h2>Cluster overlay</h2><img src=\"{overlay_uri}\" alt=\"cluster overlay\"></section>"
        "</div>"
        f"<section class=\"panel\"><h2>Settings used</h2><pre>{settings_pretty}</pre></section>"
        "</div>"
        f"<div class=\"clusters\">{''.join(cards) if cards else '<p>No clusters were produced.</p>'}</div>"
        "</div></body></html>"
    )
    report_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an HTML report that shows the source image and the cluster crops, without any labeling stage."
    )
    parser.add_argument("--processed-id", default="", help="Processed image id, like processed_1. If omitted, picks a random one.")
    parser.add_argument("--no-recluster", action="store_true", help="Visualize the existing ClusterMaps/ClusterRenders instead of reclustering in memory.")
    parser.add_argument("--coarse", action="store_true", help="Apply a strong coarse clustering preset: much larger grouping radius, dynamic split off, looser cross-colour merge.")
    parser.add_argument("--very-coarse", action="store_true", help="Apply an even more aggressive merge preset to force substantially fewer clusters.")
    parser.add_argument("--group-radius", type=float, default=None, help="Grouping radius in pixels for same-colour stroke clustering.")
    parser.add_argument("--grid-cell", type=float, default=None, help="Spatial hash cell size in pixels.")
    parser.add_argument("--enable-dynamic-split", action="store_true", help="Force the dynamic internal split pass on.")
    parser.add_argument("--disable-dynamic-split", action="store_true", help="Force the dynamic internal split pass off.")
    parser.add_argument("--dynamic-split-min-strokes", type=int, default=None, help="Min stroke count before the dynamic split pass is allowed to fire.")
    parser.add_argument("--dynamic-split-min-spread-ratio", type=float, default=None, help="How spread out a cluster must be before split is considered.")
    parser.add_argument("--dynamic-split-bridge-ratio", type=float, default=None, help="How strong a bridge separation must be before split is accepted.")
    parser.add_argument("--dynamic-split-min-component-strokes", type=int, default=None, help="Minimum stroke count per split piece.")
    parser.add_argument("--enable-cross-color-merge", action="store_true", help="Force cross-colour bbox merge on.")
    parser.add_argument("--disable-cross-color-merge", action="store_true", help="Force cross-colour bbox merge off.")
    parser.add_argument("--cross-color-growth-max", type=float, default=None, help="Max shared-bbox growth allowed when merging clusters across colours.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the generated report automatically.")
    args = parser.parse_args()

    idx = _choose_processed_index(str(args.processed_id))
    processed_id = f"processed_{int(idx)}"
    image_path = _processed_png_path(int(idx))
    image_rgb = ImageClusters._load_rgb_image(image_path)

    settings_used = ImageClusters.get_runtime_cluster_settings()
    if bool(args.no_recluster):
        clusters, renders_mask_rgb = _load_existing_clusters(int(idx))
        reclustered = False
    else:
        overrides = _build_cluster_overrides(args)
        clusters, renders_mask_rgb, settings_used = _recluster_processed(int(idx), overrides)
        reclustered = True

    out_dir = REPORT_ROOT / processed_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "cluster_split_report.html"
    meta_path = out_dir / "cluster_split_report.json"

    _write_report(
        report_path=report_path,
        processed_id=processed_id,
        source_image_path=image_path,
        image_rgb=image_rgb,
        clusters=clusters,
        renders_mask_rgb=renders_mask_rgb,
        settings_used=settings_used,
        reclustered=reclustered,
    )
    meta_path.write_text(
        json.dumps(
            {
                "processed_id": processed_id,
                "source_image_path": str(image_path),
                "cluster_count": len(clusters),
                "reclustered": bool(reclustered),
                "settings_used": settings_used,
                "clusters": clusters,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"processed_id: {processed_id}")
    print(f"source_image: {image_path.resolve()}")
    print(f"cluster_count: {len(clusters)}")
    print(f"report_json: {meta_path.resolve()}")
    print(f"report_html: {report_path.resolve()}")
    if not bool(args.no_open):
        opened = _open_report(report_path)
        print(f"report_opened: {opened}")


if __name__ == "__main__":
    main()
