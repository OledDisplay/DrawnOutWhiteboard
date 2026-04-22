#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import webbrowser
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from PIL import Image


BACKEND_DIR = Path(__file__).resolve().parent
PROCESSED_ID_RE = re.compile(r"^processed_(\d+)$", re.IGNORECASE)
DEFAULT_WORKER = "http://127.0.0.1:8009"


def _clean_labels(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        for part in str(value or "").split(","):
            label = part.strip()
            if not label:
                continue
            key = label.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(label)
    return out


def _parse_processed_index(processed_id: str) -> int:
    match = PROCESSED_ID_RE.match(str(processed_id or "").strip())
    if not match:
        raise ValueError(f"Bad processed id: {processed_id!r}. Expected something like processed_1.")
    return int(match.group(1))


def _choose_processed_index(qwentest: Any, requested: str) -> int:
    if requested:
        return _parse_processed_index(requested)

    indices = qwentest.find_indices_from_cluster_maps()
    if not indices:
        raise RuntimeError(
            "No ClusterMaps/processed_N/clusters.json files found. "
            "Run image vectorization/clustering first, or pass an existing --processed-id."
        )
    return int(random.choice(indices))


def _open_report(path: Path) -> bool:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
            return True
        return bool(webbrowser.open(path.resolve().as_uri()))
    except Exception:
        return False


def _labels_from_c2(
    *,
    depicts: str,
    image_name: str,
    processed_id: str,
    worker: str,
    steps: int,
    limit: int,
    timeout: int,
    output_root: Path,
    visual_component_batch_size: int,
    skip_visual_stage: bool,
) -> List[str]:
    from C2.Orchestrator import run_orchestrator_bundle

    prompt = str(depicts or image_name or processed_id).strip()
    if not prompt:
        return []

    result = run_orchestrator_bundle(
        prompts=[
            {
                "prompt": prompt,
                "qid": "",
                "diagram": 1,
                "processed_id": processed_id,
                "image_name": image_name or processed_id,
                "requested_components": [],
                "diagram_required_objects": [],
            }
        ],
        worker_url=str(worker).rstrip("/"),
        mode="normal",
        steps=int(steps),
        limit=int(limit),
        timeout=int(timeout),
        output_root=str(output_root),
        visual_component_batch_size=int(visual_component_batch_size),
        skip_visual_stage=bool(skip_visual_stage),
        max_workers=1,
    )

    reports = result.get("prompts") if isinstance(result, dict) else {}
    report = reports.get(prompt) if isinstance(reports, dict) else {}
    if not isinstance(report, dict):
        return []

    labels: List[str] = []
    for key in ("accepted_components", "requested_components_resolved"):
        rows = report.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or row.get("name") or row.get("component") or "").strip()
            if label:
                labels.append(label)

    visual_stage = report.get("visual_stage") if isinstance(report.get("visual_stage"), dict) else {}
    components = visual_stage.get("components") if isinstance(visual_stage, dict) else []
    if isinstance(components, list):
        for row in components:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or row.get("name") or "").strip()
            if not label and isinstance(row.get("component"), dict):
                label = str(row["component"].get("label") or row["component"].get("name") or "").strip()
            if label:
                labels.append(label)

    return _clean_labels(labels)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run C2 label discovery plus the Qwen cluster labeler for one already-processed image and open the visual HTML report."
    )
    parser.add_argument(
        "--processed-id",
        default="",
        help="Processed image id, for example processed_1. If omitted, picks a random processed image with clusters.",
    )
    parser.add_argument(
        "--image-name",
        default="",
        help="Human name for the image, for example 'cell diagram'. Used only as context in the test output.",
    )
    parser.add_argument(
        "--depicts",
        default="",
        help="What the image depicts, for example 'eukaryotic cell diagram'. This is sent to C2 to discover labels.",
    )
    parser.add_argument(
        "--labels",
        action="append",
        default=[],
        help="Optional override: comma-separated labels to match against clusters. If omitted, labels come from C2.",
    )
    parser.add_argument(
        "--base-context",
        default="",
        help="Optional context override for the cluster labeler. Defaults to --depicts, then processed JSON context.",
    )
    parser.add_argument("--worker", default=DEFAULT_WORKER, help="C2 Qwen worker base URL.")
    parser.add_argument("--c2-steps", type=int, default=4, help="C2 orchestrator agent cycles.")
    parser.add_argument("--c2-limit", type=int, default=8, help="C2 Wikidata scan row limit.")
    parser.add_argument("--c2-timeout", type=int, default=30, help="C2 HTTP timeout.")
    parser.add_argument("--visual-component-batch-size", type=int, default=4)
    parser.add_argument("--skip-visual-stage", action="store_true", help="Skip C2 visual descriptions; still uses C2 component discovery.")
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Only print the report path; do not try to open it.",
    )
    parser.add_argument(
        "--batch-size",
        default="1",
        help="Qwen batch size. Default is 1 for lower VRAM.",
    )
    parser.add_argument(
        "--proc-longest-edge",
        default="600",
        help="Qwen processed composite image edge. Default 600.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # qwentest reads these at import time, so set CLI defaults before importing it.
    os.environ.setdefault("QWEN_BATCH_SIZE", str(args.batch_size))
    os.environ.setdefault("QWEN_PROC_LONGEST_EDGE", str(args.proc_longest_edge))
    sys.path.insert(0, str(BACKEND_DIR))
    import qwentest  # noqa: WPS433

    idx = _choose_processed_index(qwentest, str(args.processed_id or "").strip())
    pid = f"processed_{idx}"

    img_index, json_index = qwentest.ensure_indexes()
    full_image_path_s: Optional[str] = img_index.get(str(idx))
    if not full_image_path_s:
        raise FileNotFoundError(f"Could not find ProccessedImages/{pid}.png in qwentest's processed image index.")
    full_image_path = Path(full_image_path_s)
    if not full_image_path.is_file():
        raise FileNotFoundError(full_image_path)

    cluster_map_path = qwentest.cluster_map_path_for_idx(idx)
    renders_dir = qwentest.cluster_renders_dir_for_idx(idx)
    if not cluster_map_path.is_file():
        raise FileNotFoundError(cluster_map_path)
    if not renders_dir.is_dir():
        raise FileNotFoundError(renders_dir)

    cluster_map = json.loads(cluster_map_path.read_text(encoding="utf-8"))
    clusters = cluster_map.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        raise RuntimeError(f"{cluster_map_path} has no clusters.")

    auto_labels, auto_base_context = qwentest.load_candidate_labels(idx, json_index)
    depicts = str(args.depicts or "").strip()
    image_name = str(args.image_name or depicts or pid).strip()
    labels = _clean_labels(list(args.labels or []))
    if labels:
        label_source = "--labels"
    else:
        if not depicts:
            depicts = str(args.base_context or auto_base_context or image_name).strip()
        if not depicts:
            raise RuntimeError("Pass --depicts \"what the image shows\" so C2 can discover labels.")
        print(f"[c2] discovering labels for {pid}: {depicts}")
        labels = _labels_from_c2(
            depicts=depicts,
            image_name=image_name,
            processed_id=pid,
            worker=str(args.worker),
            steps=int(args.c2_steps),
            limit=int(args.c2_limit),
            timeout=int(args.c2_timeout),
            output_root=BACKEND_DIR / "PipelineOutputs" / "cluster_label_report_c2",
            visual_component_batch_size=int(args.visual_component_batch_size),
            skip_visual_stage=bool(args.skip_visual_stage),
        )
        label_source = "c2"
    if not labels:
        labels = _clean_labels(auto_labels)
        label_source = "processed_json_words"
    if not labels:
        raise RuntimeError(
            "C2 did not return labels and no processed JSON labels were available. "
            "Either make sure the C2 worker is running, or pass --labels as a manual override."
        )

    base_context = str(args.base_context or depicts or auto_base_context or image_name or pid).strip()

    renders_mask_rgb = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        mask_name = str(cluster.get("crop_file_mask", "") or "").strip()
        if not mask_name:
            continue
        mask_path = renders_dir / mask_name
        if mask_path.is_file():
            renders_mask_rgb[mask_name] = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint8)

    if not renders_mask_rgb:
        raise RuntimeError(f"No cluster render PNGs were found in {renders_dir}.")

    print(f"[load] Qwen model for {pid}...")
    model, processor, device, used_quant = qwentest.load_model_and_processor()
    print(f"[load] done device={device} used_quant={used_quant}")
    print(f"[labels] source={label_source} count={len(labels)} labels={', '.join(labels[:40])}")

    full_img_rgb = np.asarray(Image.open(full_image_path).convert("RGB"), dtype=np.uint8)
    result_by_idx = qwentest.label_clusters_transformers(
        {
            idx: {
                "base_context": base_context,
                "candidate_labels_raw": labels,
                "candidate_labels_refined": labels,
                "full_img_rgb": full_img_rgb,
                "clusters": clusters,
                "renders_mask_rgb": renders_mask_rgb,
            }
        },
        model=model,
        processor=processor,
        device=device,
        save_outputs=True,
        skip_existing_labels=False,
    )

    result = result_by_idx.get(idx) if isinstance(result_by_idx, dict) else {}
    report_path = Path(str((result or {}).get("cluster_label_report_path") or ""))
    labels_path = BACKEND_DIR / "ClustersLabeled" / pid / "labels.json"
    debug_path = BACKEND_DIR / "ClustersLabeled" / "debug" / pid / "labels.json"

    print("")
    print(f"processed_id: {pid}")
    print(f"image_name: {image_name}")
    print(f"depicts: {depicts or base_context}")
    print(f"label_source: {label_source}")
    print(f"labels_json: {labels_path.resolve()}")
    print(f"debug_json: {debug_path.resolve()}")
    print(f"cluster_label_report: {report_path.resolve() if str(report_path) else '(not written)'}")

    if report_path.is_file() and not args.no_open:
        opened = _open_report(report_path)
        print(f"opened_report: {opened}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
