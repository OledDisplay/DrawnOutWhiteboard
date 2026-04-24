#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests


BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_WORKER = "http://127.0.0.1:8009"
DEFAULT_MODEL = "cyankiwi/Qwen3.5-4B-AWQ-4bit"
DEFAULT_EUKARYOTIC_CELL_QID = "Q3307661"
LOG = logging.getLogger("c2_test")


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path.resolve())


def _clean_list(items: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items or []:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _slug(text: str, fallback: str = "prompt") -> str:
    value = re.sub(r"[^\w.-]+", "_", str(text or "").strip(), flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("._")
    return (value or fallback)[:96]


def _infer_required_objects(base_context: str) -> List[str]:
    text = str(base_context or "")
    if not text:
        return []

    lowered = text.lower()
    if "eukaryotic cell" in lowered:
        known = [
            "nucleus",
            "mitochondria",
            "chloroplasts",
            "rough ER",
            "smooth ER",
            "Golgi",
            "secretory vesicles",
            "lysosomes",
            "peroxisomes",
            "ribosomes (80S)",
            "cytoskeleton",
            "centrosome/centrioles",
            "cilia/flagella",
            "extracellular matrix",
        ]
        return [item for item in known if item.casefold().replace(" (80s)", "") in lowered or item in known]

    match = re.search(r"\bshowing\b(.+)", text, flags=re.IGNORECASE)
    tail = match.group(1) if match else text
    tail = re.split(r"[.;:]", tail, maxsplit=1)[0]
    parts = re.split(r",|\band\b|/|\(|\)", tail)
    banned = {
        "large",
        "labeled",
        "diagram",
        "showing",
        "with",
        "and",
        "the",
        "a",
        "an",
    }
    out = []
    for part in parts:
        value = re.sub(r"\s+", " ", part).strip(" -")
        if not value:
            continue
        if value.casefold() in banned:
            continue
        if len(value) < 3:
            continue
        out.append(value)
    return _clean_list(out)[:80]


def _resolve_processed_paths(processed_id: str) -> Dict[str, Path]:
    pid = str(processed_id or "").strip()
    if not pid:
        raise ValueError("processed id is empty")
    return {
        "json": BACKEND_DIR / "ProccessedImages" / f"{pid}.json",
        "png": BACKEND_DIR / "ProccessedImages" / f"{pid}.png",
        "stroke_descriptions": BACKEND_DIR / "StrokeDescriptions" / f"{pid}_described.json",
        "cluster_map": BACKEND_DIR / "ClusterMaps" / pid / "clusters.json",
    }


def _check_worker(worker_url: str, timeout: int) -> Dict[str, Any]:
    base = worker_url.rstrip("/")
    health = requests.get(f"{base}/health", timeout=timeout)
    health.raise_for_status()
    status = requests.get(f"{base}/status", timeout=timeout)
    status.raise_for_status()
    return {"health": health.json(), "status": status.json()}


def _load_worker(
    *,
    worker_url: str,
    model: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    timeout: int,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "max_model_len": int(max_model_len),
        "warmup": True,
    }
    response = requests.post(f"{worker_url.rstrip('/')}/load/text", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _build_component_artifact(
    *,
    out_dir: Path,
    prompt: str,
    diagram_mode: int,
    requested_components: List[str],
    report: Dict[str, Any],
) -> Dict[str, Any]:
    visual_stage = report.get("visual_stage") if isinstance(report, dict) else {}
    components_raw = visual_stage.get("components") if isinstance(visual_stage, dict) else []
    if not isinstance(components_raw, list):
        components_raw = []

    components: List[Dict[str, Any]] = []
    labels: List[str] = []
    seen = set()

    for row in components_raw:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label", "") or "").strip()
        if not label and isinstance(row.get("component"), dict):
            label = str(row["component"].get("label", "") or "").strip()
        if not label:
            continue
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
        components.append(
            {
                "name": label,
                "qid": str(row.get("qid", "") or "").strip(),
                "query": str(row.get("query", row.get("search_query", "")) or "").strip(),
                "wikipedia_visual_description": str(row.get("wikipedia_visual_description", "") or "").strip(),
                "refined_visual_description": str(row.get("refined_visual_description", "") or "").strip(),
                "component_dir": str(row.get("component_dir", "") or "").strip(),
                "json_path": str(row.get("json_path", "") or "").strip(),
                "error": str(row.get("error", "") or "").strip(),
            }
        )

    for label in requested_components:
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
        components.append(
            {
                "name": label,
                "qid": "",
                "query": label,
                "wikipedia_visual_description": "",
                "refined_visual_description": "",
                "component_dir": "",
                "json_path": "",
                "error": "requested_component_fallback",
            }
        )

    artifact = {
        "schema": "c2_label_process_test_components_v1",
        "prompt": prompt,
        "diagram_mode": int(diagram_mode or 0),
        "requested_components": list(requested_components),
        "objects": list(labels),
        "refined_labels": list(labels),
        "components": components,
        "c2_report": report,
    }
    artifact_path = out_dir / "c2_research" / "c2_component_report.json"
    _write_json(artifact_path, artifact)
    return {"path": str(artifact_path.resolve()), "artifact": artifact}


def _write_compact_outputs(
    *,
    out_dir: Path,
    processed_id: str,
    labels: List[str],
    full_artifact_path: str,
) -> Dict[str, str]:
    full_map = {
        "schema": "c2_label_process_test_label_stroke_map_v1",
        "processed_id": processed_id,
        "parser_kind": "c2_research_only",
        "status": "ok",
        "matched_labels": 0,
        "refined_labels": list(labels),
        "refined_labels_file": full_artifact_path,
        "refined_label_to_strokes": {
            label: {
                "stroke_indexes": [],
                "source_type": "not_run",
                "reason": "This test script runs C2 component research; stroke matching is a separate VLM stage.",
            }
            for label in labels
        },
    }
    compact = {
        "processed_id": processed_id,
        "labels": [
            {"label": label, "matched_stroke_count": 0, "stroke_indexes": []}
            for label in labels
        ],
    }
    final_dir = out_dir / "final_outputs"
    return {
        "full_final_map": _write_json(final_dir / f"{processed_id}.label_stroke_map.full.json", full_map),
        "compact_final_map": _write_json(final_dir / f"{processed_id}.part_stroke_map.compact.json", compact),
    }


def _configure_logging(out_dir: Path, level_name: str) -> str:
    log_dir = out_dir / "debug"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "c2_label_process_test.log"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, str(level_name).upper(), logging.INFO))
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(console_handler)
    return str(log_path.resolve())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a C2 label-process test against one processed lesson image.")
    parser.add_argument("--processed-id", default="processed_1", help="Processed image id, for example processed_1.")
    parser.add_argument("--prompt", default="", help="Override the prompt/base_context from the processed JSON.")
    parser.add_argument(
        "--target-qid",
        default=os.environ.get("C2_TEST_TARGET_QID", DEFAULT_EUKARYOTIC_CELL_QID),
        help="Explicit Wikidata target QID. Default anchors processed_1 to eukaryotic cell (Q3307661).",
    )
    parser.add_argument(
        "--no-target-qid",
        action="store_true",
        help="Disable explicit target anchoring and let C2/Qwen resolve the target from the prompt.",
    )
    parser.add_argument(
        "--required",
        action="append",
        default=[],
        help="Required component label. Repeat this flag, or pass comma-separated labels.",
    )
    parser.add_argument("--worker", default=DEFAULT_WORKER, help="Qwen vLLM worker base URL.")
    parser.add_argument("--load-worker", action="store_true", help="Load/warm the text worker before running C2.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id to load when --load-worker is used.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80, help="vLLM GPU memory utilization.")
    parser.add_argument("--max-model-len", type=int, default=8192, help="vLLM max model length.")
    parser.add_argument("--steps", type=int, default=4, help="C2 orchestrator agent cycles.")
    parser.add_argument("--limit", type=int, default=8, help="C2 Wikidata scan row limit.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout for Wikidata/worker checks.")
    parser.add_argument("--worker-timeout", type=int, default=240, help="Timeout for Qwen worker generation calls.")
    parser.add_argument("--visual-component-batch-size", type=int, default=4)
    parser.add_argument("--skip-visual-stage", action="store_true", help="Skip C2's web visual-description stage.")
    parser.add_argument("--output-root", default=str(BACKEND_DIR / "PipelineOutputs"))
    parser.add_argument("--log-level", default="INFO", help="Console log level. File logs are always DEBUG.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    processed_id = str(args.processed_id or "").strip()
    target_qid = "" if args.no_target_qid else str(args.target_qid or "").strip()
    paths = _resolve_processed_paths(processed_id)
    if not paths["json"].exists():
        raise FileNotFoundError(paths["json"])
    if not paths["png"].exists():
        raise FileNotFoundError(paths["png"])

    processed = _load_json(paths["json"])
    prompt = str(args.prompt or processed.get("base_context", "") or processed_id).strip()
    required_from_args: List[str] = []
    for chunk in args.required or []:
        required_from_args.extend([part.strip() for part in str(chunk).split(",") if part.strip()])
    required = _clean_list(required_from_args)
    if not required:
        raw = processed.get("diagram_required_objects")
        required = _clean_list(raw if isinstance(raw, list) else [])
    if not required:
        required = _infer_required_objects(prompt)

    out_dir = Path(args.output_root) / f"c2_label_process_test_{processed_id}_{_now_stamp()}"
    log_path = _configure_logging(out_dir, args.log_level)
    LOG.info("C2 label-process test starting processed_id=%s output=%s", processed_id, out_dir.resolve())
    LOG.info("Sample paths json=%s png=%s strokes=%s clusters=%s", paths["json"], paths["png"], paths["stroke_descriptions"], paths["cluster_map"])
    LOG.info("Prompt chars=%s target_qid=%s requested_components=%s", len(prompt), target_qid or "(auto)", required)

    source_dir = out_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths["json"], source_dir / paths["json"].name)
    shutil.copy2(paths["png"], source_dir / paths["png"].name)
    if paths["stroke_descriptions"].exists():
        shutil.copy2(paths["stroke_descriptions"], source_dir / paths["stroke_descriptions"].name)
    if paths["cluster_map"].exists():
        shutil.copy2(paths["cluster_map"], source_dir / "clusters.json")

    qwen_stage_io_dir = out_dir / "qwen_stage_io"
    os.environ["QWEN_STAGE_IO_DIR"] = str(qwen_stage_io_dir)
    os.environ["QWEN_VLLM_SERVER_URL"] = str(args.worker).rstrip("/")

    worker_load_result: Optional[Dict[str, Any]] = None
    if args.load_worker:
        LOG.info("Loading Qwen worker url=%s model=%s gpu_memory_utilization=%s max_model_len=%s", args.worker, args.model, args.gpu_memory_utilization, args.max_model_len)
        worker_load_result = _load_worker(
            worker_url=args.worker,
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            timeout=max(args.worker_timeout, 600),
        )
        LOG.info("Qwen worker load complete ok=%s mode=%s", worker_load_result.get("ok"), worker_load_result.get("mode"))

    LOG.info("Checking Qwen worker health/status url=%s", args.worker)
    worker_probe = _check_worker(args.worker, timeout=args.timeout)
    LOG.info("Worker probe loaded=%s mode=%s gpu=%s", (worker_probe.get("status") or {}).get("loaded"), (worker_probe.get("status") or {}).get("mode"), (worker_probe.get("health") or {}).get("gpu"))

    sys.path.insert(0, str(BACKEND_DIR))
    from C2.Orchestrator import run_orchestrator_bundle

    prompt_row = {
        "prompt": prompt,
        "qid": target_qid,
        "diagram": 1,
        "processed_id": processed_id,
        "requested_components": required,
        "diagram_required_objects": required,
    }

    LOG.info("Running C2 orchestrator bundle steps=%s limit=%s skip_visual_stage=%s", args.steps, args.limit, args.skip_visual_stage)
    t0 = time.perf_counter()
    result = run_orchestrator_bundle(
        prompts=[prompt_row],
        worker_url=str(args.worker).rstrip("/"),
        mode="normal",
        steps=int(args.steps),
        limit=int(args.limit),
        timeout=int(args.timeout),
        output_root=str(out_dir / "vstage"),
        visual_component_batch_size=int(args.visual_component_batch_size),
        skip_visual_stage=bool(args.skip_visual_stage),
        max_workers=1,
    )
    LOG.info("C2 orchestrator bundle finished elapsed_s=%.3f ok=%s count=%s errors=%s", time.perf_counter() - t0, result.get("ok") if isinstance(result, dict) else None, result.get("count") if isinstance(result, dict) else None, result.get("errors") if isinstance(result, dict) else None)

    c2_reports = result.get("prompts") if isinstance(result, dict) else {}
    c2_report = c2_reports.get(prompt) if isinstance(c2_reports, dict) else {}
    if not isinstance(c2_report, dict):
        c2_report = {}

    visual_stage = c2_report.get("visual_stage") if isinstance(c2_report.get("visual_stage"), dict) else {}
    LOG.info(
        "C2 report summary accepted=%s requested_resolved=%s visual_components=%s visual_error=%s",
        len(c2_report.get("accepted_components") or []),
        len(c2_report.get("requested_components_resolved") or []),
        len(visual_stage.get("components") or []),
        visual_stage.get("error"),
    )

    bundle_path = _write_json(out_dir / "c2_research" / "c2_bundle_result.json", result)
    component_bundle = _build_component_artifact(
        out_dir=out_dir,
        prompt=prompt,
        diagram_mode=1,
        requested_components=required,
        report=c2_report,
    )
    refined_labels = _clean_list(component_bundle["artifact"].get("refined_labels") or [])
    final_paths = _write_compact_outputs(
        out_dir=out_dir,
        processed_id=processed_id,
        labels=refined_labels,
        full_artifact_path=component_bundle["path"],
    )

    summary = {
        "schema": "c2_label_process_test_summary_v2",
        "created_at_epoch": time.time(),
        "sample_processed_id": processed_id,
        "prompt": prompt,
        "target_qid": target_qid or None,
        "requested_components": required,
        "sample_image": str((source_dir / paths["png"].name).resolve()),
        "output_folder": str(out_dir.resolve()),
        "worker": {
            "url": str(args.worker).rstrip("/"),
            "load_result": worker_load_result,
            "probe": worker_probe,
        },
        "c2_research_result": {
            "bundle_ok": bool(result.get("ok", False)) if isinstance(result, dict) else False,
            "errors": result.get("errors", {}) if isinstance(result, dict) else {},
            "accepted_component_count": len(c2_report.get("accepted_components") or []),
            "requested_components_resolved_count": len(c2_report.get("requested_components_resolved") or []),
            "visual_stage_component_count": len(visual_stage.get("components") or []),
            "visual_stage_error": visual_stage.get("error"),
        },
        "pipeline_result": {
            "status": "ok" if isinstance(result, dict) and bool(result.get("ok", False)) else "error",
            "parser_kind": "c2_research_only",
            "label_count": len(refined_labels),
            "matched_labels": 0,
            "labels_with_strokes": 0,
        },
        "final_labels": [
            {"label": label, "matched_stroke_count": 0, "stroke_indexes": []}
            for label in refined_labels
        ],
        "important_files": {
            "log_file": log_path,
            "c2_bundle_result": bundle_path,
            "c2_component_report": component_bundle["path"],
            "qwen_stage_io_dir": str(qwen_stage_io_dir.resolve()),
            **final_paths,
        },
    }
    summary_path = _write_json(out_dir / "summary.json", summary)
    print(json.dumps({"ok": True, "output_folder": str(out_dir.resolve()), "summary": summary_path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
