#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image


BACKEND_DIR = Path(__file__).resolve().parent
PROCESSED_ID_RE = re.compile(r"^processed_(\d+)$", re.IGNORECASE)
DEFAULT_WORKER = "http://127.0.0.1:8009"
DEFAULT_C2_WORKER_MODEL = "cyankiwi/Qwen3.5-4B-AWQ-4bit"
DEFAULT_FAST_VISION_WORKER_MODEL = os.environ.get(
    "DIAGRAM_VISUAL_DESCRIBER_FAST_MODEL_ID",
    "cyankiwi/InternVL3_5-8B-AWQ-4bit",
)


def _resolve_vision_processor_model(model_id: Any) -> str:
    requested = str(model_id or "").strip()
    lowered = requested.casefold()
    if lowered == "cyankiwi/internvl3_5-8b-awq-4bit":
        return "OpenGVLab/InternVL3_5-8B-HF"
    if lowered == "opengvlab/internvl3_5-8b":
        return "OpenGVLab/InternVL3_5-8B-HF"
    return requested


def _resolve_vision_quantization(model_id: Any, requested_quantization: Any) -> str:
    requested_model = str(model_id or "").strip()
    lowered_model = requested_model.casefold()
    quant = str(requested_quantization or "").strip()

    # This repo name says AWQ, but its vLLM model config advertises
    # `compressed-tensors`. Forcing `awq` causes ModelConfig validation to fail.
    if lowered_model == "cyankiwi/internvl3_5-8b-awq-4bit":
        if quant.casefold() == "awq":
            print(
                "[vision] quantization override: requested awq for "
                f"{requested_model}, but the model config expects compressed-tensors; "
                "letting vLLM infer quantization from the repo config instead."
            )
        return ""

    if not quant and "awq" in lowered_model:
        return "awq"
    return quant


def _vision_supports_mm_dynamic_patch(model_id: Any) -> bool:
    # vLLM's InternVL loader currently forwards mm_processor_kwargs to both the
    # image and video processors. InternVLVideoProcessor rejects max_dynamic_patch,
    # so this CLI knob must be disabled for InternVL-family repos.
    model = str(model_id or "").casefold()
    processor = _resolve_vision_processor_model(model_id).casefold()
    return "internvl" not in model and "internvl" not in processor


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


# Visual-signature source precedence (higher wins). Visual-stage refined
# descriptions beat the short Wikidata terminology that stage 1 emits.
_VISUAL_SIG_SOURCE_RANK = {
    "": 0,
    "fallback_labels": 1,
    "c2:accepted_components:description": 2,
    "c2:accepted_components:stage1": 2,
    "c2:requested_components_resolved": 2,
    "c2:visual_stage:wikipedia": 3,
    "c2:visual_stage:refined": 4,
    "c2:visual_stage": 4,
}


def _coerce_catalog_label(row: Dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return ""
    label = str(row.get("label") or row.get("name") or row.get("component") or "").strip()
    if not label and isinstance(row.get("component"), dict):
        label = str(row["component"].get("label") or row["component"].get("name") or "").strip()
    return label


def _catalog_signature_candidates(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return (text, source_tag) pairs in preferred order for one C2 row."""
    if not isinstance(row, dict):
        return []
    out: List[Tuple[str, str]] = []
    refined = str(row.get("refined_visual_description") or "").strip()
    if refined:
        out.append((refined, "c2:visual_stage:refined"))
    wiki_vis = str(row.get("wikipedia_visual_description") or "").strip()
    if wiki_vis:
        out.append((wiki_vis, "c2:visual_stage:wikipedia"))
    existing_sig = str(row.get("visual_signature") or "").strip()
    if existing_sig:
        out.append((existing_sig, "c2:visual_stage"))
    vis_desc = str(row.get("visual_description") or "").strip()
    if vis_desc:
        out.append((vis_desc, "c2:visual_stage"))
    # component-level stage-1 description (short Wikidata terminology). Weakest.
    comp = row.get("component") if isinstance(row.get("component"), dict) else {}
    stage1 = str(comp.get("stage1_description") or row.get("stage1_description") or row.get("description") or "").strip()
    if stage1:
        out.append((stage1, "c2:accepted_components:stage1"))
    return out


def _clean_component_catalog(
    rows: List[Dict[str, Any]],
    *,
    visual_stage_manifest: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Merge C2 rows by normalized label.

    Later rows do NOT overwrite earlier rows blindly; we prefer a stronger
    visual signature according to ``_VISUAL_SIG_SOURCE_RANK``. Stage-1
    terminology (the short Wikidata description) is preserved separately as
    ``stage1_description``.
    """

    # Fold a visual-stage manifest in first so its refined descriptions are
    # considered authoritative when the in-memory rows are missing them.
    manifest_rows: List[Dict[str, Any]] = []
    if isinstance(visual_stage_manifest, dict):
        comps = visual_stage_manifest.get("components")
        if isinstance(comps, list):
            for comp in comps:
                if isinstance(comp, dict):
                    enriched = dict(comp)
                    enriched.setdefault("source", ["c2:visual_stage"])
                    manifest_rows.append(enriched)
    merged_rows = list(manifest_rows) + [r for r in (rows or []) if isinstance(r, dict)]

    by_key: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for row in merged_rows:
        label = _coerce_catalog_label(row)
        if not label:
            continue
        key = label.casefold()

        comp = row.get("component") if isinstance(row.get("component"), dict) else {}
        stage1_desc = str(
            comp.get("stage1_description")
            or row.get("stage1_description")
            or row.get("description")
            or ""
        ).strip()

        candidates = _catalog_signature_candidates(row)

        synonyms_raw = row.get("synonyms") or row.get("aliases") or []
        synonyms = [str(x).strip() for x in synonyms_raw if str(x).strip()] if isinstance(synonyms_raw, list) else []

        incoming_source = row.get("source") or ["c2"]
        if isinstance(incoming_source, str):
            incoming_source = [incoming_source]
        incoming_source = [str(s).strip() for s in incoming_source if str(s).strip()]

        canonical_candidate = str(row.get("qid") or row.get("id") or comp.get("qid") or "").strip()

        if key not in by_key:
            best_sig = ""
            best_src = ""
            if candidates:
                best_sig, best_src = candidates[0]
            by_key[key] = {
                "label": label,
                "visual_signature": best_sig,
                "visual_signature_source": best_src,
                "stage1_description": stage1_desc,
                "synonyms": list(synonyms),
                "source": list(incoming_source),
                "canonical_candidate": canonical_candidate,
            }
            order.append(key)
            continue

        existing = by_key[key]
        # Upgrade signature if we have a stronger candidate.
        for text, src in candidates:
            if not text:
                continue
            cur_rank = _VISUAL_SIG_SOURCE_RANK.get(existing.get("visual_signature_source", ""), 0)
            new_rank = _VISUAL_SIG_SOURCE_RANK.get(src, 0)
            if new_rank > cur_rank or (new_rank == cur_rank and not existing.get("visual_signature")):
                existing["visual_signature"] = text
                existing["visual_signature_source"] = src
                break
        if stage1_desc and not existing.get("stage1_description"):
            existing["stage1_description"] = stage1_desc
        for syn in synonyms:
            if syn and syn not in existing["synonyms"]:
                existing["synonyms"].append(syn)
        for src in incoming_source:
            if src and src not in existing["source"]:
                existing["source"].append(src)
        if canonical_candidate and not existing.get("canonical_candidate"):
            existing["canonical_candidate"] = canonical_candidate

    return [by_key[k] for k in order]


def _load_visual_stage_manifest(result: Any) -> Optional[Dict[str, Any]]:
    """Load the C2 visual_stage_manifest.json if the orchestrator emitted one."""
    if not isinstance(result, dict):
        return None
    reports = result.get("prompts") if isinstance(result.get("prompts"), dict) else {}
    for report in reports.values():
        if not isinstance(report, dict):
            continue
        visual_stage = report.get("visual_stage") if isinstance(report.get("visual_stage"), dict) else {}
        manifest_path = str(visual_stage.get("manifest_path") or "").strip()
        if not manifest_path:
            continue
        p = Path(manifest_path)
        if not p.is_file():
            continue
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[c2] failed to parse visual_stage_manifest at {p}: {type(exc).__name__}: {exc}")
    return None


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


def _ensure_c2_text_worker_loaded(
    *,
    worker: str,
    model: str,
    timeout: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    cpu_offload_gb: float,
    enforce_eager: bool,
    skip_load: bool,
) -> None:
    if skip_load:
        return
    base = str(worker or DEFAULT_WORKER).rstrip("/")
    status = requests.get(f"{base}/status", timeout=max(5, int(timeout))).json()
    if isinstance(status, dict) and bool(status.get("loaded")) and str(status.get("mode", "") or "").lower() == "text":
        print(f"[c2] worker text model already loaded: {status.get('worker_config') or ''}")
        return
    payload = {
        "model": str(model or DEFAULT_C2_WORKER_MODEL),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "max_model_len": int(max_model_len),
        "max_num_batched_tokens": int(max_num_batched_tokens) if int(max_num_batched_tokens or 0) > 0 else None,
        "max_num_seqs": int(max_num_seqs) if int(max_num_seqs or 0) > 0 else None,
        "cpu_offload_gb": float(cpu_offload_gb or 0.0),
        "enforce_eager": bool(enforce_eager),
        "disable_log_stats": True,
        "warmup": True,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    print(f"[c2] loading worker text model: {payload['model']}")
    response = requests.post(
        f"{base}/load/text",
        json=payload,
        timeout=max(600, int(timeout)),
    )
    response.raise_for_status()
    print(f"[c2] worker load result: {response.json()}")


def _load_docker_vision_worker(
    *,
    worker: str,
    model: str,
    timeout: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    cpu_offload_gb: float,
    enforce_eager: bool,
    quantization: str = "",
    mm_max_dynamic_patch: int = 0,
) -> None:
    base = str(worker or DEFAULT_WORKER).rstrip("/")
    print("[vision] unloading current Docker worker before loading VLM")
    try:
        unload = requests.post(f"{base}/unload", json={}, timeout=max(60, int(timeout)))
        unload.raise_for_status()
        print(f"[vision] unload result: {unload.json()}")
        # Let the previous model's CUDA allocations fully release before vLLM
        # re-probes free VRAM (avoids spurious "free memory < desired utilization"
        # on ~12GB laptop GPUs when switching text -> vision).
        time.sleep(2.0)
    except Exception as exc:
        print(f"[vision] unload warning: {type(exc).__name__}: {exc}")

    payload: Dict[str, Any] = {
        "model": str(model),
        "processor": _resolve_vision_processor_model(model),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "max_model_len": int(max_model_len),
        "max_num_batched_tokens": int(max_num_batched_tokens) if int(max_num_batched_tokens or 0) > 0 else None,
        "max_num_seqs": int(max_num_seqs) if int(max_num_seqs or 0) > 0 else None,
        "cpu_offload_gb": float(cpu_offload_gb or 0.0),
        "enforce_eager": bool(enforce_eager),
        "limit_mm_per_prompt": {"image": 1},
        "disable_log_stats": True,
        "warmup": False,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    quant = _resolve_vision_quantization(model, quantization)
    if quant:
        payload["quantization"] = quant
    if int(mm_max_dynamic_patch or 0) > 0 and _vision_supports_mm_dynamic_patch(model):
        payload["mm_processor_kwargs"] = {
            "max_dynamic_patch": int(mm_max_dynamic_patch),
        }
    elif int(mm_max_dynamic_patch or 0) > 0:
        print(
            "[vision] ignoring --vision-worker-mm-max-dynamic-patch for this model: "
            "vLLM InternVL rejects max_dynamic_patch during video-processor preflight"
        )
    print(f"[vision] loading Docker VLM: {payload}")
    response = requests.post(
        f"{base}/load/vision",
        json=payload,
        timeout=max(1200, int(timeout)),
    )
    response.raise_for_status()
    print(f"[vision] worker load result: {response.json()}")


def _run_vlm_smoke_check(
    *,
    worker: str,
    timeout: int,
    max_new_tokens: int = 48,
) -> Tuple[bool, str, str]:
    """Submit one small JSON-only image prompt through the loaded VLM.

    Returns ``(ok, preview, error)``. ``ok`` is ``True`` only if the worker
    returned parsable JSON containing the expected ``ok: true`` field; the
    preview is the first ~200 characters of raw output so callers can log it.
    """
    base = str(worker or DEFAULT_WORKER).rstrip("/")
    temp_dir = BACKEND_DIR / "_remote_vlm_inputs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = temp_dir / "vlm_smoke.png"
    try:
        Image.new("RGB", (64, 64), (200, 200, 200)).save(smoke_path, format="PNG")
    except Exception as exc:
        return False, "", f"smoke_image_write_failed:{type(exc).__name__}:{exc}"

    try:
        rel = smoke_path.resolve().relative_to(BACKEND_DIR.resolve())
        docker_path = "/workspace/whiteboard_backend/" + str(rel).replace("\\", "/")
    except Exception:
        docker_path = str(smoke_path)

    body = {
        "requests": [
            {
                "prompt": 'Return ONLY JSON on one line: {"ok": true, "shape": "square"}',
                "images": [docker_path],
            }
        ],
        "system_prompt": "You are a JSON-only probe. Reply with JSON only.",
        "generation": {
            "max_new_tokens": int(max_new_tokens),
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "skip_special_tokens": True,
        },
        "use_tqdm": False,
    }
    try:
        resp = requests.post(f"{base}/generate/vision", json=body, timeout=max(120, int(timeout)))
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
    except Exception as exc:
        return False, "", f"smoke_http_failed:{type(exc).__name__}:{exc}"

    rows = data.get("responses") if isinstance(data, dict) else []
    text = ""
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        text = str(rows[0].get("text", "") or "")
    preview = text[:200]
    if not text.strip():
        return False, preview, "empty_vlm_output"
    # lightweight JSON probe (mirrors qwentest.extract_json_object, local to avoid import cycle)
    try:
        start = text.index("{")
        end = text.rindex("}")
        obj = json.loads(text[start : end + 1])
    except Exception as exc:
        return False, preview, f"smoke_json_parse_failed:{type(exc).__name__}:{exc}"
    if not isinstance(obj, dict):
        return False, preview, "smoke_json_not_object"
    if obj.get("ok") is not True:
        return False, preview, "smoke_ok_field_missing"
    return True, preview, ""


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
) -> Tuple[List[str], List[Dict[str, Any]]]:
    from C2.Orchestrator import run_orchestrator_bundle

    prompt = str(depicts or image_name or processed_id).strip()
    if not prompt:
        return [], []

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
        return [], []

    # Visual-stage rows first so their refined descriptions win during merge.
    visual_rows: List[Dict[str, Any]] = []
    visual_stage = report.get("visual_stage") if isinstance(report.get("visual_stage"), dict) else {}
    vis_components = visual_stage.get("components") if isinstance(visual_stage, dict) else []
    if isinstance(vis_components, list):
        for row in vis_components:
            if not isinstance(row, dict):
                continue
            label = _coerce_catalog_label(row)
            if not label:
                continue
            enriched = dict(row)
            enriched["label"] = label
            enriched["source"] = ["c2:visual_stage"]
            visual_rows.append(enriched)

    stage1_rows: List[Dict[str, Any]] = []
    for key in ("accepted_components", "requested_components_resolved"):
        rows = report.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            label = _coerce_catalog_label(row)
            if not label:
                continue
            enriched = dict(row)
            enriched["label"] = label
            enriched["source"] = [f"c2:{key}"]
            stage1_rows.append(enriched)

    combined = visual_rows + stage1_rows

    # If the in-memory report is missing visual-stage rows, try the manifest
    # on disk produced by the orchestrator as an authoritative fallback.
    manifest = None
    if not visual_rows:
        manifest = _load_visual_stage_manifest(result)

    catalog = _clean_component_catalog(combined, visual_stage_manifest=manifest)
    labels = [row["label"] for row in catalog if row.get("label")]
    return _clean_labels(labels), catalog


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
    parser.add_argument("--c2-worker-model", default=DEFAULT_C2_WORKER_MODEL, help="Text model to load in the Docker C2 worker.")
    parser.add_argument("--c2-worker-gpu-memory", type=float, default=0.82, help="GPU memory utilization for loading the C2 text model.")
    parser.add_argument("--c2-worker-max-model-len", type=int, default=4096, help="Max model length for the C2 text model.")
    parser.add_argument("--c2-worker-max-batched-tokens", type=int, default=4096, help="Max batched tokens for the C2 text model.")
    parser.add_argument("--c2-worker-max-seqs", type=int, default=8, help="Max concurrent sequences for the C2 text model.")
    parser.add_argument("--c2-worker-cpu-offload-gb", type=float, default=0.0, help="CPU offload GB for the C2 text model.")
    parser.set_defaults(c2_worker_enforce_eager=True)
    parser.add_argument("--c2-worker-enforce-eager", dest="c2_worker_enforce_eager", action="store_true", help="Load the C2 text model with vLLM enforce_eager.")
    parser.add_argument("--no-c2-worker-enforce-eager", dest="c2_worker_enforce_eager", action="store_false", help="Allow vLLM compile/cudagraph for the C2 text model.")
    parser.add_argument("--no-load-c2-worker", action="store_true", help="Do not auto-load the C2 text worker before discovery.")
    parser.add_argument("--vision-worker-model", default=DEFAULT_FAST_VISION_WORKER_MODEL, help="VLM to load into the Docker worker. Defaults to a fast AWQ InternVL3.5-8B variant for the report tester.")
    parser.add_argument(
        "--vision-worker-gpu-memory",
        type=float,
        default=0.84,
        help=(
            "vLLM gpu_memory_utilization for the Docker VLM (fraction of *total* VRAM to reserve). "
            "On 12GB GPUs 0.93 often fails with ValueError: free memory < desired utilization after "
            "unloading the text model; use 0.80–0.88. Raise only if you have headroom and see OOM later."
        ),
    )
    parser.add_argument("--vision-worker-max-model-len", type=int, default=1024, help="Max model length for the Docker VLM. 1024 is the fast minimum that fits one InternVL cluster image prompt.")
    parser.add_argument("--vision-worker-max-batched-tokens", type=int, default=1024, help="Max batched tokens for the Docker VLM. Keep aligned with --vision-worker-max-model-len for single-cluster batches.")
    parser.add_argument("--vision-worker-max-seqs", type=int, default=1, help="Max concurrent sequences for the Docker VLM.")
    parser.add_argument("--vision-worker-cpu-offload-gb", type=float, default=0.0, help="CPU offload GB for the Docker VLM. Keep 0 for speed; CPU offload is a last-resort fit mode.")
    parser.set_defaults(vision_worker_enforce_eager=False)
    parser.add_argument("--vision-worker-enforce-eager", dest="vision_worker_enforce_eager", action="store_true", help="Load the Docker VLM with vLLM enforce_eager.")
    parser.add_argument("--no-vision-worker-enforce-eager", dest="vision_worker_enforce_eager", action="store_false", help="Allow vLLM compile/cudagraph for the Docker VLM.")
    parser.add_argument("--vision-worker-quantization", default="", help="Optional vLLM quantization name for the Docker VLM.")
    parser.add_argument(
        "--vision-worker-mm-max-dynamic-patch",
        type=int,
        default=0,
        help=(
            "Deprecated for InternVL/vLLM. Leave at 0; nonzero values are ignored for "
            "InternVL because vLLM forwards max_dynamic_patch to InternVLVideoProcessor, "
            "which rejects it."
        ),
    )
    parser.add_argument("--visual-component-batch-size", type=int, default=4)
    parser.add_argument("--skip-visual-stage", action="store_true", help="Skip C2 visual descriptions; still uses C2 component discovery.")
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=0,
        help="Randomly sample at most this many cluster renders. Use 0 for all clusters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Optional random seed for choosing the processed image and cluster sample.",
    )
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
        default="448",
        help="Qwen processed composite image edge. Default 448 for faster VLM image encoding.",
    )
    parser.add_argument(
        "--cluster-visual-max-new-tokens",
        type=int,
        default=96,
        help="Max visual-description tokens per cluster. 96 keeps InternVL JSON complete while staying fast.",
    )
    parser.add_argument(
        "--skip-siglip",
        action="store_true",
        help="Skip local SigLIP scoring. Useful when the Docker VLM already occupies the GPU and you only need a quick report.",
    )
    parser.add_argument(
        "--allow-visual-fallback",
        action="store_true",
        help="Allow falling back from the requested visual describer to Qwen VLM models. Default is off to avoid surprise downloads.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if int(args.seed or 0):
        random.seed(int(args.seed))

    # qwentest reads these at import time, so set CLI defaults before importing it.
    os.environ.setdefault("QWEN_BATCH_SIZE", str(args.batch_size))
    os.environ.setdefault("QWEN_PROC_LONGEST_EDGE", str(args.proc_longest_edge))
    os.environ.setdefault("QWEN_CLUSTER_VISUAL_MAX_NEW_TOKENS", str(max(32, int(args.cluster_visual_max_new_tokens or 80))))
    if bool(args.skip_siglip):
        os.environ["QWEN_SKIP_CLUSTER_SIGLIP"] = "1"
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
    component_catalog: List[Dict[str, Any]] = []
    if labels:
        label_source = "--labels"
    else:
        if not depicts:
            depicts = str(args.base_context or auto_base_context or image_name).strip()
        if not depicts:
            raise RuntimeError("Pass --depicts \"what the image shows\" so C2 can discover labels.")
        _ensure_c2_text_worker_loaded(
            worker=str(args.worker),
            model=str(args.c2_worker_model),
            timeout=int(args.c2_timeout),
            gpu_memory_utilization=float(args.c2_worker_gpu_memory),
            max_model_len=int(args.c2_worker_max_model_len),
            max_num_batched_tokens=int(args.c2_worker_max_batched_tokens),
            max_num_seqs=int(args.c2_worker_max_seqs),
            cpu_offload_gb=float(args.c2_worker_cpu_offload_gb),
            enforce_eager=bool(args.c2_worker_enforce_eager),
            skip_load=bool(args.no_load_c2_worker),
        )
        print(f"[c2] discovering labels for {pid}: {depicts}")
        labels, component_catalog = _labels_from_c2(
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
    if not component_catalog:
        component_catalog = [{"label": label, "visual_signature": "", "synonyms": [], "source": [label_source]} for label in labels]
    if not labels:
        raise RuntimeError(
            "C2 did not return labels and no processed JSON labels were available. "
            "Either make sure the C2 worker is running, or pass --labels as a manual override."
        )

    base_context = str(args.base_context or depicts or auto_base_context or image_name or pid).strip()

    max_clusters = max(0, int(args.max_clusters or 0))
    clusters_total = len(clusters)
    if max_clusters and len(clusters) > max_clusters:
        clusters = random.sample(clusters, max_clusters)

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

    requested_vision_model = str(args.vision_worker_model or qwentest.DIAGRAM_VISUAL_DESCRIBER_MODEL_ID).strip()
    vision_model = qwentest._resolved_multimodal_model_id(requested_vision_model)
    try:
        _load_docker_vision_worker(
            worker=str(args.worker),
            model=vision_model,
            timeout=int(args.c2_timeout),
            gpu_memory_utilization=float(args.vision_worker_gpu_memory),
            max_model_len=int(args.vision_worker_max_model_len),
            max_num_batched_tokens=int(args.vision_worker_max_batched_tokens),
            max_num_seqs=int(args.vision_worker_max_seqs),
            cpu_offload_gb=float(args.vision_worker_cpu_offload_gb),
            enforce_eager=bool(args.vision_worker_enforce_eager),
            quantization=str(args.vision_worker_quantization or ""),
            mm_max_dynamic_patch=int(args.vision_worker_mm_max_dynamic_patch),
        )
    except Exception:
        if not bool(args.allow_visual_fallback):
            raise
        fallback_model = qwentest._resolved_multimodal_model_id(qwentest.DIAGRAM_VISUAL_DESCRIBER_FALLBACK_MODEL_ID)
        print(f"[vision] primary VLM failed; trying explicit fallback because --allow-visual-fallback was set: {fallback_model}")
        _load_docker_vision_worker(
            worker=str(args.worker),
            model=fallback_model,
            timeout=int(args.c2_timeout),
            gpu_memory_utilization=float(args.vision_worker_gpu_memory),
            max_model_len=int(args.vision_worker_max_model_len),
            max_num_batched_tokens=int(args.vision_worker_max_batched_tokens),
            max_num_seqs=int(args.vision_worker_max_seqs),
            cpu_offload_gb=float(args.vision_worker_cpu_offload_gb),
            enforce_eager=bool(args.vision_worker_enforce_eager),
            quantization=str(args.vision_worker_quantization or ""),
            mm_max_dynamic_patch=int(args.vision_worker_mm_max_dynamic_patch),
        )
        vision_model = fallback_model

    print(f"[load] Docker visual describer ready for {pid}: {vision_model}")

    # Smoke-test the freshly-loaded VLM on one tiny image so we fail fast instead
    # of producing a report full of blank-visual rows when the model can't emit JSON.
    smoke_ok, smoke_preview, smoke_error = _run_vlm_smoke_check(
        worker=str(args.worker),
        timeout=int(args.c2_timeout),
        max_new_tokens=max(32, int(args.cluster_visual_max_new_tokens or 48)),
    )
    print(f"[vision] smoke_check ok={smoke_ok} preview={smoke_preview!r} error={smoke_error!r}")
    if not smoke_ok:
        fallback_model = qwentest._resolved_multimodal_model_id(qwentest.DIAGRAM_VISUAL_DESCRIBER_FALLBACK_MODEL_ID)
        if bool(args.allow_visual_fallback) and fallback_model and fallback_model != vision_model:
            print(f"[vision] smoke check failed, switching to fallback VLM: {fallback_model}")
            _load_docker_vision_worker(
                worker=str(args.worker),
                model=fallback_model,
                timeout=int(args.c2_timeout),
                gpu_memory_utilization=float(args.vision_worker_gpu_memory),
                max_model_len=int(args.vision_worker_max_model_len),
                max_num_batched_tokens=int(args.vision_worker_max_batched_tokens),
                max_num_seqs=int(args.vision_worker_max_seqs),
                cpu_offload_gb=float(args.vision_worker_cpu_offload_gb),
                enforce_eager=bool(args.vision_worker_enforce_eager),
                quantization=str(args.vision_worker_quantization or ""),
                mm_max_dynamic_patch=int(args.vision_worker_mm_max_dynamic_patch),
            )
            vision_model = fallback_model
            smoke_ok, smoke_preview, smoke_error = _run_vlm_smoke_check(
                worker=str(args.worker),
                timeout=int(args.c2_timeout),
                max_new_tokens=max(32, int(args.cluster_visual_max_new_tokens or 48)),
            )
            print(f"[vision] smoke_check_fallback ok={smoke_ok} preview={smoke_preview!r} error={smoke_error!r}")
        if not smoke_ok:
            raise RuntimeError(
                "VLM smoke check failed; refusing to label clusters with a broken visual describer. "
                f"model={vision_model} error={smoke_error} preview={smoke_preview!r}. "
                "Pass --allow-visual-fallback to auto-switch to DIAGRAM_VISUAL_DESCRIBER_FALLBACK_MODEL_ID."
            )

    visual_bundle = qwentest.create_remote_qwen_server_bundle(
        base_url=str(args.worker),
        mode="vision",
        model_id=vision_model,
        temp_dir=str(BACKEND_DIR / "_remote_vlm_inputs"),
    )
    model = visual_bundle["model"]
    processor = visual_bundle["processor"]
    device = visual_bundle["device"]
    used_quant = False
    loaded_model_id = str(visual_bundle.get("model_id") or qwentest.DIAGRAM_VISUAL_DESCRIBER_MODEL_ID)
    print(f"[load] done model={loaded_model_id} device={device} used_quant={used_quant}")
    print(f"[labels] source={label_source} count={len(labels)} labels={', '.join(labels[:40])}")
    if max_clusters:
        print(f"[clusters] sampled={len(clusters)} of total={clusters_total} max_clusters={max_clusters}")

    full_img_rgb = np.asarray(Image.open(full_image_path).convert("RGB"), dtype=np.uint8)

    postfacto_bundle: Dict[str, Any] = {}

    def _reload_text_worker_for_final_match() -> Dict[str, Any]:
        nonlocal postfacto_bundle
        _ensure_c2_text_worker_loaded(
            worker=str(args.worker),
            model=str(args.c2_worker_model),
            timeout=int(args.c2_timeout),
            gpu_memory_utilization=float(args.c2_worker_gpu_memory),
            max_model_len=int(args.c2_worker_max_model_len),
            max_num_batched_tokens=int(args.c2_worker_max_batched_tokens),
            max_num_seqs=int(args.c2_worker_max_seqs),
            cpu_offload_gb=float(args.c2_worker_cpu_offload_gb),
            enforce_eager=bool(args.c2_worker_enforce_eager),
            skip_load=bool(args.no_load_c2_worker),
        )
        postfacto_bundle = qwentest.create_remote_qwen_server_bundle(
            base_url=str(args.worker),
            mode="text",
            model_id=str(args.c2_worker_model),
        )
        return postfacto_bundle

    result_by_idx = qwentest.label_clusters_transformers(
        {
            idx: {
                "base_context": base_context,
                "candidate_labels_raw": labels,
                "candidate_labels_refined": labels,
                "component_catalog": component_catalog,
                "skip_siglip_scoring": bool(args.skip_siglip),
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
        postfacto_model_loader=_reload_text_worker_for_final_match,
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
    print(f"clusters_used: {len(clusters)} of {clusters_total}")
    print(f"labels_json: {labels_path.resolve()}")
    print(f"debug_json: {debug_path.resolve()}")
    print(f"cluster_label_report: {report_path.resolve() if str(report_path) else '(not written)'}")

    if report_path.is_file() and not args.no_open:
        opened = _open_report(report_path)
        print(f"opened_report: {opened}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
