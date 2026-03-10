#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
import random
import re
import threading
import time
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
DEFAULT_WORKFLOW_PATH = Path(
    os.getenv(
        "COMFY_WORKFLOW_JSON",
        str(BASE_DIR / "ComfyWorkflows" / "flux2_klein_4b_fp8.json"),
    )
)

UNIQUE_ROOT = BASE_DIR / "ResearchImages" / "UniqueImages" / "ComfyGenerated"
META_PATH = UNIQUE_ROOT / "image_metadata_context.json"
_META_LOCK = threading.Lock()


def _slug(s: str, cap: int = 80) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s or "").strip())
    s = s.strip("._-")
    if not s:
        s = "item"
    return s[:cap]


def _http_json(method: str, url: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 3600) -> Dict[str, Any]:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    req = urllib.request.Request(url, data=body, headers=headers, method=method.upper())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()

    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _http_bytes(url: str, timeout: int = 3600) -> bytes:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _load_workflow_template(workflow_path: Optional[str | Path] = None) -> Dict[str, Any]:
    p = Path(workflow_path or DEFAULT_WORKFLOW_PATH)
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_output_root(output_root: Optional[str | Path]) -> Path:
    if output_root is None:
        return UNIQUE_ROOT
    p = Path(output_root)
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    return p


def _convert_ui_workflow_to_api_prompt(workflow: Dict[str, Any]) -> Dict[str, Any]:
    nodes = workflow.get("nodes")
    links = workflow.get("links")
    if not isinstance(nodes, list) or not isinstance(links, list):
        raise ValueError("Workflow JSON is neither API prompt format nor UI workflow format.")

    link_map: Dict[int, List[Any]] = {}
    for row in links:
        if isinstance(row, list) and len(row) >= 6:
            try:
                link_map[int(row[0])] = row
            except Exception:
                continue

    prompt: Dict[str, Any] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue

        nid = str(node.get("id"))
        class_type = str(node.get("type") or "").strip()
        if not nid or not class_type:
            continue

        node_inputs: Dict[str, Any] = {}
        widget_values = list(node.get("widgets_values") or [])
        widget_i = 0

        for inp in (node.get("inputs") or []):
            if not isinstance(inp, dict):
                continue

            name = str(inp.get("name") or "").strip()
            if not name:
                continue

            link_id = inp.get("link", None)
            if link_id is not None:
                try:
                    row = link_map[int(link_id)]
                    src_node_id = str(row[1])
                    src_output_index = int(row[2])
                    node_inputs[name] = [src_node_id, src_output_index]
                    continue
                except Exception:
                    pass

            if isinstance(inp.get("widget"), dict) and widget_i < len(widget_values):
                node_inputs[name] = widget_values[widget_i]
                widget_i += 1

        prompt[nid] = {
            "class_type": class_type,
            "inputs": node_inputs,
        }

    if not prompt:
        raise ValueError("UI workflow conversion produced an empty prompt graph.")
    return prompt


def _ensure_api_prompt_graph(workflow: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(workflow.get("nodes"), list) and isinstance(workflow.get("links"), list):
        return _convert_ui_workflow_to_api_prompt(workflow)

    prompt = workflow.get("prompt")
    if isinstance(prompt, dict):
        return copy.deepcopy(prompt)

    if all(isinstance(v, dict) and "class_type" in v for v in workflow.values()):
        return copy.deepcopy(workflow)

    raise ValueError("Unsupported workflow JSON shape.")


def _patch_prompt_graph(
    prompt_graph: Dict[str, Any],
    *,
    positive_prompt: str,
    filename_prefix: str,
    negative_prompt: Optional[str] = None,
    batch_size: int = 1,
) -> Dict[str, Any]:
    graph = copy.deepcopy(prompt_graph)

    positive_nodes: List[str] = []
    negative_nodes: List[str] = []
    preview_nodes: List[str] = []
    has_save_node = False

    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue

        class_type = str(node.get("class_type") or "").strip()
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}
            node["inputs"] = inputs

        if class_type == "CLIPTextEncode":
            txt = inputs.get("text")
            if isinstance(txt, str) and txt.strip():
                positive_nodes.append(node_id)
            else:
                negative_nodes.append(node_id)

        if class_type == "PreviewImage":
            preview_nodes.append(node_id)
        elif class_type == "SaveImage":
            has_save_node = True
            inputs["filename_prefix"] = filename_prefix

        if class_type == "RandomNoise" and "noise_seed" in inputs:
            inputs["noise_seed"] = random.randint(1, 2**63 - 1)

        if class_type == "EmptyFlux2LatentImage" and "batch_size" in inputs:
            inputs["batch_size"] = int(batch_size)

    if not positive_nodes:
        raise RuntimeError("No positive CLIPTextEncode node found in workflow template.")

    for node_id in positive_nodes:
        graph[node_id]["inputs"]["text"] = positive_prompt

    if negative_prompt is not None:
        for node_id in negative_nodes:
            graph[node_id]["inputs"]["text"] = negative_prompt

    if preview_nodes:
        for node_id in preview_nodes:
            graph[node_id]["class_type"] = "SaveImage"
            inputs = graph[node_id].setdefault("inputs", {})
            inputs["filename_prefix"] = filename_prefix
    elif not has_save_node:
        raise RuntimeError("Workflow template has neither PreviewImage nor SaveImage node.")

    return graph


def _queue_prompt(comfy_url: str, prompt_graph: Dict[str, Any]) -> str:
    payload = {
        "client_id": uuid.uuid4().hex,
        "prompt": prompt_graph,
    }
    data = _http_json("POST", f"{comfy_url}/prompt", payload, timeout=3600)
    prompt_id = str(data.get("prompt_id") or "").strip()
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return prompt_id. Response={data!r}")
    return prompt_id


def _unwrap_history_entry(data: Dict[str, Any], prompt_id: str) -> Optional[Dict[str, Any]]:
    if prompt_id in data and isinstance(data[prompt_id], dict):
        return data[prompt_id]
    if data.get("prompt_id") == prompt_id:
        return data
    return None


def _extract_image_specs(history_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    outputs = history_entry.get("outputs") or {}
    if not isinstance(outputs, dict):
        return out

    for node_out in outputs.values():
        if not isinstance(node_out, dict):
            continue
        images = node_out.get("images") or []
        if not isinstance(images, list):
            continue

        for img in images:
            if not isinstance(img, dict):
                continue
            filename = str(img.get("filename") or "").strip()
            if not filename:
                continue

            out.append(
                {
                    "filename": filename,
                    "subfolder": str(img.get("subfolder") or "").strip(),
                    "type": str(img.get("type") or "output").strip() or "output",
                }
            )

    return out


def _wait_for_history_images(comfy_url: str, prompt_id: str, timeout_sec: float = 3600.0, poll_sec: float = 0.6) -> List[Dict[str, Any]]:
    t0 = time.perf_counter()
    while True:
        data = _http_json("GET", f"{comfy_url}/history/{prompt_id}", None, timeout=3600)
        entry = _unwrap_history_entry(data, prompt_id)
        if entry is not None:
            imgs = _extract_image_specs(entry)
            if imgs:
                return imgs

        if (time.perf_counter() - t0) >= timeout_sec:
            raise TimeoutError(f"Timed out waiting for ComfyUI history outputs for prompt_id={prompt_id}")

        time.sleep(poll_sec)


def _download_history_images(
    *,
    comfy_url: str,
    image_specs: List[Dict[str, Any]],
    prompt: str,
    topic: str,
    prompt_id: str,
    output_root: Path,
    clean_prompt_dir: bool = True,
) -> List[str]:
    prompt_dir = output_root / _slug(topic or "misc") / _slug(prompt)
    if clean_prompt_dir and prompt_dir.exists():
        for old in prompt_dir.glob("*"):
            if old.is_file():
                try:
                    old.unlink()
                except Exception:
                    pass
    prompt_dir.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    for idx, spec in enumerate(image_specs, start=1):
        q = {
            "filename": spec["filename"],
            "type": spec.get("type") or "output",
        }
        subfolder = spec.get("subfolder") or ""
        if subfolder:
            q["subfolder"] = subfolder

        blob = _http_bytes(f"{comfy_url}/view?{urllib.parse.urlencode(q)}", timeout=3600)

        suffix = Path(str(spec["filename"])).suffix or ".png"
        out_name = f"{_slug(prompt_id, 32)}_{idx:02d}{suffix}"
        out_path = prompt_dir / out_name
        out_path.write_bytes(blob)
        saved.append(str(out_path))

    _write_metadata(
        saved,
        prompt=prompt,
        topic=topic,
        prompt_id=prompt_id,
        meta_path=(output_root / "image_metadata_context.json"),
    )
    return saved


def _write_metadata(
    paths: List[str],
    *,
    prompt: str,
    topic: str,
    prompt_id: str,
    meta_path: Path,
) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with _META_LOCK:
        payload: Dict[str, Any] = {}
        if meta_path.is_file():
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8")) or {}
            except Exception:
                payload = {}

        if not isinstance(payload, dict):
            payload = {}

        for p in paths:
            payload[str(p)] = {
                "base_context": prompt,
                "topic": topic,
                "source": "comfy_flux2_klein_4b_fp8",
                "prompt_id": prompt_id,
                "final_score": 1.0,
                "contexts": [],
            }

        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_many(
    prompt_to_topic: Dict[str, str],
    *,
    workflow_path: Optional[str | Path] = None,
    comfy_url: Optional[str] = None,
    output_root: Optional[str | Path] = None,
    negative_prompt: Optional[str] = None,
    batch_size: int = 1,
    wait_timeout_sec: float = 3600.0,
    download_workers: int = 4,
    clean_prompt_dirs: bool = True,
) -> Dict[str, Dict[str, Any]]:
    comfy_url = (comfy_url or DEFAULT_COMFY_URL).rstrip("/")
    resolved_output_root = _resolve_output_root(output_root)
    template = _load_workflow_template(workflow_path)
    prompt_graph_template = _ensure_api_prompt_graph(template)

    jobs: Dict[str, Dict[str, Any]] = {}
    for prompt, topic in (prompt_to_topic or {}).items():
        p = str(prompt or "").strip()
        t = str(topic or "").strip()
        if not p:
            continue

        prefix = f"comfy_flux_{_slug(t or 'misc', 40)}_{_slug(p, 50)}_{uuid.uuid4().hex[:8]}"
        graph = _patch_prompt_graph(
            prompt_graph_template,
            positive_prompt=p,
            negative_prompt=negative_prompt,
            filename_prefix=prefix,
            batch_size=batch_size,
        )
        prompt_id = _queue_prompt(comfy_url, graph)

        jobs[p] = {
            "topic": t,
            "prompt_id": prompt_id,
            "filename_prefix": prefix,
        }

    def _wait_and_download(one_prompt: str, info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        prompt_id = str(info["prompt_id"])
        topic = str(info.get("topic") or "")

        specs = _wait_for_history_images(comfy_url, prompt_id, timeout_sec=wait_timeout_sec)
        saved = _download_history_images(
            comfy_url=comfy_url,
            image_specs=specs,
            prompt=one_prompt,
            topic=topic,
            prompt_id=prompt_id,
            output_root=resolved_output_root,
            clean_prompt_dir=bool(clean_prompt_dirs),
        )

        return one_prompt, {
            "topic": topic,
            "prompt_id": prompt_id,
            "saved_paths": saved,
            "count": len(saved),
            "output_root": str(resolved_output_root),
        }

    out: Dict[str, Dict[str, Any]] = {}
    max_workers = max(1, min(int(download_workers), max(1, len(jobs))))

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="comfy_dl") as ex:
        futs = {ex.submit(_wait_and_download, prompt, info): prompt for prompt, info in jobs.items()}
        for fut in as_completed(futs):
            prompt = futs[fut]
            out[prompt] = fut.result()[1]

    return out


def free_all_models(
    *,
    comfy_url: Optional[str] = None,
    unload_models: bool = True,
    free_memory: bool = True,
) -> Dict[str, Any]:
    """
    Ask ComfyUI to release loaded model VRAM while keeping the server alive.
    """
    comfy_url = (comfy_url or DEFAULT_COMFY_URL).rstrip("/")
    payloads = [
        {"unload_models": bool(unload_models), "free_memory": bool(free_memory)},
        {"unload_models": bool(unload_models), "free_memory": bool(free_memory), "clear_queue": False},
        {"free_memory": bool(free_memory)},
    ]

    last_err: Optional[Exception] = None
    for payload in payloads:
        try:
            return _http_json("POST", f"{comfy_url}/free", payload, timeout=120)
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"ComfyUI /free call failed: {last_err}")


def warmup_server_models(
    *,
    workflow_path: Optional[str | Path] = None,
    comfy_url: Optional[str] = None,
    prompt_text: str = "simple black line drawing of a circle on white background",
    negative_prompt: Optional[str] = None,
    batch_size: int = 1,
    wait_timeout_sec: float = 900.0,
) -> Dict[str, Any]:
    """
    Queues one cheap dummy generation to keep Comfy model weights hot.
    """
    comfy_url = (comfy_url or DEFAULT_COMFY_URL).rstrip("/")
    template = _load_workflow_template(workflow_path)
    prompt_graph_template = _ensure_api_prompt_graph(template)

    prefix = f"comfy_warmup_{uuid.uuid4().hex[:8]}"
    graph = _patch_prompt_graph(
        prompt_graph_template,
        positive_prompt=str(prompt_text or "").strip() or "warmup",
        negative_prompt=negative_prompt,
        filename_prefix=prefix,
        batch_size=max(1, int(batch_size)),
    )
    prompt_id = _queue_prompt(comfy_url, graph)
    specs = _wait_for_history_images(comfy_url, prompt_id, timeout_sec=float(wait_timeout_sec))

    return {
        "prompt_id": prompt_id,
        "images_reported": len(specs),
        "filename_prefix": prefix,
    }
