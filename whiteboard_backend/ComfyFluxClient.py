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
                "source_kind": "generated",
                "source_name": "comfy_flux2_klein_4b_fp8",
                "prompt_id": prompt_id,
                "final_score": 1.0,
                "contexts": [],
            }

        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_vector(vec: Any) -> Optional[List[float]]:
    if not isinstance(vec, list) or not vec:
        return None
    out: List[float] = []
    for x in vec:
        if not isinstance(x, (int, float)):
            return None
        out.append(float(x))
    return out or None


def _resolve_hot_minilm_bundle(minilm_bundle: Any = None) -> Any:
    if minilm_bundle is not None:
        return minilm_bundle
    try:
        from shared_models import get_minilm
        return get_minilm()
    except Exception:
        return None


def _resolve_hot_siglip_bundle(siglip_bundle: Any = None) -> Any:
    if siglip_bundle is not None:
        return siglip_bundle
    try:
        from shared_models import get_siglip
        return get_siglip()
    except Exception:
        return None


def _embed_prompts_minilm(texts: List[str], minilm_bundle: Any) -> Optional[List[List[float]]]:
    if not texts:
        return []
    if minilm_bundle is None:
        return None

    try:
        if getattr(minilm_bundle, "use_sentence_transformers", False) and hasattr(minilm_bundle.model, "encode"):
            vecs = minilm_bundle.model.encode(texts, normalize_embeddings=True)
            return vecs.tolist() if hasattr(vecs, "tolist") else [list(map(float, v)) for v in vecs]
    except Exception:
        return None

    try:
        import torch

        tok = getattr(minilm_bundle, "tokenizer", None)
        mdl = getattr(minilm_bundle, "model", None)
        dev = getattr(minilm_bundle, "device", None)
        if tok is None or mdl is None:
            return None

        device = dev
        if not isinstance(device, torch.device):
            device = torch.device(str(device) if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        with torch.inference_mode():
            batch = tok(texts, return_tensors="pt", padding=True, truncation=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = mdl(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).expand(out.size()).float()
            summed = (out * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / denom
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            return pooled.detach().cpu().tolist()
    except Exception:
        return None


def _embed_image_paths_siglip(paths: List[str], siglip_bundle: Any) -> Optional[List[Optional[List[float]]]]:
    if not paths:
        return []
    if siglip_bundle is None:
        return None

    try:
        import torch
        from PIL import Image

        proc = getattr(siglip_bundle, "processor", None)
        mdl = getattr(siglip_bundle, "model", None)
        dev = getattr(siglip_bundle, "device", None)
        if proc is None or mdl is None or not hasattr(mdl, "get_image_features"):
            return None

        valid_pils: List[Any] = []
        valid_idx: List[int] = []
        out: List[Optional[List[float]]] = [None] * len(paths)
        for i, p in enumerate(paths):
            try:
                with Image.open(p) as im:
                    valid_pils.append(im.convert("RGB"))
                valid_idx.append(i)
            except Exception:
                continue

        if not valid_pils:
            return out

        device = dev
        if not isinstance(device, torch.device):
            device = torch.device(str(device) if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        with torch.inference_mode():
            inputs = proc(images=valid_pils, return_tensors="pt", padding=True)
            try:
                inputs = inputs.to(device)
            except Exception:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = mdl.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            vecs = feats.detach().cpu().tolist()

        for idx, vec in zip(valid_idx, vecs):
            out[idx] = _json_vector(vec)
        return out
    except Exception:
        return None


def enrich_generation_metadata(
    generated_map: Dict[str, Dict[str, Any]],
    *,
    siglip_bundle: Any = None,
    minilm_bundle: Any = None,
    meta_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    mp = Path(meta_path or META_PATH)
    mp.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {}
    if mp.is_file():
        try:
            payload = json.loads(mp.read_text(encoding="utf-8")) or {}
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}

    prompt_rows: List[Tuple[str, Dict[str, Any], List[str]]] = []
    all_paths: List[str] = []
    for prompt, info in (generated_map or {}).items():
        if not isinstance(info, dict):
            continue
        saved_paths = [str(x) for x in (info.get("saved_paths") or []) if str(x).strip()]
        if not saved_paths:
            continue
        pp = str(prompt or "").strip()
        if not pp:
            continue
        prompt_rows.append((pp, info, saved_paths))
        all_paths.extend(saved_paths)

    prompts = [row[0] for row in prompt_rows]
    minilm_bundle = _resolve_hot_minilm_bundle(minilm_bundle)
    siglip_bundle = _resolve_hot_siglip_bundle(siglip_bundle)

    prompt_vecs_raw = _embed_prompts_minilm(prompts, minilm_bundle)
    prompt_vecs_by_prompt: Dict[str, List[float]] = {}
    if isinstance(prompt_vecs_raw, list):
        for prompt, vec in zip(prompts, prompt_vecs_raw):
            jv = _json_vector(vec)
            if jv is not None:
                prompt_vecs_by_prompt[prompt] = jv

    clip_vecs_raw = _embed_image_paths_siglip(all_paths, siglip_bundle)
    clip_vecs_by_path: Dict[str, List[float]] = {}
    if isinstance(clip_vecs_raw, list):
        for path, vec in zip(all_paths, clip_vecs_raw):
            if vec is not None:
                clip_vecs_by_path[path] = vec

    prompt_count = 0
    clip_count = 0
    context_count = 0

    with _META_LOCK:
        for prompt, info, saved_paths in prompt_rows:
            prompt_vec = prompt_vecs_by_prompt.get(prompt)
            topic = str(info.get("topic") or "").strip()
            prompt_id = str(info.get("prompt_id") or "").strip()

            for p in saved_paths:
                entry = payload.get(p)
                if not isinstance(entry, dict):
                    entry = {}

                entry["base_context"] = str(entry.get("base_context") or prompt).strip() or prompt
                if topic:
                    entry["topic"] = topic
                if prompt_id:
                    entry["prompt_id"] = prompt_id
                entry["source"] = str(entry.get("source") or "comfy_flux2_klein_4b_fp8").strip() or "comfy_flux2_klein_4b_fp8"
                entry["source_kind"] = str(entry.get("source_kind") or "generated").strip() or "generated"
                entry["source_name"] = str(entry.get("source_name") or "comfy_flux2_klein_4b_fp8").strip() or "comfy_flux2_klein_4b_fp8"
                entry["final_score"] = float(entry.get("final_score") or 1.0)

                if prompt_vec is not None:
                    entry["prompt_embedding"] = prompt_vec
                    entry["contexts"] = [
                        {
                            "source_kind": entry["source_kind"],
                            "source_name": entry["source_name"],
                            "ctx_text": prompt,
                            "ctx_score": 1.0,
                            "ctx_sem_score": 1.0,
                            "ctx_confidence": 1.0,
                            "ctx_embedding": prompt_vec,
                        }
                    ]
                    prompt_count += 1
                    context_count += 1

                clip_vec = clip_vecs_by_path.get(p)
                if clip_vec is not None:
                    entry["clip_embedding"] = clip_vec
                    clip_count += 1

                payload[p] = entry

        mp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "meta_path": str(mp),
        "prompts": len(prompt_rows),
        "images": len(all_paths),
        "prompt_embeddings": int(prompt_count),
        "clip_embeddings": int(clip_count),
        "context_embeddings": int(context_count),
        "siglip_available": bool(siglip_bundle is not None),
        "minilm_available": bool(minilm_bundle is not None),
    }


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
            data = _http_json("POST", f"{comfy_url}/free", payload, timeout=120)
            if isinstance(data, dict):
                err = data.get("error")
                if isinstance(err, str) and err.strip():
                    raise RuntimeError(f"ComfyUI /free returned error: {err.strip()}")
            return data
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
