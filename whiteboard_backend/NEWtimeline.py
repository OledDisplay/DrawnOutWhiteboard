from __future__ import annotations

import difflib
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from LLMstuff.input_builder_responce_parser import (
    build_chapter_speech_request,
    build_image_request_request,
    build_qwen_c2_component_verifier_prompt,
    build_qwen_diagram_component_stroke_match_prompt,
    build_qwen_diagram_action_planner_prompt,
    build_qwen_full_speech_action_sync_prompt,
    build_qwen_non_semantic_image_description_prompt,
    build_qwen_stroke_meaning_filter_prompt,
    build_logical_timeline_request,
    build_qwen_step_prompt,
    build_qwen_space_planner_prompt,
    build_text_request_request,
    collect_step_objects,
    extract_pause_markers,
    normalize_ws,
    parse_chapter_speech_output,
    parse_image_request_output,
    parse_logical_timeline_output,
    parse_qwen_c2_component_verifier_output,
    parse_qwen_diagram_component_stroke_match_output,
    parse_qwen_diagram_action_planner_output,
    parse_qwen_full_speech_action_sync_output,
    parse_qwen_non_semantic_image_description_output,
    parse_qwen_space_planner_output,
    parse_qwen_stroke_meaning_filter_output,
    parse_qwen_step_output,
    parse_text_request_output,
)
try:
    from LLMstuff.qwen_server_client import QwenServerClient
except Exception:
    from LLMstuff.qwen_vllm_server import QwenServerClient


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


DEFAULT_MODEL_CANDIDATES = ["gpt-5-mini"]
DEFAULT_QWEN_BASE_URL = str(os.getenv("QWEN_VLLM_SERVER_URL", "http://127.0.0.1:8009") or "http://127.0.0.1:8009").strip()
DEFAULT_QWEN_MODEL = str(os.getenv("QWEN_VISION_MODEL_ID", os.getenv("QWEN_TEXT_MODEL_ID", "cyankiwi/Qwen3.5-4B-AWQ-4bit")) or "cyankiwi/Qwen3.5-4B-AWQ-4bit").strip()
DEFAULT_QWEN_CONTEXT_LEN = int(os.getenv("QWEN_SERVER_MAX_MODEL_LEN", "20000") or 20000)
DEFAULT_QWEN_GPU_UTIL = float(os.getenv("QWEN_SERVER_GPU_UTIL", "0.80") or 0.80)
DEFAULT_COMPONENT_REFINED_VISUAL_DESC_MAX_CHARS = int(os.getenv("NEWTIMELINE_COMPONENT_REFINED_VISUAL_DESC_MAX_CHARS", "700") or 700)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


DEFAULT_PRELOAD_SIGLIP_TEXT = _env_bool("NEWTIMELINE_PRELOAD_SIGLIP_TEXT", False)


@dataclass
class ImageAsset:
    prompt: str
    diagram: int
    processed_ids: List[str]
    refined_labels_file: Optional[str]
    bbox_px: Optional[Tuple[int, int]] = None
    objects: Optional[List[Dict[str, Any]]] = None


def build_client() -> OpenAI:
    api_key = os.getenv("OPEN_AI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set OPEN_AI_KEY or OPENAI_API_KEY.")
    return OpenAI(api_key=api_key)


def call_responses_text(
    client: OpenAI,
    model_candidates: Sequence[str],
    input_items: List[Dict[str, str]],
    *,
    reasoning_effort: str = "low",
    max_output_tokens: Optional[int] = None,
    text_format: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    last_err: Optional[Exception] = None
    for model in model_candidates:
        try:
            kwargs: Dict[str, Any] = {
                "model": str(model),
                "input": input_items,
                "reasoning": {"effort": reasoning_effort},
            }
            if max_output_tokens is not None:
                kwargs["max_output_tokens"] = int(max_output_tokens)
            if text_format is not None:
                kwargs["text"] = text_format
            resp = client.responses.create(**kwargs)
            return resp.output_text, str(model)
        except Exception as exc:
            if text_format is not None and "invalid_json_schema" in repr(exc):
                try:
                    kwargs = {
                        "model": str(model),
                        "input": input_items,
                        "reasoning": {"effort": reasoning_effort},
                        "text": {"format": {"type": "json_object"}},
                    }
                    if max_output_tokens is not None:
                        kwargs["max_output_tokens"] = int(max_output_tokens)
                    resp = client.responses.create(**kwargs)
                    return resp.output_text, str(model)
                except Exception as retry_exc:
                    last_err = retry_exc
                    continue
            last_err = exc
    raise RuntimeError(f"All model candidates failed. Last error: {last_err!r}")


class NewTimelineFirstModule:
    def __init__(
        self,
        *,
        model_candidates: Optional[List[str]] = None,
        reasoning_effort: str = "low",
        gpu_index: int = 0,
        cpu_threads: int = 4,
        chapter_cap: int = 3,
        debug_print: bool = True,
        debug_out_dir: str = "PipelineOutputs",
        gpt_cache: bool = False,
        gpt_cache_file: Optional[str] = None,
        qwen_base_url: str = DEFAULT_QWEN_BASE_URL,
        qwen_model: str = DEFAULT_QWEN_MODEL,
    ) -> None:
        self.client = build_client()
        self.model_candidates = list(model_candidates or DEFAULT_MODEL_CANDIDATES)
        self.reasoning_effort = str(reasoning_effort or "low")
        self.gpu_index = int(gpu_index)
        self.cpu_threads = int(cpu_threads)
        self.chapter_cap = max(1, int(chapter_cap))
        self.debug_print = bool(debug_print)
        self.debug_out_dir = str(debug_out_dir or "PipelineOutputs")
        self.qwen_base_url = str(qwen_base_url or DEFAULT_QWEN_BASE_URL).rstrip("/")
        self.qwen_model = str(qwen_model or DEFAULT_QWEN_MODEL).strip()
        self.qwen_client = QwenServerClient(self.qwen_base_url)
        self._legacy_group_helper = None

        self.gpt_cache_enabled = bool(gpt_cache)
        self.gpt_cache_file = str(gpt_cache_file or (Path(self.debug_out_dir) / "_gpt_call_cache_newtimeline.json"))
        self._gpt_cache_lock = threading.Lock()
        self._gpt_cache: Dict[str, Dict[str, str]] = {}
        if self.gpt_cache_enabled:
            try:
                cache_path = Path(self.gpt_cache_file)
                if cache_path.is_file():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        self._gpt_cache = {
                            str(key): value
                            for key, value in raw.items()
                            if isinstance(value, dict)
                            and isinstance(value.get("text"), str)
                            and isinstance(value.get("model"), str)
                        }
            except Exception:
                self._gpt_cache = {}

        self._qwen_load_lock = threading.Lock()
        self._qwen_load_thread: Optional[threading.Thread] = None
        self._qwen_load_state: Dict[str, Any] = {
            "started": False,
            "ready": False,
            "error": None,
            "response": None,
            "mode": "vision",
        }
        self._qwen_sleeping = False
        self._diagram_research_state: Optional[Dict[str, Any]] = None
        self._diagram_research_thread: Optional[threading.Thread] = None
        self._dimensions_config_cache: Optional[Dict[str, Any]] = None
        self._font_metrics_cache: Optional[Dict[str, float]] = None
        self._glyph_metrics_cache: Dict[int, Dict[str, float]] = {}

    def _dbg(self, message: str, *, data: Any = None) -> None:
        if not self.debug_print:
            return
        if data is None:
            print(f"[NEWtimeline][DBG] {message}", flush=True)
            return
        try:
            text = json.dumps(data, ensure_ascii=False)
            if len(text) > 1800:
                text = text[:1800] + " ...<truncated>"
        except Exception:
            text = "<unserializable>"
        print(f"[NEWtimeline][DBG] {message} | {text}", flush=True)

    def _import_image_researcher(self):
        for module_name in ("ImageResearcher", "Imageresearcher"):
            try:
                module = importlib.import_module(module_name)
                sys.modules.setdefault("ImageResearcher", module)
                return module
            except ModuleNotFoundError:
                continue
        for path in (BACKEND_DIR / "Imageresearcher.py", BACKEND_DIR / "ImageResearcher.py"):
            if not path.is_file():
                continue
            spec = importlib.util.spec_from_file_location("ImageResearcher", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules["ImageResearcher"] = module
            spec.loader.exec_module(module)
            return module
        raise ModuleNotFoundError("No module named 'ImageResearcher' or 'Imageresearcher'")

    def _gpu_memory_snapshot(self) -> Dict[str, Any]:
        try:
            import torch

            if not torch.cuda.is_available():
                return {"cuda": False}
            try:
                torch.cuda.set_device(self.gpu_index)
            except Exception:
                pass
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            return {
                "cuda": True,
                "gpu_index": int(self.gpu_index),
                "free_gib": round(float(free_bytes) / (1024.0 ** 3), 3),
                "total_gib": round(float(total_bytes) / (1024.0 ** 3), 3),
                "used_gib": round(float(total_bytes - free_bytes) / (1024.0 ** 3), 3),
                "torch_allocated_gib": round(float(allocated) / (1024.0 ** 3), 3),
                "torch_reserved_gib": round(float(reserved) / (1024.0 ** 3), 3),
            }
        except Exception as exc:
            return {"cuda": None, "error": f"{type(exc).__name__}: {exc}"}

    def _gpu_memory_delta(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in ("free_gib", "used_gib", "torch_allocated_gib", "torch_reserved_gib"):
            try:
                out[f"{key}_delta"] = round(float(after.get(key, 0.0)) - float(before.get(key, 0.0)), 3)
            except Exception:
                continue
        return out

    def _write_json_file(self, path: Path, payload: Any) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def _internal_runtime_dir(self) -> Path:
        return Path(__file__).resolve().parent / "_newtimeline_internal"

    def _internal_cache_dir(self, stage_name: str) -> Path:
        return self._internal_runtime_dir() / "cache" / str(stage_name or "generic").strip()

    def _stable_json_dumps(self, payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _stage_cache_key(self, *, stage_name: str, payload: Any) -> str:
        raw = self._stable_json_dumps(
            {
                "stage": str(stage_name or "").strip(),
                "payload": payload,
            }
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_stage_cache(self, *, stage_name: str, key: str) -> Optional[Dict[str, Any]]:
        path = self._internal_cache_dir(stage_name) / f"{key}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _store_stage_cache(
        self,
        *,
        stage_name: str,
        key: str,
        data: Any,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        path = self._internal_cache_dir(stage_name) / f"{key}.json"
        payload = {
            "stage": str(stage_name or "").strip(),
            "key": str(key),
            "written_at_unix": time.time(),
            "meta": dict(meta or {}),
            "data": data,
        }
        return self._write_json_file(path, payload)

    def _load_cached_stage_data(self, *, stage_name: str, payload: Any) -> Tuple[Optional[Any], str]:
        key = self._stage_cache_key(stage_name=stage_name, payload=payload)
        cached = self._load_stage_cache(stage_name=stage_name, key=key)
        if not isinstance(cached, dict):
            return None, key
        return cached.get("data"), key

    def _load_latest_stage_data(self, stage_name: str) -> Tuple[Optional[Any], Optional[str]]:
        cache_dir = self._internal_cache_dir(stage_name)
        if not cache_dir.is_dir():
            return None, None
        best_payload: Optional[Dict[str, Any]] = None
        best_path: Optional[Path] = None
        best_written = -1.0
        for path in cache_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            try:
                written = float(payload.get("written_at_unix", 0.0) or 0.0)
            except Exception:
                written = 0.0
            if best_payload is None or written > best_written:
                best_payload = payload
                best_path = path
                best_written = written
        if not isinstance(best_payload, dict):
            return None, None
        return best_payload.get("data"), str(best_path) if best_path is not None else None

    def _store_cached_stage_data(
        self,
        *,
        stage_name: str,
        payload: Any,
        data: Any,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        key = self._stage_cache_key(stage_name=stage_name, payload=payload)
        path = self._store_stage_cache(stage_name=stage_name, key=key, data=data, meta=meta)
        return path, key

    def _load_first_module_result_cache(self, topic: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not self.gpt_cache_enabled:
            return None, None
        wanted = normalize_ws(topic).casefold()
        if not wanted:
            return None, None
        cache_dir = self._internal_cache_dir("first_module_result")
        if not cache_dir.is_dir():
            return None, None
        best_payload: Optional[Dict[str, Any]] = None
        best_path: Optional[Path] = None
        best_written = -1.0
        for path in cache_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            data = payload.get("data")
            if not isinstance(data, dict):
                continue
            if normalize_ws(data.get("topic")).casefold() != wanted:
                continue
            try:
                written = float(payload.get("written_at_unix", 0.0) or 0.0)
            except Exception:
                written = 0.0
            if best_payload is None or written > best_written:
                best_payload = data
                best_path = path
                best_written = written
        if best_payload is None:
            return None, None
        return best_payload, str(best_path) if best_path is not None else None

    def _store_first_module_result_cache(self, topic: str, result: Dict[str, Any]) -> Optional[str]:
        if not self.gpt_cache_enabled:
            return None
        payload = {
            "topic": normalize_ws(topic),
            "chapter_cap": int(self.chapter_cap),
            "models": list(self.model_candidates),
            "qwen_model": self.qwen_model,
        }
        path, _key = self._store_cached_stage_data(
            stage_name="first_module_result",
            payload=payload,
            data=result,
        )
        return path

    def _load_logical_timeline_from_chapter_flow_cache(self, topic: str) -> Optional[List[Dict[str, Any]]]:
        if not self.gpt_cache_enabled:
            return None
        cache_dir = self._internal_cache_dir("chapter_flow")
        if not cache_dir.is_dir():
            return None
        topic_words = {
            part.lower()
            for part in re.findall(r"[A-Za-z0-9]+", normalize_ws(topic))
            if len(part) >= 4
        }
        chapters: Dict[str, Dict[str, Any]] = {}
        for path in cache_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, dict):
                continue
            chapter_id = normalize_ws(data.get("chapter_id"))
            logical_steps = data.get("logical_steps")
            if not chapter_id or not isinstance(logical_steps, dict) or not logical_steps:
                continue
            haystack = " ".join(
                [
                    normalize_ws(data.get("title")),
                    normalize_ws(json.dumps(data.get("image_requests") or [], ensure_ascii=False)),
                    normalize_ws(json.dumps(logical_steps, ensure_ascii=False)),
                ]
            ).lower()
            if topic_words and not all(word in haystack for word in topic_words):
                continue
            chapters[chapter_id] = {
                "chapter_id": chapter_id,
                "title": normalize_ws(data.get("title")),
                "steps": OrderedDict((str(k), str(v)) for k, v in logical_steps.items()),
            }
        if not chapters:
            return None
        return [
            chapters[key]
            for key in sorted(
                chapters.keys(),
                key=lambda item: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", item)],
            )
        ][: max(1, int(self.chapter_cap))]

    def _load_chapter_flow_cache_for_chapter(self, topic: str, chapter: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not self.gpt_cache_enabled:
            return None, None
        cache_dir = self._internal_cache_dir("chapter_flow")
        if not cache_dir.is_dir():
            return None, None
        wanted_id = normalize_ws(chapter.get("chapter_id"))
        wanted_title = normalize_ws(chapter.get("title")).casefold()
        wanted_steps = {
            str(key): normalize_ws(value)
            for key, value in (chapter.get("steps") or {}).items()
        }
        best_data: Optional[Dict[str, Any]] = None
        best_path: Optional[Path] = None
        best_score = -1
        best_written = -1.0
        for path in cache_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, dict):
                continue
            if wanted_id and normalize_ws(data.get("chapter_id")) != wanted_id:
                continue
            cached_steps = {
                str(key): normalize_ws(value)
                for key, value in ((data.get("logical_steps") or {}).items() if isinstance(data.get("logical_steps"), dict) else [])
            }
            score = 0
            if wanted_title and normalize_ws(data.get("title")).casefold() == wanted_title:
                score += 2
            if wanted_steps and cached_steps:
                shared = sum(1 for key, value in wanted_steps.items() if cached_steps.get(key) == value)
                score += shared * 4
            if data.get("speech_steps") and data.get("image_requests") is not None and data.get("text_requests") is not None:
                score += 3
            if not wanted_steps and wanted_title and wanted_title in normalize_ws(data.get("title")).casefold():
                score += 1
            try:
                written = float(payload.get("written_at_unix", 0.0) or 0.0)
            except Exception:
                written = 0.0
            if score > best_score or (score == best_score and written > best_written):
                best_data = data
                best_path = path
                best_score = score
                best_written = written
        if not isinstance(best_data, dict) or best_score < 3:
            return None, None
        return best_data, str(best_path) if best_path is not None else None

    def _generate_qwen_text_batch_cached(
        self,
        *,
        stage_name: str,
        prompts: List[str],
        system_prompt: str,
        generation: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not prompts:
            return {
                "responses": [],
                "qwen_ready": None,
                "cache_hits": 0,
                "cache_misses": 0,
            }

        cache_stage = f"qwen_text_exact.{str(stage_name or '').strip()}"
        responses: List[Optional[Dict[str, Any]]] = [None] * len(prompts)
        miss_indexes: List[int] = []
        miss_prompts: List[str] = []
        miss_keys: List[str] = []
        cache_hits = 0

        for idx, prompt in enumerate(prompts):
            cache_payload = {
                "model": self.qwen_model,
                "system_prompt": str(system_prompt or ""),
                "prompt": str(prompt or ""),
                "generation": dict(generation or {}),
            }
            key = self._stage_cache_key(stage_name=cache_stage, payload=cache_payload)
            cached = self._load_stage_cache(stage_name=cache_stage, key=key)
            cached_data = cached.get("data") if isinstance(cached, dict) else None
            cached_response = cached_data.get("response") if isinstance(cached_data, dict) else None
            if isinstance(cached_response, dict):
                responses[idx] = cached_response
                cache_hits += 1
                continue
            miss_indexes.append(idx)
            miss_prompts.append(str(prompt or ""))
            miss_keys.append(key)

        qwen_ready = None
        self._dbg(
            "Qwen text batch cache status",
            data={"stage": stage_name, "requests": len(prompts), "hits": cache_hits, "misses": len(miss_prompts)},
        )
        if miss_prompts:
            self._dbg("Qwen text batch starting", data={"stage": stage_name, "misses": len(miss_prompts)})
            qwen_ready = self._ensure_qwen_text_worker()
            batch_result = self.qwen_client.generate_text_batch(
                miss_prompts,
                system_prompt=system_prompt,
                generation=generation,
                use_tqdm=False,
            )
            self._dbg("Qwen text batch returned", data={"stage": stage_name, "misses": len(miss_prompts)})
            raw_rows = batch_result.get("responses")
            if not isinstance(raw_rows, list) or len(raw_rows) != len(miss_prompts):
                raise RuntimeError(f"Qwen {stage_name} response count does not match request count")
            for idx, key, raw_row, prompt in zip(miss_indexes, miss_keys, raw_rows, miss_prompts):
                one_row = dict(raw_row or {})
                responses[idx] = one_row
                self._store_stage_cache(
                    stage_name=cache_stage,
                    key=key,
                    data={
                        "response": one_row,
                        "prompt": prompt,
                        "system_prompt": str(system_prompt or ""),
                        "generation": dict(generation or {}),
                    },
                )

        final_rows = [row if isinstance(row, dict) else {"text": ""} for row in responses]
        return {
            "responses": final_rows,
            "qwen_ready": qwen_ready,
            "cache_hits": int(cache_hits),
            "cache_misses": int(len(miss_prompts)),
        }

    def _generate_qwen_vision_batch_cached(
        self,
        *,
        stage_name: str,
        requests: List[Dict[str, Any]],
        system_prompt: str,
        generation: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not requests:
            return {
                "responses": [],
                "qwen_ready": None,
                "cache_hits": 0,
                "cache_misses": 0,
            }

        cache_stage = f"qwen_vision_exact.{str(stage_name or '').strip()}"
        responses: List[Optional[Dict[str, Any]]] = [None] * len(requests)
        miss_indexes: List[int] = []
        miss_requests: List[Dict[str, Any]] = []
        miss_keys: List[str] = []
        cache_hits = 0

        for idx, request in enumerate(requests):
            images = [str(x) for x in (request.get("images") or []) if str(x).strip()]
            cache_payload = {
                "model": self.qwen_model,
                "system_prompt": str(system_prompt or ""),
                "prompt": str(request.get("prompt", "") or ""),
                "images": images,
                "image_mtimes": [
                    os.path.getmtime(path) if os.path.isfile(path) else None
                    for path in images
                ],
                "generation": dict(generation or {}),
            }
            key = self._stage_cache_key(stage_name=cache_stage, payload=cache_payload)
            cached = self._load_stage_cache(stage_name=cache_stage, key=key)
            cached_data = cached.get("data") if isinstance(cached, dict) else None
            cached_response = cached_data.get("response") if isinstance(cached_data, dict) else None
            if isinstance(cached_response, dict):
                responses[idx] = cached_response
                cache_hits += 1
                continue
            miss_indexes.append(idx)
            miss_requests.append({"prompt": str(request.get("prompt", "") or ""), "images": images})
            miss_keys.append(key)

        qwen_ready = None
        self._dbg(
            "Qwen vision batch cache status",
            data={"stage": stage_name, "requests": len(requests), "hits": cache_hits, "misses": len(miss_requests)},
        )
        if miss_requests:
            self._dbg("Qwen vision batch starting", data={"stage": stage_name, "misses": len(miss_requests)})
            qwen_ready = self._ensure_qwen_text_worker()
            batch_result = self.qwen_client.generate_vision_batch(
                miss_requests,
                system_prompt=system_prompt,
                generation=generation,
                use_tqdm=False,
            )
            self._dbg("Qwen vision batch returned", data={"stage": stage_name, "misses": len(miss_requests)})
            raw_rows = batch_result.get("responses")
            if not isinstance(raw_rows, list) or len(raw_rows) != len(miss_requests):
                raise RuntimeError(f"Qwen vision {stage_name} response count does not match request count")
            for idx, key, raw_row, request in zip(miss_indexes, miss_keys, raw_rows, miss_requests):
                one_row = dict(raw_row or {})
                responses[idx] = one_row
                self._store_stage_cache(
                    stage_name=cache_stage,
                    key=key,
                    data={
                        "response": one_row,
                        "request": request,
                        "system_prompt": str(system_prompt or ""),
                        "generation": dict(generation or {}),
                    },
                )

        final_rows = [row if isinstance(row, dict) else {"text": ""} for row in responses]
        return {
            "responses": final_rows,
            "qwen_ready": qwen_ready,
            "cache_hits": int(cache_hits),
            "cache_misses": int(len(miss_requests)),
        }

    def _llm_cache_key(
        self,
        *,
        input_items: List[Dict[str, str]],
        reasoning_effort: str,
        max_output_tokens: Optional[int],
        text_format: Optional[Dict[str, Any]],
    ) -> str:
        payload = {
            "models": list(self.model_candidates),
            "input": input_items,
            "reasoning_effort": str(reasoning_effort),
            "max_output_tokens": int(max_output_tokens) if max_output_tokens is not None else None,
            "text_format": text_format,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _persist_gpt_cache(self) -> None:
        if not self.gpt_cache_enabled:
            return
        try:
            path = Path(self.gpt_cache_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._gpt_cache, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _call_responses_text_cached(
        self,
        input_items: List[Dict[str, str]],
        *,
        reasoning_effort: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        text_format: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        effort = str(reasoning_effort or self.reasoning_effort)
        if not self.gpt_cache_enabled:
            return call_responses_text(
                self.client,
                self.model_candidates,
                input_items,
                reasoning_effort=effort,
                max_output_tokens=max_output_tokens,
                text_format=text_format,
            )

        cache_key = self._llm_cache_key(
            input_items=input_items,
            reasoning_effort=effort,
            max_output_tokens=max_output_tokens,
            text_format=text_format,
        )
        with self._gpt_cache_lock:
            hit = self._gpt_cache.get(cache_key)
            if isinstance(hit, dict):
                text = hit.get("text")
                model = hit.get("model")
                if isinstance(text, str) and isinstance(model, str):
                    self._dbg("LLM cache hit", data={"key": cache_key[:12], "model": model})
                    return text, model

        text, model = call_responses_text(
            self.client,
            self.model_candidates,
            input_items,
            reasoning_effort=effort,
            max_output_tokens=max_output_tokens,
            text_format=text_format,
        )
        with self._gpt_cache_lock:
            self._gpt_cache[cache_key] = {"text": text, "model": model}
            self._persist_gpt_cache()
        self._dbg("LLM cache store", data={"key": cache_key[:12], "model": model})
        return text, model

    def _start_minilm_loader(self) -> Tuple[threading.Thread, Dict[str, Any]]:
        state: Dict[str, Any] = {"ok": None, "error": None}

        def _load() -> None:
            before = self._gpu_memory_snapshot()
            state["gpu_before"] = before
            try:
                from shared_models import init_minilm_hot

                init_minilm_hot(
                    gpu_index=self.gpu_index,
                    cpu_threads=self.cpu_threads,
                    warmup=True,
                )
                state["ok"] = True
            except Exception as exc:
                state["ok"] = False
                state["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                after = self._gpu_memory_snapshot()
                state["gpu_after"] = after
                state["gpu_delta"] = self._gpu_memory_delta(before, after)

        thread = threading.Thread(target=_load, name="newtimeline_preload_minilm", daemon=False)
        thread.start()
        return thread, state

    def _start_siglip_text_loader(self) -> Tuple[threading.Thread, Dict[str, Any]]:
        state: Dict[str, Any] = {"ok": True, "error": None, "skipped": True, "reason": "disabled"}
        if not DEFAULT_PRELOAD_SIGLIP_TEXT:
            def _skip() -> None:
                return None

            thread = threading.Thread(target=_skip, name="newtimeline_preload_siglip_text_skipped", daemon=False)
            thread.start()
            return thread, state

        state = {"ok": None, "error": None, "skipped": False}

        def _load() -> None:
            before = self._gpu_memory_snapshot()
            state["gpu_before"] = before
            try:
                from shared_models import init_siglip_text_hot

                init_siglip_text_hot(
                    gpu_index=self.gpu_index,
                    cpu_threads=self.cpu_threads,
                    warmup=True,
                    prefer_fp8=True,
                )
                state["ok"] = True
            except Exception as exc:
                state["ok"] = False
                state["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                after = self._gpu_memory_snapshot()
                state["gpu_after"] = after
                state["gpu_delta"] = self._gpu_memory_delta(before, after)

        thread = threading.Thread(target=_load, name="newtimeline_preload_siglip_text", daemon=False)
        thread.start()
        return thread, state

    def _start_qwen_text_loader(self) -> Tuple[Optional[threading.Thread], Dict[str, Any]]:
        with self._qwen_load_lock:
            if self._qwen_load_thread is not None:
                if self._qwen_load_thread.is_alive() or self._qwen_load_state.get("ready"):
                    return self._qwen_load_thread, self._qwen_load_state
                self._qwen_load_thread = None

            self._qwen_load_state = {
                "started": True,
                "ready": False,
                "error": None,
                "response": None,
                "mode": "vision",
            }

            def _load() -> None:
                try:
                    status = self.qwen_client.status()
                    loaded = bool(status.get("loaded"))
                    sleeping = bool(status.get("sleeping"))
                    mode = str(status.get("mode") or status.get("sleep_mode") or "").strip()
                    if loaded and not sleeping and mode in {"vision", "text"}:
                        response = {"ok": True, "mode": mode, "already_loaded": True, "status": status}
                    elif loaded and sleeping and mode == "vision":
                        response = self.qwen_client.wake_vision(
                            model=self.qwen_model,
                            warmup=False,
                            gpu_memory_utilization=DEFAULT_QWEN_GPU_UTIL,
                            max_model_len=DEFAULT_QWEN_CONTEXT_LEN,
                        )
                    else:
                        response = {
                            "ok": False,
                            "error": "qwen_worker_not_hot",
                            "message": "NEWtimeline assumes the Qwen server worker is already loaded; no load/reload was attempted.",
                            "status": status,
                        }
                    self._qwen_sleeping = False
                    self._qwen_load_state["response"] = response
                    self._qwen_load_state["ready"] = bool(response.get("ok", False))
                    if not self._qwen_load_state["ready"]:
                        self._qwen_load_state["error"] = json.dumps(response, ensure_ascii=False)
                except Exception as exc:
                    self._qwen_load_state["ready"] = False
                    self._qwen_load_state["error"] = f"{type(exc).__name__}: {exc}"

            self._qwen_load_thread = threading.Thread(target=_load, name="newtimeline_qwen_status_wake", daemon=False)
            self._qwen_load_thread.start()
            return self._qwen_load_thread, self._qwen_load_state

    def _reset_qwen_text_loader_state(self) -> None:
        with self._qwen_load_lock:
            self._qwen_load_thread = None
            self._qwen_load_state = {
                "started": False,
                "ready": False,
                "error": None,
                "response": None,
                "mode": "vision",
            }

    def _ensure_qwen_text_worker(self) -> Dict[str, Any]:
        thread, state = self._start_qwen_text_loader()
        if thread is not None:
            thread.join()
        if state.get("ready"):
            return dict(state)
        try:
            status = self.qwen_client.status()
            loaded = bool(status.get("loaded"))
            sleeping = bool(status.get("sleeping"))
            mode = str(status.get("mode") or status.get("sleep_mode") or "").strip()
            if loaded and not sleeping and mode in {"vision", "text"}:
                response = {"ok": True, "mode": mode, "already_loaded": True, "status": status}
            elif loaded and sleeping and mode == "vision":
                response = self.qwen_client.wake_vision(
                    model=self.qwen_model,
                    warmup=False,
                    gpu_memory_utilization=DEFAULT_QWEN_GPU_UTIL,
                    max_model_len=DEFAULT_QWEN_CONTEXT_LEN,
                )
            else:
                response = {
                    "ok": False,
                    "error": "qwen_worker_not_hot",
                    "message": "NEWtimeline assumes the Qwen server worker is already loaded; no load/reload was attempted.",
                    "status": status,
                }
            self._qwen_sleeping = False
            state["response"] = response
            state["ready"] = bool(response.get("ok", False))
            state["error"] = None if state["ready"] else json.dumps(response, ensure_ascii=False)
        except Exception as exc:
            state["ready"] = False
            state["error"] = f"{type(exc).__name__}: {exc}"
        if not state.get("ready"):
            raise RuntimeError(f"Qwen text worker is not ready: {state.get('error')}")
        return dict(state)

    def _save_stage_debug(
        self,
        *,
        stage_name: str,
        chapter_id: Optional[str],
        step_id: Optional[str],
        request_payload: Any,
        raw_response: str,
        parsed_payload: Any,
    ) -> str:
        safe_chapter = normalize_ws(chapter_id).replace(".", "_") or "global"
        safe_step = normalize_ws(step_id).replace(".", "_") if step_id else ""
        filename = f"{stage_name}_{safe_chapter}"
        if safe_step:
            filename += f"_{safe_step}"
        path = Path(self.debug_out_dir) / "LLM_RAW_NEW" / f"{filename}.json"
        return self._write_json_file(
            path,
            {
                "stage": stage_name,
                "chapter_id": chapter_id,
                "step_id": step_id,
                "request": request_payload,
                "raw_response": raw_response,
                "parsed": parsed_payload,
            },
        )

    def generate_logical_timeline(self, topic: str) -> Tuple[List[Dict[str, Any]], str]:
        cache_payload = {
            "topic": normalize_ws(topic),
            "chapter_cap": int(self.chapter_cap),
            "models": list(self.model_candidates),
        }
        cached, cache_key = self._load_cached_stage_data(stage_name="logical_timeline", payload=cache_payload)
        if self.gpt_cache_enabled and isinstance(cached, dict) and isinstance(cached.get("chapters"), list):
            self._dbg("Logical timeline stage cache hit", data={"key": cache_key[:12]})
            return list(cached.get("chapters") or []), str(cached.get("model", "cache"))

        chapter_flow_chapters = self._load_logical_timeline_from_chapter_flow_cache(topic)
        if chapter_flow_chapters:
            self._dbg(
                "Logical timeline reconstructed from chapter_flow cache",
                data={"chapters": len(chapter_flow_chapters)},
            )
            if self.gpt_cache_enabled:
                self._store_cached_stage_data(
                    stage_name="logical_timeline",
                    payload=cache_payload,
                    data={
                        "chapters": chapter_flow_chapters,
                        "model": "chapter_flow_cache",
                    },
                )
            return chapter_flow_chapters, "chapter_flow_cache"

        input_items, text_format = build_logical_timeline_request(topic, chapter_cap=self.chapter_cap)
        raw_text, model = self._call_responses_text_cached(
            input_items,
            reasoning_effort=self.reasoning_effort,
            text_format=text_format,
            max_output_tokens=5000,
        )
        chapters = parse_logical_timeline_output(raw_text, chapter_cap=self.chapter_cap)
        self._save_stage_debug(
            stage_name="logical_timeline",
            chapter_id=None,
            step_id=None,
            request_payload=input_items,
            raw_response=raw_text,
            parsed_payload=chapters,
        )
        if self.gpt_cache_enabled:
            self._store_cached_stage_data(
                stage_name="logical_timeline",
                payload=cache_payload,
                data={
                    "chapters": chapters,
                    "model": model,
                },
            )
        return chapters, model

    def generate_chapter_speech(self, topic: str, chapter: Dict[str, Any]) -> Tuple["OrderedDict[str, str]", str]:
        input_items, text_format = build_chapter_speech_request(topic, chapter)
        raw_text, model = self._call_responses_text_cached(
            input_items,
            reasoning_effort=self.reasoning_effort,
            text_format=text_format,
            max_output_tokens=7000,
        )
        speech_steps = parse_chapter_speech_output(raw_text, chapter=chapter)
        self._save_stage_debug(
            stage_name="chapter_speech",
            chapter_id=str(chapter.get("chapter_id")),
            step_id=None,
            request_payload=input_items,
            raw_response=raw_text,
            parsed_payload=speech_steps,
        )
        return speech_steps, model

    def generate_chapter_images(self, topic: str, chapter: Dict[str, Any], speech_steps: Dict[str, str]) -> Tuple[List[Dict[str, Any]], str]:
        input_items, text_format = build_image_request_request(topic, chapter, speech_steps)
        raw_text, model = self._call_responses_text_cached(
            input_items,
            reasoning_effort=self.reasoning_effort,
            text_format=text_format,
            max_output_tokens=3500,
        )
        image_requests = parse_image_request_output(raw_text, chapter=chapter)
        self._save_stage_debug(
            stage_name="chapter_images",
            chapter_id=str(chapter.get("chapter_id")),
            step_id=None,
            request_payload=input_items,
            raw_response=raw_text,
            parsed_payload=image_requests,
        )
        return image_requests, model

    def generate_chapter_texts(self, topic: str, chapter: Dict[str, Any], speech_steps: Dict[str, str]) -> Tuple[List[Dict[str, Any]], str]:
        input_items, text_format = build_text_request_request(topic, chapter, speech_steps)
        raw_text, model = self._call_responses_text_cached(
            input_items,
            reasoning_effort=self.reasoning_effort,
            text_format=text_format,
            max_output_tokens=3500,
        )
        text_requests = parse_text_request_output(raw_text, chapter=chapter)
        self._save_stage_debug(
            stage_name="chapter_texts",
            chapter_id=str(chapter.get("chapter_id")),
            step_id=None,
            request_payload=input_items,
            raw_response=raw_text,
            parsed_payload=text_requests,
        )
        return text_requests, model

    def _run_chapter_flow(self, topic: str, chapter: Dict[str, Any]) -> Dict[str, Any]:
        cache_payload = {
            "topic": normalize_ws(topic),
            "chapter": {
                "chapter_id": str(chapter.get("chapter_id", "") or ""),
                "title": str(chapter.get("title", "") or ""),
                "steps": dict(chapter.get("steps") or {}),
            },
            "models": list(self.model_candidates),
        }
        cached, cache_key = self._load_cached_stage_data(stage_name="chapter_flow", payload=cache_payload)
        if self.gpt_cache_enabled and isinstance(cached, dict) and cached.get("chapter_id"):
            self._dbg("Chapter flow stage cache hit", data={"chapter_id": cached.get("chapter_id"), "key": cache_key[:12]})
            return cached

        cached_by_chapter, cached_by_chapter_path = self._load_chapter_flow_cache_for_chapter(topic, chapter)
        if isinstance(cached_by_chapter, dict) and cached_by_chapter.get("chapter_id"):
            self._dbg(
                "Chapter flow existing cache hit",
                data={"chapter_id": cached_by_chapter.get("chapter_id"), "path": cached_by_chapter_path},
            )
            return cached_by_chapter

        speech_steps, speech_model = self.generate_chapter_speech(topic, chapter)

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"chapter_{chapter['chapter_id']}_objects") as executor:
            future_images = executor.submit(self.generate_chapter_images, topic, chapter, speech_steps)
            future_texts = executor.submit(self.generate_chapter_texts, topic, chapter, speech_steps)
            image_requests, image_model = future_images.result()
            text_requests, text_model = future_texts.result()

        chapter_out = {
            "chapter_id": str(chapter.get("chapter_id")),
            "title": str(chapter.get("title")),
            "logical_steps": OrderedDict((str(k), str(v)) for k, v in (chapter.get("steps") or {}).items()),
            "speech_steps": OrderedDict((str(k), str(v)) for k, v in speech_steps.items()),
            "image_requests": image_requests,
            "text_requests": text_requests,
            "qwen_steps": OrderedDict(),
            "step_objects": OrderedDict(),
            "models_used": {
                "speech": speech_model,
                "image_requests": image_model,
                "text_requests": text_model,
            },
        }
        if self.gpt_cache_enabled:
            self._store_cached_stage_data(
                stage_name="chapter_flow",
                payload=cache_payload,
                data=chapter_out,
            )
        return chapter_out

    def _get_legacy_group_helper(self):
        if self._legacy_group_helper is not None:
            return self._legacy_group_helper
        try:
            from timeline import LessonTimeline
        except ModuleNotFoundError:
            try:
                from UNUSED.timeline import LessonTimeline
            except ModuleNotFoundError:
                from whiteboard_backend.UNUSED.timeline import LessonTimeline

        self._legacy_group_helper = LessonTimeline(
            model_candidates=list(self.model_candidates),
            reasoning_effort=self.reasoning_effort,
            gpu_index=self.gpu_index,
            cpu_threads=self.cpu_threads,
            debug_print=self.debug_print,
            debug_out_dir=self.debug_out_dir,
            gpt_cache=False,
        )
        return self._legacy_group_helper

    def apply_repeating_diagram_filter(self, *, topic: str, chapters_out: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_prompts_meta: Dict[str, Dict[str, Any]] = {}
        all_diagram_required_objects: Dict[str, List[str]] = {}
        fake_chapters: List[Dict[str, Any]] = []

        for chapter in chapters_out:
            fake_images: List[Dict[str, Any]] = []
            fake_maps: List[Dict[str, Any]] = []
            for index, row in enumerate(chapter.get("image_requests") or []):
                if int(row.get("diagram", 0) or 0) != 1:
                    continue
                name = normalize_ws(row.get("name"))
                if not name:
                    continue
                required_objects = [normalize_ws(item) for item in (row.get("required_objects") or []) if normalize_ws(item)]
                if name not in all_prompts_meta:
                    all_prompts_meta[name] = {"topic": normalize_ws(topic), "diagram": 1}
                if name not in all_diagram_required_objects:
                    all_diagram_required_objects[name] = required_objects
                payload = {
                    "__index": index,
                    "content": name,
                    "diagram": 1,
                    "text": 0,
                    "diagram_required_objects": required_objects,
                }
                fake_images.append(dict(payload))
                fake_maps.append(
                    {
                        "__index": index,
                        "query": name,
                        "content": name,
                        "diagram": 1,
                        "text_tag": 0,
                        "diagram_required_objects": required_objects,
                    }
                )
            fake_chapters.append({"image_plan": {"images": fake_images}, "image_text_maps": fake_maps})

        if len(all_prompts_meta) < 2:
            return {
                "ok": True,
                "note": "not_enough_diagram_prompts",
                "replacement_map": {},
            }

        helper = self._get_legacy_group_helper()
        debug = helper._group_and_rewrite_diagram_prompts(
            all_prompts_meta=all_prompts_meta,
            all_diagram_required_objects=all_diagram_required_objects,
            chapters_out=fake_chapters,
            similarity_threshold=0.84,
            relaxed_topic_similarity=0.56,
            centroid_margin_for_umbrella=0.06,
        )

        for chapter, fake in zip(chapters_out, fake_chapters):
            fake_images = (((fake or {}).get("image_plan") or {}).get("images") or [])
            for row in fake_images:
                if not isinstance(row, dict):
                    continue
                try:
                    index = int(row.get("__index"))
                except Exception:
                    continue
                if not (0 <= index < len(chapter.get("image_requests") or [])):
                    continue
                chapter["image_requests"][index]["name"] = normalize_ws(row.get("content"))
                chapter["image_requests"][index]["diagram"] = 1 if int(row.get("diagram", 0) or 0) == 1 else 0
                chapter["image_requests"][index]["required_objects"] = [
                    normalize_ws(item)
                    for item in (row.get("diagram_required_objects") or [])
                    if normalize_ws(item)
                ]
        return debug

    def start_diagram_research(self, *, topic: str, chapters_out: List[Dict[str, Any]]) -> Dict[str, Any]:
        diagram_prompts: Dict[str, str] = {}
        for chapter in chapters_out:
            for row in chapter.get("image_requests") or []:
                if int(row.get("diagram", 0) or 0) != 1:
                    continue
                name = normalize_ws(row.get("name"))
                if name and name not in diagram_prompts:
                    diagram_prompts[name] = normalize_ws(topic)

        state: Dict[str, Any] = {
            "started": bool(diagram_prompts),
            "prompt_count": len(diagram_prompts),
            "prompts": list(diagram_prompts.keys()),
            "error": None,
        }
        if not diagram_prompts:
            state["note"] = "no_diagram_prompts"
            return state

        def _research() -> None:
            try:
                ImageResearcher = self._import_image_researcher()

                research_many = getattr(ImageResearcher, "research_many_mechanical", None)
                if callable(research_many):
                    research_many(
                        diagram_prompts,
                        max_workers=max(1, len(diagram_prompts)),
                        reset_images_dir=False,
                    )
                    return

                research_one = getattr(ImageResearcher, "research", None)
                if not callable(research_one):
                    raise RuntimeError("ImageResearcher.research_many_mechanical and ImageResearcher.research are both unavailable")

                with ThreadPoolExecutor(max_workers=max(1, len(diagram_prompts)), thread_name_prefix="newtimeline_research") as executor:
                    futures = [
                        executor.submit(research_one, prompt, prompt_topic, None, False, None)
                        for prompt, prompt_topic in diagram_prompts.items()
                    ]
                    for future in as_completed(futures):
                        future.result()
            except Exception as exc:
                state["error"] = f"{type(exc).__name__}: {exc}"

        thread = threading.Thread(target=_research, name="newtimeline_diagram_research", daemon=True)
        thread.start()
        state["thread_name"] = thread.name
        self._diagram_research_state = state
        self._diagram_research_thread = thread
        return state

    def run_qwen_step_batch(self, *, chapters_out: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts: List[str] = []
        metas: List[Dict[str, Any]] = []
        system_prompt: Optional[str] = None
        for chapter in chapters_out:
            speech_steps = chapter.get("speech_steps") or {}
            image_requests = chapter.get("image_requests") or []
            text_requests = chapter.get("text_requests") or []
            for step_id, speech_text in speech_steps.items():
                step_objects = collect_step_objects(
                    step_id=str(step_id),
                    image_requests=image_requests,
                    text_requests=text_requests,
                )
                chapter["step_objects"][str(step_id)] = step_objects
                step_system_prompt, prompt = build_qwen_step_prompt(
                    step_id=str(step_id),
                    speech_text=str(speech_text or ""),
                    step_objects=step_objects,
                )
                system_prompt = system_prompt or step_system_prompt
                prompts.append(prompt)
                metas.append(
                    {
                        "chapter_id": str(chapter.get("chapter_id")),
                        "step_id": str(step_id),
                        "object_names": [str(row.get("name")) for row in step_objects if str(row.get("name", "")).strip()],
                    }
                )

        if not prompts:
            return {
                "enabled": True,
                "request_count": 0,
                "responses": [],
                "qwen_ready": None,
                "cache_hits": 0,
                "cache_misses": 0,
            }

        response = self._generate_qwen_text_batch_cached(
            stage_name="speech_compression",
            prompts=prompts,
            system_prompt=system_prompt,
            generation={
                "max_new_tokens": 1800,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.02,
            },
        )

        responses = response.get("responses")
        if not isinstance(responses, list) or len(responses) != len(metas):
            raise RuntimeError("Qwen batch response count does not match request count")

        synced_objects: List[Dict[str, Any]] = []
        stage_io_dir = Path(self.debug_out_dir) / "qwen_stage_io"
        stage_io_dir.mkdir(parents=True, exist_ok=True)

        chapter_map = {str(chapter.get("chapter_id")): chapter for chapter in chapters_out}
        for index, (meta, raw_row) in enumerate(zip(metas, responses), start=1):
            raw_text = str((raw_row or {}).get("text", "") or "").strip()
            parsed = parse_qwen_step_output(raw_text, allowed_object_names=meta["object_names"])
            chapter = chapter_map[meta["chapter_id"]]
            chapter["qwen_steps"][meta["step_id"]] = {
                "speech": parsed["speech"],
                "sync_map": parsed["sync_map"],
                "pause_markers": extract_pause_markers(parsed["speech"]),
            }

            sync_lookup = {str(row["name"]): row for row in parsed["sync_map"]}
            step_objects = chapter["step_objects"].get(meta["step_id"]) or []
            for obj in step_objects:
                sync = sync_lookup.get(str(obj.get("name")))
                synced_objects.append(
                    {
                        "object_type": str(obj.get("object_type")),
                        "chapter_id": meta["chapter_id"],
                        "step_id": meta["step_id"],
                        "name": str(obj.get("name")),
                        "diagram": int(obj.get("diagram", 0) or 0) if obj.get("object_type") == "image" else 0,
                        "required_objects": list(obj.get("required_objects") or []),
                        "text_style_description": str(obj.get("text_style_description", "") or ""),
                        "start_silence_index": None if sync is None else int(sync.get("start", 0)),
                        "end_silence_index": None if sync is None or sync.get("end") is None else int(sync.get("end")),
                    }
                )

            file_path = stage_io_dir / f"{index:05d}_step_{meta['step_id'].replace('.', '_')}.json"
            self._write_json_file(
                file_path,
                {
                    "request_index": index,
                    "chapter_id": meta["chapter_id"],
                    "step_id": meta["step_id"],
                    "prompt": prompts[index - 1],
                    "raw_response": raw_text,
                    "parsed_response": parsed,
                    "usage": raw_row,
                },
            )

        return {
            "enabled": True,
            "request_count": len(prompts),
            "responses": responses,
            "qwen_ready": response.get("qwen_ready"),
            "cache_hits": int(response.get("cache_hits", 0) or 0),
            "cache_misses": int(response.get("cache_misses", 0) or 0),
            "synced_objects": synced_objects,
        }

    def _ensure_first_module_qwen_sync(self, first_module_result: Dict[str, Any]) -> Dict[str, Any]:
        chapters_out = list(first_module_result.get("chapters") or [])
        existing_synced = list(first_module_result.get("synced_objects") or [])
        existing_qwen_steps = 0
        for chapter in chapters_out:
            qwen_steps = chapter.get("qwen_steps") if isinstance(chapter, dict) and isinstance(chapter.get("qwen_steps"), dict) else {}
            existing_qwen_steps += len(qwen_steps)
        if existing_synced and existing_qwen_steps:
            return {
                "enabled": True,
                "reused": True,
                "synced_objects": existing_synced,
                "existing_qwen_steps": existing_qwen_steps,
            }
        self._dbg(
            "First module cache missing Qwen speech sync; running speech compression now",
            data={"chapters": len(chapters_out), "existing_synced_objects": len(existing_synced), "existing_qwen_steps": existing_qwen_steps},
        )
        qwen_debug = self.run_qwen_step_batch(chapters_out=chapters_out)
        first_module_result["chapters"] = chapters_out
        first_module_result["synced_objects"] = list(qwen_debug.get("synced_objects") or [])
        first_module_result["qwen"] = {
            key: value
            for key, value in qwen_debug.items()
            if key != "responses"
        }
        self._dbg(
            "First module Qwen speech sync ready",
            data={
                "request_count": int(qwen_debug.get("request_count", 0) or 0),
                "synced_objects": len(first_module_result.get("synced_objects") or []),
                "cache_hits": int(qwen_debug.get("cache_hits", 0) or 0),
                "cache_misses": int(qwen_debug.get("cache_misses", 0) or 0),
            },
        )
        return qwen_debug

    def run_first_module(self, topic: str) -> Dict[str, Any]:
        run_started_at = time.time()
        cached_result, cached_path = self._load_first_module_result_cache(topic)
        if isinstance(cached_result, dict):
            out = dict(cached_result)
            out["cache_hit"] = True
            out["cache_path"] = cached_path
            out["loaded_at_unix"] = time.time()
            self._dbg("First module result cache hit", data={"topic": normalize_ws(topic), "path": cached_path})
            return out

        chapters, timeline_model = self.generate_logical_timeline(topic)

        chapter_slots: List[Optional[Dict[str, Any]]] = [None] * len(chapters)
        with ThreadPoolExecutor(max_workers=max(1, len(chapters)), thread_name_prefix="newtimeline_chapter") as executor:
            future_map = {
                executor.submit(self._run_chapter_flow, topic, chapter): index
                for index, chapter in enumerate(chapters)
            }
            for future in as_completed(future_map):
                index = future_map[future]
                chapter_out = future.result()
                chapter_slots[index] = chapter_out
                self._dbg(
                    "Chapter speech + object requests complete",
                    data={
                        "chapter_id": chapter_out.get("chapter_id"),
                        "image_requests": len(chapter_out.get("image_requests") or []),
                        "text_requests": len(chapter_out.get("text_requests") or []),
                    },
                )

        chapters_out = [chapter for chapter in chapter_slots if chapter is not None]

        preload_threads: List[Tuple[str, threading.Thread, Dict[str, Any]]] = []
        minilm_thread, minilm_state = self._start_minilm_loader()
        preload_threads.append(("minilm", minilm_thread, minilm_state))
        if DEFAULT_PRELOAD_SIGLIP_TEXT:
            siglip_thread, siglip_state = self._start_siglip_text_loader()
            preload_threads.append(("siglip_text", siglip_thread, siglip_state))
        else:
            self._dbg("SigLIP text preload skipped", data={"env": "NEWTIMELINE_PRELOAD_SIGLIP_TEXT", "default": False})
        qwen_thread, qwen_state = self._start_qwen_text_loader()
        if qwen_thread is not None:
            preload_threads.append(("qwen_text", qwen_thread, qwen_state))

        preload_status: Dict[str, Any] = {}
        for label, thread, state in preload_threads:
            thread.join()
            preload_status[label] = dict(state)

        repeat_filter_debug = self.apply_repeating_diagram_filter(topic=topic, chapters_out=chapters_out)
        research_state = self.start_diagram_research(topic=topic, chapters_out=chapters_out)

        result = {
            "topic": normalize_ws(topic),
            "started_at_unix": run_started_at,
            "finished_at_unix": time.time(),
            "cache_hit": False,
            "models_used": {
                "logical_timeline": timeline_model,
                "speech": [chapter["models_used"]["speech"] for chapter in chapters_out],
                "image_requests": [chapter["models_used"]["image_requests"] for chapter in chapters_out],
                "text_requests": [chapter["models_used"]["text_requests"] for chapter in chapters_out],
                "qwen_server_model": self.qwen_model,
            },
            "preload_status": preload_status,
            "logical_timeline": [
                {
                    "chapter_id": chapter["chapter_id"],
                    "title": chapter["title"],
                    "steps": chapter["logical_steps"],
                }
                for chapter in chapters_out
            ],
            "chapters": chapters_out,
            "diagram_repeat_filter": repeat_filter_debug,
            "diagram_research": research_state,
            "qwen": {
                "enabled": True,
                "request_count": 0,
                "note": "deferred_until_after_research_c2_siglip",
            },
            "synced_objects": [],
        }
        cache_path = self._store_first_module_result_cache(topic, result)
        if cache_path:
            result["cache_path"] = cache_path
        return result

    def _collect_prompt_meta(self, *, topic: str, chapters_out: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        prompt_meta: Dict[str, Dict[str, Any]] = {}
        for chapter in chapters_out:
            chapter_id = str(chapter.get("chapter_id", "") or "").strip()
            for row in chapter.get("image_requests") or []:
                name = normalize_ws(row.get("name"))
                if not name:
                    continue
                entry = prompt_meta.setdefault(
                    name,
                    {
                        "topic": normalize_ws(topic),
                        "diagram": 0,
                        "diagram_required_objects": [],
                        "chapter_ids": [],
                    },
                )
                entry["diagram"] = max(int(entry.get("diagram", 0) or 0), int(row.get("diagram", 0) or 0))
                if chapter_id and chapter_id not in entry["chapter_ids"]:
                    entry["chapter_ids"].append(chapter_id)
                for obj in row.get("required_objects") or []:
                    cleaned = normalize_ws(obj)
                    if cleaned and cleaned not in entry["diagram_required_objects"]:
                        entry["diagram_required_objects"].append(cleaned)
        return prompt_meta

    def _project_root_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def _dimensions_json_path(self) -> Path:
        return self._project_root_dir() / "dimentions.json"

    def _default_dimensions_config(self) -> Dict[str, Any]:
        return {
            "schema": "whiteboard_dimensions_v1",
            "whiteboard_size_px": {"width": 2000.0, "height": 2000.0},
            "normalize_size_px": 1000.0,
            "image": {"default_scale": 0.75},
            "text": {
                "default_scale": 0.25,
                "letter_gap_px": 20.0,
                "base_font_size_ref_px": 200.0,
                "space_width_factor": 0.5,
                "default_stroke_slowdown": 8.0,
                "stroke_base_time_sec": 0.017,
                "stroke_curve_extra_frac": 0.25,
                "letter_pause_sec": 0.0,
            },
        }

    def _load_dimensions_config(self) -> Dict[str, Any]:
        if self._dimensions_config_cache is not None:
            return self._dimensions_config_cache
        data = self._default_dimensions_config()
        path = self._dimensions_json_path()
        try:
            if path.is_file():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    merged = self._default_dimensions_config()
                    merged.update({k: v for k, v in raw.items() if k not in {"image", "text", "whiteboard_size_px"}})
                    if isinstance(raw.get("whiteboard_size_px"), dict):
                        merged["whiteboard_size_px"] = {
                            **dict(merged.get("whiteboard_size_px") or {}),
                            **dict(raw.get("whiteboard_size_px") or {}),
                        }
                    if isinstance(raw.get("image"), dict):
                        merged["image"] = {**dict(merged.get("image") or {}), **dict(raw.get("image") or {})}
                    if isinstance(raw.get("text"), dict):
                        merged["text"] = {**dict(merged.get("text") or {}), **dict(raw.get("text") or {})}
                    data = merged
        except Exception:
            data = self._default_dimensions_config()
        self._dimensions_config_cache = data
        return data

    def _font_metrics_path(self) -> Path:
        return Path(__file__).resolve().parent / "Font" / "font_metrics.json"

    def _load_normalized_font_metrics(self) -> Dict[str, float]:
        if self._font_metrics_cache is not None:
            return self._font_metrics_cache
        dims = self._load_dimensions_config()
        normalize_size = float(dims.get("normalize_size_px", 1000.0) or 1000.0)
        line_height = normalize_size * 0.5
        image_height = normalize_size
        image_width = normalize_size
        raw_path = self._font_metrics_path()
        try:
            if raw_path.is_file():
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    raw_image_height = float(payload.get("image_height", normalize_size) or normalize_size)
                    raw_image_width = float(payload.get("image_width", raw_image_height) or raw_image_height)
                    raw_line_height = float(payload.get("line_height_px", line_height) or line_height)
                    longest = max(raw_image_width, raw_image_height, 1.0)
                    factor = normalize_size / longest
                    line_height = raw_line_height * factor
                    image_height = raw_image_height * factor
                    image_width = raw_image_width * factor
        except Exception:
            pass
        self._font_metrics_cache = {
            "line_height_px": float(line_height),
            "image_height_px": float(image_height),
            "image_width_px": float(image_width),
            "normalize_size_px": float(normalize_size),
        }
        return self._font_metrics_cache

    def _font_glyph_path(self, code_unit: int) -> Path:
        hex_code = f"{int(code_unit):04x}"
        return Path(__file__).resolve().parent / "Font" / f"{hex_code}.json"

    def _load_glyph_metrics(self, code_unit: int) -> Optional[Dict[str, float]]:
        if code_unit in self._glyph_metrics_cache:
            return self._glyph_metrics_cache[code_unit]
        path = self._font_glyph_path(code_unit)
        if not path.is_file():
            return None
        dims = self._load_dimensions_config()
        normalize_size = float(dims.get("normalize_size_px", 1000.0) or 1000.0)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        strokes = payload.get("strokes")
        if not isinstance(strokes, list):
            return None
        raw_width = float(payload.get("width", normalize_size) or normalize_size)
        raw_height = float(payload.get("height", normalize_size) or normalize_size)
        longest = max(raw_width, raw_height, 1.0)
        factor = normalize_size / longest

        min_x: Optional[float] = None
        min_y: Optional[float] = None
        max_x: Optional[float] = None
        max_y: Optional[float] = None

        for stroke in strokes:
            if not isinstance(stroke, dict):
                continue
            segments = stroke.get("segments")
            if not isinstance(segments, list):
                continue
            for seg in segments:
                if not isinstance(seg, list) or len(seg) < 8:
                    continue
                xs = [float(seg[i]) * factor for i in (0, 2, 4, 6)]
                ys = [float(seg[i]) * factor for i in (1, 3, 5, 7)]
                cur_min_x = min(xs)
                cur_max_x = max(xs)
                cur_min_y = min(ys)
                cur_max_y = max(ys)
                min_x = cur_min_x if min_x is None else min(min_x, cur_min_x)
                max_x = cur_max_x if max_x is None else max(max_x, cur_max_x)
                min_y = cur_min_y if min_y is None else min(min_y, cur_min_y)
                max_y = cur_max_y if max_y is None else max(max_y, cur_max_y)

        if min_x is None or max_x is None or min_y is None or max_y is None:
            return None
        metrics = {
            "left": float(min_x),
            "top": float(min_y),
            "width": float(max(0.0, max_x - min_x)),
            "height": float(max(0.0, max_y - min_y)),
            "source_width": float(raw_width),
            "source_height": float(raw_height),
            "normalize_factor": float(factor),
        }
        self._glyph_metrics_cache[code_unit] = metrics
        return metrics

    def _text_dimensions_px(self, text: str, *, scale: Optional[float] = None) -> Dict[str, Any]:
        dims = self._load_dimensions_config()
        text_cfg = dims.get("text") if isinstance(dims.get("text"), dict) else {}
        metrics = self._load_normalized_font_metrics()
        default_text_scale = float((text_cfg or {}).get("default_scale", 0.25) or 0.25)
        safe_scale = float(scale if scale is not None else default_text_scale)
        if safe_scale <= 0:
            safe_scale = default_text_scale
        base_font_size_ref_px = float((text_cfg or {}).get("base_font_size_ref_px", 200.0) or 200.0)
        letter_size_px = base_font_size_ref_px * safe_scale
        letter_gap_px = float((text_cfg or {}).get("letter_gap_px", 20.0) or 20.0)
        space_width_factor = float((text_cfg or {}).get("space_width_factor", 0.5) or 0.5)
        line_height_px = float(metrics.get("line_height_px", max(1.0, letter_size_px)) or max(1.0, letter_size_px))
        font_scale = letter_size_px / max(1.0, line_height_px)

        cursor_x = 0.0
        max_height = 0.0
        visible_glyphs = 0
        normalized_text = str(text or "")

        for index, char in enumerate(normalized_text):
            if char == " ":
                cursor_x += letter_size_px * space_width_factor
                continue
            glyph_metrics = self._load_glyph_metrics(ord(char))
            if glyph_metrics is None:
                cursor_x += letter_size_px * space_width_factor
                continue
            glyph_width = float(glyph_metrics.get("width", 0.0) or 0.0) * font_scale
            glyph_height = float(glyph_metrics.get("height", 0.0) or 0.0) * font_scale
            cursor_x += glyph_width
            max_height = max(max_height, glyph_height)
            visible_glyphs += 1
            if index < len(normalized_text) - 1:
                cursor_x += letter_gap_px

        if not normalized_text:
            cursor_x = 0.0
        if max_height <= 0.0:
            max_height = letter_size_px

        return {
            "width": int(round(max(0.0, cursor_x))),
            "height": int(round(max(1.0, max_height))),
            "letter_size_px": float(letter_size_px),
            "letter_gap_px": float(letter_gap_px),
            "font_scale": float(font_scale),
            "line_height_px": float(line_height_px),
            "normalize_size_px": float(dims.get("normalize_size_px", 1000.0) or 1000.0),
            "base_font_size_ref_px": float(base_font_size_ref_px),
            "space_width_factor": float(space_width_factor),
            "visible_glyphs": int(visible_glyphs),
        }

    def _resolve_processed_png_path(self, processed_id: str) -> Optional[Path]:
        pid = str(processed_id or "").strip()
        if not pid:
            return None
        base = Path(__file__).resolve().parent
        candidates = [
            Path("ProcessedImages") / f"{pid}.png",
            Path("ProccessedImages") / f"{pid}.png",
            base / "ProcessedImages" / f"{pid}.png",
            base / "ProccessedImages" / f"{pid}.png",
            base / "PipelineOutputs" / "ProcessedImages" / f"{pid}.png",
            base / "PipelineOutputs" / "ProccessedImages" / f"{pid}.png",
        ]
        for path in candidates:
            try:
                if path.is_file():
                    return path
            except Exception:
                continue
        return None

    def _compute_bbox_px_for_asset(self, asset: ImageAsset) -> None:
        if asset.bbox_px is not None:
            return
        if not asset.processed_ids:
            asset.bbox_px = (400, 300)
            return
        img_path = self._resolve_processed_png_path(asset.processed_ids[0])
        if img_path is None:
            asset.bbox_px = (400, 300)
            return
        try:
            from PIL import Image

            with Image.open(img_path) as image:
                asset.bbox_px = (int(image.size[0]), int(image.size[1]))
        except Exception:
            asset.bbox_px = (400, 300)

    def _image_dimensions_from_processed_id(self, processed_id: str) -> Dict[str, Any]:
        path = self._resolve_processed_png_path(processed_id)
        if path is None:
            return {
                "width": 400,
                "height": 300,
                "processed_png_path": "",
                "exists": False,
            }
        try:
            from PIL import Image

            with Image.open(path) as image:
                width, height = image.size
            return {
                "width": int(width),
                "height": int(height),
                "processed_png_path": str(path),
                "exists": True,
            }
        except Exception:
            return {
                "width": 400,
                "height": 300,
                "processed_png_path": str(path),
                "exists": False,
            }

    def _normalize_processed_id(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if text.lower().endswith(".png") or text.lower().endswith(".json"):
            text = Path(text).stem
        if text.lower().startswith("processed_"):
            suffix = text.split("_", 1)[1]
            return f"processed_{int(suffix)}" if suffix.isdigit() else text
        if text.isdigit():
            return f"processed_{int(text)}"
        return text

    def _cluster_map_path_for_processed_id(self, processed_id: str) -> Path:
        return Path(__file__).resolve().parent / "ClusterMaps" / self._normalize_processed_id(processed_id) / "clusters.json"

    def _load_image_cluster_stroke_groups(self, processed_id: str) -> Dict[str, Any]:
        pid = self._normalize_processed_id(processed_id)
        path = self._cluster_map_path_for_processed_id(pid)
        if not path.is_file():
            return {"processed_id": pid, "cluster_map_path": str(path), "exists": False, "clusters": []}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "processed_id": pid,
                "cluster_map_path": str(path),
                "exists": False,
                "clusters": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        clusters: List[Dict[str, Any]] = []
        for idx, row in enumerate(list((payload or {}).get("clusters") or [])):
            if not isinstance(row, dict):
                continue
            strokes = sorted({int(v) for v in (row.get("stroke_indexes") or []) if str(v).strip().lstrip("-").isdigit()})
            if not strokes:
                continue
            clusters.append(
                {
                    "cluster_index": int(idx),
                    "stroke_indexes": strokes,
                    "crop_file_mask": str(row.get("crop_file_mask", "") or ""),
                    "color_name": str(row.get("color_name", "") or ""),
                    "bbox_xyxy": list(row.get("bbox_xyxy") or []),
                }
            )
        return {"processed_id": pid, "cluster_map_path": str(path), "exists": True, "clusters": clusters}

    def _score_sam_candidate_against_image_clusters(
        self,
        *,
        candidate: Dict[str, Any],
        image_clusters: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        candidate_strokes = sorted({int(v) for v in (candidate.get("stroke_indexes") or [])})
        if not candidate_strokes:
            return None
        cand_set = set(candidate_strokes)
        evidence: List[Dict[str, Any]] = []
        best_candidate_coverage = 0.0
        best_cluster_coverage = 0.0
        best_overlap_count = 0
        for cluster in image_clusters:
            cluster_strokes = sorted({int(v) for v in (cluster.get("stroke_indexes") or [])})
            if not cluster_strokes:
                continue
            cluster_set = set(cluster_strokes)
            overlap = sorted(cand_set & cluster_set)
            if not overlap:
                continue
            candidate_coverage = float(len(overlap)) / float(max(1, len(cand_set)))
            cluster_coverage = float(len(overlap)) / float(max(1, len(cluster_set)))
            strong_match = candidate_coverage >= 0.85 or cluster_coverage >= 0.85
            best_candidate_coverage = max(best_candidate_coverage, candidate_coverage)
            best_cluster_coverage = max(best_cluster_coverage, cluster_coverage)
            best_overlap_count = max(best_overlap_count, len(overlap))
            evidence.append(
                {
                    "cluster_index": int(cluster.get("cluster_index", -1)),
                    "overlap_count": int(len(overlap)),
                    "candidate_coverage": round(candidate_coverage, 4),
                    "cluster_coverage": round(cluster_coverage, 4),
                    "strong_match": bool(strong_match),
                    "crop_file_mask": str(cluster.get("crop_file_mask", "") or ""),
                    "color_name": str(cluster.get("color_name", "") or ""),
                }
            )
        if not evidence:
            return None
        evidence.sort(
            key=lambda row: (
                bool(row.get("strong_match")),
                float(row.get("candidate_coverage", 0.0) or 0.0),
                float(row.get("cluster_coverage", 0.0) or 0.0),
                int(row.get("overlap_count", 0) or 0),
            ),
            reverse=True,
        )
        strong_count = sum(1 for row in evidence if row.get("strong_match"))
        confidence = (
            0.20
            + 0.50 * best_candidate_coverage
            + 0.20 * best_cluster_coverage
            + 0.06 * min(1.0, best_overlap_count / float(max(1, len(cand_set))))
            + 0.04 * min(1.0, strong_count / 2.0)
        )
        out = dict(candidate)
        out["stroke_indexes"] = candidate_strokes
        out["image_cluster_confidence"] = round(min(1.0, confidence), 4)
        out["image_cluster_evidence"] = evidence[:6]
        out["best_image_cluster_candidate_coverage"] = round(best_candidate_coverage, 4)
        out["best_image_cluster_cluster_coverage"] = round(best_cluster_coverage, 4)
        out["strong_image_cluster_match_count"] = int(strong_count)
        return out

    def _sleep_qwen_text_worker(self) -> Optional[str]:
        try:
            response = self.qwen_client.sleep()
            self._qwen_sleeping = True
            self._reset_qwen_text_loader_state()
            if isinstance(response, dict) and response.get("ok") is False:
                return json.dumps(response, ensure_ascii=False)
            return None
        except Exception as exc:
            self._reset_qwen_text_loader_state()
            return f"{type(exc).__name__}: {exc}"

    def _stroke_description_path_for_processed_id(self, processed_id: str) -> Path:
        return Path(__file__).resolve().parent / "StrokeDescriptions" / f"{self._normalize_processed_id(processed_id)}_described.json"

    def _load_stroke_description_payload(self, processed_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
        path = self._stroke_description_path_for_processed_id(processed_id)
        if not path.is_file():
            return None, str(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None, str(path)
        return (payload if isinstance(payload, dict) else None), str(path)

    def run_diagram_stroke_meaning_filter_batch(
        self,
        *,
        prompt_meta: Dict[str, Dict[str, Any]],
        selected_ids_by_prompt: Dict[str, List[str]],
        pipeline_handle: Any = None,
    ) -> Dict[str, Any]:
        wait_ok = True
        wait_error: Optional[str] = None
        try:
            wait_fn = getattr(pipeline_handle, "wait_line_descriptors", None)
            if callable(wait_fn):
                wait_ok = bool(wait_fn(timeout=None))

            jobs_by_pid: Dict[str, Dict[str, Any]] = {}
            for prompt, meta in (prompt_meta or {}).items():
                if int((meta or {}).get("diagram", 0) or 0) != 1:
                    continue
                clean_prompt = normalize_ws(prompt)
                components = [normalize_ws(item) for item in ((meta or {}).get("diagram_required_objects") or []) if normalize_ws(item)]
                for raw_pid in selected_ids_by_prompt.get(prompt, []) or []:
                    pid = self._normalize_processed_id(raw_pid)
                    if not pid:
                        continue
                    job = jobs_by_pid.setdefault(
                        pid,
                        {
                            "processed_id": pid,
                            "diagram_names": [],
                            "components": [],
                        },
                    )
                    if clean_prompt and clean_prompt not in job["diagram_names"]:
                        job["diagram_names"].append(clean_prompt)
                    for component in components:
                        if component and component not in job["components"]:
                            job["components"].append(component)

            if not wait_ok:
                wait_error = "line_descriptors_wait_failed"
            if not jobs_by_pid:
                return {
                    "enabled": True,
                    "request_count": 0,
                    "by_processed_id": {},
                    "wait_ok": bool(wait_ok),
                    "wait_error": wait_error,
                    "note": "no_diagram_processed_ids",
                }

            prompts: List[str] = []
            metas: List[Dict[str, Any]] = []
            descriptors: Dict[str, Dict[str, Any]] = {}
            descriptor_paths: Dict[str, str] = {}
            system_prompt: Optional[str] = None
            errors: Dict[str, str] = {}

            for pid, job in sorted(jobs_by_pid.items()):
                descriptor, descriptor_path = self._load_stroke_description_payload(pid)
                descriptor_paths[pid] = descriptor_path
                if not isinstance(descriptor, dict) or not list(descriptor.get("described_lines") or []):
                    errors[pid] = "missing_or_empty_stroke_description"
                    continue
                diagram_name = " / ".join(job.get("diagram_names") or [pid])
                sys_prompt, user_prompt = build_qwen_stroke_meaning_filter_prompt(
                    diagram_name=diagram_name,
                    components=list(job.get("components") or []),
                    stroke_description_payload=descriptor,
                )
                system_prompt = system_prompt or sys_prompt
                prompts.append(user_prompt)
                metas.append(
                    {
                        "processed_id": pid,
                        "diagram_name": diagram_name,
                        "components": list(job.get("components") or []),
                        "stroke_description_path": descriptor_path,
                    }
                )
                descriptors[pid] = descriptor

            if not prompts:
                result = {
                    "enabled": True,
                    "request_count": 0,
                    "by_processed_id": {},
                    "descriptor_paths": descriptor_paths,
                    "errors": errors,
                    "wait_ok": bool(wait_ok),
                    "wait_error": wait_error,
                }
                result["saved_path"] = self._write_json_file(self._internal_runtime_dir() / "diagram_stroke_meaning_payload.json", result)
                return result

            qwen_result = self._generate_qwen_text_batch_cached(
                stage_name="diagram_stroke_meaning_filter",
                prompts=prompts,
                system_prompt=system_prompt or "",
                generation={
                    "max_new_tokens": 2200,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repetition_penalty": 1.03,
                },
            )
            responses = qwen_result.get("responses") if isinstance(qwen_result.get("responses"), list) else []
            by_processed_id: Dict[str, Any] = {}
            raw_io_rows: List[Dict[str, Any]] = []
            for index, (meta, raw_row, prompt_text) in enumerate(zip(metas, responses, prompts), start=1):
                pid = str(meta.get("processed_id"))
                raw_text = str((raw_row or {}).get("text", "") or "").strip()
                parsed = parse_qwen_stroke_meaning_filter_output(
                    raw_text,
                    stroke_description_payload=descriptors.get(pid, {}),
                )
                by_processed_id[pid] = {
                    "processed_id": pid,
                    "diagram_name": meta.get("diagram_name"),
                    "components": list(meta.get("components") or []),
                    "stroke_description_path": meta.get("stroke_description_path"),
                    "optimized": parsed,
                }
                raw_io_rows.append(
                    {
                        "request_index": index,
                        "processed_id": pid,
                        "prompt": prompt_text,
                        "raw_response": raw_text,
                        "parsed": parsed,
                        "usage": raw_row,
                    }
                )

            raw_path = self._write_json_file(
                self._internal_runtime_dir() / "qwen_raw_io" / "diagram_stroke_meaning_filter.json",
                {
                    "system_prompt": system_prompt,
                    "rows": raw_io_rows,
                },
            )
            result = {
                "enabled": True,
                "request_count": len(prompts),
                "by_processed_id": by_processed_id,
                "descriptor_paths": descriptor_paths,
                "errors": errors,
                "wait_ok": bool(wait_ok),
                "wait_error": wait_error,
                "raw_io_path": raw_path,
                "cache_hits": int(qwen_result.get("cache_hits", 0) or 0),
                "cache_misses": int(qwen_result.get("cache_misses", 0) or 0),
            }
            result["saved_path"] = self._write_json_file(self._internal_runtime_dir() / "diagram_stroke_meaning_payload.json", result)
            return result
        except Exception as exc:
            return {
                "enabled": False,
                "error": f"{type(exc).__name__}: {exc}",
                "wait_ok": bool(wait_ok),
                "wait_error": wait_error,
                "by_processed_id": {},
            }
        finally:
            release_fn = getattr(pipeline_handle, "release_after_line_descriptors", None)
            if callable(release_fn):
                release_fn()

    def _prepare_diagram_object_candidates(
        self,
        *,
        selected_ids_by_prompt: Dict[str, List[str]],
        pipeline_handle: Any = None,
    ) -> Dict[str, Any]:
        processed_ids = sorted(
            {
                self._normalize_processed_id(pid)
                for ids in (selected_ids_by_prompt or {}).values()
                for pid in (ids or [])
                if self._normalize_processed_id(pid)
            }
        )
        out: Dict[str, Any] = {
            "schema": "newtimeline_diagram_object_candidates_v1",
            "processed_ids": processed_ids,
            "by_processed_id": {},
            "flat_candidates": [],
            "errors": {},
        }
        if not processed_ids:
            out["note"] = "no_selected_diagram_processed_ids"
            return out

        self._dbg("Diagram object candidates starting", data={"processed_ids": processed_ids})
        if pipeline_handle is not None:
            try:
                pipeline_done_evt = getattr(pipeline_handle, "pipeline_done", None)
                wait_pipeline = getattr(pipeline_done_evt, "wait", None)
                if callable(wait_pipeline):
                    self._dbg("Diagram object candidates waiting for image pipeline clusters")
                    wait_pipeline(timeout=None)
                    self._dbg("Diagram object candidates image pipeline clusters ready")
                else:
                    wait_fn = getattr(pipeline_handle, "wait_colours", None)
                    if callable(wait_fn):
                        self._dbg("Diagram object candidates waiting for image pipeline colours fallback")
                        wait_fn(timeout=None)
                        self._dbg("Diagram object candidates image pipeline colours fallback ready")
            except Exception as exc:
                out["errors"]["image_pipeline_wait"] = f"{type(exc).__name__}: {exc}"

        try:
            import DiagramMaskClusters
        except Exception as exc:
            out["errors"]["diagram_mask_clusters_import"] = f"{type(exc).__name__}: {exc}"
            return out

        for pid in processed_ids:
            row: Dict[str, Any] = {
                "processed_id": pid,
                "sam_dino_candidates": [],
                "image_clusters": [],
                "prepared_candidates": [],
                "dropped_no_image_cluster_match": [],
            }
            cluster_payload = self._load_image_cluster_stroke_groups(pid)
            row["image_cluster_map_path"] = cluster_payload.get("cluster_map_path")
            row["image_clusters"] = list(cluster_payload.get("clusters") or [])
            if cluster_payload.get("error"):
                out["errors"][f"{pid}:image_clusters"] = str(cluster_payload.get("error"))
            if not cluster_payload.get("exists"):
                out["errors"][f"{pid}:image_clusters_missing"] = str(cluster_payload.get("cluster_map_path", ""))

            try:
                self._dbg(
                    "DiagramMaskClusters starting",
                    data={"processed_id": pid, "image_clusters": len(row["image_clusters"])},
                )
                t0 = time.perf_counter()
                cfg = None
                build_cfg = getattr(DiagramMaskClusters, "build_config", None)
                if callable(build_cfg):
                    try:
                        cfg = build_cfg(log_progress=True)
                    except Exception:
                        cfg = None
                sam_result = DiagramMaskClusters.ensure_processed_clusters(pid, save_outputs=False, config=cfg)
                self._dbg(
                    "DiagramMaskClusters finished",
                    data={"processed_id": pid, "elapsed_s": round(time.perf_counter() - t0, 3)},
                )
                sam_payload = sam_result.get("stroke_candidates") if isinstance(sam_result, dict) else {}
                row["sam_dino_meta"] = {
                    key: value
                    for key, value in dict(sam_payload or {}).items()
                    if key != "candidates"
                }
                row["sam_dino_candidates"] = list((sam_payload or {}).get("candidates") or [])
            except Exception as exc:
                out["errors"][f"{pid}:sam_dino"] = f"{type(exc).__name__}: {exc}"
                row["sam_dino_error"] = f"{type(exc).__name__}: {exc}"

            for candidate in row["sam_dino_candidates"]:
                scored = self._score_sam_candidate_against_image_clusters(
                    candidate=candidate,
                    image_clusters=row["image_clusters"],
                )
                if scored is None:
                    row["dropped_no_image_cluster_match"].append(
                        {
                            "candidate_id": str((candidate or {}).get("candidate_id", "") or ""),
                            "stroke_indexes": list((candidate or {}).get("stroke_indexes") or []),
                            "render_file": str((candidate or {}).get("render_file", "") or ""),
                        }
                    )
                    continue
                bbox = scored.get("bbox_xyxy") if isinstance(scored.get("bbox_xyxy"), list) else []
                if len(bbox) == 4:
                    try:
                        scored["xy"] = [
                            round((float(bbox[0]) + float(bbox[2])) / 2.0, 2),
                            round((float(bbox[1]) + float(bbox[3])) / 2.0, 2),
                        ]
                    except Exception:
                        scored["xy"] = [None, None]
                row["prepared_candidates"].append(scored)

            row["prepared_candidates"].sort(
                key=lambda item: (
                    float(item.get("image_cluster_confidence", 0.0) or 0.0),
                    int(item.get("strong_image_cluster_match_count", 0) or 0),
                    len(item.get("stroke_indexes") or []),
                ),
                reverse=True,
            )
            out["by_processed_id"][pid] = row
            out["flat_candidates"].extend(row["prepared_candidates"])
            self._dbg(
                "Diagram object candidates prepared",
                data={
                    "processed_id": pid,
                    "sam_dino": len(row["sam_dino_candidates"]),
                    "prepared": len(row["prepared_candidates"]),
                    "dropped": len(row["dropped_no_image_cluster_match"]),
                },
            )

        saved_path = self._write_json_file(
            self._internal_runtime_dir() / "diagram_object_candidates.json",
            out,
        )
        out["saved_path"] = saved_path
        self._dbg(
            "Diagram object candidates saved",
            data={"saved_path": saved_path, "flat_candidates": len(out["flat_candidates"]), "errors": len(out["errors"])},
        )
        return out

    def _component_rows_from_c2_reports(
        self,
        *,
        prompt_meta: Dict[str, Dict[str, Any]],
        c2_reports_by_prompt: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        by_prompt: Dict[str, List[Dict[str, Any]]] = {}
        for prompt, meta in (prompt_meta or {}).items():
            if int((meta or {}).get("diagram", 0) or 0) != 1:
                continue
            rows: List[Dict[str, Any]] = []
            seen: set[str] = set()
            report = c2_reports_by_prompt.get(prompt) if isinstance(c2_reports_by_prompt, dict) else {}
            visual_stage = report.get("visual_stage") if isinstance(report, dict) and isinstance(report.get("visual_stage"), dict) else {}
            components = visual_stage.get("components") if isinstance(visual_stage.get("components"), list) else []
            for row in components:
                if not isinstance(row, dict):
                    continue
                component = row.get("component") if isinstance(row.get("component"), dict) else {}
                name = normalize_ws(row.get("label") or component.get("label"))
                if not name or name.casefold() in seen:
                    continue
                seen.add(name.casefold())
                json_path = str(row.get("json_path", "") or "").strip()
                canonical_path = str(row.get("canonical_candidate_local_path", "") or "").strip()
                canonical_id = str(row.get("canonical_candidate_id", "") or "").strip()
                canonical_score = float(row.get("canonical_candidate_score", 0.0) or 0.0)
                if (not canonical_path or not canonical_id) and json_path and os.path.isfile(json_path):
                    try:
                        profile = json.loads(Path(json_path).read_text(encoding="utf-8")) or {}
                        canonical_path = canonical_path or str(profile.get("canonical_candidate_local_path", "") or "").strip()
                        canonical_id = canonical_id or str(profile.get("canonical_candidate_id", "") or "").strip()
                        canonical_score = canonical_score or float(profile.get("canonical_candidate_score", 0.0) or 0.0)
                    except Exception:
                        pass
                rows.append(
                    {
                        "name": name,
                        "qid": str(row.get("qid", component.get("qid", "")) or "").strip(),
                        "component_key": str(row.get("component_key", "") or "").strip(),
                        "refined_visual_description": normalize_ws(row.get("refined_visual_description")),
                        "wikipedia_visual_description": normalize_ws(row.get("wikipedia_visual_description")),
                        "canonical_candidate_id": canonical_id,
                        "canonical_candidate_local_path": canonical_path,
                        "canonical_candidate_score": canonical_score,
                        "json_path": json_path,
                    }
                )
            for item in list((meta or {}).get("diagram_required_objects") or []):
                name = normalize_ws(item)
                if not name or name.casefold() in seen:
                    continue
                seen.add(name.casefold())
                rows.append(
                    {
                        "name": name,
                        "qid": "",
                        "component_key": "",
                        "refined_visual_description": "",
                        "wikipedia_visual_description": "",
                        "canonical_candidate_id": "",
                        "canonical_candidate_local_path": "",
                        "canonical_candidate_score": 0.0,
                        "json_path": "",
                    }
                )
            by_prompt[prompt] = rows
        return by_prompt

    def run_diagram_visual_description_batch(
        self,
        *,
        diagram_object_candidates: Dict[str, Any],
        prompt_meta: Dict[str, Dict[str, Any]],
        c2_reports_by_prompt: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        component_rows_by_prompt = self._component_rows_from_c2_reports(
            prompt_meta=prompt_meta,
            c2_reports_by_prompt=c2_reports_by_prompt,
        )
        system_prompt, user_prompt = build_qwen_non_semantic_image_description_prompt()
        requests: List[Dict[str, Any]] = []
        metas: List[Dict[str, Any]] = []
        errors: Dict[str, str] = {}

        by_processed = diagram_object_candidates.get("by_processed_id") if isinstance(diagram_object_candidates.get("by_processed_id"), dict) else {}
        for pid, bundle in sorted(by_processed.items()):
            candidates = bundle.get("prepared_candidates") if isinstance(bundle, dict) else []
            for candidate_index, candidate in enumerate(candidates if isinstance(candidates, list) else []):
                if not isinstance(candidate, dict):
                    continue
                render_path = str(candidate.get("render_path", "") or "").strip()
                if not render_path or not os.path.isfile(render_path):
                    errors[f"{pid}:object:{candidate.get('candidate_id', candidate_index)}"] = "missing_render_path"
                    continue
                requests.append({"prompt": user_prompt, "images": [render_path]})
                metas.append(
                    {
                        "kind": "object",
                        "processed_id": str(pid),
                        "candidate_index": candidate_index,
                        "candidate_id": str(candidate.get("candidate_id", "") or ""),
                        "image_path": render_path,
                    }
                )

        for prompt, components in sorted(component_rows_by_prompt.items()):
            for component_index, component in enumerate(components):
                image_path = str(component.get("canonical_candidate_local_path", "") or "").strip()
                if not image_path:
                    continue
                if not os.path.isfile(image_path):
                    errors[f"{prompt}:component:{component.get('name', component_index)}"] = "missing_canonical_image"
                    continue
                requests.append({"prompt": user_prompt, "images": [image_path]})
                metas.append(
                    {
                        "kind": "component",
                        "prompt": prompt,
                        "component_index": component_index,
                        "component_name": str(component.get("name", "") or ""),
                        "image_path": image_path,
                    }
                )

        self._dbg(
            "Diagram visual description requests ready",
            data={
                "requests": len(requests),
                "objects": sum(1 for meta in metas if meta.get("kind") == "object"),
                "components": sum(1 for meta in metas if meta.get("kind") == "component"),
                "errors": len(errors),
            },
        )
        qwen_result = self._generate_qwen_vision_batch_cached(
            stage_name="diagram_non_semantic_visual_descriptions",
            requests=requests,
            system_prompt=system_prompt,
            generation={
                "max_new_tokens": 220,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.02,
            },
        ) if requests else {"responses": [], "qwen_ready": None, "cache_hits": 0, "cache_misses": 0}

        raw_rows: List[Dict[str, Any]] = []
        responses = qwen_result.get("responses") if isinstance(qwen_result.get("responses"), list) else []
        for meta, raw_row in zip(metas, responses):
            raw_text = str((raw_row or {}).get("text", "") or "").strip()
            parsed = parse_qwen_non_semantic_image_description_output(raw_text)
            description = str(parsed.get("description", "") or "").strip()
            if meta.get("kind") == "object":
                bundle = by_processed.get(str(meta.get("processed_id"))) if isinstance(by_processed, dict) else {}
                candidates = bundle.get("prepared_candidates") if isinstance(bundle, dict) else []
                try:
                    candidate = candidates[int(meta.get("candidate_index", -1))]
                    if isinstance(candidate, dict):
                        candidate["qwen_visual_description"] = description
                        candidate["qwen_visual_description_image_path"] = str(meta.get("image_path", "") or "")
                except Exception:
                    pass
            elif meta.get("kind") == "component":
                components = component_rows_by_prompt.get(str(meta.get("prompt")), [])
                try:
                    component = components[int(meta.get("component_index", -1))]
                    if isinstance(component, dict):
                        component["qwen_visual_description"] = description
                except Exception:
                    pass
                json_path = ""
                try:
                    json_path = str(components[int(meta.get("component_index", -1))].get("json_path", "") or "")
                except Exception:
                    json_path = ""
                if json_path and os.path.isfile(json_path):
                    try:
                        payload = json.loads(Path(json_path).read_text(encoding="utf-8")) or {}
                        payload["qwen_canonical_image_visual_description"] = description
                        self._write_json_file(Path(json_path), payload)
                    except Exception as exc:
                        errors[f"{meta.get('prompt')}:{meta.get('component_name')}:profile_update"] = f"{type(exc).__name__}: {exc}"
            raw_rows.append(
                {
                    "meta": meta,
                    "raw_response": raw_text,
                    "parsed": parsed,
                    "usage": raw_row,
                }
            )

        out = {
            "enabled": True,
            "request_count": len(requests),
            "object_candidates_path": diagram_object_candidates.get("saved_path"),
            "component_payloads_by_prompt": component_rows_by_prompt,
            "errors": errors,
            "qwen_ready": qwen_result.get("qwen_ready"),
            "cache_hits": int(qwen_result.get("cache_hits", 0) or 0),
            "cache_misses": int(qwen_result.get("cache_misses", 0) or 0),
            "raw_rows": raw_rows,
        }
        if diagram_object_candidates.get("saved_path"):
            try:
                self._write_json_file(Path(str(diagram_object_candidates.get("saved_path"))), diagram_object_candidates)
            except Exception:
                pass
        out["saved_path"] = self._write_json_file(
            self._internal_runtime_dir() / "diagram_visual_description_payload.json",
            out,
        )
        self._dbg(
            "Diagram visual descriptions saved",
            data={
                "saved_path": out["saved_path"],
                "cache_hits": out["cache_hits"],
                "cache_misses": out["cache_misses"],
                "errors": len(errors),
            },
        )
        return out

    def _diagram_component_match_output_dir(self, *, processed_id: str, diagram_name: str) -> Path:
        pid = self._normalize_processed_id(processed_id) or "processed_unknown"
        slug = self._safe_diagram_component_slug(diagram_name)
        return Path(__file__).resolve().parent / "DiagramComponents" / f"{pid}_{slug}"

    def run_diagram_component_stroke_match_batch(
        self,
        *,
        prompt_meta: Dict[str, Dict[str, Any]],
        selected_ids_by_prompt: Dict[str, List[str]],
        diagram_object_candidates: Dict[str, Any],
        diagram_visual_descriptions: Dict[str, Any],
        stroke_meaning_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompts: List[str] = []
        metas: List[Dict[str, Any]] = []
        system_prompt: Optional[str] = None
        errors: Dict[str, str] = {}
        by_processed = diagram_object_candidates.get("by_processed_id") if isinstance(diagram_object_candidates.get("by_processed_id"), dict) else {}
        component_payloads_by_prompt = diagram_visual_descriptions.get("component_payloads_by_prompt") if isinstance(diagram_visual_descriptions.get("component_payloads_by_prompt"), dict) else {}
        stroke_meaning_by_pid = stroke_meaning_result.get("by_processed_id") if isinstance(stroke_meaning_result.get("by_processed_id"), dict) else {}

        for prompt, meta in (prompt_meta or {}).items():
            if int((meta or {}).get("diagram", 0) or 0) != 1:
                continue
            components = component_payloads_by_prompt.get(prompt)
            if not isinstance(components, list) or not components:
                errors[f"{prompt}:components"] = "missing_component_payload"
                continue
            for raw_pid in list(selected_ids_by_prompt.get(prompt) or [])[:1]:
                pid = self._normalize_processed_id(raw_pid)
                if not pid:
                    continue
                descriptor, descriptor_path = self._load_stroke_description_payload(pid)
                if not isinstance(descriptor, dict):
                    errors[f"{prompt}:{pid}:line_descriptors"] = f"missing_stroke_description:{descriptor_path}"
                    continue
                candidate_bundle = by_processed.get(pid) if isinstance(by_processed, dict) else {}
                candidate_objects = list((candidate_bundle or {}).get("prepared_candidates") or []) if isinstance(candidate_bundle, dict) else []
                if not candidate_objects:
                    errors[f"{prompt}:{pid}:candidate_objects"] = "missing_candidate_objects"
                stroke_meaning_payload = stroke_meaning_by_pid.get(pid) if isinstance(stroke_meaning_by_pid, dict) else {}
                sys_prompt, user_prompt = build_qwen_diagram_component_stroke_match_prompt(
                    diagram_name=prompt,
                    candidate_objects=candidate_objects,
                    stroke_description_payload=descriptor,
                    stroke_meaning_payload=stroke_meaning_payload if isinstance(stroke_meaning_payload, dict) else {},
                    components=components,
                    refined_visual_description_max_chars=DEFAULT_COMPONENT_REFINED_VISUAL_DESC_MAX_CHARS,
                )
                system_prompt = system_prompt or sys_prompt
                prompts.append(user_prompt)
                metas.append(
                    {
                        "prompt": prompt,
                        "processed_id": pid,
                        "component_names": [normalize_ws(row.get("name")) for row in components if isinstance(row, dict) and normalize_ws(row.get("name"))],
                        "stroke_description_path": descriptor_path,
                    }
                )

        if not prompts:
            out = {
                "enabled": True,
                "request_count": 0,
                "by_diagram": {},
                "errors": errors,
                "note": "no_final_match_jobs",
            }
            out["saved_path"] = self._write_json_file(self._internal_runtime_dir() / "diagram_component_stroke_matches.json", out)
            self._dbg("Diagram component stroke match skipped", data={"errors": len(errors), "saved_path": out["saved_path"]})
            return out

        self._dbg(
            "Diagram component stroke match requests ready",
            data={"requests": len(prompts), "errors": len(errors)},
        )
        qwen_result = self._generate_qwen_text_batch_cached(
            stage_name="diagram_component_stroke_match",
            prompts=prompts,
            system_prompt=system_prompt or "",
            generation={
                "max_new_tokens": 5200,
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.03,
            },
        )
        responses = qwen_result.get("responses") if isinstance(qwen_result.get("responses"), list) else []
        by_diagram: Dict[str, Any] = {}
        raw_io_rows: List[Dict[str, Any]] = []
        for index, (meta, raw_row, prompt_text) in enumerate(zip(metas, responses, prompts), start=1):
            raw_text = str((raw_row or {}).get("text", "") or "").strip()
            parsed = parse_qwen_diagram_component_stroke_match_output(
                raw_text,
                component_names=list(meta.get("component_names") or []),
            )
            out_dir = self._diagram_component_match_output_dir(
                processed_id=str(meta.get("processed_id", "") or ""),
                diagram_name=str(meta.get("prompt", "") or ""),
            )
            components_path = self._write_json_file(out_dir / "components.json", parsed.get("components", {}))
            key = f"{meta.get('processed_id')}:{meta.get('prompt')}"
            by_diagram[key] = {
                "prompt": meta.get("prompt"),
                "processed_id": meta.get("processed_id"),
                "component_count": len(parsed.get("components") or {}),
                "components_path": components_path,
                "stroke_description_path": meta.get("stroke_description_path"),
                "components": parsed.get("components", {}),
            }
            raw_io_rows.append(
                {
                    "request_index": index,
                    "prompt": meta.get("prompt"),
                    "processed_id": meta.get("processed_id"),
                    "request_text": prompt_text,
                    "raw_response": raw_text,
                    "parsed": parsed,
                    "usage": raw_row,
                }
            )

        raw_io_path = self._write_json_file(
            self._internal_runtime_dir() / "qwen_raw_io" / "diagram_component_stroke_match.json",
            {
                "system_prompt": system_prompt,
                "rows": raw_io_rows,
            },
        )
        out = {
            "enabled": True,
            "request_count": len(prompts),
            "by_diagram": by_diagram,
            "errors": errors,
            "raw_io_path": raw_io_path,
            "qwen_ready": qwen_result.get("qwen_ready"),
            "cache_hits": int(qwen_result.get("cache_hits", 0) or 0),
            "cache_misses": int(qwen_result.get("cache_misses", 0) or 0),
        }
        out["saved_path"] = self._write_json_file(self._internal_runtime_dir() / "diagram_component_stroke_matches.json", out)
        self._dbg(
            "Diagram component stroke matches saved",
            data={
                "saved_path": out["saved_path"],
                "request_count": len(prompts),
                "cache_hits": out["cache_hits"],
                "cache_misses": out["cache_misses"],
                "errors": len(errors),
            },
        )
        return out

    def _load_objects_list_from_refined_file(self, refined_file: str) -> List[Any]:
        try:
            data = json.loads(Path(refined_file).read_text(encoding="utf-8"))
        except Exception:
            return []
        raw = None
        if isinstance(data, list):
            raw = data
        elif isinstance(data, dict):
            raw = data.get("objects")
            if not isinstance(raw, list):
                raw = data.get("refined_labels")
            if not isinstance(raw, list):
                raw = data.get("labels")
        if not isinstance(raw, list):
            return []
        out: List[Any] = []
        seen = set()
        for item in raw:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned and cleaned.casefold() not in seen:
                    seen.add(cleaned.casefold())
                    out.append(cleaned)
                continue
            if isinstance(item, dict):
                name = normalize_ws(item.get("name"))
                if name and name.casefold() not in seen:
                    seen.add(name.casefold())
                    out.append(name)
        return out

    def _diagram_components_dir(self) -> Path:
        return Path(self.debug_out_dir) / "_diagram_components"

    def _safe_diagram_component_slug(self, prompt: str) -> str:
        text = re.sub(r"[^\w.-]+", "_", str(prompt or "").strip(), flags=re.UNICODE)
        text = re.sub(r"_+", "_", text).strip("._")
        if not text:
            text = hashlib.sha1(str(prompt or "").encode("utf-8")).hexdigest()[:12]
        return text[:96]

    def _write_c2_component_artifact(
        self,
        *,
        prompt: str,
        diagram_mode: int,
        requested_components: List[str],
        report: Dict[str, Any],
    ) -> Optional[str]:
        prompt = str(prompt or "").strip()
        if not prompt:
            return None
        visual_stage = report.get("visual_stage") if isinstance(report, dict) else {}
        components_raw = visual_stage.get("components") if isinstance(visual_stage, dict) else []
        if not isinstance(components_raw, list):
            components_raw = []

        compact_components: List[Dict[str, Any]] = []
        object_names: List[str] = []
        seen = set()
        for row in components_raw:
            if not isinstance(row, dict):
                continue
            label = normalize_ws(row.get("label"))
            if not label and isinstance(row.get("component"), dict):
                label = normalize_ws((row.get("component") or {}).get("label"))
            if not label:
                continue
            key = label.casefold()
            if key in seen:
                continue
            seen.add(key)
            object_names.append(label)
            compact_components.append(
                {
                    "name": label,
                    "qid": str(row.get("qid", "") or "").strip(),
                    "component_key": str(row.get("component_key", "") or "").strip(),
                    "query": str(row.get("query", row.get("search_query", "")) or "").strip(),
                    "wikipedia_visual_description": str(row.get("wikipedia_visual_description", "") or "").strip(),
                    "refined_visual_description": str(row.get("refined_visual_description", "") or "").strip(),
                    "canonical_candidate_id": str(row.get("canonical_candidate_id", "") or "").strip(),
                    "canonical_candidate_local_path": str(row.get("canonical_candidate_local_path", "") or "").strip(),
                    "canonical_candidate_score": float(row.get("canonical_candidate_score", 0.0) or 0.0),
                    "json_path": str(row.get("json_path", "") or "").strip(),
                    "error": str(row.get("error", "") or "").strip(),
                }
            )
        for item in requested_components or []:
            label = normalize_ws(item)
            if label and label.casefold() not in seen:
                seen.add(label.casefold())
                object_names.append(label)
                compact_components.append(
                    {
                        "name": label,
                        "qid": "",
                        "component_key": "",
                        "query": label,
                        "wikipedia_visual_description": "",
                        "refined_visual_description": "",
                        "canonical_candidate_id": "",
                        "canonical_candidate_local_path": "",
                        "canonical_candidate_score": 0.0,
                        "json_path": "",
                        "error": "",
                    }
                )
        if not object_names:
            return None
        payload = {
            "schema": "diagram_components_c2_v2",
            "prompt": prompt,
            "diagram_mode": int(diagram_mode or 0),
            "requested_components": list(requested_components or []),
            "objects": list(object_names),
            "refined_labels": list(object_names),
            "components": compact_components,
            "c2_report": report if isinstance(report, dict) else {},
        }
        out_path = self._diagram_components_dir() / f"{self._safe_diagram_component_slug(prompt)}.json"
        return self._write_json_file(out_path, payload)

    def _apply_c2_diagram_components_to_assets(
        self,
        *,
        assets: Dict[str, ImageAsset],
        prompt_meta: Dict[str, Dict[str, Any]],
        c2_reports_by_prompt: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        applied: Dict[str, Any] = {"artifacts_by_prompt": {}, "errors": {}}
        for prompt, asset in (assets or {}).items():
            if int(asset.diagram or 0) != 1:
                continue
            meta = prompt_meta.get(prompt) or {}
            requested_components = [normalize_ws(x) for x in (meta.get("diagram_required_objects") or []) if normalize_ws(x)]
            report = c2_reports_by_prompt.get(prompt) or {}
            artifact_path = self._write_c2_component_artifact(
                prompt=prompt,
                diagram_mode=int(meta.get("diagram", asset.diagram) or asset.diagram or 0),
                requested_components=requested_components,
                report=report if isinstance(report, dict) else {},
            )
            if not artifact_path:
                applied["errors"][prompt] = "no_c2_components"
                continue
            asset.refined_labels_file = artifact_path
            asset.objects = self._load_objects_list_from_refined_file(artifact_path)
            applied["artifacts_by_prompt"][prompt] = artifact_path
        return applied

    def _diagram_component_labels_from_c2_report(
        self,
        report: Optional[Dict[str, Any]],
        *,
        requested_components: Optional[List[str]] = None,
    ) -> List[str]:
        out: List[str] = []
        seen = set()

        def _push(value: Any) -> None:
            cleaned = normalize_ws(value)
            if cleaned and cleaned.casefold() not in seen:
                seen.add(cleaned.casefold())
                out.append(cleaned)

        if isinstance(report, dict):
            visual_stage = report.get("visual_stage")
            components = visual_stage.get("components") if isinstance(visual_stage, dict) else []
            if isinstance(components, list):
                for row in components:
                    if not isinstance(row, dict):
                        continue
                    _push(row.get("label"))
                    component = row.get("component")
                    if isinstance(component, dict):
                        _push(component.get("label"))
        for item in requested_components or []:
            _push(item)
        return out

    def _norm_diagram_label_text(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _diagram_ocr_match_score(self, *, canonical_labels: List[str], ocr_labels: List[str]) -> Dict[str, Any]:
        canonical = [self._norm_diagram_label_text(x) for x in (canonical_labels or []) if self._norm_diagram_label_text(x)]
        ocr = [self._norm_diagram_label_text(x) for x in (ocr_labels or []) if self._norm_diagram_label_text(x)]
        if not canonical or not ocr:
            return {"score": 0.0, "matched_count": 0, "best_pairs": []}
        pairs: List[Dict[str, Any]] = []
        total_score = 0.0
        matched_count = 0
        for label in canonical:
            best_other = ""
            best_score = 0.0
            label_tokens = set(label.split())
            for candidate in ocr:
                candidate_tokens = set(candidate.split())
                seq = difflib.SequenceMatcher(None, label, candidate).ratio()
                overlap = 0.0
                if label_tokens and candidate_tokens:
                    overlap = len(label_tokens & candidate_tokens) / float(max(1, len(label_tokens | candidate_tokens)))
                contains = 1.0 if (label in candidate or candidate in label) else 0.0
                score = max(seq, overlap, contains)
                if score > best_score:
                    best_score = score
                    best_other = candidate
            total_score += best_score
            if best_score >= 0.72:
                matched_count += 1
            pairs.append({"canonical": label, "ocr": best_other, "score": round(float(best_score), 4)})
        return {
            "score": float(total_score / float(max(1, len(canonical)))),
            "matched_count": int(matched_count),
            "best_pairs": pairs,
        }

    def _cosine_similarity_lists(self, left: Any, right: Any) -> float:
        if not isinstance(left, list) or not isinstance(right, list) or not left or not right:
            return 0.0
        try:
            import numpy as np

            lv = np.asarray(left, dtype=np.float32)
            rv = np.asarray(right, dtype=np.float32)
            if lv.size == 0 or rv.size == 0 or lv.shape != rv.shape:
                return 0.0
            lnorm = float(np.linalg.norm(lv))
            rnorm = float(np.linalg.norm(rv))
            if lnorm <= 0.0 or rnorm <= 0.0:
                return 0.0
            return float(np.dot(lv, rv) / (lnorm * rnorm))
        except Exception:
            return 0.0

    def _siglip_embed_pil_images_with_bundle(self, *, siglip_bundle: Any, pil_images: List[Any]) -> List[List[float]]:
        if siglip_bundle is None or not pil_images:
            return []
        try:
            import torch

            processor = getattr(siglip_bundle, "processor", None)
            model = getattr(siglip_bundle, "model", None)
            device = getattr(siglip_bundle, "device", None)
            if processor is None or model is None:
                return []
            torch_device = device if isinstance(device, torch.device) else torch.device(str(device) if device else ("cuda" if torch.cuda.is_available() else "cpu"))
            with torch.inference_mode():
                inputs = processor(images=pil_images, return_tensors="pt", padding=True)
                try:
                    inputs = inputs.to(torch_device)
                except Exception:
                    inputs = {k: v.to(torch_device) for k, v in inputs.items()}
                feats = model.get_image_features(**inputs)
                feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
                return feats.detach().cpu().tolist()
        except Exception:
            return []

    def _build_diagram_siglip_rule_text(self, base_context: str) -> str:
        cleaned = normalize_ws(base_context) or "unknown"
        return (
            f"Desired diagram image content: {cleaned}. "
            "Prefer a simple single-object visual with one full object centered in frame. "
            "Prefer clearly separated compact internal detail and identifiable parts of the main object. "
            "Prefer visuals that look like one diagram of the requested object instead of many spread-out unrelated things. "
            "Avoid flowcharts, boxes linked with arrows, dense text, and visuals dominated by labels instead of the object."
        )

    def _rank_c2_component_candidate_images(
        self,
        *,
        c2_reports_by_prompt: Dict[str, Dict[str, Any]],
        siglip_bundle: Any,
    ) -> Dict[str, Any]:
        import PineconeFetch
        from PIL import Image

        PineconeFetch.configure_hot_models(siglip_bundle=siglip_bundle, clear_siglip=False)
        by_prompt: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        for prompt, report in (c2_reports_by_prompt or {}).items():
            visual_stage = report.get("visual_stage") if isinstance(report, dict) else {}
            components = visual_stage.get("components") if isinstance(visual_stage, dict) else []
            if not isinstance(components, list):
                continue

            prompt_rows: List[Dict[str, Any]] = []
            for row in components:
                if not isinstance(row, dict):
                    continue
                component = row.get("component") if isinstance(row.get("component"), dict) else {}
                label = str(row.get("label", "") or component.get("label", "") or "").strip()
                refined_desc = str(row.get("refined_visual_description", "") or row.get("wikipedia_visual_description", "") or component.get("description", "") or label).strip()
                candidates = row.get("image_candidates") if isinstance(row.get("image_candidates"), list) else []
                component_dir = str(row.get("component_dir", "") or "").strip()
                json_path = str(row.get("json_path", "") or "").strip()

                ranked_candidates: List[Dict[str, Any]] = []
                best_candidate: Optional[Dict[str, Any]] = None
                if refined_desc and candidates:
                    try:
                        text_embedding = PineconeFetch.embed_siglip_text(refined_desc)
                    except Exception:
                        text_embedding = []
                    pil_images: List[Any] = []
                    pil_rows: List[Dict[str, Any]] = []
                    for candidate in candidates:
                        if not isinstance(candidate, dict):
                            continue
                        local_path = str(candidate.get("local_path", "") or "").strip()
                        if not local_path or not os.path.isfile(local_path):
                            continue
                        try:
                            with Image.open(local_path) as image:
                                pil_images.append(image.convert("RGB"))
                            pil_rows.append(candidate)
                        except Exception:
                            continue
                    image_embeddings = self._siglip_embed_pil_images_with_bundle(siglip_bundle=siglip_bundle, pil_images=pil_images)
                    for candidate, emb in zip(pil_rows, image_embeddings):
                        scored = dict(candidate)
                        scored["siglip_refined_description_score"] = round(float(self._cosine_similarity_lists(text_embedding, emb)), 6)
                        ranked_candidates.append(scored)
                    ranked_candidates.sort(key=lambda item: float(item.get("siglip_refined_description_score", 0.0) or 0.0), reverse=True)
                    best_candidate = ranked_candidates[0] if ranked_candidates else None

                canonical = {
                    "mode": "siglip_refined_visual_description",
                    "candidate_id": str((best_candidate or {}).get("id", "") or "").strip(),
                    "local_path": str((best_candidate or {}).get("local_path", "") or "").strip(),
                    "score": float((best_candidate or {}).get("siglip_refined_description_score", 0.0) or 0.0),
                    "title": str((best_candidate or {}).get("title", "") or "").strip(),
                    "source_kind": str((best_candidate or {}).get("source_kind", "") or "").strip(),
                    "page_url": str((best_candidate or {}).get("page_url", "") or "").strip(),
                    "image_url": str((best_candidate or {}).get("image_url", "") or "").strip(),
                }

                row["canonical_candidate"] = canonical
                row["canonical_candidate_id"] = canonical["candidate_id"]
                row["canonical_candidate_local_path"] = canonical["local_path"]
                row["canonical_candidate_score"] = canonical["score"]

                if json_path and os.path.isfile(json_path):
                    try:
                        payload = json.loads(Path(json_path).read_text(encoding="utf-8")) or {}
                        payload["canonical_candidate"] = canonical
                        payload["canonical_candidate_id"] = canonical["candidate_id"]
                        payload["canonical_candidate_local_path"] = canonical["local_path"]
                        payload["canonical_candidate_score"] = canonical["score"]
                        self._write_json_file(Path(json_path), payload)
                    except Exception as exc:
                        errors[f"{prompt}:{label or component_dir or json_path}"] = f"{type(exc).__name__}: {exc}"

                prompt_rows.append(
                    {
                        "label": label,
                        "component_key": str(row.get("component_key", "") or "").strip(),
                        "component_dir": component_dir,
                        "json_path": json_path,
                        "candidate_count": len(candidates),
                        "canonical_candidate": canonical,
                        "ranked_candidates": [
                            {
                                "id": str(c.get("id", "") or "").strip(),
                                "local_path": str(c.get("local_path", "") or "").strip(),
                                "score": float(c.get("siglip_refined_description_score", 0.0) or 0.0),
                            }
                            for c in ranked_candidates
                        ],
                    }
                )
            by_prompt[prompt] = prompt_rows
        return {"by_prompt": by_prompt, "errors": errors}

    def _rank_diagram_candidate_paths_with_siglip(
        self,
        *,
        diagram_prompts: List[str],
        rerank_paths_by_prompt: Dict[str, List[str]],
        siglip_bundle: Any,
    ) -> Dict[str, Any]:
        cached, cached_path = self._load_latest_stage_data("diagram_siglip_ranking")
        cached_by_prompt = cached.get("by_prompt") if isinstance(cached, dict) else None
        if isinstance(cached_by_prompt, dict):
            requested = {normalize_ws(prompt) for prompt in diagram_prompts if normalize_ws(prompt)}
            available = {str(prompt) for prompt in cached_by_prompt.keys()}
            if requested and requested.issubset(available):
                self._dbg("Diagram SigLIP ranking existing cache hit", data={"path": cached_path, "prompts": len(requested)})
                return cached
        import ImagePipeline
        import PineconeFetch

        meta_ctx_map = ImagePipeline.load_image_metadata_context_map()
        PineconeFetch.configure_hot_models(siglip_bundle=siglip_bundle, clear_siglip=False)

        by_prompt: Dict[str, List[Dict[str, Any]]] = {}
        clip_embeddings_by_processed_id: Dict[str, List[float]] = {}
        for prompt in diagram_prompts:
            candidate_paths = [str(x) for x in (rerank_paths_by_prompt.get(prompt) or []) if str(x).strip()]
            if not candidate_paths:
                by_prompt[prompt] = []
                continue
            rule_text = self._build_diagram_siglip_rule_text(prompt)
            try:
                rule_embedding = PineconeFetch.embed_siglip_text(rule_text)
            except Exception:
                rule_embedding = None
            ranked_rows: List[Dict[str, Any]] = []
            for path in candidate_paths:
                meta_entry = ImagePipeline._resolve_meta_for_source(meta_ctx_map, path)
                clip_embedding = meta_entry.get("clip_embedding") if isinstance(meta_entry, dict) else None
                score = self._cosine_similarity_lists(rule_embedding, clip_embedding)
                ranked_rows.append(
                    {
                        "source_path": path,
                        "clip_embedding": clip_embedding if isinstance(clip_embedding, list) else [],
                        "siglip_rule_score": float(score),
                        "final_score": float((meta_entry or {}).get("final_score", 0.0) or 0.0) if isinstance(meta_entry, dict) else 0.0,
                    }
                )
            ranked_rows.sort(
                key=lambda row: (
                    float(row.get("siglip_rule_score", 0.0) or 0.0),
                    float(row.get("final_score", 0.0) or 0.0),
                ),
                reverse=True,
            )
            by_prompt[prompt] = ranked_rows
            for row in ranked_rows:
                source_path = str(row.get("source_path", "") or "").strip()
                emb = row.get("clip_embedding")
                if source_path and isinstance(emb, list) and emb:
                    clip_embeddings_by_processed_id[source_path] = list(emb)
        return {"by_prompt": by_prompt, "clip_embeddings_by_source_path": clip_embeddings_by_processed_id}

    def _select_diagram_processed_ids_from_ranked_candidates(
        self,
        *,
        prompt_meta: Dict[str, Dict[str, Any]],
        ranked_candidates_by_prompt: Dict[str, List[Dict[str, Any]]],
        c2_reports_by_prompt: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        import ImagePipeline
        import ImageText

        selected_ids_by_prompt: Dict[str, List[str]] = {}
        candidate_processed_ids_by_prompt: Dict[str, List[str]] = {}
        selected_source_path_by_prompt: Dict[str, str] = {}
        ocr_debug: Dict[str, Any] = {}

        for prompt, ranked_rows in (ranked_candidates_by_prompt or {}).items():
            rows = [row for row in (ranked_rows or []) if isinstance(row, dict)]
            if not rows:
                selected_ids_by_prompt[prompt] = []
                candidate_processed_ids_by_prompt[prompt] = []
                continue
            image_items = ImagePipeline.load_unique_images_once([str(row.get("source_path", "") or "") for row in rows])
            for item in image_items:
                item["base_context"] = prompt
            processed_items: List[Dict[str, Any]] = []
            path_map: Dict[str, str] = {}
            if image_items:
                try:
                    processed_items, path_map = ImageText.process_images_in_memory(
                        image_items=image_items,
                        start_index=0,
                        save_outputs=True,
                        return_path_map=True,
                    )
                except Exception:
                    processed_items, path_map = [], {}

            candidate_processed_ids_by_prompt[prompt] = [
                str(path_map.get(str(row.get("source_path", "") or ""), "") or "").strip()
                for row in rows
                if str(path_map.get(str(row.get("source_path", "") or ""), "") or "").strip()
            ]
            canonical_labels = self._diagram_component_labels_from_c2_report(
                c2_reports_by_prompt.get(prompt),
                requested_components=list((prompt_meta.get(prompt) or {}).get("diagram_required_objects") or []),
            )

            best_ocr: Optional[Dict[str, Any]] = None
            for item in processed_items:
                payload = item.get("payload_json") if isinstance(item.get("payload_json"), dict) else {}
                words = payload.get("words") if isinstance(payload, dict) else []
                ocr_labels: List[str] = []
                seen_words = set()
                if isinstance(words, list):
                    for row in words:
                        if not isinstance(row, dict):
                            continue
                        text = str(row.get("text", "") or "").strip()
                        normalized = self._norm_diagram_label_text(text)
                        if not normalized or normalized in seen_words:
                            continue
                        seen_words.add(normalized)
                        ocr_labels.append(text)
                score_obj = self._diagram_ocr_match_score(canonical_labels=canonical_labels, ocr_labels=ocr_labels)
                row = {
                    "processed_id": str(item.get("processed_id", "") or "").strip(),
                    "source_path": str(item.get("source_path", "") or "").strip(),
                    "ocr_labels": ocr_labels,
                    "score": float(score_obj.get("score", 0.0) or 0.0),
                    "matched_count": int(score_obj.get("matched_count", 0) or 0),
                    "best_pairs": score_obj.get("best_pairs", []),
                }
                if best_ocr is None or (
                    float(row["score"]) > float(best_ocr.get("score", 0.0))
                    or (
                        float(row["score"]) == float(best_ocr.get("score", 0.0))
                        and int(row["matched_count"]) > int(best_ocr.get("matched_count", 0))
                    )
                ):
                    best_ocr = row

            chosen_processed_id = ""
            chosen_source_path = ""
            decision = "siglip_fallback"
            if best_ocr is not None and (
                float(best_ocr.get("score", 0.0)) >= 0.68
                or (
                    float(best_ocr.get("score", 0.0)) >= 0.55
                    and int(best_ocr.get("matched_count", 0)) >= 2
                )
            ):
                chosen_processed_id = str(best_ocr.get("processed_id", "") or "").strip()
                chosen_source_path = str(best_ocr.get("source_path", "") or "").strip()
                decision = "ocr_canonical_match"
            if not chosen_processed_id and rows:
                top_path = str(rows[0].get("source_path", "") or "").strip()
                chosen_processed_id = str(path_map.get(top_path, "") or "").strip()
                chosen_source_path = top_path

            selected_ids_by_prompt[prompt] = [chosen_processed_id] if chosen_processed_id else []
            selected_source_path_by_prompt[prompt] = chosen_source_path
            ocr_debug[prompt] = {
                "canonical_labels": canonical_labels,
                "best_ocr_match": best_ocr,
                "decision": decision,
                "selected_processed_id": chosen_processed_id,
                "selected_source_path": chosen_source_path,
            }
        return {
            "selected_ids_by_prompt": selected_ids_by_prompt,
            "candidate_processed_ids_by_prompt": candidate_processed_ids_by_prompt,
            "selected_source_path_by_prompt": selected_source_path_by_prompt,
            "ocr_debug": ocr_debug,
        }

    def _await_diagram_research(self) -> Dict[str, Any]:
        state = dict(self._diagram_research_state or {})
        thread = self._diagram_research_thread
        if thread is not None:
            self._dbg("Waiting for diagram research thread", data={"prompts": state.get("prompts", [])})
            thread.join()
            state["finished"] = True
            self._dbg("Diagram research thread finished", data={"error": state.get("error")})
            self._diagram_research_thread = None
            self._diagram_research_state = dict(state)
        return state

    def _unload_diagram_mask_cluster_models(self) -> Optional[str]:
        try:
            import DiagramMaskClusters

            unload_fn = getattr(DiagramMaskClusters, "unload_hot_models", None)
            if callable(unload_fn):
                unload_fn()
            return None
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    def _run_c2_component_verifier_batch(
        self,
        *,
        c2_bundle_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        usable_rows = []
        prompts: List[str] = []
        system_prompt: Optional[str] = None
        for row in c2_bundle_rows or []:
            prompt = normalize_ws(row.get("prompt"))
            required_objects = [normalize_ws(item) for item in (row.get("requested_components") or []) if normalize_ws(item)]
            if not prompt or not required_objects:
                continue
            prompt_system, prompt_text = build_qwen_c2_component_verifier_prompt(
                prompt=prompt,
                required_objects=required_objects,
            )
            usable_rows.append(
                {
                    "prompt": prompt,
                    "required_objects": required_objects,
                }
            )
            system_prompt = system_prompt or prompt_system
            prompts.append(prompt_text)

        if not prompts:
            return {
                "enabled": True,
                "by_prompt": {},
                "qwen_ready": None,
                "cache_hits": 0,
                "cache_misses": 0,
            }

        generation = {
            "max_new_tokens": int(os.getenv("QWEN_C2_COMPONENT_VERIFIER_MAX_NEW_TOKENS", "220") or 220),
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.01,
        }
        result = self._generate_qwen_text_batch_cached(
            stage_name="c2_component_verifier",
            prompts=prompts,
            system_prompt=str(system_prompt or ""),
            generation=generation,
        )
        responses = result.get("responses")
        if not isinstance(responses, list) or len(responses) != len(prompts):
            raise RuntimeError("Qwen c2_component_verifier response count does not match request count")

        by_prompt: Dict[str, Dict[str, Any]] = {}
        rows_out: List[Dict[str, Any]] = []
        for row, raw_row in zip(usable_rows, responses):
            raw_text = str((raw_row or {}).get("text", "") or "").strip()
            parsed = parse_qwen_c2_component_verifier_output(raw_text)
            verdict = {
                "prompt": row["prompt"],
                "required_objects": list(row["required_objects"]),
                "skip_stage1": int(parsed.get("skip_stage1", 0) or 0),
                "missing": list(parsed.get("missing") or []),
                "raw_response": raw_text,
            }
            by_prompt[row["prompt"]] = verdict
            rows_out.append(verdict)

        saved_path = self._write_json_file(
            self._internal_runtime_dir() / "c2_component_verifier.json",
            {
                "schema": "c2_component_verifier_v1",
                "cache_hits": int(result.get("cache_hits", 0) or 0),
                "cache_misses": int(result.get("cache_misses", 0) or 0),
                "rows": rows_out,
            },
        )
        return {
            "enabled": True,
            "by_prompt": by_prompt,
            "saved_path": saved_path,
            "qwen_ready": result.get("qwen_ready"),
            "cache_hits": int(result.get("cache_hits", 0) or 0),
            "cache_misses": int(result.get("cache_misses", 0) or 0),
        }

    def _run_c2_bundle(
        self,
        *,
        c2_bundle_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not c2_bundle_rows:
            return {"ok": True, "prompts": {}, "errors": {}, "count": 0}
        cached = self._load_c2_bundle_cache_for_rows(c2_bundle_rows)
        if isinstance(cached, dict):
            return cached
        self._ensure_qwen_text_worker()
        from C2.Orchestrator import LocalWorkerClient, run_orchestrator_bundle
        from C2.QwenWorker import ServerQwenWorker

        stage_io_dir = str(Path(self.debug_out_dir) / "qwen_stage_io")
        worker = ServerQwenWorker(
            model_name=self.qwen_model,
            server_base_url=self.qwen_base_url,
            stage_io_dir=stage_io_dir,
        )
        return run_orchestrator_bundle(
            prompts=c2_bundle_rows,
            worker_client=LocalWorkerClient(worker),
            mode="normal",
            steps=max(1, int(os.environ.get("C2_AGENT_STEPS", "4") or 4)),
            limit=max(4, int(os.environ.get("C2_AGENT_LIMIT", "8") or 8)),
            timeout=max(18, int(os.environ.get("C2_HTTP_TIMEOUT", "18") or 18)),
            output_root=str(self._diagram_components_dir()),
            visual_component_batch_size=max(2, int(os.environ.get("C2_VISUAL_BATCH_SIZE", "4") or 4)),
            skip_visual_stage=False,
            max_workers=max(1, min(len(c2_bundle_rows), int(os.environ.get("C2_DIAGRAM_BUNDLE_WORKERS", "6") or 6))),
        )

    def _load_c2_bundle_cache_for_rows(self, c2_bundle_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not c2_bundle_rows:
            return {"ok": True, "prompts": {}, "errors": {}, "count": 0}
        cached, cached_path = self._load_latest_stage_data("c2_bundle")
        requested_prompts = {normalize_ws(row.get("prompt")) for row in c2_bundle_rows if normalize_ws(row.get("prompt"))}
        cached_prompts = set((cached.get("prompts") or {}).keys()) if isinstance(cached, dict) and isinstance(cached.get("prompts"), dict) else set()
        if requested_prompts and requested_prompts.issubset(cached_prompts):
            self._dbg("C2 bundle existing cache hit", data={"path": cached_path, "prompts": len(requested_prompts)})
            return cached
        return None

    def _unload_qwen_server(self) -> Optional[str]:
        # The Qwen worker is intentionally kept resident. Heavy stages may ask
        # for this legacy helper, but the correct behavior is sleep, not unload.
        return self._sleep_qwen_text_worker()

    def _unload_siglip_text(self) -> Optional[str]:
        try:
            from shared_models import unload_siglip_text

            unload_siglip_text()
            return None
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    def _load_full_siglip(self) -> Tuple[Any, Optional[str]]:
        try:
            from shared_models import get_siglip, init_siglip_hot

            if get_siglip() is None:
                init_siglip_hot(
                    gpu_index=self.gpu_index,
                    cpu_threads=self.cpu_threads,
                    warmup=True,
                )
            return get_siglip(), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _unload_full_siglip(self) -> Optional[str]:
        try:
            from shared_models import unload_siglip

            unload_siglip()
            return None
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    def _unload_comfy_models(self) -> Optional[str]:
        try:
            import ComfyFluxClient

            ComfyFluxClient.free_all_models(
                comfy_url=os.getenv("COMFY_URL") or None,
                unload_models=True,
                free_memory=True,
            )
            return None
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    def _finalize_research_rerank(self) -> Dict[str, List[str]]:
        cached, cached_path = self._load_latest_stage_data("diagram_research_rerank_finalize")
        if isinstance(cached, dict) and cached:
            self._dbg("Diagram research rerank existing cache hit", data={"path": cached_path, "prompts": len(cached)})
            return {
                str(key): [str(x) for x in (value or []) if str(x).strip()]
                for key, value in cached.items()
                if isinstance(key, str)
            }
        ImageResearcher = self._import_image_researcher()

        finalize_fn = getattr(ImageResearcher, "finalize_research_from_saved_ranker_state", None)
        if not callable(finalize_fn):
            finalize_fn = getattr(ImageResearcher, "collect_unique_images_all_prompts", None)
        if not callable(finalize_fn):
            return {}
        backend_cls = getattr(ImageResearcher, "_SharedSiglipBackend", None)
        if not callable(backend_cls):
            return {}
        backend = backend_cls()
        finalize_kwargs: Dict[str, Any] = {"backend": backend}
        img_root = getattr(ImageResearcher, "IMAGES_PATH", None)
        if isinstance(img_root, str) and img_root.strip():
            finalize_kwargs["images_root"] = img_root
        out_map = finalize_fn(**finalize_kwargs)
        if not isinstance(out_map, dict):
            return {}
        return {
            str(key): [str(x) for x in (value or []) if str(x).strip()]
            for key, value in out_map.items()
            if isinstance(key, str)
        }

    def _generate_non_diagram_images(self, *, prompt_meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        import ComfyFluxClient
        import ImagePipeline
        import ImageText
        from shared_models import get_minilm

        prompt_to_topic = {
            prompt: str((meta or {}).get("topic") or "").strip()
            for prompt, meta in (prompt_meta or {}).items()
            if int((meta or {}).get("diagram", 0) or 0) == 0
        }
        if not prompt_to_topic:
            return {
                "generated": {},
                "processed_ids_by_prompt": {},
                "saved_paths_by_prompt": {},
                "metadata_enrichment": None,
                "error": None,
            }

        comfy_root = (
            str(os.getenv("COMFY_CAPTURE_OUTPUT_ROOT") or "").strip()
            or os.path.join("ResearchImages", "UniqueImages", "ComfyGenerated")
        )
        generated = ComfyFluxClient.generate_many(
            prompt_to_topic,
            workflow_path=os.getenv("COMFY_WORKFLOW_JSON") or None,
            comfy_url=os.getenv("COMFY_URL") or None,
            output_root=comfy_root,
            batch_size=1,
            clean_prompt_dirs=True,
        )
        metadata_enrichment = None
        try:
            metadata_enrichment = ComfyFluxClient.enrich_generation_metadata(
                generated,
                siglip_bundle=None,
                minilm_bundle=get_minilm(),
                meta_path=Path(comfy_root) / "image_metadata_context.json",
            )
        except Exception:
            metadata_enrichment = None

        processed_ids_by_prompt: Dict[str, List[str]] = {}
        saved_paths_by_prompt: Dict[str, List[str]] = {}
        for prompt, info in (generated or {}).items():
            if not isinstance(info, dict):
                continue
            saved_paths = [str(x) for x in (info.get("saved_paths") or []) if str(x).strip()]
            saved_paths_by_prompt[prompt] = saved_paths
            if not saved_paths:
                processed_ids_by_prompt[prompt] = []
                continue
            image_items = ImagePipeline.load_unique_images_once(saved_paths)
            for item in image_items:
                item["base_context"] = prompt
            try:
                _, path_map = ImageText.process_images_in_memory(
                    image_items=image_items,
                    start_index=0,
                    save_outputs=True,
                    return_path_map=True,
                )
            except Exception:
                path_map = {}
            processed_ids_by_prompt[prompt] = [
                str(path_map.get(path, "") or "").strip()
                for path in saved_paths
                if str(path_map.get(path, "") or "").strip()
            ]
        return {
            "generated": generated,
            "processed_ids_by_prompt": processed_ids_by_prompt,
            "saved_paths_by_prompt": saved_paths_by_prompt,
            "metadata_enrichment": metadata_enrichment,
            "error": None,
        }

    def _write_c2_link_maps(
        self,
        *,
        prompt_meta: Dict[str, Dict[str, Any]],
        c2_reports_by_prompt: Dict[str, Dict[str, Any]],
        selected_ids_by_prompt: Dict[str, List[str]],
        candidate_processed_ids_by_prompt: Dict[str, List[str]],
        selected_source_path_by_prompt: Dict[str, str],
    ) -> Dict[str, str]:
        written: Dict[str, str] = {}
        for prompt, report in (c2_reports_by_prompt or {}).items():
            if not isinstance(report, dict):
                continue
            visual_stage = report.get("visual_stage") if isinstance(report.get("visual_stage"), dict) else {}
            prompt_dir_text = str(visual_stage.get("prompt_dir", "") or "").strip()
            if not prompt_dir_text:
                continue
            prompt_dir = Path(prompt_dir_text)
            components = visual_stage.get("components") if isinstance(visual_stage.get("components"), list) else []
            prompt_dir.mkdir(parents=True, exist_ok=True)
            selected_processed_id = str(((selected_ids_by_prompt.get(prompt) or [""])[0]) or "").strip()
            selected_source_path = str(selected_source_path_by_prompt.get(prompt, "") or "").strip()
            link_payload = {
                "schema": "c2_prompt_component_links_v1",
                "prompt": prompt,
                "diagram_mode": int((prompt_meta.get(prompt) or {}).get("diagram", 0) or 0),
                "requested_components": list((prompt_meta.get(prompt) or {}).get("diagram_required_objects") or []),
                "selected_processed_id": selected_processed_id,
                "selected_source_path": selected_source_path,
                "candidate_processed_ids": list(candidate_processed_ids_by_prompt.get(prompt) or []),
                "components": [],
            }
            for row in components:
                if not isinstance(row, dict):
                    continue
                component_json = Path(str(row.get("json_path", "") or "").strip())
                component_entry = {
                    "component_key": str(row.get("component_key", "") or "").strip(),
                    "qid": str(row.get("qid", "") or "").strip(),
                    "label": normalize_ws(row.get("label")),
                    "query": str(row.get("query", row.get("search_query", "")) or "").strip(),
                    "json_path": str(component_json) if component_json else "",
                    "canonical_candidate_id": str(row.get("canonical_candidate_id", "") or "").strip(),
                    "canonical_candidate_local_path": str(row.get("canonical_candidate_local_path", "") or "").strip(),
                    "selected_processed_id": selected_processed_id,
                    "selected_source_path": selected_source_path,
                }
                link_payload["components"].append(component_entry)
                if component_json.is_file():
                    try:
                        payload = json.loads(component_json.read_text(encoding="utf-8")) or {}
                        payload["component_key"] = component_entry["component_key"]
                        payload["diagram_prompt"] = prompt
                        payload["diagram_selected_processed_id"] = selected_processed_id
                        payload["diagram_selected_source_path"] = selected_source_path
                        payload["diagram_candidate_processed_ids"] = list(candidate_processed_ids_by_prompt.get(prompt) or [])
                        component_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass
            out_path = prompt_dir / "diagram_component_links.json"
            out_path.write_text(json.dumps(link_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            written[prompt] = str(out_path)
        return written

    def _sync_request_dimensions_into_chapters(
        self,
        *,
        chapters_out: List[Dict[str, Any]],
        prepared_objects: List[Dict[str, Any]],
    ) -> None:
        image_by_name: Dict[str, Dict[str, Any]] = {}
        text_by_name: Dict[str, Dict[str, Any]] = {}
        for row in prepared_objects:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "") or "").strip()
            if not name:
                continue
            if str(row.get("object_type", "") or "") == "image":
                image_by_name[name] = row
            elif str(row.get("object_type", "") or "") == "text":
                text_by_name[name] = row
        for chapter in chapters_out:
            for row in chapter.get("image_requests") or []:
                name = normalize_ws(row.get("name"))
                prepared = image_by_name.get(name)
                if prepared is None:
                    continue
                row["processed_id"] = prepared.get("processed_id")
                row["processed_png_path"] = prepared.get("processed_png_path")
                row["dimensions_px"] = prepared.get("dimensions_px")
                row["dimension_source"] = prepared.get("dimension_source")
                row["comfy_generated_paths"] = prepared.get("comfy_generated_paths", [])
                row["c2_artifact_path"] = prepared.get("c2_artifact_path")
                row["c2_link_map_path"] = prepared.get("c2_link_map_path")
                row["canonical_objects"] = list(prepared.get("canonical_objects") or [])
            for row in chapter.get("text_requests") or []:
                name = str(row.get("name", "") or "").strip()
                prepared = text_by_name.get(name)
                if prepared is None:
                    continue
                row["dimensions_px"] = prepared.get("dimensions_px")
                row["dimension_source"] = prepared.get("dimension_source")
                row["text_layout"] = prepared.get("text_layout")

    def _prepare_space_planner_objects(
        self,
        *,
        first_module_result: Dict[str, Any],
        post_research_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        chapters_out = list(first_module_result.get("chapters") or [])
        synced_objects = list(first_module_result.get("synced_objects") or [])
        prompt_meta = dict(post_research_result.get("prompt_meta") or {})
        assets_by_prompt = dict(post_research_result.get("assets_by_prompt") or {})
        debug = dict(post_research_result.get("debug") or {})

        c2_artifacts = dict(debug.get("c2_artifacts_by_prompt") or {})
        c2_link_maps = dict(debug.get("c2_link_maps") or {})
        comfy_generated = dict(debug.get("comfy_generated") or {})
        comfy_processed_ids_by_prompt = dict(debug.get("comfy_processed_ids_by_prompt") or {})

        prepared_objects: List[Dict[str, Any]] = []
        verification: Dict[str, Any] = {
            "diagram_processed_links_ok": [],
            "diagram_processed_links_missing": [],
            "non_diagram_processed_links_ok": [],
            "non_diagram_processed_links_missing": [],
            "c2_artifacts_ok": [],
            "c2_artifacts_missing": [],
        }

        for obj in synced_objects:
            if not isinstance(obj, dict):
                continue
            row = dict(obj)
            object_type = str(row.get("object_type", "") or "").strip()
            name = str(row.get("name", "") or "").strip()
            if not name:
                continue
            if object_type == "image":
                asset = assets_by_prompt.get(name)
                processed_ids = list((asset.processed_ids if isinstance(asset, ImageAsset) else row.get("processed_ids")) or [])
                processed_id = str(processed_ids[0] if processed_ids else "" or "").strip()
                dimension_row = self._image_dimensions_from_processed_id(processed_id)
                is_diagram = int(row.get("diagram", 0) or 0) == 1
                c2_artifact_path = str(c2_artifacts.get(name, "") or "").strip() if is_diagram else ""
                c2_link_map_path = str(c2_link_maps.get(name, "") or "").strip() if is_diagram else ""
                canonical_objects = list((prompt_meta.get(name) or {}).get("diagram_required_objects") or [])
                if isinstance(asset, ImageAsset) and isinstance(asset.objects, list) and asset.objects:
                    canonical_objects = list(asset.objects)
                comfy_paths = list(comfy_generated.get(name) or [])

                row.update(
                    {
                        "processed_id": processed_id,
                        "processed_ids": processed_ids,
                        "processed_png_path": str(dimension_row.get("processed_png_path", "") or ""),
                        "dimensions_px": {
                            "width": int(dimension_row.get("width", 400) or 400),
                            "height": int(dimension_row.get("height", 300) or 300),
                        },
                        "dimension_source": "processed_png",
                        "canonical_objects": canonical_objects,
                        "c2_artifact_path": c2_artifact_path,
                        "c2_link_map_path": c2_link_map_path,
                        "comfy_generated_paths": comfy_paths,
                        "image_prompt": name,
                    }
                )

                if processed_id and bool(dimension_row.get("exists")):
                    if is_diagram:
                        verification["diagram_processed_links_ok"].append(name)
                    else:
                        verification["non_diagram_processed_links_ok"].append(name)
                else:
                    if is_diagram:
                        verification["diagram_processed_links_missing"].append(name)
                    else:
                        verification["non_diagram_processed_links_missing"].append(name)

                if is_diagram:
                    if c2_artifact_path and Path(c2_artifact_path).is_file():
                        verification["c2_artifacts_ok"].append(name)
                    else:
                        verification["c2_artifacts_missing"].append(name)
            elif object_type == "text":
                text_layout = self._text_dimensions_px(name)
                row.update(
                    {
                        "processed_id": "",
                        "processed_png_path": "",
                        "dimensions_px": {
                            "width": int(text_layout.get("width", 0) or 0),
                            "height": int(text_layout.get("height", 0) or 0),
                        },
                        "dimension_source": "font_metrics_and_glyph_bounds",
                        "text_layout": text_layout,
                    }
                )
            prepared_objects.append(row)

        self._sync_request_dimensions_into_chapters(
            chapters_out=chapters_out,
            prepared_objects=prepared_objects,
        )

        out_path = Path(self.debug_out_dir) / "space_planner_objects_ready.json"
        saved_path = self._write_json_file(
            out_path,
            {
                "schema": "space_planner_objects_ready_v1",
                "objects": prepared_objects,
                "verification": verification,
                "dimensions_config": self._load_dimensions_config(),
            },
        )
        return {
            "objects": prepared_objects,
            "verification": verification,
            "saved_path": saved_path,
        }

    def _space_planner_node_config(self) -> Dict[str, Any]:
        dims = self._load_dimensions_config()
        board_px = dims.get("whiteboard_size_px") if isinstance(dims.get("whiteboard_size_px"), dict) else {}
        board_width_px = float((board_px or {}).get("width", 2000.0) or 2000.0)
        board_height_px = float((board_px or {}).get("height", 2000.0) or 2000.0)
        node_edge_px = max(10, int(os.getenv("SPACE_PLANNER_NODE_EDGE_PX", "100") or 100))
        return {
            "edge_px": int(node_edge_px),
            "board_width_px": float(board_width_px),
            "board_height_px": float(board_height_px),
            "width": max(1, int(math.ceil(board_width_px / float(node_edge_px)))),
            "height": max(1, int(math.ceil(board_height_px / float(node_edge_px)))),
        }

    def _object_dimensions_to_nodes(
        self,
        *,
        width_px: Any,
        height_px: Any,
        node_config: Dict[str, Any],
    ) -> Tuple[int, int]:
        edge_px = max(1, int(node_config.get("edge_px", 100) or 100))
        board_w = max(1, int(node_config.get("width", 20) or 20))
        board_h = max(1, int(node_config.get("height", 20) or 20))
        try:
            width_value = max(1.0, float(width_px or 1.0))
        except Exception:
            width_value = 1.0
        try:
            height_value = max(1.0, float(height_px or 1.0))
        except Exception:
            height_value = 1.0
        node_w = max(1, min(board_w, int(math.ceil(width_value / float(edge_px)))))
        node_h = max(1, min(board_h, int(math.ceil(height_value / float(edge_px)))))
        return int(node_w), int(node_h)

    def _render_node_block(self, *, node_w: int, node_h: int) -> str:
        safe_w = max(1, int(node_w))
        safe_h = max(1, int(node_h))
        row = "1" * safe_w
        return "\n".join(row for _ in range(safe_h))

    def _build_space_planner_chunks(
        self,
        *,
        chapters_out: List[Dict[str, Any]],
        prepared_objects: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        node_config = self._space_planner_node_config()
        objects_by_chapter: Dict[str, List[Dict[str, Any]]] = {}
        for row in (prepared_objects or []):
            if not isinstance(row, dict):
                continue
            chapter_id = str(row.get("chapter_id", "") or "").strip()
            if not chapter_id:
                continue
            objects_by_chapter.setdefault(chapter_id, []).append(row)

        chunks: List[Dict[str, Any]] = []
        for chapter in (chapters_out or []):
            if not isinstance(chapter, dict):
                continue
            chapter_id = str(chapter.get("chapter_id", "") or "").strip()
            if not chapter_id:
                continue
            title = str(chapter.get("title", "") or "").strip()
            logical_steps = chapter.get("logical_steps") if isinstance(chapter.get("logical_steps"), dict) else {}
            speech_steps = chapter.get("speech_steps") if isinstance(chapter.get("speech_steps"), dict) else {}
            qwen_steps = chapter.get("qwen_steps") if isinstance(chapter.get("qwen_steps"), dict) else {}
            step_order = [str(step_id) for step_id in logical_steps.keys()]
            if not step_order:
                step_order = [str(step_id) for step_id in speech_steps.keys()]

            merged_speech_parts: List[str] = []
            step_silence_rows: List[Dict[str, Any]] = []
            step_offsets: Dict[str, int] = {}
            step_silence_counts: Dict[str, int] = {}
            total_silences = 0

            for step_id in step_order:
                qwen_row = qwen_steps.get(step_id) if isinstance(qwen_steps.get(step_id), dict) else {}
                speech_text = str((qwen_row or {}).get("speech") or speech_steps.get(step_id) or "").strip()
                pause_markers = list((qwen_row or {}).get("pause_markers") or extract_pause_markers(speech_text))
                silence_count = int(len(pause_markers))
                step_offsets[step_id] = int(total_silences)
                step_silence_counts[step_id] = int(silence_count)
                step_silence_rows.append(
                    {
                        "step_id": step_id,
                        "global_silence_offset": int(total_silences),
                        "silence_count": int(silence_count),
                    }
                )
                if speech_text:
                    merged_speech_parts.append(speech_text)
                total_silences += silence_count

            merged_speech = "\n\n".join(part for part in merged_speech_parts if part).strip()
            collapsed: Dict[str, Dict[str, Any]] = {}
            for obj in (objects_by_chapter.get(chapter_id) or []):
                step_id = str(obj.get("step_id", "") or "").strip()
                step_offset = int(step_offsets.get(step_id, 0) or 0)
                step_silence_count = int(step_silence_counts.get(step_id, 0) or 0)

                local_start_raw = obj.get("start_silence_index")
                local_end_raw = obj.get("end_silence_index")
                try:
                    local_start = max(0, int(local_start_raw))
                except Exception:
                    local_start = 0
                if step_silence_count > 0:
                    local_start = min(local_start, max(0, step_silence_count - 1))
                global_start = int(step_offset + local_start)
                step_last_global = int(step_offset + max(0, step_silence_count - 1)) if step_silence_count > 0 else int(global_start)

                if local_end_raw is None:
                    global_end = int(step_last_global)
                else:
                    try:
                        local_end = max(0, int(local_end_raw))
                    except Exception:
                        local_end = int(local_start)
                    if step_silence_count > 0:
                        local_end = min(local_end, max(0, step_silence_count - 1))
                    global_end = int(step_offset + local_end)
                if global_end < global_start:
                    global_end = int(global_start)

                dimensions_px = obj.get("dimensions_px") if isinstance(obj.get("dimensions_px"), dict) else {}
                width_px = float((dimensions_px or {}).get("width", 1) or 1)
                height_px = float((dimensions_px or {}).get("height", 1) or 1)
                node_w, node_h = self._object_dimensions_to_nodes(
                    width_px=width_px,
                    height_px=height_px,
                    node_config=node_config,
                )

                name = str(obj.get("name", "") or "").strip()
                object_type = "text" if str(obj.get("object_type", "") or "").strip().lower() == "text" else "image"
                merge_key = f"{object_type}\u241f{name.casefold()}"
                existing = collapsed.get(merge_key)
                if existing is None:
                    existing = {
                        "object_key": f"{chapter_id}:{object_type}:{name}",
                        "chapter_id": chapter_id,
                        "name": name,
                        "type": object_type,
                        "diagram": int(obj.get("diagram", 0) or 0) if object_type == "image" else 0,
                        "start": int(global_start),
                        "end": int(global_end),
                        "range": int(max(0, global_end - global_start)),
                        "node_w": int(node_w),
                        "node_h": int(node_h),
                        "node_size": f"{int(node_w)}x{int(node_h)}",
                        "nodes": self._render_node_block(node_w=node_w, node_h=node_h),
                        "dimensions_px": {"width": int(round(width_px)), "height": int(round(height_px))},
                        "processed_id": str(obj.get("processed_id", "") or "").strip(),
                        "processed_ids": [str(x) for x in (obj.get("processed_ids") or []) if str(x).strip()],
                        "processed_png_path": str(obj.get("processed_png_path", "") or "").strip(),
                        "text_style_description": str(obj.get("text_style_description", "") or "").strip(),
                        "canonical_objects": list(obj.get("canonical_objects") or []),
                        "c2_link_map_path": str(obj.get("c2_link_map_path", "") or "").strip(),
                        "c2_artifact_path": str(obj.get("c2_artifact_path", "") or "").strip(),
                        "comfy_generated_paths": [str(x) for x in (obj.get("comfy_generated_paths") or []) if str(x).strip()],
                        "step_ids": [step_id] if step_id else [],
                        "mentions": [
                            {
                                "step_id": step_id,
                                "global_start": int(global_start),
                                "global_end": int(global_end),
                            }
                        ],
                    }
                    collapsed[merge_key] = existing
                    continue

                existing["start"] = int(min(int(existing.get("start", global_start) or global_start), int(global_start)))
                existing["end"] = int(max(int(existing.get("end", global_end) or global_end), int(global_end)))
                existing["range"] = int(max(0, int(existing["end"]) - int(existing["start"])))
                existing["node_w"] = int(max(int(existing.get("node_w", node_w) or node_w), int(node_w)))
                existing["node_h"] = int(max(int(existing.get("node_h", node_h) or node_h), int(node_h)))
                existing["node_size"] = f"{int(existing['node_w'])}x{int(existing['node_h'])}"
                existing["nodes"] = self._render_node_block(
                    node_w=int(existing["node_w"]),
                    node_h=int(existing["node_h"]),
                )
                existing_dims = existing.get("dimensions_px") if isinstance(existing.get("dimensions_px"), dict) else {}
                existing["dimensions_px"] = {
                    "width": int(max(int(existing_dims.get("width", 0) or 0), int(round(width_px)))),
                    "height": int(max(int(existing_dims.get("height", 0) or 0), int(round(height_px)))),
                }
                if step_id and step_id not in (existing.get("step_ids") or []):
                    existing.setdefault("step_ids", []).append(step_id)
                existing.setdefault("mentions", []).append(
                    {
                        "step_id": step_id,
                        "global_start": int(global_start),
                        "global_end": int(global_end),
                    }
                )
                for pid in [str(x) for x in (obj.get("processed_ids") or []) if str(x).strip()]:
                    if pid not in existing.setdefault("processed_ids", []):
                        existing["processed_ids"].append(pid)
                if not str(existing.get("processed_id", "") or "").strip():
                    existing["processed_id"] = str(obj.get("processed_id", "") or "").strip()
                if not str(existing.get("processed_png_path", "") or "").strip():
                    existing["processed_png_path"] = str(obj.get("processed_png_path", "") or "").strip()
                for path in [str(x) for x in (obj.get("comfy_generated_paths") or []) if str(x).strip()]:
                    if path not in existing.setdefault("comfy_generated_paths", []):
                        existing["comfy_generated_paths"].append(path)
                for component in list(obj.get("canonical_objects") or []):
                    if component not in existing.setdefault("canonical_objects", []):
                        existing["canonical_objects"].append(component)

            chunk_objects = list(collapsed.values())
            chunk_objects.sort(key=lambda row: (int(row.get("start", 0) or 0), int(row.get("end", 0) or 0), str(row.get("name", "")).lower()))
            prompt_objects = [
                {
                    "name": str(row.get("name", "") or ""),
                    "type": str(row.get("type", "") or ""),
                    "start": int(row.get("start", 0) or 0),
                    "end": int(row.get("end", 0) or 0),
                    "range": int(row.get("range", 0) or 0),
                    "node_w": int(row.get("node_w", 1) or 1),
                    "node_h": int(row.get("node_h", 1) or 1),
                    "nodes": str(row.get("nodes", "") or ""),
                }
                for row in chunk_objects
            ]
            chunks.append(
                {
                    "chunk_id": chapter_id,
                    "chapter_id": chapter_id,
                    "chunk_title": title,
                    "board_nodes": {"width": int(node_config["width"]), "height": int(node_config["height"])},
                    "node_edge_px": int(node_config["edge_px"]),
                    "total_silences": int(total_silences),
                    "merged_speech": merged_speech,
                    "step_silence_rows": step_silence_rows,
                    "objects": chunk_objects,
                    "prompt_objects": prompt_objects,
                }
            )

        saved_path = self._write_json_file(
            Path(self.debug_out_dir) / "space_planner_chunks_ready.json",
            {
                "schema": "space_planner_chunks_ready_v1",
                "node_config": node_config,
                "dimensions_config": self._load_dimensions_config(),
                "chunks": chunks,
            },
        )
        return {
            "chunks": chunks,
            "node_config": node_config,
            "saved_path": saved_path,
        }

    def _first_fit_space_for_object(
        self,
        *,
        board_width: int,
        board_height: int,
        node_w: int,
        node_h: int,
        occupied: set[Tuple[int, int]],
    ) -> List[List[int]]:
        safe_w = max(1, int(node_w))
        safe_h = max(1, int(node_h))
        max_x = max(0, int(board_width) - safe_w)
        max_y = max(0, int(board_height) - safe_h)
        for top in range(max_y + 1):
            for left in range(max_x + 1):
                blocked = False
                for yy in range(top, top + safe_h):
                    for xx in range(left, left + safe_w):
                        if (xx, yy) in occupied:
                            blocked = True
                            break
                    if blocked:
                        break
                if blocked:
                    continue
                for yy in range(top, top + safe_h):
                    for xx in range(left, left + safe_w):
                        occupied.add((xx, yy))
                return [
                    [left, top],
                    [left, top + safe_h - 1],
                    [left + safe_w - 1, top],
                    [left + safe_w - 1, top + safe_h - 1],
                ]
        left = 0
        top = 0
        return [
            [left, top],
            [left, min(max_y, top + safe_h - 1)],
            [min(max_x, left + safe_w - 1), top],
            [min(max_x, left + safe_w - 1), min(max_y, top + safe_h - 1)],
        ]

    def _fallback_space_planner_actions_for_chunk(self, *, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        board_nodes = dict(chunk.get("board_nodes") or {})
        board_width = max(1, int(board_nodes.get("width", 20) or 20))
        board_height = max(1, int(board_nodes.get("height", 20) or 20))
        occupied: set[Tuple[int, int]] = set()
        out: List[Dict[str, Any]] = []
        for row in list(chunk.get("prompt_objects") or []):
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "") or "").strip()
            action_type = str(row.get("type", "") or "").strip().lower()
            if not name or action_type not in {"image", "text"}:
                continue
            corners = self._first_fit_space_for_object(
                board_width=board_width,
                board_height=board_height,
                node_w=max(1, int(row.get("node_w", 1) or 1)),
                node_h=max(1, int(row.get("node_h", 1) or 1)),
                occupied=occupied,
            )
            start = max(0, int(row.get("start", 0) or 0))
            end = max(start, int(row.get("end", start) or start))
            out.append(
                {
                    "draw": 1,
                    "type": action_type,
                    "name": name,
                    "start": start,
                    "end": end,
                    "range": max(0, int(row.get("range", end - start) or end - start)),
                    "corners": corners,
                }
            )
        return out

    def run_space_planner_batch(
        self,
        *,
        chapters_out: List[Dict[str, Any]],
        prepared_objects: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        chunk_bundle = self._build_space_planner_chunks(
            chapters_out=chapters_out,
            prepared_objects=prepared_objects,
        )
        chunks = list(chunk_bundle.get("chunks") or [])
        if not chunks:
            return {
                "enabled": True,
                "request_count": 0,
                "qwen_ready": None,
                "chunks": [],
                "chunks_ready_path": chunk_bundle.get("saved_path"),
                "responses_path": None,
            }

        prompts: List[str] = []
        metas: List[Dict[str, Any]] = []
        system_prompt: Optional[str] = None
        for chunk in chunks:
            step_system_prompt, prompt = build_qwen_space_planner_prompt(
                chunk_id=str(chunk.get("chunk_id", "") or ""),
                chunk_title=str(chunk.get("chunk_title", "") or ""),
                merged_speech=str(chunk.get("merged_speech", "") or ""),
                board_nodes=dict(chunk.get("board_nodes") or {}),
                total_silences=int(chunk.get("total_silences", 0) or 0),
                objects=list(chunk.get("prompt_objects") or []),
            )
            system_prompt = system_prompt or step_system_prompt
            prompts.append(prompt)
            metas.append(
                {
                    "chunk_id": str(chunk.get("chunk_id", "") or ""),
                    "chapter_id": str(chunk.get("chapter_id", "") or ""),
                    "prompt_payload": {
                        "board_nodes": dict(chunk.get("board_nodes") or {}),
                        "total_silences": int(chunk.get("total_silences", 0) or 0),
                        "merged_speech": str(chunk.get("merged_speech", "") or ""),
                        "objects": list(chunk.get("prompt_objects") or []),
                    },
                }
            )

        generation = {
            "max_new_tokens": int(os.getenv("QWEN_SPACE_PLANNER_MAX_NEW_TOKENS", "3200") or 3200),
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.02,
        }
        response = self._generate_qwen_text_batch_cached(
            stage_name="space_planner",
            prompts=prompts,
            system_prompt=str(system_prompt or ""),
            generation=generation,
        )
        responses = response.get("responses")
        if not isinstance(responses, list) or len(responses) != len(prompts):
            raise RuntimeError("Qwen space planner response count does not match request count")

        stage_dir = Path(self.debug_out_dir) / "space_planner_qwen"
        stage_dir.mkdir(parents=True, exist_ok=True)
        out_chunks: List[Dict[str, Any]] = []
        for index, (meta, prompt, chunk, raw_row) in enumerate(zip(metas, prompts, chunks, responses), start=1):
            raw_text = str((raw_row or {}).get("text", "") or "").strip()
            parse_error = None
            try:
                parsed_actions = list((parse_qwen_space_planner_output(raw_text) or {}).get("actions") or [])
            except Exception as exc:
                parse_error = f"{type(exc).__name__}: {exc}"
                parsed_actions = self._fallback_space_planner_actions_for_chunk(chunk=chunk)
            file_path = self._write_json_file(
                stage_dir / f"{index:05d}_chunk_{str(meta.get('chunk_id', '')).replace('.', '_')}.json",
                {
                    "request_index": index,
                    "chunk_id": meta.get("chunk_id"),
                    "chapter_id": meta.get("chapter_id"),
                    "prompt": prompt,
                    "prompt_payload": meta.get("prompt_payload"),
                    "raw_response": raw_text,
                    "parsed_actions": parsed_actions,
                    "parse_error": parse_error,
                    "usage": raw_row,
                    "chunk_context": {
                        "step_silence_rows": chunk.get("step_silence_rows"),
                        "objects": chunk.get("objects"),
                    },
                },
            )
            out_chunks.append(
                {
                    "chunk_id": str(meta.get("chunk_id", "") or ""),
                    "chapter_id": str(meta.get("chapter_id", "") or ""),
                    "raw_response": raw_text,
                    "actions": parsed_actions,
                    "parse_error": parse_error,
                    "stage_io_path": file_path,
                    "prompt_payload": meta.get("prompt_payload"),
                    "board_nodes": dict(chunk.get("board_nodes") or {}),
                    "node_edge_px": int(chunk.get("node_edge_px", 100) or 100),
                    "step_silence_rows": list(chunk.get("step_silence_rows") or []),
                    "objects": list(chunk.get("objects") or []),
                }
            )

        responses_path = self._write_json_file(
            Path(self.debug_out_dir) / "space_planner_qwen_raw.json",
            {
                "schema": "space_planner_qwen_raw_v1",
                "request_count": len(prompts),
                "chunks_ready_path": chunk_bundle.get("saved_path"),
                "cache_hits": int(response.get("cache_hits", 0) or 0),
                "cache_misses": int(response.get("cache_misses", 0) or 0),
                "chunks": out_chunks,
            },
        )
        return {
            "enabled": True,
            "request_count": len(prompts),
            "qwen_ready": response.get("qwen_ready"),
            "cache_hits": int(response.get("cache_hits", 0) or 0),
            "cache_misses": int(response.get("cache_misses", 0) or 0),
            "chunks_ready_path": chunk_bundle.get("saved_path"),
            "responses_path": responses_path,
            "chunks": out_chunks,
        }

    def _strip_pause_tokens(self, speech_text: str) -> str:
        return re.sub(r"%(?:\s*)(?:\d+(?:\.\d+)?|\.\d+)", " ", str(speech_text or ""))

    def _count_spoken_words(self, speech_text: str) -> int:
        clean = self._strip_pause_tokens(speech_text)
        return len(re.findall(r"\b[\w']+\b", clean, flags=re.UNICODE))

    def _pause_index_to_local_word_map(self, speech_text: str) -> Dict[int, int]:
        text = str(speech_text or "")
        markers = extract_pause_markers(text)
        if not markers:
            return {}
        out: Dict[int, int] = {}
        for marker in markers:
            pause_index = int(marker.get("pause_index", 0) or 0)
            prefix = self._strip_pause_tokens(text[: int(marker.get("char_start", 0) or 0)])
            word_count = len(re.findall(r"\b[\w']+\b", prefix, flags=re.UNICODE))
            out[pause_index] = max(1, int(word_count))
        return out

    def _build_step_sync_contexts(
        self,
        *,
        chapters_out: List[Dict[str, Any]],
        use_full_speech: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        lesson_word_offset = 0
        lesson_pause_offset = 0
        for chapter in chapters_out or []:
            chapter_id = str(chapter.get("chapter_id", "") or "").strip()
            chapter_word_offset = 0
            logical_steps = chapter.get("logical_steps") if isinstance(chapter.get("logical_steps"), dict) else {}
            speech_steps = chapter.get("speech_steps") if isinstance(chapter.get("speech_steps"), dict) else {}
            qwen_steps = chapter.get("qwen_steps") if isinstance(chapter.get("qwen_steps"), dict) else {}
            step_order = [str(step_id) for step_id in logical_steps.keys()] or [str(step_id) for step_id in speech_steps.keys()]
            for step_id in step_order:
                qwen_row = qwen_steps.get(step_id) if isinstance(qwen_steps.get(step_id), dict) else {}
                if use_full_speech:
                    speech = str(speech_steps.get(step_id) or (qwen_row or {}).get("speech") or "").strip()
                else:
                    speech = str((qwen_row or {}).get("speech") or speech_steps.get(step_id) or "").strip()
                word_count = max(1, self._count_spoken_words(speech))
                pause_map = self._pause_index_to_local_word_map(speech)
                pause_markers = list((qwen_row or {}).get("pause_markers") or extract_pause_markers(speech))
                out[step_id] = {
                    "step_id": step_id,
                    "chapter_id": chapter_id,
                    "speech": speech,
                    "word_count": word_count,
                    "chapter_word_offset": int(chapter_word_offset),
                    "lesson_word_offset": int(lesson_word_offset),
                    "global_pause_offset": int(lesson_pause_offset),
                    "pause_count": int(len(pause_markers)),
                    "pause_to_local_word_index": pause_map,
                }
                chapter_word_offset += word_count
                lesson_word_offset += word_count
                lesson_pause_offset += int(len(pause_markers))
        return out

    def _fallback_full_speech_word_index(
        self,
        *,
        compressed_word_index: int,
        compressed_speech: str,
        full_speech: str,
    ) -> int:
        compressed_count = max(1, self._count_spoken_words(compressed_speech))
        full_count = max(1, self._count_spoken_words(full_speech))
        compressed_local = max(1, min(compressed_count, int(compressed_word_index or 1)))
        if compressed_count <= 1 or full_count <= 1:
            return max(1, min(full_count, compressed_local))
        scale = float(max(0, full_count - 1)) / float(max(1, compressed_count - 1))
        return max(1, min(full_count, 1 + int(round((compressed_local - 1) * scale))))

    def _run_full_speech_action_sync_batch(
        self,
        *,
        chapters_out: List[Dict[str, Any]],
        actions_by_step: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        prompts: List[str] = []
        metas: List[Dict[str, Any]] = []
        system_prompt: Optional[str] = None
        chapter_map = {
            str(chapter.get("chapter_id", "") or "").strip(): chapter
            for chapter in chapters_out or []
            if isinstance(chapter, dict)
        }

        for chapter in chapters_out or []:
            chapter_id = str(chapter.get("chapter_id", "") or "").strip()
            speech_steps = chapter.get("speech_steps") if isinstance(chapter.get("speech_steps"), dict) else {}
            qwen_steps = chapter.get("qwen_steps") if isinstance(chapter.get("qwen_steps"), dict) else {}
            logical_steps = chapter.get("logical_steps") if isinstance(chapter.get("logical_steps"), dict) else {}
            step_order = [str(step_id) for step_id in logical_steps.keys()] or [str(step_id) for step_id in speech_steps.keys()]
            for step_id in step_order:
                step_actions = [dict(row) for row in (actions_by_step.get(step_id) or []) if isinstance(row, dict)]
                if not step_actions:
                    continue
                compressed_speech = str(((qwen_steps.get(step_id) or {}) if isinstance(qwen_steps.get(step_id), dict) else {}).get("speech") or speech_steps.get(step_id) or "").strip()
                full_speech = str(speech_steps.get(step_id) or compressed_speech).strip()
                action_payload = [
                    {
                        "action_id": str(row.get("action_id", "") or "").strip(),
                        "kind": str(row.get("kind", "") or "").strip(),
                        "action": str(row.get("action", "") or "").strip(),
                        "type": str(row.get("type", "") or "").strip(),
                        "name": str(row.get("name", row.get("target", "")) or "").strip(),
                        "data": str(row.get("data", "") or "").strip(),
                        "compressed_word_index": max(1, int(row.get("compressed_local_step_sync_word_index", 1) or 1)),
                    }
                    for row in step_actions
                    if str(row.get("action_id", "") or "").strip()
                ]
                if not action_payload:
                    continue
                prompt_system, prompt_text = build_qwen_full_speech_action_sync_prompt(
                    step_id=step_id,
                    compressed_speech=compressed_speech,
                    full_speech=full_speech,
                    actions=action_payload,
                )
                system_prompt = system_prompt or prompt_system
                prompts.append(prompt_text)
                metas.append(
                    {
                        "chapter_id": chapter_id,
                        "step_id": step_id,
                        "compressed_speech": compressed_speech,
                        "full_speech": full_speech,
                        "actions": action_payload,
                    }
                )

        if not prompts:
            out = {
                "enabled": True,
                "request_count": 0,
                "by_step": {},
                "rows": [],
                "raw_io_path": None,
                "cache_hits": 0,
                "cache_misses": 0,
                "qwen_ready": None,
            }
            out["saved_path"] = self._write_json_file(self._internal_runtime_dir() / "full_chunk_speech_action_sync.json", out)
            return out

        result = self._generate_qwen_text_batch_cached(
            stage_name="full_chunk_speech_action_sync",
            prompts=prompts,
            system_prompt=str(system_prompt or ""),
            generation={
                "max_new_tokens": int(os.getenv("QWEN_FULL_SPEECH_ACTION_SYNC_MAX_NEW_TOKENS", "2200") or 2200),
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.02,
            },
        )
        responses = result.get("responses") if isinstance(result.get("responses"), list) else []
        if len(responses) != len(prompts):
            raise RuntimeError("Qwen full_chunk_speech_action_sync response count does not match request count")

        by_step: Dict[str, Dict[str, Any]] = {}
        rows_out: List[Dict[str, Any]] = []
        raw_io_rows: List[Dict[str, Any]] = []
        for index, (meta, prompt_text, raw_row) in enumerate(zip(metas, prompts, responses), start=1):
            raw_text = str((raw_row or {}).get("text", "") or "").strip()
            parsed: Dict[str, Any]
            parse_error: Optional[str] = None
            try:
                parsed = parse_qwen_full_speech_action_sync_output(
                    raw_text,
                    action_ids=[str(row.get("action_id", "") or "").strip() for row in meta["actions"]],
                )
            except Exception as exc:
                parse_error = f"{type(exc).__name__}: {exc}"
                parsed = {"actions": []}

            full_count = max(1, self._count_spoken_words(str(meta.get("full_speech", "") or "")))
            sync_lookup = {
                str(row.get("action_id", "") or "").strip(): max(1, min(full_count, int(row.get("full_word_index", 1) or 1)))
                for row in list(parsed.get("actions") or [])
                if isinstance(row, dict) and str(row.get("action_id", "") or "").strip()
            }
            resolved_actions: List[Dict[str, Any]] = []
            for action in meta["actions"]:
                action_id = str(action.get("action_id", "") or "").strip()
                compressed_word_index = max(1, int(action.get("compressed_word_index", 1) or 1))
                full_word_index = sync_lookup.get(action_id)
                if full_word_index is None:
                    full_word_index = self._fallback_full_speech_word_index(
                        compressed_word_index=compressed_word_index,
                        compressed_speech=str(meta.get("compressed_speech", "") or ""),
                        full_speech=str(meta.get("full_speech", "") or ""),
                    )
                resolved_actions.append(
                    {
                        "action_id": action_id,
                        "compressed_word_index": compressed_word_index,
                        "full_word_index": max(1, min(full_count, int(full_word_index or 1))),
                    }
                )

            step_key = str(meta.get("step_id", "") or "").strip()
            by_step[step_key] = {
                "chapter_id": str(meta.get("chapter_id", "") or "").strip(),
                "step_id": step_key,
                "compressed_speech": str(meta.get("compressed_speech", "") or ""),
                "full_speech": str(meta.get("full_speech", "") or ""),
                "actions": resolved_actions,
                "parse_error": parse_error,
            }
            rows_out.append(by_step[step_key])
            raw_io_rows.append(
                {
                    "request_index": index,
                    "chapter_id": meta.get("chapter_id"),
                    "step_id": meta.get("step_id"),
                    "prompt": prompt_text,
                    "raw_response": raw_text,
                    "parsed": parsed,
                    "resolved_actions": resolved_actions,
                    "parse_error": parse_error,
                    "usage": raw_row,
                }
            )

        raw_io_path = self._write_json_file(
            self._internal_runtime_dir() / "qwen_raw_io" / "full_chunk_speech_action_sync.json",
            {
                "system_prompt": system_prompt,
                "rows": raw_io_rows,
            },
        )
        out = {
            "enabled": True,
            "request_count": len(prompts),
            "by_step": by_step,
            "rows": rows_out,
            "raw_io_path": raw_io_path,
            "cache_hits": int(result.get("cache_hits", 0) or 0),
            "cache_misses": int(result.get("cache_misses", 0) or 0),
            "qwen_ready": result.get("qwen_ready"),
        }
        out["saved_path"] = self._write_json_file(self._internal_runtime_dir() / "full_chunk_speech_action_sync.json", out)
        return out

    def _local_pause_to_word_index(self, *, context: Dict[str, Any], pause_index: int) -> int:
        pause_map = dict(context.get("pause_to_local_word_index") or {})
        word_count = max(1, int(context.get("word_count", 1) or 1))
        if pause_map:
            if int(pause_index) in pause_map:
                return max(1, min(word_count, int(pause_map[int(pause_index)] or 1)))
            nearest = max((idx for idx in pause_map.keys() if int(idx) <= int(pause_index)), default=None)
            if nearest is not None:
                return max(1, min(word_count, int(pause_map[nearest] or 1)))
        return max(1, min(word_count, int(pause_index) + 1))

    def _spread_local_action_indexes(self, *, indexes: List[int], word_count: int) -> List[int]:
        if not indexes:
            return []
        raw_values = [int(value or 1) for value in indexes]
        min_raw = min(raw_values)
        max_raw = max(raw_values)
        if min_raw < 1 or max_raw > int(word_count):
            if max_raw == min_raw:
                capped = [max(1, min(int(word_count), 1)) for _ in raw_values]
            else:
                scale = float(max(1, int(word_count) - 1)) / float(max_raw - min_raw)
                capped = [
                    max(1, min(int(word_count), 1 + int(round((value - min_raw) * scale))))
                    for value in raw_values
                ]
        else:
            capped = [max(1, min(int(word_count), value)) for value in raw_values]
        groups: "OrderedDict[int, List[int]]" = OrderedDict()
        for original_idx, base in enumerate(capped):
            groups.setdefault(base, []).append(original_idx)
        bases = list(groups.keys())
        assigned = [1] * len(capped)
        for pos, base in enumerate(bases):
            members = groups[base]
            next_base = bases[pos + 1] if pos + 1 < len(bases) else int(word_count) + 1
            slot_cap = max(1, next_base - base)
            for offset, original_idx in enumerate(members):
                assigned[original_idx] = min(int(word_count), base + min(offset, slot_cap - 1))
        return assigned

    def _locate_step_for_global_pause(
        self,
        *,
        step_silence_rows: List[Dict[str, Any]],
        pause_index: int,
    ) -> Dict[str, Any]:
        rows = [row for row in (step_silence_rows or []) if isinstance(row, dict)]
        if not rows:
            return {"step_id": "", "global_silence_offset": 0, "silence_count": 0}
        rows.sort(key=lambda row: int(row.get("global_silence_offset", 0) or 0))
        for idx, row in enumerate(rows):
            start = int(row.get("global_silence_offset", 0) or 0)
            next_start = int(rows[idx + 1].get("global_silence_offset", 0) or 0) if idx + 1 < len(rows) else None
            if next_start is None or int(pause_index) < next_start:
                return row
        return rows[-1]

    def _node_corners_to_allocated_space(self, *, corners: List[List[int]], node_edge_px: int) -> Dict[str, Any]:
        xs = [int(corner[0]) for corner in corners if isinstance(corner, list) and len(corner) == 2]
        ys = [int(corner[1]) for corner in corners if isinstance(corner, list) and len(corner) == 2]
        if len(xs) != 4 or len(ys) != 4:
            return {"node_corners": corners, "corners_px": []}
        left = min(xs) * int(node_edge_px)
        right = (max(xs) + 1) * int(node_edge_px)
        top = min(ys) * int(node_edge_px)
        bottom = (max(ys) + 1) * int(node_edge_px)
        return {
            "node_corners": corners,
            "corners_px": [
                [float(left), float(top)],
                [float(left), float(bottom)],
                [float(right), float(top)],
                [float(right), float(bottom)],
            ],
        }

    def run_diagram_action_planner_batch(
        self,
        *,
        chapters_out: List[Dict[str, Any]],
        prepared_objects: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        step_contexts = self._build_step_sync_contexts(chapters_out=chapters_out)
        step_images: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for obj in prepared_objects or []:
            if not isinstance(obj, dict) or str(obj.get("object_type", "") or "").strip() != "image":
                continue
            step_id = str(obj.get("step_id", "") or "").strip()
            context = step_contexts.get(step_id)
            if context is None:
                continue
            name = str(obj.get("name", "") or "").strip()
            if not name:
                continue
            start_pause = max(0, int(obj.get("start_silence_index", 0) or 0))
            end_raw = obj.get("end_silence_index")
            if end_raw is None:
                end_word = int(context.get("word_count", 1) or 1)
            else:
                end_word = self._local_pause_to_word_index(context=context, pause_index=max(0, int(end_raw or 0)))
            image_row = step_images.setdefault(step_id, {}).setdefault(
                name.casefold(),
                {
                    "name": name,
                    "components": [],
                    "word_sync": {
                        "start": self._local_pause_to_word_index(context=context, pause_index=start_pause),
                        "end": max(1, int(end_word)),
                    },
                },
            )
            image_row["word_sync"]["start"] = min(
                int(image_row["word_sync"]["start"]),
                self._local_pause_to_word_index(context=context, pause_index=start_pause),
            )
            image_row["word_sync"]["end"] = max(
                int(image_row["word_sync"]["end"]),
                max(1, int(end_word)),
            )
            for component in list(obj.get("canonical_objects") or obj.get("required_objects") or []):
                cleaned = normalize_ws(component)
                if cleaned and cleaned not in image_row["components"]:
                    image_row["components"].append(cleaned)

        prompts: List[str] = []
        metas: List[Dict[str, Any]] = []
        system_prompt: Optional[str] = None
        for chapter in chapters_out or []:
            speech_steps = chapter.get("speech_steps") if isinstance(chapter.get("speech_steps"), dict) else {}
            qwen_steps = chapter.get("qwen_steps") if isinstance(chapter.get("qwen_steps"), dict) else {}
            logical_steps = chapter.get("logical_steps") if isinstance(chapter.get("logical_steps"), dict) else {}
            step_order = [str(step_id) for step_id in logical_steps.keys()] or [str(step_id) for step_id in speech_steps.keys()]
            for step_id in step_order:
                image_rows = list((step_images.get(step_id) or {}).values())
                if not image_rows:
                    continue
                speech = str(((qwen_steps.get(step_id) or {}) if isinstance(qwen_steps.get(step_id), dict) else {}).get("speech") or speech_steps.get(step_id) or "").strip()
                prompt_system, prompt_text = build_qwen_diagram_action_planner_prompt(
                    step_id=step_id,
                    speech=speech,
                    images=image_rows,
                )
                system_prompt = system_prompt or prompt_system
                prompts.append(prompt_text)
                metas.append(
                    {
                        "step_id": step_id,
                        "chapter_id": str(chapter.get("chapter_id", "") or "").strip(),
                        "images": image_rows,
                    }
                )

        if not prompts:
            return {
                "enabled": True,
                "request_count": 0,
                "rows": [],
                "responses_path": None,
                "qwen_ready": None,
                "cache_hits": 0,
                "cache_misses": 0,
            }

        generation = {
            "max_new_tokens": int(os.getenv("QWEN_DIAGRAM_ACTION_PLANNER_MAX_NEW_TOKENS", "1400") or 1400),
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.03,
        }
        result = self._generate_qwen_text_batch_cached(
            stage_name="diagram_action_planner",
            prompts=prompts,
            system_prompt=str(system_prompt or ""),
            generation=generation,
        )
        responses = result.get("responses")
        if not isinstance(responses, list) or len(responses) != len(prompts):
            raise RuntimeError("Qwen diagram_action_planner response count does not match request count")

        rows_out: List[Dict[str, Any]] = []
        for meta, raw_row in zip(metas, responses):
            raw_text = str((raw_row or {}).get("text", "") or "").strip()
            parsed = parse_qwen_diagram_action_planner_output(raw_text)
            image_ranges: Dict[str, Tuple[int, int]] = {}
            for image in list(meta.get("images") or []):
                if not isinstance(image, dict):
                    continue
                name = str(image.get("name", "") or "").strip()
                word_sync = image.get("word_sync") if isinstance(image.get("word_sync"), dict) else {}
                if name:
                    image_ranges[name.casefold()] = (
                        max(1, int(word_sync.get("start", 1) or 1)),
                        max(1, int(word_sync.get("end", word_sync.get("start", 1)) or word_sync.get("start", 1) or 1)),
                    )
            cleaned_actions: List[Dict[str, Any]] = []
            for action in list(parsed.get("actions") or []):
                if not isinstance(action, dict):
                    continue
                target = str(action.get("target", "") or "").strip()
                image_name = target.split(" : ", 1)[0].strip() if " : " in target else target
                range_start, range_end = image_ranges.get(image_name.casefold(), (1, max(1, int(step_contexts.get(meta["step_id"], {}).get("word_count", 1) or 1))))
                sync_index = max(range_start, min(range_end, int(action.get("sync_index", range_start) or range_start)))
                cleaned_actions.append(
                    {
                        "type": str(action.get("type", "") or "").strip(),
                        "target": target,
                        "data": str(action.get("data", "") or ""),
                        "sync_index": sync_index,
                        "init": 1 if int(action.get("init", 0) or 0) == 1 else 0,
                    }
                )
            rows_out.append(
                {
                    "step_id": meta["step_id"],
                    "chapter_id": meta["chapter_id"],
                    "images": list(meta.get("images") or []),
                    "raw_response": raw_text,
                    "actions": cleaned_actions,
                }
            )

        responses_path = self._write_json_file(
            self._internal_runtime_dir() / "qwen_raw_responses" / "diagram_action_planner.json",
            {
                "stage": "diagram_action_planner",
                "responses": [str((row or {}).get("raw_response", "") or "") for row in rows_out],
            },
        )
        return {
            "enabled": True,
            "request_count": len(prompts),
            "rows": rows_out,
            "responses_path": responses_path,
            "qwen_ready": result.get("qwen_ready"),
            "cache_hits": int(result.get("cache_hits", 0) or 0),
            "cache_misses": int(result.get("cache_misses", 0) or 0),
        }

    def _build_combined_lesson_action_payload(
        self,
        *,
        chapters_out: List[Dict[str, Any]],
        space_planner_result: Dict[str, Any],
        diagram_action_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        compressed_step_contexts = self._build_step_sync_contexts(chapters_out=chapters_out, use_full_speech=False)
        full_step_contexts = self._build_step_sync_contexts(chapters_out=chapters_out, use_full_speech=True)
        step_order = [
            str(step_id)
            for chapter in chapters_out or []
            for step_id in (
                list((chapter.get("logical_steps") or {}).keys())
                or list((chapter.get("speech_steps") or {}).keys())
            )
        ]

        actions_by_step: Dict[str, List[Dict[str, Any]]] = {}
        next_action_number = 1

        def _register_step_action(step_id: str, row: Dict[str, Any]) -> None:
            nonlocal next_action_number
            clean_step_id = str(step_id or "").strip()
            if not clean_step_id:
                return
            payload = dict(row)
            payload["step_id"] = clean_step_id
            payload["action_id"] = f"A{next_action_number}"
            next_action_number += 1
            actions_by_step.setdefault(clean_step_id, []).append(payload)

        for step_id in step_order:
            context = compressed_step_contexts.get(step_id)
            if context is None:
                continue
            speech = str(context.get("speech", "") or "")
            pause_markers = extract_pause_markers(speech)
            for marker in pause_markers:
                pause_index = int(marker.get("pause_index", 0) or 0)
                local_word = self._local_pause_to_word_index(context=context, pause_index=pause_index)
                _register_step_action(
                    step_id,
                    {
                        "kind": "silence",
                        "action": "silence",
                        "type": "silence",
                        "chunk_id": str(context.get("chapter_id", "") or ""),
                        "chapter_id": str(context.get("chapter_id", "") or ""),
                        "pause_index": pause_index,
                        "global_pause_index": int(context.get("global_pause_offset", 0) or 0) + pause_index,
                        "length": float(marker.get("seconds", 0.0) or 0.0),
                        "compressed_local_step_sync_word_index": local_word,
                    },
                )

        for chunk in list(space_planner_result.get("chunks") or []):
            if not isinstance(chunk, dict):
                continue
            step_rows = list(chunk.get("step_silence_rows") or [])
            node_edge_px = int(chunk.get("node_edge_px", 100) or 100)
            for action in list(chunk.get("actions") or []):
                if not isinstance(action, dict):
                    continue
                pause_index = (
                    int(action.get("start", 0) or 0)
                    if int(action.get("draw", 0) or 0) == 1
                    else int(action.get("end", action.get("start", 0)) or action.get("start", 0) or 0)
                )
                step_row = self._locate_step_for_global_pause(step_silence_rows=step_rows, pause_index=pause_index)
                step_id = str(step_row.get("step_id", "") or "").strip()
                context = compressed_step_contexts.get(step_id, {})
                local_pause = max(0, pause_index - int(step_row.get("global_silence_offset", 0) or 0))
                local_word = self._local_pause_to_word_index(context=context, pause_index=local_pause) if context else 1
                allocated_space = self._node_corners_to_allocated_space(
                    corners=list(action.get("corners") or []),
                    node_edge_px=node_edge_px,
                )
                corners_px = list(allocated_space.get("corners_px") or [])
                location = {"x": 0.0, "y": 0.0}
                if len(corners_px) == 4:
                    xs = [float(corner[0]) for corner in corners_px]
                    ys = [float(corner[1]) for corner in corners_px]
                    location = {"x": float((min(xs) + max(xs)) / 2.0), "y": float((min(ys) + max(ys)) / 2.0)}
                _register_step_action(
                    step_id,
                    {
                        "kind": "space",
                        "action": "draw" if int(action.get("draw", 0) or 0) == 1 else "delete",
                        "chunk_id": str(chunk.get("chunk_id", "") or ""),
                        "chapter_id": str(chunk.get("chapter_id", "") or ""),
                        "name": str(action.get("name", "") or "").strip(),
                        "type": str(action.get("type", "") or "").strip(),
                        "global_pause_index": pause_index,
                        "local_step_silence_index": local_pause,
                        "compressed_local_step_sync_word_index": local_word,
                        "location": location,
                        "allocated_space": allocated_space,
                    },
                )

        for row in list(diagram_action_result.get("rows") or []):
            if not isinstance(row, dict):
                continue
            step_id = str(row.get("step_id", "") or "").strip()
            context = compressed_step_contexts.get(step_id)
            if context is None:
                continue
            raw_indexes = [
                max(1, int(action.get("sync_index", 1) or 1))
                for action in list(row.get("actions") or [])
                if isinstance(action, dict)
            ]
            spread_indexes = self._spread_local_action_indexes(
                indexes=raw_indexes,
                word_count=max(1, int(context.get("word_count", 1) or 1)),
            )
            spread_iter = iter(spread_indexes)
            for action in list(row.get("actions") or []):
                if not isinstance(action, dict):
                    continue
                local_word = next(spread_iter, max(1, int(action.get("sync_index", 1) or 1)))
                _register_step_action(
                    step_id,
                    {
                        "kind": "diagram",
                        "action": str(action.get("type", "") or "").strip(),
                        "chunk_id": str(context.get("chapter_id", "") or ""),
                        "chapter_id": str(context.get("chapter_id", "") or ""),
                        "target": str(action.get("target", "") or "").strip(),
                        "type": str(action.get("type", "") or "").strip(),
                        "data": str(action.get("data", "") or ""),
                        "init": 1 if int(action.get("init", 0) or 0) == 1 else 0,
                        "compressed_local_step_sync_word_index": local_word,
                    },
                )

        sync_result = self._run_full_speech_action_sync_batch(
            chapters_out=chapters_out,
            actions_by_step=actions_by_step,
        )
        sync_by_step = sync_result.get("by_step") if isinstance(sync_result.get("by_step"), dict) else {}

        silence_actions: List[Dict[str, Any]] = []
        space_actions: List[Dict[str, Any]] = []
        diagram_actions: List[Dict[str, Any]] = []

        for step_id in step_order:
            full_context = full_step_contexts.get(step_id, {})
            compressed_context = compressed_step_contexts.get(step_id, {})
            resolved_rows = list((sync_by_step.get(step_id) or {}).get("actions") or [])
            resolved_lookup = {
                str(row.get("action_id", "") or "").strip(): max(1, int(row.get("full_word_index", 1) or 1))
                for row in resolved_rows
                if isinstance(row, dict) and str(row.get("action_id", "") or "").strip()
            }
            for action_row in list(actions_by_step.get(step_id) or []):
                if not isinstance(action_row, dict):
                    continue
                action_id = str(action_row.get("action_id", "") or "").strip()
                compressed_local_word = max(1, int(action_row.get("compressed_local_step_sync_word_index", 1) or 1))
                full_local_word = resolved_lookup.get(action_id)
                if full_local_word is None:
                    full_local_word = self._fallback_full_speech_word_index(
                        compressed_word_index=compressed_local_word,
                        compressed_speech=str(compressed_context.get("speech", "") or ""),
                        full_speech=str(full_context.get("speech", "") or compressed_context.get("speech", "") or ""),
                    )
                base_row = dict(action_row)
                base_row["local_step_sync_word_index"] = full_local_word
                base_row["compressed_local_step_sync_word_index"] = compressed_local_word
                base_row["chapter_sync_word_index"] = int(full_context.get("chapter_word_offset", 0) or 0) + full_local_word - 1
                base_row["sync_word_index"] = int(full_context.get("lesson_word_offset", 0) or 0) + full_local_word - 1
                base_row["compressed_sync_word_index"] = int(compressed_context.get("lesson_word_offset", 0) or 0) + compressed_local_word - 1
                base_row["compressed_chapter_sync_word_index"] = int(compressed_context.get("chapter_word_offset", 0) or 0) + compressed_local_word - 1
                if base_row.get("kind") == "silence":
                    silence_actions.append(base_row)
                elif base_row.get("kind") == "space":
                    space_actions.append(base_row)
                elif base_row.get("kind") == "diagram":
                    diagram_actions.append(base_row)

        lesson_actions = list(silence_actions) + list(space_actions) + list(diagram_actions)
        lesson_actions.sort(
            key=lambda row: (
                int(row.get("sync_word_index", 0) or 0),
                int(0 if str(row.get("action", "") or "") == "silence" else 1),
                str(row.get("chapter_id", "") or ""),
                str(row.get("step_id", "") or ""),
                str(row.get("action", "") or ""),
            )
        )
        for action_index, row in enumerate(lesson_actions):
            row["action_index"] = int(action_index)
            row["action_order_id"] = int(action_index + 1)

        sync_chunks: List[Dict[str, Any]] = []
        for step_id in step_order:
            rows = [row for row in lesson_actions if str(row.get("step_id", "") or "") == step_id]
            if not rows:
                continue
            rows.sort(
                key=lambda row: (
                    int(row.get("local_step_sync_word_index", 0) or 0),
                    int(row.get("action_index", 0) or 0),
                )
            )
            full_context = full_step_contexts.get(step_id, {})
            sync_chunks.append(
                {
                    "chunk_id": str(full_context.get("chapter_id", "") or ""),
                    "chapter_id": str(full_context.get("chapter_id", "") or ""),
                    "step_id": step_id,
                    "action_indexes": {
                        str(row.get("action_id", "") or ""): int(row.get("chapter_sync_word_index", 0) or 0)
                        for row in rows
                        if str(row.get("action_id", "") or "").strip()
                    },
                    "refined_count": max(0, len({int(row.get("chapter_sync_word_index", 0) or 0) for row in rows}) - 1),
                }
            )
        sync_chunks.sort(key=lambda row: (str(row.get("chapter_id", "")), str(row.get("step_id", ""))))

        saved_path = self._write_json_file(
            Path(self.debug_out_dir) / "final_actions_by_global_word_index.json",
            {
                "schema": "final_actions_by_global_word_index_v1",
                "lesson_actions": lesson_actions,
                "space_actions": space_actions,
                "diagram_actions": diagram_actions,
                "silence_actions": silence_actions,
                "full_chunk_speech_action_sync_path": sync_result.get("saved_path"),
            },
        )
        return {
            "saved_path": saved_path,
            "lesson_actions": lesson_actions,
            "space_actions": space_actions,
            "diagram_actions": diagram_actions,
            "silence_actions": silence_actions,
            "full_chunk_speech_action_sync": {
                "saved_path": sync_result.get("saved_path"),
                "raw_io_path": sync_result.get("raw_io_path"),
                "chunks": sync_chunks,
                "cache_hits": int(sync_result.get("cache_hits", 0) or 0),
                "cache_misses": int(sync_result.get("cache_misses", 0) or 0),
            },
        }

    def run_post_research_pipeline(self, *, topic: str, first_module_result: Dict[str, Any]) -> Dict[str, Any]:
        chapters_out = list(first_module_result.get("chapters") or [])
        prompt_meta = self._collect_prompt_meta(topic=topic, chapters_out=chapters_out)
        diagram_prompts = [prompt for prompt, meta in prompt_meta.items() if int((meta or {}).get("diagram", 0) or 0) == 1]
        c2_rows = [
            {
                "prompt": prompt,
                "requested_components": list((prompt_meta.get(prompt) or {}).get("diagram_required_objects") or []),
            }
            for prompt in diagram_prompts
        ]

        debug: Dict[str, Any] = {
            "prompt_meta": prompt_meta,
            "diagram_prompts": diagram_prompts,
            "non_diagram_prompts": [prompt for prompt in prompt_meta.keys() if prompt not in diagram_prompts],
            "pipeline_errors": {},
            "qwen_speech_sync": {"status": "pending_until_after_research_c2_siglip"},
        }

        if diagram_prompts and self._diagram_research_thread is None:
            self._dbg("Starting diagram research from post-research stage", data={"prompts": diagram_prompts})
            self.start_diagram_research(topic=topic, chapters_out=chapters_out)

        cached_c2_bundle = self._load_c2_bundle_cache_for_rows(c2_rows)
        c2_branch_holder: Dict[str, Any] = {
            "verifier": {"enabled": False, "by_prompt": {}, "qwen_ready": None, "cache_hits": 0, "cache_misses": 0, "note": "not_started"},
            "bundle": {"ok": True, "prompts": {}, "errors": {}, "count": 0},
            "errors": {},
        }

        def _run_c2_branch() -> None:
            branch_rows = [dict(row) for row in c2_rows]
            verifier_result: Dict[str, Any] = {"enabled": False, "by_prompt": {}, "qwen_ready": None, "cache_hits": 0, "cache_misses": 0}
            if isinstance(cached_c2_bundle, dict):
                verifier_result = {
                    "enabled": False,
                    "by_prompt": {},
                    "qwen_ready": None,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "note": "skipped_existing_c2_bundle_cache",
                }
                c2_branch_holder["verifier"] = verifier_result
                c2_branch_holder["bundle"] = cached_c2_bundle
                return

            if _env_bool("NEWTIMELINE_RUN_C2_VERIFIER", True):
                try:
                    verifier_result = self._run_c2_component_verifier_batch(c2_bundle_rows=branch_rows)
                except Exception as exc:
                    verifier_result = {"enabled": False, "error": f"{type(exc).__name__}: {exc}", "by_prompt": {}}
                    c2_branch_holder["errors"]["verifier"] = verifier_result["error"]
            else:
                verifier_result = {
                    "enabled": False,
                    "by_prompt": {},
                    "qwen_ready": None,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "note": "disabled_by_env",
                }
            c2_branch_holder["verifier"] = verifier_result

            verifier_by_prompt = dict(verifier_result.get("by_prompt") or {})
            for row in branch_rows:
                prompt = normalize_ws(row.get("prompt"))
                verdict = verifier_by_prompt.get(prompt) or {}
                row["skip_stage1"] = 1 if int(verdict.get("skip_stage1", 0) or 0) == 1 else 0
                row["verifier_missing"] = list(verdict.get("missing") or [])

            try:
                c2_branch_holder["bundle"] = self._run_c2_bundle(c2_bundle_rows=branch_rows)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                c2_branch_holder["errors"]["bundle"] = err
                c2_branch_holder["bundle"] = {"ok": False, "prompts": {}, "errors": {"bundle": err}, "count": 0}

        c2_thread = threading.Thread(target=_run_c2_branch, name="newtimeline_c2_branch", daemon=False)
        c2_thread.start()
        research_state = self._await_diagram_research()
        c2_thread.join()

        c2_verifier_result = dict(c2_branch_holder.get("verifier") or {})
        verifier_by_prompt = dict(c2_verifier_result.get("by_prompt") or {})
        c2_result_holder = dict(c2_branch_holder.get("bundle") or {})

        debug["diagram_research"] = research_state
        debug["c2_component_verifier"] = {
            "enabled": bool(c2_verifier_result.get("enabled", True)),
            "saved_path": c2_verifier_result.get("saved_path"),
            "cache_hits": int(c2_verifier_result.get("cache_hits", 0) or 0),
            "cache_misses": int(c2_verifier_result.get("cache_misses", 0) or 0),
            "rows": [
                {
                    "prompt": prompt,
                    "skip_stage1": int((verifier_by_prompt.get(prompt) or {}).get("skip_stage1", 0) or 0),
                    "missing": list((verifier_by_prompt.get(prompt) or {}).get("missing") or []),
                }
                for prompt in diagram_prompts
                if prompt in verifier_by_prompt
            ],
        }
        debug["c2_bundle"] = {
            "ok": bool(c2_result_holder.get("ok", True)),
            "count": int(c2_result_holder.get("count", 0) or 0),
            "errors": c2_result_holder.get("errors", {}),
        }
        if c2_branch_holder.get("errors"):
            for key, value in dict(c2_branch_holder.get("errors") or {}).items():
                debug["pipeline_errors"][f"c2_{key}"] = value
        if research_state.get("error"):
            debug["pipeline_errors"]["diagram_research"] = research_state.get("error")

        siglip_cache_payload = {
            "schema": "post_research_siglip_v1",
            "diagram_prompts": list(diagram_prompts),
            "c2_reports_by_prompt": c2_result_holder.get("prompts") or {},
            "research_prompts": list(research_state.get("prompts") or []),
        }
        siglip_cached_data, siglip_cache_key = self._load_cached_stage_data(
            stage_name="post_research_siglip",
            payload=siglip_cache_payload,
        )
        if not isinstance(siglip_cached_data, dict):
            old_rerank, old_rerank_path = self._load_latest_stage_data("diagram_research_rerank_finalize")
            old_ranking, old_ranking_path = self._load_latest_stage_data("diagram_siglip_ranking")
            old_canonical, old_canonical_path = self._load_latest_stage_data("c2_component_canonical_images")
            old_ranked_by_prompt = old_ranking.get("by_prompt") if isinstance(old_ranking, dict) else None
            if isinstance(old_rerank, dict) and isinstance(old_ranked_by_prompt, dict):
                requested = {normalize_ws(prompt) for prompt in diagram_prompts if normalize_ws(prompt)}
                available = {str(prompt) for prompt in old_ranked_by_prompt.keys()}
                if requested and requested.issubset(available):
                    siglip_cached_data = {
                        "rerank_paths_by_prompt": old_rerank,
                        "c2_canonical": old_canonical if isinstance(old_canonical, dict) else {"by_prompt": {}, "errors": {}},
                        "ranked_diagram_candidates": old_ranked_by_prompt,
                    }
                    siglip_cache_key = "existing-stage-cache"
                    self._dbg(
                        "Post-research SigLIP reconstructed from existing caches",
                        data={
                            "rerank": old_rerank_path,
                            "ranking": old_ranking_path,
                            "canonical": old_canonical_path,
                        },
                    )
        comfy_cache_payload = {
            "schema": "post_research_comfy_v1",
            "prompt_meta": {
                prompt: meta
                for prompt, meta in (prompt_meta or {}).items()
                if int((meta or {}).get("diagram", 0) or 0) == 0
            },
        }
        comfy_cached_data, comfy_cache_key = self._load_cached_stage_data(
            stage_name="post_research_comfy",
            payload=comfy_cache_payload,
        )
        siglip_cache_hit = isinstance(siglip_cached_data, dict)
        comfy_cache_hit = isinstance(comfy_cached_data, dict)
        debug["stage_cache"] = {
            "siglip": {"hit": siglip_cache_hit, "key": siglip_cache_key[:12]},
            "comfy": {"hit": comfy_cache_hit, "key": comfy_cache_key[:12]},
        }

        qwen_slept_for_heavy_stage = False
        if not (siglip_cache_hit and comfy_cache_hit):
            qwen_sleep_error = self._sleep_qwen_text_worker()
            qwen_slept_for_heavy_stage = True
            if qwen_sleep_error:
                debug["pipeline_errors"]["qwen_sleep_before_heavy_stages"] = qwen_sleep_error
            if DEFAULT_PRELOAD_SIGLIP_TEXT:
                siglip_text_unload_error = self._unload_siglip_text()
                if siglip_text_unload_error:
                    debug["pipeline_errors"]["siglip_text_unload_before_heavy_stages"] = siglip_text_unload_error

        siglip_bundle = None
        siglip_load_error = None

        rerank_paths_by_prompt: Dict[str, List[str]] = {}
        c2_canonical: Dict[str, Any] = {"by_prompt": {}, "errors": {}}
        ranked_diagram_candidates: Dict[str, List[Dict[str, Any]]] = {}
        if siglip_cache_hit:
            rerank_paths_by_prompt = {
                str(key): [str(x) for x in (value or []) if str(x).strip()]
                for key, value in (siglip_cached_data.get("rerank_paths_by_prompt") or {}).items()
                if isinstance(key, str)
            }
            cached_canonical = siglip_cached_data.get("c2_canonical")
            if isinstance(cached_canonical, dict):
                c2_canonical = cached_canonical
            ranked_diagram_candidates = {
                str(key): list(value or [])
                for key, value in (siglip_cached_data.get("ranked_diagram_candidates") or {}).items()
                if isinstance(key, str)
            }
        else:
            siglip_bundle, siglip_load_error = self._load_full_siglip()
            if siglip_load_error:
                debug["pipeline_errors"]["siglip_load"] = siglip_load_error
        if siglip_bundle is not None:
            try:
                rerank_paths_by_prompt = self._finalize_research_rerank()
            except Exception as exc:
                debug["pipeline_errors"]["research_finalize_rerank"] = f"{type(exc).__name__}: {exc}"
            try:
                c2_canonical = self._rank_c2_component_candidate_images(
                    c2_reports_by_prompt=(c2_result_holder.get("prompts") or {}),
                    siglip_bundle=siglip_bundle,
                )
            except Exception as exc:
                debug["pipeline_errors"]["c2_component_canonical_images"] = f"{type(exc).__name__}: {exc}"
                c2_canonical = {"by_prompt": {}, "errors": {"rank": f"{type(exc).__name__}: {exc}"}}
            try:
                ranked = self._rank_diagram_candidate_paths_with_siglip(
                    diagram_prompts=diagram_prompts,
                    rerank_paths_by_prompt=rerank_paths_by_prompt,
                    siglip_bundle=siglip_bundle,
                )
                ranked_diagram_candidates = dict(ranked.get("by_prompt") or {})
                debug["diagram_siglip_rule_ranking"] = ranked_diagram_candidates
            except Exception as exc:
                debug["pipeline_errors"]["diagram_siglip_rule_ranking"] = f"{type(exc).__name__}: {exc}"
        else:
            debug["diagram_siglip_rule_ranking"] = {}

        if not siglip_cache_hit and (
            rerank_paths_by_prompt
            or ranked_diagram_candidates
            or (isinstance(c2_canonical, dict) and (c2_canonical.get("by_prompt") or c2_canonical.get("errors")))
        ):
            self._store_cached_stage_data(
                stage_name="post_research_siglip",
                payload=siglip_cache_payload,
                data={
                    "rerank_paths_by_prompt": rerank_paths_by_prompt,
                    "c2_canonical": c2_canonical,
                    "ranked_diagram_candidates": ranked_diagram_candidates,
                },
            )

        debug["diagram_rerank_paths_by_prompt"] = rerank_paths_by_prompt
        debug["c2_component_canonical_images"] = c2_canonical.get("by_prompt", {})
        if c2_canonical.get("errors"):
            debug["pipeline_errors"]["c2_component_canonical_images"] = c2_canonical.get("errors")

        if siglip_bundle is not None:
            siglip_unload_error = self._unload_full_siglip()
            if siglip_unload_error:
                debug["pipeline_errors"]["siglip_unload_before_comfy"] = siglip_unload_error

        comfy_result_holder: Dict[str, Any] = {
            "generated": {},
            "processed_ids_by_prompt": {},
            "saved_paths_by_prompt": {},
            "metadata_enrichment": None,
            "error": None,
        }

        def _run_comfy() -> None:
            nonlocal comfy_result_holder
            try:
                comfy_result_holder = self._generate_non_diagram_images(prompt_meta=prompt_meta)
            except Exception as exc:
                comfy_result_holder = {
                    "generated": {},
                    "processed_ids_by_prompt": {},
                    "saved_paths_by_prompt": {},
                    "metadata_enrichment": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        comfy_thread: Optional[threading.Thread] = None
        if comfy_cache_hit:
            comfy_result_holder = dict(comfy_cached_data or {})
        else:
            comfy_thread = threading.Thread(target=_run_comfy, name="newtimeline_comfy_non_diagrams", daemon=False)
            comfy_thread.start()
        diagram_selection = self._select_diagram_processed_ids_from_ranked_candidates(
            prompt_meta=prompt_meta,
            ranked_candidates_by_prompt=ranked_diagram_candidates,
            c2_reports_by_prompt=(c2_result_holder.get("prompts") or {}),
        )
        if comfy_thread is not None:
            comfy_thread.join()

        if not comfy_cache_hit:
            self._store_cached_stage_data(
                stage_name="post_research_comfy",
                payload=comfy_cache_payload,
                data=comfy_result_holder,
            )
            comfy_unload_error = self._unload_comfy_models()
            if comfy_unload_error:
                debug["pipeline_errors"]["comfy_unload_after_generation"] = comfy_unload_error
        if qwen_slept_for_heavy_stage:
            qwen_wake_thread, qwen_wake_state = self._start_qwen_text_loader()
            debug["qwen_wake_after_comfy"] = {
                "started": bool(qwen_wake_thread is not None or qwen_wake_state.get("started")),
                "state": dict(qwen_wake_state or {}),
            }
        else:
            debug["qwen_wake_after_comfy"] = {
                "started": False,
                "state": {"ready": True, "note": "qwen_kept_loaded_due_to_heavy_stage_cache_hits"},
            }

        if comfy_result_holder.get("error"):
            debug["pipeline_errors"]["comfy_generation"] = comfy_result_holder.get("error")
        debug["comfy_generated"] = {
            prompt: list((info or {}).get("saved_paths") or [])
            for prompt, info in (comfy_result_holder.get("generated") or {}).items()
            if isinstance(info, dict)
        }
        debug["comfy_processed_ids_by_prompt"] = comfy_result_holder.get("processed_ids_by_prompt") or {}
        debug["comfy_metadata_enrichment"] = comfy_result_holder.get("metadata_enrichment")
        debug["diagram_ocr_selection"] = diagram_selection.get("ocr_debug") or {}

        selected_ids_by_prompt: Dict[str, List[str]] = {}
        for prompt, meta in prompt_meta.items():
            if int((meta or {}).get("diagram", 0) or 0) == 1:
                selected_ids_by_prompt[prompt] = list((diagram_selection.get("selected_ids_by_prompt") or {}).get(prompt) or [])
            else:
                selected_ids_by_prompt[prompt] = list((comfy_result_holder.get("processed_ids_by_prompt") or {}).get(prompt) or [])

        assets: Dict[str, ImageAsset] = {}
        for prompt, meta in prompt_meta.items():
            assets[prompt] = ImageAsset(
                prompt=prompt,
                diagram=int((meta or {}).get("diagram", 0) or 0),
                processed_ids=list(selected_ids_by_prompt.get(prompt) or []),
                refined_labels_file=None,
                bbox_px=None,
                objects=None,
            )

        c2_apply = self._apply_c2_diagram_components_to_assets(
            assets=assets,
            prompt_meta=prompt_meta,
            c2_reports_by_prompt=(c2_result_holder.get("prompts") or {}),
        )
        debug["c2_artifacts_by_prompt"] = c2_apply.get("artifacts_by_prompt", {})
        if c2_apply.get("errors"):
            debug["pipeline_errors"]["c2_apply"] = c2_apply.get("errors")

        link_maps = self._write_c2_link_maps(
            prompt_meta=prompt_meta,
            c2_reports_by_prompt=(c2_result_holder.get("prompts") or {}),
            selected_ids_by_prompt=selected_ids_by_prompt,
            candidate_processed_ids_by_prompt=(diagram_selection.get("candidate_processed_ids_by_prompt") or {}),
            selected_source_path_by_prompt=(diagram_selection.get("selected_source_path_by_prompt") or {}),
        )
        debug["c2_link_maps"] = link_maps

        accepted_processed_ids_by_base_context = {
            prompt: [pid for pid in (ids or []) if str(pid).strip()]
            for prompt, ids in selected_ids_by_prompt.items()
            if any(str(pid).strip() for pid in (ids or []))
        }
        for asset in assets.values():
            self._compute_bbox_px_for_asset(asset)

        import ImagePipeline
        from shared_models import get_minilm

        pipeline_handle = ImagePipeline.start_pipeline_background(
            workers=ImagePipeline.PipelineWorkers(
                siglip_bundle=None,
                minilm_bundle=get_minilm(),
                precomputed_clip_embeddings_by_processed_id={},
            ),
            model_id="",
            gpu_index=self.gpu_index,
            allowed_base_contexts=list(accepted_processed_ids_by_base_context.keys()),
            diagram_base_contexts=[prompt for prompt, meta in prompt_meta.items() if int((meta or {}).get("diagram", 0) or 0) == 1],
            diagram_mode_by_base_context={prompt: int((meta or {}).get("diagram", 0) or 0) for prompt, meta in prompt_meta.items()},
            diagram_required_objects_by_base_context={
                prompt: list((meta or {}).get("diagram_required_objects") or [])
                for prompt, meta in prompt_meta.items()
            },
            accepted_processed_ids_by_base_context=accepted_processed_ids_by_base_context,
            pause_after_line_descriptors=True,
        )

        debug["selected_ids_by_prompt"] = selected_ids_by_prompt
        debug["accepted_processed_ids_by_base_context"] = accepted_processed_ids_by_base_context
        debug["pipeline_started"] = True

        self._dbg("Qwen speech sync/compression starting after research/C2/SigLIP")
        qwen_sync_debug = self._ensure_first_module_qwen_sync(first_module_result)
        chapters_out = list(first_module_result.get("chapters") or chapters_out)
        debug["qwen_speech_sync"] = {
            key: value
            for key, value in (qwen_sync_debug or {}).items()
            if key != "responses"
        }
        self._dbg(
            "Qwen speech sync/compression done",
            data={
                "requests": int((qwen_sync_debug or {}).get("request_count", 0) or 0),
                "synced_objects": len(first_module_result.get("synced_objects") or []),
                "cache_hits": int((qwen_sync_debug or {}).get("cache_hits", 0) or 0),
                "cache_misses": int((qwen_sync_debug or {}).get("cache_misses", 0) or 0),
            },
        )

        self._dbg("Preparing space planner objects")
        prepared_objects_bundle = self._prepare_space_planner_objects(
            first_module_result=first_module_result,
            post_research_result={
                "prompt_meta": prompt_meta,
                "assets_by_prompt": assets,
                "debug": debug,
            },
        )
        self._dbg(
            "Space planner objects ready",
            data={"objects": len(prepared_objects_bundle.get("objects") or []), "saved_path": prepared_objects_bundle.get("saved_path")},
        )
        first_module_result["synced_objects"] = list(prepared_objects_bundle.get("objects") or [])
        debug["space_planner_objects_ready_path"] = prepared_objects_bundle.get("saved_path")
        debug["space_planner_object_verification"] = prepared_objects_bundle.get("verification")
        self._dbg("Space planner batch starting")
        space_planner_result = self.run_space_planner_batch(
            chapters_out=chapters_out,
            prepared_objects=list(prepared_objects_bundle.get("objects") or []),
        )
        self._dbg(
            "Space planner batch done",
            data={
                "requests": int(space_planner_result.get("request_count", 0) or 0),
                "cache_hits": int(space_planner_result.get("cache_hits", 0) or 0),
                "cache_misses": int(space_planner_result.get("cache_misses", 0) or 0),
            },
        )
        debug["space_planner"] = {
            "request_count": int(space_planner_result.get("request_count", 0) or 0),
            "chunks_ready_path": space_planner_result.get("chunks_ready_path"),
            "responses_path": space_planner_result.get("responses_path"),
            "cache_hits": int(space_planner_result.get("cache_hits", 0) or 0),
            "cache_misses": int(space_planner_result.get("cache_misses", 0) or 0),
        }
        self._dbg("Diagram action planner batch starting")
        diagram_action_result = self.run_diagram_action_planner_batch(
            chapters_out=chapters_out,
            prepared_objects=list(prepared_objects_bundle.get("objects") or []),
        )
        self._dbg(
            "Diagram action planner batch done",
            data={
                "requests": int(diagram_action_result.get("request_count", 0) or 0),
                "cache_hits": int(diagram_action_result.get("cache_hits", 0) or 0),
                "cache_misses": int(diagram_action_result.get("cache_misses", 0) or 0),
            },
        )
        debug["diagram_action_planner"] = {
            "request_count": int(diagram_action_result.get("request_count", 0) or 0),
            "responses_path": diagram_action_result.get("responses_path"),
            "cache_hits": int(diagram_action_result.get("cache_hits", 0) or 0),
            "cache_misses": int(diagram_action_result.get("cache_misses", 0) or 0),
        }
        self._dbg("Full speech action sync + combined lesson actions starting")
        combined_actions = self._build_combined_lesson_action_payload(
            chapters_out=chapters_out,
            space_planner_result=space_planner_result,
            diagram_action_result=diagram_action_result,
        )
        self._dbg(
            "Full speech action sync + combined lesson actions done",
            data={
                "lesson_actions": len(combined_actions.get("lesson_actions") or []),
                "sync_cache_hits": int(((combined_actions.get("full_chunk_speech_action_sync") or {}).get("cache_hits", 0) or 0)),
                "sync_cache_misses": int(((combined_actions.get("full_chunk_speech_action_sync") or {}).get("cache_misses", 0) or 0)),
            },
        )
        debug["combined_lesson_actions"] = {
            "saved_path": combined_actions.get("saved_path"),
            "lesson_action_count": len(combined_actions.get("lesson_actions") or []),
            "space_action_count": len(combined_actions.get("space_actions") or []),
            "diagram_action_count": len(combined_actions.get("diagram_actions") or []),
            "silence_action_count": len(combined_actions.get("silence_actions") or []),
            "full_chunk_speech_action_sync_path": ((combined_actions.get("full_chunk_speech_action_sync") or {}).get("saved_path")),
            "full_chunk_speech_action_sync_raw_io_path": ((combined_actions.get("full_chunk_speech_action_sync") or {}).get("raw_io_path")),
            "full_chunk_speech_action_sync_cache_hits": int(((combined_actions.get("full_chunk_speech_action_sync") or {}).get("cache_hits", 0) or 0)),
            "full_chunk_speech_action_sync_cache_misses": int(((combined_actions.get("full_chunk_speech_action_sync") or {}).get("cache_misses", 0) or 0)),
        }
        self._dbg("Diagram stroke meaning batch starting")
        stroke_meaning_result = self.run_diagram_stroke_meaning_filter_batch(
            prompt_meta=prompt_meta,
            selected_ids_by_prompt=selected_ids_by_prompt,
            pipeline_handle=pipeline_handle,
        )
        self._dbg(
            "Diagram stroke meaning batch done",
            data={
                "requests": int(stroke_meaning_result.get("request_count", 0) or 0),
                "cache_hits": int(stroke_meaning_result.get("cache_hits", 0) or 0),
                "cache_misses": int(stroke_meaning_result.get("cache_misses", 0) or 0),
                "error": stroke_meaning_result.get("error"),
            },
        )
        debug["diagram_stroke_meaning"] = {
            "enabled": bool(stroke_meaning_result.get("enabled", True)),
            "request_count": int(stroke_meaning_result.get("request_count", 0) or 0),
            "saved_path": stroke_meaning_result.get("saved_path"),
            "raw_io_path": stroke_meaning_result.get("raw_io_path"),
            "cache_hits": int(stroke_meaning_result.get("cache_hits", 0) or 0),
            "cache_misses": int(stroke_meaning_result.get("cache_misses", 0) or 0),
            "errors": dict(stroke_meaning_result.get("errors") or {}),
            "error": stroke_meaning_result.get("error"),
        }
        if stroke_meaning_result.get("error") or stroke_meaning_result.get("errors"):
            debug["pipeline_errors"]["diagram_stroke_meaning"] = stroke_meaning_result.get("error") or stroke_meaning_result.get("errors")
        self._dbg("Qwen sleep before SAM/DINO candidates starting")
        qwen_sleep_error = self._sleep_qwen_text_worker()
        self._dbg("Qwen sleep before SAM/DINO candidates done", data={"error": qwen_sleep_error})
        if qwen_sleep_error:
            debug["pipeline_errors"]["qwen_sleep_before_sam_dino_candidates"] = qwen_sleep_error

        diagram_selected_ids_by_prompt = {
            prompt: list(ids or [])
            for prompt, ids in selected_ids_by_prompt.items()
            if int((prompt_meta.get(prompt) or {}).get("diagram", 0) or 0) == 1
        }
        diagram_object_candidates = self._prepare_diagram_object_candidates(
            selected_ids_by_prompt=diagram_selected_ids_by_prompt,
            pipeline_handle=pipeline_handle,
        )
        debug["diagram_object_candidates"] = {
            "saved_path": diagram_object_candidates.get("saved_path"),
            "processed_ids": list(diagram_object_candidates.get("processed_ids") or []),
            "candidate_count": len(diagram_object_candidates.get("flat_candidates") or []),
            "errors": dict(diagram_object_candidates.get("errors") or {}),
        }
        if diagram_object_candidates.get("errors"):
            debug["pipeline_errors"]["diagram_object_candidates"] = diagram_object_candidates.get("errors")

        self._dbg("Diagram SAM/DINO unload before Qwen vision descriptions starting")
        sam_dino_unload_error = self._unload_diagram_mask_cluster_models()
        self._dbg("Diagram SAM/DINO unload before Qwen vision descriptions done", data={"error": sam_dino_unload_error})
        if sam_dino_unload_error:
            debug["pipeline_errors"]["diagram_sam_dino_unload_before_qwen_vision"] = sam_dino_unload_error

        self._dbg("Diagram visual description batch starting")
        diagram_visual_descriptions = self.run_diagram_visual_description_batch(
            diagram_object_candidates=diagram_object_candidates,
            prompt_meta=prompt_meta,
            c2_reports_by_prompt=(c2_result_holder.get("prompts") or {}),
        )
        self._dbg(
            "Diagram visual description batch done",
            data={
                "requests": int(diagram_visual_descriptions.get("request_count", 0) or 0),
                "cache_hits": int(diagram_visual_descriptions.get("cache_hits", 0) or 0),
                "cache_misses": int(diagram_visual_descriptions.get("cache_misses", 0) or 0),
                "errors": len(diagram_visual_descriptions.get("errors") or {}),
            },
        )
        debug["diagram_visual_descriptions"] = {
            "request_count": int(diagram_visual_descriptions.get("request_count", 0) or 0),
            "saved_path": diagram_visual_descriptions.get("saved_path"),
            "cache_hits": int(diagram_visual_descriptions.get("cache_hits", 0) or 0),
            "cache_misses": int(diagram_visual_descriptions.get("cache_misses", 0) or 0),
            "errors": dict(diagram_visual_descriptions.get("errors") or {}),
        }
        if diagram_visual_descriptions.get("errors"):
            debug["pipeline_errors"]["diagram_visual_descriptions"] = diagram_visual_descriptions.get("errors")

        self._dbg("Diagram component stroke match batch starting")
        diagram_component_stroke_matches = self.run_diagram_component_stroke_match_batch(
            prompt_meta=prompt_meta,
            selected_ids_by_prompt=selected_ids_by_prompt,
            diagram_object_candidates=diagram_object_candidates,
            diagram_visual_descriptions=diagram_visual_descriptions,
            stroke_meaning_result=stroke_meaning_result,
        )
        self._dbg(
            "Diagram component stroke match batch done",
            data={
                "requests": int(diagram_component_stroke_matches.get("request_count", 0) or 0),
                "cache_hits": int(diagram_component_stroke_matches.get("cache_hits", 0) or 0),
                "cache_misses": int(diagram_component_stroke_matches.get("cache_misses", 0) or 0),
                "errors": len(diagram_component_stroke_matches.get("errors") or {}),
            },
        )
        debug["diagram_component_stroke_matches"] = {
            "request_count": int(diagram_component_stroke_matches.get("request_count", 0) or 0),
            "saved_path": diagram_component_stroke_matches.get("saved_path"),
            "raw_io_path": diagram_component_stroke_matches.get("raw_io_path"),
            "cache_hits": int(diagram_component_stroke_matches.get("cache_hits", 0) or 0),
            "cache_misses": int(diagram_component_stroke_matches.get("cache_misses", 0) or 0),
            "errors": dict(diagram_component_stroke_matches.get("errors") or {}),
        }
        if diagram_component_stroke_matches.get("errors"):
            debug["pipeline_errors"]["diagram_component_stroke_matches"] = diagram_component_stroke_matches.get("errors")
        return {
            "topic": normalize_ws(topic),
            "prompt_meta": prompt_meta,
            "assets_by_prompt": assets,
            "debug": debug,
            "pipeline_handle": pipeline_handle,
            "space_planner_objects": prepared_objects_bundle.get("objects") or [],
            "space_planner_object_verification": prepared_objects_bundle.get("verification") or {},
            "space_planner": space_planner_result,
            "diagram_action_planner": diagram_action_result,
            "diagram_stroke_meaning": stroke_meaning_result,
            "diagram_object_candidates": diagram_object_candidates,
            "diagram_visual_descriptions": diagram_visual_descriptions,
            "diagram_component_stroke_matches": diagram_component_stroke_matches,
            "combined_lesson_actions": combined_actions,
        }

    def run_full_timeline(self, topic: str) -> Dict[str, Any]:
        first_module_result = self.run_first_module(topic)
        post_research_result = self.run_post_research_pipeline(topic=topic, first_module_result=first_module_result)
        return {
            "topic": normalize_ws(topic),
            "first_module": first_module_result,
            "post_research_pipeline": post_research_result,
        }


def _json_safe_for_main(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_for_main(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_for_main(item) for item in value]
    return repr(value)


def _build_main_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NEWtimeline.py end-to-end for one topic.")
    parser.add_argument("topic", help="Lesson topic / prompt to run through the pipeline.")
    parser.add_argument("--cache", action="store_true", help="Use NEWtimeline cached stages where available.")
    parser.add_argument("--out", default="PipelineOutputs/newtimeline_run_result.json")
    parser.add_argument("--debug-out-dir", default="PipelineOutputs")
    return parser


def main() -> None:
    args = _build_main_arg_parser().parse_args()
    topic = normalize_ws(args.topic)
    if not topic:
        raise SystemExit("topic cannot be empty")
    runner = NewTimelineFirstModule(
        debug_out_dir=str(args.debug_out_dir),
        gpt_cache=bool(args.cache),
        debug_print=True,
    )
    result = runner.run_full_timeline(topic)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "schema": "newtimeline_direct_main_run_v1",
                "topic": topic,
                "cache_enabled": bool(args.cache),
                "written_at_unix": time.time(),
                "result": _json_safe_for_main(result),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[NEWtimeline] complete. Result written to {out_path}")


if __name__ == "__main__":
    main()

