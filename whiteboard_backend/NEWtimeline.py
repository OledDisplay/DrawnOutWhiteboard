from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from LLMstuff.input_builder_responce_parser import (
    build_chapter_speech_request,
    build_image_request_request,
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
    parse_qwen_step_output,
    parse_text_request_output,
)
from LLMstuff.qwen_vllm_server import QwenServerClient


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


DEFAULT_MODEL_CANDIDATES = ["gpt-5-mini"]
DEFAULT_QWEN_BASE_URL = str(os.getenv("QWEN_VLLM_SERVER_URL", "http://127.0.0.1:8009") or "http://127.0.0.1:8009").strip()
DEFAULT_QWEN_MODEL = str(os.getenv("QWEN_TEXT_MODEL_ID", "cyankiwi/Qwen3.5-4B-AWQ-4bit") or "cyankiwi/Qwen3.5-4B-AWQ-4bit").strip()


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
        }
        self._diagram_research_state: Optional[Dict[str, Any]] = None
        self._diagram_research_thread: Optional[threading.Thread] = None
        self._dimensions_config_cache: Optional[Dict[str, Any]] = None
        self._font_metrics_cache: Optional[Dict[str, float]] = None
        self._glyph_metrics_cache: Dict[int, Dict[str, float]] = {}

    def _dbg(self, message: str, *, data: Any = None) -> None:
        if not self.debug_print:
            return
        if data is None:
            print(f"[NEWtimeline][DBG] {message}")
            return
        try:
            text = json.dumps(data, ensure_ascii=False)
            if len(text) > 1800:
                text = text[:1800] + " ...<truncated>"
        except Exception:
            text = "<unserializable>"
        print(f"[NEWtimeline][DBG] {message} | {text}")

    def _write_json_file(self, path: Path, payload: Any) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

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

        thread = threading.Thread(target=_load, name="newtimeline_preload_minilm", daemon=False)
        thread.start()
        return thread, state

    def _start_siglip_text_loader(self) -> Tuple[threading.Thread, Dict[str, Any]]:
        state: Dict[str, Any] = {"ok": None, "error": None}

        def _load() -> None:
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

        thread = threading.Thread(target=_load, name="newtimeline_preload_siglip_text", daemon=False)
        thread.start()
        return thread, state

    def _start_qwen_text_loader(self) -> Tuple[Optional[threading.Thread], Dict[str, Any]]:
        with self._qwen_load_lock:
            if self._qwen_load_thread is not None:
                return self._qwen_load_thread, self._qwen_load_state

            self._qwen_load_state = {
                "started": True,
                "ready": False,
                "error": None,
                "response": None,
            }

            def _load() -> None:
                try:
                    response = self.qwen_client.load_text(
                        model=self.qwen_model,
                        warmup=True,
                        gpu_memory_utilization=float(os.getenv("QWEN_SERVER_GPU_UTIL", "0.98") or 0.98),
                        max_model_len=int(os.getenv("QWEN_SERVER_MAX_MODEL_LEN", "32768") or 32768),
                    )
                    self._qwen_load_state["response"] = response
                    self._qwen_load_state["ready"] = bool(response.get("ok", False))
                    if not self._qwen_load_state["ready"]:
                        self._qwen_load_state["error"] = json.dumps(response, ensure_ascii=False)
                except Exception as exc:
                    self._qwen_load_state["ready"] = False
                    self._qwen_load_state["error"] = f"{type(exc).__name__}: {exc}"

            self._qwen_load_thread = threading.Thread(target=_load, name="newtimeline_qwen_text_load", daemon=False)
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
            }

    def _ensure_qwen_text_worker(self) -> Dict[str, Any]:
        thread, state = self._start_qwen_text_loader()
        if thread is not None:
            thread.join()
        if state.get("ready"):
            return dict(state)
        try:
            response = self.qwen_client.load_text(
                model=self.qwen_model,
                warmup=True,
                gpu_memory_utilization=float(os.getenv("QWEN_SERVER_GPU_UTIL", "0.98") or 0.98),
                max_model_len=int(os.getenv("QWEN_SERVER_MAX_MODEL_LEN", "32768") or 32768),
            )
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
        return chapter_out

    def _get_legacy_group_helper(self):
        if self._legacy_group_helper is not None:
            return self._legacy_group_helper
        from timeline import LessonTimeline

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
                import ImageResearcher

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
        qwen_ready = self._ensure_qwen_text_worker()

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
                "qwen_ready": qwen_ready,
            }

        response = self.qwen_client.generate_text_batch(
            prompts,
            system_prompt=system_prompt,
            generation={
                "max_new_tokens": 1800,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.02,
            },
            use_tqdm=False,
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
            "qwen_ready": qwen_ready,
            "synced_objects": synced_objects,
        }

    def run_first_module(self, topic: str) -> Dict[str, Any]:
        run_started_at = time.time()
        preload_threads: List[Tuple[str, threading.Thread, Dict[str, Any]]] = []

        minilm_thread, minilm_state = self._start_minilm_loader()
        preload_threads.append(("minilm", minilm_thread, minilm_state))
        siglip_thread, siglip_state = self._start_siglip_text_loader()
        preload_threads.append(("siglip_text", siglip_thread, siglip_state))
        qwen_thread, qwen_state = self._start_qwen_text_loader()
        if qwen_thread is not None:
            preload_threads.append(("qwen_text", qwen_thread, qwen_state))

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

        preload_status: Dict[str, Any] = {}
        for label, thread, state in preload_threads:
            thread.join()
            preload_status[label] = dict(state)

        repeat_filter_debug = self.apply_repeating_diagram_filter(topic=topic, chapters_out=chapters_out)
        research_state = self.start_diagram_research(topic=topic, chapters_out=chapters_out)
        qwen_debug = self.run_qwen_step_batch(chapters_out=chapters_out)

        return {
            "topic": normalize_ws(topic),
            "started_at_unix": run_started_at,
            "finished_at_unix": time.time(),
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
                key: value
                for key, value in qwen_debug.items()
                if key != "responses"
            },
            "synced_objects": list(qwen_debug.get("synced_objects") or []),
        }

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
            thread.join()
            state["finished"] = True
        return state

    def _run_c2_bundle(
        self,
        *,
        c2_bundle_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not c2_bundle_rows:
            return {"ok": True, "prompts": {}, "errors": {}, "count": 0}
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

    def _unload_qwen_server(self) -> Optional[str]:
        try:
            self.qwen_client.unload()
            self._reset_qwen_text_loader_state()
            return None
        except Exception as exc:
            self._reset_qwen_text_loader_state()
            return f"{type(exc).__name__}: {exc}"

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
        import ImageResearcher

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

        qwen_ready = self._ensure_qwen_text_worker()
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

        response = self.qwen_client.generate_text_batch(
            prompts,
            system_prompt=system_prompt,
            generation={
                "max_new_tokens": int(os.getenv("QWEN_SPACE_PLANNER_MAX_NEW_TOKENS", "3200") or 3200),
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.02,
            },
            use_tqdm=False,
        )
        responses = response.get("responses")
        if not isinstance(responses, list) or len(responses) != len(prompts):
            raise RuntimeError("Qwen space planner response count does not match request count")

        stage_dir = Path(self.debug_out_dir) / "space_planner_qwen"
        stage_dir.mkdir(parents=True, exist_ok=True)
        out_chunks: List[Dict[str, Any]] = []
        for index, (meta, prompt, chunk, raw_row) in enumerate(zip(metas, prompts, chunks, responses), start=1):
            raw_text = str((raw_row or {}).get("text", "") or "").strip()
            file_path = self._write_json_file(
                stage_dir / f"{index:05d}_chunk_{str(meta.get('chunk_id', '')).replace('.', '_')}.json",
                {
                    "request_index": index,
                    "chunk_id": meta.get("chunk_id"),
                    "chapter_id": meta.get("chapter_id"),
                    "prompt": prompt,
                    "prompt_payload": meta.get("prompt_payload"),
                    "raw_response": raw_text,
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
                    "stage_io_path": file_path,
                    "prompt_payload": meta.get("prompt_payload"),
                }
            )

        responses_path = self._write_json_file(
            Path(self.debug_out_dir) / "space_planner_qwen_raw.json",
            {
                "schema": "space_planner_qwen_raw_v1",
                "request_count": len(prompts),
                "chunks_ready_path": chunk_bundle.get("saved_path"),
                "chunks": out_chunks,
            },
        )
        return {
            "enabled": True,
            "request_count": len(prompts),
            "qwen_ready": qwen_ready,
            "chunks_ready_path": chunk_bundle.get("saved_path"),
            "responses_path": responses_path,
            "chunks": out_chunks,
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
        }

        c2_result_holder: Dict[str, Any] = {"ok": True, "prompts": {}, "errors": {}, "count": 0}
        c2_error: Optional[str] = None

        def _run_c2() -> None:
            nonlocal c2_result_holder, c2_error
            try:
                c2_result_holder = self._run_c2_bundle(c2_bundle_rows=c2_rows)
            except Exception as exc:
                c2_error = f"{type(exc).__name__}: {exc}"
                c2_result_holder = {"ok": False, "prompts": {}, "errors": {"bundle": c2_error}, "count": 0}

        c2_thread = threading.Thread(target=_run_c2, name="newtimeline_c2_bundle", daemon=False)
        c2_thread.start()
        research_state = self._await_diagram_research()
        c2_thread.join()

        debug["diagram_research"] = research_state
        debug["c2_bundle"] = {
            "ok": bool(c2_result_holder.get("ok", True)),
            "count": int(c2_result_holder.get("count", 0) or 0),
            "errors": c2_result_holder.get("errors", {}),
        }
        if c2_error:
            debug["pipeline_errors"]["c2_bundle"] = c2_error
        if research_state.get("error"):
            debug["pipeline_errors"]["diagram_research"] = research_state.get("error")

        qwen_unload_error = self._unload_qwen_server()
        if qwen_unload_error:
            debug["pipeline_errors"]["qwen_unload_before_siglip"] = qwen_unload_error
        siglip_text_unload_error = self._unload_siglip_text()
        if siglip_text_unload_error:
            debug["pipeline_errors"]["siglip_text_unload_before_siglip"] = siglip_text_unload_error

        siglip_bundle, siglip_load_error = self._load_full_siglip()
        if siglip_load_error:
            debug["pipeline_errors"]["siglip_load"] = siglip_load_error

        rerank_paths_by_prompt: Dict[str, List[str]] = {}
        c2_canonical: Dict[str, Any] = {"by_prompt": {}, "errors": {}}
        ranked_diagram_candidates: Dict[str, List[Dict[str, Any]]] = {}
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

        debug["diagram_rerank_paths_by_prompt"] = rerank_paths_by_prompt
        debug["c2_component_canonical_images"] = c2_canonical.get("by_prompt", {})
        if c2_canonical.get("errors"):
            debug["pipeline_errors"]["c2_component_canonical_images"] = c2_canonical.get("errors")

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

        comfy_thread = threading.Thread(target=_run_comfy, name="newtimeline_comfy_non_diagrams", daemon=False)
        comfy_thread.start()
        diagram_selection = self._select_diagram_processed_ids_from_ranked_candidates(
            prompt_meta=prompt_meta,
            ranked_candidates_by_prompt=ranked_diagram_candidates,
            c2_reports_by_prompt=(c2_result_holder.get("prompts") or {}),
        )
        comfy_thread.join()

        comfy_unload_error = self._unload_comfy_models()
        if comfy_unload_error:
            debug["pipeline_errors"]["comfy_unload_after_generation"] = comfy_unload_error
        qwen_reload_thread, qwen_reload_state = self._start_qwen_text_loader()
        debug["qwen_reload_after_comfy"] = {
            "started": bool(qwen_reload_thread is not None or qwen_reload_state.get("started")),
            "state": dict(qwen_reload_state or {}),
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
        )

        debug["selected_ids_by_prompt"] = selected_ids_by_prompt
        debug["accepted_processed_ids_by_base_context"] = accepted_processed_ids_by_base_context
        debug["pipeline_started"] = True
        prepared_objects_bundle = self._prepare_space_planner_objects(
            first_module_result=first_module_result,
            post_research_result={
                "prompt_meta": prompt_meta,
                "assets_by_prompt": assets,
                "debug": debug,
            },
        )
        first_module_result["synced_objects"] = list(prepared_objects_bundle.get("objects") or [])
        debug["space_planner_objects_ready_path"] = prepared_objects_bundle.get("saved_path")
        debug["space_planner_object_verification"] = prepared_objects_bundle.get("verification")
        space_planner_result = self.run_space_planner_batch(
            chapters_out=chapters_out,
            prepared_objects=list(prepared_objects_bundle.get("objects") or []),
        )
        debug["space_planner"] = {
            "request_count": int(space_planner_result.get("request_count", 0) or 0),
            "chunks_ready_path": space_planner_result.get("chunks_ready_path"),
            "responses_path": space_planner_result.get("responses_path"),
        }
        return {
            "topic": normalize_ws(topic),
            "prompt_meta": prompt_meta,
            "assets_by_prompt": assets,
            "debug": debug,
            "pipeline_handle": pipeline_handle,
            "space_planner_objects": prepared_objects_bundle.get("objects") or [],
            "space_planner_object_verification": prepared_objects_bundle.get("verification") or {},
            "space_planner": space_planner_result,
        }

    def run_full_timeline(self, topic: str) -> Dict[str, Any]:
        first_module_result = self.run_first_module(topic)
        post_research_result = self.run_post_research_pipeline(topic=topic, first_module_result=first_module_result)
        return {
            "topic": normalize_ws(topic),
            "first_module": first_module_result,
            "post_research_pipeline": post_research_result,
        }
