#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

try:
    from transformers import AutoProcessor, AutoTokenizer
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "transformers is required for chat templating and multimodal prompt building"
    ) from exc

try:
    from vllm import LLM, SamplingParams
    _VLLM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    LLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]
    _VLLM_IMPORT_ERROR = exc

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    import uvicorn
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "fastapi, pydantic and uvicorn are required for server mode. Install: pip install fastapi uvicorn pydantic"
    ) from exc


DEFAULT_MODEL_ID = "cyankiwi/Qwen3.5-4B-AWQ-4bit"
DEFAULT_MAX_MODEL_LEN = 20_000
DEFAULT_GPU_MEMORY_UTILIZATION = 0.80
DEFAULT_KV_CACHE_DTYPE = "auto"
DEFAULT_ATTENTION_CONFIG = {"flash_attn_version": 2}
DEFAULT_COMPILATION_CONFIG = 3
DEFAULT_BATCH_LONG_PREFILL_THRESHOLD = 4096
DEFAULT_MM_LIMIT_PER_PROMPT = {"image": 8}
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8009

LOG = logging.getLogger("qwen_vllm_server")


def _resolve_vision_processor_model(model_id: Any, explicit_processor: Any = None) -> str:
    """Return the repo/path to use for AutoProcessor assets.

    Some quantized vision repos ship model weights for vLLM but omit the
    `preprocessor_config.json` / image processor files required by
    `AutoProcessor.from_pretrained(...)`. In that case we must load processor
    assets from the matching base model repo instead of the quantized weights
    repo.
    """
    explicit = str(explicit_processor or "").strip()
    if explicit:
        return explicit

    requested = str(model_id or "").strip()
    lowered = requested.casefold()

    if lowered == "cyankiwi/internvl3_5-8b-awq-4bit":
        return "OpenGVLab/InternVL3_5-8B-HF"
    if lowered == "opengvlab/internvl3_5-8b":
        return "OpenGVLab/InternVL3_5-8B-HF"

    return requested


def _is_internvl_model(*values: Any) -> bool:
    return any("internvl" in str(value or "").casefold() for value in values)


def _sanitize_mm_processor_kwargs(
    *,
    model: Any,
    processor: Any = None,
    mm_processor_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[Optional[dict[str, Any]], list[str]]:
    """Return vLLM-safe multimodal processor kwargs.

    vLLM passes ``mm_processor_kwargs`` to every processor object it builds for a
    model. InternVL's image processor accepts some dynamic-patch knobs in
    certain HF code paths, but vLLM also constructs an ``InternVLVideoProcessor``
    during engine initialization. That video processor currently rejects
    ``max_dynamic_patch`` / ``min_dynamic_patch``, which makes the whole load fail
    before weights are usable. Strip those known-bad keys here so stale clients
    and old CLI commands cannot crash the worker after a long load attempt.
    """
    if not isinstance(mm_processor_kwargs, dict) or not mm_processor_kwargs:
        return None, []

    sanitized = dict(mm_processor_kwargs)
    warnings: list[str] = []
    if _is_internvl_model(model, processor):
        for key in ("max_dynamic_patch", "min_dynamic_patch"):
            if key in sanitized:
                removed = sanitized.pop(key)
                warnings.append(
                    f"removed unsupported InternVL mm_processor_kwargs.{key}={removed!r}; "
                    "vLLM forwards it to InternVLVideoProcessor, which rejects it"
                )

    return (sanitized or None), warnings


@dataclass(slots=True)
class WorkerConfig:
    model: str = DEFAULT_MODEL_ID
    tokenizer: Optional[str] = None
    processor: Optional[str] = None
    trust_remote_code: bool = True
    tensor_parallel_size: int = 1
    dtype: str = "half"
    quantization: Optional[str] = None
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    kv_cache_dtype: str = DEFAULT_KV_CACHE_DTYPE
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    scheduler_reserve_full_isl: bool = True
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: Optional[int] = None
    cpu_offload_gb: float = 0.0
    max_num_partial_prefills: int = 1
    max_long_partial_prefills: int = 1
    long_prefill_token_threshold: int = DEFAULT_BATCH_LONG_PREFILL_THRESHOLD
    compilation_config: int | dict[str, Any] = DEFAULT_COMPILATION_CONFIG
    attention_config: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_ATTENTION_CONFIG))
    language_model_only: bool = True
    limit_mm_per_prompt: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_MM_LIMIT_PER_PROMPT))
    mm_encoder_tp_mode: Optional[str] = None
    mm_processor_cache_type: Optional[str] = None
    mm_processor_cache_gb: Optional[int] = None
    mm_shm_cache_max_object_size_mb: Optional[int] = None
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    enable_speculative: bool = False
    num_speculative_tokens: int = 1
    default_enable_thinking: bool = False
    seed: int = 0
    disable_log_stats: bool = False
    enable_logging_iteration_details: bool = False
    kv_cache_metrics: bool = False
    cudagraph_metrics: bool = False
    hf_token: Optional[str] = None
    config_warnings: list[str] = field(default_factory=list)

    def for_vllm(self) -> dict[str, Any]:
        compilation_config = 0 if self.enforce_eager else self.compilation_config
        tokenizer_id = self.tokenizer or self.processor or self.model
        kwargs: dict[str, Any] = {
            "model": self.model,
            "tokenizer": tokenizer_id,
            "trust_remote_code": self.trust_remote_code,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "kv_cache_dtype": self.kv_cache_dtype,
            "enforce_eager": self.enforce_eager,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "scheduler_reserve_full_isl": self.scheduler_reserve_full_isl,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "cpu_offload_gb": self.cpu_offload_gb,
            "max_num_partial_prefills": self.max_num_partial_prefills,
            "max_long_partial_prefills": self.max_long_partial_prefills,
            "long_prefill_token_threshold": self.long_prefill_token_threshold,
            "compilation_config": compilation_config,
            "attention_config": self.attention_config,
            "language_model_only": self.language_model_only,
            "limit_mm_per_prompt": self.limit_mm_per_prompt,
            "seed": self.seed,
            "disable_log_stats": self.disable_log_stats,
            "enable_logging_iteration_details": self.enable_logging_iteration_details,
            "kv_cache_metrics": self.kv_cache_metrics,
            "cudagraph_metrics": self.cudagraph_metrics,
        }
        if self.quantization:
            kwargs["quantization"] = self.quantization
        if self.hf_token:
            kwargs["hf_token"] = self.hf_token
        if self.mm_encoder_tp_mode:
            kwargs["mm_encoder_tp_mode"] = self.mm_encoder_tp_mode
        if self.mm_processor_cache_type:
            kwargs["mm_processor_cache_type"] = self.mm_processor_cache_type
        if self.mm_processor_cache_gb is not None:
            kwargs["mm_processor_cache_gb"] = self.mm_processor_cache_gb
        if self.mm_shm_cache_max_object_size_mb is not None:
            kwargs["mm_shm_cache_max_object_size_mb"] = self.mm_shm_cache_max_object_size_mb
        if self.mm_processor_kwargs:
            mm_kwargs, warnings = _sanitize_mm_processor_kwargs(
                model=self.model,
                processor=self.processor,
                mm_processor_kwargs=self.mm_processor_kwargs,
            )
            for warning in warnings:
                if warning not in self.config_warnings:
                    self.config_warnings.append(warning)
                LOG.warning("Sanitized vLLM config: %s", warning)
            if mm_kwargs:
                kwargs["mm_processor_kwargs"] = mm_kwargs
        if self.enable_speculative:
            kwargs["speculative_config"] = {
                "method": "mtp",
                "num_speculative_tokens": int(self.num_speculative_tokens),
            }
        return kwargs


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    stop: Optional[list[str]] = None
    skip_special_tokens: bool = True

    def to_sampling_params(self) -> SamplingParams:
        if SamplingParams is None:
            raise RuntimeError(
                "vllm is required to build local SamplingParams. Use the Docker/vLLM server for HTTP-only clients."
            ) from _VLLM_IMPORT_ERROR
        kwargs: dict[str, Any] = {
            "max_tokens": int(self.max_new_tokens),
            "skip_special_tokens": bool(self.skip_special_tokens),
            "repetition_penalty": float(self.repetition_penalty),
        }
        temp = float(self.temperature)
        if temp <= 0.0:
            kwargs["temperature"] = 0.0
        else:
            kwargs["temperature"] = temp
            kwargs["top_p"] = float(self.top_p)
            if self.top_k is not None and int(self.top_k) >= 0:
                kwargs["top_k"] = int(self.top_k)
        if self.stop:
            kwargs["stop"] = list(self.stop)
        return SamplingParams(**kwargs)


@dataclass(slots=True)
class ResponseRecord:
    request_id: str
    text: str
    finish_reason: Optional[str]
    prompt_tokens: int
    cached_prompt_tokens: Optional[int]
    output_tokens: int
    total_tokens: int
    metrics: Optional[dict[str, Any]]


class QwenVLLMWorker:
    def __init__(self, *, config: WorkerConfig, mode: str) -> None:
        self.config = config
        self.mode = mode
        self.model_id = config.model
        self.hf_token = config.hf_token or _read_hf_token()
        self._tokenizer = None
        self._processor = None

        if self.mode not in {"text", "vision"}:
            raise ValueError(f"unsupported mode: {self.mode}")

        self._load_template_adapter()
        self.llm = self._build_engine()

    def _load_template_adapter(self) -> None:
        if self.mode == "text":
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer or self.model_id,
                trust_remote_code=self.config.trust_remote_code,
                token=self.hf_token,
            )
            if getattr(self._tokenizer, "pad_token_id", None) is None:
                eos = getattr(self._tokenizer, "eos_token_id", None)
                if eos is not None:
                    self._tokenizer.pad_token_id = eos
        else:
            processor_model = _resolve_vision_processor_model(self.model_id, self.config.processor)
            if processor_model != self.model_id:
                LOG.info(
                    "Vision processor repo differs from weights repo: model=%s processor=%s",
                    self.model_id,
                    processor_model,
                )
            self._processor = AutoProcessor.from_pretrained(
                processor_model,
                trust_remote_code=self.config.trust_remote_code,
                token=self.hf_token,
            )
            tok = getattr(self._processor, "tokenizer", None)
            if tok is not None and getattr(tok, "pad_token_id", None) is None:
                eos = getattr(tok, "eos_token_id", None)
                if eos is not None:
                    tok.pad_token_id = eos

    def _build_engine(self) -> LLM:
        if LLM is None:
            raise RuntimeError(
                "vllm is required for this worker server. Install it inside the WSL/Linux environment that will host the server."
            ) from _VLLM_IMPORT_ERROR
        kwargs = self.config.for_vllm()
        LOG.info("Creating vLLM engine with config: %s", json.dumps(_json_safe(kwargs), indent=2))
        return LLM(**kwargs)

    def _apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        adapter = self._processor if self._processor is not None else self._tokenizer
        if adapter is None:
            raise RuntimeError("tokenizer/processor is not initialized")

        common_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        thinking = bool(self.config.default_enable_thinking)

        try:
            return adapter.apply_chat_template(
                messages,
                enable_thinking=thinking,
                **common_kwargs,
            )
        except TypeError:
            try:
                return adapter.apply_chat_template(
                    messages,
                    chat_template_kwargs={"enable_thinking": thinking},
                    **common_kwargs,
                )
            except TypeError:
                return adapter.apply_chat_template(messages, **common_kwargs)

    def build_text_prompt(self, user_prompt: str, *, system_prompt: Optional[str] = None) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
        })
        return self._apply_chat_template(messages)

    def build_vision_prompt(
        self,
        user_prompt: str,
        *,
        num_images: int,
        system_prompt: Optional[str] = None,
    ) -> str:
        if _is_internvl_model(self.model_id, self.config.processor):
            # vLLM's InternVL multimodal processor searches the text prompt for
            # literal "<image>" placeholders and replaces each one with the
            # model-specific "<img><IMG_CONTEXT>...</img>" feature tokens.
            # Hugging Face's chat template eagerly turns image content into
            # "<IMG_CONTEXT>", which makes vLLM's replacement pass fail with:
            # "Failed to apply prompt replacement for mm_items['image'][0]".
            parts: list[str] = []
            if system_prompt:
                parts.append(f"<|im_start|>system\n{str(system_prompt).strip()}<|im_end|>")
            image_tokens = "\n".join("<image>" for _ in range(max(1, int(num_images or 1))))
            user_text = str(user_prompt or "").strip()
            parts.append(f"<|im_start|>user\n{image_tokens}\n{user_text}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })
        content: list[dict[str, Any]] = []
        for _ in range(int(num_images)):
            content.append({"type": "image"})
        content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": content})
        return self._apply_chat_template(messages)

    def generate_text_batch(
        self,
        prompts: Sequence[str],
        *,
        system_prompt: Optional[str] = None,
        generation: Optional[GenerationConfig] = None,
        use_tqdm: bool = False,
    ) -> list[ResponseRecord]:
        if self.mode not in {"text", "vision"}:
            raise RuntimeError("generate_text_batch() requires a loaded Qwen worker")
        if not prompts:
            return []

        sampling_params = (generation or GenerationConfig()).to_sampling_params()
        rendered_prompts = [
            self.build_text_prompt(prompt, system_prompt=system_prompt)
            for prompt in prompts
        ]
        outputs = self.llm.generate(rendered_prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
        return [self._to_response_record(output) for output in outputs]

    def generate_vision_batch(
        self,
        requests: Sequence[dict[str, Any]],
        *,
        system_prompt: Optional[str] = None,
        generation: Optional[GenerationConfig] = None,
        use_tqdm: bool = False,
    ) -> list[ResponseRecord]:
        if self.mode != "vision":
            raise RuntimeError("generate_vision_batch() requires a multimodal worker")
        if not requests:
            return []

        sampling_params = (generation or GenerationConfig()).to_sampling_params()
        payloads: list[dict[str, Any]] = []

        for item in requests:
            prompt = str(item["prompt"])
            images = _normalize_images(item.get("images"))
            if not images:
                raise ValueError("vision request is missing images")
            rendered_prompt = self.build_vision_prompt(
                prompt,
                num_images=len(images),
                system_prompt=system_prompt,
            )
            payloads.append(
                {
                    "prompt": rendered_prompt,
                    "multi_modal_data": {"image": images[0] if len(images) == 1 else images},
                }
            )

        outputs = self.llm.generate(payloads, sampling_params=sampling_params, use_tqdm=use_tqdm)
        return [self._to_response_record(output) for output in outputs]

    def warmup(self) -> None:
        if self.mode != "text":
            raise RuntimeError("vision warmup needs a sample image; call generate_vision() explicitly")
        self.generate_text_batch(
            ["Reply with the single token: warm."],
            generation=GenerationConfig(max_new_tokens=8, temperature=0.0),
            use_tqdm=False,
        )

    def close(self) -> None:
        if getattr(self, "llm", None) is not None:
            try:
                del self.llm
            except Exception:
                pass
        self._tokenizer = None
        self._processor = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _to_response_record(self, output: Any) -> ResponseRecord:
        completion = output.outputs[0] if getattr(output, "outputs", None) else None
        text = str(getattr(completion, "text", "") or "")
        finish_reason = getattr(completion, "finish_reason", None)
        prompt_tokens = len(getattr(output, "prompt_token_ids", []) or [])
        cached_prompt_tokens = getattr(output, "num_cached_tokens", None)
        output_token_ids = getattr(completion, "token_ids", None)
        output_tokens = len(output_token_ids or [])
        metrics = _json_safe(getattr(output, "metrics", None))
        return ResponseRecord(
            request_id=str(getattr(output, "request_id", "")),
            text=text,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=int(cached_prompt_tokens) if cached_prompt_tokens is not None else None,
            output_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens,
            metrics=metrics if isinstance(metrics, dict) else None,
        )


def create_text_worker(
    *,
    model: str = DEFAULT_MODEL_ID,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_num_batched_tokens: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    cpu_offload_gb: float = 0.0,
    enforce_eager: bool = False,
    debug_scheduler: bool = False,
    enable_speculative: bool = False,
    num_speculative_tokens: int = 1,
    quantization: Optional[str] = None,
    hf_token: Optional[str] = None,
    mm_processor_kwargs: Optional[dict[str, Any]] = None,
    disable_log_stats: bool = False,
) -> QwenVLLMWorker:
    cfg = WorkerConfig(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        cpu_offload_gb=float(cpu_offload_gb or 0.0),
        enforce_eager=enforce_eager,
        language_model_only=True,
        enable_prefix_caching=not enable_speculative,
        enable_speculative=enable_speculative,
        num_speculative_tokens=num_speculative_tokens,
        quantization=quantization,
        enable_logging_iteration_details=debug_scheduler,
        kv_cache_metrics=debug_scheduler,
        cudagraph_metrics=debug_scheduler,
        disable_log_stats=disable_log_stats,
        hf_token=hf_token,
    )
    return QwenVLLMWorker(config=cfg, mode="text")


def create_vision_worker(
    *,
    model: str = DEFAULT_MODEL_ID,
    processor: Optional[str] = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_num_batched_tokens: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    cpu_offload_gb: float = 0.0,
    enforce_eager: bool = False,
    debug_scheduler: bool = False,
    enable_speculative: bool = False,
    num_speculative_tokens: int = 1,
    quantization: Optional[str] = None,
    hf_token: Optional[str] = None,
    mm_processor_kwargs: Optional[dict[str, Any]] = None,
    limit_mm_per_prompt: Optional[dict[str, int]] = None,
    disable_log_stats: bool = True,
) -> QwenVLLMWorker:
    processor_model = _resolve_vision_processor_model(model, processor)
    safe_mm_processor_kwargs, config_warnings = _sanitize_mm_processor_kwargs(
        model=model,
        processor=processor_model,
        mm_processor_kwargs=mm_processor_kwargs,
    )
    for warning in config_warnings:
        LOG.warning("Sanitized requested vision config: %s", warning)
    cfg = WorkerConfig(
        model=model,
        processor=processor_model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        cpu_offload_gb=float(cpu_offload_gb or 0.0),
        enforce_eager=enforce_eager,
        language_model_only=False,
        limit_mm_per_prompt=dict(limit_mm_per_prompt or {"image": 1}),
        mm_encoder_tp_mode="data",
        mm_processor_cache_type="shm",
        mm_processor_cache_gb=4,
        mm_shm_cache_max_object_size_mb=256,
        mm_processor_kwargs=safe_mm_processor_kwargs,
        enable_prefix_caching=not enable_speculative,
        enable_speculative=enable_speculative,
        num_speculative_tokens=num_speculative_tokens,
        quantization=quantization,
        enable_logging_iteration_details=debug_scheduler,
        kv_cache_metrics=debug_scheduler,
        cudagraph_metrics=debug_scheduler,
        disable_log_stats=disable_log_stats,
        hf_token=hf_token,
        config_warnings=config_warnings,
    )
    return QwenVLLMWorker(config=cfg, mode="vision")


def create_qwen_worker(*, mode: str = "text", **kwargs: Any) -> QwenVLLMWorker:
    if mode == "text":
        return create_text_worker(**kwargs)
    if mode == "vision":
        return create_vision_worker(**kwargs)
    raise ValueError(f"unsupported mode: {mode}")


class LoadWorkerRequest(BaseModel):
    model: str = DEFAULT_MODEL_ID
    processor: Optional[str] = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: Optional[int] = None
    cpu_offload_gb: float = 0.0
    enforce_eager: bool = False
    debug_scheduler: bool = False
    enable_speculative: bool = False
    num_speculative_tokens: int = 1
    quantization: Optional[str] = None
    hf_token: Optional[str] = None
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    limit_mm_per_prompt: Optional[dict[str, int]] = None
    disable_log_stats: bool = False
    warmup: bool = False


class GenerationConfigPayload(BaseModel):
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    stop: Optional[list[str]] = None
    skip_special_tokens: bool = True

    def to_generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stop=self.stop,
            skip_special_tokens=self.skip_special_tokens,
        )


class TextGenerateRequest(BaseModel):
    prompts: list[str] = Field(default_factory=list)
    system_prompt: Optional[str] = None
    generation: GenerationConfigPayload = Field(default_factory=GenerationConfigPayload)
    use_tqdm: bool = False


class VisionItem(BaseModel):
    prompt: str
    images: list[str]


class VisionGenerateRequest(BaseModel):
    requests: list[VisionItem] = Field(default_factory=list)
    system_prompt: Optional[str] = None
    generation: GenerationConfigPayload = Field(default_factory=GenerationConfigPayload)
    use_tqdm: bool = False


class QwenWorkerManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._worker: Optional[QwenVLLMWorker] = None
        self._mode: Optional[str] = None
        self._loaded_at: Optional[float] = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "loaded": self._worker is not None,
                "mode": self._mode,
                "loaded_at_unix": self._loaded_at,
                "gpu": _gpu_snapshot(),
                "worker_config": _json_safe(asdict(self._worker.config)) if self._worker is not None else None,
            }

    def endpoint_map(self, host: str, port: int) -> dict[str, str]:
        base = f"http://{host}:{port}"
        return {
            "base": base,
            "status": f"{base}/status",
            "load_text": f"{base}/load/text",
            "load_vision": f"{base}/load/vision",
            "unload": f"{base}/unload",
            "generate_text": f"{base}/generate/text",
            "generate_vision": f"{base}/generate/vision",
            "health": f"{base}/health",
            "shutdown": f"{base}/shutdown",
        }

    def load(self, mode: str, request: LoadWorkerRequest) -> dict[str, Any]:
        if mode not in {"text", "vision"}:
            raise ValueError(f"unsupported mode: {mode}")

        with self._lock:
            previous_mode = self._mode
            if self._worker is not None:
                self._unload_locked()

            worker_kwargs: dict[str, Any] = {
                "mode": mode,
                "model": request.model,
                "tensor_parallel_size": request.tensor_parallel_size,
                "gpu_memory_utilization": request.gpu_memory_utilization,
                "max_model_len": request.max_model_len,
                "max_num_batched_tokens": request.max_num_batched_tokens,
                "max_num_seqs": request.max_num_seqs,
                "cpu_offload_gb": request.cpu_offload_gb,
                "enforce_eager": request.enforce_eager,
                "debug_scheduler": request.debug_scheduler,
                "enable_speculative": request.enable_speculative,
                "num_speculative_tokens": request.num_speculative_tokens,
                "quantization": request.quantization,
                "hf_token": request.hf_token,
                "mm_processor_kwargs": request.mm_processor_kwargs,
                "disable_log_stats": request.disable_log_stats,
            }
            if mode == "vision":
                worker_kwargs["processor"] = request.processor
                worker_kwargs["limit_mm_per_prompt"] = request.limit_mm_per_prompt

            worker = create_qwen_worker(
                **worker_kwargs,
            )
            if request.warmup and mode == "text":
                worker.warmup()

            self._worker = worker
            self._mode = mode
            self._loaded_at = time.time()
            return {
                "ok": True,
                "mode": mode,
                "replaced_previous_mode": previous_mode,
                "worker_config": _json_safe(asdict(worker.config)),
                "gpu_after_load": _gpu_snapshot(),
            }

    def unload(self) -> dict[str, Any]:
        with self._lock:
            previous_mode = self._mode
            self._unload_locked()
            return {
                "ok": True,
                "unloaded_mode": previous_mode,
                "gpu_after_unload": _gpu_snapshot(),
            }

    def generate_text(self, request: TextGenerateRequest) -> dict[str, Any]:
        with self._lock:
            worker = self._require_text_capable_worker_locked()
            started = time.perf_counter()
            responses = worker.generate_text_batch(
                request.prompts,
                system_prompt=request.system_prompt,
                generation=request.generation.to_generation_config(),
                use_tqdm=request.use_tqdm,
            )
            ended = time.perf_counter()
            return self._make_generation_result(
                mode=str(self._mode or "text"),
                elapsed_s=ended - started,
                responses=responses,
                request_count=len(request.prompts),
            )

    def generate_vision(self, request: VisionGenerateRequest) -> dict[str, Any]:
        with self._lock:
            worker = self._require_mode_locked("vision")
            started = time.perf_counter()
            responses = worker.generate_vision_batch(
                [{"prompt": item.prompt, "images": item.images} for item in request.requests],
                system_prompt=request.system_prompt,
                generation=request.generation.to_generation_config(),
                use_tqdm=request.use_tqdm,
            )
            ended = time.perf_counter()
            return self._make_generation_result(
                mode="vision",
                elapsed_s=ended - started,
                responses=responses,
                request_count=len(request.requests),
            )

    def _make_generation_result(
        self,
        *,
        mode: str,
        elapsed_s: float,
        responses: Sequence[ResponseRecord],
        request_count: int,
    ) -> dict[str, Any]:
        output_tokens = sum(item.output_tokens for item in responses)
        prompt_tokens = sum(item.prompt_tokens for item in responses)
        cached_prompt_tokens = sum((item.cached_prompt_tokens or 0) for item in responses)
        return {
            "ok": True,
            "mode": mode,
            "request_count": request_count,
            "elapsed_s": round(max(elapsed_s, 0.0), 6),
            "prompt_tokens_total": prompt_tokens,
            "cached_prompt_tokens_total": cached_prompt_tokens,
            "output_tokens_total": output_tokens,
            "output_tokens_per_s": round(output_tokens / max(elapsed_s, 1e-9), 3),
            "responses": [_json_safe(asdict(item)) for item in responses],
            "gpu_after": _gpu_snapshot(),
        }

    def _unload_locked(self) -> None:
        worker = self._worker
        self._worker = None
        self._mode = None
        self._loaded_at = None
        if worker is not None:
            worker.close()

    def _require_mode_locked(self, expected_mode: str) -> QwenVLLMWorker:
        if self._worker is None or self._mode is None:
            raise HTTPException(status_code=409, detail="no worker is loaded")
        if self._mode != expected_mode:
            raise HTTPException(status_code=409, detail=f"loaded worker mode is '{self._mode}', not '{expected_mode}'")
        return self._worker

    def _require_text_capable_worker_locked(self) -> QwenVLLMWorker:
        if self._worker is None or self._mode is None:
            raise HTTPException(status_code=409, detail="no worker is loaded")
        if self._mode not in {"text", "vision"}:
            raise HTTPException(status_code=409, detail=f"loaded worker mode is '{self._mode}', not text-capable")
        return self._worker


class QwenServerClient:
    """
    Optional client helper for the main app.
    Keep this on the Windows side if you want the pipeline to call the WSL server
    with a worker-like interface instead of hand-building HTTP requests.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        try:
            import requests  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("requests is required for QwenServerClient") from exc
        self._requests = requests

    def endpoints(self) -> dict[str, str]:
        return {
            "base": self.base_url,
            "status": f"{self.base_url}/status",
            "load_text": f"{self.base_url}/load/text",
            "load_vision": f"{self.base_url}/load/vision",
            "unload": f"{self.base_url}/unload",
            "generate_text": f"{self.base_url}/generate/text",
            "generate_vision": f"{self.base_url}/generate/vision",
            "health": f"{self.base_url}/health",
            "shutdown": f"{self.base_url}/shutdown",
        }

    def load_text(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/load/text", kwargs)

    def load_vision(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/load/vision", kwargs)

    def unload(self) -> dict[str, Any]:
        return self._post("/unload", {})

    def status(self) -> dict[str, Any]:
        return self._get("/status")

    def generate_text_batch(
        self,
        prompts: Sequence[str],
        *,
        system_prompt: Optional[str] = None,
        generation: Optional[dict[str, Any]] = None,
        use_tqdm: bool = False,
    ) -> dict[str, Any]:
        return self._post(
            "/generate/text",
            {
                "prompts": list(prompts),
                "system_prompt": system_prompt,
                "generation": generation or {},
                "use_tqdm": use_tqdm,
            },
        )

    def generate_vision_batch(
        self,
        requests: Sequence[dict[str, Any]],
        *,
        system_prompt: Optional[str] = None,
        generation: Optional[dict[str, Any]] = None,
        use_tqdm: bool = False,
    ) -> dict[str, Any]:
        return self._post(
            "/generate/vision",
            {
                "requests": list(requests),
                "system_prompt": system_prompt,
                "generation": generation or {},
                "use_tqdm": use_tqdm,
            },
        )

    def text_endpoint(self) -> str:
        return f"{self.base_url}/generate/text"

    def vision_endpoint(self) -> str:
        return f"{self.base_url}/generate/vision"

    def _get(self, path: str) -> dict[str, Any]:
        resp = self._requests.get(f"{self.base_url}{path}", timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._requests.post(f"{self.base_url}{path}", json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()


def create_app(*, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> FastAPI:
    app = FastAPI(title="Qwen vLLM Worker Server", version="1.0.0")
    manager = QwenWorkerManager()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "gpu": _gpu_snapshot()}

    @app.get("/status")
    def status() -> dict[str, Any]:
        data = manager.status()
        data["endpoints"] = manager.endpoint_map(host, port)
        return data

    @app.post("/load/text")
    def load_text(request: LoadWorkerRequest) -> dict[str, Any]:
        return manager.load("text", request)

    @app.post("/load/vision")
    def load_vision(request: LoadWorkerRequest) -> dict[str, Any]:
        return manager.load("vision", request)

    @app.post("/unload")
    def unload() -> dict[str, Any]:
        return manager.unload()

    @app.post("/generate/text")
    def generate_text(request: TextGenerateRequest) -> dict[str, Any]:
        return manager.generate_text(request)

    @app.post("/generate/vision")
    def generate_vision(request: VisionGenerateRequest) -> dict[str, Any]:
        return manager.generate_vision(request)

    @app.post("/shutdown")
    def shutdown() -> dict[str, Any]:
        manager.unload()
        return {"ok": True, "message": "worker unloaded; stop the process externally or via Ctrl+C"}

    return app


def _normalize_images(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raw = [raw]

    out: list[Any] = []
    for item in raw:
        if Image is not None and isinstance(item, Image.Image):
            out.append(item.convert("RGB"))
            continue
        if isinstance(item, (str, Path)):
            if Image is None:
                raise RuntimeError("Pillow is required for loading local images")
            path = Path(item)
            with Image.open(path) as img:
                out.append(img.convert("RGB"))
            continue
        raise TypeError(f"unsupported image input type: {type(item)!r}")
    return out


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _read_hf_token() -> Optional[str]:
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = str(os.environ.get(name, "") or "").strip()
        if value:
            return value
    return None


def _gpu_snapshot() -> dict[str, Any]:
    if torch is None or not torch.cuda.is_available():
        return {"cuda": False}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
    return {
        "cuda": True,
        "device_index": idx,
        "device_name": torch.cuda.get_device_name(idx),
        "total_vram_gib": round(total_bytes / (1024 ** 3), 2),
        "free_vram_gib": round(free_bytes / (1024 ** 3), 2),
        "used_vram_gib": round((total_bytes - free_bytes) / (1024 ** 3), 2),
        "compute_capability": f"{props.major}.{props.minor}",
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resident Qwen vLLM worker server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = create_app(host=args.host, port=args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info" if args.verbose else "warning")


if __name__ == "__main__":
    main()
