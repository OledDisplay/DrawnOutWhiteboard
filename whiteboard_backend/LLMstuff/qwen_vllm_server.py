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

if not str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "") or "").strip():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

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
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "vllm is required for this worker server. Install it inside the WSL/Linux environment that will host the server."
    ) from exc

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    import uvicorn
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "fastapi, pydantic and uvicorn are required for server mode. Install: pip install fastapi uvicorn pydantic"
    ) from exc


DEFAULT_MODEL_ID = "cyankiwi/Qwen3.5-4B-AWQ-4bit"
DEFAULT_MAX_MODEL_LEN = int(os.getenv("QWEN_SERVER_MAX_MODEL_LEN_DEFAULT", "20000") or 20000)
DEFAULT_GPU_MEMORY_UTILIZATION = float(os.getenv("QWEN_SERVER_GPU_UTIL_DEFAULT", "0.80") or 0.80)
DEFAULT_TEXT_MAX_NUM_SEQS = int(os.getenv("QWEN_SERVER_TEXT_MAX_NUM_SEQS", "24") or 24)
DEFAULT_ENFORCE_EAGER = bool(int(os.getenv("QWEN_SERVER_ENFORCE_EAGER", "0") or 0))
DEFAULT_KV_CACHE_DTYPE = "fp8"
DEFAULT_ATTENTION_CONFIG = {"flash_attn_version": 2}
DEFAULT_COMPILATION_CONFIG = int(os.getenv("QWEN_SERVER_COMPILATION_CONFIG", "3") or 3)
DEFAULT_BATCH_LONG_PREFILL_THRESHOLD = 4096
DEFAULT_MAX_BATCHED_TOKENS = int(os.getenv("QWEN_SERVER_MAX_BATCHED_TOKENS_DEFAULT", "8192") or 8192)
DEFAULT_MM_LIMIT_PER_PROMPT = {"image": 8}
DEFAULT_GPU_UTILIZATION_HARD_CAP = float(os.getenv("QWEN_SERVER_GPU_UTIL_HARD_CAP", "0.98") or 0.98)
DEFAULT_TEXT_FREE_VRAM_RESERVE_GIB = float(os.getenv("QWEN_SERVER_TEXT_FREE_VRAM_RESERVE_GIB", "0.35") or 0.35)
DEFAULT_VISION_FREE_VRAM_RESERVE_GIB = float(os.getenv("QWEN_SERVER_VISION_FREE_VRAM_RESERVE_GIB", "0.0") or 0.0)
DEFAULT_SPECULATIVE_EXTRA_RESERVE_GIB = float(os.getenv("QWEN_SERVER_SPECULATIVE_EXTRA_RESERVE_GIB", "0.75") or 0.75)
DEFAULT_MTP_EXTRA_RESERVE_GIB = float(os.getenv("QWEN_SERVER_MTP_EXTRA_RESERVE_GIB", "1.00") or 1.00)
DEFAULT_MTP_EXTRA_RESERVE_PER_TOKEN_GIB = float(os.getenv("QWEN_SERVER_MTP_EXTRA_RESERVE_PER_TOKEN_GIB", "0.35") or 0.35)
DEFAULT_ENABLE_SLEEP_MODE = bool(int(os.getenv("QWEN_SERVER_ENABLE_SLEEP_MODE", "1") or 1))
DEFAULT_SLEEP_LEVEL = max(1, min(2, int(os.getenv("QWEN_SERVER_SLEEP_LEVEL", "1") or 1)))
DEFAULT_AUTO_LOAD_TEXT_ON_STARTUP = bool(int(os.getenv("QWEN_SERVER_AUTO_LOAD_TEXT_ON_STARTUP", "0") or 0))
DEFAULT_AUTO_LOAD_VISION_ON_STARTUP = bool(int(os.getenv("QWEN_SERVER_AUTO_LOAD_VISION_ON_STARTUP", "1") or 1))
DEFAULT_AUTO_WARMUP_ON_STARTUP = bool(int(os.getenv("QWEN_SERVER_AUTO_WARMUP_ON_STARTUP", "1") or 1))
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8009

LOG = logging.getLogger("qwen_vllm_server")


def _optional_positive_int_env(name: str) -> Optional[int]:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


DEFAULT_VISION_MM_PROCESSOR_CACHE_TYPE = str(os.getenv("QWEN_SERVER_VISION_MM_PROCESSOR_CACHE_TYPE", "") or "").strip() or None
DEFAULT_VISION_MM_PROCESSOR_CACHE_GB = _optional_positive_int_env("QWEN_SERVER_VISION_MM_PROCESSOR_CACHE_GB")
DEFAULT_VISION_MM_SHM_CACHE_MAX_OBJECT_SIZE_MB = _optional_positive_int_env("QWEN_SERVER_VISION_MM_SHM_CACHE_MAX_OBJECT_SIZE_MB")


def _is_qwen35_family(model: str) -> bool:
    model_l = (model or "").lower()
    return "qwen3.5" in model_l or "qwen3_5" in model_l or "qwen3.6" in model_l or "qwen3_6" in model_l


def _is_qwen3_next_family(model: str) -> bool:
    model_l = (model or "").lower()
    return "qwen3-next" in model_l or "qwen3_next" in model_l


def _default_speculative_method(model: str) -> str:
    env_override = str(os.getenv("QWEN_SERVER_SPECULATIVE_METHOD_DEFAULT", "") or "").strip()
    if env_override:
        return env_override

    if _is_qwen3_next_family(model):
        return "qwen3_next_mtp"
    if _is_qwen35_family(model):
        return "mtp"
    return "mtp"


def _default_enable_speculative_for_model(model: str, *, mode: str) -> bool:
    return False


def _fallback_speculative_methods(method: str, model: str) -> list[str]:
    methods: list[str] = []
    normalized = str(method or "").strip()
    if normalized:
        methods.append(normalized)

    if _is_qwen3_next_family(model):
        for candidate in ("qwen3_next_mtp", "mtp"):
            if candidate not in methods:
                methods.append(candidate)
    elif _is_qwen35_family(model):
        for candidate in ("mtp",):
            if candidate not in methods:
                methods.append(candidate)
    elif "mtp" not in methods:
        methods.append("mtp")

    return methods


def _speculative_reserve_gib(*, enable_speculative: bool, speculative_method: Optional[str], num_speculative_tokens: Any) -> float:
    if not enable_speculative:
        return 0.0
    try:
        spec_tokens = max(1, int(num_speculative_tokens))
    except Exception:
        spec_tokens = 1
    method = str(speculative_method or "").strip().lower()
    if method in {"mtp", "qwen3_next_mtp"}:
        return DEFAULT_MTP_EXTRA_RESERVE_GIB + (DEFAULT_MTP_EXTRA_RESERVE_PER_TOKEN_GIB * spec_tokens)
    return DEFAULT_SPECULATIVE_EXTRA_RESERVE_GIB


def _estimate_gpu_load_delta_gib(before: Any, after: Any) -> Optional[float]:
    before_gpu = dict(before or {}) if isinstance(before, dict) else {}
    after_gpu = dict(after or {}) if isinstance(after, dict) else {}
    if not before_gpu.get("cuda") or not after_gpu.get("cuda"):
        return None
    before_used = before_gpu.get("used_vram_gib")
    after_used = after_gpu.get("used_vram_gib")
    try:
        delta = float(after_used) - float(before_used)
    except Exception:
        return None
    return round(delta, 3)


def _clamp_gpu_memory_utilization(
    requested: Any,
    *,
    mode: str,
    enable_speculative: bool = False,
    speculative_method: Optional[str] = None,
    num_speculative_tokens: Any = 1,
) -> float:
    try:
        requested_value = float(requested)
    except Exception:
        requested_value = DEFAULT_GPU_MEMORY_UTILIZATION
    snapshot = _gpu_snapshot()
    if not snapshot.get("cuda"):
        return round(min(requested_value, DEFAULT_GPU_UTILIZATION_HARD_CAP), 3)
    free_gib = float(snapshot.get("free_vram_gib") or 0.0)
    total_gib = float(snapshot.get("total_vram_gib") or 0.0)
    reserve_gib = DEFAULT_VISION_FREE_VRAM_RESERVE_GIB if mode == "vision" else DEFAULT_TEXT_FREE_VRAM_RESERVE_GIB
    reserve_gib += _speculative_reserve_gib(
        enable_speculative=enable_speculative,
        speculative_method=speculative_method,
        num_speculative_tokens=num_speculative_tokens,
    )
    if free_gib <= 0.0 or total_gib <= 0.0:
        return round(min(requested_value, DEFAULT_GPU_UTILIZATION_HARD_CAP), 3)
    free_limited_util = max(0.55, (free_gib - reserve_gib) / total_gib)
    return round(min(requested_value, free_limited_util, DEFAULT_GPU_UTILIZATION_HARD_CAP), 3)


def _recommended_max_num_seqs(
    *,
    mode: str,
    requested: Any,
    max_model_len: Any,
    enable_speculative: bool = False,
    speculative_method: Optional[str] = None,
    num_speculative_tokens: Any = 1,
) -> int:
    try:
        requested_value = max(1, int(requested))
    except Exception:
        requested_value = DEFAULT_TEXT_MAX_NUM_SEQS
    try:
        model_len_value = max(1024, int(max_model_len))
    except Exception:
        model_len_value = DEFAULT_MAX_MODEL_LEN

    snapshot = _gpu_snapshot()
    total_gib = float(snapshot.get("total_vram_gib") or 0.0)
    free_gib = float(snapshot.get("free_vram_gib") or 0.0)
    free_ratio = (free_gib / total_gib) if total_gib > 0.0 else 1.0
    long_context = model_len_value >= 12000

    if mode == "vision":
        if total_gib <= 8.5:
            safe_cap = 2
        elif total_gib <= 12.5:
            safe_cap = 4
        else:
            safe_cap = 6
    else:
        if total_gib <= 8.5:
            safe_cap = 12 if long_context else 16
        elif total_gib <= 12.5:
            safe_cap = 20 if long_context else 28
        elif total_gib <= 16.5:
            safe_cap = 28 if long_context else 40
        elif total_gib <= 24.5:
            safe_cap = 40 if long_context else 64
        else:
            safe_cap = requested_value

    if free_ratio < 0.80:
        safe_cap = max(2 if mode == "vision" else 4, int(round(safe_cap * 0.75)))
    if free_ratio < 0.65:
        safe_cap = max(2 if mode == "vision" else 4, int(round(safe_cap * 0.5)))
    if enable_speculative and mode == "text":
        try:
            spec_tokens = max(1, int(num_speculative_tokens))
        except Exception:
            spec_tokens = 1
        method = str(speculative_method or "").strip().lower()
        if method in {"mtp", "qwen3_next_mtp"}:
            safe_cap = max(4, int(round(safe_cap * max(0.45, 0.72 - (0.06 * (spec_tokens - 1))))))
        else:
            safe_cap = max(4, int(round(safe_cap * 0.65)))
    return max(1, min(requested_value, safe_cap))


def _shutdown_process_group() -> None:
    if torch is None:
        return
    try:
        import torch.distributed as dist
    except Exception:
        return
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def _shutdown_vllm_runtime(llm: Any) -> None:
    if llm is None:
        return
    targets = [
        llm,
        getattr(llm, "llm_engine", None),
    ]
    llm_engine = getattr(llm, "llm_engine", None)
    if llm_engine is not None:
        targets.extend(
            [
                getattr(llm_engine, "engine_core", None),
                getattr(llm_engine, "model_executor", None),
                getattr(llm_engine, "executor", None),
            ]
        )
    seen: set[int] = set()
    for target in targets:
        if target is None:
            continue
        target_id = id(target)
        if target_id in seen:
            continue
        seen.add(target_id)
        for method_name in ("shutdown", "close"):
            try:
                method = getattr(target, method_name, None)
            except Exception:
                method = None
            if callable(method):
                try:
                    method()
                except Exception:
                    pass


@dataclass(slots=True)
class WorkerConfig:
    model: str = DEFAULT_MODEL_ID
    tokenizer: Optional[str] = None
    trust_remote_code: bool = True
    tensor_parallel_size: int = 1
    dtype: str = "half"
    quantization: Optional[str] = None
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    kv_cache_dtype: str = DEFAULT_KV_CACHE_DTYPE
    enforce_eager: bool = DEFAULT_ENFORCE_EAGER
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    scheduler_reserve_full_isl: bool = True
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: Optional[int] = None
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
    enable_speculative: bool = False
    speculative_method: Optional[str] = None
    num_speculative_tokens: int = 1
    default_enable_thinking: bool = False
    seed: int = 0
    disable_log_stats: bool = False
    enable_logging_iteration_details: bool = False
    kv_cache_metrics: bool = False
    cudagraph_metrics: bool = False
    hf_token: Optional[str] = None
    enable_sleep_mode: bool = DEFAULT_ENABLE_SLEEP_MODE

    def for_vllm(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "tokenizer": self.tokenizer or self.model,
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
            "compilation_config": self.compilation_config,
            "attention_config": self.attention_config,
            "language_model_only": self.language_model_only,
            "limit_mm_per_prompt": self.limit_mm_per_prompt,
            "seed": self.seed,
            "disable_log_stats": self.disable_log_stats,
            "enable_logging_iteration_details": self.enable_logging_iteration_details,
            "kv_cache_metrics": self.kv_cache_metrics,
            "cudagraph_metrics": self.cudagraph_metrics,
            "enable_sleep_mode": self.enable_sleep_mode,
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
        if self.enable_speculative:
            kwargs["speculative_config"] = {
                "method": self.speculative_method or _default_speculative_method(self.model),
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
        self._is_sleeping = False

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
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=self.config.trust_remote_code,
                token=self.hf_token,
            )
            tok = getattr(self._processor, "tokenizer", None)
            if tok is not None and getattr(tok, "pad_token_id", None) is None:
                eos = getattr(tok, "eos_token_id", None)
                if eos is not None:
                    tok.pad_token_id = eos

    def _build_engine(self) -> LLM:
        kwargs = self.config.for_vllm()
        LOG.info("Creating vLLM engine with config: %s", json.dumps(_json_safe(kwargs), indent=2))
        try:
            return LLM(**kwargs)
        except Exception as exc:
            if not self.config.enable_speculative:
                raise

            speculative_config = dict(kwargs.get("speculative_config") or {})
            configured_method = str(speculative_config.get("method") or "")
            fallback_methods = _fallback_speculative_methods(configured_method, self.config.model)
            if len(fallback_methods) <= 1:
                raise

            last_exc = exc
            for fallback_method in fallback_methods[1:]:
                retry_kwargs = dict(kwargs)
                retry_speculative = dict(speculative_config)
                retry_speculative["method"] = fallback_method
                retry_kwargs["speculative_config"] = retry_speculative
                LOG.warning(
                    "Speculative method '%s' failed for model '%s'; retrying with '%s'. Original error: %r",
                    configured_method,
                    self.config.model,
                    fallback_method,
                    exc,
                )
                try:
                    self.config.speculative_method = fallback_method
                    return LLM(**retry_kwargs)
                except Exception as retry_exc:
                    last_exc = retry_exc
            raise last_exc

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
        generation = GenerationConfig(max_new_tokens=8, temperature=0.0)
        if self.mode == "text":
            self.generate_text_batch(
                ["Reply with the single token: warm."],
                generation=generation,
                use_tqdm=False,
            )
            return
        if self.mode == "vision":
            if Image is None:
                raise RuntimeError("Pillow is required for vision warmup")
            sample = Image.new("RGB", (32, 32), color=(255, 255, 255))
            self.generate_vision_batch(
                [{"prompt": "Reply with the single token: warm.", "images": [sample]}],
                generation=generation,
                use_tqdm=False,
            )
            return
        raise RuntimeError(f"unsupported warmup mode: {self.mode}")

    def sleep(self, *, level: int = DEFAULT_SLEEP_LEVEL) -> None:
        sleep_fn = getattr(self.llm, "sleep", None)
        if not callable(sleep_fn):
            raise RuntimeError("vLLM sleep mode is not available on this build")
        sleep_fn(level=int(level))
        self._is_sleeping = True

    def wake_up(self) -> None:
        wake_fn = getattr(self.llm, "wake_up", None)
        if not callable(wake_fn):
            raise RuntimeError("vLLM wake_up is not available on this build")
        wake_fn()
        self._is_sleeping = False

    def mark_sleep_failed(self) -> None:
        self._is_sleeping = False

    def close(self) -> None:
        if getattr(self, "llm", None) is not None:
            try:
                _shutdown_vllm_runtime(self.llm)
            except Exception:
                pass
            try:
                del self.llm
            except Exception:
                pass
        self._tokenizer = None
        self._processor = None
        gc.collect()
        _shutdown_process_group()
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
    max_num_seqs: int = DEFAULT_TEXT_MAX_NUM_SEQS,
    debug_scheduler: bool = False,
    enforce_eager: bool = DEFAULT_ENFORCE_EAGER,
    enable_prefix_caching: Optional[bool] = None,
    enable_chunked_prefill: bool = True,
    max_num_batched_tokens: Optional[int] = None,
    max_num_partial_prefills: int = 1,
    max_long_partial_prefills: int = 1,
    long_prefill_token_threshold: int = DEFAULT_BATCH_LONG_PREFILL_THRESHOLD,
    enable_speculative: bool = False,
    speculative_method: Optional[str] = None,
    num_speculative_tokens: int = 1,
    hf_token: Optional[str] = None,
) -> QwenVLLMWorker:
    if enable_prefix_caching is None:
        enable_prefix_caching = True

    if max_num_batched_tokens is None and enable_chunked_prefill and not enable_speculative:
        max_num_batched_tokens = DEFAULT_MAX_BATCHED_TOKENS

    cfg = WorkerConfig(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max(1, int(max_num_seqs)),
        language_model_only=True,
        enforce_eager=bool(enforce_eager),
        enable_prefix_caching=bool(enable_prefix_caching),
        enable_chunked_prefill=bool(enable_chunked_prefill),
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_partial_prefills=max(1, int(max_num_partial_prefills)),
        max_long_partial_prefills=max(1, int(max_long_partial_prefills)),
        long_prefill_token_threshold=max(0, int(long_prefill_token_threshold)),
        enable_speculative=enable_speculative,
        speculative_method=speculative_method,
        num_speculative_tokens=max(1, int(num_speculative_tokens)),
        enable_logging_iteration_details=debug_scheduler,
        kv_cache_metrics=debug_scheduler,
        cudagraph_metrics=debug_scheduler,
        hf_token=hf_token,
    )
    return QwenVLLMWorker(config=cfg, mode="text")


def create_vision_worker(
    *,
    model: str = DEFAULT_MODEL_ID,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_num_seqs: int = DEFAULT_TEXT_MAX_NUM_SEQS,
    debug_scheduler: bool = False,
    enforce_eager: bool = DEFAULT_ENFORCE_EAGER,
    enable_prefix_caching: Optional[bool] = None,
    enable_chunked_prefill: bool = True,
    max_num_batched_tokens: Optional[int] = None,
    max_num_partial_prefills: int = 1,
    max_long_partial_prefills: int = 1,
    long_prefill_token_threshold: int = DEFAULT_BATCH_LONG_PREFILL_THRESHOLD,
    enable_speculative: bool = False,
    speculative_method: Optional[str] = None,
    num_speculative_tokens: int = 1,
    hf_token: Optional[str] = None,
) -> QwenVLLMWorker:
    if enable_prefix_caching is None:
        enable_prefix_caching = True

    if max_num_batched_tokens is None and enable_chunked_prefill and not enable_speculative:
        max_num_batched_tokens = DEFAULT_MAX_BATCHED_TOKENS

    cfg = WorkerConfig(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max(1, int(max_num_seqs)),
        language_model_only=False,
        enforce_eager=bool(enforce_eager),
        mm_encoder_tp_mode="data",
        mm_processor_cache_type=DEFAULT_VISION_MM_PROCESSOR_CACHE_TYPE,
        mm_processor_cache_gb=DEFAULT_VISION_MM_PROCESSOR_CACHE_GB,
        mm_shm_cache_max_object_size_mb=DEFAULT_VISION_MM_SHM_CACHE_MAX_OBJECT_SIZE_MB,
        enable_prefix_caching=bool(enable_prefix_caching),
        enable_chunked_prefill=bool(enable_chunked_prefill),
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_partial_prefills=max(1, int(max_num_partial_prefills)),
        max_long_partial_prefills=max(1, int(max_long_partial_prefills)),
        long_prefill_token_threshold=max(0, int(long_prefill_token_threshold)),
        enable_speculative=enable_speculative,
        speculative_method=speculative_method,
        num_speculative_tokens=max(1, int(num_speculative_tokens)),
        enable_logging_iteration_details=debug_scheduler,
        kv_cache_metrics=debug_scheduler,
        cudagraph_metrics=debug_scheduler,
        hf_token=hf_token,
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
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    max_num_seqs: int = DEFAULT_TEXT_MAX_NUM_SEQS
    debug_scheduler: bool = False
    enforce_eager: bool = DEFAULT_ENFORCE_EAGER
    enable_prefix_caching: Optional[bool] = None
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: Optional[int] = None
    max_num_partial_prefills: int = 1
    max_long_partial_prefills: int = 1
    long_prefill_token_threshold: int = DEFAULT_BATCH_LONG_PREFILL_THRESHOLD
    enable_speculative: bool = False
    speculative_method: Optional[str] = None
    num_speculative_tokens: int = 1
    hf_token: Optional[str] = None
    warmup: bool = False


class WakeWorkerRequest(BaseModel):
    model: Optional[str] = None
    tensor_parallel_size: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    max_model_len: Optional[int] = None
    max_num_seqs: Optional[int] = None
    debug_scheduler: Optional[bool] = None
    enforce_eager: Optional[bool] = None
    enable_prefix_caching: Optional[bool] = None
    enable_chunked_prefill: Optional[bool] = None
    max_num_batched_tokens: Optional[int] = None
    max_num_partial_prefills: Optional[int] = None
    max_long_partial_prefills: Optional[int] = None
    long_prefill_token_threshold: Optional[int] = None
    enable_speculative: Optional[bool] = None
    speculative_method: Optional[str] = None
    num_speculative_tokens: Optional[int] = None
    hf_token: Optional[str] = None
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
        self._sleeping: bool = False
        self._sleep_mode: Optional[str] = None
        self._sleep_request: Optional[LoadWorkerRequest] = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "loaded": self._worker is not None,
                "mode": self._mode,
                "loaded_at_unix": self._loaded_at,
                "sleeping": self._sleeping,
                "sleep_mode": self._sleep_mode,
                "gpu": _gpu_snapshot(),
                "worker_config": _json_safe(asdict(self._worker.config)) if self._worker is not None else None,
                "sleep_request": self._sleep_request.model_dump() if self._sleep_request is not None else None,
            }

    def endpoint_map(self, host: str, port: int) -> dict[str, str]:
        base = f"http://{host}:{port}"
        return {
            "base": base,
            "status": f"{base}/status",
            "load_text": f"{base}/load/text",
            "reload_text": f"{base}/reload/text",
            "wake_text": f"{base}/wake/text",
            "load_vision": f"{base}/load/vision",
            "reload_vision": f"{base}/reload/vision",
            "wake_vision": f"{base}/wake/vision",
            "sleep": f"{base}/sleep",
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
            gpu_before = _gpu_snapshot()
            resolved_request = self._resolve_load_request_defaults(mode=mode, request=request)
            if self._worker is not None:
                self._unload_locked()
            worker = self._build_worker(mode=mode, request=resolved_request)
            gpu_after = _gpu_snapshot()

            self._worker = worker
            self._mode = mode
            self._loaded_at = time.time()
            self._sleeping = False
            self._sleep_mode = mode
            self._sleep_request = resolved_request.model_copy(deep=True)
            return {
                "ok": True,
                "mode": mode,
                "replaced_previous_mode": previous_mode,
                "reloaded": previous_mode is not None,
                "worker_config": _json_safe(asdict(worker.config)),
                "gpu_before_load": gpu_before,
                "gpu_after_load": gpu_after,
                "estimated_model_vram_delta_gib": _estimate_gpu_load_delta_gib(gpu_before, gpu_after),
            }

    def wake(self, mode: str, request: WakeWorkerRequest) -> dict[str, Any]:
        if mode not in {"text", "vision"}:
            raise ValueError(f"unsupported mode: {mode}")

        with self._lock:
            if self._worker is not None and self._mode == mode:
                if self._sleeping:
                    gpu_before = _gpu_snapshot()
                    try:
                        self._worker.wake_up()
                        gpu_after = _gpu_snapshot()
                        self._sleeping = False
                        self._sleep_mode = mode
                        self._sleep_request = self._merge_wake_request(base_request=self._sleep_request or LoadWorkerRequest(), override=request)
                        return {
                            "ok": True,
                            "mode": mode,
                            "already_loaded": True,
                            "woke_from_sleep": True,
                            "worker_config": _json_safe(asdict(self._worker.config)),
                            "gpu_before_load": gpu_before,
                            "gpu_after_load": gpu_after,
                            "estimated_model_vram_delta_gib": _estimate_gpu_load_delta_gib(gpu_before, gpu_after),
                        }
                    except Exception as exc:
                        LOG.warning("vLLM wake_up failed for mode '%s'; rebuilding worker instead. Error: %r", mode, exc)
                        try:
                            self._worker.mark_sleep_failed()
                        except Exception:
                            pass
                        self._unload_locked()
                        self._sleeping = False
                    if self._worker is None:
                        # Native wake failed and the worker was unloaded; continue into
                        # the rebuild-from-saved-config path below.
                        pass
                    else:
                        return {
                            "ok": True,
                            "mode": mode,
                            "already_loaded": True,
                            "woke_from_sleep": False,
                            "worker_config": _json_safe(asdict(self._worker.config)),
                            "gpu_after_load": _gpu_snapshot(),
                        }
                else:
                    return {
                        "ok": True,
                        "mode": mode,
                        "already_loaded": True,
                        "woke_from_sleep": False,
                        "worker_config": _json_safe(asdict(self._worker.config)),
                        "gpu_after_load": _gpu_snapshot(),
                    }

            base_request = self._sleep_request
            base_mode = self._sleep_mode
            if base_request is None or base_mode != mode:
                raise HTTPException(status_code=409, detail=f"no sleeping worker config is available for mode '{mode}'")

            merged_request = self._merge_wake_request(base_request=base_request, override=request)
            resolved_request = self._resolve_load_request_defaults(mode=mode, request=merged_request)
            gpu_before = _gpu_snapshot()
            if self._worker is not None:
                self._unload_locked()
            worker = self._build_worker(mode=mode, request=resolved_request)
            gpu_after = _gpu_snapshot()

            self._worker = worker
            self._mode = mode
            self._loaded_at = time.time()
            self._sleeping = False
            self._sleep_mode = mode
            self._sleep_request = resolved_request.model_copy(deep=True)
            return {
                "ok": True,
                "mode": mode,
                "already_loaded": False,
                "woke_from_sleep": True,
                "worker_config": _json_safe(asdict(worker.config)),
                "gpu_before_load": gpu_before,
                "gpu_after_load": gpu_after,
                "estimated_model_vram_delta_gib": _estimate_gpu_load_delta_gib(gpu_before, gpu_after),
            }

    def sleep(self) -> dict[str, Any]:
        with self._lock:
            if self._worker is None:
                return {
                    "ok": True,
                    "sleeping": bool(self._sleeping and self._sleep_request is not None),
                    "slept_mode": self._sleep_mode,
                    "gpu_after_sleep": _gpu_snapshot(),
                }
            slept_mode = self._mode
            self._sleep_mode = self._mode
            self._worker.sleep(level=DEFAULT_SLEEP_LEVEL)
            self._sleeping = True
            return {
                "ok": True,
                "sleeping": True,
                "slept_mode": slept_mode,
                "gpu_after_sleep": _gpu_snapshot(),
            }

    def unload(self) -> dict[str, Any]:
        with self._lock:
            previous_mode = self._mode
            self._unload_locked()
            self._sleeping = False
            self._sleep_mode = None
            self._sleep_request = None
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

    def _resolve_load_request_defaults(self, *, mode: str, request: LoadWorkerRequest) -> LoadWorkerRequest:
        data = request.model_dump()

        if "enable_speculative" not in request.model_fields_set:
            data["enable_speculative"] = _default_enable_speculative_for_model(request.model, mode=mode)

        if "speculative_method" not in request.model_fields_set and data.get("enable_speculative"):
            data["speculative_method"] = _default_speculative_method(request.model)

        requested_gpu_util = data.get("gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTILIZATION)
        speculative_reserve = _speculative_reserve_gib(
            enable_speculative=bool(data.get("enable_speculative")),
            speculative_method=data.get("speculative_method"),
            num_speculative_tokens=data.get("num_speculative_tokens", 1),
        )
        clamped_gpu_util = _clamp_gpu_memory_utilization(
            requested_gpu_util,
            mode=mode,
            enable_speculative=bool(data.get("enable_speculative")),
            speculative_method=data.get("speculative_method"),
            num_speculative_tokens=data.get("num_speculative_tokens", 1),
        )
        if abs(float(clamped_gpu_util) - float(requested_gpu_util)) > 1e-6:
            LOG.info(
                "Clamped gpu_memory_utilization for %s worker from %.3f to %.3f based on currently free VRAM%s.",
                mode,
                float(requested_gpu_util),
                float(clamped_gpu_util),
                (
                    f"; speculative reserve={speculative_reserve:.2f} GiB"
                    if speculative_reserve > 0.0
                    else ""
                ),
            )
        data["gpu_memory_utilization"] = clamped_gpu_util

        requested_max_num_seqs = data.get("max_num_seqs", DEFAULT_TEXT_MAX_NUM_SEQS)
        clamped_max_num_seqs = _recommended_max_num_seqs(
            mode=mode,
            requested=requested_max_num_seqs,
            max_model_len=data.get("max_model_len", DEFAULT_MAX_MODEL_LEN),
            enable_speculative=bool(data.get("enable_speculative")),
            speculative_method=data.get("speculative_method"),
            num_speculative_tokens=data.get("num_speculative_tokens", 1),
        )
        if int(clamped_max_num_seqs) != int(requested_max_num_seqs):
            LOG.info(
                "Clamped max_num_seqs for %s worker from %s to %s based on VRAM budget and context length.",
                mode,
                requested_max_num_seqs,
                clamped_max_num_seqs,
            )
        data["max_num_seqs"] = clamped_max_num_seqs

        if "enable_prefix_caching" not in request.model_fields_set:
            data["enable_prefix_caching"] = True if mode == "text" else bool(data.get("enable_prefix_caching", True))

        if (
            "max_num_batched_tokens" not in request.model_fields_set
            and bool(data.get("enable_chunked_prefill"))
            and not bool(data.get("enable_speculative"))
        ):
            data["max_num_batched_tokens"] = DEFAULT_MAX_BATCHED_TOKENS

        return LoadWorkerRequest(**data)

    def _build_worker(self, *, mode: str, request: LoadWorkerRequest) -> QwenVLLMWorker:
        request = self._resolve_load_request_defaults(mode=mode, request=request)
        worker = create_qwen_worker(
            mode=mode,
            model=request.model,
            tensor_parallel_size=request.tensor_parallel_size,
            gpu_memory_utilization=request.gpu_memory_utilization,
            max_model_len=request.max_model_len,
            max_num_seqs=request.max_num_seqs,
            debug_scheduler=request.debug_scheduler,
            enforce_eager=request.enforce_eager,
            enable_prefix_caching=request.enable_prefix_caching,
            enable_chunked_prefill=request.enable_chunked_prefill,
            max_num_batched_tokens=request.max_num_batched_tokens,
            max_num_partial_prefills=request.max_num_partial_prefills,
            max_long_partial_prefills=request.max_long_partial_prefills,
            long_prefill_token_threshold=request.long_prefill_token_threshold,
            enable_speculative=request.enable_speculative,
            speculative_method=request.speculative_method,
            num_speculative_tokens=request.num_speculative_tokens,
            hf_token=request.hf_token,
        )
        if request.warmup:
            worker.warmup()
        return worker

    def _merge_wake_request(self, *, base_request: LoadWorkerRequest, override: WakeWorkerRequest) -> LoadWorkerRequest:
        data = base_request.model_dump()
        override_data = override.model_dump(exclude_none=True)
        data.update(override_data)
        return LoadWorkerRequest(**data)

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
        if self._sleeping:
            raise HTTPException(status_code=409, detail=f"worker mode '{self._mode}' is sleeping; call wake first")
        return self._worker

    def _require_text_capable_worker_locked(self) -> QwenVLLMWorker:
        if self._worker is None or self._mode is None:
            raise HTTPException(status_code=409, detail="no worker is loaded")
        if self._mode not in {"text", "vision"}:
            raise HTTPException(status_code=409, detail=f"loaded worker mode is '{self._mode}', not text-capable")
        if self._sleeping:
            raise HTTPException(status_code=409, detail=f"worker mode '{self._mode}' is sleeping; call wake first")
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
            "reload_text": f"{self.base_url}/reload/text",
            "wake_text": f"{self.base_url}/wake/text",
            "load_vision": f"{self.base_url}/load/vision",
            "reload_vision": f"{self.base_url}/reload/vision",
            "wake_vision": f"{self.base_url}/wake/vision",
            "sleep": f"{self.base_url}/sleep",
            "unload": f"{self.base_url}/unload",
            "generate_text": f"{self.base_url}/generate/text",
            "generate_vision": f"{self.base_url}/generate/vision",
            "health": f"{self.base_url}/health",
            "shutdown": f"{self.base_url}/shutdown",
        }

    def load_text(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/load/text", kwargs)

    def reload_text(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/reload/text", kwargs)

    def wake_text(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/wake/text", kwargs)

    def load_vision(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/load/vision", kwargs)

    def reload_vision(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/reload/vision", kwargs)

    def wake_vision(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/wake/vision", kwargs)

    def sleep(self) -> dict[str, Any]:
        return self._post("/sleep", {})

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

    @app.on_event("startup")
    def startup_load_default_worker() -> None:
        request = LoadWorkerRequest(
            warmup=DEFAULT_AUTO_WARMUP_ON_STARTUP,
            gpu_memory_utilization=float(os.getenv("QWEN_SERVER_GPU_UTIL", str(DEFAULT_GPU_MEMORY_UTILIZATION)) or DEFAULT_GPU_MEMORY_UTILIZATION),
            max_model_len=int(os.getenv("QWEN_SERVER_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN)) or DEFAULT_MAX_MODEL_LEN),
        )
        try:
            if DEFAULT_AUTO_LOAD_VISION_ON_STARTUP:
                print(
                    "[qwen_vllm_server] startup loading VISION worker "
                    f"model={request.model} max_model_len={request.max_model_len} "
                    f"gpu_util={request.gpu_memory_utilization} warmup={request.warmup}",
                    flush=True,
                )
                manager.load("vision", request)
                print("[qwen_vllm_server] startup vision worker ready", flush=True)
            elif DEFAULT_AUTO_LOAD_TEXT_ON_STARTUP:
                print(
                    "[qwen_vllm_server] startup loading TEXT worker "
                    f"model={request.model} max_model_len={request.max_model_len} "
                    f"gpu_util={request.gpu_memory_utilization} warmup={request.warmup}",
                    flush=True,
                )
                manager.load("text", request)
                print("[qwen_vllm_server] startup text worker ready", flush=True)
            else:
                print("[qwen_vllm_server] startup auto-load disabled; waiting for explicit /load request", flush=True)
        except Exception:
            LOG.exception("startup Qwen worker auto-load failed; server will stay up so clients can retry explicit load")

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

    @app.post("/reload/text")
    def reload_text(request: LoadWorkerRequest) -> dict[str, Any]:
        return manager.load("text", request)

    @app.post("/wake/text")
    def wake_text(request: WakeWorkerRequest) -> dict[str, Any]:
        return manager.wake("text", request)

    @app.post("/load/vision")
    def load_vision(request: LoadWorkerRequest) -> dict[str, Any]:
        return manager.load("vision", request)

    @app.post("/reload/vision")
    def reload_vision(request: LoadWorkerRequest) -> dict[str, Any]:
        return manager.load("vision", request)

    @app.post("/wake/vision")
    def wake_vision(request: WakeWorkerRequest) -> dict[str, Any]:
        return manager.wake("vision", request)

    @app.post("/sleep")
    def sleep() -> dict[str, Any]:
        return manager.sleep()

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
    allocated_bytes = 0
    reserved_bytes = 0
    try:
        allocated_bytes = int(torch.cuda.memory_allocated(idx))
        reserved_bytes = int(torch.cuda.memory_reserved(idx))
    except Exception:
        allocated_bytes = 0
        reserved_bytes = 0
    return {
        "cuda": True,
        "device_index": idx,
        "device_name": torch.cuda.get_device_name(idx),
        "total_vram_gib": round(total_bytes / (1024 ** 3), 2),
        "free_vram_gib": round(free_bytes / (1024 ** 3), 2),
        "used_vram_gib": round((total_bytes - free_bytes) / (1024 ** 3), 2),
        "torch_allocated_gib": round(allocated_bytes / (1024 ** 3), 3),
        "torch_reserved_gib": round(reserved_bytes / (1024 ** 3), 3),
        "torch_unallocated_reserved_gib": round(max(0, reserved_bytes - allocated_bytes) / (1024 ** 3), 3),
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
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = create_app(host=args.host, port=args.port)
    print(
        f"[qwen_vllm_server] starting FastAPI/vLLM host={args.host} port={args.port} "
        f"auto_vision={DEFAULT_AUTO_LOAD_VISION_ON_STARTUP} gpu_util_default={DEFAULT_GPU_MEMORY_UTILIZATION} "
        f"max_model_len_default={DEFAULT_MAX_MODEL_LEN}",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
