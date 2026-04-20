#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

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
        "vllm is required for this worker. Install a recent vLLM build/new enough nightly."
    ) from exc


DEFAULT_MODEL_ID = "cyankiwi/Qwen3.5-4B-AWQ-4bit"
DEFAULT_MAX_MODEL_LEN = 32_768
DEFAULT_GPU_MEMORY_UTILIZATION = 0.98
DEFAULT_KV_CACHE_DTYPE = "turboquant_k8v4"
DEFAULT_ATTENTION_CONFIG = {"flash_attn_version": 2}
DEFAULT_COMPILATION_CONFIG = 3
DEFAULT_BATCH_LONG_PREFILL_THRESHOLD = 4096
DEFAULT_MM_LIMIT_PER_PROMPT = {"image": 8}

LOG = logging.getLogger("qwen_vllm_worker")


@dataclass(slots=True)
class WorkerConfig:
    model: str = DEFAULT_MODEL_ID
    tokenizer: Optional[str] = None
    trust_remote_code: bool = True
    tensor_parallel_size: int = 1
    dtype: str = "half"
    quantization: Optional[str] = "awq"
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    kv_cache_dtype: str = DEFAULT_KV_CACHE_DTYPE
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    scheduler_reserve_full_isl: bool = True
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: Optional[int] = None
    max_num_partial_prefills: int = 2
    max_long_partial_prefills: int = 1
    long_prefill_token_threshold: int = DEFAULT_BATCH_LONG_PREFILL_THRESHOLD
    compilation_config: int | dict[str, Any] = DEFAULT_COMPILATION_CONFIG
    attention_config: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_ATTENTION_CONFIG)
    )
    language_model_only: bool = True
    limit_mm_per_prompt: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_MM_LIMIT_PER_PROMPT)
    )
    mm_encoder_tp_mode: Optional[str] = None
    mm_processor_cache_type: Optional[str] = None
    mm_processor_cache_gb: Optional[int] = None
    mm_shm_cache_max_object_size_mb: Optional[int] = None
    enable_speculative: bool = False
    num_speculative_tokens: int = 1
    default_enable_thinking: bool = False
    seed: int = 0
    disable_log_stats: bool = False
    enable_logging_iteration_details: bool = False
    kv_cache_metrics: bool = False
    cudagraph_metrics: bool = False
    hf_token: Optional[str] = None

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
            "max_num_partial_prefills": self.max_num_partial_prefills,
            "max_long_partial_prefills": self.max_long_partial_prefills,
            "long_prefill_token_threshold": self.long_prefill_token_threshold,
            "compilation_config": self.compilation_config,
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
    """
    Resident in-process vLLM worker.

    Design notes:
    - One engine per worker instance, kept hot in-process.
    - No app-level batch size. Hand the full request list to vLLM and let its
      scheduler do continuous/in-flight batching under the live KV/VRAM limits.
    - Text-only mode uses the same checkpoint with `language_model_only=True`,
      which skips the vision encoder and frees memory for KV cache.
    - Multimodal mode keeps the vision path enabled and uses HF processor chat
      templating plus `multi_modal_data` payloads for vLLM generate().
    """

    def __init__(
        self,
        *,
        config: WorkerConfig,
        mode: str,
    ) -> None:
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
        return LLM(**kwargs)

    @property
    def tokenizer(self) -> Any:
        if self._processor is not None and getattr(self._processor, "tokenizer", None) is not None:
            return self._processor.tokenizer
        return self._tokenizer

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
        if self.mode != "text":
            raise RuntimeError("generate_text_batch() requires a text worker")
        if not prompts:
            return []

        sampling_params = (generation or GenerationConfig()).to_sampling_params()
        rendered_prompts = [
            self.build_text_prompt(prompt, system_prompt=system_prompt)
            for prompt in prompts
        ]
        outputs = self.llm.generate(rendered_prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
        return [self._to_response_record(output) for output in outputs]

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        generation: Optional[GenerationConfig] = None,
    ) -> ResponseRecord:
        return self.generate_text_batch(
            [prompt],
            system_prompt=system_prompt,
            generation=generation,
            use_tqdm=False,
        )[0]

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

    def generate_vision(
        self,
        prompt: str,
        *,
        images: Sequence[Any],
        system_prompt: Optional[str] = None,
        generation: Optional[GenerationConfig] = None,
    ) -> ResponseRecord:
        return self.generate_vision_batch(
            [{"prompt": prompt, "images": list(images)}],
            system_prompt=system_prompt,
            generation=generation,
            use_tqdm=False,
        )[0]

    def count_tokens(self, text: str) -> int:
        tok = self.tokenizer
        if tok is None:
            return 0
        try:
            encoded = tok(text, add_special_tokens=False)
            ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
            return len(ids or [])
        except Exception:
            return 0

    def warmup(self) -> None:
        if self.mode == "text":
            self.generate_text(
                "Reply with the single token: warm.",
                generation=GenerationConfig(max_new_tokens=8, temperature=0.0),
            )
        else:
            raise RuntimeError(
                "vision warmup needs a sample image; call generate_vision() explicitly"
            )

    def close(self) -> None:
        llm = getattr(self, "llm", None)
        if llm is not None:
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
    debug_scheduler: bool = False,
    enable_speculative: bool = False,
    num_speculative_tokens: int = 1,
    hf_token: Optional[str] = None,
) -> QwenVLLMWorker:
    cfg = WorkerConfig(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        language_model_only=True,
        enable_prefix_caching=not enable_speculative,
        enable_speculative=enable_speculative,
        num_speculative_tokens=num_speculative_tokens,
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
    debug_scheduler: bool = False,
    enable_speculative: bool = False,
    num_speculative_tokens: int = 1,
    hf_token: Optional[str] = None,
) -> QwenVLLMWorker:
    cfg = WorkerConfig(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        language_model_only=False,
        mm_encoder_tp_mode="data",
        mm_processor_cache_type="shm",
        mm_processor_cache_gb=4,
        mm_shm_cache_max_object_size_mb=256,
        enable_prefix_caching=not enable_speculative,
        enable_speculative=enable_speculative,
        num_speculative_tokens=num_speculative_tokens,
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


def destroy_worker(worker: Optional[QwenVLLMWorker]) -> None:
    if worker is None:
        return
    worker.close()


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


def _render_shared_system_prompt(repeat: int) -> str:
    base = (
        "You are a fast structured inference worker. "
        "Honor the instruction exactly, do not add extra framing, and keep outputs concise."
    )
    return "\n".join(base for _ in range(max(1, int(repeat))))


def _print_engine_summary(worker: QwenVLLMWorker) -> None:
    print("\n=== Worker Summary ===")
    print(f"mode: {worker.mode}")
    print(f"model: {worker.model_id}")
    print(json.dumps(_json_safe(asdict(worker.config)), indent=2))
    print("gpu:")
    print(json.dumps(_gpu_snapshot(), indent=2))


def _print_benchmark_result(
    *,
    label: str,
    started_at: float,
    ended_at: float,
    prompts: Sequence[str],
    responses: Sequence[ResponseRecord],
    shared_system_prompt: Optional[str],
) -> None:
    elapsed = max(ended_at - started_at, 1e-9)
    prompt_chars = sum(len(p) for p in prompts)
    prompt_tokens = sum(r.prompt_tokens for r in responses)
    cached_tokens = sum((r.cached_prompt_tokens or 0) for r in responses)
    output_tokens = sum(r.output_tokens for r in responses)
    print(f"\n=== {label} ===")
    print(f"requests: {len(prompts)}")
    print(f"elapsed_s: {elapsed:.3f}")
    print(f"req_per_s: {len(prompts) / elapsed:.3f}")
    print(f"prompt_chars_total: {prompt_chars}")
    print(f"prompt_tokens_total: {prompt_tokens}")
    print(f"cached_prompt_tokens_total: {cached_tokens}")
    print(f"output_tokens_total: {output_tokens}")
    print(f"output_tok_per_s: {output_tokens / elapsed:.3f}")
    if shared_system_prompt:
        print(f"shared_system_prompt_chars: {len(shared_system_prompt)}")
        tok = 0
        try:
            tok = worker_token_count_from_system_prompt(shared_system_prompt, responses, prompt_tokens)
        except Exception:
            tok = 0
        if tok:
            print(f"shared_system_prompt_tokens_est: {tok}")
    print("gpu_after:")
    print(json.dumps(_gpu_snapshot(), indent=2))
    print("sample_output_0:")
    if responses:
        print(responses[0].text.strip())


def worker_token_count_from_system_prompt(
    system_prompt: str,
    responses: Sequence[ResponseRecord],
    prompt_tokens_total: int,
) -> int:
    del responses
    del prompt_tokens_total
    return 0


def _run_text_debug(worker: QwenVLLMWorker, args: argparse.Namespace) -> None:
    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    shared_system_prompt = _render_shared_system_prompt(args.system_prefix_repeat)

    single_prompt = args.single_prompt or "State the purpose of this worker in one sentence."
    t0 = time.perf_counter()
    single = worker.generate_text(
        single_prompt,
        system_prompt=shared_system_prompt,
        generation=generation,
    )
    t1 = time.perf_counter()
    _print_benchmark_result(
        label="single_request",
        started_at=t0,
        ended_at=t1,
        prompts=[single_prompt],
        responses=[single],
        shared_system_prompt=shared_system_prompt,
    )

    batch_prompts = [
        f"Batch item {i}: summarize the payload id {i} in 8 words or fewer."
        for i in range(args.batch_size)
    ]
    t2 = time.perf_counter()
    batch = worker.generate_text_batch(
        batch_prompts,
        system_prompt=shared_system_prompt,
        generation=generation,
    )
    t3 = time.perf_counter()
    _print_benchmark_result(
        label="batched_requests",
        started_at=t2,
        ended_at=t3,
        prompts=batch_prompts,
        responses=batch,
        shared_system_prompt=shared_system_prompt,
    )

    print(
        "\nNote: for scheduler split/admission detail under live VRAM pressure, run with "
        "--debug-scheduler and watch the vLLM iteration logs."
    )


def _run_vision_debug(worker: QwenVLLMWorker, args: argparse.Namespace) -> None:
    if not args.image:
        raise SystemExit("vision mode requires --image /path/to/image")

    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    shared_system_prompt = _render_shared_system_prompt(args.system_prefix_repeat)
    image_path = Path(args.image)

    single_req = {
        "prompt": args.single_prompt or "Describe the image in one sentence.",
        "images": [image_path],
    }
    t0 = time.perf_counter()
    single = worker.generate_vision(
        single_req["prompt"],
        images=single_req["images"],
        system_prompt=shared_system_prompt,
        generation=generation,
    )
    t1 = time.perf_counter()
    _print_benchmark_result(
        label="single_request",
        started_at=t0,
        ended_at=t1,
        prompts=[single_req["prompt"]],
        responses=[single],
        shared_system_prompt=shared_system_prompt,
    )

    batch_reqs = [
        {
            "prompt": f"Batch item {i}: identify the main visible subject and one detail.",
            "images": [image_path],
        }
        for i in range(args.batch_size)
    ]
    t2 = time.perf_counter()
    batch = worker.generate_vision_batch(
        batch_reqs,
        system_prompt=shared_system_prompt,
        generation=generation,
    )
    t3 = time.perf_counter()
    _print_benchmark_result(
        label="batched_requests",
        started_at=t2,
        ended_at=t3,
        prompts=[req["prompt"] for req in batch_reqs],
        responses=batch,
        shared_system_prompt=shared_system_prompt,
    )

    print(
        "\nNote: for scheduler split/admission detail under live VRAM pressure, run with "
        "--debug-scheduler and watch the vLLM iteration logs."
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="In-process vLLM Qwen worker")
    parser.add_argument("--mode", choices=["text", "vision"], default="text")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--single-prompt", default="")
    parser.add_argument("--system-prefix-repeat", type=int, default=128)
    parser.add_argument("--image", default="")
    parser.add_argument("--debug-scheduler", action="store_true")
    parser.add_argument("--enable-speculative", action="store_true")
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose or args.debug_scheduler else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.enable_speculative:
        print(
            "speculative decoding is enabled. This is a latency profile; under heavy concurrency it can reduce effective batch size."
        )

    worker = create_qwen_worker(
        mode=args.mode,
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        debug_scheduler=args.debug_scheduler,
        enable_speculative=args.enable_speculative,
        num_speculative_tokens=args.num_speculative_tokens,
    )

    try:
        _print_engine_summary(worker)
        if args.mode == "text":
            _run_text_debug(worker, args)
        else:
            _run_vision_debug(worker, args)
    finally:
        worker.close()


if __name__ == "__main__":
    main()
