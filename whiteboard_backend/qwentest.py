#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
import gc
import itertools
import logging
import threading
import base64
import html as html_lib
from io import BytesIO
from difflib import SequenceMatcher
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

try:
    from transformers import Qwen3VLForConditionalGeneration  # type: ignore
except Exception:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import Qwen3_5ForConditionalGeneration  # type: ignore
except Exception:
    Qwen3_5ForConditionalGeneration = None

try:
    from vllm import LLM as VllmLLM, SamplingParams as VllmSamplingParams
    _HAS_VLLM = True
except Exception:
    VllmLLM = None
    VllmSamplingParams = None
    _HAS_VLLM = False



# bitsandbytes optional
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False


# ============================
# PATHS
# ============================
BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "ProccessedImages"
CLUSTER_RENDER_DIR = BASE_DIR / "ClusterRenders"
CLUSTER_MAP_DIR = BASE_DIR / "ClusterMaps"

OUT_DIR = BASE_DIR / "ClustersLabeled"
OUT_DEBUG_DIR = OUT_DIR / "debug"
SCHEMATIC_OUT_DIR = BASE_DIR / "LineLabels"
CACHE_DIR = BASE_DIR / "_path_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
IMG_CACHE_PATH = CACHE_DIR / "processed_png_index.json"
JSON_CACHE_PATH = CACHE_DIR / "processed_json_index.json"
_QWEN_STAGE_IO_GLOBAL_COUNTER = itertools.count(1)

SKIP_EXISTING_LABELS = True



# ============================
# MODEL CONFIG 
# ============================
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
FALLBACK_TEXT_MODEL_ID = os.environ.get("QWEN_TEXT_MODEL_FALLBACK", "Qwen/Qwen2.5-0.5B-Instruct")
C2_TEXT_MODEL_ID = os.environ.get("QWEN_TEXT_MODEL_ID", "Qwen/Qwen3.5-0.8B")
ACTION_PLANNER_TEXT_MODEL_ID = os.environ.get("QWEN_ACTION_MODEL_ID", C2_TEXT_MODEL_ID)
DIAGRAM_MULTIMODAL_MODEL_ID = os.environ.get(
    "QWEN_DIAGRAM_MULTIMODAL_MODEL_ID",
    os.environ.get("QWEN_MULTIMODAL_MODEL_ID", MODEL_ID),
)

QWEN_MIN_PIXELS = 256 * 28 * 28
QWEN_MAX_PIXELS = 512 * 28 * 28
QWEN_TEXT_BACKEND = str(os.environ.get("QWEN_TEXT_BACKEND", "vllm") or "vllm").strip().lower()
QWEN_VLLM_TP_SIZE = max(1, int(os.environ.get("QWEN_VLLM_TP_SIZE", "1") or 1))
QWEN_VLLM_GPU_MEMORY_UTILIZATION = float(os.environ.get("QWEN_VLLM_GPU_MEMORY_UTILIZATION", "0.9") or 0.9)
QWEN_VLLM_MAX_MODEL_LEN = max(512, int(os.environ.get("QWEN_VLLM_MAX_MODEL_LEN", "4096") or 4096))
QWEN_VLLM_ENFORCE_EAGER = str(os.environ.get("QWEN_VLLM_ENFORCE_EAGER", "0") or "0").strip().lower() not in {"0", "false", "no", "off", ""}



GPU_INDEX = 0
FORCE_CPU = False

QUANT_MODE = None # "4bit" | "8bit" | "none"
INT8_CPU_OFFLOAD = False 


# ============================
# BATCHING 
# ============================
def _env_int(name: str, default: int, min_value: int = 1) -> int:
    try:
        return max(int(min_value), int(os.getenv(name, str(default)) or default))
    except Exception:
        return max(int(min_value), int(default))


# Set to 1 => strict cluster-by-cluster (lowest VRAM).
# Set >1 => batching (faster, more VRAM).
BATCH_SIZE = _env_int("QWEN_BATCH_SIZE", 8, 1)

# ============================
# SPEED / MEMORY LEVERS
# ============================
# ONE composite image = two square panels side-by-side.
# PANEL_EDGE = PROC_LONGEST_EDGE//2
PROC_LONGEST_EDGE = _env_int("QWEN_PROC_LONGEST_EDGE", 600, 128) # composite max side;

SUGGESTION_LIMIT = 40
MAX_NEW_TOKENS = _env_int("QWEN_MAX_NEW_TOKENS", 70, 8)
CLUSTER_VISUAL_MAX_NEW_TOKENS = _env_int("QWEN_CLUSTER_VISUAL_MAX_NEW_TOKENS", 220, 32)
DIAGRAM_CLUSTER_VISUAL_MAX_NEW_TOKENS = _env_int("QWEN_DIAGRAM_CLUSTER_VISUAL_MAX_NEW_TOKENS", 420, 64)
SCHEMATIC_LINE_MATCH_MAX_NEW_TOKENS = _env_int("QWEN_SCHEMATIC_LINE_MATCH_MAX_NEW_TOKENS", 420, 64)
POSTFACTO_MATCH_MAX_NEW_TOKENS = _env_int("QWEN_POSTFACTO_MATCH_MAX_NEW_TOKENS", 700, 64)
DO_SAMPLE = False

RECT_THICKNESS_PX = 3
DTYPE = torch.float16
MIN_ACCEPT_CONF = 0.65

# Planner prompt/memory guardrails (env-tunable).
def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw not in ("0", "false", "no", "off", "")


def _is_qwen_35_08b_model(model_id: Any) -> bool:
    text = re.sub(r"\s+", "", str(model_id or "")).strip().lower()
    return "qwen3.5-0.8b" in text


def _should_use_vllm_for_text_model(model_id: Any) -> bool:
    if QWEN_TEXT_BACKEND in {"transformers", "hf"}:
        return False
    if QWEN_TEXT_BACKEND == "vllm":
        return _is_qwen_35_08b_model(model_id)
    return False


def _resolved_text_model_id(model_id: Any) -> str:
    requested = str(model_id or "").strip() or str(FALLBACK_TEXT_MODEL_ID)
    if _is_qwen_35_08b_model(requested) and Qwen3_5ForConditionalGeneration is None:
        print(
            f"[qwen][DBG] text_model_fallback requested={requested} "
            f"fallback={FALLBACK_TEXT_MODEL_ID} reason=transformers_missing_qwen3_5"
        )
        return str(FALLBACK_TEXT_MODEL_ID)
    return requested


def _resolved_multimodal_model_id(model_id: Any) -> str:
    requested = str(model_id or "").strip() or str(MODEL_ID)
    if _is_qwen_35_08b_model(requested):
        print(
            f"[qwen][DBG] multimodal_model_fallback requested={requested} "
            f"fallback={MODEL_ID} reason=text_model_not_multimodal"
        )
        return str(MODEL_ID)
    return requested


def _ensure_vllm_text_backend(model_id: Any) -> None:
    if not _HAS_VLLM:
        raise RuntimeError(
            f"vllm_not_installed_for_text_model:{model_id}. Install vllm or set QWEN_TEXT_BACKEND=transformers."
        )
    if FORCE_CPU or not torch.cuda.is_available():
        raise RuntimeError(f"vllm_text_backend_requires_cuda_for:{model_id}")


def _build_vllm_text_engine(model_id: str):
    _ensure_vllm_text_backend(model_id)
    engine = VllmLLM(
        model=model_id,
        tokenizer=model_id,
        trust_remote_code=True,
        tensor_parallel_size=int(QWEN_VLLM_TP_SIZE),
        dtype="auto",
        gpu_memory_utilization=float(QWEN_VLLM_GPU_MEMORY_UTILIZATION),
        max_model_len=int(QWEN_VLLM_MAX_MODEL_LEN),
        enforce_eager=bool(QWEN_VLLM_ENFORCE_EAGER),
    )
    return engine


def _get_vllm_tokenizer(llm: Any):
    tok = llm.get_tokenizer()
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def _apply_chat_template_text(tokenizer: Any, messages: List[Dict[str, Any]], *, thinking_enabled: bool = False) -> str:
    common_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=thinking_enabled, **common_kwargs)
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                chat_template_kwargs={"enable_thinking": thinking_enabled},
                **common_kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **common_kwargs)


def _make_vllm_sampling_params(
    *,
    temperature: float,
    max_new_tokens: int,
    thinking_enabled: bool = False,
    do_sample: bool = False,
):
    kwargs: Dict[str, Any] = {
        "max_tokens": int(max_new_tokens),
        "skip_special_tokens": False,
    }
    if thinking_enabled:
        kwargs.update(
            {
                "temperature": max(0.6, float(temperature)),
                "top_p": 0.95,
                "top_k": 20,
            }
        )
    elif do_sample:
        kwargs.update(
            {
                "temperature": max(0.0, float(temperature)),
                "top_p": 0.9,
            }
        )
    else:
        kwargs["temperature"] = 0.0
    return VllmSamplingParams(**kwargs)


def _vllm_generate_texts(
    *,
    llm: Any,
    prompts: List[str],
    temperature: float,
    max_new_tokens: int,
    thinking_enabled: bool = False,
    do_sample: bool = False,
) -> List[str]:
    if not prompts:
        return []
    sampling_params = _make_vllm_sampling_params(
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        thinking_enabled=bool(thinking_enabled),
        do_sample=bool(do_sample),
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    texts: List[str] = []
    for row in outputs:
        value = ""
        try:
            if getattr(row, "outputs", None):
                value = str(row.outputs[0].text or "")
        except Exception:
            value = ""
        texts.append(value)
    return texts


def _split_vllm_generated_text(raw_text: str, *, thinking_enabled: bool) -> Dict[str, Any]:
    raw_text = str(raw_text or "").replace("<|im_end|>", " ").replace("<|endoftext|>", " ").strip()
    if not thinking_enabled:
        return {
            "raw_text": raw_text,
            "thinking_text": "",
            "final_text": raw_text,
            "saw_close_tag": False,
        }
    head, marker, tail = raw_text.partition("</think>")
    if marker:
        return {
            "raw_text": raw_text,
            "thinking_text": (head + marker).strip(),
            "final_text": tail.strip(),
            "saw_close_tag": True,
        }
    return {
        "raw_text": raw_text,
        "thinking_text": raw_text,
        "final_text": "",
        "saw_close_tag": False,
    }


PLAN_PROMPT_CHAR_CAP = max(512, int(os.getenv("QWEN_PLAN_PROMPT_CHAR_CAP", "12000") or 12000))
PLAN_DO_SAMPLE = _env_flag("QWEN_PLAN_DO_SAMPLE", False)
PLAN_USE_CACHE = _env_flag("QWEN_PLAN_USE_CACHE", True)
PLAN_HEAVY_BATCH_TOKEN_BUDGET = max(1024, int(os.getenv("QWEN_PLAN_HEAVY_BATCH_TOKEN_BUDGET", "4096") or 4096))
PLAN_HEAVY_BATCH_PROMPT_CHAR_BUDGET = max(4096, int(os.getenv("QWEN_PLAN_HEAVY_BATCH_PROMPT_CHAR_BUDGET", "24000") or 24000))
QWEN_DISABLE_ASYNC_LOAD = _env_flag("QWEN_DISABLE_ASYNC_LOAD", True)
QWEN_DISABLE_ALLOCATOR_WARMUP = _env_flag("QWEN_DISABLE_ALLOCATOR_WARMUP", True)

# Prints for every completed cluster (even in batching).
PRINT_EVERY_CLUSTER = True

# For speed: don't empty_cache each cluster.
EMPTY_CACHE_EVERY = 0  # 0 disables

# Optional cheap reject to skip obvious garbage (OFF by default)
ENABLE_FAST_REJECT = False
MIN_COLORED_PIXELS = 25


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ============================
# LABEL CLEANING
# ============================
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "from", "by", "with", "without",
    "is", "are", "was", "were", "be", "been", "being", "as", "that", "this", "these", "those",
    "it", "its", "into", "over", "under", "up", "down", "left", "right", "top", "bottom",
    "label", "labels", "figure", "fig", "diagram", "image", "picture", "illustration",
}

def _clean_word(w: str) -> Optional[str]:
    if not w:
        return None
    s = w.strip()
    s = re.sub(r"^[^\w]+|[^\w]+$", "", s)
    if not s or len(s) == 1:
        return None
    low = s.lower()
    if low in STOPWORDS:
        return None
    if re.fullmatch(r"\d+", s):
        return None
    return s


# ----------------------------
# FAST INFERENCE CONFIG
# ----------------------------
USE_SDPA = True
USE_TORCH_COMPILE = True          # keep on only if your hot path is shape-stable
TRY_STATIC_CACHE = False          # transformers StaticLayer cache has been unstable here
COMPILE_MODE = "reduce-overhead"
PAD_TO_MULTIPLE_OF = 8            # reduces shape churn a bit without wasting too much padding


def _qwen_attn_impl() -> Optional[str]:
    if FORCE_CPU or not torch.cuda.is_available() or not USE_SDPA:
        return None
    return "sdpa"


def _processor_batch(
    processor,
    *,
    text=None,
    images=None,
):
    kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
    }

    if text is not None:
        kwargs["text"] = text
        if PAD_TO_MULTIPLE_OF:
            kwargs["pad_to_multiple_of"] = int(PAD_TO_MULTIPLE_OF)

    if images is not None:
        kwargs["images"] = images

    try:
        return processor(**kwargs)
    except TypeError:
        kwargs.pop("pad_to_multiple_of", None)
        return processor(**kwargs)


def _move_inputs_to_device(inputs, device: torch.device):
    img_dtype = _img_tensor_dtype(device)

    for k, v in list(inputs.items()):
        if not torch.is_tensor(v):
            continue
        if k == "pixel_values":
            inputs[k] = v.to(device=device, dtype=img_dtype, non_blocking=True)
        else:
            inputs[k] = v.to(device=device, non_blocking=True)

    return inputs


def _maybe_enable_qwen_fast_inference(
    model,
    device: torch.device,
    *,
    allow_compile: bool = True,
):
    flags = {
        "attn_implementation": None,
        "static_cache": False,
        "compiled": False,
        "compile_fullgraph": None,
    }

    if device.type != "cuda":
        setattr(model, "_qwen_fast_flags", flags)
        return model

    try:
        if USE_SDPA and hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("sdpa")
            flags["attn_implementation"] = "sdpa"
    except Exception as e:
        print(f"[warn] set_attn_implementation(sdpa) failed: {e}")

    if TRY_STATIC_CACHE and allow_compile:
        try:
            model.generation_config.cache_implementation = "static"
            flags["static_cache"] = True
        except Exception as e:
            print(f"[warn] static cache unavailable: {e}")

    if USE_TORCH_COMPILE and allow_compile and flags["static_cache"]:
        try:
            model.forward = torch.compile(
                model.forward,
                mode=COMPILE_MODE,
                fullgraph=True,
            )
            flags["compiled"] = True
            flags["compile_fullgraph"] = True
        except Exception:
            try:
                model.forward = torch.compile(
                    model.forward,
                    mode=COMPILE_MODE,
                    fullgraph=False,
                )
                flags["compiled"] = True
                flags["compile_fullgraph"] = False
            except Exception as e:
                print(f"[warn] torch.compile disabled: {e}")
                try:
                    model.generation_config.cache_implementation = None
                except Exception:
                    pass
                flags["static_cache"] = False

    setattr(model, "_qwen_fast_flags", flags)
    return model


def _is_static_cache_batch_error(err: BaseException) -> bool:
    msg = f"{type(err).__name__}: {err}".lower()
    return ("staticlayer" in msg) and ("max_batch_size" in msg)


def _downgrade_cache_mode_for_runtime(model) -> None:
    changed: List[str] = []

    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        try:
            if getattr(gen_cfg, "cache_implementation", None) != "dynamic":
                gen_cfg.cache_implementation = "dynamic"
                changed.append("cache_implementation=dynamic")
        except Exception:
            try:
                gen_cfg.cache_implementation = None
                changed.append("cache_implementation=None")
            except Exception:
                pass

    try:
        if hasattr(model, "config"):
            model.config.use_cache = False
            changed.append("model.config.use_cache=False")
    except Exception:
        pass

    for attr in ("_cache", "_past_key_values", "past_key_values"):
        try:
            if hasattr(model, attr):
                setattr(model, attr, None)
        except Exception:
            pass

    flags = getattr(model, "_qwen_fast_flags", None)
    if isinstance(flags, dict):
        flags["static_cache"] = False
        flags["compiled"] = False
        flags["compile_fullgraph"] = None

    if changed:
        print(f"[warn] qwen cache fallback applied ({', '.join(changed)})")


def _safe_generate(
    model,
    *,
    inputs: Dict[str, Any],
    gen_kwargs: Dict[str, Any],
):
    try:
        return model.generate(**inputs, **gen_kwargs)
    except Exception as e:
        if not _is_static_cache_batch_error(e):
            raise
        print(f"[warn] qwen static cache mismatch, retrying without cache: {e}")
        _downgrade_cache_mode_for_runtime(model)
        retry_kwargs = dict(gen_kwargs or {})
        retry_kwargs["use_cache"] = False
        return model.generate(**inputs, **retry_kwargs)


def _generation_special_token_ids(model, processor) -> Tuple[List[int], int]:
    eos_ids: List[int] = []
    pad_id = 0

    tok = getattr(processor, "tokenizer", None)
    gen_cfg = getattr(model, "generation_config", None)

    eos_raw = None
    if gen_cfg is not None:
        eos_raw = getattr(gen_cfg, "eos_token_id", None)
        pad_id = int(getattr(gen_cfg, "pad_token_id", 0) or 0)
    if eos_raw is None and tok is not None:
        eos_raw = getattr(tok, "eos_token_id", None)
    if not pad_id and tok is not None:
        pad_id = int(getattr(tok, "pad_token_id", 0) or 0)

    if isinstance(eos_raw, (list, tuple, set)):
        for one in eos_raw:
            try:
                eos_ids.append(int(one))
            except Exception:
                pass
    elif eos_raw is not None:
        try:
            eos_ids.append(int(eos_raw))
        except Exception:
            pass

    eos_ids = list(dict.fromkeys([int(x) for x in eos_ids if int(x) >= 0]))
    if not eos_ids and pad_id > 0:
        eos_ids = [int(pad_id)]
    if pad_id <= 0 and eos_ids:
        pad_id = int(eos_ids[0])
    return eos_ids, int(pad_id)


def _sample_next_tokens_from_logits(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    if not do_sample:
        return torch.argmax(logits, dim=-1)

    temp = max(1e-5, float(temperature or 1.0))
    probs = torch.softmax(logits / temp, dim=-1)
    top_p = max(1e-5, min(1.0, float(top_p or 1.0)))
    if top_p < 0.999:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
        denom = sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sorted_probs = sorted_probs / denom
        sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(-1, sampled_sorted).squeeze(-1)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _select_past_key_values_batch(past_key_values: Any, keep_indices: torch.Tensor) -> Any:
    if past_key_values is None:
        return None

    if hasattr(past_key_values, "batch_select_indices"):
        past_key_values.batch_select_indices(keep_indices)
        return past_key_values

    if isinstance(past_key_values, tuple):
        new_layers = []
        for layer in past_key_values:
            if isinstance(layer, tuple):
                new_layer = []
                for tensor in layer:
                    if torch.is_tensor(tensor) and tensor.dim() > 0 and tensor.shape[0] >= int(keep_indices.numel()):
                        new_layer.append(tensor.index_select(0, keep_indices))
                    else:
                        new_layer.append(tensor)
                new_layers.append(tuple(new_layer))
            else:
                new_layers.append(layer)
        return tuple(new_layers)

    return past_key_values


def _slice_generation_model_kwargs_batch(
    model_kwargs: Dict[str, Any],
    *,
    keep_indices: torch.Tensor,
    batch_size_before: int,
) -> Dict[str, Any]:
    out = dict(model_kwargs or {})
    for key, value in list(out.items()):
        if key == "cache_position":
            continue
        if key == "past_key_values":
            out[key] = _select_past_key_values_batch(value, keep_indices)
            continue
        if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == int(batch_size_before):
            out[key] = value.index_select(0, keep_indices)
    return out


def _generate_with_rolling_cache_release(
    *,
    model,
    processor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    debug_label: str,
) -> List[str]:
    batch_size = int(input_ids.shape[0])
    if batch_size <= 0:
        return []

    eos_ids, _pad_id = _generation_special_token_ids(model, processor)
    eos_id_set = set(int(x) for x in eos_ids)
    tok = getattr(processor, "tokenizer", None)
    decode_fn = getattr(processor, "decode", None)
    if not callable(decode_fn) and tok is not None:
        decode_fn = getattr(tok, "decode", None)
    active_input_ids = input_ids
    active_indices = torch.arange(batch_size, device=input_ids.device, dtype=torch.long)
    generated_token_rows: List[List[int]] = [[] for _ in range(batch_size)]

    model_kwargs: Dict[str, Any] = {
        "attention_mask": attention_mask,
        "use_cache": True,
        "cache_position": torch.arange(int(input_ids.shape[1]), device=input_ids.device, dtype=torch.long),
    }
    next_input_ids = active_input_ids

    for step in range(int(max_new_tokens)):
        prepared_inputs = model.prepare_inputs_for_generation(
            next_input_ids,
            is_first_iteration=(step == 0),
            **model_kwargs,
        )
        outputs = model(**prepared_inputs, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = _sample_next_tokens_from_logits(
            next_token_logits,
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
        )

        active_indices_cpu = active_indices.detach().cpu().tolist()
        next_tokens_cpu = next_tokens.detach().cpu().tolist()
        finished_positions: List[int] = []
        for pos, orig_idx in enumerate(active_indices_cpu):
            tok_id = int(next_tokens_cpu[pos])
            generated_token_rows[int(orig_idx)].append(tok_id)
            if tok_id in eos_id_set:
                finished_positions.append(int(pos))

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=False,
            num_new_tokens=1,
        )

        if len(finished_positions) >= len(active_indices_cpu):
            break

        if finished_positions:
            keep_positions = [i for i in range(len(active_indices_cpu)) if i not in set(finished_positions)]
            keep_indices = torch.tensor(keep_positions, device=active_indices.device, dtype=torch.long)
            batch_before = int(active_indices.shape[0])
            active_indices = active_indices.index_select(0, keep_indices)
            next_tokens = next_tokens.index_select(0, keep_indices)
            model_kwargs = _slice_generation_model_kwargs_batch(
                model_kwargs,
                keep_indices=keep_indices,
                batch_size_before=batch_before,
            )
            try:
                del outputs
            except Exception:
                pass
            try:
                del prepared_inputs
            except Exception:
                pass
            try:
                del next_token_logits
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            print(
                f"[qwen][DBG] {debug_label} rolling_cache_release step={step} "
                f"finished={len(finished_positions)} active_left={int(active_indices.shape[0])} vram={_vram_str()}"
            )

        next_input_ids = next_tokens.unsqueeze(-1)
        try:
            del outputs
        except Exception:
            pass
        try:
            del prepared_inputs
        except Exception:
            pass
        try:
            del next_token_logits
        except Exception:
            pass
        if int(active_indices.shape[0]) <= 0:
            break

    decoded_rows: List[str] = []
    for token_ids in generated_token_rows:
        if token_ids and callable(decode_fn):
            decoded_rows.append(str(decode_fn(token_ids, skip_special_tokens=True)))
        else:
            decoded_rows.append("")
    return decoded_rows


def _noop_caching_allocator_warmup(*args, **kwargs) -> None:
    return None


@contextmanager
def _qwen_safe_loader_context():
    orig_async_load = os.environ.get("HF_DEACTIVATE_ASYNC_LOAD")
    orig_allocator_warmup = None
    modeling_utils = None

    try:
        if QWEN_DISABLE_ASYNC_LOAD:
            os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

        if QWEN_DISABLE_ALLOCATOR_WARMUP:
            try:
                import transformers.modeling_utils as modeling_utils
                orig_allocator_warmup = getattr(modeling_utils, "caching_allocator_warmup", None)
                if orig_allocator_warmup is not None:
                    modeling_utils.caching_allocator_warmup = _noop_caching_allocator_warmup
            except Exception as e:
                print(f"[warn] failed to disable transformers caching allocator warmup: {e}")

        yield
    finally:
        if modeling_utils is not None and orig_allocator_warmup is not None:
            try:
                modeling_utils.caching_allocator_warmup = orig_allocator_warmup
            except Exception:
                pass

        if orig_async_load is None:
            os.environ.pop("HF_DEACTIVATE_ASYNC_LOAD", None)
        else:
            os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = orig_async_load


# ============================
# DISCOVERY / INDEXING
# ============================
_FULL_IMG_RE = re.compile(r"processed_(\d+)\.png$", re.IGNORECASE)
_FULL_JSON_RE = re.compile(r"processed_(\d+)\.json$", re.IGNORECASE)
_CLUSTER_DIR_RE = re.compile(r"processed_(\d+)$", re.IGNORECASE)

def find_indices_from_cluster_maps() -> List[int]:
    out: List[int] = []
    if not CLUSTER_MAP_DIR.exists():
        return out
    for d in CLUSTER_MAP_DIR.iterdir():
        if not d.is_dir():
            continue
        m = _CLUSTER_DIR_RE.match(d.name)
        if not m:
            continue
        if (d / "clusters.json").exists():
            out.append(int(m.group(1)))
    out.sort()
    return out

def cluster_map_path_for_idx(idx: int) -> Path:
    return CLUSTER_MAP_DIR / f"processed_{idx}" / "clusters.json"

def cluster_renders_dir_for_idx(idx: int) -> Path:
    return CLUSTER_RENDER_DIR / f"processed_{idx}"

def parse_colour_hint_from_filename(name: str) -> Optional[str]:
    n = (name or "").lower()
    for c in ["white","yellow","orange","cyan","green","magenta","red","blue","purple","gray","black"]:
        if c in n:
            return c
    return None

def _save_cache(path: Path, mapping: Dict[str, str]) -> None:
    payload = {"created_utc": time.time(), "root": str(PROCESSED_DIR), "map": mapping}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _build_processed_index(regex: re.Pattern, suffix_hint: str) -> Dict[str, str]:
    t0 = time.time()
    mapping: Dict[str, str] = {}
    seen = 0
    matched = 0

    for root, _, files in os.walk(PROCESSED_DIR):
        for fn in files:
            seen += 1
            if (seen % 20000) == 0:
                print(f"[scan] {suffix_hint} files seen={seen} matched={matched} ({time.time()-t0:.1f}s)")
            m = regex.match(fn)
            if not m:
                continue
            idx = str(int(m.group(1)))
            p = str(Path(root) / fn)

            if idx in mapping:
                prev = Path(mapping[idx]).name.lower()
                cur = fn.lower()
                if prev.startswith("proccessed_") and cur.startswith("processed_"):
                    mapping[idx] = p
            else:
                mapping[idx] = p
            matched += 1

    print(f"[scan] {suffix_hint} index built: entries={len(mapping)} files_seen={seen} ({time.time()-t0:.1f}s)")
    return mapping

def _load_cache(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """
    Returns (map, meta). meta may contain 'root', 'created_utc', etc.
    """
    if not path.exists():
        return {}, {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("map"), dict):
            m = {str(k): str(v) for k, v in obj["map"].items()}
            return m, obj
    except Exception:
        pass
    return {}, {}


def _prune_missing_files(mapping: dict[str, str], *, must_be_file: bool = True) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in mapping.items():
        p = Path(v)
        try:
            if must_be_file:
                if p.is_file():
                    out[k] = v
            else:
                if p.exists():
                    out[k] = v
        except Exception:
            # Path() can throw on weird Windows path edge cases; treat as invalid.
            pass
    return out


def ensure_indexes() -> tuple[dict[str, str], dict[str, str]]:
    # Load (map, meta)
    img_map, img_meta = _load_cache(IMG_CACHE_PATH)
    json_map, json_meta = _load_cache(JSON_CACHE_PATH)

    # If cache was built for a different root, rebuild (prevents stale-project issues)
    cur_root = str(PROCESSED_DIR.resolve())
    img_root = str(img_meta.get("root", ""))
    json_root = str(json_meta.get("root", ""))

    # Prune missing entries first (cheap)
    if img_map:
        img_map = _prune_missing_files(img_map, must_be_file=True)
    if json_map:
        json_map = _prune_missing_files(json_map, must_be_file=True)

    # Rebuild conditions
    def need_rebuild(map_now: dict[str, str], root_now: str) -> bool:
        if not map_now:
            return True
        if root_now and (Path(root_now).resolve().as_posix() != Path(cur_root).resolve().as_posix()):
            return True
        return False

    if need_rebuild(img_map, img_root):
        print("[scan] rebuilding PNG index cache...")
        img_map = _build_processed_index(_FULL_IMG_RE, "png")
        _save_cache(IMG_CACHE_PATH, img_map)

    if need_rebuild(json_map, json_root):
        print("[scan] rebuilding JSON index cache...")
        json_map = _build_processed_index(_FULL_JSON_RE, "json")
        _save_cache(JSON_CACHE_PATH, json_map)

    return img_map, json_map


def load_candidate_labels(idx: int, json_index: Dict[str, str]) -> Tuple[List[str], str]:
    p = json_index.get(str(idx))
    if not p:
        return [], ""
    path = Path(p)
    if not path.exists():
        return [], ""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [], ""

    # NEW: base_context comes from processed_<idx>.json (added by your other script)
    base_context = data.get("base_context")
    if not isinstance(base_context, str):
        base_context = ""
    base_context = base_context.strip()

    words = data.get("words") or []
    out: List[str] = []
    seen = set()

    for item in words:
        t = item.get("text") if isinstance(item, dict) else None
        if not isinstance(t, str):
            continue
        c = _clean_word(t)
        if not c:
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)

    return out, base_context



# ============================
# Image helpers (COMPOSITE)
# ============================
def _fit_into_square(img: Image.Image, side: int) -> Tuple[Image.Image, float, int, int, int, int]:
    img = img.convert("RGB")
    ow, oh = img.size
    if ow <= 0 or oh <= 0:
        square = Image.new("RGB", (side, side), (0, 0, 0))
        return square, 1.0, side, side, 0, 0

    scale = min(side / ow, side / oh)
    rw = max(1, int(round(ow * scale)))
    rh = max(1, int(round(oh * scale)))
    resized = img.resize((rw, rh), resample=Image.BICUBIC)

    square = Image.new("RGB", (side, side), (0, 0, 0))
    pad_x = (side - rw) // 2
    pad_y = (side - rh) // 2
    square.paste(resized, (pad_x, pad_y))

    return square, float(scale), rw, rh, pad_x, pad_y

def _draw_red_rect_pil(img: Image.Image, bbox_xyxy: List[int], thickness: int) -> None:
    if thickness <= 0:
        return
    w, h = img.size
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]

    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))

    x2 = max(0, min(w - 1, x1 - 1))
    y2 = max(0, min(h - 1, y1 - 1))
    if x2 <= x0 or y2 <= y0:
        return

    draw = ImageDraw.Draw(img)
    for t in range(int(thickness)):
        draw.rectangle([x0 - t, y0 - t, x2 + t, y2 + t], outline=(255, 0, 0))

def _make_composite(left_sq: Image.Image, right_sq: Image.Image) -> Image.Image:
    side = left_sq.size[0]
    canvas = Image.new("RGB", (side * 2, side), (0, 0, 0))
    canvas.paste(left_sq, (0, 0))
    canvas.paste(right_sq, (side, 0))

    draw = ImageDraw.Draw(canvas)
    x = side
    draw.line([(x - 1, 0), (x - 1, side)], fill=(80, 80, 80), width=1)
    draw.line([(x, 0), (x, side)], fill=(80, 80, 80), width=1)
    return canvas

def _bbox_norm_xyxy(bbox_xyxy: List[int], w: int, h: int) -> List[float]:
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    return [x0 / max(1, w), y0 / max(1, h), x1 / max(1, w), y1 / max(1, h)]

def _scale_bbox_xyxy(bbox_xyxy: List[int], scale: float) -> List[int]:
    return [int(round(v * scale)) for v in bbox_xyxy]

def _clamp_bbox_xyxy(bbox_xyxy: List[int], w: int, h: int) -> List[int]:
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    return [x0, y0, x1, y1]

def build_label_refine_prompt(base_context: str, raw_candidates: List[str]) -> str:
    bc = (base_context or "").strip()
    if not bc:
        bc = "unknown"

    # keep it short to avoid garbage / token spam
    cand = raw_candidates[:80]
    cand_text = ", ".join(cand) if cand else "(none)"

    return (
        f"You are given a noisy list of candidate words, describing parts of a(n) {base_context} - MAIN .\n\n"
        f"Return a NEW list of OBJECT COMPONENTS / PARTS that the {base_context} would be built from.\n"
        "Fill in gaps in the pruned list so that it becomes a set of ALL objects that build up MAIN.\n\n"
        "Fill the new list with as many unique objects as possible"
        "Examples:\n"
        "- face -> f each face part\n"
        "- eukaryotic cell -> all organels\n"
        "- car -> wheel, tire, rim, door...n\n"
        f"RAW CANDIDATES: {cand_text}\n\n"
        "Include a very consise description of how each object is represented and differentiated VISUALLY"
        "RULES:\n"
        "- Output ONLY JSON\n"
        "- Example JSON in format:{"
        "{\"labels\": [{label : description, label : description}]}\n"
        "- labels must be short strings (1-4 words max)\n"
        "- no duplicates\n"
    )


def _parse_refined_labels(text_out: str) -> List[str]:
    obj = extract_json_object(text_out)

    if isinstance(obj, dict):
        labels = obj.get("labels")

        if isinstance(labels, list):
            out: List[str] = []
            seen = set()

            for item in labels:
                name = None

                if isinstance(item, str):
                    name = item
                elif isinstance(item, dict):
                    n = item.get("name")
                    if isinstance(n, str):
                        name = n

                if not isinstance(name, str):
                    continue

                c = _clean_word(name)
                if not c:
                    continue
                k = c.lower()
                if k in seen:
                    continue
                seen.add(k)
                out.append(c)

            if out:
                return out

    # fallback: try comma/newline split if model didn't follow JSON
    out: List[str] = []
    seen = set()
    for part in re.split(r"[,;\n]+", (text_out or "").strip()):
        c = _clean_word(part)
        if not c:
            continue
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out



def refine_candidate_labels_with_qwen(
    model,
    processor,
    device,
    base_context: str,
    raw_candidates: List[str],
) -> List[str]:
    SYSTEM_MSG = (
        "You are a label-refinement engine.\n"
        "You produce component/part labels, not generic category names.\n"
        "Return only JSON.\n"
    )

    prompt_text = build_label_refine_prompt(base_context, raw_candidates)

    conv = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]
    prompt_str = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    # slightly more tokens than the main run because it must output a list
    gen_kwargs_refine = dict(
        max_new_tokens=140,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

    try:
        inputs = _processor_batch(
            processor,
            text=[prompt_str],
        )
        inputs = _move_inputs_to_device(inputs, device)

        with torch.inference_mode():
            out_ids = _safe_generate(model, inputs=inputs, gen_kwargs=gen_kwargs_refine)

        if "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
            prompt_len = int(inputs["input_ids"].shape[1])
            gen_only_ids = out_ids[:, prompt_len:]
        else:
            gen_only_ids = out_ids

        text_out = processor.batch_decode(gen_only_ids, skip_special_tokens=True)[0]
        refined = _parse_refined_labels(text_out)

        # if model produced nothing useful -> fallback to original candidates
        if not refined:
            return raw_candidates[:SUGGESTION_LIMIT]

        return refined

    except Exception as e:
        print(f"[warn] refine_candidate_labels_with_qwen failed: {e}")
        return raw_candidates[:SUGGESTION_LIMIT]
    

# ============================
# Prompt + parse (ONE IMAGE)
# ============================
def build_prompt(
    colour_hint: Optional[str],
    suggestions: List[str],
    base_context: str,
) -> str:

    colour_txt = colour_hint if colour_hint else "unknown"
    sug = suggestions[:SUGGESTION_LIMIT]

    # Keep suggestions readable but short
    if sug:
        sug_text = ",".join(sug)
    else:
        sug_text = "(none)"

    return (
        "Input : Two images merged horizontally\n"
        f"- LEFT: Cropped box with with coloured {colour_txt} main object and grayscaled surroundings\n"
        f"- RIGHT: A(n) {base_context}, LEFT crop from there\n\n"
        f"Visually charecterize the features of LEFT's object and its placement in the {base_context} based on foreground.\n"
        "Look at RIGHT and LEFTS FOREGROUND and link the crop to WIDER surroundings (their shapes, colours) for RELATIVE context \n"
        "Based on LEFT's full semantic profile label it - what it is\n\n"
        "Return ONLY JSON with REQUIRED keys:\n"
        "{"
        "\"label\":string|null,"
        "\"full_visual_LEFT\": string,"
        "\"LEFT_SURROUNDINGS\": string,"
        "\"geometry_keywords\": strings"
        "}\n"
    )

def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    if not isinstance(text, str):
        return []

    s = text.strip()
    if not s:
        return []

    out: List[Dict[str, Any]] = []
    seen = set()

    n = len(s)
    for start in range(n):
        if s[start] != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for end in range(start, n):
            ch = s[end]
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = s[start:end + 1]
                    if chunk in seen:
                        break
                    seen.add(chunk)
                    try:
                        obj = json.loads(chunk)
                        if isinstance(obj, dict):
                            out.append(obj)
                    except Exception:
                        pass
                    break

    return out

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    objs = extract_json_objects(text)
    return objs[0] if objs else None

def _ensure_cluster_containers(results: Dict[str, Any]) -> None:
    if "clusters" not in results or not isinstance(results.get("clusters"), dict):
        results["clusters"] = {}
    if "cluster_order" not in results or not isinstance(results.get("cluster_order"), list):
        results["cluster_order"] = []

def _store_cluster_result(results: Dict[str, Any], cluster_key: str, payload: Dict[str, Any]) -> None:
    _ensure_cluster_containers(results)
    clusters = results["clusters"]
    order = results["cluster_order"]

    if cluster_key not in clusters:
        order.append(cluster_key)
    clusters[cluster_key] = payload

def build_cluster_visual_prompt(
    colour_hint: Optional[str],
    base_context: str,
) -> str:
    colour_txt = colour_hint if colour_hint else "unknown"
    bc = (base_context or "").strip() or "unknown"

    return (
        "Input : Two images merged horizontally\n"
        f"- LEFT: Cropped box with with coloured {colour_txt} main object and grayscaled surroundings\n"
        f"- RIGHT: A(n) {base_context}, LEFT crop from there\n\n"
        f"Visually charecterize the features of LEFT's object and its placement in the {base_context} based on foreground.\n"
        "Look at RIGHT and LEFTS FOREGROUND and link the crop to WIDER surroundings (their shapes, colours) for RELATIVE context \n"
        "Based on LEFT's full semantic profile label it - what it is\n\n"
        "Return ONLY JSON with REQUIRED keys:\n"
        "{"
        "\"label\":string|null,"
        "\"full_visual_LEFT\": string,"
        "\"LEFT_SURROUNDINGS\": string,"
        "\"geometry_keywords\": strings"
        "}\n"
    )


def _json_unescape_loose(value: str) -> str:
    s = str(value or "")
    if not s:
        return ""
    try:
        return json.loads('"' + s + '"')
    except Exception:
        return (
            s.replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t")
            .replace("\\/", "/")
            .strip()
        )


def _extract_jsonish_string_field(text: str, field_name: str) -> str:
    if not isinstance(text, str) or not field_name:
        return ""

    exact = re.search(
        rf'"{re.escape(field_name)}"\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if exact:
        return _json_unescape_loose(exact.group(1)).strip()

    partial = re.search(
        rf'"{re.escape(field_name)}"\s*:\s*"([^\n\r]*)',
        text,
        flags=re.IGNORECASE,
    )
    if partial:
        return _json_unescape_loose(partial.group(1)).strip().rstrip(",")

    return ""


def _extract_jsonish_keywords_field(text: str, field_name: str) -> List[str]:
    if not isinstance(text, str) or not field_name:
        return []

    block = re.search(
        rf'"{re.escape(field_name)}"\s*:\s*(\[[\s\S]*?\])',
        text,
        flags=re.IGNORECASE,
    )
    if block:
        try:
            arr = json.loads(block.group(1))
        except Exception:
            arr = None
        if isinstance(arr, list):
            out: List[str] = []
            for item in arr:
                if not isinstance(item, str):
                    continue
                cleaned = _clean_short_label(item)
                if cleaned:
                    out.append(cleaned)
            if out:
                return out[:12]

    inline = _extract_jsonish_string_field(text, field_name)
    if not inline:
        return []

    out = []
    seen = set()
    for part in re.split(r"[,;/|]+", inline):
        cleaned = _clean_short_label(part)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out[:12]


def _parse_cluster_visual(text_out: str) -> Dict[str, Any]:
    obj = extract_json_object(text_out or "")
    fv = obj.get("full_visual_LEFT") if isinstance(obj, dict) else None
    ls = obj.get("LEFT_SURROUNDINGS") if isinstance(obj, dict) else None
    gk = obj.get("geometry_keywords") if isinstance(obj, dict) else None
    label_guess = obj.get("label") if isinstance(obj, dict) else None

    out = {
        "label_guess": _clean_short_label(label_guess) if isinstance(label_guess, str) else "",
        "full_visual_LEFT": fv if isinstance(fv, str) else "",
        "LEFT_SURROUNDINGS": ls if isinstance(ls, str) else "",
        "geometry_keywords": [],
    }

    if isinstance(gk, list):
        kk = []
        for it in gk:
            if isinstance(it, str):
                s = _clean_short_label(it)
                if s:
                    kk.append(s)
        out["geometry_keywords"] = kk[:12]

    if not out["label_guess"]:
        out["label_guess"] = _clean_short_label(_extract_jsonish_string_field(text_out or "", "label")) or ""
    if not out["full_visual_LEFT"]:
        out["full_visual_LEFT"] = _extract_jsonish_string_field(text_out or "", "full_visual_LEFT")
    if not out["LEFT_SURROUNDINGS"]:
        out["LEFT_SURROUNDINGS"] = _extract_jsonish_string_field(text_out or "", "LEFT_SURROUNDINGS")
    if not out["geometry_keywords"]:
        out["geometry_keywords"] = _extract_jsonish_keywords_field(text_out or "", "geometry_keywords")

    return out


def build_postfacto_label_match_prompt(
    base_context: str,
    refined_labels: List[str],
    cluster_visual_map: Dict[str, Dict[str, Any]],
) -> str:
    bc = (base_context or "").strip() or "unknown"

    # compact the payload
    labels = [str(x).strip() for x in (refined_labels or []) if str(x).strip()]
    labels = labels[:120]

    clusters_compact = {}
    for k, v in (cluster_visual_map or {}).items():
        if not isinstance(v, dict):
            continue
        clusters_compact[str(k)] = {
            "stage1_label_guess": str(v.get("label_guess", "") or "")[:120],
            "geometry_keywords": v.get("geometry_keywords", []) if isinstance(v.get("geometry_keywords"), list) else [],
            "full_visual_LEFT": str(v.get("full_visual_LEFT", "") or "")[:600],
            "LEFT_SURROUNDINGS": str(v.get("LEFT_SURROUNDINGS", "") or "")[:600],
        }

    return (
        f"BASE CONTEXT: {bc}\n\n"
        "You are given:\n"
        "- A list of refined component labels (things that build up something - cell, face, ex)\n"
        "- A map objects, described ONLY by visual characteristics\n\n"
        "Your job:\n"
        "- For EACH object, choose the single BEST label from the refined label list.\n"
        "- Match by how that label would LOOK visually (its known characteristics).\n"
        "- If none fit, output null.\n\n"
        "Output JSON ONLY with schema:\n"
        "{"
        "\"matches\":["
        "{\"object_key\":\"...\",\"label\":string|null,\"confidence\":0.0,\"reason\":\"...\"}"
        "]"
        "}\n\n"
        "Rules:\n"
        "- every label is used only ONCE, ot at least for ONE TYPE of visual description\n"
        "- use ONLY provided labels - no inventing, every label HAS TO find a partner description"
        "- label must be exactly one of refined_labels or null\n"
        "- confidence is 0..1\n"
        "REFINED LABELS:\n"
        f"{json.dumps(labels, ensure_ascii=False)}\n\n"
        "OBJECT VISUAL MAP, INDEXED:\n"
        f"{json.dumps(clusters_compact, ensure_ascii=False)}\n\n"
    )


def _match_hint_lines_for_labels(refined_labels: List[str]) -> List[str]:
    label_set = {str(x or "").strip().lower() for x in (refined_labels or []) if str(x or "").strip()}
    hints: List[str] = []

    def _has(*names: str) -> bool:
        return any(str(name).strip().lower() in label_set for name in names)

    if _has("nucleus", "nucleolus", "nuclear envelope", "nuclear pores"):
        hints.append(
            "- Distinguish nuclear family carefully: nucleus is the large round/oval compartment, nucleolus is a dense spot inside it, nuclear envelope is the boundary/ring around it, nuclear pores are tiny openings/dots on that boundary."
        )
    if _has("rough er", "smooth er", "golgi", "vesicles"):
        hints.append(
            "- Distinguish membrane systems carefully: rough ER is an extended sheet/tube network often near nucleus, smooth ER is smoother tubular membrane network, Golgi is stacked curved flattened sacs, vesicles are small round membrane bubbles."
        )
    if _has("mitochondria", "mitochondrion", "lysosome", "peroxisome", "vacuole"):
        hints.append(
            "- Distinguish round vesicle-like organelles from mitochondria: mitochondria are bean/oval with inner folds/cristae; lysosome/peroxisome/vacuole are more bubble-like or dense round compartments without visible folded inner membranes."
        )
    if _has("microtubules", "microfilaments", "cytoskeleton"):
        hints.append(
            "- Distinguish cytoskeletal labels carefully: microtubules are longer tube-like filaments, microfilaments are thinner strand-like filaments, cytoskeleton is the broader supporting network rather than one single filament."
        )
    if _has("clathrin-coated pit", "snare proteins", "rab gtpase", "na+/k+ atpase"):
        hints.append(
            "- Protein/process labels are usually small membrane-localized structures or markers; do not map a large organelle to them unless the object is truly a small membrane feature."
        )
    return hints


def build_single_cluster_label_match_prompt(
    *,
    base_context: str,
    object_key: str,
    refined_labels: List[str],
    visual: Dict[str, Any],
    force_assignment: bool,
) -> str:
    labels = _dedupe_clean_labels(refined_labels)[:120]
    compact_visual = {
        "object_key": str(object_key or "").strip(),
        "stage1_label_guess": str((visual or {}).get("label_guess", "") or "")[:120],
        "geometry_keywords": (visual or {}).get("geometry_keywords", []) if isinstance((visual or {}).get("geometry_keywords"), list) else [],
        "full_visual_LEFT": str((visual or {}).get("full_visual_LEFT", "") or "")[:700],
        "LEFT_SURROUNDINGS": str((visual or {}).get("LEFT_SURROUNDINGS", "") or "")[:700],
    }
    match_mode = (
        "There are enough remaining labels for the remaining objects, so avoid null and choose the closest allowed label unless the fit is genuinely impossible."
        if force_assignment
        else "Use null only if none of the allowed labels fit the visual evidence."
    )
    hint_lines = _match_hint_lines_for_labels(labels)
    hints_text = "\n".join(hint_lines)
    if hints_text:
        hints_text += "\n"

    return (
        f"BASE CONTEXT: {(base_context or '').strip() or 'unknown'}\n\n"
        "You are matching ONE visual object to ONE allowed label.\n"
        "The stage1 label guess may be wrong. Do NOT echo it blindly.\n"
        "You must actually compare the object's visual description against the allowed labels and choose the best visual-semantic fit.\n"
        f"{match_mode}\n"
        "Use ONLY one of the allowed labels exactly as written.\n"
        "Prefer the most specific correct label over a vague broad label.\n"
        "Reason about shape, membrane style, stacked sacs vs tubular network, bubble-like vesicles, inner folds, boundary/ring vs interior spot, and filament-like structure when relevant.\n"
        f"{hints_text}"
        "Return JSON only with schema:\n"
        "{"
        "\"object_key\": string,"
        "\"label\": string|null,"
        "\"confidence\": 0.0,"
        "\"reason\": string"
        "}\n\n"
        "ALLOWED LABELS:\n"
        f"{json.dumps(labels, ensure_ascii=False)}\n\n"
        "OBJECT VISUAL:\n"
        f"{json.dumps(compact_visual, ensure_ascii=False, indent=2)}\n"
    )


def _normalize_label_for_match(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip().lower()
    if not t:
        return ""
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9+/]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _label_variants_for_match(s: str) -> List[str]:
    base = _normalize_label_for_match(s)
    if not base:
        return []

    variants = {base}
    irregular = {
        "mitochondrion": "mitochondria",
        "mitochondria": "mitochondrion",
        "vesicle": "vesicles",
        "vesicles": "vesicle",
        "microtubule": "microtubules",
        "microtubules": "microtubule",
        "microfilament": "microfilaments",
        "microfilaments": "microfilament",
        "nuclear pore": "nuclear pores",
        "nuclear pores": "nuclear pore",
        "cell nucleus": "nucleus",
        "cell membrane": "plasma membrane",
    }
    if base in irregular:
        variants.add(irregular[base])

    if base.startswith("cell "):
        variants.add(base[5:].strip())
    if base.endswith(" cell"):
        variants.add(base[:-5].strip())

    words = base.split()
    if len(words) > 1:
        variants.add(words[-1])

    if base.endswith("ies") and len(base) > 3:
        variants.add(base[:-3] + "y")
    if base.endswith("es") and len(base) > 2:
        variants.add(base[:-2])
    if base.endswith("s") and len(base) > 1:
        variants.add(base[:-1])
    else:
        variants.add(base + "s")

    return [x for x in variants if x]


def _resolve_refined_label_match(label_guess: Any, refined_labels: List[str]) -> Tuple[Optional[str], float, str]:
    if not isinstance(label_guess, str) or not label_guess.strip():
        return None, 0.0, "empty_label_guess"

    cleaned_guess = label_guess.strip()
    variants = _label_variants_for_match(cleaned_guess)
    if not variants:
        return None, 0.0, "empty_label_guess"

    allowed: List[Tuple[str, str]] = []
    for label in refined_labels or []:
        if not isinstance(label, str):
            continue
        exact = label.strip()
        if not exact:
            continue
        allowed.append((exact, _normalize_label_for_match(exact)))
    if not allowed:
        return None, 0.0, "no_refined_labels"

    for variant in variants:
        for exact, normalized in allowed:
            if variant == normalized:
                return exact, 0.99, f"normalized_exact:{cleaned_guess}"

    for variant in variants:
        for exact, normalized in allowed:
            if variant and normalized and (variant in normalized or normalized in variant):
                return exact, 0.9, f"normalized_overlap:{cleaned_guess}"

    best_label = None
    best_score = 0.0
    for variant in variants:
        for exact, normalized in allowed:
            score = SequenceMatcher(None, variant, normalized).ratio()
            if score > best_score:
                best_score = score
                best_label = exact

    if best_label and best_score >= 0.84:
        return best_label, float(best_score), f"fuzzy_stage1_label:{cleaned_guess}"

    return None, float(best_score), f"no_refined_match:{cleaned_guess}"


def _parse_postfacto_matches(text_out: str, refined_labels: List[str]) -> Dict[str, Dict[str, Any]]:
    obj = extract_json_object(text_out or "")
    matches = obj.get("matches") if isinstance(obj, dict) else None
    if not isinstance(matches, list):
        matches = [x for x in extract_json_objects(text_out or "") if isinstance(x, dict) and isinstance(x.get("object_key"), str)]
    if not isinstance(matches, list):
        return {}

    out: Dict[str, Dict[str, Any]] = {}

    for it in matches:
        if not isinstance(it, dict):
            continue
        ck = it.get("object_key")
        if not isinstance(ck, str) or not ck.strip():
            continue
        ck = ck.strip()

        lab = it.get("label")
        if lab is not None:
            if not isinstance(lab, str):
                lab = None
            else:
                lab, _, _ = _resolve_refined_label_match(lab, refined_labels)

        conf = it.get("confidence", 0.0)
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        conf_f = max(0.0, min(1.0, conf_f))

        reason = it.get("reason")
        reason_s = str(reason or "").strip()

        out[ck] = {
            "label": lab,
            "confidence": conf_f,
            "reason": reason_s,
        }

    return out


def _parse_single_postfacto_match_obj(
    obj: Dict[str, Any],
    *,
    expected_object_key: str,
    refined_labels: List[str],
) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {
            "label": None,
            "confidence": 0.0,
            "reason": "invalid_or_empty_match_object",
        }

    lab = obj.get("label")
    if lab is not None:
        if not isinstance(lab, str):
            lab = None
        else:
            lab, _, _ = _resolve_refined_label_match(lab, refined_labels)

    conf = obj.get("confidence", 0.0)
    try:
        conf_f = float(conf)
    except Exception:
        conf_f = 0.0
    conf_f = max(0.0, min(1.0, conf_f))
    reason_s = str(obj.get("reason", "") or "").strip()
    if not reason_s:
        reason_s = "match_completed_without_reason"

    return {
        "label": lab,
        "confidence": conf_f,
        "reason": reason_s,
        "object_key": str(obj.get("object_key", "") or expected_object_key),
    }


def _stage1_match_fallback_for_allowed_labels(
    *,
    visual: Dict[str, Any],
    allowed_labels: List[str],
) -> Optional[Dict[str, Any]]:
    if not isinstance(visual, dict):
        return None
    label_guess = str(visual.get("label_guess", "") or "").strip()
    resolved, score, reason = _resolve_refined_label_match(label_guess, allowed_labels)
    if not resolved:
        return None
    return {
        "label": resolved,
        "confidence": max(0.55, min(0.9, float(score))),
        "reason": f"stage1_exact_fallback:{reason}",
    }


def _fallback_matches_from_stage1_labels(
    refined_labels: List[str],
    cluster_visual_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for ck, vis in (cluster_visual_map or {}).items():
        if not isinstance(vis, dict):
            continue
        label_guess = str(vis.get("label_guess", "") or "").strip()
        resolved, score, reason = _resolve_refined_label_match(label_guess, refined_labels)
        if not resolved:
            continue
        out[str(ck)] = {
            "label": resolved,
            "confidence": max(0.65, min(0.99, float(score))),
            "reason": reason,
        }
    return out


def _dedupe_matches_by_label(
    matches: Dict[str, Dict[str, Any]],
    cluster_order: List[str],
) -> Dict[str, Dict[str, Any]]:
    if not matches:
        return {}

    order_rank: Dict[str, int] = {str(k): i for i, k in enumerate(cluster_order or [])}
    best_by_label: Dict[str, Dict[str, Any]] = {}

    for ck, m in matches.items():
        if not isinstance(m, dict):
            continue
        lab = m.get("label")
        if not isinstance(lab, str) or not lab.strip():
            continue
        conf = float(m.get("confidence", 0.0) or 0.0)
        rank = int(order_rank.get(str(ck), 10**9))

        prev = best_by_label.get(lab)
        if prev is None or conf > float(prev["confidence"]) or (conf == float(prev["confidence"]) and rank < int(prev["rank"])):
            best_by_label[lab] = {"cluster_key": str(ck), "confidence": conf, "rank": rank}

    out: Dict[str, Dict[str, Any]] = {}
    for ck, m in matches.items():
        if not isinstance(m, dict):
            continue
        row = dict(m)
        lab = row.get("label")
        if isinstance(lab, str) and lab.strip():
            keep = best_by_label.get(lab)
            if not keep or str(keep.get("cluster_key")) != str(ck):
                prev_reason = str(row.get("reason", "") or "").strip()
                row["label"] = None
                row["confidence"] = 0.0
                row["reason"] = (prev_reason + " | duplicate_label_dropped").strip(" |")
        out[str(ck)] = row

    return out


def _infer_cluster_index(cluster_key: str, entry: Dict[str, Any]) -> Optional[int]:
    if isinstance(entry, dict):
        for k in ("cluster_index", "cluster_idx", "index"):
            v = entry.get(k)
            try:
                if v is not None:
                    return int(v)
            except Exception:
                pass
    m = re.match(r"^(\d+)_", str(cluster_key or "").strip())
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _build_label_cluster_outputs(
    *,
    refined_labels: List[str],
    clusters: Dict[str, Dict[str, Any]],
    cluster_order: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    label_cluster_map: Dict[str, Any] = {}
    cleaned_labels: List[str] = []
    seen_labels = set()
    for x in (refined_labels or []):
        sx = str(x).strip()
        if not sx:
            continue
        k = sx.lower()
        if k in seen_labels:
            continue
        seen_labels.add(k)
        cleaned_labels.append(sx)

    rows: List[Dict[str, Any]] = []
    for ck in (cluster_order or []):
        rec = clusters.get(ck) if isinstance(clusters, dict) else None
        if not isinstance(rec, dict):
            continue
        entry = rec.get("entry") if isinstance(rec.get("entry"), dict) else {}
        rows.append({
            "cluster_key": str(ck),
            "cluster_index": _infer_cluster_index(str(ck), entry),
            "mask_name": str(entry.get("crop_file_mask", "") or "") if isinstance(entry, dict) else "",
            "visual": rec.get("visual") if isinstance(rec.get("visual"), dict) else {},
            "matched_label": rec.get("matched_label"),
            "match_confidence": rec.get("match_confidence"),
            "match_reason": rec.get("match_reason", ""),
        })

    by_label: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        lab = row.get("matched_label")
        if not isinstance(lab, str) or not lab.strip():
            continue
        by_label[lab] = {
            "cluster_key": row.get("cluster_key"),
            "cluster_index": row.get("cluster_index"),
            "mask_name": row.get("mask_name"),
            "visual": row.get("visual", {}),
            "confidence": row.get("match_confidence"),
            "reason": row.get("match_reason", ""),
        }

    for lab in cleaned_labels:
        label_cluster_map[lab] = by_label.get(lab, {
            "cluster_key": None,
            "cluster_index": None,
            "mask_name": None,
            "visual": None,
            "confidence": 0.0,
            "reason": "",
        })

    return label_cluster_map, rows


def _dedupe_clean_labels(refined_labels: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in (refined_labels or []):
        sx = str(x or "").strip()
        if not sx:
            continue
        key = sx.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(sx)
    return out


def _coerce_int_list(values: Any) -> List[int]:
    out: List[int] = []
    seen = set()
    if not isinstance(values, list):
        return out
    for v in values:
        try:
            i = int(v)
        except Exception:
            continue
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def _coerce_float_01(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        out = 0.0
    return max(0.0, min(1.0, out))


def _build_clean_cluster_labels_output(
    *,
    image_index: int,
    diagram_name: str,
    refined_labels: List[str],
    label_cluster_map: Dict[str, Any],
    clusters: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for label in _dedupe_clean_labels(refined_labels):
        rec = label_cluster_map.get(label) if isinstance(label_cluster_map, dict) else None
        rec = rec if isinstance(rec, dict) else {}
        cluster_key = str(rec.get("cluster_key", "") or "").strip()
        cluster_rec = clusters.get(cluster_key) if cluster_key and isinstance(clusters, dict) else None
        cluster_rec = cluster_rec if isinstance(cluster_rec, dict) else {}
        entry = cluster_rec.get("entry") if isinstance(cluster_rec.get("entry"), dict) else {}
        stroke_indexes = _coerce_int_list(entry.get("stroke_indexes"))
        target_key = str(rec.get("mask_name", "") or entry.get("crop_file_mask", "") or cluster_key or "").strip()
        row = {
            "matched_label": label,
            "match_confidence": _coerce_float_01(rec.get("confidence", 0.0)),
            "stroke_indexes": stroke_indexes,
            "target_type": "cluster",
            "target_key": target_key or None,
            "cluster_index": rec.get("cluster_index"),
        }
        rows.append(row)

    return {
        "schema": "diagram_label_matches_clean_v1",
        "image_index": int(image_index),
        "diagram_type": 1,
        "diagram_name": str(diagram_name or "").strip(),
        "base_context": str(diagram_name or "").strip(),
        "labels": rows,
        "matched_labels_count": int(sum(1 for row in rows if row.get("stroke_indexes"))),
    }


def _image_path_to_data_uri(path: Any) -> str:
    p = Path(str(path or ""))
    if not p.is_file():
        return ""
    suffix = p.suffix.lower().lstrip(".")
    mime = "image/png"
    if suffix in {"jpg", "jpeg"}:
        mime = "image/jpeg"
    elif suffix == "webp":
        mime = "image/webp"
    try:
        encoded = base64.b64encode(p.read_bytes()).decode("ascii")
    except Exception:
        return ""
    return f"data:{mime};base64,{encoded}"


def _rgb_array_to_png_data_uri(value: Any) -> str:
    arr = _as_rgb_uint8(value)
    if arr is None:
        return ""
    try:
        buf = BytesIO()
        Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""
    return f"data:image/png;base64,{encoded}"


def _cluster_report_visual_text(visual: Any) -> str:
    if not isinstance(visual, dict):
        return ""
    parts: List[str] = []
    label_guess = str(visual.get("label_guess", "") or "").strip()
    if label_guess:
        parts.append(f"stage1 guess: {label_guess}")
    full_visual = str(visual.get("full_visual_LEFT", "") or visual.get("visual_summary", "") or "").strip()
    if full_visual:
        parts.append(full_visual)
    surroundings = str(visual.get("LEFT_SURROUNDINGS", "") or visual.get("neighbor_context", "") or "").strip()
    if surroundings:
        parts.append(f"context: {surroundings}")
    keywords = visual.get("geometry_keywords")
    if isinstance(keywords, list):
        clean = [str(x).strip() for x in keywords if str(x).strip()]
        if clean:
            parts.append("keywords: " + ", ".join(clean[:12]))
    return " | ".join(parts)


def _write_cluster_label_visual_report(
    *,
    report_path: Path,
    image_index: int,
    base_context: str,
    results: Dict[str, Any],
    renders_mask_rgb: Optional[Dict[str, Any]] = None,
) -> str:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    clusters = results.get("clusters") if isinstance(results.get("clusters"), dict) else {}
    order = results.get("cluster_order") if isinstance(results.get("cluster_order"), list) else []
    rows_html: List[str] = []

    for pos, cluster_key in enumerate(order, start=1):
        rec = clusters.get(cluster_key) if isinstance(clusters, dict) else None
        if not isinstance(rec, dict):
            continue
        mask_name = str(rec.get("mask_name", "") or "").strip()
        if not mask_name:
            entry = rec.get("entry") if isinstance(rec.get("entry"), dict) else {}
            mask_name = str(entry.get("crop_file_mask", "") or "").strip()

        data_uri = ""
        if renders_mask_rgb and mask_name in renders_mask_rgb:
            data_uri = _rgb_array_to_png_data_uri(renders_mask_rgb.get(mask_name))
        if not data_uri and mask_name:
            data_uri = _image_path_to_data_uri(CLUSTER_RENDER_DIR / f"processed_{int(image_index)}" / mask_name)

        label = str(rec.get("matched_label", "") or "").strip() or "(unmatched)"
        confidence = rec.get("match_confidence")
        try:
            confidence_text = f"{float(confidence):.2f}"
        except Exception:
            confidence_text = ""
        reason = str(rec.get("match_reason", "") or "").strip()
        visual_text = _cluster_report_visual_text(rec.get("visual"))
        error = str(rec.get("error", "") or "").strip()

        image_html = (
            f'<img src="{data_uri}" alt="cluster {html_lib.escape(str(cluster_key))}">'
            if data_uri
            else '<div class="missing">missing render</div>'
        )
        rows_html.append(
            "<section class=\"cluster-card\">"
            f"<div class=\"cluster-image\">{image_html}</div>"
            "<div class=\"cluster-meta\">"
            f"<h2>{html_lib.escape(label)}</h2>"
            f"<p><strong>Cluster:</strong> {html_lib.escape(str(cluster_key))}</p>"
            f"<p><strong>Mask:</strong> {html_lib.escape(mask_name or '(none)')}</p>"
            f"<p><strong>Confidence:</strong> {html_lib.escape(confidence_text or '(none)')}</p>"
            f"<p><strong>Reason:</strong> {html_lib.escape(reason or '(none)')}</p>"
            f"<p><strong>Visual read:</strong> {html_lib.escape(visual_text or error or '(none)')}</p>"
            "</div>"
            f"<div class=\"cluster-number\">#{pos}</div>"
            "</section>"
        )

    if not rows_html:
        rows_html.append("<p>No clusters were available in this labeling result.</p>")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Cluster Label Report - processed_{int(image_index)}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #172033; }}
    header {{ padding: 24px 28px; background: #172033; color: white; }}
    header h1 {{ margin: 0 0 8px; font-size: 24px; }}
    header p {{ margin: 0; color: #d9e1f2; max-width: 980px; }}
    main {{ display: grid; gap: 14px; padding: 18px; }}
    .cluster-card {{ position: relative; display: grid; grid-template-columns: 260px 1fr; gap: 18px; align-items: stretch; background: white; border: 1px solid #dfe4ec; border-radius: 8px; padding: 14px; }}
    .cluster-image {{ display: flex; align-items: center; justify-content: center; min-height: 220px; background: #fff; border: 1px solid #e6e9ef; border-radius: 6px; overflow: hidden; }}
    .cluster-image img {{ max-width: 100%; max-height: 240px; object-fit: contain; image-rendering: auto; }}
    .missing {{ color: #6b7280; font-size: 14px; }}
    .cluster-meta h2 {{ margin: 2px 42px 10px 0; font-size: 21px; }}
    .cluster-meta p {{ margin: 7px 0; line-height: 1.4; }}
    .cluster-number {{ position: absolute; top: 12px; right: 14px; color: #64748b; font-size: 13px; }}
    @media (max-width: 720px) {{
      .cluster-card {{ grid-template-columns: 1fr; }}
      .cluster-image {{ min-height: 180px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Cluster Label Report - processed_{int(image_index)}</h1>
    <p>{html_lib.escape(str(base_context or ""))}</p>
  </header>
  <main>
    {''.join(rows_html)}
  </main>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    return str(report_path.resolve())


def _build_clean_schematic_labels_output(
    *,
    image_index: int,
    diagram_name: str,
    refined_labels: List[str],
    label_line_map: Dict[str, Any],
    line_keyed: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for label in _dedupe_clean_labels(refined_labels):
        rec = label_line_map.get(label) if isinstance(label_line_map, dict) else None
        rec = rec if isinstance(rec, dict) else {}
        raw_line_keys = rec.get("line_keys") if isinstance(rec.get("line_keys"), list) else []
        line_keys: List[str] = []
        line_indexes: List[int] = []
        stroke_indexes: List[int] = []
        seen_line_keys = set()
        seen_line_indexes = set()
        seen_strokes = set()
        for raw_key in raw_line_keys:
            key = str(raw_key or "").strip()
            if not key or key in seen_line_keys:
                continue
            seen_line_keys.add(key)
            line_keys.append(key)
            line_row = line_keyed.get(key) if isinstance(line_keyed, dict) else None
            if not isinstance(line_row, dict):
                continue
            try:
                line_index = int(line_row.get("line_index"))
            except Exception:
                line_index = None
            if line_index is not None and line_index not in seen_line_indexes:
                seen_line_indexes.add(line_index)
                line_indexes.append(line_index)
            try:
                stroke_index = int(line_row.get("source_stroke_index", line_index if line_index is not None else -1))
            except Exception:
                stroke_index = None
            if stroke_index is not None and stroke_index >= 0 and stroke_index not in seen_strokes:
                seen_strokes.add(stroke_index)
                stroke_indexes.append(stroke_index)

        target_key = str(rec.get("target_key", "") or "").strip() or None
        target_type = str(rec.get("target_type", "") or "").strip().lower()
        if not line_keys and not target_key:
            target_type = None
        elif target_type not in ("line", "group"):
            target_type = "group" if len(line_keys) > 1 else "line"

        rows.append(
            {
                "matched_label": label,
                "match_confidence": _coerce_float_01(rec.get("confidence", 0.0)),
                "stroke_indexes": stroke_indexes,
                "target_type": target_type,
                "target_key": target_key,
                "line_keys": line_keys,
                "line_indexes": line_indexes,
            }
        )

    return {
        "schema": "diagram_label_matches_clean_v1",
        "image_index": int(image_index),
        "diagram_type": 2,
        "diagram_name": str(diagram_name or "").strip(),
        "base_context": str(diagram_name or "").strip(),
        "labels": rows,
        "matched_labels_count": int(sum(1 for row in rows if row.get("stroke_indexes"))),
    }


def postfacto_match_labels_with_qwen(
    model,
    processor,
    device: torch.device,
    *,
    base_context: str,
    refined_labels: List[str],
    cluster_visual_map: Dict[str, Dict[str, Any]],
    debug_sink: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    if not refined_labels or not cluster_visual_map:
        if isinstance(debug_sink, dict):
            debug_sink["mode"] = "skipped_empty_inputs"
        return {}
    cleaned_labels = _dedupe_clean_labels(refined_labels)
    cluster_keys = [
        str(k)
        for k, v in (cluster_visual_map or {}).items()
        if str(k or "").strip() and isinstance(v, dict)
    ]
    if not cleaned_labels or not cluster_keys:
        if isinstance(debug_sink, dict):
            debug_sink["mode"] = "skipped_empty_cleaned_inputs"
        return {}

    final_matches: Dict[str, Dict[str, Any]] = {}
    remaining_labels = list(cleaned_labels)
    pending_keys = list(cluster_keys)
    rounds_debug: List[Dict[str, Any]] = []
    model_raw_parts: List[str] = []

    try:
        max_rounds = min(6, max(2, len(cleaned_labels)))
        for round_i in range(max_rounds):
            if not pending_keys or not remaining_labels:
                break

            force_assignment = len(remaining_labels) >= len(pending_keys)
            prompts: List[str] = []
            for ck in pending_keys:
                prompts.append(
                    build_single_cluster_label_match_prompt(
                        base_context=base_context,
                        object_key=ck,
                        refined_labels=remaining_labels,
                        visual=cluster_visual_map.get(ck) if isinstance(cluster_visual_map.get(ck), dict) else {},
                        force_assignment=force_assignment,
                    )
                )

            parsed_objs, raw_texts = _generate_json_objects_from_prompts(
                model=model,
                processor=processor,
                device=device,
                prompts=prompts,
                temperature=0.0,
                max_new_tokens=min(220, int(POSTFACTO_MATCH_MAX_NEW_TOKENS)),
                batch_size=max(1, min(6, len(prompts))),
                debug_label=f"postfacto_match_round_{round_i+1}",
                stage_io_contexts=[
                    _build_qwen_stage_io_context(
                        bundle=None,
                        debug_label=f"postfacto_match_round_{round_i+1}",
                        job_key=str(ck),
                        stage_kind="multimodal",
                        rendered_input_text=str(prompts[idx] if idx < len(prompts) else ""),
                        input_prompt_text=str(prompts[idx] if idx < len(prompts) else ""),
                        extra={
                            "round_index": int(round_i + 1),
                            "object_key": str(ck),
                            "force_assignment": bool(force_assignment),
                        },
                    )
                    for idx, ck in enumerate(pending_keys)
                ],
            )

            round_matches: Dict[str, Dict[str, Any]] = {}
            round_raw: List[Dict[str, Any]] = []
            for idx, ck in enumerate(pending_keys):
                obj = parsed_objs[idx] if idx < len(parsed_objs) and isinstance(parsed_objs[idx], dict) else {}
                parsed = _parse_single_postfacto_match_obj(
                    obj,
                    expected_object_key=ck,
                    refined_labels=remaining_labels,
                )
                if not parsed.get("label"):
                    fallback_row = _stage1_match_fallback_for_allowed_labels(
                        visual=cluster_visual_map.get(ck) if isinstance(cluster_visual_map.get(ck), dict) else {},
                        allowed_labels=remaining_labels,
                    )
                    if isinstance(fallback_row, dict):
                        parsed.update(fallback_row)
                round_matches[str(ck)] = {
                    "label": parsed.get("label"),
                    "confidence": float(parsed.get("confidence", 0.0) or 0.0),
                    "reason": str(parsed.get("reason", "") or ""),
                }
                round_raw.append(
                    {
                        "object_key": str(ck),
                        "raw_text": raw_texts[idx] if idx < len(raw_texts) else "",
                        "parsed": round_matches[str(ck)],
                    }
                )

            deduped_round = _dedupe_matches_by_label(round_matches, pending_keys)
            accepted_labels: List[str] = []
            accepted_keys: List[str] = []
            for ck in pending_keys:
                row = deduped_round.get(ck) if isinstance(deduped_round, dict) else None
                if not isinstance(row, dict):
                    continue
                lab = row.get("label")
                if isinstance(lab, str) and lab.strip():
                    final_matches[str(ck)] = row
                    accepted_keys.append(str(ck))
                    accepted_labels.append(str(lab))

            if raw_texts:
                model_raw_parts.append(f"[round {round_i+1}]\n" + "\n".join(str(x or "") for x in raw_texts))
            rounds_debug.append(
                {
                    "round_index": int(round_i + 1),
                    "force_assignment": bool(force_assignment),
                    "pending_in": list(pending_keys),
                    "remaining_labels_in": list(remaining_labels),
                    "accepted_keys": list(accepted_keys),
                    "accepted_labels": list(accepted_labels),
                    "raw_rows": round_raw,
                }
            )

            if not accepted_keys:
                break

            pending_keys = [ck for ck in pending_keys if ck not in set(accepted_keys)]
            remaining_labels = [lab for lab in remaining_labels if lab not in set(accepted_labels)]

        fallback = _fallback_matches_from_stage1_labels(remaining_labels, {k: cluster_visual_map.get(k, {}) for k in pending_keys})
        for ck in pending_keys:
            row = fallback.get(str(ck))
            if isinstance(row, dict) and isinstance(row.get("label"), str) and row.get("label"):
                final_matches[str(ck)] = row

        if isinstance(debug_sink, dict):
            debug_sink["mode"] = "iterative_per_object_qwen"
            debug_sink["rounds"] = rounds_debug
            debug_sink["parsed_matches"] = len(final_matches)
            debug_sink["pending_unmatched"] = [ck for ck in cluster_keys if ck not in final_matches]
            debug_sink["model_raw"] = "\n\n".join(model_raw_parts)[:20000]
        return final_matches

    except Exception as e:
        print(f"[warn] postfacto_match_labels_with_qwen failed: {e}")
        fallback = _fallback_matches_from_stage1_labels(refined_labels, cluster_visual_map)
        if isinstance(debug_sink, dict):
            debug_sink["mode"] = "exception_stage1_label_fallback"
            debug_sink["error"] = f"{type(e).__name__}: {e}"
            debug_sink["fallback_matches"] = len(fallback)
        return fallback




# ============================
# VRAM debug
# ============================
def _vram_str() -> str:
    if not torch.cuda.is_available():
        return "cuda=off"
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    reserv = torch.cuda.memory_reserved() / (1024 ** 2)
    return f"alloc={alloc:.0f}MiB reserved={reserv:.0f}MiB"


def _clear_model_runtime_cache_refs(model: Any) -> None:
    for attr in ("_cache", "_past_key_values", "past_key_values"):
        try:
            if hasattr(model, attr):
                setattr(model, attr, None)
        except Exception:
            pass


# ============================
# Optional fast reject (noise)
# ============================
def _count_colored_pixels(mask_rgb: Image.Image) -> int:
    arr = np.asarray(mask_rgb.convert("RGB"), dtype=np.uint8)
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    colored = (np.abs(r - g) + np.abs(r - b) + np.abs(g - b)) > 20
    return int(colored.sum())


# ============================
# Model loading
# ============================
def load_model_and_processor():
    have_cuda = torch.cuda.is_available() and not FORCE_CPU
    device = torch.device(f"cuda:{GPU_INDEX}" if have_cuda else "cpu")

    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            min_pixels=QWEN_MIN_PIXELS,
            max_pixels=QWEN_MAX_PIXELS,
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(MODEL_ID)

    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.padding_side = "left"
        tok.truncation_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

    used_quant = False
    quant_config = None

    if have_cuda and QUANT_MODE in ("4bit", "8bit") and _HAS_BNB:
        if QUANT_MODE == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=DTYPE,
            )
            used_quant = True
        elif QUANT_MODE == "8bit":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=bool(INT8_CPU_OFFLOAD),
            )
            used_quant = True

    load_kwargs = {
        "low_cpu_mem_usage": True,
    }

    attn_impl = _qwen_attn_impl()
    if attn_impl is not None:
        load_kwargs["attn_implementation"] = attn_impl

    if have_cuda:
        with _qwen_safe_loader_context():
            if used_quant:
                load_kwargs["quantization_config"] = quant_config
                load_kwargs["device_map"] = "auto" if INT8_CPU_OFFLOAD else {"": GPU_INDEX}
                model = _load_qwen_vl_direct(
                    MODEL_ID,
                    load_kwargs=load_kwargs,
                )
            else:
                load_kwargs["torch_dtype"] = DTYPE
                load_kwargs["device_map"] = {"": GPU_INDEX}
                model = _load_qwen_vl_direct(
                    MODEL_ID,
                    load_kwargs=load_kwargs,
                )
    else:
        model = _load_qwen_vl_direct(
            MODEL_ID,
            load_kwargs={
                **load_kwargs,
                "torch_dtype": torch.float32,
            },
        ).to(device)

    if tok is not None and hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id

    try:
        model.config.use_cache = True
    except Exception:
        pass

    if have_cuda:
        model = _maybe_enable_qwen_fast_inference(
            model,
            device,
            allow_compile=not used_quant,
        )

    return model, processor, device, used_quant

# ============================
# Preload + warmup (for orchestrator)
# ============================
def _make_dummy_vl_sample() -> tuple[str, Image.Image]:
    # Small image keeps warmup cheap
    img = Image.new("RGB", (32, 32), (0, 0, 0))
    SYSTEM_MSG = "You are a test harness. Reply with JSON only."
    prompt_text = 'Return ONLY JSON: {"ok": true}'
    conv = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
    ]
    return conv, img

def _img_tensor_dtype(device: torch.device) -> torch.dtype:
    # CPU fp16 is a common source of pain; keep CPU on fp32.
    return DTYPE if device.type == "cuda" else torch.float32


def _as_rgb_uint8(arr: Any) -> Optional[np.ndarray]:
    """
    Accepts: np arrays in RGB/BGR, PIL Image, etc.
    Returns: np.ndarray RGB uint8 (H,W,3) or None.
    """
    if arr is None:
        return None

    if isinstance(arr, Image.Image):
        arr = np.asarray(arr.convert("RGB"), dtype=np.uint8)

    if not isinstance(arr, np.ndarray):
        return None

    if arr.dtype != np.uint8:
        try:
            arr = arr.astype(np.uint8, copy=False)
        except Exception:
            return None

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim != 3 or arr.shape[2] < 3:
        return None

    # If 4ch, drop alpha.
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]

    return arr


def _extract_mask_map(pack: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Tries multiple key shapes that ImageClusters might use.
    Returns dict: mask_filename -> RGB uint8 array.
    """
    # Common dict-style keys
    dict_keys_rgb = [
        "renders_mask_rgb",
        "mask_renders_rgb",
        "mask_crops_rgb",
        "render_masks_rgb",
    ]
    dict_keys_bgr = [
        "renders_mask_bgr",
        "mask_renders_bgr",
        "mask_crops_bgr",
        "render_masks_bgr",
    ]

    for k in dict_keys_rgb:
        mm = pack.get(k)
        if isinstance(mm, dict):
            out: Dict[str, np.ndarray] = {}
            for name, img in mm.items():
                rgb = _as_rgb_uint8(img)
                if rgb is not None:
                    out[str(name)] = rgb
            if out:
                return out

    for k in dict_keys_bgr:
        mm = pack.get(k)
        if isinstance(mm, dict):
            out: Dict[str, np.ndarray] = {}
            for name, img in mm.items():
                bgr = _as_rgb_uint8(img)
                if bgr is None:
                    continue
                # treat as BGR -> convert to RGB
                rgb = bgr[:, :, ::-1].copy()
                out[str(name)] = rgb
            if out:
                return out

    # List-style payloads (less common)
    list_keys = [
        "mask_crops",
        "mask_renders",
        "renders_masks",
        "mask_images",
    ]
    for k in list_keys:
        lst = pack.get(k)
        if isinstance(lst, list):
            out: Dict[str, np.ndarray] = {}
            for it in lst:
                if not isinstance(it, dict):
                    continue
                name = it.get("name") or it.get("file") or it.get("filename")
                img = it.get("rgb") or it.get("img_rgb") or it.get("image_rgb")
                if img is None:
                    img = it.get("bgr") or it.get("img_bgr") or it.get("image_bgr")
                    bgr = _as_rgb_uint8(img)
                    if bgr is not None:
                        out[str(name)] = bgr[:, :, ::-1].copy()
                    continue
                rgb = _as_rgb_uint8(img)
                if rgb is not None:
                    out[str(name)] = rgb
            if out:
                return out

    return {}



def warmup_qwen_once(model, processor, device: torch.device) -> None:
    conv, img = _make_dummy_vl_sample()
    prompt_str = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    inputs = _processor_batch(
        processor,
        text=[prompt_str],
        images=[img],
    )
    inputs = _move_inputs_to_device(inputs, device)

    fast_flags = getattr(model, "_qwen_fast_flags", {})
    warmup_iters = 2 if fast_flags.get("compiled") else 1

    with torch.inference_mode():
        warmup_kwargs = {
            "max_new_tokens": 8,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
        }
        for _ in range(warmup_iters):
            _ = _safe_generate(model, inputs=inputs, gen_kwargs=warmup_kwargs)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

def _clean_short_label(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    t = re.sub(r"\s+", " ", s.strip())
    if not t:
        return None

    # keep short
    parts = t.split(" ")
    if len(parts) > 4:
        t = " ".join(parts[:4]).strip()

    if len(t) > 60:
        t = t[:60].strip()

    return t or None


def _resize_longest_side(img: Image.Image, max_side: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (max_side, max_side), (0, 0, 0))

    longest = max(w, h)
    if longest <= max_side:
        return img

    scale = float(max_side) / float(longest)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), resample=Image.BICUBIC)




def preload_qwen_cpu_only(
    model_id: str = MODEL_ID,
    *,
    warmup: bool = False,
) -> tuple[Any, Any, torch.device]:
    device = torch.device("cpu")

    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=QWEN_MIN_PIXELS,
            max_pixels=QWEN_MAX_PIXELS,
            trust_remote_code=True,
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.padding_side = "left"
        tok.truncation_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

    model = _load_qwen_vl_direct(
        model_id,
        load_kwargs={
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        },
    ).to(device)

    if tok is not None and hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id

    try:
        model.config.use_cache = True
    except Exception:
        pass

    if warmup:
        warmup_qwen_once(model, processor, device)

    return model, processor, device



def move_qwen_to_cuda_if_available(
    model,
    *,
    gpu_index: int = GPU_INDEX,
) -> tuple[Any, torch.device]:
    if torch.cuda.is_available() and not FORCE_CPU:
        device = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(gpu_index)

        model = model.to(device)
        model.eval()

        try:
            model = model.to(dtype=DTYPE)
        except Exception:
            pass

        model = _maybe_enable_qwen_fast_inference(
            model,
            device,
            allow_compile=True,
        )
        return model, device

    return model, torch.device("cpu")


def create_c2_local_worker(
    model_id: str = C2_TEXT_MODEL_ID,
    *,
    max_new_tokens: int = 256,
):
    from C2.QwenWorker import ServerQwenWorker

    model_id = _resolved_text_model_id(model_id)
    return ServerQwenWorker(
        model_name=model_id,
        max_new_tokens=max_new_tokens,
        server_base_url=str(os.getenv("QWEN_VLLM_SERVER_URL", "http://127.0.0.1:8009") or "http://127.0.0.1:8009").strip(),
        stage_io_dir=str(os.getenv("QWEN_STAGE_IO_DIR", "") or ""),
    )


def destroy_c2_local_worker(worker: Any) -> None:
    return


# ============================
# In-memory entrypoint (called by Orchestrator)
# ============================
def label_clusters_transformers(
    clusters_state: Dict[int, Dict[str, Any]],
    model,
    processor,
    device: torch.device,
    *,
    save_outputs: bool = False,
    skip_existing_labels: Optional[bool] = None,
) -> Dict[int, Dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    panel_edge = max(64, int(PROC_LONGEST_EDGE) // 2)

    gen_kwargs = dict(
        max_new_tokens=int(CLUSTER_VISUAL_MAX_NEW_TOKENS),
        do_sample=bool(DO_SAMPLE),
        num_beams=1,
        use_cache=True,
    )

    results_by_idx: Dict[int, Dict[str, Any]] = {}
    do_skip_existing = SKIP_EXISTING_LABELS if skip_existing_labels is None else bool(skip_existing_labels)

    for idx in sorted(clusters_state.keys()):
        folder = OUT_DIR / f"processed_{idx}"
        folder.mkdir(parents=True, exist_ok=True)
        labels_path = folder / "labels.json"
        debug_folder = OUT_DEBUG_DIR / f"processed_{idx}"
        debug_folder.mkdir(parents=True, exist_ok=True)
        debug_labels_path = debug_folder / "labels.json"

        if save_outputs and do_skip_existing and debug_labels_path.exists():
            try:
                results_by_idx[idx] = json.loads(debug_labels_path.read_text(encoding="utf-8"))
                print(f"[skip] idx={idx}: existing debug labels.json loaded")
                continue
            except Exception:
                pass

        pack = clusters_state[idx]
        t0 = time.time()

        base_context = str(pack.get("base_context", "") or "")

        suggestions_all = list(pack.get("candidate_labels_raw", []) or [])
        refined_upstream = list(pack.get("candidate_labels_refined", []) or [])

        # NOTE: refined_upstream now used ONLY in post-facto stage2 matching
        refined_labels = refined_upstream or suggestions_all
        refined_labels = [str(x).strip() for x in refined_labels if str(x).strip()]

        full_img_rgb = _as_rgb_uint8(pack.get("full_img_rgb"))
        if full_img_rgb is None:
            print(f"[skip] idx={idx}: missing full_img_rgb")
            continue

        full_img0 = Image.fromarray(full_img_rgb, mode="RGB")
        right_panel_base, full_scale, full_rw, full_rh, full_pad_x, full_pad_y = _fit_into_square(full_img0, panel_edge)

        results: Dict[str, Any] = {
            "image_index": int(idx),
            "candidate_labels_raw": suggestions_all[: int(SUGGESTION_LIMIT)],
            "candidate_labels_refined": refined_labels[: int(SUGGESTION_LIMIT)],
            "base_context": base_context,
            "model_id": str(getattr(model, "name_or_path", MODEL_ID)),
            "proc_longest_edge": int(PROC_LONGEST_EDGE),
            "panel_edge": int(panel_edge),
            "batch_size": int(BATCH_SIZE),
            "clusters": {},
            "cluster_order": [],
            "postfacto_matches": {},
            "postfacto_debug": {},
            "label_cluster_map": {},
            "cluster_label_rows": [],
        }
        results_by_idx[idx] = results

        entries = pack.get("clusters") or []
        if not isinstance(entries, list) or not entries:
            print(f"[skip] idx={idx}: pack has no clusters list")
            continue

        keep_mask_names_raw = (
            pack.get("sam_keep_mask_names")
            or pack.get("kept_mask_names")
            or pack.get("allowed_mask_names")
            or []
        )
        keep_mask_names = {
            str(x or "").strip()
            for x in keep_mask_names_raw
            if str(x or "").strip()
        }
        results["clusters_input_total"] = int(len(entries))
        if keep_mask_names:
            results["sam_keep_mask_names"] = sorted(keep_mask_names)

        mask_map = _extract_mask_map(pack)
        renders_dir = CLUSTER_RENDER_DIR / f"processed_{idx}"

        def _load_mask_from_disk(mask_name: str) -> Optional[np.ndarray]:
            p = renders_dir / mask_name
            if not p.exists():
                return None
            try:
                img = Image.open(p).convert("RGB")
                return np.asarray(img, dtype=np.uint8)
            except Exception:
                return None

        def _store(cluster_key: str, payload: Dict[str, Any]) -> None:
            if cluster_key not in results["clusters"]:
                results["cluster_order"].append(cluster_key)
            results["clusters"][cluster_key] = payload

        tasks: List[Dict[str, Any]] = []
        for ci, entry in enumerate(entries, start=1):
            bbox_xyxy = entry.get("bbox_xyxy")
            mask_name = entry.get("crop_file_mask")

            if not (isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4):
                continue
            if not isinstance(mask_name, str) or not mask_name.strip():
                continue

            mask_name = mask_name.strip()
            if keep_mask_names and mask_name not in keep_mask_names:
                continue
            mask_rgb = mask_map.get(mask_name)
            if mask_rgb is None:
                mask_rgb = _load_mask_from_disk(mask_name)

            if mask_rgb is None:
                continue

            colour_hint = None
            if isinstance(entry.get("color_name"), str) and entry.get("color_name").strip():
                colour_hint = entry["color_name"].strip().lower()
            else:
                colour_hint = parse_colour_hint_from_filename(mask_name)

            tasks.append({
                "ci": ci,
                "entry": entry,
                "bbox_xyxy": [int(x) for x in bbox_xyxy],
                "mask_name": mask_name,
                "mask_rgb": mask_rgb,
                "colour_hint": colour_hint,
            })

        results["clusters_after_sam_filter"] = int(len(tasks))

        bs = max(1, int(BATCH_SIZE))
        for start in range(0, len(tasks), bs):
            chunk = tasks[start:start + bs]
            if not chunk:
                continue

            texts_batch: List[str] = []
            images_batch: List[Image.Image] = []
            meta_batch: List[Dict[str, Any]] = []

            for task in chunk:
                ci = int(task["ci"])
                entry = task["entry"]
                bbox_xyxy = task["bbox_xyxy"]
                mask_rgb = task["mask_rgb"]
                colour_hint = task.get("colour_hint")

                cluster_key = f"{ci:04d}_{task['mask_name']}"

                left_img = Image.fromarray(_as_rgb_uint8(mask_rgb), mode="RGB")
                left_sq, _, _, _, _, _ = _fit_into_square(left_img, panel_edge)

                bbox_resized = _scale_bbox_xyxy(bbox_xyxy, full_scale)
                bbox_resized = _clamp_bbox_xyxy(bbox_resized, full_rw, full_rh)
                bbox_panel = [
                    bbox_resized[0] + full_pad_x,
                    bbox_resized[1] + full_pad_y,
                    bbox_resized[2] + full_pad_x,
                    bbox_resized[3] + full_pad_y,
                ]

                right_sq = right_panel_base.copy()
                _draw_red_rect_pil(right_sq, bbox_panel, thickness=RECT_THICKNESS_PX)

                composite = _make_composite(left_sq, right_sq)

                prompt_text = build_cluster_visual_prompt(colour_hint, base_context)

                SYSTEM_MSG = (
                    "You are a strict visual transcription engine.\n"
                    "No guessing identity.\n"
                    "Return JSON only.\n"
                )

                conv = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
                ]
                prompt_str = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

                texts_batch.append(prompt_str)
                images_batch.append(composite)

                meta_batch.append({
                    "ci": ci,
                    "cluster_key": cluster_key,
                    "entry": entry,
                    "mask_name": task["mask_name"],
                    "colour_hint": colour_hint,
                    "bbox_xyxy": bbox_xyxy,
                    "bbox_right_panel_xyxy": bbox_panel,
                })

            if not texts_batch:
                continue

            t_req0 = time.time()
            try:
                inputs = _processor_batch(
                    processor,
                    text=texts_batch,
                    images=images_batch,
                )
                inputs = _move_inputs_to_device(inputs, device)

                with torch.inference_mode():
                    out_ids = _safe_generate(model, inputs=inputs, gen_kwargs=gen_kwargs)

                if "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
                    prompt_len = int(inputs["input_ids"].shape[1])
                    gen_only_ids = out_ids[:, prompt_len:]
                else:
                    gen_only_ids = out_ids

                texts_out = processor.batch_decode(gen_only_ids, skip_special_tokens=True)
                dt = time.time() - t_req0

            except Exception as e:
                for meta in meta_batch:
                    _store(meta["cluster_key"], {"entry": meta["entry"], "error": f"batch_inference_failed: {e}"})
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            n_out = min(len(texts_out), len(meta_batch))
            for j in range(n_out):
                text_out = texts_out[j]
                meta = meta_batch[j]

                parsed_visual = _parse_cluster_visual(text_out)

                _store(meta["cluster_key"], {
                    "cluster_index": int(meta["ci"]),
                    "entry": meta["entry"],
                    "mask_name": meta.get("mask_name"),
                    "colour_hint": meta["colour_hint"],
                    "bbox_xyxy": meta["bbox_xyxy"],
                    "bbox_right_panel_xyxy": meta["bbox_right_panel_xyxy"],
                    "timing_s": float(f"{dt:.4f}"),
                    "model_raw": text_out,
                    "visual": parsed_visual,
                    "matched_label": None,
                    "match_confidence": None,
                    "match_reason": "",
                })

        # ---- Stage 2: post-facto label matching across all clusters (text-only)
        cluster_visual_map: Dict[str, Dict[str, Any]] = {}
        for ck in results.get("cluster_order", []):
            rec = results["clusters"].get(ck, {})
            vis = rec.get("visual")
            if isinstance(vis, dict):
                cluster_visual_map[ck] = vis

        postfacto_debug: Dict[str, Any] = {}
        matches = postfacto_match_labels_with_qwen(
            model=model,
            processor=processor,
            device=device,
            base_context=base_context,
            refined_labels=refined_labels,
            cluster_visual_map=cluster_visual_map,
            debug_sink=postfacto_debug,
        )
        matches = _dedupe_matches_by_label(matches, results.get("cluster_order", []))

        results["postfacto_matches"] = matches
        results["postfacto_debug"] = postfacto_debug

        for ck, m in matches.items():
            if ck in results["clusters"] and isinstance(m, dict):
                results["clusters"][ck]["matched_label"] = m.get("label")
                results["clusters"][ck]["match_confidence"] = m.get("confidence")
                results["clusters"][ck]["match_reason"] = m.get("reason", "")

        label_cluster_map, cluster_label_rows = _build_label_cluster_outputs(
            refined_labels=refined_labels,
            clusters=results.get("clusters", {}),
            cluster_order=results.get("cluster_order", []),
        )
        results["label_cluster_map"] = label_cluster_map
        results["cluster_label_rows"] = cluster_label_rows

        if save_outputs:
            try:
                results["cluster_label_report_path"] = _write_cluster_label_visual_report(
                    report_path=folder / "cluster_label_report.html",
                    image_index=idx,
                    base_context=base_context,
                    results=results,
                    renders_mask_rgb=pack.get("renders_mask_rgb") if isinstance(pack.get("renders_mask_rgb"), dict) else None,
                )
            except Exception as e:
                results["cluster_label_report_error"] = f"{type(e).__name__}: {e}"

            clean_out = _build_clean_cluster_labels_output(
                image_index=idx,
                diagram_name=base_context,
                refined_labels=refined_labels,
                label_cluster_map=label_cluster_map,
                clusters=results.get("clusters", {}),
            )
            labels_path.write_text(json.dumps(clean_out, ensure_ascii=False, indent=2), encoding="utf-8")
            debug_labels_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[ok] idx={idx}: clusters={len(tasks)} dt={time.time()-t0:.2f}s")

    return results_by_idx


def _as_float_pair(v: Any) -> Tuple[float, float]:
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        return (0.0, 0.0)
    try:
        return (float(v[0]), float(v[1]))
    except Exception:
        return (0.0, 0.0)


def _build_schematic_key_payload(line_desc: Dict[str, Any]) -> Dict[str, Any]:
    described_lines = line_desc.get("described_lines") if isinstance(line_desc, dict) else None
    groups = line_desc.get("groups") if isinstance(line_desc, dict) else None
    described_lines = described_lines if isinstance(described_lines, list) else []
    groups = groups if isinstance(groups, list) else []

    line_rows: List[Dict[str, Any]] = []
    for src_order, row in enumerate(described_lines):
        if not isinstance(row, dict):
            continue
        line_idx = row.get("described_line_index", row.get("line_index"))
        try:
            line_idx_i = int(line_idx)
        except Exception:
            continue

        geom = row.get("geometry") if isinstance(row.get("geometry"), dict) else {}
        centroid = _as_float_pair(row.get("centroid"))
        if centroid == (0.0, 0.0):
            centroid = _as_float_pair(geom.get("centroid"))

        bbox0 = {}
        if isinstance(row.get("bbox"), dict):
            bbox0 = row.get("bbox") or {}
        elif isinstance(geom.get("bbox"), dict):
            bbox0 = geom.get("bbox") or {}
        elif isinstance(geom.get("bbox_xyxy"), dict):
            bbox0 = geom.get("bbox_xyxy") or {}
        bbox = {
            "min_x": float(bbox0.get("min_x", 0.0) or 0.0),
            "min_y": float(bbox0.get("min_y", 0.0) or 0.0),
            "max_x": float(bbox0.get("max_x", 0.0) or 0.0),
            "max_y": float(bbox0.get("max_y", 0.0) or 0.0),
        }
        group_index = row.get("group_index", None)
        try:
            group_index = int(group_index) if group_index is not None else None
        except Exception:
            group_index = None

        desc = str(row.get("description", "") or row.get("full_description", "") or "").strip()
        if len(desc) > 320:
            desc = desc[:320].rstrip() + "..."

        left_to_right_rank = row.get("left_to_right_rank", src_order)
        try:
            left_to_right_rank = int(left_to_right_rank)
        except Exception:
            left_to_right_rank = int(src_order)

        line_rows.append(
            {
                "line_index": line_idx_i,
                "source_stroke_index": int(row.get("source_stroke_index", line_idx_i) or line_idx_i),
                "group_index": group_index,
                "centroid": [round(centroid[0], 2), round(centroid[1], 2)],
                "bbox_xyxy": bbox,
                "description": desc,
                "_left_to_right_rank": left_to_right_rank,
            }
        )

    line_rows.sort(
        key=lambda r: (
            int(r.get("_left_to_right_rank", 10**9)),
            float(r["centroid"][0]),
            float(r["centroid"][1]),
            int(r["line_index"]),
        )
    )

    line_key_order: List[str] = []
    line_keyed: Dict[str, Dict[str, Any]] = {}
    line_key_by_index: Dict[int, str] = {}
    line_rank: Dict[int, int] = {}
    for rank, row in enumerate(line_rows):
        key = f"L{int(row['line_index']):04d}"
        line_key_order.append(key)
        line_rank[int(row["line_index"])] = rank
        line_key_by_index[int(row["line_index"])] = key
        row_out = dict(row)
        row_out.pop("_left_to_right_rank", None)
        line_keyed[key] = row_out

    group_rows: List[Dict[str, Any]] = []
    for g in groups:
        if not isinstance(g, dict):
            continue
        gi = g.get("group_index", None)
        try:
            gi_i = int(gi)
        except Exception:
            continue

        members_raw = g.get("member_line_indices")
        members_raw = members_raw if isinstance(members_raw, list) else []
        members: List[int] = []
        for m in members_raw:
            try:
                members.append(int(m))
            except Exception:
                continue

        member_keys = [line_key_by_index[m] for m in members if m in line_key_by_index]
        if not member_keys:
            continue
        member_keys.sort(key=lambda k: line_rank.get(int(line_keyed.get(k, {}).get("line_index", -1)), 10**9))

        centroid = _as_float_pair(g.get("centroid"))
        shape = g.get("shape_inference") if isinstance(g.get("shape_inference"), dict) else {}
        g_desc = str(
            g.get("description")
            or g.get("shape_summary")
            or shape.get("group_summary")
            or shape.get("shape_summary")
            or ""
        ).strip()
        if len(g_desc) > 240:
            g_desc = g_desc[:240].rstrip() + "..."

        rank_min = min((line_rank.get(int(x), 10**9) for x in members), default=10**9)
        group_rows.append(
            {
                "group_index": gi_i,
                "member_line_keys": member_keys,
                "centroid": [round(centroid[0], 2), round(centroid[1], 2)],
                "description": g_desc,
                "_rank_min": rank_min,
            }
        )

    group_rows.sort(key=lambda r: (int(r.get("_rank_min", 10**9)), float(r["centroid"][0]), int(r["group_index"])))

    group_keyed: Dict[str, Dict[str, Any]] = {}
    for g in group_rows:
        key = f"G{int(g['group_index']):04d}"
        out = dict(g)
        out.pop("_rank_min", None)
        group_keyed[key] = out

    return {
        "line_key_order": line_key_order,
        "line_keyed": line_keyed,
        "group_keyed": group_keyed,
    }


def _build_schematic_label_prompt(
    *,
    base_context: str,
    refined_labels: List[str],
    key_payload: Dict[str, Any],
) -> str:
    ctx = str(base_context or "").strip() or "unknown"
    labels = [str(x).strip() for x in (refined_labels or []) if str(x).strip()][:140]
    line_key_order = key_payload.get("line_key_order") if isinstance(key_payload, dict) else []
    line_keyed = key_payload.get("line_keyed") if isinstance(key_payload, dict) else {}
    group_keyed = key_payload.get("group_keyed") if isinstance(key_payload, dict) else {}

    return (
        "You are matching refined diagram labels to schematic lines/groups.\n"
        "The line keys are already sorted LEFT to RIGHT.\n"
        "Use line geometry/location/group context only; infer meaning by structure and context.\n"
        "Every label must map to one unique line key or one unique group key.\n"
        "Do not reuse a target for multiple labels unless absolutely unavoidable.\n"
        "Output JSON only.\n\n"
        f"BASE_CONTEXT:\n{ctx}\n\n"
        f"REFINED_LABELS:\n{json.dumps(labels, ensure_ascii=False)}\n\n"
        f"LINE_KEY_ORDER_LEFT_TO_RIGHT:\n{json.dumps(line_key_order, ensure_ascii=False)}\n\n"
        f"LINES_KEYED:\n{json.dumps(line_keyed, ensure_ascii=False)}\n\n"
        f"GROUPS_KEYED:\n{json.dumps(group_keyed, ensure_ascii=False)}\n\n"
        "Return this JSON schema exactly:\n"
        "{"
        "\"matches\":["
        "{\"label\":\"...\",\"target_type\":\"line|group|null\",\"target_key\":\"L0001|G0001|null\",\"line_keys\":[\"L0001\"],\"confidence\":0.0,\"reason\":\"short\"}"
        "]"
        "}\n"
    )


def _parse_schematic_matches(
    text_out: str,
    *,
    refined_labels: List[str],
    line_key_order: List[str],
    group_keyed: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    line_set = {str(k) for k in (line_key_order or [])}
    group_to_lines: Dict[str, List[str]] = {}
    for gk, gv in (group_keyed or {}).items():
        if not isinstance(gv, dict):
            continue
        raw = gv.get("member_line_keys")
        if not isinstance(raw, list):
            continue
        group_to_lines[str(gk)] = [str(x) for x in raw if str(x) in line_set]

    labels_clean: List[str] = []
    label_set = set()
    for lb in (refined_labels or []):
        s = str(lb or "").strip()
        if not s:
            continue
        lk = s.lower()
        if lk in label_set:
            continue
        label_set.add(lk)
        labels_clean.append(s)

    obj = extract_json_object(text_out or "")
    raw_matches = obj.get("matches") if isinstance(obj, dict) else None
    if not isinstance(raw_matches, list):
        raw_matches = []

    by_label: Dict[str, Dict[str, Any]] = {}
    for it in raw_matches:
        if not isinstance(it, dict):
            continue
        label = str(it.get("label", "") or "").strip()
        if label.lower() not in label_set:
            continue

        ttype = str(it.get("target_type", "") or "").strip().lower()
        if ttype not in ("line", "group"):
            ttype = "line"

        target_key = str(it.get("target_key", "") or "").strip()
        raw_line_keys = it.get("line_keys")
        if not isinstance(raw_line_keys, list):
            raw_line_keys = []
        line_keys = [str(x) for x in raw_line_keys if str(x) in line_set]

        if ttype == "group":
            if target_key in group_to_lines:
                line_keys = group_to_lines[target_key]
            elif target_key in line_set:
                ttype = "line"
                line_keys = [target_key]
        else:
            if not line_keys and target_key in line_set:
                line_keys = [target_key]
            if not line_keys and target_key in group_to_lines:
                ttype = "group"
                line_keys = group_to_lines[target_key]

        if not line_keys:
            continue

        conf = 0.0
        try:
            conf = float(it.get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))

        reason = str(it.get("reason", "") or "").strip()
        if len(reason) > 200:
            reason = reason[:200].rstrip() + "..."

        rec = {
            "label": label,
            "target_type": ttype,
            "target_key": target_key if target_key else (line_keys[0] if ttype == "line" else ""),
            "line_keys": line_keys,
            "confidence": conf,
            "reason": reason,
        }
        prev = by_label.get(label.lower())
        if prev is None or float(rec["confidence"]) > float(prev.get("confidence", 0.0)):
            by_label[label.lower()] = rec

    # enforce one unique target per label map
    used_targets: Dict[Tuple[str, ...], str] = {}
    deduped_rows: List[Dict[str, Any]] = []
    for lb in labels_clean:
        rec = by_label.get(lb.lower())
        if not isinstance(rec, dict):
            deduped_rows.append(
                {
                    "label": lb,
                    "target_type": None,
                    "target_key": None,
                    "line_keys": [],
                    "confidence": 0.0,
                    "reason": "unmatched",
                }
            )
            continue

        rec = dict(rec)
        rec["label"] = lb
        sig = tuple(sorted(str(x) for x in (rec.get("line_keys") or [])))
        prev_label = used_targets.get(sig)
        if sig and prev_label and prev_label != lb:
            deduped_rows.append(
                {
                    "label": lb,
                    "target_type": None,
                    "target_key": None,
                    "line_keys": [],
                    "confidence": 0.0,
                    "reason": "target_already_used",
                }
            )
            continue
        if sig:
            used_targets[sig] = lb
        deduped_rows.append(rec)

    out_map: Dict[str, Dict[str, Any]] = {}
    for row in deduped_rows:
        out_map[str(row.get("label", ""))] = {
            "target_type": row.get("target_type"),
            "target_key": row.get("target_key"),
            "line_keys": row.get("line_keys") if isinstance(row.get("line_keys"), list) else [],
            "confidence": float(row.get("confidence", 0.0) or 0.0),
            "reason": str(row.get("reason", "") or ""),
        }
    return out_map, deduped_rows


def label_schematic_lines_transformers(
    schematic_state: Dict[int, Dict[str, Any]],
    model,
    processor,
    device: torch.device,
    *,
    batch_size: int = 8,
    save_outputs: bool = False,
    skip_existing_labels: Optional[bool] = None,
) -> Dict[int, Dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMATIC_OUT_DIR.mkdir(parents=True, exist_ok=True)
    do_skip_existing = SKIP_EXISTING_LABELS if skip_existing_labels is None else bool(skip_existing_labels)

    jobs: List[Dict[str, Any]] = []
    results_by_idx: Dict[int, Dict[str, Any]] = {}
    total_inputs = 0
    skipped_missing_labels_or_lines = 0

    for idx in sorted((schematic_state or {}).keys()):
        total_inputs += 1
        folder = OUT_DIR / f"processed_{idx}"
        folder.mkdir(parents=True, exist_ok=True)
        labels_path = folder / "labels.json"
        debug_folder = OUT_DEBUG_DIR / f"processed_{idx}"
        debug_folder.mkdir(parents=True, exist_ok=True)
        debug_labels_path = debug_folder / "labels.json"
        legacy_folder = SCHEMATIC_OUT_DIR / f"processed_{idx}"
        legacy_folder.mkdir(parents=True, exist_ok=True)
        legacy_labels_path = legacy_folder / "labels.json"

        if save_outputs and do_skip_existing and debug_labels_path.exists():
            try:
                results_by_idx[idx] = json.loads(debug_labels_path.read_text(encoding="utf-8"))
                print(f"[skip] schematic idx={idx}: existing debug labels.json loaded")
                continue
            except Exception:
                pass

        pack = schematic_state[idx] if isinstance(schematic_state.get(idx), dict) else {}
        base_context = str(pack.get("base_context", "") or "").strip()
        labels = list(pack.get("candidate_labels_refined", []) or pack.get("candidate_labels_raw", []) or [])
        labels = [str(x).strip() for x in labels if str(x).strip()]
        labels = labels[:140]
        line_desc = pack.get("line_descriptions") if isinstance(pack.get("line_descriptions"), dict) else {}
        key_payload = _build_schematic_key_payload(line_desc)

        if not labels or not key_payload.get("line_key_order"):
            skipped_missing_labels_or_lines += 1
            results_by_idx[idx] = {
                "image_index": int(idx),
                "base_context": base_context,
                "candidate_labels_refined": labels,
                "line_key_order": key_payload.get("line_key_order", []),
                "line_keyed": key_payload.get("line_keyed", {}),
                "group_keyed": key_payload.get("group_keyed", {}),
                "label_line_map": {},
                "line_label_rows": [],
                "error": "missing_labels_or_lines",
            }
            print(
                f"[qwen][DBG] schematic idx={idx} skipped: missing_labels_or_lines "
                f"(labels={len(labels)} lines={len(key_payload.get('line_key_order', []))})"
            )
            continue

        prompt = _build_schematic_label_prompt(
            base_context=base_context,
            refined_labels=labels,
            key_payload=key_payload,
        )
        jobs.append(
            {
                "idx": int(idx),
                "prompt": prompt,
                "labels": labels,
                "key_payload": key_payload,
                "labels_path": labels_path,
                "debug_labels_path": debug_labels_path,
                "legacy_labels_path": legacy_labels_path,
                "base_context": base_context,
            }
        )

    print(
        f"[qwen][DBG] schematic prepare total={total_inputs} "
        f"jobs={len(jobs)} skipped_missing={skipped_missing_labels_or_lines}"
    )

    if jobs:
        prompts = [j["prompt"] for j in jobs]
        _parsed_objs, raw_texts = _generate_json_objects_from_prompts(
            model=model,
            processor=processor,
            device=device,
            prompts=prompts,
            temperature=0.10,
            max_new_tokens=int(SCHEMATIC_LINE_MATCH_MAX_NEW_TOKENS),
            batch_size=max(1, int(batch_size)),
            debug_label="schematic_line_match",
            stage_io_contexts=[
                _build_qwen_stage_io_context(
                    bundle=None,
                    debug_label="schematic_line_match",
                    job_key=f"processed_{int(job.get('idx', 0))}",
                    stage_kind="multimodal",
                    rendered_input_text=str(prompts[idx] if idx < len(prompts) else ""),
                    input_prompt_text=str(prompts[idx] if idx < len(prompts) else ""),
                    extra={
                        "processed_index": int(job.get("idx", 0) or 0),
                        "base_context": str(job.get("base_context", "") or ""),
                    },
                )
                for idx, job in enumerate(jobs)
            ],
        )

        for i, job in enumerate(jobs):
            idx = int(job["idx"])
            raw = raw_texts[i] if i < len(raw_texts) else ""
            labels = job["labels"]
            key_payload = job["key_payload"]
            label_line_map, line_label_rows = _parse_schematic_matches(
                str(raw),
                refined_labels=labels,
                line_key_order=key_payload.get("line_key_order", []),
                group_keyed=key_payload.get("group_keyed", {}),
            )

            out = {
                "image_index": int(idx),
                "base_context": job.get("base_context", ""),
                "candidate_labels_refined": labels,
                "line_key_order": key_payload.get("line_key_order", []),
                "line_keyed": key_payload.get("line_keyed", {}),
                "group_keyed": key_payload.get("group_keyed", {}),
                "label_line_map": label_line_map,
                "line_label_rows": line_label_rows,
                "model_raw": raw,
            }
            results_by_idx[idx] = out
            if save_outputs:
                clean_out = _build_clean_schematic_labels_output(
                    image_index=idx,
                    diagram_name=str(job.get("base_context", "") or ""),
                    refined_labels=labels,
                    label_line_map=label_line_map,
                    line_keyed=key_payload.get("line_keyed", {}),
                )
                Path(job["labels_path"]).write_text(json.dumps(clean_out, ensure_ascii=False, indent=2), encoding="utf-8")
                Path(job["debug_labels_path"]).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
                Path(job["legacy_labels_path"]).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] schematic idx={idx}: labels={len(labels)} lines={len(key_payload.get('line_key_order', []))}")
    else:
        print("[qwen][DBG] schematic no generation jobs were created.")

    return results_by_idx


def build_stage1_visual_probe_prompt() -> str:
    # Strict schema; no semantic inference; small + consistent keys for stage2.
    return (
        "Make a strict VISUAL description of the image with no regard to semantic meaning.\n"
        "Describe only visible shapes, blobs, membranes, clusters, compact detail, and layout.\n"
        "Describe how the visible parts are distributed across the image.\n"
        "Do NOT infer meaning or identity.\n"
        "Output JSON ONLY with:\n"
        "{\"objects\":[\"shape\"],\"structure\":\"1-4 words\",\"dominant\":\"1-4 words\",\"simplicity\":1}\n"
        "Rules:\n"
        "- objects: 3-8 items, each 1-4 words\n"
        "- structure: short layout phrase like center, spread out, left-heavy, ring-like\n"
        "- dominant: the most visually dominant visible thing, 1-4 words\n"
        "- simplicity: integer 1 to 5, where 1 is very simple and 5 is very dense\n"
        "- no extra keys, no prose\n"
    )


def _as_meta_score(v: Any) -> Optional[float]:
    """
    Normalize external score (final_score) into a float if possible.
    Keep it as-is (0..1 expected), clamp to sane bounds.
    """
    if v is None:
        return None
    try:
        s = float(v)
    except Exception:
        return None
    # clamp to avoid prompt garbage if something went wrong upstream
    if s != s:  # NaN
        return None
    return max(0.0, min(1.0, s))


def _parse_stage1_visual_probe(text_out: str) -> Dict[str, Any]:
    """
    New stage-1 structure:
      { "objects":[...], "structure":"...", "dominant":"...", "simplicity":1..5 }

    Parser is tolerant to minor key drift (labels->objects, layout->structure, etc).
    """
    obj = extract_json_object(text_out or "")
    if not isinstance(obj, dict):
        return {"objects": [], "structure": "", "dominant": "", "simplicity": 5}

    raw_objects = (
        obj.get("objects")
        if obj.get("objects") is not None else
        obj.get("labels")  # tolerate old key
    )
    raw_structure = (
        obj.get("structure")
        if obj.get("structure") is not None else
        obj.get("layout")  # tolerate variants
    )
    raw_dominant = (
        obj.get("dominant")
        if obj.get("dominant") is not None else
        obj.get("main_object")
    )
    raw_simplicity = (
        obj.get("simplicity")
        if obj.get("simplicity") is not None else
        obj.get("complexity")
    )

    out_objects: List[str] = []
    if isinstance(raw_objects, list):
        seen = set()
        for it in raw_objects:
            if not isinstance(it, str):
                continue
            c = _clean_short_label(it)
            if not c:
                continue
            k = c.lower()
            if k in seen:
                continue
            seen.add(k)
            out_objects.append(c)
            if len(out_objects) >= 8:
                break

    structure = ""
    if isinstance(raw_structure, str):
        structure = _clean_short_label(raw_structure) or ""

    dominant = ""
    if isinstance(raw_dominant, str):
        dominant = _clean_short_label(raw_dominant) or ""

    simp_i = 5
    try:
        simp_i = int(raw_simplicity)
    except Exception:
        simp_i = 5
    simp_i = max(1, min(5, simp_i))

    # Ensure minimum objects count
    if len(out_objects) < 3:
        # salvage from raw text (comma/newline split) if model didn't follow schema
        for part in re.split(r"[,;\n]+", (text_out or "").strip()):
            c = _clean_short_label(part)
            if not c:
                continue
            out_objects.append(c)
            if len(out_objects) >= 3:
                break

    return {
        "objects": out_objects[:8],
        "structure": structure,
        "dominant": dominant,
        "simplicity": simp_i,
    }


def probe_processed_images_stage1(
    *,
    model,
    processor,
    device: torch.device,
    items: List[Dict[str, Any]],
    batch_size: int = 8,
    longest_side: int = 600,
) -> Dict[str, Dict[str, Any]]:
    """
    items: [
      {
        "processed_id": "processed_12",
        "image_pil": PIL.Image,
        "base_context": "...",
        # NEW (optional): "final_score": 0.83  (or "score": 0.83)
      }, ...
    ]

    Returns:
      {
        processed_id: {
          "objects":[...],
          "structure":"...",
          "dominant":"...",
          "simplicity":1..5,
          "score": Optional[float]   # passthrough external meta score
        },
        ...
      }
    """
    SYSTEM_MSG = (
        "You are a visual transcription engine.\n"
        "Describe only visible shapes/layout.\n"
        "No guessing. Output JSON only.\n"
    )

    prompt_text = build_stage1_visual_probe_prompt()
    conv = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
    ]
    prompt_str = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    gen_kwargs = dict(
        max_new_tokens=160,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

    out: Dict[str, Dict[str, Any]] = {}

    bs = max(1, int(batch_size))
    for start in range(0, len(items), bs):
        chunk = items[start:start + bs]
        if not chunk:
            continue

        texts_batch: List[str] = []
        images_batch: List[Image.Image] = []
        ids_batch: List[str] = []
        scores_batch: List[Optional[float]] = []

        for it in chunk:
            pid = str(it.get("processed_id", "") or "").strip()
            img = it.get("image_pil")
            if not pid or not isinstance(img, Image.Image):
                continue

            # NEW: passthrough meta score (final_score) so stage2 can use it
            score = _as_meta_score(it.get("final_score", None))
            if score is None:
                score = _as_meta_score(it.get("score", None))

            img2 = _resize_longest_side(img, int(longest_side))

            texts_batch.append(prompt_str)
            images_batch.append(img2)
            ids_batch.append(pid)
            scores_batch.append(score)

        if not texts_batch:
            continue

        try:
            inputs = _processor_batch(
                processor,
                text=texts_batch,
                images=images_batch,
            )
            inputs = _move_inputs_to_device(inputs, device)

            with torch.inference_mode():
                out_ids = _safe_generate(model, inputs=inputs, gen_kwargs=gen_kwargs)

            if "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
                prompt_len = int(inputs["input_ids"].shape[1])
                gen_only_ids = out_ids[:, prompt_len:]
            else:
                gen_only_ids = out_ids

            texts_out = processor.batch_decode(gen_only_ids, skip_special_tokens=True)

        except Exception as e:
            for pid, sc in zip(ids_batch, scores_batch):
                out[pid] = {
                    "objects": [],
                    "structure": "",
                    "dominant": "",
                    "simplicity": 5,
                    "score": sc,
                    "error": f"stage1_failed: {e}",
                }
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        n_out = min(len(texts_out), len(ids_batch))
        for j in range(n_out):
            pid = ids_batch[j]
            parsed = _parse_stage1_visual_probe(texts_out[j])
            parsed["score"] = scores_batch[j]  # NEW: attach external meta score
            out[pid] = parsed

    return out


def _probe_objects_summary(objects: Any, limit: int = 4) -> str:
    vals = [str(x).strip() for x in (objects or []) if str(x).strip()]
    if not vals:
        return ""
    return ", ".join(vals[: max(1, int(limit))])


def _probe_to_line_description(probe: Dict[str, Any], *, fallback_name: str = "") -> str:
    if not isinstance(probe, dict):
        probe = {}

    dominant = _clean_short_label(str(probe.get("dominant", "") or "")) or ""
    structure = _clean_short_label(str(probe.get("structure", "") or "")) or ""
    objects = _probe_objects_summary(probe.get("objects"), limit=4)

    parts: List[str] = []
    if dominant:
        parts.append(f"dominant {dominant}")
    if objects:
        parts.append(f"objects {objects}")
    if structure:
        parts.append(f"layout {structure}")

    desc = "; ".join(parts).strip(" ;")
    if desc:
        return desc
    if fallback_name:
        return f"cluster {fallback_name}"
    return "visual cluster"


def build_zero_shot_schematic_descriptor_from_clusters(
    *,
    processed_id: str,
    base_context: str,
    refined_labels: List[str],
    clusters: List[Dict[str, Any]],
    renders_mask_rgb: Dict[str, Any],
    model,
    processor,
    device: torch.device,
    batch_size: int = 8,
    longest_side: int = 384,
) -> Dict[str, Any]:
    """
    Fallback when StrokeDescriptions are missing:
      - probe each cluster render with the stage1 visual prompt
      - synthesize described_lines/groups in the same shape as LineDescriptors output
      - feed that into the normal schematic matcher
    """
    candidates: List[Dict[str, Any]] = []
    group_members: Dict[int, List[int]] = {}

    for src_order, entry in enumerate(clusters or []):
        if not isinstance(entry, dict):
            continue
        mask_name = str(entry.get("crop_file_mask", "") or "").strip()
        if not mask_name:
            continue
        mask_rgb = renders_mask_rgb.get(mask_name)
        mask_arr = _as_rgb_uint8(mask_rgb)
        if mask_arr is None:
            continue

        bbox_xyxy = entry.get("bbox_xyxy")
        if not (isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4):
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        except Exception:
            continue

        centroid_x = round((float(x1) + float(x2)) / 2.0, 2)
        centroid_y = round((float(y1) + float(y2)) / 2.0, 2)

        group_index = entry.get("group_index")
        if group_index is None and entry.get("group_index_in_color") is not None:
            try:
                # ClusterMaps commonly uses group_index_in_color; combine it with
                # color_id so zero-shot schematic grouping stays stable/unique.
                color_id_i = int(entry.get("color_id", 0) or 0)
                group_in_color_i = int(entry.get("group_index_in_color"))
                group_index = int(color_id_i * 1000 + group_in_color_i)
            except Exception:
                group_index = entry.get("group_index_in_color")
        try:
            group_index = int(group_index) if group_index is not None else None
        except Exception:
            group_index = None

        line_index = int(len(candidates) + 1)
        if group_index is not None:
            group_members.setdefault(group_index, []).append(line_index)

        candidates.append(
            {
                "line_index": line_index,
                "mask_name": mask_name,
                "bbox_xyxy": [x1, y1, x2, y2],
                "centroid": [centroid_x, centroid_y],
                "group_index": group_index,
                "image_pil": Image.fromarray(mask_arr, mode="RGB"),
                "probe_id": f"{processed_id}::{line_index:04d}::{mask_name}",
            }
        )

    candidates.sort(
        key=lambda row: (
            float((row.get("centroid") or [0.0, 0.0])[0]),
            float((row.get("centroid") or [0.0, 0.0])[1]),
            int(row.get("line_index", 10**9)),
        )
    )

    if not candidates:
        return {
            "schema": "zero_shot_schematic_descriptor_v1",
            "processed_id": processed_id,
            "base_context": base_context,
            "refined_labels": list(refined_labels or []),
            "source": "cluster_renders",
            "described_lines": [],
            "groups": [],
        }

    stage1_items: List[Dict[str, Any]] = []
    for row in candidates:
        stage1_items.append(
            {
                "processed_id": row["probe_id"],
                "image_pil": row["image_pil"],
                "base_context": base_context,
            }
        )

    stage1_map = probe_processed_images_stage1(
        model=model,
        processor=processor,
        device=device,
        items=stage1_items,
        batch_size=max(1, int(batch_size)),
        longest_side=max(128, int(longest_side)),
    )

    described_lines: List[Dict[str, Any]] = []
    for rank, row in enumerate(candidates):
        probe = stage1_map.get(row["probe_id"], {}) if isinstance(stage1_map, dict) else {}
        bbox = row["bbox_xyxy"]
        line_index = int(row["line_index"])
        described_lines.append(
            {
                "described_line_index": line_index,
                "line_index": line_index,
                "source_stroke_index": line_index,
                "group_index": row.get("group_index"),
                "centroid": list(row["centroid"]),
                "bbox": {
                    "min_x": float(bbox[0]),
                    "min_y": float(bbox[1]),
                    "max_x": float(bbox[2]),
                    "max_y": float(bbox[3]),
                },
                "geometry": {
                    "centroid": list(row["centroid"]),
                    "bbox": {
                        "min_x": float(bbox[0]),
                        "min_y": float(bbox[1]),
                        "max_x": float(bbox[2]),
                        "max_y": float(bbox[3]),
                    },
                },
                "description": _probe_to_line_description(probe, fallback_name=row.get("mask_name", "")),
                "mask_name": row.get("mask_name"),
                "left_to_right_rank": int(rank),
                "visual_probe": probe,
            }
        )

    groups: List[Dict[str, Any]] = []
    for group_index in sorted(group_members.keys()):
        member_ids = [int(x) for x in group_members.get(group_index, []) if int(x) > 0]
        if not member_ids:
            continue
        member_rows = [r for r in described_lines if int(r.get("line_index", -1)) in member_ids]
        if not member_rows:
            continue
        cx = sum(float(r["centroid"][0]) for r in member_rows) / float(len(member_rows))
        cy = sum(float(r["centroid"][1]) for r in member_rows) / float(len(member_rows))
        groups.append(
            {
                "group_index": int(group_index),
                "member_line_indices": member_ids,
                "centroid": [round(cx, 2), round(cy, 2)],
                "description": f"cluster group of {len(member_ids)} regions",
                "shape_inference": {"group_summary": f"group of {len(member_ids)} regions"},
            }
        )

    return {
        "schema": "zero_shot_schematic_descriptor_v1",
        "processed_id": processed_id,
        "base_context": base_context,
        "refined_labels": list(refined_labels or []),
        "source": "cluster_renders",
        "described_lines": described_lines,
        "groups": groups,
    }


def build_stage2_pick_prompt(
    base_context: str,
    stage1_map: Dict[str, Dict[str, Any]],
    processed_to_unique: Optional[Dict[str, str]] = None,
) -> str:
    bc = (base_context or "").strip() or "unknown"

    lines: List[str] = []
    for pid, info in stage1_map.items():
        objects = info.get("objects") or []
        structure = info.get("structure") or ""
        dominant = info.get("dominant") or ""
        simplicity = info.get("simplicity", 5)
        score = info.get("score", None)

        try:
            simp_i = int(simplicity)
        except Exception:
            simp_i = 5
        simp_i = max(1, min(5, simp_i))

        # keep objects short
        if isinstance(objects, list):
            obj_s = ",".join(str(x) for x in objects[:8])
        else:
            obj_s = ""

        # format score compactly
        sc_s = "NA"
        try:
            if isinstance(score, (int, float)):
                sc_s = f"{float(score):.3f}"
        except Exception:
            sc_s = "NA"

        up = ""
        if processed_to_unique and pid in processed_to_unique:
            up = processed_to_unique[pid]

        if up:
            lines.append(
                f"{pid} | score={sc_s} | simp={simp_i} | struct={structure} | dom={dominant} | objs={obj_s} | path={up}"
            )
        else:
            lines.append(
                f"{pid} | score={sc_s} | simp={simp_i} | struct={structure} | dom={dominant} | objs={obj_s}"
            )

    joined = "\n".join(lines)

    return ( 
        f"DESIRED IMAGE CONTENT: {bc}\n\n" 
        "You must pick EXACTLY 1 best image to keep.\n" 
        "Goal: a SIMPLE visual OBJECT - one full thing identified in the center of the screen.\n" 
        "Seperated complex compact detail is wanted. Example : Eukaryotic cell > nucleus + organells" 
        "Each candidate comes with an external score. When between two good candidates, pick based on score" 
        "REJECT:\n" "- Images filled with many things spread out\n" 
        "- flowcharts, clusters linked with arrouws, any dense text\n"
        "Candidates:\n" f"{joined}\n\n" 
        "Return JSON ONLY with this exact schema:\n" 
        "{" "\"candidate\":[" 
        "{\"processed_id\":\"...\",\"reason\":\"...\",\"main_object\":\"...\",\"simplicity\":1}" "]" "}\n" 
        "Rules:\n" "- exactly 1 candidates\n" 
        "- processed_id must match one of the listed ids\n" 
        "- reason and main_object must be 1-8 words (short)\n"
        "- simplicity integer 1..5\n" "- no extra keys\n" 
        )


def build_diagram_siglip_rule_text(base_context: str) -> str:
    bc = (base_context or "").strip() or "unknown"
    return (
        f"Desired diagram image content: {bc}. "
        "Prefer a simple single-object visual with one full object centered in frame. "
        "Prefer clearly separated compact internal detail and identifiable parts of the main object. "
        "Prefer visuals that look like one diagram of the requested object instead of many spread-out unrelated things. "
        "Avoid flowcharts, boxes linked with arrows, dense text, and visuals dominated by labels instead of the object."
    )


def _parse_stage2_candidates(text_out: str) -> List[Dict[str, Any]]:
    obj = extract_json_object(text_out or "")
    if not isinstance(obj, dict):
        return []

    # Prefer the schema your prompt actually requests: "candidate": [...]
    cands = obj.get("candidate")
    if not isinstance(cands, list):
        # tolerate old key if the model drifts
        cands = obj.get("candidates")
    if not isinstance(cands, list):
        return []

    out: List[Dict[str, Any]] = []
    seen = set()

    for it in cands:
        if not isinstance(it, dict):
            continue

        pid = it.get("processed_id")
        if not isinstance(pid, str) or not pid.strip():
            continue
        pid = pid.strip()
        if pid in seen:
            continue
        seen.add(pid)

        reason = it.get("reason")
        if not isinstance(reason, str):
            reason = ""

        main_obj = it.get("main_object")
        if not isinstance(main_obj, str):
            main_obj = ""

        simp = it.get("simplicity", 5)
        try:
            simp_i = int(simp)
        except Exception:
            simp_i = 5
        simp_i = max(1, min(5, simp_i))

        out.append({
            "processed_id": pid,
            "reason": _clean_short_label(reason) or "",
            "main_object": _clean_short_label(main_obj) or "",
            "simplicity": simp_i,
        })

        # EXACTLY 1 candidate
        break

    return out


def pick_two_processed_candidates_transformers(
    *,
    model,
    processor,
    device: torch.device,
    processed_items: List[Dict[str, Any]],
    processed_to_unique: Dict[str, str],
    base_context: str,
    batch_size: int = 8,
    longest_side: int = 600,
) -> Dict[str, Any]:
    """
    Stage2 returns/keeps EXACTLY 1 candidate (as your prompt/schema expects).
    Prompts are untouched; we enforce 1 in parsing + fallback + packaging.
    """
    # Stage 1 (vision)
    stage1 = probe_processed_images_stage1(
        model=model,
        processor=processor,
        device=device,
        items=processed_items,
        batch_size=batch_size,
        longest_side=longest_side,
    )

    # Stage 2 (text-only decision)  <-- PROMPT TEXT IS UNCHANGED (build_stage2_pick_prompt)
    SYSTEM_MSG = (
        "You are a strict JSON decision engine.\n"
        "Pick exactly 2 candidates.\n"
        "Bias heavily toward simple single-object visuals.\n"
        "Use score to break ties.\n"
        "Output JSON only.\n"
    )
    prompt_text = build_stage2_pick_prompt(base_context, stage1, processed_to_unique)

    conv = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]
    prompt_str = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    gen_kwargs = dict(
        max_new_tokens=260,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

    chosen: List[Dict[str, Any]] = []
    try:
        inputs = _processor_batch(
            processor,
            text=[prompt_str],
        )
        inputs = _move_inputs_to_device(inputs, device)

        with torch.inference_mode():
            out_ids = _safe_generate(model, inputs=inputs, gen_kwargs=gen_kwargs)

        if "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
            prompt_len = int(inputs["input_ids"].shape[1])
            gen_only_ids = out_ids[:, prompt_len:]
        else:
            gen_only_ids = out_ids

        text_out = processor.batch_decode(gen_only_ids, skip_special_tokens=True)[0]
        chosen = _parse_stage2_candidates(text_out)

    except Exception as e:
        print(f"[warn] stage2 selection failed: {e}")
        chosen = []

    # Validate + keep ONLY 1
    valid_ids = set(stage1.keys())
    chosen = [c for c in chosen if c.get("processed_id") in valid_ids][:1]

    # Fallback if model didn't return 1 valid candidate
    if len(chosen) < 1:
        ranked = []
        for pid, info in stage1.items():
            simp = info.get("simplicity", 5)
            try:
                simp_i = int(simp)
            except Exception:
                simp_i = 5
            simp_i = max(1, min(5, simp_i))

            objs = info.get("objects") or []
            nobj = len(objs) if isinstance(objs, list) else 999

            sc = info.get("score", None)
            try:
                sc_f = float(sc) if isinstance(sc, (int, float)) else -1.0
            except Exception:
                sc_f = -1.0

            # lower simplicity better, higher score better, fewer objects better
            ranked.append((simp_i, -sc_f, nobj, pid))

        ranked.sort()
        fallback_pid = ranked[0][3] if ranked else None

        chosen = []
        if fallback_pid:
            chosen.append({
                "processed_id": fallback_pid,
                "reason": "fallback rank",
                "main_object": stage1.get(fallback_pid, {}).get("dominant", "") or "",
                "simplicity": int(stage1.get(fallback_pid, {}).get("simplicity", 5)),
            })

    # Attach unique path + score (authoritative) and keep ONLY 1
    final_candidates: List[Dict[str, Any]] = []
    if chosen:
        c = chosen[0]
        pid = c.get("processed_id", "")
        info = stage1.get(pid, {}) if isinstance(stage1.get(pid, {}), dict) else {}

        sc = info.get("score", None)
        sc_out = None
        try:
            if isinstance(sc, (int, float)):
                sc_out = float(sc)
        except Exception:
            sc_out = None

        final_candidates.append({
            "processed_id": pid,
            "unique_path": str(processed_to_unique.get(pid, "")),
            "reason": str(c.get("reason", "")),
            "main_object": str(c.get("main_object", "")),
            "simplicity": int(c.get("simplicity", 5)),
            "score": sc_out,
        })

    return {
        "base_context": (base_context or "").strip(),
        "stage1": stage1,
        "candidates": final_candidates,  # list with exactly 1 dict (or empty if stage1 empty)
    }


import json
import re
from typing import Any, Dict, List, Optional

import torch


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Pull the first {...} JSON object out of model text, tolerant of pre/post junk.
    """
    if not isinstance(text, str):
        return None

    # fast path
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    # find first balanced {...}
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1]
                try:
                    return json.loads(chunk)
                except Exception:
                    return None
    return None


def _extract_all_json_objects(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(text, str):
        return out
    source = str(text or "")
    for start in range(len(source)):
        if source[start] != "{":
            continue
        depth = 0
        for end in range(start, len(source)):
            ch = source[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = source[start:end + 1]
                    try:
                        obj = json.loads(chunk)
                    except Exception:
                        obj = None
                    if isinstance(obj, dict):
                        out.append(obj)
                    break
    return out


def _extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
    rows = _extract_all_json_objects(text)
    return rows[-1] if rows else None


def _strip_qwen_thinking_text(text: str) -> str:
    raw = str(text or "")
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    elif "<think>" in raw:
        return ""
    raw = re.sub(r"<\|[^>]+\|>", " ", raw)
    raw = raw.replace("<think>", " ").replace("</think>", " ")
    return raw.strip()


def _decode_token_ids_text(processor_or_tokenizer: Any, token_ids: Any, *, skip_special_tokens: bool = True) -> str:
    if token_ids is None:
        return ""
    try:
        if hasattr(processor_or_tokenizer, "batch_decode"):
            rows = processor_or_tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
            if isinstance(rows, list) and rows:
                return str(rows[0] or "")
        tok = getattr(processor_or_tokenizer, "tokenizer", None)
        if tok is not None and hasattr(tok, "batch_decode"):
            rows = tok.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
            if isinstance(rows, list) and rows:
                return str(rows[0] or "")
        if hasattr(processor_or_tokenizer, "decode"):
            item = token_ids[0] if isinstance(token_ids, (list, tuple)) and token_ids else token_ids
            return str(processor_or_tokenizer.decode(item, skip_special_tokens=skip_special_tokens) or "")
        if tok is not None and hasattr(tok, "decode"):
            item = token_ids[0] if isinstance(token_ids, (list, tuple)) and token_ids else token_ids
            return str(tok.decode(item, skip_special_tokens=skip_special_tokens) or "")
    except Exception:
        pass
    return ""


def _infer_text_model_device(model: Any, fallback: Any = None) -> torch.device:
    if isinstance(fallback, torch.device):
        return fallback
    try:
        if isinstance(fallback, str) and fallback:
            return torch.device(fallback)
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_action_planner_text_bundle(
    model_id: str = ACTION_PLANNER_TEXT_MODEL_ID,
    stage_io_dir: Optional[str] = None,
) -> Dict[str, Any]:
    model_id = _resolved_text_model_id(model_id)
    if _should_use_vllm_for_text_model(model_id):
        try:
            llm = _build_vllm_text_engine(model_id)
            tokenizer = _get_vllm_tokenizer(llm)
            return {
                "model_id": model_id,
                "backend": "vllm",
                "llm": llm,
                "tokenizer": tokenizer,
                "device": torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() and not FORCE_CPU else "cpu"),
                "stage_io_dir": str(stage_io_dir or "").strip(),
            }
        except Exception as e:
            print(f"[qwen][DBG] action_planner_text_bundle vllm_unavailable fallback err={type(e).__name__}: {e}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return {
        "model_id": model_id,
        "backend": "transformers",
        "model": model,
        "tokenizer": tokenizer,
        "device": _infer_text_model_device(model),
        "stage_io_dir": str(stage_io_dir or "").strip(),
    }


def destroy_action_planner_text_bundle(bundle: Any) -> None:
    if not isinstance(bundle, dict):
        return
    try:
        bundle["llm"] = None
    except Exception:
        pass
    try:
        bundle["model"] = None
    except Exception:
        pass
    try:
        bundle["tokenizer"] = None
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_qwen_multimodal_model(
    model_id: str,
    *,
    device: torch.device,
    load_kwargs: Dict[str, Any],
):
    tried: List[str] = []
    for loader_name, loader in (
        ("Qwen3_5ForConditionalGeneration", Qwen3_5ForConditionalGeneration),
        ("AutoModelForImageTextToText", AutoModelForImageTextToText),
        ("Qwen3VLForConditionalGeneration", Qwen3VLForConditionalGeneration),
    ):
        if loader is None:
            tried.append(f"{loader_name}:missing")
            continue
        try:
            return loader.from_pretrained(model_id, **load_kwargs).eval()
        except Exception as e:
            tried.append(f"{loader_name}:{type(e).__name__}")
            continue
    raise RuntimeError(f"no_multimodal_loader_succeeded:{','.join(tried)}")


def _load_qwen_vl_direct(
    model_id: str,
    *,
    load_kwargs: Dict[str, Any],
):
    if Qwen3VLForConditionalGeneration is not None:
        return Qwen3VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs).eval()
    return AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs).eval()


def create_diagram_multimodal_bundle(
    model_id: str = DIAGRAM_MULTIMODAL_MODEL_ID,
    stage_io_dir: Optional[str] = None,
) -> Dict[str, Any]:
    model_id = _resolved_multimodal_model_id(model_id)
    have_cuda = torch.cuda.is_available() and not FORCE_CPU
    device = torch.device(f"cuda:{GPU_INDEX}" if have_cuda else "cpu")

    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=QWEN_MIN_PIXELS,
            max_pixels=QWEN_MAX_PIXELS,
            trust_remote_code=True,
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.padding_side = "left"
        tok.truncation_side = "left"
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id

    load_kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    attn_impl = _qwen_attn_impl()
    if attn_impl is not None:
        load_kwargs["attn_implementation"] = attn_impl

    if have_cuda:
        load_kwargs["torch_dtype"] = DTYPE
        load_kwargs["device_map"] = {"": GPU_INDEX}
        with _qwen_safe_loader_context():
            model = _load_qwen_multimodal_model(model_id, device=device, load_kwargs=load_kwargs)
        model = _maybe_enable_qwen_fast_inference(
            model,
            device,
            allow_compile=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float32
        model = _load_qwen_multimodal_model(model_id, device=device, load_kwargs=load_kwargs).to(device)

    if tok is not None and hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id
    try:
        model.config.use_cache = True
    except Exception:
        pass

    return {
        "model_id": model_id,
        "model": model,
        "processor": processor,
        "tokenizer": tok if tok is not None else processor,
        "device": device,
        "stage_io_dir": str(stage_io_dir or "").strip(),
    }


def destroy_diagram_multimodal_bundle(bundle: Any) -> None:
    if not isinstance(bundle, dict):
        return
    try:
        bundle["model"] = None
    except Exception:
        pass
    try:
        bundle["processor"] = None
    except Exception:
        pass
    try:
        bundle["tokenizer"] = None
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def create_diagram_text_bundle(
    model_id: str = DIAGRAM_MULTIMODAL_MODEL_ID,
) -> Dict[str, Any]:
    return create_diagram_multimodal_bundle(model_id=model_id)


def destroy_diagram_text_bundle(bundle: Any) -> None:
    destroy_diagram_multimodal_bundle(bundle)


def _coerce_unique_ints(values: Any) -> List[int]:
    out: List[int] = []
    seen = set()
    if not isinstance(values, list):
        return out
    for item in values:
        try:
            value = int(item)
        except Exception:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _coerce_unique_strs(values: Any, *, cap: int = 32) -> List[str]:
    out: List[str] = []
    seen = set()
    if not isinstance(values, list):
        return out
    for item in values:
        value = str(item or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if cap > 0 and len(out) >= cap:
            break
    return out


def _clip_text_block(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def _run_text_model_json_jobs(
    *,
    bundle: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    prompt_builder,
    parser,
    debug_label: str,
    temperature: float,
    max_new_tokens: int,
    thinking_enabled: bool = False,
    max_rounds: int = 4,
    batch_size: int = 8,
) -> Dict[str, Any]:
    if not isinstance(bundle, dict):
        return {}
    active_jobs = [job for job in (jobs or []) if isinstance(job, dict) and str(job.get("job_key", "") or "").strip()]
    if not active_jobs:
        return {}

    prompts: List[str] = [str(prompt_builder(job) or "") for job in active_jobs]
    print(
        f"[qwen][DBG] {str(debug_label or 'diagram_text_jobs')} jobs_start "
        f"count={len(active_jobs)} thinking={int(bool(thinking_enabled))} "
        f"max_rounds={int(max_rounds)} batch_size={max(1, int(batch_size or 1))} "
        f"prompt_chars_total={sum(len(p) for p in prompts)} "
        f"prompt_chars_max={max((len(p) for p in prompts), default=0)} "
        f"vram={_vram_str()}"
    )
    parsed_objs, raws = _generate_json_objects_from_prompts_text_model(
        llm=bundle.get("llm"),
        model=bundle.get("model"),
        tokenizer=bundle.get("tokenizer"),
        device=bundle.get("device"),
        prompts=prompts,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        debug_label=str(debug_label or "diagram_text_jobs"),
        thinking_enabled=bool(thinking_enabled),
        max_rounds=int(max_rounds),
        batch_size=max(1, int(batch_size or 1)),
        stage_io_contexts=[
            _build_qwen_stage_io_context(
                bundle=bundle,
                debug_label=str(debug_label or "diagram_text_jobs"),
                job_key=str(job.get("job_key", "") or f"job_{idx + 1}"),
                stage_kind="text",
                rendered_input_text=str(prompts[idx] if idx < len(prompts) else ""),
                input_prompt_text=str(prompts[idx] if idx < len(prompts) else ""),
                extra={
                    "thinking_enabled": bool(thinking_enabled),
                    "max_rounds": int(max_rounds),
                    "temperature": float(temperature),
                    "max_new_tokens": int(max_new_tokens),
                    "job_input": job,
                },
            )
            for idx, job in enumerate(active_jobs)
        ],
    )
    print(
        f"[qwen][DBG] {str(debug_label or 'diagram_text_jobs')} jobs_done "
        f"count={len(active_jobs)} parsed={len(parsed_objs)} raws={len(raws)} vram={_vram_str()}"
    )

    out: Dict[str, Any] = {}
    for idx, job in enumerate(active_jobs):
        job_key = str(job.get("job_key", "") or "").strip()
        parsed = parsed_objs[idx] if idx < len(parsed_objs) and isinstance(parsed_objs[idx], dict) else {}
        raw_text = raws[idx] if idx < len(raws) else ""
        try:
            out[job_key] = parser(job=job, parsed_obj=parsed, raw_text=raw_text)
        except Exception as e:
            out[job_key] = {
                "error": f"{type(e).__name__}: {e}",
                "raw_text": raw_text,
            }
    return out


def _sanitize_qwen_stage_name(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or fallback


def _qwen_tokenizer_from_any(value: Any) -> Any:
    if value is None:
        return None
    tok = getattr(value, "tokenizer", None)
    if tok is not None:
        return tok
    return value


def _safe_qwen_token_count(tokenizer_or_processor: Any, text: Any) -> Optional[int]:
    tok = _qwen_tokenizer_from_any(tokenizer_or_processor)
    if tok is None:
        return None
    try:
        encoded = tok(str(text or ""), add_special_tokens=False, return_attention_mask=False)
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
        if isinstance(input_ids, list):
            return int(len(input_ids))
    except Exception:
        pass
    try:
        if hasattr(tok, "encode"):
            return int(len(tok.encode(str(text or ""), add_special_tokens=False)))
    except Exception:
        pass
    return None


def _build_qwen_stage_io_context(
    *,
    bundle: Optional[Dict[str, Any]],
    debug_label: str,
    job_key: str,
    stage_kind: str,
    rendered_input_text: str,
    input_prompt_text: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "bundle": bundle if isinstance(bundle, dict) else None,
        "debug_label": str(debug_label or ""),
        "job_key": str(job_key or ""),
        "stage_kind": str(stage_kind or ""),
        "rendered_input_text": str(rendered_input_text or ""),
    }
    if input_prompt_text is not None:
        ctx["input_prompt_text"] = str(input_prompt_text or "")
    if system_prompt is not None:
        ctx["system_prompt"] = str(system_prompt or "")
    if user_prompt is not None:
        ctx["user_prompt"] = str(user_prompt or "")
    if isinstance(extra, dict) and extra:
        ctx["extra"] = extra
    return ctx


def _write_qwen_stage_io_json(
    bundle: Optional[Dict[str, Any]],
    *,
    tokenizer_or_processor: Any = None,
    debug_label: str,
    job_key: str,
    prompt_text: str,
    raw_text: str,
    parsed_obj: Any,
    parsed_payload: Any,
    stage_kind: str,
    rendered_input_text: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    out_dir_raw = ""
    if isinstance(bundle, dict):
        out_dir_raw = str(bundle.get("stage_io_dir", "") or "").strip()
    if not out_dir_raw:
        out_dir_raw = str(os.getenv("QWEN_STAGE_IO_DIR", "") or "").strip()
    if not out_dir_raw:
        return None
    try:
        out_dir = Path(out_dir_raw)
        out_dir.mkdir(parents=True, exist_ok=True)
        counter = 1
        if isinstance(bundle, dict):
            counter = int(bundle.get("_stage_io_counter", 0) or 0) + 1
            bundle["_stage_io_counter"] = counter
        else:
            counter = int(next(_QWEN_STAGE_IO_GLOBAL_COUNTER))
        input_text = str(rendered_input_text if rendered_input_text is not None else prompt_text or "")
        output_text = str(raw_text or "")
        filename = (
            f"{counter:05d}_"
            f"{_sanitize_qwen_stage_name(debug_label, fallback='qwen_stage')}_"
            f"{_sanitize_qwen_stage_name(job_key, fallback='job')}.json"
        )
        payload: Dict[str, Any] = {
            "schema": "qwen_stage_io_v1",
            "stage_kind": str(stage_kind or ""),
            "stage_label": str(debug_label or ""),
            "job_key": str(job_key or ""),
            "created_at_epoch_ms": int(time.time() * 1000),
            "model_id": str(bundle.get("model_id", "") or "") if isinstance(bundle, dict) else "",
            "backend": str(bundle.get("backend", "") or "") if isinstance(bundle, dict) else "",
            "input": {
                "system_prompt": str(system_prompt or ""),
                "user_prompt": str(user_prompt or ""),
                "prompt_text": str(prompt_text or ""),
                "rendered_model_input": input_text,
            },
            "output": {
                "raw_text": output_text,
            },
            "token_counts": {
                "input": _safe_qwen_token_count(tokenizer_or_processor, input_text),
                "output": _safe_qwen_token_count(tokenizer_or_processor, output_text),
            },
            "parsed_output": parsed_payload,
            "parsed_model_object": parsed_obj,
        }
        if isinstance(extra, dict) and extra:
            payload["extra"] = extra
        path = out_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
    except Exception:
        return None


def _open_rgb_image_for_qwen(path: Any) -> Optional[Image.Image]:
    clean = str(path or "").strip()
    if not clean:
        return None
    try:
        with Image.open(clean) as im:
            return im.convert("RGB")
    except Exception:
        return None


def _run_multimodal_json_jobs(
    *,
    bundle: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    user_prompt_builder,
    parser,
    image_builder,
    debug_label: str,
    temperature: float,
    max_new_tokens: int,
    system_prompt: str,
    batch_size: int = 8,
) -> Dict[str, Any]:
    if not isinstance(bundle, dict):
        return {}
    processor = bundle.get("processor")
    if processor is None or not hasattr(processor, "apply_chat_template"):
        return {}

    active_jobs: List[Dict[str, Any]] = []
    user_prompts: List[str] = []
    chat_texts: List[str] = []
    images: List[Image.Image] = []
    for job in jobs or []:
        if not isinstance(job, dict):
            continue
        job_key = str(job.get("job_key", "") or "").strip()
        if not job_key:
            continue
        image = image_builder(job)
        if not isinstance(image, Image.Image):
            continue
        prompt_text = str(user_prompt_builder(job) or "").strip()
        if not prompt_text:
            continue
        conv = [
            {"role": "system", "content": [{"type": "text", "text": str(system_prompt or "").strip()}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
        ]
        chat_text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        active_jobs.append(job)
        user_prompts.append(prompt_text)
        chat_texts.append(str(chat_text or ""))
        images.append(image)

    if not active_jobs:
        return {}

    parsed_objs, raws = _generate_json_objects_from_prompts(
        model=bundle.get("model"),
        processor=processor,
        device=bundle.get("device"),
        prompts=chat_texts,
        images=images,
        prompts_are_chat_text=True,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        batch_size=max(1, int(batch_size or 1)),
        debug_label=str(debug_label or "diagram_vl_jobs"),
        stage_io_contexts=[
            _build_qwen_stage_io_context(
                bundle=bundle,
                debug_label=str(debug_label or "diagram_vl_jobs"),
                job_key=str(job.get("job_key", "") or f"job_{idx + 1}"),
                stage_kind="multimodal",
                rendered_input_text=str(chat_texts[idx] if idx < len(chat_texts) else ""),
                input_prompt_text=str(user_prompts[idx] if idx < len(user_prompts) else ""),
                system_prompt=str(system_prompt or ""),
                user_prompt=str(user_prompts[idx] if idx < len(user_prompts) else ""),
                extra={
                    "temperature": float(temperature),
                    "max_new_tokens": int(max_new_tokens),
                    "job_input": job,
                },
            )
            for idx, job in enumerate(active_jobs)
        ],
    )

    out: Dict[str, Any] = {}
    for idx, job in enumerate(active_jobs):
        job_key = str(job.get("job_key", "") or "").strip()
        parsed = parsed_objs[idx] if idx < len(parsed_objs) and isinstance(parsed_objs[idx], dict) else {}
        raw_text = raws[idx] if idx < len(raws) else ""
        try:
            out[job_key] = parser(job=job, parsed_obj=parsed, raw_text=raw_text)
        except Exception as e:
            out[job_key] = {
                "error": f"{type(e).__name__}: {e}",
                "raw_text": raw_text,
            }
    return out


def build_diagram_snapshot_refinement_prompt(
    *,
    base_context: str,
    refined_labels: List[str],
    diagram_payload: Dict[str, Any],
) -> str:
    return (
        "You are compressing raw per-stroke diagram descriptions into a stroke-faithful mental map.\n"
        "Your job is NOT to solve the diagram yet.\n"
        "Your job is to bundle nearby or mechanically related strokes into a small set of SNAPSHOTS.\n"
        "A snapshot is a local cutout of the diagram with a location phrase, a bundle reason, and preserved stroke-level notes.\n"
        "You must preserve stroke ids carefully.\n"
        "You may create snapshots from one stroke, one group, several neighboring groups, or mixed local bundles.\n"
        "Standout strokes may become their own snapshot if they are unusually large, central, or detailed.\n"
        "Mention neighboring snapshots when useful.\n"
        "Use flowing language, but keep the JSON compact.\n"
        "Do not invent stroke ids.\n"
        "Do not omit stroke-level notes entirely.\n"
        "Prefer 4 to 12 snapshots unless the drawing is truly tiny or huge.\n"
        "Each snapshot should feel like a local reading of one region of the diagram.\n\n"
        f"BASE_CONTEXT:\n{str(base_context or '').strip() or 'unknown'}\n\n"
        "CANONICAL_LABEL_HINTS:\n"
        f"{json.dumps([str(x).strip() for x in (refined_labels or []) if str(x).strip()][:80], ensure_ascii=False)}\n\n"
        "RAW_DIAGRAM_PAYLOAD:\n"
        f"{json.dumps(diagram_payload, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON only with this schema:\n"
        "{"
        "\"snapshots\":["
        "{"
        "\"snapshot_id\":\"S1\","
        "\"location\":\"bottom-left / center-right / upper-middle style phrase\","
        "\"stroke_ids\":[1,2,3],"
        "\"group_indexes\":[4,5],"
        "\"neighbor_snapshot_ids\":[\"S2\"],"
        "\"summary\":\"compact flowing description of the local cutout\","
        "\"bundle_reason\":\"why these strokes belong together\","
        "\"stroke_notes\":[{\"stroke_id\":1,\"note\":\"short note that preserves a stroke-specific detail\"}]"
        "}"
        "]"
        "}\n"
    )


def _parse_diagram_snapshot_refinement(
    *,
    job: Dict[str, Any],
    parsed_obj: Dict[str, Any],
    raw_text: str,
) -> Dict[str, Any]:
    raw_rows = parsed_obj.get("snapshots") if isinstance(parsed_obj.get("snapshots"), list) else []
    snapshots: List[Dict[str, Any]] = []
    used_snapshot_ids = set()

    for index, row in enumerate(raw_rows, start=1):
        if not isinstance(row, dict):
            continue
        snapshot_id = str(row.get("snapshot_id", "") or f"S{index}").strip() or f"S{index}"
        if snapshot_id in used_snapshot_ids:
            snapshot_id = f"S{index}"
        used_snapshot_ids.add(snapshot_id)

        stroke_ids = _coerce_unique_ints(row.get("stroke_ids"))
        if not stroke_ids:
            continue

        stroke_notes: List[Dict[str, Any]] = []
        for note_row in row.get("stroke_notes") if isinstance(row.get("stroke_notes"), list) else []:
            if not isinstance(note_row, dict):
                continue
            try:
                stroke_id = int(note_row.get("stroke_id"))
            except Exception:
                continue
            if stroke_id not in stroke_ids:
                continue
            stroke_note = _clip_text_block(note_row.get("note", ""), 180)
            if not stroke_note:
                continue
            stroke_notes.append({"stroke_id": stroke_id, "note": stroke_note})

        snapshots.append(
            {
                "snapshot_id": snapshot_id,
                "location": _clip_text_block(row.get("location", ""), 80),
                "stroke_ids": stroke_ids,
                "group_indexes": _coerce_unique_ints(row.get("group_indexes")),
                "neighbor_snapshot_ids": _coerce_unique_strs(row.get("neighbor_snapshot_ids"), cap=8),
                "summary": _clip_text_block(row.get("summary", ""), 420),
                "bundle_reason": _clip_text_block(row.get("bundle_reason", ""), 220),
                "stroke_notes": stroke_notes[:16],
            }
        )

    return {
        "snapshots": snapshots,
        "raw_text": raw_text,
    }


def refine_diagram_snapshots_text_model(
    *,
    bundle: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    temperature: float = 0.15,
    max_new_tokens: int = 900,
) -> Dict[str, Any]:
    return _run_text_model_json_jobs(
        bundle=bundle,
        jobs=jobs,
        prompt_builder=lambda job: build_diagram_snapshot_refinement_prompt(
            base_context=str(job.get("base_context", "") or ""),
            refined_labels=list(job.get("refined_labels", []) or []),
            diagram_payload=job.get("diagram_payload") if isinstance(job.get("diagram_payload"), dict) else {},
        ),
        parser=_parse_diagram_snapshot_refinement,
        debug_label="diagram_snapshot_refine",
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        thinking_enabled=False,
        max_rounds=3,
    )


def build_cluster_visual_summary_prompt(
    *,
    base_context: str,
    refined_labels: List[str],
    cluster_payload: Dict[str, Any],
) -> str:
    return (
        "You are making a compact visual reading of one diagram cluster from pixels.\n"
        "You are looking at a two-panel image.\n"
        "LEFT shows the isolated cluster render.\n"
        "RIGHT shows the full diagram with the same region boxed for context.\n"
        "Stay grounded in what is visibly present.\n"
        "Describe structure, shape, density, topology, orientation, branching, cavities, pipes, membranes, loops, or other visible geometry when relevant.\n"
        "You may interpret broad mechanical or biological-looking forms if the image strongly supports it, but do not force a canonical label.\n"
        "Mention local surroundings or neighboring context when visible.\n\n"
        f"BASE_CONTEXT:\n{str(base_context or '').strip() or 'unknown'}\n\n"
        "CANONICAL_LABEL_HINTS:\n"
        f"{json.dumps([str(x).strip() for x in (refined_labels or []) if str(x).strip()][:80], ensure_ascii=False)}\n\n"
        "CLUSTER_METADATA:\n"
        f"{json.dumps(cluster_payload, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON only with this schema:\n"
        "{"
        "\"visual_summary\":\"compact but detailed visual description\","
        "\"bundle_reason\":\"why these strokes form one cluster\","
        "\"location\":\"short location phrase\","
        "\"neighbor_context\":\"short note about nearby context in the larger diagram\","
        "\"dominant_cues\":[\"cue\"],"
        "\"visual_parts\":[\"part-like visible region or subform\"]"
        "}\n"
    )


def _parse_cluster_visual_summary(
    *,
    job: Dict[str, Any],
    parsed_obj: Dict[str, Any],
    raw_text: str,
) -> Dict[str, Any]:
    return {
        "visual_summary": _clip_text_block(parsed_obj.get("visual_summary", ""), 420),
        "bundle_reason": _clip_text_block(parsed_obj.get("bundle_reason", ""), 220),
        "location": _clip_text_block(parsed_obj.get("location", ""), 80),
        "neighbor_context": _clip_text_block(parsed_obj.get("neighbor_context", ""), 180),
        "dominant_cues": _coerce_unique_strs(parsed_obj.get("dominant_cues"), cap=10),
        "visual_parts": _coerce_unique_strs(parsed_obj.get("visual_parts"), cap=10),
        "raw_text": raw_text,
    }


def _build_cluster_visual_job_image(job: Dict[str, Any]) -> Optional[Image.Image]:
    cluster_path = str(job.get("cluster_render_path", "") or "").strip()
    full_path = str(job.get("full_image_path", "") or "").strip()
    bbox_xyxy = job.get("bbox_xyxy")
    if not cluster_path or not full_path or not (isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4):
        return None

    cluster_img = _open_rgb_image_for_qwen(cluster_path)
    full_img = _open_rgb_image_for_qwen(full_path)
    if cluster_img is None or full_img is None:
        return None

    panel_edge = max(64, int(PROC_LONGEST_EDGE) // 2)
    left_sq, _, _, _, _, _ = _fit_into_square(cluster_img, panel_edge)
    right_panel_base, full_scale, full_rw, full_rh, full_pad_x, full_pad_y = _fit_into_square(full_img, panel_edge)

    try:
        bbox_panel = _clamp_bbox_xyxy(_scale_bbox_xyxy([int(x) for x in bbox_xyxy], full_scale), full_rw, full_rh)
    except Exception:
        return None
    bbox_panel = [
        int(bbox_panel[0]) + int(full_pad_x),
        int(bbox_panel[1]) + int(full_pad_y),
        int(bbox_panel[2]) + int(full_pad_x),
        int(bbox_panel[3]) + int(full_pad_y),
    ]
    right_sq = right_panel_base.copy()
    _draw_red_rect_pil(right_sq, bbox_panel, thickness=RECT_THICKNESS_PX)
    return _make_composite(left_sq, right_sq)


def describe_diagram_clusters_multimodal_model(
    *,
    bundle: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    temperature: float = 0.7,
    max_new_tokens: int = DIAGRAM_CLUSTER_VISUAL_MAX_NEW_TOKENS,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, Any]:
    return _run_multimodal_json_jobs(
        bundle=bundle,
        jobs=jobs,
        user_prompt_builder=lambda job: build_cluster_visual_summary_prompt(
            base_context=str(job.get("base_context", "") or ""),
            refined_labels=list(job.get("refined_labels", []) or []),
            cluster_payload=job.get("cluster_payload") if isinstance(job.get("cluster_payload"), dict) else {},
        ),
        parser=_parse_cluster_visual_summary,
        image_builder=_build_cluster_visual_job_image,
        debug_label="diagram_cluster_visual_text",
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        system_prompt=(
            "You are a strict visual transcription engine. "
            "Stay grounded in visible pixels, summarize the pictured cluster, and return JSON only."
        ),
        batch_size=int(batch_size),
    )


def build_canonical_part_visual_prompt(
    *,
    base_context: str,
    component_payload: Dict[str, Any],
) -> str:
    return (
        "You are reading one canonical part reference image.\n"
        "Describe what the visible part looks like in image-grounded terms.\n"
        "You may use broad interpretation when the image strongly supports it, such as pipe, chamber, branch, wheel-like ring, layered membrane, blade, arm, connector, or housing.\n"
        "Focus on shape, topology, openings, branching, symmetry, density, silhouette, and visually diagnostic cues.\n"
        "Use the metadata only as weak support and never override the pixels.\n\n"
        f"BASE_CONTEXT:\n{str(base_context or '').strip() or 'unknown'}\n\n"
        "CANONICAL_PART_PAYLOAD:\n"
        f"{json.dumps(component_payload, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON only with this schema:\n"
        "{"
        "\"canonical_visual_parts\":\"compact target visual description\","
        "\"notable_cues\":[\"cue\"],"
        "\"candidate_note\":\"optional short note about the chosen canonical candidate\""
        "}\n"
    )


def _parse_canonical_part_visual(
    *,
    job: Dict[str, Any],
    parsed_obj: Dict[str, Any],
    raw_text: str,
) -> Dict[str, Any]:
    return {
        "canonical_visual_parts": _clip_text_block(parsed_obj.get("canonical_visual_parts", ""), 360),
        "notable_cues": _coerce_unique_strs(parsed_obj.get("notable_cues"), cap=10),
        "candidate_note": _clip_text_block(parsed_obj.get("candidate_note", ""), 180),
        "raw_text": raw_text,
    }


def _build_canonical_part_job_image(job: Dict[str, Any]) -> Optional[Image.Image]:
    image_path = str(job.get("candidate_image_path", "") or "").strip()
    if not image_path:
        return None
    image = _open_rgb_image_for_qwen(image_path)
    if image is None:
        return None
    return _resize_longest_side(image, int(PROC_LONGEST_EDGE))


def describe_canonical_parts_multimodal_model(
    *,
    bundle: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    temperature: float = 0.65,
    max_new_tokens: int = 280,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, Any]:
    return _run_multimodal_json_jobs(
        bundle=bundle,
        jobs=jobs,
        user_prompt_builder=lambda job: build_canonical_part_visual_prompt(
            base_context=str(job.get("base_context", "") or ""),
            component_payload=job.get("component_payload") if isinstance(job.get("component_payload"), dict) else {},
        ),
        parser=_parse_canonical_part_visual,
        image_builder=_build_canonical_part_job_image,
        debug_label="diagram_canonical_part_visual_text",
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        system_prompt=(
            "You are a strict visual reference summarizer. "
            "Describe the visible canonical part in grounded image terms and return JSON only."
        ),
        batch_size=int(batch_size),
    )


def describe_diagram_clusters_text_model(
    *,
    bundle: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    temperature: float = 0.7,
    max_new_tokens: int = 420,
) -> Dict[str, Any]:
    return describe_diagram_clusters_multimodal_model(
        bundle=bundle,
        jobs=jobs,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
    )


def describe_canonical_parts_text_model(
    *,
    bundle: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    temperature: float = 0.65,
    max_new_tokens: int = 280,
) -> Dict[str, Any]:
    return describe_canonical_parts_multimodal_model(
        bundle=bundle,
        jobs=jobs,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
    )


def build_diagram_final_match_prompt(
    *,
    base_context: str,
    refined_labels: List[str],
    snapshots: List[Dict[str, Any]],
    clusters: List[Dict[str, Any]],
    canonical_parts: List[Dict[str, Any]],
    stroke_lookup: Dict[str, Dict[str, Any]],
) -> str:
    return (
        "You are matching canonical diagram parts to exact stroke ids.\n"
        "Work stroke-first.\n"
        "You must produce exactly one match row for every canonical label.\n"
        "A match may come from:\n"
        "- one ready cluster,\n"
        "- one full snapshot,\n"
        "- a custom subset of strokes pulled from one or more snapshots,\n"
        "- one single stroke.\n"
        "Choose whichever is best.\n"
        "Use the cluster package when it is already a good bundle.\n"
        "Break clusters apart or recombine strokes when the cluster is too coarse or wrong.\n"
        "Use the snapshot mental map and the per-stroke notes to reason locally.\n"
        "Every label should get one best stroke group.\n"
        "Do not waste space on prose outside the JSON.\n"
        "Be explicit about which ids justify the match.\n"
        "Avoid reusing the same stroke ids for many labels unless the overlap is genuinely unavoidable.\n\n"
        f"BASE_CONTEXT:\n{str(base_context or '').strip() or 'unknown'}\n\n"
        "LABEL_ORDER:\n"
        f"{json.dumps([str(x).strip() for x in (refined_labels or []) if str(x).strip()][:120], ensure_ascii=False)}\n\n"
        "CANONICAL_PARTS:\n"
        "For each label, CANONICAL_VISUAL_PARTS is the direct description of the chosen canonical reference image.\n"
        "GENERAL_VISUAL_DESCRIPTION is the upstream refined textual description for that part.\n"
        f"{json.dumps(canonical_parts, ensure_ascii=False, indent=2)}\n\n"
        "SNAPSHOTS:\n"
        f"{json.dumps(snapshots, ensure_ascii=False, indent=2)}\n\n"
        "CLUSTERS:\n"
        "Each cluster carries its stroke ids and a direct image-grounded visual description from the cluster render.\n"
        f"{json.dumps(clusters, ensure_ascii=False, indent=2)}\n\n"
        "STROKE_LOOKUP:\n"
        f"{json.dumps(stroke_lookup, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON only with this schema:\n"
        "{"
        "\"matches\":["
        "{"
        "\"label\":\"exact canonical label\","
        "\"source_type\":\"cluster|snapshot|strokes|single_stroke|unmatched\","
        "\"source_key\":\"cluster_id or snapshot_id or custom key\","
        "\"stroke_ids\":[1,2,3],"
        "\"snapshot_ids\":[\"S1\"],"
        "\"cluster_ids\":[\"C1\"],"
        "\"confidence\":0.0,"
        "\"reason\":\"short stroke-level explanation\""
        "}"
        "]"
        "}\n"
    )


def _parse_diagram_final_matches(
    *,
    job: Dict[str, Any],
    parsed_obj: Dict[str, Any],
    raw_text: str,
) -> Dict[str, Any]:
    refined_labels = [str(x).strip() for x in (job.get("refined_labels") or []) if str(x).strip()]
    label_set = {label.lower(): label for label in refined_labels}
    raw_matches = parsed_obj.get("matches") if isinstance(parsed_obj.get("matches"), list) else []
    by_label: Dict[str, Dict[str, Any]] = {}

    for row in raw_matches:
        if not isinstance(row, dict):
            continue
        label_raw = str(row.get("label", "") or "").strip()
        label = label_set.get(label_raw.lower())
        if not label:
            continue
        source_type = str(row.get("source_type", "") or "").strip().lower()
        if source_type not in {"cluster", "snapshot", "strokes", "single_stroke", "unmatched"}:
            source_type = "strokes"
        stroke_ids = _coerce_unique_ints(row.get("stroke_ids"))
        if source_type == "single_stroke" and len(stroke_ids) > 1:
            stroke_ids = stroke_ids[:1]
        confidence = _coerce_float_01(row.get("confidence", 0.0))
        reason = _clip_text_block(row.get("reason", ""), 240)
        candidate = {
            "label": label,
            "source_type": source_type if stroke_ids else "unmatched",
            "source_key": _clip_text_block(row.get("source_key", ""), 80),
            "stroke_ids": stroke_ids,
            "snapshot_ids": _coerce_unique_strs(row.get("snapshot_ids"), cap=12),
            "cluster_ids": _coerce_unique_strs(row.get("cluster_ids"), cap=12),
            "confidence": confidence if stroke_ids else 0.0,
            "reason": reason if reason else ("unmatched" if not stroke_ids else ""),
        }
        prev = by_label.get(label.lower())
        if prev is None or float(candidate["confidence"]) > float(prev.get("confidence", 0.0)):
            by_label[label.lower()] = candidate

    matches: List[Dict[str, Any]] = []
    for label in refined_labels:
        row = by_label.get(label.lower())
        if not isinstance(row, dict):
            row = {
                "label": label,
                "source_type": "unmatched",
                "source_key": "",
                "stroke_ids": [],
                "snapshot_ids": [],
                "cluster_ids": [],
                "confidence": 0.0,
                "reason": "unmatched",
            }
        matches.append(row)

    return {
        "matches": matches,
        "raw_text": raw_text,
    }


def match_diagram_parts_text_model(
    *,
    bundle: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_new_tokens: int = 1200,
    thinking_enabled: bool = False,
) -> Dict[str, Any]:
    return _run_text_model_json_jobs(
        bundle=bundle,
        jobs=jobs,
        prompt_builder=lambda job: build_diagram_final_match_prompt(
            base_context=str(job.get("base_context", "") or ""),
            refined_labels=list(job.get("refined_labels", []) or []),
            snapshots=list(job.get("snapshots", []) or []),
            clusters=list(job.get("clusters", []) or []),
            canonical_parts=list(job.get("canonical_parts", []) or []),
            stroke_lookup=job.get("stroke_lookup") if isinstance(job.get("stroke_lookup"), dict) else {},
        ),
        parser=_parse_diagram_final_matches,
        debug_label="diagram_final_match_text",
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        thinking_enabled=bool(thinking_enabled),
        max_rounds=4,
    )


def _extract_action_list_from_model_obj(obj: Any) -> List[Dict[str, Any]]:
    """
    Accept a few common action-list key variants so planner prompt wording does not
    accidentally force us into the fallback path.
    """
    if not isinstance(obj, dict):
        return []

    candidate_keys = (
        "actions",
        "planned_actions",
        "visual_actions",
        "example actions",
        "example_actions",
        "steps",
    )
    for key in candidate_keys:
        acts = obj.get(key)
        if isinstance(acts, list):
            return acts

    for value in obj.values():
        if not isinstance(value, list) or not value:
            continue
        if all(isinstance(item, dict) and str(item.get("type", "") or "").strip() for item in value):
            return value
    return []


def _clamp_i(v: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(v), int(hi)))


def _bbox_overlap_xywh(a: Dict[str, int], b: Dict[str, int]) -> bool:
    ax0 = int(a.get("x", 0))
    ay0 = int(a.get("y", 0))
    ax1 = ax0 + int(a.get("w", 0))
    ay1 = ay0 + int(a.get("h", 0))
    bx0 = int(b.get("x", 0))
    by0 = int(b.get("y", 0))
    bx1 = bx0 + int(b.get("w", 0))
    by1 = by0 + int(b.get("h", 0))
    if ax1 <= bx0 or bx1 <= ax0:
        return False
    if ay1 <= by0 or by1 <= ay0:
        return False
    return True


def _normalize_print_bbox(
    pb: Optional[Dict[str, Any]],
    *,
    default_w: int,
    default_h: int,
    board_w: int,
    board_h: int,
) -> Dict[str, int]:
    if not isinstance(pb, dict):
        pb = {}
    w = int(pb.get("w", default_w) or default_w)
    h = int(pb.get("h", default_h) or default_h)
    # keep space boxes large, but cap around quadrant-ish scale so multiple entries can coexist
    max_w = max(80, int(board_w) // 2)
    max_h = max(60, int(board_h) // 2)
    w = max(80, min(w, max_w))
    h = max(60, min(h, max_h))

    x = int(pb.get("x", 20) or 20)
    y = int(pb.get("y", 20) or 20)
    max_x = max(0, int(board_w) - w)
    max_y = max(0, int(board_h) - h)
    x = _clamp_i(x, 0, max_x)
    y = _clamp_i(y, 0, max_y)
    return {"x": x, "y": y, "w": int(w), "h": int(h)}


def _normalize_text_coord(
    tc: Optional[Dict[str, Any]],
    *,
    print_bbox: Dict[str, int],
    text_w: int,
    text_h: int,
    board_w: int,
    board_h: int,
) -> Dict[str, int]:
    pbx = int(print_bbox.get("x", 0) or 0)
    pby = int(print_bbox.get("y", 0) or 0)
    pbw = int(print_bbox.get("w", 0) or 0)
    pbh = int(print_bbox.get("h", 0) or 0)

    min_x = pbx
    max_x = pbx + max(0, pbw - max(1, int(text_w)))
    min_y = pby
    max_y = pby + max(0, pbh - max(1, int(text_h)))

    if not isinstance(tc, dict):
        tc = {}
    x = int(tc.get("x", pbx) or pbx)
    y_default = pby + min(max(0, pbh - 1), max(0, int(text_h)))
    y = int(tc.get("y", y_default) or y_default)

    x = _clamp_i(x, min_x, max_x if max_x >= min_x else min_x)
    y = _clamp_i(y, min_y, max_y if max_y >= min_y else min_y)
    x = _clamp_i(x, 0, max(0, int(board_w) - 1))
    y = _clamp_i(y, 0, max(0, int(board_h) - 1))
    return {"x": int(x), "y": int(y)}


def _find_non_overlapping_box(
    desired: Dict[str, int],
    *,
    placed: List[Dict[str, int]],
    board_w: int,
    board_h: int,
    step: int = 40,
) -> Dict[str, int]:
    cand = _normalize_print_bbox(
        desired,
        default_w=int(desired.get("w", 240) or 240),
        default_h=int(desired.get("h", 180) or 180),
        board_w=board_w,
        board_h=board_h,
    )
    if all(not _bbox_overlap_xywh(cand, p) for p in placed):
        return cand

    max_x = max(0, int(board_w) - int(cand["w"]))
    max_y = max(0, int(board_h) - int(cand["h"]))

    def _scan(cur: Dict[str, int]) -> Optional[Dict[str, int]]:
        mx = max(0, int(board_w) - int(cur["w"]))
        my = max(0, int(board_h) - int(cur["h"]))
        yy = 0
        while yy <= my:
            xx = 0
            while xx <= mx:
                test = {"x": int(xx), "y": int(yy), "w": int(cur["w"]), "h": int(cur["h"])}
                if all(not _bbox_overlap_xywh(test, p) for p in placed):
                    return test
                xx += int(step)
            yy += int(step)
        return None

    found = _scan(cand)
    if found is not None:
        return found

    # If no space with current size, shrink progressively to force non-overlap placement.
    cur = dict(cand)
    for _ in range(8):
        cur["w"] = max(80, int(round(cur["w"] * 0.88)))
        cur["h"] = max(60, int(round(cur["h"] * 0.88)))
        found = _scan(cur)
        if found is not None:
            return found

    return cand


def _render_chat_text(processor, prompt: str, *, thinking_enabled: bool = False) -> str:
    try:
        tok = processor if hasattr(processor, "apply_chat_template") else getattr(processor, "tokenizer", None)
        if tok is not None and hasattr(tok, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                return tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking_enabled,
                )
            except TypeError:
                return tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": thinking_enabled},
                )
    except Exception:
        pass
    return prompt


def _clip_str(v: Any, max_chars: int) -> str:
    s = str(v or "").strip()
    if max_chars > 0 and len(s) > max_chars:
        return s[:max_chars].rstrip()
    return s


def _clip_str_list(values: Any, *, max_items: int, max_chars: int) -> List[str]:
    out: List[str] = []
    if not isinstance(values, list):
        return out
    for it in values:
        s = _clip_str(it, max_chars=max_chars)
        if not s:
            continue
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def _compact_diagram_cluster_labels(value: Any, *, max_items: int, max_chars: int) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    diagram_name = _clip_str(value.get("diagram_name", ""), 56)
    refined_labels = _clip_str_list(value.get("refined_labels"), max_items=max_items, max_chars=max_chars)
    if not diagram_name and not refined_labels:
        return None
    out: Dict[str, Any] = {}
    if diagram_name:
        out["diagram_name"] = diagram_name
    if refined_labels:
        out["refined_labels"] = refined_labels
    return out


def _visual_action_prompt_payload(ev: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event": _compact_visual_event_for_prompt(ev),
        "allowed_actions": {
           "Base actions schema": [
                "draw_image: {type,target,x,y,sync_local}",
                "write_text: {type,target,text,x,y,scale,sync_local}",
                "move_inside_bbox: {type,target,new_x,new_y,sync_local}",
                "link_to_image: {type,target,image_name,sync_local}",
                "delete_self: {type,sync_local}",
                "delete_by_name: {type,target,sync_local}",
            ],
            "diagram_extra": [
                "draw_cluster",
                "highlight_cluster",
                "zoom_cluster",
                "(all above use current-diagram cluster target/ref as \"diagram_name : cluster_name\")",
                "write_label: {type,cluster_name,text,sync_local}",
                "connect_cluster_to_cluster: {type,from_cluster,to_cluster,sync_local} where from_cluster/to_cluster are exactly \"diagram_name : cluster\"",
            ],
        },
    }


def build_visual_action_prompt(ev: Dict[str, Any]) -> str:
    payload = _visual_action_prompt_payload(ev)
    return (
        "You plan actions around drawing an image on a whiteboard under speech\n"
        "Use ONLY the allowed actions and exactly their field names\n"
        "Action MUST include sync_local {start_word_offset,end_word_offset} to link it to parts of the speech\n"
        "Actions must happen only inside print_bbox - respect image dimensions when picking print coords\n"
        f"The whiteboard is {WHITEBOARD_BOARD_WIDTH_PX} x {WHITEBOARD_BOARD_HEIGHT_PX} px, centered at 0,0.\n"
        f"Board text is large: default writing is about {WHITEBOARD_TEXT_DEFAULT_HEIGHT_PX}px tall, and write_text scale 1.0 is about {WHITEBOARD_TEXT_REFERENCE_HEIGHT_PX}px tall with about {WHITEBOARD_TEXT_LETTER_GAP_PX}px letter spacing.\n"
        "Sometimes images come with diagram = 1 or 2 - for them you have a list of interactable objects that comprise them\n"
        "Diagrams often come with large ranges with many inner objects mentioned - highlight and interact with them at every possible point.\n"
        "With diagrams you can either start with draw_image (self) OR draw them cluster by cluster (drawing them as they're mentioned) - one at a time\n"
        "You can use extra write_text actions to add detail in unused space\n"
        "Create the most rich and complex sync -> match what is said with as many different actions as possible\n"
        "For diagram == 1 or 2 you may also use diagram_extra\n"
        "If event.current_diagram_cluster_labels exists, those refined_labels belong ONLY to the current image being planned.\n"
        "If static_plan_context[i].context_other_diagram_cluster_labels exists, those refined_labels belong ONLY to context diagram.\n"
        "For diagram-only cluster actions, always identify the target cluster as \"diagram_name : cluster\".\n"
        "Use connect_cluster_to_cluster to link clusters of two diagrams in comparisons, both endpoints = \"diagram_name : cluster\".\n"
        "Use links with other images in the plan on shared ranges\n"
        "Use delete_self only if image is very temporary - unrelated to other images / speech\n"
        "When you need targeted cleanup, use delete_by_id with image_id from event/static_plan_context processed_id values.\n"
        "There is NO hard upper action cap per image.\n"
        "If the speech supports it, output a long actions array with many micro-synced actions.\n"
        "Generally with long speech and diagrams with many objects, go through most of the object list - either draw 1 by one or highlight"
        "Drawing a diagram takes around 5 seconds so only start actions on it after around 10 words"
        "Return JSON only.\n"
        "\n"
        "Input:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nOutput schema:\n"
        + json.dumps(
            {
                "actions": [
                    {"type": "draw_image", "target": "img_name", "x": 100, "y": 180, "sync_local": {"start_word_offset": 0, "end_word_offset": 2}},
                    {"type": "move_inside_bbox", "target": "img_name", "new_x": 180, "new_y": 220, "sync_local": {"start_word_offset": 4, "end_word_offset": 6}},
                    {"type": "highlight_cluster", "cluster_name": "Cell diagram : nucleus", "sync_local": {"start_word_offset": 7, "end_word_offset": 9}},
                    {"type": "write_label", "cluster_name": "Cell diagram : nucleus", "text": "Nucleus", "sync_local": {"start_word_offset": 8, "end_word_offset": 10}},
                    {"type": "connect_cluster_to_cluster", "from_cluster": "Cell diagram : nucleus", "to_cluster": "Mitosis diagram : prophase", "duration_sec": 2.5, "sync_local": {"start_word_offset": 9, "end_word_offset": 11}},
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def estimate_visual_action_prompt_chars(ev: Dict[str, Any]) -> int:
    try:
        return int(len(build_visual_action_prompt(ev)))
    except Exception:
        return 0


def _same_ci(a: Any, b: Any) -> bool:
    sa = str(a or "").strip()
    sb = str(b or "").strip()
    return bool(sa) and bool(sb) and sa.casefold() == sb.casefold()


def _blank_non_priority_overlaps(payload: Dict[str, Any], *, priority: str, fields: List[str]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    pval = str(payload.get(priority, "") or "").strip()
    if not pval:
        return payload
    for k in fields:
        if k == priority:
            continue
        if _same_ci(payload.get(k, ""), pval):
            payload[k] = ""
    return payload


def _planner_words(text: Any) -> List[str]:
    return [
        str(tok).strip().casefold()
        for tok in re.findall(r"[A-Za-z0-9]+(?:[+/\.-][A-Za-z0-9]+)*", str(text or ""))
        if str(tok).strip()
    ]


def _label_phrase_variants(label: Any) -> List[List[str]]:
    raw = str(label or "").strip()
    if not raw:
        return []

    candidates = [
        raw,
        raw.split("(", 1)[0].strip(),
        raw.replace("/", " "),
        raw.split(":", 1)[0].strip(),
    ]
    out: List[List[str]] = []
    seen = set()
    for cand in candidates:
        toks = tuple(_planner_words(cand))
        if not toks or toks in seen:
            continue
        seen.add(toks)
        out.append(list(toks))
    return out


def _find_phrase_offsets_in_words(words: List[str], phrase_words: List[str]) -> Optional[Tuple[int, int]]:
    if not words or not phrase_words:
        return None
    plen = len(phrase_words)
    if plen > len(words):
        return None
    for start in range(0, len(words) - plen + 1):
        if words[start : start + plen] == phrase_words:
            return start, start + plen - 1
    return None


def _map_word_idx_to_sync_offset(word_idx: int, *, total_words: int, span: int) -> int:
    if span <= 0:
        return 0
    if total_words <= 1:
        return _clamp_i(word_idx, 0, span)
    frac = float(max(0, min(word_idx, total_words - 1))) / float(max(1, total_words - 1))
    return _clamp_i(int(round(frac * span)), 0, span)


def _diagram_labels_for_event(ev: Dict[str, Any]) -> List[str]:
    raw_sources: List[Any] = []
    current = ev.get("current_diagram_cluster_labels")
    if isinstance(current, dict):
        raw_sources.extend(current.get("refined_labels") or [])
    raw_sources.extend(ev.get("objects_that_comprise_image") or [])

    out: List[str] = []
    seen = set()
    for item in raw_sources:
        label = _clean_short_label(str(item or "").strip())
        if not label:
            continue
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(label)
    return out


def _default_diagram_actions_for_event(
    ev: Dict[str, Any],
    *,
    span: int,
    draw_target: str,
) -> List[Dict[str, Any]]:
    labels = _diagram_labels_for_event(ev)
    if not labels:
        labels = [
            _clean_short_label(str(x or "").strip())
            for x in (ev.get("objects_that_comprise_image") or [])
        ]
        labels = [x for x in labels if x]
    if not labels:
        return []

    speech_words = _planner_words(ev.get("speech_text_in_range", ""))
    matched_rows: List[Tuple[str, int, int]] = []
    used = set()
    for label in labels:
        for variant in _label_phrase_variants(label):
            found = _find_phrase_offsets_in_words(speech_words, variant)
            if found is None:
                continue
            start_w, end_w = found
            matched_rows.append((label, start_w, end_w))
            used.add(label.casefold())
            break

    matched_rows.sort(key=lambda row: (row[1], row[2], row[0].casefold()))
    target_cluster_count = min(
        len(labels),
        max(3, min(12, max(4, (span // 28) + 1))),
    )
    selected_labels: List[Tuple[str, Optional[int], Optional[int]]] = [
        (label, start_w, end_w)
        for label, start_w, end_w in matched_rows[:target_cluster_count]
    ]

    if len(selected_labels) < target_cluster_count:
        remaining = [label for label in labels if label.casefold() not in used]
        need = max(0, target_cluster_count - len(selected_labels))
        for idx, label in enumerate(remaining[:need]):
            selected_labels.append((label, None, None))

    if not selected_labels:
        return []

    actions: List[Dict[str, Any]] = [
        {
            "type": "draw_image",
            "target": draw_target,
            "x": int((ev.get("print_bbox") or {}).get("x", 0) or 0),
            "y": int((ev.get("print_bbox") or {}).get("y", 0) or 0),
            "sync_local": {"start_word_offset": 0, "end_word_offset": min(span, 2)},
        }
    ]

    total_words = max(1, len(speech_words))
    label_window = max(2, min(14, max(2, span // max(1, len(selected_labels) * 2))))
    unmatched_count = sum(1 for _, sw, ew in selected_labels if sw is None or ew is None)
    unmatched_seen = 0
    for idx, (label, start_w, end_w) in enumerate(selected_labels):
        if start_w is None or end_w is None:
            unmatched_seen += 1
            base_start = _clamp_i(
                int(round((float(unmatched_seen) / float(max(1, unmatched_count + 1))) * span)),
                0,
                span,
            )
        else:
            base_start = _map_word_idx_to_sync_offset(start_w, total_words=total_words, span=span)
        base_end = min(span, max(base_start, base_start + label_window))
        actions.append(
            {
                "type": "highlight_cluster",
                "cluster_name": label,
                "sync_local": {
                    "start_word_offset": int(base_start),
                    "end_word_offset": int(base_end),
                },
            }
        )
        label_start = min(span, max(base_start, min(span, base_start + 1)))
        label_end = min(span, max(label_start, label_start + max(1, label_window // 2)))
        actions.append(
            {
                "type": "write_label",
                "cluster_name": label,
                "text": label,
                "sync_local": {
                    "start_word_offset": int(label_start),
                    "end_word_offset": int(label_end),
                },
            }
        )
        if idx < 3 and span >= 8:
            zoom_start = min(span, label_end)
            zoom_end = min(span, max(zoom_start, zoom_start + max(1, label_window // 2)))
            actions.append(
                {
                    "type": "zoom_cluster",
                    "cluster_name": label,
                    "sync_local": {
                        "start_word_offset": int(zoom_start),
                        "end_word_offset": int(zoom_end),
                    },
                }
            )

    return actions


WHITEBOARD_BOARD_WIDTH_PX = 2000
WHITEBOARD_BOARD_HEIGHT_PX = 2000
WHITEBOARD_TEXT_DEFAULT_HEIGHT_PX = 180
WHITEBOARD_TEXT_REFERENCE_HEIGHT_PX = 200
WHITEBOARD_TEXT_CHAR_WIDTH_PX = 140
WHITEBOARD_TEXT_LETTER_GAP_PX = 20


def _estimate_text_bbox_for_whiteboard(
    row: Dict[str, Any],
    *,
    board_w: int,
    board_h: int,
) -> Tuple[int, int]:
    bbox_px = row.get("bbox_px") if isinstance(row.get("bbox_px"), dict) else {}
    text_payload = str(row.get("write_text", "") or row.get("content", "") or row.get("name", "") or "").strip()
    text_chars = max(1, len(text_payload)) if text_payload else max(1, int(row.get("text_chars", 1) or 1))
    default_w = min(int(board_w), max(WHITEBOARD_TEXT_CHAR_WIDTH_PX * 2, text_chars * WHITEBOARD_TEXT_CHAR_WIDTH_PX))
    default_h = min(int(board_h), WHITEBOARD_TEXT_DEFAULT_HEIGHT_PX)
    bw = int(bbox_px.get("w", default_w) or default_w)
    bh = int(bbox_px.get("h", default_h) or default_h)
    bw = max(WHITEBOARD_TEXT_CHAR_WIDTH_PX, min(int(board_w), bw))
    bh = max(WHITEBOARD_TEXT_DEFAULT_HEIGHT_PX, min(int(board_h), bh))
    return int(bw), int(bh)


def _compact_space_chunk_for_prompt(chunk: Dict[str, Any], *, board_width: int, board_height: int) -> Dict[str, Any]:
    steps_src = chunk.get("steps") if isinstance(chunk, dict) else None
    entries_src = chunk.get("entries") if isinstance(chunk, dict) else None

    steps_out: List[Dict[str, Any]] = []
    for s in (steps_src or []):
        if not isinstance(s, dict):
            continue
        steps_out.append(
            {
                "key": _clip_str(s.get("key", ""), 24),
                "start_word_index": int(s.get("start_word_index", 0) or 0),
                "end_word_index": int(s.get("end_word_index", 0) or 0),
            }
        )

    entries_out: List[Dict[str, Any]] = []
    for e in (entries_src or []):
        if not isinstance(e, dict):
            continue
        et = _clip_str(e.get("type", ""), 16)
        name = _clip_str(e.get("name", ""), 56)
        row: Dict[str, Any] = {
            "entry_index": int(e.get("entry_index", 0) or 0),
            "type": et,
            "name": name,
            "step_key": _clip_str(e.get("step_key", ""), 24),
            "range_start": int(e.get("range_start", 0) or 0),
            "range_end": int(e.get("range_end", 0) or 0),
            "range_words": max(
                1,
                int(e.get("range_end", 0) or 0) - int(e.get("range_start", 0) or 0) + 1,
            ),
        }
        if et == "image":
            bbox_px = e.get("bbox_px") if isinstance(e.get("bbox_px"), dict) else None
            if isinstance(bbox_px, dict):
                row["bbox_px"] = {
                    "w": int(bbox_px.get("w", 400) or 400),
                    "h": int(bbox_px.get("h", 300) or 300),
                }
            if int(e.get("diagram", 0) or 0) > 0:
                row["diagram"] = 1
        elif et == "text":
            wt = str(e.get("write_text", "") or e.get("content", "") or "")
            wt = wt.strip()
            tw, th = _estimate_text_bbox_for_whiteboard(
                e,
                board_w=int(board_width),
                board_h=int(board_height),
            )
            row["bbox_px"] = {"w": int(tw), "h": int(th)}
            if wt:
                row["text_chars"] = int(len(wt))
        elif et == "silence":
            dur = e.get("duration_sec")
            try:
                if dur is not None:
                    row["duration_sec"] = round(float(dur), 2)
            except Exception:
                pass
            if bool(e.get("delete_all", False)):
                row["delete_all"] = True
            if bool(e.get("chunk_boundary_silence", False)):
                row["chunk_boundary_silence"] = True
        entries_out.append(row)

    return {
        "chapter_index": int(chunk.get("chapter_index", 0) or 0),
        "board": {"w": int(board_width), "h": int(board_height)},
        "steps": steps_out,
        "entries": entries_out,
    }


def _compact_visual_event_for_prompt(ev: Dict[str, Any]) -> Dict[str, Any]:
    ev_diagram_mode = int(ev.get("diagram", 0) or 0)
    cur_start = int(ev.get("range_start", 0) or 0)
    cur_end = int(ev.get("range_end", cur_start) or cur_start)
    ctx_candidates: List[Tuple[Tuple[int, int, int, int], Dict[str, Any]]] = []
    for c in (ev.get("static_plan_context") or []):
        if not isinstance(c, dict):
            continue
        ctype = _clip_str(c.get("type", ""), 16)
        crow: Dict[str, Any] = {
            "name": _clip_str(c.get("name", ""), 56),
            "type": ctype,
        }
        cpid = _clip_str(c.get("processed_id", ""), 40)
        if cpid:
            crow["processed_id"] = cpid
        c_start = int(c.get("range_start", 0) or 0)
        c_end = int(c.get("range_end", c_start) or c_start)
        crow["range_start"] = c_start
        crow["range_end"] = c_end
        diagram_mode = int(c.get("diagram", 0) or 0)
        if ctype == "image" and diagram_mode > 0:
            crow["diagram"] = diagram_mode
            if ev_diagram_mode > 0:
                other_diagram_labels = _compact_diagram_cluster_labels(
                    c.get("context_other_diagram_cluster_labels"),
                    max_items=8,
                    max_chars=40,
                )
                if other_diagram_labels:
                    crow["context_other_diagram_cluster_labels"] = other_diagram_labels

        overlap_words = max(0, min(cur_end, c_end) - max(cur_start, c_start) + 1)
        active_at_start = 1 if (c_start <= cur_start <= c_end) else 0
        same_step = 1 if _same_ci(c.get("step_key", ""), ev.get("step_key", "")) else 0
        start_delta = abs(c_start - cur_start)
        ctx_candidates.append(
            (
                (-active_at_start, -same_step, -overlap_words, start_delta),
                crow,
            )
        )

    ctx_candidates.sort(key=lambda row: row[0])
    max_context_rows = 6 if ev_diagram_mode > 0 else 4
    ctx_out = [row for _, row in ctx_candidates[:max_context_rows]]

    ev_type = _clip_str(ev.get("type", ""), 16)
    name = _clip_str(ev.get("name", ""), 56)
    image_name = _clip_str(ev.get("image_name", "") or name, 56)
    out: Dict[str, Any] = {
        "name": _clip_str(ev.get("name", ""), 56),
        "image_name": image_name,
        "type": _clip_str(ev.get("type", ""), 16),
        "range_start": int(ev.get("range_start", 0) or 0),
        "range_end": int(ev.get("range_end", 0) or 0),
        "static_plan_context": ctx_out,
    }
    pid = _clip_str(ev.get("processed_id", ""), 40)
    if pid:
        out["processed_id"] = pid
    _blank_non_priority_overlaps(out, priority="name", fields=["name", "image_name"])
    if isinstance(ev.get("print_bbox"), dict):
        out["print_bbox"] = ev.get("print_bbox")
    speech_span = _clip_str(ev.get("speech_text_in_range", ""), 900)
    if speech_span:
        out["speech_text_in_range"] = speech_span

    is_text = int(ev.get("text_tag", 0) or 0) == 1 or ev_type == "text"
    if is_text:
        out["text_tag"] = 1
        wt = _clip_str(ev.get("write_text", "") or ev.get("content", ""), 120)
        if wt:
            out["write_text"] = wt
        return out

    content = _clip_str(ev.get("content", ""), 120)
    query = _clip_str(ev.get("query", ""), 120)
    out["content"] = content
    out["query"] = query
    _blank_non_priority_overlaps(out, priority="name", fields=["name", "content", "query"])

    diagram_mode = int(ev.get("diagram", 0) or 0)
    if diagram_mode > 0:
        out["diagram"] = diagram_mode
        current_diagram_labels = _compact_diagram_cluster_labels(
            ev.get("current_diagram_cluster_labels"),
            max_items=16,
            max_chars=40,
        )
        if current_diagram_labels:
            out["current_diagram_cluster_labels"] = current_diagram_labels
        else:
            objs = _clip_str_list(
                ev.get("objects_that_comprise_image"),
                max_items=12,
                max_chars=40,
            )
            if objs:
                out["objects_that_comprise_image"] = objs
    return out


def _generate_json_objects_from_prompts(
    *,
    model,
    processor,
    device,
    prompts: List[str],
    images: Optional[List[Any]] = None,
    prompts_are_chat_text: bool = False,
    temperature: float,
    max_new_tokens: int,
    batch_size: int = 8,
    debug_label: str = "qwen_json",
    stage_io_contexts: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not prompts:
        return [], []

    capped_prompts: List[str] = []
    for p in prompts:
        one = str(p or "")
        if PLAN_PROMPT_CHAR_CAP > 0 and len(one) > PLAN_PROMPT_CHAR_CAP:
            one = one[:PLAN_PROMPT_CHAR_CAP].rstrip() + "\n\n[TRUNCATED_FOR_VRAM]"
        capped_prompts.append(one)

    chat_texts = list(capped_prompts) if prompts_are_chat_text else [_render_chat_text(processor, p) for p in capped_prompts]
    chat_images: Optional[List[Any]] = None
    if images is not None:
        chat_images = list(images[: len(chat_texts)])
        if len(chat_images) < len(chat_texts):
            chat_texts = chat_texts[: len(chat_images)]
    bs = max(1, int(batch_size or 1))
    all_raws: List[str] = []
    empty_cache_each_batch = str(os.getenv("QWEN_PLAN_EMPTY_CACHE_EACH_BATCH", "0")).strip().lower() not in ("0", "false", "no")

    def _cleanup_cuda_cache() -> None:
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

    def _cuda_mem_numbers() -> Optional[Dict[str, float]]:
        try:
            if not torch.cuda.is_available():
                return None
            free_b, total_b = torch.cuda.mem_get_info()
            return {
                "alloc_mb": float(torch.cuda.memory_allocated() / (1024 ** 2)),
                "reserved_mb": float(torch.cuda.memory_reserved() / (1024 ** 2)),
                "free_mb": float(free_b / (1024 ** 2)),
                "total_mb": float(total_b / (1024 ** 2)),
            }
        except Exception:
            return None

    def _cuda_cache_trim_needed(mem: Optional[Dict[str, float]]) -> bool:
        if not mem:
            return False
        alloc_mb = float(mem.get("alloc_mb", 0.0) or 0.0)
        reserved_mb = float(mem.get("reserved_mb", 0.0) or 0.0)
        free_mb = float(mem.get("free_mb", 0.0) or 0.0)
        total_mb = float(mem.get("total_mb", 0.0) or 0.0)
        cache_slack_mb = max(512.0, total_mb * 0.08) if total_mb > 0.0 else 512.0
        low_free_mb = max(768.0, total_mb * 0.10) if total_mb > 0.0 else 768.0
        return (reserved_mb - alloc_mb) > cache_slack_mb or free_mb < low_free_mb

    def _is_cuda_oom_error(err: BaseException) -> bool:
        msg = f"{type(err).__name__}: {err}".lower()
        return "out of memory" in msg and "cuda" in msg

    def _run_generate(texts_batch: List[str], *, use_cache_now: bool) -> List[str]:
        inputs = None
        out_ids = None
        gen_only_ids = None
        decoded = None
        try:
            with torch.inference_mode():
                t_encode0 = time.perf_counter()
                inputs = _processor_batch(processor, text=texts_batch)
                encode_ms = round((time.perf_counter() - t_encode0) * 1000.0, 2)

                input_token_max = None
                input_token_total = None
                try:
                    input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
                    attention_mask = inputs.get("attention_mask") if isinstance(inputs, dict) else None
                    if torch.is_tensor(input_ids):
                        input_token_max = int(input_ids.shape[1])
                    if torch.is_tensor(attention_mask):
                        input_token_total = int(attention_mask.sum().item())
                    elif torch.is_tensor(input_ids):
                        input_token_total = int(input_ids.numel())
                except Exception:
                    input_token_max = None
                    input_token_total = None
                print(
                    f"[qwen][DBG] {debug_label} tokenize_done size={len(texts_batch)} "
                    f"encode_ms={encode_ms} input_tokens_max={input_token_max} input_tokens_total={input_token_total} "
                    f"vram={_vram_str()}"
                )

                t_move0 = time.perf_counter()
                inputs = _move_inputs_to_device(inputs, device)
                move_ms = round((time.perf_counter() - t_move0) * 1000.0, 2)
                print(
                    f"[qwen][DBG] {debug_label} move_done size={len(texts_batch)} "
                    f"move_ms={move_ms} device={device} vram={_vram_str()}"
                )

                if bool(use_cache_now) and len(texts_batch) > 1:
                    t_gen0 = time.perf_counter()
                    input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
                    attention_mask = inputs.get("attention_mask") if isinstance(inputs, dict) else None
                    if not torch.is_tensor(input_ids) or not torch.is_tensor(attention_mask):
                        raise RuntimeError("rolling_cache_release_requires_input_ids_and_attention_mask")
                    decoded = _generate_with_rolling_cache_release(
                        model=model,
                        processor=processor,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=int(max_new_tokens),
                        temperature=float(temperature),
                        top_p=0.9,
                        do_sample=bool(PLAN_DO_SAMPLE),
                        debug_label=debug_label,
                    )
                    gen_ms = round((time.perf_counter() - t_gen0) * 1000.0, 2)
                    out_token_max = max((len(str(x or "")) for x in decoded), default=0)
                    print(
                        f"[qwen][DBG] {debug_label} rolling_generate_done size={len(texts_batch)} "
                        f"gen_ms={gen_ms} use_cache={int(use_cache_now)} out_chars_max={out_token_max} vram={_vram_str()}"
                    )
                    return list(decoded)

                gen_kwargs = {
                    "max_new_tokens": int(max_new_tokens),
                    "do_sample": bool(PLAN_DO_SAMPLE),
                    "temperature": float(temperature),
                    "top_p": 0.9,
                    "num_beams": 1,
                    "use_cache": bool(use_cache_now),
                }
                t_gen0 = time.perf_counter()
                out_ids = _safe_generate(model, inputs=inputs, gen_kwargs=gen_kwargs)
                gen_ms = round((time.perf_counter() - t_gen0) * 1000.0, 2)
                print(
                    f"[qwen][DBG] {debug_label} generate_done size={len(texts_batch)} "
                    f"gen_ms={gen_ms} use_cache={int(use_cache_now)} vram={_vram_str()}"
                )

                if "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
                    prompt_len = int(inputs["input_ids"].shape[1])
                    gen_only_ids = out_ids[:, prompt_len:]
                else:
                    gen_only_ids = out_ids

                t_decode0 = time.perf_counter()
                decoded = processor.batch_decode(gen_only_ids, skip_special_tokens=True)
                decode_ms = round((time.perf_counter() - t_decode0) * 1000.0, 2)
                out_token_max = None
                try:
                    if torch.is_tensor(gen_only_ids):
                        out_token_max = int(gen_only_ids.shape[1])
                except Exception:
                    out_token_max = None
                print(
                    f"[qwen][DBG] {debug_label} decode_done size={len(texts_batch)} "
                    f"decode_ms={decode_ms} out_tokens_max={out_token_max} vram={_vram_str()}"
                )
                return list(decoded)
        finally:
            _clear_model_runtime_cache_refs(model)
            try:
                del decoded
            except Exception:
                pass
            try:
                del gen_only_ids
            except Exception:
                pass
            try:
                del out_ids
            except Exception:
                pass
            try:
                del inputs
            except Exception:
                pass

    start = 0
    adaptive_bs = int(bs)
    while start < len(chat_texts):
        requested_len = max(1, min(int(adaptive_bs), len(chat_texts) - start))
        attempt_len = int(requested_len)
        while attempt_len >= 1:
            batch = chat_texts[start:start + attempt_len]
            batch_images = chat_images[start:start + attempt_len] if chat_images is not None else None
            if not batch:
                break

            batch_len = len(batch)
            t0 = time.perf_counter()
            batch_chars = sum(len(x) for x in batch)
            max_chars = max((len(x) for x in batch), default=0)
            use_cache_now = bool(PLAN_USE_CACHE)
            mem_before = _cuda_mem_numbers()
            print(
                f"[qwen][DBG] {debug_label} batch_start start={start} size={batch_len} "
                f"requested_bs={bs} adaptive_bs={adaptive_bs} "
                f"chars_total={batch_chars} chars_max={max_chars} use_cache={int(use_cache_now)} "
                f"vram={_vram_str()}"
            )

            batch_raws: List[str]
            batch_failed = False
            try:
                if batch_images is not None:
                    def _run_generate_with_images(texts_batch: List[str], images_batch: List[Any], *, use_cache_now: bool) -> List[str]:
                        inputs = None
                        out_ids = None
                        gen_only_ids = None
                        decoded = None
                        try:
                            with torch.inference_mode():
                                t_encode0 = time.perf_counter()
                                inputs = _processor_batch(processor, text=texts_batch, images=images_batch)
                                encode_ms = round((time.perf_counter() - t_encode0) * 1000.0, 2)
                                print(
                                    f"[qwen][DBG] {debug_label} tokenize_done size={len(texts_batch)} "
                                    f"encode_ms={encode_ms} multimodal=1 vram={_vram_str()}"
                                )
                                t_move0 = time.perf_counter()
                                inputs = _move_inputs_to_device(inputs, device)
                                move_ms = round((time.perf_counter() - t_move0) * 1000.0, 2)
                                print(
                                    f"[qwen][DBG] {debug_label} move_done size={len(texts_batch)} "
                                    f"move_ms={move_ms} device={device} multimodal=1 vram={_vram_str()}"
                                )

                                gen_kwargs = {
                                    "max_new_tokens": int(max_new_tokens),
                                    "do_sample": bool(PLAN_DO_SAMPLE),
                                    "temperature": float(temperature),
                                    "top_p": 0.9,
                                    "num_beams": 1,
                                    "use_cache": bool(use_cache_now),
                                }
                                if not bool(PLAN_DO_SAMPLE):
                                    gen_kwargs.pop("temperature", None)
                                    gen_kwargs.pop("top_p", None)
                                t_gen0 = time.perf_counter()
                                out_ids = _safe_generate(model, inputs=inputs, gen_kwargs=gen_kwargs)
                                gen_ms = round((time.perf_counter() - t_gen0) * 1000.0, 2)
                                print(
                                    f"[qwen][DBG] {debug_label} generate_done size={len(texts_batch)} "
                                    f"gen_ms={gen_ms} use_cache={int(use_cache_now)} multimodal=1 vram={_vram_str()}"
                                )
                                if "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
                                    prompt_len = int(inputs["input_ids"].shape[1])
                                    gen_only_ids = out_ids[:, prompt_len:]
                                else:
                                    gen_only_ids = out_ids
                                t_decode0 = time.perf_counter()
                                decoded = processor.batch_decode(gen_only_ids, skip_special_tokens=True)
                                decode_ms = round((time.perf_counter() - t_decode0) * 1000.0, 2)
                                print(
                                    f"[qwen][DBG] {debug_label} decode_done size={len(texts_batch)} "
                                    f"decode_ms={decode_ms} multimodal=1 vram={_vram_str()}"
                                )
                                return list(decoded)
                        finally:
                            _clear_model_runtime_cache_refs(model)
                            try:
                                del decoded
                            except Exception:
                                pass
                            try:
                                del gen_only_ids
                            except Exception:
                                pass
                            try:
                                del out_ids
                            except Exception:
                                pass
                            try:
                                del inputs
                            except Exception:
                                pass

                    batch_raws = _run_generate_with_images(batch, batch_images, use_cache_now=use_cache_now)
                else:
                    batch_raws = _run_generate(batch, use_cache_now=use_cache_now)
            except Exception as e:
                if _is_cuda_oom_error(e) and batch_len > 1:
                    next_attempt = max(1, batch_len // 2)
                    adaptive_bs = min(adaptive_bs, next_attempt)
                    print(
                        f"[qwen][DBG] {debug_label} batch_oom_retry start={start} size={batch_len} "
                        f"next_size={next_attempt} err={type(e).__name__}: {e}"
                    )
                    _cleanup_cuda_cache()
                    attempt_len = int(next_attempt)
                    continue
                print(f"[qwen][DBG] {debug_label} batch_fail start={start} size={batch_len} err={type(e).__name__}: {e}")
                _cleanup_cuda_cache()
                batch_raws = [""] * batch_len
                batch_failed = True
            finally:
                mem_pre_cleanup = _cuda_mem_numbers()
                vram_pre_cleanup = _vram_str()
                should_trim = bool(empty_cache_each_batch or _cuda_cache_trim_needed(mem_pre_cleanup))
                if should_trim:
                    _cleanup_cuda_cache()
                mem_post_cleanup = _cuda_mem_numbers()
                vram_post_cleanup = _vram_str()
                trim_reason = "manual" if empty_cache_each_batch else ("over_budget" if should_trim else "none")
                print(
                    f"[qwen][DBG] {debug_label} batch_end start={start} size={batch_len} "
                    f"elapsed_ms={round((time.perf_counter() - t0) * 1000.0, 2)} "
                    f"trim_reason={trim_reason} "
                    f"vram_pre_cleanup={vram_pre_cleanup} vram_post_cleanup={vram_post_cleanup} "
                    f"mem_before={mem_before} mem_after={mem_post_cleanup}"
                )

            if len(batch_raws) < batch_len:
                batch_raws = list(batch_raws) + [""] * (batch_len - len(batch_raws))
            all_raws.extend(batch_raws[:batch_len])
            start += batch_len
            if not batch_failed:
                adaptive_bs = max(1, min(adaptive_bs, batch_len))
            break

    objs: List[Dict[str, Any]] = []
    for idx, raw in enumerate(all_raws):
        parsed = _extract_first_json_object(raw)
        if not isinstance(parsed, dict):
            parsed = {}
        parsed["raw_text"] = raw
        objs.append(parsed)
        ctx = stage_io_contexts[idx] if isinstance(stage_io_contexts, list) and idx < len(stage_io_contexts) else None
        _write_qwen_stage_io_json(
            (ctx or {}).get("bundle") if isinstance(ctx, dict) else None,
            tokenizer_or_processor=processor,
            debug_label=str((ctx or {}).get("debug_label", "") or debug_label or "qwen_json"),
            job_key=str((ctx or {}).get("job_key", "") or f"prompt_{idx + 1}"),
            prompt_text=str((ctx or {}).get("input_prompt_text", "") or (capped_prompts[idx] if idx < len(capped_prompts) else "")),
            raw_text=str(raw or ""),
            parsed_obj=parsed,
            parsed_payload=parsed,
            stage_kind=str((ctx or {}).get("stage_kind", "") or "generic"),
            rendered_input_text=str((ctx or {}).get("rendered_input_text", "") or (chat_texts[idx] if idx < len(chat_texts) else "")),
            system_prompt=str((ctx or {}).get("system_prompt", "") or ""),
            user_prompt=str((ctx or {}).get("user_prompt", "") or ""),
            extra=((ctx or {}).get("extra") if isinstance((ctx or {}).get("extra"), dict) else {"prompt_index": int(idx), "multimodal": bool(images is not None)}),
        )
    return objs, all_raws


def _generate_json_objects_from_prompts_text_model(
    *,
    llm=None,
    model,
    tokenizer,
    device,
    prompts: List[str],
    temperature: float,
    max_new_tokens: int,
    debug_label: str = "qwen_text_json",
    thinking_enabled: bool = False,
    max_rounds: int = 4,
    batch_size: int = 8,
    stage_io_contexts: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not prompts:
        return [], []

    if llm is not None:
        capped_prompts: List[str] = []
        for p in prompts:
            one = str(p or "")
            if PLAN_PROMPT_CHAR_CAP > 0 and len(one) > PLAN_PROMPT_CHAR_CAP:
                one = one[:PLAN_PROMPT_CHAR_CAP].rstrip() + "\n\n[TRUNCATED_FOR_VRAM]"
            capped_prompts.append(one)

        rendered_prompts = [
            _apply_chat_template_text(
                tokenizer,
                [{"role": "user", "content": str(p or "")}],
                thinking_enabled=bool(thinking_enabled),
            )
            for p in capped_prompts
        ]
        print(
            f"[qwen][DBG] {debug_label} vllm_batch_mode prompts={len(rendered_prompts)} "
            f"thinking={int(bool(thinking_enabled))} batch_size={max(1, int(batch_size or 1))}"
        )
        raws = _vllm_generate_texts(
            llm=llm,
            prompts=rendered_prompts,
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
            thinking_enabled=bool(thinking_enabled),
            do_sample=bool(PLAN_DO_SAMPLE),
        )
        objs: List[Dict[str, Any]] = []
        out_raws: List[str] = []
        for raw_text in raws:
            split = _split_vllm_generated_text(raw_text, thinking_enabled=bool(thinking_enabled))
            parse_text = (
                _normalize_raw_jsonish_text(str(split.get("final_text", "") or ""))
                if thinking_enabled
                else _normalize_raw_jsonish_text(str(split.get("raw_text", "") or ""))
            )
            if thinking_enabled and not parse_text:
                parse_text = _normalize_raw_jsonish_text(str(split.get("raw_text", "") or ""))
            parsed = _extract_last_json_object(parse_text) or _extract_first_json_object(parse_text)
            if not isinstance(parsed, dict):
                parsed = {}
            parsed["raw_text"] = str(split.get("raw_text", "") or raw_text or "")
            parsed["parse_text"] = parse_text
            objs.append(parsed)
            out_raws.append(str(split.get("raw_text", "") or raw_text or ""))
            idx = len(objs) - 1
            ctx = stage_io_contexts[idx] if isinstance(stage_io_contexts, list) and idx < len(stage_io_contexts) else None
            _write_qwen_stage_io_json(
                (ctx or {}).get("bundle") if isinstance(ctx, dict) else None,
                tokenizer_or_processor=tokenizer,
                debug_label=str((ctx or {}).get("debug_label", "") or debug_label or "qwen_text_json"),
                job_key=str((ctx or {}).get("job_key", "") or f"prompt_{len(objs)}"),
                prompt_text=str((ctx or {}).get("input_prompt_text", "") or (capped_prompts[idx] if idx < len(capped_prompts) else "")),
                raw_text=str(split.get("raw_text", "") or raw_text or ""),
                parsed_obj=parsed,
                parsed_payload=parsed,
                stage_kind=str((ctx or {}).get("stage_kind", "") or "generic_text"),
                rendered_input_text=str((ctx or {}).get("rendered_input_text", "") or (rendered_prompts[idx] if idx < len(rendered_prompts) else "")),
                system_prompt=str((ctx or {}).get("system_prompt", "") or ""),
                user_prompt=str((ctx or {}).get("user_prompt", "") or ""),
                extra=((ctx or {}).get("extra") if isinstance((ctx or {}).get("extra"), dict) else {"prompt_index": int(idx), "thinking_enabled": bool(thinking_enabled), "backend": "vllm"}),
            )
        return objs, out_raws

    capped_prompts: List[str] = []
    for p in prompts:
        one = str(p or "")
        if PLAN_PROMPT_CHAR_CAP > 0 and len(one) > PLAN_PROMPT_CHAR_CAP:
            one = one[:PLAN_PROMPT_CHAR_CAP].rstrip() + "\n\n[TRUNCATED_FOR_VRAM]"
        capped_prompts.append(one)

    raws: List[str] = []
    parse_texts: List[str] = []
    round_limit = max(1, int(max_rounds or 1))
    active_device = _infer_text_model_device(model, device)
    if not bool(thinking_enabled):
        effective_bs = max(1, min(int(batch_size or 1), len(capped_prompts)))
        print(
            f"[qwen][DBG] {debug_label} transformers_batch_mode prompts={len(capped_prompts)} "
            f"batch_size={effective_bs} thinking=0 vram={_vram_str()}"
        )
        _, raws = _generate_json_objects_from_prompts(
            model=model,
            processor=tokenizer,
            device=active_device,
            prompts=capped_prompts,
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
            batch_size=int(effective_bs),
            debug_label=str(debug_label or "qwen_text_json"),
            stage_io_contexts=stage_io_contexts,
        )
        objs: List[Dict[str, Any]] = []
        for raw_text in raws:
            parse_text = str(raw_text or "").strip()
            parsed = _extract_last_json_object(parse_text) or _extract_first_json_object(parse_text)
            if not isinstance(parsed, dict):
                parsed = {}
            parsed["raw_text"] = raw_text
            parsed["parse_text"] = parse_text
            objs.append(parsed)
        return objs, raws

    for prompt_idx, prompt in enumerate(capped_prompts):
        chat_text = _render_chat_text(tokenizer, prompt, thinking_enabled=thinking_enabled)
        raw_text = ""
        parse_text = ""
        inputs = None
        generated = None
        current_ids = None
        current_attention = None
        try:
            with torch.inference_mode():
                print(
                    f"[qwen][DBG] {debug_label} prompt_start index={prompt_idx + 1}/{len(capped_prompts)} "
                    f"thinking={int(bool(thinking_enabled))} chars={len(chat_text)} vram={_vram_str()}"
                )
                inputs = _processor_batch(tokenizer, text=[chat_text])
                inputs = _move_inputs_to_device(inputs, active_device)
                current_ids = inputs.get("input_ids")
                current_attention = inputs.get("attention_mask")
                if not torch.is_tensor(current_ids):
                    raise RuntimeError("text_planner_missing_input_ids")
                if not torch.is_tensor(current_attention):
                    current_attention = torch.ones_like(current_ids, device=current_ids.device)

                prompt_len = int(current_ids.shape[1])
                for round_idx in range(round_limit):
                    print(
                        f"[qwen][DBG] {debug_label} prompt_round_start index={prompt_idx + 1}/{len(capped_prompts)} "
                        f"round={round_idx + 1}/{round_limit} prompt_len={prompt_len} vram={_vram_str()}"
                    )
                    gen_kwargs = {
                        "input_ids": current_ids,
                        "attention_mask": current_attention,
                        "max_new_tokens": int(max_new_tokens),
                        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
                        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                        "use_cache": True,
                    }
                    if thinking_enabled:
                        gen_kwargs.update(
                            {
                                "do_sample": True,
                                "temperature": max(0.6, float(temperature)),
                                "top_p": 0.95,
                                "top_k": 20,
                            }
                        )
                    else:
                        gen_kwargs["do_sample"] = bool(PLAN_DO_SAMPLE)
                        if bool(PLAN_DO_SAMPLE):
                            gen_kwargs["temperature"] = float(temperature)
                            gen_kwargs["top_p"] = 0.9

                    generated = model.generate(**gen_kwargs)
                    if not torch.is_tensor(generated) or generated.shape[-1] <= current_ids.shape[-1]:
                        print(
                            f"[qwen][DBG] {debug_label} prompt_round_empty index={prompt_idx + 1}/{len(capped_prompts)} "
                            f"round={round_idx + 1}/{round_limit} vram={_vram_str()}"
                        )
                        break

                    gen_only_ids = generated[:, prompt_len:]
                    raw_text = _decode_token_ids_text(tokenizer, gen_only_ids, skip_special_tokens=False)
                    parse_text = _strip_qwen_thinking_text(raw_text) if thinking_enabled else str(raw_text or "").strip()
                    parsed = _extract_last_json_object(parse_text) or _extract_first_json_object(parse_text)
                    print(
                        f"[qwen][DBG] {debug_label} prompt_round_done index={prompt_idx + 1}/{len(capped_prompts)} "
                        f"round={round_idx + 1}/{round_limit} raw_chars={len(str(raw_text or ''))} "
                        f"parse_chars={len(parse_text)} parsed={int(isinstance(parsed, dict))} "
                        f"has_think_close={int('</think>' in str(raw_text or ''))} vram={_vram_str()}"
                    )

                    if thinking_enabled and "</think>" not in str(raw_text or ""):
                        current_ids = generated
                        current_attention = torch.ones_like(current_ids, device=current_ids.device)
                        continue

                    if isinstance(parsed, dict):
                        break

                    current_ids = generated
                    current_attention = torch.ones_like(current_ids, device=current_ids.device)

                print(
                    f"[qwen][DBG] {debug_label} prompt_done chars={len(chat_text)} "
                    f"thinking={int(bool(thinking_enabled))} parse_chars={len(parse_text)} vram={_vram_str()}"
                )
        except Exception as e:
            print(f"[qwen][DBG] {debug_label} prompt_fail err={type(e).__name__}: {e}")
        finally:
            _clear_model_runtime_cache_refs(model)
            try:
                del generated
            except Exception:
                pass
            try:
                del current_attention
            except Exception:
                pass
            try:
                del current_ids
            except Exception:
                pass
            try:
                del inputs
            except Exception:
                pass

        raws.append(raw_text)
        parse_texts.append(parse_text)

    objs: List[Dict[str, Any]] = []
    for idx, (raw, parse_text) in enumerate(zip(raws, parse_texts)):
        parsed = _extract_last_json_object(parse_text) or _extract_first_json_object(parse_text)
        if not isinstance(parsed, dict):
            parsed = {}
        parsed["raw_text"] = raw
        parsed["parse_text"] = parse_text
        objs.append(parsed)
        ctx = stage_io_contexts[idx] if isinstance(stage_io_contexts, list) and idx < len(stage_io_contexts) else None
        _write_qwen_stage_io_json(
            (ctx or {}).get("bundle") if isinstance(ctx, dict) else None,
            tokenizer_or_processor=tokenizer,
            debug_label=str((ctx or {}).get("debug_label", "") or debug_label or "qwen_text_json"),
            job_key=str((ctx or {}).get("job_key", "") or f"prompt_{idx + 1}"),
            prompt_text=str((ctx or {}).get("input_prompt_text", "") or (capped_prompts[idx] if idx < len(capped_prompts) else "")),
            raw_text=str(raw or ""),
            parsed_obj=parsed,
            parsed_payload=parsed,
            stage_kind=str((ctx or {}).get("stage_kind", "") or "generic_text"),
            rendered_input_text=str((ctx or {}).get("rendered_input_text", "") or (_render_chat_text(tokenizer, capped_prompts[idx], thinking_enabled=thinking_enabled) if idx < len(capped_prompts) else "")),
            system_prompt=str((ctx or {}).get("system_prompt", "") or ""),
            user_prompt=str((ctx or {}).get("user_prompt", "") or ""),
            extra=((ctx or {}).get("extra") if isinstance((ctx or {}).get("extra"), dict) else {"prompt_index": int(idx), "thinking_enabled": bool(thinking_enabled), "backend": "transformers_thinking"}),
        )
    return objs, raws

def plan_whiteboard_actions_transformers(
    *,
    model,
    processor,
    device,
    event: Dict[str, Any],
    whiteboard_state: Dict[str, Any],
    base_actions: List[str],
    diagram_actions: List[str],
    temperature: float = 0.2,
    max_new_tokens: int = 700,
) -> Dict[str, Any]:
    """
    Qwen simple call:
      - event: image or silence packet (includes fine speech text + bbox + objects for diagrams)
      - whiteboard_state: current objects and bboxes on the board
      - base_actions: list of allowed base actions
      - diagram_actions: extra actions allowed only if diagram=1 (operate on objects_that_comprise_image)

    Returns dict:
      {
        "actions": [ ... ],   # list of dicts OR strings
        "raw_text": "...",
        "notes": "...",
      }
    """

    # Build instruction prompt (NO loops/state management here)
    kind = str(event.get("kind", "") or "")
    is_diagram = int(event.get("diagram", 0) or 0) > 0

    # NEW: text tag (0/1). When 1, this "image event" represents board text.
    text_tag = int(event.get("text", 0) or 0)
    write_text = str(event.get("write_text", "") or "")

    # Strong constraints: produce JSON only, avoid code fences.
    system_rules = (
        "You are planning actions with images to build a synced animation under some narration.\n"
        "You have to fill as much different actions for a rich result, using your base actions.\n"
        "Images will come sometimes come with a DIAGRAM tag.\n"
        "For these use the special diagram actions with each of their objects, pushing for text.\n\n"

        "You are provided with a whiteboard state with all present objects.\n"
        "Draw in empty spaces not already taken up by an object and manage object layout based on their relations.\n"
        "If there's no space either shift the board to make space (center shifts, you can draw at further coords)"
        "or delete an irrelevant object.\n"
        "Try and find an empty space first, if you erase plan your actions - > first erase then draw\n"

        "After you finish with an image dont erase it by default if it wasnt very \"temporary\""
        "During SILENCE: prefer erasing many objects and/or shifting the board to reset space.\n"

        "Try to keep something happening most of the time: output many actions when appropriate.\n"

        # --- ONLY ADDITION: TEXT TAG RULES ---
        "TEXT TAG RULES:\n"
        "- If CURRENT EVENT includes text_tag=1, this is NOT an image draw request.\n"
        "- For text_tag=1 you should primarily output a 'write' action to place the provided write_text.\n"
        "- Avoid complex layout/multi-step work for text_tag=1; only erase/shift if needed to make space.\n"
        f"- Board text is large: think roughly {WHITEBOARD_TEXT_CHAR_WIDTH_PX}px of width per character and about {WHITEBOARD_TEXT_REFERENCE_HEIGHT_PX}px of height at write scale 1.0, with about {WHITEBOARD_TEXT_LETTER_GAP_PX}px between letters. If you must clear space, erase first, then write.\n"
        "- Do NOT output diagram-only actions for text_tag=1.\n\n"
        # ------------------------------------

        "You MUST output ONLY a single JSON object, no markdown, no code fences.\n"
        "Do not invent new action types.\n"
        "All actions MUST be from the allowed action sets.\n"
        "Do not invent new action types.\n"
        f"Whiteboard size is {WHITEBOARD_BOARD_WIDTH_PX} x {WHITEBOARD_BOARD_HEIGHT_PX} px.\n"
        f"Coordinates are in whiteboard coords with center 0,0, so the visible board is roughly x/y from -{WHITEBOARD_BOARD_WIDTH_PX // 2} to {WHITEBOARD_BOARD_WIDTH_PX // 2}.\n"
    )

    allowed = {
        "base_actions": base_actions,
        "diagram_actions": diagram_actions if is_diagram else [],
    }

    # Event payload
    event_desc = {
        "kind": kind,
        "duration_sec": float(event.get("duration_sec", 0.0) or 0.0),
        "chapter_index": int(event.get("chapter_index", 0) or 0),
    }

    if kind == "image":
        event_desc.update(
            {
                "image_name": event.get("image_name"),
                "image_text": event.get("image_text", ""),
                "bbox_px": event.get("bbox_px", {}),
                "diagram": 1 if is_diagram else 0,
                "objects_that_comprise_image": event.get("objects_that_comprise_image", []) if is_diagram else [],
                # NEW: propagate text tag & the text to write (if any)
                "text_tag": int(text_tag),
                "write_text": write_text if int(text_tag) == 1 else "",
            }
        )
    else:
        event_desc.update(
            {
                "context_before": event.get("context_before", ""),
                "context_after": event.get("context_after", ""),
            }
        )

    # Required output schema
    schema = {
        "actions": [
            # base:
            # {"type":"draw","target":"<image name>","x":123,"y":456}
            # {"type":"erase","target":"<image name or text object name>"}
            # {"type":"link","a":"<image name>","b":"<image name>"}
            # {"type":"write","text":"...","x":100,"y":200,"scale":1.0}
            # {"type":"move","target":"<image name>","x":400,"y":200}
            #
            # diagram-only (do not affect board geometry, still output them):
            # {"type":"highlight","image":"<image name>","object":"<object name>","time":2.0}
            # {"type":"zoom","image":"<image name>","object":"<object name>","time":3.0}
            # {"type":"refine_detail","image":"<image name>","object":"<object name>"}
        ],
        "notes": "short optional notes",
    }

    prompt = (
        system_rules
        + "\n\nALLOWED ACTIONS:\n"
        + json.dumps(allowed, ensure_ascii=False, indent=2)
        + "\n\nCURRENT WHITEBOARD STATE:\n"
        + json.dumps(whiteboard_state, ensure_ascii=False, indent=2)
        + "\n\nCURRENT EVENT:\n"
        + json.dumps(event_desc, ensure_ascii=False, indent=2)
        + "\n\nOUTPUT JSON SCHEMA (follow this shape):\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + "\n\nNow output the JSON object only."
    )

    # Try to use chat template if available
    chat_text = None
    try:
        tok = getattr(processor, "tokenizer", None)
        if tok is not None and hasattr(tok, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            chat_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        chat_text = None

    if chat_text is None:
        chat_text = prompt

    with torch.no_grad():
        inputs = _processor_batch(
            processor,
            text=[chat_text],
        )
        inputs = _move_inputs_to_device(inputs, device)

        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": True,
            "temperature": float(temperature),
            "top_p": 0.9,
        }

        out_ids = _safe_generate(model, inputs=inputs, gen_kwargs=gen_kwargs)
        if "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
            prompt_len = int(inputs["input_ids"].shape[1])
            gen_only_ids = out_ids[:, prompt_len:]
        else:
            gen_only_ids = out_ids
        try:
            raw = processor.batch_decode(gen_only_ids, skip_special_tokens=True)[0]
        except Exception:
            # some processors expose tokenizer.decode
            raw = processor.tokenizer.decode(gen_only_ids[0], skip_special_tokens=True)

    parsed = _extract_first_json_object(raw)
    if not isinstance(parsed, dict):
        return {
            "actions": [],
            "raw_text": raw,
            "notes": "failed_to_parse_json",
        }

    # Ensure keys exist
    if "actions" not in parsed or not isinstance(parsed["actions"], list):
        parsed["actions"] = []
    if "notes" not in parsed:
        parsed["notes"] = ""

    parsed["raw_text"] = raw
    return parsed


def _repair_space_plan_chunk(
    *,
    chunk: Dict[str, Any],
    planned_entries: Optional[List[Dict[str, Any]]],
    board_w: int,
    board_h: int,
) -> Dict[str, Any]:
    entries_in = chunk.get("entries") or []
    if not isinstance(entries_in, list):
        entries_in = []
    plan_rows = planned_entries if isinstance(planned_entries, list) else []
    plan_by_idx: Dict[int, Dict[str, Any]] = {}
    for row in plan_rows:
        if not isinstance(row, dict):
            continue
        try:
            k = int(row.get("entry_index", -1))
        except Exception:
            k = -1
        if k >= 0:
            plan_by_idx[k] = row

    merged: List[Dict[str, Any]] = []
    for e in entries_in:
        if not isinstance(e, dict):
            continue
        out = dict(e)
        idx = int(out.get("entry_index", 0) or 0)
        p = plan_by_idx.get(idx) or {}
        t = str(out.get("type", "") or "").lower()
        if t in ("image", "text"):
            bbox_px = out.get("bbox_px") or {}
            if t == "text":
                bw, bh = _estimate_text_bbox_for_whiteboard(
                    out,
                    board_w=int(board_w),
                    board_h=int(board_h),
                )
            else:
                bw = int(bbox_px.get("w", 400) or 400)
                bh = int(bbox_px.get("h", 300) or 300)
            # "very big bbox" policy: image/text occupancy can approach quadrant scale.
            if t == "text":
                target_w = max(bw + 80, board_w // 8)
                target_h = max(bh + 60, board_h // 10)
            else:
                target_w = max(bw + 140, board_w // 5)
                target_h = max(bh + 120, board_h // 5)
            target_w = min(target_w, board_w)
            target_h = min(target_h, board_h)
            pb = p.get("print_bbox")
            out["print_bbox"] = _normalize_print_bbox(
                pb if isinstance(pb, dict) else None,
                default_w=int(target_w),
                default_h=int(target_h),
                board_w=board_w,
                board_h=board_h,
            )
            if t == "text":
                tc = p.get("text_coord") if isinstance(p, dict) else None
                out["text_coord"] = _normalize_text_coord(
                    tc if isinstance(tc, dict) else None,
                    print_bbox=out["print_bbox"],
                    text_w=bw,
                    text_h=bh,
                    board_w=board_w,
                    board_h=board_h,
                )
        elif t == "silence":
            default_delete = bool(out.get("chunk_boundary_silence", False))
            planner_delete = p.get("delete_all") if isinstance(p, dict) and "delete_all" in p else None
            # Chapter-boundary cleanup silences are deterministic wipe points and
            # should not be disabled by the model. Later diagram keepalive logic
            # can still preserve marked exceptions by filtering delete targets.
            out["delete_all"] = bool(default_delete or planner_delete) if planner_delete is not None else default_delete
        merged.append(out)

    # deterministic no-overlap repair for print boxes
    placed: List[Dict[str, int]] = []
    for row in sorted(
        [x for x in merged if str(x.get("type", "") or "") in ("image", "text")],
        key=lambda x: (int(x.get("range_start", 0) or 0), int(x.get("entry_index", 0) or 0)),
    ):
        pb = row.get("print_bbox") if isinstance(row.get("print_bbox"), dict) else None
        bbox_px = row.get("bbox_px") or {}
        dw = int(bbox_px.get("w", 400) or 400)
        dh = int(bbox_px.get("h", 300) or 300)
        desired = _normalize_print_bbox(
            pb,
            default_w=max(dw + 120, board_w // 6),
            default_h=max(dh + 100, board_h // 6),
            board_w=board_w,
            board_h=board_h,
        )
        fixed = _find_non_overlapping_box(desired, placed=placed, board_w=board_w, board_h=board_h, step=40)
        row["print_bbox"] = fixed
        if str(row.get("type", "") or "").strip().lower() == "text":
            tw, th = _estimate_text_bbox_for_whiteboard(
                row,
                board_w=int(board_w),
                board_h=int(board_h),
            )
            row["text_coord"] = _normalize_text_coord(
                row.get("text_coord") if isinstance(row.get("text_coord"), dict) else None,
                print_bbox=fixed,
                text_w=tw,
                text_h=th,
                board_w=board_w,
                board_h=board_h,
            )
        placed.append(fixed)

    merged.sort(key=lambda x: (int(x.get("range_start", 0) or 0), 0 if str(x.get("type", "") or "") == "silence" else 1))
    for i, row in enumerate(merged):
        row["entry_index"] = int(i)

    return {
        "chapter_index": int(chunk.get("chapter_index", 0) or 0),
        "board": {"w": int(board_w), "h": int(board_h)},
        "step_order": list(chunk.get("step_order") or []),
        "steps": list(chunk.get("steps") or []),
        "entries": merged,
    }


def plan_space_timeline_batch_transformers(
    *,
    model,
    processor,
    device,
    chunk_maps: List[Dict[str, Any]],
    board_width: int = WHITEBOARD_BOARD_WIDTH_PX,
    board_height: int = WHITEBOARD_BOARD_HEIGHT_PX,
    batch_size: int = 8,
    temperature: float = 0.15,
    max_new_tokens: int = 900,
) -> Dict[str, Any]:
    """
    Plans static print space per chunk:
      - assigns print_bbox for image/text entries
      - marks silence entries with delete_all true/false
    Always validates + repairs output to ensure valid non-overlapping bboxes.
    """
    chunks = [c for c in (chunk_maps or []) if isinstance(c, dict)]
    if not chunks:
        return {"chunks": []}

    prompts: List[str] = []
    for ch in chunks:
        payload = _compact_space_chunk_for_prompt(
            ch,
            board_width=int(board_width),
            board_height=int(board_height),
        )
        prompt = (
            "You are managing where images will be printed on a whiteboard\n"
            "Goal: assign each image/text a non-overlapping print_bbox and choose which silence rows should be cleanup points\n"
            "Entries are chronologically synced with range_start/range_end, range_words, and step_key\n"
            "For each image/text entry you MUST output print_bbox\n"
            "For each silence you MUST output delete_all true/false\n"
            "Set delete_all=true only for pauses where board cleanup is useful and appropriate before the next cluster of visuals\n"
            "Keep nearby chronology conceptually close where possible\n"
            "Use large boxes, stay inside board bounds.\n"
            "Try and fill up the board as much a possible before deletion (cram optimized close bboxes)\n"
            "For images with long ranges you might have to use manual delete actions around them to make space, while they are still active\n"
            "With that, and as a general rule, only delete an image after it's active range is ended\n"
            "Diagram images are generally more substantial - so know to give them a solid free space in a good space"
            "Board dimentions are -1000 to 1000 on x and -700 and 700 - 0,0 is the center, go from left to right"
            "Dont place text on the same y level -> space them apart by at least 50."
            "Output JSON only.\n\n"
            "ADDED RULES FOR TEXT ENTRIES (append-only):\n"
            "- For each entry where type='text', also output text_coord with exact write coordinates.\n"
            "- text_coord must be: {\"x\": <int>, \"y\": <int>}.\n"
            "- Keep text_coord inside its print_bbox.\n"
            f"- Text on this board is large: assume about {WHITEBOARD_TEXT_CHAR_WIDTH_PX}px of width per character and about {WHITEBOARD_TEXT_DEFAULT_HEIGHT_PX}px of height at the usual whiteboard text size.\n"
            "- Keep providing print_bbox for text entries as usual.\n\n"
            "Input chunk:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n\nOutput schema:\n"
            + json.dumps(
                {
                    "entries": [
                        {"entry_index": 0, "type": "image", "print_bbox": {"x": 0, "y": 0, "w": -800, "h": 800}},
                        {"entry_index": 1, "type": "text", "print_bbox": {"x": 900, "y": 100, "w": 500, "h": -300}, "text_coord": {"x": 920, "y": 140}},
                        {"entry_index": 2, "type": "silence", "delete_all": False},
                    ],
                    "notes": "short optional note",
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        prompts.append(prompt)

    parsed_objs, _ = _generate_json_objects_from_prompts(
        model=model,
        processor=processor,
        device=device,
        prompts=prompts,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        batch_size=int(batch_size),
        debug_label="plan_space",
        stage_io_contexts=[
            _build_qwen_stage_io_context(
                bundle=None,
                debug_label="plan_space",
                job_key=str(ch.get("job_key", "") or f"chunk_{idx}"),
                stage_kind="text",
                rendered_input_text=str(prompts[idx] if idx < len(prompts) else ""),
                input_prompt_text=str(prompts[idx] if idx < len(prompts) else ""),
                extra={
                    "temperature": float(temperature),
                    "max_new_tokens": int(max_new_tokens),
                    "job_input": ch,
                },
            )
            for idx, ch in enumerate(chunks)
        ],
    )

    out_chunks: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        obj = parsed_objs[i] if i < len(parsed_objs) else {}
        plan_entries = obj.get("entries") if isinstance(obj, dict) else None
        repaired = _repair_space_plan_chunk(
            chunk=ch,
            planned_entries=plan_entries if isinstance(plan_entries, list) else None,
            board_w=int(board_width),
            board_h=int(board_height),
        )
        out_chunks.append(repaired)

    return {"chunks": out_chunks}


def plan_space_timeline_batch_text_model(
    *,
    bundle: Dict[str, Any],
    chunk_maps: List[Dict[str, Any]],
    board_width: int = WHITEBOARD_BOARD_WIDTH_PX,
    board_height: int = WHITEBOARD_BOARD_HEIGHT_PX,
    temperature: float = 0.15,
    max_new_tokens: int = 900,
    thinking_enabled: bool = False,
) -> Dict[str, Any]:
    chunks = [c for c in (chunk_maps or []) if isinstance(c, dict)]
    if not chunks:
        return {"chunks": []}

    prompts: List[str] = []
    for ch in chunks:
        payload = _compact_space_chunk_for_prompt(
            ch,
            board_width=int(board_width),
            board_height=int(board_height),
        )
        prompt = (
            "You are managing where images will be printed on a whiteboard\n"
            "Goal: assign each image/text a non-overlapping print_bbox and choose which silence rows should be cleanup points\n"
            "Entries are chronologically synced with range_start/range_end, range_words, and step_key\n"
            "For each image/text entry you MUST output print_bbox\n"
            "For each silence you MUST output delete_all true/false\n"
            "Set delete_all=true only for pauses where board cleanup is useful and appropriate before the next cluster of visuals\n"
            "Keep nearby chronology conceptually close where possible\n"
            "Use large boxes, stay inside board bounds.\n"
            "Try and fill up the board as much a possible before deletion (cram optimized close bboxes)\n"
            "For images with long ranges you might have to use manual delete actions around them to make space, while they are still active\n"
            "With that, and as a general rule, only delete an image after it's active range is ended\n"
            "Diagram images are generally more substantial - so know to give them a solid free space in a good space"
            "Board dimentions are -1000 to 1000 on x and -700 and 700 - 0,0 is the center, go from left to right"
            "Dont place text on the same y level -> space them apart by at least 50."
            "Output JSON only.\n\n"
            "ADDED RULES FOR TEXT ENTRIES (append-only):\n"
            "- For each entry where type='text', also output text_coord with exact write coordinates.\n"
            "- text_coord must be: {\"x\": <int>, \"y\": <int>}.\n"
            "- Keep text_coord inside its print_bbox.\n"
            f"- Text on this board is large: assume about {WHITEBOARD_TEXT_CHAR_WIDTH_PX}px of width per character and about {WHITEBOARD_TEXT_DEFAULT_HEIGHT_PX}px of height at the usual whiteboard text size.\n"
            "- Keep providing print_bbox for text entries as usual.\n\n"
            "Input chunk:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n\nOutput schema:\n"
            + json.dumps(
                {
                    "entries": [
                        {"entry_index": 0, "type": "image", "print_bbox": {"x": 0, "y": 0, "w": -800, "h": 800}},
                        {"entry_index": 1, "type": "text", "print_bbox": {"x": 900, "y": 100, "w": 500, "h": -300}, "text_coord": {"x": 920, "y": 140}},
                        {"entry_index": 2, "type": "silence", "delete_all": False},
                    ],
                    "notes": "short optional note",
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        prompts.append(prompt)

    parsed_objs, _ = _generate_json_objects_from_prompts_text_model(
        llm=bundle.get("llm"),
        model=bundle.get("model"),
        tokenizer=bundle.get("tokenizer"),
        device=bundle.get("device"),
        prompts=prompts,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        debug_label="plan_space_text",
        thinking_enabled=bool(thinking_enabled),
        batch_size=max(1, len(chunks)),
        stage_io_contexts=[
            _build_qwen_stage_io_context(
                bundle=bundle,
                debug_label="plan_space_text",
                job_key=str(ch.get("job_key", "") or f"chunk_{idx}"),
                stage_kind="text",
                rendered_input_text=str(prompts[idx] if idx < len(prompts) else ""),
                input_prompt_text=str(prompts[idx] if idx < len(prompts) else ""),
                extra={
                    "thinking_enabled": bool(thinking_enabled),
                    "temperature": float(temperature),
                    "max_new_tokens": int(max_new_tokens),
                    "job_input": ch,
                },
            )
            for idx, ch in enumerate(chunks)
        ],
    )

    out_chunks: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        obj = parsed_objs[i] if i < len(parsed_objs) else {}
        plan_entries = obj.get("entries") if isinstance(obj, dict) else None
        repaired = _repair_space_plan_chunk(
            chunk=ch,
            planned_entries=plan_entries if isinstance(plan_entries, list) else None,
            board_w=int(board_width),
            board_h=int(board_height),
        )
        out_chunks.append(repaired)
    return {"chunks": out_chunks}


def _default_visual_actions_for_event(ev: Dict[str, Any]) -> List[Dict[str, Any]]:
    s = int(ev.get("range_start", 0) or 0)
    e = int(ev.get("range_end", s) or s)
    if e < s:
        e = s
    span = max(0, e - s)
    pb = ev.get("print_bbox") or {"x": 0, "y": 0, "w": 400, "h": 300}
    x = int(pb.get("x", 0) or 0)
    y = int(pb.get("y", 0) or 0)

    acts: List[Dict[str, Any]] = []
    if int(ev.get("text_tag", 0) or 0) == 1:
        acts.append(
            {
                "type": "write_text",
                "target": str(ev.get("name", "") or ""),
                "text": str(ev.get("write_text", "") or ev.get("content", "")),
                "x": x,
                "y": y,
                "sync_local": {"start_word_offset": 0, "end_word_offset": span},
            }
        )
        return acts

    if int(ev.get("diagram", 0) or 0) > 0:
        diagram_acts = _default_diagram_actions_for_event(
            ev,
            span=span,
            draw_target=str(ev.get("name", "") or ""),
        )
        if diagram_acts:
            return diagram_acts

    acts.append(
        {
            "type": "draw_image",
            "target": str(ev.get("name", "") or ""),
            "x": x,
            "y": y,
            "sync_local": {"start_word_offset": 0, "end_word_offset": min(span, 2)},
        }
    )
    if int(ev.get("diagram", 0) or 0) > 0:
        objs = ev.get("objects_that_comprise_image") or []
        first_obj = str(objs[0] if isinstance(objs, list) and objs else "").strip()
        acts.append(
            {
                "type": "highlight_cluster",
                "cluster_name": first_obj,
                "sync_local": {"start_word_offset": min(span, 1), "end_word_offset": min(span, max(1, span // 2))},
            }
        )
    return acts


_VISUAL_ACTION_ALIASES = {
    "draw_image": "draw_image",
    "draw": "draw_image",
    "write_text": "write_text",
    "write": "write_text",
    "annotate": "write_text",
    "move_inside_bbox": "move_inside_bbox",
    "move_inside_print_bbox": "move_inside_bbox",
    "move": "move_inside_bbox",
    "link_to_image": "link_to_image",
    "link_to_previous": "link_to_image",
    "link": "link_to_image",
    "delete_self": "delete_self",
    "delete_by_id": "delete_by_id",
    "delete_image_by_id": "delete_by_id",
    "delete_by_image_id": "delete_by_id",
    "delete_by_processed_id": "delete_by_id",
    "delete_processed_id": "delete_by_id",
    "highlight_cluster": "highlight_cluster",
    "unhighlight_cluster": "unhighlight_cluster",
    "zoom_cluster": "zoom_cluster",
    "unzoom_cluster": "unzoom_cluster",
    "write_label": "write_label",
    "connect_cluster_to_cluster": "connect_cluster_to_cluster",
    "connect_clusters": "connect_cluster_to_cluster",
}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _normalize_diagram_cluster_ref(value: Any, *, default_diagram_name: str = "") -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if ":" not in raw:
        diagram_name = str(default_diagram_name or "").strip()
        return f"{diagram_name} : {raw}" if diagram_name else raw
    left, right = raw.split(":", 1)
    diagram_name = str(left or "").strip() or str(default_diagram_name or "").strip()
    cluster_name = str(right or "").strip()
    if not diagram_name or not cluster_name:
        return ""
    return f"{diagram_name} : {cluster_name}"


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return int(default)


def _normalize_sync_local(sync: Any, *, span: int) -> Dict[str, int]:
    if not isinstance(sync, dict):
        sync = {}
    s = _safe_int(sync.get("start_word_offset", sync.get("start", 0)), 0)
    e = _safe_int(sync.get("end_word_offset", sync.get("end", s)), s)
    s = _clamp_i(s, 0, max(0, span))
    e = _clamp_i(e, s, max(0, span))
    return {
        "start_word_offset": int(s),
        "end_word_offset": int(e),
    }


def _extract_xy(raw: Dict[str, Any], *, default_x: int, default_y: int, x_key: str = "x", y_key: str = "y") -> Tuple[int, int]:
    loc = raw.get("location")
    xv = raw.get(x_key, None)
    yv = raw.get(y_key, None)
    if isinstance(loc, dict):
        if xv is None:
            xv = loc.get("x")
        if yv is None:
            yv = loc.get("y")
    x = _safe_int(xv, default_x)
    y = _safe_int(yv, default_y)
    return x, y


def _clean_visual_action_for_event(ev: Dict[str, Any], a: Dict[str, Any], *, action_index: int) -> Optional[Dict[str, Any]]:
    raw_t = str(a.get("type", "") or "").strip().lower()
    if not raw_t:
        return None
    t = _VISUAL_ACTION_ALIASES.get(raw_t)
    if not t:
        return None

    is_diagram = int(ev.get("diagram", 0) or 0) > 0
    diagram_types = (
        "highlight_cluster",
        "unhighlight_cluster",
        "zoom_cluster",
        "unzoom_cluster",
        "write_label",
        "connect_cluster_to_cluster",
    )
    if t in diagram_types and not is_diagram:
        return None

    s = int(ev.get("range_start", 0) or 0)
    e = int(ev.get("range_end", s) or s)
    if e < s:
        e = s
    span = max(0, e - s)

    pb = ev.get("print_bbox") if isinstance(ev.get("print_bbox"), dict) else {"x": 0, "y": 0, "w": 400, "h": 300}
    default_x = _safe_int(pb.get("x", 0), 0)
    default_y = _safe_int(pb.get("y", 0), 0)
    default_name = str(ev.get("name", "") or "").strip()

    out: Dict[str, Any] = {"type": t}

    if t == "draw_image":
        x, y = _extract_xy(a, default_x=default_x, default_y=default_y, x_key="x", y_key="y")
        out["target"] = str(a.get("target", "") or default_name).strip()
        out["x"] = int(x)
        out["y"] = int(y)
    elif t == "write_text":
        x, y = _extract_xy(a, default_x=default_x, default_y=default_y, x_key="x", y_key="y")
        text_fallback = str(ev.get("write_text", "") or ev.get("content", "")).strip()
        out["text"] = str(a.get("text", "") or text_fallback)
        out["x"] = int(x)
        out["y"] = int(y)
        out["scale"] = float(_safe_float(a.get("scale", 1.0), 1.0))
        tgt = str(a.get("target", "") or "").strip()
        if not tgt:
            tgt = default_name if int(ev.get("text_tag", 0) or 0) == 1 else f"{default_name}__text_{action_index + 1}"
        out["target"] = tgt
    elif t == "move_inside_bbox":
        nx, ny = _extract_xy(a, default_x=default_x, default_y=default_y, x_key="new_x", y_key="new_y")
        if "new_x" not in a and "new_y" not in a:
            nx, ny = _extract_xy(a, default_x=default_x, default_y=default_y, x_key="x", y_key="y")
        out["target"] = str(a.get("target", "") or default_name).strip()
        out["new_x"] = int(nx)
        out["new_y"] = int(ny)
    elif t == "link_to_image":
        other = (
            str(a.get("image_name", "") or a.get("target_image", "") or a.get("link_to_image", "") or a.get("to", "") or "")
            .strip()
        )
        if not other:
            return None
        out["target"] = default_name
        out["image_name"] = other
    elif t == "delete_self":
        pass
    elif t == "delete_by_id":
        image_id = str(
            a.get("image_id", "")
            or a.get("processed_id", "")
            or a.get("target_id", "")
            or a.get("id", "")
            or a.get("target", "")
            or ev.get("processed_id", "")
            or ""
        ).strip()
        if not image_id:
            return None
        out["image_id"] = image_id
    elif t == "connect_cluster_to_cluster":
        from_default = str(a.get("from_diagram_name", "") or a.get("diagram_name", "") or default_name).strip()
        to_default = str(a.get("to_diagram_name", "") or a.get("other_diagram_name", "") or default_name).strip()
        from_cluster = _normalize_diagram_cluster_ref(
            a.get("from_cluster", "") or a.get("source_cluster", "") or a.get("cluster_name", "") or "",
            default_diagram_name=from_default,
        )
        to_cluster = _normalize_diagram_cluster_ref(
            a.get("to_cluster", "")
            or a.get("target_cluster", "")
            or a.get("other_cluster", "")
            or a.get("other_cluster_name", "")
            or "",
            default_diagram_name=to_default,
        )
        if not from_cluster or not to_cluster:
            return None
        out["from_cluster"] = from_cluster
        out["to_cluster"] = to_cluster
        out["duration_sec"] = max(
            0.1,
            float(_safe_float(a.get("duration_sec", a.get("duration_in_sec", 2.0)), 2.0)),
        )
    elif t in diagram_types:
        objs = ev.get("objects_that_comprise_image") if isinstance(ev.get("objects_that_comprise_image"), list) else []
        first_obj = str(objs[0] if objs else "").strip()
        raw_cluster_name = str(
            a.get("target", "")
            or a.get("cluster_name", "")
            or a.get("cluster", "")
            or a.get("object", "")
            or a.get("name", "")
            or first_obj
        ).strip()
        cluster_name = _normalize_diagram_cluster_ref(
            raw_cluster_name,
            default_diagram_name=default_name,
        )
        if not cluster_name:
            return None
        out["cluster_name"] = cluster_name
        if t == "write_label":
            out["text"] = str(a.get("text", "") or raw_cluster_name)

    out["sync_local"] = _normalize_sync_local(a.get("sync_local"), span=span)
    return out


def plan_visual_actions_batch_transformers(
    *,
    model,
    processor,
    device,
    events: List[Dict[str, Any]],
    batch_size: int = 8,
    temperature: float = 0.2,
    max_new_tokens: int = 1400,
) -> Dict[str, Any]:
    """
    Plans actions for image/text/diagram events.
    Uses static timeline context for visuals that overlap the current event's active range.
    """
    rows = [e for e in (events or []) if isinstance(e, dict)]
    if not rows:
        return {"items": []}

    prompts: List[str] = []
    for ev in rows:
        prompts.append(build_visual_action_prompt(ev))

    parsed_objs, _ = _generate_json_objects_from_prompts(
        model=model,
        processor=processor,
        device=device,
        prompts=prompts,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        batch_size=int(batch_size),
        debug_label="plan_visual",
        stage_io_contexts=[
            _build_qwen_stage_io_context(
                bundle=None,
                debug_label="plan_visual",
                job_key=str(ev.get("job_key", "") or f"event_{idx}"),
                stage_kind="text",
                rendered_input_text=str(prompts[idx] if idx < len(prompts) else ""),
                input_prompt_text=str(prompts[idx] if idx < len(prompts) else ""),
                extra={
                    "temperature": float(temperature),
                    "max_new_tokens": int(max_new_tokens),
                    "job_input": ev,
                },
            )
            for idx, ev in enumerate(rows)
        ],
    )

    out_items: List[Dict[str, Any]] = []
    for i, ev in enumerate(rows):
        obj = parsed_objs[i] if i < len(parsed_objs) else {}
        acts = _extract_action_list_from_model_obj(obj)

        cleaned: List[Dict[str, Any]] = []
        for ai, a in enumerate(acts):
            if not isinstance(a, dict):
                continue
            one = _clean_visual_action_for_event(ev, a, action_index=ai)
            if one is None:
                continue
            cleaned.append(one)

        if not cleaned:
            cleaned = _default_visual_actions_for_event(ev)

        name = _clip_str(ev.get("name", ""), 56)
        image_name = _clip_str(ev.get("image_name", "") or name, 56)
        item: Dict[str, Any] = {
            "name": name,
            "image_name": image_name,
            "actions": cleaned,
        }
        _blank_non_priority_overlaps(item, priority="name", fields=["name", "image_name"])
        notes = str(obj.get("notes", "") or "").strip()
        if notes:
            item["notes"] = notes
        out_items.append(
            item
        )

    return {"items": out_items}


def plan_visual_actions_batch_text_model(
    *,
    bundle: Dict[str, Any],
    events: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_new_tokens: int = 1400,
) -> Dict[str, Any]:
    rows = [e for e in (events or []) if isinstance(e, dict)]
    if not rows:
        return {"items": []}

    prompts: List[str] = [build_visual_action_prompt(ev) for ev in rows]
    parsed_objs, _ = _generate_json_objects_from_prompts_text_model(
        llm=bundle.get("llm"),
        model=bundle.get("model"),
        tokenizer=bundle.get("tokenizer"),
        device=bundle.get("device"),
        prompts=prompts,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        debug_label="plan_visual_text",
        thinking_enabled=False,
        batch_size=max(1, len(rows)),
        stage_io_contexts=[
            _build_qwen_stage_io_context(
                bundle=bundle,
                debug_label="plan_visual_text",
                job_key=str(ev.get("job_key", "") or f"event_{idx}"),
                stage_kind="text",
                rendered_input_text=str(prompts[idx] if idx < len(prompts) else ""),
                input_prompt_text=str(prompts[idx] if idx < len(prompts) else ""),
                extra={
                    "temperature": float(temperature),
                    "max_new_tokens": int(max_new_tokens),
                    "job_input": ev,
                },
            )
            for idx, ev in enumerate(rows)
        ],
    )

    out_items: List[Dict[str, Any]] = []
    for i, ev in enumerate(rows):
        obj = parsed_objs[i] if i < len(parsed_objs) else {}
        acts = _extract_action_list_from_model_obj(obj)

        cleaned: List[Dict[str, Any]] = []
        for ai, a in enumerate(acts):
            if not isinstance(a, dict):
                continue
            one = _clean_visual_action_for_event(ev, a, action_index=ai)
            if one is None:
                continue
            cleaned.append(one)

        if not cleaned:
            cleaned = _default_visual_actions_for_event(ev)

        name = _clip_str(ev.get("name", ""), 56)
        image_name = _clip_str(ev.get("image_name", "") or name, 56)
        item: Dict[str, Any] = {
            "name": name,
            "image_name": image_name,
            "actions": cleaned,
        }
        _blank_non_priority_overlaps(item, priority="name", fields=["name", "image_name"])
        notes = str(obj.get("notes", "") or "").strip()
        if notes:
            item["notes"] = notes
        out_items.append(item)

    return {"items": out_items}


# ============================
# MAIN - kept for testing
# ============================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[env] torch: {torch.__version__}")
    print(f"[env] cuda_available: {torch.cuda.is_available()} count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [env] cuda {i} : {torch.cuda.get_device_name(i)}")

    print("[load] model...")
    model, processor, device, used_quant = load_model_and_processor()
    print(f"[load] used_quant={used_quant} QUANT_MODE={QUANT_MODE} _HAS_BNB={_HAS_BNB} INT8_CPU_OFFLOAD={INT8_CPU_OFFLOAD}")
    print(f"[cfg] device={device} dtype={DTYPE} PROC_LONGEST_EDGE={PROC_LONGEST_EDGE} panel_edge={PROC_LONGEST_EDGE//2}")
    print(f"[cfg] BATCH_SIZE={BATCH_SIZE} max_new_tokens={MAX_NEW_TOKENS} suggestion_limit={SUGGESTION_LIMIT}")
    print(f"[vram] after load: {_vram_str()}")

    idxs = find_indices_from_cluster_maps()
    if not idxs:
        print(f"[error] no ClusterMaps/processed_<idx>/clusters.json found in {CLUSTER_MAP_DIR}")
        return

    img_index, json_index = ensure_indexes()

    gen_kwargs = dict(
        max_new_tokens=int(MAX_NEW_TOKENS),
        do_sample=bool(DO_SAMPLE),
        num_beams=1,
        use_cache=True,
    )

    PANEL_EDGE = max(64, int(PROC_LONGEST_EDGE) // 2)

    for idx in idxs:
        t0 = time.time()

        full_img_path_s = img_index.get(str(idx))
        if not full_img_path_s:
            print(f"[skip] processed_{idx}: full image path not found in cache")
            continue
        full_img_path = Path(full_img_path_s)
        if not full_img_path.exists():
            print(f"[skip] processed_{idx}: cached full image missing: {full_img_path}")
            continue

        cmap_path = cluster_map_path_for_idx(idx)
        renders_dir = cluster_renders_dir_for_idx(idx)
        if not cmap_path.exists() or not renders_dir.exists():
            print(f"[skip] processed_{idx}: missing clusters.json or renders dir")
            continue

        try:
            cmap = json.loads(cmap_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[skip] processed_{idx}: bad clusters.json: {e}")
            continue

        entries = cmap.get("clusters") or []
        if not isinstance(entries, list) or not entries:
            print(f"[skip] processed_{idx}: no clusters")
            continue

        suggestions_all, base_context = load_candidate_labels(idx, json_index)

        # NEW: first Qwen run to refine candidate labels into component/part labels
        refined_labels = refine_candidate_labels_with_qwen(
            model=model,
            processor=processor,
            device=device,
            base_context=base_context,
            raw_candidates=suggestions_all,
        )

        suggestions = refined_labels[: int(SUGGESTION_LIMIT)]



        # Prepare full image base once
        try:
            full_img0 = Image.open(full_img_path).convert("RGB")
            full_w0, full_h0 = full_img0.size

            right_panel_base, full_scale, full_rw, full_rh, full_pad_x, full_pad_y = _fit_into_square(full_img0, PANEL_EDGE)
            resized_full_base = full_img0.resize((full_rw, full_rh), resample=Image.BICUBIC)
        except Exception as e:
            print(f"[skip] processed_{idx}: cannot open/prepare full: {e}")
            continue

        out_folder = OUT_DIR / f"processed_{idx}"
        out_folder.mkdir(parents=True, exist_ok=True)
        out_json_path = out_folder / "labels.json"

        if SKIP_EXISTING_LABELS and out_json_path.exists():
            print(f"[skip] processed_{idx}: labels already exist: {out_json_path}")
            continue


        results: Dict[str, Any] = {
            "image_index": idx,
            "full_image_source": str(full_img_path),
            "clusters_json": str(cmap_path),
            "candidate_labels_raw": suggestions_all[: int(SUGGESTION_LIMIT)],
            "candidate_labels_refined": suggestions,
            "base_context": base_context,
            "model_id": MODEL_ID,
            "quant_mode": QUANT_MODE,
            "used_quant": bool(used_quant),
            "device": str(device),
            "proc_longest_edge": int(PROC_LONGEST_EDGE),
            "panel_edge": int(PANEL_EDGE),
            "batch_size": int(BATCH_SIZE),
            "clusters": {},
            "cluster_order": [],
        }

        print(f"[run] processed_{idx}: clusters={len(entries)} full={full_img_path.name} vram={_vram_str()}")

        # Build work list (entry + computed static fields)
        work: List[Tuple[int, Any, Path, str, str, Optional[str], List[int], List[float]]] = []
        for ci, entry in enumerate(entries, start=1):
            bbox_xyxy = entry.get("bbox_xyxy")
            cluster_key = f"cluster_{ci:04d}"
            mask_name_for_key = entry.get("crop_file_mask")
            if isinstance(mask_name_for_key, str) and mask_name_for_key.strip():
                cluster_key = f"{ci:04d}_{mask_name_for_key.strip()}"

            if not (isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4):
                _store_cluster_result(results, cluster_key, {"entry": entry, "error": "bad_bbox"})
                if PRINT_EVERY_CLUSTER:
                    print(f"  [cluster {ci}/{len(entries)}] bad_bbox vram={_vram_str()}")
                continue

            bbox_xyxy = [int(x) for x in bbox_xyxy]
            bbox_norm = _bbox_norm_xyxy(bbox_xyxy, full_w0, full_h0)

            mask_name = entry.get("crop_file_mask")
            if not isinstance(mask_name, str):
                _store_cluster_result(results, cluster_key, {"entry": entry, "error": "missing_mask_name"})
                if PRINT_EVERY_CLUSTER:
                    print(f"  [cluster {ci}/{len(entries)}] missing_mask_name vram={_vram_str()}")
                continue

            mask_name = mask_name.strip()
            if not mask_name:
                _store_cluster_result(results, cluster_key, {"entry": entry, "error": "missing_mask_name"})
                if PRINT_EVERY_CLUSTER:
                    print(f"  [cluster {ci}/{len(entries)}] missing_mask_name vram={_vram_str()}")
                continue

            mask_path = renders_dir / mask_name
            if not mask_path.exists():
                _store_cluster_result(results, cluster_key, {"entry": entry, "mask_path": str(mask_path), "error": "mask_missing"})
                if PRINT_EVERY_CLUSTER:
                    print(f"  [cluster {ci}/{len(entries)}] mask_missing vram={_vram_str()}")
                continue

            colour_hint = (
                (entry.get("color_name").strip().lower()
                 if isinstance(entry.get("color_name"), str) and entry.get("color_name").strip()
                 else None)
                or parse_colour_hint_from_filename(mask_name)
            )

            work.append((ci, entry, mask_path, cluster_key, mask_name, colour_hint, bbox_xyxy, bbox_norm))

        # Run in fixed batches
        bs = max(1, int(BATCH_SIZE))
        for start in range(0, len(work), bs):
            chunk = work[start:start + bs]
            if not chunk:
                continue

            texts_batch: List[str] = []
            images_batch: List[Image.Image] = []

            meta_batch: List[Dict[str, Any]] = []

            # Build batch inputs
            for (ci, entry, mask_path, cluster_key, mask_file, colour_hint, bbox_xyxy, bbox_norm) in chunk:
                try:
                    img_mask0 = Image.open(mask_path).convert("RGB")
                except Exception as e:
                    _store_cluster_result(results, cluster_key, {"entry": entry, "mask_path": str(mask_path), "error": f"mask_load_failed: {e}"})
                    if PRINT_EVERY_CLUSTER:
                        print(f"  [cluster {ci}/{len(entries)}] mask_load_failed vram={_vram_str()}")
                    continue

                if ENABLE_FAST_REJECT:
                    try:
                        colored = _count_colored_pixels(img_mask0)
                        if colored < int(MIN_COLORED_PIXELS):
                            _store_cluster_result(results, cluster_key, {
                                "entry": entry,
                                "mask_path": str(mask_path),
                                "bbox_xyxy": bbox_xyxy,
                                "fast_reject": True,
                                "colored_pixels": int(colored),
                                "final": {"decision": "reject", "label": None, "from_suggestions": False, "confidence": 0.0, "reason": "too few colored pixels"},
                            })
                            if PRINT_EVERY_CLUSTER:
                                print(f"  [cluster {ci}/{len(entries)}] fast_reject colored={colored} vram={_vram_str()}")
                            continue
                    except Exception:
                        pass

                left_panel_sq, _, _, _, _, _ = _fit_into_square(img_mask0, PANEL_EDGE)

                bbox_resized = _scale_bbox_xyxy(bbox_xyxy, full_scale)
                bbox_resized = _clamp_bbox_xyxy(bbox_resized, full_rw, full_rh)

                bbox_panel = [
                    bbox_resized[0] + full_pad_x,
                    bbox_resized[1] + full_pad_y,
                    bbox_resized[2] + full_pad_x,
                    bbox_resized[3] + full_pad_y,
                ]

                # right_panel_base is already a square with the resized full image pasted at pad_x/pad_y
                right_panel_sq = right_panel_base.copy()
                _draw_red_rect_pil(right_panel_sq, bbox_panel, thickness=RECT_THICKNESS_PX)

                composite = _make_composite(left_panel_sq, right_panel_sq)

                prompt_text = build_prompt(
                    colour_hint=colour_hint,
                    suggestions=suggestions,
                    base_context=base_context
                )
                SYSTEM_MSG = (
                    "You are a visual object labeling engine\n"
                    "Assosiate objects only to provided materials as opposed to applying general knowledge\n"
                    "If the image is too ambigous reject\n"
                )

                # SmolVLM2 expects chat-template formatting (role/content with image+text).
                # One "conversation" per sample; batching is a list of conversations.
                conv = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                prompt_str = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)


                texts_batch.append(prompt_str)
                images_batch.append(composite)

                meta_batch.append({
                    "ci": ci,
                    "cluster_key": cluster_key,
                    "mask_file": str(mask_file),
                    "entry": entry,
                    "mask_path": str(mask_path),
                    "colour_hint": colour_hint,
                    "bbox_xyxy": bbox_xyxy,
                    "bbox_resized_xyxy": bbox_resized,
                    "bbox_right_panel_xyxy": bbox_panel,
                    "bbox_norm_xyxy": bbox_norm,
                })

            if not texts_batch:
                continue

            # Batch encode -> move -> generate
            t1 = time.time()
            try:
                inputs = _processor_batch(
                    processor,
                    text=texts_batch,
                    images=images_batch,
                )
                inputs = _move_inputs_to_device(inputs, device)

                with torch.inference_mode():
                    out_ids = _safe_generate(model, inputs=inputs, gen_kwargs=gen_kwargs)

                # Decode ONLY the generated continuation (not the prompt).
                # generate() returns [prompt + new_tokens] by default.
                if "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
                    prompt_len = int(inputs["input_ids"].shape[1])
                    gen_only_ids = out_ids[:, prompt_len:]
                else:
                    gen_only_ids = out_ids

                texts_out = processor.batch_decode(gen_only_ids, skip_special_tokens=True)
                dt_batch = time.time() - t1

            except Exception as e:
                if torch.cuda.is_available() and not FORCE_CPU:
                    torch.cuda.empty_cache()
                # Record failure for each element of this batch
                for meta in meta_batch:
                    meta_err = dict(meta)
                    meta_err["error"] = f"batch_inference_failed: {e}"
                    _store_cluster_result(results, str(meta.get("cluster_key", f"cluster_{meta.get('ci','?')}")), meta_err)
                    if PRINT_EVERY_CLUSTER:
                        ci = meta.get("ci", "?")
                        print(f"  [cluster {ci}/{len(entries)}] batch_inference_failed ({e}) vram={_vram_str()}")
                continue

            # Unpack per sample
            # Per-cluster dt is not truly measurable inside a batch without CUDA events per sample,
            # so we print batch dt and also dt/len as an average.
            avg_dt = dt_batch / max(1, len(texts_out))

            n_out = min(len(texts_out), len(meta_batch))
            for j in range(n_out):
                text_out = texts_out[j]
                meta = meta_batch[j]
                ci = int(meta["ci"])
                cluster_key = str(meta.get("cluster_key", f"cluster_{ci:04d}"))

                json_objects = extract_json_objects(text_out)
                parsed = json_objects[0] if json_objects else None
                final = parsed
                capped = False
                if isinstance(final, dict):
                    conf = final.get("confidence")
                    dec = final.get("decision")
                    if isinstance(conf, (int, float)) and isinstance(dec, str):
                        if dec.lower() == "accept" and float(conf) < float(MIN_ACCEPT_CONF):
                            final = dict(final)
                            final["decision"] = "reject"
                            final["label"] = None
                            final["reason"] = (final.get("reason") or "") + " | auto-reject"
                            capped = True

                _store_cluster_result(results, cluster_key, {
                    "entry": meta["entry"],
                    "mask_file": meta.get("mask_file"),
                    "mask_path": meta["mask_path"],
                    "colour_hint": meta["colour_hint"],
                    "bbox_xyxy": meta["bbox_xyxy"],
                    "bbox_resized_xyxy": meta["bbox_resized_xyxy"],
                    "bbox_right_panel_xyxy": meta["bbox_right_panel_xyxy"],
                    "bbox_norm_xyxy": meta["bbox_norm_xyxy"],
                    "batch_size": len(texts_out),
                    "batch_timing_s": float(f"{dt_batch:.4f}"),
                    "avg_per_item_s": float(f"{avg_dt:.4f}"),
                    "model_raw": text_out,
                    "json_objects": json_objects,
                    "parsed": parsed,
                    "final": final,
                    "auto_capped": capped,
                })

                if PRINT_EVERY_CLUSTER:
                    print(f"  [cluster {ci}/{len(entries)}] ok batch_dt={dt_batch:.2f}s avg={avg_dt:.2f}s vram={_vram_str()}")

            if EMPTY_CACHE_EVERY and ((start // bs + 1) % int(EMPTY_CACHE_EVERY) == 0):
                if torch.cuda.is_available() and not FORCE_CPU:
                    torch.cuda.empty_cache()

        out_json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] processed_{idx}: wrote {out_json_path} ({time.time()-t0:.2f}s) vram={_vram_str()}")

        if torch.cuda.is_available() and not FORCE_CPU:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
