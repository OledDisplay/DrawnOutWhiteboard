# shared_models.py
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, List

# -----------------------------
# Thread pool clamps (process-wide)
# Call configure_threads() ASAP, ideally BEFORE importing numpy/torch/opencv in your main entrypoint.
# -----------------------------
def configure_threads(cpu_threads: int = 4) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(cpu_threads))


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    th = threading.current_thread().name
    print(f"[{ts}][{th}] {msg}")


_SIGLIP_DEFAULT_FALLBACK = "google/siglip-base-patch16-384"


def resolve_siglip_model_id(explicit: Optional[str] = None) -> str:
    raw = (
        explicit
        or os.getenv("SIGLIP_MODEL_ID")
        or os.getenv("SIGLIP_NAME")
        or _SIGLIP_DEFAULT_FALLBACK
    )
    model_id = str(raw or "").strip()
    return model_id or _SIGLIP_DEFAULT_FALLBACK


@dataclass
class QwenBundle:
    model: Any
    processor: Any
    device: Any
    model_id: str
    gpu_index: int


@dataclass
class SiglipBundle:
    model: Any
    processor: Any
    device: str
    model_id: str


@dataclass
class SiglipTextBundle:
    model: Any
    processor: Any
    device: str
    model_id: str
    dtype: str


@dataclass
class MiniLMBundle:
    model: Any
    tokenizer: Any
    device: str
    model_id: str
    use_sentence_transformers: bool


@dataclass
class EfficientSAM3Bundle:
    model: Any
    processor: Any
    device: str
    model_id: str
    gpu_index: int


_INIT_LOCK = threading.Lock()
_QWEN_LOCK = threading.Lock()
_SIGLIP_LOCK = threading.Lock()
_SIGLIP_TEXT_LOCK = threading.Lock()
_MINILM_LOCK = threading.Lock()
_SAM_LOCK = threading.Lock()

_QWEN: Optional[QwenBundle] = None
_SIGLIP: Optional[SiglipBundle] = None
_SIGLIP_TEXT: Optional[SiglipTextBundle] = None
_MINILM: Optional[MiniLMBundle] = None
_SAM: Optional[EfficientSAM3Bundle] = None


def _torch_and_cv2_post_config(cpu_threads: int) -> None:
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass

    try:
        import torch
        torch.set_num_threads(max(1, int(cpu_threads)))
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _sync_pinecone_hot_models(*, clear_siglip: bool = False, clear_minilm: bool = False) -> None:
    """
    Keep PineconeFetch in sync with shared workers.
    """
    try:
        siglip_for_pinecone = _SIGLIP_TEXT if _SIGLIP_TEXT is not None else _SIGLIP
        import PineconeFetch
        PineconeFetch.configure_hot_models(
            siglip_bundle=siglip_for_pinecone,
            minilm_bundle=_MINILM,
            clear_siglip=bool(clear_siglip),
            clear_minilm=bool(clear_minilm),
        )
    except Exception:
        pass


def _batch_to_device(batch: Any, device: str) -> Any:
    try:
        return batch.to(device)
    except Exception:
        if isinstance(batch, dict):
            return {k: v.to(device) for k, v in batch.items()}
        return batch


def _clear_cuda_cache(*, gpu_index: Optional[int] = None, reset_peak: bool = False) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            dev = None
            dev_count = int(torch.cuda.device_count() or 0)
            if gpu_index is not None and 0 <= int(gpu_index) < dev_count:
                try:
                    torch.cuda.set_device(int(gpu_index))
                except Exception:
                    pass
                dev = torch.device(f"cuda:{int(gpu_index)}")
            try:
                if dev is not None:
                    torch.cuda.synchronize(dev)
                else:
                    torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            if bool(reset_peak):
                try:
                    if dev is not None:
                        torch.cuda.reset_peak_memory_stats(dev)
                    else:
                        torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
        else:
            return
    except Exception:
        pass


def _release_model_cuda_refs(model: Any) -> None:
    if model is None:
        return

    try:
        import qwentest

        clear_runtime = getattr(qwentest, "_clear_model_runtime_cache_refs", None)
        if callable(clear_runtime):
            clear_runtime(model)
    except Exception:
        pass

    for attr in ("_cache", "_past_key_values", "past_key_values"):
        try:
            if hasattr(model, attr):
                setattr(model, attr, None)
        except Exception:
            pass

    try:
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None and hasattr(gen_cfg, "cache_implementation"):
            gen_cfg.cache_implementation = None
    except Exception:
        pass

    moved = False
    try:
        if hasattr(model, "to_empty"):
            model.to_empty(device="meta")
            moved = True
    except Exception:
        moved = False

    if moved:
        return

    try:
        hf_device_map = getattr(model, "hf_device_map", None)
    except Exception:
        hf_device_map = None

    if hf_device_map:
        return

    try:
        if hasattr(model, "to"):
            model.to("cpu")
        elif hasattr(model, "cpu"):
            model.cpu()
    except Exception:
        pass


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw not in ("0", "false", "no", "off", "")


def _resolve_efficientsam3_checkpoint_path(checkpoint_path: Optional[str]) -> str:
    """
    Resolution order:
      1) explicit checkpoint_path arg
      2) env EFFICIENTSAM3_CKPT
      3) Hugging Face download defaults (env-overridable)
    """
    explicit = str(checkpoint_path or os.getenv("EFFICIENTSAM3_CKPT") or "").strip()
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise RuntimeError(f"EfficientSAM3 checkpoint not found: {explicit}")
        return str(p)

    repo_id = str(os.getenv("EFFICIENTSAM3_HF_REPO_ID", "Simon7108528/EfficientSAM3") or "").strip()
    subfolder = str(os.getenv("EFFICIENTSAM3_HF_SUBFOLDER", "stage1_all_converted") or "").strip()
    filename = str(os.getenv("EFFICIENTSAM3_HF_FILENAME", "efficient_sam3_tinyvit_m.pt") or "").strip()
    local_dir = str(os.getenv("EFFICIENTSAM3_HF_LOCAL_DIR", "checkpoints") or "").strip()
    local_files_only = _env_flag("EFFICIENTSAM3_HF_LOCAL_FILES_ONLY", False)
    token = str(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip() or None

    if not repo_id or not filename:
        raise RuntimeError(
            "Missing EfficientSAM3 HuggingFace config. Set EFFICIENTSAM3_HF_REPO_ID and EFFICIENTSAM3_HF_FILENAME."
        )

    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required to auto-download EfficientSAM3 checkpoint. "
            "Install it or set EFFICIENTSAM3_CKPT."
        ) from e

    _log(
        "Resolving EfficientSAM3 checkpoint via HuggingFace "
        f"(repo={repo_id}, subfolder={subfolder or '<root>'}, file={filename}, local_dir={local_dir})"
    )
    kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "local_dir": local_dir,
        "local_files_only": bool(local_files_only),
    }
    if subfolder:
        kwargs["subfolder"] = subfolder
    if token:
        kwargs["token"] = token

    try:
        ckpt = hf_hub_download(**kwargs)
    except TypeError:
        # Backward compatibility with older huggingface_hub signatures.
        kwargs.pop("token", None)
        ckpt = hf_hub_download(**kwargs)

    p = Path(str(ckpt))
    if not p.is_file():
        raise RuntimeError(f"EfficientSAM3 checkpoint download did not produce a file: {ckpt}")
    return str(p)


def _resolve_efficientsam3_bpe_path() -> str:
    """
    Resolve tokenizer BPE file required by sam3 text encoder.
    Some wheel installs do not package this asset, so we download it if needed.
    """
    filename = str(os.getenv("EFFICIENTSAM3_BPE_FILENAME", "bpe_simple_vocab_16e6.txt.gz") or "").strip()
    explicit = str(os.getenv("EFFICIENTSAM3_BPE_PATH", "") or "").strip()
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise RuntimeError(f"EfficientSAM3 BPE file not found: {explicit}")
        return str(p)

    candidates = [
        Path("checkpoints") / filename,
        Path(__file__).resolve().parent / "checkpoints" / filename,
    ]
    for p in candidates:
        if p.is_file():
            return str(p)

    url = str(
        os.getenv(
            "EFFICIENTSAM3_BPE_URL",
            "https://raw.githubusercontent.com/SimonZeng7108/efficientsam3/main/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        )
        or ""
    ).strip()
    out_dir = Path(str(os.getenv("EFFICIENTSAM3_BPE_LOCAL_DIR", "checkpoints") or "checkpoints").strip())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    _log(f"Resolving EfficientSAM3 BPE file (url={url}, local={out_path})")
    try:
        from urllib.request import urlopen
        with urlopen(url, timeout=60) as resp:
            payload = resp.read()
        out_path.write_bytes(payload)
    except Exception as e:
        raise RuntimeError(
            "Failed to resolve EfficientSAM3 BPE file. "
            "Set EFFICIENTSAM3_BPE_PATH to an existing file."
        ) from e

    if not out_path.is_file():
        raise RuntimeError(f"EfficientSAM3 BPE download did not produce a file: {out_path}")
    return str(out_path)


def _preferred_siglip_devices(gpu_index: int) -> List[str]:
    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(int(gpu_index))
            except Exception:
                pass
            return ["cuda", "cpu"]
    except Exception:
        pass
    return ["cpu"]


class _SiglipTextOnlyModelAdapter:
    """
    Minimal adapter so callers can keep using `get_text_features` shape/API
    while only the SigLIP text tower is loaded.
    """
    def __init__(self, text_model: Any):
        self.text_model = text_model

    def to(self, *args, **kwargs):
        self.text_model = self.text_model.to(*args, **kwargs)
        return self

    def eval(self):
        self.text_model.eval()
        return self

    def cpu(self):
        self.text_model.cpu()
        return self

    def get_text_features(
        self,
        input_ids: Any,
        attention_mask: Any = None,
        position_ids: Any = None,
        **kwargs,
    ) -> Any:
        out = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        return out.pooler_output

    def __getattr__(self, name: str) -> Any:
        return getattr(self.text_model, name)


def init_siglip_text_hot(
    *,
    siglip_model_id: Optional[str] = None,
    gpu_index: int = 0,
    cpu_threads: int = 4,
    warmup: bool = True,
    prefer_fp8: bool = False,
) -> None:
    """
    Loads only SigLIP text tower for Pinecone text->clip embeddings.
    Tries CUDA first using BF16/FP16, then falls back to CPU immediately.
    """
    global _SIGLIP_TEXT

    siglip_model_id = resolve_siglip_model_id(siglip_model_id)
    configure_threads(cpu_threads)
    _torch_and_cv2_post_config(cpu_threads)

    with _INIT_LOCK:
        if _SIGLIP_TEXT is None:
            _log(f"Loading SigLIP text-only once: {siglip_model_id}")
            try:
                import gc
                import torch
                from transformers import AutoProcessor, SiglipTextModel

                processor = AutoProcessor.from_pretrained(siglip_model_id)
                last_err: Optional[Exception] = None

                for device in _preferred_siglip_devices(gpu_index):
                    text_model = None
                    try:
                        load_dtype = torch.float32
                        if device == "cuda":
                            # Lower VRAM pressure before optional FP8 cast.
                            load_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

                        text_model = SiglipTextModel.from_pretrained(
                            siglip_model_id,
                            torch_dtype=load_dtype,
                            low_cpu_mem_usage=True,
                        )
                        adapter = _SiglipTextOnlyModelAdapter(text_model=text_model)
                        adapter.to(device)

                        active_dtype = str(load_dtype).replace("torch.", "")
                        if device == "cuda" and prefer_fp8 and hasattr(torch, "float8_e4m3fn"):
                            try:
                                adapter.to(dtype=torch.float8_e4m3fn)
                                active_dtype = "float8_e4m3fn"
                            except Exception as fp8_err:
                                _log(f"[WARN] SigLIP text FP8 cast failed on CUDA, keeping {active_dtype}: {fp8_err}")

                        adapter.eval()

                        if warmup:
                            inputs = processor(
                                text=["warmup"],
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                            )
                            inputs = _batch_to_device(inputs, device)
                            with torch.inference_mode():
                                _ = adapter.get_text_features(**inputs)

                        _SIGLIP_TEXT = SiglipTextBundle(
                            model=adapter,
                            processor=processor,
                            device=device,
                            model_id=siglip_model_id,
                            dtype=active_dtype,
                        )
                        _log(f"SigLIP text-only loaded and kept hot on {device} ({active_dtype}).")
                        break
                    except Exception as e:
                        last_err = e
                        if text_model is not None:
                            try:
                                del text_model
                            except Exception:
                                pass
                        gc.collect()
                        if device == "cuda":
                            _clear_cuda_cache()
                            _log(f"[WARN] SigLIP text CUDA load failed, falling back to CPU: {e}")

                if _SIGLIP_TEXT is None:
                    _log(f"[WARN] SigLIP text-only load failed (skipping): {last_err}")
            except Exception as e:
                _log(f"[WARN] SigLIP text-only setup failed (skipping): {e}")
                _SIGLIP_TEXT = None

        _sync_pinecone_hot_models(clear_siglip=True, clear_minilm=False)


def init_siglip_hot(
    *,
    siglip_model_id: Optional[str] = None,
    gpu_index: int = 0,
    cpu_threads: int = 4,
    warmup: bool = True,
) -> None:
    global _SIGLIP

    siglip_model_id = resolve_siglip_model_id(siglip_model_id)
    configure_threads(cpu_threads)
    _torch_and_cv2_post_config(cpu_threads)

    with _INIT_LOCK:
        if _SIGLIP is None:
            _log(f"Loading SigLIP once: {siglip_model_id}")
            try:
                import gc
                import torch
                from transformers import AutoProcessor, AutoModel

                processor = AutoProcessor.from_pretrained(siglip_model_id)
                last_err: Optional[Exception] = None

                for device in _preferred_siglip_devices(gpu_index):
                    model = None
                    try:
                        load_kwargs: dict[str, Any] = {
                            "low_cpu_mem_usage": True,
                        }
                        if device == "cuda":
                            load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                        else:
                            load_kwargs["torch_dtype"] = torch.float32

                        model = AutoModel.from_pretrained(siglip_model_id, **load_kwargs)
                        model.to(device)
                        model.eval()

                        if warmup:
                            try:
                                from PIL import Image
                                img = Image.new("RGB", (64, 64), (0, 0, 0))
                                inputs = processor(images=img, return_tensors="pt")
                                inputs = _batch_to_device(inputs, device)
                                with torch.inference_mode():
                                    _ = model.get_image_features(**inputs)
                            except Exception as e:
                                _log(f"[WARN] SigLIP warmup failed: {e}")

                        _SIGLIP = SiglipBundle(model=model, processor=processor, device=device, model_id=siglip_model_id)
                        _log(f"SigLIP loaded and kept hot on {device}.")
                        break
                    except Exception as e:
                        last_err = e
                        if model is not None:
                            try:
                                del model
                            except Exception:
                                pass
                        gc.collect()
                        if device == "cuda":
                            _clear_cuda_cache()
                            _log(f"[WARN] SigLIP CUDA load failed, falling back to CPU: {e}")

                if _SIGLIP is None:
                    _log(f"[WARN] SigLIP load failed (skipping): {last_err}")
            except Exception as e:
                _log(f"[WARN] SigLIP load failed (skipping): {e}")
                _SIGLIP = None

        _sync_pinecone_hot_models(clear_siglip=True, clear_minilm=False)


def init_minilm_hot(
    *,
    minilm_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    gpu_index: int = 0,
    cpu_threads: int = 4,
    warmup: bool = True,
) -> None:
    global _MINILM

    configure_threads(cpu_threads)
    _torch_and_cv2_post_config(cpu_threads)

    with _INIT_LOCK:
        if _MINILM is None:
            _log(f"Loading MiniLM once: {minilm_model_id}")
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cuda":
                    torch.cuda.set_device(gpu_index)

                use_st = False
                try:
                    from sentence_transformers import SentenceTransformer  # type: ignore
                    st = SentenceTransformer(minilm_model_id, device=device)
                    if warmup:
                        _ = st.encode(["warmup"], normalize_embeddings=True)
                    _MINILM = MiniLMBundle(model=st, tokenizer=None, device=device, model_id=minilm_model_id, use_sentence_transformers=True)
                    use_st = True
                    _log("MiniLM loaded via sentence-transformers and kept hot.")
                except Exception:
                    use_st = False

                if not use_st:
                    from transformers import AutoTokenizer, AutoModel
                    tok = AutoTokenizer.from_pretrained(minilm_model_id)
                    mdl = AutoModel.from_pretrained(minilm_model_id)
                    mdl.to(device)
                    mdl.eval()
                    if warmup:
                        with torch.inference_mode():
                            inputs = tok(["warmup"], return_tensors="pt", padding=True, truncation=True).to(device)
                            _ = mdl(**inputs).last_hidden_state
                    _MINILM = MiniLMBundle(model=mdl, tokenizer=tok, device=device, model_id=minilm_model_id, use_sentence_transformers=False)
                    _log("MiniLM loaded via transformers and kept hot.")

            except Exception as e:
                _log(f"[WARN] MiniLM load failed (skipping): {e}")
                _MINILM = None

        _sync_pinecone_hot_models(clear_siglip=False, clear_minilm=True)


def init_siglip_minilm_hot(
    *,
    siglip_model_id: Optional[str] = None,
    minilm_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    gpu_index: int = 0,
    cpu_threads: int = 4,
    warmup: bool = True,
) -> None:
    init_siglip_hot(
        siglip_model_id=siglip_model_id,
        gpu_index=gpu_index,
        cpu_threads=cpu_threads,
        warmup=warmup,
    )
    init_minilm_hot(
        minilm_model_id=minilm_model_id,
        gpu_index=gpu_index,
        cpu_threads=cpu_threads,
        warmup=warmup,
    )


def init_qwen_hot(
    *,
    qwen_model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    gpu_index: int = 0,
    cpu_threads: int = 4,
    warmup: bool = True,
) -> None:
    global _QWEN

    configure_threads(cpu_threads)
    _torch_and_cv2_post_config(cpu_threads)

    with _INIT_LOCK:
        if _QWEN is None:
            _log(f"Loading Qwen once: {qwen_model_id} (gpu_index={gpu_index})")
            import qwentest  # local module

            try:
                # Prefer direct low-peak load path (avoids CPU float32 staging spikes).
                orig_model_id = getattr(qwentest, "MODEL_ID", None)
                orig_gpu_index = getattr(qwentest, "GPU_INDEX", None)
                try:
                    qwentest.MODEL_ID = qwen_model_id
                    qwentest.GPU_INDEX = int(gpu_index)
                    model, processor, device, _used_quant = qwentest.load_model_and_processor()
                finally:
                    if orig_model_id is not None:
                        qwentest.MODEL_ID = orig_model_id
                    if orig_gpu_index is not None:
                        qwentest.GPU_INDEX = orig_gpu_index
            except Exception as direct_load_err:
                _log(f"[WARN] Qwen direct load path failed, falling back to CPU staging: {direct_load_err}")
                model, processor, device = qwentest.preload_qwen_cpu_only(qwen_model_id, warmup=False)
                try:
                    model, device = qwentest.move_qwen_to_cuda_if_available(model, gpu_index=gpu_index)
                except Exception as e:
                    _log(f"[WARN] Qwen CUDA move failed, staying on CPU: {e}")

            try:
                model.eval()
            except Exception:
                pass

            if warmup:
                try:
                    qwentest.warmup_qwen_once(model, processor, device)
                except Exception as e:
                    _log(f"[WARN] Qwen warmup failed: {e}")

            _QWEN = QwenBundle(model=model, processor=processor, device=device, model_id=qwen_model_id, gpu_index=gpu_index)
            _log("Qwen loaded and kept hot.")


def init_efficientsam3_hot(
    *,
    gpu_index: int = 0,
    cpu_threads: int = 4,
    warmup: bool = True,
    checkpoint_path: Optional[str] = None,
    backbone_type: str = "tinyvit",
    model_name: str = "11m",
    text_encoder_type: str = "MobileCLIP-S1",
) -> None:
    """
    Loads EfficientSAM3 (text prompt) once per process.
    Checkpoint resolution:
      - checkpoint_path arg or env EFFICIENTSAM3_CKPT
      - otherwise auto-download from HuggingFace (env-overridable defaults)
    """
    global _SAM

    configure_threads(cpu_threads)
    _torch_and_cv2_post_config(cpu_threads)

    ckpt = _resolve_efficientsam3_checkpoint_path(checkpoint_path)
    bpe_path = _resolve_efficientsam3_bpe_path()

    with _INIT_LOCK:
        if _SAM is None:
            _log(f"Loading EfficientSAM3 once: ckpt={ckpt} bpe={bpe_path} gpu_index={gpu_index}")
            import torch
            from sam3.model_builder import build_efficientsam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                torch.cuda.set_device(int(gpu_index))

            model = build_efficientsam3_image_model(
                bpe_path=bpe_path,
                checkpoint_path=ckpt,
                backbone_type=backbone_type,
                model_name=model_name,
                text_encoder_type=text_encoder_type,
            )

            try:
                model.to(device)
            except Exception:
                pass

            try:
                model.eval()
            except Exception:
                pass

            raw_thr = str(os.getenv("EFFICIENTSAM3_CONF_THRESHOLD", "0.01") or "0.01").strip()
            try:
                conf_thr = float(raw_thr)
            except Exception:
                conf_thr = 0.01
            conf_thr = max(0.0, min(1.0, conf_thr))

            processor = Sam3Processor(model, confidence_threshold=conf_thr)
            _log(f"EfficientSAM3 processor confidence_threshold={conf_thr:.4f}")

            if warmup:
                try:
                    from PIL import Image
                    img = Image.new("RGB", (128, 128), (255, 255, 255))
                    st = processor.set_image(img)
                    _ = processor.set_text_prompt(prompt="warmup", state=dict(st))
                except Exception as e:
                    _log(f"[WARN] EfficientSAM3 warmup failed: {e}")

            _SAM = EfficientSAM3Bundle(model=model, processor=processor, device=device, model_id="efficientsam3_text_prompt", gpu_index=int(gpu_index))
            _log("EfficientSAM3 loaded and kept hot.")


def unload_qwen() -> None:
    """
    Actually releases refs + empties CUDA cache.
    """
    global _QWEN
    with _INIT_LOCK:
        if _QWEN is None:
            return
        _log("Unloading Qwen...")
        bundle = _QWEN
        _QWEN = None
        try:
            import gc

            try:
                _release_model_cuda_refs(bundle.model)
            except Exception:
                pass

            try:
                bundle.model = None
            except Exception:
                pass
            try:
                bundle.processor = None
            except Exception:
                pass
            gc.collect()
            gc.collect()
            _clear_cuda_cache(gpu_index=bundle.gpu_index, reset_peak=True)
        except Exception:
            _QWEN = None


def unload_siglip() -> None:
    """
    Releases SigLIP worker and GPU memory.
    """
    global _SIGLIP
    with _INIT_LOCK:
        if _SIGLIP is None:
            _sync_pinecone_hot_models(clear_siglip=True, clear_minilm=False)
            return
        _log("Unloading SigLIP...")
        try:
            import gc

            try:
                if hasattr(_SIGLIP.model, "cpu"):
                    _SIGLIP.model.cpu()
            except Exception:
                pass

            _SIGLIP = None
            gc.collect()
            _clear_cuda_cache()
        except Exception:
            _SIGLIP = None
        finally:
            _sync_pinecone_hot_models(clear_siglip=True, clear_minilm=False)


def unload_siglip_text() -> None:
    """
    Releases SigLIP text-only worker and GPU memory.
    """
    global _SIGLIP_TEXT
    with _INIT_LOCK:
        if _SIGLIP_TEXT is None:
            _sync_pinecone_hot_models(clear_siglip=True, clear_minilm=False)
            return
        _log("Unloading SigLIP text-only...")
        try:
            import gc

            try:
                if hasattr(_SIGLIP_TEXT.model, "cpu"):
                    _SIGLIP_TEXT.model.cpu()
            except Exception:
                pass

            _SIGLIP_TEXT = None
            gc.collect()
            _clear_cuda_cache()
        except Exception:
            _SIGLIP_TEXT = None
        finally:
            _sync_pinecone_hot_models(clear_siglip=True, clear_minilm=False)


def unload_minilm() -> None:
    """
    Releases MiniLM worker and GPU memory.
    """
    global _MINILM
    with _INIT_LOCK:
        if _MINILM is None:
            _sync_pinecone_hot_models(clear_siglip=False, clear_minilm=True)
            return
        _log("Unloading MiniLM...")
        try:
            import torch
            import gc

            try:
                mdl = _MINILM.model
                if hasattr(mdl, "cpu"):
                    mdl.cpu()
                elif hasattr(mdl, "to"):
                    mdl.to("cpu")
            except Exception:
                pass

            _MINILM = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            _MINILM = None
        finally:
            _sync_pinecone_hot_models(clear_siglip=False, clear_minilm=True)


def unload_siglip_minilm() -> None:
    unload_siglip()
    unload_minilm()


def unload_efficientsam3() -> None:
    global _SAM
    with _INIT_LOCK:
        if _SAM is None:
            return
        _log("Unloading EfficientSAM3...")
        bundle = _SAM
        _SAM = None
        try:
            import gc

            try:
                _release_model_cuda_refs(bundle.model)
            except Exception:
                pass

            try:
                bundle.model = None
            except Exception:
                pass
            try:
                bundle.processor = None
            except Exception:
                pass
            gc.collect()
            gc.collect()
            _clear_cuda_cache(gpu_index=bundle.gpu_index, reset_peak=True)
        except Exception:
            _SAM = None


def init_hot_models(
    *,
    qwen_model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    siglip_model_id: Optional[str] = None,
    minilm_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    gpu_index: int = 0,
    cpu_threads: int = 4,
    warmup: bool = True,
) -> None:
    init_siglip_minilm_hot(
        siglip_model_id=siglip_model_id,
        minilm_model_id=minilm_model_id,
        gpu_index=gpu_index,
        cpu_threads=cpu_threads,
        warmup=warmup,
    )
    init_qwen_hot(
        qwen_model_id=qwen_model_id,
        gpu_index=gpu_index,
        cpu_threads=cpu_threads,
        warmup=warmup,
    )


def get_siglip() -> Optional[SiglipBundle]:
    return _SIGLIP


def get_siglip_text() -> Optional[SiglipTextBundle]:
    return _SIGLIP_TEXT


def get_minilm() -> Optional[MiniLMBundle]:
    return _MINILM


def get_qwen() -> Optional[QwenBundle]:
    return _QWEN


def get_efficientsam3() -> Optional[EfficientSAM3Bundle]:
    return _SAM


def qwen_lock() -> threading.Lock:
    return _QWEN_LOCK


def sam_lock() -> threading.Lock:
    return _SAM_LOCK


# ---------- Optional embedding helpers ----------
def siglip_embed_pil_images(pil_images: List[Any]) -> Optional[List[List[float]]]:
    if _SIGLIP is None:
        return None
    import torch

    with _SIGLIP_LOCK, torch.inference_mode():
        proc = _SIGLIP.processor
        mdl = _SIGLIP.model
        dev = _SIGLIP.device
        inputs = proc(images=pil_images, return_tensors="pt", padding=True).to(dev)
        feats = mdl.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.detach().cpu().tolist()


def minilm_embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    if _MINILM is None:
        return None

    if _MINILM.use_sentence_transformers:
        with _MINILM_LOCK:
            vecs = _MINILM.model.encode(texts, normalize_embeddings=True)
            return vecs.tolist() if hasattr(vecs, "tolist") else [list(map(float, v)) for v in vecs]

    import torch
    with _MINILM_LOCK, torch.inference_mode():
        tok = _MINILM.tokenizer
        mdl = _MINILM.model
        dev = _MINILM.device

        batch = tok(texts, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(dev) for k, v in batch.items()}
        out = mdl(**batch).last_hidden_state

        mask = batch["attention_mask"].unsqueeze(-1).expand(out.size()).float()
        summed = (out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)

        return pooled.detach().cpu().tolist()
