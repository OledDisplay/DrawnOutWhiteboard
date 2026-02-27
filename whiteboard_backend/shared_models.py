# shared_models.py
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
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
_MINILM_LOCK = threading.Lock()
_SAM_LOCK = threading.Lock()

_QWEN: Optional[QwenBundle] = None
_SIGLIP: Optional[SiglipBundle] = None
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


def init_siglip_minilm_hot(
    *,
    siglip_model_id: str = "google/siglip-so400m-patch14-384",
    minilm_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    gpu_index: int = 0,
    cpu_threads: int = 4,
    warmup: bool = True,
) -> None:
    global _SIGLIP, _MINILM

    configure_threads(cpu_threads)
    _torch_and_cv2_post_config(cpu_threads)

    with _INIT_LOCK:
        # SIGLIP
        if _SIGLIP is None:
            _log(f"Loading SigLIP once: {siglip_model_id}")
            try:
                import torch
                from transformers import AutoProcessor, AutoModel

                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cuda":
                    torch.cuda.set_device(gpu_index)

                processor = AutoProcessor.from_pretrained(siglip_model_id)
                model = AutoModel.from_pretrained(siglip_model_id)
                model.to(device)
                model.eval()

                if warmup:
                    try:
                        from PIL import Image
                        img = Image.new("RGB", (64, 64), (0, 0, 0))
                        inputs = processor(images=img, return_tensors="pt").to(device)
                        with torch.inference_mode():
                            _ = model.get_image_features(**inputs)
                    except Exception as e:
                        _log(f"[WARN] SigLIP warmup failed: {e}")

                _SIGLIP = SiglipBundle(model=model, processor=processor, device=device, model_id=siglip_model_id)
                _log("SigLIP loaded and kept hot.")
            except Exception as e:
                _log(f"[WARN] SigLIP load failed (skipping): {e}")
                _SIGLIP = None

        # MINILM
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

            model, processor, device = qwentest.preload_qwen_cpu_only(qwen_model_id)

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
    Requires checkpoint path (env EFFICIENTSAM3_CKPT or explicit).
    """
    global _SAM

    configure_threads(cpu_threads)
    _torch_and_cv2_post_config(cpu_threads)

    ckpt = (checkpoint_path or os.getenv("EFFICIENTSAM3_CKPT") or "").strip()
    if not ckpt:
        raise RuntimeError("Missing EfficientSAM3 checkpoint. Set EFFICIENTSAM3_CKPT or pass checkpoint_path=...")

    with _INIT_LOCK:
        if _SAM is None:
            _log(f"Loading EfficientSAM3 once: ckpt={ckpt} gpu_index={gpu_index}")
            import torch
            from sam3.model_builder import build_efficientsam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                torch.cuda.set_device(int(gpu_index))

            model = build_efficientsam3_image_model(
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

            processor = Sam3Processor(model)

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
        try:
            import torch
            import gc

            try:
                if hasattr(_QWEN.model, "cpu"):
                    _QWEN.model.cpu()
            except Exception:
                pass

            _QWEN = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            _QWEN = None


def unload_efficientsam3() -> None:
    global _SAM
    with _INIT_LOCK:
        if _SAM is None:
            return
        _log("Unloading EfficientSAM3...")
        try:
            import torch
            import gc

            try:
                if hasattr(_SAM.model, "cpu"):
                    _SAM.model.cpu()
            except Exception:
                pass

            _SAM = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            _SAM = None


def init_hot_models(
    *,
    qwen_model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    siglip_model_id: str = "google/siglip-so400m-patch14-384",
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