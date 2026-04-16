#!/usr/bin/env python3
from __future__ import annotations

# --------------------------------------------
# Process-wide thread + backend clamps
# IMPORTANT: keep this BEFORE importing cv2/numpy/torch/transformers
# --------------------------------------------
import os 
_PIPE_CPU_THREADS = int(os.getenv("PIPE_CPU_THREADS", "4") or "4")
os.environ.setdefault("OMP_NUM_THREADS", str(_PIPE_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_PIPE_CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_PIPE_CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_PIPE_CPU_THREADS))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(_PIPE_CPU_THREADS))

import sys
import threading
import time
import traceback
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from contextlib import contextmanager

import cv2
import numpy as np

# clamp OpenCV internal threads (avoids “random” oversubscription stalls)
try:
    cv2.setNumThreads(0)
except Exception:
    pass

import ImagePreprocessor
import ImageSkeletonizer
import ImageVectorizer
import ImageClusters
import ImageColours
import LineDescriptors

import PineconeSave
import PineconeFetch

import shared_models

from typing import Any, Dict, List, Optional

@dataclass
class PipelineWorkers:
    # SigLIP + MiniLM are used by PineconeFetch (configured externally but passed through here)
    siglip_bundle: Any = None
    minilm_bundle: Any = None
    precomputed_clip_embeddings_by_processed_id: Dict[str, List[float]] = field(default_factory=dict)


IN_DIR_ROOT = Path("ResearchImages\\UniqueImages")
BASE_DIR = Path(__file__).resolve().parent

# -------------------------
# ADDED: metadata context json path (as you described)
# -------------------------
METADATA_CONTEXT_PATH = IN_DIR_ROOT / "image_metadata_context.json"


# ============================================
# DEBUG TIMING (stage profiler)
# ============================================
_STAGE = defaultdict(float)

@contextmanager
def stage(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _STAGE[name] += (time.perf_counter() - t0)

def stage_report() -> None:
    items = sorted(_STAGE.items(), key=lambda x: x[1], reverse=True)
    total = sum(v for _, v in items) or 1e-9
    print("\n=== STAGE REPORT ===")
    for k, v in items:
        print(f"{k:34s} {v:8.2f}s ({(v/total*100):5.1f}%)")
    print(f"{'TOTAL':34s} {total:8.2f}s\n")


# ============================================
# HOT MODEL REGISTRY (SigLIP + MiniLM)
# Keep models loaded once per process
# ============================================
@dataclass
class _SiglipBundle:
    model: Any
    processor: Any
    device: str
    model_id: str

@dataclass
class _MiniLMBundle:
    model: Any
    tokenizer: Any
    device: str
    model_id: str
    use_sentence_transformers: bool

_HOT_INIT_LOCK = threading.Lock()
_SIGLIP_LOCK = threading.Lock()
_MINILM_LOCK = threading.Lock()

_HOT_SIGLIP: Optional[_SiglipBundle] = None
_HOT_MINILM: Optional[_MiniLMBundle] = None

import urllib.request
import urllib.error

REMOTE_API = os.getenv("PIPE_REMOTE_API", "").strip()  # e.g. "http://127.0.0.1:8787"

def _http_post_json(url: str, payload: Dict[str, Any], timeout: int = 3600) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8") or "{}")



def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    th = threading.current_thread().name
    print(f"[{ts}][{th}] {msg}")


def _torch_post_config(cpu_threads: int) -> None:
    try:
        import torch
        torch.set_num_threads(max(1, int(cpu_threads)))
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def ensure_hot_models(
    *,
    qwen_model_id: Optional[str] = None,
    siglip_model_id: Optional[str] = None,
    minilm_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    gpu_index: int = 0,
    cpu_threads: int = _PIPE_CPU_THREADS,
    warmup: bool = True,
    load_siglip: bool = True,
    load_minilm: bool = True,
) -> None:
    """
    Loads SigLIP/MiniLM exactly once per PROCESS and keeps them resident.
    If you run this script as a persistent service (--serve), all calls reuse the same hot models.
    `qwen_model_id` is accepted only for backward compatibility and is ignored.
    """
    global _HOT_SIGLIP, _HOT_MINILM

    siglip_model_id = shared_models.resolve_siglip_model_id(siglip_model_id)
    _torch_post_config(cpu_threads)

    with _HOT_INIT_LOCK:
        # ---- SIGLIP ----
        if load_siglip and _HOT_SIGLIP is None:
            _log(f"Loading SigLIP (one-time): {siglip_model_id}")
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

                _HOT_SIGLIP = _SiglipBundle(model=model, processor=processor, device=device, model_id=siglip_model_id)
                _log("SigLIP is hot.")
            except Exception as e:
                _HOT_SIGLIP = None
                _log(f"[WARN] SigLIP load failed (skipping): {repr(e)}")

        # ---- MINILM ----
        if load_minilm and _HOT_MINILM is None:
            _log(f"Loading MiniLM (one-time): {minilm_model_id}")
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cuda":
                    torch.cuda.set_device(gpu_index)

                # Prefer sentence-transformers if available
                try:
                    from sentence_transformers import SentenceTransformer  # type: ignore
                    st = SentenceTransformer(minilm_model_id, device=device)
                    if warmup:
                        _ = st.encode(["warmup"], normalize_embeddings=True)
                    _HOT_MINILM = _MiniLMBundle(
                        model=st,
                        tokenizer=None,
                        device=device,
                        model_id=minilm_model_id,
                        use_sentence_transformers=True,
                    )
                    _log("MiniLM is hot (sentence-transformers).")
                except Exception:
                    from transformers import AutoTokenizer, AutoModel
                    tok = AutoTokenizer.from_pretrained(minilm_model_id)
                    mdl = AutoModel.from_pretrained(minilm_model_id)
                    mdl.to(device)
                    mdl.eval()
                    if warmup:
                        with torch.inference_mode():
                            inputs = tok(["warmup"], return_tensors="pt", padding=True, truncation=True).to(device)
                            _ = mdl(**inputs).last_hidden_state
                    _HOT_MINILM = _MiniLMBundle(
                        model=mdl,
                        tokenizer=tok,
                        device=device,
                        model_id=minilm_model_id,
                        use_sentence_transformers=False,
                    )
                    _log("MiniLM is hot (transformers).")
            except Exception as e:
                _HOT_MINILM = None
                _log(f"[WARN] MiniLM load failed (skipping): {repr(e)}")

        # ---- wire hot SigLIP/MiniLM into PineconeFetch (same process) ----
        try:
            import PineconeFetch
            PineconeFetch.configure_hot_models(
                siglip_bundle=_HOT_SIGLIP,
                minilm_bundle=_HOT_MINILM,
                clear_siglip=True,
                clear_minilm=True,
            )
        except Exception:
            pass

# ============================================
# Pinecone embedding helpers (SigLIP image + MiniLM text)
# ============================================
_EMBED_SIGLIP_LOCK = threading.Lock()
_EMBED_MINILM_LOCK = threading.Lock()

def _mean_pool_trf(last_hidden, attention_mask):
    # last_hidden: (B,T,H), attention_mask: (B,T)
    import torch
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom

def _minilm_embed_texts_hot(texts: List[str], minilm_bundle: Any) -> Optional[List[List[float]]]:
    """
    Supports either:
      - sentence-transformers bundle (.use_sentence_transformers True, model.encode)
      - transformers bundle (.tokenizer + .model)
    """
    if not texts:
        return []
    if minilm_bundle is None:
        return None

    # sentence-transformers path
    try:
        if getattr(minilm_bundle, "use_sentence_transformers", False) and hasattr(minilm_bundle.model, "encode"):
            with _EMBED_MINILM_LOCK:
                vecs = minilm_bundle.model.encode(texts, normalize_embeddings=True)
            # vecs can be np.ndarray
            return vecs.tolist() if hasattr(vecs, "tolist") else [list(map(float, v)) for v in vecs]
    except Exception:
        pass

    # transformers path
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

        with _EMBED_MINILM_LOCK, torch.inference_mode():
            batch = tok(texts, return_tensors="pt", padding=True, truncation=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = mdl(**batch).last_hidden_state  # (B,T,H)
            pooled = _mean_pool_trf(out, batch["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            return pooled.detach().cpu().tolist()
    except Exception:
        return None

def _siglip_embed_pil_images_hot(pil_images: List[Any], siglip_bundle: Any) -> Optional[List[List[float]]]:
    """
    Uses injected SigLIP bundle from shared_models (AutoProcessor + AutoModel typically).
    Returns normalized image embeddings.
    """
    if not pil_images:
        return []
    if siglip_bundle is None:
        return None

    try:
        import torch
        proc = getattr(siglip_bundle, "processor", None)
        mdl = getattr(siglip_bundle, "model", None)
        dev = getattr(siglip_bundle, "device", None)

        if proc is None or mdl is None:
            return None

        device = dev
        if not isinstance(device, torch.device):
            device = torch.device(str(device) if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        with _EMBED_SIGLIP_LOCK, torch.inference_mode():
            inputs = proc(images=pil_images, return_tensors="pt", padding=True)
            try:
                inputs = inputs.to(device)
            except Exception:
                inputs = {k: v.to(device) for k, v in inputs.items()}

            feats = mdl.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            return feats.detach().cpu().tolist()
    except Exception:
        return None

def _build_pinecone_jobs_from_text_items(
    *,
    text_items_all: List[Dict[str, Any]],
    meta_ctx_map: Dict[str, Dict[str, Any]],
    workers: PipelineWorkers,
    handle: PipelineHandle,
) -> List[Dict[str, Any]]:
    """
    Builds PineconeSave jobs:
      {
        "processed_id": ...,
        "unique_path": ...,
        "base_context": ...,
        "prompt_embedding": [...],
        "clip_embedding": [...],
        "context_embedding": [...],
        "meta": {...}
      }

    Diagram tagging:
      - base_contexts flagged as diagrams are marked with is_diagram=1
      - if labels are present in handle.refined_labels_by_processed_id they are included in metadata
    """
    rows: List[Dict[str, Any]] = []
    for t in (text_items_all or []):
        pid = str(t.get("processed_id", "") or "").strip()
        if not pid:
            continue

        bc = str(t.get("base_context", "") or "").strip()
        src = str(t.get("source_path", "") or "").strip()
        if not src:
            src = str(handle.processed_to_unique.get(pid, "") or "").strip()

        mbgr = t.get("masked_bgr", None)
        rows.append({"processed_id": pid, "base_context": bc, "source_path": src, "masked_bgr": mbgr})

    if not rows:
        return []

    prompts = [r["base_context"] for r in rows]
    prompt_vecs = _minilm_embed_texts_hot(prompts, workers.minilm_bundle)

    pil_images: List[Any] = []
    pil_ok_mask: List[bool] = []
    for r in rows:
        mbgr = r.get("masked_bgr", None)
        if mbgr is None:
            pil_images.append(None)
            pil_ok_mask.append(False)
            continue
        try:
            pil_images.append(_bgr_to_pil_fast(mbgr, longest_side=600))
            pil_ok_mask.append(True)
        except Exception:
            pil_images.append(None)
            pil_ok_mask.append(False)

    clip_vecs_full: List[Optional[List[float]]] = [None] * len(rows)
    valid_pils = [pil_images[i] for i, ok in enumerate(pil_ok_mask) if ok]
    if valid_pils:
        clip_vecs = _siglip_embed_pil_images_hot(valid_pils, workers.siglip_bundle)
        if isinstance(clip_vecs, list) and len(clip_vecs) == len(valid_pils):
            j = 0
            for i, ok in enumerate(pil_ok_mask):
                if ok:
                    clip_vecs_full[i] = clip_vecs[j]
                    j += 1

    jobs: List[Dict[str, Any]] = []

    for i, r in enumerate(rows):
        pid = r["processed_id"]
        bc = r["base_context"]
        src = r["source_path"]

        pvec = None
        if isinstance(prompt_vecs, list) and i < len(prompt_vecs):
            pvec = prompt_vecs[i]

        cvec = clip_vecs_full[i]
        try:
            precomputed_clip = workers.precomputed_clip_embeddings_by_processed_id.get(pid)
        except Exception:
            precomputed_clip = None
        if cvec is None and isinstance(precomputed_clip, list) and precomputed_clip:
            cvec = list(precomputed_clip)

        ctx_vec = None
        final_score = None
        meta_entry = None
        try:
            if src:
                meta_entry = _resolve_meta_for_source(meta_ctx_map, src)
                if meta_entry:
                    final_score = meta_entry.get("final_score", None)
                    ctx_vec = _pick_best_context_embedding(meta_entry)
        except Exception:
            ctx_vec = None

        # If live MiniLM prompt embedding is unavailable, reuse metadata prompt embedding when present.
        if pvec is None and isinstance(meta_entry, dict):
            try:
                pvec = _pick_prompt_embedding(meta_entry)
            except Exception:
                pvec = None

        # If live SigLIP image embedding is unavailable, reuse metadata clip embedding when present.
        if cvec is None and isinstance(meta_entry, dict):
            try:
                cvec = _pick_clip_embedding(meta_entry)
            except Exception:
                cvec = None

        # Generated/comfy images do not have smart_hits context, so use prompt space as context.
        if ctx_vec is None and isinstance(meta_entry, dict) and _is_generated_or_comfy_meta(meta_entry):
            if isinstance(pvec, list):
                ctx_vec = pvec
            else:
                try:
                    ctx_vec = _pick_prompt_embedding(meta_entry)
                except Exception:
                    ctx_vec = None

        # -------------------------
        # Diagram label + tag
        # -------------------------
        refined = handle.refined_labels_by_processed_id.get(pid, None)
        diagram_mode = int(handle.diagram_mode_by_base_context.get(bc, 0) or 0)
        is_diagram = 1 if diagram_mode > 0 or (isinstance(refined, list) and len(refined) > 0) else 0

        meta: Dict[str, Any] = {
            "final_score": final_score,
            "is_diagram": int(is_diagram),
            "diagram": int(is_diagram),
        }
        if is_diagram:
            # Keep it small-ish; Pinecone metadata has limits.
            cleaned: List[str] = []
            for s in refined:
                ss = str(s or "").strip()
                if ss:
                    cleaned.append(ss[:64])
                if len(cleaned) >= 60:
                    break
            meta["labels"] = cleaned

        jobs.append(
            {
                "processed_id": pid,
                "unique_path": src,
                "base_context": bc,
                "prompt_embedding": pvec,
                "clip_embedding": cvec,
                "context_embedding": ctx_vec,
                "meta": meta,
            }
        )

    kept: List[Dict[str, Any]] = []
    for j in jobs:
        if isinstance(j.get("prompt_embedding"), list) or isinstance(j.get("clip_embedding"), list) or isinstance(j.get("context_embedding"), list):
            kept.append(j)

    return kept

def _pinecone_upsert_after_colours(
    *,
    text_items_all: List[Dict[str, Any]],
    meta_ctx_map: Dict[str, Dict[str, Any]],
    workers: PipelineWorkers,
    handle: PipelineHandle,
    out_dir: Path,
) -> None:
    """
    Runs PineconeSave.upsert_image_metadata_embeddings(jobs)
    and writes a small report file.
    """
    try:
        jobs = _build_pinecone_jobs_from_text_items(
            text_items_all=text_items_all,
            meta_ctx_map=meta_ctx_map,
            workers=workers,
            handle=handle,
        )
        if not jobs:
            handle.errors["pinecone"] = "no_jobs_built"
            return

        with stage("PineconeSave.upsert_image_metadata_embeddings"):
            summary = PineconeSave.upsert_image_metadata_embeddings(jobs)

        try:
            (out_dir / "pinecone_upsert_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    except Exception as e:
        handle.errors["pinecone"] = f"{type(e).__name__}: {e}"



def get_hot_qwen() -> None:
    return None

def qwen_lock() -> threading.Lock:
    return _HOT_INIT_LOCK


def unload_hot_qwen() -> None:
    # Backward-compatible no-op: ImagePipeline no longer owns Qwen residency.
    return None


# Optional: expose hot SigLIP/MiniLM to other scripts in the SAME process
def hot_siglip_bundle() -> Optional[_SiglipBundle]:
    return _HOT_SIGLIP

def hot_minilm_bundle() -> Optional[_MiniLMBundle]:
    return _HOT_MINILM


def _norm_path(p: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def _norm_diagram_mode(v: Any) -> int:
    try:
        d = int(v)
    except Exception:
        return 0
    if d == 1:
        return 1
    if d == 2:
        return 2
    return 0


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

def discover_unique_images(root: Path) -> List[str]:
    if not root.exists():
        return []
    out: List[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMG_EXTS:
            out.append(str(p))
    out.sort()
    return out

def discover_metadata_context_paths(root: Path) -> List[Path]:
    if not root.exists():
        return []
    out = list(root.rglob("image_metadata_context.json"))
    out.sort()
    return out



# =========================
# Pipeline helpers
# =========================
def load_unique_images_once(unique_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Loads the UniqueImages ONCE into numpy arrays (BGR).
    """
    items: List[Dict[str, Any]] = []
    for p in unique_paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[skip] cannot read {p}")
            continue

        # normalize into BGR 3ch; same handling as your other scripts
        if img.ndim == 3 and img.shape[2] == 4:
            b, g, r, a = cv2.split(img)
            rgb = cv2.merge([b, g, r])
            mask = (a == 0)
            if np.any(mask):
                rgb[mask] = [255, 255, 255]
            img = rgb
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:
            pass
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        items.append({"source_path": str(p), "img_bgr": img})
    return items


def _processed_dir_candidates() -> List[Path]:
    return [
        BASE_DIR / "ProccessedImages",
        BASE_DIR / "ProcessedImages",
        BASE_DIR / "PipelineOutputs" / "ProccessedImages",
        BASE_DIR / "PipelineOutputs" / "ProcessedImages",
        Path("ProccessedImages"),
        Path("ProcessedImages"),
        Path("PipelineOutputs") / "ProccessedImages",
        Path("PipelineOutputs") / "ProcessedImages",
    ]


def _resolve_processed_png_path(pid: str) -> Optional[Path]:
    for root in _processed_dir_candidates():
        p = root / f"{pid}.png"
        try:
            if p.is_file():
                return p
        except Exception:
            continue
    return None


def _resolve_processed_json_path(pid: str) -> Optional[Path]:
    for root in _processed_dir_candidates():
        p = root / f"{pid}.json"
        try:
            if p.is_file():
                return p
        except Exception:
            continue
    return None


def load_processed_items_once(
    *,
    accepted_processed_ids_by_base_context: Optional[Dict[str, List[str]]] = None,
    allowed_base_contexts: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str]]:
    items: List[Dict[str, Any]] = []
    unique_to_processed: Dict[str, str] = {}
    processed_to_unique: Dict[str, str] = {}

    allowed_set = {
        str(x or "").strip()
        for x in (allowed_base_contexts or [])
        if str(x or "").strip()
    }

    candidate_pairs: List[Tuple[str, str]] = []
    seen_pairs = set()
    if isinstance(accepted_processed_ids_by_base_context, dict) and accepted_processed_ids_by_base_context:
        for bc, ids in accepted_processed_ids_by_base_context.items():
            base_context = str(bc or "").strip()
            if allowed_set and base_context and base_context not in allowed_set:
                continue
            if not isinstance(ids, list):
                continue
            for pid_raw in ids:
                pid = str(pid_raw or "").strip()
                if not pid:
                    continue
                key = (base_context, pid)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                candidate_pairs.append((base_context, pid))

    if not candidate_pairs:
        seen_pid = set()
        for root in _processed_dir_candidates():
            try:
                if not root.is_dir():
                    continue
            except Exception:
                continue
            for json_path in sorted(root.glob("processed_*.json")):
                pid = json_path.stem
                if pid in seen_pid:
                    continue
                try:
                    payload = json.loads(json_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                base_context = str((payload or {}).get("base_context", "") or "").strip()
                if allowed_set and base_context not in allowed_set:
                    continue
                seen_pid.add(pid)
                candidate_pairs.append((base_context, pid))

    for base_context_hint, pid in candidate_pairs:
        png_path = _resolve_processed_png_path(pid)
        json_path = _resolve_processed_json_path(pid)
        if png_path is None or json_path is None:
            continue

        img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[skip] cannot read processed image {png_path}")
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        try:
            payload = json.loads(json_path.read_text(encoding="utf-8")) or {}
        except Exception:
            payload = {}

        base_context = str(payload.get("base_context", "") or base_context_hint or "").strip()
        if allowed_set and base_context not in allowed_set:
            continue

        try:
            idx = int(str(pid).split("_")[-1])
        except Exception:
            idx = len(items)

        source_path = str(payload.get("source_path", "") or "").strip()
        if source_path:
            unique_to_processed[source_path] = pid
            processed_to_unique[pid] = source_path

        items.append(
            {
                "idx": int(idx),
                "source_path": source_path,
                "base_context": base_context,
                "masked_bgr": img,
                "payload_json": payload if isinstance(payload, dict) else {},
                "processed_id": pid,
            }
        )

    items.sort(key=lambda x: int(x.get("idx", 0) or 0))
    return items, unique_to_processed, processed_to_unique


def extract_candidate_words_from_text_payload(payload_json: dict) -> List[str]:
    """
    Extract unique OCR candidate words from a processed image payload.
    """
    words = payload_json.get("words") or []
    out: List[str] = []
    seen = set()
    for w in words:
        t = w.get("text") if isinstance(w, dict) else None
        if not isinstance(t, str):
            continue
        tt = t.strip()
        if not tt:
            continue
        k = tt.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(tt)
    return out


# -------------------------
# ADDED: load image_metadata_context.json robustly
# -------------------------
def load_image_metadata_context_map() -> Dict[str, Dict[str, Any]]:
    paths = discover_metadata_context_paths(IN_DIR_ROOT)
    if not paths:
        print(f"[WARN] Missing metadata json under: {IN_DIR_ROOT}")
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    loaded_files = 0
    loaded_entries = 0

    def _add_keys(k: str, v: Dict[str, Any]) -> None:
        kn = _norm_path(k)
        out[kn] = v
        out[os.path.basename(kn).lower()] = v
        try:
            abs_k = _norm_path(os.path.abspath(k))
            out[abs_k] = v
            try:
                rel = _norm_path(os.path.relpath(abs_k, str(IN_DIR_ROOT.resolve()))).lower()
                out[rel] = v
            except Exception:
                pass
        except Exception:
            pass

    for mp in paths:
        try:
            data = json.loads(mp.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"[WARN] bad metadata json: {mp} -> {e}")
            continue

        if isinstance(data, dict):
            for k, v in data.items():
                if not isinstance(v, dict):
                    continue
                _add_keys(k, v)
                loaded_entries += 1

        loaded_files += 1

    print(f"[META] loaded metadata files: {loaded_files} entries: {loaded_entries} keys_in_map: {len(out)}")
    return out



def _pick_best_context_embedding(meta_entry: Dict[str, Any]) -> Optional[List[float]]:
    ctxs = meta_entry.get("contexts") or []
    if not isinstance(ctxs, list):
        return None

    best_vec = None
    best_score = -1e9
    for c in ctxs:
        if not isinstance(c, dict):
            continue
        vec = c.get("ctx_embedding")
        if not (isinstance(vec, list) and vec and all(isinstance(x, (int, float)) for x in vec)):
            continue
        # prefer ctx_confidence, then ctx_score
        score = c.get("ctx_confidence", None)
        if score is None:
            score = c.get("ctx_score", 0.0)
        try:
            s = float(score)
        except Exception:
            s = 0.0
        if s > best_score:
            best_score = s
            best_vec = vec

    return best_vec


def _pick_clip_embedding(meta_entry: Dict[str, Any]) -> Optional[List[float]]:
    vec = meta_entry.get("clip_embedding")
    if isinstance(vec, list) and vec and all(isinstance(x, (int, float)) for x in vec):
        return vec
    return None


def _pick_prompt_embedding(meta_entry: Dict[str, Any]) -> Optional[List[float]]:
    vec = meta_entry.get("prompt_embedding")
    if isinstance(vec, list) and vec and all(isinstance(x, (int, float)) for x in vec):
        return vec
    return None


def _meta_source_text(meta_entry: Optional[Dict[str, Any]]) -> str:
    if not isinstance(meta_entry, dict):
        return ""
    bits: List[str] = []
    for key in ("source", "source_name", "source_kind"):
        value = meta_entry.get(key)
        if isinstance(value, str) and value.strip():
            bits.append(value.strip().lower())
    return " ".join(bits)


def _is_generated_or_comfy_meta(meta_entry: Optional[Dict[str, Any]]) -> bool:
    txt = _meta_source_text(meta_entry)
    return ("comfy" in txt) or ("generated" in txt)


def _resolve_meta_for_source(meta_map: Dict[str, Dict[str, Any]], source_path: str) -> Optional[Dict[str, Any]]:
    sp = str(source_path)

    abs_sp = ""
    rel_sp = ""
    try:
        abs_sp = _norm_path(os.path.abspath(sp))
        try:
            rel_sp = _norm_path(os.path.relpath(abs_sp, str(IN_DIR_ROOT.resolve()))).lower()
        except Exception:
            rel_sp = ""
    except Exception:
        abs_sp = _norm_path(sp)

    keys = [
        _norm_path(sp),
        abs_sp,
        os.path.basename(abs_sp).lower(),
    ]
    if rel_sp:
        keys.append(rel_sp)

    for k in keys:
        if k in meta_map:
            return meta_map[k]
    return None



# =========================
# Async handle
# =========================

@dataclass
class PipelineHandle:
    thread: threading.Thread
    colours_done: threading.Event
    qwen_done: threading.Event
    pipeline_done: threading.Event

    selection_done: threading.Event
    pinecone_done: threading.Event

    out_dir: Path
    errors: Dict[str, str] = field(default_factory=dict)

    unique_paths: List[str] = field(default_factory=list)
    unique_to_processed: Dict[str, str] = field(default_factory=dict)
    processed_to_unique: Dict[str, str] = field(default_factory=dict)

    selected_processed_ids: List[str] = field(default_factory=list)
    selected_ids_by_base_context: Dict[str, List[str]] = field(default_factory=dict)

    refined_labels_dir: Path = field(default_factory=lambda: Path("PipelineOutputs") / "_refined_labels")
    refined_labels_by_processed_id: Dict[str, List[str]] = field(default_factory=dict)

    diagram_mode_by_base_context: Dict[str, int] = field(default_factory=dict)
    diagram_required_objects_by_base_context: Dict[str, List[str]] = field(default_factory=dict)


    def wait_colours(self, timeout: Optional[float] = None) -> bool:
        return self.colours_done.wait(timeout)

    def wait_qwen(self, timeout: Optional[float] = None) -> bool:
        return self.qwen_done.wait(timeout)

    def wait_selection(self, timeout: Optional[float] = None) -> bool:
        return self.selection_done.wait(timeout)

    def join(self, timeout: Optional[float] = None) -> None:
        self.thread.join(timeout)


def _bgr_to_pil_fast(mbgr: np.ndarray, longest_side: int = 600):
    import cv2
    from PIL import Image

    h, w = mbgr.shape[:2]
    s = max(h, w)
    if s > longest_side:
        scale = float(longest_side) / float(s)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        mbgr = cv2.resize(mbgr, (nw, nh), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(mbgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)



# =========================
# Core pipeline (runs inside background thread)
# =========================
def _pipeline_worker(
    *,
    handle: PipelineHandle,
    workers: PipelineWorkers,
    model_id: str,
    debug_save: bool,
    parallel_cpu: bool,
    gpu_index: int,
    allowed_base_contexts: Optional[List[str]] = None,
    diagram_base_contexts: Optional[List[str]] = None,
    diagram_mode_by_base_context: Optional[Dict[str, int]] = None,
    diagram_required_objects_by_base_context: Optional[Dict[str, List[str]]] = None,
    accepted_processed_ids_by_base_context: Optional[Dict[str, List[str]]] = None,
) -> None:

    out_dir = handle.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # normalized diagram modes by base context:
    # 0 = bypass pipeline stages, 1 = visual diagram, 2 = schematic diagram
    diagram_mode_map: Dict[str, int] = {}
    if isinstance(diagram_mode_by_base_context, dict):
        for k, v in diagram_mode_by_base_context.items():
            kk = str(k or "").strip()
            if not kk:
                continue
            diagram_mode_map[kk] = _norm_diagram_mode(v)

    if diagram_base_contexts:
        # Backward-compatible fallback for callers that only pass a diagram context list.
        for x in diagram_base_contexts:
            kk = str(x or "").strip()
            if not kk:
                continue
            diagram_mode_map.setdefault(kk, 1)

    diagram_set: set[str] = {k for k, v in diagram_mode_map.items() if int(v) > 0}

    req_map: Dict[str, List[str]] = {}
    if isinstance(diagram_required_objects_by_base_context, dict):
        for k, v in diagram_required_objects_by_base_context.items():
            kk = str(k or "").strip()
            if not kk:
                continue
            if not isinstance(v, list):
                continue
            vv = [str(x).strip() for x in v if str(x).strip()]
            if vv:
                req_map[kk] = vv

    try:
        handle.diagram_mode_by_base_context = dict(diagram_mode_map)
    except Exception:
        pass

    try:
        handle.diagram_required_objects_by_base_context = dict(req_map)
    except Exception:
        pass

    # refined labels output folder
    refined_dir = out_dir / "_refined_labels"
    handle.refined_labels_dir = refined_dir
    refined_dir.mkdir(parents=True, exist_ok=True)

    try:
        with stage("load_image_metadata_context_map"):
            meta_ctx_map = load_image_metadata_context_map()

        print("[2/10] Load accepted processed images...")
        with stage("load_processed_items_once"):
            text_items, unique_to_processed, processed_to_unique = load_processed_items_once(
                accepted_processed_ids_by_base_context=accepted_processed_ids_by_base_context,
                allowed_base_contexts=allowed_base_contexts,
            )

        if not text_items:
            handle.errors["pipeline"] = "no_processed_items_loaded"
            return

        handle.unique_paths = [
            str(x.get("source_path", "") or "")
            for x in text_items
            if str(x.get("source_path", "") or "").strip()
        ]
        handle.unique_to_processed = dict(unique_to_processed)
        handle.processed_to_unique = dict(processed_to_unique)
        try:
            (out_dir / "unique_paths.json").write_text(
                json.dumps(handle.unique_paths, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            (out_dir / "unique_to_processed.json").write_text(
                json.dumps(handle.unique_to_processed, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            (out_dir / "processed_to_unique.json").write_text(
                json.dumps(handle.processed_to_unique, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception:
            pass

        print("[3/10] Accept processed images without internal selection...")
        selected_ids_all: List[str] = []
        selection_payload_by_ctx: Dict[str, Any] = {}
        selected_ids_by_ctx: Dict[str, List[str]] = {}
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for t in text_items:
            bc = str(t.get("base_context", "") or "").strip()
            groups.setdefault(bc, []).append(t)

        for bc in sorted(groups.keys(), key=lambda s: (s == "", s.lower())):
            picked_passthrough: List[str] = []
            for t in groups.get(bc) or []:
                pid = str(t.get("processed_id", "") or "").strip()
                if pid and pid not in picked_passthrough:
                    picked_passthrough.append(pid)
            selected_ids_by_ctx[bc] = picked_passthrough
            selection_payload_by_ctx[bc] = {
                "selection_mode": "accepted_processed_ids_passthrough",
                "count_out": len(picked_passthrough),
            }
            for pid in picked_passthrough:
                if pid and pid not in selected_ids_all:
                    selected_ids_all.append(pid)

        handle.selected_processed_ids = list(selected_ids_all)
        handle.selected_ids_by_base_context = dict(selected_ids_by_ctx)
        try:
            print(
                "[dbg][selection] contexts_total=%d diagram_contexts=%d selected_total=%d"
                % (len(selected_ids_by_ctx), len(diagram_set), len(selected_ids_all))
            )
        except Exception:
            pass

        try:
            (out_dir / "selection.json").write_text(
                json.dumps(
                    {
                        "selected_processed_ids": handle.selected_processed_ids,
                        "selected_ids_by_base_context": selected_ids_by_ctx,
                        "selection_payload_by_base_context": selection_payload_by_ctx,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

        handle.refined_labels_by_processed_id = {}

        # Signal: accepted processed ids are ready. Diagram components come from the external C2 pipeline.
        handle.selection_done.set()
        handle.qwen_done.set()

        try:
            import gc
            gc.collect()
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # ---------------------------------------------------------
        # Continue heavy processing for ALL selected prompts.
        # Clustering remains diagram-only later.
        # ---------------------------------------------------------
        selected_keep: set[str] = set()
        visual_diagram_keep: set[str] = set()
        schematic_diagram_keep: set[str] = set()
        for bc, picked in selected_ids_by_ctx.items():
            mode = int(diagram_mode_map.get(bc, 0) or 0)
            for pid in picked:
                if pid:
                    selected_keep.add(pid)
                    if mode == 1:
                        visual_diagram_keep.add(pid)
                    elif mode == 2:
                        schematic_diagram_keep.add(pid)
        try:
            print(
                "[dbg][diagram-modes] selected_keep=%d visual_keep=%d schematic_keep=%d"
                % (len(selected_keep), len(visual_diagram_keep), len(schematic_diagram_keep))
            )
        except Exception:
            pass

        # Keep Pinecone payload aligned with selected outputs used downstream.
        text_items_all_for_pinecone = [
            t for t in text_items
            if str(t.get("processed_id", "") or "").strip() in selected_keep
        ]
        selected_text_items = list(text_items_all_for_pinecone)

        # Diagram items still control later labeling/clustering, but colour/vector JSONs
        # should exist for every selected image.
        diagram_text_items = [
            t for t in text_items_all_for_pinecone
            if int(diagram_mode_map.get(str(t.get("base_context", "") or "").strip(), 0) or 0) > 0
        ]

        if len(text_items_all_for_pinecone) < 1:
            # Upsert what we have from the externally accepted processed outputs.
            _pinecone_upsert_after_colours(
                text_items_all=text_items_all_for_pinecone,
                meta_ctx_map=meta_ctx_map,
                workers=workers,
                handle=handle,
                out_dir=out_dir,
            )
            handle.pinecone_done.set()

            handle.colours_done.set()
            print("[done] selected_keep empty after filtering.")
            return

        print("[6/10] ImagePreprocessor (in-memory)...")
        with stage("ImagePreprocessor.process_images_in_memory"):
            preproc_by_idx = ImagePreprocessor.process_images_in_memory(
                text_items=selected_text_items,
                save_outputs=debug_save,
                parallel=parallel_cpu,
            )

        print("[7/10] ImageSkeletonizer (in-memory)...")
        with stage("ImageSkeletonizer.skeletonize_in_memory"):
            skel_by_idx = ImageSkeletonizer.skeletonize_in_memory(
                preproc_by_idx=preproc_by_idx,
                save_outputs=debug_save,
                parallel=parallel_cpu,
            )

        print("[8/10] ImageVectorizer (in-memory)...")
        with stage("ImageVectorizer.vectorize_in_memory"):
            vectors_by_idx = ImageVectorizer.vectorize_in_memory(
                skel_by_idx=skel_by_idx,
                save_outputs=True,
                parallel=parallel_cpu,
            )

        # ---------------------------------------------------------
        # Cluster labeling is no longer run inside ImagePipeline.
        # Pipeline returns after colours are done.
        # ---------------------------------------------------------

        def _run_colours() -> None:
            try:
                print("[9/10] ImageColours (async)...")
                ImageColours.apply_colours_in_memory(
                    preproc_by_idx=preproc_by_idx,
                    vectors_by_idx=vectors_by_idx,
                    save_outputs=True,
                )
            except BaseException as e:
                handle.errors["colours"] = f"{type(e).__name__}: {e}"
                print("[bg][ERR] ImageColours thread crashed:")
                traceback.print_exc()
            finally:
                handle.colours_done.set()
                print("[bg] ImageColours done. You can work with coloured outputs now.")

        colours_thread = threading.Thread(target=_run_colours, name="image_colours", daemon=False)
        colours_thread.start()
        colours_thread.join()

        print("Colours Done.")

            # Pinecone upsert happens AFTER colours done (your requested placement)
        _pinecone_upsert_after_colours(
            text_items_all=text_items_all_for_pinecone,
            meta_ctx_map=meta_ctx_map,
            workers=workers,
            handle=handle,
            out_dir=out_dir,
        )
        handle.pinecone_done.set()

        diagram_keep: set[str] = set(visual_diagram_keep) | set(schematic_diagram_keep)

        if not diagram_keep:
            handle.colours_done.set()
            print("[done] no diagram prompts -> finished after colours.")
            return

        # Every accepted diagram now produces stroke-level descriptors so the
        # downstream matcher can reason from the same substrate regardless of
        # the original diagram mode.
        diagram_indices: set[int] = set()
        for t in selected_text_items:
            pid = str(t.get("processed_id", "") or "").strip()
            if pid not in diagram_keep:
                continue
            try:
                diagram_indices.add(int(t.get("idx")))
            except Exception:
                continue

        vec_json_paths: List[Path] = []
        for idx in sorted(diagram_indices):
            p = BASE_DIR / "StrokeVectors" / f"processed_{idx}.json"
            if p.is_file():
                vec_json_paths.append(p)

        if vec_json_paths:
            print(f"[10/10] LineDescriptors (all diagrams) for {len(vec_json_paths)} files...")
            with stage("LineDescriptors.describe_vectorizer_jsons_batch"):
                desc_summary = LineDescriptors.describe_vectorizer_jsons_batch(
                    vec_json_paths,
                    output_dir=BASE_DIR / "StrokeDescriptions",
                    max_workers=min(8, max(1, len(vec_json_paths))),
                )
            try:
                (out_dir / "line_descriptors_summary.json").write_text(
                    json.dumps(desc_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
            if desc_summary.get("errors"):
                handle.errors["line_descriptors"] = json.dumps(desc_summary.get("errors", [])[:5], ensure_ascii=False)
        else:
            handle.errors["line_descriptors"] = "no_strokevectors_for_diagram_indices"

        # Every accepted diagram also produces clusters so the final matcher can
        # choose between ready-made groups and stroke-level regrouping.
        diagram_text_items = [
            t for t in selected_text_items
            if str(t.get("processed_id", "") or "").strip() in diagram_keep
        ]
        diagram_preproc_by_idx = {idx: preproc_by_idx[idx] for idx in diagram_indices if idx in preproc_by_idx}
        diagram_vectors_by_idx = {idx: vectors_by_idx[idx] for idx in diagram_indices if idx in vectors_by_idx}

        try:
            print(
                "[dbg][clusters] diagram_text_items=%d diagram_indices=%d preproc_hits=%d vector_hits=%d"
                % (
                    len(diagram_text_items),
                    len(diagram_indices),
                    len(diagram_preproc_by_idx),
                    len(diagram_vectors_by_idx),
                )
            )
        except Exception:
            pass
        if not diagram_text_items or not diagram_preproc_by_idx or not diagram_vectors_by_idx:
            try:
                pre_keys = set(int(k) for k in preproc_by_idx.keys())
                vec_keys = set(int(k) for k in vectors_by_idx.keys())
                miss_pre = sorted(diagram_indices - pre_keys)
                miss_vec = sorted(diagram_indices - vec_keys)
                print(
                    "[dbg][clusters] empty_diagram_input_details missing_preproc=%s missing_vectors=%s"
                    % (miss_pre[:10], miss_vec[:10])
                )
            except Exception:
                pass
            handle.colours_done.set()
            print("[done] no diagram items available for clusters -> finished after colours.")
            return

        print("[11/11] ImageClusters (all diagrams, in-memory)...")
        with stage("ImageClusters.cluster_in_memory"):
            ImageClusters.cluster_in_memory(
                preproc_by_idx=diagram_preproc_by_idx,
                vectors_by_idx=diagram_vectors_by_idx,
                save_outputs=True,
            )

    except BaseException as e:
        handle.errors["pipeline_fatal"] = f"{type(e).__name__}: {e}"
        print("[bg][ERR] pipeline fatal crash:")
        traceback.print_exc()
    finally:
        handle.selection_done.set()
        handle.qwen_done.set()
        handle.colours_done.set()
        handle.pinecone_done.set()
        handle.pipeline_done.set()
        stage_report()



# =========================
# Public API: start in background thread
# =========================
def start_pipeline_background(
    *,
    workers: PipelineWorkers,
    model_id: str = "",
    debug_save: bool = False,
    parallel_cpu: bool = True,
    gpu_index: int = 0,
    allowed_base_contexts: Optional[List[str]] = None,
    diagram_base_contexts: Optional[List[str]] = None,
    diagram_mode_by_base_context: Optional[Dict[str, int]] = None,
    diagram_required_objects_by_base_context: Optional[Dict[str, List[str]]] = None,
    accepted_processed_ids_by_base_context: Optional[Dict[str, List[str]]] = None,
) -> PipelineHandle:

    if REMOTE_API:
        return run_pipeline_blocking(
            workers=workers,
            model_id=model_id,
            debug_save=debug_save,
            parallel_cpu=parallel_cpu,
            gpu_index=gpu_index,
            allowed_base_contexts=allowed_base_contexts,
            diagram_base_contexts=diagram_base_contexts,
            diagram_mode_by_base_context=diagram_mode_by_base_context,
            diagram_required_objects_by_base_context=diagram_required_objects_by_base_context,
            accepted_processed_ids_by_base_context=accepted_processed_ids_by_base_context,
        )

    out_dir = BASE_DIR / "PipelineOutputs"

    colours_done = threading.Event()
    qwen_done = threading.Event()
    pipeline_done = threading.Event()
    selection_done = threading.Event()
    pinecone_done = threading.Event()

    dummy_thread = threading.Thread(target=lambda: None)

    handle = PipelineHandle(
        thread=dummy_thread,
        colours_done=colours_done,
        qwen_done=qwen_done,
        pipeline_done=pipeline_done,
        selection_done=selection_done,
        pinecone_done=pinecone_done,
        out_dir=out_dir,
        errors={},
        diagram_mode_by_base_context=dict(diagram_mode_by_base_context or {}),
        diagram_required_objects_by_base_context=dict(diagram_required_objects_by_base_context or {}),
    )

    # Model loading is owned by the timeline orchestrator now.

    t = threading.Thread(
        target=_pipeline_worker,
        name="pipeline_bg",
        daemon=False,
        kwargs=dict(
            handle=handle,
            workers=workers,
            model_id=model_id,
            debug_save=debug_save,
            parallel_cpu=parallel_cpu,
            gpu_index=gpu_index,
            allowed_base_contexts=allowed_base_contexts,
            diagram_base_contexts=diagram_base_contexts,
            diagram_mode_by_base_context=diagram_mode_by_base_context,
            diagram_required_objects_by_base_context=diagram_required_objects_by_base_context,
            accepted_processed_ids_by_base_context=accepted_processed_ids_by_base_context,
        ),
    )
    handle.thread = t
    t.start()
    return handle


def run_pipeline_blocking(
    *,
    workers: PipelineWorkers,
    model_id: str = "",
    debug_save: bool = False,
    parallel_cpu: bool = True,
    gpu_index: int = 0,
    allowed_base_contexts: Optional[List[str]] = None,
    diagram_base_contexts: Optional[List[str]] = None,
    diagram_mode_by_base_context: Optional[Dict[str, int]] = None,
    diagram_required_objects_by_base_context: Optional[Dict[str, List[str]]] = None,
    accepted_processed_ids_by_base_context: Optional[Dict[str, List[str]]] = None,
) -> PipelineHandle:
    if REMOTE_API:
        out = _http_post_json(
            f"{REMOTE_API}/run_pipeline",
            {
                "model_id": model_id,
                "debug_save": bool(debug_save),
                "parallel_cpu": bool(parallel_cpu),
                "gpu_index": int(gpu_index),
                "allowed_base_contexts": allowed_base_contexts or [],
                "diagram_base_contexts": diagram_base_contexts or [],
                "diagram_mode_by_base_context": diagram_mode_by_base_context or {},
                "diagram_required_objects_by_base_context": diagram_required_objects_by_base_context or {},
                "accepted_processed_ids_by_base_context": accepted_processed_ids_by_base_context or {},
            },
            timeout=3600,
        )

        out_dir = BASE_DIR / "PipelineOutputs"
        h = PipelineHandle(
            thread=threading.Thread(target=lambda: None),
            colours_done=threading.Event(),
            qwen_done=threading.Event(),
            pipeline_done=threading.Event(),
            selection_done=threading.Event(),
            pinecone_done=threading.Event(),
            out_dir=out_dir,
            errors={},
            diagram_mode_by_base_context=out.get("diagram_mode_by_base_context") or {},
            diagram_required_objects_by_base_context=out.get("diagram_required_objects_by_base_context") or {},
        )
        h.colours_done.set()
        h.qwen_done.set()
        h.selection_done.set()
        h.pinecone_done.set()
        h.pipeline_done.set()

        h.errors = out.get("errors") or {}
        h.unique_paths = out.get("unique_paths") or []
        h.unique_to_processed = out.get("unique_to_processed") or {}
        h.processed_to_unique = out.get("processed_to_unique") or {}
        h.selected_processed_ids = out.get("selected_processed_ids") or []
        h.selected_ids_by_base_context = out.get("selected_ids_by_base_context") or {}
        h.refined_labels_by_processed_id = out.get("refined_labels_by_processed_id") or {}
        h.diagram_mode_by_base_context = out.get("diagram_mode_by_base_context") or {}
        return h

    h = start_pipeline_background(
        workers=workers,
        model_id=model_id,
        debug_save=debug_save,
        parallel_cpu=parallel_cpu,
        gpu_index=gpu_index,
        allowed_base_contexts=allowed_base_contexts,
        diagram_base_contexts=diagram_base_contexts,
        diagram_mode_by_base_context=diagram_mode_by_base_context,
        diagram_required_objects_by_base_context=diagram_required_objects_by_base_context,
        accepted_processed_ids_by_base_context=accepted_processed_ids_by_base_context,
    )
    h.pipeline_done.wait()
    return h



def _parse_cli(argv: List[str]) -> Dict[str, Any]:
    args: Dict[str, Any] = {
        "model_id": "",
        "debug_save": False,
        "parallel_cpu": True,
        "gpu_index": 0,
        "blocking": False,
        "detach_after_colours": False,
        "serve": False,
        "host": "127.0.0.1",
        "port": 8787,
    }
    for a in argv[1:]:
        if a.startswith("--model="):
            args["model_id"] = a.split("=", 1)[1]
        elif a == "--debug":
            args["debug_save"] = True
        elif a == "--no-parallel":
            args["parallel_cpu"] = False
        elif a.startswith("--gpu="):
            args["gpu_index"] = int(a.split("=", 1)[1])
        elif a == "--blocking":
            args["blocking"] = True
        elif a == "--detach-after-colours":
            args["detach_after_colours"] = True
        elif a == "--serve":
            args["serve"] = True
        elif a.startswith("--host="):
            args["host"] = a.split("=", 1)[1]
        elif a.startswith("--port="):
            args["port"] = int(a.split("=", 1)[1])
    return args

def get_images_background(
    prompt_or_map: Any,
    subj: Optional[str] = None,
    *,
    workers: PipelineWorkers,
    top_n_per_prompt: int = 2,
    min_final_score: float = 0.78,
    min_modalities: int = 3,
    top_k_per_modality: int = 50,
    model_id: str = "",
    gpu_index: int = 0,
) -> tuple[Dict[str, List[str]], Optional[PipelineHandle], Dict[str, int]]:
    """
    Returns:
      (pinecone_hits, pipeline_handle_or_none, diagram_mode_by_prompt)

    - Pinecone is checked first.
    - If misses exist, this function starts ImagePipeline for those prompts.
    - The caller is responsible for populating ResearchImages/UniqueImages first
      (researcher for diagrams, Comfy for non-diagrams, etc.).
    - Caller waits:
        handle.wait_selection()  -> read handle.selected_ids_by_base_context
        handle.wait_colours()    -> colours/upsert ready
    """
    prompt_to_topic: Dict[str, str] = {}
    diagram_flags: Dict[str, int] = {}

    if isinstance(prompt_or_map, str):
        p = prompt_or_map.strip()
        t = (subj or "").strip()
        if p and t:
            prompt_to_topic[p] = t
            diagram_flags[p] = 0
    elif isinstance(prompt_or_map, dict):
        for k, v in prompt_or_map.items():
            if not isinstance(k, str) or not k.strip():
                continue
            prompt = k.strip()

            if isinstance(v, str):
                topic = v.strip()
                if topic:
                    prompt_to_topic[prompt] = topic
                    diagram_flags[prompt] = 0
                continue

            if isinstance(v, dict):
                topic = str(v.get("topic") or v.get("subj") or v.get("subject") or "").strip()
                if not topic:
                    continue
                prompt_to_topic[prompt] = topic
                d = v.get("diagram", 0)
                diagram_flags[prompt] = _norm_diagram_mode(d)

    if not prompt_to_topic:
        return {}, None, {}

    try:
        PineconeFetch.configure_hot_models(
            siglip_bundle=workers.siglip_bundle,
            minilm_bundle=workers.minilm_bundle,
            clear_siglip=True,
            clear_minilm=True,
        )
    except Exception:
        pass

    results: Dict[str, List[str]] = {}
    misses: Dict[str, str] = {}
    fetch_attempts: List[Tuple[float, int, bool]] = [
        (float(min_final_score), int(min_modalities), True),
        (0.62, 2, True),
    ]
    allow_lax_reuse = str(os.getenv("PINECONE_ALLOW_LAX_REUSE", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if allow_lax_reuse:
        fetch_attempts.append((0.55, 1, False))

    for prompt, topic in prompt_to_topic.items():
        try:
            ids: List[str] = []
            for score_floor, min_mods, require_ctx in fetch_attempts:
                cur = PineconeFetch.fetch_processed_ids_for_prompt(
                    prompt,
                    top_n=top_n_per_prompt,
                    top_k_per_modality=top_k_per_modality,
                    min_modalities=int(min_mods),
                    min_final_score=float(score_floor),
                    require_base_context_match=bool(require_ctx),
                )
                if cur:
                    ids = [str(x) for x in cur if str(x)]
                    break
            if ids:
                results[prompt] = ids
            else:
                misses[prompt] = topic
        except Exception:
            misses[prompt] = topic

    if not misses:
        return results, None, diagram_flags

    diagram_prompts = [p for p in misses.keys() if int(diagram_flags.get(p, 0) or 0) > 0]
    diagram_mode_map = {p: int(diagram_flags.get(p, 0) or 0) for p in misses.keys()}

    h = start_pipeline_background(
        workers=workers,
        model_id=model_id,
        debug_save=False,
        parallel_cpu=True,
        gpu_index=gpu_index,
        allowed_base_contexts=list(misses.keys()),
        diagram_base_contexts=diagram_prompts,
        diagram_mode_by_base_context=diagram_mode_map,
    )
    return results, h, diagram_flags


def get_images(
    prompt_or_map: Any,
    subj: Optional[str] = None,
    *,
    workers: PipelineWorkers,
    top_n_per_prompt: int = 2,
    min_final_score: float = 0.78,
    min_modalities: int = 3,
    top_k_per_modality: int = 50,
    model_id: str = "",
    gpu_index: int = 0,
) -> Dict[str, List[str]]:
    hits, h, diagram_flags = get_images_background(
        prompt_or_map,
        subj=subj,
        workers=workers,
        top_n_per_prompt=top_n_per_prompt,
        min_final_score=min_final_score,
        min_modalities=min_modalities,
        top_k_per_modality=top_k_per_modality,
        model_id=model_id,
        gpu_index=gpu_index,
    )
    if not h:
        return hits

    # wait full finish (old behavior)
    h.pipeline_done.wait()

    # merge selection output
    by_ctx = getattr(h, "selected_ids_by_base_context", {}) or {}
    for prompt, ids in by_ctx.items():
        if isinstance(ids, list) and ids:
            hits[prompt] = [str(x) for x in ids if str(x)]
    return hits


# ============================================
# HOT LOCAL API MODE (one process stays alive)
# ============================================
def serve_hot_api(
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    model_id: str = "",
    gpu_index: int = 0,
) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    ensure_hot_models(
        gpu_index=gpu_index,
        cpu_threads=_PIPE_CPU_THREADS,
        warmup=True,
        load_siglip=True,
        load_minilm=True,
    )

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: Dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True})
                return
            self._send(404, {"ok": False, "err": "not_found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(n) if n > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                self._send(400, {"ok": False, "err": "bad_json"})
                return

            if self.path == "/get_images":
                prompts = body.get("prompts") or {}
                subj = body.get("subj", None)
                try:
                    out = get_images(
                        prompts if isinstance(prompts, dict) else prompts,
                        subj=subj if isinstance(subj, str) else None,
                        top_n_per_prompt=int(body.get("top_n_per_prompt", 2) or 2),
                        min_final_score=float(body.get("min_final_score", 0.78) or 0.78),
                        min_modalities=int(body.get("min_modalities", 3) or 3),
                        top_k_per_modality=int(body.get("top_k_per_modality", 50) or 50),
                        model_id=model_id,
                        gpu_index=gpu_index,
                    )
                    self._send(200, {"ok": True, "result": out})
                except Exception as e:
                    self._send(500, {"ok": False, "err": repr(e)})
                return

            self._send(404, {"ok": False, "err": "not_found"})

        def log_message(self, format, *args):
            # keep console clean
            return

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"[hot_api] listening on http://{host}:{port}  (SigLIP/MiniLM hot)")
    httpd.serve_forever()


def _hardcoded_get_images_test() -> int:
    # HARD TEST INPUTS
    prompts = {
        "Eukaryotic Cell": "Cell biology",
    }

    print("[TEST] get_images hardcoded run")
    print("[TEST] prompts:", list(prompts.keys()))

    try:
        shared_models.init_siglip_hot(gpu_index=0, cpu_threads=_PIPE_CPU_THREADS, warmup=True)
        shared_models.init_minilm_hot(gpu_index=0, cpu_threads=_PIPE_CPU_THREADS, warmup=True)
        siglip = shared_models.get_siglip()
        minilm = shared_models.get_minilm()
        workers = PipelineWorkers(
            siglip_bundle=siglip,
            minilm_bundle=minilm,
        )
    except Exception as e:
        print("[TEST][ERR] shared_models init failed:", repr(e))
        return 2

    try:
        out = get_images(
            prompts,
            workers=workers,
            top_n_per_prompt=2,
            min_final_score=0.78,
            min_modalities=3,
            top_k_per_modality=50,
            model_id="",
            gpu_index=0,
        )
    except Exception as e:
        print("[TEST][ERR] get_images crashed:", repr(e))
        return 2

    print("\n[TEST] RESULTS (processed_n only):")
    missing = []
    for p in prompts.keys():
        ids = out.get(p) or []
        print(f"  - {p}: {ids}")
        if not ids:
            missing.append(p)

    if missing:
        print("\n[TEST][FAIL] Missing processed ids for:", missing)
        return 1

    print("\n[TEST][OK] All prompts returned processed ids.")
    return 0


if __name__ == "__main__":
    args = _parse_cli(sys.argv)

    # Persistent hot service mode (keeps models loaded and reused)
    if args.get("serve"):
        serve_hot_api(
            host=str(args.get("host") or "127.0.0.1"),
            port=int(args.get("port") or 8787),
            model_id=str(args.get("model_id") or ""),
            gpu_index=int(args.get("gpu_index") or 0),
        )
        raise SystemExit(0)

    # Default behavior: run test flow
    raise SystemExit(_hardcoded_get_images_test())
