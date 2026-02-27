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
from typing import Any, Dict, List, Optional
from collections import Counter, defaultdict
from contextlib import contextmanager

import cv2
import numpy as np

# clamp OpenCV internal threads (avoids “random” oversubscription stalls)
try:
    cv2.setNumThreads(0)
except Exception:
    pass

import ImageText
import ImagePreprocessor
import ImageSkeletonizer
import ImageVectorizer
import ImageClusters
import ImageColours

import PineconeSave
import PineconeFetch

import ImageResearcher
import shared_models

from typing import Any, Dict, List, Optional

@dataclass
class PipelineWorkers:
    # Qwen used for selection + refinement only
    qwen_model: Any
    qwen_processor: Any
    qwen_device: Any
    qwen_lock: threading.Lock

    # SigLIP + MiniLM are used by PineconeFetch (configured externally but passed through here)
    siglip_bundle: Any = None
    minilm_bundle: Any = None


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
# HOT MODEL REGISTRY (Qwen + SigLIP + MiniLM)
# Keep models loaded once per process
# ============================================
@dataclass
class _QwenBundle:
    model: Any
    processor: Any
    device: Any
    model_id: str
    gpu_index: int

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
_QWEN_LOCK = threading.Lock()
_SIGLIP_LOCK = threading.Lock()
_MINILM_LOCK = threading.Lock()

_HOT_QWEN: Optional[_QwenBundle] = None
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
    qwen_model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    siglip_model_id: str = "google/siglip-so400m-patch14-384",
    minilm_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    gpu_index: int = 0,
    cpu_threads: int = _PIPE_CPU_THREADS,
    warmup: bool = True,
    load_siglip: bool = True,
    load_minilm: bool = True,
) -> None:
    """
    Loads Qwen/SigLIP/MiniLM exactly once per PROCESS and keeps them resident.
    If you run this script as a persistent service (--serve), all calls reuse the same hot models.
    """
    global _HOT_QWEN, _HOT_SIGLIP, _HOT_MINILM

    _torch_post_config(cpu_threads)

    with _HOT_INIT_LOCK:
        # ---- QWEN ----
        if _HOT_QWEN is None or _HOT_QWEN.model_id != qwen_model_id or _HOT_QWEN.gpu_index != gpu_index:
            _log(f"Loading Qwen (one-time): {qwen_model_id} gpu_index={gpu_index}")
            try:
                import qwentest  # local module
                model, processor, device = qwentest.preload_qwen_cpu_only(qwen_model_id)

                # Move ONCE to CUDA if available and keep it there
                try:
                    model, device = qwentest.move_qwen_to_cuda_if_available(model, gpu_index=gpu_index)
                except Exception as e:
                    _log(f"[WARN] Qwen CUDA move failed; staying on CPU: {e}")

                try:
                    model.eval()
                except Exception:
                    pass

                if warmup:
                    try:
                        qwentest.warmup_qwen_once(model, processor, device)
                    except Exception as e:
                        _log(f"[WARN] Qwen warmup failed: {e}")

                _HOT_QWEN = _QwenBundle(model=model, processor=processor, device=device, model_id=qwen_model_id, gpu_index=gpu_index)
                _log("Qwen is hot.")
            except Exception as e:
                _HOT_QWEN = None
                _log(f"[ERR] Qwen load failed: {repr(e)}")
                traceback.print_exc()

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
            PineconeFetch.configure_hot_models(siglip_bundle=_HOT_SIGLIP, minilm_bundle=_HOT_MINILM)
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

    Diagram-only label saving:
      - If pid is in handle.refined_labels_by_processed_id => is_diagram=1 and meta["labels"]=[...]
      - Otherwise is_diagram=0 and NO labels field is written.
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

        ctx_vec = None
        final_score = None
        try:
            if src:
                meta_entry = _resolve_meta_for_source(meta_ctx_map, src)
                if meta_entry:
                    final_score = meta_entry.get("final_score", None)
                    ctx_vec = _pick_best_context_embedding(meta_entry)
        except Exception:
            ctx_vec = None

        if ctx_vec is None:
            ctx_vec = pvec

        # -------------------------
        # Diagram label + tag
        # -------------------------
        refined = handle.refined_labels_by_processed_id.get(pid, None)
        is_diagram = 1 if isinstance(refined, list) and len(refined) > 0 else 0

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



def get_hot_qwen() -> Optional[_QwenBundle]:
    return _HOT_QWEN

def qwen_lock() -> threading.Lock:
    return _QWEN_LOCK


# Optional: expose hot SigLIP/MiniLM to other scripts in the SAME process
def hot_siglip_bundle() -> Optional[_SiglipBundle]:
    return _HOT_SIGLIP

def hot_minilm_bundle() -> Optional[_MiniLMBundle]:
    return _HOT_MINILM


def _norm_path(p: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


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


def extract_candidate_words_from_text_payload(payload_json: dict) -> List[str]:
    """
    qwentest previously got its candidates from processed_<idx>.json.
    Now we build them directly from the in-memory OCR payload.
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
    diagram_required_objects_by_base_context: Optional[Dict[str, List[str]]] = None,
) -> None:

    out_dir = handle.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # normalized set for diagram prompts
    diagram_set: set[str] = set()
    if diagram_base_contexts:
        diagram_set = {str(x or "").strip() for x in diagram_base_contexts if str(x or "").strip()}

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
        handle.diagram_required_objects_by_base_context = dict(req_map)
    except Exception:
        pass

    # refined labels output folder
    refined_dir = out_dir / "_refined_labels"
    handle.refined_labels_dir = refined_dir
    refined_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Ensure models are hot inside worker too (safe no-op if already loaded)
        # (model loading is owned by the timeline orchestrator now)

        with stage("discover_unique_images"):
            unique_paths = discover_unique_images(IN_DIR_ROOT)

        handle.unique_paths = [str(p) for p in unique_paths]
        try:
            (out_dir / "unique_paths.json").write_text(
                json.dumps(handle.unique_paths, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception:
            pass

        print("[2/10] Load ImageText base_context map + metadata map...")
        with stage("ImageText.load_base_context_map"):
            base_context_map = ImageText.load_base_context_map()

        with stage("load_image_metadata_context_map"):
            meta_ctx_map = load_image_metadata_context_map()

        allowed_set: Optional[set[str]] = None
        if allowed_base_contexts:
            allowed_set = {
                str(x or "").strip()
                for x in allowed_base_contexts
                if str(x or "").strip()
            }

        path_base_context: Dict[str, str] = {}
        with stage("assign_base_context_for_paths"):
            for sp in unique_paths:
                sp_norm = _norm_path(sp)
                bc = base_context_map.get(sp_norm, "")

                if not bc:
                    meta_entry = _resolve_meta_for_source(meta_ctx_map, sp)
                    if meta_entry and isinstance(meta_entry.get("base_context"), str):
                        bc = meta_entry["base_context"].strip()

                path_base_context[sp] = (bc or "").strip()

        try:
            dist = Counter(str(bc or "").strip() for bc in path_base_context.values())
            print("[ctx] base_context distribution:", dict(dist))
        except Exception:
            pass

        if allowed_set is not None:
            before = len(unique_paths)
            unique_paths = [sp for sp in unique_paths if str(path_base_context.get(sp, "") or "").strip() in allowed_set]
            print(f"[filter] base_context allowed={len(allowed_set)} images {before} -> {len(unique_paths)}")
            if not unique_paths:
                handle.errors["pipeline"] = "no_images_after_base_context_filter"
                return

        print("[3/10] Load UniqueImages ONCE -> numpy...")
        with stage("load_unique_images_once"):
            img_items = load_unique_images_once(unique_paths)

        if not img_items:
            handle.errors["pipeline"] = "no_images_loaded"
            return

        for it in img_items:
            it["base_context"] = path_base_context.get(it["source_path"], "")

        print("[4/10] ImageText (in-memory)...")
        with stage("ImageText.process_images_in_memory"):
            text_items, unique_to_processed = ImageText.process_images_in_memory(
                image_items=img_items,
                start_index=0,
                save_outputs=True,
                return_path_map=True,
            )
        if not text_items:
            handle.errors["pipeline"] = "imagetext_no_outputs"
            return
        
        text_items_all_for_pinecone = list(text_items)

        handle.unique_to_processed = dict(unique_to_processed)
        handle.processed_to_unique = {v: k for k, v in handle.unique_to_processed.items()}
        try:
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

        # =========================
        # Qwen selection step (EXTERNAL worker)
        # =========================
        print("[5/10] qwentest selection (processed images, external worker)...")
        import qwentest

        qwen_model = workers.qwen_model
        qwen_processor = workers.qwen_processor
        qwen_device = workers.qwen_device

        selected_ids_all: List[str] = []
        selection_payload_by_ctx: Dict[str, Any] = {}
        selected_ids_by_ctx: Dict[str, List[str]] = {}

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for t in text_items:
            bc = str(t.get("base_context", "") or "").strip()
            groups.setdefault(bc, []).append(t)

        group_keys = sorted(groups.keys(), key=lambda s: (s == "", s.lower()))

        with stage("Qwen.selection.total"):
            for bc in group_keys:
                items = groups.get(bc) or []
                if not items:
                    continue

                processed_items_for_qwen: List[Dict[str, Any]] = []
                for t in items:
                    pid = str(t.get("processed_id", "") or "").strip()
                    mbgr = t.get("masked_bgr")
                    if not pid or mbgr is None:
                        continue
                    try:
                        pil = _bgr_to_pil_fast(mbgr, longest_side=600)

                        src = str(t.get("source_path", "") or "")
                        if not src:
                            src = str(handle.processed_to_unique.get(pid, "") or "")

                        final_score = None
                        try:
                            if src:
                                meta_entry = _resolve_meta_for_source(meta_ctx_map, src)
                                if meta_entry:
                                    final_score = meta_entry.get("final_score", None)
                        except Exception:
                            final_score = None

                        processed_items_for_qwen.append({
                            "processed_id": pid,
                            "image_pil": pil,
                            "base_context": bc,
                            "final_score": final_score,
                        })

                    except Exception:
                        continue

                if not processed_items_for_qwen:
                    continue

                picked: List[str] = []
                payload: Dict[str, Any] = {}

                try:
                    with workers.qwen_lock:
                        payload = qwentest.pick_two_processed_candidates_transformers(
                            model=qwen_model,
                            processor=qwen_processor,
                            device=qwen_device,
                            processed_items=processed_items_for_qwen,
                            processed_to_unique=handle.processed_to_unique,
                            base_context=bc,
                            batch_size=8,
                            longest_side=600,
                        )

                    cand = payload.get("candidates") or []
                    if isinstance(cand, list):
                        for c in cand:
                            if isinstance(c, dict):
                                pid = str(c.get("processed_id", "") or "").strip()
                                if pid:
                                    picked.append(pid)

                except BaseException as e:
                    handle.errors[f"qwen_select[{bc}]"] = f"{type(e).__name__}: {e}"
                    print("[bg][ERR] Qwen selection crashed for base_context:", bc)
                    traceback.print_exc()

                if len(picked) < 1:
                    tmp = []
                    for t in items:
                        pid = str(t.get("processed_id", "") or "").strip()
                        if pid:
                            tmp.append(pid)
                        if len(tmp) >= 2:
                            break
                    picked = tmp[:1]

                selection_payload_by_ctx[bc] = payload
                selected_ids_by_ctx[bc] = list(picked)

                for pid in picked:
                    if pid and pid not in selected_ids_all:
                        selected_ids_all.append(pid)

        handle.selected_processed_ids = list(selected_ids_all)
        handle.selected_ids_by_base_context = dict(selected_ids_by_ctx)

        try:
            (out_dir / "qwen_selection.json").write_text(
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

        refined_by_pid: Dict[str, List[str]] = {}

        with stage("Qwen.refine_labels.after_selection"):
            for bc, picked in selected_ids_by_ctx.items():
                if bc not in diagram_set:
                    continue

                for pid in picked[:1]:  # keep consistent with "pick 1"
                    pid = str(pid or "").strip()
                    if not pid:
                        continue

                    # find its text_item to get payload_json words
                    t_match = None
                    for t in text_items:
                        if str(t.get("processed_id", "") or "").strip() == pid:
                            t_match = t
                            break
                    if not t_match:
                        continue

                    raw_candidates_imagetext = extract_candidate_words_from_text_payload(t_match.get("payload_json") or {})
                    raw_candidates_imagetext = raw_candidates_imagetext[:120]

                    required_objs = req_map.get(bc) or []
                    if not isinstance(required_objs, list):
                        required_objs = []
                    required_objs = [str(x).strip() for x in required_objs if str(x).strip()]

                    merged: List[str] = []
                    seen = set()

                    def _push(s: str) -> None:
                        ss = str(s or "").strip()
                        if not ss:
                            return
                        kk = ss.lower()
                        if kk in seen:
                            return
                        seen.add(kk)
                        merged.append(ss)

                    for s in required_objs:
                        _push(s)
                    for s in raw_candidates_imagetext:
                        _push(s)

                    raw_candidates = merged[:80]

                    try:
                        with workers.qwen_lock:
                            refined = qwentest.refine_candidate_labels_with_qwen(
                                model=qwen_model,
                                processor=qwen_processor,
                                device=qwen_device,
                                base_context=bc,
                                raw_candidates=raw_candidates,
                            )
                    except Exception as e:
                        handle.errors[f"qwen_refine[{pid}]"] = f"{type(e).__name__}: {e}"
                        refined = raw_candidates[:40]

                    # force clean list[str]
                    refined_list: List[str] = []
                    if isinstance(refined, list):
                        for s in refined:
                            ss = str(s or "").strip()
                            if ss:
                                refined_list.append(ss)

                    refined_by_pid[pid] = refined_list

                    # store both keys:
                    # - "objects": list[str] (semantic name for downstream qwen actions)
                    # - "refined_labels": list[str] (back-compat / human clarity)
                    try:
                        (refined_dir / f"{pid}.json").write_text(
                            json.dumps(
                                {
                                    "processed_id": pid,
                                    "base_context": bc,
                                    "diagram_required_objects": required_objs,
                                    "raw_candidates_imagetext": raw_candidates_imagetext[:80],
                                    "raw_candidates_merged": raw_candidates,
                                    "objects": refined_list,
                                    "refined_labels": refined_list,
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

        handle.refined_labels_by_processed_id = dict(refined_by_pid)

        # Signal: selection + refine is done (timeline orchestrator can proceed)
        handle.selection_done.set()
        handle.qwen_done.set()

        try:
            workers.qwen_model = None
            workers.qwen_processor = None
            workers.qwen_device = None
        except Exception:
            pass

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
        diagram_keep: set[str] = set()
        for bc, picked in selected_ids_by_ctx.items():
            for pid in picked:
                if pid:
                    selected_keep.add(pid)
                    if bc in diagram_set:
                        diagram_keep.add(pid)

        # Keep Pinecone payload aligned with qwen champions used downstream.
        text_items_all_for_pinecone = [
            t for t in text_items
            if str(t.get("processed_id", "") or "").strip() in selected_keep
        ]

        # Cut down to selected images for all contexts
        text_items = [t for t in text_items if str(t.get("processed_id", "") or "").strip() in selected_keep]
        if len(text_items) < 1:
            # Upsert what we have (selection/refine already produced processed outputs)
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
                text_items=text_items,
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
        # REMOVED: final cluster labeling stage call (qwentest.label_clusters_transformers)
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

        if not diagram_keep:
            # nothing diagram-like: skip clusters only
            handle.colours_done.set()
            print("[done] no diagram prompts -> finished after colours.")
            return

        diagram_text_items = [
            t for t in text_items
            if str(t.get("processed_id", "") or "").strip() in diagram_keep
        ]
        diagram_preproc_by_idx = {
            i: preproc_by_idx[i] for i, t in enumerate(text_items)
            if str(t.get("processed_id", "") or "").strip() in diagram_keep and i in preproc_by_idx
        }
        diagram_vectors_by_idx = {
            i: vectors_by_idx[i] for i, t in enumerate(text_items)
            if str(t.get("processed_id", "") or "").strip() in diagram_keep and i in vectors_by_idx
        }
        if not diagram_text_items or not diagram_preproc_by_idx or not diagram_vectors_by_idx:
            handle.colours_done.set()
            print("[done] no diagram items available for clusters -> finished after colours.")
            return

        print("[10/10] ImageClusters (in-memory)...")
        with stage("ImageClusters.cluster_in_memory"):
            clusters_by_idx = ImageClusters.cluster_in_memory(
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
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    debug_save: bool = False,
    parallel_cpu: bool = True,
    gpu_index: int = 0,
    allowed_base_contexts: Optional[List[str]] = None,
    diagram_base_contexts: Optional[List[str]] = None,
    diagram_required_objects_by_base_context: Optional[Dict[str, List[str]]] = None,
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
            diagram_required_objects_by_base_context=diagram_required_objects_by_base_context,
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
        diagram_required_objects_by_base_context=dict(diagram_required_objects_by_base_context or {}),
    )

    # HOT LOAD ONCE at startup (keeps Qwen/SigLIP/MiniLM resident for the life of the process)\
    # (model loading is owned by the timeline orchestrator now)

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
            diagram_required_objects_by_base_context=diagram_required_objects_by_base_context,
        ),
    )
    handle.thread = t
    t.start()
    return handle


def run_pipeline_blocking(
    *,
    workers: PipelineWorkers,
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    debug_save: bool = False,
    parallel_cpu: bool = True,
    gpu_index: int = 0,
    allowed_base_contexts: Optional[List[str]] = None,
    diagram_base_contexts: Optional[List[str]] = None,
    diagram_required_objects_by_base_context: Optional[Dict[str, List[str]]] = None,
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
                "diagram_required_objects_by_base_context": diagram_required_objects_by_base_context or {},
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
        return h

    h = start_pipeline_background(
        workers=workers,
        model_id=model_id,
        debug_save=debug_save,
        parallel_cpu=parallel_cpu,
        gpu_index=gpu_index,
        allowed_base_contexts=allowed_base_contexts,
        diagram_base_contexts=diagram_base_contexts,
        diagram_required_objects_by_base_context=diagram_required_objects_by_base_context,
    )
    h.pipeline_done.wait()
    return h



def _parse_cli(argv: List[str]) -> Dict[str, Any]:
    args: Dict[str, Any] = {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
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
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    gpu_index: int = 0,
) -> tuple[Dict[str, List[str]], Optional[PipelineHandle], Dict[str, int]]:
    """
    Returns:
      (pinecone_hits, pipeline_handle_or_none, diagram_flags_by_prompt)

    - If misses exist: runs ImageResearcher.research_many(misses) then starts pipeline BG.
    - Caller waits:
        handle.wait_selection()  -> read handle.selected_ids_by_base_context + refined labels
        handle.wait_colours()    -> colours ready (only for diagram prompts)
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
                try:
                    diagram_flags[prompt] = 1 if int(d) == 1 else 0
                except Exception:
                    diagram_flags[prompt] = 0

    if not prompt_to_topic:
        return {}, None, {}

    # configure PineconeFetch with SigLIP/MiniLM workers (owned by orchestrator)
    try:
        PineconeFetch.configure_hot_models(siglip_bundle=workers.siglip_bundle, minilm_bundle=workers.minilm_bundle)
    except Exception:
        pass

    results: Dict[str, List[str]] = {}
    misses: Dict[str, str] = {}

    for prompt, topic in prompt_to_topic.items():
        try:
            ids = PineconeFetch.fetch_processed_ids_for_prompt(
                prompt,
                top_n=top_n_per_prompt,
                top_k_per_modality=top_k_per_modality,
                min_modalities=min_modalities,
                min_final_score=min_final_score,
                require_base_context_match=True,
            )
            if ids:
                results[prompt] = ids
            else:
                misses[prompt] = topic
        except Exception:
            misses[prompt] = topic

    if not misses:
        return results, None, diagram_flags

    # research missing prompts (bundled)
    ImageResearcher.research_many(misses)

    # pipeline only for missing prompts
    diagram_prompts = [p for p in misses.keys() if diagram_flags.get(p, 0) == 1]

    h = start_pipeline_background(
        workers=workers,
        model_id=model_id,
        debug_save=False,
        parallel_cpu=True,
        gpu_index=gpu_index,
        allowed_base_contexts=list(misses.keys()),
        diagram_base_contexts=diagram_prompts,
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
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
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
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    gpu_index: int = 0,
) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    ensure_hot_models(
        qwen_model_id=model_id,
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
    print(f"[hot_api] listening on http://{host}:{port}  (Qwen/SigLIP/MiniLM hot)")
    httpd.serve_forever()


def _hardcoded_get_images_test() -> int:
    # HARD TEST INPUTS
    prompts = {
        "Eukaryotic Cell": "Cell biology",
        "Human Heart": "Anatomy",
    }

    print("[TEST] get_images hardcoded run")
    print("[TEST] prompts:", list(prompts.keys()))

    try:
        shared_models.init_hot_models(
            qwen_model_id="Qwen/Qwen3-VL-2B-Instruct",
            gpu_index=0,
            cpu_threads=_PIPE_CPU_THREADS,
            warmup=True,
        )
        qwen = shared_models.get_qwen()
        siglip = shared_models.get_siglip()
        minilm = shared_models.get_minilm()
        if qwen is None:
            print("[TEST][ERR] qwen model not loaded from shared_models.")
            return 2
        workers = PipelineWorkers(
            qwen_model=qwen.model,
            qwen_processor=qwen.processor,
            qwen_device=qwen.device,
            qwen_lock=shared_models.qwen_lock(),
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
            model_id="Qwen/Qwen3-VL-2B-Instruct",
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
            model_id=str(args.get("model_id") or "Qwen/Qwen3-VL-2B-Instruct"),
            gpu_index=int(args.get("gpu_index") or 0),
        )
        raise SystemExit(0)

    # Default behavior: run test flow
    raise SystemExit(_hardcoded_get_images_test())
