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

    # -------------------------
    # Pinecone event
    # -------------------------
    pinecone_done: threading.Event

    out_dir: Path
    errors: Dict[str, str] = field(default_factory=dict)

    # -------------------------
    # expose these to caller
    # -------------------------
    unique_paths: List[str] = field(default_factory=list)
    unique_to_processed: Dict[str, str] = field(default_factory=dict)
    processed_to_unique: Dict[str, str] = field(default_factory=dict)

    # -------------------------
    #selection result
    # -------------------------
    selected_processed_ids: List[str] = field(default_factory=list)

    selected_ids_by_base_context: Dict[str, List[str]] = field(default_factory=dict)


    def wait_colours(self, timeout: Optional[float] = None) -> bool:
        return self.colours_done.wait(timeout)

    def wait_qwen(self, timeout: Optional[float] = None) -> bool:
        return self.qwen_done.wait(timeout)

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
    model_id: str,
    debug_save: bool,
    parallel_cpu: bool,
    gpu_index: int,
    allowed_base_contexts: Optional[List[str]] = None,
) -> None:

    out_dir = handle.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Ensure models are hot inside worker too (safe no-op if already loaded)
        ensure_hot_models(
            qwen_model_id=model_id,
            gpu_index=gpu_index,
            cpu_threads=_PIPE_CPU_THREADS,
            warmup=False,
            load_siglip=True,
            load_minilm=True,
        )

        with stage("discover_unique_images"):
            unique_paths = discover_unique_images(IN_DIR_ROOT)

        # -------------------------
        # ADDED: expose unique paths to caller + save them
        # -------------------------
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

        # Build allowed set ONCE (don’t do it inside the loop)
        allowed_set: Optional[set[str]] = None
        if allowed_base_contexts:
            allowed_set = {
                str(x or "").strip()
                for x in allowed_base_contexts
                if str(x or "").strip()
            }

        # Assign base_context for ALL paths first (NO image decode yet)
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

        # Optional: debug distribution (this will instantly show if contexts are missing)
        try:
            dist = Counter(str(bc or "").strip() for bc in path_base_context.values())
            print("[ctx] base_context distribution:", dict(dist))
        except Exception:
            pass

        # Filter BEFORE decoding images
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

        # Attach base_context onto items (no extra lookup later)
        for it in img_items:
            it["base_context"] = path_base_context.get(it["source_path"], "")

        print("[4/10] ImageText (in-memory)...")
        # -------------------------
        # CHANGED: ImageText now returns (text_items, unique_to_processed)
        # -------------------------
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

        # -------------------------
        # ADDED: store maps on handle + persist for later scripts
        # -------------------------
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
        # Qwen selection step (HOT model, GPU stays loaded)
        # =========================
        print("[5/10] qwentest selection (processed images, hot GPU, per-prompt)...")
        import qwentest

        qb = get_hot_qwen()
        qwen_model = qb.model if qb else None
        qwen_processor = qb.processor if qb else None
        qwen_device = qb.device if qb else None

        selected_ids_all: List[str] = []
        selection_payload_by_ctx: Dict[str, Any] = {}
        selected_ids_by_ctx: Dict[str, List[str]] = {}

        # group text_items by base_context (prompt)
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for t in text_items:
            bc = str(t.get("base_context", "") or "").strip()
            groups.setdefault(bc, []).append(t)

        # deterministic order
        group_keys = sorted(groups.keys(), key=lambda s: (s == "", s.lower()))

        with stage("Qwen.selection.total"):
            for bc in group_keys:
                items = groups.get(bc) or []
                if not items:
                    continue

                # build processed items for THIS prompt only
                processed_items_for_qwen: List[Dict[str, Any]] = []
                for t in items:
                    pid = str(t.get("processed_id", "") or "").strip()
                    mbgr = t.get("masked_bgr")
                    if not pid or mbgr is None:
                        continue
                    try:
                        pil = _bgr_to_pil_fast(mbgr, longest_side=600)
                        # --- inside: for t in items: (selection step) ---
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
                            "final_score": final_score,   # <-- NEW
                        })

                    except Exception:
                        continue

                if not processed_items_for_qwen:
                    continue

                picked: List[str] = []
                payload: Dict[str, Any] = {}

                if qwen_model is not None:
                    try:
                        # serialize Qwen inference across threads
                        with qwen_lock():
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

                # fallback per group
                if len(picked) < 1:
                    tmp = []
                    for t in items:
                        pid = str(t.get("processed_id", "") or "").strip()
                        if pid:
                            tmp.append(pid)
                        if len(tmp) >= 2:
                            break
                    picked = tmp[:1]

                # store
                selection_payload_by_ctx[bc] = payload
                selected_ids_by_ctx[bc] = list(picked)

                for pid in picked:
                    if pid and pid not in selected_ids_all:
                        selected_ids_all.append(pid)

        handle.selected_processed_ids = list(selected_ids_all)
        handle.selected_ids_by_base_context = dict(selected_ids_by_ctx)

        # Save selection results
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

        # Cut down pipeline to only selected (2 per prompt)
        keep = set(handle.selected_processed_ids)
        text_items = [t for t in text_items if str(t.get("processed_id", "") or "").strip() in keep]

        if len(text_items) < 1:
            handle.errors["pipeline"] = "selection_removed_all_images"
            return

        print(f"[sel] kept processed_ids total={len(keep)} groups={len(selected_ids_by_ctx)}")


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

        # -------------------------
        # ADDED: Build Pinecone embedding jobs now
        # -------------------------
        embedding_jobs: List[Dict[str, Any]] = []
        for t in text_items:
            src = str(t.get("source_path", ""))
            processed_id = str(t.get("processed_id", f"processed_{int(t.get('idx', 0))}"))
            meta_entry = _resolve_meta_for_source(meta_ctx_map, src)
            if not meta_entry:
                continue

            prompt_emb = meta_entry.get("prompt_embedding")
            clip_emb = meta_entry.get("clip_embedding")
            ctx_emb = _pick_best_context_embedding(meta_entry)

            # Keep metadata small
            meta_small = {
                "clip_score": meta_entry.get("clip_score"),
                "confidence_score": meta_entry.get("confidence_score"),
                "final_score": meta_entry.get("final_score"),
            }

            embedding_jobs.append({
                "processed_id": processed_id,
                "unique_path": src,
                "base_context": str(t.get("base_context", "") or "").strip(),
                "prompt_embedding": prompt_emb,
                "clip_embedding": clip_emb,
                "context_embedding": ctx_emb,
                "meta": meta_small,
            })


        # -------------------------
        # ADDED: Pinecone save thread (PARALLEL to ImageClusters)
        # -------------------------
        def _run_pinecone_save() -> None:
            try:
                if not embedding_jobs:
                    return
                summary = PineconeSave.upsert_image_metadata_embeddings(embedding_jobs)
                try:
                    (out_dir / "pinecone_upsert_summary.json").write_text(
                        json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8"
                    )
                except Exception:
                    pass
            except BaseException as e:
                handle.errors["pinecone"] = f"{type(e).__name__}: {e}"
                print("[bg][ERR] Pinecone save thread crashed:")
                traceback.print_exc()
            finally:
                handle.pinecone_done.set()

        #pine_thread = threading.Thread(target=_run_pinecone_save, name="pinecone_save", daemon=False)
        #pine_thread.start()

        print("[9/10] ImageClusters (in-memory)...")
        with stage("ImageClusters.cluster_in_memory"):
            clusters_by_idx = ImageClusters.cluster_in_memory(
                preproc_by_idx=preproc_by_idx,
                vectors_by_idx=vectors_by_idx,
                save_outputs=True,
            )

        #pine_thread.join()  # <-- ensure Pinecone is finished before exiting pipeline section

        # Build qwen packs (now only the selected images exist in text_items)
        qwen_packs: Dict[int, Dict[str, Any]] = {}
        for t in text_items:
            idx = int(t["idx"])
            if idx not in clusters_by_idx:
                continue
            cleaned_bgr = preproc_by_idx[idx]["cleaned_bgr"]
            full_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)

            pack = clusters_by_idx[idx]
            pack["idx"] = idx
            pack["full_img_rgb"] = full_rgb
            pack["base_context"] = t.get("base_context", "") or ""
            pack["candidate_labels_raw"] = extract_candidate_words_from_text_payload(t["payload_json"])
            qwen_packs[idx] = pack

        # If Qwen couldn’t load, still set qwen_done so callers don't hang.
        if qwen_model is None:
            handle.qwen_done.set()

        # ---- RUN QWEN + IMAGECOLOURS IN PARALLEL ----

        def _run_qwen() -> None:
            try:
                if qwen_model is None:
                    return
                print("[10/11] qwentest (Transformers, async, hot GPU)...")
                with qwen_lock():
                    qwentest.label_clusters_transformers(
                        clusters_state=qwen_packs,
                        model=qwen_model,
                        processor=qwen_processor,
                        device=qwen_device,
                        save_outputs=True,
                    )
            except BaseException as e:
                handle.errors["qwen"] = f"{type(e).__name__}: {e}"
                print("[bg][ERR] Qwen thread crashed:")
                traceback.print_exc()
            finally:
                handle.qwen_done.set()

        def _run_colours() -> None:
            try:
                print("[11/11] ImageColours (async)...")
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
                # This is the event you asked for: signals you can start using coloured outputs now.
                handle.colours_done.set()
                print("[bg] ImageColours done. You can work with coloured outputs now.")

        qwen_thread = threading.Thread(target=_run_qwen, name="qwen_infer", daemon=False)
        colours_thread = threading.Thread(target=_run_colours, name="image_colours", daemon=False)

        if qwen_model is None:
            handle.qwen_done.set()
        else:
            qwen_thread.start()

        colours_thread.start()

        colours_thread.join()

        if qwen_model is not None:
            qwen_thread.join()

        print("[done] background pipeline finished.")

    except BaseException as e:
        handle.errors["pipeline_fatal"] = f"{type(e).__name__}: {e}"
        print("[bg][ERR] pipeline fatal crash:")
        traceback.print_exc()
    finally:
        handle.colours_done.set()
        handle.qwen_done.set()
        handle.pinecone_done.set()
        handle.pipeline_done.set()
        stage_report()



# =========================
# Public API: start in background thread
# =========================
def start_pipeline_background(
    *,
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    debug_save: bool = False,
    parallel_cpu: bool = True,
    gpu_index: int = 0,
    allowed_base_contexts: Optional[List[str]] = None,
) -> PipelineHandle:
    

    if REMOTE_API:
        # In remote mode, we never start local threads or load models.
        # Just call remote run and return a completed handle (same as blocking).
        return run_pipeline_blocking(
            model_id=model_id,
            debug_save=debug_save,
            parallel_cpu=parallel_cpu,
            gpu_index=gpu_index,
            allowed_base_contexts=allowed_base_contexts,
        )


    out_dir = BASE_DIR / "PipelineOutputs"

    colours_done = threading.Event()
    qwen_done = threading.Event()
    pipeline_done = threading.Event()

    # -------------------------
    # ADDED: pinecone event
    # -------------------------
    pinecone_done = threading.Event()

    dummy_thread = threading.Thread(target=lambda: None)

    handle = PipelineHandle(
        thread=dummy_thread,
        colours_done=colours_done,
        qwen_done=qwen_done,
        pipeline_done=pipeline_done,
        pinecone_done=pinecone_done,
        out_dir=out_dir,
        errors={},
    )

    # HOT LOAD ONCE at startup (keeps Qwen/SigLIP/MiniLM resident for the life of the process)\
    if not REMOTE_API:
        ensure_hot_models(
            qwen_model_id=model_id,
            gpu_index=gpu_index,
            cpu_threads=_PIPE_CPU_THREADS,
            warmup=True,
            load_siglip=True,
            load_minilm=True,
        )

    # IMPORTANT: daemon=False so the process does NOT exit while Qwen is still running.
    t = threading.Thread(
        target=_pipeline_worker,
        name="pipeline_bg",
        daemon=False,
        kwargs=dict(
            handle=handle,
            model_id=model_id,
            debug_save=debug_save,
            parallel_cpu=parallel_cpu,
            gpu_index=gpu_index,
            allowed_base_contexts=allowed_base_contexts,
        ),
    )
    handle.thread = t
    t.start()
    return handle


def run_pipeline_blocking(
    *,
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    debug_save: bool = False,
    parallel_cpu: bool = True,
    gpu_index: int = 0,
    allowed_base_contexts: Optional[List[str]] = None,
) -> PipelineHandle:
    # If remote API is set, DO NOT load models locally, DO NOT run pipeline locally.
    if REMOTE_API:
        out = _http_post_json(
            f"{REMOTE_API}/run_pipeline",
            {
                "model_id": model_id,
                "debug_save": bool(debug_save),
                "parallel_cpu": bool(parallel_cpu),
                "gpu_index": int(gpu_index),
                "allowed_base_contexts": allowed_base_contexts or [],
            },
            timeout=3600,
        )

        # create a fake handle that looks like a completed local run
        out_dir = BASE_DIR / "PipelineOutputs"
        h = PipelineHandle(
            thread=threading.Thread(target=lambda: None),
            colours_done=threading.Event(),
            qwen_done=threading.Event(),
            pipeline_done=threading.Event(),
            pinecone_done=threading.Event(),
            out_dir=out_dir,
            errors={},
        )
        h.colours_done.set()
        h.qwen_done.set()
        h.pinecone_done.set()
        h.pipeline_done.set()

        # bring back fields you actually use
        h.errors = out.get("errors") or {}
        h.unique_paths = out.get("unique_paths") or []
        h.unique_to_processed = out.get("unique_to_processed") or {}
        h.processed_to_unique = out.get("processed_to_unique") or {}
        h.selected_processed_ids = out.get("selected_processed_ids") or []
        h.selected_ids_by_base_context = out.get("selected_ids_by_base_context") or {}
        return h

    # Local fallback (your existing behavior)
    h = start_pipeline_background(
        model_id=model_id,
        debug_save=debug_save,
        parallel_cpu=parallel_cpu,
        gpu_index=gpu_index,
        allowed_base_contexts=allowed_base_contexts,
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


def get_images(
    prompt_or_map: Any,
    subj: Optional[str] = None,
    *,
    top_n_per_prompt: int = 2,
    min_final_score: float = 0.78,
    min_modalities: int = 3,
    top_k_per_modality: int = 50,
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    gpu_index: int = 0,
) -> Dict[str, List[str]]:
    """
    Input:
      - single: get_images("Eukaryotic cell", "Biology")
      - batch:  get_images({"Eukaryotic cell":"Biology", "Prokaryotic cell":"Biology"})

    Output:
      { prompt_text: ["processed_12","processed_81"] , ... }

    Behavior:
      1) Try PineconeFetch for each prompt.
      2) If accepted => return processed ids (no images).
      3) If not accepted => bundle misses, run researcher on them, then pipeline on those prompts only.
      4) Pipeline upserts to Pinecone + writes jsons + returns selected processed ids per prompt.
    """
    # normalize input to {prompt: topic}
    prompt_to_topic: Dict[str, str] = {}
    if isinstance(prompt_or_map, str):
        p = prompt_or_map.strip()
        t = (subj or "").strip()
        if p and t:
            prompt_to_topic[p] = t
    elif isinstance(prompt_or_map, dict):
        for k, v in prompt_or_map.items():
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                prompt_to_topic[k.strip()] = v.strip()

    if not prompt_to_topic:
        return {}

    # Keep models hot for the caller process
    if not REMOTE_API:
        ensure_hot_models(
            qwen_model_id=model_id,
            gpu_index=gpu_index,
            cpu_threads=_PIPE_CPU_THREADS,
            warmup=False,
            load_siglip=True,
            load_minilm=True,
        )

    results: Dict[str, List[str]] = {}
    misses: Dict[str, str] = {}

    # 1) pinecone fetch first
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
        return results

    # 2) research missing prompts (bundled)
    try:
        ImageResearcher.research_many(misses)
    except Exception:
        return results

    # 3) run pipeline ONLY for missing prompts
    h = run_pipeline_blocking(
        model_id=model_id,
        debug_save=False,
        parallel_cpu=True,
        gpu_index=gpu_index,
        allowed_base_contexts=list(misses.keys()),
    )

    # 4) pull per-prompt processed ids from pipeline selection
    by_ctx = getattr(h, "selected_ids_by_base_context", {}) or {}
    for prompt in misses.keys():
        picked = by_ctx.get(prompt, [])
        if isinstance(picked, list) and picked:
            results[prompt] = [str(x) for x in picked if str(x)]
        else:
            # fallback: try pinecone again after upsert
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
            except Exception:
                pass

    return results


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
    # HARD TEST INPUTS (edit these)
    prompts = {
        "Eukaryotic cell": "Biology",
        "Pythagorean theorem": "Math",
        "Glucose" : "Biology",
        "Human heart": "Biology",
    }

    print("[TEST] get_images hardcoded run")
    print("[TEST] prompts:", list(prompts.keys()))

    try:
        out = get_images(
            prompts,
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
