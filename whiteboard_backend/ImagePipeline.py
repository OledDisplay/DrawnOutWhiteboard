#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import cv2
import numpy as np

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

def _lazy_import_qwentest():
    import qwentest  # local import so spawned workers don't load torch
    return qwentest



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

    # Start Qwen preload ASAP (CPU-side) while the rest runs.
    qwentest = _lazy_import_qwentest()
    qwen_model = None
    qwen_processor = None
    qwen_device = None

    preload_err: Optional[BaseException] = None
    preload_done = threading.Event()

    def _preload_qwen() -> None:
        nonlocal qwen_model, qwen_processor, qwen_device, preload_err
        try:
            qwen_model, qwen_processor, qwen_device = qwentest.preload_qwen_cpu_only(model_id)
            # warmup is optional
            try:
                qwentest.warmup_qwen_once(qwen_model, qwen_processor, qwen_device)
            except Exception:
                pass
        except BaseException as e:
            preload_err = e
            print("[bg][ERR] qwen preload crashed:")
            traceback.print_exc()
        finally:
            preload_done.set()
    preload_thread = threading.Thread(target=_preload_qwen, name="qwen_preload", daemon=False)
    preload_thread.start()

    try:
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

        print("[2/9] Load UniqueImages ONCE -> numpy...")
        img_items = load_unique_images_once(unique_paths)
        if not img_items:
            handle.errors["pipeline"] = "no_images_loaded"
            return

        print("[3/9] ImageText (in-memory)...")
        base_context_map = ImageText.load_base_context_map()

        # -------------------------
        # ADDED: metadata context map load once
        # -------------------------
        meta_ctx_map = load_image_metadata_context_map()

               # Build allowed set ONCE (don’t do it inside the loop)
        allowed_set: Optional[set[str]] = None
        if allowed_base_contexts:
            allowed_set = {
                str(x or "").strip()
                for x in allowed_base_contexts
                if str(x or "").strip()
            }

        # 1) Assign base_context for ALL images first
        for it in img_items:
            sp_norm = _norm_path(it["source_path"])
            bc = base_context_map.get(sp_norm, "")

            if not bc:
                meta_entry = _resolve_meta_for_source(meta_ctx_map, it["source_path"])
                if meta_entry and isinstance(meta_entry.get("base_context"), str):
                    bc = meta_entry["base_context"].strip()

            it["base_context"] = (bc or "").strip()

        # Optional: debug distribution (this will instantly show if contexts are missing)
        try:
            dist = Counter(str(it.get("base_context", "") or "").strip() for it in img_items)
            print("[ctx] base_context distribution:", dict(dist))
        except Exception:
            pass

        # 2) Filter ONCE after all items have base_context
        if allowed_set is not None:
            before = len(img_items)
            img_items = [
                it for it in img_items
                if str(it.get("base_context", "") or "").strip() in allowed_set
            ]
            print(f"[filter] base_context allowed={len(allowed_set)} images {before} -> {len(img_items)}")
            if not img_items:
                handle.errors["pipeline"] = "no_images_after_base_context_filter"
                return




        # -------------------------
        # CHANGED: ImageText now returns (text_items, unique_to_processed)
        # -------------------------
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
        # Qwen selection step (BLOCKING, no parallel threading)
        # =========================
        print("[4/9] qwentest selection (processed images, blocking, per-prompt)...")

        preload_done.wait()
        if preload_err is not None:
            handle.errors["qwen_preload"] = f"{type(preload_err).__name__}: {preload_err}"
            qwen_model = None

        selected_ids_all: List[str] = []
        selection_payload_by_ctx: Dict[str, Any] = {}
        selected_ids_by_ctx: Dict[str, List[str]] = {}

        if qwen_model is not None:
            try:
                qwen_model, qwen_device = qwentest.move_qwen_to_cuda_if_available(qwen_model, gpu_index=gpu_index)
                try:
                    qwentest.warmup_qwen_once(qwen_model, qwen_processor, qwen_device)
                except Exception:
                    pass
            except BaseException as e:
                handle.errors["qwen_move_cuda"] = f"{type(e).__name__}: {e}"
                qwen_model = None

        # group text_items by base_context (prompt)
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for t in text_items:
            bc = str(t.get("base_context", "") or "").strip()
            groups.setdefault(bc, []).append(t)

        # deterministic order
        group_keys = sorted(groups.keys(), key=lambda s: (s == "", s.lower()))

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
                    rgb = cv2.cvtColor(mbgr, cv2.COLOR_BGR2RGB)
                    from PIL import Image
                    pil = Image.fromarray(rgb, mode="RGB")
                    processed_items_for_qwen.append({
                        "processed_id": pid,
                        "image_pil": pil,
                        "base_context": bc,
                    })
                except Exception:
                    continue

            if not processed_items_for_qwen:
                continue

            picked: List[str] = []
            payload: Dict[str, Any] = {}

            if qwen_model is not None:
                try:
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
            if len(picked) < 2:
                tmp = []
                for t in items:
                    pid = str(t.get("processed_id", "") or "").strip()
                    if pid:
                        tmp.append(pid)
                    if len(tmp) >= 2:
                        break
                picked = tmp[:2]

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


        print("[5/9] ImagePreprocessor (in-memory)...")
        preproc_by_idx = ImagePreprocessor.process_images_in_memory(
            text_items=text_items,
            save_outputs=debug_save,
            parallel=parallel_cpu,
        )

        print("[6/9] ImageSkeletonizer (in-memory)...")
        skel_by_idx = ImageSkeletonizer.skeletonize_in_memory(
            preproc_by_idx=preproc_by_idx,
            save_outputs=debug_save,
            parallel=parallel_cpu,
        )

        print("[7/9] ImageVectorizer (in-memory)...")
        vectors_by_idx = ImageVectorizer.vectorize_in_memory(
            skel_by_idx=skel_by_idx,
            save_outputs=True,
            parallel=parallel_cpu,
        )

        # -------------------------
        # ADDED: Build Pinecone embedding jobs now (pipeline passes ALL info)
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

        pine_thread = threading.Thread(target=_run_pinecone_save, name="pinecone_save", daemon=False)
        pine_thread.start()

        print("[8/9] ImageClusters (in-memory)...")
        clusters_by_idx = ImageClusters.cluster_in_memory(
            preproc_by_idx=preproc_by_idx,
            vectors_by_idx=vectors_by_idx,
            save_outputs=True,
        )
        pine_thread.join()  # <-- ensure Pinecone is finished before exiting pipeline section

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
                print("[9/10] qwentest (Transformers, async)...")
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
                print("[10/10] ImageColours (async)...")
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
    except Exception as e:
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

def _hardcoded_get_images_test() -> int:
    # HARD TEST INPUTS (edit these)
    prompts = {
        "Eukaryotic cell": "Biology",
        "Pythagorean theorem": "Math",
        # "Human heart anatomy": "Biology",
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
    # HARDCODED TEST MODE: run ONLY the new PineconeFetch -> fallback researcher -> pipeline flow.
    raise SystemExit(_hardcoded_get_images_test())
