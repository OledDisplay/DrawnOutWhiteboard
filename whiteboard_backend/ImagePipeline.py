#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

import ImageText
import ImagePreprocessor
import ImageSkeletonizer
import ImageVectorizer
import ImageClusters
import ImageColours
import qwentest  # transformers-based (must provide preload/move/warmup/label funcs)


IN_DIR = Path("ResearchImages\\UniqueImages")
BASE_DIR = Path(__file__).resolve().parent


def _norm_path(p: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


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


# =========================
# Async handle
# =========================
@dataclass
class PipelineHandle:
    thread: threading.Thread
    colours_done: threading.Event
    qwen_done: threading.Event
    pipeline_done: threading.Event
    out_dir: Path
    errors: Dict[str, str] = field(default_factory=dict)

    def wait_colours(self, timeout: Optional[float] = None) -> bool:
        return self.colours_done.wait(timeout)

    def wait_qwen(self, timeout: Optional[float] = None) -> bool:
        return self.qwen_done.wait(timeout)

    def join(self, timeout: Optional[float] = None) -> None:
        self.thread.join(timeout)


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
) -> None:
    out_dir = handle.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start Qwen preload ASAP (CPU-side) while the rest runs.
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
        unique_paths = [
            os.path.join(IN_DIR, f)
            for f in os.listdir(IN_DIR)
            if os.path.isfile(os.path.join(IN_DIR, f))
        ]

        print("[2/8] Load UniqueImages ONCE -> numpy...")
        img_items = load_unique_images_once(unique_paths)
        if not img_items:
            handle.errors["pipeline"] = "no_images_loaded"
            return

        print("[3/8] ImageText (in-memory)...")
        base_context_map = ImageText.load_base_context_map()
        for it in img_items:
            it["base_context"] = base_context_map.get(_norm_path(it["source_path"]), "")
        text_items = ImageText.process_images_in_memory(
            image_items=img_items,
            start_index=0,
            save_outputs=debug_save,
        )
        if not text_items:
            handle.errors["pipeline"] = "imagetext_no_outputs"
            return

        print("[4/8] ImagePreprocessor (in-memory)...")
        preproc_by_idx = ImagePreprocessor.process_images_in_memory(
            text_items=text_items,
            save_outputs=debug_save,
            parallel=parallel_cpu,
        )

        print("[5/8] ImageSkeletonizer (in-memory)...")
        skel_by_idx = ImageSkeletonizer.skeletonize_in_memory(
            preproc_by_idx=preproc_by_idx,
            save_outputs=debug_save,
            parallel=parallel_cpu,
        )

        print("[6/8] ImageVectorizer (in-memory)...")
        vectors_by_idx = ImageVectorizer.vectorize_in_memory(
            skel_by_idx=skel_by_idx,
            save_outputs=True,
            parallel=parallel_cpu,
        )

        print("[7/8] ImageClusters (in-memory)...")
        clusters_by_idx = ImageClusters.cluster_in_memory(
            preproc_by_idx=preproc_by_idx,
            vectors_by_idx=vectors_by_idx,
            save_outputs=True,
        )

        # Build qwen packs
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

        # Wait for preload to finish
        preload_done.wait()
        if preload_err is not None:
            handle.errors["qwen_preload"] = f"{type(preload_err).__name__}: {preload_err}"
            qwen_model = None

        # Move to GPU
        if qwen_model is not None:
            try:
                # Expected signature:
                # move_qwen_to_cuda_if_available(model, gpu_index=int) -> (model, device)
                qwen_model, qwen_device = qwentest.move_qwen_to_cuda_if_available(qwen_model, gpu_index=gpu_index)
            except Exception as e:
                handle.errors["qwen_cuda_move"] = f"{type(e).__name__}: {e}"
                print("[bg][ERR] qwen cuda move crashed:")
                traceback.print_exc()

            # Optional second warmup (after cuda move)
            try:
                qwentest.warmup_qwen_once(qwen_model, qwen_processor, qwen_device)
            except Exception:
                pass

        # ---- RUN QWEN + IMAGECOLOURS IN PARALLEL ----

        def _run_qwen() -> None:
            try:
                if qwen_model is None:
                    return
                print("[8/9] qwentest (Transformers, async)...")
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
                print("[9/9] ImageColours (async)...")
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

        # If Qwen couldn’t load, still set qwen_done so callers don't hang.
        if qwen_model is None:
            handle.qwen_done.set()
        else:
            qwen_thread.start()

        colours_thread.start()

        # Wait for colours first so event is reliable
        colours_thread.join()

        # Then wait for Qwen if it ran
        if qwen_model is not None:
            qwen_thread.join()

        print("[done] background pipeline finished.")

    except BaseException as e:
        handle.errors["pipeline_fatal"] = f"{type(e).__name__}: {e}"
        print("[bg][ERR] pipeline fatal crash:")
        traceback.print_exc()
    finally:
        # Ensure events are always released
        handle.colours_done.set()
        handle.qwen_done.set()
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
) -> PipelineHandle:
    out_dir = BASE_DIR / "PipelineOutputs"

    colours_done = threading.Event()
    qwen_done = threading.Event()
    pipeline_done = threading.Event()

    dummy_thread = threading.Thread(target=lambda: None)

    handle = PipelineHandle(
        thread=dummy_thread,
        colours_done=colours_done,
        qwen_done=qwen_done,
        pipeline_done=pipeline_done,
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
        ),
    )
    handle.thread = t
    t.start()
    return handle


# Optional: synchronous wrapper (waits for everything)
def run_pipeline_blocking(
    *,
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    debug_save: bool = False,
    parallel_cpu: bool = True,
    gpu_index: int = 0,
) -> PipelineHandle:
    h = start_pipeline_background(
        model_id=model_id,
        debug_save=debug_save,
        parallel_cpu=parallel_cpu,
        gpu_index=gpu_index,
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
        # If False: script stays alive until Qwen finishes (prevents “it stopped”)
        # If True: script exits after colours are ready (will terminate Qwen work)
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


if __name__ == "__main__":
    kw = _parse_cli(sys.argv)

    detach = bool(kw.pop("detach_after_colours"))

    if kw.pop("blocking"):
        h = run_pipeline_blocking(**kw)
        if h.errors:
            print("[errors]", h.errors)
        raise SystemExit(0)

    h = start_pipeline_background(**kw)
    print("[bg] pipeline started.")
    print("[bg] waiting for colours_done...")

    h.colours_done.wait()
    print("[bg] colours_done set. Coloured outputs should be usable now.")
    if h.errors.get("colours"):
        print("[bg][WARN] colours had an error:", h.errors["colours"])

    # If you detach here, Qwen WILL be killed once the process exits.
    if detach:
        print("[bg] detach requested -> exiting now (Qwen will be terminated).")
        if h.errors:
            print("[errors]", h.errors)
        raise SystemExit(0)

    # Keep process alive so Qwen actually keeps running.
    if not h.qwen_done.is_set():
        print("[bg] Qwen still running. Keeping process alive until it finishes.")
        try:
            last = time.time()
            while not h.qwen_done.wait(timeout=0.5):
                # heartbeat every ~10s so you know it's alive
                if (time.time() - last) >= 10.0:
                    print("[bg] still running... (CTRL+C to abort)")
                    last = time.time()
        except KeyboardInterrupt:
            print("\n[bg] CTRL+C received. Exiting (threads will be terminated).")
            if h.errors:
                print("[errors]", h.errors)
            raise SystemExit(130)

    # Ensure pipeline is fully done
    h.pipeline_done.wait()

    print("[bg] Qwen done. Pipeline finished.")
    if h.errors:
        print("[errors]", h.errors)
