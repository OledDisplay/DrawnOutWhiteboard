from typing import List, Dict, Any, Optional
from collections import Counter
import os
import traceback
import numpy as np
from PIL import Image, UnidentifiedImageError

# keep your SiglipBackend as-is, but I strongly recommend this small safety tweak:
# (this does NOT change your existing comments)
class SiglipBackend:
    def __init__(self, model_name: str = "google/siglip-base-patch16-384", device: Optional[str] = None):
        import torch
        from transformers import SiglipProcessor, SiglipModel

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = SiglipModel.from_pretrained(model_name).to(self.device)
        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.model.eval()

    @staticmethod
    def _open_rgb(path: str) -> Image.Image:
        # fails loud and clear if Pillow cannot decode the file
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def encode_text(self, text: str) -> np.ndarray:
        import torch
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        with torch.no_grad():
            inputs = self.processor(
                text=[text],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            out = self.model.get_text_features(**inputs)
            out = out / out.norm(dim=-1, keepdim=True)
            return out.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_image(self, path: str) -> np.ndarray:
        import torch
        with torch.no_grad():
            img = self._open_rgb(path)
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)

            out = self.model.get_image_features(**inputs)
            out = out / out.norm(dim=-1, keepdim=True)
            return out.squeeze(0).cpu().numpy().astype(np.float32)


def _np_from_vec(v: Any) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    if a.ndim > 1:
        a = a.reshape(-1)
    return a

def _l2_normalize_np(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm == 0.0:
        return x
    return x / norm

def _cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def rerank_image_candidates_siglip(
    candidates: List[Dict[str, Any]],
    prompt_text: str,
    backend: Optional[SiglipBackend] = None,
    top_n: int = 30,
    final_k: int = 10,
    debug: bool = True,
    raise_on_empty: bool = True,
) -> List[Dict[str, Any]]:

    if not candidates:
        if raise_on_empty:
            raise ValueError("[RERANK] candidates was empty.")
        return []

    if backend is None:
        backend = SiglipBackend()

    skip = Counter()
    prepared: List[Dict[str, Any]] = []

    # show what your incoming dicts actually look like (first one only)
    if debug:
        print("[RERANK] first_candidate_keys:", list(candidates[0].keys()))

    for i, c in enumerate(candidates):
        img_path = c.get("image_path") or c.get("path")
        if not img_path:
            skip["missing_image_path_key_(expected_image_path_or_path)"] += 1
            if debug and skip["missing_image_path_key_(expected_image_path_or_path)"] <= 5:
                print(f"[RERANK][SKIP] idx={i} missing image path key. keys={list(c.keys())}")
            continue

        if not os.path.exists(img_path):
            skip["image_path_does_not_exist"] += 1
            if debug and skip["image_path_does_not_exist"] <= 5:
                print(f"[RERANK][SKIP] idx={i} path not found: {img_path}")
            continue

        ctx_emb = c.get("ctx_embedding")
        if ctx_emb is None:
            skip["missing_ctx_embedding_key_(expected_ctx_embedding)"] += 1
            if debug and skip["missing_ctx_embedding_key_(expected_ctx_embedding)"] <= 5:
                print(f"[RERANK][SKIP] idx={i} missing ctx_embedding. keys={list(c.keys())}")
            continue

        prompt_emb = c.get("prompt_embedding")
        if prompt_emb is None:
            skip["missing_prompt_embedding_key_(expected_prompt_embedding)"] += 1
            if debug and skip["missing_prompt_embedding_key_(expected_prompt_embedding)"] <= 5:
                print(f"[RERANK][SKIP] idx={i} missing prompt_embedding. keys={list(c.keys())}")
            continue

        ctx_conf = c.get("ctx_confidence")
        if ctx_conf is None:
            ctx_conf = c.get("ctx_sem_score", c.get("ctx_score", 0.0))

        ctx_text = c.get("ctx_text") or c.get("snippet") or c.get("heading_text") or ""

        prepared.append(
            {
                # passthrough an id if you have one (so upstream can mark “processed”)
                "id": c.get("id") or c.get("image_id") or c.get("sha") or c.get("hash"),
                "image_path": img_path,
                "ctx_embedding": ctx_emb,
                "prompt_embedding": prompt_emb,
                "ctx_confidence": float(ctx_conf),
                "ctx_score": c.get("ctx_score"),
                "ctx_sem_score": c.get("ctx_sem_score"),
                "ctx_text": ctx_text,
            }
        )

    if not prepared:
        msg = f"[RERANK][EMPTY] All candidates filtered out. skip_summary={dict(skip)}"
        if debug:
            print(msg)
        if raise_on_empty:
            raise ValueError(msg)
        return []

    prepared.sort(key=lambda d: d["ctx_confidence"], reverse=True)
    prepared = prepared[: min(top_n, len(prepared))]
    n = len(prepared)

    # ---- STEP 1: SigLIP stage ----
    try:
        backend_prompt_vec = backend.encode_text(prompt_text)
    except Exception as e:
        raise RuntimeError(f"[RERANK] encode_text(prompt_text) failed: {e!r}")

    scored: List[Dict[str, Any]] = []
    for i, c in enumerate(prepared):
        text_for_backend = c["ctx_text"] if c["ctx_text"] else prompt_text
        try:
            backend_ctx_vec = backend.encode_text(text_for_backend)
            backend_img_vec = backend.encode_image(c["image_path"])
        except UnidentifiedImageError as e:
            skip["pillow_cannot_decode_image_(likely_webp_support)"] += 1
            if debug and skip["pillow_cannot_decode_image_(likely_webp_support)"] <= 5:
                print(f"[RERANK][DECODE_FAIL] idx={i} path={c['image_path']} err={e!r}")
            continue
        except Exception as e:
            skip["siglip_encode_exception"] += 1
            if debug and skip["siglip_encode_exception"] <= 5:
                print(f"[RERANK][SIGLIP_FAIL] idx={i} path={c['image_path']} err={e!r}")
                traceback.print_exc()
            continue

        sim_img_ctx = _cosine_np(backend_img_vec, backend_ctx_vec)
        sim_img_prompt = _cosine_np(backend_img_vec, backend_prompt_vec)
        clip_like_score = 0.60 * sim_img_ctx + 0.40 * sim_img_prompt

        c["clip_embedding"] = backend_img_vec
        c["clip_score"] = float(clip_like_score)
        scored.append(c)

    if not scored:
        msg = f"[RERANK][EMPTY] No images survived SigLIP encoding. skip_summary={dict(skip)}"
        if debug:
            print(msg)
        if raise_on_empty:
            raise RuntimeError(msg)
        return []

    prepared = scored
    n = len(prepared)

    # ---- STEP 2: Context centroid stage ----
    prompt_ctx_vec = _l2_normalize_np(_np_from_vec(prepared[0]["prompt_embedding"]))

    for c in prepared:
        c["_ctx_emb_np"] = _l2_normalize_np(_np_from_vec(c["ctx_embedding"]))

    K = max(1, 1 + n // 10)
    centroid = np.zeros_like(prepared[0]["_ctx_emb_np"], dtype=np.float32)
    for c in prepared[:K]:
        centroid += c["_ctx_emb_np"]
    centroid = _l2_normalize_np(centroid / float(K))

    for c in prepared:
        s_centroid = _cosine_np(c["_ctx_emb_np"], centroid)
        s_prompt = _cosine_np(c["_ctx_emb_np"], prompt_ctx_vec)
        c["confidence_score"] = float(0.70 * s_centroid + 0.30 * s_prompt)

    # ---- STEP 3: Final score ----

    for c in prepared:
        c["final_score"] = float(0.60 * c["confidence_score"] + 0.40 * c["clip_score"])

    prepared.sort(key=lambda d: d["final_score"], reverse=True)
    winners = prepared[: min(final_k, len(prepared))]

    results: List[Dict[str, Any]] = []
    for c in winners:
        c.pop("_ctx_emb_np", None)
        results.append(
            {
                "id": c.get("id"),
                "image_path": c["image_path"],
                "ctx_embedding": c["ctx_embedding"],
                "clip_embedding": c["clip_embedding"],
                "ctx_confidence": c["ctx_confidence"],
                "ctx_score": c.get("ctx_score"),
                "ctx_sem_score": c.get("ctx_sem_score"),
                "clip_score": c["clip_score"],
                "confidence_score": c["confidence_score"],
                "final_score": c["final_score"],
            }
        )

    if debug:
        print(f"[RERANK] done. in={len(candidates)} prepared={len(prepared)} winners={len(results)} skip={dict(skip)}")

    return results
