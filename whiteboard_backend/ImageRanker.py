"""
SigLIP-based image reranking with context centroids and weighted scores.

Pipeline for up to N=30 initial candidates:

1) SigLIP IMAGE STAGE (60% img↔context, 40% img↔prompt):
   - For each candidate image:
       * Encode image with SigLIP (image embedding).
       * Encode its own context text with SigLIP text encoder.
       * Encode the prompt text with SigLIP text encoder (once).
       * siglip_score):
           0.60 * sim_img_ctx + 0.40 * sim_img_prompt

2) CONTEXT CONFIDENCE STAGE (MiniLM / context embedding space, 70% centroid, 30% prompt):
   - Candidates already have:
       * ctx_embedding      (from MiniLM / text encoder)
       * prompt_embedding   (same space as ctx_embedding)
       * ctx_confidence / ctx_sem_score / ctx_score
   - Take the top K = 1 + N//10 by ctx_confidence as prototypes.
   - Build a centroid vector in context-embedding space from those K.
   - For each candidate:
       * s_centroid = cos(ctx_embedding, centroid)
       * s_prompt   = cos(ctx_embedding, prompt_embedding)
       * confidence_score = 0.70 * s_centroid + 0.30 * s_prompt

3) FINAL SCORE (balance image vs confidence, 35% / 65%):
   - For each candidate:
       * final_score = 0.65 * confidence_score + 0.35 * clip_score
   - Sort all candidates by final_score and keep top M (default 10).
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
from PIL import Image
from transformers import SiglipProcessor, SiglipModel


# =========================
# SigLIP backend
# =========================

class SiglipBackend:
    """
    Simple SigLIP backend:
      - encode_text(text: str)  -> np.ndarray (L2-normalized)
      - encode_image(path: str) -> np.ndarray (L2-normalized)
    """

    def __init__(self, model_name: str = "google/siglip-base-patch16-224", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # THIS was missing in your version:
        self.model = SiglipModel.from_pretrained(model_name).to(self.device)

        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        inputs = self.processor(
            text=[text],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        out = self.model.get_text_features(**inputs)  # (1, d)
        out = out / out.norm(dim=-1, keepdim=True)
        return out.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_image(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        inputs = self.processor(
            images=img,
            return_tensors="pt",
        ).to(self.device)

        out = self.model.get_image_features(**inputs)  # (1, d)
        out = out / out.norm(dim=-1, keepdim=True)
        return out.squeeze(0).cpu().numpy().astype(np.float32)



# =========================
# NUMPY HELPERS
# =========================

def _np_from_vec(v: Any) -> np.ndarray:
    """Convert list/tuple/np-array to 1D float32 numpy array."""
    a = np.array(v, dtype=np.float32)
    if a.ndim > 1:
        a = a.reshape(-1)
    return a


def _l2_normalize_np(x: np.ndarray) -> np.ndarray:
    """L2-normalize a numpy vector (safe for zero-norm)."""
    norm = float(np.linalg.norm(x))
    if norm == 0.0:
        return x
    return x / norm


def _cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for 1D numpy vectors (assumed already normalized)."""
    return float(np.dot(a, b))


# =========================
# MAIN RERANKING LOGIC (SigLIP integrated)
# =========================

def rerank_image_candidates_siglip(
    candidates: List[Dict[str, Any]],
    prompt_text: str,
    backend: Optional[SiglipBackend] = None,
    top_n: int = 30,
    final_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Rerank image candidates using SigLIP + context centroids + combined final scores.

    Parameters
    ----------
    candidates : list of dict
        Each candidate must have:
          - 'image_path' or 'path'          : local image path
          - 'ctx_embedding'                 : context embedding (MiniLM or similar)
          - 'prompt_embedding'              : prompt embedding (same space as ctx_embedding)
          - 'ctx_confidence' (preferred) OR 'ctx_sem_score' / 'ctx_score'
          - 'ctx_text' (preferred) OR 'snippet' OR 'heading_text'
    """
    if not candidates:
        return []

    if backend is None:
        backend = SiglipBackend()

    # ---- STEP 0: prepare and sort by original context confidence ---- #

    prepared: List[Dict[str, Any]] = []
    for c in candidates:
        img_path = c.get("image_path") or c.get("path")
        if not img_path:
            continue

        ctx_emb = c.get("ctx_embedding")
        if ctx_emb is None:
            continue

        prompt_emb = c.get("prompt_embedding")
        if prompt_emb is None:
            raise ValueError("Candidate is missing 'prompt_embedding' in context space.")

        ctx_conf = c.get("ctx_confidence")
        if ctx_conf is None:
            ctx_conf = c.get("ctx_sem_score", c.get("ctx_score", 0.0))

        ctx_text = c.get("ctx_text") or c.get("snippet") or c.get("heading_text") or ""

        prepared.append(
            {
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
        return []

    # sort by original ctx_confidence, best first
    prepared.sort(key=lambda d: d["ctx_confidence"], reverse=True)
    prepared = prepared[: min(top_n, len(prepared))]
    n = len(prepared)

    # ---- STEP 1: SigLIP stage (60% image↔context, 40% image↔prompt) ---- #

    backend_prompt_vec = backend.encode_text(prompt_text)  # L2-normalized

    for c in prepared:
        text_for_backend = c["ctx_text"] if c["ctx_text"] else prompt_text

        backend_ctx_vec = backend.encode_text(text_for_backend)
        backend_img_vec = backend.encode_image(c["image_path"])

        sim_img_ctx = _cosine_np(backend_img_vec, backend_ctx_vec)
        sim_img_prompt = _cosine_np(backend_img_vec, backend_prompt_vec)

        # 60% image↔context, 40% image↔prompt
        clip_like_score = 0.60 * sim_img_ctx + 0.40 * sim_img_prompt

        # keep API naming compatible: 'clip_embedding' + 'clip_score'
        c["clip_embedding"] = backend_img_vec          # SigLIP image embedding
        c["clip_score"] = clip_like_score              # SigLIP-based score

    # ---- STEP 2: Context centroid + prompt (70% / 30%) in context/MiniLM space ---- #

    # prompt embedding in context space
    prompt_ctx_vec = _l2_normalize_np(_np_from_vec(prepared[0]["prompt_embedding"]))

    # normalize all context embeddings and stash
    for c in prepared:
        emb = _np_from_vec(c["ctx_embedding"])
        c["_ctx_emb_np"] = _l2_normalize_np(emb)

    # prototypes K = 1 + n//10
    K = max(1, 1 + n // 10)
    proto = prepared[:K]

    centroid = np.zeros_like(prepared[0]["_ctx_emb_np"], dtype=np.float32)
    for c in proto:
        centroid += c["_ctx_emb_np"]
    centroid /= float(K)
    centroid = _l2_normalize_np(centroid)

    # compute confidence_score per candidate
    for c in prepared:
        ctx_vec = c["_ctx_emb_np"]
        s_centroid = _cosine_np(ctx_vec, centroid)
        s_prompt = _cosine_np(ctx_vec, prompt_ctx_vec)
        confidence_score = 0.70 * s_centroid + 0.30 * s_prompt
        c["confidence_score"] = confidence_score

    # ---- STEP 3: Final score = 65% confidence + 35% SigLIP ("clip") ---- #

    for c in prepared:
        conf = c["confidence_score"]
        clip_s = c["clip_score"]
        final_score = 0.65 * conf + 0.35 * clip_s
        c["final_score"] = final_score

    # sort by final_score and keep top final_k
    prepared.sort(key=lambda d: d["final_score"], reverse=True)
    winners = prepared[: min(final_k, len(prepared))]

    # strip internals and build return dicts
    results: List[Dict[str, Any]] = []
    for c in winners:
        c.pop("_ctx_emb_np", None)
        results.append(
            {
                "image_path": c["image_path"],
                "ctx_embedding": c["ctx_embedding"],
                "clip_embedding": c["clip_embedding"],  # SigLIP image embedding
                "ctx_confidence": c["ctx_confidence"],
                "ctx_score": c.get("ctx_score"),
                "ctx_sem_score": c.get("ctx_sem_score"),
                "clip_score": c["clip_score"],
                "confidence_score": c["confidence_score"],
                "final_score": c["final_score"],
            }
        )

    return results
