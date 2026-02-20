# ImagePineconeFetch.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

from dotenv import load_dotenv

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec  # not used here, but keep consistent with your save script

# Embedders
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoProcessor, SiglipModel


# ------------------------------------------------------------
# .env load (same pattern as your save script)
# Put .env next to this file unless you change the path.
# ------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


# ------------------------------------------------------------
# ENV SETTINGS
# ------------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_INDEX_PREFIX = os.getenv("PINECONE_INDEX_PREFIX", "img-meta")

PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")

# Optional: if you saved your plan to disk, fetch will reuse it.
# If missing, it will rebuild the same deterministic plan from dims.
PLAN_PATH = Path(os.getenv("PINECONE_PLAN_PATH", "")) if os.getenv("PINECONE_PLAN_PATH") else (Path(__file__).resolve().parent / "pinecone_index_plan.json")


# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------
MINILM_NAME = os.getenv("MINILM_NAME", "all-MiniLM-L6-v2")
SIGLIP_NAME = os.getenv("SIGLIP_NAME", "google/siglip-base-patch16-384")


# ------------------------------------------------------------
# INTERNAL CACHES (so repeated calls don't reload models)
# ------------------------------------------------------------
_minilm_model: Optional[SentenceTransformer] = None
_siglip_processor: Optional[Any] = None
_siglip_model: Optional[SiglipModel] = None
_siglip_device: Optional[torch.device] = None
_pc: Optional[Pinecone] = None
_opened_indexes: Dict[str, Any] = {}
_plan_cache: Dict[Tuple[int, int, int], Dict[str, Tuple[str, str]]] = {}

import threading

# ... keep your existing caches ...

# Optional externally-provided (hot) models from ImagePipeline.py
_minilm_tok: Optional[Any] = None
_minilm_trf_model: Optional[Any] = None
_minilm_device: Optional[torch.device] = None

_EMBED_MINILM_LOCK = threading.Lock()
_EMBED_SIGLIP_LOCK = threading.Lock()

def configure_hot_models(*, siglip_bundle: Any = None, minilm_bundle: Any = None) -> None:
    """
    Allows ImagePipeline.py (same process) to inject already-loaded models here,
    so PineconeFetch does NOT reload them.

    Expected bundle shapes:
      - SigLIP bundle: has .model, .processor, .device
      - MiniLM bundle: either sentence-transformers (.model has encode)
        OR transformers (.model + .tokenizer)
    """
    global _siglip_processor, _siglip_model, _siglip_device
    global _minilm_model, _minilm_tok, _minilm_trf_model, _minilm_device

    if siglip_bundle is not None:
        try:
            _siglip_model = siglip_bundle.model
            _siglip_processor = siglip_bundle.processor
            dev = getattr(siglip_bundle, "device", None)
            if isinstance(dev, torch.device):
                _siglip_device = dev
            else:
                _siglip_device = torch.device(str(dev) if dev else ("cuda" if torch.cuda.is_available() else "cpu"))
        except Exception:
            # If injection fails, just keep local lazy-loading behavior
            pass

    if minilm_bundle is not None:
        try:
            # sentence-transformers path
            if getattr(minilm_bundle, "use_sentence_transformers", False) and hasattr(minilm_bundle.model, "encode"):
                _minilm_model = minilm_bundle.model
                _minilm_tok = None
                _minilm_trf_model = None
                _minilm_device = None
            else:
                # transformers path
                _minilm_model = None
                _minilm_trf_model = minilm_bundle.model
                _minilm_tok = minilm_bundle.tokenizer
                dev = getattr(minilm_bundle, "device", None)
                if isinstance(dev, torch.device):
                    _minilm_device = dev
                else:
                    _minilm_device = torch.device(str(dev) if dev else ("cuda" if torch.cuda.is_available() else "cpu"))
        except Exception:
            pass

# ------------------------------------------------------------
# SMALL UTILS
# ------------------------------------------------------------
def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 0:
        return v
    return v / n


def _vec_dim(v: Any) -> Optional[int]:
    if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
        return len(v)
    return None


def _open_index(pc: Pinecone, index_name: str):
    desc = pc.describe_index(index_name)
    host = getattr(desc, "host", None)
    if not host and isinstance(desc, dict):
        host = desc.get("host")
    if not host:
        raise RuntimeError(f"Could not resolve host for index: {index_name}")
    return pc.Index(host=host)


def _index_exists(pc: Pinecone, index_name: str) -> bool:
    try:
        pc.describe_index(index_name)
        return True
    except Exception:
        return False
    

def _get_pc() -> Pinecone:
    global _pc
    if _pc is None:
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY missing. Put it in .env next to this file.")
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    return _pc

def _get_index(index_name: str):
    global _opened_indexes
    if index_name in _opened_indexes:
        return _opened_indexes[index_name]
    pc = _get_pc()
    if not _index_exists(pc, index_name):
        raise RuntimeError(f"Index not found: {index_name}")
    idx = _open_index(pc, index_name)
    _opened_indexes[index_name] = idx
    return idx



# ------------------------------------------------------------
# PLAN
# ------------------------------------------------------------
def _minimal_index_plan(prompt_dim: Optional[int], clip_dim: Optional[int], context_dim: Optional[int]) -> Dict[str, Tuple[str, str]]:
    """
    Returns:
      {
        "prompt":  (index_name, namespace),
        "clip":    (index_name, namespace),
        "context": (index_name, namespace),
      }

    Strategy mirrors your save script:
      - If all 3 dims match -> single index, namespaces prompt/clip/context.
      - Else if prompt_dim == context_dim -> share one index (prompt/context), clip separate.
      - Else -> 3 indexes.
    """
    def idx(name: str, dim: Optional[int]) -> str:
        d = dim if dim is not None else 0
        return f"{PINECONE_INDEX_PREFIX}-{name}-{d}"

    if prompt_dim and clip_dim and context_dim and (prompt_dim == clip_dim == context_dim):
        shared = f"{PINECONE_INDEX_PREFIX}-all-{prompt_dim}"
        return {
            "prompt": (shared, "prompt"),
            "clip": (shared, "clip"),
            "context": (shared, "context"),
        }

    if prompt_dim and context_dim and (prompt_dim == context_dim):
        shared = f"{PINECONE_INDEX_PREFIX}-promptctx-{prompt_dim}"
        return {
            "prompt": (shared, "prompt"),
            "context": (shared, "context"),
            "clip": (idx("clip", clip_dim), "clip"),
        }

    return {
        "prompt": (idx("prompt", prompt_dim), "prompt"),
        "clip": (idx("clip", clip_dim), "clip"),
        "context": (idx("context", context_dim), "context"),
    }


def load_plan_or_build(prompt_dim: int, clip_dim: int, context_dim: int) -> Dict[str, Tuple[str, str]]:
    """
    If PLAN_PATH exists, load it.
    Else, build a deterministic plan from dims (same naming as save script).
    """
    if PLAN_PATH.exists():
        raw = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        out: Dict[str, Tuple[str, str]] = {}
        for k, v in raw.items():
            # support {"prompt": ["index","ns"]} OR {"prompt": ("index","ns")}
            if isinstance(v, (list, tuple)) and len(v) == 2:
                out[str(k)] = (str(v[0]), str(v[1]))
        if out:
            return out

    return _minimal_index_plan(prompt_dim, clip_dim, context_dim)


# ------------------------------------------------------------
# EMBEDDERS
# ------------------------------------------------------------
def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def embed_minilm(text: str) -> List[float]:
    global _minilm_model, _minilm_tok, _minilm_trf_model, _minilm_device

    # If ImagePipeline injected a hot sentence-transformers model, use it.
    if _minilm_model is not None:
        with _EMBED_MINILM_LOCK:
            vec = _minilm_model.encode([text], normalize_embeddings=True)
        v = np.asarray(vec[0], dtype=np.float32)
        return v.tolist()

    # If ImagePipeline injected a hot transformers model+tokenizer, use it.
    if _minilm_trf_model is not None and _minilm_tok is not None and _minilm_device is not None:
        with _EMBED_MINILM_LOCK, torch.inference_mode():
            inputs = _minilm_tok([text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(_minilm_device) for k, v in inputs.items()}
            out = _minilm_trf_model(**inputs)
            pooled = _mean_pool(out.last_hidden_state, inputs["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            feats = pooled[0].detach().cpu().float().numpy().astype(np.float32)
        return feats.tolist()

    # Fallback: local lazy-load (old behavior)
    if _minilm_model is None:
        _minilm_model = SentenceTransformer(MINILM_NAME)

    with _EMBED_MINILM_LOCK:
        vec = _minilm_model.encode([text], normalize_embeddings=True)
    v = np.asarray(vec[0], dtype=np.float32)
    return v.tolist()



def embed_siglip_text(text: str) -> List[float]:
    global _siglip_processor, _siglip_model, _siglip_device

    # If ImagePipeline injected hot SigLIP, use it; otherwise lazy-load locally.
    if _siglip_processor is None or _siglip_model is None or _siglip_device is None:
        _siglip_processor = AutoProcessor.from_pretrained(SIGLIP_NAME)
        _siglip_model = SiglipModel.from_pretrained(SIGLIP_NAME)
        _siglip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _siglip_model.to(_siglip_device)
        _siglip_model.eval()

    with _EMBED_SIGLIP_LOCK, torch.inference_mode():
        inputs = _siglip_processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # works for BatchEncoding or dict-like
        try:
            inputs = inputs.to(_siglip_device)
        except Exception:
            inputs = {k: v.to(_siglip_device) for k, v in inputs.items()}

        feats = _siglip_model.get_text_features(**inputs)

    feats = feats[0].detach().cpu().float().numpy()
    feats = _l2_normalize(feats)
    return feats.astype(np.float32).tolist()



# ------------------------------------------------------------
# QUERY + FUSION
# ------------------------------------------------------------
@dataclass
class Match:
    processed_id: str
    score: float
    metadata: Dict[str, Any]


def _extract_matches(res: Any) -> List[Match]:
    """
    Works with:
      - GRPC object response: res.matches
      - dict response: res["matches"]
    """
    matches = []
    raw = None

    if hasattr(res, "matches"):
        raw = res.matches
    elif isinstance(res, dict):
        raw = res.get("matches", [])
    else:
        raw = []

    for m in raw:
        if isinstance(m, dict):
            pid = str(m.get("id", "")).strip()
            sc = float(m.get("score", 0.0))
            md = m.get("metadata", {}) if isinstance(m.get("metadata", {}), dict) else {}
        else:
            pid = str(getattr(m, "id", "")).strip()
            sc = float(getattr(m, "score", 0.0))
            md = getattr(m, "metadata", None)
            md = md if isinstance(md, dict) else {}

        if not pid:
            continue
        matches.append(Match(processed_id=pid, score=sc, metadata=md))

    return matches


def _minmax_norm(scores_by_id: Dict[str, float]) -> Dict[str, float]:
    if not scores_by_id:
        return {}

    vals = list(scores_by_id.values())
    mn = float(min(vals))
    mx = float(max(vals))
    if mx - mn < 1e-9:
        # all equal -> collapse to 1.0
        return {k: 1.0 for k in scores_by_id.keys()}

    return {k: (float(v) - mn) / (mx - mn) for k, v in scores_by_id.items()}


def _query_one(
    idx,
    *,
    namespace: str,
    vector: List[float],
    top_k: int,
) -> List[Match]:
    # Pinecone docs show include_metadata=True usage in Python query examples. :contentReference[oaicite:2]{index=2}
    res = idx.query(
        namespace=namespace,
        vector=vector,
        top_k=int(top_k),
        include_values=False,
        include_metadata=True,
    )
    return _extract_matches(res)


def fetch_best_processed(
    prompt: str,
    *,
    top_k_per_modality: int = 50,
    min_modalities: int = 2,
    return_top_n: int = 5,
    weights: Tuple[float, float, float] = (0.35, 0.40, 0.25),  # (prompt, siglip, context)
) -> Dict[str, Any]:
    """
    Returns:
      {
        "best_processed_id": "processed_12",
        "ranking": [
          {
            "processed_id": "...",
            "final_score": 0.83,
            "hits": 3,
            "scores": {"prompt": 0.7, "clip": 0.9, "context": 0.8},
            "metadata_any": {...}
          },
          ...
        ],
        "used_plan": {...}
      }
    """
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY missing. Put it in your .env next to this script.")

    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt is empty")

    # 1) Embed query
    q_prompt = embed_minilm(prompt)         # for prompt DB
    q_context = q_prompt                    # context DB is also MiniLM in your pipeline assumption
    q_clip = embed_siglip_text(prompt)      # for siglip/clip DB

    prompt_dim = len(q_prompt)
    context_dim = len(q_context)
    clip_dim = len(q_clip)

    # 2) Resolve index plan (load from file or rebuild deterministically)
    plan = load_plan_or_build(prompt_dim, clip_dim, context_dim)

    # 3) Connect + open needed indexes
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # unique index names -> open once
    needed_index_names = sorted({plan[k][0] for k in plan.keys()})

    for name in needed_index_names:
        if not _index_exists(pc, name):
            raise RuntimeError(
                f"Index not found: {name}. "
                f"Either your save script created different names, or you need to save/load the plan."
            )

    opened = {name: _open_index(pc, name) for name in needed_index_names}

    # 4) Query each modality
    prompt_matches: List[Match] = []
    clip_matches: List[Match] = []
    context_matches: List[Match] = []

    # prompt
    i_name, ns = plan["prompt"]
    prompt_matches = _query_one(opened[i_name], namespace=ns, vector=q_prompt, top_k=top_k_per_modality)

    # clip (siglip)
    i_name, ns = plan["clip"]
    clip_matches = _query_one(opened[i_name], namespace=ns, vector=q_clip, top_k=top_k_per_modality)

    # context
    i_name, ns = plan["context"]
    context_matches = _query_one(opened[i_name], namespace=ns, vector=q_context, top_k=top_k_per_modality)

    # 5) Build per-modality score dicts
    s_prompt = {m.processed_id: float(m.score) for m in prompt_matches}
    s_clip = {m.processed_id: float(m.score) for m in clip_matches}
    s_ctx = {m.processed_id: float(m.score) for m in context_matches}

    # 6) Normalize within each modality so weights are meaningful
    n_prompt = _minmax_norm(s_prompt)
    n_clip = _minmax_norm(s_clip)
    n_ctx = _minmax_norm(s_ctx)

    w_prompt, w_clip, w_ctx = weights

    # 7) Fuse (weighted sum) + enforce overlap requirement
    all_ids = set(n_prompt.keys()) | set(n_clip.keys()) | set(n_ctx.keys())

    fused = []
    meta_by_id: Dict[str, Dict[str, Any]] = {}

    # keep some metadata (whatever modality returned it first)
    for m in prompt_matches + clip_matches + context_matches:
        if m.processed_id not in meta_by_id and isinstance(m.metadata, dict):
            meta_by_id[m.processed_id] = m.metadata

    for pid in all_ids:
        hits = 0
        sp = float(n_prompt.get(pid, 0.0))
        sc = float(n_clip.get(pid, 0.0))
        sx = float(n_ctx.get(pid, 0.0))

        if pid in n_prompt:
            hits += 1
        if pid in n_clip:
            hits += 1
        if pid in n_ctx:
            hits += 1

        if hits < int(min_modalities):
            continue

        final = (w_prompt * sp) + (w_clip * sc) + (w_ctx * sx)

        fused.append({
            "processed_id": pid,
            "final_score": float(final),
            "hits": int(hits),
            "scores": {"prompt": sp, "clip": sc, "context": sx},
            "metadata_any": meta_by_id.get(pid, {}),
        })

    fused.sort(key=lambda x: (x["final_score"], x["hits"]), reverse=True)

    best = fused[0]["processed_id"] if fused else None

    return {
        "best_processed_id": best,
        "ranking": fused[: int(return_top_n)],
        "used_plan": {k: [v[0], v[1]] for k, v in plan.items()},
        "dims": {"prompt": prompt_dim, "clip": clip_dim, "context": context_dim},
    }

def fetch_processed_ids_for_prompt(
    prompt: str,
    *,
    top_n: int = 2,
    top_k_per_modality: int = 50,
    min_modalities: int = 3,
    min_final_score: float = 0.78,
    require_base_context_match: bool = True,
) -> List[str]:
    """
    Returns [] if not accepted.
    Returns list of processed_ids (length up to top_n) if accepted.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return []

    # embed prompt twice (minilm prompt+context), and siglip once
    q_prompt = embed_minilm(prompt)
    q_context = q_prompt
    q_clip = embed_siglip_text(prompt)

    dims_key = (len(q_prompt), len(q_clip), len(q_context))

    # plan caching
    plan = _plan_cache.get(dims_key)
    if plan is None:
        plan = load_plan_or_build(dims_key[0], dims_key[1], dims_key[2])
        _plan_cache[dims_key] = plan

    # open indexes once
    idx_prompt = _get_index(plan["prompt"][0])
    idx_clip = _get_index(plan["clip"][0])
    idx_ctx = _get_index(plan["context"][0])

    prompt_matches = _query_one(idx_prompt, namespace=plan["prompt"][1], vector=q_prompt, top_k=top_k_per_modality)
    clip_matches   = _query_one(idx_clip,   namespace=plan["clip"][1],   vector=q_clip,   top_k=top_k_per_modality)
    ctx_matches    = _query_one(idx_ctx,    namespace=plan["context"][1],vector=q_context,top_k=top_k_per_modality)

    s_prompt = {m.processed_id: float(m.score) for m in prompt_matches}
    s_clip   = {m.processed_id: float(m.score) for m in clip_matches}
    s_ctx    = {m.processed_id: float(m.score) for m in ctx_matches}

    n_prompt = _minmax_norm(s_prompt)
    n_clip   = _minmax_norm(s_clip)
    n_ctx    = _minmax_norm(s_ctx)

    # keep metadata (first seen)
    meta_by_id: Dict[str, Dict[str, Any]] = {}
    for m in prompt_matches + clip_matches + ctx_matches:
        if m.processed_id not in meta_by_id and isinstance(m.metadata, dict):
            meta_by_id[m.processed_id] = m.metadata

    fused = []
    all_ids = set(n_prompt.keys()) | set(n_clip.keys()) | set(n_ctx.keys())

    # weights: (prompt, siglip, context)
    w_prompt, w_clip, w_ctx = (0.35, 0.40, 0.25)

    prompt_norm = prompt.strip().lower()

    for pid in all_ids:
        hits = 0
        if pid in n_prompt: hits += 1
        if pid in n_clip:   hits += 1
        if pid in n_ctx:    hits += 1
        if hits < int(min_modalities):
            continue

        sp = float(n_prompt.get(pid, 0.0))
        sc = float(n_clip.get(pid, 0.0))
        sx = float(n_ctx.get(pid, 0.0))
        final = (w_prompt * sp) + (w_clip * sc) + (w_ctx * sx)

        md = meta_by_id.get(pid, {}) or {}
        if require_base_context_match:
            bc = str(md.get("base_context", "") or "").strip().lower()
            if bc and bc != prompt_norm:
                continue

        fused.append((float(final), int(hits), pid))

    fused.sort(key=lambda x: (x[0], x[1]), reverse=True)

    if not fused:
        return []

    # accept policy: top candidate must pass min_final_score
    if fused[0][0] < float(min_final_score):
        return []

    # return top_n pids that also satisfy score >= min_final_score
    out: List[str] = []
    for final, hits, pid in fused:
        if final < float(min_final_score):
            continue
        out.append(pid)
        if len(out) >= int(top_n):
            break
    return out



# ------------------------------------------------------------
# CLI quick test
# ------------------------------------------------------------
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        print("Usage: python ImagePineconeFetch.py \"your image prompt here\"")
        raise SystemExit(2)

    res = fetch_best_processed(q, top_k_per_modality=50, min_modalities=2, return_top_n=5)
    print(json.dumps(res, indent=2, ensure_ascii=False))
