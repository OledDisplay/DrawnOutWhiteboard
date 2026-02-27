# PineconeSave.py
from __future__ import annotations

import os
import time
import json
from typing import Any, Dict, List, Optional, Tuple

from pinecone.grpc import PineconeGRPC as Pinecone  # per docs
from pinecone import ServerlessSpec

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


# -----------------------------
# SETTINGS (env-driven)
# -----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_INDEX_PREFIX = os.getenv("PINECONE_INDEX_PREFIX", "img-meta")

# If you want a different metric, change it, but cosine is typical for embeddings
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")

UPSERT_BATCH = int(os.getenv("PINECONE_UPSERT_BATCH", "100"))

# Optional plan path so fetch can reuse exact index/namespace mapping
PLAN_PATH = (
    Path(os.getenv("PINECONE_PLAN_PATH", "")).resolve()
    if os.getenv("PINECONE_PLAN_PATH")
    else (Path(__file__).resolve().parent / "pinecone_index_plan.json")
)


def _chunked(lst: List[Any], n: int) -> List[List[Any]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]


def _vec_dim(v: Any) -> Optional[int]:
    if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
        return len(v)
    return None


def _ensure_index(pc: Pinecone, index_name: str, dim: int) -> None:
    if pc.has_index(index_name):
        return

    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=int(dim),
        metric=str(PINECONE_METRIC),
        spec=ServerlessSpec(cloud=str(PINECONE_CLOUD), region=str(PINECONE_REGION)),
        deletion_protection="disabled",
        tags={"component": "image_pipeline"},
    )
    # Create is async; wait until it exists
    while not pc.has_index(index_name):
        time.sleep(0.5)


def _open_index(pc: Pinecone, index_name: str):
    desc = pc.describe_index(index_name)
    host = getattr(desc, "host", None)
    if not host:
        # Some SDK variants return dict-like
        host = desc.get("host") if isinstance(desc, dict) else None
    if not host:
        raise RuntimeError(f"Could not resolve host for index: {index_name}")
    return pc.Index(host=host)


def _minimal_index_plan(prompt_dim: Optional[int], clip_dim: Optional[int], context_dim: Optional[int]) -> Dict[str, Tuple[str, str]]:
    """
    Returns:
      {
        "prompt":  (index_name, namespace),
        "clip":    (index_name, namespace),
        "context": (index_name, namespace),
      }
    Strategy:
      - If all 3 dims match -> single index, namespaces prompt/clip/context.
      - Else if prompt_dim == context_dim -> share one index with namespaces prompt/context, clip separate.
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


def _clean_labels(x: Any, *, max_items: int = 60, max_len: int = 64) -> List[str]:
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for s in x:
        ss = str(s or "").strip()
        if not ss:
            continue
        if len(ss) > max_len:
            ss = ss[:max_len]
        out.append(ss)
        if len(out) >= max_items:
            break
    return out


def upsert_image_metadata_embeddings(jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    jobs item format expected:
      {
        "processed_id": "processed_0",
        "unique_path": "...",
        "base_context": "...",
        "prompt_embedding": [...],
        "clip_embedding": [...],
        "context_embedding": [...],
        "meta": {... small metadata ...}
      }
    """
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is missing in environment variables.")

    prompt_dim = None
    clip_dim = None
    context_dim = None

    for j in jobs:
        if prompt_dim is None:
            prompt_dim = _vec_dim(j.get("prompt_embedding"))
        if clip_dim is None:
            clip_dim = _vec_dim(j.get("clip_embedding"))
        if context_dim is None:
            context_dim = _vec_dim(j.get("context_embedding"))
        if prompt_dim and clip_dim and context_dim:
            break

    plan = _minimal_index_plan(prompt_dim, clip_dim, context_dim)

    # Persist plan so fetch can reuse the exact mapping
    try:
        PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        PLAN_PATH.write_text(
            json.dumps({k: [v[0], v[1]] for k, v in plan.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    pc = Pinecone(api_key=PINECONE_API_KEY)

    needed_indexes = {}
    for kind, (index_name, _) in plan.items():
        dim = {"prompt": prompt_dim, "clip": clip_dim, "context": context_dim}.get(kind)
        if dim is None:
            continue
        needed_indexes[index_name] = dim

    for index_name, dim in needed_indexes.items():
        _ensure_index(pc, index_name, dim)

    opened = {name: _open_index(pc, name) for name in needed_indexes.keys()}

    counts = {"prompt": 0, "clip": 0, "context": 0}

    per_kind_vectors: Dict[str, List[Dict[str, Any]]] = {"prompt": [], "clip": [], "context": []}

    for j in jobs:
        pid = str(j.get("processed_id", "")).strip()
        upath = str(j.get("unique_path", "")).strip()
        meta = j.get("meta") if isinstance(j.get("meta"), dict) else {}

        base_ctx = str(j.get("base_context", "") or "").strip()

        # -----------------------------
        # ENFORCE: labels only if diagram
        # -----------------------------
        is_diagram = 0
        try:
            is_diagram = 1 if int(meta.get("is_diagram", meta.get("diagram", 0)) or 0) == 1 else 0
        except Exception:
            is_diagram = 0

        labels = _clean_labels(meta.get("labels", None))

        base_meta = {
            "processed_id": pid,
            "unique_path": upath,
            "base_context": base_ctx,
            "is_diagram": int(is_diagram),
            "diagram": int(is_diagram),  # alias, so old code can read either
            **meta,
        }

        # Strip diagram-only payloads for non-diagrams
        if not is_diagram:
            base_meta.pop("labels", None)
            base_meta.pop("refined_labels", None)
            base_meta.pop("objects", None)
        else:
            # Normalize to ONE key for fetch: "labels"
            base_meta["labels"] = labels
            base_meta.pop("refined_labels", None)
            base_meta.pop("objects", None)

        v_prompt = j.get("prompt_embedding")
        if _vec_dim(v_prompt) is not None:
            per_kind_vectors["prompt"].append({"id": pid, "values": v_prompt, "metadata": base_meta})

        v_clip = j.get("clip_embedding")
        if _vec_dim(v_clip) is not None:
            per_kind_vectors["clip"].append({"id": pid, "values": v_clip, "metadata": base_meta})

        v_ctx = j.get("context_embedding")
        if _vec_dim(v_ctx) is not None:
            per_kind_vectors["context"].append({"id": pid, "values": v_ctx, "metadata": base_meta})

    for kind, vectors in per_kind_vectors.items():
        if not vectors:
            continue
        index_name, namespace = plan[kind]
        if index_name not in opened:
            continue
        idx = opened[index_name]

        for batch in _chunked(vectors, UPSERT_BATCH):
            idx.upsert(vectors=batch, namespace=namespace)
            counts[kind] += len(batch)

    return {
        "index_plan": plan,
        "dims": {"prompt": prompt_dim, "clip": clip_dim, "context": context_dim},
        "upserted": counts,
        "plan_path": str(PLAN_PATH),
    }