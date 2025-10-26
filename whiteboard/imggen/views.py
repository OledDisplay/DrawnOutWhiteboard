# generator/views.py
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponseBadRequest
from pathlib import Path
import json, time, uuid, requests
from typing import List, Dict, Any, Optional

# ── COMFY / PIPELINE CONFIG ──────────────────────────────────────────────
COMFY_SERVER = "http://127.0.0.1:8188"
HTTP_TIMEOUT = 60          # per request to Comfy
POLL_SLEEP   = 0.20        # seconds between /history polls
POLL_MAX_S   = 300         # hard cap per job

# Your workflow JSON must exist at <project_root>/Model_Fasr.json
# Adjust node IDs (they are STRING KEYS in the JSON!)
POS_NODE_ID = "8"          # CLIPTextEncode (positive)
KS_NODE_ID  = "14"         # KSampler (optional tuning: seed/steps/cfg)

# ── PROMPT PADDING (locks token shapes) ──────────────────────────────────
# If your text encoder is CLIP: 77 tokens. If it’s T5 (some Flux stacks): 256.
PAD_PROMPTS     = True
TOKENIZER_NAME  = "openai/clip-vit-base-patch32"  # CLIP
TARGET_TOKENS   = 77

# If you confirm you’re on a T5 text encoder, switch to:
# TOKENIZER_NAME = "google/t5-v1_1-base"
# TARGET_TOKENS  = 256

try:
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
except Exception:
    _TOKENIZER = None  # padding will no-op if transformers not installed

def pad_prompt_fixed_tokens(text: str, target_len: int = TARGET_TOKENS) -> str:
    """Pad/truncate to fixed token length so attention shapes don’t change."""
    if not PAD_PROMPTS or not _TOKENIZER or not text:
        return text
    enc = _TOKENIZER(text, truncation=True, max_length=target_len)
    ids = enc["input_ids"]
    pad_id = _TOKENIZER.eos_token_id or _TOKENIZER.pad_token_id
    if pad_id is None:
        pad_id = ids[-1] if ids else 0
    if len(ids) < target_len:
        ids = ids + [pad_id] * (target_len - len(ids))
    elif len(ids) > target_len:
        ids = ids[:target_len]
    return _TOKENIZER.decode(ids)

# ── HELPER FUNCTIONS ─────────────────────────────────────────────────────
def _base_dir() -> Path:
    return Path(__file__).resolve().parent.parent

def _load_workflow(wf_path: Path) -> Dict[str, Any]:
    return json.loads(wf_path.read_text(encoding="utf-8"))

def _set_prompt(workflow: Dict[str, Any], prompt: str) -> None:
    if POS_NODE_ID not in workflow:
        raise KeyError(f"Workflow missing node id '{POS_NODE_ID}'")
    workflow[POS_NODE_ID]["inputs"]["text"] = prompt

def _set_sampler(workflow: Dict[str, Any],
                 seed: Optional[int] = None,
                 steps: Optional[int] = None,
                 cfg: Optional[float] = None) -> None:
    if KS_NODE_ID in workflow and isinstance(workflow[KS_NODE_ID].get("inputs"), dict):
        ks = workflow[KS_NODE_ID]["inputs"]
        if seed is not None:  ks["seed"]  = int(seed)
        if steps is not None: ks["steps"] = int(steps)
        if cfg is not None:   ks["cfg"]   = float(cfg)

def _queue_workflow(workflow: Dict[str, Any]) -> str:
    r = requests.post(f"{COMFY_SERVER}/prompt",
                      json={"prompt": workflow},
                      timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    pid = r.json().get("prompt_id")
    if not pid:
        raise RuntimeError(f"No prompt_id from /prompt: {r.text}")
    return pid

def _wait_for_images(pid: str) -> List[Dict[str, Any]]:
    """Poll /history until images appear OR status==completed/error OR timeout."""
    deadline = time.time() + POLL_MAX_S
    while time.time() < deadline:
        h = requests.get(f"{COMFY_SERVER}/history/{pid}", timeout=HTTP_TIMEOUT)
        h.raise_for_status()
        hist = h.json()
        rec = hist.get(pid)
        if rec:
            # collect images if present NOW (break immediately)
            outs = rec.get("outputs") or {}
            imgs: List[Dict[str, Any]] = []
            for node_out in outs.values():
                imgs += node_out.get("images", [])
            if imgs:
                return imgs
            status = rec.get("status")
            if status == "error":
                raise RuntimeError(rec.get("error", "Comfy error"))
            if status == "completed":
                # completed but no images -> let caller decide
                return []
        time.sleep(POLL_SLEEP)
    raise TimeoutError("Timeout waiting for outputs")

def _download_images(images: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for img in images:
        fn  = img.get("filename")
        sub = img.get("subfolder", "")
        typ = img.get("type", "output")
        if not fn:
            continue
        v = requests.get(f"{COMFY_SERVER}/view",
                         params={"filename": fn, "subfolder": sub, "type": typ},
                         timeout=HTTP_TIMEOUT)
        v.raise_for_status()
        name = f"gen_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
        path = out_dir / name
        path.write_bytes(v.content)
        saved.append(str(path))
    return saved

# ── MULTI-IMAGE (SEQUENTIAL) ENDPOINT ────────────────────────────────────
@csrf_exempt
def generate_images_batch(request):
    """
    POST /generate_batch/
    {
      "prompts": ["p1","p2", ...],            // REQUIRED
      "path_out": "outputs",                  // optional dir
      "seed": 12345, "steps": 4, "cfg": 1.0   // optional; seed increments per prompt
    }
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST JSON only")

    try:
        body = json.loads((request.body or b"{}").decode("utf-8"))
    except json.JSONDecodeError:
        return HttpResponseBadRequest("Invalid JSON")

    prompts = body.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        return HttpResponseBadRequest("'prompts' must be a non-empty list")

    path_out = (body.get("path_out") or "").strip()
    seed  = body.get("seed")
    steps = body.get("steps")
    cfg   = body.get("cfg")

    base = _base_dir()
    out_dir = Path(path_out) if path_out else (base / "outputs")
    wf_path = base / "Model_Fasr.json"
    if not wf_path.exists():
        return JsonResponse({"ok": False, "error": f"Workflow not found: {wf_path}"}, status=500)

    results: List[Dict[str, Any]] = []
    for i, raw_prompt in enumerate(prompts, start=1):
        p = (str(raw_prompt) or "").strip()
        if not p:
            results.append({"prompt": raw_prompt, "saved": [], "error": "empty prompt"})
            continue

        # pad to fixed tokens
        p = pad_prompt_fixed_tokens(p)

        try:
            wf = _load_workflow(wf_path)
            _set_prompt(wf, p)

            # optional sampler (seed increments to vary images)
            s = (int(seed) + (i - 1)) if seed is not None and str(seed).strip() != "" else None
            _set_sampler(
                wf,
                seed=s,
                steps=int(steps) if steps is not None and str(steps).strip() != "" else None,
                cfg=float(cfg)   if cfg   is not None and str(cfg).strip()   != "" else None,
            )

            pid = _queue_workflow(wf)
            images = _wait_for_images(pid)
            if not images:
                results.append({"prompt": raw_prompt, "saved": [], "error": "no images"})
                continue
            saved = _download_images(images, out_dir)
            results.append({"prompt": raw_prompt, "saved": saved})
        except Exception as e:
            results.append({"prompt": raw_prompt, "saved": [], "error": str(e)})

    return JsonResponse({"ok": True, "results": results})
