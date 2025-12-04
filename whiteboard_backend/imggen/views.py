# generator/views.py
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseNotAllowed

from pathlib import Path
import json, time, uuid, requests
from typing import List, Dict, Any, Optional

from .models import WhiteboardObject
from .utils import validate_image_json, validate_text_prompt

from Imageresearcher import research
from ImagePreproccessor import proccess_images
from ImageSkeletonizer import skeletonize_images
from ImageVectorizer import vectorize_images
from ImageVecOrganizer import organize_images


"""

"http://127.0.0.1:8000/research/"

{
"query": "eukaryotic cell",
"subject": "Biology"
}

"""

"""

http://127.0.0.1:8000/preprocess/

{
"inputdir": "ResearchImages/UniqueImages"
}

"""

# ── CONFIG ───────────────────────────────────────────────────────────────
COMFY_SERVER = "http://127.0.0.1:8188"
HTTP_TIMEOUT = 60
POLL_SLEEP   = 0.20
POLL_MAX_S   = 300

POS_NODE_ID = "8"    # CLIPTextEncode (positive)
KS_NODE_ID  = "14"   # KSampler (optional)

PAD_PROMPTS     = True
TOKENIZER_NAME  = "openai/clip-vit-base-patch32"  # switch to T5 if needed
TARGET_TOKENS   = 77

# ── CLIENT ───────────────────────────────────────────────────────────────
class ComfyClient:
    def __init__(
        self,
        server: str = COMFY_SERVER,
        http_timeout: int = HTTP_TIMEOUT,
        poll_sleep: float = POLL_SLEEP,
        poll_max_s: int = POLL_MAX_S,
        pos_node_id: str = POS_NODE_ID,
        ks_node_id: str = KS_NODE_ID,
        pad_prompts: bool = PAD_PROMPTS,
        tokenizer_name: str = TOKENIZER_NAME,
        target_tokens: int = TARGET_TOKENS,
    ):
        self.server = server.rstrip("/")
        self.http_timeout = http_timeout
        self.poll_sleep = poll_sleep
        self.poll_max_s = poll_max_s
        self.pos_node_id = pos_node_id
        self.ks_node_id  = ks_node_id
        self.pad_prompts = pad_prompts
        self.target_tokens = target_tokens

        # tokenizer is optional
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception:
            self._tokenizer = None

    # ── workflow utils ──
    @staticmethod
    def base_dir() -> Path:
        return Path(__file__).resolve().parent.parent

    @staticmethod
    def load_workflow(wf_path: Path) -> Dict[str, Any]:
        return json.loads(wf_path.read_text(encoding="utf-8"))

    def set_prompt(self, workflow: Dict[str, Any], prompt: str) -> None:
        if self.pos_node_id not in workflow:
            raise KeyError(f"Workflow missing node id '{self.pos_node_id}'")
        workflow[self.pos_node_id]["inputs"]["text"] = prompt

    def set_sampler(
        self,
        workflow: Dict[str, Any],
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
    ) -> None:
        if self.ks_node_id in workflow and isinstance(workflow[self.ks_node_id].get("inputs"), dict):
            ks = workflow[self.ks_node_id]["inputs"]
            if seed is not None:  ks["seed"]  = int(seed)
            if steps is not None: ks["steps"] = int(steps)
            if cfg is not None:   ks["cfg"]   = float(cfg)

    # ── prompt padding ──
    def pad_prompt_fixed_tokens(self, text: str) -> str:
        if not (self.pad_prompts and self._tokenizer and text):
            return text
        enc = self._tokenizer(text, truncation=True, max_length=self.target_tokens)
        ids = enc["input_ids"]
        pad_id = self._tokenizer.eos_token_id or self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = ids[-1] if ids else 0
        if len(ids) < self.target_tokens:
            ids = ids + [pad_id] * (self.target_tokens - len(ids))
        elif len(ids) > self.target_tokens:
            ids = ids[:self.target_tokens]
        return self._tokenizer.decode(ids)

    # ── comfy http ──
    def queue_workflow(self, workflow: Dict[str, Any]) -> str:
        r = requests.post(
            f"{self.server}/prompt",
            json={"prompt": workflow},
            timeout=self.http_timeout
        )
        r.raise_for_status()
        pid = r.json().get("prompt_id")
        if not pid:
            raise RuntimeError(f"No prompt_id from /prompt: {r.text}")
        return pid

    def wait_for_images(self, pid: str) -> List[Dict[str, Any]]:
        deadline = time.time() + self.poll_max_s
        while time.time() < deadline:
            h = requests.get(f"{self.server}/history/{pid}", timeout=self.http_timeout)
            h.raise_for_status()
            hist = h.json()
            rec = hist.get(pid)
            if rec:
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
                    return []
            time.sleep(self.poll_sleep)
        raise TimeoutError("Timeout waiting for outputs")

    def download_images(self, images: List[Dict[str, Any]], out_dir: Path) -> List[str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: List[str] = []
        for img in images:
            fn  = img.get("filename")
            sub = img.get("subfolder", "")
            typ = img.get("type", "output")
            if not fn:
                continue
            v = requests.get(
                f"{self.server}/view",
                params={"filename": fn, "subfolder": sub, "type": typ},
                timeout=self.http_timeout
            )
            v.raise_for_status()
            name = f"gen_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
            path = out_dir / name
            path.write_bytes(v.content)
            saved.append(str(path))
        return saved

    # ── end-to-end (one prompt) ──
    def run_once(
        self,
        wf_path: Path,
        prompt: str,
        out_dir: Path,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
    ) -> List[str]:
        wf = self.load_workflow(wf_path)
        self.set_prompt(wf, self.pad_prompt_fixed_tokens(prompt))
        self.set_sampler(wf, seed=seed, steps=steps, cfg=cfg)
        pid = self.queue_workflow(wf)
        images = self.wait_for_images(pid)
        return self.download_images(images, out_dir) if images else []

# ── VIEW ─────────────────────────────────────────────────────────────────
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

    base = ComfyClient.base_dir()
    out_dir = Path(path_out) if path_out else (base / "outputs")
    wf_path = base / "Model_Fasr.json"
    if not wf_path.exists():
        return JsonResponse({"ok": False, "error": f"Workflow not found: {wf_path}"}, status=500)

    client = ComfyClient()

    results: List[Dict[str, Any]] = []
    for i, raw_prompt in enumerate(prompts, start=1):
        p = (str(raw_prompt) or "").strip()
        if not p:
            results.append({"prompt": raw_prompt, "saved": [], "error": "empty prompt"})
            continue

        try:
            s = (int(seed) + (i - 1)) if seed is not None and str(seed).strip() != "" else None
            saved = client.run_once(
                wf_path=wf_path,
                prompt=p,
                out_dir=out_dir,
                seed=s,
                steps=int(steps) if steps not in (None, "") else None,
                cfg=float(cfg)   if cfg   not in (None, "") else None,
            )
            if saved:
                results.append({"prompt": raw_prompt, "saved": saved})
            else:
                results.append({"prompt": raw_prompt, "saved": [], "error": "no images"})
        except Exception as e:
            results.append({"prompt": raw_prompt, "saved": [], "error": str(e)})

    return JsonResponse({"ok": True, "results": results})


@csrf_exempt
def research_images(request):

    if request.method != "POST":
        return HttpResponseBadRequest("POST JSON only")

    ct = request.META.get("CONTENT_TYPE")
    raw = request.body or b""
    print("CT =", ct)
    print("LEN =", len(raw))
    print("RAW-REPR =", repr(raw[:200]))  # shows hidden chars

    try:
        body = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        return JsonResponse({
            "error": "Invalid JSON",
            "where": {"lineno": e.lineno, "colno": e.colno, "msg": e.msg},
            "debug": {
                "content_type": ct,
                "len": len(raw),
                "preview": raw[:200].decode("utf-8", errors="replace")
            }
        }, status=400)

    prompt = (body.get("query") or "").strip()
    subj   = (body.get("subject") or "").strip()
    if not prompt or not subj:
        return HttpResponseBadRequest("Missing 'query' or 'subject'")

    try:
        results = research(prompt, subj)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"ok": True, "results": results})

@csrf_exempt
def process_images(request):

    if request.method != "POST":
        return HttpResponseBadRequest("POST JSON only")
    
    ct = request.META.get("CONTENT_TYPE")
    raw = request.body or b""

    try:
        body = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        return JsonResponse({
            "error": "Invalid JSON",
            "where": {"lineno": e.lineno, "colno": e.colno, "msg": e.msg},
            "debug": {
                "content_type": ct,
                "len": len(raw),
                "preview": raw[:200].decode("utf-8", errors="replace")
            }
        }, status=400)

    inpt = (body.get("inputdir") or "").strip()
    if not inpt:
        return HttpResponseBadRequest("Missing 'inpt'")
    
    base = Path(__file__).resolve().parent.parent
    i_dir  = base / inpt
    

    proccess_images(i_dir)
    skeletonize_images()
    vectorize_images()
    organize_images()

    return JsonResponse({"ok": True})

@csrf_exempt
@require_http_methods(["POST"])
def create_image_object(request):
    """
    Body JSON:
    {
      "file_name": "edges_0_skeleton.json",
      "x": 0.0,
      "y": 0.0,
      "scale": 1.0
    }
    Looks up file in StrokeVectors, validates, then stores board object.
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    file_name = payload.get("file_name")
    if not file_name:
        return JsonResponse({"error": "file_name is required"}, status=400)

    x = float(payload.get("x", 0.0))
    y = float(payload.get("y", 0.0))
    scale = float(payload.get("scale", 1.0))

    # Validate the JSON file in StrokeVectors
    try:
        validate_image_json(file_name)
    except FileNotFoundError as e:
        return JsonResponse({"error": f"JSON file not found: {e}"}, status=404)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)

    # Upsert behavior: one object per name, just like _drawnJsonNames in UI
    obj, created = WhiteboardObject.objects.update_or_create(
        name=file_name,
        defaults={
            "kind": WhiteboardObject.KIND_IMAGE,
            "pos_x": x,
            "pos_y": y,
            "scale": scale,
            "letter_size": None,
            "letter_gap": None,
        },
    )

    return JsonResponse(
        {
            "name": obj.name,
            "kind": obj.kind,
            "pos_x": obj.pos_x,
            "pos_y": obj.pos_y,
            "scale": obj.scale,
            "created": created,
        },
        status=201 if created else 200,
    )


# ---------- 2) CREATE TEXT OBJECT (WRITE TEXT) ----------

@csrf_exempt
@require_http_methods(["POST"])
def create_text_object(request):
    """
    Body JSON:
    {
      "prompt": "Hello, world",
      "x": 0.0,
      "y": 0.0,
      "letter_size": 180.0,
      "letter_gap": 20.0
    }
    Mirrors _writeTextPrompt in spirit: stores a text 'object' on the board.
    Name = prompt, same as jsonName in your UI.
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    prompt = payload.get("prompt")
    if not prompt:
        return JsonResponse({"error": "prompt is required"}, status=400)

    x = float(payload.get("x", 0.0))
    y = float(payload.get("y", 0.0))
    letter_size = float(payload.get("letter_size", 180.0))
    letter_gap = float(payload.get("letter_gap", 20.0))

    if letter_size <= 0:
        return JsonResponse({"error": "letter_size must be > 0"}, status=400)

    # Validate that at least one glyph exists in Font
    try:
        validate_text_prompt(prompt)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)

    # Again, upsert by name, like _drawnJsonNames
    obj, created = WhiteboardObject.objects.update_or_create(
        name=prompt,
        defaults={
            "kind": WhiteboardObject.KIND_TEXT,
            "pos_x": x,
            "pos_y": y,
            "scale": 1.0,  # you can compute real scale from metrics if you care
            "letter_size": letter_size,
            "letter_gap": letter_gap,
        },
    )

    return JsonResponse(
        {
            "name": obj.name,
            "kind": obj.kind,
            "pos_x": obj.pos_x,
            "pos_y": obj.pos_y,
            "letter_size": obj.letter_size,
            "letter_gap": obj.letter_gap,
            "created": created,
        },
        status=201 if created else 200,
    )


# ---------- 3) LIST ALL OBJECTS ----------

@require_http_methods(["GET"])
def list_objects(request):
    """
    Returns:
    {
      "objects": [
        {"name": "edges_0_skeleton.json", "kind": "image"},
        {"name": "Hello, world", "kind": "text"}
      ]
    }
    """
    objs = WhiteboardObject.objects.all()
    data = [
        {
            "name": o.name,
            "kind": o.kind,
            "pos_x": o.pos_x,
            "pos_y": o.pos_y,
            "scale": o.scale,
            "letter_size": o.letter_size,
            "letter_gap": o.letter_gap,
        }
        for o in objs
    ]
    return JsonResponse({"objects": data}, status=200)


# ---------- 4) DELETE OBJECT BY NAME ----------

@csrf_exempt
@require_http_methods(["DELETE"])
def delete_object(request):
    """
    Body JSON:
    {
      "name": "edges_0_skeleton.json"
    }
    or
    {
      "name": "Hello, world"
    }
    Deletes a single WhiteboardObject row (does NOT delete files on disk).
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    name = payload.get("name")
    if not name:
        return JsonResponse({"error": "name is required"}, status=400)

    try:
        obj = WhiteboardObject.objects.get(name=name)
    except WhiteboardObject.DoesNotExist:
        return JsonResponse({"error": "Object not found"}, status=404)

    obj.delete()
    return JsonResponse({"deleted": name}, status=200)