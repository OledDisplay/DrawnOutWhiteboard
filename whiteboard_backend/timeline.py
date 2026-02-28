"""
timeline_pipeline.py

3-stage lesson timeline pipeline:
1) topic -> logical timeline split into chapters with '|'
2) each chapter -> teacher-style narration with pause markers like '>3.0'
3) each narration chunk -> JSON image plan with diagram flag + word index ranges

PLUS:
4) Integrates with ImagePipeline orchestrator:
   - Loads SigLIP+MiniLM while GPT calls run
   - When image collection is needed:
       - Pinecone fetch first
       - If misses:
           - Load Qwen in parallel with ImageResearcher.research_many
           - Start ImagePipeline background run using workers
           - Wait selection_done (selection + refine) -> attach image ids + refined labels
           - THEN (new) run Qwen ACTION PLANNING per image+silence with a whiteboard simulator
           - Finally wait colours_done (pipeline finished)

Requires:
  pip install openai pillow
Env:
  OPEN_AI_KEY (preferred in this file) or OPENAI_API_KEY
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typing import DefaultDict



from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from openai import OpenAI


# ----------------------------
# Models / client
# ----------------------------

DEFAULT_MODEL_CANDIDATES = [
    "gpt-5-mini"
]


def build_client() -> OpenAI:
    api_key = os.getenv("OPEN_AI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set env var OPEN_AI_KEY (or OPENAI_API_KEY).")
    return OpenAI(api_key=api_key)


def call_responses_text(
    client: OpenAI,
    model_candidates: List[str],
    input_items: List[Dict[str, str]],
    *,
    reasoning_effort: str = "low",
    temperature: float = 0.4,
    max_output_tokens: Optional[int] = None,
    text_format: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    Tries models in order. Returns (output_text, model_used).
    """
    last_err: Optional[Exception] = None

    for model in model_candidates:
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "input": input_items,
                "reasoning": {"effort": reasoning_effort},
            }
            if max_output_tokens is not None:
                kwargs["max_output_tokens"] = max_output_tokens
            if text_format is not None:
                kwargs["text"] = text_format
            resp = client.responses.create(**kwargs)
            return resp.output_text, model
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"All model candidates failed. Last error: {last_err!r}")


def _run_efficientsam3_bboxes(
    self,
    *,
    assets_by_name: Dict[str, ImageAsset],
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builds {processed_id: refined_labels(list[str])} for diagram images only,
    runs EfficientSAM3 to get bboxes per label, returns map.
    """
    from shared_models import get_efficientsam3, sam_lock
    from EfficientSAM3Clusters import EfficientSAM3Bundle, compute_label_bboxes_for_processed_images

    pid_to_labels: Dict[str, List[str]] = {}

    for name, a in assets_by_name.items():
        if int(a.diagram) != 1:
            continue
        if not a.processed_ids:
            continue
        pid = str(a.processed_ids[0] or "").strip()
        if not pid:
            continue
        if not a.refined_labels_file:
            continue

        # your refined file format in ImagePipeline: dict with "objects": list[str] AND "refined_labels": list[str]
        try:
            data = json.loads(Path(a.refined_labels_file).read_text(encoding="utf-8"))
        except Exception:
            continue

        labels = []
        if isinstance(data, dict):
            # prefer "objects" (your new semantic name), fall back to refined_labels
            raw = data.get("objects")
            if not isinstance(raw, list):
                raw = data.get("refined_labels")
            if isinstance(raw, list):
                for s in raw:
                    ss = str(s or "").strip()
                    if ss:
                        labels.append(ss)

        if labels:
            pid_to_labels[pid] = labels

    if not pid_to_labels:
        return {"ok": True, "note": "no_diagram_refined_labels"}

    sam = get_efficientsam3()
    if sam is None:
        return {"ok": False, "err": "efficientsam3_not_loaded"}

    bundle = EfficientSAM3Bundle(model=sam.model, processor=sam.processor, device=sam.device, model_id=sam.model_id)

    # write into PipelineOutputs by default (same place ImagePipeline uses)
    save_path = None
    if out_dir:
        save_path = str(Path(out_dir) / "efficientsam3_label_bboxes.json")
    else:
        save_path = str(Path("PipelineOutputs") / "efficientsam3_label_bboxes.json")

    with sam_lock():
        bbox_map = compute_label_bboxes_for_processed_images(
            bundle=bundle,
            processed_id_to_labels=pid_to_labels,
            processed_images_roots=None,
            top_k=3,
            min_score=0.0,
            save_json_path=save_path,
        )

    return {"ok": True, "save_json": save_path, "result": bbox_map}


# ----------------------------
# Prompts
# ----------------------------

CHAPTER_CAP = 3
MAX_WORDS_PER_CHAPTER = 200
MAX_DIAGRAMS_PER_CHAPTER = 2




def prompt_logical_timeline(topic: str) -> List[Dict[str, str]]:
    developer = (
        "You are a lesson-planning engine. You must output a FULL logical lesson timeline "
        "for the given topic.\n\n"
        "Definition of 'logical timeline': every small step a good teacher would do in order, "
        "including: introduction, framing what will be explored, transitions, pacing notes, "
        "examples, mini-check questions, and a short recap.\n\n"
        "Cover every micro part and step in explaining a concept - in a litteral deep analytical way,"
        "You MUST split the lesson into multiple chapters (separate logical parts). "
        "Chapters MUST be separated by a line that contains ONLY this character:\n"
        "|\n\n"
        "Inside each chapter:\n"
        "- Start with: CHAPTER: <short title>\n"
        "- Then provide numbered steps: 1., 2., 3., ...\n"
        "- Each step must be explicit and actionable (like a robot could follow it).\n"
        "- Include the content 'meat' (what is actually taught), not just headings.\n"
        "- Include pacing hints in-line like: (pace: slow) / (pace: medium) / (pace: fast)\n\n"
        "Try to not have too many chapters (3-6 is good), but do cover the topic fully.\n"
        "Each chapter should be a concrete seperated part of the lesson.\n\n"
        f"Current chapter cap : {CHAPTER_CAP}"
        "Hard rules:\n"
        "- Do NOT use the '|' character anywhere except the chapter separator line.\n"
        "- Do NOT output markdown code fences.\n"
    )
    user = f"Topic: {topic}\nGenerate the full logical timeline now."
    return [{"role": "developer", "content": developer}, {"role": "user", "content": user}]


def prompt_teacher_speech(topic: str, chapter_timeline: str, chapter_index: int, chapter_count: int) -> List[Dict[str, str]]:
    developer = (
        "You are generating the SPOKEN narration for a lesson.\n\n"
        "Input is a single chapter of a 'logical timeline' (robotic steps). "
        "You must convert it into natural teacher speech that follows those steps closely.\n\n"
        "Requirements:\n"
        "- Explain extensively and in-depth. Cover every nuance mentioned in the timeline.\n"
        "- Keep it natural like a teacher talking to students.\n"
        "- Smooth transitions.\n"
        "- Include occasional short rhetorical questions or checks for understanding.\n\n"
        "Pause markers:\n"
        "- Insert pauses using EXACTLY this format: >3.0 where 3.0 is a float seconds value, and \">\" is your special char.\n"
        "- Place pause markers between sentences/paragraphs, not inside words.\n"
        "- Use them at chapter sections, after dense explanations, or before a key idea.\n"
        "- Typical values: 1.0 to 3.0.\n\n"
        f"word cap : {MAX_WORDS_PER_CHAPTER}"
        "Hard rules:\n"
        "- Output ONLY the narration text (no titles, no step numbers, no bullet lists).\n"
        "- Do NOT output markdown code fences.\n"
    )
    user = (
        f"Topic: {topic}\n"
        f"Chapter: {chapter_index}/{chapter_count}\n\n"
        "Chapter timeline:\n"
        f"{chapter_timeline}\n\n"
        "Generate the full narration for this chapter only."
    )
    return [{"role": "developer", "content": developer}, {"role": "user", "content": user}]


IMAGE_PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "word_index_base": {"type": "integer", "enum": [0]},
        "end_index_inclusive": {"type": "boolean", "enum": [True]},
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "minLength": 1},
                    "topic": {"type": "string", "minLength": 1},
                    "diagram": {"type": "integer", "enum": [0, 1]},
                    # NEW: text tag (0/1). When 1, this entry is board text (not an image request).
                    "text": {"type": "integer", "enum": [0, 1]},
                    "diagram_required_objects": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "range_start": {"type": "integer", "minimum": 0},
                    "range_end": {"type": "integer", "minimum": 0},
                },
                "required": ["content", "topic", "diagram", "text", "diagram_required_objects", "range_start", "range_end"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["word_index_base", "end_index_inclusive", "images"],
    "additionalProperties": False,
}


def prompt_image_requests(speech_text: str) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    developer = (
        "You generate image requests for a lesson background/whiteboard animation.\n\n"
        "Given narration text, identify places where a concept/object should appear visually.\n"
        "Do NOT be obnoxious: pick visuals that actually help understanding, like a teacher drawing.\n\n"
        "In every place where an image could link to speech well, do request an image"
        "Diagram behavior:\n"
        "- Some concepts can persist for a while, using or being represented with some image, so leave a special image there \n"
        "- Mark those as diagram=1.\n"
        "- Anything that will actually be interacted with after the fisrt seconds of being drawn is a diagram = 1"
        "- you SHOULDNT constantly request diagrams though as they take ~30secs to load -> try and use them"
        "  only after some \"time\" has passed in the chunk."
        "- Small one-off visuals are diagram=0.\n\n"
        "Sync behavior:\n"
        "- You must sync each image to the narration via word indices.\n"
        "- Indices are ZERO-BASED word positions after splitting the entire narration by whitespace.\n"
        "- Provide range_start and range_end (end is INCLUSIVE) -> end should be only where the image."
        "You can also request TEXT to be written on the board:\n"
        "- use the same structure as requesting an image, but with text = 1\n" 
        "- Text is the most powerful tool for good sync, its cheap, so request it often, for" \
        "title, labels, keywords, formulas, numbers, etc. that appear in the narration and should be on the board\n"
        "- you need to still do word index sync with text requests, just like images\n"
        "- Text should be literral stirngs to be printed on the board -> not \"lesson\" map -> its text that'll be written 1:1\n"
        "- PUT NOTHING ELSE BUT THE TEXT TO PRINT IN THE CONTENT FIELD"
        "stops being relevant, so try persist solid ranges\n\n"
        "DIAGRAM CONSOLIDATION / PARTS RULES:\n"
        "- If multiple related concepts can be explained using one shared diagram, prefer ONE diagram over many separate visuals.\n"
        "- When diagram=1, you must populate 'diagram_required_objects' with the exact parts/sub-objects that will be referenced or interacted with.\n"
        "- If diagram=0, set 'diagram_required_objects' to [].\n"
        "- Keep a diagram active (large relevant range) while it can still support explanation.\n"
        "- Just try and force a digram to optimize a low image count.\n"
        "- Reverse rule: if nearby concepts can logically unify into one object/scene, request a combined diagram that contains all of them.\n"
        "- Explicit example: organelles explanation -> one eukaryotic cell diagram + diagram_required_objects list of mentioned organelles.\n"
        "- Explicit example: Pythagorean theorem -> one triangle diagram + diagram_required_objects like side a, side b, side c.\n"
        "Extra behaviour rules:\n"
        "- If not many images try to at least request text entries for important keywords/concepts OFTEN\n"
        "- You are aiming for a rich visualization of what is said so just animate a lot\n"
        "- You are trying to have 15 + objects per chunk - USE A LOT OF TEXT, A LOT OF NON DIAGRAM IMAGES"
        f"Diagram cap: {MAX_DIAGRAMS_PER_CHAPTER}\n"
        "Output in content:\n"
        "- Use ONE field only: 'content'.\n"
        "- If image (text=0): content is the concrete object/query.\n"
        "- Fill content only clear optimized queries for JUST the object -> no extra explanations or detail"
        "- Request just plain REAL objects without ANY SORT of EXPLANATIONS in content - NO SPECIFICATIONS FOR STYLE OR DETAIL, JUST AN OPTIMIZED SHORT QUERY"
        "- If text (text=1): content is the exact text to write on the board.\n"
        "Topic behavior:\n"
        "- You must include a 'topic' for every entry.\n"
        "- For image entries (text=0), topic must be the broader subject umbrella that contains the specific concept.\n"
        "- Keep topic specific enough to be useful, but still general enough to include nearby concepts.\n"
        "- Example: image content='map of the German invasion of France in WW2' -> topic='WW2 History of Europe'.\n"
        "- For text entries (text=1), set topic to a stable lesson-level umbrella topic too (not empty).\n"
        "- Output must follow the provided JSON schema exactly.\n"
        "- Keep requested images as concrete things, easily got with img generation / search engines.\n"
        "- Have different objects all around -> do not request the same thing in different places -> if it's used for a while JUST GIVE A BIG RANGE FOR IT\n"
        "\n\n"
    )

    user = (
        "Narration text:\n"
        f"{speech_text}\n\n"
        "Return the JSON image plan now."
    )

    input_items = [{"role": "developer", "content": developer}, {"role": "user", "content": user}]
    text_format = {
        "format": {
            "type": "json_schema",
            "name": "image_plan",
            "schema": IMAGE_PLAN_SCHEMA,
            "strict": True,
        }
    }
    return input_items, text_format


# ----------------------------
# Parsing / validation
# ----------------------------

@dataclass
class Chapter:
    raw: str


def split_chapters(timeline_text: str) -> List[Chapter]:
    text = timeline_text.strip()
    parts = re.split(r"\n\s*\|\s*\n", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1 and "|" in text:
        parts = [p.strip() for p in text.split("|") if p.strip()]
    return [Chapter(raw=p) for p in parts]


def clamp_image_ranges(plan: Dict[str, Any], speech_text: str) -> Dict[str, Any]:
    words = speech_text.split()
    n = len(words)
    if n == 0:
        return plan

    for img in plan.get("images", []):
        s = int(img["range_start"])
        e = int(img["range_end"])
        s = max(0, min(s, n - 1))
        e = max(0, min(e, n - 1))
        if e < s:
            s, e = e, s
        img["range_start"] = s
        img["range_end"] = e

    return plan


def build_image_text_maps(speech_text: str, image_plan: dict) -> list[dict]:
    words = speech_text.split()
    n = len(words)

    out: list[dict] = []
    for img in image_plan.get("images", []):
        s = int(img["range_start"])
        e = int(img["range_end"])

        s = max(0, min(s, n - 1)) if n else 0
        e = max(0, min(e, n - 1)) if n else -1
        if e < s:
            s, e = e, s

        span_text = " ".join(words[s : e + 1]) if (n and e >= 0) else ""

        content = str(img.get("content", "") or "").strip()
        topic = str(img.get("topic", "") or "").strip()
        diagram_required_objects = img.get("diagram_required_objects") or []
        if not isinstance(diagram_required_objects, list):
            diagram_required_objects = []
        diagram_required_objects = [str(x).strip() for x in diagram_required_objects if str(x).strip()]

        # NEW: text tag handling
        text_tag = 1 if int(img.get("text", 0) or 0) == 1 else 0
        write_text = content if text_tag == 1 else ""

        # For images: query is used for search/pinecone.
        # For text-tag entries: they are NOT image prompts, so query must be empty.
        query = ""
        if text_tag == 0:
            query = content  # THIS is what should be used for search/pinecone

        # Diagram must be 0 for text entries.
        diag = int(img.get("diagram", 0) or 0)
        if text_tag == 1:
            diag = 0

        out.append(
            {
                "content": content,
                "topic": topic,
                "query": query,                 # NEW (empty when text_tag==1)
                "span_text": span_text,         # NEW (was "text")
                "range_start": s,
                "range_end": e,
                "diagram": diag,
                "diagram_required_objects": diagram_required_objects,
                "text_tag": text_tag,           # NEW
                "write_text": write_text,       # NEW (what to write on board)
            }
        )

    return out


# ----------------------------
# Image orchestration integration
# ----------------------------

@dataclass
class ImageAsset:
    prompt: str
    diagram: int
    processed_ids: List[str]
    refined_labels_file: Optional[str]  # path to refined labels json, if produced
    bbox_px: Optional[Tuple[int, int]] = None
    objects: Optional[List[Dict[str, Any]]] = None  # diagram objects that comprise the image


def _dbg(self, msg: str, *, data: Any = None) -> None:
    if not self.debug_print:
        return
    if data is None:
        print(f"[timeline][DBG] {msg}")
        return
    try:
        s = json.dumps(data, ensure_ascii=False)
        if len(s) > 1500:
            s = s[:1500] + " ...<truncated>"
        print(f"[timeline][DBG] {msg} | {s}")
    except Exception:
        print(f"[timeline][DBG] {msg} | <unserializable>")

def _write_json_file(self, path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


# ----------------------------
# Whiteboard state simulator
# ----------------------------

@dataclass
class BoardObject:
    name: str
    kind: str  # "image" | "text"
    x: float
    y: float
    w: float
    h: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


class WhiteboardState:
    """
    Global whiteboard simulator. Coordinates are "whiteboard coords" (pixel-like).
    """
    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        self.width = float(width)
        self.height = float(height)
        self.objects: Dict[str, BoardObject] = {}
        self.links: List[Dict[str, str]] = []
        self._text_id = 0

    def snapshot_for_prompt(self) -> Dict[str, Any]:
        objs = []
        for o in self.objects.values():
            objs.append(
                {
                    "name": o.name,
                    "kind": o.kind,
                    "x": round(o.x, 2),
                    "y": round(o.y, 2),
                    "w": round(o.w, 2),
                    "h": round(o.h, 2),
                    "bbox": [round(v, 2) for v in o.bbox()],
                    "meta": o.meta,
                }
            )
        return {
            "whiteboard_width": self.width,
            "whiteboard_height": self.height,
            "objects": objs,
            "links": list(self.links),
        }

    def _cull_offscreen(self) -> None:
        to_del = []
        for name, o in self.objects.items():
            x1, y1, x2, y2 = o.bbox()
            # fully outside
            if x2 < 0 or y2 < 0 or x1 > self.width or y1 > self.height:
                to_del.append(name)
        for name in to_del:
            self.objects.pop(name, None)

    @staticmethod
    def _bboxes_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        if ax1 <= bx0 or bx1 <= ax0:
            return False
        if ay1 <= by0 or by1 <= ay0:
            return False
        return True

    def find_empty_spot(self, w: float, h: float, *, padding: float = 20.0, step: float = 80.0) -> Optional[Tuple[float, float]]:
        """
        Very simple packing: scan grid for a non-overlapping top-left location within board bounds.
        """
        if w <= 0 or h <= 0:
            return None

        max_x = max(0.0, self.width - w - padding)
        max_y = max(0.0, self.height - h - padding)

        y = padding
        while y <= max_y:
            x = padding
            while x <= max_x:
                bb = (x, y, x + w, y + h)
                ok = True
                for o in self.objects.values():
                    if self._bboxes_overlap(bb, o.bbox()):
                        ok = False
                        break
                if ok:
                    return (float(x), float(y))
                x += step
            y += step
        return None

    def apply_base_action(self, action: Dict[str, Any], image_size_lookup: Dict[str, Tuple[int, int]]) -> None:
        t = str(action.get("type", "") or "").strip().lower()

        if t == "shift":
            dx = float(action.get("dx", 0.0) or 0.0)
            dy = float(action.get("dy", 0.0) or 0.0)
            for o in self.objects.values():
                o.x += dx
                o.y += dy
            self._cull_offscreen()
            return

        if t == "erase":
            target = str(action.get("target", "") or "").strip()
            if target:
                self.objects.pop(target, None)
            return

        if t == "draw":
            target = str(action.get("target", "") or "").strip()
            if not target:
                return
            x = float(action.get("x", 0.0) or 0.0)
            y = float(action.get("y", 0.0) or 0.0)
            w, h = image_size_lookup.get(target, (400, 300))
            meta = dict(action.get("meta", {}) or {})
            self.objects[target] = BoardObject(
                name=target,
                kind="image",
                x=x,
                y=y,
                w=float(w),
                h=float(h),
                meta=meta,
            )
            self._cull_offscreen()
            return

        if t == "move":
            target = str(action.get("target", "") or "").strip()
            if not target or target not in self.objects:
                return
            x = float(action.get("x", 0.0) or 0.0)
            y = float(action.get("y", 0.0) or 0.0)
            self.objects[target].x = x
            self.objects[target].y = y
            self._cull_offscreen()
            return

        if t == "write":
            txt = str(action.get("text", "") or "")
            x = float(action.get("x", 0.0) or 0.0)
            y = float(action.get("y", 0.0) or 0.0)
            scale = float(action.get("scale", 1.0) or 1.0)

            # NEW: allow stable naming via optional "target"
            target = str(action.get("target", "") or "").strip()
            if target:
                name = target
            else:
                self._text_id += 1
                name = f"text_{self._text_id:04d}"

            # crude bbox estimate for spacing (NEW: 15px per letter)
            w = max(20.0, float(len(txt)) * 15.0 * scale)
            h = max(16.0, 30.0 * scale)
            self.objects[name] = BoardObject(
                name=name,
                kind="text",
                x=x,
                y=y,
                w=w,
                h=h,
                meta={"text": txt, "scale": scale},
            )
            self._cull_offscreen()
            return

        if t == "link":
            a = str(action.get("a", "") or "").strip()
            b = str(action.get("b", "") or "").strip()
            if a and b:
                self.links.append({"a": a, "b": b})
            return

    @staticmethod
    def parse_actions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Accepts either:
          - payload["actions"] list of dicts
          - payload["actions"] list of strings, parsed into dicts
        """
        out: List[Dict[str, Any]] = []
        acts = payload.get("actions")
        if not isinstance(acts, list):
            return out

        for a in acts:
            if isinstance(a, dict):
                if "type" in a:
                    out.append(a)
                continue

            if not isinstance(a, str):
                continue

            s = a.strip()

            # draw [name] at : x y
            m = re.match(r"^draw\s*\[(.+?)\]\s*at\s*:?\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$", s, re.I)
            if m:
                out.append({"type": "draw", "target": m.group(1).strip(), "x": float(m.group(2)), "y": float(m.group(3))})
                continue

            # erase [name]
            m = re.match(r"^erase\s*\[(.+?)\]\s*$", s, re.I)
            if m:
                out.append({"type": "erase", "target": m.group(1).strip()})
                continue

            # shift whiteboard horizontally / vertically [ amount ]
            m = re.match(r"^shift\s*whiteboard\s*(horizontally|vertically)\s*\[\s*(-?\d+(?:\.\d+)?)\s*\]\s*$", s, re.I)
            if m:
                axis = m.group(1).lower()
                amt = float(m.group(2))
                out.append({"type": "shift", "dx": amt if axis.startswith("h") else 0.0, "dy": amt if axis.startswith("v") else 0.0})
                continue

            # link [a] to [b]
            m = re.match(r"^link\s*\[(.+?)\]\s*to\s*\[(.+?)\]\s*$", s, re.I)
            if m:
                out.append({"type": "link", "a": m.group(1).strip(), "b": m.group(2).strip()})
                continue

            # write [text] at x , y  scale = float
            m = re.match(r"^write\s*\[(.+?)\]\s*at\s*(-?\d+(?:\.\d+)?)\s*,?\s*(-?\d+(?:\.\d+)?)(?:\s*scale\s*=\s*([0-9.]+))?\s*$", s, re.I)
            if m:
                scale = float(m.group(4)) if m.group(4) else 1.0
                out.append({"type": "write", "text": m.group(1), "x": float(m.group(2)), "y": float(m.group(3)), "scale": scale})
                continue

            # move [name] to x, y
            m = re.match(r"^move\s*\[(.+?)\]\s*to\s*(-?\d+(?:\.\d+)?)\s*,?\s*(-?\d+(?:\.\d+)?)\s*$", s, re.I)
            if m:
                out.append({"type": "move", "target": m.group(1).strip(), "x": float(m.group(2)), "y": float(m.group(3))})
                continue

        return out


# ----------------------------
# Action events
# ----------------------------

@dataclass
class ActionEvent:
    kind: str  # "image" | "silence"
    chapter_index: int
    start_word_index: int
    duration_sec: float
    end_word_index: Optional[int] = None
    # for image
    image_name: Optional[str] = None
    image_text: Optional[str] = None
    diagram: int = 0
    processed_id: Optional[str] = None
    bbox_px: Optional[Tuple[int, int]] = None
    objects: Optional[List[Dict[str, Any]]] = None
    refined_labels_file: Optional[str] = None
    # NEW: text-tag for board writes (still uses kind="image")
    text_tag: int = 0
    write_text: Optional[str] = None
    # for silence
    context_before: Optional[str] = None
    context_after: Optional[str] = None


def _is_pause_token(tok: str) -> Optional[float]:
    m = re.fullmatch(r">(\d+(?:\.\d+)?)", tok.strip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _strip_pause_tokens(text: str) -> str:
    # remove standalone >3.0 tokens
    toks = text.split()
    kept = []
    for t in toks:
        if _is_pause_token(t) is None:
            kept.append(t)
    return " ".join(kept)


def build_events_for_chapter(
    speech_text: str,
    image_text_maps: List[Dict[str, Any]],
    chapter_index: int,
    assets_by_name: Dict[str, ImageAsset],
) -> List[ActionEvent]:
    words = speech_text.split()
    n = len(words)

    events: List[ActionEvent] = []

    # image events: by range_start order
   # image events: by range_start order
    for im in (image_text_maps or []):
        content = str(im.get("content", "") or "").strip()
        name = content

        s = int(im.get("range_start", 0) or 0)
        s = max(0, min(s, max(0, n - 1))) if n else 0

        e = int(im.get("range_end", s) or s)
        e = max(0, min(e, max(0, n - 1))) if n else s
        if e < s:
            s, e = e, s

        # NEW: text tag
        text_tag = int(im.get("text_tag", 0) or 0)
        if text_tag == 1:
            # For text entries, always source text content from content/write_text payload.
            write_text = str(im.get("write_text", "") or content).strip()
            if not name:
                name = write_text[:48] if write_text else "text_object"
        else:
            write_text = None
            if not name:
                continue

        asset = None if text_tag == 1 else assets_by_name.get(name)
        processed_id = None
        bbox_px = None
        objects = None
        refined_file = None
        diag = int(im.get("diagram", 0) or 0)

        # Diagram must be 0 for text entries
        if text_tag == 1:
            diag = 0

        if asset:
            processed_id = asset.processed_ids[0] if asset.processed_ids else None
            bbox_px = asset.bbox_px
            objects = asset.objects
            refined_file = asset.refined_labels_file

        span_text = str(im.get("span_text", "") or "")
        span_text = _strip_pause_tokens(span_text)

        # duration heuristic:
        # - for text writes: use write length
        # - for images: use narration span
        if text_tag == 1 and write_text:
            wc = max(1, len(write_text.split()))
            dur = max(1.0, min(6.0, wc / 2.0))
        else:
            wc = len(span_text.split())
            dur = max(1.5, min(10.0, wc / 2.5))

        events.append(
            ActionEvent(
                kind="image",
                chapter_index=chapter_index,
                start_word_index=s,
                end_word_index=e,                 # NEW
                duration_sec=float(dur),
                image_name=name,
                image_text=span_text,
                diagram=diag,
                processed_id=processed_id,
                bbox_px=bbox_px,
                objects=objects,
                refined_labels_file=refined_file,
                text_tag=text_tag,                # NEW
                write_text=write_text,            # NEW
            )
        )

    # silence events: each pause token
    for idx, tok in enumerate(words):
        sec = _is_pause_token(tok)
        if sec is None:
            continue
        before = " ".join(words[max(0, idx - 20) : idx])
        after = " ".join(words[idx + 1 : min(n, idx + 21)])
        before = _strip_pause_tokens(before)
        after = _strip_pause_tokens(after)
        events.append(
            ActionEvent(
                kind="silence",
                chapter_index=chapter_index,
                start_word_index=idx,
                end_word_index=idx,               # NEW
                duration_sec=float(sec),
                context_before=before,
                context_after=after,
            )
        )

    # end-of-chunk 3 sec silence
    tail = " ".join(words[max(0, n - 25) : n])
    tail = _strip_pause_tokens(tail)
    events.append(
        ActionEvent(
            kind="silence",
            chapter_index=chapter_index,
            start_word_index=n + 1,
            duration_sec=3.0,
            context_before=tail,
            context_after="",
        )
    )

    # ordering: by position; silences first if same index
    events.sort(key=lambda e: (e.start_word_index, 0 if e.kind == "silence" else 1))
    return events

def _merge_unique_str_list(dst: List[str], src: List[str], *, cap: int = 80) -> List[str]:
        """
        In-place unique merge (case-insensitive), preserves order, clamps length.
        Returns dst for convenience.
        """
        if not isinstance(dst, list):
            dst = []
        if not isinstance(src, list):
            return dst

        seen = {str(x).strip().lower() for x in dst if str(x).strip()}
        for it in src:
            s = str(it or "").strip()
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            dst.append(s)
            seen.add(k)
            if cap and len(dst) >= cap:
                break
        return dst


# ----------------------------
# Pipeline
# ----------------------------

class LessonTimelinePipeline:
    def __init__(
    self,
    *,
    model_candidates: Optional[List[str]] = None,
    reasoning_effort: str = "low",
    temperature: float = 0.4,
    gpu_index: int = 0,
    cpu_threads: int = 4,
    whiteboard_width: int = 1920,
    whiteboard_height: int = 1080,
    processed_images_dir: str = "ProcessedImages",
    debug_timeline_json: bool = False,     # NEW: controls extra debug fields in OUTPUT JSON
    debug_print: bool = True,              # NEW: heavy console debug
    debug_out_dir: str = "PipelineOutputs" # NEW: where we save action timeline json
    ) -> None:
        self.client = build_client()
        self.model_candidates = model_candidates or DEFAULT_MODEL_CANDIDATES
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.gpu_index = int(gpu_index)
        self.cpu_threads = int(cpu_threads)
        self.whiteboard_width = int(whiteboard_width)
        self.whiteboard_height = int(whiteboard_height)
        self.processed_images_dir = processed_images_dir

        self.debug_timeline_json = bool(debug_timeline_json)
        self.debug_print = bool(debug_print)
        self.debug_out_dir = str(debug_out_dir or "PipelineOutputs")
        self._timings_ms: List[Dict[str, Any]] = []

    def _dbg(self, msg: str, *, data: Any = None) -> None:
        if not self.debug_print:
            return
        if data is None:
            print(f"[timeline][DBG] {msg}")
            return
        try:
            s = json.dumps(data, ensure_ascii=False)
            if len(s) > 1500:
                s = s[:1500] + " ...<truncated>"
            print(f"[timeline][DBG] {msg} | {s}")
        except Exception:
            print(f"[timeline][DBG] {msg} | <unserializable>")

    def _write_json_file(self, path: Path, payload: Any) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    @contextmanager
    def _timed(self, label: str, **meta: Any):
        t0 = time.perf_counter()
        if meta:
            self._dbg(f"[TIMER][START] {label}", data=meta)
        else:
            self._dbg(f"[TIMER][START] {label}")
        try:
            yield
        finally:
            ms = round((time.perf_counter() - t0) * 1000.0, 2)
            row: Dict[str, Any] = {"label": label, "elapsed_ms": ms}
            if meta:
                row.update(meta)
            self._timings_ms.append(row)
            self._dbg(f"[TIMER][END] {label}", data=row)

    def generate_timeline(self, topic: str) -> Tuple[str, List[Chapter], str]:
        with self._timed("llm.timeline", topic=topic):
            timeline_text, used_model = call_responses_text(
                self.client,
                self.model_candidates,
                prompt_logical_timeline(topic),
                reasoning_effort=self.reasoning_effort,
                temperature=self.temperature,
            )
        chapters = split_chapters(timeline_text)
        self._dbg("Timeline split", data={"chapters": len(chapters), "chars": len(timeline_text)})
        if not chapters:
            raise RuntimeError("Timeline produced zero chapters. Prompt or formatting failed.")
        return timeline_text, chapters, used_model

    def generate_speech_for_chapter(
        self, topic: str, chapter: Chapter, chapter_index: int, chapter_count: int
    ) -> Tuple[str, str]:
        with self._timed("llm.speech", chapter_index=chapter_index, chapter_count=chapter_count):
            speech_text, used_model = call_responses_text(
                self.client,
                self.model_candidates,
                prompt_teacher_speech(topic, chapter.raw, chapter_index, chapter_count),
                reasoning_effort=self.reasoning_effort,
                temperature=self.temperature,
            )
        self._dbg("Speech generated", data={"chapter_index": chapter_index, "chars": len(speech_text)})
        return speech_text, used_model

    def generate_images_for_speech(self, speech_text: str, *, chapter_index: Optional[int] = None) -> Tuple[Dict[str, Any], str]:
        with self._timed("llm.image_plan", chapter_index=chapter_index):
            input_items, text_format = prompt_image_requests(speech_text)
            json_text, used_model = call_responses_text(
                self.client,
                self.model_candidates,
                input_items,
                reasoning_effort=self.reasoning_effort,
                temperature=0.2,
                text_format=text_format,
            )

        # ---- DEBUG: dump raw model output + request payload ----
        try:
            out_dir = Path(self.debug_out_dir) / "LLM_RAW"
            out_dir.mkdir(parents=True, exist_ok=True)

            tag = f"ch{int(chapter_index):02d}_" if chapter_index is not None else ""
            raw_path = out_dir / f"{tag}image_plan_raw_{used_model}.txt"
            req_path = out_dir / f"{tag}image_plan_request.json"

            raw_path.write_text(json_text, encoding="utf-8")
            req_path.write_text(json.dumps(input_items, ensure_ascii=False, indent=2), encoding="utf-8")

            if self.debug_print:
                preview = json_text if len(json_text) <= 4000 else (json_text[:4000] + "\n...<truncated>\n")
                print(f"\n[LLM][IMAGE_PLAN][RAW_PREVIEW] saved_full={raw_path}")
                print(preview)
                print("[LLM][IMAGE_PLAN][END]\n")
        except Exception as e:
            if self.debug_print:
                print(f"[WARN] Failed to save LLM raw image plan: {type(e).__name__}: {e}")
        # ---------------------------------------------

        with self._timed("image_plan.parse_and_clamp", chapter_index=chapter_index):
            plan = json.loads(json_text)
            plan = clamp_image_ranges(plan, speech_text)
        self._dbg(
            "Image plan generated",
            data={"chapter_index": chapter_index, "images": len(plan.get("images", [])), "chars": len(json_text)},
        )
        return plan, used_model

    def _load_siglip_minilm_async(self) -> threading.Thread:
        """
        Starts SigLIP+MiniLM load while GPT calls run.
        """
        def _loader():
            from shared_models import init_siglip_minilm_hot
            with self._timed("models.load_siglip_minilm", gpu_index=self.gpu_index, cpu_threads=self.cpu_threads):
                init_siglip_minilm_hot(
                    gpu_index=self.gpu_index,
                    cpu_threads=self.cpu_threads,
                    warmup=True,
                )

        t = threading.Thread(target=_loader, name="load_siglip_minilm", daemon=False)
        t.start()
        return t

    def _resolve_processed_png_path(self, processed_id: str) -> Optional[Path]:
        """
        Looks for processed_id.png in:
          - ProcessedImages/
          - ProccessedImages/ (common typo mentioned)
        """
        pid = str(processed_id or "").strip()
        if not pid:
            return None

        p1 = Path(self.processed_images_dir) / f"{pid}.png"
        if p1.is_file():
            return p1

        p2 = Path("ProccessedImages") / f"{pid}.png"
        if p2.is_file():
            return p2

        # fallback: also check in cwd
        p3 = Path(f"{pid}.png")
        if p3.is_file():
            return p3

        return None

    def _compute_bbox_px_for_asset(self, asset: ImageAsset) -> None:
        """
        bbox calculator: after selection, processed_n -> ProcessedImages/processed_n.png
        bbox is literally pixel size.
        """
        if asset.bbox_px is not None:
            return
        if not asset.processed_ids:
            asset.bbox_px = (400, 300)
            return

        pid = asset.processed_ids[0]
        img_path = self._resolve_processed_png_path(pid)
        if not img_path:
            asset.bbox_px = (400, 300)
            return

        try:
            from PIL import Image
            with self._timed("image.open_for_bbox", processed_id=pid):
                with Image.open(img_path) as im:
                    w, h = im.size
            asset.bbox_px = (int(w), int(h))
        except Exception:
            asset.bbox_px = (400, 300)

    def _load_objects_list_from_refined_file(self, refined_file: str) -> List[Dict[str, Any]]:
        """
        refined labels file is expected to be:
          - a list of objects, OR
          - a dict with "objects": [...]
        """
        try:
            data = json.loads(Path(refined_file).read_text(encoding="utf-8"))
        except Exception:
            return []

        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            objs = data.get("objects")
            if isinstance(objs, list):
                return [x for x in objs if isinstance(x, dict)]
        return []

    


    def _orchestrate_images_selection_only(
        self,
        *,
        lesson_topic: str,
        all_image_requests: Dict[str, Any],  # content -> (diagram int) OR {"topic": str, "diagram": 0/1}
        diagram_required_objects_by_base_context: Optional[Dict[str, List[str]]] = None,  # NEW
    ) -> Tuple[Dict[str, ImageAsset], Dict[str, Any], Optional[Any]]:
        """
        Returns:
        (assets_by_prompt, debug_info, pipeline_handle_or_none)

        IMPORTANT:
        - waits only for selection_done (selection + refine)
        - DOES NOT wait colours_done (so we can run Qwen actions immediately)
        """
        orch_t0 = time.perf_counter()
        self._dbg("Image orchestration start", data={"requests": len(all_image_requests)})

        import PineconeFetch
        import ImageResearcher
        import ImagePipeline

        from shared_models import (
            get_siglip,
            get_minilm,
            init_qwen_hot,
            get_qwen,
            qwen_lock,
        )

        PineconeFetch.configure_hot_models(siglip_bundle=get_siglip(), minilm_bundle=get_minilm())

        prompt_meta: Dict[str, Dict[str, Any]] = {}
        for p_raw, v in (all_image_requests or {}).items():
            p = str(p_raw or "").strip()
            if not p:
                continue
            topic_val = str(lesson_topic or "").strip()
            diagram_val = 0

            if isinstance(v, dict):
                vv_topic = str(v.get("topic") or v.get("subj") or v.get("subject") or "").strip()
                if vv_topic:
                    topic_val = vv_topic
                try:
                    diagram_val = 1 if int(v.get("diagram", 0) or 0) == 1 else 0
                except Exception:
                    diagram_val = 0
            else:
                try:
                    diagram_val = 1 if int(v or 0) == 1 else 0
                except Exception:
                    diagram_val = 0

            prompt_meta[p] = {"topic": topic_val, "diagram": diagram_val}

        prompts = list(prompt_meta.keys())
        diagram_set = {p for p, m in prompt_meta.items() if int(m.get("diagram", 0) or 0) == 1}

        hits: Dict[str, List[str]] = {}
        misses: Dict[str, str] = {}

        for p in prompts:
            t0 = time.perf_counter()
            try:
                ids = PineconeFetch.fetch_processed_ids_for_prompt(
                    p,
                    top_n=2,
                    top_k_per_modality=50,
                    min_modalities=3,
                    min_final_score=0.78,
                    require_base_context_match=True,
                )
                if ids:
                    hits[p] = [str(x) for x in ids if str(x)]
                else:
                    misses[p] = str(prompt_meta.get(p, {}).get("topic") or lesson_topic or "").strip()
            except Exception:
                misses[p] = str(prompt_meta.get(p, {}).get("topic") or lesson_topic or "").strip()
            self._dbg(
                "Pinecone fetch done",
                data={
                    "prompt": p,
                    "hit": bool(p in hits),
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                },
            )

        debug: Dict[str, Any] = {
            "pinecone_hits": list(hits.keys()),
            "misses": list(misses.keys()),
            "pipeline_started": False,
            "selection_done": False,
            "colours_done": False,
            "pipeline_errors": {},
            "refined_labels_dir": None,
        }

        assets: Dict[str, ImageAsset] = {}
        for p in prompts:
            assets[p] = ImageAsset(
                prompt=p,
                diagram=int(prompt_meta.get(p, {}).get("diagram", 0) or 0),
                processed_ids=list(hits.get(p, [])),
                refined_labels_file=None,
                bbox_px=None,
                objects=None,
            )

        if not misses:
            debug["selection_done"] = True
            debug["colours_done"] = True

            for a in assets.values():
                self._compute_bbox_px_for_asset(a)
                if int(a.diagram) == 1:
                    if not a.refined_labels_file and a.processed_ids:
                        cand = self._find_refined_labels_file_for_pid(a.processed_ids[0])
                        if cand:
                            a.refined_labels_file = cand
                    if a.refined_labels_file:
                        a.objects = self._load_objects_list_from_refined_file(a.refined_labels_file)

            self._dbg(
                "Image orchestration done (pinecone-only)",
                data={"elapsed_ms": round((time.perf_counter() - orch_t0) * 1000.0, 2)},
            )
            return assets, debug, None

        # overlap qwen load + research
        qwen_err: Optional[str] = None
        research_err: Optional[str] = None

        def _load_qwen():
            nonlocal qwen_err
            try:
                init_qwen_hot(
                    gpu_index=self.gpu_index,
                    cpu_threads=self.cpu_threads,
                    warmup=True,
                )
            except Exception as e:
                qwen_err = repr(e)

        def _research():
            nonlocal research_err
            try:
                ImageResearcher.research_many(misses)
            except Exception as e:
                research_err = repr(e)

        t_qwen = threading.Thread(target=_load_qwen, name="load_qwen", daemon=False)
        t_res = threading.Thread(target=_research, name="research_many", daemon=False)

        t_qwen.start()
        t_res.start()
        threads_t0 = time.perf_counter()
        t_res.join()
        t_qwen.join()
        self._dbg("Qwen load + research done", data={"elapsed_ms": round((time.perf_counter() - threads_t0) * 1000.0, 2)})

        if research_err:
            debug["pipeline_errors"]["research_many"] = research_err
            debug["selection_done"] = True
            debug["colours_done"] = True
            return assets, debug, None

        qb = get_qwen()
        if qb is None:
            debug["pipeline_errors"]["qwen_load"] = qwen_err or "qwen_bundle_none"
            debug["selection_done"] = True
            debug["colours_done"] = True
            return assets, debug, None

        workers = ImagePipeline.PipelineWorkers(
            qwen_model=qb.model,
            qwen_processor=qb.processor,
            qwen_device=qb.device,
            qwen_lock=qwen_lock(),
            siglip_bundle=get_siglip(),
            minilm_bundle=get_minilm(),
        )

        # NEW: pass required diagram objects map into ImagePipeline
        handle = ImagePipeline.start_pipeline_background(
            workers=workers,
            model_id=qb.model_id,
            gpu_index=self.gpu_index,
            allowed_base_contexts=list(misses.keys()),
            diagram_base_contexts=[p for p in misses.keys() if p in diagram_set],
            diagram_required_objects_by_base_context=(diagram_required_objects_by_base_context or {}),  # NEW
        )
        debug["pipeline_started"] = True

        selection_t0 = time.perf_counter()
        handle.selection_done.wait()
        self._dbg("Image pipeline selection_done", data={"elapsed_ms": round((time.perf_counter() - selection_t0) * 1000.0, 2)})
        debug["selection_done"] = True

        by_ctx = getattr(handle, "selected_ids_by_base_context", {}) or {}
        for p in misses.keys():
            picked = by_ctx.get(p, []) or []
            if picked:
                assets[p].processed_ids = [str(x) for x in picked if str(x)]

        refined_dir = getattr(handle, "refined_labels_dir", None)
        if refined_dir is not None:
            debug["refined_labels_dir"] = str(refined_dir)

            for p in misses.keys():
                if not assets[p].processed_ids:
                    continue
                pid = assets[p].processed_ids[0]
                cand_path = os.path.join(str(refined_dir), f"{pid}.json")
                if os.path.isfile(cand_path):
                    assets[p].refined_labels_file = cand_path

        for a in assets.values():
            self._compute_bbox_px_for_asset(a)
            if int(a.diagram) == 1 and a.refined_labels_file:
                a.objects = self._load_objects_list_from_refined_file(a.refined_labels_file)

        self._dbg(
            "Image orchestration done",
            data={
                "elapsed_ms": round((time.perf_counter() - orch_t0) * 1000.0, 2),
                "hits": len(hits),
                "misses": len(misses),
            },
        )
        return assets, debug, handle
    def _run_qwen_actions_over_events(
        self,
        *,
        chapters_out: List[Dict[str, Any]],
        assets_by_name: Dict[str, ImageAsset],
    ) -> Dict[str, Any]:
        """
        Runs through the list of each image + silence.
        Calls a SINGLE exported Qwen function per event, and updates a global whiteboard simulator
        using BASE actions only.
        """
        from shared_models import get_qwen, qwen_lock
        import qwentest

        qb = get_qwen()
        if qb is None:
            return {"enabled": False, "error": "qwen_not_loaded"}

        BASE_ACTIONS = [
            "draw [ image name ] at : x y",
            "erase [ image name ]",
            "shift whiteboard horizontally [ amount ]",
            "shift whiteboard vertically [ amount ]",
            "link [ image name ] to [ image name ]",
            "write [text] at x , y  scale = float",
            "move [ image name ] to x, y",
        ]

        DIAGRAM_ACTIONS = [
            "highlight object for [ time ]",
            "zoom into object [ time ]",
            "refine object detail",
        ]

        # map image name -> (w,h) for state updates
        image_size_lookup: Dict[str, Tuple[int, int]] = {}
        for name, a in assets_by_name.items():
            if a.bbox_px:
                image_size_lookup[name] = a.bbox_px
            else:
                image_size_lookup[name] = (400, 300)

        board = WhiteboardState(width=self.whiteboard_width, height=self.whiteboard_height)

        all_events: List[ActionEvent] = []
        for ch in chapters_out:
            ci = int(ch.get("chapter_index", 0) or 0)
            speech = str(ch.get("speech", "") or "")
            maps = ch.get("image_text_maps") or []
            if not isinstance(maps, list):
                maps = []
            evs = build_events_for_chapter(speech, maps, ci, assets_by_name)
            all_events.extend(evs)

        # global ordering: chapter, then word index
        all_events.sort(key=lambda e: (e.chapter_index, e.start_word_index, 0 if e.kind == "silence" else 1))

        results: List[Dict[str, Any]] = []

        self._dbg("Qwen action planning start", data={"events_total": len(all_events)})
        planning_t0 = time.perf_counter()

        for idx, ev in enumerate(all_events):
            event_payload: Dict[str, Any] = {
                "event_index": idx,
                "kind": ev.kind,
                "chapter_index": ev.chapter_index,
                "start_word_index": int(ev.start_word_index),                # NEW
                "end_word_index": int(ev.end_word_index) if ev.end_word_index is not None else None,  # NEW
                "duration_sec": float(ev.duration_sec),
            }
            self._dbg("Qwen event begin", data={
                "event_index": idx,
                "kind": ev.kind,
                "chapter": ev.chapter_index,
                "start_word_index": ev.start_word_index,
                "end_word_index": ev.end_word_index,
                "duration_sec": ev.duration_sec,
                "image": ev.image_name if ev.kind == "image" else None,
                "text_tag": int(ev.text_tag) if ev.kind == "image" else 0,
            })

            if ev.kind == "image":
                event_payload.update(
                    {
                        "image_name": ev.image_name,
                        "image_text": ev.image_text or "",
                        "bbox_px": {"w": (ev.bbox_px[0] if ev.bbox_px else 400), "h": (ev.bbox_px[1] if ev.bbox_px else 300)},
                        "diagram": int(ev.diagram),
                        "processed_id": ev.processed_id,
                        "objects_that_comprise_image": ev.objects or [],
                        "refined_labels_file": ev.refined_labels_file,
                        # NEW: text tag propagation to Qwen payload shape
                        "text": int(ev.text_tag),
                        "write_text": (ev.write_text or ""),
                    }
                )
            else:
                event_payload.update(
                    {
                        "context_before": ev.context_before or "",
                        "context_after": ev.context_after or "",
                    }
                )

            wb_snapshot = board.snapshot_for_prompt()

            # If text_tag==1: DO NOT call Qwen. Still generate a normal "write" base action and update state.
            if ev.kind == "image" and int(ev.text_tag) == 1:
                text_to_write = str(ev.write_text or "").strip()
                if not text_to_write:
                    text_to_write = str(ev.image_text or "").strip()

                scale = 1.0
                est_w = max(20.0, float(len(text_to_write)) * 15.0 * scale)
                est_h = max(16.0, 30.0 * scale)

                pos = board.find_empty_spot(est_w, est_h, padding=20.0, step=80.0)
                if pos is None:
                    pos = (20.0, 20.0)

                x, y = pos
                target_name = str(ev.image_name or "").strip() or "text_auto"

                # If we are forced into a spot, erase overlappers (up to a few) before writing.
                bb_new = (x, y, x + est_w, y + est_h)
                erase_first: List[Dict[str, Any]] = []
                for oname, o in list(board.objects.items()):
                    if WhiteboardState._bboxes_overlap(bb_new, o.bbox()):
                        erase_first.append({"type": "erase", "target": oname})
                        if len(erase_first) >= 3:
                            break

                write_action = {
                    "type": "write",
                    "target": target_name,
                    "text": text_to_write,
                    "x": float(x),
                    "y": float(y),
                    "scale": float(scale),
                }

                qwen_payload = {
                    "actions": erase_first + [write_action],
                    "raw_text": "",
                    "notes": "text_auto",
                }
                base_actions_parsed = WhiteboardState.parse_actions(qwen_payload)

                for a in base_actions_parsed:
                    board.apply_base_action(a, image_size_lookup=image_size_lookup)

                results.append(
                    {
                        "event": event_payload,
                        "qwen_payload": qwen_payload,
                        "base_actions_parsed": base_actions_parsed,
                        "whiteboard_after": board.snapshot_for_prompt(),
                    }
                )
                continue

            # Call Qwen (simple function only)
            qwen_t0 = time.perf_counter()
            with qwen_lock():
                qwen_payload = qwentest.plan_whiteboard_actions_transformers(
                    model=qb.model,
                    processor=qb.processor,
                    device=qb.device,
                    event=event_payload,
                    whiteboard_state=wb_snapshot,
                    base_actions=BASE_ACTIONS,
                    diagram_actions=DIAGRAM_ACTIONS,
                    temperature=0.2,
                    max_new_tokens=700,
                )
            self._dbg(
                "Qwen event done",
                data={
                    "event_index": idx,
                    "kind": ev.kind,
                    "elapsed_ms": round((time.perf_counter() - qwen_t0) * 1000.0, 2),
                },
            )

            base_actions_parsed = WhiteboardState.parse_actions(qwen_payload)
            # apply BASE actions to board state
            for a in base_actions_parsed:
                board.apply_base_action(a, image_size_lookup=image_size_lookup)

            # Debug print the whiteboard state, after each image (NOT text-tag images)
            if ev.kind == "image" and int(ev.text_tag) == 0:
                self._dbg("Whiteboard state after image", data=board.snapshot_for_prompt())

            results.append(
                {
                    "event": event_payload,
                    "qwen_payload": qwen_payload,
                    "base_actions_parsed": base_actions_parsed,
                    "whiteboard_after": board.snapshot_for_prompt(),
                }
            )
        
        self._dbg(
            "Qwen action planning done",
            data={"events_total": len(results), "elapsed_ms": round((time.perf_counter() - planning_t0) * 1000.0, 2)},
        )

        return {
            "enabled": True,
            "whiteboard_width": self.whiteboard_width,
            "whiteboard_height": self.whiteboard_height,
            "events_count": len(results),
            "events": results,
            "final_whiteboard": board.snapshot_for_prompt(),
        }
    
    def _run_qwen_diagram_cluster_labeling(
        self,
        *,
        assets_by_name: Dict[str, ImageAsset],
        sam_bbox_map_any: Any,
    ) -> Dict[str, Any]:
        """
        For each DIAGRAM image:
          - read ClusterMaps/processed_n/clusters.json
          - match SAM bboxes -> closest cluster bboxes
          - load ONLY matched cluster renders from ClusterRenders
          - run qwentest.label_clusters_transformers (visual-only stage1 + post-facto stage2 label match)
          - writes ClustersLabeled/processed_n/labels.json via qwentest
        """
        import qwentest
        from PIL import Image
        import numpy as np

        # normalize SAM root structure
        # expected outer shape from _run_efficientsam3_bboxes: {"ok":True,"result": bbox_map}
        bbox_root = None
        if isinstance(sam_bbox_map_any, dict):
            bbox_root = sam_bbox_map_any.get("result") if "result" in sam_bbox_map_any else sam_bbox_map_any

        if not isinstance(bbox_root, dict):
            bbox_root = {}

        # lazily load Qwen (must be loaded already by the caller)
        from shared_models import get_qwen
        qb = get_qwen()
        if qb is None:
            return {"ok": False, "error": "qwen_not_loaded_for_diagram_labeling"}

        base_dir = Path(__file__).resolve().parent
        cluster_maps_root = base_dir / "ClusterMaps"
        cluster_renders_root = base_dir / "ClusterRenders"

        summary: Dict[str, Any] = {
            "ok": True,
            "images": [],
        }

        all_t0 = time.perf_counter()
        for name, asset in assets_by_name.items():
            one_t0 = time.perf_counter()
            if int(asset.diagram) != 1:
                continue
            if not asset.processed_ids:
                continue

            pid = str(asset.processed_ids[0] or "").strip()
            if not pid:
                continue

            # parse processed_n -> n
            m = re.match(r"^processed_(\d+)$", pid.strip(), re.IGNORECASE)
            if not m:
                summary["images"].append({"image_name": name, "processed_id": pid, "skipped": True, "reason": "bad_processed_id"})
                continue
            idx = int(m.group(1))

            cmap_path = cluster_maps_root / f"processed_{idx}" / "clusters.json"
            renders_dir = cluster_renders_root / f"processed_{idx}"
            img_path = self._resolve_processed_png_path(pid)

            if not cmap_path.is_file() or not renders_dir.is_dir() or not img_path:
                summary["images"].append({
                    "image_name": name,
                    "processed_id": pid,
                    "skipped": True,
                    "reason": "missing_cluster_maps_or_renders_or_image",
                    "clusters_json": str(cmap_path),
                    "renders_dir": str(renders_dir),
                    "image_path": str(img_path) if img_path else None,
                })
                continue

            try:
                cmap = json.loads(cmap_path.read_text(encoding="utf-8"))
            except Exception as e:
                summary["images"].append({"image_name": name, "processed_id": pid, "skipped": True, "reason": f"bad_clusters_json: {e}"})
                continue

            clusters = cmap.get("clusters") or []
            if not isinstance(clusters, list) or not clusters:
                summary["images"].append({"image_name": name, "processed_id": pid, "skipped": True, "reason": "no_clusters"})
                continue

            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    img_w, img_h = im.size
                    full_img_rgb = np.asarray(im, dtype=np.uint8)
            except Exception as e:
                summary["images"].append({"image_name": name, "processed_id": pid, "skipped": True, "reason": f"image_open_failed: {e}"})
                continue

            # refined labels list: prefer refined_labels_file; fallback to SAM label keys
            refined_file = asset.refined_labels_file
            refined_labels = self._load_refined_label_strings(refined_file)

            sam_pid_payload = bbox_root.get(pid)
            sam_items = self._extract_sam_items_for_pid(sam_pid_payload)

            if not refined_labels:
                # fallback to label keys from SAM items
                labs = []
                seen = set()
                for it in sam_items:
                    lab = str(it.get("label", "") or "").strip()
                    if lab and lab.lower() not in seen:
                        seen.add(lab.lower())
                        labs.append(lab)
                refined_labels = labs

            if not sam_items:
                summary["images"].append({
                    "image_name": name,
                    "processed_id": pid,
                    "skipped": True,
                    "reason": "no_sam_bboxes_for_pid",
                })
                continue

            matched_clusters, debug_rows = self._match_sam_bboxes_to_cluster_entries(
                pid=pid,
                sam_items=sam_items,
                clusters=clusters,
                img_w=img_w,
                img_h=img_h,
            )

            if not matched_clusters:
                summary["images"].append({
                    "image_name": name,
                    "processed_id": pid,
                    "skipped": True,
                    "reason": "sam_to_cluster_match_empty",
                    "debug_matches": debug_rows[:50],
                })
                continue

            # solid debug prints (as requested)
            print(f"[sam-match] {pid} matched_clusters={len(matched_clusters)} total_clusters={len(clusters)}")
            for r in debug_rows[:40]:
                print(
                    f"  [sam-match] label={r.get('sam_label')} iou={r.get('iou'):.3f} score={r.get('match_score'):.4f} -> {r.get('matched_mask')}"
                )

            # Load ONLY matched cluster renders
            renders_mask_rgb: Dict[str, Any] = {}
            for c in matched_clusters:
                mname = str(c.get("crop_file_mask", "") or "").strip()
                if not mname:
                    continue
                p = renders_dir / mname
                if not p.is_file():
                    continue
                try:
                    with Image.open(p) as mi:
                        mi = mi.convert("RGB")
                        renders_mask_rgb[mname] = np.asarray(mi, dtype=np.uint8)
                except Exception:
                    continue

            filtered_entries = []
            keep = set(renders_mask_rgb.keys())
            for c in matched_clusters:
                mname = str(c.get("crop_file_mask", "") or "").strip()
                if mname in keep:
                    filtered_entries.append(c)

            if not filtered_entries:
                summary["images"].append({
                    "image_name": name,
                    "processed_id": pid,
                    "skipped": True,
                    "reason": "matched_clusters_have_no_renders",
                    "debug_matches": debug_rows[:50],
                })
                continue

            clusters_state = {
                idx: {
                    "base_context": name,  # best available here (image content)
                    "candidate_labels_raw": refined_labels,
                    "candidate_labels_refined": refined_labels,
                    "full_img_rgb": full_img_rgb,
                    "clusters": filtered_entries,
                    "renders_mask_rgb": renders_mask_rgb,
                }
            }

            # Run OLD labeling function (now visual-only + post-facto match) and SAVE into ClustersLabeled
            res = qwentest.label_clusters_transformers(
                clusters_state=clusters_state,
                model=qb.model,
                processor=qb.processor,
                device=qb.device,
                save_outputs=True,
            )

            summary["images"].append({
                "image_name": name,
                "processed_id": pid,
                "clusters_total": len(clusters),
                "clusters_matched": len(filtered_entries),
                "debug_matches_count": len(debug_rows),
                "debug_matches_sample": debug_rows[:80],
                "qwen_output_keys": list(res.get(idx, {}).keys()) if isinstance(res.get(idx), dict) else [],
                "elapsed_ms": round((time.perf_counter() - one_t0) * 1000.0, 2),
            })

        self._dbg(
            "Diagram cluster labeling done",
            data={"elapsed_ms": round((time.perf_counter() - all_t0) * 1000.0, 2), "images": len(summary["images"])},
        )
        return summary
    
    def _build_whiteboard_action_timeline(
        self,
        *,
        chapters_out: List[Dict[str, Any]],
        whiteboard_actions_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Produces an index-synced timeline:
        - Every action synced to a word index in the CHAPTER, or to a SILENCE word index.
        - No seconds->word conversion. duration_sec stays contextual only.
        """
        t0 = time.perf_counter()
        events = whiteboard_actions_payload.get("events") or []
        if not isinstance(events, list):
            events = []

        chapters_meta: Dict[int, Dict[str, Any]] = {}
        for ch in (chapters_out or []):
            ci = int(ch.get("chapter_index", 0) or 0)
            speech = str(ch.get("speech", "") or "")
            words = speech.split()

            pauses = []
            for wi, tok in enumerate(words):
                sec = _is_pause_token(tok)
                if sec is not None:
                    pauses.append({"word_index": int(wi), "seconds": float(sec)})

            chapters_meta[ci] = {
                "chapter_index": ci,
                "speech": speech,
                "word_count": len(words),
                "pause_markers": pauses,
            }

            if self.debug_timeline_json:
                chapters_meta[ci]["speech_words"] = words

        # Flatten actions with sync info
        flat_actions: List[Dict[str, Any]] = []
        global_action_i = 0

        for ev_i, ev_pack in enumerate(events):
            ev = ev_pack.get("event") or {}
            acts = ev_pack.get("base_actions_parsed") or []
            if not isinstance(acts, list):
                acts = []

            chapter_index = int(ev.get("chapter_index", 0) or 0)
            kind = str(ev.get("kind", "") or "")
            start_wi = int(ev.get("start_word_index", 0) or 0)
            end_wi = ev.get("end_word_index", None)
            try:
                end_wi = int(end_wi) if end_wi is not None else None
            except Exception:
                end_wi = None

            sync = {
                "sync_kind": "silence" if kind == "silence" else "word",
                "word_index_in_chapter": start_wi,
            }
            if kind == "silence":
                sync["silence_seconds"] = float(ev.get("duration_sec", 0.0) or 0.0)

            # Every base action becomes an entry
            for local_i, a in enumerate(acts):
                if not isinstance(a, dict):
                    continue
                flat_actions.append({
                    "global_action_index": global_action_i,
                    "event_index": int(ev.get("event_index", ev_i)),
                    "action_index_in_event": int(local_i),
                    "chapter_index": chapter_index,
                    "event_kind": kind,
                    "event_duration_sec": float(ev.get("duration_sec", 0.0) or 0.0),
                    "event_start_word_index": start_wi,
                    "event_end_word_index": end_wi,
                    "sync": sync,
                    "action": a,  # draw/erase/shift/write/move/link
                })
                global_action_i += 1

        out = {
            "schema": "whiteboard_action_timeline_v1",
            "chapters": [chapters_meta[k] for k in sorted(chapters_meta.keys())],
            "actions": flat_actions,
            "counts": {
                "events": len(events),
                "actions": len(flat_actions),
            }
        }

        if self.debug_timeline_json:
            # keep intermediate event payloads (heavy)
            out["debug_events_raw"] = events

        self._dbg(
            "Built whiteboard action timeline",
            data={
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "events": len(events),
                "actions": len(flat_actions),
            },
        )
        return out


    def _save_whiteboard_action_timeline_json(
        self,
        *,
        chapters_out: List[Dict[str, Any]],
        whiteboard_actions_payload: Dict[str, Any],
    ) -> str:
        timeline = self._build_whiteboard_action_timeline(
            chapters_out=chapters_out,
            whiteboard_actions_payload=whiteboard_actions_payload,
        )
        out_dir = Path(self.debug_out_dir)
        path = out_dir / "whiteboard_action_timeline.json"
        saved = self._write_json_file(path, timeline)
        self._dbg("Saved whiteboard_action_timeline.json", data={"path": saved, "events": timeline["counts"]["events"], "actions": timeline["counts"]["actions"]})
        return saved

    def run_full(self, topic: str, *, integrate_image_pipeline: bool = True) -> Dict[str, Any]:
        """
        Runs:
          timeline -> chapter speeches -> per-speech image plans
          + optional ImagePipeline integration
          + new: Qwen action planning right after selection/refine
          + then wait colours_done
        """
        self._timings_ms = []
        run_t0 = time.perf_counter()
        siglip_thread = None
        if integrate_image_pipeline:
            siglip_thread = self._load_siglip_minilm_async()

        timeline_text, chapters, model_timeline = self.generate_timeline(topic)

        out: Dict[str, Any] = {
            "topic": topic,
            "models_used": {
                "timeline": model_timeline,
                "speech": [],
                "images": [],
            },
            "timeline_raw": timeline_text,
            "chapters": [],
            "image_pipeline": {
                "enabled": bool(integrate_image_pipeline),
                "assets_by_prompt": {},
                "debug": {},
            },
            "whiteboard_actions": {
                "enabled": False,
            },
        }

        chapter_count = len(chapters)
        self._dbg("Parallel chapter generation start", data={"chapters": chapter_count})

        all_prompts_meta: Dict[str, Dict[str, Any]] = {}
        all_diagram_required_objects: Dict[str, List[str]] = {}  # NEW

        chapter_slots: List[Optional[Dict[str, Any]]] = [None] * chapter_count
        speech_models: List[Optional[str]] = [None] * chapter_count
        image_models: List[Optional[str]] = [None] * chapter_count

        def _chapter_worker(i: int, ch: Chapter) -> Dict[str, Any]:
            with self._timed("chapter.full", chapter_index=i):
                speech, model_speech = self.generate_speech_for_chapter(topic, ch, i, chapter_count)
                images_plan, model_images = self.generate_images_for_speech(speech, chapter_index=i)
                image_text_maps = build_image_text_maps(speech, images_plan)

            local_prompts_meta: Dict[str, Dict[str, Any]] = {}
            local_required_objs: Dict[str, List[str]] = {}  # NEW

            for img in images_plan.get("images", []):
                # skip text-tag entries for image pipeline orchestration
                if int(img.get("text", 0) or 0) == 1:
                    continue

                query = str(img.get("content", "") or "").strip()
                if not query:
                    continue

                d = 1 if int(img.get("diagram", 0) or 0) == 1 else 0
                topic_val = str(img.get("topic", "") or "").strip()
                prev_meta = local_prompts_meta.get(query) or {}
                prev_d = int(prev_meta.get("diagram", 0) or 0)
                prev_t = str(prev_meta.get("topic", "") or "").strip()
                local_prompts_meta[query] = {
                    "topic": topic_val or prev_t or str(topic or "").strip(),
                    "diagram": 1 if (prev_d == 1 or d == 1) else 0,
                }

                # NEW: accumulate required objects only for diagrams
                if d == 1:
                    dro = img.get("diagram_required_objects") or []
                    if not isinstance(dro, list):
                        dro = []
                    dro = [str(x).strip() for x in dro if str(x).strip()]
                    if dro:
                        _merge_unique_str_list(local_required_objs.setdefault(query, []), dro, cap=80)

            return {
                "chapter_index": i,
                "chapter_payload": {
                    "chapter_index": i,
                    "timeline_chunk": ch.raw,
                    "speech": speech,
                    "image_plan": images_plan,
                    "image_text_maps": image_text_maps,
                },
                "speech_model": model_speech,
                "image_model": model_images,
                "prompts_meta": local_prompts_meta,
                "required_objects_by_prompt": local_required_objs,  # NEW
            }

        max_workers = max(1, min(chapter_count, 8))
        with self._timed("chapters.parallel_generation", chapters=chapter_count, workers=max_workers):
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="chapter") as ex:
                fut_to_idx = {
                    ex.submit(_chapter_worker, i, ch): i
                    for i, ch in enumerate(chapters, start=1)
                }
                for fut in as_completed(fut_to_idx):
                    res = fut.result()
                    slot = int(res["chapter_index"]) - 1
                    chapter_slots[slot] = res["chapter_payload"]
                    speech_models[slot] = str(res["speech_model"])
                    image_models[slot] = str(res["image_model"])

                    local_pm = res.get("prompts_meta") or {}
                    for query, meta in local_pm.items():
                        if not isinstance(meta, dict):
                            continue
                        d = 1 if int(meta.get("diagram", 0) or 0) == 1 else 0
                        t = str(meta.get("topic", "") or "").strip()

                        prev = all_prompts_meta.get(query) or {}
                        prev_d = int(prev.get("diagram", 0) or 0)
                        prev_t = str(prev.get("topic", "") or "").strip()

                        all_prompts_meta[query] = {
                            "topic": prev_t or t or str(topic or "").strip(),
                            "diagram": 1 if (prev_d == 1 or d == 1) else 0,
                        }

                    # NEW: merge required objects upward
                    local_ro = res.get("required_objects_by_prompt") or {}
                    if isinstance(local_ro, dict):
                        for query, objs in local_ro.items():
                            if not isinstance(objs, list):
                                continue
                            _merge_unique_str_list(all_diagram_required_objects.setdefault(query, []), objs, cap=80)

                    self._dbg(
                        "Chapter complete",
                        data={"chapter_index": int(res["chapter_index"]), "prompts_found": len(local_pm)},
                    )



        out["models_used"]["speech"] = [m or "" for m in speech_models]
        out["models_used"]["images"] = [m or "" for m in image_models]
        out["chapters"] = [c for c in chapter_slots if c is not None]

        if not integrate_image_pipeline:
            out["timings_ms"] = list(self._timings_ms)
            out["run_elapsed_ms"] = round((time.perf_counter() - run_t0) * 1000.0, 2)
            return out

        if siglip_thread is not None:
            with self._timed("models.wait_siglip_minilm"):
                siglip_thread.join()

        if not all_prompts_meta:
            out["image_pipeline"]["debug"] = {"note": "no_image_prompts"}
            out["timings_ms"] = list(self._timings_ms)
            out["run_elapsed_ms"] = round((time.perf_counter() - run_t0) * 1000.0, 2)
            return out

        # selection + refinement only (returns handle so colours can keep running)
        with self._timed("image_pipeline.selection_only"):
            assets, dbg, handle = self._orchestrate_images_selection_only(
                lesson_topic=topic,
                all_image_requests=all_prompts_meta,
                diagram_required_objects_by_base_context=all_diagram_required_objects,
            )

        assets_by_name: Dict[str, ImageAsset] = {}

        for ch in out["chapters"]:
            maps = ch.get("image_text_maps") or []
            if not isinstance(maps, list):
                continue
            for im in maps:
                if not isinstance(im, dict):
                    continue
                # NEW: skip text-tag entries
                if int(im.get("text_tag", 0) or 0) == 1:
                    continue

                name = str(im.get("content", "") or "").strip()
                query = str(im.get("query", "") or "").strip()
                if name and query and (query in assets):
                    assets_by_name[name] = assets[query]

        assets_json: Dict[str, Any] = {}
        for k, v in assets.items():
            assets_json[k] = {
                "diagram": int(v.diagram),
                "processed_ids": list(v.processed_ids),
                "refined_labels_file": v.refined_labels_file,
                "bbox_px": {"w": v.bbox_px[0], "h": v.bbox_px[1]} if v.bbox_px else None,
                "objects_that_comprise_image": v.objects or [],
            }

        out["image_pipeline"]["assets_by_prompt"] = assets_json
        out["image_pipeline"]["debug"] = dbg

        # Attach resolved ids/refined/objects/bbox into each chapter's image_text_maps
        for ch in out["chapters"]:
            maps = ch.get("image_text_maps") or []
            if not isinstance(maps, list):
                continue
            for im in maps:
                if not isinstance(im, dict):
                    continue
                # NEW: skip text-tag entries
                if int(im.get("text_tag", 0) or 0) == 1:
                    continue

                query = str(im.get("query", "") or im.get("content", "") or "").strip()
                if not query or query not in assets:
                    continue
                a = assets[query]
                im["processed_ids"] = list(a.processed_ids)
                im["refined_labels_file"] = a.refined_labels_file
                im["bbox_px"] = {"w": a.bbox_px[0], "h": a.bbox_px[1]} if a.bbox_px else None
                im["objects_that_comprise_image"] = a.objects or []

        # 1) Qwen action planning (Qwen must still be loaded here)
        with self._timed("qwen.whiteboard_action_planning"):
            out["whiteboard_actions"] = self._run_qwen_actions_over_events(
                chapters_out=out["chapters"],
                assets_by_name=assets_by_name,
            )

        # save index-synced action timeline immediately after Qwen stage
        try:
            saved_path = self._save_whiteboard_action_timeline_json(
                chapters_out=out["chapters"],
                whiteboard_actions_payload=out["whiteboard_actions"],
            )
            out["whiteboard_actions"]["saved_action_timeline_json"] = saved_path
        except Exception as e:
            out["whiteboard_actions"]["saved_action_timeline_json"] = None
            out["whiteboard_actions"]["save_error"] = f"{type(e).__name__}: {e}"
            self._dbg("Failed saving action timeline json", data={"err": out["whiteboard_actions"]["save_error"]})

        # 2) UNLOAD QWEN (free GPU before SAM)
        try:
            from shared_models import unload_qwen
            with self._timed("models.unload_qwen"):
                unload_qwen()
        except Exception:
            pass

        # 3) LOAD EfficientSAM3 + run bbox search per refined label
        out["efficientsam3_bboxes"] = {"ok": True, "note": "skipped"}

        try:
            from shared_models import init_efficientsam3_hot
            with self._timed("models.load_efficientsam3"):
                init_efficientsam3_hot(gpu_index=self.gpu_index, cpu_threads=self.cpu_threads, warmup=True)
            out_dir = str(getattr(handle, "out_dir", Path("PipelineOutputs"))) if handle is not None else "PipelineOutputs"
            with self._timed("efficientsam3.compute_bboxes"):
                out["efficientsam3_bboxes"] = self._run_efficientsam3_bboxes(
                    assets_by_name=assets,
                    out_dir=out_dir,
                )
        except Exception as e:
            out["efficientsam3_bboxes"] = {"ok": False, "err": f"{type(e).__name__}: {e}"}

        # 4) Optionally unload SAM (same logic discipline)
        try:
            from shared_models import unload_efficientsam3
            with self._timed("models.unload_efficientsam3"):
                unload_efficientsam3()
        except Exception:
            pass

        # 4.5) NEW: preload Qwen again AFTER SAM, but do NOT run it until colours_done
        qwen_reload_err: Optional[str] = None
        qwen_reload_thread: Optional[threading.Thread] = None

        def _reload_qwen_after_sam():
            nonlocal qwen_reload_err
            try:
                from shared_models import init_qwen_hot
                init_qwen_hot(
                    gpu_index=self.gpu_index,
                    cpu_threads=self.cpu_threads,
                    warmup=True,
                )
            except Exception as e:
                qwen_reload_err = repr(e)

        qwen_reload_thread = threading.Thread(target=_reload_qwen_after_sam, name="reload_qwen_after_sam", daemon=False)
        qwen_reload_thread.start()

        # 5) Now wait for colours_done (pipeline finished)
        if handle is not None:
            with self._timed("image_pipeline.wait_colours_done"):
                handle.colours_done.wait()
            out["image_pipeline"]["debug"]["colours_done"] = True
            out["image_pipeline"]["debug"]["pipeline_errors"] = dict(getattr(handle, "errors", {}) or {})

        # 6) Ensure Qwen reload completed
        if qwen_reload_thread is not None:
            with self._timed("models.wait_qwen_reload"):
                qwen_reload_thread.join()

        if qwen_reload_err:
            out["diagram_cluster_labels"] = {"ok": False, "error": f"qwen_reload_failed: {qwen_reload_err}"}
        else:
            # 7) NEW: run diagram cluster labeling (ONLY matched clusters via SAM bboxes)
            with self._timed("qwen.diagram_cluster_labeling"):
                out["diagram_cluster_labels"] = self._run_qwen_diagram_cluster_labeling(
                    assets_by_name=assets,
                    sam_bbox_map_any=out.get("efficientsam3_bboxes", {}),
                )

        out["timings_ms"] = list(self._timings_ms)
        out["run_elapsed_ms"] = round((time.perf_counter() - run_t0) * 1000.0, 2)
        self._dbg("run_full complete", data={"run_elapsed_ms": out["run_elapsed_ms"], "timings_count": len(self._timings_ms)})
        return out

    def _find_refined_labels_file_for_pid(self, pid: str) -> Optional[str]:
        """
        For Pinecone hits (no ImagePipeline handle), try to locate an existing refined labels artifact on disk.
        If it doesn't exist, return None.
        """
        pid = str(pid or "").strip()
        if not pid:
            return None

        candidates = [
            Path("PipelineOutputs") / "RefinedLabels" / f"{pid}.json",
            Path("PipelineOutputs") / "refined_labels" / f"{pid}.json",
            Path("PipelineOutputs") / f"{pid}.json",
            Path("RefinedLabels") / f"{pid}.json",
        ]
        for p in candidates:
            try:
                if p.is_file():
                    return str(p)
            except Exception:
                continue
        return None

    def _load_refined_label_strings(self, refined_file: Optional[str]) -> List[str]:
        """
        Returns refined labels as list[str].
        Accepts:
          - dict with "objects": [str] OR "refined_labels": [str] OR "labels": [...]
          - list[str]
          - list[dict] with {"name": "..."}
        """
        if not refined_file:
            return []
        try:
            data = json.loads(Path(refined_file).read_text(encoding="utf-8"))
        except Exception:
            return []

        raw = None
        if isinstance(data, dict):
            raw = data.get("objects")
            if not isinstance(raw, list):
                raw = data.get("refined_labels")
            if not isinstance(raw, list):
                raw = data.get("labels")
        elif isinstance(data, list):
            raw = data

        if not isinstance(raw, list):
            return []

        out: List[str] = []
        seen = set()
        for it in raw:
            s = None
            if isinstance(it, str):
                s = it
            elif isinstance(it, dict):
                n = it.get("name")
                if isinstance(n, str):
                    s = n
            if not isinstance(s, str):
                continue
            ss = s.strip()
            if not ss:
                continue
            key = ss.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(ss)
        return out

    @staticmethod
    def _bbox_iou(a: List[int], b: List[int]) -> float:
        ax0, ay0, ax1, ay1 = [int(x) for x in a]
        bx0, by0, bx1, by1 = [int(x) for x in b]
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        iw = max(0, ix1 - ix0)
        ih = max(0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        aa = max(1, (ax1 - ax0) * (ay1 - ay0))
        ba = max(1, (bx1 - bx0) * (by1 - by0))
        return float(inter / float(aa + ba - inter))

    def _bbox_match_score(self, sam_bb: List[int], cl_bb: List[int], img_w: int, img_h: int) -> float:
        # lower = better
        sx0, sy0, sx1, sy1 = [int(x) for x in sam_bb]
        cx0, cy0, cx1, cy1 = [int(x) for x in cl_bb]

        scx = (sx0 + sx1) * 0.5
        scy = (sy0 + sy1) * 0.5
        ccx = (cx0 + cx1) * 0.5
        ccy = (cy0 + cy1) * 0.5

        diag = max(1.0, (img_w * img_w + img_h * img_h) ** 0.5)
        center_d = (((scx - ccx) ** 2 + (scy - ccy) ** 2) ** 0.5) / diag

        sa = max(1.0, float((sx1 - sx0) * (sy1 - sy0)))
        ca = max(1.0, float((cx1 - cx0) * (cy1 - cy0)))
        # size similarity in log-space
        import math
        size_d = abs(math.log(ca / sa))

        iou = self._bbox_iou(sam_bb, cl_bb)
        # combine: center + size + iou penalty
        score = center_d + 0.70 * size_d + (1.0 - iou)
        return float(score)

    def _extract_sam_items_for_pid(self, sam_pid_payload: Any) -> List[Dict[str, Any]]:
        """
        Normalize SAM output into:
          [{"label": "...", "bbox_xyxy":[...], "score": optional}, ...]
        Tolerant to multiple possible shapes.
        """
        out: List[Dict[str, Any]] = []
        if not sam_pid_payload:
            return out

        if isinstance(sam_pid_payload, dict):
            # expected: {label: [bboxes...]} or {label: [{"bbox_xyxy":...,"score":...}, ...]}
            for lab, arr in sam_pid_payload.items():
                label = str(lab or "").strip()
                if not label:
                    continue
                if isinstance(arr, list):
                    for it in arr:
                        bb = None
                        sc = None
                        if isinstance(it, list) and len(it) == 4:
                            bb = [int(x) for x in it]
                        elif isinstance(it, dict):
                            bb0 = it.get("bbox_xyxy") or it.get("bbox") or it.get("box")
                            if isinstance(bb0, list) and len(bb0) == 4:
                                bb = [int(x) for x in bb0]
                            sc0 = it.get("score") or it.get("iou") or it.get("confidence")
                            try:
                                sc = float(sc0) if sc0 is not None else None
                            except Exception:
                                sc = None
                        if isinstance(bb, list) and len(bb) == 4:
                            out.append({"label": label, "bbox_xyxy": bb, "score": sc})
        elif isinstance(sam_pid_payload, list):
            # already list of items
            for it in sam_pid_payload:
                if not isinstance(it, dict):
                    continue
                label = str(it.get("label", "") or "").strip()
                bb0 = it.get("bbox_xyxy") or it.get("bbox") or it.get("box")
                if not label:
                    continue
                if isinstance(bb0, list) and len(bb0) == 4:
                    out.append({"label": label, "bbox_xyxy": [int(x) for x in bb0], "score": it.get("score")})
        return out

    def _match_sam_bboxes_to_cluster_entries(
        self,
        *,
        pid: str,
        sam_items: List[Dict[str, Any]],
        clusters: List[Dict[str, Any]],
        img_w: int,
        img_h: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Returns: (matched_cluster_entries, debug_rows)
        - Matches each SAM bbox to the closest cluster bbox by combined center+size+IoU score.
        - Keeps UNIQUE cluster mask targets (dedup by crop_file_mask).
        """
        debug_rows: List[Dict[str, Any]] = []
        if not sam_items or not clusters:
            return [], debug_rows

        best_by_mask: Dict[str, Tuple[float, Dict[str, Any], Dict[str, Any]]] = {}

        for si, it in enumerate(sam_items, start=1):
            bb = it.get("bbox_xyxy")
            lab = str(it.get("label", "") or "").strip()
            if not (isinstance(bb, list) and len(bb) == 4):
                continue

            best = None
            best_score = None

            for c in clusters:
                cbb = c.get("bbox_xyxy")
                mname = c.get("crop_file_mask")
                if not (isinstance(cbb, list) and len(cbb) == 4):
                    continue
                if not isinstance(mname, str) or not mname.strip():
                    continue

                score = self._bbox_match_score(bb, cbb, img_w, img_h)

                if best_score is None or score < best_score:
                    best_score = score
                    best = c

            if best is None or best_score is None:
                continue

            mname = str(best.get("crop_file_mask", "") or "").strip()
            row = {
                "processed_id": pid,
                "sam_index": si,
                "sam_label": lab,
                "sam_bbox_xyxy": [int(x) for x in bb],
                "matched_mask": mname,
                "cluster_bbox_xyxy": [int(x) for x in best.get("bbox_xyxy", [0, 0, 0, 0])],
                "match_score": float(best_score),
                "iou": float(self._bbox_iou(bb, best.get("bbox_xyxy", [0, 0, 0, 0]))),
            }
            debug_rows.append(row)

            # keep best (lowest score) per mask_name
            prev = best_by_mask.get(mname)
            if prev is None or best_score < prev[0]:
                best_by_mask[mname] = (best_score, best, row)

        # stable order: by bbox position
        matched = [v[1] for v in best_by_mask.values()]
        matched.sort(key=lambda c: (int(c.get("bbox_xyxy", [0, 0, 0, 0])[0]), int(c.get("bbox_xyxy", [0, 0, 0, 0])[1])))

        return matched, debug_rows


# ----------------------------
# Bind module-level helper as method if missing
# (keeps compatibility if _run_efficientsam3_bboxes was originally module-level)
# ----------------------------
try:
    if not hasattr(LessonTimelinePipeline, "_run_efficientsam3_bboxes"):
        LessonTimelinePipeline._run_efficientsam3_bboxes = _run_efficientsam3_bboxes  # type: ignore
except Exception:
    pass


# ----------------------------
# Minimal CLI usage
# ----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="Lesson topic, e.g. 'Eukaryotic cell structure'")
    parser.add_argument("--out", default="lesson_output.json", help="Output JSON file path")
    parser.add_argument("--no-images", action="store_true", help="Disable ImagePipeline integration")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--wb-w", type=int, default=1920)
    parser.add_argument("--wb-h", type=int, default=1080)
    parser.add_argument("--processed-dir", default="ProcessedImages")
    args = parser.parse_args()

    pipe = LessonTimelinePipeline(
        gpu_index=args.gpu,
        cpu_threads=args.cpu_threads,
        whiteboard_width=args.wb_w,
        whiteboard_height=args.wb_h,
        processed_images_dir=args.processed_dir,
    )
    result = pipe.run_full(args.topic, integrate_image_pipeline=(not args.no_images))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {args.out}")
