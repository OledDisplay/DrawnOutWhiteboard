"""
timeline_pipeline.py

3-stage lesson timeline pipeline:
1) topic -> logical timeline split into chapters with '|'
2) each chapter -> teacher-style narration with pause markers like '>3.0'
3) each narration chunk -> JSON image plan with diagram flag + word index ranges

PLUS:
4) Integrates with ImagePipeline orchestrator:
   - Preloads SigLIP text-only + MiniLM while GPT requests are running
   - When image collection is needed:
       - Pinecone fetch first (called directly here with shared workers)
       - Unload SigLIP workers before Comfy/Flux generation
       - If misses:
           - After Flux/Comfy completes, free Comfy models via API, then load full SigLIP and Qwen
           - Wait for mechanical research + SigLIP reload, then run final rerank
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
import hashlib
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
SPACE_PLANNER_BATCH_SIZE = max(1, int(os.getenv("QWEN_SPACE_BATCH_SIZE", "4") or 4))
ACTION_PLANNER_BATCH_SIZE = max(1, int(os.getenv("QWEN_ACTION_BATCH_SIZE", "2") or 2))
SPACE_PLANNER_MAX_NEW_TOKENS = max(64, int(os.getenv("QWEN_SPACE_MAX_NEW_TOKENS", "320") or 320))
VISUAL_PLANNER_MAX_NEW_TOKENS = max(64, int(os.getenv("QWEN_VISUAL_MAX_NEW_TOKENS", "220") or 220))
SILENCE_PLANNER_MAX_NEW_TOKENS = max(64, int(os.getenv("QWEN_SILENCE_MAX_NEW_TOKENS", "140") or 140))
QWEN_RELOAD_TIMEOUT_SEC = max(10.0, float(os.getenv("QWEN_RELOAD_TIMEOUT_SEC", "180") or 180))
INTER_CHUNK_SILENCE_SEC = 3.0


SPEECH_PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "full_speech": {"type": "string", "minLength": 1},
        "step_speech": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_key": {"type": "string", "minLength": 1},
                    "speech": {"type": "string", "minLength": 1},
                },
                "required": ["step_key", "speech"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["full_speech", "step_speech"],
    "additionalProperties": False,
}


def _normalize_ws(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def parse_chapter_timeline_steps(chapter_timeline: str) -> List[Dict[str, str]]:
    """
    Parse timeline steps in stable order:
      - top level: "1. ..."
      - alpha substeps: "a. ..." / "a) ..."
      - bullet substeps: "- ..."
    """
    steps: List[Dict[str, str]] = []
    current_top: Optional[str] = None
    bullet_counts: Dict[str, int] = {}
    last_idx: Optional[int] = None

    for raw in str(chapter_timeline or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("CHAPTER:"):
            continue

        m_top = re.match(r"^(\d+)\.\s*(.+)$", line)
        if m_top:
            current_top = m_top.group(1)
            key = current_top
            steps.append({"key": key, "timeline_text": m_top.group(2).strip()})
            last_idx = len(steps) - 1
            continue

        m_alpha = re.match(r"^([a-zA-Z])[\.\)]\s*(.+)$", line)
        if m_alpha and current_top:
            sub = m_alpha.group(1).lower()
            key = f"{current_top}.{sub}"
            steps.append({"key": key, "timeline_text": m_alpha.group(2).strip()})
            last_idx = len(steps) - 1
            continue

        m_bullet = re.match(r"^[-*]\s*(.+)$", line)
        if m_bullet and current_top:
            c = int(bullet_counts.get(current_top, 0)) + 1
            bullet_counts[current_top] = c
            key = f"{current_top}.sub{c}"
            steps.append({"key": key, "timeline_text": m_bullet.group(1).strip()})
            last_idx = len(steps) - 1
            continue

        if last_idx is not None:
            prev = steps[last_idx]["timeline_text"]
            steps[last_idx]["timeline_text"] = (prev + " " + line).strip()

    if not steps:
        fallback = _normalize_ws(chapter_timeline)
        if fallback:
            return [{"key": "1", "timeline_text": fallback}]
        return [{"key": "1", "timeline_text": "lesson step"}]
    return steps


def _build_even_step_ranges(word_count: int, step_order: List[str]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    n = max(0, int(word_count))
    if not step_order:
        return out
    if n <= 0:
        for k in step_order:
            out[k] = {"start_word_index": 0, "end_word_index": -1}
        return out

    base = n // len(step_order)
    rem = n % len(step_order)
    cursor = 0
    for i, k in enumerate(step_order):
        ln = base + (1 if i < rem else 0)
        s = cursor
        e = cursor + ln - 1
        out[k] = {"start_word_index": s, "end_word_index": e}
        cursor = e + 1
    return out


def build_flat_speech_and_ranges(
    *,
    full_speech: str,
    step_order: List[str],
    step_speech_map: Dict[str, str],
) -> Tuple[str, Dict[str, Dict[str, int]]]:
    parts: List[str] = []
    ranges: Dict[str, Dict[str, int]] = {}
    cursor = 0
    non_empty_parts = 0

    for key in step_order:
        seg = _normalize_ws(step_speech_map.get(key, ""))
        if not seg:
            ranges[key] = {"start_word_index": cursor, "end_word_index": cursor - 1}
            continue
        words = seg.split()
        s = cursor
        e = cursor + len(words) - 1
        ranges[key] = {"start_word_index": s, "end_word_index": e}
        cursor = e + 1
        parts.append(seg)
        non_empty_parts += 1

    if non_empty_parts > 0:
        return " ".join(parts).strip(), ranges

    flat = _normalize_ws(full_speech)
    words = flat.split()
    return flat, _build_even_step_ranges(len(words), step_order)




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


def prompt_teacher_speech(
    topic: str,
    chapter_timeline: str,
    chapter_index: int,
    chapter_count: int,
    step_keys: List[str],
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
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
        "STRUCTURE RULES:\n"
        "- The speech is ONE continuous chapter narration for the whole chunk.\n"
        "- You must also split it mechanically by step keys for formatting only.\n"
        "- These step chunks are not standalone mini speeches; they are slices of one continuous narration.\n"
        "- Keep every provided step key exactly once in step_speech.\n\n"
        f"word cap : {MAX_WORDS_PER_CHAPTER}"
        "Hard rules:\n"
        "- Output JSON only, following the provided schema.\n"
        "- Do NOT output markdown code fences.\n"
    )
    user = (
        f"Topic: {topic}\n"
        f"Chapter: {chapter_index}/{chapter_count}\n\n"
        "Required step keys (must appear exactly once in step_speech.step_key, in the same order):\n"
        f"{json.dumps(step_keys, ensure_ascii=False)}\n\n"
        "Chapter timeline:\n"
        f"{chapter_timeline}\n\n"
        "Generate the full narration for this chapter only."
    )
    input_items = [{"role": "developer", "content": developer}, {"role": "user", "content": user}]
    text_format = {
        "format": {
            "type": "json_schema",
            "name": "speech_plan",
            "schema": SPEECH_PLAN_SCHEMA,
            "strict": True,
        }
    }
    return input_items, text_format


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
        "- If text (text=1): content is the exact text to write on the board.\n"
        "OBJECTS QUERYING COMPARISON:\n"
        "- With diagrams you should only request small optimized queries - NO EXPLANATIONS FOR CONNTENT, DETAILS VISUAL,\n"
        "just the most simple possible request\n"
        "- YOU ARE NAMING JUST A SINGLE OBJECT WITH ONLY A COUPLE OF WORDS, NO CLARIFICATION"
        "- You request detail for diagrams by listing concrete objects in diagram_required_objects\n"
        "- With normal images (diagram=0) you should include explanation on specific visual in the content field."
        "- You are to describe both the normal characterisctics of your image, but also STYLISTIC characteristics:"
        "- That includes description of composition of inner (mentioned) elements + requesting either a preffered cartoonish / drawn look, or realistic (in edge cases)\n"
        "- INCLUDE VISUAL DESCRIPTIONS ONLY FOR NORMAL IMAGES - NO SORT OF DESCRIPTION FOR DIAGRAMS - NO VISUAL DESCRIPTION"
        "- With text you putting a litteral string that is printed 1:1 on the board"
        "On ranges:\n"
        "- you really gotta focus on having a good active range for each image -> as long as needed\n"
        "- ranges can overlap ESPECIALLY when its a diagram -> it can be active and used for a long time\n"
        "Topic behavior:\n"
        "- You must include a 'topic' for every entry.\n"
        "- For image entries (text=0), topic must be the broader subject umbrella that contains the specific concept.\n"
        "- Keep topic specific enough to be useful, but still general enough to include nearby concepts.\n"
        "- Example: image content='map of the German invasion of France in WW2' -> topic='WW2 History of Europe'.\n"
        "- THE TOPIC NEEDS TO BE THE BROADEST POSSIBLE CONCEPT THE PROMPT CAN SIT IN (ex. nucleus -> cell biology)"
        "- For text entries (text=1), set topic to a stable lesson-level umbrella topic too (not empty).\n"
        "- Output must follow the provided JSON schema exactly.\n"
        "Keep requested images as concrete things, easily got with img generation / search engines.\n"
        "Have different objects all around -> do not request the same thing in different places -> if it's used for a while JUST GIVE A BIG RANGE FOR IT\n"
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


def _estimate_text_bbox_px(write_text: str, scale: float = 1.0) -> Tuple[int, int]:
    txt = str(write_text or "")
    w = int(max(20.0, float(len(txt)) * 15.0 * float(scale)))
    h = int(max(16.0, 30.0 * float(scale)))
    return (w, h)


def _step_key_for_span(
    *,
    start_word_index: int,
    end_word_index: int,
    step_order: List[str],
    step_ranges: Dict[str, Dict[str, int]],
) -> str:
    if not step_order:
        return "1"

    best_key = step_order[0]
    best_overlap = -1
    best_dist = 10**9

    s0 = int(start_word_index)
    e0 = int(end_word_index)
    if e0 < s0:
        s0, e0 = e0, s0

    for key in step_order:
        rr = step_ranges.get(key) or {}
        ss = int(rr.get("start_word_index", 0) or 0)
        ee = int(rr.get("end_word_index", ss) or ss)
        if ee < ss:
            ee = ss

        ov = max(0, min(e0, ee) - max(s0, ss) + 1)
        if ov > best_overlap:
            best_overlap = ov
            best_key = key
            best_dist = abs(s0 - ss)
            continue
        if ov == best_overlap:
            d = abs(s0 - ss)
            if d < best_dist:
                best_dist = d
                best_key = key

    return best_key


def build_chunk_sync_map_for_chapter(
    *,
    chapter_index: int,
    speech_text: str,
    image_text_maps: List[Dict[str, Any]],
    speech_step_order: List[str],
    speech_step_ranges: Dict[str, Dict[str, int]],
    timeline_steps: Optional[List[Dict[str, Any]]] = None,
    assets_by_name: Dict[str, ImageAsset],
    board_w: int = 4000,
    board_h: int = 4000,
    include_inter_chunk_silence: bool = True,
) -> Dict[str, Any]:
    words = str(speech_text or "").split()
    n = len(words)

    step_order = [str(x or "").strip() for x in (speech_step_order or []) if str(x or "").strip()]
    if not step_order:
        step_order = ["1"]
    if not isinstance(speech_step_ranges, dict) or not speech_step_ranges:
        speech_step_ranges = _build_even_step_ranges(n, step_order)

    entries: List[Dict[str, Any]] = []
    entry_index = 0

    for im in (image_text_maps or []):
        if not isinstance(im, dict):
            continue
        s = int(im.get("range_start", 0) or 0)
        e = int(im.get("range_end", s) or s)
        if n > 0:
            s = max(0, min(s, n - 1))
            e = max(0, min(e, n - 1))
        if e < s:
            s, e = e, s

        text_tag = 1 if int(im.get("text_tag", 0) or 0) == 1 else 0
        diag = 1 if int(im.get("diagram", 0) or 0) == 1 else 0
        content = str(im.get("content", "") or "").strip()
        name = content or f"entry_{entry_index + 1}"
        write_text = str(im.get("write_text", "") or "").strip() if text_tag == 1 else ""
        query = str(im.get("query", "") or "").strip()
        span_text = str(im.get("span_text", "") or "").strip()
        if not span_text and n > 0 and e >= s:
            span_text = " ".join(words[s : e + 1])
        span_text = _strip_pause_tokens(span_text)

        bbox = None
        processed_id = None
        objects: List[Any] = []
        refined_file = None

        req_objects = im.get("diagram_required_objects")
        if isinstance(req_objects, list):
            for it in req_objects:
                s0 = str(it or "").strip()
                if s0:
                    objects.append(s0)

        if text_tag == 1:
            bbox = _estimate_text_bbox_px(write_text or content, scale=1.0)
        else:
            asset = assets_by_name.get(name) if name else None
            if asset is None and query:
                asset = assets_by_name.get(query)
            if asset is not None:
                bbox = asset.bbox_px or (400, 300)
                processed_id = asset.processed_ids[0] if asset.processed_ids else None
                if isinstance(asset.objects, list):
                    for o in asset.objects:
                        if isinstance(o, str):
                            s0 = o.strip()
                            if s0:
                                objects.append(s0)
                        elif isinstance(o, dict):
                            nm = str(o.get("name", "") or "").strip()
                            if nm:
                                objects.append(nm)
                refined_file = asset.refined_labels_file
            else:
                bbox = (400, 300)

        # keep stable order but dedupe strings case-insensitively
        norm_objs: List[str] = []
        seen_obj = set()
        for o in objects:
            s0 = str(o or "").strip()
            if not s0:
                continue
            k0 = s0.lower()
            if k0 in seen_obj:
                continue
            seen_obj.add(k0)
            norm_objs.append(s0)

        step_key = _step_key_for_span(
            start_word_index=s,
            end_word_index=e,
            step_order=step_order,
            step_ranges=speech_step_ranges,
        )

        entries.append(
            {
                "entry_index": int(entry_index),
                "type": "text" if text_tag == 1 else "image",
                "name": name,
                "content": content,
                "query": query,
                "step_key": step_key,
                "range_start": int(s),
                "range_end": int(e),
                "duration_sec": None,
                "diagram": int(diag),
                "text_tag": int(text_tag),
                "write_text": write_text,
                "speech_text_in_range": span_text,
                "bbox_px": {"w": int(bbox[0]), "h": int(bbox[1])} if bbox else {"w": 400, "h": 300},
                "processed_id": processed_id,
                "objects_that_comprise_image": norm_objs,
                "refined_labels_file": refined_file,
                "chunk_boundary_silence": False,
                "delete_all": False,
            }
        )
        entry_index += 1

    for wi, tok in enumerate(words):
        sec = _is_pause_token(tok)
        if sec is None:
            continue
        sk = _step_key_for_span(
            start_word_index=wi,
            end_word_index=wi,
            step_order=step_order,
            step_ranges=speech_step_ranges,
        )
        entries.append(
            {
                "entry_index": int(entry_index),
                "type": "silence",
                "name": f"silence_{chapter_index}_{entry_index}",
                "content": "",
                "query": "",
                "step_key": sk,
                "range_start": int(wi),
                "range_end": int(wi),
                "duration_sec": float(sec),
                "diagram": 0,
                "text_tag": 0,
                "write_text": "",
                "speech_text_in_range": _strip_pause_tokens(" ".join(words[max(0, wi - 10) : min(n, wi + 11)])),
                "bbox_px": None,
                "processed_id": None,
                "objects_that_comprise_image": [],
                "refined_labels_file": None,
                "chunk_boundary_silence": False,
                "delete_all": False,
            }
        )
        entry_index += 1

    if include_inter_chunk_silence:
        wi = n + 1
        sk = _step_key_for_span(
            start_word_index=wi,
            end_word_index=wi,
            step_order=step_order,
            step_ranges=speech_step_ranges,
        )
        entries.append(
            {
                "entry_index": int(entry_index),
                "type": "silence",
                "name": f"silence_boundary_{chapter_index}",
                "content": "",
                "query": "",
                "step_key": sk,
                "range_start": int(wi),
                "range_end": int(wi),
                "duration_sec": float(INTER_CHUNK_SILENCE_SEC),
                "diagram": 0,
                "text_tag": 0,
                "write_text": "",
                "speech_text_in_range": _strip_pause_tokens(" ".join(words[max(0, n - 20) : n])),
                "bbox_px": None,
                "processed_id": None,
                "objects_that_comprise_image": [],
                "refined_labels_file": None,
                "chunk_boundary_silence": True,
                "delete_all": True,
            }
        )

    entries.sort(key=lambda e: (int(e.get("range_start", 0) or 0), 0 if str(e.get("type", "")) == "silence" else 1))
    for i, e in enumerate(entries):
        e["entry_index"] = int(i)

    step_rows = []
    timeline_map: Dict[str, str] = {}
    if isinstance(timeline_steps, list):
        for row in timeline_steps:
            if not isinstance(row, dict):
                continue
            k = str(row.get("key", "") or "").strip()
            if not k:
                continue
            timeline_map[k] = str(row.get("timeline_text", "") or "").strip()
    for sk in step_order:
        rr = speech_step_ranges.get(sk) or {}
        step_rows.append(
            {
                "key": sk,
                "timeline_text": timeline_map.get(sk, ""),
                "start_word_index": int(rr.get("start_word_index", 0) or 0),
                "end_word_index": int(rr.get("end_word_index", -1) or -1),
            }
        )

    return {
        "chapter_index": int(chapter_index),
        "board": {"w": int(board_w), "h": int(board_h)},
        "step_order": step_order,
        "steps": step_rows,
        "entries": entries,
    }


def split_entries_by_deletion_silence(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns ordered chunks:
      [{"visual_entries":[...], "deletion_silence": {...} | None}, ...]
    """
    out: List[Dict[str, Any]] = []
    cur: List[Dict[str, Any]] = []

    for e in (entries or []):
        t = str(e.get("type", "") or "").strip().lower()
        if t != "silence":
            cur.append(e)
            continue
        if not bool(e.get("delete_all", False)):
            continue
        out.append({"visual_entries": list(cur), "deletion_silence": e})
        cur = []

    if cur:
        out.append({"visual_entries": list(cur), "deletion_silence": None})
    return out


def _chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    size = max(1, int(size))
    out: List[List[Any]] = []
    for i in range(0, len(items), size):
        out.append(items[i : i + size])
    return out


def convert_local_sync_to_absolute(
    action: Dict[str, Any],
    *,
    event_start_word: int,
    event_end_word: int,
) -> Dict[str, Any]:
    a = dict(action or {})
    s_evt = int(event_start_word)
    e_evt = int(event_end_word)
    if e_evt < s_evt:
        e_evt = s_evt
    span = max(0, e_evt - s_evt)

    sync = a.get("sync_local")
    if not isinstance(sync, dict):
        sync = {}
    ls = int(sync.get("start_word_offset", sync.get("start", 0)) or 0)
    le = int(sync.get("end_word_offset", sync.get("end", ls)) or ls)

    ls = max(0, min(ls, span))
    le = max(ls, min(le, span))

    abs_s = s_evt + ls
    abs_e = s_evt + le

    a["sync_local"] = {
        "start_word_offset": int(ls),
        "end_word_offset": int(le),
    }
    a["sync_absolute"] = {
        "start_word_index": int(abs_s),
        "end_word_index": int(abs_e),
    }
    return a


_VISUAL_ACTION_ALIAS_STATIC: Dict[str, str] = {
    "draw_image": "draw_image",
    "draw": "draw_image",
    "write_text": "write_text",
    "write": "write_text",
    "annotate": "write_text",
    "move_inside_bbox": "move_inside_bbox",
    "move_inside_print_bbox": "move_inside_bbox",
    "move": "move_inside_bbox",
    "link_to_image": "link_to_image",
    "link_to_previous": "link_to_image",
    "link": "link_to_image",
    "delete_self": "delete_self",
    "delete_by_name": "delete_by_name",
    "highlight_cluster": "highlight_cluster",
    "unhighlight_cluster": "unhighlight_cluster",
    "zoom_cluster": "zoom_cluster",
    "unzoom_cluster": "unzoom_cluster",
    "write_label": "write_label",
}

_DIAGRAM_ACTION_TYPES_STATIC = {
    "highlight_cluster",
    "unhighlight_cluster",
    "zoom_cluster",
    "unzoom_cluster",
    "write_label",
}


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return int(default)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _normalize_sync_local_for_req(sync: Any, *, req_start: int, req_end: int) -> Dict[str, int]:
    if req_end < req_start:
        req_end = req_start
    span = max(0, req_end - req_start)
    if not isinstance(sync, dict):
        sync = {}
    s = _safe_int(sync.get("start_word_offset", sync.get("start", 0)), 0)
    e = _safe_int(sync.get("end_word_offset", sync.get("end", s)), s)
    s = max(0, min(s, span))
    e = max(s, min(e, span))
    return {"start_word_offset": int(s), "end_word_offset": int(e)}


def _xy_from_action_fields(
    action: Dict[str, Any],
    *,
    default_x: int,
    default_y: int,
    x_key: str = "x",
    y_key: str = "y",
) -> Tuple[int, int]:
    loc = action.get("location")
    xv = action.get(x_key, None)
    yv = action.get(y_key, None)
    if isinstance(loc, dict):
        if xv is None:
            xv = loc.get("x")
        if yv is None:
            yv = loc.get("y")
    x = _safe_int(xv, default_x)
    y = _safe_int(yv, default_y)
    return x, y


def normalize_static_visual_action(
    action: Dict[str, Any],
    *,
    req: Dict[str, Any],
    action_index: int,
) -> Optional[Dict[str, Any]]:
    if not isinstance(action, dict):
        return None
    raw_t = str(action.get("type", "") or "").strip().lower()
    if not raw_t:
        return None
    t = _VISUAL_ACTION_ALIAS_STATIC.get(raw_t)
    if not t:
        return None

    is_diagram = int(req.get("diagram", 0) or 0) == 1
    if t in _DIAGRAM_ACTION_TYPES_STATIC and not is_diagram:
        return None

    req_name = str(req.get("name", "") or "").strip()
    req_start = int(req.get("range_start", 0) or 0)
    req_end = int(req.get("range_end", req_start) or req_start)
    pb = req.get("print_bbox") if isinstance(req.get("print_bbox"), dict) else {"x": 0, "y": 0, "w": 400, "h": 300}
    pbx = _safe_int(pb.get("x", 0), 0)
    pby = _safe_int(pb.get("y", 0), 0)

    out: Dict[str, Any] = {"type": t}

    if t == "draw_image":
        x, y = _xy_from_action_fields(action, default_x=pbx, default_y=pby, x_key="x", y_key="y")
        out["target"] = str(action.get("target", "") or req_name).strip()
        out["x"] = int(x)
        out["y"] = int(y)
    elif t == "write_text":
        x, y = _xy_from_action_fields(action, default_x=pbx, default_y=pby, x_key="x", y_key="y")
        text_fallback = str(req.get("write_text", "") or req.get("content", "")).strip()
        out["text"] = str(action.get("text", "") or text_fallback)
        out["x"] = int(x)
        out["y"] = int(y)
        out["scale"] = float(_safe_float(action.get("scale", 1.0), 1.0))
        tgt = str(action.get("target", "") or "").strip()
        if not tgt:
            tgt = req_name if int(req.get("text_tag", 0) or 0) == 1 else f"{req_name}__text_{action_index + 1}"
        out["target"] = tgt
    elif t == "move_inside_bbox":
        nx, ny = _xy_from_action_fields(action, default_x=pbx, default_y=pby, x_key="new_x", y_key="new_y")
        if "new_x" not in action and "new_y" not in action:
            nx, ny = _xy_from_action_fields(action, default_x=pbx, default_y=pby, x_key="x", y_key="y")
        out["target"] = str(action.get("target", "") or req_name).strip()
        out["new_x"] = int(nx)
        out["new_y"] = int(ny)
    elif t == "link_to_image":
        other = (
            str(action.get("image_name", "") or action.get("target_image", "") or action.get("link_to_image", "") or action.get("to", "") or "")
            .strip()
        )
        if not other:
            other = str(action.get("target", "") or "").strip()
            if other.lower() == req_name.lower():
                other = ""
        if not other:
            return None
        out["target"] = req_name
        out["image_name"] = other
    elif t == "delete_self":
        pass
    elif t == "delete_by_name":
        tgt = str(action.get("target", "") or "").strip()
        if not tgt:
            tgt = req_name
        if not tgt:
            return None
        out["target"] = tgt
    elif t in _DIAGRAM_ACTION_TYPES_STATIC:
        objects = req.get("objects_that_comprise_image") if isinstance(req.get("objects_that_comprise_image"), list) else []
        first_obj = str(objects[0] if objects else "").strip()
        cname = str(
            action.get("cluster_name", "")
            or action.get("cluster", "")
            or action.get("object", "")
            or action.get("name", "")
            or first_obj
        ).strip()
        if not cname:
            return None
        out["cluster_name"] = cname
        if t == "write_label":
            out["text"] = str(action.get("text", "") or cname)

    out["sync_local"] = _normalize_sync_local_for_req(
        action.get("sync_local"),
        req_start=req_start,
        req_end=req_end,
    )
    return out


def build_active_object_from_action(
    action: Dict[str, Any],
    *,
    req: Dict[str, Any],
    action_index: int,
    chapter_index: int,
    chunk_index: int,
    segment_index: int,
    batch_index: int,
) -> Optional[Dict[str, Any]]:
    if not isinstance(action, dict):
        return None
    t = str(action.get("type", "") or "").strip().lower()
    if t == "draw_image":
        nm = str(action.get("target", "") or req.get("name", "") or "").strip()
        if not nm:
            return None
        pb = req.get("print_bbox") if isinstance(req.get("print_bbox"), dict) else {}
        x = _safe_int(action.get("x", pb.get("x", 0)), _safe_int(pb.get("x", 0), 0))
        y = _safe_int(action.get("y", pb.get("y", 0)), _safe_int(pb.get("y", 0), 0))
        bb = {
            "x": int(x),
            "y": int(y),
            "w": _safe_int((req.get("bbox_px") or {}).get("w", pb.get("w", 400)), 400),
            "h": _safe_int((req.get("bbox_px") or {}).get("h", pb.get("h", 300)), 300),
        }
        return {
            "name": nm,
            "kind": "image",
            "source_type": str(req.get("type", "image") or "image"),
            "bbox": bb,
            "chapter_index": int(chapter_index),
            "chunk_index": int(chunk_index),
            "segment_index": int(segment_index),
            "batch_index": int(batch_index),
            "action_index": int(action_index),
        }
    if t == "write_text":
        nm = str(action.get("target", "") or "").strip()
        if not nm:
            nm = f"{str(req.get('name', '') or 'text').strip()}__text_{action_index + 1}"
        txt = str(action.get("text", "") or "")
        scale = float(_safe_float(action.get("scale", 1.0), 1.0))
        w, h = _estimate_text_bbox_px(txt, scale=scale)
        x = _safe_int(action.get("x", 0), 0)
        y = _safe_int(action.get("y", 0), 0)
        return {
            "name": nm,
            "kind": "text",
            "source_type": str(req.get("type", "text") or "text"),
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "text": txt,
            "scale": scale,
            "chapter_index": int(chapter_index),
            "chunk_index": int(chunk_index),
            "segment_index": int(segment_index),
            "batch_index": int(batch_index),
            "action_index": int(action_index),
        }
    return None


def sorted_active_objects_for_cleanup(active_objects_by_key: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [v for v in (active_objects_by_key or {}).values() if isinstance(v, dict)]
    rows.sort(
        key=lambda r: (
            int(((r.get("bbox") or {}).get("y", 10_000_000) or 10_000_000)),
            int(((r.get("bbox") or {}).get("x", 10_000_000) or 10_000_000)),
            int(r.get("created_order", 10_000_000) or 10_000_000),
            str(r.get("name", "") or "").lower(),
        )
    )
    return rows


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

class LessonTimeline:
    def __init__(
    self,
    *,
    model_candidates: Optional[List[str]] = None,
    reasoning_effort: str = "low",
    temperature: float = 0.4,
    gpu_index: int = 0,
    cpu_threads: int = 4,
    whiteboard_width: int = 4000,
    whiteboard_height: int = 4000,
    processed_images_dir: str = "ProcessedImages",
    debug_timeline_json: bool = False,     # NEW: controls extra debug fields in OUTPUT JSON
    debug_print: bool = True,              # NEW: heavy console debug
    debug_out_dir: str = "PipelineOutputs", # NEW: where we save action timeline json
    gpt_cache: bool = False,
    gpt_cache_file: Optional[str] = None,
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
        self.gpt_cache_enabled = bool(gpt_cache)
        self.gpt_cache_file = str(gpt_cache_file or (Path(self.debug_out_dir) / "_gpt_call_cache.json"))
        self._gpt_cache_lock = threading.Lock()
        self._gpt_cache: Dict[str, Dict[str, str]] = {}
        if self.gpt_cache_enabled:
            try:
                p = Path(self.gpt_cache_file)
                if p.is_file():
                    raw = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        self._gpt_cache = {
                            str(k): v
                            for k, v in raw.items()
                            if isinstance(v, dict)
                            and isinstance(v.get("text"), str)
                            and isinstance(v.get("model"), str)
                        }
            except Exception:
                self._gpt_cache = {}

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

    def _llm_cache_key(
        self,
        *,
        model_candidates: List[str],
        input_items: List[Dict[str, str]],
        reasoning_effort: str,
        temperature: float,
        max_output_tokens: Optional[int],
        text_format: Optional[Dict[str, Any]],
    ) -> str:
        payload = {
            "models": list(model_candidates or []),
            "input": input_items,
            "reasoning_effort": str(reasoning_effort),
            "temperature": float(temperature),
            "max_output_tokens": int(max_output_tokens) if max_output_tokens is not None else None,
            "text_format": text_format,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _persist_gpt_cache(self) -> None:
        if not self.gpt_cache_enabled:
            return
        try:
            p = Path(self.gpt_cache_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self._gpt_cache, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _call_responses_text_cached(
        self,
        input_items: List[Dict[str, str]],
        *,
        reasoning_effort: str,
        temperature: float,
        max_output_tokens: Optional[int] = None,
        text_format: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        if not self.gpt_cache_enabled:
            return call_responses_text(
                self.client,
                self.model_candidates,
                input_items,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                text_format=text_format,
            )

        key = self._llm_cache_key(
            model_candidates=self.model_candidates,
            input_items=input_items,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            text_format=text_format,
        )

        with self._gpt_cache_lock:
            hit = self._gpt_cache.get(key)
            if isinstance(hit, dict):
                txt = hit.get("text")
                mdl = hit.get("model")
                if isinstance(txt, str) and isinstance(mdl, str):
                    self._dbg("LLM cache hit", data={"key": key[:10]})
                    return txt, mdl

        txt, mdl = call_responses_text(
            self.client,
            self.model_candidates,
            input_items,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            text_format=text_format,
        )

        with self._gpt_cache_lock:
            self._gpt_cache[key] = {"text": txt, "model": mdl}
            self._persist_gpt_cache()
        self._dbg("LLM cache store", data={"key": key[:10], "model": mdl})
        return txt, mdl

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
            timeline_text, used_model = self._call_responses_text_cached(
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
    ) -> Tuple[Dict[str, Any], str]:
        parsed_steps = parse_chapter_timeline_steps(chapter.raw)
        step_order = [str(x.get("key", "") or "").strip() for x in parsed_steps]
        step_order = [x for x in step_order if x]
        if not step_order:
            step_order = ["1"]

        with self._timed("llm.speech", chapter_index=chapter_index, chapter_count=chapter_count):
            input_items, text_format = prompt_teacher_speech(
                topic,
                chapter.raw,
                chapter_index,
                chapter_count,
                step_order,
            )
            speech_json, used_model = self._call_responses_text_cached(
                input_items,
                reasoning_effort=self.reasoning_effort,
                temperature=self.temperature,
                text_format=text_format,
            )
        parsed = json.loads(speech_json)
        full_speech = _normalize_ws(str(parsed.get("full_speech", "") or ""))
        raw_step_speech = parsed.get("step_speech") if isinstance(parsed, dict) else []

        step_speech_map: Dict[str, str] = {k: "" for k in step_order}
        if isinstance(raw_step_speech, list):
            for row in raw_step_speech:
                if not isinstance(row, dict):
                    continue
                key = str(row.get("step_key", "") or "").strip()
                if key not in step_speech_map:
                    continue
                step_speech_map[key] = _normalize_ws(str(row.get("speech", "") or ""))

        flat_speech, step_ranges = build_flat_speech_and_ranges(
            full_speech=full_speech,
            step_order=step_order,
            step_speech_map=step_speech_map,
        )

        if not flat_speech:
            flat_speech = full_speech
        if not flat_speech:
            flat_speech = "Lesson continuation >1.0"

        payload = {
            "speech": flat_speech,
            "speech_step_order": list(step_order),
            "speech_step_map": step_speech_map,
            "speech_step_ranges": step_ranges,
            "timeline_steps": parsed_steps,
        }
        self._dbg("Speech generated", data={"chapter_index": chapter_index, "chars": len(flat_speech), "steps": len(step_order)})
        return payload, used_model

    def generate_images_for_speech(self, speech_text: str, *, chapter_index: Optional[int] = None) -> Tuple[Dict[str, Any], str]:
        with self._timed("llm.image_plan", chapter_index=chapter_index):
            input_items, text_format = prompt_image_requests(speech_text)
            json_text, used_model = self._call_responses_text_cached(
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

    def _start_siglip_loader(self, *, label: str, warmup: bool = True) -> threading.Thread:
        def _loader():
            from shared_models import init_siglip_text_hot
            with self._timed(label, gpu_index=self.gpu_index, cpu_threads=self.cpu_threads):
                init_siglip_text_hot(
                    gpu_index=self.gpu_index,
                    cpu_threads=self.cpu_threads,
                    warmup=bool(warmup),
                    prefer_fp8=True,
                )

        t = threading.Thread(target=_loader, name=f"{label}_thread", daemon=False)
        t.start()
        return t

    def _start_minilm_loader(self, *, label: str, warmup: bool = True) -> threading.Thread:
        def _loader():
            from shared_models import init_minilm_hot
            with self._timed(label, gpu_index=self.gpu_index, cpu_threads=self.cpu_threads):
                init_minilm_hot(
                    gpu_index=self.gpu_index,
                    cpu_threads=self.cpu_threads,
                    warmup=bool(warmup),
                )

        t = threading.Thread(target=_loader, name=f"{label}_thread", daemon=False)
        t.start()
        return t

    def _load_siglip_minilm_async(self) -> threading.Thread:
        """
        Backward-compatible wrapper that loads MiniLM + SigLIP-text in one helper thread.
        New flow prefers split loaders.
        """
        def _loader():
            from shared_models import init_minilm_hot, init_siglip_text_hot
            with self._timed("models.load_minilm", gpu_index=self.gpu_index, cpu_threads=self.cpu_threads):
                init_minilm_hot(gpu_index=self.gpu_index, cpu_threads=self.cpu_threads, warmup=True)
            with self._timed("models.load_siglip_text", gpu_index=self.gpu_index, cpu_threads=self.cpu_threads):
                init_siglip_text_hot(gpu_index=self.gpu_index, cpu_threads=self.cpu_threads, warmup=True, prefer_fp8=True)

        t = threading.Thread(target=_loader, name="load_siglip_minilm", daemon=False)
        t.start()
        return t

    def _group_and_rewrite_diagram_prompts(
        self,
        *,
        all_prompts_meta: Dict[str, Dict[str, Any]],
        all_diagram_required_objects: Dict[str, List[str]],
        chapters_out: List[Dict[str, Any]],
        similarity_threshold: float = 0.84,
        relaxed_topic_similarity: float = 0.56,
        centroid_margin_for_umbrella: float = 0.06,
    ) -> Dict[str, Any]:
        """
        Diagram-only prompt consolidation:
          1) embed each unique diagram prompt
          2) compare all-vs-all with cosine similarity
          3) form connected groups:
             - strict semantic match (>= similarity_threshold)
             - relaxed, topic-aware umbrella match for broad reusable diagrams
          4) pick representative by centroid proximity with umbrella/coverage bias
          5) rewrite downstream prompt usage to representative
        """
        try:
            import numpy as np
            from shared_models import get_minilm, init_minilm_hot, minilm_embed_texts
        except Exception as e:
            return {
                "ok": False,
                "error": f"imports_failed: {type(e).__name__}: {e}",
            }

        diag_prompts: List[str] = []
        for p_raw, meta in (all_prompts_meta or {}).items():
            p = str(p_raw or "").strip()
            if not p:
                continue
            d = 0
            try:
                d = 1 if int((meta or {}).get("diagram", 0) or 0) == 1 else 0
            except Exception:
                d = 0
            if d == 1:
                diag_prompts.append(p)

        # Preserve insertion order, unique by exact string.
        uniq_diag_prompts = list(dict.fromkeys(diag_prompts))
        if len(uniq_diag_prompts) < 2:
            return {
                "ok": True,
                "diagram_prompts": len(uniq_diag_prompts),
                "groups_found": 0,
                "groups_rewritten": 0,
                "replacement_map": {},
                "note": "not_enough_diagram_prompts",
            }

        def _norm_topic(s: str) -> str:
            return re.sub(r"\s+", " ", str(s or "").strip().lower())

        _stop_terms = {
            "the", "and", "with", "for", "from", "into", "near", "showing", "shown", "clean",
            "style", "cartoon", "teacher", "drawn", "whiteboard", "look", "simple", "diagram",
            "label", "labeled", "labels", "image", "visual", "view",
        }

        def _tokenize_terms(s: str) -> set:
            toks = re.findall(r"[a-z0-9]+", str(s or "").lower())
            return {t for t in toks if len(t) >= 3 and t not in _stop_terms}

        relaxed_topic_similarity = max(0.40, min(float(relaxed_topic_similarity), float(similarity_threshold) - 0.03))
        centroid_margin_for_umbrella = max(0.0, min(float(centroid_margin_for_umbrella), 0.20))

        if get_minilm() is None:
            with self._timed("models.load_minilm_for_diagram_grouping", gpu_index=self.gpu_index, cpu_threads=self.cpu_threads):
                init_minilm_hot(
                    gpu_index=self.gpu_index,
                    cpu_threads=self.cpu_threads,
                    warmup=True,
                )

        vecs = minilm_embed_texts(uniq_diag_prompts)
        if not isinstance(vecs, list) or len(vecs) != len(uniq_diag_prompts):
            return {
                "ok": False,
                "diagram_prompts": len(uniq_diag_prompts),
                "groups_found": 0,
                "groups_rewritten": 0,
                "replacement_map": {},
                "error": "minilm_embed_failed",
            }

        mat = np.asarray(vecs, dtype=np.float32)
        if mat.ndim != 2 or mat.shape[0] != len(uniq_diag_prompts):
            return {
                "ok": False,
                "diagram_prompts": len(uniq_diag_prompts),
                "groups_found": 0,
                "groups_rewritten": 0,
                "replacement_map": {},
                "error": "invalid_embedding_shape",
            }

        # Normalize once, then cosine matrix via dot product.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1e-9)
        mat = mat / norms
        sims = np.matmul(mat, mat.T)

        n = len(uniq_diag_prompts)
        prompt_topics = [
            _norm_topic(str((all_prompts_meta.get(p) or {}).get("topic", "") or ""))
            for p in uniq_diag_prompts
        ]
        prompt_topic_terms = [_tokenize_terms(t) for t in prompt_topics]

        required_objs_norm: List[List[str]] = []
        concept_terms: List[set] = []
        coverage_scores: List[float] = []
        umbrella_flags: List[int] = []

        for p in uniq_diag_prompts:
            text = str(p or "").strip()
            low = text.lower()
            objs_raw = all_diagram_required_objects.get(p) or []
            objs = [str(x).strip().lower() for x in objs_raw if str(x).strip()]
            required_objs_norm.append(objs)

            rt: set = set()
            for o in objs:
                rt |= _tokenize_terms(o)

            pt = _tokenize_terms(low)
            ct = set(pt) | set(rt)
            concept_terms.append(ct)

            cc = text.count(",")
            wc = len(re.findall(r"\w+", text))

            cov = float(len(objs))
            cov += min(12.0, float(cc) * 0.8)
            if "labeled" in low or "labelled" in low:
                cov += 3.0
            if "cell" in low:
                cov += 2.5
            if "cross section" in low or "cross-section" in low:
                cov += 1.5
            if "overview" in low:
                cov += 1.0
            coverage_scores.append(cov)

            is_umbrella = (
                (len(objs) >= 6)
                or (cc >= 5 and ("cell" in low or "diagram" in low))
                or (wc >= 20 and ("labeled" in low or "labelled" in low))
            )
            umbrella_flags.append(1 if is_umbrella else 0)

        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra = _find(a)
            rb = _find(b)
            if ra != rb:
                parent[rb] = ra

        merge_rule_counts: Dict[str, int] = {
            "strict_similarity": 0,
            "topic_required_overlap": 0,
            "topic_umbrella_similarity": 0,
            "topic_umbrella_anchor": 0,
            "topic_semantic_overlap": 0,
            "cross_topic_anchor": 0,
        }

        for i in range(n):
            for j in range(i + 1, n):
                sim_ij = float(sims[i, j])
                same_topic = bool(prompt_topics[i] and (prompt_topics[i] == prompt_topics[j]))
                tt_i = prompt_topic_terms[i]
                tt_j = prompt_topic_terms[j]
                min_tt = max(1, min(len(tt_i), len(tt_j)))
                topic_term_overlap = float(len(tt_i & tt_j)) / float(min_tt)
                topic_related = (
                    same_topic
                    or topic_term_overlap >= 0.20
                    or not prompt_topics[i]
                    or not prompt_topics[j]
                )

                terms_i = concept_terms[i]
                terms_j = concept_terms[j]
                common_terms = terms_i & terms_j
                min_terms = max(1, min(len(terms_i), len(terms_j)))
                lex_overlap = float(len(common_terms)) / float(min_terms)

                objs_i = set(required_objs_norm[i])
                objs_j = set(required_objs_norm[j])
                req_overlap = 0.0
                if objs_i and objs_j:
                    req_overlap = float(len(objs_i & objs_j)) / float(max(1, min(len(objs_i), len(objs_j))))

                if sim_ij >= float(similarity_threshold):
                    _union(i, j)
                    merge_rule_counts["strict_similarity"] += 1
                    continue

                if not topic_related:
                    continue

                if req_overlap >= 0.45 and sim_ij >= 0.50:
                    _union(i, j)
                    merge_rule_counts["topic_required_overlap"] += 1
                    continue

                has_umbrella = bool(umbrella_flags[i] == 1 or umbrella_flags[j] == 1)
                if has_umbrella and sim_ij >= float(relaxed_topic_similarity) and lex_overlap >= 0.03:
                    _union(i, j)
                    merge_rule_counts["topic_umbrella_similarity"] += 1
                    continue

                if has_umbrella and sim_ij >= 0.44 and len(common_terms) >= 1:
                    _union(i, j)
                    merge_rule_counts["topic_umbrella_anchor"] += 1
                    continue

                if sim_ij >= 0.74 and lex_overlap >= 0.10:
                    _union(i, j)
                    merge_rule_counts["topic_semantic_overlap"] += 1
                    continue

                # Last resort: broad reusable diagram anchor across weak/no topic tags.
                if has_umbrella and sim_ij >= 0.40 and len(common_terms) >= 2:
                    _union(i, j)
                    merge_rule_counts["cross_topic_anchor"] += 1

        comp: Dict[int, List[int]] = {}
        for i in range(n):
            r = _find(i)
            comp.setdefault(r, []).append(i)

        groups: List[List[int]] = [idxs for idxs in comp.values() if len(idxs) >= 2]
        if not groups:
            return {
                "ok": True,
                "diagram_prompts": len(uniq_diag_prompts),
                "groups_found": 0,
                "groups_rewritten": 0,
                "replacement_map": {},
                "similarity_threshold": float(similarity_threshold),
                "relaxed_topic_similarity": float(relaxed_topic_similarity),
                "merge_rule_counts": merge_rule_counts,
                "note": "no_very_close_diagram_groups",
            }

        replacement_map: Dict[str, str] = {}
        group_debug: List[Dict[str, Any]] = []

        for idxs in groups:
            centroid = np.mean(mat[idxs], axis=0).astype(np.float32)
            cn = float(np.linalg.norm(centroid))
            if cn > 1e-9:
                centroid = centroid / cn

            centroid_scores: Dict[int, float] = {}
            avg_pair_scores: Dict[int, float] = {}

            for i in idxs:
                centroid_scores[i] = float(np.dot(mat[i], centroid))
                avg_pair_scores[i] = float(np.mean([sims[i, j] for j in idxs if j != i])) if len(idxs) > 1 else 1.0

            # Candidate pool around centroid winner; then bias to reusable umbrella prompt.
            best_centroid = max(centroid_scores.values()) if centroid_scores else 0.0
            pool = [
                i for i in idxs
                if (best_centroid - float(centroid_scores.get(i, 0.0))) <= float(centroid_margin_for_umbrella)
            ]
            if not pool:
                pool = list(idxs)

            best_i = max(
                pool,
                key=lambda i: (
                    int(umbrella_flags[i] == 1),
                    coverage_scores[i],
                    avg_pair_scores.get(i, 0.0),
                    centroid_scores.get(i, 0.0),
                    -i,  # deterministic tie-break toward earlier prompt
                ),
            )
            rep = uniq_diag_prompts[best_i]

            members = [uniq_diag_prompts[i] for i in idxs]
            for m in members:
                replacement_map[m] = rep

            group_debug.append(
                {
                    "representative": rep,
                    "members": members,
                    "member_count": len(members),
                    "centroid_scores": {
                        uniq_diag_prompts[i]: round(float(centroid_scores.get(i, 0.0)), 6)
                        for i in idxs
                    },
                    "avg_pairwise_scores": {
                        uniq_diag_prompts[i]: round(float(avg_pair_scores.get(i, 0.0)), 6)
                        for i in idxs
                    },
                    "coverage_scores": {
                        uniq_diag_prompts[i]: round(float(coverage_scores[i]), 6)
                        for i in idxs
                    },
                    "umbrella_flags": {
                        uniq_diag_prompts[i]: int(umbrella_flags[i] == 1)
                        for i in idxs
                    },
                    "centroid_margin_for_umbrella": float(centroid_margin_for_umbrella),
                }
            )

        changed_map = {k: v for k, v in replacement_map.items() if k != v}
        if not changed_map:
            return {
                "ok": True,
                "diagram_prompts": len(uniq_diag_prompts),
                "groups_found": len(groups),
                "groups_rewritten": 0,
                "replacement_map": {},
                "similarity_threshold": float(similarity_threshold),
                "relaxed_topic_similarity": float(relaxed_topic_similarity),
                "merge_rule_counts": merge_rule_counts,
                "groups": group_debug,
                "note": "groups_found_but_no_replacements",
            }

        # 1) Collapse all prompts meta by representative prompt.
        collapsed_prompts_meta: Dict[str, Dict[str, Any]] = {}
        for p, meta in (all_prompts_meta or {}).items():
            old_p = str(p or "").strip()
            if not old_p:
                continue
            new_p = changed_map.get(old_p, old_p)
            prev = collapsed_prompts_meta.get(new_p) or {}
            prev_topic = str(prev.get("topic", "") or "").strip()
            cur_topic = str((meta or {}).get("topic", "") or "").strip()
            prev_d = 1 if int(prev.get("diagram", 0) or 0) == 1 else 0
            cur_d = 1 if int((meta or {}).get("diagram", 0) or 0) == 1 else 0
            collapsed_prompts_meta[new_p] = {
                "topic": prev_topic or cur_topic,
                "diagram": 1 if (prev_d == 1 or cur_d == 1) else 0,
            }
        all_prompts_meta.clear()
        all_prompts_meta.update(collapsed_prompts_meta)

        # 2) Merge required diagram objects into representative keys.
        collapsed_required_objects: Dict[str, List[str]] = {}
        for p, objs in (all_diagram_required_objects or {}).items():
            old_p = str(p or "").strip()
            if not old_p:
                continue
            new_p = changed_map.get(old_p, old_p)
            if not isinstance(objs, list):
                continue
            _merge_unique_str_list(
                collapsed_required_objects.setdefault(new_p, []),
                [str(x).strip() for x in objs if str(x).strip()],
                cap=80,
            )
        all_diagram_required_objects.clear()
        all_diagram_required_objects.update(collapsed_required_objects)

        # 3) Rewrite chapter image map/query content for diagram entries.
        image_map_rewrites = 0
        image_plan_rewrites = 0
        for ch in chapters_out or []:
            if not isinstance(ch, dict):
                continue

            maps = ch.get("image_text_maps") or []
            if isinstance(maps, list):
                for im in maps:
                    if not isinstance(im, dict):
                        continue
                    if int(im.get("text_tag", 0) or 0) == 1:
                        continue
                    if int(im.get("diagram", 0) or 0) != 1:
                        continue

                    cur_q = str(im.get("query", "") or "").strip()
                    cur_c = str(im.get("content", "") or "").strip()
                    rep = changed_map.get(cur_q) or changed_map.get(cur_c)
                    if not rep:
                        continue
                    if cur_q != rep:
                        im["query"] = rep
                    if cur_c != rep:
                        im["content"] = rep
                    image_map_rewrites += 1

            plan = ch.get("image_plan") or {}
            imgs = plan.get("images") if isinstance(plan, dict) else None
            if isinstance(imgs, list):
                for row in imgs:
                    if not isinstance(row, dict):
                        continue
                    if int(row.get("text", 0) or 0) == 1:
                        continue
                    if int(row.get("diagram", 0) or 0) != 1:
                        continue
                    cur_c = str(row.get("content", "") or "").strip()
                    rep = changed_map.get(cur_c)
                    if not rep:
                        continue
                    if cur_c != rep:
                        row["content"] = rep
                        image_plan_rewrites += 1

        return {
            "ok": True,
            "diagram_prompts": len(uniq_diag_prompts),
            "embedded_diagram_prompts": int(mat.shape[0]),
            "groups_found": len(groups),
            "groups_rewritten": len(group_debug),
            "similarity_threshold": float(similarity_threshold),
            "relaxed_topic_similarity": float(relaxed_topic_similarity),
            "merge_rule_counts": merge_rule_counts,
            "replacement_map": changed_map,
            "image_text_map_rewrites": int(image_map_rewrites),
            "image_plan_rewrites": int(image_plan_rewrites),
            "groups": group_debug,
        }

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

    def _load_objects_list_from_refined_file(self, refined_file: str) -> List[Any]:
        """
        refined labels file is expected to be:
          - a list[str] or list[dict], OR
          - a dict with "objects"/"refined_labels"/"labels"
        """
        try:
            data = json.loads(Path(refined_file).read_text(encoding="utf-8"))
        except Exception:
            return []

        raw = None
        if isinstance(data, list):
            raw = data
        elif isinstance(data, dict):
            raw = data.get("objects")
            if not isinstance(raw, list):
                raw = data.get("refined_labels")
            if not isinstance(raw, list):
                raw = data.get("labels")

        if not isinstance(raw, list):
            return []

        out: List[Any] = []
        seen = set()
        for it in raw:
            if isinstance(it, str):
                s = it.strip()
                if not s:
                    continue
                k = s.lower()
                if k in seen:
                    continue
                seen.add(k)
                out.append(s)
                continue

            if isinstance(it, dict):
                nm = it.get("name")
                if isinstance(nm, str):
                    s = nm.strip()
                    if not s:
                        continue
                    k = s.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(s)
                    continue
                out.append(it)

        return out

    


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
        import ComfyFluxClient

        from shared_models import (
            get_siglip,
            get_siglip_text,
            get_minilm,
            init_siglip_hot,
            init_siglip_text_hot,
            init_minilm_hot,
            unload_siglip,
            unload_siglip_text,
            init_qwen_hot,
            get_qwen,
            qwen_lock,
        )

        # Ensure Pinecone fetch is executed with explicit hot workers from shared_models.
        if get_minilm() is None:
            with self._timed("models.load_minilm_for_pinecone", gpu_index=self.gpu_index, cpu_threads=self.cpu_threads):
                init_minilm_hot(gpu_index=self.gpu_index, cpu_threads=self.cpu_threads, warmup=True)
        if get_siglip_text() is None:
            with self._timed("models.load_siglip_text_for_pinecone", gpu_index=self.gpu_index, cpu_threads=self.cpu_threads):
                init_siglip_text_hot(
                    gpu_index=self.gpu_index,
                    cpu_threads=self.cpu_threads,
                    warmup=True,
                    prefer_fp8=True,
                )

        PineconeFetch.configure_hot_models(
            siglip_bundle=(get_siglip_text() or get_siglip()),
            minilm_bundle=get_minilm(),
            clear_siglip=True,
            clear_minilm=True,
        )

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
        self._dbg(
            "Image prompt meta prepared",
            data={
                "prompts_total": len(prompts),
                "diagram_prompts": len(diagram_set),
                "non_diagram_prompts": max(0, len(prompts) - len(diagram_set)),
            },
        )

        hits: Dict[str, List[str]] = {}
        misses: Dict[str, str] = {}
        pinecone_attempts_by_prompt: Dict[str, List[Dict[str, Any]]] = {}

        # First try strict precision, then progressively relax to avoid false misses.
        pinecone_fetch_attempts: List[Tuple[float, int]] = [
            (0.78, 3),
            (0.70, 3),
            (0.62, 2),
        ]

        for p in prompts:
            t0 = time.perf_counter()
            attempt_log: List[Dict[str, Any]] = []
            ids: List[str] = []
            try:
                for score_floor, min_mods in pinecone_fetch_attempts:
                    cur_ids = PineconeFetch.fetch_processed_ids_for_prompt(
                        p,
                        top_n=2,
                        top_k_per_modality=50,
                        min_modalities=int(min_mods),
                        min_final_score=float(score_floor),
                        require_base_context_match=True,
                    )
                    attempt_log.append(
                        {
                            "min_final_score": float(score_floor),
                            "min_modalities": int(min_mods),
                            "hit_count": int(len(cur_ids or [])),
                        }
                    )
                    if cur_ids:
                        ids = [str(x) for x in cur_ids if str(x)]
                        break
            except Exception as e:
                attempt_log.append({"error": repr(e)})

            if ids:
                hits[p] = ids
            else:
                misses[p] = str(prompt_meta.get(p, {}).get("topic") or lesson_topic or "").strip()

            pinecone_attempts_by_prompt[p] = attempt_log

            self._dbg(
                "Pinecone fetch done",
                data={
                    "prompt": p,
                    "hit": bool(p in hits),
                    "attempts": attempt_log,
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                },
            )

        # Release SigLIP-text (pinecone only) immediately after Pinecone fetch.
        siglip_text_unload_err: Optional[str] = None
        try:
            with self._timed("models.unload_siglip_text_after_pinecone", gpu_index=self.gpu_index):
                unload_siglip_text()
        except Exception as e:
            siglip_text_unload_err = repr(e)

        # Ensure full SigLIP is also not resident before any Comfy/Flux generation.
        siglip_unload_err: Optional[str] = None
        try:
            with self._timed("models.unload_siglip_before_comfy", gpu_index=self.gpu_index):
                unload_siglip()
        except Exception as e:
            siglip_unload_err = repr(e)

        debug: Dict[str, Any] = {
            "pinecone_hits": list(hits.keys()),
            "misses": list(misses.keys()),
            "pinecone_attempts_by_prompt": pinecone_attempts_by_prompt,
            "pipeline_started": False,
            "selection_done": False,
            "colours_done": False,
            "pipeline_errors": {},
            "refined_labels_dir": None,
            "diagram_misses": [],
            "comfy_misses": [],
            "comfy_generated": {},
        }
        if siglip_text_unload_err:
            debug["pipeline_errors"]["siglip_text_unload_after_pinecone"] = siglip_text_unload_err
        if siglip_unload_err:
            debug["pipeline_errors"]["siglip_unload_before_comfy"] = siglip_unload_err

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

        diagram_misses = {p: misses[p] for p in misses.keys() if p in diagram_set}
        non_diagram_misses = {p: misses[p] for p in misses.keys() if p not in diagram_set}

        debug["diagram_misses"] = list(diagram_misses.keys())
        debug["comfy_misses"] = list(non_diagram_misses.keys())
        self._dbg(
            "Image misses split",
            data={
                "diagram_misses": len(diagram_misses),
                "comfy_misses": len(non_diagram_misses),
                "diagram_miss_prompts_preview": list(diagram_misses.keys())[:5],
            },
        )

        qwen_err: Optional[str] = None
        research_err: Optional[str] = None
        rerank_err: Optional[str] = None
        comfy_err: Optional[str] = None
        comfy_free_err: Optional[str] = None
        siglip_reload_err: Optional[str] = None
        comfy_generated: Dict[str, Any] = {}
        rerank_out: Dict[str, List[str]] = {}
        research_done_evt = threading.Event()
        comfy_done_evt = threading.Event()
        if not diagram_misses:
            research_done_evt.set()
        if not non_diagram_misses:
            comfy_done_evt.set()

        comfy_capture_root = (
            str(os.getenv("COMFY_CAPTURE_OUTPUT_ROOT") or "").strip()
            or os.path.join(
                str(getattr(ImageResearcher, "IMAGES_PATH", "ResearchImages") or "ResearchImages"),
                "UniqueImages",
                "ComfyGenerated",
            )
        )
        debug["comfy_capture_root"] = comfy_capture_root

        if diagram_misses:
            try:
                reset_fn = getattr(ImageResearcher, "_reset_research_images_dir", None)
                if callable(reset_fn):
                    with self._timed("research.reset_images_dir_before_parallel"):
                        reset_fn()
            except Exception as e:
                debug["pipeline_errors"]["research_reset_before_parallel"] = repr(e)

        def _research_diagrams():
            nonlocal research_err
            if not diagram_misses:
                return
            try:
                mech_many = getattr(ImageResearcher, "research_many_mechanical", None)
                if callable(mech_many):
                    mech_many(
                        diagram_misses,
                        # User-requested behavior: fan out to all diagram prompts at once.
                        max_workers=max(1, len(diagram_misses)),
                        reset_images_dir=False,
                    )
                    return

                # Keep only the mechanical/crawl phase here.
                # Final SigLIP rerank is executed after BOTH research + comfy complete.
                items: List[Tuple[str, str]] = []
                for prompt, topic in (diagram_misses or {}).items():
                    p = str(prompt or "").strip()
                    t = str(topic or "").strip()
                    if p and t:
                        items.append((p, t))
                if not items:
                    return

                # User-requested behavior: fan out to all diagram prompts at once.
                max_workers = max(1, len(items))
                with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="research_mech") as ex:
                    futures = [
                        ex.submit(
                            ImageResearcher.research,
                            prompt,
                            topic,
                            None,   # prompt_id auto-derived by researcher
                            False,  # rank_images=False => mechanical phase only
                            None,   # siglip_backend
                        )
                        for prompt, topic in items
                    ]
                    for fut in as_completed(futures):
                        fut.result()
            except Exception as e:
                research_err = repr(e)
            finally:
                research_done_evt.set()

        def _generate_non_diagrams():
            nonlocal comfy_err, comfy_generated
            if not non_diagram_misses:
                return
            try:
                comfy_generated = ComfyFluxClient.generate_many(
                    non_diagram_misses,
                    workflow_path=os.getenv("COMFY_WORKFLOW_JSON") or None,
                    comfy_url=os.getenv("COMFY_URL") or None,
                    output_root=comfy_capture_root,
                    batch_size=1,
                    clean_prompt_dirs=True,
                )
            except Exception as e:
                comfy_err = repr(e)
            finally:
                comfy_done_evt.set()

        def _prepare_models_after_flux():
            nonlocal siglip_reload_err, qwen_err, comfy_free_err
            research_done_evt.wait()
            comfy_done_evt.wait()

            try:
                with self._timed("comfy.free_after_flux_before_siglip", gpu_index=self.gpu_index):
                    ComfyFluxClient.free_all_models(
                        comfy_url=os.getenv("COMFY_URL") or None,
                        unload_models=True,
                        free_memory=True,
                    )
            except Exception as e:
                comfy_free_err = repr(e)

            try:
                with self._timed("models.reload_siglip_after_comfy", gpu_index=self.gpu_index, cpu_threads=self.cpu_threads):
                    init_siglip_hot(
                        gpu_index=self.gpu_index,
                        cpu_threads=self.cpu_threads,
                        warmup=True,
                    )
            except Exception as e:
                siglip_reload_err = repr(e)

            # Intentionally do not preload Qwen here.
            # Keeping Qwen lazy avoids post-Flux VRAM spikes and lets pipeline continue
            # even when Qwen cannot be loaded at this stage.

        threads: List[threading.Thread] = []
        post_flux_models_thread = threading.Thread(target=_prepare_models_after_flux, name="prepare_models_after_flux", daemon=False)

        if diagram_misses:
            threads.append(threading.Thread(target=_research_diagrams, name="research_diagrams", daemon=False))
        if non_diagram_misses:
            threads.append(threading.Thread(target=_generate_non_diagrams, name="generate_non_diagrams_comfy", daemon=False))

        threads_t0 = time.perf_counter()
        for t in threads:
            t.start()
        post_flux_models_thread.start()
        for t in threads:
            t.join()
        with self._timed("models.wait_post_flux_model_prep"):
            post_flux_models_thread.join()

        debug["comfy_generated"] = {
            k: list((v or {}).get("saved_paths") or [])
            for k, v in (comfy_generated or {}).items()
            if isinstance(v, dict)
        }

        self._dbg(
            "Diagram research + comfy generation + post-flux SigLIP prep done",
            data={
                "elapsed_ms": round((time.perf_counter() - threads_t0) * 1000.0, 2),
                "diagram_misses": len(diagram_misses),
                "comfy_misses": len(non_diagram_misses),
            },
        )

        if research_err:
            debug["pipeline_errors"]["research_mechanical"] = research_err
        if comfy_err:
            debug["pipeline_errors"]["comfy_many"] = comfy_err
        if comfy_free_err:
            debug["pipeline_errors"]["comfy_free_after_flux"] = comfy_free_err
        if siglip_reload_err:
            debug["pipeline_errors"]["siglip_reload_after_comfy"] = siglip_reload_err

        # Strict ordering: only after BOTH mechanical research + siglip-reload are done,
        # execute final researcher rerank.
        if not research_err and diagram_misses:
            try:
                if siglip_reload_err:
                    raise RuntimeError(f"siglip_reload_failed: {siglip_reload_err}")
                if get_siglip() is None:
                    raise RuntimeError("siglip_bundle_none_after_reload")

                finalize_fn = getattr(ImageResearcher, "finalize_research_from_saved_ranker_state", None)
                if not callable(finalize_fn):
                    finalize_fn = getattr(ImageResearcher, "collect_unique_images_all_prompts", None)
                if callable(finalize_fn):
                    shared_backend_cls = getattr(ImageResearcher, "_SharedSiglipBackend", None)
                    if not callable(shared_backend_cls):
                        raise RuntimeError("_SharedSiglipBackend_missing")

                    backend = shared_backend_cls()

                    finalize_kwargs: Dict[str, Any] = {}
                    img_root = getattr(ImageResearcher, "IMAGES_PATH", None)
                    if isinstance(img_root, str) and img_root.strip():
                        finalize_kwargs["images_root"] = img_root
                    finalize_kwargs["backend"] = backend

                    with self._timed("research.finalize_siglip_rerank"):
                        out_map = finalize_fn(**finalize_kwargs)
                    if isinstance(out_map, dict):
                        rerank_out = {
                            str(k): [str(x) for x in (v or []) if str(x)]
                            for k, v in out_map.items()
                            if isinstance(k, str)
                        }
            except Exception as e:
                rerank_err = repr(e)

        if rerank_err:
            debug["pipeline_errors"]["research_finalize_rerank"] = rerank_err
        if rerank_out:
            debug["diagram_rerank_counts"] = {k: len(v) for k, v in rerank_out.items()}

        # Enforce strict memory sequencing:
        # 1) finish SigLIP-based rerank
        # 2) unload heavy SigLIP
        # 3) load Qwen
        # 4) start ImagePipeline
        siglip_unload_before_pipeline_err: Optional[str] = None
        try:
            with self._timed("models.unload_siglip_before_pipeline_qwen", gpu_index=self.gpu_index):
                unload_siglip()
        except Exception as e:
            siglip_unload_before_pipeline_err = repr(e)
        if siglip_unload_before_pipeline_err:
            debug["pipeline_errors"]["siglip_unload_before_pipeline_qwen"] = siglip_unload_before_pipeline_err

        try:
            with self._timed("models.load_qwen_for_pipeline_after_rerank", gpu_index=self.gpu_index, cpu_threads=self.cpu_threads):
                init_qwen_hot(
                    gpu_index=self.gpu_index,
                    cpu_threads=self.cpu_threads,
                    warmup=True,
                )
        except Exception as e:
            qwen_err = repr(e)

        qb = get_qwen()
        if qb is None:
            debug["pipeline_warnings"] = debug.get("pipeline_warnings") or {}
            debug["pipeline_warnings"]["qwen_unavailable_for_pipeline_start"] = qwen_err or "qwen_bundle_none"

        workers = ImagePipeline.PipelineWorkers(
            qwen_model=(qb.model if qb is not None else None),
            qwen_processor=(qb.processor if qb is not None else None),
            qwen_device=(qb.device if qb is not None else None),
            qwen_lock=qwen_lock(),
            siglip_bundle=get_siglip(),
            minilm_bundle=get_minilm(),
        )

        # NEW: pass required diagram objects map into ImagePipeline
        handle = ImagePipeline.start_pipeline_background(
            workers=workers,
            model_id=(qb.model_id if qb is not None else "Qwen/Qwen3-VL-2B-Instruct"),
            gpu_index=self.gpu_index,
            allowed_base_contexts=list(misses.keys()),
            diagram_base_contexts=[p for p in misses.keys() if p in diagram_set],
            diagram_required_objects_by_base_context=(diagram_required_objects_by_base_context or {}),
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

    def _build_chunk_sync_maps(
        self,
        *,
        chapters_out: List[Dict[str, Any]],
        assets_by_name: Dict[str, ImageAsset],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for ch in (chapters_out or []):
            ci = int(ch.get("chapter_index", 0) or 0)
            speech = str(ch.get("speech", "") or "")
            maps = ch.get("image_text_maps") or []
            if not isinstance(maps, list):
                maps = []
            speech_step_order = ch.get("speech_step_order") or []
            speech_step_ranges = ch.get("speech_step_ranges") or {}
            timeline_steps = ch.get("timeline_steps") or []
            one = build_chunk_sync_map_for_chapter(
                chapter_index=ci,
                speech_text=speech,
                image_text_maps=maps,
                speech_step_order=speech_step_order if isinstance(speech_step_order, list) else [],
                speech_step_ranges=speech_step_ranges if isinstance(speech_step_ranges, dict) else {},
                timeline_steps=timeline_steps if isinstance(timeline_steps, list) else [],
                assets_by_name=assets_by_name,
                board_w=self.whiteboard_width,
                board_h=self.whiteboard_height,
                include_inter_chunk_silence=True,
            )
            out.append(one)

        out.sort(key=lambda x: int(x.get("chapter_index", 0) or 0))
        return out

    def _run_qwen_actions_static_batched(
        self,
        *,
        chapters_out: List[Dict[str, Any]],
        assets_by_name: Dict[str, ImageAsset],
    ) -> Dict[str, Any]:
        from shared_models import get_qwen, qwen_lock
        import qwentest

        qb = get_qwen()
        if qb is None:
            return {"enabled": False, "error": "qwen_not_loaded"}

        def _cuda_mem_snapshot() -> Dict[str, Any]:
            try:
                import torch
                if not torch.cuda.is_available():
                    return {"cuda": False}
                alloc = float(torch.cuda.memory_allocated() / (1024 ** 2))
                reserved = float(torch.cuda.memory_reserved() / (1024 ** 2))
                max_alloc = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                return {
                    "cuda": True,
                    "alloc_mb": round(alloc, 2),
                    "reserved_mb": round(reserved, 2),
                    "max_alloc_mb": round(max_alloc, 2),
                }
            except Exception as e:
                return {"cuda": "unknown", "err": f"{type(e).__name__}: {e}"}

        chunk_sync_maps = self._build_chunk_sync_maps(
            chapters_out=chapters_out,
            assets_by_name=assets_by_name,
        )
        self._dbg("Chunk sync maps built", data={"chunks": len(chunk_sync_maps)})
        self._dbg(
            "Qwen static planner start",
            data={
                "chunks": len(chunk_sync_maps),
                "space_batch_size": int(SPACE_PLANNER_BATCH_SIZE),
                "action_batch_size": int(ACTION_PLANNER_BATCH_SIZE),
                "space_max_new_tokens": int(SPACE_PLANNER_MAX_NEW_TOKENS),
                "visual_max_new_tokens": int(VISUAL_PLANNER_MAX_NEW_TOKENS),
                "silence_max_new_tokens": int(SILENCE_PLANNER_MAX_NEW_TOKENS),
                "cuda_mem": _cuda_mem_snapshot(),
            },
        )

        # 1) Space planner by chunk in batches (size=8)
        planned_chunks: List[Dict[str, Any]] = []
        space_planner_batches = _chunk_list(chunk_sync_maps, SPACE_PLANNER_BATCH_SIZE)
        for bi, batch in enumerate(space_planner_batches):
            approx_prompt_chars = 0
            try:
                approx_prompt_chars = sum(
                    len(json.dumps({"steps": b.get("steps") or [], "entries": b.get("entries") or []}, ensure_ascii=False))
                    for b in batch
                    if isinstance(b, dict)
                )
            except Exception:
                approx_prompt_chars = 0
            self._dbg(
                "Space planner batch start",
                data={
                    "batch_index": bi,
                    "chunks_in_batch": len(batch),
                    "approx_prompt_chars": int(approx_prompt_chars),
                    "cuda_mem": _cuda_mem_snapshot(),
                },
            )
            qwen_t0 = time.perf_counter()
            with qwen_lock():
                batch_out = qwentest.plan_space_timeline_batch_transformers(
                    model=qb.model,
                    processor=qb.processor,
                    device=qb.device,
                    chunk_maps=batch,
                    board_width=self.whiteboard_width,
                    board_height=self.whiteboard_height,
                    batch_size=SPACE_PLANNER_BATCH_SIZE,
                    temperature=0.15,
                    max_new_tokens=SPACE_PLANNER_MAX_NEW_TOKENS,
                )
            got = batch_out.get("chunks") if isinstance(batch_out, dict) else None
            if not isinstance(got, list):
                got = []
            by_ci: Dict[int, Dict[str, Any]] = {}
            for row in got:
                if not isinstance(row, dict):
                    continue
                ci = int(row.get("chapter_index", 0) or 0)
                by_ci[ci] = row
            for orig in batch:
                ci = int(orig.get("chapter_index", 0) or 0)
                planned_chunks.append(by_ci.get(ci) or orig)

            self._dbg(
                "Space planner batch done",
                data={
                    "batch_index": bi,
                    "chunks_in_batch": len(batch),
                    "elapsed_ms": round((time.perf_counter() - qwen_t0) * 1000.0, 2),
                    "cuda_mem": _cuda_mem_snapshot(),
                },
            )

        planned_chunks.sort(key=lambda x: int(x.get("chapter_index", 0) or 0))

        # 2) Build visual and silence jobs from static timeline
        visual_events_out: List[Dict[str, Any]] = []
        silence_jobs: List[Dict[str, Any]] = []
        active_objects_by_key: Dict[str, Dict[str, Any]] = {}
        active_object_counter = 0

        for chunk_i, ch in enumerate(planned_chunks):
            chapter_index = int(ch.get("chapter_index", 0) or 0)
            entries = ch.get("entries") or []
            if not isinstance(entries, list):
                entries = []
            entries.sort(key=lambda e: (int(e.get("range_start", 0) or 0), 0 if str(e.get("type", "") or "") == "silence" else 1))

            split_groups = split_entries_by_deletion_silence(entries)

            for seg_i, seg in enumerate(split_groups):
                visuals = seg.get("visual_entries") or []
                if not isinstance(visuals, list):
                    visuals = []
                visuals = [x for x in visuals if str(x.get("type", "") or "") in ("image", "text")]

                visual_batches = _chunk_list(visuals, ACTION_PLANNER_BATCH_SIZE)
                for batch_i, batch_rows in enumerate(visual_batches):
                    reqs: List[Dict[str, Any]] = []
                    for row_i, row in enumerate(batch_rows):
                        context_rows = []
                        for prev in batch_rows[:row_i]:
                            context_rows.append(
                                {
                                    "name": str(prev.get("name", "") or ""),
                                    "type": str(prev.get("type", "") or ""),
                                    "print_bbox": prev.get("print_bbox"),
                                    "range_start": int(prev.get("range_start", 0) or 0),
                                    "range_end": int(prev.get("range_end", 0) or 0),
                                    "diagram": int(prev.get("diagram", 0) or 0),
                                    "text_tag": int(prev.get("text_tag", 0) or 0),
                                    "speech_text_in_range": str(prev.get("speech_text_in_range", "") or ""),
                                }
                            )
                        reqs.append(
                            {
                                "chapter_index": chapter_index,
                                "chunk_index": chunk_i,
                                "segment_index": seg_i,
                                "batch_index": batch_i,
                                "name": str(row.get("name", "") or ""),
                                "type": str(row.get("type", "") or ""),
                                "content": str(row.get("content", "") or ""),
                                "write_text": str(row.get("write_text", "") or ""),
                                "diagram": int(row.get("diagram", 0) or 0),
                                "text_tag": int(row.get("text_tag", 0) or 0),
                                "range_start": int(row.get("range_start", 0) or 0),
                                "range_end": int(row.get("range_end", 0) or 0),
                                "speech_text_in_range": str(row.get("speech_text_in_range", "") or ""),
                                "step_key": str(row.get("step_key", "") or ""),
                                "bbox_px": row.get("bbox_px") or {"w": 400, "h": 300},
                                "print_bbox": row.get("print_bbox") or {"x": 0, "y": 0, "w": 400, "h": 300},
                                "processed_id": row.get("processed_id"),
                                "objects_that_comprise_image": row.get("objects_that_comprise_image") or [],
                                "static_plan_context": context_rows,
                            }
                        )

                    if not reqs:
                        continue

                    self._dbg(
                        "Visual action batch start",
                        data={
                            "chapter_index": chapter_index,
                            "segment_index": seg_i,
                            "batch_index": batch_i,
                            "items": len(reqs),
                            "context_items_total": int(sum(len(r.get("static_plan_context") or []) for r in reqs)),
                            "cuda_mem": _cuda_mem_snapshot(),
                        },
                    )
                    qwen_t0 = time.perf_counter()
                    with qwen_lock():
                        resp = qwentest.plan_visual_actions_batch_transformers(
                            model=qb.model,
                            processor=qb.processor,
                            device=qb.device,
                            events=reqs,
                            batch_size=ACTION_PLANNER_BATCH_SIZE,
                            temperature=0.2,
                            max_new_tokens=VISUAL_PLANNER_MAX_NEW_TOKENS,
                        )
                    rows = resp.get("items") if isinstance(resp, dict) else None
                    if not isinstance(rows, list):
                        rows = []

                    def _upsert_active(rec: Dict[str, Any]) -> None:
                        nonlocal active_object_counter
                        nm = str((rec or {}).get("name", "") or "").strip()
                        if not nm:
                            return
                        key = nm.lower()
                        prev = active_objects_by_key.get(key)
                        if isinstance(prev, dict):
                            rec["created_order"] = int(prev.get("created_order", 0) or 0)
                        else:
                            active_object_counter += 1
                            rec["created_order"] = int(active_object_counter)
                        active_objects_by_key[key] = rec

                    for i, req in enumerate(reqs):
                        row = rows[i] if (i < len(rows) and isinstance(rows[i], dict)) else {"actions": [], "notes": "missing_visual_result"}
                        acts = row.get("actions") if isinstance(row, dict) else []
                        if not isinstance(acts, list):
                            acts = []

                        normalized_actions: List[Dict[str, Any]] = []
                        for ai, a in enumerate(acts):
                            nrm = normalize_static_visual_action(a, req=req, action_index=ai) if isinstance(a, dict) else None
                            if isinstance(nrm, dict):
                                normalized_actions.append(nrm)

                        if not normalized_actions:
                            fallback_acts = qwentest._default_visual_actions_for_event(req)
                            if isinstance(fallback_acts, list):
                                for ai, a in enumerate(fallback_acts):
                                    nrm = normalize_static_visual_action(a, req=req, action_index=ai) if isinstance(a, dict) else None
                                    if isinstance(nrm, dict):
                                        normalized_actions.append(nrm)

                        parsed_actions: List[Dict[str, Any]] = []
                        for ai, one in enumerate(normalized_actions):
                            t = str(one.get("type", "") or "").strip().lower()
                            one = convert_local_sync_to_absolute(
                                one,
                                event_start_word=int(req.get("range_start", 0) or 0),
                                event_end_word=int(req.get("range_end", 0) or 0),
                            )
                            parsed_actions.append(one)

                            if t in ("delete_self", "delete_by_name"):
                                tgt = str(one.get("target", "") or req.get("name", "") or "").strip()
                                if tgt:
                                    active_objects_by_key.pop(tgt.lower(), None)
                                continue

                            if t in ("draw_image", "write_text"):
                                rec = build_active_object_from_action(
                                    one,
                                    req=req,
                                    action_index=ai,
                                    chapter_index=int(req.get("chapter_index", 0) or 0),
                                    chunk_index=int(req.get("chunk_index", 0) or 0),
                                    segment_index=int(req.get("segment_index", 0) or 0),
                                    batch_index=int(req.get("batch_index", 0) or 0),
                                )
                                if isinstance(rec, dict):
                                    _upsert_active(rec)

                        duration_words = max(1, int(req.get("range_end", 0) or 0) - int(req.get("range_start", 0) or 0) + 1)
                        duration_sec = float(max(1.0, min(12.0, duration_words / 2.5)))

                        visual_events_out.append(
                            {
                                "event": {
                                    "kind": "image",
                                    "chapter_index": int(req["chapter_index"]),
                                    "chunk_index": int(req["chunk_index"]),
                                    "segment_index": int(req["segment_index"]),
                                    "batch_index": int(req["batch_index"]),
                                    "name": req.get("name"),
                                    "image_name": req.get("name"),
                                    "type": req.get("type"),
                                    "diagram": int(req.get("diagram", 0) or 0),
                                    "text_tag": int(req.get("text_tag", 0) or 0),
                                    "start_word_index": int(req.get("range_start", 0) or 0),
                                    "end_word_index": int(req.get("range_end", 0) or 0),
                                    "speech_text_in_range": str(req.get("speech_text_in_range", "") or ""),
                                    "duration_sec": duration_sec,
                                    "step_key": req.get("step_key"),
                                    "bbox_px": req.get("bbox_px"),
                                    "print_bbox": req.get("print_bbox"),
                                    "objects_that_comprise_image": req.get("objects_that_comprise_image") or [],
                                },
                                "planner": "visual",
                                "qwen_payload": row,
                                "base_actions_parsed": parsed_actions,
                            }
                        )

                    self._dbg(
                        "Visual action batch done",
                        data={
                            "chapter_index": chapter_index,
                            "segment_index": seg_i,
                            "batch_index": batch_i,
                            "items": len(reqs),
                            "active_objects_after_batch": len(active_objects_by_key),
                            "elapsed_ms": round((time.perf_counter() - qwen_t0) * 1000.0, 2),
                            "cuda_mem": _cuda_mem_snapshot(),
                        },
                    )

                del_sil = seg.get("deletion_silence")
                if isinstance(del_sil, dict):
                    active_objects_rows = sorted_active_objects_for_cleanup(active_objects_by_key)
                    active_names = [str(r.get("name", "") or "") for r in active_objects_rows if str(r.get("name", "") or "")]

                    silence_jobs.append(
                        {
                            "chapter_index": chapter_index,
                            "chunk_index": chunk_i,
                            "segment_index": seg_i,
                            "name": str(del_sil.get("name", "") or f"silence_{chapter_index}_{seg_i}"),
                            "start_word_index": int(del_sil.get("range_start", 0) or 0),
                            "end_word_index": int(del_sil.get("range_end", del_sil.get("range_start", 0)) or 0),
                            "duration_sec": float(del_sil.get("duration_sec", INTER_CHUNK_SILENCE_SEC) or INTER_CHUNK_SILENCE_SEC),
                            "step_key": str(del_sil.get("step_key", "") or ""),
                            "active_names": active_names,
                            "active_objects": active_objects_rows,
                            "active_objects_count": len(active_names),
                        }
                    )

        self._dbg(
            "Silence planner jobs built",
            data={
                "jobs": len(silence_jobs),
                "batches": len(_chunk_list(silence_jobs, ACTION_PLANNER_BATCH_SIZE)),
                "active_objects_current": len(active_objects_by_key),
                "cuda_mem": _cuda_mem_snapshot(),
            },
        )

        # 3) Silence planner in batches (delete-tagged silences only)
        silence_events_out: List[Dict[str, Any]] = []
        silence_batches = _chunk_list(silence_jobs, ACTION_PLANNER_BATCH_SIZE)
        for batch_i, s_batch in enumerate(silence_batches):
            if not s_batch:
                continue
            self._dbg(
                "Silence action batch start",
                data={
                    "batch_index": batch_i,
                    "items": len(s_batch),
                    "cuda_mem": _cuda_mem_snapshot(),
                },
            )
            qwen_t0 = time.perf_counter()
            with qwen_lock():
                resp = qwentest.plan_silence_actions_batch_transformers(
                    model=qb.model,
                    processor=qb.processor,
                    device=qb.device,
                    events=s_batch,
                    batch_size=ACTION_PLANNER_BATCH_SIZE,
                    temperature=0.1,
                    max_new_tokens=SILENCE_PLANNER_MAX_NEW_TOKENS,
                )
            rows = resp.get("items") if isinstance(resp, dict) else None
            if not isinstance(rows, list):
                rows = []

            for i, job in enumerate(s_batch):
                row = rows[i] if (i < len(rows) and isinstance(rows[i], dict)) else {"actions": [], "notes": "missing_silence_result"}
                acts = row.get("actions") if isinstance(row, dict) else []
                if not isinstance(acts, list):
                    acts = []

                active_rows = job.get("active_objects") if isinstance(job.get("active_objects"), list) else []
                ordered_names: List[str] = []
                for r in active_rows:
                    if not isinstance(r, dict):
                        continue
                    nm = str(r.get("name", "") or "").strip()
                    if nm:
                        ordered_names.append(nm)
                name_rank = {nm.lower(): idx for idx, nm in enumerate(ordered_names)}

                parsed_actions: List[Dict[str, Any]] = []
                seen_targets: set = set()
                for a in acts:
                    if not isinstance(a, dict):
                        continue
                    if not ordered_names:
                        continue
                    t = str(a.get("type", "") or "").strip().lower()
                    if t not in ("delete_by_name", "delete_self"):
                        continue
                    target = str(a.get("target", "") or "").strip()
                    if not target:
                        continue
                    lk = target.lower()
                    if name_rank and lk not in name_rank:
                        continue
                    if lk in seen_targets:
                        continue
                    one = {
                        "type": "delete_by_name",
                        "target": target,
                        "sync_local": _normalize_sync_local_for_req(
                            a.get("sync_local"),
                            req_start=int(job.get("start_word_index", 0) or 0),
                            req_end=int(job.get("end_word_index", 0) or 0),
                        ),
                    }
                    one = convert_local_sync_to_absolute(
                        one,
                        event_start_word=int(job.get("start_word_index", 0) or 0),
                        event_end_word=int(job.get("end_word_index", 0) or 0),
                    )
                    parsed_actions.append(one)
                    seen_targets.add(lk)

                # Enforce deletion coverage for all active objects in top-to-bottom order.
                s_word = int(job.get("start_word_index", 0) or 0)
                e_word = int(job.get("end_word_index", s_word) or s_word)
                if e_word < s_word:
                    e_word = s_word
                span = max(0, e_word - s_word)
                den = max(1, len(ordered_names))
                for nm in ordered_names:
                    lk = nm.lower()
                    if lk in seen_targets:
                        continue
                    pos = int(name_rank.get(lk, 0))
                    start_off = 0 if span <= 0 else int(round((float(pos) / float(max(1, den - 1))) * float(span)))
                    hard = {
                        "type": "delete_by_name",
                        "target": str(nm),
                        "source": "active_pool_backfill",
                        "sync_local": {"start_word_offset": int(start_off), "end_word_offset": int(start_off)},
                    }
                    parsed_actions.append(
                        convert_local_sync_to_absolute(
                            hard,
                            event_start_word=s_word,
                            event_end_word=e_word,
                        )
                    )
                    seen_targets.add(lk)

                # Keep active pool in sync immediately after this silence.
                for lk in seen_targets:
                    active_objects_by_key.pop(lk, None)

                silence_events_out.append(
                    {
                        "event": {
                            "kind": "silence",
                            "chapter_index": int(job.get("chapter_index", 0) or 0),
                            "chunk_index": int(job.get("chunk_index", 0) or 0),
                            "segment_index": int(job.get("segment_index", 0) or 0),
                            "batch_index": int(batch_i),
                            "name": str(job.get("name", "") or ""),
                            "start_word_index": int(job.get("start_word_index", 0) or 0),
                            "end_word_index": int(job.get("end_word_index", 0) or 0),
                            "duration_sec": float(job.get("duration_sec", INTER_CHUNK_SILENCE_SEC) or INTER_CHUNK_SILENCE_SEC),
                            "step_key": str(job.get("step_key", "") or ""),
                            "active_objects_count_before": int(len(ordered_names)),
                            "active_objects_count_after": int(len(active_objects_by_key)),
                        },
                        "planner": "silence",
                        "qwen_payload": row,
                        "base_actions_parsed": parsed_actions,
                    }
                )

            self._dbg(
                "Silence action batch done",
                data={
                    "batch_index": batch_i,
                    "items": len(s_batch),
                    "elapsed_ms": round((time.perf_counter() - qwen_t0) * 1000.0, 2),
                    "cuda_mem": _cuda_mem_snapshot(),
                },
            )

        all_events = visual_events_out + silence_events_out
        all_events.sort(
            key=lambda p: (
                int((p.get("event") or {}).get("chapter_index", 0) or 0),
                int((p.get("event") or {}).get("start_word_index", 0) or 0),
                0 if str((p.get("event") or {}).get("kind", "") or "") == "silence" else 1,
            )
        )
        for i, ev in enumerate(all_events):
            e0 = ev.setdefault("event", {})
            e0["event_index"] = int(i)

        return {
            "enabled": True,
            "mode": "static_batch_v2",
            "whiteboard_width": self.whiteboard_width,
            "whiteboard_height": self.whiteboard_height,
            "space_planner_batch_size": int(SPACE_PLANNER_BATCH_SIZE),
            "action_planner_batch_size": int(ACTION_PLANNER_BATCH_SIZE),
            "space_planner_max_new_tokens": int(SPACE_PLANNER_MAX_NEW_TOKENS),
            "visual_planner_max_new_tokens": int(VISUAL_PLANNER_MAX_NEW_TOKENS),
            "silence_planner_max_new_tokens": int(SILENCE_PLANNER_MAX_NEW_TOKENS),
            "events_count": len(all_events),
            "events": all_events,
            "space_planner_chunks": planned_chunks,
            "active_objects_final": sorted_active_objects_for_cleanup(active_objects_by_key),
        }

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
        # bbox_map can be:
        #   - {processed_id: {...}} (legacy)
        #   - {"by_processed_id": {processed_id: {"labels": {...}}}} (current EfficientSAM3Clusters)
        bbox_root = None
        if isinstance(sam_bbox_map_any, dict):
            bbox_root = sam_bbox_map_any.get("result") if "result" in sam_bbox_map_any else sam_bbox_map_any

        if not isinstance(bbox_root, dict):
            bbox_root = {}

        by_pid_root = bbox_root
        if isinstance(bbox_root.get("by_processed_id"), dict):
            by_pid_root = bbox_root.get("by_processed_id") or {}

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

            sam_pid_payload = by_pid_root.get(pid)
            if isinstance(sam_pid_payload, dict) and isinstance(sam_pid_payload.get("labels"), dict):
                sam_pid_payload = sam_pid_payload.get("labels")
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
        - Every action has absolute chapter word-index sync.
        - Also keeps local action sync offsets from event range.
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
            if end_wi is None:
                end_wi = start_wi
            if end_wi < start_wi:
                end_wi = start_wi

            for local_i, a in enumerate(acts):
                if not isinstance(a, dict):
                    continue
                abs_sync = a.get("sync_absolute")
                if not isinstance(abs_sync, dict):
                    abs_sync = {"start_word_index": start_wi, "end_word_index": end_wi}

                local_sync = a.get("sync_local")
                if not isinstance(local_sync, dict):
                    local_sync = {"start_word_offset": 0, "end_word_offset": max(0, end_wi - start_wi)}

                flat_actions.append({
                    "global_action_index": global_action_i,
                    "event_index": int(ev.get("event_index", ev_i)),
                    "action_index_in_event": int(local_i),
                    "chapter_index": chapter_index,
                    "chunk_index": int(ev.get("chunk_index", 0) or 0),
                    "segment_index": int(ev.get("segment_index", 0) or 0),
                    "batch_index": int(ev.get("batch_index", 0) or 0),
                    "event_kind": kind,
                    "event_duration_sec": float(ev.get("duration_sec", 0.0) or 0.0),
                    "event_start_word_index": start_wi,
                    "event_end_word_index": end_wi,
                    "sync": {
                        "sync_kind": "silence" if kind == "silence" else "word",
                        "absolute": {
                            "start_word_index": int(abs_sync.get("start_word_index", start_wi) or start_wi),
                            "end_word_index": int(abs_sync.get("end_word_index", end_wi) or end_wi),
                        },
                        "local": {
                            "start_word_offset": int(local_sync.get("start_word_offset", local_sync.get("start", 0)) or 0),
                            "end_word_offset": int(local_sync.get("end_word_offset", local_sync.get("end", 0)) or 0),
                        },
                        "silence_seconds": float(ev.get("duration_sec", 0.0) or 0.0) if kind == "silence" else None,
                    },
                    "action": a,
                })
                global_action_i += 1

        out = {
            "schema": "whiteboard_action_timeline_v2",
            "mode": str(whiteboard_actions_payload.get("mode", "") or ""),
            "chapters": [chapters_meta[k] for k in sorted(chapters_meta.keys())],
            "actions": flat_actions,
            "counts": {
                "events": len(events),
                "actions": len(flat_actions),
            },
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

        # Preload embedding workers while GPT timeline/chapter generation runs.
        embed_preload_threads: List[threading.Thread] = []
        if integrate_image_pipeline:
            embed_preload_threads.append(self._start_minilm_loader(label="models.preload_minilm", warmup=True))
            embed_preload_threads.append(self._start_siglip_loader(label="models.preload_siglip", warmup=True))

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
                speech_bundle, model_speech = self.generate_speech_for_chapter(topic, ch, i, chapter_count)
                speech_flat = str(speech_bundle.get("speech", "") or "")
                images_plan, model_images = self.generate_images_for_speech(speech_flat, chapter_index=i)
                image_text_maps = build_image_text_maps(speech_flat, images_plan)

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
                    "speech": speech_flat,
                    "speech_step_order": speech_bundle.get("speech_step_order", []),
                    "speech_step_map": speech_bundle.get("speech_step_map", {}),
                    "speech_step_ranges": speech_bundle.get("speech_step_ranges", {}),
                    "timeline_steps": speech_bundle.get("timeline_steps", []),
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

        diagram_prompt_grouping_debug: Dict[str, Any] = {
            "ok": True,
            "note": "skipped",
            "reason": "integrate_image_pipeline_disabled",
        }
        if integrate_image_pipeline:
            with self._timed("diagram.group_close_prompts"):
                diagram_prompt_grouping_debug = self._group_and_rewrite_diagram_prompts(
                    all_prompts_meta=all_prompts_meta,
                    all_diagram_required_objects=all_diagram_required_objects,
                    chapters_out=out["chapters"],
                    similarity_threshold=0.84,
                    relaxed_topic_similarity=0.56,
                    centroid_margin_for_umbrella=0.06,
                )
            self._dbg(
                "Diagram prompt grouping",
                data={
                    "groups_found": int((diagram_prompt_grouping_debug or {}).get("groups_found", 0) or 0),
                    "rewrites": len((diagram_prompt_grouping_debug or {}).get("replacement_map", {}) or {}),
                },
            )

        if not integrate_image_pipeline:
            out["timings_ms"] = list(self._timings_ms)
            out["run_elapsed_ms"] = round((time.perf_counter() - run_t0) * 1000.0, 2)
            return out

        if not all_prompts_meta:
            if embed_preload_threads:
                with self._timed("models.wait_preload_siglip_minilm_no_prompts"):
                    for t in embed_preload_threads:
                        t.join()
            out["image_pipeline"]["debug"] = {
                "note": "no_image_prompts",
                "diagram_prompt_grouping": diagram_prompt_grouping_debug,
            }
            out["timings_ms"] = list(self._timings_ms)
            out["run_elapsed_ms"] = round((time.perf_counter() - run_t0) * 1000.0, 2)
            return out

        if embed_preload_threads:
            with self._timed("models.wait_preload_siglip_minilm"):
                for t in embed_preload_threads:
                    t.join()

        # selection + refinement only (returns handle so colours can keep running)
        with self._timed("image_pipeline.selection_only"):
            assets, dbg, handle = self._orchestrate_images_selection_only(
                lesson_topic=topic,
                all_image_requests=all_prompts_meta,
                diagram_required_objects_by_base_context=all_diagram_required_objects,
            )
        dbg["diagram_prompt_grouping"] = diagram_prompt_grouping_debug

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

        # 1) Qwen action planning (static batched v2; Qwen must still be loaded here)
        qwen_actions_err: Optional[str] = None
        try:
            with self._timed("qwen.whiteboard_action_planning"):
                out["whiteboard_actions"] = self._run_qwen_actions_static_batched(
                    chapters_out=out["chapters"],
                    assets_by_name=assets_by_name,
                )
        except Exception as e:
            qwen_actions_err = f"{type(e).__name__}: {e}"
            out["whiteboard_actions"] = {
                "enabled": False,
                "error": qwen_actions_err,
                "events_count": 0,
                "events": [],
            }
            pipe_dbg = out.setdefault("image_pipeline", {}).setdefault("debug", {})
            pipe_errs = pipe_dbg.setdefault("pipeline_errors", {})
            pipe_errs["qwen_action_planning"] = qwen_actions_err
            self._dbg("Qwen action planning failed", data={"err": qwen_actions_err})

        # save index-synced action timeline immediately after Qwen stage
        if qwen_actions_err is None:
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
        else:
            out["whiteboard_actions"]["saved_action_timeline_json"] = None

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

        qwen_reload_thread = threading.Thread(target=_reload_qwen_after_sam, name="reload_qwen_after_sam", daemon=True)
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
                qwen_reload_thread.join(timeout=QWEN_RELOAD_TIMEOUT_SEC)
            if qwen_reload_thread.is_alive():
                qwen_reload_err = qwen_reload_err or f"timeout_after_{QWEN_RELOAD_TIMEOUT_SEC:.1f}s"
                self._dbg(
                    "Qwen reload timeout",
                    data={"timeout_sec": QWEN_RELOAD_TIMEOUT_SEC},
                )

        if qwen_reload_err:
            out["diagram_cluster_labels"] = {"ok": False, "error": f"qwen_reload_failed: {qwen_reload_err}"}
        else:
            # 7) NEW: run diagram cluster labeling (ONLY matched clusters via SAM bboxes)
            with self._timed("qwen.diagram_cluster_labeling"):
                out["diagram_cluster_labels"] = self._run_qwen_diagram_cluster_labeling(
                    assets_by_name=assets,
                    sam_bbox_map_any=out.get("efficientsam3_bboxes", {}),
                )

        # 8) Qwen is done; unload it and keep Comfy hot for the next run.
        qwen_final_unload_err: Optional[str] = None
        comfy_warmup_err: Optional[str] = None
        try:
            from shared_models import unload_qwen
            with self._timed("models.unload_qwen_after_diagram_labeling"):
                unload_qwen()
        except Exception as e:
            qwen_final_unload_err = repr(e)

        try:
            import ComfyFluxClient
            with self._timed("comfy.warmup_after_qwen", gpu_index=self.gpu_index):
                ComfyFluxClient.warmup_server_models(
                    workflow_path=os.getenv("COMFY_WORKFLOW_JSON") or None,
                    comfy_url=os.getenv("COMFY_URL") or None,
                    prompt_text=(os.getenv("COMFY_WARMUP_PROMPT") or "simple black line drawing of one circle"),
                    batch_size=1,
                    wait_timeout_sec=900.0,
                )
        except Exception as e:
            comfy_warmup_err = repr(e)

        pipe_dbg = out.setdefault("image_pipeline", {}).setdefault("debug", {})
        pipe_errs = pipe_dbg.setdefault("pipeline_errors", {})
        if qwen_final_unload_err:
            pipe_errs["qwen_unload_after_diagram_labeling"] = qwen_final_unload_err
        if comfy_warmup_err:
            pipe_errs["comfy_warmup_after_qwen"] = comfy_warmup_err

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
    if not hasattr(LessonTimeline, "_run_efficientsam3_bboxes"):
        LessonTimeline._run_efficientsam3_bboxes = _run_efficientsam3_bboxes  # type: ignore
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
    parser.add_argument("--wb-w", type=int, default=4000)
    parser.add_argument("--wb-h", type=int, default=4000)
    parser.add_argument("--processed-dir", default="ProcessedImages")
    parser.add_argument("--gpt-cache", action="store_true", help="Cache GPT Responses API calls to a local json file for quick test reruns.")
    parser.add_argument("--gpt-cache-file", default="", help="Optional cache file path (default: PipelineOutputs/_gpt_call_cache.json).")
    args = parser.parse_args()

    pipe = LessonTimeline(
        gpu_index=args.gpu,
        cpu_threads=args.cpu_threads,
        whiteboard_width=args.wb_w,
        whiteboard_height=args.wb_h,
        processed_images_dir=args.processed_dir,
        gpt_cache=bool(args.gpt_cache),
        gpt_cache_file=(str(args.gpt_cache_file).strip() or None),
    )
    result = pipe.run_full(args.topic, integrate_image_pipeline=(not args.no_images))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {args.out}")
