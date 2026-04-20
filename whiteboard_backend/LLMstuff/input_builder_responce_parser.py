from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from LLMstuff.prompts import (
    CHAPTER_SPEECH_SCHEMA,
    IMAGE_REQUEST_SCHEMA,
    LOGICAL_TIMELINE_SCHEMA,
    TEXT_REQUEST_SCHEMA,
    chapter_speech_system_prompt,
    image_request_system_prompt,
    logical_timeline_system_prompt,
    qwen_space_planner_system_prompt,
    qwen_step_rewrite_system_prompt,
    text_request_system_prompt,
)


def normalize_ws(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def normalize_step_id(raw_step_id: Any, *, chapter_id: str, fallback_index: int) -> str:
    raw = normalize_ws(raw_step_id).lower()
    raw = raw.replace(")", ".").replace("_", ".").replace("-", ".")
    raw = re.sub(r"\s+", "", raw)
    raw = raw.strip(".")
    raw = raw.replace("chapter", "c").replace("step", "s")
    if raw.startswith(f"{chapter_id}.s"):
        suffix = raw.split(f"{chapter_id}.s", 1)[1]
        digits = re.findall(r"\d+", suffix)
        if digits:
            return f"{chapter_id}.s{int(digits[0])}"
    if raw.startswith(f"{chapter_id}."):
        suffix = raw.split(f"{chapter_id}.", 1)[1]
        digits = re.findall(r"\d+", suffix)
        if digits:
            return f"{chapter_id}.s{int(digits[0])}"
    digits = re.findall(r"\d+", raw)
    if digits:
        return f"{chapter_id}.s{int(digits[-1])}"
    return f"{chapter_id}.s{max(1, int(fallback_index))}"


def build_openai_json_schema(schema_name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": schema_name,
            "schema": schema,
            "strict": True,
        }
    }


def build_logical_timeline_request(topic: str, *, chapter_cap: int = 3) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    user = (
        f"Topic: {normalize_ws(topic)}\n\n"
        "Build the logical timeline now."
    )
    input_items = [
        {"role": "developer", "content": logical_timeline_system_prompt(chapter_cap=chapter_cap)},
        {"role": "user", "content": user},
    ]
    return input_items, build_openai_json_schema("new_timeline_plan", LOGICAL_TIMELINE_SCHEMA)


def parse_logical_timeline_output(raw_text: str, *, chapter_cap: int = 3) -> List[Dict[str, Any]]:
    payload = json.loads(raw_text)
    chapters_raw = payload.get("chapters")
    if not isinstance(chapters_raw, list) or not chapters_raw:
        raise ValueError("logical timeline output is missing chapters")

    out: List[Dict[str, Any]] = []
    for chapter_index, chapter_row in enumerate(chapters_raw[: max(1, int(chapter_cap))], start=1):
        if not isinstance(chapter_row, dict):
            continue
        chapter_id = f"c{chapter_index}"
        chapter_title = normalize_ws(chapter_row.get("title")) or f"Chapter {chapter_index}"
        steps_raw = chapter_row.get("steps")
        if not isinstance(steps_raw, dict) or not steps_raw:
            continue

        steps: "OrderedDict[str, str]" = OrderedDict()
        seen: set[str] = set()
        for fallback_index, (raw_key, raw_value) in enumerate(steps_raw.items(), start=1):
            step_text = normalize_ws(raw_value)
            if not step_text:
                continue
            step_id = normalize_step_id(raw_key, chapter_id=chapter_id, fallback_index=fallback_index)
            if step_id in seen:
                step_id = f"{chapter_id}.s{len(seen) + 1}"
            seen.add(step_id)
            steps[step_id] = step_text

        if not steps:
            continue

        out.append(
            {
                "chapter_id": chapter_id,
                "title": chapter_title,
                "steps": steps,
            }
        )

    if not out:
        raise ValueError("logical timeline output contained no valid chapters")
    return out


def render_chapter_timeline_for_speech(chapter: Dict[str, Any]) -> str:
    chapter_id = normalize_ws(chapter.get("chapter_id"))
    title = normalize_ws(chapter.get("title"))
    steps = chapter.get("steps") if isinstance(chapter.get("steps"), dict) else {}
    lines = [f"Chapter: {chapter_id} - {title}"]
    lines.append("Step timeline:")
    for step_id, text in steps.items():
        lines.append(f"[{step_id}] {normalize_ws(text)}")
    return "\n".join(lines).strip()


def build_chapter_speech_request(topic: str, chapter: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    chapter_text = render_chapter_timeline_for_speech(chapter)
    step_ids = list((chapter.get("steps") or {}).keys())
    user = (
        f"Topic: {normalize_ws(topic)}\n"
        f"Chapter id: {normalize_ws(chapter.get('chapter_id'))}\n"
        f"Chapter title: {normalize_ws(chapter.get('title'))}\n\n"
        "Required step order:\n"
        f"{json.dumps(step_ids, ensure_ascii=False)}\n\n"
        "Chapter logical timeline:\n"
        f"{chapter_text}"
    )
    input_items = [
        {"role": "developer", "content": chapter_speech_system_prompt()},
        {"role": "user", "content": user},
    ]
    return input_items, build_openai_json_schema("chapter_speech_plan", CHAPTER_SPEECH_SCHEMA)


def parse_chapter_speech_output(raw_text: str, *, chapter: Dict[str, Any]) -> "OrderedDict[str, str]":
    payload = json.loads(raw_text)
    steps_raw = payload.get("steps")
    if not isinstance(steps_raw, dict):
        raise ValueError("chapter speech output is missing steps")

    expected_steps = list((chapter.get("steps") or {}).keys())
    chapter_id = normalize_ws(chapter.get("chapter_id"))
    out: "OrderedDict[str, str]" = OrderedDict()
    alias_map = {step_id: step_id for step_id in expected_steps}
    for step_id in expected_steps:
        alias_map[step_id.split(".")[-1]] = step_id

    for fallback_index, step_id in enumerate(expected_steps, start=1):
        raw_value = steps_raw.get(step_id)
        if raw_value is None:
            alt = steps_raw.get(step_id.split(".")[-1])
            if alt is not None:
                raw_value = alt
        if raw_value is None:
            for candidate_key, candidate_value in steps_raw.items():
                normalized = normalize_step_id(candidate_key, chapter_id=chapter_id, fallback_index=fallback_index)
                if normalized == step_id:
                    raw_value = candidate_value
                    break
        cleaned = normalize_ws(raw_value)
        if not cleaned:
            cleaned = normalize_ws((chapter.get("steps") or {}).get(step_id))
        out[step_id] = cleaned

    if not out:
        raise ValueError("chapter speech output contained no valid step speech")
    return out


def render_step_text_map(step_map: Dict[str, str], *, heading: str) -> str:
    lines = [heading]
    for step_id, text in step_map.items():
        lines.append(f"[{step_id}] {normalize_ws(text)}")
    return "\n".join(lines).strip()


def build_image_request_request(topic: str, chapter: Dict[str, Any], speech_steps: Dict[str, str]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    user = (
        f"Topic: {normalize_ws(topic)}\n"
        f"Chapter id: {normalize_ws(chapter.get('chapter_id'))}\n"
        f"Chapter title: {normalize_ws(chapter.get('title'))}\n\n"
        "Speech, split by step:\n"
        f"{render_step_text_map(speech_steps, heading='Chapter speech:')}"
    )
    input_items = [
        {"role": "developer", "content": image_request_system_prompt()},
        {"role": "user", "content": user},
    ]
    return input_items, build_openai_json_schema("chapter_image_requests", IMAGE_REQUEST_SCHEMA)


def build_text_request_request(topic: str, chapter: Dict[str, Any], speech_steps: Dict[str, str]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    user = (
        f"Topic: {normalize_ws(topic)}\n"
        f"Chapter id: {normalize_ws(chapter.get('chapter_id'))}\n"
        f"Chapter title: {normalize_ws(chapter.get('title'))}\n\n"
        "Speech, split by step:\n"
        f"{render_step_text_map(speech_steps, heading='Chapter speech:')}"
    )
    input_items = [
        {"role": "developer", "content": text_request_system_prompt()},
        {"role": "user", "content": user},
    ]
    return input_items, build_openai_json_schema("chapter_text_requests", TEXT_REQUEST_SCHEMA)


def _normalize_step_list(raw_steps: Any, *, allowed_steps: Sequence[str]) -> List[str]:
    allowed = {str(step_id): str(step_id) for step_id in allowed_steps}
    allowed_local = {str(step_id).split(".")[-1]: str(step_id) for step_id in allowed_steps}
    out: List[str] = []
    if not isinstance(raw_steps, list):
        return out
    for item in raw_steps:
        raw = normalize_ws(item)
        if not raw:
            continue
        step_id = allowed.get(raw) or allowed_local.get(raw)
        if step_id and step_id not in out:
            out.append(step_id)
    return out


def parse_image_request_output(raw_text: str, *, chapter: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = json.loads(raw_text)
    rows = payload.get("images")
    if not isinstance(rows, list):
        raise ValueError("image request output is missing images")
    allowed_steps = list((chapter.get("steps") or {}).keys())
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = normalize_ws(row.get("name"))
        if not name:
            continue
        diagram = 1 if int(row.get("diagram", 0) or 0) == 1 else 0
        required_objects = row.get("required_objects")
        if not isinstance(required_objects, list):
            required_objects = []
        cleaned_required_objects = [normalize_ws(item) for item in required_objects if normalize_ws(item)]
        relevant_steps = _normalize_step_list(row.get("relevant_steps"), allowed_steps=allowed_steps)
        if not relevant_steps:
            relevant_steps = allowed_steps[:1]
        out.append(
            {
                "name": name,
                "diagram": diagram,
                "required_objects": cleaned_required_objects if diagram == 1 else [],
                "relevant_steps": relevant_steps,
            }
        )
    return out


def parse_text_request_output(raw_text: str, *, chapter: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = json.loads(raw_text)
    rows = payload.get("texts")
    if not isinstance(rows, list):
        raise ValueError("text request output is missing texts")
    allowed_steps = list((chapter.get("steps") or {}).keys())
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "") or "").strip()
        if not name:
            continue
        style_tag = normalize_ws(row.get("text_style_description")) or "note"
        relevant_steps = _normalize_step_list(row.get("relevant_steps"), allowed_steps=allowed_steps)
        if not relevant_steps:
            relevant_steps = allowed_steps[:1]
        out.append(
            {
                "name": name,
                "text_style_description": style_tag[:40],
                "relevant_steps": relevant_steps,
            }
        )
    return out


def collect_step_objects(
    *,
    step_id: str,
    image_requests: Sequence[Dict[str, Any]],
    text_requests: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    objects: List[Dict[str, Any]] = []
    for row in image_requests:
        if step_id not in (row.get("relevant_steps") or []):
            continue
        objects.append(
            {
                "object_type": "image",
                "name": normalize_ws(row.get("name")),
                "diagram": 1 if int(row.get("diagram", 0) or 0) == 1 else 0,
                "required_objects": list(row.get("required_objects") or []),
                "relevant_steps": list(row.get("relevant_steps") or []),
            }
        )
    for row in text_requests:
        if step_id not in (row.get("relevant_steps") or []):
            continue
        objects.append(
            {
                "object_type": "text",
                "name": str(row.get("name", "") or "").strip(),
                "text_style_description": normalize_ws(row.get("text_style_description")) or "note",
                "relevant_steps": list(row.get("relevant_steps") or []),
            }
        )
    return [row for row in objects if row.get("name")]


def build_qwen_step_prompt(*, step_id: str, speech_text: str, step_objects: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    prompt = (
        f"Step id: {normalize_ws(step_id)}\n\n"
        "Speech:\n"
        f"{str(speech_text or '').strip()}\n\n"
        "Relevant objects:\n"
        f"{json.dumps(list(step_objects), ensure_ascii=False, indent=2)}\n\n"
        "Return the rewritten speech and sync map now."
    )
    return qwen_step_rewrite_system_prompt(), prompt


def build_qwen_space_planner_prompt(
    *,
    chunk_id: str,
    chunk_title: str,
    merged_speech: str,
    board_nodes: Dict[str, Any],
    total_silences: int,
    objects: Sequence[Dict[str, Any]],
) -> Tuple[str, str]:
    board_payload = {
        "width": int((board_nodes or {}).get("width", 0) or 0),
        "height": int((board_nodes or {}).get("height", 0) or 0),
    }
    prompt = (
        f"Chunk id: {normalize_ws(chunk_id)}\n"
        f"Chunk title: {normalize_ws(chunk_title)}\n\n"
        "Board dimensions in nodes:\n"
        f"{json.dumps(board_payload, ensure_ascii=False)}\n\n"
        f"Global silence count in this merged chunk: {int(total_silences)}\n\n"
        "Merged chunk speech with silence markers kept intact:\n"
        f"{str(merged_speech or '').strip()}\n\n"
        "Objects already converted into node format:\n"
        f"{json.dumps(list(objects), ensure_ascii=False, indent=2)}\n\n"
        "Remember:\n"
        "- The objects are already globally synced to the merged chunk silences.\n"
        "- range is end minus start.\n"
        "- nodes is the object's filled footprint.\n"
        "- Work only with board nodes.\n"
        "- Return the JSON action list now."
    )
    return qwen_space_planner_system_prompt(), prompt


def parse_qwen_step_output(
    raw_text: str,
    *,
    allowed_object_names: Iterable[str],
) -> Dict[str, Any]:
    payload = json.loads(raw_text)
    speech = str(payload.get("speech", "") or "").strip()
    if not speech:
        raise ValueError("qwen step output is missing speech")
    allowed = {str(name): str(name) for name in allowed_object_names if str(name).strip()}
    sync_rows = payload.get("sync_map")
    if not isinstance(sync_rows, list):
        sync_rows = []

    out_sync: List[Dict[str, Any]] = []
    for row in sync_rows:
        if not isinstance(row, dict):
            continue
        raw_name = str(row.get("name", "") or "").strip()
        name = allowed.get(raw_name)
        if not name:
            continue
        start = row.get("start")
        end = row.get("end")
        try:
            start_index = int(start)
        except Exception:
            continue
        end_index: Optional[int]
        if end is None:
            end_index = None
        else:
            try:
                end_index = int(end)
            except Exception:
                end_index = None
        out_sync.append(
            {
                "name": name,
                "start": max(0, start_index),
                "end": None if end_index is None else max(0, end_index),
            }
        )

    return {
        "speech": speech,
        "sync_map": out_sync,
    }


def extract_pause_markers(speech_text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for index, match in enumerate(re.finditer(r"%(\d+(?:\.\d{3}))", str(speech_text or ""))):
        try:
            seconds = float(match.group(1))
        except Exception:
            continue
        out.append(
            {
                "pause_index": index,
                "seconds": seconds,
                "token": match.group(0),
                "char_start": match.start(),
                "char_end": match.end(),
            }
        )
    return out
