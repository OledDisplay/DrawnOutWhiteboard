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
    qwen_c2_component_verifier_system_prompt,
    qwen_diagram_component_stroke_match_system_prompt,
    qwen_diagram_action_planner_system_prompt,
    qwen_non_semantic_image_description_system_prompt,
    qwen_stroke_meaning_filter_system_prompt,
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


def build_qwen_diagram_action_planner_prompt(
    *,
    step_id: str,
    speech: str,
    images: Sequence[Dict[str, Any]],
) -> Tuple[str, str]:
    payload = {
        "step_id": normalize_ws(step_id),
        "speech": str(speech or "").strip(),
        "images": list(images or []),
    }
    prompt = (
        "Input JSON (compact, no extra whitespace):\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )
    return qwen_diagram_action_planner_system_prompt(), prompt


def build_qwen_c2_component_verifier_prompt(
    *,
    prompt: str,
    required_objects: Sequence[str],
) -> Tuple[str, str]:
    payload = {
        "prompt": normalize_ws(prompt),
        "required_objects": [normalize_ws(item) for item in (required_objects or []) if normalize_ws(item)],
    }
    user = (
        "Diagram prompt and required object list:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return qwen_c2_component_verifier_system_prompt(), user


def _clip_compact_text(value: Any, max_chars: int) -> str:
    text = normalize_ws(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def compact_stroke_description_map(stroke_description_payload: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    rows = stroke_description_payload.get("described_lines")
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            stroke_id = int(row.get("source_stroke_index", row.get("described_line_index")))
        except Exception:
            continue
        look = normalize_ws(row.get("look"))
        loc = normalize_ws(row.get("location"))
        if look or loc:
            text = f"{look}; loc {loc}".strip("; ")
        else:
            text = normalize_ws(row.get("description"))
        if text:
            out[str(stroke_id)] = _clip_compact_text(text, 220)
    return dict(sorted(out.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9))


def compact_group_description_map(stroke_description_payload: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    rows = stroke_description_payload.get("groups")
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            group_id = int(row.get("group_index"))
        except Exception:
            continue
        stroke_ids = []
        for value in row.get("source_stroke_indices") if isinstance(row.get("source_stroke_indices"), list) else []:
            try:
                stroke_ids.append(int(value))
            except Exception:
                continue
        desc = _clip_compact_text(row.get("description"), 150)
        if stroke_ids and desc:
            out[str(group_id)] = f"strokes {','.join(str(x) for x in stroke_ids)}; {desc}"
    return dict(sorted(out.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9))


def build_qwen_stroke_meaning_filter_prompt(
    *,
    diagram_name: str,
    components: Sequence[str],
    stroke_description_payload: Dict[str, Any],
) -> Tuple[str, str]:
    payload = {
        "diagram": normalize_ws(diagram_name),
        "components": [normalize_ws(item) for item in (components or []) if normalize_ws(item)][:80],
        "strokes": compact_stroke_description_map(stroke_description_payload),
        "groups": compact_group_description_map(stroke_description_payload),
    }
    user = (
        "Compact diagram stroke map:\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "Return the compressed accepted/groups/rejected JSON now."
    )
    return qwen_stroke_meaning_filter_system_prompt(), user


def build_qwen_non_semantic_image_description_prompt() -> Tuple[str, str]:
    return qwen_non_semantic_image_description_system_prompt(), "Describe the colored non-grayscale visual content in this image now."


def parse_qwen_non_semantic_image_description_output(raw_text: str) -> Dict[str, Any]:
    payload = _extract_first_json_object(raw_text)
    description = _clip_compact_text(payload.get("description"), 900)
    if not description:
        description = _clip_compact_text(str(raw_text or ""), 900)
    return {"description": description}


def _compact_component_description(value: Any, max_chars: int = 700) -> str:
    return _clip_compact_text(value, max(80, int(max_chars or 700)))


def build_qwen_diagram_component_stroke_match_prompt(
    *,
    diagram_name: str,
    candidate_objects: Sequence[Dict[str, Any]],
    stroke_description_payload: Dict[str, Any],
    stroke_meaning_payload: Optional[Dict[str, Any]] = None,
    components: Sequence[Dict[str, Any]],
    refined_visual_description_max_chars: int = 700,
) -> Tuple[str, str]:
    candidates_out: List[Dict[str, Any]] = []
    for index, row in enumerate(candidate_objects or [], start=1):
        if not isinstance(row, dict):
            continue
        stroke_ids = _coerce_int_list_light(row.get("stroke_indexes", row.get("stroke_ids")), cap=500)
        if not stroke_ids:
            continue
        xy = row.get("xy")
        if not isinstance(xy, list) or len(xy) != 2:
            bbox = row.get("bbox_xyxy") if isinstance(row.get("bbox_xyxy"), list) else []
            if len(bbox) == 4:
                try:
                    xy = [round((float(bbox[0]) + float(bbox[2])) / 2.0, 2), round((float(bbox[1]) + float(bbox[3])) / 2.0, 2)]
                except Exception:
                    xy = [None, None]
            else:
                xy = [None, None]
        desc = _clip_compact_text(
            row.get("qwen_visual_description")
            or row.get("visual_description")
            or row.get("non_semantic_visual_description"),
            900,
        )
        if not desc:
            continue
        candidates_out.append(
            {
                "id": normalize_ws(row.get("candidate_id")) or f"C{index}",
                "stroke_ids": stroke_ids,
                "xy": xy,
                "visual": desc,
            }
        )

    component_rows: List[Dict[str, Any]] = []
    seen_components: set[str] = set()
    for row in components or []:
        if not isinstance(row, dict):
            continue
        name = normalize_ws(row.get("name") or row.get("label"))
        if not name or name.casefold() in seen_components:
            continue
        seen_components.add(name.casefold())
        component_rows.append(
            {
                "name": name,
                "image_visual": _compact_component_description(
                    row.get("qwen_visual_description") or row.get("image_visual_description"),
                    max_chars=900,
                ),
                "refined_visual": _compact_component_description(
                    row.get("refined_visual_description"),
                    max_chars=refined_visual_description_max_chars,
                ),
            }
        )

    line_map = {
        "strokes": compact_stroke_description_map(stroke_description_payload),
        "groups": compact_group_description_map(stroke_description_payload),
    }
    if isinstance(stroke_meaning_payload, dict):
        optimized = stroke_meaning_payload.get("optimized") if isinstance(stroke_meaning_payload.get("optimized"), dict) else stroke_meaning_payload
        line_map["filtered"] = {
            "accepted": list((optimized or {}).get("accepted") or []),
            "groups": list((optimized or {}).get("groups") or []),
        }

    payload = {
        "diagram": normalize_ws(diagram_name),
        "candidate_objects": candidates_out,
        "line_descriptors": line_map,
        "components": component_rows,
    }
    user = (
        "Diagram component matching input:\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "Return the component-to-stroke match JSON now."
    )
    return qwen_diagram_component_stroke_match_system_prompt(), user


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
    for index, match in enumerate(re.finditer(r"%(?:\s*)(\d+(?:\.\d+)?|\.\d+)", str(speech_text or ""))):
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


def _extract_action_objects_from_truncated_json(raw_text: str, *, field_name: str = "actions") -> List[Dict[str, Any]]:
    text = str(raw_text or "")
    field_pos = text.find(f'"{field_name}"')
    if field_pos < 0:
        return []
    bracket_pos = text.find("[", field_pos)
    if bracket_pos < 0:
        return []

    out: List[Dict[str, Any]] = []
    in_string = False
    escape = False
    object_depth = 0
    buffer: List[str] = []

    for ch in text[bracket_pos + 1 :]:
        if object_depth == 0:
            if ch == "{":
                object_depth = 1
                buffer = ["{"]
            elif ch == "]":
                break
            continue

        buffer.append(ch)
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            object_depth += 1
            continue
        if ch == "}":
            object_depth -= 1
            if object_depth == 0:
                try:
                    parsed = json.loads("".join(buffer))
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    out.append(parsed)
                buffer = []
    return out


def parse_qwen_space_planner_output(raw_text: str) -> Dict[str, Any]:
    actions_raw: Any = None
    try:
        payload = json.loads(raw_text)
        actions_raw = payload.get("actions")
    except Exception:
        actions_raw = _extract_action_objects_from_truncated_json(raw_text, field_name="actions")

    if not isinstance(actions_raw, list):
        actions_raw = []

    out_actions: List[Dict[str, Any]] = []
    for row in actions_raw:
        if not isinstance(row, dict):
            continue
        name = normalize_ws(row.get("name"))
        action_type = normalize_ws(row.get("type")).lower()
        corners = row.get("corners")
        if not name or action_type not in {"image", "text"} or not isinstance(corners, list) or len(corners) < 2:
            continue
        try:
            draw = 1 if int(row.get("draw", 0) or 0) == 1 else 0
            if row.get("start") is None and row.get("sync_index") is not None:
                start = max(0, int(row.get("sync_index", 0) or 0))
            else:
                start = max(0, int(row.get("start", 0) or 0))
            if row.get("end") is None and row.get("sync_index") is not None:
                end = start
            else:
                end = max(0, int(row.get("end", start) or start))
            span = max(0, int(row.get("range", max(0, end - start)) or max(0, end - start)))
        except Exception:
            continue
        cleaned_corners: List[List[int]] = []
        valid = True
        for corner in corners:
            if not isinstance(corner, (list, tuple)) or len(corner) != 2:
                valid = False
                break
            try:
                cleaned_corners.append([int(corner[0]), int(corner[1])])
            except Exception:
                valid = False
                break
        if not valid:
            continue
        if len(cleaned_corners) == 2:
            xs = [int(cleaned_corners[0][0]), int(cleaned_corners[1][0])]
            ys = [int(cleaned_corners[0][1]), int(cleaned_corners[1][1])]
            cleaned_corners = [
                [min(xs), min(ys)],
                [min(xs), max(ys)],
                [max(xs), min(ys)],
                [max(xs), max(ys)],
            ]
        elif len(cleaned_corners) != 4:
            xs = [corner[0] for corner in cleaned_corners]
            ys = [corner[1] for corner in cleaned_corners]
            cleaned_corners = [
                [min(xs), min(ys)],
                [min(xs), max(ys)],
                [max(xs), min(ys)],
                [max(xs), max(ys)],
            ]
        out_actions.append(
            {
                "draw": draw,
                "type": action_type,
                "name": name,
                "start": start,
                "end": max(start, end),
                "range": span,
                "corners": cleaned_corners,
            }
        )

    if not out_actions:
        raise ValueError("space planner output is missing actions")
    return {"actions": out_actions}


def parse_qwen_diagram_action_planner_output(raw_text: str) -> Dict[str, Any]:
    actions_raw: Any = None
    try:
        payload = json.loads(raw_text)
        actions_raw = payload.get("actions")
    except Exception:
        actions_raw = _extract_action_objects_from_truncated_json(raw_text, field_name="actions")

    if not isinstance(actions_raw, list):
        actions_raw = []

    allowed_types = {
        "highlight_image",
        "write_text_image",
        "highlight_component",
        "zoom_component",
        "label_component",
        "connect_component_to_component",
        "write_text_component",
    }
    out_actions: List[Dict[str, Any]] = []
    for row in actions_raw:
        if not isinstance(row, dict):
            continue
        action_type = normalize_ws(row.get("type"))
        target = str(row.get("target", "") or "").strip()
        data = str(row.get("data", "") or "")
        if action_type not in allowed_types or not target:
            continue
        try:
            sync_index = max(1, int(row.get("sync_index", 1) or 1))
            init = 1 if int(row.get("init", 0) or 0) == 1 else 0
        except Exception:
            continue
        out_actions.append(
            {
                "type": action_type,
                "target": target,
                "data": data,
                "sync_index": sync_index,
                "init": init,
            }
        )
    return {"actions": out_actions}


def parse_qwen_c2_component_verifier_output(raw_text: str) -> Dict[str, Any]:
    try:
        payload = json.loads(raw_text)
    except Exception:
        payload = {}
    skip_stage1 = 1 if int(payload.get("skip_stage1", 0) or 0) == 1 else 0
    missing_raw = payload.get("missing")
    missing = [normalize_ws(item) for item in (missing_raw if isinstance(missing_raw, list) else []) if normalize_ws(item)]
    if skip_stage1 == 1:
        missing = []
    return {
        "skip_stage1": skip_stage1,
        "missing": missing,
    }


def _extract_first_json_object(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        return {}
    in_string = False
    escape = False
    depth = 0
    for index in range(start, len(text)):
        ch = text[index]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(text[start:index + 1])
                    return payload if isinstance(payload, dict) else {}
                except Exception:
                    return {}
    return {}


def _coerce_int_list_light(value: Any, *, cap: int = 1000) -> List[int]:
    raw_items: List[Any]
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = re.findall(r"-?\d+", value)
    else:
        raw_items = []
    out: List[int] = []
    seen: set[int] = set()
    for item in raw_items:
        try:
            number = int(item)
        except Exception:
            continue
        if number < 0 or number in seen:
            continue
        seen.add(number)
        out.append(number)
        if len(out) >= cap:
            break
    return out


def _stroke_lookup_from_description_payload(stroke_description_payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}
    for row in stroke_description_payload.get("described_lines") if isinstance(stroke_description_payload.get("described_lines"), list) else []:
        if not isinstance(row, dict):
            continue
        try:
            stroke_id = int(row.get("source_stroke_index", row.get("described_line_index")))
        except Exception:
            continue
        lookup[stroke_id] = row
    return lookup


def _fallback_stroke_summary(row: Dict[str, Any]) -> Tuple[str, str]:
    look = _clip_compact_text(row.get("look") or row.get("description"), 140)
    loc = _clip_compact_text(row.get("location"), 80)
    if not loc:
        loc = "from descriptor"
    return look, loc


def parse_qwen_stroke_meaning_filter_output(
    raw_text: str,
    *,
    stroke_description_payload: Dict[str, Any],
) -> Dict[str, Any]:
    payload = _extract_first_json_object(raw_text)
    stroke_lookup = _stroke_lookup_from_description_payload(stroke_description_payload)
    valid_strokes = set(stroke_lookup.keys())

    accepted_by_stroke: Dict[int, Dict[str, Any]] = {}
    accepted_raw = payload.get("accepted")
    if isinstance(accepted_raw, dict):
        accepted_iter = [
            {"s": key, "d": value}
            for key, value in accepted_raw.items()
        ]
    elif isinstance(accepted_raw, list):
        accepted_iter = accepted_raw
    else:
        accepted_iter = []

    for row in accepted_iter:
        if isinstance(row, dict):
            raw_id = row.get("s", row.get("stroke", row.get("stroke_id", row.get("id"))))
            try:
                stroke_id = int(raw_id)
            except Exception:
                continue
            if stroke_id not in valid_strokes:
                continue
            fallback_d, fallback_loc = _fallback_stroke_summary(stroke_lookup[stroke_id])
            accepted_by_stroke[stroke_id] = {
                "stroke": stroke_id,
                "d": _clip_compact_text(row.get("d", row.get("description", fallback_d)), 160) or fallback_d,
                "loc": _clip_compact_text(row.get("loc", row.get("location", fallback_loc)), 90) or fallback_loc,
            }

    parsed_groups: List[Dict[str, Any]] = []
    groups_raw = payload.get("groups")
    if not isinstance(groups_raw, list):
        groups_raw = []
    for index, row in enumerate(groups_raw, start=1):
        if not isinstance(row, dict):
            continue
        stroke_ids = _coerce_int_list_light(row.get("strokes", row.get("stroke_ids")), cap=500)
        stroke_ids = [sid for sid in stroke_ids if sid in valid_strokes]
        if not stroke_ids:
            continue
        for sid in stroke_ids:
            if sid not in accepted_by_stroke:
                fallback_d, fallback_loc = _fallback_stroke_summary(stroke_lookup[sid])
                accepted_by_stroke[sid] = {
                    "stroke": sid,
                    "d": fallback_d,
                    "loc": fallback_loc,
                    "auto_accepted_from_group": True,
                }
        parsed_groups.append(
            {
                "id": normalize_ws(row.get("id")) or f"G{index}",
                "strokes": stroke_ids,
                "d": _clip_compact_text(row.get("d", row.get("description", "")), 180),
                "source": _clip_compact_text(row.get("source", "new"), 40) or "new",
            }
        )

    rejected_ranges: List[Dict[str, Any]] = []
    rejected_raw = payload.get("rejected")
    if not isinstance(rejected_raw, list):
        rejected_raw = []
    for row in rejected_raw:
        if not isinstance(row, dict):
            continue
        rng = _coerce_int_list_light(row.get("range"), cap=2)
        if len(rng) < 2:
            continue
        start, end = sorted([rng[0], rng[1]])
        rejected_ranges.append(
            {
                "range": [start, end],
                "why": _clip_compact_text(row.get("why", row.get("reason", "")), 120),
            }
        )

    accepted = [accepted_by_stroke[sid] for sid in sorted(accepted_by_stroke.keys())]
    return {
        "accepted": accepted,
        "groups": parsed_groups,
        "rejected": rejected_ranges,
        "stats": {
            "accepted_strokes": len(accepted),
            "groups": len(parsed_groups),
            "rejected_ranges": len(rejected_ranges),
            "source_strokes": len(valid_strokes),
        },
    }


def parse_qwen_diagram_component_stroke_match_output(
    raw_text: str,
    *,
    component_names: Sequence[str],
) -> Dict[str, Any]:
    payload = _extract_first_json_object(raw_text)
    raw_components = payload.get("components")
    if isinstance(raw_components, list):
        converted: Dict[str, Any] = {}
        for row in raw_components:
            if not isinstance(row, dict):
                continue
            name = normalize_ws(row.get("name") or row.get("component"))
            if name:
                converted[name] = row
        raw_components = converted
    if not isinstance(raw_components, dict):
        raw_components = {}

    expected = [normalize_ws(name) for name in (component_names or []) if normalize_ws(name)]
    expected_lookup = {name.casefold(): name for name in expected}
    out_components: Dict[str, Dict[str, Any]] = {}
    for raw_name, raw_value in raw_components.items():
        name = expected_lookup.get(normalize_ws(raw_name).casefold()) or normalize_ws(raw_name)
        if not name or name not in expected:
            continue
        row = raw_value if isinstance(raw_value, dict) else {}
        raw_strokes = row.get("stroke_ids", row.get("strokes", row.get("stroke_indexes")))
        stroke_ids = None if raw_strokes is None else _coerce_int_list_light(raw_strokes, cap=1000)
        out_components[name] = {
            "stroke_ids": stroke_ids,
            "visual_description_of_match": _clip_compact_text(
                row.get("visual_description_of_match", row.get("visual", row.get("match_visual_description", ""))),
                1000,
            ),
            "reason": _clip_compact_text(row.get("reason", ""), 1200),
        }

    for name in expected:
        if name not in out_components:
            out_components[name] = {
                "stroke_ids": None,
                "visual_description_of_match": "",
                "reason": "missing from raw Qwen output",
            }

    return {"components": out_components}
