from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import edge_tts
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: edge-tts\n"
        "Install with: pip install edge-tts pygame websockets"
    ) from exc

try:
    import pygame
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pygame\n"
        "Install with: pip install edge-tts pygame websockets"
    ) from exc

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: websockets\n"
        "Install with: pip install edge-tts pygame websockets"
    ) from exc


PAUSE_TOKEN_RE = re.compile(r"^>(\d+(?:\.\d+)?)$")
DEFAULT_INTER_CHAPTER_PAUSE_SEC = 3.0
POLL_INTERVAL_SEC = 0.01
DEFAULT_MIXER_FREQUENCY = 24_000
AUDIO_TICK_TO_SECONDS = 10_000_000.0


# ----------------------------
# Data models
# ----------------------------


@dataclass
class RawToken:
    raw_index: int
    text: str
    is_pause: bool
    pause_seconds: float = 0.0


@dataclass
class WordBoundary:
    offset_sec: float
    duration_sec: float
    text: str


@dataclass
class SegmentPlan:
    chapter_index: int
    segment_index: int
    spoken_text: str
    raw_spoken_indices: List[int]
    raw_pause_indices_after: List[int]
    pause_after_sec: float
    audio_path: Optional[str] = None
    boundaries: List[WordBoundary] = field(default_factory=list)
    estimated_audio_duration_sec: float = 0.0
    chapter_time_offset_sec: float = 0.0
    raw_token_start_times_sec: Dict[int, float] = field(default_factory=dict)


@dataclass
class ChapterPlan:
    chapter_index: int
    raw_speech: str
    raw_tokens: List[RawToken]
    segments: List[SegmentPlan]
    raw_token_start_times_sec: Dict[int, float] = field(default_factory=dict)
    duration_sec: float = 0.0


@dataclass
class ActionPacketSchedule:
    emit_time_sec: float
    emit_word_index: int
    emit_phase: str  # start | end
    payload: Dict[str, Any]


@dataclass
class LessonRuntimeData:
    topic: str
    chapter_map: Dict[int, ChapterPlan]
    chapter_packets: Dict[int, List[ActionPacketSchedule]]


# ----------------------------
# Parsing helpers
# ----------------------------


def _is_pause_token(token: str) -> Optional[float]:
    match = PAUSE_TOKEN_RE.fullmatch(token.strip())
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


WORDISH_RE = re.compile(r"[\w]+", re.UNICODE)


def _comparison_norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.casefold()
    return re.sub(r"[^a-z0-9]+", "", value)


# ----------------------------
# Lesson parsing
# ----------------------------


def parse_raw_tokens(raw_speech: str) -> List[RawToken]:
    tokens = []
    for raw_index, token in enumerate(str(raw_speech or "").split()):
        pause = _is_pause_token(token)
        tokens.append(
            RawToken(
                raw_index=raw_index,
                text=token,
                is_pause=pause is not None,
                pause_seconds=float(pause or 0.0),
            )
        )
    return tokens


def build_segments_for_chapter(chapter_index: int, raw_speech: str) -> ChapterPlan:
    raw_tokens = parse_raw_tokens(raw_speech)
    segments: List[SegmentPlan] = []

    current_spoken_tokens: List[str] = []
    current_spoken_indices: List[int] = []
    pending_pause_indices: List[int] = []
    pending_pause_seconds = 0.0
    segment_index = 0

    def flush_segment() -> None:
        nonlocal current_spoken_tokens, current_spoken_indices
        nonlocal pending_pause_indices, pending_pause_seconds, segment_index

        if not current_spoken_indices:
            return

        segments.append(
            SegmentPlan(
                chapter_index=chapter_index,
                segment_index=segment_index,
                spoken_text=" ".join(current_spoken_tokens).strip(),
                raw_spoken_indices=list(current_spoken_indices),
                raw_pause_indices_after=list(pending_pause_indices),
                pause_after_sec=float(pending_pause_seconds),
            )
        )
        segment_index += 1
        current_spoken_tokens = []
        current_spoken_indices = []
        pending_pause_indices = []
        pending_pause_seconds = 0.0

    for token in raw_tokens:
        if token.is_pause:
            if current_spoken_indices:
                pending_pause_indices.append(token.raw_index)
                pending_pause_seconds += token.pause_seconds
                flush_segment()
            else:
                # Leading / consecutive pauses before any spoken word of a segment.
                # Keep them in a zero-word segment placeholder by attaching them to
                # the previous segment if one exists, otherwise preserve them later.
                if segments:
                    segments[-1].raw_pause_indices_after.append(token.raw_index)
                    segments[-1].pause_after_sec += token.pause_seconds
                else:
                    segments.append(
                        SegmentPlan(
                            chapter_index=chapter_index,
                            segment_index=segment_index,
                            spoken_text="",
                            raw_spoken_indices=[],
                            raw_pause_indices_after=[token.raw_index],
                            pause_after_sec=float(token.pause_seconds),
                        )
                    )
                    segment_index += 1
            continue

        current_spoken_tokens.append(token.text)
        current_spoken_indices.append(token.raw_index)

    if current_spoken_indices:
        flush_segment()

    return ChapterPlan(
        chapter_index=chapter_index,
        raw_speech=raw_speech,
        raw_tokens=raw_tokens,
        segments=segments,
    )


# ----------------------------
# edge-tts synthesis + timing
# ----------------------------


async def synthesize_segment_audio(
    segment: SegmentPlan,
    *,
    voice: str,
    rate: str,
    volume: str,
    pitch: str,
    audio_dir: Path,
) -> None:
    if not segment.spoken_text.strip():
        segment.boundaries = []
        segment.estimated_audio_duration_sec = 0.0
        return

    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"chapter_{segment.chapter_index:02d}_segment_{segment.segment_index:02d}.mp3"

    communicate = edge_tts.Communicate(
        segment.spoken_text,
        voice=voice,
        rate=rate,
        volume=volume,
        pitch=pitch,
        boundary="WordBoundary",
    )

    audio_bytes = bytearray()
    boundaries: List[WordBoundary] = []

    async for chunk in communicate.stream():
        chunk_type = chunk.get("type")
        if chunk_type == "audio":
            audio_bytes.extend(chunk["data"])
        elif chunk_type == "WordBoundary":
            boundaries.append(
                WordBoundary(
                    offset_sec=float(chunk["offset"]) / AUDIO_TICK_TO_SECONDS,
                    duration_sec=float(chunk["duration"]) / AUDIO_TICK_TO_SECONDS,
                    text=str(chunk.get("text", "") or ""),
                )
            )

    audio_path.write_bytes(bytes(audio_bytes))
    segment.audio_path = str(audio_path)
    segment.boundaries = boundaries

    if boundaries:
        segment.estimated_audio_duration_sec = max(
            0.0,
            boundaries[-1].offset_sec + boundaries[-1].duration_sec,
        )
    else:
        # Fallback for unusual metadata failure.
        word_count = max(1, len(segment.spoken_text.split()))
        segment.estimated_audio_duration_sec = word_count * 0.42


# ----------------------------
# Boundary -> raw token alignment
# ----------------------------


def _map_boundaries_to_whitespace_tokens(
    whitespace_tokens: List[str],
    boundaries: List[WordBoundary],
) -> List[Optional[float]]:
    """
    Map edge-tts word boundaries back onto the original whitespace-token sequence.

    This is needed because your lesson indices are based on `speech.split()`, while
    edge-tts boundary events are linguistic words and may split or normalize tokens.
    """
    if not whitespace_tokens:
        return []

    token_norms = [_comparison_norm(token) for token in whitespace_tokens]
    boundary_norms = [_comparison_norm(boundary.text) for boundary in boundaries]
    mapped: List[Optional[float]] = [None] * len(whitespace_tokens)

    boundary_i = 0
    for token_i, token_norm in enumerate(token_norms):
        if not token_norm:
            continue
        if boundary_i >= len(boundary_norms):
            break

        candidate_start = boundary_i
        combined = ""
        combined_end = candidate_start - 1

        while combined != token_norm and len(combined) < len(token_norm) and boundary_i < len(boundary_norms):
            part = boundary_norms[boundary_i]
            boundary_i += 1
            if not part:
                continue
            combined += part
            combined_end = boundary_i - 1

        if combined == token_norm and candidate_start <= combined_end:
            mapped[token_i] = boundaries[candidate_start].offset_sec
            continue

        # Fallback: if a direct concat match failed, try current boundary alone.
        boundary_i = max(candidate_start + 1, boundary_i)
        mapped[token_i] = boundaries[candidate_start].offset_sec if candidate_start < len(boundaries) else None

    # Interpolate gaps.
    known = [(i, t) for i, t in enumerate(mapped) if t is not None]
    if not known:
        if boundaries:
            total = max(0.0, boundaries[-1].offset_sec + boundaries[-1].duration_sec)
        else:
            total = max(0.3, len(whitespace_tokens) * 0.4)
        if len(whitespace_tokens) == 1:
            return [0.0]
        step = total / max(1, len(whitespace_tokens) - 1)
        return [i * step for i in range(len(whitespace_tokens))]

    first_idx, first_time = known[0]
    for i in range(0, first_idx):
        mapped[i] = max(0.0, first_time * (i / max(1, first_idx)))

    for (left_idx, left_time), (right_idx, right_time) in zip(known, known[1:]):
        gap = right_idx - left_idx
        if gap <= 1:
            continue
        step = (right_time - left_time) / gap
        for i in range(left_idx + 1, right_idx):
            mapped[i] = left_time + step * (i - left_idx)

    last_idx, last_time = known[-1]
    tail_count = len(mapped) - last_idx - 1
    if tail_count > 0:
        tail_step = 0.25
        for i in range(last_idx + 1, len(mapped)):
            mapped[i] = last_time + tail_step * (i - last_idx)

    return [0.0 if t is None else float(t) for t in mapped]


def finalize_chapter_timing(chapter: ChapterPlan) -> None:
    chapter_elapsed = 0.0
    raw_times: Dict[int, float] = {}

    for segment in chapter.segments:
        segment.chapter_time_offset_sec = chapter_elapsed

        if segment.raw_spoken_indices and segment.spoken_text.strip():
            spoken_tokens = segment.spoken_text.split()
            local_times = _map_boundaries_to_whitespace_tokens(spoken_tokens, segment.boundaries)

            for raw_idx, local_time in zip(segment.raw_spoken_indices, local_times):
                absolute_time = chapter_elapsed + float(local_time)
                raw_times[raw_idx] = absolute_time
                segment.raw_token_start_times_sec[raw_idx] = absolute_time

            segment_audio_end = chapter_elapsed + segment.estimated_audio_duration_sec
        else:
            segment_audio_end = chapter_elapsed

        # Pause tokens are part of the raw lesson index space, so they also get
        # a start time. The start time of a pause token is the exact moment the
        # silence begins.
        pause_cursor = segment_audio_end
        for pause_raw_idx in segment.raw_pause_indices_after:
            raw_times[pause_raw_idx] = pause_cursor
            segment.raw_token_start_times_sec[pause_raw_idx] = pause_cursor

            token = next((t for t in chapter.raw_tokens if t.raw_index == pause_raw_idx), None)
            if token is not None:
                pause_cursor += token.pause_seconds

        chapter_elapsed = segment_audio_end + segment.pause_after_sec

    chapter.raw_token_start_times_sec = raw_times
    chapter.duration_sec = chapter_elapsed


async def synthesize_lesson(
    lesson_payload: Dict[str, Any],
    *,
    voice: str,
    rate: str,
    volume: str,
    pitch: str,
    audio_dir: Path,
) -> Dict[int, ChapterPlan]:
    chapter_map: Dict[int, ChapterPlan] = {}

    for speech_row in lesson_payload.get("speech_map", []):
        if not isinstance(speech_row, dict):
            continue
        chapter_index = _safe_int(speech_row.get("chapter_index"), 0)
        raw_speech = str(speech_row.get("speech", "") or "")
        if chapter_index <= 0 or not raw_speech.strip():
            continue

        chapter = build_segments_for_chapter(chapter_index, raw_speech)
        chapter_map[chapter_index] = chapter

        for segment in chapter.segments:
            await synthesize_segment_audio(
                segment,
                voice=voice,
                rate=rate,
                volume=volume,
                pitch=pitch,
                audio_dir=audio_dir,
            )

        finalize_chapter_timing(chapter)

    return chapter_map


# ----------------------------
# Action packet building
# ----------------------------


def _derive_absolute_sync(event_row: Dict[str, Any], action_row: Dict[str, Any]) -> Tuple[int, int]:
    sync_absolute = action_row.get("sync_absolute")
    if isinstance(sync_absolute, dict):
        start_idx = _safe_int(sync_absolute.get("start_word_index"), _safe_int(event_row.get("start_word_index"), 0))
        end_idx = _safe_int(sync_absolute.get("end_word_index"), start_idx)
        if end_idx < start_idx:
            end_idx = start_idx
        return start_idx, end_idx

    event_start = _safe_int(event_row.get("start_word_index"), 0)
    event_end = _safe_int(event_row.get("end_word_index"), event_start)
    if event_end < event_start:
        event_end = event_start

    sync_local = action_row.get("sync_local")
    if isinstance(sync_local, dict):
        local_start = _safe_int(sync_local.get("start_word_offset", sync_local.get("start", 0)), 0)
        local_end = _safe_int(sync_local.get("end_word_offset", sync_local.get("end", local_start)), local_start)
        if local_end < local_start:
            local_end = local_start
        start_idx = event_start + local_start
        end_idx = event_start + local_end
        return start_idx, max(start_idx, end_idx)

    return event_start, event_end


def _lookup_emit_time(chapter: ChapterPlan, raw_word_index: int) -> float:
    if raw_word_index in chapter.raw_token_start_times_sec:
        return chapter.raw_token_start_times_sec[raw_word_index]

    if not chapter.raw_token_start_times_sec:
        return 0.0

    keys = sorted(chapter.raw_token_start_times_sec.keys())
    if raw_word_index <= keys[0]:
        return chapter.raw_token_start_times_sec[keys[0]]
    if raw_word_index >= keys[-1]:
        return chapter.raw_token_start_times_sec[keys[-1]]

    lower = max(k for k in keys if k < raw_word_index)
    upper = min(k for k in keys if k > raw_word_index)
    lower_t = chapter.raw_token_start_times_sec[lower]
    upper_t = chapter.raw_token_start_times_sec[upper]
    span = max(1, upper - lower)
    ratio = (raw_word_index - lower) / span
    return lower_t + ((upper_t - lower_t) * ratio)


def build_action_schedules(
    lesson_payload: Dict[str, Any],
    chapter_map: Dict[int, ChapterPlan],
) -> Dict[int, List[ActionPacketSchedule]]:
    per_chapter: Dict[int, List[ActionPacketSchedule]] = {chapter_idx: [] for chapter_idx in chapter_map}

    for event_row in lesson_payload.get("parsed_action_events", []):
        if not isinstance(event_row, dict):
            continue

        chapter_index = _safe_int(event_row.get("chapter_index"), 0)
        if chapter_index not in chapter_map:
            continue

        chapter = chapter_map[chapter_index]
        event_start_idx = _safe_int(event_row.get("start_word_index"), 0)
        event_end_idx = _safe_int(event_row.get("end_word_index"), event_start_idx)
        event_start_time = _lookup_emit_time(chapter, event_start_idx)
        event_end_time = _lookup_emit_time(chapter, event_end_idx)

        if (
            str(event_row.get("planner", "") or "") == "silence_static"
            and str(event_row.get("event_kind", "") or "") == "silence"
        ):
            deletion_payload: Dict[str, Any] = {
                "schema": "whiteboard_deletion_silence_v1",
                "chapter_index": chapter_index,
                "chunk_index": _safe_int(event_row.get("chunk_index"), 0),
                "segment_index": _safe_int(event_row.get("segment_index"), 0),
                "batch_index": _safe_int(event_row.get("batch_index"), -1),
                "event_index": _safe_int(event_row.get("event_index"), 0),
                "planner": str(event_row.get("planner", "") or ""),
                "event_kind": str(event_row.get("event_kind", "") or ""),
                "event_type": str(event_row.get("event_type", "") or ""),
                "name": str(event_row.get("name", "") or ""),
                "image_name": str(event_row.get("image_name", "") or ""),
                "processed_id": str(event_row.get("processed_id", "") or ""),
                "step_key": str(event_row.get("step_key", "") or ""),
                "event_start_word_index": event_start_idx,
                "event_end_word_index": event_end_idx,
                "event_duration_sec": _safe_float(event_row.get("duration_sec"), 0.0),
                "delete_targets_original": list(event_row.get("delete_targets_original") or []),
                "delete_targets_blocked_due_active_diagram": list(
                    event_row.get("delete_targets_blocked_due_active_diagram") or []
                ),
                "manual_shift_targets_due_active_diagram": list(
                    event_row.get("manual_shift_targets_due_active_diagram") or []
                ),
                "repeat_occurrence_name": str(event_row.get("repeat_occurrence_name", "") or ""),
                "emit_phase": "start",
                "emit_word_index": event_start_idx,
            }
            per_chapter[chapter_index].append(
                ActionPacketSchedule(
                    emit_time_sec=event_start_time,
                    emit_word_index=event_start_idx,
                    emit_phase="start",
                    payload=deletion_payload,
                )
            )
            continue

        parsed_actions = event_row.get("parsed_actions") or []
        if not isinstance(parsed_actions, list):
            continue

        for local_action_index, action_row in enumerate(parsed_actions):
            if not isinstance(action_row, dict):
                continue

            start_idx, end_idx = _derive_absolute_sync(event_row, action_row)
            start_time = _lookup_emit_time(chapter, start_idx)
            end_time = _lookup_emit_time(chapter, end_idx)

            base_payload: Dict[str, Any] = {
                "schema": "whiteboard_action_emit_v1",
                "chapter_index": chapter_index,
                "chunk_index": _safe_int(event_row.get("chunk_index"), 0),
                "segment_index": _safe_int(event_row.get("segment_index"), 0),
                "batch_index": _safe_int(event_row.get("batch_index"), -1),
                "event_index": _safe_int(event_row.get("event_index"), 0),
                "planner": str(event_row.get("planner", "") or ""),
                "event_kind": str(event_row.get("event_kind", "") or ""),
                "event_type": str(event_row.get("event_type", "") or ""),
                "name": str(event_row.get("name", "") or ""),
                "image_name": str(event_row.get("image_name", "") or ""),
                "processed_id": str(event_row.get("processed_id", "") or ""),
                "step_key": str(event_row.get("step_key", "") or ""),
                "event_start_word_index": event_start_idx,
                "event_end_word_index": event_end_idx,
                "event_duration_sec": _safe_float(event_row.get("duration_sec"), 0.0),
                "delete_targets_original": list(event_row.get("delete_targets_original") or []),
                "delete_targets_blocked_due_active_diagram": list(
                    event_row.get("delete_targets_blocked_due_active_diagram") or []
                ),
                "manual_shift_targets_due_active_diagram": list(
                    event_row.get("manual_shift_targets_due_active_diagram") or []
                ),
                "repeat_occurrence_name": str(event_row.get("repeat_occurrence_name", "") or ""),
                "action": dict(action_row),
                "global_action_index": _safe_int(
                    action_row.get("global_action_index"),
                    local_action_index,
                ),
                "action_index_in_event": _safe_int(action_row.get("action_index_in_event"), local_action_index),
                "sync_absolute": {
                    "start_word_index": start_idx,
                    "end_word_index": end_idx,
                },
                "sync_local": dict(action_row.get("sync_local") or {}),
            }

            start_payload = dict(base_payload)
            start_payload["emit_phase"] = "start"
            start_payload["emit_word_index"] = start_idx

            end_payload = dict(base_payload)
            end_payload["emit_phase"] = "end"
            end_payload["emit_word_index"] = end_idx

            per_chapter[chapter_index].append(
                ActionPacketSchedule(
                    emit_time_sec=start_time,
                    emit_word_index=start_idx,
                    emit_phase="start",
                    payload=start_payload,
                )
            )
            per_chapter[chapter_index].append(
                ActionPacketSchedule(
                    emit_time_sec=end_time,
                    emit_word_index=end_idx,
                    emit_phase="end",
                    payload=end_payload,
                )
            )

    for chapter_index, packets in per_chapter.items():
        packets.sort(
            key=lambda packet: (
                packet.emit_time_sec,
                packet.emit_word_index,
                0 if packet.emit_phase == "start" else 1,
                _safe_int(packet.payload.get("global_action_index"), 0),
            )
        )

    return per_chapter


# ----------------------------
# WebSocket broadcaster
# ----------------------------


class Broadcaster:
    def __init__(self) -> None:
        self.clients: set[WebSocketServerProtocol] = set()
        self._has_client_event = asyncio.Event()

    async def register(self, websocket: WebSocketServerProtocol) -> None:
        self.clients.add(websocket)
        self._has_client_event.set()

    async def unregister(self, websocket: WebSocketServerProtocol) -> None:
        self.clients.discard(websocket)
        if not self.clients:
            self._has_client_event.clear()

    async def wait_for_client(self) -> None:
        await self._has_client_event.wait()

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        if not self.clients:
            return
        message = json.dumps(payload, ensure_ascii=False)
        dead: List[WebSocketServerProtocol] = []
        for client in list(self.clients):
            try:
                await client.send(message)
            except Exception:
                dead.append(client)
        for client in dead:
            self.clients.discard(client)
        if not self.clients:
            self._has_client_event.clear()


async def websocket_handler(websocket: WebSocketServerProtocol, broadcaster: Broadcaster, runtime: LessonRuntimeData) -> None:
    await broadcaster.register(websocket)
    try:
        await websocket.send(
            json.dumps(
                {
                    "type": "hello",
                    "schema": "lesson_stream_socket_v1",
                    "topic": runtime.topic,
                    "chapters": sorted(runtime.chapter_map.keys()),
                },
                ensure_ascii=False,
            )
        )
        async for message in websocket:
            # The frontend may later want to send control messages. Right now
            # the server simply acknowledges them.
            try:
                data = json.loads(message)
            except Exception:
                data = {"raw": message}
            await websocket.send(
                json.dumps(
                    {
                        "type": "ack",
                        "received": data,
                    },
                    ensure_ascii=False,
                )
            )
    finally:
        await broadcaster.unregister(websocket)


# ----------------------------
# Playback runtime
# ----------------------------


def init_audio() -> None:
    if pygame.mixer.get_init():
        return
    pygame.mixer.init(frequency=DEFAULT_MIXER_FREQUENCY)


async def _wait_music_finish() -> None:
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(POLL_INTERVAL_SEC)


async def _emit_due_packets(
    broadcaster: Broadcaster,
    chapter_packets: List[ActionPacketSchedule],
    next_packet_index: int,
    chapter_elapsed_sec: float,
) -> int:
    while next_packet_index < len(chapter_packets):
        packet = chapter_packets[next_packet_index]
        if packet.emit_time_sec > chapter_elapsed_sec + 1e-3:
            break
        outgoing = dict(packet.payload)
        outgoing["type"] = (
            "deletion_silence"
            if outgoing.get("schema") == "whiteboard_deletion_silence_v1"
            else "action_packet"
        )
        outgoing["emitted_monotonic_sec"] = time.monotonic()
        outgoing["scheduled_emit_time_sec"] = packet.emit_time_sec
        await broadcaster.broadcast(outgoing)
        next_packet_index += 1
    return next_packet_index


async def play_chapter(
    chapter: ChapterPlan,
    chapter_packets: List[ActionPacketSchedule],
    broadcaster: Broadcaster,
) -> None:
    next_packet_index = 0

    await broadcaster.broadcast(
        {
            "type": "chapter_started",
            "schema": "lesson_stream_socket_v1",
            "chapter_index": chapter.chapter_index,
            "chapter_duration_sec": chapter.duration_sec,
        }
    )

    # Emit any packets at time zero before audio begins.
    next_packet_index = await _emit_due_packets(
        broadcaster,
        chapter_packets,
        next_packet_index,
        0.0,
    )

    for segment in chapter.segments:
        # Zero-word segment: pure silence.
        if not segment.raw_spoken_indices or not segment.spoken_text.strip() or not segment.audio_path:
            if segment.pause_after_sec > 0:
                pause_start = time.monotonic()
                while True:
                    elapsed = time.monotonic() - pause_start
                    chapter_elapsed = segment.chapter_time_offset_sec + elapsed
                    next_packet_index = await _emit_due_packets(
                        broadcaster,
                        chapter_packets,
                        next_packet_index,
                        chapter_elapsed,
                    )
                    if elapsed >= segment.pause_after_sec:
                        break
                    await asyncio.sleep(POLL_INTERVAL_SEC)
            continue

        pygame.mixer.music.load(segment.audio_path)
        pygame.mixer.music.play()

        # Poll playback position and emit due packets against the actual music clock.
        while True:
            pos_ms = pygame.mixer.music.get_pos()
            if pos_ms < 0 and not pygame.mixer.music.get_busy():
                break
            pos_sec = max(0.0, pos_ms / 1000.0)
            chapter_elapsed = segment.chapter_time_offset_sec + pos_sec
            next_packet_index = await _emit_due_packets(
                broadcaster,
                chapter_packets,
                next_packet_index,
                chapter_elapsed,
            )
            if not pygame.mixer.music.get_busy() and pos_ms < 0:
                break
            await asyncio.sleep(POLL_INTERVAL_SEC)

        # Make sure we flush anything that lands exactly at the segment tail.
        next_packet_index = await _emit_due_packets(
            broadcaster,
            chapter_packets,
            next_packet_index,
            segment.chapter_time_offset_sec + segment.estimated_audio_duration_sec,
        )

        if segment.pause_after_sec > 0:
            pause_start = time.monotonic()
            while True:
                elapsed = time.monotonic() - pause_start
                chapter_elapsed = segment.chapter_time_offset_sec + segment.estimated_audio_duration_sec + elapsed
                next_packet_index = await _emit_due_packets(
                    broadcaster,
                    chapter_packets,
                    next_packet_index,
                    chapter_elapsed,
                )
                if elapsed >= segment.pause_after_sec:
                    break
                await asyncio.sleep(POLL_INTERVAL_SEC)

    # Flush any leftovers.
    next_packet_index = await _emit_due_packets(
        broadcaster,
        chapter_packets,
        next_packet_index,
        chapter.duration_sec + 0.01,
    )

    await broadcaster.broadcast(
        {
            "type": "chapter_finished",
            "schema": "lesson_stream_socket_v1",
            "chapter_index": chapter.chapter_index,
        }
    )


async def play_lesson(
    runtime: LessonRuntimeData,
    broadcaster: Broadcaster,
    *,
    inter_chapter_pause_sec: float,
) -> None:
    ordered_chapters = sorted(runtime.chapter_map.keys())
    total_chapters = len(ordered_chapters)

    await broadcaster.broadcast(
        {
            "type": "lesson_started",
            "schema": "lesson_stream_socket_v1",
            "topic": runtime.topic,
            "chapter_count": total_chapters,
        }
    )

    for chapter_pos, chapter_index in enumerate(ordered_chapters):
        chapter = runtime.chapter_map[chapter_index]
        packets = runtime.chapter_packets.get(chapter_index, [])
        await play_chapter(chapter, packets, broadcaster)

        if chapter_pos < total_chapters - 1 and inter_chapter_pause_sec > 0:
            pause_start = time.monotonic()
            await broadcaster.broadcast(
                {
                    "type": "inter_chapter_pause_started",
                    "schema": "lesson_stream_socket_v1",
                    "chapter_index": chapter_index,
                    "seconds": inter_chapter_pause_sec,
                }
            )
            while (time.monotonic() - pause_start) < inter_chapter_pause_sec:
                await asyncio.sleep(POLL_INTERVAL_SEC)

    await broadcaster.broadcast(
        {
            "type": "lesson_finished",
            "schema": "lesson_stream_socket_v1",
            "topic": runtime.topic,
        }
    )


# ----------------------------
# IO / CLI
# ----------------------------


def load_lesson_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


async def build_runtime(
    lesson_payload: Dict[str, Any],
    *,
    voice: str,
    rate: str,
    volume: str,
    pitch: str,
    audio_dir: Path,
) -> LessonRuntimeData:
    chapter_map = await synthesize_lesson(
        lesson_payload,
        voice=voice,
        rate=rate,
        volume=volume,
        pitch=pitch,
        audio_dir=audio_dir,
    )
    chapter_packets = build_action_schedules(lesson_payload, chapter_map)
    return LessonRuntimeData(
        topic=str(lesson_payload.get("topic", "") or ""),
        chapter_map=chapter_map,
        chapter_packets=chapter_packets,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read lesson speech with TTS and emit synced action packets over WebSocket.",
    )
    parser.add_argument("--lesson-json", required=True, help="Path to lesson_ready_output_v1 JSON file.")
    parser.add_argument("--host", default="127.0.0.1", help="WebSocket bind host.")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket bind port.")
    parser.add_argument("--voice", default="en-US-AriaNeural", help="edge-tts voice name.")
    parser.add_argument("--rate", default="+0%", help="edge-tts rate.")
    parser.add_argument("--volume", default="+0%", help="edge-tts volume.")
    parser.add_argument("--pitch", default="+0Hz", help="edge-tts pitch.")
    parser.add_argument(
        "--inter-chapter-pause-sec",
        type=float,
        default=DEFAULT_INTER_CHAPTER_PAUSE_SEC,
        help="Pause inserted between chapters. Defaults to 3 seconds.",
    )
    parser.add_argument(
        "--audio-cache-dir",
        default="",
        help="Optional directory for generated MP3 files. A temp directory is used if omitted.",
    )
    parser.add_argument(
        "--wait-for-client",
        action="store_true",
        help="Do not start the lesson until at least one WebSocket client is connected.",
    )
    parser.add_argument(
        "--dump-schedule-json",
        default="",
        help="Optional path to dump the parsed timing schedule for debugging.",
    )
    return parser


def dump_schedule(runtime: LessonRuntimeData, out_path: Path) -> None:
    payload: Dict[str, Any] = {
        "topic": runtime.topic,
        "chapters": {},
    }
    for chapter_index, chapter in sorted(runtime.chapter_map.items()):
        payload["chapters"][chapter_index] = {
            "duration_sec": chapter.duration_sec,
            "raw_token_start_times_sec": chapter.raw_token_start_times_sec,
            "segments": [
                {
                    "chapter_index": segment.chapter_index,
                    "segment_index": segment.segment_index,
                    "spoken_text": segment.spoken_text,
                    "raw_spoken_indices": segment.raw_spoken_indices,
                    "raw_pause_indices_after": segment.raw_pause_indices_after,
                    "pause_after_sec": segment.pause_after_sec,
                    "audio_path": segment.audio_path,
                    "estimated_audio_duration_sec": segment.estimated_audio_duration_sec,
                    "chapter_time_offset_sec": segment.chapter_time_offset_sec,
                    "boundaries": [asdict(boundary) for boundary in segment.boundaries],
                }
                for segment in chapter.segments
            ],
            "packets": [
                {
                    "emit_time_sec": packet.emit_time_sec,
                    "emit_word_index": packet.emit_word_index,
                    "emit_phase": packet.emit_phase,
                    "payload": packet.payload,
                }
                for packet in runtime.chapter_packets.get(chapter_index, [])
            ],
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def async_main() -> None:
    args = build_arg_parser().parse_args()

    lesson_path = Path(args.lesson_json).expanduser().resolve()
    if not lesson_path.exists():
        raise SystemExit(f"Lesson JSON not found: {lesson_path}")

    lesson_payload = load_lesson_json(lesson_path)
    if str(lesson_payload.get("schema", "") or "") != "lesson_ready_output_v1":
        raise SystemExit(
            "The provided file does not look like lesson_ready_output_v1. "
            "This runner expects the same structure as your timeline builder output."
        )

    init_audio()

    if args.audio_cache_dir:
        audio_dir = Path(args.audio_cache_dir).expanduser().resolve()
        audio_dir.mkdir(parents=True, exist_ok=True)
    else:
        audio_dir = Path(tempfile.mkdtemp(prefix="lesson_tts_audio_"))

    runtime = await build_runtime(
        lesson_payload,
        voice=args.voice,
        rate=args.rate,
        volume=args.volume,
        pitch=args.pitch,
        audio_dir=audio_dir,
    )

    if args.dump_schedule_json:
        dump_schedule(runtime, Path(args.dump_schedule_json).expanduser().resolve())

    broadcaster = Broadcaster()

    async def handler(websocket: WebSocketServerProtocol) -> None:
        await websocket_handler(websocket, broadcaster, runtime)

    async with websockets.serve(handler, args.host, args.port):
        print(f"[lesson-streamer] WebSocket server listening on ws://{args.host}:{args.port}")
        print(f"[lesson-streamer] Topic: {runtime.topic}")
        print(f"[lesson-streamer] Chapters: {sorted(runtime.chapter_map.keys())}")
        print(f"[lesson-streamer] Audio cache: {audio_dir}")

        if args.wait_for_client:
            print("[lesson-streamer] Waiting for at least one client...")
            await broadcaster.wait_for_client()
            print("[lesson-streamer] Client connected. Starting lesson.")

        await play_lesson(
            runtime,
            broadcaster,
            inter_chapter_pause_sec=float(args.inter_chapter_pause_sec),
        )

        # Keep the server alive briefly after finishing so a frontend can still
        # receive the final packet if needed.
        await asyncio.sleep(0.25)


def main() -> None:
    try:
        asyncio.run(async_main())
    finally:
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        try:
            pygame.mixer.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
