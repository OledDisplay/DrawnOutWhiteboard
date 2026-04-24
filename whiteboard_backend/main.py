from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from NEWtimeline import NewTimelineFirstModule, normalize_ws


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NEWtimeline.py end-to-end for one topic.")
    parser.add_argument("topic", help="Lesson topic / prompt to run through the pipeline.")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable NEWtimeline GPT cache for OpenAI stages. Qwen/internal stage caches are managed by NEWtimeline.",
    )
    parser.add_argument(
        "--out",
        default="PipelineOutputs/newtimeline_run_result.json",
        help="Path to write a JSON-safe copy of the pipeline result.",
    )
    parser.add_argument(
        "--debug-out-dir",
        default="PipelineOutputs",
        help="NEWtimeline debug output directory.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    topic = normalize_ws(args.topic)
    if not topic:
        raise SystemExit("topic cannot be empty")

    runner = NewTimelineFirstModule(
        debug_out_dir=str(args.debug_out_dir),
        gpt_cache=bool(args.cache),
        debug_print=True,
    )
    result = runner.run_full_timeline(topic)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "newtimeline_main_run_v1",
        "topic": topic,
        "cache_enabled": bool(args.cache),
        "written_at_unix": time.time(),
        "result": _json_safe(result),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[main] NEWtimeline complete. Result written to {out_path}")


if __name__ == "__main__":
    main()
