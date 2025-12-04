# whiteboard_api/utils.py
import json
from pathlib import Path


def resolve_backend_root(max_up: int = 10) -> Path:
    """
    Walk up from this file's directory until we find 'whiteboard_backend'.
    Same idea as your Flutter _resolveBackendSubdir.
    """
    dir_path = Path(__file__).resolve().parent
    for _ in range(max_up):
        candidate = dir_path / "whiteboard_backend"
        if candidate.is_dir():
            return candidate
        parent = dir_path.parent
        if parent == dir_path:
            break
        dir_path = parent
    raise RuntimeError("whiteboard_backend folder not found while resolving paths")


BACKEND_ROOT = resolve_backend_root()
STROKE_VECTORS_DIR = BACKEND_ROOT / "StrokeVectors"
FONT_DIR = BACKEND_ROOT / "Font"
FONT_METRICS_PATH = FONT_DIR / "font_metrics.json"


def validate_image_json(file_name: str) -> None:
    """
    Ensure the JSON exists in StrokeVectors and has a 'strokes' list.
    Does NOT modify anything, only validates.
    """
    path = STROKE_VECTORS_DIR / file_name
    if not path.is_file():
        raise FileNotFoundError(str(path))

    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)

    if not isinstance(data, dict) or "strokes" not in data:
        raise ValueError(f"{file_name} is missing 'strokes' key")

    if not isinstance(data["strokes"], list):
        raise ValueError(f"{file_name} 'strokes' must be a list")


def validate_text_prompt(prompt: str) -> None:
    """
    Make sure at least one non-space character in the prompt
    has a glyph JSON in the Font folder.
    This mirrors _getGlyphForCode's file naming: <hex>.json.
    """
    for ch in prompt:
        if ch == " ":
            continue
        code = ord(ch)
        hexname = f"{code:04x}.json"
        path = FONT_DIR / hexname
        if path.is_file():
            return
    raise ValueError("No glyph JSONs found for any non-space character in prompt")
