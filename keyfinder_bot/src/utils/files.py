"""File management helpers for temporary audio storage."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional
from uuid import uuid4

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".opus"}
ALLOWED_MIME_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/x-wav",
    "audio/wav",
    "audio/mp4",
    "audio/aac",
    "audio/ogg",
    "audio/opus",
    "audio/webm",
}

_TEMP_DIR: Optional[Path] = None


def ensure_temp_dir(base: str | Path | None = None) -> Path:
    """Create and return the temporary directory path."""

    global _TEMP_DIR
    base_path = Path(base) if base is not None else Path.cwd() / "temp"
    base_path.mkdir(parents=True, exist_ok=True)
    _TEMP_DIR = base_path
    return base_path


def unique_stem() -> str:
    """Return a unique file stem using UUID4."""

    return uuid4().hex


def safe_join_temp(extension: str) -> Path:
    """Return a path inside the temp directory with a random filename."""

    if not extension.startswith("."):
        extension = f".{extension}"
    temp_dir = _TEMP_DIR or ensure_temp_dir()
    filename = f"{unique_stem()}{extension.lower()}"
    return temp_dir / filename


def remove_silent(path: str | Path) -> None:
    """Remove file ignoring missing file errors."""

    path_obj = Path(path)
    try:
        path_obj.unlink()
    except FileNotFoundError:
        pass


def detect_extension(filename: str | None, mime_type: str | None = None) -> Optional[str]:
    """Return a lowercase extension (with dot) inferred from name or mime type."""

    if filename:
        suffix = Path(filename).suffix
        if suffix:
            return suffix.lower()
    if mime_type:
        guessed = mimetypes.guess_extension(mime_type, strict=False)
        if guessed:
            return guessed.lower()
    return None


def is_supported_audio(filename: str | None, mime_type: str | None = None) -> bool:
    """Return True if filename or mime type indicates supported audio."""

    extension = detect_extension(filename, mime_type)

    if extension and extension in ALLOWED_EXTENSIONS:
        return True

    if mime_type and mime_type.lower() in ALLOWED_MIME_TYPES:
        return True

    if mime_type and mime_type.startswith("audio/"):
        return True

    return False


__all__ = [
    "ALLOWED_EXTENSIONS",
    "detect_extension",
    "ensure_temp_dir",
    "is_supported_audio",
    "remove_silent",
    "safe_join_temp",
    "unique_stem",
]
