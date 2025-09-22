"""Utility helpers for working with temporary audio files."""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS: Final[set[str]] = {".mp3", ".wav", ".m4a", ".ogg", ".opus"}
MIME_EXTENSION_MAP: Final[dict[str, str]] = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/vnd.wave": ".wav",
    "audio/mp4": ".m4a",
    "audio/aac": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aacp": ".m4a",
    "audio/ogg": ".ogg",
    "application/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/webm": ".ogg",
}

TEMP_DIR_NAME: Final[str] = "keyfinder_bot"


def ensure_temp_dir() -> Path:
    """Ensure that the temporary directory exists and return it."""

    tmp_dir = Path(tempfile.gettempdir()) / TEMP_DIR_NAME
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def unique_stem() -> str:
    """Generate a unique identifier suitable for file stems."""

    return uuid.uuid4().hex


def safe_join_temp(extension: str | None) -> Path:
    """Build a unique temporary file path with the provided extension."""

    temp_dir = ensure_temp_dir()
    suffix = (extension or "").lower()
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"
    return temp_dir / f"{unique_stem()}{suffix}"


def remove_silent(path: str | Path) -> None:
    """Remove a temporary file, ignoring missing-file errors."""

    try:
        Path(path).unlink(missing_ok=True)
    except Exception:  # pragma: no cover - best-effort cleanup
        logger.warning("Failed to remove temporary file %s", path, exc_info=True)


def get_extension(file_name: str | None, mime_type: str | None) -> str | None:
    """Determine a supported file extension from the provided metadata."""

    if file_name:
        suffix = Path(file_name).suffix.lower()
        if suffix in ALLOWED_EXTENSIONS:
            return suffix

    if mime_type:
        normalized = mime_type.split(";")[0].strip().lower()
        if normalized in MIME_EXTENSION_MAP:
            return MIME_EXTENSION_MAP[normalized]

    return None


def is_supported_audio(file_name: str | None, mime_type: str | None) -> bool:
    """Return True if the provided metadata matches supported formats."""

    return get_extension(file_name, mime_type) is not None


def display_name(file_name: str | None, extension: str | None, *, fallback: str = "audio") -> str:
    """Return a human-friendly file name for messages."""

    if file_name:
        name = Path(file_name).name
        if name.strip():
            return name

    suffix = (extension or "").lower()
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"
    return f"{fallback}{suffix}"
