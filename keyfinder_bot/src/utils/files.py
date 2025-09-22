"""Utilities for working with temporary audio files."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

TEMP_DIR = Path(tempfile.gettempdir()) / "keyfinder-bot"
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".opus"}
MIME_EXTENSION_MAP = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/vnd.wave": ".wav",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/webm": ".ogg",
    "audio/mp4": ".m4a",
    "audio/aac": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/3gpp": ".m4a",
}


def ensure_temp_dir() -> Path:
    """Ensure that the temporary directory exists and return it."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_DIR


def unique_stem() -> str:
    """Generate a random filename stem."""
    return uuid.uuid4().hex


def safe_join_temp(ext: str) -> Path:
    """Return a unique path inside the temporary directory with the given extension."""
    ensure_temp_dir()
    clean_ext = ext if ext.startswith(".") else f".{ext}"
    return TEMP_DIR / f"{unique_stem()}{clean_ext.lower()}"


def remove_silent(path: Path | str) -> None:
    """Remove file if it exists, ignoring any errors."""
    try:
        Path(path).unlink(missing_ok=True)
    except Exception as exc:  # pragma: no cover - best effort cleanup
        logger.debug("Failed to remove temporary file %s: %s", path, exc)


def extract_extension(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    suffix = Path(filename).suffix.lower()
    return suffix or None


def resolve_extension(filename: Optional[str], mime_type: Optional[str]) -> Optional[str]:
    ext = extract_extension(filename)
    if ext and ext in SUPPORTED_EXTENSIONS:
        return ext
    if mime_type:
        mime_ext = MIME_EXTENSION_MAP.get(mime_type.lower())
        if mime_ext in SUPPORTED_EXTENSIONS:
            return mime_ext
    if ext in SUPPORTED_EXTENSIONS:
        return ext
    return None


def is_supported(filename: Optional[str], mime_type: Optional[str]) -> bool:
    return resolve_extension(filename, mime_type) in SUPPORTED_EXTENSIONS


def pick_filename(original_name: Optional[str], default_ext: str) -> str:
    """Return a safe filename for displaying to the user."""
    if original_name:
        name = os.path.basename(original_name)
        if name:
            return name
    clean_ext = default_ext if default_ext.startswith(".") else f".{default_ext}"
    return f"audio{clean_ext}"


__all__ = [
    "ensure_temp_dir",
    "unique_stem",
    "safe_join_temp",
    "remove_silent",
    "resolve_extension",
    "is_supported",
    "pick_filename",
]
