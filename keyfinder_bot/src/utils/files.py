"""Utility helpers for temporary files and audio validation."""
from __future__ import annotations

import logging
import mimetypes
import tempfile
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".ogg", ".opus", ".m4a"}
SUPPORTED_MIME_TYPES = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/x-mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/ogg": ".ogg",
    "audio/vorbis": ".ogg",
    "audio/opus": ".opus",
    "audio/webm": ".webm",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".m4a",
    "video/mp4": ".m4a",
}

TEMP_DIR_NAME = "keyfinder_bot_tmp"
_TEMP_DIR = Path(tempfile.gettempdir()) / TEMP_DIR_NAME


def ensure_temp_dir() -> Path:
    """Ensure that the temporary directory exists and return it."""

    _TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return _TEMP_DIR


def unique_stem() -> str:
    """Generate a unique file stem."""

    return uuid.uuid4().hex


def _normalize_extension(extension: str | None) -> str | None:
    if not extension:
        return None
    extension = extension.lower()
    if not extension.startswith('.'):
        extension = f'.{extension}'
    return extension


def determine_extension(filename: str | None, mime_type: str | None) -> Optional[str]:
    """Determine a valid extension from file name or mime type."""

    if filename:
        ext = _normalize_extension(Path(filename).suffix)
        if ext in SUPPORTED_EXTENSIONS:
            return ext
    if mime_type:
        normalized_mime = mime_type.lower()
        direct = SUPPORTED_MIME_TYPES.get(normalized_mime)
        if direct:
            ext = _normalize_extension(direct)
            if ext in SUPPORTED_EXTENSIONS:
                return ext
        guessed = mimetypes.guess_extension(normalized_mime)
        guessed = _normalize_extension(guessed)
        if guessed in SUPPORTED_EXTENSIONS:
            return guessed
    return None


def is_supported_audio(filename: str | None, mime_type: str | None) -> bool:
    """Check whether a file can be processed by the analyzer."""

    return determine_extension(filename, mime_type) in SUPPORTED_EXTENSIONS


def safe_join_temp(extension: str) -> Path:
    """Build a temporary file path using the given extension."""

    ensure_temp_dir()
    ext = _normalize_extension(extension) or ""
    if ext and ext not in SUPPORTED_EXTENSIONS:
        logger.debug("Extension %s not explicitly supported; still creating file.", ext)
    return _TEMP_DIR / f"{unique_stem()}{ext}"


def remove_silent(path: str | Path) -> None:
    """Remove a file ignoring errors."""

    try:
        Path(path).unlink(missing_ok=True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to remove temporary file %s: %s", path, exc)
