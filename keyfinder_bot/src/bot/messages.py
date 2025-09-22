"""User-facing message templates."""
from __future__ import annotations

from typing import Any

START_MESSAGE = (
    "Привет! Отправьте аудио-файл (mp3, wav, ogg/opus, m4a) размером до 70 МБ, "
    "и я определю тональность, темп и длительность бита."
)

FILE_TOO_LARGE_MESSAGE = "Файл слишком большой. Максимальный размер: {limit_mb} МБ."
UNSUPPORTED_FORMAT_MESSAGE = (
    "Неподдерживаемый формат. Допустимы: mp3, wav, ogg/opus, m4a."
)
DOWNLOAD_FAILED_MESSAGE = "Не удалось скачать файл. Попробуйте ещё раз позже."
ANALYSIS_FAILED_MESSAGE = (
    "Не получилось проанализировать аудио. Убедитесь, что это поддерживаемый формат и попробуйте снова."
)


def format_key_string(note: str, mode: str, enharmonic: str | None) -> str:
    """Format key string with optional enharmonic alternative.

    >>> format_key_string('A#', 'min', 'Bb')
    'A#min (Bbmin)'
    >>> format_key_string('C', 'Maj', None)
    'CMaj'
    """

    base = f"{note}{mode}"
    if enharmonic:
        return f"{base} ({enharmonic}{mode})"
    return base


def format_analysis_message(data: dict[str, Any]) -> str:
    """Build the final analysis message following the required specification."""

    tone_line = format_key_string(data["note"], data["mode"], data.get("enharmonic"))
    lines = [
        f"Тональность бита: {data['filename']}",
        f"Тон: {tone_line}",
        f"Темп: {data['bpm']} BPM ({data['bpm2x']} / {data['bpm1_2']})",
        f"Длительность: {data['duration']}",
    ]

    close_key = data.get("close_key")
    if close_key:
        close_line = format_key_string(
            close_key["note"],
            close_key["mode"],
            close_key.get("enharmonic"),
        )
        lines.append(f"Близкий вариант: {close_line}")

    return "\n".join(lines)


def file_too_large(limit_mb: int) -> str:
    return FILE_TOO_LARGE_MESSAGE.format(limit_mb=limit_mb)


def unsupported_format() -> str:
    return UNSUPPORTED_FORMAT_MESSAGE


def analysis_failed() -> str:
    return ANALYSIS_FAILED_MESSAGE


def download_failed() -> str:
    return DOWNLOAD_FAILED_MESSAGE
