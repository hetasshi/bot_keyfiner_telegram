"""Text templates and formatting helpers for Telegram responses."""

from __future__ import annotations

from typing import Mapping, Optional

START_MESSAGE = (
    "👋 Привет! Пришли мне аудио (mp3, wav, m4a, ogg/opus) до 70 МБ, "
    "и я определю тональность, темп и длительность трека."
)
FILE_TOO_LARGE_MESSAGE = "Файл слишком большой. Максимальный размер — 70 МБ."
UNSUPPORTED_FORMAT_MESSAGE = (
    "Этот тип файла не поддерживается. Отправьте mp3, wav, m4a или ogg/opus."
)
ANALYSIS_ERROR_MESSAGE = (
    "Не удалось проанализировать аудио. Попробуйте другой файл или повторите позже."
)


def format_key(note: str, mode: str, enharmonic: Optional[str]) -> str:
    """Format musical key string.

    >>> format_key("F#", "min", "Gb")
    'F#min (Gbmin)'
    >>> format_key("C", "Maj", None)
    'CMaj'
    """

    main = f"{note}{mode}"
    if enharmonic:
        return f"{main} ({enharmonic}{mode})"
    return main


def format_analysis_result(result: Mapping[str, object]) -> str:
    """Build the final message with the analysis results."""

    tone_line = format_key(
        str(result.get("note")),
        str(result.get("mode")),
        result.get("enharmonic") or None,
    )
    bpm = int(result.get("bpm", 0))
    bpm_double = int(result.get("bpm_double", bpm * 2))
    bpm_half = int(result.get("bpm_half", max(bpm // 2, 1)))
    duration = str(result.get("duration", "00:00"))
    filename = str(result.get("filename", "audio"))

    lines = [
        f"Тональность бита: {filename}",
        f"Тон: {tone_line}",
        f"Темп: {bpm} BPM ({bpm_double} / {bpm_half})",
        f"Длительность: {duration}",
    ]

    close_key = result.get("close_key")
    if isinstance(close_key, Mapping):
        close_line = format_key(
            str(close_key.get("note")),
            str(close_key.get("mode")),
            close_key.get("enharmonic") or None,
        )
        lines.append(f"Близкий вариант: {close_line}")

    return "\n".join(lines)


__all__ = [
    "START_MESSAGE",
    "FILE_TOO_LARGE_MESSAGE",
    "UNSUPPORTED_FORMAT_MESSAGE",
    "ANALYSIS_ERROR_MESSAGE",
    "format_key",
    "format_analysis_result",
]
