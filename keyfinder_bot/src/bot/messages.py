"""Text messages and formatting helpers used by the bot."""

from __future__ import annotations

from typing import Any

SUPPORTED_FORMATS = "mp3, wav, m4a, ogg/opus"


def start_message(max_file_mb: int) -> str:
    """Return the greeting shown on /start."""

    return (
        "Привет! Я помогу определить тональность, темп и длительность бита.\n"
        f"Пришлите аудио-файл ({SUPPORTED_FORMATS}) до {max_file_mb} MB, и я верну результат."
    )


def file_too_large(max_file_mb: int) -> str:
    return (
        "Файл слишком большой. Максимальный размер — "
        f"{max_file_mb} MB. Попробуйте отправить более лёгкую версию."
    )


def unsupported_format() -> str:
    return (
        "Неподдерживаемый формат файла. Допустимы: "
        f"{SUPPORTED_FORMATS}."
    )


def processing_error() -> str:
    return (
        "Не удалось проанализировать аудио. Попробуйте другой файл или повторите попытку позже."
    )


def format_key(note: str, mode: str, enharmonic: str | None) -> str:
    """Format a note/mode pair for output.

    >>> format_key('F#', 'min', 'Gb')
    'F#min (Gbmin)'
    >>> format_key('C', 'Maj', None)
    'CMaj'
    """

    base = f"{note}{mode}"
    if enharmonic:
        return f"{base} ({enharmonic}{mode})"
    return base


def format_analysis_result(result: dict[str, Any], *, include_close: bool = False) -> str:
    tone_str = format_key(result["note"], result["mode"], result.get("enharmonic"))

    lines = [
        f"Тональность бита: {result.get('filename', 'audio')}",
        f"Тон: {tone_str}",
    ]

    if include_close and result.get("close_key"):
        close = result["close_key"]
        close_str = format_key(close["note"], close["mode"], close.get("enharmonic"))
        lines.append(f"Близкий вариант: {close_str}")

    lines.append(
        "Темп: {bpm} BPM ({double_bpm} / {half_bpm})".format(
            bpm=result["bpm"],
            double_bpm=result["bpm_double"],
            half_bpm=result["bpm_half"],
        )
    )
    lines.append(f"Длительность: {result['duration']}")

    return "\n".join(lines)
