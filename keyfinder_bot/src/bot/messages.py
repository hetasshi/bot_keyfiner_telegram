"""Text messages and formatting helpers for the bot."""
from __future__ import annotations

from typing import Any, Dict

START_MESSAGE = (
    "Привет! Я помогу определить тональность и темп бита.\n"
    "Отправьте аудио-файл (mp3, wav, m4a, ogg, opus) размером до 70 МБ,"
    " и я пришлю тональность, темп и длительность трека."
)


def file_too_large(limit_mb: int) -> str:
    return (
        "Файл слишком большой. Максимально допустимый размер — "
        f"{limit_mb} МБ. Попробуйте отправить более лёгкий файл."
    )


def unsupported_file() -> str:
    return (
        "Не удалось определить тип файла. Поддерживаются форматы:"
        " mp3, wav, m4a, ogg, opus."
    )


def processing_error() -> str:
    return (
        "Не получилось обработать файл. Убедитесь, что это поддерживаемый"
        " аудио-формат, и попробуйте снова позже."
    )


def format_analysis_result(result: Dict[str, Any]) -> str:
    """Format analysis result into a user-friendly text message.

    >>> sample = {
    ...     "filename": "demo.wav",
    ...     "tone": "A#min (Bbmin)",
    ...     "bpm": 140,
    ...     "bpm_double": 280,
    ...     "bpm_half": 70,
    ...     "duration": "03:30",
    ...     "close_key": None,
    ... }
    >>> "A#min" in format_analysis_result(sample)
    True
    """

    lines = [f"Тональность бита: {result.get('filename', 'audio')}"]
    lines.append(f"Тон: {result['tone']}")
    lines.append(
        "Темп: {bpm} BPM ({double} / {half})".format(
            bpm=result['bpm'],
            double=result['bpm_double'],
            half=result['bpm_half'],
        )
    )
    lines.append(f"Длительность: {result['duration']}")

    close_key = result.get("close_key")
    if close_key:
        lines.append(f"Близкий вариант: {close_key}")

    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual smoke-test helper
    print(
        format_analysis_result(
            {
                "filename": "track.mp3",
                "tone": "F#Maj (GbMaj)",
                "bpm": 128,
                "bpm_double": 256,
                "bpm_half": 64,
                "duration": "02:45",
                "close_key": None,
            }
        )
    )
