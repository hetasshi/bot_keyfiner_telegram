"""Text messages and formatting helpers for the bot."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..audio_processing.analyzer import AnalysisResult


def start_message(limit_mb: int) -> str:
    return (
        "Привет! Я помогу определить тональность и темп бита.\n"
        "Отправьте аудио-файл (mp3, wav, m4a, ogg, opus) размером до "
        f"{limit_mb} МБ, и я пришлю тональность, темп и длительность трека."
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


def format_analysis_result(result: "AnalysisResult") -> str:
    """Format analysis result into a user-friendly text message."""

    lines = [f"Тональность бита: {result.filename}"]
    lines.append(f"Тон: {result.tone_display}")
    lines.append(
        "Темп: {bpm} BPM ({double} / {half})".format(
            bpm=result.bpm,
            double=result.bpm_double,
            half=result.bpm_half,
        )
    )
    lines.append(f"Длительность: {result.duration}")

    if result.close_key:
        lines.append(f"Близкий вариант: {result.close_key}")

    return "\n".join(lines)


__all__ = [
    "start_message",
    "file_too_large",
    "unsupported_file",
    "processing_error",
    "format_analysis_result",
]
