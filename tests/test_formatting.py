from src.audio_processing.analyzer import (
    AnalysisResult,
    format_duration_seconds,
    format_tone_display,
)
from src.bot import messages


def test_enharmonic_pairs() -> None:
    assert format_tone_display("F#", "Maj") == "F#Maj (GbMaj)"
    assert format_tone_display("A#", "min") == "A#min (Bbmin)"


def test_duration_formatting() -> None:
    assert format_duration_seconds(5) == "00:05"
    assert format_duration_seconds(187) == "03:07"


def test_message_template() -> None:
    result = AnalysisResult(
        filename="track.mp3",
        note="A#",
        mode="min",
        enharmonic="Bb",
        bpm=140,
        bpm_double=280,
        bpm_half=70,
        duration="03:25",
        close_key=None,
    )

    formatted = messages.format_analysis_result(result)
    lines = formatted.splitlines()

    assert len(lines) == 4
    assert lines[0] == "Тональность бита: track.mp3"
    assert lines[1] == "Тон: A#min (Bbmin)"
    assert lines[2] == "Темп: 140 BPM (280 / 70)"
    assert lines[3] == "Длительность: 03:25"


def test_message_template_with_close_key() -> None:
    result = AnalysisResult(
        filename="beat.wav",
        note="C",
        mode="Maj",
        enharmonic=None,
        bpm=128,
        bpm_double=256,
        bpm_half=64,
        duration="02:30",
        close_key="G#Maj (AbMaj)",
    )

    formatted = messages.format_analysis_result(result)
    lines = formatted.splitlines()

    assert len(lines) == 5
    assert lines[0] == "Тональность бита: beat.wav"
    assert lines[1] == "Тон: CMaj"
    assert lines[2] == "Темп: 128 BPM (256 / 64)"
    assert lines[3] == "Длительность: 02:30"
    assert lines[4] == "Близкий вариант: G#Maj (AbMaj)"
