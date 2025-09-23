from src.bot import messages
from src.audio_processing.analyzer import AnalysisResult


def test_file_too_large_message() -> None:
    message = messages.file_too_large(70)

    assert "слишком большой" in message
    assert "70" in message


def test_unsupported_file_mentions_webm() -> None:
    assert "webm" in messages.unsupported_file().lower()


def test_message_html_escape() -> None:
    result = AnalysisResult(
        filename="bad<&>.mp3",
        note="C",
        mode="Maj",
        enharmonic="Db",
        tone_frequency_hz=261.63,
        tuning_reference_hz=440.0,
        key_confidence=0.9,
        bpm=120,
        bpm_double=240,
        bpm_half=60,
        duration="01:00",
        close_key=None,
    )

    formatted = messages.format_analysis_result(result)
    first_line = formatted.splitlines()[0]

    assert first_line == "Тональность бита: bad&lt;&amp;&gt;.mp3"
