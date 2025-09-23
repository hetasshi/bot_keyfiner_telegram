import numpy as np
import soundfile as sf
import librosa

from src.audio_processing.analyzer import (
    KeyDetectionResult,
    _find_close_key,
    _detect_bpm,
    analyze_file,
)

SR = 44100


def _write_audio(tmp_path, name: str, data: np.ndarray, sr: int = SR) -> str:
    path = tmp_path / f"{name}.wav"
    sf.write(path, data, sr)
    return str(path)


def _sine_wave(note: str, duration: float, *, detune: float = 0.0) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    freq = librosa.note_to_hz(note) * (2 ** (detune / 12))
    return np.sin(2 * np.pi * freq * t)


def _chord_wave(notes: tuple[str, ...], duration: float) -> np.ndarray:
    return sum(_sine_wave(note, duration) for note in notes)


def _normalize(signal: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal.astype(np.float32)
    return (signal / peak).astype(np.float32)


def test_detect_minor_key(tmp_path) -> None:
    chord = _chord_wave(("A2", "A3", "C3", "C4", "E3", "E4"), 4.0)
    path = _write_audio(tmp_path, "amin", _normalize(chord))

    result = analyze_file(path)

    assert result.mode == "min"
    assert result.note == "A"
    assert result.key_confidence > 0.7
    expected_freq = librosa.note_to_hz("A4")
    assert abs(result.tone_frequency_hz - expected_freq) < 2.0
    assert abs(result.tuning_reference_hz - 440.0) < 2.0


def test_silence_low_confidence(tmp_path) -> None:
    silence = np.zeros(int(SR * 1.0), dtype=np.float32)
    path = _write_audio(tmp_path, "silent", silence)

    result = analyze_file(path)

    assert result.note == "0"
    assert result.mode == ""
    assert result.key_confidence == 0.0
    assert result.tone_frequency_hz == 0.0


def test_percussive_click_has_no_key(tmp_path) -> None:
    click_track = _click_track(120, duration=4.0)
    path = _write_audio(tmp_path, "click", click_track)

    result = analyze_file(path)

    assert result.note == "0"
    assert result.mode == ""
    assert result.tone_frequency_hz == 0.0


def test_detect_major_key(tmp_path) -> None:
    chord_progression = np.concatenate(
        [
            _chord_wave(("C3", "E3", "G3", "C4"), 1.0),
            _chord_wave(("F3", "A3", "C4", "F4"), 1.0),
            _chord_wave(("G3", "B3", "D4", "G4"), 1.0),
        ]
    )
    path = _write_audio(tmp_path, "cmaj", _normalize(chord_progression))

    result = analyze_file(path)

    assert result.mode == "Maj"
    assert result.note == "C"
    assert result.key_confidence > 0.55
    expected_freq = librosa.note_to_hz("C4")
    assert abs(result.tone_frequency_hz - expected_freq) < 1.0
    assert abs(result.tuning_reference_hz - 440.0) < 2.0


def test_key_with_detuning(tmp_path) -> None:
    detune = -0.2  # semitones
    chord = sum(
        _sine_wave(note, 3.0, detune=detune)
        for note in ("E3", "E4", "G#3", "B3")
    )
    path = _write_audio(tmp_path, "detuned_e", _normalize(chord))

    result = analyze_file(path)

    assert result.note == "E"
    assert result.mode == "Maj"
    assert result.key_confidence > 0.4
    expected_detune_factor = 2 ** (detune / 12)
    expected_tuning = 440.0 * expected_detune_factor
    assert abs(result.tuning_reference_hz - expected_tuning) < 2.0
    expected_freq = librosa.note_to_hz("E4") * expected_detune_factor
    assert abs(result.tone_frequency_hz - expected_freq) < 2.0


def _click_track(bpm: float, duration: float) -> np.ndarray:
    total_samples = int(SR * duration)
    click = np.zeros(total_samples, dtype=np.float32)
    interval = max(1, int(round(SR * 60.0 / bpm)))
    pulse_length = int(0.01 * SR)
    for start in range(0, total_samples, interval):
        end = min(start + pulse_length, total_samples)
        click[start:end] = 1.0
    return click


def test_detect_bpm(tmp_path) -> None:
    data = _click_track(140, duration=20.0)
    path = _write_audio(tmp_path, "tempo", data)

    result = analyze_file(path)

    assert abs(result.bpm - 140) <= 1
    assert result.bpm_double == result.bpm * 2
    expected_half = int(round(result.bpm / 2)) if result.bpm else 0
    assert result.bpm_half == expected_half


def test_bpm_not_found(tmp_path) -> None:
    silence = np.zeros(int(SR * 0.2), dtype=np.float32)
    path = _write_audio(tmp_path, "silence", silence)

    result = analyze_file(path)

    assert result.bpm == 0
    assert result.bpm_double == 0
    assert result.bpm_half == 0


def _tempo_stub(responses):
    arrays = [np.asarray(response, dtype=float) for response in responses]
    state = {"count": 0}

    def fake_tempo(*args, **kwargs):
        idx = state["count"]
        if idx >= len(arrays):
            raise AssertionError("Unexpected librosa.beat.tempo call")
        state["count"] += 1
        return arrays[idx]

    return fake_tempo, state


def _candidate(note: str, mode: str, score: float, confidence: float) -> KeyDetectionResult:
    return KeyDetectionResult(
        note=note,
        mode=mode,
        correlation=score,
        score=score,
        positive_support=0.5,
        dominant_share=0.4,
        confidence=confidence,
    )


def test_bpm_double_correction(monkeypatch) -> None:
    responses = [[170.0] * 100 + [85.0] * 90]
    fake_tempo, state = _tempo_stub(responses)
    monkeypatch.setattr(librosa.beat, "tempo", fake_tempo)

    y_perc = np.ones(4096, dtype=np.float32)
    bpm, bpm_double, bpm_half = _detect_bpm(y_perc, SR)

    assert bpm == 85
    assert bpm_double == 170
    assert bpm_half in {42, 43}
    assert state["count"] == 1


def test_bpm_low_correction(monkeypatch) -> None:
    responses = [[50.0] * 30 + [100.0] * 5]
    fake_tempo, state = _tempo_stub(responses)
    monkeypatch.setattr(librosa.beat, "tempo", fake_tempo)

    y_perc = np.ones(4096, dtype=np.float32)
    bpm, bpm_double, bpm_half = _detect_bpm(y_perc, SR)

    assert bpm == 100
    assert bpm_double == 200
    assert bpm_half == 50
    assert state["count"] == 1


def test_bpm_harmonic_fallback(monkeypatch) -> None:
    responses = [[120.0] * 40 + [60.0] * 36]
    fake_tempo, state = _tempo_stub(responses)
    monkeypatch.setattr(librosa.beat, "tempo", fake_tempo)

    y_perc = np.zeros(4096, dtype=np.float32)
    harmonic = _sine_wave("A4", 2.0)
    bpm, bpm_double, bpm_half = _detect_bpm(y_perc, SR, y_harm=harmonic, y_original=None)

    assert bpm == 120
    assert bpm_double == 240
    assert bpm_half == 60
    assert state["count"] == 1


def test_find_close_key_returns_candidate() -> None:
    best = _candidate("C", "Maj", score=0.82, confidence=0.36)
    second = _candidate("G", "Maj", score=0.75, confidence=0.25)
    close = _find_close_key(best, [best, second])

    assert close is second


def test_find_close_key_rejects_low_confidence() -> None:
    best = _candidate("D", "min", score=0.78, confidence=0.21)
    second = _candidate("A", "min", score=0.72, confidence=0.28)

    assert _find_close_key(best, [best, second]) is None


def test_find_close_key_requires_second_confidence() -> None:
    best = _candidate("F", "Maj", score=0.81, confidence=0.34)
    second = _candidate("C", "Maj", score=0.74, confidence=0.18)

    assert _find_close_key(best, [best, second]) is None
