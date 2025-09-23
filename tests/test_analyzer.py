import librosa
import numpy as np
import soundfile as sf

from src.audio_processing.analyzer import analyze_file

SR = 44100


def _write_audio(tmp_path, name: str, data: np.ndarray, sr: int = SR) -> str:
    path = tmp_path / f"{name}.wav"
    sf.write(path, data, sr)
    return str(path)


def _sine_wave(note: str, duration: float) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    freq = librosa.note_to_hz(note)
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
    assert result.note in {"A", "G#"}  # допускаем редкую энгармонику
    assert abs(result.tuning_hz - 440.0) < 2.0


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
    assert abs(result.tuning_hz - 440.0) < 2.0


def test_key_detection_with_quiet_distractors(tmp_path) -> None:
    loud_chord = _chord_wave(("C3", "E3", "G3", "C4"), 2.0)
    quiet_ratio = 10 ** (-45 / 20)
    quiet_chord = quiet_ratio * _chord_wave(("F#3", "A#3", "C#4"), 6.0)
    mixture = np.concatenate([loud_chord, quiet_chord, loud_chord])
    path = _write_audio(tmp_path, "cmaj_weighted", _normalize(mixture))

    result = analyze_file(path)

    assert result.mode == "Maj"
    assert result.note == "C"


def _click_track(bpm: float, duration: float) -> np.ndarray:
    total_samples = int(SR * duration)
    click = np.zeros(total_samples, dtype=np.float32)
    interval = max(1, int(round(SR * 60.0 / bpm)))
    pulse_length = int(0.01 * SR)
    for start in range(0, total_samples, interval):
        end = min(start + pulse_length, total_samples)
        click[start:end] = 1.0
    return click


def _click_track_with_eighths(bpm: float, duration: float) -> np.ndarray:
    total_samples = int(SR * duration)
    click = np.zeros(total_samples, dtype=np.float32)
    interval = max(1, int(round(SR * 60.0 / (bpm * 2))))
    pulse_length = int(0.01 * SR)
    idx = 0
    for start in range(0, total_samples, interval):
        end = min(start + pulse_length, total_samples)
        amplitude = 1.0 if idx % 2 == 0 else 0.35
        click[start:end] = amplitude
        idx += 1
    return click


def test_detect_bpm(tmp_path) -> None:
    data = _click_track(140, duration=20.0)
    path = _write_audio(tmp_path, "tempo", data)

    result = analyze_file(path)

    assert abs(result.bpm - 140) <= 1
    assert result.bpm_double == result.bpm * 2
    expected_half = int(round(result.bpm / 2)) if result.bpm else 0
    assert result.bpm_half == expected_half


def test_detect_bpm_double_time_normalization(tmp_path) -> None:
    data = _click_track_with_eighths(140, duration=20.0)
    path = _write_audio(tmp_path, "tempo_double", data)

    result = analyze_file(path)

    assert abs(result.bpm - 140) <= 1
    assert result.bpm_double == int(round(result.bpm * 2))
    expected_half = int(round(result.bpm / 2)) if result.bpm else 0
    assert result.bpm_half == expected_half


def test_detect_tuning_from_detuned_signal(tmp_path) -> None:
    chord = _chord_wave(("C3", "E3", "G3", "C4"), 4.0)
    detune_steps = -0.35
    detuned = librosa.effects.pitch_shift(chord, sr=SR, n_steps=detune_steps)
    path = _write_audio(tmp_path, "cmaj_detuned", _normalize(detuned))

    result = analyze_file(path)

    assert result.mode == "Maj"
    assert result.note == "C"
    expected_a4 = 440.0 * (2 ** (detune_steps / 12.0))
    assert abs(result.tuning_hz - expected_a4) < 1.0


def test_bpm_not_found(tmp_path) -> None:
    silence = np.zeros(int(SR * 0.2), dtype=np.float32)
    path = _write_audio(tmp_path, "silence", silence)

    result = analyze_file(path)

    assert result.bpm == 0
    assert result.bpm_double == 0
    assert result.bpm_half == 0
