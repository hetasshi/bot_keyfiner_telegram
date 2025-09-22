"""Core audio analysis routines."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import librosa
import numpy as np

from .profiles import ENHARMONIC, MAJOR_PROFILES, MINOR_PROFILES, NOTES_SHARP

logger = logging.getLogger(__name__)

CLOSE_KEY_THRESHOLD = 0.02


@dataclass(slots=True)
class KeyDetectionResult:
    note: str
    mode: str
    correlation: float


class AnalysisError(RuntimeError):
    """Raised when audio analysis cannot be completed."""


def analyze_file(path: str | Path, *, want_close_key: bool = False) -> Dict[str, Any]:
    """Analyze the audio file and return musical metrics."""
    audio_path = Path(path)
    logger.info("Loading audio file %s", audio_path)

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as exc:  # pragma: no cover - delegated to caller
        raise AnalysisError(f"Failed to load audio file: {exc}") from exc

    if y.size == 0:
        raise AnalysisError("Empty audio stream received.")

    y_harm, y_perc = librosa.effects.hpss(y)

    key_info, candidates = _detect_key(y_harm, sr)
    bpm, bpm_double, bpm_half = _detect_bpm(y_perc, sr)
    duration = _format_duration(len(y), sr)

    result: Dict[str, Any] = {
        "filename": audio_path.name,
        "note": key_info.note,
        "mode": key_info.mode,
        "enharmonic": ENHARMONIC.get(key_info.note),
        "tone": _format_tone_display(key_info.note, key_info.mode),
        "bpm": bpm,
        "bpm_double": bpm_double,
        "bpm_half": bpm_half,
        "duration": duration,
        "close_key": None,
    }

    if key_info.correlation < 0:
        logger.debug(
            "Detected negative correlation %.3f for %s",
            key_info.correlation,
            result["tone"],
        )

    if want_close_key:
        close_candidate = _find_close_key(key_info, candidates)
        if close_candidate is not None:
            result["close_key"] = _format_tone_display(
                close_candidate.note, close_candidate.mode
            )

    logger.info(
        "Analyzed %s: tone=%s, bpm=%s, duration=%s",
        audio_path.name,
        result["tone"],
        bpm,
        duration,
    )

    return result


def _detect_key(y_harm: np.ndarray, sr: int) -> Tuple[KeyDetectionResult, list[KeyDetectionResult]]:
    chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    candidates: list[KeyDetectionResult] = []
    for idx, note in enumerate(NOTES_SHARP):
        corr_major = _safe_correlation(chroma_mean, MAJOR_PROFILES[idx])
        candidates.append(KeyDetectionResult(note=note, mode="Maj", correlation=corr_major))

        corr_minor = _safe_correlation(chroma_mean, MINOR_PROFILES[idx])
        candidates.append(KeyDetectionResult(note=note, mode="min", correlation=corr_minor))

    if not candidates:
        raise AnalysisError("No key candidates produced by the algorithm.")

    candidates.sort(key=lambda item: item.correlation, reverse=True)
    best = candidates[0]
    return best, candidates


def _find_close_key(
    best: KeyDetectionResult, candidates: Sequence[KeyDetectionResult]
) -> Optional[KeyDetectionResult]:
    if len(candidates) < 2:
        return None

    second = candidates[1]
    delta = abs(best.correlation - second.correlation)
    logger.debug(
        "Close key delta=%.5f (best=%s%s, second=%s%s)",
        delta,
        best.note,
        best.mode,
        second.note,
        second.mode,
    )
    if delta < CLOSE_KEY_THRESHOLD:
        return second
    return None


def _detect_bpm(y_perc: np.ndarray, sr: int) -> Tuple[int, int, int]:
    tempo = librosa.beat.tempo(y=y_perc, sr=sr)
    tempo_value = float(tempo[0]) if tempo.size > 0 else 0.0
    if np.isnan(tempo_value):
        tempo_value = 0.0
    bpm = int(round(tempo_value)) if tempo_value > 0 else 0
    bpm_double = int(round(bpm * 2)) if bpm else 0
    bpm_half = int(round(bpm / 2)) if bpm else 0
    return bpm, bpm_double, bpm_half


def _format_duration(num_samples: int, sr: int) -> str:
    """Return mm:ss formatted duration for a given number of samples.

    >>> _format_duration(44100 * 65, 44100)
    '01:05'
    >>> _format_duration(22050 * 3, 22050)
    '00:03'
    """

    total_seconds = num_samples / sr
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def _format_tone_display(note: str, mode: str) -> str:
    """Compose tone string with optional enharmonic alternative.

    >>> _format_tone_display('A#', 'min')
    'A#min (Bbmin)'
    >>> _format_tone_display('C', 'Maj')
    'CMaj'
    """

    tone = f"{note}{mode}"
    enharmonic = ENHARMONIC.get(note)
    if enharmonic:
        return f"{tone} ({enharmonic}{mode})"
    return tone


def _safe_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if not np.any(arr_a) or not np.any(arr_b):
        return 0.0
    corr_matrix = np.corrcoef(arr_a, arr_b)
    corr = float(corr_matrix[0, 1])
    if np.isnan(corr):
        return 0.0
    return corr


if __name__ == "__main__":  # pragma: no cover - doctest helper
    import doctest

    doctest.testmod()
