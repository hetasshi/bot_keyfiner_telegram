"""Audio analysis routines for the KeyFinder bot."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np

from .profiles import ENHARMONIC, MAJOR_PROFILES, MINOR_PROFILES, NOTES_SHARP

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class KeyCandidate:
    """Representation of a detected key candidate."""

    note_index: int
    mode: str
    score: float

    @property
    def note(self) -> str:
        return NOTES_SHARP[self.note_index]

    @property
    def enharmonic(self) -> str | None:
        return ENHARMONIC.get(self.note)


def analyze_file(path: str | Path, *, want_close_key: bool = False) -> dict[str, Any]:
    """Analyze an audio file and return musical metrics.

    Parameters
    ----------
    path:
        Path to the audio file.
    want_close_key:
        Whether to include a near-key alternative if correlations are similar.
    """

    file_path = Path(path)
    logger.info("Loading audio file: %s", file_path)
    y, sr = librosa.load(str(file_path), sr=None, mono=True)
    logger.debug("Loaded %s samples at %s Hz", len(y), sr)

    y_harm, y_perc = librosa.effects.hpss(y)
    key_result, close_candidate = _detect_key(y_harm, sr, want_close_key=want_close_key)
    tempo_result = _detect_bpm(y_perc, sr)
    duration = _format_duration(len(y) / float(sr))

    result: dict[str, Any] = {
        "filename": file_path.name,
        "note": key_result.note,
        "mode": key_result.mode,
        "enharmonic": key_result.enharmonic,
        "bpm": tempo_result["bpm"],
        "bpm2x": tempo_result["bpm2x"],
        "bpm1_2": tempo_result["bpm1_2"],
        "duration": duration,
    }

    if close_candidate is not None:
        result["close_key"] = {
            "note": close_candidate.note,
            "mode": close_candidate.mode,
            "enharmonic": close_candidate.enharmonic,
            "score": close_candidate.score,
        }

    logger.info(
        "Analysis complete: %s %s, %s BPM, duration %s",
        result["note"],
        result["mode"],
        result["bpm"],
        result["duration"],
    )
    return result


def _detect_key(y_harm: np.ndarray, sr: int, *, want_close_key: bool) -> tuple[KeyCandidate, KeyCandidate | None]:
    """Detect the musical key using the Krumhansl–Schmuckler algorithm."""

    chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    candidates: list[KeyCandidate] = []
    for idx in range(len(NOTES_SHARP)):
        major_score = _safe_correlation(chroma_mean, MAJOR_PROFILES[idx])
        candidates.append(KeyCandidate(note_index=idx, mode="Maj", score=major_score))
        minor_score = _safe_correlation(chroma_mean, MINOR_PROFILES[idx])
        candidates.append(KeyCandidate(note_index=idx, mode="min", score=minor_score))

    # Sort by score descending while preserving order of equal items.
    candidates.sort(key=lambda item: item.score, reverse=True)
    best = candidates[0]
    close_candidate: KeyCandidate | None = None
    if want_close_key and len(candidates) > 1:
        second = candidates[1]
        if abs(best.score - second.score) < 0.02:
            close_candidate = second

    return best, close_candidate


def _detect_bpm(y_perc: np.ndarray, sr: int) -> dict[str, int]:
    """Detect tempo (BPM) using the percussive component."""

    tempo_values = librosa.beat.tempo(y=y_perc, sr=sr)
    bpm = int(round(float(tempo_values[0]))) if tempo_values.size else 0
    bpm = max(bpm, 0)
    bpm2x = int(round(bpm * 2))
    bpm1_2 = int(round(bpm / 2))
    return {"bpm": bpm, "bpm2x": bpm2x, "bpm1_2": bpm1_2}


def _format_duration(duration_seconds: float) -> str:
    """Format duration in seconds into ``mm:ss`` string.

    >>> _format_duration(65.8)
    '01:05'
    >>> _format_duration(0)
    '00:00'
    """

    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def _safe_correlation(a: Iterable[float], b: Iterable[float]) -> float:
    """Calculate Pearson correlation coefficient safeguarding against NaNs."""

    a_arr = np.asarray(list(a), dtype=np.float_)
    b_arr = np.asarray(list(b), dtype=np.float_)
    if a_arr.ndim != 1 or b_arr.ndim != 1:
        raise ValueError("Inputs must be one-dimensional arrays")

    if np.allclose(a_arr, a_arr[0]) or np.allclose(b_arr, b_arr[0]):
        return float("-inf")

    corr_matrix = np.corrcoef(a_arr, b_arr)
    corr = float(corr_matrix[0, 1])
    if np.isnan(corr):
        return float("-inf")
    return corr


if __name__ == "__main__":  # pragma: no cover - utility test run
    logging.basicConfig(level=logging.INFO)
    logger.info("Duration test: %s", _format_duration(123.4))
