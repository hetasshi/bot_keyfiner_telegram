"""Audio analysis utilities for determining musical properties."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from .profiles import ENHARMONIC, MAJOR_PROFILES, MINOR_PROFILES, NOTES_SHARP

logger = logging.getLogger(__name__)

CLOSE_KEY_THRESHOLD = 0.02


def analyze_file(path: str | Path, *, want_close_key: bool = False) -> dict[str, Any]:
    """Analyze an audio file to determine key, BPM and duration."""

    file_path = Path(path)
    logger.info("Starting analysis for %s", file_path)
    y, sr = librosa.load(file_path, sr=None, mono=True)
    logger.debug("Loaded audio: %s samples at %s Hz", len(y), sr)

    # Harmonic/percussive source separation
    y_harm, y_perc = librosa.effects.hpss(y)
    logger.debug("HPSS complete: harm=%s, perc=%s", y_harm.shape, y_perc.shape)

    key_result = _detect_key(y_harm, sr, want_close_key=want_close_key)
    bpm_result = _detect_bpm(y_perc, sr)
    duration_str = _format_duration(len(y) / float(sr))

    result: dict[str, Any] = {
        "filename": file_path.name,
        "note": key_result["note"],
        "mode": key_result["mode"],
        "enharmonic": key_result.get("enharmonic"),
        "bpm": bpm_result["bpm"],
        "bpm_double": bpm_result["bpm_double"],
        "bpm_half": bpm_result["bpm_half"],
        "duration": duration_str,
        "correlation": key_result["correlation"],
    }

    if want_close_key and key_result.get("close_key"):
        result["close_key"] = key_result["close_key"]

    logger.info(
        "Analysis finished for %s: %s%s, %s BPM", file_path,
        result["note"], result["mode"], result["bpm"],
    )
    return result


def _detect_key(y_harm: np.ndarray, sr: int, *, want_close_key: bool) -> dict[str, Any]:
    chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    candidates: list[dict[str, Any]] = []
    for idx, note in enumerate(NOTES_SHARP):
        major_corr = _pearson_corr(chroma_mean, MAJOR_PROFILES[idx])
        candidates.append({"note": note, "mode": "Maj", "corr": major_corr})

        minor_corr = _pearson_corr(chroma_mean, MINOR_PROFILES[idx])
        candidates.append({"note": note, "mode": "min", "corr": minor_corr})

    best = max(candidates, key=lambda item: item["corr"])
    logger.debug("Best key candidate: %s %s corr=%.4f", best["note"], best["mode"], best["corr"])

    close_key: dict[str, Any] | None = None
    if want_close_key and len(candidates) > 1:
        sorted_candidates = sorted(candidates, key=lambda item: item["corr"], reverse=True)
        second = sorted_candidates[1]
        if np.isfinite(second["corr"]) and abs(best["corr"] - second["corr"]) < CLOSE_KEY_THRESHOLD:
            close_key = _build_key_entry(second)
            logger.debug("Close key alternative: %s %s corr=%.4f", second["note"], second["mode"], second["corr"])

    result = _build_key_entry(best)
    if close_key:
        result["close_key"] = close_key
    return result


def _build_key_entry(candidate: dict[str, Any]) -> dict[str, Any]:
    note = candidate["note"]
    mode = candidate["mode"]
    corr = float(candidate["corr"])
    enharmonic = ENHARMONIC.get(note)
    if enharmonic == note:
        enharmonic = None

    return {
        "note": note,
        "mode": mode,
        "enharmonic": enharmonic,
        "correlation": corr,
    }


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size != b.size:
        raise ValueError("Arrays must have the same size to compute correlation")

    a_mean = a - np.mean(a)
    b_mean = b - np.mean(b)
    denom = np.linalg.norm(a_mean) * np.linalg.norm(b_mean)
    if denom == 0:
        return float("-inf")
    return float(np.dot(a_mean, b_mean) / denom)


def _detect_bpm(y_perc: np.ndarray, sr: int) -> dict[str, int]:
    tempo = librosa.beat.tempo(y=y_perc, sr=sr)
    bpm = int(round(float(tempo[0]))) if tempo.size else 0
    if bpm < 0:
        bpm = 0

    bpm_double = int(round(bpm * 2))
    bpm_half = int(round(bpm / 2)) if bpm else 0

    return {"bpm": bpm, "bpm_double": bpm_double, "bpm_half": bpm_half}


def _format_duration(duration_seconds: float) -> str:
    """Format seconds into ``mm:ss`` representation.

    >>> _format_duration(125.2)
    '02:05'
    >>> _format_duration(59)
    '00:59'
    """

    if duration_seconds < 0:
        duration_seconds = 0

    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


__all__ = ["analyze_file"]
