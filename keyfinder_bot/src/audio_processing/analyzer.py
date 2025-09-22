"""Audio feature extraction for key, tempo and duration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

from .profiles import ENHARMONIC, MAJOR_PROFILES, MINOR_PROFILES, NOTES_SHARP

logger = logging.getLogger(__name__)

CLOSE_KEY_THRESHOLD = 0.02


def analyze_file(path: str | Path, *, want_close_key: bool = False) -> Dict[str, object]:
    """Analyze audio file and return musical attributes.

    The returned dictionary contains keys:
    - ``filename``: исходное имя файла (без пути).
    - ``note`` / ``mode`` / ``enharmonic``: итоговая тональность.
    - ``bpm`` / ``bpm_double`` / ``bpm_half``: темп и его половина/удвоение.
    - ``duration``: продолжительность в формате mm:ss.
    - ``close_key`` (опционально): альтернатива при включённом флаге.
    """

    source_path = Path(path)
    logger.debug("Starting analysis for %s", source_path)
    y, sr = librosa.load(source_path, sr=None, mono=True)
    if y.size == 0:
        raise ValueError("Empty audio file provided")

    y_harm, y_perc = librosa.effects.hpss(y)

    key_info, close_info = _detect_key(y_harm, sr, want_close_key=want_close_key)
    bpm, bpm_double, bpm_half = _detect_bpm(y_perc, sr)
    duration = _detect_duration(y, sr)

    result: Dict[str, object] = {
        "filename": source_path.name,
        "note": key_info["note"],
        "mode": key_info["mode"],
        "enharmonic": key_info.get("enharmonic"),
        "bpm": bpm,
        "bpm_double": bpm_double,
        "bpm_half": bpm_half,
        "duration": duration,
    }
    if want_close_key and close_info is not None:
        result["close_key"] = close_info

    logger.debug("Analysis completed for %s", source_path)
    return result


def _detect_key(
    y_harm: np.ndarray,
    sr: int,
    *,
    want_close_key: bool = False,
) -> Tuple[Dict[str, object], Optional[Dict[str, object]]]:
    """Detect the musical key using the Krumhansl-Schmuckler method."""

    chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    candidates: List[Tuple[float, str, str]] = []
    for idx, note in enumerate(NOTES_SHARP):
        corr_major = _safe_corr(chroma_mean, MAJOR_PROFILES[idx])
        corr_minor = _safe_corr(chroma_mean, MINOR_PROFILES[idx])
        candidates.append((corr_major, note, "Maj"))
        candidates.append((corr_minor, note, "min"))

    # Sort by correlation descending.
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_corr, best_note, best_mode = candidates[0]
    best_info = _build_key_result(best_note, best_mode, best_corr)

    close_info: Optional[Dict[str, object]] = None
    if want_close_key and len(candidates) > 1:
        close_corr, close_note, close_mode = candidates[1]
        if abs(best_corr - close_corr) < CLOSE_KEY_THRESHOLD:
            close_info = _build_key_result(close_note, close_mode, close_corr)

    return best_info, close_info


def _detect_bpm(y_perc: np.ndarray, sr: int) -> Tuple[int, int, int]:
    """Detect tempo (BPM) using the percussive component."""

    tempo = librosa.beat.tempo(y=y_perc, sr=sr)
    if tempo.size == 0:
        bpm = 0
    else:
        bpm = int(round(float(tempo[0])))

    bpm = max(bpm, 1)
    bpm_double = int(round(bpm * 2))
    bpm_half = max(int(round(bpm / 2)), 1)
    return bpm, bpm_double, bpm_half


def _detect_duration(y: np.ndarray, sr: int) -> str:
    """Return formatted track duration."""

    duration_seconds = float(len(y)) / float(sr)
    return _format_duration(duration_seconds)


def _build_key_result(note: str, mode: str, correlation: float) -> Dict[str, object]:
    """Create dictionary with key information."""

    enharmonic = ENHARMONIC.get(note)
    return {
        "note": note,
        "mode": mode,
        "enharmonic": enharmonic,
        "correlation": float(correlation),
    }


def _safe_corr(chroma: np.ndarray, profile: np.ndarray) -> float:
    """Compute Pearson correlation while handling NaN results."""

    corr_matrix = np.corrcoef(chroma, profile)
    if corr_matrix.shape[0] < 2 or corr_matrix.shape[1] < 2:
        return 0.0
    corr = float(corr_matrix[0, 1])
    if np.isnan(corr):
        return 0.0
    return corr


def _format_duration(seconds: float) -> str:
    """Format seconds as mm:ss string.

    >>> _format_duration(65.8)
    '01:05'
    >>> _format_duration(3600)
    '60:00'
    """

    total_seconds = max(int(seconds), 0)
    minutes = total_seconds // 60
    remaining = total_seconds % 60
    return f"{minutes:02d}:{remaining:02d}"


if __name__ == "__main__":  # pragma: no cover - simple sanity checks
    assert _format_duration(0) == "00:00"
    assert _format_duration(65.2) == "01:05"
    demo = _build_key_result("C#", "min", 0.9)
    assert demo["note"] == "C#" and demo["mode"] == "min"
