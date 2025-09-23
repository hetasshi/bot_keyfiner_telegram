"""Core audio analysis routines."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy.typing as npt

import librosa
import numpy as np
import soundfile as sf

from .profiles import ENHARMONIC, MAJOR_PROFILES, MINOR_PROFILES, NOTES_SHARP

logger = logging.getLogger(__name__)

CLOSE_KEY_THRESHOLD = 0.02
RMS_DROP_DB = 50.0
CHROMA_HOP_LENGTH = 512


@dataclass(slots=True)
class KeyDetectionResult:
    note: str
    mode: str
    correlation: float


@dataclass(slots=True)
class AnalysisResult:
    """Structured result of key, tempo and duration analysis."""

    filename: str
    note: str
    mode: str
    enharmonic: Optional[str]
    tuning_hz: float
    bpm: int
    bpm_double: int
    bpm_half: int
    duration: str
    close_key: Optional[str] = None

    @property
    def tone_display(self) -> str:
        return format_tone_display(self.note, self.mode)

    @property
    def tuning_display(self) -> str:
        return f"A4 ≈ {self.tuning_hz:.2f} Гц"


class AnalysisError(RuntimeError):
    """Raised when audio analysis cannot be completed."""


def _load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load audio file, falling back to soundfile if needed."""

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) != "aifc":
            raise
        y, sr = sf.read(audio_path, always_2d=False)
        if isinstance(y, tuple):
            y = y[0]
        y = np.asarray(y)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return np.asarray(y, dtype=np.float32), int(sr)
    return y, int(sr)


def analyze_file(path: str | Path, *, want_close_key: bool = False) -> AnalysisResult:
    """Analyze the audio file and return musical metrics."""
    audio_path = Path(path)
    logger.info("Loading audio file %s", audio_path)

    try:
        y, sr = _load_audio(audio_path)
    except Exception as exc:  # pragma: no cover - delegated to caller
        raise AnalysisError(f"Failed to load audio file: {exc}") from exc

    if y.size == 0:
        raise AnalysisError("Empty audio stream received.")

    y_harm, y_perc = librosa.effects.hpss(y)

    key_info, candidates, tuning_hz = _detect_key(y_harm, sr)
    bpm, bpm_double, bpm_half = _detect_bpm(y_perc, sr)
    duration = format_duration_seconds(len(y) / sr)

    close_key_display: Optional[str] = None
    if want_close_key:
        close_candidate = _find_close_key(key_info, candidates)
        if close_candidate is not None:
            close_key_display = format_tone_display(
                close_candidate.note, close_candidate.mode
            )

    result = AnalysisResult(
        filename=audio_path.name,
        note=key_info.note,
        mode=key_info.mode,
        enharmonic=ENHARMONIC.get(key_info.note),
        tuning_hz=tuning_hz,
        bpm=bpm,
        bpm_double=bpm_double,
        bpm_half=bpm_half,
        duration=duration,
        close_key=close_key_display,
    )

    if key_info.correlation < 0:
        logger.debug(
            "Detected negative correlation %.3f for %s", key_info.correlation, result.tone_display
        )

    logger.info(
        "Analyzed %s: tone=%s, bpm=%s, duration=%s",
        audio_path.name,
        result.tone_display,
        bpm,
        duration,
    )

    return result


def _detect_key(
    y_harm: np.ndarray, sr: int
) -> Tuple[KeyDetectionResult, list[KeyDetectionResult], float]:
    if np.any(y_harm):
        estimated_tuning = float(librosa.estimate_tuning(y=y_harm, sr=sr))
        if np.isnan(estimated_tuning):
            estimated_tuning = 0.0

        chroma = librosa.feature.chroma_stft(
            y=y_harm,
            sr=sr,
            hop_length=CHROMA_HOP_LENGTH,
            tuning=estimated_tuning,
        )

        rms = librosa.feature.rms(y=y_harm, hop_length=CHROMA_HOP_LENGTH)
        rms = rms.squeeze(axis=0)
    else:
        estimated_tuning = 0.0
        chroma = np.zeros((len(NOTES_SHARP), 1), dtype=float)
        rms = np.zeros(1, dtype=float)

    if chroma.shape[1] != rms.shape[0]:
        frame_count = min(chroma.shape[1], rms.shape[0])
        chroma = chroma[:, :frame_count]
        rms = rms[:frame_count]

    chroma_mean = _weighted_chroma_mean(chroma, rms)

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

    a4_hz = float(librosa.tuning_to_A4(estimated_tuning))
    if not np.isfinite(a4_hz):
        a4_hz = 440.0

    return best, candidates, a4_hz


def _weighted_chroma_mean(
    chroma: npt.NDArray[np.float_],
    rms: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    if chroma.size == 0:
        return np.zeros(chroma.shape[0], dtype=float)

    rms = np.asarray(rms, dtype=float)
    if rms.ndim != 1 or chroma.shape[1] != rms.shape[0]:
        return chroma.mean(axis=1)

    peak = float(np.max(rms))
    if peak <= 0.0:
        return chroma.mean(axis=1)

    threshold = peak * 10 ** (-RMS_DROP_DB / 20.0)
    mask = rms >= threshold

    if not np.any(mask):
        mask = rms > 0

    if np.any(mask):
        weights = rms[mask]
        weight_sum = float(np.sum(weights))
        if weight_sum > 0:
            return np.average(chroma[:, mask], axis=1, weights=weights)

    return chroma.mean(axis=1)


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
    if not np.any(y_perc):
        return 0, 0, 0

    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr)
    if onset_env.size == 0 or not np.any(onset_env):
        return 0, 0, 0

    tempo_candidates = librosa.beat.tempo(
        onset_envelope=onset_env,
        sr=sr,
        aggregate=None,
    )

    bpm = _select_base_tempo(tempo_candidates)
    bpm_double = int(round(bpm * 2)) if bpm else 0
    bpm_half = int(round(bpm / 2)) if bpm else 0
    return bpm, bpm_double, bpm_half


def _select_base_tempo(tempo_candidates: npt.ArrayLike) -> int:
    if tempo_candidates is None:
        return 0

    values = np.asarray(tempo_candidates, dtype=float)
    if values.size == 0:
        return 0

    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return 0

    buckets: dict[int, list[float]] = {}
    for tempo_value in values:
        normalized = _normalize_tempo(float(tempo_value))
        if normalized <= 0:
            continue
        bucket = int(round(normalized))
        buckets.setdefault(bucket, []).append(normalized)

    if not buckets:
        return 0

    # Pick the bucket with the most support; on tie prefer the slower tempo.
    _, best_values = max(
        buckets.items(),
        key=lambda item: (len(item[1]), -item[0]),
    )

    averaged = float(np.mean(best_values))
    return int(round(averaged)) if averaged > 0 else 0


def _normalize_tempo(value: float, *, min_bpm: float = 60.0, max_bpm: float = 190.0) -> float:
    if not np.isfinite(value) or value <= 0:
        return 0.0

    tempo = value
    while tempo > max_bpm:
        tempo /= 2.0
    while tempo < min_bpm:
        tempo *= 2.0
        if tempo == value:  # safeguard against zero/denormals
            break
        if tempo > max_bpm:
            tempo /= 2.0
            break

    if tempo < min_bpm or tempo > max_bpm:
        return 0.0

    return tempo


def format_duration_seconds(total_seconds: float) -> str:
    """Return mm:ss formatted duration for a given number of seconds."""

    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def format_tone_display(note: str, mode: str) -> str:
    """Compose tone string with optional enharmonic alternative."""

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


__all__ = [
    "AnalysisResult",
    "AnalysisError",
    "analyze_file",
    "format_duration_seconds",
    "format_tone_display",
]
