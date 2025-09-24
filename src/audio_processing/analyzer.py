"""Core audio analysis routines."""
from __future__ import annotations

import os
import math
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import librosa
import librosa.util
import numpy as np
import soundfile as sf

from .profiles import ENHARMONIC, MAJOR_PROFILES, MINOR_PROFILES, NOTES_SHARP

logger = logging.getLogger(__name__)

CLOSE_KEY_THRESHOLD = 0.12
CLOSE_KEY_MIN_CONF = 0.30
CLOSE_KEY_ALT_MIN_CONF = 0.20
POSITIVE_CORRELATION_THRESHOLD = 0.6
HARMONIC_HOP_LENGTH = 512
HARMONIC_RMS_VOID = 1e-5


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - configuration error
        raise RuntimeError(f"Invalid float value for {name!r}: {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - configuration error
        raise RuntimeError(f"Invalid integer value for {name!r}: {raw!r}") from exc


BPM_MIN = _env_int("BPM_MIN", 60)
BPM_MAX = _env_int("BPM_MAX", 180)
BPM_DOUBLE_SWITCH = _env_int("BPM_DOUBLE_SWITCH", 165)
BPM_HALF_SWITCH = _env_int("BPM_HALF_SWITCH", 65)
BPM_CANDIDATE_TOLERANCE = _env_float("BPM_CANDIDATE_TOLERANCE", 1.5)
BPM_SUPPORT_RATIO = _env_float("BPM_SUPPORT_RATIO", 0.9)

CONF_VOID = _env_float("KEYFINDER_KEY_CONF_VOID", 0.05)

NOTE_TO_INDEX = {note: idx for idx, note in enumerate(NOTES_SHARP)}


@dataclass(slots=True)
class _Profile:
    note: str
    mode: str
    centered: np.ndarray
    norm: float


_PROFILE_DATA: list[_Profile] = []
for idx, note in enumerate(NOTES_SHARP):
    major_vec = np.asarray(MAJOR_PROFILES[idx], dtype=float)
    major_centered = major_vec - major_vec.mean()
    major_norm = np.linalg.norm(major_centered)
    _PROFILE_DATA.append(
        _Profile(
            note=note,
            mode="Maj",
            centered=major_centered,
            norm=major_norm if major_norm > 0 else 1.0,
        )
    )

    minor_vec = np.asarray(MINOR_PROFILES[idx], dtype=float)
    minor_centered = minor_vec - minor_vec.mean()
    minor_norm = np.linalg.norm(minor_centered)
    _PROFILE_DATA.append(
        _Profile(
            note=note,
            mode="min",
            centered=minor_centered,
            norm=minor_norm if minor_norm > 0 else 1.0,
        )
    )

PROFILE_CENTERS = np.stack([profile.centered for profile in _PROFILE_DATA], axis=0)
PROFILE_NORMS = np.array([profile.norm for profile in _PROFILE_DATA], dtype=float)


@dataclass(slots=True)
class KeyDetectionResult:
    note: str
    mode: str
    correlation: float
    score: float
    positive_support: float
    dominant_share: float
    confidence: float


@dataclass(slots=True)
class AnalysisResult:
    """Structured result of key, tempo and duration analysis."""

    filename: str
    note: str
    mode: str
    enharmonic: Optional[str]
    tone_frequency_hz: float
    tuning_reference_hz: float
    key_confidence: float
    bpm: int
    bpm_double: int
    bpm_half: int
    duration: str
    close_key: Optional[str] = None

    @property
    def tone_display(self) -> str:
        if not self.note or self.note == "0":
            return "0"
        return format_tone_display(self.note, self.mode)


class AnalysisError(RuntimeError):
    """Raised when audio analysis cannot be completed."""


def _load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load audio file, falling back to soundfile if needed."""

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as first_exc:
        logger.debug(
            "librosa.load failed for %s (%s); trying soundfile fallback",
            audio_path,
            first_exc,
        )
        try:
            y, sr = sf.read(audio_path, always_2d=False)
        except Exception as sf_exc:  # pragma: no cover - delegated to caller
            raise RuntimeError(
                "librosa.load failed: "
                f"{first_exc}; soundfile.read failed: {sf_exc}"
            ) from sf_exc

    if isinstance(y, tuple):
        y = y[0]
    y = np.asarray(y)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return np.asarray(y, dtype=np.float32), int(sr)


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

    key_info, candidates, tuning_offset = _detect_key(y_harm, sr)

    rms_vec = librosa.feature.rms(
        y=y_harm, hop_length=HARMONIC_HOP_LENGTH
    ).flatten()
    harm_rms = float(np.mean(rms_vec)) if rms_vec.size else 0.0

    no_key = (harm_rms < HARMONIC_RMS_VOID) or (
        key_info.confidence < CONF_VOID
    )
    if no_key:
        note, mode, enharmonic = "0", "", None
    else:
        note, mode = key_info.note, key_info.mode
        enharmonic = ENHARMONIC.get(key_info.note)

    bpm, bpm_double, bpm_half = _detect_bpm(
        y_perc, sr, y_harm=y_harm, y_original=y
    )
    duration = format_duration_seconds(len(y) / sr)

    tuning_reference_hz = _tuning_reference_hz(tuning_offset)
    tone_frequency_hz = _tone_frequency_hz(note, tuning_offset)

    close_key_display: Optional[str] = None
    if want_close_key and not no_key:
        close_candidate = _find_close_key(key_info, candidates)
        if close_candidate is not None:
            close_key_display = format_tone_display(
                close_candidate.note, close_candidate.mode
            )

    result = AnalysisResult(
        filename=audio_path.name,
        note=note,
        mode=mode,
        enharmonic=enharmonic,
        tone_frequency_hz=tone_frequency_hz,
        tuning_reference_hz=tuning_reference_hz,
        key_confidence=key_info.confidence,
        bpm=bpm,
        bpm_double=bpm_double,
        bpm_half=bpm_half,
        duration=duration,
        close_key=close_key_display,
    )

    logger.info(
        (
            "Analyzed %s: tone=%s (confidence %.2f, harm_rms %.6f, no_key=%s), "
            "bpm=%s, duration=%s, A4=%.2f"
        ),
        audio_path.name,
        result.tone_display,
        key_info.confidence,
        harm_rms,
        no_key,
        bpm,
        duration,
        tuning_reference_hz,
    )

    return result


def _detect_key(
    y_harm: np.ndarray, sr: int
) -> Tuple[KeyDetectionResult, list[KeyDetectionResult], float]:
    """Detect the musical key using harmonic chroma analysis."""

    if y_harm.size == 0:
        raise AnalysisError("Empty harmonic component for key detection.")

    tuning = float(librosa.estimate_tuning(y=y_harm, sr=sr))
    if math.isnan(tuning):
        tuning = 0.0

    chroma_cqt = librosa.feature.chroma_cqt(
        y=y_harm,
        sr=sr,
        hop_length=HARMONIC_HOP_LENGTH,
        bins_per_octave=36,
        tuning=tuning,
    )
    chroma_cens = librosa.feature.chroma_cens(
        C=chroma_cqt,
        hop_length=HARMONIC_HOP_LENGTH,
    )
    chroma = 0.65 * chroma_cqt + 0.35 * chroma_cens
    chroma = np.maximum(chroma, 0.0)

    frames = chroma.shape[1]
    if frames == 0:
        chroma = np.zeros((12, 1), dtype=float)
        frames = 1

    frame_sum = chroma.sum(axis=0)
    valid_sum_mask = frame_sum > 0
    if np.any(valid_sum_mask):
        chroma[:, valid_sum_mask] /= frame_sum[valid_sum_mask]

    rms = librosa.feature.rms(
        y=y_harm, hop_length=HARMONIC_HOP_LENGTH, center=True
    ).flatten()
    if rms.size != frames:
        rms = librosa.util.fix_length(rms, frames)
    weights = np.maximum(rms, 0.0)

    frame_centered = chroma - chroma.mean(axis=0, keepdims=True)
    frame_norm = np.linalg.norm(frame_centered, axis=0)
    valid_frames = frame_norm > 1e-6
    frame_centered[:, ~valid_frames] = 0.0
    frame_norm[~valid_frames] = 1.0

    weights = weights * valid_frames
    weights_sum = float(weights.sum())
    has_active_frames = weights_sum > 0
    if not has_active_frames:
        weights = np.ones(frames, dtype=float)
        weights_sum = float(weights.sum())
    weights /= weights_sum

    corr_matrix = PROFILE_CENTERS @ frame_centered
    corr_matrix /= PROFILE_NORMS[:, None]
    corr_matrix /= frame_norm[None, :]
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

    weighted_corr = corr_matrix @ weights
    positive_support = (corr_matrix >= POSITIVE_CORRELATION_THRESHOLD).astype(float) @ weights

    if has_active_frames:
        dominant_indices = np.argmax(corr_matrix, axis=0)
        dominant_share = np.bincount(
            dominant_indices, weights=weights, minlength=len(_PROFILE_DATA)
        )
    else:
        dominant_share = np.zeros(len(_PROFILE_DATA), dtype=float)

    global_chroma = chroma @ weights

    candidates: list[KeyDetectionResult] = []
    for idx, profile in enumerate(_PROFILE_DATA):
        tonic_idx = NOTE_TO_INDEX[profile.note]
        tonic_strength = float(global_chroma[tonic_idx])
        dominant_strength = float(global_chroma[(tonic_idx + 7) % 12])
        major_third = float(global_chroma[(tonic_idx + 4) % 12])
        minor_third = float(global_chroma[(tonic_idx + 3) % 12])

        if profile.mode == "Maj":
            mode_bonus = major_third - minor_third
        else:
            mode_bonus = minor_third - major_third

        base_corr = float(weighted_corr[idx])
        support_value = float(positive_support[idx])
        dominant_value = float(dominant_share[idx])

        heuristics = (
            0.12 * tonic_strength
            + 0.08 * dominant_strength
            + 0.10 * mode_bonus
        )
        score = base_corr + 0.15 * support_value + 0.10 * dominant_value + heuristics

        corr_component = max(0.0, base_corr)
        corr_term = corr_component ** 0.4 if corr_component > 0 else 0.0
        confidence = min(
            1.0,
            corr_term * 0.6
            + support_value * 0.2
            + dominant_value * 0.1
            + tonic_strength * 0.05
            + max(0.0, mode_bonus) * 0.05,
        )

        candidates.append(
            KeyDetectionResult(
                note=profile.note,
                mode=profile.mode,
                correlation=base_corr,
                score=score,
                positive_support=support_value,
                dominant_share=dominant_value,
                confidence=confidence,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    top = candidates[:3]
    logger.debug(
        "Key candidates: %s",
        ", ".join(
            f"{c.note}{c.mode}: corr={c.correlation:.3f}, score={c.score:.3f}, conf={c.confidence:.2f}"
            for c in top
        ),
    )
    best = candidates[0]

    return best, candidates, tuning


def _find_close_key(
    best: KeyDetectionResult, candidates: Sequence[KeyDetectionResult]
) -> Optional[KeyDetectionResult]:
    if len(candidates) < 2:
        return None

    second = candidates[1]
    delta = best.score - second.score
    logger.debug(
        "Close key delta=%.5f (best=%s%s, second=%s%s)",
        delta,
        best.note,
        best.mode,
        second.note,
        second.mode,
    )
    if (
        best.confidence >= CLOSE_KEY_MIN_CONF
        and delta < CLOSE_KEY_THRESHOLD
        and second.confidence >= CLOSE_KEY_ALT_MIN_CONF
    ):
        return second
    return None


def _tuning_reference_hz(tuning_offset: float) -> float:
    """Convert tuning offset in semitones to the implied A4 frequency."""

    return 440.0 * (2.0 ** (tuning_offset / 12.0))


def _tone_frequency_hz(note: str, tuning_offset: float) -> float:
    """Return the tonic frequency (around the 4th octave) for the detected key."""

    if not note or note == "0":
        return 0.0

    base = float(librosa.note_to_hz(f"{note}4"))
    return base * (2.0 ** (tuning_offset / 12.0))


def _candidate_support(candidates: np.ndarray, target: float) -> int:
    if target <= 0 or candidates.size == 0:
        return 0
    return int(np.sum(np.abs(candidates - target) <= BPM_CANDIDATE_TOLERANCE))


def _refine_bpm_with_candidates(bpm: int, candidates: np.ndarray) -> int:
    if bpm <= 0:
        return 0
    if candidates.size == 0:
        return bpm

    finite = candidates[np.isfinite(candidates)]
    if finite.size == 0:
        return bpm

    primary_support = _candidate_support(finite, bpm)

    half_value = bpm / 2.0
    half_int = int(round(half_value))
    half_support = _candidate_support(finite, half_value)

    if (
        bpm >= BPM_DOUBLE_SWITCH
        and half_int >= BPM_MIN
        and half_support >= max(int(math.ceil(primary_support * BPM_SUPPORT_RATIO)), 1)
    ):
        logger.debug(
            "Refined BPM %s -> %s based on half-tempo support (%s vs %s)",
            bpm,
            half_int,
            half_support,
            primary_support,
        )
        return half_int

    double_value = bpm * 2.0
    double_int = int(round(double_value))
    double_support = _candidate_support(finite, double_value)

    if (
        bpm <= BPM_HALF_SWITCH
        and double_int <= BPM_MAX
        and double_support >= max(int(math.ceil(primary_support * BPM_SUPPORT_RATIO)), 1)
    ):
        logger.debug(
            "Refined BPM %s -> %s based on double-tempo support (%s vs %s)",
            bpm,
            double_int,
            double_support,
            primary_support,
        )
        return double_int

    return bpm


def _adjust_bpm(bpm: int, candidates: Optional[np.ndarray]) -> int:
    if bpm <= 0:
        return 0

    adjusted = bpm
    if bpm < BPM_MIN and bpm * 2 <= BPM_MAX:
        adjusted = int(round(bpm * 2))
        logger.debug("Doubling BPM %s -> %s (below minimum)", bpm, adjusted)
    elif bpm > BPM_MAX and (bpm / 2) >= BPM_MIN:
        adjusted = int(round(bpm / 2))
        logger.debug("Halving BPM %s -> %s (above maximum)", bpm, adjusted)

    if candidates is not None and candidates.size:
        refined = _refine_bpm_with_candidates(adjusted, candidates)
        if refined != adjusted:
            logger.debug("Refined BPM %s -> %s using candidate support", adjusted, refined)
            adjusted = refined

    return adjusted


def _estimate_bpm_from_signal(
    y_signal: Optional[np.ndarray], sr: int, *, source: str
) -> int:
    if y_signal is None or not np.any(y_signal):
        return 0

    candidates = librosa.beat.tempo(y=y_signal, sr=sr, aggregate=None)
    candidates_array = np.asarray(candidates, dtype=float)
    if candidates_array.size == 0:
        return 0

    finite_candidates = candidates_array[np.isfinite(candidates_array)]
    if finite_candidates.size == 0:
        return 0

    positive_candidates = finite_candidates[finite_candidates > 0]
    if positive_candidates.size == 0:
        return 0

    median_value = float(np.median(positive_candidates))
    if math.isnan(median_value) or median_value <= 0:
        return 0

    bpm_raw = int(round(median_value))
    if bpm_raw <= 0:
        return 0

    bpm = _adjust_bpm(bpm_raw, positive_candidates)
    if bpm != bpm_raw:
        logger.debug("Adjusted BPM %s -> %s for %s signal", bpm_raw, bpm, source)
    return bpm


def _detect_bpm(
    y_perc: np.ndarray,
    sr: int,
    *,
    y_harm: Optional[np.ndarray] = None,
    y_original: Optional[np.ndarray] = None,
) -> Tuple[int, int, int]:
    bpm = _estimate_bpm_from_signal(y_perc, sr, source="percussive")

    if bpm == 0 and y_harm is not None:
        logger.debug("No percussive BPM detected, analyzing harmonic component")
        bpm = _estimate_bpm_from_signal(y_harm, sr, source="harmonic")

    if bpm == 0 and y_original is not None:
        logger.debug("No harmonic BPM detected, analyzing full mix")
        bpm = _estimate_bpm_from_signal(y_original, sr, source="full")

    bpm_double = int(round(bpm * 2)) if bpm else 0
    bpm_half = int(round(bpm / 2)) if bpm else 0
    return bpm, bpm_double, bpm_half


def format_duration_seconds(total_seconds: float) -> str:
    """Return mm:ss formatted duration for a given number of seconds."""

    total_seconds_int = int(round(total_seconds))
    minutes, seconds = divmod(total_seconds_int, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_tone_display(note: str, mode: str) -> str:
    """Compose tone string with optional enharmonic alternative."""

    tone = f"{note}{mode}"
    enharmonic = ENHARMONIC.get(note)
    if enharmonic:
        return f"{tone} ({enharmonic}{mode})"
    return tone


__all__ = [
    "AnalysisResult",
    "AnalysisError",
    "analyze_file",
    "format_duration_seconds",
    "format_tone_display",
]
