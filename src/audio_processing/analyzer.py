"""Core audio analysis routines."""
from __future__ import annotations

import os
import math
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import librosa
import librosa.util
import numpy as np
import soundfile as sf

try:  # Optional enhanced key detection backend
    import essentia.standard as es  # type: ignore

    _ESSENTIA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    es = None  # type: ignore
    _ESSENTIA_AVAILABLE = False

try:  # Optional enhanced tempo backend
    from madmom.audio.signal import Signal  # type: ignore
    from madmom.features.beats import RNNBeatProcessor  # type: ignore
    from madmom.features.tempo import TempoEstimationProcessor  # type: ignore

    _MADMOM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Signal = None  # type: ignore
    RNNBeatProcessor = None  # type: ignore
    TempoEstimationProcessor = None  # type: ignore
    _MADMOM_AVAILABLE = False

from .profiles import ENHARMONIC, MAJOR_PROFILES, MINOR_PROFILES, NOTES_SHARP

logger = logging.getLogger(__name__)

CLOSE_KEY_THRESHOLD = 0.12
CLOSE_KEY_MIN_CONF = 0.30
CLOSE_KEY_ALT_MIN_CONF = 0.20
POSITIVE_CORRELATION_THRESHOLD = 0.6
HARMONIC_HOP_LENGTH = 512
HARMONIC_RMS_VOID = 1e-5


_ESSENTIA_KEY_EXTRACTORS: dict[int, Any] = {}
_ESSENTIA_RHYTHM_EXTRACTORS: dict[int, Any] = {}
_MADMOM_BEAT_PROCESSOR: Optional[Any] = None
_MADMOM_TEMPO_PROCESSOR: Optional[Any] = None


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


CONF_VOID = _env_float("KEYFINDER_KEY_CONF_VOID", 0.05)
BPM_MIN = _env_int("BPM_MIN", 60)
BPM_MAX = _env_int("BPM_MAX", 180)
BPM_DOUBLE_SWITCH = _env_int("BPM_DOUBLE_SWITCH", 165)
BPM_HALF_SWITCH = _env_int("BPM_HALF_SWITCH", 65)
BPM_CANDIDATE_TOLERANCE = _env_float("BPM_CANDIDATE_TOLERANCE", 1.5)
BPM_SUPPORT_RATIO = _env_float("BPM_SUPPORT_RATIO", 0.9)

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


def _normalize_essentia_note(note: str) -> Optional[str]:
    """Map Essentia note naming to the sharp-based system used by the bot."""

    if not note:
        return None

    cleaned = note.strip().replace("♯", "#").replace("♭", "b")
    if not cleaned:
        return None

    base = cleaned[0].upper()
    accidental = ""
    if len(cleaned) > 1:
        symbol = cleaned[1]
        if symbol in {"#", "b"}:
            accidental = symbol

    normalized = f"{base}{accidental}"
    if normalized in NOTES_SHARP:
        return normalized

    enharmonic = ENHARMONIC.get(normalized)
    if enharmonic in NOTES_SHARP:
        return enharmonic
    return None


def _merge_key_detection(
    primary: KeyDetectionResult, secondary: KeyDetectionResult
) -> KeyDetectionResult:
    return KeyDetectionResult(
        note=primary.note,
        mode=primary.mode,
        correlation=max(primary.correlation, secondary.correlation),
        score=max(primary.score, secondary.score),
        positive_support=max(primary.positive_support, secondary.positive_support),
        dominant_share=max(primary.dominant_share, secondary.dominant_share),
        confidence=max(primary.confidence, secondary.confidence),
    )


def _get_essentia_key_extractor(sr: int) -> Optional[Any]:
    if not _ESSENTIA_AVAILABLE or es is None:
        return None

    extractor = _ESSENTIA_KEY_EXTRACTORS.get(sr)
    if extractor is not None:
        return extractor

    candidate_kwargs = [
        {
            "profileType": "temperley",
            "pcpSize": 36,
            "maxNumPeaks": 10,
            "windowSize": 0.5,
            "sampleRate": sr,
        },
        {
            "profileType": "temperley",
            "pcpSize": 36,
            "maxNumPeaks": 10,
            "sampleRate": sr,
        },
        {"profileType": "temperley", "pcpSize": 36, "sampleRate": sr},
        {"profileType": "temperley", "sampleRate": sr},
        {"profileType": "temperley"},
        {},
    ]

    for kwargs in candidate_kwargs:
        try:
            extractor = es.KeyExtractor(**kwargs)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug(
                "Failed to create Essentia KeyExtractor with %s: %s", kwargs, exc
            )
            continue
        _ESSENTIA_KEY_EXTRACTORS[sr] = extractor
        return extractor

    logger.warning(
        "Unable to initialise Essentia KeyExtractor for sample rate %s", sr
    )
    return None


def _get_essentia_rhythm_extractor(sr: int) -> Optional[Any]:
    if not _ESSENTIA_AVAILABLE or es is None:
        return None

    extractor = _ESSENTIA_RHYTHM_EXTRACTORS.get(sr)
    if extractor is not None:
        return extractor

    candidate_kwargs = [
        {
            "method": "multifeature",
            "minTempo": BPM_MIN,
            "maxTempo": BPM_MAX,
            "sampleRate": sr,
        },
        {
            "method": "multifeature",
            "minTempo": BPM_MIN,
            "maxTempo": BPM_MAX,
        },
        {"method": "multifeature"},
        {},
    ]

    for kwargs in candidate_kwargs:
        try:
            extractor = es.RhythmExtractor2013(**kwargs)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug(
                "Failed to create Essentia RhythmExtractor with %s: %s", kwargs, exc
            )
            continue
        _ESSENTIA_RHYTHM_EXTRACTORS[sr] = extractor
        return extractor

    logger.warning(
        "Unable to initialise Essentia RhythmExtractor for sample rate %s", sr
    )
    return None


def _get_madmom_processors() -> tuple[Optional[Any], Optional[Any]]:
    if not _MADMOM_AVAILABLE or RNNBeatProcessor is None or TempoEstimationProcessor is None:
        return None, None

    global _MADMOM_BEAT_PROCESSOR, _MADMOM_TEMPO_PROCESSOR

    if _MADMOM_BEAT_PROCESSOR is None:
        try:
            _MADMOM_BEAT_PROCESSOR = RNNBeatProcessor()
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug("Failed to initialise madmom beat processor: %s", exc)
            return None, None

    if _MADMOM_TEMPO_PROCESSOR is None:
        try:
            _MADMOM_TEMPO_PROCESSOR = TempoEstimationProcessor(fps=100)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug("Failed to initialise madmom tempo processor: %s", exc)
            return None, None

    return _MADMOM_BEAT_PROCESSOR, _MADMOM_TEMPO_PROCESSOR


def _detect_key_essentia(
    y_harm: np.ndarray, sr: int
) -> Optional[KeyDetectionResult]:
    if not _ESSENTIA_AVAILABLE or es is None:
        return None
    if y_harm.size == 0 or not np.any(y_harm):
        return None

    extractor = _get_essentia_key_extractor(sr)
    if extractor is None:
        return None

    audio = np.ascontiguousarray(y_harm, dtype=np.float32)
    try:
        raw_result = extractor(audio)
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"Essentia key extraction failed: {exc}") from exc

    key_raw: Any = None
    scale_raw: Any = "major"
    strength_raw: Any = 0.0

    if isinstance(raw_result, (tuple, list)):
        if raw_result:
            key_raw = raw_result[0]
        if len(raw_result) >= 2:
            scale_raw = raw_result[1]
        if len(raw_result) >= 3:
            strength_raw = raw_result[2]
    else:
        key_raw = raw_result

    if key_raw is None:
        return None

    key_text = str(key_raw)
    if " " in key_text:
        key_part = key_text.split()[0]
    else:
        key_part = key_text

    normalized_note = _normalize_essentia_note(key_part)
    if normalized_note is None:
        logger.debug("Unable to normalise Essentia key %s", key_text)
        return None

    mode_text = str(scale_raw).lower()
    mode = "Maj" if mode_text.startswith("maj") else "min"

    try:
        strength_value = float(strength_raw)
    except (TypeError, ValueError):
        strength_value = 0.0

    if math.isnan(strength_value):
        strength_value = 0.0

    strength_value = float(np.clip(strength_value, 0.0, 1.0))
    confidence = float(np.clip(strength_value * 1.25, 0.0, 1.0))

    logger.debug(
        "Essentia key candidate %s%s (strength %.3f, confidence %.3f)",
        normalized_note,
        mode,
        strength_value,
        confidence,
    )

    return KeyDetectionResult(
        note=normalized_note,
        mode=mode,
        correlation=strength_value,
        score=strength_value,
        positive_support=strength_value,
        dominant_share=strength_value,
        confidence=confidence,
    )


def _combine_key_candidates(
    fallback_best: KeyDetectionResult,
    fallback_candidates: Sequence[KeyDetectionResult],
    essentia_candidate: KeyDetectionResult,
) -> tuple[KeyDetectionResult, list[KeyDetectionResult]]:
    candidates = list(fallback_candidates)

    match_idx: Optional[int] = None
    for idx, candidate in enumerate(candidates):
        if candidate.note == essentia_candidate.note and candidate.mode == essentia_candidate.mode:
            match_idx = idx
            break

    if match_idx is not None:
        merged_candidate = _merge_key_detection(
            candidates[match_idx], essentia_candidate
        )
        candidates[match_idx] = merged_candidate
        essentia_candidate = merged_candidate
    else:
        candidates.insert(0, essentia_candidate)

    same_key = (
        fallback_best.note == essentia_candidate.note
        and fallback_best.mode == essentia_candidate.mode
    )

    if same_key:
        merged_best = _merge_key_detection(fallback_best, essentia_candidate)
        for idx, candidate in enumerate(candidates):
            if candidate.note == merged_best.note and candidate.mode == merged_best.mode:
                candidates[idx] = merged_best
                break
        return merged_best, candidates

    if fallback_best.confidence < CONF_VOID and essentia_candidate.confidence >= CONF_VOID:
        logger.debug(
            "Switching to Essentia key %s%s due to low confidence fallback (%.2f)",
            essentia_candidate.note,
            essentia_candidate.mode,
            fallback_best.confidence,
        )
        return essentia_candidate, candidates

    if essentia_candidate.confidence >= fallback_best.confidence + 0.1:
        logger.debug(
            "Switching to Essentia key %s%s (conf %.2f) over librosa %s%s (conf %.2f)",
            essentia_candidate.note,
            essentia_candidate.mode,
            essentia_candidate.confidence,
            fallback_best.note,
            fallback_best.mode,
            fallback_best.confidence,
        )
        return essentia_candidate, candidates

    return fallback_best, candidates


def _detect_key(
    y_harm: np.ndarray, sr: int
) -> Tuple[KeyDetectionResult, list[KeyDetectionResult], float]:
    best, candidates, tuning = _detect_key_librosa(y_harm, sr)

    if not _ESSENTIA_AVAILABLE or es is None:
        return best, candidates, tuning

    try:
        essentia_candidate = _detect_key_essentia(y_harm, sr)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("Essentia key detection failed: %s", exc)
        return best, candidates, tuning

    if essentia_candidate is None:
        return best, candidates, tuning

    best_combined, combined_candidates = _combine_key_candidates(
        best, candidates, essentia_candidate
    )
    return best_combined, combined_candidates, tuning


def _detect_bpm_madmom(y: np.ndarray, sr: int) -> Tuple[int, int, int]:
    if not _MADMOM_AVAILABLE or Signal is None:
        return 0, 0, 0
    if y.size == 0 or not np.any(y):
        return 0, 0, 0

    beat_processor, tempo_processor = _get_madmom_processors()
    if beat_processor is None or tempo_processor is None:
        return 0, 0, 0

    try:
        signal = Signal(y, sample_rate=sr)
        activations = beat_processor(signal)
        tempo_candidates = tempo_processor(activations)
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"madmom tempo estimation failed: {exc}") from exc

    tempo_array = np.asarray(tempo_candidates, dtype=float)
    if tempo_array.size == 0:
        return 0, 0, 0

    if tempo_array.ndim >= 2:
        bpm_value = float(tempo_array[0][0])
        bpm_candidates = tempo_array[:, 0]
    else:
        bpm_value = float(tempo_array[0])
        bpm_candidates = tempo_array

    if math.isnan(bpm_value) or bpm_value <= 0:
        return 0, 0, 0

    bpm_int = int(round(bpm_value))
    bpm_adjusted = _adjust_bpm(bpm_int, bpm_candidates)

    bpm = bpm_adjusted
    bpm_double = int(round(bpm * 2)) if bpm else 0
    bpm_half = int(round(bpm / 2)) if bpm else 0
    return bpm, bpm_double, bpm_half


def _detect_bpm_essentia(y: np.ndarray, sr: int) -> Tuple[int, int, int]:
    if not _ESSENTIA_AVAILABLE or es is None:
        return 0, 0, 0
    if y.size == 0 or not np.any(y):
        return 0, 0, 0

    extractor = _get_essentia_rhythm_extractor(sr)
    if extractor is None:
        return 0, 0, 0

    audio = np.ascontiguousarray(y, dtype=np.float32)
    try:
        raw_result = extractor(audio)
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"Essentia tempo extraction failed: {exc}") from exc

    bpm_value = 0.0
    beat_intervals: Optional[np.ndarray] = None

    if isinstance(raw_result, (tuple, list)):
        if raw_result:
            bpm_value = float(raw_result[0])
        if len(raw_result) >= 5:
            beat_intervals = np.asarray(raw_result[4], dtype=float)
        elif len(raw_result) >= 3 and isinstance(raw_result[2], (list, np.ndarray)):
            beat_intervals = np.asarray(raw_result[2], dtype=float)
    else:
        bpm_value = float(raw_result)

    if math.isnan(bpm_value) or bpm_value <= 0:
        return 0, 0, 0

    bpm_int = int(round(bpm_value))
    tempo_candidates: Optional[np.ndarray] = None

    if beat_intervals is not None and beat_intervals.size:
        with np.errstate(divide="ignore", invalid="ignore"):
            tempo_candidates = 60.0 / beat_intervals
        if tempo_candidates is not None:
            tempo_candidates = tempo_candidates[np.isfinite(tempo_candidates)]
            if tempo_candidates.size == 0:
                tempo_candidates = None

    bpm_adjusted = _adjust_bpm(bpm_int, tempo_candidates)

    bpm = bpm_adjusted
    bpm_double = int(round(bpm * 2)) if bpm else 0
    bpm_half = int(round(bpm / 2)) if bpm else 0
    return bpm, bpm_double, bpm_half


def _detect_bpm(
    y_perc: np.ndarray,
    sr: int,
    *,
    y_harm: Optional[np.ndarray] = None,
    y_original: Optional[np.ndarray] = None,
) -> Tuple[int, int, int]:
    if _MADMOM_AVAILABLE and Signal is not None and y_original is not None and y_original.size:
        try:
            bpm, bpm_double, bpm_half = _detect_bpm_madmom(y_original, sr)
            if bpm:
                return bpm, bpm_double, bpm_half
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug("Madmom BPM detection failed: %s", exc)

    if _ESSENTIA_AVAILABLE and es is not None:
        target = None
        if y_perc is not None and y_perc.size:
            target = y_perc
        elif y_original is not None and y_original.size:
            target = y_original
        elif y_harm is not None and y_harm.size:
            target = y_harm

        if target is not None:
            try:
                bpm, bpm_double, bpm_half = _detect_bpm_essentia(target, sr)
                if bpm:
                    return bpm, bpm_double, bpm_half
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.debug("Essentia BPM detection failed: %s", exc)

    return _detect_bpm_librosa(y_perc, sr, y_harm=y_harm, y_original=y_original)


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


def _detect_key_librosa(
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


def _detect_bpm_librosa(
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
