"""Definitions of tonal profiles used by the key detection algorithm."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

KRUMHANSL_MAJOR: List[float] = [
    6.35,
    2.23,
    3.48,
    2.33,
    4.38,
    4.09,
    2.52,
    5.19,
    2.39,
    3.66,
    2.29,
    2.88,
]

KRUMHANSL_MINOR: List[float] = [
    6.33,
    2.68,
    3.52,
    5.38,
    2.60,
    3.53,
    2.54,
    4.75,
    3.98,
    2.69,
    3.34,
    3.17,
]

NOTES_SHARP: List[str] = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]

ENHARMONIC: Dict[str, str] = {
    "C#": "Db",
    "Db": "C#",
    "D#": "Eb",
    "Eb": "D#",
    "F#": "Gb",
    "Gb": "F#",
    "G#": "Ab",
    "Ab": "G#",
    "A#": "Bb",
    "Bb": "A#",
    "B": "Cb",
    "Cb": "B",
    "E": "Fb",
    "Fb": "E",
}

MAJOR_PROFILES = np.array(
    [np.roll(KRUMHANSL_MAJOR, i) for i in range(len(NOTES_SHARP))], dtype=float
)
MINOR_PROFILES = np.array(
    [np.roll(KRUMHANSL_MINOR, i) for i in range(len(NOTES_SHARP))], dtype=float
)

__all__ = [
    "KRUMHANSL_MAJOR",
    "KRUMHANSL_MINOR",
    "NOTES_SHARP",
    "ENHARMONIC",
    "MAJOR_PROFILES",
    "MINOR_PROFILES",
]
