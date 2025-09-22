"""Key detection profiles and note utilities."""

from __future__ import annotations

import numpy as np

KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

NOTES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

ENHARMONIC = {
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
}

MAJOR_PROFILES = [np.roll(KRUMHANSL_MAJOR, i) for i in range(len(NOTES_SHARP))]
MINOR_PROFILES = [np.roll(KRUMHANSL_MINOR, i) for i in range(len(NOTES_SHARP))]

__all__ = [
    "KRUMHANSL_MAJOR",
    "KRUMHANSL_MINOR",
    "MAJOR_PROFILES",
    "MINOR_PROFILES",
    "NOTES_SHARP",
    "ENHARMONIC",
]
