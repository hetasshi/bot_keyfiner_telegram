"""Key-detection templates and note metadata."""
from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

KRUMHANSL_MAJOR: Final[NDArray[np.float_]] = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float_,
)
KRUMHANSL_MINOR: Final[NDArray[np.float_]] = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float_,
)

NOTES_SHARP: Final[list[str]] = [
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

ENHARMONIC: Final[dict[str, str]] = {
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

MAJOR_PROFILES: Final[NDArray[np.float_]] = np.stack(
    [np.roll(KRUMHANSL_MAJOR, i) for i in range(len(NOTES_SHARP))]
)
MINOR_PROFILES: Final[NDArray[np.float_]] = np.stack(
    [np.roll(KRUMHANSL_MINOR, i) for i in range(len(NOTES_SHARP))]
)
