from typing import NamedTuple

import numpy as np


class Wav(NamedTuple):
    fs: int
    a: np.ndarray


class F0(NamedTuple):
    t: np.ndarray
    f0: np.ndarray


class Record(NamedTuple):
    fs: int | None = None
    a: np.ndarray | None = None
    t: np.ndarray | None = None
    f0: np.ndarray | None = None
