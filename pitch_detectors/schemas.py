import dataclasses

import numpy as np


@dataclasses.dataclass
class Wav:
    fs: int
    a: np.ndarray


@dataclasses.dataclass
class F0:
    t: np.ndarray
    f0: np.ndarray


@dataclasses.dataclass
class Record:
    fs: int | None = None
    a: np.ndarray | None = None
    t: np.ndarray | None = None
    f0: np.ndarray | None = None
