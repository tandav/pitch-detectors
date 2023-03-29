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

    def __hash__(self) -> int:
        return hash((
            self.fs,
            self.a.tobytes() if self.a is not None else None,
            self.t.tobytes() if self.t is not None else None,
            self.f0.tobytes() if self.f0 is not None else None,
        ))
