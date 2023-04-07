import numpy as np
from pydantic import BaseModel


class ArbitraryBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class Wav(ArbitraryBaseModel):
    fs: int
    a: np.ndarray


class F0(ArbitraryBaseModel):
    t: np.ndarray
    f0: np.ndarray


class Record(ArbitraryBaseModel):
    fs: int | None = None
    a: np.ndarray | None = None
    t: np.ndarray | None = None
    f0: np.ndarray | None = None
