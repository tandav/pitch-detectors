import numpy as np
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import model_validator
from typing_extensions import Self


class ArbitraryBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Wav(ArbitraryBaseModel):
    fs: int
    a: np.ndarray


class F0(ArbitraryBaseModel):
    t: np.ndarray
    f0: np.ndarray

    @model_validator(mode='after')
    def check_shape(self) -> Self:
        if self.t.shape != self.f0.shape:
            raise ValueError('t and f0 must have the same shape')
        return self


class Record(ArbitraryBaseModel):
    fs: int | None = None
    a: np.ndarray | None = None
    t: np.ndarray | None = None
    f0: np.ndarray | None = None
