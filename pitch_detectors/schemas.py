from typing import Any

import numpy as np
from pydantic import BaseModel
from pydantic import root_validator


class ArbitraryBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class Wav(ArbitraryBaseModel):
    fs: int
    a: np.ndarray


class F0(ArbitraryBaseModel):
    t: np.ndarray
    f0: np.ndarray

    @root_validator
    def check_shape(cls, values: dict[str, Any]) -> dict[str, Any]:  # pylint: disable=no-self-argument
        if values['t'].shape != values['f0'].shape:
            raise ValueError('t and f0 must have the same shape')
        return values


class Record(ArbitraryBaseModel):
    fs: int | None = None
    a: np.ndarray | None = None
    t: np.ndarray | None = None
    f0: np.ndarray | None = None
