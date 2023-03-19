import math
import typing as tp


def nan_to_none(x: list[float]) -> list[tp.Optional[float]]:
    return [None if math.isnan(v) else v for v in x]


def none_to_nan(x: list[tp.Optional[float]]) -> list[float]:
    return [float('nan') if v is None else v for v in x]
