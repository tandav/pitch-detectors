import math


def nan_to_none(x: list[float]) -> list[float | None]:
    return [None if math.isnan(v) else v for v in x]


def none_to_nan(x: list[float | None]) -> list[float]:
    return [float('nan') if v is None else v for v in x]
