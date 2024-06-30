from typing import Any
from typing import TypeAlias

import numpy as np

from pitch_detectors.algorithms.base import PitchDetector
from pitch_detectors.algorithms.base import TensorflowGPU
from pitch_detectors.algorithms.base import TorchGPU
from pitch_detectors.schemas import F0

PDT: TypeAlias = type[PitchDetector]


def vote_and_median(
    algorithms: dict[str, F0],
    seconds: float,
    pitch_fs: int = 1024,
    min_duration: float = 1,
    min_algorithms: int = 3,
    # algorithm_weights: dict[str, float] = {},
) -> F0:
    if len(algorithms) < min_algorithms:
        raise ValueError(f'at least {min_algorithms} algorithms must be provided, because min_algorithms={min_algorithms}')

    single_n = int(seconds * pitch_fs)
    t_resampled = np.linspace(0, seconds, single_n)
    f0_resampled = {}
    F0_arr = np.empty((len(algorithms), single_n))
    for i, (name, data) in enumerate(algorithms.items()):
        t = data.t
        f0 = data.f0
        if len(f0) == 0:
            raise ValueError(f'algorithm {name} returned an empty f0 array')
        f0_resampled[name] = np.full_like(t_resampled, fill_value=np.nan)
        notna_slices = np.ma.clump_unmasked(np.ma.masked_invalid(f0))

        for sl in notna_slices:
            t_slice = t[sl]
            f0_slice = f0[sl]
            t_start, t_stop = t_slice[0], t_slice[-1]
            duration = t_stop - t_start
            if duration < min_duration:
                continue
            mask = (t_start < t_resampled) & (t_resampled < t_stop)
            t_interp = t_resampled[mask]
            f0_interp = np.interp(t_interp, t_slice, f0_slice)
            f0_resampled[name][mask] = f0_interp
        F0_arr[i] = f0_resampled[name]

    F0_mask = np.isfinite(F0_arr).astype(int)
    F0_mask_sum = F0_mask.sum(axis=0)
    min_alg_mask = F0_mask_sum > min_algorithms
    f0_mean = np.full_like(t_resampled, fill_value=np.nan)
    f0_mean[min_alg_mask] = np.nanmedian(F0_arr[:, min_alg_mask], axis=0)
    return F0(t=t_resampled, f0=f0_mean)


class Ensemble(TensorflowGPU, TorchGPU, PitchDetector):
    """https://github.com/tandav/pitch-detectors/blob/master/pitch_detectors/algorithms/ensemble.py"""

    def __init__(
        self,
        a: np.ndarray,
        fs: int,
        algorithms: tuple[PDT, ...] | None = None,
        algorithms_kwargs: dict[PDT, dict[str, Any]] | None = None,
        gpu: bool | None = None,
        vote_and_median_kwargs: dict[str, Any] | None = None,
    ):
        TensorflowGPU.__init__(self, gpu)
        TorchGPU.__init__(self, gpu)
        PitchDetector.__init__(self, a, fs)

        if algorithms is None:
            from pitch_detectors.algorithms import ALGORITHMS as algorithms_
        else:
            algorithms_ = algorithms

        self._algorithms = {}
        algorithms_kwargs = algorithms_kwargs or {}

        for cls in algorithms_:
            self._algorithms[cls] = cls(a, fs, **algorithms_kwargs.get(cls, {}))

        f0 = vote_and_median(
            {k.name(): F0(t=v.t, f0=v.f0) for k, v in self._algorithms.items()},
            self.seconds,
            **(vote_and_median_kwargs or {}),
        )
        self.t = f0.t
        self.f0 = f0.f0
