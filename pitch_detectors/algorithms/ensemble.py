from typing import Any

import numpy as np

from pitch_detectors.algorithms.base import PitchDetector


class Ensemble(PitchDetector):
    """https://github.com/tandav/pitch-detectors/blob/master/pitch_detectors/algorithms/ensemble.py"""

    def __init__(
        self,
        a: np.ndarray,
        fs: int,
        pitch_fs: int = 1024,
        min_duration: float = 1,
        min_algorithms: int = 3,
        algorithms: tuple[type[PitchDetector], ...] | None = None,
        algorithms_kwargs: dict[type[PitchDetector], dict[str, Any]] | None = None,
        algorithms_cache: dict[type[PitchDetector], PitchDetector] | None = None,
        # algorithm_weights: dict[type[PitchDetector], float] = {},
    ):
        super().__init__(a, fs)

        if algorithms_cache is not None:
            if (algorithms is not None or algorithms_kwargs is not None):
                raise ValueError('algorithms and algorithms_kwargs cannot be used with algorithms_cache')
            self._algorithms = algorithms_cache
        elif algorithms is None:
            raise ValueError('algorithms or algorithms_cache must be provided')
        else:
            self._algorithms = {}
            algorithms_kwargs = algorithms_kwargs or {}

            for algorithm_cls in algorithms:
                self._algorithms[algorithm_cls] = algorithm_cls(a, fs, **algorithms_kwargs.get(algorithm_cls, {}))
        single_n = int(self.seconds * pitch_fs)
        t_resampled = np.linspace(0, self.seconds, single_n)
        f0_resampled = {}
        F0 = np.empty((len(self._algorithms), single_n))
        for i, (alg_cls, algorithm) in enumerate(self._algorithms.items()):
            t = algorithm.t
            f0 = algorithm.f0
            f0_resampled[alg_cls] = np.full_like(t_resampled, fill_value=np.nan)
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
                f0_resampled[alg_cls][mask] = f0_interp
            F0[i] = f0_resampled[alg_cls]

        F0_mask = np.isfinite(F0).astype(int)
        F0_mask_sum = F0_mask.sum(axis=0)
        min_alg_mask = F0_mask_sum > min_algorithms
        f0_mean = np.full_like(t_resampled, fill_value=np.nan)
        f0_mean[min_alg_mask] = np.nanmedian(F0[:, min_alg_mask], axis=0)
        self.t = t_resampled
        self.f0 = f0_mean
