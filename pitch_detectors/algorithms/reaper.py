import numpy as np

from pitch_detectors import config
from pitch_detectors.algorithms.base import PitchDetector


class Reaper(PitchDetector):
    """https://github.com/r9y9/pyreaper"""

    def __init__(
        self,
        a: np.ndarray,
        fs: int,
        hz_min: float = config.HZ_MIN,
        hz_max: float = config.HZ_MAX,
    ):
        import pyreaper
        from dsplib.scale import minmax_scaler_array
        int16_info = np.iinfo(np.int16)
        a = minmax_scaler_array(a, np.min(a), np.max(a), int16_info.min, int16_info.max).round().astype(np.int16)
        super().__init__(a, fs)
        pm_times, pm, f0_times, f0, corr = pyreaper.reaper(self.a, fs=self.fs, minf0=hz_min, maxf0=hz_max, frame_period=0.01)
        f0[f0 == -1] = np.nan
        self.f0 = f0
        self.t = f0_times
