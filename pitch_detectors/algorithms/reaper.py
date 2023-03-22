import numpy as np

from pitch_detectors.algorithms.base import PitchDetector


class Reaper(PitchDetector):
    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600):
        import pyreaper
        from dsplib.scale import minmax_scaler
        int16_info = np.iinfo(np.int16)
        a = minmax_scaler(a, np.min(a), np.max(a), int16_info.min, int16_info.max).round().astype(np.int16)
        super().__init__(a, fs, hz_min, hz_max)
        pm_times, pm, f0_times, f0, corr = pyreaper.reaper(self.a, fs=self.fs, minf0=self.hz_min, maxf0=self.hz_max, frame_period=0.01)
        f0[f0 == -1] = np.nan
        self.f0 = f0
        self.t = f0_times
