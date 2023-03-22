import numpy as np

from pitch_detectors.algorithms.base import PitchDetector


class Yaapt(PitchDetector):
    """http://bjbschmitt.github.io/AMFM_decompy/pYAAPT.html#amfm_decompy.pYAAPT.yaapt"""

    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600):
        import amfm_decompy.basic_tools as basic
        from amfm_decompy import pYAAPT
        super().__init__(a, fs, hz_min, hz_max)
        self.signal = basic.SignalObj(data=self.a, fs=self.fs)
        f0 = pYAAPT.yaapt(self.signal, f0_min=self.hz_min, f0_max=self.hz_max, frame_length=15)
        f0 = f0.samp_values
        f0[f0 == 0] = np.nan
        self.f0 = f0
        self.t = np.linspace(0, self.seconds, f0.shape[0])
