import numpy as np

from pitch_detectors.algorithms.base import PitchDetector


class PraatAC(PitchDetector):
    """https://parselmouth.readthedocs.io/en/stable/api_reference.html#parselmouth.Sound.to_pitch_ac"""

    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600):
        import parselmouth
        super().__init__(a, fs, hz_min, hz_max)
        self.signal = parselmouth.Sound(self.a, sampling_frequency=self.fs)
        self.pitch_obj = self.signal.to_pitch_ac(pitch_floor=self.hz_min, pitch_ceiling=self.hz_max, very_accurate=True)
        self.f0 = self.pitch_obj.selected_array['frequency']
        self.f0[self.f0 == 0] = np.nan
        self.t = self.pitch_obj.xs()
