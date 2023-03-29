import numpy as np

from pitch_detectors import config
from pitch_detectors.algorithms.base import PitchDetector


class PraatSHS(PitchDetector):
    """https://parselmouth.readthedocs.io/en/stable/api_reference.html#parselmouth.Sound.to_pitch_shs"""

    def __init__(
        self,
        a: np.ndarray,
        fs: int,
        hz_min: float = config.HZ_MIN,
        hz_max: float = config.HZ_MAX,
    ):
        import parselmouth
        super().__init__(a, fs)
        self.signal = parselmouth.Sound(self.a, sampling_frequency=self.fs)
        self.pitch_obj = self.signal.to_pitch_shs(minimum_pitch=hz_min, maximum_frequency_component=hz_max)
        self.f0 = self.pitch_obj.selected_array['frequency']
        self.f0[self.f0 == 0] = np.nan
        self.t = self.pitch_obj.xs()
