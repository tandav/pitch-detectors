import numpy as np

from pitch_detectors.algorithms.base import PitchDetector


class TorchYin(PitchDetector):
    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600):
        import torch
        import torchyin
        super().__init__(a, fs, hz_min, hz_max)
        _a = torch.from_numpy(a)
        f0 = torchyin.estimate(_a, sample_rate=self.fs, pitch_min=self.hz_min, pitch_max=self.hz_max)
        f0[f0 == 0] = np.nan
        self.f0 = f0[:-1]
        self.t = np.linspace(0, self.seconds, f0.shape[0])[1:]
