import numpy as np

from pitch_detectors import config
from pitch_detectors.algorithms.base import PitchDetector


class TorchYin(PitchDetector):
    """https://github.com/brentspell/torch-yin"""

    uses_gpu_framework = True

    def __init__(
        self,
        a: np.ndarray,
        fs: int,
        hz_min: float = config.HZ_MIN,
        hz_max: float = config.HZ_MAX,
    ):
        import torch
        import torchyin
        super().__init__(a, fs)
        _a = torch.from_numpy(a)
        f0 = torchyin.estimate(_a, sample_rate=self.fs, pitch_min=hz_min, pitch_max=hz_max).numpy()
        f0[f0 == 0] = np.nan
        self.f0 = f0[:-1]
        self.t = np.linspace(0, self.seconds, f0.shape[0])[1:]
