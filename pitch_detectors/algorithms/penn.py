import numpy as np

from pitch_detectors.algorithms.base import PitchDetector
from pitch_detectors.algorithms.base import TorchGPU


class Penn(TorchGPU, PitchDetector):
    def __init__(
        self,
        a: np.ndarray,
        fs: int,
        hz_min: float = 75,
        hz_max: float = 600,
        periodicity_threshold: float = 0.1,
        checkpoint: str = '/fcnf0++.pt',
    ):
        import torch
        from penn.core import from_audio
        super().__init__(a, fs, hz_min, hz_max)
        f0, periodicity = from_audio(
            audio=torch.tensor(a.reshape(1, -1)),
            sample_rate=fs,
            fmin=self.hz_min,
            fmax=self.hz_max,
            checkpoint=checkpoint,
        )
        periodicity = periodicity.numpy().ravel()
        f0 = f0.numpy().ravel()
        f0[periodicity < periodicity_threshold] = np.nan
        self.f0 = f0
        self.periodicity = periodicity
        self.t = np.linspace(0, self.seconds, self.f0.shape[0] + 1)
