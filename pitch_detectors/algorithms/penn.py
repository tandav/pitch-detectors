import numpy as np

from pitch_detectors import config
from pitch_detectors.algorithms.base import PitchDetector
from pitch_detectors.algorithms.base import TorchGPU


class Penn(TorchGPU, PitchDetector):
    """https://github.com/interactiveaudiolab/penn"""

    def __init__(
        self,
        a: np.ndarray,
        fs: int,
        hz_min: float = config.HZ_MIN,
        hz_max: float = config.HZ_MAX,
        periodicity_threshold: float = 0.1,
        checkpoint: str = '/fcnf0++.pt',
    ):
        import torch
        from penn.core import from_audio
        super().__init__(a, fs)
        f0, periodicity = from_audio(
            audio=torch.tensor(a.reshape(1, -1)),
            sample_rate=fs,
            fmin=hz_min,
            fmax=hz_max,
            checkpoint=checkpoint,
        )
        periodicity = periodicity.numpy().ravel()
        f0 = f0.numpy().ravel()
        f0[periodicity < periodicity_threshold] = np.nan
        self.f0 = f0
        self.periodicity = periodicity
        self.t = np.linspace(0, self.seconds, self.f0.shape[0])
