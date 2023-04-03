import os

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
        checkpoint: str | None = None,
        gpu: bool | None = None,
    ):
        import torch
        from penn.core import from_audio

        TorchGPU.__init__(self, gpu)
        PitchDetector.__init__(self, a, fs)

        if checkpoint is None:
            checkpoint = os.environ.get('PITCH_DETECTORS_PENN_CHECKPOINT_PATH', '/fcnf0++.pt')

        f0, periodicity = from_audio(
            audio=torch.tensor(a.reshape(1, -1)),
            sample_rate=fs,
            fmin=hz_min,
            fmax=hz_max,
            checkpoint=checkpoint,
            gpu=0 if self.gpu else None,
        )
        if self.gpu:
            f0 = f0.cpu()
            periodicity = periodicity.cpu()
        periodicity = periodicity.numpy().ravel()
        f0 = f0.numpy().ravel()
        f0[periodicity < periodicity_threshold] = np.nan
        self.f0 = f0
        self.periodicity = periodicity
        self.t = np.linspace(0, self.seconds, self.f0.shape[0])
        torch.cuda.empty_cache()
