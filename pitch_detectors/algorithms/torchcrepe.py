import numpy as np

from pitch_detectors import config
from pitch_detectors.algorithms.base import PitchDetector
from pitch_detectors.algorithms.base import TorchGPU


class TorchCrepe(TorchGPU, PitchDetector):
    """https://github.com/maxrmorrison/torchcrepe"""

    def __init__(
        self,
        a: np.ndarray,
        fs: int,
        hz_min: float = config.HZ_MIN,
        hz_max: float = config.HZ_MAX,
        confidence_threshold: float = 0.8,
        batch_size: int = 2048,
        device: str | None = None,
    ):
        import torch
        import torchcrepe

        TorchGPU.__init__(self)
        PitchDetector.__init__(self, a, fs)

        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            torch.device(device)

        f0, confidence = torchcrepe.predict(
            torch.from_numpy(a[np.newaxis, ...]),
            fs,
            hop_length=int(fs / 100),  # 10 ms
            fmin=hz_min,
            fmax=hz_max,
            batch_size=batch_size,
            device=device,
            return_periodicity=True,
        )
        win_length = 3
        f0 = torchcrepe.filter.mean(f0, win_length)
        confidence = torchcrepe.filter.median(confidence, win_length)

        f0 = f0.ravel().numpy()
        confidence = confidence.ravel().numpy()
        f0[confidence < confidence_threshold] = np.nan
        self.f0 = f0
        self.t = np.linspace(0, self.seconds, f0.shape[0])
        torch.cuda.empty_cache()
