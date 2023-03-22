import numpy as np

from pitch_detectors.algorithms.base import PitchDetector
from pitch_detectors.algorithms.base import TorchGPU


class TorchCrepe(TorchGPU, PitchDetector):
    def __init__(
        self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600, confidence_threshold: float = 0.8,
        batch_size: int = 2048,
        device: str | None = None,
    ):
        import torch
        import torchcrepe
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            torch.device(device)

        super().__init__(a, fs, hz_min, hz_max)

        f0, confidence = torchcrepe.predict(
            torch.from_numpy(a[np.newaxis, ...]),
            fs,
            hop_length=int(fs / 100),  # 10 ms
            fmin=self.hz_min,
            fmax=self.hz_max,
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
