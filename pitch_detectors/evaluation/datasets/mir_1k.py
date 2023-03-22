import os
from pathlib import Path

import numpy as np
from musiclib.pitch import Pitch
from scipy.io import wavfile

from pitch_detectors.evaluation.datasets.base import Dataset
from pitch_detectors.schemas import F0
from pitch_detectors.schemas import Wav


class Mir1K(Dataset):

    @classmethod
    def dataset_dir(cls) -> Path:
        return Path(os.environ.get('DATASET_DIR_MIR_1K', 'f0-datasets/mir-1k/MIR-1K'))

    @classmethod
    def wav_dir(cls) -> Path:
        return cls.dataset_dir() / 'Wavfile'

    @classmethod
    def load_wav(cls, wav_path: Path) -> Wav:
        fs, a = wavfile.read(wav_path)
        a = a[:, 1].astype(np.float32)
        return Wav(fs=fs, a=a)

    @classmethod
    def load_true(cls, wav_path: Path, seconds: float) -> F0:
        p = Pitch()
        pitch_label_dir = wav_path.parent.parent / 'PitchLabel'
        f0_path = (pitch_label_dir / wav_path.stem).with_suffix('.pv')
        f0 = []
        with open(f0_path) as f:
            for _line in f:
                line = _line.strip()
                if line == '0':
                    f0.append(float('nan'))
                else:
                    f0.append(p.note_i_to_hz(float(line)))
        f0 = np.array(f0)
        # t = np.arange(0.02, seconds - 0.02, 0.02)
        # assert t.shape == f0.shape
        t = np.linspace(0.02, seconds, len(f0))
        return F0(t=t, f0=f0)
