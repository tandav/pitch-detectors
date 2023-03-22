import hashlib
import math
from pathlib import Path

import numpy as np
from dsplib.scale import minmax_scaler
from scipy.io import wavfile


def nan_to_none(x: list[float]) -> list[float | None]:
    return [None if math.isnan(v) else v for v in x]


def none_to_nan(x: list[float | None]) -> list[float]:
    return [float('nan') if v is None else v for v in x]


def load_wav(path: Path | str, rescale: float = 100000) -> tuple[int, np.ndarray]:
    fs, a = wavfile.read(path)
    a = minmax_scaler(a, a.min(), a.max(), -rescale, rescale).astype(np.float32)
    return fs, a


def source_hashes() -> dict[str, str]:
    alg_dir = Path(__file__).parent / 'algorithms'
    base = alg_dir / 'base.py'
    base_hash = hashlib.sha256(base.read_bytes()).digest()
    hashes = {}
    for p in alg_dir.glob('*.py'):
        if p.name in {'__init__.py', 'base.py'}:
            continue
        h = hashlib.sha256()
        h.update(base_hash)
        h.update(p.read_bytes())
        hashes[p.stem] = h.hexdigest()
    return hashes
