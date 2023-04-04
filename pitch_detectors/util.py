import hashlib
import math
import sys
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
    base_bytes = base.read_bytes()
    hashes = {}
    for p in alg_dir.glob('*.py'):
        if p.name in {'__init__.py', 'base.py'}:
            continue
        h = hashlib.sha256()
        h.update(base_bytes)
        h.update(p.read_bytes())
        hashes[p.stem] = h.hexdigest()
    return hashes


def ld_library_path() -> str:
    site_packages = f'{sys.exec_prefix}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    libs = [
        f'{site_packages}/nvidia/curand/lib',
        f'{site_packages}/nvidia/cuda_runtime/lib',
        f'{site_packages}/nvidia/cusparse/lib',
        f'{site_packages}/nvidia/cudnn/lib',
        f'{site_packages}/nvidia/cuda_nvrtc/lib',
        f'{site_packages}/nvidia/cuda_cupti/lib',
        f'{site_packages}/nvidia/nccl/lib',
        f'{site_packages}/nvidia/cusolver/lib',
        f'{site_packages}/nvidia/nvtx/lib',
        f'{site_packages}/nvidia/cufft/lib',
        f'{site_packages}/nvidia/cublas/lib',
        f'{site_packages}/tensorrt',
    ]
    return ':'.join(libs)


if __name__ == '__main__':
    supported_actions = {'ld_library_path'}
    if len(sys.argv) != 2:  # noqa: PLR2004
        raise ValueError('Pass action as argument. Supported_actions:', supported_actions)
    if sys.argv[1] == 'ld_library_path':
        print(ld_library_path())
    else:
        raise ValueError(f'Action {sys.argv[1]} not supported. Supported_actions:', supported_actions)
