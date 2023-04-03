from pathlib import Path

import pytest

from pitch_detectors import util
from pitch_detectors.schemas import Record


@pytest.fixture
def record():
    fs, a = util.load_wav(Path(__file__).parent.parent / 'data' / 'b1a5da49d564a7341e7e1327aa3f229a.wav')
    return Record(fs, a)


@pytest.fixture
def environ():
    return {
        'PITCH_DETECTORS_GPU_MEMORY_LIMIT': 'true',
        'PITCH_DETECTORS_PENN_CHECKPOINT_PATH': '/home/tandav/docs/bhairava/libmv/data/fcnf0++.pt',
        'PITCH_DETECTORS_SPICE_MODEL_PATH': '/home/tandav/docs/bhairava/libmv/data/spice_model',
        'LD_LIBRARY_PATH': '/home/tandav/.virtualenvs/pitch-detectors/lib/python3.10/site-packages/nvidia/curand/lib:/home/tandav/.virtualenvs/pitch-detectors/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/tandav/.virtualenvs/pitch-detectors/lib/python3.10/site-packages/nvidia/cusparse/lib:/home/tandav/.virtualenvs/pitch-detectors/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/tandav/.virtualenvs/pitch-detectors/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/home/tandav/.virtualenvs/pitch-detectors/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:/home/tandav/.virtualenvs/pitch-detectors/lib/python3.10/site-packages/nvidia/nccl/lib:/home/tandav/.virtualenvs/pitch-detectors/lib/python3.10/site-packages/nvidia/cusolver/lib:/home/tandav/.virtualenvs/pitch-detectors/lib/python3.10/site-packages/nvidia/nvtx/lib:/home/tandav/.virtualenvs/pitch-detectors/lib/python3.10/site-packages/nvidia/cufft/lib:/home/tandav/.virtualenvs/pitch-detectors/lib/python3.10/site-packages/nvidia/cublas/lib',
    }
