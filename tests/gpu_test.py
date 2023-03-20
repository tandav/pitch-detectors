import subprocess

import pytest
import tensorflow as tf
import torch


@pytest.mark.order(0)
def test_nvidia_smi():
    subprocess.check_call('nvidia-smi')


@pytest.mark.order(1)
def test_tensorflow():
    assert tf.config.experimental.list_physical_devices('GPU')


@pytest.mark.order(2)
def test_pytorch():
    assert torch.cuda.is_available()
