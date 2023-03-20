import subprocess

import tensorflow as tf
import torch


def test_nvidia_smi():
    subprocess.check_call('nvidia-smi')


def test_tensorflow():
    assert tf.config.experimental.list_physical_devices('GPU')


def test_pytorch():
    assert torch.cuda.is_available()
