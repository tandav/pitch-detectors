# FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04 - SUCCESS
# FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04 - FAIL
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# https://github.com/NVIDIA/nvidia-docker/wiki/Usage
# https://github.com/NVIDIA/nvidia-docker/issues/531
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.10-venv libsndfile-dev libasound-dev portaudio19-dev

# https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV VIRTUAL_ENV=/venv
RUN python3.10 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install .[dev]
