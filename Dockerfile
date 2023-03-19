# FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04

# https://github.com/NVIDIA/nvidia-docker/wiki/Usage
# https://github.com/NVIDIA/nvidia-docker/issues/531
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility

# RUN apt-get update && apt-get install -y python3-pip
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.10-venv libsndfile-dev libasound-dev portaudio19-dev

#  - apt-get install -y software-properties-common curl git openjdk-8-jdk make libgomp1
#     - add-apt-repository -y ppa:deadsnakes/ppa
#     - apt-get install -y python3.7-venv python3.8-venv python3.9-venv python3.10-venv


# https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV VIRTUAL_ENV=/venv
RUN python3.10 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install tensorflow

WORKDIR /app
COPY pyproject.toml /app/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install .[dev]

# COPY libmv /app/libmv
# COPY scripts /app/scripts
