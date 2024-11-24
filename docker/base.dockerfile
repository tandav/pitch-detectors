ARG BASE_IMAGE_CUDA=scratch
FROM ${BASE_IMAGE_CUDA}
# https://github.com/NVIDIA/nvidia-docker/wiki/Usage
# https://github.com/NVIDIA/nvidia-docker/issues/531
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
ENV DEBIAN_FRONTEND=noninteractive

ARG PYTHON_VERSION=3.12

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION}-dev \
    libsndfile-dev \
    libasound-dev \
    portaudio19-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
