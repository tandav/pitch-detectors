FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# https://github.com/NVIDIA/nvidia-docker/wiki/Usage
# https://github.com/NVIDIA/nvidia-docker/issues/531
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility

RUN apt-get update && apt-get install -y python3-pip

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install tensorflow
