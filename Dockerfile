FROM alpine/curl as downloader
RUN curl -L https://tfhub.dev/google/spice/2?tf-hub-format=compressed --output spice_2.tar.gz && \
    mkdir /spice_model && \
    tar xvf spice_2.tar.gz --directory /spice_model && \
    rm spice_2.tar.gz && \
    curl -L https://huggingface.co/maxrmorrison/fcnf0-plus-plus/resolve/main/fcnf0%2B%2B.pt --output fcnf0++.pt

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# https://github.com/NVIDIA/nvidia-docker/wiki/Usage
# https://github.com/NVIDIA/nvidia-docker/issues/531
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility

COPY --from=downloader /spice_model /spice_model
COPY --from=downloader /fcnf0++.pt /fcnf0++.pt

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

COPY pitch_detectors /app/pitch_detectors
COPY tests /app/tests
COPY data /app/data
