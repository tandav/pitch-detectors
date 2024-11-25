
ARG BASE_IMAGE_CUDA=scratch
ARG PYTHON_VERSION=3.12
ARG UV_IMAGE=ghcr.io/astral-sh/uv:latest
FROM python:${PYTHON_VERSION}-slim AS python
FROM ${UV_IMAGE} AS uv
FROM ${BASE_IMAGE_CUDA}
# disable banner from nvidia/cuda entrypoint
ENTRYPOINT []

# Copy Python binaries and libraries from the official Python image
COPY --from=python /usr/local /usr/local
ENV PATH="/usr/local/bin:$PATH"

COPY --from=uv /uv /uvx /bin/

WORKDIR /app
COPY pitch_detectors /app/pitch_detectors

ENV UV_LINK_MODE=copy \
    UV_FROZEN=1

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync

COPY tests /app/tests
COPY scripts/ /app/scripts
COPY data /app/data

ENV PATH="/app/.venv/bin:$PATH"
