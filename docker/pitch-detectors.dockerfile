ARG BASE_IMAGE=scratch
ARG UV_IMAGE=ghcr.io/astral-sh/uv:latest
FROM ${UV_IMAGE} AS uv

FROM ${BASE_IMAGE} AS base

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
