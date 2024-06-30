ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /app
COPY pyproject.toml .
ENV PIP_INDEX_URL=https://pypi.tandav.me/index/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install .[dev]

COPY pitch_detectors /app/pitch_detectors

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps .

COPY tests /app/tests
COPY scripts/ /app/scripts
COPY data /app/data
