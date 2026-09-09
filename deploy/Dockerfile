# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

# llama-cpp-python compiles against the host CPU here: the container runs on
# CPU (no Metal inside Docker on macOS), which is stated in the README because
# it makes the containerised numbers slower than the ones on the host.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md CMakeLists.txt ./
COPY src ./src
COPY bench ./bench

RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir \
      fastapi "uvicorn[standard]" httpx pydantic-settings numpy pandas pyarrow \
      scikit-learn matplotlib prometheus-client opentelemetry-sdk \
      opentelemetry-instrumentation-fastapi opentelemetry-exporter-otlp \
 && pip install --no-cache-dir llama-cpp-python

ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "cadence.api.app"]
