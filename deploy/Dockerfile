# syntax=docker/dockerfile:1
#
# Three stages, for three different reasons.
#
#   weights  the GGUF, fetched once at build time and checksummed. A model
#            pulled on boot is a cold start that scales with the network, and
#            a cold start is a latency outlier in exactly the tail this
#            project claims to control. Baked, or not deployed.
#   build    the wheel, including the C++17 KV core. Compiled here so the
#            runtime image carries no compiler.
#   runtime  Python, the wheel, the weights. Nothing else.
#
# Build from the repository root:
#   docker build -f deploy/Dockerfile -t cadence .

# --- weights ---------------------------------------------------------------
FROM debian:bookworm-slim AS weights

ARG MODEL_FILE=qwen2.5-0.5b-instruct-q4_k_m.gguf
ARG MODEL_URL=https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
# Pinned, and verified below. "resolve/main" is a moving reference: without
# the checksum the image would be reproducible in name only, and a silently
# requantised model would show up as an unexplained shift in every latency
# number the deployment reports.
ARG MODEL_SHA256=74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /weights
RUN curl -fL --retry 5 --retry-delay 2 -o "${MODEL_FILE}" "${MODEL_URL}" \
    && echo "${MODEL_SHA256}  ${MODEL_FILE}" | sha256sum -c -

# --- build -----------------------------------------------------------------
FROM python:3.11-slim AS build

# llama-cpp-python compiles against the container's CPU here: there is no
# Metal inside Docker, so a containerised gateway is materially slower than
# the host numbers in the README. That is stated there rather than papered
# over -- the deployment exists to be a working deployment, not to set the
# record.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PROJECT_ENVIRONMENT=/venv

# The lock file first, so a source-only change does not reinstall the world.
COPY pyproject.toml uv.lock README.md CMakeLists.txt ./
RUN uv sync --frozen --no-install-project --no-dev

# Then the project itself, which is what builds the C++17 extension.
COPY src ./src
RUN uv sync --frozen --no-dev

# The extension must exist. Without this line the image would still start,
# fall back to the Python reference, and quietly serve different code from
# the one the A/B in the README measured.
RUN /venv/bin/python -c "import cadence._core; from cadence.engine.kv import core_name; \
assert core_name('auto') == 'cpp', 'the C++ core did not build'; print('kv core: cpp')"

# --- runtime ---------------------------------------------------------------
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 cadence

COPY --from=build /venv /venv
COPY --from=weights /weights/qwen2.5-0.5b-instruct-q4_k_m.gguf /app/models/
COPY --chown=cadence:cadence models/admission.pkl models/admission.json /app/models/
COPY deploy/entrypoint.sh /app/entrypoint.sh
RUN chmod 0555 /app/entrypoint.sh

WORKDIR /app
USER cadence
ENV PATH=/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    CADENCE_HOST=0.0.0.0 \
    CADENCE_PORT=8000 \
    CADENCE_MODEL_PATH=/app/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    CADENCE_ADMISSION_MODEL=/app/models/admission.pkl

EXPOSE 8000

# Liveness only. Readiness is /ready, and the difference is the point: this
# container is alive within a second of starting and must not be sent traffic
# for the tens of seconds the model takes to load. See deploy/fly.toml, where
# the two are wired to different checks.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=6 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# SIGTERM reaches the server rather than a shell: exec form, no shell wrapper.
# The drain in cadence.api.app depends on receiving that signal.
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["cadence-serve"]
