# syntax=docker/dockerfile:1
# Two-stage uv image. The builder compiles the native core (`_eclipse`:
# vendored CSPICE + ERFA through scikit-build-core / CMake) into a wheel on the
# full bookworm image, which has a C/C++ toolchain (cmake and ninja come from
# PyPI wheels); the runtime stays the slim image plus libgomp1 (the OpenMP
# runtime the core's parallel loops link) and installs that wheel into the
# locked venv. SPICE for the Python oracle comes from the `spiceypy` wheel.

# ---- builder: `uv build --wheel` of this project (only the native inputs)
FROM ghcr.io/astral-sh/uv:python3.11-bookworm AS builder
WORKDIR /src
# ccache keeps the ~2,200 CSPICE translation units across image rebuilds
# (CMakeLists.txt picks it up with find_program).
RUN apt-get update && apt-get install -y --no-install-recommends ccache \
    && rm -rf /var/lib/apt/lists/*
ENV CCACHE_DIR=/root/.cache/ccache
COPY pyproject.toml uv.lock README.md CMakeLists.txt CMakePresets.json ./
COPY cmake ./cmake
COPY core ./core
COPY bindings ./bindings
COPY third_party ./third_party
RUN --mount=type=cache,target=/root/.cache/ccache \
    uv build --wheel --out-dir /dist

# ---- runtime
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos '' appuser
WORKDIR /app

# Install dependencies first for layer caching (`--no-install-project` skips
# the scikit-build-core build here; the wheel from the builder replaces it).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY --from=builder /dist/*.whl /tmp/wheels/
RUN uv pip install --python /app/.venv/bin/python --no-deps /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels \
    && /app/.venv/bin/python -c "import _eclipse as e; print(e.toolkit_version(), e.erfa_version())"

# Application code, frontend and kernel tooling.
COPY app ./app
COPY frontend ./frontend
COPY --chown=appuser:appuser kernels ./kernels

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app"

EXPOSE 8000
USER appuser

# Download SPICE kernels if they are not already present (idempotent), then
# serve. Mount a volume at /app/kernels to pre-bake them and skip the download.
# The API computes with the native core (the wheel above). Do NOT add
# `--preload` to gunicorn: libgomp is not fork-safe
# once its thread pool has started, and a preloaded parent that warmed the
# native grid or catalog would hand forked workers a dead pool.
#
# OpenMP team size: libgomp defaults to the host's core count (it ignores a
# cgroup CPU quota) and every worker runs its own team, so split the cores
# between the workers unless OMP_NUM_THREADS is set (at least 1). Idle teams
# sleep rather than spin (OMP_WAIT_POLICY). WEB_CONCURRENCY is gunicorn's own
# worker-count variable. See app/core.py PARALLEL_LOCK.
ENV WEB_CONCURRENCY=4 \
    OMP_WAIT_POLICY=PASSIVE
CMD ["sh", "-c", "python -m kernels.bootstrap || true; : \"${OMP_NUM_THREADS:=$(( $(nproc) / WEB_CONCURRENCY > 0 ? $(nproc) / WEB_CONCURRENCY : 1 ))}\"; export OMP_NUM_THREADS; exec gunicorn -w \"$WEB_CONCURRENCY\" -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app.main:app"]
