# uv-based image; SPICE comes from the `spiceypy` wheel (no native build).
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

RUN adduser --disabled-password --gecos '' appuser
WORKDIR /app

# Install dependencies first for layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application code and kernel tooling.
COPY app ./app
COPY --chown=appuser:appuser kernels ./kernels

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app"

EXPOSE 8000
USER appuser

# Download SPICE kernels if they are not already present (idempotent), then
# serve. Mount a volume at /app/kernels to pre-bake them and skip the download.
CMD ["sh", "-c", "python -m kernels.bootstrap || true; exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app.main:app"]
