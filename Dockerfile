# =============================================================================
# e-voice Dockerfile
# Single image for CPU + GPU (nvidia-container-toolkit injects CUDA at runtime)
# =============================================================================

# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:0.8-python3.12-bookworm AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY .git/ ./.git/
COPY uv.lock pyproject.toml README.md ./

RUN git config --global --add safe.directory /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

RUN rm -rf .venv/lib/python*/site-packages/pip* \
           .venv/lib/python*/site-packages/setuptools* \
           .venv/include

# -----------------------------------------------------------------------------
# Runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN groupadd -g 1000 app && \
    useradd -m -u 1000 -g app -d /app -s /bin/bash app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/pyproject.toml /app/pyproject.toml
COPY --from=builder --chown=app:app /app/README.md /app/README.md

RUN mkdir -p /app/data/models /app/data/config && \
    chown -R app:app /app/data

USER 1000

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 5500 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5500/health || exit 1

CMD ["python", "-m", "e_voice"]
