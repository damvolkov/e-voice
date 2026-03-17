# =============================================================================
# e-voice Dockerfile
# Single image for CPU + GPU (nvidia-container-toolkit injects CUDA at runtime)
# =============================================================================

# -----------------------------------------------------------------------------
# Builder — resolve deps + version from git
# -----------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:0.8-python3.12-bookworm AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY .git/ .git/
COPY uv.lock pyproject.toml README.md ./

RUN git config --global --add safe.directory /app

COPY src/ src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Bake version into package metadata so runtime doesn't need git
RUN .venv/bin/python -c "import importlib.metadata; print(importlib.metadata.version('e-voice'))" > /app/.version

# Trim build artifacts
RUN rm -rf .venv/lib/python*/site-packages/{pip,setuptools}* .venv/include

# -----------------------------------------------------------------------------
# Runtime — minimal, no git, no build tools
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN groupadd -g 1000 app && \
    useradd -m -u 1000 -g app -d /app -s /bin/bash app

COPY --from=builder --chown=app:app /app/.venv .venv/
COPY --from=builder --chown=app:app /app/src src/
COPY --from=builder --chown=app:app /app/pyproject.toml pyproject.toml
COPY --from=builder --chown=app:app /app/README.md README.md

RUN mkdir -p data/models data/config && chown -R app:app data/

USER 1000

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 5500 5600

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5500/health || exit 1

CMD ["python", "-m", "e_voice"]
