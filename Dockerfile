# =============================================================================
# e-voice Dockerfile
# Single image for CPU + GPU (nvidia-container-toolkit injects CUDA at runtime)
# =============================================================================

# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:0.8-python3.13-bookworm AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY .git/ ./.git/
COPY uv.lock pyproject.toml ./

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
FROM ghcr.io/astral-sh/uv:0.8-python3.13-bookworm-slim

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
COPY --from=builder --chown=app:app /app/.git /app/.git
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/pyproject.toml /app/pyproject.toml
COPY --from=builder --chown=app:app /app/uv.lock /app/uv.lock

RUN mkdir -p /app/models && chown app:app /app/models

USER 1000

RUN git config --global --add safe.directory /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV WHISPER_DEVICE="cuda"
ENV WHISPER_COMPUTE_TYPE="default"
ENV WHISPER_MODEL="mobiuslabsgmbh/faster-whisper-large-v3-turbo"
ENV MODELS_PATH="/app/models"

EXPOSE 5500

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5500/health || exit 1

CMD ["python", "-m", "e_voice"]
