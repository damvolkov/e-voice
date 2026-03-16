# =============================================================================
# robyn-ml-api Dockerfile
# Multi-stage build for minimal production image
# =============================================================================

# -----------------------------------------------------------------------------
# Builder Stage
# -----------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:0.8-python3.12-bookworm AS builder

ARG ENABLE_DEBUG=false

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

COPY uv.lock pyproject.toml README.md ./
COPY .git/ ./.git/

RUN git config --global --add safe.directory /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY app/ ./app/

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$ENABLE_DEBUG" = "true" ]; then \
        uv sync --frozen && uv pip install debugpy; \
    else \
        uv sync --frozen --no-dev; \
    fi

RUN rm -rf .venv/lib/python*/site-packages/pip* \
           .venv/lib/python*/site-packages/setuptools* \
           .venv/include && \
    echo "✅ Virtual environment ready:" && du -sh .venv

# -----------------------------------------------------------------------------
# Runtime Stage
# -----------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:0.8-python3.12-bookworm

ARG ENABLE_DEBUG=false

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

RUN groupadd -g 1000 appuser && \
    useradd -m -u 1000 -g appuser -d /app -s /bin/bash appuser && \
    chown -R appuser:appuser /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/.git /app/.git
COPY --from=builder --chown=appuser:appuser /app/app /app/app
COPY --from=builder --chown=appuser:appuser /app/pyproject.toml /app/pyproject.toml
COPY --from=builder --chown=appuser:appuser /app/uv.lock /app/uv.lock
COPY --from=builder --chown=appuser:appuser /app/README.md /app/README.md

USER 1000

RUN git config --global --add safe.directory /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENABLE_DEBUG=${ENABLE_DEBUG}

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD if [ "$ENABLE_DEBUG" = "true" ]; then \
        python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m app.main; \
    else \
        python -m app.main; \
    fi
