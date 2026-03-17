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

RUN rm -rf .venv/lib/python*/site-packages/{pip,setuptools}* .venv/include

FROM python:3.12-slim-bookworm

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libsndfile1 nginx && \
    rm -rf /var/lib/apt/lists/* /etc/nginx/sites-enabled/default

WORKDIR /app

RUN groupadd -g 1000 app && \
    useradd -m -u 1000 -g app -d /app -s /bin/bash app

COPY --from=builder --chown=app:app /app/.venv .venv/
COPY --from=builder --chown=app:app /app/src src/
COPY --from=builder --chown=app:app /app/pyproject.toml pyproject.toml
COPY --from=builder --chown=app:app /app/README.md README.md

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

RUN mkdir -p data/models data/config && chown -R app:app data/
COPY --chown=app:app data/config/config.yaml data/config/config.yaml
COPY --chown=app:app assets/ assets/
COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh

RUN chown -R app:app /var/log/nginx /var/lib/nginx /run

USER 1000

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
