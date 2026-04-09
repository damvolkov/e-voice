FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git libsndfile1 nginx && \
    rm -rf /var/lib/apt/lists/* /etc/nginx/sites-enabled/default

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /usr/local/bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-system

COPY .git/ .git/
COPY uv.lock pyproject.toml README.md ./

RUN git config --global --add safe.directory /app

COPY src/ src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group qwen \
        --no-install-package torch \
        --no-install-package torchaudio \
        --no-install-package triton \
        --no-install-package nvidia-cuda-runtime-cu13 \
        --no-install-package nvidia-cublas-cu13 \
        --no-install-package nvidia-cudnn-cu13 \
        --no-install-package nvidia-cufft-cu13 \
        --no-install-package nvidia-curand-cu13 \
        --no-install-package nvidia-cusolver-cu13 \
        --no-install-package nvidia-cusparse-cu13 \
        --no-install-package nvidia-nccl-cu13 \
        --no-install-package nvidia-nvjitlink-cu13 \
        --no-install-package nvidia-nvtx-cu13 && \
    rm -rf .venv/lib/python*/site-packages/{pip,setuptools}* .venv/include && \
    echo "include-system-site-packages = true" >> .venv/pyvenv.cfg

RUN usermod -d /app ubuntu 2>/dev/null; mkdir -p /app && chown 1000:1000 /app

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

RUN mkdir -p data/models/stt data/models/tts/kokoro/models data/models/tts/qwen/models data/models/tts/qwen/voices \
    data/config /defaults .cache/huggingface && chown -R 1000:1000 data/ /defaults .cache/
COPY --chown=1000:1000 data/config/config.yaml /defaults/config.yaml
COPY --chown=1000:1000 data/config/config.yaml data/config/config.yaml
COPY --chown=1000:1000 data/models/tts/qwen/voices/ data/models/tts/qwen/voices/
COPY --chown=1000:1000 data/models/tts/qwen/reference_samples/ data/models/tts/qwen/reference_samples/
COPY --chown=1000:1000 assets/ assets/
COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh

RUN chown -R 1000:1000 /var/log/nginx /var/lib/nginx /run

USER 1000

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME="/app/.cache/huggingface" \
    HF_HUB_DISABLE_XET=1

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
