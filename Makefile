# =============================================================================
# e-voice Makefile
# =============================================================================
PROJECT ?= e-voice
PACKAGE ?= src/e_voice
SERVICE_PORT ?= 5500
GRADIO_PORT ?= 5600

OS := $(shell uname -s)

BOLD   := \033[1m
RESET  := \033[0m
GREEN  := \033[1;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
CYAN   := \033[0;36m
RED    := \033[0;31m

export PYTHONPATH := $(CURDIR)/src

NVIDIA_LIBS := $(CURDIR)/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$(CURDIR)/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH := $(NVIDIA_LIBS):$(LD_LIBRARY_PATH)

COMPOSE_FILE := compose.yml

.PHONY: help install sync lock lint type test test-integration check \
        dev stt tts docker-up docker-down docker-build log clean

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:
	@echo "$(BOLD)$(BLUE)e-voice$(RESET) — Speech API (Robyn + faster-whisper + Kokoro)"
	@echo ""
	@echo "$(BOLD)Setup:$(RESET)"
	@echo "  $(GREEN)make install$(RESET)          Install uv, deps, pre-commit hooks"
	@echo "  $(GREEN)make sync$(RESET)             Sync dependencies from lockfile"
	@echo ""
	@echo "$(BOLD)Development:$(RESET)"
	@echo "  $(GREEN)make dev$(RESET)              API on :$(SERVICE_PORT), Gradio UI on :$(GRADIO_PORT)"
	@echo ""
	@echo "$(BOLD)Live test (requires running server):$(RESET)"
	@echo "  $(GREEN)make stt$(RESET)              mic → WebSocket STT  (ffmpeg + websocat)"
	@echo "  $(GREEN)make tts$(RESET)              text → WebSocket TTS (websocat)"
	@echo ""
	@echo "$(BOLD)Quality:$(RESET)"
	@echo "  $(GREEN)make lint$(RESET)             Ruff check + format"
	@echo "  $(GREEN)make type$(RESET)             ty type checker"
	@echo "  $(GREEN)make test$(RESET)             Unit tests (parallel, coverage >90%)"
	@echo "  $(GREEN)make check$(RESET)            lint + type + test"
	@echo ""
	@echo "$(BOLD)Docker:$(RESET)"
	@echo "  $(GREEN)make docker-up$(RESET)        Build + start (GPU, port :$(SERVICE_PORT))"
	@echo "  $(GREEN)make docker-down$(RESET)      Stop"
	@echo "  $(GREEN)make docker-build$(RESET)     Build image only"
	@echo "  $(GREEN)make log$(RESET)              Tail container logs"
	@echo ""
	@echo "$(BOLD)Cleanup:$(RESET)"
	@echo "  $(GREEN)make clean$(RESET)            Remove caches and build artifacts"

# -----------------------------------------------------------------------------
# Setup & Dependencies
# -----------------------------------------------------------------------------
install:
	@echo "$(GREEN)=== Installing system dependencies ===$(RESET)"
ifeq ($(OS),Linux)
	@curl -LsSf https://astral.sh/uv/install.sh | sh
else ifeq ($(OS),Darwin)
	@command -v brew >/dev/null 2>&1 || { echo "$(RED)Error: Homebrew required$(RESET)"; exit 1; }
	@brew install uv
else
	@echo "$(RED)Error: Unsupported OS: $(OS)$(RESET)" && exit 1
endif
	@echo "$(GREEN)=== Syncing Python dependencies ===$(RESET)"
	@uv sync --frozen
	@echo "$(GREEN)=== Installing pre-commit hooks ===$(RESET)"
	@uv run pre-commit install
	@echo "$(GREEN)=== Setup complete ===$(RESET)"

sync:
	@uv sync --dev

lock:
	@uv lock

# -----------------------------------------------------------------------------
# Quality & Testing
# -----------------------------------------------------------------------------
lint:
	@uv run ruff check --fix $(PACKAGE) tests/
	@uv run ruff format $(PACKAGE) tests/

type:
	@uv run ty check

test:
	@uv run pytest tests/unit -n auto -v -m 'not slow' --cov --cov-report=term-missing

test-integration:
	@uv run pytest tests/integration -v -m slow

check: lint type test

# -----------------------------------------------------------------------------
# Development — API on :5500, Gradio on :5600
# -----------------------------------------------------------------------------
dev:
	@echo "$(CYAN)=== API: http://localhost:$(SERVICE_PORT) | Gradio: http://localhost:$(GRADIO_PORT) ===$(RESET)"
	@LD_LIBRARY_PATH="$(NVIDIA_LIBS):$$LD_LIBRARY_PATH" uv run python -c "from e_voice.main import main; main()"

# -----------------------------------------------------------------------------
# Live test (requires running server)
# Deps: sudo apt install ffmpeg && cargo install websocat
# -----------------------------------------------------------------------------
STT_LANG ?= es
STT_FMT ?= text

stt:
	@command -v ffmpeg >/dev/null 2>&1 || { echo "$(RED)ffmpeg not found. Install: sudo apt install ffmpeg$(RESET)"; exit 1; }
	@command -v websocat >/dev/null 2>&1 || { echo "$(RED)websocat not found. Install: cargo install websocat$(RESET)"; exit 1; }
	@echo "$(CYAN)=== STT: mic → ws://localhost:$(SERVICE_PORT)/v1/audio/transcriptions?language=$(STT_LANG) ===$(RESET)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(RESET)"
	@ffmpeg -loglevel quiet -f pulse -i default -ac 1 -ar 16000 -f s16le pipe:1 \
		| uv run python -uc "import sys,base64;[print(base64.b64encode(c).decode(),flush=True)for c in iter(lambda:sys.stdin.buffer.read(32000),b'')]" \
		| websocat "ws://localhost:$(SERVICE_PORT)/v1/audio/transcriptions?language=$(STT_LANG)&response_format=$(STT_FMT)"

tts:
	@command -v websocat >/dev/null 2>&1 || { echo "$(RED)websocat not found. Install: cargo install websocat$(RESET)"; exit 1; }
	@echo "$(CYAN)=== TTS: ws://localhost:$(SERVICE_PORT)/v1/audio/speech ===$(RESET)"
	@echo "$(YELLOW)Type JSON + Enter. Example: {\"input\":\"Hello world\",\"voice\":\"af_heart\"}$(RESET)"
	@websocat "ws://localhost:$(SERVICE_PORT)/v1/audio/speech"

# -----------------------------------------------------------------------------
# Docker — self-contained image (API + Gradio + nginx on :80)
# -----------------------------------------------------------------------------
docker-build:
	@echo "$(CYAN)=== Building Docker image ===$(RESET)"
	@docker compose -f $(COMPOSE_FILE) build
	@echo "$(GREEN)=== Build complete ===$(RESET)"

docker-up: docker-build
	@docker compose -f $(COMPOSE_FILE) up -d
	@echo "$(GREEN)=== Running at http://localhost:$(SERVICE_PORT) (UI + API + docs) ===$(RESET)"

docker-down:
	@docker compose -f $(COMPOSE_FILE) down

log:
	@docker compose -f $(COMPOSE_FILE) logs -f

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf dist/ build/ *.egg-info/
	@echo "$(GREEN)=== Clean ===$(RESET)"
