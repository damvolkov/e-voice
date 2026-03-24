PROJECT ?= e-voice
PACKAGE ?= src/e_voice
SERVICE_PORT ?= 5500
WS_PORT ?= 5700
GRADIO_PORT ?= 5600
WEBSOCAT_VERSION ?= 1.13.0

OS := $(shell uname -s)
ARCH := $(shell uname -m)

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
        kill dev stt tts docker-up docker-down docker-build log clean


# Help

help:
	@echo "$(BOLD)$(BLUE)e-voice$(RESET) — Speech API (Robyn + faster-whisper + Kokoro)"
	@echo ""
	@echo "$(BOLD)Setup:$(RESET)"
	@echo "  $(GREEN)make install$(RESET)          Install everything (uv, deps, ffmpeg, websocat, pre-commit)"
	@echo "  $(GREEN)make sync$(RESET)             Sync dependencies from lockfile"
	@echo ""
	@echo "$(BOLD)Development:$(RESET)"
	@echo "  $(GREEN)make dev$(RESET)              API on :$(SERVICE_PORT), WS on :$(WS_PORT), Gradio on :$(GRADIO_PORT)"
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


# Setup & Dependencies

install:
	@echo "$(GREEN)[1/5] Installing uv$(RESET)"
ifeq ($(OS),Linux)
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
else ifeq ($(OS),Darwin)
	@command -v uv >/dev/null 2>&1 || brew install uv
endif
	@echo "$(GREEN)[2/5] Installing system deps (ffmpeg, websocat)$(RESET)"
ifeq ($(OS),Linux)
	@command -v ffmpeg >/dev/null 2>&1 || sudo apt-get install -y -qq ffmpeg > /dev/null
	@command -v websocat >/dev/null 2>&1 || { \
		WEBSOCAT_ARCH=$$([ "$(ARCH)" = "aarch64" ] && echo "aarch64" || echo "x86_64"); \
		wget -qO /tmp/websocat "https://github.com/vi/websocat/releases/download/v$(WEBSOCAT_VERSION)/websocat.$${WEBSOCAT_ARCH}-unknown-linux-musl" && \
		chmod +x /tmp/websocat && sudo mv /tmp/websocat /usr/local/bin/websocat; \
	}
else ifeq ($(OS),Darwin)
	@command -v ffmpeg >/dev/null 2>&1 || brew install -q ffmpeg
	@command -v websocat >/dev/null 2>&1 || brew install -q websocat
endif
	@echo "$(GREEN)[3/5] Syncing Python dependencies$(RESET)"
	@uv sync --dev --quiet
	@echo "$(GREEN)[4/5] Installing pre-commit hooks$(RESET)"
	@uv run pre-commit install > /dev/null
	@echo "$(GREEN)[5/5] Done$(RESET)"

sync:
	@uv sync --dev

lock:
	@uv lock


# Quality & Testing

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


# Development — API on :5500, WS on :5700, Gradio on :5600

kill:
	@for port in $(SERVICE_PORT) $(WS_PORT) $(GRADIO_PORT); do \
		lsof -i :$$port -t 2>/dev/null | xargs -r kill -9; \
	done
	@echo "$(GREEN)=== Ports $(SERVICE_PORT)/$(WS_PORT)/$(GRADIO_PORT) freed ===$(RESET)"

dev: kill
	@echo "$(CYAN)=== API: http://localhost:$(SERVICE_PORT) | WS: ws://localhost:$(WS_PORT) | Gradio: http://localhost:$(GRADIO_PORT) ===$(RESET)"
	@trap 'make -s kill' EXIT; \
	LD_LIBRARY_PATH="$(NVIDIA_LIBS):$$LD_LIBRARY_PATH" uv run python -m robyn src/e_voice/main.py --dev


# Live test (requires running server — works with both local and docker)
# Local:  API on :5500
# Docker: API on :5500 (nginx proxies from :80)

STT_LANG ?= es
STT_FMT ?= text

stt:
	@command -v ffmpeg >/dev/null 2>&1 || { echo "$(RED)ffmpeg not found. Run: make install$(RESET)"; exit 1; }
	@command -v websocat >/dev/null 2>&1 || { echo "$(RED)websocat not found. Run: make install$(RESET)"; exit 1; }
	@echo "$(CYAN)=== STT: mic → ws://localhost:$(WS_PORT)/v1/audio/transcriptions?language=$(STT_LANG) ===$(RESET)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(RESET)"
	@ffmpeg -loglevel quiet -f pulse -i default -ac 1 -ar 16000 -f s16le pipe:1 \
		| websocat --binary "ws://localhost:$(WS_PORT)/v1/audio/transcriptions?language=$(STT_LANG)&response_format=$(STT_FMT)"

TTS_VOICE ?= af_heart
TTS_FMT ?= pcm

tts:
	@command -v ffplay >/dev/null 2>&1 || { echo "$(RED)ffplay not found. Install: sudo apt-get install ffmpeg$(RESET)"; exit 1; }
	@echo "$(CYAN)=== TTS: http://localhost:$(SERVICE_PORT)/v1/audio/speech voice=$(TTS_VOICE) ===$(RESET)"
	@echo "$(YELLOW)Type text + Enter to speak. Ctrl+C to stop.$(RESET)"
	@while IFS= read -r line; do \
		[ -z "$$line" ] && continue; \
		curl -s http://localhost:$(SERVICE_PORT)/v1/audio/speech \
			-H 'Content-Type: application/json' \
			-d "{\"input\":\"$$line\",\"voice\":\"$(TTS_VOICE)\",\"response_format\":\"$(TTS_FMT)\",\"stream\":false}" \
			| ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -loglevel quiet -; \
	done



# Docker — self-contained image (API + Gradio + nginx on :80)

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


# Cleanup

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf dist/ build/ *.egg-info/
	@echo "$(GREEN)=== Clean ===$(RESET)"
