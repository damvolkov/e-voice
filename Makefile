# =============================================================================
# robyn-ml-api Makefile
# =============================================================================
PROJECT ?= robyn-ml-api
VERSION ?= latest
DEBUG ?= True
ENVIRONMENT ?= DEV
PACKAGE ?= app
SERVICE_PORT ?= 8000

# OS Detection
OS := $(shell uname -s)

# ANSI Escape codes
BOLD   := \033[1m
RESET  := \033[0m
GREEN  := \033[1;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
CYAN   := \033[0;36m
RED    := \033[0;31m

# Environment
-include .env
ifneq (,$(wildcard .env))
    $(eval export $(shell sed -ne 's/ *#.*$$//; /./ s/=.*$$// p' .env))
endif
export PYTHONPATH := $(CURDIR)

COMPOSE_FILE := compose.yml

.PHONY: help install sync lock lint format typecheck test dev prod run docker-build docker-up docker-down docker-test clean

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:
	@echo "$(BOLD)$(BLUE)robyn-ml-api$(RESET) - ML API Template powered by Robyn"
	@echo ""
	@echo "$(BOLD)Setup:$(RESET)"
	@echo "  $(GREEN)make install$(RESET)      Install uv, dependencies, and pre-commit hooks"
	@echo "  $(GREEN)make sync$(RESET)         Sync dependencies from lockfile (frozen)"
	@echo "  $(GREEN)make lock$(RESET)         Update lockfile with current dependencies"
	@echo ""
	@echo "$(BOLD)Development:$(RESET)"
	@echo "  $(GREEN)make dev$(RESET)          Start development server with DEBUG=True"
	@echo "  $(GREEN)make prod$(RESET)         Start production server with DEBUG=False"
	@echo "  $(GREEN)make run$(RESET)          Alias for 'make dev'"
	@echo ""
	@echo "$(BOLD)Quality:$(RESET)"
	@echo "  $(GREEN)make lint$(RESET)         Run ruff linter with auto-fix"
	@echo "  $(GREEN)make format$(RESET)       Format code with ruff"
	@echo "  $(GREEN)make typecheck$(RESET)    Run ty type checker"
	@echo "  $(GREEN)make test$(RESET)         Run unit tests"
	@echo ""
	@echo "$(BOLD)Docker:$(RESET)"
	@echo "  $(GREEN)make docker-build$(RESET) Build Docker image"
	@echo "  $(GREEN)make docker-up$(RESET)    Start service in Docker"
	@echo "  $(GREEN)make docker-down$(RESET)  Stop Docker services"
	@echo "  $(GREEN)make docker-test$(RESET)  Build, start, and test Docker deployment"
	@echo "  $(GREEN)make log$(RESET)          Tail container logs"
	@echo ""
	@echo "$(BOLD)Cleanup:$(RESET)"
	@echo "  $(GREEN)make clean$(RESET)        Remove cache and build artifacts"

# -----------------------------------------------------------------------------
# Setup & Dependencies
# -----------------------------------------------------------------------------
install:
	@echo "$(GREEN)=== Installing system dependencies ===$(RESET)"
ifeq ($(OS),Linux)
	@echo "$(GREEN)=== Installing uv ===$(RESET)"
	@curl -LsSf https://astral.sh/uv/install.sh | sh
else ifeq ($(OS),Darwin)
	@command -v brew >/dev/null 2>&1 || { echo "$(RED)Error: Homebrew required$(RESET)"; exit 1; }
	@echo "$(GREEN)=== Installing uv ===$(RESET)"
	@brew install uv
else
	@echo "$(RED)Error: Unsupported OS: $(OS)$(RESET)"
	@exit 1
endif
	@echo "$(GREEN)=== Syncing Python dependencies ===$(RESET)"
	@uv sync --frozen
	@echo "$(GREEN)=== Installing pre-commit hooks ===$(RESET)"
	@uv run pre-commit install
	@echo "$(GREEN)=== Setup complete ===$(RESET)"

sync:
	@echo "$(GREEN)=== Syncing dependencies ===$(RESET)"
	@uv sync --dev
	@echo "$(GREEN)=== Sync complete ===$(RESET)"

lock:
	@echo "$(GREEN)=== Updating lockfile ===$(RESET)"
	@uv lock
	@echo "$(GREEN)=== Lockfile updated ===$(RESET)"

# -----------------------------------------------------------------------------
# Quality & Testing
# -----------------------------------------------------------------------------
lint:
	@echo "$(GREEN)=== Running linter ===$(RESET)"
	@uv run ruff check --fix $(PACKAGE)
	@echo "$(GREEN)=== Lint complete ===$(RESET)"

format:
	@echo "$(GREEN)=== Formatting code ===$(RESET)"
	@uv run ruff format $(PACKAGE)
	@echo "$(GREEN)=== Format complete ===$(RESET)"

typecheck:
	@echo "$(GREEN)=== Running type checker ===$(RESET)"
	@uv run ty check
	@echo "$(GREEN)=== Type check complete ===$(RESET)"

test:
	@echo "$(GREEN)=== Running unit tests ===$(RESET)"
	@uv run pytest test/unit -v
	@echo "$(GREEN)=== Tests complete ===$(RESET)"

# -----------------------------------------------------------------------------
# Development
# -----------------------------------------------------------------------------
dev:
	@echo "$(GREEN)=== Starting Development Server ===$(RESET)"
	@DEBUG=True ENVIRONMENT=DEV uv run python -m app.main

prod:
	@echo "$(GREEN)=== Starting Production Server ===$(RESET)"
	@ENVIRONMENT=PROD DEBUG=False uv run python -m app.main

run: dev

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------
docker-build:
	@echo "$(CYAN)=== Building Docker image ===$(RESET)"
	@docker compose -f $(COMPOSE_FILE) build $(PROJECT)
	@echo "$(GREEN)=== Build complete ===$(RESET)"

docker-up:
	@echo "$(CYAN)=== Starting service in Docker ===$(RESET)"
	@docker compose -f $(COMPOSE_FILE) up -d $(PROJECT)
	@echo "$(GREEN)=== Service started ===$(RESET)"
	@echo "$(CYAN)API: http://localhost:$(SERVICE_PORT)$(RESET)"
	@echo "$(CYAN)Logs: make log$(RESET)"

docker-down:
	@echo "$(YELLOW)=== Stopping Docker services ===$(RESET)"
	@docker compose -f $(COMPOSE_FILE) down
	@echo "$(GREEN)=== Services stopped ===$(RESET)"

docker-test: docker-build docker-up
	@echo "$(CYAN)=== Testing Docker deployment ===$(RESET)"
	@echo "$(YELLOW)Waiting for service (10s)...$(RESET)"
	@sleep 10
	@curl -sf http://localhost:$(SERVICE_PORT)/health | jq . || echo "$(RED)Health check failed$(RESET)"
	@echo "$(GREEN)=== Docker test complete ===$(RESET)"

log:
	@docker compose -f $(COMPOSE_FILE) logs -f $(PROJECT)

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
clean:
	@echo "$(YELLOW)=== Cleaning cache and artifacts ===$(RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf dist/ build/ *.egg-info/
	@echo "$(GREEN)=== Clean complete ===$(RESET)"
