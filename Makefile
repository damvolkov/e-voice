.DEFAULT_GOAL := help
MAKEFLAGS += --no-print-directory
export PATH := $(HOME)/.cargo/bin:$(HOME)/.local/bin:$(PATH)
SHELL := bash
PREK_VERSION ?= 0.5.4
ZENSICAL_VERSION ?= 0.0.67
COVERAGE ?= 90
BUMP ?= patch
CONFIG ?=
CONFIGURED = $(if $(CONFIG),--config $(CONFIG))
WORD ?= hey eager
MODE ?= full
COMPOSE = $(if $(filter split,$(MODE)),-f docker/compose.stt.yml -f docker/compose.tts.yml,-f docker/compose.$(MODE).yml)
PACKAGES ?= --workspace
BINS ?= e-voice ecli
DIST ?= data/ops/dist
TARGET = $(or $(CARGO_TARGET_DIR),data/ops/target)
SUDO = $(if $(filter 0,$(shell id -u 2>/dev/null)),,sudo)
KERNEL = $(firstword $(subst _, ,$(shell uname -s 2>/dev/null)))
HOST_Linux = ubuntu
HOST_Darwin = mac
HOST_MINGW64 = windows
HOST_MINGW32 = windows
HOST_MSYS = windows
HOST_CYGWIN = windows
PLATFORM = $(or $(filter ubuntu mac windows,$(OS)),$(HOST_$(KERNEL)))
ARGS = $(filter-out $(firstword $(MAKECMDGOALS)),$(MAKECMDGOALS))
.PHONY: help install system system-ubuntu system-mac system-windows toolchain tools hooks coverage sherpa setup export annex gates release dist docs docs-serve wake datasets bench report serve stt image up down build fmt lint bounds test backends check

help:  ## list targets
	@awk 'BEGIN{FS=":.*##"} /^[a-z][a-zA-Z0-9_-]*:.*##/{printf "  \033[32m%-9s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)

install: system toolchain tools  ## everything a dev machine needs [OS=ubuntu|mac|windows, detected by default]

system: system-$(PLATFORM)  ## OS build packages: compiler, cmake, pkg-config, curl, bzip2 (+ ALSA headers on Linux)

system-ubuntu:  # Debian family (apt): also what the Dockerfiles run
	@$(SUDO) apt-get update -qq
	@$(SUDO) env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
	  build-essential bzip2 ca-certificates cmake curl libasound2-dev pkg-config

system-mac:  # Homebrew; bash ≥ 4 for the ops scripts, Xcode command line tools for the compiler
	@xcode-select -p >/dev/null 2>&1 || xcode-select --install
	@brew install bash cmake pkg-config

system-windows:  # winget from Git Bash: MSVC build tools, cmake
	@winget install --exact --silent --accept-package-agreements --accept-source-agreements --id Microsoft.VisualStudio.2022.BuildTools \
	  --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
	@winget install --exact --silent --accept-package-agreements --accept-source-agreements --id Kitware.CMake

toolchain:  ## rustup + the toolchain pinned in rust-toolchain.toml, and uv
	@command -v rustup >/dev/null || curl -fsSL https://sh.rustup.rs | sh -s -- -y --no-modify-path --default-toolchain none
	@command -v uv >/dev/null || curl -fsSL https://astral.sh/uv/install.sh | sh
	@rustup toolchain install

tools: sherpa  ## prek and cargo-llvm-cov, then the git hooks
	@uv tool install --quiet prek==$(PREK_VERSION)
	@cargo install --quiet --locked cargo-llvm-cov
	@prek install --force

hooks:  ## run every prek hook on every file (what CI runs)
	@uvx prek==$(PREK_VERSION) run --all-files --show-diff-on-failure

sherpa:  ## vendor the pinned sherpa-onnx libraries for this platform into data/stt/ops/sherpa
	@stt/ops/sherpa.sh

setup: sherpa  ## toolkit + exactly the models the config declares [CONFIG=path] [ARGS=--all adds tool/test models]
	@cargo run -q --release -p e-voice-stt -- $(CONFIGURED) pull $(ARGS)

wake:  ## calibrate a kws wake phrase: make wake WORD="eager" (prints the [stt.pipeline.ww] block)
	@cargo run -q --release -p e-voice-stt -- pull kws-gigaspeech piper-en-amy-low piper-en-lessac-medium piper-en-ryan-medium
	@cargo run -q --release -p e-voice-stt -- $(CONFIGURED) wake "$(WORD)" $(ARGS)

export:  ## export models without a published ONNX (emotion2vec+ large) into data/stt/ops/exports, then install them
	@uv run -q --script eval/export/emotion2vec.py --out data/stt/ops/exports/emotion2vec-plus-large
	@cargo run -q --release -p e-voice-stt -- pull emotion2vec-plus-large

datasets:  ## fetch the pinned evaluation sets into data/stt/ops/datasets
	@cd eval && for set in "fleurs-es 200" "fleurs-en 200" "mesd" "crema 400"; do set -- $$set; uv run -q python -m e_voice_eval fetch $$1 $${2:+--limit $$2}; done

bench: build  ## run the benchmark matrix (stt/ops/bench/*.toml × datasets) [configs forwarded]
	@stt/ops/bench.sh $(ARGS)

report:  ## compare every results file in data/stt/ops/results
	@cd eval && uv run -q python -m e_voice_eval report ../data/stt/ops/results/*.jsonl --out ../data/stt/ops/results/report.md

annex:  ## benchmark annex (table, CSV, charts) of every results file: make annex OUT=docs/history/<date>-<topic>
	@cd eval && uv run -q python -m e_voice_eval annex ../data/stt/ops/results/*.jsonl --out ../$(OUT)

serve: sherpa  ## run the gateway (release) [args forwarded]
	@cargo run -q --release -p e-voice-stt -- $(CONFIGURED) serve $(ARGS)

stt:  ## live mic transcription: make stt ARGS="--flat | --struct [--lang es]"
	@cargo run -q --release -p ecli -- stt $(ARGS)

docs:  ## the OpenAPI document, then the documentation site into data/ops/site
	@cargo run -q --release -p e-voice-stt -- openapi > docs/api/openapi.json
	@uvx zensical==$(ZENSICAL_VERSION) build --clean

docs-serve:  ## live documentation preview on :8000
	@cargo run -q --release -p e-voice-stt -- openapi > docs/api/openapi.json
	@uvx zensical==$(ZENSICAL_VERSION) serve

image:  ## build the runtime image(s): MODE=full (default) | stt | tts | split → e-voice:<mode>
	@docker compose $(COMPOSE) build

up:  ## start with compose, installing configured models first [MODE=full|stt|tts|split]
	@docker compose $(COMPOSE) up -d --build

down:  ## stop the compose services [MODE=full|stt|tts|split]
	@docker compose $(COMPOSE) down

build: sherpa  ## release build [PACKAGES="--package e-voice-stt"]
	@cargo build --release --locked $(PACKAGES) $(ARGS)

dist: build  ## binaries + native libraries into DIST/{bin,lib} [PACKAGES=… BINS=… DIST=…]
	@for bin in $(BINS); do install -D $(TARGET)/release/$$bin $(DIST)/bin/$$bin; done
	@mkdir -p $(DIST)/lib && cp -a data/stt/ops/sherpa/lib/. $(DIST)/lib/

fmt:  ## format the tree
	@cargo fmt --all

lint: sherpa  ## rustfmt check + clippy, warnings are errors
	@cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings

bounds:  ## inner layers never import outer ones
	@! grep -rnE 'crate::(core|workflow|api|ops)' stt/src/config \
	  && ! grep -rnE 'crate::(workflow|core|config|api|ops)' stt/src/schema \
	  && ! grep -rnE 'crate::(api|ops)' stt/src/workflow stt/src/core \
	  && for node in denoise ww vad lid asr ser; do ! grep -rnE "crate::workflow::($$(echo denoise ww vad lid asr ser | tr ' ' '\n' | grep -vx $$node | paste -sd'|'))::" stt/src/workflow/$$node || exit 1; done \
	  && printf '\033[32mbounds ok\033[0m\n'

test: sherpa  ## unit, property and integration tests [args forwarded]
	@cargo test --workspace $(ARGS)

backends: sherpa  ## model-backed tests against installed models (make setup ARGS=--all first)
	@cargo test --release --workspace -- --ignored $(ARGS)

coverage: sherpa  ## line coverage over every test incl. model-backed ones; fails under $(COVERAGE)%
	@cargo llvm-cov --workspace --quiet --ignore-filename-regex '(stt/src/main\.rs|cli/)' --fail-under-lines $(COVERAGE) --summary-only -- --include-ignored

gates:  ## every vars.* a workflow reads is declared, disabled, in .github/ci.vars.example
	@used="$$(grep -rhoE 'vars\.[A-Z_]+' .github/workflows | sed 's/vars\.//' | sort -u)"; \
	  for gate in $$used; do grep -qE "^$$gate=false" .github/ci.vars.example || { echo "gate $$gate is undeclared or ships enabled"; exit 1; }; done; \
	  ! grep -qE '^[A-Z_]+=true' .github/ci.vars.example && printf '\033[32mgates ok\033[0m\n'

release:  ## cut a release: make release BUMP=patch|minor|major (tag, GitHub release, gated images)
	@gh workflow run release.yml -f bump=$(BUMP)

check: lint bounds gates test  ## the local gate

%:
	@:
