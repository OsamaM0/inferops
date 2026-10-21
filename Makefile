SHELL := /bin/bash
APP_DIR := .
VENV_PY := $(CURDIR)/$(APP_DIR)/.venv/bin/python
PY ?= $(if $(wildcard $(VENV_PY)),$(VENV_PY),python3)
PROJECT ?= $(DOMAIN)
PROJECT_ROOT ?=
MODE ?= cpu
API_PORT ?= 8080
GPU ?= 0
LABELS ?= person,helmet,vest
FORMAT ?= yolo
SELECT = $(if $(PROJECT_ROOT),--domain-root "$(abspath $(PROJECT_ROOT))",$(if $(PROJECT),--domain "$(PROJECT)",))
SELECT_ENV = $(if $(PROJECT_ROOT),INFEROPS_DOMAIN_ROOT="$(abspath $(PROJECT_ROOT))",$(if $(PROJECT),INFEROPS_ACTIVE_DOMAIN="$(PROJECT)",))
LAUNCH = $(PY) $(APP_DIR)/scripts/run_mode.py --mode $(MODE) --port $(API_PORT) $(SELECT)
GPU_ARG = $(if $(filter 1,$(GPU)),--gpu,)

.DEFAULT_GOAL := help
.PHONY: help setup install dev-install setup-models projects domains new-project new-domain new-model
.PHONY: test test-core test-project test-domain test-all lint fmt validate preflight
.PHONY: run run-container serve serve-api stop stop-mode up down logs status
.PHONY: configure-accelerated setup-container-runtime worker-pipeline worker-topic clean

help: ## Show common commands; select PROJECT=<name> or PROJECT_ROOT=/path/to/project
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {print $$1 "  " $$2}' $(MAKEFILE_LIST)

setup: ## Create the Python environment; install project dependencies and declared weights
	@if [[ ! -x "$(VENV_PY)" ]]; then python3 -m venv $(APP_DIR)/.venv; fi
	cd $(APP_DIR) && $(VENV_PY) scripts/setup_inferops.py $(SELECT) $(GPU_ARG)

install: ## Install dependencies only (GPU=1 selects ONNX Runtime CUDA)
	@if [[ ! -x "$(VENV_PY)" ]]; then python3 -m venv $(APP_DIR)/.venv; fi
	cd $(APP_DIR) && $(VENV_PY) scripts/setup_inferops.py $(SELECT) --skip-models $(GPU_ARG)

dev-install: install

setup-models: ## Install missing assets declared by the selected project
	cd $(APP_DIR) && $(PY) -m inferops.adapters.inbound.cli project setup-assets $(SELECT)

projects: ## List available project blocks
	cd $(APP_DIR) && $(PY) -m inferops.adapters.inbound.cli project list

domains: projects

new-project: ## Create a project: NAME=ppe_site LABELS=person,helmet,vest
	@test -n "$(NAME)" || { echo "Provide NAME=<identifier>"; exit 1; }
	cd $(APP_DIR) && $(PY) -m inferops.adapters.inbound.cli project init "$(NAME)" --labels "$(LABELS)"

new-domain: new-project

new-model: ## Add a model port: PROJECT=ppe_site NAME=detector FORMAT=yolo|detr LABELS=person,helmet
	@test -n "$(NAME)" || { echo "Provide NAME=<identifier>"; exit 1; }
	cd $(APP_DIR) && $(PY) -m inferops.adapters.inbound.cli models init "$(NAME)" $(SELECT) --format "$(FORMAT)" --labels "$(LABELS)"

validate: ## Validate YAML and references between models, pipelines, fusion and alerts
	cd $(APP_DIR) && $(PY) -m inferops.adapters.inbound.cli validate $(SELECT)

preflight: ## Check configuration, plugins, dependencies and artifacts for MODE
	cd $(APP_DIR) && INFEROPS_EXECUTION_MODE=$(MODE) $(PY) -m inferops.adapters.inbound.cli preflight $(SELECT)

run: ## Start the native API; optional services via RUN_ARGS='--profiles storage,broker,monitoring'
	$(LAUNCH) $(RUN_ARGS)

run-container: ## Start the reusable Docker engine with the selected project mounted
	$(LAUNCH) --container $(RUN_ARGS)

serve: run

serve-api: ## Start only the native API, using local storage or already running optional services
	$(LAUNCH) --no-services $(RUN_ARGS)

up: run-container

stop: ## Stop the native API launched by this repository
	$(PY) $(APP_DIR)/scripts/run_mode.py --stop

stop-mode: stop

down: ## Stop Docker services; retain containers and their database/object-store data
	$(LAUNCH) --services-down

logs: ## Follow Docker service logs
	$(LAUNCH) --logs

status: ## Show Docker service status
	$(LAUNCH) --status

configure-accelerated: ## Validate Compose configuration for MODE=cuda|triton
	$(LAUNCH) --compose-check

setup-container-runtime: ## Install NVIDIA Container Toolkit (sudo; restarts Docker)
	bash $(APP_DIR)/scripts/setup_container_runtime.sh

test: test-core test-project ## Run core and selected project tests in separate processes

test-core: ## Run the engine suite
	cd $(APP_DIR) && env -u INFEROPS_ACTIVE_DOMAIN -u INFEROPS_DOMAIN_ROOT -u INFEROPS_EXECUTION_MODE $(PY) -m pytest -q tests

test-project: ## Run the selected project's tests
	@if [[ -z "$(PROJECT)$(PROJECT_ROOT)" ]]; then \
		echo "No project selected; skipping project tests"; \
	else \
		cd $(APP_DIR) && $(SELECT_ENV) $(PY) -c 'from inferops.adapters.outbound.projects.settings import get_config_stack; import subprocess,sys; root=get_config_stack().domain_root; tests=root/"tests"; sys.exit(subprocess.call([sys.executable,"-m","pytest","-q",str(tests)]) if tests.is_dir() else 0)'; \
	fi

test-domain: test-project

test-all: test-core ## Run the engine and every shipped project's tests
	@for tests in $(APP_DIR)/projects/*/tests; do \
		[[ -d "$$tests" ]] || continue; \
		project_root=$$(realpath "$$(dirname "$$tests")"); \
		(cd $(APP_DIR) && env -u INFEROPS_ACTIVE_DOMAIN -u INFEROPS_EXECUTION_MODE INFEROPS_DOMAIN_ROOT="$$project_root" $(PY) -m pytest -q "$$project_root/tests") || exit 1; \
	done

lint: ## Check Python code
	cd $(APP_DIR) && $(PY) -m ruff check .

fmt: ## Format Python code
	cd $(APP_DIR) && $(PY) -m ruff format .

worker-pipeline: ## Run a source-driven worker: PIPELINE=<name>
	cd $(APP_DIR) && $(SELECT_ENV) INFEROPS_EXECUTION_MODE=$(MODE) $(PY) -m inferops.adapters.inbound.cli worker pipeline "$(PIPELINE)"

worker-topic: ## Run a topic worker: PIPELINE=<name> INPUT=<topic> OUTPUT=<topic>
	cd $(APP_DIR) && $(SELECT_ENV) INFEROPS_EXECUTION_MODE=$(MODE) $(PY) -m inferops.adapters.inbound.cli worker topic "$(PIPELINE)" --input-topic "$(INPUT)" --output-topic "$(OUTPUT)"

clean: ## Remove caches and Python packaging output (CLEAN_ARGS=--dry-run)
	$(PY) $(APP_DIR)/scripts/clean_workspace.py $(CLEAN_ARGS)
