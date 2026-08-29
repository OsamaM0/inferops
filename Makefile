SHELL := /bin/bash
APP_DIR := inferops
VENV_PY := $(CURDIR)/$(APP_DIR)/.venv/bin/python
PY ?= $(if $(wildcard $(VENV_PY)),$(VENV_PY),python3)
RUN_DIR := .run
LOG_DIR := logs
DATA_DIR := .data
TOOLS_BIN_DIR := $(CURDIR)/.tools/bin
API_PORT ?= 8080
DASHBOARD_PORT ?= 8090
INFEROPS_API_URL ?= http://127.0.0.1:$(API_PORT)
DOMAIN ?=
DOMAIN_ENV := $(if $(DOMAIN),INFEROPS_ACTIVE_DOMAIN=$(DOMAIN),)
DOMAIN_ARG := $(if $(DOMAIN),--domain $(DOMAIN),)
# Everything domain-specific (Triton auto-start, model repository, tests,
# prewarm) is read from the pack's domain.yaml through `inferops domain env`;
# nothing here names a particular domain.
ifneq ($(DOMAIN),)
DOMAIN_ENV_LINES := $(shell cd $(APP_DIR) && $(PY) -m inferops.cli domain env --domain $(DOMAIN) 2>/dev/null)
domain_var = $(patsubst $(1)=%,%,$(filter $(1)=%,$(DOMAIN_ENV_LINES)))
DOMAIN_ROOT := $(call domain_var,INFEROPS_DOMAIN_ROOT)
DOMAIN_TRITON := $(call domain_var,INFEROPS_DOMAIN_TRITON)
DOMAIN_MODEL_REPOSITORY := $(call domain_var,INFEROPS_DOMAIN_MODEL_REPOSITORY)
DOMAIN_HAS_TESTS := $(call domain_var,INFEROPS_DOMAIN_HAS_TESTS)
ifeq ($(DOMAIN_ROOT),)
$(warning DOMAIN=$(DOMAIN) could not be resolved; run `make domains` to list packs)
endif
endif
MODEL_REPOSITORY_DIR ?= $(if $(DOMAIN_MODEL_REPOSITORY),$(DOMAIN_MODEL_REPOSITORY),$(CURDIR)/$(APP_DIR)/model_repository)
GPU ?= 0
GPU_ARG := $(if $(filter $(GPU),1),--gpu,)
# serve-mock supplies a mock S3 (moto) but used to leave INFEROPS_PG_URL at its
# Postgres default, so every event/alert insert failed against a server that was
# never started and the dashboard's activity panels stayed permanently empty.
# The DAO already declares SQLite variants for its JSONB columns, so point the
# mock stack at a local file and the whole surface works offline.
MOCK_PG_URL ?= sqlite+aiosqlite:///$(CURDIR)/$(DATA_DIR)/inferops-mock.db
MINIO_PORT ?= 9000
MINIO_CONSOLE_PORT ?= 9001
REDPANDA_PORT ?= 19092
REDPANDA_ADMIN_PORT ?= 19644
MINIO_ROOT_USER ?= minio
MINIO_ROOT_PASSWORD ?= minio12345
WITH_TRITON ?= auto
TRITON_IMAGE ?= nvcr.io/nvidia/tritonserver:24.04-py3
TRITON_CONTAINER ?= inferops-triton
TRITON_BIN ?= $(shell command -v tritonserver 2>/dev/null || echo /opt/tritonserver/bin/tritonserver)
TRITON_HTTP_PORT ?= 8000
TRITON_GRPC_PORT ?= 8001
TRITON_METRICS_PORT ?= 8002
PROMETHEUS_PORT ?= 9090
GRAFANA_PORT ?= 3000
PROMETHEUS_BIN ?= $(shell command -v prometheus 2>/dev/null || echo $(TOOLS_BIN_DIR)/prometheus)
GRAFANA_DIR ?= $(CURDIR)/.tools/grafana
GRAFANA_BIN ?= $(shell test -x $(GRAFANA_DIR)/bin/grafana && echo $(GRAFANA_DIR)/bin/grafana || echo $(GRAFANA_DIR)/bin/grafana-server)
API_PID_FILE := $(RUN_DIR)/inferops-api.pid
DASHBOARD_PID_FILE := $(RUN_DIR)/dashboard.pid
MINIO_PID_FILE := $(RUN_DIR)/minio.pid
REDPANDA_PID_FILE := $(RUN_DIR)/redpanda.pid
MOTO_PID_FILE := $(RUN_DIR)/moto.pid
TRITON_PID_FILE := $(RUN_DIR)/triton.pid
PROMETHEUS_PID_FILE := $(RUN_DIR)/prometheus.pid
GRAFANA_PID_FILE := $(RUN_DIR)/grafana.pid
API_LOG_FILE := $(LOG_DIR)/inferops-api.log
DASHBOARD_LOG_FILE := $(LOG_DIR)/dashboard.log
MINIO_LOG_FILE := $(LOG_DIR)/minio.log
REDPANDA_LOG_FILE := $(LOG_DIR)/redpanda.log
MOTO_LOG_FILE := $(LOG_DIR)/moto-s3.log
TRITON_LOG_FILE := $(LOG_DIR)/triton.log
PROMETHEUS_LOG_FILE := $(LOG_DIR)/prometheus.log
GRAFANA_LOG_FILE := $(LOG_DIR)/grafana.log
# Mirror the PY fallback: prefer the project venv, otherwise take whatever is on
# PATH. This was pinned to the venv path with no fallback, so `make serve-mock`
# failed outright in any environment that installs into a conda/system env.
VENV_MOTO := $(CURDIR)/$(APP_DIR)/.venv/bin/moto_server
MOTO_SERVER ?= $(if $(wildcard $(VENV_MOTO)),$(VENV_MOTO),$(shell command -v moto_server 2>/dev/null || echo moto_server))
MINIO_BIN ?= $(shell command -v minio 2>/dev/null || echo $(TOOLS_BIN_DIR)/minio)
MC_BIN ?= $(shell command -v mc 2>/dev/null || echo $(TOOLS_BIN_DIR)/mc)
REDPANDA_BIN ?= $(shell command -v redpanda 2>/dev/null)
RPK_BIN ?= $(shell command -v rpk 2>/dev/null)
TRITON_ENABLED := 0
ifeq ($(WITH_TRITON),auto)
ifeq ($(DOMAIN_TRITON),1)
TRITON_ENABLED := 1
endif
else ifeq ($(WITH_TRITON),1)
TRITON_ENABLED := 1
endif
SERVE_DEPS := stop start-minio start-redpanda start-prometheus start-grafana
ifeq ($(TRITON_ENABLED),1)
SERVE_DEPS += start-triton
endif

.PHONY: help setup install dev-install setup-models test test-core test-domain test-all scenario lint fmt validate preflight \
	domains new-domain serve serve-mock serve-api serve-dashboard stop up down logs \
	start-minio start-redpanda start-triton start-prometheus start-grafana \
	stop-minio stop-redpanda stop-triton stop-prometheus stop-grafana \
	ensure-topics ensure-buckets status \
	worker-pipeline worker-topic setup-runtime

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "%-22s %s\n",$$1,$$2}'
	@echo
	@echo "Most targets take DOMAIN=<pack> (see 'make domains'). GPU=1 installs onnxruntime-gpu."

setup: ## Install InferOps core (+ DOMAIN=<pack> extras and model assets) and native serve prerequisites
	@if [[ ! -x "$(VENV_PY)" ]]; then python3 -m venv $(APP_DIR)/.venv; fi
	cd $(APP_DIR) && $(VENV_PY) scripts/setup_inferops.py $(DOMAIN_ARG) $(GPU_ARG)
	@if [[ "$(SKIP_RUNTIME)" != "1" ]]; then $(MAKE) setup-runtime; fi

setup-runtime: ## Install native MinIO/mc and Redpanda/rpk prerequisites for make serve
	cd $(APP_DIR) && $(PY) scripts/setup_native_runtime.py

install: ## Install InferOps with dev dependencies (GPU=1 for the CUDA onnxruntime wheel)
	cd $(APP_DIR) && $(PY) scripts/setup_inferops.py --skip-models $(GPU_ARG)

dev-install: install ## Alias for install

setup-models: ## Build the core tiny model and download DOMAIN=<pack> assets declared in its domain.yaml
	cd $(APP_DIR) && $(PY) scripts/make_tiny_model.py
	@if [[ -n "$(DOMAIN)" ]]; then cd $(APP_DIR) && $(PY) -m inferops.cli domain setup-assets --domain $(DOMAIN); fi

domains: ## List available domain packs
	cd $(APP_DIR) && $(PY) -m inferops.cli domain list

new-domain: ## Scaffold a runnable domain pack: make new-domain NAME=my_project LABELS=person,helmet
	@if [[ -z "$(NAME)" ]]; then echo "usage: make new-domain NAME=<identifier> [LABELS=a,b,c]"; exit 1; fi
	cd $(APP_DIR) && $(PY) -m inferops.cli domain init $(NAME) $(if $(LABELS),--labels $(LABELS),)

test: test-core test-domain ## Run the core suite, then the DOMAIN=<pack> suite when one is selected

test-core: ## Run the domain-free core test suite
	cd $(APP_DIR) && env -u INFEROPS_ACTIVE_DOMAIN -u INFEROPS_DOMAIN_ROOT $(PY) -m pytest -q tests

test-domain: ## Run the selected DOMAIN=<pack> test suite (examples/<pack>/tests)
	@if [[ -z "$(DOMAIN)" ]]; then echo "no DOMAIN selected; skipping domain tests"; exit 0; fi
	@if [[ "$(DOMAIN_HAS_TESTS)" != "1" ]]; then echo "domain $(DOMAIN) has no tests/ directory"; exit 0; fi
	cd $(APP_DIR) && INFEROPS_ACTIVE_DOMAIN=$(DOMAIN) $(PY) -m pytest -q $(DOMAIN_ROOT)/tests

test-all: test-core ## Run the core suite plus every domain pack that ships tests
	@for d in $(APP_DIR)/examples/*/tests $(APP_DIR)/domains/*/tests; do \
		[[ -d "$$d" ]] || continue; \
		name=$$(basename $$(dirname $$d)); \
		echo "== domain $$name =="; \
		(cd $(APP_DIR) && INFEROPS_ACTIVE_DOMAIN=$$name $(PY) -m pytest -q $$(realpath $$d)) || exit 1; \
	done

scenario: ## Drive the dashboard end-to-end in a real browser and screenshot each step (uses the served domain)
	@command -v curl >/dev/null || { echo "curl is required"; exit 1; }
	@curl -fsS $(INFEROPS_API_URL)/healthz >/dev/null 2>&1 || { \
		echo "API is not up on $(INFEROPS_API_URL) — run: make serve-mock DOMAIN=<pack>"; exit 1; }
	cd $(APP_DIR) && $(PY) scripts/e2e_dashboard_scenario.py --api $(INFEROPS_API_URL) --out $(CURDIR)/$(RUN_DIR)/scenario $(SCENARIO_ARGS)
	@echo "Screenshots: $(CURDIR)/$(RUN_DIR)/scenario"

lint: ## Lint InferOps
	cd $(APP_DIR) && $(PY) -m ruff check .

fmt: ## Format InferOps
	cd $(APP_DIR) && $(PY) -m ruff format . && $(PY) -m ruff check --fix .

validate: ## Validate InferOps YAML configs (DOMAIN=<pack> for that pack's surface)
	cd $(APP_DIR) && $(PY) -m inferops.cli validate $(DOMAIN_ARG)

preflight: ## Validate plus dependency, plugin-import and model-artifact checks
	cd $(APP_DIR) && $(PY) -m inferops.cli preflight $(DOMAIN_ARG)


serve-api: ## Run only the InferOps API in the foreground
	cd $(APP_DIR) && $(DOMAIN_ENV) $(PY) -m uvicorn inferops.api.main:app --host 0.0.0.0 --port $(API_PORT) --reload

serve-dashboard: ## Run only the dashboard static server in the foreground
	$(PY) -m http.server $(DASHBOARD_PORT) --directory $(CURDIR)/frontend

start-minio: ## Start MinIO server (S3) in the background
	@mkdir -p $(RUN_DIR) $(LOG_DIR) $(DATA_DIR)/minio
	@if [[ ! -x "$(MINIO_BIN)" ]]; then \
		echo "minio binary not found at $(MINIO_BIN)"; \
		echo "Run 'make setup-runtime' to download the native serve prerequisites."; \
		echo "Manual MinIO install: curl -fsSL https://dl.min.io/server/minio/release/linux-amd64/minio -o $(MINIO_BIN) && chmod +x $(MINIO_BIN)"; \
		exit 1; \
	fi
	@if ss -ltn "( sport = :$(MINIO_PORT) )" 2>/dev/null | grep -q ":$(MINIO_PORT)"; then \
		echo "MinIO already listening on $(MINIO_PORT)"; \
	else \
		MINIO_ROOT_USER=$(MINIO_ROOT_USER) MINIO_ROOT_PASSWORD=$(MINIO_ROOT_PASSWORD) \
		nohup $(MINIO_BIN) server $(CURDIR)/$(DATA_DIR)/minio \
			--address ":$(MINIO_PORT)" --console-address ":$(MINIO_CONSOLE_PORT)" \
			> $(CURDIR)/$(MINIO_LOG_FILE) 2>&1 < /dev/null & echo $$! > $(CURDIR)/$(MINIO_PID_FILE); \
		for attempt in {1..100}; do \
			if curl -s -o /dev/null http://127.0.0.1:$(MINIO_PORT)/minio/health/live; then break; fi; \
			sleep 0.2; \
		done; \
	fi
	@$(MC_BIN) alias set inferops-local http://127.0.0.1:$(MINIO_PORT) $(MINIO_ROOT_USER) $(MINIO_ROOT_PASSWORD) >/dev/null 2>&1 || true
	@for b in inferops inferops-clips; do $(MC_BIN) mb -p inferops-local/$$b >/dev/null 2>&1 || true; done
	@echo "MinIO:         http://127.0.0.1:$(MINIO_PORT)  console http://127.0.0.1:$(MINIO_CONSOLE_PORT)  (user $(MINIO_ROOT_USER))"

start-redpanda: ## Start Redpanda broker in the background
	@mkdir -p $(RUN_DIR) $(LOG_DIR) $(DATA_DIR)/redpanda
	@if [[ -z "$(RPK_BIN)" ]]; then \
		echo "rpk binary not found in PATH"; \
		echo "Run 'make setup-runtime' to install the native serve prerequisites."; \
		echo "Manual Redpanda install: curl -1sLf 'https://dl.redpanda.com/nzc4ZYQK3WRGd9sy/redpanda/cfg/setup/bash.deb.sh' | sudo -E bash && sudo apt install redpanda -y"; \
		exit 1; \
	fi
	@if ss -ltn "( sport = :$(REDPANDA_PORT) )" 2>/dev/null | grep -q ":$(REDPANDA_PORT)"; then \
		echo "Redpanda already listening on $(REDPANDA_PORT)"; \
	else \
		printf 'redpanda:\n  data_directory: %s/%s/redpanda\n  node_id: 0\n  empty_seed_starts_cluster: true\n  seed_servers: []\n  rpc_server:\n    address: 0.0.0.0\n    port: 33145\n  advertised_rpc_api:\n    address: 127.0.0.1\n    port: 33145\n  kafka_api:\n    - address: 0.0.0.0\n      port: %s\n  advertised_kafka_api:\n    - address: 127.0.0.1\n      port: %s\n  admin:\n    - address: 0.0.0.0\n      port: %s\n  developer_mode: true\n  auto_create_topics_enabled: true\nrpk:\n  overprovisioned: true\n' "$(CURDIR)" "$(DATA_DIR)" "$(REDPANDA_PORT)" "$(REDPANDA_PORT)" "$(REDPANDA_ADMIN_PORT)" > /etc/redpanda/redpanda.yaml; \
		nohup $(RPK_BIN) redpanda start --mode dev-container --check=false --smp 1 \
			> $(CURDIR)/$(REDPANDA_LOG_FILE) 2>&1 < /dev/null & echo $$! > $(CURDIR)/$(REDPANDA_PID_FILE); \
		for attempt in {1..200}; do \
			if $(RPK_BIN) --brokers 127.0.0.1:$(REDPANDA_PORT) cluster info >/dev/null 2>&1; then break; fi; \
			sleep 0.2; \
		done; \
	fi
	@if $(RPK_BIN) --brokers 127.0.0.1:$(REDPANDA_PORT) cluster info >/dev/null 2>&1; then \
		echo "Redpanda:      127.0.0.1:$(REDPANDA_PORT)  (admin :$(REDPANDA_ADMIN_PORT))"; \
	else \
		echo "WARNING: Redpanda failed to start; see $(REDPANDA_LOG_FILE)"; \
	fi

stop-minio:
	@if [[ -f "$(MINIO_PID_FILE)" ]]; then pid=$$(cat $(MINIO_PID_FILE)); kill $$pid 2>/dev/null || true; rm -f $(MINIO_PID_FILE); fi
	@pids=$$(ss -ltnp "( sport = :$(MINIO_PORT) )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u); \
	if [[ -n "$$pids" ]]; then kill $$pids 2>/dev/null || true; fi

stop-redpanda:
	@if [[ -f "$(REDPANDA_PID_FILE)" ]]; then pid=$$(cat $(REDPANDA_PID_FILE)); kill $$pid 2>/dev/null || true; rm -f $(REDPANDA_PID_FILE); fi
	@pids=$$(ss -ltnp "( sport = :$(REDPANDA_PORT) )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u); \
	if [[ -n "$$pids" ]]; then kill $$pids 2>/dev/null || true; fi

start-triton: ## Start Triton Inference Server in Docker (requires NVIDIA Container Toolkit)
	@mkdir -p $(RUN_DIR) $(LOG_DIR)
	@if command -v docker >/dev/null 2>&1; then \
		if docker ps --format '{{.Names}}' | grep -qx "$(TRITON_CONTAINER)"; then \
			echo "Triton already running as $(TRITON_CONTAINER)"; \
		else \
			docker rm -f $(TRITON_CONTAINER) >/dev/null 2>&1 || true; \
			nohup docker run --rm --name $(TRITON_CONTAINER) --gpus all \
				-p $(TRITON_HTTP_PORT):8000 -p $(TRITON_GRPC_PORT):8001 -p $(TRITON_METRICS_PORT):8002 \
				-v $(MODEL_REPOSITORY_DIR):/models:ro \
				$(TRITON_IMAGE) tritonserver --model-repository=/models --strict-model-config=false --model-control-mode=poll --repository-poll-secs=5 \
				> $(CURDIR)/$(TRITON_LOG_FILE) 2>&1 < /dev/null & echo $$! > $(CURDIR)/$(TRITON_PID_FILE); \
		fi; \
	elif [[ -x "$(TRITON_BIN)" ]]; then \
		if ss -ltn "( sport = :$(TRITON_HTTP_PORT) )" 2>/dev/null | grep -q ":$(TRITON_HTTP_PORT)"; then \
			echo "Triton already listening on $(TRITON_HTTP_PORT)"; \
		else \
			nohup $(TRITON_BIN) --model-repository=$(MODEL_REPOSITORY_DIR) \
				--http-port=$(TRITON_HTTP_PORT) --grpc-port=$(TRITON_GRPC_PORT) --metrics-port=$(TRITON_METRICS_PORT) \
				--strict-model-config=false --model-control-mode=poll --repository-poll-secs=5 \
				> $(CURDIR)/$(TRITON_LOG_FILE) 2>&1 < /dev/null & echo $$! > $(CURDIR)/$(TRITON_PID_FILE); \
		fi; \
	else \
		echo "Neither docker nor a native Triton binary was found."; \
		echo "Install Docker plus NVIDIA Container Toolkit, or make sure $(TRITON_BIN) exists and is executable."; \
		exit 1; \
	fi
	@ready=""; \
	for attempt in {1..240}; do \
		if curl -fsS http://127.0.0.1:$(TRITON_HTTP_PORT)/v2/health/ready >/dev/null 2>&1; then ready=1; break; fi; \
		sleep 0.5; \
	done; \
	if [[ -n "$$ready" ]]; then \
		echo "Triton:        http://127.0.0.1:$(TRITON_HTTP_PORT)  gRPC 127.0.0.1:$(TRITON_GRPC_PORT)  metrics :$(TRITON_METRICS_PORT)"; \
	else \
		echo "WARNING: Triton failed to become ready; see $(TRITON_LOG_FILE)"; \
	fi

stop-triton:
	@if command -v docker >/dev/null 2>&1; then docker rm -f $(TRITON_CONTAINER) >/dev/null 2>&1 || true; fi
	@if [[ -f "$(TRITON_PID_FILE)" ]]; then pid=$$(cat $(TRITON_PID_FILE)); kill $$pid 2>/dev/null || true; rm -f $(TRITON_PID_FILE); fi
	@for port in $(TRITON_HTTP_PORT) $(TRITON_GRPC_PORT) $(TRITON_METRICS_PORT); do \
		pids=$$(ss -ltnp "( sport = :$$port )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u); \
		if [[ -n "$$pids" ]]; then kill $$pids 2>/dev/null || true; fi; \
		for attempt in {1..50}; do \
			if ! ss -ltn "( sport = :$$port )" 2>/dev/null | grep -q ":$$port"; then break; fi; \
			sleep 0.2; \
		done; \
	done

start-prometheus: ## Start Prometheus in the background (scrapes API, Triton, DCGM, MinIO, Redpanda)
	@mkdir -p $(RUN_DIR) $(LOG_DIR) $(DATA_DIR)/prometheus
	@if [[ ! -x "$(PROMETHEUS_BIN)" ]]; then \
		echo "prometheus binary not found at $(PROMETHEUS_BIN)"; \
		echo "Run 'make setup-runtime' to download Prometheus."; \
		exit 1; \
	fi
	@if ss -ltn "( sport = :$(PROMETHEUS_PORT) )" 2>/dev/null | grep -q ":$(PROMETHEUS_PORT)"; then \
		echo "Prometheus already listening on $(PROMETHEUS_PORT)"; \
	else \
		nohup $(PROMETHEUS_BIN) \
			--config.file=$(CURDIR)/$(APP_DIR)/observability/prometheus.yml \
			--storage.tsdb.path=$(CURDIR)/$(DATA_DIR)/prometheus \
			--web.listen-address=:$(PROMETHEUS_PORT) \
			--web.enable-lifecycle \
			> $(CURDIR)/$(PROMETHEUS_LOG_FILE) 2>&1 < /dev/null & echo $$! > $(CURDIR)/$(PROMETHEUS_PID_FILE); \
		for attempt in {1..60}; do \
			if curl -fsS http://127.0.0.1:$(PROMETHEUS_PORT)/-/ready >/dev/null 2>&1; then break; fi; \
			sleep 0.2; \
		done; \
	fi
	@echo "Prometheus:    http://127.0.0.1:$(PROMETHEUS_PORT)"

stop-prometheus:
	@if [[ -f "$(PROMETHEUS_PID_FILE)" ]]; then pid=$$(cat $(PROMETHEUS_PID_FILE)); kill $$pid 2>/dev/null || true; rm -f $(PROMETHEUS_PID_FILE); fi
	@pids=$$(ss -ltnp "( sport = :$(PROMETHEUS_PORT) )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u); \
	if [[ -n "$$pids" ]]; then kill $$pids 2>/dev/null || true; fi

start-grafana: ## Start Grafana with auto-provisioned Prometheus datasource and InferOps dashboards
	@mkdir -p $(RUN_DIR) $(LOG_DIR) $(DATA_DIR)/grafana
	@if [[ ! -x "$(GRAFANA_BIN)" ]]; then \
		echo "grafana binary not found at $(GRAFANA_BIN)"; \
		echo "Run 'make setup-runtime' to download Grafana."; \
		exit 1; \
	fi
	@if ss -ltn "( sport = :$(GRAFANA_PORT) )" 2>/dev/null | grep -q ":$(GRAFANA_PORT)"; then \
		echo "Grafana already listening on $(GRAFANA_PORT)"; \
	else \
		GF_SECURITY_ADMIN_USER=admin GF_SECURITY_ADMIN_PASSWORD=admin \
		GF_PATHS_DATA=$(CURDIR)/$(DATA_DIR)/grafana \
		GF_PATHS_LOGS=$(CURDIR)/$(LOG_DIR) \
		GF_PATHS_PLUGINS=$(CURDIR)/$(DATA_DIR)/grafana/plugins \
		GF_PATHS_PROVISIONING=$(CURDIR)/$(APP_DIR)/observability/grafana/provisioning \
		GF_SERVER_HTTP_PORT=$(GRAFANA_PORT) \
		GF_SERVER_HTTP_ADDR=0.0.0.0 \
		GF_AUTH_ANONYMOUS_ENABLED=true \
		GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
		nohup $(GRAFANA_BIN) --homepath $(GRAFANA_DIR) \
			> $(CURDIR)/$(GRAFANA_LOG_FILE) 2>&1 < /dev/null & echo $$! > $(CURDIR)/$(GRAFANA_PID_FILE); \
		for attempt in {1..120}; do \
			if curl -fsS http://127.0.0.1:$(GRAFANA_PORT)/api/health >/dev/null 2>&1; then break; fi; \
			sleep 0.3; \
		done; \
	fi
	@echo "Grafana:       http://127.0.0.1:$(GRAFANA_PORT)  (admin/admin)"

stop-grafana:
	@if [[ -f "$(GRAFANA_PID_FILE)" ]]; then pid=$$(cat $(GRAFANA_PID_FILE)); kill $$pid 2>/dev/null || true; rm -f $(GRAFANA_PID_FILE); fi
	@pids=$$(ss -ltnp "( sport = :$(GRAFANA_PORT) )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u); \
	if [[ -n "$$pids" ]]; then kill $$pids 2>/dev/null || true; fi

ensure-topics: ## Create Redpanda topics required by InferOps (alerts, fusion, frames, evidence)
	@if [[ -z "$(RPK_BIN)" ]]; then echo "rpk not found; run 'make setup-runtime'"; exit 1; fi
	@for t in inferops.alerts inferops.fusion inferops.frames inferops.evidence; do \
		$(RPK_BIN) --brokers 127.0.0.1:$(REDPANDA_PORT) topic create $$t -p 3 -r 1 >/dev/null 2>&1 || true; \
	done
	@$(RPK_BIN) --brokers 127.0.0.1:$(REDPANDA_PORT) topic list 2>/dev/null | grep '^inferops\.' || true

ensure-buckets: ## Create MinIO buckets required by InferOps (inferops, inferops-clips)
	@if [[ ! -x "$(MC_BIN)" ]]; then echo "mc not found; run 'make setup-runtime'"; exit 1; fi
	@$(MC_BIN) alias set inferops-local http://127.0.0.1:$(MINIO_PORT) $(MINIO_ROOT_USER) $(MINIO_ROOT_PASSWORD) >/dev/null 2>&1 || true
	@for b in inferops inferops-clips; do $(MC_BIN) mb -p inferops-local/$$b >/dev/null 2>&1 || true; done
	@$(MC_BIN) ls inferops-local 2>/dev/null || true

status: ## Show health of every InferOps stack component
	@printf "%-14s %s\n" "API"        "$$(curl -fsS -o /dev/null -w %{http_code} http://127.0.0.1:$(API_PORT)/healthz 2>/dev/null || echo down)"
	@printf "%-14s %s\n" "Dashboard"  "$$(curl -fsS -o /dev/null -w %{http_code} http://127.0.0.1:$(DASHBOARD_PORT) 2>/dev/null || echo down)"
	@printf "%-14s %s\n" "MinIO"      "$$(curl -fsS -o /dev/null -w %{http_code} http://127.0.0.1:$(MINIO_PORT)/minio/health/live 2>/dev/null || echo down)"
	@printf "%-14s %s\n" "Redpanda"   "$$($(RPK_BIN) --brokers 127.0.0.1:$(REDPANDA_PORT) cluster info >/dev/null 2>&1 && echo up || echo down)"
	@printf "%-14s %s\n" "Triton"     "$$(curl -fsS -o /dev/null -w %{http_code} http://127.0.0.1:$(TRITON_HTTP_PORT)/v2/health/ready 2>/dev/null || echo down)"
	@printf "%-14s %s\n" "Prometheus" "$$(curl -fsS -o /dev/null -w %{http_code} http://127.0.0.1:$(PROMETHEUS_PORT)/-/ready 2>/dev/null || echo down)"
	@printf "%-14s %s\n" "Grafana"    "$$(curl -fsS -o /dev/null -w %{http_code} http://127.0.0.1:$(GRAFANA_PORT)/api/health 2>/dev/null || echo down)"
	@printf "%-14s %s\n" "DCGM"       "$$(curl -fsS -o /dev/null -w %{http_code} http://127.0.0.1:9400/metrics 2>/dev/null || echo down)"

serve: $(SERVE_DEPS) ## Start API, dashboard, MinIO, Redpanda, Prometheus, Grafana, and Triton when WITH_TRITON=1 or the domain.yaml says triton: true
	@mkdir -p $(RUN_DIR) $(LOG_DIR)
	@cd $(APP_DIR) && { \
		$(DOMAIN_ENV) \
		INFEROPS_TRANSPORT_KIND=redpanda \
		INFEROPS_TRANSPORT_BOOTSTRAP=127.0.0.1:$(REDPANDA_PORT) \
		INFEROPS_S3_ENDPOINT=http://127.0.0.1:$(MINIO_PORT) \
		INFEROPS_S3_ACCESS_KEY=$(MINIO_ROOT_USER) \
		INFEROPS_S3_SECRET_KEY=$(MINIO_ROOT_PASSWORD) \
		INFEROPS_TRITON_URL=127.0.0.1:$(TRITON_GRPC_PORT) \
		INFEROPS_TRITON_HTTP_URL=http://127.0.0.1:$(TRITON_HTTP_PORT) \
		nohup $(PY) -m uvicorn inferops.api.main:app --host 0.0.0.0 --port $(API_PORT) \
			> $(CURDIR)/$(API_LOG_FILE) 2>&1 < /dev/null & echo $$! > $(CURDIR)/$(API_PID_FILE); }
	@nohup $(PY) -m http.server $(DASHBOARD_PORT) --directory $(CURDIR)/frontend > $(CURDIR)/$(DASHBOARD_LOG_FILE) 2>&1 < /dev/null & echo $$! > $(CURDIR)/$(DASHBOARD_PID_FILE)
	@api_ready=""; \
	for attempt in {1..300}; do \
		if curl -fsS http://127.0.0.1:$(API_PORT)/healthz >/dev/null 2>&1; then api_ready=1; break; fi; \
		sleep 0.2; \
	done; \
	if [[ -z "$$api_ready" ]]; then echo "WARNING: API may not have started cleanly."; fi
	@dashboard_ready=""; \
	for attempt in {1..150}; do \
		if ss -ltn "( sport = :$(DASHBOARD_PORT) )" 2>/dev/null | grep -q ":$(DASHBOARD_PORT)"; then dashboard_ready=1; break; fi; \
		sleep 0.2; \
	done; \
	if [[ -z "$$dashboard_ready" ]]; then echo "WARNING: dashboard on port $(DASHBOARD_PORT) may not have started."; fi
	@echo "InferOps API:  http://127.0.0.1:$(API_PORT)"
	@echo "Dashboard:     http://127.0.0.1:$(DASHBOARD_PORT)/dashboard.html"
	@echo "Logs:          $(API_LOG_FILE)  $(DASHBOARD_LOG_FILE)  $(MINIO_LOG_FILE)  $(REDPANDA_LOG_FILE)"

serve-mock: stop ## Fallback: API + dashboard + Moto S3 mock + SQLite + inproc transport (no Redpanda/MinIO)
	@mkdir -p $(RUN_DIR) $(LOG_DIR) $(DATA_DIR)
	@cd $(APP_DIR) && { $(DOMAIN_ENV) INFEROPS_TRANSPORT_KIND=inproc INFEROPS_PG_URL=$(MOCK_PG_URL) nohup $(PY) -m uvicorn inferops.api.main:app --host 0.0.0.0 --port $(API_PORT) > $(CURDIR)/$(API_LOG_FILE) 2>&1 < /dev/null & echo $$! > $(CURDIR)/$(API_PID_FILE); }
	@nohup $(PY) -m http.server $(DASHBOARD_PORT) --directory $(CURDIR)/frontend > $(CURDIR)/$(DASHBOARD_LOG_FILE) 2>&1 < /dev/null & echo $$! > $(CURDIR)/$(DASHBOARD_PID_FILE)
	@nohup $(MOTO_SERVER) -p $(MINIO_PORT) > $(CURDIR)/$(MOTO_LOG_FILE) 2>&1 < /dev/null & echo $$! > $(CURDIR)/$(MOTO_PID_FILE)
	@for port in $(API_PORT) $(DASHBOARD_PORT) $(MINIO_PORT); do \
		for attempt in {1..150}; do \
			if ss -ltn "( sport = :$$port )" 2>/dev/null | grep -q ":$$port"; then break; fi; \
			sleep 0.2; \
		done; \
	done
	@$(PY) -c "exec(\"import boto3\\ns3 = boto3.client('s3', endpoint_url='http://127.0.0.1:$(MINIO_PORT)', region_name='us-east-1', aws_access_key_id='minio', aws_secret_access_key='minio12345')\\nfor b in ['inferops', 'inferops-clips']:\\n    try:\\n        s3.create_bucket(Bucket=b)\\n    except Exception:\\n        pass\\n\")" 2>/dev/null || true
	@echo "InferOps API (mock mode):  http://127.0.0.1:$(API_PORT)"
	@echo "Dashboard:                 http://127.0.0.1:$(DASHBOARD_PORT)/dashboard.html"
	@echo "S3 (moto mock):            http://127.0.0.1:$(MINIO_PORT)"

stop: stop-minio stop-redpanda stop-triton stop-prometheus stop-grafana ## Stop local API, dashboard, MinIO, Redpanda, Triton, Prometheus, Grafana, and moto mock
	@set +e; \
	for pidfile in "$(API_PID_FILE)" "$(DASHBOARD_PID_FILE)" "$(MOTO_PID_FILE)"; do \
		if [[ -f "$$pidfile" ]]; then \
			pid=$$(cat "$$pidfile"); \
			if kill -0 $$pid 2>/dev/null; then kill $$pid 2>/dev/null || true; fi; \
			rm -f "$$pidfile"; \
		fi; \
	done; \
	for port in $(API_PORT) $(DASHBOARD_PORT); do \
		pids=$$(ss -ltnp "( sport = :$$port )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u); \
		if [[ -n "$$pids" ]]; then kill $$pids 2>/dev/null || true; fi; \
		for attempt in {1..50}; do \
			if ! ss -ltn "( sport = :$$port )" 2>/dev/null | grep -q ":$$port"; then break; fi; \
			sleep 0.2; \
		done; \
	done; \
	echo "Stopped local services."

up: ## Start the InferOps compose stack
	cd $(APP_DIR) && docker compose up -d

down: ## Stop the InferOps compose stack
	cd $(APP_DIR) && docker compose down -v

logs: ## Follow compose logs
	cd $(APP_DIR) && docker compose logs -f --tail=200

worker-pipeline: ## Run a configured source-driven pipeline worker: DOMAIN=<pack> PIPELINE=<name>
	cd $(APP_DIR) && $(DOMAIN_ENV) $(PY) -m inferops.workers.pipeline_worker $(PIPELINE)

worker-topic: ## Run topic worker: DOMAIN=<pack> PIPELINE=<name> INPUT=in OUTPUT=out
	cd $(APP_DIR) && $(DOMAIN_ENV) $(PY) -m inferops.workers.topic_worker $(PIPELINE) $(INPUT) $(OUTPUT)
