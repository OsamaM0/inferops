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
DOMAIN ?=
DOMAIN_ENV := $(if $(DOMAIN),INFEROPS_ACTIVE_DOMAIN=$(DOMAIN),)
DOMAIN_ARG := $(if $(DOMAIN),--domain $(DOMAIN),)
MINIO_PORT ?= 9000
MINIO_CONSOLE_PORT ?= 9001
REDPANDA_PORT ?= 19092
REDPANDA_ADMIN_PORT ?= 19644
MINIO_ROOT_USER ?= minio
MINIO_ROOT_PASSWORD ?= minio12345
API_PID_FILE := $(RUN_DIR)/inferops-api.pid
DASHBOARD_PID_FILE := $(RUN_DIR)/dashboard.pid
MINIO_PID_FILE := $(RUN_DIR)/minio.pid
REDPANDA_PID_FILE := $(RUN_DIR)/redpanda.pid
MOTO_PID_FILE := $(RUN_DIR)/moto.pid
API_LOG_FILE := $(LOG_DIR)/inferops-api.log
DASHBOARD_LOG_FILE := $(LOG_DIR)/dashboard.log
MINIO_LOG_FILE := $(LOG_DIR)/minio.log
REDPANDA_LOG_FILE := $(LOG_DIR)/redpanda.log
MOTO_LOG_FILE := $(LOG_DIR)/moto-s3.log
MOTO_SERVER := $(CURDIR)/$(APP_DIR)/.venv/bin/moto_server
MINIO_BIN ?= $(shell command -v minio 2>/dev/null || echo $(TOOLS_BIN_DIR)/minio)
MC_BIN ?= $(shell command -v mc 2>/dev/null || echo $(TOOLS_BIN_DIR)/mc)
REDPANDA_BIN ?= $(shell command -v redpanda 2>/dev/null)
RPK_BIN ?= $(shell command -v rpk 2>/dev/null)

.PHONY: help setup install dev-install setup-models test lint fmt validate serve serve-mock serve-api serve-dashboard stop up down logs \
	start-minio start-redpanda stop-minio stop-redpanda worker-pipeline worker-topic setup-runtime

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "%-22s %s\n",$$1,$$2}'

setup: ## Install InferOps core and native serve prerequisites, optionally with DOMAIN=driver_behaviour
	@if [[ ! -x "$(VENV_PY)" ]]; then python3 -m venv $(APP_DIR)/.venv; fi
	cd $(APP_DIR) && $(VENV_PY) scripts/setup_inferops.py $(DOMAIN_ARG)
	@if [[ "$(SKIP_RUNTIME)" != "1" ]]; then $(MAKE) setup-runtime; fi

setup-runtime: ## Install native MinIO/mc and Redpanda/rpk prerequisites for make serve
	cd $(APP_DIR) && $(PY) scripts/setup_native_runtime.py

install: ## Install InferOps with dev dependencies
	cd $(APP_DIR) && $(PY) -m pip install -e ".[dev]"

dev-install: install ## Alias for install

setup-models: ## Build/download example models
	cd $(APP_DIR) && $(PY) scripts/make_tiny_model.py && $(PY) scripts/setup_inferops.py --domain driver_behaviour --skip-install

test: ## Run InferOps tests
	cd $(APP_DIR) && $(PY) -m pytest -q

lint: ## Lint InferOps
	cd $(APP_DIR) && $(PY) -m ruff check .

fmt: ## Format InferOps
	cd $(APP_DIR) && $(PY) -m ruff format . && $(PY) -m ruff check --fix .

validate: ## Validate InferOps YAML configs
	cd $(APP_DIR) && $(PY) -m inferops.cli validate $(DOMAIN_ARG)


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

serve: stop start-minio start-redpanda ## Start API, dashboard, real MinIO, and Redpanda locally (primary stack)
	@mkdir -p $(RUN_DIR) $(LOG_DIR)
	@cd $(APP_DIR) && { \
		$(DOMAIN_ENV) \
		INFEROPS_TRANSPORT_KIND=redpanda \
		INFEROPS_TRANSPORT_BOOTSTRAP=127.0.0.1:$(REDPANDA_PORT) \
		INFEROPS_S3_ENDPOINT=http://127.0.0.1:$(MINIO_PORT) \
		INFEROPS_S3_ACCESS_KEY=$(MINIO_ROOT_USER) \
		INFEROPS_S3_SECRET_KEY=$(MINIO_ROOT_PASSWORD) \
		nohup $(PY) -m uvicorn inferops.api.main:app --host 0.0.0.0 --port $(API_PORT) \
			> $(CURDIR)/$(API_LOG_FILE) 2>&1 < /dev/null & echo $$! > $(CURDIR)/$(API_PID_FILE); }
	@nohup $(PY) -m http.server $(DASHBOARD_PORT) --directory $(CURDIR)/frontend > $(CURDIR)/$(DASHBOARD_LOG_FILE) 2>&1 < /dev/null & echo $$! > $(CURDIR)/$(DASHBOARD_PID_FILE)
	@for port in $(API_PORT) $(DASHBOARD_PORT); do \
		ready=""; \
		for attempt in {1..150}; do \
			if ss -ltn "( sport = :$$port )" 2>/dev/null | grep -q ":$$port"; then ready=1; break; fi; \
			sleep 0.2; \
		done; \
		if [[ -z "$$ready" ]]; then echo "WARNING: service on port $$port may not have started."; fi; \
	done
	@echo "InferOps API:  http://127.0.0.1:$(API_PORT)"
	@echo "Dashboard:     http://127.0.0.1:$(DASHBOARD_PORT)/dashboard.html"
	@echo "Logs:          $(API_LOG_FILE)  $(DASHBOARD_LOG_FILE)  $(MINIO_LOG_FILE)  $(REDPANDA_LOG_FILE)"

serve-mock: stop ## Fallback: API + dashboard + Moto S3 mock + inproc transport (no Redpanda/MinIO)
	@mkdir -p $(RUN_DIR) $(LOG_DIR)
	@cd $(APP_DIR) && { $(DOMAIN_ENV) INFEROPS_TRANSPORT_KIND=inproc nohup $(PY) -m uvicorn inferops.api.main:app --host 0.0.0.0 --port $(API_PORT) > $(CURDIR)/$(API_LOG_FILE) 2>&1 < /dev/null & echo $$! > $(CURDIR)/$(API_PID_FILE); }
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

stop: stop-minio stop-redpanda ## Stop local API, dashboard, MinIO, Redpanda, and moto mock
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

worker-pipeline: ## Run a configured source-driven pipeline worker, e.g. PIPELINE=driver_stream
	cd $(APP_DIR) && $(DOMAIN_ENV) $(PY) -m inferops.workers.pipeline_worker $(PIPELINE)

worker-topic: ## Run topic worker, e.g. PIPELINE=driver_video INPUT=in OUTPUT=out
	cd $(APP_DIR) && $(DOMAIN_ENV) $(PY) -m inferops.workers.topic_worker $(PIPELINE) $(INPUT) $(OUTPUT)
