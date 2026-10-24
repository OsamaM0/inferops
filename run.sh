#!/usr/bin/env bash
# Docker-only entry point: no host Python environment or model dependencies.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

usage() {
    cat <<'EOF'
Usage: ./run.sh [up|repair|stop|status|logs|check] [options]

  up             Build and start enabled services; wait until the API is ready (default)
  repair         Recreate the API container and restore its network; keep all data volumes
  stop           Stop services, keeping containers and data
  status         Show service status
  logs           Follow the last 100 lines of service logs
  check          Check Docker, configuration, API networking and port availability

Options:
  --mode MODE    cpu, auto, cuda or triton; otherwise use .env (default: cpu)
  --port PORT    Published API port; otherwise use .env (default: 8080)
  --no-build     Reuse existing images for a faster start
  -h, --help     Show this help

Examples:
  ./run.sh
  ./run.sh --mode cpu --port 8082
  ./run.sh repair
  ./run.sh logs
EOF
}

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
action=up
build=(--build)
while (($#)); do
    case "$1" in
        up|repair|stop|status|logs|check) action=$1; shift ;;
        --mode)
            (($# >= 2)) || fail '--mode requires cpu, auto, cuda or triton'
            case "$2" in cpu|auto|cuda|triton) ;; *) fail "Unknown runtime: $2" ;; esac
            export INFEROPS_EXECUTION_MODE=$2
            shift 2 ;;
        --port)
            (($# >= 2)) || fail '--port requires a port number'
            [[ "$2" =~ ^[0-9]{1,5}$ ]] || fail 'Port must be between 1 and 65535'
            ((10#$2 > 0 && 10#$2 <= 65535)) || fail 'Port must be between 1 and 65535'
            export INFEROPS_API_PORT=$((10#$2))
            shift 2 ;;
        --no-build) build=(); shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; fail "Unknown argument: $1" ;;
    esac
done

command -v docker >/dev/null || fail 'Install Docker Engine or Docker Desktop, then run ./run.sh again.'
docker compose version >/dev/null 2>&1 || fail 'Docker Compose 2.20.3+ is required.'
if ! docker info >/dev/null 2>&1; then
    fail 'Cannot access Docker. Start Docker Engine/Desktop and check your Docker socket permissions (docker info).'
fi
if [[ ! -f .env ]]; then
    cp -- .env.example .env
    printf 'Created .env from .env.example.\n'
fi

compose=(docker compose --project-directory "$PWD" -f "$PWD/compose.yaml")
"${compose[@]}" config --quiet
case "$action" in
    stop) exec "${compose[@]}" stop -t 60 ;;
    status) exec "${compose[@]}" ps -a ;;
    logs) exec "${compose[@]}" logs -f --tail=100 ;;
esac

# Ask Compose to resolve .env interpolation and shell overrides; never source .env.
configuration=$("${compose[@]}" config --environment)
setting() { awk -v key="$1" 'index($0, key "=") == 1 {print substr($0, length(key) + 2); exit}' <<< "$configuration"; }
mode=$(setting INFEROPS_EXECUTION_MODE)
port=$(setting INFEROPS_API_PORT)
mode=${mode:-cpu}
port=${port:-8080}
[[ "$port" =~ ^[0-9]{1,5}$ ]] || fail 'INFEROPS_API_PORT must be between 1 and 65535'
((10#$port > 0 && 10#$port <= 65535)) || fail 'INFEROPS_API_PORT must be between 1 and 65535'
port=$((10#$port))
printf 'InferOps: runtime=%s, API port=%s\n' "$mode" "$port"

api_id=$("${compose[@]}" ps -a -q api)
# A detached existing container can start without joining the Compose network.
# Recreating that container restores service-name DNS without deleting volumes.
detached=false
if [[ -n "$api_id" ]]; then
    networks=$(docker inspect --format '{{len .NetworkSettings.Networks}}' "$api_id")
    if [[ "$networks" == 0 ]]; then
        detached=true
        printf 'The existing API container has no Docker network; it needs recreation.\n'
    fi
fi

# Diagnose Docker port owners even when no host listener is visible (Docker Desktop/WSL).
containers=$(docker ps --format '{{.ID}} {{.Names}} {{.Ports}}')
owner=$(awk -v port=":$port->" -v api="$api_id" 'index($0, port) && !(api != "" && index(api, $1) == 1) {print $2; exit}' <<< "$containers")
if [[ -n "$owner" ]]; then
    fail "Port $port is used by Docker container '$owner'. Choose a free port: ./run.sh --port <port>"
fi
api_owns_port=$(awk -v port=":$port->" -v api="$api_id" 'index($0, port) && api != "" && index(api, $1) == 1 {print "true"; exit}' <<< "$containers")
if [[ "$api_owns_port" != true ]] && (echo >"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    fail "Port $port is already in use. Choose a free port: ./run.sh --port <port>"
fi

if [[ "$action" == check ]]; then
    if [[ "$detached" == true ]]; then
        fail 'API networking needs repair. Run ./run.sh repair (add --port <port> if needed).'
    fi
    printf 'Docker and Compose configuration are ready.\n'
    exit 0
fi
if [[ "$action" == repair || "$detached" == true ]]; then
    "${compose[@]}" stop -t 60 api
    "${compose[@]}" rm -f api
fi

printf 'Preparing models and starting services. First startup may download weights and images.\n'
if ! "${compose[@]}" up -d "${build[@]}" --wait --wait-timeout 600; then
    printf '\nStartup failed. Service status and recent logs:\n' >&2
    "${compose[@]}" ps -a || true
    "${compose[@]}" logs --tail=60 api prepare postgres || true
    printf '\nFor missing API networking: ./run.sh repair\nFor all logs: ./run.sh logs\n' >&2
    exit 1
fi
printf '\nInferOps is ready: http://localhost:%s/ui/\nAPI docs: http://localhost:%s/docs\nStop: ./run.sh stop\nLogs: ./run.sh logs\n' "$port" "$port"
