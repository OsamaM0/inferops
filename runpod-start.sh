#!/usr/bin/env bash
# Single-RunPod-Pod deployment: all services run as native processes (no Docker).
# Project, model weights, database, recordings, secrets and logs live on /workspace.
#
# Runtime: INFEROPS_EXECUTION_MODE=cpu|cuda|triton|auto, else runtime.mode in
# configs/inferops.yaml. Works on any root Debian/Ubuntu x86_64 image; triton mode
# needs an NVIDIA Triton image (nvcr.io/nvidia/tritonserver:<yy.mm>-py3).
set -Eeuo pipefail
umask 077

APP="${INFEROPS_APP:-/workspace/inferops}"
DATA="${INFEROPS_RUNPOD_DATA:-/workspace/inferops-data}"
PGPORT=55432  # Internal-only port; avoids an apt-created PostgreSQL cluster on 5432.
MTX_VERSION=1.21.2  # Match deploy/compose.yaml.
TRITON_BIN="${INFEROPS_TRITONSERVER:-/opt/tritonserver/bin/tritonserver}"
PG_PID_STARTED=0
MTX_PID=''
TRITON_PID=''
API_PID=''

info() { printf '[inferops/runpod] %s\n' "$*"; }
fail() { printf '[inferops/runpod] ERROR: %s\n' "$*" >&2; exit 1; }

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    for pid in "$API_PID" "$TRITON_PID" "$MTX_PID"; do
        if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
    done
    # Shut down only the PostgreSQL cluster managed by this script.
    if [[ "$PG_PID_STARTED" == 1 ]]; then
        runuser -u postgres -- "$PGBIN/pg_ctl" -D "$PGDATA" -m fast -w stop >/dev/null 2>&1 || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "$(id -u)" == 0 ]] || fail 'This RunPod bootstrap needs a root-based Pod image.'
[[ "$(uname -m)" == x86_64 ]] || fail 'This bootstrap currently supports x86_64 Pods.'
[[ -f "$APP/pyproject.toml" ]] || fail "Repository not found at $APP (clone it before calling this script)."
mkdir -p "$DATA"/{bin,logs,secrets,recordings,app}
# Root owns the shared /workspace data folder, while PostgreSQL runs as an
# unprivileged OS user. Grant directory *traversal* (not listing) so postgres
# can access its owned database/socket and log file. Keep secrets root-only.
chmod 0711 "$DATA" "$DATA/logs"
chmod 0700 "$DATA/secrets"

# One instance per Pod: a second copy would fight over ports and the database.
# Services started below close fd 9, so a killed script never leaves the lock held.
exec 9>"$DATA/runpod-start.lock"
flock -n 9 || fail "Another runpod-start.sh is already running (see $DATA/logs/runpod-start.log)."

# Explicit environment wins, matching how the server resolves configs/inferops.yaml.
MODE="${INFEROPS_EXECUTION_MODE:-$(awk '/^runtime:/ {r=1; next} /^[^[:space:]#]/ {r=0}
    r && $1 == "mode:" {print $2; exit}' "$APP/configs/inferops.yaml")}"
MODE="${MODE:-auto}"
if [[ "$MODE" == auto ]]; then
    if nvidia-smi -L >/dev/null 2>&1; then MODE=cuda; else MODE=cpu; fi
fi
case "$MODE" in
    cpu) EXTRAS=server,video,cpu ;;
    cuda) EXTRAS=server,video,cuda ;;
    # The API only talks to Triton; CPU ONNX Runtime covers local helper imports.
    triton) EXTRAS=server,video,triton,cpu ;;
    *) fail "Unsupported INFEROPS_EXECUTION_MODE=$MODE (use cpu, cuda, triton or auto)." ;;
esac
export INFEROPS_EXECUTION_MODE="$MODE"
if [[ "$MODE" != cpu ]]; then
    nvidia-smi -L >/dev/null 2>&1 || fail "Runtime $MODE needs a GPU Pod (nvidia-smi found no GPU)."
fi
if [[ "$MODE" == triton && ! -x "$TRITON_BIN" ]]; then
    fail "Runtime triton needs tritonserver at $TRITON_BIN; use the nvcr.io/nvidia/tritonserver image."
fi
info "Runtime: $MODE"

# OS packages disappear on a rebuilt Pod. Python/model artifacts stay on /workspace.
python_ok() { python3 -c 'import sys, ensurepip; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; }
if ! compgen -G '/usr/lib/postgresql/*/bin/initdb' >/dev/null || ! command -v curl >/dev/null \
    || ! command -v ffmpeg >/dev/null || ! python_ok; then
    info 'Installing PostgreSQL, Python venv, curl and FFmpeg...'
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends postgresql postgresql-client ca-certificates curl ffmpeg \
        python3 python3-venv python3-dev libgl1 libglib2.0-0
fi
python_ok || fail 'Python 3.10+ with venv support is required.'
PGBIN="$(compgen -G '/usr/lib/postgresql/*/bin' | sort -V | tail -n 1)"
[[ -x "$PGBIN/initdb" ]] || fail 'PostgreSQL server binaries are missing.'

# Database passwords are generated once and persisted; no development defaults.
password_file="$DATA/secrets/postgres_password"
if [[ ! -s "$password_file" ]]; then
    python3 -c 'import secrets; print(secrets.token_hex(24))' > "$password_file"
fi
chmod 600 "$password_file"
PGPASSWORD_LOCAL="$(cat "$password_file")"
PGDATA="$DATA/postgres"
PGSOCK="$DATA/postgres-socket"
PGLOG="$DATA/logs/postgres.log"
install -d -m 700 -o postgres -g postgres "$PGDATA" "$PGSOCK"
touch "$PGLOG"
chown postgres:postgres "$PGLOG"
# The postgres UID can differ between images; keep the persistent cluster readable.
chown -R postgres:postgres "$PGDATA" "$PGSOCK"

# The default apt-created 'main' cluster, if any, is not this persistent cluster.
# Stop it to avoid wasting resources or accidentally using an ephemeral DB.
service postgresql stop >/dev/null 2>&1 || true

if [[ ! -f "$PGDATA/PG_VERSION" ]]; then
    info 'Initializing persistent PostgreSQL database...'
    pw_temp="$PGSOCK/initdb-password"
    printf '%s\n' "$PGPASSWORD_LOCAL" > "$pw_temp"
    chown postgres:postgres "$pw_temp"
    chmod 600 "$pw_temp"
    runuser -u postgres -- test -r "$pw_temp" || fail "postgres cannot read its initdb password file; check /workspace permissions"
    runuser -u postgres -- "$PGBIN/initdb" -D "$PGDATA" --username=postgres \
        --auth-local=peer --auth-host=scram-sha-256 --pwfile="$pw_temp" >/dev/null
    rm -f "$pw_temp"
fi
# The cluster persists across images; it must run with its own major version.
cluster_version="$(cat "$PGDATA/PG_VERSION")"
PGBIN="/usr/lib/postgresql/$cluster_version/bin"
[[ -x "$PGBIN/postgres" ]] || fail "Database in $PGDATA needs PostgreSQL $cluster_version; this image has another \
version. Use an image with the same Ubuntu/Debian release, or move $PGDATA aside to start a new database."

if ! "$PGBIN/pg_isready" -q -h 127.0.0.1 -p "$PGPORT"; then
    # After a Pod restart, the old postmaster PID may belong to an unrelated process.
    # The flock above guarantees no other copy of this script owns the cluster.
    rm -f "$PGDATA/postmaster.pid" "$PGSOCK"/.s.PGSQL.*
    info 'Starting PostgreSQL on loopback...'
    runuser -u postgres -- "$PGBIN/pg_ctl" -D "$PGDATA" \
        -o "-c listen_addresses=127.0.0.1 -p $PGPORT -c unix_socket_directories=$PGSOCK" \
        -l "$PGLOG" -w start >/dev/null 9>&- || fail "PostgreSQL did not start; see $PGLOG"
    PG_PID_STARTED=1
fi

if ! PGPASSWORD="$PGPASSWORD_LOCAL" "$PGBIN/psql" \
    -h 127.0.0.1 -p "$PGPORT" -U postgres -d postgres \
    -Atqc "SELECT 1 FROM pg_database WHERE datname='inferops'" | grep -qx 1; then
    info 'Creating inferops database...'
    PGPASSWORD="$PGPASSWORD_LOCAL" "$PGBIN/createdb" \
        -h 127.0.0.1 -p "$PGPORT" -U postgres inferops
fi

# Intentionally replace old RunPod env values pointing to nonexistent DB/Media hosts.
unset INFEROPS_PG_URL
export INFEROPS_DATABASE_URL="postgresql+asyncpg://postgres:${PGPASSWORD_LOCAL}@127.0.0.1:${PGPORT}/inferops"

# Download and verify the version pinned by this project's Compose stack.
MTX="$DATA/bin/mediamtx-v${MTX_VERSION}"
if [[ ! -x "$MTX" ]]; then
    info "Downloading MediaMTX v${MTX_VERSION}..."
    temp_dir="$(mktemp -d)"
    base_url="https://github.com/bluenviron/mediamtx/releases/download/v${MTX_VERSION}"
    archive="mediamtx_v${MTX_VERSION}_linux_amd64.tar.gz"
    curl -fsSL --retry 3 "$base_url/$archive" -o "$temp_dir/$archive"
    curl -fsSL --retry 3 "$base_url/checksums.sha256" -o "$temp_dir/checksums.sha256"
    (cd "$temp_dir" && sha256sum --check --ignore-missing checksums.sha256)
    tar -xzf "$temp_dir/$archive" -C "$temp_dir" mediamtx
    install -m 0755 "$temp_dir/mediamtx" "$MTX"
    rm -rf "$temp_dir"
fi

media_password_file="$DATA/secrets/mediamtx_password"
if [[ ! -s "$media_password_file" ]]; then
    python3 -c 'import secrets; print(secrets.token_hex(24))' > "$media_password_file"
fi
chmod 600 "$media_password_file"
MEDIA_PASSWORD="$(cat "$media_password_file")"
MTX_CONFIG="$DATA/mediamtx.yml"
python3 - "$APP/deploy/media/mediamtx.yml" "$MTX_CONFIG" "$DATA/recordings" "$MEDIA_PASSWORD" <<'PY'
from pathlib import Path
import sys
source, target, recordings, password = sys.argv[1:]
config = Path(source).read_text()
replacements = {
    'pass: inferops-local-api': f'pass: {password}',
    'recordPath: /recordings/': f'recordPath: {recordings}/',
    'apiAddress: :9997': 'apiAddress: 127.0.0.1:9997',
}
for original, replacement in replacements.items():
    if original not in config:
        raise SystemExit(f'MediaMTX project config changed: expected {original!r}')
    config = config.replace(original, replacement, 1)
Path(target).write_text(config)
PY

info 'Starting MediaMTX...'
"$MTX" "$MTX_CONFIG" >> "$DATA/logs/mediamtx.log" 2>&1 9>&- &
MTX_PID=$!
export INFEROPS_MEDIAMTX_API_URL=http://127.0.0.1:9997
export INFEROPS_MEDIAMTX_RTSP_URL=rtsp://127.0.0.1:8554
export INFEROPS_MEDIAMTX_PLAYBACK_URL=http://127.0.0.1:9996
export INFEROPS_MEDIAMTX_API_USER=inferops
export INFEROPS_MEDIAMTX_API_PASSWORD="$MEDIA_PASSWORD"
for attempt in $(seq 1 60); do
    if curl -fsS -u "inferops:$MEDIA_PASSWORD" \
        'http://127.0.0.1:9997/v3/config/global/get' >/dev/null 2>&1; then
        break
    fi
    kill -0 "$MTX_PID" 2>/dev/null || fail "MediaMTX exited; see $DATA/logs/mediamtx.log"
    if [[ "$attempt" == 60 ]]; then fail "MediaMTX not ready; see $DATA/logs/mediamtx.log"; fi
    sleep 1
done

cd "$APP"
export INFEROPS_CONFIGS_DIR="$APP/configs"
export HF_HOME="$DATA/huggingface"
mkdir -p "$HF_HOME"

# CPU and CUDA ONNX Runtime share one module name, so each runtime gets its own
# environment. Reinstall when the dependency spec, extras or Python changes.
VENV="/workspace/.venvs/inferops-$MODE"
if [[ ! -x "$VENV/bin/python" ]] || ! "$VENV/bin/python" -c '' 2>/dev/null; then
    info "Creating InferOps Python virtual environment ($VENV)..."
    rm -rf "$VENV"
    python3 -m venv "$VENV"
fi
spec_hash="$( { sha256sum pyproject.toml; echo "$EXTRAS"; "$VENV/bin/python" -V; } | sha256sum | cut -d' ' -f1)"
stamp="$VENV/.inferops-requirements.sha256"
if [[ "$(cat "$stamp" 2>/dev/null || true)" != "$spec_hash" ]]; then
    info "Installing InferOps dependencies [$EXTRAS]..."
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install -e ".[$EXTRAS]"
    printf '%s\n' "$spec_hash" > "$stamp"
fi

info 'Preparing project model artifacts...'
"$VENV/bin/inferops" project prepare

if [[ "$MODE" == triton ]]; then
    REPOSITORY="$("$VENV/bin/python" -c 'from scripts.run_mode import runtime_environment
print(runtime_environment(None, None)["INFEROPS_TRITON_REPOSITORY"])' | tail -n 1)"
    # Load only complete model folders; e.g. a checkpoint folder holds no config.pbtxt.
    load_args=()
    for config in "$REPOSITORY"/*/config.pbtxt; do
        [[ -f "$config" ]] && load_args+=("--load-model=$(basename "$(dirname "$config")")")
    done
    ((${#load_args[@]})) || fail "No Triton models with config.pbtxt in $REPOSITORY"
    info "Starting Triton with ${#load_args[@]} models from $REPOSITORY..."
    # Flags match the triton service in deploy/compose.yaml.
    "$TRITON_BIN" --model-repository="$REPOSITORY" --strict-model-config=true \
        --model-control-mode=explicit "${load_args[@]}" --exit-on-error=true \
        --rate-limit=execution_count --pinned-memory-pool-byte-size=134217728 \
        --cuda-memory-pool-byte-size=0:134217728 --allow-metrics=true --allow-gpu-metrics=true \
        >> "$DATA/logs/triton.log" 2>&1 9>&- &
    TRITON_PID=$!
    export INFEROPS_TRITON_URL=127.0.0.1:8001
    export INFEROPS_TRITON_HTTP_URL=http://127.0.0.1:8000
    export INFEROPS_TRITON_METRICS_URL=http://127.0.0.1:8002
    for attempt in $(seq 1 300); do
        if curl -fsS http://127.0.0.1:8000/v2/health/ready >/dev/null 2>&1; then break; fi
        kill -0 "$TRITON_PID" 2>/dev/null || fail "Triton exited; see $DATA/logs/triton.log"
        if [[ "$attempt" == 300 ]]; then fail "Triton not ready; see $DATA/logs/triton.log"; fi
        sleep 1
    done
fi

info "Starting InferOps API on port ${INFEROPS_API_PORT:-runtime.api_port} (runtime $MODE)..."
"$VENV/bin/python" scripts/serve_local.py --data "$DATA/app" 9>&- &
API_PID=$!
wait "$API_PID"
