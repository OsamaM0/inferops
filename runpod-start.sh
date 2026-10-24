#!/usr/bin/env bash
# Single-RunPod-Pod development deployment: all services run as native processes.
# Project, model weights, database, recordings, secrets and logs live on /workspace.
set -Eeuo pipefail
umask 077

APP=/workspace/inferops
DATA=/workspace/inferops-data
VENV=/workspace/.venvs/inferops
PGPORT=55432  # Internal-only port; avoids an apt-created PostgreSQL cluster on 5432.
MTX_VERSION=1.21.2  # Match deploy/compose.yaml.
PG_PID_STARTED=0
MTX_PID=''
API_PID=''

info() { printf '[inferops/runpod] %s\n' "$*"; }
fail() { printf '[inferops/runpod] ERROR: %s\n' "$*" >&2; exit 1; }

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ -n "$API_PID" ]]; then kill "$API_PID" 2>/dev/null || true; wait "$API_PID" 2>/dev/null || true; fi
    if [[ -n "$MTX_PID" ]]; then kill "$MTX_PID" 2>/dev/null || true; wait "$MTX_PID" 2>/dev/null || true; fi
    # Shut down only the PostgreSQL cluster managed by this script.
    if [[ "$PG_PID_STARTED" == 1 ]]; then
        runuser -u postgres -- "$PGBIN/pg_ctl" -D "$PGDATA" -m fast -w stop >/dev/null 2>&1 || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "$(id -u)" == 0 ]] || fail 'This RunPod bootstrap needs a root-based CUDA Pod image.'
[[ "$(uname -m)" == x86_64 ]] || fail 'This bootstrap currently supports x86_64 CUDA Pods.'
[[ -f "$APP/pyproject.toml" ]] || fail "Repository not found at $APP (clone it before calling this script)."
mkdir -p "$DATA"/{bin,logs,secrets,recordings,app}
# Root owns the shared /workspace data folder, while PostgreSQL runs as an
# unprivileged OS user. Grant directory *traversal* (not listing) so postgres
# can access its owned database/socket and log file. Keep secrets root-only.
chmod 0711 "$DATA" "$DATA/logs"
chmod 0700 "$DATA/secrets"

# OS packages disappear on a rebuilt Pod. Python/model artifacts stay on /workspace.
if ! command -v pg_config >/dev/null || ! command -v curl >/dev/null || ! command -v ffmpeg >/dev/null; then
    info 'Installing PostgreSQL, curl and FFmpeg...'
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends postgresql postgresql-client ca-certificates curl ffmpeg
fi
command -v python3.11 >/dev/null || fail 'Use a RunPod CUDA base image with Python 3.11.'
PGBIN="$(pg_config --bindir)"
[[ -x "$PGBIN/initdb" ]] || fail 'PostgreSQL server binaries are missing.'

# Database passwords are generated once and persisted; no development defaults.
password_file="$DATA/secrets/postgres_password"
if [[ ! -s "$password_file" ]]; then
    python3.11 -c 'import secrets; print(secrets.token_hex(24))' > "$password_file"
fi
chmod 600 "$password_file"
PGPASSWORD_LOCAL="$(cat "$password_file")"
PGDATA="$DATA/postgres"
PGSOCK="$DATA/postgres-socket"
PGLOG="$DATA/logs/postgres.log"
install -d -m 700 -o postgres -g postgres "$PGDATA" "$PGSOCK"
touch "$PGLOG"
chown postgres:postgres "$PGLOG"

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

if ! runuser -u postgres -- "$PGBIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
    info 'Starting PostgreSQL on loopback...'
    runuser -u postgres -- "$PGBIN/pg_ctl" -D "$PGDATA" \
        -o "-c listen_addresses=127.0.0.1 -p $PGPORT -c unix_socket_directories=$PGSOCK" \
        -l "$PGLOG" -w start >/dev/null
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
    python3.11 -c 'import secrets; print(secrets.token_hex(24))' > "$media_password_file"
fi
chmod 600 "$media_password_file"
MEDIA_PASSWORD="$(cat "$media_password_file")"
MTX_CONFIG="$DATA/mediamtx.yml"
python3.11 - "$APP/deploy/media/mediamtx.yml" "$MTX_CONFIG" "$DATA/recordings" "$MEDIA_PASSWORD" <<'PY'
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
"$MTX" "$MTX_CONFIG" >> "$DATA/logs/mediamtx.log" 2>&1 &
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
export INFEROPS_EXECUTION_MODE=cuda
export INFEROPS_DOMAIN_ROOT="$APP/projects/driver_behaviour"
export INFEROPS_CONFIGS_DIR="$APP/configs"
export INFEROPS_STORAGE_KIND=filesystem
export INFEROPS_MEDIA_ROUTER=mediamtx
export HF_HOME="$DATA/huggingface"
mkdir -p "$HF_HOME"

# Install/update Python requirements when the project dependency spec changes.
if [[ ! -x "$VENV/bin/inferops" ]]; then
    info 'Creating InferOps Python virtual environment...'
    python3.11 -m venv "$VENV"
fi
spec_hash="$(sha256sum pyproject.toml | cut -d' ' -f1)"
if [[ "$(cat "$DATA/python-requirements.sha256" 2>/dev/null || true)" != "$spec_hash" ]]; then
    info 'Installing InferOps CUDA dependencies...'
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install -e '.[server,cuda,video]'
    printf '%s\n' "$spec_hash" > "$DATA/python-requirements.sha256"
fi

info 'Preparing project model artifacts...'
"$VENV/bin/inferops" project prepare
info 'Starting InferOps at http://localhost:8080/ui/ (RunPod HTTP port 8080)...'
"$VENV/bin/python" scripts/serve_local.py \
    --runtime cuda --host 0.0.0.0 --port 8080 --data "$DATA/app" &
API_PID=$!
wait "$API_PID"
