
#!/usr/bin/env bash
set -euo pipefail

APP=/workspace/inferops
VENV=/workspace/.venvs/inferops

# Download source on first boot; update on later starts
if [ ! -d "$APP/.git" ]; then
    git clone --depth 1 \
      https://github.com/OsamaM0/inferops.git "$APP"
else
    git -C "$APP" pull --ff-only
fi

cd "$APP"

# Install CUDA inference dependencies once
if [ ! -x "$VENV/bin/inferops" ]; then
    python3.11 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install -e '.[server,cuda,video]'
fi

# External services required by the native server
: "${INFEROPS_DATABASE_URL:?Set PostgreSQL connection URL}"
: "${INFEROPS_MEDIAMTX_API_URL:?Set MediaMTX API URL}"
: "${INFEROPS_MEDIAMTX_RTSP_URL:?Set MediaMTX RTSP URL}"

export INFEROPS_EXECUTION_MODE=cuda
export INFEROPS_DOMAIN_ROOT="$APP/projects/driver_behaviour"
export INFEROPS_CONFIGS_DIR="$APP/configs"
export INFEROPS_STORAGE_KIND=filesystem
export INFEROPS_MEDIA_ROUTER=mediamtx

mkdir -p /workspace/inferops-data

# Prepare project model weights
"$VENV/bin/inferops" project prepare

# Launch API and dashboard on the GPU Pod
exec "$VENV/bin/python" scripts/serve_local.py \
    --runtime cuda \
    --host 0.0.0.0 \
    --port 8080 \
    --data /workspace/inferops-data
