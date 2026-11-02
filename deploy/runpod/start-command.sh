#!/usr/bin/env bash
# RunPod boot script: SSH, clone on first boot, then keep InferOps running.
# The Pod's one-line start command downloads this file from GitHub and runs it
# (see deploy/runpod/start-<mode>.txt), so every boot uses the version on main.
set -e

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server git ca-certificates

mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
/usr/sbin/sshd
echo "SSH server started successfully"

if [ ! -f /workspace/inferops/pyproject.toml ]; then
  git clone --branch "${INFEROPS_GIT_REF:-main}" \
    "${INFEROPS_REPO_URL:-https://github.com/OsamaM0/inferops.git}" /workspace/inferops \
    || echo "git clone failed; clone it over SSH"
fi

# The app restarts after a crash; SSH stays up even if it never starts.
mkdir -p /workspace/inferops-data/logs
(
  while true; do
    if [ -f /workspace/inferops/runpod-start.sh ]; then
      bash /workspace/inferops/runpod-start.sh || echo "[inferops/runpod] stopped (exit $?); restarting in 15s"
    fi
    sleep 15
  done
) >> /workspace/inferops-data/logs/runpod-start.log 2>&1 &

exec sleep infinity
