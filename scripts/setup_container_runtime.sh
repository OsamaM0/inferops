#!/usr/bin/env bash
# NVIDIA's Debian/Ubuntu installation path. Installs the toolkit only;
# starts no Triton or DeepStream container.
set -euo pipefail

printf '%s\n' 'Installing NVIDIA Container Toolkit; Docker will be restarted.'
task_tmp_dir=$(mktemp -d)
trap 'rm -rf "$task_tmp_dir"' EXIT
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey -o "$task_tmp_dir/key.asc"
gpg --batch --dearmor -o "$task_tmp_dir/key.gpg" "$task_tmp_dir/key.asc"
sudo install -m 0644 "$task_tmp_dir/key.gpg" /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  -o "$task_tmp_dir/toolkit.list"
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  "$task_tmp_dir/toolkit.list" > "$task_tmp_dir/signed.list"
sudo install -m 0644 "$task_tmp_dir/signed.list" /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info --format '{{range $name, $runtime := .Runtimes}}{{$name}} {{end}}'
printf '%s\n' 'Toolkit configured. Triton and DeepStream have not been started.'
