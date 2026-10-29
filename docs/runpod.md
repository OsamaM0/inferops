# RunPod native startup

`runpod-start.sh` runs PostgreSQL, MediaMTX and InferOps as native processes in
a root-based x86_64 Pod. In Triton mode it also starts an installed native model
server for an unused local endpoint. It does not invoke Docker Compose;
`COMPOSE_PROFILES` does not start services here.

The checkout is `/workspace/inferops`. Python environments, model weights,
database, recordings and logs persist under `/workspace`.

## Image for the observed R595 host

For the host reporting NVIDIA driver **595.91.07**, the deployment target is:

```text
nvcr.io/nvidia/tritonserver:26.04-py3
```

NVIDIA's [26.04 release notes](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2680/release-notes/rel-26-04.html)
list Triton 2.68.0, Ubuntu 24.04, Python 3.12 and CUDA 13.2.1. The [CUDA driver
table](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html)
lists R595 for CUDA 13.2. The GPU must satisfy the release's compute capability
requirement (7.5 or newer). GPU inference must still be verified on the Pod.

The Compose deployment retains its separate 26.09 pin. The R595 Pod's 26.09
entrypoint reported that driver compatibility was unavailable. Choose a compatible
image or another RunPod host when this happens; Python packages cannot upgrade
the host kernel driver. A PyTorch image's CUDA tag does not identify that driver.

## Environment and startup command

For the bundled project and the image above, configure:

```ini
APP_ENV=prod
INFEROPS_PROJECT_ROOT=./projects/driver_behaviour
INFEROPS_EXECUTION_MODE=triton
INFEROPS_API_PORT=8082
INFEROPS_TRITON_URL=127.0.0.1:8001
INFEROPS_TRITON_HTTP_URL=http://127.0.0.1:8000
INFEROPS_TRITON_CLIENT_VERSION=2.68.0
INFEROPS_TRITON_AUTOSTART=auto
INFEROPS_TRITON_STARTUP_TIMEOUT_SECONDS=60
```

Use this command after merging the native startup changes:

```bash
bash -lc 'if [ -x /start.sh ]; then /start.sh & fi; if [ -d /workspace/inferops/.git ]; then git -C /workspace/inferops pull --ff-only; else git clone https://github.com/OsamaM0/inferops.git /workspace/inferops; fi && exec bash /workspace/inferops/runpod-start.sh'
```

`/start.sh` is optional and belongs to the old RunPod template. NVIDIA's raw
image does not supply that template's SSH, Jupyter or code-server setup. Expose
**8082 as an HTTP port** to access InferOps at `/ui/`. The managed Triton
services bind to loopback and need no public port exposure in this layout.

## SSH, desktop VS Code and browser editor on the NVIDIA image

For development access on the raw NVIDIA image, use the optional access launcher.
It starts access before inference dependencies/model preparation, and keeps the
Pod running for repairs if InferOps exits. This deliberately changes failure
behavior: an inference failure is logged but does not restart the Pod while
access is enabled. Restart the Pod after repairing the app/configuration.

Add these Pod environment variables, keeping the inference settings above:

```ini
INFEROPS_SSH_ENABLED=true
PUBLIC_KEY=<contents of your local id_ed25519.pub, not the private key>
INFEROPS_CODE_SERVER_ENABLED=true
INFEROPS_CODE_SERVER_PASSWORD=<a new password of at least 12 characters>
```

SSH also accepts `SSH_PUBLIC_KEY` and preserves already injected authorized keys.
Only public-key SSH authentication is enabled; no SSH password is created.
The private key stays on your computer. SSH host keys persist under
`/workspace/inferops-data/secrets`, while editor settings/extensions and binaries
persist under `/workspace/inferops-data`. The pinned code-server release is
verified against its SHA-256 before extraction. Neither credential is logged.
`JUPYTER_PASSWORD` does not configure this editor; Jupyter is not installed.

In **Edit Pod / template**, expose **22 as TCP**, **8080 as HTTP**, and keep
**8082 as HTTP** for InferOps. Full SSH needs a Pod with a public IP. The
external SSH port is assigned by RunPod; it is usually different from 22.
Use this startup command instead of the previous one (do not also run `/start.sh`):

```bash
bash -lc 'if [ -d /workspace/inferops/.git ]; then git -C /workspace/inferops pull --ff-only; else git clone https://github.com/OsamaM0/inferops.git /workspace/inferops; fi && exec python3 /workspace/inferops/scripts/runpod_access.py'
```

After restart:

- Terminal: copy **Connect → SSH over exposed TCP**. Use the corresponding local
  private key with `-i`. A public key alone cannot authenticate your computer.
- Desktop VS Code: install **Remote - SSH**, connect using the same public IP,
  mapped external port and private key, then open `/workspace/inferops`.
- Browser editor: open **Connect → HTTP service on port 8080** (or
  `https://<pod-id>-8080.proxy.runpod.net`) and enter `INFEROPS_CODE_SERVER_PASSWORD`.
  The editor includes a terminal. Use RunPod's HTTPS proxy for this HTTP service.
- InferOps: open the HTTP service on **8082**, path `/ui/`.

Each access service can be disabled with its flag set to `false`; both default
to `false`, in which case the launcher directly runs the regular bootstrap.
An occupied access port fails clearly instead of replacing an existing service.
Access logs are `logs/sshd.log` and `logs/code-server.log` under the data directory.
The launcher stops only its own process groups on Pod shutdown. It does not
install an SSH gateway for RunPod's basic `ssh.runpod.io` connection method.

See RunPod's [full SSH setup](https://docs.runpod.io/pods/configuration/use-ssh)
and [VS Code guide](https://docs.runpod.io/pods/configuration/connect-to-ide).

## Python and persistent state

The bootstrap supports Python 3.10+ and uses `python3`; set `INFEROPS_PYTHON`
to override the executable. It installs venv support and uses an interpreter-specific
environment such as `/workspace/.venvs/inferops-py3.12`. Previous environments
are preserved, so changing from a Python 3.11 image does not reuse its environment.

It installs `.[server,cuda,triton,video,driver_face]`. The optional client version constraint
supports 2.68.x through 2.73.x. The dependency cache includes the package spec,
bootstrap, Python version and client pin.

PostgreSQL and MediaMTX use persistent generated credentials and loopback
endpoints. Old database/media endpoint overrides are replaced at startup.
If existing PostgreSQL data belongs to another major server version, startup
stops with a migration message. It preserves the data and does not attempt an
automatic major upgrade; use matching binaries or a planned PostgreSQL migration.

## Triton process lifecycle

After preparing model artifacts, startup resolves the effective project settings.
For a local `127.0.0.1`/`localhost` endpoint with no listener, it finds
`tritonserver` on PATH or at `/opt/tritonserver/bin/tritonserver`, starts it
with the selected repository and explicit loading, and waits for gRPC readiness
before starting the API.

The launcher supervises its API and native Triton process together. It stops
owned processes on shutdown or failure. An existing model server is reused and
never killed. Native server failures print the current attempt's log into Pod output;
the full log persists at `/workspace/inferops-data/logs/triton.log`.

`INFEROPS_TRITON_AUTOSTART=false` uses an independently managed server.
`true` requires an installed binary for an unused local endpoint.
The default `auto` starts a native server when available and otherwise checks
the configured server. Override the executable with `INFEROPS_TRITON_BINARY`.

The default repository belongs to the selected project. Set
`INFEROPS_TRITON_REPOSITORY` for a separately prepared repository. The bundled
project serves RF-DETR, SCRFD, 6DRepNet360, 2D106 and BlinkCNN through Triton from
`/workspace/inferops/projects/driver_behaviour/model_repository`.
Head direction is estimated by 6DRepNet360; 2D106 locates the eyes and BlinkCNN
classifies their closure. The `driver_face` extra supplies Pillow for preprocessing.
Project preparation validates the five prepared ONNX graphs and their checksums.
The head graph is exported once from the pinned official checkpoint in a separate
PyTorch export environment; PyTorch is not needed by the API or Triton.
See [Triton serving](triton-serving.md) for qualified batching repositories.
API model loading still validates model names and tensor contracts; readiness
alone is not full inference qualification.

For a remote server, set its reachable gRPC `host:port` and HTTP endpoint.
An HTTP-only proxy is not a gRPC endpoint. The bootstrap preserves proxy
exclusions and adds loopback exclusions. If a listener returns HTTP/1.x, check
which service owns that port and the proxy configuration. The launcher does not
terminate unrelated listeners. CPU and CUDA modes do not start or probe Triton.
