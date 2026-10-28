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

## Python and persistent state

The bootstrap supports Python 3.10+ and uses `python3`; set `INFEROPS_PYTHON`
to override the executable. It installs venv support and uses an interpreter-specific
environment such as `/workspace/.venvs/inferops-py3.12`. Previous environments
are preserved, so changing from a Python 3.11 image does not reuse its environment.

It installs `.[server,cuda,triton,video]`. The optional client version constraint
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
never killed. Native server failures print the last log lines into Pod output;
the full log persists at `/workspace/inferops-data/logs/triton.log`.

`INFEROPS_TRITON_AUTOSTART=false` uses an independently managed server.
`true` requires an installed binary for an unused local endpoint.
The default `auto` starts a native server when available and otherwise checks
the configured server. Override the executable with `INFEROPS_TRITON_BINARY`.

The default repository belongs to the selected project. Set
`INFEROPS_TRITON_REPOSITORY` for a separately prepared repository. The bundled
project loads `face_mesh_onnx` and `rfdetr_driver_behaviour_onnx` from
`/workspace/inferops/projects/driver_behaviour/model_repository`.
See [Triton serving](triton-serving.md) for qualified batching repositories.
API model loading still validates model names and tensor contracts; readiness
alone is not full inference qualification.

For a remote server, set its reachable gRPC `host:port` and HTTP endpoint.
An HTTP-only proxy is not a gRPC endpoint. The bootstrap preserves proxy
exclusions and adds loopback exclusions. If a listener returns HTTP/1.x, check
which service owns that port and the proxy configuration. The launcher does not
terminate unrelated listeners. CPU and CUDA modes do not start or probe Triton.
