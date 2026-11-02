<div align="center">

# InferOps

**Multi-model video inference pipelines with incidents, evidence and a built-in dashboard.**
Ships with a driver-behaviour project. Runs on CPU, CUDA or NVIDIA Triton.

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Runtimes](https://img.shields.io/badge/runtime-CPU%20%7C%20CUDA%20%7C%20Triton-76B900?logo=nvidia&logoColor=white)
![Docker](https://img.shields.io/badge/docker-compose-2496ED?logo=docker&logoColor=white)

<!-- Maintainers: replace TRITON_TEMPLATE_ID, CUDA_TEMPLATE_ID and CPU_TEMPLATE_ID with the IDs printed by deploy/runpod/create-template.sh -->
[![Deploy on RunPod](https://img.shields.io/badge/Deploy%20on-RunPod-673AB7?style=for-the-badge)](https://console.runpod.io/deploy?template=TRITON_TEMPLATE_ID)

[Quick start](#quick-start) ·
[RunPod](#deploy-on-runpod) ·
[Docker](#run-with-docker) ·
[Configure](#configure) ·
[Python](#use-as-a-python-library) ·
[Docs](#documentation)

</div>

---

## What you get

- **Model pipelines.** Detection, face mesh, blendshapes and custom stages run as one ordered pipeline per source.
- **Cameras, videos and images.** Register RTSP cameras, server videos or uploads, with annotated live previews and history.
- **Incidents with evidence.** Unsafe behaviour is confirmed over time, then saved with video from before and after.
- **One dashboard** at `/ui/` for sources, pipelines, models, settings and GPU capacity. No frontend build.
- **Three runtimes**, chosen by one setting: ONNX Runtime on CPU or CUDA, or a separate Triton server.

PostgreSQL stores records, disk stores evidence, and [MediaMTX](docs/media-router.md) routes and records camera streams. MinIO/S3, Redpanda and Prometheus/Grafana are optional.

## Quick start

| Where | Best for | Run |
| --- | --- | --- |
| **RunPod** | A cloud GPU, nothing to install | [Deploy button](#deploy-on-runpod) |
| **Docker** | Your own Linux server or laptop | `./run.sh` |
| **Python** | Calling models from your own code | `pip install -e '.[cpu]'` |

---

## Deploy on RunPod

The Pod runs InferOps as plain processes (RunPod has no Docker): SSH, PostgreSQL, MediaMTX, Triton if chosen, and the API. Add your SSH key under **RunPod → Settings → SSH Public Keys** first.

| Image | Mode | Pod | Deploy | Command |
| --- | --- | --- | --- | --- |
| `nvcr.io/nvidia/tritonserver:26.04-py3` | `triton` | GPU, CUDA 13.0+ | [![Deploy](https://img.shields.io/badge/Deploy-673AB7)](https://console.runpod.io/deploy?template=TRITON_TEMPLATE_ID) | [Start command](docs/runpod.md#triton) |
| `nvidia/cuda:12.8.1-runtime-ubuntu24.04` | `cuda` | GPU, CUDA 12.8+ | [![Deploy](https://img.shields.io/badge/Deploy-673AB7)](https://console.runpod.io/deploy?template=CUDA_TEMPLATE_ID) | [Start command](docs/runpod.md#cuda) |
| `ubuntu:24.04` | `cpu` | CPU | [![Deploy](https://img.shields.io/badge/Deploy-673AB7)](https://console.runpod.io/deploy?template=CPU_TEMPLATE_ID) | [Start command](docs/runpod.md#cpu) |

1. Click **Deploy** for your image and pick a GPU.
2. Click **Deploy On-Demand**. The first boot takes several minutes.
3. Open `https://<pod-id>-8082.proxy.runpod.net/ui/`.

Setting the Pod up by hand needs only an image, a start command, a 50 GB volume at `/workspace`, HTTP port `8082` and TCP port `22`. The [RunPod guide](docs/runpod.md) has each start command, all settings, template creation, operations and troubleshooting.

---

## Run with Docker

Needs Docker Engine with **Compose 2.20.3+**. GPU modes also need the NVIDIA driver and Container Toolkit (`make setup-container-runtime`). No host Python needed.

```bash
./run.sh            # creates .env if missing, builds, starts, waits until ready
```

Open **http://localhost:8080/ui/** (or the port it prints). The first start downloads images and model weights.

| Command | What it does |
| --- | --- |
| `./run.sh --mode cuda --port 8082` | Override runtime or port for this run |
| `./run.sh --no-build` | Start faster with existing images |
| `./run.sh status` · `logs` · `stop` | Show status, follow logs, stop and keep data |
| `./run.sh check` · `repair` | Check Docker and ports; recreate the API container |

| Interface | Address | Login |
| --- | --- | --- |
| Workspace | [localhost:8080/ui/](http://localhost:8080/ui/) | None |
| API docs | [localhost:8080/docs](http://localhost:8080/docs) | None |
| Pages, ports and routes | [localhost:8080/ui/guide](http://localhost:8080/ui/guide) | None |
| MinIO console (`storage`) | [localhost:9001](http://localhost:9001/) | `minio` / `minio12345` |
| Grafana (`monitoring`) | [localhost:3000](http://localhost:3000/d/inferops-overview/inferops-overview) | Viewer; `admin` / `admin` to edit |
| Prometheus (`monitoring`) | [localhost:9090](http://localhost:9090/graph) | None |

> These are development credentials with no login on the workspace. Before exposing a server, read [server access](docs/hosting.md#server-access-and-deployment-checks).

<details>
<summary><b>Manual Compose commands and rebuilds</b></summary>

```bash
./run.sh configure                                              # generate Docker inputs from the YAML and .env
docker compose up -d --build --wait --wait-timeout 600          # start without the launcher's checks
docker compose logs prepare api

# After changing a model source or tensor shapes:
docker compose run --rm --no-deps prepare inferops project prepare --force
docker compose up -d --build --force-recreate api               # Triton modes: recreate "triton api"
```

Named volumes keep databases, objects, broker state and caches across `docker compose down`. `down -v` deletes them. Project weights stay in `projects/<name>/model_repository/`. More: [startup troubleshooting](deploy/README.md#startup-troubleshooting).

</details>

---

## Configure

Workspace settings live in [`configs/inferops.yaml`](configs/inferops.yaml), and the dashboard editors save to it. Secrets and host overrides go in `.env`.

```yaml
runtime:
  project: driver_behaviour   # folder under projects/
  mode: cuda                  # cpu | cuda | triton | auto
  api_port: 8080
  services: []                # optional: storage, broker, monitoring
```

| Mode | Runs models with | Needs |
| --- | --- | --- |
| `cpu` | ONNX Runtime CPU | Nothing extra |
| `cuda` | ONNX Runtime GPU | NVIDIA GPU and driver |
| `triton` | A separate Triton server | NVIDIA GPU, recent driver ([Triton serving](docs/triton-serving.md)) |
| `auto` | CPU (on RunPod: CUDA if the Pod has a GPU) | |

Environment variables such as `INFEROPS_EXECUTION_MODE` override the YAML. Startup validates the project and downloads its weights; missing or invalid weights stop it. For MinIO/S3 set `storage.kind: s3` and `runtime.services: [storage]` ([storage options](deploy/README.md#what-starts-and-where)). Every key and its precedence: [central configuration](docs/configuration.md).

---

## Use as a Python library

```bash
pip install -e '.[cpu]'                     # or .[cuda], .[triton], .[tensorrt]: one ONNX Runtime per environment
pip install -e projects/driver_behaviour    # optional driver-behaviour adapter
```

```python
from inferops import Engine

async def run_project():
    async with Engine.from_project("projects/driver_behaviour", runtime="cpu") as engine:
        summary = await engine.run("driver_behaviour", "clip.mp4")
        print(summary)
```

Results stay in memory with evidence capture off. The [Python guide](docs/python-library.md) shows durable storage and sessions, and [adapter authoring](docs/adapters.md) covers your own project packages. To work on the code (`make setup`, tests, new projects), see [development](docs/development.md).

---

## Documentation

| I want to… | Read |
| --- | --- |
| Deploy on RunPod | [RunPod guide](docs/runpod.md) |
| Operate a server, size it, secure it | [Deployment guide](deploy/README.md) · [Hosting reference](docs/hosting.md) |
| Change settings | [Central configuration](docs/configuration.md) |
| Use the dashboard | [Unified dashboard](docs/unified-dashboard.md) · [Source control and capacity](docs/source-control.md) |
| Serve models with Triton | [Triton serving](docs/triton-serving.md) |
| Route, record and play cameras | [MediaMTX routing](docs/media-router.md) · [Interfaces and ports](docs/interfaces.md) |
| Call the API | [API reference](docs/api.md), or `/docs` on a running server |
| Build a project or adapter | [Project configuration](docs/project-blocks.md) · [Adapter authoring](docs/adapters.md) · [Examples](projects/README.md) |
| Develop and test | [Development](docs/development.md) |
| Convert RF-DETR checkpoints | [Checkpoint conversion](docs/checkpoint-conversion.md) |
| Run fixed camera shards | [Static workers](docs/static-workers.md) |
| Understand the design | [Architecture](docs/architecture.md) · [Lifecycle](docs/lifecycle.md) · [Messaging](docs/messaging.md) · [Structure](docs/structure.md) |
| Ship releases | [CI/CD and Docker Hub](deploy/README.md#cicd-and-docker-hub) · [Upgrade and rollback](docs/lifecycle.md#upgrade-and-rollback) |

Everything else is in the [documentation index](docs/README.md).
