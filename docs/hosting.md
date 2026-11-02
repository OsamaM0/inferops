# Hosting and operations reference

Server sizing, software, GPU compatibility and access rules. For steps to start the server, see the [README](../README.md) and the [deployment guide](../deploy/README.md).

The following are **planning estimates**, not benchmarked minimums or guarantees
of real-time performance. They assume one Linux server hosts the API, inference,
PostgreSQL, MediaMTX and optional MinIO, Redpanda, Prometheus and Grafana services. The default driver
project runs RF-DETR at 576 × 576 and FaceMesh at 256 × 256. Size other projects
for their own models, input resolution, sampled FPS and concurrent jobs.

## Hardware sizing

| Deployment | CPU | System RAM | GPU | Free SSD space at setup |
| --- | --- | --- | --- | --- |
| CPU evaluation, short clips or occasional images | 4 vCPU | 8 GB | None | 50 GB |
| CPU hosting starting point | 8 vCPU | 16–32 GB | None | 100 GB |
| CUDA hosting starting point | 8 vCPU | 32 GB | Compatible NVIDIA GPU; 8 GB VRAM starting point, 16 GB preferred | 100 GB |
| Triton hosting starting point | 8–16 vCPU | 32–64 GB | Compatible NVIDIA GPU; 16 GB VRAM or more | 150 GB |

CPU mode is useful for evaluation and workloads that tolerate slower inference.
For sustained video processing, start by evaluating CUDA; choose Triton for
separate model serving. The shared engine uses
incremental PyAV decoding in every runtime mode. The Triton reference configuration uses one instance per model, shared GPU rate limiting and FP32 CUDA execution. A separate generated repository enables dynamic batching only after batch qualification. The original RF-DETR ONNX remains batch-one; the supplied checkpoint was genuinely re-exported and verified with dynamic batches on the local RTX 4060. See [Triton serving](triton-serving.md). GPU memory and end-to-end camera capacity must be measured.

**The shared engine decodes video incrementally.** Its memory depends on bounded
processing/evidence buffers rather than the full clip duration. Benchmark the
actual models, concurrency and evidence workload when sizing the server.

The SSD estimates cover an initial allowance for images, build cache, weights
and service data. Add capacity for uploaded media, evidence retention, database
growth and backups. Keep Docker's data directory and the writable project/model
repository on persistent storage. Named volumes survive `docker compose down`;
`docker compose down -v` removes them. Back up volumes and project configuration
and weights separately, and verify restoration.

## Operating system and software

- **Host:** use a Linux x86_64 server as the deployment baseline. Ubuntu 22.04
  LTS is a practical common starting point for the pinned GPU images; Ubuntu
  24.04 LTS is another option to evaluate for CPU/CUDA. ARM/Jetson deployments
  require separate image, wheel and runtime compatibility validation.
- **Containers:** install Docker Engine and **Docker Compose 2.20.3 or newer**.
  The root Compose file uses
  [Compose `include`](https://docs.docker.com/compose/how-tos/multiple-compose-files/include/).
  The deployment account needs permission to run Docker and write the selected
  project directory so model preparation can save weights.
- **Python:** the package requires Python **3.10+**; use **3.11** for a native
  server environment to match the CPU/CUDA API image. Docker supplies Python
  and OS dependencies, including FFmpeg; host Python is only needed for native
  setup or the Python launcher. Native setup also needs `venv`, `pip`, `make`
  for Make targets, and system libraries required by the selected video runtime.
- **Dependencies:** [pyproject.toml](../pyproject.toml) is the source of truth.
  For a native CPU API/video server, install `pip install -e '.[server,cpu,video]'`;
  use `.[server,cuda,video]` for CUDA. Choose one ONNX Runtime wheel per
  environment. Docker installs the server, runtime and video dependencies.
  The UI needs no Node.js or frontend build.
- **Network:** allow outbound HTTPS during builds and model preparation to
  package indexes, image registries and configured artifact sources. The driver
  project's weights are downloaded from Google Drive and Wasabi; model files
  are ignored by Git. Provide `HF_TOKEN` when a selected project uses gated
  Hugging Face artifacts. Allow access to configured camera/source endpoints.

## GPU runtime compatibility

GPU modes need a working host NVIDIA driver, GPU access for the container, and
the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The helper `make setup-container-runtime` configures the toolkit on Ubuntu/Debian
and restarts Docker; it does **not** install a GPU driver.

| Mode | Runtime supplied by this checkout | Compatibility to check on the server |
| --- | --- | --- |
| `cpu` / `auto` | ONNX Runtime CPU | No NVIDIA software required; `auto` selects CPU |
| `cuda` | `onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27` | CUDA 12 and cuDNN 9 library compatibility, supported GPU and matching driver; check the resolved wheel against the [ONNX Runtime requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) |
| `triton` | `nvcr.io/nvidia/tritonserver:26.09-py3` | Triton 2.73.0, CUDA 13.4.1, ONNX Runtime 1.28.0; see [version qualification and driver notes](triton-serving.md) |

The CUDA extra supplies user-space CUDA/cuDNN libraries through pip; GPU images
supply their runtime libraries. A host CUDA compiler toolkit is not required
for these container workflows. Driver versions above describe the pinned
releases, not a universal driver recommendation: choose a maintained driver
compatible with the GPU and every selected runtime. Newer GPU generations may
need newer images. Validate compatibility before provisioning the server.
`tensorrt` is a Python library runtime option; it is not a shipped Compose mode.

## Server access and deployment checks

The supplied Compose stack publishes ports on all host interfaces and uses
development credentials. Before exposing it to remote users, change MinIO and
Grafana and PostgreSQL credentials in `.env`, and
restrict access to internal services. The workspace/API has no built-in login;
put it behind an authenticated HTTPS reverse proxy or private access gateway
with WebSocket support. Grafana enables anonymous viewer access by default.

Expose the application through the gateway (normally HTTPS 443). Keep PostgreSQL
5432, Kafka 19092 and Triton 8000–8002 private. Restrict MinIO
9000/9001, Grafana 3000 and Prometheus 9090 to trusted users/networks. Configure
the browser-facing `INFEROPS_*_PUBLIC_URL` values for remote service links and
evidence access; see [interfaces and ports](interfaces.md) for all settings.
Docker-published ports require Docker-aware firewall or upstream network rules;
see [Docker's firewall guidance](https://docs.docker.com/engine/network/packet-filtering-firewalls/).

From this directory, after selecting the project/runtime in `configs/inferops.yaml`:

```bash
docker version
docker compose version
./run.sh configure
docker compose config --quiet
# GPU modes only: check the host driver, then verify container GPU access.
# nvidia-smi
# docker run --rm --gpus all ubuntu:22.04 nvidia-smi
docker compose up -d --build
docker compose ps
docker compose logs prepare api
curl --fail http://localhost:8080/healthz
curl --fail http://localhost:8080/readyz
```

Use your configured API port if it differs from 8080. Startup prepares and
validates real weights; missing artifacts or incompatible runtimes block it.
Health/readiness checks verify startup, not capacity. Run representative images
and videos at the intended concurrency, confirm GPU execution in accelerated
modes, and monitor CPU, RAM, VRAM, inference latency and disk growth before
settling on server sizing. The [verification scenarios](../tests/scenarios/README.md)
provide examples for exercising the running application.

## Notes on earlier versions

MinIO uses the optional `storage` profile; Grafana and Prometheus use `monitoring`.
Persistent disk is the default evidence store. The workspace opens the enabled services’ own
interfaces in new tabs. `/frontend/` remains a compatibility address for
`/ui/`; the separate frontend directory and hand-built MinIO evidence page
have been removed. HTML and assets are served by FastAPI and included in
installed Python packages. No frontend server, npm setup or UI build is required.

When enabled, the full MinIO admin console requires an older official image: newer upstream
releases removed that console. The version choice, optional storage setup,
remote access and troubleshooting are covered in the
[interfaces and ports guide](interfaces.md). This is a development stack;
its example credentials and public MinIO metrics are configured for local use.

Previous benchmark reports describe their recorded configuration. They do not
qualify the new PostgreSQL/persistent-disk/MediaMTX defaults. The workspace reset discards
retired local runtime data. Startup uses fresh PostgreSQL and evidence-disk storage and has
no SQLite import or migration step.
