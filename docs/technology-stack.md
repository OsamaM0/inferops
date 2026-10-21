# Technology stack and defaults

The durable server uses PostgreSQL, persistent disk and MediaMTX by default. Its inference
runtime defaults to CPU; CUDA, TensorRT and Triton are selected explicitly.
The embedded Python library can still use memory and explicitly injected storage.

| Technology | Role | Usage |
| --- | --- | --- |
| Python, asyncio | Core | Engine, isolated sessions, pipelines and background work |
| Pydantic, pydantic-settings | Core | Data contracts, validation and settings |
| YAML/PyYAML, python-dotenv | Core | Project/pipeline declarations and environment configuration |
| NumPy, OpenCV | Core | Tensors, image processing, annotations and snapshots |
| Typer, Rich | Core | CLI and terminal output |
| structlog, orjson, filelock | Core | Logs, JSON and exclusive local resource ownership |
| FastAPI, Uvicorn, HTTPX, AnyIO, websockets, python-multipart | Server/SDK | HTTP, uploads, asynchronous networking and WebSockets |
| HTML, CSS, vanilla JavaScript | Dashboard | Sources, live players, recordings, results and evidence |
| ONNX, ONNX Runtime CPU | Default inference | Deployed model graphs and local CPU execution |
| PyAV, FFmpeg | Video processing support | Decode pixels for AI and encode annotated incident clips |
| PostgreSQL, SQLAlchemy, asyncpg | Default durable database | Sources, history, sessions, jobs, incidents, evidence metadata, inbox/outbox and leases |
| Persistent disk, FilesystemStorage | Default evidence store | Verified snapshots, annotated clips and manifests on the `engine-data` volume |
| MinIO/S3, boto3 | Optional evidence adapter | Shared object storage through the same `ObjectStore` port; enable the `storage` profile for local MinIO |
| Persistent recovery spool | Required for durable capture | Recover unfinished captures/uploads after a crash |
| MediaMTX | Default live-media service | Shared/on-demand RTSP paths, WebRTC/HLS, original recording, playback and metrics |
| In-process transport | Default messaging | Event delivery inside a combined process |
| prometheus-client | Core instrumentation | Counters, gauges and histograms |
| CUDA, cuDNN, ONNX Runtime GPU | Optional runtime | Local NVIDIA GPU inference |
| TensorRT | Optional runtime | ONNX Runtime TensorRT execution provider |
| Triton, tritonclient, gRPC/HTTP | Optional model service | Shared remote inference and qualified batching |
| Redpanda/Kafka, aiokafka | Optional broker | Durable external event transport and topic workers |
| Prometheus, Grafana | Optional monitoring | Metrics retention, queries and dashboards |
| OpenTelemetry/OTLP | Optional tracing | Export traces to a configured collector |
| NVIDIA management tools | Optional inspection | GPU utilization, memory and runtime diagnostics |
| ONNX tooling, Hugging Face Hub | Optional preparation | Validate/download declared model artifacts |
| RF-DETR, FaceMesh-compatible ONNX | Driver project | Driver objects, face landmarks and temporal safety signals |
| PyTorch, torchvision, RF-DETR toolkit | Optional export/research | CPU checkpoint conversion, validation and training |
| Docker, Docker Compose | Deployment | Run the application and selected services |
| pip, setuptools, wheel, build, Make | Development | Packaging, installation and common commands |
| pytest, pytest-asyncio, pytest-cov, Moto | Development | Tests, asynchronous checks, coverage and simulated S3 |
| Ruff, GitHub Actions | Development/CI | Linting, architecture contracts and integration checks |
| Playwright, Chromium | Optional verification | Browser and media-player scenarios |
| Jupyter/Colab, Roboflow, Supervision, gdown | Research | Notebooks, datasets, evaluation and downloads |
| Pandas, Matplotlib, Plotly, Pillow, tqdm | Research | Analysis, charts, image handling and progress |

## MediaMTX boundaries

MediaMTX remuxes/routes compressed streams, records them to disk and serves
recorded timespans. Those recordings complement incident evidence. The AI engine
still needs decoded pixels, and annotated evidence needs encoding. MediaMTX's
own [re-encoding](https://mediamtx.org/docs/features/remuxing-reencoding-compression)
and [snapshot](https://mediamtx.org/docs/features/extract-snapshots) examples use
external processing tools. PyAV/FFmpeg therefore remain processing dependencies.

MediaMTX is not the application's object store. Its
[recording documentation](https://mediamtx.org/docs/features/record) writes local
segments and uses external tools for remote uploads. Persistent disk holds verified
incident snapshots/clips/manifests; PostgreSQL holds their lifecycle and delivery
state; the persistent spool preserves unfinished captures. MinIO/S3 can replace
the completed-evidence disk adapter when shared object storage is needed. Native original-video
recordings have a separate, default 24-hour retention policy.

See [deployment](../deploy/README.md), [MediaMTX](media-router.md), and the
[PostgreSQL test setup](../README.md#postgresql-backed-tests).
