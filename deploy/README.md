# Deployment

Run `docker compose up -d --build` from the Git repository root, which contains
`README.md`, `pyproject.toml` and `compose.yaml`. `.env` selects `projects/<name>`
and the runtime overlay.

- `docker/`: CPU/CUDA API and DeepStream images, including the packaged API UI.
- `compose/`: runtime overrides for the single API service.
- `observability/`: Grafana provisioning and container/native Prometheus configs.

[The project entry point](../compose.yaml) includes [the base stack](compose.yaml)
and one runtime overlay using Compose include (Docker Compose 2.20.3+). Build
context and bind mounts resolve from the project root. Grafana and Prometheus
are enabled by default; Triton is enabled by triton/deepstream profiles.

Named volumes retain storage across recreation. Project code, configuration
and weights are mounted at `/project`; engine/UI changes require `--build`.
Read the [server requirements](../README.md#hosting-server-requirements),
[startup guide](../README.md), [interfaces and ports](../docs/interfaces.md)
and [API reference](../docs/api.md).
