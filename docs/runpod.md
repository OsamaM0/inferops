# RunPod native startup

`runpod-start.sh` runs PostgreSQL, MediaMTX and the InferOps API as native
processes in a root-based x86_64 CUDA Pod with Python 3.11. The checkout lives
at `/workspace/inferops`; Python packages, weights, database, recordings and
logs persist under `/workspace`. It does not run Docker Compose, so
`COMPOSE_PROFILES` does not start services in this deployment.

Use the existing clone/pull startup command:

```bash
bash -lc '/start.sh & if [ -d /workspace/inferops/.git ]; then git -C /workspace/inferops pull --ff-only; else git clone https://github.com/OsamaM0/inferops.git /workspace/inferops; fi && exec bash /workspace/inferops/runpod-start.sh'
```

`/start.sh` belongs to the RunPod template; this command assumes it exists.
Select the project, mode and API port in `configs/inferops.yaml`, or override
them with Pod environment variables:

```ini
INFEROPS_PROJECT_ROOT=./projects/driver_behaviour
INFEROPS_EXECUTION_MODE=triton
INFEROPS_API_PORT=8082
```

Expose the API's selected HTTP port through RunPod. The bootstrap configures
its own persistent loopback PostgreSQL and MediaMTX services and credentials.
It replaces old database/media endpoint environment values at startup.

## Triton requires a model server

The bootstrap installs `.[server,cuda,triton,video]` so CPU, CUDA and Triton
client modes work in the same persistent virtual environment. Its dependency
fingerprint includes `pyproject.toml` and `runpod-start.sh`; upgrading from an
older CUDA-only bootstrap automatically reinstalls the required extras, even
when `pyproject.toml` is unchanged.

The Triton Python client sends inference requests to **Triton Inference
Server**. Installing the client or selecting `triton` does not install or
launch the model server. Use a separately managed compatible server in the
same Pod, or a reachable external server. Configure the native API endpoint:

```ini
# Same Pod; use the external server's reachable address for a separate host.
INFEROPS_TRITON_URL=127.0.0.1:8001
INFEROPS_TRITON_HTTP_URL=http://127.0.0.1:8000
INFEROPS_TRITON_STARTUP_TIMEOUT_SECONDS=60
```

`INFEROPS_TRITON_URL` is a gRPC `host:port`, without an `http://` prefix.
It must be reachable from the InferOps process; an HTTP-only RunPod proxy is
not a substitute for the gRPC endpoint. HTTP status/metrics configuration
does not change the inference endpoint.

Prepare and load the same project's model repository on the Triton host.
For the bundled driver project, the server must load `face_mesh_onnx` and
`rfdetr_driver_behaviour_onnx`. The bootstrap downloads the original weights
into `/workspace/inferops/projects/driver_behaviour/model_repository`; an
external server needs its own copy or shared access. A generated serving
repository may be selected separately; see [Triton serving](triton-serving.md).

The repository pins Triton Server `nvcr.io/nvidia/tritonserver:26.09-py3`
and client `>=2.73,<2.74`. Choose a server image compatible with the Pod's
GPU and host driver; the CUDA Python packages installed for local inference
do not supply Triton Server. Changing the Pod image also requires preserving
the bootstrap's Python 3.11 requirement. See the [release and driver
notes](triton-serving.md) before changing images.

After model preparation, startup checks the effective configured gRPC
endpoint before launching the API. It retries until the startup timeout,
then exits with the endpoint and setup guidance if the server is unreachable
or not ready. CPU and CUDA modes skip this check. Readiness confirms server
availability; the API's model loading still validates the required model
names and tensor contracts.
