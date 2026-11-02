# Deploy on RunPod

RunPod Pods can't run Docker, so InferOps runs there as plain processes. On boot the Pod starts SSH, clones this repository into `/workspace/inferops` (first boot only), then runs [`runpod-start.sh`](../runpod-start.sh). That script installs what is missing, starts PostgreSQL, MediaMTX, Triton (triton mode only) and the API, and restarts them if they crash.

**Before you start:** add your SSH key in **RunPod → Settings → SSH Public Keys**. RunPod passes it to the Pod as `PUBLIC_KEY`, so it never goes in a template.

- [1. Pick an image](#1-pick-an-image)
- [2. Start commands](#2-start-commands)
- [3. Pod settings](#3-pod-settings)
- [4. Deploy](#4-deploy)
- [5. After deployment](#5-after-deployment)
- [6. Troubleshooting](#6-troubleshooting)

## 1. Pick an image

| Image | Mode | Pod type | Choose it for |
| --- | --- | --- | --- |
| `nvcr.io/nvidia/tritonserver:26.04-py3` | `triton` | GPU, **CUDA 13.0+** | Models served by Triton. The image includes `tritonserver`, and it also runs `cuda` mode. |
| `nvidia/cuda:12.8.1-runtime-ubuntu24.04` | `cuda` | GPU, CUDA 12.8+ | Models run in the API with ONNX Runtime GPU, no Triton. Runs on hosts that can't take the Triton image. |
| `ubuntu:24.04` | `cpu` | CPU | Trying it out, or short clips. |

The script needs a root Debian/Ubuntu x86_64 image, so other images of that kind should work for `cpu` and `cuda`, but only the three above have been written for. To try another, set `RUNPOD_IMAGE=...` for [`create-template.sh`](#create-the-template). Keep one image for the life of a volume (see [troubleshooting](#6-troubleshooting)). Set the **CUDA Versions** filter when you pick a GPU.

## 2. Start commands

Paste the one for your image into **Container Start Command**, as a single line. Each command installs `curl`, downloads [`start-command.sh`](../deploy/runpod/start-command.sh), sets the default mode and runs it. That script starts SSH, clones the repository and keeps the app running. It is downloaded on every boot, so a fix pushed to `main` reaches every Pod without editing the template.

### Triton

Image: `nvcr.io/nvidia/tritonserver:26.04-py3`

```bash
bash -c 'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates && curl -fsSL https://raw.githubusercontent.com/OsamaM0/inferops/main/deploy/runpod/start-command.sh -o /opt/inferops-boot.sh && export INFEROPS_EXECUTION_MODE=${INFEROPS_EXECUTION_MODE:-triton} && exec bash /opt/inferops-boot.sh'
```

### CUDA

Image: `nvidia/cuda:12.8.1-runtime-ubuntu24.04`

```bash
bash -c 'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates && curl -fsSL https://raw.githubusercontent.com/OsamaM0/inferops/main/deploy/runpod/start-command.sh -o /opt/inferops-boot.sh && export INFEROPS_EXECUTION_MODE=${INFEROPS_EXECUTION_MODE:-cuda} && exec bash /opt/inferops-boot.sh'
```

### CPU

Image: `ubuntu:24.04`

```bash
bash -c 'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates && curl -fsSL https://raw.githubusercontent.com/OsamaM0/inferops/main/deploy/runpod/start-command.sh -o /opt/inferops-boot.sh && export INFEROPS_EXECUTION_MODE=${INFEROPS_EXECUTION_MODE:-cpu} && exec bash /opt/inferops-boot.sh'
```

The three differ only in the default mode. A Pod environment variable named `INFEROPS_EXECUTION_MODE` overrides it. The same lines are kept in [`deploy/runpod/`](../deploy/runpod/) as `start-triton.txt`, `start-cuda.txt` and `start-cpu.txt`, which the template script sends.

## 3. Pod settings

| Setting | Value |
| --- | --- |
| Container disk | 40 GB (system packages only) |
| Volume disk | 50 GB, mount path `/workspace` (code, weights, Python environments, database, recordings) |
| Expose HTTP ports | `8082` |
| Expose TCP ports | `22` |

Only 8082 needs to be public. Live video is relayed through the API as HLS, and MediaMTX, PostgreSQL and Triton stay inside the Pod. WebRTC is unavailable because RunPod doesn't expose UDP.

**Environment variables:**

| Variable | Value | Purpose |
| --- | --- | --- |
| `INFEROPS_EXECUTION_MODE` | `triton`, `cuda`, `cpu` or `auto` | Runtime. Optional: the start command sets a default. |
| `INFEROPS_API_PORT` | `8082` | Must match the exposed HTTP port. |
| `INFEROPS_PROJECT_ROOT` | `/workspace/inferops/projects/driver_behaviour` | Project to run. |
| `INFEROPS_STORAGE_KIND` | `filesystem` | Evidence on the volume. |
| `INFEROPS_MEDIA_ROUTER` | `mediamtx` | Camera routing. |
| `HF_TOKEN` | *(optional)* | Hugging Face downloads that need a login. |
| `INFEROPS_TRITON_REPOSITORY` | *(optional)* | Another Triton model folder, e.g. `/workspace/inferops/.run/triton/models` from `scripts/prepare_triton.py`. |
| `INFEROPS_REPO_URL`, `INFEROPS_GIT_REF` | *(optional)* | Clone a fork or branch on first boot. Default: this repository, `main`. |

Don't set `INFEROPS_DATABASE_URL` or `INFEROPS_MEDIAMTX_*`: the script generates them and overrides your values. `COMPOSE_PROFILES` is for Docker only.

## 4. Deploy

### Deploy button

Click **Deploy** for your image in the [README](../README.md#deploy-on-runpod), pick a GPU (or a CPU Pod for `cpu`), and click **Deploy On-Demand**. The template carries everything from sections 1 to 3; under **Edit Template** you can change the mode or volume size first.

### Create the template

You need this once per image, to make the button. Create a RunPod API key (**Settings → API Keys**, read/write), and push your changes to GitHub first, because a new Pod clones from there.

```bash
export RUNPOD_API_KEY=...
deploy/runpod/create-template.sh triton      # or: cuda, cpu
```

It prints the template ID. Put it in the README in place of `TRITON_TEMPLATE_ID` (or `CUDA_…`, `CPU_…`).

| Option | Effect |
| --- | --- |
| `DRY_RUN=1` | Print the request without sending it |
| `RUNPOD_IMAGE=...` | Use another image |
| `RUNPOD_TEMPLATE_NAME=...` | Set the name (default `InferOps (<mode>)`) |
| `RUNPOD_TEMPLATE_PUBLIC=true` | List the template for other RunPod users |

Prefer the console? **Templates → New Template**, enter sections 1 to 3, then use **Copy link** in the template menu for the deploy link.

### Manual deployment

1. **Storage.** A *Pod volume* is created with the Pod and deleted on Terminate. A *network volume* (**Storage → New Network Volume**, 50 GB) survives Terminate and attaches to a new Pod in the same data center.
2. **Pods → Deploy.** Pick a GPU and apply the CUDA filter from section 1. Attach the network volume if you made one.
3. **Edit Template.** Set the image from section 1, the start command from section 2, and the disks, ports and variables from section 3.
4. **Set Overrides → Deploy On-Demand.**

## 5. After deployment

The first boot takes several minutes: system packages, Python dependencies, MediaMTX and model weights are installed onto `/workspace`. Later boots skip what is already there.

Open `https://<pod-id>-8082.proxy.runpod.net/ui/` (**Connect → HTTP 8082**, plus `/ui/`). Over SSH (**Connect** shows the host and port):

```bash
ssh root@<public-ip> -p <mapped-port> -i ~/.ssh/<your-key>
tail -f /workspace/inferops-data/logs/runpod-start.log   # progress; ready at "Starting InferOps API"
curl -fsS localhost:8082/readyz                          # API check
```

| Task | How |
| --- | --- |
| Restart the app | `pkill -TERM -f '^bash /workspace/inferops/runpod-start.sh'`. It returns within 15 s. Keep the `^`, or you also kill the restart loop. |
| Update the code | `cd /workspace/inferops && git pull`, then restart the app. New dependencies install automatically. |
| Switch mode | Change `INFEROPS_EXECUTION_MODE`, then restart the Pod. Each mode has its own environment in `/workspace/.venvs/`. |
| Stop paying for the GPU | **Stop** the Pod. Data on `/workspace` stays; a Pod volume is deleted only on **Terminate**. |

Everything is under `/workspace`:

| Path | Contents |
| --- | --- |
| `inferops/` | Code and model weights |
| `inferops-data/logs/` | `runpod-start.log` (bootstrap and API), `triton.log`, `postgres.log`, `mediamtx.log` |
| `inferops-data/postgres/` | Database |
| `inferops-data/recordings/` | Camera recordings |
| `inferops-data/app/` | Sources and evidence |
| `inferops-data/secrets/` | Generated database and MediaMTX passwords |
| `.venvs/inferops-<mode>/` | Python environments |

## 6. Troubleshooting

| Problem | Cause and fix |
| --- | --- |
| HTTP 8082 link shows *Bad Gateway* | The app is still installing or starting. Follow `runpod-start.log`. |
| `Runtime triton needs tritonserver` | Not a Triton image. Use the Triton image, or set `INFEROPS_EXECUTION_MODE=cuda`. |
| `needs a GPU Pod` | A GPU mode on a CPU Pod. Use `cpu`, or a GPU Pod. |
| Triton exits at start; `triton.log` mentions the driver or CUDA | The host is too old for the image. Redeploy on a host with CUDA 13.0 or newer, or use the CUDA image. |
| `Database ... needs PostgreSQL <n>` | The volume was created on an image from another Ubuntu/Debian release. Use that image again, or move `/workspace/inferops-data/postgres` aside to start a new database. |
| `Another runpod-start.sh is already running` | You ran the script by hand while the boot loop was running. Use the restart command above. |
| Can't SSH in | Check your key under **Settings → SSH Public Keys**, then restart the Pod. |
| `git clone failed` in the log | The repository isn't reachable. Check `INFEROPS_REPO_URL`, or clone into `/workspace/inferops` over SSH. |
