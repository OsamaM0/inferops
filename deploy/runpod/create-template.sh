#!/usr/bin/env bash
# Create the RunPod Pod template behind a "Deploy on RunPod" button (run once per image).
# Usage: RUNPOD_API_KEY=... deploy/runpod/create-template.sh [triton|cuda|cpu]
#        DRY_RUN=1 deploy/runpod/create-template.sh triton   # print the request only
# Options: RUNPOD_IMAGE=<image> (use another image), RUNPOD_TEMPLATE_NAME=<name>,
#          RUNPOD_TEMPLATE_PUBLIC=true (list it for other RunPod users).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
mode=${1:-triton}
case "$mode" in
    triton) image=nvcr.io/nvidia/tritonserver:26.04-py3; category=NVIDIA ;;
    cuda) image=nvidia/cuda:12.8.1-runtime-ubuntu24.04; category=NVIDIA ;;
    cpu) image=ubuntu:24.04; category=CPU ;;
    *) fail "Unknown mode: $mode (use triton, cuda or cpu)" ;;
esac
image=${RUNPOD_IMAGE:-$image}

# RunPod injects PUBLIC_KEY from Settings -> SSH Public Keys, so the template carries no key.
payload=$(python3 - "$mode" "$image" "$category" <<'PY'
import json, os, shlex, sys
from pathlib import Path
mode, image, category = sys.argv[1:]
print(json.dumps({
    "name": os.environ.get("RUNPOD_TEMPLATE_NAME", f"InferOps ({mode})"),
    "imageName": image,
    "category": category,
    "containerDiskInGb": 40,
    "volumeInGb": 50,
    "volumeMountPath": "/workspace",
    "ports": ["8082/http", "22/tcp"],
    # The same one-liner operators paste into the console, split into argv.
    "dockerStartCmd": shlex.split(Path(f"start-{mode}.txt").read_text()),
    "env": {
        "INFEROPS_EXECUTION_MODE": mode,
        "INFEROPS_API_PORT": "8082",
        "INFEROPS_PROJECT_ROOT": "/workspace/inferops/projects/driver_behaviour",
        "INFEROPS_STORAGE_KIND": "filesystem",
        "INFEROPS_MEDIA_ROUTER": "mediamtx",
    },
    "isPublic": os.environ.get("RUNPOD_TEMPLATE_PUBLIC", "false") == "true",
    "isServerless": False,
    "readme": "InferOps driver-behaviour workspace. Open the HTTP 8082 link and add /ui/. "
              "Guide: https://github.com/OsamaM0/inferops/blob/main/docs/runpod.md",
}, indent=2))
PY
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%s\n' "$payload"; exit 0; fi
: "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY (RunPod console -> Settings -> API Keys)}"

response=$(curl -fsS -X POST https://rest.runpod.io/v1/templates \
    -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
    --data-binary @- <<< "$payload") || fail 'RunPod rejected the request (a template with this name may already exist; set RUNPOD_TEMPLATE_NAME).'
id=$(python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])' <<< "$response")
printf 'Template created (%s): %s\nDeploy link: https://console.runpod.io/deploy?template=%s\n' "$mode" "$id" "$id"
printf 'Put this ID in README.md in place of %s_TEMPLATE_ID.\n' "$(tr a-z A-Z <<< "$mode")"
