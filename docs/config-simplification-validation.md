# Configuration and Docker validation — 10 October 2026

The project now starts through root Compose, using `project.yaml`, logical model
references and model-owned artifact sources. Existing thresholds, tensors,
tracking, rules, alert actions and evidence settings were retained.

Initial CPU validation used an isolated Compose project (`inferops-config-verify`)
on port 18080. The current deployment now runs the CUDA Docker API on port 8080,
after stopping the registered native API that occupied that port. The original
database, broker, cache and object-store data were copied into persistent named
volumes; the original storage volumes were retained.

| Check | Result |
| --- | --- |
| Engine suite | 159 tests passed |
| Driver suite, including real video inference | 9 tests passed |
| CPU Docker video | 187 frames, 9 alerts, 195 evidence objects, including an H.264 clip |
| Dashboard and API | HTTP 200; both real ONNX models loaded on CPU |
| Persistence after Compose down/up | Both weight checksums, 9 alerts, 10 fusion events and all 195 evidence objects retained |
| Empty project, same Docker image | Healthy dashboard, zero models and zero pipelines |
| Offline preparation | Both real models validated inside a container with networking disabled |
| Actual Google Drive download | RF-DETR downloaded and validated; checksum matched existing trained weights |
| Actual archive download | FaceMesh selected archive member downloaded and validated; checksum matched existing trained weights |
| Native CUDA | Both real models executed with CUDAExecutionProvider, without device downgrade |
| CUDA Docker video | 187 frames, 9 alerts, 195 evidence objects, including 7 JPEG snapshots and 1 H.264 clip; 25.101 seconds |
| Browser dashboard | 2 loaded logical models, 2 configured; API and MinIO 3/3 online; console link opens the authenticated object browser |
| CUDA API recreation | Both models reload on GPU; 54 events, 49 alerts, 935 JSON objects and 32 clip-bucket objects preserved |
| Offline preparation after project move | Both real weights validated with container networking disabled |
| Compose configurations | CPU, CUDA, Triton, DeepStream and legacy auto rendered successfully with one API service |
| Triton preparation | Real tensor contracts validated; supplied config.pbtxt files retained |
| Changed Python files and dashboard JavaScript | Ruff and JavaScript syntax checks passed |
| Consolidated workspace | One `inferops/projects/` folder; source at `inferops/inferops/`; deployment, dashboard, docs and research moved under the workspace |
| Tests after reorganization | 159 engine tests and 9 driver tests passed; all five Compose modes rendered through the root entry point |
| CUDA video after reorganization | 187 frames, 9 alerts and 195 evidence objects; both models remained on GPU |
| Browser scenario after reorganization | Real sample frames and video, alerts, persisted activity and config reload passed; 10 screenshots, zero browser errors |
| Storage after reorganization | All 54 events, 49 alerts, 935 JSON records and 32 clip-bucket objects present before new verification runs |

The engine tests cover Hugging Face repository/filename/revision/token handling,
Google Drive confirmation, direct downloads, tar/zip extraction, custom installers,
offline caching, source changes, refresh, interrupted downloads, invalid graphs,
tensors and checksums. Provider unit tests use isolated test doubles. The real
driver download checks additionally contacted the configured public sources.

Logical and explicit variants, per-stage overrides, tracking filters, custom
stages, project/model editor validation and source changes are covered. New
scaffolds are empty, and model scaffolds contain no synthetic variants.

The CPU video verification checks WebSocket delivery, Postgres alerts and fusion
events, per-frame JSON metadata, JPEG snapshots, H.264 clips, evidence links and
HTTP range retrieval. Recreating containers preserves storage through named
volumes and retains weights in the mounted project folder.

CUDA checks ran on an NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04.
The NVIDIA Container Toolkit was installed by the user through the repository
installer. Docker GPU access and both models' CUDA execution were verified;
ONNX Runtime 1.26.0 activated CUDAExecutionProvider without device downgrade.
Triton serving and DeepStream decoding have not been executed.
Whole-repository Ruff still reports existing
findings in unrelated files; the files changed for this implementation pass.

Raw local reports and the engine/driver JUnit files are grouped in
`.run/verification/configuration/` relative to the current repository root. Earlier scratch projects, duplicate reports
and obsolete build/test logs were removed during workspace cleanup. Current
startup, browser, CUDA video, offline and persistence reports use the `current-`
prefix. The native PID file was removed after stopping the old API; Docker now
owns the running API.

The root `compose.yaml` is now a thin include entry point. Deployment definitions
live in `deploy/`. Empty `domains/` and `examples/` folders, the project
compatibility symlink and old Compose wrappers were removed. Test scenarios
live in `inferops/tests/scenarios/`; tiny model generation belongs to
`inferops/testing/`. The `layout-` reports record this reorganization. The earlier
`inferops/` workspace has since become the Git repository root; paths in the
historical validation table above refer to its previous parent checkout.
