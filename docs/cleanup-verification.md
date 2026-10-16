# CPU/CUDA video verification and source cleanup

Verified on 15 October 2026 with the actual
`research/driver_training/test.mp4`, existing RF-DETR/FaceMesh weights, Python
3.12, ONNX Runtime 1.26 and an NVIDIA RTX 4060 laptop GPU (8 GiB).
The video SHA-256 is
`e45dced12684ae6155fa30d34ce796288ace8ff5b64a8285d5646a6965462dc8`.

## Real video results

The full driver pipeline ran before cleanup and again against the final source
and simplified driver configuration. Frames, model call counts, executed provider node counts, events and accepted
incident counts match the baseline within each runtime. Persisted decision types,
severities and media timestamps also match; audit wall times and generated IDs differ.

| Check | CPU | CUDA |
| --- | --- | --- |
| Decoded/processed frames | 187 / 187 | 187 / 187 |
| RF-DETR / FaceMesh calls | 187 / 67 | 187 / 67 |
| Actual profiled provider | CPU | CUDA for both models |
| Accepted incidents / events | 9 / 10 | 9 / 10 |
| Checksum-verified artifacts | 27 | 27 |
| Evidence outcome | 2 Saved, 7 Incomplete | 2 Saved, 7 Incomplete |

Every incident has a verified JPEG snapshot, playable MP4 clip and checksummed
artifact manifest. Incomplete records expose missing media intervals: the finite
source cannot supply all requested pre/post context. No final record failed or
remained pending. CPU and CUDA detection totals differ slightly across providers;
cross-provider numerical differences are not a cleanup regression.

Provider traces prove execution rather than merely listing installed providers.
RF-DETR executed 160,072 CPU nodes / 158,576 CUDA nodes; FaceMesh executed
24,187 CPU nodes / 14,807 CUDA nodes. Raw reports, profiles, SQLite records,
artifacts and parity checks remain under `.run/verification/video-final-cpu/`,
`video-final-cuda/` and `cleanup-parity.json`.

These are offline correctness runs with capture enabled, four inference threads,
profiling, filesystem artifacts, zero test disk reserve and an explicit 1,800-second
capture deadline. The initial profiled run used the ordinary 60-second capture
budget and exposed timeout-related incomplete states. The final offline budget
avoids confusing slow verification with missing media. Production defaults were
not raised. Concurrent verification, profiling and fsync overhead mean elapsed
times (CPU 284.9 seconds; CUDA 195.4 seconds) are not live-capacity benchmarks.
PostgreSQL/MinIO correctness is exercised separately by integration tests.

## Cleanup

Removed 28 obsolete source modules: duplicate HTTP implementations, cached
pipeline runner/global stores, old per-frame evidence/delivery sinks, unused
global PostgreSQL/S3/Redis clients and frame cache, and the superseded runtime
validation script. Removed active-project/transport factories and ignored driver
evidence trigger settings. The manifest is
`.run/verification/pre-cleanup/removed-modules.json`.

Preserved runtime implementations, preprocessing/postprocessing, tracking,
fusion/window/expression algorithms, driver calculations, annotations/encoding,
public `Engine`/`Model`/`InferenceSession`, SDK and useful extension aliases.
The pure annotation/encoding helpers and live serializer now have dedicated
modules. Python, HTTP, CLI and workers use one execution engine. Project-relative
plugins require an explicit project context. Architecture documentation/gallery
show the current engine, evidence recovery and deployment; obsolete diagrams
were removed.

Rollback is tested using the archived compatibility wheel in an isolated process,
rather than retaining duplicate execution code. Restore the matching project
configuration with the designated release; the pre-cleanup driver YAML snapshot
is archived in `.run/verification/pre-cleanup/driver-configuration.json`.
The test reads new-system records,
serves clips, recovers a pending upload, processes additional work, then upgrades
again without replacing artifacts or downgrading schemas.

## Regression and packaging checks

- Core regression suite: **284 passed**, 26 optional cases skipped; one upstream
  Starlette/httpx deprecation warning.
- PostgreSQL/MinIO transactions, recovery, idempotency and archived rollback:
  **45 passed**, no skips, using disposable schemas/buckets/containers.
- Driver project suite: **15 passed**, including the full real video regression.
- Ruff, Compose configuration and Git whitespace checks passed.
- Clean wheel excludes removed modules; public library imports from the wheel
  create no server/configuration/database modules.
- Three current SVGs rendered and loaded; mobile page overflow and zoom controls
  passed Playwright checks.

An executable boundary test prevents obsolete execution modules and imports from
returning. WebSocket cancellation checks cover quiet subscriber removal and
session/resource release. Job tests now submit real image data instead of
expecting successful inference from an empty payload.

## Repeat

Use a fresh output directory for each verification run:

```bash
.venv/bin/python scripts/verify_video.py --runtime cpu --output .run/verification/recheck-cpu
.venv/bin/python scripts/verify_video.py --runtime cuda --output .run/verification/recheck-cuda
.venv/bin/python -m pytest tests -q -o addopts=''
.venv/bin/python -m pytest projects/driver_behaviour/tests -q -o addopts=''
```

For production-adapter tests set `INFEROPS_TEST_PG_URL` and
`INFEROPS_TEST_S3_ENDPOINT` to dedicated test services, then run
`tests/test_redesign_persistence.py` and `tests/test_redesign_production.py`.
`INFEROPS_TEST_ROLLBACK_WHEEL` selects the designated compatibility release.
Resource defaults remain provisional until deployment-host live latency,
concurrency and overload benchmarks pass with evidence capture enabled.

## Additional legacy file removal

The remaining `inferops/compat/` folder was removed. Its active stage factory now
lives in `inferops/adapters/outbound/stages/factory.py`. Redundant
`inferops/domain/fusion/engine.py` and `inferops/alerts/engine.py` shims were removed;
current code imports the domain implementations directly. Package-level public
exports still resolve to those implementations. The unused active-project query
was removed. The outdated redesign and configuration-simplification verification
documents were removed, and links now point to this current report.

The workspace cleanup removed 44 generated build, metadata and cache folders,
including cached copies of deleted modules. Environments, weights, persistent
data and release/recovery verification artifacts remain intact. The executable
boundary guard now rejects the removed shims as well. Regression results are in
`.run/verification/legacy-removal-core.xml` and `legacy-removal-driver.xml`: **284
core tests and 15 driver tests passed**, including the full test video. The clean
wheel excludes the deleted shims and imports the public library without server
configuration. All local documentation links and current SVG XML are valid.

## Physical structure consolidation

The package now follows the [consolidated layout](structure.md). Public handles
live in `public.py`; `library.py` is removed. Domain schemas, tracking and fusion
have moved into `domain/`. Inbound HTTP/UI, CLI and workers and outbound runtimes,
plugins, projects and observability now live under `adapters/`. The six remaining
top-level directories are enforced by an executable architecture test. Inward
layers have no exceptions allowing imports from old modules.

The local Python wrappers now use the same application `ModelPool` as engine
predictions. Logical and explicit model handles share backend instances; closing
waits for active inference. Session identity/time/state, ordered stage execution,
incident acceptance, revision activation and engine resource lifecycle have
separate application modules. Standalone library use still needs no server stack.

| Verification after consolidation | Result |
| --- | --- |
| Core suite | 287 passed; 26 optional cases skipped; one upstream deprecation warning |
| Driver suite including real video | 15 passed |
| PostgreSQL/MinIO persistence, recovery and archived rollback | 45 passed |
| Actual full CPU and CUDA video runs | 187 frames each; RF-DETR 187 / FaceMesh 67 calls; 27 verified artifacts each |
| Video parity | Detection counts, model selection, event and incident counts match the prior per-runtime baseline |
| Evidence coverage | Each runtime: 2 Saved, 7 Incomplete due to missing finite-source context |
| CLI and deployment | Editable CLI entry point refreshed; project validation and Compose configuration passed |
| Packaging and UI | Wheel excludes old trees; standalone exports and packaged UI HTTP routes passed |
| Architecture gallery | Four SVGs rendered and loaded; mobile layout and zoom controls passed |

Video reports/profiles/artifacts are in `.run/verification/structure-video-cpu/`
and `structure-video-cuda/`; comparisons are in `structure-parity.json`. Tests
are recorded in `structure-core.xml`, `structure-driver.xml` and
`structure-production.xml`. Internal import paths and plugin YAML references
changed; public imports from `inferops` and the HTTP/SDK contracts remain stable.
Archived release rollback restores matching project code/imports with its wheel.
These offline runs do not establish live capacity; defaults remain provisional.

## Test-generated folder correction

The root `model_repository/` was created by the tiny ONNX test fixture. Tests now
use process-owned system temporary storage for models, the test database and
spool. The test model configuration and all fixture consumers use that location;
temporary storage is cleaned at process exit. Concurrent suites no longer share
or delete one another's model. The previous root test model and lock were removed
without changing `projects/*/model_repository/` or real weights.

Pytest and Ruff caches now live under `.run/cache/`. `make clean` removes that
cache directory while preserving runtime reports, spools, models and other
`.run/` contents. `.venv/` contains installed dependencies; `inferops.egg-info/`
is normal editable-install metadata, not an alternate implementation.
