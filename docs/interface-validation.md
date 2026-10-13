# Interface and layout validation

Verified on 10 October 2026 from the consolidated project root.

- Core suite and shipped project tests passed with `make test-all` (162 engine,
  9 project tests). Targeted API/UI, startup, WebSocket and device checks passed
  again after the final changes.
- Compose configuration validated for cpu, auto, cuda, triton and deepstream.
  The complete CPU preparation image, CUDA API image and original MinIO image
  built successfully. The current deployment loads two real models on CUDA.
- The wheel includes HTML, CSS, JavaScript and the UI router. UI pages, guide
  and assets served successfully from an extracted wheel outside the checkout.
- Browser checks covered desktop (1440 px), mobile (390 px), service cards,
  native service pages, the interface guide and the ten-second stream refresh.
  The workspace has no page-level horizontal overflow at these sizes. A real
  image ran through the selected driver pipeline from the UI with no browser
  console errors; it reported one processed frame and two loaded models.
- The MinIO admin console login succeeds and exposes Buckets, Policies,
  Identity, Monitoring and Configuration. Its browser displays the original
  1,311 evidence records and 50 clip/snapshot objects.
- All 1,361 migrated objects were checked by SHA-256 readback, content type,
  user metadata and tags. The previous volume and file backup are retained;
  `.run/verification/minio-migration.json` records the transfer (11,093,187 bytes).
- Grafana provisions 16 panels with the correct WebSocket metric; Prometheus
  scrapes the API, MinIO and Redpanda successfully. The inactive Triton target
  is expected in CUDA mode.

Triton and DeepStream configuration was checked, but their hardware execution
was not exercised in this UI change. The full original MinIO console is pinned
with an older upstream release; its version tradeoff and storage migration are
explained in [the interface guide](interfaces.md).
