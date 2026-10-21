# End-to-end scenarios

Run these checks from the project root. Reports belong in `.run/`; sample clips
live in `research/`.

## Real video inference

`scripts/verify_video.py` checks every decoded frame, actual execution providers
and persisted, playable evidence using the shared engine and real project
weights. Prepare the project models and provide a dedicated disposable
PostgreSQL database before running it:

```bash
.venv/bin/python scripts/verify_video.py \
  --runtime cpu --database-url "$INFEROPS_TEST_PG_URL" \
  --output .run/verification/cpu
```

Use `--runtime cuda` to verify GPU execution. This check writes to the supplied
database and output directory; use resources reserved for verification.

## Media router verification

To verify optional MediaMTX with real driver footage/models, first start the
media profile and the qualified Triton server, then run:

```bash
docker compose --profile media up -d mediamtx
.venv/bin/python tests/scenarios/verify_media_router.py --runtime triton
```

This scenario uses an isolated API/database, FFmpeg publisher, disposable RTSP
origin and Playwright browser. It checks moving WebRTC/HLS video while inference
is active, annotated previews, camera/router/API recovery, delayed router startup,
preview-disabled processing, image/video uploads, persisted results and playable
incident evidence. It restarts the selected router; run while user cameras are
idle. Reports remain in `.run/verification/media-real/`. It does not change the
user API or its databases. Native CPU/CUDA are selectable with `--runtime`.

## Unified console browser verification

Install Playwright/Chromium for browser scenarios. The canonical workspace
address is `/ui/`; `/frontend/` remains available for older automation.

`verify_unified_console.browser.js` exercises all eleven views in one document,
persisted preferences, the command palette, JSON/path/header/multipart API calls,
HTTP errors, real source registration and preview, asynchronous jobs, manual
refresh and mobile layouts. It creates and removes its own source registration.
Run only against an isolated server using the core `tiny_classifier` pipeline at
`http://127.0.0.1:18083`; never use an operator's running project. The server needs
a disposable PostgreSQL database. The helper creates the tiny model and copies
configuration into temporary storage, so browser saves cannot edit the checkout.

```bash
docker compose -f deploy/testing/compose.postgres.yaml up -d --wait
INFEROPS_TEST_PG_URL=postgresql://postgres:inferops-test@127.0.0.1:55432/inferops_test \
  .venv/bin/python tests/scenarios/serve_unified_console.py
# In another terminal:
playwright-cli -s=inferops-console open http://127.0.0.1:18083/ui/
playwright-cli -s=inferops-console run-code --filename=tests/scenarios/verify_unified_console.browser.js
playwright-cli -s=inferops-console close
```

Screenshots are saved under `.run/verification/unified-console/`.

`verify_configuration_steps.browser.js` uses the same disposable server to build
a new pipeline entirely through configuration forms. It adds and reorders steps,
selects a logical model, configures a custom stage and a live output, validates
without saving, saves, and runs real tiny-model inference. It also checks mobile
layout and arbitrary plugin fields. It creates configuration only inside the
helper's temporary directory:

```bash
playwright-cli -s=inferops-config open http://127.0.0.1:18083/ui/#settings
playwright-cli -s=inferops-config run-code --filename=tests/scenarios/verify_configuration_steps.browser.js
playwright-cli -s=inferops-config close
```
