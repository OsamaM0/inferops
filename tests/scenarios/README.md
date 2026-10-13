# End-to-end scenarios

These scripts verify a running API with real project weights. Run from the
project root after starting the selected runtime:

```bash
.venv/bin/python tests/scenarios/verify_video.py \
  --mode cuda --out .run/verification/cuda.json
make scenario PROJECT=driver_behaviour
```

`verify_video.py` checks real inference, WebSocket alerts, database persistence,
MinIO evidence and media playback. The browser scenario checks the packaged
API workspace and can refresh screenshots using `SCENARIO_ARGS=--docs`.
Install Playwright/Chromium for browser scenarios. Reports belong in `.run/`;
sample clips live in `research/`. The canonical workspace address is `/ui/`;
`/frontend/` remains available for older automation.
