"""Print items to stdout (debug)."""
from __future__ import annotations

import orjson

from inferops.pipeline.stage import SinkStage


class SinkStdout(SinkStage):
    async def process(self, item):
        print(orjson.dumps(item.model_dump(mode="json")).decode())
        return None
