"""At-least-once delivery from transactionally committed outbox records."""
from __future__ import annotations

import asyncio
import logging
import uuid

log = logging.getLogger(__name__)


class OutboxRelay:
    def __init__(self, repository, delivery, *, interval: float = .25):
        self.repository, self.delivery = repository, delivery
        self.owner = uuid.uuid4().hex
        self.interval = interval
        self.task = None

    async def flush(self) -> int:
        messages = await self.repository.claim_outbox(self.owner)
        delivered = 0
        for message in messages:
            try:
                await self.delivery.publish(message["topic"], message.get("key", ""), message["payload"], message["id"])
                await self.repository.ack_outbox(message["id"], self.owner, message["generation"])
                delivered += 1
            except Exception as exc:
                await self.repository.retry_outbox(message["id"], self.owner, message["generation"], str(exc), 5)
                log.warning("outbox_delivery_failed", extra={"event_id": message["id"], "error": str(exc)})
        return delivered

    async def start(self):
        async def loop():
            while True:
                try:
                    await self.flush()
                except Exception:
                    log.exception("outbox_claim_failed")
                await asyncio.sleep(self.interval)
        self.task = asyncio.create_task(loop(), name="inferops-outbox")

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        try:
            await asyncio.wait_for(self.flush(), timeout=5)
        except Exception:
            log.warning("outbox_shutdown_pending; durable records remain retryable")
        await self.delivery.close()
