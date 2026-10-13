"""Transport factory."""
from __future__ import annotations

from inferops.adapters.transport.base import Consumer, Message, Producer, Transport
from inferops.adapters.transport.inproc import InProcTransport
from inferops.core.config import settings

__all__ = [
    "Consumer", "InProcTransport", "Message", "Producer", "RedpandaTransport", "Transport",
    "get_transport", "set_transport",
]

_default: Transport | None = None


def RedpandaTransport(*args, **kwargs):  # lazy import
    from inferops.adapters.transport.redpanda import RedpandaTransport as _RP
    return _RP(*args, **kwargs)


def get_transport() -> Transport:
    global _default
    if _default is not None:
        return _default
    _default = InProcTransport() if settings.transport.kind == "inproc" else RedpandaTransport()
    return _default


def set_transport(t: Transport) -> None:
    global _default
    _default = t
