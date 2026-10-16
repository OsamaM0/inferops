"""ASGI entry point for the shared application services."""
from inferops.adapters.inbound.http.app import create_application


def create_app(*, engine=None):
    return create_application(engine)

app = create_app()
