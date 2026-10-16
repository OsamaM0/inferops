"""Explicit construction for shared-service workers."""
def server_engine():
    from inferops.bootstrap import build_server_engine
    return build_server_engine()[0]
