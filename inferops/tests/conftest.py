"""Core test suite: runs with NO domain pack active.

Domain packs carry their own tests under ``examples/<name>/tests`` and run in
a separate pytest process with that domain selected (``make test DOMAIN=...``).
"""
from inferops.testing import configure_test_env

configure_test_env(domain=None)

from inferops.testing.fixtures import *  # noqa: E402,F403
