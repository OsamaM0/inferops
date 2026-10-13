from pathlib import Path

from inferops.testing import configure_test_env

configure_test_env(domain_root=Path(__file__).resolve().parents[1])

from inferops.testing.fixtures import *  # noqa: E402,F403
