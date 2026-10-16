"""Test assets stay outside the checkout and belong to one test process."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from inferops.adapters.outbound.projects.models import load_model_file
from inferops.testing import PROJECT_ROOT, TEST_MODEL_REPOSITORY, TINY_MODEL_PATH


def test_model_configuration_uses_process_temporary_storage():
    assert not TINY_MODEL_PATH.is_relative_to(PROJECT_ROOT)
    assert TINY_MODEL_PATH.is_file()
    assert os.environ["INFEROPS_MODEL_REPOSITORY"] == str(TEST_MODEL_REPOSITORY)
    spec = load_model_file(PROJECT_ROOT / "inferops/testing/configs/models/tiny_classifier.yaml")
    assert Path(spec.artifact.path) == TINY_MODEL_PATH


def test_other_test_process_has_its_own_model_storage():
    result = subprocess.run([
        sys.executable, "-c",
        "from inferops.testing import configure_test_env, TINY_MODEL_PATH; "
        "configure_test_env(); print(TINY_MODEL_PATH)",
    ], capture_output=True, text=True, check=True)
    other = Path(result.stdout.strip())
    assert other != TINY_MODEL_PATH
    assert not other.is_relative_to(PROJECT_ROOT)
    assert TINY_MODEL_PATH.is_file()
