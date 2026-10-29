"""RunPod startup checks without a GPU, model server or database."""
from __future__ import annotations

import contextlib
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts import runpod_preflight


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class TritonReadinessTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.clock = Clock()
        self.client = MagicMock()
        self.client.__enter__.return_value = self.client
        self.grpc = types.SimpleNamespace(InferenceServerClient=MagicMock(return_value=self.client))
        self.importer = self.stack.enter_context(patch.object(runpod_preflight.importlib, "import_module", return_value=self.grpc))
        self.stack.enter_context(patch.object(runpod_preflight.time, "monotonic", self.clock.monotonic))
        self.stack.enter_context(patch.object(runpod_preflight.time, "sleep", self.clock.sleep))

    def test_ready_server_uses_configured_grpc_endpoint_and_closes_client(self):
        self.client.is_server_ready.return_value = True
        runpod_preflight.wait_for_triton("model-host:18001", timeout_s=30)
        self.grpc.InferenceServerClient.assert_called_once_with(url="model-host:18001")
        self.client.is_server_ready.assert_called_once_with(client_timeout=5.0)
        self.client.__exit__.assert_called_once()
        self.assertEqual(self.importer.call_args_list[0].args, ("tritonclient.grpc.aio",))

    def test_retries_connection_errors_and_loading_server_until_ready(self):
        self.client.is_server_ready.side_effect = [ConnectionError("connection refused"), False, True]
        runpod_preflight.wait_for_triton("127.0.0.1:8001", timeout_s=10)
        self.assertEqual(self.client.is_server_ready.call_count, 3)
        self.assertEqual(self.clock.now, 2.0)

    def test_unready_server_has_bounded_wait_and_actionable_error(self):
        self.client.is_server_ready.return_value = False
        with self.assertRaisesRegex(RuntimeError, "model-host:18001.*server reported not ready.*INFEROPS_TRITON_URL"):
            runpod_preflight.wait_for_triton("model-host:18001", timeout_s=2.5)
        self.assertEqual(self.clock.now, 2.5)
        self.assertEqual(
            [call.kwargs["client_timeout"] for call in self.client.is_server_ready.call_args_list],
            [2.5, 1.5, 0.5],
        )
        self.client.__exit__.assert_called_once()

    def test_connection_failure_is_included_in_timeout_diagnostic(self):
        self.client.is_server_ready.side_effect = ConnectionError("connection refused")
        with self.assertRaisesRegex(RuntimeError, "connection refused.*Native Triton autostart"):
            runpod_preflight.wait_for_triton("127.0.0.1:8001", timeout_s=1)
        self.client.__exit__.assert_called_once()

    def test_missing_async_client_reports_installation_fix_without_waiting(self):
        self.importer.side_effect = ImportError("No module named tritonclient")
        with self.assertRaisesRegex(RuntimeError, r"python -m pip install -e '.\[triton\]'"):
            runpod_preflight.wait_for_triton("127.0.0.1:8001", timeout_s=30)
        self.grpc.InferenceServerClient.assert_not_called()
        self.assertEqual(self.clock.now, 0)

    def test_rpc_time_counts_toward_deadline(self):
        def unavailable(*, client_timeout):
            self.clock.now += client_timeout
            raise TimeoutError("RPC timed out")

        self.client.is_server_ready.side_effect = unavailable
        with self.assertRaisesRegex(RuntimeError, "RPC timed out"):
            runpod_preflight.wait_for_triton("127.0.0.1:8001", timeout_s=2)
        self.assertEqual(self.clock.now, 2)
        self.client.is_server_ready.assert_called_once_with(client_timeout=2)

    def test_invalid_timeouts_fail_before_connecting(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, "finite and positive"):
                runpod_preflight.wait_for_triton("127.0.0.1:8001", timeout_s=timeout)
        self.grpc.InferenceServerClient.assert_not_called()


class RunPodModeTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        settings_module = types.ModuleType("inferops.adapters.outbound.projects.settings")
        self.settings = types.SimpleNamespace(execution_mode="triton", triton=types.SimpleNamespace(url="remote:9001"))
        settings_module.load_settings = lambda: self.settings
        self.stack.enter_context(patch.dict("sys.modules", {settings_module.__name__: settings_module}))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.wait = self.stack.enter_context(patch.object(runpod_preflight, "wait_for_triton"))
        self.stack.enter_context(patch.object(runpod_preflight.sys, "argv", ["runpod_preflight.py"]))

    def test_uses_effective_settings_and_environment_timeout(self):
        with patch.dict("os.environ", {"INFEROPS_TRITON_STARTUP_TIMEOUT_SECONDS": "12"}):
            runpod_preflight.main()
        self.wait.assert_called_once_with("remote:9001", timeout_s=12)

    def test_command_line_timeout_overrides_environment(self):
        with patch.dict("os.environ", {"INFEROPS_TRITON_STARTUP_TIMEOUT_SECONDS": "12"}), patch.object(
            runpod_preflight.sys, "argv", ["runpod_preflight.py", "--timeout", "3"],
        ):
            runpod_preflight.main()
        self.wait.assert_called_once_with("remote:9001", timeout_s=3)

    def test_local_modes_skip_triton_even_with_invalid_triton_timeout(self):
        for mode in ("cpu", "cuda", "auto", "tensorrt"):
            with self.subTest(mode=mode), patch.dict(
                "os.environ", {"INFEROPS_TRITON_STARTUP_TIMEOUT_SECONDS": "unused"},
            ):
                self.settings.execution_mode = mode
                runpod_preflight.main()
        self.wait.assert_not_called()

    def test_unavailable_server_exits_cleanly_before_api_start(self):
        self.wait.side_effect = RuntimeError("Triton is unavailable")
        with self.assertRaisesRegex(SystemExit, r"\[inferops/runpod\] ERROR: Triton is unavailable"):
            runpod_preflight.main()


class RunPodDependencyTests(unittest.TestCase):
    """Execute the real install block with pip stubbed; never install GPU wheels."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        self.bin = self.root / "venv/bin"
        self.data.mkdir()
        self.bin.mkdir(parents=True)
        checkout = Path(__file__).resolve().parents[1]
        self.script = (checkout / "runpod-start.sh").read_text()
        (self.root / "runpod-start.sh").write_text(self.script)
        self.spec = (checkout / "pyproject.toml").read_bytes()
        (self.root / "pyproject.toml").write_bytes(self.spec)
        self.log = self.root / "calls"
        for name in ("python", "python3", "inferops"):
            executable = self.bin / name
            executable.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n')
            executable.chmod(0o755)

    def install(self, **overrides):
        # Run the shipped dependency block without starting database/media services.
        start = self.script.index('if [[ ! -x "$VENV/bin/inferops" ]]; then')
        end = self.script.index("info 'Preparing project model artifacts...'")
        result = subprocess.run(
            ["bash", "-euo", "pipefail", "-c", "info() { :; }\nfail() { echo \"$*\" >&2; exit 1; }\n" + self.script[start:end]],
            cwd=self.root, check=True, capture_output=True, text=True,
            env={**os.environ, "VENV": str(self.bin.parent), "DATA": str(self.data),
                 "CALL_LOG": str(self.log), "PATH": f"{self.bin}:{os.environ['PATH']}",
                 "PYTHON": str(self.bin / "python3"), "PYTHON_VERSION": "3.12",
                 "INFEROPS_TRITON_CLIENT_VERSION": "", **overrides},
        )
        self.assertEqual(result.stderr, "")
        calls = self.log.read_text() if self.log.exists() else ""
        self.log.unlink(missing_ok=True)
        return calls

    def test_old_cuda_only_cache_installs_triton_then_reuses_cache(self):
        (self.data / "python-requirements.sha256").write_text(hashlib.sha256(self.spec).hexdigest() + "\n")
        self.assertIn("-m pip install -e .[server,cuda,triton,video,driver_face]", self.install())
        self.assertEqual(self.install(), "")

    def test_bootstrap_change_invalidates_dependency_cache(self):
        self.install()
        (self.root / "runpod-start.sh").write_text(self.script + "\n# bootstrap updated\n")
        self.assertIn("-m pip install -e", self.install())

    def test_recreated_virtualenv_does_not_reuse_old_marker(self):
        self.install()
        (self.bin / "inferops").unlink()
        calls = self.install()
        self.assertIn("-m venv", calls)
        self.assertIn("-m pip install -e", calls)

    def test_changing_client_pin_reinstalls_matching_sdk(self):
        self.install()
        pin = {"INFEROPS_TRITON_CLIENT_VERSION": "2.68.0"}
        self.assertIn("tritonclient[grpc,http]==2.68.0", self.install(**pin))
        self.assertEqual(self.install(**pin), "")

    def test_changing_python_version_invalidates_cache(self):
        self.install()
        self.assertIn("-m pip install -e", self.install(PYTHON_VERSION="3.11"))


class RunPodImageTests(unittest.TestCase):
    def setUp(self):
        self.script = (Path(__file__).resolve().parents[1] / "runpod-start.sh").read_text()

    def test_bootstrap_accepts_current_interpreter_and_isolates_its_environment(self):
        start = self.script.index('command -v "$PYTHON"')
        end = self.script.index("# Local services must bypass")
        result = subprocess.run(
            ["bash", "-euo", "pipefail", "-c", "info() { :; }\nfail() { exit 1; }\n" +
             self.script[start:end] + '\nprintf "%s" "$VENV"'],
            env={**os.environ, "PYTHON": sys.executable}, text=True, capture_output=True, check=True,
        )
        self.assertEqual(result.stdout, f"/workspace/.venvs/inferops-py{sys.version_info.major}.{sys.version_info.minor}")

    def test_loopback_proxy_bypass_preserves_existing_exclusions(self):
        lines = [line for line in self.script.splitlines() if line.startswith(("export no_proxy=", "export NO_PROXY=", "export no_grpc_proxy="))]
        result = subprocess.run(
            ["bash", "-eu", "-c", "\n".join(lines) + '\nprintf "%s" "$no_grpc_proxy"'],
            env={"no_proxy": "internal.example", "NO_PROXY": "corp.example", "no_grpc_proxy": "grpc.example"},
            text=True, capture_output=True, check=True,
        )
        for host in ("internal.example", "corp.example", "grpc.example", "127.0.0.1", "localhost", "::1"):
            self.assertIn(host, result.stdout.split(","))

    def test_postgres_major_mismatch_preserves_existing_cluster(self):
        start = self.script.index('if [[ -f "$PGDATA/PG_VERSION" ]]; then')
        end = self.script.index('install -d -m 700 -o postgres', start)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            version = root / "PG_VERSION"
            version.write_text("14\n")
            postgres = root / "postgres"
            postgres.write_text('#!/bin/sh\necho "postgres (PostgreSQL) 16.15 (Ubuntu 16.15)"\n')
            postgres.chmod(0o755)
            result = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", 'fail() { echo "$*" >&2; exit 1; }\n' + self.script[start:end]],
                env={**os.environ, "PGDATA": str(root), "PGBIN": str(root)}, text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Persistent PostgreSQL 14 data requires matching server", result.stderr)
            self.assertEqual(version.read_text(), "14\n")


if __name__ == "__main__":
    unittest.main()
