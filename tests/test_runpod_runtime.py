"""Native RunPod process ordering and ownership without a GPU or database."""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts import runpod_preflight, runpod_runtime


class NativeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.repository = self.root / "model repository"
        model = self.repository / "model"
        model.mkdir(parents=True)
        (model / "config.pbtxt").write_text('name: "model"')
        self.log = self.root / "logs/triton.log"
        self.settings = types.SimpleNamespace(
            execution_mode="triton",
            triton=types.SimpleNamespace(url="127.0.0.1:8001", http_url="http://127.0.0.1:8000"),
        )
        self.stack.enter_context(patch.dict("os.environ", {}, clear=True))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.triton = MagicMock()
        self.triton.poll.return_value = None
        self.api = MagicMock()
        self.api.poll.return_value = 0
        self.spawn = self.stack.enter_context(patch.object(
            runpod_runtime.subprocess, "Popen", side_effect=[self.triton, self.api],
        ))
        self.wait = self.stack.enter_context(patch.object(runpod_preflight, "wait_for_triton"))
        self.listener = self.stack.enter_context(patch.object(runpod_runtime, "listening", return_value=False))
        self.which = self.stack.enter_context(patch.object(
            runpod_runtime.shutil, "which", return_value="/opt/tritonserver/bin/tritonserver",
        ))
        self.stack.enter_context(patch.object(runpod_runtime.time, "sleep"))

    def run_runtime(self, **kwargs):
        return runpod_runtime.run_runtime(
            self.settings, repository=self.repository, data=self.root / "app data", triton_log=self.log, **kwargs,
        )

    def test_starts_model_server_then_checks_readiness_then_starts_api(self):
        order = []

        def spawn(command, **_kwargs):
            if len(order) == 0:
                order.append("triton")
                self.assertEqual(command[0], "/opt/tritonserver/bin/tritonserver")
                self.assertIn(f"--model-repository={self.repository}", command)
                self.assertIn("--load-model=model", command)
                self.assertNotIn("--load-model=*", command)
                self.assertIn("--rate-limit=execution_count", command)
                return self.triton
            self.assertEqual(order, ["triton", "ready"])
            self.assertEqual(command[-2:], ["--data", str(self.root / "app data")])
            order.append("api")
            return self.api

        self.spawn.side_effect = spawn
        self.wait.side_effect = lambda *_args, **_kwargs: order.append("ready")
        self.assertEqual(self.run_runtime(), 0)
        self.assertEqual(order, ["triton", "ready", "api"])
        self.triton.terminate.assert_called_once()
        self.api.terminate.assert_not_called()

    def test_cuda_and_cpu_do_not_probe_or_start_triton(self):
        for mode in ("cpu", "cuda", "auto"):
            with self.subTest(mode=mode):
                self.settings.execution_mode = mode
                self.spawn.side_effect = [self.api]
                self.assertEqual(self.run_runtime(), 0)
        self.wait.assert_not_called()
        self.which.assert_not_called()
        self.listener.assert_not_called()

    def test_remote_server_is_reused_without_starting_local_triton(self):
        self.settings.triton.url = "remote-model-host:19001"
        self.spawn.side_effect = [self.api]
        self.assertEqual(self.run_runtime(), 0)
        self.assertEqual(self.spawn.call_count, 1)
        self.wait.assert_called_once()
        self.triton.terminate.assert_not_called()

    def test_existing_local_listener_is_checked_but_never_killed(self):
        self.listener.return_value = True
        self.spawn.side_effect = [self.api]
        self.assertEqual(self.run_runtime(), 0)
        self.assertEqual(self.spawn.call_count, 1)
        self.triton.terminate.assert_not_called()

    def test_disabled_autostart_uses_managed_external_server(self):
        self.spawn.side_effect = [self.api]
        self.assertEqual(self.run_runtime(autostart="false"), 0)
        self.assertEqual(self.spawn.call_count, 1)
        self.wait.assert_called_once()

    def test_missing_required_binary_fails_before_starting_api(self):
        self.which.return_value = None
        with patch.object(Path, "is_file", return_value=False), self.assertRaisesRegex(RuntimeError, "tritonserver is missing"):
            self.run_runtime(autostart="true")
        self.spawn.assert_not_called()

    def test_readiness_failure_stops_owned_server_and_prints_its_log(self):
        self.log.parent.mkdir()
        self.log.write_text("previous attempt: stale error\n")

        def spawn(*_args, **_kwargs):
            with self.log.open("a") as output:
                output.write("model failed to load: CUDA initialization failed\n" + "shutdown detail\n" * 50)
            return self.triton

        self.spawn.side_effect = spawn
        self.wait.side_effect = RuntimeError("not ready")
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            self.run_runtime()
        self.assertEqual(self.spawn.call_count, 1)
        self.triton.terminate.assert_called_once()
        self.assertIn("CUDA initialization failed", self.stderr.getvalue())
        self.assertNotIn("previous attempt: stale error", self.stderr.getvalue())

    def test_native_server_exit_during_startup_reports_log_and_never_starts_api(self):
        self.triton.poll.return_value = 1

        def ready(*_args, check_running, **_kwargs):
            check_running()

        self.wait.side_effect = ready
        with self.assertRaisesRegex(RuntimeError, "exited with status 1.*triton.log"):
            self.run_runtime()
        self.assertEqual(self.spawn.call_count, 1)

    def test_native_server_death_after_startup_stops_owned_api(self):
        self.triton.poll.side_effect = [None, 8, 8]
        self.api.poll.return_value = None
        with self.assertRaisesRegex(RuntimeError, "exited with status 8"):
            self.run_runtime()
        self.api.terminate.assert_called_once()

    def test_interrupt_cleans_up_native_server(self):
        self.wait.side_effect = SystemExit(143)
        with self.assertRaises(SystemExit):
            self.run_runtime()
        self.triton.terminate.assert_called_once()

    def test_api_failure_propagates_and_stops_native_server(self):
        self.api.poll.return_value = 7
        self.assertEqual(self.run_runtime(), 7)
        self.triton.terminate.assert_called_once()

    def test_rejects_inconsistent_native_endpoints(self):
        self.settings.triton.http_url = "http://remote:8000"
        with self.assertRaisesRegex(ValueError, "local gRPC, HTTP and metrics"):
            self.run_runtime()
        self.spawn.assert_not_called()

    def test_rejects_reusing_same_port_for_http_and_grpc(self):
        self.settings.triton.http_url = "http://127.0.0.1:8001"
        with self.assertRaisesRegex(ValueError, "ports must be distinct"):
            self.run_runtime()
        self.spawn.assert_not_called()

    def test_invalid_timeout_does_not_start_processes(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self.run_runtime(timeout_s=float("nan"))
        self.spawn.assert_not_called()

    def test_custom_ports_are_used_by_native_server(self):
        self.settings.triton.url = "localhost:18001"
        self.settings.triton.http_url = "http://localhost:18000"
        with patch.dict("os.environ", {"INFEROPS_TRITON_METRICS_URL": "http://localhost:18002"}):
            command = runpod_runtime.native_command(self.settings, self.repository, "tritonserver")
        self.assertIn("--grpc-port=18001", command)
        self.assertIn("--http-port=18000", command)
        self.assertIn("--metrics-port=18002", command)

    def test_rejects_missing_model_repository_before_spawning(self):
        (self.repository / "model/config.pbtxt").unlink()
        with self.assertRaisesRegex(ValueError, "No prepared Triton model"):
            self.run_runtime()
        self.spawn.assert_not_called()

    def test_checkpoint_directory_does_not_become_a_serving_model(self):
        (self.repository / "model/config.pbtxt").unlink()
        for name in ("face_mesh_onnx", "rfdetr_driver_behaviour_onnx"):
            model = self.repository / name
            model.mkdir()
            (model / "config.pbtxt").write_text(f'name: "{name}"')
        checkpoint = self.repository / "rfdetr_driver_behaviour_pth/1/model.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"preserve this checkpoint")
        command = runpod_runtime.native_command(self.settings, self.repository, "tritonserver")
        self.assertEqual(
            [argument for argument in command if argument.startswith("--load-model=")],
            ["--load-model=face_mesh_onnx", "--load-model=rfdetr_driver_behaviour_onnx"],
        )
        self.assertEqual(checkpoint.read_bytes(), b"preserve this checkpoint")

    def test_wildcard_directory_name_cannot_reenable_loading_all_artifacts(self):
        model = self.repository / "*"
        model.mkdir()
        (model / "config.pbtxt").touch()
        with self.assertRaisesRegex(ValueError, "cannot be named"):
            runpod_runtime.native_command(self.settings, self.repository, "tritonserver")


class OwnedProcessTests(unittest.TestCase):
    def test_unresponsive_owned_process_is_killed_after_grace_period(self):
        process = MagicMock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("child", 10), 0]
        runpod_runtime.stop_owned(process)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()

    def test_stops_real_owned_subprocess(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            runpod_runtime.stop_owned(process)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


class StartupLogTests(unittest.TestCase):
    def test_large_output_is_bounded_and_explicitly_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "triton.log"
            path.write_text("backend error\n" + "x" * 1000)
            captured = io.StringIO()
            with contextlib.redirect_stderr(captured):
                runpod_runtime.print_startup_log(path, 0, max_bytes=100)
            self.assertIn("backend error", captured.getvalue())
            self.assertIn("truncated at 100 bytes", captured.getvalue())
            self.assertNotIn("x" * 101, captured.getvalue())

    def test_replaced_log_is_read_from_beginning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "triton.log"
            path.write_bytes(b"backend error: bad byte \xff\n")
            captured = io.StringIO()
            with contextlib.redirect_stderr(captured):
                runpod_runtime.print_startup_log(path, 1000)
            self.assertIn("backend error", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
