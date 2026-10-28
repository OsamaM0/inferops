"""Access configuration and real subprocess lifetime checks without root or a GPU."""
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import runpod_access


class AccessTests(unittest.TestCase):
    def test_merge_keys_preserves_existing_and_is_idempotent(self):
        existing = '# injected\nssh-ed25519 AAAA existing\n'
        supplied = 'ssh-ed25519 BBBB added\nssh-ed25519 AAAA existing\n'
        merged = runpod_access.authorized_keys(existing, supplied)
        self.assertEqual(merged, existing + 'ssh-ed25519 BBBB added\n')
        self.assertEqual(runpod_access.authorized_keys(merged, supplied), merged)

    def test_no_keys_fails_without_including_environment_values(self):
        with self.assertRaisesRegex(ValueError, 'SSH needs PUBLIC_KEY'):
            runpod_access.authorized_keys('# comment\n', '')

    def test_ssh_key_only_auth_and_vscode_forwarding(self):
        config = runpod_access.ssh_config('/key', '/authorized', '/pid')
        for option in ('PasswordAuthentication no', 'KbdInteractiveAuthentication no',
                       'PermitRootLogin prohibit-password', 'AllowTcpForwarding local',
                       'Subsystem sftp internal-sftp', 'UsePAM yes'):
            self.assertIn(option, config)

    def test_flags_are_explicit_and_default_off(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(runpod_access.enabled('ACCESS'))
        with patch.dict(os.environ, {'ACCESS': 'true'}):
            self.assertTrue(runpod_access.enabled('ACCESS'))
        with patch.dict(os.environ, {'ACCESS': 'maybe'}), self.assertRaises(ValueError):
            runpod_access.enabled('ACCESS')

    def test_editor_requires_password_before_installing_or_writing(self):
        with patch.dict(os.environ, {'INFEROPS_SSH_ENABLED': 'false',
                                    'INFEROPS_CODE_SERVER_ENABLED': 'true',
                                    'INFEROPS_CODE_SERVER_PASSWORD': ''}), patch.object(os, 'geteuid', return_value=0), self.assertRaisesRegex(ValueError, 'at least 12 characters'):
            runpod_access.main()

    def test_access_survives_app_failure_then_signal_stops_owned_children(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pidfile = root / 'access.pid'
            helper = ('import os,time; from pathlib import Path; '
                      f'Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)')
            module_root = str(Path(__file__).resolve().parents[1])
            supervisor = (
                'from pathlib import Path; from scripts.runpod_access import supervise; '
                f'supervise([{sys.executable!r}, "-c", "raise SystemExit(7)"], '
                f'[("access", [{sys.executable!r}, "-c", {helper!r}], None)], Path({directory!r}))'
            )
            with (root / 'output').open('w+') as output:
                process = subprocess.Popen([sys.executable, '-c', supervisor], cwd=module_root,
                                           stdout=output, stderr=output)
                try:
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        output.seek(0)
                        if 'access remains available for repair' in output.read():
                            break
                        if process.poll() is not None:
                            self.fail('Supervisor exited before inference failure was handled')
                        time.sleep(0.1)
                    else:
                        self.fail('Inference failure was not reported')
                    child_pid = int(pidfile.read_text())
                    os.kill(child_pid, 0)
                    process.send_signal(signal.SIGTERM)
                    self.assertEqual(process.wait(timeout=10), 143)
                    with self.assertRaises(ProcessLookupError):
                        os.kill(child_pid, 0)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=10)


if __name__ == '__main__':
    unittest.main()
