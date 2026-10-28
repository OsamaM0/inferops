"""Optional RunPod development access, independent of inference startup success."""
from __future__ import annotations

import hashlib
import os
import shutil
import signal
import socket
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

CODE_VERSION = "4.143.1"
CODE_SHA256 = "614877cf71005f14ea252bd2d9b1c12237cf7d73681ef150f0dff7f13849fdb5"


def info(message):
    print(f"[inferops/access] {message}", flush=True)


def enabled(name):
    value = os.environ.get(name, "false").lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


def authorized_keys(existing, supplied):
    """Preserve injected keys and add each supplied key once, never log key text."""
    lines = existing.splitlines()
    for line in supplied.splitlines():
        line = line.strip()
        if line and line not in lines:
            lines.append(line)
    if not any(line.strip() and not line.lstrip().startswith("#") for line in lines):
        raise ValueError("SSH needs PUBLIC_KEY, SSH_PUBLIC_KEY or an existing /root/.ssh/authorized_keys")
    return "\n".join(lines) + "\n"


def ssh_config(key, keys, pid):
    return f"""Port 22
ListenAddress 0.0.0.0
HostKey {key}
PidFile {pid}
AuthorizedKeysFile {keys}
PermitRootLogin prohibit-password
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM yes
AllowUsers root
AllowTcpForwarding local
GatewayPorts no
X11Forwarding no
Subsystem sftp internal-sftp
"""


def install_code_server(data):
    destination = data / "bin" / f"code-server-{CODE_VERSION}-linux-amd64"
    executable = destination / "bin/code-server"
    if executable.is_file():
        return executable
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        staging = Path(directory)
        archive = staging / "code.tar.gz"
        url = f"https://github.com/coder/code-server/releases/download/v{CODE_VERSION}/{destination.name}.tar.gz"
        info(f"Downloading code-server {CODE_VERSION}...")
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as target:
            shutil.copyfileobj(response, target)
        digest = hashlib.sha256()
        with archive.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != CODE_SHA256:
            raise RuntimeError("code-server archive checksum mismatch")
        # Only the pinned, checksum-verified upstream release is extracted.
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                if not (staging / member.name).resolve().is_relative_to(staging.resolve()):
                    raise RuntimeError("Invalid code-server archive path")
            source.extractall(staging)
        (staging / destination.name).rename(destination)
    return executable


def stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def supervise(app_command, access_commands, log_dir):
    """Keep access alive when inference fails; stop every owned process on exit."""
    children = []
    logs = []

    def shutdown(signum, _frame):
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, shutdown)
    try:
        for name, command, env in access_commands:
            log = (log_dir / f"{name}.log").open("a")
            logs.append(log)
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
            children.append((name, process))
        # Detect immediate configuration/port errors before announcing access.
        time.sleep(1)
        for name, process in children:
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited; see {log_dir / (name + '.log')}")
        info("Development access started; expose 22/tcp for SSH and 8080/http for enabled code-server.")
        app = subprocess.Popen(app_command, start_new_session=True)
        children.append(("inferops", app))
        reported = False
        while True:
            for name, process in children[:-1]:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited; see {log_dir / (name + '.log')}")
            if not reported and app.poll() is not None:
                info(f"InferOps exited with status {app.returncode}; access remains available for repair. Restart the Pod after fixing settings/code.")
                reported = True
            time.sleep(1)
    finally:
        for _name, process in reversed(children):
            stop(process)
        for log in logs:
            log.close()


def main():
    app = Path(__file__).resolve().parents[1]
    command = ["bash", str(app / "runpod-start.sh")]
    ssh = enabled("INFEROPS_SSH_ENABLED")
    code = enabled("INFEROPS_CODE_SERVER_ENABLED")
    if not ssh and not code:
        os.execvp(command[0], command)
    if os.geteuid() != 0 or os.uname().machine != "x86_64":
        raise ValueError("Development access needs a root-based x86_64 Pod")
    password = os.environ.get("INFEROPS_CODE_SERVER_PASSWORD", "")
    if code and len(password) < 12:
        raise ValueError("Set INFEROPS_CODE_SERVER_PASSWORD to at least 12 characters")
    data = Path("/workspace/inferops-data")
    log_dir = data / "logs"
    secrets = data / "secrets"
    for path in (log_dir, secrets):
        path.mkdir(parents=True, exist_ok=True)
    secrets.chmod(0o700)
    commands = []
    # Fail clearly on a port collision; never replace an existing SSH/editor service.
    for port in ([22] if ssh else []) + ([8080] if code else []):
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", port))
    if ssh:
        if not Path("/usr/sbin/sshd").is_file():
            env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
            subprocess.run(["apt-get", "update"], env=env, check=True)
            subprocess.run(["apt-get", "install", "-y", "--no-install-recommends", "openssh-server"], env=env, check=True)
            # Some Debian package hooks start a default daemon. Port 22 was unused
            # before this installation; stop that newly installed service only.
            subprocess.run(["service", "ssh", "stop"], check=False)
        ssh_dir = Path("/root/.ssh")
        ssh_dir.mkdir(mode=0o700, exist_ok=True)
        ssh_dir.chmod(0o700)
        keys = ssh_dir / "authorized_keys"
        merged = authorized_keys(keys.read_text() if keys.exists() else "", "\n".join(
            os.environ.get(name, "") for name in ("PUBLIC_KEY", "SSH_PUBLIC_KEY")))
        keys.write_text(merged)
        keys.chmod(0o600)
        Path("/run/sshd").mkdir(mode=0o755, exist_ok=True)
        host_key = secrets / "ssh_host_ed25519_key"
        if not host_key.exists():
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(host_key)], check=True)
        config = secrets / "sshd_config"
        config.write_text(ssh_config(host_key, keys, data / "sshd.pid"))
        subprocess.run(["/usr/sbin/sshd", "-t", "-f", str(config)], check=True)
        commands.append(("sshd", ["/usr/sbin/sshd", "-D", "-e", "-f", str(config)], os.environ.copy()))
    if code:
        executable = install_code_server(data)
        # Write no password into logs or command arguments. Ignore unrelated template config.
        config = secrets / "code-server.yaml"
        config.write_text("{}\n")
        config.chmod(0o600)
        env = {**os.environ, "PASSWORD": password}
        env.pop("HASHED_PASSWORD", None)
        commands.append(("code-server", [str(executable), "--config", str(config),
            "--bind-addr", "0.0.0.0:8080", "--auth", "password", "--disable-telemetry",
            "--user-data-dir", str(data / "code-server/user-data"),
            "--extensions-dir", str(data / "code-server/extensions"), str(app)], env))
    supervise(command, commands, log_dir)


if __name__ == "__main__":
    os.umask(0o077)
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"[inferops/access] ERROR: {exc}") from exc
