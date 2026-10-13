# Original upstream MinIO binary and embedded Console, not a fork or replacement UI.
# The old official registry tag is no longer available; use the official GitHub release.
FROM python:3.11-slim AS download
ARG TARGETARCH
ARG MINIO_RELEASE=RELEASE.2025-04-22T22-12-26Z
RUN TARGETARCH="$TARGETARCH" MINIO_RELEASE="$MINIO_RELEASE" python - <<'PY'
import hashlib
import os
import pathlib
import platform
import urllib.request

arch = os.environ.get("TARGETARCH") or {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine())
if arch not in {"amd64", "arm64"}:
    raise SystemExit(f"Unsupported MinIO architecture: {arch}")
release = os.environ["MINIO_RELEASE"]
asset = f"minio.linux-{arch}.{release}"
base = f"https://github.com/minio/minio/releases/download/{release}/{asset}"
path = pathlib.Path("/minio")
with urllib.request.urlopen(base, timeout=120) as source, path.open("wb") as output:
    while chunk := source.read(1024 * 1024):
        output.write(chunk)
with urllib.request.urlopen(base + ".sha256sum", timeout=30) as response:
    expected = response.read().decode().split()[0]
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit("MinIO release SHA-256 verification failed")
path.chmod(0o755)
PY

FROM python:3.11-slim
COPY --from=download /minio /usr/local/bin/minio
EXPOSE 9000 9001
HEALTHCHECK --interval=5s --timeout=3s --start-period=15s --retries=12 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/minio/health/ready', timeout=2)"
ENTRYPOINT ["/usr/local/bin/minio"]
CMD ["server", "/data", "--console-address", ":9001"]
