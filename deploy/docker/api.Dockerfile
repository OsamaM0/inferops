FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*

# Build from the project root. Only the engine and generic UI belong to
# this image; project configs, plugins and weights arrive through a mount.
COPY pyproject.toml ./
RUN python -c "import pathlib,tomllib; p=tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']; pathlib.Path('/tmp/requirements.txt').write_text('\n'.join(p['dependencies'] + p['optional-dependencies']['server'] + p['optional-dependencies']['cpu'] + p['optional-dependencies']['triton'] + p['optional-dependencies']['video']))" \
    && pip install --no-cache-dir -r /tmp/requirements.txt
ARG INSTALL_CUDA=false
RUN if [ "$INSTALL_CUDA" = "true" ]; then \
    pip uninstall -y onnxruntime && \
    pip install --no-cache-dir 'onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27'; fi
COPY inferops ./inferops
COPY README.md ./
RUN pip install --no-cache-dir --no-deps .
COPY configs/platform.yaml ./configs/platform.yaml
RUN mkdir -p /app/model_repository /project

EXPOSE 8080
HEALTHCHECK --interval=5s --timeout=3s --start-period=120s --retries=12 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz', timeout=2)"
CMD ["inferops", "serve"]
