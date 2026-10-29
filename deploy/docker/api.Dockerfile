FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libegl1 libgles2 ffmpeg && rm -rf /var/lib/apt/lists/*

# Build from the project root. Only the engine and generic UI belong to
# this image; project configs, plugins and weights arrive through a mount.
COPY pyproject.toml ./
ARG INFERENCE_EXTRAS=cpu
ARG SERVER_EXTRAS=server,video,driver_face
ENV INFEROPS_IMAGE_RUNTIME=${INFERENCE_EXTRAS}
RUN INFERENCE_EXTRAS="$INFERENCE_EXTRAS" SERVER_EXTRAS="$SERVER_EXTRAS" python -c "import os,pathlib,tomllib; p=tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']; extras=(os.environ['SERVER_EXTRAS']+','+os.environ['INFERENCE_EXTRAS']).split(','); pathlib.Path('/tmp/requirements.txt').write_text('\n'.join(p['dependencies'] + [d for extra in extras for d in p['optional-dependencies'][extra]]))" \
    && pip install --no-cache-dir -r /tmp/requirements.txt
ARG INSTALL_CUDA=false
RUN if [ "$INSTALL_CUDA" = "true" ]; then \
    pip uninstall -y onnxruntime && \
    pip install --no-cache-dir 'onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27'; fi
COPY inferops ./inferops
COPY README.md ./
RUN pip install --no-cache-dir --no-deps .
# The generic image carries no selected project. Compose mounts the workspace's
# authoritative configs/inferops.yaml; standalone users can supply their own.
RUN mkdir -p /app/configs /app/model_repository /project \
    && printf '{}\n' > /app/configs/inferops.yaml

EXPOSE 8080
HEALTHCHECK --interval=5s --timeout=3s --start-period=120s --retries=12 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:'+os.environ.get('INFEROPS_API_PORT','8080')+'/readyz', timeout=2)"
CMD ["inferops", "serve"]
