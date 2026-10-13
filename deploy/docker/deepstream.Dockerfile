# DeepStream decoder and the reusable API, with projects supplied through a mount.
FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch
WORKDIR /app
RUN /opt/nvidia/deepstream/deepstream/user_additional_install.sh && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-gi gir1.2-gst-plugins-base-1.0 && \
    rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY inferops ./inferops
RUN python3 -m pip install --no-cache-dir '.[server,deepstream]'
COPY configs/platform.yaml ./configs/platform.yaml
HEALTHCHECK --interval=5s --timeout=3s --start-period=120s --retries=12 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz', timeout=2)"
CMD ["inferops", "serve"]
