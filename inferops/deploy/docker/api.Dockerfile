FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e . || true
COPY inferops ./inferops
COPY scripts ./scripts
COPY configs ./configs
RUN pip install --no-cache-dir -e .
# Domain packs are plain directories; compose mounts ./examples and ./domains
# over these so a pack can be edited without rebuilding the image.
RUN mkdir -p /app/examples /app/domains /app/model_repository

EXPOSE 8080
CMD ["uvicorn", "inferops.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
