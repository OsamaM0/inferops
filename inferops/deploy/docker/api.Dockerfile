FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e . || true
COPY inferops ./inferops
COPY scripts ./scripts
RUN pip install --no-cache-dir -e .

EXPOSE 8080
CMD ["uvicorn", "inferops.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
