# A Docker-only launcher can read YAML without a host Python environment.
FROM python:3.11-slim
RUN pip install --no-cache-dir PyYAML==6.0.3 python-dotenv==1.2.1
WORKDIR /workspace
ENTRYPOINT ["python", "scripts/configure.py"]
