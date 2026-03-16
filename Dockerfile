FROM python:3.11-slim

LABEL maintainer="Gabriel Demetrios Lafis"
LABEL description="ML Experiment Tracking Platform"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY main.py .
COPY config/ config/

RUN mkdir -p /app/data /app/artifacts

ENV MLTRACK_DB_PATH=/app/data/tracking.db
ENV MLTRACK_ARTIFACT_ROOT=/app/artifacts
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
