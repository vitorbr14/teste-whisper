# Worker Serverless para rodar pipeline.py com GPU NVIDIA na RunPod.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.* \
    && pip install -r requirements.txt

ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib"

# Core da transcrição
COPY pipeline.py .

# Cliente do Google Drive
COPY drive_client.py .

# Interface entre RunPod Serverless e o pipeline
COPY handler.py .

RUN mkdir -p /workspace

WORKDIR /workspace

ENTRYPOINT ["python", "/app/handler.py"]
