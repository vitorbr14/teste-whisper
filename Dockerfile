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

# cuBLAS/cuDNN via pip: o ctranslate2 (usado pelo faster-whisper) carrega essas
# libs em runtime, então não precisamos da imagem nvidia/cuda "runtime" inteira
# (isso sozinho economiza uns 2-3GB em relação à base cudnn-runtime).
RUN pip install --upgrade pip \
    && pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.* \
    && pip install -r requirements.txt

ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib"

# Core da transcrição
COPY pipeline.py .

# Interface entre a RunPod Serverless e o pipeline
COPY handler.py .

# Área de trabalho do worker
RUN mkdir -p /workspace
WORKDIR /workspace

# Inicia o worker Serverless e fica esperando jobs
ENTRYPOINT ["python", "/app/handler.py"]
