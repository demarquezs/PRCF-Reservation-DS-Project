# Base image with CUDA runtime
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Bogota \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

# Combine system dependencies installation in single layer
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    build-essential \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create virtual environment and upgrade core tools
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy and install dependencies (separate layer for better caching)
COPY requirements_mlops.txt .
RUN pip install --no-cache-dir numpy==1.26.4 \
    && pip install --no-cache-dir -r requirements_mlops.txt \
    && pip install --no-cache-dir xgboost==1.7.5 \
    && rm -rf /root/.cache/pip

# Copy project code
COPY . .

# Create models directory for persistence
RUN mkdir -p /app/models

# Default command
CMD ["python", "src/pipeline.py"]