# Base image with CUDA runtime
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Bogota

# System dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    build-essential \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python 3.12 from deadsnakes PPA
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade core tools
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements_mlops.txt .
RUN pip install numpy==1.26.4 && \
    pip install -r requirements_mlops.txt && \
    pip install xgboost==1.7.5

# Copy project code
COPY . .

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create models directory for persistence
RUN mkdir -p /app/models

# Default command
CMD ["python", "src/pipeline.py"]




