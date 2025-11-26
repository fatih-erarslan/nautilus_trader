# GPU-Optimized Trading Platform for Fly.io
FROM rapidsai/rapidsai:23.10-cuda11.8-runtime-ubuntu22.04-py3.10

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY benchmark/ ./benchmark/
COPY tests/ ./tests/
COPY memory/ ./memory/

# Copy optimization modules
COPY src/optimization/ ./src/optimization/
COPY src/trading/strategies/ ./src/trading/strategies/

# Set environment variables for GPU acceleration
ENV CUDA_VISIBLE_DEVICES=0
ENV RAPIDS_NO_INITIALIZE=1
ENV CUPY_CACHE_DIR=/tmp/.cupy

# Create necessary directories
RUN mkdir -p /app/logs /app/results /app/memory/backups

# Expose port for web interface
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the trading platform
CMD ["python", "-m", "src.main", "--gpu", "--port", "8080"]