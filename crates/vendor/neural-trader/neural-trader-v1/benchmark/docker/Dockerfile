# Multi-stage Dockerfile for AI News Trading Benchmark System
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    git \
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p \
    /app/data \
    /app/results \
    /app/logs \
    /app/monitoring_data \
    /app/pipeline_data

# Set proper permissions
RUN chmod +x /app/docker/entrypoint.sh

# Create non-root user
RUN useradd --create-home --shell /bin/bash benchmark && \
    chown -R benchmark:benchmark /app
USER benchmark

# Expose ports
EXPOSE 8000 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Entry point
ENTRYPOINT ["/app/docker/entrypoint.sh"]

# Default command
CMD ["benchmark", "--help"]


# Development stage
FROM base as development

USER root

# Install development dependencies
RUN pip install \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy \
    jupyter \
    ipdb

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    nano \
    less \
    tree \
    && rm -rf /var/lib/apt/lists/*

USER benchmark

# Development command
CMD ["python", "-m", "benchmark.cli", "status"]


# Production stage
FROM base as production

# Remove unnecessary packages to reduce image size
RUN pip uninstall -y pip setuptools

# Set production environment
ENV ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    WORKERS=4

# Production command
CMD ["python", "-m", "benchmark.cli", "start", "--production"]


# Testing stage
FROM development as testing

# Copy test configuration
COPY tests/ /app/tests/
COPY pytest.ini /app/

# Run tests as default command
CMD ["pytest", "-v", "--tb=short"]


# Monitoring stage
FROM base as monitoring

# Install monitoring tools
RUN pip install \
    prometheus-client \
    grafana-api \
    influxdb-client

# Install system monitoring tools
USER root
RUN apt-get update && apt-get install -y \
    prometheus-node-exporter \
    && rm -rf /var/lib/apt/lists/*

USER benchmark

# Expose monitoring ports
EXPOSE 9100 9090

# Monitoring command
CMD ["python", "-m", "benchmark.monitoring.dashboard"]


# CLI stage for command-line operations
FROM base as cli

# Set CLI-specific environment
ENV BENCHMARK_MODE=cli

# CLI command
CMD ["python", "-m", "benchmark.cli"]