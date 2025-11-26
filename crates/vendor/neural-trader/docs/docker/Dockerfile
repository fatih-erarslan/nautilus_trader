# Low-latency Trading APIs Docker Image
FROM python:3.11-slim

# Install system dependencies for optimal performance
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc-dev \
    linux-tools-generic \
    linux-tools-common \
    numactl \
    cpufrequtils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data

# Set Python optimizations
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONOPTIMIZE=2

# CPU affinity and priority settings will be handled by run script
# Default command
CMD ["python", "-m", "src.main"]