# Getting Started Guide

## Quick Start

This guide will help you get the Cerebellar-Norse neural network system up and running in just a few minutes.

## Prerequisites

### System Requirements

**Minimum:**
- Ubuntu 22.04 LTS (or compatible Linux distribution)
- 8 CPU cores, 32GB RAM
- 100GB free disk space
- NVIDIA GPU with 8GB+ VRAM (optional but recommended)

**Recommended:**
- Ubuntu 22.04 LTS with latest kernel
- 16+ CPU cores, 128GB RAM
- 500GB NVMe SSD
- NVIDIA RTX 4090 or Tesla V100 with 24GB+ VRAM

### Software Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install build tools
sudo apt install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    cmake

# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup default stable

# Install CUDA (for GPU acceleration)
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
sudo sh cuda_12.0.0_525.60.13_linux.run --silent --toolkit

# Add CUDA to PATH
echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

## Installation

### Option 1: Using Pre-built Binaries (Recommended)

```bash
# Download latest release
wget https://github.com/nautilus_trader/cerebellar-norse/releases/latest/download/cerebellar-norse-x86_64-unknown-linux-gnu.tar.gz

# Extract binary
tar -xzf cerebellar-norse-x86_64-unknown-linux-gnu.tar.gz
sudo mv cerebellar-norse /usr/local/bin/
sudo chmod +x /usr/local/bin/cerebellar-norse

# Verify installation
cerebellar-norse --version
```

### Option 2: Building from Source

```bash
# Clone repository
git clone https://github.com/nautilus_trader/cerebellar-norse.git
cd cerebellar-norse

# Build with all features
cargo build --release --features "lightning-gpu,parallel,simd"

# Install binary
sudo cp target/release/cerebellar-norse /usr/local/bin/

# Verify installation
cerebellar-norse --version
```

### Option 3: Using Docker (Easiest)

```bash
# Pull latest image
docker pull cerebellar-norse:latest

# Run with GPU support
docker run --gpus all -p 8080:8080 -d \
    --name cerebellar-norse \
    cerebellar-norse:latest

# Check status
docker logs cerebellar-norse
```

## First Run

### 1. Create Configuration

Create a basic configuration file:

```bash
mkdir -p ~/.config/cerebellar-norse
cat > ~/.config/cerebellar-norse/config.toml << 'EOF'
[neural]
# Small network for testing
granule_size = 10000
purkinje_size = 100
golgi_size = 50
dcn_size = 10
learning_rate = 0.01
sparsity = 0.1

[performance]
device = "cpu"  # Change to "cuda" if GPU available
batch_size = 32
max_threads = 4

[server]
host = "127.0.0.1"
port = 8080
workers = 4

[logging]
level = "info"
format = "human"

[monitoring]
enabled = true
metrics_port = 9090
EOF
```

### 2. Start the Service

```bash
# Start Cerebellar-Norse
cerebellar-norse --config ~/.config/cerebellar-norse/config.toml

# Or run in background
nohup cerebellar-norse --config ~/.config/cerebellar-norse/config.toml > cerebellar.log 2>&1 &
```

### 3. Verify Operation

```bash
# Check health endpoint
curl http://localhost:8080/health

# Expected output:
# {
#   "status": "healthy",
#   "timestamp": "2024-01-01T12:00:00Z",
#   "uptime_seconds": 45,
#   "neural_metrics": {
#     "total_neurons": 10160,
#     "memory_usage_bytes": 1048576
#   }
# }
```

## Basic Usage

### 1. Process Market Data

```bash
# Send a simple market tick
curl -X POST http://localhost:8080/neural/process \
  -H "Content-Type: application/json" \
  -d '{
    "price": 100.50,
    "volume": 1000.0,
    "timestamp": 1640995200000,
    "market_id": "BTCUSD"
  }'

# Expected response:
# {
#   "trading_signals": [0.12, 0.85, 0.03],
#   "processing_time_ns": 2500,
#   "confidence": 0.87,
#   "neural_metrics": {
#     "spike_counts": {
#       "granule_layer": 1250,
#       "purkinje_layer": 87
#     }
#   }
# }
```

### 2. Monitor Performance

```bash
# Get system metrics
curl http://localhost:8080/neural/metrics

# Get Prometheus metrics
curl http://localhost:9090/metrics
```

### 3. Configure Neural Network

```bash
# Update network configuration
curl -X POST http://localhost:8080/neural/configure \
  -H "Content-Type: application/json" \
  -d '{
    "granule_size": 20000,
    "purkinje_size": 200,
    "learning_rate": 0.005,
    "sparsity": 0.05
  }'
```

## Python Integration Example

### Install Python Client

```bash
pip install cerebellar-norse-client
```

### Basic Usage

```python
import asyncio
from cerebellar_norse_client import CerebellarNorseClient

async def main():
    # Create client
    client = CerebellarNorseClient("http://localhost:8080")
    
    # Check system health
    health = await client.get_health()
    print(f"System status: {health['status']}")
    
    # Process market data
    market_data = {
        "price": 100.50,
        "volume": 1000.0,
        "timestamp": 1640995200000,
        "market_id": "BTCUSD"
    }
    
    result = await client.process_market_data(market_data)
    print(f"Trading signals: {result['trading_signals']}")
    print(f"Processing time: {result['processing_time_ns']}ns")
    print(f"Confidence: {result['confidence']}")
    
    # Get performance metrics
    metrics = await client.get_metrics()
    print(f"Average latency: {metrics['performance_metrics']['avg_latency_ns']}ns")

if __name__ == "__main__":
    asyncio.run(main())
```

## Web Dashboard

### Setup Monitoring Dashboard

```bash
# Clone dashboard configuration
git clone https://github.com/nautilus_trader/cerebellar-norse-dashboard.git
cd cerebellar-norse-dashboard

# Start monitoring stack
docker-compose up -d

# Access dashboards:
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

### Dashboard Features

- **Real-time Performance Monitoring**
  - Processing latency trends
  - Throughput metrics
  - Error rates

- **Neural Network Visualization**
  - Spike pattern displays
  - Weight distribution histograms
  - Layer activation heatmaps

- **System Health**
  - CPU and memory usage
  - GPU utilization
  - Network statistics

## Training Your First Model

### 1. Prepare Training Data

```python
import pandas as pd
import numpy as np

# Load market data
data = pd.read_csv('market_data.csv')

# Feature engineering
data['returns'] = data['price'].pct_change()
data['sma_10'] = data['price'].rolling(10).mean()
data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

# Create labels (example: predict price direction)
data['label'] = (data['returns'].shift(-1) > 0).astype(int)

# Save processed data
training_data = {
    "training_examples": []
}

for idx, row in data.dropna().iterrows():
    training_data["training_examples"].append({
        "input": {
            "price": row['price'],
            "volume": row['volume'],
            "features": [row['sma_10'], row['volume_ratio']]
        },
        "target": {
            "action": "buy" if row['label'] == 1 else "sell",
            "confidence": 0.8
        }
    })

import json
with open('training_data.json', 'w') as f:
    json.dump(training_data, f)
```

### 2. Start Training

```bash
# Upload training data and start training
curl -X POST http://localhost:8080/training/start \
  -H "Content-Type: application/json" \
  -d @training_data.json

# Monitor training progress
curl http://localhost:8080/training/status/latest
```

### 3. Training with Python

```python
import asyncio
from cerebellar_norse_client import CerebellarNorseClient

async def train_model():
    client = CerebellarNorseClient("http://localhost:8080")
    
    # Load training data
    with open('training_data.json', 'r') as f:
        training_data = json.load(f)
    
    # Configure training
    training_config = {
        "epochs": 1000,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "validation_split": 0.2
    }
    
    # Start training
    training_response = await client.start_training(
        training_data, 
        training_config
    )
    
    training_id = training_response['training_id']
    print(f"Training started with ID: {training_id}")
    
    # Monitor training progress
    while True:
        status = await client.get_training_status(training_id)
        print(f"Epoch {status['current_epoch']}/{status['total_epochs']}: "
              f"Loss = {status['metrics']['loss']:.6f}")
        
        if status['status'] == 'completed':
            print("Training completed!")
            break
        elif status['status'] == 'failed':
            print("Training failed!")
            break
        
        await asyncio.sleep(10)  # Check every 10 seconds

asyncio.run(train_model())
```

## Performance Optimization

### GPU Acceleration

If you have an NVIDIA GPU, enable CUDA acceleration:

```toml
# Update config.toml
[performance]
device = "cuda"
batch_size = 1024      # Larger batches for GPU
cuda_streams = 4       # Parallel CUDA streams
memory_pool_size = "20GB"  # GPU memory pool

# Larger network for GPU
[neural]
granule_size = 1000000  # 1M granule cells
purkinje_size = 5000    # 5K Purkinje cells
```

### Memory Optimization

```toml
[performance]
# Enable memory optimizations
zero_copy_enabled = true
memory_mapping_enabled = true
garbage_collection_threshold = 0.8

# Pre-allocate buffers
pre_allocate_buffers = true
buffer_pool_size = "16GB"
```

### Multi-Instance Setup

For high-throughput applications:

```bash
# Start multiple instances
for i in {0..3}; do
    PORT=$((8080 + i))
    GPU_ID=$i
    
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    cerebellar-norse \
        --config config.toml \
        --port $PORT \
        --instance-id $i &
done

# Use load balancer (nginx example)
upstream cerebellar_backend {
    server 127.0.0.1:8080;
    server 127.0.0.1:8081;
    server 127.0.0.1:8082;
    server 127.0.0.1:8083;
}
```

## Next Steps

### 1. Learn More

- **[API Documentation](../api/)** - Complete API reference
- **[Training Guide](../training/)** - Advanced training techniques
- **[Architecture Overview](../architecture/)** - System design details
- **[Performance Tuning](../operations/performance-tuning.md)** - Optimization strategies

### 2. Advanced Features

- **Distributed Training** - Scale across multiple GPUs/nodes
- **Custom Loss Functions** - Implement domain-specific objectives
- **Real-time Streaming** - Process live market data feeds
- **Model Ensemble** - Combine multiple neural networks

### 3. Production Deployment

- **[Deployment Guide](../operations/deployment.md)** - Production deployment
- **[Monitoring Setup](../operations/monitoring.md)** - Comprehensive monitoring
- **[Security Configuration](../operations/security.md)** - Security best practices
- **[Backup Procedures](../operations/backup-recovery.md)** - Data protection

### 4. Community and Support

- **GitHub Repository**: https://github.com/nautilus_trader/cerebellar-norse
- **Documentation**: https://docs.cerebellar-norse.ai
- **Discord Community**: https://discord.gg/cerebellar-norse
- **Stack Overflow**: Tag your questions with `cerebellar-norse`

## Troubleshooting

### Common Issues

**Service won't start:**
```bash
# Check configuration
cerebellar-norse --config config.toml --validate-only

# Check logs
journalctl -u cerebellar-norse -f

# Check port availability
netstat -tuln | grep 8080
```

**Poor performance:**
```bash
# Check system resources
htop
nvidia-smi  # For GPU

# Optimize configuration
[performance]
batch_size = 512  # Reduce if memory issues
max_threads = 8   # Match CPU cores
```

**CUDA errors:**
```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Getting Help

1. **Check the [FAQ](../knowledge-base/faq.md)** for common questions
2. **Read the [Troubleshooting Guide](../operations/troubleshooting.md)** for detailed solutions
3. **Search existing [GitHub issues](https://github.com/nautilus_trader/cerebellar-norse/issues)**
4. **Create a new issue** with system information and error logs

---

**Congratulations!** You now have Cerebellar-Norse running and can start experimenting with biologically-inspired neural trading systems. 

*For production deployments, please review the [deployment guide](../operations/deployment.md) and [security recommendations](../operations/security.md).*