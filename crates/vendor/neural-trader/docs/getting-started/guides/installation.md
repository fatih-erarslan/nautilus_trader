# Installation Guide

Comprehensive installation guide for the AI News Trading Platform with Neural Forecasting capabilities.

## System Requirements

### Hardware Requirements

#### Minimum Configuration
- **CPU**: 4 cores, 2.5+ GHz
- **RAM**: 16GB
- **Storage**: 50GB SSD
- **GPU**: Optional but recommended
- **Network**: Stable internet connection

#### Recommended Configuration
- **CPU**: 8+ cores, 3.0+ GHz (Intel i7/i9 or AMD Ryzen 7/9)
- **RAM**: 32GB DDR4
- **Storage**: 100GB+ NVMe SSD
- **GPU**: NVIDIA RTX 3080+ or Tesla V100+
- **Network**: High-speed broadband

#### Production Configuration
- **CPU**: 16+ cores, 3.5+ GHz
- **RAM**: 64GB+ DDR4/DDR5
- **Storage**: 500GB+ NVMe SSD with backup
- **GPU**: NVIDIA A100, H100, or multiple RTX cards
- **Network**: Dedicated high-speed connection

### Software Requirements

#### Operating System
- **Linux**: Ubuntu 20.04+, RHEL 8+, CentOS 8+
- **macOS**: 11.0+ (Big Sur or later)
- **Windows**: Windows 10/11 with WSL2

#### Python Environment
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **Package Manager**: pip 21.0+
- **Virtual Environment**: venv, conda, or virtualenv

#### GPU Support (Optional but Recommended)
- **NVIDIA GPU**: Compute Capability 3.7+
- **CUDA**: 11.8 or 12.0+
- **cuDNN**: 8.0+
- **Driver**: 525.60.13+ (Linux), 527.41+ (Windows)

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-org/ai-news-trader.git
cd ai-news-trader

# Check out the latest stable release
git checkout $(git describe --tags --abbrev=0)
```

#### Step 2: Create Python Environment

**Using venv (Recommended):**
```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

**Using conda:**
```bash
# Create conda environment
conda create -n ai-news-trader python=3.10
conda activate ai-news-trader
```

#### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install -r requirements-mcp.txt

# Install neural forecasting dependencies
pip install neuralforecast[gpu]

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Step 4: Install Additional Tools

```bash
# Install development tools
pip install -r requirements-dev.txt

# Install optional dependencies
pip install optuna              # Hyperparameter optimization
pip install shap               # Model interpretability
pip install plotly             # Advanced plotting
pip install streamlit          # Web UI components
```

### Method 2: Docker Installation

#### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Container Toolkit (for GPU support)

#### GPU-enabled Docker Setup

```bash
# Install NVIDIA Container Toolkit (Ubuntu)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

#### Build and Run

```bash
# Build Docker image
docker build -t ai-news-trader:latest .

# Run with GPU support
docker run --gpus all -it --rm \
  -p 3000:3000 \
  -v $(pwd):/workspace \
  -e NEURAL_FORECAST_GPU=true \
  ai-news-trader:latest

# Or use Docker Compose
docker-compose up -d
```

### Method 3: Development Installation

For contributors and advanced users:

```bash
# Clone with submodules
git clone --recursive https://github.com/your-org/ai-news-trader.git
cd ai-news-trader

# Install in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run development setup
python setup.py develop
```

## GPU Setup

### NVIDIA CUDA Installation

#### Ubuntu/Debian

```bash
# Install NVIDIA driver
sudo apt update
sudo apt install nvidia-driver-525

# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA Toolkit
sudo apt-get install cuda-toolkit-11-8

# Install cuDNN
sudo apt-get install libcudnn8 libcudnn8-dev
```

#### CentOS/RHEL

```bash
# Install NVIDIA driver
sudo dnf install nvidia-driver nvidia-settings

# Add CUDA repository
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# Install CUDA
sudo dnf install cuda-toolkit-11-8
```

#### Windows

1. Download CUDA Toolkit from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
2. Run installer and follow prompts
3. Add CUDA to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`

### Verify GPU Installation

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Test PyTorch GPU support
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

## Configuration

### Environment Variables

Create `.env` file in the project root:

```bash
# System Configuration
NEURAL_FORECAST_GPU=true
NEURAL_FORECAST_DEVICE=cuda
NEURAL_FORECAST_BATCH_SIZE=32
NEURAL_FORECAST_MAX_MEMORY=8

# MCP Server Configuration
MCP_SERVER_PORT=3000
MCP_SERVER_HOST=localhost
MCP_NEURAL_ENABLED=true
MCP_GPU_ACCELERATION=true

# Claude-Flow Configuration
CLAUDE_WORKING_DIR=/path/to/ai-news-trader
CLAUDE_GPU_ENABLED=true
CLAUDE_NEURAL_FORECASTING=true

# API Keys (replace with your keys)
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Database Configuration
DATABASE_URL=sqlite:///ai_news_trader.db
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/ai-news-trader.log
```

### Configuration Files

#### Neural Forecasting Configuration

Create `config/neural_forecast.yaml`:

```yaml
models:
  nhits:
    input_size: 168
    horizon: 30
    n_freq_downsample: [168, 24, 1]
    stack_types: ["trend", "seasonality"]
    n_blocks: [1, 1]
    mlp_units: [[512, 512], [512, 512]]
    interpolation_mode: "linear"
    pooling_mode: "MaxPool1d"
    
  nbeats:
    input_size: 168
    horizon: 30
    stack_types: ["trend", "seasonality"]
    n_blocks: [1, 1]
    mlp_units: [[512, 512], [512, 512]]
    
training:
  max_epochs: 100
  learning_rate: 0.001
  batch_size: 32
  early_stopping_patience: 10
  accelerator: "gpu"
  devices: [0]
  
gpu:
  enabled: true
  memory_fraction: 0.8
  allow_growth: true
  mixed_precision: true
```

#### MCP Server Configuration

Create `config/mcp_server.json`:

```json
{
  "server": {
    "name": "ai-news-trader-gpu",
    "version": "1.0.0",
    "host": "localhost",
    "port": 3000,
    "neural_forecasting": true,
    "gpu_acceleration": true
  },
  "tools": {
    "neural_forecast": {
      "enabled": true,
      "default_horizon": 30,
      "max_symbols": 50,
      "cache_forecasts": true
    },
    "quick_analysis": {
      "neural_enhancement": true,
      "gpu_acceleration": true,
      "cache_duration": 300
    },
    "backtest": {
      "gpu_acceleration": true,
      "parallel_processing": true,
      "max_concurrent": 8
    }
  },
  "neural_models": {
    "preload": ["nhits", "nbeats"],
    "cache_size": "4GB",
    "optimization_level": "high"
  }
}
```

#### Claude-Flow Configuration

Create `.claude/config.yaml`:

```yaml
system:
  neural_forecasting:
    enabled: true
    default_model: "nhits"
    gpu_acceleration: true
    max_memory_gb: 8
    preload_models: true
    
  mcp_server:
    auto_start: true
    port: 3000
    timeout: 30
    neural_tools: true
    
  gpu:
    enabled: true
    device_id: 0
    memory_growth: true
    mixed_precision: true
    
agents:
  max_concurrent: 8
  neural_forecaster:
    enabled: true
    gpu_access: true
    memory_limit: "4GB"
    
memory:
  neural_context:
    enabled: true
    max_forecasts: 1000
    compression: true
    cache_duration: 3600
    
logging:
  level: "INFO"
  file: "logs/claude-flow.log"
  neural_debug: false
```

## Database Setup

### SQLite (Default)

SQLite is used by default and requires no additional setup.

### PostgreSQL (Recommended for Production)

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE ai_news_trader;
CREATE USER ai_trader WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE ai_news_trader TO ai_trader;
\q

# Update connection string
export DATABASE_URL="postgresql://ai_trader:your_password@localhost/ai_news_trader"
```

### Redis (Optional, for Caching)

```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test connection
redis-cli ping
```

## Verification

### Quick System Test

```bash
# Test basic functionality
python -c "
import sys
print(f'Python version: {sys.version}')

# Test imports
try:
    import torch
    import neuralforecast
    import fastmcp
    print('✓ All core packages imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')

# Test GPU
import torch
if torch.cuda.is_available():
    print(f'✓ GPU available: {torch.cuda.get_device_name(0)}')
else:
    print('! GPU not available (CPU mode)')
"
```

### Neural Forecasting Test

```bash
# Run quick neural forecast test
python -c "
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
import pandas as pd
import numpy as np

# Create test data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
data = pd.DataFrame({
    'ds': dates,
    'unique_id': 'TEST',
    'y': np.random.randn(100).cumsum()
})

# Test model
model = NHITS(input_size=30, h=7, max_epochs=1)
nf = NeuralForecast(models=[model], freq='D')

print('Testing neural forecasting...')
nf.fit(data)
forecasts = nf.predict()
print('✓ Neural forecasting test passed')
print(f'Forecast shape: {forecasts.shape}')
"
```

### MCP Server Test

```bash
# Test MCP server
python mcp_server_enhanced.py &
SERVER_PID=$!
sleep 5

# Test with curl
curl -X POST http://localhost:3000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"ping","id":1}' \
  && echo "✓ MCP server test passed" \
  || echo "✗ MCP server test failed"

# Cleanup
kill $SERVER_PID
```

### Claude-Flow Test

```bash
# Test claude-flow
./claude-flow status --check-gpu --check-models
./claude-flow neural forecast TEST --horizon 7 --quick-test
```

## Troubleshooting

### Common Issues

#### 1. CUDA Version Mismatch

```bash
# Check CUDA version
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Out of Memory Errors

```bash
# Reduce batch size
export NEURAL_FORECAST_BATCH_SIZE=16

# Or edit config file
sed -i 's/batch_size: 32/batch_size: 16/' config/neural_forecast.yaml
```

#### 3. Permission Errors

```bash
# Fix permissions
chmod +x claude-flow
sudo chown -R $USER:$USER .
```

#### 4. Port Already in Use

```bash
# Find process using port
lsof -i :3000

# Kill process or use different port
export MCP_SERVER_PORT=3001
```

### Performance Issues

#### Slow Installation

```bash
# Use pip cache
pip install --cache-dir ~/.cache/pip -r requirements-mcp.txt

# Use conda for faster dependency resolution
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Restart NVIDIA services
sudo systemctl restart nvidia-persistenced
sudo modprobe nvidia

# Check Docker GPU access (if using Docker)
docker run --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

## Uninstallation

### Remove Virtual Environment

```bash
# Deactivate environment
deactivate

# Remove environment
rm -rf venv/
# or for conda
conda env remove -n ai-news-trader
```

### Remove Docker Containers

```bash
# Stop and remove containers
docker-compose down -v
docker rmi ai-news-trader:latest

# Remove volumes
docker volume prune
```

### Clean System

```bash
# Remove configuration files
rm -rf .claude/
rm -rf config/
rm .env

# Remove logs
rm -rf logs/

# Remove cached models
rm -rf models/
rm -rf ~/.cache/neuralforecast/
```

## Next Steps

After successful installation:

1. **Read the [Quick Start Guide](quickstart.md)** for immediate usage
2. **Configure your system** with the [Configuration Guide](../configuration/system_config.md)
3. **Follow tutorials** starting with [Basic Forecasting](../tutorials/basic_forecasting.md)
4. **Explore examples** in the [Examples directory](../examples/)

## Support

If you encounter issues during installation:

1. **Check the [Troubleshooting Guide](troubleshooting.md)**
2. **Search existing [GitHub Issues](https://github.com/your-org/ai-news-trader/issues)**
3. **Create a new issue** with installation logs and system information
4. **Join our Discord** for community support

### System Information for Support

When reporting issues, please include:

```bash
# Generate system information
python -c "
import sys, platform, torch
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
" > system_info.txt

# Include GPU information
nvidia-smi >> system_info.txt

# Include pip packages
pip list >> system_info.txt
```