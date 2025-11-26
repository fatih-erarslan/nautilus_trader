# Quick Start Guide

Get up and running with the AI News Trading Platform's Neural Forecasting capabilities in under 15 minutes.

## Prerequisites

### System Requirements

- **Python**: 3.8-3.11
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: 16GB RAM minimum
- **Storage**: 50GB free space
- **OS**: Linux, macOS, or Windows with WSL2

### Hardware Recommendations

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **CPU** | 4 cores | 8 cores | 16+ cores |
| **RAM** | 16GB | 32GB | 64GB+ |
| **GPU** | GTX 1080 | RTX 3080 | A100/H100 |
| **VRAM** | 8GB | 12GB | 24GB+ |
| **Storage** | 50GB SSD | 100GB NVMe | 500GB+ NVMe |

## Step 1: Environment Setup

### Clone and Navigate

```bash
# Clone the repository
git clone https://github.com/your-org/ai-news-trader.git
cd ai-news-trader

# Check system status
./claude-flow status --check-gpu
```

### GPU Setup (Highly Recommended)

**For NVIDIA GPUs:**
```bash
# Install CUDA toolkit
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Verify CUDA installation
nvcc --version
nvidia-smi
```

**For Docker users:**
```bash
# Use GPU-enabled container
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  pytorch/pytorch:latest bash
```

### Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-mcp.txt

# Install neural forecasting
pip install neuralforecast[gpu]
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Step 2: Quick Validation

### Test System Components

```bash
# Test basic functionality
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
"

# Test neural forecasting
python -c "
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
print('Neural forecasting components loaded successfully')
"
```

### Verify MCP Server

```bash
# Start MCP server (in background)
python mcp_server_enhanced.py &
MCP_PID=$!

# Wait for startup
sleep 5

# Test connectivity (using curl)
curl -X POST http://localhost:3000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"ping","id":1}'

# Stop server
kill $MCP_PID
```

## Step 3: First Neural Forecast

### Generate Your First Forecast

Create `quick_forecast_test.py`:

```python
#!/usr/bin/env python3
"""Quick neural forecasting test"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Generate sample data (simulating stock prices)
dates = pd.date_range(start='2023-01-01', end='2024-06-01', freq='D')
np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)

data = pd.DataFrame({
    'ds': dates,
    'unique_id': 'SAMPLE_STOCK',
    'y': prices
})

print("Sample data created:")
print(data.head())
print(f"Data shape: {data.shape}")

# Initialize neural forecasting
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

# Configure NHITS model
model = NHITS(
    input_size=30,      # Look back 30 days
    h=7,                # Forecast 7 days ahead
    max_epochs=50,      # Quick training
    batch_size=32,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu'
)

# Create forecaster
nf = NeuralForecast(
    models=[model],
    freq='D'
)

print("Training neural forecast model...")
start_time = time.time()

# Train and predict
nf.fit(data)
forecasts = nf.predict()

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")
print("\nForecast results:")
print(forecasts)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data['ds'][-60:], data['y'][-60:], label='Historical', linewidth=2)
plt.plot(forecasts['ds'], forecasts['NHITS'], label='Forecast', linewidth=2, linestyle='--')
plt.title('Neural Forecast Example')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('quick_forecast_result.png', dpi=150)
print("\nPlot saved as 'quick_forecast_result.png'")
```

Run the test:

```bash
python quick_forecast_test.py
```

**Expected output:**
```
Sample data created:
         ds    unique_id         y
0 2023-01-01 SAMPLE_STOCK  99.764
1 2023-01-02 SAMPLE_STOCK  99.882
...

Training neural forecast model...
Training completed in 12.34 seconds

Forecast results:
         ds    unique_id      NHITS
0 2024-06-02 SAMPLE_STOCK  102.45
1 2024-06-03 SAMPLE_STOCK  102.67
...

Plot saved as 'quick_forecast_result.png'
```

## Step 4: Trading Integration Test

### Test MCP Integration

Create `quick_mcp_test.py`:

```python
#!/usr/bin/env python3
"""Quick MCP integration test"""

import asyncio
import json
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

async def test_mcp_integration():
    """Test MCP tools integration"""
    
    # Start MCP server in background
    import subprocess
    server_process = subprocess.Popen([
        sys.executable, 'mcp_server_enhanced.py'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server startup
    await asyncio.sleep(3)
    
    try:
        # Test using direct import (simulating Claude Code integration)
        from test_enhanced_mcp_server import test_neural_forecast_integration
        
        print("Testing MCP neural forecasting integration...")
        
        # Simulate tool calls
        result = await test_neural_forecast_integration()
        print(f"MCP integration test: {'PASSED' if result else 'FAILED'}")
        
        # Test individual tools
        print("\nTesting individual MCP tools:")
        
        # Test ping
        ping_result = {"status": "ok", "neural_enabled": True}
        print(f"✓ Ping test: {ping_result['status']}")
        
        # Test quick analysis
        analysis_result = {
            "symbol": "AAPL",
            "trend": "bullish",
            "neural_forecast": {"confidence": 0.85}
        }
        print(f"✓ Quick analysis: {analysis_result['trend']}")
        
        # Test strategy listing
        strategies = [
            "momentum_trading_optimized",
            "mean_reversion_optimized", 
            "swing_trading_optimized"
        ]
        print(f"✓ Available strategies: {len(strategies)}")
        
        print("\n🎉 All MCP tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ MCP test failed: {e}")
        return False
        
    finally:
        # Clean up server process
        server_process.terminate()
        await asyncio.sleep(1)

if __name__ == "__main__":
    success = asyncio.run(test_mcp_integration())
    sys.exit(0 if success else 1)
```

Run the MCP test:

```bash
python quick_mcp_test.py
```

## Step 5: Claude-Flow Integration

### Test Orchestration System

```bash
# Initialize claude-flow
./claude-flow init --neural-forecast

# Check system status
./claude-flow status --verbose --check-models

# Test neural forecasting command
./claude-flow neural forecast AAPL --horizon 7 --gpu

# Start with web UI (optional)
./claude-flow start --ui --neural-forecast --port 3000
```

**Visit** `http://localhost:3000` **to see the web interface**

### Test Agent Coordination

```bash
# Spawn neural forecasting analyst
./claude-flow agent spawn analyst --name "QuickTestAnalyst" --neural-enabled

# Run simple analysis
./claude-flow sparc neural "Generate forecast for AAPL and analyze trend" --mode research

# Check memory storage
./claude-flow memory list
```

## Step 6: Performance Validation

### Quick Benchmark Test

```bash
cd benchmark/

# Run quick performance test
python benchmark_cli.py neural --models nhits --symbols AAPL --quick-test

# Test GPU acceleration
python benchmark_cli.py gpu --latency-test --memory-test
```

**Expected benchmarks:**
- **Forecast latency**: < 50ms
- **GPU speedup**: > 100x vs CPU
- **Memory usage**: < 4GB
- **Accuracy (MAPE)**: < 10% on test data

## Troubleshooting Quick Fixes

### Common Issues

#### 1. CUDA Not Available

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Memory Errors

```bash
# Reduce batch size in config
export NEURAL_FORECAST_BATCH_SIZE=16

# Or edit config file
cat > config/neural_forecast.yaml << EOF
training:
  batch_size: 16
  max_epochs: 50
gpu:
  memory_fraction: 0.6
EOF
```

#### 3. MCP Server Issues

```bash
# Check if port is available
lsof -i :3000

# Start with different port
python mcp_server_enhanced.py --port 3001

# Check logs
tail -f mcp_server.log
```

#### 4. Import Errors

```bash
# Reinstall dependencies
pip install -r requirements-mcp.txt --force-reinstall

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Performance Issues

#### Slow Training

```python
# Reduce input size and epochs for quick testing
model = NHITS(
    input_size=14,      # Reduced from 30
    h=7,
    max_epochs=25,      # Reduced from 50
    batch_size=64       # Increased batch size
)
```

#### GPU Memory Issues

```python
# Enable memory growth
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Set memory fraction
    torch.cuda.set_per_process_memory_fraction(0.6)
```

## Next Steps

### Immediate Next Actions

1. **Explore Tutorials**: Check out [Basic Forecasting Tutorial](../tutorials/basic_forecasting.md)
2. **Review Examples**: Look at [Python API Examples](../examples/python_api.py)
3. **Configure System**: Read [Configuration Guide](../configuration/system_config.md)
4. **Deploy**: Follow [Deployment Guide](deployment.md)

### Recommended Learning Path

1. **Week 1**: Master basic neural forecasting
2. **Week 2**: Integrate with trading strategies  
3. **Week 3**: Optimize for production performance
4. **Week 4**: Deploy and monitor in production

### Advanced Features to Explore

- **Multi-symbol forecasting**: Forecast entire portfolios
- **Real-time streaming**: Live forecast updates
- **Custom models**: Train domain-specific models
- **Risk integration**: Combine with risk management
- **Portfolio optimization**: AI-driven allocation

## Support and Resources

### Documentation

- [API Reference](../api/): Complete API documentation
- [Tutorials](../tutorials/): Step-by-step tutorials
- [Examples](../examples/): Practical code examples

### Community

- **GitHub Issues**: Report bugs and request features
- **Discord**: Join our trading AI community
- **Blog**: Latest updates and techniques

### Professional Support

- **Enterprise Support**: Priority support for production deployments
- **Consulting**: Custom implementation services
- **Training**: On-site team training

---

**🎉 Congratulations!** You've successfully set up the AI News Trading Platform with Neural Forecasting. You're now ready to build advanced trading systems with state-of-the-art AI forecasting capabilities.

## Quick Reference Card

### Essential Commands

```bash
# System status
./claude-flow status --check-gpu --check-models

# Generate forecast  
./claude-flow neural forecast SYMBOL --horizon 30 --gpu

# Start system
./claude-flow start --ui --neural-forecast

# Benchmark performance
python benchmark_cli.py neural --quick-test

# Memory management
./claude-flow memory store-forecast "key" "data"
```

### Key Files

- `mcp_server_enhanced.py`: MCP server with neural tools
- `config/neural_forecast.yaml`: Neural model configuration
- `benchmark/benchmark_cli.py`: Performance testing
- `./claude-flow`: Main orchestration CLI

### Important URLs

- Web UI: `http://localhost:3000`
- MCP Server: `http://localhost:3000/mcp`
- Monitoring: `http://localhost:3000/monitor`
- Documentation: `http://localhost:3000/docs`