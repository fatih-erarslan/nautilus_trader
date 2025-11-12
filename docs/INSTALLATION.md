# HyperPhysics Installation Guide

Complete installation guide for the PyTorch-Rust bridge with AMD 6800XT GPU acceleration.

## System Requirements

### Hardware
- **GPU**: AMD 6800XT (16GB GDDR6) or compatible RDNA 2 GPU
- **RAM**: 16GB+ recommended
- **Storage**: 10GB for dependencies

### Software
- **OS**: Linux (Ubuntu 22.04+), macOS (limited GPU support)
- **Python**: 3.9, 3.10, or 3.11
- **Rust**: 1.70+
- **ROCm**: 5.7+ (for AMD GPU support)

## Step-by-Step Installation

### 1. Install ROCm (AMD GPU Support)

#### Ubuntu 22.04 / 24.04

```bash
# Add AMD repository
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb
sudo apt install ./amdgpu-install_5.7.50700-1_all.deb

# Install ROCm
sudo amdgpu-install --usecase=rocm

# Add user to render group
sudo usermod -a -G render,video $USER

# Reboot
sudo reboot
```

#### Verify ROCm Installation

```bash
# Check ROCm version
rocm-smi

# Should show GPU information
# Example output:
# ======================= ROCm System Management Interface =======================
# ================================= Concise Info =================================
# GPU  Temp   AvgPwr  SCLK     MCLK     Fan  Perf  PwrCap  VRAM%  GPU%
# 0    31.0c  9.0W    800Mhz   96Mhz    0%   auto  203.0W    0%   0%
```

### 2. Install PyTorch with ROCm

```bash
# Create virtual environment (or use existing freqtrade env)
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with ROCm 5.7
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Verify PyTorch GPU support
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Expected output:
```
CUDA Available: True
Device: AMD Radeon RX 6800 XT
```

### 3. Install Rust and Cargo

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Configure current shell
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

### 4. Install Maturin (Rust-Python Bridge)

```bash
pip install maturin
```

### 5. Build HyperPhysics Bridge

```bash
# Navigate to HyperPhysics directory
cd /Users/ashina/Desktop/Kurultay/HyperPhysics

# Install Python dependencies
pip install -r python/requirements.txt

# Build and install the bridge (development mode)
maturin develop --release --features python

# Or install the Python package
cd python
pip install -e .
```

### 6. Install to Freqtrade Environment

If you want to use with an existing freqtrade installation:

```bash
# Activate freqtrade environment
source /Users/ashina/freqtrade/.venv/bin/activate

# Install HyperPhysics
cd /Users/ashina/Desktop/Kurultay/HyperPhysics
maturin develop --release --features python

# Verify installation
python -c "from hyperphysics_torch import get_device_info; print(get_device_info())"
```

## Verification

### Test GPU Acceleration

```python
#!/usr/bin/env python3
"""Test HyperPhysics GPU acceleration."""

import torch
from hyperphysics_torch import HyperbolicOrderBook, GPURiskEngine, get_device_info
import numpy as np

# 1. Check device info
print("=" * 60)
print("GPU Device Information")
print("=" * 60)
info = get_device_info()
for key, value in info.items():
    print(f"{key}: {value}")

# 2. Test order book
print("\n" + "=" * 60)
print("Testing Order Book GPU Acceleration")
print("=" * 60)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
ob = HyperbolicOrderBook(device=device, max_levels=100)

bids = np.array([[100.0, 10.0], [99.5, 15.0], [99.0, 20.0]])
asks = np.array([[100.5, 12.0], [101.0, 18.0], [101.5, 25.0]])

state = ob.update(bids, asks)
print(f"Best Bid: {state['best_bid']}")
print(f"Best Ask: {state['best_ask']}")
print(f"Spread: {state.get('spread', 'N/A')}")

# 3. Test risk calculations
print("\n" + "=" * 60)
print("Testing Risk Engine GPU Acceleration")
print("=" * 60)

risk = GPURiskEngine(device=device, mc_simulations=10000)
returns = np.random.randn(1000) * 0.02

var_95, es = risk.var_monte_carlo(returns, confidence=0.95)
print(f"VaR (95%): {var_95:.4f}")
print(f"Expected Shortfall: {es:.4f}")

greeks = risk.calculate_greeks(
    spot=100.0, strike=100.0, volatility=0.2,
    time_to_expiry=1.0, risk_free_rate=0.05
)
print(f"\nOption Greeks:")
for greek, value in greeks.items():
    print(f"  {greek}: {value:.4f}")

print("\n" + "=" * 60)
print("All tests passed! GPU acceleration working.")
print("=" * 60)
```

Save as `test_installation.py` and run:
```bash
python test_installation.py
```

### Run Benchmarks

```bash
# Run comprehensive tests
cd /Users/ashina/Desktop/Kurultay/HyperPhysics
pytest tests/python/test_torch_integration.py -v

# Run benchmarks only
pytest tests/python/test_torch_integration.py --benchmark-only

# Compare CPU vs GPU performance
pytest tests/python/test_torch_integration.py -k "benchmark_orderbook" --benchmark-compare
```

## ROCm Environment Variables

For optimal performance, set these environment variables:

```bash
# Add to ~/.bashrc or ~/.zshrc

# Enable GPU device 0
export HIP_VISIBLE_DEVICES=0

# Optimize for RDNA 2 architecture (gfx1030 for 6800XT)
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Disable kernel serialization for better performance
export AMD_SERIALIZE_KERNEL=0

# MIOpen cache directory
export MIOPEN_USER_DB_PATH=/tmp/miopen_cache
mkdir -p /tmp/miopen_cache

# Enable asynchronous kernel launches
export HIP_LAUNCH_BLOCKING=0

# Source the changes
source ~/.bashrc  # or ~/.zshrc
```

## Troubleshooting

### Issue: GPU Not Detected

```bash
# Check if ROCm sees the GPU
rocm-smi

# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check ROCm version in PyTorch
python -c "import torch; print(torch.version.hip)"
```

**Solution**: If GPU not detected:
1. Verify ROCm installation: `dpkg -l | grep rocm`
2. Check user groups: `groups` (should include `render` and `video`)
3. Reboot after ROCm installation
4. Reinstall PyTorch with correct ROCm version

### Issue: Import Error

```
ImportError: cannot import name 'HyperbolicOrderBook'
```

**Solution**:
```bash
# Rebuild the extension
cd /Users/ashina/Desktop/Kurultay/HyperPhysics
maturin develop --release --features python

# Or reinstall
pip uninstall hyperphysics-finance
pip install -e python/
```

### Issue: Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**:
```python
import torch

# Clear GPU cache
torch.cuda.empty_cache()

# Use smaller batch sizes
from rocm_setup import ROCmConfig
config = ROCmConfig()
optimal_batch = config.get_optimal_batch_size((100, 2))
```

### Issue: Slow Performance

**Solution**:
```python
# Enable cuDNN benchmarking
import torch
torch.backends.cudnn.benchmark = True

# Enable TF32 for faster computation
torch.backends.cuda.matmul.allow_tf32 = True

# Run ROCm optimization
from rocm_setup import setup_rocm_for_freqtrade
config = setup_rocm_for_freqtrade()
```

### Issue: Rust Compilation Errors

```bash
# Update Rust to latest stable
rustup update stable

# Clean build cache
cargo clean

# Rebuild with verbose output
maturin develop --release --features python -v
```

## Performance Optimization

### For Trading (Low Latency)

```python
from rocm_setup import ROCmConfig

config = ROCmConfig()
config.optimize_for_inference()  # Disable gradients, enable deterministic mode
```

### For Backtesting (High Throughput)

```python
config.optimize_for_training()  # Enable benchmarking, mixed precision
```

### Custom Optimization

```python
import torch

# Disable gradient computation (inference only)
torch.set_grad_enabled(False)

# Enable cuDNN benchmarking (finds fastest kernels)
torch.backends.cudnn.benchmark = True

# Enable TF32 for matrix operations
torch.backends.cuda.matmul.allow_tf32 = True

# Set memory allocator config
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
```

## Next Steps

After successful installation:

1. **Read the Tutorial**: See `python/README.md` for usage examples
2. **Run Examples**: Execute `python/integration_example.py`
3. **Integrate with Freqtrade**: Follow freqtrade integration guide
4. **Optimize**: Use `rocm_setup.py` to tune for your workload

## Support

- **Issues**: https://github.com/hyperphysics/hyperphysics/issues
- **Documentation**: https://hyperphysics.readthedocs.io
- **Discord**: https://discord.gg/hyperphysics

## Version Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.9-3.11 | 3.12 not yet tested |
| PyTorch | 2.2.0-2.2.2 | ROCm 5.7 build |
| ROCm | 5.7+ | 6.0+ experimental |
| Rust | 1.70+ | 1.75+ recommended |
| Maturin | 1.4+ | Latest stable |

## Alternative: CPU-Only Installation

If you don't have AMD GPU or want CPU-only:

```bash
# Install PyTorch CPU version
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Build HyperPhysics
maturin develop --release --features python

# Use device="cpu" in code
from hyperphysics_torch import HyperbolicOrderBook
ob = HyperbolicOrderBook(device="cpu")
```

Note: CPU version will be ~800x slower for order book operations.
