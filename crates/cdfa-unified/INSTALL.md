# CDFA Unified Installation Guide

Complete installation guide for CDFA Unified across multiple platforms and environments.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Platform-Specific Installation](#platform-specific-installation)
- [Package Manager Installation](#package-manager-installation)
- [Build from Source](#build-from-source)
- [Docker Installation](#docker-installation)
- [Language Bindings](#language-bindings)
- [Verification](#verification)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | x86_64 or ARM64 | x86_64 with AVX2+ |
| **Memory** | 4 GB RAM | 16 GB RAM |
| **Storage** | 2 GB free space | 10 GB free space |
| **OS** | Linux, macOS, Windows | Linux (Ubuntu 20.04+) |

### Required Dependencies

#### Rust Development

```bash
# Install Rust (required for all installations)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify installation
rustc --version  # Should be >= 1.70
cargo --version
```

#### System Libraries

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    libc6-dev \
    curl \
    git
```

**Linux (CentOS/RHEL/Fedora):**
```bash
sudo dnf install -y \
    gcc \
    gcc-c++ \
    pkgconfig \
    openssl-devel \
    glibc-devel \
    curl \
    git
```

**macOS:**
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install openssl pkg-config
```

**Windows:**
```powershell
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Install Git for Windows
winget install Git.Git

# Install LLVM (for SIMD features)
winget install LLVM.LLVM
```

## 🚀 Platform-Specific Installation

### Linux

#### Quick Installation
```bash
# Add to Cargo.toml
[dependencies]
cdfa-unified = { version = "0.1", features = ["default"] }

# Or install globally
cargo install cdfa-unified
```

#### Full Performance Installation
```bash
# Install with all performance features
cargo install cdfa-unified --features "full-performance"

# Or add to Cargo.toml
[dependencies]
cdfa-unified = { version = "0.1", features = ["full-performance"] }
```

#### Hardware-Specific Optimizations

**Intel CPUs with AVX-512:**
```bash
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx512f" \
cargo install cdfa-unified --features "full-performance,avx512"
```

**AMD CPUs:**
```bash
RUSTFLAGS="-C target-cpu=native" \
cargo install cdfa-unified --features "full-performance"
```

**ARM64/Apple Silicon:**
```bash
cargo install cdfa-unified --features "full-performance" --target aarch64-unknown-linux-gnu
```

### macOS

#### Intel Macs
```bash
# Standard installation
cargo install cdfa-unified --features "default,accelerate"

# With Apple Accelerate framework
RUSTFLAGS="-C link-arg=-framework -C link-arg=Accelerate" \
cargo install cdfa-unified --features "accelerate"
```

#### Apple Silicon (M1/M2/M3)
```bash
# Native ARM64 build
cargo install cdfa-unified --features "full-performance" --target aarch64-apple-darwin

# Universal binary (Intel + ARM)
cargo install cdfa-unified --features "full-performance" --target universal-apple-darwin
```

### Windows

#### Standard Installation
```powershell
# Install via Cargo
cargo install cdfa-unified --features "default"

# Or add to Cargo.toml
[dependencies]
cdfa-unified = { version = "0.1", features = ["default"] }
```

#### With Intel MKL
```powershell
# Download Intel MKL from Intel website
# Set environment variables
$env:MKLROOT = "C:\Program Files (x86)\Intel\oneAPI\mkl\latest"

# Install with MKL support
cargo install cdfa-unified --features "mkl"
```

## 📦 Package Manager Installation

### Rust (crates.io)

```bash
# Basic installation
cargo add cdfa-unified

# With specific features
cargo add cdfa-unified --features "simd,parallel,ml"

# Latest development version
cargo add cdfa-unified --git https://github.com/tengri/nautilus-trader
```

### Python (PyPI)

```bash
# Install Python bindings
pip install cdfa-unified

# With specific extras
pip install "cdfa-unified[gpu,distributed]"

# Development version
pip install git+https://github.com/tengri/nautilus-trader.git#egg=cdfa-unified&subdirectory=crates/cdfa-unified
```

### Conda

```bash
# Install from conda-forge (when available)
conda install -c conda-forge cdfa-unified

# Or using mamba
mamba install -c conda-forge cdfa-unified
```

### vcpkg (Windows)

```bash
# Install vcpkg dependencies
vcpkg install openssl:x64-windows

# Build with vcpkg
cargo install cdfa-unified --features "default" --target x86_64-pc-windows-msvc
```

## 🔨 Build from Source

### Complete Source Build

```bash
# Clone repository
git clone https://github.com/tengri/nautilus-trader.git
cd nautilus-trader/crates/cdfa-unified

# Build with default features
./scripts/build.sh

# Build with all features
./scripts/build.sh --profile release --features "full-performance"

# Build for specific target
./scripts/build.sh --target aarch64-unknown-linux-gnu

# Build with documentation
./scripts/build.sh --docs --tests
```

### Development Build

```bash
# Development build with watch
cargo install cargo-watch
cargo watch -x "build --all-features"

# Debug build with symbols
cargo build --profile dev --features "default"

# Release build with optimizations
cargo build --release --features "full-performance"
```

### Cross-Compilation

```bash
# Install cross compilation tool
cargo install cross

# ARM64 Linux
cross build --target aarch64-unknown-linux-gnu --features "default"

# Windows from Linux
cross build --target x86_64-pc-windows-gnu --features "default"

# macOS from Linux (requires special setup)
# See: https://github.com/cross-rs/cross
```

## 🐳 Docker Installation

### Pre-built Images

```bash
# Pull latest image
docker pull ghcr.io/tengri/cdfa-unified:latest

# Pull specific version
docker pull ghcr.io/tengri/cdfa-unified:v0.1.0

# Pull development image
docker pull ghcr.io/tengri/cdfa-unified:development
```

### Build Custom Image

```bash
# Build production image
docker build -t cdfa-unified:custom \
  --build-arg FEATURES="full-performance" \
  --build-arg PROFILE="release" \
  .

# Build development image
docker build -t cdfa-unified:dev \
  --target development \
  .

# Build Python integration image
docker build -t cdfa-unified:python \
  --target python-runtime \
  .
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  cdfa-unified:
    image: ghcr.io/tengri/cdfa-unified:latest
    environment:
      - CDFA_LOG_LEVEL=info
      - CDFA_NUM_THREADS=8
    volumes:
      - ./data:/data
      - ./config:/etc/cdfa-unified
    ports:
      - "8080:8080"
```

## 🌐 Language Bindings

### Python

#### Via pip (Recommended)
```bash
pip install cdfa-unified
```

#### Build from Source
```bash
# Install maturin for Python builds
pip install maturin

# Build Python wheel
cd nautilus-trader/crates/cdfa-unified
maturin build --release --features python

# Install built wheel
pip install target/wheels/*.whl
```

#### Development Setup
```bash
# Editable install for development
maturin develop --features python

# Test Python bindings
python -c "import cdfa_unified; print('Success!')"
```

### C/C++

#### Header Generation
```bash
# Generate C headers
cargo build --features c-bindings
cbindgen --config cbindgen.toml --crate cdfa-unified --output cdfa_unified.h

# Install library and headers
sudo cp target/release/libcdfa_unified.so /usr/local/lib/
sudo cp cdfa_unified.h /usr/local/include/
sudo ldconfig
```

#### CMake Integration
```cmake
# FindCDFA.cmake
find_library(CDFA_LIBRARY cdfa_unified)
find_path(CDFA_INCLUDE_DIR cdfa_unified.h)

target_link_libraries(your_target ${CDFA_LIBRARY})
target_include_directories(your_target PRIVATE ${CDFA_INCLUDE_DIR})
```

### JavaScript/Node.js (via WebAssembly)

```bash
# Install wasm-pack
cargo install wasm-pack

# Build WebAssembly module
wasm-pack build --target nodejs --features "core,algorithms"

# Install in Node.js project
npm install ./pkg
```

## ✅ Verification

### Basic Verification

```bash
# Verify Rust installation
cargo --version

# Test basic functionality
cargo test --package cdfa-unified --features default

# Run example
cargo run --example performance_demo --features default
```

### Python Verification

```python
# Test Python bindings
import cdfa_unified as cdfa
import numpy as np

# Basic test
data = np.random.rand(100, 5)
result = cdfa.pearson_correlation(data[:, 0], data[:, 1])
print(f"Correlation: {result}")

# Verify features
print(f"Available features: {cdfa.features()}")
```

### Performance Verification

```bash
# Run benchmarks
cargo bench --features "benchmarks"

# Run validation suite
./scripts/validate.sh benchmarks

# Check SIMD availability
cargo run --example simd_test --features simd
```

### Docker Verification

```bash
# Test Docker image
docker run --rm ghcr.io/tengri/cdfa-unified:latest cdfa-demo

# Test with volume mount
docker run --rm -v $(pwd)/data:/data ghcr.io/tengri/cdfa-unified:latest
```

## ⚙️ Configuration

### Environment Configuration

```bash
# Create configuration directory
mkdir -p ~/.config/cdfa-unified

# Copy default configuration
cp examples/config.toml ~/.config/cdfa-unified/config.toml

# Edit configuration
vim ~/.config/cdfa-unified/config.toml
```

### Runtime Configuration

```bash
# Set environment variables
export CDFA_LOG_LEVEL=debug
export CDFA_NUM_THREADS=16
export CDFA_SIMD_LEVEL=avx2
export CDFA_GPU_DEVICE=0
export CDFA_CACHE_SIZE=2GB
export CDFA_CONFIG_PATH=~/.config/cdfa-unified/config.toml
```

### Performance Tuning

```toml
# ~/.config/cdfa-unified/config.toml
[performance]
threads = 16
simd_level = "avx2"
memory_pool_size = "2GB"
enable_gpu = true
gpu_device = 0

[algorithms]
correlation_method = "pearson_simd"
fusion_method = "adaptive"
cache_results = true

[logging]
level = "info"
target = "stdout"
format = "json"
```

## 🔧 Troubleshooting

### Common Issues

#### Build Failures

**Error**: `failed to run custom build command for 'openssl-sys'`
```bash
# Linux
sudo apt install libssl-dev pkg-config

# macOS
brew install openssl pkg-config
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig"

# Windows
vcpkg install openssl:x64-windows
```

**Error**: `linker error: could not find library`
```bash
# Update library path
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
sudo ldconfig

# Or install missing libraries
sudo apt install build-essential
```

#### Performance Issues

**Slow Performance**:
```bash
# Check CPU features
cat /proc/cpuinfo | grep flags

# Rebuild with native optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Enable specific features
cargo build --features "simd,parallel" --release
```

**Memory Issues**:
```bash
# Reduce thread count
export CDFA_NUM_THREADS=4

# Reduce cache size
export CDFA_CACHE_SIZE=512MB

# Use memory-efficient allocator
cargo build --features mimalloc --release
```

#### Python Integration Issues

**Import Error**:
```bash
# Check Python version
python --version  # Should be >= 3.8

# Reinstall with verbose output
pip install --force-reinstall --verbose cdfa-unified

# Check for missing dependencies
pip install numpy>=1.20
```

**Performance Issues in Python**:
```python
# Check if native extension loaded
import cdfa_unified
print(cdfa_unified.__file__)  # Should be .so/.dll, not .py

# Check for missing optimizations
print(cdfa_unified.features())
```

### Getting Help

1. **Check Documentation**: [docs.rs/cdfa-unified](https://docs.rs/cdfa-unified)
2. **Search Issues**: [GitHub Issues](https://github.com/tengri/nautilus-trader/issues)
3. **Create Issue**: Include system info, error messages, and reproduction steps
4. **Discord/Community**: Join our development community
5. **Professional Support**: Contact swarm@tengri.ai

### Debug Information

```bash
# Collect debug information
echo "=== System Information ===" > debug_info.txt
uname -a >> debug_info.txt
cat /etc/os-release >> debug_info.txt
echo "=== Rust Information ===" >> debug_info.txt
rustc --version >> debug_info.txt
cargo --version >> debug_info.txt
echo "=== CPU Information ===" >> debug_info.txt
cat /proc/cpuinfo | grep "model name" | head -1 >> debug_info.txt
cat /proc/cpuinfo | grep flags | head -1 >> debug_info.txt
echo "=== Memory Information ===" >> debug_info.txt
free -h >> debug_info.txt
echo "=== Build Log ===" >> debug_info.txt
cargo build --verbose 2>&1 >> debug_info.txt
```

---

**Need more help?** Contact us at swarm@tengri.ai or create an issue on GitHub.