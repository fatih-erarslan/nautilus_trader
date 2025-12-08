# CDFA Unified

[![CI/CD](https://github.com/tengri/nautilus-trader/actions/workflows/ci.yml/badge.svg)](https://github.com/tengri/nautilus-trader/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/tengri/nautilus-trader/branch/main/graph/badge.svg)](https://codecov.io/gh/tengri/nautilus-trader)
[![Crates.io](https://img.shields.io/crates/v/cdfa-unified.svg)](https://crates.io/crates/cdfa-unified)
[![Documentation](https://docs.rs/cdfa-unified/badge.svg)](https://docs.rs/cdfa-unified)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

Unified Cross-Domain Feature Alignment (CDFA) library consolidating 15+ specialized crates into a single, high-performance, comprehensive solution for financial data analysis, pattern detection, and machine learning.

## 🚀 Features

- **🏃‍♂️ High Performance**: SIMD optimizations, parallel processing, GPU acceleration
- **🔬 Mathematical Accuracy**: >99.99% compatibility with Python implementations  
- **🧩 Modular Design**: Use only the features you need
- **⚡ Multiple Backends**: CPU, GPU, distributed computing
- **🌐 Language Bindings**: Python, C/C++ FFI support
- **⏱️ Real-time Processing**: Sub-microsecond performance for critical operations
- **🧠 Advanced ML**: Neural networks, classical ML, ensemble methods
- **📊 Comprehensive Analytics**: 50+ diversity metrics, fusion algorithms, pattern detectors

## 📦 Quick Installation

### Rust

```toml
[dependencies]
cdfa-unified = { version = "0.1", features = ["full-performance"] }
```

### Python

```bash
pip install cdfa-unified
```

### Docker

```bash
docker pull ghcr.io/tengri/cdfa-unified:latest
```

## 🔧 Quick Start

### Basic Usage

```rust
use cdfa_unified::prelude::*;
use ndarray::array;

// Basic diversity and fusion analysis
let data = array![
    [0.8, 0.6, 0.9, 0.3, 0.7],
    [0.7, 0.8, 0.6, 0.4, 0.9],
    [0.9, 0.5, 0.8, 0.5, 0.6]
];

// Calculate diversity metrics
let correlation = pearson_correlation(&data.row(0), &data.row(1))?;
let kendall = kendall_tau(&data.row(0), &data.row(1))?;

// Perform fusion
let fused = CdfaFusion::fuse(&data.view(), FusionMethod::Average, None)?;

println!("Correlation: {:.3}, Kendall: {:.3}", correlation, kendall);
println!("Fused result: {:?}", fused);
```

### Advanced Pattern Detection

```rust
use cdfa_unified::detectors::*;

// Fibonacci pattern detection
let detector = FibonacciPatternDetector::new()?;
let patterns = detector.detect_patterns(&price_data)?;

// Black swan event detection
let swan_detector = BlackSwanDetector::with_threshold(3.0)?;
let extreme_events = swan_detector.analyze(&market_data)?;

// Antifragility analysis
let antifragility = AntifragilityAnalyzer::new();
let metrics = antifragility.analyze(&volatility_data)?;
```

### Machine Learning Integration

```rust
use cdfa_unified::ml::*;

// Neural pattern detection
let neural_detector = NeuralPatternDetector::new()?;
let patterns = neural_detector.detect_patterns(&time_series)?;

// Classical ML ensemble
let ensemble = EnsembleDetector::new()
    .add_detector(Box::new(SvmDetector::new()))
    .add_detector(Box::new(RandomForestDetector::new()));
    
let predictions = ensemble.predict(&features)?;
```

### GPU Acceleration

```rust
use cdfa_unified::gpu::*;

// GPU-accelerated processing
let gpu_context = GpuContext::new()?;
let gpu_detector = GpuFibonacciDetector::new(&gpu_context)?;
let patterns = gpu_detector.detect_realtime(&price_stream)?;
```

### Python Integration

```python
import cdfa_unified as cdfa
import numpy as np

# Load data
data = np.random.rand(1000, 5)

# Calculate metrics
correlation = cdfa.pearson_correlation(data[:, 0], data[:, 1])
fusion_result = cdfa.fuse_scores(data, method="rank_based")

# Pattern detection
detector = cdfa.FibonacciDetector()
patterns = detector.detect(price_data)
```

## 🏗️ Build System

### Development Build

```bash
# Quick development build
./scripts/build.sh --profile debug --features "core,algorithms"

# Full performance build
./scripts/build.sh --profile release --features "full-performance"

# Build with documentation
./scripts/build.sh --docs --tests
```

### Advanced Build Options

```bash
# Cross-compilation
./scripts/build.sh --target aarch64-unknown-linux-gnu

# Python bindings
./scripts/build.sh --python --features "python"

# Docker containerization
./scripts/build.sh --docker

# Complete validation suite
./scripts/validate.sh --features "full-performance" --benchmarks
```

## 🚀 Deployment

### Local Deployment

```bash
./scripts/deploy.sh local
```

### Production Deployment

```bash
# Docker deployment
./scripts/deploy.sh docker --version 1.0.0

# Kubernetes deployment  
./scripts/deploy.sh kubernetes --version 1.0.0

# Package registry deployment
./scripts/deploy.sh crates --version 1.0.0
./scripts/deploy.sh pypi --version 1.0.0
```

## 🔧 Configuration

### Cargo Features

| Feature | Description | Performance Impact |
|---------|-------------|-------------------|
| `default` | Core + algorithms + SIMD + parallel | ⭐⭐⭐ |
| `full-performance` | All performance features | ⭐⭐⭐⭐⭐ |
| `minimal` | Core functionality only | ⭐ |
| `simd` | SIMD optimizations | ⭐⭐⭐⭐ |
| `parallel` | Multi-threading support | ⭐⭐⭐⭐ |
| `gpu` | GPU acceleration | ⭐⭐⭐⭐⭐ |
| `ml` | Machine learning features | ⭐⭐⭐ |
| `python` | Python bindings | ⭐⭐ |
| `distributed` | Distributed computing | ⭐⭐⭐⭐ |

### Environment Variables

```bash
export CDFA_LOG_LEVEL=info          # Logging level
export CDFA_NUM_THREADS=8           # Thread count for parallel processing
export CDFA_SIMD_LEVEL=avx2         # SIMD instruction set
export CDFA_GPU_DEVICE=0            # GPU device index
export CDFA_CACHE_SIZE=1GB          # Memory cache size
```

## 📊 Performance Benchmarks

### Core Operations

| Operation | Rust (μs) | Python (μs) | Speedup |
|-----------|-----------|-------------|---------|
| Pearson Correlation | 0.8 | 45.2 | 56.5x |
| Kendall Tau | 1.2 | 89.4 | 74.5x |
| Score Fusion | 0.6 | 32.1 | 53.5x |
| Fibonacci Detection | 2.4 | 156.8 | 65.3x |
| Black Swan Analysis | 1.8 | 94.2 | 52.3x |

### SIMD Optimizations

| Feature | Scalar (μs) | SIMD (μs) | Speedup |
|---------|-------------|-----------|---------|
| Vector Operations | 12.4 | 3.1 | 4.0x |
| Matrix Multiplication | 45.6 | 8.9 | 5.1x |
| Signal Processing | 23.8 | 4.2 | 5.7x |

### Memory Usage

| Configuration | Memory (MB) | Performance |
|---------------|-------------|-------------|
| Minimal | 8.2 | ⭐⭐ |
| Default | 24.6 | ⭐⭐⭐⭐ |
| Full Performance | 78.4 | ⭐⭐⭐⭐⭐ |

## 🧪 Testing & Validation

### Automated Testing

```bash
# Run complete test suite
cargo test --all-features

# Run with validation
./scripts/validate.sh all

# Performance regression testing
./scripts/validate.sh benchmarks --benchmark-threshold 1000

# Memory leak detection
./scripts/validate.sh memory
```

### Manual Testing

```bash
# Example execution
cargo run --example performance_demo --features full-performance

# Python integration test
python examples/python/integration_test.py

# GPU acceleration test
cargo run --example gpu_demo --features gpu
```

## 📚 Documentation

- **[API Documentation](https://docs.rs/cdfa-unified)** - Complete API reference
- **[User Guide](docs/user-guide.md)** - Comprehensive usage guide
- **[Performance Guide](docs/performance.md)** - Optimization strategies
- **[Migration Guide](docs/migration.md)** - Migrating from individual crates
- **[Examples](examples/)** - Complete working examples

## 🏗️ Architecture

### Unified Crate Structure

```
cdfa-unified/
├── src/
│   ├── core/           # Core diversity and fusion algorithms
│   ├── algorithms/     # Signal processing and statistical algorithms  
│   ├── detectors/      # Pattern detection (Fibonacci, Black Swan, etc.)
│   ├── ml/            # Machine learning integration
│   ├── simd/          # SIMD optimizations
│   ├── parallel/      # Parallel processing
│   ├── gpu/           # GPU acceleration
│   ├── integration/   # External system integrations
│   ├── ffi/           # Foreign function interfaces
│   └── unified.rs     # Unified high-level API
├── benches/           # Performance benchmarks
├── examples/          # Usage examples
├── scripts/           # Build and deployment automation
└── tests/             # Integration tests
```

### Consolidated Functionality

This crate unifies the functionality from:

- **cdfa-core** → `src/core/`
- **cdfa-algorithms** → `src/algorithms/`
- **cdfa-parallel** → `src/parallel/`
- **cdfa-simd** → `src/simd/`
- **cdfa-ml** → `src/ml/`
- **cdfa-advanced-detectors** → `src/detectors/`
- **cdfa-fibonacci-pattern-detector** → `src/detectors/fibonacci/`
- **cdfa-black-swan-detector** → `src/detectors/black_swan/`
- **cdfa-antifragility-analyzer** → `src/detectors/antifragility/`
- **cdfa-panarchy-analyzer** → `src/detectors/panarchy/`
- **cdfa-soc-analyzer** → `src/detectors/soc/`
- **cdfa-torchscript-fusion** → `src/ml/torch/`
- **cdfa-stdp-optimizer** → `src/ml/stdp/`
- **cdfa-examples** → `examples/`
- **cdfa-ffi** → `src/ffi/`

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/tengri/nautilus-trader.git
cd nautilus-trader/crates/cdfa-unified

# Install development dependencies
cargo install cargo-watch cargo-audit cargo-llvm-cov

# Run development server
cargo watch -x "test --all-features"

# Run full validation
./scripts/validate.sh all
```

## 📄 License

This project is licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE))
* MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## 🙏 Acknowledgments

- **TENGRI Trading Team** - Core development and financial domain expertise
- **Rust Community** - Excellent ecosystem and tooling
- **NumPy/SciPy** - Mathematical foundation and API inspiration
- **OpenBLAS/Intel MKL** - High-performance linear algebra backends

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/tengri/nautilus-trader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/tengri/nautilus-trader/discussions)
- **Documentation**: [docs.rs/cdfa-unified](https://docs.rs/cdfa-unified)
- **Email**: swarm@tengri.ai

---

**Built with ❤️ by the TENGRI Trading Team**