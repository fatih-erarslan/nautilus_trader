# Neural Networks Documentation

**Neural Trader Rust - AI/ML Model Documentation**

This directory contains comprehensive documentation for neural network models, training, optimization, and inference in the Neural Trader Rust implementation.

---

## 📚 Quick Navigation

### 🚀 Getting Started

**Start here:**
- [Quick Start Guide](core/QUICKSTART.md) - Get up and running quickly
- [API Reference](core/API.md) - Complete neural network API
- [Models Overview](core/MODELS.md) - Available neural architectures

### 🏗️ Core Documentation

**Architecture & Design:**
- [Architecture](core/ARCHITECTURE.md) - Neural network system design
- [Neural Crate Status](core/NEURAL_CRATE_STATUS.md) - Implementation status
- [Neural Crate Enabled](core/NEURAL_CRATE_ENABLED.md) - Feature availability

### 🎓 Training

**Model Training Guides:**
- [Training Overview](training/TRAINING.md) - General training guide
- [CPU Training Guide](training/CPU_TRAINING_GUIDE.md) - CPU-based training
- [NHITS Training Guide](training/NHITS_TRAINING_GUIDE.md) - NHITS model training
- [Training Validation Summary](training/CPU_TRAINING_VALIDATION_SUMMARY.md) - Validation results

**What you'll learn:**
- Training neural models on CPU
- Hyperparameter tuning
- Training data preparation
- Model validation and testing

### ⚡ Inference

**Inference & Prediction:**
- [Inference Guide](inference/INFERENCE.md) - Running predictions
- [CPU Inference Performance](inference/CPU_INFERENCE_PERFORMANCE.md) - Performance optimization

**Topics covered:**
- Sub-10ms inference latency
- Batch inference optimization
- Real-time prediction pipelines
- Model serving strategies

### 🚀 Optimization

**Performance Optimization:**
- [Optimization Guide](optimization/CPU_OPTIMIZATION_GUIDE.md) - Complete optimization guide
- [Optimization Implementation](optimization/OPTIMIZATION_IMPLEMENTATION_GUIDE.md) - Implementation details
- [Optimization Summary](optimization/OPTIMIZATION_SUMMARY.md) - Key optimizations
- [CPU Best Practices](optimization/CPU_BEST_PRACTICES.md) - CPU optimization tips
- [Performance Targets](optimization/CPU_PERFORMANCE_TARGETS.md) - Target benchmarks
- [Optimization Analysis](optimization/CPU_OPTIMIZATION_ANALYSIS.md) - Analysis results
- [Performance Overview](optimization/PERFORMANCE.md) - Overall performance

**Optimization areas:**
- CPU vectorization (SIMD)
- Memory efficiency
- Batch processing
- Cache optimization
- Thread parallelism

### 💾 Memory Management

**Memory Optimization:**
- [CPU Memory Optimization](memory/CPU_MEMORY_OPTIMIZATION.md) - Memory efficiency techniques
- [Memory Optimization Summary](memory/MEMORY_OPTIMIZATION_SUMMARY.md) - Key optimizations

**Topics:**
- Memory pool management
- Zero-copy operations
- Memory profiling
- Allocation strategies

### 🔢 SIMD Optimizations

**CPU Vectorization:**
- [SIMD Quick Start](simd/SIMD_QUICK_START.md) - Get started with SIMD
- [CPU SIMD Optimizations](simd/CPU_SIMD_OPTIMIZATIONS.md) - SIMD implementation
- [SIMD Implementation Summary](simd/SIMD_IMPLEMENTATION_SUMMARY.md) - Implementation details
- [CPU Preprocessing Validation](simd/CPU_PREPROCESSING_VALIDATION.md) - Preprocessing with SIMD

**SIMD features:**
- AVX2/AVX-512 support
- Vectorized operations
- 2-4x speedups
- Cross-platform compatibility

### 📊 Benchmarking & Profiling

**Performance Measurement:**
- [Benchmark Quick Start](benchmarking/BENCHMARK_QUICK_START.md) - Run benchmarks
- [Benchmark Deliverables](benchmarking/BENCHMARK_DELIVERABLES.md) - Benchmark results
- [CPU Benchmark Results](benchmarking/CPU_BENCHMARK_RESULTS.md) - Detailed results
- [Profiling Summary](benchmarking/PROFILING_SUMMARY.md) - Profiling overview
- [CPU Profiling Report](benchmarking/CPU_PROFILING_REPORT.md) - Detailed profiling
- [CPU Profiling Deliverables](benchmarking/CPU_PROFILING_DELIVERABLES.md) - Profiling results

**Benchmarking tools:**
- Criterion.rs integration
- Memory profiling
- CPU profiling
- Flamegraph generation

### 🔌 Integration

**External Integrations:**
- [AgentDB Integration](integration/AGENTDB.md) - Vector database integration
- [Rust ML Ecosystem](integration/RUST_ML_ECOSYSTEM.md) - ML ecosystem overview

**Integration features:**
- AgentDB vector storage
- 150x faster searches
- Persistent memory
- Learning trajectories

### 📝 Summaries & Reviews

**Documentation & Reviews:**
- [Documentation Summary](summaries/DOCUMENTATION_SUMMARY.md) - Overall documentation status
- [CPU Documentation Summary](summaries/CPU_DOCUMENTATION_SUMMARY.md) - CPU-specific docs
- [CPU Code Review](summaries/CPU_CODE_REVIEW.md) - Code review findings

---

## 🎯 Available Neural Models

### Time Series Models
1. **NHITS** - Neural Hierarchical Interpolation for Time Series
   - Hierarchical structure
   - Multi-horizon forecasting
   - State-of-the-art accuracy

2. **LSTM** - Long Short-Term Memory
   - Recurrent architecture
   - Sequence learning
   - Temporal dependencies

3. **GRU** - Gated Recurrent Unit
   - Simplified LSTM
   - Faster training
   - Good performance

4. **TCN** - Temporal Convolutional Network
   - Causal convolutions
   - Dilated convolutions
   - Parallel training

5. **Transformer** - Attention-based model
   - Multi-head attention
   - Positional encoding
   - State-of-the-art performance

6. **N-BEATS** - Neural Basis Expansion Analysis
   - Interpretable forecasting
   - Trend & seasonality decomposition
   - No external features needed

---

## 🚀 Quick Start Examples

### Training a Model

```bash
# Train NHITS model
cargo run --example train_nhits -- \
  --data data/prices.csv \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001
```

### Running Inference

```rust
use nt_neural::{NeuralModel, InferenceEngine};

// Load trained model
let model = NeuralModel::load("models/nhits.bin")?;

// Run prediction
let predictions = model.predict(&input_data)?;
println!("Predictions: {:?}", predictions);
```

### Benchmarking

```bash
# Run CPU benchmarks
cargo bench --package nt-neural

# Run with profiling
cargo bench --package nt-neural -- --profile-time=5
```

---

## 📊 Performance Targets

### Inference Performance
- **Target:** <10ms per prediction
- **Achieved:** 3-8ms (CPU)
- **Speedup:** 5-10x vs Python

### Training Performance
- **Target:** 2000+ samples/sec
- **Achieved:** 2500-3500 samples/sec
- **Memory:** <2GB for typical models

### Memory Efficiency
- **Model Size:** 10-50MB typical
- **Inference Memory:** <100MB
- **Training Memory:** <2GB

---

## 🔬 Optimization Features

### CPU Optimizations
✅ SIMD vectorization (AVX2/AVX-512)
✅ Memory pooling
✅ Batch processing
✅ Cache-friendly layouts
✅ Thread parallelism

### Memory Optimizations
✅ Zero-copy operations
✅ Memory reuse
✅ Efficient allocation
✅ Memory profiling

### Inference Optimizations
✅ Batch inference
✅ Model quantization
✅ Operator fusion
✅ Graph optimization

---

## 🛠️ Development Workflow

### 1. Create Model Architecture

```rust
use nt_neural::models::{NHITS, ModelConfig};

let config = ModelConfig {
    input_size: 30,
    output_size: 7,
    hidden_size: 128,
    num_layers: 3,
};

let model = NHITS::new(config);
```

### 2. Train Model

```rust
use nt_neural::training::Trainer;

let trainer = Trainer::new(model, optimizer);
trainer.train(&train_data, epochs)?;
```

### 3. Evaluate Performance

```rust
let metrics = model.evaluate(&test_data)?;
println!("MAE: {:.4}", metrics.mae);
println!("RMSE: {:.4}", metrics.rmse);
```

### 4. Deploy for Inference

```rust
model.save("models/my_model.bin")?;

// Later...
let model = NeuralModel::load("models/my_model.bin")?;
let predictions = model.predict(&new_data)?;
```

---

## 📈 Benchmarking Guide

### Run Standard Benchmarks

```bash
# All benchmarks
cargo bench --package nt-neural

# Specific benchmark
cargo bench --package nt-neural inference_latency

# With profiling
cargo bench --package nt-neural -- --profile-time=10
```

### Generate Flamegraph

```bash
# Install flamegraph
cargo install flamegraph

# Generate flamegraph
cargo flamegraph --bench inference_latency
```

---

## 🔍 Profiling & Debugging

### Memory Profiling

```bash
# Using valgrind
valgrind --tool=massif target/release/train_nhits

# Using heaptrack
heaptrack target/release/train_nhits
```

### CPU Profiling

```bash
# Using perf
perf record -g target/release/train_nhits
perf report

# Using samply
samply record target/release/train_nhits
```

---

## 🧪 Testing

### Run Tests

```bash
# All neural tests
cargo test --package nt-neural

# Specific test
cargo test --package nt-neural test_nhits_training

# With output
cargo test --package nt-neural -- --nocapture
```

### Property-Based Tests

```bash
# Run property tests
cargo test --package nt-neural property_tests
```

---

## 📚 Learning Resources

### Recommended Reading Order

1. **Beginners:**
   - Start with [Quick Start](core/QUICKSTART.md)
   - Read [Models Overview](core/MODELS.md)
   - Follow [Training Guide](training/TRAINING.md)

2. **Intermediate:**
   - Study [Architecture](core/ARCHITECTURE.md)
   - Learn [CPU Training](training/CPU_TRAINING_GUIDE.md)
   - Explore [Inference Guide](inference/INFERENCE.md)

3. **Advanced:**
   - Master [Optimization Guide](optimization/CPU_OPTIMIZATION_GUIDE.md)
   - Deep dive into [SIMD Optimizations](simd/CPU_SIMD_OPTIMIZATIONS.md)
   - Review [Benchmarking](benchmarking/BENCHMARK_QUICK_START.md)

---

## 🤝 Contributing

When adding neural network documentation:

1. Place in appropriate subdirectory
2. Follow naming conventions
3. Include code examples
4. Add performance metrics
5. Update this README

---

## 📞 Support

- **Issues:** File on GitHub
- **Questions:** Contact neural team
- **Main Docs:** [/docs/README.md](../README.md)

---

**Last Updated:** 2025-11-13
**Documentation Version:** 2.0
**Neural Crate Version:** 0.1.0
