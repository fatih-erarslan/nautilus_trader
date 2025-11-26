# CPU Best Practices

## Overview

This guide provides production-ready best practices for deploying the `nt-neural` crate on CPU-only infrastructure. Follow these recommendations to achieve optimal performance, reliability, and cost-efficiency.

## Table of Contents

1. [Batch Size Selection](#batch-size-selection)
2. [Preprocessing Optimization](#preprocessing-optimization)
3. [Model Selection for CPU](#model-selection-for-cpu)
4. [Memory Management](#memory-management)
5. [Production Deployment](#production-deployment)
6. [Troubleshooting](#troubleshooting)

---

## Batch Size Selection

### How to Choose Batch Size

Batch size significantly impacts latency and throughput. Follow this decision tree:

```
Is latency critical (<50ms)?
├─ YES → Use batch size 1-4
│   └─ Single predictions: batch_size = 1
│   └─ Micro-batching: batch_size = 4
└─ NO → Optimize for throughput
    ├─ Available CPU cores ≤ 4 → batch_size = 16
    ├─ Available CPU cores 5-8 → batch_size = 32
    └─ Available CPU cores ≥ 16 → batch_size = 64
```

### Batch Size Trade-offs

| Batch Size | Latency | Throughput | Memory | Best For |
|------------|---------|------------|--------|----------|
| 1 | 15-25ms | 40-60/s | 10 KB | Real-time API |
| 4 | 50-70ms | 200-300/s | 40 KB | Micro-batching |
| 16 | 150-200ms | 800-1200/s | 150 KB | Balanced workload |
| 32 | 280-380ms | 1500-2500/s | 300 KB | High throughput |
| 64 | 530-720ms | 2200-3400/s | 600 KB | Batch processing |
| 128 | 1000-1400ms | 2800-4200/s | 1.2 MB | Offline processing |

### Dynamic Batch Size Selection

Implement adaptive batching based on load:

```rust
use nt_neural::inference::BatchPredictor;

fn select_batch_size(queue_length: usize, target_latency_ms: f64) -> usize {
    if target_latency_ms < 50.0 {
        // Latency-critical
        if queue_length < 4 {
            1  // Single prediction
        } else {
            4  // Micro-batch
        }
    } else if target_latency_ms < 200.0 {
        // Balanced
        16
    } else {
        // Throughput-optimized
        if queue_length > 64 {
            64
        } else {
            32
        }
    }
}

// Usage
let batch_size = select_batch_size(pending_requests.len(), 100.0);
let predictor = BatchPredictor::new(model, device, batch_size);
```

### Benchmark Your Workload

Always benchmark with your specific data and hardware:

```rust
use criterion::{black_box, Criterion};
use nt_neural::inference::BatchPredictor;

fn benchmark_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_sizes");

    for batch_size in [1, 4, 16, 32, 64] {
        let predictor = BatchPredictor::new(model.clone(), device.clone(), batch_size);

        group.bench_with_input(
            format!("batch_{}", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let inputs = generate_inputs(batch_size);
                    let results = predictor.predict_batch(inputs).unwrap();
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}
```

### Recommendation by Use Case

| Use Case | Batch Size | Rationale |
|----------|------------|-----------|
| REST API (sync) | 1 | Minimize latency per request |
| REST API (async queue) | 16-32 | Balance latency & throughput |
| WebSocket streaming | 1-4 | Real-time updates |
| Background job processing | 64-128 | Maximum throughput |
| Offline batch scoring | 128-256 | Minimize total time |
| Edge device (Raspberry Pi) | 1-8 | Limited memory/CPU |

---

## Preprocessing Optimization

### Pre-compute Normalization Parameters

Don't recalculate statistics for every prediction:

```rust
use nt_neural::utils::preprocessing::{normalize, NormalizationParams};
use std::sync::Arc;

// ❌ BAD: Recalculate every time
fn predict_slow(data: &[f64]) -> Vec<f64> {
    let (normalized, _params) = normalize(data);
    // ... inference
}

// ✅ GOOD: Cache parameters
struct CachedPredictor {
    norm_params: Arc<NormalizationParams>,
    model: Model,
}

impl CachedPredictor {
    fn new(training_data: &[f64], model: Model) -> Self {
        let (_normalized, params) = normalize(training_data);
        Self {
            norm_params: Arc::new(params),
            model,
        }
    }

    fn predict(&self, data: &[f64]) -> Vec<f64> {
        // Use cached parameters
        let normalized: Vec<f64> = data
            .iter()
            .map(|x| (x - self.norm_params.mean) / self.norm_params.std)
            .collect();

        // ... inference
    }
}
```

**Performance Gain**: 2-3x faster preprocessing

### Batch Preprocessing

Process multiple time series together:

```rust
use rayon::prelude::*;

// ✅ GOOD: Parallel preprocessing
fn preprocess_batch(data: Vec<Vec<f64>>, params: &NormalizationParams) -> Vec<Vec<f64>> {
    data.par_iter()
        .map(|series| {
            series
                .iter()
                .map(|x| (x - params.mean) / params.std)
                .collect()
        })
        .collect()
}
```

### Minimize Feature Engineering

Only compute features that improve accuracy:

```rust
use nt_neural::utils::features::{create_lags, rolling_mean};

// ❌ BAD: Compute all possible features
fn extract_features_slow(data: &[f64]) -> Vec<Vec<f64>> {
    let mut features = vec![];
    features.extend(create_lags(data, &[1, 2, 3, 7, 14, 21, 28]));  // 7 features
    features.push(rolling_mean(data, 5));
    features.push(rolling_mean(data, 10));
    features.push(rolling_mean(data, 20));
    // ... 15+ features
    features
}

// ✅ GOOD: Only essential features (verified by ablation study)
fn extract_features_fast(data: &[f64]) -> Vec<Vec<f64>> {
    vec![
        create_lags(data, &[1, 7])[0].clone(),  // Only critical lags
        rolling_mean(data, 7),                   // Single rolling stat
    ]
}
```

**Performance Gain**: 3-5x faster feature engineering

### Reuse Buffers

Avoid repeated allocations:

```rust
struct PreprocessorWithBuffers {
    buffer: Vec<f64>,
    output: Vec<f64>,
}

impl PreprocessorWithBuffers {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            output: Vec::with_capacity(capacity),
        }
    }

    fn normalize(&mut self, data: &[f64], params: &NormalizationParams) -> &[f64] {
        self.output.clear();
        self.output.extend(
            data.iter()
                .map(|x| (x - params.mean) / params.std)
        );
        &self.output
    }
}
```

### Preprocessing Pipeline Example

Complete optimized pipeline:

```rust
use nt_neural::utils::{normalize, create_lags, rolling_mean};
use std::sync::Arc;

pub struct OptimizedPipeline {
    norm_params: Arc<NormalizationParams>,
    lag_periods: Vec<usize>,
    rolling_window: usize,
}

impl OptimizedPipeline {
    pub fn new(training_data: &[f64]) -> Self {
        let (_normalized, params) = normalize(training_data);

        Self {
            norm_params: Arc::new(params),
            lag_periods: vec![1, 7],      // Only essential lags
            rolling_window: 7,            // Single rolling stat
        }
    }

    pub fn transform(&self, data: &[f64]) -> Vec<f64> {
        // Step 1: Normalize (using cached params)
        let normalized: Vec<f64> = data
            .iter()
            .map(|x| (x - self.norm_params.mean) / self.norm_params.std)
            .collect();

        // Step 2: Create lags
        let lagged = create_lags(&normalized, &self.lag_periods);

        // Step 3: Rolling features
        let rolling = rolling_mean(&normalized, self.rolling_window);

        // Combine features
        let mut features = Vec::with_capacity(normalized.len() + lagged.len() + rolling.len());
        features.extend(normalized);
        features.extend(lagged.into_iter().flatten());
        features.extend(rolling);

        features
    }
}
```

---

## Model Selection for CPU

### CPU-Friendly Models

Ranked by CPU performance (best to worst):

| Rank | Model | CPU Latency | Complexity | Accuracy | Recommendation |
|------|-------|-------------|------------|----------|----------------|
| 1 | **TCN** | 12-18ms | Medium | High | ✅ **BEST FOR CPU** |
| 2 | **GRU** | 15-22ms | Low | Medium-High | ✅ Good balance |
| 3 | **N-BEATS** | 20-28ms | Medium-High | High | ✅ Interpretable |
| 4 | **Prophet** | 25-35ms | Low | Medium | ✅ Simple deployment |
| 5 | NHITS | 14-22ms | High | Very High | ⚠️ Better on GPU |
| 6 | LSTM-Attention | 18-28ms | High | High | ⚠️ Better on GPU |
| 7 | Transformer | 25-40ms | Very High | Very High | ⚠️ Better on GPU |
| 8 | DeepAR | 22-35ms | High | High | ⚠️ Better on GPU |

### Model Selection Guide

```rust
fn select_model_for_cpu(
    accuracy_requirement: &str,
    latency_budget_ms: f64,
    interpretability_needed: bool,
) -> &'static str {
    if interpretability_needed {
        return "N-BEATS";  // Decomposable forecasts
    }

    if latency_budget_ms < 20.0 {
        if accuracy_requirement == "high" {
            "TCN"  // Best balance
        } else {
            "GRU"  // Fastest
        }
    } else if latency_budget_ms < 30.0 {
        if accuracy_requirement == "very_high" {
            "N-BEATS"
        } else {
            "GRU"
        }
    } else {
        "Prophet"  // Simplest
    }
}
```

### Model Size Tuning

Reduce model complexity for CPU:

```rust
use nt_neural::models::{GRUModel, GRUConfig, ModelConfig};

// ❌ BAD: Large model for CPU
let config_slow = GRUConfig {
    base: ModelConfig {
        input_size: 168,
        horizon: 24,
        hidden_size: 512,  // Too large!
        num_layers: 4,     // Too deep!
        ..Default::default()
    },
    ..Default::default()
};

// ✅ GOOD: CPU-optimized model
let config_fast = GRUConfig {
    base: ModelConfig {
        input_size: 168,
        horizon: 24,
        hidden_size: 128,  // Smaller hidden size
        num_layers: 2,     // Fewer layers
        ..Default::default()
    },
    ..Default::default()
};
```

**Performance**: 3x faster with <10% accuracy loss

### Ensemble on CPU

Use simple averaging instead of weighted ensembles:

```rust
// ✅ Simple ensemble (fast)
fn simple_ensemble(predictions: Vec<Vec<f64>>) -> Vec<f64> {
    let horizon = predictions[0].len();
    let num_models = predictions.len() as f64;

    (0..horizon)
        .map(|i| {
            predictions.iter().map(|p| p[i]).sum::<f64>() / num_models
        })
        .collect()
}

// ⚠️ Weighted ensemble (slower but more accurate)
fn weighted_ensemble(predictions: Vec<Vec<f64>>, weights: &[f64]) -> Vec<f64> {
    let horizon = predictions[0].len();

    (0..horizon)
        .map(|i| {
            predictions
                .iter()
                .zip(weights)
                .map(|(p, &w)| p[i] * w)
                .sum()
        })
        .collect()
}
```

---

## Memory Management

### Memory Pooling Configuration

Enable for high-throughput scenarios:

```rust
use nt_neural::inference::{BatchPredictor, BatchConfig};

let config = BatchConfig {
    batch_size: 32,
    num_threads: num_cpus::get(),
    memory_pooling: true,   // ✅ Enable pooling
    max_queue_size: 1000,
};

let predictor = BatchPredictor::with_config(model, device, config);
```

**When to Use**:
- ✅ Processing >1000 predictions
- ✅ Long-running services
- ✅ Memory-constrained environments
- ❌ Single predictions (overhead not worth it)

### Memory Limits

Set explicit limits to prevent OOM:

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct MemoryLimitedPredictor {
    predictor: BatchPredictor<Model>,
    current_memory: Arc<AtomicUsize>,
    max_memory: usize,
}

impl MemoryLimitedPredictor {
    fn predict_if_within_limit(&self, inputs: Vec<Vec<f64>>) -> Result<Vec<PredictionResult>> {
        let estimated_memory = inputs.len() * inputs[0].len() * 8;  // bytes

        let current = self.current_memory.load(Ordering::Relaxed);
        if current + estimated_memory > self.max_memory {
            return Err(NeuralError::inference("Memory limit exceeded"));
        }

        self.current_memory.fetch_add(estimated_memory, Ordering::Relaxed);

        let result = self.predictor.predict_batch(inputs);

        self.current_memory.fetch_sub(estimated_memory, Ordering::Relaxed);

        result
    }
}
```

### Periodic Pool Cleanup

Clear tensor pool periodically:

```rust
use std::time::{Duration, Instant};

struct ManagedPredictor {
    predictor: Arc<BatchPredictor<Model>>,
    last_cleanup: Instant,
    cleanup_interval: Duration,
}

impl ManagedPredictor {
    fn predict(&mut self, inputs: Vec<Vec<f64>>) -> Result<Vec<PredictionResult>> {
        // Periodic cleanup
        if self.last_cleanup.elapsed() > self.cleanup_interval {
            self.predictor.clear_pool();
            self.last_cleanup = Instant::now();
        }

        self.predictor.predict_batch(inputs)
    }
}
```

### Memory Profiling

Monitor memory usage:

```rust
#[cfg(target_os = "linux")]
fn get_memory_usage_mb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let kb: f64 = line
                .split_whitespace()
                .nth(1)
                .unwrap()
                .parse()
                .unwrap();
            return kb / 1024.0;
        }
    }
    0.0
}

// Usage
println!("Memory before: {:.2} MB", get_memory_usage_mb());
let results = predictor.predict_batch(inputs)?;
println!("Memory after: {:.2} MB", get_memory_usage_mb());
```

---

## Production Deployment

### Docker Deployment

Optimized Dockerfile:

```dockerfile
FROM rust:1.83-slim as builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY neural-trader-rust/ ./neural-trader-rust/

# Build with optimizations
ENV RUSTFLAGS="-C target-cpu=native -C opt-level=3"
RUN cargo build --release --package nt-neural

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary
COPY --from=builder /app/target/release/libnt_neural.so /app/

# Set environment
ENV RAYON_NUM_THREADS=8
ENV RUST_LOG=info

# Resource limits
ENV MEMORY_LIMIT=2GB
ENV CPU_QUOTA=800000  # 8 cores

CMD ["./neural-trader"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-trader-cpu
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neural-trader
  template:
    metadata:
      labels:
        app: neural-trader
    spec:
      containers:
      - name: neural-trader
        image: neural-trader:latest
        resources:
          requests:
            cpu: "4"
            memory: "2Gi"
          limits:
            cpu: "8"
            memory: "4Gi"
        env:
        - name: RAYON_NUM_THREADS
          value: "8"
        - name: RUST_LOG
          value: "info"
        - name: BATCH_SIZE
          value: "32"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### AWS Lambda (Serverless)

CPU-optimized Lambda configuration:

```yaml
# serverless.yml
service: neural-trader-lambda

provider:
  name: aws
  runtime: rust
  architecture: x86_64  # or arm64 for Graviton
  memorySize: 2048      # 2 GB
  timeout: 30           # 30 seconds
  environment:
    RAYON_NUM_THREADS: 2
    RUST_LOG: warn

functions:
  predict:
    handler: rust.predict
    reservedConcurrency: 10
    events:
      - http:
          path: /predict
          method: post
```

### Load Balancing Strategy

Distribute load based on CPU utilization:

```rust
use std::sync::atomic::{AtomicU64, Ordering};

struct LoadBalancer {
    predictors: Vec<Arc<BatchPredictor<Model>>>,
    request_counts: Vec<Arc<AtomicU64>>,
}

impl LoadBalancer {
    fn select_predictor(&self) -> Arc<BatchPredictor<Model>> {
        // Round-robin with least connections
        let (idx, _) = self.request_counts
            .iter()
            .enumerate()
            .min_by_key(|(_, count)| count.load(Ordering::Relaxed))
            .unwrap();

        self.request_counts[idx].fetch_add(1, Ordering::Relaxed);
        self.predictors[idx].clone()
    }

    async fn predict(&self, inputs: Vec<Vec<f64>>) -> Result<Vec<PredictionResult>> {
        let predictor = self.select_predictor();

        let result = predictor.predict_batch_async(inputs).await;

        // Decrement counter
        let idx = Arc::ptr_eq(&predictor, &self.predictors[0]) as usize;
        self.request_counts[idx].fetch_sub(1, Ordering::Relaxed);

        result
    }
}
```

### Monitoring & Metrics

Collect performance metrics:

```rust
use prometheus::{Counter, Histogram, Registry};

lazy_static! {
    static ref PREDICTIONS_TOTAL: Counter = Counter::new(
        "predictions_total",
        "Total predictions made"
    ).unwrap();

    static ref PREDICTION_LATENCY: Histogram = Histogram::new(
        "prediction_latency_seconds",
        "Prediction latency in seconds"
    ).unwrap();

    static ref BATCH_SIZE_HISTOGRAM: Histogram = Histogram::new(
        "batch_size",
        "Batch size distribution"
    ).unwrap();
}

fn predict_with_metrics(predictor: &BatchPredictor<Model>, inputs: Vec<Vec<f64>>)
    -> Result<Vec<PredictionResult>> {
    let start = Instant::now();
    let batch_size = inputs.len();

    let results = predictor.predict_batch(inputs)?;

    let elapsed = start.elapsed().as_secs_f64();

    PREDICTIONS_TOTAL.inc_by(batch_size as f64);
    PREDICTION_LATENCY.observe(elapsed);
    BATCH_SIZE_HISTOGRAM.observe(batch_size as f64);

    Ok(results)
}
```

---

## Troubleshooting

### "My inference is too slow"

**Symptom**: Latency >50ms for single prediction

**Solutions**:

1. **Check batch size**
   ```bash
   # Verify batch size is 1 for single predictions
   echo "Current batch size: $BATCH_SIZE"
   ```

2. **Verify compiler optimizations**
   ```bash
   # Ensure release build
   cargo build --release --package nt-neural

   # Check rustflags
   echo $RUSTFLAGS
   # Should include: -C target-cpu=native -C opt-level=3
   ```

3. **Profile hot paths**
   ```bash
   # Linux
   perf record --call-graph dwarf ./target/release/neural-trader
   perf report

   # macOS
   instruments -t "Time Profiler" ./target/release/neural-trader
   ```

4. **Check model size**
   ```rust
   let params = model.num_parameters();
   println!("Model parameters: {}", params);
   // Should be <500K for CPU
   ```

5. **Try faster model**
   ```rust
   // Switch from NHITS → TCN or GRU
   let model = TCNModel::new(config)?;  // Faster on CPU
   ```

### "Memory usage is high"

**Symptom**: Memory >500 MB or growing over time

**Solutions**:

1. **Enable memory pooling**
   ```rust
   let config = BatchConfig {
       memory_pooling: true,
       ..Default::default()
   };
   ```

2. **Clear tensor pool periodically**
   ```rust
   predictor.clear_pool();
   ```

3. **Reduce batch size**
   ```rust
   // From 64 → 32 → 16
   let predictor = BatchPredictor::new(model, device, 16);
   ```

4. **Check for memory leaks**
   ```bash
   # Valgrind
   valgrind --leak-check=full ./target/release/neural-trader

   # Heaptrack (Linux)
   heaptrack ./target/release/neural-trader
   ```

5. **Use smaller model**
   ```rust
   let config = GRUConfig {
       hidden_size: 64,  // Reduce from 128/256
       num_layers: 1,    // Reduce from 2/3
       ..Default::default()
   };
   ```

### "Training doesn't converge"

**Symptom**: Loss not decreasing or NaN

**Solutions**:

1. **Check data normalization**
   ```rust
   let (normalized, params) = normalize(&data);
   println!("Mean: {}, Std: {}", params.mean, params.std);
   // Std should not be 0 or very small
   ```

2. **Reduce learning rate**
   ```rust
   let trainer = Trainer::new(
       model,
       optimizer,
       TrainerConfig {
           learning_rate: 0.0001,  // Reduce from 0.001
           ..Default::default()
       },
   );
   ```

3. **Check for outliers**
   ```rust
   use nt_neural::utils::preprocessing::remove_outliers;
   let clean_data = remove_outliers(&data, 3.0);
   ```

4. **Gradient clipping**
   ```rust
   let config = TrainerConfig {
       gradient_clip: Some(1.0),
       ..Default::default()
   };
   ```

5. **Verify data quality**
   ```rust
   // Check for NaN/Inf
   assert!(data.iter().all(|x| x.is_finite()));
   ```

### "NaN in outputs"

**Symptom**: Predictions contain NaN

**Solutions**:

1. **Check input data**
   ```rust
   if inputs.iter().any(|x| !x.is_finite()) {
       return Err(NeuralError::inference("Invalid input data"));
   }
   ```

2. **Verify normalization**
   ```rust
   // Avoid division by zero
   let std = if params.std < 1e-10 { 1.0 } else { params.std };
   let normalized = (value - params.mean) / std;
   ```

3. **Check model parameters**
   ```rust
   // Reload model from checkpoint
   let model = Model::load("model.safetensors")?;
   ```

4. **Reduce numerical instability**
   ```rust
   // Use log-softmax instead of softmax
   // Clip gradients
   // Use float64 instead of float32
   ```

### "Throughput lower than expected"

**Symptom**: <1000 predictions/sec with batch_size=32

**Solutions**:

1. **Check CPU utilization**
   ```bash
   top -H -p $(pgrep neural-trader)
   # Should see 100% utilization on multiple cores
   ```

2. **Increase thread count**
   ```rust
   let config = BatchConfig {
       num_threads: num_cpus::get(),  // Use all cores
       ..Default::default()
   };
   ```

3. **Profile with flamegraph**
   ```bash
   cargo flamegraph --bench neural_benchmarks
   ```

4. **Verify SIMD usage**
   ```bash
   objdump -d target/release/libnt_neural.so | grep -c "vmul\|vadd\|vfma"
   # Should see many SIMD instructions
   ```

5. **Enable LTO**
   ```toml
   [profile.release]
   lto = "fat"
   codegen-units = 1
   ```

---

## Quick Reference

### Performance Checklist

- [ ] Using release build (`cargo build --release`)
- [ ] Enabled `-C target-cpu=native` in rustflags
- [ ] Set `RAYON_NUM_THREADS` to physical core count
- [ ] Batch size appropriate for use case
- [ ] Memory pooling enabled for high-throughput
- [ ] Pre-computed normalization parameters
- [ ] CPU-friendly model selected (TCN/GRU)
- [ ] Monitoring latency and throughput
- [ ] Profiled hot paths with criterion/perf
- [ ] Set resource limits in production

### Common Configurations

**Low Latency (<20ms)**:
```rust
let config = BatchConfig {
    batch_size: 1,
    num_threads: 4,
    memory_pooling: false,
    ..Default::default()
};
let model = TCNModel::new(config)?;  // Fastest
```

**Balanced (100ms, 2000/s)**:
```rust
let config = BatchConfig {
    batch_size: 32,
    num_threads: 8,
    memory_pooling: true,
    ..Default::default()
};
let model = GRUModel::new(config)?;  // Good balance
```

**High Throughput (>3000/s)**:
```rust
let config = BatchConfig {
    batch_size: 64,
    num_threads: 16,
    memory_pooling: true,
    ..Default::default()
};
let model = TCNModel::new(config)?;  // Parallel-friendly
```

---

## See Also

- [CPU Optimization Guide](CPU_OPTIMIZATION_GUIDE.md) - Technical optimizations
- [CPU Performance Targets](CPU_PERFORMANCE_TARGETS.md) - Benchmarks and targets
- [Architecture Guide](ARCHITECTURE.md) - System design
- [Training Guide](TRAINING.md) - Model training workflows
