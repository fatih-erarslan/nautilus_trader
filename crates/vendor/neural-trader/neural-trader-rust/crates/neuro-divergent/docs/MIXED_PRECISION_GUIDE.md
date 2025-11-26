# Mixed Precision Training (FP16) Guide

## Overview

Mixed precision training uses FP16 (half precision) for forward passes and FP32 (full precision) for backward passes, delivering:

- **1.5-2x training speedup** on modern GPUs with Tensor Cores
- **50% memory reduction** for activations and model weights
- **Same accuracy** as FP32 training with proper loss scaling

## Quick Start

```rust
use neuro_divergent::optimizations::mixed_precision::*;

// Create configuration
let config = MixedPrecisionConfig::default();
let mut trainer = MixedPrecisionTrainer::new(config);

// Initialize with model weights
let weights = vec![Array2::zeros((input_dim, hidden_dim))];
trainer.initialize_weights(weights);

// Training loop
for epoch in 0..epochs {
    let loss = trainer.train_step(
        &x_batch,
        &y_batch,
        |input_fp16, targets| compute_gradients(input_fp16, targets),
        learning_rate,
    )?;

    println!("Epoch {}: Loss = {:.6}, Scale = {}",
             epoch, loss, trainer.current_scale());
}

// Check statistics
let stats = trainer.stats();
println!("Overflow rate: {:.2}%", stats.overflow_rate * 100.0);
```

## Architecture

### Components

1. **F16 Type**: FP16 representation with proper range clamping
2. **GradScaler**: Dynamic loss scaling to prevent gradient underflow
3. **WeightManager**: Maintains master weights in FP32
4. **MixedPrecisionTrainer**: Complete training infrastructure

### How It Works

```
┌─────────────────────────────────────────────────┐
│              Mixed Precision Flow               │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. Forward Pass (FP16)                        │
│     ├─ Convert inputs to FP16                  │
│     ├─ Compute activations in FP16             │
│     └─ Calculate loss in FP16                  │
│                                                 │
│  2. Loss Scaling                               │
│     └─ loss_scaled = loss × scale (65536)      │
│                                                 │
│  3. Backward Pass (FP32)                       │
│     ├─ Compute gradients in FP32               │
│     └─ Unscale: grad = grad / scale            │
│                                                 │
│  4. Overflow Detection                         │
│     ├─ Check for NaN/Inf in gradients          │
│     ├─ If overflow: reduce scale × 0.5         │
│     └─ If stable: increase scale × 2.0         │
│                                                 │
│  5. Weight Update (FP32)                       │
│     ├─ Update master weights in FP32           │
│     └─ Sync FP16 working weights               │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Configuration

### Default Configuration

```rust
let config = MixedPrecisionConfig {
    enabled: true,
    initial_scale: 65536.0,        // 2^16
    scale_growth_factor: 2.0,       // Double when stable
    scale_backoff_factor: 0.5,      // Halve on overflow
    growth_interval: 2000,          // Steps before increase
    min_scale: 1.0,
    max_scale: 262144.0,            // 2^18
    dynamic_scaling: true,
    check_finite: true,
    master_weights: true,
};
```

### Custom Configuration

```rust
// Conservative scaling (fewer overflows)
let conservative = MixedPrecisionConfig {
    initial_scale: 32768.0,         // Lower initial scale
    growth_interval: 5000,          // Slower growth
    scale_growth_factor: 1.5,       // Smaller increases
    ..Default::default()
};

// Aggressive scaling (faster training)
let aggressive = MixedPrecisionConfig {
    initial_scale: 131072.0,        // Higher initial scale
    growth_interval: 1000,          // Faster growth
    scale_growth_factor: 2.5,       // Larger increases
    ..Default::default()
};
```

## Advanced Features

### 1. Dynamic Loss Scaling

Automatically adjusts loss scale based on gradient stability:

```rust
let mut scaler = GradScaler::new(&config);

// Training loop
loop {
    // ... compute gradients ...

    scaler.unscale(&mut gradients);
    let is_finite = scaler.check_finite_gradients(&gradients);

    let status = scaler.update(!is_finite);
    match status {
        UpdateStatus::ScaleIncreased { old, new } => {
            println!("Scale increased: {} -> {}", old, new);
        },
        UpdateStatus::ScaleDecreased { old, new } => {
            println!("Overflow detected! Scale reduced: {} -> {}", old, new);
        },
        UpdateStatus::NoUpdate => {},
    }
}
```

### 2. Master Weight Management

Keeps high-precision master weights for numerical stability:

```rust
let mut weight_manager = WeightManager::new(weights, true);

// Training loop
loop {
    // Use FP16 weights for forward pass
    let fp16_weights = weight_manager.get_fp16_weights();

    // ... compute gradients ...

    // Update master weights in FP32
    weight_manager.update_master_weights(&gradients, learning_rate);

    // FP16 weights automatically synced
}
```

### 3. FP16 Safety Checks

Verify data is safe for FP16 conversion:

```rust
use neuro_divergent::optimizations::mixed_precision::conversion;

let data = Array2::random((1000, 100), Uniform::new(-1.0, 1.0));

// Check for overflow/underflow
let (overflow_count, underflow_count) = conversion::check_fp16_safe(&data);

if overflow_count > 0 {
    println!("Warning: {} values would overflow in FP16", overflow_count);
}

if underflow_count > 0 {
    println!("Warning: {} values would underflow in FP16", underflow_count);
}

// Safe conversion
if overflow_count == 0 && underflow_count == 0 {
    let fp16_data = conversion::to_fp16(&data);
}
```

## Performance Optimization

### 1. Batch Size Scaling

Larger batches benefit more from FP16:

```rust
// Optimal batch sizes for FP16
let batch_sizes = [64, 128, 256, 512];

for &batch_size in &batch_sizes {
    let (x, y) = load_batch(batch_size);
    let loss = trainer.train_step(&x, &y, compute_grads, lr)?;
    // Larger batches = better FP16 speedup
}
```

### 2. Memory Efficiency

Monitor memory usage:

```rust
let stats = trainer.stats();

println!("Training Statistics:");
println!("  Total steps: {}", stats.total_steps);
println!("  Overflow rate: {:.2}%", stats.overflow_rate * 100.0);
println!("  Current scale: {}", stats.current_scale);
println!("  Avg gradient norm: {:.6}", stats.avg_grad_norm);
println!("  Scale adjustments: +{} / -{}",
         stats.scale_increases, stats.scale_decreases);
```

### 3. Gradient Accumulation

Combine with gradient accumulation for effective larger batches:

```rust
let accumulation_steps = 4;
let mut accumulated_grads = Vec::new();

for step in 0..accumulation_steps {
    let loss = trainer.train_step(&x_batch, &y_batch,
        |input, target| {
            let grads = compute_gradients(input, target);
            accumulated_grads.push(grads);
            grads
        },
        lr / accumulation_steps as f64
    )?;
}

// Accumulated gradients reduce memory pressure
```

## Benchmarking

### Run Benchmarks

```bash
# Full benchmark suite
cargo bench --bench mixed_precision_benchmark

# Specific benchmarks
cargo bench --bench mixed_precision_benchmark -- fp16_conversion
cargo bench --bench mixed_precision_benchmark -- training_step
cargo bench --bench mixed_precision_benchmark -- gradient_scaling
```

### Expected Results

```
FP16 Conversion:
  to_fp16/100          50.2 ns
  to_fp16/1000        483.1 ns
  to_fp16/10000      4.82 µs

Training Step Comparison:
  fp32/32            25.3 µs
  fp16_mixed/32      18.7 µs    (1.35x faster)
  fp32/128          101.2 µs
  fp16_mixed/128     56.4 µs    (1.79x faster)

Memory Efficiency:
  FP32: 100MB activations
  FP16: 50MB activations     (50% reduction)
```

## Troubleshooting

### High Overflow Rate

If overflow rate exceeds 5%:

```rust
let config = MixedPrecisionConfig {
    initial_scale: 16384.0,     // Lower initial scale
    growth_interval: 5000,       // Slower scaling
    scale_growth_factor: 1.5,    // Smaller increases
    ..Default::default()
};
```

### Convergence Issues

If model doesn't converge:

```rust
// Ensure master weights are enabled
let config = MixedPrecisionConfig {
    master_weights: true,        // Critical for stability
    check_finite: true,          // Detect issues early
    ..Default::default()
};

// Monitor gradient norms
let stats = trainer.stats();
if stats.avg_grad_norm > 10.0 {
    println!("Warning: Large gradients detected!");
    // Consider gradient clipping
}
```

### NaN/Inf Detection

Enable strict checking:

```rust
let config = MixedPrecisionConfig {
    check_finite: true,          // Always check
    dynamic_scaling: true,       // Auto-adjust scale
    ..Default::default()
};

// Manual checks
if !trainer.scaler.check_finite_gradients(&gradients) {
    println!("Non-finite gradients detected!");
    trainer.reset(); // Reset to safe state
}
```

## Best Practices

### 1. Start Conservative

```rust
// Begin with conservative settings
let config = MixedPrecisionConfig {
    initial_scale: 32768.0,
    growth_interval: 5000,
    ..Default::default()
};

// Monitor first 1000 steps
for step in 0..1000 {
    // ... train ...
    if step % 100 == 0 {
        let stats = trainer.stats();
        println!("Step {}: overflow rate = {:.2}%",
                 step, stats.overflow_rate * 100.0);
    }
}

// Adjust based on results
```

### 2. Use with Gradient Clipping

```rust
// Combine FP16 with gradient clipping for stability
fn clip_gradients(gradients: &mut [Array2<f64>], max_norm: f64) {
    let total_norm = compute_gradient_norm(gradients);
    if total_norm > max_norm {
        let scale = max_norm / total_norm;
        for grad in gradients.iter_mut() {
            grad.mapv_inplace(|x| x * scale);
        }
    }
}

let loss = trainer.train_step(&x, &y,
    |input, target| {
        let mut grads = compute_gradients(input, target);
        clip_gradients(&mut grads, 1.0);
        grads
    },
    lr
)?;
```

### 3. Monitor Long-Term Stability

```rust
// Track statistics over time
let mut overflow_history = Vec::new();
let mut loss_history = Vec::new();

for epoch in 0..epochs {
    let loss = trainer.train_step(&x, &y, compute_grads, lr)?;

    loss_history.push(loss);
    overflow_history.push(trainer.stats().overflow_rate);

    // Check for divergence
    if epoch > 100 {
        let recent_overflow = overflow_history[epoch - 10..].iter()
            .sum::<f64>() / 10.0;

        if recent_overflow > 0.1 {
            println!("Warning: High recent overflow rate!");
            // Consider reducing scale or learning rate
        }
    }
}
```

## Integration with Training Loop

Complete example:

```rust
use neuro_divergent::optimizations::mixed_precision::*;

fn train_model(
    x_train: &Array2<f64>,
    y_train: &Array1<f64>,
    epochs: usize,
    batch_size: usize,
) -> Result<Vec<f32>> {
    // Setup
    let config = MixedPrecisionConfig::default();
    let mut trainer = MixedPrecisionTrainer::new(config);

    let weights = vec![Array2::zeros((x_train.ncols(), 1))];
    trainer.initialize_weights(weights);

    let mut losses = Vec::new();
    let n_batches = (x_train.nrows() + batch_size - 1) / batch_size;

    // Training loop
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(x_train.nrows());

            let x_batch = x_train.slice(s![start..end, ..]).to_owned();
            let y_batch = y_train.slice(s![start..end]).to_owned();

            let loss = trainer.train_step(
                &x_batch,
                &y_batch,
                |input_fp16, targets| {
                    compute_gradients(input_fp16, targets)
                },
                0.001,
            )?;

            epoch_loss += loss;
        }

        let avg_loss = epoch_loss / n_batches as f32;
        losses.push(avg_loss);

        // Logging
        if epoch % 10 == 0 {
            let stats = trainer.stats();
            println!(
                "Epoch {}/{}: loss={:.6}, scale={}, overflow={:.2}%",
                epoch + 1, epochs, avg_loss,
                stats.current_scale,
                stats.overflow_rate * 100.0
            );
        }
    }

    Ok(losses)
}
```

## References

- [Mixed Precision Training (NVIDIA)](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- [Automatic Mixed Precision (PyTorch)](https://pytorch.org/docs/stable/amp.html)
- [FP16 Training (TensorFlow)](https://www.tensorflow.org/guide/mixed_precision)

## License

MIT License - see repository for details
