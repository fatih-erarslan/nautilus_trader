# SIMD Quick Start Guide

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
neuro-divergent = { version = "2.0", features = ["simd"] }
```

## Basic Usage

### 1. Matrix Operations

```rust
use neuro_divergent::optimizations::simd::matmul;

// Matrix multiplication
let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
let c = matmul::gemm(&a, &b);

// Matrix-vector multiplication
let matrix = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
let vector = vec![1.0, 2.0, 3.0];
let result = matmul::gemv(&matrix, &vector);

// Dot product
let a = vec![1.0, 2.0, 3.0, 4.0];
let b = vec![5.0, 6.0, 7.0, 8.0];
let dot = matmul::dot_product(&a, &b);
```

### 2. Activation Functions

```rust
use neuro_divergent::optimizations::simd::activations;

let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

// ReLU activation
let relu = activations::relu(&x);
// Output: [0.0, 0.0, 0.0, 1.0, 2.0]

// GELU activation
let gelu = activations::gelu(&x);

// Sigmoid activation
let sigmoid = activations::sigmoid(&x);

// Softmax activation
let softmax = activations::softmax(&x);

// Leaky ReLU with alpha=0.01
let leaky = activations::leaky_relu(&x, 0.01);
```

### 3. Loss Functions

```rust
use neuro_divergent::optimizations::simd::losses;

let predictions = vec![1.0, 2.0, 3.0, 4.0];
let targets = vec![1.1, 2.1, 2.9, 4.1];

// Mean Squared Error
let mse = losses::mse(&predictions, &targets);

// Mean Absolute Error
let mae = losses::mae(&predictions, &targets);

// Compute gradients
let mse_grad = losses::mse_gradient(&predictions, &targets);
let mae_grad = losses::mae_gradient(&predictions, &targets);

// Huber loss (delta=1.0)
let huber = losses::huber_loss(&predictions, &targets, 1.0);
```

### 4. Utility Functions

```rust
use neuro_divergent::optimizations::simd::utils;

let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

// Reductions
let sum = utils::reduce_sum(&x);      // 15.0
let max = utils::reduce_max(&x);      // 5.0
let min = utils::reduce_min(&x);      // 1.0

// Scalar operations
let scaled = utils::scalar_mul(&x, 2.0);   // [2.0, 4.0, 6.0, 8.0, 10.0]
let shifted = utils::scalar_add(&x, 10.0); // [11.0, 12.0, 13.0, 14.0, 15.0]

// Clamping
let clamped = utils::clamp(&x, 2.0, 4.0);  // [2.0, 2.0, 3.0, 4.0, 4.0]

// L2 norm
let norm = utils::norm_l2(&[3.0, 4.0]);    // 5.0
```

## CPU Feature Detection

```rust
use neuro_divergent::optimizations::simd::{detect_cpu_features, is_simd_available};

// Check available SIMD features
let features = detect_cpu_features();
println!("AVX2: {}", features.has_avx2);
println!("AVX-512: {}", features.has_avx512f);
println!("NEON: {}", features.has_neon);

// Check if any SIMD is available
if is_simd_available() {
    println!("SIMD acceleration enabled");
} else {
    println!("Using scalar fallback");
}

// Get recommended lane size
let f32_lanes = features.recommended_f32_lanes();
println!("Recommended f32 lanes: {}", f32_lanes);
```

## Neural Network Example

```rust
use neuro_divergent::optimizations::simd::{matmul, activations, losses};

fn forward_pass(
    inputs: &[Vec<f32>],
    weights_1: &[Vec<f32>],
    weights_2: &[Vec<f32>]
) -> Vec<f32> {
    // Layer 1: Linear + ReLU
    let hidden = matmul::gemm(inputs, weights_1);
    let hidden_flat: Vec<f32> = hidden.into_iter().flatten().collect();
    let activated = activations::relu(&hidden_flat);

    // Layer 2: Linear + Sigmoid
    let hidden_2: Vec<Vec<f32>> = vec![activated];
    let output = matmul::gemm(&hidden_2, weights_2);
    let output_flat: Vec<f32> = output.into_iter().flatten().collect();

    activations::sigmoid(&output_flat)
}

fn train_step(
    inputs: &[Vec<f32>],
    weights_1: &[Vec<f32>],
    weights_2: &[Vec<f32>],
    targets: &[f32]
) -> f32 {
    // Forward pass
    let predictions = forward_pass(inputs, weights_1, weights_2);

    // Compute loss
    let loss = losses::mse(&predictions, targets);

    // Compute gradients (for backprop)
    let grad = losses::mse_gradient(&predictions, targets);

    loss
}
```

## Performance Tips

### 1. Enable Native CPU Features

Compile with native CPU optimizations:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### 2. Use Appropriate Vector Sizes

Best performance with vectors that are multiples of SIMD width:

```rust
// Good: 1024 elements (divisible by 8 for AVX2)
let x = vec![0.0; 1024];

// Less efficient: 1000 elements (remainder of 8)
let y = vec![0.0; 1000];
```

### 3. Batch Operations

Process multiple vectors together when possible:

```rust
// Better: Batch matrix multiplication
let results = batches.iter()
    .map(|batch| matmul::gemm(batch, &weights))
    .collect();
```

### 4. Reuse Allocations

Pre-allocate output vectors to avoid repeated allocations:

```rust
// Good: Reuse buffer
let mut buffer = vec![0.0; 1024];
for data in datasets {
    let result = activations::relu(&data);
    buffer.copy_from_slice(&result);
    // Process buffer...
}
```

## Benchmarking

Run benchmarks to verify performance:

```bash
# All benchmarks
cargo bench --bench simd_benchmarks

# Specific operation
cargo bench --bench simd_benchmarks -- matmul

# Compare with baseline
cargo bench --bench simd_benchmarks -- --baseline scalar
```

## Testing

Verify correctness:

```bash
# Run SIMD tests
cargo test --lib simd

# Run correctness tests
cargo test --test simd_correctness

# Run with specific features
cargo test --features simd
```

## Troubleshooting

### Issue: No speedup observed

**Solutions:**
1. Compile with `--release` flag
2. Enable native CPU features: `RUSTFLAGS="-C target-cpu=native"`
3. Verify SIMD is available: `detect_cpu_features()`
4. Check vector sizes (prefer multiples of 8 for AVX2)

### Issue: Different results from scalar

**Solutions:**
1. SIMD uses fast approximations for some functions (tanh, sigmoid)
2. Floating-point operations may have different rounding
3. Use higher epsilon in comparisons: `assert!((a - b).abs() < 1e-4)`

### Issue: Compilation errors on ARM

**Solutions:**
1. ARM NEON implementations use scalar fallback (safe but slower)
2. Full NEON implementation coming in future version
3. For now, disable SIMD on ARM: `--no-default-features --features cpu`

## Feature Flags

```toml
[features]
default = ["cpu", "simd"]
cpu = []        # CPU-only (no SIMD)
simd = []       # SIMD vectorization
```

**Enable SIMD (default):**
```toml
neuro-divergent = "2.0"
```

**Disable SIMD:**
```toml
neuro-divergent = { version = "2.0", default-features = false, features = ["cpu"] }
```

## Expected Performance

| Operation | Size | AVX2 Speedup | AVX-512 Speedup |
|-----------|------|--------------|-----------------|
| GEMM | 256×256 | 2.5x | 3.5x |
| GEMV | 1024 | 2.8x | 3.8x |
| ReLU | 16384 | 2.2x | 3.0x |
| Softmax | 16384 | 2.0x | 2.8x |
| MSE | 16384 | 3.2x | 4.5x |
| Dot Product | 16384 | 3.5x | 4.8x |

## Next Steps

1. Read the [full documentation](SIMD_OPTIMIZATION.md)
2. Check [implementation summary](SIMD_IMPLEMENTATION_SUMMARY.md)
3. Run benchmarks: `cargo bench --bench simd_benchmarks`
4. Integrate into your neural network training loop

## Support

For issues or questions:
- Check [SIMD_OPTIMIZATION.md](SIMD_OPTIMIZATION.md) for detailed documentation
- Run tests: `cargo test --lib simd`
- File an issue on GitHub with benchmark results
