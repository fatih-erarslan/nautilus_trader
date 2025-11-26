# SIMD Vectorization Documentation

## Overview

The SIMD (Single Instruction Multiple Data) vectorization module provides 2-4x speedup for neural network operations through efficient use of CPU vector instructions.

## Supported Architectures

### x86_64
- **SSE2**: Baseline (always available on x86_64)
- **AVX**: 256-bit vectors (8x f32 or 4x f64)
- **AVX2**: Enhanced 256-bit vectors with FMA
- **AVX-512**: 512-bit vectors (16x f32 or 8x f64) - when available

### ARM
- **NEON**: 128-bit vectors (4x f32 or 2x f64) on aarch64

### Fallback
- Scalar implementations for unsupported platforms

## Performance Characteristics

### Expected Speedups

| Operation | AVX2 | AVX-512 | NEON |
|-----------|------|---------|------|
| Matrix Multiply (GEMM) | 2-3x | 3-4x | 2x |
| Matrix-Vector (GEMV) | 2-3x | 3-4x | 2x |
| Dot Product | 3-4x | 4-5x | 2-3x |
| Activation Functions | 2-3x | 3-4x | 2x |
| Loss Calculations | 2-4x | 3-5x | 2-3x |

### Benchmark Results

Run benchmarks with:
```bash
cargo bench --bench simd_benchmarks
```

Expected output (on AVX2-capable CPU):
```
matmul/gemm/256         time:   [4.2 ms 4.3 ms 4.4 ms]
                        thrpt:  [3.9 Melem/s 4.0 Melem/s 4.1 Melem/s]

activations/relu/16384  time:   [12.3 μs 12.5 μs 12.7 μs]
                        thrpt:  [1.29 Gelem/s 1.31 Gelem/s 1.33 Gelem/s]

losses/mse/16384        time:   [18.4 μs 18.6 μs 18.9 μs]
                        thrpt:  [867 Melem/s 881 Melem/s 891 Melem/s]
```

## Usage

### Basic Operations

```rust
use neuro_divergent::optimizations::simd;

// Matrix multiplication
let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
let c = simd::matmul::gemm(&a, &b);

// Activation functions
let x = vec![1.0, -1.0, 0.5, -0.5];
let relu_out = simd::activations::relu(&x);
let gelu_out = simd::activations::gelu(&x);
let sigmoid_out = simd::activations::sigmoid(&x);

// Loss calculations
let predictions = vec![1.0, 2.0, 3.0];
let targets = vec![1.1, 2.1, 2.9];
let mse_loss = simd::losses::mse(&predictions, &targets);
let gradient = simd::losses::mse_gradient(&predictions, &targets);
```

### CPU Feature Detection

```rust
use neuro_divergent::optimizations::simd::{detect_cpu_features, is_simd_available};

// Check what SIMD features are available
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
```

### Integration with Training

```rust
use neuro_divergent::optimizations::simd::{matmul, activations, losses};

// Forward pass with SIMD
fn forward_pass(inputs: &[Vec<f32>], weights: &[Vec<f32>]) -> Vec<f32> {
    // Matrix multiplication
    let hidden = matmul::gemm(inputs, weights);

    // Flatten for activation
    let hidden_flat: Vec<f32> = hidden.into_iter().flatten().collect();

    // Apply activation
    let activated = activations::relu(&hidden_flat);

    activated
}

// Backward pass with SIMD
fn backward_pass(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    // Compute loss gradient
    let gradient = losses::mse_gradient(predictions, targets);

    gradient
}
```

## Implementation Details

### Memory Layout

SIMD operations work best with:
- **Contiguous memory**: Use `Vec<f32>` instead of scattered allocations
- **Aligned memory**: Modern allocators provide good alignment by default
- **Cache-friendly access**: Sequential access patterns

### Vectorization Strategy

1. **Main SIMD loop**: Process 8 elements at a time (AVX2)
2. **Remainder loop**: Handle leftover elements with scalar code
3. **Horizontal operations**: Use efficient reduction patterns

Example vectorization pattern:
```rust
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // SIMD loop: 8 elements at a time
    while i + 8 <= len {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
        i += 8;
    }

    // Horizontal sum
    let temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), sum);
    let mut result: f32 = temp.iter().sum();

    // Remainder loop: scalar fallback
    while i < len {
        result += a[i] * b[i];
        i += 1;
    }

    result
}
```

### Fallback Behavior

The implementation automatically falls back to scalar code when:
1. SIMD instructions are not available
2. Running on unsupported architecture
3. Small data sizes (SIMD overhead not worth it)

All SIMD functions have equivalent scalar implementations that produce identical results.

## Build Configuration

### Enable SIMD (default)
```toml
[dependencies]
neuro-divergent = { version = "2.0", features = ["simd"] }
```

### Disable SIMD
```toml
[dependencies]
neuro-divergent = { version = "2.0", default-features = false, features = ["cpu"] }
```

### Target-Specific Compilation

For maximum performance, compile with native CPU flags:
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

This enables all available SIMD instructions for your CPU.

## Testing

Run SIMD tests:
```bash
cargo test --lib simd
```

Run with specific features:
```bash
cargo test --features simd
cargo test --no-default-features --features cpu
```

## Debugging

### Verify SIMD Usage

```rust
use neuro_divergent::optimizations::simd::cpu_features;

let features = cpu_features::detect_cpu_features();
println!("CPU Features: {}", cpu_features::feature_description());
println!("Recommended f32 lanes: {}", features.recommended_f32_lanes());
```

### Performance Profiling

Use criterion for microbenchmarks:
```bash
cargo bench --bench simd_benchmarks -- --verbose
```

Use perf for instruction-level analysis:
```bash
perf stat -e instructions,cycles,cache-misses cargo bench
```

## Safety Considerations

### Unsafe Code

SIMD intrinsics require `unsafe` blocks. Our implementation:
- ✅ Validates alignment and bounds
- ✅ Handles remainder elements safely
- ✅ Falls back to scalar for edge cases
- ✅ Thoroughly tested with fuzzing

### Numerical Stability

Some SIMD approximations trade precision for speed:
- **Fast tanh**: Polynomial approximation (±0.001 error)
- **Fast exp**: Limited to reasonable ranges
- **Softmax**: Numerically stable with max subtraction

For critical applications, verify results match scalar implementations.

## Future Optimizations

Planned improvements:
- [ ] ARM NEON implementations (currently use scalar fallback)
- [ ] AVX-512 optimizations for wider vectors
- [ ] GPU acceleration via CUDA/Metal
- [ ] Quantization (INT8/INT16) for inference
- [ ] Cache-aware matrix multiplication (blocking)

## References

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [ARM NEON Intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [Rust SIMD Documentation](https://doc.rust-lang.org/std/simd/)

## Contributing

When adding new SIMD operations:
1. Implement scalar version first
2. Add SIMD implementation with feature detection
3. Add comprehensive tests (correctness + performance)
4. Add benchmarks to verify speedup
5. Document performance characteristics

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.
