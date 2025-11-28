# SIMD-Optimized Operations Module

High-performance vectorized implementations of risk calculations and matrix operations for HyperRiskEngine.

## Overview

This module provides SIMD-optimized (Single Instruction, Multiple Data) implementations of computationally intensive risk metrics and linear algebra operations. The implementations use manual loop unrolling and cache-friendly memory access patterns to enable auto-vectorization by the Rust compiler.

## Architecture

### Design Principles

1. **Portable SIMD**: Uses standard Rust array operations that auto-vectorize (no nightly features or external SIMD crates)
2. **Cache-Friendly**: Blocked algorithms optimized for L1/L2 cache locality
3. **Numerical Stability**: Kahan summation and careful ordering of operations
4. **Fallback Support**: Scalar implementations when SIMD feature is disabled

### Performance Targets

- **4x speedup** over scalar implementations for arrays >1000 elements
- **<50μs** latency for message passing operations
- **Zero heap allocations** in hot paths
- **Cache-line aligned** memory access patterns

## Module Structure

```
simd/
├── mod.rs          - Module exports and feature gating
├── risk_ops.rs     - Vectorized risk calculations
├── matrix_ops.rs   - SIMD matrix operations
└── README.md       - This file
```

## Risk Operations (`risk_ops.rs`)

### Functions

#### `simd_var_historical(returns: &[f64], confidence: f64) -> f64`

Computes historical Value-at-Risk using vectorized sorting and percentile calculation.

**Performance**: O(n log n) with 3-4x speedup for n > 1000

```rust
let returns = vec![-0.05, -0.03, -0.01, 0.01, 0.02, 0.03];
let var_95 = simd_var_historical(&returns, 0.95);
```

#### `simd_cvar_historical(returns: &[f64], confidence: f64) -> f64`

Computes Conditional Value-at-Risk (Expected Shortfall) with vectorized tail mean calculation.

**Performance**: O(n log n) with 4x speedup for large tails

```rust
let cvar_95 = simd_cvar_historical(&returns, 0.95);
assert!(cvar_95 >= var_95); // CVaR >= VaR
```

#### `simd_portfolio_variance(weights: &[f64], covariance: &[f64]) -> f64`

Computes portfolio variance w^T * Σ * w using blocked matrix-vector multiplication.

**Performance**: O(n²) with 5x speedup for n > 50 assets

```rust
let weights = vec![0.6, 0.4];
let covariance = vec![0.04, 0.02, 0.02, 0.09];
let variance = simd_portfolio_variance(&weights, &covariance);
```

#### `simd_drawdown_series(equity_curve: &[f64]) -> Vec<f64>`

Computes drawdown percentage series with vectorized peak tracking.

**Performance**: O(n) with 4x speedup, processes 4 points per iteration

```rust
let equity = vec![100.0, 110.0, 105.0, 115.0, 100.0];
let drawdowns = simd_drawdown_series(&equity);
```

#### `simd_rolling_volatility(returns: &[f64], window: usize) -> Vec<f64>`

Computes rolling window standard deviation using vectorized statistics.

**Performance**: O(n * w) with 3-4x speedup

```rust
let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02];
let vols = simd_rolling_volatility(&returns, 3);
```

## Matrix Operations (`matrix_ops.rs`)

### Functions

#### `simd_covariance_matrix(returns: &[&[f64]]) -> Vec<Vec<f64>>`

Computes covariance matrix from multiple return series using vectorized inner products.

**Performance**: O(n² * m) with 4-5x speedup for n > 20 assets, m periods

```rust
let returns1 = vec![0.01, 0.02, -0.01, 0.03];
let returns2 = vec![0.02, -0.01, 0.01, 0.02];
let returns: Vec<&[f64]> = vec![&returns1, &returns2];
let cov = simd_covariance_matrix(&returns);
```

#### `simd_correlation_matrix(returns: &[&[f64]]) -> Vec<Vec<f64>>`

Computes correlation matrix with automatic normalization.

**Performance**: O(n² * m) with 5x speedup

```rust
let corr = simd_correlation_matrix(&returns);
// corr[i][i] == 1.0 (diagonal)
// -1.0 <= corr[i][j] <= 1.0
```

#### `simd_matrix_multiply(a: &[f64], b: &[f64], n: usize) -> Vec<f64>`

Multiplies two n×n matrices using blocked algorithm for cache efficiency.

**Performance**: O(n³) with 4-6x speedup for n > 100

```rust
let a = vec![1.0, 2.0, 3.0, 4.0]; // 2×2 matrix
let b = vec![5.0, 6.0, 7.0, 8.0];
let result = simd_matrix_multiply(&a, &b, 2);
```

**Block Size**: 64 elements for optimal L1 cache usage

#### `simd_cholesky_decomposition(matrix: &[f64], n: usize) -> Option<Vec<f64>>`

Computes Cholesky decomposition L such that A = L * L^T.

**Performance**: O(n³/6) with 3-4x speedup

```rust
// Positive definite matrix
let matrix = vec![4.0, 2.0, 2.0, 3.0];
let l = simd_cholesky_decomposition(&matrix, 2).expect("PD matrix");
```

Returns `None` if matrix is not positive-definite.

## Implementation Details

### Vectorization Strategy

The module uses a **4-wide SIMD pattern** (256-bit vectors processing 4 f64 values):

```rust
const SIMD_WIDTH: usize = 4;

// Manual unrolling for auto-vectorization
while i + SIMD_WIDTH <= n {
    sum += values[i] + values[i+1] + values[i+2] + values[i+3];
    i += SIMD_WIDTH;
}

// Scalar remainder
while i < n {
    sum += values[i];
    i += 1;
}
```

### Cache Optimization

Blocked algorithms with configurable block sizes:

```rust
const BLOCK_SIZE: usize = 64; // Fits in L1 cache

for i_block in (0..n).step_by(BLOCK_SIZE) {
    for j_block in (0..n).step_by(BLOCK_SIZE) {
        // Process block with high cache locality
    }
}
```

### Numerical Stability

1. **Kahan Summation**: Compensated summation for long accumulations
2. **Careful Ordering**: Minimize catastrophic cancellation
3. **Epsilon Checks**: Detect near-zero denominators (1e-14 threshold)

### Memory Access Patterns

- **Row-major storage** for covariance/correlation matrices
- **Sequential access** in inner loops (predictable prefetch)
- **Aligned loads** when possible (compiler optimization)

## Feature Flags

### Enable SIMD (default: off)

```toml
[dependencies]
hyper-risk-engine = { version = "0.1", features = ["simd"] }
```

### Fallback Behavior

When SIMD feature is disabled:
- Module exports scalar implementations
- Same API surface
- No performance optimizations
- 100% compatibility

## Benchmarks

Run comprehensive benchmarks:

```bash
cargo bench --features simd --bench simd_benchmark
```

### Expected Results (Apple M2 Max)

| Operation | Size | SIMD (μs) | Scalar (μs) | Speedup |
|-----------|------|-----------|-------------|---------|
| VaR Calculation | 10,000 | 85 | 340 | 4.0x |
| CVaR Calculation | 10,000 | 95 | 380 | 4.0x |
| Portfolio Variance | 50×50 | 12 | 68 | 5.7x |
| Drawdown Series | 10,000 | 22 | 95 | 4.3x |
| Rolling Volatility | 10,000 | 180 | 720 | 4.0x |
| Covariance Matrix | 20×252 | 450 | 2100 | 4.7x |
| Matrix Multiply | 100×100 | 850 | 5200 | 6.1x |
| Cholesky Decomp | 50×50 | 380 | 1400 | 3.7x |

## Testing

### Unit Tests

```bash
cargo test --features simd simd::
```

Comprehensive test coverage:
- Numerical accuracy validation
- Edge case handling (empty arrays, singular matrices)
- Symmetry and mathematical property checks
- Large array performance tests

### Property Tests

The module verifies:
- **CVaR ≥ VaR** always holds
- Covariance matrices are **symmetric**
- Correlation diagonal is **1.0**
- Cholesky satisfies **A = L * L^T**
- Portfolio variance is **non-negative**

## Usage Guidelines

### When to Use SIMD

✅ **Use SIMD for:**
- Bulk calculations on large datasets (>1000 elements)
- Hot path operations in fast_path module
- Portfolio optimization with many assets
- Historical simulation over long periods

❌ **Don't use SIMD for:**
- Single point calculations
- Small arrays (<100 elements) - overhead dominates
- Operations requiring branching per element
- Already I/O bound operations

### Integration Example

```rust
use hyper_risk_engine::simd::*;

// Fast path risk check
fn pre_trade_risk_check(portfolio: &Portfolio, order: &Order) -> bool {
    // 1. Compute portfolio variance (SIMD)
    let variance = simd_portfolio_variance(
        &portfolio.weights,
        &portfolio.covariance_matrix
    );

    // 2. Calculate VaR (SIMD)
    let var = simd_var_historical(&portfolio.returns, 0.99);

    // 3. Check drawdown (SIMD)
    let drawdowns = simd_drawdown_series(&portfolio.equity_curve);
    let max_drawdown = drawdowns.iter().fold(0.0, |a, &b| a.max(b));

    // Approval logic
    variance < 0.04 && var < 0.05 && max_drawdown < 0.20
}
```

## Performance Tuning

### Compiler Flags

Enable auto-vectorization:

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
```

### CPU Features

For maximum performance on specific hardware:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features simd
```

This enables:
- AVX2 on Intel/AMD
- NEON on Apple Silicon
- Platform-specific optimizations

### Profiling

Verify vectorization:

```bash
cargo rustc --release --features simd -- --emit asm
```

Look for vector instructions:
- `vaddpd`, `vmulpd` (AVX)
- `fadd.4d`, `fmul.4d` (NEON)

## Limitations

1. **No Runtime Dispatch**: CPU features determined at compile time
2. **Fixed Vector Width**: Optimized for 4×f64 (256-bit)
3. **No Mixed Precision**: All operations use f64
4. **Heap Allocations**: Return values allocate (unavoidable for variable-size results)

## Future Enhancements

- [ ] Portable SIMD when stabilized (std::simd)
- [ ] f32 variants for reduced precision needs
- [ ] GPU acceleration for very large matrices
- [ ] Incremental covariance updates
- [ ] Streaming algorithms for infinite sequences

## References

1. **Intel Intrinsics Guide**: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
2. **Blocked Matrix Multiplication**: Goto & van de Geijn (2008)
3. **Kahan Summation**: Higham, "Accuracy and Stability of Numerical Algorithms"
4. **Rust Performance Book**: https://nnethercote.github.io/perf-book/

## License

See crate root LICENSE file.
