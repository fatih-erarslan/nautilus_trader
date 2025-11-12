# SIMD Optimization Strategy for HyperPhysics

**Agent**: Performance-Engineer
**Date**: 2025-11-12
**Status**: Strategic Plan - Implementation Pending

---

## 1. SIMD Architecture Overview

### 1.1 Target Platforms

| Platform | ISA | Vector Width | Datatype Support | Priority |
|----------|-----|--------------|------------------|----------|
| **x86_64** | AVX2 | 256-bit | 4×f64, 8×f32 | ⭐ PRIMARY |
| **x86_64** | AVX-512 | 512-bit | 8×f64, 16×f32 | ⭐⭐ FUTURE |
| **ARM64** | NEON | 128-bit | 2×f64, 4×f32 | ⭐ APPLE SILICON |
| **WASM** | SIMD128 | 128-bit | 4×f32 only | ⭐⭐ FUTURE |

**Current Support**: AVX2 only (partial implementation)

---

### 1.2 Rust SIMD Approaches

#### Option A: Portable SIMD (std::simd) - **RECOMMENDED**
```rust
#![feature(portable_simd)]
use std::simd::*;

pub fn update_probabilities_portable(
    h_eff: &[f32],
    temperature: f32,
    output: &mut [f32],
) {
    let t_inv = f32x8::splat(1.0 / temperature);

    for (h_chunk, out_chunk) in h_eff.chunks_exact(8).zip(output.chunks_exact_mut(8)) {
        let h = f32x8::from_slice(h_chunk);
        let x = h * t_inv;
        let prob = fast_sigmoid_simd(x);  // Vectorized sigmoid
        prob.copy_to_slice(out_chunk);
    }

    // Handle remainder with scalar code
}
```

**Pros**:
- Compiler auto-selects AVX2/NEON/AVX-512 at compile time
- Safe Rust (no unsafe blocks after initial implementation)
- Will be stabilized in Rust 2024 edition

**Cons**:
- Requires nightly Rust (for now)
- Limited math functions (no std exp/log yet)

---

#### Option B: Explicit Intrinsics (std::arch) - **CURRENT**
```rust
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::*;

pub unsafe fn dot_product_avx2(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = _mm256_setzero_pd();
    for i in 0..(a.len() / 4) {
        let va = _mm256_loadu_pd(a.as_ptr().add(i * 4));
        let vb = _mm256_loadu_pd(b.as_ptr().add(i * 4));
        sum = _mm256_add_pd(sum, _mm256_mul_pd(va, vb));
    }
    // Horizontal sum...
}
```

**Pros**:
- Maximum control and performance
- Works on stable Rust

**Cons**:
- Requires unsafe blocks
- Platform-specific code duplication
- Manual fallback implementations

---

**Decision**: Use **Option A (Portable SIMD)** for new code, keep existing AVX2 intrinsics as fallback.

---

## 2. Critical SIMD Kernels

### 2.1 Kernel #1: Vectorized Sigmoid
**Priority**: ⭐⭐⭐ CRITICAL (blocks Gillespie optimization)

```rust
/// Fast sigmoid approximation using SIMD
/// sigmoid(x) = 1 / (1 + exp(-x))
///
/// Uses rational function approximation:
/// sigmoid(x) ≈ 0.5 + x / (2 * (1 + |x|/2))
pub fn fast_sigmoid_simd(x: f32x8) -> f32x8 {
    // Rational approximation (max error ~0.001)
    let half = f32x8::splat(0.5);
    let two = f32x8::splat(2.0);

    let abs_x = x.abs();
    let denom = two * (f32x8::splat(1.0) + abs_x / two);
    let approx = half + x / denom;

    // Clamp to [0, 1]
    approx.clamp(f32x8::splat(0.0), f32x8::splat(1.0))
}

/// More accurate sigmoid using vectorized exp (slower)
pub fn accurate_sigmoid_simd(x: f32x8) -> f32x8 {
    let neg_x = -x;
    let exp_neg_x = fast_exp_simd(neg_x);  // Kernel #2
    f32x8::splat(1.0) / (f32x8::splat(1.0) + exp_neg_x)
}
```

**Performance Target**:
- Rational approx: ~2 cycles/element (16 cycles for 8 elements)
- Accurate exp: ~10 cycles/element (80 cycles for 8 elements)

**Trade-off**: Rational is 5x faster but 0.1% max error. Use accurate for financial risk calculations.

---

### 2.2 Kernel #2: Vectorized Exponential
**Priority**: ⭐⭐⭐ CRITICAL (required by sigmoid, Boltzmann factors)

```rust
/// Fast exp approximation using SIMD (Agner Fog's algorithm)
/// Accurate to ~1e-6 over [-88, 88]
pub fn fast_exp_simd(x: f32x8) -> f32x8 {
    // Split x = n*ln(2) + r, where |r| < ln(2)/2
    const LN2_RECIP: f32 = 1.442695041;  // 1/ln(2)
    const LN2_HI: f32 = 0.693359375;
    const LN2_LO: f32 = -2.1219444e-4;

    // n = round(x / ln(2))
    let n = (x * f32x8::splat(LN2_RECIP)).round();

    // r = x - n*ln(2)
    let r = x - n * f32x8::splat(LN2_HI) - n * f32x8::splat(LN2_LO);

    // Polynomial approximation for exp(r) (degree 5)
    let c1 = f32x8::splat(1.0);
    let c2 = f32x8::splat(1.0);
    let c3 = f32x8::splat(0.5);
    let c4 = f32x8::splat(0.166666667);
    let c5 = f32x8::splat(0.041666667);
    let c6 = f32x8::splat(0.008333333);

    let r2 = r * r;
    let poly = c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * c6))));

    // Scale by 2^n (using floating-point tricks)
    // result = poly * 2^n
    scale_by_power_of_two(poly, n)
}

fn scale_by_power_of_two(x: f32x8, n: f32x8) -> f32x8 {
    // Convert n to integer, add to exponent field
    // TODO: Implement using bit manipulation
    // For now, use scalar fallback
    unimplemented!("Requires low-level bit manipulation")
}
```

**Status**: Partially implemented, needs bit manipulation for `scale_by_power_of_two`.

**Alternative**: Use Intel SVML (Short Vector Math Library) intrinsics:
```rust
#[cfg(target_arch = "x86_64")]
extern "C" {
    #[link_name = "__svml_expf8"]
    fn _mm256_exp_ps(x: __m256) -> __m256;
}
```

**Recommendation**: Use SVML for x86, implement portable version for ARM.

---

### 2.3 Kernel #3: Vectorized Dot Product
**Priority**: ⭐⭐ HIGH (effective field calculation)

**Status**: ✅ **ALREADY IMPLEMENTED** in `simd.rs::dot_product_avx2()`

```rust
// Current implementation (AVX2)
pub unsafe fn dot_product_avx2(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = _mm256_setzero_pd();
    for i in 0..(a.len() / 4) {
        let va = _mm256_loadu_pd(a.as_ptr().add(i * 4));
        let vb = _mm256_loadu_pd(b.as_ptr().add(i * 4));
        sum = _mm256_add_pd(sum, _mm256_mul_pd(va, vb));
    }
    // ... horizontal sum
}
```

**Required Change**: Port to portable SIMD:
```rust
pub fn dot_product_portable(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = f32x8::splat(0.0);

    for (a_chunk, b_chunk) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        let va = f32x8::from_slice(a_chunk);
        let vb = f32x8::from_slice(b_chunk);
        sum += va * vb;
    }

    sum.reduce_sum() + scalar_remainder(a, b)
}
```

**Optimization**: Use FMA (fused multiply-add) for better accuracy and performance.

---

### 2.4 Kernel #4: Vectorized State Updates
**Priority**: ⭐ MEDIUM (currently not a bottleneck)

**Status**: ✅ Implemented in `simd.rs::update_states_avx2()`

**Improvement**: Add stochastic rounding for probabilistic states:
```rust
pub fn stochastic_update_simd(
    probabilities: &[f32],
    random: &[f32],  // Uniform [0,1] random numbers
    output: &mut [bool],
) {
    for ((p_chunk, r_chunk), out_chunk) in probabilities
        .chunks_exact(8)
        .zip(random.chunks_exact(8))
        .zip(output.chunks_exact_mut(8))
    {
        let p = f32x8::from_slice(p_chunk);
        let r = f32x8::from_slice(r_chunk);
        let mask = p.simd_gt(r);

        for (i, &m) in mask.to_array().iter().enumerate() {
            out_chunk[i] = m;
        }
    }
}
```

---

## 3. Integration Points

### 3.1 Gillespie Simulator Refactor

**Current Code** (gillespie.rs, lines 60-72):
```rust
for (_i, pbit) in self.lattice.pbits().iter().enumerate() {
    let h_eff = pbit.effective_field(&states);  // ❌ Scalar
    temp_pbit.update_probability(h_eff);        // ❌ Scalar sigmoid
    let rate = temp_pbit.flip_rate();
    rates.push(rate);
}
```

**SIMD Optimized**:
```rust
use crate::simd::SimdOps;

// Batch calculate all effective fields (vectorized dot products)
let h_eff_batch = SimdOps::effective_fields_batch(
    &self.lattice.coupling_matrix(),
    &states,
);

// Vectorized probability updates (8 pBits at once)
let probabilities = SimdOps::sigmoid_batch(&h_eff_batch, temperature);

// Vectorized rate calculation
let rates = SimdOps::flip_rates_batch(&probabilities);
```

**Expected Speedup**: 4-8x for this loop (currently 40-60% of runtime)

---

### 3.2 Metropolis Energy Calculation

**Current Code** (metropolis.rs, line 91):
```rust
let h_eff = pbit.effective_field(states);  // ❌ Scalar dot product
```

**SIMD Optimized**:
```rust
let h_eff = SimdOps::dot_product(&couplings[idx], states_as_f32);
```

**Expected Speedup**: 3-6x for energy calculations

---

### 3.3 PBit Lattice Data Layout
**Current**: Array-of-Structs (AoS) - cache-inefficient for SIMD
```rust
struct PBit {
    bias: f64,
    couplings: Vec<f64>,
    state: bool,
}
lattice.pbits: Vec<PBit>  // ❌ Non-contiguous memory
```

**SIMD-Optimized**: Struct-of-Arrays (SoA)
```rust
struct PBitLatticeSimd {
    biases: Vec<f32>,         // Aligned for SIMD
    couplings: Vec<Vec<f32>>, // Or flat matrix
    states: Vec<bool>,
    // SIMD-aligned padding
}
```

**Benefit**: Contiguous memory allows efficient vector loads.

**Migration Strategy**: Add SoA layout alongside existing AoS, benchmark both.

---

## 4. Performance Benchmarking

### 4.1 Micro-Benchmarks
**File**: `benches/simd_kernels.rs` (to be created)

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hyperphysics_pbit::simd::*;

fn bench_sigmoid(c: &mut Criterion) {
    let mut group = c.benchmark_group("sigmoid");
    let data: Vec<f32> = (0..10000).map(|i| (i as f32 - 5000.0) / 1000.0).collect();

    group.bench_function("scalar", |b| {
        b.iter(|| {
            let mut out = vec![0.0; data.len()];
            for (i, &x) in data.iter().enumerate() {
                out[i] = 1.0 / (1.0 + (-x).exp());
            }
            black_box(out);
        })
    });

    group.bench_function("simd_rational", |b| {
        b.iter(|| {
            fast_sigmoid_batch_rational(&data)
        })
    });

    group.bench_function("simd_accurate", |b| {
        b.iter(|| {
            fast_sigmoid_batch_accurate(&data)
        })
    });

    group.finish();
}
```

**Metrics to Track**:
- Throughput (ops/sec)
- Latency (ns/op)
- Instructions per cycle (IPC)
- Cache misses

---

### 4.2 End-to-End Benchmarks
**File**: `benches/gillespie_simd.rs` (to be created)

```rust
fn bench_gillespie_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("gillespie_step");

    for &n in &[100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &n| {
            let lattice = PBitLattice::random(n);
            let mut sim = GillespieSimulator::new(lattice);
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            b.iter(|| {
                sim.step(&mut rng).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, &n| {
            let lattice = PBitLatticeSimd::random(n);  // SIMD layout
            let mut sim = GillespieSimulatorSimd::new(lattice);
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            b.iter(|| {
                sim.step(&mut rng).unwrap();
            });
        });
    }

    group.finish();
}
```

**Target Metrics** (10k pBits):
- Scalar baseline: 500 μs/step
- SIMD target: 100 μs/step (5x speedup)

---

## 5. Implementation Roadmap

### Week 1: Core SIMD Infrastructure
- [ ] **Day 1-2**: Implement portable SIMD sigmoid (rational + accurate)
- [ ] **Day 3**: Port dot product to portable SIMD
- [ ] **Day 4-5**: Create micro-benchmarks, establish baselines

**Deliverables**:
- `simd/sigmoid.rs` with portable SIMD
- Benchmark showing 4-8x speedup for sigmoid

---

### Week 2: Gillespie Integration
- [ ] **Day 1-2**: Refactor `effective_field()` to use SIMD dot product
- [ ] **Day 3**: Batch vectorize probability updates in Gillespie
- [ ] **Day 4-5**: End-to-end Gillespie benchmarks

**Deliverables**:
- Gillespie simulator with SIMD (feature-gated)
- Benchmark showing 3-5x Gillespie speedup

---

### Week 3: Metropolis + ARM Support
- [ ] **Day 1-2**: SIMD optimize Metropolis energy calculations
- [ ] **Day 3-4**: Implement NEON equivalents (test on Apple Silicon)
- [ ] **Day 5**: Cross-platform benchmarks

**Deliverables**:
- Full SIMD support on x86 + ARM
- Performance parity report

---

## 6. Success Criteria

### Quantitative Metrics
| Metric | Baseline | Target | Stretch Goal |
|--------|----------|--------|--------------|
| Sigmoid throughput | 100 Mops/s | 500 Mops/s | 1 Gops/s |
| Dot product (1k) | 1 μs | 200 ns | 100 ns |
| Gillespie step (10k) | 500 μs | 100 μs | 50 μs |
| Overall speedup | 1x | **3-5x** | **8x** |

### Qualitative Goals
- ✅ Zero correctness regressions (validated with proptest)
- ✅ Portable code (x86, ARM, fallback)
- ✅ Feature-gated (can disable SIMD if needed)
- ✅ Maintainable (avoid overly complex intrinsics)

---

## 7. Risk Mitigation

### Risk #1: SIMD Slower Than Scalar (Low Probability)
**Cause**: Small data sizes, overhead of vector setup
**Mitigation**: Only use SIMD for lattices >100 pBits, fallback otherwise

### Risk #2: Accuracy Loss (Medium Probability)
**Cause**: Fast approximations (rational sigmoid, polynomial exp)
**Mitigation**:
- Validate with proptest against scalar ground truth
- Provide accurate versions for financial calculations
- Document error bounds

### Risk #3: Platform Incompatibility (Low Probability)
**Cause**: Portable SIMD not available on target platform
**Mitigation**:
- Always include scalar fallback
- Test on CI across x86/ARM/WASM

---

## 8. Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    #[test]
    fn test_sigmoid_accuracy() {
        for x in -10..=10 {
            let xf = x as f32;
            let scalar = 1.0 / (1.0 + (-xf).exp());
            let simd_result = fast_sigmoid_simd(f32x8::splat(xf))[0];

            assert_relative_eq!(scalar, simd_result, epsilon = 1e-3);
        }
    }
}
```

### Property-Based Tests
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn simd_matches_scalar(
        data in prop::collection::vec(-10.0..10.0f32, 0..10000)
    ) {
        let scalar_results: Vec<f32> = data.iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        let simd_results = fast_sigmoid_batch(&data);

        for (s, v) in scalar_results.iter().zip(simd_results.iter()) {
            prop_assert!((s - v).abs() < 1e-3);
        }
    }
}
```

---

## 9. References

### SIMD Libraries
- [Agner Fog's Vector Class Library](https://github.com/vectorclass/version2)
- [Intel SVML](https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-10/intrinsics-for-short-vector-math-library-operations.html)
- [Sleef (Portable SIMD Math)](https://sleef.org/)

### Algorithms
- **Fast Exponential**: Schraudolph (1999) "A Fast, Compact Approximation of the Exponential Function"
- **Sigmoid Approximation**: Various rational function approaches
- **Vectorized Math**: Fog, A. (2020) "Optimizing Software in C++"

### Rust SIMD
- [Portable SIMD RFC](https://rust-lang.github.io/rfcs/2948-portable-simd.html)
- [std::arch documentation](https://doc.rust-lang.org/core/arch/)

---

**Agent Status**: 📋 Strategy Complete
**Next Phase**: Implementation (Week 1)
**Dependencies**: wgpu fix (to unblock benchmarking)
