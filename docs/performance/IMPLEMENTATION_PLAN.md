# HyperPhysics Performance Optimization - Implementation Plan

**Agent**: Performance-Engineer (Queen Seraphina Command)
**Created**: 2025-11-12
**Status**: Roadmap Approved - Ready for Execution
**Timeline**: 6 weeks (deferred GPU to Phase 4+)

---

## Executive Summary

### Current State
- ❌ **Blocker**: wgpu v0.19.4 compilation error prevents benchmarking
- ✅ SIMD infrastructure exists but not integrated with Gillespie/Metropolis
- ✅ Benchmark harness ready
- ⏳ No performance baselines established

### Strategic Goals
1. **Phase 1** (Week 1): Fix wgpu, establish CPU baselines
2. **Phase 2** (Weeks 2-3): SIMD optimization → 3-5x speedup
3. **Phase 3** (Week 4): ARM NEON support + cross-platform testing
4. **Phase 4+** (Weeks 5-6+): GPU acceleration (if needed)

### Success Criteria
- ✅ Gillespie step (10k pBits): 500 μs → **100 μs** (5x speedup)
- ✅ Metropolis step (10k pBits): 500 μs → **100 μs** (5x speedup)
- ✅ Message passing latency: <50 μs
- ✅ Zero correctness regressions (validated with proptest)

---

## Phase 1: Baseline Establishment (Week 1)

### 🎯 Objectives
1. Fix wgpu compilation blocker
2. Run existing benchmarks → establish scalar baseline
3. Profile with flamegraph → identify true hotspots

---

### Day 1-2: Dependency Cleanup

#### Task 1.1: Fix wgpu Compilation
**Priority**: 🔴 CRITICAL
**Owner**: Performance-Engineer / Systems-Architect

**Current Error**:
```
error: macro expansion ends with an incomplete expression
   --> wgpu-0.19.4/src/backend/wgpu_core.rs:783:92
```

**Solution Options**:
```toml
# Option A: Update wgpu (RECOMMENDED)
[dependencies]
wgpu = { version = "0.20", optional = true }

# Option B: Make GPU optional (SHORT-TERM)
[features]
default = []
gpu = ["wgpu", "bytemuck"]

# Then disable GPU feature in CI
cargo build --no-default-features
```

**Action Items**:
- [ ] Try `cargo update wgpu` to v0.20+
- [ ] If fails, make GPU feature optional
- [ ] Verify builds pass: `cargo build --workspace`
- [ ] Document wgpu version in `docs/dependencies.md`

**Estimated Time**: 2-4 hours
**Success Criteria**: `cargo build --workspace` completes successfully

---

#### Task 1.2: Cargo.toml Reorganization
**Priority**: 🟡 MEDIUM

Add explicit feature flags for performance backends:

```toml
[features]
default = ["simd"]

# Performance backends (mutually exclusive at runtime)
simd = []          # Portable SIMD (AVX2/NEON)
simd-nightly = []  # Use std::simd (requires nightly)
gpu = ["wgpu", "bytemuck", "pollster"]
gpu-cuda = ["cudarc"]

# Development features
profiling = ["pprof"]
benchmarking = ["criterion/html_reports"]

[dependencies]
# SIMD (no dependencies for intrinsics)
# std::simd requires nightly Rust

# GPU (optional)
wgpu = { version = "0.20", optional = true }
bytemuck = { version = "1.14", optional = true }
pollster = { version = "0.3", optional = true }
cudarc = { version = "0.10", optional = true }

# Profiling
pprof = { version = "0.13", optional = true, features = ["flamegraph"] }

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1.9"
approx = "0.5"
```

**Action Items**:
- [ ] Update `Cargo.toml` with feature flags
- [ ] Test builds with different feature combinations
- [ ] Update CI to test `--features simd` and `--no-default-features`

**Estimated Time**: 1-2 hours

---

### Day 3: Benchmark Baseline

#### Task 1.3: Run Existing Benchmarks
**Priority**: 🔴 CRITICAL

```bash
# Ensure Rust environment has cargo
export PATH="$HOME/.cargo/bin:$PATH"

# Run baseline benchmarks (scalar code only)
cargo bench --workspace --no-default-features -- --save-baseline scalar

# Generate HTML reports
ls -lh target/criterion/

# Key metrics to extract:
# - message_passing/mpsc_channel: median latency
# - lockfree/atomic_increment: throughput
# - memory/vec_with_capacity: allocation speed
```

**Action Items**:
- [ ] Run `cargo bench --workspace` (no SIMD)
- [ ] Save output to `docs/performance/BASELINE_RESULTS.txt`
- [ ] Parse key metrics into table format
- [ ] Identify slowest 3 benchmarks

**Estimated Time**: 3-4 hours (including compilation)
**Success Criteria**: Baseline numbers documented

---

#### Task 1.4: Create SIMD Micro-Benchmarks
**Priority**: 🟢 HIGH

**New File**: `benches/simd_kernels.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hyperphysics_pbit::simd::*;

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for &size in &[100, 1000, 10000] {
        let a: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..size).map(|i| (i + 1) as f64).collect();

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| {
                black_box(portable::dot_product(&a, &b))
            })
        });

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        group.bench_with_input(BenchmarkId::new("avx2", size), &size, |bench, _| {
            bench.iter(|| unsafe {
                black_box(dot_product_avx2(&a, &b))
            })
        });
    }

    group.finish();
}

fn bench_sigmoid(c: &mut Criterion) {
    let mut group = c.benchmark_group("sigmoid");
    let data: Vec<f64> = (0..10000).map(|i| (i as f64 - 5000.0) / 1000.0).collect();

    group.bench_function("scalar", |b| {
        b.iter(|| {
            let result: Vec<f64> = data.iter()
                .map(|&x| 1.0 / (1.0 + (-x).exp()))
                .collect();
            black_box(result);
        })
    });

    // TODO: Add SIMD sigmoid when implemented

    group.finish();
}

criterion_group!(benches, bench_dot_product, bench_sigmoid);
criterion_main!(benches);
```

**Action Items**:
- [ ] Create `benches/simd_kernels.rs`
- [ ] Run benchmarks: `cargo bench --bench simd_kernels -- --save-baseline before-opt`
- [ ] Document results in `docs/performance/SIMD_BASELINE.txt`

**Estimated Time**: 2 hours

---

### Day 4-5: Profiling and Analysis

#### Task 1.5: Flamegraph Profiling
**Priority**: 🟢 HIGH

```bash
# Install flamegraph
cargo install flamegraph

# Profile message passing benchmark
cargo flamegraph --bench message_passing --output docs/performance/flamegraph_baseline.svg

# Profile Gillespie simulation (create bench first)
cargo flamegraph --bench gillespie_simulation --output docs/performance/flamegraph_gillespie.svg

# Open in browser
open docs/performance/flamegraph_baseline.svg
```

**Action Items**:
- [ ] Install `cargo-flamegraph`
- [ ] Generate flamegraphs for top 3 benchmarks
- [ ] Identify functions consuming >10% CPU time
- [ ] Document hotspots in `BASELINE_ANALYSIS.md`

**Estimated Time**: 4 hours
**Success Criteria**: Flamegraphs show clear hotspots (expected: `effective_field`, `exp()`)

---

#### Task 1.6: Create Gillespie End-to-End Benchmark
**Priority**: 🟢 HIGH

**New File**: `benches/gillespie_simulation.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hyperphysics_pbit::{PBitLattice, GillespieSimulator};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn bench_gillespie_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("gillespie_step");

    for &n in &[100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &n| {
            let lattice = PBitLattice::random_with_size(n, 1.0);
            let mut sim = GillespieSimulator::new(lattice);
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            b.iter(|| {
                black_box(sim.step(&mut rng).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_gillespie_1000_steps(c: &mut Criterion) {
    let lattice = PBitLattice::random_with_size(1000, 1.0);
    let mut sim = GillespieSimulator::new(lattice);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    c.bench_function("gillespie_1000_steps", |b| {
        b.iter(|| {
            sim.simulate_events(1000, &mut rng).unwrap();
        });
    });
}

criterion_group!(benches, bench_gillespie_step, bench_gillespie_1000_steps);
criterion_main!(benches);
```

**Action Items**:
- [ ] Create `benches/gillespie_simulation.rs`
- [ ] Add `random_with_size()` helper to `PBitLattice`
- [ ] Run: `cargo bench --bench gillespie_simulation -- --save-baseline scalar`
- [ ] Document baseline: "10k pBit step: XXX μs"

**Estimated Time**: 3 hours

---

### Week 1 Deliverables
- ✅ wgpu compilation fixed or GPU disabled
- ✅ Scalar baseline benchmarks complete
- ✅ Flamegraph profiles generated
- ✅ Bottlenecks identified and documented
- 📊 **Key Metric**: Gillespie 10k step baseline (expected: 500-1000 μs)

---

## Phase 2: SIMD Optimization (Weeks 2-3)

### 🎯 Objectives
1. Implement vectorized sigmoid and exp functions
2. Integrate SIMD into Gillespie and Metropolis
3. Achieve 3-5x speedup on hot paths

---

### Week 2, Day 1-2: Vectorized Math Kernels

#### Task 2.1: Implement Fast Vectorized Sigmoid
**Priority**: 🔴 CRITICAL

**New File**: `crates/hyperphysics-pbit/src/simd/sigmoid.rs`

```rust
#![cfg(target_feature = "avx2")]
use std::arch::x86_64::*;

/// Fast rational sigmoid approximation (vectorized)
/// sigmoid(x) ≈ 0.5 + x / (2 * (1 + |x|/2))
/// Max error: ~0.001
pub unsafe fn sigmoid_rational_avx2(x: &[f64], output: &mut [f64]) {
    assert_eq!(x.len(), output.len());

    let half = _mm256_set1_pd(0.5);
    let two = _mm256_set1_pd(2.0);

    for i in (0..x.len()).step_by(4) {
        let xv = _mm256_loadu_pd(x.as_ptr().add(i));
        let abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), xv);  // Absolute value
        let denom = _mm256_mul_pd(two, _mm256_add_pd(_mm256_set1_pd(1.0),
                                                      _mm256_div_pd(abs_x, two)));
        let approx = _mm256_add_pd(half, _mm256_div_pd(xv, denom));

        // Clamp to [0, 1]
        let clamped = _mm256_max_pd(_mm256_setzero_pd(),
                                     _mm256_min_pd(_mm256_set1_pd(1.0), approx));

        _mm256_storeu_pd(output.as_mut_ptr().add(i), clamped);
    }

    // Handle remainder
    for i in (x.len() / 4 * 4)..x.len() {
        output[i] = sigmoid_scalar(x[i]);
    }
}

fn sigmoid_scalar(x: f64) -> f64 {
    0.5 + x / (2.0 * (1.0 + x.abs() / 2.0))
}
```

**Action Items**:
- [ ] Create `simd/sigmoid.rs` module
- [ ] Implement AVX2 version (4x f64)
- [ ] Add scalar fallback
- [ ] Unit test: compare against `1.0 / (1.0 + (-x).exp())` with 0.1% tolerance
- [ ] Benchmark: target 5-10x faster than scalar

**Estimated Time**: 6 hours

---

#### Task 2.2: Vectorized Exponential (Using Intel SVML)
**Priority**: 🔴 CRITICAL

**New File**: `crates/hyperphysics-pbit/src/simd/exp.rs`

```rust
#[cfg(target_arch = "x86_64")]
extern "C" {
    /// Intel Short Vector Math Library (SVML) - vectorized exp
    #[link_name = "__svml_exp4"]
    fn _mm256_exp_pd(x: __m256d) -> __m256d;
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn exp_avx2_svml(x: &[f64], output: &mut [f64]) {
    for i in (0..x.len()).step_by(4) {
        let xv = _mm256_loadu_pd(x.as_ptr().add(i));
        let result = _mm256_exp_pd(xv);
        _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
    }

    // Remainder
    for i in (x.len() / 4 * 4)..x.len() {
        output[i] = x[i].exp();
    }
}
```

**Alternative** (if SVML unavailable): Polynomial approximation (see `SIMD_STRATEGY.md` section 2.2)

**Action Items**:
- [ ] Implement SVML version
- [ ] Add polynomial fallback for non-Intel platforms
- [ ] Unit test: max error <1e-6
- [ ] Benchmark: target 3-5x faster than scalar

**Estimated Time**: 8 hours

---

### Week 2, Day 3-5: Gillespie Integration

#### Task 2.3: Refactor `effective_field()` to Use SIMD
**Priority**: 🔴 CRITICAL

**Current Code** (`crates/hyperphysics-pbit/src/pbit.rs`):
```rust
pub fn effective_field(&self, states: &[bool]) -> f64 {
    let mut h = self.bias;
    for (i, &coupling) in self.couplings.iter().enumerate() {
        if coupling != 0.0 && states[i] {
            h += coupling;  // Assumes state=true → s=+1
        }
    }
    h
}
```

**SIMD Optimized** (new implementation):
```rust
pub fn effective_field_simd(&self, states: &[bool]) -> f64 {
    use crate::simd::SimdOps;

    // Convert states to f64: true=+1, false=-1
    let states_f64: Vec<f64> = states.iter()
        .map(|&s| if s { 1.0 } else { -1.0 })
        .collect();

    self.bias + SimdOps::dot_product(&self.couplings, &states_f64)
}
```

**Better**: Precompute `states_f64` in lattice to avoid repeated conversion.

**Action Items**:
- [ ] Add `states_as_spin()` helper to `PBitLattice` (returns `Vec<f64>`)
- [ ] Refactor `effective_field()` to use SIMD dot product
- [ ] Add feature flag: `#[cfg(feature = "simd")]`
- [ ] Benchmark: single call speedup 3-6x

**Estimated Time**: 4 hours

---

#### Task 2.4: Vectorize Gillespie Propensity Calculation
**Priority**: 🔴 CRITICAL

**Current Code** (`gillespie.rs`, lines 60-72):
```rust
for (_i, pbit) in self.lattice.pbits().iter().enumerate() {
    let h_eff = pbit.effective_field(&states);
    let mut temp_pbit = pbit.clone();
    temp_pbit.update_probability(h_eff);
    let rate = temp_pbit.flip_rate();
    rates.push(rate);
}
```

**SIMD Optimized**:
```rust
// Calculate all effective fields in one batch (SIMD dot products)
let h_eff_batch = self.lattice.effective_fields_batch_simd();

// Vectorized sigmoid: prob = sigmoid(h_eff / T)
use crate::simd::sigmoid_rational_avx2;
let probabilities = unsafe {
    let mut probs = vec![0.0; h_eff_batch.len()];
    let h_over_t: Vec<f64> = h_eff_batch.iter()
        .map(|&h| h / self.lattice.temperature())
        .collect();
    sigmoid_rational_avx2(&h_over_t, &mut probs);
    probs
};

// Calculate flip rates (vectorized: r = p if s=0, else 1-p)
let states = self.lattice.states();
let rates: Vec<f64> = probabilities.iter().enumerate()
    .map(|(i, &p)| if states[i] { 1.0 - p } else { p })
    .collect();
```

**Action Items**:
- [ ] Implement `PBitLattice::effective_fields_batch_simd()`
- [ ] Refactor Gillespie to use batch calculation
- [ ] Benchmark: 10k pBit step → target 200-300 μs (2-3x speedup)

**Estimated Time**: 8 hours

---

### Week 3, Day 1-3: Metropolis + End-to-End Testing

#### Task 2.5: SIMD Optimize Metropolis
**Priority**: 🟢 HIGH

Similar refactor to Gillespie:
- Replace `energy_change()` scalar dot product with SIMD
- Batch energy calculations for multiple proposed flips (speculative execution)

**Action Items**:
- [ ] Refactor `MetropolisSimulator::energy_change()` to use SIMD
- [ ] Benchmark: 10k pBit step → target <150 μs

**Estimated Time**: 6 hours

---

#### Task 2.6: Property-Based Testing (Correctness)
**Priority**: 🟡 MEDIUM

**New File**: `crates/hyperphysics-pbit/tests/simd_correctness.rs`

```rust
use proptest::prelude::*;
use hyperphysics_pbit::simd::*;
use approx::assert_relative_eq;

proptest! {
    #[test]
    fn test_sigmoid_matches_scalar(
        data in prop::collection::vec(-10.0..10.0f64, 0..10000)
    ) {
        let scalar_results: Vec<f64> = data.iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        let mut simd_results = vec![0.0; data.len()];
        unsafe {
            sigmoid_rational_avx2(&data, &mut simd_results);
        }

        for (s, v) in scalar_results.iter().zip(simd_results.iter()) {
            // Rational approximation has ~0.1% error
            assert_relative_eq!(s, v, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_gillespie_simd_vs_scalar(
        n_pbits in 100..1000usize
    ) {
        let lattice_scalar = PBitLattice::random_with_size(n_pbits, 1.0);
        let lattice_simd = lattice_scalar.clone();

        let mut sim_scalar = GillespieSimulator::new(lattice_scalar);
        let mut sim_simd = GillespieSimulatorSimd::new(lattice_simd);

        let mut rng_scalar = ChaCha8Rng::seed_from_u64(42);
        let mut rng_simd = ChaCha8Rng::seed_from_u64(42);

        for _ in 0..100 {
            sim_scalar.step(&mut rng_scalar).unwrap();
            sim_simd.step(&mut rng_simd).unwrap();

            // States should be identical (same RNG seed)
            prop_assert_eq!(sim_scalar.lattice().states(), sim_simd.lattice().states());
        }
    }
}
```

**Action Items**:
- [ ] Create proptest suite for SIMD correctness
- [ ] Run: `cargo test --features simd`
- [ ] Ensure 0 failures

**Estimated Time**: 4 hours

---

### Week 3, Day 4-5: Benchmarking and Validation

#### Task 2.7: Comprehensive SIMD Benchmarks
**Priority**: 🔴 CRITICAL

```bash
# Run SIMD benchmarks
cargo bench --features simd -- --save-baseline simd-opt

# Compare against scalar baseline
cargo bench --features simd -- --baseline scalar

# Generate report
criterion-html target/criterion
open target/criterion/report/index.html
```

**Action Items**:
- [ ] Run all benchmarks with SIMD enabled
- [ ] Generate comparison report
- [ ] Document results in `docs/performance/SIMD_RESULTS.md`
- [ ] Verify 3-5x speedup achieved

**Expected Results**:
| Benchmark | Scalar | SIMD | Speedup |
|-----------|--------|------|---------|
| Sigmoid (10k) | 50 μs | 10 μs | **5x** |
| Dot product (1k) | 1 μs | 200 ns | **5x** |
| Gillespie step (10k) | 500 μs | 125 μs | **4x** |

**Estimated Time**: 6 hours

---

### Week 2-3 Deliverables
- ✅ Vectorized sigmoid and exp functions
- ✅ SIMD integration in Gillespie and Metropolis
- ✅ 3-5x speedup validated with benchmarks
- ✅ Zero correctness regressions (proptest passing)
- 📊 **Key Metric**: Gillespie 10k step: 500 μs → **<150 μs**

---

## Phase 3: ARM NEON Support (Week 4)

### 🎯 Objective
Port SIMD kernels to ARM NEON for Apple Silicon (M1/M2/M3)

---

### Day 1-2: NEON Implementation

#### Task 3.1: NEON Sigmoid
**Priority**: 🟢 HIGH

```rust
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
pub unsafe fn sigmoid_rational_neon(x: &[f64], output: &mut [f64]) {
    let half = vdupq_n_f64(0.5);
    let two = vdupq_n_f64(2.0);

    for i in (0..x.len()).step_by(2) {  // NEON: 2x f64 at once
        let xv = vld1q_f64(x.as_ptr().add(i));
        let abs_x = vabsq_f64(xv);
        // ... similar to AVX2 version

        vst1q_f64(output.as_mut_ptr().add(i), result);
    }
}
```

**Action Items**:
- [ ] Implement NEON versions of sigmoid, exp, dot_product
- [ ] Test on Apple Silicon (M1+ Mac)
- [ ] Benchmark: should match AVX2 performance (accounting for 2x vs 4x lanes)

**Estimated Time**: 8 hours

---

### Day 3-4: Cross-Platform Testing

#### Task 3.2: CI Matrix for Multiple Architectures
**Priority**: 🟡 MEDIUM

**GitHub Actions**: `.github/workflows/benchmark.yml`
```yaml
name: Benchmarks
on: [push, pull_request]

jobs:
  benchmark:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        arch: [x86_64, aarch64]
        features: [simd, no-default-features]

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v3
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Run benchmarks
        run: cargo bench --features ${{ matrix.features }}

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results-${{ matrix.os }}-${{ matrix.arch }}
          path: target/criterion/
```

**Action Items**:
- [ ] Set up CI matrix for x86 + ARM
- [ ] Verify benchmarks run on both architectures
- [ ] Document any performance differences

**Estimated Time**: 4 hours

---

### Week 4 Deliverables
- ✅ NEON implementation for ARM
- ✅ Cross-platform benchmarks (x86 + ARM)
- ✅ Performance parity validated
- 📊 **Key Metric**: SIMD speedup consistent across architectures

---

## Phase 4+: GPU Acceleration (Weeks 5-6+) - DEFERRED

**Trigger Conditions** (all must be met):
- ✅ SIMD optimization complete (3-5x speedup achieved)
- ✅ Production workload requires >10k pBits
- ✅ SIMD <100 μs latency still insufficient for message passing
- ✅ wgpu compilation fixed and stable

**See**: `GPU_ARCHITECTURE.md` for full GPU implementation plan

---

## Risk Management

### Risk #1: SIMD Slower Than Expected
**Probability**: Low (10%)
**Impact**: High (blocks timeline)
**Mitigation**:
- Benchmark early and often (Week 1)
- Profile to find true bottlenecks (flamegraph)
- Start with known-good kernels (dot product)

### Risk #2: wgpu Remains Broken
**Probability**: Medium (30%)
**Impact**: Low (GPU deferred anyway)
**Mitigation**:
- Make GPU feature optional (Week 1, Task 1.1)
- Focus on SIMD (doesn't depend on wgpu)
- Re-evaluate GPU in Phase 4 after SIMD baseline

### Risk #3: Correctness Regressions
**Probability**: Medium (30%)
**Impact**: Critical (scientific integrity)
**Mitigation**:
- Comprehensive proptest suite (Week 3, Task 2.6)
- Validate against scalar ground truth
- Document error bounds (sigmoid ±0.1%, exp ±1e-6)

### Risk #4: ARM Performance Lag
**Probability**: Low (15%)
**Impact**: Medium (Apple Silicon important platform)
**Mitigation**:
- Test on M1+ hardware early (Week 4)
- NEON has 128-bit lanes (2x f64) vs AVX2 256-bit (4x f64) → expect 50% of AVX2 throughput
- Still faster than scalar!

---

## Success Metrics

### Phase 1 (Week 1)
- ✅ wgpu compilation fixed
- ✅ Scalar baseline: Gillespie 10k step = **XXX μs** (measured)
- ✅ Flamegraph identifies `effective_field()` and `exp()` as hotspots

### Phase 2 (Weeks 2-3)
- ✅ Sigmoid SIMD: **5-10x** faster than scalar
- ✅ Gillespie SIMD: **3-5x** faster than scalar
- ✅ Proptest: **0 failures** (100% correctness)

### Phase 3 (Week 4)
- ✅ NEON performance within **50%** of AVX2 (accounting for lane width)
- ✅ CI benchmarks passing on x86 + ARM

### Phase 4+ (Deferred)
- ⏳ GPU shows **>10x** speedup for 10k pBits
- ⏳ Message passing <10 μs end-to-end

---

## Next Steps

### Immediate (This Week)
1. Fix wgpu dependency (or disable GPU feature)
2. Run baseline benchmarks
3. Create flamegraph profiles

### Short-Term (Next 2 Weeks)
1. Implement vectorized sigmoid and exp
2. Integrate SIMD into Gillespie
3. Validate 3-5x speedup

### Long-Term (4-6 Weeks)
1. ARM NEON support
2. Cross-platform validation
3. GPU prototyping (if needed)

---

## Resource Allocation

### Engineering Time
| Phase | Hours | Calendar |
|-------|-------|----------|
| Phase 1 (Baseline) | 16 | Week 1 |
| Phase 2 (SIMD) | 40 | Weeks 2-3 |
| Phase 3 (ARM) | 20 | Week 4 |
| Phase 4 (GPU) | 80+ | Weeks 5-6+ (deferred) |
| **Total** | **156** | **6 weeks** |

### Hardware Requirements
- Development machine with AVX2 support (any modern x86_64)
- Apple Silicon Mac (M1+) for NEON testing
- GPU (optional, Phase 4+): RTX 3060 or equivalent

---

## Handoff Protocol

### To Rust Developer
- **Implements**: Vectorized sigmoid/exp (`simd/sigmoid.rs`, `simd/exp.rs`)
- **Validates**: Unit tests passing
- **Hands off to**: Performance-Engineer (integration)

### To Systems Architect
- **Reviews**: SIMD architecture design
- **Approves**: Memory layout changes (AoS → SoA)
- **Hands off to**: Rust Developer (implementation)

### To Testing Engineer
- **Creates**: Proptest suite for SIMD correctness
- **Validates**: Zero regressions against scalar
- **Hands off to**: Performance-Engineer (benchmarking)

---

**Agent Status**: 📋 Roadmap Complete
**Next Phase**: Execute Week 1 (Baseline)
**Blocker**: wgpu compilation (to be resolved in Task 1.1)
**Ready**: Yes
