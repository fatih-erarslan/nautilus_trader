# HyperPhysics Performance Baseline Analysis

**Generated**: 2025-11-12
**Agent**: Performance-Engineer (Queen Seraphina Command)
**Status**: Code Review Complete, Benchmarking Blocked by wgpu Error

---

## Executive Summary

### Current State
- ✅ SIMD infrastructure exists (`crates/hyperphysics-pbit/src/simd.rs`)
- ✅ Benchmark harness ready (`benches/message_passing.rs`)
- ❌ **BLOCKER**: wgpu v0.19.4 compilation failure prevents benchmark execution
- 🔄 GPU features premature (need CPU/SIMD baseline first)

### Critical Path Forward
1. **IMMEDIATE**: Fix wgpu dependency or make GPU optional
2. **Phase 1**: Establish CPU baseline (scalar code)
3. **Phase 2**: Optimize with SIMD (AVX2/NEON)
4. **Phase 3**: Prototype GPU acceleration (after SIMD proves 3-5x gains)

---

## 1. Identified Performance Bottlenecks

### 1.1 Gillespie Algorithm Hot Paths
**File**: `crates/hyperphysics-pbit/src/gillespie.rs`

```rust
// BOTTLENECK #1: O(N) propensity calculation per step (lines 60-72)
for (_i, pbit) in self.lattice.pbits().iter().enumerate() {
    let h_eff = pbit.effective_field(&states);  // ❌ Pointer chasing
    let mut temp_pbit = pbit.clone();           // ❌ Clone overhead
    temp_pbit.update_probability(h_eff);
    let rate = temp_pbit.flip_rate();
    rates.push(rate);  // ❌ Vec reallocation possible
    total_rate += rate;
}
```

**Impact**:
- Executed **every simulation step**
- Dominates runtime for large lattices (>1000 pBits)
- Clone operation allocates memory unnecessarily

**SIMD Opportunity**: ✅ **HIGH**
- Effective field calculation is embarrassingly parallel
- Probability updates use `sigmoid(h/T)` → vectorizable
- Rate accumulation is a reduction (horizontal sum)

**Estimated Speedup**: 4-8x with AVX2 (8x f32 lanes)

---

### 1.2 Metropolis Energy Calculations
**File**: `crates/hyperphysics-pbit/src/metropolis.rs`

```rust
// BOTTLENECK #2: Energy change calculation (lines 89-96)
fn energy_change(&self, idx: usize, states: &[bool]) -> f64 {
    let pbit = &self.lattice.pbits()[idx];
    let h_eff = pbit.effective_field(states);  // ❌ Same issue as Gillespie
    let si = pbit.spin();
    2.0 * h_eff * si
}
```

**Impact**:
- Called **every Metropolis step** (potentially 10k-100k steps/equilibration)
- `effective_field()` likely does dot product with coupling matrix

**SIMD Opportunity**: ✅ **HIGH**
- Dot product is textbook SIMD use case
- Already implemented in `simd.rs::dot_product_avx2()` but NOT USED

**Estimated Speedup**: 3-6x with AVX2 dot product

---

### 1.3 Memory Allocation Patterns
**Current Benchmark**: `benches/message_passing.rs` (lines 76-107)

```rust
// INEFFICIENCY: Vec allocation without capacity (line 88)
let mut v = Vec::new();  // ❌ Multiple reallocations
for i in 0..1000 {
    v.push(i as f64);
}

// vs OPTIMIZED (line 98)
let mut v = Vec::with_capacity(1000);  // ✅ Single allocation
```

**Measured in Gillespie**:
```rust
let mut rates = Vec::with_capacity(self.lattice.size());  // ✅ Good!
```

**Status**: ✅ Lattice code already optimized. Benchmark needed to quantify gains.

---

### 1.4 Parallelization Opportunities

**Current State**: All code is **single-threaded**
- `rayon` dependency exists but **NOT USED**
- Gillespie is inherently sequential (event-driven)
- Metropolis can parallelize multiple chains (replica exchange)

**SIMD vs Thread Parallelism**:
| Approach | Latency Target | Use Case |
|----------|---------------|----------|
| SIMD | <50 μs | Single lattice update |
| Rayon threads | 1-10 ms | Batch processing, ensemble averages |
| GPU | <5 μs | Large lattices (>10k pBits) |

**Recommendation**: Start with SIMD (lowest overhead), add threading later.

---

## 2. SIMD Implementation Status

### 2.1 Existing SIMD Code
**File**: `crates/hyperphysics-pbit/src/simd.rs`

✅ **Implemented**:
- AVX2 state updates (4x f64 parallelism)
- AVX2 dot product (4x f64)
- Portable fallback for non-x86

❌ **Missing**:
- Fast vectorized `exp()` for Boltzmann factors (line 74, TODO)
- ARM NEON implementation (compile-time checks exist, no code)
- AVX-512 support (16x f32 lanes)
- Integration with Gillespie/Metropolis (SIMD code not called!)

### 2.2 Critical Gap: Vectorized Exponential
```rust
// TODO: Implement Remez polynomial approximation for vectorized exp
// Current: Falls back to scalar exp() (line 75)
for i in 0..len {
    result[i] = x[i].exp();  // ❌ NOT VECTORIZED
}
```

**Sigmoid Function** (used in probability updates):
```rust
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())  // ❌ Scalar exp dominates
}
```

**Solution**: Use Agner Fog's fast exp approximation or Intel SVML intrinsics.

**Impact**: Vectorized exp can improve Gillespie by **2-4x** alone.

---

### 2.3 Integration Issue
**Problem**: SIMD functions exist but Gillespie/Metropolis don't call them!

**Evidence**:
```bash
$ rg "SimdOps" crates/hyperphysics-pbit/src/
# Only appears in simd.rs tests, NOT in gillespie.rs or metropolis.rs
```

**Recommendation**: Refactor Gillespie to use `SimdOps::dot_product()` for effective fields.

---

## 3. GPU Acceleration Readiness

### 3.1 Dependency Blocker
```toml
# Cargo.toml
wgpu = { version = "0.19.4", optional = true }  # ❌ Compilation fails
```

**Error**:
```
error: macro expansion ends with an incomplete expression
   --> wgpu-0.19.4/src/backend/wgpu_core.rs:783:92
```

**Root Cause**: wgpu v0.19.4 incompatible with current Rust nightly (likely)

**Fix Options**:
1. Update to wgpu v0.20+ (latest stable)
2. Downgrade Rust toolchain to match wgpu v0.19 era
3. Remove GPU feature until SIMD baseline established

**Recommendation**: **Option 3** - Delay GPU, focus on CPU/SIMD first

---

### 3.2 GPU Architecture Analysis (From Code)
No GPU kernels found. All GPU infrastructure is placeholder.

**Required Work** (80-120 hours):
1. Design wgpu compute shaders for Gillespie propensities
2. Implement efficient GPU↔CPU memory transfers
3. Batch processing to amortize transfer overhead
4. Benchmark to prove >10x speedup over SIMD

**Reality Check**: GPU only worth it for **>10k pBits** (transfer overhead)

---

## 4. Hottest Code Paths (Profiling Needed)

### Predicted Hotspots
Based on algorithmic complexity:

| Function | File | Complexity | Calls/Sec | Est. % Runtime |
|----------|------|-----------|-----------|----------------|
| `effective_field()` | pbit.rs | O(N) | 10k-100k | **40-60%** |
| `sigmoid/exp()` | metropolis.rs | O(1) | 10k-100k | **20-30%** |
| `Vec::push()` | gillespie.rs | O(1) amortized | 10k | **5-10%** |
| RNG sampling | gillespie.rs | O(1) | 10k | **5-10%** |

**Verification Needed**:
```bash
# TODO: Run flamegraph after fixing wgpu
cargo flamegraph --bench message_passing
```

---

## 5. Recommended Optimization Roadmap

### Phase 1: SIMD Integration (Week 1)
- [ ] Fix wgpu compilation or disable GPU feature
- [ ] Run existing benchmarks to establish scalar baseline
- [ ] Integrate `SimdOps::dot_product()` into Gillespie/Metropolis
- [ ] Benchmark SIMD vs scalar (target: 3-5x speedup)

### Phase 2: Vectorized Math (Week 2)
- [ ] Implement fast vectorized `exp()` using Agner Fog's library
- [ ] Replace scalar `sigmoid()` with SIMD version
- [ ] Benchmark probability updates (target: 4-8x speedup)
- [ ] Profile with `cargo flamegraph` to find remaining hotspots

### Phase 3: ARM NEON (Week 3)
- [ ] Implement NEON equivalents of AVX2 kernels
- [ ] Test on Apple Silicon (M1/M2/M3)
- [ ] Benchmark ARM vs x86 performance parity

### Phase 4: GPU Prototype (Weeks 4-5) - ONLY if SIMD proves insufficient
- [ ] Update wgpu to v0.20+
- [ ] Write compute shader for Gillespie propensities
- [ ] Implement async memory transfers
- [ ] Benchmark 10k pBit lattice (target: >10x vs SIMD)

---

## 6. Performance Targets

### Latency Targets (from Mission Brief)

| Operation | Current (Est.) | SIMD Target | GPU Target |
|-----------|---------------|-------------|------------|
| Gillespie step (10k) | **500 μs** | 100 μs | 10 μs |
| Metropolis step (10k) | **500 μs** | 100 μs | 10 μs |
| Effective field calc | **100 μs** | 20 μs | 5 μs |
| Message passing | **Unknown** | N/A | <50 μs |

**Critical**: Message passing <50 μs required for real-time coordination.

---

## 7. Next Steps

### Immediate (This Session)
1. ✅ Document performance analysis (this file)
2. ⏳ Create SIMD optimization strategy
3. ⏳ Create GPU architecture plan
4. ⏳ Create implementation roadmap

### Next Session (Performance Engineer)
1. Fix wgpu dependency or disable GPU feature
2. Run `cargo bench` to get actual numbers
3. Profile with `cargo flamegraph`
4. Implement SIMD integration (Phase 1)

### Handoff to Other Agents
- **Systems Architect**: Review SIMD/GPU architecture
- **Rust Developer**: Implement vectorized exp()
- **Testing Engineer**: Validate SIMD correctness (proptest)

---

## 8. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| SIMD slower than scalar | Low | High | Benchmark before committing |
| GPU overhead > gains | Medium | Medium | Only use for >10k pBits |
| wgpu incompatibility | **High** | Low | Use v0.20+ or disable |
| ARM performance lag | Low | Medium | Test on Apple Silicon early |

---

## Appendix: Benchmark Commands

```bash
# Fix wgpu first
cargo update wgpu --precise 0.20.0  # or latest

# Run benchmarks
cargo bench --workspace -- --save-baseline before-simd

# Profile hotspots
cargo install flamegraph
cargo flamegraph --bench message_passing

# SIMD-specific benchmarks (after integration)
cargo bench --features simd -- --save-baseline simd-optimized

# Compare baselines
cargo bench --features simd -- --baseline before-simd
```

---

**Agent Status**: 📊 Analysis Complete
**Blocker**: wgpu compilation error
**Recommendation**: Disable GPU, focus on SIMD
**Next Agent**: Rust Developer (implement vectorized exp)
