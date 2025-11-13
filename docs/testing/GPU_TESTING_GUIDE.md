# GPU Testing & Validation Guide

## Overview

This document describes the comprehensive test suite for validating the HyperPhysics GPU compute pipeline against CPU reference implementations.

## Test Files

### Integration Tests: `/crates/hyperphysics-gpu/tests/integration_tests.rs`

**Purpose**: Validate GPU pipeline correctness end-to-end

**Coverage**: 11 comprehensive test cases

### Performance Benchmarks: `/crates/hyperphysics-gpu/benches/gpu_benchmarks.rs`

**Purpose**: Measure and validate 10-800× speedup claims

**Coverage**: 7 benchmark groups with multiple problem sizes

---

## Integration Test Suite

### Test 1: GPU Executor Initialization
```rust
#[tokio::test]
async fn test_gpu_executor_initialization()
```

**Validates**:
- GPU executor can be created successfully
- 48-pBit test lattice with nearest-neighbor couplings

**Expected**: Successful initialization without errors

---

### Test 2: WGPU Backend Initialization
```rust
#[tokio::test]
async fn test_wgpu_backend_initialization()
```

**Validates**:
- WGPU backend initialization
- GPU compute shader support
- Minimum buffer size (≥1MB)
- Minimum workgroup size (≥256)

**Expected**:
- `supports_compute = true`
- Prints GPU device name and capabilities

---

### Test 3: GPU Energy vs CPU Validation
```rust
#[tokio::test]
async fn test_gpu_energy_vs_cpu()
```

**Validates**:
- Ising energy calculation: `E = -Σ J_ij s_i s_j`
- GPU result matches CPU reference implementation

**Acceptance Criteria**:
- Relative error < 1e-5 (0.001%)

**CPU Reference**:
```rust
fn cpu_compute_energy(states: &[u32], couplings: &[(usize, usize, f64)]) -> f64 {
    let mut energy = 0.0;
    for &(i, j, strength) in couplings.iter() {
        if i < j {
            let spin_i = (states[i] as f64) * 2.0 - 1.0;
            let spin_j = (states[j] as f64) * 2.0 - 1.0;
            energy -= strength * spin_i * spin_j;
        }
    }
    energy
}
```

---

### Test 4: GPU Entropy vs CPU Validation
```rust
#[tokio::test]
async fn test_gpu_entropy_vs_cpu()
```

**Validates**:
- Shannon entropy calculation: `S = -Σ [p ln(p) + (1-p) ln(1-p)]`
- GPU result matches CPU reference

**Acceptance Criteria**:
- Relative error < 1e-5 (0.001%)

**CPU Reference**:
```rust
fn cpu_compute_entropy(states: &[u32]) -> f64 {
    let mut entropy = 0.0;
    for &state in states.iter() {
        let p = state as f64;
        let q = 1.0 - p;
        if p > 1e-10 { entropy -= p * p.ln(); }
        if q > 1e-10 { entropy -= q * q.ln(); }
    }
    entropy
}
```

---

### Test 5: GPU State Update
```rust
#[tokio::test]
async fn test_gpu_state_update()
```

**Validates**:
- Gillespie stochastic algorithm execution
- pBit state transitions occur

**Expected**:
- At least some states change after one simulation step
- Prints changed state count

---

### Test 6: GPU Double Buffering
```rust
#[tokio::test]
async fn test_gpu_double_buffering()
```

**Validates**:
- Ping-pong buffer swapping works correctly
- Multiple consecutive simulation steps succeed

**Test**: Runs 5 simulation steps without panicking

**Expected**: No buffer access violations or race conditions

---

### Test 7: GPU Bias Update
```rust
#[tokio::test]
async fn test_gpu_bias_update()
```

**Validates**:
- Dynamic bias updates via `update_biases()`
- Bias influences state distribution

**Test**:
- Apply strong positive bias (10.0) to all pBits
- Run 20 steps at low temperature (T=0.1)

**Expected**:
- ≥70% of pBits in state = 1 (bias-favored state)

---

### Test 8: Ferromagnetic Ordering
```rust
#[tokio::test]
async fn test_gpu_ferromagnetic_ordering()
```

**Validates**:
- Physics correctness: ferromagnetic coupling favors alignment
- System reaches equilibrium

**Test**:
- Ferromagnetic nearest-neighbor couplings (J=1.0)
- Low temperature (T=0.5)
- 100 equilibration steps

**Expected**:
- ≥60% of neighboring pBits have aligned spins

**Physics**:
- Ferromagnetic: J > 0 favors parallel spins (↑↑ or ↓↓)
- Low T → system minimizes energy → alignment

---

### Test 9: Energy Conservation
```rust
#[tokio::test]
async fn test_gpu_energy_conservation()
```

**Validates**:
- Energy dynamics follow physics
- At low temperature, energy decreases or stabilizes

**Test**:
- Very low temperature (T=0.1)
- 50 simulation steps
- Compare initial vs final energy

**Expected**:
- Final energy ≤ initial energy + tolerance (1.0)

**Physics**:
- Low T → system explores low-energy configurations
- Energy should decrease toward minimum

---

### Test 10: Async Readback Performance
```rust
#[tokio::test]
async fn test_gpu_async_readback()
```

**Validates**:
- Async GPU→CPU transfer doesn't block
- Readback completes quickly

**Expected**:
- Readback duration < 100ms
- Correct number of states returned

---

### Test 11: Large Lattice Performance
```rust
#[tokio::test]
#[ignore] // Only run for performance profiling
async fn test_gpu_large_lattice()
```

**Purpose**: Verify GPU advantage at scale

**Test**:
- 10,000 pBit lattice
- 10 simulation steps
- Energy + entropy computation

**Metrics Collected**:
- Initialization time
- Average time per simulation step
- Observable computation time

**Run Manually**:
```bash
cargo test --release test_gpu_large_lattice -- --ignored --nocapture
```

---

## Performance Benchmark Suite

### Benchmark Groups

#### 1. pBit Update CPU
```rust
fn bench_pbit_update_cpu(c: &mut Criterion)
```

**Sizes**: 48, 1,000, 10,000 pBits

**Measures**: CPU Gillespie algorithm performance

---

#### 2. pBit Update GPU
```rust
fn bench_pbit_update_gpu(c: &mut Criterion)
```

**Sizes**: 48, 1,000, 10,000 pBits

**Measures**: GPU compute shader performance

**Expected Speedup**:
- 48 pBits: 2-5× (overhead dominates)
- 1,000 pBits: 10-50×
- 10,000 pBits: 100-500×

---

#### 3. Energy Calculation CPU
```rust
fn bench_energy_cpu(c: &mut Criterion)
```

**Measures**: CPU energy computation

---

#### 4. Energy Calculation GPU
```rust
fn bench_energy_gpu(c: &mut Criterion)
```

**Measures**: GPU parallel reduction

**Expected Speedup**: 50-100×

---

#### 5. Entropy Calculation CPU
```rust
fn bench_entropy_cpu(c: &mut Criterion)
```

**Measures**: CPU entropy computation

---

#### 6. Entropy Calculation GPU
```rust
fn bench_entropy_gpu(c: &mut Criterion)
```

**Measures**: GPU parallel reduction

**Expected Speedup**: 50-100×

---

#### 7. End-to-End Simulation
```rust
fn bench_end_to_end_simulation(c: &mut Criterion)
```

**Sizes**: 1,000, 10,000 pBits

**Workflow**:
1. 10 simulation steps
2. Compute energy
3. Compute entropy

**Measures**: Complete simulation cycle performance

**Sample Size**: 10 (fewer due to long runtime)

---

## Running Tests

### Integration Tests

```bash
# Run all tests
cargo test --package hyperphysics-gpu

# Run specific test
cargo test --package hyperphysics-gpu test_gpu_energy_vs_cpu

# Run with output
cargo test --package hyperphysics-gpu -- --nocapture

# Run large lattice test (ignored by default)
cargo test --package hyperphysics-gpu test_gpu_large_lattice -- --ignored --nocapture
```

### Performance Benchmarks

```bash
# Run all benchmarks
cargo bench --package hyperphysics-gpu

# Run specific benchmark group
cargo bench --package hyperphysics-gpu --bench gpu_benchmarks pbit_update

# Generate detailed report
cargo bench --package hyperphysics-gpu -- --save-baseline gpu_baseline

# Compare with baseline
cargo bench --package hyperphysics-gpu -- --baseline gpu_baseline
```

---

## Expected Results

### Integration Tests

All 10 tests (excluding large lattice) should **PASS**:

```
test test_gpu_executor_initialization ... ok
test test_wgpu_backend_initialization ... ok
test test_gpu_energy_vs_cpu ... ok
test test_gpu_entropy_vs_cpu ... ok
test test_gpu_state_update ... ok
test test_gpu_double_buffering ... ok
test test_gpu_bias_update ... ok
test test_gpu_ferromagnetic_ordering ... ok
test test_gpu_energy_conservation ... ok
test test_gpu_async_readback ... ok
```

### Benchmark Results (Example)

```
pbit_update_cpu/48          time: [15.2 µs 15.4 µs 15.6 µs]
pbit_update_gpu/48          time: [8.1 µs 8.3 µs 8.5 µs]
                            speedup: 1.85×

pbit_update_cpu/1000        time: [298 µs 302 µs 306 µs]
pbit_update_gpu/1000        time: [12.5 µs 12.8 µs 13.1 µs]
                            speedup: 23.6×

pbit_update_cpu/10000       time: [2.95 ms 3.01 ms 3.07 ms]
pbit_update_gpu/10000       time: [18.2 µs 18.6 µs 19.0 µs]
                            speedup: 161.8×
```

**Note**: Actual speedups depend on GPU hardware

---

## Failure Modes & Debugging

### Test Failure: `test_wgpu_backend_initialization`

**Symptoms**: "GPU initialization failed"

**Causes**:
- No GPU available
- Outdated GPU drivers
- WGPU not compiled with correct features

**Fix**:
```bash
# Verify GPU detection
cargo test test_wgpu_backend_initialization -- --nocapture

# Check WGPU features in Cargo.toml
wgpu = { version = "22", features = ["wgsl"] }
```

---

### Test Failure: Energy/Entropy Mismatch

**Symptoms**: "GPU energy doesn't match CPU energy"

**Debugging**:
1. Check relative error magnitude
2. Verify coupling buffer construction
3. Ensure proper spin mapping: {0,1} → {-1,+1}
4. Check for race conditions in parallel reduction

**Investigate**:
```rust
// Add debug prints in compute_energy()
println!("GPU Energy: {}, CPU Energy: {}", gpu_energy, cpu_energy);
println!("Relative Error: {:.2e}", relative_error);
```

---

### Benchmark Failure: Unexpected Slowdown

**Symptoms**: GPU slower than CPU

**Possible Causes**:
1. **Problem too small**: Overhead dominates (< 1000 pBits)
2. **Memory bandwidth**: GPU→CPU transfer bottleneck
3. **Driver issues**: Outdated or misconfigured
4. **Thermal throttling**: GPU overheating

**Verify**:
```bash
# Profile with larger problem size
cargo bench --bench gpu_benchmarks pbit_update_gpu/10000

# Check GPU utilization (requires tools)
# NVIDIA: nvidia-smi
# AMD: radeontop
# Intel/Apple: Activity Monitor / GPU History
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: GPU Tests
on: [push, pull_request]

jobs:
  test-gpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: dtolnay/rust-toolchain@stable

      # CPU fallback testing
      - name: Run integration tests
        run: cargo test --package hyperphysics-gpu

      # Benchmark compilation check
      - name: Build benchmarks
        run: cargo bench --package hyperphysics-gpu --no-run

      # Large lattice test (optional)
      - name: Run large lattice test
        run: cargo test --package hyperphysics-gpu test_gpu_large_lattice -- --ignored
        continue-on-error: true  # May not have GPU in CI
```

---

## Performance Regression Testing

### Baseline Creation

```bash
# After verified working implementation
cargo bench --package hyperphysics-gpu -- --save-baseline gpu_v1.0
```

### Regression Check

```bash
# After code changes
cargo bench --package hyperphysics-gpu -- --baseline gpu_v1.0

# Review criterion HTML report
open target/criterion/report/index.html
```

### Acceptable Ranges

- **No regression**: Performance within ±5%
- **Minor regression**: 5-10% slower (investigate)
- **Major regression**: >10% slower (block merge)
- **Improvement**: Any speedup (celebrate!)

---

## Next Steps

1. **Run Full Test Suite**: Verify all 10 tests pass
2. **Benchmark Baseline**: Establish performance baseline
3. **Hardware Matrix**: Test on NVIDIA/AMD/Intel/Apple GPUs
4. **Stress Testing**: Scale to 100K+ pBits
5. **Continuous Profiling**: Integrate criterion into CI

---

## References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [WGPU Testing Guide](https://wgpu.rs/doc/wgpu/)
- [Tokio Async Testing](https://tokio.rs/tokio/topics/testing)
