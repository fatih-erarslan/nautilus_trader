# CPU Optimization Implementation Guide

## Quick Start: Top 5 Optimizations

### 1. Enable SIMD Everywhere (30 minutes, 3-4x speedup)

**File**: `Cargo.toml`
```toml
[features]
default = ["simd"]  # Enable by default
simd = []
```

**Implementation**:
```bash
cd /workspaces/neural-trader/neural-trader-rust/crates/neural
cargo build --features simd
cargo test --features simd
cargo bench --features simd
```

---

### 2. Add Inline Hints (15 minutes, 5-8% speedup)

**Files to modify**:
- `src/utils/preprocessing.rs`
- `src/utils/metrics.rs`
- `src/models/layers.rs`

**Pattern**:
```rust
// Before
pub fn normalize(data: &[f64]) -> (Vec<f64>, NormalizationParams) {
    // ...
}

// After
#[inline]
pub fn normalize(data: &[f64]) -> (Vec<f64>, NormalizationParams) {
    // ...
}

// For small getters
#[inline(always)]
pub fn get_param(&self) -> f64 {
    self.param
}
```

**Command**:
```bash
# Add to 20-30 hot functions
rg "pub fn" src/utils/preprocessing.rs | wc -l  # Find candidates
```

---

### 3. Parallel Data Loading (45 minutes, 2-3x speedup)

**File**: `src/training/data_loader.rs`

**Current Code** (line ~100-150):
```rust
impl DataLoader {
    pub fn next_batch(&mut self, device: &Device) -> Result<Option<(Tensor, Tensor)>> {
        let mut batch_x = Vec::new();
        let mut batch_y = Vec::new();

        for _ in 0..self.batch_size {
            if let Some((x, y)) = self.get_next_sample()? {
                batch_x.push(x);
                batch_y.push(y);
            }
        }
        // ...
    }
}
```

**Optimized**:
```rust
use rayon::prelude::*;

impl DataLoader {
    pub fn next_batch(&mut self, device: &Device) -> Result<Option<(Tensor, Tensor)>> {
        let indices: Vec<usize> = (0..self.batch_size).collect();

        let samples: Result<Vec<_>> = indices
            .par_iter()
            .map(|_| self.get_next_sample())
            .collect();

        let samples = samples?;
        // ... create tensors from samples
    }
}
```

**Testing**:
```bash
cargo test --package nt-neural data_loader
cargo bench --bench neural_benchmarks -- data_loader
```

---

### 4. Memory Pool for Inference (2 hours, 2.5-3x speedup)

**File**: `src/inference/batch.rs`

**Step 1**: Add pool to predictor
```rust
use crate::utils::memory_pool::TensorPool;

pub struct BatchPredictor {
    model: Box<dyn Model>,
    tensor_pool: TensorPool,  // ← Add this
}

impl BatchPredictor {
    pub fn new(model: Box<dyn Model>) -> Self {
        Self {
            model,
            tensor_pool: TensorPool::new(32),  // Pool size = 32 buffers
        }
    }
}
```

**Step 2**: Use pool in predict
```rust
pub fn predict(&mut self, inputs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let batch_size = inputs.len();

    // Get buffer from pool instead of allocating
    let mut input_buffer = self.tensor_pool.get(inputs[0].len() * batch_size);

    // Fill buffer
    for (i, input) in inputs.iter().enumerate() {
        let start = i * input.len();
        input_buffer[start..start + input.len()].copy_from_slice(input);
    }

    // Create tensor from pooled buffer
    let input_tensor = Tensor::from_vec(
        input_buffer.clone(),
        (batch_size, inputs[0].len()),
        &self.device,
    )?;

    let output = self.model.forward(&input_tensor)?;

    // Return buffer to pool
    self.tensor_pool.return_buffer(input_buffer);

    // Convert output
    Ok(/* ... */)
}
```

---

### 5. Reduce Clones in Training (1 hour, 30-40% speedup)

**File**: `src/training/nhits_trainer.rs`

**Pattern to fix**:
```rust
// Before (line ~150)
pub fn train_step(&mut self, x: Tensor, y: Tensor) -> Result<f32> {
    let x = x.clone();  // ❌ Unnecessary
    let y = y.clone();  // ❌ Unnecessary

    let output = self.model.forward(&x)?;
    let loss = self.loss_fn.compute(&output, &y)?;
    // ...
}

// After
pub fn train_step(&mut self, x: &Tensor, y: &Tensor) -> Result<f32> {
    let output = self.model.forward(x)?;  // Use reference
    let loss = self.loss_fn.compute(&output, y)?;
    // ...
}
```

**Find all clones**:
```bash
cd /workspaces/neural-trader/neural-trader-rust/crates/neural
rg "\.clone\(\)" src/training/ src/models/ | grep -v "// OK" | wc -l
```

---

## Verification Commands

### Run Benchmarks
```bash
cd /workspaces/neural-trader/neural-trader-rust

# Quick benchmarks
cargo bench --package nt-neural --bench neural_benchmarks -- --quick

# Full benchmarks
cargo bench --package nt-neural --bench neural_benchmarks

# Specific benchmark
cargo bench --package nt-neural --bench neural_benchmarks -- normalization

# Save baseline
cargo bench --package nt-neural -- --save-baseline before_opt

# After optimization, compare
cargo bench --package nt-neural -- --baseline before_opt
```

### Generate Flamegraph
```bash
# Install
cargo install flamegraph

# Generate for specific benchmark
cd /workspaces/neural-trader/neural-trader-rust
cargo flamegraph --bench neural_benchmarks -- --bench

# Output: flamegraph.svg
# View in browser or VS Code
```

### Cache Analysis (Linux only)
```bash
# Install valgrind
sudo apt-get install valgrind

# Run cache analysis
valgrind --tool=cachegrind \
  --cachegrind-out-file=cachegrind.out \
  /workspaces/neural-trader/neural-trader-rust/target/release/deps/neural_benchmarks-*

# Annotate source with cache stats
cg_annotate cachegrind.out src/utils/preprocessing.rs
```

### Performance Regression Test
```bash
# Create performance test
cat > tests/performance_regression.rs << 'EOF'
#[test]
fn test_inference_performance() {
    let start = std::time::Instant::now();

    // Run 1000 inferences
    for _ in 0..1000 {
        model.forward(&input);
    }

    let elapsed = start.elapsed();

    // Must complete in < 5 seconds
    assert!(elapsed.as_secs() < 5, "Performance regression detected!");
}
EOF

cargo test --release performance_regression
```

---

## Measurement & Validation

### Before Optimization
```bash
# Run baseline benchmarks
cd /workspaces/neural-trader/neural-trader-rust
cargo bench --package nt-neural --bench neural_benchmarks \
  -- --save-baseline baseline_before

# Save results
cp target/criterion/*/baseline_before.json docs/neural/benchmark_before.json
```

### After Each Optimization
```bash
# Run benchmarks
cargo bench --package nt-neural --bench neural_benchmarks \
  -- --baseline baseline_before

# Compare results
cargo benchcmp baseline_before current

# Save if improvement is good
cargo bench --package nt-neural -- --save-baseline optimized_v1
```

### Expected Results

**Baseline** (before optimization):
```
normalize/1000          50 μs
normalize/10000        450 μs
model_forward/32        15 ms
training_step           45 ms
```

**After Opt #1** (SIMD):
```
normalize/1000          12 μs  ← 4.2x speedup
normalize/10000        110 μs  ← 4.1x speedup
model_forward/32        15 ms  ← no change yet
training_step           45 ms  ← no change yet
```

**After Opt #1-3** (SIMD + Inline + Parallel):
```
normalize/1000          12 μs  ← 4.2x
normalize/10000        110 μs  ← 4.1x
model_forward/32        12 ms  ← 1.25x
training_step           38 ms  ← 1.18x
```

**After All Optimizations**:
```
normalize/1000          12 μs  ← 4.2x
normalize/10000        110 μs  ← 4.1x
model_forward/32         6 ms  ← 2.5x
training_step           18 ms  ← 2.5x
```

---

## Safety Checklist

Before each optimization:
- [ ] Run full test suite: `cargo test --all-features`
- [ ] Check clippy: `cargo clippy --all-features`
- [ ] Run benchmarks: `cargo bench`
- [ ] Verify correctness: `cargo test --release`

After each optimization:
- [ ] Run full test suite again
- [ ] Compare benchmark results
- [ ] Check for numerical differences
- [ ] Update documentation
- [ ] Commit changes

---

## Common Pitfalls

### 1. SIMD Without Feature Flag
```rust
// ❌ Wrong - always uses SIMD
use packed_simd::f64x4;

// ✅ Correct - feature-gated
#[cfg(feature = "simd")]
use packed_simd::f64x4;
```

### 2. Memory Pool Without Return
```rust
// ❌ Memory leak
let buffer = pool.get(size);
// ... use buffer ...
// Forgot to return!

// ✅ Correct
let buffer = pool.get(size);
// ... use buffer ...
pool.return_buffer(buffer);  // Return to pool
```

### 3. Parallel Iterator on Small Data
```rust
// ❌ Overhead > benefit for small data
let small_data: Vec<_> = (0..10).into_par_iter().collect();

// ✅ Use parallel only for large data
let large_data: Vec<_> = if data.len() > 1000 {
    data.into_par_iter().collect()
} else {
    data.into_iter().collect()
};
```

---

## Quick Reference

### Performance Targets

| Optimization | Time Investment | Expected Speedup | Difficulty |
|--------------|-----------------|------------------|------------|
| SIMD Enable | 30 min | 3-4x | Easy |
| Inline Hints | 15 min | 5-8% | Easy |
| Parallel Loading | 45 min | 2-3x | Easy |
| Memory Pool | 2 hours | 2-3x | Medium |
| Reduce Clones | 1 hour | 30-40% | Easy |
| Cache Layout | 4 hours | 50-100% | Hard |

### Total Time Investment: ~8 hours
### Total Expected Speedup: **3-5x overall**

---

## Support & Resources

### Documentation
- Rust Performance Book: https://nnethercote.github.io/perf-book/
- Criterion.rs Guide: https://bheisler.github.io/criterion.rs/book/
- Rayon Documentation: https://docs.rs/rayon/

### Tools
- cargo-flamegraph: `cargo install flamegraph`
- cargo-benchcmp: `cargo install cargo-benchcmp`
- perf (Linux): `sudo apt-get install linux-tools-generic`

### Questions?
- Check `/workspaces/neural-trader/docs/neural/CPU_OPTIMIZATION_ANALYSIS.md`
- Run benchmarks: `cargo bench --help`
- Open issue: https://github.com/neural-trader/neural-trader-rust/issues

---

**Ready to optimize? Start with #1 (SIMD) for biggest immediate impact!**
