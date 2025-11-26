# Compilation Fixes Required for Profiling

**Priority:** 🔴 CRITICAL - Blocking all profiling work

## Quick Fix Summary

Total errors: 34 compiler errors, 18 warnings
Estimated fix time: 30 minutes

## Fix 1: Remove Module File Conflict

**Error:**
```
error[E0761]: file for module `models` found at both
  "crates/neuro-divergent/src/models.rs" and
  "crates/neuro-divergent/src/models/mod.rs"
```

**Solution:**
```bash
rm /workspaces/neural-trader/neural-trader-rust/crates/neuro-divergent/src/models.rs
```

**Verification:**
```bash
# Should only show models/ directory
ls -la /workspaces/neural-trader/neural-trader-rust/crates/neuro-divergent/src/ | grep models
```

## Fix 2: Add Missing Error Variants

**File:** `/workspaces/neural-trader/neural-trader-rust/crates/neuro-divergent/src/error.rs`

**Add after line 24:**
```rust
    #[error("Training operation error: {0}")]
    Training(String),

    #[error("Optimization operation error: {0}")]
    Optimization(String),
```

**Full error enum should be:**
```rust
#[derive(Error, Debug)]
pub enum NeuroDivergentError {
    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Training error: {0}")]
    TrainingError(String),

    #[error("Training operation error: {0}")]  // NEW
    Training(String),                           // NEW

    #[error("Data error: {0}")]
    DataError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Optimization operation error: {0}")]  // NEW
    Optimization(String),                           // NEW

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    // ... rest of variants
}
```

## Fix 3: Add TrainingMetrics Type

**File:** `/workspaces/neural-trader/neural-trader-rust/crates/neuro-divergent/src/training/metrics.rs`

**Add after the existing functions:**
```rust
/// Training metrics collected during model training
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingMetrics {
    /// Training loss history
    pub train_loss: Vec<f64>,

    /// Validation loss history (if available)
    pub val_loss: Option<Vec<f64>>,

    /// Training MAE history
    pub train_mae: Option<Vec<f64>>,

    /// Validation MAE history
    pub val_mae: Option<Vec<f64>>,

    /// Training RMSE history
    pub train_rmse: Option<Vec<f64>>,

    /// Validation RMSE history
    pub val_rmse: Option<Vec<f64>>,

    /// Best epoch (lowest validation loss)
    pub best_epoch: Option<usize>,

    /// Total training time (seconds)
    pub training_time: f64,

    /// Number of epochs completed
    pub epochs_completed: usize,

    /// Final learning rate
    pub final_lr: Option<f64>,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            train_loss: Vec::new(),
            val_loss: None,
            train_mae: None,
            val_mae: None,
            train_rmse: None,
            val_rmse: None,
            best_epoch: None,
            training_time: 0.0,
            epochs_completed: 0,
            final_lr: None,
        }
    }
}

impl TrainingMetrics {
    /// Create new empty metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record training metrics for an epoch
    pub fn record_epoch(&mut self, train_loss: f64, val_loss: Option<f64>) {
        self.train_loss.push(train_loss);

        if let Some(vl) = val_loss {
            self.val_loss.get_or_insert_with(Vec::new).push(vl);
        }

        self.epochs_completed += 1;
    }

    /// Get best validation loss and epoch
    pub fn best_val_loss(&self) -> Option<(usize, f64)> {
        self.val_loss.as_ref().and_then(|losses| {
            losses.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, &loss)| (idx, loss))
        })
    }

    /// Get final training loss
    pub fn final_train_loss(&self) -> Option<f64> {
        self.train_loss.last().copied()
    }

    /// Get final validation loss
    pub fn final_val_loss(&self) -> Option<f64> {
        self.val_loss.as_ref().and_then(|v| v.last().copied())
    }
}
```

## Fix 4: Fix TimeSeriesDataFrame API Usage

**File:** `/workspaces/neural-trader/neural-trader-rust/crates/neuro-divergent/src/training/trainer.rs`

**Line 420 - Change from:**
```rust
let values = data.values();
```

**To:**
```rust
let values = data.values.clone();  // or use &data.values if possible
```

**Alternative (if values is private):**
Check the TimeSeriesDataFrame API and use the correct method/field accessor.

## Fix 5: Fix Mutable Borrow Issue

**File:** `/workspaces/neural-trader/neural-trader-rust/crates/neuro-divergent/src/training/trainer.rs`

**Line 236 - Change from:**
```rust
let val_loader = x_val.map(|x| {
```

**To:**
```rust
let mut val_loader = x_val.map(|x| {
```

## Fix 6: Remove Unused Variables (Optional - Warnings)

These are warnings, not errors, but fixing improves code quality:

**src/training/trainer.rs:219**
```rust
// Change from:
mut set_params_fn: impl FnMut(Vec<Array2<f64>>),

// To:
_set_params_fn: impl FnMut(Vec<Array2<f64>>),  // Prefix with _ if truly unused
```

**src/optimizations/flash_attention.rs**
```rust
// Line 89: Remove unused d_k
let (batch_size, seq_len, _d_k) = q.dim();

// Line 231: Remove unused scale_vec (or use it)
// This variable appears to be intended for use but not used

// Line 336: Remove unused parameters
pub fn memory_usage(&self, _seq_len: usize, _d_k: usize, d_v: usize, batch_size: usize) -> usize {
```

## Fix 7: Remove Unused Imports (Optional - Warnings)

Clean up unused imports to reduce warnings (18 total).

## Verification Steps

After applying all fixes:

```bash
# 1. Clean build
cd /workspaces/neural-trader/neural-trader-rust
cargo clean

# 2. Build with all features
cargo build --release --package neuro-divergent --all-features

# 3. Run tests
cargo test --package neuro-divergent

# 4. Run clippy for additional warnings
cargo clippy --package neuro-divergent --all-features

# 5. Check benchmarks compile
cargo bench --package neuro-divergent --no-run
```

## Expected Build Output

**Success indicators:**
```
   Compiling neuro-divergent v2.1.0 (/workspaces/neural-trader/neural-trader-rust/crates/neuro-divergent)
    Finished release [optimized] target(s) in X.XXs
```

**No more errors:**
- 0 errors (was 34)
- 0-5 warnings (was 18, after cleanup)

## After Successful Build

Once compilation succeeds, proceed with:

1. **Build with profiling symbols:**
   ```bash
   CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release \
     --package neuro-divergent --features "simd"
   ```

2. **Run benchmarks:**
   ```bash
   cargo bench --package neuro-divergent --all-features
   ```

3. **Profile with perf:**
   ```bash
   perf record -g cargo bench --bench simd_benchmarks
   cargo flamegraph --bench simd_benchmarks
   ```

4. **Analyze results** and proceed with optimization work.

## File Checklist

- [ ] `/src/models.rs` removed
- [ ] `/src/error.rs` - Added `Training` and `Optimization` variants
- [ ] `/src/training/metrics.rs` - Added `TrainingMetrics` struct
- [ ] `/src/training/trainer.rs` - Fixed `values()` → `values`
- [ ] `/src/training/trainer.rs` - Made `val_loader` mutable
- [ ] (Optional) Fixed unused variable warnings
- [ ] (Optional) Removed unused imports
- [ ] Verified build succeeds
- [ ] Verified tests pass
- [ ] Benchmarks compile and run

## Timeline

| Step | Duration | Status |
|------|----------|--------|
| Fix 1: Remove models.rs | 1 min | ⏳ Pending |
| Fix 2: Add error variants | 5 min | ⏳ Pending |
| Fix 3: Add TrainingMetrics | 10 min | ⏳ Pending |
| Fix 4: Fix values() call | 2 min | ⏳ Pending |
| Fix 5: Fix val_loader | 1 min | ⏳ Pending |
| Fix 6-7: Cleanup warnings | 5 min | ⏳ Optional |
| Build verification | 5 min | ⏳ Pending |
| **Total** | **30 min** | ⏳ Pending |

---

**Next Steps:**
1. Apply all fixes listed above
2. Run verification steps
3. Proceed with profiling workflow from `PROFILING_ANALYSIS_REPORT.md`
