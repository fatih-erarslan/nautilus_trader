# Quick Fixes Required - Neural Crate CPU

## Critical Issues (Fix Immediately)

### 1. Add SmallVec Dependency

**File**: `neural-trader-rust/crates/neural/Cargo.toml`

```toml
[dependencies]
smallvec = "1.15"  # Add this line
```

### 2. Fix Missing PI Constant

**File**: `neural-trader-rust/crates/neural/src/models/nbeats.rs`

Add at top:
```rust
use std::f64::consts::PI;
```

**File**: `neural-trader-rust/crates/neural/src/models/prophet.rs`

Add at top:
```rust
use std::f64::consts::PI;
```

### 3. Add SIMD Feature to Cargo.toml

**File**: `neural-trader-rust/crates/neural/Cargo.toml`

```toml
[features]
default = []
simd = []  # Add this line
candle = ["candle-core", "candle-nn"]
cuda = ["candle", "cudarc", "candle-core/cuda"]
metal = ["candle", "candle-core/metal"]
accelerate = ["candle", "candle-core/accelerate"]
```

### 4. Fix Division by Zero

**File**: `neural-trader-rust/crates/neural/src/utils/preprocessing.rs`

Line ~35, change:
```rust
// OLD (unsafe):
let std_val = params.std;

// NEW (safe):
let std_val = if params.std > 1e-8 { params.std } else { 1.0 };
```

### 5. Fix Unused Imports

**File**: `neural-trader-rust/crates/neural/src/training/cpu_trainer.rs`

Remove:
```rust
use crate::error::NeuralError;  // Remove
use ndarray::{Array1, Array2, Array3, Axis};  // Remove Array3, Axis
use std::path::Path;  // Remove
```

**File**: `neural-trader-rust/crates/neural/src/utils/synthetic.rs`

Remove:
```rust
use rand::Rng;  // Remove (unused)
```

**File**: `neural-trader-rust/crates/neural/src/utils/preprocessing_optimized.rs`

Line 9, change:
```rust
// OLD:
use super::memory_pool::{TensorPool, SmallBuffer};

// NEW:
use super::memory_pool::TensorPool;
```

Line 310, add import:
```rust
use std::borrow::Cow;
```

### 6. Fix Lifetime Annotations

**File**: `neural-trader-rust/crates/neural/src/utils/preprocessing_optimized.rs`

Lines 221 and 241, change:
```rust
// OLD:
pub fn maybe_normalize(data: &[f64]) -> Cow<[f64]> {

// NEW:
pub fn maybe_normalize(data: &[f64]) -> Cow<'_, [f64]> {
```

## After Applying Fixes

```bash
# Clean workspace
cd /workspaces/neural-trader/neural-trader-rust
cargo clean

# Build library
cargo build --package nt-neural --lib

# Run tests
cargo test --package nt-neural --lib

# Run simple training example
cargo run --release --example cpu_train_simple
```

## Expected Results

✅ Zero compilation errors  
✅ Zero warnings (after running cargo fix)  
✅ 42/42 tests passing  
✅ SimpleMLP trains in <10 seconds  

