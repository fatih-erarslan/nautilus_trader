# Detailed Remediation Plan - Fix All 206 Errors

## Priority 1: Quick Wins (13 errors fixed immediately)

### Fix ndarray s! macro imports
```bash
# Add to these files:
echo "use ndarray::s;" >> src/ml/nhits/financial/price_prediction.rs
```

Files needing this import:
- `src/ml/nhits/financial/price_prediction.rs` - Lines 472, 479, 481, 528, 537, 543, etc.

## Priority 2: Test File Reorganization (146 errors fixed)

### Move test files to proper location
```bash
# Create proper test structure
mkdir -p tests/ml/nhits/{unit,integration,property,benchmarks}

# Move test files OUT of src
mv src/ml/nhits/tests/unit_tests.rs tests/ml/nhits/unit/
mv src/ml/nhits/tests/integration_tests.rs tests/ml/nhits/integration/
mv src/ml/nhits/tests/property_tests.rs tests/ml/nhits/property/
mv src/ml/nhits/tests/benchmarks.rs benches/nhits_benchmarks.rs
```

### Update Cargo.toml for benchmarks
```toml
[[bench]]
name = "nhits_benchmarks"
harness = false
path = "benches/nhits_benchmarks.rs"
```

## Priority 3: Fix Type Name Casing (25 errors fixed)

### Global search and replace
```bash
# Fix all instances of wrong casing
find src -name "*.rs" -exec sed -i 's/NHiTSConfig/NHITSConfig/g' {} \;
find src -name "*.rs" -exec sed -i 's/NHiTSModel/NHITSModel/g' {} \;
```

## Priority 4: Add Missing Imports (22 errors fixed)

### Standard library imports
Add to files missing HashMap, VecDeque, etc:
```rust
use std::collections::HashMap;
use std::collections::VecDeque;
```

Files needing fixes:
- `src/ml/nhits/tests/property_tests.rs:540` - HashMap
- `src/ml/nhits/optimization/benchmarking.rs:4` - VecDeque removed but imported

## Priority 5: Create Missing API Module

### Create the API module structure
```bash
mkdir -p src/api
```

Create `src/api/mod.rs`:
```rust
pub mod nhits_api;
```

Create `src/api/nhits_api.rs`:
```rust
use serde::{Deserialize, Serialize};
use crate::ml::nhits::model::NHITSModel;

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingRequest {
    pub data: Vec<f32>,
    pub config: NHITSConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionRequest {
    pub data: Vec<f32>,
    pub steps: usize,
}

pub struct NHITSService {
    model: NHITSModel,
}

impl NHITSService {
    pub fn new(config: NHITSConfig) -> Self {
        Self {
            model: NHITSModel::new(config),
        }
    }
    
    pub async fn train(&mut self, request: TrainingRequest) -> Result<(), String> {
        // Implementation
        Ok(())
    }
    
    pub async fn predict(&self, request: PredictionRequest) -> Result<Vec<f32>, String> {
        // Implementation
        Ok(vec![])
    }
}
```

## Priority 6: Fix Feature Gates

### Update files using optional dependencies
For files in src/ that use test dependencies:

```rust
// src/ml/nhits/tests/benchmarks.rs
#[cfg(feature = "benchmarks")]
use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(not(feature = "benchmarks"))]
compile_error!("This module requires the 'benchmarks' feature");
```

## Priority 7: Implement Missing Types

### Create missing NHITSModel implementation
In `src/ml/nhits/model.rs`, ensure proper implementation:
```rust
#[derive(Debug, Clone)]
pub struct NHITSModel {
    config: NHITSConfig,
    // ... other fields
}

impl NHITSModel {
    pub fn new(config: NHITSConfig) -> Self {
        Self { config }
    }
    
    // Add missing methods referenced in tests
    pub fn forward(&self, input: &Array3<f32>) -> Result<Array3<f32>, Error> {
        // Implementation
    }
}
```

## Execution Order

1. **Immediate** (5 minutes):
   - Fix ndarray imports
   - Fix type casing

2. **Short-term** (30 minutes):
   - Add missing standard library imports
   - Create API module structure

3. **Medium-term** (2 hours):
   - Reorganize test files
   - Update Cargo.toml
   - Add feature gates

4. **Long-term** (4 hours):
   - Implement missing types
   - Complete API implementations
   - Validate all fixes

## Validation Commands

```bash
# After each phase, run:
cargo check
cargo check --all-features
cargo check --no-default-features

# Final validation:
cargo build --release
cargo test --all-features
cargo bench
```

## Success Criteria
- `cargo check` shows 0 errors
- All 212 warnings resolved
- Tests can run with `cargo test`
- Benchmarks can run with `cargo bench`
- No compilation errors in any feature combination