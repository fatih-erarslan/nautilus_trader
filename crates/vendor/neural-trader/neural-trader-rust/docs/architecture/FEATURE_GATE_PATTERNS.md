# Feature Gate Patterns for Rust Crates

## Quick Reference for Agent Coordination

This document provides reusable patterns for implementing optional dependencies with feature gates in Rust crates.

## Pattern 1: Stub Types for Optional Dependencies

### Problem
You want a crate to compile without an optional dependency, but still provide the API surface.

### Solution
Create stub types that mirror the real types but return `NotImplemented` errors.

```rust
// src/stubs.rs (only compiled when feature is disabled)
#![cfg(not(feature = "your-feature"))]

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct YourType;

impl YourType {
    pub fn new() -> Result<Self> {
        Err(Error::not_implemented(
            "YourType requires the 'your-feature' feature to be enabled"
        ))
    }
}
```

### Usage in Main Module
```rust
// src/lib.rs
#[cfg(not(feature = "your-feature"))]
pub mod stubs;

// Conditional exports
#[cfg(feature = "your-feature")]
pub use real_crate::{YourType, AnotherType};

#[cfg(not(feature = "your-feature"))]
pub use stubs::{YourType, AnotherType};
```

## Pattern 2: Feature-Gated Error Variants

### Problem
Error types that wrap external crate errors can't compile if the external crate isn't available.

### Solution
Feature-gate the error variant itself.

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MyError {
    #[cfg(feature = "external-crate")]
    #[error("External crate error: {0}")]
    External(#[from] external_crate::Error),

    #[error("Feature not available: {0}")]
    NotImplemented(String),

    // Other variants...
}
```

## Pattern 3: Feature-Gated Modules

### Problem
Entire modules depend on an optional feature.

### Solution
Feature-gate module declarations and re-exports.

```rust
// src/lib.rs
#[cfg(feature = "ml")]
pub mod neural;

#[cfg(feature = "ml")]
pub mod training;

// Conditional re-exports
#[cfg(feature = "ml")]
pub use neural::{Model, Predictor};

#[cfg(feature = "ml")]
pub use training::{Trainer, DataLoader};

// Always available types
pub use neural::Config; // Config doesn't depend on ML libs
```

## Pattern 4: Dual Trait Implementations

### Problem
A trait needs different implementations based on feature availability.

### Solution
Implement the trait twice with different feature gates.

```rust
// Full implementation when feature is enabled
#[cfg(feature = "advanced")]
impl Model {
    pub fn predict(&self, input: &Tensor) -> Result<Tensor> {
        // Real ML prediction
        self.neural_net.forward(input)
    }
}

// Stub implementation when feature is disabled
#[cfg(not(feature = "advanced"))]
impl Model {
    pub fn predict(&self, input: &Tensor) -> Result<Tensor> {
        Err(Error::not_implemented(
            "Model prediction requires 'advanced' feature"
        ))
    }
}
```

## Pattern 5: Feature-Gated Tests

### Problem
Tests should verify both with and without features.

### Solution
Write separate tests for each configuration.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Test with feature enabled
    #[test]
    #[cfg(feature = "your-feature")]
    fn test_real_functionality() {
        let model = Model::new().unwrap();
        let result = model.predict(&input).unwrap();
        assert!(result.is_valid());
    }

    // Test without feature (stub mode)
    #[test]
    #[cfg(not(feature = "your-feature"))]
    fn test_stub_returns_not_implemented() {
        let model = Model::new().unwrap();
        let result = model.predict(&input);

        match result {
            Err(Error::NotImplemented(_)) => {}, // Expected
            _ => panic!("Should return NotImplemented"),
        }
    }

    // Test that always runs (feature-independent)
    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("input_size"));
    }
}
```

## Pattern 6: Optional Fields in Structs

### Problem
A struct has fields that only exist when a feature is enabled.

### Solution
Feature-gate the fields and handle in Default implementation.

```rust
#[derive(Debug, Clone)]
pub struct Config {
    pub input_size: usize,
    pub output_size: usize,

    #[cfg(feature = "gpu")]
    pub device: Option<Device>,

    #[cfg(feature = "advanced")]
    pub optimization_level: u8,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input_size: 100,
            output_size: 10,
            #[cfg(feature = "gpu")]
            device: None,
            #[cfg(feature = "advanced")]
            optimization_level: 2,
        }
    }
}
```

## Pattern 7: Cargo.toml Configuration

### Problem
How to properly declare optional dependencies and features.

### Solution
Use clear feature definitions with optional dependencies.

```toml
[dependencies]
# Always required
serde = { version = "1.0", features = ["derive"] }
thiserror = "1.0"

# Optional dependencies
candle-core = { version = "0.6", optional = true }
cuda-runtime = { version = "0.5", optional = true }

[features]
default = []  # Minimal by default

# Basic feature
ml = ["candle-core"]

# Feature that depends on another feature
gpu = ["ml", "cuda-runtime"]

# Feature that enables other features
full = ["ml", "gpu"]

[dev-dependencies]
# Test dependencies
```

## Pattern 8: Documentation with Feature Gates

### Problem
API docs should indicate when features are required.

### Solution
Use doc comments with feature information.

```rust
/// Neural network model for time series prediction.
///
/// # Features
///
/// This type requires the `ml` feature to be enabled:
///
/// ```toml
/// [dependencies]
/// my-crate = { version = "0.1", features = ["ml"] }
/// ```
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "ml")]
/// # {
/// use my_crate::Model;
///
/// let model = Model::new()?;
/// let prediction = model.predict(&input)?;
/// # }
/// ```
#[cfg(feature = "ml")]
pub struct Model { /* ... */ }
```

## Complete Example: Neural Crate

See `/workspaces/neural-trader/neural-trader-rust/crates/neural/` for a complete implementation using all these patterns:

1. **stubs.rs** - Stub types for Device and Tensor
2. **error.rs** - Feature-gated error variant for candle
3. **lib.rs** - Conditional exports and initialization
4. **models/mod.rs** - Feature-gated model implementations
5. **training/mod.rs** - Feature-gated training modules
6. **inference/mod.rs** - Feature-gated inference modules

## Quick Checklist for Feature Gating

- [ ] Add optional dependency to Cargo.toml
- [ ] Define feature in Cargo.toml
- [ ] Create stubs module for key types
- [ ] Feature-gate error variants that wrap external errors
- [ ] Add conditional imports with #[cfg(feature = "...")]
- [ ] Feature-gate module declarations
- [ ] Feature-gate trait implementations
- [ ] Update Default implementations for structs with optional fields
- [ ] Write tests for both configurations
- [ ] Document feature requirements in API docs
- [ ] Verify `cargo build` works without features
- [ ] Verify `cargo build --features all-features` works

## Testing Both Configurations

```bash
# Test without features (default)
cargo build --package your-crate
cargo test --package your-crate

# Test with features
cargo build --package your-crate --features your-feature
cargo test --package your-crate --features your-feature

# Test all features
cargo build --package your-crate --all-features
cargo test --package your-crate --all-features
```

## Common Pitfalls

1. **Forgetting to feature-gate error variants** - Causes compilation errors
2. **Not providing Default for optional fields** - Breaks default() calls
3. **Mixing stub and real types** - Use clear module boundaries
4. **Testing only one configuration** - Always test both with and without
5. **Circular feature dependencies** - Keep feature dependency graph simple

## Benefits of This Approach

1. **Compilation without dependencies** - Reduces build time and binary size
2. **Clear API boundaries** - Users know what features enable what
3. **Gradual adoption** - Users can enable features as needed
4. **Testing flexibility** - Can test stub behavior separately
5. **Documentation clarity** - Feature requirements are explicit
