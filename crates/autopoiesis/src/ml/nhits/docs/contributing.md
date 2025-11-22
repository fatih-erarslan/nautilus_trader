# Contributing Guide

Welcome to the NHITS project! This guide will help you contribute effectively to the development of this consciousness-aware neural time series forecasting system.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Style](#code-style)
- [Architecture Guidelines](#architecture-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Community](#community)

## Getting Started

### Prerequisites

- Rust 1.70+ with Cargo
- Git
- Basic understanding of time series forecasting
- Familiarity with neural networks and Rust async programming

### Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/autopoiesis.git
cd autopoiesis

# Add upstream remote
git remote add upstream https://github.com/autopoiesis/autopoiesis.git

# Create a development branch
git checkout -b feature/your-feature-name
```

### First Build

```bash
# Install dependencies and build
cargo build

# Run tests to ensure everything works
cargo test

# Run examples
cargo run --example basic_forecasting
```

## Development Environment

### Recommended Tools

- **IDE**: VS Code with rust-analyzer extension
- **Formatter**: rustfmt (configured in rustfmt.toml)
- **Linter**: clippy for code quality
- **Documentation**: cargo doc for API docs

### Environment Setup

```bash
# Install additional tools
rustup component add clippy rustfmt

# Install cargo extensions
cargo install cargo-watch cargo-expand cargo-audit

# Setup pre-commit hooks (optional)
cp scripts/pre-commit .git/hooks/
chmod +x .git/hooks/pre-commit
```

### Development Workflow

```bash
# Watch for changes and rebuild
cargo watch -x check -x test

# Format code
cargo fmt

# Run linter
cargo clippy -- -D warnings

# Generate documentation
cargo doc --open
```

## Code Style

### Rust Conventions

Follow the official Rust style guide with these project-specific additions:

```rust
// ✅ Good: Clear, descriptive names
struct HierarchicalBlock {
    config: BlockConfig,
    layers: Vec<NeuralLayer>,
    activation: ActivationType,
}

impl HierarchicalBlock {
    /// Creates a new hierarchical block with the given configuration.
    /// 
    /// # Arguments
    /// 
    /// * `config` - Block configuration specifying architecture parameters
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let config = BlockConfig::default();
    /// let block = HierarchicalBlock::new(config);
    /// ```
    pub fn new(config: BlockConfig) -> Self {
        Self {
            config,
            layers: Vec::new(),
            activation: ActivationType::GELU,
        }
    }
    
    /// Performs forward pass through the hierarchical block.
    /// 
    /// # Errors
    /// 
    /// Returns `NHITSError::ComputationError` if the computation fails.
    pub fn forward(
        &mut self,
        input: &Array3<f64>,
        consciousness_modulation: f64,
    ) -> Result<Array3<f64>, NHITSError> {
        // Implementation
        todo!()
    }
}

// ❌ Bad: Unclear names, missing documentation
struct HB {
    cfg: BC,
    l: Vec<NL>,
}

impl HB {
    pub fn new(c: BC) -> Self { /* ... */ }
    pub fn fwd(&mut self, i: &Array3<f64>) -> Array3<f64> { /* ... */ }
}
```

### Error Handling

Use the established error types and provide descriptive error messages:

```rust
// ✅ Good: Descriptive error handling
fn validate_input_shape(
    input: &Array3<f64>,
    expected_features: usize,
) -> Result<(), NHITSError> {
    let actual_features = input.shape()[2];
    if actual_features != expected_features {
        return Err(NHITSError::ShapeMismatch {
            expected: vec![0, 0, expected_features],
            got: vec![input.shape()[0], input.shape()[1], actual_features],
        });
    }
    Ok(())
}

// ❌ Bad: Generic error handling
fn validate_input(input: &Array3<f64>) -> Result<(), Box<dyn std::error::Error>> {
    if input.shape()[2] != 1 {
        return Err("wrong shape".into());
    }
    Ok(())
}
```

### Async Code

Follow async best practices:

```rust
// ✅ Good: Proper async patterns
pub async fn train_async(
    &mut self,
    train_data: &Array3<f64>,
    epochs: usize,
) -> Result<TrainingHistory, NHITSError> {
    let mut history = TrainingHistory::new();
    
    for epoch in 0..epochs {
        // Use yield points in long-running loops
        if epoch % 10 == 0 {
            tokio::task::yield_now().await;
        }
        
        let loss = self.train_epoch(train_data).await?;
        history.train_losses.push(loss);
    }
    
    Ok(history)
}

// ❌ Bad: Blocking operations in async context
pub async fn bad_train(&mut self, data: &Array3<f64>) -> Result<(), NHITSError> {
    // Don't do blocking operations without yield points
    for i in 0..1000000 {
        self.compute_something_expensive(); // Blocks event loop
    }
    Ok(())
}
```

### Consciousness Integration

When working with consciousness-aware components:

```rust
// ✅ Good: Proper consciousness integration
impl TemporalAttention {
    pub fn apply_with_consciousness(
        &self,
        input: &Array3<f64>,
        consciousness_state: &ConsciousnessState,
    ) -> Result<Array3<f64>, NHITSError> {
        // Check consciousness coherence
        if consciousness_state.coherence < self.min_coherence_threshold {
            return self.apply_fallback(input);
        }
        
        // Apply consciousness-modulated attention
        let attention_weights = self.compute_attention_weights(input)?;
        let modulated_weights = self.modulate_with_consciousness(
            attention_weights,
            consciousness_state,
        )?;
        
        self.apply_attention(input, &modulated_weights)
    }
    
    fn apply_fallback(&self, input: &Array3<f64>) -> Result<Array3<f64>, NHITSError> {
        // Fallback when consciousness is not available
        self.apply_standard_attention(input)
    }
}
```

## Architecture Guidelines

### Module Organization

```
src/ml/nhits/
├── core/           # Core NHITS implementation
├── blocks/         # Hierarchical blocks
├── attention/      # Attention mechanisms
├── decomposition/  # Time series decomposition
├── adaptation/     # Adaptive structures
├── consciousness/  # Consciousness integration
├── configs/        # Configuration system
├── utils/          # Utility functions
├── forecasting/    # Production pipeline
├── api/           # REST/WebSocket APIs
└── tests/         # Test modules
```

### Adding New Components

When adding new components, follow this structure:

```rust
// src/ml/nhits/new_component/mod.rs

//! New Component Module
//! 
//! This module implements [description of what the component does].
//! It integrates with the consciousness system for [specific purpose].

use crate::consciousness::ConsciousnessField;
use crate::core::autopoiesis::AutopoieticSystem;
use super::{NHITSError, /* other imports */};

/// Main component struct
#[derive(Debug, Clone)]
pub struct NewComponent {
    config: NewComponentConfig,
    consciousness: Option<Arc<ConsciousnessField>>,
    state: ComponentState,
}

/// Configuration for the new component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewComponentConfig {
    // Configuration fields with sensible defaults
    pub parameter1: f64,
    pub parameter2: usize,
    pub enable_consciousness: bool,
}

impl Default for NewComponentConfig {
    fn default() -> Self {
        Self {
            parameter1: 1.0,
            parameter2: 128,
            enable_consciousness: true,
        }
    }
}

impl NewComponent {
    /// Creates a new component instance
    pub fn new(
        config: NewComponentConfig,
        consciousness: Option<Arc<ConsciousnessField>>,
    ) -> Self {
        Self {
            config,
            consciousness,
            state: ComponentState::default(),
        }
    }
    
    /// Main processing function
    pub fn process(
        &mut self,
        input: &Array3<f64>,
    ) -> Result<Array3<f64>, NHITSError> {
        // Implementation with consciousness integration
        if let Some(ref consciousness) = self.consciousness {
            let consciousness_state = consciousness.get_current_state();
            self.process_with_consciousness(input, &consciousness_state)
        } else {
            self.process_standard(input)
        }
    }
    
    fn process_with_consciousness(
        &mut self,
        input: &Array3<f64>,
        consciousness_state: &ConsciousnessState,
    ) -> Result<Array3<f64>, NHITSError> {
        // Consciousness-aware processing
        todo!()
    }
    
    fn process_standard(
        &mut self,
        input: &Array3<f64>,
    ) -> Result<Array3<f64>, NHITSError> {
        // Standard processing without consciousness
        todo!()
    }
}

#[derive(Debug, Clone, Default)]
struct ComponentState {
    // Internal state
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_component_creation() {
        let config = NewComponentConfig::default();
        let component = NewComponent::new(config, None);
        // Test assertions
    }
    
    #[test]
    fn test_component_processing() {
        // Test the main functionality
    }
}
```

### Configuration System

All components should use the builder pattern for configuration:

```rust
#[derive(Debug, Clone)]
pub struct ComponentConfigBuilder {
    config: ComponentConfig,
}

impl ComponentConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: ComponentConfig::default(),
        }
    }
    
    pub fn with_parameter1(mut self, value: f64) -> Self {
        self.config.parameter1 = value;
        self
    }
    
    pub fn with_parameter2(mut self, value: usize) -> Self {
        self.config.parameter2 = value;
        self
    }
    
    pub fn with_consciousness(mut self, enabled: bool) -> Self {
        self.config.enable_consciousness = enabled;
        self
    }
    
    pub fn build(self) -> Result<ComponentConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}
```

## Testing

### Test Categories

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test component interactions
3. **Property Tests**: Test invariants using quickcheck
4. **Benchmark Tests**: Performance testing using criterion

### Test Organization

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::quickcheck;
    use approx::assert_abs_diff_eq;
    
    // Unit tests
    #[test]
    fn test_component_initialization() {
        let config = ComponentConfig::default();
        let component = Component::new(config);
        assert!(component.is_initialized());
    }
    
    // Integration tests
    #[tokio::test]
    async fn test_component_with_consciousness() {
        let consciousness = Arc::new(ConsciousnessField::new());
        let component = Component::new(config, Some(consciousness));
        
        let input = Array3::zeros((1, 100, 1));
        let result = component.process(&input).await?;
        
        assert_eq!(result.shape(), &[1, 100, 1]);
    }
    
    // Property tests
    quickcheck! {
        fn test_output_shape_invariant(
            batch_size: usize,
            seq_len: usize,
            features: usize
        ) -> bool {
            if batch_size == 0 || seq_len == 0 || features == 0 {
                return true; // Skip invalid inputs
            }
            
            let input = Array3::zeros((batch_size, seq_len, features));
            let component = Component::new(ComponentConfig::default());
            
            match component.process(&input) {
                Ok(output) => output.shape()[0] == batch_size,
                Err(_) => true, // Allow errors for invalid configurations
            }
        }
    }
    
    // Numerical stability tests
    #[test]
    fn test_numerical_stability() {
        let component = Component::new(ComponentConfig::default());
        
        // Test with extreme values
        let extreme_input = Array3::from_elem((1, 10, 1), 1e6);
        let result = component.process(&extreme_input).unwrap();
        
        // Ensure no NaN or Inf values
        for value in result.iter() {
            assert!(value.is_finite());
        }
    }
    
    // Consciousness integration tests
    #[test]
    fn test_consciousness_fallback() {
        let consciousness = Arc::new(ConsciousnessField::new());
        
        // Simulate low coherence
        consciousness.set_coherence(0.1); // Below threshold
        
        let component = Component::new(config, Some(consciousness));
        let input = Array3::zeros((1, 10, 1));
        
        // Should still work with fallback mechanism
        let result = component.process(&input).unwrap();
        assert!(result.shape() == input.shape());
    }
}
```

### Benchmark Tests

```rust
// benches/component_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use autopoiesis::ml::nhits::*;

fn benchmark_component_processing(c: &mut Criterion) {
    let component = Component::new(ComponentConfig::default());
    let input = Array3::zeros((32, 168, 1));
    
    c.bench_function("component_forward", |b| {
        b.iter(|| {
            let result = component.process(black_box(&input));
            black_box(result)
        });
    });
}

criterion_group!(benches, benchmark_component_processing);
criterion_main!(benches);
```

### Test Data

Create realistic test data:

```rust
// tests/test_utils.rs
pub fn create_synthetic_time_series(
    samples: usize,
    sequence_length: usize,
    features: usize,
) -> Array3<f64> {
    let mut rng = SmallRng::seed_from_u64(42); // Deterministic seed
    let mut data = Array3::zeros((samples, sequence_length, features));
    
    for sample in 0..samples {
        for feature in 0..features {
            // Create realistic time series with trend and seasonality
            for t in 0..sequence_length {
                let trend = 0.01 * t as f64;
                let seasonal = 5.0 * (2.0 * PI * t as f64 / 24.0).sin();
                let noise = rng.gen_range(-1.0..1.0);
                
                data[[sample, t, feature]] = 100.0 + trend + seasonal + noise;
            }
        }
    }
    
    data
}

pub fn create_consciousness_field_for_testing() -> Arc<ConsciousnessField> {
    let consciousness = Arc::new(ConsciousnessField::new());
    consciousness.set_coherence(0.8); // Good coherence for testing
    consciousness
}
```

## Documentation

### Code Documentation

Use comprehensive doc comments:

```rust
/// Performs hierarchical time series forecasting with consciousness integration.
/// 
/// This function implements the NHITS algorithm with adaptive structure evolution
/// guided by the consciousness field. The model can adapt its architecture
/// based on performance feedback and consciousness coherence levels.
/// 
/// # Arguments
/// 
/// * `input` - Input time series data with shape `(batch_size, sequence_length, features)`
/// * `lookback_window` - Number of historical time steps to consider
/// * `forecast_horizon` - Number of future time steps to predict
/// 
/// # Returns
/// 
/// Returns a `Result` containing:
/// * `Ok(Array3<f64>)` - Predictions with shape `(batch_size, forecast_horizon, output_features)`
/// * `Err(NHITSError)` - Error if computation fails
/// 
/// # Errors
/// 
/// This function will return an error if:
/// * Input dimensions don't match the model configuration
/// * Consciousness coherence falls below the minimum threshold
/// * Numerical computation becomes unstable
/// 
/// # Examples
/// 
/// ```rust
/// use autopoiesis::ml::nhits::prelude::*;
/// use std::sync::Arc;
/// 
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let consciousness = Arc::new(ConsciousnessField::new());
/// let autopoietic = Arc::new(AutopoieticSystem::new());
/// let config = NHITSConfig::default();
/// let mut model = NHITS::new(config, consciousness, autopoietic);
/// 
/// let input = Array3::zeros((32, 168, 1)); // 32 samples, 168 hours, 1 feature
/// let predictions = model.forward(&input, 168, 24)?; // 24-hour forecast
/// 
/// assert_eq!(predictions.shape(), &[32, 24, 1]);
/// # Ok(())
/// # }
/// ```
/// 
/// # Performance Notes
/// 
/// * Time complexity: O(n * h * b) where n=sequence_length, h=hidden_size, b=num_blocks
/// * Space complexity: O(n * h) for intermediate computations
/// * Consciousness integration adds ~10-15% computational overhead
/// 
/// # See Also
/// 
/// * [`train`](Self::train) - For training the model
/// * [`predict`](Self::predict) - For single-sequence predictions
/// * [`NHITSConfig`] - For configuration options
pub fn forward(
    &mut self,
    input: &Array3<f64>,
    lookback_window: usize,
    forecast_horizon: usize,
) -> Result<Array3<f64>, NHITSError> {
    // Implementation
    todo!()
}
```

### Module Documentation

Each module should have comprehensive documentation:

```rust
//! # Attention Mechanisms
//! 
//! This module implements various attention mechanisms for the NHITS architecture,
//! including consciousness-aware attention that can adapt based on the current
//! consciousness field state.
//! 
//! ## Overview
//! 
//! The attention mechanisms in NHITS serve to focus the model's processing
//! on the most relevant parts of the input sequence. The consciousness integration
//! allows the attention patterns to evolve based on higher-level understanding
//! of the data patterns.
//! 
//! ## Available Attention Types
//! 
//! * [`TemporalAttention`] - Standard temporal attention with consciousness modulation
//! * [`MultiHeadAttention`] - Multi-head attention for complex patterns
//! * [`SparseAttention`] - Memory-efficient sparse attention
//! * [`LocalWindowAttention`] - Attention with limited receptive field
//! 
//! ## Examples
//! 
//! ```rust
//! use autopoiesis::ml::nhits::attention::*;
//! 
//! let config = AttentionConfig {
//!     num_heads: 8,
//!     head_dim: 64,
//!     consciousness_integration: true,
//!     ..Default::default()
//! };
//! 
//! let attention = TemporalAttention::new(&config);
//! ```
//! 
//! ## Consciousness Integration
//! 
//! When consciousness integration is enabled, the attention mechanism:
//! 
//! 1. Receives consciousness state updates
//! 2. Modulates attention weights based on coherence levels
//! 3. Adapts attention patterns based on field strength
//! 4. Falls back to standard attention when coherence is low
//! 
//! ## Performance Considerations
//! 
//! * Standard attention: O(n²) time and space complexity
//! * Sparse attention: O(n√n) with configurable sparsity
//! * Local window: O(n*w) where w is window size
//! * Consciousness integration adds ~5% overhead
```

### README Updates

When adding features, update relevant README sections:

```markdown
## Recent Changes

### v0.2.0 - New Attention Mechanisms

- Added sparse attention for memory efficiency
- Implemented local window attention for streaming data
- Enhanced consciousness integration with attention modulation
- Performance improvements: 15% faster inference, 30% less memory usage

### Usage

```rust
let config = NHITSConfigBuilder::new()
    .with_attention(AttentionType::Sparse { sparsity_factor: 0.9 })
    .build()?;
```

See [examples/sparse_attention.rs](examples/sparse_attention.rs) for complete example.
```

## Pull Request Process

### Before Submitting

1. **Run the full test suite**:
   ```bash
   cargo test --all-features
   cargo clippy -- -D warnings
   cargo fmt --check
   ```

2. **Update documentation**:
   ```bash
   cargo doc --no-deps
   # Check that docs build without warnings
   ```

3. **Run benchmarks** if performance-related:
   ```bash
   cargo bench
   ```

4. **Update CHANGELOG.md** with your changes

### PR Description Template

```markdown
## Description

Brief description of the changes and their motivation.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing

- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have added integration tests for new features
- [ ] I have run benchmarks and performance has not regressed

## Documentation

- [ ] I have updated the documentation accordingly
- [ ] I have added examples for new features
- [ ] I have updated the changelog

## Checklist

- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published

## Related Issues

Closes #[issue_number]
```

### Review Process

1. **Automated Checks**: All CI checks must pass
2. **Code Review**: At least one maintainer review required
3. **Testing**: New features must include tests
4. **Documentation**: Public APIs must be documented
5. **Backwards Compatibility**: Breaking changes require major version bump

### Merging

- Use "Squash and merge" for feature branches
- Use "Merge commit" for release branches
- Ensure commit messages follow conventional commit format:
  ```
  feat(attention): add sparse attention mechanism
  
  - Implements O(n√n) sparse attention
  - Reduces memory usage by 30%
  - Maintains accuracy within 1% of dense attention
  
  Closes #123
  ```

## Issue Guidelines

### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Create configuration '...'
2. Run model with input '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- Autopoiesis version: [e.g. 0.1.0]
- Rust version: [e.g. 1.70.0]
- OS: [e.g. Ubuntu 22.04]

**Additional Context**
- Configuration used
- Input data characteristics
- Error messages and stack traces
- Any relevant logs

**Minimal Reproducible Example**
```rust
// Minimal code that reproduces the issue
```

### Feature Requests

```markdown
**Feature Description**
A clear and concise description of the feature you'd like to see.

**Motivation**
Why is this feature needed? What problem does it solve?

**Proposed Solution**
How do you envision this feature working?

**Alternatives Considered**
What alternatives have you considered?

**Implementation Notes**
Any thoughts on how this could be implemented?

**Examples**
```rust
// Example of how the feature would be used
```

**Additional Context**
Any other context, references, or examples.
```

### Performance Issues

Include benchmarking information:

```rust
// Include benchmark code that demonstrates the issue
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_slow_operation(c: &mut Criterion) {
    c.bench_function("slow_operation", |b| {
        b.iter(|| {
            // Code that's slower than expected
        });
    });
}
```

## Community

### Communication Channels

- **GitHub Discussions**: Design discussions, questions
- **Discord**: Real-time chat, community support
- **Issues**: Bug reports, feature requests
- **Email**: Security issues, private concerns

### Code of Conduct

We follow the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). In summary:

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what's best for the community
- Show empathy towards other community members

### Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Annual contributor highlights
- Conference presentations (with permission)

### Becoming a Maintainer

Regular contributors may be invited to become maintainers based on:
- Consistent high-quality contributions
- Deep understanding of the codebase
- Positive community interactions
- Commitment to the project's goals

Maintainers have additional responsibilities:
- Reviewing pull requests
- Triaging issues
- Guiding project direction
- Mentoring new contributors

Thank you for contributing to NHITS! Your efforts help advance the state of consciousness-aware time series forecasting.