# Isolation Forest Implementation

## Overview

This is a complete, high-performance Rust implementation of the Isolation Forest algorithm for anomaly detection in trading data. The implementation achieves sub-50 microsecond inference times as required.

## Features

- **200 Isolation Trees**: Default configuration with 200 trees for robust anomaly detection
- **Anomaly Scoring**: Accurate anomaly scoring algorithm based on path lengths
- **Contamination Parameter**: Configurable contamination rate (default: 0.1)
- **Feature Importance**: Calculate feature contributions to anomaly detection
- **Parallel Processing**: Uses Rayon for parallel tree building and prediction
- **High Performance**: <50μs inference time per sample
- **Serialization Support**: Full serde support for model persistence

## Key Components

### IsolationForestConfig
```rust
pub struct IsolationForestConfig {
    pub n_estimators: usize,      // Number of trees (default: 200)
    pub max_samples: usize,       // Max samples per tree (default: 256)
    pub contamination: f32,       // Expected anomaly rate (default: 0.1)
    pub max_depth: Option<usize>, // Max tree depth
    pub random_seed: Option<u64>, // For reproducibility
    pub n_jobs: Option<usize>,    // Number of parallel threads
}
```

### Usage Example
```rust
use nn_models::isolation_forest::IsolationForest;

// Create and train model
let mut forest = IsolationForest::builder()
    .n_estimators(200)
    .contamination(0.1)
    .random_seed(42)
    .build();

forest.fit(&training_data);

// Detect anomalies
let predictions = forest.predict(&test_data);
let scores = forest.decision_function(&test_data);

// Get feature importances
let importances = forest.feature_importances();
```

## Performance

- **Training**: Parallel tree construction for fast training
- **Inference**: <50μs per prediction (verified by tests)
- **Memory**: Efficient tree storage with shared Arc pointers

## Algorithm Details

1. **Tree Construction**: Each tree is built by randomly selecting features and split points
2. **Path Length**: Anomalies have shorter average path lengths in the trees
3. **Anomaly Score**: Calculated as 2^(-average_path_length / c(n))
4. **Threshold**: Automatically determined based on contamination rate

## Testing

The implementation includes comprehensive tests:
- Basic functionality tests
- Performance benchmarks (<50μs requirement)
- Feature importance validation
- Reproducibility tests
- Edge case handling

## Integration

The module is ready to be integrated with the rest of the ATS CP Trader system. It follows the same patterns as other ML models in the codebase and can be used for real-time anomaly detection in trading strategies.