# Neural Crate Quick Start Guide

Get started with `nt-neural` in under 5 minutes.

## Table of Contents

1. [Installation](#installation)
2. [Basic Data Processing](#basic-data-processing)
3. [Feature Engineering](#feature-engineering)
4. [Model Training (GPU)](#model-training-gpu)
5. [Model Storage](#model-storage)
6. [Next Steps](#next-steps)

## Installation

### CPU-Only Mode (Recommended for Getting Started)

Add to your `Cargo.toml`:

```toml
[dependencies]
nt-neural = "0.1.0"
```

This gives you access to all preprocessing, feature engineering, and metrics without requiring GPU dependencies.

### GPU-Accelerated Mode

For neural model training and inference:

```toml
[dependencies]
# CUDA (NVIDIA GPUs)
nt-neural = { version = "0.1.0", features = ["candle", "cuda"] }

# Metal (Apple Silicon)
nt-neural = { version = "0.1.0", features = ["candle", "metal"] }

# Accelerate (Apple CPU optimization)
nt-neural = { version = "0.1.0", features = ["candle", "accelerate"] }
```

**Note**: GPU features are currently unavailable due to upstream dependency conflicts. Use CPU-only mode for now.

## Basic Data Processing

### Load and Normalize Data

```rust
use nt_neural::utils::preprocessing::*;

fn main() -> anyhow::Result<()> {
    // Your raw price data
    let prices = vec![100.0, 102.5, 101.8, 103.2, 105.0];

    // Normalize to zero mean, unit variance
    let (normalized, params) = normalize(&prices);
    println!("Normalized: {:?}", normalized);

    // Denormalize back to original scale
    let original = denormalize(&normalized, &params);
    assert_eq!(prices, original);

    Ok(())
}
```

### Handle Outliers

```rust
use nt_neural::utils::preprocessing::*;

fn main() -> anyhow::Result<()> {
    let data = vec![100.0, 102.0, 101.0, 999.0, 103.0]; // 999.0 is outlier

    // Remove outliers (z-score > 3)
    let clean = remove_outliers(&data, 3.0);

    // Or winsorize (cap at percentiles)
    let winsorized = winsorize(&data, 0.05, 0.95);

    println!("Clean: {:?}", clean);
    println!("Winsorized: {:?}", winsorized);

    Ok(())
}
```

### Detrend Time Series

```rust
use nt_neural::utils::preprocessing::*;

fn main() -> anyhow::Result<()> {
    let data = vec![100.0, 102.0, 104.0, 106.0, 108.0]; // Linear trend

    // Remove linear trend
    let (detrended, slope, intercept) = detrend(&data);
    println!("Detrended: {:?}", detrended);
    println!("Slope: {}, Intercept: {}", slope, intercept);

    Ok(())
}
```

## Feature Engineering

### Create Lagged Features

```rust
use nt_neural::utils::features::*;

fn main() -> anyhow::Result<()> {
    let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0, 104.0];

    // Create lags: t-1, t-3, t-7
    let lagged = create_lags(&prices, &[1, 3, 7]);

    for (i, lags) in lagged.iter().enumerate() {
        println!("Row {}: {:?}", i, lags);
    }

    Ok(())
}
```

### Rolling Statistics

```rust
use nt_neural::utils::features::*;

fn main() -> anyhow::Result<()> {
    let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0];
    let window = 3;

    // Calculate rolling mean
    let ma = rolling_mean(&prices, window);
    println!("Moving Average: {:?}", ma);

    // Calculate rolling std
    let std = rolling_std(&prices, window);
    println!("Rolling Std: {:?}", std);

    // Calculate rolling min/max
    let min_vals = rolling_min(&prices, window);
    let max_vals = rolling_max(&prices, window);

    Ok(())
}
```

### Technical Indicators

```rust
use nt_neural::utils::features::*;

fn main() -> anyhow::Result<()> {
    let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0];

    // Exponential Moving Average
    let alpha = 0.3;
    let ema_values = ema(&prices, alpha);
    println!("EMA: {:?}", ema_values);

    // Rate of Change
    let period = 1;
    let roc = rate_of_change(&prices, period);
    println!("ROC: {:?}", roc);

    Ok(())
}
```

### Seasonality Features

```rust
use nt_neural::utils::features::*;
use chrono::{DateTime, Utc};

fn main() -> anyhow::Result<()> {
    let n = 24; // 24 hours
    let period = 24; // Daily seasonality
    let order = 3; // 3 Fourier terms

    // Generate Fourier features for seasonality
    let fourier = fourier_features(n, period, order);

    // Create calendar features from timestamps
    let timestamps: Vec<DateTime<Utc>> = vec![/* your timestamps */];
    let calendar = calendar_features(&timestamps);

    // calendar contains: hour, day_of_week, day_of_month, etc.

    Ok(())
}
```

## Model Training (GPU)

**Note**: This section requires the `candle` feature which is currently unavailable.

### NHITS Model

```rust
#[cfg(feature = "candle")]
use nt_neural::{NHITSModel, NHITSConfig, Trainer, TrainingConfig};

#[cfg(feature = "candle")]
async fn train_model() -> anyhow::Result<()> {
    // Model configuration
    let config = NHITSConfig {
        input_size: 168,      // 1 week of hourly data
        horizon: 24,          // Predict next 24 hours
        num_stacks: 3,
        num_blocks_per_stack: vec![1, 1, 1],
        hidden_size: 512,
        pooling_sizes: vec![4, 4, 1],
        dropout: 0.1,
        ..Default::default()
    };

    // Initialize model
    let model = NHITSModel::new(config)?;

    // Training configuration
    let train_config = TrainingConfig {
        batch_size: 32,
        num_epochs: 100,
        learning_rate: 1e-3,
        weight_decay: 1e-5,
        gradient_clip: Some(1.0),
        early_stopping_patience: 10,
        validation_split: 0.2,
        mixed_precision: true,
    };

    // Create trainer
    let trainer = Trainer::new(train_config);

    // Train model (prepare your data first)
    // let trained_model = trainer.train(&model, &train_data).await?;

    Ok(())
}
```

### LSTM-Attention Model

```rust
#[cfg(feature = "candle")]
use nt_neural::{LSTMAttentionModel, LSTMAttentionConfig};

#[cfg(feature = "candle")]
fn create_lstm_model() -> anyhow::Result<()> {
    let config = LSTMAttentionConfig {
        input_size: 168,
        horizon: 24,
        hidden_size: 256,
        num_layers: 3,
        num_attention_heads: 8,
        dropout: 0.2,
        ..Default::default()
    };

    let model = LSTMAttentionModel::new(config)?;

    Ok(())
}
```

### DeepAR (Probabilistic Forecasting)

```rust
#[cfg(feature = "candle")]
use nt_neural::{DeepARModel, DeepARConfig, DistributionType};

#[cfg(feature = "candle")]
fn create_deepar_model() -> anyhow::Result<()> {
    let config = DeepARConfig {
        input_size: 168,
        horizon: 24,
        hidden_size: 128,
        num_layers: 3,
        distribution: DistributionType::Gaussian,
        quantiles: vec![0.1, 0.5, 0.9], // 10%, 50%, 90% quantiles
        dropout: 0.2,
        ..Default::default()
    };

    let model = DeepARModel::new(config)?;

    Ok(())
}
```

## Model Storage

### Save Model to AgentDB

```rust
use nt_neural::storage::{AgentDbStorage, ModelMetadata};

async fn save_model_example() -> anyhow::Result<()> {
    // Initialize storage
    let storage = AgentDbStorage::new("./data/models/agentdb.db").await?;

    // Create metadata
    let metadata = ModelMetadata {
        name: "btc-hourly-predictor".to_string(),
        model_type: "NHITS".to_string(),
        version: "1.0.0".to_string(),
        description: Some("Bitcoin hourly price predictor".to_string()),
        tags: vec!["crypto".to_string(), "hourly".to_string(), "production".to_string()],
        hyperparameters: Some(serde_json::json!({
            "input_size": 168,
            "horizon": 24,
            "hidden_size": 512
        })),
        ..Default::default()
    };

    // Save model (model_bytes would come from serialization)
    let model_bytes = vec![/* your model bytes */];
    let model_id = storage.save_model(&model_bytes, metadata).await?;

    println!("Model saved with ID: {}", model_id);

    Ok(())
}
```

### Load Model from AgentDB

```rust
use nt_neural::storage::AgentDbStorage;

async fn load_model_example() -> anyhow::Result<()> {
    let storage = AgentDbStorage::new("./data/models/agentdb.db").await?;

    // Load by model ID
    let model_id = "your-model-id";
    let model_bytes = storage.load_model(model_id).await?;
    let metadata = storage.get_metadata(model_id).await?;

    println!("Loaded model: {}", metadata.name);
    println!("Model type: {}", metadata.model_type);

    Ok(())
}
```

### Search Similar Models

```rust
use nt_neural::storage::AgentDbStorage;

async fn search_models_example() -> anyhow::Result<()> {
    let storage = AgentDbStorage::new("./data/models/agentdb.db").await?;

    // Search by embedding similarity
    let embedding = vec![/* your query embedding */];
    let similar_models = storage.search_similar_models(&embedding, 5).await?;

    for result in similar_models {
        println!("Model: {} (score: {})", result.metadata.name, result.score);
    }

    Ok(())
}
```

## Evaluation

### Calculate Metrics

```rust
use nt_neural::utils::metrics::EvaluationMetrics;

fn evaluate_predictions() -> anyhow::Result<()> {
    let y_true = vec![100.0, 102.0, 101.0, 103.0, 105.0];
    let y_pred = vec![100.5, 101.8, 101.2, 102.8, 104.5];

    // Calculate all metrics at once
    let metrics = EvaluationMetrics::calculate(&y_true, &y_pred, None)?;

    println!("MAE: {:.4}", metrics.mae);
    println!("RMSE: {:.4}", metrics.rmse);
    println!("MAPE: {:.2}%", metrics.mape);
    println!("R²: {:.4}", metrics.r2_score);
    println!("sMAPE: {:.2}%", metrics.smape);

    // With sample weights
    let weights = vec![1.0, 1.0, 1.0, 2.0, 2.0]; // More weight on recent
    let weighted_metrics = EvaluationMetrics::calculate(&y_true, &y_pred, Some(&weights))?;

    Ok(())
}
```

### Cross-Validation

```rust
use nt_neural::utils::validation::*;

fn cross_validate() -> anyhow::Result<()> {
    let data = vec![/* your time series data */];

    // Create time series splits
    let n_splits = 5;
    let test_size = 24; // 24 hours
    let splits = time_series_split(&data, n_splits, test_size);

    for (i, (train_idx, test_idx)) in splits.iter().enumerate() {
        println!("Fold {}: {} train, {} test", i, train_idx.len(), test_idx.len());
        // Train and evaluate on this split
    }

    Ok(())
}
```

## Next Steps

1. **Explore Models**: Read [MODELS.md](MODELS.md) for detailed model comparisons
2. **Advanced Training**: See [TRAINING.md](TRAINING.md) for training best practices
3. **Production Inference**: Check [INFERENCE.md](INFERENCE.md) for deployment
4. **AgentDB Deep Dive**: Learn more in [AGENTDB.md](AGENTDB.md)
5. **API Reference**: Browse [API.md](API.md) for complete API documentation

## Common Patterns

### Complete Preprocessing Pipeline

```rust
use nt_neural::utils::preprocessing::*;
use nt_neural::utils::features::*;

fn preprocess_pipeline(prices: &[f64]) -> anyhow::Result<Vec<Vec<f64>>> {
    // 1. Handle outliers
    let clean = remove_outliers(prices, 3.0);

    // 2. Normalize
    let (normalized, _params) = normalize(&clean);

    // 3. Create features
    let lags = create_lags(&normalized, &[1, 3, 7, 14]);
    let ma = rolling_mean(&normalized, 20);
    let std = rolling_std(&normalized, 20);

    // 4. Combine features
    let mut features = Vec::new();
    for i in 0..lags.len() {
        let mut row = lags[i].clone();
        row.push(ma[i]);
        row.push(std[i]);
        features.push(row);
    }

    Ok(features)
}
```

### Model Selection Pattern

```rust
use nt_neural::ModelType;

fn select_model(use_case: &str) -> ModelType {
    match use_case {
        "multi_horizon" => ModelType::NHITS,
        "sequential" => ModelType::LSTMAttention,
        "long_range" => ModelType::Transformer,
        "uncertainty" => ModelType::DeepAR,
        "interpretable" => ModelType::NBeats,
        "trend_seasonal" => ModelType::Prophet,
        _ => ModelType::NHITS, // Default
    }
}
```

## Troubleshooting

### Issue: Candle features won't compile

**Solution**: Use CPU-only mode. Remove `features = ["candle"]` from your `Cargo.toml`.

### Issue: NaN values in preprocessing

**Solution**: Check for division by zero in normalization:

```rust
let (normalized, params) = normalize(&data);
// Check for NaN
if normalized.iter().any(|&x| x.is_nan()) {
    // Handle constant data or apply different scaling
}
```

### Issue: Out of memory during feature engineering

**Solution**: Process in batches:

```rust
let batch_size = 1000;
for chunk in data.chunks(batch_size) {
    let features = create_lags(chunk, &[1, 3, 7]);
    // Process this batch
}
```

## Resources

- [Examples Directory](../../neural-trader-rust/crates/neural/examples/)
- [Full API Documentation](API.md)
- [Model Comparison Guide](MODELS.md)
- [Training Best Practices](TRAINING.md)

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/your-org/neural-trader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/neural-trader/discussions)
- **Documentation**: [docs.rs/nt-neural](https://docs.rs/nt-neural)
