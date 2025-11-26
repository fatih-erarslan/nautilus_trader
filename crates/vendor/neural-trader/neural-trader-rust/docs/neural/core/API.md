# API Reference

Complete API documentation for `nt-neural`.

## Table of Contents

1. [Core Types](#core-types)
2. [Models](#models)
3. [Training](#training)
4. [Inference](#inference)
5. [Storage](#storage)
6. [Utilities](#utilities)

## Core Types

### ModelType

Model architecture enumeration.

```rust
pub enum ModelType {
    NHITS,
    LSTMAttention,
    Transformer,
    GRU,
    TCN,
    DeepAR,
    NBeats,
    Prophet,
}
```

**Methods:**
- `to_string() -> String` - Get display name

### ModelConfig

Common configuration for all models.

```rust
pub struct ModelConfig {
    pub input_size: usize,      // Input sequence length
    pub horizon: usize,          // Forecast horizon
    pub hidden_size: usize,      // Hidden layer size
    pub num_features: usize,     // Number of input features
    pub dropout: f64,            // Dropout rate
    #[cfg(feature = "candle")]
    pub device: Option<Device>,  // Compute device
}
```

**Methods:**
- `default() -> Self` - Default configuration

### Device

Compute device (CPU/GPU).

```rust
#[cfg(feature = "candle")]
pub use candle_core::Device;

#[cfg(not(feature = "candle"))]
pub struct Device;  // Stub
```

**Methods:**
- `cpu() -> Self` - CPU device
- `new_cuda(ordinal: usize) -> Result<Self>` - CUDA device
- `new_metal(ordinal: usize) -> Result<Self>` - Metal device
- `is_cpu(&self) -> bool`
- `is_cuda(&self) -> bool`
- `is_metal(&self) -> bool`

## Models

### NHITS

Neural Hierarchical Interpolation for Time Series.

```rust
#[cfg(feature = "candle")]
pub struct NHITSModel { /* ... */ }

pub struct NHITSConfig {
    pub input_size: usize,
    pub horizon: usize,
    pub num_stacks: usize,
    pub num_blocks_per_stack: Vec<usize>,
    pub hidden_size: usize,
    pub pooling_sizes: Vec<usize>,
    pub dropout: f64,
    pub activation: Activation,
}
```

**Methods:**
- `new(config: NHITSConfig) -> Result<Self>`
- `forward(&self, input: &Tensor) -> Result<Tensor>`
- `save_weights(&self, path: &str) -> Result<()>`
- `load_weights(&mut self, path: &str) -> Result<()>`

### LSTM-Attention

LSTM with multi-head attention.

```rust
#[cfg(feature = "candle")]
pub struct LSTMAttentionModel { /* ... */ }

pub struct LSTMAttentionConfig {
    pub input_size: usize,
    pub horizon: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub dropout: f64,
    pub bidirectional: bool,
}
```

**Methods:**
- `new(config: LSTMAttentionConfig) -> Result<Self>`
- `forward(&self, input: &Tensor) -> Result<Tensor>`

### Transformer

Transformer architecture for time series.

```rust
#[cfg(feature = "candle")]
pub struct TransformerModel { /* ... */ }

pub struct TransformerConfig {
    pub input_size: usize,
    pub horizon: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
    pub d_ff: usize,
    pub dropout: f64,
    pub activation: Activation,
}
```

**Methods:**
- `new(config: TransformerConfig) -> Result<Self>`
- `forward(&self, input: &Tensor) -> Result<Tensor>`

### GRU

Gated Recurrent Unit.

```rust
pub struct GRUModel { /* ... */ }

pub struct GRUConfig {
    pub input_size: usize,
    pub horizon: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub bidirectional: bool,
}
```

**Methods:**
- `new(config: GRUConfig) -> Result<Self>`

### TCN

Temporal Convolutional Network.

```rust
pub struct TCNModel { /* ... */ }

pub struct TCNConfig {
    pub input_size: usize,
    pub horizon: usize,
    pub num_channels: Vec<usize>,
    pub kernel_size: usize,
    pub dropout: f64,
    pub dilation_base: usize,
}
```

**Methods:**
- `new(config: TCNConfig) -> Result<Self>`

### DeepAR

Probabilistic forecasting with LSTM.

```rust
pub struct DeepARModel { /* ... */ }

pub struct DeepARConfig {
    pub input_size: usize,
    pub horizon: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub distribution: DistributionType,
    pub quantiles: Vec<f64>,
    pub likelihood_weight: f64,
    pub dropout: f64,
}

pub enum DistributionType {
    Gaussian,
    StudentT,
    NegativeBinomial,
    ZeroInflatedNegativeBinomial,
}
```

**Methods:**
- `new(config: DeepARConfig) -> Result<Self>`
- `forward(&self, input: &Tensor) -> Result<Tensor>`
- `sample(&self, input: &Tensor, n_samples: usize) -> Result<Tensor>`

### N-BEATS

Neural Basis Expansion Analysis.

```rust
pub struct NBeatsModel { /* ... */ }

pub struct NBeatsConfig {
    pub input_size: usize,
    pub horizon: usize,
    pub num_stacks: usize,
    pub stack_types: Vec<StackType>,
    pub num_blocks_per_stack: usize,
    pub hidden_size: usize,
    pub expansion_coefficient_dim: usize,
}

pub enum StackType {
    Trend,
    Seasonality,
    Generic,
}
```

**Methods:**
- `new(config: NBeatsConfig) -> Result<Self>`
- `forward(&self, input: &Tensor) -> Result<(Tensor, Tensor, Tensor)>` - Returns (forecast, trend, seasonality)

### Prophet

Time series decomposition model.

```rust
pub struct ProphetModel { /* ... */ }

pub struct ProphetConfig {
    pub growth: GrowthModel,
    pub yearly_seasonality: bool,
    pub weekly_seasonality: bool,
    pub daily_seasonality: bool,
    pub seasonality_prior_scale: f64,
    pub changepoint_prior_scale: f64,
}

pub enum GrowthModel {
    Linear,
    Logistic,
}
```

**Methods:**
- `new(config: ProphetConfig) -> Result<Self>`
- `fit(&mut self, ds: &[DateTime<Utc>], y: &[f64]) -> Result<()>`
- `predict(&self, ds: &[DateTime<Utc>]) -> Result<Vec<f64>>`

## Training

### Trainer

Main training orchestrator.

```rust
#[cfg(feature = "candle")]
pub struct Trainer {
    pub config: TrainingConfig,
}
```

**Methods:**
- `new(config: TrainingConfig) -> Self`
- `train(&self, model: &impl NeuralModel, train_data: &DataLoader, val_data: Option<&DataLoader>) -> Result<TrainedModel>`
- `add_callback(&mut self, callback: TrainingCallback)`

### TrainingConfig

Training configuration.

```rust
pub struct TrainingConfig {
    pub batch_size: usize,
    pub num_epochs: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub gradient_clip: Option<f64>,
    pub early_stopping_patience: usize,
    pub early_stopping_min_delta: f64,
    pub validation_split: f32,
    pub mixed_precision: bool,
    pub num_workers: usize,
    pub lr_scheduler: Option<LRScheduler>,
}
```

**Methods:**
- `default() -> Self`

### TrainingMetrics

Training metrics.

```rust
pub struct TrainingMetrics {
    pub train_loss: f64,
    pub val_loss: f64,
    pub train_mae: f64,
    pub val_mae: f64,
    pub train_rmse: f64,
    pub val_rmse: f64,
    pub r2_score: f64,
    pub learning_rate: f64,
    pub epoch_time_secs: f64,
    pub total_epochs: usize,
}
```

### DataLoader

Data loading with batching.

```rust
#[cfg(feature = "candle")]
pub struct DataLoader {
    // ...
}
```

**Methods:**
- `new(dataset: TimeSeriesDataset, batch_size: usize, shuffle: bool, num_workers: usize) -> Result<Self>`
- `len(&self) -> usize`
- `iter(&self) -> impl Iterator<Item = Batch>`

### TimeSeriesDataset

Time series dataset.

```rust
#[cfg(feature = "candle")]
pub struct TimeSeriesDataset {
    // ...
}
```

**Methods:**
- `new(data: Vec<f64>, input_size: usize, horizon: usize, stride: usize) -> Result<Self>`
- `len(&self) -> usize`
- `get(&self, idx: usize) -> Result<(Tensor, Tensor)>`
- `split_at(&self, idx: usize) -> (Self, Self)`

## Inference

### Predictor

Single prediction interface.

```rust
#[cfg(feature = "candle")]
pub struct Predictor {
    // ...
}
```

**Methods:**
- `new(model: impl NeuralModel) -> Result<Self>`
- `predict(&self, input: &[f64]) -> Result<PredictionResult>`
- `predict_with_intervals(&self, input: &[f64], confidence: f64) -> Result<PredictionResult>`

### BatchPredictor

Batch prediction interface.

```rust
#[cfg(feature = "candle")]
pub struct BatchPredictor {
    // ...
}
```

**Methods:**
- `new(model: impl NeuralModel, batch_size: usize) -> Result<Self>`
- `predict_batch(&self, inputs: &[Vec<f64>]) -> Result<Vec<PredictionResult>>`
- `throughput(&self) -> f64` - Predictions per second

### PredictionResult

Prediction output.

```rust
pub struct PredictionResult {
    pub values: Vec<f64>,
    pub confidence: Option<f64>,
    pub intervals: Option<(Vec<f64>, Vec<f64>)>,  // (lower, upper)
    pub inference_time_ms: f64,
}
```

## Storage

### AgentDbStorage

AgentDB storage backend.

```rust
pub struct AgentDbStorage {
    // ...
}
```

**Methods:**
- `new(db_path: impl AsRef<Path>) -> Result<Self>`
- `with_config(config: AgentDbConfig) -> Result<Self>`
- `save_model(&self, model_bytes: &[u8], metadata: ModelMetadata) -> Result<ModelId>`
- `load_model(&self, model_id: &str) -> Result<Vec<u8>>`
- `get_metadata(&self, model_id: &str) -> Result<ModelMetadata>`
- `list_models(&self, filter: Option<SearchFilter>) -> Result<Vec<ModelMetadata>>`
- `search_similar_models(&self, embedding: &[f32], k: usize) -> Result<Vec<SearchResult>>`
- `search_similar_models_with_metric(&self, embedding: &[f32], k: usize, metric: SimilarityMetric) -> Result<Vec<SearchResult>>`
- `save_checkpoint(&self, model_id: &str, checkpoint: ModelCheckpoint, state_bytes: &[u8]) -> Result<String>`
- `load_checkpoint(&self, checkpoint_id: &str) -> Result<(ModelCheckpoint, Vec<u8>)>`
- `get_stats(&self) -> Result<serde_json::Value>`
- `export(&self, output_path: impl AsRef<Path>, compress: bool) -> Result<()>`

### ModelMetadata

Model metadata.

```rust
pub struct ModelMetadata {
    pub name: String,
    pub model_type: String,
    pub version: String,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub hyperparameters: Option<serde_json::Value>,
    pub training_config: Option<serde_json::Value>,
    pub metrics: Option<serde_json::Value>,
    pub dataset_info: Option<serde_json::Value>,
    pub hardware: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub model_size_bytes: usize,
}
```

### ModelCheckpoint

Training checkpoint.

```rust
pub struct ModelCheckpoint {
    pub checkpoint_id: String,
    pub model_id: String,
    pub epoch: usize,
    pub loss: f64,
    pub metrics: Option<serde_json::Value>,
    pub optimizer_config: Option<serde_json::Value>,
    pub timestamp: DateTime<Utc>,
}
```

### SearchFilter

Model search filter.

```rust
pub struct SearchFilter {
    pub model_type: Option<String>,
    pub tags: Option<Vec<String>>,
    pub min_val_loss: Option<f64>,
    pub max_val_loss: Option<f64>,
    pub created_after: Option<DateTime<Utc>>,
    pub created_before: Option<DateTime<Utc>>,
}
```

### SearchResult

Similarity search result.

```rust
pub struct SearchResult {
    pub model_id: String,
    pub score: f64,
    pub metadata: ModelMetadata,
}
```

### SimilarityMetric

Similarity metric for vector search.

```rust
pub enum SimilarityMetric {
    Cosine,
    L2,
    L1,
}
```

## Utilities

### Preprocessing

Data preprocessing functions.

```rust
pub mod preprocessing {
    pub fn normalize(data: &[f64]) -> (Vec<f64>, NormParams);
    pub fn denormalize(data: &[f64], params: &NormParams) -> Vec<f64>;
    pub fn min_max_normalize(data: &[f64]) -> (Vec<f64>, MinMaxParams);
    pub fn robust_scale(data: &[f64]) -> (Vec<f64>, f64, f64);
    pub fn difference(data: &[f64], lag: usize) -> Vec<f64>;
    pub fn inverse_difference(diff: &[f64], initial: &[f64], lag: usize) -> Vec<f64>;
    pub fn detrend(data: &[f64]) -> (Vec<f64>, f64, f64);
    pub fn seasonal_decompose(data: &[f64], period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>);
    pub fn remove_outliers(data: &[f64], threshold: f64) -> Vec<f64>;
    pub fn winsorize(data: &[f64], lower_percentile: f64, upper_percentile: f64) -> Vec<f64>;
}
```

### Features

Feature engineering functions.

```rust
pub mod features {
    pub fn create_lags(data: &[f64], lags: &[usize]) -> Vec<Vec<f64>>;
    pub fn rolling_mean(data: &[f64], window: usize) -> Vec<f64>;
    pub fn rolling_std(data: &[f64], window: usize) -> Vec<f64>;
    pub fn rolling_min(data: &[f64], window: usize) -> Vec<f64>;
    pub fn rolling_max(data: &[f64], window: usize) -> Vec<f64>;
    pub fn ema(data: &[f64], alpha: f64) -> Vec<f64>;
    pub fn rate_of_change(data: &[f64], period: usize) -> Vec<f64>;
    pub fn fourier_features(n: usize, period: usize, order: usize) -> Vec<Vec<f64>>;
    pub fn calendar_features(timestamps: &[DateTime<Utc>]) -> Vec<Vec<f64>>;
}
```

### Metrics

Evaluation metrics.

```rust
pub mod metrics {
    pub struct EvaluationMetrics {
        pub mae: f64,
        pub rmse: f64,
        pub mape: f64,
        pub smape: f64,
        pub r2_score: f64,
    }

    impl EvaluationMetrics {
        pub fn calculate(y_true: &[f64], y_pred: &[f64], sample_weight: Option<&[f64]>) -> Result<Self>;
    }

    pub fn directional_accuracy(y_true: &[f64], y_pred: &[f64]) -> f64;
    pub fn prediction_interval_coverage(y_true: &[f64], lower: &[f64], upper: &[f64]) -> f64;
}
```

### Validation

Cross-validation utilities.

```rust
pub mod validation {
    pub fn time_series_split(data: &[f64], n_splits: usize, test_size: usize) -> Vec<(Vec<usize>, Vec<usize>)>;
    pub fn rolling_window_cv(data: &[f64], window_size: usize, horizon: usize) -> Vec<(Vec<usize>, Vec<usize>)>;
    pub fn expanding_window_cv(data: &[f64], min_train_size: usize, horizon: usize) -> Vec<(Vec<usize>, Vec<usize>)>;
    pub fn grid_search<F>(param_grid: &[(String, Vec<f64>)], eval_fn: F) -> Result<HashMap<String, f64>>
    where
        F: Fn(&HashMap<String, f64>) -> Result<f64>;
}
```

## Error Handling

### NeuralError

Error type for neural operations.

```rust
pub enum NeuralError {
    ModelInitialization(String),
    TrainingError(String),
    InferenceError(String),
    StorageError(String),
    InputShapeMismatch { expected: usize, got: usize },
    NotImplemented(String),
    IO(std::io::Error),
    Serialization(serde_json::Error),
}
```

**Methods:**
- `not_implemented(msg: impl Into<String>) -> Self`

## Feature Flags

| Feature | Description |
|---------|-------------|
| `default` | CPU-only mode |
| `candle` | Enable Candle ML framework |
| `cuda` | CUDA GPU acceleration |
| `metal` | Metal GPU acceleration (macOS) |
| `accelerate` | Apple Accelerate CPU optimization |

## Examples

See the [`examples/`](../../neural-trader-rust/crates/neural/examples/) directory for complete usage examples.

## Related Documentation

- [Quick Start Guide](QUICKSTART.md)
- [Model Comparison](MODELS.md)
- [Training Guide](TRAINING.md)
- [Inference Guide](INFERENCE.md)
- [AgentDB Integration](AGENTDB.md)
