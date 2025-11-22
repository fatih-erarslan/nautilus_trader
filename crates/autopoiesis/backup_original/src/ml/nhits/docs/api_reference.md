# API Reference

Complete API documentation for the NHITS (Neural Hierarchical Interpolation for Time Series) system.

## Table of Contents

- [Core Types](#core-types)
- [NHITS Model](#nhits-model)
- [Configuration](#configuration)
- [Neural Blocks](#neural-blocks)
- [Attention Mechanisms](#attention-mechanisms)
- [Decomposition](#decomposition)
- [Adaptation](#adaptation)
- [Forecasting Pipeline](#forecasting-pipeline)
- [Error Types](#error-types)
- [Utilities](#utilities)

## Core Types

### NHITS

The main model struct that orchestrates consciousness-aware hierarchical time series forecasting.

```rust
pub struct NHITS {
    config: NHITSConfig,
    blocks: Vec<HierarchicalBlock>,
    attention: TemporalAttention,
    decomposer: MultiScaleDecomposer,
    adapter: AdaptiveStructure,
    consciousness: Arc<ConsciousnessField>,
    autopoietic_system: Arc<AutopoieticSystem>,
    state: ModelState,
}
```

#### Constructor

```rust
impl NHITS {
    pub fn new(
        config: NHITSConfig,
        consciousness: Arc<ConsciousnessField>,
        autopoietic_system: Arc<AutopoieticSystem>,
    ) -> Self
}
```

**Parameters:**
- `config`: Model configuration specifying architecture and hyperparameters
- `consciousness`: Shared consciousness field for attention modulation
- `autopoietic_system`: Self-organizing system for adaptive behavior

**Example:**
```rust
let consciousness = Arc::new(ConsciousnessField::new());
let autopoietic = Arc::new(AutopoieticSystem::new());
let config = NHITSConfig::default();
let model = NHITS::new(config, consciousness, autopoietic);
```

#### Forward Pass

```rust
pub fn forward(
    &mut self,
    input: &Array3<f64>,
    lookback_window: usize,
    forecast_horizon: usize,
) -> Result<Array3<f64>, NHITSError>
```

**Parameters:**
- `input`: Input time series data with shape `(batch_size, sequence_length, features)`
- `lookback_window`: Number of historical time steps to consider
- `forecast_horizon`: Number of future time steps to predict

**Returns:**
- `Array3<f64>`: Predictions with shape `(batch_size, forecast_horizon, output_features)`

**Example:**
```rust
let input = Array3::zeros((32, 168, 1)); // 32 samples, 168 hours, 1 feature
let predictions = model.forward(&input, 168, 24)?; // 24-hour forecast
```

#### Training

```rust
pub fn train(
    &mut self,
    train_data: &Array3<f64>,
    val_data: Option<&Array3<f64>>,
    epochs: usize,
) -> Result<TrainingHistory, NHITSError>
```

**Parameters:**
- `train_data`: Training dataset
- `val_data`: Optional validation dataset
- `epochs`: Number of training epochs

**Returns:**
- `TrainingHistory`: Training metrics and performance history

#### Prediction

```rust
pub fn predict(
    &self,
    input: &Array1<f64>,
    horizon: usize,
) -> Result<Array1<f64>, NHITSError>
```

**Parameters:**
- `input`: 1D input sequence
- `horizon`: Forecast horizon

**Returns:**
- `Array1<f64>`: 1D predictions

#### Online Learning

```rust
pub fn update_online(
    &mut self,
    inputs: &[Array1<f64>],
    targets: &[Array1<f64>],
) -> Result<(), NHITSError>
```

**Parameters:**
- `inputs`: New input sequences
- `targets`: Corresponding target values

#### State Management

```rust
pub fn save_state(&self) -> Result<ModelState, NHITSError>
pub fn load_state(&mut self, state: ModelState) -> Result<(), NHITSError>
pub fn reset_weights(&mut self) -> Result<(), NHITSError>
```

### ModelState

Tracks model training state and performance metrics.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    pub epoch: usize,
    pub total_predictions: usize,
    pub performance_history: Vec<f64>,
    pub structural_changes: Vec<StructuralChange>,
    pub consciousness_coherence: f64,
}
```

### TrainingHistory

Records training progress and metrics.

```rust
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    pub train_losses: Vec<f64>,
    pub val_losses: Vec<f64>,
    pub best_epoch: usize,
    pub best_val_loss: f64,
}
```

## Configuration

### NHITSConfig

Main configuration struct for the NHITS model.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NHITSConfig {
    // Architecture
    pub block_configs: Vec<BlockConfig>,
    pub attention_config: AttentionConfig,
    pub decomposer_config: DecomposerConfig,
    pub adaptation_config: AdaptationConfig,
    
    // Time series parameters
    pub lookback_window: usize,
    pub forecast_horizon: usize,
    pub input_features: usize,
    pub output_features: usize,
    
    // Training parameters
    pub learning_rate: f64,
    pub batch_size: usize,
    pub gradient_clip: f64,
    pub weight_decay: f64,
    
    // Consciousness integration
    pub consciousness_enabled: bool,
    pub coherence_weight: f64,
    pub min_coherence_threshold: f64,
    
    // Early stopping
    pub early_stop_patience: usize,
    pub early_stop_threshold: f64,
    
    // Advanced features
    pub use_residual_connections: bool,
    pub use_layer_norm: bool,
    pub use_dropout: bool,
    pub dropout_rate: f64,
}
```

#### Methods

```rust
impl NHITSConfig {
    pub fn default() -> Self
    pub fn minimal() -> Self
    pub fn for_use_case(use_case: UseCase) -> Self
    pub fn validate(&self) -> Result<(), ConfigError>
}
```

### NHITSConfigBuilder

Fluent builder for configuration creation.

```rust
impl NHITSConfigBuilder {
    pub fn new() -> Self
    pub fn with_lookback(self, window: usize) -> Self
    pub fn with_horizon(self, horizon: usize) -> Self
    pub fn with_features(self, input: usize, output: usize) -> Self
    pub fn with_consciousness(self, enabled: bool, weight: f64) -> Self
    pub fn with_adaptation(self, strategy: AdaptationStrategy, rate: f64) -> Self
    pub fn with_blocks(self, blocks: Vec<BlockConfig>) -> Self
    pub fn build(self) -> Result<NHITSConfig, ConfigError>
}
```

**Example:**
```rust
let config = NHITSConfigBuilder::new()
    .with_lookback(168)
    .with_horizon(24)
    .with_features(1, 1)
    .with_consciousness(true, 0.1)
    .build()?;
```

### UseCase

Predefined configuration templates.

```rust
#[derive(Debug, Clone, Copy)]
pub enum UseCase {
    ShortTermForecasting,
    LongTermForecasting,
    MultivariateSeries,
    HighFrequencyTrading,
    AnomalyDetection,
    SeasonalDecomposition,
}
```

## Neural Blocks

### HierarchicalBlock

Multi-scale neural processing unit with basis expansion.

```rust
pub struct HierarchicalBlock {
    config: BlockConfig,
    // Internal layers and parameters
}
```

### BlockConfig

Configuration for individual hierarchical blocks.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_basis: usize,
    pub pooling_factor: usize,
    pub pooling_type: PoolingType,
    pub interpolation_type: InterpolationType,
    pub dropout_rate: f64,
    pub activation: ActivationType,
}
```

### ActivationType

Available activation functions.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    GELU,
    Swish,
    Tanh,
    Sigmoid,
}
```

### PoolingType

Pooling operation variants.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolingType {
    Average,
    Max,
    Adaptive,
}
```

### InterpolationType

Interpolation methods for upsampling.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterpolationType {
    Linear,
    Cubic,
    Spline { tension: f64 },
}
```

## Attention Mechanisms

### TemporalAttention

Consciousness-aware attention mechanism for temporal sequences.

```rust
pub struct TemporalAttention {
    config: AttentionConfig,
    // Attention parameters
}
```

#### Methods

```rust
impl TemporalAttention {
    pub fn new(config: &AttentionConfig) -> Self
    pub fn apply(
        &self,
        input: &Array3<f64>,
        consciousness_weights: Option<Array2<f64>>,
    ) -> Result<Array3<f64>, NHITSError>
    pub fn reconfigure_heads(&mut self, num_heads: usize)
}
```

### AttentionConfig

Configuration for attention mechanisms.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub dropout_rate: f64,
    pub temperature: f64,
    pub use_causal_mask: bool,
    pub attention_type: AttentionType,
    pub consciousness_integration: bool,
}
```

### AttentionType

Different attention mechanism variants.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionType {
    Standard,
    Relative,
    Sparse { sparsity_factor: f64 },
    LocalWindow { window_size: usize },
}
```

## Decomposition

### MultiScaleDecomposer

Advanced time series decomposition methods.

```rust
pub struct MultiScaleDecomposer {
    config: DecomposerConfig,
    // Decomposition components
}
```

#### Methods

```rust
impl MultiScaleDecomposer {
    pub fn new(config: &DecomposerConfig) -> Self
    pub fn decompose(&self, input: &Array3<f64>) -> Result<Array3<f64>, NHITSError>
}
```

### DecomposerConfig

Configuration for time series decomposition.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecomposerConfig {
    pub decomposition_type: DecompositionType,
    pub num_scales: usize,
    pub seasonal_periods: Vec<usize>,
    pub trend_filter_size: usize,
    pub use_stl: bool,
    pub robust: bool,
}
```

### DecompositionType

Available decomposition methods.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecompositionType {
    Additive,
    Multiplicative,
    STL,
    EMD,
    Hybrid,
}
```

## Adaptation

### AdaptiveStructure

Manages dynamic model architecture evolution.

```rust
pub struct AdaptiveStructure {
    config: AdaptationConfig,
    // Adaptation state
}
```

#### Methods

```rust
impl AdaptiveStructure {
    pub fn new(config: &AdaptationConfig) -> Self
    pub fn evaluate(
        &self,
        performance_history: &[f64],
        consciousness_state: &ConsciousnessState,
        config: &AdaptationConfig,
    ) -> Result<Option<StructuralChange>, NHITSError>
}
```

### AdaptationConfig

Configuration for adaptive structure evolution.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationConfig {
    pub adaptation_rate: f64,
    pub performance_window: usize,
    pub change_threshold: f64,
    pub max_depth: usize,
    pub min_depth: usize,
    pub consciousness_weight: f64,
    pub exploration_rate: f64,
    pub adaptation_strategy: AdaptationStrategy,
}
```

### AdaptationStrategy

Strategies for model adaptation.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    Conservative,
    Balanced,
    Aggressive,
    ConsciousnessGuided,
}
```

### StructuralChange

Records architectural modifications.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralChange {
    pub timestamp: u64,
    pub change_type: ChangeType,
    pub performance_impact: f64,
    pub consciousness_influence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    BlockAdded { depth: usize, units: usize },
    BlockRemoved { depth: usize },
    AttentionReconfigured { heads: usize },
    PoolingAdjusted { factor: usize },
    BasisExpanded { new_basis: usize },
}
```

## Forecasting Pipeline

### ForecastingPipeline

Production-ready forecasting system with ensemble methods and online learning.

```rust
pub struct ForecastingPipeline {
    config: ForecastingConfig,
    models: Vec<NHITS>,
    consciousness: Arc<ConsciousnessField>,
    autopoietic_system: Arc<AutopoieticSystem>,
}
```

#### Methods

```rust
impl ForecastingPipeline {
    pub async fn new(
        config: ForecastingConfig,
        consciousness: Arc<ConsciousnessField>,
        autopoietic_system: Arc<AutopoieticSystem>,
    ) -> Result<Self, NHITSError>
    
    pub async fn forecast(
        &self,
        input: &Array2<f64>,
        external_features: Option<&Array2<f64>>,
    ) -> Result<ForecastResult, NHITSError>
    
    pub async fn update(
        &mut self,
        input: &Array1<f64>,
        target: &Array1<f64>,
    ) -> Result<(), NHITSError>
    
    pub fn subscribe(&self) -> Receiver<ForecastingEvent>
}
```

### ForecastingConfig

Configuration for the forecasting pipeline.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastingConfig {
    pub horizons: Vec<usize>,
    pub ensemble_size: usize,
    pub confidence_levels: Vec<f64>,
    pub online_learning: bool,
    pub online_window_size: usize,
    pub update_frequency: usize,
    pub anomaly_threshold: f64,
    pub retraining_config: RetrainingConfig,
    pub preprocessing_config: PreprocessingConfig,
    pub persistence_config: PersistenceConfig,
}
```

### ForecastResult

Comprehensive forecast results with uncertainty quantification.

```rust
#[derive(Debug, Clone)]
pub struct ForecastResult {
    pub forecasts: HashMap<usize, Array1<f64>>,
    pub intervals: HashMap<(usize, f64), (Array1<f64>, Array1<f64>)>,
    pub anomaly_scores: Array1<f64>,
    pub performance_metrics: PerformanceMetrics,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
```

### PerformanceMetrics

Model performance tracking.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub mae: f64,
    pub rmse: f64,
    pub mape: f64,
    pub interval_coverage: HashMap<f64, f64>,
    pub forecast_bias: f64,
}
```

### ForecastingEvent

Event system for monitoring and notifications.

```rust
#[derive(Debug, Clone)]
pub enum ForecastingEvent {
    ForecastGenerated {
        timestamp: chrono::DateTime<chrono::Utc>,
        horizons: Vec<usize>,
    },
    AnomalyDetected {
        timestamp: chrono::DateTime<chrono::Utc>,
        score: f64,
        value: f64,
    },
    ModelRetrained {
        timestamp: chrono::DateTime<chrono::Utc>,
        version: String,
        performance_improvement: f64,
    },
    PerformanceDegraded {
        timestamp: chrono::DateTime<chrono::Utc>,
        metric: String,
        degradation: f64,
    },
    ConceptDriftDetected {
        timestamp: chrono::DateTime<chrono::Utc>,
        drift_score: f64,
    },
}
```

## Error Types

### NHITSError

Comprehensive error handling for the NHITS system.

```rust
#[derive(Debug, thiserror::Error)]
pub enum NHITSError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { 
        expected: Vec<usize>, 
        got: Vec<usize> 
    },
    
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    #[error("Computation error: {0}")]
    ComputationError(String),
    
    #[error("Adaptation error: {0}")]
    AdaptationError(String),
}
```

### ConfigError

Configuration validation errors.

```rust
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Incompatible configuration: {0}")]
    IncompatibleConfig(String),
}
```

## Utilities

### Prelude Module

Convenient imports for common usage.

```rust
pub mod prelude {
    pub use super::core::*;
    pub use super::configs::*;
    pub use super::blocks::BlockConfig;
    pub use super::attention::AttentionConfig;
    pub use super::decomposition::DecomposerConfig;
    pub use super::adaptation::AdaptationConfig;
    pub use super::forecasting::{
        ForecastingPipeline, 
        ForecastingConfig, 
        ForecastResult
    };
}
```

**Usage:**
```rust
use autopoiesis::ml::nhits::prelude::*;
```

## Feature Flags

The NHITS system supports conditional compilation with feature flags:

```toml
[dependencies]
autopoiesis = { version = "0.1.0", features = ["ml", "optimization"] }
```

Available features:
- `ml`: Core machine learning functionality
- `optimization`: Advanced optimization algorithms
- `metrics`: Performance monitoring and metrics
- `full`: All features enabled

## Version Compatibility

| Version | Rust Version | Key Features |
|---------|-------------|--------------|
| 0.1.0   | 1.70+       | Initial release with consciousness integration |

## Thread Safety

All public APIs are designed to be thread-safe where appropriate:

- `NHITS`: Not `Send`/`Sync` due to mutable training state
- `NHITSConfig`: `Send` + `Sync` (immutable configuration)
- `ForecastingPipeline`: `Send` + `Sync` with internal synchronization
- Shared systems (`ConsciousnessField`, `AutopoieticSystem`): `Send` + `Sync`

## Memory Management

The NHITS system is designed for efficient memory usage:

- Zero-copy operations where possible using `ndarray` views
- Memory-mapped model persistence
- Streaming processing for large time series
- Configurable buffer sizes for online learning

## Performance Characteristics

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Forward Pass | O(n * h * b) | O(n * h) |
| Training Step | O(n * h * b) | O(n * h) |
| Online Update | O(h * b) | O(w) |

Where:
- n = sequence length
- h = hidden size
- b = number of blocks
- w = online window size