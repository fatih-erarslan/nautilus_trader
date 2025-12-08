//! Configuration for ML training infrastructure

use serde::{Serialize, Deserialize};
use std::path::PathBuf;

/// Main training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Data configuration
    pub data: DataConfig,
    
    /// Model configurations
    pub models: ModelConfigs,
    
    /// Training parameters
    pub training: TrainingParams,
    
    /// Validation configuration
    pub validation: ValidationConfig,
    
    /// Optimization configuration
    pub optimization: OptimizationConfig,
    
    /// Persistence configuration
    pub persistence: PersistenceConfig,
    
    /// Experiment tracking configuration
    pub experiments: ExperimentConfig,
    
    /// Deployment configuration
    pub deployment: DeploymentConfig,
    
    /// GPU configuration
    #[cfg(feature = "gpu")]
    pub gpu: GPUConfig,
}

/// Data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Data source path
    pub source_path: PathBuf,
    
    /// Batch size for training
    pub batch_size: usize,
    
    /// Sequence length for time series
    pub sequence_length: usize,
    
    /// Prediction horizon
    pub horizon: usize,
    
    /// Features to use
    pub features: Vec<String>,
    
    /// Target variable
    pub target: String,
    
    /// Train/validation/test split ratios
    pub split_ratios: (f32, f32, f32),
    
    /// Data normalization method
    pub normalization: NormalizationMethod,
    
    /// Cache preprocessed data
    pub enable_cache: bool,
    
    /// Parallel data loading
    pub num_workers: usize,
}

/// Model configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfigs {
    /// Transformer configuration
    pub transformer: TransformerConfig,
    
    /// LSTM configuration
    pub lstm: LSTMConfig,
    
    /// XGBoost configuration
    pub xgboost: XGBoostConfig,
    
    /// LightGBM configuration
    pub lightgbm: LightGBMConfig,
    
    /// Neural network configuration
    pub neural: NeuralConfig,
}

/// Transformer model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Number of layers
    pub num_layers: usize,
    
    /// Hidden dimension
    pub d_model: usize,
    
    /// Number of attention heads
    pub num_heads: usize,
    
    /// Feedforward dimension
    pub d_ff: usize,
    
    /// Dropout rate
    pub dropout: f32,
    
    /// Maximum sequence length
    pub max_seq_length: usize,
    
    /// Use positional encoding
    pub use_positional_encoding: bool,
}

/// LSTM model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTMConfig {
    /// Number of LSTM layers
    pub num_layers: usize,
    
    /// Hidden size
    pub hidden_size: usize,
    
    /// Dropout rate
    pub dropout: f32,
    
    /// Bidirectional LSTM
    pub bidirectional: bool,
    
    /// Use attention mechanism
    pub use_attention: bool,
}

/// XGBoost configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XGBoostConfig {
    /// Number of boosting rounds
    pub n_estimators: u32,
    
    /// Maximum tree depth
    pub max_depth: u32,
    
    /// Learning rate
    pub learning_rate: f32,
    
    /// Subsample ratio
    pub subsample: f32,
    
    /// Column subsample ratio
    pub colsample_bytree: f32,
    
    /// Regularization alpha
    pub reg_alpha: f32,
    
    /// Regularization lambda
    pub reg_lambda: f32,
    
    /// Objective function
    pub objective: String,
    
    /// Evaluation metric
    pub eval_metric: String,
    
    /// Early stopping rounds
    pub early_stopping_rounds: Option<u32>,
    
    /// Use GPU
    pub tree_method: String,
}

/// LightGBM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightGBMConfig {
    /// Boosting type
    pub boosting_type: String,
    
    /// Number of iterations
    pub num_iterations: u32,
    
    /// Learning rate
    pub learning_rate: f32,
    
    /// Number of leaves
    pub num_leaves: u32,
    
    /// Maximum depth
    pub max_depth: i32,
    
    /// Feature fraction
    pub feature_fraction: f32,
    
    /// Bagging fraction
    pub bagging_fraction: f32,
    
    /// Bagging frequency
    pub bagging_freq: u32,
    
    /// Lambda L1
    pub lambda_l1: f32,
    
    /// Lambda L2
    pub lambda_l2: f32,
    
    /// Objective function
    pub objective: String,
    
    /// Evaluation metric
    pub metric: String,
    
    /// Early stopping rounds
    pub early_stopping_rounds: Option<u32>,
    
    /// Device type
    pub device_type: String,
}

/// Neural network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    /// Layer sizes
    pub layer_sizes: Vec<usize>,
    
    /// Activation function
    pub activation: ActivationType,
    
    /// Dropout rates per layer
    pub dropout_rates: Vec<f32>,
    
    /// Use batch normalization
    pub batch_norm: bool,
    
    /// Weight initialization
    pub weight_init: WeightInit,
}

/// Training parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    /// Number of epochs
    pub epochs: usize,
    
    /// Learning rate
    pub learning_rate: f32,
    
    /// Learning rate scheduler
    pub lr_scheduler: LRScheduler,
    
    /// Optimizer type
    pub optimizer: OptimizerType,
    
    /// Gradient clipping
    pub gradient_clip: Option<f32>,
    
    /// Early stopping patience
    pub early_stopping_patience: usize,
    
    /// Checkpoint frequency
    pub checkpoint_every: usize,
    
    /// Mixed precision training
    pub mixed_precision: bool,
    
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Cross-validation strategy
    pub cv_strategy: CVStrategy,
    
    /// Number of folds
    pub n_folds: usize,
    
    /// Time series gap
    pub gap: usize,
    
    /// Validation metrics
    pub metrics: Vec<MetricType>,
    
    /// Walk-forward analysis
    pub walk_forward: bool,
    
    /// Purged cross-validation
    pub purged: bool,
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Hyperparameter optimization method
    pub method: OptimizationMethod,
    
    /// Number of trials
    pub n_trials: usize,
    
    /// Timeout in seconds
    pub timeout: Option<u64>,
    
    /// Random seed
    pub seed: Option<u64>,
    
    /// Parallel trials
    pub n_jobs: usize,
    
    /// Pruning strategy
    pub pruning: bool,
}

/// Persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Model storage path
    pub model_path: PathBuf,
    
    /// Checkpoint path
    pub checkpoint_path: PathBuf,
    
    /// Save format
    pub format: SaveFormat,
    
    /// Compression
    pub compression: bool,
    
    /// Versioning strategy
    pub versioning: VersioningStrategy,
    
    /// Maximum saved versions
    pub max_versions: usize,
}

/// Experiment tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Database URL
    pub database_url: String,
    
    /// Experiment name prefix
    pub name_prefix: String,
    
    /// Track parameters
    pub track_params: bool,
    
    /// Track metrics
    pub track_metrics: bool,
    
    /// Track artifacts
    pub track_artifacts: bool,
    
    /// Log frequency
    pub log_frequency: usize,
}

/// Deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    /// Model registry path
    pub registry_path: PathBuf,
    
    /// Serving format
    pub serving_format: ServingFormat,
    
    /// Model optimization
    pub optimize_for_inference: bool,
    
    /// Quantization
    pub quantization: Option<QuantizationType>,
    
    /// Maximum model size (MB)
    pub max_model_size: Option<usize>,
}

/// GPU configuration
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUConfig {
    /// Device ID
    pub device_id: usize,
    
    /// Memory fraction
    pub memory_fraction: f32,
    
    /// Enable TensorRT
    pub tensorrt: bool,
    
    /// Enable cuDNN
    pub cudnn: bool,
}

/// Normalization methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Standard normalization (z-score)
    Standard,
    /// Min-max normalization
    MinMax,
    /// Robust scaling
    Robust,
    /// No normalization
    None,
}

/// Activation types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationType {
    /// ReLU activation
    ReLU,
    /// Leaky ReLU
    LeakyReLU,
    /// GELU
    GELU,
    /// Tanh
    Tanh,
    /// Sigmoid
    Sigmoid,
    /// Swish
    Swish,
}

/// Weight initialization methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WeightInit {
    /// Xavier/Glorot initialization
    Xavier,
    /// He initialization
    He,
    /// Normal distribution
    Normal,
    /// Uniform distribution
    Uniform,
}

/// Learning rate schedulers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LRScheduler {
    /// Constant learning rate
    Constant,
    /// Step decay
    StepLR { step_size: usize, gamma: f32 },
    /// Exponential decay
    ExponentialLR { gamma: f32 },
    /// Cosine annealing
    CosineAnnealingLR { t_max: usize },
    /// Reduce on plateau
    ReduceLROnPlateau { patience: usize, factor: f32 },
}

/// Optimizer types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Adam optimizer
    Adam,
    /// AdamW optimizer
    AdamW,
    /// SGD optimizer
    SGD,
    /// RMSprop optimizer
    RMSprop,
    /// LAMB optimizer
    LAMB,
}

/// Cross-validation strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CVStrategy {
    /// Time series split
    TimeSeriesSplit,
    /// Purged K-fold
    PurgedKFold,
    /// Walk-forward analysis
    WalkForward,
    /// Combinatorial purged CV
    CombinatorialPurged,
}

/// Metric types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MetricType {
    /// Mean Squared Error
    MSE,
    /// Mean Absolute Error
    MAE,
    /// Root Mean Squared Error
    RMSE,
    /// Mean Absolute Percentage Error
    MAPE,
    /// R-squared
    R2,
    /// Sharpe Ratio
    SharpeRatio,
    /// Maximum Drawdown
    MaxDrawdown,
    /// Hit Rate
    HitRate,
    /// Profit Factor
    ProfitFactor,
}

/// Optimization methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationMethod {
    /// Bayesian optimization
    Bayesian,
    /// Grid search
    GridSearch,
    /// Random search
    RandomSearch,
    /// Optuna
    Optuna,
    /// Hyperband
    Hyperband,
}

/// Save formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SaveFormat {
    /// PyTorch format
    PyTorch,
    /// ONNX format
    ONNX,
    /// TensorFlow SavedModel
    TensorFlow,
    /// Custom binary format
    Binary,
}

/// Versioning strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VersioningStrategy {
    /// Semantic versioning
    Semantic,
    /// Timestamp-based
    Timestamp,
    /// Hash-based
    Hash,
    /// Sequential
    Sequential,
}

/// Serving formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ServingFormat {
    /// REST API
    REST,
    /// gRPC
    GRPC,
    /// WebSocket
    WebSocket,
    /// Embedded
    Embedded,
}

/// Quantization types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QuantizationType {
    /// INT8 quantization
    INT8,
    /// FP16 quantization
    FP16,
    /// Dynamic quantization
    Dynamic,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            data: DataConfig::default(),
            models: ModelConfigs::default(),
            training: TrainingParams::default(),
            validation: ValidationConfig::default(),
            optimization: OptimizationConfig::default(),
            persistence: PersistenceConfig::default(),
            experiments: ExperimentConfig::default(),
            deployment: DeploymentConfig::default(),
            #[cfg(feature = "gpu")]
            gpu: GPUConfig::default(),
        }
    }
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            source_path: PathBuf::from("data"),
            batch_size: 32,
            sequence_length: 100,
            horizon: 10,
            features: vec![
                "open".to_string(),
                "high".to_string(),
                "low".to_string(),
                "close".to_string(),
                "volume".to_string(),
            ],
            target: "close".to_string(),
            split_ratios: (0.7, 0.15, 0.15),
            normalization: NormalizationMethod::Standard,
            enable_cache: true,
            num_workers: 4,
        }
    }
}

impl Default for ModelConfigs {
    fn default() -> Self {
        Self {
            transformer: TransformerConfig::default(),
            lstm: LSTMConfig::default(),
            xgboost: XGBoostConfig::default(),
            lightgbm: LightGBMConfig::default(),
            neural: NeuralConfig::default(),
        }
    }
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            num_layers: 6,
            d_model: 512,
            num_heads: 8,
            d_ff: 2048,
            dropout: 0.1,
            max_seq_length: 100,
            use_positional_encoding: true,
        }
    }
}

impl Default for LSTMConfig {
    fn default() -> Self {
        Self {
            num_layers: 2,
            hidden_size: 256,
            dropout: 0.1,
            bidirectional: true,
            use_attention: false,
        }
    }
}

impl Default for XGBoostConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: 6,
            learning_rate: 0.1,
            subsample: 0.8,
            colsample_bytree: 0.8,
            reg_alpha: 0.0,
            reg_lambda: 1.0,
            objective: "reg:squarederror".to_string(),
            eval_metric: "rmse".to_string(),
            early_stopping_rounds: Some(10),
            tree_method: "hist".to_string(),
        }
    }
}

impl Default for LightGBMConfig {
    fn default() -> Self {
        Self {
            boosting_type: "gbdt".to_string(),
            num_iterations: 100,
            learning_rate: 0.1,
            num_leaves: 31,
            max_depth: -1,
            feature_fraction: 0.9,
            bagging_fraction: 0.8,
            bagging_freq: 5,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            objective: "regression".to_string(),
            metric: "rmse".to_string(),
            early_stopping_rounds: Some(10),
            device_type: "cpu".to_string(),
        }
    }
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![128, 64, 32],
            activation: ActivationType::ReLU,
            dropout_rates: vec![0.2, 0.2, 0.1],
            batch_norm: true,
            weight_init: WeightInit::He,
        }
    }
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            epochs: 100,
            learning_rate: 0.001,
            lr_scheduler: LRScheduler::ReduceLROnPlateau { patience: 10, factor: 0.5 },
            optimizer: OptimizerType::Adam,
            gradient_clip: Some(1.0),
            early_stopping_patience: 20,
            checkpoint_every: 10,
            mixed_precision: false,
            gradient_accumulation_steps: 1,
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            cv_strategy: CVStrategy::TimeSeriesSplit,
            n_folds: 5,
            gap: 10,
            metrics: vec![MetricType::MSE, MetricType::MAE, MetricType::SharpeRatio],
            walk_forward: false,
            purged: true,
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            method: OptimizationMethod::Bayesian,
            n_trials: 50,
            timeout: Some(3600),
            seed: Some(42),
            n_jobs: 4,
            pruning: true,
        }
    }
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models"),
            checkpoint_path: PathBuf::from("checkpoints"),
            format: SaveFormat::Binary,
            compression: true,
            versioning: VersioningStrategy::Timestamp,
            max_versions: 10,
        }
    }
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            database_url: "sqlite://experiments.db".to_string(),
            name_prefix: "exp".to_string(),
            track_params: true,
            track_metrics: true,
            track_artifacts: true,
            log_frequency: 10,
        }
    }
}

#[cfg(feature = "gpu")]
impl Default for GPUConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            memory_fraction: 0.9,
            tensorrt: false,
            cudnn: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_serialization() {
        let config = TrainingConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: TrainingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.data.batch_size, deserialized.data.batch_size);
    }
}