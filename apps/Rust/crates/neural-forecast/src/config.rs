//! Configuration management for neural forecasting models
//!
//! This module provides centralized configuration management for all neural forecasting
//! components, including model architectures, training parameters, and system settings.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::collections::HashMap;

/// Main configuration structure for neural forecasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralForecastConfig {
    /// Model configurations
    pub models: ModelConfigs,
    
    /// Ensemble configuration
    pub ensemble: EnsembleConfig,
    
    /// GPU configuration
    pub gpu: GPUConfig,
    
    /// Preprocessing configuration
    pub preprocessing: PreprocessingConfig,
    
    /// Inference configuration
    pub inference: InferenceConfig,
    
    /// Storage configuration
    pub storage: StorageConfig,
    
    /// Batch processing configuration
    pub batch: BatchConfig,
    
    /// System configuration
    pub system: SystemConfig,
}

/// Model architecture configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfigs {
    /// NHITS model configuration
    pub nhits: NHITSConfig,
    
    /// NBEATS model configuration
    pub nbeats: NBEATSConfig,
    
    /// Transformer model configuration
    pub transformer: TransformerConfig,
    
    /// LSTM model configuration
    pub lstm: LSTMConfig,
    
    /// GRU model configuration
    pub gru: GRUConfig,
}

/// NHITS (Neural Hierarchical Interpolation for Time Series) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NHITSConfig {
    /// Input sequence length
    pub input_length: usize,
    
    /// Prediction horizon
    pub output_length: usize,
    
    /// Number of stacks
    pub n_stacks: usize,
    
    /// Number of blocks per stack
    pub n_blocks: Vec<usize>,
    
    /// Number of layers per block
    pub n_layers: Vec<usize>,
    
    /// Layer widths
    pub layer_widths: Vec<usize>,
    
    /// Pooling kernel sizes
    pub pooling_sizes: Vec<usize>,
    
    /// Interpolation modes
    pub interpolation_modes: Vec<String>,
    
    /// Activation function
    pub activation: String,
    
    /// Dropout rate
    pub dropout: f32,
    
    /// Learning rate
    pub learning_rate: f32,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Number of epochs
    pub epochs: usize,
    
    /// Early stopping patience
    pub patience: usize,
    
    /// Loss function
    pub loss_function: String,
    
    /// Optimizer
    pub optimizer: String,
    
    /// L2 regularization
    pub l2_regularization: f32,
    
    /// Gradient clipping
    pub gradient_clip: f32,
    
    /// Enable batch normalization
    pub batch_norm: bool,
    
    /// Enable residual connections
    pub residual_connections: bool,
    
    /// Enable attention mechanism
    pub attention: bool,
}

/// NBEATS (Neural Basis Expansion Analysis for Time Series) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NBEATSConfig {
    /// Input sequence length
    pub input_length: usize,
    
    /// Prediction horizon
    pub output_length: usize,
    
    /// Number of stacks
    pub n_stacks: usize,
    
    /// Number of blocks per stack
    pub n_blocks: usize,
    
    /// Number of layers per block
    pub n_layers: usize,
    
    /// Layer width
    pub layer_width: usize,
    
    /// Expansion coefficient dimension
    pub expansion_coefficient_dim: usize,
    
    /// Activation function
    pub activation: String,
    
    /// Dropout rate
    pub dropout: f32,
    
    /// Learning rate
    pub learning_rate: f32,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Number of epochs
    pub epochs: usize,
    
    /// Early stopping patience
    pub patience: usize,
    
    /// Loss function
    pub loss_function: String,
    
    /// Optimizer
    pub optimizer: String,
    
    /// L2 regularization
    pub l2_regularization: f32,
    
    /// Gradient clipping
    pub gradient_clip: f32,
    
    /// Enable batch normalization
    pub batch_norm: bool,
    
    /// Enable residual connections
    pub residual_connections: bool,
    
    /// Stack types ("trend", "seasonality", "generic")
    pub stack_types: Vec<String>,
    
    /// Enable share weights in stack
    pub share_weights_in_stack: bool,
    
    /// Enable harmonics for seasonality
    pub harmonics: bool,
}

/// Transformer model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Input sequence length
    pub input_length: usize,
    
    /// Prediction horizon
    pub output_length: usize,
    
    /// Model dimension
    pub d_model: usize,
    
    /// Number of heads in multi-head attention
    pub n_heads: usize,
    
    /// Number of encoder layers
    pub n_encoder_layers: usize,
    
    /// Number of decoder layers
    pub n_decoder_layers: usize,
    
    /// Feedforward dimension
    pub d_ff: usize,
    
    /// Dropout rate
    pub dropout: f32,
    
    /// Activation function
    pub activation: String,
    
    /// Learning rate
    pub learning_rate: f32,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Number of epochs
    pub epochs: usize,
    
    /// Early stopping patience
    pub patience: usize,
    
    /// Loss function
    pub loss_function: String,
    
    /// Optimizer
    pub optimizer: String,
    
    /// L2 regularization
    pub l2_regularization: f32,
    
    /// Gradient clipping
    pub gradient_clip: f32,
    
    /// Enable positional encoding
    pub positional_encoding: bool,
    
    /// Enable layer normalization
    pub layer_norm: bool,
    
    /// Enable pre-normalization
    pub pre_norm: bool,
    
    /// Enable causal masking
    pub causal_mask: bool,
    
    /// Enable cross-attention
    pub cross_attention: bool,
    
    /// Maximum position for positional encoding
    pub max_position: usize,
}

/// LSTM model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTMConfig {
    /// Input sequence length
    pub input_length: usize,
    
    /// Prediction horizon
    pub output_length: usize,
    
    /// Hidden size
    pub hidden_size: usize,
    
    /// Number of layers
    pub num_layers: usize,
    
    /// Dropout rate
    pub dropout: f32,
    
    /// Bidirectional
    pub bidirectional: bool,
    
    /// Learning rate
    pub learning_rate: f32,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Number of epochs
    pub epochs: usize,
    
    /// Early stopping patience
    pub patience: usize,
    
    /// Loss function
    pub loss_function: String,
    
    /// Optimizer
    pub optimizer: String,
    
    /// L2 regularization
    pub l2_regularization: f32,
    
    /// Gradient clipping
    pub gradient_clip: f32,
    
    /// Enable batch normalization
    pub batch_norm: bool,
}

/// GRU model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GRUConfig {
    /// Input sequence length
    pub input_length: usize,
    
    /// Prediction horizon
    pub output_length: usize,
    
    /// Hidden size
    pub hidden_size: usize,
    
    /// Number of layers
    pub num_layers: usize,
    
    /// Dropout rate
    pub dropout: f32,
    
    /// Bidirectional
    pub bidirectional: bool,
    
    /// Learning rate
    pub learning_rate: f32,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Number of epochs
    pub epochs: usize,
    
    /// Early stopping patience
    pub patience: usize,
    
    /// Loss function
    pub loss_function: String,
    
    /// Optimizer
    pub optimizer: String,
    
    /// L2 regularization
    pub l2_regularization: f32,
    
    /// Gradient clipping
    pub gradient_clip: f32,
    
    /// Enable batch normalization
    pub batch_norm: bool,
}

/// Ensemble configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Voting strategy
    pub voting_strategy: String,
    
    /// Model weights
    pub model_weights: HashMap<String, f32>,
    
    /// Enable dynamic weighting
    pub dynamic_weighting: bool,
    
    /// Confidence threshold
    pub confidence_threshold: f32,
    
    /// Enable uncertainty quantification
    pub uncertainty_quantification: bool,
    
    /// Temperature scaling factor
    pub temperature_scaling: f32,
    
    /// Enable calibration
    pub calibration: bool,
    
    /// Calibration method
    pub calibration_method: String,
}

/// GPU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUConfig {
    /// Enable GPU acceleration
    pub enabled: bool,
    
    /// Preferred GPU backend
    pub backend: String,
    
    /// Device index
    pub device_index: Option<u32>,
    
    /// Memory limit (MB)
    pub memory_limit: Option<u32>,
    
    /// Enable memory pooling
    pub memory_pooling: bool,
    
    /// Enable mixed precision
    pub mixed_precision: bool,
    
    /// Enable tensor cores
    pub tensor_cores: bool,
    
    /// Batch size for GPU
    pub gpu_batch_size: usize,
    
    /// Number of concurrent streams
    pub num_streams: usize,
    
    /// Enable asynchronous execution
    pub async_execution: bool,
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Normalization method
    pub normalization: String,
    
    /// Feature engineering methods
    pub feature_engineering: Vec<String>,
    
    /// Enable outlier detection
    pub outlier_detection: bool,
    
    /// Outlier detection method
    pub outlier_method: String,
    
    /// Enable missing value imputation
    pub missing_value_imputation: bool,
    
    /// Imputation method
    pub imputation_method: String,
    
    /// Enable trend removal
    pub trend_removal: bool,
    
    /// Enable seasonality removal
    pub seasonality_removal: bool,
    
    /// Enable differencing
    pub differencing: bool,
    
    /// Differencing order
    pub differencing_order: usize,
    
    /// Enable log transformation
    pub log_transform: bool,
    
    /// Enable box-cox transformation
    pub box_cox: bool,
    
    /// Window size for rolling statistics
    pub rolling_window: usize,
    
    /// Enable technical indicators
    pub technical_indicators: bool,
    
    /// Technical indicator list
    pub indicator_list: Vec<String>,
}

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Target latency (microseconds)
    pub target_latency_us: u64,
    
    /// Number of inference threads
    pub num_threads: usize,
    
    /// Enable real-time inference
    pub real_time: bool,
    
    /// Enable streaming inference
    pub streaming: bool,
    
    /// Buffer size for streaming
    pub stream_buffer_size: usize,
    
    /// Enable model caching
    pub model_caching: bool,
    
    /// Cache size (MB)
    pub cache_size: usize,
    
    /// Enable prediction caching
    pub prediction_caching: bool,
    
    /// Prediction cache TTL (seconds)
    pub prediction_cache_ttl: u64,
    
    /// Enable warm-up
    pub warm_up: bool,
    
    /// Number of warm-up iterations
    pub warm_up_iterations: usize,
    
    /// Enable profiling
    pub profiling: bool,
    
    /// Profile output path
    pub profile_output: Option<PathBuf>,
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Model storage path
    pub model_path: PathBuf,
    
    /// Enable memory mapping
    pub memory_mapping: bool,
    
    /// Enable compression
    pub compression: bool,
    
    /// Compression method
    pub compression_method: String,
    
    /// Enable encryption
    pub encryption: bool,
    
    /// Encryption key
    pub encryption_key: Option<String>,
    
    /// Enable checksums
    pub checksums: bool,
    
    /// Enable versioning
    pub versioning: bool,
    
    /// Maximum versions to keep
    pub max_versions: usize,
    
    /// Enable automatic cleanup
    pub auto_cleanup: bool,
    
    /// Cleanup threshold (days)
    pub cleanup_threshold: u64,
}

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    
    /// Number of assets to process simultaneously
    pub concurrent_assets: usize,
    
    /// Enable parallel processing
    pub parallel_processing: bool,
    
    /// Number of worker threads
    pub num_workers: usize,
    
    /// Enable load balancing
    pub load_balancing: bool,
    
    /// Load balancing strategy
    pub load_balancing_strategy: String,
    
    /// Enable dynamic batching
    pub dynamic_batching: bool,
    
    /// Batch timeout (milliseconds)
    pub batch_timeout: u64,
    
    /// Enable result aggregation
    pub result_aggregation: bool,
    
    /// Aggregation method
    pub aggregation_method: String,
}

/// System configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// Enable logging
    pub logging: bool,
    
    /// Log level
    pub log_level: String,
    
    /// Log output path
    pub log_path: Option<PathBuf>,
    
    /// Enable metrics collection
    pub metrics: bool,
    
    /// Metrics collection interval (seconds)
    pub metrics_interval: u64,
    
    /// Enable monitoring
    pub monitoring: bool,
    
    /// Monitoring port
    pub monitoring_port: u16,
    
    /// Enable health checks
    pub health_checks: bool,
    
    /// Health check interval (seconds)
    pub health_check_interval: u64,
    
    /// Enable graceful shutdown
    pub graceful_shutdown: bool,
    
    /// Shutdown timeout (seconds)
    pub shutdown_timeout: u64,
}

impl Default for NeuralForecastConfig {
    fn default() -> Self {
        Self {
            models: ModelConfigs::default(),
            ensemble: EnsembleConfig::default(),
            gpu: GPUConfig::default(),
            preprocessing: PreprocessingConfig::default(),
            inference: InferenceConfig::default(),
            storage: StorageConfig::default(),
            batch: BatchConfig::default(),
            system: SystemConfig::default(),
        }
    }
}

impl Default for ModelConfigs {
    fn default() -> Self {
        Self {
            nhits: NHITSConfig::default(),
            nbeats: NBEATSConfig::default(),
            transformer: TransformerConfig::default(),
            lstm: LSTMConfig::default(),
            gru: GRUConfig::default(),
        }
    }
}

impl Default for NHITSConfig {
    fn default() -> Self {
        Self {
            input_length: 168,
            output_length: 24,
            n_stacks: 3,
            n_blocks: vec![1, 1, 1],
            n_layers: vec![2, 2, 2],
            layer_widths: vec![512, 512, 512],
            pooling_sizes: vec![2, 2, 2],
            interpolation_modes: vec!["linear".to_string(); 3],
            activation: "relu".to_string(),
            dropout: 0.1,
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            patience: 10,
            loss_function: "mse".to_string(),
            optimizer: "adam".to_string(),
            l2_regularization: 0.0001,
            gradient_clip: 1.0,
            batch_norm: true,
            residual_connections: true,
            attention: false,
        }
    }
}

impl Default for NBEATSConfig {
    fn default() -> Self {
        Self {
            input_length: 168,
            output_length: 24,
            n_stacks: 3,
            n_blocks: 3,
            n_layers: 4,
            layer_width: 512,
            expansion_coefficient_dim: 5,
            activation: "relu".to_string(),
            dropout: 0.1,
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            patience: 10,
            loss_function: "mse".to_string(),
            optimizer: "adam".to_string(),
            l2_regularization: 0.0001,
            gradient_clip: 1.0,
            batch_norm: true,
            residual_connections: true,
            stack_types: vec!["trend".to_string(), "seasonality".to_string(), "generic".to_string()],
            share_weights_in_stack: false,
            harmonics: true,
        }
    }
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            input_length: 168,
            output_length: 24,
            d_model: 512,
            n_heads: 8,
            n_encoder_layers: 6,
            n_decoder_layers: 6,
            d_ff: 2048,
            dropout: 0.1,
            activation: "relu".to_string(),
            learning_rate: 0.0001,
            batch_size: 32,
            epochs: 100,
            patience: 10,
            loss_function: "mse".to_string(),
            optimizer: "adam".to_string(),
            l2_regularization: 0.0001,
            gradient_clip: 1.0,
            positional_encoding: true,
            layer_norm: true,
            pre_norm: true,
            causal_mask: true,
            cross_attention: true,
            max_position: 1000,
        }
    }
}

impl Default for LSTMConfig {
    fn default() -> Self {
        Self {
            input_length: 168,
            output_length: 24,
            hidden_size: 128,
            num_layers: 2,
            dropout: 0.1,
            bidirectional: false,
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            patience: 10,
            loss_function: "mse".to_string(),
            optimizer: "adam".to_string(),
            l2_regularization: 0.0001,
            gradient_clip: 1.0,
            batch_norm: true,
        }
    }
}

impl Default for GRUConfig {
    fn default() -> Self {
        Self {
            input_length: 168,
            output_length: 24,
            hidden_size: 128,
            num_layers: 2,
            dropout: 0.1,
            bidirectional: false,
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            patience: 10,
            loss_function: "mse".to_string(),
            optimizer: "adam".to_string(),
            l2_regularization: 0.0001,
            gradient_clip: 1.0,
            batch_norm: true,
        }
    }
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        let mut model_weights = HashMap::new();
        model_weights.insert("nhits".to_string(), 0.4);
        model_weights.insert("nbeats".to_string(), 0.3);
        model_weights.insert("transformer".to_string(), 0.3);
        
        Self {
            voting_strategy: "weighted_average".to_string(),
            model_weights,
            dynamic_weighting: true,
            confidence_threshold: 0.8,
            uncertainty_quantification: true,
            temperature_scaling: 1.0,
            calibration: true,
            calibration_method: "platt".to_string(),
        }
    }
}

impl Default for GPUConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            backend: "webgpu".to_string(),
            device_index: None,
            memory_limit: None,
            memory_pooling: true,
            mixed_precision: true,
            tensor_cores: true,
            gpu_batch_size: 128,
            num_streams: 2,
            async_execution: true,
        }
    }
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            normalization: "zscore".to_string(),
            feature_engineering: vec!["rolling_mean".to_string(), "rolling_std".to_string()],
            outlier_detection: true,
            outlier_method: "iqr".to_string(),
            missing_value_imputation: true,
            imputation_method: "linear".to_string(),
            trend_removal: false,
            seasonality_removal: false,
            differencing: false,
            differencing_order: 1,
            log_transform: false,
            box_cox: false,
            rolling_window: 24,
            technical_indicators: true,
            indicator_list: vec!["sma".to_string(), "ema".to_string(), "rsi".to_string()],
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            target_latency_us: 100,
            num_threads: num_cpus::get(),
            real_time: true,
            streaming: true,
            stream_buffer_size: 1024,
            model_caching: true,
            cache_size: 1024,
            prediction_caching: true,
            prediction_cache_ttl: 300,
            warm_up: true,
            warm_up_iterations: 100,
            profiling: false,
            profile_output: None,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models"),
            memory_mapping: true,
            compression: true,
            compression_method: "zstd".to_string(),
            encryption: false,
            encryption_key: None,
            checksums: true,
            versioning: true,
            max_versions: 10,
            auto_cleanup: true,
            cleanup_threshold: 30,
        }
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 128,
            concurrent_assets: 128,
            parallel_processing: true,
            num_workers: num_cpus::get(),
            load_balancing: true,
            load_balancing_strategy: "round_robin".to_string(),
            dynamic_batching: true,
            batch_timeout: 100,
            result_aggregation: true,
            aggregation_method: "weighted_average".to_string(),
        }
    }
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            logging: true,
            log_level: "info".to_string(),
            log_path: None,
            metrics: true,
            metrics_interval: 60,
            monitoring: true,
            monitoring_port: 8080,
            health_checks: true,
            health_check_interval: 30,
            graceful_shutdown: true,
            shutdown_timeout: 30,
        }
    }
}

// ModelConfig implementations for each config type
use crate::models::{ModelConfig, ModelType, TrainingParams, OptimizerType};

impl ModelConfig for NHITSConfig {
    fn validate(&self) -> crate::Result<()> {
        if self.input_length == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Input length must be greater than 0".to_string()
            ));
        }
        if self.output_length == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Output length must be greater than 0".to_string()
            ));
        }
        if self.n_stacks == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Number of stacks must be greater than 0".to_string()
            ));
        }
        Ok(())
    }

    fn model_type(&self) -> ModelType {
        ModelType::NHITS
    }

    fn dimensions(&self) -> (usize, usize) {
        (self.input_length, self.output_length)
    }

    fn training_params(&self) -> TrainingParams {
        TrainingParams {
            learning_rate: self.learning_rate,
            batch_size: self.batch_size,
            epochs: self.epochs,
            patience: self.patience,
            validation_split: 0.2,
            l2_reg: self.l2_regularization,
            dropout: self.dropout,
            grad_clip: self.gradient_clip,
            optimizer: OptimizerType::Adam,
        }
    }
}

impl ModelConfig for NBEATSConfig {
    fn validate(&self) -> crate::Result<()> {
        if self.input_length == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Input length must be greater than 0".to_string()
            ));
        }
        if self.output_length == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Output length must be greater than 0".to_string()
            ));
        }
        if self.n_stacks == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Number of stacks must be greater than 0".to_string()
            ));
        }
        Ok(())
    }

    fn model_type(&self) -> ModelType {
        ModelType::NBEATS
    }

    fn dimensions(&self) -> (usize, usize) {
        (self.input_length, self.output_length)
    }

    fn training_params(&self) -> TrainingParams {
        TrainingParams {
            learning_rate: self.learning_rate,
            batch_size: self.batch_size,
            epochs: self.epochs,
            patience: self.patience,
            validation_split: 0.2,
            l2_reg: self.l2_regularization,
            dropout: self.dropout,
            grad_clip: self.gradient_clip,
            optimizer: OptimizerType::Adam,
        }
    }
}

impl ModelConfig for TransformerConfig {
    fn validate(&self) -> crate::Result<()> {
        if self.input_length == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Input length must be greater than 0".to_string()
            ));
        }
        if self.output_length == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Output length must be greater than 0".to_string()
            ));
        }
        if self.d_model == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Model dimension must be greater than 0".to_string()
            ));
        }
        if self.n_heads == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Number of heads must be greater than 0".to_string()
            ));
        }
        Ok(())
    }

    fn model_type(&self) -> ModelType {
        ModelType::Transformer
    }

    fn dimensions(&self) -> (usize, usize) {
        (self.input_length, self.output_length)
    }

    fn training_params(&self) -> TrainingParams {
        TrainingParams {
            learning_rate: self.learning_rate,
            batch_size: self.batch_size,
            epochs: self.epochs,
            patience: self.patience,
            validation_split: 0.2,
            l2_reg: self.l2_regularization,
            dropout: self.dropout,
            grad_clip: self.gradient_clip,
            optimizer: OptimizerType::Adam,
        }
    }
}

impl ModelConfig for LSTMConfig {
    fn validate(&self) -> crate::Result<()> {
        if self.input_length == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Input length must be greater than 0".to_string()
            ));
        }
        if self.output_length == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Output length must be greater than 0".to_string()
            ));
        }
        if self.hidden_size == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Hidden size must be greater than 0".to_string()
            ));
        }
        if self.num_layers == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Number of layers must be greater than 0".to_string()
            ));
        }
        Ok(())
    }

    fn model_type(&self) -> ModelType {
        ModelType::LSTM
    }

    fn dimensions(&self) -> (usize, usize) {
        (self.input_length, self.output_length)
    }

    fn training_params(&self) -> TrainingParams {
        TrainingParams {
            learning_rate: self.learning_rate,
            batch_size: self.batch_size,
            epochs: self.epochs,
            patience: self.patience,
            validation_split: 0.2,
            l2_reg: self.l2_regularization,
            dropout: self.dropout,
            grad_clip: self.gradient_clip,
            optimizer: OptimizerType::Adam,
        }
    }
}

impl ModelConfig for GRUConfig {
    fn validate(&self) -> crate::Result<()> {
        if self.input_length == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Input length must be greater than 0".to_string()
            ));
        }
        if self.output_length == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Output length must be greater than 0".to_string()
            ));
        }
        if self.hidden_size == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Hidden size must be greater than 0".to_string()
            ));
        }
        if self.num_layers == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Number of layers must be greater than 0".to_string()
            ));
        }
        Ok(())
    }

    fn model_type(&self) -> ModelType {
        ModelType::GRU
    }

    fn dimensions(&self) -> (usize, usize) {
        (self.input_length, self.output_length)
    }

    fn training_params(&self) -> TrainingParams {
        TrainingParams {
            learning_rate: self.learning_rate,
            batch_size: self.batch_size,
            epochs: self.epochs,
            patience: self.patience,
            validation_split: 0.2,
            l2_reg: self.l2_regularization,
            dropout: self.dropout,
            grad_clip: self.gradient_clip,
            optimizer: OptimizerType::Adam,
        }
    }
}

impl NeuralForecastConfig {
    /// Load configuration from file
    pub fn load_from_file(path: &std::path::Path) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::NeuralForecastError::IoError(e.to_string()))?;
        
        let config = match path.extension().and_then(|s| s.to_str()) {
            Some("json") => serde_json::from_str(&content)?,
            Some("toml") => toml::from_str(&content)
                .map_err(|e| crate::NeuralForecastError::ConfigError(e.to_string()))?,
            _ => return Err(crate::NeuralForecastError::ConfigError(
                "Unsupported config file format".to_string()
            )),
        };
        
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_to_file(&self, path: &std::path::Path) -> crate::Result<()> {
        let content = match path.extension().and_then(|s| s.to_str()) {
            Some("json") => serde_json::to_string_pretty(self)?,
            Some("toml") => toml::to_string_pretty(self)
                .map_err(|e| crate::NeuralForecastError::ConfigError(e.to_string()))?,
            _ => return Err(crate::NeuralForecastError::ConfigError(
                "Unsupported config file format".to_string()
            )),
        };
        
        std::fs::write(path, content)
            .map_err(|e| crate::NeuralForecastError::IoError(e.to_string()))?;
        
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate(&self) -> crate::Result<()> {
        // Validate model configurations
        if self.models.nhits.input_length == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "NHITS input length must be greater than 0".to_string()
            ));
        }
        
        if self.models.nbeats.output_length == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "NBEATS output length must be greater than 0".to_string()
            ));
        }
        
        // Validate ensemble configuration
        let total_weight: f32 = self.ensemble.model_weights.values().sum();
        if (total_weight - 1.0).abs() > 1e-6 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Ensemble model weights must sum to 1.0".to_string()
            ));
        }
        
        // Validate GPU configuration
        if self.gpu.enabled && self.gpu.gpu_batch_size == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "GPU batch size must be greater than 0".to_string()
            ));
        }
        
        // Validate inference configuration
        if self.inference.target_latency_us == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Target latency must be greater than 0".to_string()
            ));
        }
        
        // Validate batch configuration
        if self.batch.max_batch_size == 0 {
            return Err(crate::NeuralForecastError::ConfigError(
                "Max batch size must be greater than 0".to_string()
            ));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_default_config() {
        let config = NeuralForecastConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_config_serialization() {
        let config = NeuralForecastConfig::default();
        
        // Test JSON serialization
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: NeuralForecastConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.models.nhits.input_length, deserialized.models.nhits.input_length);
        
        // Test TOML serialization
        let toml = toml::to_string(&config).unwrap();
        let deserialized: NeuralForecastConfig = toml::from_str(&toml).unwrap();
        assert_eq!(config.models.nbeats.output_length, deserialized.models.nbeats.output_length);
    }
    
    #[test]
    fn test_config_file_io() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.json");
        
        let config = NeuralForecastConfig::default();
        config.save_to_file(&config_path).unwrap();
        
        let loaded_config = NeuralForecastConfig::load_from_file(&config_path).unwrap();
        assert_eq!(config.models.transformer.d_model, loaded_config.models.transformer.d_model);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = NeuralForecastConfig::default();
        
        // Test valid configuration
        assert!(config.validate().is_ok());
        
        // Test invalid configuration
        config.models.nhits.input_length = 0;
        assert!(config.validate().is_err());
    }
}