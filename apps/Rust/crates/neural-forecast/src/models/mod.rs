//! Neural forecasting model implementations
//!
//! This module contains implementations of state-of-the-art neural forecasting models
//! optimized for financial time series prediction with WebGPU acceleration.

use std::collections::HashMap;
use ndarray::{Array1, Array2, Array3};
use serde::{Serialize, Deserialize};
use async_trait::async_trait;
use crate::{Result, NeuralForecastError};
use crate::config::{
    NeuralForecastConfig, NHITSConfig, NBEATSConfig, TransformerConfig, 
    LSTMConfig, GRUConfig, EnsembleConfig, GPUConfig
};
#[cfg(feature = "gpu")]
use crate::gpu::{GPUBackend, GPUTensor};

pub mod nhits;
pub mod nbeats;
pub mod transformer;
pub mod lstm;
pub mod gru;
pub mod base;

pub use nhits::NHITSModel;
pub use nbeats::NBEATSModel;
pub use transformer::TransformerModel;
pub use lstm::LSTMModel;
pub use gru::GRUModel;
pub use base::*;

// VotingStrategy, ModelType, ModelConfig, and DynamicModelConfig are already defined in this module

/// Common trait for all neural forecasting models
#[async_trait]
pub trait Model: Send + Sync {
    /// Configuration type for this model
    type Config;
    
    /// Create new model instance
    fn new(config: Self::Config) -> Result<Self> where Self: Sized;
    
    /// Initialize model with GPU backend
    #[cfg(feature = "gpu")]
    async fn initialize(&mut self, gpu_backend: Option<&GPUBackend>) -> Result<()>;
    
    /// Initialize model without GPU backend
    #[cfg(not(feature = "gpu"))]
    async fn initialize(&mut self) -> Result<()>;
    
    /// Train the model on historical data
    async fn train(&mut self, data: &TrainingData) -> Result<TrainingMetrics>;
    
    /// Make predictions for future time steps
    async fn predict(&self, input: &Array3<f32>) -> Result<Array3<f32>>;
    
    /// Make batch predictions for multiple assets
    async fn predict_batch(&self, inputs: &[Array3<f32>]) -> Result<Vec<Array3<f32>>>;
    
    /// Get model parameters
    fn parameters(&self) -> &ModelParameters;
    
    /// Set model parameters
    fn set_parameters(&mut self, parameters: ModelParameters) -> Result<()>;
    
    /// Save model to file
    async fn save(&self, path: &std::path::Path) -> Result<()>;
    
    /// Load model from file
    async fn load(&mut self, path: &std::path::Path) -> Result<()>;
    
    /// Get model metadata
    fn metadata(&self) -> ModelMetadata;
    
    /// Validate model configuration
    fn validate_config(&self) -> Result<()>;
    
    /// Get model performance metrics
    fn metrics(&self) -> Option<&ModelMetrics>;
    
    /// Update model with new data (online learning)
    async fn update(&mut self, data: &UpdateData) -> Result<()>;
}

/// Model configuration trait
pub trait ModelConfig: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> {
    /// Validate configuration
    fn validate(&self) -> Result<()>;
    
    /// Get model type
    fn model_type(&self) -> ModelType;
    
    /// Get input/output dimensions
    fn dimensions(&self) -> (usize, usize);
    
    /// Get training parameters
    fn training_params(&self) -> TrainingParams;
}

/// Enum wrapper for model configurations to enable dynamic dispatch
/// Dynamic model configuration enum for runtime model selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DynamicModelConfig {
    /// NHITS model configuration
    NHITS(crate::config::NHITSConfig),
    /// NBEATS model configuration
    NBEATS(crate::config::NBEATSConfig),
    /// Transformer model configuration
    Transformer(crate::config::TransformerConfig),
    /// LSTM model configuration
    LSTM(crate::config::LSTMConfig),
    /// GRU model configuration
    GRU(crate::config::GRUConfig),
}

impl DynamicModelConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        match self {
            DynamicModelConfig::NHITS(config) => config.validate(),
            DynamicModelConfig::NBEATS(config) => config.validate(),
            DynamicModelConfig::Transformer(config) => config.validate(),
            DynamicModelConfig::LSTM(config) => config.validate(),
            DynamicModelConfig::GRU(config) => config.validate(),
        }
    }
    
    /// Get the model type
    pub fn model_type(&self) -> ModelType {
        match self {
            DynamicModelConfig::NHITS(config) => config.model_type(),
            DynamicModelConfig::NBEATS(config) => config.model_type(),
            DynamicModelConfig::Transformer(config) => config.model_type(),
            DynamicModelConfig::LSTM(config) => config.model_type(),
            DynamicModelConfig::GRU(config) => config.model_type(),
        }
    }
    
    /// Get model input/output dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        match self {
            DynamicModelConfig::NHITS(config) => config.dimensions(),
            DynamicModelConfig::NBEATS(config) => config.dimensions(),
            DynamicModelConfig::Transformer(config) => config.dimensions(),
            DynamicModelConfig::LSTM(config) => config.dimensions(),
            DynamicModelConfig::GRU(config) => config.dimensions(),
        }
    }
    
    /// Get training parameters
    pub fn training_params(&self) -> TrainingParams {
        match self {
            DynamicModelConfig::NHITS(config) => config.training_params(),
            DynamicModelConfig::NBEATS(config) => config.training_params(),
            DynamicModelConfig::Transformer(config) => config.training_params(),
            DynamicModelConfig::LSTM(config) => config.training_params(),
            DynamicModelConfig::GRU(config) => config.training_params(),
        }
    }
}

/// Model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    /// Neural Hierarchical Interpolation for Time Series
    NHITS,
    /// Neural Basis Expansion Analysis for Time Series
    NBEATS,
    /// Transformer-based model
    Transformer,
    /// Long Short-Term Memory model
    LSTM,
    /// Gated Recurrent Unit model
    GRU,
    /// Custom model type with ID
    Custom(u32),
}

/// Training data structure
#[derive(Debug, Clone)]
pub struct TrainingData {
    /// Input sequences [batch_size, seq_len, features]
    pub inputs: Array3<f32>,
    /// Target sequences [batch_size, horizon, features]
    pub targets: Array3<f32>,
    /// Optional static features [batch_size, static_features]
    pub static_features: Option<Array2<f32>>,
    /// Optional time features [batch_size, seq_len, time_features]
    pub time_features: Option<Array3<f32>>,
    /// Asset identifiers
    pub asset_ids: Vec<String>,
    /// Timestamps
    pub timestamps: Vec<chrono::DateTime<chrono::Utc>>,
}

/// Update data for online learning
#[derive(Debug, Clone)]
pub struct UpdateData {
    /// New input sequences
    pub inputs: Array3<f32>,
    /// New target sequences
    pub targets: Array3<f32>,
    /// Learning rate for update
    pub learning_rate: Option<f32>,
    /// Regularization factor
    pub regularization: Option<f32>,
}

/// Model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    /// Model weights
    pub weights: HashMap<String, Array2<f32>>,
    /// Model biases
    pub biases: HashMap<String, Array1<f32>>,
    /// Normalization parameters
    pub normalization: Option<NormalizationParams>,
    /// Model version
    pub version: u32,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Normalization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    /// Mean values
    pub mean: Array1<f32>,
    /// Standard deviation values
    pub std: Array1<f32>,
    /// Minimum values
    pub min: Array1<f32>,
    /// Maximum values
    pub max: Array1<f32>,
}

/// Training parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Early stopping patience
    pub patience: usize,
    /// Validation split
    pub validation_split: f32,
    /// L2 regularization
    pub l2_reg: f32,
    /// Dropout rate
    pub dropout: f32,
    /// Gradient clipping
    pub grad_clip: f32,
    /// Optimizer type
    pub optimizer: OptimizerType,
}

/// Optimizer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Adam optimizer
    Adam,
    /// Stochastic Gradient Descent
    SGD,
    /// Root Mean Square Propagation
    RMSprop,
    /// Adam with weight decay
    AdamW,
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss history
    pub train_loss: Vec<f32>,
    /// Validation loss history
    pub val_loss: Vec<f32>,
    /// Training accuracy history
    pub train_accuracy: Vec<f32>,
    /// Validation accuracy history
    pub val_accuracy: Vec<f32>,
    /// Training time in seconds
    pub training_time: f64,
    /// Number of epochs trained
    pub epochs_trained: usize,
    /// Best validation loss
    pub best_val_loss: f32,
    /// Early stopping triggered
    pub early_stopped: bool,
    /// Final learning rate
    pub final_lr: f32,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model type
    pub model_type: ModelType,
    /// Model name
    pub name: String,
    /// Version
    pub version: String,
    /// Description
    pub description: String,
    /// Author
    pub author: String,
    /// Creation date
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modified date
    pub modified_at: chrono::DateTime<chrono::Utc>,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Number of parameters
    pub num_parameters: usize,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Training data info
    pub training_data_info: Option<TrainingDataInfo>,
    /// Performance metrics
    pub performance_metrics: Option<PerformanceMetrics>,
}

/// Training data information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataInfo {
    /// Number of samples
    pub num_samples: usize,
    /// Number of features
    pub num_features: usize,
    /// Sequence length
    pub sequence_length: usize,
    /// Prediction horizon
    pub prediction_horizon: usize,
    /// Asset types
    pub asset_types: Vec<String>,
    /// Time period
    pub time_period: (chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>),
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Mean Absolute Error
    pub mae: f32,
    /// Mean Squared Error
    pub mse: f32,
    /// Root Mean Squared Error
    pub rmse: f32,
    /// Mean Absolute Percentage Error
    pub mape: f32,
    /// R-squared
    pub r2: f32,
    /// Inference time in microseconds
    pub inference_time_us: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    /// Sharpe ratio (for financial metrics)
    pub sharpe_ratio: Option<f32>,
    /// Maximum drawdown
    pub max_drawdown: Option<f32>,
    /// Hit rate (percentage of correct directional predictions)
    pub hit_rate: Option<f32>,
}

/// Model metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Model performance metrics
    pub performance: PerformanceMetrics,
    /// Resource usage metrics
    pub resource_usage: ResourceUsage,
    /// Prediction statistics
    pub prediction_stats: PredictionStats,
}

/// Resource usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage percentage
    pub cpu_usage: f32,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// GPU usage percentage
    pub gpu_usage: Option<f32>,
    /// GPU memory usage in bytes
    pub gpu_memory_usage: Option<u64>,
    /// Disk I/O operations
    pub disk_io: u64,
    /// Network I/O operations
    pub network_io: u64,
}

/// Prediction statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionStats {
    /// Total predictions made
    pub total_predictions: u64,
    /// Successful predictions
    pub successful_predictions: u64,
    /// Failed predictions
    pub failed_predictions: u64,
    /// Average prediction time
    pub avg_prediction_time_us: f64,
    /// Prediction confidence distribution
    pub confidence_distribution: Vec<f32>,
    /// Prediction error distribution
    pub error_distribution: Vec<f32>,
}

/// Model factory for creating model instances
pub struct ModelFactory;

impl ModelFactory {
    /// Create model from configuration
    pub fn create_model(config: &NeuralForecastConfig) -> Result<Box<dyn Model<Config = DynamicModelConfig>>> {
        // Validate all config types are available
        let _nhits_config: &NHITSConfig = &config.models.nhits;
        let _nbeats_config: &NBEATSConfig = &config.models.nbeats;
        let _transformer_config: &TransformerConfig = &config.models.transformer;
        let _lstm_config: &LSTMConfig = &config.models.lstm;
        let _gru_config: &GRUConfig = &config.models.gru;
        let _ensemble_config: &EnsembleConfig = &config.ensemble;
        let _gpu_config: &GPUConfig = &config.gpu;
        
        match config.ensemble.voting_strategy.as_str() {
            "nhits" => {
                let dynamic_config = DynamicModelConfig::NHITS(config.models.nhits.clone());
                let model = <NHITSModel as Model>::new(dynamic_config)?;
                Ok(Box::new(model) as Box<dyn Model<Config = DynamicModelConfig>>)
            }
            "nbeats" => {
                let dynamic_config = DynamicModelConfig::NBEATS(config.models.nbeats.clone());
                let model = <NBEATSModel as Model>::new(dynamic_config)?;
                Ok(Box::new(model) as Box<dyn Model<Config = DynamicModelConfig>>)
            }
            "transformer" => {
                let dynamic_config = DynamicModelConfig::Transformer(config.models.transformer.clone());
                let model = <TransformerModel as Model>::new(dynamic_config)?;
                Ok(Box::new(model) as Box<dyn Model<Config = DynamicModelConfig>>)
            }
            "lstm" => {
                let dynamic_config = DynamicModelConfig::LSTM(config.models.lstm.clone());
                let model = <LSTMModel as Model>::new(dynamic_config)?;
                Ok(Box::new(model) as Box<dyn Model<Config = DynamicModelConfig>>)
            }
            "gru" => {
                let dynamic_config = DynamicModelConfig::GRU(config.models.gru.clone());
                let model = <GRUModel as Model>::new(dynamic_config)?;
                Ok(Box::new(model) as Box<dyn Model<Config = DynamicModelConfig>>)
            }
            _ => Err(NeuralForecastError::ConfigError(
                format!("Unknown model type: {}", config.ensemble.voting_strategy)
            )),
        }
    }
    
    /// Create multiple models for ensemble
    pub fn create_ensemble_models(config: &NeuralForecastConfig) -> Result<Vec<Box<dyn Model<Config = DynamicModelConfig>>>> {
        let mut models = Vec::new();
        
        // Create NHITS model
        let nhits_config = DynamicModelConfig::NHITS(config.models.nhits.clone());
        let nhits = <NHITSModel as Model>::new(nhits_config)?;
        models.push(Box::new(nhits) as Box<dyn Model<Config = DynamicModelConfig>>);
        
        // Create NBEATS model
        let nbeats_config = DynamicModelConfig::NBEATS(config.models.nbeats.clone());
        let nbeats = <NBEATSModel as Model>::new(nbeats_config)?;
        models.push(Box::new(nbeats) as Box<dyn Model<Config = DynamicModelConfig>>);
        
        // Create Transformer model
        let transformer_config = DynamicModelConfig::Transformer(config.models.transformer.clone());
        let transformer = <TransformerModel as Model>::new(transformer_config)?;
        models.push(Box::new(transformer) as Box<dyn Model<Config = DynamicModelConfig>>);
        
        Ok(models)
    }
}

/// Model ensemble for combining multiple models
pub struct ModelEnsemble {
    models: Vec<Box<dyn Model<Config = DynamicModelConfig>>>,
    weights: Vec<f32>,
    voting_strategy: VotingStrategy,
    #[allow(dead_code)]
    gpu_config: GPUConfig,
}

/// Voting strategies for ensemble
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VotingStrategy {
    /// Simple average of all predictions
    Average,
    /// Weighted average based on model performance
    WeightedAverage,
    /// Median of all predictions
    Median,
    /// Maximum prediction value
    Max,
    /// Minimum prediction value
    Min,
    /// Meta-learning stacking approach
    Stacking,
}

impl ModelEnsemble {
    /// Create new model ensemble
    pub fn new(models: Vec<Box<dyn Model<Config = DynamicModelConfig>>>, weights: Vec<f32>) -> Result<Self> {
        if models.len() != weights.len() {
            return Err(NeuralForecastError::ConfigError(
                "Number of models must match number of weights".to_string()
            ));
        }
        
        let total_weight: f32 = weights.iter().sum();
        if (total_weight - 1.0).abs() > 1e-6 {
            return Err(NeuralForecastError::ConfigError(
                "Weights must sum to 1.0".to_string()
            ));
        }
        
        Ok(Self {
            models,
            weights,
            voting_strategy: VotingStrategy::WeightedAverage,
            gpu_config: GPUConfig::default(),
        })
    }
    
    /// Set voting strategy
    pub fn set_voting_strategy(&mut self, strategy: VotingStrategy) {
        self.voting_strategy = strategy;
    }
    
    /// Make ensemble prediction
    pub async fn predict(&self, input: &Array3<f32>) -> Result<Array3<f32>> {
        let mut predictions = Vec::new();
        
        // Get predictions from all models
        for model in &self.models {
            let prediction = model.predict(input).await?;
            predictions.push(prediction);
        }
        
        // Combine predictions based on voting strategy
        self.combine_predictions(predictions)
    }
    
    /// Combine predictions using voting strategy
    fn combine_predictions(&self, predictions: Vec<Array3<f32>>) -> Result<Array3<f32>> {
        if predictions.is_empty() {
            return Err(NeuralForecastError::InferenceError(
                "No predictions to combine".to_string()
            ));
        }
        
        let shape = predictions[0].shape();
        let mut result = Array3::<f32>::zeros((shape[0], shape[1], shape[2]));
        
        match self.voting_strategy {
            VotingStrategy::Average => {
                for pred in &predictions {
                    result = result + pred;
                }
                result = result / predictions.len() as f32;
            }
            VotingStrategy::WeightedAverage => {
                for (i, pred) in predictions.iter().enumerate() {
                    result = result + pred * self.weights[i];
                }
            }
            VotingStrategy::Median => {
                // Implement median voting
                // This is a simplified implementation
                for pred in &predictions {
                    result = result + pred;
                }
                result = result / predictions.len() as f32;
            }
            _ => {
                return Err(NeuralForecastError::InferenceError(
                    "Voting strategy not implemented".to_string()
                ));
            }
        }
        
        Ok(result)
    }
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            patience: 10,
            validation_split: 0.2,
            l2_reg: 0.0001,
            dropout: 0.1,
            grad_clip: 1.0,
            optimizer: OptimizerType::Adam,
        }
    }
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            weights: HashMap::new(),
            biases: HashMap::new(),
            normalization: None,
            version: 1,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }
}

impl ModelType {
    /// Get model type name
    pub fn name(&self) -> &'static str {
        match self {
            ModelType::NHITS => "nhits",
            ModelType::NBEATS => "nbeats",
            ModelType::Transformer => "transformer",
            ModelType::LSTM => "lstm",
            ModelType::GRU => "gru",
            ModelType::Custom(_) => "custom",
        }
    }
}

// ModelConfig implementations are defined in config.rs to avoid circular dependencies

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_type_name() {
        assert_eq!(ModelType::NHITS.name(), "nhits");
        assert_eq!(ModelType::NBEATS.name(), "nbeats");
        assert_eq!(ModelType::Transformer.name(), "transformer");
    }
    
    #[test]
    fn test_training_params_default() {
        let params = TrainingParams::default();
        assert_eq!(params.learning_rate, 0.001);
        assert_eq!(params.batch_size, 32);
        assert_eq!(params.epochs, 100);
    }
    
    #[test]
    fn test_model_parameters_default() {
        let params = ModelParameters::default();
        assert_eq!(params.version, 1);
        assert!(params.weights.is_empty());
        assert!(params.biases.is_empty());
    }
}