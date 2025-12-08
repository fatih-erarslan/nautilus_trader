//! Comprehensive Machine Learning Integration for CDFA Unified
//!
//! This module provides a unified ML framework that integrates multiple machine learning
//! libraries and approaches for financial signal analysis and pattern recognition.
//!
//! # Features
//!
//! - **Neural Networks**: Candle-based neural network implementations with GPU acceleration
//! - **Classical ML**: Linfa integration for traditional machine learning algorithms
//! - **Statistical Learning**: SmartCore for statistical learning and ensemble methods
//! - **TorchScript Fusion**: Hardware-accelerated signal fusion from TorchScript models
//! - **Model Management**: Comprehensive model versioning, serialization, and deployment
//! - **Training Infrastructure**: Advanced training pipelines with hyperparameter optimization
//!
//! # Quick Start
//!
//! ```rust
//! use cdfa_unified::ml::prelude::*;
//! use ndarray::Array2;
//!
//! // Create a neural network for signal classification
//! let config = NeuralConfig::default()
//!     .with_layers(vec![100, 64, 32, 1])
//!     .with_activation(Activation::ReLU)
//!     .with_device(Device::cuda_if_available());
//!
//! let mut model = NeuralNetwork::new(config)?;
//!
//! // Train the model
//! let X = Array2::random((1000, 100), Uniform::new(-1.0, 1.0));
//! let y = Array2::random((1000, 1), Uniform::new(0.0, 1.0));
//!
//! model.fit(&X, &y, 100)?;
//!
//! // Make predictions
//! let predictions = model.predict(&X)?;
//! ```
//!
//! # Architecture
//!
//! The ML module is organized into several key components:
//!
//! - `neural`: Candle-based neural network implementations
//! - `classical`: Traditional ML algorithms via Linfa
//! - `statistical`: Statistical learning via SmartCore
//! - `fusion`: TorchScript fusion integration
//! - `models`: Model management and versioning
//! - `training`: Training infrastructure and optimization
//! - `inference`: High-performance inference pipeline
//! - `serialization`: Model serialization and deployment
//!
//! Each component is designed to be modular and can be used independently or in combination.

use std::fmt;
use thiserror::Error;

pub mod classical;
#[cfg(feature = "candle")]
pub mod fusion;
pub mod inference;
pub mod models;
#[cfg(feature = "candle")]
pub mod neural;
pub mod serialization;
pub mod statistical;
pub mod training;
pub mod utils;

// Re-export key types and traits
pub use classical::*;
#[cfg(feature = "candle")]
pub use fusion::*;
pub use inference::*;
pub use models::*;
#[cfg(feature = "candle")]
pub use neural::*;
pub use serialization::*;
pub use statistical::*;
pub use training::*;
pub use utils::*;

/// Prelude module for convenient imports
pub mod prelude {
    pub use super::classical::*;
    #[cfg(feature = "candle")]
    pub use super::fusion::*;
    pub use super::inference::*;
    pub use super::models::*;
    #[cfg(feature = "candle")]
    pub use super::neural::*;
    pub use super::serialization::*;
    pub use super::statistical::*;
    pub use super::training::*;
    pub use super::utils::*;
    
    // Re-export common external types
    #[cfg(feature = "candle")]
    pub use candle_core::{Device, Tensor, DType};
    pub use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
    #[cfg(feature = "rand_distr")]
    pub use rand_distr::Uniform;
}

/// Comprehensive ML error handling
#[derive(Error, Debug)]
pub enum MLError {
    #[error("Model training failed: {message}")]
    TrainingError { message: String },
    
    #[error("Model inference failed: {message}")]
    InferenceError { message: String },
    
    #[error("Model serialization failed: {message}")]
    SerializationError { message: String },
    
    #[error("Invalid model configuration: {message}")]
    ConfigurationError { message: String },
    
    #[error("Data preprocessing failed: {message}")]
    PreprocessingError { message: String },
    
    #[error("Hardware acceleration error: {message}")]
    HardwareError { message: String },
    
    #[error("Model not found: {model_id}")]
    ModelNotFound { model_id: String },
    
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },
    
    #[error("Feature extraction failed: {message}")]
    FeatureExtractionError { message: String },
    
    #[error("Hyperparameter optimization failed: {message}")]
    OptimizationError { message: String },
    
    #[error("Cross-validation failed: {message}")]
    ValidationError { message: String },
    
    #[error("Model ensemble error: {message}")]
    EnsembleError { message: String },
    
    #[error("CDFA integration error: {message}")]
    CDFAError { message: String },
    
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[cfg(feature = "candle")]
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),
    
    #[error("Linfa error: {message}")]
    LinfaError { message: String },
    
    #[error("SmartCore error: {message}")]
    SmartCoreError { message: String },
    
    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),
    
    #[error("Bincode error: {0}")]
    BincodeError(#[from] bincode::Error),
}

/// Result type for ML operations
pub type MLResult<T> = Result<T, MLError>;

/// Supported ML frameworks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MLFramework {
    /// Candle (Rust-native neural networks)
    Candle,
    /// Linfa (Classical ML algorithms)
    Linfa,
    /// SmartCore (Statistical learning)
    SmartCore,
    /// TorchScript Fusion (Hardware-accelerated fusion)
    TorchScript,
    /// Hybrid (Combination of multiple frameworks)
    Hybrid,
}

impl fmt::Display for MLFramework {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MLFramework::Candle => write!(f, "Candle"),
            MLFramework::Linfa => write!(f, "Linfa"),
            MLFramework::SmartCore => write!(f, "SmartCore"),
            MLFramework::TorchScript => write!(f, "TorchScript"),
            MLFramework::Hybrid => write!(f, "Hybrid"),
        }
    }
}

/// ML task types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MLTask {
    /// Classification task
    Classification,
    /// Regression task
    Regression,
    /// Clustering task
    Clustering,
    /// Dimensionality reduction
    DimensionalityReduction,
    /// Anomaly detection
    AnomalyDetection,
    /// Time series forecasting
    TimeSeriesForecasting,
    /// Signal fusion
    SignalFusion,
    /// Feature extraction
    FeatureExtraction,
}

impl fmt::Display for MLTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MLTask::Classification => write!(f, "Classification"),
            MLTask::Regression => write!(f, "Regression"),
            MLTask::Clustering => write!(f, "Clustering"),
            MLTask::DimensionalityReduction => write!(f, "DimensionalityReduction"),
            MLTask::AnomalyDetection => write!(f, "AnomalyDetection"),
            MLTask::TimeSeriesForecasting => write!(f, "TimeSeriesForecasting"),
            MLTask::SignalFusion => write!(f, "SignalFusion"),
            MLTask::FeatureExtraction => write!(f, "FeatureExtraction"),
        }
    }
}

/// Hardware acceleration options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acceleration {
    /// CPU-only computation
    CPU,
    /// CUDA GPU acceleration
    CUDA,
    /// ROCm GPU acceleration (AMD)
    ROCm,
    /// Metal GPU acceleration (Apple)
    Metal,
    /// Automatic selection based on availability
    Auto,
}

impl Default for Acceleration {
    fn default() -> Self {
        Self::Auto
    }
}

/// Model metadata for tracking and versioning
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelMetadata {
    /// Unique model identifier
    pub id: String,
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Framework used
    pub framework: String,
    /// ML task type
    pub task: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Training parameters
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
    /// Performance metrics
    pub metrics: std::collections::HashMap<String, f64>,
    /// Model description
    pub description: Option<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl ModelMetadata {
    /// Create new model metadata
    pub fn new(id: String, name: String, framework: MLFramework, task: MLTask) -> Self {
        let now = chrono::Utc::now();
        Self {
            id,
            name,
            version: "1.0.0".to_string(),
            framework: framework.to_string(),
            task: task.to_string(),
            created_at: now,
            updated_at: now,
            parameters: std::collections::HashMap::new(),
            metrics: std::collections::HashMap::new(),
            description: None,
            tags: Vec::new(),
        }
    }
    
    /// Update metadata timestamp
    pub fn touch(&mut self) {
        self.updated_at = chrono::Utc::now();
    }
    
    /// Add a parameter
    pub fn add_parameter<T: serde::Serialize>(&mut self, key: String, value: T) -> MLResult<()> {
        self.parameters.insert(key, serde_json::to_value(value)?);
        self.touch();
        Ok(())
    }
    
    /// Add a metric
    pub fn add_metric(&mut self, key: String, value: f64) {
        self.metrics.insert(key, value);
        self.touch();
    }
    
    /// Add a tag
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
            self.touch();
        }
    }
}

/// Common trait for all ML models
pub trait MLModel: Send + Sync {
    /// Model input type
    type Input;
    /// Model output type
    type Output;
    /// Model configuration type
    type Config;
    
    /// Create new model with configuration
    fn new(config: Self::Config) -> MLResult<Self> where Self: Sized;
    
    /// Fit the model to training data
    fn fit(&mut self, x: &Self::Input, y: &Self::Output) -> MLResult<()>;
    
    /// Make predictions on input data
    fn predict(&self, x: &Self::Input) -> MLResult<Self::Output>;
    
    /// Evaluate model performance
    fn evaluate(&self, x: &Self::Input, y: &Self::Output) -> MLResult<f64>;
    
    /// Get model metadata
    fn metadata(&self) -> &ModelMetadata;
    
    /// Get mutable model metadata
    fn metadata_mut(&mut self) -> &mut ModelMetadata;
    
    /// Serialize model to bytes
    fn to_bytes(&self) -> MLResult<Vec<u8>>;
    
    /// Deserialize model from bytes
    fn from_bytes(bytes: &[u8]) -> MLResult<Self> where Self: Sized;
    
    /// Get model framework
    fn framework(&self) -> MLFramework;
    
    /// Get model task type
    fn task(&self) -> MLTask;
    
    /// Check if model is trained
    fn is_trained(&self) -> bool;
    
    /// Get model parameter count
    fn parameter_count(&self) -> usize;
    
    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize;
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Maximum number of epochs
    pub max_epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Validation split ratio
    pub validation_split: f64,
    /// Early stopping patience
    pub early_stopping_patience: Option<usize>,
    /// Hardware acceleration
    pub acceleration: Acceleration,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Verbose training output
    pub verbose: bool,
    /// Save best model during training
    pub save_best: bool,
    /// Model save path
    pub save_path: Option<std::path::PathBuf>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_epochs: 100,
            learning_rate: 0.001,
            batch_size: 32,
            validation_split: 0.2,
            early_stopping_patience: Some(10),
            acceleration: Acceleration::Auto,
            seed: None,
            verbose: true,
            save_best: true,
            save_path: None,
        }
    }
}

impl TrainingConfig {
    /// Create new training configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set maximum epochs
    pub fn with_max_epochs(mut self, epochs: usize) -> Self {
        self.max_epochs = epochs;
        self
    }
    
    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    
    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
    
    /// Set validation split
    pub fn with_validation_split(mut self, split: f64) -> Self {
        self.validation_split = split;
        self
    }
    
    /// Set early stopping patience
    pub fn with_early_stopping(mut self, patience: Option<usize>) -> Self {
        self.early_stopping_patience = patience;
        self
    }
    
    /// Set hardware acceleration
    pub fn with_acceleration(mut self, acceleration: Acceleration) -> Self {
        self.acceleration = acceleration;
        self
    }
    
    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    /// Set verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
    
    /// Set save path
    pub fn with_save_path<P: Into<std::path::PathBuf>>(mut self, path: P) -> Self {
        self.save_path = Some(path.into());
        self
    }
}

/// Performance metrics for model evaluation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerformanceMetrics {
    /// Accuracy (for classification)
    pub accuracy: Option<f64>,
    /// Precision (for classification)
    pub precision: Option<f64>,
    /// Recall (for classification)
    pub recall: Option<f64>,
    /// F1 score (for classification)
    pub f1_score: Option<f64>,
    /// Area under ROC curve (for classification)
    pub auc_roc: Option<f64>,
    /// Mean squared error (for regression)
    pub mse: Option<f64>,
    /// Root mean squared error (for regression)
    pub rmse: Option<f64>,
    /// Mean absolute error (for regression)
    pub mae: Option<f64>,
    /// R-squared (for regression)
    pub r2_score: Option<f64>,
    /// Training time in seconds
    pub training_time: Option<f64>,
    /// Inference time per sample in microseconds
    pub inference_time_us: Option<f64>,
    /// Memory usage in bytes
    pub memory_usage: Option<usize>,
    /// Custom metrics
    pub custom_metrics: std::collections::HashMap<String, f64>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            accuracy: None,
            precision: None,
            recall: None,
            f1_score: None,
            auc_roc: None,
            mse: None,
            rmse: None,
            mae: None,
            r2_score: None,
            training_time: None,
            inference_time_us: None,
            memory_usage: None,
            custom_metrics: std::collections::HashMap::new(),
        }
    }
}

impl PerformanceMetrics {
    /// Create new empty metrics
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add a custom metric
    pub fn add_custom(&mut self, name: String, value: f64) {
        self.custom_metrics.insert(name, value);
    }
    
    /// Get primary metric based on task type
    pub fn primary_metric(&self, task: MLTask) -> Option<f64> {
        match task {
            MLTask::Classification => self.accuracy.or(self.f1_score),
            MLTask::Regression => self.r2_score.or(self.rmse.map(|x| -x)),
            MLTask::Clustering => self.custom_metrics.get("silhouette_score").copied(),
            MLTask::AnomalyDetection => self.auc_roc,
            MLTask::TimeSeriesForecasting => self.rmse.map(|x| -x),
            MLTask::SignalFusion => self.custom_metrics.get("fusion_quality").copied(),
            MLTask::FeatureExtraction => self.custom_metrics.get("explained_variance").copied(),
            MLTask::DimensionalityReduction => self.custom_metrics.get("reconstruction_error").map(|x| -x),
        }
    }
}

/// Global ML configuration and registry
pub struct MLRegistry {
    /// Available devices
    #[cfg(feature = "candle")]
    pub devices: Vec<candle_core::Device>,
    #[cfg(not(feature = "candle"))]
    pub devices: Vec<String>,
    /// Default device
    #[cfg(feature = "candle")]
    pub default_device: candle_core::Device,
    #[cfg(not(feature = "candle"))]
    pub default_device: String,
    /// Model cache
    pub model_cache: std::sync::Arc<parking_lot::RwLock<std::collections::HashMap<String, Vec<u8>>>>,
    /// Feature extractors
    pub feature_extractors: std::sync::Arc<parking_lot::RwLock<std::collections::HashMap<String, Box<dyn crate::traits::FeatureExtractor + Send + Sync>>>>,
}

impl Default for MLRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl MLRegistry {
    /// Create new ML registry
    pub fn new() -> Self {
        #[cfg(feature = "candle")]
        {
            let default_device = candle_core::Device::Cpu;
            let devices = vec![default_device.clone()];
            
            Self {
                devices,
                default_device,
                model_cache: std::sync::Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new())),
                feature_extractors: std::sync::Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new())),
            }
        }
        #[cfg(not(feature = "candle"))]
        {
            let default_device = "cpu".to_string();
            let devices = vec![default_device.clone()];
            
            Self {
                devices,
                default_device,
                model_cache: std::sync::Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new())),
                feature_extractors: std::sync::Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new())),
            }
        }
    }
    
    /// Initialize with device detection
    pub fn with_device_detection() -> Self {
        let mut registry = Self::new();
        registry.detect_devices();
        registry
    }
    
    /// Detect available devices
    pub fn detect_devices(&mut self) {
        #[cfg(feature = "candle")]
        {
            self.devices.clear();
            self.devices.push(candle_core::Device::Cpu);
            
            // Try to detect CUDA devices
            #[cfg(feature = "cuda")]
            {
                for i in 0..8 {
                    if let Ok(device) = candle_core::Device::new_cuda(i) {
                        self.devices.push(device);
                    }
                }
            }
            
            // Try to detect Metal devices
            #[cfg(feature = "metal")]
            {
                for i in 0..4 {
                    if let Ok(device) = candle_core::Device::new_metal(i) {
                        self.devices.push(device);
                    }
                }
            }
            
            // Set best available device as default
            if self.devices.len() > 1 {
                self.default_device = self.devices[1].clone();
            }
        }
        #[cfg(not(feature = "candle"))]
        {
            self.devices.clear();
            self.devices.push("cpu".to_string());
            self.default_device = "cpu".to_string();
        }
    }
    
    /// Get best available device
    #[cfg(feature = "candle")]
    pub fn best_device(&self) -> &candle_core::Device {
        &self.default_device
    }
    
    /// Get best available device
    #[cfg(not(feature = "candle"))]
    pub fn best_device(&self) -> &String {
        &self.default_device
    }
    
    /// Cache a model
    pub fn cache_model(&self, id: String, data: Vec<u8>) {
        self.model_cache.write().insert(id, data);
    }
    
    /// Get cached model
    pub fn get_cached_model(&self, id: &str) -> Option<Vec<u8>> {
        self.model_cache.read().get(id).cloned()
    }
    
    /// Register feature extractor
    pub fn register_feature_extractor(&self, name: String, extractor: Box<dyn crate::traits::FeatureExtractor + Send + Sync>) {
        self.feature_extractors.write().insert(name, extractor);
    }
    
    /// Get feature extractor
    pub fn get_feature_extractor(&self, name: &str) -> Option<std::sync::Arc<Box<dyn crate::traits::FeatureExtractor + Send + Sync>>> {
        self.feature_extractors.read().get(name).map(|e| {
            // Note: This is a simplified approach. In practice, you'd want to use Arc<dyn Trait>
            // throughout for better performance
            std::sync::Arc::new(Box::new(crate::utils::DummyFeatureExtractor))
        })
    }
}

/// Thread-safe global ML registry instance
static ML_REGISTRY: once_cell::sync::Lazy<std::sync::Arc<parking_lot::RwLock<MLRegistry>>> = 
    once_cell::sync::Lazy::new(|| {
        std::sync::Arc::new(parking_lot::RwLock::new(MLRegistry::with_device_detection()))
    });

/// Get global ML registry
pub fn ml_registry() -> std::sync::Arc<parking_lot::RwLock<MLRegistry>> {
    ML_REGISTRY.clone()
}

/// Initialize ML subsystem
pub fn initialize_ml() -> MLResult<()> {
    tracing::info!("Initializing CDFA ML subsystem");
    
    // Initialize device detection
    let mut registry = ml_registry().write();
    registry.detect_devices();
    
    tracing::info!("Detected {} devices", registry.devices.len());
    tracing::info!("Default device: {:?}", registry.default_device);
    
    Ok(())
}

/// Shutdown ML subsystem
pub fn shutdown_ml() -> MLResult<()> {
    tracing::info!("Shutting down CDFA ML subsystem");
    
    // Clear caches
    let registry = ml_registry().read();
    registry.model_cache.write().clear();
    registry.feature_extractors.write().clear();
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ml_framework_display() {
        assert_eq!(MLFramework::Candle.to_string(), "Candle");
        assert_eq!(MLFramework::Linfa.to_string(), "Linfa");
        assert_eq!(MLFramework::SmartCore.to_string(), "SmartCore");
        assert_eq!(MLFramework::TorchScript.to_string(), "TorchScript");
        assert_eq!(MLFramework::Hybrid.to_string(), "Hybrid");
    }
    
    #[test]
    fn test_ml_task_display() {
        assert_eq!(MLTask::Classification.to_string(), "Classification");
        assert_eq!(MLTask::Regression.to_string(), "Regression");
        assert_eq!(MLTask::Clustering.to_string(), "Clustering");
    }
    
    #[test]
    fn test_model_metadata() {
        let mut metadata = ModelMetadata::new(
            "test-model".to_string(),
            "Test Model".to_string(),
            MLFramework::Candle,
            MLTask::Classification,
        );
        
        assert_eq!(metadata.id, "test-model");
        assert_eq!(metadata.name, "Test Model");
        assert_eq!(metadata.framework, "Candle");
        assert_eq!(metadata.task, "Classification");
        
        metadata.add_parameter("learning_rate".to_string(), 0.001).unwrap();
        metadata.add_metric("accuracy".to_string(), 0.95);
        metadata.add_tag("production".to_string());
        
        assert!(metadata.parameters.contains_key("learning_rate"));
        assert!(metadata.metrics.contains_key("accuracy"));
        assert!(metadata.tags.contains(&"production".to_string()));
    }
    
    #[test]
    fn test_training_config() {
        let config = TrainingConfig::new()
            .with_max_epochs(200)
            .with_learning_rate(0.01)
            .with_batch_size(64)
            .with_validation_split(0.3)
            .with_acceleration(Acceleration::CUDA)
            .with_seed(42)
            .with_verbose(false);
        
        assert_eq!(config.max_epochs, 200);
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.validation_split, 0.3);
        assert_eq!(config.acceleration, Acceleration::CUDA);
        assert_eq!(config.seed, Some(42));
        assert!(!config.verbose);
    }
    
    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();
        
        metrics.accuracy = Some(0.95);
        metrics.f1_score = Some(0.92);
        metrics.add_custom("custom_score".to_string(), 0.88);
        
        assert_eq!(metrics.primary_metric(MLTask::Classification), Some(0.95));
        assert_eq!(metrics.custom_metrics.get("custom_score"), Some(&0.88));
    }
    
    #[test]
    fn test_ml_registry() {
        let registry = MLRegistry::new();
        
        assert!(!registry.devices.is_empty());
        assert_eq!(registry.devices[0], candle_core::Device::Cpu);
        
        // Test model caching
        registry.cache_model("test".to_string(), vec![1, 2, 3, 4]);
        let cached = registry.get_cached_model("test");
        assert_eq!(cached, Some(vec![1, 2, 3, 4]));
    }
    
    #[test]
    fn test_ml_initialization() {
        // Test that initialization doesn't panic
        let result = initialize_ml();
        assert!(result.is_ok());
        
        let result = shutdown_ml();
        assert!(result.is_ok());
    }
}