//! # Neuro-Divergent: 27+ Neural Forecasting Models
//!
//! A comprehensive time series forecasting library with 27+ state-of-the-art neural models.
//!
//! ## Features
//!
//! - **27+ Neural Models**: NHITS, LSTM, GRU, Transformers, N-BEATS, DeepAR, and more
//! - **High Performance**: Parallel processing with Rayon, optimized matrix operations
//! - **GPU Acceleration**: Optional CUDA, Metal, and Accelerate support
//! - **Production Ready**: Comprehensive error handling, serialization, and testing
//! - **Easy Integration**: Simple API with sensible defaults
//!
//! ## Model Categories
//!
//! ### Basic Models
//! - **MLP**: Multi-Layer Perceptron
//! - **DLinear**: Decomposition Linear
//! - **NLinear**: Normalization Linear
//! - **MLPMultivariate**: Multi-output MLP
//!
//! ### Recurrent Models
//! - **RNN**: Recurrent Neural Network
//! - **LSTM**: Long Short-Term Memory
//! - **GRU**: Gated Recurrent Unit
//!
//! ### Advanced Models
//! - **NBEATS**: Neural Basis Expansion Analysis
//! - **NBEATSx**: Extended N-BEATS
//! - **NHITS**: Neural Hierarchical Interpolation for Time Series
//! - **TiDE**: Time-series Dense Encoder
//!
//! ### Transformer Models
//! - **TFT**: Temporal Fusion Transformer
//! - **Informer**: Informer Transformer
//! - **AutoFormer**: Auto-Correlation Transformer
//! - **FedFormer**: Frequency Enhanced Decomposed Transformer
//! - **PatchTST**: Patch Time Series Transformer
//! - **ITransformer**: Inverted Transformer
//!
//! ### Specialized Models
//! - **DeepAR**: Deep AutoRegressive
//! - **DeepNPTS**: Deep Non-Parametric Time Series
//! - **TCN**: Temporal Convolutional Network
//! - **BiTCN**: Bidirectional TCN
//! - **TimesNet**: TimesNet architecture
//! - **StemGNN**: Spectral Temporal Graph Neural Network
//! - **TSMixer**: Time Series Mixer
//! - **TimeLLM**: Large Language Model for Time Series
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use neuro_divergent::{
//!     NeuralModel, ModelConfig, TimeSeriesDataFrame,
//!     models::nhits::NHITSModel,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create time series data
//! let data = TimeSeriesDataFrame::from_values(
//!     vec![1.0, 2.0, 3.0, 4.0, 5.0],
//!     None,
//! )?;
//!
//! // Configure model
//! let config = ModelConfig::default()
//!     .with_input_size(24)
//!     .with_horizon(12);
//!
//! // Create and train model
//! let mut model = NHITSModel::new(config)?;
//! model.fit(&data)?;
//!
//! // Make predictions
//! let predictions = model.predict(12)?;
//! # Ok(())
//! # }
//! ```

pub mod error;
pub mod config;
pub mod data;
pub mod models;
pub mod training;
pub mod inference;
pub mod registry;
pub mod optimizations;

// NAPI bindings (feature-gated)
#[cfg(feature = "napi-bindings")]
pub mod napi_bindings;

// Re-export main types
pub use error::{NeuroDivergentError, Result};
pub use config::{ModelConfig, TrainingConfig};
pub use data::{TimeSeriesDataFrame, DataPreprocessor};
pub use registry::{ModelRegistry, ModelFactory};

/// Core trait that all neural models must implement
pub trait NeuralModel: Send + Sync {
    /// Train the model on the given time series data
    fn fit(&mut self, data: &TimeSeriesDataFrame) -> Result<()>;

    /// Make predictions for the specified horizon
    fn predict(&self, horizon: usize) -> Result<Vec<f64>>;

    /// Make predictions with prediction intervals
    fn predict_intervals(
        &self,
        horizon: usize,
        levels: &[f64],
    ) -> Result<inference::PredictionIntervals>;

    /// Get the model name
    fn name(&self) -> &str;

    /// Get model configuration
    fn config(&self) -> &ModelConfig;

    /// Save model to disk
    fn save(&self, path: &std::path::Path) -> Result<()>;

    /// Load model from disk
    fn load(path: &std::path::Path) -> Result<Self>
    where
        Self: Sized;
}

/// Version information for model tracking
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelVersion {
    pub version: String,
    pub model_id: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub model_name: String,
    pub config: serde_json::Value,
    pub metrics: Option<training::TrainingMetrics>,
}

impl ModelVersion {
    pub fn new(model_name: String, config: serde_json::Value) -> Self {
        Self {
            version: "0.1.0".to_string(),
            model_id: uuid::Uuid::new_v4().to_string(),
            created_at: chrono::Utc::now(),
            model_name,
            config,
            metrics: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_creation() {
        let config = serde_json::json!({"test": "value"});
        let version = ModelVersion::new("TestModel".to_string(), config);

        assert_eq!(version.version, "0.1.0");
        assert_eq!(version.model_name, "TestModel");
        assert!(!version.model_id.is_empty());
    }
}
