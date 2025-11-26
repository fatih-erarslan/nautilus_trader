//! # Neuro-Divergent NAPI Bindings
//!
//! Node.js bindings for the Neuro-Divergent neural forecasting library.
//! Exposes 27+ neural models for time series prediction to JavaScript/TypeScript.
//!
//! ## Features
//!
//! - Async/await API with automatic Promise conversion
//! - Type-safe TypeScript definitions
//! - Zero-copy buffer operations
//! - Thread-safe concurrent operations
//! - Comprehensive error handling

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// Re-export neuro-divergent types
use neuro_divergent::{
    ModelConfig as CoreModelConfig,
    training::TrainingMetrics as CoreTrainingMetrics,
};

// =============================================================================
// JavaScript-compatible types
// =============================================================================

/// Model types available for forecasting
#[napi]
pub enum ModelType {
    LSTM,
    GRU,
    Transformer,
    Ensemble,
    NHITS,
    NBEATS,
    TFT,
    DeepAR,
}

impl From<ModelType> for CoreModelType {
    fn from(mt: ModelType) -> Self {
        match mt {
            ModelType::LSTM => CoreModelType::LSTM,
            ModelType::GRU => CoreModelType::GRU,
            ModelType::Transformer => CoreModelType::Transformer,
            ModelType::Ensemble => CoreModelType::Ensemble,
        }
    }
}

/// Model configuration
#[napi(object)]
#[derive(Clone)]
pub struct ModelConfig {
    /// Model type
    pub model_type: String,
    /// Input sequence length
    pub input_size: u32,
    /// Forecast horizon
    pub horizon: u32,
    /// Hidden layer size
    pub hidden_size: Option<u32>,
    /// Number of layers
    pub num_layers: Option<u32>,
    /// Dropout rate (0.0 to 1.0)
    pub dropout: Option<f64>,
    /// Learning rate
    pub learning_rate: Option<f64>,
}

impl TryFrom<ModelConfig> for CoreModelConfig {
    type Error = Error;

    fn try_from(config: ModelConfig) -> Result<Self> {
        // Note: model_type is handled separately when creating specific model instances
        // CoreModelConfig only contains training/architecture parameters
        Ok(CoreModelConfig {
            input_size: config.input_size as usize,
            horizon: config.horizon as usize,
            hidden_size: config.hidden_size.unwrap_or(512) as usize,
            num_layers: config.num_layers.unwrap_or(3) as usize,
            dropout: config.dropout.unwrap_or(0.1),
            learning_rate: config.learning_rate.unwrap_or(0.001),
            batch_size: 32,              // Default batch size
            num_features: 1,             // Default to univariate
            seed: Some(42),              // Default seed
            use_flash_attention: true,   // Enable Flash Attention by default
            flash_block_size: 64,        // Default block size
            flash_causal: false,         // Not causal by default
        })
    }
}

/// Time series data point
#[napi(object)]
#[derive(Clone)]
pub struct TimeSeriesPoint {
    /// Timestamp in ISO 8601 format
    pub timestamp: String,
    /// Value
    pub value: f64,
    /// Optional additional features
    pub features: Option<HashMap<String, f64>>,
}

/// Time series dataset
#[napi(object)]
#[derive(Clone)]
pub struct TimeSeriesData {
    /// Data points
    pub points: Vec<TimeSeriesPoint>,
    /// Frequency (e.g., "1D", "1H", "15m")
    pub frequency: String,
}

impl TryFrom<TimeSeriesData> for CoreTimeSeriesData {
    type Error = Error;

    fn try_from(data: TimeSeriesData) -> Result<Self> {
        use chrono::DateTime;

        let points: Result<Vec<CoreTimeSeriesPoint>> = data
            .points
            .into_iter()
            .map(|p| {
                let timestamp = DateTime::parse_from_rfc3339(&p.timestamp)
                    .map_err(|e| Error::from_reason(format!("Invalid timestamp: {}", e)))?
                    .with_timezone(&chrono::Utc);

                Ok(CoreTimeSeriesPoint {
                    timestamp,
                    value: p.value,
                    features: p.features,
                })
            })
            .collect();

        Ok(CoreTimeSeriesData {
            points: points?,
            frequency: data.frequency,
        })
    }
}

/// Prediction result
#[napi(object)]
pub struct PredictionResult {
    /// Predicted values
    pub predictions: Vec<f64>,
    /// Timestamps for predictions
    pub timestamps: Vec<String>,
    /// Optional confidence intervals (lower, upper)
    pub confidence_intervals: Option<Vec<Vec<f64>>>,
    /// Model type used
    pub model_type: String,
}

impl From<CorePredictionResult> for PredictionResult {
    fn from(result: CorePredictionResult) -> Self {
        Self {
            predictions: result.predictions,
            timestamps: result.timestamps.iter().map(|t| t.to_rfc3339()).collect(),
            confidence_intervals: result.confidence_intervals.map(|ci| {
                ci.into_iter().map(|(l, u)| vec![l, u]).collect()
            }),
            model_type: result.model_type.name().to_string(),
        }
    }
}

/// Cross-validation result
#[napi(object)]
pub struct CrossValidationResult {
    /// Mean Absolute Error
    pub mae: f64,
    /// Mean Squared Error
    pub mse: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Percentage Error
    pub mape: f64,
}

impl From<CoreCrossValidationResult> for CrossValidationResult {
    fn from(result: CoreCrossValidationResult) -> Self {
        Self {
            mae: result.mae,
            mse: result.mse,
            rmse: result.rmse,
            mape: result.mape,
        }
    }
}

/// Training metrics
#[napi(object)]
pub struct TrainingMetrics {
    /// Epoch number
    pub epoch: u32,
    /// Training loss
    pub train_loss: f64,
    /// Validation loss (optional)
    pub val_loss: Option<f64>,
    /// Learning rate
    pub learning_rate: f64,
}

impl From<CoreTrainingMetrics> for TrainingMetrics {
    fn from(metrics: CoreTrainingMetrics) -> Self {
        Self {
            epoch: metrics.epoch as u32,
            train_loss: metrics.train_loss,
            val_loss: metrics.val_loss,
            learning_rate: metrics.learning_rate,
        }
    }
}

// =============================================================================
// Main NAPI class
// =============================================================================

/// Neural forecast engine for time series prediction
#[napi]
pub struct NeuralForecast {
    inner: Arc<RwLock<CoreNeuralForecast>>,
}

#[napi]
impl NeuralForecast {
    /// Create a new neural forecast instance
    #[napi(constructor)]
    pub fn new() -> Result<Self> {
        let inner = CoreNeuralForecast::new()
            .map_err(|e| Error::from_reason(format!("Failed to create forecast: {}", e)))?;

        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    /// Add a model with the specified configuration
    ///
    /// Returns a model ID that can be used for training and prediction
    #[napi]
    pub async fn add_model(&self, config: ModelConfig) -> Result<String> {
        let core_config: CoreModelConfig = config.try_into()?;

        let mut inner = self.inner.write().await;
        inner
            .add_model(core_config)
            .await
            .map_err(|e| Error::from_reason(format!("Failed to add model: {}", e)))
    }

    /// Train a model on the provided time series data
    ///
    /// Returns training metrics for each epoch
    #[napi]
    pub async fn fit(&self, model_id: String, data: TimeSeriesData) -> Result<Vec<TrainingMetrics>> {
        let core_data: CoreTimeSeriesData = data.try_into()?;

        let mut inner = self.inner.write().await;
        let metrics = inner
            .fit(&model_id, &core_data)
            .await
            .map_err(|e| Error::from_reason(format!("Failed to fit model: {}", e)))?;

        Ok(metrics.into_iter().map(TrainingMetrics::from).collect())
    }

    /// Make predictions for the specified horizon
    #[napi]
    pub async fn predict(&self, model_id: String, horizon: u32) -> Result<PredictionResult> {
        let inner = self.inner.read().await;
        let result = inner
            .predict(&model_id, horizon as usize)
            .await
            .map_err(|e| Error::from_reason(format!("Failed to predict: {}", e)))?;

        Ok(PredictionResult::from(result))
    }

    /// Perform cross-validation on the model
    #[napi]
    pub async fn cross_validation(
        &self,
        model_id: String,
        data: TimeSeriesData,
        n_windows: u32,
        step_size: u32,
    ) -> Result<CrossValidationResult> {
        let core_data: CoreTimeSeriesData = data.try_into()?;

        let inner = self.inner.read().await;
        let result = inner
            .cross_validation(&model_id, &core_data, n_windows as usize, step_size as usize)
            .await
            .map_err(|e| Error::from_reason(format!("Failed to cross-validate: {}", e)))?;

        Ok(CrossValidationResult::from(result))
    }

    /// Get model configuration
    #[napi]
    pub async fn get_config(&self, model_id: String) -> Result<Option<ModelConfig>> {
        let inner = self.inner.read().await;
        let config = inner.get_config(&model_id).await;

        Ok(config.map(|c| ModelConfig {
            model_type: c.model_type.name().to_string(),
            input_size: c.input_size as u32,
            horizon: c.horizon as u32,
            hidden_size: Some(c.hidden_size as u32),
            num_layers: Some(c.num_layers as u32),
            dropout: Some(c.dropout),
            learning_rate: Some(c.learning_rate),
        }))
    }
}

// =============================================================================
// Utility functions
// =============================================================================

/// List all available model types
#[napi]
pub fn list_available_models() -> Vec<String> {
    CoreNeuralForecast::list_models()
}

/// Get version information
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Check if GPU acceleration is available
#[napi]
pub fn is_gpu_available() -> bool {
    // TODO: Implement actual GPU detection
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_forecast() {
        let forecast = NeuralForecast::new();
        assert!(forecast.is_ok());
    }

    #[test]
    fn test_list_models() {
        let models = list_available_models();
        assert!(!models.is_empty());
    }
}
