//! Model implementations module

use crate::{Result, TrainingError};
use crate::data::TrainingData;
use crate::config::*;
use async_trait::async_trait;
use ndarray::{Array1, Array2, Array3};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::path::Path;

/// Deep learning models
pub mod deep_learning;
/// Gradient boosting models
pub mod gradient_boosting;
/// Classical ML models
pub mod classical;
/// Ensemble models
pub mod ensemble;

// Re-exports
pub use deep_learning::{TransformerModel, LSTMModel};
pub use gradient_boosting::{XGBoostModel, LightGBMModel};
pub use classical::NeuralNetworkModel;
pub use ensemble::EnsembleModel;

/// Model trait for all ML models
#[async_trait]
pub trait Model: Send + Sync {
    /// Train the model
    async fn train(&mut self, data: &TrainingData, config: &TrainingParams) -> Result<TrainingMetrics>;
    
    /// Make predictions
    async fn predict(&self, inputs: &Array3<f32>) -> Result<Array3<f32>>;
    
    /// Batch prediction
    async fn predict_batch(&self, inputs: &[Array3<f32>]) -> Result<Vec<Array3<f32>>>;
    
    /// Save model to disk
    async fn save(&self, path: &Path) -> Result<()>;
    
    /// Load model from disk
    async fn load(&mut self, path: &Path) -> Result<()>;
    
    /// Get model parameters
    fn parameters(&self) -> ModelParameters;
    
    /// Set model parameters
    fn set_parameters(&mut self, params: ModelParameters) -> Result<()>;
    
    /// Get model metadata
    fn metadata(&self) -> ModelMetadata;
    
    /// Validate input dimensions
    fn validate_input(&self, input: &Array3<f32>) -> Result<()>;
    
    /// Get model type
    fn model_type(&self) -> ModelType;
}

/// Model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    /// Transformer model
    Transformer,
    /// LSTM model
    LSTM,
    /// XGBoost model
    XGBoost,
    /// LightGBM model
    LightGBM,
    /// Neural network model
    NeuralNetwork,
    /// Ensemble model
    Ensemble,
}

/// Model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    /// Model weights
    pub weights: Vec<Array2<f32>>,
    /// Model biases
    pub biases: Vec<Array1<f32>>,
    /// Additional parameters
    pub extra: serde_json::Value,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model type
    pub model_type: ModelType,
    /// Model version
    pub version: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modified
    pub modified_at: chrono::DateTime<chrono::Utc>,
    /// Training configuration
    pub config: serde_json::Value,
    /// Performance metrics
    pub metrics: Option<TrainingMetrics>,
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss history
    pub train_loss: Vec<f32>,
    /// Validation loss history
    pub val_loss: Vec<f32>,
    /// Training metrics per epoch
    pub train_metrics: Vec<MetricSet>,
    /// Validation metrics per epoch
    pub val_metrics: Vec<MetricSet>,
    /// Best epoch
    pub best_epoch: usize,
    /// Total training time
    pub training_time_secs: f64,
    /// Early stopped
    pub early_stopped: bool,
}

/// Set of metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSet {
    /// Mean squared error
    pub mse: f32,
    /// Mean absolute error
    pub mae: f32,
    /// Root mean squared error
    pub rmse: f32,
    /// Mean absolute percentage error
    pub mape: Option<f32>,
    /// R-squared
    pub r2: Option<f32>,
    /// Custom metrics
    pub custom: std::collections::HashMap<String, f32>,
}

/// Model factory
pub fn create_model(model_type: ModelType, config: &TrainingConfig) -> Result<Box<dyn Model>> {
    match model_type {
        ModelType::Transformer => {
            let model = TransformerModel::new(config.models.transformer.clone())?;
            Ok(Box::new(model))
        }
        ModelType::LSTM => {
            let model = LSTMModel::new(config.models.lstm.clone())?;
            Ok(Box::new(model))
        }
        ModelType::XGBoost => {
            let model = XGBoostModel::new(config.models.xgboost.clone())?;
            Ok(Box::new(model))
        }
        ModelType::LightGBM => {
            let model = LightGBMModel::new(config.models.lightgbm.clone())?;
            Ok(Box::new(model))
        }
        ModelType::NeuralNetwork => {
            let model = NeuralNetworkModel::new(config.models.neural.clone())?;
            Ok(Box::new(model))
        }
        ModelType::Ensemble => {
            // Create ensemble with multiple models
            let models = vec![
                create_model(ModelType::Transformer, config)?,
                create_model(ModelType::LSTM, config)?,
                create_model(ModelType::XGBoost, config)?,
                create_model(ModelType::LightGBM, config)?,
            ];
            let model = EnsembleModel::new(models)?;
            Ok(Box::new(model))
        }
    }
}

/// Calculate metrics from predictions and targets
pub fn calculate_metrics(predictions: &Array3<f32>, targets: &Array3<f32>) -> Result<MetricSet> {
    let pred_flat = predictions.as_slice().ok_or_else(|| {
        TrainingError::Training("Failed to flatten predictions".to_string())
    })?;
    let target_flat = targets.as_slice().ok_or_else(|| {
        TrainingError::Training("Failed to flatten targets".to_string())
    })?;
    
    let n = pred_flat.len() as f32;
    
    // MSE
    let mse = pred_flat.iter()
        .zip(target_flat.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>() / n;
    
    // MAE
    let mae = pred_flat.iter()
        .zip(target_flat.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f32>() / n;
    
    // RMSE
    let rmse = mse.sqrt();
    
    // MAPE (if no zeros in targets)
    let mape = if target_flat.iter().all(|&t| t.abs() > 1e-8) {
        Some(
            pred_flat.iter()
                .zip(target_flat.iter())
                .map(|(p, t)| ((p - t) / t).abs())
                .sum::<f32>() / n * 100.0
        )
    } else {
        None
    };
    
    // R²
    let target_mean = target_flat.iter().sum::<f32>() / n;
    let ss_tot = target_flat.iter()
        .map(|t| (t - target_mean).powi(2))
        .sum::<f32>();
    let ss_res = pred_flat.iter()
        .zip(target_flat.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum::<f32>();
    
    let r2 = if ss_tot > 1e-8 {
        Some(1.0 - ss_res / ss_tot)
    } else {
        None
    };
    
    Ok(MetricSet {
        mse,
        mae,
        rmse,
        mape,
        r2,
        custom: std::collections::HashMap::new(),
    })
}

/// Model state for checkpointing
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelState {
    /// Model type
    pub model_type: ModelType,
    /// Model parameters
    pub parameters: ModelParameters,
    /// Training epoch
    pub epoch: usize,
    /// Best metrics
    pub best_metrics: Option<MetricSet>,
    /// Optimizer state
    pub optimizer_state: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calculate_metrics() {
        let predictions = Array3::<f32>::ones((10, 5, 3));
        let targets = Array3::<f32>::ones((10, 5, 3));
        
        let metrics = calculate_metrics(&predictions, &targets).unwrap();
        assert!((metrics.mse - 0.0).abs() < 1e-6);
        assert!((metrics.mae - 0.0).abs() < 1e-6);
    }
}