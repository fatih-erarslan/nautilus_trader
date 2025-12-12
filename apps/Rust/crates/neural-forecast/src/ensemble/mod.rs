//! Ensemble forecasting system

use crate::models::{Model, VotingStrategy};
use crate::config::EnsembleConfig;
use crate::{Result, NeuralForecastError};
use ndarray::Array3;
use async_trait::async_trait;

/// Forecast ensemble for combining multiple models
pub struct ForecastEnsemble {
    models: Vec<Box<dyn Model<Config = crate::models::DynamicModelConfig>>>,
    config: EnsembleConfig,
}

impl ForecastEnsemble {
    /// Create new forecast ensemble
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            config: EnsembleConfig::default(),
        }
    }
    
    /// Add model to ensemble
    pub fn add_model(mut self, model: Box<dyn Model<Config = crate::models::DynamicModelConfig>>) -> Self {
        self.models.push(model);
        self
    }
    
    /// Make ensemble prediction
    pub async fn predict(&self, input: &Array3<f32>) -> Result<Array3<f32>> {
        if self.models.is_empty() {
            return Err(NeuralForecastError::InferenceError(
                "No models in ensemble".to_string()
            ));
        }
        
        // Get predictions from all models
        let mut predictions = Vec::new();
        for model in &self.models {
            let prediction = model.predict(input).await?;
            predictions.push(prediction);
        }
        
        // Combine predictions based on voting strategy
        self.combine_predictions(predictions)
    }
    
    fn combine_predictions(&self, predictions: Vec<Array3<f32>>) -> Result<Array3<f32>> {
        if predictions.is_empty() {
            return Err(NeuralForecastError::InferenceError(
                "No predictions to combine".to_string()
            ));
        }
        
        let shape = predictions[0].shape();
        let mut result = Array3::<f32>::zeros((shape[0], shape[1], shape[2]));
        
        // Simple average for now
        for pred in &predictions {
            result = result + pred;
        }
        result = result / predictions.len() as f32;
        
        Ok(result)
    }
}

impl Default for ForecastEnsemble {
    fn default() -> Self {
        Self::new()
    }
}

// Re-exports are already handled in the use statements above