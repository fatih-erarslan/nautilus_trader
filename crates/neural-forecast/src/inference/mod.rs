//! Real-time inference engine

use crate::config::InferenceConfig;
use crate::models::Model;
use crate::{Result, NeuralForecastError};
use ndarray::Array3;
use tokio::sync::RwLock;
use std::sync::Arc;

/// Real-time inference engine
pub struct InferenceEngine {
    config: InferenceConfig,
    models: Vec<Arc<RwLock<Box<dyn Model<Config = crate::models::DynamicModelConfig>>>>>,
}

impl InferenceEngine {
    /// Create a new inference engine with the given configuration
    pub fn new(config: InferenceConfig) -> Self {
        Self {
            config,
            models: Vec::new(),
        }
    }
    
    /// Make a prediction using the loaded models
    pub async fn predict(&self, input: &Array3<f32>) -> Result<Array3<f32>> {
        if self.models.is_empty() {
            return Err(NeuralForecastError::InferenceError(
                "No models loaded".to_string()
            ));
        }
        
        // Use first model for now
        let model = self.models[0].read().await;
        model.predict(input).await
    }
    
    /// Make batch predictions using the loaded models
    pub async fn predict_batch(&self, inputs: &[Array3<f32>]) -> Result<Vec<Array3<f32>>> {
        let mut results = Vec::new();
        
        for input in inputs {
            let prediction = self.predict(input).await?;
            results.push(prediction);
        }
        
        Ok(results)
    }
}

// Re-export is already handled in the use statement above