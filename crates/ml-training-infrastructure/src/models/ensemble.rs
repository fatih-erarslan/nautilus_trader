//! Ensemble model implementation

use crate::{Result, TrainingError};
use crate::data::TrainingData;
use crate::config::TrainingParams;
use crate::models::{Model, ModelType, ModelParameters, ModelMetadata, TrainingMetrics};
use async_trait::async_trait;
use ndarray::Array3;
use std::path::Path;

/// Ensemble model combining multiple models
pub struct EnsembleModel {
    models: Vec<Box<dyn Model>>,
    weights: Vec<f32>,
    metadata: ModelMetadata,
}

impl EnsembleModel {
    /// Create new ensemble model
    pub fn new(models: Vec<Box<dyn Model>>) -> Result<Self> {
        let n_models = models.len();
        let weights = vec![1.0 / n_models as f32; n_models];
        
        Ok(Self {
            models,
            weights,
            metadata: ModelMetadata {
                model_type: ModelType::Ensemble,
                version: "1.0.0".to_string(),
                created_at: chrono::Utc::now(),
                modified_at: chrono::Utc::now(),
                config: serde_json::json!({}),
                metrics: None,
            },
        })
    }
}

#[async_trait]
impl Model for EnsembleModel {
    async fn train(&mut self, data: &TrainingData, config: &TrainingParams) -> Result<TrainingMetrics> {
        // Train all models in parallel
        let mut handles = vec![];
        
        // Note: In a real implementation, we'd need to handle the async training properly
        // For now, train sequentially
        for model in &mut self.models {
            model.train(data, config).await?;
        }
        
        Ok(TrainingMetrics {
            train_loss: vec![0.0],
            val_loss: vec![0.0],
            train_metrics: vec![],
            val_metrics: vec![],
            best_epoch: 0,
            training_time_secs: 0.0,
            early_stopped: false,
        })
    }
    
    async fn predict(&self, inputs: &Array3<f32>) -> Result<Array3<f32>> {
        let mut predictions = vec![];
        
        // Get predictions from all models
        for model in &self.models {
            predictions.push(model.predict(inputs).await?);
        }
        
        // Weighted average
        let mut result = Array3::<f32>::zeros(predictions[0].dim());
        for (pred, weight) in predictions.iter().zip(&self.weights) {
            result = result + pred * *weight;
        }
        
        Ok(result)
    }
    
    async fn predict_batch(&self, inputs: &[Array3<f32>]) -> Result<Vec<Array3<f32>>> {
        let mut results = Vec::new();
        for input in inputs {
            results.push(self.predict(input).await?);
        }
        Ok(results)
    }
    
    async fn save(&self, path: &Path) -> Result<()> {
        // Save each model with a unique suffix
        for (i, model) in self.models.iter().enumerate() {
            let model_path = path.with_file_name(format!("{}_model_{}", path.file_stem().unwrap().to_str().unwrap(), i));
            model.save(&model_path).await?;
        }
        Ok(())
    }
    
    async fn load(&mut self, path: &Path) -> Result<()> {
        // Load each model
        for (i, model) in self.models.iter_mut().enumerate() {
            let model_path = path.with_file_name(format!("{}_model_{}", path.file_stem().unwrap().to_str().unwrap(), i));
            model.load(&model_path).await?;
        }
        Ok(())
    }
    
    fn parameters(&self) -> ModelParameters {
        ModelParameters {
            weights: Vec::new(),
            biases: Vec::new(),
            extra: serde_json::json!({
                "ensemble_weights": self.weights,
                "n_models": self.models.len(),
            }),
        }
    }
    
    fn set_parameters(&mut self, _params: ModelParameters) -> Result<()> {
        Ok(())
    }
    
    fn metadata(&self) -> ModelMetadata {
        self.metadata.clone()
    }
    
    fn validate_input(&self, input: &Array3<f32>) -> Result<()> {
        // Validate with first model
        if let Some(model) = self.models.first() {
            model.validate_input(input)
        } else {
            Err(TrainingError::Validation("No models in ensemble".to_string()))
        }
    }
    
    fn model_type(&self) -> ModelType {
        ModelType::Ensemble
    }
}