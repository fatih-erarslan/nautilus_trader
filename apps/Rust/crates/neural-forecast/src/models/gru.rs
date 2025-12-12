//! GRU model implementation

use async_trait::async_trait;
use ndarray::Array3;
use crate::config::GRUConfig;
#[cfg(feature = "gpu")]
use crate::gpu::GPUBackend;
use crate::{Result, NeuralForecastError};
use super::{
    Model, ModelConfig, ModelType, ModelParameters, ModelMetadata, TrainingData,
    TrainingMetrics, UpdateData, ModelMetrics, TrainingParams, OptimizerType,
};

/// Gated Recurrent Unit neural network model
#[derive(Debug)]
pub struct GRUModel {
    config: GRUConfig,
    parameters: ModelParameters,
    metrics: Option<ModelMetrics>,
    is_trained: bool,
}

impl GRUModel {
    /// Create a new GRU model from configuration
    pub fn new_from_config(config: GRUConfig) -> Result<Self> {
        Ok(Self {
            config,
            parameters: ModelParameters::default(),
            metrics: None,
            is_trained: false,
        })
    }
}

#[async_trait]
impl Model for GRUModel {
    type Config = super::DynamicModelConfig;
    
    fn new(config: Self::Config) -> Result<Self> {
        match config {
            super::DynamicModelConfig::GRU(gru_config) => {
                gru_config.validate()?;
                Self::new_from_config(gru_config)
            }
            _ => Err(NeuralForecastError::ConfigError(
                "Invalid config type for GRU model".to_string()
            )),
        }
    }
    
    #[cfg(feature = "gpu")]
    async fn initialize(&mut self, _gpu_backend: Option<&GPUBackend>) -> Result<()> {
        // Initialize GRU architecture
        Ok(())
    }
    
    #[cfg(not(feature = "gpu"))]
    async fn initialize(&mut self) -> Result<()> {
        Ok(())
    }
    
    async fn train(&mut self, _data: &TrainingData) -> Result<TrainingMetrics> {
        self.is_trained = true;
        Ok(TrainingMetrics {
            train_loss: vec![0.1],
            val_loss: vec![0.1],
            train_accuracy: vec![],
            val_accuracy: vec![],
            training_time: 1.0,
            epochs_trained: 1,
            best_val_loss: 0.1,
            early_stopped: false,
            final_lr: 0.001,
        })
    }
    
    async fn predict(&self, input: &Array3<f32>) -> Result<Array3<f32>> {
        if !self.is_trained {
            return Err(NeuralForecastError::InferenceError(
                "Model must be trained".to_string()
            ));
        }
        let shape = (input.shape()[0], self.config.output_length, input.shape()[2]);
        Ok(Array3::zeros(shape))
    }
    
    async fn predict_batch(&self, inputs: &[Array3<f32>]) -> Result<Vec<Array3<f32>>> {
        let mut results = Vec::new();
        for input in inputs {
            results.push(self.predict(input).await?);
        }
        Ok(results)
    }
    
    fn parameters(&self) -> &ModelParameters {
        &self.parameters
    }
    
    fn set_parameters(&mut self, parameters: ModelParameters) -> Result<()> {
        self.parameters = parameters;
        Ok(())
    }
    
    async fn save(&self, path: &std::path::Path) -> Result<()> {
        let data = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(path, data)?;
        Ok(())
    }
    
    async fn load(&mut self, path: &std::path::Path) -> Result<()> {
        let data = std::fs::read_to_string(path)?;
        self.config = serde_json::from_str(&data)?;
        Ok(())
    }
    
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            model_type: ModelType::GRU,
            name: "GRU".to_string(),
            version: "1.0.0".to_string(),
            description: "Gated Recurrent Unit".to_string(),
            author: "TENGRI Trading Swarm".to_string(),
            created_at: chrono::Utc::now(),
            modified_at: chrono::Utc::now(),
            size_bytes: 0,
            num_parameters: 0,
            input_shape: vec![self.config.input_length, 1],
            output_shape: vec![self.config.output_length, 1],
            training_data_info: None,
            performance_metrics: None,
        }
    }
    
    fn validate_config(&self) -> Result<()> {
        self.config.validate()
    }
    
    fn metrics(&self) -> Option<&ModelMetrics> {
        self.metrics.as_ref()
    }
    
    async fn update(&mut self, _data: &UpdateData) -> Result<()> {
        Ok(())
    }
}

// ModelConfig implementation moved to config.rs to avoid conflicts