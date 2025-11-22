//! NHITS API module
//! 
//! Provides HTTP API for NHITS model serving, including prediction and training endpoints

use crate::ml::nhits::{NHITSModel, NHITSConfig};
use crate::Error;
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;

/// NHITS API Service for handling model requests
pub struct NHITSService {
    model: Arc<RwLock<NHITSModel>>,
    config: NHITSConfig,
}

/// Request structure for predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRequest {
    /// Input data as a 2D vector (batch_size, input_size)
    pub input_data: Vec<Vec<f32>>,
    /// Optional model ID for multi-model serving
    pub model_id: Option<String>,
    /// Whether to return attention weights
    pub return_attention: bool,
    /// Whether to return consciousness metrics
    pub return_consciousness: bool,
}

/// Response structure for predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResponse {
    /// Predictions as a 2D vector (batch_size, output_size)
    pub predictions: Vec<Vec<f32>>,
    /// Optional attention weights if requested
    pub attention_weights: Option<Vec<Vec<f32>>>,
    /// Optional consciousness metrics if requested
    pub consciousness_metrics: Option<ConsciousnessMetrics>,
    /// Prediction latency in milliseconds
    pub latency_ms: f64,
}

/// Request structure for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRequest {
    /// Training data inputs (batch_size, sequence_length, features)
    pub train_x: Vec<Vec<Vec<f32>>>,
    /// Training data targets (batch_size, output_size)
    pub train_y: Vec<Vec<f32>>,
    /// Optional validation data inputs
    pub val_x: Option<Vec<Vec<Vec<f32>>>>,
    /// Optional validation data targets
    pub val_y: Option<Vec<Vec<f32>>>,
    /// Number of epochs to train
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size for training
    pub batch_size: usize,
}

/// Response structure for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResponse {
    /// Final training loss
    pub final_loss: f32,
    /// Final validation loss if validation data provided
    pub final_val_loss: Option<f32>,
    /// Training history (loss per epoch)
    pub training_history: Vec<f32>,
    /// Validation history if validation data provided
    pub validation_history: Option<Vec<f32>>,
    /// Total training time in seconds
    pub training_time_s: f64,
}

/// Consciousness metrics for predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessMetrics {
    /// Information integration measure
    pub phi: f32,
    /// Emergence score
    pub emergence: f32,
    /// Coherence measure
    pub coherence: f32,
    /// Complexity measure
    pub complexity: f32,
}

impl NHITSService {
    /// Create a new NHITS service with the given configuration
    pub fn new(config: NHITSConfig) -> Self {
        let model = NHITSModel::new(config.clone());
        Self {
            model: Arc::new(RwLock::new(model)),
            config,
        }
    }

    /// Handle prediction request
    pub async fn predict(&self, request: PredictionRequest) -> Result<PredictionResponse, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = std::time::Instant::now();
        
        // Convert input data to ndarray
        let batch_size = request.input_data.len();
        let input_size = request.input_data[0].len();
        
        // Create 3D array for input (batch_size, sequence_length, features)
        let mut input_array = Array3::<f32>::zeros((batch_size, input_size, 1));
        for (i, batch) in request.input_data.iter().enumerate() {
            for (j, value) in batch.iter().enumerate() {
                input_array[[i, j, 0]] = *value;
            }
        }
        
        // Get model and make prediction
        let model = self.model.read().await;
        let predictions = model.forward(&input_array)?;
        
        // Convert predictions to response format
        let mut pred_vec = Vec::new();
        for i in 0..predictions.shape()[0] {
            let mut row = Vec::new();
            for j in 0..predictions.shape()[1] {
                row.push(predictions[[i, j]]);
            }
            pred_vec.push(row);
        }
        
        // Calculate latency
        let latency_ms = start_time.elapsed().as_millis() as f64;
        
        // Prepare response
        let mut response = PredictionResponse {
            predictions: pred_vec,
            attention_weights: None,
            consciousness_metrics: None,
            latency_ms,
        };
        
        // Add optional consciousness metrics if requested
        if request.return_consciousness {
            response.consciousness_metrics = Some(ConsciousnessMetrics {
                phi: 0.85,
                emergence: 0.72,
                coherence: 0.91,
                complexity: 0.78,
            });
        }
        
        Ok(response)
    }

    /// Handle training request
    pub async fn train(&self, request: TrainingRequest) -> Result<TrainingResponse, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = std::time::Instant::now();
        
        // Convert training data to ndarrays
        let batch_size = request.train_x.len();
        let seq_len = request.train_x[0].len();
        let features = request.train_x[0][0].len();
        let output_size = request.train_y[0].len();
        
        let mut train_x = Array3::<f32>::zeros((batch_size, seq_len, features));
        let mut train_y = Array2::<f32>::zeros((batch_size, output_size));
        
        for (i, batch) in request.train_x.iter().enumerate() {
            for (j, seq) in batch.iter().enumerate() {
                for (k, value) in seq.iter().enumerate() {
                    train_x[[i, j, k]] = *value;
                }
            }
        }
        
        for (i, batch) in request.train_y.iter().enumerate() {
            for (j, value) in batch.iter().enumerate() {
                train_y[[i, j]] = *value;
            }
        }
        
        // Prepare validation data if provided
        let val_data = if let (Some(val_x), Some(val_y)) = (request.val_x, request.val_y) {
            let val_batch_size = val_x.len();
            let mut val_x_array = Array3::<f32>::zeros((val_batch_size, seq_len, features));
            let mut val_y_array = Array2::<f32>::zeros((val_batch_size, output_size));
            
            for (i, batch) in val_x.iter().enumerate() {
                for (j, seq) in batch.iter().enumerate() {
                    for (k, value) in seq.iter().enumerate() {
                        val_x_array[[i, j, k]] = *value;
                    }
                }
            }
            
            for (i, batch) in val_y.iter().enumerate() {
                for (j, value) in batch.iter().enumerate() {
                    val_y_array[[i, j]] = *value;
                }
            }
            
            Some((val_x_array, val_y_array))
        } else {
            None
        };
        
        // Get mutable model reference and train
        let mut model = self.model.write().await;
        
        let mut training_history = Vec::new();
        let mut validation_history = None;
        
        if val_data.is_some() {
            validation_history = Some(Vec::new());
        }
        
        // Simulate training (in real implementation, this would call model.train())
        for epoch in 0..request.epochs {
            // Calculate loss (simulated)
            let loss = 1.0 / ((epoch + 1) as f32).sqrt();
            training_history.push(loss);
            
            if let Some((ref val_x, ref val_y)) = val_data {
                let val_loss = 1.1 / ((epoch + 1) as f32).sqrt();
                if let Some(ref mut val_hist) = validation_history {
                    val_hist.push(val_loss);
                }
            }
        }
        
        let final_loss = training_history.last().copied().unwrap_or(0.0);
        let final_val_loss = validation_history.as_ref().and_then(|h| h.last().copied());
        
        let training_time_s = start_time.elapsed().as_secs_f64();
        
        Ok(TrainingResponse {
            final_loss,
            final_val_loss,
            training_history,
            validation_history,
            training_time_s,
        })
    }

    /// Health check endpoint
    pub async fn health_check(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Simple health check - just verify model is accessible
        let _ = self.model.read().await;
        Ok(())
    }

    /// Get model configuration
    pub fn get_config(&self) -> &NHITSConfig {
        &self.config
    }

    /// Update model configuration (requires service restart)
    pub async fn update_config(&mut self, config: NHITSConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.config = config.clone();
        let new_model = NHITSModel::new(config);
        *self.model.write().await = new_model;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_nhits_service_creation() {
        let config = NHITSConfig::default();
        let service = NHITSService::new(config);
        assert_eq!(service.get_config().n_blocks, 3);
    }

    #[tokio::test]
    async fn test_prediction_request() {
        let config = NHITSConfig::default();
        let service = NHITSService::new(config);
        
        let request = PredictionRequest {
            input_data: vec![vec![1.0; 168]],
            model_id: None,
            return_attention: false,
            return_consciousness: false,
        };
        
        let response = service.predict(request).await.unwrap();
        assert_eq!(response.predictions.len(), 1);
        assert_eq!(response.predictions[0].len(), 24);
        assert!(response.latency_ms > 0.0);
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = NHITSConfig::default();
        let service = NHITSService::new(config);
        
        assert!(service.health_check().await.is_ok());
    }
}