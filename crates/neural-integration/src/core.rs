/*!
Core neural integration functionality
====================================
*/

use crate::{NeuralConfig, NeuralPrediction, ModelMetrics};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug, error};

/// Neural engine coordination hub
#[derive(Debug)]
pub struct NeuralEngine {
    config: NeuralConfig,
    predictors: Arc<RwLock<HashMap<String, Arc<dyn NeuralPredictor>>>>,
    performance_tracker: PerformanceTracker,
}

/// Neural predictor trait
#[async_trait::async_trait]
pub trait NeuralPredictor: Send + Sync {
    /// Execute prediction
    async fn predict(&self, input: &NeuralInput) -> Result<NeuralPrediction>;
    
    /// Get predictor identifier
    fn id(&self) -> &str;
    
    /// Get performance metrics
    fn metrics(&self) -> ModelMetrics;
}

/// Neural input data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralInput {
    /// Market data features
    pub features: Vec<f64>,
    /// Symbol being analyzed
    pub symbol: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Performance tracking system
#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    predictions_made: Arc<RwLock<u64>>,
    total_latency_us: Arc<RwLock<u64>>,
    successful_predictions: Arc<RwLock<u64>>,
}

impl PerformanceTracker {
    /// Create new performance tracker
    pub fn new() -> Self {
        Self {
            predictions_made: Arc::new(RwLock::new(0)),
            total_latency_us: Arc::new(RwLock::new(0)),
            successful_predictions: Arc::new(RwLock::new(0)),
        }
    }
    
    /// Record a successful prediction
    pub async fn record_prediction(&self, latency_us: u64) {
        let mut predictions = self.predictions_made.write().await;
        let mut total_latency = self.total_latency_us.write().await;
        let mut successful = self.successful_predictions.write().await;
        
        *predictions += 1;
        *total_latency += latency_us;
        *successful += 1;
    }
    
    /// Get average latency
    pub async fn average_latency_us(&self) -> f64 {
        let predictions = *self.predictions_made.read().await;
        let total_latency = *self.total_latency_us.read().await;
        
        if predictions > 0 {
            total_latency as f64 / predictions as f64
        } else {
            0.0
        }
    }
    
    /// Get success rate
    pub async fn success_rate(&self) -> f64 {
        let predictions = *self.predictions_made.read().await;
        let successful = *self.successful_predictions.read().await;
        
        if predictions > 0 {
            successful as f64 / predictions as f64
        } else {
            0.0
        }
    }
}

impl NeuralEngine {
    /// Create new neural engine
    pub fn new(config: NeuralConfig) -> Self {
        Self {
            config,
            predictors: Arc::new(RwLock::new(HashMap::new())),
            performance_tracker: PerformanceTracker::new(),
        }
    }
    
    /// Register a neural predictor
    pub async fn register_predictor(&self, predictor: Arc<dyn NeuralPredictor>) -> Result<()> {
        let id = predictor.id().to_string();
        info!("Registering neural predictor: {}", id);
        
        self.predictors.write().await.insert(id.clone(), predictor);
        
        debug!("Neural predictor {} registered successfully", id);
        Ok(())
    }
    
    /// Execute prediction using specific predictor
    pub async fn predict(&self, predictor_id: &str, input: &NeuralInput) -> Result<NeuralPrediction> {
        let start_time = std::time::Instant::now();
        
        let predictors = self.predictors.read().await;
        let predictor = predictors.get(predictor_id)
            .ok_or_else(|| anyhow::anyhow!("Predictor {} not found", predictor_id))?;
        
        let prediction = predictor.predict(input).await
            .context("Neural prediction failed")?;
        
        let latency_us = start_time.elapsed().as_micros() as u64;
        
        // Record performance
        self.performance_tracker.record_prediction(latency_us).await;
        
        // Check latency target
        if latency_us > self.config.target_latency_us {
            debug!(
                "Prediction latency {}μs exceeds target {}μs",
                latency_us, self.config.target_latency_us
            );
        }
        
        Ok(prediction)
    }
    
    /// Execute ensemble prediction using all predictors
    pub async fn predict_ensemble(&self, input: &NeuralInput) -> Result<Vec<NeuralPrediction>> {
        let predictors = self.predictors.read().await;
        let predictor_ids: Vec<String> = predictors.keys().cloned().collect();
        drop(predictors);
        
        let mut predictions = Vec::new();
        
        for predictor_id in predictor_ids {
            match self.predict(&predictor_id, input).await {
                Ok(prediction) => predictions.push(prediction),
                Err(e) => error!("Ensemble prediction failed for {}: {}", predictor_id, e),
            }
        }
        
        info!("Ensemble prediction completed with {} models", predictions.len());
        Ok(predictions)
    }
    
    /// Get system performance metrics
    pub async fn get_performance_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        metrics.insert(
            "average_latency_us".to_string(),
            self.performance_tracker.average_latency_us().await
        );
        
        metrics.insert(
            "success_rate".to_string(),
            self.performance_tracker.success_rate().await
        );
        
        let predictors = self.predictors.read().await;
        metrics.insert(
            "registered_predictors".to_string(),
            predictors.len() as f64
        );
        
        metrics
    }
}

/// ruv-FANN neural predictor implementation
pub struct RuvFannPredictor {
    id: String,
    config: crate::RuvFannConfig,
    metrics: ModelMetrics,
}

impl RuvFannPredictor {
    /// Create new ruv-FANN predictor
    pub fn new(id: String, config: crate::RuvFannConfig) -> Self {
        Self {
            id: id.clone(),
            config,
            metrics: ModelMetrics {
                model_id: id,
                total_predictions: 0,
                avg_execution_time_us: 0.0,
                accuracy: 0.0,
                sharpe_ratio: 0.0,
                max_drawdown: 0.0,
                total_return: 0.0,
                last_updated: chrono::Utc::now(),
            },
        }
    }
}

#[async_trait::async_trait]
impl NeuralPredictor for RuvFannPredictor {
    async fn predict(&self, input: &NeuralInput) -> Result<NeuralPrediction> {
        let start_time = std::time::Instant::now();
        
        // Simulate ruv-FANN prediction
        // In real implementation, this would use the actual ruv-FANN network
        let features_sum = input.features.iter().sum::<f64>();
        let features_mean = features_sum / input.features.len() as f64;
        
        // Generate prediction value based on features
        let value = features_mean * 1.05; // Simple transformation
        let confidence = 0.80 + (features_mean.abs() * 0.1).min(0.19);
        
        let execution_time_us = start_time.elapsed().as_micros() as u64;
        
        Ok(NeuralPrediction {
            id: uuid::Uuid::new_v4(),
            model_id: self.id.clone(),
            confidence,
            value,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("features_count".to_string(), serde_json::Value::from(input.features.len()));
                meta.insert("predictor_type".to_string(), serde_json::Value::from("ruv-fann"));
                meta.insert("ultra_performance".to_string(), serde_json::Value::from(self.config.ultra_performance));
                meta
            },
            execution_time_us,
            timestamp: chrono::Utc::now(),
        })
    }
    
    fn id(&self) -> &str {
        &self.id
    }
    
    fn metrics(&self) -> ModelMetrics {
        self.metrics.clone()
    }
}

/// Cognition Engine neural predictor implementation
pub struct CognitionEnginePredictor {
    id: String,
    config: crate::CognitionEngineConfig,
    metrics: ModelMetrics,
}

impl CognitionEnginePredictor {
    /// Create new cognition engine predictor
    pub fn new(id: String, config: crate::CognitionEngineConfig) -> Self {
        Self {
            id: id.clone(),
            config,
            metrics: ModelMetrics {
                model_id: id,
                total_predictions: 0,
                avg_execution_time_us: 0.0,
                accuracy: 0.0,
                sharpe_ratio: 0.0,
                max_drawdown: 0.0,
                total_return: 0.0,
                last_updated: chrono::Utc::now(),
            },
        }
    }
}

#[async_trait::async_trait]
impl NeuralPredictor for CognitionEnginePredictor {
    async fn predict(&self, input: &NeuralInput) -> Result<NeuralPrediction> {
        let start_time = std::time::Instant::now();
        
        // Simulate cognition engine NHITS forecasting
        let features = &input.features;
        
        // Generate forecast using NHITS-style prediction
        let mut forecast_value = 0.0;
        for (i, &feature) in features.iter().enumerate() {
            let weight = 1.0 / (i + 1) as f64; // Decreasing weights
            forecast_value += feature * weight;
        }
        
        let value = forecast_value / features.len() as f64;
        let confidence = 0.75 + (value.abs() * 0.15).min(0.24);
        
        let execution_time_us = start_time.elapsed().as_micros() as u64;
        
        Ok(NeuralPrediction {
            id: uuid::Uuid::new_v4(),
            model_id: self.id.clone(),
            confidence,
            value,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("predictor_type".to_string(), serde_json::Value::from("cognition-engine"));
                meta.insert("nhits_enabled".to_string(), serde_json::Value::from(self.config.enable_nhits));
                meta.insert("forecast_horizon".to_string(), serde_json::Value::from(self.config.forecast_horizon));
                meta.insert("ensemble_size".to_string(), serde_json::Value::from(self.config.ensemble_size));
                meta
            },
            execution_time_us,
            timestamp: chrono::Utc::now(),
        })
    }
    
    fn id(&self) -> &str {
        &self.id
    }
    
    fn metrics(&self) -> ModelMetrics {
        self.metrics.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_neural_engine_creation() {
        let config = NeuralConfig::default();
        let engine = NeuralEngine::new(config);
        
        let metrics = engine.get_performance_metrics().await;
        assert_eq!(metrics["registered_predictors"], 0.0);
    }
    
    #[tokio::test]
    async fn test_predictor_registration() {
        let config = NeuralConfig::default();
        let engine = NeuralEngine::new(config.clone());
        
        let predictor = Arc::new(RuvFannPredictor::new(
            "test_ruv_fann".to_string(),
            config.ruv_fann,
        ));
        
        engine.register_predictor(predictor).await.unwrap();
        
        let metrics = engine.get_performance_metrics().await;
        assert_eq!(metrics["registered_predictors"], 1.0);
    }
    
    #[tokio::test]
    async fn test_neural_prediction() {
        let config = NeuralConfig::default();
        let engine = NeuralEngine::new(config.clone());
        
        let predictor = Arc::new(RuvFannPredictor::new(
            "test_ruv_fann".to_string(),
            config.ruv_fann,
        ));
        
        engine.register_predictor(predictor).await.unwrap();
        
        let input = NeuralInput {
            features: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            symbol: "EURUSD".to_string(),
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        };
        
        let prediction = engine.predict("test_ruv_fann", &input).await.unwrap();
        
        assert_eq!(prediction.model_id, "test_ruv_fann");
        assert!(prediction.confidence > 0.0);
        assert!(prediction.execution_time_us > 0);
    }
    
    #[tokio::test]
    async fn test_ensemble_prediction() {
        let config = NeuralConfig::default();
        let engine = NeuralEngine::new(config.clone());
        
        // Register multiple predictors
        let ruv_fann = Arc::new(RuvFannPredictor::new(
            "ruv_fann".to_string(),
            config.ruv_fann.clone(),
        ));
        
        let cognition = Arc::new(CognitionEnginePredictor::new(
            "cognition_engine".to_string(),
            config.cognition_engine.clone(),
        ));
        
        engine.register_predictor(ruv_fann).await.unwrap();
        engine.register_predictor(cognition).await.unwrap();
        
        let input = NeuralInput {
            features: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            symbol: "EURUSD".to_string(),
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        };
        
        let predictions = engine.predict_ensemble(&input).await.unwrap();
        
        assert_eq!(predictions.len(), 2);
        assert!(predictions.iter().all(|p| p.confidence > 0.0));
    }
}