//! Ensemble predictor for whale detection
//! 
//! This module implements an ensemble of different models for robust
//! whale detection with interpretability features.

use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use crate::error::{Result, WhaleMLError};
use crate::transformer::{TransformerWhaleDetector, TransformerConfig};
use crate::metrics::{InferenceTimer, MetricsCollector};

/// Result of whale detection prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Probability of whale activity (0.0 to 1.0)
    pub whale_probability: f32,
    /// Individual model predictions
    pub model_predictions: HashMap<String, f32>,
    /// Confidence score (based on model agreement)
    pub confidence: f32,
    /// Threat level (1-5)
    pub threat_level: u8,
    /// Inference time in microseconds
    pub inference_time_us: u64,
    /// Interpretability features
    pub interpretability: InterpretabilityInfo,
}

/// Interpretability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretabilityInfo {
    /// Most important features contributing to the prediction
    pub top_features: Vec<(String, f32)>,
    /// Attention weights from transformer (if available)
    pub attention_weights: Option<Vec<Vec<f32>>>,
    /// Feature importance scores
    pub feature_importance: HashMap<String, f32>,
    /// Anomaly score (0.0 = normal, 1.0 = highly anomalous)
    pub anomaly_score: f32,
}

/// Model weights for ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleWeights {
    /// Weight for transformer model
    pub transformer: f32,
    /// Weight for LSTM model (future)
    pub lstm: f32,
    /// Weight for gradient boosting (future)
    pub gradient_boost: f32,
    /// Weight for anomaly detection
    pub anomaly: f32,
}

impl Default for EnsembleWeights {
    fn default() -> Self {
        Self {
            transformer: 0.7,  // Primary model gets highest weight
            lstm: 0.0,         // Not implemented yet
            gradient_boost: 0.0,
            anomaly: 0.3,
        }
    }
}

/// Ensemble predictor combining multiple models
pub struct EnsemblePredictor {
    /// Device for computation
    device: Device,
    /// Transformer model
    transformer: Arc<TransformerWhaleDetector>,
    /// Model weights
    weights: EnsembleWeights,
    /// Metrics collector
    metrics: MetricsCollector,
    /// Feature importance tracker
    feature_importance: Arc<RwLock<HashMap<String, f32>>>,
    /// Anomaly detection threshold
    anomaly_threshold: f32,
}

impl EnsemblePredictor {
    /// Create a new ensemble predictor
    pub fn new(device: Device) -> Result<Self> {
        let config = TransformerConfig::default();
        let transformer = TransformerWhaleDetector::new(config, device.clone())?;
        
        Ok(Self {
            device,
            transformer: Arc::new(transformer),
            weights: EnsembleWeights::default(),
            metrics: MetricsCollector::new(),
            feature_importance: Arc::new(RwLock::new(HashMap::new())),
            anomaly_threshold: 0.7,
        })
    }
    
    /// Create with custom configuration
    pub fn with_config(
        device: Device,
        transformer_config: TransformerConfig,
        weights: EnsembleWeights,
    ) -> Result<Self> {
        let transformer = TransformerWhaleDetector::new(transformer_config, device.clone())?;
        
        Ok(Self {
            device,
            transformer: Arc::new(transformer),
            weights,
            metrics: MetricsCollector::new(),
            feature_importance: Arc::new(RwLock::new(HashMap::new())),
            anomaly_threshold: 0.7,
        })
    }
    
    /// Make a prediction on market data
    pub fn predict(&self, features: &Tensor) -> Result<PredictionResult> {
        let timer = InferenceTimer::start();
        let mut model_predictions = HashMap::new();
        
        // Transformer prediction
        let transformer_prob = self.transformer.predict(features)?;
        model_predictions.insert("transformer".to_string(), transformer_prob);
        
        // Calculate anomaly score based on feature statistics
        let anomaly_score = self.calculate_anomaly_score(features)?;
        model_predictions.insert("anomaly".to_string(), anomaly_score);
        
        // Weighted ensemble prediction
        let whale_probability = self.calculate_ensemble_prediction(&model_predictions);
        
        // Calculate confidence based on model agreement
        let confidence = self.calculate_confidence(&model_predictions);
        
        // Determine threat level
        let threat_level = self.calculate_threat_level(whale_probability, anomaly_score);
        
        // Get interpretability information
        let interpretability = self.generate_interpretability_info(features, &model_predictions)?;
        
        let inference_time_us = timer.stop();
        
        // Check performance constraint
        if inference_time_us > 500 {
            tracing::warn!(
                "Ensemble inference time {}μs exceeds 500μs target",
                inference_time_us
            );
        }
        
        Ok(PredictionResult {
            whale_probability,
            model_predictions,
            confidence,
            threat_level,
            inference_time_us,
            interpretability,
        })
    }
    
    /// Calculate anomaly score based on statistical analysis
    fn calculate_anomaly_score(&self, features: &Tensor) -> Result<f32> {
        // Simple anomaly detection based on statistical outliers
        // In production, this would use a trained isolation forest or similar
        
        let features_vec = features.flatten_all()?.to_vec1::<f32>()?;
        
        // Calculate z-scores for each feature
        let mean = features_vec.iter().sum::<f32>() / features_vec.len() as f32;
        let variance = features_vec
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / features_vec.len() as f32;
        let std_dev = variance.sqrt();
        
        // Count features that are outliers (>3 std devs)
        let outlier_count = features_vec
            .iter()
            .filter(|&&x| ((x - mean) / std_dev).abs() > 3.0)
            .count();
        
        let outlier_ratio = outlier_count as f32 / features_vec.len() as f32;
        
        // Convert to anomaly score (0-1)
        Ok((outlier_ratio * 2.0).min(1.0))
    }
    
    /// Calculate weighted ensemble prediction
    fn calculate_ensemble_prediction(&self, predictions: &HashMap<String, f32>) -> f32 {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        
        if let Some(&transformer_pred) = predictions.get("transformer") {
            weighted_sum += transformer_pred * self.weights.transformer;
            total_weight += self.weights.transformer;
        }
        
        if let Some(&anomaly_pred) = predictions.get("anomaly") {
            // Anomaly score contributes to whale probability
            let anomaly_contribution = if anomaly_pred > self.anomaly_threshold {
                anomaly_pred
            } else {
                anomaly_pred * 0.5  // Reduce impact of low anomaly scores
            };
            weighted_sum += anomaly_contribution * self.weights.anomaly;
            total_weight += self.weights.anomaly;
        }
        
        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }
    
    /// Calculate confidence based on model agreement
    fn calculate_confidence(&self, predictions: &HashMap<String, f32>) -> f32 {
        if predictions.len() < 2 {
            return 0.5;  // Default confidence when only one model
        }
        
        let values: Vec<f32> = predictions.values().copied().collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        
        // Calculate standard deviation
        let variance = values
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();
        
        // Higher agreement (lower std dev) means higher confidence
        // Map std dev to confidence score
        1.0 - (std_dev * 2.0).min(1.0)
    }
    
    /// Calculate threat level based on probability and anomaly
    fn calculate_threat_level(&self, whale_probability: f32, anomaly_score: f32) -> u8 {
        let combined_score = whale_probability * 0.7 + anomaly_score * 0.3;
        
        match combined_score {
            x if x >= 0.9 => 5,  // Critical threat
            x if x >= 0.7 => 4,  // High threat
            x if x >= 0.5 => 3,  // Medium threat
            x if x >= 0.3 => 2,  // Low threat
            _ => 1,              // Minimal threat
        }
    }
    
    /// Generate interpretability information
    fn generate_interpretability_info(
        &self,
        features: &Tensor,
        predictions: &HashMap<String, f32>,
    ) -> Result<InterpretabilityInfo> {
        // Get feature values
        let feature_values = features.flatten_all()?.to_vec1::<f32>()?;
        
        // Feature names (should match feature extraction)
        let feature_names = vec![
            "price", "volume", "sma_20", "ema_20", "rsi_14",
            "bb_position", "macd", "vwap", "volume_sma_20", "volume_ratio",
            "spread", "relative_spread", "price_change_1m", "price_change_5m",
            "volatility", "skewness", "kurtosis", "bid", "ask",
        ];
        
        // Calculate feature importance (simplified - in production use SHAP or similar)
        let mut feature_importance = HashMap::new();
        let mut top_features = Vec::new();
        
        for (i, name) in feature_names.iter().enumerate() {
            if i < feature_values.len() {
                let importance = (feature_values[i].abs() / 100.0).min(1.0);
                feature_importance.insert(name.to_string(), importance);
                top_features.push((name.to_string(), importance));
            }
        }
        
        // Sort by importance
        top_features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        top_features.truncate(5);  // Top 5 features
        
        // Get anomaly score
        let anomaly_score = predictions.get("anomaly").copied().unwrap_or(0.0);
        
        Ok(InterpretabilityInfo {
            top_features,
            attention_weights: None,  // Would be extracted from transformer
            feature_importance,
            anomaly_score,
        })
    }
    
    /// Load model weights
    pub fn load_weights(&self, path: &str) -> Result<()> {
        self.transformer.load_weights(path)?;
        Ok(())
    }
    
    /// Save model weights
    pub fn save_weights(&self, path: &str) -> Result<()> {
        self.transformer.save_weights(path)?;
        Ok(())
    }
    
    /// Get performance metrics
    pub fn get_metrics(&self) -> crate::metrics::PerformanceMetrics {
        self.metrics.get_metrics()
    }
    
    /// Update metrics with actual outcome
    pub fn update_metrics(&self, prediction: &PredictionResult, actual_whale: bool) -> Result<()> {
        let predicted_whale = prediction.whale_probability >= 0.5;
        self.metrics.record_prediction(
            predicted_whale,
            actual_whale,
            prediction.inference_time_us,
        )?;
        Ok(())
    }
}

/// Batch prediction for multiple samples
pub fn batch_predict(
    predictor: &EnsemblePredictor,
    features_batch: &Tensor,
) -> Result<Vec<PredictionResult>> {
    let batch_size = features_batch.dim(0)?;
    let mut results = Vec::with_capacity(batch_size);
    
    for i in 0..batch_size {
        let features = features_batch.get(i)?;
        let result = predictor.predict(&features)?;
        results.push(result);
    }
    
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_ensemble_creation() {
        let device = Device::Cpu;
        let ensemble = EnsemblePredictor::new(device);
        assert!(ensemble.is_ok());
    }
    
    #[test]
    fn test_prediction() {
        let device = Device::Cpu;
        let ensemble = EnsemblePredictor::new(device.clone()).unwrap();
        
        // Create dummy features: seq_len=10, features=19
        let features = Tensor::randn(0f32, 1f32, (1, 10, 19), &device).unwrap();
        
        let result = ensemble.predict(&features);
        assert!(result.is_ok());
        
        let prediction = result.unwrap();
        assert!(prediction.whale_probability >= 0.0 && prediction.whale_probability <= 1.0);
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        assert!(prediction.threat_level >= 1 && prediction.threat_level <= 5);
    }
    
    #[test]
    fn test_threat_level_calculation() {
        let device = Device::Cpu;
        let ensemble = EnsemblePredictor::new(device).unwrap();
        
        assert_eq!(ensemble.calculate_threat_level(0.95, 0.9), 5);
        assert_eq!(ensemble.calculate_threat_level(0.8, 0.6), 4);
        assert_eq!(ensemble.calculate_threat_level(0.6, 0.4), 3);
        assert_eq!(ensemble.calculate_threat_level(0.4, 0.2), 2);
        assert_eq!(ensemble.calculate_threat_level(0.1, 0.1), 1);
    }
}