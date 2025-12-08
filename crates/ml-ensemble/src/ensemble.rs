use crate::{
    calibration::ConfidenceCalibrator,
    features::FeatureEngineering,
    market_detector::MarketConditionDetector,
    model_selector::ModelSelector,
    types::*,
    weights::WeightManager,
};
use anyhow::Result;
use ats_core::types::MarketData;
use metrics::{counter, histogram};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::Instant,
};
use tokio::sync::RwLock as AsyncRwLock;
use tracing::{debug, info, warn};

/// Unified ML ensemble predictor
pub struct EnsemblePredictor {
    /// Configuration
    config: EnsembleConfig,
    
    /// Individual models
    models: HashMap<ModelType, Arc<dyn ModelPredictor>>,
    
    /// Feature engineering
    feature_engine: Arc<FeatureEngineering>,
    
    /// Market condition detector
    market_detector: Arc<MarketConditionDetector>,
    
    /// Model selector
    model_selector: Arc<ModelSelector>,
    
    /// Weight manager
    weight_manager: Arc<AsyncRwLock<WeightManager>>,
    
    /// Confidence calibrator
    calibrator: Arc<AsyncRwLock<ConfidenceCalibrator>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<HashMap<ModelType, ModelMetrics>>>,
}

/// Trait for individual model predictors
pub trait ModelPredictor: Send + Sync {
    /// Get model type
    fn model_type(&self) -> ModelType;
    
    /// Make prediction
    fn predict(&self, features: &Array1<f32>) -> Result<(f64, f64)>; // (prediction, confidence)
    
    /// Batch prediction
    fn predict_batch(&self, features: &Array2<f32>) -> Result<Vec<(f64, f64)>>;
    
    /// Update model (online learning)
    fn update(&mut self, features: &Array1<f32>, target: f64) -> Result<()>;
}

impl EnsemblePredictor {
    /// Create new ensemble predictor
    pub async fn new(config: EnsembleConfig, models: HashMap<ModelType, Arc<dyn ModelPredictor>>) -> Result<Self> {
        info!("Initializing ML ensemble predictor with {} models", models.len());
        
        let feature_engine = Arc::new(FeatureEngineering::new(config.features.clone()));
        let market_detector = Arc::new(MarketConditionDetector::new());
        let model_selector = Arc::new(ModelSelector::new());
        let weight_manager = Arc::new(AsyncRwLock::new(WeightManager::new(config.model_weights.clone())));
        let calibrator = Arc::new(AsyncRwLock::new(ConfidenceCalibrator::new(config.calibration.clone())));
        
        let metrics = Arc::new(RwLock::new(
            models.keys()
                .map(|&model_type| (model_type, ModelMetrics { model_type, ..Default::default() }))
                .collect()
        ));
        
        Ok(Self {
            config,
            models,
            feature_engine,
            market_detector,
            model_selector,
            weight_manager,
            calibrator,
            metrics,
        })
    }
    
    /// Make ensemble prediction
    pub async fn predict(&self, market_data: &MarketData) -> Result<EnsemblePrediction> {
        let start = Instant::now();
        counter!("ensemble.predictions").increment(1);
        
        // Extract features
        let features = self.feature_engine.extract_features(market_data)?;
        debug!("Extracted {} features", features.len());
        
        // Detect market condition
        let market_condition = self.market_detector.detect_condition(market_data)?;
        debug!("Detected market condition: {:?}", market_condition);
        
        // Select models based on market condition
        let selected_models = self.model_selector.select_models(market_condition, &self.models)?;
        debug!("Selected {} models for prediction", selected_models.len());
        
        // Get current weights
        let weights = self.weight_manager.read().await.get_weights(&selected_models);
        
        // Parallel model predictions
        let model_predictions: Vec<ModelPrediction> = selected_models
            .par_iter()
            .filter_map(|model| {
                let model_start = Instant::now();
                
                match model.predict(&features) {
                    Ok((prediction, confidence)) => {
                        let latency_us = model_start.elapsed().as_micros() as f64;
                        
                        histogram!("model.latency", "model" => model.model_type().to_string())
                            .record(latency_us);
                        
                        Some(ModelPrediction {
                            model_type: model.model_type(),
                            prediction,
                            confidence,
                            weight: weights.get(&model.model_type()).copied().unwrap_or(0.0),
                            latency_us,
                        })
                    }
                    Err(e) => {
                        warn!("Model {:?} prediction failed: {}", model.model_type(), e);
                        counter!("model.errors", "model" => model.model_type().to_string())
                            .increment(1);
                        None
                    }
                }
            })
            .collect();
        
        if model_predictions.is_empty() {
            return Err(anyhow::anyhow!("No models produced predictions"));
        }
        
        // Weighted ensemble prediction
        let (ensemble_value, raw_confidence) = self.compute_weighted_prediction(&model_predictions);
        
        // Calibrate confidence
        let calibrated_confidence = self.calibrator.read().await
            .calibrate_confidence(raw_confidence, market_condition)?;
        
        // Extract feature importance
        let feature_importance = self.extract_feature_importance(&model_predictions, &features)?;
        
        let latency_us = start.elapsed().as_micros() as f64;
        histogram!("ensemble.latency").record(latency_us);
        
        // Check latency constraint
        if latency_us > self.config.max_latency_us {
            warn!("Ensemble prediction exceeded latency limit: {:.0}μs > {:.0}μs", 
                  latency_us, self.config.max_latency_us);
        }
        
        // Update metrics
        self.update_metrics(&model_predictions).await;
        
        Ok(EnsemblePrediction {
            value: ensemble_value,
            confidence: calibrated_confidence,
            model_predictions,
            market_condition,
            feature_importance,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            latency_us,
        })
    }
    
    /// Compute weighted ensemble prediction
    fn compute_weighted_prediction(&self, predictions: &[ModelPrediction]) -> (f64, f64) {
        let total_weight: f64 = predictions.iter().map(|p| p.weight).sum();
        
        if total_weight == 0.0 {
            // Fallback to equal weights
            let avg_prediction = predictions.iter().map(|p| p.prediction).sum::<f64>() / predictions.len() as f64;
            let avg_confidence = predictions.iter().map(|p| p.confidence).sum::<f64>() / predictions.len() as f64;
            return (avg_prediction, avg_confidence);
        }
        
        // Weighted average
        let weighted_prediction = predictions.iter()
            .map(|p| p.prediction * p.weight)
            .sum::<f64>() / total_weight;
        
        // Weighted confidence with diversity bonus
        let weighted_confidence = predictions.iter()
            .map(|p| p.confidence * p.weight)
            .sum::<f64>() / total_weight;
        
        // Apply diversity bonus to confidence
        let prediction_std = self.calculate_prediction_std(predictions);
        let diversity_factor = 1.0 - (prediction_std / 0.1).min(1.0); // Lower std = higher confidence
        let adjusted_confidence = weighted_confidence * (0.8 + 0.2 * diversity_factor);
        
        (weighted_prediction, adjusted_confidence.min(1.0))
    }
    
    /// Calculate standard deviation of predictions
    fn calculate_prediction_std(&self, predictions: &[ModelPrediction]) -> f64 {
        if predictions.len() < 2 {
            return 0.0;
        }
        
        let mean = predictions.iter().map(|p| p.prediction).sum::<f64>() / predictions.len() as f64;
        let variance = predictions.iter()
            .map(|p| (p.prediction - mean).powi(2))
            .sum::<f64>() / predictions.len() as f64;
        
        variance.sqrt()
    }
    
    /// Extract feature importance from models
    fn extract_feature_importance(
        &self,
        predictions: &[ModelPrediction],
        features: &Array1<f32>
    ) -> Result<Vec<(String, f64)>> {
        // For now, return top features based on variance
        // In production, we'd extract from tree models and attention weights
        let feature_names = self.feature_engine.get_feature_names();
        let mut importance_scores: Vec<(String, f64)> = Vec::new();
        
        for (i, name) in feature_names.iter().enumerate() {
            if i < features.len() {
                // Simple variance-based importance
                let importance = features[i].abs() as f64;
                importance_scores.push((name.clone(), importance));
            }
        }
        
        // Sort by importance and take top 10
        importance_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        importance_scores.truncate(10);
        
        Ok(importance_scores)
    }
    
    /// Update model metrics
    async fn update_metrics(&self, predictions: &[ModelPrediction]) {
        let mut metrics = self.metrics.write().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        for pred in predictions {
            if let Some(metric) = metrics.get_mut(&pred.model_type) {
                metric.prediction_count += 1;
                metric.avg_latency_us = (metric.avg_latency_us * (metric.prediction_count - 1) as f64 
                    + pred.latency_us) / metric.prediction_count as f64;
                metric.last_update = now;
            }
        }
    }
    
    /// Update ensemble with new observation
    pub async fn update(&mut self, market_data: &MarketData, actual_return: f64) -> Result<()> {
        // Extract features
        let features = self.feature_engine.extract_features(market_data)?;
        
        // Update individual models that support online learning
        for model in self.models.values_mut() {
            if let Ok(model_mut) = Arc::get_mut(model) {
                model_mut.update(&features, actual_return)?;
            }
        }
        
        // Update weight manager with performance feedback
        self.weight_manager.write().await
            .update_performance(actual_return)?;
        
        // Update calibrator
        self.calibrator.write().await
            .update(actual_return)?;
        
        Ok(())
    }
    
    /// Get current model weights
    pub async fn get_weights(&self) -> HashMap<ModelType, f64> {
        self.weight_manager.read().await.get_all_weights()
    }
    
    /// Get performance metrics
    pub fn get_metrics(&self) -> HashMap<ModelType, ModelMetrics> {
        self.metrics.read().unwrap().clone()
    }
}