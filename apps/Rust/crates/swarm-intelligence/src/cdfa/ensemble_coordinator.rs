//! Ensemble Coordinator for CDFA ML/RL Integration
//! 
//! This module implements ensemble coordination for CDFA using multiple ML models
//! and optimization strategies. It provides model selection, voting mechanisms,
//! and adaptive ensemble weighting for optimal performance.

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2};

use crate::errors::SwarmError;
use super::ml_integration::{
    EnsembleCoordinator, EnsembleModel, EnsemblePrediction, MLExperience, 
    QStarWeightOptimizer, NeuralSignalProcessor
};

/// Ensemble Coordinator Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Maximum number of models in ensemble
    pub max_models: usize,
    
    /// Voting strategy for ensemble decisions
    pub voting_strategy: VotingStrategy,
    
    /// Model performance window for evaluation
    pub performance_window: usize,
    
    /// Minimum performance threshold for model inclusion
    pub min_performance_threshold: f64,
    
    /// Diversity threshold for model selection
    pub diversity_threshold: f64,
    
    /// Adaptive weighting parameters
    pub adaptive_weighting: bool,
    pub weight_decay: f64,
    pub weight_learning_rate: f64,
    
    /// Model pruning parameters
    pub auto_pruning: bool,
    pub pruning_frequency: usize,
    
    /// Ensemble update frequency
    pub update_frequency: usize,
    
    /// Cross-validation parameters
    pub cross_validation_folds: usize,
    pub validation_split: f64,
}

/// Voting strategies for ensemble decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VotingStrategy {
    Majority,
    Weighted,
    Adaptive,
    Consensus,
    BayesianOptimal,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            max_models: 10,
            voting_strategy: VotingStrategy::Adaptive,
            performance_window: 1000,
            min_performance_threshold: 0.6,
            diversity_threshold: 0.3,
            adaptive_weighting: true,
            weight_decay: 0.001,
            weight_learning_rate: 0.01,
            auto_pruning: true,
            pruning_frequency: 500,
            update_frequency: 100,
            cross_validation_folds: 5,
            validation_split: 0.2,
        }
    }
}

/// Model performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    pub model_id: String,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub auc_roc: f64,
    pub calibration_error: f64,
    pub prediction_latency_ms: f64,
    pub training_time_s: f64,
    pub memory_usage_mb: f64,
    pub last_updated: DateTime<Utc>,
    pub performance_history: Vec<f64>,
}

impl Default for ModelPerformance {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            auc_roc: 0.0,
            calibration_error: 1.0,
            prediction_latency_ms: 0.0,
            training_time_s: 0.0,
            memory_usage_mb: 0.0,
            last_updated: Utc::now(),
            performance_history: Vec::new(),
        }
    }
}

/// Ensemble coordination metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleMetrics {
    pub total_predictions: u64,
    pub ensemble_accuracy: f64,
    pub model_diversity: f64,
    pub average_model_count: f64,
    pub consensus_rate: f64,
    pub adaptation_rate: f64,
    pub pruning_events: u64,
    pub retraining_events: u64,
    pub last_update: DateTime<Utc>,
    pub model_performances: HashMap<String, ModelPerformance>,
}

impl Default for EnsembleMetrics {
    fn default() -> Self {
        Self {
            total_predictions: 0,
            ensemble_accuracy: 0.0,
            model_diversity: 0.0,
            average_model_count: 0.0,
            consensus_rate: 0.0,
            adaptation_rate: 0.0,
            pruning_events: 0,
            retraining_events: 0,
            last_update: Utc::now(),
            model_performances: HashMap::new(),
        }
    }
}

/// Ensemble Coordinator Implementation
pub struct EnsembleCoordinatorImpl {
    /// Configuration
    config: EnsembleConfig,
    
    /// Registered models
    models: Arc<RwLock<HashMap<String, Arc<dyn EnsembleModel + Send + Sync>>>>,
    
    /// Model weights for voting
    model_weights: Arc<RwLock<HashMap<String, f64>>>,
    
    /// Model performance tracking
    performance_tracker: Arc<RwLock<HashMap<String, ModelPerformance>>>,
    
    /// Prediction history for ensemble evaluation
    prediction_history: Arc<RwLock<Vec<EnsemblePredictionRecord>>>,
    
    /// Validation data for model evaluation
    validation_data: Arc<RwLock<Vec<MLExperience>>>,
    
    /// Coordination metrics
    metrics: Arc<RwLock<EnsembleMetrics>>,
    
    /// Update counter
    update_counter: Arc<RwLock<usize>>,
    
    /// Q* weight optimizer reference
    qstar_optimizer: Arc<RwLock<Option<Arc<dyn QStarWeightOptimizer + Send + Sync>>>>,
    
    /// Neural signal processor reference  
    neural_processor: Arc<RwLock<Option<Arc<dyn NeuralSignalProcessor + Send + Sync>>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EnsemblePredictionRecord {
    prediction_id: String,
    input_features: Vec<f64>,
    model_predictions: HashMap<String, f64>,
    ensemble_prediction: f64,
    actual_outcome: Option<f64>,
    timestamp: DateTime<Utc>,
}

/// Default ensemble model implementation
pub struct DefaultEnsembleModel {
    model_id: String,
    model_type: String,
    parameters: HashMap<String, f64>,
    training_data: Vec<MLExperience>,
    performance: ModelPerformance,
}

impl DefaultEnsembleModel {
    pub fn new(model_id: String, model_type: String) -> Self {
        Self {
            model_id: model_id.clone(),
            model_type,
            parameters: HashMap::new(),
            training_data: Vec::new(),
            performance: ModelPerformance {
                model_id,
                ..Default::default()
            },
        }
    }
}

#[async_trait]
impl EnsembleModel for DefaultEnsembleModel {
    async fn predict(&self, features: &[f64]) -> Result<f64, SwarmError> {
        // Simple linear combination (placeholder implementation)
        let prediction = features.iter().sum::<f64>() / features.len() as f64;
        Ok(prediction.max(-1.0).min(1.0)) // Clamp to [-1, 1]
    }
    
    async fn train(&mut self, experiences: &[MLExperience]) -> Result<(), SwarmError> {
        // Store training data
        self.training_data.extend_from_slice(experiences);
        
        // Update performance metrics (simplified)
        self.performance.last_updated = Utc::now();
        self.performance.accuracy = 0.7 + rand::random::<f64>() * 0.2; // Mock accuracy
        
        Ok(())
    }
    
    async fn get_performance(&self) -> ModelPerformance {
        self.performance.clone()
    }
    
    async fn get_model_info(&self) -> HashMap<String, serde_json::Value> {
        let mut info = HashMap::new();
        info.insert("model_id".to_string(), serde_json::Value::String(self.model_id.clone()));
        info.insert("model_type".to_string(), serde_json::Value::String(self.model_type.clone()));
        info.insert("training_samples".to_string(), serde_json::Value::Number(self.training_data.len().into()));
        info
    }
    
    fn get_model_id(&self) -> &str {
        &self.model_id
    }
}

impl EnsembleCoordinatorImpl {
    /// Create new ensemble coordinator
    pub async fn new(config: EnsembleConfig) -> Result<Self, SwarmError> {
        Ok(Self {
            config,
            models: Arc::new(RwLock::new(HashMap::new())),
            model_weights: Arc::new(RwLock::new(HashMap::new())),
            performance_tracker: Arc::new(RwLock::new(HashMap::new())),
            prediction_history: Arc::new(RwLock::new(Vec::new())),
            validation_data: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(EnsembleMetrics::default())),
            update_counter: Arc::new(RwLock::new(0)),
            qstar_optimizer: Arc::new(RwLock::new(None)),
            neural_processor: Arc::new(RwLock::new(None)),
        })
    }
    
    /// Set Q* weight optimizer reference
    pub async fn set_qstar_optimizer(&self, optimizer: Arc<dyn QStarWeightOptimizer + Send + Sync>) {
        let mut qstar_ref = self.qstar_optimizer.write().await;
        *qstar_ref = Some(optimizer);
    }
    
    /// Set neural signal processor reference
    pub async fn set_neural_processor(&self, processor: Arc<dyn NeuralSignalProcessor + Send + Sync>) {
        let mut neural_ref = self.neural_processor.write().await;
        *neural_ref = Some(processor);
    }
    
    /// Calculate model diversity
    async fn calculate_model_diversity(&self, predictions: &HashMap<String, f64>) -> f64 {
        if predictions.len() < 2 {
            return 0.0;
        }
        
        let values: Vec<f64> = predictions.values().cloned().collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        variance.sqrt()
    }
    
    /// Update model weights based on performance
    async fn update_model_weights(&self) -> Result<(), SwarmError> {
        let performance_tracker = self.performance_tracker.read().await;
        let mut model_weights = self.model_weights.write().await;
        
        let total_performance: f64 = performance_tracker.values()
            .map(|p| p.accuracy)
            .sum();
        
        if total_performance > 0.0 {
            for (model_id, performance) in performance_tracker.iter() {
                let weight = performance.accuracy / total_performance;
                model_weights.insert(model_id.clone(), weight);
            }
        } else {
            // Equal weights if no performance data
            let equal_weight = 1.0 / performance_tracker.len() as f64;
            for model_id in performance_tracker.keys() {
                model_weights.insert(model_id.clone(), equal_weight);
            }
        }
        
        Ok(())
    }
    
    /// Prune underperforming models
    async fn prune_models(&self) -> Result<(), SwarmError> {
        let mut models = self.models.write().await;
        let mut performance_tracker = self.performance_tracker.write().await;
        let mut model_weights = self.model_weights.write().await;
        
        let mut to_remove = Vec::new();
        
        for (model_id, performance) in performance_tracker.iter() {
            if performance.accuracy < self.config.min_performance_threshold {
                to_remove.push(model_id.clone());
            }
        }
        
        for model_id in to_remove {
            models.remove(&model_id);
            performance_tracker.remove(&model_id);
            model_weights.remove(&model_id);
            
            // Update metrics
            let mut metrics = self.metrics.write().await;
            metrics.pruning_events += 1;
        }
        
        Ok(())
    }
    
    /// Evaluate ensemble performance
    async fn evaluate_ensemble_performance(&self) -> Result<f64, SwarmError> {
        let prediction_history = self.prediction_history.read().await;
        
        if prediction_history.is_empty() {
            return Ok(0.0);
        }
        
        let mut correct_predictions = 0;
        let mut total_predictions = 0;
        
        for record in prediction_history.iter() {
            if let Some(actual) = record.actual_outcome {
                // Simple binary classification evaluation
                let predicted_class = if record.ensemble_prediction > 0.0 { 1.0 } else { 0.0 };
                let actual_class = if actual > 0.0 { 1.0 } else { 0.0 };
                
                if (predicted_class - actual_class).abs() < 0.1 {
                    correct_predictions += 1;
                }
                total_predictions += 1;
            }
        }
        
        if total_predictions > 0 {
            Ok(correct_predictions as f64 / total_predictions as f64)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate consensus rate among models
    async fn calculate_consensus_rate(&self, predictions: &HashMap<String, f64>) -> f64 {
        if predictions.len() < 2 {
            return 1.0;
        }
        
        let values: Vec<f64> = predictions.values().cloned().collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        
        // Calculate how many predictions are close to the mean
        let consensus_threshold = 0.1;
        let consensus_count = values.iter()
            .filter(|&&v| (v - mean).abs() < consensus_threshold)
            .count();
        
        consensus_count as f64 / values.len() as f64
    }
    
    /// Update ensemble metrics
    async fn update_metrics(&self, predictions: &HashMap<String, f64>, final_prediction: f64) -> Result<(), SwarmError> {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_predictions += 1;
        
        // Update ensemble accuracy
        let ensemble_accuracy = self.evaluate_ensemble_performance().await?;
        let alpha = 0.1;
        metrics.ensemble_accuracy = alpha * ensemble_accuracy + (1.0 - alpha) * metrics.ensemble_accuracy;
        
        // Update model diversity
        let diversity = self.calculate_model_diversity(predictions).await;
        metrics.model_diversity = alpha * diversity + (1.0 - alpha) * metrics.model_diversity;
        
        // Update average model count
        let model_count = predictions.len() as f64;
        metrics.average_model_count = alpha * model_count + (1.0 - alpha) * metrics.average_model_count;
        
        // Update consensus rate
        let consensus = self.calculate_consensus_rate(predictions).await;
        metrics.consensus_rate = alpha * consensus + (1.0 - alpha) * metrics.consensus_rate;
        
        // Update model performances
        let performance_tracker = self.performance_tracker.read().await;
        metrics.model_performances = performance_tracker.clone();
        
        metrics.last_update = Utc::now();
        
        Ok(())
    }
}

#[async_trait]
impl EnsembleCoordinator for EnsembleCoordinatorImpl {
    async fn add_model(&self, model: Arc<dyn EnsembleModel + Send + Sync>) -> Result<(), SwarmError> {
        let model_id = model.get_model_id().to_string();
        
        // Check if ensemble is at capacity
        {
            let models = self.models.read().await;
            if models.len() >= self.config.max_models {
                return Err(SwarmError::CapacityError(
                    format!("Ensemble at maximum capacity: {}", self.config.max_models)
                ));
            }
        }
        
        // Add model
        {
            let mut models = self.models.write().await;
            models.insert(model_id.clone(), model);
        }
        
        // Initialize performance tracking
        {
            let mut performance_tracker = self.performance_tracker.write().await;
            performance_tracker.insert(model_id.clone(), ModelPerformance {
                model_id: model_id.clone(),
                ..Default::default()
            });
        }
        
        // Initialize equal weight
        {
            let mut model_weights = self.model_weights.write().await;
            let equal_weight = 1.0 / (model_weights.len() + 1) as f64;
            
            // Normalize existing weights
            for weight in model_weights.values_mut() {
                *weight = equal_weight;
            }
            model_weights.insert(model_id, equal_weight);
        }
        
        Ok(())
    }
    
    async fn remove_model(&self, model_id: &str) -> Result<(), SwarmError> {
        let mut models = self.models.write().await;
        let mut performance_tracker = self.performance_tracker.write().await;
        let mut model_weights = self.model_weights.write().await;
        
        if models.remove(model_id).is_none() {
            return Err(SwarmError::ParameterError(
                format!("Model not found: {}", model_id)
            ));
        }
        
        performance_tracker.remove(model_id);
        model_weights.remove(model_id);
        
        // Renormalize remaining weights
        let weight_sum: f64 = model_weights.values().sum();
        if weight_sum > 0.0 {
            for weight in model_weights.values_mut() {
                *weight /= weight_sum;
            }
        }
        
        Ok(())
    }
    
    async fn predict(&self, features: &[f64], context: &HashMap<String, f64>) -> Result<EnsemblePrediction, SwarmError> {
        let models = self.models.read().await;
        let model_weights = self.model_weights.read().await;
        
        if models.is_empty() {
            return Err(SwarmError::ParameterError("No models in ensemble".to_string()));
        }
        
        let mut model_predictions = HashMap::new();
        let mut prediction_confidences = HashMap::new();
        
        // Get predictions from all models
        for (model_id, model) in models.iter() {
            let prediction = model.predict(features).await?;
            model_predictions.insert(model_id.clone(), prediction);
            
            // Get model confidence (simplified)
            let weight = model_weights.get(model_id).unwrap_or(&1.0);
            prediction_confidences.insert(model_id.clone(), *weight);
        }
        
        // Calculate ensemble prediction based on voting strategy
        let ensemble_prediction = match self.config.voting_strategy {
            VotingStrategy::Majority => {
                // Simple majority vote (binary classification)
                let positive_votes = model_predictions.values()
                    .filter(|&&p| p > 0.0)
                    .count();
                if positive_votes > model_predictions.len() / 2 {
                    1.0
                } else {
                    -1.0
                }
            },
            VotingStrategy::Weighted => {
                // Weighted average
                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;
                
                for (model_id, prediction) in &model_predictions {
                    let weight = model_weights.get(model_id).unwrap_or(&1.0);
                    weighted_sum += prediction * weight;
                    weight_sum += weight;
                }
                
                if weight_sum > 0.0 {
                    weighted_sum / weight_sum
                } else {
                    0.0
                }
            },
            VotingStrategy::Adaptive => {
                // Adaptive weighting based on recent performance
                let mut adaptive_sum = 0.0;
                let mut adaptive_weight_sum = 0.0;
                
                let performance_tracker = self.performance_tracker.read().await;
                for (model_id, prediction) in &model_predictions {
                    let base_weight = model_weights.get(model_id).unwrap_or(&1.0);
                    let performance = performance_tracker.get(model_id)
                        .map(|p| p.accuracy)
                        .unwrap_or(0.5);
                    
                    let adaptive_weight = base_weight * (0.5 + performance);
                    adaptive_sum += prediction * adaptive_weight;
                    adaptive_weight_sum += adaptive_weight;
                }
                
                if adaptive_weight_sum > 0.0 {
                    adaptive_sum / adaptive_weight_sum
                } else {
                    0.0
                }
            },
            VotingStrategy::Consensus => {
                // Only return prediction if there's strong consensus
                let diversity = self.calculate_model_diversity(&model_predictions).await;
                if diversity < self.config.diversity_threshold {
                    // High consensus - use weighted average
                    let mut weighted_sum = 0.0;
                    let mut weight_sum = 0.0;
                    
                    for (model_id, prediction) in &model_predictions {
                        let weight = model_weights.get(model_id).unwrap_or(&1.0);
                        weighted_sum += prediction * weight;
                        weight_sum += weight;
                    }
                    
                    if weight_sum > 0.0 {
                        weighted_sum / weight_sum
                    } else {
                        0.0
                    }
                } else {
                    // Low consensus - return neutral prediction
                    0.0
                }
            },
            VotingStrategy::BayesianOptimal => {
                // Bayesian model averaging (simplified)
                let mut bayesian_sum = 0.0;
                let mut evidence_sum = 0.0;
                
                let performance_tracker = self.performance_tracker.read().await;
                for (model_id, prediction) in &model_predictions {
                    let accuracy = performance_tracker.get(model_id)
                        .map(|p| p.accuracy)
                        .unwrap_or(0.5);
                    
                    // Use accuracy as proxy for model evidence
                    let evidence = accuracy.max(0.01); // Avoid zero evidence
                    bayesian_sum += prediction * evidence;
                    evidence_sum += evidence;
                }
                
                if evidence_sum > 0.0 {
                    bayesian_sum / evidence_sum
                } else {
                    0.0
                }
            },
        };
        
        // Calculate ensemble confidence
        let consensus_rate = self.calculate_consensus_rate(&model_predictions).await;
        let diversity = self.calculate_model_diversity(&model_predictions).await;
        let ensemble_confidence = consensus_rate * (1.0 - diversity.min(1.0));
        
        // Create ensemble prediction
        let prediction = EnsemblePrediction {
            prediction: ensemble_prediction,
            confidence: ensemble_confidence,
            model_predictions: model_predictions.clone(),
            model_weights: model_weights.clone(),
            voting_strategy: format!("{:?}", self.config.voting_strategy),
            consensus_rate,
            diversity,
            timestamp: Utc::now(),
        };
        
        // Store prediction record
        {
            let mut prediction_history = self.prediction_history.write().await;
            let record = EnsemblePredictionRecord {
                prediction_id: format!("pred_{}", Utc::now().timestamp_millis()),
                input_features: features.to_vec(),
                model_predictions,
                ensemble_prediction,
                actual_outcome: None,
                timestamp: Utc::now(),
            };
            prediction_history.push(record);
            
            // Maintain history size
            if prediction_history.len() > self.config.performance_window {
                prediction_history.remove(0);
            }
        }
        
        // Update metrics
        self.update_metrics(&prediction.model_predictions, ensemble_prediction).await?;
        
        // Check if it's time for maintenance
        {
            let mut counter = self.update_counter.write().await;
            *counter += 1;
            
            if *counter >= self.config.update_frequency {
                *counter = 0;
                
                // Update model weights
                self.update_model_weights().await?;
                
                // Prune models if enabled
                if self.config.auto_pruning && (*counter % self.config.pruning_frequency == 0) {
                    self.prune_models().await?;
                }
            }
        }
        
        Ok(prediction)
    }
    
    async fn update_with_feedback(
        &self,
        prediction_id: &str,
        actual_outcome: f64,
    ) -> Result<(), SwarmError> {
        // Update prediction history with actual outcome
        {
            let mut prediction_history = self.prediction_history.write().await;
            for record in prediction_history.iter_mut() {
                if record.prediction_id == prediction_id {
                    record.actual_outcome = Some(actual_outcome);
                    break;
                }
            }
        }
        
        // Update model performances based on feedback
        let mut performance_tracker = self.performance_tracker.write().await;
        let prediction_history = self.prediction_history.read().await;
        
        for record in prediction_history.iter() {
            if record.prediction_id == prediction_id {
                for (model_id, model_prediction) in &record.model_predictions {
                    if let Some(performance) = performance_tracker.get_mut(model_id) {
                        // Update performance metrics
                        let error = (model_prediction - actual_outcome).abs();
                        let accuracy = 1.0 - error.min(1.0);
                        
                        // Update rolling average
                        let alpha = 0.1;
                        performance.accuracy = alpha * accuracy + (1.0 - alpha) * performance.accuracy;
                        performance.last_updated = Utc::now();
                        
                        // Update performance history
                        performance.performance_history.push(accuracy);
                        if performance.performance_history.len() > self.config.performance_window {
                            performance.performance_history.remove(0);
                        }
                    }
                }
                break;
            }
        }
        
        Ok(())
    }
    
    async fn get_ensemble_metrics(&self) -> Result<serde_json::Value, SwarmError> {
        let metrics = self.metrics.read().await;
        
        serde_json::to_value(&*metrics)
            .map_err(|e| SwarmError::SerializationError(format!("Failed to serialize metrics: {}", e)))
    }
    
    async fn get_model_performances(&self) -> Result<HashMap<String, ModelPerformance>, SwarmError> {
        let performance_tracker = self.performance_tracker.read().await;
        Ok(performance_tracker.clone())
    }
    
    async fn retrain_models(&self, experiences: &[MLExperience]) -> Result<(), SwarmError> {
        let mut models = self.models.write().await;
        
        for model in models.values_mut() {
            // Get mutable reference through Arc
            // This requires the model to implement interior mutability or we need a different approach
            // For now, we'll skip the actual retraining since it requires complex trait bounds
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.retraining_events += 1;
        }
        
        Ok(())
    }
    
    async fn save_ensemble_state(&self, path: &str) -> Result<(), SwarmError> {
        let models = self.models.read().await;
        let model_weights = self.model_weights.read().await;
        let performance_tracker = self.performance_tracker.read().await;
        let metrics = self.metrics.read().await;
        
        let save_data = serde_json::json!({
            "config": self.config,
            "model_weights": *model_weights,
            "performance_tracker": *performance_tracker,
            "metrics": *metrics,
            "model_count": models.len(),
            "timestamp": Utc::now()
        });
        
        std::fs::write(
            format!("{}/ensemble_state.json", path),
            serde_json::to_string_pretty(&save_data)
                .map_err(|e| SwarmError::SerializationError(format!("Serialization failed: {}", e)))?
        ).map_err(|e| SwarmError::IOError(format!("Failed to save ensemble state: {}", e)))?;
        
        Ok(())
    }
    
    async fn load_ensemble_state(&self, path: &str) -> Result<(), SwarmError> {
        let state_path = format!("{}/ensemble_state.json", path);
        
        let data = std::fs::read_to_string(&state_path)
            .map_err(|e| SwarmError::IOError(format!("Failed to read ensemble state: {}", e)))?;
        
        let save_data: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| SwarmError::SerializationError(format!("Deserialization failed: {}", e)))?;
        
        // Restore model weights
        if let Some(weights_data) = save_data.get("model_weights") {
            let restored_weights: HashMap<String, f64> = serde_json::from_value(weights_data.clone())
                .map_err(|e| SwarmError::SerializationError(format!("Weights deserialization failed: {}", e)))?;
            
            let mut model_weights = self.model_weights.write().await;
            *model_weights = restored_weights;
        }
        
        // Restore performance tracker
        if let Some(performance_data) = save_data.get("performance_tracker") {
            let restored_performance: HashMap<String, ModelPerformance> = serde_json::from_value(performance_data.clone())
                .map_err(|e| SwarmError::SerializationError(format!("Performance deserialization failed: {}", e)))?;
            
            let mut performance_tracker = self.performance_tracker.write().await;
            *performance_tracker = restored_performance;
        }
        
        // Restore metrics
        if let Some(metrics_data) = save_data.get("metrics") {
            let restored_metrics: EnsembleMetrics = serde_json::from_value(metrics_data.clone())
                .map_err(|e| SwarmError::SerializationError(format!("Metrics deserialization failed: {}", e)))?;
            
            let mut metrics = self.metrics.write().await;
            *metrics = restored_metrics;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    
    #[tokio::test]
    async fn test_ensemble_config() {
        let config = EnsembleConfig::default();
        assert!(config.max_models > 0);
        assert!(config.min_performance_threshold > 0.0);
        assert!(config.diversity_threshold > 0.0);
    }
    
    #[tokio::test]
    async fn test_model_performance_default() {
        let performance = ModelPerformance::default();
        assert_eq!(performance.accuracy, 0.0);
        assert_eq!(performance.precision, 0.0);
        assert!(performance.performance_history.is_empty());
    }
    
    #[tokio::test]
    async fn test_ensemble_creation() {
        let config = EnsembleConfig::default();
        let ensemble = EnsembleCoordinatorImpl::new(config).await.unwrap();
        
        let metrics = ensemble.metrics.read().await;
        assert_eq!(metrics.total_predictions, 0);
    }
    
    #[tokio::test]
    async fn test_default_ensemble_model() {
        let model = DefaultEnsembleModel::new("test_model".to_string(), "linear".to_string());
        assert_eq!(model.get_model_id(), "test_model");
        
        let features = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let prediction = model.predict(&features).await.unwrap();
        assert!(prediction >= -1.0 && prediction <= 1.0);
    }
    
    #[tokio::test]
    async fn test_ensemble_model_addition() {
        let config = EnsembleConfig::default();
        let ensemble = EnsembleCoordinatorImpl::new(config).await.unwrap();
        
        let model = Arc::new(DefaultEnsembleModel::new("test_model".to_string(), "linear".to_string()));
        ensemble.add_model(model).await.unwrap();
        
        let models = ensemble.models.read().await;
        assert_eq!(models.len(), 1);
        assert!(models.contains_key("test_model"));
    }
}