//! Ensemble coordination for Neural Forge
//! 
//! Provides intelligent ensemble coordination across multiple neural systems
//! Supports model combination, voting strategies, and performance optimization

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug};

use crate::prelude::*;
use crate::integration::{EnsembleConfig, ModelSelectionStrategy, VotingStrategy};

/// Ensemble coordinator
pub struct EnsembleCoordinator {
    config: EnsembleConfig,
    models: HashMap<String, EnsembleModel>,
    performance_tracker: Arc<RwLock<EnsemblePerformance>>,
    strategy_engine: StrategyEngine,
    model_selector: ModelSelector,
}

/// Ensemble model wrapper
#[derive(Debug, Clone)]
pub struct EnsembleModel {
    /// Model identifier
    pub id: String,
    
    /// Model type and source
    pub model_type: ModelSourceType,
    
    /// Model metadata
    pub metadata: ModelMetadata,
    
    /// Performance statistics
    pub performance: ModelPerformanceStats,
    
    /// Current status
    pub status: ModelStatus,
    
    /// Configuration
    pub config: ModelConfig,
}

/// Model source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSourceType {
    CognitionEngine,
    RuvFann,
    ClaudeFlow,
    NeuralForge,
    External(String),
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub architecture: String,
    pub training_date: String,
    pub accuracy: f64,
    pub latency_ms: f64,
    pub memory_mb: f64,
    pub specialization: Vec<String>,
}

/// Model performance statistics
#[derive(Debug, Clone, Default)]
pub struct ModelPerformanceStats {
    pub total_predictions: u64,
    pub successful_predictions: u64,
    pub average_accuracy: f64,
    pub average_latency_us: f64,
    pub error_rate: f64,
    pub confidence_score: f64,
    pub reliability_score: f64,
    pub recent_performance: Vec<PerformancePoint>,
}

/// Performance point
#[derive(Debug, Clone)]
pub struct PerformancePoint {
    pub timestamp: std::time::SystemTime,
    pub accuracy: f64,
    pub latency_us: u64,
    pub confidence: f64,
}

/// Model status
#[derive(Debug, Clone, PartialEq)]
pub enum ModelStatus {
    Active,
    Standby,
    Training,
    Maintenance,
    Failed,
    Disabled,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub weight: f64,
    pub priority: u8,
    pub timeout_ms: u64,
    pub retry_count: usize,
    pub enabled: bool,
    pub constraints: ModelConstraints,
}

/// Model constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConstraints {
    pub max_latency_ms: Option<u64>,
    pub min_accuracy: Option<f64>,
    pub max_memory_mb: Option<f64>,
    pub required_features: Vec<String>,
    pub excluded_patterns: Vec<String>,
}

/// Ensemble performance tracker
#[derive(Debug, Clone, Default)]
pub struct EnsemblePerformance {
    pub total_predictions: u64,
    pub ensemble_accuracy: f64,
    pub individual_accuracies: HashMap<String, f64>,
    pub voting_effectiveness: HashMap<String, f64>,
    pub selection_accuracy: f64,
    pub coordination_overhead_us: u64,
    pub model_utilization: HashMap<String, f64>,
    pub consensus_rate: f64,
}

/// Strategy engine for ensemble coordination
pub struct StrategyEngine {
    voting_strategies: HashMap<String, Box<dyn VotingStrategyTrait + Send + Sync>>,
    selection_strategies: HashMap<String, Box<dyn SelectionStrategyTrait + Send + Sync>>,
    current_voting_strategy: String,
    current_selection_strategy: String,
    adaptation_threshold: f64,
}

/// Model selector for intelligent model choice
pub struct ModelSelector {
    selection_history: Vec<SelectionDecision>,
    performance_predictors: HashMap<String, PerformancePredictor>,
    context_analyzer: ContextAnalyzer,
    learning_enabled: bool,
}

/// Ensemble prediction request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsemblePredictionRequest {
    /// Input data
    pub data: Vec<Vec<f64>>,
    
    /// Prediction options
    pub options: EnsemblePredictionOptions,
    
    /// Context information
    pub context: PredictionContext,
    
    /// Request metadata
    pub metadata: RequestMetadata,
}

/// Ensemble prediction options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsemblePredictionOptions {
    /// Voting strategy to use
    pub voting_strategy: Option<String>,
    
    /// Model selection strategy
    pub selection_strategy: Option<String>,
    
    /// Maximum models to use
    pub max_models: Option<usize>,
    
    /// Minimum consensus threshold
    pub consensus_threshold: Option<f64>,
    
    /// Require uncertainty quantification
    pub require_uncertainty: bool,
    
    /// Performance vs accuracy trade-off
    pub performance_mode: PerformanceMode,
    
    /// Timeout for ensemble prediction
    pub timeout_ms: Option<u64>,
}

/// Performance mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMode {
    HighAccuracy,
    HighSpeed,
    Balanced,
    Conservative,
    Aggressive,
}

/// Prediction context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionContext {
    /// Domain or category
    pub domain: String,
    
    /// Data characteristics
    pub data_characteristics: DataCharacteristics,
    
    /// Quality requirements
    pub quality_requirements: QualityRequirements,
    
    /// Historical performance context
    pub historical_context: Option<HistoricalContext>,
}

/// Data characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCharacteristics {
    pub size: usize,
    pub dimensionality: usize,
    pub noise_level: f64,
    pub complexity: f64,
    pub temporal_patterns: bool,
    pub seasonality: bool,
    pub trend: bool,
}

/// Quality requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRequirements {
    pub min_accuracy: f64,
    pub max_latency_ms: u64,
    pub confidence_level: f64,
    pub reliability_requirement: f64,
}

/// Historical context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalContext {
    pub similar_requests: Vec<String>,
    pub best_performing_models: Vec<String>,
    pub known_difficult_patterns: Vec<String>,
    pub seasonal_adjustments: HashMap<String, f64>,
}

/// Ensemble prediction response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsemblePredictionResponse {
    /// Final ensemble prediction
    pub prediction: Vec<f64>,
    
    /// Individual model predictions
    pub individual_predictions: HashMap<String, Vec<f64>>,
    
    /// Confidence scores
    pub confidence: Vec<f64>,
    
    /// Uncertainty estimates
    pub uncertainty: Option<Vec<f64>>,
    
    /// Ensemble metadata
    pub ensemble_metadata: EnsembleMetadata,
    
    /// Response metadata
    pub metadata: ResponseMetadata,
}

/// Ensemble metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleMetadata {
    /// Models used in ensemble
    pub models_used: Vec<String>,
    
    /// Voting strategy applied
    pub voting_strategy: String,
    
    /// Model weights
    pub model_weights: HashMap<String, f64>,
    
    /// Consensus level achieved
    pub consensus_level: f64,
    
    /// Selection rationale
    pub selection_rationale: String,
    
    /// Performance prediction
    pub predicted_accuracy: f64,
}

/// Request metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetadata {
    pub request_id: String,
    pub timestamp: u64,
    pub priority: u8,
    pub timeout_ms: u64,
}

/// Response metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    pub request_id: String,
    pub processing_time_us: u64,
    pub ensemble_version: String,
    pub coordination_overhead_us: u64,
    pub status: ResponseStatus,
}

/// Response status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseStatus {
    Success,
    PartialSuccess,
    Failed(String),
    Timeout,
    InsufficientConsensus,
    NoModelsAvailable,
}

/// Voting strategy trait
pub trait VotingStrategyTrait {
    fn vote(&self, predictions: &HashMap<String, Vec<f64>>, weights: &HashMap<String, f64>) -> Result<Vec<f64>>;
    fn name(&self) -> &str;
    fn supports_uncertainty(&self) -> bool;
}

/// Selection strategy trait
pub trait SelectionStrategyTrait {
    fn select_models(
        &self,
        available_models: &[EnsembleModel],
        context: &PredictionContext,
        options: &EnsemblePredictionOptions,
    ) -> Result<Vec<String>>;
    fn name(&self) -> &str;
}

/// Selection decision
#[derive(Debug, Clone)]
pub struct SelectionDecision {
    pub timestamp: std::time::SystemTime,
    pub selected_models: Vec<String>,
    pub context: PredictionContext,
    pub actual_performance: Option<f64>,
    pub rationale: String,
}

/// Performance predictor
#[derive(Debug, Clone)]
pub struct PerformancePredictor {
    pub model_id: String,
    pub predictor_type: PredictorType,
    pub accuracy_history: Vec<f64>,
    pub context_performance: HashMap<String, f64>,
    pub last_updated: std::time::SystemTime,
}

/// Predictor types
#[derive(Debug, Clone)]
pub enum PredictorType {
    MovingAverage,
    ExponentialSmoothing,
    LinearRegression,
    NeuralNetwork,
    EnsemblePredictor,
}

/// Context analyzer
#[derive(Debug, Clone)]
pub struct ContextAnalyzer {
    pub pattern_library: HashMap<String, PatternInfo>,
    pub complexity_estimator: ComplexityEstimator,
    pub similarity_threshold: f64,
}

/// Pattern information
#[derive(Debug, Clone)]
pub struct PatternInfo {
    pub pattern_id: String,
    pub description: String,
    pub characteristics: Vec<String>,
    pub best_models: Vec<String>,
    pub difficulty_score: f64,
}

/// Complexity estimator
#[derive(Debug, Clone)]
pub struct ComplexityEstimator {
    pub linear_threshold: f64,
    pub nonlinear_threshold: f64,
    pub chaotic_threshold: f64,
}

impl EnsembleCoordinator {
    /// Create new ensemble coordinator
    pub fn new(config: EnsembleConfig) -> Result<Self> {
        info!("Initializing Ensemble Coordinator");
        
        // Validate configuration
        config.validate()?;
        
        let performance_tracker = Arc::new(RwLock::new(EnsemblePerformance::default()));
        let strategy_engine = StrategyEngine::new(&config)?;
        let model_selector = ModelSelector::new(&config)?;
        
        Ok(Self {
            config,
            models: HashMap::new(),
            performance_tracker,
            strategy_engine,
            model_selector,
        })
    }
    
    /// Register model with ensemble
    pub async fn register_model(&mut self, model: EnsembleModel) -> Result<()> {
        info!("Registering model in ensemble: {}", model.id);
        
        // Validate model
        self.validate_model(&model)?;
        
        // Add to models
        self.models.insert(model.id.clone(), model);
        
        // Update performance tracking
        self.update_model_registry().await;
        
        Ok(())
    }
    
    /// Unregister model from ensemble
    pub async fn unregister_model(&mut self, model_id: &str) -> Result<()> {
        info!("Unregistering model from ensemble: {}", model_id);
        
        match self.models.remove(model_id) {
            Some(_) => {
                self.update_model_registry().await;
                Ok(())
            }
            None => Err(NeuralForgeError::validation(&format!("Model not found: {}", model_id))),
        }
    }
    
    /// Make ensemble prediction
    pub async fn predict(&mut self, request: EnsemblePredictionRequest) -> Result<EnsemblePredictionResponse> {
        let start_time = std::time::Instant::now();
        
        info!("Starting ensemble prediction: {}", request.metadata.request_id);
        
        // Select models for this prediction
        let selected_models = self.model_selector.select_models(
            &self.get_available_models(),
            &request.context,
            &request.options,
        ).await?;
        
        if selected_models.is_empty() {
            return Err(NeuralForgeError::backend("No models available for prediction"));
        }
        
        info!("Selected {} models for ensemble: {:?}", selected_models.len(), selected_models);
        
        // Get predictions from selected models
        let individual_predictions = self.get_individual_predictions(&selected_models, &request).await?;
        
        // Apply voting strategy
        let voting_strategy = request.options.voting_strategy
            .as_ref()
            .unwrap_or(&self.strategy_engine.current_voting_strategy);
        
        let model_weights = self.calculate_model_weights(&selected_models, &request.context).await;
        
        let ensemble_prediction = self.strategy_engine.apply_voting_strategy(
            voting_strategy,
            &individual_predictions,
            &model_weights,
        )?;
        
        // Calculate confidence and uncertainty
        let confidence = self.calculate_confidence(&individual_predictions, &ensemble_prediction).await;
        let uncertainty = if request.options.require_uncertainty {
            Some(self.calculate_uncertainty(&individual_predictions, &ensemble_prediction).await)
        } else {
            None
        };
        
        // Calculate consensus level
        let consensus_level = self.calculate_consensus(&individual_predictions).await;
        
        // Check consensus threshold
        if let Some(threshold) = request.options.consensus_threshold {
            if consensus_level < threshold {
                return Err(NeuralForgeError::backend(&format!(
                    "Insufficient consensus: {:.2} < {:.2}",
                    consensus_level, threshold
                )));
            }
        }
        
        // Update performance statistics
        let processing_time_us = start_time.elapsed().as_micros() as u64;
        self.update_performance_stats(&selected_models, processing_time_us).await;
        
        // Build response
        let response = EnsemblePredictionResponse {
            prediction: ensemble_prediction,
            individual_predictions,
            confidence,
            uncertainty,
            ensemble_metadata: EnsembleMetadata {
                models_used: selected_models.clone(),
                voting_strategy: voting_strategy.clone(),
                model_weights,
                consensus_level,
                selection_rationale: self.model_selector.get_last_rationale(),
                predicted_accuracy: self.predict_ensemble_accuracy(&selected_models, &request.context).await,
            },
            metadata: ResponseMetadata {
                request_id: request.metadata.request_id,
                processing_time_us,
                ensemble_version: "1.0.0".to_string(),
                coordination_overhead_us: processing_time_us / 10, // Estimate 10% overhead
                status: ResponseStatus::Success,
            },
        };
        
        info!("Ensemble prediction completed in {}μs", processing_time_us);
        Ok(response)
    }
    
    /// Get ensemble performance statistics
    pub async fn get_performance_stats(&self) -> EnsemblePerformance {
        self.performance_tracker.read().await.clone()
    }
    
    /// Update ensemble configuration
    pub async fn update_config(&mut self, new_config: EnsembleConfig) -> Result<()> {
        info!("Updating ensemble configuration");
        
        // Validate new configuration
        new_config.validate()?;
        
        // Update strategy engine
        self.strategy_engine.update_strategies(&new_config)?;
        
        // Update model selector
        self.model_selector.update_config(&new_config)?;
        
        self.config = new_config;
        Ok(())
    }
    
    /// Get available models
    fn get_available_models(&self) -> Vec<EnsembleModel> {
        self.models.values()
            .filter(|model| model.status == ModelStatus::Active && model.config.enabled)
            .cloned()
            .collect()
    }
    
    /// Validate model before registration
    fn validate_model(&self, model: &EnsembleModel) -> Result<()> {
        // Check if model ID is unique
        if self.models.contains_key(&model.id) {
            return Err(NeuralForgeError::validation(&format!("Model already registered: {}", model.id)));
        }
        
        // Validate model configuration
        if model.config.weight < 0.0 || model.config.weight > 1.0 {
            return Err(NeuralForgeError::validation("Model weight must be between 0.0 and 1.0"));
        }
        
        // Validate metadata
        if model.metadata.accuracy < 0.0 || model.metadata.accuracy > 1.0 {
            return Err(NeuralForgeError::validation("Model accuracy must be between 0.0 and 1.0"));
        }
        
        Ok(())
    }
    
    /// Update model registry
    async fn update_model_registry(&self) {
        let mut performance = self.performance_tracker.write().await;
        
        // Update model utilization tracking
        for model_id in self.models.keys() {
            performance.model_utilization.entry(model_id.clone()).or_insert(0.0);
            performance.individual_accuracies.entry(model_id.clone()).or_insert(0.0);
        }
    }
    
    /// Get individual predictions from selected models
    async fn get_individual_predictions(
        &self,
        selected_models: &[String],
        request: &EnsemblePredictionRequest,
    ) -> Result<HashMap<String, Vec<f64>>> {
        let mut predictions = HashMap::new();
        
        // In a real implementation, this would call the actual models
        // For now, simulate predictions
        for model_id in selected_models {
            if let Some(model) = self.models.get(model_id) {
                let prediction = self.simulate_model_prediction(model, &request.data).await?;
                predictions.insert(model_id.clone(), prediction);
            }
        }
        
        Ok(predictions)
    }
    
    /// Simulate model prediction (placeholder)
    async fn simulate_model_prediction(&self, model: &EnsembleModel, data: &[Vec<f64>]) -> Result<Vec<f64>> {
        // Simulate processing time based on model characteristics
        let processing_time_us = (model.metadata.latency_ms * 1000.0) as u64;
        tokio::time::sleep(std::time::Duration::from_micros(processing_time_us)).await;
        
        // Generate realistic prediction based on model type
        let prediction = match model.model_type {
            ModelSourceType::CognitionEngine => {
                // NHITS-style prediction with trend
                data.iter().map(|row| {
                    let sum: f64 = row.iter().sum();
                    let avg = sum / row.len() as f64;
                    avg * (1.0 + model.metadata.accuracy * 0.1)
                }).collect()
            }
            ModelSourceType::RuvFann => {
                // FANN-style prediction with neural patterns
                data.iter().map(|row| {
                    let weighted_sum: f64 = row.iter().enumerate()
                        .map(|(i, &val)| val * (i as f64 + 1.0) * model.config.weight)
                        .sum();
                    weighted_sum / row.len() as f64
                }).collect()
            }
            ModelSourceType::ClaudeFlow => {
                // AI-optimized prediction
                data.iter().map(|row| {
                    let complex_transform: f64 = row.iter()
                        .enumerate()
                        .map(|(i, &val)| val * (i as f64).sin() * model.metadata.accuracy)
                        .sum();
                    complex_transform / row.len() as f64
                }).collect()
            }
            ModelSourceType::NeuralForge => {
                // High-accuracy optimized prediction
                data.iter().map(|row| {
                    let optimized: f64 = row.iter()
                        .map(|&val| val * model.metadata.accuracy * model.config.weight)
                        .sum();
                    optimized / row.len() as f64
                }).collect()
            }
            ModelSourceType::External(_) => {
                // Generic external model prediction
                data.iter().map(|row| row.iter().sum::<f64>() / row.len() as f64).collect()
            }
        };
        
        Ok(prediction)
    }
    
    /// Calculate model weights based on context
    async fn calculate_model_weights(&self, models: &[String], context: &PredictionContext) -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        
        for model_id in models {
            if let Some(model) = self.models.get(model_id) {
                let base_weight = model.config.weight;
                
                // Adjust weight based on context and historical performance
                let context_adjustment = self.calculate_context_adjustment(model, context).await;
                let performance_adjustment = self.calculate_performance_adjustment(model).await;
                
                let final_weight = base_weight * context_adjustment * performance_adjustment;
                weights.insert(model_id.clone(), final_weight);
            }
        }
        
        // Normalize weights
        let total_weight: f64 = weights.values().sum();
        if total_weight > 0.0 {
            for weight in weights.values_mut() {
                *weight /= total_weight;
            }
        }
        
        weights
    }
    
    /// Calculate context-based weight adjustment
    async fn calculate_context_adjustment(&self, model: &EnsembleModel, context: &PredictionContext) -> f64 {
        let mut adjustment = 1.0;
        
        // Adjust based on model specialization
        if model.metadata.specialization.contains(&context.domain) {
            adjustment *= 1.2;
        }
        
        // Adjust based on data characteristics
        if context.data_characteristics.complexity > 0.8 && model.metadata.accuracy > 0.95 {
            adjustment *= 1.1;
        }
        
        // Adjust based on quality requirements
        if context.quality_requirements.min_accuracy > model.metadata.accuracy {
            adjustment *= 0.5; // Penalize models that don't meet requirements
        }
        
        adjustment.max(0.1).min(2.0) // Clamp between 0.1 and 2.0
    }
    
    /// Calculate performance-based weight adjustment
    async fn calculate_performance_adjustment(&self, model: &EnsembleModel) -> f64 {
        let recent_accuracy = if model.performance.recent_performance.len() >= 5 {
            let recent: f64 = model.performance.recent_performance
                .iter()
                .rev()
                .take(5)
                .map(|p| p.accuracy)
                .sum();
            recent / 5.0
        } else {
            model.performance.average_accuracy
        };
        
        // Scale based on recent performance vs baseline
        let baseline = model.metadata.accuracy;
        if recent_accuracy > baseline {
            1.0 + (recent_accuracy - baseline) * 2.0
        } else {
            1.0 - (baseline - recent_accuracy) * 1.5
        }.max(0.1).min(2.0)
    }
    
    /// Calculate confidence scores
    async fn calculate_confidence(&self, predictions: &HashMap<String, Vec<f64>>, ensemble: &[f64]) -> Vec<f64> {
        let mut confidence = vec![0.0; ensemble.len()];
        
        for i in 0..ensemble.len() {
            let individual_values: Vec<f64> = predictions.values()
                .map(|pred| pred.get(i).copied().unwrap_or(0.0))
                .collect();
            
            // Calculate confidence based on agreement between models
            if !individual_values.is_empty() {
                let mean: f64 = individual_values.iter().sum::<f64>() / individual_values.len() as f64;
                let variance: f64 = individual_values.iter()
                    .map(|&val| (val - mean).powi(2))
                    .sum::<f64>() / individual_values.len() as f64;
                
                // Higher agreement (lower variance) = higher confidence
                confidence[i] = 1.0 / (1.0 + variance).sqrt();
            }
        }
        
        confidence
    }
    
    /// Calculate uncertainty estimates
    async fn calculate_uncertainty(&self, predictions: &HashMap<String, Vec<f64>>, ensemble: &[f64]) -> Vec<f64> {
        let mut uncertainty = vec![0.0; ensemble.len()];
        
        for i in 0..ensemble.len() {
            let individual_values: Vec<f64> = predictions.values()
                .map(|pred| pred.get(i).copied().unwrap_or(0.0))
                .collect();
            
            if !individual_values.is_empty() {
                let mean: f64 = individual_values.iter().sum::<f64>() / individual_values.len() as f64;
                let std_dev: f64 = {
                    let variance: f64 = individual_values.iter()
                        .map(|&val| (val - mean).powi(2))
                        .sum::<f64>() / individual_values.len() as f64;
                    variance.sqrt()
                };
                
                uncertainty[i] = std_dev;
            }
        }
        
        uncertainty
    }
    
    /// Calculate consensus level
    async fn calculate_consensus(&self, predictions: &HashMap<String, Vec<f64>>) -> f64 {
        if predictions.is_empty() {
            return 0.0;
        }
        
        let num_predictions = predictions.values().next().map(|p| p.len()).unwrap_or(0);
        if num_predictions == 0 {
            return 0.0;
        }
        
        let mut total_consensus = 0.0;
        
        for i in 0..num_predictions {
            let values: Vec<f64> = predictions.values()
                .map(|pred| pred.get(i).copied().unwrap_or(0.0))
                .collect();
            
            if values.len() >= 2 {
                let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                let max_deviation: f64 = values.iter()
                    .map(|&val| (val - mean).abs())
                    .fold(0.0, f64::max);
                
                // Consensus is inversely related to maximum deviation
                let consensus = if max_deviation > 0.0 {
                    1.0 / (1.0 + max_deviation)
                } else {
                    1.0
                };
                
                total_consensus += consensus;
            }
        }
        
        total_consensus / num_predictions as f64
    }
    
    /// Predict ensemble accuracy
    async fn predict_ensemble_accuracy(&self, models: &[String], context: &PredictionContext) -> f64 {
        let mut total_accuracy = 0.0;
        let mut total_weight = 0.0;
        
        for model_id in models {
            if let Some(model) = self.models.get(model_id) {
                let weight = model.config.weight;
                let accuracy = model.metadata.accuracy;
                
                total_accuracy += accuracy * weight;
                total_weight += weight;
            }
        }
        
        if total_weight > 0.0 {
            total_accuracy / total_weight
        } else {
            0.0
        }
    }
    
    /// Update performance statistics
    async fn update_performance_stats(&self, models: &[String], processing_time_us: u64) {
        let mut performance = self.performance_tracker.write().await;
        
        performance.total_predictions += 1;
        performance.coordination_overhead_us = processing_time_us / 10; // Estimate
        
        // Update model utilization
        for model_id in models {
            let current_utilization = performance.model_utilization.get(model_id).copied().unwrap_or(0.0);
            let new_utilization = current_utilization + (1.0 / performance.total_predictions as f64);
            performance.model_utilization.insert(model_id.clone(), new_utilization);
        }
    }
}

// Stub implementations for strategy engine and model selector
impl StrategyEngine {
    pub fn new(config: &EnsembleConfig) -> Result<Self> {
        let mut voting_strategies: HashMap<String, Box<dyn VotingStrategyTrait + Send + Sync>> = HashMap::new();
        let mut selection_strategies: HashMap<String, Box<dyn SelectionStrategyTrait + Send + Sync>> = HashMap::new();
        
        // Add default strategies
        voting_strategies.insert("weighted_average".to_string(), Box::new(WeightedAverageVoting));
        voting_strategies.insert("majority_vote".to_string(), Box::new(MajorityVote));
        
        selection_strategies.insert("performance_based".to_string(), Box::new(PerformanceBasedSelection));
        selection_strategies.insert("consensus_based".to_string(), Box::new(ConsensusBasedSelection));
        
        Ok(Self {
            voting_strategies,
            selection_strategies,
            current_voting_strategy: "weighted_average".to_string(),
            current_selection_strategy: "performance_based".to_string(),
            adaptation_threshold: 0.85,
        })
    }
    
    pub fn apply_voting_strategy(
        &self,
        strategy_name: &str,
        predictions: &HashMap<String, Vec<f64>>,
        weights: &HashMap<String, f64>,
    ) -> Result<Vec<f64>> {
        match self.voting_strategies.get(strategy_name) {
            Some(strategy) => strategy.vote(predictions, weights),
            None => Err(NeuralForgeError::validation(&format!("Unknown voting strategy: {}", strategy_name))),
        }
    }
    
    pub fn update_strategies(&mut self, config: &EnsembleConfig) -> Result<()> {
        // Update strategy configuration
        Ok(())
    }
}

impl ModelSelector {
    pub fn new(config: &EnsembleConfig) -> Result<Self> {
        Ok(Self {
            selection_history: Vec::new(),
            performance_predictors: HashMap::new(),
            context_analyzer: ContextAnalyzer {
                pattern_library: HashMap::new(),
                complexity_estimator: ComplexityEstimator {
                    linear_threshold: 0.3,
                    nonlinear_threshold: 0.7,
                    chaotic_threshold: 0.9,
                },
                similarity_threshold: 0.8,
            },
            learning_enabled: true,
        })
    }
    
    pub async fn select_models(
        &mut self,
        available_models: &[EnsembleModel],
        context: &PredictionContext,
        options: &EnsemblePredictionOptions,
    ) -> Result<Vec<String>> {
        // Simple selection logic - would be more sophisticated in practice
        let max_models = options.max_models.unwrap_or(3).min(available_models.len());
        
        let mut selected: Vec<_> = available_models.iter()
            .filter(|model| {
                model.status == ModelStatus::Active &&
                model.config.enabled &&
                self.meets_requirements(model, context)
            })
            .collect();
        
        // Sort by performance score
        selected.sort_by(|a, b| {
            let score_a = self.calculate_selection_score(a, context);
            let score_b = self.calculate_selection_score(b, context);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        let selected_ids: Vec<String> = selected.into_iter()
            .take(max_models)
            .map(|model| model.id.clone())
            .collect();
        
        Ok(selected_ids)
    }
    
    pub fn get_last_rationale(&self) -> String {
        "Selected models based on performance and context compatibility".to_string()
    }
    
    pub fn update_config(&mut self, config: &EnsembleConfig) -> Result<()> {
        // Update selector configuration
        Ok(())
    }
    
    fn meets_requirements(&self, model: &EnsembleModel, context: &PredictionContext) -> bool {
        // Check if model meets context requirements
        model.metadata.accuracy >= context.quality_requirements.min_accuracy &&
        model.metadata.latency_ms <= context.quality_requirements.max_latency_ms as f64
    }
    
    fn calculate_selection_score(&self, model: &EnsembleModel, context: &PredictionContext) -> f64 {
        let mut score = model.metadata.accuracy;
        
        // Boost score for specialized models
        if model.metadata.specialization.contains(&context.domain) {
            score *= 1.2;
        }
        
        // Consider recent performance
        if !model.performance.recent_performance.is_empty() {
            let recent_avg: f64 = model.performance.recent_performance.iter()
                .rev()
                .take(5)
                .map(|p| p.accuracy)
                .sum::<f64>() / 5.0;
            score = (score + recent_avg) / 2.0;
        }
        
        score
    }
}

// Voting strategy implementations
#[derive(Debug)]
pub struct WeightedAverageVoting;

impl VotingStrategyTrait for WeightedAverageVoting {
    fn vote(&self, predictions: &HashMap<String, Vec<f64>>, weights: &HashMap<String, f64>) -> Result<Vec<f64>> {
        if predictions.is_empty() {
            return Ok(vec![]);
        }
        
        let prediction_length = predictions.values().next().unwrap().len();
        let mut result = vec![0.0; prediction_length];
        
        for i in 0..prediction_length {
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;
            
            for (model_id, pred) in predictions {
                if let Some(value) = pred.get(i) {
                    let weight = weights.get(model_id).copied().unwrap_or(1.0);
                    weighted_sum += value * weight;
                    total_weight += weight;
                }
            }
            
            if total_weight > 0.0 {
                result[i] = weighted_sum / total_weight;
            }
        }
        
        Ok(result)
    }
    
    fn name(&self) -> &str {
        "weighted_average"
    }
    
    fn supports_uncertainty(&self) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct MajorityVote;

impl VotingStrategyTrait for MajorityVote {
    fn vote(&self, predictions: &HashMap<String, Vec<f64>>, _weights: &HashMap<String, f64>) -> Result<Vec<f64>> {
        if predictions.is_empty() {
            return Ok(vec![]);
        }
        
        let prediction_length = predictions.values().next().unwrap().len();
        let mut result = vec![0.0; prediction_length];
        
        for i in 0..prediction_length {
            let values: Vec<f64> = predictions.values()
                .filter_map(|pred| pred.get(i).copied())
                .collect();
            
            if !values.is_empty() {
                // Simple majority vote using median
                let mut sorted_values = values;
                sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                result[i] = sorted_values[sorted_values.len() / 2];
            }
        }
        
        Ok(result)
    }
    
    fn name(&self) -> &str {
        "majority_vote"
    }
    
    fn supports_uncertainty(&self) -> bool {
        false
    }
}

// Selection strategy implementations
#[derive(Debug)]
pub struct PerformanceBasedSelection;

impl SelectionStrategyTrait for PerformanceBasedSelection {
    fn select_models(
        &self,
        available_models: &[EnsembleModel],
        context: &PredictionContext,
        options: &EnsemblePredictionOptions,
    ) -> Result<Vec<String>> {
        let max_models = options.max_models.unwrap_or(3);
        
        let mut scored_models: Vec<_> = available_models.iter()
            .filter(|model| model.status == ModelStatus::Active && model.config.enabled)
            .map(|model| (model.id.clone(), model.metadata.accuracy))
            .collect();
        
        scored_models.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(scored_models.into_iter()
            .take(max_models)
            .map(|(id, _)| id)
            .collect())
    }
    
    fn name(&self) -> &str {
        "performance_based"
    }
}

#[derive(Debug)]
pub struct ConsensusBasedSelection;

impl SelectionStrategyTrait for ConsensusBasedSelection {
    fn select_models(
        &self,
        available_models: &[EnsembleModel],
        context: &PredictionContext,
        options: &EnsemblePredictionOptions,
    ) -> Result<Vec<String>> {
        // Select models that are likely to provide good consensus
        let max_models = options.max_models.unwrap_or(5);
        
        let mut selected: Vec<_> = available_models.iter()
            .filter(|model| {
                model.status == ModelStatus::Active &&
                model.config.enabled &&
                model.metadata.accuracy >= context.quality_requirements.min_accuracy
            })
            .take(max_models)
            .map(|model| model.id.clone())
            .collect();
        
        // Ensure we have an odd number for better consensus
        if selected.len() % 2 == 0 && selected.len() > 1 {
            selected.pop();
        }
        
        Ok(selected)
    }
    
    fn name(&self) -> &str {
        "consensus_based"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_ensemble_coordinator_creation() {
        let config = EnsembleConfig::default();
        let coordinator = EnsembleCoordinator::new(config);
        assert!(coordinator.is_ok());
    }
    
    #[tokio::test]
    async fn test_model_registration() {
        let config = EnsembleConfig::default();
        let mut coordinator = EnsembleCoordinator::new(config).unwrap();
        
        let model = EnsembleModel {
            id: "test_model".to_string(),
            model_type: ModelSourceType::NeuralForge,
            metadata: ModelMetadata {
                name: "Test Model".to_string(),
                version: "1.0".to_string(),
                description: "Test".to_string(),
                architecture: "feedforward".to_string(),
                training_date: "2024-01-01".to_string(),
                accuracy: 0.95,
                latency_ms: 10.0,
                memory_mb: 100.0,
                specialization: vec!["crypto".to_string()],
            },
            performance: ModelPerformanceStats::default(),
            status: ModelStatus::Active,
            config: ModelConfig {
                weight: 1.0,
                priority: 5,
                timeout_ms: 5000,
                retry_count: 3,
                enabled: true,
                constraints: ModelConstraints {
                    max_latency_ms: Some(1000),
                    min_accuracy: Some(0.9),
                    max_memory_mb: Some(500.0),
                    required_features: vec![],
                    excluded_patterns: vec![],
                },
            },
        };
        
        let result = coordinator.register_model(model).await;
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_weighted_average_voting() {
        let voting = WeightedAverageVoting;
        
        let mut predictions = HashMap::new();
        predictions.insert("model1".to_string(), vec![1.0, 2.0, 3.0]);
        predictions.insert("model2".to_string(), vec![2.0, 3.0, 4.0]);
        
        let mut weights = HashMap::new();
        weights.insert("model1".to_string(), 0.6);
        weights.insert("model2".to_string(), 0.4);
        
        let result = voting.vote(&predictions, &weights).unwrap();
        assert_eq!(result.len(), 3);
        
        // Expected: [1.0*0.6 + 2.0*0.4, 2.0*0.6 + 3.0*0.4, 3.0*0.6 + 4.0*0.4]
        //         = [1.4, 2.4, 3.4]
        assert!((result[0] - 1.4).abs() < 1e-6);
        assert!((result[1] - 2.4).abs() < 1e-6);
        assert!((result[2] - 3.4).abs() < 1e-6);
    }
    
    #[test]
    fn test_majority_vote() {
        let voting = MajorityVote;
        
        let mut predictions = HashMap::new();
        predictions.insert("model1".to_string(), vec![1.0, 5.0, 3.0]);
        predictions.insert("model2".to_string(), vec![2.0, 3.0, 4.0]);
        predictions.insert("model3".to_string(), vec![3.0, 4.0, 5.0]);
        
        let weights = HashMap::new();
        
        let result = voting.vote(&predictions, &weights).unwrap();
        assert_eq!(result.len(), 3);
        
        // Expected median values: [2.0, 4.0, 4.0]
        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 4.0).abs() < 1e-6);
        assert!((result[2] - 4.0).abs() < 1e-6);
    }
}