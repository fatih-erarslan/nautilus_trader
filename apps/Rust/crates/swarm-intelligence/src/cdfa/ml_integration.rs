//! Machine Learning and Reinforcement Learning integration for CDFA
//!
//! This module provides advanced ML/RL capabilities for signal processing,
//! weight optimization, and adaptive fusion strategies, leveraging our
//! existing Q* algorithm, neural networks, and ensemble methods.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use ndarray::{Array1, Array2};
use parking_lot::Mutex;
use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::core::SwarmError;

/// ML/RL integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    /// Enable Q* reinforcement learning
    pub enable_qstar: bool,
    
    /// Enable neural network signal processing
    pub enable_neural: bool,
    
    /// Enable ensemble methods
    pub enable_ensemble: bool,
    
    /// Learning rate for adaptive weights
    pub learning_rate: f64,
    
    /// Discount factor for Q* algorithm
    pub discount_factor: f64,
    
    /// Exploration rate for Q* decisions
    pub exploration_rate: f64,
    
    /// Update frequency for neural networks
    pub update_frequency: usize,
    
    /// Batch size for training
    pub batch_size: usize,
    
    /// Memory capacity for experience replay
    pub memory_capacity: usize,
    
    /// Performance threshold for adaptation
    pub performance_threshold: f64,
    
    /// Maximum training iterations
    pub max_training_iterations: usize,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            enable_qstar: true,
            enable_neural: true,
            enable_ensemble: true,
            learning_rate: 0.001,
            discount_factor: 0.99,
            exploration_rate: 0.1,
            update_frequency: 100,
            batch_size: 64,
            memory_capacity: 10000,
            performance_threshold: 0.7,
            max_training_iterations: 1000,
        }
    }
}

/// Signal processing context for ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalContext {
    /// Symbol identifier
    pub symbol: String,
    
    /// Signal source name
    pub source: String,
    
    /// Signal values time series
    pub values: Vec<f64>,
    
    /// Signal timestamps
    pub timestamps: Vec<DateTime<Utc>>,
    
    /// Signal metadata
    pub metadata: HashMap<String, serde_json::Value>,
    
    /// Performance metrics
    pub performance: Option<f64>,
}

/// Q* state representation for signal fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionState {
    /// Current signal values
    pub signals: HashMap<String, f64>,
    
    /// Signal diversities
    pub diversities: HashMap<String, f64>,
    
    /// Current weights
    pub weights: HashMap<String, f64>,
    
    /// Market context
    pub market_context: Vec<f64>,
    
    /// Performance history
    pub performance_history: Vec<f64>,
    
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Q* action for weight adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightAction {
    /// Signal name to adjust
    pub signal: String,
    
    /// Weight adjustment (-1.0 to 1.0)
    pub adjustment: f64,
    
    /// Action confidence
    pub confidence: f64,
}

/// Experience for Q* learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionExperience {
    /// State before action
    pub state: FusionState,
    
    /// Action taken
    pub action: WeightAction,
    
    /// Reward received
    pub reward: f64,
    
    /// Next state
    pub next_state: FusionState,
    
    /// Whether episode terminated
    pub done: bool,
    
    /// Experience timestamp
    pub timestamp: DateTime<Utc>,
}

/// Neural network model for signal processing
#[async_trait]
pub trait NeuralSignalProcessor: Send + Sync {
    /// Process signals using neural network
    async fn process_signals(&self, signals: &[SignalContext]) -> Result<Vec<f64>, SwarmError>;
    
    /// Update model with training data
    async fn train(&mut self, data: &[SignalContext], targets: &[f64]) -> Result<(), SwarmError>;
    
    /// Get model performance metrics
    async fn get_metrics(&self) -> Result<HashMap<String, f64>, SwarmError>;
}

/// Q* agent for weight optimization
#[async_trait]
pub trait QStarWeightOptimizer: Send + Sync {
    /// Decide weight adjustments based on current state
    async fn decide_weights(&self, state: &FusionState) -> Result<HashMap<String, f64>, SwarmError>;
    
    /// Learn from experience
    async fn learn(&mut self, experiences: &[FusionExperience]) -> Result<(), SwarmError>;
    
    /// Get Q* algorithm metrics
    async fn get_qstar_metrics(&self) -> Result<HashMap<String, f64>, SwarmError>;
}

/// Ensemble coordinator for multiple models
#[async_trait]
pub trait EnsembleCoordinator: Send + Sync {
    /// Coordinate multiple models for prediction
    async fn coordinate_prediction(&self, contexts: &[SignalContext]) -> Result<Vec<f64>, SwarmError>;
    
    /// Update ensemble weights based on performance
    async fn update_ensemble_weights(&mut self, performances: &[f64]) -> Result<(), SwarmError>;
    
    /// Get ensemble metrics
    async fn get_ensemble_metrics(&self) -> Result<HashMap<String, f64>, SwarmError>;
}

/// Main ML/RL integration engine for CDFA
pub struct CDFAMLEngine {
    /// Configuration
    config: MLConfig,
    
    /// Q* weight optimizer
    qstar_optimizer: Option<Arc<Mutex<dyn QStarWeightOptimizer>>>,
    
    /// Neural signal processors
    neural_processors: HashMap<String, Arc<Mutex<dyn NeuralSignalProcessor>>>,
    
    /// Ensemble coordinator
    ensemble_coordinator: Option<Arc<Mutex<dyn EnsembleCoordinator>>>,
    
    /// Experience replay memory
    experience_memory: Arc<RwLock<Vec<FusionExperience>>>,
    
    /// Performance tracking
    performance_history: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    
    /// Adaptive weights
    adaptive_weights: Arc<RwLock<HashMap<String, f64>>>,
    
    /// Training iteration counter
    iteration_counter: Arc<Mutex<usize>>,
}

impl CDFAMLEngine {
    /// Create new ML/RL engine
    pub fn new(config: MLConfig) -> Self {
        Self {
            config,
            qstar_optimizer: None,
            neural_processors: HashMap::new(),
            ensemble_coordinator: None,
            experience_memory: Arc::new(RwLock::new(Vec::new())),
            performance_history: Arc::new(RwLock::new(HashMap::new())),
            adaptive_weights: Arc::new(RwLock::new(HashMap::new())),
            iteration_counter: Arc::new(Mutex::new(0)),
        }
    }
    
    /// Register Q* weight optimizer
    pub fn register_qstar_optimizer(&mut self, optimizer: Arc<Mutex<dyn QStarWeightOptimizer>>) {
        self.qstar_optimizer = Some(optimizer);
    }
    
    /// Register neural signal processor
    pub fn register_neural_processor(&mut self, name: String, processor: Arc<Mutex<dyn NeuralSignalProcessor>>) {
        self.neural_processors.insert(name, processor);
    }
    
    /// Register ensemble coordinator
    pub fn register_ensemble_coordinator(&mut self, coordinator: Arc<Mutex<dyn EnsembleCoordinator>>) {
        self.ensemble_coordinator = Some(coordinator);
    }
    
    /// Process signals using ML/RL pipeline
    pub async fn process_signals(&self, contexts: &[SignalContext]) -> Result<ProcessingResult, SwarmError> {
        let mut results = ProcessingResult::default();
        
        // Neural signal processing
        if self.config.enable_neural {
            results.neural_outputs = self.process_with_neural(contexts).await?;
        }
        
        // Ensemble coordination
        if self.config.enable_ensemble {
            if let Some(ensemble) = &self.ensemble_coordinator {
                let coordinator = ensemble.lock();
                results.ensemble_prediction = coordinator.coordinate_prediction(contexts).await?;
            }
        }
        
        // Q* weight optimization
        if self.config.enable_qstar {
            results.optimized_weights = self.optimize_weights_qstar(contexts).await?;
        }
        
        // Adaptive learning
        results.adaptive_weights = self.update_adaptive_weights(contexts, &results).await?;
        
        Ok(results)
    }
    
    /// Process signals with neural networks
    async fn process_with_neural(&self, contexts: &[SignalContext]) -> Result<HashMap<String, Vec<f64>>, SwarmError> {
        let mut neural_outputs = HashMap::new();
        
        for (name, processor) in &self.neural_processors {
            let processor_guard = processor.lock();
            let outputs = processor_guard.process_signals(contexts).await?;
            neural_outputs.insert(name.clone(), outputs);
        }
        
        Ok(neural_outputs)
    }
    
    /// Optimize weights using Q* algorithm
    async fn optimize_weights_qstar(&self, contexts: &[SignalContext]) -> Result<HashMap<String, f64>, SwarmError> {
        if let Some(qstar) = &self.qstar_optimizer {
            // Convert signal contexts to fusion state
            let state = self.contexts_to_fusion_state(contexts).await?;
            
            // Get Q* decision
            let optimizer = qstar.lock();
            let weights = optimizer.decide_weights(&state).await?;
            
            Ok(weights)
        } else {
            Ok(HashMap::new())
        }
    }
    
    /// Update adaptive weights based on performance
    async fn update_adaptive_weights(
        &self,
        contexts: &[SignalContext],
        results: &ProcessingResult,
    ) -> Result<HashMap<String, f64>, SwarmError> {
        let mut adaptive_weights = self.adaptive_weights.write().await;
        
        // Calculate performance for each signal
        for context in contexts {
            if let Some(performance) = context.performance {
                let current_weight = adaptive_weights.get(&context.source).copied().unwrap_or(1.0);
                
                // Adaptive learning rule
                let weight_update = self.config.learning_rate * (performance - self.config.performance_threshold);
                let new_weight = (current_weight + weight_update).clamp(0.0, 2.0);
                
                adaptive_weights.insert(context.source.clone(), new_weight);
            }
        }
        
        Ok(adaptive_weights.clone())
    }
    
    /// Convert signal contexts to fusion state for Q*
    async fn contexts_to_fusion_state(&self, contexts: &[SignalContext]) -> Result<FusionState, SwarmError> {
        let mut signals = HashMap::new();
        let mut diversities = HashMap::new();
        let adaptive_weights = self.adaptive_weights.read().await;
        
        for context in contexts {
            if let Some(value) = context.values.last() {
                signals.insert(context.source.clone(), *value);
            }
            
            // Calculate signal diversity (simplified)
            let diversity = if context.values.len() > 1 {
                let variance: f64 = context.values.iter()
                    .map(|x| (*x - context.values.iter().sum::<f64>() / context.values.len() as f64).powi(2))
                    .sum::<f64>() / context.values.len() as f64;
                variance.sqrt()
            } else {
                0.0
            };
            diversities.insert(context.source.clone(), diversity);
        }
        
        Ok(FusionState {
            signals,
            diversities,
            weights: adaptive_weights.clone(),
            market_context: vec![], // To be filled with market data
            performance_history: vec![], // Recent performance values
            timestamp: Utc::now(),
        })
    }
    
    /// Learn from fusion experience
    pub async fn learn_from_experience(&self, experience: FusionExperience) -> Result<(), SwarmError> {
        // Store experience in memory
        {
            let mut memory = self.experience_memory.write().await;
            memory.push(experience.clone());
            
            // Limit memory size
            if memory.len() > self.config.memory_capacity {
                memory.remove(0);
            }
        }
        
        // Train Q* optimizer if available
        if let Some(qstar) = &self.qstar_optimizer {
            let memory = self.experience_memory.read().await;
            if memory.len() >= self.config.batch_size {
                let batch: Vec<FusionExperience> = memory.iter()
                    .rev()
                    .take(self.config.batch_size)
                    .cloned()
                    .collect();
                
                let mut optimizer = qstar.lock();
                optimizer.learn(&batch).await?;
            }
        }
        
        // Update iteration counter and trigger training
        {
            let mut counter = self.iteration_counter.lock();
            *counter += 1;
            
            if *counter % self.config.update_frequency == 0 {
                self.train_neural_processors().await?;
            }
        }
        
        Ok(())
    }
    
    /// Train neural processors with accumulated data
    async fn train_neural_processors(&self) -> Result<(), SwarmError> {
        let memory = self.experience_memory.read().await;
        
        if memory.len() < self.config.batch_size {
            return Ok(());
        }
        
        // Prepare training data from experiences
        let mut training_contexts = Vec::new();
        let mut targets = Vec::new();
        
        for experience in memory.iter().rev().take(self.config.batch_size) {
            // Convert fusion state to signal context
            for (signal_name, signal_value) in &experience.state.signals {
                let context = SignalContext {
                    symbol: "training".to_string(),
                    source: signal_name.clone(),
                    values: vec![*signal_value],
                    timestamps: vec![experience.timestamp],
                    metadata: HashMap::new(),
                    performance: Some(experience.reward),
                };
                training_contexts.push(context);
                targets.push(experience.reward);
            }
        }
        
        // Train each neural processor
        for (name, processor) in &self.neural_processors {
            let mut processor_guard = processor.lock();
            processor_guard.train(&training_contexts, &targets).await?;
        }
        
        Ok(())
    }
    
    /// Get comprehensive ML/RL metrics
    pub async fn get_ml_metrics(&self) -> Result<MLMetrics, SwarmError> {
        let mut metrics = MLMetrics::default();
        
        // Q* metrics
        if let Some(qstar) = &self.qstar_optimizer {
            let optimizer = qstar.lock();
            metrics.qstar_metrics = optimizer.get_qstar_metrics().await?;
        }
        
        // Neural metrics
        for (name, processor) in &self.neural_processors {
            let processor_guard = processor.lock();
            let neural_metrics = processor_guard.get_metrics().await?;
            metrics.neural_metrics.insert(name.clone(), neural_metrics);
        }
        
        // Ensemble metrics
        if let Some(ensemble) = &self.ensemble_coordinator {
            let coordinator = ensemble.lock();
            metrics.ensemble_metrics = coordinator.get_ensemble_metrics().await?;
        }
        
        // Experience memory stats
        let memory = self.experience_memory.read().await;
        metrics.memory_size = memory.len();
        metrics.memory_capacity = self.config.memory_capacity;
        
        // Adaptive weights
        let adaptive_weights = self.adaptive_weights.read().await;
        metrics.adaptive_weights = adaptive_weights.clone();
        
        Ok(metrics)
    }
}

/// Result of ML/RL signal processing
#[derive(Debug, Clone, Default)]
pub struct ProcessingResult {
    /// Neural network outputs
    pub neural_outputs: HashMap<String, Vec<f64>>,
    
    /// Ensemble prediction
    pub ensemble_prediction: Vec<f64>,
    
    /// Q* optimized weights
    pub optimized_weights: HashMap<String, f64>,
    
    /// Adaptive weights
    pub adaptive_weights: HashMap<String, f64>,
}

/// Comprehensive ML/RL metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MLMetrics {
    /// Q* algorithm metrics
    pub qstar_metrics: HashMap<String, f64>,
    
    /// Neural network metrics by processor
    pub neural_metrics: HashMap<String, HashMap<String, f64>>,
    
    /// Ensemble metrics
    pub ensemble_metrics: HashMap<String, f64>,
    
    /// Experience memory size
    pub memory_size: usize,
    
    /// Memory capacity
    pub memory_capacity: usize,
    
    /// Current adaptive weights
    pub adaptive_weights: HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_ml_engine_creation() {
        let config = MLConfig::default();
        let engine = CDFAMLEngine::new(config);
        
        assert!(engine.qstar_optimizer.is_none());
        assert!(engine.neural_processors.is_empty());
        assert!(engine.ensemble_coordinator.is_none());
    }
    
    #[tokio::test]
    async fn test_signal_context_creation() {
        let context = SignalContext {
            symbol: "BTCUSD".to_string(),
            source: "rsi".to_string(),
            values: vec![0.3, 0.4, 0.5],
            timestamps: vec![Utc::now(); 3],
            metadata: HashMap::new(),
            performance: Some(0.8),
        };
        
        assert_eq!(context.symbol, "BTCUSD");
        assert_eq!(context.values.len(), 3);
        assert_eq!(context.performance, Some(0.8));
    }
    
    #[tokio::test]
    async fn test_fusion_state_creation() {
        let mut signals = HashMap::new();
        signals.insert("signal1".to_string(), 0.5);
        signals.insert("signal2".to_string(), 0.7);
        
        let state = FusionState {
            signals,
            diversities: HashMap::new(),
            weights: HashMap::new(),
            market_context: vec![],
            performance_history: vec![],
            timestamp: Utc::now(),
        };
        
        assert_eq!(state.signals.len(), 2);
    }
}