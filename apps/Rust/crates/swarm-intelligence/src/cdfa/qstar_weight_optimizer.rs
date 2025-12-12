//! Q* Algorithm-based Weight Optimizer for CDFA
//! 
//! This module implements weight optimization using the Q* reinforcement learning algorithm
//! integrated with the existing q-star-core infrastructure. It provides adaptive weight
//! optimization for CDFA fusion strategies based on market conditions and performance feedback.

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

// Import Q* core components
use q_star_core::{
    QStarEngine, QStarConfig, QStarError, QStarMetrics, Experience,
    MarketState, QStarAction, Policy, ValueFunction, ExperienceMemory, SearchTree
};

use crate::errors::SwarmError;
use super::ml_integration::{QStarWeightOptimizer, MLExperience, WeightUpdate};

/// Q* Weight Optimizer Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QStarOptimizerConfig {
    /// Q* algorithm configuration
    pub qstar_config: QStarConfig,
    
    /// Weight learning rate
    pub weight_learning_rate: f64,
    
    /// Weight decay factor
    pub weight_decay: f64,
    
    /// Minimum weight value
    pub min_weight: f64,
    
    /// Maximum weight value  
    pub max_weight: f64,
    
    /// Experience buffer size
    pub experience_buffer_size: usize,
    
    /// Update frequency (number of experiences before update)
    pub update_frequency: usize,
    
    /// Performance target for weight optimization
    pub performance_target: f64,
}

impl Default for QStarOptimizerConfig {
    fn default() -> Self {
        Self {
            qstar_config: QStarConfig::default(),
            weight_learning_rate: 0.01,
            weight_decay: 0.0001,
            min_weight: 0.001,
            max_weight: 1.0,
            experience_buffer_size: 10000,
            update_frequency: 100,
            performance_target: 0.75,
        }
    }
}

/// Weight optimization state for Q* algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightState {
    /// Current weights for each signal
    pub weights: HashMap<String, f64>,
    
    /// Market conditions vector
    pub market_conditions: Vec<f64>,
    
    /// Performance metrics
    pub performance_metrics: Vec<f64>,
    
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl WeightState {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            market_conditions: vec![],
            performance_metrics: vec![],
            timestamp: Utc::now(),
        }
    }
    
    pub fn to_state_vector(&self) -> Vec<f64> {
        let mut state_vector = Vec::new();
        
        // Add normalized weights
        for (_, weight) in &self.weights {
            state_vector.push(*weight);
        }
        
        // Add market conditions
        state_vector.extend_from_slice(&self.market_conditions);
        
        // Add performance metrics
        state_vector.extend_from_slice(&self.performance_metrics);
        
        state_vector
    }
}

/// Weight adjustment action for Q* algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightAction {
    /// Signal identifier
    pub signal_id: String,
    
    /// Weight adjustment direction and magnitude
    pub adjustment: f64,
    
    /// Action type
    pub action_type: WeightActionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightActionType {
    Increase,
    Decrease,
    Reset,
    Maintain,
}

/// Q* Weight Optimizer Implementation
pub struct QStarWeightOptimizerImpl {
    /// Configuration
    config: QStarOptimizerConfig,
    
    /// Q* algorithm engine
    qstar_engine: Arc<QStarEngine>,
    
    /// Current weight state
    current_state: Arc<RwLock<WeightState>>,
    
    /// Experience buffer for training
    experience_buffer: Arc<RwLock<Vec<MLExperience>>>,
    
    /// Performance history
    performance_history: Arc<RwLock<Vec<f64>>>,
    
    /// Update counter
    update_counter: Arc<RwLock<usize>>,
    
    /// Metrics
    metrics: Arc<RwLock<QStarOptimizerMetrics>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QStarOptimizerMetrics {
    pub total_optimizations: u64,
    pub average_performance: f64,
    pub weight_stability: f64,
    pub convergence_rate: f64,
    pub last_update: DateTime<Utc>,
    pub qstar_metrics: QStarMetrics,
}

impl Default for QStarOptimizerMetrics {
    fn default() -> Self {
        Self {
            total_optimizations: 0,
            average_performance: 0.0,
            weight_stability: 0.0,
            convergence_rate: 0.0,
            last_update: Utc::now(),
            qstar_metrics: QStarMetrics::default(),
        }
    }
}

impl QStarWeightOptimizerImpl {
    /// Create new Q* weight optimizer
    pub async fn new(
        config: QStarOptimizerConfig,
        policy: Arc<dyn Policy + Send + Sync>,
        value_function: Arc<dyn ValueFunction + Send + Sync>,
        memory: Arc<dyn ExperienceMemory + Send + Sync>,
        search: Arc<dyn SearchTree + Send + Sync>,
    ) -> Result<Self, SwarmError> {
        let qstar_engine = Arc::new(QStarEngine::new(
            config.qstar_config.clone(),
            policy,
            value_function,
            memory,
            search,
        ));
        
        Ok(Self {
            config,
            qstar_engine,
            current_state: Arc::new(RwLock::new(WeightState::new())),
            experience_buffer: Arc::new(RwLock::new(Vec::new())),
            performance_history: Arc::new(RwLock::new(Vec::new())),
            update_counter: Arc::new(RwLock::new(0)),
            metrics: Arc::new(RwLock::new(QStarOptimizerMetrics::default())),
        })
    }
    
    /// Convert CDFA state to Q* MarketState
    async fn convert_to_market_state(&self, weights: &HashMap<String, f64>, market_data: &[f64]) -> MarketState {
        let mut state_vector = Vec::new();
        
        // Add weights as state features
        for (_, weight) in weights {
            state_vector.push(*weight);
        }
        
        // Add market data
        state_vector.extend_from_slice(market_data);
        
        // Create Q* MarketState (implementation depends on q-star-core specifics)
        MarketState::new(state_vector)
    }
    
    /// Convert Q* action to weight updates
    async fn convert_from_qstar_action(&self, action: &QStarAction) -> Vec<WeightUpdate> {
        let mut updates = Vec::new();
        
        // Extract weight adjustments from Q* action
        // This conversion depends on how actions are encoded in q-star-core
        let action_data = action.get_data();
        
        for (i, adjustment) in action_data.iter().enumerate() {
            if adjustment.abs() > 1e-6 {
                updates.push(WeightUpdate {
                    signal_id: format!("signal_{}", i),
                    old_weight: 0.0, // Will be filled with current weight
                    new_weight: adjustment.min(self.config.max_weight).max(self.config.min_weight),
                    confidence: adjustment.abs(),
                    reason: format!("Q* optimization: {:.4}", adjustment),
                });
            }
        }
        
        updates
    }
    
    /// Calculate reward for weight optimization
    async fn calculate_reward(
        &self,
        old_performance: f64,
        new_performance: f64,
        weight_changes: &[WeightUpdate],
    ) -> f64 {
        // Performance improvement reward
        let performance_reward = (new_performance - old_performance) * 10.0;
        
        // Weight stability penalty (prefer smaller changes)
        let stability_penalty = weight_changes
            .iter()
            .map(|update| (update.new_weight - update.old_weight).abs())
            .sum::<f64>() * -0.1;
        
        // Convergence bonus
        let convergence_bonus = if new_performance > self.config.performance_target {
            1.0
        } else {
            0.0
        };
        
        performance_reward + stability_penalty + convergence_bonus
    }
    
    /// Update experience buffer and train if needed
    async fn update_and_train(&self, experience: MLExperience) -> Result<(), SwarmError> {
        // Add to experience buffer
        {
            let mut buffer = self.experience_buffer.write().await;
            buffer.push(experience);
            
            // Maintain buffer size limit
            if buffer.len() > self.config.experience_buffer_size {
                buffer.remove(0);
            }
        }
        
        // Check if it's time to update
        {
            let mut counter = self.update_counter.write().await;
            *counter += 1;
            
            if *counter >= self.config.update_frequency {
                *counter = 0;
                
                // Convert experiences to Q* format and train
                let buffer = self.experience_buffer.read().await;
                let mut qstar_experiences = Vec::new();
                
                for exp in buffer.iter() {
                    let market_state = MarketState::new(exp.state_vector.clone());
                    let next_market_state = MarketState::new(exp.next_state_vector.clone());
                    let action = QStarAction::new(exp.action_vector.clone());
                    
                    qstar_experiences.push(Experience {
                        state: market_state,
                        action,
                        reward: exp.reward,
                        next_state: next_market_state,
                        done: exp.done,
                        timestamp: exp.timestamp,
                    });
                }
                
                // Train Q* algorithm
                self.qstar_engine.train(&qstar_experiences).await
                    .map_err(|e| SwarmError::ParameterError(format!("Q* training failed: {}", e)))?;
            }
        }
        
        Ok(())
    }
    
    /// Update metrics
    async fn update_metrics(&self, performance: f64, weights: &HashMap<String, f64>) -> Result<(), SwarmError> {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_optimizations += 1;
        
        // Update average performance
        let alpha = 0.1;
        metrics.average_performance = alpha * performance + (1.0 - alpha) * metrics.average_performance;
        
        // Calculate weight stability (variance of weights)
        let weight_values: Vec<f64> = weights.values().cloned().collect();
        let mean_weight = weight_values.iter().sum::<f64>() / weight_values.len() as f64;
        let variance = weight_values
            .iter()
            .map(|w| (w - mean_weight).powi(2))
            .sum::<f64>() / weight_values.len() as f64;
        metrics.weight_stability = variance.sqrt();
        
        // Update convergence rate (based on performance target)
        metrics.convergence_rate = if performance > self.config.performance_target {
            metrics.convergence_rate * 0.9 + 0.1
        } else {
            metrics.convergence_rate * 0.95
        };
        
        // Get Q* metrics
        metrics.qstar_metrics = self.qstar_engine.get_metrics().await;
        
        metrics.last_update = Utc::now();
        
        Ok(())
    }
}

#[async_trait]
impl QStarWeightOptimizer for QStarWeightOptimizerImpl {
    async fn optimize_weights(
        &self,
        current_weights: &HashMap<String, f64>,
        performance_metrics: &[f64],
        market_data: &[f64],
    ) -> Result<Vec<WeightUpdate>, SwarmError> {
        // Convert to Q* state
        let market_state = self.convert_to_market_state(current_weights, market_data).await;
        
        // Get Q* decision
        let action = self.qstar_engine.decide(&market_state).await
            .map_err(|e| SwarmError::ParameterError(format!("Q* decision failed: {}", e)))?;
        
        // Convert to weight updates
        let mut updates = self.convert_from_qstar_action(&action).await;
        
        // Fill in current weights
        for update in &mut updates {
            if let Some(&current_weight) = current_weights.get(&update.signal_id) {
                update.old_weight = current_weight;
            }
        }
        
        // Update state
        {
            let mut state = self.current_state.write().await;
            state.weights = current_weights.clone();
            state.market_conditions = market_data.to_vec();
            state.performance_metrics = performance_metrics.to_vec();
            state.timestamp = Utc::now();
        }
        
        Ok(updates)
    }
    
    async fn learn_from_experience(&self, experience: MLExperience) -> Result<(), SwarmError> {
        // Update experience buffer and train if needed
        self.update_and_train(experience).await?;
        
        // Update metrics
        let current_weights = self.current_state.read().await.weights.clone();
        self.update_metrics(experience.reward, &current_weights).await?;
        
        Ok(())
    }
    
    async fn get_optimization_metrics(&self) -> Result<serde_json::Value, SwarmError> {
        let metrics = self.metrics.read().await;
        
        serde_json::to_value(&*metrics)
            .map_err(|e| SwarmError::SerializationError(format!("Failed to serialize metrics: {}", e)))
    }
    
    async fn reset_optimization_state(&self) -> Result<(), SwarmError> {
        // Reset state
        {
            let mut state = self.current_state.write().await;
            *state = WeightState::new();
        }
        
        // Clear experience buffer
        {
            let mut buffer = self.experience_buffer.write().await;
            buffer.clear();
        }
        
        // Reset counter
        {
            let mut counter = self.update_counter.write().await;
            *counter = 0;
        }
        
        // Reset metrics
        {
            let mut metrics = self.metrics.write().await;
            *metrics = QStarOptimizerMetrics::default();
        }
        
        Ok(())
    }
    
    async fn save_optimization_state(&self, path: &str) -> Result<(), SwarmError> {
        let state = self.current_state.read().await;
        let metrics = self.metrics.read().await;
        
        let save_data = serde_json::json!({
            "state": *state,
            "metrics": *metrics,
            "config": self.config
        });
        
        std::fs::write(path, serde_json::to_string_pretty(&save_data)
            .map_err(|e| SwarmError::SerializationError(format!("Serialization failed: {}", e)))?)
            .map_err(|e| SwarmError::IOError(format!("Failed to save state: {}", e)))?;
        
        Ok(())
    }
    
    async fn load_optimization_state(&self, path: &str) -> Result<(), SwarmError> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| SwarmError::IOError(format!("Failed to read state: {}", e)))?;
        
        let save_data: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| SwarmError::SerializationError(format!("Deserialization failed: {}", e)))?;
        
        // Restore state
        if let Some(state_data) = save_data.get("state") {
            let restored_state: WeightState = serde_json::from_value(state_data.clone())
                .map_err(|e| SwarmError::SerializationError(format!("State deserialization failed: {}", e)))?;
            
            let mut state = self.current_state.write().await;
            *state = restored_state;
        }
        
        // Restore metrics
        if let Some(metrics_data) = save_data.get("metrics") {
            let restored_metrics: QStarOptimizerMetrics = serde_json::from_value(metrics_data.clone())
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
    use std::collections::HashMap;
    
    #[tokio::test]
    async fn test_qstar_optimizer_config() {
        let config = QStarOptimizerConfig::default();
        assert!(config.weight_learning_rate > 0.0);
        assert!(config.max_weight > config.min_weight);
        assert!(config.experience_buffer_size > 0);
    }
    
    #[tokio::test]
    async fn test_weight_state_creation() {
        let state = WeightState::new();
        assert!(state.weights.is_empty());
        assert!(state.market_conditions.is_empty());
        assert!(state.performance_metrics.is_empty());
    }
    
    #[tokio::test]
    async fn test_weight_state_vector_conversion() {
        let mut state = WeightState::new();
        state.weights.insert("signal_1".to_string(), 0.5);
        state.weights.insert("signal_2".to_string(), 0.3);
        state.market_conditions = vec![0.1, 0.2, 0.3];
        state.performance_metrics = vec![0.7, 0.8];
        
        let vector = state.to_state_vector();
        assert_eq!(vector.len(), 7); // 2 weights + 3 market + 2 performance
        assert!(vector.contains(&0.5));
        assert!(vector.contains(&0.3));
    }
    
    #[tokio::test]
    async fn test_optimizer_metrics_default() {
        let metrics = QStarOptimizerMetrics::default();
        assert_eq!(metrics.total_optimizations, 0);
        assert_eq!(metrics.average_performance, 0.0);
        assert_eq!(metrics.weight_stability, 0.0);
        assert_eq!(metrics.convergence_rate, 0.0);
    }
}