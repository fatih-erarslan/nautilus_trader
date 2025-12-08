//! Q* Algorithm Implementation
//! 
//! Core Q* algorithm that combines Q-learning with advanced search techniques
//! for ultra-low latency trading decisions.

use crate::{QStarError, MarketState, QStarAction, Experience};
use async_trait::async_trait;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Q* algorithm trait defining core functionality
#[async_trait]
pub trait QStarAlgorithm {
    /// Execute Q* decision process
    async fn decide(&self, state: &MarketState) -> Result<QStarAction, QStarError>;
    
    /// Update algorithm with new experience
    async fn update(&mut self, experience: &Experience) -> Result<(), QStarError>;
    
    /// Get Q-value for state-action pair
    async fn get_q_value(&self, state: &MarketState, action: &QStarAction) -> Result<f64, QStarError>;
    
    /// Get algorithm convergence status
    async fn is_converged(&self) -> Result<bool, QStarError>;
}

/// Q* algorithm implementation combining neural networks and tree search
pub struct QStarAlgorithmImpl {
    /// Q-table for discrete state-action pairs
    q_table: HashMap<(StateHash, ActionHash), f64>,
    
    /// Learning rate for Q-value updates
    learning_rate: f64,
    
    /// Discount factor for future rewards
    discount_factor: f64,
    
    /// Exploration rate
    exploration_rate: f64,
    
    /// Search depth for tree search
    search_depth: usize,
    
    /// Convergence threshold
    convergence_threshold: f64,
    
    /// Performance metrics
    metrics: AlgorithmMetrics,
}

/// State hash for efficient lookup
pub type StateHash = u64;

/// Action hash for efficient lookup  
pub type ActionHash = u64;

/// Algorithm performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmMetrics {
    pub total_updates: u64,
    pub average_q_value: f64,
    pub convergence_rate: f64,
    pub last_convergence_check: f64,
    pub performance_score: f64,
}

impl Default for AlgorithmMetrics {
    fn default() -> Self {
        Self {
            total_updates: 0,
            average_q_value: 0.0,
            convergence_rate: 0.0,
            last_convergence_check: 0.0,
            performance_score: 0.0,
        }
    }
}

impl QStarAlgorithmImpl {
    /// Create new Q* algorithm instance
    pub fn new(
        learning_rate: f64,
        discount_factor: f64,
        exploration_rate: f64,
        search_depth: usize,
        convergence_threshold: f64,
    ) -> Self {
        Self {
            q_table: HashMap::new(),
            learning_rate,
            discount_factor,
            exploration_rate,
            search_depth,
            convergence_threshold,
            metrics: AlgorithmMetrics::default(),
        }
    }
    
    /// Hash state for efficient lookup
    fn hash_state(&self, state: &MarketState) -> StateHash {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash key market indicators
        state.price.to_bits().hash(&mut hasher);
        state.volume.to_bits().hash(&mut hasher);
        state.volatility.to_bits().hash(&mut hasher);
        
        // Hash discretized technical indicators
        ((state.rsi * 100.0) as u32).hash(&mut hasher);
        ((state.macd * 1000.0) as i32).hash(&mut hasher);
        
        // Hash market regime
        state.market_regime.hash(&mut hasher);
        
        hasher.finish()
    }
    
    /// Hash action for efficient lookup
    fn hash_action(&self, action: &QStarAction) -> ActionHash {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        action.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Execute Q* search with tree exploration
    fn q_star_search<'a>(
        &'a self, 
        state: &'a MarketState, 
        depth: usize
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(QStarAction, f64), QStarError>> + Send + 'a>> {
        Box::pin(async move {
            if depth == 0 {
                // Base case: use learned Q-values
                return self.get_best_action_greedy(state).await;
            }
            
            let actions = state.get_legal_actions();
            let mut best_action = actions[0].clone();
            let mut best_value = f64::NEG_INFINITY;
            
            for action in actions {
                // Calculate Q* value recursively
                let q_value = self.calculate_q_star_value(state, &action, depth).await?;
                
                if q_value > best_value {
                    best_value = q_value;
                    best_action = action;
                }
            }
            
            Ok((best_action, best_value))
        })
    }
    
    /// Calculate Q* value with recursive search
    fn calculate_q_star_value<'a>(
        &'a self,
        state: &'a MarketState,
        action: &'a QStarAction,
        depth: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<f64, QStarError>> + Send + 'a>> {
        Box::pin(async move {
            // Get immediate reward and next state
            let next_state = state.apply_action(action)?;
            let immediate_reward = self.calculate_immediate_reward(state, action, &next_state);
            
            if depth <= 1 {
                // Base case: return immediate reward + learned Q-value
                let learned_q = self.get_q_value_from_table(state, action);
                return Ok(immediate_reward + self.discount_factor * learned_q);
            }
            
            // Recursive case: search deeper
            let (_, future_value) = self.q_star_search(&next_state, depth - 1).await?;
            
            Ok(immediate_reward + self.discount_factor * future_value)
        })
    }
    
    /// Calculate immediate reward for state transition
    fn calculate_immediate_reward(
        &self,
        _state: &MarketState,
        _action: &QStarAction,
        _next_state: &MarketState,
    ) -> f64 {
        // Placeholder for reward calculation
        // This will be implemented in q-star-trading crate
        0.0
    }
    
    /// Get Q-value from lookup table
    fn get_q_value_from_table(&self, state: &MarketState, action: &QStarAction) -> f64 {
        let state_hash = self.hash_state(state);
        let action_hash = self.hash_action(action);
        
        self.q_table.get(&(state_hash, action_hash)).copied().unwrap_or(0.0)
    }
    
    /// Get best action using greedy policy
    async fn get_best_action_greedy(&self, state: &MarketState) -> Result<(QStarAction, f64), QStarError> {
        let actions = state.get_legal_actions();
        let mut best_action = actions[0].clone();
        let mut best_value = f64::NEG_INFINITY;
        
        for action in actions {
            let q_value = self.get_q_value_from_table(state, &action);
            if q_value > best_value {
                best_value = q_value;
                best_action = action;
            }
        }
        
        Ok((best_action, best_value))
    }
    
    /// Get best action using epsilon-greedy policy
    async fn get_best_action_epsilon_greedy(&self, state: &MarketState) -> Result<(QStarAction, f64), QStarError> {
        if rand::random::<f64>() < self.exploration_rate {
            // Explore: random action
            let actions = state.get_legal_actions();
            let random_action = actions[rand::random::<usize>() % actions.len()].clone();
            let q_value = self.get_q_value_from_table(state, &random_action);
            Ok((random_action, q_value))
        } else {
            // Exploit: best known action
            self.get_best_action_greedy(state).await
        }
    }
    
    /// Update Q-table with new experience
    fn update_q_table(&mut self, experience: &Experience) {
        let state_hash = self.hash_state(&experience.state);
        let action_hash = self.hash_action(&experience.action);
        let key = (state_hash, action_hash);
        
        // Get current Q-value
        let current_q = self.q_table.get(&key).copied().unwrap_or(0.0);
        
        // Calculate target Q-value
        let next_state_hash = self.hash_state(&experience.next_state);
        let max_next_q = if experience.done {
            0.0 // Terminal state
        } else {
            // Find maximum Q-value for next state
            let next_actions = experience.next_state.get_legal_actions();
            next_actions.iter()
                .map(|a| {
                    let next_action_hash = self.hash_action(a);
                    self.q_table.get(&(next_state_hash, next_action_hash)).copied().unwrap_or(0.0)
                })
                .fold(f64::NEG_INFINITY, f64::max)
        };
        
        // Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max Q(s',a') - Q(s,a)]
        let target_q = experience.reward + self.discount_factor * max_next_q;
        let new_q = current_q + self.learning_rate * (target_q - current_q);
        
        // Update Q-table
        self.q_table.insert(key, new_q);
        
        // Update metrics
        self.metrics.total_updates += 1;
        self.update_average_q_value(new_q);
    }
    
    /// Update average Q-value metric
    fn update_average_q_value(&mut self, new_q: f64) {
        let alpha = 0.01; // Smoothing factor
        self.metrics.average_q_value = 
            alpha * new_q + (1.0 - alpha) * self.metrics.average_q_value;
    }
    
    /// Check algorithm convergence
    fn check_convergence(&mut self) -> bool {
        // Simple convergence check based on Q-value stability
        let current_avg = self.metrics.average_q_value;
        let last_avg = self.metrics.last_convergence_check;
        
        let convergence_rate = (current_avg - last_avg).abs();
        self.metrics.convergence_rate = convergence_rate;
        self.metrics.last_convergence_check = current_avg;
        
        convergence_rate < self.convergence_threshold
    }
    
    /// Get algorithm metrics
    pub fn get_metrics(&self) -> &AlgorithmMetrics {
        &self.metrics
    }
    
    /// Reset algorithm state
    pub fn reset(&mut self) {
        self.q_table.clear();
        self.metrics = AlgorithmMetrics::default();
    }
}

#[async_trait]
impl QStarAlgorithm for QStarAlgorithmImpl {
    async fn decide(&self, state: &MarketState) -> Result<QStarAction, QStarError> {
        // Use Q* search for decision making
        let (action, _) = self.q_star_search(state, self.search_depth).await?;
        Ok(action)
    }
    
    async fn update(&mut self, experience: &Experience) -> Result<(), QStarError> {
        // Update Q-table with experience
        self.update_q_table(experience);
        Ok(())
    }
    
    async fn get_q_value(&self, state: &MarketState, action: &QStarAction) -> Result<f64, QStarError> {
        Ok(self.get_q_value_from_table(state, action))
    }
    
    async fn is_converged(&self) -> Result<bool, QStarError> {
        Ok(self.metrics.convergence_rate < self.convergence_threshold)
    }
}

/// Advanced Q* algorithm with neural network integration
pub struct NeuralQStarAlgorithm {
    /// Base Q* algorithm
    base_algorithm: QStarAlgorithmImpl,
    
    /// Neural network for value approximation
    value_network: Option<Box<dyn ValueNetwork + Send + Sync>>,
    
    /// Neural network for policy approximation
    policy_network: Option<Box<dyn PolicyNetwork + Send + Sync>>,
}

/// Value network trait for neural value approximation
#[async_trait]
pub trait ValueNetwork {
    async fn evaluate(&self, state: &MarketState) -> Result<f64, QStarError>;
    async fn update(&mut self, experiences: &[Experience]) -> Result<(), QStarError>;
}

/// Policy network trait for neural policy approximation
#[async_trait]
pub trait PolicyNetwork {
    async fn get_action_probabilities(&self, state: &MarketState) -> Result<Vec<f64>, QStarError>;
    async fn update(&mut self, experiences: &[Experience]) -> Result<(), QStarError>;
}

impl NeuralQStarAlgorithm {
    /// Create new neural Q* algorithm
    pub fn new(
        base_config: (f64, f64, f64, usize, f64), // learning_rate, discount_factor, exploration_rate, search_depth, convergence_threshold
        value_network: Option<Box<dyn ValueNetwork + Send + Sync>>,
        policy_network: Option<Box<dyn PolicyNetwork + Send + Sync>>,
    ) -> Self {
        let (learning_rate, discount_factor, exploration_rate, search_depth, convergence_threshold) = base_config;
        
        Self {
            base_algorithm: QStarAlgorithmImpl::new(
                learning_rate,
                discount_factor,
                exploration_rate,
                search_depth,
                convergence_threshold,
            ),
            value_network,
            policy_network,
        }
    }
}

#[async_trait]
impl QStarAlgorithm for NeuralQStarAlgorithm {
    async fn decide(&self, state: &MarketState) -> Result<QStarAction, QStarError> {
        // Combine base algorithm with neural networks
        let base_action = self.base_algorithm.decide(state).await?;
        
        // If policy network is available, use it to refine the decision
        if let Some(policy_net) = &self.policy_network {
            let action_probs = policy_net.get_action_probabilities(state).await?;
            // TODO: Combine base_action with neural policy
        }
        
        Ok(base_action)
    }
    
    async fn update(&mut self, experience: &Experience) -> Result<(), QStarError> {
        // Update base algorithm
        self.base_algorithm.update(experience).await?;
        
        // Update neural networks if available
        let experiences = vec![experience.clone()];
        
        if let Some(value_net) = &mut self.value_network {
            value_net.update(&experiences).await?;
        }
        
        if let Some(policy_net) = &mut self.policy_network {
            policy_net.update(&experiences).await?;
        }
        
        Ok(())
    }
    
    async fn get_q_value(&self, state: &MarketState, action: &QStarAction) -> Result<f64, QStarError> {
        // Combine base Q-value with neural value if available
        let base_q = self.base_algorithm.get_q_value(state, action).await?;
        
        if let Some(value_net) = &self.value_network {
            let neural_value = value_net.evaluate(state).await?;
            // Weighted combination of base and neural values
            Ok(0.7 * base_q + 0.3 * neural_value)
        } else {
            Ok(base_q)
        }
    }
    
    async fn is_converged(&self) -> Result<bool, QStarError> {
        self.base_algorithm.is_converged().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MarketState, QStarAction, MarketRegime};
    use chrono::Utc;
    
    fn create_test_state() -> MarketState {
        MarketState {
            price: 50000.0,
            volume: 1000000.0,
            volatility: 0.02,
            rsi: 0.5,
            macd: 0.001,
            market_regime: MarketRegime::Trending,
            timestamp: Utc::now(),
            features: vec![0.1, 0.2, 0.3],
        }
    }
    
    fn create_test_experience() -> Experience {
        Experience {
            state: create_test_state(),
            action: QStarAction::Buy { amount: 0.1 },
            reward: 0.05,
            next_state: create_test_state(),
            done: false,
            timestamp: Utc::now(),
        }
    }
    
    #[tokio::test]
    async fn test_q_star_algorithm_creation() {
        let algorithm = QStarAlgorithmImpl::new(0.01, 0.99, 0.1, 3, 1e-6);
        assert_eq!(algorithm.learning_rate, 0.01);
        assert_eq!(algorithm.discount_factor, 0.99);
    }
    
    #[tokio::test]
    async fn test_q_star_algorithm_update() {
        let mut algorithm = QStarAlgorithmImpl::new(0.01, 0.99, 0.1, 3, 1e-6);
        let experience = create_test_experience();
        
        let result = algorithm.update(&experience).await;
        assert!(result.is_ok());
        assert!(algorithm.metrics.total_updates > 0);
    }
    
    #[tokio::test]
    async fn test_state_hashing() {
        let algorithm = QStarAlgorithmImpl::new(0.01, 0.99, 0.1, 3, 1e-6);
        let state1 = create_test_state();
        let state2 = create_test_state();
        
        let hash1 = algorithm.hash_state(&state1);
        let hash2 = algorithm.hash_state(&state2);
        
        assert_eq!(hash1, hash2); // Same state should have same hash
    }
    
    #[tokio::test]
    async fn test_action_hashing() {
        let algorithm = QStarAlgorithmImpl::new(0.01, 0.99, 0.1, 3, 1e-6);
        let action1 = QStarAction::Buy { amount: 0.1 };
        let action2 = QStarAction::Buy { amount: 0.1 };
        
        let hash1 = algorithm.hash_action(&action1);
        let hash2 = algorithm.hash_action(&action2);
        
        assert_eq!(hash1, hash2); // Same action should have same hash
    }
}