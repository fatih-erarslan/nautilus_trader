//! Regret minimization algorithms for hedge systems

use std::collections::HashMap;
use crate::{HedgeError, HedgeConfig};

/// Regret minimization framework
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RegretMinimizer {
    /// External regret per expert
    pub external_regret: HashMap<String, f64>,
    /// Internal regret matrix
    pub internal_regret: HashMap<String, HashMap<String, f64>>,
    /// Cumulative regret
    pub cumulative_regret: f64,
    /// Regret bounds
    pub regret_bounds: HashMap<String, (f64, f64)>,
    /// Time step
    pub time_step: usize,
    /// Configuration
    config: HedgeConfig,
}

impl RegretMinimizer {
    /// Create new regret minimizer
    pub fn new(config: HedgeConfig) -> Self {
        Self {
            external_regret: HashMap::new(),
            internal_regret: HashMap::new(),
            cumulative_regret: 0.0,
            regret_bounds: HashMap::new(),
            time_step: 0,
            config,
        }
    }
    
    /// Initialize expert regret tracking
    pub fn initialize_expert(&mut self, expert_name: &str) -> Result<(), HedgeError> {
        self.external_regret.insert(expert_name.to_string(), 0.0);
        self.internal_regret.insert(expert_name.to_string(), HashMap::new());
        self.regret_bounds.insert(expert_name.to_string(), (f64::NEG_INFINITY, f64::INFINITY));
        Ok(())
    }
    
    /// Update external regret
    pub fn update_external_regret(
        &mut self,
        expert_predictions: &HashMap<String, f64>,
        portfolio_prediction: f64,
        actual_outcome: f64,
    ) -> Result<(), HedgeError> {
        for (expert_name, prediction) in expert_predictions {
            let expert_loss = (prediction - actual_outcome).powi(2);
            let portfolio_loss = (portfolio_prediction - actual_outcome).powi(2);
            
            let regret = portfolio_loss - expert_loss;
            
            self.external_regret
                .entry(expert_name.clone())
                .and_modify(|r| *r += regret)
                .or_insert(regret);
        }
        
        Ok(())
    }
    
    /// Update internal regret
    pub fn update_internal_regret(
        &mut self,
        expert_predictions: &HashMap<String, f64>,
        expert_weights: &HashMap<String, f64>,
        actual_outcome: f64,
    ) -> Result<(), HedgeError> {
        for (expert_i, prediction_i) in expert_predictions {
            let weight_i = expert_weights.get(expert_i).unwrap_or(&0.0);
            
            for (expert_j, prediction_j) in expert_predictions {
                if expert_i != expert_j {
                    let weight_j = expert_weights.get(expert_j).unwrap_or(&0.0);
                    
                    let loss_i = (prediction_i - actual_outcome).powi(2);
                    let loss_j = (prediction_j - actual_outcome).powi(2);
                    
                    let regret = weight_i * loss_i - weight_j * loss_j;
                    
                    self.internal_regret
                        .entry(expert_i.clone())
                        .or_insert_with(HashMap::new)
                        .entry(expert_j.clone())
                        .and_modify(|r| *r += regret)
                        .or_insert(regret);
                }
            }
        }
        
        Ok(())
    }
    
    /// Update cumulative regret
    pub fn update_cumulative_regret(
        &mut self,
        expert_predictions: &HashMap<String, f64>,
        portfolio_prediction: f64,
        actual_outcome: f64,
    ) -> Result<(), HedgeError> {
        let portfolio_loss = (portfolio_prediction - actual_outcome).powi(2);
        
        let best_expert_loss = expert_predictions
            .values()
            .map(|prediction| (prediction - actual_outcome).powi(2))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        
        let regret = portfolio_loss - best_expert_loss;
        self.cumulative_regret += regret;
        
        Ok(())
    }
    
    /// Get regret bounds
    pub fn get_regret_bound(&self, _expert_name: &str) -> Result<f64, HedgeError> {
        let t = self.time_step as f64;
        
        if t <= 0.0 {
            return Ok(0.0);
        }
        
        // Theoretical regret bound for exponential weights
        let bound = (t.ln() / self.config.learning_rate).sqrt();
        
        Ok(bound)
    }
    
    /// Check if regret is within bounds
    pub fn is_regret_bounded(&self, expert_name: &str) -> Result<bool, HedgeError> {
        let current_regret = self.external_regret.get(expert_name).unwrap_or(&0.0);
        let bound = self.get_regret_bound(expert_name)?;
        
        Ok(current_regret.abs() <= bound)
    }
    
    /// Get regret statistics
    pub fn get_regret_statistics(&self) -> RegretStatistics {
        let total_experts = self.external_regret.len();
        
        let max_external_regret = self.external_regret
            .values()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);
        
        let min_external_regret = self.external_regret
            .values()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);
        
        let avg_external_regret = if total_experts > 0 {
            self.external_regret.values().sum::<f64>() / total_experts as f64
        } else {
            0.0
        };
        
        let max_internal_regret = self.internal_regret
            .values()
            .flat_map(|inner| inner.values())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);
        
        let avg_cumulative_regret = if self.time_step > 0 {
            self.cumulative_regret / self.time_step as f64
        } else {
            0.0
        };
        
        RegretStatistics {
            total_experts,
            max_external_regret,
            min_external_regret,
            avg_external_regret,
            max_internal_regret,
            cumulative_regret: self.cumulative_regret,
            avg_cumulative_regret,
            time_step: self.time_step,
        }
    }
    
    /// Reset regret tracking
    pub fn reset(&mut self) -> Result<(), HedgeError> {
        self.external_regret.clear();
        self.internal_regret.clear();
        self.cumulative_regret = 0.0;
        self.regret_bounds.clear();
        self.time_step = 0;
        
        Ok(())
    }
    
    /// Update time step
    pub fn update_time_step(&mut self) {
        self.time_step += 1;
    }
    
    /// Get external regret for expert
    pub fn get_external_regret(&self, expert_name: &str) -> Option<f64> {
        self.external_regret.get(expert_name).copied()
    }
    
    /// Get internal regret between experts
    pub fn get_internal_regret(&self, expert_i: &str, expert_j: &str) -> Option<f64> {
        self.internal_regret
            .get(expert_i)?
            .get(expert_j)
            .copied()
    }
    
    /// Get cumulative regret
    pub fn get_cumulative_regret(&self) -> f64 {
        self.cumulative_regret
    }
    
    /// Get average regret per time step
    pub fn get_average_regret(&self) -> f64 {
        if self.time_step > 0 {
            self.cumulative_regret / self.time_step as f64
        } else {
            0.0
        }
    }
    
    /// Get regret variance
    pub fn get_regret_variance(&self) -> f64 {
        let regrets: Vec<f64> = self.external_regret.values().copied().collect();
        
        if regrets.is_empty() {
            return 0.0;
        }
        
        let mean = regrets.iter().sum::<f64>() / regrets.len() as f64;
        let variance = regrets.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / regrets.len() as f64;
        
        variance
    }
}

/// Regret statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RegretStatistics {
    /// Total number of experts
    pub total_experts: usize,
    /// Maximum external regret
    pub max_external_regret: f64,
    /// Minimum external regret
    pub min_external_regret: f64,
    /// Average external regret
    pub avg_external_regret: f64,
    /// Maximum internal regret
    pub max_internal_regret: f64,
    /// Cumulative regret
    pub cumulative_regret: f64,
    /// Average cumulative regret
    pub avg_cumulative_regret: f64,
    /// Time step
    pub time_step: usize,
}

/// Online gradient descent regret minimizer
#[derive(Debug, Clone)]
pub struct OnlineGradientDescent {
    /// Weights
    weights: HashMap<String, f64>,
    /// Learning rate
    learning_rate: f64,
    /// Gradient history
    gradient_history: HashMap<String, Vec<f64>>,
    /// Time step
    time_step: usize,
}

impl OnlineGradientDescent {
    /// Create new online gradient descent
    pub fn new(learning_rate: f64) -> Self {
        Self {
            weights: HashMap::new(),
            learning_rate,
            gradient_history: HashMap::new(),
            time_step: 0,
        }
    }
    
    /// Initialize expert
    pub fn initialize_expert(&mut self, expert_name: &str, initial_weight: f64) -> Result<(), HedgeError> {
        self.weights.insert(expert_name.to_string(), initial_weight);
        self.gradient_history.insert(expert_name.to_string(), Vec::new());
        Ok(())
    }
    
    /// Update weights with gradient
    pub fn update_weights(
        &mut self,
        expert_predictions: &HashMap<String, f64>,
        actual_outcome: f64,
    ) -> Result<(), HedgeError> {
        for (expert_name, prediction) in expert_predictions {
            let gradient = 2.0 * (prediction - actual_outcome); // Gradient of squared loss
            
            self.gradient_history
                .entry(expert_name.clone())
                .or_insert_with(Vec::new)
                .push(gradient);
            
            let weight = self.weights.get(expert_name).unwrap_or(&0.0);
            let new_weight = weight - self.learning_rate * gradient;
            
            self.weights.insert(expert_name.clone(), new_weight.max(0.0));
        }
        
        // Normalize weights
        let total_weight: f64 = self.weights.values().sum();
        if total_weight > 0.0 {
            for weight in self.weights.values_mut() {
                *weight /= total_weight;
            }
        }
        
        self.time_step += 1;
        
        Ok(())
    }
    
    /// Get current weights
    pub fn get_weights(&self) -> &HashMap<String, f64> {
        &self.weights
    }
    
    /// Get regret bound
    pub fn get_regret_bound(&self) -> f64 {
        let t = self.time_step as f64;
        
        if t <= 0.0 {
            return 0.0;
        }
        
        // Regret bound for online gradient descent
        (t / self.learning_rate).sqrt()
    }
}

/// Follow the Regularized Leader
#[derive(Debug, Clone)]
pub struct FollowTheRegularizedLeader {
    /// Weights
    weights: HashMap<String, f64>,
    /// Regularization parameter
    regularization: f64,
    /// Loss history
    loss_history: HashMap<String, Vec<f64>>,
    /// Time step
    time_step: usize,
}

impl FollowTheRegularizedLeader {
    /// Create new FTRL
    pub fn new(regularization: f64) -> Self {
        Self {
            weights: HashMap::new(),
            regularization,
            loss_history: HashMap::new(),
            time_step: 0,
        }
    }
    
    /// Initialize expert
    pub fn initialize_expert(&mut self, expert_name: &str) -> Result<(), HedgeError> {
        self.weights.insert(expert_name.to_string(), 0.0);
        self.loss_history.insert(expert_name.to_string(), Vec::new());
        Ok(())
    }
    
    /// Update weights
    pub fn update_weights(
        &mut self,
        expert_predictions: &HashMap<String, f64>,
        actual_outcome: f64,
    ) -> Result<(), HedgeError> {
        // Update loss history
        for (expert_name, prediction) in expert_predictions {
            let loss = (prediction - actual_outcome).powi(2);
            
            self.loss_history
                .entry(expert_name.clone())
                .or_insert_with(Vec::new)
                .push(loss);
        }
        
        // Compute new weights using FTRL
        let mut new_weights = HashMap::new();
        
        for expert_name in self.weights.keys() {
            let losses = self.loss_history.get(expert_name).unwrap();
            let cumulative_loss: f64 = losses.iter().sum();
            
            let weight = (-cumulative_loss / self.regularization).exp();
            new_weights.insert(expert_name.clone(), weight);
        }
        
        // Normalize weights
        let total_weight: f64 = new_weights.values().sum();
        if total_weight > 0.0 {
            for weight in new_weights.values_mut() {
                *weight /= total_weight;
            }
        }
        
        self.weights = new_weights;
        self.time_step += 1;
        
        Ok(())
    }
    
    /// Get current weights
    pub fn get_weights(&self) -> &HashMap<String, f64> {
        &self.weights
    }
    
    /// Get regret bound
    pub fn get_regret_bound(&self) -> f64 {
        let t = self.time_step as f64;
        
        if t <= 0.0 {
            return 0.0;
        }
        
        // Regret bound for FTRL
        (t * self.regularization.ln()).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regret_minimizer_creation() {
        let config = HedgeConfig::default();
        let regret_minimizer = RegretMinimizer::new(config);
        
        assert_eq!(regret_minimizer.external_regret.len(), 0);
        assert_eq!(regret_minimizer.cumulative_regret, 0.0);
        assert_eq!(regret_minimizer.time_step, 0);
    }

    #[test]
    fn test_regret_minimizer_expert_initialization() {
        let config = HedgeConfig::default();
        let mut regret_minimizer = RegretMinimizer::new(config);
        
        regret_minimizer.initialize_expert("expert1").unwrap();
        regret_minimizer.initialize_expert("expert2").unwrap();
        
        assert_eq!(regret_minimizer.external_regret.len(), 2);
        assert!(regret_minimizer.external_regret.contains_key("expert1"));
        assert!(regret_minimizer.external_regret.contains_key("expert2"));
    }

    #[test]
    fn test_external_regret_update() {
        let config = HedgeConfig::default();
        let mut regret_minimizer = RegretMinimizer::new(config);
        
        regret_minimizer.initialize_expert("expert1").unwrap();
        regret_minimizer.initialize_expert("expert2").unwrap();
        
        let mut expert_predictions = HashMap::new();
        expert_predictions.insert("expert1".to_string(), 0.05);
        expert_predictions.insert("expert2".to_string(), -0.02);
        
        let portfolio_prediction = 0.015;
        let actual_outcome = 0.03;
        
        regret_minimizer.update_external_regret(
            &expert_predictions,
            portfolio_prediction,
            actual_outcome,
        ).unwrap();
        
        assert!(regret_minimizer.external_regret.get("expert1").unwrap() != &0.0);
        assert!(regret_minimizer.external_regret.get("expert2").unwrap() != &0.0);
    }

    #[test]
    fn test_online_gradient_descent() {
        let mut ogd = OnlineGradientDescent::new(0.01);
        
        ogd.initialize_expert("expert1", 0.5).unwrap();
        ogd.initialize_expert("expert2", 0.5).unwrap();
        
        let mut expert_predictions = HashMap::new();
        expert_predictions.insert("expert1".to_string(), 0.05);
        expert_predictions.insert("expert2".to_string(), -0.02);
        
        let actual_outcome = 0.03;
        
        ogd.update_weights(&expert_predictions, actual_outcome).unwrap();
        
        let weights = ogd.get_weights();
        assert_eq!(weights.len(), 2);
        
        let total_weight: f64 = weights.values().sum();
        assert!((total_weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ftrl() {
        let mut ftrl = FollowTheRegularizedLeader::new(0.1);
        
        ftrl.initialize_expert("expert1").unwrap();
        ftrl.initialize_expert("expert2").unwrap();
        
        let mut expert_predictions = HashMap::new();
        expert_predictions.insert("expert1".to_string(), 0.05);
        expert_predictions.insert("expert2".to_string(), -0.02);
        
        let actual_outcome = 0.03;
        
        ftrl.update_weights(&expert_predictions, actual_outcome).unwrap();
        
        let weights = ftrl.get_weights();
        assert_eq!(weights.len(), 2);
        
        let total_weight: f64 = weights.values().sum();
        assert!((total_weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_regret_statistics() {
        let config = HedgeConfig::default();
        let mut regret_minimizer = RegretMinimizer::new(config);
        
        regret_minimizer.initialize_expert("expert1").unwrap();
        regret_minimizer.initialize_expert("expert2").unwrap();
        
        let mut expert_predictions = HashMap::new();
        expert_predictions.insert("expert1".to_string(), 0.05);
        expert_predictions.insert("expert2".to_string(), -0.02);
        
        let portfolio_prediction = 0.015;
        let actual_outcome = 0.03;
        
        regret_minimizer.update_external_regret(
            &expert_predictions,
            portfolio_prediction,
            actual_outcome,
        ).unwrap();
        
        let stats = regret_minimizer.get_regret_statistics();
        assert_eq!(stats.total_experts, 2);
        assert!(stats.max_external_regret != 0.0 || stats.min_external_regret != 0.0);
    }
}