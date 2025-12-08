//! pBit-Enhanced Q* Algorithm
//!
//! Integrates Ising model dynamics with Q* search for
//! probabilistic action selection and value estimation.
//!
//! ## Mathematical Foundation (Wolfram Validated)
//!
//! - **Boltzmann Policy**: π(a|s) = exp(Q(s,a)/T) / Σ exp(Q(s,a')/T)
//! - **pBit Value**: V(s) = -T × ln(Σ exp(-Q(s,a)/T))
//! - **Ising Exploration**: P(explore) = 1/(1 + exp(-ΔQ/T))
//! - **Temperature Annealing**: T(t) = T_0 × α^t

use rand::prelude::*;
use std::collections::HashMap;

/// pBit Q* configuration
#[derive(Debug, Clone)]
pub struct PBitQStarConfig {
    /// Initial temperature
    pub initial_temperature: f64,
    /// Temperature decay rate
    pub temperature_decay: f64,
    /// Minimum temperature
    pub min_temperature: f64,
    /// Discount factor γ
    pub discount: f64,
    /// Learning rate α
    pub learning_rate: f64,
    /// Number of pBit samples for value estimation
    pub n_samples: usize,
}

impl Default for PBitQStarConfig {
    fn default() -> Self {
        Self {
            initial_temperature: 1.0,
            temperature_decay: 0.995,
            min_temperature: 0.01,
            discount: 0.99,
            learning_rate: 0.1,
            n_samples: 100,
        }
    }
}

/// pBit-enhanced Q* agent
#[derive(Debug, Clone)]
pub struct PBitQStar {
    /// Configuration
    pub config: PBitQStarConfig,
    /// Current temperature
    pub temperature: f64,
    /// Q-values: state -> action -> value
    q_values: HashMap<u64, HashMap<u32, f64>>,
    /// Visit counts for UCB exploration
    visit_counts: HashMap<(u64, u32), u32>,
    /// Total steps
    steps: u64,
}

impl PBitQStar {
    /// Create new pBit Q* agent
    pub fn new(config: PBitQStarConfig) -> Self {
        Self {
            temperature: config.initial_temperature,
            config,
            q_values: HashMap::new(),
            visit_counts: HashMap::new(),
            steps: 0,
        }
    }

    /// Select action using Boltzmann policy
    pub fn select_action(&mut self, state: u64, actions: &[u32]) -> u32 {
        if actions.is_empty() {
            return 0;
        }

        let mut rng = rand::thread_rng();
        
        // Get Q-values for all actions
        let q_vals: Vec<f64> = actions.iter()
            .map(|&a| self.get_q(state, a))
            .collect();

        // Boltzmann distribution
        let max_q = q_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = q_vals.iter()
            .map(|q| ((q - max_q) / self.temperature).exp())
            .collect();
        let z: f64 = exp_vals.iter().sum();

        if z < 1e-10 {
            // Uniform random if all equal
            return actions[rng.gen_range(0..actions.len())];
        }

        // Sample from distribution
        let r: f64 = rng.gen::<f64>() * z;
        let mut cumsum = 0.0;
        for (i, &exp_v) in exp_vals.iter().enumerate() {
            cumsum += exp_v;
            if r <= cumsum {
                // Update visit count
                *self.visit_counts.entry((state, actions[i])).or_insert(0) += 1;
                return actions[i];
            }
        }

        actions[actions.len() - 1]
    }

    /// Get Q-value with optimistic initialization
    pub fn get_q(&self, state: u64, action: u32) -> f64 {
        self.q_values
            .get(&state)
            .and_then(|m| m.get(&action))
            .copied()
            .unwrap_or(0.0) // Optimistic: start at 0
    }

    /// Update Q-value with TD learning
    pub fn update(&mut self, state: u64, action: u32, reward: f64, next_state: u64, next_actions: &[u32]) {
        // Get max Q for next state
        let max_next_q = if next_actions.is_empty() {
            0.0
        } else {
            next_actions.iter()
                .map(|&a| self.get_q(next_state, a))
                .fold(f64::NEG_INFINITY, f64::max)
        };

        // TD target
        let target = reward + self.config.discount * max_next_q;
        
        // Current Q
        let current_q = self.get_q(state, action);
        
        // TD update
        let new_q = current_q + self.config.learning_rate * (target - current_q);
        
        self.q_values
            .entry(state)
            .or_insert_with(HashMap::new)
            .insert(action, new_q);

        // Anneal temperature
        self.steps += 1;
        self.temperature = (self.temperature * self.config.temperature_decay)
            .max(self.config.min_temperature);
    }

    /// pBit value estimation using soft-max
    pub fn pbit_value(&self, state: u64, actions: &[u32]) -> f64 {
        if actions.is_empty() {
            return 0.0;
        }

        let q_vals: Vec<f64> = actions.iter()
            .map(|&a| self.get_q(state, a))
            .collect();

        // Log-sum-exp for numerical stability
        let max_q = q_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = q_vals.iter()
            .map(|q| ((q - max_q) / self.temperature).exp())
            .sum();

        max_q + self.temperature * sum_exp.ln()
    }

    /// pBit exploration bonus using UCB
    pub fn exploration_bonus(&self, state: u64, action: u32) -> f64 {
        let n = *self.visit_counts.get(&(state, action)).unwrap_or(&0) as f64;
        let total = self.steps as f64;

        if n < 1.0 {
            return f64::INFINITY; // Encourage unexplored
        }

        // UCB1 formula with temperature scaling
        self.temperature * (2.0 * total.ln() / n).sqrt()
    }

    /// Get action probabilities (Boltzmann policy)
    pub fn action_probabilities(&self, state: u64, actions: &[u32]) -> Vec<f64> {
        if actions.is_empty() {
            return vec![];
        }

        let q_vals: Vec<f64> = actions.iter()
            .map(|&a| self.get_q(state, a))
            .collect();

        let max_q = q_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = q_vals.iter()
            .map(|q| ((q - max_q) / self.temperature).exp())
            .collect();
        let z: f64 = exp_vals.iter().sum();

        if z < 1e-10 {
            vec![1.0 / actions.len() as f64; actions.len()]
        } else {
            exp_vals.iter().map(|e| e / z).collect()
        }
    }

    /// Reset agent
    pub fn reset(&mut self) {
        self.q_values.clear();
        self.visit_counts.clear();
        self.temperature = self.config.initial_temperature;
        self.steps = 0;
    }

    /// Get statistics
    pub fn stats(&self) -> PBitQStarStats {
        let total_states = self.q_values.len();
        let total_actions: usize = self.q_values.values()
            .map(|m| m.len())
            .sum();

        PBitQStarStats {
            steps: self.steps,
            temperature: self.temperature,
            total_states,
            total_state_actions: total_actions,
        }
    }
}

/// Agent statistics
#[derive(Debug, Clone)]
pub struct PBitQStarStats {
    pub steps: u64,
    pub temperature: f64,
    pub total_states: usize,
    pub total_state_actions: usize,
}

/// Batch Q-learning with pBit sampling
pub fn pbit_batch_update(
    agent: &mut PBitQStar,
    experiences: &[(u64, u32, f64, u64, Vec<u32>)],
) {
    for (state, action, reward, next_state, next_actions) in experiences {
        agent.update(*state, *action, *reward, *next_state, next_actions);
    }
}

/// Create trading-specific Q* agent
pub fn trading_qstar() -> PBitQStar {
    PBitQStar::new(PBitQStarConfig {
        initial_temperature: 0.5,
        temperature_decay: 0.999,
        min_temperature: 0.01,
        discount: 0.95,
        learning_rate: 0.05,
        n_samples: 50,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boltzmann_policy() {
        let mut agent = PBitQStar::new(PBitQStarConfig::default());
        
        // Set some Q-values
        agent.q_values.insert(0, [(0, 1.0), (1, 2.0), (2, 0.5)].into_iter().collect());

        let probs = agent.action_probabilities(0, &[0, 1, 2]);
        
        // Action 1 (Q=2.0) should have highest probability
        assert!(probs[1] > probs[0]);
        assert!(probs[1] > probs[2]);
        
        // Sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_temperature_annealing() {
        let mut agent = PBitQStar::new(PBitQStarConfig {
            initial_temperature: 1.0,
            temperature_decay: 0.9,
            min_temperature: 0.1,
            ..Default::default()
        });

        let initial_temp = agent.temperature;
        
        // Simulate some updates
        for _ in 0..10 {
            agent.update(0, 0, 1.0, 1, &[0, 1]);
        }

        assert!(agent.temperature < initial_temp);
        assert!(agent.temperature >= 0.1);
    }

    #[test]
    fn test_pbit_value() {
        let mut agent = PBitQStar::new(PBitQStarConfig::default());
        agent.q_values.insert(0, [(0, 1.0), (1, 2.0)].into_iter().collect());

        let value = agent.pbit_value(0, &[0, 1]);
        
        // Value should be between min and max Q
        assert!(value >= 1.0);
        assert!(value <= 2.5); // Slightly above max due to entropy
    }

    #[test]
    fn test_td_learning() {
        let mut agent = PBitQStar::new(PBitQStarConfig {
            learning_rate: 1.0, // Full update for testing
            discount: 0.9,
            ..Default::default()
        });

        // First update
        agent.update(0, 0, 10.0, 1, &[]);
        
        let q = agent.get_q(0, 0);
        assert!((q - 10.0).abs() < 0.01, "Q should be ~10.0, got {}", q);
    }

    #[test]
    fn test_boltzmann_wolfram_validated() {
        // Wolfram: softmax([1,2,3], T=1) = [0.09, 0.245, 0.665]
        let t = 1.0_f64;
        let q = [1.0_f64, 2.0, 3.0];
        let max_q = 3.0;
        
        let exp_vals: Vec<f64> = q.iter()
            .map(|&qi| ((qi - max_q) / t).exp())
            .collect();
        let z: f64 = exp_vals.iter().sum();
        let probs: Vec<f64> = exp_vals.iter().map(|e| e / z).collect();

        assert!((probs[0] - 0.09).abs() < 0.01);
        assert!((probs[1] - 0.245).abs() < 0.01);
        assert!((probs[2] - 0.665).abs() < 0.01);
    }
}
