//! Load balancing for expert utilization
//!
//! Implements auxiliary losses and capacity management for MoE systems.

// Error types available for future extension
#[allow(unused_imports)]
use crate::{RouterError, Result};
use serde::{Deserialize, Serialize};

/// Configuration for load balancing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerConfig {
    /// Number of experts
    pub num_experts: usize,
    /// Capacity factor (tokens per expert / tokens per batch)
    pub capacity_factor: f64,
    /// Coefficient for auxiliary balance loss
    pub balance_loss_coef: f64,
    /// Coefficient for router z-loss
    pub router_z_loss_coef: f64,
    /// Whether to use auxiliary losses
    pub use_auxiliary_loss: bool,
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            capacity_factor: 1.25,
            balance_loss_coef: 0.01,
            router_z_loss_coef: 0.001,
            use_auxiliary_loss: true,
        }
    }
}

/// Load balancer for expert utilization
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    config: LoadBalancerConfig,
    /// Current load per expert
    expert_loads: Vec<f64>,
    /// Total tokens routed
    total_tokens: usize,
    /// Historical balance loss (for trend analysis)
    #[allow(dead_code)]
    balance_loss_history: Vec<f64>,
}

impl LoadBalancer {
    /// Create new load balancer
    pub fn new(config: LoadBalancerConfig) -> Self {
        let expert_loads = vec![0.0; config.num_experts];
        Self {
            config,
            expert_loads,
            total_tokens: 0,
            balance_loss_history: Vec::new(),
        }
    }

    /// Update loads with new routing decision
    ///
    /// # Arguments
    ///
    /// * `expert_indices` - Selected expert indices
    /// * `weights` - Routing weights for each selected expert
    pub fn update_loads(&mut self, routing: &[(usize, f64)]) {
        for &(expert_id, weight) in routing {
            if expert_id < self.expert_loads.len() {
                self.expert_loads[expert_id] += weight;
            }
        }
        self.total_tokens += 1;
    }

    /// Compute auxiliary balance loss
    ///
    /// Encourages uniform expert utilization to prevent collapse.
    ///
    /// Loss = α · n · Σᵢ fᵢ · pᵢ
    ///
    /// where fᵢ is fraction of tokens routed to expert i,
    /// and pᵢ is average routing probability for expert i.
    pub fn compute_balance_loss(&self, routing_probs: &[Vec<f64>]) -> f64 {
        if routing_probs.is_empty() || self.total_tokens == 0 {
            return 0.0;
        }

        let n = self.config.num_experts as f64;

        // Compute fraction of tokens to each expert
        let total = self.total_tokens as f64;
        let fractions: Vec<f64> = self.expert_loads.iter().map(|&l| l / total).collect();

        // Compute average probability for each expert
        let batch_size = routing_probs.len();
        let mut avg_probs = vec![0.0; self.config.num_experts];

        for probs in routing_probs {
            for (i, &p) in probs.iter().enumerate() {
                if i < avg_probs.len() {
                    avg_probs[i] += p;
                }
            }
        }

        for p in &mut avg_probs {
            *p /= batch_size as f64;
        }

        // Balance loss: encourages uniform distribution
        let loss: f64 = fractions.iter()
            .zip(avg_probs.iter())
            .map(|(&f, &p)| f * p)
            .sum();

        self.config.balance_loss_coef * n * loss
    }

    /// Compute router z-loss
    ///
    /// Penalizes very confident routing to encourage exploration.
    ///
    /// z_loss = β · (1/n) · Σᵢ log(Σⱼ exp(logits_ij))²
    pub fn compute_z_loss(&self, logits_batch: &[Vec<f64>]) -> f64 {
        if logits_batch.is_empty() {
            return 0.0;
        }

        let mut z_loss = 0.0;

        for logits in logits_batch {
            // Log-sum-exp
            let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lse: f64 = logits.iter().map(|&l| (l - max_logit).exp()).sum::<f64>().ln() + max_logit;
            z_loss += lse.powi(2);
        }

        self.config.router_z_loss_coef * z_loss / (logits_batch.len() as f64)
    }

    /// Check if expert is over capacity
    pub fn is_over_capacity(&self, expert_id: usize) -> bool {
        if expert_id >= self.expert_loads.len() || self.total_tokens == 0 {
            return false;
        }

        let expected_load = self.total_tokens as f64 / self.config.num_experts as f64;
        let capacity = expected_load * self.config.capacity_factor;

        self.expert_loads[expert_id] > capacity
    }

    /// Get current load for expert
    pub fn expert_load(&self, expert_id: usize) -> f64 {
        self.expert_loads.get(expert_id).copied().unwrap_or(0.0)
    }

    /// Get load imbalance metric
    ///
    /// Returns coefficient of variation of expert loads.
    pub fn load_imbalance(&self) -> f64 {
        if self.total_tokens == 0 {
            return 0.0;
        }

        let n = self.config.num_experts as f64;
        let mean: f64 = self.expert_loads.iter().sum::<f64>() / n;

        if mean < 1e-10 {
            return 0.0;
        }

        let variance: f64 = self.expert_loads.iter()
            .map(|&l| (l - mean).powi(2))
            .sum::<f64>() / n;

        variance.sqrt() / mean
    }

    /// Get total auxiliary loss
    pub fn total_auxiliary_loss(&self, routing_probs: &[Vec<f64>], logits: &[Vec<f64>]) -> f64 {
        if !self.config.use_auxiliary_loss {
            return 0.0;
        }

        self.compute_balance_loss(routing_probs) + self.compute_z_loss(logits)
    }

    /// Reset load counters
    pub fn reset(&mut self) {
        self.expert_loads.fill(0.0);
        self.total_tokens = 0;
    }

    /// Get load summary
    pub fn summary(&self) -> LoadSummary {
        LoadSummary {
            total_tokens: self.total_tokens,
            expert_loads: self.expert_loads.clone(),
            imbalance: self.load_imbalance(),
            over_capacity_count: (0..self.config.num_experts)
                .filter(|&i| self.is_over_capacity(i))
                .count(),
        }
    }
}

/// Summary of load balancer state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadSummary {
    /// Total tokens processed
    pub total_tokens: usize,
    /// Load per expert
    pub expert_loads: Vec<f64>,
    /// Load imbalance metric
    pub imbalance: f64,
    /// Number of over-capacity experts
    pub over_capacity_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_balancer_creation() {
        let lb = LoadBalancer::new(LoadBalancerConfig::default());
        assert_eq!(lb.expert_loads.len(), 8);
    }

    #[test]
    fn test_load_update() {
        let mut lb = LoadBalancer::new(LoadBalancerConfig {
            num_experts: 4,
            ..Default::default()
        });

        lb.update_loads(&[(0, 0.6), (1, 0.4)]);

        assert!((lb.expert_load(0) - 0.6).abs() < 1e-10);
        assert!((lb.expert_load(1) - 0.4).abs() < 1e-10);
        assert_eq!(lb.expert_load(2), 0.0);
    }

    #[test]
    fn test_balanced_loads() {
        let mut lb = LoadBalancer::new(LoadBalancerConfig {
            num_experts: 4,
            ..Default::default()
        });

        // Perfectly balanced routing
        for _ in 0..100 {
            lb.update_loads(&[(0, 0.25), (1, 0.25), (2, 0.25), (3, 0.25)]);
        }

        // Imbalance should be near zero
        assert!(lb.load_imbalance() < 0.01);
    }

    #[test]
    fn test_imbalanced_loads() {
        let mut lb = LoadBalancer::new(LoadBalancerConfig {
            num_experts: 4,
            ..Default::default()
        });

        // All to expert 0
        for _ in 0..100 {
            lb.update_loads(&[(0, 1.0)]);
        }

        // Should be highly imbalanced
        assert!(lb.load_imbalance() > 1.0);
    }

    #[test]
    fn test_over_capacity() {
        let mut lb = LoadBalancer::new(LoadBalancerConfig {
            num_experts: 4,
            capacity_factor: 1.25,
            ..Default::default()
        });

        // Route everything to expert 0
        for _ in 0..100 {
            lb.update_loads(&[(0, 1.0)]);
        }

        assert!(lb.is_over_capacity(0));
        assert!(!lb.is_over_capacity(1));
    }

    #[test]
    fn test_balance_loss() {
        let mut lb = LoadBalancer::new(LoadBalancerConfig {
            num_experts: 4,
            balance_loss_coef: 0.01,
            ..Default::default()
        });

        lb.update_loads(&[(0, 1.0)]);

        let probs = vec![vec![0.5, 0.3, 0.1, 0.1]];
        let loss = lb.compute_balance_loss(&probs);

        // Should have non-zero loss for imbalanced routing
        assert!(loss > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut lb = LoadBalancer::new(LoadBalancerConfig::default());

        lb.update_loads(&[(0, 1.0)]);
        lb.update_loads(&[(1, 0.5)]);

        lb.reset();

        assert_eq!(lb.total_tokens, 0);
        assert!(lb.expert_loads.iter().all(|&l| l == 0.0));
    }
}
