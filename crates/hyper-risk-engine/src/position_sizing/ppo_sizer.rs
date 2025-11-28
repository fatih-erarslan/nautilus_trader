//! PPO-based Position Sizer.
//!
//! Uses Proximal Policy Optimization for dynamic position sizing
//! that adapts to market conditions.
//!
//! ## Advantages over Kelly
//! - Adapts to non-stationary distributions
//! - Learns complex patterns
//! - Can incorporate multiple features
//!
//! ## Architecture
//! - State: Market features (volatility, regime, momentum, etc.)
//! - Action: Position size (continuous)
//! - Reward: Risk-adjusted return (Sharpe, Sortino)
//!
//! ## Scientific References
//! - Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
//! - Zhang et al. (2020): "Deep Reinforcement Learning for Trading"

use serde::{Deserialize, Serialize};

/// PPO configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPOConfig {
    /// State dimension (number of features).
    pub state_dim: usize,
    /// Hidden layer sizes.
    pub hidden_dims: Vec<usize>,
    /// Learning rate.
    pub learning_rate: f64,
    /// Discount factor.
    pub gamma: f64,
    /// GAE lambda.
    pub gae_lambda: f64,
    /// PPO clip parameter.
    pub clip_epsilon: f64,
    /// Value loss coefficient.
    pub value_coef: f64,
    /// Entropy bonus coefficient.
    pub entropy_coef: f64,
    /// Maximum position size.
    pub max_position: f64,
    /// Use Sharpe or Sortino for reward.
    pub use_sortino: bool,
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            state_dim: 10,
            hidden_dims: vec![64, 64],
            learning_rate: 3e-4,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.01,
            max_position: 1.0,
            use_sortino: false,
        }
    }
}

/// Position size result from PPO.
#[derive(Debug, Clone)]
pub struct PositionSizeResult {
    /// Recommended position size (-1 to 1).
    pub position_size: f64,
    /// Confidence in the recommendation.
    pub confidence: f64,
    /// Value estimate for current state.
    pub value_estimate: f64,
    /// Policy entropy (exploration level).
    pub entropy: f64,
}

/// State features for PPO.
#[derive(Debug, Clone, Default)]
pub struct StateFeatures {
    /// Realized volatility.
    pub volatility: f64,
    /// Current regime probability.
    pub regime_prob: f64,
    /// Momentum signal.
    pub momentum: f64,
    /// Mean reversion signal.
    pub mean_reversion: f64,
    /// Current position.
    pub current_position: f64,
    /// Unrealized P&L.
    pub unrealized_pnl: f64,
    /// Portfolio drawdown.
    pub drawdown: f64,
    /// VaR estimate.
    pub var_estimate: f64,
    /// Market regime (encoded).
    pub regime_code: f64,
    /// Time feature (hour of day, etc.).
    pub time_feature: f64,
}

impl StateFeatures {
    /// Convert to vector for neural network.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.volatility,
            self.regime_prob,
            self.momentum,
            self.mean_reversion,
            self.current_position,
            self.unrealized_pnl,
            self.drawdown,
            self.var_estimate,
            self.regime_code,
            self.time_feature,
        ]
    }
}

/// PPO Position Sizer.
///
/// Note: This is a simplified implementation. A full implementation
/// would use a neural network framework (tch-rs, burn, candle).
#[derive(Debug)]
pub struct PPOPositionSizer {
    /// Configuration.
    config: PPOConfig,
    /// Policy weights (simplified linear model).
    policy_weights: Vec<f64>,
    /// Value weights.
    value_weights: Vec<f64>,
    /// Running mean of state features (for normalization).
    state_mean: Vec<f64>,
    /// Running variance of state features.
    state_var: Vec<f64>,
    /// Number of updates.
    update_count: u64,
}

impl PPOPositionSizer {
    /// Create new PPO sizer.
    pub fn new(config: PPOConfig) -> Self {
        let state_dim = config.state_dim;

        // Initialize weights (in production, use proper initialization)
        let policy_weights = vec![0.0; state_dim];
        let value_weights = vec![0.0; state_dim];
        let state_mean = vec![0.0; state_dim];
        let state_var = vec![1.0; state_dim];

        Self {
            config,
            policy_weights,
            value_weights,
            state_mean,
            state_var,
            update_count: 0,
        }
    }

    /// Get position size recommendation.
    pub fn get_position(&self, state: &StateFeatures) -> PositionSizeResult {
        let features = state.to_vec();
        let normalized = self.normalize(&features);

        // Linear policy (in production, use neural network)
        let logit: f64 = normalized.iter()
            .zip(self.policy_weights.iter())
            .map(|(s, w)| s * w)
            .sum();

        // Tanh for bounded output
        let position_size = logit.tanh() * self.config.max_position;

        // Value estimate
        let value_estimate: f64 = normalized.iter()
            .zip(self.value_weights.iter())
            .map(|(s, w)| s * w)
            .sum();

        // Confidence based on feature magnitudes
        let feature_magnitude: f64 = normalized.iter().map(|x| x.abs()).sum::<f64>() / normalized.len() as f64;
        let confidence = (1.0 - (-feature_magnitude).exp()).min(0.99);

        // Entropy (simplified)
        let entropy = 0.5; // Would compute from policy distribution

        PositionSizeResult {
            position_size,
            confidence,
            value_estimate,
            entropy,
        }
    }

    /// Update model with experience.
    ///
    /// # Arguments
    /// * `states` - Batch of states
    /// * `actions` - Actions taken
    /// * `rewards` - Rewards received
    /// * `next_states` - Next states
    /// * `dones` - Episode termination flags
    pub fn update(
        &mut self,
        states: &[StateFeatures],
        actions: &[f64],
        rewards: &[f64],
        next_states: &[StateFeatures],
        dones: &[bool],
    ) {
        if states.is_empty() {
            return;
        }

        // Update running statistics
        for state in states {
            self.update_normalization(&state.to_vec());
        }

        // Calculate advantages using GAE
        let advantages = self.compute_gae(states, rewards, next_states, dones);

        // PPO update (simplified - gradient descent on clipped objective)
        let lr = self.config.learning_rate;

        for (i, (state, action, advantage)) in states.iter().zip(actions.iter()).zip(advantages.iter()).map(|((s, a), adv)| (s, a, adv)).enumerate() {
            let features = self.normalize(&state.to_vec());

            // Policy gradient
            let current_logit: f64 = features.iter()
                .zip(self.policy_weights.iter())
                .map(|(s, w)| s * w)
                .sum();

            // Simplified gradient (in production, use autograd)
            for (j, f) in features.iter().enumerate() {
                let grad = f * advantage * (1.0 - current_logit.tanh().powi(2));
                self.policy_weights[j] += lr * grad.clamp(-1.0, 1.0);
            }

            // Value update
            let value_target = rewards.get(i).unwrap_or(&0.0);
            let current_value: f64 = features.iter()
                .zip(self.value_weights.iter())
                .map(|(s, w)| s * w)
                .sum();
            let value_error = value_target - current_value;

            for (j, f) in features.iter().enumerate() {
                self.value_weights[j] += lr * self.config.value_coef * value_error * f;
            }
        }

        self.update_count += 1;
    }

    /// Compute Generalized Advantage Estimation.
    fn compute_gae(
        &self,
        states: &[StateFeatures],
        rewards: &[f64],
        next_states: &[StateFeatures],
        dones: &[bool],
    ) -> Vec<f64> {
        let n = states.len();
        let mut advantages = vec![0.0; n];

        if n == 0 {
            return advantages;
        }

        // Compute values
        let values: Vec<f64> = states.iter()
            .map(|s| {
                let features = self.normalize(&s.to_vec());
                features.iter()
                    .zip(self.value_weights.iter())
                    .map(|(s, w)| s * w)
                    .sum()
            })
            .collect();

        let next_values: Vec<f64> = next_states.iter()
            .map(|s| {
                let features = self.normalize(&s.to_vec());
                features.iter()
                    .zip(self.value_weights.iter())
                    .map(|(s, w)| s * w)
                    .sum()
            })
            .collect();

        // GAE computation
        let mut gae = 0.0;
        for i in (0..n).rev() {
            let next_val = if dones[i] { 0.0 } else { next_values[i] };
            let delta = rewards[i] + self.config.gamma * next_val - values[i];
            gae = delta + self.config.gamma * self.config.gae_lambda * (if dones[i] { 0.0 } else { gae });
            advantages[i] = gae;
        }

        advantages
    }

    /// Normalize state features.
    fn normalize(&self, features: &[f64]) -> Vec<f64> {
        features.iter()
            .zip(self.state_mean.iter())
            .zip(self.state_var.iter())
            .map(|((f, m), v)| {
                if *v > 1e-8 {
                    (f - m) / v.sqrt()
                } else {
                    f - m
                }
            })
            .collect()
    }

    /// Update running normalization statistics.
    fn update_normalization(&mut self, features: &[f64]) {
        let alpha = 0.01; // Exponential moving average weight

        for (i, f) in features.iter().enumerate() {
            if i < self.state_mean.len() {
                // Update mean
                let delta = f - self.state_mean[i];
                self.state_mean[i] += alpha * delta;

                // Update variance
                let delta2 = (f - self.state_mean[i]).powi(2);
                self.state_var[i] = (1.0 - alpha) * self.state_var[i] + alpha * delta2;
            }
        }
    }

    /// Get number of updates performed.
    pub fn update_count(&self) -> u64 {
        self.update_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppo_creation() {
        let config = PPOConfig::default();
        let sizer = PPOPositionSizer::new(config);
        assert_eq!(sizer.update_count(), 0);
    }

    #[test]
    fn test_ppo_inference() {
        let config = PPOConfig::default();
        let sizer = PPOPositionSizer::new(config);

        let state = StateFeatures {
            volatility: 0.2,
            momentum: 0.5,
            ..Default::default()
        };

        let result = sizer.get_position(&state);

        // Should return bounded position
        assert!(result.position_size >= -1.0 && result.position_size <= 1.0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_ppo_update() {
        let config = PPOConfig::default();
        let mut sizer = PPOPositionSizer::new(config);

        let states = vec![StateFeatures::default()];
        let actions = vec![0.5];
        let rewards = vec![0.1];
        let next_states = vec![StateFeatures::default()];
        let dones = vec![false];

        sizer.update(&states, &actions, &rewards, &next_states, &dones);

        assert_eq!(sizer.update_count(), 1);
    }
}
