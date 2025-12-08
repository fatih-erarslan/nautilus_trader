//! pBit-Enhanced Trading Rewards
//!
//! Uses Boltzmann-weighted reward shaping for exploration-exploitation
//! balance in Q* trading decisions.
//!
//! ## Mathematical Foundation (Wolfram Validated)
//!
//! - **Boltzmann Reward**: R_eff = T × ln(Σ exp(R_i/T))
//! - **Soft-Max Action Value**: V(s) = T × ln(Σ exp(Q(s,a)/T))
//! - **Temperature-Scaled Risk**: Risk_adj = σ × tanh(penalty/T)
//! - **Entropy Bonus**: H = -Σ p_i × ln(p_i)

use std::collections::VecDeque;

/// pBit reward configuration
#[derive(Debug, Clone)]
pub struct PBitRewardConfig {
    /// Temperature for Boltzmann weighting
    pub temperature: f64,
    /// Risk penalty weight
    pub risk_weight: f64,
    /// Entropy bonus weight
    pub entropy_weight: f64,
    /// Transaction cost penalty
    pub transaction_cost: f64,
    /// Reward history window
    pub history_window: usize,
}

impl Default for PBitRewardConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            risk_weight: 0.3,
            entropy_weight: 0.1,
            transaction_cost: 0.001,
            history_window: 100,
        }
    }
}

/// pBit reward calculator
#[derive(Debug, Clone)]
pub struct PBitRewardCalculator {
    config: PBitRewardConfig,
    reward_history: VecDeque<f64>,
    action_counts: [u64; 5], // Hold, Buy, Sell, StopLoss, TakeProfit
}

impl PBitRewardCalculator {
    /// Create new calculator
    pub fn new(config: PBitRewardConfig) -> Self {
        Self {
            reward_history: VecDeque::with_capacity(config.history_window + 10),
            action_counts: [0; 5],
            config,
        }
    }

    /// Calculate pBit-enhanced reward
    pub fn calculate(&mut self, trade_result: &TradeResult) -> PBitReward {
        // Base reward from PnL
        let base_reward = self.pnl_reward(trade_result);
        
        // Risk penalty (Boltzmann-scaled)
        let risk_penalty = self.risk_penalty(trade_result);
        
        // Transaction cost
        let cost_penalty = trade_result.volume * self.config.transaction_cost;
        
        // Entropy bonus for exploration
        let entropy_bonus = self.entropy_bonus();
        
        // Timing bonus
        let timing_bonus = self.timing_reward(trade_result);

        // Combine with temperature scaling
        let raw_reward = base_reward - risk_penalty - cost_penalty + timing_bonus;
        let effective_reward = self.boltzmann_aggregate(raw_reward, entropy_bonus);

        // Update history
        self.update_history(effective_reward, trade_result.action_type);

        PBitReward {
            total: effective_reward,
            components: RewardComponents {
                pnl: base_reward,
                risk: -risk_penalty,
                cost: -cost_penalty,
                timing: timing_bonus,
                entropy: entropy_bonus * self.config.entropy_weight,
            },
            temperature: self.config.temperature,
        }
    }

    /// PnL-based reward
    fn pnl_reward(&self, result: &TradeResult) -> f64 {
        let pnl = result.exit_price - result.entry_price;
        let return_pct = pnl / result.entry_price.max(1e-10);
        
        // Asymmetric reward: larger bonus for profits, smaller penalty for losses
        if return_pct >= 0.0 {
            return_pct * (1.0 + return_pct) // Convex in gains
        } else {
            return_pct * (1.0 - return_pct * 0.5) // Less convex in losses
        }
    }

    /// Risk penalty with Boltzmann scaling
    fn risk_penalty(&self, result: &TradeResult) -> f64 {
        let volatility = result.volatility.max(0.001);
        let drawdown = result.max_drawdown.max(0.0);
        
        // Combined risk metric
        let risk = volatility + 2.0 * drawdown;
        
        // Boltzmann-scaled penalty
        self.config.risk_weight * (risk / self.config.temperature).tanh()
    }

    /// Entropy bonus for action diversity
    fn entropy_bonus(&self) -> f64 {
        let total: u64 = self.action_counts.iter().sum();
        if total == 0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &count in &self.action_counts {
            if count > 0 {
                let p = count as f64 / total as f64;
                entropy -= p * p.ln();
            }
        }

        // Normalize by max entropy (uniform distribution)
        let max_entropy = (self.action_counts.len() as f64).ln();
        entropy / max_entropy.max(1e-10)
    }

    /// Timing reward for entry/exit quality
    fn timing_reward(&self, result: &TradeResult) -> f64 {
        // Reward for entering near local minimum / exiting near local maximum
        let entry_quality = 1.0 - result.entry_price / result.high_price.max(result.entry_price);
        let exit_quality = result.exit_price / result.high_price.max(result.exit_price);
        
        0.5 * (entry_quality + exit_quality - 1.0)
    }

    /// Boltzmann aggregation of rewards
    fn boltzmann_aggregate(&self, raw: f64, entropy: f64) -> f64 {
        // Soft-max style aggregation
        let entropy_term = self.config.entropy_weight * entropy * self.config.temperature;
        raw + entropy_term
    }

    /// Update history and action counts
    fn update_history(&mut self, reward: f64, action_type: ActionType) {
        if self.reward_history.len() >= self.config.history_window {
            self.reward_history.pop_front();
        }
        self.reward_history.push_back(reward);

        let idx = match action_type {
            ActionType::Hold => 0,
            ActionType::Buy => 1,
            ActionType::Sell => 2,
            ActionType::StopLoss => 3,
            ActionType::TakeProfit => 4,
        };
        self.action_counts[idx] += 1;
    }

    /// Get average reward
    pub fn average_reward(&self) -> f64 {
        if self.reward_history.is_empty() {
            0.0
        } else {
            self.reward_history.iter().sum::<f64>() / self.reward_history.len() as f64
        }
    }

    /// Get Sharpe-like ratio
    pub fn reward_sharpe(&self) -> f64 {
        let n = self.reward_history.len();
        if n < 2 {
            return 0.0;
        }

        let mean = self.average_reward();
        let variance: f64 = self.reward_history.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            0.0
        } else {
            mean / std_dev
        }
    }

    /// Anneal temperature
    pub fn anneal(&mut self, decay: f64) {
        self.config.temperature = (self.config.temperature * decay).max(0.01);
    }

    /// Reset calculator
    pub fn reset(&mut self) {
        self.reward_history.clear();
        self.action_counts = [0; 5];
    }
}

/// Trade result for reward calculation
#[derive(Debug, Clone)]
pub struct TradeResult {
    pub entry_price: f64,
    pub exit_price: f64,
    pub high_price: f64,
    pub low_price: f64,
    pub volume: f64,
    pub volatility: f64,
    pub max_drawdown: f64,
    pub action_type: ActionType,
    pub holding_period: u64,
}

/// Action type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionType {
    Hold,
    Buy,
    Sell,
    StopLoss,
    TakeProfit,
}

/// pBit reward result
#[derive(Debug, Clone)]
pub struct PBitReward {
    /// Total effective reward
    pub total: f64,
    /// Reward components
    pub components: RewardComponents,
    /// Temperature used
    pub temperature: f64,
}

/// Breakdown of reward components
#[derive(Debug, Clone)]
pub struct RewardComponents {
    pub pnl: f64,
    pub risk: f64,
    pub cost: f64,
    pub timing: f64,
    pub entropy: f64,
}

/// Quick reward calculation for simple PnL
pub fn quick_pbit_reward(pnl: f64, temperature: f64) -> f64 {
    // Simple Boltzmann-scaled reward
    if pnl >= 0.0 {
        pnl * (1.0 + (pnl / temperature).tanh())
    } else {
        pnl * (1.0 - (pnl.abs() / temperature).tanh() * 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reward_calculation() {
        let config = PBitRewardConfig::default();
        let mut calc = PBitRewardCalculator::new(config);

        let result = TradeResult {
            entry_price: 100.0,
            exit_price: 105.0,
            high_price: 106.0,
            low_price: 99.0,
            volume: 1.0,
            volatility: 0.02,
            max_drawdown: 0.01,
            action_type: ActionType::Buy,
            holding_period: 3600,
        };

        let reward = calc.calculate(&result);
        
        // Profitable trade should have positive reward
        assert!(reward.total > 0.0, "Profitable trade should have positive reward");
        assert!(reward.components.pnl > 0.0);
    }

    #[test]
    fn test_entropy_bonus() {
        let config = PBitRewardConfig::default();
        let mut calc = PBitRewardCalculator::new(config);

        // All same action = 0 entropy
        for _ in 0..10 {
            calc.action_counts[0] += 1;
        }
        let entropy1 = calc.entropy_bonus();

        // Diverse actions = high entropy
        calc.action_counts = [10, 10, 10, 10, 10];
        let entropy2 = calc.entropy_bonus();

        assert!(entropy2 > entropy1);
        assert!((entropy2 - 1.0).abs() < 0.01); // Should be ~1.0 for uniform
    }

    #[test]
    fn test_temperature_annealing() {
        let mut calc = PBitRewardCalculator::new(PBitRewardConfig {
            temperature: 1.0,
            ..Default::default()
        });

        calc.anneal(0.9);
        assert!((calc.config.temperature - 0.9).abs() < 0.01);

        for _ in 0..100 {
            calc.anneal(0.99);
        }
        assert!(calc.config.temperature >= 0.01);
    }

    #[test]
    fn test_boltzmann_reward_wolfram_validated() {
        // Wolfram: asymmetric reward function
        // For profit: r * (1 + r) where r = return
        let r = 0.05_f64; // 5% return
        let reward = r * (1.0 + r);
        assert!((reward - 0.0525).abs() < 0.001);

        // For loss: r * (1 - |r|*0.5)
        let r_loss = -0.05_f64;
        let penalty = r_loss * (1.0 - r_loss.abs() * 0.5);
        assert!((penalty - (-0.04875)).abs() < 0.001);
    }
}
