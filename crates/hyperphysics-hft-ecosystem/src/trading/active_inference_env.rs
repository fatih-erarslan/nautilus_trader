//! Active Inference Trading Environment
//!
//! Bridges Active Inference agents with HyperPhysics market dynamics

use active_inference_agent::{ActiveInferenceAgent, GenerativeModel};
use nalgebra as na;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Market state representation for Active Inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketState {
    /// Normalized price movements (returns)
    pub returns: na::DVector<f64>,
    /// Volume indicators
    pub volume: na::DVector<f64>,
    /// Volatility estimate
    pub volatility: f64,
    /// Market regime (0 = bearish, 0.5 = neutral, 1 = bullish)
    pub regime: f64,
    /// Timestamp (seconds since epoch)
    pub timestamp: f64,
}

impl MarketState {
    /// Convert market state to observation vector for Active Inference
    pub fn to_observation(&self) -> na::DVector<f64> {
        let mut obs = Vec::new();

        // Add returns
        obs.extend_from_slice(self.returns.as_slice());

        // Add volume
        obs.extend_from_slice(self.volume.as_slice());

        // Add scalar features
        obs.push(self.volatility);
        obs.push(self.regime);

        na::DVector::from_vec(obs)
    }

    /// Dimension of observation vector
    pub fn observation_dim(return_window: usize, volume_window: usize) -> usize {
        return_window + volume_window + 2 // +2 for volatility and regime
    }
}

/// Trading action for Active Inference agents
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TradingAction {
    /// Hold current position
    Hold,
    /// Buy (long)
    Buy(f64), // Position size [0.0, 1.0]
    /// Sell (short)
    Sell(f64), // Position size [0.0, 1.0]
}

impl TradingAction {
    /// Convert to action vector for Active Inference
    pub fn to_vector(&self) -> na::DVector<f64> {
        match self {
            TradingAction::Hold => na::DVector::from_vec(vec![0.0, 0.0]),
            TradingAction::Buy(size) => na::DVector::from_vec(vec![*size, 0.0]),
            TradingAction::Sell(size) => na::DVector::from_vec(vec![0.0, *size]),
        }
    }

    /// Convert from action vector
    pub fn from_vector(vec: &na::DVector<f64>) -> Self {
        if vec.len() < 2 {
            return TradingAction::Hold;
        }

        let buy_signal = vec[0];
        let sell_signal = vec[1];

        if buy_signal > sell_signal && buy_signal > 0.1 {
            TradingAction::Buy(buy_signal.min(1.0))
        } else if sell_signal > buy_signal && sell_signal > 0.1 {
            TradingAction::Sell(sell_signal.min(1.0))
        } else {
            TradingAction::Hold
        }
    }
}

/// Active Inference Trading Environment
#[derive(Debug, Clone)]
pub struct TradingEnvironment {
    /// Window size for returns
    pub return_window: usize,
    /// Window size for volume
    pub volume_window: usize,
    /// Historical market states
    pub history: VecDeque<MarketState>,
    /// Current position
    pub position: f64, // [-1.0, 1.0] where negative = short
    /// Cumulative PnL
    pub pnl: f64,
    /// Sharpe ratio estimate
    pub sharpe_ratio: f64,
    /// Returns history for Sharpe calculation
    returns_history: VecDeque<f64>,
}

impl TradingEnvironment {
    /// Create new trading environment
    pub fn new(return_window: usize, volume_window: usize) -> Self {
        Self {
            return_window,
            volume_window,
            history: VecDeque::with_capacity(100),
            position: 0.0,
            pnl: 0.0,
            sharpe_ratio: 0.0,
            returns_history: VecDeque::with_capacity(100),
        }
    }

    /// Add market observation
    pub fn observe(&mut self, state: MarketState) {
        self.history.push_back(state);
        if self.history.len() > 100 {
            self.history.pop_front();
        }
    }

    /// Get current observation vector
    pub fn get_observation(&self) -> Option<na::DVector<f64>> {
        self.history.back().map(|state| state.to_observation())
    }

    /// Execute trading action
    pub fn execute_action(&mut self, action: TradingAction) -> f64 {
        let reward = match action {
            TradingAction::Hold => 0.0,
            TradingAction::Buy(size) => {
                let new_position = (self.position + size).min(1.0);
                let position_change = new_position - self.position;
                self.position = new_position;
                position_change
            }
            TradingAction::Sell(size) => {
                let new_position = (self.position - size).max(-1.0);
                let position_change = self.position - new_position;
                self.position = new_position;
                -position_change
            }
        };

        // Calculate reward based on position and market return
        if let Some(state) = self.history.back() {
            let market_return = state.returns.iter().last().copied().unwrap_or(0.0);
            let position_return = self.position * market_return;

            // Update PnL
            self.pnl += position_return;

            // Update returns history
            self.returns_history.push_back(position_return);
            if self.returns_history.len() > 100 {
                self.returns_history.pop_front();
            }

            // Calculate Sharpe ratio
            if self.returns_history.len() > 10 {
                let mean_return: f64 =
                    self.returns_history.iter().sum::<f64>() / self.returns_history.len() as f64;
                let variance: f64 = self
                    .returns_history
                    .iter()
                    .map(|r| (r - mean_return).powi(2))
                    .sum::<f64>()
                    / self.returns_history.len() as f64;
                let std_dev = variance.sqrt();

                self.sharpe_ratio = if std_dev > 1e-6 {
                    mean_return / std_dev
                } else {
                    0.0
                };
            }

            position_return
        } else {
            0.0
        }
    }

    /// Build generative model for Active Inference agent
    pub fn build_generative_model(&self) -> GenerativeModel {
        let state_dim = 10; // Simplified latent market states
        let obs_dim = MarketState::observation_dim(self.return_window, self.volume_window);

        let mut model = GenerativeModel::new(state_dim, obs_dim);

        // Set transition matrix (mean-reverting dynamics)
        for i in 0..state_dim {
            model.transition[(i, i)] = 0.9; // Persistence
            if i > 0 {
                model.transition[(i, i - 1)] = 0.05; // Coupling
            }
            if i < state_dim - 1 {
                model.transition[(i, i + 1)] = 0.05;
            }
        }

        // Set likelihood matrix (observation model)
        // Map each observation dimension to a subset of state dimensions
        let ratio = state_dim as f64 / obs_dim as f64;
        for i in 0..obs_dim {
            let state_idx = ((i as f64 * ratio).floor() as usize).min(state_dim - 1);
            model.likelihood[(i, state_idx)] = 1.0;
            // Add smoothing across nearby states
            if state_idx > 0 {
                model.likelihood[(i, state_idx - 1)] = 0.3;
            }
            if state_idx < state_dim - 1 {
                model.likelihood[(i, state_idx + 1)] = 0.3;
            }
        }

        // Set preferences (profit-seeking states)
        model.preferences = na::DVector::from_fn(state_dim, |i, _| {
            if i > state_dim / 2 {
                1.0 / state_dim as f64 // Prefer higher-indexed states (bullish)
            } else {
                0.5 / state_dim as f64
            }
        });

        model
    }

    /// Create trading action repertoire
    pub fn create_action_repertoire(state_dim: usize) -> Vec<na::DVector<f64>> {
        // Create action vectors matching the state dimensionality
        // The first two dimensions correspond to buy and sell signals; remaining dimensions are zero.
        let mut hold = na::DVector::zeros(state_dim);
        let mut buy_small = na::DVector::zeros(state_dim);
        buy_small[0] = 0.25;
        let mut buy_medium = na::DVector::zeros(state_dim);
        buy_medium[0] = 0.5;
        let mut buy_full = na::DVector::zeros(state_dim);
        buy_full[0] = 1.0;
        let mut sell_small = na::DVector::zeros(state_dim);
        sell_small[1] = 0.25;
        let mut sell_medium = na::DVector::zeros(state_dim);
        sell_medium[1] = 0.5;
        let mut sell_full = na::DVector::zeros(state_dim);
        sell_full[1] = 1.0;

        vec![
            hold,
            buy_small,
            buy_medium,
            buy_full,
            sell_small,
            sell_medium,
            sell_full,
        ]
    }
}

/// Active Inference Trading Agent
pub struct AITradingAgent {
    /// Core Active Inference agent
    pub agent: ActiveInferenceAgent,
    /// Trading environment
    pub environment: TradingEnvironment,
}

impl AITradingAgent {
    /// Create new AI trading agent
    pub fn new(return_window: usize, volume_window: usize) -> Self {
        let environment = TradingEnvironment::new(return_window, volume_window);
        let model = environment.build_generative_model();

        let initial_belief =
            na::DVector::from_element(model.state_dim, 1.0 / model.state_dim as f64);
        let mut agent = ActiveInferenceAgent::new(model, initial_belief);

        // Set action repertoire
        // Use the model's state dimensionality to create action vectors of matching size
        agent.actions = TradingEnvironment::create_action_repertoire(agent.model.state_dim);
        agent.precision = 2.0; // Higher precision for more deterministic action selection

        Self { agent, environment }
    }

    /// Process market update and select action
    pub fn step(&mut self, state: MarketState) -> Option<TradingAction> {
        // Add market state to environment
        self.environment.observe(state);

        // Get observation
        let observation = self.environment.get_observation()?;

        // Update Active Inference agent
        self.agent.step(&observation);

        // Select action
        self.agent
            .select_action()
            .map(|action_vec| TradingAction::from_vector(&action_vec))
    }

    /// Execute selected action
    pub fn execute(&mut self, action: TradingAction) -> f64 {
        self.environment.execute_action(action)
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> TradingMetrics {
        TradingMetrics {
            pnl: self.environment.pnl,
            sharpe_ratio: self.environment.sharpe_ratio,
            position: self.environment.position,
            free_energy: if let Some(obs) = self.environment.get_observation() {
                self.agent.compute_free_energy(&obs)
            } else {
                f64::NAN
            },
        }
    }
}

/// Trading performance metrics
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TradingMetrics {
    pub pnl: f64,
    pub sharpe_ratio: f64,
    pub position: f64,
    pub free_energy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_state_conversion() {
        let state = MarketState {
            returns: na::DVector::from_vec(vec![0.01, -0.005]),
            volume: na::DVector::from_vec(vec![1000.0, 1200.0]),
            volatility: 0.02,
            regime: 0.6,
            timestamp: 1000.0,
        };

        let obs = state.to_observation();
        assert_eq!(obs.len(), 6); // 2 returns + 2 volume + 2 scalars
    }

    #[test]
    fn test_trading_action_conversion() {
        let buy = TradingAction::Buy(0.5);
        let vec = buy.to_vector();
        let recovered = TradingAction::from_vector(&vec);

        match recovered {
            TradingAction::Buy(size) => assert!((size - 0.5).abs() < 0.01),
            _ => panic!("Expected Buy action"),
        }
    }

    #[test]
    fn test_trading_environment() {
        let mut env = TradingEnvironment::new(5, 5);

        let state = MarketState {
            returns: na::DVector::from_vec(vec![0.01; 5]),
            volume: na::DVector::from_vec(vec![1000.0; 5]),
            volatility: 0.02,
            regime: 0.5,
            timestamp: 1000.0,
        };

        env.observe(state);
        assert!(env.get_observation().is_some());
    }

    #[test]
    fn test_ai_trading_agent() {
        let mut agent = AITradingAgent::new(3, 3);

        let state = MarketState {
            returns: na::DVector::from_vec(vec![0.01, 0.005, -0.002]),
            volume: na::DVector::from_vec(vec![1000.0, 1100.0, 900.0]),
            volatility: 0.015,
            regime: 0.6,
            timestamp: 1000.0,
        };

        let action = agent.step(state);
        assert!(action.is_some());

        if let Some(act) = action {
            let reward = agent.execute(act);
            assert!(reward.is_finite());
        }

        let metrics = agent.get_metrics();
        assert!(metrics.free_energy.is_finite());
    }
}
