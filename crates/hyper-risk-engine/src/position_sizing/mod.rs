//! Position sizing algorithms for optimal capital allocation.
//!
//! Implements both classical Kelly Criterion and modern reinforcement
//! learning approaches (PPO) for dynamic position sizing.
//!
//! ## Scientific References
//! - Kelly (1956): "A New Interpretation of Information Rate"
//! - Thorp (2006): "The Kelly Criterion in Blackjack, Sports Betting and the Stock Market"
//! - Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
//!
//! ## Implementation Notes
//!
//! For production trading:
//! - Use fractional Kelly (0.25-0.5) for more conservative sizing
//! - Adjust for transaction costs
//! - Consider regime-dependent sizing

pub mod kelly;
pub mod fractional_kelly;
pub mod ppo_sizer;

pub use kelly::{KellyCriterion, KellyConfig, KellyResult};
pub use fractional_kelly::{FractionalKelly, RiskTolerance};
pub use ppo_sizer::{PPOPositionSizer, PPOConfig, PositionSizeResult};
