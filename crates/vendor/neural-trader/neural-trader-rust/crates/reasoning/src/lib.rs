//! ReasoningBank Self-Learning Engine
//!
//! This crate provides adaptive pattern learning capabilities for trading systems.
//! It implements experience recording, verdict judgment, memory distillation, and
//! adaptive threshold adjustment based on performance.

pub mod pattern_learning;
pub mod types;
pub mod metrics;

pub use pattern_learning::PatternLearningEngine;
pub use types::{
    PatternExperience, MarketContext, PatternVerdict, Adaptation,
    DistilledPattern, PatternTrajectory, TradingSignal,
};
pub use metrics::{calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown};
