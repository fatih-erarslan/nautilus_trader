//! Backtesting framework for strategy validation
//!
//! Provides comprehensive backtesting capabilities with:
//! - Historical data replay
//! - Realistic slippage and commission models
//! - Performance metrics and analytics
//! - Walk-forward analysis support

pub mod engine;
pub mod performance;
pub mod slippage;

pub use engine::{BacktestEngine, BacktestResult, Trade, TradeSide, EquityPoint};
pub use performance::PerformanceMetrics;
pub use slippage::SlippageModel;
