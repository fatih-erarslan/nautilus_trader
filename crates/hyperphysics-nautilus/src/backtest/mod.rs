//! Backtest integration module.
//!
//! This module provides utilities for backtesting HyperPhysics strategies
//! using either Nautilus Trader's backtest engine or standalone simulation.

mod runner;
mod data_loader;

pub use runner::BacktestRunner;
pub use data_loader::DataLoader;

use crate::types::{NautilusBar, NautilusQuoteTick, NautilusTradeTick};
use serde::{Deserialize, Serialize};

/// Backtest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Start timestamp (nanoseconds)
    pub start_time_ns: u64,
    /// End timestamp (nanoseconds)
    pub end_time_ns: u64,
    /// Initial capital
    pub initial_capital: f64,
    /// Commission rate (as decimal, e.g., 0.001 = 0.1%)
    pub commission_rate: f64,
    /// Slippage model
    pub slippage_model: SlippageModel,
    /// Enable detailed logging
    pub verbose: bool,
}

/// Slippage model for backtest execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlippageModel {
    /// No slippage
    None,
    /// Fixed basis points slippage
    FixedBps(f64),
    /// Volatility-based slippage
    VolatilityBased { multiplier: f64 },
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            start_time_ns: 0,
            end_time_ns: u64::MAX,
            initial_capital: 100_000.0,
            commission_rate: 0.001,
            slippage_model: SlippageModel::FixedBps(1.0),
            verbose: false,
        }
    }
}

/// Backtest results summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResults {
    /// Total return (as decimal)
    pub total_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Maximum drawdown (as decimal)
    pub max_drawdown: f64,
    /// Total trades executed
    pub total_trades: u64,
    /// Win rate (0-1)
    pub win_rate: f64,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Final equity
    pub final_equity: f64,
    /// Average latency (microseconds)
    pub avg_latency_us: f64,
    /// Total runtime (seconds)
    pub backtest_runtime_secs: f64,
}

/// Market data event for backtesting
#[derive(Debug, Clone)]
pub enum MarketDataEvent {
    /// Quote tick
    Quote(NautilusQuoteTick),
    /// Trade tick
    Trade(NautilusTradeTick),
    /// Bar/candle
    Bar(NautilusBar),
}

impl MarketDataEvent {
    /// Get event timestamp
    pub fn timestamp(&self) -> u64 {
        match self {
            MarketDataEvent::Quote(q) => q.ts_event,
            MarketDataEvent::Trade(t) => t.ts_event,
            MarketDataEvent::Bar(b) => b.ts_event,
        }
    }

    /// Get instrument ID
    pub fn instrument_id(&self) -> u64 {
        match self {
            MarketDataEvent::Quote(q) => q.instrument_id,
            MarketDataEvent::Trade(t) => t.instrument_id,
            MarketDataEvent::Bar(b) => b.instrument_id,
        }
    }
}
