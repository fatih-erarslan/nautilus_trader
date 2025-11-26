//! Portfolio management and performance tracking
//!
//! Provides portfolio optimization, rebalancing, and performance analytics.

pub mod metrics;
pub mod pnl;
pub mod tracker;

pub use metrics::{MetricsCalculator, PerformanceMetrics};
pub use pnl::PnLCalculator;
pub use tracker::Portfolio;

use thiserror::Error;

/// Portfolio management errors
#[derive(Error, Debug)]
pub enum PortfolioError {
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    #[error("Invalid calculation: {0}")]
    InvalidCalculation(String),

    #[error("Invalid position: {0}")]
    InvalidPosition(String),

    #[error("Position not found: {0}")]
    PositionNotFound(String),

    #[error("Rebalancing failed: {0}")]
    RebalancingFailed(String),
}

/// Result type for portfolio operations
pub type Result<T> = std::result::Result<T, PortfolioError>;
