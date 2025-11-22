//! # Autopoiesis Trading System
//! 
//! A self-organizing, biomimetic trading system inspired by autopoietic systems.
//! 
//! ## Overview
//! 
//! Autopoiesis implements a trading system that exhibits self-organization,
//! adaptation, and emergent behavior through the interaction of multiple
//! specialized components (observers).
//! 
//! ## Architecture
//! 
//! The system is organized into several key modules:
//! 
//! - `core`: Core traits and types that define the system's behavior
//! - `models`: Data models for market data, orders, and system state
//! - `observers`: Specialized components that observe and react to market conditions
//! - `engines`: Execution engines for different trading strategies
//! - `market_data`: Market data ingestion and processing
//! - `analysis`: Technical and statistical analysis tools
//! - `execution`: Order execution and management
//! - `risk`: Risk management and position sizing
//! - `portfolio`: Portfolio management and optimization
//! - `utils`: Utility functions and helpers

#![cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), feature(stdarch_x86_avx512))]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod core;
pub mod models;
pub mod observers;
pub mod engines;
pub mod market_data;
pub mod analysis;
pub mod execution;
pub mod risk;
pub mod portfolio;
pub mod utils;
pub mod consciousness;
pub mod ml;
pub mod api;
pub mod security;

// Re-export commonly used types
pub use crate::core::{Observer, ObserverContext, SystemState};
pub use crate::models::{MarketData, Order, Position, Trade};

/// Error type for the autopoiesis system
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// Market data error
    #[error("Market data error: {0}")]
    MarketData(String),
    
    /// Execution error
    #[error("Execution error: {0}")]
    Execution(String),
    
    /// Risk management error
    #[error("Risk management error: {0}")]
    Risk(String),
    
    /// Analysis error
    #[error("Analysis error: {0}")]
    Analysis(String),
    
    /// Database error
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Other errors
    #[error("Other error: {0}")]
    Other(String),
}

/// Result type for the autopoiesis system
pub type Result<T> = std::result::Result<T, Error>;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::core::*;
    pub use crate::models::*;
    pub use crate::{Error, Result};
    pub use async_trait::async_trait;
    pub use serde::{Deserialize, Serialize};
    pub use tracing::{debug, error, info, trace, warn};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::Config("Invalid configuration".to_string());
        assert_eq!(err.to_string(), "Configuration error: Invalid configuration");
    }
}