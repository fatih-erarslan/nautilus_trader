//! # Cryptocurrency Data Collector
//! 
//! A comprehensive data collection system for cryptocurrency market data,
//! supporting multiple exchanges and time periods for backtesting.
//! 
//! ## Features
//! 
//! - **Multi-Exchange Support**: Binance, Coinbase, Kraken, OKX, Bybit
//! - **5+ Years Historical Data**: Complete market history for backtesting
//! - **Multiple Data Types**: OHLCV, trades, order book, funding rates
//! - **High Performance**: Concurrent downloads with rate limiting
//! - **Data Validation**: Comprehensive quality checks and cleaning
//! - **Storage Formats**: Parquet, CSV, SQLite for optimal performance
//! - **Progress Tracking**: Real-time download progress monitoring
//! 
//! ## Usage
//! 
//! ```rust
//! use data_collector::DataCollector;
//! 
//! let collector = DataCollector::new().await?;
//! 
//! // Download 5 years of BTC/USDT data from Binance
//! collector.download_historical_data(
//!     "binance",
//!     "BTCUSDT", 
//!     "1h",
//!     "2019-01-01",
//!     "2024-01-01"
//! ).await?;
//! ```

pub mod types;
pub mod config;
pub mod rate_limiter;
pub mod collectors;
pub mod storage;

// Re-export main components
pub use collectors::DataCollector;
pub use config::CollectorConfig;
pub use types::*;

// Error handling
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataCollectorError {
    #[error("Exchange API error: {0}")]
    ExchangeApi(String),
    
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    #[error("Storage error: {0}")]
    Storage(String),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),
    
    #[error("Data quality issue: {0}")]
    DataQuality(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Parse error: {0}")]
    Parse(#[from] chrono::ParseError),
    
    #[error("Number parse error: {0}")]
    ParseFloat(#[from] std::num::ParseFloatError),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

pub type Result<T> = std::result::Result<T, DataCollectorError>;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Initialize the data collector system
pub async fn init() -> Result<()> {
    tracing::info!("Initializing Cryptocurrency Data Collector v{}", VERSION);
    
    // Storage initialization is handled per backend
    
    // Initialize rate limiters
    rate_limiter::init().await?;
    
    tracing::info!("Data Collector system initialized successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_init() {
        init().await.unwrap();
    }
}