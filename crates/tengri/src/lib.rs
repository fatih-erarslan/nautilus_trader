//! # Tengri - Advanced Cryptocurrency Trading Strategy
//! 
//! A high-performance cryptocurrency trading strategy implementation for Binance and Binance Futures,
//! featuring multi-source data integration and GPU acceleration capabilities.
//! 
//! ## Features
//! 
//! - **Multi-Exchange Trading**: Binance Spot and Futures support
//! - **Multi-Source Data**: Integration with Polymarket, Databento, and Tardis
//! - **GPU Acceleration**: Advanced mathematical computations with CUDA support
//! - **Risk Management**: Sophisticated position sizing and risk controls
//! - **Real-time Analytics**: Live market analysis and signal generation
//! - **Python Integration**: Seamless Python bridge for configuration and monitoring
//! 
//! ## Architecture
//! 
//! The strategy is built on Nautilus Trader core with the following components:
//! 
//! - `strategy/`: Core trading strategy implementation
//! - `data/`: Multi-source data aggregation and normalization
//! - `signals/`: Technical analysis and signal generation
//! - `risk/`: Risk management and position sizing
//! - `execution/`: Order management and execution logic
//! - `monitoring/`: Real-time performance and system monitoring
//! 
//! ## Usage
//! 
//! ```rust
//! use tengri::strategy::TengriStrategy;
//! use tengri::config::TengriConfig;
//! 
//! // Initialize strategy with configuration
//! let config = TengriConfig::from_file("tengri_config.toml")?;
//! let mut strategy = TengriStrategy::new(config).await?;
//! 
//! // Start trading
//! strategy.run().await?;
//! ```

pub mod config;
pub mod data;
pub mod events;
pub mod execution;
pub mod monitoring;
pub mod neural_networks;
pub mod neuromorphic;
pub mod python;
pub mod risk;
pub mod signals;
pub mod spike_swarm;
pub mod strategy;
pub mod neural_strategy;
pub mod quantum_integration;
pub mod types;
pub mod utils;

// Re-export main components
pub use config::TengriConfig;
pub use events::{EventQueue, EventProcessor, EventEnvelope, NeuralEvent, EventFilter};
pub use neuromorphic::{
    SpikingNeuron, NeuronConfig, SpikeEvent,
    STDPSynapse, SynapseConfig, LearningRule,
    EventQueue as NeuromorphicEventQueue, NeuralEvent as NeuromorphicEvent, EventPriority,
    NeuromorphicConfig, NeuromorphicSystem, PerformanceMetrics, HardwareBackend
};
pub use spike_swarm::{SpikeSwarm, SpikeSwarmConfig, SpikeEncoding, SwarmStatus};
pub use strategy::TengriStrategy;
pub use types::*;

// Error handling
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TengriError {
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Data source error: {0}")]
    DataSource(String),
    
    #[error("Strategy execution error: {0}")]
    Strategy(String),
    
    #[error("Risk management error: {0}")]
    Risk(String),
    
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("System time error: {0}")]
    SystemTime(#[from] std::time::SystemTimeError),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

pub type Result<T> = std::result::Result<T, TengriError>;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Initialize the Tengri trading system
pub async fn init() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    
    tracing::info!("Initializing Tengri Trading System v{}", VERSION);
    
    // Perform any global initialization here
    
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