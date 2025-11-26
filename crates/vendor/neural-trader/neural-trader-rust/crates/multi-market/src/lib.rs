//! Multi-Market Trading Support
//!
//! This crate provides comprehensive support for trading across multiple market types:
//! - Sports betting (The Odds API integration)
//! - Prediction markets (Polymarket integration)
//! - Cryptocurrency trading (DeFi and cross-exchange)
//!
//! # Features
//!
//! ## Sports Betting
//! - Live odds tracking for 30+ sports
//! - Kelly Criterion for optimal stake sizing
//! - Arbitrage opportunity detection
//! - Syndicate management for pooled betting
//!
//! ## Prediction Markets
//! - Polymarket CLOB API integration
//! - Sentiment analysis and market manipulation detection
//! - Expected value calculation
//! - Multi-market arbitrage
//!
//! ## Cryptocurrency
//! - CCXT integration for 100+ exchanges
//! - DeFi yield optimization
//! - Cross-exchange arbitrage
//! - Gas optimization
//!
//! # Example
//!
//! ```rust,no_run
//! use multi_market::{
//!     sports::{OddsApiClient, KellyOptimizer},
//!     prediction::PolymarketClient,
//!     crypto::DefiManager,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Sports betting
//!     let odds_client = OddsApiClient::new("api_key");
//!     let kelly = KellyOptimizer::new(10000.0, 0.25);
//!
//!     // Prediction markets
//!     let polymarket = PolymarketClient::new("api_key");
//!
//!     // Cryptocurrency
//!     let defi = DefiManager::new();
//!
//!     Ok(())
//! }
//! ```

pub mod error;
pub mod types;

#[cfg(feature = "sports")]
pub mod sports;

#[cfg(feature = "prediction")]
pub mod prediction;

#[cfg(feature = "crypto")]
pub mod crypto;

// Re-exports for convenience
pub use error::{MultiMarketError, Result};
pub use types::*;

#[cfg(feature = "sports")]
pub use sports::{
    OddsApiClient, KellyOptimizer, Syndicate, ArbitrageDetector as SportsArbitrageDetector,
    PollingOddsStreamer, WebSocketOddsStreamer, BettingOpportunity, KellyResult,
};

#[cfg(feature = "prediction")]
pub use prediction::{
    PolymarketClient, SentimentAnalyzer, ExpectedValueCalculator, OrderbookAnalyzer,
    MarketMakingStrategy, ArbitrageDetector as PredictionArbitrageDetector,
};

#[cfg(feature = "crypto")]
pub use crypto::{
    DefiManager, ArbitrageEngine, YieldFarmingStrategy, GasOptimizer, DexArbitrageStrategy,
    LiquidityPoolStrategy,
};
