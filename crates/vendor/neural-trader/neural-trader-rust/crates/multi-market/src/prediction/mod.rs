//! Prediction Markets Module
//!
//! Provides comprehensive prediction market functionality including:
//! - Polymarket API integration
//! - Sentiment analysis and market probability tracking
//! - Expected value calculation
//! - Order book analysis
//! - Market making strategies

pub mod polymarket;
pub mod sentiment;
pub mod expected_value;
pub mod orderbook;
pub mod strategies;

// Re-exports
pub use polymarket::{PolymarketClient, Market, Order as PolyOrder, Position as PolyPosition};
pub use sentiment::{SentimentAnalyzer, MarketSentiment, SentimentScore};
pub use expected_value::{ExpectedValueCalculator, EVOpportunity};
pub use orderbook::{OrderbookAnalyzer, OrderbookDepth, LiquidityMetrics};
pub use strategies::{MarketMakingStrategy, ArbitrageDetector};
