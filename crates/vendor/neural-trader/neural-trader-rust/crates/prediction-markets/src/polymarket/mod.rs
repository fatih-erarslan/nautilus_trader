//! Polymarket CLOB client and trading strategies

pub mod auth;
pub mod client;
pub mod websocket;
pub mod mm;
pub mod arbitrage;

pub use auth::{Credentials, RateLimiter};
pub use client::{ClientConfig, PolymarketClient};
pub use websocket::{PolymarketStream, StreamBuilder};
pub use mm::{MarketMakerConfig, PolymarketMM};
pub use arbitrage::{ArbitrageConfig, PolymarketArbitrage, ArbitrageOpportunity, RiskLevel};
