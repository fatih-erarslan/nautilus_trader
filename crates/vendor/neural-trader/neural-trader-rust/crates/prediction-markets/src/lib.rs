//! Prediction markets integration (Polymarket)
//!
//! Provides CLOB client, orderbook streaming, market making, and arbitrage detection.
//!
//! # Features
//! - REST API client for Polymarket CLOB
//! - WebSocket streaming for real-time data
//! - Automated market making
//! - Arbitrage detection and execution
//!
//! # Example
//! ```rust,no_run
//! use nt_prediction_markets::polymarket::{ClientConfig, PolymarketClient};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = ClientConfig::new("your_api_key");
//!     let client = PolymarketClient::new(config)?;
//!
//!     let markets = client.get_markets().await?;
//!     println!("Found {} markets", markets.len());
//!
//!     Ok(())
//! }
//! ```

pub mod error;
pub mod models;
pub mod polymarket;

pub use error::{PredictionMarketError, Result};
pub use models::{Market, Order, OrderBook, OrderRequest, OrderResponse, Position};
pub use polymarket::PolymarketClient;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
