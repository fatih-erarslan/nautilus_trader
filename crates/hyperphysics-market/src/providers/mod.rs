//! Market data provider interfaces and implementations

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::data::{Bar, Timeframe};
use crate::error::MarketResult;

pub mod alpaca;
pub mod binance;
pub mod binance_websocket;
pub mod bybit;
pub mod coinbase;
// NOTE: interactive_brokers temporarily disabled due to rustc ICE (internal compiler error)
// tracking: https://github.com/rust-lang/rust/issues - evaluate_obligation crash
// pub mod interactive_brokers;
pub mod kraken;
pub mod okx;

pub use alpaca::AlpacaProvider;
pub use binance::BinanceProvider;
pub use binance_websocket::BinanceWebSocketClient;
pub use bybit::BybitProvider;
pub use coinbase::CoinbaseProvider;
// pub use interactive_brokers::InteractiveBrokersProvider;
pub use kraken::KrakenProvider;
pub use okx::OKXProvider;

/// Trait defining the interface for market data providers
#[async_trait]
pub trait MarketDataProvider: Send + Sync {
    /// Fetch historical bars for a symbol
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading symbol (e.g., "AAPL", "BTCUSD")
    /// * `timeframe` - Bar timeframe
    /// * `start` - Start time for historical data
    /// * `end` - End time for historical data
    ///
    /// # Returns
    ///
    /// Vector of bars ordered chronologically
    async fn fetch_bars(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MarketResult<Vec<Bar>>;

    /// Fetch the latest bar for a symbol
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading symbol
    ///
    /// # Returns
    ///
    /// Most recent bar data
    async fn fetch_latest_bar(&self, symbol: &str) -> MarketResult<Bar>;

    /// Get the name of the provider
    fn provider_name(&self) -> &str;

    /// Check if the provider supports real-time data
    fn supports_realtime(&self) -> bool {
        false
    }

    /// Check if the provider supports the given symbol
    async fn supports_symbol(&self, symbol: &str) -> MarketResult<bool>;
}

/// Provider configuration
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// API key for authentication
    pub api_key: String,

    /// API secret for authentication
    pub api_secret: String,

    /// Base URL for API endpoints
    pub base_url: String,

    /// Whether to use paper trading endpoints
    pub paper_trading: bool,

    /// Request timeout in seconds
    pub timeout_secs: u64,

    /// Maximum retries for failed requests
    pub max_retries: u32,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            api_secret: String::new(),
            base_url: String::new(),
            paper_trading: true,
            timeout_secs: 30,
            max_retries: 3,
        }
    }
}
