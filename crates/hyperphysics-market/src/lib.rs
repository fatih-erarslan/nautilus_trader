//! # HyperPhysics Market Data Module
//!
//! This module provides interfaces to various market data providers
//! and maps financial data to topological spaces for analysis.
//!
//! ## Features
//!
//! - Multiple market data provider support (Alpaca, Interactive Brokers, Binance)
//! - Real-time and historical data fetching
//! - Topological mapping of market structures
//! - Unified data models for cross-provider compatibility
//!
//! ## Example
//!
//! ```no_run
//! use hyperphysics_market::providers::{AlpacaProvider, MarketDataProvider};
//! use hyperphysics_market::data::Timeframe;
//! use chrono::Utc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let provider = AlpacaProvider::new(
//!         "your_api_key".to_string(),
//!         "your_api_secret".to_string(),
//!         "https://paper-api.alpaca.markets".to_string(), // paper trading URL
//!     );
//!
//!     let bar = provider.fetch_latest_bar("AAPL").await?;
//!     println!("Latest AAPL bar: {:?}", bar);
//!
//!     Ok(())
//! }
//! ```

pub mod arbitrage;
pub mod backtest;
pub mod data;
pub mod enactive;
pub mod error;
pub mod hnsw;
pub mod providers;
pub mod risk;
pub mod subsumption;
pub mod topology;
pub mod validation;

// Re-export commonly used types
pub use data::{Bar, Tick, Timeframe};
pub use data::tick::Quote;
pub use error::MarketError;
pub use providers::{
    AlpacaProvider,
    BinanceProvider,
    BinanceWebSocketClient,
    BybitProvider,
    CoinbaseProvider,
    // InteractiveBrokersProvider disabled due to rustc ICE
    KrakenProvider,
    OKXProvider,
    MarketDataProvider
};
pub use arbitrage::{ArbitrageDetector, ArbitrageOpportunity, ArbitrageType};
pub use backtest::{
    Strategy,
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    PerformanceMetrics,
    Portfolio,
    Position,
    Signal,
    Trade,
    Side,
    Commission,
    Slippage,
};
pub use risk::{
    RiskManager,
    RiskConfig,
    RiskLimits,
    PositionSizingStrategy,
    StopLossType,
    Position as RiskPosition,
    PortfolioMetrics,
    RiskViolation,
};
pub use validation::{
    DataValidator,
    ValidationConfig,
    ValidationError,
    SymbolValidationRules,
};
pub use enactive::{
    EnactiveMarketPerception,
    EnactiveConfig,
    CouplingState,
    CouplingEvent,
    EnactedPatterns,
    MarketRegime,
    ActionSignal,
};
pub use subsumption::{
    SubsumptionTradingSystem,
    SubsumptionConfig,
    SurvivalConfig,
    RiskConfig as SubsumptionRiskConfig,
    PositionConfig,
    ExecutionConfig,
    StrategyConfig,
    ExplorationConfig,
    MarketState as SubsumptionMarketState,
    StateUpdate,
    TradingAction,
    Order,
    OrderSide,
};
pub use hnsw::{
    HNSWIndex,
    HNSWNode,
    HNSWLayer,
    SearchResult,
    HNSWStats,
};
