//! Real-time market data integration
//!
//! Provides production-grade integration with real market data sources
//! including Binance WebSocket streams with comprehensive validation,
//! fault tolerance, and compliance monitoring.

pub mod binance_websocket_client;
pub mod integration_demo;

pub use binance_websocket_client::{
    BinanceWebSocketClient, DataSourceError, HealthCheckStatus, MarketDataStream,
};

pub use integration_demo::{demonstrate_real_data_integration, demonstrate_validation_only};
