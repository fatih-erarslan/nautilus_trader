//! Interactive Brokers TWS API provider (stub)
//!
//! This module will provide integration with Interactive Brokers
//! through their Trader Workstation (TWS) API.
//!
//! Phase 2 implementation will include:
//! - TWS connection via IB Gateway
//! - Contract definitions
//! - Real-time and historical data requests
//! - Market depth (Level 2) data

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::data::{Bar, Timeframe};
use crate::error::{MarketError, MarketResult};
use super::MarketDataProvider;

/// Interactive Brokers provider (stub)
pub struct InteractiveBrokersProvider {
    // TODO: Add TWS connection fields
    // - host: String
    // - port: u16
    // - client_id: i32
    // - connection: Option<TwsConnection>
}

impl InteractiveBrokersProvider {
    /// Create new IB provider instance (stub)
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl MarketDataProvider for InteractiveBrokersProvider {
    async fn fetch_bars(
        &self,
        _symbol: &str,
        _timeframe: Timeframe,
        _start: DateTime<Utc>,
        _end: DateTime<Utc>,
    ) -> MarketResult<Vec<Bar>> {
        Err(MarketError::ApiError(
            "Interactive Brokers provider not yet implemented".to_string()
        ))
    }

    async fn fetch_latest_bar(&self, _symbol: &str) -> MarketResult<Bar> {
        Err(MarketError::ApiError(
            "Interactive Brokers provider not yet implemented".to_string()
        ))
    }

    fn provider_name(&self) -> &str {
        "Interactive Brokers"
    }

    fn supports_realtime(&self) -> bool {
        true
    }

    async fn supports_symbol(&self, _symbol: &str) -> MarketResult<bool> {
        Ok(false)
    }
}
