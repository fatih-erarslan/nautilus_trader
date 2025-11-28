//! Data feed sentinel for monitoring market data quality.
//!
//! Operates in the fast path (<20μs) to validate market data
//! freshness, consistency, and integrity before order processing.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use crate::core::types::{Order, Portfolio, Symbol, Timestamp};
use crate::core::error::{Result, RiskError};

use super::base::{Sentinel, SentinelConfig, SentinelId, SentinelStats, SentinelStatus};

/// Configuration for the data feed sentinel.
#[derive(Debug, Clone)]
pub struct DataFeedConfig {
    /// Base sentinel configuration.
    pub base: SentinelConfig,
    /// Maximum data staleness in milliseconds.
    pub max_staleness_ms: u64,
    /// Maximum price deviation from previous tick (percentage).
    pub max_price_deviation_pct: f64,
    /// Minimum required data feeds for trading.
    pub min_feed_count: usize,
    /// Enable tick validation.
    pub validate_ticks: bool,
}

impl Default for DataFeedConfig {
    fn default() -> Self {
        Self {
            base: SentinelConfig {
                name: "data_feed_sentinel".to_string(),
                enabled: true,
                priority: 1,
                verbose: false,
            },
            max_staleness_ms: 1000,
            max_price_deviation_pct: 10.0,
            min_feed_count: 1,
            validate_ticks: true,
        }
    }
}

/// Data feed status for a symbol.
#[derive(Debug, Clone)]
pub struct FeedStatus {
    /// Symbol.
    pub symbol: Symbol,
    /// Last update timestamp.
    pub last_update: Timestamp,
    /// Last price.
    pub last_price: f64,
    /// Feed is healthy.
    pub healthy: bool,
    /// Consecutive stale count.
    pub stale_count: u32,
}

/// Data feed sentinel.
#[derive(Debug)]
pub struct DataFeedSentinel {
    id: SentinelId,
    config: DataFeedConfig,
    status: AtomicU8,
    stats: SentinelStats,
    /// Feed status by symbol.
    feed_status: RwLock<HashMap<Symbol, FeedStatus>>,
}

impl DataFeedSentinel {
    /// Create a new data feed sentinel.
    pub fn new(config: DataFeedConfig) -> Self {
        Self {
            id: SentinelId::new(&config.base.name),
            config,
            status: AtomicU8::new(SentinelStatus::Active as u8),
            stats: SentinelStats::new(),
            feed_status: RwLock::new(HashMap::new()),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(DataFeedConfig::default())
    }

    /// Update feed status for a symbol.
    pub fn update_feed(&self, symbol: Symbol, price: f64, timestamp: Timestamp) {
        let mut feeds = self.feed_status.write();

        if let Some(existing) = feeds.get_mut(&symbol) {
            // Check for price deviation
            let deviation_pct = if existing.last_price > 0.0 {
                ((price - existing.last_price) / existing.last_price).abs() * 100.0
            } else {
                0.0
            };

            existing.last_update = timestamp;
            existing.last_price = price;
            existing.healthy = deviation_pct <= self.config.max_price_deviation_pct;
            existing.stale_count = 0;
        } else {
            feeds.insert(
                symbol.clone(),
                FeedStatus {
                    symbol,
                    last_update: timestamp,
                    last_price: price,
                    healthy: true,
                    stale_count: 0,
                },
            );
        }
    }

    /// Check if feed is fresh for a symbol.
    pub fn is_feed_fresh(&self, symbol: &Symbol) -> bool {
        let feeds = self.feed_status.read();
        if let Some(feed) = feeds.get(symbol) {
            let now_ns = Timestamp::now().as_nanos();
            let feed_ns = feed.last_update.as_nanos();
            let age_ms = (now_ns.saturating_sub(feed_ns)) / 1_000_000;
            age_ms <= self.config.max_staleness_ms
        } else {
            false
        }
    }

    /// Get all feed statuses.
    pub fn get_feed_statuses(&self) -> Vec<FeedStatus> {
        self.feed_status.read().values().cloned().collect()
    }

    /// Get healthy feed count.
    pub fn healthy_feed_count(&self) -> usize {
        self.feed_status
            .read()
            .values()
            .filter(|f| f.healthy)
            .count()
    }

    /// Convert u8 to SentinelStatus.
    fn status_from_u8(value: u8) -> SentinelStatus {
        match value {
            0 => SentinelStatus::Active,
            1 => SentinelStatus::Disabled,
            2 => SentinelStatus::Triggered,
            _ => SentinelStatus::Error,
        }
    }
}

impl Sentinel for DataFeedSentinel {
    fn id(&self) -> SentinelId {
        self.id.clone()
    }

    fn status(&self) -> SentinelStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn check(&self, order: &Order, _portfolio: &Portfolio) -> Result<()> {
        let start = Instant::now();

        // Check if disabled
        if self.status() == SentinelStatus::Disabled {
            return Ok(());
        }

        // Check minimum feed count
        let healthy_count = self.healthy_feed_count();
        if healthy_count < self.config.min_feed_count {
            self.stats.record_trigger();
            self.status.store(SentinelStatus::Triggered as u8, Ordering::SeqCst);
            let latency_ns = start.elapsed().as_nanos() as u64;
            self.stats.record_check(latency_ns);
            return Err(RiskError::InternalError(format!(
                "Insufficient healthy data feeds: {} < {}",
                healthy_count, self.config.min_feed_count
            )));
        }

        // Check feed freshness for order symbol
        if !self.is_feed_fresh(&order.symbol) {
            self.stats.record_trigger();
            self.status.store(SentinelStatus::Triggered as u8, Ordering::SeqCst);
            let latency_ns = start.elapsed().as_nanos() as u64;
            self.stats.record_check(latency_ns);
            return Err(RiskError::InternalError(format!(
                "Stale data feed for symbol: {}",
                order.symbol.as_str()
            )));
        }

        // Check feed health
        let feeds = self.feed_status.read();
        if let Some(feed) = feeds.get(&order.symbol) {
            if !feed.healthy {
                drop(feeds);
                self.stats.record_trigger();
                self.status.store(SentinelStatus::Triggered as u8, Ordering::SeqCst);
                let latency_ns = start.elapsed().as_nanos() as u64;
                self.stats.record_check(latency_ns);
                return Err(RiskError::InternalError(format!(
                    "Unhealthy data feed for symbol: {}",
                    order.symbol.as_str()
                )));
            }
        }

        let latency_ns = start.elapsed().as_nanos() as u64;
        self.stats.record_check(latency_ns);
        Ok(())
    }

    fn reset(&self) {
        self.stats.reset();
        self.status.store(SentinelStatus::Active as u8, Ordering::SeqCst);

        // Reset all feeds to healthy
        let mut feeds = self.feed_status.write();
        for feed in feeds.values_mut() {
            feed.healthy = true;
            feed.stale_count = 0;
        }
    }

    fn enable(&self) {
        self.status.store(SentinelStatus::Active as u8, Ordering::SeqCst);
    }

    fn disable(&self) {
        self.status.store(SentinelStatus::Disabled as u8, Ordering::SeqCst);
    }

    fn check_count(&self) -> u64 {
        self.stats.checks.load(Ordering::Relaxed)
    }

    fn trigger_count(&self) -> u64 {
        self.stats.triggers.load(Ordering::Relaxed)
    }

    fn avg_latency_ns(&self) -> u64 {
        self.stats.avg_latency_ns()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{OrderSide, Price, Quantity};

    fn create_test_order(symbol: &str) -> Order {
        Order {
            symbol: Symbol::new(symbol),
            side: OrderSide::Buy,
            quantity: Quantity::from_f64(100.0),
            limit_price: Some(Price::from_f64(100.0)),
            strategy_id: 1,
            timestamp: Timestamp::now(),
        }
    }

    #[test]
    fn test_data_feed_sentinel_creation() {
        let sentinel = DataFeedSentinel::with_defaults();
        assert_eq!(sentinel.status(), SentinelStatus::Active);
        assert_eq!(sentinel.check_count(), 0);
    }

    #[test]
    fn test_feed_update() {
        let sentinel = DataFeedSentinel::with_defaults();
        let symbol = Symbol::new("AAPL");

        sentinel.update_feed(symbol.clone(), 150.0, Timestamp::now());

        assert!(sentinel.is_feed_fresh(&symbol));
        assert_eq!(sentinel.healthy_feed_count(), 1);
    }

    #[test]
    fn test_feed_staleness_check() {
        let mut config = DataFeedConfig::default();
        config.max_staleness_ms = 100; // 100ms
        config.min_feed_count = 0; // Don't require feeds for this test

        let sentinel = DataFeedSentinel::new(config);
        let symbol = Symbol::new("AAPL");

        // Update with old timestamp
        let old_time = Timestamp::from_nanos(
            Timestamp::now().as_nanos().saturating_sub(200_000_000) // 200ms ago
        );
        sentinel.update_feed(symbol.clone(), 150.0, old_time);

        assert!(!sentinel.is_feed_fresh(&symbol));
    }

    #[test]
    fn test_check_passes_with_fresh_data() {
        let mut config = DataFeedConfig::default();
        config.min_feed_count = 1;

        let sentinel = DataFeedSentinel::new(config);
        let symbol = Symbol::new("AAPL");

        sentinel.update_feed(symbol.clone(), 150.0, Timestamp::now());

        let order = create_test_order("AAPL");
        let portfolio = Portfolio::default();

        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_fails_without_feed() {
        let mut config = DataFeedConfig::default();
        config.min_feed_count = 1;

        let sentinel = DataFeedSentinel::new(config);

        // No feed updated for AAPL
        let order = create_test_order("AAPL");
        let portfolio = Portfolio::default();

        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_err());
    }

    #[test]
    fn test_sentinel_lifecycle() {
        let sentinel = DataFeedSentinel::with_defaults();

        assert_eq!(sentinel.status(), SentinelStatus::Active);

        sentinel.disable();
        assert_eq!(sentinel.status(), SentinelStatus::Disabled);

        sentinel.enable();
        assert_eq!(sentinel.status(), SentinelStatus::Active);

        sentinel.reset();
        assert_eq!(sentinel.status(), SentinelStatus::Active);
        assert_eq!(sentinel.check_count(), 0);
    }
}
