//! Whale Detection Sentinel.
//!
//! Detects and monitors large institutional flows that could impact markets.
//! Target latency: <15μs
//!
//! ## Scientific References
//!
//! - Kyle (1985): "Continuous Auctions and Insider Trading"
//! - Easley, Lopez de Prado (2012): "Flow Toxicity and Liquidity"

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use crate::core::error::{Result, RiskError};
use crate::core::types::{Order, Portfolio};
use crate::sentinels::base::{Sentinel, SentinelId, SentinelStats, SentinelStatus};

/// Whale detection configuration.
#[derive(Debug, Clone)]
pub struct WhaleConfig {
    /// Threshold as multiple of average daily volume (ADV).
    /// Order > adv_threshold * ADV triggers detection.
    pub adv_threshold: f64,
    /// Absolute notional threshold (USD).
    pub notional_threshold: f64,
    /// Percentage of portfolio considered large.
    pub portfolio_pct_threshold: f64,
    /// Order flow imbalance threshold.
    pub imbalance_threshold: f64,
    /// VPIN (Volume-synchronized PIN) threshold.
    pub vpin_threshold: f64,
    /// Number of buckets for flow tracking.
    pub flow_buckets: usize,
}

impl Default for WhaleConfig {
    fn default() -> Self {
        Self {
            adv_threshold: 0.01,          // 1% of ADV
            notional_threshold: 1_000_000.0, // $1M
            portfolio_pct_threshold: 0.05,   // 5% of portfolio
            imbalance_threshold: 0.70,       // 70% buy or sell imbalance
            vpin_threshold: 0.80,            // VPIN > 0.80 indicates toxicity
            flow_buckets: 50,
        }
    }
}

impl WhaleConfig {
    /// Configuration for small portfolios.
    pub fn small_portfolio() -> Self {
        Self {
            adv_threshold: 0.001,
            notional_threshold: 100_000.0,
            portfolio_pct_threshold: 0.10,
            imbalance_threshold: 0.60,
            vpin_threshold: 0.70,
            flow_buckets: 30,
        }
    }

    /// Configuration for large portfolios.
    pub fn large_portfolio() -> Self {
        Self {
            adv_threshold: 0.05,
            notional_threshold: 10_000_000.0,
            portfolio_pct_threshold: 0.02,
            imbalance_threshold: 0.80,
            vpin_threshold: 0.85,
            flow_buckets: 100,
        }
    }
}

/// Flow bucket for VPIN calculation.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
struct FlowBucket {
    /// Buy volume.
    buy_volume: f64,
    /// Sell volume.
    sell_volume: f64,
    /// Bucket timestamp.
    timestamp: u64,
}

/// Whale detection sentinel.
///
/// Monitors for large flows that could indicate informed trading
/// or market impact concerns.
#[derive(Debug)]
pub struct WhaleSentinel {
    /// Sentinel ID.
    id: SentinelId,
    /// Configuration.
    config: WhaleConfig,
    /// Current status.
    status: AtomicU8,
    /// Current ADV (scaled by 1000).
    adv_scaled: AtomicU64,
    /// Current VPIN (scaled by 1M).
    vpin_scaled: AtomicU64,
    /// Buy volume in current window (scaled).
    buy_volume_scaled: AtomicU64,
    /// Sell volume in current window (scaled).
    sell_volume_scaled: AtomicU64,
    /// Total volume in current window (scaled).
    total_volume_scaled: AtomicU64,
    /// Statistics.
    stats: SentinelStats,
}

impl WhaleSentinel {
    const SCALE: f64 = 1_000_000.0;

    /// Create new whale sentinel.
    pub fn new(config: WhaleConfig) -> Self {
        Self {
            id: SentinelId::new("whale"),
            config,
            status: AtomicU8::new(SentinelStatus::Active as u8),
            adv_scaled: AtomicU64::new(0),
            vpin_scaled: AtomicU64::new(0),
            buy_volume_scaled: AtomicU64::new(0),
            sell_volume_scaled: AtomicU64::new(0),
            total_volume_scaled: AtomicU64::new(0),
            stats: SentinelStats::new(),
        }
    }

    /// Update average daily volume for a symbol.
    pub fn update_adv(&self, adv: f64) {
        let scaled = (adv * Self::SCALE) as u64;
        self.adv_scaled.store(scaled, Ordering::Relaxed);
    }

    /// Update VPIN estimate.
    pub fn update_vpin(&self, vpin: f64) {
        let scaled = (vpin * Self::SCALE) as u64;
        self.vpin_scaled.store(scaled, Ordering::Relaxed);
    }

    /// Record trade for flow tracking.
    pub fn record_trade(&self, is_buy: bool, volume: f64) {
        let scaled = (volume * Self::SCALE) as u64;

        if is_buy {
            self.buy_volume_scaled.fetch_add(scaled, Ordering::Relaxed);
        } else {
            self.sell_volume_scaled.fetch_add(scaled, Ordering::Relaxed);
        }
        self.total_volume_scaled.fetch_add(scaled, Ordering::Relaxed);
    }

    /// Reset flow tracking (call periodically).
    pub fn reset_flow(&self) {
        self.buy_volume_scaled.store(0, Ordering::SeqCst);
        self.sell_volume_scaled.store(0, Ordering::SeqCst);
        self.total_volume_scaled.store(0, Ordering::SeqCst);
    }

    /// Calculate order flow imbalance.
    #[inline]
    fn calculate_imbalance(&self) -> f64 {
        let buy = self.buy_volume_scaled.load(Ordering::Relaxed) as f64;
        let sell = self.sell_volume_scaled.load(Ordering::Relaxed) as f64;
        let total = buy + sell;

        if total == 0.0 {
            return 0.5; // Neutral
        }

        // Imbalance = |buy - sell| / total
        (buy - sell).abs() / total
    }

    /// Check if order is whale-sized relative to ADV.
    #[inline]
    fn check_adv_threshold(&self, order: &Order) -> Result<()> {
        let adv = self.adv_scaled.load(Ordering::Relaxed) as f64 / Self::SCALE;

        if adv <= 0.0 {
            return Ok(()); // No ADV data, skip check
        }

        let order_qty = order.quantity.as_f64();
        let adv_pct = order_qty / adv;

        if adv_pct > self.config.adv_threshold {
            return Err(RiskError::WhaleOrderDetected {
                order_size: order_qty,
                adv_pct,
                threshold: self.config.adv_threshold,
            });
        }

        Ok(())
    }

    /// Check if order exceeds notional threshold.
    #[inline]
    fn check_notional_threshold(&self, order: &Order) -> Result<()> {
        let price = order.limit_price.map(|p| p.as_f64()).unwrap_or(0.0);
        let notional = order.quantity.as_f64() * price;

        if notional > self.config.notional_threshold {
            return Err(RiskError::LargeNotionalOrder {
                notional,
                threshold: self.config.notional_threshold,
            });
        }

        Ok(())
    }

    /// Check if order is large relative to portfolio.
    #[inline]
    fn check_portfolio_pct(&self, order: &Order, portfolio: &Portfolio) -> Result<()> {
        let price = order.limit_price.map(|p| p.as_f64()).unwrap_or(0.0);
        let order_value = order.quantity.as_f64() * price;
        let portfolio_value = portfolio.total_value;

        if portfolio_value <= 0.0 {
            return Ok(());
        }

        let pct = order_value / portfolio_value;

        if pct > self.config.portfolio_pct_threshold {
            return Err(RiskError::LargePortfolioOrder {
                order_pct: pct,
                threshold: self.config.portfolio_pct_threshold,
            });
        }

        Ok(())
    }

    /// Check order flow imbalance.
    #[inline]
    fn check_flow_imbalance(&self) -> Result<()> {
        let imbalance = self.calculate_imbalance();

        if imbalance > self.config.imbalance_threshold {
            return Err(RiskError::FlowImbalanceDetected {
                imbalance,
                threshold: self.config.imbalance_threshold,
            });
        }

        Ok(())
    }

    /// Check VPIN toxicity.
    #[inline]
    fn check_vpin(&self) -> Result<()> {
        let vpin = self.vpin_scaled.load(Ordering::Relaxed) as f64 / Self::SCALE;

        if vpin > self.config.vpin_threshold {
            return Err(RiskError::ToxicFlowDetected {
                vpin,
                threshold: self.config.vpin_threshold,
            });
        }

        Ok(())
    }

    /// Get current VPIN value.
    pub fn current_vpin(&self) -> f64 {
        self.vpin_scaled.load(Ordering::Relaxed) as f64 / Self::SCALE
    }

    /// Get current flow imbalance.
    pub fn current_imbalance(&self) -> f64 {
        self.calculate_imbalance()
    }
}

impl Default for WhaleSentinel {
    fn default() -> Self {
        Self::new(WhaleConfig::default())
    }
}

impl Sentinel for WhaleSentinel {
    fn id(&self) -> SentinelId {
        self.id.clone()
    }

    fn status(&self) -> SentinelStatus {
        match self.status.load(Ordering::Relaxed) {
            0 => SentinelStatus::Active,
            1 => SentinelStatus::Disabled,
            2 => SentinelStatus::Triggered,
            _ => SentinelStatus::Error,
        }
    }

    /// Check whale conditions.
    ///
    /// Target: <15μs
    #[inline]
    fn check(&self, order: &Order, portfolio: &Portfolio) -> Result<()> {
        let start = std::time::Instant::now();

        // Check all whale conditions
        self.check_adv_threshold(order)?;
        self.check_notional_threshold(order)?;
        self.check_portfolio_pct(order, portfolio)?;
        self.check_flow_imbalance()?;
        self.check_vpin()?;

        self.stats.record_check(start.elapsed().as_nanos() as u64);
        Ok(())
    }

    fn reset(&self) {
        self.reset_flow();
        self.status.store(SentinelStatus::Active as u8, Ordering::SeqCst);
        self.stats.reset();
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
    use crate::core::types::{OrderSide, Price, Quantity, Symbol, Timestamp};

    fn test_order(quantity: f64, price: f64) -> Order {
        Order {
            symbol: Symbol::new("AAPL"),
            side: OrderSide::Buy,
            quantity: Quantity::from_f64(quantity),
            limit_price: Some(Price::from_f64(price)),
            strategy_id: 1,
            timestamp: Timestamp::now(),
        }
    }

    #[test]
    fn test_normal_order() {
        let sentinel = WhaleSentinel::default();
        sentinel.update_adv(10_000_000.0); // 10M shares ADV

        let order = test_order(1000.0, 150.0); // Small order
        let portfolio = Portfolio::new(10_000_000.0);

        assert!(sentinel.check(&order, &portfolio).is_ok());
    }

    #[test]
    fn test_whale_order_adv() {
        let sentinel = WhaleSentinel::default();
        sentinel.update_adv(100_000.0); // 100k shares ADV

        // Order for 2k shares = 2% of ADV (> 1% threshold)
        let order = test_order(2000.0, 150.0);
        let portfolio = Portfolio::new(10_000_000.0);

        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_err());
    }

    #[test]
    fn test_notional_threshold() {
        let sentinel = WhaleSentinel::default();
        sentinel.update_adv(10_000_000.0);

        // $1.5M notional (> $1M threshold)
        let order = test_order(10000.0, 150.0);
        let portfolio = Portfolio::new(100_000_000.0);

        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_err());
    }

    #[test]
    fn test_flow_imbalance() {
        let sentinel = WhaleSentinel::default();
        sentinel.update_adv(10_000_000.0);

        // Create heavy buy imbalance
        for _ in 0..100 {
            sentinel.record_trade(true, 1000.0);
        }
        for _ in 0..10 {
            sentinel.record_trade(false, 1000.0);
        }

        let order = test_order(100.0, 150.0);
        let portfolio = Portfolio::new(10_000_000.0);

        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_err());
    }

    #[test]
    fn test_vpin_threshold() {
        let sentinel = WhaleSentinel::default();
        sentinel.update_adv(10_000_000.0);
        sentinel.update_vpin(0.85); // High VPIN

        let order = test_order(100.0, 150.0);
        let portfolio = Portfolio::new(10_000_000.0);

        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_err());
    }

    #[test]
    fn test_latency() {
        let sentinel = WhaleSentinel::default();
        sentinel.update_adv(10_000_000.0);

        let order = test_order(100.0, 150.0);
        let portfolio = Portfolio::new(10_000_000.0);

        // Warm up
        for _ in 0..1000 {
            let _ = sentinel.check(&order, &portfolio);
        }

        // Measure
        let start = std::time::Instant::now();
        for _ in 0..10000 {
            let _ = sentinel.check(&order, &portfolio);
        }
        let elapsed = start.elapsed();
        let avg_ns = elapsed.as_nanos() / 10000;

        assert!(avg_ns < 15000, "Whale check too slow: {}ns average", avg_ns);
    }
}
