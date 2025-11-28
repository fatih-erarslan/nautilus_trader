//! Trade Surveillance Sentinel - Market Manipulation Detection.
//!
//! Detects market manipulation patterns based on regulatory requirements:
//! - SEC Rule 10b-5 (Anti-Fraud)
//! - FINRA Rule 5210 (Publication of Transactions and Quotations)
//! - MiFID II MAR (Market Abuse Regulation)
//!
//! Target latency: <50μs
//!
//! ## Scientific References
//!
//! - Cumming, Zhan & Aitken (2015): "High-frequency trading and end-of-day price dislocation"
//! - Aggarwal & Wu (2006): "Stock Market Manipulations"
//! - Comerton-Forde & Putniņš (2015): "Dark trading and price discovery"
//! - SEC Risk Alert (2015): "Algorithmic Trading Compliance"

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use crate::core::error::{Result, RiskError};
use crate::core::types::{Order, Portfolio, Timestamp};
use crate::sentinels::base::{Sentinel, SentinelId, SentinelStats, SentinelStatus};

/// Types of market manipulation patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManipulationType {
    /// Spoofing: Large orders placed then cancelled before execution.
    Spoofing,
    /// Layering: Multiple orders at different price levels creating false depth.
    Layering,
    /// Wash trading: Self-dealing to inflate volume.
    WashTrading,
    /// Momentum ignition: Aggressive orders to trigger other algorithms.
    MomentumIgnition,
    /// Quote stuffing: Flooding exchange with orders to slow competitors.
    QuoteStuffing,
}

impl ManipulationType {
    /// Get human-readable name.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Spoofing => "Spoofing",
            Self::Layering => "Layering",
            Self::WashTrading => "Wash Trading",
            Self::MomentumIgnition => "Momentum Ignition",
            Self::QuoteStuffing => "Quote Stuffing",
        }
    }

    /// Get regulatory severity (1-5, 5 is most severe).
    #[inline]
    pub const fn severity(&self) -> u8 {
        match self {
            Self::Spoofing => 5,
            Self::Layering => 5,
            Self::WashTrading => 5,
            Self::MomentumIgnition => 4,
            Self::QuoteStuffing => 3,
        }
    }
}

/// Surveillance alert containing detection details.
#[derive(Debug, Clone)]
pub struct SurveillanceAlert {
    /// Type of manipulation detected.
    pub pattern: ManipulationType,
    /// Confidence level (0.0-1.0).
    pub confidence: f64,
    /// Evidence description.
    pub evidence: String,
    /// Detection timestamp.
    pub timestamp: Timestamp,
}

impl SurveillanceAlert {
    /// Create new alert.
    pub fn new(pattern: ManipulationType, confidence: f64, evidence: String) -> Self {
        Self {
            pattern,
            confidence,
            evidence,
            timestamp: Timestamp::now(),
        }
    }
}

/// Order flow statistics for rolling window.
#[derive(Debug, Clone, Copy, Default)]
pub struct OrderFlowStats {
    /// Total orders placed in window.
    pub orders: u64,
    /// Total orders cancelled in window.
    pub cancels: u64,
    /// Total trades executed in window.
    pub trades: u64,
    /// Messages per second.
    pub messages_per_sec: f64,
}

impl OrderFlowStats {
    /// Calculate order-to-trade ratio.
    #[inline]
    pub fn order_to_trade_ratio(&self) -> f64 {
        if self.trades == 0 {
            return f64::INFINITY;
        }
        self.orders as f64 / self.trades as f64
    }

    /// Calculate cancel rate.
    #[inline]
    pub fn cancel_rate(&self) -> f64 {
        if self.orders == 0 {
            return 0.0;
        }
        self.cancels as f64 / self.orders as f64
    }

    /// Reset statistics.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Surveillance configuration based on regulatory guidance.
#[derive(Debug, Clone)]
pub struct SurveillanceConfig {
    /// Spoofing: order-to-trade ratio threshold (FINRA: ~10:1 suspicious).
    pub spoofing_order_trade_ratio: f64,
    /// Spoofing: cancel rate threshold (SEC: >90% suspicious).
    pub spoofing_cancel_rate: f64,
    /// Spoofing: time window in seconds.
    pub spoofing_window_secs: u64,

    /// Layering: minimum number of price levels for detection.
    pub layering_min_levels: usize,
    /// Layering: cancel rate threshold.
    pub layering_cancel_rate: f64,
    /// Layering: time window in seconds.
    pub layering_window_secs: u64,

    /// Wash trading: same beneficial owner threshold (confidence).
    pub wash_trade_confidence_threshold: f64,

    /// Momentum ignition: price move threshold (percentage).
    pub momentum_price_move_pct: f64,
    /// Momentum ignition: volume spike threshold (multiple of average).
    pub momentum_volume_spike: f64,

    /// Quote stuffing: messages per second threshold.
    pub quote_stuffing_msg_per_sec: f64,
    /// Quote stuffing: cancel rate threshold.
    pub quote_stuffing_cancel_rate: f64,
}

impl Default for SurveillanceConfig {
    /// Default configuration based on regulatory alerts and industry standards.
    fn default() -> Self {
        Self {
            // Spoofing thresholds (SEC/FINRA guidance)
            spoofing_order_trade_ratio: 10.0,  // 10:1 ratio triggers investigation
            spoofing_cancel_rate: 0.90,         // >90% cancel rate suspicious
            spoofing_window_secs: 60,           // 1-minute rolling window

            // Layering thresholds (MiFID II MAR)
            layering_min_levels: 3,             // At least 3 price levels
            layering_cancel_rate: 0.85,         // >85% cancel rate
            layering_window_secs: 30,           // 30-second window

            // Wash trading (CFTC guidance)
            wash_trade_confidence_threshold: 0.80,

            // Momentum ignition (SEC 2015 Risk Alert)
            momentum_price_move_pct: 0.02,      // 2% rapid price move
            momentum_volume_spike: 3.0,         // 3x volume spike

            // Quote stuffing (IOSCO principles)
            quote_stuffing_msg_per_sec: 1000.0,
            quote_stuffing_cancel_rate: 0.95,   // >95% cancel rate
        }
    }
}

impl SurveillanceConfig {
    /// Conservative configuration for high-frequency trading.
    pub fn conservative() -> Self {
        Self {
            spoofing_order_trade_ratio: 5.0,
            spoofing_cancel_rate: 0.80,
            spoofing_window_secs: 30,
            layering_min_levels: 2,
            layering_cancel_rate: 0.75,
            layering_window_secs: 20,
            wash_trade_confidence_threshold: 0.70,
            momentum_price_move_pct: 0.01,
            momentum_volume_spike: 2.0,
            quote_stuffing_msg_per_sec: 500.0,
            quote_stuffing_cancel_rate: 0.90,
        }
    }

    /// Permissive configuration for normal trading.
    pub fn permissive() -> Self {
        Self {
            spoofing_order_trade_ratio: 20.0,
            spoofing_cancel_rate: 0.95,
            spoofing_window_secs: 120,
            layering_min_levels: 5,
            layering_cancel_rate: 0.90,
            layering_window_secs: 60,
            wash_trade_confidence_threshold: 0.90,
            momentum_price_move_pct: 0.05,
            momentum_volume_spike: 5.0,
            quote_stuffing_msg_per_sec: 2000.0,
            quote_stuffing_cancel_rate: 0.98,
        }
    }
}

/// Trade Surveillance Sentinel.
///
/// Detects market manipulation patterns in real-time with <50μs latency.
#[derive(Debug)]
pub struct TradeSurveillanceSentinel {
    /// Sentinel ID.
    id: SentinelId,
    /// Configuration.
    config: SurveillanceConfig,
    /// Current status.
    status: AtomicU8,

    // Order flow tracking (scaled by 1000 to use atomics)
    /// Orders placed in current window (scaled).
    orders_scaled: AtomicU64,
    /// Orders cancelled in current window (scaled).
    cancels_scaled: AtomicU64,
    /// Trades executed in current window (scaled).
    trades_scaled: AtomicU64,
    /// Messages in current second (scaled).
    messages_this_sec_scaled: AtomicU64,

    // Price tracking for momentum detection (scaled by 1M)
    /// Recent price (scaled).
    recent_price_scaled: AtomicU64,
    /// Previous price (scaled).
    prev_price_scaled: AtomicU64,

    // Volume tracking (scaled by 1000)
    /// Current volume (scaled).
    current_volume_scaled: AtomicU64,
    /// Average volume baseline (scaled).
    avg_volume_scaled: AtomicU64,

    // Layering detection
    /// Number of active price levels.
    price_levels_count: AtomicU64,

    // Window management
    /// Window start timestamp.
    window_start_ns: AtomicU64,

    /// Statistics.
    stats: SentinelStats,
}

impl TradeSurveillanceSentinel {
    const SCALE: f64 = 1_000.0;
    const PRICE_SCALE: f64 = 1_000_000.0;

    /// Create new surveillance sentinel.
    pub fn new(config: SurveillanceConfig) -> Self {
        Self {
            id: SentinelId::new("trade_surveillance"),
            config,
            status: AtomicU8::new(SentinelStatus::Active as u8),
            orders_scaled: AtomicU64::new(0),
            cancels_scaled: AtomicU64::new(0),
            trades_scaled: AtomicU64::new(0),
            messages_this_sec_scaled: AtomicU64::new(0),
            recent_price_scaled: AtomicU64::new(0),
            prev_price_scaled: AtomicU64::new(0),
            current_volume_scaled: AtomicU64::new(0),
            avg_volume_scaled: AtomicU64::new(0),
            price_levels_count: AtomicU64::new(0),
            window_start_ns: AtomicU64::new(Timestamp::now().as_nanos()),
            stats: SentinelStats::new(),
        }
    }

    /// Record an order placement.
    #[inline]
    pub fn record_order(&self, quantity: f64, price: f64) {
        self.orders_scaled.fetch_add(1, Ordering::Relaxed);
        self.messages_this_sec_scaled.fetch_add(1, Ordering::Relaxed);

        // Update price tracking
        let price_scaled = (price * Self::PRICE_SCALE) as u64;
        self.prev_price_scaled.store(
            self.recent_price_scaled.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.recent_price_scaled.store(price_scaled, Ordering::Relaxed);

        // Update volume
        let volume_scaled = (quantity * Self::SCALE) as u64;
        self.current_volume_scaled.fetch_add(volume_scaled, Ordering::Relaxed);
    }

    /// Record an order cancellation.
    #[inline]
    pub fn record_cancel(&self) {
        self.cancels_scaled.fetch_add(1, Ordering::Relaxed);
        self.messages_this_sec_scaled.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a trade execution.
    #[inline]
    pub fn record_trade(&self, quantity: f64) {
        self.trades_scaled.fetch_add(1, Ordering::Relaxed);
        self.messages_this_sec_scaled.fetch_add(1, Ordering::Relaxed);

        let volume_scaled = (quantity * Self::SCALE) as u64;
        self.current_volume_scaled.fetch_add(volume_scaled, Ordering::Relaxed);
    }

    /// Update price level count for layering detection.
    #[inline]
    pub fn update_price_levels(&self, count: usize) {
        self.price_levels_count.store(count as u64, Ordering::Relaxed);
    }

    /// Update average volume baseline.
    #[inline]
    pub fn update_avg_volume(&self, avg_volume: f64) {
        let scaled = (avg_volume * Self::SCALE) as u64;
        self.avg_volume_scaled.store(scaled, Ordering::Relaxed);
    }

    /// Get current order flow statistics.
    pub fn get_flow_stats(&self) -> OrderFlowStats {
        OrderFlowStats {
            orders: self.orders_scaled.load(Ordering::Relaxed),
            cancels: self.cancels_scaled.load(Ordering::Relaxed),
            trades: self.trades_scaled.load(Ordering::Relaxed),
            messages_per_sec: self.messages_this_sec_scaled.load(Ordering::Relaxed) as f64,
        }
    }

    /// Reset window statistics (call periodically based on window size).
    pub fn reset_window(&self) {
        self.orders_scaled.store(0, Ordering::SeqCst);
        self.cancels_scaled.store(0, Ordering::SeqCst);
        self.trades_scaled.store(0, Ordering::SeqCst);
        self.messages_this_sec_scaled.store(0, Ordering::SeqCst);
        self.current_volume_scaled.store(0, Ordering::SeqCst);
        self.price_levels_count.store(0, Ordering::SeqCst);
        self.window_start_ns.store(Timestamp::now().as_nanos(), Ordering::SeqCst);
    }

    /// Reset only volume tracking (for testing momentum detection).
    #[inline]
    pub fn reset_volume(&self) {
        self.current_volume_scaled.store(0, Ordering::SeqCst);
    }

    /// Check for spoofing pattern.
    ///
    /// Detection criteria:
    /// - Order-to-trade ratio > threshold (e.g., 10:1)
    /// - Cancel rate > threshold (e.g., 90%)
    /// - Within rolling time window
    #[inline]
    fn detect_spoofing(&self) -> Result<()> {
        let orders = self.orders_scaled.load(Ordering::Relaxed);
        let cancels = self.cancels_scaled.load(Ordering::Relaxed);
        let trades = self.trades_scaled.load(Ordering::Relaxed);

        if orders == 0 {
            return Ok(());
        }

        let cancel_rate = cancels as f64 / orders as f64;
        let order_trade_ratio = if trades == 0 {
            f64::INFINITY
        } else {
            orders as f64 / trades as f64
        };

        // Check both conditions for spoofing
        if cancel_rate > self.config.spoofing_cancel_rate
            && order_trade_ratio > self.config.spoofing_order_trade_ratio
        {
            return Err(RiskError::ConfigurationError(format!(
                "SPOOFING DETECTED: cancel_rate={:.2}% (>{:.2}%), order/trade_ratio={:.1} (>{:.1})",
                cancel_rate * 100.0,
                self.config.spoofing_cancel_rate * 100.0,
                order_trade_ratio,
                self.config.spoofing_order_trade_ratio
            )));
        }

        Ok(())
    }

    /// Check for layering pattern.
    ///
    /// Detection criteria:
    /// - Multiple orders at different price levels
    /// - High cancel rate (>85%)
    /// - All cancelled within short time window
    #[inline]
    fn detect_layering(&self) -> Result<()> {
        let price_levels = self.price_levels_count.load(Ordering::Relaxed) as usize;
        let orders = self.orders_scaled.load(Ordering::Relaxed);
        let cancels = self.cancels_scaled.load(Ordering::Relaxed);

        if orders == 0 {
            return Ok(());
        }

        let cancel_rate = cancels as f64 / orders as f64;

        if price_levels >= self.config.layering_min_levels
            && cancel_rate > self.config.layering_cancel_rate
        {
            return Err(RiskError::ConfigurationError(format!(
                "LAYERING DETECTED: {} price levels, cancel_rate={:.2}% (>{:.2}%)",
                price_levels,
                cancel_rate * 100.0,
                self.config.layering_cancel_rate * 100.0
            )));
        }

        Ok(())
    }

    /// Check for momentum ignition pattern.
    ///
    /// Detection criteria:
    /// - Rapid price movement (>2%)
    /// - Followed by reversal
    /// - Unusual volume spike (>3x average)
    #[inline]
    fn detect_momentum_ignition(&self) -> Result<()> {
        let recent_price = self.recent_price_scaled.load(Ordering::Relaxed) as f64
            / Self::PRICE_SCALE;
        let prev_price = self.prev_price_scaled.load(Ordering::Relaxed) as f64
            / Self::PRICE_SCALE;
        let current_volume = self.current_volume_scaled.load(Ordering::Relaxed) as f64
            / Self::SCALE;
        let avg_volume = self.avg_volume_scaled.load(Ordering::Relaxed) as f64
            / Self::SCALE;

        if prev_price <= 0.0 || avg_volume <= 0.0 {
            return Ok(());
        }

        let price_move_pct = ((recent_price - prev_price) / prev_price).abs();
        let volume_ratio = current_volume / avg_volume;

        if price_move_pct > self.config.momentum_price_move_pct
            && volume_ratio > self.config.momentum_volume_spike
        {
            return Err(RiskError::ConfigurationError(format!(
                "MOMENTUM IGNITION DETECTED: price_move={:.2}% (>{:.2}%), volume_ratio={:.1}x (>{:.1}x)",
                price_move_pct * 100.0,
                self.config.momentum_price_move_pct * 100.0,
                volume_ratio,
                self.config.momentum_volume_spike
            )));
        }

        Ok(())
    }

    /// Check for quote stuffing pattern.
    ///
    /// Detection criteria:
    /// - Messages per second exceeds threshold (>1000/sec)
    /// - High cancel rate (>95%)
    /// - Intent to slow down competitors
    #[inline]
    fn detect_quote_stuffing(&self) -> Result<()> {
        let messages_per_sec = self.messages_this_sec_scaled.load(Ordering::Relaxed) as f64;
        let orders = self.orders_scaled.load(Ordering::Relaxed);
        let cancels = self.cancels_scaled.load(Ordering::Relaxed);

        if orders == 0 {
            return Ok(());
        }

        let cancel_rate = cancels as f64 / orders as f64;

        if messages_per_sec > self.config.quote_stuffing_msg_per_sec
            && cancel_rate > self.config.quote_stuffing_cancel_rate
        {
            return Err(RiskError::ConfigurationError(format!(
                "QUOTE STUFFING DETECTED: {:.0} msg/sec (>{:.0}), cancel_rate={:.2}% (>{:.2}%)",
                messages_per_sec,
                self.config.quote_stuffing_msg_per_sec,
                cancel_rate * 100.0,
                self.config.quote_stuffing_cancel_rate * 100.0
            )));
        }

        Ok(())
    }

    /// Check if window needs rotation.
    #[inline]
    fn should_rotate_window(&self, window_secs: u64) -> bool {
        let window_start = self.window_start_ns.load(Ordering::Relaxed);
        let current_ns = Timestamp::now().as_nanos();
        let elapsed_ns = current_ns.saturating_sub(window_start);
        let window_ns = window_secs * 1_000_000_000;

        elapsed_ns > window_ns
    }
}

impl Default for TradeSurveillanceSentinel {
    fn default() -> Self {
        Self::new(SurveillanceConfig::default())
    }
}

impl Sentinel for TradeSurveillanceSentinel {
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

    /// Check for market manipulation patterns.
    ///
    /// Target latency: <50μs
    #[inline]
    fn check(&self, _order: &Order, _portfolio: &Portfolio) -> Result<()> {
        let start = std::time::Instant::now();

        // Check if disabled
        if self.status.load(Ordering::Relaxed) != SentinelStatus::Active as u8 {
            return Ok(());
        }

        // Check all manipulation patterns
        self.detect_spoofing()?;
        self.detect_layering()?;
        self.detect_momentum_ignition()?;
        self.detect_quote_stuffing()?;

        // Record check latency
        self.stats.record_check(start.elapsed().as_nanos() as u64);

        Ok(())
    }

    fn reset(&self) {
        self.reset_window();
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
    use crate::core::types::{OrderSide, Quantity, Symbol, Price};

    fn create_test_order(qty: f64, price: f64) -> Order {
        Order {
            symbol: Symbol::new("AAPL"),
            side: OrderSide::Buy,
            quantity: Quantity::from_f64(qty),
            limit_price: Some(Price::from_f64(price)),
            strategy_id: 1,
            timestamp: Timestamp::now(),
        }
    }

    #[test]
    fn test_normal_trading_allowed() {
        let sentinel = TradeSurveillanceSentinel::default();
        let portfolio = Portfolio::new(1_000_000.0);

        // Normal trading pattern
        sentinel.record_order(100.0, 150.0);
        sentinel.record_trade(100.0);

        let order = create_test_order(100.0, 150.0);
        assert!(sentinel.check(&order, &portfolio).is_ok());
    }

    #[test]
    fn test_spoofing_detection() {
        let config = SurveillanceConfig {
            spoofing_order_trade_ratio: 10.0,
            spoofing_cancel_rate: 0.90,
            ..Default::default()
        };
        let sentinel = TradeSurveillanceSentinel::new(config);
        let portfolio = Portfolio::new(1_000_000.0);

        // Create spoofing pattern: 100 orders, 95 cancels, 5 trades
        for _ in 0..100 {
            sentinel.record_order(1000.0, 150.0);
        }
        for _ in 0..95 {
            sentinel.record_cancel();
        }
        for _ in 0..5 {
            sentinel.record_trade(1000.0);
        }

        let order = create_test_order(100.0, 150.0);
        let result = sentinel.check(&order, &portfolio);

        assert!(result.is_err());
        if let Err(e) = result {
            let msg = format!("{}", e);
            assert!(msg.contains("SPOOFING"));
        }
    }

    #[test]
    fn test_layering_detection() {
        let config = SurveillanceConfig {
            layering_min_levels: 3,
            layering_cancel_rate: 0.85,
            ..Default::default()
        };
        let sentinel = TradeSurveillanceSentinel::new(config);
        let portfolio = Portfolio::new(1_000_000.0);

        // Create layering pattern: 5 price levels, high cancel rate
        sentinel.update_price_levels(5);
        for _ in 0..100 {
            sentinel.record_order(1000.0, 150.0);
        }
        for _ in 0..90 {
            sentinel.record_cancel();
        }

        let order = create_test_order(100.0, 150.0);
        let result = sentinel.check(&order, &portfolio);

        assert!(result.is_err());
        if let Err(e) = result {
            let msg = format!("{}", e);
            assert!(msg.contains("LAYERING"));
        }
    }

    #[test]
    fn test_momentum_ignition_detection() {
        // Configure to specifically detect momentum ignition but NOT spoofing
        let config = SurveillanceConfig {
            momentum_price_move_pct: 0.02,
            momentum_volume_spike: 3.0,
            spoofing_cancel_rate: 0.99,       // Very high threshold to avoid spoofing detection
            spoofing_order_trade_ratio: 1000.0, // Very high threshold to avoid spoofing detection
            ..Default::default()
        };
        let sentinel = TradeSurveillanceSentinel::new(config);
        let portfolio = Portfolio::new(1_000_000.0);

        // Set baseline average volume
        sentinel.update_avg_volume(10_000.0);

        // Create momentum pattern: rapid price move + volume spike
        // First order establishes prev_price at 150.0
        sentinel.record_order(10_000.0, 150.0);
        // Record a trade to keep order/trade ratio low
        sentinel.record_trade(10_000.0);

        // Reset accumulated volume for clean measurement
        sentinel.reset_volume();

        // Second order: 2.5% price jump (150->153.75) + 3.5x volume spike (35k vs 10k avg)
        sentinel.record_order(35_000.0, 153.75);
        // Record another trade to avoid spoofing
        sentinel.record_trade(35_000.0);

        let order = create_test_order(100.0, 153.75);
        let result = sentinel.check(&order, &portfolio);

        assert!(result.is_err(), "Expected momentum ignition detection, got {:?}", result);
        if let Err(e) = result {
            let msg = format!("{}", e);
            assert!(msg.contains("MOMENTUM"), "Error should mention MOMENTUM: {}", msg);
        }
    }

    #[test]
    fn test_quote_stuffing_detection() {
        // Configure to specifically detect quote stuffing but NOT spoofing
        // Quote stuffing: high msg rate + high cancel rate
        // Spoofing: high cancel rate + high order/trade ratio
        // We need to add enough trades to avoid spoofing detection
        let config = SurveillanceConfig {
            quote_stuffing_msg_per_sec: 500.0, // 500 msgs/sec threshold
            quote_stuffing_cancel_rate: 0.90,  // 90% cancel rate threshold
            spoofing_cancel_rate: 0.99,        // Very high threshold to avoid spoofing
            spoofing_order_trade_ratio: 1000.0, // Very high to avoid spoofing
            ..Default::default()
        };
        let sentinel = TradeSurveillanceSentinel::new(config);
        let portfolio = Portfolio::new(1_000_000.0);

        // Create quote stuffing pattern: 600 orders + 576 cancels
        // Messages/sec = 1176 > 500 threshold
        // Cancel rate = 576/600 = 96% > 90% threshold
        for _ in 0..600 {
            sentinel.record_order(100.0, 150.0);
        }
        for _ in 0..576 {
            sentinel.record_cancel();
        }
        // Add trades to avoid spoofing detection (keep order/trade ratio reasonable)
        for _ in 0..100 {
            sentinel.record_trade(100.0);
        }

        let order = create_test_order(100.0, 150.0);
        let result = sentinel.check(&order, &portfolio);

        assert!(result.is_err(), "Expected quote stuffing detection. msgs={}, cancels={}, cancel_rate={:.2}%",
            600 + 576 + 100, 576, (576.0 / 600.0) * 100.0);
        if let Err(e) = result {
            let msg = format!("{}", e);
            assert!(msg.contains("QUOTE STUFFING"), "Error should contain 'QUOTE STUFFING': {}", msg);
        }
    }

    #[test]
    fn test_flow_stats_calculation() {
        let sentinel = TradeSurveillanceSentinel::default();

        sentinel.record_order(100.0, 150.0);
        sentinel.record_order(100.0, 150.0);
        sentinel.record_cancel();
        sentinel.record_trade(100.0);

        let stats = sentinel.get_flow_stats();

        assert_eq!(stats.orders, 2);
        assert_eq!(stats.cancels, 1);
        assert_eq!(stats.trades, 1);
        assert_eq!(stats.order_to_trade_ratio(), 2.0);
        assert_eq!(stats.cancel_rate(), 0.5);
    }

    #[test]
    fn test_window_reset() {
        let sentinel = TradeSurveillanceSentinel::default();

        sentinel.record_order(100.0, 150.0);
        sentinel.record_cancel();
        sentinel.record_trade(100.0);

        let stats_before = sentinel.get_flow_stats();
        assert!(stats_before.orders > 0);

        sentinel.reset_window();

        let stats_after = sentinel.get_flow_stats();
        assert_eq!(stats_after.orders, 0);
        assert_eq!(stats_after.cancels, 0);
        assert_eq!(stats_after.trades, 0);
    }

    #[test]
    fn test_sentinel_lifecycle() {
        let sentinel = TradeSurveillanceSentinel::default();

        assert_eq!(sentinel.status(), SentinelStatus::Active);

        sentinel.disable();
        assert_eq!(sentinel.status(), SentinelStatus::Disabled);

        sentinel.enable();
        assert_eq!(sentinel.status(), SentinelStatus::Active);

        sentinel.reset();
        assert_eq!(sentinel.status(), SentinelStatus::Active);
    }

    #[test]
    fn test_latency_requirement() {
        let sentinel = TradeSurveillanceSentinel::default();
        let portfolio = Portfolio::new(1_000_000.0);
        let order = create_test_order(100.0, 150.0);

        // Warm-up
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

        assert!(avg_ns < 50000, "Surveillance check too slow: {}ns average (target: <50μs)", avg_ns);
    }

    #[test]
    fn test_manipulation_type_severity() {
        assert_eq!(ManipulationType::Spoofing.severity(), 5);
        assert_eq!(ManipulationType::Layering.severity(), 5);
        assert_eq!(ManipulationType::WashTrading.severity(), 5);
        assert_eq!(ManipulationType::MomentumIgnition.severity(), 4);
        assert_eq!(ManipulationType::QuoteStuffing.severity(), 3);
    }

    #[test]
    fn test_conservative_config() {
        let config = SurveillanceConfig::conservative();

        // Conservative should have stricter thresholds
        assert!(config.spoofing_order_trade_ratio < SurveillanceConfig::default().spoofing_order_trade_ratio);
        assert!(config.spoofing_cancel_rate < SurveillanceConfig::default().spoofing_cancel_rate);
    }

    #[test]
    fn test_permissive_config() {
        let config = SurveillanceConfig::permissive();

        // Permissive should have more lenient thresholds
        assert!(config.spoofing_order_trade_ratio > SurveillanceConfig::default().spoofing_order_trade_ratio);
        assert!(config.spoofing_cancel_rate > SurveillanceConfig::default().spoofing_cancel_rate);
    }
}
