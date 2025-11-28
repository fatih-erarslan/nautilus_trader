//! Drawdown Sentinel.
//!
//! Monitors portfolio drawdown and triggers protective actions.
//! Target latency: <5μs

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use crate::core::error::{Result, RiskError};
use crate::core::types::{Order, Portfolio};
use crate::sentinels::base::{Sentinel, SentinelId, SentinelStats, SentinelStatus};

/// Drawdown thresholds configuration.
#[derive(Debug, Clone)]
pub struct DrawdownConfig {
    /// Warning threshold (e.g., 0.05 = 5% drawdown).
    pub warning_threshold: f64,
    /// Critical threshold - restrict new positions.
    pub critical_threshold: f64,
    /// Emergency threshold - halt trading.
    pub emergency_threshold: f64,
    /// Daily loss limit as fraction of portfolio.
    pub daily_loss_limit: f64,
    /// Weekly loss limit as fraction of portfolio.
    pub weekly_loss_limit: f64,
}

impl Default for DrawdownConfig {
    fn default() -> Self {
        Self {
            warning_threshold: 0.05,     // 5%
            critical_threshold: 0.10,    // 10%
            emergency_threshold: 0.15,   // 15%
            daily_loss_limit: 0.02,      // 2%
            weekly_loss_limit: 0.05,     // 5%
        }
    }
}

impl DrawdownConfig {
    /// Conservative thresholds for low-risk strategies.
    pub fn conservative() -> Self {
        Self {
            warning_threshold: 0.02,
            critical_threshold: 0.05,
            emergency_threshold: 0.08,
            daily_loss_limit: 0.01,
            weekly_loss_limit: 0.03,
        }
    }

    /// Aggressive thresholds for high-risk strategies.
    pub fn aggressive() -> Self {
        Self {
            warning_threshold: 0.10,
            critical_threshold: 0.20,
            emergency_threshold: 0.30,
            daily_loss_limit: 0.05,
            weekly_loss_limit: 0.10,
        }
    }
}

/// Drawdown level for risk classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawdownLevel {
    /// Normal operation.
    Normal,
    /// Warning level - increased monitoring.
    Warning,
    /// Critical level - restrict new positions.
    Critical,
    /// Emergency level - halt trading.
    Emergency,
}

impl DrawdownLevel {
    /// Get numeric severity (0-3).
    pub fn severity(&self) -> u8 {
        match self {
            Self::Normal => 0,
            Self::Warning => 1,
            Self::Critical => 2,
            Self::Emergency => 3,
        }
    }
}

/// Drawdown sentinel.
///
/// Monitors current drawdown and enforces loss limits.
/// Uses atomic operations for lock-free updates.
#[derive(Debug)]
pub struct DrawdownSentinel {
    /// Sentinel ID.
    id: SentinelId,
    /// Configuration.
    config: DrawdownConfig,
    /// Current status.
    status: AtomicU8,
    /// Current drawdown level.
    drawdown_level: AtomicU8,
    /// High water mark (scaled by 1000 for atomic storage).
    high_water_mark_scaled: AtomicU64,
    /// Daily starting value (scaled).
    daily_start_scaled: AtomicU64,
    /// Weekly starting value (scaled).
    weekly_start_scaled: AtomicU64,
    /// Statistics.
    stats: SentinelStats,
}

impl DrawdownSentinel {
    /// Scaling factor for atomic storage of f64 values.
    const SCALE: f64 = 1_000_000.0;

    /// Create new drawdown sentinel.
    pub fn new(config: DrawdownConfig) -> Self {
        Self {
            id: SentinelId::new("drawdown"),
            config,
            status: AtomicU8::new(SentinelStatus::Active as u8),
            drawdown_level: AtomicU8::new(DrawdownLevel::Normal as u8),
            high_water_mark_scaled: AtomicU64::new(0),
            daily_start_scaled: AtomicU64::new(0),
            weekly_start_scaled: AtomicU64::new(0),
            stats: SentinelStats::new(),
        }
    }

    /// Initialize with starting portfolio value.
    pub fn initialize(&self, initial_value: f64) {
        let scaled = (initial_value * Self::SCALE) as u64;
        self.high_water_mark_scaled.store(scaled, Ordering::SeqCst);
        self.daily_start_scaled.store(scaled, Ordering::SeqCst);
        self.weekly_start_scaled.store(scaled, Ordering::SeqCst);
    }

    /// Update high water mark if portfolio value increased.
    pub fn update_high_water_mark(&self, current_value: f64) {
        let current_scaled = (current_value * Self::SCALE) as u64;
        let mut hwm = self.high_water_mark_scaled.load(Ordering::Relaxed);

        while current_scaled > hwm {
            match self.high_water_mark_scaled.compare_exchange_weak(
                hwm,
                current_scaled,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => hwm = x,
            }
        }
    }

    /// Reset daily tracking (call at market open).
    pub fn reset_daily(&self, current_value: f64) {
        let scaled = (current_value * Self::SCALE) as u64;
        self.daily_start_scaled.store(scaled, Ordering::SeqCst);
    }

    /// Reset weekly tracking (call at week start).
    pub fn reset_weekly(&self, current_value: f64) {
        let scaled = (current_value * Self::SCALE) as u64;
        self.weekly_start_scaled.store(scaled, Ordering::SeqCst);
    }

    /// Calculate current drawdown from high water mark.
    #[inline]
    fn calculate_drawdown(&self, current_value: f64) -> f64 {
        let hwm = self.high_water_mark_scaled.load(Ordering::Relaxed) as f64 / Self::SCALE;
        if hwm <= 0.0 {
            return 0.0;
        }
        (hwm - current_value) / hwm
    }

    /// Calculate daily loss.
    #[inline]
    fn calculate_daily_loss(&self, current_value: f64) -> f64 {
        let start = self.daily_start_scaled.load(Ordering::Relaxed) as f64 / Self::SCALE;
        if start <= 0.0 {
            return 0.0;
        }
        (start - current_value) / start
    }

    /// Calculate weekly loss.
    #[inline]
    fn calculate_weekly_loss(&self, current_value: f64) -> f64 {
        let start = self.weekly_start_scaled.load(Ordering::Relaxed) as f64 / Self::SCALE;
        if start <= 0.0 {
            return 0.0;
        }
        (start - current_value) / start
    }

    /// Determine drawdown level based on current drawdown.
    #[inline]
    fn determine_level(&self, drawdown: f64) -> DrawdownLevel {
        if drawdown >= self.config.emergency_threshold {
            DrawdownLevel::Emergency
        } else if drawdown >= self.config.critical_threshold {
            DrawdownLevel::Critical
        } else if drawdown >= self.config.warning_threshold {
            DrawdownLevel::Warning
        } else {
            DrawdownLevel::Normal
        }
    }

    /// Get current drawdown level.
    pub fn current_level(&self) -> DrawdownLevel {
        match self.drawdown_level.load(Ordering::Relaxed) {
            0 => DrawdownLevel::Normal,
            1 => DrawdownLevel::Warning,
            2 => DrawdownLevel::Critical,
            _ => DrawdownLevel::Emergency,
        }
    }
}

impl Default for DrawdownSentinel {
    fn default() -> Self {
        Self::new(DrawdownConfig::default())
    }
}

impl Sentinel for DrawdownSentinel {
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

    /// Check drawdown limits.
    ///
    /// Target: <5μs
    #[inline]
    fn check(&self, _order: &Order, portfolio: &Portfolio) -> Result<()> {
        let start = std::time::Instant::now();
        let current_value = portfolio.total_value;

        // Update high water mark
        self.update_high_water_mark(current_value);

        // Calculate drawdown metrics
        let drawdown = self.calculate_drawdown(current_value);
        let daily_loss = self.calculate_daily_loss(current_value);
        let weekly_loss = self.calculate_weekly_loss(current_value);

        // Determine and store level
        let level = self.determine_level(drawdown);
        self.drawdown_level.store(level as u8, Ordering::Relaxed);

        // Check emergency threshold
        if level == DrawdownLevel::Emergency {
            self.status.store(SentinelStatus::Triggered as u8, Ordering::SeqCst);
            self.stats.record_trigger();
            self.stats.record_check(start.elapsed().as_nanos() as u64);
            return Err(RiskError::EmergencyDrawdown {
                current: drawdown,
                threshold: self.config.emergency_threshold,
            });
        }

        // Check critical threshold - reject new positions
        if level == DrawdownLevel::Critical {
            self.stats.record_check(start.elapsed().as_nanos() as u64);
            return Err(RiskError::DrawdownLimitExceeded {
                current: drawdown,
                limit: self.config.critical_threshold,
            });
        }

        // Check daily loss limit
        if daily_loss > self.config.daily_loss_limit {
            self.stats.record_trigger();
            self.stats.record_check(start.elapsed().as_nanos() as u64);
            return Err(RiskError::DailyLossLimitExceeded {
                loss: daily_loss,
                limit: self.config.daily_loss_limit,
            });
        }

        // Check weekly loss limit
        if weekly_loss > self.config.weekly_loss_limit {
            self.stats.record_trigger();
            self.stats.record_check(start.elapsed().as_nanos() as u64);
            return Err(RiskError::WeeklyLossLimitExceeded {
                loss: weekly_loss,
                limit: self.config.weekly_loss_limit,
            });
        }

        self.stats.record_check(start.elapsed().as_nanos() as u64);
        Ok(())
    }

    fn reset(&self) {
        self.drawdown_level.store(DrawdownLevel::Normal as u8, Ordering::SeqCst);
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

    fn test_order() -> Order {
        Order {
            symbol: Symbol::new("AAPL"),
            side: OrderSide::Buy,
            quantity: Quantity::from_f64(100.0),
            limit_price: Some(Price::from_f64(150.0)),
            strategy_id: 1,
            timestamp: Timestamp::now(),
        }
    }

    #[test]
    fn test_normal_drawdown() {
        let sentinel = DrawdownSentinel::default();
        sentinel.initialize(100_000.0);

        let order = test_order();
        let portfolio = Portfolio::new(98_000.0); // 2% drawdown

        assert!(sentinel.check(&order, &portfolio).is_ok());
        assert_eq!(sentinel.current_level(), DrawdownLevel::Normal);
    }

    #[test]
    fn test_warning_drawdown() {
        let sentinel = DrawdownSentinel::default();
        sentinel.initialize(100_000.0);

        let order = test_order();
        // 6% drawdown is above warning (5%) but below critical (10%)
        // Warning level should allow trading to continue
        let portfolio = Portfolio::new(94_000.0);

        // Need to set daily/weekly start values to avoid triggering loss limits
        sentinel.reset_daily(94_000.0);
        sentinel.reset_weekly(94_000.0);

        // The check should pass (warning allows trading), just sets level to Warning
        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_ok(), "Warning level should allow trading: {:?}", result);
        assert_eq!(sentinel.current_level(), DrawdownLevel::Warning);
    }

    #[test]
    fn test_critical_drawdown() {
        let sentinel = DrawdownSentinel::default();
        sentinel.initialize(100_000.0);

        let order = test_order();
        let portfolio = Portfolio::new(88_000.0); // 12% drawdown (> 10% critical)

        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_err());
        assert_eq!(sentinel.current_level(), DrawdownLevel::Critical);
    }

    #[test]
    fn test_emergency_drawdown() {
        let sentinel = DrawdownSentinel::default();
        sentinel.initialize(100_000.0);

        let order = test_order();
        let portfolio = Portfolio::new(82_000.0); // 18% drawdown (> 15% emergency)

        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_err());
        assert_eq!(sentinel.current_level(), DrawdownLevel::Emergency);
    }

    #[test]
    fn test_high_water_mark_update() {
        let sentinel = DrawdownSentinel::default();
        sentinel.initialize(100_000.0);

        // Portfolio increases
        sentinel.update_high_water_mark(105_000.0);

        // Now check drawdown from new HWM
        let order = test_order();
        let portfolio = Portfolio::new(100_000.0); // ~4.8% from 105k HWM

        assert!(sentinel.check(&order, &portfolio).is_ok());
    }

    #[test]
    fn test_latency() {
        let sentinel = DrawdownSentinel::default();
        sentinel.initialize(100_000.0);

        let order = test_order();
        let portfolio = Portfolio::new(98_000.0);

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

        assert!(avg_ns < 5000, "Drawdown check too slow: {}ns average", avg_ns);
    }
}
