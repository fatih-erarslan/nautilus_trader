//! Circuit Breaker Sentinel.
//!
//! Implements volatility and loss-based trading halts.
//! Target latency: <10μs

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use crate::core::error::{Result, RiskError};
use crate::core::types::{Order, Portfolio, Timestamp};
use crate::sentinels::base::{Sentinel, SentinelId, SentinelStats, SentinelStatus};

/// Circuit breaker states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation - all trading allowed.
    Closed,
    /// Half-open - limited trading allowed (testing recovery).
    HalfOpen,
    /// Open - trading halted.
    Open,
}

impl CircuitState {
    /// Convert to atomic storage format.
    fn as_u8(self) -> u8 {
        match self {
            Self::Closed => 0,
            Self::HalfOpen => 1,
            Self::Open => 2,
        }
    }

    /// Convert from atomic storage format.
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Closed,
            1 => Self::HalfOpen,
            _ => Self::Open,
        }
    }
}

/// Circuit breaker configuration.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Volatility threshold (realized vol as decimal, e.g., 0.5 = 50% annualized).
    pub volatility_threshold: f64,
    /// Rapid loss threshold - loss in short period.
    pub rapid_loss_threshold: f64,
    /// Rapid loss window in nanoseconds.
    pub rapid_loss_window_ns: u64,
    /// Consecutive losses before trip.
    pub max_consecutive_losses: u32,
    /// Cooldown period in nanoseconds after trip.
    pub cooldown_ns: u64,
    /// Maximum trips before requiring manual reset.
    pub max_trips_before_lockout: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            volatility_threshold: 0.50,           // 50% annualized vol
            rapid_loss_threshold: 0.02,           // 2% in rapid period
            rapid_loss_window_ns: 300_000_000_000, // 5 minutes
            max_consecutive_losses: 5,
            cooldown_ns: 60_000_000_000,          // 1 minute
            max_trips_before_lockout: 3,
        }
    }
}

impl CircuitBreakerConfig {
    /// Sensitive configuration for conservative strategies.
    pub fn sensitive() -> Self {
        Self {
            volatility_threshold: 0.30,
            rapid_loss_threshold: 0.01,
            rapid_loss_window_ns: 180_000_000_000, // 3 minutes
            max_consecutive_losses: 3,
            cooldown_ns: 120_000_000_000,         // 2 minutes
            max_trips_before_lockout: 2,
        }
    }

    /// Tolerant configuration for aggressive strategies.
    pub fn tolerant() -> Self {
        Self {
            volatility_threshold: 0.80,
            rapid_loss_threshold: 0.05,
            rapid_loss_window_ns: 600_000_000_000, // 10 minutes
            max_consecutive_losses: 10,
            cooldown_ns: 30_000_000_000,          // 30 seconds
            max_trips_before_lockout: 5,
        }
    }
}

/// Circuit breaker sentinel.
///
/// Implements exponential backoff circuit breaker pattern.
#[derive(Debug)]
pub struct CircuitBreakerSentinel {
    /// Sentinel ID.
    id: SentinelId,
    /// Configuration.
    config: CircuitBreakerConfig,
    /// Current status.
    status: AtomicU8,
    /// Current circuit state.
    circuit_state: AtomicU8,
    /// Time of last trip (nanoseconds since epoch).
    last_trip_time: AtomicU64,
    /// Trip count since last reset.
    trip_count: AtomicU64,
    /// Consecutive losses counter.
    consecutive_losses: AtomicU64,
    /// Current realized volatility (scaled by 1M).
    current_vol_scaled: AtomicU64,
    /// Recent high value for rapid loss calc (scaled).
    recent_high_scaled: AtomicU64,
    /// Recent high timestamp.
    recent_high_time: AtomicU64,
    /// Statistics.
    stats: SentinelStats,
}

impl CircuitBreakerSentinel {
    const SCALE: f64 = 1_000_000.0;

    /// Create new circuit breaker sentinel.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            id: SentinelId::new("circuit_breaker"),
            config,
            status: AtomicU8::new(SentinelStatus::Active as u8),
            circuit_state: AtomicU8::new(CircuitState::Closed.as_u8()),
            last_trip_time: AtomicU64::new(0),
            trip_count: AtomicU64::new(0),
            consecutive_losses: AtomicU64::new(0),
            current_vol_scaled: AtomicU64::new(0),
            recent_high_scaled: AtomicU64::new(0),
            recent_high_time: AtomicU64::new(0),
            stats: SentinelStats::new(),
        }
    }

    /// Update realized volatility measurement.
    pub fn update_volatility(&self, realized_vol: f64) {
        let scaled = (realized_vol * Self::SCALE) as u64;
        self.current_vol_scaled.store(scaled, Ordering::Relaxed);
    }

    /// Record a trade outcome.
    pub fn record_trade(&self, profit: f64, current_value: f64) {
        let now = Timestamp::now().as_nanos();

        if profit < 0.0 {
            self.consecutive_losses.fetch_add(1, Ordering::Relaxed);
        } else {
            self.consecutive_losses.store(0, Ordering::Relaxed);
        }

        // Update recent high for rapid loss tracking
        let current_high = self.recent_high_scaled.load(Ordering::Relaxed);
        let current_scaled = (current_value * Self::SCALE) as u64;
        let high_time = self.recent_high_time.load(Ordering::Relaxed);

        // Reset high if outside window or if new high
        if now.saturating_sub(high_time) > self.config.rapid_loss_window_ns
            || current_scaled > current_high
        {
            self.recent_high_scaled.store(current_scaled, Ordering::Relaxed);
            self.recent_high_time.store(now, Ordering::Relaxed);
        }
    }

    /// Get current circuit state.
    pub fn state(&self) -> CircuitState {
        CircuitState::from_u8(self.circuit_state.load(Ordering::Relaxed))
    }

    /// Trip the circuit breaker.
    fn trip(&self, reason: &'static str) -> RiskError {
        let now = Timestamp::now().as_nanos();

        self.circuit_state.store(CircuitState::Open.as_u8(), Ordering::SeqCst);
        self.last_trip_time.store(now, Ordering::SeqCst);
        self.trip_count.fetch_add(1, Ordering::Relaxed);
        self.status.store(SentinelStatus::Triggered as u8, Ordering::SeqCst);
        self.stats.record_trigger();

        RiskError::CircuitBreakerTripped { reason }
    }

    /// Check if cooldown has elapsed.
    #[inline]
    fn check_cooldown(&self) -> bool {
        let last_trip = self.last_trip_time.load(Ordering::Relaxed);
        if last_trip == 0 {
            return true;
        }

        let now = Timestamp::now().as_nanos();
        now.saturating_sub(last_trip) >= self.config.cooldown_ns
    }

    /// Check volatility condition.
    #[inline]
    fn check_volatility(&self) -> Result<()> {
        let vol = self.current_vol_scaled.load(Ordering::Relaxed) as f64 / Self::SCALE;

        if vol > self.config.volatility_threshold {
            return Err(self.trip("Volatility threshold exceeded"));
        }
        Ok(())
    }

    /// Check consecutive losses condition.
    #[inline]
    fn check_consecutive_losses(&self) -> Result<()> {
        let losses = self.consecutive_losses.load(Ordering::Relaxed) as u32;

        if losses >= self.config.max_consecutive_losses {
            return Err(self.trip("Maximum consecutive losses exceeded"));
        }
        Ok(())
    }

    /// Check rapid loss condition.
    #[inline]
    fn check_rapid_loss(&self, current_value: f64) -> Result<()> {
        let recent_high = self.recent_high_scaled.load(Ordering::Relaxed) as f64 / Self::SCALE;
        let high_time = self.recent_high_time.load(Ordering::Relaxed);
        let now = Timestamp::now().as_nanos();

        // Only check if within window
        if now.saturating_sub(high_time) <= self.config.rapid_loss_window_ns && recent_high > 0.0 {
            let loss_pct = (recent_high - current_value) / recent_high;

            if loss_pct > self.config.rapid_loss_threshold {
                return Err(self.trip("Rapid loss threshold exceeded"));
            }
        }
        Ok(())
    }

    /// Attempt to close circuit after cooldown.
    pub fn try_close(&self) -> bool {
        if self.state() != CircuitState::Open {
            return true;
        }

        if !self.check_cooldown() {
            return false;
        }

        // Move to half-open
        self.circuit_state.store(CircuitState::HalfOpen.as_u8(), Ordering::SeqCst);
        true
    }

    /// Confirm circuit close after successful half-open test.
    pub fn confirm_close(&self) {
        if self.state() == CircuitState::HalfOpen {
            self.circuit_state.store(CircuitState::Closed.as_u8(), Ordering::SeqCst);
            self.status.store(SentinelStatus::Active as u8, Ordering::SeqCst);
        }
    }

    /// Get trip count.
    pub fn trip_count(&self) -> u64 {
        self.trip_count.load(Ordering::Relaxed)
    }
}

impl Default for CircuitBreakerSentinel {
    fn default() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }
}

impl Sentinel for CircuitBreakerSentinel {
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

    /// Check circuit breaker conditions.
    ///
    /// Target: <10μs
    #[inline]
    fn check(&self, _order: &Order, portfolio: &Portfolio) -> Result<()> {
        let start = std::time::Instant::now();

        // Check circuit state first
        match self.state() {
            CircuitState::Open => {
                // Check if cooldown elapsed
                if self.check_cooldown() {
                    // Transition to half-open
                    self.circuit_state.store(CircuitState::HalfOpen.as_u8(), Ordering::SeqCst);
                } else {
                    self.stats.record_check(start.elapsed().as_nanos() as u64);
                    return Err(RiskError::CircuitBreakerOpen {
                        remaining_cooldown_ns: self.config.cooldown_ns
                            .saturating_sub(
                                Timestamp::now().as_nanos()
                                    .saturating_sub(self.last_trip_time.load(Ordering::Relaxed))
                            ),
                    });
                }
            }
            CircuitState::HalfOpen => {
                // Allow limited trading, will trip again if conditions worsen
            }
            CircuitState::Closed => {
                // Normal operation
            }
        }

        // Check trip count for lockout
        if self.trip_count.load(Ordering::Relaxed) >= self.config.max_trips_before_lockout as u64 {
            self.stats.record_check(start.elapsed().as_nanos() as u64);
            return Err(RiskError::CircuitBreakerLockout {
                trips: self.trip_count.load(Ordering::Relaxed),
                max: self.config.max_trips_before_lockout,
            });
        }

        // Check all conditions
        self.check_volatility()?;
        self.check_consecutive_losses()?;
        self.check_rapid_loss(portfolio.total_value)?;

        // If we get here in half-open, transition to closed
        if self.state() == CircuitState::HalfOpen {
            self.confirm_close();
        }

        self.stats.record_check(start.elapsed().as_nanos() as u64);
        Ok(())
    }

    fn reset(&self) {
        self.circuit_state.store(CircuitState::Closed.as_u8(), Ordering::SeqCst);
        self.last_trip_time.store(0, Ordering::SeqCst);
        self.trip_count.store(0, Ordering::SeqCst);
        self.consecutive_losses.store(0, Ordering::SeqCst);
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
    use crate::core::types::{OrderSide, Price, Quantity, Symbol};

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
    fn test_circuit_closed() {
        let sentinel = CircuitBreakerSentinel::default();
        let order = test_order();
        let portfolio = Portfolio::new(100_000.0);

        assert_eq!(sentinel.state(), CircuitState::Closed);
        assert!(sentinel.check(&order, &portfolio).is_ok());
    }

    #[test]
    fn test_volatility_trip() {
        let sentinel = CircuitBreakerSentinel::default();
        sentinel.update_volatility(0.60); // Above 0.50 threshold

        let order = test_order();
        let portfolio = Portfolio::new(100_000.0);

        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_err());
        assert_eq!(sentinel.state(), CircuitState::Open);
    }

    #[test]
    fn test_consecutive_losses_trip() {
        let sentinel = CircuitBreakerSentinel::default();

        // Record 5 consecutive losses
        for _ in 0..5 {
            sentinel.record_trade(-100.0, 100_000.0);
        }

        let order = test_order();
        let portfolio = Portfolio::new(100_000.0);

        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_err());
    }

    #[test]
    fn test_circuit_recovery() {
        let mut config = CircuitBreakerConfig::default();
        config.cooldown_ns = 1_000; // 1 microsecond for testing

        let sentinel = CircuitBreakerSentinel::new(config);
        sentinel.update_volatility(0.60); // Trip it

        let order = test_order();
        let portfolio = Portfolio::new(100_000.0);

        // First check trips
        let _ = sentinel.check(&order, &portfolio);
        assert_eq!(sentinel.state(), CircuitState::Open);

        // Reset volatility
        sentinel.update_volatility(0.30);

        // Wait for cooldown
        std::thread::sleep(std::time::Duration::from_micros(10));

        // Should recover
        assert!(sentinel.check(&order, &portfolio).is_ok());
        assert_eq!(sentinel.state(), CircuitState::Closed);
    }

    #[test]
    fn test_latency() {
        let sentinel = CircuitBreakerSentinel::default();
        let order = test_order();
        let portfolio = Portfolio::new(100_000.0);

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

        assert!(
            avg_ns < 10000,
            "Circuit breaker check too slow: {}ns average",
            avg_ns
        );
    }
}
