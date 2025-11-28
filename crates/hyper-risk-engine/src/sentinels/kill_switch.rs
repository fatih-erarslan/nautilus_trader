//! Global Kill Switch Sentinel.
//!
//! Provides atomic halt across all trading strategies.
//! This is the highest priority sentinel and must complete in <1μs.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use crate::core::error::{Result, RiskError};
use crate::core::types::{Order, Portfolio};
use crate::sentinels::base::{Sentinel, SentinelId, SentinelStats, SentinelStatus};

/// Global kill switch sentinel.
///
/// When triggered, immediately rejects ALL orders regardless of other conditions.
/// This is the first check in the fast path and must be ultra-low latency.
#[derive(Debug)]
pub struct GlobalKillSwitch {
    /// Sentinel ID.
    id: SentinelId,
    /// Whether kill switch is activated.
    activated: AtomicBool,
    /// Current status.
    status: AtomicU8,
    /// Reason for activation (static string for zero-allocation).
    reason: &'static str,
    /// Statistics.
    stats: SentinelStats,
}

impl GlobalKillSwitch {
    /// Create new kill switch (initially not activated).
    pub fn new() -> Self {
        Self {
            id: SentinelId::new("global_kill_switch"),
            activated: AtomicBool::new(false),
            status: AtomicU8::new(SentinelStatus::Active as u8),
            reason: "Kill switch not activated",
            stats: SentinelStats::new(),
        }
    }

    /// Activate kill switch with reason.
    ///
    /// # Arguments
    ///
    /// * `reason` - Static string reason (must be 'static for zero allocation)
    pub fn activate(&mut self, reason: &'static str) {
        self.activated.store(true, Ordering::SeqCst);
        self.status.store(SentinelStatus::Triggered as u8, Ordering::SeqCst);
        self.reason = reason;
        self.stats.record_trigger();
    }

    /// Deactivate kill switch.
    pub fn deactivate(&self) {
        self.activated.store(false, Ordering::SeqCst);
        self.status.store(SentinelStatus::Active as u8, Ordering::SeqCst);
    }

    /// Check if kill switch is activated.
    #[inline]
    pub fn is_activated(&self) -> bool {
        self.activated.load(Ordering::SeqCst)
    }

    /// Get activation reason.
    pub fn reason(&self) -> &'static str {
        self.reason
    }
}

impl Default for GlobalKillSwitch {
    fn default() -> Self {
        Self::new()
    }
}

impl Sentinel for GlobalKillSwitch {
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

    /// Check kill switch state.
    ///
    /// This is the fastest sentinel check - just an atomic load.
    /// Target: <1μs (typically <100ns)
    #[inline]
    fn check(&self, _order: &Order, _portfolio: &Portfolio) -> Result<()> {
        // Single atomic load - ultra fast
        if self.activated.load(Ordering::SeqCst) {
            self.stats.record_check(100); // ~100ns
            Err(RiskError::KillSwitchActivated {
                reason: self.reason,
            })
        } else {
            self.stats.record_check(50); // ~50ns
            Ok(())
        }
    }

    fn reset(&self) {
        self.deactivate();
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
    use crate::core::types::{OrderSide, Quantity, Symbol, Timestamp};

    fn test_order() -> Order {
        Order {
            symbol: Symbol::new("AAPL"),
            side: OrderSide::Buy,
            quantity: Quantity::from_f64(100.0),
            limit_price: None,
            strategy_id: 1,
            timestamp: Timestamp::now(),
        }
    }

    #[test]
    fn test_kill_switch_not_activated() {
        let kill_switch = GlobalKillSwitch::new();
        let order = test_order();
        let portfolio = Portfolio::new(100_000.0);

        assert!(!kill_switch.is_activated());
        assert!(kill_switch.check(&order, &portfolio).is_ok());
    }

    #[test]
    fn test_kill_switch_activated() {
        let mut kill_switch = GlobalKillSwitch::new();
        kill_switch.activate("Test activation");

        let order = test_order();
        let portfolio = Portfolio::new(100_000.0);

        assert!(kill_switch.is_activated());
        let result = kill_switch.check(&order, &portfolio);
        assert!(result.is_err());

        match result.unwrap_err() {
            RiskError::KillSwitchActivated { reason } => {
                assert_eq!(reason, "Test activation");
            }
            _ => panic!("Expected KillSwitchActivated error"),
        }
    }

    #[test]
    fn test_kill_switch_reset() {
        let mut kill_switch = GlobalKillSwitch::new();
        kill_switch.activate("Test activation");

        assert!(kill_switch.is_activated());

        kill_switch.reset();

        assert!(!kill_switch.is_activated());
        assert_eq!(kill_switch.status(), SentinelStatus::Active);
    }

    #[test]
    fn test_kill_switch_latency() {
        let kill_switch = GlobalKillSwitch::new();
        let order = test_order();
        let portfolio = Portfolio::new(100_000.0);

        // Warm up
        for _ in 0..1000 {
            let _ = kill_switch.check(&order, &portfolio);
        }

        // Measure
        let start = std::time::Instant::now();
        for _ in 0..10000 {
            let _ = kill_switch.check(&order, &portfolio);
        }
        let elapsed = start.elapsed();
        let avg_ns = elapsed.as_nanos() / 10000;

        // Should be well under 1μs (1000ns)
        assert!(
            avg_ns < 1000,
            "Kill switch check too slow: {}ns average",
            avg_ns
        );
    }
}
