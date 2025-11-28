//! Base traits and types for sentinels.

use std::fmt::Debug;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::core::error::Result;
use crate::core::types::{Order, Portfolio};

/// Unique identifier for a sentinel.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SentinelId(String);

impl SentinelId {
    /// Create new sentinel ID.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get ID as string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SentinelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Sentinel operational status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SentinelStatus {
    /// Sentinel is active and monitoring.
    Active,
    /// Sentinel is disabled (not checking).
    Disabled,
    /// Sentinel has been triggered.
    Triggered,
    /// Sentinel encountered an error.
    Error,
}

/// Base configuration for sentinels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentinelConfig {
    /// Sentinel name.
    pub name: String,
    /// Whether sentinel is enabled.
    pub enabled: bool,
    /// Priority (lower = higher priority, checked first).
    pub priority: u8,
    /// Log all checks (for debugging).
    pub verbose: bool,
}

impl Default for SentinelConfig {
    fn default() -> Self {
        Self {
            name: "unnamed".to_string(),
            enabled: true,
            priority: 100,
            verbose: false,
        }
    }
}

/// Core trait for all sentinels.
///
/// Sentinels must be:
/// - **Fast**: Check must complete in <20μs
/// - **Deterministic**: Same input produces same output
/// - **Lock-free**: No blocking operations in check path
/// - **Allocation-free**: No heap allocation during check
pub trait Sentinel: Send + Sync + Debug {
    /// Get sentinel identifier.
    fn id(&self) -> SentinelId;

    /// Get current status.
    fn status(&self) -> SentinelStatus;

    /// Check order against sentinel rules.
    ///
    /// # Performance Requirements
    ///
    /// - Must complete in <20μs
    /// - Must not allocate
    /// - Must not block
    ///
    /// # Returns
    ///
    /// - `Ok(())` if check passes
    /// - `Err(RiskError)` if check fails
    fn check(&self, order: &Order, portfolio: &Portfolio) -> Result<()>;

    /// Reset sentinel state (after trigger or recovery).
    fn reset(&self);

    /// Enable sentinel.
    fn enable(&self);

    /// Disable sentinel.
    fn disable(&self);

    /// Get check count.
    fn check_count(&self) -> u64;

    /// Get trigger count.
    fn trigger_count(&self) -> u64;

    /// Get average check latency in nanoseconds.
    fn avg_latency_ns(&self) -> u64;
}

/// Statistics tracking for sentinels.
#[derive(Debug, Default)]
pub struct SentinelStats {
    /// Total checks performed.
    pub checks: AtomicU64,
    /// Total triggers.
    pub triggers: AtomicU64,
    /// Total latency (for average calculation).
    pub total_latency_ns: AtomicU64,
    /// Maximum latency observed.
    pub max_latency_ns: AtomicU64,
}

impl SentinelStats {
    /// Create new stats tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a check.
    #[inline]
    pub fn record_check(&self, latency_ns: u64) {
        self.checks.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ns.fetch_add(latency_ns, Ordering::Relaxed);

        // Update max
        let prev_max = self.max_latency_ns.load(Ordering::Relaxed);
        if latency_ns > prev_max {
            self.max_latency_ns.store(latency_ns, Ordering::Relaxed);
        }
    }

    /// Record a trigger.
    #[inline]
    pub fn record_trigger(&self) {
        self.triggers.fetch_add(1, Ordering::Relaxed);
    }

    /// Get average latency.
    pub fn avg_latency_ns(&self) -> u64 {
        let checks = self.checks.load(Ordering::Relaxed);
        if checks == 0 {
            return 0;
        }
        self.total_latency_ns.load(Ordering::Relaxed) / checks
    }

    /// Reset statistics.
    pub fn reset(&self) {
        self.checks.store(0, Ordering::Relaxed);
        self.triggers.store(0, Ordering::Relaxed);
        self.total_latency_ns.store(0, Ordering::Relaxed);
        self.max_latency_ns.store(0, Ordering::Relaxed);
    }
}

/// Macro for implementing common sentinel boilerplate.
#[macro_export]
macro_rules! impl_sentinel_common {
    ($sentinel:ty) => {
        impl $sentinel {
            /// Get check count.
            pub fn check_count(&self) -> u64 {
                self.stats.checks.load(std::sync::atomic::Ordering::Relaxed)
            }

            /// Get trigger count.
            pub fn trigger_count(&self) -> u64 {
                self.stats.triggers.load(std::sync::atomic::Ordering::Relaxed)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentinel_id() {
        let id = SentinelId::new("test_sentinel");
        assert_eq!(id.as_str(), "test_sentinel");
    }

    #[test]
    fn test_sentinel_stats() {
        let stats = SentinelStats::new();

        stats.record_check(100);
        stats.record_check(200);
        stats.record_check(150);

        assert_eq!(stats.checks.load(Ordering::Relaxed), 3);
        assert_eq!(stats.avg_latency_ns(), 150);
        assert_eq!(stats.max_latency_ns.load(Ordering::Relaxed), 200);
    }
}
