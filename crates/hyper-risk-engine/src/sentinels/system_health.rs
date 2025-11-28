//! System health sentinel for monitoring infrastructure status.
//!
//! Operates in the fast path (<20μs) to validate system health
//! before allowing order processing.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use crate::core::types::{Order, Portfolio, Timestamp};
use crate::core::error::{Result, RiskError};

use super::base::{Sentinel, SentinelConfig, SentinelId, SentinelStats, SentinelStatus};

/// Configuration for the system health sentinel.
#[derive(Debug, Clone)]
pub struct SystemHealthConfig {
    /// Base sentinel configuration.
    pub base: SentinelConfig,
    /// Maximum acceptable latency in microseconds.
    pub max_latency_us: u64,
    /// Maximum memory usage percentage.
    pub max_memory_pct: f64,
    /// Maximum error rate threshold.
    pub max_error_rate: f64,
    /// Health check interval in milliseconds.
    pub check_interval_ms: u64,
    /// Minimum uptime required in seconds.
    pub min_uptime_secs: u64,
}

impl Default for SystemHealthConfig {
    fn default() -> Self {
        Self {
            base: SentinelConfig {
                name: "system_health_sentinel".to_string(),
                enabled: true,
                priority: 1,
                verbose: false,
            },
            max_latency_us: 1000, // 1ms
            max_memory_pct: 90.0,
            max_error_rate: 0.01, // 1%
            check_interval_ms: 1000,
            min_uptime_secs: 0,
        }
    }
}

/// System health metrics.
#[derive(Debug, Clone)]
pub struct HealthMetrics {
    /// Current latency in microseconds.
    pub current_latency_us: u64,
    /// Memory usage percentage.
    pub memory_pct: f64,
    /// Current error rate.
    pub error_rate: f64,
    /// System uptime in seconds.
    pub uptime_secs: u64,
    /// Number of active connections.
    pub active_connections: u32,
    /// Last update timestamp.
    pub last_update: Timestamp,
}

impl Default for HealthMetrics {
    fn default() -> Self {
        Self {
            current_latency_us: 0,
            memory_pct: 0.0,
            error_rate: 0.0,
            uptime_secs: 0,
            active_connections: 0,
            last_update: Timestamp::now(),
        }
    }
}

/// Health status summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthLevel {
    /// All systems healthy.
    Healthy,
    /// Some metrics degraded but acceptable.
    Degraded,
    /// Critical issues detected.
    Critical,
    /// System unhealthy, trading should stop.
    Unhealthy,
}

/// System health sentinel.
#[derive(Debug)]
pub struct SystemHealthSentinel {
    id: SentinelId,
    config: SystemHealthConfig,
    status: AtomicU8,
    stats: SentinelStats,
    /// Current health metrics.
    metrics: RwLock<HealthMetrics>,
    /// Total requests processed.
    total_requests: AtomicU64,
    /// Total errors encountered.
    total_errors: AtomicU64,
    /// Start time for uptime calculation.
    start_time: Timestamp,
}

impl SystemHealthSentinel {
    /// Create a new system health sentinel.
    pub fn new(config: SystemHealthConfig) -> Self {
        Self {
            id: SentinelId::new(&config.base.name),
            config,
            status: AtomicU8::new(SentinelStatus::Active as u8),
            stats: SentinelStats::new(),
            metrics: RwLock::new(HealthMetrics::default()),
            total_requests: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            start_time: Timestamp::now(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SystemHealthConfig::default())
    }

    /// Update health metrics.
    pub fn update_metrics(&self, metrics: HealthMetrics) {
        *self.metrics.write() = metrics;
    }

    /// Record a request.
    pub fn record_request(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an error.
    pub fn record_error(&self) {
        self.total_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current health metrics.
    pub fn get_metrics(&self) -> HealthMetrics {
        self.metrics.read().clone()
    }

    /// Get current health level.
    pub fn get_health_level(&self) -> HealthLevel {
        let metrics = self.metrics.read();

        // Check for critical conditions
        if metrics.memory_pct > self.config.max_memory_pct {
            return HealthLevel::Unhealthy;
        }
        if metrics.error_rate > self.config.max_error_rate * 2.0 {
            return HealthLevel::Unhealthy;
        }

        // Check for degraded conditions
        if metrics.current_latency_us > self.config.max_latency_us {
            return HealthLevel::Degraded;
        }
        if metrics.error_rate > self.config.max_error_rate * 0.5 {
            return HealthLevel::Degraded;
        }
        if metrics.memory_pct > self.config.max_memory_pct * 0.8 {
            return HealthLevel::Degraded;
        }

        HealthLevel::Healthy
    }

    /// Calculate current error rate.
    pub fn calculate_error_rate(&self) -> f64 {
        let requests = self.total_requests.load(Ordering::Relaxed);
        let errors = self.total_errors.load(Ordering::Relaxed);

        if requests == 0 {
            return 0.0;
        }
        errors as f64 / requests as f64
    }

    /// Get system uptime in seconds.
    pub fn uptime_secs(&self) -> u64 {
        let now = Timestamp::now().as_nanos();
        let start = self.start_time.as_nanos();
        (now.saturating_sub(start)) / 1_000_000_000
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

impl Sentinel for SystemHealthSentinel {
    fn id(&self) -> SentinelId {
        self.id.clone()
    }

    fn status(&self) -> SentinelStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn check(&self, _order: &Order, _portfolio: &Portfolio) -> Result<()> {
        let start = Instant::now();

        // Check if disabled
        if self.status() == SentinelStatus::Disabled {
            return Ok(());
        }

        // Record this request
        self.record_request();

        // Check uptime requirement
        let uptime = self.uptime_secs();
        if uptime < self.config.min_uptime_secs {
            self.stats.record_trigger();
            self.status.store(SentinelStatus::Triggered as u8, Ordering::SeqCst);
            let latency_ns = start.elapsed().as_nanos() as u64;
            self.stats.record_check(latency_ns);
            return Err(RiskError::InternalError(format!(
                "System uptime {} secs below minimum {} secs",
                uptime, self.config.min_uptime_secs
            )));
        }

        // Check health level
        let health_level = self.get_health_level();
        if health_level == HealthLevel::Unhealthy {
            self.stats.record_trigger();
            self.status.store(SentinelStatus::Triggered as u8, Ordering::SeqCst);
            let latency_ns = start.elapsed().as_nanos() as u64;
            self.stats.record_check(latency_ns);
            return Err(RiskError::InternalError(
                "System health is unhealthy, trading suspended".to_string(),
            ));
        }

        // Check error rate
        let error_rate = self.calculate_error_rate();
        if error_rate > self.config.max_error_rate {
            self.stats.record_trigger();
            self.status.store(SentinelStatus::Triggered as u8, Ordering::SeqCst);
            let latency_ns = start.elapsed().as_nanos() as u64;
            self.stats.record_check(latency_ns);
            return Err(RiskError::InternalError(format!(
                "Error rate {:.2}% exceeds maximum {:.2}%",
                error_rate * 100.0,
                self.config.max_error_rate * 100.0
            )));
        }

        // Check metrics
        let metrics = self.metrics.read();
        let current_latency_us = metrics.current_latency_us;
        let memory_pct = metrics.memory_pct;
        drop(metrics);

        // Check latency
        if current_latency_us > self.config.max_latency_us {
            self.stats.record_trigger();
            self.status.store(SentinelStatus::Triggered as u8, Ordering::SeqCst);
            let latency_ns = start.elapsed().as_nanos() as u64;
            self.stats.record_check(latency_ns);
            return Err(RiskError::InternalError(format!(
                "System latency {} μs exceeds maximum {} μs",
                current_latency_us, self.config.max_latency_us
            )));
        }

        // Check memory
        if memory_pct > self.config.max_memory_pct {
            self.stats.record_trigger();
            self.status.store(SentinelStatus::Triggered as u8, Ordering::SeqCst);
            let latency_ns = start.elapsed().as_nanos() as u64;
            self.stats.record_check(latency_ns);
            return Err(RiskError::InternalError(format!(
                "Memory usage {:.1}% exceeds maximum {:.1}%",
                memory_pct, self.config.max_memory_pct
            )));
        }

        let latency_ns = start.elapsed().as_nanos() as u64;
        self.stats.record_check(latency_ns);
        Ok(())
    }

    fn reset(&self) {
        self.stats.reset();
        self.status.store(SentinelStatus::Active as u8, Ordering::SeqCst);
        self.total_requests.store(0, Ordering::Relaxed);
        self.total_errors.store(0, Ordering::Relaxed);
        *self.metrics.write() = HealthMetrics::default();
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

    fn create_test_order() -> Order {
        Order {
            symbol: Symbol::new("AAPL"),
            side: OrderSide::Buy,
            quantity: Quantity::from_f64(100.0),
            limit_price: Some(Price::from_f64(100.0)),
            strategy_id: 1,
            timestamp: Timestamp::now(),
        }
    }

    #[test]
    fn test_system_health_sentinel_creation() {
        let sentinel = SystemHealthSentinel::with_defaults();
        assert_eq!(sentinel.status(), SentinelStatus::Active);
        assert_eq!(sentinel.check_count(), 0);
    }

    #[test]
    fn test_health_level_healthy() {
        let sentinel = SystemHealthSentinel::with_defaults();

        sentinel.update_metrics(HealthMetrics {
            current_latency_us: 100,
            memory_pct: 50.0,
            error_rate: 0.0,
            uptime_secs: 3600,
            active_connections: 10,
            last_update: Timestamp::now(),
        });

        assert_eq!(sentinel.get_health_level(), HealthLevel::Healthy);
    }

    #[test]
    fn test_health_level_degraded() {
        let mut config = SystemHealthConfig::default();
        config.max_latency_us = 500;

        let sentinel = SystemHealthSentinel::new(config);

        sentinel.update_metrics(HealthMetrics {
            current_latency_us: 1000, // Above threshold
            memory_pct: 50.0,
            error_rate: 0.0,
            uptime_secs: 3600,
            active_connections: 10,
            last_update: Timestamp::now(),
        });

        assert_eq!(sentinel.get_health_level(), HealthLevel::Degraded);
    }

    #[test]
    fn test_health_level_unhealthy() {
        let mut config = SystemHealthConfig::default();
        config.max_memory_pct = 80.0;

        let sentinel = SystemHealthSentinel::new(config);

        sentinel.update_metrics(HealthMetrics {
            current_latency_us: 100,
            memory_pct: 95.0, // Above threshold
            error_rate: 0.0,
            uptime_secs: 3600,
            active_connections: 10,
            last_update: Timestamp::now(),
        });

        assert_eq!(sentinel.get_health_level(), HealthLevel::Unhealthy);
    }

    #[test]
    fn test_check_passes_when_healthy() {
        let sentinel = SystemHealthSentinel::with_defaults();

        sentinel.update_metrics(HealthMetrics {
            current_latency_us: 100,
            memory_pct: 50.0,
            error_rate: 0.0,
            uptime_secs: 3600,
            active_connections: 10,
            last_update: Timestamp::now(),
        });

        let order = create_test_order();
        let portfolio = Portfolio::default();

        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_fails_when_unhealthy() {
        let mut config = SystemHealthConfig::default();
        config.max_memory_pct = 80.0;

        let sentinel = SystemHealthSentinel::new(config);

        sentinel.update_metrics(HealthMetrics {
            current_latency_us: 100,
            memory_pct: 95.0,
            error_rate: 0.0,
            uptime_secs: 3600,
            active_connections: 10,
            last_update: Timestamp::now(),
        });

        let order = create_test_order();
        let portfolio = Portfolio::default();

        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_rate_calculation() {
        let sentinel = SystemHealthSentinel::with_defaults();

        // Record 100 requests
        for _ in 0..100 {
            sentinel.record_request();
        }

        // Record 5 errors
        for _ in 0..5 {
            sentinel.record_error();
        }

        let error_rate = sentinel.calculate_error_rate();
        assert!((error_rate - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_sentinel_lifecycle() {
        let sentinel = SystemHealthSentinel::with_defaults();

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
