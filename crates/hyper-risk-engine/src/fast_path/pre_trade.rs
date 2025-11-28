//! Pre-trade risk check optimized for minimal latency.
//!
//! Performs all necessary checks before order submission in <50μs.

use std::time::Instant;

use crate::core::types::{Order, Portfolio, RiskDecision, RiskLevel, Timestamp};
use super::limit_checker::{LimitChecker, LimitConfig};
use super::anomaly_detector::{FastAnomalyDetector, AnomalyConfig};

/// Pre-trade check configuration.
#[derive(Debug, Clone)]
pub struct PreTradeConfig {
    /// Limit check configuration.
    pub limits: LimitConfig,
    /// Anomaly detection configuration.
    pub anomaly: AnomalyConfig,
    /// Enable strict mode (reject on any warning).
    pub strict_mode: bool,
    /// Target latency in nanoseconds.
    pub target_latency_ns: u64,
}

impl Default for PreTradeConfig {
    fn default() -> Self {
        Self {
            limits: LimitConfig::default(),
            anomaly: AnomalyConfig::default(),
            strict_mode: false,
            target_latency_ns: 50_000, // 50μs
        }
    }
}

/// Pre-trade check result.
#[derive(Debug, Clone)]
pub struct PreTradeResult {
    /// Is order allowed?
    pub allowed: bool,
    /// Risk level.
    pub risk_level: RiskLevel,
    /// Rejection reason (if not allowed).
    pub rejection_reason: Option<String>,
    /// Suggested size adjustment (1.0 = no change).
    pub size_adjustment: f64,
    /// Check latency in nanoseconds.
    pub latency_ns: u64,
    /// Individual check timings.
    pub check_timings: CheckTimings,
}

/// Timings for individual checks.
#[derive(Debug, Clone, Default)]
pub struct CheckTimings {
    /// Limit check time.
    pub limit_check_ns: u64,
    /// Anomaly check time.
    pub anomaly_check_ns: u64,
    /// Portfolio lookup time.
    pub portfolio_lookup_ns: u64,
}

/// Optimized pre-trade checker.
#[derive(Debug)]
pub struct PreTradeChecker {
    /// Configuration.
    config: PreTradeConfig,
    /// Limit checker.
    limit_checker: LimitChecker,
    /// Anomaly detector.
    anomaly_detector: FastAnomalyDetector,
    /// Check counter.
    check_count: std::sync::atomic::AtomicU64,
    /// Slow check counter (exceeded latency target).
    slow_count: std::sync::atomic::AtomicU64,
}

impl PreTradeChecker {
    /// Create new pre-trade checker.
    pub fn new(config: PreTradeConfig) -> Self {
        Self {
            limit_checker: LimitChecker::new(config.limits.clone()),
            anomaly_detector: FastAnomalyDetector::new(config.anomaly.clone()),
            config,
            check_count: std::sync::atomic::AtomicU64::new(0),
            slow_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Perform pre-trade check.
    ///
    /// # Performance
    /// Target: <50μs
    /// - Limit checks: ~10μs
    /// - Anomaly detection: ~15μs
    /// - Portfolio lookup: ~5μs
    /// - Overhead: ~20μs
    #[inline]
    pub fn check(&self, order: &Order, portfolio: &Portfolio) -> PreTradeResult {
        let start = Instant::now();
        let mut timings = CheckTimings::default();

        // 1. Limit checks (fastest first)
        let limit_start = Instant::now();
        let limit_result = self.limit_checker.check(order, portfolio);
        timings.limit_check_ns = limit_start.elapsed().as_nanos() as u64;

        if let Some(violation) = limit_result {
            let latency = start.elapsed().as_nanos() as u64;
            self.record_check(latency);
            return PreTradeResult {
                allowed: false,
                risk_level: violation.severity,
                rejection_reason: Some(violation.message),
                size_adjustment: 0.0,
                latency_ns: latency,
                check_timings: timings,
            };
        }

        // 2. Anomaly detection
        let anomaly_start = Instant::now();
        let anomaly_score = self.anomaly_detector.score(order, portfolio);
        timings.anomaly_check_ns = anomaly_start.elapsed().as_nanos() as u64;

        // Check if anomalous
        if anomaly_score.is_anomaly {
            let latency = start.elapsed().as_nanos() as u64;
            self.record_check(latency);

            if self.config.strict_mode || anomaly_score.score > 0.9 {
                return PreTradeResult {
                    allowed: false,
                    risk_level: RiskLevel::High,
                    rejection_reason: Some(format!(
                        "Anomaly detected: score {:.2}",
                        anomaly_score.score
                    )),
                    size_adjustment: 0.0,
                    latency_ns: latency,
                    check_timings: timings,
                };
            }
        }

        // 3. Calculate size adjustment based on risk
        let size_adjustment = self.calculate_size_adjustment(&anomaly_score, portfolio);

        let latency = start.elapsed().as_nanos() as u64;
        self.record_check(latency);

        PreTradeResult {
            allowed: true,
            risk_level: if anomaly_score.score > 0.5 {
                RiskLevel::Elevated
            } else {
                RiskLevel::Normal
            },
            rejection_reason: None,
            size_adjustment,
            latency_ns: latency,
            check_timings: timings,
        }
    }

    /// Calculate position size adjustment based on risk.
    #[inline]
    fn calculate_size_adjustment(
        &self,
        anomaly_score: &super::anomaly_detector::AnomalyScore,
        portfolio: &Portfolio,
    ) -> f64 {
        let mut adjustment = 1.0;

        // Reduce size if anomalous
        if anomaly_score.score > 0.3 {
            adjustment *= 1.0 - anomaly_score.score * 0.5;
        }

        // Reduce size if in drawdown
        let drawdown = portfolio.drawdown_pct();
        if drawdown > 5.0 {
            adjustment *= 1.0 - (drawdown - 5.0) / 100.0;
        }

        adjustment.max(0.1) // Minimum 10%
    }

    /// Record check and update statistics.
    #[inline]
    fn record_check(&self, latency_ns: u64) {
        use std::sync::atomic::Ordering;
        self.check_count.fetch_add(1, Ordering::Relaxed);

        if latency_ns > self.config.target_latency_ns {
            self.slow_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get check count.
    pub fn check_count(&self) -> u64 {
        self.check_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get slow check count.
    pub fn slow_count(&self) -> u64 {
        self.slow_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get slow check percentage.
    pub fn slow_percentage(&self) -> f64 {
        let total = self.check_count();
        if total == 0 {
            return 0.0;
        }
        self.slow_count() as f64 / total as f64 * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{Symbol, OrderSide, Quantity};

    fn create_test_order() -> Order {
        Order {
            symbol: Symbol::new("TEST"),
            side: OrderSide::Buy,
            quantity: Quantity::from_f64(100.0),
            limit_price: None,
            strategy_id: 1,
            timestamp: Timestamp::now(),
        }
    }

    #[test]
    fn test_pre_trade_checker_creation() {
        let config = PreTradeConfig::default();
        let checker = PreTradeChecker::new(config);
        assert_eq!(checker.check_count(), 0);
    }

    #[test]
    fn test_pre_trade_check() {
        let config = PreTradeConfig::default();
        let checker = PreTradeChecker::new(config);

        let order = create_test_order();
        let portfolio = Portfolio::new(100_000.0);

        let result = checker.check(&order, &portfolio);

        assert!(result.allowed);
        assert_eq!(checker.check_count(), 1);
        // Should complete within target latency in most cases
    }

    #[test]
    fn test_size_adjustment() {
        let config = PreTradeConfig::default();
        let checker = PreTradeChecker::new(config);

        // Portfolio in drawdown should reduce size
        let mut portfolio = Portfolio::new(100_000.0);
        portfolio.peak_value = 100_000.0;
        portfolio.total_value = 85_000.0; // 15% drawdown

        let order = create_test_order();
        let result = checker.check(&order, &portfolio);

        // Size should be reduced due to drawdown
        assert!(result.size_adjustment < 1.0);
    }
}
