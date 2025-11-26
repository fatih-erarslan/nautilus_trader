//! Comprehensive Validation Test Suite
//!
//! This module contains all validation tests for the Neural Trader Rust port.
//! Tests are organized by functional area and should be run after successful compilation.

#![cfg(test)]

pub mod test_strategies;
pub mod test_brokers;
pub mod test_neural;
pub mod test_multi_market;
pub mod test_risk;
pub mod test_mcp;
pub mod test_distributed;
pub mod test_memory;
pub mod test_integration;
pub mod test_performance;

/// Test utilities and helpers
pub mod helpers {
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;
    use chrono::{DateTime, Utc};

    /// Generate sample historical data for testing
    pub fn generate_sample_bars(count: usize) -> Vec<TestBar> {
        (0..count)
            .map(|i| TestBar {
                timestamp: Utc::now(),
                open: dec!(100.0) + Decimal::from(i),
                high: dec!(102.0) + Decimal::from(i),
                low: dec!(98.0) + Decimal::from(i),
                close: dec!(101.0) + Decimal::from(i),
                volume: dec!(1000000),
            })
            .collect()
    }

    pub struct TestBar {
        pub timestamp: DateTime<Utc>,
        pub open: Decimal,
        pub high: Decimal,
        pub low: Decimal,
        pub close: Decimal,
        pub volume: Decimal,
    }

    /// Assert performance target is met
    pub fn assert_performance_target(actual_ms: f64, target_ms: f64, tolerance: f64) {
        let max_allowed = target_ms * (1.0 + tolerance);
        assert!(
            actual_ms <= max_allowed,
            "Performance target not met: {:.2}ms > {:.2}ms (target: {:.2}ms, tolerance: {:.0}%)",
            actual_ms, max_allowed, target_ms, tolerance * 100.0
        );
    }

    /// Calculate Sharpe ratio for testing
    pub fn calculate_sharpe_ratio(returns: &[Decimal], risk_free_rate: Decimal) -> Decimal {
        if returns.is_empty() {
            return Decimal::ZERO;
        }

        let mean_return = returns.iter().sum::<Decimal>() / Decimal::from(returns.len());
        let variance = returns
            .iter()
            .map(|r| (*r - mean_return).powi(2))
            .sum::<Decimal>()
            / Decimal::from(returns.len());
        let std_dev = variance.sqrt().unwrap_or(Decimal::ONE);

        if std_dev == Decimal::ZERO {
            return Decimal::ZERO;
        }

        (mean_return - risk_free_rate) / std_dev
    }
}
