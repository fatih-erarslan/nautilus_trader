//! Position Limit Sentinel.
//!
//! Enforces per-asset and total portfolio position limits.
//! Target latency: <5μs

use std::sync::atomic::{AtomicU8, Ordering};

use dashmap::DashMap;

use crate::core::error::{Result, RiskError};
use crate::core::types::{Order, Portfolio, Symbol};
use crate::sentinels::base::{Sentinel, SentinelId, SentinelStats, SentinelStatus};

/// Position limit configuration.
#[derive(Debug, Clone)]
pub struct PositionLimitConfig {
    /// Maximum position size per asset (in base currency).
    pub max_position_value: f64,
    /// Maximum position as fraction of portfolio (e.g., 0.10 = 10%).
    pub max_position_pct: f64,
    /// Maximum total exposure (sum of absolute positions).
    pub max_total_exposure: f64,
    /// Maximum number of concurrent positions.
    pub max_positions: usize,
}

impl Default for PositionLimitConfig {
    fn default() -> Self {
        Self {
            max_position_value: 100_000.0,
            max_position_pct: 0.10,
            max_total_exposure: 500_000.0,
            max_positions: 50,
        }
    }
}

impl PositionLimitConfig {
    /// Conservative configuration for risk-averse strategies.
    pub fn conservative() -> Self {
        Self {
            max_position_value: 50_000.0,
            max_position_pct: 0.05,
            max_total_exposure: 200_000.0,
            max_positions: 20,
        }
    }

    /// Aggressive configuration for high-conviction strategies.
    pub fn aggressive() -> Self {
        Self {
            max_position_value: 250_000.0,
            max_position_pct: 0.20,
            max_total_exposure: 1_000_000.0,
            max_positions: 100,
        }
    }
}

/// Position limit sentinel.
///
/// Checks order against position limits with <5μs latency.
/// Uses lock-free data structures for concurrent access.
#[derive(Debug)]
pub struct PositionLimitSentinel {
    /// Sentinel ID.
    id: SentinelId,
    /// Configuration.
    config: PositionLimitConfig,
    /// Current status.
    status: AtomicU8,
    /// Per-symbol position limits (for custom overrides).
    symbol_limits: DashMap<u64, f64>,
    /// Statistics.
    stats: SentinelStats,
}

impl PositionLimitSentinel {
    /// Create new position limit sentinel.
    pub fn new(config: PositionLimitConfig) -> Self {
        Self {
            id: SentinelId::new("position_limit"),
            config,
            status: AtomicU8::new(SentinelStatus::Active as u8),
            symbol_limits: DashMap::new(),
            stats: SentinelStats::new(),
        }
    }

    /// Set custom limit for specific symbol.
    pub fn set_symbol_limit(&self, symbol: &Symbol, max_value: f64) {
        self.symbol_limits.insert(symbol.hash_value(), max_value);
    }

    /// Get effective limit for symbol.
    #[inline]
    fn get_limit(&self, symbol: &Symbol) -> f64 {
        self.symbol_limits
            .get(&symbol.hash_value())
            .map(|v| *v)
            .unwrap_or(self.config.max_position_value)
    }

    /// Check if order would exceed position limits.
    #[inline]
    fn check_position_limit(
        &self,
        order: &Order,
        portfolio: &Portfolio,
    ) -> Result<()> {
        // Get current position for symbol
        let current_position = portfolio
            .get_position_value(&order.symbol)
            .unwrap_or(0.0);

        // Calculate new position value
        let order_price = order.limit_price.map(|p| p.as_f64()).unwrap_or(0.0);
        let order_value = order.quantity.as_f64() * order_price;
        let new_position = if order.side.is_buy() {
            current_position + order_value
        } else {
            current_position - order_value
        };

        // Check absolute position limit
        let limit = self.get_limit(&order.symbol);
        if new_position.abs() > limit {
            return Err(RiskError::PositionLimitExceeded {
                symbol: order.symbol.as_str().to_string(),
                current: current_position,
                attempted: order_value,
                limit,
            });
        }

        // Check position as percentage of portfolio
        let position_pct = new_position.abs() / portfolio.total_value;
        if position_pct > self.config.max_position_pct {
            return Err(RiskError::ConcentrationLimitExceeded {
                symbol: order.symbol.as_str().to_string(),
                concentration: position_pct,
                limit: self.config.max_position_pct,
            });
        }

        Ok(())
    }

    /// Check total exposure limit.
    #[inline]
    fn check_total_exposure(&self, order: &Order, portfolio: &Portfolio) -> Result<()> {
        let order_price = order.limit_price.map(|p| p.as_f64()).unwrap_or(0.0);
        let order_value = order.quantity.as_f64() * order_price;
        let new_exposure = portfolio.total_exposure() + order_value.abs();

        if new_exposure > self.config.max_total_exposure {
            return Err(RiskError::ExposureLimitExceeded {
                current: portfolio.total_exposure(),
                attempted: order_value,
                limit: self.config.max_total_exposure,
            });
        }

        Ok(())
    }

    /// Check position count limit.
    #[inline]
    fn check_position_count(&self, order: &Order, portfolio: &Portfolio) -> Result<()> {
        // Only check for new positions
        let is_new_position = portfolio.get_position(&order.symbol).is_none();

        if is_new_position && portfolio.positions.len() >= self.config.max_positions {
            return Err(RiskError::MaxPositionsExceeded {
                current: portfolio.positions.len(),
                limit: self.config.max_positions,
            });
        }

        Ok(())
    }
}

impl Default for PositionLimitSentinel {
    fn default() -> Self {
        Self::new(PositionLimitConfig::default())
    }
}

impl Sentinel for PositionLimitSentinel {
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

    /// Check position limits.
    ///
    /// Target: <5μs
    #[inline]
    fn check(&self, order: &Order, portfolio: &Portfolio) -> Result<()> {
        let start = std::time::Instant::now();

        // Check all limits
        self.check_position_limit(order, portfolio)?;
        self.check_total_exposure(order, portfolio)?;
        self.check_position_count(order, portfolio)?;

        let latency_ns = start.elapsed().as_nanos() as u64;
        self.stats.record_check(latency_ns);

        Ok(())
    }

    fn reset(&self) {
        self.stats.reset();
        self.status.store(SentinelStatus::Active as u8, Ordering::SeqCst);
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
    use crate::core::types::{OrderSide, Price, Quantity, Timestamp};

    fn test_order(symbol: &str, quantity: f64, price: f64) -> Order {
        Order {
            symbol: Symbol::new(symbol),
            side: OrderSide::Buy,
            quantity: Quantity::from_f64(quantity),
            limit_price: Some(Price::from_f64(price)),
            strategy_id: 1,
            timestamp: Timestamp::now(),
        }
    }

    #[test]
    fn test_position_limit_pass() {
        let sentinel = PositionLimitSentinel::new(PositionLimitConfig {
            max_position_value: 100_000.0,
            max_position_pct: 0.10,
            max_total_exposure: 500_000.0,
            max_positions: 50,
        });

        let order = test_order("AAPL", 100.0, 150.0); // $15,000 order
        let portfolio = Portfolio::new(1_000_000.0);

        assert!(sentinel.check(&order, &portfolio).is_ok());
    }

    #[test]
    fn test_position_limit_exceeded() {
        let sentinel = PositionLimitSentinel::new(PositionLimitConfig {
            max_position_value: 10_000.0, // Low limit
            max_position_pct: 0.10,
            max_total_exposure: 500_000.0,
            max_positions: 50,
        });

        let order = test_order("AAPL", 100.0, 150.0); // $15,000 exceeds $10,000 limit
        let portfolio = Portfolio::new(1_000_000.0);

        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_symbol_limit() {
        let sentinel = PositionLimitSentinel::new(PositionLimitConfig {
            max_position_value: 10_000.0,
            ..Default::default()
        });

        // Set higher limit for AAPL
        sentinel.set_symbol_limit(&Symbol::new("AAPL"), 50_000.0);

        let order = test_order("AAPL", 100.0, 200.0); // $20,000 order
        let portfolio = Portfolio::new(1_000_000.0);

        // Should pass because AAPL has custom $50k limit
        assert!(sentinel.check(&order, &portfolio).is_ok());
    }

    #[test]
    fn test_latency() {
        let sentinel = PositionLimitSentinel::default();
        let order = test_order("AAPL", 100.0, 150.0);
        let portfolio = Portfolio::new(1_000_000.0);

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

        // Should be under 5μs
        assert!(
            avg_ns < 5000,
            "Position limit check too slow: {}ns average",
            avg_ns
        );
    }
}
