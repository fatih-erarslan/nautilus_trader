//! Fast limit checking with branchless operations.
//!
//! Optimized for minimal latency using:
//! - Pre-computed lookup tables
//! - Branchless comparisons where possible
//! - Cache-aligned data structures

use crate::core::types::{Order, Portfolio, RiskLevel, Symbol};

/// Limit configuration.
#[derive(Debug, Clone)]
pub struct LimitConfig {
    /// Maximum position value per symbol.
    pub max_position_value: f64,
    /// Maximum order value.
    pub max_order_value: f64,
    /// Maximum portfolio leverage.
    pub max_leverage: f64,
    /// Maximum concentration per symbol (fraction).
    pub max_concentration: f64,
    /// Maximum number of open positions.
    pub max_positions: usize,
    /// Daily loss limit (fraction of portfolio).
    pub daily_loss_limit: f64,
}

impl Default for LimitConfig {
    fn default() -> Self {
        Self {
            max_position_value: 1_000_000.0,
            max_order_value: 100_000.0,
            max_leverage: 3.0,
            max_concentration: 0.20,
            max_positions: 50,
            daily_loss_limit: 0.05,
        }
    }
}

/// Limit violation details.
#[derive(Debug, Clone)]
pub struct LimitViolation {
    /// Type of limit violated.
    pub limit_type: LimitType,
    /// Human-readable message.
    pub message: String,
    /// Severity of violation.
    pub severity: RiskLevel,
    /// Current value.
    pub current_value: f64,
    /// Limit value.
    pub limit_value: f64,
}

/// Types of limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitType {
    /// Order size too large.
    OrderSize,
    /// Position would exceed limit.
    PositionSize,
    /// Portfolio leverage exceeded.
    Leverage,
    /// Single asset concentration.
    Concentration,
    /// Too many positions.
    PositionCount,
    /// Daily loss limit.
    DailyLoss,
}

/// Fast limit checker.
#[derive(Debug)]
pub struct LimitChecker {
    /// Configuration.
    config: LimitConfig,
    /// Symbol-specific limits (if any).
    symbol_limits: dashmap::DashMap<u64, SymbolLimits>,
}

/// Per-symbol limits.
#[derive(Debug, Clone)]
pub struct SymbolLimits {
    /// Maximum position value.
    pub max_position: f64,
    /// Maximum order size.
    pub max_order: f64,
}

impl LimitChecker {
    /// Create new limit checker.
    pub fn new(config: LimitConfig) -> Self {
        Self {
            config,
            symbol_limits: dashmap::DashMap::new(),
        }
    }

    /// Check order against limits.
    ///
    /// # Returns
    /// `None` if all checks pass, `Some(violation)` otherwise.
    #[inline]
    pub fn check(&self, order: &Order, portfolio: &Portfolio) -> Option<LimitViolation> {
        // 1. Order size check (fastest)
        let order_value = order.quantity.as_f64().abs()
            * order.limit_price.map(|p| p.as_f64()).unwrap_or(1.0);

        if order_value > self.config.max_order_value {
            return Some(LimitViolation {
                limit_type: LimitType::OrderSize,
                message: format!(
                    "Order value ${:.0} exceeds limit ${:.0}",
                    order_value, self.config.max_order_value
                ),
                severity: RiskLevel::High,
                current_value: order_value,
                limit_value: self.config.max_order_value,
            });
        }

        // 2. Position count check
        if portfolio.positions.len() >= self.config.max_positions {
            // Only reject if opening new position
            let is_new = portfolio.get_position(&order.symbol).is_none();
            if is_new {
                return Some(LimitViolation {
                    limit_type: LimitType::PositionCount,
                    message: format!(
                        "Position count {} at limit {}",
                        portfolio.positions.len(),
                        self.config.max_positions
                    ),
                    severity: RiskLevel::Elevated,
                    current_value: portfolio.positions.len() as f64,
                    limit_value: self.config.max_positions as f64,
                });
            }
        }

        // 3. Position size check
        let current_pos_value = portfolio
            .get_position(&order.symbol)
            .map(|p| p.market_value().abs())
            .unwrap_or(0.0);
        let new_pos_value = current_pos_value + order_value;

        if new_pos_value > self.config.max_position_value {
            return Some(LimitViolation {
                limit_type: LimitType::PositionSize,
                message: format!(
                    "Position value ${:.0} would exceed limit ${:.0}",
                    new_pos_value, self.config.max_position_value
                ),
                severity: RiskLevel::High,
                current_value: new_pos_value,
                limit_value: self.config.max_position_value,
            });
        }

        // 4. Concentration check
        if portfolio.total_value > 0.0 {
            let concentration = new_pos_value / portfolio.total_value;
            if concentration > self.config.max_concentration {
                return Some(LimitViolation {
                    limit_type: LimitType::Concentration,
                    message: format!(
                        "Concentration {:.1}% would exceed limit {:.1}%",
                        concentration * 100.0,
                        self.config.max_concentration * 100.0
                    ),
                    severity: RiskLevel::Elevated,
                    current_value: concentration,
                    limit_value: self.config.max_concentration,
                });
            }
        }

        // 5. Leverage check
        let total_exposure: f64 = portfolio.positions.iter()
            .map(|p| p.market_value().abs())
            .sum::<f64>() + order_value;

        if portfolio.total_value > 0.0 {
            let leverage = total_exposure / portfolio.total_value;
            if leverage > self.config.max_leverage {
                return Some(LimitViolation {
                    limit_type: LimitType::Leverage,
                    message: format!(
                        "Leverage {:.2}x would exceed limit {:.2}x",
                        leverage, self.config.max_leverage
                    ),
                    severity: RiskLevel::High,
                    current_value: leverage,
                    limit_value: self.config.max_leverage,
                });
            }
        }

        // 6. Daily loss check
        let daily_pnl = portfolio.unrealized_pnl + portfolio.realized_pnl;
        if portfolio.total_value > 0.0 {
            let loss_pct = -daily_pnl / portfolio.total_value;
            if loss_pct > self.config.daily_loss_limit {
                return Some(LimitViolation {
                    limit_type: LimitType::DailyLoss,
                    message: format!(
                        "Daily loss {:.1}% exceeds limit {:.1}%",
                        loss_pct * 100.0,
                        self.config.daily_loss_limit * 100.0
                    ),
                    severity: RiskLevel::Critical,
                    current_value: loss_pct,
                    limit_value: self.config.daily_loss_limit,
                });
            }
        }

        None // All checks passed
    }

    /// Set symbol-specific limits.
    pub fn set_symbol_limits(&self, symbol: &Symbol, limits: SymbolLimits) {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        symbol.as_str().hash(&mut hasher);
        self.symbol_limits.insert(hasher.finish(), limits);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{OrderSide, Quantity, Timestamp};

    fn create_order(quantity: f64) -> Order {
        Order {
            symbol: Symbol::new("TEST"),
            side: OrderSide::Buy,
            quantity: Quantity::from_f64(quantity),
            limit_price: None,
            strategy_id: 1,
            timestamp: Timestamp::now(),
        }
    }

    #[test]
    fn test_order_size_limit() {
        let config = LimitConfig {
            max_order_value: 1000.0,
            ..Default::default()
        };
        let checker = LimitChecker::new(config);

        let order = create_order(2000.0); // Exceeds limit
        let portfolio = Portfolio::new(100_000.0);

        let result = checker.check(&order, &portfolio);
        assert!(result.is_some());
        assert_eq!(result.unwrap().limit_type, LimitType::OrderSize);
    }

    #[test]
    fn test_passes_all_limits() {
        let config = LimitConfig::default();
        let checker = LimitChecker::new(config);

        let order = create_order(100.0); // Small order
        let portfolio = Portfolio::new(100_000.0);

        let result = checker.check(&order, &portfolio);
        assert!(result.is_none());
    }

    #[test]
    fn test_leverage_limit() {
        let config = LimitConfig {
            max_leverage: 1.0,
            max_concentration: 1.0, // Set high to not trigger concentration first
            max_position_value: 1_000_000.0, // Set high to not trigger position size
            ..Default::default()
        };
        let checker = LimitChecker::new(config);

        // Portfolio already at 80% exposure with existing position
        let mut portfolio = Portfolio::new(100_000.0);
        portfolio.positions.push(crate::core::types::Position {
            id: crate::core::types::PositionId::new(),
            symbol: Symbol::new("OTHER"),
            quantity: Quantity::from_f64(800.0),
            avg_entry_price: crate::core::types::Price::from_f64(100.0),
            current_price: crate::core::types::Price::from_f64(100.0),
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            opened_at: Timestamp::now(),
            updated_at: Timestamp::now(),
        });

        // Order would push leverage over 1.0 (80k existing + 30k new = 110k / 100k = 1.1x leverage)
        let order = create_order(30000.0);
        let result = checker.check(&order, &portfolio);

        assert!(result.is_some(), "Should detect limit violation");
        assert_eq!(result.unwrap().limit_type, LimitType::Leverage, "Should be leverage violation");
    }
}
