//! P&L (Profit and Loss) calculation

use crate::{Result};
use crate::types::{Portfolio};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};

/// P&L calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PnLResult {
    /// Total realized P&L (closed positions)
    pub realized: f64,
    /// Total unrealized P&L (open positions)
    pub unrealized: f64,
    /// Total P&L (realized + unrealized)
    pub total: f64,
    /// P&L percentage relative to initial capital
    pub total_percent: f64,
    /// Daily P&L
    pub daily: f64,
    /// Calculation timestamp
    pub calculated_at: DateTime<Utc>,
}

/// P&L calculator
pub struct PnLCalculator {
    initial_capital: Decimal,
    realized_pnl: Decimal,
}

impl PnLCalculator {
    /// Create a new P&L calculator
    pub fn new(initial_capital: Decimal) -> Self {
        Self {
            initial_capital,
            realized_pnl: Decimal::ZERO,
        }
    }

    /// Calculate P&L for a portfolio
    pub async fn calculate(&self, portfolio: &Portfolio) -> Result<PnLResult> {
        // Calculate unrealized P&L from open positions
        let unrealized = portfolio.total_unrealized_pnl();

        // Realized P&L (tracked separately)
        let realized = self.realized_pnl.to_f64().unwrap_or(0.0);

        // Total P&L
        let total = realized + unrealized;

        // P&L percentage
        let initial = self.initial_capital.to_f64().unwrap_or(1.0);
        let total_percent = if initial > 0.0 {
            (total / initial) * 100.0
        } else {
            0.0
        };

        // Daily P&L (simplified - would track previous day's value in production)
        let daily = unrealized; // Simplified

        Ok(PnLResult {
            realized,
            unrealized,
            total,
            total_percent,
            daily,
            calculated_at: Utc::now(),
        })
    }

    /// Record realized P&L from a closed position
    pub fn record_realized_pnl(&mut self, pnl: Decimal) {
        self.realized_pnl += pnl;
    }

    /// Get total realized P&L
    pub fn total_realized(&self) -> f64 {
        self.realized_pnl.to_f64().unwrap_or(0.0)
    }

    /// Reset realized P&L (for new trading period)
    pub fn reset_realized(&mut self) {
        self.realized_pnl = Decimal::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PositionSide, Symbol};
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_pnl_calculation() {
        let calculator = PnLCalculator::new(dec!(100000));
        let mut portfolio = Portfolio::new(dec!(100000));

        // Add a position with unrealized P&L
        portfolio.update_position(Position {
            symbol: Symbol::new("AAPL"),
            quantity: dec!(100),
            avg_entry_price: dec!(150),
            current_price: dec!(155),
            market_value: dec!(15500),
            unrealized_pnl: dec!(500),
            unrealized_pnl_percent: dec!(3.33),
            side: PositionSide::Long,
            opened_at: Utc::now(),
        });

        let result = calculator.calculate(&portfolio).await.unwrap();
        assert!(result.unrealized > 0.0);
        assert_eq!(result.realized, 0.0);
    }
}
