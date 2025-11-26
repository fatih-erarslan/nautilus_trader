//! Common types used across all market modules

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Market type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketType {
    /// Sports betting market
    Sports,
    /// Prediction market
    Prediction,
    /// Cryptocurrency market
    Crypto,
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    /// Buy order
    Buy,
    /// Sell order
    Sell,
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    /// Market order (immediate execution)
    Market,
    /// Limit order (execute at specific price)
    Limit,
    /// Stop order
    Stop,
    /// Stop-limit order
    StopLimit,
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    /// Order pending placement
    Pending,
    /// Order placed and open
    Open,
    /// Order partially filled
    PartiallyFilled,
    /// Order fully filled
    Filled,
    /// Order cancelled
    Cancelled,
    /// Order rejected
    Rejected,
    /// Order expired
    Expired,
}

/// Generic order structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Unique order identifier
    pub id: String,
    /// Market identifier
    pub market_id: String,
    /// Order side
    pub side: OrderSide,
    /// Order type
    pub order_type: OrderType,
    /// Order size/amount
    pub size: Decimal,
    /// Order price (for limit orders)
    pub price: Option<Decimal>,
    /// Order status
    pub status: OrderStatus,
    /// Amount filled
    pub filled: Decimal,
    /// Amount remaining
    pub remaining: Decimal,
    /// Order creation time
    pub created_at: DateTime<Utc>,
    /// Order update time
    pub updated_at: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Position in a market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Market identifier
    pub market_id: String,
    /// Market type
    pub market_type: MarketType,
    /// Position size
    pub size: Decimal,
    /// Average entry price
    pub entry_price: Decimal,
    /// Current market price
    pub current_price: Decimal,
    /// Unrealized PnL
    pub unrealized_pnl: Decimal,
    /// Realized PnL
    pub realized_pnl: Decimal,
    /// Position opened time
    pub opened_at: DateTime<Utc>,
    /// Position last updated
    pub updated_at: DateTime<Utc>,
}

impl Position {
    /// Calculate position value
    pub fn value(&self) -> Decimal {
        self.size * self.current_price
    }

    /// Calculate position cost
    pub fn cost(&self) -> Decimal {
        self.size * self.entry_price
    }

    /// Calculate return percentage
    pub fn return_pct(&self) -> Decimal {
        if self.entry_price.is_zero() {
            Decimal::ZERO
        } else {
            ((self.current_price - self.entry_price) / self.entry_price) * Decimal::from(100)
        }
    }
}

/// Portfolio holding multiple positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    /// Portfolio identifier
    pub id: String,
    /// Total capital
    pub total_capital: Decimal,
    /// Available capital
    pub available_capital: Decimal,
    /// Open positions
    pub positions: Vec<Position>,
    /// Total unrealized PnL
    pub total_unrealized_pnl: Decimal,
    /// Total realized PnL
    pub total_realized_pnl: Decimal,
    /// Portfolio creation time
    pub created_at: DateTime<Utc>,
    /// Portfolio last updated
    pub updated_at: DateTime<Utc>,
}

impl Portfolio {
    /// Create new portfolio
    pub fn new(id: String, capital: Decimal) -> Self {
        let now = Utc::now();
        Self {
            id,
            total_capital: capital,
            available_capital: capital,
            positions: Vec::new(),
            total_unrealized_pnl: Decimal::ZERO,
            total_realized_pnl: Decimal::ZERO,
            created_at: now,
            updated_at: now,
        }
    }

    /// Calculate total portfolio value
    pub fn total_value(&self) -> Decimal {
        self.available_capital + self.positions.iter().map(|p| p.value()).sum::<Decimal>()
    }

    /// Calculate total PnL
    pub fn total_pnl(&self) -> Decimal {
        self.total_unrealized_pnl + self.total_realized_pnl
    }

    /// Calculate portfolio return percentage
    pub fn return_pct(&self) -> Decimal {
        if self.total_capital.is_zero() {
            Decimal::ZERO
        } else {
            (self.total_pnl() / self.total_capital) * Decimal::from(100)
        }
    }

    /// Get position by market ID
    pub fn get_position(&self, market_id: &str) -> Option<&Position> {
        self.positions.iter().find(|p| p.market_id == market_id)
    }

    /// Update position
    pub fn update_position(&mut self, position: Position) {
        if let Some(pos) = self
            .positions
            .iter_mut()
            .find(|p| p.market_id == position.market_id)
        {
            *pos = position;
        } else {
            self.positions.push(position);
        }
        self.updated_at = Utc::now();
    }
}

/// Arbitrage opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    /// Opportunity identifier
    pub id: String,
    /// Market A identifier
    pub market_a: String,
    /// Market B identifier
    pub market_b: String,
    /// Price in market A
    pub price_a: Decimal,
    /// Price in market B
    pub price_b: Decimal,
    /// Expected profit percentage
    pub profit_pct: Decimal,
    /// Expected profit amount
    pub profit_amount: Decimal,
    /// Confidence score (0-1)
    pub confidence: Decimal,
    /// Required capital
    pub required_capital: Decimal,
    /// Opportunity detected time
    pub detected_at: DateTime<Utc>,
    /// Opportunity expiry time
    pub expires_at: Option<DateTime<Utc>>,
}

impl ArbitrageOpportunity {
    /// Check if opportunity is still valid
    pub fn is_valid(&self) -> bool {
        if let Some(expiry) = self.expires_at {
            Utc::now() < expiry
        } else {
            true
        }
    }

    /// Calculate risk-adjusted profit
    pub fn risk_adjusted_profit(&self) -> Decimal {
        self.profit_amount * self.confidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_calculations() {
        let position = Position {
            market_id: "test_market".to_string(),
            market_type: MarketType::Sports,
            size: Decimal::from(100),
            entry_price: Decimal::from(10),
            current_price: Decimal::from(12),
            unrealized_pnl: Decimal::from(200),
            realized_pnl: Decimal::ZERO,
            opened_at: Utc::now(),
            updated_at: Utc::now(),
        };

        assert_eq!(position.value(), Decimal::from(1200));
        assert_eq!(position.cost(), Decimal::from(1000));
        assert_eq!(position.return_pct(), Decimal::from(20));
    }

    #[test]
    fn test_portfolio_calculations() {
        let mut portfolio = Portfolio::new("test".to_string(), Decimal::from(10000));

        let position = Position {
            market_id: "test_market".to_string(),
            market_type: MarketType::Sports,
            size: Decimal::from(100),
            entry_price: Decimal::from(10),
            current_price: Decimal::from(12),
            unrealized_pnl: Decimal::from(200),
            realized_pnl: Decimal::ZERO,
            opened_at: Utc::now(),
            updated_at: Utc::now(),
        };

        portfolio.update_position(position);
        portfolio.available_capital = Decimal::from(9000);
        portfolio.total_unrealized_pnl = Decimal::from(200);

        assert_eq!(portfolio.total_value(), Decimal::from(10200));
        assert_eq!(portfolio.total_pnl(), Decimal::from(200));
        assert_eq!(portfolio.return_pct(), Decimal::from(2));
    }

    #[test]
    fn test_arbitrage_validity() {
        let opp = ArbitrageOpportunity {
            id: "test".to_string(),
            market_a: "market_a".to_string(),
            market_b: "market_b".to_string(),
            price_a: Decimal::from(100),
            price_b: Decimal::from(105),
            profit_pct: Decimal::from(5),
            profit_amount: Decimal::from(50),
            confidence: Decimal::new(8, 1), // 0.8
            required_capital: Decimal::from(1000),
            detected_at: Utc::now(),
            expires_at: Some(Utc::now() + chrono::Duration::hours(1)),
        };

        assert!(opp.is_valid());
        assert_eq!(opp.risk_adjusted_profit(), Decimal::from(40));
    }
}
