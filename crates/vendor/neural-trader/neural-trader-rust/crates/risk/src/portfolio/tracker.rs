//! Real-time portfolio tracking with position monitoring

use crate::{Result};
use crate::types::{Portfolio, Position, Symbol};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use std::sync::Arc;
use tracing::{debug, info};

/// Real-time portfolio tracker
pub struct PortfolioTracker {
    portfolio: Arc<RwLock<Portfolio>>,
    position_updates: Arc<DashMap<Symbol, Position>>,
    last_update: Arc<RwLock<DateTime<Utc>>>,
}

impl PortfolioTracker {
    /// Create a new portfolio tracker
    pub fn new(initial_cash: Decimal) -> Self {
        Self {
            portfolio: Arc::new(RwLock::new(Portfolio::new(initial_cash))),
            position_updates: Arc::new(DashMap::new()),
            last_update: Arc::new(RwLock::new(Utc::now())),
        }
    }

    /// Update a position in the portfolio
    pub async fn update_position(&self, position: Position) -> Result<()> {
        debug!("Updating position: {}", position.symbol);

        // Store in concurrent map for real-time access
        self.position_updates.insert(position.symbol.clone(), position.clone());

        // Update main portfolio
        let mut portfolio = self.portfolio.write();
        portfolio.update_position(position);

        // Update timestamp
        *self.last_update.write() = Utc::now();

        Ok(())
    }

    /// Get current portfolio snapshot
    pub fn get_portfolio(&self) -> Portfolio {
        self.portfolio.read().clone()
    }

    /// Get specific position
    pub fn get_position(&self, symbol: &Symbol) -> Option<Position> {
        self.position_updates.get(symbol).map(|entry| entry.clone())
    }

    /// Calculate total portfolio value
    pub async fn total_value(&self) -> f64 {
        self.portfolio.read().total_value()
    }

    /// Calculate total unrealized P&L
    pub async fn unrealized_pnl(&self) -> f64 {
        self.portfolio.read().total_unrealized_pnl()
    }

    /// Get portfolio concentration
    pub async fn concentration(&self) -> f64 {
        self.portfolio.read().concentration()
    }

    /// Get number of open positions
    pub fn position_count(&self) -> usize {
        self.position_updates.len()
    }

    /// Get last update timestamp
    pub fn last_updated(&self) -> DateTime<Utc> {
        *self.last_update.read()
    }

    /// Clear all positions (for testing or reset)
    pub async fn clear_positions(&self) -> Result<()> {
        info!("Clearing all positions");
        self.position_updates.clear();
        let mut portfolio = self.portfolio.write();
        portfolio.positions.clear();
        *self.last_update.write() = Utc::now();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PositionSide;
    use rust_decimal_macros::dec;

    fn create_test_position(symbol: &str, quantity: i64, price: f64) -> Position {
        let qty = Decimal::from(quantity);
        let p = Decimal::from_f64_retain(price).unwrap();
        Position {
            symbol: Symbol::new(symbol),
            quantity: qty,
            avg_entry_price: p,
            current_price: p,
            market_value: qty * p,
            unrealized_pnl: dec!(0),
            unrealized_pnl_percent: dec!(0),
            side: PositionSide::Long,
            opened_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_portfolio_tracker() {
        let tracker = PortfolioTracker::new(dec!(100000));

        let position = create_test_position("AAPL", 100, 150.0);
        tracker.update_position(position).await.unwrap();

        assert_eq!(tracker.position_count(), 1);

        let retrieved = tracker.get_position(&Symbol::new("AAPL"));
        assert!(retrieved.is_some());
    }

    #[tokio::test]
    async fn test_portfolio_value() {
        let tracker = PortfolioTracker::new(dec!(100000));

        tracker
            .update_position(create_test_position("AAPL", 100, 150.0))
            .await
            .unwrap();
        tracker
            .update_position(create_test_position("MSFT", 50, 300.0))
            .await
            .unwrap();

        let total = tracker.total_value().await;
        assert!(total > 100000.0); // Cash + positions
    }
}
