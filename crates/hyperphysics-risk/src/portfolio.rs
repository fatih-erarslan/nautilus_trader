use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Represents a single position in a portfolio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub cost_basis: f64,
}

impl Position {
    pub fn new(symbol: impl Into<String>, quantity: f64, cost_basis: f64) -> Self {
        Self {
            symbol: symbol.into(),
            quantity,
            cost_basis,
        }
    }

    /// Calculate current value of position
    pub fn value(&self, current_price: f64) -> f64 {
        self.quantity * current_price
    }

    /// Calculate unrealized P&L
    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        (current_price - self.cost_basis) * self.quantity
    }
}

/// Portfolio containing positions and cash
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    pub positions: HashMap<String, Position>,
    pub cash: f64,
}

impl Portfolio {
    pub fn new(cash: f64) -> Self {
        Self {
            positions: HashMap::new(),
            cash,
        }
    }

    /// Add or update a position
    pub fn add_position(&mut self, position: Position) {
        self.positions.insert(position.symbol.clone(), position);
    }

    /// Remove a position
    pub fn remove_position(&mut self, symbol: &str) -> Option<Position> {
        self.positions.remove(symbol)
    }

    /// Calculate total portfolio value at current prices
    pub fn total_value(&self, prices: &HashMap<String, f64>) -> f64 {
        let position_value: f64 = self.positions
            .iter()
            .map(|(symbol, pos)| {
                prices.get(symbol).unwrap_or(&0.0) * pos.quantity
            })
            .sum();

        self.cash + position_value
    }

    /// Get portfolio weights (normalized position values)
    ///
    /// Returns weights as a fraction of total portfolio value (including cash).
    /// Weights sum to less than 1.0 if cash is present.
    /// To get position-only weights that sum to 1.0, use position_weights().
    pub fn weights(&self, prices: &HashMap<String, f64>) -> Vec<f64> {
        let position_value: f64 = self.positions
            .iter()
            .map(|(symbol, pos)| {
                prices.get(symbol).unwrap_or(&0.0) * pos.quantity
            })
            .sum();

        if position_value <= 0.0 {
            return vec![0.0; self.positions.len()];
        }

        self.positions
            .iter()
            .map(|(symbol, pos)| {
                (prices.get(symbol).unwrap_or(&0.0) * pos.quantity) / position_value
            })
            .collect()
    }

    /// Get symbols in portfolio
    pub fn symbols(&self) -> Vec<String> {
        self.positions.keys().cloned().collect()
    }

    /// Calculate total unrealized P&L
    pub fn unrealized_pnl(&self, prices: &HashMap<String, f64>) -> f64 {
        self.positions
            .iter()
            .map(|(symbol, pos)| {
                prices
                    .get(symbol)
                    .map(|&price| pos.unrealized_pnl(price))
                    .unwrap_or(0.0)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_value() {
        let pos = Position::new("AAPL", 100.0, 150.0);
        assert_eq!(pos.value(160.0), 16000.0);
        assert_eq!(pos.unrealized_pnl(160.0), 1000.0);
    }

    #[test]
    fn test_portfolio_weights() {
        let mut portfolio = Portfolio::new(10000.0);
        portfolio.add_position(Position::new("AAPL", 100.0, 150.0));
        portfolio.add_position(Position::new("GOOGL", 50.0, 2800.0));

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), 160.0);
        prices.insert("GOOGL".to_string(), 2900.0);

        let total = portfolio.total_value(&prices);
        let weights = portfolio.weights(&prices);

        // Total should be cash + positions
        // Cash: 10000, AAPL: 16000, GOOGL: 145000 = 171000
        assert!((total - 171000.0).abs() < 1e-6);

        // Weights are position_value / total_value (excluding cash weight)
        // So weights sum to (positions / total) = 161000 / 171000 ≈ 0.9415
        let weight_sum: f64 = weights.iter().sum();
        let expected_weight_sum = 161000.0 / 171000.0;
        assert!((weight_sum - expected_weight_sum).abs() < 1e-6);
    }
}
