//! Core types for risk management

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Asset identifier
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct Symbol(pub String);

impl Symbol {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self(symbol.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for Symbol {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for Symbol {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Trading position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: Symbol,
    pub quantity: Decimal,
    pub avg_entry_price: Decimal,
    pub current_price: Decimal,
    pub market_value: Decimal,
    pub unrealized_pnl: Decimal,
    pub unrealized_pnl_percent: Decimal,
    pub side: PositionSide,
    pub opened_at: DateTime<Utc>,
}

impl Position {
    /// Calculate position exposure (market value)
    pub fn exposure(&self) -> f64 {
        self.market_value.to_f64().unwrap_or(0.0)
    }

    /// Calculate unrealized P&L
    pub fn unrealized_pnl(&self) -> f64 {
        let price_diff = self.current_price - self.avg_entry_price;
        let pnl = price_diff * self.quantity;
        match self.side {
            PositionSide::Long => pnl.to_f64().unwrap_or(0.0),
            PositionSide::Short => -pnl.to_f64().unwrap_or(0.0),
        }
    }

    /// Calculate position risk (absolute value of exposure)
    pub fn risk(&self) -> f64 {
        self.exposure().abs()
    }
}

/// Position side (long or short)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PositionSide {
    Long,
    Short,
}

/// Portfolio containing multiple positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    pub positions: HashMap<Symbol, Position>,
    pub cash: Decimal,
    pub total_equity: Decimal,
    pub buying_power: Decimal,
    pub updated_at: DateTime<Utc>,
}

impl Portfolio {
    /// Create empty portfolio
    pub fn new(initial_cash: Decimal) -> Self {
        Self {
            positions: HashMap::new(),
            cash: initial_cash,
            total_equity: initial_cash,
            buying_power: initial_cash,
            updated_at: Utc::now(),
        }
    }

    /// Calculate total portfolio value
    pub fn total_value(&self) -> f64 {
        let positions_value: f64 = self
            .positions
            .values()
            .map(|p| p.market_value.to_f64().unwrap_or(0.0))
            .sum();
        positions_value + self.cash.to_f64().unwrap_or(0.0)
    }

    /// Calculate total unrealized P&L
    pub fn total_unrealized_pnl(&self) -> f64 {
        self.positions.values().map(|p| p.unrealized_pnl()).sum()
    }

    /// Get position by symbol
    pub fn get_position(&self, symbol: &Symbol) -> Option<&Position> {
        self.positions.get(symbol)
    }

    /// Add or update position
    pub fn update_position(&mut self, position: Position) {
        self.positions.insert(position.symbol.clone(), position);
        self.updated_at = Utc::now();
    }

    /// Remove position
    pub fn remove_position(&mut self, symbol: &Symbol) -> Option<Position> {
        self.updated_at = Utc::now();
        self.positions.remove(symbol)
    }

    /// Calculate portfolio concentration (Herfindahl index)
    pub fn concentration(&self) -> f64 {
        let total_value = self.total_value();
        if total_value == 0.0 {
            return 0.0;
        }

        self.positions
            .values()
            .map(|p| {
                let weight = p.exposure() / total_value;
                weight * weight
            })
            .sum()
    }
}

/// Asset for correlation and VaR calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub symbol: Symbol,
    pub returns: Vec<f64>,
    pub volatility: f64,
    pub weight: f64,
}

/// Value at Risk (VaR) calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaRResult {
    /// Value at Risk at specified confidence level
    pub var_95: f64,
    pub var_99: f64,
    /// Conditional VaR (Expected Shortfall)
    pub cvar_95: f64,
    pub cvar_99: f64,
    /// Expected return
    pub expected_return: f64,
    /// Portfolio volatility
    pub volatility: f64,
    /// Worst case loss in simulation
    pub worst_case: f64,
    /// Best case gain in simulation
    pub best_case: f64,
    /// Number of simulations performed
    pub num_simulations: usize,
    /// Time horizon in days
    pub time_horizon_days: usize,
    /// Calculation timestamp
    pub calculated_at: DateTime<Utc>,
}

/// Stress test scenario result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResult {
    pub scenario_name: String,
    pub immediate_impact: f64,
    pub final_returns_distribution: Vec<f64>,
    pub survival_probability: f64,
    pub var_95: f64,
    pub cvar_95: f64,
    pub expected_return: f64,
    pub volatility: f64,
    pub worst_case: f64,
    pub best_case: f64,
}

/// Correlation matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationMatrix {
    pub symbols: Vec<Symbol>,
    pub matrix: Vec<Vec<f64>>,
    pub calculated_at: DateTime<Utc>,
}

impl CorrelationMatrix {
    /// Get correlation between two symbols
    pub fn get_correlation(&self, symbol1: &Symbol, symbol2: &Symbol) -> Option<f64> {
        let idx1 = self.symbols.iter().position(|s| s == symbol1)?;
        let idx2 = self.symbols.iter().position(|s| s == symbol2)?;
        self.matrix.get(idx1)?.get(idx2).copied()
    }

    /// Get number of assets
    pub fn size(&self) -> usize {
        self.symbols.len()
    }
}

/// Risk limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimit {
    pub limit_type: RiskLimitType,
    pub threshold: f64,
    pub current_value: f64,
    pub enabled: bool,
}

impl RiskLimit {
    /// Check if limit is breached
    pub fn is_breached(&self) -> bool {
        self.enabled && self.current_value > self.threshold
    }

    /// Calculate utilization percentage
    pub fn utilization(&self) -> f64 {
        if self.threshold == 0.0 {
            0.0
        } else {
            (self.current_value / self.threshold) * 100.0
        }
    }
}

/// Types of risk limits
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RiskLimitType {
    /// Maximum position size per symbol
    PositionSize,
    /// Portfolio-level VaR limit
    PortfolioVaR,
    /// Maximum drawdown percentage
    MaxDrawdown,
    /// Maximum leverage
    MaxLeverage,
    /// Maximum concentration in single asset
    MaxConcentration,
    /// Maximum correlation exposure
    MaxCorrelation,
}

/// Alert severity level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Risk metrics for a portfolio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub var_95: f64,
    pub cvar_95: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub concentration: f64,
    pub leverage: f64,
    pub beta: Option<f64>,
    pub calculated_at: DateTime<Utc>,
}

/// Emergency action to take
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyAction {
    /// Halt all trading
    HaltTrading,
    /// Close specific position
    ClosePosition(Symbol),
    /// Close all positions
    CloseAllPositions,
    /// Reduce position size by percentage
    ReducePosition { symbol: Symbol, percentage: f64 },
    /// Activate circuit breaker
    CircuitBreaker { duration_seconds: u64 },
    /// Send alert to operators
    Alert { level: AlertLevel, message: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_position_unrealized_pnl() {
        let position = Position {
            symbol: Symbol::new("AAPL"),
            quantity: dec!(100),
            avg_entry_price: dec!(150.00),
            current_price: dec!(155.00),
            market_value: dec!(15500),
            unrealized_pnl: dec!(500),
            unrealized_pnl_percent: dec!(3.33),
            side: PositionSide::Long,
            opened_at: Utc::now(),
        };

        let pnl = position.unrealized_pnl();
        assert!((pnl - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_portfolio_concentration() {
        let mut portfolio = Portfolio::new(dec!(100000));

        portfolio.update_position(Position {
            symbol: Symbol::new("AAPL"),
            quantity: dec!(100),
            avg_entry_price: dec!(150.00),
            current_price: dec!(150.00),
            market_value: dec!(15000),
            unrealized_pnl: dec!(0),
            unrealized_pnl_percent: dec!(0),
            side: PositionSide::Long,
            opened_at: Utc::now(),
        });

        let concentration = portfolio.concentration();
        assert!(concentration > 0.0);
    }

    #[test]
    fn test_risk_limit_breach() {
        let limit = RiskLimit {
            limit_type: RiskLimitType::PortfolioVaR,
            threshold: 10000.0,
            current_value: 12000.0,
            enabled: true,
        };

        assert!(limit.is_breached());
        assert_eq!(limit.utilization(), 120.0);
    }

    #[test]
    fn test_correlation_matrix() {
        let matrix = CorrelationMatrix {
            symbols: vec![Symbol::new("AAPL"), Symbol::new("MSFT")],
            matrix: vec![vec![1.0, 0.8], vec![0.8, 1.0]],
            calculated_at: Utc::now(),
        };

        let corr = matrix.get_correlation(&Symbol::new("AAPL"), &Symbol::new("MSFT"));
        assert_eq!(corr, Some(0.8));
    }
}
