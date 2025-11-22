//! Risk Management and Position Sizing Module
//!
//! This module provides comprehensive risk management capabilities including:
//! - Position sizing strategies (Fixed, Percentage, Kelly Criterion, Volatility-based)
//! - Risk metrics (VaR, CVaR, Sharpe, Sortino ratios)
//! - Risk limits (max position size, max drawdown, max daily loss)
//! - Portfolio risk analysis (correlation, diversification)
//! - Stop loss management (Fixed, Trailing, ATR-based)
//! - Margin and leverage calculations
//! - Real-time risk monitoring
//!
//! # Example
//!
//! ```
//! use hyperphysics_market::risk::{RiskManager, RiskConfig, PositionSizingStrategy};
//!
//! let config = RiskConfig::default()
//!     .with_max_position_size(0.1)  // 10% max position
//!     .with_max_drawdown(0.2)        // 20% max drawdown
//!     .with_max_daily_loss(0.05);    // 5% max daily loss
//!
//! let mut risk_manager = RiskManager::new(100000.0, config);
//!
//! // Calculate position size using Kelly Criterion
//! let position_size = risk_manager.calculate_position_size(
//!     PositionSizingStrategy::Kelly { win_rate: 0.6, win_loss_ratio: 1.5 },
//!     50.0, // current price
//! );
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Position sizing strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PositionSizingStrategy {
    /// Fixed dollar amount per position
    Fixed { amount: f64 },

    /// Percentage of total capital
    Percentage { percentage: f64 },

    /// Kelly Criterion for optimal position sizing
    /// Formula: f = (p * b - q) / b
    /// where p = win rate, q = loss rate, b = win/loss ratio
    Kelly { win_rate: f64, win_loss_ratio: f64 },

    /// Volatility-based sizing (ATR multiplier)
    Volatility {
        atr: f64,           // Average True Range
        risk_per_trade: f64, // Risk amount per trade
    },
}

/// Stop loss types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum StopLossType {
    /// Fixed percentage stop loss
    Fixed { percentage: f64 },

    /// Trailing stop loss
    Trailing {
        percentage: f64,
        activation_profit: f64, // Profit % before trailing activates
    },

    /// ATR-based stop loss
    AtrBased {
        atr: f64,
        multiplier: f64, // ATR multiplier
    },
}

/// Risk limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    /// Maximum position size as percentage of portfolio
    pub max_position_size: f64,

    /// Maximum drawdown allowed (as percentage)
    pub max_drawdown: f64,

    /// Maximum daily loss allowed (as percentage)
    pub max_daily_loss: f64,

    /// Maximum leverage allowed
    pub max_leverage: f64,

    /// Maximum correlation between positions
    pub max_correlation: f64,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_position_size: 0.1,  // 10%
            max_drawdown: 0.2,        // 20%
            max_daily_loss: 0.05,     // 5%
            max_leverage: 1.0,        // No leverage
            max_correlation: 0.7,     // 70% max correlation
        }
    }
}

/// Risk manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    pub limits: RiskLimits,
    pub risk_free_rate: f64, // Annual risk-free rate for Sharpe/Sortino
    pub confidence_level: f64, // For VaR/CVaR calculations (e.g., 0.95)
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            limits: RiskLimits::default(),
            risk_free_rate: 0.02, // 2% annual
            confidence_level: 0.95, // 95% confidence
        }
    }
}

impl RiskConfig {
    pub fn with_max_position_size(mut self, size: f64) -> Self {
        self.limits.max_position_size = size;
        self
    }

    pub fn with_max_drawdown(mut self, drawdown: f64) -> Self {
        self.limits.max_drawdown = drawdown;
        self
    }

    pub fn with_max_daily_loss(mut self, loss: f64) -> Self {
        self.limits.max_daily_loss = loss;
        self
    }

    pub fn with_max_leverage(mut self, leverage: f64) -> Self {
        self.limits.max_leverage = leverage;
        self
    }

    pub fn with_risk_free_rate(mut self, rate: f64) -> Self {
        self.risk_free_rate = rate;
        self
    }
}

/// Position information for risk tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub entry_price: f64,
    pub current_price: f64,
    pub quantity: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub entry_time: DateTime<Utc>,
}

impl Position {
    /// Calculate unrealized P&L
    pub fn unrealized_pnl(&self) -> f64 {
        (self.current_price - self.entry_price) * self.quantity
    }

    /// Calculate unrealized P&L percentage
    pub fn unrealized_pnl_pct(&self) -> f64 {
        ((self.current_price - self.entry_price) / self.entry_price) * 100.0
    }

    /// Get position value at current price
    pub fn market_value(&self) -> f64 {
        self.current_price * self.quantity.abs()
    }

    /// Update current price
    pub fn update_price(&mut self, price: f64) {
        self.current_price = price;
    }
}

/// Portfolio metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioMetrics {
    pub total_value: f64,
    pub total_pnl: f64,
    pub daily_pnl: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub var_95: f64,        // Value at Risk (95% confidence)
    pub cvar_95: f64,       // Conditional VaR (95% confidence)
    pub diversification_score: f64,
    pub leverage: f64,
}

/// Main risk manager
pub struct RiskManager {
    /// Initial capital
    initial_capital: f64,

    /// Current capital
    current_capital: f64,

    /// Peak capital (for drawdown calculation)
    peak_capital: f64,

    /// Risk configuration
    config: RiskConfig,

    /// Active positions
    positions: HashMap<String, Position>,

    /// Historical returns for metrics calculation
    returns: Vec<f64>,

    /// Daily P&L tracking
    daily_pnl: f64,
    daily_start_capital: f64,

    /// Historical daily returns
    daily_returns: Vec<f64>,
}

impl RiskManager {
    /// Create a new risk manager
    pub fn new(initial_capital: f64, config: RiskConfig) -> Self {
        Self {
            initial_capital,
            current_capital: initial_capital,
            peak_capital: initial_capital,
            config,
            positions: HashMap::new(),
            returns: Vec::new(),
            daily_pnl: 0.0,
            daily_start_capital: initial_capital,
            daily_returns: Vec::new(),
        }
    }

    /// Calculate position size based on strategy
    pub fn calculate_position_size(
        &self,
        strategy: PositionSizingStrategy,
        price: f64,
    ) -> f64 {
        let size = match strategy {
            PositionSizingStrategy::Fixed { amount } => amount / price,

            PositionSizingStrategy::Percentage { percentage } => {
                let amount = self.current_capital * percentage;
                amount / price
            }

            PositionSizingStrategy::Kelly { win_rate, win_loss_ratio } => {
                let loss_rate = 1.0 - win_rate;
                let kelly_fraction = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio;
                let kelly_fraction = kelly_fraction.max(0.0).min(0.25); // Cap at 25%
                let amount = self.current_capital * kelly_fraction;
                amount / price
            }

            PositionSizingStrategy::Volatility { atr, risk_per_trade } => {
                // Position size = Risk Amount / (ATR * multiplier)
                risk_per_trade / atr
            }
        };

        // Apply max position size limit
        let max_position_value = self.current_capital * self.config.limits.max_position_size;
        let max_shares = max_position_value / price;

        size.min(max_shares)
    }

    /// Calculate stop loss level
    pub fn calculate_stop_loss(
        &self,
        entry_price: f64,
        stop_type: StopLossType,
        is_long: bool,
    ) -> f64 {
        match stop_type {
            StopLossType::Fixed { percentage } => {
                if is_long {
                    entry_price * (1.0 - percentage)
                } else {
                    entry_price * (1.0 + percentage)
                }
            }

            StopLossType::Trailing { percentage, .. } => {
                // Initial stop same as fixed
                if is_long {
                    entry_price * (1.0 - percentage)
                } else {
                    entry_price * (1.0 + percentage)
                }
            }

            StopLossType::AtrBased { atr, multiplier } => {
                if is_long {
                    entry_price - (atr * multiplier)
                } else {
                    entry_price + (atr * multiplier)
                }
            }
        }
    }

    /// Update trailing stop loss
    pub fn update_trailing_stop(
        &self,
        position: &Position,
        stop_type: StopLossType,
        is_long: bool,
    ) -> Option<f64> {
        if let StopLossType::Trailing { percentage, activation_profit } = stop_type {
            let profit_pct = position.unrealized_pnl_pct() / 100.0;

            // Check if trailing stop should activate
            if profit_pct >= activation_profit {
                let current_stop = position.stop_loss?;

                if is_long {
                    // Move stop up with price
                    let new_stop = position.current_price * (1.0 - percentage);
                    Some(new_stop.max(current_stop))
                } else {
                    // Move stop down with price
                    let new_stop = position.current_price * (1.0 + percentage);
                    Some(new_stop.min(current_stop))
                }
            } else {
                position.stop_loss
            }
        } else {
            position.stop_loss
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

    /// Update position price
    pub fn update_position_price(&mut self, symbol: &str, price: f64) {
        if let Some(position) = self.positions.get_mut(symbol) {
            position.update_price(price);
        }
    }

    /// Calculate Value at Risk (VaR)
    /// Using historical simulation method
    pub fn calculate_var(&self, confidence_level: f64) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }

        let mut sorted_returns = self.returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));

        let index = ((1.0 - confidence_level) * sorted_returns.len() as f64) as usize;
        let var = -sorted_returns[index.min(sorted_returns.len() - 1)];

        var * self.current_capital
    }

    /// Calculate Conditional VaR (Expected Shortfall)
    pub fn calculate_cvar(&self, confidence_level: f64) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }

        let mut sorted_returns = self.returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));

        let index = ((1.0 - confidence_level) * sorted_returns.len() as f64) as usize;
        let tail_returns: Vec<f64> = sorted_returns.iter().take(index + 1).copied().collect();

        if tail_returns.is_empty() {
            return 0.0;
        }

        let cvar = -tail_returns.iter().sum::<f64>() / tail_returns.len() as f64;
        cvar * self.current_capital
    }

    /// Calculate Sharpe ratio
    pub fn calculate_sharpe_ratio(&self, _period_days: usize) -> f64 {
        if self.returns.len() < 2 {
            return 0.0;
        }

        let mean_return = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
        let variance = self.returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / (self.returns.len() - 1) as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        // Annualize (assuming daily returns)
        let annualized_return = mean_return * 252.0;
        let annualized_std = std_dev * (252.0_f64).sqrt();
        let risk_free_rate = self.config.risk_free_rate;

        (annualized_return - risk_free_rate) / annualized_std
    }

    /// Calculate Sortino ratio (using downside deviation)
    pub fn calculate_sortino_ratio(&self, _period_days: usize) -> f64 {
        if self.returns.len() < 2 {
            return 0.0;
        }

        let mean_return = self.returns.iter().sum::<f64>() / self.returns.len() as f64;

        // Calculate downside deviation (only negative returns)
        let downside_returns: Vec<f64> = self.returns.iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();

        if downside_returns.is_empty() {
            return f64::INFINITY; // No downside risk
        }

        let downside_variance = downside_returns.iter()
            .map(|r| r.powi(2))
            .sum::<f64>() / downside_returns.len() as f64;
        let downside_std = downside_variance.sqrt();

        if downside_std == 0.0 {
            return f64::INFINITY;
        }

        // Annualize
        let annualized_return = mean_return * 252.0;
        let annualized_downside_std = downside_std * (252.0_f64).sqrt();
        let risk_free_rate = self.config.risk_free_rate;

        (annualized_return - risk_free_rate) / annualized_downside_std
    }

    /// Calculate portfolio diversification score
    /// Simple implementation based on position count and correlation
    pub fn calculate_diversification_score(&self) -> f64 {
        let position_count = self.positions.len() as f64;
        if position_count == 0.0 {
            return 0.0;
        }

        // Simple Herfindahl index for concentration
        let total_value = self.get_total_position_value();
        if total_value == 0.0 {
            return 0.0;
        }

        let herfindahl: f64 = self.positions.values()
            .map(|p| {
                let weight = p.market_value() / total_value;
                weight * weight
            })
            .sum();

        // Diversification score: 1 - H
        // Score of 1.0 means perfectly diversified
        // Score of 0.0 means concentrated in one position
        1.0 - herfindahl
    }

    /// Calculate current leverage
    pub fn calculate_leverage(&self) -> f64 {
        let total_position_value = self.get_total_position_value();
        if self.current_capital == 0.0 {
            return 0.0;
        }
        total_position_value / self.current_capital
    }

    /// Calculate margin requirement
    pub fn calculate_margin_requirement(&self, leverage: f64) -> f64 {
        let total_position_value = self.get_total_position_value();
        if leverage == 0.0 {
            return total_position_value;
        }
        total_position_value / leverage
    }

    /// Get total position value
    fn get_total_position_value(&self) -> f64 {
        self.positions.values()
            .map(|p| p.market_value())
            .sum()
    }

    /// Get total unrealized P&L
    pub fn get_total_unrealized_pnl(&self) -> f64 {
        self.positions.values()
            .map(|p| p.unrealized_pnl())
            .sum()
    }

    /// Calculate current drawdown
    pub fn calculate_drawdown(&self) -> f64 {
        if self.peak_capital == 0.0 {
            return 0.0;
        }
        (self.peak_capital - self.current_capital) / self.peak_capital
    }

    /// Update capital and metrics
    pub fn update_capital(&mut self, new_capital: f64) {
        let return_pct = (new_capital - self.current_capital) / self.current_capital;
        self.returns.push(return_pct);

        self.current_capital = new_capital;

        // Update peak capital
        if new_capital > self.peak_capital {
            self.peak_capital = new_capital;
        }

        // Update daily P&L
        self.daily_pnl = new_capital - self.daily_start_capital;
    }

    /// Reset daily tracking (call at start of each trading day)
    pub fn reset_daily(&mut self) {
        let daily_return = if self.daily_start_capital > 0.0 {
            self.daily_pnl / self.daily_start_capital
        } else {
            0.0
        };

        self.daily_returns.push(daily_return);
        self.daily_pnl = 0.0;
        self.daily_start_capital = self.current_capital;
    }

    /// Check if risk limits are violated
    pub fn check_risk_limits(&self) -> Vec<RiskViolation> {
        let mut violations = Vec::new();

        // Check max drawdown
        let current_drawdown = self.calculate_drawdown();
        if current_drawdown > self.config.limits.max_drawdown {
            violations.push(RiskViolation::MaxDrawdown {
                current: current_drawdown,
                limit: self.config.limits.max_drawdown,
            });
        }

        // Check max daily loss
        let daily_loss_pct = if self.daily_start_capital > 0.0 {
            -self.daily_pnl / self.daily_start_capital
        } else {
            0.0
        };

        if daily_loss_pct > self.config.limits.max_daily_loss {
            violations.push(RiskViolation::MaxDailyLoss {
                current: daily_loss_pct,
                limit: self.config.limits.max_daily_loss,
            });
        }

        // Check max leverage
        let current_leverage = self.calculate_leverage();
        if current_leverage > self.config.limits.max_leverage {
            violations.push(RiskViolation::MaxLeverage {
                current: current_leverage,
                limit: self.config.limits.max_leverage,
            });
        }

        // Check individual position sizes
        let total_value = self.current_capital + self.get_total_unrealized_pnl();
        for (symbol, position) in &self.positions {
            let position_pct = position.market_value() / total_value;
            if position_pct > self.config.limits.max_position_size {
                violations.push(RiskViolation::MaxPositionSize {
                    symbol: symbol.clone(),
                    current: position_pct,
                    limit: self.config.limits.max_position_size,
                });
            }
        }

        violations
    }

    /// Get portfolio metrics
    pub fn get_metrics(&self) -> PortfolioMetrics {
        PortfolioMetrics {
            total_value: self.current_capital + self.get_total_unrealized_pnl(),
            total_pnl: self.current_capital - self.initial_capital + self.get_total_unrealized_pnl(),
            daily_pnl: self.daily_pnl,
            max_drawdown: self.calculate_drawdown(),
            sharpe_ratio: self.calculate_sharpe_ratio(252),
            sortino_ratio: self.calculate_sortino_ratio(252),
            var_95: self.calculate_var(self.config.confidence_level),
            cvar_95: self.calculate_cvar(self.config.confidence_level),
            diversification_score: self.calculate_diversification_score(),
            leverage: self.calculate_leverage(),
        }
    }

    /// Get current positions
    pub fn get_positions(&self) -> &HashMap<String, Position> {
        &self.positions
    }

    /// Get risk configuration
    pub fn get_config(&self) -> &RiskConfig {
        &self.config
    }

    /// Update risk configuration
    pub fn update_config(&mut self, config: RiskConfig) {
        self.config = config;
    }
}

/// Risk violation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskViolation {
    MaxDrawdown { current: f64, limit: f64 },
    MaxDailyLoss { current: f64, limit: f64 },
    MaxLeverage { current: f64, limit: f64 },
    MaxPositionSize { symbol: String, current: f64, limit: f64 },
    MaxCorrelation { symbol1: String, symbol2: String, correlation: f64, limit: f64 },
}

impl std::fmt::Display for RiskViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskViolation::MaxDrawdown { current, limit } => {
                write!(f, "Max drawdown exceeded: {:.2}% > {:.2}%", current * 100.0, limit * 100.0)
            }
            RiskViolation::MaxDailyLoss { current, limit } => {
                write!(f, "Max daily loss exceeded: {:.2}% > {:.2}%", current * 100.0, limit * 100.0)
            }
            RiskViolation::MaxLeverage { current, limit } => {
                write!(f, "Max leverage exceeded: {:.2}x > {:.2}x", current, limit)
            }
            RiskViolation::MaxPositionSize { symbol, current, limit } => {
                write!(f, "Max position size exceeded for {}: {:.2}% > {:.2}%",
                    symbol, current * 100.0, limit * 100.0)
            }
            RiskViolation::MaxCorrelation { symbol1, symbol2, correlation, limit } => {
                write!(f, "Max correlation exceeded between {} and {}: {:.2} > {:.2}",
                    symbol1, symbol2, correlation, limit)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_position_sizing_fixed() {
        let config = RiskConfig::default();
        let risk_manager = RiskManager::new(100000.0, config);

        let strategy = PositionSizingStrategy::Fixed { amount: 10000.0 };
        let size = risk_manager.calculate_position_size(strategy, 50.0);

        assert_relative_eq!(size, 200.0, epsilon = 0.01); // 10000 / 50 = 200 shares
    }

    #[test]
    fn test_position_sizing_percentage() {
        let config = RiskConfig::default();
        let risk_manager = RiskManager::new(100000.0, config);

        let strategy = PositionSizingStrategy::Percentage { percentage: 0.1 };
        let size = risk_manager.calculate_position_size(strategy, 100.0);

        assert_relative_eq!(size, 100.0, epsilon = 0.01); // 10% of 100k = 10k / 100 = 100 shares
    }

    #[test]
    fn test_position_sizing_kelly() {
        let config = RiskConfig::default();
        let risk_manager = RiskManager::new(100000.0, config);

        // Win rate 60%, Win/Loss ratio 1.5
        // Kelly = (0.6 * 1.5 - 0.4) / 1.5 = (0.9 - 0.4) / 1.5 = 0.333
        // But capped at 25%
        let strategy = PositionSizingStrategy::Kelly {
            win_rate: 0.6,
            win_loss_ratio: 1.5
        };
        let size = risk_manager.calculate_position_size(strategy, 100.0);

        // Should be capped at 10% (max_position_size)
        assert_relative_eq!(size, 100.0, epsilon = 0.01);
    }

    #[test]
    fn test_position_sizing_volatility() {
        let config = RiskConfig::default();
        let risk_manager = RiskManager::new(100000.0, config);

        let strategy = PositionSizingStrategy::Volatility {
            atr: 2.0,
            risk_per_trade: 1000.0
        };
        let size = risk_manager.calculate_position_size(strategy, 50.0);

        // Volatility formula gives 1000 / 2 = 500 shares
        // But capped by max position size (10% of $100k = $10k / $50 = 200 shares)
        assert_relative_eq!(size, 200.0, epsilon = 0.01);
    }

    #[test]
    fn test_stop_loss_fixed() {
        let config = RiskConfig::default();
        let risk_manager = RiskManager::new(100000.0, config);

        let stop_type = StopLossType::Fixed { percentage: 0.02 }; // 2% stop
        let stop_price = risk_manager.calculate_stop_loss(100.0, stop_type, true);

        assert_relative_eq!(stop_price, 98.0, epsilon = 0.01);
    }

    #[test]
    fn test_stop_loss_atr() {
        let config = RiskConfig::default();
        let risk_manager = RiskManager::new(100000.0, config);

        let stop_type = StopLossType::AtrBased { atr: 1.5, multiplier: 2.0 };
        let stop_price = risk_manager.calculate_stop_loss(100.0, stop_type, true);

        assert_relative_eq!(stop_price, 97.0, epsilon = 0.01); // 100 - (1.5 * 2)
    }

    #[test]
    fn test_position_pnl() {
        let position = Position {
            symbol: "AAPL".to_string(),
            entry_price: 150.0,
            current_price: 160.0,
            quantity: 100.0,
            stop_loss: Some(145.0),
            take_profit: Some(165.0),
            entry_time: Utc::now(),
        };

        assert_relative_eq!(position.unrealized_pnl(), 1000.0, epsilon = 0.01);
        assert_relative_eq!(position.unrealized_pnl_pct(), 6.6666, epsilon = 0.01);
        assert_relative_eq!(position.market_value(), 16000.0, epsilon = 0.01);
    }

    #[test]
    fn test_risk_manager_positions() {
        let config = RiskConfig::default();
        let mut risk_manager = RiskManager::new(100000.0, config);

        let position = Position {
            symbol: "AAPL".to_string(),
            entry_price: 150.0,
            current_price: 160.0,
            quantity: 100.0,
            stop_loss: Some(145.0),
            take_profit: Some(165.0),
            entry_time: Utc::now(),
        };

        risk_manager.add_position(position);

        assert_eq!(risk_manager.get_positions().len(), 1);
        assert_relative_eq!(risk_manager.get_total_unrealized_pnl(), 1000.0, epsilon = 0.01);
    }

    #[test]
    fn test_sharpe_ratio() {
        let config = RiskConfig::default().with_risk_free_rate(0.02);
        let mut risk_manager = RiskManager::new(100000.0, config);

        // Add some returns
        risk_manager.returns = vec![0.01, 0.02, -0.01, 0.03, 0.01, 0.02, -0.005, 0.015];

        let sharpe = risk_manager.calculate_sharpe_ratio(252);
        assert!(sharpe > 0.0); // Should be positive with mostly positive returns
    }

    #[test]
    fn test_sortino_ratio() {
        let config = RiskConfig::default().with_risk_free_rate(0.02);
        let mut risk_manager = RiskManager::new(100000.0, config);

        // Add some returns (mix of positive and negative)
        risk_manager.returns = vec![0.01, 0.02, -0.01, 0.03, 0.01, 0.02, -0.005, 0.015];

        let sortino = risk_manager.calculate_sortino_ratio(252);
        assert!(sortino > 0.0);

        // Sortino should be higher than Sharpe for same data
        // (because it only penalizes downside volatility)
        let sharpe = risk_manager.calculate_sharpe_ratio(252);
        assert!(sortino > sharpe);
    }

    #[test]
    fn test_var_calculation() {
        let config = RiskConfig::default();
        let mut risk_manager = RiskManager::new(100000.0, config);

        // Add returns with known distribution
        risk_manager.returns = vec![-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05];

        let var = risk_manager.calculate_var(0.95);
        assert!(var > 0.0); // VaR should be positive
    }

    #[test]
    fn test_cvar_calculation() {
        let config = RiskConfig::default();
        let mut risk_manager = RiskManager::new(100000.0, config);

        risk_manager.returns = vec![-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05];

        let var = risk_manager.calculate_var(0.95);
        let cvar = risk_manager.calculate_cvar(0.95);

        // CVaR should be greater than VaR
        assert!(cvar >= var);
    }

    #[test]
    fn test_diversification_score() {
        let config = RiskConfig::default();
        let mut risk_manager = RiskManager::new(100000.0, config);

        // Add multiple positions
        risk_manager.add_position(Position {
            symbol: "AAPL".to_string(),
            entry_price: 150.0,
            current_price: 160.0,
            quantity: 50.0,
            stop_loss: None,
            take_profit: None,
            entry_time: Utc::now(),
        });

        risk_manager.add_position(Position {
            symbol: "GOOGL".to_string(),
            entry_price: 2800.0,
            current_price: 2900.0,
            quantity: 2.0,
            stop_loss: None,
            take_profit: None,
            entry_time: Utc::now(),
        });

        let score = risk_manager.calculate_diversification_score();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_leverage_calculation() {
        let config = RiskConfig::default();
        let mut risk_manager = RiskManager::new(100000.0, config);

        risk_manager.add_position(Position {
            symbol: "AAPL".to_string(),
            entry_price: 150.0,
            current_price: 150.0,
            quantity: 1000.0, // 150k position on 100k capital = 1.5x leverage
            stop_loss: None,
            take_profit: None,
            entry_time: Utc::now(),
        });

        let leverage = risk_manager.calculate_leverage();
        assert_relative_eq!(leverage, 1.5, epsilon = 0.01);
    }

    #[test]
    fn test_drawdown_calculation() {
        let config = RiskConfig::default();
        let mut risk_manager = RiskManager::new(100000.0, config);

        // Simulate some gains then losses
        risk_manager.update_capital(120000.0); // +20%
        assert_relative_eq!(risk_manager.calculate_drawdown(), 0.0, epsilon = 0.001);

        risk_manager.update_capital(96000.0); // Down to 96k from peak of 120k
        let drawdown = risk_manager.calculate_drawdown();
        assert_relative_eq!(drawdown, 0.2, epsilon = 0.001); // 20% drawdown
    }

    #[test]
    fn test_risk_violations() {
        let config = RiskConfig::default()
            .with_max_drawdown(0.1)
            .with_max_daily_loss(0.02);

        let mut risk_manager = RiskManager::new(100000.0, config);

        // Create large drawdown
        risk_manager.update_capital(120000.0);
        risk_manager.update_capital(85000.0);

        let violations = risk_manager.check_risk_limits();
        assert!(!violations.is_empty());

        // Should have max drawdown violation
        assert!(violations.iter().any(|v| matches!(v, RiskViolation::MaxDrawdown { .. })));
    }

    #[test]
    fn test_trailing_stop_update() {
        let config = RiskConfig::default();
        let risk_manager = RiskManager::new(100000.0, config);

        let position = Position {
            symbol: "AAPL".to_string(),
            entry_price: 100.0,
            current_price: 110.0, // 10% profit
            quantity: 100.0,
            stop_loss: Some(98.0),
            take_profit: None,
            entry_time: Utc::now(),
        };

        let stop_type = StopLossType::Trailing {
            percentage: 0.05,  // 5% trailing
            activation_profit: 0.05  // Activate after 5% profit
        };

        let new_stop = risk_manager.update_trailing_stop(&position, stop_type, true);

        // Stop should trail at 5% below current price
        assert!(new_stop.is_some());
        assert_relative_eq!(new_stop.unwrap(), 104.5, epsilon = 0.01); // 110 * 0.95
    }

    #[test]
    fn test_margin_requirement() {
        let config = RiskConfig::default();
        let mut risk_manager = RiskManager::new(100000.0, config);

        risk_manager.add_position(Position {
            symbol: "AAPL".to_string(),
            entry_price: 150.0,
            current_price: 150.0,
            quantity: 1000.0,
            stop_loss: None,
            take_profit: None,
            entry_time: Utc::now(),
        });

        // With 2x leverage, margin requirement should be half the position value
        let margin = risk_manager.calculate_margin_requirement(2.0);
        assert_relative_eq!(margin, 75000.0, epsilon = 0.01);
    }

    #[test]
    fn test_portfolio_metrics() {
        let config = RiskConfig::default();
        let mut risk_manager = RiskManager::new(100000.0, config);

        // Add some returns
        risk_manager.returns = vec![0.01, 0.02, -0.01, 0.015];

        // Add a position
        risk_manager.add_position(Position {
            symbol: "AAPL".to_string(),
            entry_price: 150.0,
            current_price: 160.0,
            quantity: 100.0,
            stop_loss: None,
            take_profit: None,
            entry_time: Utc::now(),
        });

        let metrics = risk_manager.get_metrics();

        assert!(metrics.total_value > 0.0);
        assert!(metrics.sharpe_ratio != 0.0);
        assert!(metrics.diversification_score >= 0.0);
        assert!(metrics.leverage >= 0.0);
    }
}
