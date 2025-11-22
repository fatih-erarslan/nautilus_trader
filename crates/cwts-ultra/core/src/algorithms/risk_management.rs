use crate::validation::ieee754_arithmetic::ArithmeticError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RiskError {
    #[error("Invalid position size: {0}")]
    InvalidPositionSize(f64),
    #[error("Insufficient margin: required {required}, available {available}")]
    InsufficientMargin { required: f64, available: f64 },
    #[error("Maximum drawdown exceeded: {0}%")]
    MaxDrawdownExceeded(f64),
    #[error("Correlation limit exceeded: {0}")]
    CorrelationLimitExceeded(f64),
    #[error("Risk calculation error: {0}")]
    CalculationError(String),
    #[error("IEEE 754 arithmetic error: {0}")]
    ArithmeticError(#[from] ArithmeticError),
    #[error("Regulatory compliance violation: {0}")]
    RegulatoryViolation(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub size: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub margin_used: f64,
    pub leverage: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub portfolio_value: f64,
    pub total_margin_used: f64,
    pub free_margin: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub max_drawdown: f64,
    pub current_drawdown: f64,
    pub var_95: f64, // Value at Risk 95%
    pub var_99: f64, // Value at Risk 99%
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_leverage: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct RiskParameters {
    pub max_position_size_pct: f64,
    pub max_portfolio_risk_pct: f64,
    pub max_drawdown_pct: f64,
    pub max_leverage: f64,
    pub max_correlation: f64,
    pub kelly_lookback: usize,
    pub var_confidence: f64,
    pub risk_free_rate: f64,
}

impl Default for RiskParameters {
    fn default() -> Self {
        Self {
            max_position_size_pct: 2.0,
            max_portfolio_risk_pct: 20.0,
            max_drawdown_pct: 15.0,
            max_leverage: 10.0,
            max_correlation: 0.7,
            kelly_lookback: 252,
            var_confidence: 0.05,
            risk_free_rate: 0.02,
        }
    }
}

pub struct RiskManager {
    parameters: RiskParameters,
    positions: HashMap<String, Position>,
    historical_returns: HashMap<String, Vec<f64>>,
    portfolio_history: Vec<f64>,
    high_water_mark: f64,
}

impl RiskManager {
    pub fn new(parameters: RiskParameters) -> Self {
        Self {
            parameters,
            positions: HashMap::new(),
            historical_returns: HashMap::new(),
            portfolio_history: Vec::new(),
            high_water_mark: 0.0,
        }
    }

    /// Calculate position size using Kelly criterion
    pub fn calculate_kelly_position_size(
        &self,
        _symbol: &str,
        win_rate: f64,
        avg_win: f64,
        avg_loss: f64,
        portfolio_value: f64,
    ) -> Result<f64, RiskError> {
        if win_rate <= 0.0 || win_rate >= 1.0 {
            return Err(RiskError::CalculationError(
                "Win rate must be between 0 and 1".to_string(),
            ));
        }

        // Kelly formula: f = (bp - q) / b
        // where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        let b = if avg_loss.abs() > 0.0 {
            avg_win / avg_loss.abs()
        } else {
            0.0
        };
        let p = win_rate;
        let q = 1.0 - win_rate;

        let kelly_fraction = (b * p - q) / b;
        let kelly_capped = kelly_fraction.min(self.parameters.max_position_size_pct / 100.0);

        let position_size = portfolio_value * kelly_capped.max(0.0);

        Ok(position_size)
    }

    /// Calculate position size using fixed fractional method
    pub fn calculate_fixed_fractional_size(
        &self,
        portfolio_value: f64,
        risk_per_trade: f64,
        entry_price: f64,
        stop_loss_price: f64,
    ) -> Result<f64, RiskError> {
        let risk_amount = portfolio_value * (risk_per_trade / 100.0);
        let price_diff = (entry_price - stop_loss_price).abs();

        if price_diff <= 0.0 {
            return Err(RiskError::CalculationError(
                "Invalid stop loss price".to_string(),
            ));
        }

        let position_size = risk_amount / price_diff;
        Ok(position_size)
    }

    /// Calculate ATR-based stop loss
    pub fn calculate_atr_stop_loss(
        &self,
        _symbol: &str,
        entry_price: f64,
        atr: f64,
        multiplier: f64,
        is_long: bool,
    ) -> f64 {
        if is_long {
            entry_price - (atr * multiplier)
        } else {
            entry_price + (atr * multiplier)
        }
    }

    /// Calculate percentage-based stop loss
    pub fn calculate_percentage_stop_loss(
        &self,
        entry_price: f64,
        stop_percentage: f64,
        is_long: bool,
    ) -> f64 {
        if is_long {
            entry_price * (1.0 - stop_percentage / 100.0)
        } else {
            entry_price * (1.0 + stop_percentage / 100.0)
        }
    }

    /// Calculate trailing stop loss
    pub fn calculate_trailing_stop(
        &self,
        _entry_price: f64,
        current_price: f64,
        trail_percentage: f64,
        is_long: bool,
        previous_stop: Option<f64>,
    ) -> f64 {
        let trail_amount = current_price * (trail_percentage / 100.0);

        let new_stop = if is_long {
            current_price - trail_amount
        } else {
            current_price + trail_amount
        };

        match previous_stop {
            Some(prev) => {
                if is_long {
                    new_stop.max(prev) // Only move stop up for long positions
                } else {
                    new_stop.min(prev) // Only move stop down for short positions
                }
            }
            None => new_stop,
        }
    }

    /// Calculate risk-reward ratio
    pub fn calculate_risk_reward_ratio(
        &self,
        entry_price: f64,
        stop_loss: f64,
        take_profit: f64,
        is_long: bool,
    ) -> Result<f64, RiskError> {
        let risk = (entry_price - stop_loss).abs();
        let reward = if is_long {
            take_profit - entry_price
        } else {
            entry_price - take_profit
        };

        if risk <= 0.0 {
            return Err(RiskError::CalculationError(
                "Risk cannot be zero".to_string(),
            ));
        }

        Ok(reward / risk)
    }

    /// Update position and check risk limits
    pub fn update_position(&mut self, position: Position) -> Result<(), RiskError> {
        let _old_position = self.positions.get(&position.symbol);

        // Validate position size
        if position.size.abs() > self.parameters.max_position_size_pct {
            return Err(RiskError::InvalidPositionSize(position.size));
        }

        // Update position
        self.positions
            .insert(position.symbol.clone(), position.clone());

        // Check portfolio risk
        self.validate_portfolio_risk()?;

        // Update historical data for future calculations
        self.update_historical_data(&position);

        Ok(())
    }

    /// Calculate maximum drawdown protection
    pub fn calculate_max_drawdown(
        &mut self,
        current_portfolio_value: f64,
    ) -> Result<f64, RiskError> {
        if current_portfolio_value > self.high_water_mark {
            self.high_water_mark = current_portfolio_value;
        }

        let current_drawdown =
            ((self.high_water_mark - current_portfolio_value) / self.high_water_mark) * 100.0;

        if current_drawdown > self.parameters.max_drawdown_pct {
            return Err(RiskError::MaxDrawdownExceeded(current_drawdown));
        }

        Ok(current_drawdown)
    }

    /// Calculate portfolio correlation matrix
    pub fn calculate_correlation_matrix(&self) -> HashMap<(String, String), f64> {
        let mut correlations = HashMap::new();
        let symbols: Vec<_> = self.historical_returns.keys().collect();

        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                let symbol1 = symbols[i];
                let symbol2 = symbols[j];

                if let (Some(returns1), Some(returns2)) = (
                    self.historical_returns.get(symbol1),
                    self.historical_returns.get(symbol2),
                ) {
                    let correlation = self.calculate_correlation(returns1, returns2);
                    correlations.insert((symbol1.clone(), symbol2.clone()), correlation);
                }
            }
        }

        correlations
    }

    /// Check correlation risk
    pub fn validate_correlation_risk(&self) -> Result<(), RiskError> {
        let correlations = self.calculate_correlation_matrix();

        for ((_symbol1, _symbol2), correlation) in correlations {
            if correlation.abs() > self.parameters.max_correlation {
                return Err(RiskError::CorrelationLimitExceeded(correlation));
            }
        }

        Ok(())
    }

    /// Calculate Value at Risk (VaR)
    pub fn calculate_var(&self, confidence_level: f64, portfolio_value: f64) -> f64 {
        if self.portfolio_history.len() < 30 {
            return 0.0;
        }

        let mut returns: Vec<f64> = self
            .portfolio_history
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((1.0 - confidence_level) * returns.len() as f64) as usize;
        let var_return = returns.get(index).unwrap_or(&0.0);

        portfolio_value * var_return.abs()
    }

    /// Calculate Sharpe ratio
    pub fn calculate_sharpe_ratio(&self) -> f64 {
        if self.portfolio_history.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = self
            .portfolio_history
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        (mean_return * 252.0 - self.parameters.risk_free_rate) / (std_dev * (252.0_f64).sqrt())
    }

    /// Calculate Sortino ratio (using downside deviation)
    pub fn calculate_sortino_ratio(&self) -> f64 {
        if self.portfolio_history.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = self
            .portfolio_history
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

        if downside_returns.is_empty() {
            return f64::INFINITY;
        }

        let downside_variance =
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64;
        let downside_deviation = downside_variance.sqrt();

        if downside_deviation == 0.0 {
            return f64::INFINITY;
        }

        (mean_return * 252.0 - self.parameters.risk_free_rate)
            / (downside_deviation * (252.0_f64).sqrt())
    }

    /// Get current risk metrics
    pub fn get_risk_metrics(&self) -> RiskMetrics {
        let portfolio_value: f64 = self
            .positions
            .values()
            .map(|p| p.size * p.current_price)
            .sum();

        let total_margin_used: f64 = self.positions.values().map(|p| p.margin_used).sum();

        let unrealized_pnl: f64 = self.positions.values().map(|p| p.unrealized_pnl).sum();

        let current_drawdown = if self.high_water_mark > 0.0 {
            ((self.high_water_mark - portfolio_value) / self.high_water_mark) * 100.0
        } else {
            0.0
        };

        let var_95 = self.calculate_var(0.95, portfolio_value);
        let var_99 = self.calculate_var(0.99, portfolio_value);

        RiskMetrics {
            portfolio_value,
            total_margin_used,
            free_margin: portfolio_value - total_margin_used,
            unrealized_pnl,
            realized_pnl: 0.0, // Would be tracked separately
            max_drawdown: self.parameters.max_drawdown_pct,
            current_drawdown,
            var_95,
            var_99,
            sharpe_ratio: self.calculate_sharpe_ratio(),
            sortino_ratio: self.calculate_sortino_ratio(),
            max_leverage: self.parameters.max_leverage,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    // Helper methods

    fn validate_portfolio_risk(&self) -> Result<(), RiskError> {
        let total_risk: f64 = self
            .positions
            .values()
            .map(|p| (p.size * p.current_price) / (p.current_price - 0.0)) // Simplified risk calc
            .sum();

        let portfolio_value: f64 = self
            .positions
            .values()
            .map(|p| p.size * p.current_price)
            .sum();

        if portfolio_value > 0.0 {
            let risk_percentage = (total_risk / portfolio_value) * 100.0;
            if risk_percentage > self.parameters.max_portfolio_risk_pct {
                return Err(RiskError::CalculationError(format!(
                    "Portfolio risk {}% exceeds limit {}%",
                    risk_percentage, self.parameters.max_portfolio_risk_pct
                )));
            }
        }

        Ok(())
    }

    fn update_historical_data(&mut self, position: &Position) {
        // Update historical returns for correlation calculations
        let symbol = &position.symbol;
        let returns = self.historical_returns.entry(symbol.clone()).or_default();

        if let Some(last_price) = returns.last() {
            let return_rate = (position.current_price - last_price) / last_price;
            returns.push(return_rate);
        } else {
            returns.push(0.0); // First entry
        }

        // Keep only recent data for performance
        if returns.len() > self.parameters.kelly_lookback {
            returns.drain(0..returns.len() - self.parameters.kelly_lookback);
        }

        // Update portfolio history
        let portfolio_value: f64 = self
            .positions
            .values()
            .map(|p| p.size * p.current_price)
            .sum();

        self.portfolio_history.push(portfolio_value);
        if self.portfolio_history.len() > self.parameters.kelly_lookback {
            self.portfolio_history
                .drain(0..self.portfolio_history.len() - self.parameters.kelly_lookback);
        }
    }

    fn calculate_correlation(&self, returns1: &[f64], returns2: &[f64]) -> f64 {
        let min_len = returns1.len().min(returns2.len());
        if min_len < 2 {
            return 0.0;
        }

        let returns1 = &returns1[returns1.len() - min_len..];
        let returns2 = &returns2[returns2.len() - min_len..];

        let mean1 = returns1.iter().sum::<f64>() / min_len as f64;
        let mean2 = returns2.iter().sum::<f64>() / min_len as f64;

        let mut covariance = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for i in 0..min_len {
            let diff1 = returns1[i] - mean1;
            let diff2 = returns2[i] - mean2;
            covariance += diff1 * diff2;
            var1 += diff1 * diff1;
            var2 += diff2 * diff2;
        }

        let std1 = (var1 / min_len as f64).sqrt();
        let std2 = (var2 / min_len as f64).sqrt();

        if std1 == 0.0 || std2 == 0.0 {
            return 0.0;
        }

        (covariance / min_len as f64) / (std1 * std2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_position_sizing() {
        let risk_manager = RiskManager::new(RiskParameters::default());
        let position_size = risk_manager
            .calculate_kelly_position_size(
                "BTC", 0.6,     // 60% win rate
                100.0,   // Average win
                50.0,    // Average loss
                10000.0, // Portfolio value
            )
            .unwrap();

        assert!(position_size > 0.0);
        assert!(position_size <= 10000.0 * 0.02); // Max 2% position size
    }

    #[test]
    fn test_atr_stop_loss() {
        let risk_manager = RiskManager::new(RiskParameters::default());
        let stop_loss = risk_manager.calculate_atr_stop_loss(
            "BTC", 50000.0, // Entry price
            1000.0,  // ATR
            2.0,     // Multiplier
            true,    // Long position
        );

        assert_eq!(stop_loss, 48000.0); // 50000 - (1000 * 2)
    }

    #[test]
    fn test_risk_reward_ratio() {
        let risk_manager = RiskManager::new(RiskParameters::default());
        let ratio = risk_manager
            .calculate_risk_reward_ratio(
                100.0, // Entry
                95.0,  // Stop loss
                110.0, // Take profit
                true,  // Long
            )
            .unwrap();

        assert_eq!(ratio, 2.0); // 10 reward / 5 risk
    }
}
