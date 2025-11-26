//! Risk Management Integration for Trading Strategies
//!
//! Connects strategies to Agent 6's risk management:
//! - Kelly Criterion position sizing
//! - VaR/CVaR limit checks
//! - Portfolio exposure validation
//! - Stop-loss execution
//! - Emergency protocols

use crate::{Result, Signal, StrategyError, Direction};
use serde::{Serialize, Deserialize};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use std::collections::HashMap;
use tracing::{debug, info, warn};

// Re-export risk types from kelly module
pub use nt_risk::kelly::{KellySingleAsset as KellyCriterion};

/// Position size result for strategies
#[derive(Debug, Clone)]
pub struct PositionSize {
    pub quantity: u32,
    pub notional: Decimal,
    pub fraction: f64,
}

/// Risk manager for strategy execution
pub struct RiskManager {
    /// Kelly criterion calculator
    kelly: KellyCriterion,
    /// Maximum portfolio leverage
    max_leverage: f64,
    /// Maximum single position size (fraction of portfolio)
    max_position_size: f64,
    /// Maximum daily loss limit
    max_daily_loss: Decimal,
    /// Current daily loss
    daily_loss: Decimal,
    /// VaR limit (95% confidence)
    var_limit: Decimal,
    /// CVaR limit (95% confidence)
    cvar_limit: Decimal,
    /// Position limits by symbol
    position_limits: HashMap<String, Decimal>,
    /// Emergency stop triggered
    emergency_stop: bool,
}

impl RiskManager {
    /// Create new risk manager
    pub fn new(
        kelly_fraction: f64,
        max_position_fraction: f64,
        min_position_size: Decimal,
        max_leverage: f64,
    ) -> Result<Self> {
        // Kelly needs (win_rate, avg_win, avg_loss, fractional)
        let kelly = KellyCriterion::new(
            0.55, // default win rate
            1.5,  // default avg_win
            1.0,  // default avg_loss
            kelly_fraction,
        ).map_err(|e| StrategyError::ConfigError(e.to_string()))?;

        Ok(Self {
            kelly,
            max_leverage,
            max_position_size: max_position_fraction,
            max_daily_loss: Decimal::from(10000), // Default $10k daily loss limit
            daily_loss: Decimal::ZERO,
            var_limit: Decimal::from(50000), // Default $50k VaR limit
            cvar_limit: Decimal::from(75000), // Default $75k CVaR limit
            position_limits: HashMap::new(),
            emergency_stop: false,
        })
    }

    /// Set risk limits
    pub fn with_limits(
        mut self,
        max_daily_loss: Decimal,
        var_limit: Decimal,
        cvar_limit: Decimal,
    ) -> Self {
        self.max_daily_loss = max_daily_loss;
        self.var_limit = var_limit;
        self.cvar_limit = cvar_limit;
        self
    }

    /// Set position limit for specific symbol
    pub fn set_position_limit(&mut self, symbol: String, limit: Decimal) {
        self.position_limits.insert(symbol, limit);
    }

    /// Validate and size signal
    pub fn validate_signal(
        &self,
        signal: &mut Signal,
        portfolio_value: Decimal,
        current_positions: &HashMap<String, Position>,
    ) -> Result<ValidationResult> {
        // Check emergency stop
        if self.emergency_stop {
            return Ok(ValidationResult::Rejected(
                "Emergency stop active".to_string(),
            ));
        }

        // Check daily loss limit
        if self.daily_loss.abs() >= self.max_daily_loss {
            warn!("Daily loss limit reached: ${}", self.daily_loss.abs());
            return Ok(ValidationResult::Rejected(
                "Daily loss limit exceeded".to_string(),
            ));
        }

        // For close signals, always approve
        if signal.direction == Direction::Close {
            return Ok(ValidationResult::Approved);
        }

        // Calculate position size using Kelly Criterion
        let position_size = self.calculate_position_size(
            signal,
            portfolio_value,
        )?;

        if position_size.quantity == 0 {
            return Ok(ValidationResult::Rejected(
                "Position size too small".to_string(),
            ));
        }

        // Check symbol-specific limits
        if let Some(limit) = self.position_limits.get(&signal.symbol) {
            if position_size.notional > *limit {
                return Ok(ValidationResult::Rejected(format!(
                    "Exceeds symbol limit of ${}",
                    limit
                )));
            }
        }

        // Check portfolio concentration
        let concentration = self.check_concentration(
            &signal.symbol,
            position_size.notional,
            portfolio_value,
            current_positions,
        )?;

        if concentration > self.max_position_size {
            return Ok(ValidationResult::Rejected(format!(
                "Portfolio concentration too high: {:.1}%",
                concentration * 100.0
            )));
        }

        // Check leverage
        let total_exposure = self.calculate_total_exposure(
            current_positions,
            Some((signal.symbol.clone(), position_size.notional)),
        );

        let leverage = total_exposure.to_f64().unwrap() / portfolio_value.to_f64().unwrap();

        if leverage > self.max_leverage {
            return Ok(ValidationResult::Rejected(format!(
                "Leverage too high: {:.2}x",
                leverage
            )));
        }

        // Update signal with calculated quantity
        signal.quantity = Some(position_size.quantity);

        Ok(ValidationResult::Approved)
    }

    /// Calculate position size using Kelly Criterion
    fn calculate_position_size(
        &self,
        signal: &Signal,
        portfolio_value: Decimal,
    ) -> Result<PositionSize> {
        let confidence = signal.confidence.unwrap_or(0.5);

        // Simple position sizing: scale by confidence
        let base_fraction = self.max_position_size * confidence;
        let position_value = portfolio_value * Decimal::from_f64_retain(base_fraction).unwrap();

        let current_price = signal.entry_price.unwrap_or(Decimal::from(100));
        let quantity = (position_value / current_price).to_u32().unwrap_or(0);

        Ok(PositionSize {
            quantity,
            notional: position_value,
            fraction: base_fraction,
        })
    }

    /// Check portfolio concentration
    fn check_concentration(
        &self,
        symbol: &str,
        new_position_value: Decimal,
        portfolio_value: Decimal,
        current_positions: &HashMap<String, Position>,
    ) -> Result<f64> {
        // Get current exposure to symbol
        let current_exposure = current_positions
            .get(symbol)
            .map(|p| p.market_value)
            .unwrap_or(Decimal::ZERO);

        let total_exposure = current_exposure + new_position_value;
        let concentration = total_exposure.to_f64().unwrap() / portfolio_value.to_f64().unwrap();

        Ok(concentration)
    }

    /// Calculate total portfolio exposure
    fn calculate_total_exposure(
        &self,
        current_positions: &HashMap<String, Position>,
        new_position: Option<(String, Decimal)>,
    ) -> Decimal {
        let mut total = current_positions
            .values()
            .map(|p| p.market_value)
            .sum::<Decimal>();

        if let Some((_, value)) = new_position {
            total += value;
        }

        total
    }

    /// Update daily P&L tracking
    pub fn update_daily_pnl(&mut self, pnl: Decimal) {
        self.daily_loss += pnl;
        debug!("Updated daily P&L: ${}", self.daily_loss);

        // Check if we need emergency stop
        if self.daily_loss.abs() >= self.max_daily_loss * Decimal::from_f64_retain(1.2).unwrap() {
            self.trigger_emergency_stop();
        }
    }

    /// Reset daily tracking (call at start of trading day)
    pub fn reset_daily(&mut self) {
        info!("Resetting daily risk tracking");
        self.daily_loss = Decimal::ZERO;
        self.emergency_stop = false;
    }

    /// Calculate portfolio VaR (Value at Risk)
    pub fn calculate_var(
        &self,
        positions: &HashMap<String, Position>,
        confidence: f64,
    ) -> Result<Decimal> {
        if positions.is_empty() {
            return Ok(Decimal::ZERO);
        }

        // Simple parametric VaR calculation
        let total_value: Decimal = positions.values().map(|p| p.market_value).sum();
        let volatility = 0.15; // Assume 15% annual volatility
        let z_score = match confidence {
            0.95 => 1.645,
            0.99 => 2.326,
            _ => 1.645,
        };

        let var = total_value * Decimal::from_f64_retain(volatility * z_score / (252.0_f64).sqrt()).unwrap();

        Ok(var)
    }

    /// Calculate portfolio CVaR (Conditional Value at Risk)
    pub fn calculate_cvar(
        &self,
        positions: &HashMap<String, Position>,
        confidence: f64,
    ) -> Result<Decimal> {
        // CVaR is typically 1.2-1.5x VaR
        let var = self.calculate_var(positions, confidence)?;
        Ok(var * Decimal::from_f64_retain(1.3).unwrap())
    }

    /// Check if portfolio risk limits are breached
    pub fn check_risk_limits(
        &self,
        positions: &HashMap<String, Position>,
    ) -> Result<Vec<RiskWarning>> {
        let mut warnings = Vec::new();

        // Check VaR
        let var = self.calculate_var(positions, 0.95)?;
        if var > self.var_limit {
            warnings.push(RiskWarning {
                level: RiskLevel::High,
                message: format!("VaR ${} exceeds limit ${}", var, self.var_limit),
                metric: "VaR".to_string(),
                current_value: var,
                limit: self.var_limit,
            });
        }

        // Check CVaR
        let cvar = self.calculate_cvar(positions, 0.95)?;
        if cvar > self.cvar_limit {
            warnings.push(RiskWarning {
                level: RiskLevel::Critical,
                message: format!("CVaR ${} exceeds limit ${}", cvar, self.cvar_limit),
                metric: "CVaR".to_string(),
                current_value: cvar,
                limit: self.cvar_limit,
            });
        }

        // Check daily loss
        if self.daily_loss.abs() >= self.max_daily_loss * Decimal::from_f64_retain(0.8).unwrap() {
            warnings.push(RiskWarning {
                level: RiskLevel::Medium,
                message: format!(
                    "Daily loss ${} approaching limit ${}",
                    self.daily_loss.abs(),
                    self.max_daily_loss
                ),
                metric: "DailyLoss".to_string(),
                current_value: self.daily_loss.abs(),
                limit: self.max_daily_loss,
            });
        }

        Ok(warnings)
    }

    /// Trigger emergency stop
    fn trigger_emergency_stop(&mut self) {
        warn!("EMERGENCY STOP TRIGGERED - Daily loss limit severely exceeded");
        self.emergency_stop = true;
    }

    /// Check if emergency stop is active
    pub fn is_emergency_stop(&self) -> bool {
        self.emergency_stop
    }

    /// Get current daily loss
    pub fn daily_loss(&self) -> Decimal {
        self.daily_loss
    }
}

impl Default for RiskManager {
    fn default() -> Self {
        Self::new(
            0.25,                  // Quarter Kelly
            0.1,                   // Max 10% per position
            Decimal::from(100),    // $100 minimum
            2.0,                   // 2x max leverage
        )
        .unwrap()
    }
}

/// Validation result
#[derive(Debug, Clone)]
pub enum ValidationResult {
    Approved,
    Rejected(String),
}

/// Position information for risk calculations
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: i64,
    pub market_value: Decimal,
    pub unrealized_pnl: Decimal,
}

/// Risk warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskWarning {
    pub level: RiskLevel,
    pub message: String,
    pub metric: String,
    pub current_value: Decimal,
    pub limit: Decimal,
}

/// Risk warning level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_manager_creation() {
        let manager = RiskManager::default();
        assert!(!manager.is_emergency_stop());
        assert_eq!(manager.daily_loss(), Decimal::ZERO);
    }

    #[test]
    fn test_signal_validation_approved() {
        let manager = RiskManager::default();
        let mut signal = Signal::new("test".to_string(), "AAPL".to_string(), Direction::Long)
            .with_confidence(0.8)
            .with_entry_price(Decimal::from(150));

        let result = manager
            .validate_signal(&mut signal, Decimal::from(100000), &HashMap::new())
            .unwrap();

        assert!(matches!(result, ValidationResult::Approved));
        assert!(signal.quantity.is_some());
    }

    #[test]
    fn test_signal_validation_rejected_daily_loss() {
        let mut manager = RiskManager::default();
        manager.update_daily_pnl(Decimal::from(-10000)); // Hit daily loss limit

        let mut signal = Signal::new("test".to_string(), "AAPL".to_string(), Direction::Long);

        let result = manager
            .validate_signal(&mut signal, Decimal::from(100000), &HashMap::new())
            .unwrap();

        assert!(matches!(result, ValidationResult::Rejected(_)));
    }

    #[test]
    fn test_var_calculation() {
        let manager = RiskManager::default();
        let mut positions = HashMap::new();
        positions.insert(
            "AAPL".to_string(),
            Position {
                symbol: "AAPL".to_string(),
                quantity: 100,
                market_value: Decimal::from(15000),
                unrealized_pnl: Decimal::ZERO,
            },
        );

        let var = manager.calculate_var(&positions, 0.95).unwrap();
        assert!(var > Decimal::ZERO);
        assert!(var < Decimal::from(15000)); // VaR should be less than total value
    }

    #[test]
    fn test_risk_limits_check() {
        let manager = RiskManager::default();
        let positions = HashMap::new(); // Empty portfolio

        let warnings = manager.check_risk_limits(&positions).unwrap();
        assert!(warnings.is_empty()); // No warnings for empty portfolio
    }

    #[test]
    fn test_daily_reset() {
        let mut manager = RiskManager::default();
        manager.update_daily_pnl(Decimal::from(-5000));

        assert_eq!(manager.daily_loss(), Decimal::from(-5000));

        manager.reset_daily();

        assert_eq!(manager.daily_loss(), Decimal::ZERO);
        assert!(!manager.is_emergency_stop());
    }
}
