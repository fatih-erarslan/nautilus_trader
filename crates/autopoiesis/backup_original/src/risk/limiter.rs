//! Risk limiting implementation

use crate::prelude::*;
use crate::models::{Order, Position};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use num_traits::ToPrimitive;
use std::collections::HashMap;

/// Risk limiter for enforcing risk constraints
#[derive(Debug, Clone)]
pub struct RiskLimiter {
    /// Limiter configuration
    config: RiskLimiterConfig,
    
    /// Current risk state
    risk_state: RiskState,
    
    /// Violation history
    violation_history: Vec<RiskViolation>,
}

#[derive(Debug, Clone)]
pub struct RiskLimiterConfig {
    /// Maximum position size per symbol
    pub max_position_size: Decimal,
    
    /// Maximum total portfolio value
    pub max_portfolio_value: Decimal,
    
    /// Maximum daily loss
    pub max_daily_loss: Decimal,
    
    /// Maximum leverage
    pub max_leverage: f64,
    
    /// Concentration limits
    pub concentration_limits: ConcentrationLimits,
    
    /// Drawdown limits
    pub drawdown_limits: DrawdownLimits,
}

#[derive(Debug, Clone)]
pub struct ConcentrationLimits {
    /// Maximum percentage in single position
    pub max_single_position_pct: f64,
    
    /// Maximum percentage in single sector
    pub max_sector_pct: f64,
    
    /// Maximum correlation between positions
    pub max_correlation: f64,
}

#[derive(Debug, Clone)]
pub struct DrawdownLimits {
    /// Maximum total drawdown
    pub max_drawdown_pct: f64,
    
    /// Maximum daily drawdown
    pub max_daily_drawdown_pct: f64,
    
    /// Stop trading threshold
    pub stop_trading_threshold_pct: f64,
}

#[derive(Debug, Clone, Default)]
struct RiskState {
    current_positions: HashMap<String, Decimal>,
    daily_pnl: Decimal,
    current_drawdown: f64,
    portfolio_value: Decimal,
    leverage_ratio: f64,
    last_reset: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct RiskViolation {
    timestamp: DateTime<Utc>,
    violation_type: ViolationType,
    severity: ViolationSeverity,
    description: String,
    action_taken: ActionTaken,
}

#[derive(Debug, Clone)]
enum ViolationType {
    PositionSize,
    PortfolioValue,
    DailyLoss,
    Leverage,
    Concentration,
    Drawdown,
    Correlation,
}

#[derive(Debug, Clone)]
enum ViolationSeverity {
    Warning,
    Breach,
    Critical,
}

#[derive(Debug, Clone)]
enum ActionTaken {
    Warning,
    OrderBlocked,
    PositionReduced,
    TradingStopped,
}

#[derive(Debug, Clone)]
pub struct RiskCheckResult {
    pub approved: bool,
    pub violations: Vec<String>,
    pub warnings: Vec<String>,
    pub adjusted_order: Option<Order>,
    pub action_required: Option<String>,
}

impl Default for RiskLimiterConfig {
    fn default() -> Self {
        Self {
            max_position_size: Decimal::from(100000),
            max_portfolio_value: Decimal::from(1000000),
            max_daily_loss: Decimal::from(50000),
            max_leverage: 3.0,
            concentration_limits: ConcentrationLimits {
                max_single_position_pct: 0.20,
                max_sector_pct: 0.30,
                max_correlation: 0.80,
            },
            drawdown_limits: DrawdownLimits {
                max_drawdown_pct: 0.15,
                max_daily_drawdown_pct: 0.05,
                stop_trading_threshold_pct: 0.10,
            },
        }
    }
}

impl RiskLimiter {
    /// Create a new risk limiter
    pub fn new(config: RiskLimiterConfig) -> Self {
        Self {
            config,
            risk_state: RiskState::default(),
            violation_history: Vec::new(),
        }
    }

    /// Check if an order is within risk limits
    pub async fn check_order(&mut self, order: &Order, current_positions: &[Position]) -> Result<RiskCheckResult> {
        // Update risk state
        self.update_risk_state(current_positions).await?;

        let mut result = RiskCheckResult {
            approved: true,
            violations: Vec::new(),
            warnings: Vec::new(),
            adjusted_order: None,
            action_required: None,
        };

        // Check position size limit
        self.check_position_size_limit(order, &mut result)?;

        // Check portfolio value limit
        self.check_portfolio_value_limit(order, &mut result)?;

        // Check daily loss limit
        self.check_daily_loss_limit(order, &mut result)?;

        // Check leverage limit
        self.check_leverage_limit(order, &mut result)?;

        // Check concentration limits
        self.check_concentration_limits(order, current_positions, &mut result)?;

        // Check drawdown limits
        self.check_drawdown_limits(order, &mut result)?;

        // Log violations
        for violation in &result.violations {
            self.log_violation(ViolationType::PositionSize, ViolationSeverity::Breach, violation.clone(), ActionTaken::OrderBlocked);
        }

        Ok(result)
    }

    /// Update position after trade execution
    pub async fn update_position(&mut self, symbol: &str, quantity_change: Decimal, price: Decimal) -> Result<()> {
        let current_position = self.risk_state.current_positions.entry(symbol.to_string()).or_insert(Decimal::ZERO);
        *current_position += quantity_change;

        // Update portfolio value
        self.risk_state.portfolio_value += quantity_change * price;

        Ok(())
    }

    /// Force risk state reset (e.g., at start of new trading day)
    pub async fn reset_daily_limits(&mut self) {
        self.risk_state.daily_pnl = Decimal::ZERO;
        self.risk_state.last_reset = Utc::now();
    }

    /// Get current risk status
    pub async fn get_risk_status(&self) -> RiskStatus {
        RiskStatus {
            total_portfolio_value: self.risk_state.portfolio_value,
            current_leverage: self.risk_state.leverage_ratio,
            daily_pnl: self.risk_state.daily_pnl,
            current_drawdown: self.risk_state.current_drawdown,
            position_count: self.risk_state.current_positions.len(),
            violations_today: self.violation_history.iter()
                .filter(|v| v.timestamp.date_naive() == Utc::now().date_naive())
                .count(),
            trading_allowed: self.is_trading_allowed(),
        }
    }

    /// Check if trading is currently allowed
    pub fn is_trading_allowed(&self) -> bool {
        // Stop trading if drawdown exceeds threshold
        if self.risk_state.current_drawdown > self.config.drawdown_limits.stop_trading_threshold_pct {
            return false;
        }

        // Stop trading if daily loss exceeds limit
        if self.risk_state.daily_pnl < -self.config.max_daily_loss {
            return false;
        }

        true
    }

    async fn update_risk_state(&mut self, current_positions: &[Position]) -> Result<()> {
        // Update current positions
        self.risk_state.current_positions.clear();
        for position in current_positions {
            self.risk_state.current_positions.insert(
                position.symbol.clone(),
                position.quantity,
            );
        }

        // Calculate portfolio value
        self.risk_state.portfolio_value = current_positions.iter()
            .map(|p| p.quantity * p.mark_price)
            .sum();

        // Update daily PnL
        let daily_pnl: Decimal = current_positions.iter()
            .map(|p| p.unrealized_pnl + p.realized_pnl)
            .sum();
        self.risk_state.daily_pnl = daily_pnl;

        // Calculate current drawdown (simplified)
        let peak_value = self.risk_state.portfolio_value * Decimal::from_f64_retain(1.2).unwrap_or_default(); // Assume 20% higher peak
        self.risk_state.current_drawdown = if peak_value > Decimal::ZERO {
            ((peak_value - self.risk_state.portfolio_value) / peak_value).to_f64().unwrap_or(0.0)
        } else {
            0.0
        };

        Ok(())
    }

    fn check_position_size_limit(&self, order: &Order, result: &mut RiskCheckResult) -> Result<()> {
        let current_position = self.risk_state.current_positions
            .get(&order.symbol)
            .copied()
            .unwrap_or(Decimal::ZERO);

        let new_position = current_position + order.quantity;

        if new_position.abs() > self.config.max_position_size {
            result.approved = false;
            result.violations.push(format!(
                "Position size limit exceeded for {}: {} > {}",
                order.symbol, new_position, self.config.max_position_size
            ));
            
            // Suggest adjusted order size
            let max_additional = self.config.max_position_size - current_position.abs();
            if max_additional > Decimal::ZERO {
                let mut adjusted_order = order.clone();
                adjusted_order.quantity = max_additional;
                result.adjusted_order = Some(adjusted_order);
                result.warnings.push(format!(
                    "Order size reduced from {} to {} to comply with position limits",
                    order.quantity, max_additional
                ));
            }
        }

        Ok(())
    }

    fn check_portfolio_value_limit(&self, order: &Order, result: &mut RiskCheckResult) -> Result<()> {
        let order_value = order.quantity * order.price.unwrap_or(Decimal::from(50000));
        let new_portfolio_value = self.risk_state.portfolio_value + order_value;

        if new_portfolio_value > self.config.max_portfolio_value {
            result.approved = false;
            result.violations.push(format!(
                "Portfolio value limit exceeded: {} > {}",
                new_portfolio_value, self.config.max_portfolio_value
            ));
        }

        Ok(())
    }

    fn check_daily_loss_limit(&self, _order: &Order, result: &mut RiskCheckResult) -> Result<()> {
        if self.risk_state.daily_pnl < -self.config.max_daily_loss {
            result.approved = false;
            result.violations.push(format!(
                "Daily loss limit exceeded: {} < -{}",
                self.risk_state.daily_pnl, self.config.max_daily_loss
            ));
            result.action_required = Some("Trading suspended due to daily loss limit".to_string());
        }

        Ok(())
    }

    fn check_leverage_limit(&self, order: &Order, result: &mut RiskCheckResult) -> Result<()> {
        let order_value = order.quantity * order.price.unwrap_or(Decimal::from(50000));
        let new_portfolio_value = self.risk_state.portfolio_value + order_value;
        
        // Simplified leverage calculation
        let estimated_equity = new_portfolio_value / Decimal::from(2); // Assume 50% equity
        let leverage = if estimated_equity > Decimal::ZERO {
            (new_portfolio_value / estimated_equity).to_f64().unwrap_or(0.0)
        } else {
            0.0
        };

        if leverage > self.config.max_leverage {
            result.approved = false;
            result.violations.push(format!(
                "Leverage limit exceeded: {:.2}x > {:.2}x",
                leverage, self.config.max_leverage
            ));
        }

        Ok(())
    }

    fn check_concentration_limits(&self, order: &Order, current_positions: &[Position], result: &mut RiskCheckResult) -> Result<()> {
        let total_portfolio_value = self.risk_state.portfolio_value;
        if total_portfolio_value <= Decimal::ZERO {
            return Ok(());
        }

        // Check single position concentration
        let current_position_value = current_positions.iter()
            .find(|p| p.symbol == order.symbol)
            .map(|p| p.quantity * p.mark_price)
            .unwrap_or(Decimal::ZERO);

        let order_value = order.quantity * order.price.unwrap_or(Decimal::from(50000));
        let new_position_value = current_position_value + order_value;
        let concentration_pct = (new_position_value / total_portfolio_value).to_f64().unwrap_or(0.0);

        if concentration_pct > self.config.concentration_limits.max_single_position_pct {
            result.approved = false;
            result.violations.push(format!(
                "Single position concentration limit exceeded for {}: {:.2}% > {:.2}%",
                order.symbol, concentration_pct * 100.0, 
                self.config.concentration_limits.max_single_position_pct * 100.0
            ));
        }

        Ok(())
    }

    fn check_drawdown_limits(&self, _order: &Order, result: &mut RiskCheckResult) -> Result<()> {
        if self.risk_state.current_drawdown > self.config.drawdown_limits.max_drawdown_pct {
            result.approved = false;
            result.violations.push(format!(
                "Drawdown limit exceeded: {:.2}% > {:.2}%",
                self.risk_state.current_drawdown * 100.0,
                self.config.drawdown_limits.max_drawdown_pct * 100.0
            ));
            result.action_required = Some("Trading suspended due to drawdown limit".to_string());
        }

        Ok(())
    }

    fn log_violation(&mut self, violation_type: ViolationType, severity: ViolationSeverity, description: String, action: ActionTaken) {
        let violation = RiskViolation {
            timestamp: Utc::now(),
            violation_type,
            severity,
            description,
            action_taken: action,
        };

        self.violation_history.push(violation);

        // Keep only recent violations (last 30 days)
        let cutoff = Utc::now() - chrono::Duration::days(30);
        self.violation_history.retain(|v| v.timestamp > cutoff);
    }

    /// Get violation history
    pub fn get_violation_history(&self, days: u32) -> Vec<&RiskViolation> {
        let cutoff = Utc::now() - chrono::Duration::days(days as i64);
        self.violation_history.iter()
            .filter(|v| v.timestamp > cutoff)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct RiskStatus {
    pub total_portfolio_value: Decimal,
    pub current_leverage: f64,
    pub daily_pnl: Decimal,
    pub current_drawdown: f64,
    pub position_count: usize,
    pub violations_today: usize,
    pub trading_allowed: bool,
}