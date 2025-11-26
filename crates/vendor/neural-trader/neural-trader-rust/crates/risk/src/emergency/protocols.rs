//! Emergency protocol handlers
//!
//! Automated responses to critical risk events:
//! - Position flattening
//! - Trading halt
//! - System shutdown
//! - Alert escalation

use crate::{Result, RiskError};
use crate::emergency::circuit_breakers::{CircuitBreaker, CircuitBreakerConfig};
use crate::types::{AlertLevel, EmergencyAction, Portfolio, Symbol};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info, warn};

/// Emergency protocol execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProtocolStatus {
    /// Monitoring, no action needed
    Normal,
    /// Warning level, increase monitoring
    Warning,
    /// Critical level, prepare for action
    Critical,
    /// Emergency level, executing protocols
    Emergency,
}

/// Emergency protocol executor
pub struct EmergencyProtocol {
    circuit_breaker: Arc<CircuitBreaker>,
    action_history: Arc<RwLock<Vec<EmergencyActionRecord>>>,
    status: Arc<RwLock<ProtocolStatus>>,
    config: EmergencyConfig,
}

/// Emergency protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyConfig {
    /// Auto-flatten positions on circuit breaker trip
    pub auto_flatten_on_trip: bool,
    /// Minimum position size to flatten (ignore small positions)
    pub min_flatten_size: f64,
    /// Enable automated emergency actions
    pub enable_auto_actions: bool,
    /// Alert escalation thresholds
    pub alert_escalation_enabled: bool,
}

impl Default for EmergencyConfig {
    fn default() -> Self {
        Self {
            auto_flatten_on_trip: true,
            min_flatten_size: 100.0,
            enable_auto_actions: true,
            alert_escalation_enabled: true,
        }
    }
}

impl EmergencyProtocol {
    /// Create new emergency protocol handler
    pub fn new(breaker_config: CircuitBreakerConfig, config: EmergencyConfig) -> Self {
        let circuit_breaker = Arc::new(CircuitBreaker::new(breaker_config));

        info!("Emergency protocols initialized");

        Self {
            circuit_breaker,
            action_history: Arc::new(RwLock::new(Vec::new())),
            status: Arc::new(RwLock::new(ProtocolStatus::Normal)),
            config,
        }
    }

    /// Execute emergency action
    pub fn execute_action(&self, action: EmergencyAction, portfolio: &Portfolio) -> Result<()> {
        if !self.config.enable_auto_actions {
            warn!("Auto actions disabled, not executing: {:?}", action);
            return Ok(());
        }

        match &action {
            EmergencyAction::HaltTrading => {
                self.halt_trading()?;
            }
            EmergencyAction::ClosePosition(symbol) => {
                self.close_position(portfolio, symbol)?;
            }
            EmergencyAction::CloseAllPositions => {
                self.close_all_positions(portfolio)?;
            }
            EmergencyAction::ReducePosition { symbol, percentage } => {
                self.reduce_position(portfolio, symbol, *percentage)?;
            }
            EmergencyAction::CircuitBreaker { duration_seconds } => {
                self.activate_circuit_breaker(*duration_seconds)?;
            }
            EmergencyAction::Alert { level, message } => {
                self.send_alert(*level, message)?;
            }
        }

        // Record action
        self.record_action(action);

        Ok(())
    }

    /// Halt all trading
    fn halt_trading(&self) -> Result<()> {
        error!("EMERGENCY: Trading halted");

        let breaker = &self.circuit_breaker;
        breaker.trip(crate::emergency::circuit_breakers::TriggerCondition::Manual)?;

        let mut status = self.status.write();
        *status = ProtocolStatus::Emergency;

        Ok(())
    }

    /// Close a single position
    fn close_position(&self, portfolio: &Portfolio, symbol: &Symbol) -> Result<()> {
        let position = portfolio.get_position(symbol).ok_or_else(|| {
            RiskError::PositionNotFound(symbol.to_string())
        })?;

        let size = position.exposure().abs();
        if size < self.config.min_flatten_size {
            info!("Position {} too small to flatten: ${:.2}", symbol, size);
            return Ok(());
        }

        warn!(
            "EMERGENCY: Closing position {} (${:.2})",
            symbol, size
        );

        // In production, this would execute actual market orders
        // Here we just log the action

        Ok(())
    }

    /// Close all positions
    fn close_all_positions(&self, portfolio: &Portfolio) -> Result<()> {
        error!("EMERGENCY: Closing all positions");

        let positions: Vec<Symbol> = portfolio
            .positions
            .values()
            .filter(|p| p.exposure().abs() >= self.config.min_flatten_size)
            .map(|p| p.symbol.clone())
            .collect();

        for symbol in positions {
            self.close_position(portfolio, &symbol)?;
        }

        Ok(())
    }

    /// Reduce position size by percentage
    fn reduce_position(
        &self,
        portfolio: &Portfolio,
        symbol: &Symbol,
        percentage: f64,
    ) -> Result<()> {
        let position = portfolio.get_position(symbol).ok_or_else(|| {
            RiskError::PositionNotFound(symbol.to_string())
        })?;

        let size = position.exposure().abs();
        let reduce_amount = size * percentage;

        if reduce_amount < self.config.min_flatten_size {
            info!("Reduction amount too small: ${:.2}", reduce_amount);
            return Ok(());
        }

        warn!(
            "EMERGENCY: Reducing position {} by {:.1}% (${:.2})",
            symbol,
            percentage * 100.0,
            reduce_amount
        );

        // In production, would execute partial close orders

        Ok(())
    }

    /// Activate circuit breaker
    fn activate_circuit_breaker(&self, duration_seconds: u64) -> Result<()> {
        error!(
            "EMERGENCY: Activating circuit breaker for {} seconds",
            duration_seconds
        );

        let breaker = &self.circuit_breaker;
        breaker.trip(crate::emergency::circuit_breakers::TriggerCondition::Manual)?;

        // In production, would schedule auto-reset after duration

        let mut status = self.status.write();
        *status = ProtocolStatus::Emergency;

        Ok(())
    }

    /// Send alert
    fn send_alert(&self, level: AlertLevel, message: &str) -> Result<()> {
        match level {
            AlertLevel::Info => info!("ALERT: {}", message),
            AlertLevel::Warning => warn!("ALERT: {}", message),
            AlertLevel::Critical => error!("ALERT (CRITICAL): {}", message),
            AlertLevel::Emergency => error!("ALERT (EMERGENCY): {}", message),
        }

        // In production, would send email/SMS/PagerDuty alerts

        Ok(())
    }

    /// Check if emergency response is needed
    pub fn assess_situation(
        &self,
        portfolio: &Portfolio,
        peak_value: f64,
    ) -> Result<Vec<EmergencyAction>> {
        let mut recommended_actions = Vec::new();

        // Check circuit breaker status
        if !self.circuit_breaker.is_trading_allowed() {
            if self.config.auto_flatten_on_trip {
                recommended_actions.push(EmergencyAction::CloseAllPositions);
            }
            return Ok(recommended_actions);
        }

        // Check drawdown
        let current_value = portfolio.total_value();
        if peak_value > 0.0 {
            let drawdown = (peak_value - current_value) / peak_value;

            if drawdown > 0.15 {
                warn!("Severe drawdown detected: {:.2}%", drawdown * 100.0);
                recommended_actions.push(EmergencyAction::Alert {
                    level: AlertLevel::Critical,
                    message: format!("Severe drawdown: {:.2}%", drawdown * 100.0),
                });

                if drawdown > 0.20 {
                    // Catastrophic drawdown
                    recommended_actions.push(EmergencyAction::CircuitBreaker {
                        duration_seconds: 1800, // 30 minutes
                    });
                    recommended_actions.push(EmergencyAction::CloseAllPositions);
                }
            }
        }

        // Check unrealized P&L
        let unrealized_pnl = portfolio.total_unrealized_pnl();
        if unrealized_pnl < -0.10 * current_value {
            // More than 10% unrealized loss
            recommended_actions.push(EmergencyAction::Alert {
                level: AlertLevel::Warning,
                message: format!("Large unrealized loss: ${:.2}", unrealized_pnl),
            });
        }

        // Check concentration risk
        let concentration = portfolio.concentration();
        if concentration > 0.5 {
            // More than 50% in single position
            recommended_actions.push(EmergencyAction::Alert {
                level: AlertLevel::Warning,
                message: format!("High concentration: {:.1}%", concentration * 100.0),
            });
        }

        Ok(recommended_actions)
    }

    /// Get current protocol status
    pub fn get_status(&self) -> ProtocolStatus {
        self.status.read().clone()
    }

    /// Get circuit breaker reference
    pub fn get_circuit_breaker(&self) -> Arc<CircuitBreaker> {
        Arc::clone(&self.circuit_breaker)
    }

    /// Get action history
    pub fn get_action_history(&self, limit: usize) -> Vec<EmergencyActionRecord> {
        let history = self.action_history.read();
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Record executed action
    fn record_action(&self, action: EmergencyAction) {
        let record = EmergencyActionRecord {
            timestamp: Utc::now(),
            action,
            status: self.get_status(),
        };

        let mut history = self.action_history.write();
        history.push(record);

        // Keep last 1000 actions
        if history.len() > 1000 {
            history.drain(0..100);
        }
    }

    /// Reset to normal status
    pub fn reset_status(&self) -> Result<()> {
        let mut status = self.status.write();
        *status = ProtocolStatus::Normal;

        info!("Emergency protocol status reset to Normal");

        Ok(())
    }
}

/// Emergency action execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyActionRecord {
    pub timestamp: DateTime<Utc>,
    pub action: EmergencyAction,
    pub status: ProtocolStatus,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PositionSide};
    use rust_decimal_macros::dec;

    fn create_test_portfolio() -> Portfolio {
        let mut portfolio = Portfolio::new(dec!(100000));

        portfolio.update_position(Position {
            symbol: Symbol::new("AAPL"),
            quantity: dec!(100),
            avg_entry_price: dec!(150.0),
            current_price: dec!(150.0),
            market_value: dec!(15000),
            unrealized_pnl: dec!(0),
            unrealized_pnl_percent: dec!(0),
            side: PositionSide::Long,
            opened_at: Utc::now(),
        });

        portfolio
    }

    #[test]
    fn test_protocol_creation() {
        let breaker_config = CircuitBreakerConfig::default();
        let _config = EmergencyConfig::default();
        let protocol = EmergencyProtocol::new(breaker_config, config);

        assert!(matches!(protocol.get_status(), ProtocolStatus::Normal));
    }

    #[test]
    fn test_halt_trading() {
        let breaker_config = CircuitBreakerConfig::default();
        let _config = EmergencyConfig::default();
        let protocol = EmergencyProtocol::new(breaker_config, config);

        let portfolio = create_test_portfolio();

        protocol
            .execute_action(EmergencyAction::HaltTrading, &portfolio)
            .unwrap();

        assert!(matches!(protocol.get_status(), ProtocolStatus::Emergency));
        assert!(!protocol.circuit_breaker.is_trading_allowed());
    }

    #[test]
    fn test_close_position() {
        let breaker_config = CircuitBreakerConfig::default();
        let _config = EmergencyConfig::default();
        let protocol = EmergencyProtocol::new(breaker_config, config);

        let portfolio = create_test_portfolio();

        let result = protocol.execute_action(
            EmergencyAction::ClosePosition(Symbol::new("AAPL")),
            &portfolio,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_assess_situation_normal() {
        let breaker_config = CircuitBreakerConfig::default();
        let _config = EmergencyConfig::default();
        let protocol = EmergencyProtocol::new(breaker_config, config);

        let portfolio = create_test_portfolio();
        let peak_value = 110000.0;

        let actions = protocol.assess_situation(&portfolio, peak_value).unwrap();

        // Should be empty or minimal with normal portfolio
        assert!(actions.len() <= 1);
    }

    #[test]
    fn test_assess_situation_severe_drawdown() {
        let breaker_config = CircuitBreakerConfig::default();
        let _config = EmergencyConfig::default();
        let protocol = EmergencyProtocol::new(breaker_config, config);

        let mut portfolio = create_test_portfolio();
        portfolio.cash = dec!(50000); // Severe loss

        let peak_value = 120000.0;

        let actions = protocol.assess_situation(&portfolio, peak_value).unwrap();

        // Should recommend emergency actions
        assert!(!actions.is_empty());
    }

    #[test]
    fn test_action_history() {
        let breaker_config = CircuitBreakerConfig::default();
        let _config = EmergencyConfig::default();
        let protocol = EmergencyProtocol::new(breaker_config, config);

        let portfolio = create_test_portfolio();

        // Execute some actions
        let _ = protocol.execute_action(
            EmergencyAction::Alert {
                level: AlertLevel::Warning,
                message: "Test alert".to_string(),
            },
            &portfolio,
        );

        let history = protocol.get_action_history(10);
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_auto_actions_disabled() {
        let breaker_config = CircuitBreakerConfig::default();
        let mut config = EmergencyConfig::default();
        config.enable_auto_actions = false;

        let protocol = EmergencyProtocol::new(breaker_config, config);
        let portfolio = create_test_portfolio();

        // Should not execute (but not error)
        let result = protocol.execute_action(EmergencyAction::HaltTrading, &portfolio);

        assert!(result.is_ok());
        // Status should remain normal since action wasn't executed
        assert!(matches!(protocol.get_status(), ProtocolStatus::Normal));
    }
}
