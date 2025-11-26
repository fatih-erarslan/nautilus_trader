//! Risk limit enforcement engine
//!
//! Real-time validation and blocking of trades that would breach limits

use crate::Result;
use crate::limits::rules::{RiskLimitRules, RiskMetrics};
use crate::types::{AlertLevel, EmergencyAction, Portfolio, Position, Symbol};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

/// Limit enforcement decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementDecision {
    pub allowed: bool,
    pub reason: Option<String>,
    pub breached_limits: Vec<String>,
    pub alert_level: AlertLevel,
    pub recommended_actions: Vec<EmergencyAction>,
}

impl EnforcementDecision {
    /// Create allowed decision
    pub fn allow() -> Self {
        Self {
            allowed: true,
            reason: None,
            breached_limits: Vec::new(),
            alert_level: AlertLevel::Info,
            recommended_actions: Vec::new(),
        }
    }

    /// Create blocked decision
    pub fn block(reason: impl Into<String>, breached_limits: Vec<String>) -> Self {
        Self {
            allowed: false,
            reason: Some(reason.into()),
            breached_limits,
            alert_level: AlertLevel::Critical,
            recommended_actions: Vec::new(),
        }
    }

    /// Add recommended action
    pub fn with_action(mut self, action: EmergencyAction) -> Self {
        self.recommended_actions.push(action);
        self
    }

    /// Set alert level
    pub fn with_alert_level(mut self, level: AlertLevel) -> Self {
        self.alert_level = level;
        self
    }
}

/// Limit enforcer with real-time validation
pub struct LimitEnforcer {
    rules: Arc<RwLock<RiskLimitRules>>,
    current_portfolio: Arc<RwLock<Portfolio>>,
    enforcement_history: Arc<RwLock<Vec<EnforcementRecord>>>,
}

impl LimitEnforcer {
    /// Create new limit enforcer
    pub fn new(rules: RiskLimitRules, portfolio: Portfolio) -> Self {
        Self {
            rules: Arc::new(RwLock::new(rules)),
            current_portfolio: Arc::new(RwLock::new(portfolio)),
            enforcement_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Check if a trade is allowed
    pub fn check_trade_allowed(
        &self,
        symbol: &Symbol,
        quantity: f64,
        price: f64,
    ) -> Result<EnforcementDecision> {
        let portfolio = self.current_portfolio.read();
        let rules = self.rules.read();

        // Simulate trade impact
        let mut test_portfolio = portfolio.clone();
        self.apply_simulated_trade(&mut test_portfolio, symbol, quantity, price)?;

        // Calculate metrics for simulated portfolio
        let metrics = RiskMetrics::from_portfolio(&test_portfolio);

        // Check for position size limits
        let position_value = quantity.abs() * price;
        let total_value = test_portfolio.total_value();

        if total_value > 0.0 {
            let concentration = position_value / total_value;
            if concentration > 0.30 {
                // More than 30% in single position
                warn!(
                    "Trade would create high concentration: {:.1}% in {}",
                    concentration * 100.0,
                    symbol
                );
            }
        }

        // Check all rules
        let breached = rules.get_breached_limits(&test_portfolio, &metrics);

        let decision = if breached.is_empty() {
            info!("Trade allowed: {} {} shares at ${}", symbol, quantity, price);
            EnforcementDecision::allow()
        } else {
            let breached_ids: Vec<String> = breached.iter().map(|(id, _)| id.clone()).collect();
            let reason = format!(
                "Trade would breach {} risk limit(s): {}",
                breached.len(),
                breached_ids.join(", ")
            );

            error!("Trade blocked: {}", reason);

            let mut decision = EnforcementDecision::block(reason, breached_ids);

            // Add recommended actions based on breaches
            for (id, limit) in &breached {
                if id.contains("var") {
                    decision = decision.with_action(EmergencyAction::Alert {
                        level: AlertLevel::Critical,
                        message: format!("VaR limit breached: {:.2}%", limit.utilization()),
                    });
                }
            }

            decision
        };

        // Record enforcement decision
        self.record_enforcement(&decision, symbol, quantity, price);

        Ok(decision)
    }

    /// Validate proposed position change
    pub fn validate_position_change(
        &self,
        symbol: &Symbol,
        new_quantity: f64,
    ) -> Result<EnforcementDecision> {
        let portfolio = self.current_portfolio.read();

        let current_price = portfolio
            .get_position(symbol)
            .map(|p| p.current_price.to_f64().unwrap_or(0.0))
            .unwrap_or(100.0); // Fallback price

        let current_quantity = portfolio
            .get_position(symbol)
            .map(|p| p.quantity.to_f64().unwrap_or(0.0))
            .unwrap_or(0.0);

        let delta_quantity = new_quantity - current_quantity;

        drop(portfolio);

        self.check_trade_allowed(symbol, delta_quantity, current_price)
    }

    /// Check if portfolio is currently within all limits
    pub fn check_portfolio_compliance(&self) -> Result<Vec<(String, f64)>> {
        let portfolio = self.current_portfolio.read();
        let rules = self.rules.read();
        let metrics = RiskMetrics::from_portfolio(&portfolio);

        let breached = rules.get_breached_limits(&portfolio, &metrics);

        Ok(breached
            .into_iter()
            .map(|(id, limit)| (id, limit.utilization()))
            .collect())
    }

    /// Get limit utilization for all rules
    pub fn get_limit_utilization(&self) -> Result<Vec<LimitUtilization>> {
        let portfolio = self.current_portfolio.read();
        let rules = self.rules.read();
        let metrics = RiskMetrics::from_portfolio(&portfolio);

        let limits = rules.check_all(&portfolio, &metrics);

        Ok(limits
            .into_iter()
            .map(|limit| LimitUtilization {
                limit_type: format!("{:?}", limit.limit_type),
                current: limit.current_value,
                threshold: limit.threshold,
                utilization_pct: limit.utilization(),
                is_breached: limit.is_breached(),
            })
            .collect())
    }

    /// Update portfolio state
    pub fn update_portfolio(&self, portfolio: Portfolio) {
        let mut current = self.current_portfolio.write();
        *current = portfolio;
        debug!("Portfolio updated for limit enforcement");
    }

    /// Update risk limit rules
    pub fn update_rules(&self, rules: RiskLimitRules) {
        let mut current = self.rules.write();
        *current = rules;
        info!("Risk limit rules updated");
    }

    /// Get enforcement history
    pub fn get_enforcement_history(&self, limit: usize) -> Vec<EnforcementRecord> {
        let history = self.enforcement_history.read();
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Clear enforcement history
    pub fn clear_history(&self) {
        let mut history = self.enforcement_history.write();
        history.clear();
        info!("Enforcement history cleared");
    }

    /// Apply simulated trade to portfolio
    fn apply_simulated_trade(
        &self,
        portfolio: &mut Portfolio,
        symbol: &Symbol,
        quantity: f64,
        price: f64,
    ) -> Result<()> {
        let existing_pos = portfolio.get_position(symbol);

        let new_quantity = if let Some(pos) = existing_pos {
            pos.quantity.to_f64().unwrap_or(0.0) + quantity
        } else {
            quantity
        };

        // Simplified position update for simulation
        if new_quantity.abs() > 0.01 {
            let market_value = new_quantity * price;
            portfolio.update_position(Position {
                symbol: symbol.clone(),
                quantity: rust_decimal::Decimal::from_f64_retain(new_quantity).unwrap(),
                avg_entry_price: rust_decimal::Decimal::from_f64_retain(price).unwrap(),
                current_price: rust_decimal::Decimal::from_f64_retain(price).unwrap(),
                market_value: rust_decimal::Decimal::from_f64_retain(market_value).unwrap(),
                unrealized_pnl: rust_decimal::Decimal::ZERO,
                unrealized_pnl_percent: rust_decimal::Decimal::ZERO,
                side: if new_quantity > 0.0 {
                    crate::types::PositionSide::Long
                } else {
                    crate::types::PositionSide::Short
                },
                opened_at: Utc::now(),
            });
        } else {
            portfolio.remove_position(symbol);
        }

        Ok(())
    }

    /// Record enforcement decision
    fn record_enforcement(
        &self,
        decision: &EnforcementDecision,
        symbol: &Symbol,
        quantity: f64,
        price: f64,
    ) {
        let record = EnforcementRecord {
            timestamp: Utc::now(),
            symbol: symbol.clone(),
            quantity,
            price,
            allowed: decision.allowed,
            reason: decision.reason.clone(),
            breached_limits: decision.breached_limits.clone(),
        };

        let mut history = self.enforcement_history.write();
        history.push(record);

        // Keep only last 1000 records
        if history.len() > 1000 {
            history.drain(0..100);
        }
    }
}

/// Enforcement record for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementRecord {
    pub timestamp: DateTime<Utc>,
    pub symbol: Symbol,
    pub quantity: f64,
    pub price: f64,
    pub allowed: bool,
    pub reason: Option<String>,
    pub breached_limits: Vec<String>,
}

/// Limit utilization information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitUtilization {
    pub limit_type: String,
    pub current: f64,
    pub threshold: f64,
    pub utilization_pct: f64,
    pub is_breached: bool,
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
    fn test_trade_allowed() {
        let rules = RiskLimitRules::with_defaults();
        let portfolio = create_test_portfolio();
        let enforcer = LimitEnforcer::new(rules, portfolio);

        let decision = enforcer
            .check_trade_allowed(&Symbol::new("MSFT"), 50.0, 200.0)
            .unwrap();

        assert!(decision.allowed);
    }

    #[test]
    fn test_trade_blocked_high_concentration() {
        let mut rules = RiskLimitRules::with_defaults();
        // Set very low concentration limit
        rules.update_threshold("max_concentration", 0.05).unwrap();

        let portfolio = create_test_portfolio();
        let enforcer = LimitEnforcer::new(rules, portfolio);

        // Try to buy huge position
        let decision = enforcer
            .check_trade_allowed(&Symbol::new("TSLA"), 1000.0, 250.0)
            .unwrap();

        // Should be blocked due to high concentration
        assert!(!decision.allowed);
    }

    #[test]
    fn test_get_utilization() {
        let rules = RiskLimitRules::with_defaults();
        let portfolio = create_test_portfolio();
        let enforcer = LimitEnforcer::new(rules, portfolio);

        let utilization = enforcer.get_limit_utilization().unwrap();
        assert!(!utilization.is_empty());
    }

    #[test]
    fn test_enforcement_history() {
        let rules = RiskLimitRules::with_defaults();
        let portfolio = create_test_portfolio();
        let enforcer = LimitEnforcer::new(rules, portfolio);

        // Make some checks
        let _ = enforcer.check_trade_allowed(&Symbol::new("AAPL"), 10.0, 150.0);
        let _ = enforcer.check_trade_allowed(&Symbol::new("MSFT"), 20.0, 200.0);

        let history = enforcer.get_enforcement_history(10);
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_portfolio_update() {
        let rules = RiskLimitRules::with_defaults();
        let portfolio = create_test_portfolio();
        let enforcer = LimitEnforcer::new(rules, portfolio.clone());

        let mut new_portfolio = portfolio;
        new_portfolio.cash = dec!(50000);

        enforcer.update_portfolio(new_portfolio.clone());

        // Verify update
        let current = enforcer.current_portfolio.read();
        assert_eq!(current.cash, dec!(50000));
    }
}
