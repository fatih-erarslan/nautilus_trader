//! Risk limit rules and configuration
//!
//! Defines rules for:
//! - Position size limits
//! - Portfolio VaR limits
//! - Drawdown thresholds
//! - Leverage constraints
//! - Concentration limits

use crate::{Result, RiskError};
use crate::types::{Portfolio, RiskLimit, RiskLimitType, Symbol};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, warn};

/// Risk limit rule with validation logic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimitRule {
    /// Rule identifier
    pub id: String,
    /// Rule description
    pub description: String,
    /// Limit type
    pub limit_type: RiskLimitType,
    /// Threshold value
    pub threshold: f64,
    /// Whether rule is enabled
    pub enabled: bool,
    /// Whether to block trades on breach
    pub block_on_breach: bool,
    /// Symbol-specific limit (None = portfolio-wide)
    pub symbol: Option<Symbol>,
}

impl RiskLimitRule {
    /// Create new limit rule
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        limit_type: RiskLimitType,
        threshold: f64,
    ) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            limit_type,
            threshold,
            enabled: true,
            block_on_breach: true,
            symbol: None,
        }
    }

    /// Set whether to block trades on breach
    pub fn block_on_breach(mut self, block: bool) -> Self {
        self.block_on_breach = block;
        self
    }

    /// Set symbol-specific limit
    pub fn for_symbol(mut self, symbol: Symbol) -> Self {
        self.symbol = Some(symbol);
        self
    }

    /// Enable/disable rule
    pub fn set_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Check if rule is breached
    pub fn check(&self, current_value: f64) -> RiskLimit {
        RiskLimit {
            limit_type: self.limit_type.clone(),
            threshold: self.threshold,
            current_value,
            enabled: self.enabled,
        }
    }

    /// Validate limit is breached and should trigger action
    pub fn is_breached(&self, current_value: f64) -> bool {
        self.enabled && current_value > self.threshold
    }
}

/// Collection of risk limit rules
#[derive(Debug, Clone)]
pub struct RiskLimitRules {
    rules: HashMap<String, RiskLimitRule>,
}

impl RiskLimitRules {
    /// Create new empty rule set
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }

    /// Create with default rules
    pub fn with_defaults() -> Self {
        let mut rules = Self::new();

        // Portfolio-level VaR limit (10% of portfolio value)
        rules.add_rule(RiskLimitRule::new(
            "portfolio_var_95",
            "Portfolio VaR must not exceed 10% of portfolio value",
            RiskLimitType::PortfolioVaR,
            0.10,
        ));

        // Maximum drawdown (20%)
        rules.add_rule(RiskLimitRule::new(
            "max_drawdown",
            "Maximum drawdown limit",
            RiskLimitType::MaxDrawdown,
            0.20,
        ));

        // Maximum leverage (2x)
        rules.add_rule(RiskLimitRule::new(
            "max_leverage",
            "Maximum portfolio leverage",
            RiskLimitType::MaxLeverage,
            2.0,
        ));

        // Maximum single position concentration (25%)
        rules.add_rule(RiskLimitRule::new(
            "max_concentration",
            "Maximum concentration in single asset",
            RiskLimitType::MaxConcentration,
            0.25,
        ));

        rules
    }

    /// Add a new rule
    pub fn add_rule(&mut self, rule: RiskLimitRule) {
        debug!("Adding risk limit rule: {} - {}", rule.id, rule.description);
        self.rules.insert(rule.id.clone(), rule);
    }

    /// Remove a rule
    pub fn remove_rule(&mut self, id: &str) -> Option<RiskLimitRule> {
        self.rules.remove(id)
    }

    /// Get a rule by ID
    pub fn get_rule(&self, id: &str) -> Option<&RiskLimitRule> {
        self.rules.get(id)
    }

    /// Get mutable rule reference
    pub fn get_rule_mut(&mut self, id: &str) -> Option<&mut RiskLimitRule> {
        self.rules.get_mut(id)
    }

    /// Get all rules
    pub fn all_rules(&self) -> impl Iterator<Item = &RiskLimitRule> {
        self.rules.values()
    }

    /// Get enabled rules only
    pub fn enabled_rules(&self) -> impl Iterator<Item = &RiskLimitRule> {
        self.rules.values().filter(|r| r.enabled)
    }

    /// Enable/disable a rule
    pub fn set_rule_enabled(&mut self, id: &str, enabled: bool) -> Result<()> {
        let rule = self.rules.get_mut(id).ok_or_else(|| {
            RiskError::InvalidConfig(format!("Rule not found: {}", id))
        })?;
        rule.enabled = enabled;
        Ok(())
    }

    /// Update rule threshold
    pub fn update_threshold(&mut self, id: &str, new_threshold: f64) -> Result<()> {
        let rule = self.rules.get_mut(id).ok_or_else(|| {
            RiskError::InvalidConfig(format!("Rule not found: {}", id))
        })?;
        rule.threshold = new_threshold;
        debug!("Updated rule {} threshold to {}", id, new_threshold);
        Ok(())
    }

    /// Check all rules against portfolio
    pub fn check_all(&self, portfolio: &Portfolio, metrics: &RiskMetrics) -> Vec<RiskLimit> {
        let mut limits = Vec::new();

        for rule in self.enabled_rules() {
            let current_value = self.calculate_current_value(portfolio, metrics, rule);
            let limit = rule.check(current_value);

            if limit.is_breached() {
                warn!(
                    "Risk limit breached: {} - current: {:.4}, threshold: {:.4}",
                    rule.description, current_value, rule.threshold
                );
            }

            limits.push(limit);
        }

        limits
    }

    /// Get all breached limits
    pub fn get_breached_limits(
        &self,
        portfolio: &Portfolio,
        metrics: &RiskMetrics,
    ) -> Vec<(String, RiskLimit)> {
        self.enabled_rules()
            .filter_map(|rule| {
                let current_value = self.calculate_current_value(portfolio, metrics, rule);
                let limit = rule.check(current_value);
                if limit.is_breached() {
                    Some((rule.id.clone(), limit))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Calculate current value for a specific rule
    fn calculate_current_value(
        &self,
        portfolio: &Portfolio,
        metrics: &RiskMetrics,
        rule: &RiskLimitRule,
    ) -> f64 {
        match rule.limit_type {
            RiskLimitType::PortfolioVaR => {
                // VaR as percentage of portfolio value
                let portfolio_value = portfolio.total_value();
                if portfolio_value > 0.0 {
                    metrics.var_95 / portfolio_value
                } else {
                    0.0
                }
            }
            RiskLimitType::MaxDrawdown => metrics.max_drawdown,
            RiskLimitType::MaxLeverage => metrics.leverage,
            RiskLimitType::MaxConcentration => {
                if let Some(ref symbol) = rule.symbol {
                    // Symbol-specific concentration
                    let total_value = portfolio.total_value();
                    if total_value > 0.0 {
                        portfolio
                            .get_position(symbol)
                            .map(|p| p.exposure().abs() / total_value)
                            .unwrap_or(0.0)
                    } else {
                        0.0
                    }
                } else {
                    // Portfolio-wide concentration (Herfindahl index)
                    metrics.concentration
                }
            }
            RiskLimitType::PositionSize => {
                // Position size for specific symbol
                if let Some(ref symbol) = rule.symbol {
                    portfolio
                        .get_position(symbol)
                        .map(|p| p.exposure().abs())
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            }
            RiskLimitType::MaxCorrelation => {
                // Would need correlation data
                0.0
            }
        }
    }
}

impl Default for RiskLimitRules {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Risk metrics for limit checking
#[derive(Debug, Clone)]
pub struct RiskMetrics {
    pub var_95: f64,
    pub cvar_95: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub concentration: f64,
    pub leverage: f64,
}

impl RiskMetrics {
    /// Create risk metrics from portfolio
    pub fn from_portfolio(portfolio: &Portfolio) -> Self {
        let total_value = portfolio.total_value();
        let leverage = if total_value > 0.0 {
            let gross_exposure: f64 = portfolio
                .positions
                .values()
                .map(|p| p.exposure().abs())
                .sum();
            gross_exposure / total_value
        } else {
            0.0
        };

        Self {
            var_95: 0.0,                        // To be calculated
            cvar_95: 0.0,                       // To be calculated
            volatility: 0.0,                    // To be calculated
            sharpe_ratio: 0.0,                  // To be calculated
            max_drawdown: 0.0,                  // To be tracked
            concentration: portfolio.concentration(),
            leverage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PositionSide};
    use chrono::Utc;
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
    fn test_rule_creation() {
        let rule = RiskLimitRule::new(
            "test_rule",
            "Test rule",
            RiskLimitType::PortfolioVaR,
            0.10,
        );

        assert_eq!(rule.id, "test_rule");
        assert_eq!(rule.threshold, 0.10);
        assert!(rule.enabled);
        assert!(rule.block_on_breach);
    }

    #[test]
    fn test_rule_breach_detection() {
        let rule = RiskLimitRule::new(
            "var_limit",
            "VaR limit",
            RiskLimitType::PortfolioVaR,
            0.10,
        );

        assert!(!rule.is_breached(0.05)); // Below limit
        assert!(!rule.is_breached(0.10)); // At limit
        assert!(rule.is_breached(0.15));  // Above limit
    }

    #[test]
    fn test_rules_collection() {
        let mut rules = RiskLimitRules::new();

        rules.add_rule(RiskLimitRule::new(
            "rule1",
            "Rule 1",
            RiskLimitType::MaxDrawdown,
            0.20,
        ));

        assert!(rules.get_rule("rule1").is_some());
        assert!(rules.get_rule("rule2").is_none());
    }

    #[test]
    fn test_default_rules() {
        let rules = RiskLimitRules::with_defaults();
        assert!(rules.get_rule("portfolio_var_95").is_some());
        assert!(rules.get_rule("max_drawdown").is_some());
        assert!(rules.get_rule("max_leverage").is_some());
    }

    #[test]
    fn test_check_all_rules() {
        let portfolio = create_test_portfolio();
        let rules = RiskLimitRules::with_defaults();
        let metrics = RiskMetrics::from_portfolio(&portfolio);

        let limits = rules.check_all(&portfolio, &metrics);
        assert!(!limits.is_empty());
    }

    #[test]
    fn test_update_threshold() {
        let mut rules = RiskLimitRules::with_defaults();

        rules.update_threshold("max_drawdown", 0.30).unwrap();
        let rule = rules.get_rule("max_drawdown").unwrap();
        assert_eq!(rule.threshold, 0.30);
    }

    #[test]
    fn test_enable_disable_rule() {
        let mut rules = RiskLimitRules::with_defaults();

        rules.set_rule_enabled("max_leverage", false).unwrap();
        let rule = rules.get_rule("max_leverage").unwrap();
        assert!(!rule.enabled);
    }

    #[test]
    fn test_symbol_specific_limit() {
        let rule = RiskLimitRule::new(
            "aapl_limit",
            "AAPL position limit",
            RiskLimitType::PositionSize,
            5000.0,
        )
        .for_symbol(Symbol::new("AAPL"));

        assert!(rule.symbol.is_some());
        assert_eq!(rule.symbol.unwrap().as_str(), "AAPL");
    }
}
