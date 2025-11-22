//! Compliance rules engine with zero tolerance for violations

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use rust_decimal::{Decimal, prelude::FromStr};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use dashmap::DashMap;
use bitflags::bitflags;
use crate::error::{ComplianceError, ComplianceResult};

/// Trading context for rule evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingContext {
    pub order_id: Uuid,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub order_type: OrderType,
    pub trader_id: String,
    pub timestamp: DateTime<Utc>,
    pub portfolio_value: Decimal,
    pub current_positions: HashMap<String, Position>,
    pub daily_pnl: Decimal,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: Decimal,
    pub average_price: Decimal,
    pub unrealized_pnl: Decimal,
    pub market_value: Decimal,
}

bitflags! {
    /// Rule categories for efficient filtering
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct RuleCategories: u32 {
        const POSITION_LIMITS = 0b00000001;
        const LEVERAGE_LIMITS = 0b00000010;
        const RISK_LIMITS = 0b00000100;
        const SURVEILLANCE = 0b00001000;
        const REGULATORY = 0b00010000;
        const LIQUIDITY = 0b00100000;
        const CONCENTRATION = 0b01000000;
        const FREQUENCY = 0b10000000;
        
        const ALL = Self::POSITION_LIMITS.bits() 
                  | Self::LEVERAGE_LIMITS.bits()
                  | Self::RISK_LIMITS.bits()
                  | Self::SURVEILLANCE.bits()
                  | Self::REGULATORY.bits()
                  | Self::LIQUIDITY.bits()
                  | Self::CONCENTRATION.bits()
                  | Self::FREQUENCY.bits();
    }
}

/// Compliance rule trait
#[async_trait]
pub trait ComplianceRule: Send + Sync {
    /// Unique rule identifier
    fn id(&self) -> &str;
    
    /// Rule description
    fn description(&self) -> &str;
    
    /// Rule categories
    fn categories(&self) -> RuleCategories;
    
    /// Rule priority (higher = more critical)
    fn priority(&self) -> u8;
    
    /// Evaluate the rule against trading context
    async fn evaluate(&self, context: &TradingContext) -> ComplianceResult<RuleResult>;
    
    /// Whether this rule should block execution on failure
    fn blocks_execution(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleResult {
    pub rule_id: String,
    pub passed: bool,
    pub details: Option<String>,
    pub risk_score: Option<f64>,
    pub suggested_action: Option<String>,
}

/// Position limit rule
pub struct PositionLimitRule {
    id: String,
    symbol_limits: HashMap<String, Decimal>,
    global_limit: Decimal,
}

impl PositionLimitRule {
    pub fn new(symbol_limits: HashMap<String, Decimal>, global_limit: Decimal) -> Self {
        Self {
            id: "position_limits".to_string(),
            symbol_limits,
            global_limit,
        }
    }
}

#[async_trait]
impl ComplianceRule for PositionLimitRule {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn description(&self) -> &str {
        "Enforces position size limits per symbol and globally"
    }
    
    fn categories(&self) -> RuleCategories {
        RuleCategories::POSITION_LIMITS
    }
    
    fn priority(&self) -> u8 {
        100
    }
    
    async fn evaluate(&self, context: &TradingContext) -> ComplianceResult<RuleResult> {
        // Check symbol-specific limit
        if let Some(limit) = self.symbol_limits.get(&context.symbol) {
            let current_position = context.current_positions
                .get(&context.symbol)
                .map(|p| p.quantity)
                .unwrap_or(Decimal::ZERO);
            
            let new_position = match context.side {
                OrderSide::Buy => current_position + context.quantity,
                OrderSide::Sell => current_position - context.quantity,
            };
            
            if new_position.abs() > *limit {
                return Err(ComplianceError::PositionLimitViolation {
                    symbol: context.symbol.clone(),
                    current: new_position.abs(),
                    limit: *limit,
                });
            }
        }
        
        // Check global exposure
        let total_exposure: Decimal = context.current_positions
            .values()
            .map(|p| p.market_value.abs())
            .sum();
        
        if total_exposure > self.global_limit {
            return Err(ComplianceError::RiskLimitBreach {
                metric: "global_exposure".to_string(),
                value: total_exposure,
                threshold: self.global_limit,
            });
        }
        
        Ok(RuleResult {
            rule_id: self.id.clone(),
            passed: true,
            details: Some(format!("Position within limits for {}", context.symbol)),
            risk_score: Some(0.1),
            suggested_action: None,
        })
    }
}

/// Leverage limit rule
pub struct LeverageRule {
    id: String,
    max_leverage: Decimal,
}

impl LeverageRule {
    pub fn new(max_leverage: Decimal) -> Self {
        Self {
            id: "leverage_limits".to_string(),
            max_leverage,
        }
    }
}

#[async_trait]
impl ComplianceRule for LeverageRule {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn description(&self) -> &str {
        "Enforces maximum leverage constraints"
    }
    
    fn categories(&self) -> RuleCategories {
        RuleCategories::LEVERAGE_LIMITS
    }
    
    fn priority(&self) -> u8 {
        95
    }
    
    async fn evaluate(&self, context: &TradingContext) -> ComplianceResult<RuleResult> {
        let total_exposure: Decimal = context.current_positions
            .values()
            .map(|p| p.market_value.abs())
            .sum();
        
        if context.portfolio_value > Decimal::ZERO {
            let current_leverage = total_exposure / context.portfolio_value;
            
            if current_leverage > self.max_leverage {
                return Err(ComplianceError::LeverageViolation {
                    current: current_leverage,
                    max_allowed: self.max_leverage,
                });
            }
        }
        
        Ok(RuleResult {
            rule_id: self.id.clone(),
            passed: true,
            details: Some("Leverage within acceptable limits".to_string()),
            risk_score: Some(0.1),
            suggested_action: None,
        })
    }
}

/// Daily loss limit rule
pub struct DailyLossLimitRule {
    id: String,
    max_daily_loss: Decimal,
}

impl DailyLossLimitRule {
    pub fn new(max_daily_loss: Decimal) -> Self {
        Self {
            id: "daily_loss_limit".to_string(),
            max_daily_loss,
        }
    }
}

#[async_trait]
impl ComplianceRule for DailyLossLimitRule {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn description(&self) -> &str {
        "Enforces daily loss limits"
    }
    
    fn categories(&self) -> RuleCategories {
        RuleCategories::RISK_LIMITS
    }
    
    fn priority(&self) -> u8 {
        90
    }
    
    async fn evaluate(&self, context: &TradingContext) -> ComplianceResult<RuleResult> {
        if context.daily_pnl < -self.max_daily_loss.abs() {
            return Err(ComplianceError::DailyLossLimitExceeded {
                current_loss: -context.daily_pnl,
                daily_limit: self.max_daily_loss.abs(),
            });
        }
        
        Ok(RuleResult {
            rule_id: self.id.clone(),
            passed: true,
            details: Some("Daily loss within acceptable limits".to_string()),
            risk_score: Some(0.2),
            suggested_action: None,
        })
    }
}

/// Concentration risk rule
pub struct ConcentrationRiskRule {
    id: String,
    max_concentration: f64, // Percentage
}

impl ConcentrationRiskRule {
    pub fn new(max_concentration: f64) -> Self {
        Self {
            id: "concentration_risk".to_string(),
            max_concentration,
        }
    }
}

#[async_trait]
impl ComplianceRule for ConcentrationRiskRule {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn description(&self) -> &str {
        "Prevents excessive concentration in single assets"
    }
    
    fn categories(&self) -> RuleCategories {
        RuleCategories::CONCENTRATION
    }
    
    fn priority(&self) -> u8 {
        80
    }
    
    async fn evaluate(&self, context: &TradingContext) -> ComplianceResult<RuleResult> {
        if context.portfolio_value > Decimal::ZERO {
            let position_value = context.current_positions
                .get(&context.symbol)
                .map(|p| p.market_value.abs())
                .unwrap_or(Decimal::ZERO);
            
            let concentration = (position_value / context.portfolio_value).to_f64().unwrap_or(0.0) * 100.0;
            
            if concentration > self.max_concentration {
                return Err(ComplianceError::ConcentrationRiskExceeded {
                    asset: context.symbol.clone(),
                    concentration,
                });
            }
        }
        
        Ok(RuleResult {
            rule_id: self.id.clone(),
            passed: true,
            details: Some("Concentration within limits".to_string()),
            risk_score: Some(0.3),
            suggested_action: None,
        })
    }
}

/// Rule engine that evaluates all compliance rules
pub struct RuleEngine {
    rules: Arc<DashMap<String, Box<dyn ComplianceRule>>>,
    enabled_categories: RuleCategories,
}

impl RuleEngine {
    pub fn new() -> Self {
        Self {
            rules: Arc::new(DashMap::new()),
            enabled_categories: RuleCategories::ALL,
        }
    }

    pub fn add_rule(&self, rule: Box<dyn ComplianceRule>) {
        self.rules.insert(rule.id().to_string(), rule);
    }

    pub fn set_enabled_categories(&mut self, categories: RuleCategories) {
        self.enabled_categories = categories;
    }

    pub async fn evaluate_all(&self, context: &TradingContext) -> ComplianceResult<Vec<RuleResult>> {
        let mut results = Vec::new();
        let mut violations = Vec::new();
        
        // Sort rules by priority (highest first)
        let mut rule_entries: Vec<_> = self.rules.iter().collect();
        rule_entries.sort_by(|a, b| b.value().priority().cmp(&a.value().priority()));
        
        for entry in rule_entries {
            let rule = entry.value();
            
            // Skip if category is disabled
            if !self.enabled_categories.intersects(rule.categories()) {
                continue;
            }
            
            match rule.evaluate(context).await {
                Ok(result) => {
                    results.push(result);
                }
                Err(e) => {
                    if rule.blocks_execution() {
                        // Critical rule failed - stop evaluation and return error
                        return Err(e);
                    } else {
                        // Non-blocking rule failed - continue but record violation
                        violations.push(e);
                        results.push(RuleResult {
                            rule_id: rule.id().to_string(),
                            passed: false,
                            details: Some(e.to_string()),
                            risk_score: Some(1.0),
                            suggested_action: Some("Review and modify order".to_string()),
                        });
                    }
                }
            }
        }
        
        // If we have non-blocking violations but execution can continue
        if !violations.is_empty() {
            tracing::warn!("Non-blocking violations detected: {}", violations.len());
        }
        
        Ok(results)
    }

    pub fn get_rule_count(&self) -> usize {
        self.rules.len()
    }

    pub fn get_enabled_categories(&self) -> RuleCategories {
        self.enabled_categories
    }
}

/// Rule set builder for common compliance configurations
pub struct RuleSet;

impl RuleSet {
    /// Create a conservative rule set for high-risk environments
    pub fn conservative() -> Vec<Box<dyn ComplianceRule>> {
        vec![
            Box::new(PositionLimitRule::new(
                HashMap::new(),
                Decimal::from(1_000_000), // $1M global limit
            )),
            Box::new(LeverageRule::new(Decimal::from(2))), // 2:1 max leverage
            Box::new(DailyLossLimitRule::new(Decimal::from(50_000))), // $50K daily loss limit
            Box::new(ConcentrationRiskRule::new(20.0)), // 20% max concentration
        ]
    }
    
    /// Create an aggressive rule set for experienced traders
    pub fn aggressive() -> Vec<Box<dyn ComplianceRule>> {
        vec![
            Box::new(PositionLimitRule::new(
                HashMap::new(),
                Decimal::from(10_000_000), // $10M global limit
            )),
            Box::new(LeverageRule::new(Decimal::from(10))), // 10:1 max leverage
            Box::new(DailyLossLimitRule::new(Decimal::from(500_000))), // $500K daily loss limit
            Box::new(ConcentrationRiskRule::new(50.0)), // 50% max concentration
        ]
    }
    
    /// Create a minimal rule set for testing
    pub fn minimal() -> Vec<Box<dyn ComplianceRule>> {
        vec![
            Box::new(DailyLossLimitRule::new(Decimal::from(1_000_000))), // $1M daily loss limit
        ]
    }
}