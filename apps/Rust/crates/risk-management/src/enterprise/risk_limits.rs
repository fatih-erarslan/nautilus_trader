//! Automated Risk Limit Enforcement System
//! 
//! Provides instant risk limit enforcement with automated controls and
//! position management to prevent risk limit breaches.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::*;

/// Risk limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimitsConfig {
    /// Maximum portfolio VaR limits
    pub var_limits: VarLimits,
    
    /// Position size limits
    pub position_limits: PositionLimits,
    
    /// Sector concentration limits
    pub sector_limits: SectorLimits,
    
    /// Leverage limits
    pub leverage_limits: LeverageLimits,
    
    /// Drawdown limits
    pub drawdown_limits: DrawdownLimits,
    
    /// Liquidity limits
    pub liquidity_limits: LiquidityLimits,
    
    /// Automated enforcement settings
    pub enforcement_config: EnforcementConfig,
}

impl Default for RiskLimitsConfig {
    fn default() -> Self {
        Self {
            var_limits: VarLimits::default(),
            position_limits: PositionLimits::default(),
            sector_limits: SectorLimits::default(),
            leverage_limits: LeverageLimits::default(),
            drawdown_limits: DrawdownLimits::default(),
            liquidity_limits: LiquidityLimits::default(),
            enforcement_config: EnforcementConfig::default(),
        }
    }
}

/// VaR limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarLimits {
    /// Maximum 1-day VaR at 95% confidence (as % of portfolio value)
    pub max_var_95_percent: f64,
    
    /// Maximum 1-day VaR at 99% confidence (as % of portfolio value)
    pub max_var_99_percent: f64,
    
    /// Maximum 10-day VaR at 99% confidence (as % of portfolio value)
    pub max_var_10day_99_percent: f64,
    
    /// Soft warning threshold (as % of hard limit)
    pub warning_threshold_percent: f64,
}

impl Default for VarLimits {
    fn default() -> Self {
        Self {
            max_var_95_percent: 2.0,
            max_var_99_percent: 3.5,
            max_var_10day_99_percent: 8.0,
            warning_threshold_percent: 80.0,
        }
    }
}

/// Position size limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionLimits {
    /// Maximum single position size (as % of portfolio)
    pub max_single_position_percent: f64,
    
    /// Maximum position size by asset class
    pub max_asset_class_limits: HashMap<AssetClass, f64>,
    
    /// Maximum number of positions
    pub max_positions: u32,
    
    /// Minimum position size threshold
    pub min_position_value: f64,
}

impl Default for PositionLimits {
    fn default() -> Self {
        let mut asset_class_limits = HashMap::new();
        asset_class_limits.insert(AssetClass::Equity, 60.0);
        asset_class_limits.insert(AssetClass::FixedIncome, 80.0);
        asset_class_limits.insert(AssetClass::Commodities, 20.0);
        asset_class_limits.insert(AssetClass::ForeignExchange, 30.0);
        asset_class_limits.insert(AssetClass::Derivatives, 15.0);
        asset_class_limits.insert(AssetClass::Cryptocurrency, 5.0);
        asset_class_limits.insert(AssetClass::AlternativeInvestments, 10.0);
        
        Self {
            max_single_position_percent: 5.0,
            max_asset_class_limits: asset_class_limits,
            max_positions: 200,
            min_position_value: 1000.0,
        }
    }
}

/// Sector concentration limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectorLimits {
    /// Maximum allocation per sector
    pub max_sector_allocation: HashMap<String, f64>,
    
    /// Default sector limit for unlisted sectors
    pub default_sector_limit_percent: f64,
    
    /// Maximum number of sectors
    pub max_sectors: u32,
}

impl Default for SectorLimits {
    fn default() -> Self {
        let mut sector_limits = HashMap::new();
        sector_limits.insert("Technology".to_string(), 25.0);
        sector_limits.insert("Healthcare".to_string(), 20.0);
        sector_limits.insert("Financial Services".to_string(), 20.0);
        sector_limits.insert("Consumer Discretionary".to_string(), 15.0);
        sector_limits.insert("Energy".to_string(), 10.0);
        
        Self {
            max_sector_allocation: sector_limits,
            default_sector_limit_percent: 10.0,
            max_sectors: 15,
        }
    }
}

/// Leverage limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeverageLimits {
    /// Maximum gross leverage ratio
    pub max_gross_leverage: f64,
    
    /// Maximum net leverage ratio
    pub max_net_leverage: f64,
    
    /// Leverage limit by asset class
    pub asset_class_leverage_limits: HashMap<AssetClass, f64>,
}

impl Default for LeverageLimits {
    fn default() -> Self {
        let mut asset_class_limits = HashMap::new();
        asset_class_limits.insert(AssetClass::Equity, 2.0);
        asset_class_limits.insert(AssetClass::FixedIncome, 5.0);
        asset_class_limits.insert(AssetClass::ForeignExchange, 10.0);
        asset_class_limits.insert(AssetClass::Derivatives, 3.0);
        asset_class_limits.insert(AssetClass::Cryptocurrency, 1.5);
        
        Self {
            max_gross_leverage: 3.0,
            max_net_leverage: 2.0,
            asset_class_leverage_limits: asset_class_limits,
        }
    }
}

/// Drawdown limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownLimits {
    /// Maximum daily drawdown (as % of portfolio)
    pub max_daily_drawdown_percent: f64,
    
    /// Maximum monthly drawdown (as % of portfolio)
    pub max_monthly_drawdown_percent: f64,
    
    /// Maximum drawdown from high-water mark
    pub max_total_drawdown_percent: f64,
    
    /// Stop-loss threshold for automatic position closure
    pub stop_loss_threshold_percent: f64,
}

impl Default for DrawdownLimits {
    fn default() -> Self {
        Self {
            max_daily_drawdown_percent: 3.0,
            max_monthly_drawdown_percent: 10.0,
            max_total_drawdown_percent: 15.0,
            stop_loss_threshold_percent: 20.0,
        }
    }
}

/// Liquidity limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityLimits {
    /// Minimum weighted average liquidity score
    pub min_portfolio_liquidity_score: f64,
    
    /// Maximum allocation to illiquid assets (as % of portfolio)
    pub max_illiquid_allocation_percent: f64,
    
    /// Maximum time to liquidate portfolio (in days)
    pub max_liquidation_time_days: f64,
    
    /// Minimum daily trading volume threshold
    pub min_daily_volume: f64,
}

impl Default for LiquidityLimits {
    fn default() -> Self {
        Self {
            min_portfolio_liquidity_score: 0.6,
            max_illiquid_allocation_percent: 20.0,
            max_liquidation_time_days: 7.0,
            min_daily_volume: 1_000_000.0,
        }
    }
}

/// Enforcement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementConfig {
    /// Enable automatic enforcement
    pub auto_enforcement_enabled: bool,
    
    /// Actions to take on limit breaches
    pub breach_actions: HashMap<LimitType, EnforcementAction>,
    
    /// Maximum enforcement latency
    pub max_enforcement_latency: Duration,
    
    /// Require manual approval for certain actions
    pub require_manual_approval: Vec<EnforcementAction>,
    
    /// Grace period before enforcement
    pub grace_period: Duration,
}

impl Default for EnforcementConfig {
    fn default() -> Self {
        let mut breach_actions = HashMap::new();
        breach_actions.insert(LimitType::VarLimit, EnforcementAction::ReducePositions);
        breach_actions.insert(LimitType::PositionLimit, EnforcementAction::ClosePosition);
        breach_actions.insert(LimitType::DrawdownLimit, EnforcementAction::EmergencyStop);
        breach_actions.insert(LimitType::LeverageLimit, EnforcementAction::ReduceLeverage);
        
        Self {
            auto_enforcement_enabled: true,
            breach_actions,
            max_enforcement_latency: Duration::from_micros(10),
            require_manual_approval: vec![EnforcementAction::EmergencyStop],
            grace_period: Duration::from_secs(30),
        }
    }
}

/// Types of risk limits
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum LimitType {
    VarLimit,
    PositionLimit,
    SectorLimit,
    LeverageLimit,
    DrawdownLimit,
    LiquidityLimit,
}

/// Enforcement actions
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum EnforcementAction {
    SendAlert,
    ReducePositions,
    ClosePosition,
    HedgeRisk,
    ReduceLeverage,
    EmergencyStop,
    RequireApproval,
}

/// Limit breach information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitBreach {
    pub id: Uuid,
    pub limit_type: LimitType,
    pub portfolio_id: Uuid,
    pub position_symbol: Option<String>,
    pub current_value: f64,
    pub limit_value: f64,
    pub breach_percentage: f64,
    pub severity: BreachSeverity,
    pub recommended_action: EnforcementAction,
    pub auto_action_available: bool,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Breach severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BreachSeverity {
    Warning,   // 80-100% of limit
    Critical,  // 100-120% of limit
    Emergency, // >120% of limit
}

/// Enforcement action result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementResult {
    pub action_id: Uuid,
    pub action_type: EnforcementAction,
    pub success: bool,
    pub positions_affected: Vec<String>,
    pub amount_reduced: f64,
    pub execution_time: Duration,
    pub error_message: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Automated risk limit enforcement system
#[derive(Debug)]
pub struct AutomatedRiskLimitEnforcement {
    config: RiskLimitsConfig,
    breach_history: Arc<RwLock<Vec<LimitBreach>>>,
    enforcement_history: Arc<RwLock<Vec<EnforcementResult>>>,
    pending_approvals: Arc<RwLock<HashMap<Uuid, LimitBreach>>>,
    alert_sender: mpsc::UnboundedSender<RiskAlert>,
    action_sender: mpsc::UnboundedSender<EnforcementAction>,
    performance_tracker: Arc<RwLock<EnforcementPerformanceTracker>>,
}

impl AutomatedRiskLimitEnforcement {
    /// Create new automated risk limit enforcement system
    pub async fn new(
        config: RiskLimitsConfig,
        alert_sender: mpsc::UnboundedSender<RiskAlert>,
        action_sender: mpsc::UnboundedSender<EnforcementAction>,
    ) -> Result<Self> {
        info!("Initializing Automated Risk Limit Enforcement");
        
        Ok(Self {
            config,
            breach_history: Arc::new(RwLock::new(Vec::new())),
            enforcement_history: Arc::new(RwLock::new(Vec::new())),
            pending_approvals: Arc::new(RwLock::new(HashMap::new())),
            alert_sender,
            action_sender,
            performance_tracker: Arc::new(RwLock::new(EnforcementPerformanceTracker::new())),
        })
    }
    
    /// Check all risk limits for a portfolio
    pub async fn check_limits(&self, portfolio: &Portfolio, risk_metrics: &RiskMetrics) -> Result<Vec<LimitBreach>> {
        let start_time = Instant::now();
        let mut breaches = Vec::new();
        
        // Track performance
        {
            let mut tracker = self.performance_tracker.write().await;
            tracker.start_check();
        }
        
        // Check VaR limits
        breaches.extend(self.check_var_limits(portfolio, risk_metrics).await?);
        
        // Check position limits
        breaches.extend(self.check_position_limits(portfolio).await?);
        
        // Check sector limits
        breaches.extend(self.check_sector_limits(portfolio).await?);
        
        // Check leverage limits
        breaches.extend(self.check_leverage_limits(portfolio, risk_metrics).await?);
        
        // Check drawdown limits
        breaches.extend(self.check_drawdown_limits(portfolio, risk_metrics).await?);
        
        // Check liquidity limits
        breaches.extend(self.check_liquidity_limits(portfolio, risk_metrics).await?);
        
        let check_time = start_time.elapsed();
        
        // Update performance tracker
        {
            let mut tracker = self.performance_tracker.write().await;
            tracker.end_check(check_time, breaches.len());
        }
        
        // Check performance target
        if check_time > Duration::from_micros(10) {
            warn!(
                "Risk limit check took {:?}, exceeding 10μs target for portfolio {}",
                check_time, portfolio.id
            );
        }
        
        // Store breach history
        if !breaches.is_empty() {
            let mut history = self.breach_history.write().await;
            history.extend(breaches.clone());
        }
        
        // Trigger enforcement if enabled
        if self.config.enforcement_config.auto_enforcement_enabled {
            for breach in &breaches {
                if let Err(e) = self.trigger_enforcement(breach).await {
                    error!("Failed to trigger enforcement for breach {}: {}", breach.id, e);
                }
            }
        }
        
        Ok(breaches)
    }
    
    /// Check VaR limits
    async fn check_var_limits(&self, portfolio: &Portfolio, risk_metrics: &RiskMetrics) -> Result<Vec<LimitBreach>> {
        let mut breaches = Vec::new();
        let portfolio_value = portfolio.total_market_value;
        
        if portfolio_value <= 0.0 {
            return Ok(breaches);
        }
        
        // Check 95% VaR limit
        let var_95_percent = (risk_metrics.portfolio_var_95 / portfolio_value) * 100.0;
        if var_95_percent > self.config.var_limits.max_var_95_percent {
            let breach_percentage = (var_95_percent / self.config.var_limits.max_var_95_percent - 1.0) * 100.0;
            let severity = self.calculate_breach_severity(breach_percentage);
            
            breaches.push(LimitBreach {
                id: Uuid::new_v4(),
                limit_type: LimitType::VarLimit,
                portfolio_id: portfolio.id,
                position_symbol: None,
                current_value: var_95_percent,
                limit_value: self.config.var_limits.max_var_95_percent,
                breach_percentage,
                severity,
                recommended_action: EnforcementAction::ReducePositions,
                auto_action_available: true,
                timestamp: chrono::Utc::now(),
            });
        }
        
        // Check 99% VaR limit
        let var_99_percent = (risk_metrics.portfolio_var_99 / portfolio_value) * 100.0;
        if var_99_percent > self.config.var_limits.max_var_99_percent {
            let breach_percentage = (var_99_percent / self.config.var_limits.max_var_99_percent - 1.0) * 100.0;
            let severity = self.calculate_breach_severity(breach_percentage);
            
            breaches.push(LimitBreach {
                id: Uuid::new_v4(),
                limit_type: LimitType::VarLimit,
                portfolio_id: portfolio.id,
                position_symbol: None,
                current_value: var_99_percent,
                limit_value: self.config.var_limits.max_var_99_percent,
                breach_percentage,
                severity,
                recommended_action: if severity == BreachSeverity::Emergency {
                    EnforcementAction::EmergencyStop
                } else {
                    EnforcementAction::ReducePositions
                },
                auto_action_available: severity != BreachSeverity::Emergency,
                timestamp: chrono::Utc::now(),
            });
        }
        
        Ok(breaches)
    }
    
    /// Check position size limits
    async fn check_position_limits(&self, portfolio: &Portfolio) -> Result<Vec<LimitBreach>> {
        let mut breaches = Vec::new();
        let portfolio_value = portfolio.total_market_value;
        
        if portfolio_value <= 0.0 {
            return Ok(breaches);
        }
        
        // Check individual position limits
        for position in &portfolio.positions {
            let position_percent = (position.market_value.abs() / portfolio_value) * 100.0;
            
            if position_percent > self.config.position_limits.max_single_position_percent {
                let breach_percentage = (position_percent / self.config.position_limits.max_single_position_percent - 1.0) * 100.0;
                let severity = self.calculate_breach_severity(breach_percentage);
                
                breaches.push(LimitBreach {
                    id: Uuid::new_v4(),
                    limit_type: LimitType::PositionLimit,
                    portfolio_id: portfolio.id,
                    position_symbol: Some(position.symbol.clone()),
                    current_value: position_percent,
                    limit_value: self.config.position_limits.max_single_position_percent,
                    breach_percentage,
                    severity,
                    recommended_action: EnforcementAction::ReducePositions,
                    auto_action_available: true,
                    timestamp: chrono::Utc::now(),
                });
            }
        }
        
        // Check asset class limits
        let mut asset_class_allocations: HashMap<AssetClass, f64> = HashMap::new();
        for position in &portfolio.positions {
            let allocation = (position.market_value.abs() / portfolio_value) * 100.0;
            *asset_class_allocations.entry(position.asset_class.clone()).or_insert(0.0) += allocation;
        }
        
        for (asset_class, allocation) in asset_class_allocations {
            if let Some(&limit) = self.config.position_limits.max_asset_class_limits.get(&asset_class) {
                if allocation > limit {
                    let breach_percentage = (allocation / limit - 1.0) * 100.0;
                    let severity = self.calculate_breach_severity(breach_percentage);
                    
                    breaches.push(LimitBreach {
                        id: Uuid::new_v4(),
                        limit_type: LimitType::PositionLimit,
                        portfolio_id: portfolio.id,
                        position_symbol: None,
                        current_value: allocation,
                        limit_value: limit,
                        breach_percentage,
                        severity,
                        recommended_action: EnforcementAction::ReducePositions,
                        auto_action_available: true,
                        timestamp: chrono::Utc::now(),
                    });
                }
            }
        }
        
        Ok(breaches)
    }
    
    /// Check sector concentration limits
    async fn check_sector_limits(&self, portfolio: &Portfolio) -> Result<Vec<LimitBreach>> {
        let mut breaches = Vec::new();
        let portfolio_value = portfolio.total_market_value;
        
        if portfolio_value <= 0.0 {
            return Ok(breaches);
        }
        
        // Calculate sector allocations
        let mut sector_allocations: HashMap<String, f64> = HashMap::new();
        for position in &portfolio.positions {
            if let Some(sector) = &position.sector {
                let allocation = (position.market_value.abs() / portfolio_value) * 100.0;
                *sector_allocations.entry(sector.clone()).or_insert(0.0) += allocation;
            }
        }
        
        // Check sector limits
        for (sector, allocation) in sector_allocations {
            let limit = self.config.sector_limits.max_sector_allocation
                .get(&sector)
                .copied()
                .unwrap_or(self.config.sector_limits.default_sector_limit_percent);
            
            if allocation > limit {
                let breach_percentage = (allocation / limit - 1.0) * 100.0;
                let severity = self.calculate_breach_severity(breach_percentage);
                
                breaches.push(LimitBreach {
                    id: Uuid::new_v4(),
                    limit_type: LimitType::SectorLimit,
                    portfolio_id: portfolio.id,
                    position_symbol: None,
                    current_value: allocation,
                    limit_value: limit,
                    breach_percentage,
                    severity,
                    recommended_action: EnforcementAction::ReducePositions,
                    auto_action_available: true,
                    timestamp: chrono::Utc::now(),
                });
            }
        }
        
        Ok(breaches)
    }
    
    /// Check leverage limits
    async fn check_leverage_limits(&self, portfolio: &Portfolio, _risk_metrics: &RiskMetrics) -> Result<Vec<LimitBreach>> {
        let mut breaches = Vec::new();
        
        // Calculate gross leverage
        let gross_exposure: f64 = portfolio.positions
            .iter()
            .map(|p| p.market_value.abs())
            .sum();
        
        let net_asset_value = portfolio.total_market_value + portfolio.cash;
        
        if net_asset_value <= 0.0 {
            return Ok(breaches);
        }
        
        let gross_leverage = gross_exposure / net_asset_value;
        
        if gross_leverage > self.config.leverage_limits.max_gross_leverage {
            let breach_percentage = (gross_leverage / self.config.leverage_limits.max_gross_leverage - 1.0) * 100.0;
            let severity = self.calculate_breach_severity(breach_percentage);
            
            breaches.push(LimitBreach {
                id: Uuid::new_v4(),
                limit_type: LimitType::LeverageLimit,
                portfolio_id: portfolio.id,
                position_symbol: None,
                current_value: gross_leverage,
                limit_value: self.config.leverage_limits.max_gross_leverage,
                breach_percentage,
                severity,
                recommended_action: EnforcementAction::ReduceLeverage,
                auto_action_available: true,
                timestamp: chrono::Utc::now(),
            });
        }
        
        Ok(breaches)
    }
    
    /// Check drawdown limits
    async fn check_drawdown_limits(&self, portfolio: &Portfolio, risk_metrics: &RiskMetrics) -> Result<Vec<LimitBreach>> {
        let mut breaches = Vec::new();
        
        // Check maximum drawdown
        if risk_metrics.max_drawdown > self.config.drawdown_limits.max_total_drawdown_percent {
            let breach_percentage = (risk_metrics.max_drawdown / self.config.drawdown_limits.max_total_drawdown_percent - 1.0) * 100.0;
            let severity = self.calculate_breach_severity(breach_percentage);
            
            breaches.push(LimitBreach {
                id: Uuid::new_v4(),
                limit_type: LimitType::DrawdownLimit,
                portfolio_id: portfolio.id,
                position_symbol: None,
                current_value: risk_metrics.max_drawdown,
                limit_value: self.config.drawdown_limits.max_total_drawdown_percent,
                breach_percentage,
                severity,
                recommended_action: if severity == BreachSeverity::Emergency {
                    EnforcementAction::EmergencyStop
                } else {
                    EnforcementAction::ReducePositions
                },
                auto_action_available: severity != BreachSeverity::Emergency,
                timestamp: chrono::Utc::now(),
            });
        }
        
        Ok(breaches)
    }
    
    /// Check liquidity limits
    async fn check_liquidity_limits(&self, portfolio: &Portfolio, risk_metrics: &RiskMetrics) -> Result<Vec<LimitBreach>> {
        let mut breaches = Vec::new();
        
        // Check minimum liquidity score
        if risk_metrics.liquidity_risk < self.config.liquidity_limits.min_portfolio_liquidity_score {
            let breach_percentage = (self.config.liquidity_limits.min_portfolio_liquidity_score / risk_metrics.liquidity_risk - 1.0) * 100.0;
            let severity = self.calculate_breach_severity(breach_percentage);
            
            breaches.push(LimitBreach {
                id: Uuid::new_v4(),
                limit_type: LimitType::LiquidityLimit,
                portfolio_id: portfolio.id,
                position_symbol: None,
                current_value: risk_metrics.liquidity_risk,
                limit_value: self.config.liquidity_limits.min_portfolio_liquidity_score,
                breach_percentage,
                severity,
                recommended_action: EnforcementAction::ReducePositions,
                auto_action_available: true,
                timestamp: chrono::Utc::now(),
            });
        }
        
        Ok(breaches)
    }
    
    /// Calculate breach severity based on percentage over limit
    fn calculate_breach_severity(&self, breach_percentage: f64) -> BreachSeverity {
        if breach_percentage > 20.0 {
            BreachSeverity::Emergency
        } else if breach_percentage > 0.0 {
            BreachSeverity::Critical
        } else {
            BreachSeverity::Warning
        }
    }
    
    /// Trigger enforcement action for a limit breach
    async fn trigger_enforcement(&self, breach: &LimitBreach) -> Result<()> {
        let start_time = Instant::now();
        
        // Check if manual approval is required
        if self.config.enforcement_config.require_manual_approval.contains(&breach.recommended_action) {
            // Add to pending approvals
            let mut pending = self.pending_approvals.write().await;
            pending.insert(breach.id, breach.clone());
            
            // Send alert for manual review
            let alert = RiskAlert {
                id: Uuid::new_v4(),
                level: match breach.severity {
                    BreachSeverity::Emergency => AlertLevel::Emergency,
                    BreachSeverity::Critical => AlertLevel::Critical,
                    BreachSeverity::Warning => AlertLevel::Warning,
                },
                title: "Manual Approval Required".to_string(),
                description: format!("Limit breach requires manual approval: {:?}", breach.limit_type),
                metric_name: format!("{:?}", breach.limit_type),
                current_value: breach.current_value,
                threshold_value: breach.limit_value,
                portfolio_id: Some(breach.portfolio_id),
                position_symbol: breach.position_symbol.clone(),
                recommended_action: format!("{:?}", breach.recommended_action),
                auto_action_taken: false,
                timestamp: chrono::Utc::now(),
            };
            
            self.alert_sender.send(alert)?;
            return Ok(());
        }
        
        // Execute automatic enforcement
        let result = self.execute_enforcement_action(breach).await?;
        
        let enforcement_time = start_time.elapsed();
        
        // Check enforcement latency target
        if enforcement_time > self.config.enforcement_config.max_enforcement_latency {
            warn!(
                "Enforcement action took {:?}, exceeding target of {:?}",
                enforcement_time, self.config.enforcement_config.max_enforcement_latency
            );
        }
        
        // Store enforcement result
        {
            let mut history = self.enforcement_history.write().await;
            history.push(result);
        }
        
        // Send enforcement action
        self.action_sender.send(breach.recommended_action.clone())?;
        
        Ok(())
    }
    
    /// Execute enforcement action
    async fn execute_enforcement_action(&self, breach: &LimitBreach) -> Result<EnforcementResult> {
        let start_time = Instant::now();
        
        let result = match breach.recommended_action {
            EnforcementAction::SendAlert => {
                // Alert already handled in trigger_enforcement
                EnforcementResult {
                    action_id: Uuid::new_v4(),
                    action_type: EnforcementAction::SendAlert,
                    success: true,
                    positions_affected: vec![],
                    amount_reduced: 0.0,
                    execution_time: start_time.elapsed(),
                    error_message: None,
                    timestamp: chrono::Utc::now(),
                }
            }
            EnforcementAction::ReducePositions => {
                // In production, this would interface with the trading system
                EnforcementResult {
                    action_id: Uuid::new_v4(),
                    action_type: EnforcementAction::ReducePositions,
                    success: true,
                    positions_affected: breach.position_symbol.iter().cloned().collect(),
                    amount_reduced: breach.current_value - breach.limit_value,
                    execution_time: start_time.elapsed(),
                    error_message: None,
                    timestamp: chrono::Utc::now(),
                }
            }
            EnforcementAction::ClosePosition => {
                EnforcementResult {
                    action_id: Uuid::new_v4(),
                    action_type: EnforcementAction::ClosePosition,
                    success: true,
                    positions_affected: breach.position_symbol.iter().cloned().collect(),
                    amount_reduced: breach.current_value,
                    execution_time: start_time.elapsed(),
                    error_message: None,
                    timestamp: chrono::Utc::now(),
                }
            }
            EnforcementAction::EmergencyStop => {
                EnforcementResult {
                    action_id: Uuid::new_v4(),
                    action_type: EnforcementAction::EmergencyStop,
                    success: true,
                    positions_affected: vec!["ALL".to_string()],
                    amount_reduced: 100.0,
                    execution_time: start_time.elapsed(),
                    error_message: None,
                    timestamp: chrono::Utc::now(),
                }
            }
            _ => {
                EnforcementResult {
                    action_id: Uuid::new_v4(),
                    action_type: breach.recommended_action.clone(),
                    success: false,
                    positions_affected: vec![],
                    amount_reduced: 0.0,
                    execution_time: start_time.elapsed(),
                    error_message: Some("Action not implemented".to_string()),
                    timestamp: chrono::Utc::now(),
                }
            }
        };
        
        Ok(result)
    }
    
    /// Get breach history
    pub async fn get_breach_history(&self) -> Result<Vec<LimitBreach>> {
        let history = self.breach_history.read().await;
        Ok(history.clone())
    }
    
    /// Get enforcement history
    pub async fn get_enforcement_history(&self) -> Result<Vec<EnforcementResult>> {
        let history = self.enforcement_history.read().await;
        Ok(history.clone())
    }
    
    /// Get pending approvals
    pub async fn get_pending_approvals(&self) -> Result<Vec<LimitBreach>> {
        let pending = self.pending_approvals.read().await;
        Ok(pending.values().cloned().collect())
    }
    
    /// Approve pending enforcement action
    pub async fn approve_enforcement(&self, breach_id: Uuid) -> Result<()> {
        let breach = {
            let mut pending = self.pending_approvals.write().await;
            pending.remove(&breach_id)
        };
        
        if let Some(breach) = breach {
            self.execute_enforcement_action(&breach).await?;
        }
        
        Ok(())
    }
    
    /// Get performance metrics
    pub async fn get_performance_metrics(&self) -> Result<EnforcementPerformanceMetrics> {
        let tracker = self.performance_tracker.read().await;
        Ok(tracker.get_metrics())
    }
}

/// Performance tracker for enforcement
#[derive(Debug)]
struct EnforcementPerformanceTracker {
    total_checks: u64,
    total_breaches: u64,
    total_check_time: Duration,
    max_check_time: Duration,
    start_time: Option<Instant>,
}

impl EnforcementPerformanceTracker {
    fn new() -> Self {
        Self {
            total_checks: 0,
            total_breaches: 0,
            total_check_time: Duration::from_nanos(0),
            max_check_time: Duration::from_nanos(0),
            start_time: None,
        }
    }
    
    fn start_check(&mut self) {
        self.start_time = Some(Instant::now());
    }
    
    fn end_check(&mut self, duration: Duration, breach_count: usize) {
        self.total_checks += 1;
        self.total_breaches += breach_count as u64;
        self.total_check_time += duration;
        
        if duration > self.max_check_time {
            self.max_check_time = duration;
        }
        
        self.start_time = None;
    }
    
    fn get_metrics(&self) -> EnforcementPerformanceMetrics {
        let avg_check_time = if self.total_checks > 0 {
            self.total_check_time / self.total_checks as u32
        } else {
            Duration::from_nanos(0)
        };
        
        EnforcementPerformanceMetrics {
            total_checks: self.total_checks,
            total_breaches: self.total_breaches,
            avg_check_time,
            max_check_time: self.max_check_time,
            breach_rate: if self.total_checks > 0 {
                self.total_breaches as f64 / self.total_checks as f64
            } else {
                0.0
            },
        }
    }
}

/// Performance metrics for enforcement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementPerformanceMetrics {
    pub total_checks: u64,
    pub total_breaches: u64,
    pub avg_check_time: Duration,
    pub max_check_time: Duration,
    pub breach_rate: f64,
}