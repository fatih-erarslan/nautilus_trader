//! Barbell strategy implementation
//! 
//! This module implements Nassim Nicholas Taleb's Barbell Strategy,
//! which combines extremely safe and extremely risky investments.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Parameters for Barbell strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarbellParams {
    /// Allocation to safe assets (0.0 to 1.0)
    pub safe_allocation: f64,
    /// Allocation to risky assets (0.0 to 1.0)
    pub risky_allocation: f64,
    /// Minimum safe allocation (safety constraint)
    pub min_safe_allocation: f64,
    /// Maximum risky allocation (risk constraint)
    pub max_risky_allocation: f64,
    /// Rebalancing threshold (trigger rebalancing when deviation exceeds this)
    pub rebalancing_threshold: f64,
    /// Enable dynamic allocation based on market conditions
    pub enable_dynamic_allocation: bool,
    /// Risk budget for risky allocation
    pub risk_budget: f64,
}

impl Default for BarbellParams {
    fn default() -> Self {
        Self {
            safe_allocation: 0.85,
            risky_allocation: 0.15,
            min_safe_allocation: 0.8,
            max_risky_allocation: 0.2,
            rebalancing_threshold: 0.05,
            enable_dynamic_allocation: true,
            risk_budget: 0.02, // 2% of portfolio value at risk
        }
    }
}

/// Strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Strategy identifier
    pub strategy_id: String,
    /// Target return for the strategy
    pub target_return: f64,
    /// Maximum acceptable drawdown
    pub max_drawdown: f64,
    /// Rebalancing frequency in days
    pub rebalancing_frequency_days: u32,
    /// Transaction cost rate
    pub transaction_cost_rate: f64,
    /// Enable position sizing optimization
    pub enable_position_sizing: bool,
    /// Enable tail risk hedging
    pub enable_tail_hedging: bool,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            strategy_id: "barbell_strategy".to_string(),
            target_return: 0.08, // 8% annual return
            max_drawdown: 0.15,
            rebalancing_frequency_days: 30,
            transaction_cost_rate: 0.001, // 0.1% transaction cost
            enable_position_sizing: true,
            enable_tail_hedging: true,
        }
    }
}

/// Asset class definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetClass {
    pub name: String,
    pub asset_type: AssetType,
    pub expected_return: f64,
    pub volatility: f64,
    pub max_drawdown: f64,
    pub liquidity_score: f64, // 0.0 to 1.0
    pub correlation_to_market: f64,
}

/// Types of assets in the barbell
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssetType {
    Safe,
    Moderate,
    Volatile,
    Risky,
    Derivative,
    Antifragile,
    Alternative,
    Hedge,
}

/// Risk levels for risky assets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Moderate,
    High,
    Extreme,
}

/// Types of hedging instruments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HedgeType {
    TailRiskHedge,
    VolatilityHedge,
    InflationHedge,
    CurrencyHedge,
}

/// Portfolio allocation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationResult {
    /// Asset allocations
    pub allocations: HashMap<String, f64>,
    /// Total safe allocation
    pub total_safe_allocation: f64,
    /// Total risky allocation
    pub total_risky_allocation: f64,
    /// Expected portfolio return
    pub expected_return: f64,
    /// Expected portfolio volatility
    pub expected_volatility: f64,
    /// Risk budget utilization
    pub risk_budget_utilization: f64,
    /// Rebalancing required
    pub rebalancing_required: bool,
    /// Allocation timestamp
    pub timestamp: DateTime<Utc>,
}

/// Barbell strategy implementation
#[derive(Debug, Clone)]
pub struct BarbellStrategy {
    strategy_id: String,
    config: StrategyConfig,
    params: BarbellParams,
    safe_assets: Vec<AssetClass>,
    risky_assets: Vec<AssetClass>,
    hedge_assets: Vec<AssetClass>,
    current_allocation: HashMap<String, f64>,
    allocation_history: Vec<AllocationResult>,
    performance_history: Vec<PerformanceRecord>,
    last_rebalance: Option<DateTime<Utc>>,
}

/// Performance record for tracking strategy performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecord {
    pub timestamp: DateTime<Utc>,
    pub portfolio_value: f64,
    pub return_since_inception: f64,
    pub return_since_last_rebalance: f64,
    pub volatility: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub safe_contribution: f64,
    pub risky_contribution: f64,
}

/// Rebalancing signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalancingSignal {
    pub signal_strength: f64,
    pub reason: RebalancingReason,
    pub recommended_action: RebalancingAction,
    pub urgency: SignalUrgency,
    pub estimated_impact: f64,
}

/// Reasons for rebalancing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RebalancingReason {
    AllocationDrift,
    RiskBudgetExceeded,
    MarketRegimeChange,
    ScheduledRebalancing,
    TailRiskElevated,
    OpportunityDetected,
}

/// Rebalancing actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RebalancingAction {
    IncreaseSafeAllocation { by: f64 },
    IncreaseRiskyAllocation { by: f64 },
    RebalanceToTarget,
    EmergencyDeRisk,
    NoAction,
}

/// Signal urgency levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalUrgency {
    Low,
    Medium,
    High,
    Critical,
}

impl BarbellStrategy {
    /// Create a new Barbell strategy
    pub fn new(
        strategy_id: String,
        config: StrategyConfig,
        params: BarbellParams,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Validate parameters
        if params.safe_allocation + params.risky_allocation > 1.0 {
            return Err("Safe and risky allocations cannot exceed 100%".into());
        }

        if params.safe_allocation < params.min_safe_allocation {
            return Err("Safe allocation below minimum threshold".into());
        }

        if params.risky_allocation > params.max_risky_allocation {
            return Err("Risky allocation above maximum threshold".into());
        }

        Ok(Self {
            strategy_id,
            config,
            params,
            safe_assets: Self::create_default_safe_assets(),
            risky_assets: Self::create_default_risky_assets(),
            hedge_assets: Self::create_default_hedge_assets(),
            current_allocation: HashMap::new(),
            allocation_history: Vec::new(),
            performance_history: Vec::new(),
            last_rebalance: None,
        })
    }

    /// Add safe asset to the barbell
    pub fn add_safe_asset(&mut self, asset: AssetClass) -> Result<(), Box<dyn std::error::Error>> {
        match asset.asset_type {
            AssetType::Safe | AssetType::Moderate => {
                self.safe_assets.push(asset);
                Ok(())
            }
            _ => Err("Asset is not classified as safe".into()),
        }
    }

    /// Add risky asset to the barbell
    pub fn add_risky_asset(&mut self, asset: AssetClass) -> Result<(), Box<dyn std::error::Error>> {
        match asset.asset_type {
            AssetType::Risky | AssetType::Volatile | AssetType::Derivative | AssetType::Antifragile | AssetType::Alternative => {
                self.risky_assets.push(asset);
                Ok(())
            }
            _ => Err("Asset is not classified as risky".into()),
        }
    }

    /// Calculate optimal allocation
    pub fn calculate_allocation(&mut self, market_conditions: &MarketConditions) -> Result<AllocationResult, Box<dyn std::error::Error>> {
        let mut allocations = HashMap::new();
        
        // Adjust allocations based on market conditions if dynamic allocation is enabled
        let (safe_alloc, risky_alloc) = if self.params.enable_dynamic_allocation {
            self.calculate_dynamic_allocation(market_conditions)?
        } else {
            (self.params.safe_allocation, self.params.risky_allocation)
        };

        // Allocate among safe assets
        let safe_allocation_per_asset = safe_alloc / self.safe_assets.len() as f64;
        for asset in &self.safe_assets {
            allocations.insert(asset.name.clone(), safe_allocation_per_asset);
        }

        // Allocate among risky assets using risk budgeting
        self.allocate_risky_assets(&mut allocations, risky_alloc)?;

        // Add hedge positions if enabled
        if self.config.enable_tail_hedging {
            self.add_hedge_positions(&mut allocations, market_conditions)?;
        }

        // Calculate portfolio metrics
        let expected_return = self.calculate_expected_return(&allocations)?;
        let expected_volatility = self.calculate_expected_volatility(&allocations)?;
        let risk_budget_utilization = self.calculate_risk_budget_utilization(&allocations)?;

        // Check if rebalancing is required
        let rebalancing_required = self.is_rebalancing_required(&allocations)?;

        let result = AllocationResult {
            allocations: allocations.clone(),
            total_safe_allocation: safe_alloc,
            total_risky_allocation: risky_alloc,
            expected_return,
            expected_volatility,
            risk_budget_utilization,
            rebalancing_required,
            timestamp: Utc::now(),
        };

        self.current_allocation = allocations;
        self.allocation_history.push(result.clone());

        Ok(result)
    }

    /// Check for rebalancing signals
    pub fn check_rebalancing_signals(&self, market_conditions: &MarketConditions) -> Result<Vec<RebalancingSignal>, Box<dyn std::error::Error>> {
        let mut signals = Vec::new();

        // Check allocation drift
        if let Some(signal) = self.check_allocation_drift()? {
            signals.push(signal);
        }

        // Check risk budget utilization
        if let Some(signal) = self.check_risk_budget()? {
            signals.push(signal);
        }

        // Check scheduled rebalancing
        if let Some(signal) = self.check_scheduled_rebalancing()? {
            signals.push(signal);
        }

        // Check market regime changes
        if let Some(signal) = self.check_market_regime_change(market_conditions)? {
            signals.push(signal);
        }

        // Check tail risk elevation
        if let Some(signal) = self.check_tail_risk_elevation(market_conditions)? {
            signals.push(signal);
        }

        Ok(signals)
    }

    /// Execute rebalancing
    pub fn execute_rebalancing(&mut self, target_allocation: HashMap<String, f64>) -> Result<RebalancingResult, Box<dyn std::error::Error>> {
        let mut trades = Vec::new();
        let mut total_transaction_cost = 0.0;

        // Calculate required trades
        for (asset_name, target_weight) in &target_allocation {
            let current_weight = self.current_allocation.get(asset_name).unwrap_or(&0.0);
            let trade_amount = target_weight - current_weight;

            if trade_amount.abs() > 0.001 { // Minimum trade threshold
                let transaction_cost = trade_amount.abs() * self.config.transaction_cost_rate;
                total_transaction_cost += transaction_cost;

                trades.push(Trade {
                    asset_name: asset_name.clone(),
                    trade_amount,
                    transaction_cost,
                });
            }
        }

        // Update current allocation
        self.current_allocation = target_allocation;
        self.last_rebalance = Some(Utc::now());

        Ok(RebalancingResult {
            trades,
            total_transaction_cost,
            rebalancing_timestamp: Utc::now(),
        })
    }

    /// Get current portfolio metrics
    pub fn get_portfolio_metrics(&self) -> Result<PortfolioMetrics, Box<dyn std::error::Error>> {
        if self.current_allocation.is_empty() {
            return Err("No current allocation available".into());
        }

        let expected_return = self.calculate_expected_return(&self.current_allocation)?;
        let expected_volatility = self.calculate_expected_volatility(&self.current_allocation)?;
        let safe_allocation = self.calculate_safe_allocation_total();
        let risky_allocation = self.calculate_risky_allocation_total();

        Ok(PortfolioMetrics {
            expected_return,
            expected_volatility,
            safe_allocation,
            risky_allocation,
            diversification_ratio: self.calculate_diversification_ratio()?,
            risk_budget_utilization: self.calculate_risk_budget_utilization(&self.current_allocation)?,
            last_rebalance: self.last_rebalance,
        })
    }

    // Private helper methods

    /// Calculate dynamic allocation based on market conditions
    fn calculate_dynamic_allocation(&self, market_conditions: &MarketConditions) -> Result<(f64, f64), Box<dyn std::error::Error>> {
        let mut safe_multiplier = 1.0;
        let mut risky_multiplier = 1.0;

        // Adjust based on volatility
        if market_conditions.volatility_level > 0.25 {
            safe_multiplier += 0.1;
            risky_multiplier -= 0.1;
        }

        // Adjust based on tail risk
        if market_conditions.risk_off_mode {
            safe_multiplier += 0.2;
            risky_multiplier -= 0.2;
        }

        let safe_alloc = (self.params.safe_allocation * safe_multiplier)
            .max(self.params.min_safe_allocation)
            .min(1.0);
        
        let risky_alloc = (self.params.risky_allocation * risky_multiplier)
            .max(0.0)
            .min(self.params.max_risky_allocation)
            .min(1.0 - safe_alloc);

        Ok((safe_alloc, risky_alloc))
    }

    /// Allocate risky assets using risk budgeting
    fn allocate_risky_assets(&self, allocations: &mut HashMap<String, f64>, total_risky_alloc: f64) -> Result<(), Box<dyn std::error::Error>> {
        if self.risky_assets.is_empty() {
            return Ok(());
        }

        // Equal risk budgeting approach (simplified)
        let risk_budget_per_asset = self.params.risk_budget / self.risky_assets.len() as f64;
        
        for asset in &self.risky_assets {
            // Calculate allocation based on inverse volatility weighting
            let weight = risk_budget_per_asset / asset.volatility;
            let normalized_weight = weight * total_risky_alloc / self.risky_assets.len() as f64;
            allocations.insert(asset.name.clone(), normalized_weight);
        }

        Ok(())
    }

    /// Add hedge positions to the allocation
    fn add_hedge_positions(&self, allocations: &mut HashMap<String, f64>, market_conditions: &MarketConditions) -> Result<(), Box<dyn std::error::Error>> {
        let hedge_budget = 0.05; // 5% for hedging

        if market_conditions.volatility_level > 0.3 {
            // Add volatility hedge
            for asset in &self.hedge_assets {
                if matches!(asset.asset_type, AssetType::Hedge) {
                    allocations.insert(asset.name.clone(), hedge_budget / 2.0);
                }
            }
        }

        if market_conditions.sentiment_extreme {
            // Add tail risk hedge
            for asset in &self.hedge_assets {
                if matches!(asset.asset_type, AssetType::Hedge) {
                    allocations.insert(asset.name.clone(), hedge_budget / 2.0);
                }
            }
        }

        Ok(())
    }

    /// Calculate expected portfolio return
    fn calculate_expected_return(&self, allocations: &HashMap<String, f64>) -> Result<f64, Box<dyn std::error::Error>> {
        let mut total_return = 0.0;

        for (asset_name, &weight) in allocations {
            let asset_return = self.get_asset_expected_return(asset_name)?;
            total_return += weight * asset_return;
        }

        Ok(total_return)
    }

    /// Calculate expected portfolio volatility (simplified)
    fn calculate_expected_volatility(&self, allocations: &HashMap<String, f64>) -> Result<f64, Box<dyn std::error::Error>> {
        let mut total_variance = 0.0;

        for (asset_name, &weight) in allocations {
            let asset_volatility = self.get_asset_volatility(asset_name)?;
            total_variance += (weight * asset_volatility).powi(2);
        }

        Ok(total_variance.sqrt())
    }

    /// Calculate risk budget utilization
    fn calculate_risk_budget_utilization(&self, allocations: &HashMap<String, f64>) -> Result<f64, Box<dyn std::error::Error>> {
        let portfolio_volatility = self.calculate_expected_volatility(allocations)?;
        let risk_budget_volatility = self.params.risk_budget.sqrt();
        
        Ok(portfolio_volatility / risk_budget_volatility)
    }

    /// Check if rebalancing is required
    fn is_rebalancing_required(&self, new_allocations: &HashMap<String, f64>) -> Result<bool, Box<dyn std::error::Error>> {
        for (asset_name, &new_weight) in new_allocations {
            let current_weight = self.current_allocation.get(asset_name).unwrap_or(&0.0);
            if (new_weight - current_weight).abs() > self.params.rebalancing_threshold {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Get asset expected return
    fn get_asset_expected_return(&self, asset_name: &str) -> Result<f64, Box<dyn std::error::Error>> {
        for asset in &self.safe_assets {
            if asset.name == asset_name {
                return Ok(asset.expected_return);
            }
        }
        for asset in &self.risky_assets {
            if asset.name == asset_name {
                return Ok(asset.expected_return);
            }
        }
        for asset in &self.hedge_assets {
            if asset.name == asset_name {
                return Ok(asset.expected_return);
            }
        }
        Err(format!("Asset {} not found", asset_name).into())
    }

    /// Get asset volatility
    fn get_asset_volatility(&self, asset_name: &str) -> Result<f64, Box<dyn std::error::Error>> {
        for asset in &self.safe_assets {
            if asset.name == asset_name {
                return Ok(asset.volatility);
            }
        }
        for asset in &self.risky_assets {
            if asset.name == asset_name {
                return Ok(asset.volatility);
            }
        }
        for asset in &self.hedge_assets {
            if asset.name == asset_name {
                return Ok(asset.volatility);
            }
        }
        Err(format!("Asset {} not found", asset_name).into())
    }

    /// Calculate total safe allocation
    fn calculate_safe_allocation_total(&self) -> f64 {
        self.safe_assets.iter()
            .map(|asset| self.current_allocation.get(&asset.name).unwrap_or(&0.0))
            .sum()
    }

    /// Calculate total risky allocation
    fn calculate_risky_allocation_total(&self) -> f64 {
        self.risky_assets.iter()
            .map(|asset| self.current_allocation.get(&asset.name).unwrap_or(&0.0))
            .sum()
    }

    /// Calculate diversification ratio
    fn calculate_diversification_ratio(&self) -> Result<f64, Box<dyn std::error::Error>> {
        // Simplified diversification ratio calculation
        let num_assets = self.current_allocation.len();
        if num_assets <= 1 {
            return Ok(1.0);
        }

        // Effective number of assets (inverse of Herfindahl index)
        let herfindahl_index: f64 = self.current_allocation.values()
            .map(|&weight| weight.powi(2))
            .sum();
        
        let effective_assets = 1.0 / herfindahl_index;
        Ok(effective_assets / num_assets as f64)
    }

    // Rebalancing signal check methods

    /// Check allocation drift
    fn check_allocation_drift(&self) -> Result<Option<RebalancingSignal>, Box<dyn std::error::Error>> {
        let safe_current = self.calculate_safe_allocation_total();
        let safe_target = self.params.safe_allocation;
        let drift = (safe_current - safe_target).abs();

        if drift > self.params.rebalancing_threshold {
            return Ok(Some(RebalancingSignal {
                signal_strength: (drift / self.params.rebalancing_threshold).min(1.0),
                reason: RebalancingReason::AllocationDrift,
                recommended_action: RebalancingAction::RebalanceToTarget,
                urgency: if drift > self.params.rebalancing_threshold * 2.0 {
                    SignalUrgency::High
                } else {
                    SignalUrgency::Medium
                },
                estimated_impact: drift,
            }));
        }

        Ok(None)
    }

    /// Check risk budget utilization
    fn check_risk_budget(&self) -> Result<Option<RebalancingSignal>, Box<dyn std::error::Error>> {
        let utilization = self.calculate_risk_budget_utilization(&self.current_allocation)?;
        
        if utilization > 1.2 { // 20% over budget
            return Ok(Some(RebalancingSignal {
                signal_strength: ((utilization - 1.0) / 0.2).min(1.0),
                reason: RebalancingReason::RiskBudgetExceeded,
                recommended_action: RebalancingAction::IncreaseSafeAllocation { by: 0.05 },
                urgency: SignalUrgency::High,
                estimated_impact: utilization - 1.0,
            }));
        }

        Ok(None)
    }

    /// Check scheduled rebalancing
    fn check_scheduled_rebalancing(&self) -> Result<Option<RebalancingSignal>, Box<dyn std::error::Error>> {
        if let Some(last_rebalance) = self.last_rebalance {
            let days_since_rebalance = Utc::now().signed_duration_since(last_rebalance).num_days();
            
            if days_since_rebalance >= self.config.rebalancing_frequency_days as i64 {
                return Ok(Some(RebalancingSignal {
                    signal_strength: (days_since_rebalance as f64 / self.config.rebalancing_frequency_days as f64).min(1.0),
                    reason: RebalancingReason::ScheduledRebalancing,
                    recommended_action: RebalancingAction::RebalanceToTarget,
                    urgency: SignalUrgency::Low,
                    estimated_impact: 0.01, // Minimal impact for scheduled rebalancing
                }));
            }
        }

        Ok(None)
    }

    /// Check market regime change
    fn check_market_regime_change(&self, market_conditions: &MarketConditions) -> Result<Option<RebalancingSignal>, Box<dyn std::error::Error>> {
        if market_conditions.volatility_level > 0.35 && market_conditions.risk_off_mode {
            return Ok(Some(RebalancingSignal {
                signal_strength: market_conditions.volatility_level / 0.5,
                reason: RebalancingReason::MarketRegimeChange,
                recommended_action: RebalancingAction::IncreaseSafeAllocation { by: 0.1 },
                urgency: SignalUrgency::High,
                estimated_impact: 0.05,
            }));
        }

        Ok(None)
    }

    /// Check tail risk elevation
    fn check_tail_risk_elevation(&self, market_conditions: &MarketConditions) -> Result<Option<RebalancingSignal>, Box<dyn std::error::Error>> {
        if market_conditions.correlation_breakdown && market_conditions.liquidity_level < 0.3 {
            return Ok(Some(RebalancingSignal {
                signal_strength: 1.0 - market_conditions.liquidity_level,
                reason: RebalancingReason::TailRiskElevated,
                recommended_action: RebalancingAction::EmergencyDeRisk,
                urgency: SignalUrgency::Critical,
                estimated_impact: 0.1,
            }));
        }

        Ok(None)
    }

    /// Create default safe assets
    fn create_default_safe_assets() -> Vec<AssetClass> {
        vec![
            AssetClass {
                name: "Treasury Bills".to_string(),
                asset_type: AssetType::Safe,
                expected_return: 0.02,
                volatility: 0.01,
                max_drawdown: 0.001,
                liquidity_score: 1.0,
                correlation_to_market: 0.1,
            },
            AssetClass {
                name: "High Grade Bonds".to_string(),
                asset_type: AssetType::Safe,
                expected_return: 0.04,
                volatility: 0.05,
                max_drawdown: 0.02,
                liquidity_score: 0.9,
                correlation_to_market: 0.2,
            },
        ]
    }

    /// Create default risky assets
    fn create_default_risky_assets() -> Vec<AssetClass> {
        vec![
            AssetClass {
                name: "Technology Stocks".to_string(),
                asset_type: AssetType::Volatile,
                expected_return: 0.15,
                volatility: 0.35,
                max_drawdown: 0.50,
                liquidity_score: 0.8,
                correlation_to_market: 0.9,
            },
            AssetClass {
                name: "Cryptocurrency".to_string(),
                asset_type: AssetType::Volatile,
                expected_return: 0.25,
                volatility: 0.80,
                max_drawdown: 0.80,
                liquidity_score: 0.6,
                correlation_to_market: 0.3,
            },
        ]
    }

    /// Create default hedge assets
    fn create_default_hedge_assets() -> Vec<AssetClass> {
        vec![
            AssetClass {
                name: "VIX Options".to_string(),
                asset_type: AssetType::Hedge,
                expected_return: -0.05,
                volatility: 0.50,
                max_drawdown: 0.60,
                liquidity_score: 0.7,
                correlation_to_market: -0.8,
            },
        ]
    }
}

/// Portfolio metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioMetrics {
    pub expected_return: f64,
    pub expected_volatility: f64,
    pub safe_allocation: f64,
    pub risky_allocation: f64,
    pub diversification_ratio: f64,
    pub risk_budget_utilization: f64,
    pub last_rebalance: Option<DateTime<Utc>>,
}

/// Trade execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub asset_name: String,
    pub trade_amount: f64,
    pub transaction_cost: f64,
}

/// Rebalancing execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalancingResult {
    pub trades: Vec<Trade>,
    pub total_transaction_cost: f64,
    pub rebalancing_timestamp: DateTime<Utc>,
}

/// Market conditions for strategy input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub volatility_level: f64,
    pub liquidity_level: f64,
    pub correlation_breakdown: bool,
    pub sentiment_extreme: bool,
    pub risk_off_mode: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barbell_strategy_creation() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        
        let strategy = BarbellStrategy::new("test_strategy".to_string(), config, params).unwrap();
        assert_eq!(strategy.strategy_id, "test_strategy");
        assert_eq!(strategy.safe_assets.len(), 2);
        assert_eq!(strategy.risky_assets.len(), 2);
    }

    #[test]
    fn test_invalid_allocation_parameters() {
        let config = StrategyConfig::default();
        let mut params = BarbellParams::default();
        params.safe_allocation = 0.9;
        params.risky_allocation = 0.2; // Total > 1.0
        
        let result = BarbellStrategy::new("test".to_string(), config, params);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_safe_asset() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let mut strategy = BarbellStrategy::new("test".to_string(), config, params).unwrap();
        
        let safe_asset = AssetClass {
            name: "Corporate Bonds".to_string(),
            asset_type: AssetType::Safe,
            expected_return: 0.05,
            volatility: 0.08,
            max_drawdown: 0.05,
            liquidity_score: 0.85,
            correlation_to_market: 0.3,
        };
        
        let result = strategy.add_safe_asset(safe_asset);
        assert!(result.is_ok());
        assert_eq!(strategy.safe_assets.len(), 3);
    }

    #[test]
    fn test_calculate_allocation() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let mut strategy = BarbellStrategy::new("test".to_string(), config, params).unwrap();
        
        let market_conditions = MarketConditions {
            volatility_level: 0.2,
            liquidity_level: 0.8,
            correlation_breakdown: false,
            sentiment_extreme: false,
            risk_off_mode: false,
        };
        
        let result = strategy.calculate_allocation(&market_conditions).unwrap();
        assert!(result.total_safe_allocation > 0.0);
        assert!(result.total_risky_allocation > 0.0);
        assert!(result.expected_return > 0.0);
        assert!(result.expected_volatility > 0.0);
    }

    #[test]
    fn test_dynamic_allocation_stress_conditions() {
        let config = StrategyConfig::default();
        let mut params = BarbellParams::default();
        params.enable_dynamic_allocation = true;
        let mut strategy = BarbellStrategy::new("test".to_string(), config, params).unwrap();
        
        let stress_conditions = MarketConditions {
            volatility_level: 0.4, // High volatility
            liquidity_level: 0.3,  // Low liquidity
            correlation_breakdown: true,
            sentiment_extreme: true,
            risk_off_mode: true,
        };
        
        let result = strategy.calculate_allocation(&stress_conditions).unwrap();
        
        // Should increase safe allocation under stress
        assert!(result.total_safe_allocation > strategy.params.safe_allocation);
    }

    #[test]
    fn test_rebalancing_signals() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let strategy = BarbellStrategy::new("test".to_string(), config, params).unwrap();
        
        let market_conditions = MarketConditions {
            volatility_level: 0.4, // High volatility should trigger signals
            liquidity_level: 0.2,  // Low liquidity
            correlation_breakdown: true,
            sentiment_extreme: true,
            risk_off_mode: true,
        };
        
        let signals = strategy.check_rebalancing_signals(&market_conditions).unwrap();
        assert!(!signals.is_empty()); // Should detect some signals under stress
    }

    #[test]
    fn test_portfolio_metrics() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let mut strategy = BarbellStrategy::new("test".to_string(), config, params).unwrap();
        
        // Set up some allocation first
        let market_conditions = MarketConditions {
            volatility_level: 0.2,
            liquidity_level: 0.8,
            correlation_breakdown: false,
            sentiment_extreme: false,
            risk_off_mode: false,
        };
        
        strategy.calculate_allocation(&market_conditions).unwrap();
        
        let metrics = strategy.get_portfolio_metrics().unwrap();
        assert!(metrics.expected_return > 0.0);
        assert!(metrics.expected_volatility > 0.0);
        assert!(metrics.safe_allocation >= 0.0);
        assert!(metrics.risky_allocation >= 0.0);
        assert!(metrics.diversification_ratio > 0.0 && metrics.diversification_ratio <= 1.0);
    }

    #[test]
    fn test_risk_budget_utilization() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let strategy = BarbellStrategy::new("test".to_string(), config, params).unwrap();
        
        let mut allocations = HashMap::new();
        allocations.insert("Treasury Bills".to_string(), 0.85);
        allocations.insert("Technology Stocks".to_string(), 0.15);
        
        let utilization = strategy.calculate_risk_budget_utilization(&allocations).unwrap();
        assert!(utilization > 0.0);
    }
}