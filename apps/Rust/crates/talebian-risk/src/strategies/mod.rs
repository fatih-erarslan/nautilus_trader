//! Talebian trading and risk management strategies
//!
//! This module implements various strategies based on Nassim Taleb's philosophical
//! and practical approaches to risk management and trading.

pub mod barbell;
pub mod via_negativa;
pub mod convexity;
pub mod lindy;
pub mod hormesis;
pub mod ensemble;
pub mod skin_in_game;
pub mod option_strategy;

// Re-export key types
pub use barbell::*;
pub use via_negativa::*;
pub use convexity::*;
pub use lindy::*;
pub use hormesis::*;
pub use ensemble::*;
pub use skin_in_game::*;
pub use option_strategy::*;

use crate::barbell::AssetType;
use crate::error::TalebianResult as Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Base trait for all Talebian strategies
pub trait TalebianStrategy {
    /// Strategy type identifier
    fn strategy_type(&self) -> StrategyType;
    
    /// Calculate position sizes for given assets
    fn calculate_position_sizes(&self, assets: &[String], market_data: &MarketData) -> Result<HashMap<String, f64>>;
    
    /// Update strategy based on market conditions
    fn update_strategy(&mut self, market_data: &MarketData) -> Result<()>;
    
    /// Calculate expected return for the strategy
    fn expected_return(&self, market_data: &MarketData) -> Result<f64>;
    
    /// Calculate strategy risk metrics
    fn risk_metrics(&self, market_data: &MarketData) -> Result<StrategyRiskMetrics>;
    
    /// Get strategy performance attribution
    fn performance_attribution(&self, returns: &[f64]) -> Result<PerformanceAttribution>;
    
    /// Assess strategy robustness
    fn robustness_assessment(&self, scenarios: &[MarketScenario]) -> Result<RobustnessAssessment>;
    
    /// Get strategy configuration
    fn get_config(&self) -> &StrategyConfig;
    
    /// Update strategy configuration
    fn update_config(&mut self, config: StrategyConfig) -> Result<()>;
    
    /// Check if strategy is suitable for current market conditions
    fn is_suitable(&self, market_data: &MarketData) -> Result<bool>;
    
    /// Calculate strategy capacity (maximum AUM)
    fn calculate_capacity(&self, market_data: &MarketData) -> Result<f64>;
}

/// Types of Talebian strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StrategyType {
    /// Barbell strategy (safe + risky)
    Barbell,
    /// Via Negativa (elimination-based)
    ViaNegativa,
    /// Convexity-based strategy
    Convexity,
    /// Lindy effect strategy
    Lindy,
    /// Hormesis-based strategy
    Hormesis,
    /// Ensemble strategy
    Ensemble,
    /// Skin in the game strategy
    SkinInGame,
    /// Option-based strategy
    OptionStrategy,
}

/// Market data for strategy calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    /// Asset prices
    pub prices: HashMap<String, f64>,
    /// Asset returns
    pub returns: HashMap<String, Vec<f64>>,
    /// Asset volatilities
    pub volatilities: HashMap<String, f64>,
    /// Asset correlations
    pub correlations: HashMap<(String, String), f64>,
    /// Asset volumes
    pub volumes: HashMap<String, f64>,
    /// Asset types
    pub asset_types: HashMap<String, AssetType>,
    /// Market timestamp
    pub timestamp: DateTime<Utc>,
    /// Market regime information
    pub regime: MarketRegime,
}

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Normal market conditions
    Normal,
    /// Bull market
    Bull,
    /// Bear market
    Bear,
    /// High volatility
    HighVolatility,
    /// Low volatility
    LowVolatility,
    /// Crisis mode
    Crisis,
    /// Recovery mode
    Recovery,
}

/// Market scenario for robustness testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketScenario {
    /// Scenario name
    pub name: String,
    /// Asset price shocks
    pub price_shocks: HashMap<String, f64>,
    /// Volatility shocks
    pub volatility_shocks: HashMap<String, f64>,
    /// Correlation shocks
    pub correlation_shocks: HashMap<(String, String), f64>,
    /// Scenario probability
    pub probability: f64,
    /// Scenario duration
    pub duration_days: usize,
}

/// Strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Maximum position size per asset
    pub max_position_size: f64,
    /// Risk budget
    pub risk_budget: f64,
    /// Rebalancing frequency (days)
    pub rebalancing_frequency: usize,
    /// Minimum position size
    pub min_position_size: f64,
    /// Transaction costs
    pub transaction_costs: f64,
    /// Risk aversion parameter
    pub risk_aversion: f64,
    /// Strategy-specific parameters
    pub strategy_params: HashMap<String, f64>,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            max_position_size: 0.2,
            risk_budget: 0.1,
            rebalancing_frequency: 30,
            min_position_size: 0.01,
            transaction_costs: 0.001,
            risk_aversion: 2.0,
            strategy_params: HashMap::new(),
        }
    }
}

/// Strategy risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyRiskMetrics {
    /// Value at Risk (VaR)
    pub var_95: f64,
    /// Conditional Value at Risk (CVaR)
    pub cvar_95: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Volatility
    pub volatility: f64,
    /// Downside deviation
    pub downside_deviation: f64,
    /// Tail ratio
    pub tail_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Antifragility score
    pub antifragility_score: f64,
    /// Black swan probability
    pub black_swan_probability: f64,
}

/// Performance attribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAttribution {
    /// Asset contributions to return
    pub asset_contributions: HashMap<String, f64>,
    /// Factor contributions to return
    pub factor_contributions: HashMap<String, f64>,
    /// Alpha (excess return)
    pub alpha: f64,
    /// Beta (market exposure)
    pub beta: f64,
    /// Luck vs skill ratio
    pub luck_skill_ratio: f64,
    /// Attribution confidence
    pub attribution_confidence: f64,
}

/// Robustness assessment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessAssessment {
    /// Performance under stress scenarios
    pub stress_performance: HashMap<String, f64>,
    /// Worst-case scenario performance
    pub worst_case_performance: f64,
    /// Best-case scenario performance
    pub best_case_performance: f64,
    /// Robustness score (0-1)
    pub robustness_score: f64,
    /// Fragility indicators
    pub fragility_indicators: Vec<String>,
    /// Recommended adjustments
    pub recommended_adjustments: Vec<String>,
}

/// Portfolio composition for strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioComposition {
    /// Asset weights
    pub weights: HashMap<String, f64>,
    /// Asset types
    pub asset_types: HashMap<String, AssetType>,
    /// Total portfolio value
    pub total_value: f64,
    /// Number of positions
    pub num_positions: usize,
    /// Concentration metrics
    pub concentration_metrics: ConcentrationMetrics,
}

/// Concentration metrics for portfolio analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcentrationMetrics {
    /// Herfindahl-Hirschman Index
    pub hhi: f64,
    /// Effective number of positions
    pub effective_positions: f64,
    /// Largest position weight
    pub largest_position: f64,
    /// Top 5 positions weight
    pub top_5_weight: f64,
    /// Concentration ratio
    pub concentration_ratio: f64,
}

impl ConcentrationMetrics {
    /// Calculate concentration metrics from weights
    pub fn calculate(weights: &HashMap<String, f64>) -> Self {
        let mut weight_values: Vec<f64> = weights.values().cloned().collect();
        weight_values.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        let hhi = weight_values.iter().map(|w| w * w).sum::<f64>();
        let effective_positions = if hhi > 0.0 { 1.0 / hhi } else { 0.0 };
        let largest_position = weight_values.first().cloned().unwrap_or(0.0);
        let top_5_weight = weight_values.iter().take(5).sum::<f64>();
        let concentration_ratio = if weights.len() > 0 {
            top_5_weight / weights.len() as f64
        } else {
            0.0
        };
        
        Self {
            hhi,
            effective_positions,
            largest_position,
            top_5_weight,
            concentration_ratio,
        }
    }
    
    /// Check if portfolio is well-diversified
    pub fn is_diversified(&self) -> bool {
        self.hhi < 0.1 && self.largest_position < 0.2 && self.effective_positions > 10.0
    }
}

/// Strategy performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyPerformance {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Information ratio
    pub information_ratio: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Maximum consecutive losses
    pub max_consecutive_losses: usize,
    /// Average holding period
    pub average_holding_period: f64,
    /// Turnover rate
    pub turnover_rate: f64,
}

/// Rebalancing trigger conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalancingTrigger {
    /// Time-based trigger (days since last rebalance)
    pub time_trigger: Option<usize>,
    /// Drift-based trigger (maximum weight deviation)
    pub drift_trigger: Option<f64>,
    /// Volatility-based trigger
    pub volatility_trigger: Option<f64>,
    /// Performance-based trigger
    pub performance_trigger: Option<f64>,
    /// Market regime change trigger
    pub regime_trigger: Option<MarketRegime>,
}

impl RebalancingTrigger {
    /// Check if rebalancing is needed
    pub fn should_rebalance(
        &self,
        days_since_rebalance: usize,
        current_weights: &HashMap<String, f64>,
        target_weights: &HashMap<String, f64>,
        current_volatility: f64,
        recent_performance: f64,
        current_regime: MarketRegime,
    ) -> bool {
        // Time-based trigger
        if let Some(time_limit) = self.time_trigger {
            if days_since_rebalance >= time_limit {
                return true;
            }
        }
        
        // Drift-based trigger
        if let Some(drift_limit) = self.drift_trigger {
            let max_drift = current_weights.iter()
                .map(|(asset, &current_weight)| {
                    let target_weight = target_weights.get(asset).cloned().unwrap_or(0.0);
                    (current_weight - target_weight).abs()
                })
                .fold(0.0, f64::max);
            
            if max_drift > drift_limit {
                return true;
            }
        }
        
        // Volatility-based trigger
        if let Some(vol_limit) = self.volatility_trigger {
            if current_volatility > vol_limit {
                return true;
            }
        }
        
        // Performance-based trigger
        if let Some(perf_limit) = self.performance_trigger {
            if recent_performance < perf_limit {
                return true;
            }
        }
        
        // Regime-based trigger
        if let Some(target_regime) = self.regime_trigger {
            if current_regime != target_regime {
                return true;
            }
        }
        
        false
    }
}

/// Utility functions for strategy implementation
pub mod utils {
    use super::*;
    use crate::error::TalebianError;
    
    /// Normalize weights to sum to 1.0
    pub fn normalize_weights(weights: &mut HashMap<String, f64>) -> Result<()> {
        let total_weight: f64 = weights.values().sum();
        
        if total_weight <= 0.0 {
            return Err(TalebianError::invalid_parameter(
                "weights",
                "Total weight must be positive"
            ));
        }
        
        for weight in weights.values_mut() {
            *weight /= total_weight;
        }
        
        Ok(())
    }
    
    /// Apply position size constraints
    pub fn apply_constraints(
        weights: &mut HashMap<String, f64>,
        config: &StrategyConfig,
    ) -> Result<()> {
        // Apply maximum position size constraint
        for weight in weights.values_mut() {
            *weight = weight.min(config.max_position_size);
        }
        
        // Remove positions below minimum size
        weights.retain(|_, &mut weight| weight >= config.min_position_size);
        
        // Renormalize after constraints
        normalize_weights(weights)?;
        
        Ok(())
    }
    
    /// Calculate transaction costs for rebalancing
    pub fn calculate_transaction_costs(
        current_weights: &HashMap<String, f64>,
        target_weights: &HashMap<String, f64>,
        transaction_cost_rate: f64,
    ) -> f64 {
        let mut total_turnover = 0.0;
        
        // Calculate turnover for existing positions
        for (asset, &current_weight) in current_weights {
            let target_weight = target_weights.get(asset).cloned().unwrap_or(0.0);
            total_turnover += (current_weight - target_weight).abs();
        }
        
        // Calculate turnover for new positions
        for (asset, &target_weight) in target_weights {
            if !current_weights.contains_key(asset) {
                total_turnover += target_weight;
            }
        }
        
        total_turnover * transaction_cost_rate
    }
    
    /// Calculate portfolio volatility
    pub fn calculate_portfolio_volatility(
        weights: &HashMap<String, f64>,
        volatilities: &HashMap<String, f64>,
        correlations: &HashMap<(String, String), f64>,
    ) -> Result<f64> {
        let mut portfolio_variance = 0.0;
        
        // Individual asset contributions
        for (asset, &weight) in weights {
            if let Some(&volatility) = volatilities.get(asset) {
                portfolio_variance += weight * weight * volatility * volatility;
            }
        }
        
        // Correlation contributions
        for (asset1, &weight1) in weights {
            for (asset2, &weight2) in weights {
                if asset1 != asset2 {
                    let vol1 = volatilities.get(asset1).cloned().unwrap_or(0.0);
                    let vol2 = volatilities.get(asset2).cloned().unwrap_or(0.0);
                    let corr = correlations.get(&(asset1.clone(), asset2.clone()))
                        .or_else(|| correlations.get(&(asset2.clone(), asset1.clone())))
                        .cloned()
                        .unwrap_or(0.0);
                    
                    portfolio_variance += weight1 * weight2 * vol1 * vol2 * corr;
                }
            }
        }
        
        Ok(portfolio_variance.sqrt())
    }
    
    /// Calculate risk-adjusted position sizes
    pub fn risk_adjusted_position_sizes(
        expected_returns: &HashMap<String, f64>,
        volatilities: &HashMap<String, f64>,
        correlations: &HashMap<(String, String), f64>,
        risk_aversion: f64,
    ) -> Result<HashMap<String, f64>> {
        // This is a simplified implementation of mean-variance optimization
        // In practice, you'd use a proper optimization library
        
        let mut weights = HashMap::new();
        let mut total_utility = 0.0;
        
        for (asset, &expected_return) in expected_returns {
            if let Some(&volatility) = volatilities.get(asset) {
                let utility = expected_return - 0.5 * risk_aversion * volatility * volatility;
                weights.insert(asset.clone(), utility.max(0.0));
                total_utility += utility.max(0.0);
            }
        }
        
        // Normalize weights
        if total_utility > 0.0 {
            for weight in weights.values_mut() {
                *weight /= total_utility;
            }
        }
        
        Ok(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::utils::*;
    
    #[test]
    fn test_normalize_weights() {
        let mut weights = HashMap::new();
        weights.insert("AAPL".to_string(), 0.3);
        weights.insert("GOOGL".to_string(), 0.5);
        weights.insert("MSFT".to_string(), 0.4);
        
        normalize_weights(&mut weights).unwrap();
        
        let total: f64 = weights.values().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_apply_constraints() {
        let mut weights = HashMap::new();
        weights.insert("AAPL".to_string(), 0.8);
        weights.insert("GOOGL".to_string(), 0.2);
        weights.insert("MSFT".to_string(), 0.005); // Below minimum
        
        let config = StrategyConfig {
            max_position_size: 0.3,
            min_position_size: 0.01,
            ..Default::default()
        };
        
        apply_constraints(&mut weights, &config).unwrap();
        
        // Should remove MSFT and cap AAPL
        assert!(!weights.contains_key("MSFT"));
        assert!(weights.get("AAPL").unwrap() <= &0.3);
        
        let total: f64 = weights.values().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_transaction_costs() {
        let mut current_weights = HashMap::new();
        current_weights.insert("AAPL".to_string(), 0.5);
        current_weights.insert("GOOGL".to_string(), 0.5);
        
        let mut target_weights = HashMap::new();
        target_weights.insert("AAPL".to_string(), 0.3);
        target_weights.insert("GOOGL".to_string(), 0.3);
        target_weights.insert("MSFT".to_string(), 0.4);
        
        let cost = calculate_transaction_costs(&current_weights, &target_weights, 0.001);
        assert!(cost > 0.0);
    }
    
    #[test]
    fn test_portfolio_volatility() {
        let mut weights = HashMap::new();
        weights.insert("AAPL".to_string(), 0.5);
        weights.insert("GOOGL".to_string(), 0.5);
        
        let mut volatilities = HashMap::new();
        volatilities.insert("AAPL".to_string(), 0.2);
        volatilities.insert("GOOGL".to_string(), 0.3);
        
        let mut correlations = HashMap::new();
        correlations.insert(("AAPL".to_string(), "GOOGL".to_string()), 0.6);
        
        let portfolio_vol = calculate_portfolio_volatility(&weights, &volatilities, &correlations).unwrap();
        assert!(portfolio_vol > 0.0);
        assert!(portfolio_vol < 0.3); // Should be less than max individual volatility due to diversification
    }
    
    #[test]
    fn test_concentration_metrics() {
        let mut weights = HashMap::new();
        weights.insert("AAPL".to_string(), 0.4);
        weights.insert("GOOGL".to_string(), 0.3);
        weights.insert("MSFT".to_string(), 0.2);
        weights.insert("AMZN".to_string(), 0.1);
        
        let metrics = ConcentrationMetrics::calculate(&weights);
        assert!(metrics.hhi > 0.0);
        assert!(metrics.effective_positions > 0.0);
        assert_eq!(metrics.largest_position, 0.4);
        assert!(metrics.top_5_weight >= 1.0);
    }
    
    #[test]
    fn test_rebalancing_trigger() {
        let trigger = RebalancingTrigger {
            time_trigger: Some(30),
            drift_trigger: Some(0.1),
            volatility_trigger: Some(0.3),
            performance_trigger: Some(-0.1),
            regime_trigger: Some(MarketRegime::Crisis),
        };
        
        let current_weights = HashMap::new();
        let target_weights = HashMap::new();
        
        // Time trigger
        assert!(trigger.should_rebalance(
            35, &current_weights, &target_weights, 0.1, 0.05, MarketRegime::Normal
        ));
        
        // Volatility trigger
        assert!(trigger.should_rebalance(
            10, &current_weights, &target_weights, 0.4, 0.05, MarketRegime::Normal
        ));
        
        // Performance trigger
        assert!(trigger.should_rebalance(
            10, &current_weights, &target_weights, 0.1, -0.15, MarketRegime::Normal
        ));
        
        // Regime trigger
        assert!(trigger.should_rebalance(
            10, &current_weights, &target_weights, 0.1, 0.05, MarketRegime::Crisis
        ));
    }
}