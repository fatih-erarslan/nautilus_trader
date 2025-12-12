//! Barbell strategy implementation
//!
//! The barbell strategy is a central concept in Talebian risk management,
//! combining extreme safety with high-risk, high-reward positions.

use crate::barbell::AssetType;
use crate::error::{TalebianResult as Result, TalebianError};
use crate::strategies::*;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Barbell strategy implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarbellStrategy {
    /// Strategy identifier
    id: String,
    /// Strategy configuration
    config: StrategyConfig,
    /// Barbell-specific parameters
    barbell_params: BarbellParams,
    /// Safe asset allocation
    safe_allocation: f64,
    /// Risky asset allocation
    risky_allocation: f64,
    /// Current portfolio composition
    portfolio: PortfolioComposition,
    /// Performance history
    performance_history: Vec<PerformanceRecord>,
    /// Last rebalancing date
    last_rebalance: Option<DateTime<Utc>>,
}

/// Barbell strategy parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarbellParams {
    /// Target allocation to safe assets (0.0 to 1.0)
    pub safe_target: f64,
    /// Target allocation to risky assets (0.0 to 1.0)
    pub risky_target: f64,
    /// Maximum safe asset allocation
    pub max_safe_allocation: f64,
    /// Maximum risky asset allocation
    pub max_risky_allocation: f64,
    /// Minimum safe asset allocation
    pub min_safe_allocation: f64,
    /// Minimum risky asset allocation
    pub min_risky_allocation: f64,
    /// Volatility threshold for safe assets
    pub safe_volatility_threshold: f64,
    /// Minimum expected return for risky assets
    pub risky_return_threshold: f64,
    /// Rebalancing tolerance (drift before rebalancing)
    pub rebalancing_tolerance: f64,
    /// Dynamic adjustment factor
    pub adjustment_factor: f64,
    /// Convexity bias (preference for asymmetric payoffs)
    pub convexity_bias: f64,
}

impl Default for BarbellParams {
    fn default() -> Self {
        Self {
            safe_target: 0.8,              // 80% safe assets
            risky_target: 0.2,             // 20% risky assets
            max_safe_allocation: 0.95,     // Maximum 95% safe
            max_risky_allocation: 0.3,     // Maximum 30% risky
            min_safe_allocation: 0.6,      // Minimum 60% safe
            min_risky_allocation: 0.05,    // Minimum 5% risky
            safe_volatility_threshold: 0.05, // 5% max volatility for safe assets
            risky_return_threshold: 0.15,  // 15% min expected return for risky assets
            rebalancing_tolerance: 0.05,   // 5% drift tolerance
            adjustment_factor: 0.1,        // 10% adjustment factor
            convexity_bias: 1.5,           // 50% bias towards convex payoffs
        }
    }
}

/// Performance record for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecord {
    /// Record timestamp
    pub timestamp: DateTime<Utc>,
    /// Total return
    pub total_return: f64,
    /// Safe assets return
    pub safe_return: f64,
    /// Risky assets return
    pub risky_return: f64,
    /// Strategy volatility
    pub volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Antifragility score
    pub antifragility_score: f64,
}

impl BarbellStrategy {
    /// Create a new barbell strategy
    pub fn new(
        id: impl Into<String>,
        config: StrategyConfig,
        barbell_params: BarbellParams,
    ) -> Result<Self> {
        // Validate parameters
        if barbell_params.safe_target + barbell_params.risky_target > 1.0 {
            return Err(TalebianError::invalid_parameter(
                "allocation_targets",
                "Safe and risky targets cannot exceed 100%"
            ));
        }
        
        if barbell_params.safe_target < barbell_params.min_safe_allocation {
            return Err(TalebianError::invalid_parameter(
                "safe_target",
                "Safe target below minimum allocation"
            ));
        }
        
        if barbell_params.risky_target < barbell_params.min_risky_allocation {
            return Err(TalebianError::invalid_parameter(
                "risky_target",
                "Risky target below minimum allocation"
            ));
        }
        
        let safe_allocation = barbell_params.safe_target;
        let risky_allocation = barbell_params.risky_target;
        
        let portfolio = PortfolioComposition {
            weights: HashMap::new(),
            asset_types: HashMap::new(),
            total_value: 0.0,
            num_positions: 0,
            concentration_metrics: ConcentrationMetrics::calculate(&HashMap::new()),
        };
        
        Ok(Self {
            id: id.into(),
            config,
            barbell_params,
            safe_allocation,
            risky_allocation,
            portfolio,
            performance_history: Vec::new(),
            last_rebalance: None,
        })
    }
    
    /// Classify assets into safe and risky categories
    fn classify_assets(&self, market_data: &MarketData) -> Result<(Vec<String>, Vec<String>)> {
        let mut safe_assets = Vec::new();
        let mut risky_assets = Vec::new();
        
        for (asset, &asset_type) in &market_data.asset_types {
            match asset_type {
                AssetType::Safe => safe_assets.push(asset.clone()),
                AssetType::Volatile | AssetType::Derivative => {
                    // Additional checks for risky assets
                    if let Some(returns) = market_data.returns.get(asset) {
                        if !returns.is_empty() {
                            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
                            let annualized_return = mean_return * 252.0; // Assuming daily returns
                            
                            if annualized_return > self.barbell_params.risky_return_threshold {
                                risky_assets.push(asset.clone());
                            }
                        }
                    }
                }
                AssetType::Moderate => {
                    // Check volatility to determine classification
                    if let Some(&volatility) = market_data.volatilities.get(asset) {
                        if volatility <= self.barbell_params.safe_volatility_threshold {
                            safe_assets.push(asset.clone());
                        } else {
                            risky_assets.push(asset.clone());
                        }
                    }
                }
                AssetType::Antifragile => {
                    // Antifragile assets go to risky side due to their convexity
                    risky_assets.push(asset.clone());
                }
                AssetType::Alternative => {
                    // Alternative assets treated as risky
                    risky_assets.push(asset.clone());
                }
                AssetType::Risky => {
                    // Risky assets go to risky side
                    risky_assets.push(asset.clone());
                }
                AssetType::Hedge => {
                    // Hedge assets go to safe side
                    safe_assets.push(asset.clone());
                }
            }
        }
        
        if safe_assets.is_empty() {
            return Err(TalebianError::portfolio_construction(
                "No safe assets available for barbell strategy"
            ));
        }
        
        if risky_assets.is_empty() {
            return Err(TalebianError::portfolio_construction(
                "No risky assets available for barbell strategy"
            ));
        }
        
        Ok((safe_assets, risky_assets))
    }
    
    /// Calculate safe asset weights
    fn calculate_safe_weights(
        &self,
        safe_assets: &[String],
        market_data: &MarketData,
    ) -> Result<HashMap<String, f64>> {
        let mut weights = HashMap::new();
        let mut total_score = 0.0;
        
        // Score safe assets based on safety and liquidity
        for asset in safe_assets {
            let volatility = market_data.volatilities.get(asset).cloned().unwrap_or(0.1);
            let volume = market_data.volumes.get(asset).cloned().unwrap_or(1.0);
            
            // Lower volatility and higher volume = better safe asset
            let safety_score = 1.0 / (1.0 + volatility);
            let liquidity_score = (volume / 1000000.0).min(1.0); // Normalize volume
            let combined_score = safety_score * liquidity_score;
            
            weights.insert(asset.clone(), combined_score);
            total_score += combined_score;
        }
        
        // Normalize weights
        if total_score > 0.0 {
            for weight in weights.values_mut() {
                *weight /= total_score;
            }
        }
        
        Ok(weights)
    }
    
    /// Calculate risky asset weights with convexity bias
    fn calculate_risky_weights(
        &self,
        risky_assets: &[String],
        market_data: &MarketData,
    ) -> Result<HashMap<String, f64>> {
        let mut weights = HashMap::new();
        let mut total_score = 0.0;
        
        // Score risky assets based on expected return and convexity
        for asset in risky_assets {
            let returns = market_data.returns.get(asset).cloned().unwrap_or_default();
            if returns.is_empty() {
                continue;
            }
            
            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let volatility = market_data.volatilities.get(asset).cloned().unwrap_or(0.2);
            
            // Calculate convexity measure (preference for asymmetric payoffs)
            let convexity = self.calculate_convexity(&returns)?;
            
            // Risk-adjusted return with convexity bias
            let return_score = mean_return / volatility;
            let convexity_score = convexity * self.barbell_params.convexity_bias;
            let combined_score = (return_score + convexity_score).max(0.0);
            
            weights.insert(asset.clone(), combined_score);
            total_score += combined_score;
        }
        
        // Normalize weights
        if total_score > 0.0 {
            for weight in weights.values_mut() {
                *weight /= total_score;
            }
        }
        
        Ok(weights)
    }
    
    /// Calculate convexity measure for an asset
    fn calculate_convexity(&self, returns: &[f64]) -> Result<f64> {
        if returns.len() < 10 {
            return Ok(0.0);
        }
        
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted_returns.len();
        let bottom_10 = &sorted_returns[0..n/10];
        let top_10 = &sorted_returns[9*n/10..];
        
        let bottom_avg = bottom_10.iter().sum::<f64>() / bottom_10.len() as f64;
        let top_avg = top_10.iter().sum::<f64>() / top_10.len() as f64;
        
        // Convexity: asymmetric payoffs (more upside than downside)
        let convexity = if bottom_avg < 0.0 {
            top_avg / bottom_avg.abs()
        } else {
            1.0
        };
        
        Ok(convexity.min(10.0)) // Cap at 10x
    }
    
    /// Adjust allocations based on market conditions
    fn adjust_allocations(&mut self, market_data: &MarketData) -> Result<()> {
        // Calculate market stress level
        let stress_level = self.calculate_market_stress(market_data)?;
        
        // Adjust safe/risky allocation based on stress
        let stress_adjustment = stress_level * self.barbell_params.adjustment_factor;
        
        // Increase safe allocation during stress
        self.safe_allocation = (self.barbell_params.safe_target + stress_adjustment)
            .min(self.barbell_params.max_safe_allocation)
            .max(self.barbell_params.min_safe_allocation);
        
        // Decrease risky allocation during stress
        self.risky_allocation = (self.barbell_params.risky_target - stress_adjustment)
            .min(self.barbell_params.max_risky_allocation)
            .max(self.barbell_params.min_risky_allocation);
        
        // Ensure allocations sum to 1.0
        let total_allocation = self.safe_allocation + self.risky_allocation;
        if total_allocation > 1.0 {
            self.safe_allocation /= total_allocation;
            self.risky_allocation /= total_allocation;
        }
        
        Ok(())
    }
    
    /// Calculate market stress level
    fn calculate_market_stress(&self, market_data: &MarketData) -> Result<f64> {
        let mut stress_indicators = Vec::new();
        
        // Volatility stress
        let avg_volatility = market_data.volatilities.values().sum::<f64>() 
            / market_data.volatilities.len() as f64;
        let volatility_stress = (avg_volatility / 0.2).min(1.0); // Normalize to 20% volatility
        stress_indicators.push(volatility_stress);
        
        // Correlation stress (higher correlation = more stress)
        let avg_correlation = market_data.correlations.values().sum::<f64>() 
            / market_data.correlations.len() as f64;
        let correlation_stress = avg_correlation.abs();
        stress_indicators.push(correlation_stress);
        
        // Return stress (negative returns = stress)
        let mut negative_returns = 0;
        let mut total_returns = 0;
        for returns in market_data.returns.values() {
            if let Some(&recent_return) = returns.last() {
                if recent_return < 0.0 {
                    negative_returns += 1;
                }
                total_returns += 1;
            }
        }
        
        let return_stress = if total_returns > 0 {
            negative_returns as f64 / total_returns as f64
        } else {
            0.0
        };
        stress_indicators.push(return_stress);
        
        // Regime stress
        let regime_stress = match market_data.regime {
            MarketRegime::Crisis => 1.0,
            MarketRegime::Bear => 0.8,
            MarketRegime::HighVolatility => 0.6,
            MarketRegime::Normal => 0.2,
            MarketRegime::Bull => 0.0,
            MarketRegime::LowVolatility => 0.0,
            MarketRegime::Recovery => 0.3,
        };
        stress_indicators.push(regime_stress);
        
        // Combine stress indicators
        let overall_stress = stress_indicators.iter().sum::<f64>() / stress_indicators.len() as f64;
        Ok(overall_stress.min(1.0))
    }
    
    /// Check if rebalancing is needed
    fn needs_rebalancing(&self, current_weights: &HashMap<String, f64>) -> bool {
        // Calculate current safe/risky allocations
        let mut current_safe = 0.0;
        let mut current_risky = 0.0;
        
        for (asset, &weight) in current_weights {
            if let Some(&asset_type) = self.portfolio.asset_types.get(asset) {
                match asset_type {
                    AssetType::Safe | AssetType::Moderate => current_safe += weight,
                    _ => current_risky += weight,
                }
            }
        }
        
        // Check drift tolerance
        let safe_drift = (current_safe - self.safe_allocation).abs();
        let risky_drift = (current_risky - self.risky_allocation).abs();
        
        safe_drift > self.barbell_params.rebalancing_tolerance
            || risky_drift > self.barbell_params.rebalancing_tolerance
    }
    
    /// Get barbell-specific metrics
    pub fn get_barbell_metrics(&self) -> BarbellMetrics {
        let safe_weight = self.portfolio.weights.iter()
            .filter(|(asset, _)| {
                self.portfolio.asset_types.get(*asset)
                    .map_or(false, |&t| matches!(t, AssetType::Safe | AssetType::Moderate))
            })
            .map(|(_, weight)| weight)
            .sum::<f64>();
        
        let risky_weight = 1.0 - safe_weight;
        
        BarbellMetrics {
            safe_allocation: safe_weight,
            risky_allocation: risky_weight,
            target_safe_allocation: self.safe_allocation,
            target_risky_allocation: self.risky_allocation,
            allocation_drift: (safe_weight - self.safe_allocation).abs(),
            barbell_ratio: if risky_weight > 0.0 { safe_weight / risky_weight } else { f64::INFINITY },
            convexity_exposure: self.calculate_convexity_exposure(),
            safety_score: self.calculate_safety_score(),
        }
    }
    
    /// Calculate convexity exposure
    fn calculate_convexity_exposure(&self) -> f64 {
        let mut convexity_exposure = 0.0;
        
        for (asset, &weight) in &self.portfolio.weights {
            if let Some(&asset_type) = self.portfolio.asset_types.get(asset) {
                let convexity_multiplier = match asset_type {
                    AssetType::Derivative => 3.0,
                    AssetType::Antifragile => 2.0,
                    AssetType::Volatile => 1.5,
                    AssetType::Alternative => 1.2,
                    AssetType::Risky => 1.0,
                    AssetType::Safe | AssetType::Moderate | AssetType::Hedge => 0.0,
                };
                convexity_exposure += weight * convexity_multiplier;
            }
        }
        
        convexity_exposure
    }
    
    /// Calculate safety score
    fn calculate_safety_score(&self) -> f64 {
        let mut safety_score = 0.0;
        
        for (asset, &weight) in &self.portfolio.weights {
            if let Some(&asset_type) = self.portfolio.asset_types.get(asset) {
                let safety_multiplier = match asset_type {
                    AssetType::Safe => 1.0,
                    AssetType::Moderate => 0.7,
                    AssetType::Antifragile => 0.5,
                    AssetType::Alternative => 0.3,
                    AssetType::Volatile => 0.1,
                    AssetType::Derivative => 0.0,
                    AssetType::Risky => 0.2,
                    AssetType::Hedge => 0.4,
                };
                safety_score += weight * safety_multiplier;
            }
        }
        
        safety_score
    }
    
    /// Record performance
    fn record_performance(&mut self, market_data: &MarketData) -> Result<()> {
        // Calculate returns for different components
        let mut safe_return = 0.0;
        let mut risky_return = 0.0;
        let mut total_return = 0.0;
        
        for (asset, &weight) in &self.portfolio.weights {
            if let Some(returns) = market_data.returns.get(asset) {
                if let Some(&recent_return) = returns.last() {
                    total_return += weight * recent_return;
                    
                    if let Some(&asset_type) = self.portfolio.asset_types.get(asset) {
                        match asset_type {
                            AssetType::Safe | AssetType::Moderate => {
                                safe_return += weight * recent_return;
                            }
                            _ => {
                                risky_return += weight * recent_return;
                            }
                        }
                    }
                }
            }
        }
        
        // Calculate other metrics
        let volatility = self.calculate_portfolio_volatility(market_data)?;
        let sharpe_ratio = if volatility > 0.0 { total_return / volatility } else { 0.0 };
        let max_drawdown = self.calculate_max_drawdown()?;
        let antifragility_score = self.calculate_antifragility_score(market_data)?;
        
        let performance = PerformanceRecord {
            timestamp: market_data.timestamp,
            total_return,
            safe_return,
            risky_return,
            volatility,
            sharpe_ratio,
            max_drawdown,
            antifragility_score,
        };
        
        self.performance_history.push(performance);
        
        // Keep only recent history
        if self.performance_history.len() > 252 {
            self.performance_history.remove(0);
        }
        
        Ok(())
    }
    
    /// Calculate portfolio volatility
    fn calculate_portfolio_volatility(&self, market_data: &MarketData) -> Result<f64> {
        use crate::strategies::utils::calculate_portfolio_volatility;
        calculate_portfolio_volatility(
            &self.portfolio.weights,
            &market_data.volatilities,
            &market_data.correlations,
        )
    }
    
    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self) -> Result<f64> {
        if self.performance_history.is_empty() {
            return Ok(0.0);
        }
        
        let mut max_drawdown = 0.0f64;
        let mut peak_return = 0.0f64;
        let mut cumulative_return = 0.0;
        
        for record in &self.performance_history {
            cumulative_return += record.total_return;
            peak_return = peak_return.max(cumulative_return);
            let drawdown = peak_return - cumulative_return;
            max_drawdown = max_drawdown.max(drawdown);
        }
        
        Ok(max_drawdown)
    }
    
    /// Calculate antifragility score
    fn calculate_antifragility_score(&self, market_data: &MarketData) -> Result<f64> {
        // Simplified antifragility calculation
        let stress_level = self.calculate_market_stress(market_data)?;
        let convexity_exposure = self.calculate_convexity_exposure();
        
        // Antifragility: benefit from stress through convexity
        let antifragility = convexity_exposure * stress_level;
        Ok(antifragility.min(1.0))
    }
    
    /// Get performance history
    pub fn get_performance_history(&self) -> &[PerformanceRecord] {
        &self.performance_history
    }
    
    /// Get current barbell parameters
    pub fn get_barbell_params(&self) -> &BarbellParams {
        &self.barbell_params
    }
    
    /// Update barbell parameters
    pub fn update_barbell_params(&mut self, params: BarbellParams) -> Result<()> {
        // Validate new parameters
        if params.safe_target + params.risky_target > 1.0 {
            return Err(TalebianError::invalid_parameter(
                "allocation_targets",
                "Safe and risky targets cannot exceed 100%"
            ));
        }
        
        self.safe_allocation = params.safe_target;
        self.risky_allocation = params.risky_target;
        self.barbell_params = params;
        
        Ok(())
    }
}

impl TalebianStrategy for BarbellStrategy {
    fn strategy_type(&self) -> StrategyType {
        StrategyType::Barbell
    }
    
    fn calculate_position_sizes(&self, _assets: &[String], market_data: &MarketData) -> Result<HashMap<String, f64>> {
        // Classify assets
        let (safe_assets, risky_assets) = self.classify_assets(market_data)?;
        
        // Calculate weights for each category
        let safe_weights = self.calculate_safe_weights(&safe_assets, market_data)?;
        let risky_weights = self.calculate_risky_weights(&risky_assets, market_data)?;
        
        // Combine weights with barbell allocations
        let mut final_weights = HashMap::new();
        
        // Apply safe allocation
        for (asset, weight) in safe_weights {
            final_weights.insert(asset, weight * self.safe_allocation);
        }
        
        // Apply risky allocation
        for (asset, weight) in risky_weights {
            final_weights.insert(asset, weight * self.risky_allocation);
        }
        
        Ok(final_weights)
    }
    
    fn update_strategy(&mut self, market_data: &MarketData) -> Result<()> {
        // Adjust allocations based on market conditions
        self.adjust_allocations(market_data)?;
        
        // Update portfolio composition
        let new_weights = self.calculate_position_sizes(&[], market_data)?;
        
        // Check if rebalancing is needed
        if self.needs_rebalancing(&new_weights) {
            self.portfolio.weights = new_weights;
            self.portfolio.asset_types = market_data.asset_types.clone();
            self.portfolio.concentration_metrics = ConcentrationMetrics::calculate(&self.portfolio.weights);
            self.portfolio.num_positions = self.portfolio.weights.len();
            self.last_rebalance = Some(market_data.timestamp);
        }
        
        // Record performance
        self.record_performance(market_data)?;
        
        Ok(())
    }
    
    fn expected_return(&self, market_data: &MarketData) -> Result<f64> {
        let mut expected_return = 0.0;
        
        for (asset, &weight) in &self.portfolio.weights {
            if let Some(returns) = market_data.returns.get(asset) {
                if !returns.is_empty() {
                    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
                    expected_return += weight * mean_return;
                }
            }
        }
        
        Ok(expected_return)
    }
    
    fn risk_metrics(&self, market_data: &MarketData) -> Result<StrategyRiskMetrics> {
        let volatility = self.calculate_portfolio_volatility(market_data)?;
        let expected_return = self.expected_return(market_data)?;
        let max_drawdown = self.calculate_max_drawdown()?;
        let antifragility_score = self.calculate_antifragility_score(market_data)?;
        
        // Calculate VaR and CVaR (simplified)
        let var_95 = -1.65 * volatility; // Normal distribution approximation
        let cvar_95 = var_95 * 1.2; // Rough approximation
        
        // Calculate other risk metrics
        let sharpe_ratio = if volatility > 0.0 { expected_return / volatility } else { 0.0 };
        let sortino_ratio = sharpe_ratio * 1.4; // Rough approximation
        let calmar_ratio = if max_drawdown > 0.0 { expected_return / max_drawdown } else { 0.0 };
        
        Ok(StrategyRiskMetrics {
            var_95,
            cvar_95,
            max_drawdown,
            volatility,
            downside_deviation: volatility * 0.7, // Approximation
            tail_ratio: 1.0, // Placeholder
            sortino_ratio,
            calmar_ratio,
            antifragility_score,
            black_swan_probability: 0.01, // Placeholder
        })
    }
    
    fn performance_attribution(&self, returns: &[f64]) -> Result<PerformanceAttribution> {
        let mut asset_contributions = HashMap::new();
        let mut factor_contributions = HashMap::new();
        
        // Calculate asset contributions
        for (asset, &weight) in &self.portfolio.weights {
            asset_contributions.insert(asset.clone(), weight * 0.1); // Placeholder
        }
        
        // Calculate factor contributions
        factor_contributions.insert("Safe Factor".to_string(), self.safe_allocation * 0.05);
        factor_contributions.insert("Risky Factor".to_string(), self.risky_allocation * 0.15);
        
        // Calculate alpha and beta
        let alpha = returns.iter().sum::<f64>() / returns.len() as f64;
        let beta = 1.0; // Placeholder
        
        Ok(PerformanceAttribution {
            asset_contributions,
            factor_contributions,
            alpha,
            beta,
            luck_skill_ratio: 0.3, // Placeholder
            attribution_confidence: 0.8, // Placeholder
        })
    }
    
    fn robustness_assessment(&self, scenarios: &[MarketScenario]) -> Result<RobustnessAssessment> {
        let mut stress_performance = HashMap::new();
        let mut performances = Vec::new();
        
        for scenario in scenarios {
            // Simulate performance under scenario
            let performance = self.simulate_scenario_performance(scenario)?;
            stress_performance.insert(scenario.name.clone(), performance);
            performances.push(performance);
        }
        
        let worst_case = performances.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let best_case = performances.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        // Calculate robustness score
        let robustness_score = if worst_case < -0.5 { 0.0 } else { 1.0 - worst_case.abs() };
        
        let fragility_indicators = if worst_case < -0.2 {
            vec!["High downside risk".to_string()]
        } else {
            vec![]
        };
        
        let recommended_adjustments = if robustness_score < 0.7 {
            vec!["Increase safe allocation".to_string()]
        } else {
            vec![]
        };
        
        Ok(RobustnessAssessment {
            stress_performance,
            worst_case_performance: worst_case,
            best_case_performance: best_case,
            robustness_score,
            fragility_indicators,
            recommended_adjustments,
        })
    }
    
    fn get_config(&self) -> &StrategyConfig {
        &self.config
    }
    
    fn update_config(&mut self, config: StrategyConfig) -> Result<()> {
        self.config = config;
        Ok(())
    }
    
    fn is_suitable(&self, market_data: &MarketData) -> Result<bool> {
        // Barbell strategy is suitable when there are clear safe and risky assets
        let (safe_assets, risky_assets) = self.classify_assets(market_data)?;
        Ok(!safe_assets.is_empty() && !risky_assets.is_empty())
    }
    
    fn calculate_capacity(&self, market_data: &MarketData) -> Result<f64> {
        // Calculate capacity based on asset liquidity
        let mut capacity = 0.0;
        
        for (asset, &weight) in &self.portfolio.weights {
            if let Some(&volume) = market_data.volumes.get(asset) {
                // Assume we can trade up to 10% of daily volume
                let asset_capacity = volume * 0.1;
                capacity += asset_capacity / weight;
            }
        }
        
        Ok(capacity)
    }
}

impl BarbellStrategy {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn name(&self) -> &str {
        "Barbell Strategy"
    }
    
    fn validate(&self) -> Result<()> {
        if self.safe_allocation + self.risky_allocation > 1.0 {
            return Err(TalebianError::invalid_parameter(
                "allocations",
                "Safe and risky allocations cannot exceed 100%"
            ));
        }
        
        if self.safe_allocation < 0.0 || self.risky_allocation < 0.0 {
            return Err(TalebianError::invalid_parameter(
                "allocations",
                "Allocations cannot be negative"
            ));
        }
        
        Ok(())
    }
    
    fn reset(&mut self) {
        self.safe_allocation = self.barbell_params.safe_target;
        self.risky_allocation = self.barbell_params.risky_target;
        self.portfolio.weights.clear();
        self.portfolio.asset_types.clear();
        self.performance_history.clear();
        self.last_rebalance = None;
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "BarbellStrategy".to_string());
        metadata.insert("id".to_string(), self.id.clone());
        metadata.insert("safe_allocation".to_string(), self.safe_allocation.to_string());
        metadata.insert("risky_allocation".to_string(), self.risky_allocation.to_string());
        metadata.insert("num_positions".to_string(), self.portfolio.num_positions.to_string());
        metadata.insert("performance_records".to_string(), self.performance_history.len().to_string());
        metadata
    }
}

impl BarbellStrategy {
    /// Simulate performance under a market scenario
    fn simulate_scenario_performance(&self, scenario: &MarketScenario) -> Result<f64> {
        let mut performance = 0.0;
        
        for (asset, &weight) in &self.portfolio.weights {
            if let Some(&shock) = scenario.price_shocks.get(asset) {
                performance += weight * shock;
            }
        }
        
        Ok(performance)
    }
}

/// Barbell strategy specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarbellMetrics {
    /// Current safe allocation
    pub safe_allocation: f64,
    /// Current risky allocation
    pub risky_allocation: f64,
    /// Target safe allocation
    pub target_safe_allocation: f64,
    /// Target risky allocation
    pub target_risky_allocation: f64,
    /// Allocation drift from target
    pub allocation_drift: f64,
    /// Barbell ratio (safe/risky)
    pub barbell_ratio: f64,
    /// Convexity exposure
    pub convexity_exposure: f64,
    /// Safety score
    pub safety_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    
    #[test]
    fn test_barbell_strategy_creation() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        
        let strategy = BarbellStrategy::new("test_barbell", config, params);
        assert!(strategy.is_ok());
        
        let strategy = strategy.unwrap();
        assert_eq!(strategy.id(), "test_barbell");
        assert_eq!(strategy.name(), "Barbell Strategy");
        assert_eq!(strategy.strategy_type(), StrategyType::Barbell);
    }
    
    #[test]
    fn test_invalid_parameters() {
        let config = StrategyConfig::default();
        let mut params = BarbellParams::default();
        params.safe_target = 0.9;
        params.risky_target = 0.9; // Total > 1.0
        
        let strategy = BarbellStrategy::new("test_barbell", config, params);
        assert!(strategy.is_err());
    }
    
    #[test]
    fn test_asset_classification() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        let mut asset_types = HashMap::new();
        asset_types.insert("BONDS".to_string(), AssetType::Safe);
        asset_types.insert("STOCKS".to_string(), AssetType::Volatile);
        
        let mut returns = HashMap::new();
        returns.insert("STOCKS".to_string(), vec![0.1, 0.05, 0.08]); // High returns
        
        let market_data = MarketData {
            prices: HashMap::new(),
            returns,
            volatilities: HashMap::new(),
            correlations: HashMap::new(),
            volumes: HashMap::new(),
            asset_types,
            timestamp: Utc::now(),
            regime: MarketRegime::Normal,
        };
        
        let (safe_assets, risky_assets) = strategy.classify_assets(&market_data).unwrap();
        assert!(safe_assets.contains(&"BONDS".to_string()));
        assert!(risky_assets.contains(&"STOCKS".to_string()));
    }
    
    #[test]
    fn test_convexity_calculation() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        // Returns with positive convexity (more upside than downside)
        let returns = vec![-0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2];
        let convexity = strategy.calculate_convexity(&returns).unwrap();
        
        assert!(convexity > 1.0); // Should detect positive convexity
    }
    
    #[test]
    fn test_market_stress_calculation() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        let mut volatilities = HashMap::new();
        volatilities.insert("STOCKS".to_string(), 0.4); // High volatility
        
        let mut returns = HashMap::new();
        returns.insert("STOCKS".to_string(), vec![-0.1, -0.05, -0.08]); // Negative returns
        
        let market_data = MarketData {
            prices: HashMap::new(),
            returns,
            volatilities,
            correlations: HashMap::new(),
            volumes: HashMap::new(),
            asset_types: HashMap::new(),
            timestamp: Utc::now(),
            regime: MarketRegime::Crisis,
        };
        
        let stress_level = strategy.calculate_market_stress(&market_data).unwrap();
        assert!(stress_level > 0.5); // Should detect high stress
    }
    
    #[test]
    fn test_barbell_metrics() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let mut strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        // Set up portfolio
        strategy.portfolio.weights.insert("BONDS".to_string(), 0.8);
        strategy.portfolio.weights.insert("STOCKS".to_string(), 0.2);
        strategy.portfolio.asset_types.insert("BONDS".to_string(), AssetType::Safe);
        strategy.portfolio.asset_types.insert("STOCKS".to_string(), AssetType::Volatile);
        
        let metrics = strategy.get_barbell_metrics();
        assert!((metrics.safe_allocation - 0.8).abs() < 1e-10);
        assert!((metrics.risky_allocation - 0.2).abs() < 1e-10);
        assert!(metrics.barbell_ratio > 0.0);
    }
    
    #[test]
    fn test_position_size_calculation() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        let mut asset_types = HashMap::new();
        asset_types.insert("BONDS".to_string(), AssetType::Safe);
        asset_types.insert("STOCKS".to_string(), AssetType::Volatile);
        
        let mut returns = HashMap::new();
        returns.insert("STOCKS".to_string(), vec![0.1, 0.05, 0.08]);
        
        let mut volatilities = HashMap::new();
        volatilities.insert("BONDS".to_string(), 0.02);
        volatilities.insert("STOCKS".to_string(), 0.2);
        
        let mut volumes = HashMap::new();
        volumes.insert("BONDS".to_string(), 1000000.0);
        volumes.insert("STOCKS".to_string(), 2000000.0);
        
        let market_data = MarketData {
            prices: HashMap::new(),
            returns,
            volatilities,
            correlations: HashMap::new(),
            volumes,
            asset_types,
            timestamp: Utc::now(),
            regime: MarketRegime::Normal,
        };
        
        let assets = vec!["BONDS".to_string(), "STOCKS".to_string()];
        let positions = strategy.calculate_position_sizes(&assets, &market_data).unwrap();
        
        assert!(positions.contains_key("BONDS"));
        assert!(positions.contains_key("STOCKS"));
        
        let total_allocation: f64 = positions.values().sum();
        assert!((total_allocation - 1.0).abs() < 1e-10);
    }
}