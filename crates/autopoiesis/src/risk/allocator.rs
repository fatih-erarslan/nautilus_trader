//! Position allocation implementation

use crate::prelude::*;
use crate::models::{Position, MarketData, Order};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use num_traits::ToPrimitive;
use std::collections::HashMap;

/// Position allocator for determining optimal position sizes
#[derive(Debug, Clone)]
pub struct PositionAllocator {
    /// Allocator configuration
    config: PositionAllocatorConfig,
    
    /// Current portfolio state
    portfolio_state: PortfolioState,
    
    /// Risk metrics cache
    risk_metrics_cache: RiskMetricsCache,
    
    /// Historical allocation performance
    allocation_history: Vec<AllocationRecord>,
}

#[derive(Debug, Clone)]
pub struct PositionAllocatorConfig {
    /// Base allocation method
    pub allocation_method: AllocationMethod,
    
    /// Maximum position size as percentage of portfolio
    pub max_position_size_pct: f64,
    
    /// Minimum position size in base currency
    pub min_position_size: Decimal,
    
    /// Risk budget per position
    pub risk_budget_per_position: f64,
    
    /// Correlation adjustment factor
    pub correlation_adjustment: f64,
    
    /// Volatility adjustment parameters
    pub volatility_params: VolatilityParams,
    
    /// Kelly criterion parameters
    pub kelly_params: KellyParams,
}

#[derive(Debug, Clone)]
pub enum AllocationMethod {
    /// Fixed position size
    Fixed(Decimal),
    
    /// Equal weight allocation
    EqualWeight,
    
    /// Risk parity allocation
    RiskParity,
    
    /// Volatility adjusted allocation
    VolatilityAdjusted,
    
    /// Kelly optimal allocation
    KellyOptimal,
    
    /// Modern portfolio theory
    MeanVariance,
    
    /// Risk budgeting approach
    RiskBudgeting,
}

#[derive(Debug, Clone)]
pub struct VolatilityParams {
    /// Lookback period for volatility calculation
    pub lookback_days: u32,
    
    /// Target volatility
    pub target_volatility: f64,
    
    /// Volatility floor
    pub min_volatility: f64,
    
    /// Volatility ceiling
    pub max_volatility: f64,
}

#[derive(Debug, Clone)]
pub struct KellyParams {
    /// Confidence level for Kelly calculation
    pub confidence_level: f64,
    
    /// Kelly fraction multiplier (safety factor)
    pub kelly_multiplier: f64,
    
    /// Minimum win rate threshold
    pub min_win_rate: f64,
    
    /// Maximum Kelly fraction
    pub max_kelly_fraction: f64,
}

#[derive(Debug, Clone, Default)]
struct PortfolioState {
    total_equity: Decimal,
    available_capital: Decimal,
    current_positions: HashMap<String, Position>,
    reserved_capital: Decimal,
    last_updated: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Default)]
struct RiskMetricsCache {
    symbol_volatilities: HashMap<String, f64>,
    correlation_matrix: HashMap<String, HashMap<String, f64>>,
    var_estimates: HashMap<String, f64>,
    last_updated: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
struct AllocationRecord {
    timestamp: DateTime<Utc>,
    symbol: String,
    allocation_method: String,
    recommended_size: Decimal,
    actual_size: Decimal,
    confidence_score: f64,
    risk_metrics: AllocationRiskMetrics,
}

#[derive(Debug, Clone)]
struct AllocationRiskMetrics {
    estimated_volatility: f64,
    position_var: f64,
    correlation_risk: f64,
    concentration_risk: f64,
}

#[derive(Debug, Clone)]
pub struct AllocationRecommendation {
    pub symbol: String,
    pub recommended_size: Decimal,
    pub allocation_method: AllocationMethod,
    pub confidence_score: f64,
    pub risk_metrics: AllocationRiskMetrics,
    pub rationale: String,
    pub constraints: Vec<String>,
    pub alternative_sizes: HashMap<String, Decimal>,
}

#[derive(Debug, Clone)]
pub struct PortfolioAllocation {
    pub allocations: HashMap<String, AllocationRecommendation>,
    pub total_allocated: Decimal,
    pub remaining_capital: Decimal,
    pub portfolio_risk_metrics: PortfolioRiskMetrics,
    pub optimization_score: f64,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct PortfolioRiskMetrics {
    pub portfolio_var: f64,
    pub expected_return: f64,
    pub sharpe_ratio: f64,
    pub diversification_ratio: f64,
    pub concentration_risk: f64,
}

impl Default for PositionAllocatorConfig {
    fn default() -> Self {
        Self {
            allocation_method: AllocationMethod::RiskParity,
            max_position_size_pct: 0.15,
            min_position_size: Decimal::from(100),
            risk_budget_per_position: 0.02,
            correlation_adjustment: 0.5,
            volatility_params: VolatilityParams {
                lookback_days: 60,
                target_volatility: 0.15,
                min_volatility: 0.05,
                max_volatility: 0.50,
            },
            kelly_params: KellyParams {
                confidence_level: 0.95,
                kelly_multiplier: 0.25,
                min_win_rate: 0.51,
                max_kelly_fraction: 0.10,
            },
        }
    }
}

impl PositionAllocator {
    /// Create a new position allocator
    pub fn new(config: PositionAllocatorConfig) -> Self {
        Self {
            config,
            portfolio_state: PortfolioState::default(),
            risk_metrics_cache: RiskMetricsCache::default(),
            allocation_history: Vec::new(),
        }
    }

    /// Update portfolio state
    pub async fn update_portfolio_state(&mut self, positions: &[Position], available_capital: Decimal) -> Result<()> {
        self.portfolio_state.available_capital = available_capital;
        self.portfolio_state.total_equity = available_capital + positions.iter()
            .map(|p| p.quantity * p.mark_price)
            .sum::<Decimal>();

        self.portfolio_state.current_positions.clear();
        for position in positions {
            self.portfolio_state.current_positions.insert(position.symbol.clone(), position.clone());
        }

        self.portfolio_state.last_updated = Some(Utc::now());
        Ok(())
    }

    /// Calculate position size recommendation for a single symbol
    pub async fn calculate_position_size(
        &mut self, 
        symbol: &str, 
        market_data: &[MarketData],
        signal_strength: f64
    ) -> Result<AllocationRecommendation> {
        // Update risk metrics
        self.update_risk_metrics(symbol, market_data).await?;

        let base_size = match &self.config.allocation_method {
            AllocationMethod::Fixed(size) => *size,
            AllocationMethod::EqualWeight => self.calculate_equal_weight_size()?,
            AllocationMethod::RiskParity => self.calculate_risk_parity_size(symbol).await?,
            AllocationMethod::VolatilityAdjusted => self.calculate_volatility_adjusted_size(symbol).await?,
            AllocationMethod::KellyOptimal => self.calculate_kelly_optimal_size(symbol, signal_strength).await?,
            AllocationMethod::MeanVariance => self.calculate_mean_variance_size(symbol).await?,
            AllocationMethod::RiskBudgeting => self.calculate_risk_budgeting_size(symbol).await?,
        };

        // Apply constraints and adjustments
        let adjusted_size = self.apply_constraints(symbol, base_size).await?;

        // Calculate risk metrics for this allocation
        let risk_metrics = self.calculate_allocation_risk_metrics(symbol, adjusted_size).await?;

        // Calculate confidence score
        let confidence_score = self.calculate_confidence_score(symbol, signal_strength, &risk_metrics);

        // Generate rationale
        let rationale = self.generate_allocation_rationale(symbol, adjusted_size, &risk_metrics);

        // Generate alternative sizes
        let alternative_sizes = self.generate_alternative_sizes(symbol, adjusted_size).await?;

        // Record allocation
        let allocation_record = AllocationRecord {
            timestamp: Utc::now(),
            symbol: symbol.to_string(),
            allocation_method: format!("{:?}", self.config.allocation_method),
            recommended_size: adjusted_size,
            actual_size: adjusted_size, // Would be updated after execution
            confidence_score,
            risk_metrics: risk_metrics.clone(),
        };
        self.allocation_history.push(allocation_record);

        // Maintain history size
        if self.allocation_history.len() > 1000 {
            self.allocation_history.drain(0..100);
        }

        Ok(AllocationRecommendation {
            symbol: symbol.to_string(),
            recommended_size: adjusted_size,
            allocation_method: self.config.allocation_method.clone(),
            confidence_score,
            risk_metrics,
            rationale,
            constraints: self.get_active_constraints(symbol, adjusted_size),
            alternative_sizes,
        })
    }

    /// Calculate optimal portfolio allocation across multiple symbols
    pub async fn calculate_portfolio_allocation(
        &mut self,
        symbols: &[String],
        market_data: &HashMap<String, Vec<MarketData>>,
        signal_strengths: &HashMap<String, f64>
    ) -> Result<PortfolioAllocation> {
        let mut individual_allocations = HashMap::new();
        let mut total_allocated = Decimal::ZERO;

        // Calculate individual allocations
        let empty_vec = Vec::new();
        for symbol in symbols {
            let symbol_data = market_data.get(symbol).unwrap_or(&empty_vec);
            let signal_strength = signal_strengths.get(symbol).copied().unwrap_or(0.0);
            
            let allocation = self.calculate_position_size(symbol, symbol_data, signal_strength).await?;
            total_allocated += allocation.recommended_size;
            individual_allocations.insert(symbol.clone(), allocation);
        }

        // Apply portfolio-level constraints
        let (optimized_allocations, final_total) = 
            self.optimize_portfolio_allocations(individual_allocations).await?;

        // Calculate portfolio risk metrics
        let portfolio_risk_metrics = self.calculate_portfolio_risk_metrics(&optimized_allocations).await?;

        // Calculate optimization score
        let optimization_score = self.calculate_optimization_score(&portfolio_risk_metrics);

        Ok(PortfolioAllocation {
            allocations: optimized_allocations,
            total_allocated: final_total,
            remaining_capital: self.portfolio_state.available_capital - final_total,
            portfolio_risk_metrics,
            optimization_score,
            generated_at: Utc::now(),
        })
    }

    async fn update_risk_metrics(&mut self, symbol: &str, market_data: &[MarketData]) -> Result<()> {
        if market_data.len() < 30 {
            return Ok(()); // Insufficient data
        }

        // Calculate volatility
        let returns: Vec<f64> = market_data
            .windows(2)
            .map(|pair| {
                let prev_price = pair[0].mid.to_f64().unwrap_or(0.0);
                let curr_price = pair[1].mid.to_f64().unwrap_or(0.0);
                if prev_price > 0.0 {
                    (curr_price - prev_price) / prev_price
                } else {
                    0.0
                }
            })
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let volatility = variance.sqrt() * (252.0_f64).sqrt(); // Annualized

        // Constrain volatility within bounds
        let constrained_volatility = volatility
            .max(self.config.volatility_params.min_volatility)
            .min(self.config.volatility_params.max_volatility);

        self.risk_metrics_cache.symbol_volatilities.insert(symbol.to_string(), constrained_volatility);

        // Calculate VaR estimate (simplified)
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let var_index = ((1.0 - 0.95) * sorted_returns.len() as f64) as usize;
        let var_95 = -sorted_returns.get(var_index).unwrap_or(&0.0);
        self.risk_metrics_cache.var_estimates.insert(symbol.to_string(), var_95);

        self.risk_metrics_cache.last_updated = Some(Utc::now());
        Ok(())
    }

    fn calculate_equal_weight_size(&self) -> Result<Decimal> {
        // Simplified equal weight - would consider number of positions
        let target_positions = 10; // Could be configurable
        Ok(self.portfolio_state.available_capital / Decimal::from(target_positions))
    }

    async fn calculate_risk_parity_size(&self, symbol: &str) -> Result<Decimal> {
        let volatility = self.risk_metrics_cache.symbol_volatilities
            .get(symbol)
            .copied()
            .unwrap_or(self.config.volatility_params.target_volatility);

        if volatility <= 0.0 {
            return Ok(self.config.min_position_size);
        }

        // Risk parity: inverse volatility weighting
        let target_risk = self.config.risk_budget_per_position;
        let position_size = self.portfolio_state.available_capital * Decimal::from_f64_retain(target_risk / volatility).unwrap_or_default();

        Ok(position_size.max(self.config.min_position_size))
    }

    async fn calculate_volatility_adjusted_size(&self, symbol: &str) -> Result<Decimal> {
        let volatility = self.risk_metrics_cache.symbol_volatilities
            .get(symbol)
            .copied()
            .unwrap_or(self.config.volatility_params.target_volatility);

        let target_vol = self.config.volatility_params.target_volatility;
        let vol_adjustment = target_vol / volatility.max(0.01);

        let base_size = self.portfolio_state.available_capital * Decimal::from_f64_retain(0.05).unwrap_or_default(); // 5% base
        let adjusted_size = base_size * Decimal::from_f64_retain(vol_adjustment).unwrap_or_default();

        Ok(adjusted_size.max(self.config.min_position_size))
    }

    async fn calculate_kelly_optimal_size(&self, symbol: &str, signal_strength: f64) -> Result<Decimal> {
        if signal_strength < self.config.kelly_params.min_win_rate {
            return Ok(self.config.min_position_size);
        }

        // Simplified Kelly calculation
        let win_probability = signal_strength.max(0.51).min(0.99);
        let win_loss_ratio = 1.5; // Simplified - would calculate from historical data
        
        let kelly_fraction = (win_probability * win_loss_ratio - (1.0 - win_probability)) / win_loss_ratio;
        let constrained_kelly = kelly_fraction
            .max(0.0)
            .min(self.config.kelly_params.max_kelly_fraction) * self.config.kelly_params.kelly_multiplier;

        let position_size = self.portfolio_state.available_capital * Decimal::from_f64_retain(constrained_kelly).unwrap_or_default();
        Ok(position_size.max(self.config.min_position_size))
    }

    async fn calculate_mean_variance_size(&self, symbol: &str) -> Result<Decimal> {
        // Simplified mean-variance optimization
        let expected_return = 0.08; // Would calculate from historical data
        let volatility = self.risk_metrics_cache.symbol_volatilities
            .get(symbol)
            .copied()
            .unwrap_or(0.15);

        let risk_aversion = 3.0; // Risk aversion parameter
        let optimal_weight = expected_return / (risk_aversion * volatility.powi(2));
        let constrained_weight = optimal_weight.max(0.0).min(self.config.max_position_size_pct);

        let position_size = self.portfolio_state.available_capital * Decimal::from_f64_retain(constrained_weight).unwrap_or_default();
        Ok(position_size.max(self.config.min_position_size))
    }

    async fn calculate_risk_budgeting_size(&self, symbol: &str) -> Result<Decimal> {
        let risk_budget = self.config.risk_budget_per_position;
        let portfolio_volatility = 0.12; // Would calculate actual portfolio volatility
        
        let volatility = self.risk_metrics_cache.symbol_volatilities
            .get(symbol)
            .copied()
            .unwrap_or(0.15);

        // Risk budgeting allocation
        let position_volatility_contribution = risk_budget * portfolio_volatility;
        let position_weight = position_volatility_contribution / volatility;
        let constrained_weight = position_weight.max(0.0).min(self.config.max_position_size_pct);

        let position_size = self.portfolio_state.available_capital * Decimal::from_f64_retain(constrained_weight).unwrap_or_default();
        Ok(position_size.max(self.config.min_position_size))
    }

    async fn apply_constraints(&self, symbol: &str, base_size: Decimal) -> Result<Decimal> {
        let mut adjusted_size = base_size;

        // Apply minimum size constraint
        adjusted_size = adjusted_size.max(self.config.min_position_size);

        // Apply maximum position size constraint
        let max_position_value = self.portfolio_state.total_equity * Decimal::from_f64_retain(self.config.max_position_size_pct).unwrap_or_default();
        adjusted_size = adjusted_size.min(max_position_value);

        // Apply available capital constraint
        adjusted_size = adjusted_size.min(self.portfolio_state.available_capital);

        // Apply existing position constraint (if increasing position)
        if let Some(existing_position) = self.portfolio_state.current_positions.get(symbol) {
            let existing_value = existing_position.quantity * existing_position.mark_price;
            let max_additional = max_position_value - existing_value;
            adjusted_size = adjusted_size.min(max_additional.max(Decimal::ZERO));
        }

        Ok(adjusted_size)
    }

    async fn calculate_allocation_risk_metrics(&self, symbol: &str, size: Decimal) -> Result<AllocationRiskMetrics> {
        let volatility = self.risk_metrics_cache.symbol_volatilities
            .get(symbol)
            .copied()
            .unwrap_or(0.15);

        let var_95 = self.risk_metrics_cache.var_estimates
            .get(symbol)
            .copied()
            .unwrap_or(0.02);

        let position_var = (size / self.portfolio_state.total_equity).to_f64().unwrap_or(0.0) * var_95;

        // Simplified correlation and concentration risk
        let correlation_risk = 0.3; // Would calculate from actual correlations
        let concentration_risk = (size / self.portfolio_state.total_equity).to_f64().unwrap_or(0.0);

        Ok(AllocationRiskMetrics {
            estimated_volatility: volatility,
            position_var: position_var,
            correlation_risk,
            concentration_risk,
        })
    }

    fn calculate_confidence_score(&self, _symbol: &str, signal_strength: f64, risk_metrics: &AllocationRiskMetrics) -> f64 {
        // Combine signal strength and risk metrics into confidence score
        let signal_score = signal_strength.max(0.0).min(1.0);
        let risk_score = (1.0 - risk_metrics.concentration_risk).max(0.0);
        let volatility_score = (1.0 - (risk_metrics.estimated_volatility - 0.15).abs() / 0.35).max(0.0);

        (signal_score * 0.5 + risk_score * 0.3 + volatility_score * 0.2).min(1.0)
    }

    fn generate_allocation_rationale(&self, symbol: &str, size: Decimal, risk_metrics: &AllocationRiskMetrics) -> String {
        let method_name = match &self.config.allocation_method {
            AllocationMethod::Fixed(_) => "Fixed Size",
            AllocationMethod::EqualWeight => "Equal Weight",
            AllocationMethod::RiskParity => "Risk Parity",
            AllocationMethod::VolatilityAdjusted => "Volatility Adjusted",
            AllocationMethod::KellyOptimal => "Kelly Optimal",
            AllocationMethod::MeanVariance => "Mean-Variance Optimization",
            AllocationMethod::RiskBudgeting => "Risk Budgeting",
        };

        format!(
            "Allocated {} to {} using {} method. Estimated volatility: {:.2}%, Position VaR: {:.2}%, Concentration risk: {:.2}%",
            size, symbol, method_name, 
            risk_metrics.estimated_volatility * 100.0,
            risk_metrics.position_var * 100.0,
            risk_metrics.concentration_risk * 100.0
        )
    }

    async fn generate_alternative_sizes(&self, symbol: &str, base_size: Decimal) -> Result<HashMap<String, Decimal>> {
        let mut alternatives = HashMap::new();

        // Conservative allocation (50% of base)
        alternatives.insert("Conservative".to_string(), base_size * Decimal::from_f64_retain(0.5).unwrap_or_default());

        // Aggressive allocation (150% of base, capped by constraints)
        let aggressive = base_size * Decimal::from_f64_retain(1.5).unwrap_or_default();
        let constrained_aggressive = self.apply_constraints(symbol, aggressive).await?;
        alternatives.insert("Aggressive".to_string(), constrained_aggressive);

        // Minimum viable allocation
        alternatives.insert("Minimum".to_string(), self.config.min_position_size);

        // Maximum allowed allocation
        let max_allowed = self.portfolio_state.total_equity * Decimal::from_f64_retain(self.config.max_position_size_pct).unwrap_or_default();
        alternatives.insert("Maximum".to_string(), max_allowed);

        Ok(alternatives)
    }

    fn get_active_constraints(&self, _symbol: &str, size: Decimal) -> Vec<String> {
        let mut constraints = Vec::new();

        if size <= self.config.min_position_size {
            constraints.push("Minimum position size constraint active".to_string());
        }

        let max_position_value = self.portfolio_state.total_equity * Decimal::from_f64_retain(self.config.max_position_size_pct).unwrap_or_default();
        if size >= max_position_value {
            constraints.push(format!("Maximum position size constraint active ({}%)", self.config.max_position_size_pct * 100.0));
        }

        if size >= self.portfolio_state.available_capital {
            constraints.push("Available capital constraint active".to_string());
        }

        constraints
    }

    async fn optimize_portfolio_allocations(
        &self,
        mut allocations: HashMap<String, AllocationRecommendation>
    ) -> Result<(HashMap<String, AllocationRecommendation>, Decimal)> {
        let total_requested: Decimal = allocations.values().map(|a| a.recommended_size).sum();
        
        // If total requested exceeds available capital, scale down proportionally
        if total_requested > self.portfolio_state.available_capital {
            let scale_factor = self.portfolio_state.available_capital / total_requested;
            
            for allocation in allocations.values_mut() {
                allocation.recommended_size = allocation.recommended_size * scale_factor;
                allocation.constraints.push(format!("Scaled down by {:.2}% due to capital constraints", (1.0 - scale_factor.to_f64().unwrap_or(1.0)) * 100.0));
            }
        }

        let final_total: Decimal = allocations.values().map(|a| a.recommended_size).sum();
        Ok((allocations, final_total))
    }

    async fn calculate_portfolio_risk_metrics(&self, allocations: &HashMap<String, AllocationRecommendation>) -> Result<PortfolioRiskMetrics> {
        let total_value: Decimal = allocations.values().map(|a| a.recommended_size).sum();
        
        if total_value <= Decimal::ZERO {
            return Ok(PortfolioRiskMetrics {
                portfolio_var: 0.0,
                expected_return: 0.0,
                sharpe_ratio: 0.0,
                diversification_ratio: 1.0,
                concentration_risk: 0.0,
            });
        }

        // Calculate weighted portfolio metrics
        let mut weighted_var = 0.0;
        let mut max_concentration = 0.0;

        for allocation in allocations.values() {
            let weight = (allocation.recommended_size / total_value).to_f64().unwrap_or(0.0);
            weighted_var += weight * allocation.risk_metrics.position_var;
            max_concentration = f64::max(max_concentration, weight);
        }

        // Simplified portfolio metrics
        let portfolio_var = weighted_var;
        let expected_return = 0.08; // Would calculate from individual expected returns
        let sharpe_ratio = expected_return / f64::max(portfolio_var, 0.01);
        let diversification_ratio = allocations.len() as f64 / (1.0 + max_concentration * allocations.len() as f64);

        Ok(PortfolioRiskMetrics {
            portfolio_var,
            expected_return,
            sharpe_ratio,
            diversification_ratio,
            concentration_risk: max_concentration,
        })
    }

    fn calculate_optimization_score(&self, risk_metrics: &PortfolioRiskMetrics) -> f64 {
        // Composite score based on risk-return characteristics
        let return_score = (risk_metrics.expected_return / 0.10).min(1.0); // Normalize to 10% target
        let risk_score = (1.0 - risk_metrics.portfolio_var / 0.05).max(0.0); // Normalize to 5% VaR
        let diversification_score = risk_metrics.diversification_ratio.min(1.0);
        let concentration_score = (1.0 - risk_metrics.concentration_risk).max(0.0);

        (return_score * 0.3 + risk_score * 0.3 + diversification_score * 0.2 + concentration_score * 0.2).min(1.0)
    }

    /// Get allocation performance statistics
    pub async fn get_allocation_performance(&self, days: u32) -> AllocationPerformance {
        let cutoff = Utc::now() - chrono::Duration::days(days as i64);
        let recent_allocations: Vec<&AllocationRecord> = self.allocation_history
            .iter()
            .filter(|record| record.timestamp > cutoff)
            .collect();

        if recent_allocations.is_empty() {
            return AllocationPerformance::empty();
        }

        let total_allocations = recent_allocations.len();
        let average_confidence = recent_allocations.iter()
            .map(|r| r.confidence_score)
            .sum::<f64>() / total_allocations as f64;

        let average_volatility = recent_allocations.iter()
            .map(|r| r.risk_metrics.estimated_volatility)
            .sum::<f64>() / total_allocations as f64;

        AllocationPerformance {
            period_days: days,
            total_allocations: total_allocations as u64,
            average_confidence_score: average_confidence,
            average_volatility: average_volatility,
            method_distribution: self.calculate_method_distribution(&recent_allocations),
            generated_at: Utc::now(),
        }
    }

    fn calculate_method_distribution(&self, allocations: &[&AllocationRecord]) -> HashMap<String, u64> {
        let mut distribution = HashMap::new();
        for allocation in allocations {
            *distribution.entry(allocation.allocation_method.clone()).or_insert(0) += 1;
        }
        distribution
    }
}

#[derive(Debug, Clone)]
pub struct AllocationPerformance {
    pub period_days: u32,
    pub total_allocations: u64,
    pub average_confidence_score: f64,
    pub average_volatility: f64,
    pub method_distribution: HashMap<String, u64>,
    pub generated_at: DateTime<Utc>,
}

impl AllocationPerformance {
    fn empty() -> Self {
        Self {
            period_days: 0,
            total_allocations: 0,
            average_confidence_score: 0.0,
            average_volatility: 0.0,
            method_distribution: HashMap::new(),
            generated_at: Utc::now(),
        }
    }
}