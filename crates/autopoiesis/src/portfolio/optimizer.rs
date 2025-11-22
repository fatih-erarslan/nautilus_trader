//! Portfolio optimization implementation

use crate::prelude::*;
use crate::models::{Position, MarketData};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use num_traits::ToPrimitive;
use std::collections::HashMap;

/// Portfolio optimizer for advanced portfolio optimization techniques
#[derive(Debug, Clone)]
pub struct PortfolioOptimizer {
    /// Optimizer configuration
    config: PortfolioOptimizerConfig,
    
    /// Optimization history
    optimization_history: Vec<OptimizationRecord>,
    
    /// Market data cache
    market_data_cache: MarketDataCache,
    
    /// Performance metrics cache
    performance_cache: PerformanceCache,
}

#[derive(Debug, Clone)]
pub struct PortfolioOptimizerConfig {
    /// Default optimization method
    pub default_method: OptimizationMethod,
    
    /// Risk-free rate for calculations
    pub risk_free_rate: f64,
    
    /// Confidence level for VaR calculations
    pub confidence_level: f64,
    
    /// Historical data lookback period in days
    pub lookback_period_days: u32,
    
    /// Rebalancing constraints
    pub rebalancing_constraints: RebalancingConstraints,
    
    /// Optimization parameters
    pub optimization_params: OptimizationParams,
    
    /// Maximum number of optimization iterations
    pub max_iterations: u32,
    
    /// Convergence tolerance
    pub convergence_tolerance: f64,
}

#[derive(Debug, Clone)]
pub enum OptimizationMethod {
    /// Mean-variance optimization (Markowitz)
    MeanVariance,
    
    /// Risk parity optimization
    RiskParity,
    
    /// Minimum variance optimization
    MinimumVariance,
    
    /// Maximum Sharpe ratio optimization
    MaximumSharpe,
    
    /// Black-Litterman optimization
    BlackLitterman,
    
    /// Hierarchical risk parity
    HierarchicalRiskParity,
    
    /// Critical line algorithm
    CriticalLine,
    
    /// Monte Carlo optimization
    MonteCarlo,
}

#[derive(Debug, Clone)]
pub struct RebalancingConstraints {
    /// Minimum allocation per asset
    pub min_allocation: f64,
    
    /// Maximum allocation per asset
    pub max_allocation: f64,
    
    /// Long-only constraint
    pub long_only: bool,
    
    /// Sector concentration limits
    pub sector_limits: HashMap<String, f64>,
    
    /// Geographic limits
    pub geographic_limits: HashMap<String, f64>,
    
    /// Turnover constraints
    pub max_turnover: Option<f64>,
    
    /// Transaction cost assumptions
    pub transaction_costs: TransactionCosts,
}

#[derive(Debug, Clone)]
pub struct TransactionCosts {
    /// Fixed cost per trade
    pub fixed_cost: Decimal,
    
    /// Variable cost as percentage of trade value
    pub variable_cost_pct: f64,
    
    /// Market impact model parameters
    pub market_impact: MarketImpactModel,
}

#[derive(Debug, Clone)]
pub struct MarketImpactModel {
    /// Temporary impact coefficient
    pub temporary_impact: f64,
    
    /// Permanent impact coefficient
    pub permanent_impact: f64,
    
    /// Liquidity adjustment factor
    pub liquidity_factor: f64,
}

#[derive(Debug, Clone)]
pub struct OptimizationParams {
    /// Target return (if applicable)
    pub target_return: Option<f64>,
    
    /// Risk aversion parameter
    pub risk_aversion: f64,
    
    /// Regularization parameter
    pub regularization: f64,
    
    /// Shrinkage intensity for covariance matrix
    pub shrinkage_intensity: f64,
    
    /// Use robust estimators
    pub use_robust_estimators: bool,
}

#[derive(Debug, Clone, Default)]
struct MarketDataCache {
    price_data: HashMap<String, Vec<PricePoint>>,
    returns_data: HashMap<String, Vec<f64>>,
    covariance_matrix: HashMap<String, HashMap<String, f64>>,
    expected_returns: HashMap<String, f64>,
    last_updated: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
struct PricePoint {
    timestamp: DateTime<Utc>,
    price: Decimal,
}

#[derive(Debug, Clone, Default)]
struct PerformanceCache {
    historical_performance: HashMap<String, AssetPerformance>,
    correlation_matrix: HashMap<String, HashMap<String, f64>>,
    risk_metrics: HashMap<String, AssetRiskMetrics>,
    last_calculated: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
struct AssetPerformance {
    symbol: String,
    expected_return: f64,
    volatility: f64,
    sharpe_ratio: f64,
    max_drawdown: f64,
    var_95: f64,
}

#[derive(Debug, Clone)]
struct AssetRiskMetrics {
    symbol: String,
    beta: f64,
    alpha: f64,
    tracking_error: f64,
    information_ratio: f64,
    downside_deviation: f64,
}

#[derive(Debug, Clone)]
struct OptimizationRecord {
    timestamp: DateTime<Utc>,
    method: OptimizationMethod,
    input_allocations: HashMap<String, f64>,
    optimized_allocations: HashMap<String, f64>,
    expected_return: f64,
    expected_risk: f64,
    sharpe_ratio: f64,
    optimization_score: f64,
    constraints_satisfied: bool,
    convergence_iterations: u32,
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub method: OptimizationMethod,
    pub optimal_allocations: HashMap<String, f64>,
    pub expected_return: f64,
    pub expected_risk: f64,
    pub sharpe_ratio: f64,
    pub optimization_metrics: OptimizationMetrics,
    pub constraint_analysis: ConstraintAnalysis,
    pub risk_attribution: RiskAttribution,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct OptimizationMetrics {
    pub convergence_iterations: u32,
    pub optimization_time_ms: u64,
    pub objective_function_value: f64,
    pub gradient_norm: f64,
    pub constraint_violations: Vec<String>,
    pub diversification_ratio: f64,
    pub effective_number_of_bets: f64,
}

#[derive(Debug, Clone)]
pub struct ConstraintAnalysis {
    pub active_constraints: Vec<String>,
    pub binding_constraints: Vec<String>,
    pub constraint_multipliers: HashMap<String, f64>,
    pub feasibility_score: f64,
}

#[derive(Debug, Clone)]
pub struct RiskAttribution {
    pub total_risk: f64,
    pub idiosyncratic_risk: HashMap<String, f64>,
    pub systematic_risk: f64,
    pub concentration_risk: f64,
    pub correlation_risk: f64,
    pub sector_risk_contribution: HashMap<String, f64>,
}

impl Default for PortfolioOptimizerConfig {
    fn default() -> Self {
        Self {
            default_method: OptimizationMethod::MeanVariance,
            risk_free_rate: 0.02,
            confidence_level: 0.95,
            lookback_period_days: 252,
            rebalancing_constraints: RebalancingConstraints {
                min_allocation: 0.0,
                max_allocation: 0.30,
                long_only: true,
                sector_limits: HashMap::new(),
                geographic_limits: HashMap::new(),
                max_turnover: Some(0.20),
                transaction_costs: TransactionCosts {
                    fixed_cost: Decimal::from(5),
                    variable_cost_pct: 0.001,
                    market_impact: MarketImpactModel {
                        temporary_impact: 0.0001,
                        permanent_impact: 0.00005,
                        liquidity_factor: 1.0,
                    },
                },
            },
            optimization_params: OptimizationParams {
                target_return: None,
                risk_aversion: 3.0,
                regularization: 0.01,
                shrinkage_intensity: 0.1,
                use_robust_estimators: false,
            },
            max_iterations: 1000,
            convergence_tolerance: 1e-6,
        }
    }
}

impl PortfolioOptimizer {
    /// Create a new portfolio optimizer
    pub fn new(config: PortfolioOptimizerConfig) -> Self {
        Self {
            config,
            optimization_history: Vec::new(),
            market_data_cache: MarketDataCache::default(),
            performance_cache: PerformanceCache::default(),
        }
    }

    /// Optimize portfolio allocations
    pub async fn optimize_portfolio(
        &mut self,
        current_positions: &[Position],
        market_data: &HashMap<String, Vec<MarketData>>,
        method: Option<OptimizationMethod>
    ) -> Result<OptimizationResult> {
        let optimization_method = method.unwrap_or(self.config.default_method.clone());
        
        // Update market data cache
        self.update_market_data_cache(market_data).await?;
        
        // Calculate expected returns and covariance matrix
        self.calculate_expected_returns().await?;
        self.calculate_covariance_matrix().await?;
        
        // Get current allocations
        let current_allocations = self.calculate_current_allocations(current_positions)?;
        
        // Perform optimization based on method
        let optimization_start = std::time::Instant::now();
        let result = match optimization_method {
            OptimizationMethod::MeanVariance => self.optimize_mean_variance(&current_allocations).await?,
            OptimizationMethod::RiskParity => self.optimize_risk_parity(&current_allocations).await?,
            OptimizationMethod::MinimumVariance => self.optimize_minimum_variance(&current_allocations).await?,
            OptimizationMethod::MaximumSharpe => self.optimize_maximum_sharpe(&current_allocations).await?,
            OptimizationMethod::BlackLitterman => self.optimize_black_litterman(&current_allocations).await?,
            OptimizationMethod::HierarchicalRiskParity => self.optimize_hierarchical_risk_parity(&current_allocations).await?,
            OptimizationMethod::CriticalLine => self.optimize_critical_line(&current_allocations).await?,
            OptimizationMethod::MonteCarlo => self.optimize_monte_carlo(&current_allocations).await?,
        };
        
        let optimization_time = optimization_start.elapsed().as_millis() as u64;
        
        // Validate constraints
        let constraint_analysis = self.analyze_constraints(&result.optimal_allocations)?;
        
        // Calculate risk attribution
        let risk_attribution = self.calculate_risk_attribution(&result.optimal_allocations).await?;
        
        // Create comprehensive result
        let optimization_result = OptimizationResult {
            method: optimization_method.clone(),
            optimal_allocations: result.optimal_allocations.clone(),
            expected_return: result.expected_return,
            expected_risk: result.expected_risk,
            sharpe_ratio: result.sharpe_ratio,
            optimization_metrics: OptimizationMetrics {
                convergence_iterations: result.convergence_iterations,
                optimization_time_ms: optimization_time,
                objective_function_value: result.objective_function_value,
                gradient_norm: result.gradient_norm,
                constraint_violations: constraint_analysis.binding_constraints.clone(),
                diversification_ratio: self.calculate_diversification_ratio(&result.optimal_allocations).await?,
                effective_number_of_bets: self.calculate_effective_number_of_bets(&result.optimal_allocations).await?,
            },
            constraint_analysis,
            risk_attribution,
            generated_at: Utc::now(),
        };

        // Record optimization
        let optimization_record = OptimizationRecord {
            timestamp: Utc::now(),
            method: optimization_method,
            input_allocations: current_allocations,
            optimized_allocations: result.optimal_allocations,
            expected_return: result.expected_return,
            expected_risk: result.expected_risk,
            sharpe_ratio: result.sharpe_ratio,
            optimization_score: optimization_result.optimization_metrics.objective_function_value,
            constraints_satisfied: optimization_result.constraint_analysis.binding_constraints.is_empty(),
            convergence_iterations: result.convergence_iterations,
        };

        self.optimization_history.push(optimization_record);

        // Maintain history size
        if self.optimization_history.len() > 1000 {
            self.optimization_history.drain(0..100);
        }

        Ok(optimization_result)
    }

    /// Calculate efficient frontier
    pub async fn calculate_efficient_frontier(
        &mut self,
        market_data: &HashMap<String, Vec<MarketData>>,
        num_portfolios: usize
    ) -> Result<Vec<EfficientFrontierPoint>> {
        // Update market data
        self.update_market_data_cache(market_data).await?;
        self.calculate_expected_returns().await?;
        self.calculate_covariance_matrix().await?;

        let mut frontier_points = Vec::new();
        let symbols: Vec<String> = self.market_data_cache.expected_returns.keys().cloned().collect();

        // Calculate minimum and maximum expected returns
        let min_return = self.market_data_cache.expected_returns.values().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_return = self.market_data_cache.expected_returns.values().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Generate target returns along the frontier
        for i in 0..num_portfolios {
            let target_return = min_return + (max_return - min_return) * (i as f64) / ((num_portfolios - 1) as f64);
            
            // Optimize for this target return
            let mut temp_config = self.config.clone();
            temp_config.optimization_params.target_return = Some(target_return);
            temp_config.default_method = OptimizationMethod::MeanVariance;

            let current_allocations = symbols.iter().map(|s| (s.clone(), 1.0 / symbols.len() as f64)).collect();
            
            if let Ok(result) = self.optimize_mean_variance(&current_allocations).await {
                frontier_points.push(EfficientFrontierPoint {
                    expected_return: result.expected_return,
                    expected_risk: result.expected_risk,
                    sharpe_ratio: result.sharpe_ratio,
                    allocations: result.optimal_allocations,
                });
            }
        }

        // Sort by risk
        frontier_points.sort_by(|a, b| a.expected_risk.partial_cmp(&b.expected_risk).unwrap());

        Ok(frontier_points)
    }

    async fn update_market_data_cache(&mut self, market_data: &HashMap<String, Vec<MarketData>>) -> Result<()> {
        for (symbol, data) in market_data {
            let price_points: Vec<PricePoint> = data.iter()
                .map(|md| PricePoint {
                    timestamp: md.timestamp,
                    price: md.mid,
                })
                .collect();

            self.market_data_cache.price_data.insert(symbol.clone(), price_points);

            // Calculate returns
            let returns = self.calculate_returns_for_symbol(symbol)?;
            self.market_data_cache.returns_data.insert(symbol.clone(), returns);
        }

        self.market_data_cache.last_updated = Some(Utc::now());
        Ok(())
    }

    fn calculate_returns_for_symbol(&self, symbol: &str) -> Result<Vec<f64>> {
        let price_data = self.market_data_cache.price_data.get(symbol).ok_or_else(|| {
            Error::Config(format!("No price data found for symbol: {}", symbol))
        })?;

        if price_data.len() < 2 {
            return Ok(Vec::new());
        }

        let returns = price_data.windows(2)
            .map(|window| {
                let prev_price = window[0].price.to_f64().unwrap_or(0.0);
                let curr_price = window[1].price.to_f64().unwrap_or(0.0);
                if prev_price > 0.0 {
                    (curr_price - prev_price) / prev_price
                } else {
                    0.0
                }
            })
            .collect();

        Ok(returns)
    }

    async fn calculate_expected_returns(&mut self) -> Result<()> {
        for (symbol, returns) in &self.market_data_cache.returns_data {
            if !returns.is_empty() {
                let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
                let annualized_return = mean_return * 252.0; // Assuming daily returns
                self.market_data_cache.expected_returns.insert(symbol.clone(), annualized_return);
            }
        }
        Ok(())
    }

    async fn calculate_covariance_matrix(&mut self) -> Result<()> {
        let symbols: Vec<String> = self.market_data_cache.returns_data.keys().cloned().collect();
        
        for symbol1 in &symbols {
            let mut row = HashMap::new();
            
            for symbol2 in &symbols {
                let covariance = if symbol1 == symbol2 {
                    // Calculate variance
                    if let Some(returns) = self.market_data_cache.returns_data.get(symbol1) {
                        if !returns.is_empty() {
                            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                            let variance = returns.iter()
                                .map(|r| (r - mean).powi(2))
                                .sum::<f64>() / returns.len() as f64;
                            variance * 252.0 // Annualized variance
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    }
                } else {
                    // Calculate covariance
                    if let (Some(returns1), Some(returns2)) = (
                        self.market_data_cache.returns_data.get(symbol1),
                        self.market_data_cache.returns_data.get(symbol2)
                    ) {
                        if returns1.len() == returns2.len() && !returns1.is_empty() {
                            let mean1 = returns1.iter().sum::<f64>() / returns1.len() as f64;
                            let mean2 = returns2.iter().sum::<f64>() / returns2.len() as f64;
                            
                            let covariance = returns1.iter().zip(returns2.iter())
                                .map(|(r1, r2)| (r1 - mean1) * (r2 - mean2))
                                .sum::<f64>() / returns1.len() as f64;
                            
                            covariance * 252.0 // Annualized covariance
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    }
                };
                
                row.insert(symbol2.clone(), covariance);
            }
            
            self.market_data_cache.covariance_matrix.insert(symbol1.clone(), row);
        }
        
        Ok(())
    }

    fn calculate_current_allocations(&self, positions: &[Position]) -> Result<HashMap<String, f64>> {
        let total_value: Decimal = positions.iter()
            .map(|p| p.quantity * p.mark_price)
            .sum();

        if total_value <= Decimal::ZERO {
            return Ok(HashMap::new());
        }

        let mut allocations = HashMap::new();
        for position in positions {
            if position.quantity > Decimal::ZERO {
                let position_value = position.quantity * position.mark_price;
                let allocation = (position_value / total_value).to_f64().unwrap_or(0.0);
                allocations.insert(position.symbol.clone(), allocation);
            }
        }

        Ok(allocations)
    }

    async fn optimize_mean_variance(&self, _current_allocations: &HashMap<String, f64>) -> Result<OptimizationResultInternal> {
        // Simplified mean-variance optimization
        let symbols: Vec<String> = self.market_data_cache.expected_returns.keys().cloned().collect();
        let n = symbols.len();

        if n == 0 {
            return Err(Error::Config("No assets available for optimization".to_string()));
        }

        // Equal weight allocation as starting point
        let equal_weight = 1.0 / n as f64;
        let mut optimal_weights = HashMap::new();
        
        for symbol in &symbols {
            optimal_weights.insert(symbol.clone(), equal_weight);
        }

        // Calculate portfolio metrics
        let expected_return = self.calculate_portfolio_return(&optimal_weights)?;
        let expected_risk = self.calculate_portfolio_risk(&optimal_weights)?;
        let sharpe_ratio = if expected_risk > 0.0 {
            (expected_return - self.config.risk_free_rate) / expected_risk
        } else {
            0.0
        };

        Ok(OptimizationResultInternal {
            optimal_allocations: optimal_weights,
            expected_return,
            expected_risk,
            sharpe_ratio,
            convergence_iterations: 1,
            objective_function_value: sharpe_ratio,
            gradient_norm: 0.0,
        })
    }

    async fn optimize_risk_parity(&self, _current_allocations: &HashMap<String, f64>) -> Result<OptimizationResultInternal> {
        // Risk parity optimization - equal risk contribution
        let symbols: Vec<String> = self.market_data_cache.covariance_matrix.keys().cloned().collect();
        let n = symbols.len();

        if n == 0 {
            return Err(Error::Config("No assets available for optimization".to_string()));
        }

        let mut optimal_weights = HashMap::new();
        
        // Calculate inverse volatility weights
        let mut total_inv_vol = 0.0;
        let mut inv_volatilities = HashMap::new();
        
        for symbol in &symbols {
            if let Some(variance) = self.market_data_cache.covariance_matrix.get(symbol)
                .and_then(|row| row.get(symbol)) {
                let volatility = variance.sqrt().max(0.01); // Avoid division by zero
                let inv_vol = 1.0 / volatility;
                inv_volatilities.insert(symbol.clone(), inv_vol);
                total_inv_vol += inv_vol;
            }
        }

        // Normalize to get risk parity weights
        for symbol in &symbols {
            if let Some(inv_vol) = inv_volatilities.get(symbol) {
                let weight = inv_vol / total_inv_vol;
                optimal_weights.insert(symbol.clone(), weight);
            }
        }

        let expected_return = self.calculate_portfolio_return(&optimal_weights)?;
        let expected_risk = self.calculate_portfolio_risk(&optimal_weights)?;
        let sharpe_ratio = if expected_risk > 0.0 {
            (expected_return - self.config.risk_free_rate) / expected_risk
        } else {
            0.0
        };

        Ok(OptimizationResultInternal {
            optimal_allocations: optimal_weights,
            expected_return,
            expected_risk,
            sharpe_ratio,
            convergence_iterations: 1,
            objective_function_value: -expected_risk, // Minimizing risk
            gradient_norm: 0.0,
        })
    }

    async fn optimize_minimum_variance(&self, _current_allocations: &HashMap<String, f64>) -> Result<OptimizationResultInternal> {
        // Minimum variance optimization (simplified)
        let symbols: Vec<String> = self.market_data_cache.covariance_matrix.keys().cloned().collect();
        let n = symbols.len();

        if n == 0 {
            return Err(Error::Config("No assets available for optimization".to_string()));
        }

        // Use inverse variance weighting as approximation
        let mut optimal_weights = HashMap::new();
        let mut total_inv_var = 0.0;
        
        for symbol in &symbols {
            if let Some(variance) = self.market_data_cache.covariance_matrix.get(symbol)
                .and_then(|row| row.get(symbol)) {
                let inv_var = 1.0 / variance.max(0.0001);
                optimal_weights.insert(symbol.clone(), inv_var);
                total_inv_var += inv_var;
            }
        }

        // Normalize weights
        for (_, weight) in optimal_weights.iter_mut() {
            *weight /= total_inv_var;
        }

        let expected_return = self.calculate_portfolio_return(&optimal_weights)?;
        let expected_risk = self.calculate_portfolio_risk(&optimal_weights)?;
        let sharpe_ratio = if expected_risk > 0.0 {
            (expected_return - self.config.risk_free_rate) / expected_risk
        } else {
            0.0
        };

        Ok(OptimizationResultInternal {
            optimal_allocations: optimal_weights,
            expected_return,
            expected_risk,
            sharpe_ratio,
            convergence_iterations: 1,
            objective_function_value: expected_risk,
            gradient_norm: 0.0,
        })
    }

    async fn optimize_maximum_sharpe(&self, _current_allocations: &HashMap<String, f64>) -> Result<OptimizationResultInternal> {
        // Maximum Sharpe ratio optimization (simplified)
        let symbols: Vec<String> = self.market_data_cache.expected_returns.keys().cloned().collect();
        let n = symbols.len();

        if n == 0 {
            return Err(Error::Config("No assets available for optimization".to_string()));
        }

        // Simple heuristic: weight by expected return / variance
        let mut optimal_weights = HashMap::new();
        let mut total_weight = 0.0;

        for symbol in &symbols {
            if let (Some(expected_return), Some(variance)) = (
                self.market_data_cache.expected_returns.get(symbol),
                self.market_data_cache.covariance_matrix.get(symbol)
                    .and_then(|row| row.get(symbol))
            ) {
                let excess_return = expected_return - self.config.risk_free_rate;
                let weight = if *variance > 0.0 && excess_return > 0.0 {
                    excess_return / variance
                } else {
                    0.0
                };
                optimal_weights.insert(symbol.clone(), weight);
                total_weight += weight;
            }
        }

        // Normalize weights
        if total_weight > 0.0 {
            for (_, weight) in optimal_weights.iter_mut() {
                *weight /= total_weight;
            }
        } else {
            // Fallback to equal weights
            let equal_weight = 1.0 / n as f64;
            for symbol in &symbols {
                optimal_weights.insert(symbol.clone(), equal_weight);
            }
        }

        let expected_return = self.calculate_portfolio_return(&optimal_weights)?;
        let expected_risk = self.calculate_portfolio_risk(&optimal_weights)?;
        let sharpe_ratio = if expected_risk > 0.0 {
            (expected_return - self.config.risk_free_rate) / expected_risk
        } else {
            0.0
        };

        Ok(OptimizationResultInternal {
            optimal_allocations: optimal_weights,
            expected_return,
            expected_risk,
            sharpe_ratio,
            convergence_iterations: 1,
            objective_function_value: sharpe_ratio,
            gradient_norm: 0.0,
        })
    }

    // Placeholder implementations for other optimization methods
    async fn optimize_black_litterman(&self, current_allocations: &HashMap<String, f64>) -> Result<OptimizationResultInternal> {
        // For now, fallback to mean-variance
        self.optimize_mean_variance(current_allocations).await
    }

    async fn optimize_hierarchical_risk_parity(&self, current_allocations: &HashMap<String, f64>) -> Result<OptimizationResultInternal> {
        // For now, fallback to risk parity
        self.optimize_risk_parity(current_allocations).await
    }

    async fn optimize_critical_line(&self, current_allocations: &HashMap<String, f64>) -> Result<OptimizationResultInternal> {
        // For now, fallback to mean-variance
        self.optimize_mean_variance(current_allocations).await
    }

    async fn optimize_monte_carlo(&self, current_allocations: &HashMap<String, f64>) -> Result<OptimizationResultInternal> {
        // For now, fallback to mean-variance
        self.optimize_mean_variance(current_allocations).await
    }

    fn calculate_portfolio_return(&self, weights: &HashMap<String, f64>) -> Result<f64> {
        let mut portfolio_return = 0.0;
        
        for (symbol, weight) in weights {
            if let Some(expected_return) = self.market_data_cache.expected_returns.get(symbol) {
                portfolio_return += weight * expected_return;
            }
        }
        
        Ok(portfolio_return)
    }

    fn calculate_portfolio_risk(&self, weights: &HashMap<String, f64>) -> Result<f64> {
        let mut portfolio_variance = 0.0;
        
        for (symbol1, weight1) in weights {
            for (symbol2, weight2) in weights {
                if let Some(covariance) = self.market_data_cache.covariance_matrix.get(symbol1)
                    .and_then(|row| row.get(symbol2)) {
                    portfolio_variance += weight1 * weight2 * covariance;
                }
            }
        }
        
        Ok(portfolio_variance.sqrt().max(0.0))
    }

    fn analyze_constraints(&self, allocations: &HashMap<String, f64>) -> Result<ConstraintAnalysis> {
        let mut active_constraints = Vec::new();
        let mut binding_constraints = Vec::new();
        let mut constraint_multipliers = HashMap::new();
        let mut feasibility_score = 1.0;

        // Check allocation bounds
        for (symbol, allocation) in allocations {
            if *allocation <= self.config.rebalancing_constraints.min_allocation + 1e-6 {
                active_constraints.push(format!("Min allocation constraint for {}", symbol));
                if *allocation < self.config.rebalancing_constraints.min_allocation {
                    binding_constraints.push(format!("Min allocation violation for {}", symbol));
                    feasibility_score *= 0.9;
                }
            }
            
            if *allocation >= self.config.rebalancing_constraints.max_allocation - 1e-6 {
                active_constraints.push(format!("Max allocation constraint for {}", symbol));
                if *allocation > self.config.rebalancing_constraints.max_allocation {
                    binding_constraints.push(format!("Max allocation violation for {}", symbol));
                    feasibility_score *= 0.9;
                }
            }
        }

        // Check sum constraint
        let total_allocation: f64 = allocations.values().sum();
        if (total_allocation - 1.0).abs() > 1e-6 {
            binding_constraints.push("Sum constraint violation".to_string());
            feasibility_score *= 0.8;
        }

        // Long-only constraint
        if self.config.rebalancing_constraints.long_only {
            for (symbol, allocation) in allocations {
                if *allocation < 0.0 {
                    binding_constraints.push(format!("Long-only constraint violation for {}", symbol));
                    feasibility_score *= 0.7;
                }
            }
        }

        Ok(ConstraintAnalysis {
            active_constraints,
            binding_constraints,
            constraint_multipliers,
            feasibility_score,
        })
    }

    async fn calculate_risk_attribution(&self, allocations: &HashMap<String, f64>) -> Result<RiskAttribution> {
        let total_risk = self.calculate_portfolio_risk(allocations)?;
        let mut idiosyncratic_risk = HashMap::new();
        let mut sector_risk_contribution = HashMap::new();

        // Calculate individual asset risk contributions
        for (symbol, weight) in allocations {
            if let Some(variance) = self.market_data_cache.covariance_matrix.get(symbol)
                .and_then(|row| row.get(symbol)) {
                let asset_risk = weight * variance.sqrt();
                idiosyncratic_risk.insert(symbol.clone(), asset_risk);
            }
        }

        // Simplified sector risk attribution
        sector_risk_contribution.insert("Technology".to_string(), total_risk * 0.4);
        sector_risk_contribution.insert("Finance".to_string(), total_risk * 0.3);
        sector_risk_contribution.insert("Other".to_string(), total_risk * 0.3);

        Ok(RiskAttribution {
            total_risk,
            idiosyncratic_risk,
            systematic_risk: total_risk * 0.7, // Simplified
            concentration_risk: self.calculate_concentration_risk(allocations),
            correlation_risk: total_risk * 0.2, // Simplified
            sector_risk_contribution,
        })
    }

    fn calculate_concentration_risk(&self, allocations: &HashMap<String, f64>) -> f64 {
        // Calculate Herfindahl index
        allocations.values().map(|w| w.powi(2)).sum::<f64>()
    }

    async fn calculate_diversification_ratio(&self, allocations: &HashMap<String, f64>) -> Result<f64> {
        let portfolio_risk = self.calculate_portfolio_risk(allocations)?;
        
        let weighted_average_volatility: f64 = allocations.iter()
            .filter_map(|(symbol, weight)| {
                self.market_data_cache.covariance_matrix.get(symbol)
                    .and_then(|row| row.get(symbol))
                    .map(|variance| weight * variance.sqrt())
            })
            .sum();

        if portfolio_risk > 0.0 {
            Ok(weighted_average_volatility / portfolio_risk)
        } else {
            Ok(1.0)
        }
    }

    async fn calculate_effective_number_of_bets(&self, allocations: &HashMap<String, f64>) -> Result<f64> {
        // Simplified effective number of bets calculation
        let sum_of_squares: f64 = allocations.values().map(|w| w.powi(2)).sum();
        Ok(1.0 / sum_of_squares)
    }

    /// Get optimization history
    pub fn get_optimization_history(&self, limit: Option<usize>) -> Vec<OptimizationRecord> {
        let limit = limit.unwrap_or(100);
        self.optimization_history.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
}

#[derive(Debug, Clone)]
struct OptimizationResultInternal {
    optimal_allocations: HashMap<String, f64>,
    expected_return: f64,
    expected_risk: f64,
    sharpe_ratio: f64,
    convergence_iterations: u32,
    objective_function_value: f64,
    gradient_norm: f64,
}

#[derive(Debug, Clone)]
pub struct EfficientFrontierPoint {
    pub expected_return: f64,
    pub expected_risk: f64,
    pub sharpe_ratio: f64,
    pub allocations: HashMap<String, f64>,
}