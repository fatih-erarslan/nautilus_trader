//! Stress testing and scenario analysis with Monte Carlo GPU acceleration

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use crate::quantum_uncertainty::QuantumUncertaintyEngine;
use rand::prelude::*;
use rand_distr::{Distribution, Normal, Uniform as UniformDist};
use statrs::distribution::StudentsT;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::config::StressConfig;
use crate::error::{RiskError, RiskResult};
use crate::gpu::GpuMonteCarloEngine;
use crate::types::{Portfolio, Position, StressScenario, MarketData};

/// Quantum portfolio data (local definition for stress testing)
#[derive(Debug, Clone)]
pub struct QuantumPortfolioData {
    /// Return matrix
    pub returns: Array2<f64>,
    /// Target returns
    pub targets: Array1<f64>,
    /// Position data
    pub positions: Vec<Position>,
    /// Market data
    pub market_data: MarketData,
}

/// Stress test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResults {
    /// Scenario results
    pub scenario_results: Vec<ScenarioResult>,
    /// Overall portfolio impact
    pub overall_impact: OverallStressImpact,
    /// Monte Carlo simulation results
    pub monte_carlo_results: Option<MonteCarloStressResults>,
    /// Tail risk analysis
    pub tail_risk_analysis: TailRiskAnalysis,
    /// Stress test timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Computation time
    pub computation_time: Duration,
}

/// Individual scenario result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    /// Scenario name
    pub scenario_name: String,
    /// Portfolio P&L under scenario
    pub portfolio_pnl: f64,
    /// Portfolio P&L percentage
    pub portfolio_pnl_percent: f64,
    /// Position-level impacts
    pub position_impacts: HashMap<String, f64>,
    /// Risk metrics under stress
    pub stressed_risk_metrics: StressedRiskMetrics,
    /// Liquidity impact
    pub liquidity_impact: f64,
    /// Recovery time estimate
    pub recovery_time_estimate: Duration,
}

/// Stressed risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressedRiskMetrics {
    /// VaR under stress
    pub stressed_var: f64,
    /// CVaR under stress
    pub stressed_cvar: f64,
    /// Volatility under stress
    pub stressed_volatility: f64,
    /// Maximum drawdown under stress
    pub stressed_max_drawdown: f64,
    /// Sharpe ratio under stress
    pub stressed_sharpe: f64,
}

/// Overall stress impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallStressImpact {
    /// Worst-case scenario loss
    pub worst_case_loss: f64,
    /// Average stress loss
    pub average_stress_loss: f64,
    /// Probability of severe loss
    pub severe_loss_probability: f64,
    /// Portfolio resilience score
    pub resilience_score: f64,
    /// Diversification effectiveness
    pub diversification_effectiveness: f64,
}

/// Monte Carlo stress results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloStressResults {
    /// Number of simulations run
    pub num_simulations: usize,
    /// Percentile losses
    pub percentile_losses: HashMap<String, f64>,
    /// Expected shortfall
    pub expected_shortfall: f64,
    /// Probability of ruin
    pub probability_of_ruin: f64,
    /// Time to recovery distribution
    pub recovery_time_distribution: Vec<Duration>,
}

/// Tail risk analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TailRiskAnalysis {
    /// Extreme value statistics
    pub extreme_value_stats: ExtremeValueStats,
    /// Black swan scenarios
    pub black_swan_scenarios: Vec<BlackSwanScenario>,
    /// Tail dependency measures
    pub tail_dependencies: HashMap<String, f64>,
    /// Quantum tail risk enhancement
    pub quantum_tail_enhancement: Option<QuantumTailRisk>,
}

/// Extreme value statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtremeValueStats {
    /// Shape parameter (xi)
    pub shape_parameter: f64,
    /// Scale parameter (sigma)
    pub scale_parameter: f64,
    /// Location parameter (mu)
    pub location_parameter: f64,
    /// Return level estimates
    pub return_levels: HashMap<String, f64>,
}

/// Black swan scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackSwanScenario {
    /// Scenario description
    pub description: String,
    /// Estimated probability
    pub probability: f64,
    /// Potential loss
    pub potential_loss: f64,
    /// Market factors involved
    pub market_factors: Vec<String>,
}

/// Quantum tail risk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumTailRisk {
    /// Quantum enhancement factor
    pub quantum_enhancement: f64,
    /// Tail dependency quantum correction
    pub tail_dependency_correction: f64,
    /// Quantum coherence in tail events
    pub tail_coherence: f64,
}

/// Stress tester with GPU acceleration
pub struct StressTester {
    /// Configuration
    config: StressConfig,
    /// GPU Monte Carlo engine
    gpu_engine: Arc<RwLock<GpuMonteCarloEngine>>,
    /// Quantum uncertainty engine
    quantum_engine: Arc<RwLock<QuantumUncertaintyEngine>>,
    /// Random number generator
    rng: Arc<RwLock<StdRng>>,
    /// Scenario cache
    scenario_cache: Arc<RwLock<HashMap<String, StressScenario>>>,
}

impl StressTester {
    /// Create new stress tester
    pub async fn new(
        config: StressConfig,
        quantum_engine: Arc<RwLock<QuantumUncertaintyEngine>>,
    ) -> Result<Self> {
        let gpu_config = crate::config::GpuConfig::default();
        let gpu_engine = Arc::new(RwLock::new(
            GpuMonteCarloEngine::new(gpu_config).await?
        ));
        
        let rng = Arc::new(RwLock::new(StdRng::from_entropy()));
        
        Ok(Self {
            config,
            gpu_engine,
            quantum_engine,
            rng,
            scenario_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Run comprehensive stress tests
    pub async fn run_stress_tests(
        &self,
        portfolio: &Portfolio,
        scenarios: &[StressScenario],
    ) -> RiskResult<StressTestResults> {
        let start_time = Instant::now();
        info!("Running comprehensive stress tests on portfolio");
        
        // Validate inputs
        if portfolio.positions.is_empty() {
            return Err(RiskError::insufficient_data("Portfolio has no positions"));
        }
        
        if scenarios.is_empty() {
            return Err(RiskError::insufficient_data("No stress scenarios provided"));
        }
        
        // Run individual scenario tests
        let scenario_results = self.run_scenario_tests(portfolio, scenarios).await?;
        
        // Calculate overall impact
        let overall_impact = self.calculate_overall_impact(&scenario_results)?;
        
        // Run Monte Carlo stress simulations if enabled
        let monte_carlo_results = if self.config.monte_carlo_simulations > 0 {
            Some(self.run_monte_carlo_stress_tests(portfolio).await?)
        } else {
            None
        };
        
        // Perform tail risk analysis
        let tail_risk_analysis = self.perform_tail_risk_analysis(portfolio, &scenario_results).await?;
        
        let computation_time = start_time.elapsed();
        
        Ok(StressTestResults {
            scenario_results,
            overall_impact,
            monte_carlo_results,
            tail_risk_analysis,
            timestamp: chrono::Utc::now(),
            computation_time,
        })
    }
    
    /// Run individual scenario tests
    async fn run_scenario_tests(
        &self,
        portfolio: &Portfolio,
        scenarios: &[StressScenario],
    ) -> RiskResult<Vec<ScenarioResult>> {
        info!("Running {} stress scenarios", scenarios.len());
        
        // Process scenarios in parallel
        let results: Result<Vec<_>, _> = scenarios
            .par_iter()
            .map(|scenario| {
                futures::executor::block_on(async {
                    self.run_single_scenario_test(portfolio, scenario).await
                })
            })
            .collect();
        
        results.map_err(|e| RiskError::stress_testing(format!("Scenario execution failed: {}", e)))
    }
    
    /// Run single scenario test
    async fn run_single_scenario_test(
        &self,
        portfolio: &Portfolio,
        scenario: &StressScenario,
    ) -> RiskResult<ScenarioResult> {
        debug!("Running stress scenario: {}", scenario.name);
        
        // Apply scenario shocks to portfolio
        let stressed_portfolio = self.apply_scenario_shocks(portfolio, scenario)?;
        
        // Calculate portfolio P&L
        let portfolio_pnl = self.calculate_scenario_pnl(portfolio, &stressed_portfolio)?;
        let portfolio_pnl_percent = portfolio_pnl / portfolio.total_value * 100.0;
        
        // Calculate position-level impacts
        let position_impacts = self.calculate_position_impacts(portfolio, &stressed_portfolio)?;
        
        // Calculate stressed risk metrics
        let stressed_risk_metrics = self.calculate_stressed_risk_metrics(&stressed_portfolio)?;
        
        // Calculate liquidity impact
        let liquidity_impact = self.calculate_liquidity_impact(scenario)?;
        
        // Estimate recovery time
        let recovery_time_estimate = self.estimate_recovery_time(scenario, portfolio_pnl_percent)?;
        
        Ok(ScenarioResult {
            scenario_name: scenario.name.clone(),
            portfolio_pnl,
            portfolio_pnl_percent,
            position_impacts,
            stressed_risk_metrics,
            liquidity_impact,
            recovery_time_estimate,
        })
    }
    
    /// Apply scenario shocks to portfolio
    fn apply_scenario_shocks(
        &self,
        portfolio: &Portfolio,
        scenario: &StressScenario,
    ) -> RiskResult<Portfolio> {
        let mut stressed_portfolio = portfolio.clone();
        
        // Apply asset-specific shocks
        for position in &mut stressed_portfolio.positions {
            if let Some(&shock) = scenario.asset_shocks.get(&position.symbol) {
                let new_price = position.price * (1.0 + shock);
                position.price = new_price;
                position.market_value = position.quantity * new_price;
                position.pnl = (new_price - position.entry_price) * position.quantity;
            }
        }
        
        // Apply volatility multipliers
        for asset in &mut stressed_portfolio.assets {
            if let Some(&multiplier) = scenario.volatility_multipliers.get(&asset.symbol) {
                asset.volatility *= multiplier;
            }
        }
        
        // Apply correlation shifts (simplified)
        if !scenario.correlation_shifts.is_empty() {
            // In practice, would adjust the full correlation matrix
            warn!("Correlation shifts not fully implemented in simplified version");
        }
        
        // Recalculate portfolio value
        stressed_portfolio.total_value = stressed_portfolio.calculate_value();
        
        Ok(stressed_portfolio)
    }
    
    /// Calculate scenario P&L
    fn calculate_scenario_pnl(
        &self,
        original_portfolio: &Portfolio,
        stressed_portfolio: &Portfolio,
    ) -> RiskResult<f64> {
        let original_value = original_portfolio.calculate_value();
        let stressed_value = stressed_portfolio.calculate_value();
        Ok(stressed_value - original_value)
    }
    
    /// Calculate position-level impacts
    fn calculate_position_impacts(
        &self,
        original_portfolio: &Portfolio,
        stressed_portfolio: &Portfolio,
    ) -> RiskResult<HashMap<String, f64>> {
        let mut impacts = HashMap::new();
        
        for (original_pos, stressed_pos) in 
            original_portfolio.positions.iter().zip(stressed_portfolio.positions.iter()) {
            
            if original_pos.symbol != stressed_pos.symbol {
                return Err(RiskError::data_validation("Position symbol mismatch"));
            }
            
            let impact = (stressed_pos.market_value - original_pos.market_value) / 
                original_pos.market_value * 100.0;
            impacts.insert(original_pos.symbol.clone(), impact);
        }
        
        Ok(impacts)
    }
    
    /// Calculate stressed risk metrics
    fn calculate_stressed_risk_metrics(
        &self,
        stressed_portfolio: &Portfolio,
    ) -> RiskResult<StressedRiskMetrics> {
        // Simplified risk metrics calculation under stress
        let volatility = stressed_portfolio.calculate_volatility();
        let expected_return = stressed_portfolio.calculate_expected_return();
        
        // Estimate stressed VaR (simplified)
        let stressed_var = volatility * 2.33; // 99% confidence level approximation
        let stressed_cvar = stressed_var * 1.3; // CVaR typically higher than VaR
        
        // Estimate other metrics
        let stressed_max_drawdown = volatility * 3.0; // Simplified estimate
        let stressed_sharpe = if volatility > 0.0 {
            expected_return / volatility
        } else {
            0.0
        };
        
        Ok(StressedRiskMetrics {
            stressed_var,
            stressed_cvar,
            stressed_volatility: volatility,
            stressed_max_drawdown,
            stressed_sharpe,
        })
    }
    
    /// Calculate liquidity impact
    fn calculate_liquidity_impact(&self, scenario: &StressScenario) -> RiskResult<f64> {
        // Average liquidity impact across all affected assets
        let total_impact: f64 = scenario.liquidity_impacts.values().sum();
        let count = scenario.liquidity_impacts.len() as f64;
        
        if count > 0.0 {
            Ok(total_impact / count)
        } else {
            Ok(0.0)
        }
    }
    
    /// Estimate recovery time
    fn estimate_recovery_time(
        &self,
        scenario: &StressScenario,
        loss_percent: f64,
    ) -> RiskResult<Duration> {
        // Simple heuristic: recovery time based on loss severity and scenario probability
        let base_recovery_days = match loss_percent.abs() {
            x if x > 50.0 => 730.0, // 2 years for extreme losses
            x if x > 30.0 => 365.0, // 1 year for severe losses
            x if x > 20.0 => 180.0, // 6 months for moderate losses
            x if x > 10.0 => 90.0,  // 3 months for mild losses
            _ => 30.0,              // 1 month for small losses
        };
        
        // Adjust based on scenario probability (rare events take longer to recover)
        let probability_factor = if scenario.probability > 0.0 {
            1.0 / scenario.probability.sqrt()
        } else {
            2.0
        };
        
        let recovery_days = base_recovery_days * probability_factor.min(5.0);
        Ok(Duration::from_secs((recovery_days * 24.0 * 3600.0) as u64))
    }
    
    /// Calculate overall impact
    fn calculate_overall_impact(
        &self,
        scenario_results: &[ScenarioResult],
    ) -> RiskResult<OverallStressImpact> {
        if scenario_results.is_empty() {
            return Err(RiskError::insufficient_data("No scenario results available"));
        }
        
        let losses: Vec<f64> = scenario_results
            .iter()
            .map(|r| r.portfolio_pnl)
            .collect();
        
        let worst_case_loss = losses.iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0)
            .abs();
        
        let average_stress_loss = losses.iter().sum::<f64>() / losses.len() as f64;
        
        // Count scenarios with losses > 20% as severe
        let severe_loss_count = scenario_results
            .iter()
            .filter(|r| r.portfolio_pnl_percent < -20.0)
            .count();
        let severe_loss_probability = severe_loss_count as f64 / scenario_results.len() as f64;
        
        // Calculate resilience score (0-100, higher is better)
        let resilience_score = 100.0 * (1.0 - severe_loss_probability).max(0.0);
        
        // Calculate diversification effectiveness
        let diversification_effectiveness = self.calculate_diversification_effectiveness(scenario_results)?;
        
        Ok(OverallStressImpact {
            worst_case_loss,
            average_stress_loss: average_stress_loss.abs(),
            severe_loss_probability,
            resilience_score,
            diversification_effectiveness,
        })
    }
    
    /// Calculate diversification effectiveness
    fn calculate_diversification_effectiveness(
        &self,
        scenario_results: &[ScenarioResult],
    ) -> RiskResult<f64> {
        // Measure how well diversification reduces risk across scenarios
        let mut position_volatilities = HashMap::new();
        
        for result in scenario_results {
            for (symbol, impact) in &result.position_impacts {
                position_volatilities
                    .entry(symbol.clone())
                    .or_insert_with(Vec::new)
                    .push(*impact);
            }
        }
        
        // Calculate coefficient of variation for each position
        let mut cv_values = Vec::new();
        for impacts in position_volatilities.values() {
            if impacts.len() > 1 {
                let mean = impacts.iter().sum::<f64>() / impacts.len() as f64;
                let variance = impacts.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / impacts.len() as f64;
                let std_dev = variance.sqrt();
                
                if mean.abs() > 1e-6 {
                    cv_values.push(std_dev / mean.abs());
                }
            }
        }
        
        if cv_values.is_empty() {
            Ok(0.0)
        } else {
            let avg_cv = cv_values.iter().sum::<f64>() / cv_values.len() as f64;
            // Lower CV indicates better diversification
            Ok((1.0 / (1.0 + avg_cv)).min(1.0))
        }
    }
    
    /// Run Monte Carlo stress tests
    async fn run_monte_carlo_stress_tests(
        &self,
        portfolio: &Portfolio,
    ) -> RiskResult<MonteCarloStressResults> {
        info!("Running Monte Carlo stress simulations");
        
        if self.config.enable_gpu {
            self.run_gpu_monte_carlo_stress(portfolio).await
        } else {
            self.run_cpu_monte_carlo_stress(portfolio).await
        }
    }
    
    /// Run GPU-accelerated Monte Carlo stress tests
    async fn run_gpu_monte_carlo_stress(
        &self,
        portfolio: &Portfolio,
    ) -> RiskResult<MonteCarloStressResults> {
        let gpu_engine = self.gpu_engine.read().await;
        
        let time_horizon = Duration::from_secs(30 * 24 * 3600); // 30 days
        let gpu_results = gpu_engine.run_simulation(
            portfolio,
            self.config.monte_carlo_simulations as u32,
            time_horizon,
        ).await.map_err(|e| RiskError::monte_carlo_simulation(e))?;
        
        // Convert GPU results to stress test format
        let mut percentile_losses = HashMap::new();
        percentile_losses.insert("99%".to_string(), gpu_results.var_estimates.get("99%").unwrap_or(&0.0).abs());
        percentile_losses.insert("95%".to_string(), gpu_results.var_estimates.get("95%").unwrap_or(&0.0).abs());
        percentile_losses.insert("90%".to_string(), gpu_results.var_estimates.get("90%").unwrap_or(&0.0).abs());
        
        let expected_shortfall = gpu_results.expected_shortfall.abs();
        
        // Calculate probability of ruin (loss > 50% of portfolio)
        let ruin_threshold = -0.5;
        let ruin_count = gpu_results.returns.iter()
            .filter(|&&r| r < ruin_threshold)
            .count();
        let probability_of_ruin = ruin_count as f64 / gpu_results.returns.len() as f64;
        
        // Estimate recovery times (simplified)
        let recovery_time_distribution = self.estimate_recovery_time_distribution(&gpu_results.returns)?;
        
        Ok(MonteCarloStressResults {
            num_simulations: gpu_results.returns.len(),
            percentile_losses,
            expected_shortfall,
            probability_of_ruin,
            recovery_time_distribution,
        })
    }
    
    /// Run CPU Monte Carlo stress tests
    async fn run_cpu_monte_carlo_stress(
        &self,
        portfolio: &Portfolio,
    ) -> RiskResult<MonteCarloStressResults> {
        let num_simulations = self.config.monte_carlo_simulations;
        
        // Calculate portfolio statistics
        let expected_return = portfolio.calculate_expected_return();
        let volatility = portfolio.calculate_volatility();
        
        // Generate random returns
        let mut rng = self.rng.write().await;
        let normal = Normal::new(expected_return / 252.0, volatility / (252.0_f64).sqrt())
            .map_err(|e| RiskError::mathematical(format!("Failed to create normal distribution: {}", e)))?;
        
        let returns: Vec<f64> = (0..num_simulations)
            .map(|_| normal.sample(&mut *rng))
            .collect();
        
        drop(rng); // Release the lock
        
        // Calculate percentile losses
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mut percentile_losses = HashMap::new();
        percentile_losses.insert("99%".to_string(), -sorted_returns[(0.01 * num_simulations as f64) as usize]);
        percentile_losses.insert("95%".to_string(), -sorted_returns[(0.05 * num_simulations as f64) as usize]);
        percentile_losses.insert("90%".to_string(), -sorted_returns[(0.10 * num_simulations as f64) as usize]);
        
        // Calculate expected shortfall (average of worst 5%)
        let tail_index = (0.05 * num_simulations as f64) as usize;
        let expected_shortfall = -sorted_returns[..tail_index].iter().sum::<f64>() / tail_index as f64;
        
        // Calculate probability of ruin
        let ruin_threshold = -0.5;
        let ruin_count = returns.iter().filter(|&&r| r < ruin_threshold).count();
        let probability_of_ruin = ruin_count as f64 / num_simulations as f64;
        
        // Estimate recovery times
        let recovery_time_distribution = self.estimate_recovery_time_distribution(&returns)?;
        
        Ok(MonteCarloStressResults {
            num_simulations,
            percentile_losses,
            expected_shortfall,
            probability_of_ruin,
            recovery_time_distribution,
        })
    }
    
    /// Estimate recovery time distribution
    fn estimate_recovery_time_distribution(
        &self,
        returns: &[f64],
    ) -> RiskResult<Vec<Duration>> {
        let mut recovery_times = Vec::new();
        
        for &return_val in returns {
            if return_val < 0.0 {
                let loss_magnitude = return_val.abs();
                let recovery_days = match loss_magnitude {
                    x if x > 0.5 => 730.0,  // 2 years for >50% loss
                    x if x > 0.3 => 365.0,  // 1 year for >30% loss
                    x if x > 0.2 => 180.0,  // 6 months for >20% loss
                    x if x > 0.1 => 90.0,   // 3 months for >10% loss
                    _ => 30.0,              // 1 month for smaller losses
                };
                
                recovery_times.push(Duration::from_secs((recovery_days * 24.0 * 3600.0) as u64));
            }
        }
        
        Ok(recovery_times)
    }
    
    /// Perform tail risk analysis
    async fn perform_tail_risk_analysis(
        &self,
        portfolio: &Portfolio,
        scenario_results: &[ScenarioResult],
    ) -> RiskResult<TailRiskAnalysis> {
        info!("Performing tail risk analysis");
        
        // Extract extreme values
        let losses: Vec<f64> = scenario_results
            .iter()
            .map(|r| r.portfolio_pnl.abs())
            .collect();
        
        // Fit extreme value distribution
        let extreme_value_stats = self.fit_extreme_value_distribution(&losses)?;
        
        // Identify black swan scenarios
        let black_swan_scenarios = self.identify_black_swan_scenarios(scenario_results)?;
        
        // Calculate tail dependencies
        let tail_dependencies = self.calculate_tail_dependencies(scenario_results)?;
        
        // Apply quantum enhancement if available
        let quantum_tail_enhancement = if self.config.tail_risk_scenarios > 0 {
            Some(self.calculate_quantum_tail_enhancement(portfolio).await?)
        } else {
            None
        };
        
        Ok(TailRiskAnalysis {
            extreme_value_stats,
            black_swan_scenarios,
            tail_dependencies,
            quantum_tail_enhancement,
        })
    }
    
    /// Fit extreme value distribution (simplified GPD)
    fn fit_extreme_value_distribution(&self, losses: &[f64]) -> RiskResult<ExtremeValueStats> {
        if losses.is_empty() {
            return Err(RiskError::insufficient_data("No loss data for extreme value analysis"));
        }
        
        // Simple method-of-moments estimators (in practice would use MLE)
        let mean = losses.iter().sum::<f64>() / losses.len() as f64;
        let variance = losses.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / losses.len() as f64;
        let std_dev = variance.sqrt();
        
        // Simplified parameter estimates
        let shape_parameter = 0.1; // Typical value for financial data
        let scale_parameter = std_dev;
        let location_parameter = mean;
        
        // Calculate return levels
        let mut return_levels = HashMap::new();
        return_levels.insert("10_year".to_string(), mean + 3.0 * std_dev);
        return_levels.insert("100_year".to_string(), mean + 4.0 * std_dev);
        return_levels.insert("1000_year".to_string(), mean + 5.0 * std_dev);
        
        Ok(ExtremeValueStats {
            shape_parameter,
            scale_parameter,
            location_parameter,
            return_levels,
        })
    }
    
    /// Identify black swan scenarios
    fn identify_black_swan_scenarios(
        &self,
        scenario_results: &[ScenarioResult],
    ) -> RiskResult<Vec<BlackSwanScenario>> {
        let mut black_swans = Vec::new();
        
        // Define black swan criteria: low probability but high impact
        for result in scenario_results {
            if result.portfolio_pnl_percent < -30.0 { // High impact threshold
                black_swans.push(BlackSwanScenario {
                    description: format!("Severe loss scenario: {}", result.scenario_name),
                    probability: 0.01, // Assumed low probability
                    potential_loss: result.portfolio_pnl.abs(),
                    market_factors: vec!["Market crash".to_string(), "Liquidity crisis".to_string()],
                });
            }
        }
        
        Ok(black_swans)
    }
    
    /// Calculate tail dependencies
    fn calculate_tail_dependencies(
        &self,
        scenario_results: &[ScenarioResult],
    ) -> RiskResult<HashMap<String, f64>> {
        let mut dependencies = HashMap::new();
        
        // Simplified tail dependency calculation
        // In practice would use more sophisticated copula-based methods
        
        for result in scenario_results {
            let avg_position_impact: f64 = result.position_impacts.values().sum::<f64>() 
                / result.position_impacts.len() as f64;
            
            dependencies.insert(
                result.scenario_name.clone(),
                (avg_position_impact / 100.0).abs().min(1.0),
            );
        }
        
        Ok(dependencies)
    }
    
    /// Calculate quantum tail enhancement
    async fn calculate_quantum_tail_enhancement(
        &self,
        portfolio: &Portfolio,
    ) -> RiskResult<QuantumTailRisk> {
        let quantum_engine = self.quantum_engine.read().await;
        
        // Convert portfolio to quantum format
        let portfolio_data = self.portfolio_to_quantum_data(portfolio).await?;
        
        let uncertainty_quantification = quantum_engine.quantify_uncertainty(
            &portfolio_data.returns,
            &portfolio_data.targets,
        ).await.map_err(|e| RiskError::quantum_uncertainty(e))?;
        
        Ok(QuantumTailRisk {
            quantum_enhancement: uncertainty_quantification.uncertainty,
            tail_dependency_correction: uncertainty_quantification.tail_risk,
            tail_coherence: uncertainty_quantification.confidence_interval.1 - uncertainty_quantification.confidence_interval.0,
        })
    }
    
    /// Convert portfolio to quantum data format
    async fn portfolio_to_quantum_data(
        &self,
        portfolio: &Portfolio,
    ) -> RiskResult<QuantumPortfolioData> {
        let returns = Array2::from_shape_vec(
            (portfolio.returns.len().max(1), 1),
            if portfolio.returns.is_empty() { vec![0.0] } else { portfolio.returns.clone() },
        ).map_err(|e| RiskError::matrix_operation(format!("Failed to create returns matrix: {}", e)))?;
        
        let targets = Array1::from_vec(
            if portfolio.targets.is_empty() { vec![0.0] } else { portfolio.targets.clone() }
        );
        
        Ok(QuantumPortfolioData {
            returns,
            targets,
            positions: portfolio.positions.clone(),
            market_data: portfolio.market_data.clone(),
        })
    }
    
    /// Reset stress tester state
    pub async fn reset(&mut self) -> RiskResult<()> {
        self.scenario_cache.write().await.clear();
        *self.rng.write().await = StdRng::from_entropy();
        self.gpu_engine.write().await.reset().await
            .map_err(|e| RiskError::gpu_acceleration(e))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use tokio_test;

    async fn create_test_scenario() -> StressScenario {
        let mut asset_shocks = HashMap::new();
        asset_shocks.insert("AAPL".to_string(), -0.20);
        asset_shocks.insert("GOOGL".to_string(), -0.15);
        
        let mut volatility_multipliers = HashMap::new();
        volatility_multipliers.insert("AAPL".to_string(), 2.0);
        volatility_multipliers.insert("GOOGL".to_string(), 1.5);
        
        let mut liquidity_impacts = HashMap::new();
        liquidity_impacts.insert("AAPL".to_string(), 0.1);
        liquidity_impacts.insert("GOOGL".to_string(), 0.05);
        
        StressScenario {
            name: "Test Market Crash".to_string(),
            description: "20% market decline with increased volatility".to_string(),
            asset_shocks,
            volatility_multipliers,
            correlation_shifts: Array2::zeros((0, 0)),
            liquidity_impacts,
            probability: 0.05,
        }
    }

    #[tokio::test]
    async fn test_stress_tester_creation() {
        let config = StressConfig::default();
        let quantum_config = quantum_uncertainty::QuantumConfig::default();
        let quantum_engine = Arc::new(RwLock::new(
            QuantumUncertaintyEngine::new(quantum_config).await.unwrap()
        ));
        
        let stress_tester = StressTester::new(config, quantum_engine).await;
        assert!(stress_tester.is_ok());
    }

    #[tokio::test]
    async fn test_single_scenario_stress_test() {
        let config = StressConfig::default();
        let quantum_config = quantum_uncertainty::QuantumConfig::default();
        let quantum_engine = Arc::new(RwLock::new(
            QuantumUncertaintyEngine::new(quantum_config).await.unwrap()
        ));
        
        let stress_tester = StressTester::new(config, quantum_engine).await.unwrap();
        let portfolio = crate::types::Portfolio::default();
        let scenario = create_test_scenario().await;
        
        let result = stress_tester.run_single_scenario_test(&portfolio, &scenario).await;
        
        // Should handle empty portfolio gracefully
        assert!(result.is_ok() || matches!(result.unwrap_err(), RiskError::InsufficientData(_)));
    }

    #[test]
    fn test_extreme_value_distribution_fitting() {
        let config = StressConfig::default();
        let quantum_config = quantum_uncertainty::QuantumConfig::default();
        let quantum_engine = Arc::new(RwLock::new(
            futures::executor::block_on(async {
                QuantumUncertaintyEngine::new(quantum_config).await.unwrap()
            })
        ));
        
        let stress_tester = futures::executor::block_on(async {
            StressTester::new(config, quantum_engine).await.unwrap()
        });
        
        let losses = vec![0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40];
        let result = stress_tester.fit_extreme_value_distribution(&losses);
        
        assert!(result.is_ok());
        let stats = result.unwrap();
        assert!(stats.shape_parameter > 0.0);
        assert!(stats.scale_parameter > 0.0);
        assert!(!stats.return_levels.is_empty());
    }

    #[test]
    fn test_diversification_effectiveness_calculation() {
        let config = StressConfig::default();
        let quantum_config = quantum_uncertainty::QuantumConfig::default();
        let quantum_engine = Arc::new(RwLock::new(
            futures::executor::block_on(async {
                QuantumUncertaintyEngine::new(quantum_config).await.unwrap()
            })
        ));
        
        let stress_tester = futures::executor::block_on(async {
            StressTester::new(config, quantum_engine).await.unwrap()
        });
        
        let mut position_impacts = HashMap::new();
        position_impacts.insert("AAPL".to_string(), -10.0);
        position_impacts.insert("GOOGL".to_string(), -5.0);
        
        let scenario_result = ScenarioResult {
            scenario_name: "Test".to_string(),
            portfolio_pnl: -1000.0,
            portfolio_pnl_percent: -10.0,
            position_impacts,
            stressed_risk_metrics: StressedRiskMetrics {
                stressed_var: 0.05,
                stressed_cvar: 0.07,
                stressed_volatility: 0.20,
                stressed_max_drawdown: 0.15,
                stressed_sharpe: 0.5,
            },
            liquidity_impact: 0.1,
            recovery_time_estimate: Duration::from_secs(30 * 24 * 3600),
        };
        
        let effectiveness = stress_tester.calculate_diversification_effectiveness(&[scenario_result]);
        assert!(effectiveness.is_ok());
        assert!(effectiveness.unwrap() >= 0.0);
        assert!(effectiveness.unwrap() <= 1.0);
    }
}