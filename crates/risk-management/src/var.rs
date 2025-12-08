//! Value-at-Risk (VaR) and Conditional VaR calculations with quantum uncertainty

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use quantum_uncertainty::{QuantumUncertaintyEngine, UncertaintyQuantification};
use rand::prelude::*;
use rand_distr::{Distribution, Normal, StandardNormal};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT, Uniform as UniformDist};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::config::{VarConfig, VarMethod};
use crate::error::{RiskError, RiskResult};
use crate::types::{Portfolio, Position};

/// VaR calculation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarResult {
    /// VaR values at different confidence levels
    pub var_values: HashMap<String, f64>,
    /// Confidence levels used
    pub confidence_levels: Vec<f64>,
    /// Calculation method used
    pub method: VarMethod,
    /// Quantum uncertainty enhancement
    pub quantum_enhancement: Option<VarQuantumEnhancement>,
    /// Portfolio value
    pub portfolio_value: f64,
    /// Calculation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Calculation duration
    pub calculation_duration: std::time::Duration,
}

/// Conditional VaR (Expected Shortfall) results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CvarResult {
    /// CVaR values at different confidence levels
    pub cvar_values: HashMap<String, f64>,
    /// Confidence levels used
    pub confidence_levels: Vec<f64>,
    /// Calculation method used
    pub method: VarMethod,
    /// Quantum uncertainty enhancement
    pub quantum_enhancement: Option<VarQuantumEnhancement>,
    /// Portfolio value
    pub portfolio_value: f64,
    /// Calculation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Calculation duration
    pub calculation_duration: std::time::Duration,
}

/// Quantum enhancement details for VaR/CVaR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarQuantumEnhancement {
    /// Quantum uncertainty level
    pub quantum_uncertainty: f64,
    /// Quantum advantage factor
    pub quantum_advantage: f64,
    /// Tail risk quantum correction
    pub tail_risk_correction: f64,
    /// Correlation uncertainty
    pub correlation_uncertainty: f64,
    /// Quantum circuit fidelity
    pub quantum_fidelity: f64,
}

/// VaR calculator with quantum uncertainty enhancement
pub struct VarCalculator {
    /// Configuration
    config: VarConfig,
    /// Quantum uncertainty engine
    quantum_engine: Arc<RwLock<QuantumUncertaintyEngine>>,
    /// Historical data cache
    historical_cache: Arc<RwLock<HashMap<String, Array1<f64>>>>,
    /// Covariance matrix cache
    covariance_cache: Arc<RwLock<Option<Array2<f64>>>>,
    /// Random number generator
    rng: Arc<RwLock<StdRng>>,
}

impl VarCalculator {
    /// Create new VaR calculator
    pub async fn new(
        config: VarConfig,
        quantum_engine: Arc<RwLock<QuantumUncertaintyEngine>>,
    ) -> Result<Self> {
        let rng = Arc::new(RwLock::new(StdRng::from_entropy()));
        
        Ok(Self {
            config,
            quantum_engine,
            historical_cache: Arc::new(RwLock::new(HashMap::new())),
            covariance_cache: Arc::new(RwLock::new(None)),
            rng,
        })
    }
    
    /// Calculate Value-at-Risk
    pub async fn calculate_var(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> RiskResult<VarResult> {
        let start_time = Instant::now();
        
        // Validate inputs
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(RiskError::invalid_parameter(
                format!("Confidence level must be between 0 and 1, got {}", confidence_level)
            ));
        }
        
        if portfolio.positions.is_empty() {
            return Err(RiskError::insufficient_data("Portfolio has no positions"));
        }
        
        // Calculate VaR using the configured method
        let var_values = match self.config.method {
            VarMethod::Historical => self.calculate_historical_var(portfolio, confidence_level).await?,
            VarMethod::Parametric => self.calculate_parametric_var(portfolio, confidence_level).await?,
            VarMethod::MonteCarlo => self.calculate_monte_carlo_var(portfolio, confidence_level).await?,
            VarMethod::QuantumMonteCarlo => self.calculate_quantum_monte_carlo_var(portfolio, confidence_level).await?,
            VarMethod::Copula => self.calculate_copula_var(portfolio, confidence_level).await?,
        };
        
        // Apply quantum enhancement if enabled
        let quantum_enhancement = if self.config.enable_quantum {
            Some(self.calculate_quantum_enhancement(portfolio, confidence_level).await?)
        } else {
            None
        };
        
        let calculation_duration = start_time.elapsed();
        
        Ok(VarResult {
            var_values,
            confidence_levels: vec![confidence_level],
            method: self.config.method.clone(),
            quantum_enhancement,
            portfolio_value: portfolio.total_value,
            timestamp: chrono::Utc::now(),
            calculation_duration,
        })
    }
    
    /// Calculate Conditional Value-at-Risk (Expected Shortfall)
    pub async fn calculate_cvar(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> RiskResult<CvarResult> {
        let start_time = Instant::now();
        
        // Validate inputs
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(RiskError::invalid_parameter(
                format!("Confidence level must be between 0 and 1, got {}", confidence_level)
            ));
        }
        
        if portfolio.positions.is_empty() {
            return Err(RiskError::insufficient_data("Portfolio has no positions"));
        }
        
        // Calculate CVaR using the configured method
        let cvar_values = match self.config.method {
            VarMethod::Historical => self.calculate_historical_cvar(portfolio, confidence_level).await?,
            VarMethod::Parametric => self.calculate_parametric_cvar(portfolio, confidence_level).await?,
            VarMethod::MonteCarlo => self.calculate_monte_carlo_cvar(portfolio, confidence_level).await?,
            VarMethod::QuantumMonteCarlo => self.calculate_quantum_monte_carlo_cvar(portfolio, confidence_level).await?,
            VarMethod::Copula => self.calculate_copula_cvar(portfolio, confidence_level).await?,
        };
        
        // Apply quantum enhancement if enabled
        let quantum_enhancement = if self.config.enable_quantum {
            Some(self.calculate_quantum_enhancement(portfolio, confidence_level).await?)
        } else {
            None
        };
        
        let calculation_duration = start_time.elapsed();
        
        Ok(CvarResult {
            cvar_values,
            confidence_levels: vec![confidence_level],
            method: self.config.method.clone(),
            quantum_enhancement,
            portfolio_value: portfolio.total_value,
            timestamp: chrono::Utc::now(),
            calculation_duration,
        })
    }
    
    /// Calculate historical VaR
    async fn calculate_historical_var(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> RiskResult<HashMap<String, f64>> {
        let returns = self.get_portfolio_returns(portfolio).await?;
        
        if returns.len() < self.config.historical_window {
            return Err(RiskError::insufficient_data(
                format!("Need at least {} historical returns, got {}", 
                    self.config.historical_window, returns.len())
            ));
        }
        
        // Sort returns for percentile calculation
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Calculate VaR as the percentile
        let percentile_index = ((1.0 - confidence_level) * sorted_returns.len() as f64) as usize;
        let var_value = -sorted_returns[percentile_index.min(sorted_returns.len() - 1)];
        
        let mut var_values = HashMap::new();
        var_values.insert(format!("{}%", (confidence_level * 100.0) as u32), var_value);
        
        Ok(var_values)
    }
    
    /// Calculate parametric VaR (assuming normal distribution)
    async fn calculate_parametric_var(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> RiskResult<HashMap<String, f64>> {
        let returns = self.get_portfolio_returns(portfolio).await?;
        
        if returns.len() < 30 {
            return Err(RiskError::insufficient_data(
                format!("Need at least 30 returns for parametric VaR, got {}", returns.len())
            ));
        }
        
        // Calculate mean and standard deviation
        let mean = returns.mean().unwrap_or(0.0);
        let std_dev = returns.std(0.0);
        
        // Calculate VaR using normal distribution
        let normal = Normal::new(mean, std_dev).map_err(|e| {
            RiskError::mathematical(format!("Failed to create normal distribution: {}", e))
        })?;
        
        let var_value = -normal.inverse_cdf(1.0 - confidence_level);
        
        let mut var_values = HashMap::new();
        var_values.insert(format!("{}%", (confidence_level * 100.0) as u32), var_value);
        
        Ok(var_values)
    }
    
    /// Calculate Monte Carlo VaR
    async fn calculate_monte_carlo_var(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> RiskResult<HashMap<String, f64>> {
        let returns = self.get_portfolio_returns(portfolio).await?;
        
        if returns.len() < 30 {
            return Err(RiskError::insufficient_data(
                format!("Need at least 30 returns for Monte Carlo VaR, got {}", returns.len())
            ));
        }
        
        // Calculate mean and standard deviation
        let mean = returns.mean().unwrap_or(0.0);
        let std_dev = returns.std(0.0);
        
        // Generate Monte Carlo simulations
        let mut rng = self.rng.write().await;
        let normal = StandardNormal;
        
        let mut simulated_returns = Vec::with_capacity(self.config.monte_carlo_simulations);
        for _ in 0..self.config.monte_carlo_simulations {
            let random_value: f64 = normal.sample(&mut *rng);
            simulated_returns.push(mean + std_dev * random_value);
        }
        
        // Sort simulated returns
        simulated_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Calculate VaR as percentile
        let percentile_index = ((1.0 - confidence_level) * simulated_returns.len() as f64) as usize;
        let var_value = -simulated_returns[percentile_index.min(simulated_returns.len() - 1)];
        
        let mut var_values = HashMap::new();
        var_values.insert(format!("{}%", (confidence_level * 100.0) as u32), var_value);
        
        Ok(var_values)
    }
    
    /// Calculate quantum-enhanced Monte Carlo VaR
    async fn calculate_quantum_monte_carlo_var(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> RiskResult<HashMap<String, f64>> {
        // First calculate classical Monte Carlo VaR
        let classical_var = self.calculate_monte_carlo_var(portfolio, confidence_level).await?;
        
        // Apply quantum uncertainty enhancement
        let quantum_engine = self.quantum_engine.read().await;
        let portfolio_data = self.portfolio_to_quantum_data(portfolio).await?;
        
        let uncertainty_quantification = quantum_engine.quantify_uncertainty(
            &portfolio_data.returns,
            &portfolio_data.targets,
        ).await.map_err(|e| RiskError::quantum_uncertainty(e))?;
        
        // Apply quantum correction to VaR
        let mut quantum_var = HashMap::new();
        for (level, &classical_value) in classical_var.iter() {
            let quantum_correction = uncertainty_quantification.mean_uncertainty() * 
                uncertainty_quantification.quantum_advantage;
            let corrected_value = classical_value * (1.0 + quantum_correction);
            quantum_var.insert(level.clone(), corrected_value);
        }
        
        Ok(quantum_var)
    }
    
    /// Calculate copula-based VaR
    async fn calculate_copula_var(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> RiskResult<HashMap<String, f64>> {
        // Simplified copula implementation - in practice would use proper copula models
        // For now, implement as enhanced Monte Carlo with correlation structure
        
        let covariance_matrix = self.get_covariance_matrix(portfolio).await?;
        let weights = self.get_portfolio_weights(portfolio);
        
        // Calculate portfolio variance using covariance matrix
        let portfolio_variance = Self::calculate_portfolio_variance(&weights, &covariance_matrix)?;
        let portfolio_std = portfolio_variance.sqrt();
        
        // Use Student's t-distribution for fat tails
        let degrees_of_freedom = 5.0; // Typical value for financial data
        let t_dist = StudentsT::new(0.0, portfolio_std, degrees_of_freedom).map_err(|e| {
            RiskError::mathematical(format!("Failed to create t-distribution: {}", e))
        })?;
        
        let var_value = -t_dist.inverse_cdf(1.0 - confidence_level);
        
        let mut var_values = HashMap::new();
        var_values.insert(format!("{}%", (confidence_level * 100.0) as u32), var_value);
        
        Ok(var_values)
    }
    
    /// Calculate historical CVaR
    async fn calculate_historical_cvar(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> RiskResult<HashMap<String, f64>> {
        let returns = self.get_portfolio_returns(portfolio).await?;
        
        if returns.len() < self.config.historical_window {
            return Err(RiskError::insufficient_data(
                format!("Need at least {} historical returns, got {}", 
                    self.config.historical_window, returns.len())
            ));
        }
        
        // Sort returns for percentile calculation
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Calculate VaR threshold
        let percentile_index = ((1.0 - confidence_level) * sorted_returns.len() as f64) as usize;
        let var_threshold = sorted_returns[percentile_index.min(sorted_returns.len() - 1)];
        
        // Calculate CVaR as average of returns beyond VaR threshold
        let tail_returns: Vec<f64> = sorted_returns.iter()
            .take(percentile_index + 1)
            .cloned()
            .collect();
        
        let cvar_value = if tail_returns.is_empty() {
            -var_threshold
        } else {
            -tail_returns.iter().sum::<f64>() / tail_returns.len() as f64
        };
        
        let mut cvar_values = HashMap::new();
        cvar_values.insert(format!("{}%", (confidence_level * 100.0) as u32), cvar_value);
        
        Ok(cvar_values)
    }
    
    /// Calculate parametric CVaR
    async fn calculate_parametric_cvar(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> RiskResult<HashMap<String, f64>> {
        let returns = self.get_portfolio_returns(portfolio).await?;
        
        if returns.len() < 30 {
            return Err(RiskError::insufficient_data(
                format!("Need at least 30 returns for parametric CVaR, got {}", returns.len())
            ));
        }
        
        // Calculate mean and standard deviation
        let mean = returns.mean().unwrap_or(0.0);
        let std_dev = returns.std(0.0);
        
        // Calculate CVaR using normal distribution
        let normal = Normal::new(mean, std_dev).map_err(|e| {
            RiskError::mathematical(format!("Failed to create normal distribution: {}", e))
        })?;
        
        let var_threshold = normal.inverse_cdf(1.0 - confidence_level);
        
        // CVaR formula for normal distribution
        let phi = StandardNormal.pdf(StandardNormal.inverse_cdf(1.0 - confidence_level));
        let cvar_value = -(mean - std_dev * phi / (1.0 - confidence_level));
        
        let mut cvar_values = HashMap::new();
        cvar_values.insert(format!("{}%", (confidence_level * 100.0) as u32), cvar_value);
        
        Ok(cvar_values)
    }
    
    /// Calculate Monte Carlo CVaR
    async fn calculate_monte_carlo_cvar(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> RiskResult<HashMap<String, f64>> {
        let returns = self.get_portfolio_returns(portfolio).await?;
        
        if returns.len() < 30 {
            return Err(RiskError::insufficient_data(
                format!("Need at least 30 returns for Monte Carlo CVaR, got {}", returns.len())
            ));
        }
        
        // Calculate mean and standard deviation
        let mean = returns.mean().unwrap_or(0.0);
        let std_dev = returns.std(0.0);
        
        // Generate Monte Carlo simulations
        let mut rng = self.rng.write().await;
        let normal = StandardNormal;
        
        let mut simulated_returns = Vec::with_capacity(self.config.monte_carlo_simulations);
        for _ in 0..self.config.monte_carlo_simulations {
            let random_value: f64 = normal.sample(&mut *rng);
            simulated_returns.push(mean + std_dev * random_value);
        }
        
        // Sort simulated returns
        simulated_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Calculate VaR threshold
        let percentile_index = ((1.0 - confidence_level) * simulated_returns.len() as f64) as usize;
        
        // Calculate CVaR as average of tail returns
        let tail_returns: Vec<f64> = simulated_returns.iter()
            .take(percentile_index + 1)
            .cloned()
            .collect();
        
        let cvar_value = if tail_returns.is_empty() {
            -simulated_returns[0]
        } else {
            -tail_returns.iter().sum::<f64>() / tail_returns.len() as f64
        };
        
        let mut cvar_values = HashMap::new();
        cvar_values.insert(format!("{}%", (confidence_level * 100.0) as u32), cvar_value);
        
        Ok(cvar_values)
    }
    
    /// Calculate quantum-enhanced Monte Carlo CVaR
    async fn calculate_quantum_monte_carlo_cvar(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> RiskResult<HashMap<String, f64>> {
        // First calculate classical Monte Carlo CVaR
        let classical_cvar = self.calculate_monte_carlo_cvar(portfolio, confidence_level).await?;
        
        // Apply quantum uncertainty enhancement
        let quantum_engine = self.quantum_engine.read().await;
        let portfolio_data = self.portfolio_to_quantum_data(portfolio).await?;
        
        let uncertainty_quantification = quantum_engine.quantify_uncertainty(
            &portfolio_data.returns,
            &portfolio_data.targets,
        ).await.map_err(|e| RiskError::quantum_uncertainty(e))?;
        
        // Apply quantum correction to CVaR
        let mut quantum_cvar = HashMap::new();
        for (level, &classical_value) in classical_cvar.iter() {
            let quantum_correction = uncertainty_quantification.uncertainty_variance() * 
                uncertainty_quantification.quantum_advantage;
            let corrected_value = classical_value * (1.0 + quantum_correction);
            quantum_cvar.insert(level.clone(), corrected_value);
        }
        
        Ok(quantum_cvar)
    }
    
    /// Calculate copula-based CVaR
    async fn calculate_copula_cvar(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> RiskResult<HashMap<String, f64>> {
        // Simplified copula CVaR implementation
        let covariance_matrix = self.get_covariance_matrix(portfolio).await?;
        let weights = self.get_portfolio_weights(portfolio);
        
        // Calculate portfolio variance using covariance matrix
        let portfolio_variance = Self::calculate_portfolio_variance(&weights, &covariance_matrix)?;
        let portfolio_std = portfolio_variance.sqrt();
        
        // Use Student's t-distribution for fat tails
        let degrees_of_freedom = 5.0;
        let t_dist = StudentsT::new(0.0, portfolio_std, degrees_of_freedom).map_err(|e| {
            RiskError::mathematical(format!("Failed to create t-distribution: {}", e))
        })?;
        
        let var_threshold = t_dist.inverse_cdf(1.0 - confidence_level);
        
        // CVaR formula for t-distribution (approximation)
        let tail_expectation = Self::calculate_t_distribution_tail_expectation(
            degrees_of_freedom, 
            portfolio_std, 
            confidence_level
        )?;
        
        let cvar_value = -tail_expectation;
        
        let mut cvar_values = HashMap::new();
        cvar_values.insert(format!("{}%", (confidence_level * 100.0) as u32), cvar_value);
        
        Ok(cvar_values)
    }
    
    /// Calculate quantum enhancement for VaR/CVaR
    async fn calculate_quantum_enhancement(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> RiskResult<VarQuantumEnhancement> {
        let quantum_engine = self.quantum_engine.read().await;
        let portfolio_data = self.portfolio_to_quantum_data(portfolio).await?;
        
        let uncertainty_quantification = quantum_engine.quantify_uncertainty(
            &portfolio_data.returns,
            &portfolio_data.targets,
        ).await.map_err(|e| RiskError::quantum_uncertainty(e))?;
        
        let quantum_fidelity = quantum_engine.validate_circuit_fidelity().await
            .map_err(|e| RiskError::quantum_circuit(e))?;
        
        Ok(VarQuantumEnhancement {
            quantum_uncertainty: uncertainty_quantification.mean_uncertainty(),
            quantum_advantage: uncertainty_quantification.quantum_advantage,
            tail_risk_correction: uncertainty_quantification.conformal_intervals.tail_risk_quantum,
            correlation_uncertainty: uncertainty_quantification.correlations.uncertainty_level,
            quantum_fidelity,
        })
    }
    
    /// Get portfolio returns
    async fn get_portfolio_returns(&self, portfolio: &Portfolio) -> RiskResult<Array1<f64>> {
        if portfolio.returns.is_empty() {
            return Err(RiskError::insufficient_data("No historical returns available"));
        }
        
        Ok(Array1::from_vec(portfolio.returns.clone()))
    }
    
    /// Get portfolio weights
    fn get_portfolio_weights(&self, portfolio: &Portfolio) -> Array1<f64> {
        let weights: Vec<f64> = portfolio.positions.iter()
            .map(|p| p.weight)
            .collect();
        Array1::from_vec(weights)
    }
    
    /// Get covariance matrix
    async fn get_covariance_matrix(&self, portfolio: &Portfolio) -> RiskResult<Array2<f64>> {
        // Check cache first
        if let Some(cached_matrix) = self.covariance_cache.read().await.as_ref() {
            return Ok(cached_matrix.clone());
        }
        
        // Calculate covariance matrix
        let n_assets = portfolio.positions.len();
        let mut covariance_matrix = Array2::zeros((n_assets, n_assets));
        
        // Simplified covariance calculation (would use actual historical data)
        for i in 0..n_assets {
            for j in 0..n_assets {
                let correlation = if i == j { 1.0 } else { 0.3 }; // Simplified
                let vol_i = portfolio.assets.get(i).map(|a| a.volatility).unwrap_or(0.2);
                let vol_j = portfolio.assets.get(j).map(|a| a.volatility).unwrap_or(0.2);
                covariance_matrix[[i, j]] = correlation * vol_i * vol_j;
            }
        }
        
        // Cache the result
        *self.covariance_cache.write().await = Some(covariance_matrix.clone());
        
        Ok(covariance_matrix)
    }
    
    /// Calculate portfolio variance
    fn calculate_portfolio_variance(
        weights: &Array1<f64>,
        covariance_matrix: &Array2<f64>,
    ) -> RiskResult<f64> {
        if weights.len() != covariance_matrix.nrows() || covariance_matrix.nrows() != covariance_matrix.ncols() {
            return Err(RiskError::matrix_operation(
                "Dimension mismatch in portfolio variance calculation"
            ));
        }
        
        // Calculate w^T * Σ * w
        let cov_weights = covariance_matrix.dot(weights);
        let variance = weights.dot(&cov_weights);
        
        if variance < 0.0 {
            return Err(RiskError::numerical_instability(
                "Negative portfolio variance calculated"
            ));
        }
        
        Ok(variance)
    }
    
    /// Calculate tail expectation for t-distribution
    fn calculate_t_distribution_tail_expectation(
        degrees_of_freedom: f64,
        scale: f64,
        confidence_level: f64,
    ) -> RiskResult<f64> {
        // Simplified tail expectation calculation
        let t_dist = StudentsT::new(0.0, scale, degrees_of_freedom).map_err(|e| {
            RiskError::mathematical(format!("Failed to create t-distribution: {}", e))
        })?;
        
        let var_threshold = t_dist.inverse_cdf(1.0 - confidence_level);
        
        // Approximate tail expectation
        let tail_expectation = var_threshold * (1.0 + 1.0 / (degrees_of_freedom - 1.0));
        
        Ok(tail_expectation)
    }
    
    /// Convert portfolio to quantum data format
    async fn portfolio_to_quantum_data(&self, portfolio: &Portfolio) -> RiskResult<crate::quantum::QuantumPortfolioData> {
        let returns = Array2::from_shape_vec(
            (portfolio.returns.len(), 1),
            portfolio.returns.clone(),
        ).map_err(|e| RiskError::matrix_operation(format!("Failed to create returns matrix: {}", e)))?;
        
        let targets = Array1::from_vec(portfolio.targets.clone());
        
        Ok(crate::quantum::QuantumPortfolioData {
            returns,
            targets,
            positions: portfolio.positions.clone(),
            market_data: portfolio.market_data.clone(),
        })
    }
    
    /// Reset calculator state
    pub async fn reset(&mut self) -> RiskResult<()> {
        self.historical_cache.write().await.clear();
        *self.covariance_cache.write().await = None;
        *self.rng.write().await = StdRng::from_entropy();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use tokio_test;
    use uuid::Uuid;

    async fn create_test_portfolio() -> Portfolio {
        let mut portfolio = Portfolio::default();
        portfolio.positions = vec![
            Position {
                symbol: "AAPL".to_string(),
                quantity: 100.0,
                price: 150.0,
                market_value: 15000.0,
                weight: 0.5,
                pnl: 0.0,
                entry_price: 150.0,
                entry_time: chrono::Utc::now(),
            },
            Position {
                symbol: "GOOGL".to_string(),
                quantity: 50.0,
                price: 2500.0,
                market_value: 125000.0,
                weight: 0.5,
                pnl: 0.0,
                entry_price: 2500.0,
                entry_time: chrono::Utc::now(),
            },
        ];
        portfolio.returns = vec![0.01, -0.02, 0.015, -0.01, 0.005]; // Sample returns
        portfolio.targets = vec![0.01, 0.01, 0.01, 0.01, 0.01];
        portfolio
    }

    #[tokio::test]
    async fn test_var_calculator_creation() {
        let config = VarConfig::default();
        let quantum_config = quantum_uncertainty::QuantumConfig::default();
        let quantum_engine = Arc::new(RwLock::new(
            QuantumUncertaintyEngine::new(quantum_config).await.unwrap()
        ));
        
        let var_calculator = VarCalculator::new(config, quantum_engine).await;
        assert!(var_calculator.is_ok());
    }

    #[tokio::test]
    async fn test_historical_var_calculation() {
        let config = VarConfig::default();
        let quantum_config = quantum_uncertainty::QuantumConfig::default();
        let quantum_engine = Arc::new(RwLock::new(
            QuantumUncertaintyEngine::new(quantum_config).await.unwrap()
        ));
        
        let var_calculator = VarCalculator::new(config, quantum_engine).await.unwrap();
        let portfolio = create_test_portfolio().await;
        
        let result = var_calculator.calculate_var(&portfolio, 0.05).await;
        assert!(result.is_ok());
        
        let var_result = result.unwrap();
        assert!(!var_result.var_values.is_empty());
        assert!(var_result.var_values.contains_key("5%"));
    }

    #[tokio::test]
    async fn test_cvar_calculation() {
        let config = VarConfig::default();
        let quantum_config = quantum_uncertainty::QuantumConfig::default();
        let quantum_engine = Arc::new(RwLock::new(
            QuantumUncertaintyEngine::new(quantum_config).await.unwrap()
        ));
        
        let var_calculator = VarCalculator::new(config, quantum_engine).await.unwrap();
        let portfolio = create_test_portfolio().await;
        
        let result = var_calculator.calculate_cvar(&portfolio, 0.05).await;
        assert!(result.is_ok());
        
        let cvar_result = result.unwrap();
        assert!(!cvar_result.cvar_values.is_empty());
        assert!(cvar_result.cvar_values.contains_key("5%"));
    }

    #[tokio::test]
    async fn test_invalid_confidence_level() {
        let config = VarConfig::default();
        let quantum_config = quantum_uncertainty::QuantumConfig::default();
        let quantum_engine = Arc::new(RwLock::new(
            QuantumUncertaintyEngine::new(quantum_config).await.unwrap()
        ));
        
        let var_calculator = VarCalculator::new(config, quantum_engine).await.unwrap();
        let portfolio = create_test_portfolio().await;
        
        let result = var_calculator.calculate_var(&portfolio, 1.5).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), RiskError::InvalidParameter(_)));
    }

    #[tokio::test]
    async fn test_portfolio_variance_calculation() {
        let weights = Array1::from_vec(vec![0.5, 0.5]);
        let covariance_matrix = Array2::from_shape_vec(
            (2, 2),
            vec![0.04, 0.012, 0.012, 0.09],
        ).unwrap();
        
        let variance = VarCalculator::calculate_portfolio_variance(&weights, &covariance_matrix);
        assert!(variance.is_ok());
        
        let expected_variance = 0.5 * 0.5 * 0.04 + 0.5 * 0.5 * 0.09 + 2.0 * 0.5 * 0.5 * 0.012;
        assert_abs_diff_eq!(variance.unwrap(), expected_variance, epsilon = 1e-10);
    }
}