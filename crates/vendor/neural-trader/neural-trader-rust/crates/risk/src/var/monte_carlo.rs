//! Monte Carlo VaR calculation with GPU acceleration
//!
//! This implementation uses Monte Carlo simulation to estimate VaR and CVaR.
//! It supports GPU acceleration via CUDA when the `gpu` feature is enabled.

use async_trait::async_trait;
use crate::{Result, RiskError};
use crate::types::{Portfolio, Position, VaRResult};
use crate::var::{VaRCalculator, VaRConfigInfo};
use chrono::Utc;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use rand::thread_rng;
use rayon::prelude::*;
use tracing::{debug, info, warn};

/// Configuration for Monte Carlo VaR calculation
#[derive(Debug, Clone)]
pub struct VaRConfig {
    /// Confidence level (e.g., 0.95 for 95% VaR)
    pub confidence_level: f64,
    /// Time horizon in days
    pub time_horizon_days: usize,
    /// Number of Monte Carlo simulations
    pub num_simulations: usize,
    /// Use GPU acceleration if available
    pub use_gpu: bool,
}

impl Default for VaRConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            time_horizon_days: 1,
            num_simulations: 10_000,
            use_gpu: false,
        }
    }
}

/// Monte Carlo VaR calculator
pub struct MonteCarloVaR {
    config: VaRConfig,
}

impl MonteCarloVaR {
    /// Create a new Monte Carlo VaR calculator
    pub fn new(config: VaRConfig) -> Self {
        if config.confidence_level <= 0.0 || config.confidence_level >= 1.0 {
            warn!(
                "Invalid confidence level: {}, using 0.95",
                config.confidence_level
            );
        }
        if config.num_simulations < 1000 {
            warn!(
                "Low number of simulations: {}, recommend at least 10,000",
                config.num_simulations
            );
        }

        Self { config }
    }

    /// Calculate VaR and CVaR using Monte Carlo simulation
    pub async fn calculate(&self, positions: &[Position]) -> Result<VaRResult> {
        if positions.is_empty() {
            return Err(RiskError::InvalidInput(
                "Cannot calculate VaR for empty portfolio".to_string(),
            ));
        }

        info!(
            "Starting Monte Carlo VaR calculation with {} simulations",
            self.config.num_simulations
        );

        // Extract portfolio parameters
        let (returns, volatilities, correlations) = self.extract_portfolio_params(positions)?;

        // Run simulations (GPU or CPU)
        let simulated_returns = if self.config.use_gpu && crate::is_gpu_available() {
            debug!("Using GPU acceleration for Monte Carlo simulation");
            self.simulate_gpu(&returns, &volatilities, &correlations).await?
        } else {
            debug!("Using CPU for Monte Carlo simulation");
            self.simulate_cpu(&returns, &volatilities, &correlations)?
        };

        // Calculate VaR and CVaR
        let result = self.calculate_var_cvar(&simulated_returns)?;

        info!(
            "Monte Carlo VaR calculated: VaR(95%)={:.2}, CVaR(95%)={:.2}",
            result.var_95, result.cvar_95
        );

        Ok(result)
    }

    /// Extract portfolio parameters (returns, volatilities, correlations)
    fn extract_portfolio_params(
        &self,
        positions: &[Position],
    ) -> Result<(Array1<f64>, Array1<f64>, Array2<f64>)> {
        let n = positions.len();

        // For this implementation, we'll use simplified assumptions
        // In production, these would come from historical data
        let mut returns = Array1::zeros(n);
        let mut volatilities = Array1::zeros(n);

        // Calculate position weights
        let total_value: f64 = positions.iter().map(|p| p.exposure().abs()).sum();

        for (i, position) in positions.iter().enumerate() {
            let weight = position.exposure().abs() / total_value;
            returns[i] = 0.0001 * weight; // Assume 1bp expected return
            volatilities[i] = 0.02 * weight; // Assume 2% daily volatility (simplified)
        }

        // Generate correlation matrix (simplified - identity matrix)
        // In production, this would be calculated from historical returns
        let mut correlations = Array2::eye(n);

        // Add some correlation structure (simplified)
        for i in 0..n {
            for j in (i + 1)..n {
                let correlation = 0.5; // Assume moderate correlation
                correlations[[i, j]] = correlation;
                correlations[[j, i]] = correlation;
            }
        }

        Ok((returns, volatilities, correlations))
    }

    /// Run Monte Carlo simulation on CPU
    fn simulate_cpu(
        &self,
        returns: &Array1<f64>,
        volatilities: &Array1<f64>,
        correlations: &Array2<f64>,
    ) -> Result<Vec<f64>> {
        let n_assets = returns.len();
        let n_sims = self.config.num_simulations;
        let n_days = self.config.time_horizon_days;

        // Cholesky decomposition for correlated random variables
        let chol = self.cholesky_decomposition(correlations)?;

        // Parallel simulation
        let simulated_returns: Vec<f64> = (0..n_sims)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();
                let mut cumulative_return = 0.0;

                for _ in 0..n_days {
                    // Generate uncorrelated random shocks
                    let z: Vec<f64> = (0..n_assets)
                        .map(|_| StandardNormal.sample(&mut rng))
                        .collect();

                    // Apply correlation structure
                    let correlated_shocks = self.apply_cholesky(&chol, &z);

                    // Calculate daily portfolio return
                    let daily_return: f64 = (0..n_assets)
                        .map(|i| {
                            returns[i] + volatilities[i] * correlated_shocks[i] / (n_days as f64).sqrt()
                        })
                        .sum();

                    cumulative_return += daily_return;
                }

                cumulative_return
            })
            .collect();

        Ok(simulated_returns)
    }

    /// Run Monte Carlo simulation on GPU (when available)
    #[cfg(feature = "gpu")]
    async fn simulate_gpu(
        &self,
        returns: &Array1<f64>,
        volatilities: &Array1<f64>,
        correlations: &Array2<f64>,
    ) -> Result<Vec<f64>> {
        use candle_core::{Device, Tensor};

        let device = Device::cuda_if_available(0)
            .map_err(|e| RiskError::GpuError(e.to_string()))?;

        debug!("GPU device initialized: {:?}", device);

        // Convert to tensors
        let returns_tensor = Tensor::from_slice(returns.as_slice().unwrap(), returns.len(), &device)
            .map_err(|e| RiskError::GpuError(e.to_string()))?;

        let volatilities_tensor =
            Tensor::from_slice(volatilities.as_slice().unwrap(), volatilities.len(), &device)
                .map_err(|e| RiskError::GpuError(e.to_string()))?;

        // Generate random numbers on GPU
        let n_sims = self.config.num_simulations;
        let n_assets = returns.len();
        let n_days = self.config.time_horizon_days;

        let random_shocks = Tensor::randn(0.0, 1.0, (n_sims, n_days, n_assets), &device)
            .map_err(|e| RiskError::GpuError(e.to_string()))?;

        // Apply correlation and volatility (simplified GPU implementation)
        // In production, implement full Cholesky decomposition on GPU
        let scaled_shocks = random_shocks.broadcast_mul(&volatilities_tensor)
            .map_err(|e| RiskError::GpuError(e.to_string()))?;

        // Sum across assets and days to get portfolio returns
        let portfolio_returns = scaled_shocks
            .sum(2)?  // Sum across assets
            .sum(1)?; // Sum across days

        // Transfer back to CPU
        let returns_vec = portfolio_returns
            .to_vec1::<f64>()
            .map_err(|e| RiskError::GpuError(e.to_string()))?;

        Ok(returns_vec)
    }

    #[cfg(not(feature = "gpu"))]
    async fn simulate_gpu(
        &self,
        returns: &Array1<f64>,
        volatilities: &Array1<f64>,
        correlations: &Array2<f64>,
    ) -> Result<Vec<f64>> {
        warn!("GPU feature not enabled, falling back to CPU simulation");
        self.simulate_cpu(returns, volatilities, correlations)
    }

    /// Cholesky decomposition for correlation matrix
    fn cholesky_decomposition(&self, correlations: &Array2<f64>) -> Result<Array2<f64>> {
        let n = correlations.nrows();
        let mut chol = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += chol[[i, k]] * chol[[j, k]];
                }

                if i == j {
                    let val = correlations[[i, i]] - sum;
                    if val < 0.0 {
                        return Err(RiskError::MatrixError(
                            "Correlation matrix is not positive definite".to_string(),
                        ));
                    }
                    chol[[i, j]] = val.sqrt();
                } else {
                    if chol[[j, j]].abs() < 1e-10 {
                        return Err(RiskError::MatrixError(
                            "Singular matrix in Cholesky decomposition".to_string(),
                        ));
                    }
                    chol[[i, j]] = (correlations[[i, j]] - sum) / chol[[j, j]];
                }
            }
        }

        Ok(chol)
    }

    /// Apply Cholesky factor to uncorrelated shocks
    fn apply_cholesky(&self, chol: &Array2<f64>, z: &[f64]) -> Vec<f64> {
        let n = z.len();
        let mut result = vec![0.0; n];

        for i in 0..n {
            for j in 0..=i {
                result[i] += chol[[i, j]] * z[j];
            }
        }

        result
    }

    /// Calculate VaR and CVaR from simulated returns
    fn calculate_var_cvar(&self, simulated_returns: &[f64]) -> Result<VaRResult> {
        if simulated_returns.is_empty() {
            return Err(RiskError::InvalidInput(
                "No simulated returns to calculate VaR".to_string(),
            ));
        }

        let mut sorted_returns = simulated_returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_returns.len();

        // Calculate VaR at different confidence levels
        let var_95_idx = ((1.0 - 0.95) * n as f64) as usize;
        let var_99_idx = ((1.0 - 0.99) * n as f64) as usize;

        let var_95 = -sorted_returns[var_95_idx]; // Negative because we want loss
        let var_99 = -sorted_returns[var_99_idx];

        // Calculate CVaR (Expected Shortfall)
        let cvar_95 = -sorted_returns[0..=var_95_idx].iter().sum::<f64>()
            / (var_95_idx + 1) as f64;
        let cvar_99 = -sorted_returns[0..=var_99_idx].iter().sum::<f64>()
            / (var_99_idx + 1) as f64;

        // Calculate statistics
        let expected_return = sorted_returns.iter().sum::<f64>() / n as f64;
        let variance = sorted_returns
            .iter()
            .map(|r| (r - expected_return).powi(2))
            .sum::<f64>()
            / n as f64;
        let volatility = variance.sqrt();

        let worst_case = -sorted_returns[0];
        let best_case = -sorted_returns[n - 1];

        Ok(VaRResult {
            var_95,
            var_99,
            cvar_95,
            cvar_99,
            expected_return,
            volatility,
            worst_case,
            best_case,
            num_simulations: n,
            time_horizon_days: self.config.time_horizon_days,
            calculated_at: Utc::now(),
        })
    }
}

#[async_trait]
impl VaRCalculator for MonteCarloVaR {
    async fn calculate_portfolio(&self, portfolio: &Portfolio) -> Result<VaRResult> {
        let positions: Vec<Position> = portfolio.positions.values().cloned().collect();
        self.calculate(&positions).await
    }

    fn method_name(&self) -> &str {
        "Monte Carlo"
    }

    fn config(&self) -> VaRConfigInfo {
        VaRConfigInfo {
            confidence_level: self.config.confidence_level,
            time_horizon_days: self.config.time_horizon_days,
            method: "Monte Carlo".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PositionSide, Symbol};
    use rust_decimal_macros::dec;

    fn create_test_position() -> Position {
        Position {
            symbol: Symbol::new("TEST"),
            quantity: dec!(100),
            avg_entry_price: dec!(100),
            current_price: dec!(100),
            market_value: dec!(10000),
            unrealized_pnl: dec!(0),
            unrealized_pnl_percent: dec!(0),
            side: PositionSide::Long,
            opened_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_monte_carlo_var_basic() {
        let _config = VaRConfig {
            confidence_level: 0.95,
            time_horizon_days: 1,
            num_simulations: 1_000,
            use_gpu: false,
        };

        let calculator = MonteCarloVaR::new(config);
        let positions = vec![create_test_position()];

        let result = calculator.calculate(&positions).await;
        assert!(result.is_ok());

        let var_result = result.unwrap();
        assert!(var_result.var_95 >= 0.0);
        assert!(var_result.cvar_95 >= var_result.var_95);
        assert_eq!(var_result.num_simulations, 1_000);
    }

    #[tokio::test]
    async fn test_empty_positions() {
        let _config = VaRConfig::default();
        let calculator = MonteCarloVaR::new(config);

        let result = calculator.calculate(&[]).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_cholesky_decomposition() {
        let _config = VaRConfig::default();
        let calculator = MonteCarloVaR::new(config);

        // Test with identity matrix
        let identity = Array2::eye(3);
        let chol = calculator.cholesky_decomposition(&identity);
        assert!(chol.is_ok());

        let chol_matrix = chol.unwrap();
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!((chol_matrix[[i, j]] - 1.0).abs() < 1e-10);
                } else if i < j {
                    assert!(chol_matrix[[i, j]].abs() < 1e-10);
                }
            }
        }
    }
}
