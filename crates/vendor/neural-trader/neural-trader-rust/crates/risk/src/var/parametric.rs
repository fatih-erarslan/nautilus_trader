//! Parametric VaR calculation using variance-covariance method

use async_trait::async_trait;
use crate::{Result, error::RiskError};
use crate::types::{Portfolio, VaRResult};
use crate::var::{VaRCalculator, VaRConfigInfo};
use chrono::Utc;
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, Normal};
use tracing::{debug, info};

/// Parametric VaR calculator using variance-covariance method
pub struct ParametricVaR {
    confidence_level: f64,
    time_horizon_days: usize,
}

impl ParametricVaR {
    /// Create a new Parametric VaR calculator
    pub fn new(confidence_level: f64, time_horizon_days: usize) -> Self {
        Self {
            confidence_level,
            time_horizon_days,
        }
    }

    /// Calculate VaR and CVaR using parametric method
    pub async fn calculate(
        &self,
        expected_returns: &[f64],
        covariance_matrix: &[Vec<f64>],
        weights: &[f64],
    ) -> Result<VaRResult> {
        if expected_returns.is_empty() || weights.is_empty() {
            return Err(RiskError::InvalidInput(
                "Empty returns or weights vector".to_string(),
            ));
        }

        if expected_returns.len() != weights.len() {
            return Err(RiskError::InvalidInput(
                "Returns and weights vectors must have same length".to_string(),
            ));
        }

        info!(
            "Calculating Parametric VaR for {} assets",
            expected_returns.len()
        );

        let n = expected_returns.len();

        // Convert to nalgebra vectors/matrices
        let returns_vec = DVector::from_vec(expected_returns.to_vec());
        let weights_vec = DVector::from_vec(weights.to_vec());

        // Flatten covariance matrix
        let cov_data: Vec<f64> = covariance_matrix.iter().flat_map(|row| row.iter().copied()).collect();
        let cov_matrix = DMatrix::from_row_slice(n, n, &cov_data);

        // Calculate portfolio expected return
        let portfolio_return = weights_vec.dot(&returns_vec);

        // Calculate portfolio variance: w^T * Σ * w
        let weighted_cov = &cov_matrix * &weights_vec;
        let portfolio_variance = weights_vec.dot(&weighted_cov);

        if portfolio_variance < 0.0 {
            return Err(RiskError::NumericalError(
                "Negative portfolio variance calculated".to_string(),
            ));
        }

        let portfolio_std = portfolio_variance.sqrt();

        // Scale for time horizon
        let horizon_factor = (self.time_horizon_days as f64).sqrt();
        let horizon_return = portfolio_return * self.time_horizon_days as f64;
        let horizon_std = portfolio_std * horizon_factor;

        // Calculate VaR using normal distribution assumption
        let normal = Normal::new(0.0, 1.0)
            .map_err(|e| RiskError::NumericalError(e.to_string()))?;

        let z_95 = normal.inverse_cdf(1.0 - 0.95);
        let z_99 = normal.inverse_cdf(1.0 - 0.99);

        let var_95 = -(horizon_return + z_95 * horizon_std);
        let var_99 = -(horizon_return + z_99 * horizon_std);

        // Calculate CVaR (for normal distribution)
        // CVaR = μ + σ * φ(Φ^-1(α)) / α
        // where φ is PDF, Φ is CDF, α is confidence level
        let pdf_95 = (-z_95.powi(2) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let pdf_99 = (-z_99.powi(2) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();

        let cvar_95 = -(horizon_return + horizon_std * pdf_95 / (1.0 - 0.95));
        let cvar_99 = -(horizon_return + horizon_std * pdf_99 / (1.0 - 0.99));

        // Calculate best/worst case (±3 sigma)
        let worst_case = -(horizon_return - 3.0 * horizon_std);
        let best_case = -(horizon_return + 3.0 * horizon_std);

        debug!(
            "Parametric VaR: VaR(95%)={:.4}, CVaR(95%)={:.4}, σ={:.4}",
            var_95, cvar_95, portfolio_std
        );

        Ok(VaRResult {
            var_95,
            var_99,
            cvar_95,
            cvar_99,
            expected_return: horizon_return,
            volatility: horizon_std,
            worst_case,
            best_case,
            num_simulations: 0, // Not applicable for parametric method
            time_horizon_days: self.time_horizon_days,
            calculated_at: Utc::now(),
        })
    }
}

#[async_trait]
impl VaRCalculator for ParametricVaR {
    async fn calculate_portfolio(&self, portfolio: &Portfolio) -> Result<VaRResult> {
        // In production, calculate actual returns, covariance, and weights
        // from portfolio positions. For now, use simplified placeholder.

        let n = portfolio.positions.len().max(1);
        let returns = vec![0.0001; n]; // Placeholder expected returns
        let weights: Vec<f64> = vec![1.0 / n as f64; n]; // Equal weights

        // Identity covariance matrix (simplified)
        let mut covariance = vec![vec![0.0; n]; n];
        for i in 0..n {
            covariance[i][i] = 0.0004; // 2% daily volatility squared
        }

        self.calculate(&returns, &covariance, &weights).await
    }

    fn method_name(&self) -> &str {
        "Parametric VaR"
    }

    fn config(&self) -> VaRConfigInfo {
        VaRConfigInfo {
            confidence_level: self.confidence_level,
            time_horizon_days: self.time_horizon_days,
            method: "Parametric".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parametric_var_basic() {
        let returns = vec![0.001, 0.002, 0.0015];
        let weights = vec![0.4, 0.3, 0.3];

        // Simple covariance matrix
        let covariance = vec![
            vec![0.0004, 0.0001, 0.0001],
            vec![0.0001, 0.0004, 0.0001],
            vec![0.0001, 0.0001, 0.0004],
        ];

        let calculator = ParametricVaR::new(0.95, 1);
        let result = calculator.calculate(&returns, &covariance, &weights).await;

        assert!(result.is_ok());
        let var_result = result.unwrap();
        assert!(var_result.var_95 >= 0.0);
        assert!(var_result.cvar_95 >= var_result.var_95);
    }

    #[tokio::test]
    async fn test_empty_inputs() {
        let calculator = ParametricVaR::new(0.95, 1);
        let result = calculator.calculate(&[], &[], &[]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mismatched_dimensions() {
        let returns = vec![0.001, 0.002];
        let weights = vec![0.5];
        let covariance = vec![vec![0.0004]];

        let calculator = ParametricVaR::new(0.95, 1);
        let result = calculator.calculate(&returns, &covariance, &weights).await;
        assert!(result.is_err());
    }
}
