//! SIMD-optimized operations for risk calculations
//!
//! This module provides vectorized implementations of risk metrics and matrix operations
//! with automatic fallback to scalar implementations when SIMD is not available.

#[cfg(feature = "simd")]
pub mod risk_ops;
#[cfg(feature = "simd")]
pub mod matrix_ops;

#[cfg(not(feature = "simd"))]
pub mod risk_ops {
    //! Scalar fallback implementations for risk operations

    pub fn simd_var_historical(returns: &[f64], confidence: f64) -> f64 {
        crate::var::var_historical(returns, confidence)
    }

    pub fn simd_cvar_historical(returns: &[f64], confidence: f64) -> f64 {
        crate::cvar::cvar_historical(returns, confidence)
    }

    pub fn simd_portfolio_variance(weights: &[f64], covariance: &[f64]) -> f64 {
        let n = weights.len();
        let mut variance = 0.0;

        for i in 0..n {
            for j in 0..n {
                variance += weights[i] * weights[j] * covariance[i * n + j];
            }
        }

        variance
    }

    pub fn simd_drawdown_series(equity_curve: &[f64]) -> Vec<f64> {
        let mut drawdowns = Vec::with_capacity(equity_curve.len());
        let mut peak = f64::NEG_INFINITY;

        for &value in equity_curve {
            peak = peak.max(value);
            let drawdown = if peak > 0.0 {
                (peak - value) / peak
            } else {
                0.0
            };
            drawdowns.push(drawdown);
        }

        drawdowns
    }

    pub fn simd_rolling_volatility(returns: &[f64], window: usize) -> Vec<f64> {
        let mut volatilities = Vec::with_capacity(returns.len().saturating_sub(window - 1));

        for i in 0..=returns.len().saturating_sub(window) {
            let window_slice = &returns[i..i + window];
            let mean = window_slice.iter().sum::<f64>() / window as f64;
            let variance = window_slice.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / window as f64;
            volatilities.push(variance.sqrt());
        }

        volatilities
    }
}

#[cfg(not(feature = "simd"))]
pub mod matrix_ops {
    //! Scalar fallback implementations for matrix operations

    pub fn simd_covariance_matrix(returns: &[&[f64]]) -> Vec<Vec<f64>> {
        let n_assets = returns.len();
        if n_assets == 0 {
            return Vec::new();
        }

        let n_periods = returns[0].len();
        let mut cov_matrix = vec![vec![0.0; n_assets]; n_assets];

        // Calculate means
        let means: Vec<f64> = returns.iter()
            .map(|r| r.iter().sum::<f64>() / n_periods as f64)
            .collect();

        // Calculate covariance
        for i in 0..n_assets {
            for j in i..n_assets {
                let mut cov = 0.0;
                for k in 0..n_periods {
                    cov += (returns[i][k] - means[i]) * (returns[j][k] - means[j]);
                }
                cov /= (n_periods - 1) as f64;
                cov_matrix[i][j] = cov;
                cov_matrix[j][i] = cov;
            }
        }

        cov_matrix
    }

    pub fn simd_correlation_matrix(returns: &[&[f64]]) -> Vec<Vec<f64>> {
        let cov_matrix = simd_covariance_matrix(returns);
        let n = cov_matrix.len();
        let mut corr_matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                let std_i = cov_matrix[i][i].sqrt();
                let std_j = cov_matrix[j][j].sqrt();
                if std_i > 0.0 && std_j > 0.0 {
                    corr_matrix[i][j] = cov_matrix[i][j] / (std_i * std_j);
                } else {
                    corr_matrix[i][j] = if i == j { 1.0 } else { 0.0 };
                }
            }
        }

        corr_matrix
    }

    pub fn simd_matrix_multiply(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut result = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a[i * n + k] * b[k * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        result
    }

    pub fn simd_cholesky_decomposition(matrix: &[f64], n: usize) -> Option<Vec<f64>> {
        let mut l = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;

                if j == i {
                    for k in 0..j {
                        sum += l[j * n + k].powi(2);
                    }
                    let diag = matrix[j * n + j] - sum;
                    if diag <= 0.0 {
                        return None; // Matrix not positive definite
                    }
                    l[j * n + j] = diag.sqrt();
                } else {
                    for k in 0..j {
                        sum += l[i * n + k] * l[j * n + k];
                    }
                    if l[j * n + j] == 0.0 {
                        return None;
                    }
                    l[i * n + j] = (matrix[i * n + j] - sum) / l[j * n + j];
                }
            }
        }

        Some(l)
    }
}

#[cfg(feature = "simd")]
pub use risk_ops::*;
#[cfg(feature = "simd")]
pub use matrix_ops::*;

#[cfg(not(feature = "simd"))]
pub use risk_ops::*;
#[cfg(not(feature = "simd"))]
pub use matrix_ops::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_computation() {
        let returns = vec![-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05];
        let var = simd_var_historical(&returns, 0.95);
        assert!(var > 0.0);
    }

    #[test]
    fn test_portfolio_variance() {
        let weights = vec![0.5, 0.5];
        let covariance = vec![
            0.04, 0.02,
            0.02, 0.09,
        ];
        let variance = simd_portfolio_variance(&weights, &covariance);
        assert!(variance > 0.0);
    }

    #[test]
    fn test_drawdown_series() {
        let equity = vec![100.0, 110.0, 105.0, 120.0, 100.0, 130.0];
        let drawdowns = simd_drawdown_series(&equity);
        assert_eq!(drawdowns.len(), equity.len());
        assert!(drawdowns.iter().all(|&d| d >= 0.0));
    }

    #[test]
    fn test_matrix_multiply() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = simd_matrix_multiply(&a, &b, 2);
        assert_eq!(result.len(), 4);
    }
}
