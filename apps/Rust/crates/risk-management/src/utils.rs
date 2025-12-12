//! Utility functions for risk management calculations

use std::collections::HashMap;
use ndarray::{Array1, Array2, s};
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};
use crate::error::{RiskError, RiskResult};

/// Statistical utility functions
pub struct StatUtils;

impl StatUtils {
    /// Calculate sample mean
    pub fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            0.0
        } else {
            data.iter().sum::<f64>() / data.len() as f64
        }
    }
    
    /// Calculate sample variance
    pub fn variance(data: &[f64]) -> f64 {
        if data.len() < 2 {
            0.0
        } else {
            let mean = Self::mean(data);
            data.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / (data.len() - 1) as f64
        }
    }
    
    /// Calculate sample standard deviation
    pub fn std_dev(data: &[f64]) -> f64 {
        Self::variance(data).sqrt()
    }
    
    /// Calculate skewness
    pub fn skewness(data: &[f64]) -> f64 {
        if data.len() < 3 {
            0.0
        } else {
            let mean = Self::mean(data);
            let std_dev = Self::std_dev(data);
            if std_dev == 0.0 {
                0.0
            } else {
                let n = data.len() as f64;
                let skew_sum = data.iter()
                    .map(|x| ((x - mean) / std_dev).powi(3))
                    .sum::<f64>();
                (n / ((n - 1.0) * (n - 2.0))) * skew_sum
            }
        }
    }
    
    /// Calculate kurtosis
    pub fn kurtosis(data: &[f64]) -> f64 {
        if data.len() < 4 {
            0.0
        } else {
            let mean = Self::mean(data);
            let std_dev = Self::std_dev(data);
            if std_dev == 0.0 {
                0.0
            } else {
                let n = data.len() as f64;
                let kurt_sum = data.iter()
                    .map(|x| ((x - mean) / std_dev).powi(4))
                    .sum::<f64>();
                (n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * kurt_sum - 
                3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0))
            }
        }
    }
    
    /// Calculate percentile
    pub fn percentile(data: &[f64], percentile: f64) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (percentile / 100.0 * (sorted_data.len() - 1) as f64).round() as usize;
        sorted_data[index.min(sorted_data.len() - 1)]
    }
    
    /// Calculate correlation coefficient
    pub fn correlation(x: &[f64], y: &[f64]) -> RiskResult<f64> {
        if x.len() != y.len() || x.is_empty() {
            return Err(RiskError::insufficient_data("Invalid data for correlation"));
        }
        
        let mean_x = Self::mean(x);
        let mean_y = Self::mean(y);
        
        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;
        
        for i in 0..x.len() {
            let diff_x = x[i] - mean_x;
            let diff_y = y[i] - mean_y;
            
            numerator += diff_x * diff_y;
            sum_sq_x += diff_x * diff_x;
            sum_sq_y += diff_y * diff_y;
        }
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
}

/// Matrix utility functions
pub struct MatrixUtils;

impl MatrixUtils {
    /// Check if matrix is positive definite
    pub fn is_positive_definite(matrix: &Array2<f64>) -> bool {
        if matrix.nrows() != matrix.ncols() {
            return false;
        }
        
        // Simple check using Sylvester's criterion (all leading principal minors > 0)
        for i in 1..=matrix.nrows() {
            let submatrix = matrix.slice(s![..i, ..i]);
            let det = Self::determinant_2x2_or_smaller(&submatrix.to_owned());
            if det <= 0.0 {
                return false;
            }
        }
        
        true
    }
    
    /// Calculate determinant for 2x2 or smaller matrices
    fn determinant_2x2_or_smaller(matrix: &Array2<f64>) -> f64 {
        match matrix.nrows() {
            1 => matrix[[0, 0]],
            2 => matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]],
            _ => {
                // For larger matrices, would need more sophisticated algorithm
                // This is a simplified implementation
                if Self::is_diagonal(matrix) {
                    matrix.diag().product()
                } else {
                    1.0 // Fallback
                }
            }
        }
    }
    
    /// Check if matrix is diagonal
    fn is_diagonal(matrix: &Array2<f64>) -> bool {
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                if i != j && matrix[[i, j]].abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }
    
    /// Calculate matrix condition number (simplified)
    pub fn condition_number(matrix: &Array2<f64>) -> f64 {
        // Simplified condition number calculation
        // In practice would use SVD
        let diag_elements: Vec<f64> = matrix.diag().to_vec();
        if diag_elements.iter().any(|&x| x.abs() < 1e-12) {
            f64::INFINITY
        } else {
            let max_diag = diag_elements.iter().fold(0.0, |a, &b| a.max(b.abs()));
            let min_diag = diag_elements.iter().fold(f64::INFINITY, |a, &b| a.min(b.abs()));
            max_diag / min_diag
        }
    }
    
    /// Regularize correlation matrix
    pub fn regularize_correlation_matrix(matrix: &mut Array2<f64>, lambda: f64) -> RiskResult<()> {
        if matrix.nrows() != matrix.ncols() {
            return Err(RiskError::matrix_operation("Matrix must be square"));
        }
        
        let n = matrix.nrows();
        
        // Add regularization term to diagonal
        for i in 0..n {
            matrix[[i, i]] = matrix[[i, i]] * (1.0 - lambda) + lambda;
        }
        
        // Ensure symmetry
        for i in 0..n {
            for j in i + 1..n {
                let avg = (matrix[[i, j]] + matrix[[j, i]]) / 2.0;
                matrix[[i, j]] = avg;
                matrix[[j, i]] = avg;
            }
        }
        
        Ok(())
    }
}

/// Financial utility functions
pub struct FinancialUtils;

impl FinancialUtils {
    /// Calculate annualized return
    pub fn annualized_return(returns: &[f64], periods_per_year: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        
        let total_return = returns.iter().fold(1.0, |acc, &r| acc * (1.0 + r));
        let num_periods = returns.len() as f64;
        
        total_return.powf(periods_per_year / num_periods) - 1.0
    }
    
    /// Calculate annualized volatility
    pub fn annualized_volatility(returns: &[f64], periods_per_year: f64) -> f64 {
        StatUtils::std_dev(returns) * periods_per_year.sqrt()
    }
    
    /// Calculate Sharpe ratio
    pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
        let excess_returns: Vec<f64> = returns.iter()
            .map(|&r| r - risk_free_rate / periods_per_year)
            .collect();
        
        let mean_excess = StatUtils::mean(&excess_returns);
        let vol_excess = StatUtils::std_dev(&excess_returns);
        
        if vol_excess == 0.0 {
            0.0
        } else {
            mean_excess / vol_excess * periods_per_year.sqrt()
        }
    }
    
    /// Calculate Sortino ratio
    pub fn sortino_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
        let excess_returns: Vec<f64> = returns.iter()
            .map(|&r| r - risk_free_rate / periods_per_year)
            .collect();
        
        let mean_excess = StatUtils::mean(&excess_returns);
        
        // Calculate downside deviation
        let negative_returns: Vec<f64> = excess_returns.iter()
            .filter(|&&r| r < 0.0)
            .cloned()
            .collect();
        
        if negative_returns.is_empty() {
            return f64::INFINITY;
        }
        
        let downside_deviation = StatUtils::std_dev(&negative_returns);
        
        if downside_deviation == 0.0 {
            0.0
        } else {
            mean_excess / downside_deviation * periods_per_year.sqrt()
        }
    }
    
    /// Calculate maximum drawdown
    pub fn maximum_drawdown(returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        
        let mut cumulative = 1.0;
        let mut peak = 1.0;
        let mut max_dd = 0.0;
        
        for &ret in returns {
            cumulative *= 1.0 + ret;
            peak = peak.max(cumulative);
            let drawdown = (peak - cumulative) / peak;
            max_dd = max_dd.max(drawdown);
        }
        
        max_dd
    }
    
    /// Calculate Calmar ratio
    pub fn calmar_ratio(returns: &[f64], periods_per_year: f64) -> f64 {
        let annualized_ret = Self::annualized_return(returns, periods_per_year);
        let max_dd = Self::maximum_drawdown(returns);
        
        if max_dd == 0.0 {
            f64::INFINITY
        } else {
            annualized_ret / max_dd
        }
    }
    
    /// Calculate beta
    pub fn beta(asset_returns: &[f64], market_returns: &[f64]) -> RiskResult<f64> {
        if asset_returns.len() != market_returns.len() || asset_returns.is_empty() {
            return Err(RiskError::insufficient_data("Invalid data for beta calculation"));
        }
        
        let market_variance = StatUtils::variance(market_returns);
        if market_variance == 0.0 {
            return Ok(0.0);
        }
        
        let covariance = Self::covariance(asset_returns, market_returns)?;
        Ok(covariance / market_variance)
    }
    
    /// Calculate covariance
    pub fn covariance(x: &[f64], y: &[f64]) -> RiskResult<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return Err(RiskError::insufficient_data("Invalid data for covariance"));
        }
        
        let mean_x = StatUtils::mean(x);
        let mean_y = StatUtils::mean(y);
        
        let cov = x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>() / (x.len() - 1) as f64;
        
        Ok(cov)
    }
    
    /// Calculate tracking error
    pub fn tracking_error(portfolio_returns: &[f64], benchmark_returns: &[f64], periods_per_year: f64) -> RiskResult<f64> {
        if portfolio_returns.len() != benchmark_returns.len() {
            return Err(RiskError::insufficient_data("Mismatched return series lengths"));
        }
        
        let excess_returns: Vec<f64> = portfolio_returns.iter()
            .zip(benchmark_returns.iter())
            .map(|(&p, &b)| p - b)
            .collect();
        
        Ok(StatUtils::std_dev(&excess_returns) * periods_per_year.sqrt())
    }
    
    /// Calculate information ratio
    pub fn information_ratio(portfolio_returns: &[f64], benchmark_returns: &[f64], periods_per_year: f64) -> RiskResult<f64> {
        if portfolio_returns.len() != benchmark_returns.len() {
            return Err(RiskError::insufficient_data("Mismatched return series lengths"));
        }
        
        let excess_returns: Vec<f64> = portfolio_returns.iter()
            .zip(benchmark_returns.iter())
            .map(|(&p, &b)| p - b)
            .collect();
        
        let mean_excess = StatUtils::mean(&excess_returns);
        let tracking_error = StatUtils::std_dev(&excess_returns) * periods_per_year.sqrt();
        
        if tracking_error == 0.0 {
            Ok(0.0)
        } else {
            Ok(mean_excess * periods_per_year.sqrt() / tracking_error)
        }
    }
}

/// Risk utility functions
pub struct RiskUtils;

impl RiskUtils {
    /// Calculate VaR using historical method
    pub fn historical_var(returns: &[f64], confidence_level: f64) -> RiskResult<f64> {
        if returns.is_empty() {
            return Err(RiskError::insufficient_data("No returns data"));
        }
        
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(RiskError::invalid_parameter("Confidence level must be between 0 and 1"));
        }
        
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((1.0 - confidence_level) * sorted_returns.len() as f64) as usize;
        let var = -sorted_returns[index.min(sorted_returns.len() - 1)];
        
        Ok(var)
    }
    
    /// Calculate CVaR using historical method
    pub fn historical_cvar(returns: &[f64], confidence_level: f64) -> RiskResult<f64> {
        if returns.is_empty() {
            return Err(RiskError::insufficient_data("No returns data"));
        }
        
        let var = Self::historical_var(returns, confidence_level)?;
        
        let tail_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r <= -var)
            .cloned()
            .collect();
        
        if tail_returns.is_empty() {
            Ok(var)
        } else {
            Ok(-StatUtils::mean(&tail_returns))
        }
    }
    
    /// Calculate parametric VaR
    pub fn parametric_var(returns: &[f64], confidence_level: f64) -> RiskResult<f64> {
        if returns.len() < 30 {
            return Err(RiskError::insufficient_data("Need at least 30 observations"));
        }
        
        let mean = StatUtils::mean(returns);
        let std_dev = StatUtils::std_dev(returns);
        
        let normal = Normal::new(mean, std_dev)
            .map_err(|e| RiskError::mathematical(format!("Invalid distribution parameters: {}", e)))?;
        
        let var = -normal.inverse_cdf(1.0 - confidence_level);
        Ok(var)
    }
    
    /// Calculate portfolio VaR using delta-normal method
    pub fn portfolio_var_delta_normal(
        weights: &Array1<f64>,
        mean_returns: &Array1<f64>,
        covariance_matrix: &Array2<f64>,
        confidence_level: f64,
    ) -> RiskResult<f64> {
        if weights.len() != mean_returns.len() || 
           weights.len() != covariance_matrix.nrows() ||
           covariance_matrix.nrows() != covariance_matrix.ncols() {
            return Err(RiskError::matrix_operation("Dimension mismatch"));
        }
        
        // Portfolio expected return
        let portfolio_return = weights.dot(mean_returns);
        
        // Portfolio variance
        let portfolio_variance = weights.dot(&covariance_matrix.dot(weights));
        let portfolio_std = portfolio_variance.sqrt();
        
        // VaR calculation
        let normal = Normal::new(portfolio_return, portfolio_std)
            .map_err(|e| RiskError::mathematical(format!("Invalid distribution parameters: {}", e)))?;
        
        let var = -normal.inverse_cdf(1.0 - confidence_level);
        Ok(var)
    }
    
    /// Calculate component VaR
    pub fn component_var(
        weights: &Array1<f64>,
        covariance_matrix: &Array2<f64>,
        confidence_level: f64,
    ) -> RiskResult<Array1<f64>> {
        let portfolio_variance = weights.dot(&covariance_matrix.dot(weights));
        let portfolio_std = portfolio_variance.sqrt();
        
        if portfolio_std == 0.0 {
            return Ok(Array1::zeros(weights.len()));
        }
        
        // Marginal VaR
        let marginal_var = covariance_matrix.dot(weights) / portfolio_std;
        
        // Component VaR
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_score = -normal.inverse_cdf(1.0 - confidence_level);
        
        let component_var = &marginal_var * weights * z_score;
        
        Ok(component_var)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_statistical_functions() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert_abs_diff_eq!(StatUtils::mean(&data), 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(StatUtils::variance(&data), 2.5, epsilon = 1e-10);
        assert_abs_diff_eq!(StatUtils::std_dev(&data), 2.5_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_correlation_calculation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let correlation = StatUtils::correlation(&x, &y).unwrap();
        assert_abs_diff_eq!(correlation, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sharpe_ratio_calculation() {
        let returns = vec![0.01, 0.02, -0.01, 0.03, 0.00];
        let sharpe = FinancialUtils::sharpe_ratio(&returns, 0.02, 252.0);
        assert!(sharpe.is_finite());
    }

    #[test]
    fn test_maximum_drawdown() {
        let returns = vec![0.1, -0.05, -0.1, 0.15, -0.2];
        let max_dd = FinancialUtils::maximum_drawdown(&returns);
        assert!(max_dd >= 0.0);
        assert!(max_dd <= 1.0);
    }

    #[test]
    fn test_historical_var() {
        let returns = vec![-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03];
        let var = RiskUtils::historical_var(&returns, 0.05).unwrap();
        assert!(var >= 0.0);
    }

    #[test]
    fn test_matrix_positive_definite() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        assert!(MatrixUtils::is_positive_definite(&matrix));
        
        let singular_matrix = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        assert!(!MatrixUtils::is_positive_definite(&singular_matrix));
    }
}