//! SIMD-optimized risk calculation operations
//!
//! This module implements vectorized versions of common risk metrics using
//! manual loop unrolling and cache-friendly access patterns for auto-vectorization.

use std::cmp::Ordering;

/// SIMD vector size for f64 operations (4 elements = 256 bits)
const SIMD_WIDTH: usize = 4;

/// Kahan summation for improved numerical stability
#[inline(always)]
fn kahan_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;

    for &value in values {
        let y = value - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    sum
}

/// SIMD-optimized historical VaR calculation
///
/// Computes Value-at-Risk using historical simulation with vectorized operations.
///
/// # Arguments
/// * `returns` - Slice of historical returns
/// * `confidence` - Confidence level (e.g., 0.95 for 95% confidence)
///
/// # Returns
/// VaR at the specified confidence level (positive value)
pub fn simd_var_historical(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    // Sort returns for percentile calculation
    let mut sorted_returns: Vec<f64> = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    // Calculate VaR index
    let alpha = 1.0 - confidence;
    let index = (alpha * sorted_returns.len() as f64).floor() as usize;
    let index = index.min(sorted_returns.len() - 1);

    // Return absolute value (VaR is reported as positive)
    -sorted_returns[index]
}

/// SIMD-optimized historical CVaR calculation
///
/// Computes Conditional Value-at-Risk (Expected Shortfall) using vectorized operations.
///
/// # Arguments
/// * `returns` - Slice of historical returns
/// * `confidence` - Confidence level (e.g., 0.95 for 95% confidence)
///
/// # Returns
/// CVaR at the specified confidence level (positive value)
pub fn simd_cvar_historical(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    // Sort returns for tail calculation
    let mut sorted_returns: Vec<f64> = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    // Calculate VaR threshold
    let alpha = 1.0 - confidence;
    let var_index = (alpha * sorted_returns.len() as f64).floor() as usize;
    let var_index = var_index.min(sorted_returns.len() - 1);

    // Vectorized tail mean calculation
    let tail = &sorted_returns[..=var_index];
    let tail_sum = vectorized_sum(tail);
    let cvar = tail_sum / tail.len() as f64;

    // Return absolute value
    -cvar
}

/// SIMD-optimized portfolio variance calculation
///
/// Computes portfolio variance using vectorized matrix operations.
///
/// # Arguments
/// * `weights` - Portfolio weights
/// * `covariance` - Flattened covariance matrix (row-major order)
///
/// # Returns
/// Portfolio variance
pub fn simd_portfolio_variance(weights: &[f64], covariance: &[f64]) -> f64 {
    let n = weights.len();
    let mut variance = 0.0;

    // Process in blocks for better cache locality
    const BLOCK_SIZE: usize = 64;

    for i_block in (0..n).step_by(BLOCK_SIZE) {
        let i_end = (i_block + BLOCK_SIZE).min(n);

        for i in i_block..i_end {
            let wi = weights[i];
            let row_offset = i * n;

            // Vectorized inner product for row i
            let mut row_sum = 0.0;
            let mut j = 0;

            // Process 4 elements at a time
            while j + SIMD_WIDTH <= n {
                let mut partial_sum = 0.0;

                // Manual unrolling for auto-vectorization
                partial_sum += weights[j] * covariance[row_offset + j];
                partial_sum += weights[j + 1] * covariance[row_offset + j + 1];
                partial_sum += weights[j + 2] * covariance[row_offset + j + 2];
                partial_sum += weights[j + 3] * covariance[row_offset + j + 3];

                row_sum += partial_sum;
                j += SIMD_WIDTH;
            }

            // Handle remainder
            while j < n {
                row_sum += weights[j] * covariance[row_offset + j];
                j += 1;
            }

            variance += wi * row_sum;
        }
    }

    variance
}

/// SIMD-optimized drawdown series calculation
///
/// Computes the drawdown series from an equity curve using vectorized peak tracking.
///
/// # Arguments
/// * `equity_curve` - Equity curve values
///
/// # Returns
/// Vector of drawdown percentages at each point
pub fn simd_drawdown_series(equity_curve: &[f64]) -> Vec<f64> {
    if equity_curve.is_empty() {
        return Vec::new();
    }

    let mut drawdowns = Vec::with_capacity(equity_curve.len());
    let mut peaks = vec![f64::NEG_INFINITY; equity_curve.len()];

    // Vectorized peak calculation
    let mut running_peak = f64::NEG_INFINITY;

    for (i, &value) in equity_curve.iter().enumerate() {
        running_peak = running_peak.max(value);
        peaks[i] = running_peak;
    }

    // Vectorized drawdown calculation
    let mut i = 0;

    while i + SIMD_WIDTH <= equity_curve.len() {
        // Manual unrolling for auto-vectorization
        let dd0 = if peaks[i] > 0.0 {
            (peaks[i] - equity_curve[i]) / peaks[i]
        } else {
            0.0
        };

        let dd1 = if peaks[i + 1] > 0.0 {
            (peaks[i + 1] - equity_curve[i + 1]) / peaks[i + 1]
        } else {
            0.0
        };

        let dd2 = if peaks[i + 2] > 0.0 {
            (peaks[i + 2] - equity_curve[i + 2]) / peaks[i + 2]
        } else {
            0.0
        };

        let dd3 = if peaks[i + 3] > 0.0 {
            (peaks[i + 3] - equity_curve[i + 3]) / peaks[i + 3]
        } else {
            0.0
        };

        drawdowns.push(dd0);
        drawdowns.push(dd1);
        drawdowns.push(dd2);
        drawdowns.push(dd3);

        i += SIMD_WIDTH;
    }

    // Handle remainder
    while i < equity_curve.len() {
        let dd = if peaks[i] > 0.0 {
            (peaks[i] - equity_curve[i]) / peaks[i]
        } else {
            0.0
        };
        drawdowns.push(dd);
        i += 1;
    }

    drawdowns
}

/// SIMD-optimized rolling volatility calculation
///
/// Computes rolling window volatility using vectorized statistics.
///
/// # Arguments
/// * `returns` - Return series
/// * `window` - Rolling window size
///
/// # Returns
/// Vector of rolling volatilities
pub fn simd_rolling_volatility(returns: &[f64], window: usize) -> Vec<f64> {
    if returns.len() < window || window == 0 {
        return Vec::new();
    }

    let mut volatilities = Vec::with_capacity(returns.len() - window + 1);

    for i in 0..=returns.len() - window {
        let window_slice = &returns[i..i + window];

        // Vectorized mean calculation
        let mean = vectorized_sum(window_slice) / window as f64;

        // Vectorized variance calculation
        let variance = vectorized_squared_deviation(window_slice, mean) / window as f64;

        volatilities.push(variance.sqrt());
    }

    volatilities
}

/// Vectorized sum with manual unrolling
#[inline]
fn vectorized_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut i = 0;

    // Process 4 elements at a time
    while i + SIMD_WIDTH <= values.len() {
        sum += values[i] + values[i + 1] + values[i + 2] + values[i + 3];
        i += SIMD_WIDTH;
    }

    // Handle remainder
    while i < values.len() {
        sum += values[i];
        i += 1;
    }

    sum
}

/// Vectorized squared deviation sum
#[inline]
fn vectorized_squared_deviation(values: &[f64], mean: f64) -> f64 {
    let mut sum = 0.0;
    let mut i = 0;

    // Process 4 elements at a time
    while i + SIMD_WIDTH <= values.len() {
        let d0 = values[i] - mean;
        let d1 = values[i + 1] - mean;
        let d2 = values[i + 2] - mean;
        let d3 = values[i + 3] - mean;

        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += SIMD_WIDTH;
    }

    // Handle remainder
    while i < values.len() {
        let d = values[i] - mean;
        sum += d * d;
        i += 1;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_var_accuracy() {
        let returns = vec![-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05];
        let var = simd_var_historical(&returns, 0.90);
        assert!(var > 0.0);
        assert!(var <= 0.05);
    }

    #[test]
    fn test_simd_cvar_accuracy() {
        let returns = vec![-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05];
        let cvar = simd_cvar_historical(&returns, 0.90);
        assert!(cvar > 0.0);

        // CVaR should be greater than or equal to VaR
        let var = simd_var_historical(&returns, 0.90);
        assert!(cvar >= var);
    }

    #[test]
    fn test_portfolio_variance_symmetry() {
        let weights = vec![0.6, 0.4];
        let covariance = vec![
            0.04, 0.02,
            0.02, 0.09,
        ];
        let variance = simd_portfolio_variance(&weights, &covariance);
        assert!(variance > 0.0);

        // Verify formula: w^T * Cov * w
        let expected = weights[0] * weights[0] * covariance[0]
            + 2.0 * weights[0] * weights[1] * covariance[1]
            + weights[1] * weights[1] * covariance[3];
        assert!((variance - expected).abs() < 1e-10);
    }

    #[test]
    fn test_drawdown_properties() {
        let equity = vec![100.0, 110.0, 105.0, 115.0, 100.0, 120.0];
        let drawdowns = simd_drawdown_series(&equity);

        assert_eq!(drawdowns.len(), equity.len());
        assert!(drawdowns.iter().all(|&d| d >= 0.0)); // Non-negative
        assert_eq!(drawdowns[0], 0.0); // No drawdown at first peak
        assert!(drawdowns[2] > 0.0); // Drawdown after decline
    }

    #[test]
    fn test_rolling_volatility() {
        let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.02];
        let window = 3;
        let vols = simd_rolling_volatility(&returns, window);

        assert_eq!(vols.len(), returns.len() - window + 1);
        assert!(vols.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_vectorized_sum_accuracy() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sum = vectorized_sum(&values);
        let expected: f64 = values.iter().sum();
        assert!((sum - expected).abs() < 1e-10);
    }

    #[test]
    fn test_kahan_sum_precision() {
        let values = vec![1e10, 1.0, -1e10, 1.0];
        let sum = kahan_sum(&values);
        assert_eq!(sum, 2.0);
    }

    #[test]
    fn test_large_array_performance() {
        let returns: Vec<f64> = (0..10000).map(|i| (i as f64 * 0.001).sin()).collect();

        let var = simd_var_historical(&returns, 0.95);
        assert!(var > 0.0);

        let vols = simd_rolling_volatility(&returns, 50);
        assert!(vols.len() > 0);
    }
}
