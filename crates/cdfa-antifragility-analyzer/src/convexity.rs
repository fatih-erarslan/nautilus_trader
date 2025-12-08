//! Convexity analysis module
//!
//! This module calculates the convexity component of antifragility, which measures
//! the correlation between performance acceleration and volatility changes.
//! 
//! Systems with positive convexity benefit from volatility increases,
//! showing performance acceleration when volatility rises.

use crate::{AntifragilityError, AntifragilityResult};
use ndarray::prelude::*;
use num_traits::Float;
use rayon::prelude::*;

#[cfg(feature = "simd")]
use wide::f64x4;

/// Calculate convexity correlation between performance acceleration and volatility change
pub fn calculate_convexity_correlation(
    perf_acceleration: &Array1<f64>,
    vol_roc_smoothed: &Array1<f64>,
    window: usize,
) -> AntifragilityResult<f64> {
    let n = perf_acceleration.len();
    
    if n != vol_roc_smoothed.len() {
        return Err(AntifragilityError::InvalidParameters {
            message: format!("Array lengths must match: {} vs {}", n, vol_roc_smoothed.len()),
        });
    }
    
    if n < window {
        return Err(AntifragilityError::InsufficientData {
            required: window,
            actual: n,
        });
    }
    
    // Calculate rolling correlation
    let correlations = calculate_rolling_correlation(perf_acceleration, vol_roc_smoothed, window)?;
    
    // Take the mean of valid correlations
    let valid_correlations: Vec<f64> = correlations.iter()
        .filter(|&&x| x.is_finite())
        .copied()
        .collect();
    
    if valid_correlations.is_empty() {
        return Ok(0.0);
    }
    
    let mean_correlation = valid_correlations.iter().sum::<f64>() / valid_correlations.len() as f64;
    
    // Convert correlation to 0-1 scale
    Ok((mean_correlation + 1.0) / 2.0)
}

/// Calculate rolling correlation between two arrays
fn calculate_rolling_correlation(
    x: &Array1<f64>,
    y: &Array1<f64>,
    window: usize,
) -> AntifragilityResult<Array1<f64>> {
    let n = x.len();
    let mut correlations = Array1::zeros(n);
    
    for i in window..n {
        let x_window = x.slice(s![(i - window)..i]);
        let y_window = y.slice(s![(i - window)..i]);
        
        let correlation = calculate_correlation(&x_window, &y_window)?;
        correlations[i] = correlation;
    }
    
    Ok(correlations)
}

/// Calculate Pearson correlation coefficient
fn calculate_correlation(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> AntifragilityResult<f64> {
    let n = x.len();
    if n != y.len() || n < 2 {
        return Ok(0.0);
    }
    
    let mean_x = x.mean().unwrap_or(0.0);
    let mean_y = y.mean().unwrap_or(0.0);
    
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }
    
    let denominator = (sum_x2 * sum_y2).sqrt();
    if denominator > 1e-9 {
        Ok(sum_xy / denominator)
    } else {
        Ok(0.0)
    }
}

/// Calculate convexity using second derivative approximation
pub fn calculate_convexity_second_derivative(
    returns: &Array1<f64>,
    volatility: &Array1<f64>,
    window: usize,
) -> AntifragilityResult<f64> {
    let n = returns.len();
    
    if n < window + 2 {
        return Err(AntifragilityError::InsufficientData {
            required: window + 2,
            actual: n,
        });
    }
    
    let mut convexity_values = Vec::new();
    
    for i in window..(n - 2) {
        let vol_window = volatility.slice(s![(i - window)..i]);
        let ret_window = returns.slice(s![(i - window)..i]);
        
        // Calculate second derivative of returns with respect to volatility
        let convexity = calculate_local_convexity(&ret_window, &vol_window)?;
        if convexity.is_finite() {
            convexity_values.push(convexity);
        }
    }
    
    if convexity_values.is_empty() {
        return Ok(0.0);
    }
    
    let mean_convexity = convexity_values.iter().sum::<f64>() / convexity_values.len() as f64;
    
    // Normalize to 0-1 range using tanh transformation
    Ok((mean_convexity.tanh() + 1.0) / 2.0)
}

/// Calculate local convexity using quadratic fit
fn calculate_local_convexity(returns: &ArrayView1<f64>, volatility: &ArrayView1<f64>) -> AntifragilityResult<f64> {
    let n = returns.len();
    if n < 3 {
        return Ok(0.0);
    }
    
    // Sort by volatility for regression
    let mut data_points: Vec<(f64, f64)> = volatility.iter()
        .zip(returns.iter())
        .map(|(&v, &r)| (v, r))
        .collect();
    
    data_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    // Fit quadratic: y = a + bx + cx^2
    let (a, b, c) = fit_quadratic(&data_points)?;
    
    // Convexity is 2*c (second derivative)
    Ok(2.0 * c)
}

/// Fit quadratic function using least squares
fn fit_quadratic(data: &[(f64, f64)]) -> AntifragilityResult<(f64, f64, f64)> {
    let n = data.len();
    if n < 3 {
        return Ok((0.0, 0.0, 0.0));
    }
    
    let mut sum_x = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_x3 = 0.0;
    let mut sum_x4 = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2y = 0.0;
    
    for &(x, y) in data {
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x3 * x;
        
        sum_x += x;
        sum_x2 += x2;
        sum_x3 += x3;
        sum_x4 += x4;
        sum_y += y;
        sum_xy += x * y;
        sum_x2y += x2 * y;
    }
    
    let n_f = n as f64;
    
    // Set up normal equations: Ax = b
    let a_matrix = [
        [n_f, sum_x, sum_x2],
        [sum_x, sum_x2, sum_x3],
        [sum_x2, sum_x3, sum_x4],
    ];
    
    let b_vector = [sum_y, sum_xy, sum_x2y];
    
    // Solve using Gaussian elimination
    let solution = solve_3x3_system(&a_matrix, &b_vector)?;
    
    Ok((solution[0], solution[1], solution[2]))
}

/// Solve 3x3 linear system using Gaussian elimination
fn solve_3x3_system(a: &[[f64; 3]; 3], b: &[f64; 3]) -> AntifragilityResult<[f64; 3]> {
    let mut matrix = *a;
    let mut rhs = *b;
    
    // Forward elimination
    for i in 0..3 {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..3 {
            if matrix[k][i].abs() > matrix[max_row][i].abs() {
                max_row = k;
            }
        }
        
        // Swap rows
        if max_row != i {
            matrix.swap(i, max_row);
            rhs.swap(i, max_row);
        }
        
        // Make diagonal element 1
        let pivot = matrix[i][i];
        if pivot.abs() < 1e-12 {
            return Ok([0.0, 0.0, 0.0]); // Singular matrix
        }
        
        for j in 0..3 {
            matrix[i][j] /= pivot;
        }
        rhs[i] /= pivot;
        
        // Eliminate column
        for k in 0..3 {
            if k != i {
                let factor = matrix[k][i];
                for j in 0..3 {
                    matrix[k][j] -= factor * matrix[i][j];
                }
                rhs[k] -= factor * rhs[i];
            }
        }
    }
    
    Ok(rhs)
}

/// Calculate convexity using option-theoretic approach
pub fn calculate_option_convexity(
    returns: &Array1<f64>,
    volatility: &Array1<f64>,
    window: usize,
) -> AntifragilityResult<f64> {
    let n = returns.len();
    
    if n < window {
        return Err(AntifragilityError::InsufficientData {
            required: window,
            actual: n,
        });
    }
    
    let mut convexity_values = Vec::new();
    
    for i in window..n {
        let vol_window = volatility.slice(s![(i - window)..i]);
        let ret_window = returns.slice(s![(i - window)..i]);
        
        // Calculate gamma-like measure
        let gamma = calculate_gamma_measure(&ret_window, &vol_window)?;
        if gamma.is_finite() {
            convexity_values.push(gamma);
        }
    }
    
    if convexity_values.is_empty() {
        return Ok(0.0);
    }
    
    let mean_gamma = convexity_values.iter().sum::<f64>() / convexity_values.len() as f64;
    
    // Normalize to 0-1 range
    Ok((mean_gamma.tanh() + 1.0) / 2.0)
}

/// Calculate gamma-like measure (second derivative of value with respect to underlying)
fn calculate_gamma_measure(returns: &ArrayView1<f64>, volatility: &ArrayView1<f64>) -> AntifragilityResult<f64> {
    let n = returns.len();
    if n < 3 {
        return Ok(0.0);
    }
    
    // Calculate discrete second derivative
    let mut second_derivatives = Vec::new();
    
    for i in 1..(n - 1) {
        let vol_prev = volatility[i - 1];
        let vol_curr = volatility[i];
        let vol_next = volatility[i + 1];
        
        let ret_prev = returns[i - 1];
        let ret_curr = returns[i];
        let ret_next = returns[i + 1];
        
        // Calculate second derivative using central difference
        let h1 = vol_curr - vol_prev;
        let h2 = vol_next - vol_curr;
        
        if h1.abs() > 1e-9 && h2.abs() > 1e-9 {
            let d2r_dv2 = (ret_next - ret_curr) / h2 - (ret_curr - ret_prev) / h1;
            let d2r_dv2_normalized = d2r_dv2 / ((h1 + h2) / 2.0);
            
            second_derivatives.push(d2r_dv2_normalized);
        }
    }
    
    if second_derivatives.is_empty() {
        return Ok(0.0);
    }
    
    let mean_second_derivative = second_derivatives.iter().sum::<f64>() / second_derivatives.len() as f64;
    Ok(mean_second_derivative)
}

/// Calculate asymmetric convexity (upside vs downside)
pub fn calculate_asymmetric_convexity(
    returns: &Array1<f64>,
    volatility: &Array1<f64>,
    window: usize,
) -> AntifragilityResult<(f64, f64)> {
    let n = returns.len();
    
    if n < window {
        return Err(AntifragilityError::InsufficientData {
            required: window,
            actual: n,
        });
    }
    
    let mut upside_convexity = Vec::new();
    let mut downside_convexity = Vec::new();
    
    for i in window..n {
        let vol_window = volatility.slice(s![(i - window)..i]);
        let ret_window = returns.slice(s![(i - window)..i]);
        
        // Separate upside and downside movements
        let mut upside_pairs = Vec::new();
        let mut downside_pairs = Vec::new();
        
        for j in 0..vol_window.len() {
            if ret_window[j] > 0.0 {
                upside_pairs.push((vol_window[j], ret_window[j]));
            } else {
                downside_pairs.push((vol_window[j], ret_window[j]));
            }
        }
        
        // Calculate convexity for each side
        if upside_pairs.len() >= 3 {
            let (_, _, c_up) = fit_quadratic(&upside_pairs)?;
            upside_convexity.push(c_up);
        }
        
        if downside_pairs.len() >= 3 {
            let (_, _, c_down) = fit_quadratic(&downside_pairs)?;
            downside_convexity.push(c_down);
        }
    }
    
    let mean_upside = if upside_convexity.is_empty() {
        0.0
    } else {
        upside_convexity.iter().sum::<f64>() / upside_convexity.len() as f64
    };
    
    let mean_downside = if downside_convexity.is_empty() {
        0.0
    } else {
        downside_convexity.iter().sum::<f64>() / downside_convexity.len() as f64
    };
    
    Ok((mean_upside, mean_downside))
}

/// SIMD-optimized correlation calculation
#[cfg(feature = "simd")]
pub fn calculate_correlation_simd(x: &[f64], y: &[f64]) -> AntifragilityResult<f64> {
    if x.len() != y.len() || x.len() < 4 {
        return calculate_correlation_scalar(x, y);
    }
    
    let n = x.len();
    let n_simd = n / 4;
    
    // Calculate means
    let mut sum_x = f64x4::splat(0.0);
    let mut sum_y = f64x4::splat(0.0);
    
    for i in 0..n_simd {
        let idx = i * 4;
        let x_vec = f64x4::new([x[idx], x[idx + 1], x[idx + 2], x[idx + 3]]);
        let y_vec = f64x4::new([y[idx], y[idx + 1], y[idx + 2], y[idx + 3]]);
        
        sum_x += x_vec;
        sum_y += y_vec;
    }
    
    let mean_x = sum_x.reduce_add() / n as f64;
    let mean_y = sum_y.reduce_add() / n as f64;
    
    // Calculate correlation components
    let mut sum_xy = f64x4::splat(0.0);
    let mut sum_x2 = f64x4::splat(0.0);
    let mut sum_y2 = f64x4::splat(0.0);
    
    let mean_x_vec = f64x4::splat(mean_x);
    let mean_y_vec = f64x4::splat(mean_y);
    
    for i in 0..n_simd {
        let idx = i * 4;
        let x_vec = f64x4::new([x[idx], x[idx + 1], x[idx + 2], x[idx + 3]]);
        let y_vec = f64x4::new([y[idx], y[idx + 1], y[idx + 2], y[idx + 3]]);
        
        let dx = x_vec - mean_x_vec;
        let dy = y_vec - mean_y_vec;
        
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }
    
    let sum_xy_total = sum_xy.reduce_add();
    let sum_x2_total = sum_x2.reduce_add();
    let sum_y2_total = sum_y2.reduce_add();
    
    let denominator = (sum_x2_total * sum_y2_total).sqrt();
    if denominator > 1e-9 {
        Ok(sum_xy_total / denominator)
    } else {
        Ok(0.0)
    }
}

/// Scalar correlation calculation fallback
fn calculate_correlation_scalar(x: &[f64], y: &[f64]) -> AntifragilityResult<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return Ok(0.0);
    }
    
    let n = x.len();
    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;
    
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }
    
    let denominator = (sum_x2 * sum_y2).sqrt();
    if denominator > 1e-9 {
        Ok(sum_xy / denominator)
    } else {
        Ok(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    fn generate_test_data() -> (Array1<f64>, Array1<f64>) {
        let n = 100;
        let mut perf_accel = Array1::zeros(n);
        let mut vol_roc = Array1::zeros(n);
        
        for i in 0..n {
            let t = i as f64 * 0.1;
            perf_accel[i] = 0.1 * t.sin() + 0.05 * (t * 2.0).cos();
            vol_roc[i] = 0.05 * (t * 0.5).sin() + 0.02 * (t * 3.0).cos();
        }
        
        (perf_accel, vol_roc)
    }
    
    #[test]
    fn test_convexity_correlation() {
        let (perf_accel, vol_roc) = generate_test_data();
        
        let result = calculate_convexity_correlation(&perf_accel, &vol_roc, 20);
        assert!(result.is_ok());
        
        let convexity = result.unwrap();
        assert!(convexity >= 0.0 && convexity <= 1.0);
    }
    
    #[test]
    fn test_rolling_correlation() {
        let (x, y) = generate_test_data();
        
        let result = calculate_rolling_correlation(&x, &y, 20);
        assert!(result.is_ok());
        
        let correlations = result.unwrap();
        assert_eq!(correlations.len(), 100);
        
        // Check that correlations are in valid range
        for i in 20..100 {
            assert!(correlations[i] >= -1.0 && correlations[i] <= 1.0);
        }
    }
    
    #[test]
    fn test_perfect_correlation() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]); // y = 2x
        
        let result = calculate_convexity_correlation(&x, &y, 3);
        assert!(result.is_ok());
        
        let convexity = result.unwrap();
        assert_relative_eq!(convexity, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_no_correlation() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0]); // constant
        
        let result = calculate_convexity_correlation(&x, &y, 3);
        assert!(result.is_ok());
        
        let convexity = result.unwrap();
        assert_relative_eq!(convexity, 0.5, epsilon = 1e-1); // Should be near 0.5 (neutral)
    }
    
    #[test]
    fn test_quadratic_fit() {
        let data = vec![
            (0.0, 0.0),
            (1.0, 1.0),
            (2.0, 4.0),
            (3.0, 9.0),
        ]; // y = x^2
        
        let result = fit_quadratic(&data);
        assert!(result.is_ok());
        
        let (a, b, c) = result.unwrap();
        assert_relative_eq!(a, 0.0, epsilon = 1e-10);
        assert_relative_eq!(b, 0.0, epsilon = 1e-10);
        assert_relative_eq!(c, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_convexity_second_derivative() {
        let n = 50;
        let mut returns = Array1::zeros(n);
        let mut volatility = Array1::zeros(n);
        
        // Create convex relationship
        for i in 0..n {
            let vol = (i as f64) / 10.0;
            volatility[i] = vol;
            returns[i] = vol * vol; // Quadratic relationship
        }
        
        let result = calculate_convexity_second_derivative(&returns, &volatility, 10);
        assert!(result.is_ok());
        
        let convexity = result.unwrap();
        assert!(convexity > 0.5); // Should detect positive convexity
    }
    
    #[test]
    fn test_asymmetric_convexity() {
        let n = 50;
        let mut returns = Array1::zeros(n);
        let mut volatility = Array1::zeros(n);
        
        for i in 0..n {
            let vol = (i as f64) / 10.0;
            volatility[i] = vol;
            
            // Different convexity for positive and negative returns
            if i % 2 == 0 {
                returns[i] = vol * vol; // Positive returns: convex
            } else {
                returns[i] = -vol; // Negative returns: linear
            }
        }
        
        let result = calculate_asymmetric_convexity(&returns, &volatility, 10);
        assert!(result.is_ok());
        
        let (upside, downside) = result.unwrap();
        assert!(upside > downside); // Upside should be more convex
    }
    
    #[test]
    fn test_insufficient_data() {
        let small_array = Array1::from_vec(vec![1.0, 2.0]);
        let result = calculate_convexity_correlation(&small_array, &small_array, 5);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_mismatched_lengths() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![1.0, 2.0]);
        
        let result = calculate_convexity_correlation(&x, &y, 2);
        assert!(result.is_err());
    }
    
    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        
        let result_simd = calculate_correlation_simd(&x, &y);
        let result_scalar = calculate_correlation_scalar(&x, &y);
        
        assert!(result_simd.is_ok());
        assert!(result_scalar.is_ok());
        
        assert_relative_eq!(result_simd.unwrap(), result_scalar.unwrap(), epsilon = 1e-10);
    }
}