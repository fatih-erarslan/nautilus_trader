//! Utility functions for antifragility analysis

use crate::{AntifragilityError, AntifragilityResult};
use ndarray::prelude::*;
use num_traits::Float;

/// Calculate rolling statistics
pub fn rolling_statistics(
    data: &Array1<f64>,
    window: usize,
) -> AntifragilityResult<(Array1<f64>, Array1<f64>)> {
    let n = data.len();
    if n < window {
        return Err(AntifragilityError::InsufficientData {
            required: window,
            actual: n,
        });
    }
    
    let mut means = Array1::zeros(n);
    let mut stds = Array1::zeros(n);
    
    for i in window..n {
        let window_data = data.slice(s![(i - window)..i]);
        means[i] = window_data.mean().unwrap_or(0.0);
        stds[i] = window_data.std(1.0);
    }
    
    Ok((means, stds))
}

/// Calculate exponential moving average
pub fn exponential_moving_average(
    data: &Array1<f64>,
    alpha: f64,
) -> Array1<f64> {
    let n = data.len();
    let mut ema = Array1::zeros(n);
    
    if n > 0 {
        ema[0] = data[0];
        for i in 1..n {
            ema[i] = alpha * data[i] + (1.0 - alpha) * ema[i-1];
        }
    }
    
    ema
}

/// Normalize array to 0-1 range
pub fn normalize_to_unit_range(data: &Array1<f64>) -> Array1<f64> {
    let min_val = data.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
    let max_val = data.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    
    if (max_val - min_val).abs() < 1e-10 {
        return Array1::from_elem(data.len(), 0.5);
    }
    
    data.mapv(|x| (x - min_val) / (max_val - min_val))
}

/// Calculate z-score normalization
pub fn z_score_normalize(data: &Array1<f64>) -> Array1<f64> {
    let mean = data.mean().unwrap_or(0.0);
    let std_dev = data.std(1.0);
    
    if std_dev < 1e-10 {
        return Array1::zeros(data.len());
    }
    
    data.mapv(|x| (x - mean) / std_dev)
}

/// Robust outlier detection using IQR method
pub fn detect_outliers(data: &Array1<f64>, multiplier: f64) -> Vec<usize> {
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let n = sorted_data.len();
    if n < 4 {
        return vec![];
    }
    
    let q1_idx = n / 4;
    let q3_idx = 3 * n / 4;
    let q1 = sorted_data[q1_idx];
    let q3 = sorted_data[q3_idx];
    let iqr = q3 - q1;
    
    let lower_bound = q1 - multiplier * iqr;
    let upper_bound = q3 + multiplier * iqr;
    
    let mut outliers = Vec::new();
    for (i, &value) in data.iter().enumerate() {
        if value < lower_bound || value > upper_bound {
            outliers.push(i);
        }
    }
    
    outliers
}

/// Calculate correlation matrix
pub fn correlation_matrix(data: &Array2<f64>) -> AntifragilityResult<Array2<f64>> {
    let (n_rows, n_cols) = data.dim();
    
    if n_rows < 2 {
        return Err(AntifragilityError::InsufficientData {
            required: 2,
            actual: n_rows,
        });
    }
    
    let mut corr_matrix = Array2::zeros((n_cols, n_cols));
    
    for i in 0..n_cols {
        for j in 0..n_cols {
            if i == j {
                corr_matrix[(i, j)] = 1.0;
            } else {
                let col_i = data.column(i);
                let col_j = data.column(j);
                
                let corr = calculate_correlation(&col_i, &col_j)?;
                corr_matrix[(i, j)] = corr;
            }
        }
    }
    
    Ok(corr_matrix)
}

/// Calculate Pearson correlation coefficient
pub fn calculate_correlation(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
) -> AntifragilityResult<f64> {
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

/// Calculate Sharpe ratio
pub fn calculate_sharpe_ratio(
    returns: &Array1<f64>,
    risk_free_rate: f64,
) -> AntifragilityResult<f64> {
    let n = returns.len();
    
    if n < 2 {
        return Err(AntifragilityError::InsufficientData {
            required: 2,
            actual: n,
        });
    }
    
    let mean_return = returns.mean().unwrap_or(0.0);
    let std_return = returns.std(1.0);
    
    if std_return < 1e-10 {
        return Ok(0.0);
    }
    
    Ok((mean_return - risk_free_rate) / std_return)
}

/// Calculate maximum drawdown
pub fn calculate_max_drawdown(prices: &Array1<f64>) -> AntifragilityResult<f64> {
    let n = prices.len();
    
    if n < 2 {
        return Err(AntifragilityError::InsufficientData {
            required: 2,
            actual: n,
        });
    }
    
    let mut max_drawdown = 0.0;
    let mut peak_price = prices[0];
    
    for &price in prices.iter().skip(1) {
        if price > peak_price {
            peak_price = price;
        } else {
            let drawdown = (peak_price - price) / peak_price;
            max_drawdown = max_drawdown.max(drawdown);
        }
    }
    
    Ok(max_drawdown)
}

/// Calculate Calmar ratio (return / max drawdown)
pub fn calculate_calmar_ratio(
    returns: &Array1<f64>,
    prices: &Array1<f64>,
) -> AntifragilityResult<f64> {
    let mean_return = returns.mean().unwrap_or(0.0);
    let max_drawdown = calculate_max_drawdown(prices)?;
    
    if max_drawdown < 1e-10 {
        return Ok(0.0);
    }
    
    Ok(mean_return / max_drawdown)
}

/// Calculate Value at Risk (VaR) using historical method
pub fn calculate_var(
    returns: &Array1<f64>,
    confidence_level: f64,
) -> AntifragilityResult<f64> {
    let n = returns.len();
    
    if n < 10 {
        return Err(AntifragilityError::InsufficientData {
            required: 10,
            actual: n,
        });
    }
    
    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let index = ((1.0 - confidence_level) * n as f64) as usize;
    let index = index.min(n - 1);
    
    Ok(-sorted_returns[index]) // VaR is typically reported as positive
}

/// Calculate Expected Shortfall (Conditional VaR)
pub fn calculate_expected_shortfall(
    returns: &Array1<f64>,
    confidence_level: f64,
) -> AntifragilityResult<f64> {
    let n = returns.len();
    
    if n < 10 {
        return Err(AntifragilityError::InsufficientData {
            required: 10,
            actual: n,
        });
    }
    
    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let index = ((1.0 - confidence_level) * n as f64) as usize;
    let index = index.min(n - 1);
    
    if index == 0 {
        return Ok(-sorted_returns[0]);
    }
    
    let tail_returns = &sorted_returns[0..index];
    let mean_tail = tail_returns.iter().sum::<f64>() / tail_returns.len() as f64;
    
    Ok(-mean_tail)
}

/// Safe division with fallback
pub fn safe_divide(numerator: f64, denominator: f64, fallback: f64) -> f64 {
    if denominator.abs() < 1e-10 {
        fallback
    } else {
        numerator / denominator
    }
}

/// Clamp value to range
pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

/// Linear interpolation
pub fn linear_interpolate(x: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    if (x2 - x1).abs() < 1e-10 {
        return y1;
    }
    
    y1 + (x - x1) * (y2 - y1) / (x2 - x1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_rolling_statistics() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let result = rolling_statistics(&data, 3);
        
        assert!(result.is_ok());
        let (means, stds) = result.unwrap();
        
        // Check first valid window (indices 0,1,2)
        assert_relative_eq!(means[3], 2.5, epsilon = 1e-10); // mean of [1,2,3] at index 3
    }
    
    #[test]
    fn test_exponential_moving_average() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let alpha = 0.3;
        let ema = exponential_moving_average(&data, alpha);
        
        assert_eq!(ema.len(), 5);
        assert_eq!(ema[0], 1.0);
        assert_relative_eq!(ema[1], 0.3 * 2.0 + 0.7 * 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_normalize_to_unit_range() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let normalized = normalize_to_unit_range(&data);
        
        assert_relative_eq!(normalized[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(normalized[4], 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_z_score_normalize() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let normalized = z_score_normalize(&data);
        
        let mean = normalized.mean().unwrap();
        let std_dev = normalized.std(1.0);
        
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
        assert_relative_eq!(std_dev, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_detect_outliers() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]);
        let outliers = detect_outliers(&data, 1.5);
        
        assert!(!outliers.is_empty());
        assert!(outliers.contains(&5)); // Index of the outlier value 100.0
    }
    
    #[test]
    fn test_correlation_matrix() {
        let data = Array2::from_shape_vec((5, 2), vec![
            1.0, 2.0,
            2.0, 4.0,
            3.0, 6.0,
            4.0, 8.0,
            5.0, 10.0,
        ]).unwrap();
        
        let result = correlation_matrix(&data);
        assert!(result.is_ok());
        
        let corr = result.unwrap();
        assert_relative_eq!(corr[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(corr[(1, 1)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(corr[(0, 1)], 1.0, epsilon = 1e-10); // Perfect correlation
    }
    
    #[test]
    fn test_sharpe_ratio() {
        let returns = Array1::from_vec(vec![0.1, 0.05, 0.15, 0.08, 0.12]);
        let risk_free_rate = 0.02;
        
        let result = calculate_sharpe_ratio(&returns, risk_free_rate);
        assert!(result.is_ok());
        
        let sharpe = result.unwrap();
        assert!(sharpe > 0.0);
    }
    
    #[test]
    fn test_max_drawdown() {
        let prices = Array1::from_vec(vec![100.0, 110.0, 105.0, 90.0, 95.0, 120.0]);
        let result = calculate_max_drawdown(&prices);
        assert!(result.is_ok());
        
        let max_dd = result.unwrap();
        assert!(max_dd > 0.0);
        assert_relative_eq!(max_dd, (110.0 - 90.0) / 110.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_var_calculation() {
        let returns = Array1::from_vec(vec![
            0.1, 0.05, -0.02, 0.08, -0.05, 0.12, -0.08, 0.03, -0.01, 0.07,
            -0.03, 0.09, -0.06, 0.04, -0.02
        ]);
        
        let result = calculate_var(&returns, 0.95);
        assert!(result.is_ok());
        
        let var = result.unwrap();
        assert!(var > 0.0);
    }
    
    #[test]
    fn test_expected_shortfall() {
        let returns = Array1::from_vec(vec![
            0.1, 0.05, -0.02, 0.08, -0.05, 0.12, -0.08, 0.03, -0.01, 0.07,
            -0.03, 0.09, -0.06, 0.04, -0.02
        ]);
        
        let result = calculate_expected_shortfall(&returns, 0.95);
        assert!(result.is_ok());
        
        let es = result.unwrap();
        assert!(es > 0.0);
    }
    
    #[test]
    fn test_safe_divide() {
        assert_eq!(safe_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(safe_divide(10.0, 0.0, 42.0), 42.0);
        assert_eq!(safe_divide(10.0, 1e-12, 42.0), 42.0);
    }
    
    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
        assert_eq!(clamp(-5.0, 0.0, 10.0), 0.0);
        assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);
    }
    
    #[test]
    fn test_linear_interpolate() {
        let result = linear_interpolate(1.5, 1.0, 10.0, 2.0, 20.0);
        assert_relative_eq!(result, 15.0, epsilon = 1e-10);
    }
}