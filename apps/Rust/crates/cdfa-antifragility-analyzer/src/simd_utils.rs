//! SIMD utilities for high-performance calculations

use crate::{AntifragilityError, AntifragilityResult};
use ndarray::prelude::*;

#[cfg(feature = "simd")]
use wide::f64x4;

/// SIMD-optimized log returns calculation
#[cfg(feature = "simd")]
pub fn calculate_log_returns_simd(
    prices: &Array1<f64>,
    log_returns: &mut Array1<f64>,
) -> AntifragilityResult<()> {
    let n = prices.len();
    
    if n < 2 {
        return Ok(());
    }
    
    log_returns[0] = 0.0;
    
    // Process in SIMD chunks
    let n_simd = (n - 1) / 4;
    
    for i in 0..n_simd {
        let base_idx = i * 4 + 1;
        
        if base_idx + 3 < n {
            let current_array = [prices[base_idx], prices[base_idx + 1], prices[base_idx + 2], prices[base_idx + 3]];
            let prev_array = [prices[base_idx - 1], prices[base_idx], prices[base_idx + 1], prices[base_idx + 2]];
            
            let current_prices = f64x4::new(current_array);
            let prev_prices = f64x4::new(prev_array);
            
            let ratios = current_prices / prev_prices;
            let log_ratios = ratios.ln();
            
            let log_array = log_ratios.to_array();
            log_returns[base_idx] = log_array[0];
            log_returns[base_idx + 1] = log_array[1];
            log_returns[base_idx + 2] = log_array[2];
            log_returns[base_idx + 3] = log_array[3];
        }
    }
    
    // Handle remaining elements
    for i in (n_simd * 4 + 1)..n {
        log_returns[i] = (prices[i] / prices[i - 1]).ln();
    }
    
    Ok(())
}

/// Fallback scalar implementation
#[cfg(not(feature = "simd"))]
pub fn calculate_log_returns_simd(
    prices: &Array1<f64>,
    log_returns: &mut Array1<f64>,
) -> AntifragilityResult<()> {
    let n = prices.len();
    
    if n < 2 {
        return Ok(());
    }
    
    log_returns[0] = 0.0;
    
    for i in 1..n {
        log_returns[i] = (prices[i] / prices[i - 1]).ln();
    }
    
    Ok(())
}

/// SIMD-optimized correlation calculation
#[cfg(feature = "simd")]
pub fn calculate_correlation_simd(
    x: &[f64],
    y: &[f64],
) -> AntifragilityResult<f64> {
    if x.len() != y.len() || x.len() < 4 {
        return calculate_correlation_scalar(x, y);
    }
    
    let n = x.len();
    let n_simd = n / 4;
    
    // Calculate means using SIMD
    let mut sum_x = f64x4::splat(0.0);
    let mut sum_y = f64x4::splat(0.0);
    
    for i in 0..n_simd {
        let start = i * 4;
        let x_array = [x[start], x[start + 1], x[start + 2], x[start + 3]];
        let y_array = [y[start], y[start + 1], y[start + 2], y[start + 3]];
        
        let x_vec = f64x4::new(x_array);
        let y_vec = f64x4::new(y_array);
        
        sum_x += x_vec;
        sum_y += y_vec;
    }
    
    let mut total_sum_x = sum_x.reduce_add();
    let mut total_sum_y = sum_y.reduce_add();
    
    // Handle remaining elements
    for i in (n_simd * 4)..n {
        total_sum_x += x[i];
        total_sum_y += y[i];
    }
    
    let mean_x = total_sum_x / n as f64;
    let mean_y = total_sum_y / n as f64;
    
    // Calculate correlation components using SIMD
    let mut sum_xy = f64x4::splat(0.0);
    let mut sum_x2 = f64x4::splat(0.0);
    let mut sum_y2 = f64x4::splat(0.0);
    
    let mean_x_vec = f64x4::splat(mean_x);
    let mean_y_vec = f64x4::splat(mean_y);
    
    for i in 0..n_simd {
        let start = i * 4;
        let x_array = [x[start], x[start + 1], x[start + 2], x[start + 3]];
        let y_array = [y[start], y[start + 1], y[start + 2], y[start + 3]];
        
        let x_vec = f64x4::new(x_array);
        let y_vec = f64x4::new(y_array);
        
        let dx = x_vec - mean_x_vec;
        let dy = y_vec - mean_y_vec;
        
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }
    
    let mut total_sum_xy = sum_xy.reduce_add();
    let mut total_sum_x2 = sum_x2.reduce_add();
    let mut total_sum_y2 = sum_y2.reduce_add();
    
    // Handle remaining elements
    for i in (n_simd * 4)..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        
        total_sum_xy += dx * dy;
        total_sum_x2 += dx * dx;
        total_sum_y2 += dy * dy;
    }
    
    let denominator = (total_sum_x2 * total_sum_y2).sqrt();
    if denominator > 1e-9 {
        Ok(total_sum_xy / denominator)
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

/// SIMD-optimized moving average
#[cfg(feature = "simd")]
pub fn calculate_moving_average_simd(
    data: &Array1<f64>,
    window: usize,
) -> AntifragilityResult<Array1<f64>> {
    let n = data.len();
    
    if n < window {
        return Err(AntifragilityError::InsufficientData {
            required: window,
            actual: n,
        });
    }
    
    let mut ma = Array1::zeros(n);
    
    for i in window..n {
        let window_data = &data.as_slice().unwrap()[(i - window)..i];
        let n_simd = window / 4;
        
        let mut sum = f64x4::splat(0.0);
        
        for j in 0..n_simd {
            let start = j * 4;
            let array = [window_data[start], window_data[start + 1], window_data[start + 2], window_data[start + 3]];
            let vec = f64x4::new(array);
            sum += vec;
        }
        
        let mut total_sum = sum.reduce_add();
        
        // Handle remaining elements
        for j in (n_simd * 4)..window {
            total_sum += window_data[j];
        }
        
        ma[i] = total_sum / window as f64;
    }
    
    Ok(ma)
}

/// Fallback scalar moving average
#[cfg(not(feature = "simd"))]
pub fn calculate_moving_average_simd(
    data: &Array1<f64>,
    window: usize,
) -> AntifragilityResult<Array1<f64>> {
    let n = data.len();
    
    if n < window {
        return Err(AntifragilityError::InsufficientData {
            required: window,
            actual: n,
        });
    }
    
    let mut ma = Array1::zeros(n);
    
    for i in window..n {
        let window_sum = data.slice(s![(i - window)..i]).sum();
        ma[i] = window_sum / window as f64;
    }
    
    Ok(ma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_log_returns_simd() {
        let prices = Array1::from_vec(vec![100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0]);
        let mut log_returns = Array1::zeros(8);
        
        let result = calculate_log_returns_simd(&prices, &mut log_returns);
        assert!(result.is_ok());
        
        assert_eq!(log_returns[0], 0.0);
        assert_relative_eq!(log_returns[1], (105.0 / 100.0).ln(), epsilon = 1e-10);
    }
    
    #[test]
    fn test_correlation_simd() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        
        let result = calculate_correlation_simd(&x, &y);
        assert!(result.is_ok());
        
        let corr = result.unwrap();
        assert_relative_eq!(corr, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_moving_average_simd() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = calculate_moving_average_simd(&data, 3);
        
        assert!(result.is_ok());
        let ma = result.unwrap();
        
        assert_relative_eq!(ma[3], 2.0, epsilon = 1e-10); // (1+2+3)/3
        assert_relative_eq!(ma[4], 3.0, epsilon = 1e-10); // (2+3+4)/3
    }
}