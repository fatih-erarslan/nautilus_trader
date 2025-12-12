//! SIMD-optimized operations for ultra-fast regime detection

use wide::f32x8;

/// Calculate mean using SIMD
#[inline(always)]
pub fn simd_mean(data: &[f32]) -> f32 {
    if data.len() < 8 {
        return data.iter().sum::<f32>() / data.len() as f32;
    }
    
    let chunks = data.chunks_exact(8);
    let remainder = chunks.remainder();
    
    let mut sum = f32x8::splat(0.0);
    for chunk in chunks {
        let vals = f32x8::from(&chunk[..8]);
        sum += vals;
    }
    
    let scalar_sum: f32 = sum.reduce_add() + remainder.iter().sum::<f32>();
    scalar_sum / data.len() as f32
}

/// Calculate variance using SIMD
#[inline(always)]
pub fn simd_variance(data: &[f32], mean: f32) -> f32 {
    if data.len() < 8 {
        return data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    }
    
    let mean_vec = f32x8::splat(mean);
    let chunks = data.chunks_exact(8);
    let remainder = chunks.remainder();
    
    let mut sum_sq = f32x8::splat(0.0);
    for chunk in chunks {
        let vals = f32x8::from(&chunk[..8]);
        let diff = vals - mean_vec;
        sum_sq += diff * diff;
    }
    
    let scalar_sum: f32 = sum_sq.reduce_add() + 
        remainder.iter().map(|&x| (x - mean).powi(2)).sum::<f32>();
    scalar_sum / data.len() as f32
}

/// Calculate linear regression slope using SIMD
#[inline(always)]
pub fn simd_linear_slope(y_values: &[f32]) -> f32 {
    let n = y_values.len() as f32;
    if n < 2.0 {
        return 0.0;
    }
    
    // Pre-compute x values (0, 1, 2, ...)
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = simd_mean(y_values);
    
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    
    // Process in chunks of 8 for SIMD
    let chunks = y_values.chunks_exact(8);
    let remainder = chunks.remainder();
    let mut x_base = 0.0;
    
    for chunk in chunks {
        let y_vec = f32x8::from(&chunk[..8]);
        let x_vec = f32x8::from([
            x_base, x_base + 1.0, x_base + 2.0, x_base + 3.0,
            x_base + 4.0, x_base + 5.0, x_base + 6.0, x_base + 7.0
        ]);
        
        let x_centered = x_vec - f32x8::splat(x_mean);
        let y_centered = y_vec - f32x8::splat(y_mean);
        
        sum_xy += (x_centered * y_centered).reduce_add();
        sum_xx += (x_centered * x_centered).reduce_add();
        
        x_base += 8.0;
    }
    
    // Handle remainder
    for (i, &y) in remainder.iter().enumerate() {
        let x = x_base + i as f32;
        sum_xy += (x - x_mean) * (y - y_mean);
        sum_xx += (x - x_mean) * (x - x_mean);
    }
    
    if sum_xx > 0.0 {
        sum_xy / sum_xx
    } else {
        0.0
    }
}

/// Calculate autocorrelation using SIMD
#[inline(always)]
pub fn simd_autocorrelation(data: &[f32], lag: usize) -> f32 {
    if data.len() <= lag {
        return 0.0;
    }
    
    let mean = simd_mean(data);
    let variance = simd_variance(data, mean);
    
    if variance < 1e-10 {
        return 0.0;
    }
    
    let n = data.len() - lag;
    let mean_vec = f32x8::splat(mean);
    
    let mut covariance = 0.0;
    
    // Process in chunks
    let chunks = data[..n].chunks_exact(8);
    let remainder_idx = chunks.len() * 8;
    
    for (i, chunk) in chunks.enumerate() {
        let vals1 = f32x8::from(&chunk[..8]);
        let vals2 = f32x8::from(&data[i*8 + lag..i*8 + lag + 8]);
        
        let diff1 = vals1 - mean_vec;
        let diff2 = vals2 - mean_vec;
        
        covariance += (diff1 * diff2).reduce_add();
    }
    
    // Handle remainder
    for i in remainder_idx..n {
        covariance += (data[i] - mean) * (data[i + lag] - mean);
    }
    
    covariance / (n as f32 * variance)
}

/// Calculate RSI using SIMD
#[inline(always)]
pub fn simd_rsi(prices: &[f32], period: usize) -> f32 {
    if prices.len() < period + 1 {
        return 50.0;
    }
    
    let mut gains = Vec::with_capacity(prices.len() - 1);
    let mut losses = Vec::with_capacity(prices.len() - 1);
    
    // Calculate price changes
    for i in 1..prices.len() {
        let change = prices[i] - prices[i-1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }
    
    // Calculate average gain and loss
    let avg_gain = simd_mean(&gains[gains.len() - period..]);
    let avg_loss = simd_mean(&losses[losses.len() - period..]);
    
    if avg_loss < 1e-10 {
        return 100.0;
    }
    
    let rs = avg_gain / avg_loss;
    100.0 - (100.0 / (1.0 + rs))
}

/// Calculate Hurst exponent using SIMD (simplified R/S analysis)
#[inline(always)]
pub fn simd_hurst_exponent(data: &[f32]) -> f32 {
    if data.len() < 10 {
        return 0.5;
    }
    
    let mean = simd_mean(data);
    let std_dev = simd_variance(data, mean).sqrt();
    
    if std_dev < 1e-10 {
        return 0.5;
    }
    
    // Calculate cumulative sum of deviations
    let mut cumsum = vec![0.0; data.len()];
    let mut sum = 0.0;
    
    for i in 0..data.len() {
        sum += data[i] - mean;
        cumsum[i] = sum;
    }
    
    // Find range
    let max_val = cumsum.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = cumsum.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let range = max_val - min_val;
    
    // Simplified Hurst calculation
    let rs = range / std_dev;
    let log_rs = rs.ln();
    let log_n = (data.len() as f32).ln();
    
    // Approximate Hurst exponent
    (log_rs / log_n).clamp(0.0, 1.0)
}

/// Fast calculation of all regime features using SIMD
pub fn calculate_features_simd(prices: &[f32], volumes: &[f32]) -> crate::types::RegimeFeatures {
    let n = prices.len();
    if n < 2 {
        return crate::types::RegimeFeatures::default();
    }
    
    // Calculate returns
    let mut returns = vec![0.0; n - 1];
    for i in 1..n {
        returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1];
    }
    
    // Calculate features in parallel
    let trend_strength = simd_linear_slope(prices);
    let volatility = simd_variance(&returns, simd_mean(&returns)).sqrt();
    let autocorrelation = simd_autocorrelation(&returns, 1);
    let rsi = simd_rsi(prices, 14.min(n - 1));
    let hurst_exponent = simd_hurst_exponent(&returns);
    
    // Calculate VWAP ratio
    let vwap = calculate_vwap_simd(prices, volumes);
    let vwap_ratio = if vwap > 0.0 {
        prices.last().unwrap_or(&1.0) / vwap
    } else {
        1.0
    };
    
    // Calculate microstructure noise (simplified)
    let microstructure_noise = estimate_noise_simd(&returns);
    
    // Calculate order flow imbalance
    let order_flow_imbalance = calculate_ofi_simd(prices, volumes);
    
    crate::types::RegimeFeatures {
        trend_strength,
        volatility,
        autocorrelation,
        vwap_ratio,
        hurst_exponent,
        rsi,
        microstructure_noise,
        order_flow_imbalance,
    }
}

#[inline(always)]
fn calculate_vwap_simd(prices: &[f32], volumes: &[f32]) -> f32 {
    if prices.is_empty() || volumes.is_empty() {
        return 0.0;
    }
    
    let len = prices.len().min(volumes.len());
    let mut pv_sum = 0.0;
    let mut v_sum = 0.0;
    
    // Process in chunks of 8
    let chunks = prices[..len].chunks_exact(8);
    let vol_chunks = volumes[..len].chunks_exact(8);
    let remainder_idx = chunks.len() * 8;
    
    for (p_chunk, v_chunk) in chunks.zip(vol_chunks) {
        let p_vec = f32x8::from(&p_chunk[..8]);
        let v_vec = f32x8::from(&v_chunk[..8]);
        
        pv_sum += (p_vec * v_vec).reduce_add();
        v_sum += v_vec.reduce_add();
    }
    
    // Handle remainder
    for i in remainder_idx..len {
        pv_sum += prices[i] * volumes[i];
        v_sum += volumes[i];
    }
    
    if v_sum > 0.0 {
        pv_sum / v_sum
    } else {
        0.0
    }
}

#[inline(always)]
fn estimate_noise_simd(returns: &[f32]) -> f32 {
    if returns.len() < 2 {
        return 0.0;
    }
    
    // First-order autocorrelation as proxy for noise
    let autocorr = simd_autocorrelation(returns, 1).abs();
    
    // Higher autocorrelation means less noise
    (1.0 - autocorr).max(0.0)
}

#[inline(always)]
fn calculate_ofi_simd(prices: &[f32], volumes: &[f32]) -> f32 {
    if prices.len() < 2 || volumes.len() < 2 {
        return 0.0;
    }
    
    let len = prices.len().min(volumes.len());
    let mut buy_volume = 0.0;
    let mut sell_volume = 0.0;
    
    for i in 1..len {
        if prices[i] > prices[i-1] {
            buy_volume += volumes[i];
        } else if prices[i] < prices[i-1] {
            sell_volume += volumes[i];
        }
    }
    
    let total_volume = buy_volume + sell_volume;
    if total_volume > 0.0 {
        (buy_volume - sell_volume) / total_volume
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_simd_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_relative_eq!(simd_mean(&data), 4.5, epsilon = 1e-6);
    }
    
    #[test]
    fn test_simd_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = simd_mean(&data);
        assert_relative_eq!(simd_variance(&data, mean), 2.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_simd_operations_performance() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        
        let start = std::time::Instant::now();
        let _ = simd_mean(&data);
        let elapsed = start.elapsed();
        
        assert!(elapsed.as_nanos() < 1000, "SIMD mean took too long: {}ns", elapsed.as_nanos());
    }
}