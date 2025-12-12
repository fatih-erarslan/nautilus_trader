//! SIMD-optimized operations for Panarchy analysis

use wide::{f64x4, f64x8};
use crate::types::PCRComponents;

/// SIMD-optimized autocorrelation calculation
#[inline(always)]
pub fn simd_autocorrelation(data: &[f64], lag: usize) -> f64 {
    let n = data.len();
    if n <= lag || n < 4 {
        return scalar_autocorrelation(data, lag);
    }
    
    // Calculate mean using SIMD
    let mean = simd_mean(data);
    
    // Prepare normalized data
    let mut norm_data = vec![0.0; n];
    for i in 0..n {
        norm_data[i] = data[i] - mean;
    }
    
    // Calculate autocorrelation using SIMD
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    // Process in chunks of 8 for AVX2
    let chunks = n / 8;
    let remainder = n % 8;
    
    if chunks > 0 && lag < n - 8 {
        for i in 0..chunks {
            let idx = i * 8;
            if idx + lag + 8 <= n {
                let chunk1 = f64x8::from(&norm_data[idx..idx + 8]);
                let chunk2 = f64x8::from(&norm_data[idx + lag..idx + lag + 8]);
                
                let prod = chunk1 * chunk2;
                numerator += prod.reduce_add();
                
                let sq = chunk1 * chunk1;
                denominator += sq.reduce_add();
            }
        }
    }
    
    // Handle remainder
    for i in (chunks * 8)..n {
        if i + lag < n {
            numerator += norm_data[i] * norm_data[i + lag];
        }
        denominator += norm_data[i] * norm_data[i];
    }
    
    if denominator < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Fallback scalar autocorrelation
#[inline]
fn scalar_autocorrelation(data: &[f64], lag: usize) -> f64 {
    let n = data.len();
    if n <= lag {
        return 0.0;
    }
    
    let mean = data.iter().sum::<f64>() / n as f64;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for i in 0..n {
        let norm = data[i] - mean;
        if i + lag < n {
            numerator += norm * (data[i + lag] - mean);
        }
        denominator += norm * norm;
    }
    
    if denominator < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

/// SIMD-optimized mean calculation
#[inline(always)]
pub fn simd_mean(data: &[f64]) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    
    let mut sum = 0.0;
    let chunks = n / 8;
    let remainder = n % 8;
    
    // Process in chunks of 8
    if chunks > 0 {
        for i in 0..chunks {
            let idx = i * 8;
            let chunk = f64x8::from(&data[idx..idx + 8]);
            sum += chunk.reduce_add();
        }
    }
    
    // Handle remainder
    for i in (chunks * 8)..n {
        sum += data[i];
    }
    
    sum / n as f64
}

/// SIMD-optimized standard deviation
#[inline(always)]
pub fn simd_std_dev(data: &[f64], mean: f64) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    
    let mut sum_sq = 0.0;
    let chunks = n / 8;
    let remainder = n % 8;
    
    let mean_vec = f64x8::splat(mean);
    
    // Process in chunks of 8
    if chunks > 0 {
        for i in 0..chunks {
            let idx = i * 8;
            let chunk = f64x8::from(&data[idx..idx + 8]);
            let diff = chunk - mean_vec;
            let sq = diff * diff;
            sum_sq += sq.reduce_add();
        }
    }
    
    // Handle remainder
    for i in (chunks * 8)..n {
        let diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    
    (sum_sq / (n - 1) as f64).sqrt()
}

/// SIMD-optimized rolling window operations
pub struct SimdRollingWindow {
    window_size: usize,
    buffer: Vec<f64>,
    position: usize,
    filled: bool,
}

impl SimdRollingWindow {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            buffer: vec![0.0; window_size],
            position: 0,
            filled: false,
        }
    }
    
    pub fn push(&mut self, value: f64) {
        self.buffer[self.position] = value;
        self.position = (self.position + 1) % self.window_size;
        if self.position == 0 {
            self.filled = true;
        }
    }
    
    pub fn mean(&self) -> f64 {
        let n = if self.filled { self.window_size } else { self.position };
        if n == 0 {
            return 0.0;
        }
        simd_mean(&self.buffer[..n])
    }
    
    pub fn std_dev(&self) -> f64 {
        let n = if self.filled { self.window_size } else { self.position };
        if n < 2 {
            return 0.0;
        }
        let mean = self.mean();
        simd_std_dev(&self.buffer[..n], mean)
    }
    
    pub fn min_max(&self) -> (f64, f64) {
        let n = if self.filled { self.window_size } else { self.position };
        if n == 0 {
            return (0.0, 0.0);
        }
        
        let mut min = self.buffer[0];
        let mut max = self.buffer[0];
        
        // Process in chunks of 4 for better SIMD utilization
        let chunks = n / 4;
        let remainder = n % 4;
        
        if chunks > 0 {
            for i in 0..chunks {
                let idx = i * 4;
                let chunk = f64x4::from(&self.buffer[idx..idx + 4]);
                
                // Extract individual values for min/max comparison
                let vals = [chunk[0], chunk[1], chunk[2], chunk[3]];
                for &val in &vals {
                    if val < min {
                        min = val;
                    }
                    if val > max {
                        max = val;
                    }
                }
            }
        }
        
        // Handle remainder
        for i in (chunks * 4)..n {
            if self.buffer[i] < min {
                min = self.buffer[i];
            }
            if self.buffer[i] > max {
                max = self.buffer[i];
            }
        }
        
        (min, max)
    }
}

/// SIMD-optimized PCR calculation for a batch of values
#[inline(always)]
pub fn simd_pcr_batch(
    prices: &[f64],
    volatilities: &[f64],
    window_size: usize,
) -> Vec<PCRComponents> {
    let n = prices.len();
    let mut results = Vec::with_capacity(n);
    
    // Initialize with default values
    for _ in 0..window_size.min(n) {
        results.push(PCRComponents::new(0.5, 0.5, 0.5));
    }
    
    // Process remaining values
    for i in window_size..n {
        let window_prices = &prices[i - window_size..=i];
        let (min, max) = simd_min_max(window_prices);
        
        // Calculate potential
        let potential = if max - min > 1e-9 {
            (prices[i] - min) / (max - min)
        } else {
            0.5
        };
        
        // Calculate connectedness using autocorrelation
        let returns = calculate_returns(&prices[i - window_size..=i]);
        let connectedness = if returns.len() > 1 {
            let autocorr = simd_autocorrelation(&returns, 1);
            (autocorr + 1.0) / 2.0
        } else {
            0.5
        };
        
        // Calculate resilience (inverse of volatility)
        let resilience = 1.0 - volatilities[i].clamp(0.0, 1.0);
        
        results.push(PCRComponents::new(potential, connectedness, resilience));
    }
    
    results
}

/// SIMD-optimized min/max calculation
#[inline(always)]
fn simd_min_max(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    
    let mut min = data[0];
    let mut max = data[0];
    
    // Process in chunks of 4
    let chunks = data.len() / 4;
    let remainder = data.len() % 4;
    
    if chunks > 0 {
        for i in 0..chunks {
            let idx = i * 4;
            let chunk = f64x4::from(&data[idx..idx + 4]);
            
            // Manual min/max since wide doesn't have built-in min/max
            for j in 0..4 {
                if chunk[j] < min {
                    min = chunk[j];
                }
                if chunk[j] > max {
                    max = chunk[j];
                }
            }
        }
    }
    
    // Handle remainder
    for i in (chunks * 4)..data.len() {
        if data[i] < min {
            min = data[i];
        }
        if data[i] > max {
            max = data[i];
        }
    }
    
    (min, max)
}

/// Calculate returns from prices
#[inline]
fn calculate_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }
    
    let mut returns = Vec::with_capacity(prices.len() - 1);
    for i in 1..prices.len() {
        if prices[i - 1] > 1e-9 {
            returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
        } else {
            returns.push(0.0);
        }
    }
    returns
}

/// SIMD-optimized ADX calculation components
#[inline(always)]
pub fn simd_directional_movement(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    period: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = highs.len();
    if n < 2 {
        return (vec![0.0; n], vec![0.0; n]);
    }
    
    let mut pos_dm = vec![0.0; n];
    let mut neg_dm = vec![0.0; n];
    
    // Calculate directional movements
    for i in 1..n {
        let high_diff = highs[i] - highs[i - 1];
        let low_diff = lows[i - 1] - lows[i];
        
        if high_diff > low_diff && high_diff > 0.0 {
            pos_dm[i] = high_diff;
        }
        if low_diff > high_diff && low_diff > 0.0 {
            neg_dm[i] = low_diff;
        }
    }
    
    // Smooth using EMA
    smooth_ema(&mut pos_dm, period);
    smooth_ema(&mut neg_dm, period);
    
    (pos_dm, neg_dm)
}

/// Exponential moving average smoothing
fn smooth_ema(data: &mut [f64], period: usize) {
    if data.len() < period {
        return;
    }
    
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut ema = simd_mean(&data[..period]);
    
    for i in period..data.len() {
        ema = alpha * data[i] + (1.0 - alpha) * ema;
        data[i] = ema;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mean = simd_mean(&data);
        assert!((mean - 5.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_simd_autocorrelation() {
        let data = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let autocorr = simd_autocorrelation(&data, 1);
        assert!(autocorr < 0.0); // Should be negative for alternating pattern
    }
    
    #[test]
    fn test_rolling_window() {
        let mut window = SimdRollingWindow::new(3);
        window.push(1.0);
        window.push(2.0);
        window.push(3.0);
        assert!((window.mean() - 2.0).abs() < 1e-10);
        
        window.push(4.0);
        assert!((window.mean() - 3.0).abs() < 1e-10);
    }
}