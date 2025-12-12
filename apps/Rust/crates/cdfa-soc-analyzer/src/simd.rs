//! SIMD utilities for SOC analysis

use wide::f32x8;

/// SIMD-accelerated distance calculation between arrays
pub fn simd_distance_array(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    let mut sum = 0.0f32;
    let chunks = a.len() / 8;
    
    // Process 8 elements at a time with SIMD
    for i in 0..chunks {
        let start_idx = i * 8;
        
        let a_vec = f32x8::new([
            a[start_idx], a[start_idx + 1], a[start_idx + 2], a[start_idx + 3],
            a[start_idx + 4], a[start_idx + 5], a[start_idx + 6], a[start_idx + 7],
        ]);
        
        let b_vec = f32x8::new([
            b[start_idx], b[start_idx + 1], b[start_idx + 2], b[start_idx + 3],
            b[start_idx + 4], b[start_idx + 5], b[start_idx + 6], b[start_idx + 7],
        ]);
        
        let diff = a_vec - b_vec;
        let squared = diff * diff;
        
        // Horizontal sum of the vector
        let array = squared.to_array();
        sum += array.iter().sum::<f32>();
    }
    
    // Handle remaining elements
    for i in (chunks * 8)..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    sum.sqrt()
}

/// SIMD-accelerated maximum absolute difference calculation
pub fn simd_max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    let mut max_diff = 0.0f32;
    let chunks = a.len() / 8;
    
    // Process 8 elements at a time with SIMD
    for i in 0..chunks {
        let start_idx = i * 8;
        
        let a_vec = f32x8::new([
            a[start_idx], a[start_idx + 1], a[start_idx + 2], a[start_idx + 3],
            a[start_idx + 4], a[start_idx + 5], a[start_idx + 6], a[start_idx + 7],
        ]);
        
        let b_vec = f32x8::new([
            b[start_idx], b[start_idx + 1], b[start_idx + 2], b[start_idx + 3],
            b[start_idx + 4], b[start_idx + 5], b[start_idx + 6], b[start_idx + 7],
        ]);
        
        let diff = (a_vec - b_vec).abs();
        let array = diff.to_array();
        
        for &val in &array {
            if val > max_diff {
                max_diff = val;
            }
        }
    }
    
    // Handle remaining elements
    for i in (chunks * 8)..a.len() {
        let diff = (a[i] - b[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    
    max_diff
}

/// SIMD-accelerated mean calculation
pub fn simd_mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    
    let mut sum = 0.0f32;
    let chunks = data.len() / 8;
    
    // Process 8 elements at a time with SIMD
    for i in 0..chunks {
        let start_idx = i * 8;
        
        let vec = f32x8::new([
            data[start_idx], data[start_idx + 1], data[start_idx + 2], data[start_idx + 3],
            data[start_idx + 4], data[start_idx + 5], data[start_idx + 6], data[start_idx + 7],
        ]);
        
        let array = vec.to_array();
        sum += array.iter().sum::<f32>();
    }
    
    // Handle remaining elements
    for i in (chunks * 8)..data.len() {
        sum += data[i];
    }
    
    sum / data.len() as f32
}

/// SIMD-accelerated variance calculation
pub fn simd_variance(data: &[f32], mean: f32) -> f32 {
    if data.len() <= 1 {
        return 0.0;
    }
    
    let mut sum_sq_diff = 0.0f32;
    let chunks = data.len() / 8;
    let mean_vec = f32x8::splat(mean);
    
    // Process 8 elements at a time with SIMD
    for i in 0..chunks {
        let start_idx = i * 8;
        
        let data_vec = f32x8::new([
            data[start_idx], data[start_idx + 1], data[start_idx + 2], data[start_idx + 3],
            data[start_idx + 4], data[start_idx + 5], data[start_idx + 6], data[start_idx + 7],
        ]);
        
        let diff = data_vec - mean_vec;
        let sq_diff = diff * diff;
        
        let array = sq_diff.to_array();
        sum_sq_diff += array.iter().sum::<f32>();
    }
    
    // Handle remaining elements
    for i in (chunks * 8)..data.len() {
        let diff = data[i] - mean;
        sum_sq_diff += diff * diff;
    }
    
    sum_sq_diff / (data.len() - 1) as f32
}

/// SIMD-accelerated element-wise comparison with tolerance
pub fn simd_within_tolerance(a: &[f32], b: &[f32], tolerance: f32) -> Vec<bool> {
    assert_eq!(a.len(), b.len());
    
    let mut result = Vec::with_capacity(a.len());
    let chunks = a.len() / 8;
    let tolerance_vec = f32x8::splat(tolerance);
    
    // Process 8 elements at a time with SIMD
    for i in 0..chunks {
        let start_idx = i * 8;
        
        let a_vec = f32x8::new([
            a[start_idx], a[start_idx + 1], a[start_idx + 2], a[start_idx + 3],
            a[start_idx + 4], a[start_idx + 5], a[start_idx + 6], a[start_idx + 7],
        ]);
        
        let b_vec = f32x8::new([
            b[start_idx], b[start_idx + 1], b[start_idx + 2], b[start_idx + 3],
            b[start_idx + 4], b[start_idx + 5], b[start_idx + 6], b[start_idx + 7],
        ]);
        
        let diff = (a_vec - b_vec).abs();
        let within = diff.cmp_le(tolerance_vec);
        
        // Convert SIMD mask to boolean values
        let mask_array = within.to_array();
        for mask_val in mask_array {
            result.push(mask_val != 0);
        }
    }
    
    // Handle remaining elements
    for i in (chunks * 8)..a.len() {
        let diff = (a[i] - b[i]).abs();
        result.push(diff <= tolerance);
    }
    
    result
}

/// SIMD-accelerated autocorrelation calculation
pub fn simd_autocorrelation(data: &[f32], lag: usize, mean: f32, variance: f32) -> f32 {
    if data.len() <= lag || variance < 1e-12 {
        return 0.0;
    }
    
    let mut covariance = 0.0f32;
    let n = data.len() - lag;
    let chunks = n / 8;
    let mean_vec = f32x8::splat(mean);
    
    // Process 8 elements at a time with SIMD
    for i in 0..chunks {
        let start_idx = i * 8;
        
        let current_vec = f32x8::new([
            data[start_idx + lag], data[start_idx + lag + 1], 
            data[start_idx + lag + 2], data[start_idx + lag + 3],
            data[start_idx + lag + 4], data[start_idx + lag + 5], 
            data[start_idx + lag + 6], data[start_idx + lag + 7],
        ]);
        
        let lagged_vec = f32x8::new([
            data[start_idx], data[start_idx + 1], 
            data[start_idx + 2], data[start_idx + 3],
            data[start_idx + 4], data[start_idx + 5], 
            data[start_idx + 6], data[start_idx + 7],
        ]);
        
        let current_centered = current_vec - mean_vec;
        let lagged_centered = lagged_vec - mean_vec;
        let product = current_centered * lagged_centered;
        
        let array = product.to_array();
        covariance += array.iter().sum::<f32>();
    }
    
    // Handle remaining elements
    for i in (chunks * 8)..n {
        let current_centered = data[i + lag] - mean;
        let lagged_centered = data[i] - mean;
        covariance += current_centered * lagged_centered;
    }
    
    covariance / (n as f32 * variance)
}

/// SIMD-accelerated rolling statistics calculation
pub struct SimdRollingStats {
    window_size: usize,
}

impl SimdRollingStats {
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }
    
    /// Calculate rolling mean using SIMD
    pub fn rolling_mean(&self, data: &[f32]) -> Vec<f32> {
        let n = data.len();
        if n < self.window_size {
            return vec![0.0; n];
        }
        
        let mut result = Vec::with_capacity(n);
        
        // Calculate first window
        let first_mean = simd_mean(&data[0..self.window_size]);
        result.push(first_mean);
        
        // Use incremental calculation for subsequent windows
        let mut window_sum = first_mean * self.window_size as f32;
        
        for i in 1..=(n - self.window_size) {
            window_sum = window_sum - data[i - 1] + data[i + self.window_size - 1];
            result.push(window_sum / self.window_size as f32);
        }
        
        result
    }
    
    /// Calculate rolling variance using SIMD
    pub fn rolling_variance(&self, data: &[f32]) -> Vec<f32> {
        let n = data.len();
        if n < self.window_size {
            return vec![0.0; n];
        }
        
        let means = self.rolling_mean(data);
        let mut result = Vec::with_capacity(means.len());
        
        for i in 0..means.len() {
            let start_idx = i;
            let end_idx = i + self.window_size;
            let window = &data[start_idx..end_idx];
            let variance = simd_variance(window, means[i]);
            result.push(variance);
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_distance() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        
        let distance = simd_distance_array(&a, &b);
        let expected = (8.0f32).sqrt(); // Each element differs by 1, so sqrt(8*1^2) = sqrt(8)
        
        assert!((distance - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_simd_max_abs_diff() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 1.0, 5.0, 4.0, 6.0, 7.0, 8.0, 9.0];
        
        let max_diff = simd_max_abs_diff(&a, &b);
        assert_eq!(max_diff, 2.0); // |3-5| = 2 is the maximum difference
    }
    
    #[test]
    fn test_simd_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mean = simd_mean(&data);
        let expected = 4.5; // (1+2+3+4+5+6+7+8)/8 = 36/8 = 4.5
        
        assert!((mean - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_simd_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mean = simd_mean(&data);
        let variance = simd_variance(&data, mean);
        
        // Expected variance for this sequence
        let expected = 6.0; // Variance of 1,2,3,4,5,6,7,8
        assert!((variance - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_simd_within_tolerance() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.1, 2.2, 2.8, 4.5];
        let tolerance = 0.3;
        
        let result = simd_within_tolerance(&a, &b, tolerance);
        assert_eq!(result, vec![true, false, true, false]); // 0.1 <= 0.3, 0.2 <= 0.3, 0.2 <= 0.3, 0.5 > 0.3
    }
    
    #[test]
    fn test_rolling_stats() {
        let stats = SimdRollingStats::new(3);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        
        let rolling_means = stats.rolling_mean(&data);
        let expected_means = vec![2.0, 3.0, 4.0, 5.0]; // (1+2+3)/3=2, (2+3+4)/3=3, etc.
        
        for (actual, expected) in rolling_means.iter().zip(expected_means.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}