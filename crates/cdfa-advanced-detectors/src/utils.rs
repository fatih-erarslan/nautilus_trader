//! Utility Functions for Advanced Detectors
//!
//! Common mathematical operations, statistical functions, and helper utilities
//! used across all detector modules with SIMD optimization where applicable.

use crate::*;

#[cfg(feature = "simd")]
use wide::f32x8;

/// Mathematical utilities for statistical analysis
pub struct MathUtils;

impl MathUtils {
    /// Calculate mean of a slice with SIMD optimization
    pub fn mean(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        
        #[cfg(feature = "simd")]
        {
            Self::mean_simd(data)
        }
        
        #[cfg(not(feature = "simd"))]
        {
            data.iter().sum::<f32>() / data.len() as f32
        }
    }
    
    #[cfg(feature = "simd")]
    fn mean_simd(data: &[f32]) -> f32 {
        let mut sum = f32x8::splat(0.0);
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let chunk_array: [f32; 8] = chunk.try_into().unwrap_or([0.0; 8]);
            let simd_chunk = f32x8::from(chunk_array);
            sum = sum + simd_chunk;
        }
        
        // Sum SIMD lanes
        let sum_array: [f32; 8] = sum.into();
        let mut total = sum_array.iter().sum::<f32>();
        
        // Add remainder
        total += remainder.iter().sum::<f32>();
        
        total / data.len() as f32
    }
    
    /// Calculate standard deviation with SIMD optimization
    pub fn std_dev(data: &[f32]) -> f32 {
        if data.len() < 2 {
            return 0.0;
        }
        
        let mean = Self::mean(data);
        let variance = Self::variance_with_mean(data, mean);
        variance.sqrt()
    }
    
    /// Calculate variance given pre-computed mean
    pub fn variance_with_mean(data: &[f32], mean: f32) -> f32 {
        if data.len() < 2 {
            return 0.0;
        }
        
        #[cfg(feature = "simd")]
        {
            Self::variance_simd(data, mean)
        }
        
        #[cfg(not(feature = "simd"))]
        {
            let sum_sq_diff = data.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>();
            sum_sq_diff / (data.len() - 1) as f32
        }
    }
    
    #[cfg(feature = "simd")]
    fn variance_simd(data: &[f32], mean: f32) -> f32 {
        let mean_simd = f32x8::splat(mean);
        let mut sum_sq_diff = f32x8::splat(0.0);
        
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let chunk_array: [f32; 8] = chunk.try_into().unwrap_or([0.0; 8]);
            let simd_chunk = f32x8::from(chunk_array);
            let diff = simd_chunk - mean_simd;
            sum_sq_diff = sum_sq_diff + (diff * diff);
        }
        
        // Sum SIMD lanes
        let sum_array: [f32; 8] = sum_sq_diff.into();
        let mut total = sum_array.iter().sum::<f32>();
        
        // Add remainder
        for &x in remainder {
            total += (x - mean).powi(2);
        }
        
        total / (data.len() - 1) as f32
    }
    
    /// Calculate correlation coefficient between two series
    pub fn correlation(x: &[f32], y: &[f32]) -> Result<f32> {
        if x.len() != y.len() || x.len() < 2 {
            return Err(DetectorError::InvalidInput {
                message: "Arrays must have same length and at least 2 elements".to_string()
            });
        }
        
        let mean_x = Self::mean(x);
        let mean_y = Self::mean(y);
        
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }
        
        let denominator = (sum_x2 * sum_y2).sqrt();
        if denominator < f32::EPSILON {
            Ok(0.0)
        } else {
            Ok(sum_xy / denominator)
        }
    }
    
    /// Linear regression: returns (slope, intercept, r_squared)
    pub fn linear_regression(x: &[f32], y: &[f32]) -> Result<(f32, f32, f32)> {
        if x.len() != y.len() || x.len() < 2 {
            return Err(DetectorError::InvalidInput {
                message: "Arrays must have same length and at least 2 elements".to_string()
            });
        }
        
        let n = x.len() as f32;
        let sum_x = x.iter().sum::<f32>();
        let sum_y = y.iter().sum::<f32>();
        let sum_xy = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum::<f32>();
        let sum_x2 = x.iter().map(|&xi| xi * xi).sum::<f32>();
        let sum_y2 = y.iter().map(|&yi| yi * yi).sum::<f32>();
        
        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < f32::EPSILON {
            return Ok((0.0, sum_y / n, 0.0));
        }
        
        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;
        
        // Calculate R-squared
        let y_mean = sum_y / n;
        let ss_tot = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f32>();
        let ss_res = x.iter().zip(y.iter())
            .map(|(&xi, &yi)| {
                let predicted = intercept + slope * xi;
                (yi - predicted).powi(2)
            })
            .sum::<f32>();
        
        let r_squared = if ss_tot > f32::EPSILON {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };
        
        Ok((slope, intercept, r_squared))
    }
    
    /// Exponential smoothing
    pub fn exponential_smooth(data: &[f32], alpha: f32) -> Vec<f32> {
        if data.is_empty() {
            return Vec::new();
        }
        
        let mut smoothed = Vec::with_capacity(data.len());
        smoothed.push(data[0]);
        
        for &value in &data[1..] {
            let last_smooth = smoothed[smoothed.len() - 1];
            smoothed.push(alpha * value + (1.0 - alpha) * last_smooth);
        }
        
        smoothed
    }
    
    /// Simple moving average
    pub fn moving_average(data: &[f32], window: usize) -> Vec<f32> {
        if data.len() < window || window == 0 {
            return vec![0.0; data.len()];
        }
        
        let mut ma = vec![0.0; data.len()];
        
        // Calculate first window
        let first_sum: f32 = data[0..window].iter().sum();
        ma[window - 1] = first_sum / window as f32;
        
        // Rolling calculation
        for i in window..data.len() {
            ma[i] = ma[i - 1] + (data[i] - data[i - window]) / window as f32;
        }
        
        ma
    }
    
    /// Bollinger Bands calculation
    pub fn bollinger_bands(data: &[f32], window: usize, std_multiplier: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let ma = Self::moving_average(data, window);
        let mut upper = vec![0.0; data.len()];
        let mut lower = vec![0.0; data.len()];
        
        for i in (window - 1)..data.len() {
            let window_data = &data[i - window + 1..=i];
            let std = Self::std_dev(window_data);
            
            upper[i] = ma[i] + std_multiplier * std;
            lower[i] = ma[i] - std_multiplier * std;
        }
        
        (upper, ma, lower)
    }
    
    /// Z-Score calculation
    pub fn z_score(data: &[f32], window: usize) -> Vec<f32> {
        if data.len() < window || window == 0 {
            return vec![0.0; data.len()];
        }
        
        let mut z_scores = vec![0.0; data.len()];
        
        for i in (window - 1)..data.len() {
            let window_data = &data[i - window + 1..=i];
            let mean = Self::mean(window_data);
            let std = Self::std_dev(window_data);
            
            if std > f32::EPSILON {
                z_scores[i] = (data[i] - mean) / std;
            }
        }
        
        z_scores
    }
    
    /// Percentile calculation
    pub fn percentile(data: &[f32], percentile: f32) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let index = (percentile / 100.0 * (sorted_data.len() - 1) as f32).round() as usize;
        sorted_data[index.min(sorted_data.len() - 1)]
    }
    
    /// Median calculation
    pub fn median(data: &[f32]) -> f32 {
        Self::percentile(data, 50.0)
    }
    
    /// Mode calculation (most frequent value, approximately)
    pub fn mode_approximate(data: &[f32], bins: usize) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        
        let min_val = data.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        
        if (max_val - min_val).abs() < f32::EPSILON {
            return min_val;
        }
        
        let bin_width = (max_val - min_val) / bins as f32;
        let mut counts = vec![0; bins];
        
        for &value in data {
            let bin_index = ((value - min_val) / bin_width).floor() as usize;
            let bin_index = bin_index.min(bins - 1);
            counts[bin_index] += 1;
        }
        
        let max_count_index = counts.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        min_val + (max_count_index as f32 + 0.5) * bin_width
    }
    
    /// Normalize data to [0, 1] range
    pub fn normalize(data: &[f32]) -> Vec<f32> {
        if data.is_empty() {
            return Vec::new();
        }
        
        let min_val = data.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        
        if (max_val - min_val).abs() < f32::EPSILON {
            return vec![0.5; data.len()];
        }
        
        data.iter()
            .map(|&x| (x - min_val) / (max_val - min_val))
            .collect()
    }
    
    /// Standardize data (zero mean, unit variance)
    pub fn standardize(data: &[f32]) -> Vec<f32> {
        if data.is_empty() {
            return Vec::new();
        }
        
        let mean = Self::mean(data);
        let std = Self::std_dev(data);
        
        if std < f32::EPSILON {
            return vec![0.0; data.len()];
        }
        
        data.iter()
            .map(|&x| (x - mean) / std)
            .collect()
    }
}

/// Signal processing utilities
pub struct SignalUtils;

impl SignalUtils {
    /// Apply Gaussian filter for smoothing
    pub fn gaussian_filter(data: &[f32], sigma: f32, window_size: usize) -> Vec<f32> {
        if data.is_empty() || window_size == 0 {
            return data.to_vec();
        }
        
        // Generate Gaussian kernel
        let kernel = Self::gaussian_kernel(sigma, window_size);
        Self::convolve(data, &kernel)
    }
    
    /// Generate Gaussian kernel
    fn gaussian_kernel(sigma: f32, size: usize) -> Vec<f32> {
        let half_size = size / 2;
        let mut kernel = Vec::with_capacity(size);
        let mut sum = 0.0;
        
        for i in 0..size {
            let x = i as f32 - half_size as f32;
            let value = (-x * x / (2.0 * sigma * sigma)).exp();
            kernel.push(value);
            sum += value;
        }
        
        // Normalize kernel
        for value in &mut kernel {
            *value /= sum;
        }
        
        kernel
    }
    
    /// 1D convolution
    pub fn convolve(signal: &[f32], kernel: &[f32]) -> Vec<f32> {
        if signal.is_empty() || kernel.is_empty() {
            return signal.to_vec();
        }
        
        let half_kernel = kernel.len() / 2;
        let mut result = vec![0.0; signal.len()];
        
        for i in 0..signal.len() {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            
            for j in 0..kernel.len() {
                let signal_idx = i as i32 + j as i32 - half_kernel as i32;
                
                if signal_idx >= 0 && (signal_idx as usize) < signal.len() {
                    sum += signal[signal_idx as usize] * kernel[j];
                    weight_sum += kernel[j];
                }
            }
            
            result[i] = if weight_sum > f32::EPSILON { sum / weight_sum } else { signal[i] };
        }
        
        result
    }
    
    /// Simple low-pass filter
    pub fn low_pass_filter(data: &[f32], alpha: f32) -> Vec<f32> {
        if data.is_empty() {
            return Vec::new();
        }
        
        let mut filtered = Vec::with_capacity(data.len());
        filtered.push(data[0]);
        
        for &value in &data[1..] {
            let last_filtered = filtered[filtered.len() - 1];
            filtered.push(alpha * value + (1.0 - alpha) * last_filtered);
        }
        
        filtered
    }
    
    /// Detect peaks in signal
    pub fn find_peaks(data: &[f32], min_height: f32, min_distance: usize) -> Vec<usize> {
        if data.len() < 3 {
            return Vec::new();
        }
        
        let mut peaks = Vec::new();
        
        for i in 1..(data.len() - 1) {
            if data[i] > data[i - 1] && 
               data[i] > data[i + 1] && 
               data[i] >= min_height {
                
                // Check minimum distance constraint
                if peaks.is_empty() || (i - peaks[peaks.len() - 1]) >= min_distance {
                    peaks.push(i);
                }
            }
        }
        
        peaks
    }
    
    /// Detect valleys in signal
    pub fn find_valleys(data: &[f32], max_height: f32, min_distance: usize) -> Vec<usize> {
        if data.len() < 3 {
            return Vec::new();
        }
        
        let mut valleys = Vec::new();
        
        for i in 1..(data.len() - 1) {
            if data[i] < data[i - 1] && 
               data[i] < data[i + 1] && 
               data[i] <= max_height {
                
                // Check minimum distance constraint
                if valleys.is_empty() || (i - valleys[valleys.len() - 1]) >= min_distance {
                    valleys.push(i);
                }
            }
        }
        
        valleys
    }
}

/// Validation utilities
pub struct ValidationUtils;

impl ValidationUtils {
    /// Validate market data consistency
    pub fn validate_market_data(prices: &[f32], volumes: &[f32], timestamps: &[i64]) -> Result<()> {
        if prices.len() != volumes.len() || prices.len() != timestamps.len() {
            return Err(DetectorError::InvalidInput {
                message: "Prices, volumes, and timestamps must have the same length".to_string()
            });
        }
        
        if prices.is_empty() {
            return Err(DetectorError::InvalidInput {
                message: "Empty data arrays".to_string()
            });
        }
        
        // Check for invalid values
        for (i, &price) in prices.iter().enumerate() {
            if !price.is_finite() || price <= 0.0 {
                return Err(DetectorError::InvalidInput {
                    message: format!("Invalid price at index {}: {}", i, price)
                });
            }
        }
        
        for (i, &volume) in volumes.iter().enumerate() {
            if !volume.is_finite() || volume < 0.0 {
                return Err(DetectorError::InvalidInput {
                    message: format!("Invalid volume at index {}: {}", i, volume)
                });
            }
        }
        
        // Check timestamp ordering
        for i in 1..timestamps.len() {
            if timestamps[i] <= timestamps[i - 1] {
                return Err(DetectorError::InvalidInput {
                    message: format!("Timestamps not in ascending order at index {}", i)
                });
            }
        }
        
        Ok(())
    }
    
    /// Check if data has sufficient points for analysis
    pub fn check_minimum_data(data_len: usize, required: usize, _analysis_type: &str) -> Result<()> {
        if data_len < required {
            return Err(DetectorError::InsufficientData {
                required,
                actual: data_len,
            });
        }
        Ok(())
    }
    
    /// Validate configuration parameters
    pub fn validate_config_range(value: f32, min: f32, max: f32, name: &str) -> Result<()> {
        if !value.is_finite() || value < min || value > max {
            return Err(DetectorError::ConfigError {
                message: format!("{} must be between {} and {}, got {}", name, min, max, value)
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_math_utils_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(MathUtils::mean(&data), 3.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_math_utils_std_dev() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = MathUtils::std_dev(&data);
        assert!(std > 0.0);
    }
    
    #[test]
    fn test_linear_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect linear relationship
        
        let (slope, intercept, r_squared) = MathUtils::linear_regression(&x, &y).unwrap();
        
        assert_relative_eq!(slope, 2.0, epsilon = 1e-6);
        assert_relative_eq!(intercept, 0.0, epsilon = 1e-6);
        assert_relative_eq!(r_squared, 1.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = MathUtils::moving_average(&data, 3);
        
        assert_relative_eq!(ma[2], 2.0, epsilon = 1e-6); // (1+2+3)/3
        assert_relative_eq!(ma[3], 3.0, epsilon = 1e-6); // (2+3+4)/3
        assert_relative_eq!(ma[4], 4.0, epsilon = 1e-6); // (3+4+5)/3
    }
    
    #[test]
    fn test_signal_utils_find_peaks() {
        let data = vec![1.0, 3.0, 1.0, 4.0, 1.0, 2.0, 1.0];
        let peaks = SignalUtils::find_peaks(&data, 2.0, 1);
        
        assert_eq!(peaks, vec![1, 3]); // Indices where peaks occur
    }
    
    #[test]
    fn test_validation_utils() {
        let prices = vec![100.0, 101.0, 102.0];
        let volumes = vec![1000.0, 1100.0, 1200.0];
        let timestamps = vec![1000, 1001, 1002];
        
        assert!(ValidationUtils::validate_market_data(&prices, &volumes, &timestamps).is_ok());
        
        // Test invalid case
        let invalid_volumes = vec![1000.0, 1100.0]; // Different length
        assert!(ValidationUtils::validate_market_data(&prices, &invalid_volumes, &timestamps).is_err());
    }
}