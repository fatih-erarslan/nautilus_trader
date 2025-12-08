//! Enhanced utility functions for Black Swan detection with IQAD integration
//!
//! This module provides comprehensive utilities for statistical analysis,
//! validation, signal processing, and performance optimization.

use crate::error::*;
use crate::types::*;
use rand::Rng;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use nalgebra::DVector;
use ndarray::Array1;
use statrs::distribution::Distribution;

/// Validation utilities for input data
pub mod validation {
    use super::*;
    
    /// Validate that data has minimum required size
    pub fn validate_min_size(data: &[f64], min_size: usize, context: &str) -> BlackSwanResult<()> {
        if data.len() < min_size {
            return Err(BlackSwanError::InsufficientData {
                required: min_size,
                actual: data.len(),
            });
        }
        Ok(())
    }
    
    /// Validate that all values are finite (not NaN or infinite)
    pub fn validate_all_finite(data: &[f64], context: &str) -> BlackSwanResult<()> {
        for (i, &value) in data.iter().enumerate() {
            if !value.is_finite() {
                return Err(BlackSwanError::InvalidInput(
                    format!("{}: Non-finite value at index {}: {}", context, i, value)
                ));
            }
        }
        Ok(())
    }
    
    /// Validate that all values are positive
    pub fn validate_all_positive(data: &[f64], context: &str) -> BlackSwanResult<()> {
        for (i, &value) in data.iter().enumerate() {
            if value <= 0.0 {
                return Err(BlackSwanError::InvalidInput(
                    format!("{}: Non-positive value at index {}: {}", context, i, value)
                ));
            }
        }
        Ok(())
    }
    
    /// Validate that data is within expected range
    pub fn validate_range(data: &[f64], min: f64, max: f64, context: &str) -> BlackSwanResult<()> {
        for (i, &value) in data.iter().enumerate() {
            if value < min || value > max {
                return Err(BlackSwanError::InvalidInput(
                    format!("{}: Value {} at index {} outside range [{}, {}]", 
                           context, value, i, min, max)
                ));
            }
        }
        Ok(())
    }
    
    /// Validate probability values (0 <= p <= 1)
    pub fn validate_probabilities(data: &[f64], context: &str) -> BlackSwanResult<()> {
        validate_range(data, 0.0, 1.0, context)
    }
    
    /// Validate that arrays have matching lengths
    pub fn validate_matching_lengths<T, U>(
        data1: &[T], 
        data2: &[U], 
        context: &str
    ) -> BlackSwanResult<()> {
        if data1.len() != data2.len() {
            return Err(BlackSwanError::InvalidInput(
                format!("{}: Array length mismatch: {} vs {}", 
                       context, data1.len(), data2.len())
            ));
        }
        Ok(())
    }
}

/// Bootstrap sampling utility
pub fn bootstrap_sample<T: Clone>(data: &[T], rng: &mut impl Rng) -> Vec<T> {
    let n = data.len();
    let mut sample = Vec::with_capacity(n);
    
    for _ in 0..n {
        let idx = rng.gen_range(0..n);
        sample.push(data[idx].clone());
    }
    
    sample
}

/// Fast median calculation using quickselect algorithm
pub fn fast_median(data: &mut [f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    
    let n = data.len();
    if n % 2 == 1 {
        quickselect(data, n / 2)
    } else {
        let mid1 = quickselect(data, n / 2 - 1);
        let mid2 = quickselect(data, n / 2);
        (mid1 + mid2) / 2.0
    }
}

/// Quickselect algorithm for k-th order statistic
fn quickselect(data: &mut [f64], k: usize) -> f64 {
    if data.len() == 1 {
        return data[0];
    }
    
    let pivot_idx = partition(data);
    
    if k == pivot_idx {
        data[k]
    } else if k < pivot_idx {
        quickselect(&mut data[..pivot_idx], k)
    } else {
        quickselect(&mut data[pivot_idx + 1..], k - pivot_idx - 1)
    }
}

/// Partition function for quickselect
fn partition(data: &mut [f64]) -> usize {
    let pivot = data[data.len() - 1];
    let mut i = 0;
    
    for j in 0..data.len() - 1 {
        if data[j] <= pivot {
            data.swap(i, j);
            i += 1;
        }
    }
    
    data.swap(i, data.len() - 1);
    i
}

/// Calculate rolling statistics efficiently
pub struct RollingStatistics {
    window_size: usize,
    sum: f64,
    sum_squares: f64,
    values: std::collections::VecDeque<f64>,
}

impl RollingStatistics {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            sum: 0.0,
            sum_squares: 0.0,
            values: std::collections::VecDeque::with_capacity(window_size),
        }
    }
    
    pub fn add(&mut self, value: f64) {
        // Remove old value if window is full
        if self.values.len() == self.window_size {
            if let Some(old_value) = self.values.pop_front() {
                self.sum -= old_value;
                self.sum_squares -= old_value * old_value;
            }
        }
        
        // Add new value
        self.values.push_back(value);
        self.sum += value;
        self.sum_squares += value * value;
    }
    
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f64
        }
    }
    
    pub fn variance(&self) -> f64 {
        if self.values.len() < 2 {
            0.0
        } else {
            let n = self.values.len() as f64;
            let mean = self.mean();
            (self.sum_squares - n * mean * mean) / (n - 1.0)
        }
    }
    
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    
    pub fn skewness(&self) -> f64 {
        if self.values.len() < 3 {
            return 0.0;
        }
        
        let mean = self.mean();
        let std_dev = self.std_dev();
        
        if std_dev == 0.0 {
            return 0.0;
        }
        
        let n = self.values.len() as f64;
        let sum_cubed: f64 = self.values.iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum();
        
        (n / ((n - 1.0) * (n - 2.0))) * sum_cubed
    }
    
    pub fn kurtosis(&self) -> f64 {
        if self.values.len() < 4 {
            return 0.0;
        }
        
        let mean = self.mean();
        let std_dev = self.std_dev();
        
        if std_dev == 0.0 {
            return 0.0;
        }
        
        let n = self.values.len() as f64;
        let sum_fourth: f64 = self.values.iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum();
        
        let factor = (n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0));
        let correction = (3.0 * (n - 1.0) * (n - 1.0)) / ((n - 2.0) * (n - 3.0));
        
        factor * sum_fourth - correction
    }
    
    pub fn len(&self) -> usize {
        self.values.len()
    }
    
    pub fn is_full(&self) -> bool {
        self.values.len() == self.window_size
    }
}

/// Efficient autocorrelation calculation
pub fn autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag {
        return 0.0;
    }
    
    let n = data.len() - lag;
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for i in 0..n {
        let x_i = data[i] - mean;
        let x_i_lag = data[i + lag] - mean;
        numerator += x_i * x_i_lag;
        denominator += x_i * x_i;
    }
    
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Ljung-Box test for autocorrelation
pub fn ljung_box_test(data: &[f64], max_lag: usize) -> f64 {
    let n = data.len() as f64;
    let mut statistic = 0.0;
    
    for lag in 1..=max_lag {
        let autocorr = autocorrelation(data, lag);
        statistic += autocorr * autocorr / (n - lag as f64);
    }
    
    statistic * n * (n + 2.0)
}

/// Efficient quantile calculation
pub fn quantile(data: &[f64], q: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let index = q * (sorted_data.len() - 1) as f64;
    let lower_idx = index.floor() as usize;
    let upper_idx = index.ceil() as usize;
    
    if lower_idx == upper_idx {
        sorted_data[lower_idx]
    } else {
        let weight = index - lower_idx as f64;
        sorted_data[lower_idx] * (1.0 - weight) + sorted_data[upper_idx] * weight
    }
}

/// Jarque-Bera test for normality
pub fn jarque_bera_test(data: &[f64]) -> f64 {
    if data.len() < 3 {
        return 0.0;
    }
    
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / data.len() as f64;
    
    if variance == 0.0 {
        return 0.0;
    }
    
    let std_dev = variance.sqrt();
    let n = data.len() as f64;
    
    let skewness = data.iter()
        .map(|x| ((x - mean) / std_dev).powi(3))
        .sum::<f64>() / n;
    
    let kurtosis = data.iter()
        .map(|x| ((x - mean) / std_dev).powi(4))
        .sum::<f64>() / n;
    
    let excess_kurtosis = kurtosis - 3.0;
    
    (n / 6.0) * (skewness * skewness + excess_kurtosis * excess_kurtosis / 4.0)
}

/// Anderson-Darling test for normality
pub fn anderson_darling_test(data: &[f64]) -> f64 {
    if data.len() < 3 {
        return 0.0;
    }
    
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let n = sorted_data.len() as f64;
    let mean = sorted_data.iter().sum::<f64>() / n;
    let variance = sorted_data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();
    
    let mut statistic = 0.0;
    
    for (i, &x) in sorted_data.iter().enumerate() {
        let z = (x - mean) / std_dev;
        let phi = 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
        let one_minus_phi = 1.0 - phi;
        
        if phi > 0.0 && one_minus_phi > 0.0 {
            let i_plus_1 = (i + 1) as f64;
            let n_minus_i = n - i_plus_1 + 1.0;
            statistic += (2.0 * i_plus_1 - 1.0) * phi.ln() 
                       + (2.0 * n_minus_i - 1.0) * one_minus_phi.ln();
        }
    }
    
    -n - statistic / n
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    
    sign * y
}

/// Efficient correlation calculation
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    
    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;
    
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }
    
    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Efficient covariance calculation
pub fn covariance(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    
    let sum: f64 = x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();
    
    sum / (n - 1.0)
}

/// Robust statistics using median absolute deviation
pub fn robust_statistics(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let median = fast_median(&mut sorted_data);
    
    let mut deviations: Vec<f64> = data.iter()
        .map(|x| (x - median).abs())
        .collect();
    
    let mad = fast_median(&mut deviations) * 1.4826; // Scale factor for consistency with standard deviation
    
    (median, mad)
}

/// Outlier detection using modified Z-score
pub fn detect_outliers(data: &[f64], threshold: f64) -> Vec<usize> {
    let (median, mad) = robust_statistics(data);
    
    if mad == 0.0 {
        return Vec::new();
    }
    
    data.iter()
        .enumerate()
        .filter_map(|(i, &x)| {
            let modified_z_score = 0.6745 * (x - median) / mad;
            if modified_z_score.abs() > threshold {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

/// Efficient hash function for floating point arrays
pub fn hash_float_array(data: &[f64]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    
    for &value in data {
        // Convert to bits for consistent hashing
        value.to_bits().hash(&mut hasher);
    }
    
    hasher.finish()
}

/// Linear interpolation
pub fn linear_interpolation(x: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    if (x1 - x0).abs() < f64::EPSILON {
        return y0;
    }
    
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// Exponential smoothing
pub fn exponential_smoothing(data: &[f64], alpha: f64) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }
    
    let mut smoothed = Vec::with_capacity(data.len());
    smoothed.push(data[0]);
    
    for i in 1..data.len() {
        let new_value = alpha * data[i] + (1.0 - alpha) * smoothed[i - 1];
        smoothed.push(new_value);
    }
    
    smoothed
}

/// Moving average filter
pub fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if data.is_empty() || window == 0 {
        return Vec::new();
    }
    
    let mut result = Vec::new();
    let mut sum = 0.0;
    
    for i in 0..data.len() {
        sum += data[i];
        
        if i >= window {
            sum -= data[i - window];
            result.push(sum / window as f64);
        } else if i == window - 1 {
            result.push(sum / window as f64);
        }
    }
    
    result
}

/// Efficient circular buffer for real-time processing
pub struct CircularBuffer<T> {
    buffer: Vec<T>,
    head: usize,
    tail: usize,
    size: usize,
    capacity: usize,
}

impl<T: Clone + Default> CircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![T::default(); capacity],
            head: 0,
            tail: 0,
            size: 0,
            capacity,
        }
    }
    
    pub fn push(&mut self, item: T) {
        self.buffer[self.tail] = item;
        self.tail = (self.tail + 1) % self.capacity;
        
        if self.size < self.capacity {
            self.size += 1;
        } else {
            self.head = (self.head + 1) % self.capacity;
        }
    }
    
    pub fn len(&self) -> usize {
        self.size
    }
    
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
    
    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }
    
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.size {
            let actual_index = (self.head + index) % self.capacity;
            Some(&self.buffer[actual_index])
        } else {
            None
        }
    }
    
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.size).map(move |i| {
            let idx = (self.head + i) % self.capacity;
            &self.buffer[idx]
        })
    }
}

/// Memory-efficient statistical accumulator
pub struct StatisticalAccumulator {
    count: usize,
    sum: f64,
    sum_squares: f64,
    min: f64,
    max: f64,
}

impl StatisticalAccumulator {
    pub fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_squares: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }
    
    pub fn add(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_squares += value * value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }
    
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }
    
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            let mean = self.mean();
            (self.sum_squares - self.count as f64 * mean * mean) / (self.count - 1) as f64
        }
    }
    
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    
    pub fn min(&self) -> f64 {
        self.min
    }
    
    pub fn max(&self) -> f64 {
        self.max
    }
    
    pub fn count(&self) -> usize {
        self.count
    }
}

impl Default for StatisticalAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_bootstrap_sample() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut rng = rand::thread_rng();
        let sample = bootstrap_sample(&data, &mut rng);
        
        assert_eq!(sample.len(), data.len());
        // All values should be from original data
        for value in sample {
            assert!(data.contains(&value));
        }
    }
    
    #[test]
    fn test_fast_median() {
        let mut data = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let median = fast_median(&mut data);
        assert_relative_eq!(median, 3.0);
        
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let median = fast_median(&mut data);
        assert_relative_eq!(median, 2.5);
    }
    
    #[test]
    fn test_rolling_statistics() {
        let mut rolling = RollingStatistics::new(3);
        
        rolling.add(1.0);
        rolling.add(2.0);
        rolling.add(3.0);
        
        assert_relative_eq!(rolling.mean(), 2.0);
        assert_relative_eq!(rolling.variance(), 1.0);
        assert_relative_eq!(rolling.std_dev(), 1.0);
        
        rolling.add(4.0); // Should remove 1.0
        assert_relative_eq!(rolling.mean(), 3.0);
    }
    
    #[test]
    fn test_autocorrelation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let autocorr = autocorrelation(&data, 1);
        assert!(autocorr >= 0.0 && autocorr <= 1.0);
    }
    
    #[test]
    fn test_quantile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert_relative_eq!(quantile(&data, 0.0), 1.0);
        assert_relative_eq!(quantile(&data, 0.5), 3.0);
        assert_relative_eq!(quantile(&data, 1.0), 5.0);
    }
    
    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = correlation(&x, &y);
        assert_relative_eq!(corr, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_robust_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // With outlier
        let (median, mad) = robust_statistics(&data);
        
        assert_relative_eq!(median, 3.0);
        assert!(mad > 0.0);
    }
    
    #[test]
    fn test_outlier_detection() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let outliers = detect_outliers(&data, 3.5);
        
        assert!(!outliers.is_empty());
        assert!(outliers.contains(&5)); // Index of outlier
    }
    
    #[test]
    fn test_circular_buffer() {
        let mut buffer = CircularBuffer::new(3);
        
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        
        assert_eq!(buffer.len(), 3);
        assert!(buffer.is_full());
        
        buffer.push(4); // Should overwrite 1
        assert_eq!(buffer.get(0), Some(&2));
        assert_eq!(buffer.get(1), Some(&3));
        assert_eq!(buffer.get(2), Some(&4));
    }
    
    #[test]
    fn test_statistical_accumulator() {
        let mut acc = StatisticalAccumulator::new();
        
        acc.add(1.0);
        acc.add(2.0);
        acc.add(3.0);
        
        assert_relative_eq!(acc.mean(), 2.0);
        assert_relative_eq!(acc.variance(), 1.0);
        assert_relative_eq!(acc.min(), 1.0);
        assert_relative_eq!(acc.max(), 3.0);
        assert_eq!(acc.count(), 3);
    }
    
    #[test]
    fn test_exponential_smoothing() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let smoothed = exponential_smoothing(&data, 0.5);
        
        assert_eq!(smoothed.len(), data.len());
        assert_relative_eq!(smoothed[0], 1.0);
        assert!(smoothed[1] > 1.0 && smoothed[1] < 2.0);
    }
    
    #[test]
    fn test_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = moving_average(&data, 3);
        
        assert_eq!(ma.len(), 3);
        assert_relative_eq!(ma[0], 2.0); // (1+2+3)/3
        assert_relative_eq!(ma[1], 3.0); // (2+3+4)/3
        assert_relative_eq!(ma[2], 4.0); // (3+4+5)/3
    }
}