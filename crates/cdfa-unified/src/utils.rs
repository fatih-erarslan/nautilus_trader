//! Utility functions for the unified CDFA library

use crate::error::{CdfaError, Result};
use crate::types::*;
use ndarray::{Array1, ArrayView1, ArrayBase, ArrayView2, Data};
use std::time::Instant;

/// Performance timer for benchmarking operations
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Start a new timer
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }
    
    /// Get elapsed time in microseconds
    pub fn elapsed_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }
    
    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

/// Memory usage tracker
pub struct MemoryTracker {
    initial_memory: usize,
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new() -> Self {
        Self {
            initial_memory: get_memory_usage(),
        }
    }
    
    /// Get memory used since creation
    pub fn memory_used(&self) -> usize {
        get_memory_usage().saturating_sub(self.initial_memory)
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current memory usage (approximation)
fn get_memory_usage() -> usize {
    // This is a simplified approximation
    // In production, you might want to use a more sophisticated memory tracking method
    0 // Placeholder
}

/// Array validation utilities
pub mod validation {
    use crate::error::{CdfaError, Result};
    use crate::types::Float;
    
    /// Validate that array is not empty
    pub fn validate_not_empty<D>(array: &ndarray::ArrayBase<D, ndarray::Ix1>) -> Result<()>
    where
        D: ndarray::Data<Elem = Float>,
    {
        if array.is_empty() {
            return Err(CdfaError::invalid_input("Array cannot be empty"));
        }
        Ok(())
    }
    
    /// Validate that arrays have the same length
    pub fn validate_same_length<D1, D2>(
        array1: &ndarray::ArrayBase<D1, ndarray::Ix1>,
        array2: &ndarray::ArrayBase<D2, ndarray::Ix1>,
    ) -> Result<()>
    where
        D1: ndarray::Data<Elem = Float>,
        D2: ndarray::Data<Elem = Float>,
    {
        if array1.len() != array2.len() {
            return Err(CdfaError::dimension_mismatch(array1.len(), array2.len()));
        }
        Ok(())
    }
    
    /// Validate that all values are finite
    pub fn validate_finite<D>(array: &ndarray::ArrayBase<D, ndarray::Ix1>) -> Result<()>
    where
        D: ndarray::Data<Elem = Float>,
    {
        for &value in array.iter() {
            if !value.is_finite() {
                return Err(CdfaError::invalid_input(format!(
                    "Array contains non-finite value: {}", value
                )));
            }
        }
        Ok(())
    }
    
    /// Validate that array has minimum length
    pub fn validate_min_length<D>(
        array: &ndarray::ArrayBase<D, ndarray::Ix1>,
        min_length: usize,
    ) -> Result<()>
    where
        D: ndarray::Data<Elem = Float>,
    {
        if array.len() < min_length {
            return Err(CdfaError::invalid_input(format!(
                "Array length {} is less than required minimum {}",
                array.len(),
                min_length
            )));
        }
        Ok(())
    }
    
    /// Validate that values are within a range
    pub fn validate_range<D>(
        array: &ndarray::ArrayBase<D, ndarray::Ix1>,
        min_val: Float,
        max_val: Float,
    ) -> Result<()>
    where
        D: ndarray::Data<Elem = Float>,
    {
        for &value in array.iter() {
            if value < min_val || value > max_val {
                return Err(CdfaError::invalid_input(format!(
                    "Value {} is outside valid range [{}, {}]",
                    value, min_val, max_val
                )));
            }
        }
        Ok(())
    }
}

/// Statistical utilities
pub mod stats {
    use crate::error::{CdfaError, Result};
    use crate::types::{Float, ArrayView1};
    use approx::AbsDiffEq;
    
    /// Calculate mean of an array
    pub fn mean(array: &ArrayView1<Float>) -> Result<Float> {
        super::validation::validate_not_empty(array)?;
        Ok(array.mean().unwrap())
    }
    
    /// Calculate standard deviation
    pub fn std(array: &ArrayView1<Float>) -> Result<Float> {
        super::validation::validate_not_empty(array)?;
        Ok(array.std(0.0))
    }
    
    /// Calculate variance
    pub fn variance(array: &ArrayView1<Float>) -> Result<Float> {
        super::validation::validate_not_empty(array)?;
        Ok(array.var(0.0))
    }
    
    /// Calculate median
    pub fn median(array: &ArrayView1<Float>) -> Result<Float> {
        super::validation::validate_not_empty(array)?;
        let mut sorted = array.to_owned();
        sorted.as_slice_mut().unwrap().sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let len = sorted.len();
        if len % 2 == 0 {
            Ok((sorted[len / 2 - 1] + sorted[len / 2]) / 2.0)
        } else {
            Ok(sorted[len / 2])
        }
    }
    
    /// Calculate quantile
    pub fn quantile(array: &ArrayView1<Float>, q: Float) -> Result<Float> {
        super::validation::validate_not_empty(array)?;
        if !(0.0..=1.0).contains(&q) {
            return Err(CdfaError::invalid_input("Quantile must be between 0 and 1"));
        }
        
        let mut sorted = array.to_owned();
        sorted.as_slice_mut().unwrap().sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = q * (sorted.len() - 1) as Float;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        
        if lower == upper {
            Ok(sorted[lower])
        } else {
            let weight = index - lower as Float;
            Ok(sorted[lower] * (1.0 - weight) + sorted[upper] * weight)
        }
    }
    
    /// Calculate skewness
    pub fn skewness(array: &ArrayView1<Float>) -> Result<Float> {
        super::validation::validate_not_empty(array)?;
        let n = array.len() as Float;
        if n < 3.0 {
            return Err(CdfaError::invalid_input("Need at least 3 points for skewness"));
        }
        
        let mean_val = mean(array)?;
        let std_val = std(array)?;
        
        if std_val.abs_diff_eq(&0.0, Float::EPSILON) {
            return Ok(0.0);
        }
        
        let mut sum = 0.0;
        for &x in array.iter() {
            sum += ((x - mean_val) / std_val).powi(3);
        }
        
        Ok(sum / n)
    }
    
    /// Calculate kurtosis
    pub fn kurtosis(array: &ArrayView1<Float>) -> Result<Float> {
        super::validation::validate_not_empty(array)?;
        let n = array.len() as Float;
        if n < 4.0 {
            return Err(CdfaError::invalid_input("Need at least 4 points for kurtosis"));
        }
        
        let mean_val = mean(array)?;
        let std_val = std(array)?;
        
        if std_val.abs_diff_eq(&0.0, Float::EPSILON) {
            return Ok(0.0);
        }
        
        let mut sum = 0.0;
        for &x in array.iter() {
            sum += ((x - mean_val) / std_val).powi(4);
        }
        
        Ok(sum / n - 3.0) // Excess kurtosis
    }
}

/// Numerical utilities
pub mod numerical {
    use crate::error::{CdfaError, Result};
    use crate::types::{Float, Array1, ArrayView1};
    
    /// Check if a value is approximately zero
    pub fn is_zero(value: Float, tolerance: Float) -> bool {
        value.abs() < tolerance
    }
    
    /// Check if two values are approximately equal
    pub fn is_equal(a: Float, b: Float, tolerance: Float) -> bool {
        (a - b).abs() < tolerance
    }
    
    /// Safe division with tolerance check
    pub fn safe_divide(numerator: Float, denominator: Float, tolerance: Float) -> Result<Float> {
        if is_zero(denominator, tolerance) {
            Err(CdfaError::numerical_error("Division by zero"))
        } else {
            Ok(numerator / denominator)
        }
    }
    
    /// Safe logarithm
    pub fn safe_log(value: Float) -> Result<Float> {
        if value <= 0.0 {
            Err(CdfaError::numerical_error(format!(
                "Logarithm of non-positive value: {}", value
            )))
        } else {
            Ok(value.ln())
        }
    }
    
    /// Safe square root
    pub fn safe_sqrt(value: Float) -> Result<Float> {
        if value < 0.0 {
            Err(CdfaError::numerical_error(format!(
                "Square root of negative value: {}", value
            )))
        } else {
            Ok(value.sqrt())
        }
    }
    
    /// Clamp value to range
    pub fn clamp(value: Float, min_val: Float, max_val: Float) -> Float {
        value.max(min_val).min(max_val)
    }
    
    /// Linear interpolation
    pub fn lerp(a: Float, b: Float, t: Float) -> Float {
        a + t * (b - a)
    }
    
    /// Sigmoid function
    pub fn sigmoid(x: Float) -> Float {
        1.0 / (1.0 + (-x).exp())
    }
    
    /// Softmax function for array
    pub fn softmax(array: &ArrayView1<Float>) -> Result<Array1<Float>> {
        super::validation::validate_not_empty(array)?;
        
        // Subtract max for numerical stability
        let max_val = array.fold(Float::NEG_INFINITY, |acc, &x| acc.max(x));
        let shifted: Array1<Float> = array.mapv(|x| x - max_val);
        let exp_vals: Array1<Float> = shifted.mapv(|x| x.exp());
        let sum_exp = exp_vals.sum();
        
        if is_zero(sum_exp, Float::EPSILON) {
            return Err(CdfaError::numerical_error("Softmax normalization failed"));
        }
        
        Ok(exp_vals / sum_exp)
    }
}

/// Array manipulation utilities
pub mod arrays {
    use crate::error::{CdfaError, Result};
    use crate::types::{Float, Array1, ArrayView1};
    use super::{stats, numerical};
    
    /// Normalize array to zero mean and unit variance
    pub fn normalize(array: &ArrayView1<Float>) -> Result<Array1<Float>> {
        super::validation::validate_not_empty(array)?;
        
        let mean_val = stats::mean(array)?;
        let std_val = stats::std(array)?;
        
        if numerical::is_zero(std_val, Float::EPSILON) {
            Ok(Array1::zeros(array.len()))
        } else {
            Ok((array - mean_val) / std_val)
        }
    }
    
    /// Standardize array to [0, 1] range
    pub fn standardize(array: &ArrayView1<Float>) -> Result<Array1<Float>> {
        super::validation::validate_not_empty(array)?;
        
        let min_val = array.fold(Float::INFINITY, |acc, &x| acc.min(x));
        let max_val = array.fold(Float::NEG_INFINITY, |acc, &x| acc.max(x));
        
        let range = max_val - min_val;
        if numerical::is_zero(range, Float::EPSILON) {
            Ok(Array1::zeros(array.len()))
        } else {
            Ok((array - min_val) / range)
        }
    }
    
    /// Remove outliers using IQR method
    pub fn remove_outliers(array: &ArrayView1<Float>, factor: Float) -> Result<Array1<Float>> {
        super::validation::validate_not_empty(array)?;
        
        let q1 = stats::quantile(array, 0.25)?;
        let q3 = stats::quantile(array, 0.75)?;
        let iqr = q3 - q1;
        let lower_bound = q1 - factor * iqr;
        let upper_bound = q3 + factor * iqr;
        
        let filtered: Array1<Float> = array
            .iter()
            .copied()
            .filter(|&x| x >= lower_bound && x <= upper_bound)
            .collect();
        
        if filtered.is_empty() {
            Err(CdfaError::invalid_input("All values were filtered out as outliers"))
        } else {
            Ok(filtered)
        }
    }
    
    /// Moving average
    pub fn moving_average(array: &ArrayView1<Float>, window: usize) -> Result<Array1<Float>> {
        super::validation::validate_not_empty(array)?;
        super::validation::validate_min_length(array, window)?;
        
        let mut result = Array1::zeros(array.len() - window + 1);
        for i in 0..result.len() {
            let window_slice = array.slice(ndarray::s![i..i + window]);
            result[i] = window_slice.mean().unwrap();
        }
        
        Ok(result)
    }
    
    /// Exponential moving average
    pub fn exponential_moving_average(array: &ArrayView1<Float>, alpha: Float) -> Result<Array1<Float>> {
        super::validation::validate_not_empty(array)?;
        super::validation::validate_range(&Array1::from_elem(1, alpha).view(), 0.0, 1.0)?;
        
        let mut result = Array1::zeros(array.len());
        result[0] = array[0];
        
        for i in 1..array.len() {
            result[i] = alpha * array[i] + (1.0 - alpha) * result[i - 1];
        }
        
        Ok(result)
    }
    
    /// Difference array (first differences)
    pub fn diff(array: &ArrayView1<Float>) -> Result<Array1<Float>> {
        super::validation::validate_min_length(array, 2)?;
        
        let mut result = Array1::zeros(array.len() - 1);
        for i in 0..result.len() {
            result[i] = array[i + 1] - array[i];
        }
        
        Ok(result)
    }
    
    /// Cumulative sum
    pub fn cumsum(array: &ArrayView1<Float>) -> Result<Array1<Float>> {
        super::validation::validate_not_empty(array)?;
        
        let mut result = Array1::zeros(array.len());
        result[0] = array[0];
        
        for i in 1..array.len() {
            result[i] = result[i - 1] + array[i];
        }
        
        Ok(result)
    }
}

/// Parallel processing utilities
#[cfg(feature = "parallel")]
pub mod parallel {
    // Test imports available in scope
    
    /// Get optimal number of threads
    pub fn get_optimal_threads() -> usize {
        num_cpus::get().max(1)
    }
    
    /// Check if parallel processing is beneficial
    pub fn should_use_parallel(data_size: usize, threshold: usize) -> bool {
        data_size >= threshold && get_optimal_threads() > 1
    }
}

/// SIMD utilities
#[cfg(feature = "simd")]
pub mod simd {
    // Test imports available in scope
    
    /// Check if SIMD operations are supported
    pub fn is_simd_supported() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
    
    /// Get SIMD lane width for the current platform
    pub fn get_simd_width() -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                8 // AVX-512 can process 8 f64 values
            } else if is_x86_feature_detected!("avx2") {
                4 // AVX2 can process 4 f64 values
            } else {
                2 // SSE can process 2 f64 values
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            1 // Fallback to scalar processing
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_timer() {
        let timer = Timer::start();
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(timer.elapsed_us() > 0);
        assert!(timer.elapsed_ms() >= 0);
    }
    
    #[test]
    fn test_validation() {
        let empty_array = array![];
        assert!(validation::validate_not_empty(&empty_array.view()).is_err());
        
        let valid_array = array![1.0, 2.0, 3.0];
        assert!(validation::validate_not_empty(&valid_array.view()).is_ok());
        
        let finite_array = array![1.0, 2.0, f64::INFINITY];
        assert!(validation::validate_finite(&finite_array.view()).is_err());
    }
    
    #[test]
    fn test_stats() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert_abs_diff_eq!(stats::mean(&data.view()).unwrap(), 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats::median(&data.view()).unwrap(), 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats::quantile(&data.view(), 0.5).unwrap(), 3.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_numerical() {
        assert!(numerical::is_zero(1e-15, 1e-10));
        assert!(!numerical::is_zero(1e-5, 1e-10));
        
        assert!(numerical::is_equal(1.0, 1.0 + 1e-15, 1e-10));
        
        assert!(numerical::safe_divide(10.0, 0.0, 1e-10).is_err());
        assert_abs_diff_eq!(numerical::safe_divide(10.0, 2.0, 1e-10).unwrap(), 5.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_arrays() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let normalized = arrays::normalize(&data.view()).unwrap();
        assert_abs_diff_eq!(stats::mean(&normalized.view()).unwrap(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats::std(&normalized.view()).unwrap(), 1.0, epsilon = 1e-10);
        
        let standardized = arrays::standardize(&data.view()).unwrap();
        assert_abs_diff_eq!(standardized[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(standardized[4], 1.0, epsilon = 1e-10);
        
        let ma = arrays::moving_average(&data.view(), 3).unwrap();
        assert_eq!(ma.len(), 3);
        assert_abs_diff_eq!(ma[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ma[1], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ma[2], 4.0, epsilon = 1e-10);
    }
    
    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_utils() {
        assert!(parallel::get_optimal_threads() >= 1);
        assert!(parallel::should_use_parallel(10000, 1000));
        assert!(!parallel::should_use_parallel(100, 1000));
    }
}