//! Utility functions and helper modules for market analysis

use crate::error::{AnalysisError, Result};
use ndarray::{Array1, Array2};
use statrs::statistics::Statistics;
use chrono::{DateTime, Utc, Duration};

/// Statistical utility functions
pub mod statistical {
    use super::*;
    
    /// Calculate rolling statistics for a time series
    pub fn rolling_statistics(data: &[f64], window: usize) -> Result<RollingStats> {
        if data.len() < window {
            return Err(AnalysisError::insufficient_data("Not enough data for rolling window"));
        }
        
        let mut means = Vec::new();
        let mut stds = Vec::new();
        let mut mins = Vec::new();
        let mut maxs = Vec::new();
        
        for i in window..=data.len() {
            let window_data = &data[i-window..i];
            means.push(window_data.mean());
            stds.push(window_data.std_dev());
            mins.push(window_data.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
            maxs.push(window_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        }
        
        Ok(RollingStats { means, stds, mins, maxs })
    }
    
    /// Calculate correlation between two time series
    pub fn correlation(x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return Err(AnalysisError::calculation_error("Invalid data for correlation"));
        }
        
        let x_mean = x.mean();
        let y_mean = y.mean();
        
        let numerator: f64 = x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
            .sum();
            
        let x_sq_sum: f64 = x.iter().map(|xi| (xi - x_mean).powi(2)).sum();
        let y_sq_sum: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        
        let denominator = (x_sq_sum * y_sq_sum).sqrt();
        
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
    
    /// Calculate autocorrelation for specified lags
    pub fn autocorrelation(data: &[f64], max_lag: usize) -> Result<Vec<f64>> {
        if data.len() <= max_lag {
            return Err(AnalysisError::insufficient_data("Not enough data for autocorrelation"));
        }
        
        let mut autocorrs = Vec::new();
        
        for lag in 1..=max_lag {
            let x = &data[..data.len()-lag];
            let y = &data[lag..];
            let corr = correlation(x, y)?;
            autocorrs.push(corr);
        }
        
        Ok(autocorrs)
    }
    
    /// Calculate skewness of a distribution
    pub fn skewness(data: &[f64]) -> Result<f64> {
        if data.len() < 3 {
            return Err(AnalysisError::insufficient_data("Need at least 3 data points for skewness"));
        }
        
        let mean = data.mean();
        let std_dev = data.std_dev();
        
        if std_dev == 0.0 {
            return Ok(0.0);
        }
        
        let n = data.len() as f64;
        let skew_sum: f64 = data.iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum();
            
        Ok((n / ((n - 1.0) * (n - 2.0))) * skew_sum)
    }
    
    /// Calculate kurtosis of a distribution
    pub fn kurtosis(data: &[f64]) -> Result<f64> {
        if data.len() < 4 {
            return Err(AnalysisError::insufficient_data("Need at least 4 data points for kurtosis"));
        }
        
        let mean = data.mean();
        let std_dev = data.std_dev();
        
        if std_dev == 0.0 {
            return Ok(0.0);
        }
        
        let n = data.len() as f64;
        let kurt_sum: f64 = data.iter()
            .map(|&x| ((x - mean) / std_dev).powi(4))
            .sum();
            
        let kurtosis = (n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0)) * kurt_sum
            - 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0));
            
        Ok(kurtosis)
    }
    
    /// Detect outliers using IQR method
    pub fn detect_outliers_iqr(data: &[f64]) -> Result<Vec<bool>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let q1_idx = sorted_data.len() / 4;
        let q3_idx = 3 * sorted_data.len() / 4;
        
        let q1 = sorted_data[q1_idx];
        let q3 = sorted_data[q3_idx];
        let iqr = q3 - q1;
        
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;
        
        Ok(data.iter().map(|&x| x < lower_bound || x > upper_bound).collect())
    }
    
    /// Calculate Hurst exponent for long memory detection
    pub fn hurst_exponent(data: &[f64]) -> Result<f64> {
        if data.len() < 10 {
            return Err(AnalysisError::insufficient_data("Need at least 10 data points for Hurst exponent"));
        }
        
        let n = data.len();
        let mean = data.mean();
        
        // Calculate cumulative deviations
        let mut cumsum = Vec::with_capacity(n);
        let mut sum = 0.0;
        for &value in data {
            sum += value - mean;
            cumsum.push(sum);
        }
        
        // Calculate range
        let max_cumsum = cumsum.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_cumsum = cumsum.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let range = max_cumsum - min_cumsum;
        
        // Calculate standard deviation
        let std_dev = data.std_dev();
        
        if std_dev == 0.0 || range == 0.0 {
            return Ok(0.5); // Random walk
        }
        
        // R/S statistic
        let rs = range / std_dev;
        
        // Hurst exponent approximation
        let hurst = rs.ln() / (n as f64).ln();
        
        Ok(hurst.clamp(0.0, 1.0))
    }
    
    #[derive(Debug, Clone)]
    pub struct RollingStats {
        pub means: Vec<f64>,
        pub stds: Vec<f64>,
        pub mins: Vec<f64>,
        pub maxs: Vec<f64>,
    }
}

/// Machine learning utility functions
pub mod ml {
    use super::*;
    
    /// Normalize data to [0, 1] range
    pub fn min_max_normalize(data: &[f64]) -> Result<(Vec<f64>, f64, f64)> {
        if data.is_empty() {
            return Err(AnalysisError::insufficient_data("Empty data for normalization"));
        }
        
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if min_val == max_val {
            return Ok((vec![0.5; data.len()], min_val, max_val));
        }
        
        let normalized: Vec<f64> = data.iter()
            .map(|&x| (x - min_val) / (max_val - min_val))
            .collect();
            
        Ok((normalized, min_val, max_val))
    }
    
    /// Z-score normalization
    pub fn z_score_normalize(data: &[f64]) -> Result<(Vec<f64>, f64, f64)> {
        if data.len() < 2 {
            return Err(AnalysisError::insufficient_data("Need at least 2 data points for z-score"));
        }
        
        let mean = data.mean();
        let std_dev = data.std_dev();
        
        if std_dev == 0.0 {
            return Ok((vec![0.0; data.len()], mean, std_dev));
        }
        
        let normalized: Vec<f64> = data.iter()
            .map(|&x| (x - mean) / std_dev)
            .collect();
            
        Ok((normalized, mean, std_dev))
    }
    
    /// Split data into train/validation/test sets
    pub fn train_test_split<T: Clone>(
        data: &[T], 
        train_ratio: f64, 
        val_ratio: f64
    ) -> Result<(Vec<T>, Vec<T>, Vec<T>)> {
        if train_ratio + val_ratio >= 1.0 || train_ratio <= 0.0 || val_ratio < 0.0 {
            return Err(AnalysisError::invalid_config("Invalid split ratios"));
        }
        
        let n = data.len();
        let train_size = (n as f64 * train_ratio) as usize;
        let val_size = (n as f64 * val_ratio) as usize;
        
        let train_data = data[..train_size].to_vec();
        let val_data = data[train_size..train_size + val_size].to_vec();
        let test_data = data[train_size + val_size..].to_vec();
        
        Ok((train_data, val_data, test_data))
    }
    
    /// Create sliding windows for time series data
    pub fn create_sliding_windows<T: Clone>(
        data: &[T], 
        window_size: usize, 
        step_size: usize
    ) -> Result<Vec<Vec<T>>> {
        if window_size == 0 || step_size == 0 {
            return Err(AnalysisError::invalid_config("Window and step size must be positive"));
        }
        
        if data.len() < window_size {
            return Err(AnalysisError::insufficient_data("Data shorter than window size"));
        }
        
        let mut windows = Vec::new();
        let mut start = 0;
        
        while start + window_size <= data.len() {
            windows.push(data[start..start + window_size].to_vec());
            start += step_size;
        }
        
        Ok(windows)
    }
    
    /// Calculate feature importance using permutation method
    pub fn permutation_feature_importance(
        features: &Array2<f64>,
        targets: &Array1<f64>,
        predict_fn: impl Fn(&Array2<f64>) -> Result<Array1<f64>>
    ) -> Result<Vec<f64>> {
        let baseline_predictions = predict_fn(features)?;
        let baseline_score = calculate_mse(&baseline_predictions, targets)?;
        
        let mut importances = Vec::new();
        
        for feature_idx in 0..features.ncols() {
            let mut permuted_features = features.clone();
            
            // Permute the feature column
            let mut feature_column = permuted_features.column_mut(feature_idx);
            let mut values: Vec<f64> = feature_column.iter().cloned().collect();
            
            // Simple shuffle (in practice, use a proper RNG)
            for i in (1..values.len()).rev() {
                let j = i % (i + 1); // Simplified shuffle
                values.swap(i, j);
            }
            
            for (i, &value) in values.iter().enumerate() {
                feature_column[i] = value;
            }
            
            let permuted_predictions = predict_fn(&permuted_features)?;
            let permuted_score = calculate_mse(&permuted_predictions, targets)?;
            
            let importance = permuted_score - baseline_score;
            importances.push(importance);
        }
        
        Ok(importances)
    }
    
    /// Calculate mean squared error
    pub fn calculate_mse(predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(AnalysisError::calculation_error("Prediction and target lengths don't match"));
        }
        
        let mse = predictions.iter()
            .zip(targets.iter())
            .map(|(pred, target)| (pred - target).powi(2))
            .sum::<f64>() / predictions.len() as f64;
            
        Ok(mse)
    }
    
    /// Calculate mean absolute error
    pub fn calculate_mae(predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(AnalysisError::calculation_error("Prediction and target lengths don't match"));
        }
        
        let mae = predictions.iter()
            .zip(targets.iter())
            .map(|(pred, target)| (pred - target).abs())
            .sum::<f64>() / predictions.len() as f64;
            
        Ok(mae)
    }
    
    /// Calculate directional accuracy
    pub fn directional_accuracy(predictions: &[f64], targets: &[f64]) -> Result<f64> {
        if predictions.len() != targets.len() || predictions.len() < 2 {
            return Err(AnalysisError::calculation_error("Invalid data for directional accuracy"));
        }
        
        let mut correct_directions = 0;
        let mut total_comparisons = 0;
        
        for i in 1..predictions.len() {
            let pred_direction = predictions[i] > predictions[i-1];
            let actual_direction = targets[i] > targets[i-1];
            
            if pred_direction == actual_direction {
                correct_directions += 1;
            }
            total_comparisons += 1;
        }
        
        Ok(correct_directions as f64 / total_comparisons as f64)
    }
}

/// SIMD optimization utilities
pub mod simd {
    use super::*;
    
    #[cfg(feature = "simd")]
    use wide::f64x4;
    
    /// SIMD-accelerated moving average calculation
    pub fn simd_moving_average(data: &[f64], window: usize) -> Result<Vec<f64>> {
        if data.len() < window {
            return Err(AnalysisError::insufficient_data("Insufficient data for moving average"));
        }
        
        #[cfg(feature = "simd")]
        {
            simd_moving_average_impl(data, window)
        }
        
        #[cfg(not(feature = "simd"))]
        {
            fallback_moving_average(data, window)
        }
    }
    
    #[cfg(feature = "simd")]
    fn simd_moving_average_impl(data: &[f64], window: usize) -> Result<Vec<f64>> {
        let mut result = Vec::with_capacity(data.len() - window + 1);
        
        for i in 0..=data.len()-window {
            let window_data = &data[i..i+window];
            let sum = simd_sum(window_data)?;
            result.push(sum / window as f64);
        }
        
        Ok(result)
    }
    
    #[cfg(feature = "simd")]
    fn simd_sum(data: &[f64]) -> Result<f64> {
        let mut sum = f64x4::ZERO;
        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let vector = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
            sum = sum + vector;
        }
        
        let mut total = sum.reduce_add();
        
        for &value in remainder {
            total += value;
        }
        
        Ok(total)
    }
    
    fn fallback_moving_average(data: &[f64], window: usize) -> Result<Vec<f64>> {
        let mut result = Vec::with_capacity(data.len() - window + 1);
        
        for i in 0..=data.len()-window {
            let window_sum: f64 = data[i..i+window].iter().sum();
            result.push(window_sum / window as f64);
        }
        
        Ok(result)
    }
    
    /// SIMD-accelerated correlation calculation
    pub fn simd_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return Err(AnalysisError::calculation_error("Invalid data for correlation"));
        }
        
        #[cfg(feature = "simd")]
        {
            simd_correlation_impl(x, y)
        }
        
        #[cfg(not(feature = "simd"))]
        {
            statistical::correlation(x, y)
        }
    }
    
    #[cfg(feature = "simd")]
    fn simd_correlation_impl(x: &[f64], y: &[f64]) -> Result<f64> {
        let x_mean = simd_mean(x)?;
        let y_mean = simd_mean(y)?;
        
        let numerator = simd_dot_product_centered(x, y, x_mean, y_mean)?;
        let x_var = simd_variance_from_mean(x, x_mean)?;
        let y_var = simd_variance_from_mean(y, y_mean)?;
        
        let denominator = (x_var * y_var).sqrt();
        
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
    
    #[cfg(feature = "simd")]
    fn simd_mean(data: &[f64]) -> Result<f64> {
        let sum = simd_sum(data)?;
        Ok(sum / data.len() as f64)
    }
    
    #[cfg(feature = "simd")]
    fn simd_dot_product_centered(x: &[f64], y: &[f64], x_mean: f64, y_mean: f64) -> Result<f64> {
        let mut result = f64x4::ZERO;
        let x_mean_vec = f64x4::splat(x_mean);
        let y_mean_vec = f64x4::splat(y_mean);
        
        let chunks_x = x.chunks_exact(4);
        let chunks_y = y.chunks_exact(4);
        let remainder_x = chunks_x.remainder();
        let remainder_y = chunks_y.remainder();
        
        for (chunk_x, chunk_y) in chunks_x.zip(chunks_y) {
            let x_vec = f64x4::new([chunk_x[0], chunk_x[1], chunk_x[2], chunk_x[3]]);
            let y_vec = f64x4::new([chunk_y[0], chunk_y[1], chunk_y[2], chunk_y[3]]);
            
            let x_centered = x_vec - x_mean_vec;
            let y_centered = y_vec - y_mean_vec;
            
            result = result + (x_centered * y_centered);
        }
        
        let mut total = result.reduce_add();
        
        for (&x_val, &y_val) in remainder_x.iter().zip(remainder_y.iter()) {
            total += (x_val - x_mean) * (y_val - y_mean);
        }
        
        Ok(total)
    }
    
    #[cfg(feature = "simd")]
    fn simd_variance_from_mean(data: &[f64], mean: f64) -> Result<f64> {
        let mut sum_sq = f64x4::ZERO;
        let mean_vec = f64x4::splat(mean);
        
        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let vector = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let diff = vector - mean_vec;
            sum_sq = sum_sq + (diff * diff);
        }
        
        let mut total = sum_sq.reduce_add();
        
        for &value in remainder {
            total += (value - mean).powi(2);
        }
        
        Ok(total)
    }
}

/// Time series utility functions
pub mod time_series {
    use super::*;
    
    /// Calculate returns from price series
    pub fn calculate_returns(prices: &[f64], return_type: ReturnType) -> Result<Vec<f64>> {
        if prices.len() < 2 {
            return Err(AnalysisError::insufficient_data("Need at least 2 prices for returns"));
        }
        
        let returns = match return_type {
            ReturnType::Simple => {
                prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect()
            }
            ReturnType::Log => {
                prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect()
            }
        };
        
        Ok(returns)
    }
    
    /// Detect structural breaks using CUSUM test
    pub fn cusum_test(data: &[f64], threshold: f64) -> Result<Vec<usize>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        let mean = data.mean();
        let std_dev = data.std_dev();
        
        if std_dev == 0.0 {
            return Ok(Vec::new());
        }
        
        let mut cumsum = 0.0;
        let mut breaks = Vec::new();
        
        for (i, &value) in data.iter().enumerate() {
            cumsum += (value - mean) / std_dev;
            
            if cumsum.abs() > threshold {
                breaks.push(i);
                cumsum = 0.0; // Reset after detecting break
            }
        }
        
        Ok(breaks)
    }
    
    /// Detrend time series using linear regression
    pub fn detrend_linear(data: &[f64]) -> Result<Vec<f64>> {
        if data.len() < 3 {
            return Err(AnalysisError::insufficient_data("Need at least 3 points for detrending"));
        }
        
        let n = data.len() as f64;
        let x_values: Vec<f64> = (0..data.len()).map(|i| i as f64).collect();
        
        // Calculate linear trend
        let sum_x: f64 = x_values.iter().sum();
        let sum_y: f64 = data.iter().sum();
        let sum_xy: f64 = x_values.iter().zip(data.iter()).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = x_values.iter().map(|x| x.powi(2)).sum();
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
        let intercept = (sum_y - slope * sum_x) / n;
        
        // Remove trend
        let detrended: Vec<f64> = x_values.iter()
            .zip(data.iter())
            .map(|(x, y)| y - (slope * x + intercept))
            .collect();
            
        Ok(detrended)
    }
    
    /// Apply exponential smoothing
    pub fn exponential_smoothing(data: &[f64], alpha: f64) -> Result<Vec<f64>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(AnalysisError::invalid_config("Alpha must be in (0, 1]"));
        }
        
        let mut smoothed = Vec::with_capacity(data.len());
        smoothed.push(data[0]);
        
        for &value in &data[1..] {
            let prev_smooth = smoothed.last().unwrap();
            smoothed.push(alpha * value + (1.0 - alpha) * prev_smooth);
        }
        
        Ok(smoothed)
    }
    
    /// Calculate GARCH volatility estimates
    pub fn garch_volatility(returns: &[f64], alpha: f64, beta: f64, omega: f64) -> Result<Vec<f64>> {
        if returns.len() < 2 {
            return Err(AnalysisError::insufficient_data("Need at least 2 returns for GARCH"));
        }
        
        if alpha + beta >= 1.0 {
            return Err(AnalysisError::invalid_config("GARCH parameters must satisfy alpha + beta < 1"));
        }
        
        let mut volatilities = Vec::with_capacity(returns.len());
        
        // Initialize with sample variance
        let initial_var = returns.iter().map(|r| r.powi(2)).sum::<f64>() / returns.len() as f64;
        volatilities.push(initial_var.sqrt());
        
        for i in 1..returns.len() {
            let prev_return_sq = returns[i-1].powi(2);
            let prev_var = volatilities[i-1].powi(2);
            
            let current_var = omega + alpha * prev_return_sq + beta * prev_var;
            volatilities.push(current_var.sqrt());
        }
        
        Ok(volatilities)
    }
    
    #[derive(Debug, Clone, Copy)]
    pub enum ReturnType {
        Simple,
        Log,
    }
}

/// Performance monitoring utilities
pub mod performance {
    use super::*;
    use std::time::{Duration, Instant};
    
    /// Performance timer for measuring execution time
    #[derive(Debug)]
    pub struct PerformanceTimer {
        start_time: Instant,
        operation_name: String,
    }
    
    impl PerformanceTimer {
        pub fn new(operation_name: String) -> Self {
            Self {
                start_time: Instant::now(),
                operation_name,
            }
        }
        
        pub fn elapsed(&self) -> Duration {
            self.start_time.elapsed()
        }
        
        pub fn finish(self) -> (String, Duration) {
            (self.operation_name.clone(), self.elapsed())
        }
    }
    
    /// Memory usage tracker
    #[derive(Debug, Default)]
    pub struct MemoryTracker {
        allocations: usize,
        peak_usage: usize,
        current_usage: usize,
    }
    
    impl MemoryTracker {
        pub fn new() -> Self {
            Self::default()
        }
        
        pub fn allocate(&mut self, size: usize) {
            self.allocations += 1;
            self.current_usage += size;
            self.peak_usage = self.peak_usage.max(self.current_usage);
        }
        
        pub fn deallocate(&mut self, size: usize) {
            self.current_usage = self.current_usage.saturating_sub(size);
        }
        
        pub fn get_stats(&self) -> MemoryStats {
            MemoryStats {
                allocations: self.allocations,
                peak_usage_bytes: self.peak_usage,
                current_usage_bytes: self.current_usage,
            }
        }
    }
    
    #[derive(Debug, Clone)]
    pub struct MemoryStats {
        pub allocations: usize,
        pub peak_usage_bytes: usize,
        pub current_usage_bytes: usize,
    }
}

/// Data validation utilities
pub mod validation {
    use super::*;
    
    /// Validate market data for completeness and consistency
    pub fn validate_market_data(data: &crate::types::MarketData) -> Result<ValidationReport> {
        let mut report = ValidationReport::default();
        
        // Check for empty data
        if data.prices.is_empty() {
            report.errors.push("Empty price data".to_string());
        }
        
        if data.volumes.is_empty() {
            report.warnings.push("Empty volume data".to_string());
        }
        
        // Check for mismatched lengths
        if !data.volumes.is_empty() && data.prices.len() != data.volumes.len() {
            report.errors.push("Price and volume data length mismatch".to_string());
        }
        
        // Check for invalid values
        for (i, &price) in data.prices.iter().enumerate() {
            if price <= 0.0 || !price.is_finite() {
                report.errors.push(format!("Invalid price at index {}: {}", i, price));
            }
        }
        
        for (i, &volume) in data.volumes.iter().enumerate() {
            if volume < 0.0 || !volume.is_finite() {
                report.errors.push(format!("Invalid volume at index {}: {}", i, volume));
            }
        }
        
        // Check for extreme outliers
        if data.prices.len() > 10 {
            let outliers = statistical::detect_outliers_iqr(&data.prices)?;
            let outlier_count = outliers.iter().filter(|&&x| x).count();
            
            if outlier_count > data.prices.len() / 10 {
                report.warnings.push(format!("High number of price outliers: {}", outlier_count));
            }
        }
        
        // Check timestamp consistency
        if data.timestamp > Utc::now() {
            report.warnings.push("Future timestamp detected".to_string());
        }
        
        if data.timestamp < Utc::now() - Duration::days(365) {
            report.warnings.push("Very old timestamp detected".to_string());
        }
        
        Ok(report)
    }
    
    #[derive(Debug, Default)]
    pub struct ValidationReport {
        pub errors: Vec<String>,
        pub warnings: Vec<String>,
    }
    
    impl ValidationReport {
        pub fn is_valid(&self) -> bool {
            self.errors.is_empty()
        }
        
        pub fn has_warnings(&self) -> bool {
            !self.warnings.is_empty()
        }
        
        pub fn summary(&self) -> String {
            format!(
                "Validation Report: {} errors, {} warnings",
                self.errors.len(),
                self.warnings.len()
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rolling_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = statistical::rolling_statistics(&data, 3).unwrap();
        
        assert_eq!(stats.means.len(), 8);
        assert!((stats.means[0] - 2.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = statistical::correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 1e-10); // Perfect positive correlation
    }
    
    #[test]
    fn test_normalization() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (normalized, min_val, max_val) = ml::min_max_normalize(&data).unwrap();
        
        assert_eq!(min_val, 1.0);
        assert_eq!(max_val, 5.0);
        assert!((normalized[0] - 0.0).abs() < 1e-10);
        assert!((normalized[4] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_returns_calculation() {
        let prices = vec![100.0, 102.0, 101.0, 103.0];
        let returns = time_series::calculate_returns(&prices, time_series::ReturnType::Simple).unwrap();
        
        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.02).abs() < 1e-10);
        assert!((returns[1] - (-0.0098039215686275)).abs() < 1e-10);
    }
    
    #[test]
    fn test_outlier_detection() {
        let data = vec![1.0, 2.0, 3.0, 100.0, 4.0, 5.0]; // 100.0 is an outlier
        let outliers = statistical::detect_outliers_iqr(&data).unwrap();
        
        assert!(outliers[3]); // Index 3 (value 100.0) should be detected as outlier
    }
}