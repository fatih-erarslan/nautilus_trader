//! Utility functions for market analysis

use anyhow::Result;

/// Statistical helper functions
pub struct Statistics;

impl Statistics {
    pub fn percentile(data: &[f64], p: f64) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (p / 100.0 * (sorted_data.len() - 1) as f64).round() as usize;
        sorted_data[index.min(sorted_data.len() - 1)]
    }

    pub fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    pub fn variance(data: &[f64]) -> f64 {
        if data.len() <= 1 {
            return 0.0;
        }
        
        let mean = Self::mean(data);
        let sum_squared_diff = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>();
        
        sum_squared_diff / (data.len() - 1) as f64
    }

    pub fn standard_deviation(data: &[f64]) -> f64 {
        Self::variance(data).sqrt()
    }

    pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        let sum_x2: f64 = x.iter().map(|&xi| xi * xi).sum();
        let sum_y2: f64 = y.iter().map(|&yi| yi * yi).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

/// Time series utility functions
pub struct TimeSeries;

impl TimeSeries {
    pub fn returns(prices: &[f64]) -> Vec<f64> {
        prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    pub fn log_returns(prices: &[f64]) -> Vec<f64> {
        prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect()
    }

    pub fn rolling_window<F, T>(data: &[f64], window_size: usize, mut func: F) -> Vec<T>
    where
        F: FnMut(&[f64]) -> T,
    {
        data.windows(window_size)
            .map(|window| func(window))
            .collect()
    }

    pub fn exponential_moving_average(data: &[f64], alpha: f64) -> Vec<f64> {
        let mut ema = Vec::with_capacity(data.len());
        
        if !data.is_empty() {
            ema.push(data[0]);
            
            for &value in &data[1..] {
                let prev_ema = ema.last().unwrap();
                ema.push(alpha * value + (1.0 - alpha) * prev_ema);
            }
        }
        
        ema
    }
}

/// Mathematical utility functions
pub struct Math;

impl Math {
    pub fn normalize(data: &[f64]) -> Vec<f64> {
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if max_val == min_val {
            return vec![0.5; data.len()];
        }
        
        data.iter()
            .map(|&x| (x - min_val) / (max_val - min_val))
            .collect()
    }

    pub fn z_score(data: &[f64]) -> Vec<f64> {
        let mean = Statistics::mean(data);
        let std = Statistics::standard_deviation(data);
        
        if std == 0.0 {
            return vec![0.0; data.len()];
        }
        
        data.iter()
            .map(|&x| (x - mean) / std)
            .collect()
    }

    pub fn linear_regression_slope(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        let sum_x2: f64 = x.iter().map(|&xi| xi * xi).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denominator
        }
    }

    pub fn fibonacci_sequence(n: usize) -> Vec<f64> {
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![0.0];
        }
        
        let mut fib = vec![0.0, 1.0];
        for i in 2..n {
            let next = fib[i-1] + fib[i-2];
            fib.push(next);
        }
        
        fib
    }

    pub fn golden_ratio() -> f64 {
        1.618033988749
    }
}

/// Performance monitoring utilities
pub struct Performance;

impl Performance {
    pub fn measure_execution_time<F, T>(mut func: F) -> (T, std::time::Duration)
    where
        F: FnMut() -> T,
    {
        let start = std::time::Instant::now();
        let result = func();
        let duration = start.elapsed();
        (result, duration)
    }

    pub fn benchmark<F>(name: &str, iterations: usize, mut func: F)
    where
        F: FnMut(),
    {
        let start = std::time::Instant::now();
        
        for _ in 0..iterations {
            func();
        }
        
        let total_time = start.elapsed();
        let avg_time = total_time / iterations as u32;
        
        println!("Benchmark {}: {} iterations, avg: {:?}", name, iterations, avg_time);
    }
}

/// Data validation utilities
pub struct Validation;

impl Validation {
    pub fn is_valid_price_data(prices: &[f64]) -> bool {
        !prices.is_empty() && 
        prices.iter().all(|&p| p > 0.0 && p.is_finite())
    }

    pub fn is_valid_volume_data(volumes: &[f64]) -> bool {
        !volumes.is_empty() && 
        volumes.iter().all(|&v| v >= 0.0 && v.is_finite())
    }

    pub fn remove_outliers(data: &[f64], threshold: f64) -> Vec<f64> {
        let mean = Statistics::mean(data);
        let std = Statistics::standard_deviation(data);
        
        data.iter()
            .filter(|&&x| (x - mean).abs() <= threshold * std)
            .copied()
            .collect()
    }

    pub fn interpolate_missing_values(data: &[Option<f64>]) -> Vec<f64> {
        let mut result = Vec::with_capacity(data.len());
        let mut last_valid = 0.0;
        
        for (i, &value) in data.iter().enumerate() {
            match value {
                Some(v) => {
                    result.push(v);
                    last_valid = v;
                }
                None => {
                    // Simple forward fill
                    if i == 0 {
                        // Find first valid value
                        let first_valid = data.iter()
                            .find_map(|&x| x)
                            .unwrap_or(0.0);
                        result.push(first_valid);
                        last_valid = first_valid;
                    } else {
                        result.push(last_valid);
                    }
                }
            }
        }
        
        result
    }
}

/// Error handling utilities
pub mod error {
    use thiserror::Error;

    #[derive(Error, Debug)]
    pub enum AnalysisError {
        #[error("Insufficient data: need at least {required} points, got {actual}")]
        InsufficientData { required: usize, actual: usize },
        
        #[error("Invalid parameter: {parameter} = {value}")]
        InvalidParameter { parameter: String, value: String },
        
        #[error("Calculation error: {message}")]
        CalculationError { message: String },
        
        #[error("Data validation error: {message}")]
        ValidationError { message: String },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert_eq!(Statistics::mean(&data), 3.0);
        assert!((Statistics::standard_deviation(&data) - 1.5811388300841898).abs() < 1e-10);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = Statistics::correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_returns() {
        let prices = vec![100.0, 110.0, 105.0, 115.0];
        let returns = TimeSeries::returns(&prices);
        
        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.1).abs() < 1e-10);
    }
}