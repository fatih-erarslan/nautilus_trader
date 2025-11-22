use crate::prelude::*;

/// Mathematical utilities for the autopoiesis framework
#[derive(Debug, Clone)]
pub struct MathUtils;

impl MathUtils {
    /// Calculate exponential moving average
    pub fn ema(values: &[f64], alpha: f64) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }
        
        let mut result = Vec::with_capacity(values.len());
        result.push(values[0]);
        
        for i in 1..values.len() {
            let ema_value = alpha * values[i] + (1.0 - alpha) * result[i - 1];
            result.push(ema_value);
        }
        
        result
    }
    
    /// Calculate simple moving average
    pub fn sma(values: &[f64], window: usize) -> Vec<f64> {
        if values.len() < window {
            return Vec::new();
        }
        
        let mut result = Vec::new();
        for i in window - 1..values.len() {
            let sum: f64 = values[i - window + 1..=i].iter().sum();
            result.push(sum / window as f64);
        }
        
        result
    }
    
    /// Calculate standard deviation
    pub fn std_dev(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        
        variance.sqrt()
    }
    
    /// Calculate correlation between two series
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
            let diff_x = x[i] - mean_x;
            let diff_y = y[i] - mean_y;
            
            numerator += diff_x * diff_y;
            sum_sq_x += diff_x * diff_x;
            sum_sq_y += diff_y * diff_y;
        }
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
    
    /// Calculate linear regression slope and intercept
    pub fn linear_regression(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
        if x.len() != y.len() || x.len() < 2 {
            return None;
        }
        
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_x_sq: f64 = x.iter().map(|xi| xi * xi).sum();
        
        let denominator = n * sum_x_sq - sum_x * sum_x;
        if denominator == 0.0 {
            return None;
        }
        
        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;
        
        Some((slope, intercept))
    }
    
    /// Calculate percentile
    pub fn percentile(values: &[f64], p: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = p * (sorted.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        
        if lower == upper {
            sorted[lower]
        } else {
            let weight = index - lower as f64;
            sorted[lower] * (1.0 - weight) + sorted[upper] * weight
        }
    }
    
    /// Calculate Z-score normalization
    pub fn z_score(values: &[f64]) -> Vec<f64> {
        if values.len() < 2 {
            return values.to_vec();
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std_dev = Self::std_dev(values);
        
        if std_dev == 0.0 {
            return vec![0.0; values.len()];
        }
        
        values.iter()
            .map(|v| (v - mean) / std_dev)
            .collect()
    }
    
    /// Calculate compound annual growth rate
    pub fn cagr(initial_value: f64, final_value: f64, years: f64) -> f64 {
        if initial_value <= 0.0 || final_value <= 0.0 || years <= 0.0 {
            return 0.0;
        }
        
        (final_value / initial_value).powf(1.0 / years) - 1.0
    }
    
    /// Calculate maximum drawdown
    pub fn max_drawdown(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut peak = values[0];
        let mut max_dd = 0.0;
        
        for &value in values.iter().skip(1) {
            if value > peak {
                peak = value;
            }
            
            let drawdown = (peak - value) / peak;
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }
        
        max_dd
    }
}