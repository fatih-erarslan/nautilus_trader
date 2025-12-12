//! Utility functions and helpers for swarm intelligence algorithms

pub mod benchmark;
pub mod visualization;
pub mod profiling;

pub use benchmark::*;
pub use visualization::*;
pub use profiling::*;

use anyhow::Result;
use nalgebra::DVector;
use std::time::{Duration, Instant};

/// Common test functions for optimization benchmarking
pub mod test_functions {
    use super::*;
    
    /// Sphere function (unimodal, separable)
    pub fn sphere(x: &DVector<f64>) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }
    
    /// Rosenbrock function (unimodal, non-separable)
    pub fn rosenbrock(x: &DVector<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            let xi = x[i];
            let xi_next = x[i + 1];
            sum += 100.0 * (xi_next - xi * xi).powi(2) + (1.0 - xi).powi(2);
        }
        sum
    }
    
    /// Rastrigin function (multimodal, separable)
    pub fn rastrigin(x: &DVector<f64>) -> f64 {
        let a = 10.0;
        let n = x.len() as f64;
        a * n + x.iter().map(|xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
    }
    
    /// Ackley function (multimodal, non-separable)
    pub fn ackley(x: &DVector<f64>) -> f64 {
        let n = x.len() as f64;
        let sum_sq = x.iter().map(|xi| xi * xi).sum::<f64>();
        let sum_cos = x.iter().map(|xi| (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>();
        
        -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() 
            - (sum_cos / n).exp() 
            + 20.0 
            + std::f64::consts::E
    }
    
    /// Griewank function (multimodal, non-separable)
    pub fn griewank(x: &DVector<f64>) -> f64 {
        let sum_sq = x.iter().map(|xi| xi * xi).sum::<f64>();
        let prod_cos = x.iter().enumerate()
            .map(|(i, xi)| (xi / ((i + 1) as f64).sqrt()).cos())
            .product::<f64>();
        
        1.0 + sum_sq / 4000.0 - prod_cos
    }
}

/// Performance measurement utilities
pub struct PerformanceMeasurement {
    start_time: Instant,
    checkpoints: Vec<(String, Instant)>,
}

impl PerformanceMeasurement {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            checkpoints: Vec::new(),
        }
    }
    
    pub fn checkpoint(&mut self, name: &str) {
        self.checkpoints.push((name.to_string(), Instant::now()));
    }
    
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    pub fn get_checkpoint_durations(&self) -> Vec<(String, Duration)> {
        let mut results = Vec::new();
        let mut last_time = self.start_time;
        
        for (name, time) in &self.checkpoints {
            results.push((name.clone(), time.duration_since(last_time)));
            last_time = *time;
        }
        
        results
    }
}

/// Statistical analysis utilities
pub mod statistics {
    use super::*;
    
    pub fn mean(values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len() as f64
    }
    
    pub fn variance(values: &[f64]) -> f64 {
        let mean_val = mean(values);
        values.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / values.len() as f64
    }
    
    pub fn std_dev(values: &[f64]) -> f64 {
        variance(values).sqrt()
    }
    
    pub fn min_max(values: &[f64]) -> (f64, f64) {
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        (min, max)
    }
    
    pub fn percentile(values: &[f64], p: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_sphere_function() {
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let result = test_functions::sphere(&x);
        assert_relative_eq!(result, 14.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_performance_measurement() {
        let mut perf = PerformanceMeasurement::new();
        
        std::thread::sleep(Duration::from_millis(10));
        perf.checkpoint("step1");
        
        std::thread::sleep(Duration::from_millis(10));
        perf.checkpoint("step2");
        
        let durations = perf.get_checkpoint_durations();
        assert_eq!(durations.len(), 2);
        assert!(durations[0].1.as_millis() >= 8); // Allow some variance
        assert!(durations[1].1.as_millis() >= 8);
    }
    
    #[test]
    fn test_statistics() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert_relative_eq!(statistics::mean(&values), 3.0, epsilon = 1e-10);
        assert_relative_eq!(statistics::variance(&values), 2.0, epsilon = 1e-10);
        assert_relative_eq!(statistics::std_dev(&values), 2.0_f64.sqrt(), epsilon = 1e-10);
        
        let (min, max) = statistics::min_max(&values);
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
        
        assert_eq!(statistics::percentile(&values, 50.0), 3.0);
    }
}