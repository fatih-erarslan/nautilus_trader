//! Utility Functions for NHITS
//! Helper functions for data processing, metrics, and visualization

use ndarray::{Array1, Array2, Array3, Axis, s};
use std::f64::consts::PI;

/// Time series preprocessing utilities
pub mod preprocessing {
    use super::*;
    
    /// Normalize time series data
    pub fn normalize(data: &Array3<f64>) -> (Array3<f64>, Vec<(f64, f64)>) {
        let (batch_size, _, features) = data.shape();
        let mut normalized = data.clone();
        let mut stats = Vec::new();
        
        for b in 0..batch_size {
            for f in 0..features {
                let series = data.slice(s![b, .., f]);
                let mean = series.mean().unwrap_or(0.0);
                let std = series.std(0.0).max(1e-8);
                
                normalized.slice_mut(s![b, .., f]).mapv_inplace(|x| (x - mean) / std);
                stats.push((mean, std));
            }
        }
        
        (normalized, stats)
    }
    
    /// Denormalize predictions
    pub fn denormalize(
        data: &Array3<f64>,
        stats: &[(f64, f64)],
    ) -> Array3<f64> {
        let (batch_size, seq_len, features) = data.shape();
        let mut denormalized = data.clone();
        
        for b in 0..batch_size {
            for f in 0..features {
                let idx = b * features + f;
                if idx < stats.len() {
                    let (mean, std) = stats[idx];
                    denormalized.slice_mut(s![b, .., f])
                        .mapv_inplace(|x| x * std + mean);
                }
            }
        }
        
        denormalized
    }
    
    /// Create sliding windows for time series
    pub fn create_windows(
        data: &Array2<f64>,
        window_size: usize,
        horizon: usize,
        stride: usize,
    ) -> (Array3<f64>, Array3<f64>) {
        let (seq_len, features) = data.shape();
        let num_windows = (seq_len - window_size - horizon) / stride + 1;
        
        let mut inputs = Array3::zeros((num_windows, window_size, features));
        let mut targets = Array3::zeros((num_windows, horizon, features));
        
        for i in 0..num_windows {
            let start = i * stride;
            let input_end = start + window_size;
            let target_end = input_end + horizon;
            
            inputs.slice_mut(s![i, .., ..]).assign(&data.slice(s![start..input_end, ..]));
            targets.slice_mut(s![i, .., ..]).assign(&data.slice(s![input_end..target_end, ..]));
        }
        
        (inputs, targets)
    }
    
    /// Apply differencing for stationarity
    pub fn difference(data: &Array3<f64>, order: usize) -> Array3<f64> {
        let mut result = data.clone();
        
        for _ in 0..order {
            let (batch_size, seq_len, features) = result.shape();
            let mut diff = Array3::zeros((batch_size, seq_len - 1, features));
            
            for b in 0..batch_size {
                for t in 1..seq_len {
                    for f in 0..features {
                        diff[[b, t - 1, f]] = result[[b, t, f]] - result[[b, t - 1, f]];
                    }
                }
            }
            
            result = diff;
        }
        
        result
    }
    
    /// Inverse differencing
    pub fn inverse_difference(
        diff_data: &Array3<f64>,
        initial_values: &Array2<f64>,
        order: usize,
    ) -> Array3<f64> {
        let mut result = diff_data.clone();
        
        for _ in 0..order {
            let (batch_size, seq_len, features) = result.shape();
            let mut integrated = Array3::zeros((batch_size, seq_len + 1, features));
            
            // Set initial values
            for b in 0..batch_size {
                for f in 0..features {
                    integrated[[b, 0, f]] = initial_values[[b, f]];
                }
            }
            
            // Integrate
            for b in 0..batch_size {
                for t in 0..seq_len {
                    for f in 0..features {
                        integrated[[b, t + 1, f]] = integrated[[b, t, f]] + result[[b, t, f]];
                    }
                }
            }
            
            result = integrated;
        }
        
        result
    }
}

/// Metrics for model evaluation
pub mod metrics {
    use super::*;
    
    /// Mean Absolute Error
    pub fn mae(predictions: &Array3<f64>, targets: &Array3<f64>) -> f64 {
        ((predictions - targets).mapv(f64::abs)).mean().unwrap_or(0.0)
    }
    
    /// Mean Squared Error
    pub fn mse(predictions: &Array3<f64>, targets: &Array3<f64>) -> f64 {
        ((predictions - targets).mapv(|x| x * x)).mean().unwrap_or(0.0)
    }
    
    /// Root Mean Squared Error
    pub fn rmse(predictions: &Array3<f64>, targets: &Array3<f64>) -> f64 {
        mse(predictions, targets).sqrt()
    }
    
    /// Mean Absolute Percentage Error
    pub fn mape(predictions: &Array3<f64>, targets: &Array3<f64>) -> f64 {
        let epsilon = 1e-8;
        ((predictions - targets) / (targets.mapv(|x| x.abs() + epsilon)))
            .mapv(f64::abs)
            .mean()
            .unwrap_or(0.0) * 100.0
    }
    
    /// Symmetric Mean Absolute Percentage Error
    pub fn smape(predictions: &Array3<f64>, targets: &Array3<f64>) -> f64 {
        let epsilon = 1e-8;
        let numerator = (predictions - targets).mapv(f64::abs);
        let denominator = (predictions.mapv(f64::abs) + targets.mapv(f64::abs)) / 2.0 + epsilon;
        (numerator / denominator).mean().unwrap_or(0.0) * 100.0
    }
    
    /// Quantile loss for probabilistic forecasting
    pub fn quantile_loss(
        predictions: &Array3<f64>,
        targets: &Array3<f64>,
        quantile: f64,
    ) -> f64 {
        let errors = targets - predictions;
        errors.mapv(|e| {
            if e >= 0.0 {
                quantile * e
            } else {
                (quantile - 1.0) * e
            }
        }).mean().unwrap_or(0.0)
    }
    
    /// Coverage for prediction intervals
    pub fn coverage(
        predictions: &Array3<f64>,
        lower_bounds: &Array3<f64>,
        upper_bounds: &Array3<f64>,
        targets: &Array3<f64>,
    ) -> f64 {
        let in_bounds = targets.mapv(|t| 1.0)
            .zip_mut_with(&lower_bounds.zip_mut_with(upper_bounds, |&l, &u| {
                targets.zip_mut_with(&predictions, |&t, &_| {
                    if t >= l && t <= u { 1.0 } else { 0.0 }
                })
            }), |_, _| 0.0);
        
        in_bounds.mean().unwrap_or(0.0)
    }
}

/// Data generation utilities for testing
pub mod data_generation {
    use super::*;
    use rand::Rng;
    
    /// Generate synthetic time series with multiple components
    pub fn generate_synthetic_series(
        length: usize,
        trend_strength: f64,
        seasonal_periods: &[usize],
        noise_level: f64,
    ) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let mut series = Array1::zeros(length);
        
        // Add trend
        for i in 0..length {
            series[i] += trend_strength * (i as f64 / length as f64);
        }
        
        // Add seasonal components
        for &period in seasonal_periods {
            for i in 0..length {
                let phase = 2.0 * PI * (i as f64) / (period as f64);
                series[i] += phase.sin();
            }
        }
        
        // Add noise
        for i in 0..length {
            series[i] += rng.gen_range(-noise_level..noise_level);
        }
        
        series
    }
    
    /// Generate autoregressive series
    pub fn generate_ar_series(
        length: usize,
        coefficients: &[f64],
        noise_std: f64,
    ) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let mut series = Array1::zeros(length);
        let order = coefficients.len();
        
        // Initialize with random values
        for i in 0..order {
            series[i] = rng.gen_range(-1.0..1.0);
        }
        
        // Generate AR process
        for i in order..length {
            let mut value = 0.0;
            for (j, &coef) in coefficients.iter().enumerate() {
                value += coef * series[i - j - 1];
            }
            value += rng.gen_range(-noise_std..noise_std);
            series[i] = value;
        }
        
        series
    }
    
    /// Generate series with anomalies
    pub fn add_anomalies(
        series: &Array1<f64>,
        anomaly_rate: f64,
        anomaly_magnitude: f64,
    ) -> (Array1<f64>, Array1<bool>) {
        let mut rng = rand::thread_rng();
        let mut anomalous = series.clone();
        let mut is_anomaly = Array1::from_elem(series.len(), false);
        
        for i in 0..series.len() {
            if rng.gen::<f64>() < anomaly_rate {
                let magnitude = rng.gen_range(-anomaly_magnitude..anomaly_magnitude);
                anomalous[i] += magnitude * series.std(0.0);
                is_anomaly[i] = true;
            }
        }
        
        (anomalous, is_anomaly)
    }
}

/// Visualization helpers
pub mod visualization {
    use super::*;
    
    /// Create data for time series plot
    pub struct PlotData {
        pub timestamps: Vec<f64>,
        pub values: Vec<f64>,
        pub label: String,
    }
    
    /// Prepare data for plotting
    pub fn prepare_plot_data(
        series: &Array1<f64>,
        start_time: f64,
        time_step: f64,
        label: &str,
    ) -> PlotData {
        let timestamps: Vec<f64> = (0..series.len())
            .map(|i| start_time + i as f64 * time_step)
            .collect();
        
        PlotData {
            timestamps,
            values: series.to_vec(),
            label: label.to_string(),
        }
    }
    
    /// Create comparison plot data
    pub fn prepare_comparison(
        actual: &Array1<f64>,
        predicted: &Array1<f64>,
        start_time: f64,
        time_step: f64,
    ) -> Vec<PlotData> {
        vec![
            prepare_plot_data(actual, start_time, time_step, "Actual"),
            prepare_plot_data(predicted, start_time, time_step, "Predicted"),
        ]
    }
}

/// Model persistence utilities
pub mod persistence {
    use super::*;
    use serde::{Serialize, Deserialize};
    use std::fs::File;
    use std::io::{Read, Write};
    use std::path::Path;
    
    /// Save model weights to file
    pub fn save_weights<P: AsRef<Path>>(
        weights: &[(String, Array2<f64>)],
        path: P,
    ) -> Result<(), std::io::Error> {
        let serialized = bincode::serialize(weights)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        
        let mut file = File::create(path)?;
        file.write_all(&serialized)?;
        Ok(())
    }
    
    /// Load model weights from file
    pub fn load_weights<P: AsRef<Path>>(
        path: P,
    ) -> Result<Vec<(String, Array2<f64>)>, std::io::Error> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        let weights = bincode::deserialize(&buffer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        
        Ok(weights)
    }
}

/// Performance profiling utilities
pub mod profiling {
    use std::time::{Duration, Instant};
    use std::collections::HashMap;
    
    /// Simple performance profiler
    #[derive(Debug, Default)]
    pub struct Profiler {
        timings: HashMap<String, Vec<Duration>>,
        current_timers: HashMap<String, Instant>,
    }
    
    impl Profiler {
        pub fn new() -> Self {
            Self::default()
        }
        
        /// Start timing a named operation
        pub fn start(&mut self, name: &str) {
            self.current_timers.insert(name.to_string(), Instant::now());
        }
        
        /// Stop timing and record duration
        pub fn stop(&mut self, name: &str) {
            if let Some(start) = self.current_timers.remove(name) {
                let duration = start.elapsed();
                self.timings.entry(name.to_string())
                    .or_default()
                    .push(duration);
            }
        }
        
        /// Get average duration for an operation
        pub fn average(&self, name: &str) -> Option<Duration> {
            self.timings.get(name).map(|durations| {
                let sum: Duration = durations.iter().sum();
                sum / durations.len() as u32
            })
        }
        
        /// Get summary statistics
        pub fn summary(&self) -> HashMap<String, (Duration, Duration, Duration)> {
            let mut summary = HashMap::new();
            
            for (name, durations) in &self.timings {
                if !durations.is_empty() {
                    let min = durations.iter().min().copied().unwrap_or_default();
                    let max = durations.iter().max().copied().unwrap_or_default();
                    let avg = self.average(name).unwrap_or_default();
                    summary.insert(name.clone(), (min, avg, max));
                }
            }
            
            summary
        }
    }
}

// Remove extern crate declarations - dependencies are in Cargo.toml

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_preprocessing() {
        let data = Array3::from_shape_fn((2, 10, 1), |(b, t, _)| {
            (b * 10 + t) as f64
        });
        
        let (normalized, stats) = preprocessing::normalize(&data);
        assert_eq!(normalized.shape(), data.shape());
        assert_eq!(stats.len(), 2);
        
        let denormalized = preprocessing::denormalize(&normalized, &stats);
        assert!(((denormalized - &data).mapv(f64::abs)).mean().unwrap() < 1e-10);
    }
    
    #[test]
    fn test_metrics() {
        let predictions = Array3::ones((1, 10, 1));
        let targets = Array3::from_elem((1, 10, 1), 2.0);
        
        assert_eq!(metrics::mae(&predictions, &targets), 1.0);
        assert_eq!(metrics::mse(&predictions, &targets), 1.0);
        assert_eq!(metrics::rmse(&predictions, &targets), 1.0);
    }
}