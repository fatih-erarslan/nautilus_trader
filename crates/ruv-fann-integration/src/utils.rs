//! Utility functions for ruv_FANN Integration
//!
//! This module provides common utility functions used across the ruv_FANN integration.

use ndarray::{Array1, Array2, Array3};
use serde::{Serialize, Deserialize};
use crate::error::{RuvFannError, RuvFannResult};

/// Activation function implementations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    Tanh,
    Sigmoid,
    Leaky(f64),
    ELU(f64),
    Swish,
    GELU,
    Mish,
    Linear,
}

impl ActivationFunction {
    /// Create activation function from string
    pub fn from_string(name: &str) -> RuvFannResult<Self> {
        match name.to_lowercase().as_str() {
            "relu" => Ok(Self::ReLU),
            "tanh" => Ok(Self::Tanh),
            "sigmoid" => Ok(Self::Sigmoid),
            "leaky_relu" => Ok(Self::Leaky(0.01)),
            "elu" => Ok(Self::ELU(1.0)),
            "swish" => Ok(Self::Swish),
            "gelu" => Ok(Self::GELU),
            "mish" => Ok(Self::Mish),
            "linear" => Ok(Self::Linear),
            _ => Err(RuvFannError::validation_error(
                format!("Unknown activation function: {}", name)
            )),
        }
    }
    
    /// Apply activation function to array
    pub fn apply(&self, input: &Array2<f64>) -> RuvFannResult<Array2<f64>> {
        match self {
            Self::ReLU => Ok(input.mapv(|x| x.max(0.0))),
            Self::Tanh => Ok(input.mapv(|x| x.tanh())),
            Self::Sigmoid => Ok(input.mapv(|x| 1.0 / (1.0 + (-x).exp()))),
            Self::Leaky(alpha) => Ok(input.mapv(|x| if x > 0.0 { x } else { alpha * x })),
            Self::ELU(alpha) => Ok(input.mapv(|x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })),
            Self::Swish => Ok(input.mapv(|x| x / (1.0 + (-x).exp()))),
            Self::GELU => Ok(input.mapv(|x| 0.5 * x * (1.0 + (x / 2.0_f64.sqrt()).tanh()))),
            Self::Mish => Ok(input.mapv(|x| x * (1.0 + x.exp()).ln().tanh())),
            Self::Linear => Ok(input.clone()),
        }
    }
    
    /// Compute derivative for backpropagation
    pub fn derivative(&self, output: &Array2<f64>) -> RuvFannResult<Array2<f64>> {
        match self {
            Self::ReLU => Ok(output.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })),
            Self::Tanh => Ok(output.mapv(|x| 1.0 - x * x)),
            Self::Sigmoid => Ok(output.mapv(|x| x * (1.0 - x))),
            Self::Leaky(alpha) => Ok(output.mapv(|x| if x > 0.0 { 1.0 } else { *alpha })),
            Self::ELU(alpha) => Ok(output.mapv(|x| if x > 0.0 { 1.0 } else { alpha * x.exp() })),
            Self::Linear => Ok(Array2::ones(output.raw_dim())),
            _ => Err(RuvFannError::neural_network_error("Derivative not implemented for this activation")),
        }
    }
}

/// Normalize data to zero mean and unit variance
pub fn normalize_data(data: &Array2<f64>) -> RuvFannResult<(Array2<f64>, NormalizationParams)> {
    let mean = data.mean_axis(ndarray::Axis(0)).unwrap();
    let std = data.std_axis(ndarray::Axis(0), 0.0);
    
    let normalized = (data - &mean) / &std;
    
    let params = NormalizationParams {
        mean,
        std,
        min_val: data.fold(f64::INFINITY, |acc, &x| acc.min(x)),
        max_val: data.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x)),
    };
    
    Ok((normalized, params))
}

/// Denormalize data using stored parameters
pub fn denormalize_data(data: &Array2<f64>, params: &NormalizationParams) -> RuvFannResult<Array2<f64>> {
    Ok(data * &params.std + &params.mean)
}

/// Normalization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
    pub min_val: f64,
    pub max_val: f64,
}

/// Calculate prediction accuracy metrics
pub fn calculate_accuracy_metrics(predictions: &Array2<f64>, targets: &Array2<f64>) -> RuvFannResult<AccuracyMetrics> {
    if predictions.shape() != targets.shape() {
        return Err(RuvFannError::validation_error("Prediction and target shapes don't match"));
    }
    
    let diff = predictions - targets;
    let squared_diff = diff.mapv(|x| x * x);
    let absolute_diff = diff.mapv(|x| x.abs());
    
    let mse = squared_diff.mean().unwrap_or(0.0);
    let rmse = mse.sqrt();
    let mae = absolute_diff.mean().unwrap_or(0.0);
    
    // Calculate R-squared
    let target_mean = targets.mean().unwrap_or(0.0);
    let ss_tot = targets.mapv(|x| (x - target_mean).powi(2)).sum();
    let ss_res = squared_diff.sum();
    let r_squared = if ss_tot != 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
    
    Ok(AccuracyMetrics {
        mse,
        rmse,
        mae,
        r_squared,
    })
}

/// Accuracy metrics structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub r_squared: f64,
}

/// Generate random weights using Xavier initialization
pub fn xavier_initialization(input_size: usize, output_size: usize) -> Array2<f64> {
    use rand::prelude::*;
    use rand_distr::StandardNormal;
    
    let mut rng = thread_rng();
    let scale = (2.0 / (input_size + output_size) as f64).sqrt();
    
    Array2::from_shape_fn((input_size, output_size), |_| {
        rng.sample::<f64, _>(StandardNormal) * scale
    })
}

/// Generate random weights using He initialization
pub fn he_initialization(input_size: usize, output_size: usize) -> Array2<f64> {
    use rand::prelude::*;
    use rand_distr::StandardNormal;
    
    let mut rng = thread_rng();
    let scale = (2.0 / input_size as f64).sqrt();
    
    Array2::from_shape_fn((input_size, output_size), |_| {
        rng.sample::<f64, _>(StandardNormal) * scale
    })
}

/// Apply dropout to array
pub fn apply_dropout(input: &Array2<f64>, dropout_rate: f64, training: bool) -> Array2<f64> {
    if !training || dropout_rate == 0.0 {
        return input.clone();
    }
    
    use rand::prelude::*;
    let mut rng = thread_rng();
    let scale = 1.0 / (1.0 - dropout_rate);
    
    input.mapv(|x| {
        if rng.gen::<f64>() < dropout_rate {
            0.0
        } else {
            x * scale
        }
    })
}

/// Calculate gradient clipping
pub fn clip_gradients(gradients: &mut Array2<f64>, max_norm: f64) {
    let grad_norm = gradients.mapv(|x| x * x).sum().sqrt();
    
    if grad_norm > max_norm {
        let scale = max_norm / grad_norm;
        gradients.mapv_inplace(|x| x * scale);
    }
}

/// Learning rate scheduling
#[derive(Debug, Clone)]
pub enum LearningRateScheduler {
    Constant(f64),
    ExponentialDecay { initial_lr: f64, decay_rate: f64, decay_steps: usize },
    StepDecay { initial_lr: f64, decay_factor: f64, step_size: usize },
    CosineAnnealing { initial_lr: f64, min_lr: f64, t_max: usize },
}

impl LearningRateScheduler {
    /// Get learning rate for current step
    pub fn get_lr(&self, step: usize) -> f64 {
        match self {
            Self::Constant(lr) => *lr,
            Self::ExponentialDecay { initial_lr, decay_rate, decay_steps } => {
                initial_lr * decay_rate.powf(step as f64 / *decay_steps as f64)
            },
            Self::StepDecay { initial_lr, decay_factor, step_size } => {
                initial_lr * decay_factor.powf((step / step_size) as f64)
            },
            Self::CosineAnnealing { initial_lr, min_lr, t_max } => {
                let progress = (step % t_max) as f64 / *t_max as f64;
                min_lr + (initial_lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0
            },
        }
    }
}

/// Moving average tracker
#[derive(Debug, Clone)]
pub struct MovingAverage {
    values: std::collections::VecDeque<f64>,
    window_size: usize,
    sum: f64,
}

impl MovingAverage {
    /// Create new moving average tracker
    pub fn new(window_size: usize) -> Self {
        Self {
            values: std::collections::VecDeque::with_capacity(window_size),
            window_size,
            sum: 0.0,
        }
    }
    
    /// Add new value and get current average
    pub fn add(&mut self, value: f64) -> f64 {
        if self.values.len() >= self.window_size {
            if let Some(old_value) = self.values.pop_front() {
                self.sum -= old_value;
            }
        }
        
        self.values.push_back(value);
        self.sum += value;
        
        self.average()
    }
    
    /// Get current average
    pub fn average(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f64
        }
    }
    
    /// Reset the tracker
    pub fn reset(&mut self) {
        self.values.clear();
        self.sum = 0.0;
    }
}

/// Exponential moving average tracker
#[derive(Debug, Clone)]
pub struct ExponentialMovingAverage {
    value: f64,
    alpha: f64,
    initialized: bool,
}

impl ExponentialMovingAverage {
    /// Create new exponential moving average tracker
    pub fn new(alpha: f64) -> Self {
        Self {
            value: 0.0,
            alpha: alpha.clamp(0.0, 1.0),
            initialized: false,
        }
    }
    
    /// Update with new value and get current average
    pub fn update(&mut self, new_value: f64) -> f64 {
        if !self.initialized {
            self.value = new_value;
            self.initialized = true;
        } else {
            self.value = self.alpha * new_value + (1.0 - self.alpha) * self.value;
        }
        
        self.value
    }
    
    /// Get current value
    pub fn value(&self) -> f64 {
        self.value
    }
    
    /// Reset the tracker
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.initialized = false;
    }
}

/// Data windowing utilities
pub fn create_sliding_windows(data: &Array2<f64>, window_size: usize, stride: usize) -> RuvFannResult<Array3<f64>> {
    let (rows, cols) = data.dim();
    
    if window_size > rows {
        return Err(RuvFannError::validation_error("Window size larger than data"));
    }
    
    let num_windows = (rows - window_size) / stride + 1;
    let mut windows = Array3::zeros((num_windows, window_size, cols));
    
    for (i, mut window) in windows.outer_iter_mut().enumerate() {
        let start_idx = i * stride;
        let end_idx = start_idx + window_size;
        window.assign(&data.slice(ndarray::s![start_idx..end_idx, ..]));
    }
    
    Ok(windows)
}

/// Time series cross-validation
pub struct TimeSeriesSplit {
    n_splits: usize,
    test_size: usize,
    gap: usize,
}

impl TimeSeriesSplit {
    /// Create new time series split
    pub fn new(n_splits: usize, test_size: usize, gap: usize) -> Self {
        Self { n_splits, test_size, gap }
    }
    
    /// Generate train/test splits
    pub fn split(&self, data_length: usize) -> RuvFannResult<Vec<(std::ops::Range<usize>, std::ops::Range<usize>)>> {
        if self.test_size + self.gap >= data_length {
            return Err(RuvFannError::validation_error("Test size + gap too large"));
        }
        
        let mut splits = Vec::new();
        let total_test_size = self.test_size + self.gap;
        let available_length = data_length - total_test_size;
        
        for i in 0..self.n_splits {
            let test_start = available_length + i * (total_test_size / self.n_splits);
            let test_end = test_start + self.test_size;
            
            let train_end = test_start - self.gap;
            
            if train_end <= 0 {
                break;
            }
            
            splits.push((0..train_end, test_start..test_end));
        }
        
        Ok(splits)
    }
}

/// Performance timing utilities
pub struct Timer {
    start: std::time::Instant,
    label: String,
}

impl Timer {
    /// Start new timer
    pub fn new(label: &str) -> Self {
        Self {
            start: std::time::Instant::now(),
            label: label.to_string(),
        }
    }
    
    /// Get elapsed time
    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }
    
    /// Stop timer and log result
    pub fn stop(&self) {
        let elapsed = self.elapsed();
        tracing::debug!("Timer '{}': {:?}", self.label, elapsed);
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Memory usage tracking
pub fn get_memory_usage() -> usize {
    #[cfg(feature = "memory-stats")]
    {
        memory_stats::memory_stats()
            .map(|usage| usage.physical_mem)
            .unwrap_or(0)
    }
    #[cfg(not(feature = "memory-stats"))]
    {
        0
    }
}

/// System information
pub fn get_system_info() -> SystemInfo {
    SystemInfo {
        cpu_count: num_cpus::get(),
        memory_total: get_total_memory(),
        memory_available: get_available_memory(),
        platform: std::env::consts::OS.to_string(),
        architecture: std::env::consts::ARCH.to_string(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub cpu_count: usize,
    pub memory_total: u64,
    pub memory_available: u64,
    pub platform: String,
    pub architecture: String,
}

fn get_total_memory() -> u64 {
    #[cfg(feature = "sysinfo")]
    {
        use sysinfo::SystemExt;
        let mut system = sysinfo::System::new_all();
        system.refresh_memory();
        system.total_memory()
    }
    #[cfg(not(feature = "sysinfo"))]
    {
        0
    }
}

fn get_available_memory() -> u64 {
    #[cfg(feature = "sysinfo")]
    {
        use sysinfo::SystemExt;
        let mut system = sysinfo::System::new_all();
        system.refresh_memory();
        system.available_memory()
    }
    #[cfg(not(feature = "sysinfo"))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    
    #[test]
    fn test_activation_functions() {
        let input = arr2(&[[1.0, -1.0], [0.5, -0.5]]);
        
        let relu = ActivationFunction::ReLU;
        let output = relu.apply(&input).unwrap();
        assert_eq!(output[[0, 0]], 1.0);
        assert_eq!(output[[0, 1]], 0.0);
        
        let sigmoid = ActivationFunction::Sigmoid;
        let output = sigmoid.apply(&input).unwrap();
        assert!(output[[0, 0]] > 0.5);
        assert!(output[[0, 1]] < 0.5);
    }
    
    #[test]
    fn test_normalization() {
        let data = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let (normalized, params) = normalize_data(&data).unwrap();
        
        // Check that normalized data has approximately zero mean
        let mean = normalized.mean_axis(ndarray::Axis(0)).unwrap();
        assert!(mean[0].abs() < 1e-10);
        assert!(mean[1].abs() < 1e-10);
        
        // Test denormalization
        let denormalized = denormalize_data(&normalized, &params).unwrap();
        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                assert!((data[[i, j]] - denormalized[[i, j]]).abs() < 1e-10);
            }
        }
    }
    
    #[test]
    fn test_moving_average() {
        let mut ma = MovingAverage::new(3);
        
        assert_eq!(ma.add(1.0), 1.0);
        assert_eq!(ma.add(2.0), 1.5);
        assert_eq!(ma.add(3.0), 2.0);
        assert_eq!(ma.add(4.0), 3.0); // Window slides
    }
    
    #[test]
    fn test_exponential_moving_average() {
        let mut ema = ExponentialMovingAverage::new(0.1);
        
        let value1 = ema.update(10.0);
        assert_eq!(value1, 10.0); // First value
        
        let value2 = ema.update(20.0);
        assert_eq!(value2, 11.0); // 0.1 * 20 + 0.9 * 10
    }
    
    #[test]
    fn test_learning_rate_scheduler() {
        let scheduler = LearningRateScheduler::ExponentialDecay {
            initial_lr: 0.01,
            decay_rate: 0.9,
            decay_steps: 100,
        };
        
        let lr0 = scheduler.get_lr(0);
        let lr100 = scheduler.get_lr(100);
        
        assert_eq!(lr0, 0.01);
        assert_eq!(lr100, 0.009);
    }
    
    #[test]
    fn test_sliding_windows() {
        let data = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
        let windows = create_sliding_windows(&data, 2, 1).unwrap();
        
        assert_eq!(windows.dim(), (3, 2, 2)); // 3 windows, 2 timesteps, 2 features
        assert_eq!(windows[[0, 0, 0]], 1.0);
        assert_eq!(windows[[0, 1, 0]], 3.0);
        assert_eq!(windows[[1, 0, 0]], 3.0);
    }
    
    #[test]
    fn test_time_series_split() {
        let splitter = TimeSeriesSplit::new(3, 10, 5);
        let splits = splitter.split(100).unwrap();
        
        assert!(!splits.is_empty());
        
        for (train_range, test_range) in &splits {
            assert!(train_range.end + 5 <= test_range.start); // Gap maintained
            assert_eq!(test_range.len(), 10); // Test size maintained
        }
    }
}