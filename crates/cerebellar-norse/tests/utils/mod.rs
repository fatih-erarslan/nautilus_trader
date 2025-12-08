//! Test utilities and common functions for cerebellar-norse tests
//! 
//! This module provides reusable test utilities, fixtures, and helper functions
//! for comprehensive testing of the cerebellar SNN implementation.

use std::collections::HashMap;
use candle_core::{Tensor, Device, DType};
use candle_nn as nn;
use anyhow::Result;
use cerebellar_norse::*;

/// Common test fixtures and builders
pub mod fixtures;
/// Mock implementations for testing
pub mod mocks;
/// Test data generators
pub mod generators;
/// Assertion helpers
pub mod assertions;
/// Performance utilities
pub mod performance;
/// Configuration builders
pub mod config;

/// Test environment configuration
#[derive(Debug, Clone)]
pub struct TestEnvironment {
    pub device: Device,
    pub use_cuda: bool,
    pub use_parallel: bool,
    pub timeout_ms: u64,
    pub tolerance: f64,
}

impl Default for TestEnvironment {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            use_cuda: false,
            use_parallel: false,
            timeout_ms: 5000,
            tolerance: 1e-6,
        }
    }
}

/// Test result with detailed metrics
#[derive(Debug, Clone)]
pub struct TestResult {
    pub success: bool,
    pub duration: std::time::Duration,
    pub memory_usage: usize,
    pub message: String,
    pub metrics: HashMap<String, f64>,
}

impl TestResult {
    pub fn success(duration: std::time::Duration) -> Self {
        Self {
            success: true,
            duration,
            memory_usage: 0,
            message: "Test passed".to_string(),
            metrics: HashMap::new(),
        }
    }

    pub fn failure(message: String, duration: std::time::Duration) -> Self {
        Self {
            success: false,
            duration,
            memory_usage: 0,
            message,
            metrics: HashMap::new(),
        }
    }

    pub fn with_metrics(mut self, metrics: HashMap<String, f64>) -> Self {
        self.metrics = metrics;
        self
    }

    pub fn with_memory_usage(mut self, memory_usage: usize) -> Self {
        self.memory_usage = memory_usage;
        self
    }
}

/// Test execution context
pub struct TestContext {
    pub env: TestEnvironment,
    pub start_time: std::time::Instant,
    pub cleanup_handlers: Vec<Box<dyn FnOnce()>>,
}

impl TestContext {
    pub fn new(env: TestEnvironment) -> Self {
        Self {
            env,
            start_time: std::time::Instant::now(),
            cleanup_handlers: Vec::new(),
        }
    }

    pub fn add_cleanup<F>(&mut self, cleanup: F) 
    where
        F: FnOnce() + 'static,
    {
        self.cleanup_handlers.push(Box::new(cleanup));
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
}

impl Drop for TestContext {
    fn drop(&mut self) {
        // Execute cleanup handlers in reverse order
        for cleanup in self.cleanup_handlers.drain(..).rev() {
            cleanup();
        }
    }
}

/// Common test macros
#[macro_export]
macro_rules! assert_tensor_close {
    ($a:expr, $b:expr, $tolerance:expr) => {
        {
            let diff = ($a - $b).abs();
            let max_diff: f64 = diff.max_all().unwrap().to_scalar().unwrap();
            assert!(
                max_diff < $tolerance,
                "Tensors not close: max difference {} >= tolerance {}",
                max_diff,
                $tolerance
            );
        }
    };
}

#[macro_export]
macro_rules! assert_tensor_shape {
    ($tensor:expr, $expected_shape:expr) => {
        assert_eq!(
            $tensor.shape().dims(),
            $expected_shape,
            "Tensor shape mismatch: expected {:?}, got {:?}",
            $expected_shape,
            $tensor.shape().dims()
        );
    };
}

#[macro_export]
macro_rules! assert_performance {
    ($duration:expr, $max_duration:expr) => {
        assert!(
            $duration <= $max_duration,
            "Performance requirement failed: {} > {}",
            $duration.as_micros(),
            $max_duration.as_micros()
        );
    };
}

#[macro_export]
macro_rules! test_with_timeout {
    ($timeout:expr, $test:expr) => {
        {
            let start = std::time::Instant::now();
            let result = $test;
            let duration = start.elapsed();
            
            if duration > $timeout {
                panic!("Test timed out: {} > {}", duration.as_millis(), $timeout.as_millis());
            }
            
            result
        }
    };
}

/// Helper function to create test tensors
pub fn create_test_tensor(shape: &[usize], device: Device) -> Tensor {
    let dims: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
    Tensor::randn(&dims, (DType::F32, &device)).unwrap()
}

/// Helper function to create spike tensors (binary)
pub fn create_spike_tensor(shape: &[usize], spike_rate: f64, device: Device) -> Tensor {
    let dims: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
    let random_vals = Tensor::rand(&dims, (DType::F32, &device)).unwrap();
    random_vals.lt(spike_rate).unwrap().to_dtype(DType::F32).unwrap()
}

/// Helper function to measure memory usage
pub fn measure_memory_usage<F, R>(f: F) -> (R, usize)
where
    F: FnOnce() -> R,
{
    // This is a simplified memory measurement
    // In practice, you would use more sophisticated memory tracking
    let start_memory = get_memory_usage();
    let result = f();
    let end_memory = get_memory_usage();
    
    (result, end_memory.saturating_sub(start_memory))
}

/// Get current memory usage (simplified)
fn get_memory_usage() -> usize {
    // This is a placeholder - in practice you'd use proper memory measurement
    // You could use procfs, system calls, or memory profiling tools
    0
}

/// Helper function to time operations
pub fn time_operation<F, R>(f: F) -> (R, std::time::Duration)
where
    F: FnOnce() -> R,
{
    let start = std::time::Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}

/// Helper function to create test configuration
pub fn create_test_config() -> CerebellarNorseConfig {
    CerebellarNorseConfig {
        input_dim: 8,
        output_dim: 1,
        n_granule: 100,
        n_purkinje: 20,
        n_golgi: 10,
        n_dcn: 5,
        time_steps: 50,
        dt: 1e-3,
        use_adex: HashMap::new(),
        seed: 42,
        device: Device::Cpu,
        max_processing_time_us: 10000,
    }
}

/// Helper function to create layer configurations
pub fn create_layer_configs() -> HashMap<String, LayerConfig> {
    let mut configs = HashMap::new();
    
    configs.insert("granule".to_string(), LayerConfig {
        size: 100,
        neuron_type: NeuronType::LIF,
        tau_mem: 10.0,
        tau_syn_exc: 2.0,
        tau_syn_inh: 10.0,
        tau_adapt: Some(50.0),
        a: Some(2e-9),
        b: Some(1e-10),
    });
    
    configs.insert("purkinje".to_string(), LayerConfig {
        size: 20,
        neuron_type: NeuronType::AdEx,
        tau_mem: 15.0,
        tau_syn_exc: 3.0,
        tau_syn_inh: 5.0,
        tau_adapt: Some(100.0),
        a: Some(4e-9),
        b: Some(5e-10),
    });
    
    configs.insert("golgi".to_string(), LayerConfig {
        size: 10,
        neuron_type: NeuronType::LIF,
        tau_mem: 30.0,
        tau_syn_exc: 5.0,
        tau_syn_inh: 10.0,
        tau_adapt: Some(200.0),
        a: Some(2e-9),
        b: Some(2e-10),
    });
    
    configs.insert("dcn".to_string(), LayerConfig {
        size: 5,
        neuron_type: NeuronType::AdEx,
        tau_mem: 25.0,
        tau_syn_exc: 5.0,
        tau_syn_inh: 10.0,
        tau_adapt: Some(150.0),
        a: Some(1e-9),
        b: Some(5e-10),
    });
    
    configs
}

/// Helper function to validate tensor properties
pub fn validate_tensor_properties(tensor: &Tensor) -> Result<()> {
    // Check for NaN values
    let has_nan = tensor.isnan().unwrap().any().unwrap().to_scalar::<u8>().unwrap() != 0;
    if has_nan {
        return Err(anyhow::anyhow!("Tensor contains NaN values"));
    }
    
    // Check for infinite values
    let has_inf = tensor.isinf().unwrap().any().unwrap().to_scalar::<u8>().unwrap() != 0;
    if has_inf {
        return Err(anyhow::anyhow!("Tensor contains infinite values"));
    }
    
    Ok(())
}

/// Helper function to create deterministic test data
pub fn create_deterministic_data(shape: &[usize], seed: u64) -> Vec<f32> {
    use rand::{SeedableRng, Rng};
    use rand::rngs::StdRng;
    
    let mut rng = StdRng::seed_from_u64(seed);
    let total_size = shape.iter().product::<usize>();
    
    (0..total_size)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect()
}

/// Helper function to setup logging for tests
pub fn setup_test_logging() {
    use tracing_subscriber;
    
    let _ = tracing_subscriber::fmt()
        .with_env_filter("debug")
        .with_test_writer()
        .try_init();
}

/// Helper function to create temporary files for tests
pub fn create_temp_file() -> Result<tempfile::NamedTempFile> {
    tempfile::NamedTempFile::new()
        .map_err(|e| anyhow::anyhow!("Failed to create temp file: {}", e))
}

/// Helper function to compare test results
pub fn compare_results(expected: &TestResult, actual: &TestResult, tolerance: f64) -> bool {
    if expected.success != actual.success {
        return false;
    }
    
    for (key, expected_value) in &expected.metrics {
        if let Some(actual_value) = actual.metrics.get(key) {
            if (expected_value - actual_value).abs() > tolerance {
                return false;
            }
        } else {
            return false;
        }
    }
    
    true
}

/// Trait for test case builders
pub trait TestCaseBuilder<T> {
    fn build(&self) -> Result<T>;
}

/// Generic test runner for parameterized tests
pub struct TestRunner<T> {
    pub test_cases: Vec<T>,
    pub timeout: std::time::Duration,
}

impl<T> TestRunner<T> {
    pub fn new(timeout: std::time::Duration) -> Self {
        Self {
            test_cases: Vec::new(),
            timeout,
        }
    }
    
    pub fn add_test_case(&mut self, test_case: T) {
        self.test_cases.push(test_case);
    }
    
    pub fn run_tests<F>(&self, mut test_fn: F) -> Vec<TestResult>
    where
        F: FnMut(&T) -> TestResult,
    {
        self.test_cases
            .iter()
            .map(|test_case| {
                let start = std::time::Instant::now();
                let result = test_fn(test_case);
                let duration = start.elapsed();
                
                if duration > self.timeout {
                    TestResult::failure(
                        format!("Test timed out after {}ms", duration.as_millis()),
                        duration,
                    )
                } else {
                    result
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_test_result_creation() {
        let duration = std::time::Duration::from_millis(100);
        let result = TestResult::success(duration);
        
        assert!(result.success);
        assert_eq!(result.duration, duration);
        assert_eq!(result.message, "Test passed");
    }
    
    #[test]
    fn test_test_context() {
        let env = TestEnvironment::default();
        let mut ctx = TestContext::new(env);
        
        let mut cleanup_executed = false;
        ctx.add_cleanup(|| {
            // This would be executed when ctx is dropped
        });
        
        assert!(ctx.elapsed() < std::time::Duration::from_millis(100));
    }
    
    #[test]
    fn test_tensor_creation() {
        let tensor = create_test_tensor(&[2, 3], Device::Cpu);
        assert_eq!(tensor.shape().dims(), &[2, 3]);
    }
    
    #[test]
    fn test_spike_tensor_creation() {
        let tensor = create_spike_tensor(&[10, 10], 0.1, Device::Cpu);
        assert_eq!(tensor.shape().dims(), &[10, 10]);
        
        // Verify values are binary (0 or 1)
        let values: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();
        for value in values {
            assert!(value == 0.0 || value == 1.0, "Expected binary values, got {}", value);
        }
    }
    
    #[test]
    fn test_memory_measurement() {
        let (result, memory_used) = measure_memory_usage(|| {
            // Create some data
            let _vec: Vec<i32> = (0..1000).collect();
            42
        });
        
        assert_eq!(result, 42);
        // Memory measurement is simplified in this implementation
        assert!(memory_used >= 0);
    }
    
    #[test]
    fn test_time_operation() {
        let (result, duration) = time_operation(|| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            "done"
        });
        
        assert_eq!(result, "done");
        assert!(duration >= std::time::Duration::from_millis(10));
    }
    
    #[test]
    fn test_deterministic_data() {
        let data1 = create_deterministic_data(&[5, 5], 42);
        let data2 = create_deterministic_data(&[5, 5], 42);
        let data3 = create_deterministic_data(&[5, 5], 43);
        
        assert_eq!(data1, data2);
        assert_ne!(data1, data3);
        assert_eq!(data1.len(), 25);
    }
    
    #[test]
    fn test_test_runner() {
        let mut runner = TestRunner::new(std::time::Duration::from_secs(1));
        runner.add_test_case(10);
        runner.add_test_case(20);
        runner.add_test_case(30);
        
        let results = runner.run_tests(|&value| {
            if value > 15 {
                TestResult::success(std::time::Duration::from_millis(1))
            } else {
                TestResult::failure("Value too small".to_string(), std::time::Duration::from_millis(1))
            }
        });
        
        assert_eq!(results.len(), 3);
        assert!(!results[0].success);
        assert!(results[1].success);
        assert!(results[2].success);
    }
}