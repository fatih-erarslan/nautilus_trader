//! Nanosecond-Precision Performance Validator
//!
//! This module provides CPU cycle-accurate timing and validation for extreme
//! sub-microsecond performance requirements in high-frequency trading.
//!
//! PERFORMANCE TARGETS (ALL MUST BE VALIDATED):
//! - Trading decisions: <500 nanoseconds
//! - Whale detection: <200 nanoseconds  
//! - GPU kernels: <100 nanoseconds
//! - API responses: <50 nanoseconds
//!
//! VALIDATION REQUIREMENTS:
//! - 99.99% of operations under target latency
//! - Zero performance regressions
//! - Consistent performance under load
//! - Memory usage within bounds
//! - Zero memory leaks or safety issues

use std::arch::x86_64::_rdtsc;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use crate::error::AtsCoreError;

/// Nanosecond precision timing using RDTSC (Read Time-Stamp Counter)
#[derive(Debug, Clone)]
pub struct NanosecondTimer {
    cpu_freq_ghz: f64,
    overhead_cycles: u64,
    warmup_iterations: usize,
}

impl NanosecondTimer {
    /// Create a new nanosecond timer with calibrated CPU frequency
    pub fn new() -> Result<Self, AtsCoreError> {
        let cpu_freq_ghz = Self::calibrate_cpu_frequency()?;
        let overhead_cycles = Self::measure_timing_overhead(cpu_freq_ghz)?;
        
        Ok(Self {
            cpu_freq_ghz,
            overhead_cycles,
            warmup_iterations: 1000,
        })
    }
    
    /// Calibrate CPU frequency using system time as reference
    fn calibrate_cpu_frequency() -> Result<f64, AtsCoreError> {
        let mut total_cycles = 0u64;
        let mut total_duration = Duration::ZERO;
        
        // Perform multiple calibration runs
        for _ in 0..10 {
            let start_time = Instant::now();
            let start_cycles = unsafe { _rdtsc() };
            
            // Busy wait for approximately 1ms
            let target_end = start_time + Duration::from_millis(1);
            while Instant::now() < target_end {
                std::hint::spin_loop();
            }
            
            let end_cycles = unsafe { _rdtsc() };
            let end_time = Instant::now();
            
            let cycles = end_cycles - start_cycles;
            let duration = end_time - start_time;
            
            total_cycles += cycles;
            total_duration += duration;
        }
        
        // Calculate average frequency in GHz
        let avg_cycles = total_cycles as f64 / 10.0;
        let avg_duration_ns = total_duration.as_nanos() as f64 / 10.0;
        let freq_ghz = avg_cycles / avg_duration_ns;
        
        Ok(freq_ghz)
    }
    
    /// Measure timing overhead of the measurement itself
    fn measure_timing_overhead(cpu_freq_ghz: f64) -> Result<u64, AtsCoreError> {
        let mut overhead_cycles = Vec::new();
        
        // Warm up
        for _ in 0..1000 {
            let start = unsafe { _rdtsc() };
            let end = unsafe { _rdtsc() };
            overhead_cycles.push(end - start);
        }
        
        // Take minimum overhead (best case)
        let min_overhead = overhead_cycles.iter().min().copied().unwrap_or(0);
        
        Ok(min_overhead)
    }
    
    /// High-precision timing using RDTSC
    pub fn time_operation<F, R>(&self, operation: F) -> Result<(R, Duration), AtsCoreError>
    where
        F: FnOnce() -> R,
    {
        // Warmup to stabilize CPU frequency
        for _ in 0..self.warmup_iterations {
            let _ = unsafe { _rdtsc() };
        }
        
        // Serialize instructions to prevent reordering
        std::sync::atomic::compiler_fence(Ordering::SeqCst);
        
        let start_cycles = unsafe { _rdtsc() };
        let result = operation();
        let end_cycles = unsafe { _rdtsc() };
        
        std::sync::atomic::compiler_fence(Ordering::SeqCst);
        
        let cycles = end_cycles - start_cycles;
        let adjusted_cycles = cycles.saturating_sub(self.overhead_cycles);
        let nanoseconds = (adjusted_cycles as f64 / self.cpu_freq_ghz) as u64;
        
        Ok((result, Duration::from_nanos(nanoseconds)))
    }
    
    /// Batch timing for statistical analysis
    pub fn time_batch<F, R>(&self, operation: F, iterations: usize) -> Result<(Vec<R>, Vec<Duration>), AtsCoreError>
    where
        F: Fn() -> R,
    {
        let mut results = Vec::with_capacity(iterations);
        let mut durations = Vec::with_capacity(iterations);
        
        for _ in 0..iterations {
            let (result, duration) = self.time_operation(&operation)?;
            results.push(result);
            durations.push(duration);
        }
        
        Ok((results, durations))
    }
    
    /// Advanced statistical timing with outlier detection
    pub fn time_statistical<F, R>(&self, operation: F, iterations: usize) -> Result<TimingStatistics<R>, AtsCoreError>
    where
        F: Fn() -> R,
    {
        let (results, durations) = self.time_batch(operation, iterations)?;
        
        let mut duration_ns: Vec<u64> = durations.iter().map(|d| d.as_nanos() as u64).collect();
        duration_ns.sort_unstable();
        
        let min = duration_ns[0];
        let max = duration_ns[duration_ns.len() - 1];
        let median = duration_ns[duration_ns.len() / 2];
        let p99 = duration_ns[(duration_ns.len() as f64 * 0.99) as usize];
        let p999 = duration_ns[(duration_ns.len() as f64 * 0.999) as usize];
        let p9999 = duration_ns[(duration_ns.len() as f64 * 0.9999) as usize];
        
        let mean = duration_ns.iter().sum::<u64>() as f64 / duration_ns.len() as f64;
        let variance = duration_ns.iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>() / duration_ns.len() as f64;
        let stddev = variance.sqrt();
        
        Ok(TimingStatistics {
            results,
            durations,
            min_ns: min,
            max_ns: max,
            median_ns: median,
            mean_ns: mean,
            stddev_ns: stddev,
            p99_ns: p99,
            p999_ns: p999,
            p9999_ns: p9999,
            iterations,
        })
    }
}

/// Comprehensive timing statistics
#[derive(Debug, Clone)]
pub struct TimingStatistics<R> {
    pub results: Vec<R>,
    pub durations: Vec<Duration>,
    pub min_ns: u64,
    pub max_ns: u64,
    pub median_ns: u64,
    pub mean_ns: f64,
    pub stddev_ns: f64,
    pub p99_ns: u64,
    pub p999_ns: u64,
    pub p9999_ns: u64,
    pub iterations: usize,
}

impl<R> TimingStatistics<R> {
    /// Check if results meet nanosecond precision targets
    pub fn validate_target(&self, target_ns: u64, success_rate: f64) -> ValidationResult {
        let under_target = self.durations.iter()
            .filter(|d| d.as_nanos() as u64 <= target_ns)
            .count();
        
        let actual_success_rate = under_target as f64 / self.iterations as f64;
        let passed = actual_success_rate >= success_rate;
        
        ValidationResult {
            target_ns,
            required_success_rate: success_rate,
            actual_success_rate,
            passed,
            min_ns: self.min_ns,
            max_ns: self.max_ns,
            median_ns: self.median_ns,
            p99_ns: self.p99_ns,
            p999_ns: self.p999_ns,
            p9999_ns: self.p9999_ns,
            iterations: self.iterations,
        }
    }
}

/// Validation result for nanosecond precision targets
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub target_ns: u64,
    pub required_success_rate: f64,
    pub actual_success_rate: f64,
    pub passed: bool,
    pub min_ns: u64,
    pub max_ns: u64,
    pub median_ns: u64,
    pub p99_ns: u64,
    pub p999_ns: u64,
    pub p9999_ns: u64,
    pub iterations: usize,
}

impl ValidationResult {
    /// Display detailed validation results
    pub fn display_results(&self) {
        println!("🎯 Nanosecond Precision Validation Results");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Target: {}ns ({}% success rate required)", self.target_ns, (self.required_success_rate * 100.0));
        println!("Result: {} ({}% success rate achieved)", 
                 if self.passed { "✅ PASSED" } else { "❌ FAILED" },
                 (self.actual_success_rate * 100.0));
        println!();
        println!("📊 Timing Statistics:");
        println!("  Min:     {}ns", self.min_ns);
        println!("  Median:  {}ns", self.median_ns);
        println!("  P99:     {}ns", self.p99_ns);
        println!("  P99.9:   {}ns", self.p999_ns);
        println!("  P99.99:  {}ns", self.p9999_ns);
        println!("  Max:     {}ns", self.max_ns);
        println!("  Samples: {}", self.iterations);
        println!();
        
        if !self.passed {
            println!("❌ VALIDATION FAILED:");
            println!("  Target success rate: {:.2}%", self.required_success_rate * 100.0);
            println!("  Actual success rate: {:.2}%", self.actual_success_rate * 100.0);
            println!("  Deficit: {:.2}%", (self.required_success_rate - self.actual_success_rate) * 100.0);
        }
    }
}

/// Performance validation framework for nanosecond precision
pub struct NanosecondValidator {
    timer: NanosecondTimer,
    results: Arc<std::sync::Mutex<HashMap<String, ValidationResult>>>,
}

impl NanosecondValidator {
    /// Create a new nanosecond validator
    pub fn new() -> Result<Self, AtsCoreError> {
        Ok(Self {
            timer: NanosecondTimer::new()?,
            results: Arc::new(std::sync::Mutex::new(HashMap::new())),
        })
    }
    
    /// Validate trading decision latency (<500ns)
    pub fn validate_trading_decision<F>(&self, operation: F, name: &str) -> Result<ValidationResult, AtsCoreError>
    where
        F: Fn() -> (),
    {
        let stats = self.timer.time_statistical(operation, 100000)?;
        let result = stats.validate_target(500, 0.9999); // 99.99% under 500ns
        
        let mut results = self.results.lock().unwrap();
        results.insert(format!("trading_decision_{}", name), result.clone());
        
        Ok(result)
    }
    
    /// Validate whale detection latency (<200ns)
    pub fn validate_whale_detection<F>(&self, operation: F, name: &str) -> Result<ValidationResult, AtsCoreError>
    where
        F: Fn() -> (),
    {
        let stats = self.timer.time_statistical(operation, 100000)?;
        let result = stats.validate_target(200, 0.9999); // 99.99% under 200ns
        
        let mut results = self.results.lock().unwrap();
        results.insert(format!("whale_detection_{}", name), result.clone());
        
        Ok(result)
    }
    
    /// Validate GPU kernel latency (<100ns)
    pub fn validate_gpu_kernel<F>(&self, operation: F, name: &str) -> Result<ValidationResult, AtsCoreError>
    where
        F: Fn() -> (),
    {
        let stats = self.timer.time_statistical(operation, 100000)?;
        let result = stats.validate_target(100, 0.9999); // 99.99% under 100ns
        
        let mut results = self.results.lock().unwrap();
        results.insert(format!("gpu_kernel_{}", name), result.clone());
        
        Ok(result)
    }
    
    /// Validate API response latency (<50ns)
    pub fn validate_api_response<F>(&self, operation: F, name: &str) -> Result<ValidationResult, AtsCoreError>
    where
        F: Fn() -> (),
    {
        let stats = self.timer.time_statistical(operation, 100000)?;
        let result = stats.validate_target(50, 0.9999); // 99.99% under 50ns
        
        let mut results = self.results.lock().unwrap();
        results.insert(format!("api_response_{}", name), result.clone());
        
        Ok(result)
    }
    
    /// Validate custom operation with specified target
    pub fn validate_custom<F>(&self, operation: F, name: &str, target_ns: u64, success_rate: f64) -> Result<ValidationResult, AtsCoreError>
    where
        F: Fn() -> (),
    {
        let stats = self.timer.time_statistical(operation, 100000)?;
        let result = stats.validate_target(target_ns, success_rate);
        
        let mut results = self.results.lock().unwrap();
        results.insert(format!("custom_{}", name), result.clone());
        
        Ok(result)
    }
    
    /// Generate comprehensive validation report
    pub fn generate_report(&self) -> ValidationReport {
        let results = self.results.lock().unwrap();
        let mut report = ValidationReport::new();
        
        for (name, result) in results.iter() {
            report.add_result(name.clone(), result.clone());
        }
        
        report
    }
    
    /// Validate memory allocation performance (no degradation)
    pub fn validate_memory_stability<F>(&self, operation: F, name: &str) -> Result<ValidationResult, AtsCoreError>
    where
        F: Fn() -> (),
    {
        // First batch
        let first_stats = self.timer.time_statistical(&operation, 10000)?;
        
        // Second batch (after potential allocations)
        let second_stats = self.timer.time_statistical(&operation, 10000)?;
        
        // Check for performance degradation
        let degradation = (second_stats.mean_ns - first_stats.mean_ns) / first_stats.mean_ns;
        let passed = degradation < 0.05; // Less than 5% degradation
        
        let result = ValidationResult {
            target_ns: first_stats.mean_ns as u64,
            required_success_rate: 0.95,
            actual_success_rate: if passed { 1.0 } else { 0.0 },
            passed,
            min_ns: second_stats.min_ns,
            max_ns: second_stats.max_ns,
            median_ns: second_stats.median_ns,
            p99_ns: second_stats.p99_ns,
            p999_ns: second_stats.p999_ns,
            p9999_ns: second_stats.p9999_ns,
            iterations: second_stats.iterations,
        };
        
        let mut results = self.results.lock().unwrap();
        results.insert(format!("memory_stability_{}", name), result.clone());
        
        Ok(result)
    }
}

/// Comprehensive validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    results: HashMap<String, ValidationResult>,
    generated_at: std::time::SystemTime,
}

impl ValidationReport {
    fn new() -> Self {
        Self {
            results: HashMap::new(),
            generated_at: std::time::SystemTime::now(),
        }
    }
    
    fn add_result(&mut self, name: String, result: ValidationResult) {
        self.results.insert(name, result);
    }
    
    /// Check if all validations passed
    pub fn all_passed(&self) -> bool {
        self.results.values().all(|r| r.passed)
    }
    
    /// Display comprehensive report
    pub fn display_comprehensive_report(&self) {
        println!("🚀 NANOSECOND PRECISION VALIDATION REPORT");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Generated: {:?}", self.generated_at);
        println!("Total Validations: {}", self.results.len());
        
        let passed_count = self.results.values().filter(|r| r.passed).count();
        let failed_count = self.results.len() - passed_count;
        
        println!("✅ Passed: {}", passed_count);
        println!("❌ Failed: {}", failed_count);
        println!();
        
        if self.all_passed() {
            println!("🎉 ALL VALIDATIONS PASSED - NANOSECOND PRECISION ACHIEVED!");
        } else {
            println!("⚠️  SOME VALIDATIONS FAILED - PERFORMANCE TARGETS NOT MET");
        }
        
        println!();
        println!("📋 Detailed Results:");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        for (name, result) in &self.results {
            println!("Test: {}", name);
            println!("  Status: {}", if result.passed { "✅ PASSED" } else { "❌ FAILED" });
            println!("  Target: {}ns", result.target_ns);
            println!("  Success Rate: {:.2}% (required: {:.2}%)", 
                     result.actual_success_rate * 100.0, 
                     result.required_success_rate * 100.0);
            println!("  Median: {}ns", result.median_ns);
            println!("  P99.99: {}ns", result.p9999_ns);
            println!();
        }
    }
    
    /// Export results as JSON
    pub fn export_json(&self) -> Result<String, AtsCoreError> {
        use serde_json::json;
        
        let json_data = json!({
            "generated_at": self.generated_at.duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default().as_secs(),
            "all_passed": self.all_passed(),
            "results": self.results.iter().map(|(name, result)| {
                json!({
                    "name": name,
                    "passed": result.passed,
                    "target_ns": result.target_ns,
                    "success_rate": result.actual_success_rate,
                    "median_ns": result.median_ns,
                    "p99_ns": result.p99_ns,
                    "p999_ns": result.p999_ns,
                    "p9999_ns": result.p9999_ns,
                })
            }).collect::<Vec<_>>()
        });
        
        Ok(serde_json::to_string_pretty(&json_data)?)
    }
}

/// Real-world scenario testing for nanosecond validation
pub struct RealWorldScenarios {
    validator: NanosecondValidator,
}

impl RealWorldScenarios {
    pub fn new() -> Result<Self, AtsCoreError> {
        Ok(Self {
            validator: NanosecondValidator::new()?,
        })
    }
    
    /// Simulate whale attack detection scenario
    pub fn simulate_whale_attack(&self) -> Result<ValidationResult, AtsCoreError> {
        // Simulate whale detection algorithm
        let whale_detection = || {
            // Simulate pattern matching and analysis
            let mut sum = 0.0;
            for i in 0..100 {
                sum += (i as f64).sin().cos();
            }
            let _condition = sum > 0.0; // Dummy condition
        };
        
        self.validator.validate_whale_detection(whale_detection, "whale_attack_simulation")
    }
    
    /// Simulate high-frequency trading decision
    pub fn simulate_hft_decision(&self) -> Result<ValidationResult, AtsCoreError> {
        // Simulate trading decision algorithm
        let trading_decision = || {
            // Simulate market data processing and decision making
            let mut price_diff = 0.0;
            for i in 0..50 {
                price_diff += (i as f64 * 0.01).tanh();
            }
            let _condition = price_diff > 0.1; // Dummy trading condition
        };
        
        self.validator.validate_trading_decision(trading_decision, "hft_decision_simulation")
    }
    
    /// Simulate GPU kernel execution
    pub fn simulate_gpu_kernel(&self) -> Result<ValidationResult, AtsCoreError> {
        // Simulate GPU kernel workload
        let gpu_kernel = || {
            // Simulate matrix multiplication or similar GPU operation
            let mut result = 0.0;
            for i in 0..16 {
                for j in 0..16 {
                    result += (i * j) as f64;
                }
            }
            let _condition = result > 0.0;
        };
        
        self.validator.validate_gpu_kernel(gpu_kernel, "gpu_kernel_simulation")
    }
    
    /// Simulate API response processing
    pub fn simulate_api_response(&self) -> Result<ValidationResult, AtsCoreError> {
        // Simulate API response processing
        let api_response = || {
            // Simulate JSON parsing and validation
            let mut checksum = 0u64;
            for i in 0..20 {
                checksum = checksum.wrapping_add(i);
            }
            let _condition = checksum > 0;
        };
        
        self.validator.validate_api_response(api_response, "api_response_simulation")
    }
    
    /// Run comprehensive real-world scenario testing
    pub fn run_comprehensive_scenarios(&self) -> Result<ValidationReport, AtsCoreError> {
        println!("🔥 Running Real-World Nanosecond Precision Scenarios...");
        
        // Run all scenarios
        let _ = self.simulate_whale_attack()?;
        let _ = self.simulate_hft_decision()?;
        let _ = self.simulate_gpu_kernel()?;
        let _ = self.simulate_api_response()?;
        
        // Generate comprehensive report
        let report = self.validator.generate_report();
        
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nanosecond_timer_creation() {
        let timer = NanosecondTimer::new();
        assert!(timer.is_ok());
    }
    
    #[test]
    fn test_basic_timing() {
        let timer = NanosecondTimer::new().unwrap();
        let (result, duration) = timer.time_operation(|| {
            // Simple operation
            let mut sum = 0;
            for i in 0..10 {
                sum += i;
            }
            sum
        }).unwrap();
        
        assert_eq!(result, 45);
        assert!(duration.as_nanos() > 0);
    }
    
    #[test]
    fn test_statistical_timing() {
        let timer = NanosecondTimer::new().unwrap();
        let stats = timer.time_statistical(|| {
            // Simple operation
            (0..10).sum::<i32>()
        }, 1000).unwrap();
        
        assert_eq!(stats.iterations, 1000);
        assert!(stats.min_ns > 0);
        assert!(stats.max_ns >= stats.min_ns);
        assert!(stats.median_ns >= stats.min_ns);
        assert!(stats.p99_ns >= stats.median_ns);
    }
    
    #[test]
    fn test_validation_framework() {
        let validator = NanosecondValidator::new().unwrap();
        
        // Test with a very fast operation
        let result = validator.validate_custom(|| {
            // Minimal operation
            let _ = 1 + 1;
        }, "minimal_operation", 1000, 0.99).unwrap();
        
        // Should pass for such a simple operation
        assert!(result.passed);
    }
    
    #[test]
    fn test_real_world_scenarios() {
        let scenarios = RealWorldScenarios::new().unwrap();
        let report = scenarios.run_comprehensive_scenarios().unwrap();
        
        // Should have all scenario results
        assert!(report.results.len() >= 4);
        
        // Display report for manual inspection
        report.display_comprehensive_report();
    }
}