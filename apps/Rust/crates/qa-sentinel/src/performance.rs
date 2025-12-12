//! Performance Testing Framework for Sub-100μs Latency Validation
//!
//! This module implements comprehensive performance testing with microsecond-level
//! precision for high-frequency trading systems.

use crate::config::QaSentinelConfig;
use crate::quality_gates::{TestResults, TestResult};
use anyhow::Result;
use tracing::{info, debug, warn, error};
use std::time::{Duration, Instant};
use std::collections::HashMap;

// Performance measurement imports
use criterion::{Criterion, BatchSize, BenchmarkId, Throughput};
use sysinfo::{System, SystemExt, ProcessExt, CpuExt, DiskExt, NetworkExt};

/// TENGRI performance requirements
const MAX_LATENCY_MICROSECONDS: u64 = 100;
const MIN_THROUGHPUT_OPS_PER_SEC: u64 = 10_000;
const MAX_MEMORY_USAGE_MB: f64 = 512.0;
const MAX_CPU_USAGE_PERCENT: f64 = 80.0;

/// Performance testing framework with microsecond precision
pub struct PerformanceTestRunner {
    config: QaSentinelConfig,
    system: System,
    baseline_metrics: Option<SystemMetrics>,
}

/// System performance metrics
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub disk_read_mb_per_sec: f64,
    pub disk_write_mb_per_sec: f64,
    pub network_rx_mb_per_sec: f64,
    pub network_tx_mb_per_sec: f64,
    pub process_count: usize,
    pub thread_count: usize,
}

/// Performance benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub operation_name: String,
    pub mean_latency_nanos: u64,
    pub p50_latency_nanos: u64,
    pub p95_latency_nanos: u64,
    pub p99_latency_nanos: u64,
    pub max_latency_nanos: u64,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub passed_latency_requirement: bool,
    pub passed_throughput_requirement: bool,
}

/// Performance test trait
pub trait PerformanceTest {
    /// Name of the performance test
    fn name(&self) -> &'static str;
    
    /// Description of the performance test
    fn description(&self) -> &'static str;
    
    /// Expected operation count for throughput testing
    fn expected_ops_per_iteration(&self) -> u64 { 1 }
    
    /// Setup test environment
    fn setup(&mut self) -> Result<()> { Ok(()) }
    
    /// The operation to benchmark
    fn execute_operation(&mut self) -> Result<()>;
    
    /// Cleanup test environment
    fn cleanup(&mut self) -> Result<()> { Ok(()) }
    
    /// Custom latency requirement (overrides default 100μs)
    fn latency_requirement_nanos(&self) -> Option<u64> { None }
    
    /// Custom throughput requirement
    fn throughput_requirement_ops_per_sec(&self) -> Option<u64> { None }
}

/// ATS-CP Temperature Scaling Performance Test
pub struct ATSCPTemperatureScalingPerfTest {
    test_data: Vec<(f64, f64)>, // (confidence, temperature) pairs
}

impl PerformanceTest for ATSCPTemperatureScalingPerfTest {
    fn name(&self) -> &'static str {
        "ats_cp_temperature_scaling_performance"
    }
    
    fn description(&self) -> &'static str {
        "ATS-CP temperature scaling performance under load"
    }
    
    fn expected_ops_per_iteration(&self) -> u64 {
        self.test_data.len() as u64
    }
    
    fn setup(&mut self) -> Result<()> {
        // Generate test data
        self.test_data = (0..1000)
            .map(|i| {
                let confidence = 0.1 + (i as f64) * 0.8 / 1000.0;
                let temperature = 0.5 + (i as f64) * 2.0 / 1000.0;
                (confidence, temperature)
            })
            .collect();
        Ok(())
    }
    
    fn execute_operation(&mut self) -> Result<()> {
        // Benchmark temperature scaling operations
        for &(confidence, temperature) in &self.test_data {
            let _scaled = self.temperature_scale(confidence, temperature);
        }
        Ok(())
    }
    
    fn latency_requirement_nanos(&self) -> Option<u64> {
        Some(50_000) // 50μs for batch operation
    }
    
    fn throughput_requirement_ops_per_sec(&self) -> Option<u64> {
        Some(20_000) // 20k operations per second
    }
}

impl ATSCPTemperatureScalingPerfTest {
    pub fn new() -> Self {
        Self {
            test_data: Vec::new(),
        }
    }
    
    fn temperature_scale(&self, confidence: f64, temperature: f64) -> f64 {
        let logit = (confidence / (1.0 - confidence)).ln();
        let scaled_logit = logit / temperature;
        1.0 / (1.0 + (-scaled_logit).exp())
    }
}

/// Conformal Prediction Performance Test
pub struct ConformalPredictionPerfTest {
    calibration_scores: Vec<f64>,
    test_scores: Vec<f64>,
}

impl PerformanceTest for ConformalPredictionPerfTest {
    fn name(&self) -> &'static str {
        "conformal_prediction_performance"
    }
    
    fn description(&self) -> &'static str {
        "Conformal prediction interval computation performance"
    }
    
    fn setup(&mut self) -> Result<()> {
        // Generate calibration and test data
        self.calibration_scores = (0..1000).map(|i| i as f64 / 1000.0).collect();
        self.test_scores = (0..100).map(|i| i as f64 / 100.0).collect();
        Ok(())
    }
    
    fn execute_operation(&mut self) -> Result<()> {
        // Benchmark conformal prediction interval computation
        let alpha = 0.1;
        for &test_score in &self.test_scores {
            let _interval = self.compute_prediction_interval(test_score, alpha);
        }
        Ok(())
    }
    
    fn latency_requirement_nanos(&self) -> Option<u64> {
        Some(10_000) // 10μs for prediction intervals
    }
}

impl ConformalPredictionPerfTest {
    pub fn new() -> Self {
        Self {
            calibration_scores: Vec::new(),
            test_scores: Vec::new(),
        }
    }
    
    fn compute_prediction_interval(&self, test_score: f64, alpha: f64) -> (f64, f64) {
        let quantile_level = 1.0 - alpha;
        let n = self.calibration_scores.len();
        let quantile_index = ((quantile_level * (n + 1) as f64).ceil() as usize).min(n);
        
        if quantile_index == 0 {
            return (test_score, test_score);
        }
        
        let mut sorted_scores = self.calibration_scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let threshold = sorted_scores[quantile_index - 1];
        (test_score - threshold, test_score + threshold)
    }
}

/// Database Query Performance Test
pub struct DatabaseQueryPerfTest;

impl PerformanceTest for DatabaseQueryPerfTest {
    fn name(&self) -> &'static str {
        "database_query_performance"
    }
    
    fn description(&self) -> &'static str {
        "Database query latency performance test"
    }
    
    fn execute_operation(&mut self) -> Result<()> {
        // Simulate database query operation
        std::thread::sleep(Duration::from_nanos(10_000)); // 10μs simulation
        Ok(())
    }
    
    fn latency_requirement_nanos(&self) -> Option<u64> {
        Some(50_000) // 50μs for database queries
    }
    
    fn throughput_requirement_ops_per_sec(&self) -> Option<u64> {
        Some(20_000) // 20k queries per second
    }
}

/// Memory Allocation Performance Test
pub struct MemoryAllocationPerfTest {
    allocations: Vec<Vec<u8>>,
}

impl PerformanceTest for MemoryAllocationPerfTest {
    fn name(&self) -> &'static str {
        "memory_allocation_performance"
    }
    
    fn description(&self) -> &'static str {
        "Memory allocation and deallocation performance"
    }
    
    fn setup(&mut self) -> Result<()> {
        self.allocations.clear();
        Ok(())
    }
    
    fn execute_operation(&mut self) -> Result<()> {
        // Allocate and deallocate memory blocks
        for size in [1024, 4096, 8192, 16384] {
            let allocation = vec![0u8; size];
            self.allocations.push(allocation);
        }
        self.allocations.clear();
        Ok(())
    }
    
    fn cleanup(&mut self) -> Result<()> {
        self.allocations.clear();
        Ok(())
    }
    
    fn latency_requirement_nanos(&self) -> Option<u64> {
        Some(5_000) // 5μs for memory operations
    }
}

impl MemoryAllocationPerfTest {
    pub fn new() -> Self {
        Self {
            allocations: Vec::new(),
        }
    }
}

impl PerformanceTestRunner {
    pub fn new(config: QaSentinelConfig) -> Self {
        Self {
            config,
            system: System::new_all(),
            baseline_metrics: None,
        }
    }
    
    /// Initialize performance testing framework
    pub async fn initialize(&mut self) -> Result<()> {
        info!("⚡ Initializing performance testing framework");
        
        // Collect baseline system metrics
        self.system.refresh_all();
        self.baseline_metrics = Some(self.collect_system_metrics());
        
        info!("✅ Performance testing framework initialized");
        Ok(())
    }
    
    /// Run comprehensive performance test suite
    pub async fn run_all_tests(&mut self) -> Result<TestResults> {
        info!("🏃 Running comprehensive performance test suite");
        
        let mut results = TestResults::new();
        
        // ATS-CP Temperature Scaling Performance
        let mut ats_cp_test = ATSCPTemperatureScalingPerfTest::new();
        let ats_cp_result = self.run_performance_test(&mut ats_cp_test).await?;
        results.add_result(ats_cp_result);
        
        // Conformal Prediction Performance
        let mut conformal_test = ConformalPredictionPerfTest::new();
        let conformal_result = self.run_performance_test(&mut conformal_test).await?;
        results.add_result(conformal_result);
        
        // Database Query Performance
        let mut db_test = DatabaseQueryPerfTest;
        let db_result = self.run_performance_test(&mut db_test).await?;
        results.add_result(db_result);
        
        // Memory Allocation Performance
        let mut memory_test = MemoryAllocationPerfTest::new();
        let memory_result = self.run_performance_test(&mut memory_test).await?;
        results.add_result(memory_result);
        
        // System-wide performance validation
        let system_result = self.validate_system_performance().await?;
        results.add_result(system_result);
        
        info!("✅ Performance test suite completed: {} passed, {} failed", 
              results.passed_count(), results.failed_count());
        
        Ok(results)
    }
    
    /// Run a single performance test with microsecond precision
    pub async fn run_performance_test<T: PerformanceTest>(&mut self, test: &mut T) -> Result<TestResult> {
        info!("🔬 Running performance test: {}", test.name());
        
        // Setup test
        test.setup()?;
        
        let start_time = Instant::now();
        let mut latencies = Vec::new();
        let iterations = 1000;
        
        // Collect baseline metrics
        self.system.refresh_all();
        let start_metrics = self.collect_system_metrics();
        
        // Run warm-up iterations
        for _ in 0..100 {
            test.execute_operation()?;
        }
        
        // Run benchmarked iterations
        for _ in 0..iterations {
            let iter_start = Instant::now();
            test.execute_operation()?;
            let iter_duration = iter_start.elapsed();
            latencies.push(iter_duration.as_nanos() as u64);
        }
        
        let total_duration = start_time.elapsed();
        
        // Collect end metrics
        self.system.refresh_all();
        let end_metrics = self.collect_system_metrics();
        
        // Cleanup test
        test.cleanup()?;
        
        // Calculate statistics
        latencies.sort_unstable();
        let mean_latency = latencies.iter().sum::<u64>() / latencies.len() as u64;
        let p50_latency = latencies[latencies.len() * 50 / 100];
        let p95_latency = latencies[latencies.len() * 95 / 100];
        let p99_latency = latencies[latencies.len() * 99 / 100];
        let max_latency = *latencies.last().unwrap();
        
        let total_ops = iterations * test.expected_ops_per_iteration();
        let throughput = total_ops as f64 / total_duration.as_secs_f64();
        
        // Calculate resource usage
        let memory_usage = end_metrics.memory_usage_mb - start_metrics.memory_usage_mb;
        let cpu_usage = (end_metrics.cpu_usage_percent + start_metrics.cpu_usage_percent) / 2.0;
        
        // Validate requirements
        let latency_requirement = test.latency_requirement_nanos()
            .unwrap_or(MAX_LATENCY_MICROSECONDS * 1000);
        let throughput_requirement = test.throughput_requirement_ops_per_sec()
            .unwrap_or(MIN_THROUGHPUT_OPS_PER_SEC);
        
        let passed_latency = mean_latency <= latency_requirement;
        let passed_throughput = throughput >= throughput_requirement as f64;
        let passed_memory = memory_usage <= MAX_MEMORY_USAGE_MB;
        let passed_cpu = cpu_usage <= MAX_CPU_USAGE_PERCENT;
        
        let overall_passed = passed_latency && passed_throughput && passed_memory && passed_cpu;
        
        if !overall_passed {
            warn!("❌ Performance test failed: {}", test.name());
            warn!("  Mean latency: {}ns (requirement: {}ns)", mean_latency, latency_requirement);
            warn!("  Throughput: {:.2} ops/sec (requirement: {} ops/sec)", throughput, throughput_requirement);
            warn!("  Memory usage: {:.2}MB (limit: {}MB)", memory_usage, MAX_MEMORY_USAGE_MB);
            warn!("  CPU usage: {:.2}% (limit: {}%)", cpu_usage, MAX_CPU_USAGE_PERCENT);
        } else {
            info!("✅ Performance test passed: {}", test.name());
            info!("  Mean latency: {}ns ({}μs)", mean_latency, mean_latency / 1000);
            info!("  P99 latency: {}ns ({}μs)", p99_latency, p99_latency / 1000);
            info!("  Throughput: {:.2} ops/sec", throughput);
        }
        
        let benchmark_result = BenchmarkResult {
            operation_name: test.name().to_string(),
            mean_latency_nanos: mean_latency,
            p50_latency_nanos: p50_latency,
            p95_latency_nanos: p95_latency,
            p99_latency_nanos: p99_latency,
            max_latency_nanos: max_latency,
            throughput_ops_per_sec: throughput,
            memory_usage_mb: memory_usage,
            cpu_usage_percent: cpu_usage,
            passed_latency_requirement: passed_latency,
            passed_throughput_requirement: passed_throughput,
        };
        
        Ok(TestResult {
            test_name: test.name().to_string(),
            passed: overall_passed,
            duration: total_duration,
            error: if overall_passed { None } else { Some("Performance requirements not met".to_string()) },
            metrics: Default::default(),
        })
    }
    
    /// Validate overall system performance
    async fn validate_system_performance(&mut self) -> Result<TestResult> {
        info!("🖥️ Validating system performance");
        
        let start_time = Instant::now();
        
        self.system.refresh_all();
        let metrics = self.collect_system_metrics();
        
        let baseline = self.baseline_metrics.as_ref().unwrap();
        
        // Check system resource usage
        let memory_growth = metrics.memory_usage_mb - baseline.memory_usage_mb;
        let cpu_elevated = metrics.cpu_usage_percent > MAX_CPU_USAGE_PERCENT;
        let excessive_processes = metrics.process_count > baseline.process_count + 100;
        
        let passed = memory_growth <= MAX_MEMORY_USAGE_MB && 
                    !cpu_elevated && 
                    !excessive_processes;
        
        let duration = start_time.elapsed();
        
        if passed {
            info!("✅ System performance validation passed");
        } else {
            error!("❌ System performance validation failed");
            error!("  Memory growth: {:.2}MB", memory_growth);
            error!("  CPU usage: {:.2}%", metrics.cpu_usage_percent);
            error!("  Process count: {}", metrics.process_count);
        }
        
        Ok(TestResult {
            test_name: "system_performance_validation".to_string(),
            passed,
            duration,
            error: if passed { None } else { Some("System performance degraded".to_string()) },
            metrics: Default::default(),
        })
    }
    
    /// Collect current system metrics
    fn collect_system_metrics(&self) -> SystemMetrics {
        let cpu_usage = self.system.global_cpu_info().cpu_usage() as f64;
        let memory_usage = self.system.used_memory() as f64 / 1024.0 / 1024.0;
        
        let process_count = self.system.processes().len();
        let thread_count = self.system.processes().values()
            .map(|p| p.tasks().len())
            .sum();
        
        SystemMetrics {
            cpu_usage_percent: cpu_usage,
            memory_usage_mb: memory_usage,
            disk_read_mb_per_sec: 0.0,  // Would require disk monitoring
            disk_write_mb_per_sec: 0.0, // Would require disk monitoring
            network_rx_mb_per_sec: 0.0, // Would require network monitoring
            network_tx_mb_per_sec: 0.0, // Would require network monitoring
            process_count,
            thread_count,
        }
    }
}

pub async fn initialize_performance_testing(config: &QaSentinelConfig) -> Result<()> {
    info!("⚡ Initializing performance testing framework");
    
    // Initialize criterion for benchmarking
    let mut criterion = Criterion::default();
    criterion = criterion.measurement_time(Duration::from_secs(5));
    criterion = criterion.sample_size(1000);
    
    info!("✅ Performance testing framework initialized");
    Ok(())
}

pub async fn run_performance_tests(config: &QaSentinelConfig) -> Result<TestResults> {
    let mut runner = PerformanceTestRunner::new(config.clone());
    runner.initialize().await?;
    runner.run_all_tests().await
}