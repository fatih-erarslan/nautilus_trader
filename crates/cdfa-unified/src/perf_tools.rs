//! Performance tools and profiling utilities for CDFA unified crate
//! 
//! This module provides comprehensive performance monitoring, profiling,
//! and optimization tools for validating that the unified crate meets
//! all performance targets.

use crate::{
    types::{CdfaArray, CdfaMatrix, CdfaFloat, CdfaResult},
    error::CdfaError,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

/// Performance targets for CDFA operations
pub struct PerformanceTargets;

impl PerformanceTargets {
    /// Core diversity calculations should complete under 10μs
    pub const CORE_DIVERSITY_MICROS: u64 = 10;
    
    /// Signal fusion should complete under 20μs
    pub const SIGNAL_FUSION_MICROS: u64 = 20;
    
    /// Pattern detection should complete under 50μs
    pub const PATTERN_DETECTION_MICROS: u64 = 50;
    
    /// Full CDFA workflow should complete under 100μs
    pub const FULL_WORKFLOW_MICROS: u64 = 100;
    
    /// Memory usage should stay under 50MB for typical workloads
    pub const MEMORY_LIMIT_MB: f64 = 50.0;
    
    /// Minimum speedup expected vs Python reference implementation
    pub const PYTHON_SPEEDUP_MIN: f64 = 10.0;
    
    /// Maximum allowed speedup vs Python (sanity check)
    pub const PYTHON_SPEEDUP_MAX: f64 = 50.0;
}

/// Performance measurement result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurement {
    pub operation: String,
    pub duration_nanos: u64,
    pub duration_micros: f64,
    pub memory_mb: f64,
    pub meets_target: bool,
    pub target_micros: u64,
    pub speedup_factor: Option<f64>,
}

impl PerformanceMeasurement {
    pub fn new(operation: &str, duration: Duration, memory_mb: f64, target_micros: u64) -> Self {
        let duration_nanos = duration.as_nanos() as u64;
        let duration_micros = duration.as_micros() as f64;
        let meets_target = duration_micros <= target_micros as f64;
        
        Self {
            operation: operation.to_string(),
            duration_nanos,
            duration_micros,
            memory_mb,
            meets_target,
            target_micros,
            speedup_factor: None,
        }
    }
    
    pub fn with_speedup(mut self, speedup: f64) -> Self {
        self.speedup_factor = Some(speedup);
        self
    }
}

/// Comprehensive performance profiler
pub struct PerformanceProfiler {
    measurements: Vec<PerformanceMeasurement>,
    memory_tracker: MemoryTracker,
    start_time: Instant,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
            memory_tracker: MemoryTracker::new(),
            start_time: Instant::now(),
        }
    }
    
    /// Time an operation and validate against performance targets
    pub fn time_operation<F, R>(&mut self, name: &str, target_micros: u64, operation: F) -> CdfaResult<R>
    where
        F: FnOnce() -> CdfaResult<R>,
    {
        let memory_before = self.memory_tracker.current_usage_mb();
        let start = Instant::now();
        
        let result = operation()?;
        
        let duration = start.elapsed();
        let memory_after = self.memory_tracker.current_usage_mb();
        let memory_used = memory_after - memory_before;
        
        let measurement = PerformanceMeasurement::new(name, duration, memory_used, target_micros);
        
        if !measurement.meets_target {
            eprintln!(
                "⚠️  Performance target missed for {}: {}μs > {}μs",
                name, measurement.duration_micros, target_micros
            );
        }
        
        self.measurements.push(measurement);
        Ok(result)
    }
    
    /// Compare performance against Python reference implementation
    pub fn compare_with_python<F, R>(&mut self, name: &str, python_duration_ms: f64, operation: F) -> CdfaResult<R>
    where
        F: FnOnce() -> CdfaResult<R>,
    {
        let start = Instant::now();
        let result = operation()?;
        let duration = start.elapsed();
        
        let rust_duration_ms = duration.as_millis() as f64;
        let speedup = python_duration_ms / rust_duration_ms;
        
        let target_micros = (python_duration_ms * 1000.0 / PerformanceTargets::PYTHON_SPEEDUP_MIN) as u64;
        let mut measurement = PerformanceMeasurement::new(name, duration, 0.0, target_micros);
        measurement = measurement.with_speedup(speedup);
        
        // Validate speedup is within expected range
        if speedup < PerformanceTargets::PYTHON_SPEEDUP_MIN {
            eprintln!(
                "⚠️  Insufficient speedup vs Python for {}: {}x < {}x",
                name, speedup, PerformanceTargets::PYTHON_SPEEDUP_MIN
            );
        } else if speedup > PerformanceTargets::PYTHON_SPEEDUP_MAX {
            eprintln!(
                "🤔 Unexpectedly high speedup vs Python for {}: {}x > {}x (check measurement)",
                name, speedup, PerformanceTargets::PYTHON_SPEEDUP_MAX
            );
        }
        
        self.measurements.push(measurement);
        Ok(result)
    }
    
    /// Generate comprehensive performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let total_operations = self.measurements.len();
        let passed_targets = self.measurements.iter().filter(|m| m.meets_target).count();
        let target_pass_rate = passed_targets as f64 / total_operations as f64;
        
        let avg_speedup = self.measurements
            .iter()
            .filter_map(|m| m.speedup_factor)
            .fold((0.0, 0), |(sum, count), speedup| (sum + speedup, count + 1))
            .0 / self.measurements.len().max(1) as f64;
        
        let peak_memory = self.measurements
            .iter()
            .map(|m| m.memory_mb)
            .fold(0.0, f64::max);
        
        PerformanceReport {
            total_operations,
            passed_targets,
            target_pass_rate,
            avg_speedup_vs_python: avg_speedup,
            peak_memory_mb: peak_memory,
            measurements: self.measurements.clone(),
            total_runtime: self.start_time.elapsed(),
        }
    }
    
    /// Validate all performance requirements are met
    pub fn validate_requirements(&self) -> CdfaResult<()> {
        let report = self.generate_report();
        
        // Check target pass rate
        if report.target_pass_rate < 0.95 {
            return Err(CdfaError::PerformanceValidation(format!(
                "Performance target pass rate {}% < 95%",
                report.target_pass_rate * 100.0
            )));
        }
        
        // Check memory usage
        if report.peak_memory_mb > PerformanceTargets::MEMORY_LIMIT_MB {
            return Err(CdfaError::PerformanceValidation(format!(
                "Peak memory usage {}MB > limit {}MB",
                report.peak_memory_mb,
                PerformanceTargets::MEMORY_LIMIT_MB
            )));
        }
        
        // Check Python speedup
        if report.avg_speedup_vs_python > 0.0 && report.avg_speedup_vs_python < PerformanceTargets::PYTHON_SPEEDUP_MIN {
            return Err(CdfaError::PerformanceValidation(format!(
                "Average Python speedup {}x < minimum {}x",
                report.avg_speedup_vs_python,
                PerformanceTargets::PYTHON_SPEEDUP_MIN
            )));
        }
        
        Ok(())
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage tracker
#[derive(Debug)]
pub struct MemoryTracker {
    allocations: HashMap<String, usize>,
    peak_usage: usize,
    current_usage: usize,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            peak_usage: 0,
            current_usage: 0,
        }
    }
    
    pub fn track_allocation(&mut self, name: &str, size_bytes: usize) {
        self.allocations.insert(name.to_string(), size_bytes);
        self.current_usage += size_bytes;
        self.peak_usage = self.peak_usage.max(self.current_usage);
    }
    
    pub fn track_deallocation(&mut self, name: &str) {
        if let Some(size) = self.allocations.remove(name) {
            self.current_usage = self.current_usage.saturating_sub(size);
        }
    }
    
    pub fn current_usage_mb(&self) -> f64 {
        self.current_usage as f64 / (1024.0 * 1024.0)
    }
    
    pub fn peak_usage_mb(&self) -> f64 {
        self.peak_usage as f64 / (1024.0 * 1024.0)
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub total_operations: usize,
    pub passed_targets: usize,
    pub target_pass_rate: f64,
    pub avg_speedup_vs_python: f64,
    pub peak_memory_mb: f64,
    pub measurements: Vec<PerformanceMeasurement>,
    pub total_runtime: Duration,
}

impl PerformanceReport {
    /// Display a human-readable performance summary
    pub fn display_summary(&self) {
        println!("\n🎯 CDFA Performance Validation Report");
        println!("=====================================");
        println!("📊 Operations: {}", self.total_operations);
        println!("✅ Passed Targets: {} ({:.1}%)", self.passed_targets, self.target_pass_rate * 100.0);
        println!("🚀 Avg Python Speedup: {:.1}x", self.avg_speedup_vs_python);
        println!("💾 Peak Memory: {:.2}MB", self.peak_memory_mb);
        println!("⏱️  Total Runtime: {:.2}s", self.total_runtime.as_secs_f64());
        
        // Show slowest operations
        let mut sorted_measurements = self.measurements.clone();
        sorted_measurements.sort_by(|a, b| b.duration_micros.partial_cmp(&a.duration_micros).unwrap());
        
        println!("\n🐌 Slowest Operations:");
        for (i, measurement) in sorted_measurements.iter().take(5).enumerate() {
            let status = if measurement.meets_target { "✅" } else { "❌" };
            println!(
                "{}. {} {} {:.1}μs (target: {}μs)",
                i + 1,
                status,
                measurement.operation,
                measurement.duration_micros,
                measurement.target_micros
            );
        }
        
        // Show Python comparisons
        let python_comparisons: Vec<_> = self.measurements
            .iter()
            .filter(|m| m.speedup_factor.is_some())
            .collect();
        
        if !python_comparisons.is_empty() {
            println!("\n🐍 Python Speedup Comparisons:");
            for measurement in python_comparisons {
                let speedup = measurement.speedup_factor.unwrap();
                let status = if speedup >= PerformanceTargets::PYTHON_SPEEDUP_MIN { "✅" } else { "❌" };
                println!(
                    "{} {}: {:.1}x speedup",
                    status,
                    measurement.operation,
                    speedup
                );
            }
        }
    }
    
    /// Export report to JSON
    pub fn to_json(&self) -> CdfaResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| CdfaError::Serialization(format!("Failed to serialize report: {}", e)))
    }
    
    /// Save report to file
    pub fn save_to_file(&self, path: &str) -> CdfaResult<()> {
        let json = self.to_json()?;
        std::fs::write(path, json)
            .map_err(|e| CdfaError::Io(format!("Failed to write report: {}", e)))
    }
}

/// CPU feature detection for SIMD optimization validation
pub struct CpuFeatureDetector;

impl CpuFeatureDetector {
    /// Detect available CPU features and validate SIMD capabilities
    pub fn detect_features() -> CpuFeatures {
        #[cfg(target_arch = "x86_64")]
        {
            CpuFeatures {
                sse: std::arch::is_x86_feature_detected!("sse"),
                sse2: std::arch::is_x86_feature_detected!("sse2"),
                sse3: std::arch::is_x86_feature_detected!("sse3"),
                sse4_1: std::arch::is_x86_feature_detected!("sse4.1"),
                sse4_2: std::arch::is_x86_feature_detected!("sse4.2"),
                avx: std::arch::is_x86_feature_detected!("avx"),
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                avx512f: std::arch::is_x86_feature_detected!("avx512f"),
                fma: std::arch::is_x86_feature_detected!("fma"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            CpuFeatures::default()
        }
    }
    
    /// Get optimal SIMD width for current CPU
    pub fn optimal_simd_width() -> usize {
        let features = Self::detect_features();
        
        if features.avx512f {
            64 // 512 bits / 8 bits per byte
        } else if features.avx2 || features.avx {
            32 // 256 bits / 8 bits per byte
        } else if features.sse2 {
            16 // 128 bits / 8 bits per byte
        } else {
            8  // Fallback to 64-bit operations
        }
    }
}

/// CPU features available for SIMD optimization
#[derive(Debug, Clone, Default)]
pub struct CpuFeatures {
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub fma: bool,
}

impl CpuFeatures {
    /// Get a summary string of available features
    pub fn summary(&self) -> String {
        let mut features = Vec::new();
        
        if self.sse { features.push("SSE"); }
        if self.sse2 { features.push("SSE2"); }
        if self.sse3 { features.push("SSE3"); }
        if self.sse4_1 { features.push("SSE4.1"); }
        if self.sse4_2 { features.push("SSE4.2"); }
        if self.avx { features.push("AVX"); }
        if self.avx2 { features.push("AVX2"); }
        if self.avx512f { features.push("AVX512F"); }
        if self.fma { features.push("FMA"); }
        
        if features.is_empty() {
            "No SIMD features detected".to_string()
        } else {
            features.join(", ")
        }
    }
}

/// Utility functions for performance testing
pub mod test_utils {
    use super::*;
    use crate::types::{CdfaArray, CdfaMatrix};
    use ndarray::{Array1, Array2};
    
    /// Generate test data for performance benchmarks
    pub fn generate_test_data(size: usize) -> (CdfaArray, CdfaMatrix) {
        let array = Array1::linspace(0.0, 1.0, size);
        let matrix = Array2::from_shape_fn((size / 10, 10), |(i, j)| {
            (i as CdfaFloat * 0.1) + (j as CdfaFloat * 0.01)
        });
        (array, matrix)
    }
    
    /// Generate correlation matrix for diversity testing
    pub fn generate_correlation_matrix(size: usize) -> CdfaMatrix {
        let mut matrix = Array2::eye(size);
        for i in 0..size {
            for j in (i + 1)..size {
                let corr = 0.5 * ((i + j) as CdfaFloat / size as CdfaFloat);
                matrix[[i, j]] = corr;
                matrix[[j, i]] = corr;
            }
        }
        matrix
    }
    
    /// Simulate Python benchmark results for comparison
    pub fn simulate_python_benchmarks() -> HashMap<String, f64> {
        let mut benchmarks = HashMap::new();
        
        // Simulated Python execution times in milliseconds
        benchmarks.insert("pearson_diversity_1000".to_string(), 15.2);
        benchmarks.insert("kendall_diversity_1000".to_string(), 28.7);
        benchmarks.insert("score_fusion_1000".to_string(), 8.9);
        benchmarks.insert("rank_fusion_1000".to_string(), 12.1);
        benchmarks.insert("volatility_estimation_10000".to_string(), 45.6);
        benchmarks.insert("entropy_calculation_10000".to_string(), 67.3);
        benchmarks.insert("statistics_calculation_10000".to_string(), 23.4);
        benchmarks.insert("full_workflow_1000".to_string(), 156.8);
        
        benchmarks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    
    #[test]
    fn test_performance_profiler() {
        let mut profiler = PerformanceProfiler::new();
        
        // Test a simple operation
        let result = profiler.time_operation(
            "test_operation",
            PerformanceTargets::CORE_DIVERSITY_MICROS,
            || {
                std::thread::sleep(Duration::from_micros(5)); // Should pass
                Ok(42)
            }
        );
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        
        let report = profiler.generate_report();
        assert_eq!(report.total_operations, 1);
        assert_eq!(report.passed_targets, 1);
    }
    
    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new();
        
        tracker.track_allocation("test_array", 1024);
        assert_eq!(tracker.current_usage_mb(), 1024.0 / (1024.0 * 1024.0));
        
        tracker.track_deallocation("test_array");
        assert_eq!(tracker.current_usage_mb(), 0.0);
    }
    
    #[test]
    fn test_cpu_feature_detection() {
        let features = CpuFeatureDetector::detect_features();
        let summary = features.summary();
        
        // At minimum, we should have some features on x86_64
        #[cfg(target_arch = "x86_64")]
        assert!(!summary.is_empty());
        
        let width = CpuFeatureDetector::optimal_simd_width();
        assert!(width >= 8);
    }
}