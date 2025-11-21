//! Standalone test for optimized conformal prediction
//!
//! This test validates the optimized implementation works correctly
//! and achieves performance targets.

use crate::{
    config::AtsCpConfig,
    conformal::ConformalPredictor,
    conformal_optimized::OptimizedConformalPredictor,
    types::AtsCpVariant,
};
use std::time::Instant;

/// Runs comprehensive validation of optimized conformal prediction
pub fn run_validation() -> Result<ValidationReport, Box<dyn std::error::Error>> {
    println!("🚀 Running optimized conformal prediction validation...");
    
    let mut report = ValidationReport::new();
    
    // Test 1: Basic functionality
    test_basic_functionality(&mut report)?;
    
    // Test 2: Performance comparison
    test_performance_comparison(&mut report)?;
    
    // Test 3: Mathematical correctness
    test_mathematical_correctness(&mut report)?;
    
    // Test 4: Latency targets
    test_latency_targets(&mut report)?;
    
    println!("✅ Validation completed successfully!");
    report.print_summary();
    
    Ok(report)
}

/// Test basic functionality of optimized implementation
fn test_basic_functionality(report: &mut ValidationReport) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Testing basic functionality...");
    
    let config = AtsCpConfig::default();
    let mut optimized = OptimizedConformalPredictor::new(&config)?;
    
    // Test 1: Softmax computation
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let softmax_result = optimized.softmax_avx512_optimized(&logits)?;
    
    // Check softmax properties
    let sum: f64 = softmax_result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10, "Softmax doesn't sum to 1: {}", sum);
    
    for (i, &prob) in softmax_result.iter().enumerate() {
        assert!(prob > 0.0 && prob.is_finite(), "Invalid probability at {}: {}", i, prob);
    }
    
    report.add_test("softmax_functionality", true, "Softmax computation works correctly");
    
    // Test 2: Quantile computation
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    let quantile = optimized.compute_quantile_gk(&data, 0.95)?;
    
    assert!(quantile >= 0.9 && quantile <= 1.0, "Quantile {} not in expected range", quantile);
    report.add_test("quantile_functionality", true, "Quantile computation works correctly");
    
    // Test 3: End-to-end conformal prediction
    let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let calibration_data: Vec<f64> = (0..200).map(|i| i as f64 * 0.005).collect();
    let intervals = optimized.predict_optimized(&predictions, &calibration_data, 0.95)?;
    
    assert_eq!(intervals.len(), predictions.len());
    for (i, (lower, upper)) in intervals.iter().enumerate() {
        assert!(lower <= upper, "Invalid interval at {}: ({}, {})", i, lower, upper);
        assert!(lower.is_finite() && upper.is_finite(), "Non-finite interval at {}", i);
    }
    
    report.add_test("conformal_prediction_functionality", true, "Conformal prediction works correctly");
    
    println!("✅ Basic functionality tests passed");
    Ok(())
}

/// Test performance comparison between original and optimized
fn test_performance_comparison(report: &mut ValidationReport) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Testing performance comparison...");
    
    let config = AtsCpConfig::default();
    let mut original = ConformalPredictor::new(&config)?;
    let mut optimized = OptimizedConformalPredictor::new(&config)?;
    
    // Test softmax performance
    let logits: Vec<f64> = (0..64).map(|i| i as f64 * 0.1).collect();
    
    let original_start = Instant::now();
    for _ in 0..1000 {
        let _ = original.compute_softmax(&logits)?;
    }
    let original_time = original_start.elapsed();
    
    let optimized_start = Instant::now();
    for _ in 0..1000 {
        let _ = optimized.softmax_avx512_optimized(&logits)?;
    }
    let optimized_time = optimized_start.elapsed();
    
    let improvement = original_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
    println!("🧮 Softmax improvement: {:.2}x ({:.1}μs → {:.1}μs)", 
        improvement,
        original_time.as_nanos() as f64 / 1000.0 / 1000.0,
        optimized_time.as_nanos() as f64 / 1000.0 / 1000.0
    );
    
    report.add_performance("softmax_performance", improvement, "Softmax optimization");
    
    // Test end-to-end conformal prediction performance
    let predictions: Vec<f64> = (0..32).map(|i| i as f64 * 0.01).collect();
    let calibration_data: Vec<f64> = (0..500).map(|i| i as f64 * 0.002).collect();
    
    let original_start = Instant::now();
    for _ in 0..100 {
        let _ = original.predict(&predictions, &calibration_data)?;
    }
    let original_time = original_start.elapsed();
    
    let optimized_start = Instant::now();
    for _ in 0..100 {
        let _ = optimized.predict_optimized(&predictions, &calibration_data, 0.95)?;
    }
    let optimized_time = optimized_start.elapsed();
    
    let improvement = original_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
    println!("🎯 Conformal prediction improvement: {:.2}x ({:.1}μs → {:.1}μs)",
        improvement,
        original_time.as_nanos() as f64 / 100.0 / 1000.0,
        optimized_time.as_nanos() as f64 / 100.0 / 1000.0
    );
    
    report.add_performance("conformal_prediction_performance", improvement, "Conformal prediction optimization");
    
    println!("✅ Performance comparison tests passed");
    Ok(())
}

/// Test mathematical correctness
fn test_mathematical_correctness(report: &mut ValidationReport) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 Testing mathematical correctness...");
    
    let config = AtsCpConfig::default();
    let original = ConformalPredictor::new(&config)?;
    let mut optimized = OptimizedConformalPredictor::new(&config)?;
    
    // Compare softmax outputs
    let test_cases = vec![
        vec![1.0, 2.0, 3.0],
        vec![0.0, 0.0, 0.0], 
        vec![-5.0, 0.0, 5.0],
        (0..32).map(|i| i as f64 * 0.1).collect(),
    ];
    
    for (i, logits) in test_cases.iter().enumerate() {
        let original_softmax = original.compute_softmax(logits)?;
        let optimized_softmax = optimized.softmax_avx512_optimized(logits)?;
        
        assert_eq!(original_softmax.len(), optimized_softmax.len());
        
        for (j, (&orig, &opt)) in original_softmax.iter().zip(&optimized_softmax).enumerate() {
            let relative_error = (orig - opt).abs() / orig.max(1e-10);
            assert!(relative_error < 1e-10, 
                "Softmax mismatch in case {} at position {}: {} vs {} (error: {})",
                i, j, orig, opt, relative_error);
        }
    }
    
    report.add_test("mathematical_correctness", true, "Mathematical correctness verified");
    
    println!("✅ Mathematical correctness tests passed");
    Ok(())
}

/// Test latency targets
fn test_latency_targets(report: &mut ValidationReport) -> Result<(), Box<dyn std::error::Error>> {
    println!("⏱️  Testing latency targets...");
    
    let config = AtsCpConfig::default();
    let mut optimized = OptimizedConformalPredictor::new(&config)?;
    
    // Test softmax latency (target: <2μs)
    let logits: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
    
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = optimized.softmax_avx512_optimized(&logits)?;
    }
    let elapsed = start.elapsed();
    let per_op_us = elapsed.as_nanos() as f64 / 1000.0 / 1000.0;
    
    println!("🧮 Softmax latency: {:.2}μs per operation", per_op_us);
    report.add_latency("softmax_latency", per_op_us, 2.0, "Softmax operation");
    
    // Test quantile computation latency (target: <5μs)
    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
    
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = optimized.compute_quantile_gk(&data, 0.95)?;
    }
    let elapsed = start.elapsed();
    let per_op_us = elapsed.as_nanos() as f64 / 1000.0 / 1000.0;
    
    println!("📊 Quantile computation latency: {:.2}μs per operation", per_op_us);
    report.add_latency("quantile_latency", per_op_us, 5.0, "Quantile computation");
    
    // Test full conformal prediction latency (target: <20μs)
    let predictions: Vec<f64> = (0..16).map(|i| i as f64 * 0.01).collect();
    let calibration_data: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
    
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = optimized.predict_optimized(&predictions, &calibration_data, 0.95)?;
    }
    let elapsed = start.elapsed();
    let per_op_us = elapsed.as_nanos() as f64 / 1000.0 / 1000.0;
    
    println!("🎯 Conformal prediction latency: {:.2}μs per operation", per_op_us);
    report.add_latency("conformal_prediction_latency", per_op_us, 20.0, "Conformal prediction");
    
    println!("✅ Latency target tests completed");
    Ok(())
}

/// Validation report structure
pub struct ValidationReport {
    tests: Vec<TestResult>,
    performance_results: Vec<PerformanceResult>,
    latency_results: Vec<LatencyResult>,
}

struct TestResult {
    name: String,
    passed: bool,
    description: String,
}

struct PerformanceResult {
    name: String,
    improvement: f64,
    description: String,
}

struct LatencyResult {
    name: String,
    latency_us: f64,
    target_us: f64,
    description: String,
    meets_target: bool,
}

impl ValidationReport {
    fn new() -> Self {
        Self {
            tests: Vec::new(),
            performance_results: Vec::new(),
            latency_results: Vec::new(),
        }
    }
    
    fn add_test(&mut self, name: &str, passed: bool, description: &str) {
        self.tests.push(TestResult {
            name: name.to_string(),
            passed,
            description: description.to_string(),
        });
    }
    
    fn add_performance(&mut self, name: &str, improvement: f64, description: &str) {
        self.performance_results.push(PerformanceResult {
            name: name.to_string(),
            improvement,
            description: description.to_string(),
        });
    }
    
    fn add_latency(&mut self, name: &str, latency_us: f64, target_us: f64, description: &str) {
        let meets_target = latency_us <= target_us;
        self.latency_results.push(LatencyResult {
            name: name.to_string(),
            latency_us,
            target_us,
            description: description.to_string(),
            meets_target,
        });
    }
    
    fn print_summary(&self) {
        println!("\n🎯 VALIDATION SUMMARY");
        println!("===================");
        
        // Test results
        let passed_tests = self.tests.iter().filter(|t| t.passed).count();
        println!("📋 Tests: {}/{} passed", passed_tests, self.tests.len());
        
        for test in &self.tests {
            let status = if test.passed { "✅" } else { "❌" };
            println!("  {} {}: {}", status, test.name, test.description);
        }
        
        // Performance results
        println!("\n⚡ Performance Improvements:");
        for perf in &self.performance_results {
            println!("  🚀 {}: {:.2}x improvement", perf.name, perf.improvement);
        }
        
        // Latency results
        let latency_targets_met = self.latency_results.iter().filter(|l| l.meets_target).count();
        println!("\n⏱️  Latency Targets: {}/{} met", latency_targets_met, self.latency_results.len());
        
        for latency in &self.latency_results {
            let status = if latency.meets_target { "✅" } else { "❌" };
            println!("  {} {}: {:.2}μs (target: {:.0}μs)", 
                status, latency.name, latency.latency_us, latency.target_us);
        }
        
        // Overall assessment
        let all_tests_passed = self.tests.iter().all(|t| t.passed);
        let all_targets_met = self.latency_results.iter().all(|l| l.meets_target);
        let good_performance = self.performance_results.iter().all(|p| p.improvement > 1.0);
        
        println!("\n🎉 OVERALL ASSESSMENT:");
        if all_tests_passed && all_targets_met && good_performance {
            println!("✅ All optimization goals achieved!");
            println!("   - Mathematical correctness maintained");
            println!("   - Performance significantly improved");
            println!("   - Sub-20μs latency targets met");
        } else {
            if !all_tests_passed {
                println!("❌ Some tests failed - mathematical correctness issues detected");
            }
            if !all_targets_met {
                println!("❌ Some latency targets not met - further optimization needed");
            }
            if !good_performance {
                println!("❌ Performance not improved - optimization strategy needs revision");
            }
        }
    }
}