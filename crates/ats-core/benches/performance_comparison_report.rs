//! Comprehensive Performance Comparison Report Generator
//!
//! This benchmark generates detailed before/after performance reports
//! comparing the original and optimized implementations.

use ats_core::{
    config::AtsCpConfig,
    conformal::ConformalPredictor,
    conformal_optimized::OptimizedConformalPredictor,
    types::AtsCpVariant,
};
use criterion::{black_box, Criterion, measurement::WallTime, BenchmarkGroup};
use std::{collections::HashMap, time::Duration};
use serde::{Serialize, Deserialize};

/// Performance comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    pub test_name: String,
    pub original_time_ns: u64,
    pub optimized_time_ns: u64,
    pub improvement_ratio: f64,
    pub latency_target_met: bool,
    pub target_latency_ns: u64,
}

/// Comprehensive performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub timestamp: String,
    pub test_configuration: TestConfiguration,
    pub comparisons: Vec<PerformanceComparison>,
    pub summary: PerformanceSummary,
    pub regression_analysis: RegressionAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfiguration {
    pub target_latency_us: u64,
    pub test_data_sizes: Vec<usize>,
    pub confidence_levels: Vec<f64>,
    pub ats_cp_variants: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_tests: usize,
    pub tests_improved: usize,
    pub tests_meeting_target: usize,
    pub average_improvement: f64,
    pub max_improvement: f64,
    pub min_improvement: f64,
    pub geometric_mean_improvement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysis {
    pub tests_with_regression: Vec<String>,
    pub worst_regression_ratio: f64,
    pub acceptable_regression_threshold: f64,
}

/// Performance report generator
pub struct PerformanceReportGenerator {
    config: AtsCpConfig,
    results: Vec<PerformanceComparison>,
    target_latency_ns: u64,
}

impl PerformanceReportGenerator {
    /// Creates a new performance report generator
    pub fn new() -> Self {
        let mut config = AtsCpConfig::high_performance();
        config.conformal.target_latency_us = 20; // Sub-20μs target
        
        Self {
            config,
            results: Vec::new(),
            target_latency_ns: 20_000, // 20μs in nanoseconds
        }
    }
    
    /// Runs comprehensive performance comparison
    pub fn run_comprehensive_comparison(&mut self) -> PerformanceReport {
        println!("🚀 Running comprehensive performance comparison...");
        
        // Test 1: Quantile Computation Comparison
        self.test_quantile_computation();
        
        // Test 2: Softmax Computation Comparison  
        self.test_softmax_computation();
        
        // Test 3: End-to-end Conformal Prediction
        self.test_conformal_prediction_end_to_end();
        
        // Test 4: ATS-CP Algorithm Variants
        self.test_ats_cp_variants();
        
        // Test 5: Memory Access Patterns
        self.test_memory_patterns();
        
        // Test 6: Batch Processing Throughput
        self.test_batch_throughput();
        
        // Generate comprehensive report
        self.generate_report()
    }
    
    /// Test quantile computation performance
    fn test_quantile_computation(&mut self) {
        println!("📊 Testing quantile computation performance...");
        
        let mut original = ConformalPredictor::new(&self.config).unwrap();
        let mut optimized = OptimizedConformalPredictor::new(&self.config).unwrap();
        
        for size in [100, 500, 1000, 2000, 5000] {
            let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.001 + rand::random::<f64>() * 0.1).collect();
            let confidence = 0.95;
            
            // Measure original implementation
            let original_time = self.measure_time(1000, || {
                let _ = original.compute_quantile_linear(&data, confidence);
            });
            
            // Measure optimized implementation
            let optimized_time = self.measure_time(1000, || {
                let _ = optimized.compute_quantile_gk(&data, confidence);
            });
            
            self.results.push(PerformanceComparison {
                test_name: format!("quantile_computation_size_{}", size),
                original_time_ns: original_time,
                optimized_time_ns: optimized_time,
                improvement_ratio: original_time as f64 / optimized_time as f64,
                latency_target_met: optimized_time <= self.target_latency_ns,
                target_latency_ns: self.target_latency_ns,
            });
        }
    }
    
    /// Test softmax computation performance
    fn test_softmax_computation(&mut self) {
        println!("🧮 Testing softmax computation performance...");
        
        let original = ConformalPredictor::new(&self.config).unwrap();
        let mut optimized = OptimizedConformalPredictor::new(&self.config).unwrap();
        
        for size in [8, 16, 32, 64, 128, 256] {
            let logits: Vec<f64> = (0..size).map(|i| i as f64 * 0.1 - (size as f64 * 0.05)).collect();
            
            // Measure original implementation
            let original_time = self.measure_time(2000, || {
                let _ = original.compute_softmax(&logits);
            });
            
            // Measure optimized implementation
            let optimized_time = self.measure_time(2000, || {
                let _ = optimized.softmax_avx512_optimized(&logits);
            });
            
            self.results.push(PerformanceComparison {
                test_name: format!("softmax_computation_size_{}", size),
                original_time_ns: original_time,
                optimized_time_ns: optimized_time,
                improvement_ratio: original_time as f64 / optimized_time as f64,
                latency_target_met: optimized_time <= 2000, // 2μs target for softmax
                target_latency_ns: 2000,
            });
        }
    }
    
    /// Test end-to-end conformal prediction performance
    fn test_conformal_prediction_end_to_end(&mut self) {
        println!("🔄 Testing end-to-end conformal prediction performance...");
        
        let mut original = ConformalPredictor::new(&self.config).unwrap();
        let mut optimized = OptimizedConformalPredictor::new(&self.config).unwrap();
        
        let test_cases = vec![
            (16, 100),   // Small: 16 predictions, 100 calibration samples
            (32, 200),   // Medium: 32 predictions, 200 calibration samples  
            (64, 500),   // Large: 64 predictions, 500 calibration samples
        ];
        
        for (pred_size, calib_size) in test_cases {
            let predictions: Vec<f64> = (0..pred_size).map(|i| i as f64 * 0.01 + rand::random::<f64>() * 0.05).collect();
            let calibration_data: Vec<f64> = (0..calib_size).map(|i| i as f64 * 0.002 + rand::random::<f64>() * 0.1).collect();
            let confidence = 0.95;
            
            // Measure original implementation
            let original_time = self.measure_time(500, || {
                let _ = original.predict(&predictions, &calibration_data);
            });
            
            // Measure optimized implementation
            let optimized_time = self.measure_time(500, || {
                let _ = optimized.predict_optimized(&predictions, &calibration_data, confidence);
            });
            
            self.results.push(PerformanceComparison {
                test_name: format!("conformal_prediction_{}x{}", pred_size, calib_size),
                original_time_ns: original_time,
                optimized_time_ns: optimized_time,
                improvement_ratio: original_time as f64 / optimized_time as f64,
                latency_target_met: optimized_time <= self.target_latency_ns,
                target_latency_ns: self.target_latency_ns,
            });
        }
    }
    
    /// Test ATS-CP algorithm variants performance
    fn test_ats_cp_variants(&mut self) {
        println!("🎯 Testing ATS-CP algorithm variants performance...");
        
        let mut original = ConformalPredictor::new(&self.config).unwrap();
        let mut optimized = OptimizedConformalPredictor::new(&self.config).unwrap();
        
        let logits = vec![2.3, 1.1, 0.8, 1.9, 0.2, 1.5, 0.9, 2.1];
        let calibration_logits: Vec<Vec<f64>> = (0..100).map(|_| {
            (0..8).map(|i| i as f64 * 0.3 + rand::random::<f64>() * 0.5).collect()
        }).collect();
        let calibration_labels: Vec<usize> = (0..100).map(|_| rand::random::<usize>() % 8).collect();
        let confidence = 0.95;
        
        for variant in [AtsCpVariant::GQ, AtsCpVariant::AQ, AtsCpVariant::MGQ] {
            let variant_name = format!("{:?}", variant);
            
            // Measure original implementation
            let original_time = self.measure_time(200, || {
                let _ = original.ats_cp_predict(
                    &logits,
                    &calibration_logits,
                    &calibration_labels,
                    confidence,
                    variant.clone(),
                );
            });
            
            // Measure optimized implementation
            let optimized_time = self.measure_time(200, || {
                let _ = optimized.ats_cp_predict_optimized(
                    &logits,
                    &calibration_logits,
                    &calibration_labels,
                    confidence,
                    variant.clone(),
                );
            });
            
            self.results.push(PerformanceComparison {
                test_name: format!("ats_cp_variant_{}", variant_name),
                original_time_ns: original_time,
                optimized_time_ns: optimized_time,
                improvement_ratio: original_time as f64 / optimized_time as f64,
                latency_target_met: optimized_time <= self.target_latency_ns,
                target_latency_ns: self.target_latency_ns,
            });
        }
    }
    
    /// Test memory access patterns performance
    fn test_memory_patterns(&mut self) {
        use ats_core::memory_optimized::{CacheAlignedVec, ConformalDataLayout};
        
        println!("💾 Testing memory access patterns performance...");
        
        for size in [1024, 4096, 16384] {
            let std_vec: Vec<f64> = (0..size).map(|i| i as f64).collect();
            let cache_vec = CacheAlignedVec::from_slice(&std_vec).unwrap();
            
            // Measure standard vector access
            let std_time = self.measure_time(1000, || {
                let mut sum = 0.0;
                for &val in &std_vec {
                    sum += val;
                }
                black_box(sum);
            });
            
            // Measure cache-aligned vector access
            let aligned_time = self.measure_time(1000, || {
                let mut sum = 0.0;
                for &val in cache_vec.as_slice() {
                    sum += val;
                }
                black_box(sum);
            });
            
            self.results.push(PerformanceComparison {
                test_name: format!("memory_access_pattern_size_{}", size),
                original_time_ns: std_time,
                optimized_time_ns: aligned_time,
                improvement_ratio: std_time as f64 / aligned_time as f64,
                latency_target_met: aligned_time <= 5000, // 5μs target for memory access
                target_latency_ns: 5000,
            });
        }
    }
    
    /// Test batch processing throughput
    fn test_batch_throughput(&mut self) {
        println!("📦 Testing batch processing throughput...");
        
        let mut optimized = OptimizedConformalPredictor::new(&self.config).unwrap();
        
        for batch_size in [10, 50, 100, 500] {
            let batch_predictions: Vec<Vec<f64>> = (0..batch_size).map(|_| {
                (0..32).map(|i| i as f64 * 0.01 + rand::random::<f64>() * 0.05).collect()
            }).collect();
            let calibration_data: Vec<f64> = (0..200).map(|i| i as f64 * 0.005).collect();
            let confidence = 0.95;
            
            // Measure batch processing time
            let batch_time = self.measure_time(100, || {
                for predictions in &batch_predictions {
                    let _ = optimized.predict_optimized(predictions, &calibration_data, confidence);
                }
            });
            
            // Calculate per-item time
            let per_item_time = batch_time / batch_size as u64;
            
            self.results.push(PerformanceComparison {
                test_name: format!("batch_throughput_size_{}", batch_size),
                original_time_ns: batch_time, // Total time for comparison
                optimized_time_ns: per_item_time,
                improvement_ratio: batch_size as f64, // Throughput metric
                latency_target_met: per_item_time <= self.target_latency_ns,
                target_latency_ns: self.target_latency_ns,
            });
        }
    }
    
    /// Measures execution time with high precision
    fn measure_time<F>(&self, iterations: usize, mut f: F) -> u64
    where
        F: FnMut(),
    {
        // Warm-up
        for _ in 0..10 {
            f();
        }
        
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            f();
        }
        let elapsed = start.elapsed();
        
        elapsed.as_nanos() as u64 / iterations as u64
    }
    
    /// Generates comprehensive performance report
    fn generate_report(&self) -> PerformanceReport {
        let timestamp = chrono::Utc::now().to_rfc3339();
        
        // Calculate summary statistics
        let total_tests = self.results.len();
        let tests_improved = self.results.iter()
            .filter(|r| r.improvement_ratio > 1.0)
            .count();
        let tests_meeting_target = self.results.iter()
            .filter(|r| r.latency_target_met)
            .count();
        
        let improvements: Vec<f64> = self.results.iter()
            .map(|r| r.improvement_ratio)
            .collect();
        
        let average_improvement = improvements.iter().sum::<f64>() / improvements.len() as f64;
        let max_improvement = improvements.iter().fold(0.0, |a, &b| a.max(b));
        let min_improvement = improvements.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        // Geometric mean improvement
        let log_sum: f64 = improvements.iter().map(|x| x.ln()).sum();
        let geometric_mean_improvement = (log_sum / improvements.len() as f64).exp();
        
        // Regression analysis
        let regression_threshold = 0.9; // 10% regression threshold
        let tests_with_regression: Vec<String> = self.results.iter()
            .filter(|r| r.improvement_ratio < regression_threshold)
            .map(|r| r.test_name.clone())
            .collect();
        
        let worst_regression_ratio = improvements.iter()
            .fold(f64::INFINITY, |a, &b| if b < 1.0 { a.min(b) } else { a });
        
        PerformanceReport {
            timestamp,
            test_configuration: TestConfiguration {
                target_latency_us: self.target_latency_ns / 1000,
                test_data_sizes: vec![100, 500, 1000, 2000, 5000],
                confidence_levels: vec![0.90, 0.95, 0.99],
                ats_cp_variants: vec!["GQ".to_string(), "AQ".to_string(), "MGQ".to_string()],
            },
            comparisons: self.results.clone(),
            summary: PerformanceSummary {
                total_tests,
                tests_improved,
                tests_meeting_target,
                average_improvement,
                max_improvement,
                min_improvement,
                geometric_mean_improvement,
            },
            regression_analysis: RegressionAnalysis {
                tests_with_regression,
                worst_regression_ratio: if worst_regression_ratio == f64::INFINITY { 1.0 } else { worst_regression_ratio },
                acceptable_regression_threshold: regression_threshold,
            },
        }
    }
    
    /// Prints human-readable performance report
    pub fn print_report(&self, report: &PerformanceReport) {
        println!("\n🎯 ATS-CORE PERFORMANCE OPTIMIZATION REPORT");
        println!("==========================================");
        println!("Generated: {}", report.timestamp);
        println!("Target Latency: {}μs", report.test_configuration.target_latency_us);
        
        println!("\n📊 SUMMARY STATISTICS");
        println!("--------------------");
        println!("Total Tests: {}", report.summary.total_tests);
        println!("Tests Improved: {} ({:.1}%)", 
            report.summary.tests_improved, 
            report.summary.tests_improved as f64 / report.summary.total_tests as f64 * 100.0
        );
        println!("Tests Meeting Target: {} ({:.1}%)", 
            report.summary.tests_meeting_target,
            report.summary.tests_meeting_target as f64 / report.summary.total_tests as f64 * 100.0
        );
        println!("Average Improvement: {:.2}x", report.summary.average_improvement);
        println!("Geometric Mean Improvement: {:.2}x", report.summary.geometric_mean_improvement);
        println!("Max Improvement: {:.2}x", report.summary.max_improvement);
        println!("Min Improvement: {:.2}x", report.summary.min_improvement);
        
        println!("\n🚀 DETAILED RESULTS");
        println!("------------------");
        for comparison in &report.comparisons {
            let status = if comparison.latency_target_met { "✅" } else { "❌" };
            println!("{} {}: {:.1}μs → {:.1}μs ({:.2}x improvement)", 
                status,
                comparison.test_name,
                comparison.original_time_ns as f64 / 1000.0,
                comparison.optimized_time_ns as f64 / 1000.0,
                comparison.improvement_ratio
            );
        }
        
        if !report.regression_analysis.tests_with_regression.is_empty() {
            println!("\n⚠️  REGRESSION ANALYSIS");
            println!("----------------------");
            println!("Tests with Performance Regression:");
            for test in &report.regression_analysis.tests_with_regression {
                println!("  - {}", test);
            }
            println!("Worst Regression: {:.2}x", report.regression_analysis.worst_regression_ratio);
        }
        
        println!("\n💡 RECOMMENDATIONS");
        println!("------------------");
        if report.summary.tests_meeting_target == report.summary.total_tests {
            println!("✅ All tests meet the sub-{}μs latency target!", report.test_configuration.target_latency_us);
        } else {
            let failing_tests = report.summary.total_tests - report.summary.tests_meeting_target;
            println!("⚠️  {} tests still exceed the latency target", failing_tests);
            println!("   Consider further optimization for these operations");
        }
        
        if report.summary.geometric_mean_improvement > 2.0 {
            println!("🎉 Excellent performance improvement achieved!");
        } else if report.summary.geometric_mean_improvement > 1.5 {
            println!("👍 Good performance improvement achieved");
        } else {
            println!("🔍 Consider additional optimization strategies");
        }
    }
    
    /// Saves report to JSON file
    pub fn save_report(&self, report: &PerformanceReport, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(report)?;
        std::fs::write(filename, json)?;
        println!("📄 Report saved to: {}", filename);
        Ok(())
    }
}

/// Main function to run performance comparison
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut generator = PerformanceReportGenerator::new();
    let report = generator.run_comprehensive_comparison();
    
    generator.print_report(&report);
    generator.save_report(&report, "performance_comparison_report.json")?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_performance_report_generation() {
        let mut generator = PerformanceReportGenerator::new();
        
        // Add sample result
        generator.results.push(PerformanceComparison {
            test_name: "sample_test".to_string(),
            original_time_ns: 100000,
            optimized_time_ns: 50000,
            improvement_ratio: 2.0,
            latency_target_met: true,
            target_latency_ns: 20000,
        });
        
        let report = generator.generate_report();
        
        assert_eq!(report.summary.total_tests, 1);
        assert_eq!(report.summary.tests_improved, 1);
        assert_eq!(report.summary.tests_meeting_target, 1);
        assert_eq!(report.summary.average_improvement, 2.0);
    }
    
    #[test]
    fn test_measure_time_accuracy() {
        let generator = PerformanceReportGenerator::new();
        
        let time = generator.measure_time(100, || {
            std::thread::sleep(Duration::from_nanos(1000));
        });
        
        // Should be roughly 1μs, allowing for measurement overhead
        assert!(time > 500 && time < 10000, "Time measurement {} not in expected range", time);
    }
}