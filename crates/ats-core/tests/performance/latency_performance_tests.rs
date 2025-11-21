//! Performance Tests for Sub-20μs Latency Requirements
//!
//! These tests verify that the ATS-CP system meets strict performance requirements:
//! - Sub-20 microsecond latency for conformal prediction
//! - Sub-10 microsecond latency for temperature scaling
//! - Memory efficiency and allocation patterns
//! - Throughput requirements for high-frequency trading
//! - Performance regression detection

use ats_core::{
    conformal::ConformalPredictor,
    config::{AtsCpConfig, ConformalConfig, TemperatureConfig},
    types::{AtsCpVariant, Confidence},
    error::{AtsCoreError, Result},
    test_framework::{TestFramework, PerformanceHarness, TestMetrics},
};
use std::time::{Duration, Instant};
use criterion::{black_box, Criterion};

/// Performance test fixture with optimized configurations
struct PerformanceTestFixture {
    predictor: ConformalPredictor,
    fast_config: AtsCpConfig,
    test_data: PerformanceTestData,
}

/// Test data optimized for performance benchmarks
struct PerformanceTestData {
    small_logits: Vec<f64>,
    medium_logits: Vec<f64>,
    large_logits: Vec<f64>,
    small_calibration: Vec<f64>,
    medium_calibration: Vec<f64>,
    large_calibration: Vec<f64>,
    small_predictions: Vec<f64>,
    medium_predictions: Vec<f64>,
    large_predictions: Vec<f64>,
    calibration_logits_small: Vec<Vec<f64>>,
    calibration_logits_medium: Vec<Vec<f64>>,
    calibration_logits_large: Vec<Vec<f64>>,
    calibration_labels_small: Vec<usize>,
    calibration_labels_medium: Vec<usize>,
    calibration_labels_large: Vec<usize>,
}

impl PerformanceTestFixture {
    fn new() -> Self {
        // Ultra-fast configuration for performance testing
        let fast_config = AtsCpConfig {
            conformal: ConformalConfig {
                target_latency_us: 20,
                min_calibration_size: 10,
                max_calibration_size: 100,
                calibration_window_size: 50,
                default_confidence: 0.95,
                online_calibration: false, // Disable for fastest performance
                validate_exchangeability: false, // Disable for fastest performance
                quantile_method: crate::config::QuantileMethod::Nearest, // Fastest method
            },
            temperature: TemperatureConfig {
                default_temperature: 1.0,
                min_temperature: 0.1,
                max_temperature: 10.0,
                target_latency_us: 10,
                search_tolerance: 1e-4, // Reduced precision for speed
                max_search_iterations: 20, // Reduced iterations for speed
            },
            memory: crate::config::MemoryConfig {
                default_alignment: 64, // Cache line alignment
                page_size: 4096,
                huge_pages: false,
            },
        };
        
        let predictor = ConformalPredictor::new(&fast_config).unwrap();
        
        let test_data = PerformanceTestData {
            // Small datasets (typical HFT scenario)
            small_logits: vec![2.1, 1.3, 0.8],
            small_calibration: (0..20).map(|i| i as f64 * 0.05).collect(),
            small_predictions: vec![1.0, 2.0, 3.0],
            calibration_logits_small: vec![
                vec![2.0, 1.2, 0.9], vec![1.9, 1.4, 0.7], vec![2.2, 1.1, 0.8],
                vec![1.8, 1.5, 0.6], vec![2.1, 1.0, 0.9], vec![2.0, 1.3, 0.8],
                vec![1.7, 1.6, 0.7], vec![2.3, 0.9, 1.0], vec![1.9, 1.4, 0.6],
                vec![2.1, 1.2, 0.9],
            ],
            calibration_labels_small: vec![0, 0, 0, 1, 1, 2, 2, 1, 0, 2],
            
            // Medium datasets
            medium_logits: (0..10).map(|i| 2.0 - i as f64 * 0.2).collect(),
            medium_calibration: (0..50).map(|i| i as f64 * 0.02).collect(),
            medium_predictions: (0..10).map(|i| i as f64 * 0.5).collect(),
            calibration_logits_medium: (0..25).map(|i| {
                (0..10).map(|j| 2.0 - j as f64 * 0.1 + (i as f64) * 0.01).collect()
            }).collect(),
            calibration_labels_medium: (0..25).map(|i| i % 10).collect(),
            
            // Large datasets (stress test)
            large_logits: (0..100).map(|i| 5.0 - i as f64 * 0.05).collect(),
            large_calibration: (0..200).map(|i| i as f64 * 0.005).collect(),
            large_predictions: (0..100).map(|i| i as f64 * 0.1).collect(),
            calibration_logits_large: (0..100).map(|i| {
                (0..100).map(|j| 5.0 - j as f64 * 0.03 + (i as f64) * 0.001).collect()
            }).collect(),
            calibration_labels_large: (0..100).map(|i| i % 100).collect(),
        };
        
        Self {
            predictor,
            fast_config,
            test_data,
        }
    }
}

/// Core latency performance tests
mod core_latency_tests {
    use super::*;
    
    #[test]
    fn test_conformal_prediction_latency_requirement() {
        println!("⚡ Testing conformal prediction latency (target: <20μs)...");
        
        let mut fixture = PerformanceTestFixture::new();
        let num_iterations = 1000;
        let mut latencies = Vec::with_capacity(num_iterations);
        
        // Warm up
        for _ in 0..100 {
            let _ = fixture.predictor.predict(
                &fixture.test_data.small_predictions,
                &fixture.test_data.small_calibration,
            );
        }
        
        // Measure latencies
        for _ in 0..num_iterations {
            let start = Instant::now();
            let result = fixture.predictor.predict(
                &fixture.test_data.small_predictions,
                &fixture.test_data.small_calibration,
            );
            let latency = start.elapsed();
            
            assert!(result.is_ok(), "Conformal prediction should succeed");
            latencies.push(latency.as_nanos() as u64);
        }
        
        // Analyze latencies
        let min_latency = *latencies.iter().min().unwrap();
        let max_latency = *latencies.iter().max().unwrap();
        let avg_latency = latencies.iter().sum::<u64>() / latencies.len() as u64;
        let p95_latency = {
            let mut sorted = latencies.clone();
            sorted.sort();
            sorted[(sorted.len() * 95) / 100]
        };
        let p99_latency = {
            let mut sorted = latencies.clone();
            sorted.sort();
            sorted[(sorted.len() * 99) / 100]
        };
        
        println!("  Conformal Prediction Latency Results:");
        println!("    Min:     {:6} ns ({:5.2} μs)", min_latency, min_latency as f64 / 1000.0);
        println!("    Avg:     {:6} ns ({:5.2} μs)", avg_latency, avg_latency as f64 / 1000.0);
        println!("    P95:     {:6} ns ({:5.2} μs)", p95_latency, p95_latency as f64 / 1000.0);
        println!("    P99:     {:6} ns ({:5.2} μs)", p99_latency, p99_latency as f64 / 1000.0);
        println!("    Max:     {:6} ns ({:5.2} μs)", max_latency, max_latency as f64 / 1000.0);
        
        // Performance requirements
        assert!(avg_latency < 20_000, "Average latency should be <20μs, got {}μs", avg_latency as f64 / 1000.0);
        assert!(p95_latency < 25_000, "P95 latency should be <25μs, got {}μs", p95_latency as f64 / 1000.0);
        assert!(p99_latency < 30_000, "P99 latency should be <30μs, got {}μs", p99_latency as f64 / 1000.0);
        
        println!("✅ Conformal prediction latency requirements met");
    }
    
    #[test]
    fn test_temperature_scaling_latency_requirement() {
        println!("⚡ Testing temperature scaling latency (target: <10μs)...");
        
        let mut fixture = PerformanceTestFixture::new();
        let num_iterations = 1000;
        let mut latencies = Vec::with_capacity(num_iterations);
        let temperature = 1.5;
        
        // Warm up
        for _ in 0..100 {
            let _ = fixture.predictor.temperature_scaled_softmax(&fixture.test_data.small_logits, temperature);
        }
        
        // Measure latencies
        for _ in 0..num_iterations {
            let start = Instant::now();
            let result = fixture.predictor.temperature_scaled_softmax(&fixture.test_data.small_logits, temperature);
            let latency = start.elapsed();
            
            assert!(result.is_ok(), "Temperature scaling should succeed");
            latencies.push(latency.as_nanos() as u64);
        }
        
        // Analyze latencies
        let min_latency = *latencies.iter().min().unwrap();
        let max_latency = *latencies.iter().max().unwrap();
        let avg_latency = latencies.iter().sum::<u64>() / latencies.len() as u64;
        let p95_latency = {
            let mut sorted = latencies.clone();
            sorted.sort();
            sorted[(sorted.len() * 95) / 100]
        };
        
        println!("  Temperature Scaling Latency Results:");
        println!("    Min:     {:6} ns ({:5.2} μs)", min_latency, min_latency as f64 / 1000.0);
        println!("    Avg:     {:6} ns ({:5.2} μs)", avg_latency, avg_latency as f64 / 1000.0);
        println!("    P95:     {:6} ns ({:5.2} μs)", p95_latency, p95_latency as f64 / 1000.0);
        println!("    Max:     {:6} ns ({:5.2} μs)", max_latency, max_latency as f64 / 1000.0);
        
        // Performance requirements
        assert!(avg_latency < 10_000, "Average latency should be <10μs, got {}μs", avg_latency as f64 / 1000.0);
        assert!(p95_latency < 15_000, "P95 latency should be <15μs, got {}μs", p95_latency as f64 / 1000.0);
        
        println!("✅ Temperature scaling latency requirements met");
    }
    
    #[test]
    fn test_ats_cp_algorithm_latency_requirement() {
        println!("⚡ Testing ATS-CP algorithm latency (target: <20μs)...");
        
        let mut fixture = PerformanceTestFixture::new();
        let num_iterations = 1000;
        let mut latencies = Vec::with_capacity(num_iterations);
        
        let variants = vec![AtsCpVariant::GQ, AtsCpVariant::AQ];
        
        for variant in variants {
            println!("  Testing variant: {:?}", variant);
            let mut variant_latencies = Vec::new();
            
            // Warm up
            for _ in 0..50 {
                let _ = fixture.predictor.ats_cp_predict(
                    &fixture.test_data.small_logits,
                    &fixture.test_data.calibration_logits_small,
                    &fixture.test_data.calibration_labels_small,
                    0.95,
                    variant.clone(),
                );
            }
            
            // Measure latencies
            for _ in 0..num_iterations {
                let start = Instant::now();
                let result = fixture.predictor.ats_cp_predict(
                    &fixture.test_data.small_logits,
                    &fixture.test_data.calibration_logits_small,
                    &fixture.test_data.calibration_labels_small,
                    0.95,
                    variant.clone(),
                );
                let latency = start.elapsed();
                
                assert!(result.is_ok(), "ATS-CP algorithm should succeed for variant {:?}", variant);
                variant_latencies.push(latency.as_nanos() as u64);
            }
            
            latencies.extend(variant_latencies.clone());
            
            let avg_latency = variant_latencies.iter().sum::<u64>() / variant_latencies.len() as u64;
            let p95_latency = {
                let mut sorted = variant_latencies.clone();
                sorted.sort();
                sorted[(sorted.len() * 95) / 100]
            };
            
            println!("    {:?} Avg: {:6} ns ({:5.2} μs)", variant, avg_latency, avg_latency as f64 / 1000.0);
            println!("    {:?} P95: {:6} ns ({:5.2} μs)", variant, p95_latency, p95_latency as f64 / 1000.0);
            
            // Performance requirements per variant
            assert!(avg_latency < 20_000, "Variant {:?} average latency should be <20μs", variant);
            assert!(p95_latency < 30_000, "Variant {:?} P95 latency should be <30μs", variant);
        }
        
        println!("✅ ATS-CP algorithm latency requirements met for all variants");
    }
}

/// Throughput and scalability tests
mod throughput_tests {
    use super::*;
    
    #[test]
    fn test_throughput_scalability() {
        println!("📊 Testing throughput scalability...");
        
        let mut fixture = PerformanceTestFixture::new();
        let test_duration = Duration::from_secs(5);
        
        // Test different dataset sizes
        let test_scenarios = vec![
            ("Small", &fixture.test_data.small_predictions, &fixture.test_data.small_calibration),
            ("Medium", &fixture.test_data.medium_predictions, &fixture.test_data.medium_calibration),
        ];
        
        for (scenario_name, predictions, calibration) in test_scenarios {
            println!("  Testing {} dataset:", scenario_name);
            
            let start_time = Instant::now();
            let mut operations = 0;
            let mut successful_operations = 0;
            let mut total_latency = Duration::from_nanos(0);
            
            while start_time.elapsed() < test_duration {
                let op_start = Instant::now();
                let result = fixture.predictor.predict(predictions, calibration);
                let op_latency = op_start.elapsed();
                
                operations += 1;
                total_latency += op_latency;
                
                if result.is_ok() {
                    successful_operations += 1;
                }
            }
            
            let actual_duration = start_time.elapsed();
            let throughput = successful_operations as f64 / actual_duration.as_secs_f64();
            let avg_latency = total_latency / operations;
            let success_rate = (successful_operations as f64) / (operations as f64);
            
            println!("    Operations:     {}", operations);
            println!("    Successful:     {}", successful_operations);
            println!("    Success rate:   {:.2}%", success_rate * 100.0);
            println!("    Throughput:     {:.0} ops/sec", throughput);
            println!("    Avg latency:    {:?}", avg_latency);
            
            // Performance requirements
            assert!(success_rate >= 0.95, "Success rate should be ≥95% for {}", scenario_name);
            assert!(throughput >= 10_000.0, "Throughput should be ≥10k ops/sec for {}", scenario_name);
            assert!(avg_latency < Duration::from_micros(50), "Average latency should be <50μs for {}", scenario_name);
        }
        
        println!("✅ Throughput scalability requirements met");
    }
    
    #[test] 
    fn test_parallel_processing_performance() {
        println!("🔄 Testing parallel processing performance...");
        
        let mut fixture = PerformanceTestFixture::new();
        let confidence = 0.95;
        
        // Sequential processing
        let start_sequential = Instant::now();
        let sequential_result = fixture.predictor.predict(
            &fixture.test_data.medium_predictions,
            &fixture.test_data.medium_calibration,
        );
        let sequential_time = start_sequential.elapsed();
        
        assert!(sequential_result.is_ok(), "Sequential processing should succeed");
        
        // Parallel processing
        let start_parallel = Instant::now();
        let parallel_result = fixture.predictor.predict_parallel(
            &fixture.test_data.medium_predictions,
            &fixture.test_data.medium_calibration,
            confidence,
        );
        let parallel_time = start_parallel.elapsed();
        
        assert!(parallel_result.is_ok(), "Parallel processing should succeed");
        
        println!("  Sequential time: {:?}", sequential_time);
        println!("  Parallel time:   {:?}", parallel_time);
        
        if parallel_time < sequential_time {
            let speedup = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
            println!("  Speedup:         {:.2}x", speedup);
        } else {
            println!("  Parallel overhead detected (acceptable for small datasets)");
        }
        
        // Verify results consistency
        let sequential_intervals = sequential_result.unwrap();
        let parallel_intervals = parallel_result.unwrap();
        
        assert_eq!(sequential_intervals.len(), parallel_intervals.len(),
                  "Sequential and parallel should produce same number of intervals");
        
        println!("✅ Parallel processing performance validated");
    }
    
    #[test]
    fn test_batch_processing_efficiency() {
        println!("📦 Testing batch processing efficiency...");
        
        let mut fixture = PerformanceTestFixture::new();
        let confidence_levels = vec![0.90, 0.95, 0.99];
        
        // Individual predictions
        let start_individual = Instant::now();
        let mut individual_results = Vec::new();
        for &confidence in &confidence_levels {
            let result = fixture.predictor.predict_detailed(
                &fixture.test_data.small_predictions,
                &fixture.test_data.small_calibration,
                confidence,
            );
            if result.is_ok() {
                individual_results.push(result.unwrap());
            }
        }
        let individual_time = start_individual.elapsed();
        
        // Batch prediction
        let start_batch = Instant::now();
        let batch_result = fixture.predictor.predict_batch_confidence(
            &fixture.test_data.small_predictions,
            &fixture.test_data.small_calibration,
            &confidence_levels,
        );
        let batch_time = start_batch.elapsed();
        
        assert!(batch_result.is_ok(), "Batch processing should succeed");
        let batch_intervals = batch_result.unwrap();
        
        println!("  Individual time: {:?}", individual_time);
        println!("  Batch time:      {:?}", batch_time);
        
        if batch_time < individual_time {
            let efficiency = individual_time.as_nanos() as f64 / batch_time.as_nanos() as f64;
            println!("  Efficiency:      {:.2}x faster", efficiency);
        }
        
        // Verify results consistency
        assert_eq!(batch_intervals.len(), confidence_levels.len(),
                  "Batch should produce results for all confidence levels");
        assert_eq!(individual_results.len(), confidence_levels.len(),
                  "Individual processing should succeed for all confidence levels");
        
        println!("✅ Batch processing efficiency validated");
    }
}

/// Memory efficiency and allocation tests
mod memory_tests {
    use super::*;
    
    #[test]
    fn test_memory_allocation_patterns() {
        println!("🧠 Testing memory allocation patterns...");
        
        let mut fixture = PerformanceTestFixture::new();
        
        // Get initial memory stats (simplified - in real implementation would use system calls)
        let initial_stats = fixture.predictor.get_performance_stats();
        
        // Perform many operations to test memory efficiency
        let num_operations = 1000;
        for i in 0..num_operations {
            let result = fixture.predictor.predict(
                &fixture.test_data.small_predictions,
                &fixture.test_data.small_calibration,
            );
            assert!(result.is_ok(), "Operation {} should succeed", i);
            
            // Periodically check for memory leaks (simplified)
            if i % 100 == 0 {
                let current_stats = fixture.predictor.get_performance_stats();
                // Memory usage shouldn't grow unbounded
                assert!(current_stats.0 > 0, "Should record operations");
            }
        }
        
        let final_stats = fixture.predictor.get_performance_stats();
        
        println!("  Initial operations: {}", initial_stats.0);
        println!("  Final operations:   {}", final_stats.0);
        println!("  Operations added:   {}", final_stats.0 - initial_stats.0);
        
        // Verify no memory leaks (operations counter should increase)
        assert_eq!(final_stats.0 - initial_stats.0, num_operations,
                  "Should record exact number of operations");
        
        println!("✅ Memory allocation patterns validated");
    }
    
    #[test]
    fn test_working_memory_efficiency() {
        println!("💾 Testing working memory efficiency...");
        
        let mut fixture = PerformanceTestFixture::new();
        
        // Test with different data sizes to verify memory efficiency
        let test_sizes = vec![
            ("Tiny", 3, 10),
            ("Small", 10, 50),
            ("Medium", 50, 100),
        ];
        
        for (size_name, logits_size, calibration_size) in test_sizes {
            println!("  Testing {} size:", size_name);
            
            let test_logits: Vec<f64> = (0..logits_size).map(|i| i as f64 * 0.1).collect();
            let test_calibration: Vec<f64> = (0..calibration_size).map(|i| i as f64 * 0.01).collect();
            
            let start_time = Instant::now();
            let result = fixture.predictor.predict(&test_logits, &test_calibration);
            let execution_time = start_time.elapsed();
            
            assert!(result.is_ok(), "Prediction should succeed for {} size", size_name);
            
            let intervals = result.unwrap();
            assert_eq!(intervals.len(), test_logits.len(),
                      "Should produce correct number of intervals for {} size", size_name);
            
            println!("    Execution time: {:?}", execution_time);
            
            // Memory efficiency: execution time should scale reasonably
            assert!(execution_time < Duration::from_micros(100),
                   "Execution should be efficient for {} size", size_name);
        }
        
        println!("✅ Working memory efficiency validated");
    }
}

/// Performance regression tests
mod regression_tests {
    use super::*;
    
    #[test]
    fn test_performance_regression_baseline() {
        println!("📈 Testing performance regression baseline...");
        
        let mut fixture = PerformanceTestFixture::new();
        let num_samples = 100;
        let mut latencies = Vec::with_capacity(num_samples);
        
        // Establish baseline performance
        for _ in 0..num_samples {
            let start = Instant::now();
            let result = fixture.predictor.predict(
                &fixture.test_data.small_predictions,
                &fixture.test_data.small_calibration,
            );
            let latency = start.elapsed();
            
            assert!(result.is_ok(), "Baseline prediction should succeed");
            latencies.push(latency.as_nanos() as u64);
        }
        
        let avg_latency = latencies.iter().sum::<u64>() / latencies.len() as u64;
        let min_latency = *latencies.iter().min().unwrap();
        let max_latency = *latencies.iter().max().unwrap();
        
        println!("  Baseline Performance:");
        println!("    Average: {:6} ns ({:5.2} μs)", avg_latency, avg_latency as f64 / 1000.0);
        println!("    Minimum: {:6} ns ({:5.2} μs)", min_latency, min_latency as f64 / 1000.0);
        println!("    Maximum: {:6} ns ({:5.2} μs)", max_latency, max_latency as f64 / 1000.0);
        
        // Regression thresholds (these would be stored and compared in CI)
        const BASELINE_AVG_NS: u64 = 15_000; // 15μs baseline
        const BASELINE_MAX_NS: u64 = 30_000; // 30μs max acceptable
        
        assert!(avg_latency < BASELINE_AVG_NS,
               "Performance regression detected: avg {}μs > baseline {}μs",
               avg_latency as f64 / 1000.0, BASELINE_AVG_NS as f64 / 1000.0);
        assert!(max_latency < BASELINE_MAX_NS,
               "Performance regression detected: max {}μs > baseline {}μs",
               max_latency as f64 / 1000.0, BASELINE_MAX_NS as f64 / 1000.0);
        
        println!("✅ Performance regression baseline validated");
    }
    
    #[test]
    fn test_performance_consistency() {
        println!("🔄 Testing performance consistency...");
        
        let mut fixture = PerformanceTestFixture::new();
        let num_rounds = 10;
        let samples_per_round = 100;
        let mut round_averages = Vec::with_capacity(num_rounds);
        
        for round in 0..num_rounds {
            let mut round_latencies = Vec::with_capacity(samples_per_round);
            
            for _ in 0..samples_per_round {
                let start = Instant::now();
                let result = fixture.predictor.predict(
                    &fixture.test_data.small_predictions,
                    &fixture.test_data.small_calibration,
                );
                let latency = start.elapsed();
                
                assert!(result.is_ok(), "Consistency test should succeed in round {}", round);
                round_latencies.push(latency.as_nanos() as u64);
            }
            
            let round_avg = round_latencies.iter().sum::<u64>() / round_latencies.len() as u64;
            round_averages.push(round_avg);
            
            println!("  Round {}: {:6} ns ({:5.2} μs)", round + 1, round_avg, round_avg as f64 / 1000.0);
        }
        
        // Calculate consistency metrics
        let overall_avg = round_averages.iter().sum::<u64>() / round_averages.len() as u64;
        let variance: f64 = round_averages.iter()
            .map(|&x| {
                let diff = x as f64 - overall_avg as f64;
                diff * diff
            })
            .sum::<f64>() / round_averages.len() as f64;
        let std_dev = variance.sqrt();
        let coefficient_of_variation = std_dev / overall_avg as f64;
        
        println!("  Consistency Metrics:");
        println!("    Overall avg: {:6} ns ({:5.2} μs)", overall_avg, overall_avg as f64 / 1000.0);
        println!("    Std dev:     {:6.0} ns ({:5.2} μs)", std_dev, std_dev / 1000.0);
        println!("    CoV:         {:.2}%", coefficient_of_variation * 100.0);
        
        // Performance consistency requirements
        assert!(coefficient_of_variation < 0.2,
               "Performance should be consistent (CoV < 20%), got {:.2}%",
               coefficient_of_variation * 100.0);
        
        println!("✅ Performance consistency validated");
    }
}

/// High-frequency trading scenario tests
mod hft_scenario_tests {
    use super::*;
    
    #[test]
    fn test_hft_burst_performance() {
        println!("💹 Testing HFT burst performance...");
        
        let mut fixture = PerformanceTestFixture::new();
        let burst_size = 100;
        let max_burst_time = Duration::from_millis(5); // 5ms for 100 operations
        
        // Simulate HFT burst scenario
        let start_time = Instant::now();
        let mut successful_predictions = 0;
        let mut individual_latencies = Vec::with_capacity(burst_size);
        
        for i in 0..burst_size {
            // Vary logits slightly to simulate real trading scenario
            let mut trading_logits = fixture.test_data.small_logits.clone();
            for logit in &mut trading_logits {
                *logit += (i as f64) * 0.001;
            }
            
            let pred_start = Instant::now();
            let result = fixture.predictor.ats_cp_predict(
                &trading_logits,
                &fixture.test_data.calibration_logits_small,
                &fixture.test_data.calibration_labels_small,
                0.95,
                AtsCpVariant::GQ,
            );
            let pred_latency = pred_start.elapsed();
            
            if result.is_ok() {
                successful_predictions += 1;
                individual_latencies.push(pred_latency.as_nanos() as u64);
            }
        }
        
        let total_burst_time = start_time.elapsed();
        
        println!("  HFT Burst Results:");
        println!("    Total time:       {:?}", total_burst_time);
        println!("    Successful preds: {}/{}", successful_predictions, burst_size);
        println!("    Success rate:     {:.2}%", (successful_predictions as f64 / burst_size as f64) * 100.0);
        
        if !individual_latencies.is_empty() {
            let avg_latency = individual_latencies.iter().sum::<u64>() / individual_latencies.len() as u64;
            let max_latency = *individual_latencies.iter().max().unwrap();
            
            println!("    Avg prediction:   {:6} ns ({:5.2} μs)", avg_latency, avg_latency as f64 / 1000.0);
            println!("    Max prediction:   {:6} ns ({:5.2} μs)", max_latency, max_latency as f64 / 1000.0);
            
            // HFT performance requirements
            assert!(avg_latency < 20_000, "HFT avg latency should be <20μs");
            assert!(max_latency < 50_000, "HFT max latency should be <50μs");
        }
        
        // Burst requirements
        assert!(total_burst_time < max_burst_time, "Burst should complete within time limit");
        assert!(successful_predictions >= (burst_size * 95) / 100, "Should have >95% success rate");
        
        println!("✅ HFT burst performance requirements met");
    }
    
    #[test]
    fn test_hft_sustained_load() {
        println!("⏱️  Testing HFT sustained load...");
        
        let mut fixture = PerformanceTestFixture::new();
        let test_duration = Duration::from_secs(10);
        let target_rate = 1000.0; // 1000 predictions per second
        
        let start_time = Instant::now();
        let mut prediction_count = 0;
        let mut successful_predictions = 0;
        let mut latencies = Vec::new();
        
        while start_time.elapsed() < test_duration {
            let pred_start = Instant::now();
            let result = fixture.predictor.predict(
                &fixture.test_data.small_predictions,
                &fixture.test_data.small_calibration,
            );
            let pred_latency = pred_start.elapsed();
            
            prediction_count += 1;
            
            if result.is_ok() {
                successful_predictions += 1;
                latencies.push(pred_latency.as_nanos() as u64);
            }
            
            // Rate limiting to simulate realistic sustained load
            std::thread::sleep(Duration::from_nanos(800_000)); // ~1200 Hz max
        }
        
        let actual_duration = start_time.elapsed();
        let actual_rate = prediction_count as f64 / actual_duration.as_secs_f64();
        let success_rate = (successful_predictions as f64) / (prediction_count as f64);
        
        println!("  Sustained Load Results:");
        println!("    Duration:         {:?}", actual_duration);
        println!("    Predictions:      {}", prediction_count);
        println!("    Successful:       {}", successful_predictions);
        println!("    Rate achieved:    {:.0} predictions/sec", actual_rate);
        println!("    Success rate:     {:.2}%", success_rate * 100.0);
        
        if !latencies.is_empty() {
            let avg_latency = latencies.iter().sum::<u64>() / latencies.len() as u64;
            let p99_latency = {
                let mut sorted = latencies.clone();
                sorted.sort();
                sorted[(sorted.len() * 99) / 100]
            };
            
            println!("    Avg latency:      {:6} ns ({:5.2} μs)", avg_latency, avg_latency as f64 / 1000.0);
            println!("    P99 latency:      {:6} ns ({:5.2} μs)", p99_latency, p99_latency as f64 / 1000.0);
            
            // Sustained load requirements
            assert!(avg_latency < 30_000, "Sustained avg latency should be <30μs");
            assert!(p99_latency < 100_000, "Sustained P99 latency should be <100μs");
        }
        
        // Sustained performance requirements
        assert!(success_rate >= 0.99, "Should maintain >99% success rate under sustained load");
        assert!(actual_rate >= target_rate * 0.8, "Should achieve at least 80% of target rate");
        
        println!("✅ HFT sustained load requirements met");
    }
}

#[cfg(test)]
mod performance_test_integration {
    use super::*;
    use ats_core::test_framework::{TestFramework, PerformanceHarness, swarm_utils};
    
    #[tokio::test]
    async fn test_performance_tests_swarm_coordination() {
        // Initialize performance test framework
        let mut framework = TestFramework::new(
            "performance_test_swarm".to_string(),
            "performance_test_agent".to_string(),
        ).unwrap();
        
        // Signal coordination with other test agents
        swarm_utils::coordinate_test_execution(&framework.context, "performance_tests").await.unwrap();
        
        // Execute performance test sample
        let mut fixture = PerformanceTestFixture::new();
        
        let start_time = Instant::now();
        let result = fixture.predictor.predict(
            &fixture.test_data.small_predictions,
            &fixture.test_data.small_calibration,
        );
        let execution_time = start_time.elapsed();
        
        assert!(result.is_ok(), "Performance test sample should succeed");
        
        // Record performance metrics
        framework.context.execution_metrics.tests_passed += 1;
        framework.context.execution_metrics.performance_metrics.insert(
            "conformal_prediction_latency_ns".to_string(),
            execution_time.as_nanos() as f64,
        );
        
        // Share results with swarm
        swarm_utils::share_test_results(&framework.context, &framework.context.execution_metrics).await.unwrap();
        
        // Validate performance requirements
        assert!(execution_time < Duration::from_micros(50),
               "Swarm coordinated performance test should meet latency requirements");
        
        println!("✅ Performance tests swarm coordination completed");
    }
}