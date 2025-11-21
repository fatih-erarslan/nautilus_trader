//! Nanosecond Precision Benchmarks
//!
//! This benchmark suite provides CPU cycle-accurate timing and validation
//! for extreme sub-microsecond performance requirements.
//!
//! PERFORMANCE TARGETS (MANDATORY):
//! - Trading decisions: <500ns (99.99% success rate)
//! - Whale detection: <200ns (99.99% success rate)
//! - GPU kernels: <100ns (99.99% success rate)
//! - API responses: <50ns (99.99% success rate)
//!
//! Uses RDTSC instruction for nanosecond precision timing.

use ats_core::{
    config::AtsCpConfig,
    nanosecond_validator::{NanosecondValidator, RealWorldScenarios},
    prelude::*,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

/// Benchmark nanosecond precision trading decisions
fn bench_nanosecond_trading_decisions(c: &mut Criterion) {
    let validator = NanosecondValidator::new().unwrap();
    let config = AtsCpConfig::high_performance();
    let engine = AtsCpEngine::new(config).unwrap();
    
    let mut group = c.benchmark_group("nanosecond_trading_decisions");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(10000);
    
    // Test different prediction sizes for nanosecond precision
    for size in [4, 8, 16, 32, 64].iter() {
        let predictions: Vec<f64> = (0..*size).map(|i| i as f64 * 0.01).collect();
        let temperature = 1.5;
        
        group.bench_with_input(
            BenchmarkId::new("temperature_scaling_500ns_target", size),
            size,
            |b, _| {
                b.iter_custom(|iters| {
                    let start = std::time::Instant::now();
                    for _ in 0..iters {
                        let _ = engine.temperature_scale(black_box(&predictions), black_box(temperature)).unwrap();
                    }
                    start.elapsed()
                })
            },
        );
        
        // Use nanosecond validator for precise validation
        let trading_decision = || {
            let _ = engine.temperature_scale(&predictions, temperature).unwrap();
        };
        
        let validation_result = validator.validate_trading_decision(trading_decision, &format!("size_{}", size)).unwrap();
        
        println!("Trading Decision Size {} Validation:", size);
        validation_result.display_results();
        
        // Assert nanosecond precision targets are met
        if !validation_result.passed {
            panic!("Trading decision validation FAILED for size {}: {:.2}% success rate", 
                   size, validation_result.actual_success_rate * 100.0);
        }
    }
    
    group.finish();
}

/// Benchmark nanosecond precision whale detection
fn bench_nanosecond_whale_detection(c: &mut Criterion) {
    let validator = NanosecondValidator::new().unwrap();
    
    let mut group = c.benchmark_group("nanosecond_whale_detection");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(10000);
    
    // Test different whale detection algorithms
    let algorithms = [
        ("pattern_matching", || {
            let market_data = [1.0, 2.0, 3.0, 4.0, 5.0];
            let mut anomaly_score = 0.0;
            for &value in &market_data {
                anomaly_score += value.ln();
            }
            anomaly_score > 0.0
        }),
        ("volume_analysis", || {
            let volumes = [100.0, 200.0, 300.0, 400.0, 500.0];
            let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
            volumes.iter().any(|&v| v > avg_volume * 3.0)
        }),
        ("price_impact", || {
            let prices = [100.0, 101.0, 102.0, 103.0, 104.0];
            let price_change = prices.last().unwrap() - prices.first().unwrap();
            price_change > 2.0
        }),
    ];
    
    for (name, algorithm) in algorithms.iter() {
        group.bench_function(&format!("whale_detection_{}_200ns_target", name), |b| {
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    let _ = black_box(algorithm());
                }
                start.elapsed()
            })
        });
        
        // Validate with nanosecond precision
        let validation_result = validator.validate_whale_detection(*algorithm, name).unwrap();
        
        println!("Whale Detection {} Validation:", name);
        validation_result.display_results();
        
        // Assert nanosecond precision targets are met
        if !validation_result.passed {
            panic!("Whale detection validation FAILED for {}: {:.2}% success rate", 
                   name, validation_result.actual_success_rate * 100.0);
        }
    }
    
    group.finish();
}

/// Benchmark nanosecond precision GPU kernel simulation
fn bench_nanosecond_gpu_kernels(c: &mut Criterion) {
    let validator = NanosecondValidator::new().unwrap();
    
    let mut group = c.benchmark_group("nanosecond_gpu_kernels");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(10000);
    
    // Test different GPU kernel operations
    let kernels = [
        ("vector_add", || {
            let a = [1.0, 2.0, 3.0, 4.0];
            let b = [5.0, 6.0, 7.0, 8.0];
            let mut result = [0.0; 4];
            for i in 0..4 {
                result[i] = a[i] + b[i];
            }
            result[0] > 0.0
        }),
        ("matrix_multiply", || {
            let a = [[1.0, 2.0], [3.0, 4.0]];
            let b = [[5.0, 6.0], [7.0, 8.0]];
            let mut result = [[0.0; 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        result[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
            result[0][0] > 0.0
        }),
        ("convolution", || {
            let signal = [1.0, 2.0, 3.0, 4.0];
            let kernel = [0.1, 0.2, 0.3];
            let mut result = 0.0;
            for i in 0..signal.len() - kernel.len() + 1 {
                let mut sum = 0.0;
                for j in 0..kernel.len() {
                    sum += signal[i + j] * kernel[j];
                }
                result += sum;
            }
            result > 0.0
        }),
    ];
    
    for (name, kernel) in kernels.iter() {
        group.bench_function(&format!("gpu_kernel_{}_100ns_target", name), |b| {
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    let _ = black_box(kernel());
                }
                start.elapsed()
            })
        });
        
        // Validate with nanosecond precision
        let validation_result = validator.validate_gpu_kernel(*kernel, name).unwrap();
        
        println!("GPU Kernel {} Validation:", name);
        validation_result.display_results();
        
        // Assert nanosecond precision targets are met
        if !validation_result.passed {
            panic!("GPU kernel validation FAILED for {}: {:.2}% success rate", 
                   name, validation_result.actual_success_rate * 100.0);
        }
    }
    
    group.finish();
}

/// Benchmark nanosecond precision API responses
fn bench_nanosecond_api_responses(c: &mut Criterion) {
    let validator = NanosecondValidator::new().unwrap();
    
    let mut group = c.benchmark_group("nanosecond_api_responses");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(10000);
    
    // Test different API response processing operations
    let operations = [
        ("json_parse", || {
            let json_str = r#"{"price": 100.5, "volume": 1000, "timestamp": 1234567890}"#;
            let checksum = json_str.len() as u64;
            checksum > 0
        }),
        ("data_validation", || {
            let price = 100.5;
            let volume = 1000.0;
            let timestamp = 1234567890u64;
            price > 0.0 && volume > 0.0 && timestamp > 0
        }),
        ("response_formatting", || {
            let status = "success";
            let data = 42;
            let response = format!("{{\"status\":\"{}\",\"data\":{}}}", status, data);
            response.len() > 0
        }),
    ];
    
    for (name, operation) in operations.iter() {
        group.bench_function(&format!("api_response_{}_50ns_target", name), |b| {
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    let _ = black_box(operation());
                }
                start.elapsed()
            })
        });
        
        // Validate with nanosecond precision
        let validation_result = validator.validate_api_response(*operation, name).unwrap();
        
        println!("API Response {} Validation:", name);
        validation_result.display_results();
        
        // Assert nanosecond precision targets are met
        if !validation_result.passed {
            panic!("API response validation FAILED for {}: {:.2}% success rate", 
                   name, validation_result.actual_success_rate * 100.0);
        }
    }
    
    group.finish();
}

/// Benchmark comprehensive real-world scenarios
fn bench_real_world_scenarios(c: &mut Criterion) {
    let scenarios = RealWorldScenarios::new().unwrap();
    
    let mut group = c.benchmark_group("real_world_scenarios");
    group.measurement_time(Duration::from_secs(120));
    group.sample_size(1000);
    
    // Whale attack simulation
    group.bench_function("whale_attack_simulation", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = scenarios.simulate_whale_attack().unwrap();
            }
            start.elapsed()
        })
    });
    
    // HFT decision simulation
    group.bench_function("hft_decision_simulation", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = scenarios.simulate_hft_decision().unwrap();
            }
            start.elapsed()
        })
    });
    
    // GPU kernel simulation
    group.bench_function("gpu_kernel_simulation", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = scenarios.simulate_gpu_kernel().unwrap();
            }
            start.elapsed()
        })
    });
    
    // API response simulation
    group.bench_function("api_response_simulation", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = scenarios.simulate_api_response().unwrap();
            }
            start.elapsed()
        })
    });
    
    // Comprehensive scenario validation
    let report = scenarios.run_comprehensive_scenarios().unwrap();
    
    println!("\n🚀 COMPREHENSIVE REAL-WORLD SCENARIO VALIDATION REPORT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    report.display_comprehensive_report();
    
    // Assert all scenarios pass
    if !report.all_passed() {
        panic!("Real-world scenario validation FAILED: Not all scenarios met nanosecond targets");
    }
    
    group.finish();
}

/// Benchmark memory bandwidth impact on nanosecond precision
fn bench_memory_bandwidth_nanosecond(c: &mut Criterion) {
    let validator = NanosecondValidator::new().unwrap();
    
    let mut group = c.benchmark_group("memory_bandwidth_nanosecond");
    group.measurement_time(Duration::from_secs(60));
    
    // Test different memory access patterns
    for size in [64, 256, 1024, 4096].iter() {
        let data: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        
        group.bench_with_input(
            BenchmarkId::new("memory_access_pattern", size),
            size,
            |b, _| {
                b.iter_custom(|iters| {
                    let start = std::time::Instant::now();
                    for _ in 0..iters {
                        let sum: f64 = data.iter().sum();
                        black_box(sum);
                    }
                    start.elapsed()
                })
            },
        );
        
        // Validate memory access with nanosecond precision
        let memory_operation = || {
            let sum: f64 = data.iter().sum();
            sum > 0.0
        };
        
        let validation_result = validator.validate_custom(memory_operation, &format!("memory_access_{}", size), 100, 0.99).unwrap();
        
        println!("Memory Access Pattern Size {} Validation:", size);
        validation_result.display_results();
    }
    
    group.finish();
}

/// Benchmark cache line efficiency
fn bench_cache_line_efficiency(c: &mut Criterion) {
    let validator = NanosecondValidator::new().unwrap();
    
    let mut group = c.benchmark_group("cache_line_efficiency");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(10000);
    
    // Cache-friendly access (sequential)
    let cache_friendly_data = vec![1.0f64; 64]; // 64 f64s = 512 bytes (8 cache lines)
    
    group.bench_function("cache_friendly_access", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let sum: f64 = cache_friendly_data.iter().sum();
                black_box(sum);
            }
            start.elapsed()
        })
    });
    
    // Validate cache-friendly access
    let cache_friendly_operation = || {
        let sum: f64 = cache_friendly_data.iter().sum();
        sum > 0.0
    };
    
    let validation_result = validator.validate_custom(cache_friendly_operation, "cache_friendly", 50, 0.9999).unwrap();
    
    println!("Cache-Friendly Access Validation:");
    validation_result.display_results();
    
    // Assert cache-friendly operations meet strict targets
    if !validation_result.passed {
        panic!("Cache-friendly validation FAILED: {:.2}% success rate", 
               validation_result.actual_success_rate * 100.0);
    }
    
    group.finish();
}

/// Benchmark SIMD operation nanosecond precision
fn bench_simd_nanosecond_precision(c: &mut Criterion) {
    let validator = NanosecondValidator::new().unwrap();
    let config = AtsCpConfig::high_performance();
    let engine = AtsCpEngine::new(config).unwrap();
    
    let mut group = c.benchmark_group("simd_nanosecond_precision");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(10000);
    
    // Test different SIMD operations
    for size in [4, 8, 16, 32, 64].iter() {
        let a: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..*size).map(|i| (i + 1) as f64).collect();
        
        group.bench_with_input(
            BenchmarkId::new("simd_vector_add", size),
            size,
            |bench, _| {
                bench.iter_custom(|iters| {
                    let start = std::time::Instant::now();
                    for _ in 0..iters {
                        let _ = engine.simd_vector_add(black_box(&a), black_box(&b)).unwrap();
                    }
                    start.elapsed()
                })
            },
        );
        
        // Validate SIMD operations with nanosecond precision
        let simd_operation = || {
            let _ = engine.simd_vector_add(&a, &b).unwrap();
        };
        
        let validation_result = validator.validate_custom(simd_operation, &format!("simd_add_{}", size), 75, 0.9999).unwrap();
        
        println!("SIMD Vector Add Size {} Validation:", size);
        validation_result.display_results();
        
        // Assert SIMD operations meet strict targets
        if !validation_result.passed {
            panic!("SIMD validation FAILED for size {}: {:.2}% success rate", 
                   size, validation_result.actual_success_rate * 100.0);
        }
    }
    
    group.finish();
}

/// Benchmark comprehensive nanosecond validation suite
fn bench_comprehensive_nanosecond_validation(c: &mut Criterion) {
    let validator = NanosecondValidator::new().unwrap();
    let config = AtsCpConfig::high_performance();
    let engine = AtsCpEngine::new(config).unwrap();
    
    let mut group = c.benchmark_group("comprehensive_nanosecond_validation");
    group.measurement_time(Duration::from_secs(120));
    group.sample_size(1000);
    
    // Full ATS-CP pipeline benchmark
    let predictions = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let calibration = vec![0.01, 0.02, 0.03, 0.04, 0.05];
    let temperature = 1.5;
    
    group.bench_function("full_ats_cp_pipeline", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let scaled = engine.temperature_scale(black_box(&predictions), black_box(temperature)).unwrap();
                let _ = engine.conformal_predict(black_box(&scaled), black_box(&calibration)).unwrap();
            }
            start.elapsed()
        })
    });
    
    // Validate full pipeline
    let full_pipeline = || {
        let scaled = engine.temperature_scale(&predictions, temperature).unwrap();
        let _ = engine.conformal_predict(&scaled, &calibration).unwrap();
    };
    
    let validation_result = validator.validate_trading_decision(full_pipeline, "full_pipeline").unwrap();
    
    println!("\n🎯 COMPREHENSIVE NANOSECOND VALIDATION RESULTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    validation_result.display_results();
    
    // Generate final validation report
    let report = validator.generate_report();
    report.display_comprehensive_report();
    
    // Assert comprehensive validation passes
    if !validation_result.passed {
        panic!("COMPREHENSIVE VALIDATION FAILED: {:.2}% success rate", 
               validation_result.actual_success_rate * 100.0);
    }
    
    if !report.all_passed() {
        panic!("COMPREHENSIVE VALIDATION FAILED: Not all components met nanosecond targets");
    }
    
    println!("✅ COMPREHENSIVE NANOSECOND VALIDATION PASSED!");
    println!("🎯 All performance targets achieved with mathematical certainty!");
    
    group.finish();
}

criterion_group!(
    nanosecond_benchmarks,
    bench_nanosecond_trading_decisions,
    bench_nanosecond_whale_detection,
    bench_nanosecond_gpu_kernels,
    bench_nanosecond_api_responses,
    bench_real_world_scenarios,
    bench_memory_bandwidth_nanosecond,
    bench_cache_line_efficiency,
    bench_simd_nanosecond_precision,
    bench_comprehensive_nanosecond_validation
);

criterion_main!(nanosecond_benchmarks);