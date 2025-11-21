//! Performance benchmarks for optimized conformal prediction implementation
//!
//! This benchmark suite validates that the optimized implementations achieve
//! sub-20μs latency targets and provide significant performance improvements.

use ats_core::{
    config::AtsCpConfig,
    conformal::ConformalPredictor,
    conformal_optimized::OptimizedConformalPredictor,
    types::AtsCpVariant,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

/// Creates a benchmark configuration optimized for sub-20μs latency
fn create_optimized_config() -> AtsCpConfig {
    let mut config = AtsCpConfig::high_performance();
    
    // Ultra-aggressive optimization settings
    config.conformal.target_latency_us = 20; // Sub-20μs target
    config.conformal.min_calibration_size = 50; // Smaller for speed
    config.conformal.max_calibration_size = 1000;
    config.simd.enabled = true;
    config.simd.min_simd_size = 16; // Lower threshold for SIMD
    config.memory.prefault_pages = true;
    config.memory.use_huge_pages = true;
    
    config
}

/// Benchmark original vs optimized quantile computation
fn bench_quantile_computation_comparison(c: &mut Criterion) {
    let config = create_optimized_config();
    let mut original = ConformalPredictor::new(&config).unwrap();
    let mut optimized = OptimizedConformalPredictor::new(&config).unwrap();
    
    let mut group = c.benchmark_group("quantile_computation_comparison");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(1000);
    
    // Test different data sizes
    for size in [100, 500, 1000, 2000, 5000].iter() {
        let data: Vec<f64> = (0..*size).map(|i| i as f64 * 0.001 + rand::random::<f64>() * 0.1).collect();
        let confidence = 0.95;
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Original O(n log n) implementation
        group.bench_with_input(
            BenchmarkId::new("original_quantile", size),
            size,
            |b, _| {
                b.iter(|| {
                    // Using internal quantile computation (simplified access)
                    original.compute_quantile_linear(black_box(&data), black_box(confidence)).unwrap()
                })
            },
        );
        
        // Optimized Greenwald-Khanna O(n) implementation  
        group.bench_with_input(
            BenchmarkId::new("greenwald_khanna_quantile", size),
            size,
            |b, _| {
                b.iter(|| {
                    optimized.compute_quantile_gk(black_box(&data), black_box(confidence)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark original vs optimized softmax computation
fn bench_softmax_computation_comparison(c: &mut Criterion) {
    let config = create_optimized_config();
    let mut original = ConformalPredictor::new(&config).unwrap();
    let mut optimized = OptimizedConformalPredictor::new(&config).unwrap();
    
    let mut group = c.benchmark_group("softmax_computation_comparison");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(1000);
    
    // Test different logit sizes
    for size in [8, 16, 32, 64, 128, 256].iter() {
        let logits: Vec<f64> = (0..*size).map(|i| i as f64 * 0.1 - (*size as f64 * 0.05)).collect();
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Original scalar softmax
        group.bench_with_input(
            BenchmarkId::new("original_softmax", size),
            size,
            |b, _| {
                b.iter(|| {
                    original.compute_softmax(black_box(&logits)).unwrap()
                })
            },
        );
        
        // Optimized AVX-512 SIMD softmax
        group.bench_with_input(
            BenchmarkId::new("avx512_softmax", size),
            size,
            |b, _| {
                b.iter(|| {
                    optimized.softmax_avx512_optimized(black_box(&logits)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark end-to-end conformal prediction performance
fn bench_conformal_prediction_end_to_end(c: &mut Criterion) {
    let config = create_optimized_config();
    let mut original = ConformalPredictor::new(&config).unwrap();
    let mut optimized = OptimizedConformalPredictor::new(&config).unwrap();
    
    let mut group = c.benchmark_group("conformal_prediction_end_to_end");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(2000); // More samples for accurate latency measurement
    
    // Real-world sized data for high-frequency trading
    let predictions: Vec<f64> = (0..32).map(|i| i as f64 * 0.01 + rand::random::<f64>() * 0.05).collect();
    let calibration_data: Vec<f64> = (0..500).map(|i| i as f64 * 0.002 + rand::random::<f64>() * 0.1).collect();
    let confidence = 0.95;
    
    // Original implementation
    group.bench_function("original_conformal_predict", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = original.predict(black_box(&predictions), black_box(&calibration_data)).unwrap();
            }
            start.elapsed()
        })
    });
    
    // Optimized implementation
    group.bench_function("optimized_conformal_predict", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = optimized.predict_optimized(
                    black_box(&predictions), 
                    black_box(&calibration_data), 
                    black_box(confidence)
                ).unwrap();
            }
            start.elapsed()
        })
    });
    
    group.finish();
}

/// Benchmark ATS-CP algorithm implementations
fn bench_ats_cp_algorithm_comparison(c: &mut Criterion) {
    let config = create_optimized_config();
    let mut original = ConformalPredictor::new(&config).unwrap();
    let mut optimized = OptimizedConformalPredictor::new(&config).unwrap();
    
    let mut group = c.benchmark_group("ats_cp_algorithm_comparison");
    group.measurement_time(Duration::from_secs(45));
    group.sample_size(500);
    
    // Test data for ATS-CP
    let logits: Vec<f64> = vec![2.3, 1.1, 0.8, 1.9, 0.2, 1.5, 0.9, 2.1]; // 8-class classification
    let calibration_logits: Vec<Vec<f64>> = (0..100).map(|_| {
        (0..8).map(|i| i as f64 * 0.3 + rand::random::<f64>() * 0.5).collect()
    }).collect();
    let calibration_labels: Vec<usize> = (0..100).map(|_| rand::random::<usize>() % 8).collect();
    let confidence = 0.95;
    
    for variant in [AtsCpVariant::GQ, AtsCpVariant::AQ, AtsCpVariant::MGQ].iter() {
        let variant_name = format!("{:?}", variant);
        
        // Original ATS-CP implementation
        group.bench_function(format!("original_ats_cp_{}", variant_name).as_str(), |b| {
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    let _ = original.ats_cp_predict(
                        black_box(&logits),
                        black_box(&calibration_logits),
                        black_box(&calibration_labels),
                        black_box(confidence),
                        black_box(variant.clone()),
                    ).unwrap();
                }
                start.elapsed()
            })
        });
        
        // Optimized ATS-CP implementation
        group.bench_function(format!("optimized_ats_cp_{}", variant_name).as_str(), |b| {
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    let _ = optimized.ats_cp_predict_optimized(
                        black_box(&logits),
                        black_box(&calibration_logits),
                        black_box(&calibration_labels),
                        black_box(confidence),
                        black_box(variant.clone()),
                    ).unwrap();
                }
                start.elapsed()
            })
        });
    }
    
    group.finish();
}

/// Latency validation benchmark - must achieve sub-20μs consistently
fn bench_latency_validation_sub_20us(c: &mut Criterion) {
    let config = create_optimized_config();
    let mut optimized = OptimizedConformalPredictor::new(&config).unwrap();
    
    let mut group = c.benchmark_group("latency_validation_sub_20us");
    group.measurement_time(Duration::from_secs(120)); // Longer measurement for accuracy
    group.sample_size(10000); // Very large sample for P99 latency accuracy
    
    // Critical path data sizes for HFT
    let small_predictions: Vec<f64> = (0..16).map(|i| i as f64 * 0.01).collect();
    let small_calibration: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
    let confidence = 0.95;
    
    // Quantile computation latency (target: <5μs)
    group.bench_function("quantile_5us_target", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = optimized.compute_quantile_gk(black_box(&small_calibration), black_box(confidence)).unwrap();
            }
            start.elapsed()
        })
    });
    
    // Softmax computation latency (target: <2μs)  
    let logits: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
    group.bench_function("softmax_2us_target", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = optimized.softmax_avx512_optimized(black_box(&logits)).unwrap();
            }
            start.elapsed()
        })
    });
    
    // Full conformal prediction latency (target: <20μs)
    group.bench_function("full_conformal_20us_target", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = optimized.predict_optimized(
                    black_box(&small_predictions), 
                    black_box(&small_calibration), 
                    black_box(confidence)
                ).unwrap();
            }
            start.elapsed()
        })
    });
    
    // ATS-CP algorithm latency (target: <20μs)
    let small_logits: Vec<f64> = (0..8).map(|i| i as f64 * 0.2).collect();
    let calib_logits: Vec<Vec<f64>> = (0..50).map(|_| {
        (0..8).map(|i| i as f64 * 0.1 + rand::random::<f64>() * 0.2).collect()
    }).collect();
    let calib_labels: Vec<usize> = (0..50).map(|_| rand::random::<usize>() % 8).collect();
    
    group.bench_function("ats_cp_20us_target", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = optimized.ats_cp_predict_optimized(
                    black_box(&small_logits),
                    black_box(&calib_logits),
                    black_box(&calib_labels),
                    black_box(confidence),
                    black_box(AtsCpVariant::GQ),
                ).unwrap();
            }
            start.elapsed()
        })
    });
    
    group.finish();
}

/// Memory access pattern benchmarks
fn bench_memory_access_patterns(c: &mut Criterion) {
    use ats_core::memory_optimized::{CacheAlignedVec, ConformalDataLayout};
    
    let mut group = c.benchmark_group("memory_access_patterns");
    group.measurement_time(Duration::from_secs(30));
    
    // Compare standard Vec vs CacheAlignedVec for large data operations
    for size in [1024, 4096, 16384].iter() {
        let std_vec: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let cache_vec = CacheAlignedVec::from_slice(&std_vec).unwrap();
        
        group.throughput(Throughput::Bytes((*size * std::mem::size_of::<f64>()) as u64));
        
        // Standard vector access
        group.bench_with_input(
            BenchmarkId::new("standard_vec_access", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut sum = 0.0;
                    for &val in black_box(&std_vec) {
                        sum += val;
                    }
                    sum
                })
            },
        );
        
        // Cache-aligned vector access
        group.bench_with_input(
            BenchmarkId::new("cache_aligned_vec_access", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut sum = 0.0;
                    for &val in black_box(cache_vec.as_slice()) {
                        sum += val;
                    }
                    sum
                })
            },
        );
    }
    
    // Conformal data layout access patterns
    let layout = ConformalDataLayout::new(1000, 2000).unwrap();
    
    group.bench_function("conformal_layout_efficiency", |b| {
        b.iter(|| {
            // Simulate typical conformal prediction access pattern
            let preds = black_box(layout.predictions.as_slice());
            let calib = black_box(layout.calibration_scores.as_slice());
            
            let mut result = 0.0;
            for (p, c) in preds.iter().zip(calib.iter().cycle()) {
                result += p * c;
            }
            result
        })
    });
    
    group.finish();
}

/// Throughput benchmarks for batch processing
fn bench_throughput_scalability(c: &mut Criterion) {
    let config = create_optimized_config();
    let mut optimized = OptimizedConformalPredictor::new(&config).unwrap();
    
    let mut group = c.benchmark_group("throughput_scalability");
    group.measurement_time(Duration::from_secs(30));
    
    // Test scalability with increasing batch sizes
    for batch_size in [10, 50, 100, 500, 1000].iter() {
        let batch_predictions: Vec<Vec<f64>> = (0..*batch_size).map(|_| {
            (0..32).map(|i| i as f64 * 0.01 + rand::random::<f64>() * 0.05).collect()
        }).collect();
        let calibration_data: Vec<f64> = (0..200).map(|i| i as f64 * 0.005).collect();
        let confidence = 0.95;
        
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_conformal_prediction", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let mut results = Vec::new();
                    for predictions in black_box(&batch_predictions) {
                        let intervals = optimized.predict_optimized(
                            predictions,
                            black_box(&calibration_data),
                            black_box(confidence)
                        ).unwrap();
                        results.push(intervals);
                    }
                    results
                })
            },
        );
    }
    
    group.finish();
}

/// Performance regression detection benchmark
fn bench_performance_regression_detection(c: &mut Criterion) {
    let config = create_optimized_config();
    let mut optimized = OptimizedConformalPredictor::new(&config).unwrap();
    
    let mut group = c.benchmark_group("performance_regression_detection");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(5000);
    
    // Baseline performance measurement
    let predictions: Vec<f64> = (0..32).map(|i| i as f64 * 0.01).collect();
    let calibration_data: Vec<f64> = (0..300).map(|i| i as f64 * 0.003).collect();
    let confidence = 0.95;
    
    // This benchmark serves as a regression test - if performance degrades
    // significantly from baseline, it indicates a regression
    group.bench_function("regression_baseline", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = optimized.predict_optimized(
                    black_box(&predictions),
                    black_box(&calibration_data),
                    black_box(confidence)
                ).unwrap();
            }
            let elapsed = start.elapsed();
            
            // Validate that we're consistently under 20μs per operation
            let per_op_ns = elapsed.as_nanos() / iters as u128;
            if per_op_ns > 20_000 {
                eprintln!("REGRESSION DETECTED: Operation took {}ns (>20μs)", per_op_ns);
            }
            
            elapsed
        })
    });
    
    group.finish();
}

criterion_group!(
    optimized_benchmarks,
    bench_quantile_computation_comparison,
    bench_softmax_computation_comparison, 
    bench_conformal_prediction_end_to_end,
    bench_ats_cp_algorithm_comparison,
    bench_latency_validation_sub_20us,
    bench_memory_access_patterns,
    bench_throughput_scalability,
    bench_performance_regression_detection
);

criterion_main!(optimized_benchmarks);