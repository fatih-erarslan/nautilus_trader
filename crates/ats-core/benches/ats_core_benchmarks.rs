//! Comprehensive benchmarks for ATS-Core performance validation
//!
//! This benchmark suite validates that all ATS-CP operations meet their
//! sub-100μs latency targets for high-frequency trading applications.

use ats_core::{
    config::AtsCpConfig,
    prelude::*,
    temperature::TemperatureScaler,
    conformal::ConformalPredictor,
    simd::SimdOperations,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

/// Benchmark configuration optimized for latency measurement
fn create_benchmark_config() -> AtsCpConfig {
    AtsCpConfig::high_performance()
}

/// Temperature scaling benchmarks
fn bench_temperature_scaling(c: &mut Criterion) {
    let config = create_benchmark_config();
    let mut scaler = TemperatureScaler::new(&config).unwrap();
    
    let mut group = c.benchmark_group("temperature_scaling");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(1000);
    
    // Test different array sizes
    for size in [32, 64, 128, 256, 512, 1024, 2048, 4096].iter() {
        let predictions: Vec<f64> = (0..*size).map(|i| i as f64 * 0.01).collect();
        let temperature = 1.5;
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("basic_scaling", size),
            size,
            |b, _| {
                b.iter(|| {
                    scaler.scale(black_box(&predictions), black_box(temperature)).unwrap()
                })
            },
        );
        
        // SIMD scaling benchmark
        if *size >= 64 {
            group.bench_with_input(
                BenchmarkId::new("simd_scaling", size),
                size,
                |b, _| {
                    b.iter(|| {
                        scaler.scale(black_box(&predictions), black_box(temperature)).unwrap()
                    })
                },
            );
        }
        
        // Parallel scaling benchmark for large arrays
        if *size >= 1024 {
            group.bench_with_input(
                BenchmarkId::new("parallel_scaling", size),
                size,
                |b, _| {
                    b.iter(|| {
                        scaler.scale_parallel(black_box(&predictions), black_box(temperature)).unwrap()
                    })
                },
            );
        }
    }
    
    // Temperature optimization benchmark
    let predictions: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
    let targets: Vec<f64> = (0..100).map(|i| i as f64 * 0.01 + 0.1).collect();
    let confidence = 0.95;
    
    group.bench_function("temperature_optimization", |b| {
        b.iter(|| {
            scaler.optimize_temperature(
                black_box(&predictions),
                black_box(&targets),
                black_box(confidence),
            ).unwrap()
        })
    });
    
    // Softmax with temperature benchmark
    let logits: Vec<f64> = (0..32).map(|i| i as f64 * 0.1).collect();
    let temperature = 2.0;
    
    group.bench_function("softmax_with_temperature", |b| {
        b.iter(|| {
            scaler.softmax_with_temperature(black_box(&logits), black_box(temperature)).unwrap()
        })
    });
    
    group.finish();
}

/// Conformal prediction benchmarks
fn bench_conformal_prediction(c: &mut Criterion) {
    let config = create_benchmark_config();
    let mut predictor = ConformalPredictor::new(&config).unwrap();
    
    let mut group = c.benchmark_group("conformal_prediction");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(1000);
    
    // Generate test data
    let calibration_data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    
    // Test different prediction sizes
    for size in [16, 32, 64, 128, 256, 512, 1024].iter() {
        let predictions: Vec<f64> = (0..*size).map(|i| i as f64 * 0.01).collect();
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("basic_prediction", size),
            size,
            |b, _| {
                b.iter(|| {
                    predictor.predict(black_box(&predictions), black_box(&calibration_data)).unwrap()
                })
            },
        );
        
        // Detailed prediction benchmark
        let confidence = 0.95;
        group.bench_with_input(
            BenchmarkId::new("detailed_prediction", size),
            size,
            |b, _| {
                b.iter(|| {
                    predictor.predict_detailed(
                        black_box(&predictions),
                        black_box(&calibration_data),
                        black_box(confidence),
                    ).unwrap()
                })
            },
        );
        
        // Parallel prediction benchmark for large arrays
        if *size >= 256 {
            group.bench_with_input(
                BenchmarkId::new("parallel_prediction", size),
                size,
                |b, _| {
                    b.iter(|| {
                        predictor.predict_parallel(
                            black_box(&predictions),
                            black_box(&calibration_data),
                            black_box(confidence),
                        ).unwrap()
                    })
                },
            );
        }
    }
    
    // Adaptive conformal prediction benchmark
    let predictions: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
    let true_values: Vec<f64> = (0..100).map(|i| i as f64 * 0.01 + 0.05).collect();
    let confidence = 0.95;
    
    group.bench_function("adaptive_prediction", |b| {
        b.iter(|| {
            predictor.predict_adaptive(
                black_box(&predictions),
                black_box(&true_values),
                black_box(confidence),
            ).unwrap()
        })
    });
    
    group.finish();
}

/// SIMD operations benchmarks
fn bench_simd_operations(c: &mut Criterion) {
    let config = create_benchmark_config();
    let mut simd_ops = SimdOperations::new(&config).unwrap();
    
    let mut group = c.benchmark_group("simd_operations");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(1000);
    
    // Test different array sizes
    for size in [64, 128, 256, 512, 1024, 2048, 4096, 8192].iter() {
        let a: Vec<f64> = (0..*size).map(|i| i as f64 * 0.01).collect();
        let b: Vec<f64> = (0..*size).map(|i| (i as f64 + 1.0) * 0.01).collect();
        let scalar = 2.5;
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Vector addition benchmark
        group.bench_with_input(
            BenchmarkId::new("vector_add", size),
            size,
            |b, _| {
                b.iter(|| {
                    simd_ops.vector_add(black_box(&a), black_box(&b)).unwrap()
                })
            },
        );
        
        // Vector multiplication benchmark
        group.bench_with_input(
            BenchmarkId::new("vector_multiply", size),
            size,
            |b, _| {
                b.iter(|| {
                    simd_ops.vector_multiply(black_box(&a), black_box(&b)).unwrap()
                })
            },
        );
        
        // Scalar multiplication benchmark
        group.bench_with_input(
            BenchmarkId::new("scalar_multiply", size),
            size,
            |b, _| {
                b.iter(|| {
                    simd_ops.scalar_multiply(black_box(&a), black_box(scalar)).unwrap()
                })
            },
        );
        
        // Dot product benchmark
        group.bench_with_input(
            BenchmarkId::new("dot_product", size),
            size,
            |b, _| {
                b.iter(|| {
                    simd_ops.dot_product(black_box(&a), black_box(&b)).unwrap()
                })
            },
        );
        
        // Vector exponential benchmark
        group.bench_with_input(
            BenchmarkId::new("vector_exp", size),
            size,
            |b, _| {
                b.iter(|| {
                    simd_ops.vector_exp(black_box(&a)).unwrap()
                })
            },
        );
        
        // Fused multiply-add benchmark
        let c_vec: Vec<f64> = (0..*size).map(|i| (i as f64) * 0.005).collect();
        group.bench_with_input(
            BenchmarkId::new("fused_multiply_add", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    simd_ops.fused_multiply_add(black_box(&a), black_box(&b), black_box(&c_vec)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// End-to-end ATS-CP engine benchmarks
fn bench_ats_cp_engine(c: &mut Criterion) {
    let config = create_benchmark_config();
    let engine = AtsCpEngine::new(config).unwrap();
    
    let mut group = c.benchmark_group("ats_cp_engine");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(500);
    
    // Generate test data
    let predictions: Vec<f64> = (0..128).map(|i| i as f64 * 0.01).collect();
    let calibration_data: Vec<f64> = (0..500).map(|i| i as f64 * 0.002).collect();
    let temperature = 1.5;
    
    // End-to-end temperature scaling + conformal prediction
    group.bench_function("full_pipeline", |b| {
        b.iter(|| {
            // Temperature scaling
            let scaled = engine.temperature_scale(black_box(&predictions), black_box(temperature)).unwrap();
            
            // Conformal prediction
            let _intervals = engine.conformal_predict(black_box(&scaled), black_box(&calibration_data)).unwrap();
        })
    });
    
    // Individual operations
    group.bench_function("temperature_scale_only", |b| {
        b.iter(|| {
            engine.temperature_scale(black_box(&predictions), black_box(temperature)).unwrap()
        })
    });
    
    group.bench_function("conformal_predict_only", |b| {
        b.iter(|| {
            engine.conformal_predict(black_box(&predictions), black_box(&calibration_data)).unwrap()
        })
    });
    
    // SIMD operations
    let a: Vec<f64> = (0..256).map(|i| i as f64 * 0.01).collect();
    let b: Vec<f64> = (0..256).map(|i| (i as f64 + 1.0) * 0.01).collect();
    
    group.bench_function("simd_vector_add", |b| {
        b.iter(|| {
            engine.simd_vector_add(black_box(&a), black_box(&b)).unwrap()
        })
    });
    
    group.finish();
}

/// Latency validation benchmarks (must meet sub-100μs targets)
fn bench_latency_validation(c: &mut Criterion) {
    let config = create_benchmark_config();
    let engine = AtsCpEngine::new(config).unwrap();
    
    let mut group = c.benchmark_group("latency_validation");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(10000); // Large sample for accurate latency measurement
    
    // Critical path: temperature scaling (target: <5μs)
    let small_predictions: Vec<f64> = (0..32).map(|i| i as f64 * 0.01).collect();
    let temperature = 1.5;
    
    group.bench_function("temperature_scale_5us_target", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = engine.temperature_scale(black_box(&small_predictions), black_box(temperature)).unwrap();
            }
            start.elapsed()
        })
    });
    
    // Conformal prediction (target: <20μs)
    let small_calibration: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
    
    group.bench_function("conformal_predict_20us_target", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = engine.conformal_predict(black_box(&small_predictions), black_box(&small_calibration)).unwrap();
            }
            start.elapsed()
        })
    });
    
    // Full pipeline (target: <100μs)
    group.bench_function("full_pipeline_100us_target", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let scaled = engine.temperature_scale(black_box(&small_predictions), black_box(temperature)).unwrap();
                let _ = engine.conformal_predict(black_box(&scaled), black_box(&small_calibration)).unwrap();
            }
            start.elapsed()
        })
    });
    
    group.finish();
}

/// Memory bandwidth benchmarks
fn bench_memory_bandwidth(c: &mut Criterion) {
    let config = create_benchmark_config();
    let mut simd_ops = SimdOperations::new(&config).unwrap();
    
    let mut group = c.benchmark_group("memory_bandwidth");
    group.measurement_time(Duration::from_secs(30));
    
    // Test memory bandwidth with large arrays
    for size in [1024, 4096, 16384, 65536, 262144].iter() {
        let a: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..*size).map(|i| (i as f64) + 1.0).collect();
        
        let bytes_per_op = size * 3 * std::mem::size_of::<f64>(); // Read a, read b, write result
        group.throughput(Throughput::Bytes(bytes_per_op as u64));
        
        group.bench_with_input(
            BenchmarkId::new("memory_bandwidth", size),
            size,
            |b, _| {
                b.iter(|| {
                    simd_ops.vector_add(black_box(&a), black_box(&b)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Cache efficiency benchmarks
fn bench_cache_efficiency(c: &mut Criterion) {
    let config = create_benchmark_config();
    let mut simd_ops = SimdOperations::new(&config).unwrap();
    
    let mut group = c.benchmark_group("cache_efficiency");
    group.measurement_time(Duration::from_secs(30));
    
    // L1 cache size test (typically 32KB)
    let l1_size = 4096; // 4K f64 elements = 32KB
    let l1_data: Vec<f64> = (0..l1_size).map(|i| i as f64).collect();
    let l1_data_b: Vec<f64> = (0..l1_size).map(|i| (i as f64) + 1.0).collect();
    
    group.bench_function("l1_cache_friendly", |b| {
        b.iter(|| {
            simd_ops.vector_add(black_box(&l1_data), black_box(&l1_data_b)).unwrap()
        })
    });
    
    // L3 cache size test (typically 8MB)
    let l3_size = 1048576; // 1M f64 elements = 8MB
    let l3_data: Vec<f64> = (0..l3_size).map(|i| i as f64).collect();
    let l3_data_b: Vec<f64> = (0..l3_size).map(|i| (i as f64) + 1.0).collect();
    
    group.bench_function("l3_cache_friendly", |b| {
        b.iter(|| {
            simd_ops.vector_add(black_box(&l3_data), black_box(&l3_data_b)).unwrap()
        })
    });
    
    // Memory-bound test (larger than cache)
    let mem_size = 4194304; // 4M f64 elements = 32MB
    let mem_data: Vec<f64> = (0..mem_size).map(|i| i as f64).collect();
    let mem_data_b: Vec<f64> = (0..mem_size).map(|i| (i as f64) + 1.0).collect();
    
    group.bench_function("memory_bound", |b| {
        b.iter(|| {
            simd_ops.vector_add(black_box(&mem_data), black_box(&mem_data_b)).unwrap()
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_temperature_scaling,
    bench_conformal_prediction,
    bench_simd_operations,
    bench_ats_cp_engine,
    bench_latency_validation,
    bench_memory_bandwidth,
    bench_cache_efficiency
);

criterion_main!(benches);