//! Comprehensive Performance Benchmarking Suite for CDFA Unified
//!
//! This benchmark suite provides comprehensive performance testing with:
//! - Black Swan detection latency (<500ns target)
//! - SOC analysis latency (~800ns target)
//! - Antifragility analysis performance
//! - STDP sub-microsecond validation
//! - SIMD vs scalar performance comparison
//! - GPU vs CPU acceleration benchmarks
//! - Memory allocation pattern analysis
//! - Throughput testing with 1M+ data points
//! - Performance regression detection

use criterion::{
    black_box, criterion_group, criterion_main, 
    BenchmarkId, Criterion, Throughput, PlotConfiguration, AxisScale, Bencher
};
use cdfa_unified::{
    detectors::black_swan::{BlackSwanDetector, BlackSwanConfig},
    analyzers::{
        soc::{SOCAnalyzer, SOCParameters},
        antifragility::{AntifragilityAnalyzer, AntifragilityParameters},
    },
    optimizers::stdp::{STDPOptimizer, STDPConfig},
    types::{CdfaArray, CdfaMatrix, CdfaFloat},
    error::CdfaResult,
};

#[cfg(feature = "simd")]
use cdfa_unified::simd::{avx::AvxOperations, basic::BasicSIMD};

#[cfg(feature = "gpu")]
use cdfa_unified::gpu::{cuda::CudaContext, webgpu::WebGPUContext};

use ndarray::{Array1, Array2};
use std::time::{Duration, Instant};
use rayon::prelude::*;

// Performance targets (in nanoseconds)
const BLACK_SWAN_TARGET_NS: u64 = 500;
const SOC_TARGET_NS: u64 = 800;
const STDP_TARGET_NS: u64 = 1000; // 1 microsecond
const ANTIFRAGILITY_TARGET_MS: u64 = 10; // 10 milliseconds

// Test data sizes
const MICRO_SIZE: usize = 10;
const SMALL_SIZE: usize = 100;
const MEDIUM_SIZE: usize = 1_000;
const LARGE_SIZE: usize = 10_000;
const XLARGE_SIZE: usize = 100_000;
const MASSIVE_SIZE: usize = 1_000_000;

/// Generate test data for various algorithms
fn generate_financial_data(size: usize, volatility: f64) -> (CdfaArray, CdfaArray) {
    let mut prices = Array1::zeros(size);
    let mut volumes = Array1::zeros(size);
    
    let mut price = 100.0;
    for i in 0..size {
        let return_rate = volatility * ((i as f64 * 0.1).sin() + (i as f64 * 0.01).cos());
        price *= 1.0 + return_rate;
        prices[i] = price;
        volumes[i] = 1000.0 + 500.0 * ((i as f64 * 0.05).sin()).abs();
    }
    
    (prices, volumes)
}

/// Generate correlated time series for diversity testing
fn generate_correlated_series(size: usize, correlation: f64) -> CdfaMatrix {
    let mut matrix = Array2::zeros((size, 3));
    let base = Array1::linspace(0.0, 1.0, size);
    
    for i in 0..size {
        let noise1 = ((i as f64 * 0.1).sin());
        let noise2 = ((i as f64 * 0.15).cos());
        
        matrix[[i, 0]] = base[i];
        matrix[[i, 1]] = correlation * base[i] + (1.0 - correlation) * noise1;
        matrix[[i, 2]] = correlation * base[i] + (1.0 - correlation) * noise2;
    }
    
    matrix
}

/// Generate synthetic Black Swan events
fn generate_black_swan_data(size: usize) -> CdfaArray {
    let mut data = Array1::zeros(size);
    
    for i in 0..size {
        if i == size / 2 {
            // Insert Black Swan event (extreme outlier)
            data[i] = 10.0 * ((i as f64).sin() + 5.0);
        } else {
            data[i] = 0.1 * (i as f64).sin();
        }
    }
    
    data
}

// === BLACK SWAN DETECTION BENCHMARKS ===

fn bench_black_swan_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("black_swan/latency");
    group.significance_level(0.1).sample_size(1000);
    
    let config = BlackSwanConfig {
        window_size: 100,
        use_simd: true,
        parallel_processing: false, // Single-threaded for latency measurement
        ..Default::default()
    };
    
    let detector = BlackSwanDetector::new(config).expect("Failed to create detector");
    let data = generate_black_swan_data(SMALL_SIZE);
    
    group.bench_function("single_detection", |b| {
        b.iter(|| {
            let start = Instant::now();
            let result = detector.detect_anomalies(black_box(&data));
            let duration = start.elapsed();
            
            // Validate latency target
            if duration.as_nanos() > BLACK_SWAN_TARGET_NS as u128 {
                eprintln!("WARNING: Black Swan detection took {}ns, target: {}ns", 
                         duration.as_nanos(), BLACK_SWAN_TARGET_NS);
            }
            
            black_box(result)
        })
    });
    
    group.finish();
}

fn bench_black_swan_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("black_swan/throughput");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for size in [MEDIUM_SIZE, LARGE_SIZE, XLARGE_SIZE, MASSIVE_SIZE].iter() {
        let config = BlackSwanConfig {
            window_size: (*size).min(1000),
            use_simd: true,
            parallel_processing: true,
            ..Default::default()
        };
        
        let detector = BlackSwanDetector::new(config).expect("Failed to create detector");
        let data = generate_black_swan_data(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("detection_throughput", size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = detector.detect_anomalies(black_box(&data));
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

// === SOC ANALYSIS BENCHMARKS ===

fn bench_soc_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("soc/latency");
    group.significance_level(0.1).sample_size(1000);
    
    let params = SOCParameters::default();
    let analyzer = SOCAnalyzer::new(params);
    let (data, _) = generate_financial_data(SMALL_SIZE, 0.02);
    
    group.bench_function("single_analysis", |b| {
        b.iter(|| {
            let start = Instant::now();
            let result = analyzer.analyze_series(black_box(&data));
            let duration = start.elapsed();
            
            // Validate latency target
            if duration.as_nanos() > SOC_TARGET_NS as u128 {
                eprintln!("WARNING: SOC analysis took {}ns, target: {}ns", 
                         duration.as_nanos(), SOC_TARGET_NS);
            }
            
            black_box(result)
        })
    });
    
    group.finish();
}

fn bench_soc_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("soc/throughput");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for size in [MEDIUM_SIZE, LARGE_SIZE, XLARGE_SIZE].iter() {
        let params = SOCParameters::default();
        let analyzer = SOCAnalyzer::new(params);
        let (data, _) = generate_financial_data(*size, 0.02);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("analysis_throughput", size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = analyzer.analyze_series(black_box(&data));
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

// === ANTIFRAGILITY BENCHMARKS ===

fn bench_antifragility_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("antifragility/performance");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        let mut params = AntifragilityParameters::default();
        params.min_data_points = (*size / 10).max(50);
        params.enable_simd = true;
        params.enable_parallel = true;
        
        let analyzer = AntifragilityAnalyzer::with_params(params)
            .expect("Failed to create analyzer");
        let (prices, volumes) = generate_financial_data(*size, 0.03);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("full_analysis", size),
            size,
            |b, _| {
                b.iter(|| {
                    let start = Instant::now();
                    let result = analyzer.analyze_prices(black_box(&prices), black_box(&volumes));
                    let duration = start.elapsed();
                    
                    // Validate latency target for small datasets
                    if *size == SMALL_SIZE && duration.as_millis() > ANTIFRAGILITY_TARGET_MS as u128 {
                        eprintln!("WARNING: Antifragility analysis took {}ms, target: {}ms", 
                                 duration.as_millis(), ANTIFRAGILITY_TARGET_MS);
                    }
                    
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

// === STDP OPTIMIZATION BENCHMARKS ===

fn bench_stdp_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("stdp/latency");
    group.significance_level(0.1).sample_size(1000);
    
    let config = STDPConfig {
        simd_width: 8, // Force SIMD
        parallel_enabled: false, // Single-threaded for latency
        ..Default::default()
    };
    
    let mut optimizer = STDPOptimizer::new(config).expect("Failed to create optimizer");
    let weights = Array2::from_elem((10, 10), 0.5);
    
    group.bench_function("weight_update", |b| {
        b.iter(|| {
            let start = Instant::now();
            let result = optimizer.update_weights(black_box(&weights), 0.01);
            let duration = start.elapsed();
            
            // Validate sub-microsecond target
            if duration.as_nanos() > STDP_TARGET_NS as u128 {
                eprintln!("WARNING: STDP update took {}ns, target: {}ns", 
                         duration.as_nanos(), STDP_TARGET_NS);
            }
            
            black_box(result)
        })
    });
    
    group.finish();
}

fn bench_stdp_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("stdp/throughput");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for neurons in [10, 50, 100, 500, 1000].iter() {
        let config = STDPConfig {
            simd_width: 8,
            parallel_enabled: true,
            ..Default::default()
        };
        
        let mut optimizer = STDPOptimizer::new(config).expect("Failed to create optimizer");
        let weights = Array2::from_elem((*neurons, *neurons), 0.5);
        
        group.throughput(Throughput::Elements((*neurons * *neurons) as u64));
        group.bench_with_input(
            BenchmarkId::new("network_update", neurons),
            neurons,
            |b, _| {
                b.iter(|| {
                    let result = optimizer.update_weights(black_box(&weights), 0.01);
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

// === SIMD VS SCALAR COMPARISON ===

#[cfg(feature = "simd")]
fn bench_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/comparison");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        let data1 = Array1::linspace(0.0, 1.0, *size);
        let data2 = Array1::linspace(1.0, 2.0, *size);
        
        // Scalar implementation
        group.bench_with_input(
            BenchmarkId::new("scalar_dot_product", size),
            size,
            |b, _| {
                b.iter(|| {
                    let result: f64 = data1.iter().zip(data2.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    black_box(result)
                })
            },
        );
        
        // SIMD implementation
        group.bench_with_input(
            BenchmarkId::new("simd_dot_product", size),
            size,
            |b, _| {
                b.iter(|| {
                    let simd_ops = AvxOperations::new();
                    let result = simd_ops.dot_product(black_box(&data1), black_box(&data2));
                    black_box(result)
                })
            },
        );
        
        // ndarray vectorized implementation
        group.bench_with_input(
            BenchmarkId::new("ndarray_dot_product", size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = data1.dot(&data2);
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

// === GPU VS CPU BENCHMARKS ===

#[cfg(feature = "gpu")]
fn bench_gpu_vs_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu/comparison");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    // Initialize GPU context
    let gpu_context = match WebGPUContext::new() {
        Ok(ctx) => Some(ctx),
        Err(_) => {
            eprintln!("GPU context not available, skipping GPU benchmarks");
            return;
        }
    };
    
    for size in [LARGE_SIZE, XLARGE_SIZE, MASSIVE_SIZE].iter() {
        let matrix = Array2::from_shape_fn((*size / 100, 100), |(i, j)| {
            (i as f64 + j as f64) / (*size as f64)
        });
        
        // CPU implementation
        group.bench_with_input(
            BenchmarkId::new("cpu_matrix_multiply", size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = matrix.dot(&matrix.t());
                    black_box(result)
                })
            },
        );
        
        // GPU implementation
        if let Some(ref ctx) = gpu_context {
            group.bench_with_input(
                BenchmarkId::new("gpu_matrix_multiply", size),
                size,
                |b, _| {
                    b.iter(|| {
                        let result = ctx.matrix_multiply(black_box(&matrix), black_box(&matrix.t()));
                        black_box(result)
                    })
                },
            );
        }
    }
    
    group.finish();
}

// === MEMORY ALLOCATION PATTERN BENCHMARKS ===

fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/patterns");
    
    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        // Stack allocation (small arrays)
        if *size <= 1000 {
            group.bench_with_input(
                BenchmarkId::new("stack_allocation", size),
                size,
                |b, &size| {
                    b.iter(|| {
                        let mut data = [0.0f64; 1000];
                        for i in 0..size.min(1000) {
                            data[i] = i as f64;
                        }
                        black_box(data)
                    })
                },
            );
        }
        
        // Heap allocation
        group.bench_with_input(
            BenchmarkId::new("heap_allocation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let data = vec![0.0f64; size];
                    black_box(data)
                })
            },
        );
        
        // ndarray allocation
        group.bench_with_input(
            BenchmarkId::new("ndarray_allocation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let data = Array1::<f64>::zeros(size);
                    black_box(data)
                })
            },
        );
        
        // Matrix allocation
        group.bench_with_input(
            BenchmarkId::new("matrix_allocation", size),
            size,
            |b, &size| {
                let dim = (size as f64).sqrt() as usize;
                b.iter(|| {
                    let matrix = Array2::<f64>::zeros((dim, dim));
                    black_box(matrix)
                })
            },
        );
    }
    
    group.finish();
}

// === PARALLEL PROCESSING BENCHMARKS ===

fn bench_parallel_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel/processing");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for size in [LARGE_SIZE, XLARGE_SIZE, MASSIVE_SIZE].iter() {
        let data = Array1::linspace(0.0, 1.0, *size);
        
        // Sequential processing
        group.bench_with_input(
            BenchmarkId::new("sequential_computation", size),
            size,
            |b, _| {
                b.iter(|| {
                    let result: f64 = data.iter()
                        .map(|&x| x.sin().cos().tan().exp().ln())
                        .sum();
                    black_box(result)
                })
            },
        );
        
        // Parallel processing
        group.bench_with_input(
            BenchmarkId::new("parallel_computation", size),
            size,
            |b, _| {
                b.iter(|| {
                    let result: f64 = data.par_iter()
                        .map(|&x| x.sin().cos().tan().exp().ln())
                        .sum();
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

// === PERFORMANCE REGRESSION TESTS ===

fn bench_regression_suite(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression/validation");
    
    // Black Swan regression test
    group.bench_function("black_swan_regression_1000", |b| {
        let config = BlackSwanConfig {
            window_size: 100,
            use_simd: true,
            ..Default::default()
        };
        let detector = BlackSwanDetector::new(config).expect("Failed to create detector");
        let data = generate_black_swan_data(1000);
        
        b.iter(|| {
            let start = Instant::now();
            let result = detector.detect_anomalies(black_box(&data));
            let duration = start.elapsed();
            
            // Regression threshold: should complete within 2x target
            if duration.as_nanos() > (BLACK_SWAN_TARGET_NS * 2) as u128 {
                panic!("Black Swan regression detected: {}ns > {}ns", 
                       duration.as_nanos(), BLACK_SWAN_TARGET_NS * 2);
            }
            
            black_box(result)
        })
    });
    
    // SOC regression test
    group.bench_function("soc_regression_1000", |b| {
        let params = SOCParameters::default();
        let analyzer = SOCAnalyzer::new(params);
        let (data, _) = generate_financial_data(1000, 0.02);
        
        b.iter(|| {
            let start = Instant::now();
            let result = analyzer.analyze_series(black_box(&data));
            let duration = start.elapsed();
            
            // Regression threshold: should complete within 2x target
            if duration.as_nanos() > (SOC_TARGET_NS * 2) as u128 {
                panic!("SOC regression detected: {}ns > {}ns", 
                       duration.as_nanos(), SOC_TARGET_NS * 2);
            }
            
            black_box(result)
        })
    });
    
    // STDP regression test
    group.bench_function("stdp_regression_100x100", |b| {
        let config = STDPConfig {
            simd_width: 8,
            parallel_enabled: false,
            ..Default::default()
        };
        let mut optimizer = STDPOptimizer::new(config).expect("Failed to create optimizer");
        let weights = Array2::from_elem((100, 100), 0.5);
        
        b.iter(|| {
            let start = Instant::now();
            let result = optimizer.update_weights(black_box(&weights), 0.01);
            let duration = start.elapsed();
            
            // Regression threshold: should complete within 2x target
            if duration.as_nanos() > (STDP_TARGET_NS * 2) as u128 {
                panic!("STDP regression detected: {}ns > {}ns", 
                       duration.as_nanos(), STDP_TARGET_NS * 2);
            }
            
            black_box(result)
        })
    });
    
    group.finish();
}

// === CRITERION GROUPS ===

criterion_group!(
    latency_benchmarks,
    bench_black_swan_latency,
    bench_soc_latency,
    bench_stdp_latency
);

criterion_group!(
    throughput_benchmarks,
    bench_black_swan_throughput,
    bench_soc_throughput,
    bench_antifragility_performance,
    bench_stdp_throughput
);

#[cfg(feature = "simd")]
criterion_group!(
    simd_benchmarks,
    bench_simd_vs_scalar
);

#[cfg(feature = "gpu")]
criterion_group!(
    gpu_benchmarks,
    bench_gpu_vs_cpu
);

criterion_group!(
    memory_benchmarks,
    bench_memory_patterns,
    bench_parallel_processing
);

criterion_group!(
    regression_benchmarks,
    bench_regression_suite
);

// Conditional criterion main based on features
#[cfg(all(feature = "simd", feature = "gpu"))]
criterion_main!(
    latency_benchmarks,
    throughput_benchmarks,
    simd_benchmarks,
    gpu_benchmarks,
    memory_benchmarks,
    regression_benchmarks
);

#[cfg(all(feature = "simd", not(feature = "gpu")))]
criterion_main!(
    latency_benchmarks,
    throughput_benchmarks,
    simd_benchmarks,
    memory_benchmarks,
    regression_benchmarks
);

#[cfg(all(not(feature = "simd"), feature = "gpu"))]
criterion_main!(
    latency_benchmarks,
    throughput_benchmarks,
    gpu_benchmarks,
    memory_benchmarks,
    regression_benchmarks
);

#[cfg(all(not(feature = "simd"), not(feature = "gpu")))]
criterion_main!(
    latency_benchmarks,
    throughput_benchmarks,
    memory_benchmarks,
    regression_benchmarks
);