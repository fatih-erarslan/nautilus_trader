//! SIMD Fibonacci Analyzer Benchmarks
//!
//! This benchmark suite specifically tests SIMD acceleration performance
//! for the CDFA Fibonacci Analyzer, comparing SIMD vs scalar implementations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use cdfa_fibonacci_analyzer::*;
use std::collections::HashMap;

/// Generate SIMD-optimized benchmark data
fn generate_simd_benchmark_data(size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut prices = Vec::with_capacity(size);
    let mut volumes = Vec::with_capacity(size);
    
    let base_price = 100.0;
    let volatility = 0.02;
    
    for i in 0..size {
        let trend = base_price + (i as f64 * 0.001);
        let noise = (i as f64 * 0.1).sin() * volatility * base_price;
        let swing = if i % 15 == 0 { 
            if i % 30 == 0 { 1.8 } else { -1.2 }
        } else { 
            0.0 
        };
        
        prices.push(trend + noise + swing);
        volumes.push(1000.0 + (i as f64 * 7.0).sin().abs() * 400.0);
    }
    
    (prices, volumes)
}

/// Benchmark SIMD vs scalar performance
fn benchmark_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar");
    
    let data_sizes = vec![100, 500, 1000, 2000];
    
    for size in data_sizes {
        let (prices, volumes) = generate_simd_benchmark_data(size);
        
        // SIMD enabled configuration
        let simd_config = FibonacciConfig::default().with_simd(true);
        let simd_analyzer = FibonacciAnalyzer::new(simd_config);
        
        // Scalar configuration
        let scalar_config = FibonacciConfig::default().with_simd(false);
        let scalar_analyzer = FibonacciAnalyzer::new(scalar_config);
        
        group.bench_with_input(
            BenchmarkId::new("simd_enabled", size),
            &size,
            |b, _| {
                b.iter(|| {
                    simd_analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("scalar_only", size),
            &size,
            |b, _| {
                b.iter(|| {
                    scalar_analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark SIMD with different swing periods
fn benchmark_simd_swing_periods(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_swing_periods");
    let (prices, volumes) = generate_simd_benchmark_data(1000);
    
    let swing_periods = vec![5, 10, 14, 20, 30, 50];
    
    for period in swing_periods {
        let simd_config = FibonacciConfig::default()
            .with_simd(true)
            .with_swing_period(period);
        let simd_analyzer = FibonacciAnalyzer::new(simd_config);
        
        let scalar_config = FibonacciConfig::default()
            .with_simd(false)
            .with_swing_period(period);
        let scalar_analyzer = FibonacciAnalyzer::new(scalar_config);
        
        group.bench_with_input(
            BenchmarkId::new("simd", period),
            &period,
            |b, _| {
                b.iter(|| {
                    simd_analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("scalar", period),
            &period,
            |b, _| {
                b.iter(|| {
                    scalar_analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark SIMD with parallel processing
fn benchmark_simd_parallel_combination(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_parallel_combination");
    let (prices, volumes) = generate_simd_benchmark_data(2000);
    
    let configurations = vec![
        ("simd_only", FibonacciConfig::default().with_simd(true).with_parallel(false)),
        ("parallel_only", FibonacciConfig::default().with_simd(false).with_parallel(true)),
        ("simd_and_parallel", FibonacciConfig::default().with_simd(true).with_parallel(true)),
        ("neither", FibonacciConfig::default().with_simd(false).with_parallel(false)),
    ];
    
    for (name, config) in configurations {
        let analyzer = FibonacciAnalyzer::new(config);
        
        group.bench_function(name, |b| {
            b.iter(|| {
                analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
            })
        });
    }
    
    group.finish();
}

/// Benchmark SIMD with different tolerance levels
fn benchmark_simd_tolerance_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_tolerance_levels");
    let (prices, volumes) = generate_simd_benchmark_data(500);
    
    let tolerance_levels = vec![0.001, 0.003, 0.006, 0.01, 0.02];
    
    for tolerance in tolerance_levels {
        let simd_config = FibonacciConfig::default()
            .with_simd(true)
            .with_alignment_tolerance(tolerance);
        let simd_analyzer = FibonacciAnalyzer::new(simd_config);
        
        group.bench_with_input(
            BenchmarkId::new("simd_tolerance", (tolerance * 1000.0) as u32),
            &tolerance,
            |b, _| {
                b.iter(|| {
                    simd_analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark SIMD with high-precision configuration
fn benchmark_simd_high_precision(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_high_precision");
    let (prices, volumes) = generate_simd_benchmark_data(1000);
    
    let high_precision_config = FibonacciPresets::high_precision()
        .with_simd(true)
        .with_parallel(true);
    let high_precision_analyzer = FibonacciAnalyzer::new(high_precision_config);
    
    let standard_config = FibonacciConfig::default()
        .with_simd(true)
        .with_parallel(true);
    let standard_analyzer = FibonacciAnalyzer::new(standard_config);
    
    group.bench_function("high_precision_simd", |b| {
        b.iter(|| {
            high_precision_analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
        })
    });
    
    group.bench_function("standard_simd", |b| {
        b.iter(|| {
            standard_analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
        })
    });
    
    group.finish();
}

/// Benchmark SIMD performance scaling
fn benchmark_simd_performance_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_performance_scaling");
    
    let data_sizes = vec![50, 100, 200, 500, 1000, 2000, 5000];
    let config = FibonacciConfig::default().with_simd(true).with_parallel(true);
    let analyzer = FibonacciAnalyzer::new(config);
    
    for size in data_sizes {
        let (prices, volumes) = generate_simd_benchmark_data(size);
        
        group.bench_with_input(
            BenchmarkId::new("simd_scaling", size),
            &size,
            |b, _| {
                b.iter(|| {
                    analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark SIMD cache efficiency
fn benchmark_simd_cache_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_cache_efficiency");
    let (prices, volumes) = generate_simd_benchmark_data(1000);
    
    let cache_sizes = vec![10, 100, 1000, 5000];
    
    for cache_size in cache_sizes {
        let config = FibonacciConfig::default()
            .with_simd(true)
            .with_cache_size(cache_size);
        let analyzer = FibonacciAnalyzer::new(config);
        
        group.bench_with_input(
            BenchmarkId::new("simd_cache", cache_size),
            &cache_size,
            |b, _| {
                b.iter(|| {
                    analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark SIMD memory alignment impact
fn benchmark_simd_memory_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_memory_alignment");
    
    // Test with different data alignment scenarios
    let size = 1000;
    let config = FibonacciConfig::default().with_simd(true);
    let analyzer = FibonacciAnalyzer::new(config);
    
    // Aligned data
    let (aligned_prices, aligned_volumes) = generate_simd_benchmark_data(size);
    
    group.bench_function("aligned_data", |b| {
        b.iter(|| {
            analyzer.analyze(black_box(&aligned_prices), black_box(&aligned_volumes)).unwrap()
        })
    });
    
    // Create slightly misaligned data by prepending a single element
    let mut misaligned_prices = vec![99.0];
    let mut misaligned_volumes = vec![999.0];
    misaligned_prices.extend_from_slice(&aligned_prices);
    misaligned_volumes.extend_from_slice(&aligned_volumes);
    
    group.bench_function("misaligned_data", |b| {
        b.iter(|| {
            analyzer.analyze(black_box(&misaligned_prices[1..]), black_box(&misaligned_volumes[1..])).unwrap()
        })
    });
    
    group.finish();
}

criterion_group!(
    simd_benches,
    benchmark_simd_vs_scalar,
    benchmark_simd_swing_periods,
    benchmark_simd_parallel_combination,
    benchmark_simd_tolerance_levels,
    benchmark_simd_high_precision,
    benchmark_simd_performance_scaling,
    benchmark_simd_cache_efficiency,
    benchmark_simd_memory_alignment
);

criterion_main!(simd_benches);