//! Fibonacci Analyzer Benchmarks
//!
//! This benchmark suite measures the performance of the CDFA Fibonacci Analyzer
//! across different configurations and data sizes to ensure sub-microsecond
//! performance requirements are met.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use cdfa_fibonacci_analyzer::*;
use std::collections::HashMap;

/// Generate benchmark data of specified size
fn generate_benchmark_data(size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut prices = Vec::with_capacity(size);
    let mut volumes = Vec::with_capacity(size);
    
    let base_price = 100.0;
    let volatility = 0.02;
    
    for i in 0..size {
        let trend = base_price + (i as f64 * 0.001);
        let noise = (i as f64 * 0.1).sin() * volatility * base_price;
        let swing = if i % 20 == 0 { 
            if i % 40 == 0 { 2.0 } else { -1.5 }
        } else { 
            0.0 
        };
        
        prices.push(trend + noise + swing);
        volumes.push(1000.0 + (i as f64 * 5.0).sin().abs() * 500.0);
    }
    
    (prices, volumes)
}

/// Benchmark basic Fibonacci analysis
fn benchmark_basic_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_analysis");
    
    let data_sizes = vec![50, 100, 200, 500, 1000];
    
    for size in data_sizes {
        let (prices, volumes) = generate_benchmark_data(size);
        let analyzer = FibonacciAnalyzer::default();
        
        group.bench_with_input(
            BenchmarkId::new("default_config", size),
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

/// Benchmark different configurations
fn benchmark_configurations(c: &mut Criterion) {
    let mut group = c.benchmark_group("configurations");
    let (prices, volumes) = generate_benchmark_data(200);
    
    let configs = vec![
        ("default", FibonacciConfig::default()),
        ("high_frequency", FibonacciConfig::high_frequency()),
        ("daily", FibonacciConfig::daily()),
        ("minimal", FibonacciConfig::minimal()),
    ];
    
    for (name, config) in configs {
        let analyzer = FibonacciAnalyzer::new(config);
        
        group.bench_function(name, |b| {
            b.iter(|| {
                analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
            })
        });
    }
    
    group.finish();
}

/// Benchmark preset configurations
fn benchmark_presets(c: &mut Criterion) {
    let mut group = c.benchmark_group("presets");
    let (prices, volumes) = generate_benchmark_data(200);
    
    let presets = vec![
        ("scalping", FibonacciPresets::scalping()),
        ("day_trading", FibonacciPresets::day_trading()),
        ("swing_trading", FibonacciPresets::swing_trading()),
        ("position_trading", FibonacciPresets::position_trading()),
        ("high_precision", FibonacciPresets::high_precision()),
    ];
    
    for (name, config) in presets {
        let analyzer = FibonacciAnalyzer::new(config);
        
        group.bench_function(name, |b| {
            b.iter(|| {
                analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
            })
        });
    }
    
    group.finish();
}

/// Benchmark swing point detection
fn benchmark_swing_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("swing_detection");
    let (prices, volumes) = generate_benchmark_data(500);
    
    let swing_periods = vec![5, 10, 14, 20, 30];
    
    for period in swing_periods {
        let config = FibonacciConfig::default().with_swing_period(period);
        let analyzer = FibonacciAnalyzer::new(config);
        
        group.bench_with_input(
            BenchmarkId::new("swing_period", period),
            &period,
            |b, _| {
                b.iter(|| {
                    analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark parallel vs sequential processing
fn benchmark_parallel_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_processing");
    let (prices, volumes) = generate_benchmark_data(1000);
    
    let parallel_config = FibonacciConfig::default().with_parallel(true);
    let sequential_config = FibonacciConfig::default().with_parallel(false);
    
    let parallel_analyzer = FibonacciAnalyzer::new(parallel_config);
    let sequential_analyzer = FibonacciAnalyzer::new(sequential_config);
    
    group.bench_function("parallel", |b| {
        b.iter(|| {
            parallel_analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
        })
    });
    
    group.bench_function("sequential", |b| {
        b.iter(|| {
            sequential_analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
        })
    });
    
    group.finish();
}

/// Benchmark cache performance
fn benchmark_cache_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_performance");
    let (prices, volumes) = generate_benchmark_data(200);
    
    let analyzer = FibonacciAnalyzer::default();
    
    // First run to populate cache
    let _ = analyzer.analyze(&prices, &volumes);
    
    group.bench_function("with_cache", |b| {
        b.iter(|| {
            analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
        })
    });
    
    // Clear cache and benchmark without cache
    analyzer.clear_cache();
    
    group.bench_function("without_cache", |b| {
        b.iter(|| {
            analyzer.clear_cache();
            analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
        })
    });
    
    group.finish();
}

/// Benchmark memory usage patterns
fn benchmark_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");
    
    let cache_sizes = vec![10, 100, 1000, 5000];
    let (prices, volumes) = generate_benchmark_data(200);
    
    for cache_size in cache_sizes {
        let config = FibonacciConfig::default().with_cache_size(cache_size);
        let analyzer = FibonacciAnalyzer::new(config);
        
        group.bench_with_input(
            BenchmarkId::new("cache_size", cache_size),
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

/// Benchmark sub-microsecond performance target
fn benchmark_sub_microsecond_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("sub_microsecond_target");
    group.measurement_time(std::time::Duration::from_secs(10));
    group.sample_size(1000);
    
    let (prices, volumes) = generate_benchmark_data(100);
    let analyzer = FibonacciAnalyzer::new(FibonacciConfig::minimal());
    
    group.bench_function("target_performance", |b| {
        b.iter(|| {
            analyzer.analyze(black_box(&prices), black_box(&volumes)).unwrap()
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_basic_analysis,
    benchmark_configurations,
    benchmark_presets,
    benchmark_swing_detection,
    benchmark_parallel_processing,
    benchmark_cache_performance,
    benchmark_memory_patterns,
    benchmark_sub_microsecond_target
);

criterion_main!(benches);