//! Comprehensive benchmark suite for CDFA core diversity metrics
//!
//! Validates that core implementations meet performance targets

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cdfa_core::prelude::*;
use cdfa_core::benchmarks::*;
use std::time::Duration;

/// Generate test data for benchmarks
fn generate_test_data(size: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();
    let y: Vec<f64> = (0..size).map(|i| (i as f64).cos()).collect();
    (x, y)
}

/// Benchmark Kendall tau correlation
fn bench_kendall_tau(c: &mut Criterion) {
    let mut group = c.benchmark_group("kendall_tau");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [64, 128, 256, 512, 1024].iter() {
        let (x, y) = generate_test_data(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(kendall_tau_correlation(black_box(&x), black_box(&y)))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark Spearman rank correlation
fn bench_spearman_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("spearman_correlation");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [64, 128, 256, 512, 1024].iter() {
        let (x, y) = generate_test_data(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(spearman_correlation(black_box(&x), black_box(&y)))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark Jensen-Shannon divergence
fn bench_jensen_shannon_divergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("jensen_shannon_divergence");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [64, 128, 256, 512].iter() {
        let (x, y) = generate_test_data(*size);
        // Normalize to probability distributions
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let p: Vec<f64> = x.iter().map(|&v| v.abs() / sum_x.abs()).collect();
        let q: Vec<f64> = y.iter().map(|&v| v.abs() / sum_y.abs()).collect();
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(jensen_shannon_divergence(black_box(&p), black_box(&q)))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark DTW distance
fn bench_dtw_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtw_distance");
    group.measurement_time(Duration::from_secs(15));
    
    for size in [32, 64, 128, 256].iter() {
        let (x, y) = generate_test_data(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(dtw_distance(black_box(&x), black_box(&y)))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark signal fusion
fn bench_signal_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("signal_fusion");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [64, 128, 256, 512, 1024].iter() {
        let (x, y) = generate_test_data(*size);
        let signals = vec![x.clone(), y.clone()];
        let weights = vec![0.6, 0.4];
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("score_based", size),
            size,
            |b, _| {
                b.iter(|| {
                    let fusion = ScoreBasedFusion::new(black_box(weights.clone()));
                    black_box(fusion.fuse(black_box(&signals)))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("rank_based", size),
            size,
            |b, _| {
                b.iter(|| {
                    let fusion = RankBasedFusion::new(black_box(weights.clone()));
                    black_box(fusion.fuse(black_box(&signals)))
                })
            },
        );
    }
    
    group.finish();
}

/// Performance validation benchmark
fn bench_performance_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_validation");
    group.measurement_time(Duration::from_secs(30));
    
    group.bench_function("full_benchmark_suite", |b| {
        b.iter(|| {
            let results = run_core_benchmarks();
            black_box(results)
        })
    });
    
    group.bench_function("performance_target_validation", |b| {
        b.iter(|| {
            let meets_targets = validate_core_performance_targets();
            black_box(meets_targets)
        })
    });
    
    group.finish();
    
    // Run validation and print results
    let results = run_core_benchmarks();
    let meets_targets = validate_core_performance_targets();
    
    println!("\n=== Core CDFA Performance Results ===");
    println!("Kendall Tau: {} ns", results.kendall_tau_ns);
    println!("Spearman: {} ns", results.spearman_ns);
    println!("Jensen-Shannon: {} ns", results.jensen_shannon_ns);
    println!("DTW Distance: {} ns", results.dtw_ns);
    println!("Score Fusion: {} ns", results.score_fusion_ns);
    println!("Rank Fusion: {} ns", results.rank_fusion_ns);
    println!("Performance Targets Met: {}", meets_targets);
    
    if meets_targets {
        println!("\n🎉 ALL CORE PERFORMANCE TARGETS MET! 🎉");
        println!("Core CDFA implementations are delivering expected performance.");
    } else {
        println!("\n⚠️  Some core performance targets not met.");
        println!("This may be due to debug build or system limitations.");
    }
}

criterion_group!(
    benches,
    bench_kendall_tau,
    bench_spearman_correlation,
    bench_jensen_shannon_divergence,
    bench_dtw_distance,
    bench_signal_fusion,
    bench_performance_validation,
);

criterion_main!(benches);