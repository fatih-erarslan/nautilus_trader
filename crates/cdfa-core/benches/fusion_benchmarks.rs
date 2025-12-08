//! Fusion strategy benchmarks for CDFA core
//!
//! Validates that fusion implementations meet performance targets

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cdfa_core::prelude::*;
use std::time::Duration;

/// Generate test signals for benchmarks
fn generate_test_signals(num_signals: usize, signal_length: usize) -> Vec<Vec<f64>> {
    (0..num_signals)
        .map(|i| {
            (0..signal_length)
                .map(|j| ((i + j) as f64).sin() + (i as f64) * 0.1)
                .collect()
        })
        .collect()
}

/// Benchmark score-based fusion
fn bench_score_based_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("score_based_fusion");
    group.measurement_time(Duration::from_secs(10));
    
    for signal_count in [2, 4, 8, 16].iter() {
        for signal_length in [64, 128, 256, 512].iter() {
            let signals = generate_test_signals(*signal_count, *signal_length);
            let weights: Vec<f64> = (0..*signal_count).map(|i| 1.0 / *signal_count as f64).collect();
            
            group.throughput(Throughput::Elements((*signal_count * *signal_length) as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}x{}", signal_count, signal_length), signal_count),
                &(signals, weights),
                |b, (signals, weights)| {
                    b.iter(|| {
                        let fusion = ScoreBasedFusion::new(black_box(weights.clone()));
                        black_box(fusion.fuse(black_box(signals)))
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark rank-based fusion
fn bench_rank_based_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank_based_fusion");
    group.measurement_time(Duration::from_secs(10));
    
    for signal_count in [2, 4, 8, 16].iter() {
        for signal_length in [64, 128, 256, 512].iter() {
            let signals = generate_test_signals(*signal_count, *signal_length);
            let weights: Vec<f64> = (0..*signal_count).map(|i| 1.0 / *signal_count as f64).collect();
            
            group.throughput(Throughput::Elements((*signal_count * *signal_length) as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}x{}", signal_count, signal_length), signal_count),
                &(signals, weights),
                |b, (signals, weights)| {
                    b.iter(|| {
                        let fusion = RankBasedFusion::new(black_box(weights.clone()));
                        black_box(fusion.fuse(black_box(signals)))
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark hybrid fusion
fn bench_hybrid_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_fusion");
    group.measurement_time(Duration::from_secs(10));
    
    for signal_count in [2, 4, 8].iter() {
        for signal_length in [64, 128, 256].iter() {
            let signals = generate_test_signals(*signal_count, *signal_length);
            let weights: Vec<f64> = (0..*signal_count).map(|i| 1.0 / *signal_count as f64).collect();
            
            group.throughput(Throughput::Elements((*signal_count * *signal_length) as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}x{}", signal_count, signal_length), signal_count),
                &(signals, weights),
                |b, (signals, weights)| {
                    b.iter(|| {
                        let fusion = HybridFusion::new(
                            black_box(weights.clone()),
                            black_box(0.5), // alpha parameter
                        );
                        black_box(fusion.fuse(black_box(signals)))
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark adaptive fusion
fn bench_adaptive_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_fusion");
    group.measurement_time(Duration::from_secs(15));
    
    for signal_count in [2, 4, 8].iter() {
        for signal_length in [64, 128, 256].iter() {
            let signals = generate_test_signals(*signal_count, *signal_length);
            
            group.throughput(Throughput::Elements((*signal_count * *signal_length) as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}x{}", signal_count, signal_length), signal_count),
                &signals,
                |b, signals| {
                    b.iter(|| {
                        let fusion = AdaptiveFusion::new();
                        black_box(fusion.fuse(black_box(signals)))
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark fusion strategy comparison
fn bench_fusion_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_comparison");
    group.measurement_time(Duration::from_secs(20));
    
    let signals = generate_test_signals(4, 256);
    let weights = vec![0.25, 0.25, 0.25, 0.25];
    
    group.bench_function("score_based", |b| {
        b.iter(|| {
            let fusion = ScoreBasedFusion::new(black_box(weights.clone()));
            black_box(fusion.fuse(black_box(&signals)))
        })
    });
    
    group.bench_function("rank_based", |b| {
        b.iter(|| {
            let fusion = RankBasedFusion::new(black_box(weights.clone()));
            black_box(fusion.fuse(black_box(&signals)))
        })
    });
    
    group.bench_function("hybrid", |b| {
        b.iter(|| {
            let fusion = HybridFusion::new(black_box(weights.clone()), black_box(0.5));
            black_box(fusion.fuse(black_box(&signals)))
        })
    });
    
    group.bench_function("adaptive", |b| {
        b.iter(|| {
            let fusion = AdaptiveFusion::new();
            black_box(fusion.fuse(black_box(&signals)))
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_score_based_fusion,
    bench_rank_based_fusion,
    bench_hybrid_fusion,
    bench_adaptive_fusion,
    bench_fusion_comparison,
);

criterion_main!(benches);