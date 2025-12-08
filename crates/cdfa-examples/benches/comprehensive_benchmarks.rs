//! Comprehensive benchmarks showcasing CDFA performance
//!
//! End-to-end benchmarks demonstrating the full CDFA pipeline

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cdfa_core::prelude::*;
use cdfa_simd::unified;
use cdfa_parallel::prelude::*;
use std::time::Duration;

/// Generate comprehensive test dataset
fn generate_comprehensive_dataset(num_signals: usize, signal_length: usize) -> Vec<Vec<f64>> {
    (0..num_signals)
        .map(|i| {
            (0..signal_length)
                .map(|j| {
                    let base = (j as f64 * 0.1).sin();
                    let trend = (i as f64) * 0.01 * j as f64;
                    let noise = ((i + j) as f64 * 0.123).sin() * 0.1;
                    base + trend + noise
                })
                .collect()
        })
        .collect()
}

/// Benchmark full CDFA pipeline
fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_cdfa_pipeline");
    group.measurement_time(Duration::from_secs(30));
    
    for num_signals in [4, 8, 16].iter() {
        for signal_length in [256, 512, 1024].iter() {
            let signals = generate_comprehensive_dataset(*num_signals, *signal_length);
            
            group.throughput(Throughput::Elements((*num_signals * *signal_length) as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}x{}", num_signals, signal_length), num_signals),
                &signals,
                |b, signals| {
                    b.iter(|| {
                        // Step 1: Calculate diversity metrics
                        let mut diversity_matrix = vec![vec![0.0; signals.len()]; signals.len()];
                        for i in 0..signals.len() {
                            for j in i+1..signals.len() {
                                let correlation = unified::correlation(&signals[i], &signals[j]);
                                diversity_matrix[i][j] = 1.0 - correlation.abs();
                                diversity_matrix[j][i] = diversity_matrix[i][j];
                            }
                        }
                        
                        // Step 2: Calculate weights based on diversity
                        let weights: Vec<f64> = (0..signals.len())
                            .map(|i| {
                                let avg_diversity: f64 = diversity_matrix[i].iter().sum::<f64>() / signals.len() as f64;
                                avg_diversity.max(0.1) // Minimum weight
                            })
                            .collect();
                        
                        // Step 3: Normalize weights
                        let weight_sum: f64 = weights.iter().sum();
                        let normalized_weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();
                        
                        // Step 4: Perform fusion
                        let fusion = ScoreBasedFusion::new(normalized_weights);
                        let result = fusion.fuse(signals);
                        
                        black_box(result)
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark SIMD vs Scalar performance comparison
fn bench_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar");
    group.measurement_time(Duration::from_secs(20));
    
    for size in [256, 512, 1024, 2048].iter() {
        let x = (0..*size).map(|i| (i as f64).sin()).collect::<Vec<_>>();
        let y = (0..*size).map(|i| (i as f64).cos()).collect::<Vec<_>>();
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("simd_correlation", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(unified::correlation(black_box(&x), black_box(&y)))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("scalar_correlation", size),
            size,
            |b, _| {
                b.iter(|| {
                    // Scalar implementation
                    let n = x.len() as f64;
                    let sum_x: f64 = x.iter().sum();
                    let sum_y: f64 = y.iter().sum();
                    let sum_xx: f64 = x.iter().map(|&v| v * v).sum();
                    let sum_yy: f64 = y.iter().map(|&v| v * v).sum();
                    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
                    
                    let numerator = n * sum_xy - sum_x * sum_y;
                    let denominator = ((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)).sqrt();
                    
                    black_box(if denominator > 0.0 { numerator / denominator } else { 0.0 })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark parallel vs sequential processing
fn bench_parallel_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential");
    group.measurement_time(Duration::from_secs(25));
    
    for num_signals in [8, 16, 32].iter() {
        let signals = generate_comprehensive_dataset(*num_signals, 512);
        
        group.throughput(Throughput::Elements(*num_signals as u64));
        
        group.bench_with_input(
            BenchmarkId::new("parallel_processing", num_signals),
            &signals,
            |b, signals| {
                b.iter(|| {
                    let processor = ParallelProcessor::new();
                    black_box(processor.process_signals(black_box(signals)))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("sequential_processing", num_signals),
            &signals,
            |b, signals| {
                b.iter(|| {
                    let mut results = Vec::new();
                    for signal in signals {
                        // Sequential processing
                        let variance = unified::variance(signal);
                        let mean = signal.iter().sum::<f64>() / signal.len() as f64;
                        results.push((mean, variance));
                    }
                    black_box(results)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory efficiency
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    group.measurement_time(Duration::from_secs(15));
    
    for size in [1024, 2048, 4096, 8192].iter() {
        group.bench_with_input(
            BenchmarkId::new("cache_aligned_operations", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let aligned_data = AlignedVec::<f64>::new(size);
                    let mut sum = 0.0;
                    for i in 0..size {
                        sum += (i as f64).sin();
                    }
                    black_box(sum)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("regular_vec_operations", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let _regular_data = vec![0.0f64; size];
                    let mut sum = 0.0;
                    for i in 0..size {
                        sum += (i as f64).sin();
                    }
                    black_box(sum)
                })
            },
        );
    }
    
    group.finish();
}

/// Real-world trading scenario benchmark
fn bench_trading_scenario(c: &mut Criterion) {
    let mut group = c.benchmark_group("trading_scenario");
    group.measurement_time(Duration::from_secs(30));
    
    // Simulate real trading data: multiple indicators across time
    let indicators = [
        "RSI", "MACD", "Bollinger", "Stochastic", "Williams%R", 
        "CCI", "ADX", "Momentum", "ROC", "Ultimate"
    ];
    
    for time_window in [100, 200, 500, 1000].iter() {
        let signals = generate_comprehensive_dataset(indicators.len(), *time_window);
        
        group.throughput(Throughput::Elements((indicators.len() * *time_window) as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("indicators_{}bars", time_window), time_window),
            &signals,
            |b, signals| {
                b.iter(|| {
                    // Real-world CDFA trading pipeline
                    
                    // 1. Calculate all pairwise diversities
                    let mut diversity_scores = Vec::new();
                    for i in 0..signals.len() {
                        for j in i+1..signals.len() {
                            let kendall = kendall_tau_correlation(&signals[i], &signals[j]);
                            let spearman = spearman_correlation(&signals[i], &signals[j]);
                            let avg_diversity = 1.0 - (kendall.abs() + spearman.abs()) / 2.0;
                            diversity_scores.push(avg_diversity);
                        }
                    }
                    
                    // 2. Calculate dynamic weights
                    let total_diversity: f64 = diversity_scores.iter().sum();
                    let weights: Vec<f64> = (0..signals.len())
                        .map(|_| 1.0 / signals.len() as f64)
                        .collect();
                    
                    // 3. Apply multiple fusion strategies
                    let score_fusion = ScoreBasedFusion::new(weights.clone());
                    let rank_fusion = RankBasedFusion::new(weights.clone());
                    let hybrid_fusion = HybridFusion::new(weights, 0.6);
                    
                    let score_result = score_fusion.fuse(signals);
                    let rank_result = rank_fusion.fuse(signals);
                    let hybrid_result = hybrid_fusion.fuse(signals);
                    
                    black_box((score_result, rank_result, hybrid_result, total_diversity))
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_full_pipeline,
    bench_simd_vs_scalar,
    bench_parallel_vs_sequential,
    bench_memory_efficiency,
    bench_trading_scenario,
);

criterion_main!(benches);