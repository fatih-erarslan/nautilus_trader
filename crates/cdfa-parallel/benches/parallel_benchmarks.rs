//! Parallel processing benchmarks for CDFA
//!
//! Validates that parallel implementations meet performance targets

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cdfa_parallel::prelude::*;
use std::time::Duration;

/// Generate test data for benchmarks
fn generate_test_data(size: usize) -> Vec<f64> {
    (0..size).map(|i| (i as f64).sin()).collect()
}

/// Generate multiple signals for parallel processing
fn generate_multiple_signals(num_signals: usize, signal_length: usize) -> Vec<Vec<f64>> {
    (0..num_signals)
        .map(|i| {
            (0..signal_length)
                .map(|j| ((i + j) as f64).sin() + (i as f64) * 0.1)
                .collect()
        })
        .collect()
}

/// Benchmark parallel signal processing
fn bench_parallel_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_processing");
    group.measurement_time(Duration::from_secs(15));
    
    for num_signals in [4, 8, 16, 32].iter() {
        for signal_length in [256, 512, 1024, 2048].iter() {
            let signals = generate_multiple_signals(*num_signals, *signal_length);
            
            group.throughput(Throughput::Elements((*num_signals * *signal_length) as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}x{}", num_signals, signal_length), num_signals),
                &signals,
                |b, signals| {
                    b.iter(|| {
                        let processor = ParallelProcessor::new();
                        black_box(processor.process_signals(black_box(signals)))
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark lock-free data structures
fn bench_lockfree_structures(c: &mut Criterion) {
    let mut group = c.benchmark_group("lockfree_structures");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [1024, 2048, 4096, 8192].iter() {
        let data = generate_test_data(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("signal_buffer", size),
            size,
            |b, _| {
                b.iter(|| {
                    let buffer = LockFreeSignalBuffer::new(*size);
                    for &value in &data {
                        buffer.push(SignalData::new(value, 0.0, std::time::Instant::now()));
                    }
                    black_box(buffer.len())
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("result_queue", size),
            size,
            |b, _| {
                b.iter(|| {
                    let queue = LockFreeResultQueue::new();
                    for (i, &value) in data.iter().enumerate() {
                        queue.push(ProcessingResult::new(i, value, Duration::from_nanos(100)));
                    }
                    black_box(queue.len())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark parallel diversity metrics
fn bench_parallel_diversity(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_diversity");
    group.measurement_time(Duration::from_secs(20));
    
    for num_pairs in [10, 25, 50, 100].iter() {
        let signals = generate_multiple_signals(*num_pairs * 2, 512);
        
        group.throughput(Throughput::Elements(*num_pairs as u64));
        group.bench_with_input(
            BenchmarkId::new("kendall_tau", num_pairs),
            &signals,
            |b, signals| {
                b.iter(|| {
                    let calculator = ParallelDiversityCalculator::new();
                    black_box(calculator.calculate_all_pairs_kendall_tau(black_box(signals)))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("spearman", num_pairs),
            &signals,
            |b, signals| {
                b.iter(|| {
                    let calculator = ParallelDiversityCalculator::new();
                    black_box(calculator.calculate_all_pairs_spearman(black_box(signals)))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark parallel fusion strategies
fn bench_parallel_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_fusion");
    group.measurement_time(Duration::from_secs(15));
    
    for num_signals in [4, 8, 16, 32].iter() {
        for signal_length in [256, 512, 1024].iter() {
            let signals = generate_multiple_signals(*num_signals, *signal_length);
            let weights: Vec<f64> = (0..*num_signals).map(|_| 1.0 / *num_signals as f64).collect();
            
            group.throughput(Throughput::Elements((*num_signals * *signal_length) as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("score_fusion_{}x{}", num_signals, signal_length), num_signals),
                &(signals.clone(), weights.clone()),
                |b, (signals, weights)| {
                    b.iter(|| {
                        let fusion = ParallelScoreFusion::new(black_box(weights.clone()));
                        black_box(fusion.fuse_parallel(black_box(signals)))
                    })
                },
            );
            
            group.bench_with_input(
                BenchmarkId::new(format!("rank_fusion_{}x{}", num_signals, signal_length), num_signals),
                &(signals, weights),
                |b, (signals, weights)| {
                    b.iter(|| {
                        let fusion = ParallelRankFusion::new(black_box(weights.clone()));
                        black_box(fusion.fuse_parallel(black_box(signals)))
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark threading efficiency
fn bench_threading_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("threading_efficiency");
    group.measurement_time(Duration::from_secs(20));
    
    let signals = generate_multiple_signals(16, 1024);
    
    for num_threads in [1, 2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("processing_threads", num_threads),
            num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let config = ParallelConfig::new()
                        .with_num_threads(num_threads)
                        .with_batch_size(64);
                    let processor = ParallelProcessor::with_config(config);
                    black_box(processor.process_signals(black_box(&signals)))
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
    
    group.bench_function("full_parallel_benchmark_suite", |b| {
        b.iter(|| {
            let results = run_parallel_benchmarks();
            black_box(results)
        })
    });
    
    group.bench_function("parallel_performance_target_validation", |b| {
        b.iter(|| {
            let meets_targets = validate_parallel_performance_targets();
            black_box(meets_targets)
        })
    });
    
    group.finish();
    
    // Run validation and print results
    let results = run_parallel_benchmarks();
    let meets_targets = validate_parallel_performance_targets();
    
    println!("\n=== Parallel CDFA Performance Results ===");
    println!("Parallel Processing: {} ns/signal", results.parallel_processing_ns);
    println!("Lock-Free Buffer: {} ns/operation", results.lockfree_buffer_ns);
    println!("Parallel Diversity: {} ns/pair", results.parallel_diversity_ns);
    println!("Parallel Fusion: {} ns/signal", results.parallel_fusion_ns);
    println!("Threading Efficiency: {}x speedup", results.threading_speedup);
    println!("Performance Targets Met: {}", meets_targets);
    
    if meets_targets {
        println!("\n🎉 ALL PARALLEL PERFORMANCE TARGETS MET! 🎉");
        println!("Parallel CDFA implementations are delivering expected performance.");
    } else {
        println!("\n⚠️  Some parallel performance targets not met.");
        println!("This may be due to debug build or system limitations.");
    }
}

criterion_group!(
    benches,
    bench_parallel_processing,
    bench_lockfree_structures,
    bench_parallel_diversity,
    bench_parallel_fusion,
    bench_threading_efficiency,
    bench_performance_validation,
);

criterion_main!(benches);