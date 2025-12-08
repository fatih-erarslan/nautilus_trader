//! Parallel scaling benchmarks for CDFA
//!
//! Tests scalability across different core counts

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cdfa_core::prelude::*;
use cdfa_parallel::prelude::*;
use std::time::Duration;

/// Generate scaling test data
fn generate_scaling_data(size: usize) -> Vec<Vec<f64>> {
    (0..size)
        .map(|i| {
            (0..1000)
                .map(|j| ((i + j) as f64 * 0.1).sin())
                .collect()
        })
        .collect()
}

/// Benchmark parallel scaling efficiency
fn bench_parallel_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_scaling");
    group.measurement_time(Duration::from_secs(30));
    
    let data_sizes = [8, 16, 32, 64];
    let thread_counts = [1, 2, 4, 8, 16];
    
    for &data_size in &data_sizes {
        let signals = generate_scaling_data(data_size);
        
        for &thread_count in &thread_counts {
            group.throughput(Throughput::Elements(data_size as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}signals_{}threads", data_size, thread_count), thread_count),
                &thread_count,
                |b, &thread_count| {
                    b.iter(|| {
                        let config = ParallelConfig::new()
                            .with_num_threads(thread_count)
                            .with_batch_size(data_size / thread_count.max(1));
                        let processor = ParallelProcessor::with_config(config);
                        black_box(processor.process_signals(black_box(&signals)))
                    })
                },
            );
        }
    }
    
    group.finish();
}

criterion_group!(benches, bench_parallel_scaling);
criterion_main!(benches);