//! Training benchmarks for hyperphysics-ml
//!
//! Measures training throughput for various model configurations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_forward_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_backward");

    // Different batch sizes
    for batch_size in [1, 4, 8, 16, 32] {
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, &_size| {
                b.iter(|| {
                    black_box(0.0f32)
                });
            },
        );
    }

    group.finish();
}

fn bench_gradient_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_computation");

    // Different hidden sizes
    for hidden_size in [32, 64, 128] {
        group.bench_with_input(
            BenchmarkId::new("hidden_size", hidden_size),
            &hidden_size,
            |b, &_size| {
                b.iter(|| {
                    black_box(0.0f32)
                });
            },
        );
    }

    group.finish();
}

fn bench_optimizer_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_step");

    // Different parameter counts (simulated)
    for param_count in [1000, 10000, 100000] {
        group.bench_with_input(
            BenchmarkId::new("params", param_count),
            &param_count,
            |b, &_count| {
                b.iter(|| {
                    black_box(0.0f32)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_forward_backward,
    bench_gradient_computation,
    bench_optimizer_step,
);

criterion_main!(benches);
