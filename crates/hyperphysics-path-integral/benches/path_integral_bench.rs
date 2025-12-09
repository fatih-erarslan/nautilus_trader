//! Benchmark for Path Integral Portfolio Optimizer

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hyperphysics_path_integral::PathIntegralOptimizer;

fn bench_path_sampling(c: &mut Criterion) {
    let optimizer = PathIntegralOptimizer::hyperphysics_default(5);
    let initial_weights = vec![0.2; 5];

    c.bench_function("path_sampling_5_assets", |b| {
        b.iter(|| {
            black_box(optimizer.optimize(&initial_weights))
        })
    });
}

fn bench_portfolio_optimization(c: &mut Criterion) {
    let optimizer = PathIntegralOptimizer::hyperphysics_default(10);
    let initial_weights = vec![0.1; 10];

    c.bench_function("portfolio_optimization_10_assets", |b| {
        b.iter(|| {
            black_box(optimizer.optimize(&initial_weights))
        })
    });
}

criterion_group!(benches, bench_path_sampling, bench_portfolio_optimization);
criterion_main!(benches);
