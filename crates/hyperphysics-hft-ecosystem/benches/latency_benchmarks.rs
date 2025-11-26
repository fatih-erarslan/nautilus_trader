//! Latency benchmarks for HFT ecosystem
//!
//! Verifies sub-millisecond performance claims for biomimetic algorithms.
//!
//! Run with: `cargo bench --features optimization-real`

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

#[cfg(feature = "optimization-real")]
use hyperphysics_hft_ecosystem::swarms::{RealOptimizer, MarketObjective};

/// Create test market objective
#[cfg(feature = "optimization-real")]
fn create_test_objective(dimension: usize) -> MarketObjective {
    MarketObjective {
        returns: vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.003, 0.012, -0.007, 0.005],
        volatility: 0.02,
        trend: 0.3,
        risk_aversion: 1.0,
    }
}

#[cfg(feature = "optimization-real")]
fn benchmark_whale_optimization(c: &mut Criterion) {
    let optimizer = RealOptimizer::hft_optimized().expect("Failed to create optimizer");
    let objective = create_test_objective(10);

    c.bench_function("whale_optimization_10d", |b| {
        b.iter(|| {
            optimizer.optimize_whale(black_box(&objective))
        })
    });
}

#[cfg(feature = "optimization-real")]
fn benchmark_bat_algorithm(c: &mut Criterion) {
    let optimizer = RealOptimizer::hft_optimized().expect("Failed to create optimizer");
    let objective = create_test_objective(10);

    c.bench_function("bat_algorithm_10d", |b| {
        b.iter(|| {
            optimizer.optimize_bat(black_box(&objective))
        })
    });
}

#[cfg(feature = "optimization-real")]
fn benchmark_firefly_algorithm(c: &mut Criterion) {
    let optimizer = RealOptimizer::hft_optimized().expect("Failed to create optimizer");
    let objective = create_test_objective(10);

    c.bench_function("firefly_algorithm_10d", |b| {
        b.iter(|| {
            optimizer.optimize_firefly(black_box(&objective))
        })
    });
}

#[cfg(feature = "optimization-real")]
fn benchmark_cuckoo_search(c: &mut Criterion) {
    let optimizer = RealOptimizer::hft_optimized().expect("Failed to create optimizer");
    let objective = create_test_objective(10);

    c.bench_function("cuckoo_search_10d", |b| {
        b.iter(|| {
            optimizer.optimize_cuckoo(black_box(&objective))
        })
    });
}

#[cfg(feature = "optimization-real")]
fn benchmark_tier1_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier1_algorithms");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(100);

    let optimizer = RealOptimizer::hft_optimized().expect("Failed to create optimizer");
    let objective = create_test_objective(10);

    group.bench_function("whale", |b| {
        b.iter(|| optimizer.optimize_whale(black_box(&objective)))
    });

    group.bench_function("bat", |b| {
        b.iter(|| optimizer.optimize_bat(black_box(&objective)))
    });

    group.bench_function("firefly", |b| {
        b.iter(|| optimizer.optimize_firefly(black_box(&objective)))
    });

    group.bench_function("cuckoo", |b| {
        b.iter(|| optimizer.optimize_cuckoo(black_box(&objective)))
    });

    group.finish();
}

#[cfg(feature = "optimization-real")]
fn benchmark_latency_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_scaling");
    group.measurement_time(Duration::from_secs(3));

    for iterations in [10, 25, 50, 100].iter() {
        let optimizer = RealOptimizer::new().expect("Failed to create optimizer");
        let objective = create_test_objective(10);

        group.bench_with_input(
            BenchmarkId::new("whale_iterations", iterations),
            iterations,
            |b, _iters| {
                b.iter(|| optimizer.optimize_whale(black_box(&objective)))
            },
        );
    }

    group.finish();
}

#[cfg(feature = "optimization-real")]
criterion_group!(
    benches,
    benchmark_whale_optimization,
    benchmark_bat_algorithm,
    benchmark_firefly_algorithm,
    benchmark_cuckoo_search,
    benchmark_tier1_latency,
    benchmark_latency_scaling,
);

#[cfg(feature = "optimization-real")]
criterion_main!(benches);

// Fallback when feature not enabled
#[cfg(not(feature = "optimization-real"))]
fn main() {
    eprintln!("Benchmarks require 'optimization-real' feature.");
    eprintln!("Run with: cargo bench --features optimization-real");
}
