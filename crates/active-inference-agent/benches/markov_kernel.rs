//! Markovian Kernel Benchmarks
//!
//! Validates performance of conscious agent Markovian operations.
//!
//! ## Mathematical Foundation
//! - K(x, A) is a probability measure-preserving transformation
//! - K * μ computes the pushforward measure
//! - Entropy rate: H(K) = -Σᵢ πᵢ Σⱼ Kᵢⱼ log(Kᵢⱼ)
//!
//! Reference: Hoffman (2012) "Conscious Realism and the Mind-Body Problem"

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use active_inference_agent::markov_kernel::{
    MarkovianKernel, PerceptionKernel, DecisionKernel, ActionKernel
};
use nalgebra as na;

/// Create deterministic stochastic matrix (NOT random)
fn create_stochastic_matrix(dim: usize, seed: usize) -> na::DMatrix<f64> {
    let mut matrix = na::DMatrix::zeros(dim, dim);

    for i in 0..dim {
        let mut row_sum = 0.0;
        for j in 0..dim {
            // Deterministic transition probabilities based on distance
            let dist = ((i as isize - j as isize).abs() + 1) as f64;
            let val = 1.0 / (dist + (seed as f64 * 0.01));
            matrix[(i, j)] = val;
            row_sum += val;
        }
        // Normalize row
        for j in 0..dim {
            matrix[(i, j)] /= row_sum;
        }
    }

    matrix
}

/// Create deterministic probability distribution (NOT random)
fn create_distribution(dim: usize, seed: usize) -> na::DVector<f64> {
    let mut vec = na::DVector::zeros(dim);
    let mut sum = 0.0;

    for i in 0..dim {
        // Deterministic distribution based on index
        let val = 1.0 / ((i + 1 + seed) as f64);
        vec[i] = val;
        sum += val;
    }

    // Normalize
    vec /= sum;
    vec
}

fn bench_kernel_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("markov_kernel_creation");

    for &dim in &[10, 50, 100, 200] {
        let matrix = create_stochastic_matrix(dim, 42);

        group.throughput(Throughput::Elements((dim * dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(MarkovianKernel::new(
                        black_box(matrix.clone()),
                        "benchmark"
                    ))
                });
            },
        );
    }

    group.finish();
}

fn bench_kernel_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("markov_kernel_apply");

    for &dim in &[10, 50, 100, 200] {
        let matrix = create_stochastic_matrix(dim, 42);
        let kernel = MarkovianKernel::new(matrix, "benchmark").unwrap();
        let dist = create_distribution(dim, 42);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(kernel.apply(black_box(&dist)))
                });
            },
        );
    }

    group.finish();
}

fn bench_kernel_apply_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("markov_kernel_apply_n");

    let dim = 50;
    let matrix = create_stochastic_matrix(dim, 42);
    let kernel = MarkovianKernel::new(matrix, "benchmark").unwrap();
    let dist = create_distribution(dim, 42);

    for &n in &[1, 10, 50, 100] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("iterations", n),
            &n,
            |b, _| {
                b.iter(|| {
                    black_box(kernel.apply_n(black_box(&dist), black_box(n)))
                });
            },
        );
    }

    group.finish();
}

fn bench_stationary_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("markov_stationary");

    for &dim in &[10, 50, 100] {
        let matrix = create_stochastic_matrix(dim, 42);
        let mut kernel = MarkovianKernel::new(matrix, "benchmark").unwrap();

        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    kernel.stationary = None;  // Reset cache
                    black_box(kernel.compute_stationary(1000, 1e-10))
                });
            },
        );
    }

    group.finish();
}

fn bench_entropy_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("markov_entropy_rate");

    for &dim in &[10, 50, 100] {
        let matrix = create_stochastic_matrix(dim, 42);
        let mut kernel = MarkovianKernel::new(matrix, "benchmark").unwrap();

        // Pre-compute stationary distribution
        kernel.compute_stationary(1000, 1e-10);

        group.throughput(Throughput::Elements((dim * dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(kernel.entropy_rate())
                });
            },
        );
    }

    group.finish();
}

fn bench_kernel_composition(c: &mut Criterion) {
    let mut group = c.benchmark_group("markov_compose");

    for &dim in &[10, 50, 100] {
        let m1 = create_stochastic_matrix(dim, 42);
        let m2 = create_stochastic_matrix(dim, 99);
        let k1 = MarkovianKernel::new(m1, "k1").unwrap();
        let k2 = MarkovianKernel::new(m2, "k2").unwrap();

        group.throughput(Throughput::Elements((dim * dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(k1.compose(black_box(&k2)))
                });
            },
        );
    }

    group.finish();
}

fn bench_mixing_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("markov_mixing_time");

    for &dim in &[10, 50, 100] {
        let matrix = create_stochastic_matrix(dim, 42);
        let kernel = MarkovianKernel::new(matrix, "benchmark").unwrap();

        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(kernel.mixing_time_estimate())
                });
            },
        );
    }

    group.finish();
}

fn bench_perception_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("perception_kernel");

    for &dim in &[10, 50, 100] {
        // Create likelihood matrix (world states -> experiences)
        let likelihood = create_stochastic_matrix(dim, 42);
        let perception = PerceptionKernel::from_likelihood(likelihood, 0.1).unwrap();
        let world_state = create_distribution(dim, 42);

        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(perception.perceive(black_box(&world_state)))
                });
            },
        );
    }

    group.finish();
}

fn bench_decision_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("decision_kernel");

    for &dim in &[10, 50, 100] {
        // Create value matrix (states x actions)
        let mut values = na::DMatrix::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                // Deterministic value function
                values[(i, j)] = (i as f64 * 0.1 + j as f64 * 0.2).sin() + 1.0;
            }
        }

        let decision = DecisionKernel::from_values(values, 1.0).unwrap();
        let experience = create_distribution(dim, 42);

        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(decision.decide(black_box(&experience)))
                });
            },
        );
    }

    group.finish();
}

fn bench_action_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("action_kernel");

    for &dim in &[10, 50, 100] {
        // Create dynamics matrix (action -> world effect)
        let dynamics = create_stochastic_matrix(dim, 42);
        let action = ActionKernel::from_dynamics(dynamics, 1.0).unwrap();
        let action_choice = create_distribution(dim, 42);

        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(action.act(black_box(&action_choice)))
                });
            },
        );
    }

    group.finish();
}

fn bench_doubly_stochastic_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("markov_doubly_stochastic");

    for &dim in &[10, 50, 100, 200] {
        let kernel = MarkovianKernel::uniform(dim);

        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(kernel.is_doubly_stochastic())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_kernel_creation,
    bench_kernel_apply,
    bench_kernel_apply_n,
    bench_stationary_distribution,
    bench_entropy_rate,
    bench_kernel_composition,
    bench_mixing_time,
    bench_perception_kernel,
    bench_decision_kernel,
    bench_action_kernel,
    bench_doubly_stochastic_check,
);
criterion_main!(benches);
