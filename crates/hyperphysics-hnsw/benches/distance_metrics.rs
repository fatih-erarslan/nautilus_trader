//! Distance Metric Benchmarks
//!
//! Validates SIMD-accelerated distance computations.
//!
//! ## Performance Targets
//! - Euclidean distance: <50ns for dim=128
//! - Cosine similarity: <50ns for dim=128
//! - Inner product: <30ns for dim=128

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use hyperphysics_hnsw::{DistanceMetric, EuclideanMetric, CosineMetric, HyperbolicMetric};

/// Generate deterministic test vectors (NOT random)
fn generate_vectors(dim: usize, seed: usize) -> (Vec<f32>, Vec<f32>) {
    let v1: Vec<f32> = (0..dim)
        .map(|i| {
            let t = (seed as f32 * 0.1) + (i as f32 * 0.01);
            t.sin()
        })
        .collect();

    let v2: Vec<f32> = (0..dim)
        .map(|i| {
            let t = (seed as f32 * 0.15) + (i as f32 * 0.012);
            t.cos()
        })
        .collect();

    (v1, v2)
}

/// Generate vectors within the Poincaré ball (norm < 1)
fn generate_poincare_vectors(dim: usize, seed: usize) -> (Vec<f32>, Vec<f32>) {
    // Scale to ensure vectors are inside the Poincaré ball
    let scale = 0.5 / (dim as f32).sqrt();

    let v1: Vec<f32> = (0..dim)
        .map(|i| {
            let t = (seed as f32 * 0.1) + (i as f32 * 0.01);
            t.sin() * scale
        })
        .collect();

    let v2: Vec<f32> = (0..dim)
        .map(|i| {
            let t = (seed as f32 * 0.15) + (i as f32 * 0.012);
            t.cos() * scale
        })
        .collect();

    (v1, v2)
}

fn bench_euclidean_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_euclidean");
    let metric = EuclideanMetric;

    for &dim in &[64, 128, 256, 512, 1024] {
        let (v1, v2) = generate_vectors(dim, 42);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(metric.distance(black_box(&v1), black_box(&v2)))
                });
            },
        );
    }

    group.finish();
}

fn bench_cosine_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_cosine");
    let metric = CosineMetric;

    for &dim in &[64, 128, 256, 512, 1024] {
        let (v1, v2) = generate_vectors(dim, 42);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(metric.distance(black_box(&v1), black_box(&v2)))
                });
            },
        );
    }

    group.finish();
}

fn bench_hyperbolic_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_hyperbolic");
    let metric = HyperbolicMetric::standard();

    for &dim in &[64, 128, 256, 512] {
        let (v1, v2) = generate_poincare_vectors(dim, 42);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(metric.distance(black_box(&v1), black_box(&v2)))
                });
            },
        );
    }

    group.finish();
}

fn bench_batch_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_batch");
    let metric = EuclideanMetric;

    let dim = 128;
    let batch_sizes = [10, 100, 1000];

    for &batch_size in &batch_sizes {
        // Generate batch of vector pairs
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
            .map(|i| generate_vectors(dim, i))
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("euclidean_batch", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    for (v1, v2) in &pairs {
                        black_box(metric.distance(black_box(v1), black_box(v2)));
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_simd_alignment_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_simd_alignment");
    let metric = EuclideanMetric;

    // Test SIMD alignment effects - aligned vs unaligned dimensions
    for &dim in &[63, 64, 127, 128, 255, 256] {
        let (v1, v2) = generate_vectors(dim, 42);

        group.bench_with_input(
            BenchmarkId::new("euclidean", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(metric.distance(black_box(&v1), black_box(&v2)))
                });
            },
        );
    }

    group.finish();
}

fn bench_metric_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_metric_comparison");

    let dim = 128;
    let (v1, v2) = generate_vectors(dim, 42);
    let (h1, h2) = generate_poincare_vectors(dim, 42);

    let euclidean = EuclideanMetric;
    let cosine = CosineMetric;
    let hyperbolic = HyperbolicMetric::standard();

    group.bench_function("euclidean_128d", |b| {
        b.iter(|| black_box(euclidean.distance(black_box(&v1), black_box(&v2))))
    });

    group.bench_function("cosine_128d", |b| {
        b.iter(|| black_box(cosine.distance(black_box(&v1), black_box(&v2))))
    });

    group.bench_function("hyperbolic_128d", |b| {
        b.iter(|| black_box(hyperbolic.distance(black_box(&h1), black_box(&h2))))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_euclidean_distance,
    bench_cosine_distance,
    bench_hyperbolic_distance,
    bench_batch_distances,
    bench_simd_alignment_effects,
    bench_metric_comparison,
);
criterion_main!(benches);
