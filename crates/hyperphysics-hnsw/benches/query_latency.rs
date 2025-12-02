//! HNSW Query Latency Benchmarks
//!
//! Validates performance against QUERY_LATENCY_BUDGET_NS = 1_000ns (1µs) target.
//!
//! ## Performance Targets (from hyperphysics-hnsw/src/lib.rs)
//! - Query latency: <1µs per query
//! - Distance computation: <100ns per vector pair
//! - Batch query: linear scaling

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use hyperphysics_hnsw::{HotIndex, IndexConfig, EuclideanMetric, CosineMetric};

/// Create benchmark index with specified parameters using EuclideanMetric
fn create_benchmark_index(dim: usize, num_vectors: usize) -> HotIndex<EuclideanMetric> {
    let config = IndexConfig::default()
        .with_dimensions(dim)
        .with_m(16)
        .with_ef_construction(100)
        .with_ef_search(50)
        .with_capacity(num_vectors);

    let mut index = HotIndex::new(config, EuclideanMetric).expect("Failed to create index");

    // Insert vectors using deterministic pattern (NOT random - real geometric data)
    for i in 0..num_vectors {
        let vector: Vec<f32> = (0..dim)
            .map(|d| {
                // Deterministic embedding: spherical harmonics pattern
                let theta = (i as f32 * 0.1) + (d as f32 * 0.01);
                let phi = (i as f32 * 0.05) + (d as f32 * 0.02);
                theta.sin() * phi.cos()
            })
            .collect();
        index.insert(&vector).expect("Insert failed");
    }

    index
}

/// Generate deterministic query vector
fn generate_query_vector(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|d| {
            let theta = (seed as f32 * 0.15) + (d as f32 * 0.01);
            let phi = (seed as f32 * 0.07) + (d as f32 * 0.02);
            theta.cos() * phi.sin()
        })
        .collect()
}

fn bench_single_query_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_single_query");

    // Test across different index sizes
    for &num_vectors in &[1000, 10_000, 50_000] {
        let index = create_benchmark_index(128, num_vectors);
        let query = generate_query_vector(128, 999);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("query", num_vectors),
            &num_vectors,
            |b, _| {
                b.iter(|| {
                    black_box(index.search(black_box(&query), 10))
                });
            },
        );
    }

    group.finish();
}

fn bench_query_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_dimension_scaling");

    // Test across different dimensions
    for &dim in &[64, 128, 256, 512] {
        let index = create_benchmark_index(dim, 10_000);
        let query = generate_query_vector(dim, 999);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(index.search(black_box(&query), 10))
                });
            },
        );
    }

    group.finish();
}

fn bench_k_neighbors_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_k_scaling");

    let index = create_benchmark_index(128, 50_000);
    let query = generate_query_vector(128, 999);

    // Test across different k values
    for &k in &[1, 10, 50, 100] {
        group.throughput(Throughput::Elements(k as u64));
        group.bench_with_input(
            BenchmarkId::new("k", k),
            &k,
            |b, &k| {
                b.iter(|| {
                    black_box(index.search(black_box(&query), k))
                });
            },
        );
    }

    group.finish();
}

fn bench_batch_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_batch_query");

    let index = create_benchmark_index(128, 50_000);

    // Test batch sizes
    for &batch_size in &[10, 100, 1000] {
        let queries: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| generate_query_vector(128, i))
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    for query in &queries {
                        black_box(index.search(black_box(query), 10));
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_ef_search_tuning(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_ef_search");

    // Test with different ef_search values
    for &ef_search in &[10, 50, 100, 200] {
        let config = IndexConfig::default()
            .with_dimensions(128)
            .with_m(16)
            .with_ef_construction(100)
            .with_ef_search(ef_search)
            .with_capacity(10_000);

        let mut index = HotIndex::new(config, EuclideanMetric).expect("Failed to create index");

        // Insert 10k vectors
        for i in 0..10_000 {
            let vector: Vec<f32> = (0..128)
                .map(|d| {
                    let theta = (i as f32 * 0.1) + (d as f32 * 0.01);
                    theta.sin()
                })
                .collect();
            index.insert(&vector).expect("Insert failed");
        }

        let query = generate_query_vector(128, 999);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("ef", ef_search),
            &ef_search,
            |b, _| {
                b.iter(|| {
                    black_box(index.search(black_box(&query), 10))
                });
            },
        );
    }

    group.finish();
}

fn bench_cosine_metric(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_cosine_metric");

    let config = IndexConfig::default()
        .with_dimensions(128)
        .with_m(16)
        .with_ef_construction(100)
        .with_ef_search(50)
        .with_capacity(10_000);

    let mut index = HotIndex::new(config, CosineMetric).expect("Failed to create index");

    // Insert 10k vectors
    for i in 0..10_000 {
        let vector: Vec<f32> = (0..128)
            .map(|d| {
                let theta = (i as f32 * 0.1) + (d as f32 * 0.01);
                theta.sin()
            })
            .collect();
        index.insert(&vector).expect("Insert failed");
    }

    let query = generate_query_vector(128, 999);

    group.bench_function("cosine_search_10k", |b| {
        b.iter(|| {
            black_box(index.search(black_box(&query), 10))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_query_latency,
    bench_query_dimension_scaling,
    bench_k_neighbors_scaling,
    bench_batch_query,
    bench_ef_search_tuning,
    bench_cosine_metric,
);
criterion_main!(benches);
