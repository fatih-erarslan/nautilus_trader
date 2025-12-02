//! LSH Streaming Insert Benchmarks
//!
//! Validates performance against INSERT_LATENCY_BUDGET_NS = 500ns target.
//!
//! ## Performance Targets (from hyperphysics-lsh/src/lib.rs)
//! - Regular insert: <500ns
//! - Stream insert: <200ns (lock-free)
//! - Query: <5µs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use hyperphysics_lsh::{StreamingLshIndex, LshConfig};

/// Generate deterministic test vector (NOT random)
fn generate_vector(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let t = (seed as f32 * 0.1) + (i as f32 * 0.01);
            t.sin()
        })
        .collect()
}

fn create_test_index(dim: usize, num_tables: usize) -> StreamingLshIndex {
    let config = LshConfig::simhash(dim, 64)
        .with_tables(num_tables)
        .with_seed(42);

    StreamingLshIndex::new(config).expect("Failed to create index")
}

fn bench_single_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_single_insert");

    for &dim in &[64, 128, 256] {
        let index = create_test_index(dim, 4);
        let vector = generate_vector(dim, 42);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(index.insert(black_box(vector.clone())))
                });
            },
        );
    }

    group.finish();
}

fn bench_stream_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_stream_insert");

    let dim = 64;

    for &num_tables in &[2, 4, 8, 16] {
        let config = LshConfig::simhash(dim, 64)
            .with_tables(num_tables)
            .with_seed(42);

        let index = StreamingLshIndex::new(config).expect("Failed to create index");
        let vector = generate_vector(dim, 42);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("tables", num_tables),
            &num_tables,
            |b, _| {
                b.iter(|| {
                    black_box(index.stream_insert(black_box(vector.clone())))
                });
            },
        );
    }

    group.finish();
}

fn bench_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_batch_insert");

    let dim = 64;
    let index = create_test_index(dim, 4);

    for &batch_size in &[10, 100, 1000] {
        let vectors: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| generate_vector(dim, i))
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    for v in &vectors {
                        black_box(index.insert(black_box(v.clone())));
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_query_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_query");

    let dim = 64;

    // Test query performance at different index sizes
    for &index_size in &[1000, 10_000, 50_000] {
        let index = create_test_index(dim, 4);

        // Pre-populate index
        for i in 0..index_size {
            let v = generate_vector(dim, i);
            index.insert(v).expect("Insert failed");
        }

        let query = generate_vector(dim, 999999);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("size", index_size),
            &index_size,
            |b, _| {
                b.iter(|| {
                    black_box(index.query(black_box(&query), 10))
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_insert_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_concurrent");

    let dim = 64;
    let index = create_test_index(dim, 4);

    // Pre-populate
    for i in 0..10_000 {
        let v = generate_vector(dim, i);
        index.insert(v).expect("Insert failed");
    }

    let insert_vector = generate_vector(dim, 99999);
    let query_vector = generate_vector(dim, 88888);

    // Benchmark interleaved insert and query
    group.bench_function("interleaved", |b| {
        b.iter(|| {
            // Insert then query pattern (simulates real workload)
            black_box(index.insert(black_box(insert_vector.clone())));
            black_box(index.query(black_box(&query_vector), 10));
        });
    });

    group.finish();
}

fn bench_process_stream_buffer(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_buffer_processing");

    let dim = 64;
    let config = LshConfig::simhash(dim, 64)
        .with_tables(4)
        .with_seed(42);

    let index = StreamingLshIndex::new(config).expect("Failed to create index");

    // Fill stream buffer with vectors
    for i in 0..1000 {
        let v = generate_vector(dim, i);
        let _ = index.stream_insert(v);
    }

    group.bench_function("process_1000", |b| {
        b.iter(|| {
            black_box(index.process_stream_buffer())
        });
    });

    group.finish();
}

fn bench_table_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_table_scaling");

    let dim = 64;

    for &num_tables in &[1, 2, 4, 8, 16] {
        let index = create_test_index(dim, num_tables);

        // Pre-populate
        for i in 0..5_000 {
            let v = generate_vector(dim, i);
            index.insert(v).expect("Insert failed");
        }

        let query = generate_vector(dim, 99999);

        group.bench_with_input(
            BenchmarkId::new("tables_query", num_tables),
            &num_tables,
            |b, _| {
                b.iter(|| {
                    black_box(index.query(black_box(&query), 10))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_insert,
    bench_stream_insert,
    bench_batch_insert,
    bench_query_performance,
    bench_concurrent_insert_query,
    bench_process_stream_buffer,
    bench_table_scaling,
);
criterion_main!(benches);
