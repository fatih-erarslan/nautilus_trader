//! LSH query latency benchmark.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hyperphysics_cortical_bus::lsh::{LshTables, LshConfig};

fn bench_lsh_store(c: &mut Criterion) {
    let config = LshConfig { num_tables: 8, num_hashes: 16, dim: 128 };
    let lsh = LshTables::new(config);

    c.bench_function("lsh_store", |b| {
        let mut id = 0u32;
        b.iter(|| {
            let pattern: Vec<f32> = (0..128).map(|i| (i as f32 + id as f32) * 0.01).collect();
            lsh.store(black_box(id), black_box(&pattern)).unwrap();
            id = id.wrapping_add(1);
        })
    });
}

fn bench_lsh_query(c: &mut Criterion) {
    let config = LshConfig { num_tables: 8, num_hashes: 16, dim: 128 };
    let lsh = LshTables::new(config);

    // Pre-populate with patterns
    for i in 0..10000 {
        let pattern: Vec<f32> = (0..128).map(|j| ((i * 128 + j) as f32).sin()).collect();
        lsh.store(i, &pattern).unwrap();
    }

    let mut group = c.benchmark_group("lsh_query");

    for k in [1, 5, 10, 20].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(k), k, |b, &k| {
            let query: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
            let mut results = vec![(0u32, 0.0f32); k];
            b.iter(|| {
                let count = lsh.query(black_box(&query), black_box(k), black_box(&mut results)).unwrap();
                black_box(count);
            })
        });
    }

    group.finish();
}

fn bench_lsh_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_scaling");

    for num_patterns in [1000, 5000, 10000, 50000].iter() {
        let config = LshConfig { num_tables: 8, num_hashes: 16, dim: 128 };
        let lsh = LshTables::new(config);

        // Populate
        for i in 0..*num_patterns {
            let pattern: Vec<f32> = (0..128).map(|j| ((i * 128 + j) as f32).sin()).collect();
            lsh.store(i as u32, &pattern).unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(num_patterns), num_patterns, |b, _| {
            let query: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
            let mut results = vec![(0u32, 0.0f32); 10];
            b.iter(|| {
                lsh.query(black_box(&query), 10, black_box(&mut results)).unwrap();
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_lsh_store, bench_lsh_query, bench_lsh_scaling);
criterion_main!(benches);
