//! LSH Hash Computation Benchmarks
//!
//! Validates performance against HASH_LATENCY_BUDGET_NS = 100ns target.
//!
//! ## Performance Targets (from hyperphysics-lsh/src/lib.rs)
//! - SimHash computation: <100ns per vector
//! - MinHash computation: <200ns per set
//! - SRP Hash computation: <100ns per vector

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use hyperphysics_lsh::{SimHash, MinHash, SrpHash, HashFamily};

/// Generate deterministic test vector (NOT random)
fn generate_vector(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let t = (seed as f32 * 0.1) + (i as f32 * 0.01);
            t.sin()
        })
        .collect()
}

/// Generate deterministic set elements (NOT random)
fn generate_set(size: usize, seed: usize) -> Vec<u64> {
    (0..size)
        .map(|i| {
            // Deterministic hash-like values
            let base = (seed as u64) * 1000000 + (i as u64);
            base ^ (base >> 7) ^ (base << 3)
        })
        .collect()
}

fn bench_simhash_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_simhash");

    // Test across different dimensions
    for &dim in &[64, 128, 256, 512] {
        let hasher = SimHash::new(dim, 128, 42);
        let vector = generate_vector(dim, 42);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(hasher.hash(black_box(&vector)))
                });
            },
        );
    }

    group.finish();
}

fn bench_simhash_bit_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_simhash_bits");

    let dim = 128;
    let vector = generate_vector(dim, 42);

    // Test across different signature sizes
    for &num_bits in &[64, 128, 192, 256] {
        let hasher = SimHash::new(dim, num_bits, 42);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("bits", num_bits),
            &num_bits,
            |b, _| {
                b.iter(|| {
                    black_box(hasher.hash(black_box(&vector)))
                });
            },
        );
    }

    group.finish();
}

fn bench_minhash_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_minhash");

    // Test across different set sizes
    for &set_size in &[10, 50, 100, 500] {
        let hasher = MinHash::new(128, 42);
        let set = generate_set(set_size, 42);

        group.throughput(Throughput::Elements(set_size as u64));
        group.bench_with_input(
            BenchmarkId::new("set_size", set_size),
            &set_size,
            |b, _| {
                b.iter(|| {
                    black_box(hasher.hash(black_box(&set)))
                });
            },
        );
    }

    group.finish();
}

fn bench_minhash_num_hashes(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_minhash_hashes");

    let set = generate_set(100, 42);

    // Test across different numbers of hash functions
    for &num_hashes in &[64, 128, 192, 256] {
        let hasher = MinHash::new(num_hashes, 42);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("hashes", num_hashes),
            &num_hashes,
            |b, _| {
                b.iter(|| {
                    black_box(hasher.hash(black_box(&set)))
                });
            },
        );
    }

    group.finish();
}

fn bench_srp_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_srp");

    // Test SRP hash across dimensions
    for &dim in &[64, 128, 256, 512] {
        let hasher = SrpHash::new(dim, 128, 42);
        let vector = generate_vector(dim, 42);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(hasher.hash(black_box(&vector)))
                });
            },
        );
    }

    group.finish();
}

fn bench_hamming_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_hamming");

    let hasher = SimHash::new(128, 256, 42);
    let v1 = generate_vector(128, 42);
    let v2 = generate_vector(128, 99);

    let sig1 = hasher.hash(&v1);
    let sig2 = hasher.hash(&v2);

    group.bench_function("hamming_256bit", |b| {
        b.iter(|| {
            black_box(sig1.hamming_distance(black_box(&sig2)))
        });
    });

    group.finish();
}

fn bench_batch_simhash(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_simhash_batch");

    let hasher = SimHash::new(128, 128, 42);

    for &batch_size in &[10, 100, 1000] {
        let vectors: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| generate_vector(128, i))
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    for v in &vectors {
                        black_box(hasher.hash(black_box(v)));
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_collision_probability(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_collision_prob");

    let hasher = SimHash::new(128, 128, 42);

    // Generate pairs of vectors with varying similarity
    let v1 = generate_vector(128, 42);
    let sig1 = hasher.hash(&v1);

    for &offset in &[0, 10, 50, 100] {
        let v2 = generate_vector(128, 42 + offset);
        let sig2 = hasher.hash(&v2);

        group.bench_with_input(
            BenchmarkId::new("offset", offset),
            &offset,
            |b, _| {
                b.iter(|| {
                    black_box(hasher.collision_probability(black_box(&sig1), black_box(&sig2)))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_simhash_computation,
    bench_simhash_bit_scaling,
    bench_minhash_computation,
    bench_minhash_num_hashes,
    bench_srp_hash,
    bench_hamming_distance,
    bench_batch_simhash,
    bench_collision_probability,
);
criterion_main!(benches);
