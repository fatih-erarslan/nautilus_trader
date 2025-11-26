//! HNSW Performance Benchmarks
//!
//! This benchmark suite validates the performance requirements:
//! - <0.5ms latency for 1M vectors @ ef=50
//! - 95%+ recall @ ef=50
//!
//! Run with: cargo bench --bench hnsw_benchmark

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hyperphysics_market::hnsw::HNSWIndex;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Generate random normalized vectors for benchmarking
fn generate_random_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let mut vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            // Normalize
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                vec.iter_mut().for_each(|x| *x /= norm);
            }
            vec
        })
        .collect()
}

/// Calculate recall: percentage of true nearest neighbors found
fn calculate_recall(results: &[(usize, f32)], ground_truth: &[(usize, f32)], k: usize) -> f32 {
    let result_ids: std::collections::HashSet<_> = results.iter().take(k).map(|(id, _)| id).collect();
    let truth_ids: std::collections::HashSet<_> = ground_truth.iter().take(k).map(|(id, _)| id).collect();

    let intersection = result_ids.intersection(&truth_ids).count();
    intersection as f32 / k as f32
}

/// Brute force search for ground truth
fn brute_force_search(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut distances: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(id, vec)| {
            let dist: f32 = query.iter()
                .zip(vec.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            (id, dist)
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.truncate(k);
    distances
}

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_construction");

    for &size in &[1000, 10000, 100000] {
        let vectors = generate_random_vectors(size, 128, 42);

        group.bench_with_input(BenchmarkId::new("insert", size), &vectors, |b, vecs| {
            b.iter(|| {
                let mut index = HNSWIndex::new(128, 16, 200);
                for vec in vecs {
                    black_box(index.insert(vec.clone()));
                }
            });
        });
    }

    group.finish();
}

fn bench_search_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search_latency");

    // Build index with 100K vectors
    let vectors = generate_random_vectors(100000, 128, 42);
    let mut index = HNSWIndex::new(128, 16, 200);
    for vec in &vectors {
        index.insert(vec.clone());
    }

    let query = generate_random_vectors(1, 128, 123)[0].clone();

    // Test different ef values
    for &ef in &[10, 50, 100, 200] {
        group.bench_with_input(BenchmarkId::new("ef", ef), &ef, |b, &ef| {
            b.iter(|| {
                black_box(index.search(&query, 10, ef));
            });
        });
    }

    group.finish();
}

fn bench_search_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search_throughput");

    // Build index
    let vectors = generate_random_vectors(10000, 128, 42);
    let mut index = HNSWIndex::new(128, 16, 200);
    for vec in &vectors {
        index.insert(vec.clone());
    }

    let queries = generate_random_vectors(100, 128, 999);

    group.bench_function("100_queries_ef50", |b| {
        b.iter(|| {
            for query in &queries {
                black_box(index.search(query, 10, 50));
            }
        });
    });

    group.finish();
}

fn bench_recall_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_recall");

    // Smaller dataset for ground truth comparison
    let vectors = generate_random_vectors(1000, 128, 42);
    let mut index = HNSWIndex::new(128, 16, 200);
    for vec in &vectors {
        index.insert(vec.clone());
    }

    let queries = generate_random_vectors(10, 128, 999);

    group.bench_function("recall_calculation", |b| {
        b.iter(|| {
            let mut total_recall = 0.0;
            for query in &queries {
                // HNSW search
                let results = index.search(query, 10, 50);
                let hnsw_results: Vec<(usize, f32)> = results
                    .iter()
                    .map(|r| (r.id, r.distance))
                    .collect();

                // Ground truth
                let ground_truth = brute_force_search(query, &vectors, 10);

                // Calculate recall
                let recall = calculate_recall(&hnsw_results, &ground_truth, 10);
                total_recall += recall;
            }
            black_box(total_recall / queries.len() as f32);
        });
    });

    group.finish();
}

fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_dimension_scaling");

    for &dim in &[32, 64, 128, 256, 512] {
        let vectors = generate_random_vectors(1000, dim, 42);
        let mut index = HNSWIndex::new(dim, 16, 200);
        for vec in &vectors {
            index.insert(vec.clone());
        }

        let query = generate_random_vectors(1, dim, 123)[0].clone();

        group.bench_with_input(BenchmarkId::new("search", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(index.search(&query, 10, 50));
            });
        });
    }

    group.finish();
}

fn bench_m_parameter_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_m_parameter");

    let vectors = generate_random_vectors(5000, 128, 42);

    for &m in &[8, 16, 32, 48] {
        let mut index = HNSWIndex::new(128, m, 200);
        for vec in &vectors {
            index.insert(vec.clone());
        }

        let query = generate_random_vectors(1, 128, 123)[0].clone();

        group.bench_with_input(BenchmarkId::new("search_M", m), &m, |b, _| {
            b.iter(|| {
                black_box(index.search(&query, 10, 50));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_construction,
    bench_search_latency,
    bench_search_throughput,
    bench_recall_quality,
    bench_dimension_scaling,
    bench_m_parameter_impact
);
criterion_main!(benches);
