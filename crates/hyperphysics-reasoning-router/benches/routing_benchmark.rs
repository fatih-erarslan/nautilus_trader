//! Benchmarks for the reasoning router.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hyperphysics_reasoning_router::prelude::*;
use hyperphysics_reasoning_router::{LatencyTier, ProblemDomain};

fn bench_signature_to_feature_vector(c: &mut Criterion) {
    let sig = ProblemSignature::new(ProblemType::Optimization, ProblemDomain::Financial)
        .with_dimensionality(1000)
        .with_sparsity(0.8)
        .with_latency_budget(LatencyTier::Fast);

    c.bench_function("signature_to_feature_vector", |b| {
        b.iter(|| {
            black_box(sig.to_feature_vector())
        })
    });
}

fn bench_signature_similarity(c: &mut Criterion) {
    let sig1 = ProblemSignature::new(ProblemType::Optimization, ProblemDomain::Financial)
        .with_dimensionality(1000);
    let sig2 = ProblemSignature::new(ProblemType::Optimization, ProblemDomain::Financial)
        .with_dimensionality(2000);

    c.bench_function("signature_similarity", |b| {
        b.iter(|| {
            black_box(sig1.similarity(&sig2))
        })
    });
}

fn bench_lsh_hash_computation(c: &mut Criterion) {
    let config = LSHConfig::default();
    let index: LSHIndex<String> = LSHIndex::new(config);
    let features = vec![0.5f32; 16];

    c.bench_function("lsh_hash_computation", |b| {
        b.iter(|| {
            black_box(index.compute_hash(&features))
        })
    });
}

fn bench_lsh_query(c: &mut Criterion) {
    let config = LSHConfig::default();
    let mut index: LSHIndex<String> = LSHIndex::new(config);

    // Insert 1000 entries
    for i in 0..1000 {
        let features: Vec<f32> = (0..16).map(|j| ((i * j) as f32 / 1000.0) % 1.0).collect();
        index.insert(features, format!("entry-{}", i));
    }

    let query_features = vec![0.5f32; 16];

    c.bench_function("lsh_query_1000_entries", |b| {
        b.iter(|| {
            black_box(index.query(&query_features, 10))
        })
    });
}

fn bench_thompson_sampling(c: &mut Criterion) {
    let mut sampler = ThompsonSampler::new();

    // Train with some data
    for i in 0..10 {
        let id = BackendId::new(format!("backend-{}", i));
        for _ in 0..100 {
            sampler.update(&id, true, 0.8);
        }
    }

    let candidates: Vec<BackendId> = (0..10).map(|i| BackendId::new(format!("backend-{}", i))).collect();
    let candidate_refs: Vec<&BackendId> = candidates.iter().collect();

    c.bench_function("thompson_sampling_select_10", |b| {
        b.iter(|| {
            black_box(sampler.select(&candidate_refs))
        })
    });
}

fn bench_lsh_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_insert");

    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let config = LSHConfig::default();
            let mut index: LSHIndex<String> = LSHIndex::new(config);

            // Pre-generate data
            let data: Vec<(Vec<f32>, String)> = (0..size)
                .map(|i| {
                    let features: Vec<f32> = (0..16).map(|j| ((i * j) as f32 / size as f32) % 1.0).collect();
                    (features, format!("entry-{}", i))
                })
                .collect();

            b.iter(|| {
                for (features, name) in &data {
                    index.insert(features.clone(), name.clone());
                }
                index.clear();
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_signature_to_feature_vector,
    bench_signature_similarity,
    bench_lsh_hash_computation,
    bench_lsh_query,
    bench_thompson_sampling,
    bench_lsh_insert,
);

criterion_main!(benches);
