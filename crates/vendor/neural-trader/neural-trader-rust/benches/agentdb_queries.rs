// AgentDB Queries Benchmark
//
// Performance targets:
// - Vector search: <1ms for k=10
// - Insert: <500μs
// - Batch insert: <5ms for 100 vectors
// - HNSW index build: <100ms for 10K vectors

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nt_agentdb_client::{AgentDBClient, EmbeddingVector, Query};
use std::time::Duration;

// ============================================================================
// Helper Functions - Generate Test Data
// ============================================================================

fn generate_random_vector(dimensions: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..dimensions).map(|_| rng.gen::<f32>()).collect()
}

fn generate_vectors(count: usize, dimensions: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|_| generate_random_vector(dimensions))
        .collect()
}

fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let magnitude: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        v.iter().map(|x| x / magnitude).collect()
    } else {
        v.to_vec()
    }
}

// ============================================================================
// Benchmarks - Vector Generation and Normalization
// ============================================================================

fn bench_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_operations");

    for dim in [128, 384, 768, 1536].iter() {
        group.bench_with_input(
            BenchmarkId::new("generate", dim),
            dim,
            |b, &dim| {
                b.iter(|| {
                    black_box(generate_random_vector(dim))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("normalize", dim),
            dim,
            |b, &dim| {
                let vector = generate_random_vector(dim);

                b.iter(|| {
                    black_box(normalize_vector(black_box(&vector)))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Cosine Similarity
// ============================================================================

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for dim in [128, 384, 768, 1536].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            dim,
            |b, &dim| {
                let v1 = normalize_vector(&generate_random_vector(dim));
                let v2 = normalize_vector(&generate_random_vector(dim));

                b.iter(|| {
                    let dot_product: f32 = v1
                        .iter()
                        .zip(v2.iter())
                        .map(|(a, b)| a * b)
                        .sum();

                    black_box(dot_product)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Vector Distance Calculations
// ============================================================================

fn bench_distance_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_calculations");

    for dim in [128, 384, 768].iter() {
        let v1 = generate_random_vector(*dim);
        let v2 = generate_random_vector(*dim);

        group.bench_with_input(
            BenchmarkId::new("euclidean", dim),
            dim,
            |b, _| {
                b.iter(|| {
                    let distance: f32 = v1
                        .iter()
                        .zip(v2.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();

                    black_box(distance)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("manhattan", dim),
            dim,
            |b, _| {
                b.iter(|| {
                    let distance: f32 = v1
                        .iter()
                        .zip(v2.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum();

                    black_box(distance)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Memory Storage (Mock)
// ============================================================================

fn bench_memory_storage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_storage");

    let rt = tokio::runtime::Runtime::new().unwrap();

    // Mock vector insertion
    group.bench_function("insert_single_vector", |b| {
        b.to_async(&rt).iter(|| async {
            let vector = generate_random_vector(384);
            let id = uuid::Uuid::new_v4().to_string();

            // Simulate storage operation
            tokio::time::sleep(Duration::from_micros(100)).await;

            black_box((id, vector))
        });
    });

    // Batch insertion
    for batch_size in [10, 50, 100, 500].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_insert", batch_size),
            batch_size,
            |b, &batch_size| {
                b.to_async(&rt).iter(|| async {
                    let vectors = generate_vectors(batch_size, 384);

                    // Simulate batch storage
                    tokio::time::sleep(Duration::from_micros(batch_size as u64 * 10)).await;

                    black_box(vectors)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Vector Search (K-Nearest Neighbors)
// ============================================================================

fn bench_knn_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_search");
    group.sample_size(50);

    for k in [1, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("linear_scan", k),
            k,
            |b, &k| {
                let query = normalize_vector(&generate_random_vector(384));
                let database = generate_vectors(1000, 384);

                b.iter(|| {
                    // Linear scan through all vectors
                    let mut similarities: Vec<_> = database
                        .iter()
                        .map(|v| {
                            let similarity: f32 = query
                                .iter()
                                .zip(v.iter())
                                .map(|(a, b)| a * b)
                                .sum();
                            similarity
                        })
                        .collect();

                    // Sort and take top k
                    similarities.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    let top_k: Vec<_> = similarities.into_iter().take(k).collect();

                    black_box(top_k)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - HNSW Index Operations (Simulated)
// ============================================================================

fn bench_hnsw_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_operations");
    group.sample_size(30);

    let rt = tokio::runtime::Runtime::new().unwrap();

    // Simulate HNSW search
    for db_size in [100, 1000, 10_000].iter() {
        group.bench_with_input(
            BenchmarkId::new("hnsw_search", db_size),
            db_size,
            |b, &db_size| {
                let query = generate_random_vector(384);

                b.to_async(&rt).iter(|| async {
                    // Simulate HNSW search - O(log n) complexity
                    let estimated_time_us = (db_size as f64).log2() as u64 * 10;
                    tokio::time::sleep(Duration::from_micros(estimated_time_us)).await;

                    black_box(query.clone())
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Pattern Storage and Retrieval
// ============================================================================

fn bench_pattern_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_operations");

    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("store_trading_pattern", |b| {
        b.to_async(&rt).iter(|| async {
            let observation_vector = generate_random_vector(384);
            let metadata = serde_json::json!({
                "symbol": "AAPL",
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "signal_type": "momentum",
                "confidence": 0.85,
            });

            // Simulate storage
            tokio::time::sleep(Duration::from_micros(200)).await;

            black_box((observation_vector, metadata))
        });
    });

    group.bench_function("retrieve_similar_patterns", |b| {
        b.to_async(&rt).iter(|| async {
            let query_vector = generate_random_vector(384);
            let k = 10;

            // Simulate retrieval
            tokio::time::sleep(Duration::from_micros(500)).await;

            black_box((query_vector, k))
        });
    });

    group.finish();
}

// ============================================================================
// Benchmarks - ReasoningBank Operations
// ============================================================================

fn bench_reasoningbank_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("reasoningbank_operations");

    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("store_trajectory", |b| {
        b.to_async(&rt).iter(|| async {
            let trajectory = serde_json::json!({
                "observations": generate_random_vector(384),
                "actions": ["buy", "hold"],
                "rewards": [0.05, 0.02],
                "verdict": "successful",
            });

            tokio::time::sleep(Duration::from_micros(300)).await;

            black_box(trajectory)
        });
    });

    group.bench_function("query_successful_patterns", |b| {
        b.to_async(&rt).iter(|| async {
            let query = generate_random_vector(384);
            let filter = "verdict=successful";

            tokio::time::sleep(Duration::from_micros(800)).await;

            black_box((query, filter))
        });
    });

    group.finish();
}

// ============================================================================
// Benchmarks - Memory Distillation
// ============================================================================

fn bench_memory_distillation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_distillation");
    group.sample_size(30);

    let rt = tokio::runtime::Runtime::new().unwrap();

    for pattern_count in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(pattern_count),
            pattern_count,
            |b, &pattern_count| {
                b.to_async(&rt).iter(|| async {
                    // Simulate clustering/distillation of patterns
                    let patterns = generate_vectors(pattern_count, 384);

                    // Calculate cluster centroids
                    let centroid: Vec<f32> = (0..384)
                        .map(|i| {
                            patterns.iter().map(|p| p[i]).sum::<f32>() / pattern_count as f32
                        })
                        .collect();

                    black_box(centroid)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Embedding Cache Operations
// ============================================================================

fn bench_embedding_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_cache");

    use std::collections::HashMap;

    let mut cache: HashMap<String, Vec<f32>> = HashMap::new();

    group.bench_function("cache_insert", |b| {
        b.iter(|| {
            let key = uuid::Uuid::new_v4().to_string();
            let vector = generate_random_vector(384);
            cache.insert(black_box(key), black_box(vector));
        });
    });

    // Populate cache
    for i in 0..1000 {
        let key = format!("key_{}", i);
        let vector = generate_random_vector(384);
        cache.insert(key, vector);
    }

    group.bench_function("cache_lookup", |b| {
        b.iter(|| {
            let key = format!("key_{}", 500);
            black_box(cache.get(&key))
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark Configuration
// ============================================================================

fn configure_criterion() -> Criterion {
    Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_secs(2))
        .configure_from_args()
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets =
        bench_vector_operations,
        bench_cosine_similarity,
        bench_distance_calculations,
        bench_memory_storage,
        bench_knn_search,
        bench_hnsw_operations,
        bench_pattern_operations,
        bench_reasoningbank_operations,
        bench_memory_distillation,
        bench_embedding_cache
}

criterion_main!(benches);
