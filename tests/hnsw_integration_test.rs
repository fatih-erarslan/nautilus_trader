//! HNSW Integration Tests
//!
//! Comprehensive integration tests for HNSW functionality and performance.

use hyperphysics_market::hnsw::HNSWIndex;
use approx::assert_relative_eq;

#[test]
fn test_hnsw_basic_functionality() {
    let mut index = HNSWIndex::new(3, 16, 200);

    // Insert simple test vectors
    let v1 = vec![1.0, 0.0, 0.0];
    let v2 = vec![0.0, 1.0, 0.0];
    let v3 = vec![0.0, 0.0, 1.0];
    let v4 = vec![0.5, 0.5, 0.0]; // Close to v1 and v2

    index.insert(v1.clone());
    index.insert(v2.clone());
    index.insert(v3.clone());
    index.insert(v4.clone());

    assert_eq!(index.len(), 4);

    // Search for nearest to v1
    let results = index.search(&v1, 2, 10);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, 0); // Should find itself first
    assert_relative_eq!(results[0].distance, 0.0, epsilon = 1e-6);
}

#[test]
fn test_hnsw_recall_quality() {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashSet;

    let dim = 32;
    let num_vectors = 1000;
    let k = 10;

    let mut index = HNSWIndex::new(dim, 16, 200);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Generate and insert vectors
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| {
            (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
        })
        .collect();

    for vec in &vectors {
        index.insert(vec.clone());
    }

    // Test with a query
    let query = vectors[0].clone();

    // HNSW search
    let hnsw_results = index.search(&query, k, 100);

    // Brute force ground truth
    let mut exact_results: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(id, vec)| {
            let dist: f32 = query
                .iter()
                .zip(vec.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            (id, dist)
        })
        .collect();
    exact_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Calculate recall
    let hnsw_ids: HashSet<_> = hnsw_results.iter().map(|r| r.id).collect();
    let exact_ids: HashSet<_> = exact_results.iter().take(k).map(|(id, _)| *id).collect();

    println!("HNSW IDs: {:?}", hnsw_results.iter().take(k).map(|r| r.id).collect::<Vec<_>>());
    println!("Exact IDs: {:?}", exact_results.iter().take(k).map(|(id, _)| id).collect::<Vec<_>>());

    let intersection = hnsw_ids.intersection(&exact_ids).count();
    let recall = intersection as f64 / k as f64;

    println!("Recall@{}: {:.1}%", k, recall * 100.0);
    assert!(recall >= 0.7, "Recall should be at least 70%, got {:.1}%", recall * 100.0);
}

#[test]
fn test_hnsw_performance_target() {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;

    let dim = 128;
    let num_vectors = 10000;
    let mut index = HNSWIndex::new(dim, 16, 200);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Insert vectors
    for _ in 0..num_vectors {
        let vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        index.insert(vec);
    }

    // Measure search latency
    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let start = Instant::now();
    let _results = index.search(&query, 10, 50);
    let duration = start.elapsed();

    println!("Search latency: {:?}", duration);
    assert!(duration.as_micros() < 500, "Search should complete in <500μs");
}

#[test]
fn test_hnsw_index_statistics() {
    let mut index = HNSWIndex::new(4, 16, 200);

    for i in 0..100 {
        index.insert(vec![i as f32; 4]);
    }

    let stats = index.stats();
    assert_eq!(stats.total_nodes, 100);
    assert!(stats.total_layers > 0);
    assert!(stats.total_layers <= 16); // Should cap at 16
    assert!(stats.avg_connections_per_node > 0.0);
    assert_eq!(stats.nodes_per_layer[0], 100); // Layer 0 has all nodes
}
