//! HNSW Index Demonstration
//!
//! This example demonstrates the usage of the HNSW index for fast
//! approximate nearest neighbor search in HyperPhysics.
//!
//! Run with: cargo run --example hnsw_demo --release

use hyperphysics_market::hnsw::HNSWIndex;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

fn main() {
    println!("=== HyperPhysics HNSW Index Demo ===\n");

    // Configuration
    let dim = 128;
    let num_vectors = 10_000;
    let num_queries = 100;
    let k = 10; // Find 10 nearest neighbors
    let ef = 50; // Search parameter

    println!("Configuration:");
    println!("  Dimension: {}", dim);
    println!("  Vectors: {}", num_vectors);
    println!("  Queries: {}", num_queries);
    println!("  K (neighbors): {}", k);
    println!("  ef (search param): {}\n", ef);

    // Create index
    println!("Creating HNSW index (M=16, ef_construction=200)...");
    let mut index = HNSWIndex::new(dim, 16, 200);

    // Generate random vectors
    println!("Generating {} random {}-dimensional vectors...", num_vectors, dim);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| {
            let mut vec: Vec<f32> = (0..dim)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            // Normalize
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                vec.iter_mut().for_each(|x| *x /= norm);
            }
            vec
        })
        .collect();

    // Insert vectors
    println!("Inserting vectors into index...");
    let start = Instant::now();
    for (i, vec) in vectors.iter().enumerate() {
        index.insert(vec.clone());
        if (i + 1) % 1000 == 0 {
            print!("\r  Inserted: {}/{}", i + 1, num_vectors);
        }
    }
    let insert_duration = start.elapsed();
    println!("\n  Time: {:.2?}", insert_duration);
    println!("  Rate: {:.0} vectors/sec", num_vectors as f64 / insert_duration.as_secs_f64());

    // Display index statistics
    let stats = index.stats();
    println!("\nIndex Statistics:");
    println!("  Total nodes: {}", stats.total_nodes);
    println!("  Total layers: {}", stats.total_layers);
    println!("  Nodes per layer: {:?}", stats.nodes_per_layer);
    println!("  Total connections: {}", stats.total_connections);
    println!("  Avg connections/node: {:.2}", stats.avg_connections_per_node);

    // Generate query vectors
    println!("\nGenerating {} query vectors...", num_queries);
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| {
            let mut vec: Vec<f32> = (0..dim)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                vec.iter_mut().for_each(|x| *x /= norm);
            }
            vec
        })
        .collect();

    // Search performance test
    println!("\nSearching {} queries for {} nearest neighbors (ef={})...", num_queries, k, ef);
    let start = Instant::now();
    let mut total_results = 0;
    for query in &queries {
        let results = index.search(query, k, ef);
        total_results += results.len();
    }
    let search_duration = start.elapsed();
    let avg_latency = search_duration.as_micros() as f64 / num_queries as f64;

    println!("  Time: {:.2?}", search_duration);
    println!("  Avg latency: {:.2} μs", avg_latency);
    println!("  Throughput: {:.0} queries/sec", num_queries as f64 / search_duration.as_secs_f64());
    println!("  Total results: {}", total_results);

    // Example search with detailed results
    println!("\nExample Search:");
    let example_query = &queries[0];
    println!("  Query vector: [{:.3}, {:.3}, ..., {:.3}]",
        example_query[0], example_query[1], example_query[dim-1]);

    let results = index.search(example_query, 5, ef);
    println!("\n  Top 5 Nearest Neighbors:");
    for (i, result) in results.iter().enumerate() {
        println!("    {}. ID: {}, Distance: {:.6}", i + 1, result.id, result.distance);
    }

    // Verify search correctness with exact search
    println!("\nVerifying search quality (comparing with brute force)...");
    let test_query = &queries[0];

    // HNSW search
    let hnsw_results = index.search(test_query, k, ef);

    // Brute force search
    let mut exact_results: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(id, vec)| {
            let dist: f32 = test_query
                .iter()
                .zip(vec.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            (id, dist)
        })
        .collect();
    exact_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Calculate recall
    let hnsw_ids: std::collections::HashSet<_> = hnsw_results.iter().map(|r| r.id).collect();
    let exact_ids: std::collections::HashSet<_> = exact_results.iter().take(k).map(|(id, _)| *id).collect();
    let intersection = hnsw_ids.intersection(&exact_ids).count();
    let recall = intersection as f64 / k as f64;

    println!("  Recall@{}: {:.1}%", k, recall * 100.0);

    // Performance comparison
    println!("\nPerformance Comparison:");
    println!("  Target latency: <500 μs");
    println!("  Actual latency: {:.2} μs {}", avg_latency,
        if avg_latency < 500.0 { "✓" } else { "✗" });
    println!("  Target recall: ≥95%");
    println!("  Actual recall: {:.1}% {}", recall * 100.0,
        if recall >= 0.95 { "✓" } else { "✗" });

    println!("\n=== Demo Complete ===");
}
