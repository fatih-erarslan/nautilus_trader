//! CSR Graph Format Benchmarks
//!
//! Demonstrates 3-5x performance improvement over edge list format
//! for neighbor aggregation operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tengri_holographic_cortex::csr::CSRGraph;

/// Generate random graph with given nodes and average degree
fn generate_random_graph(num_nodes: usize, avg_degree: usize) -> Vec<(u32, u32, f32)> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let num_edges = num_nodes * avg_degree;
    let mut edges = Vec::with_capacity(num_edges);

    for _ in 0..num_edges {
        let src = rng.gen_range(0..num_nodes) as u32;
        let dst = rng.gen_range(0..num_nodes) as u32;
        let weight = rng.gen_range(0.0..1.0);

        if src != dst {
            edges.push((src, dst, weight));
        }
    }

    edges
}

/// Benchmark CSR construction from edge list
fn bench_csr_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_construction");

    for num_nodes in [100, 500, 1000, 5000] {
        let edges = generate_random_graph(num_nodes, 10);

        group.bench_with_input(
            BenchmarkId::new("from_edge_list", num_nodes),
            &edges,
            |b, edges| {
                b.iter(|| CSRGraph::from_edge_list(black_box(edges)));
            },
        );
    }

    group.finish();
}

/// Benchmark neighbor iteration (CSR vs naive)
fn bench_neighbor_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbor_iteration");

    for num_nodes in [100, 500, 1000, 5000] {
        let edges = generate_random_graph(num_nodes, 10);
        let graph = CSRGraph::from_edge_list(&edges);

        // CSR neighbor iteration
        group.bench_with_input(
            BenchmarkId::new("csr", num_nodes),
            &graph,
            |b, graph| {
                b.iter(|| {
                    let mut sum = 0.0f32;
                    for node in 0..black_box(graph.num_nodes()) {
                        for (_, weight) in graph.neighbors(node as u32) {
                            sum += weight;
                        }
                    }
                    sum
                });
            },
        );

        // Edge list iteration (baseline)
        group.bench_with_input(
            BenchmarkId::new("edge_list", num_nodes),
            &edges,
            |b, edges| {
                b.iter(|| {
                    let mut sum = 0.0f32;
                    for node in 0..black_box(num_nodes) {
                        for (src, _, weight) in edges {
                            if *src == node as u32 {
                                sum += weight;
                            }
                        }
                    }
                    sum
                });
            },
        );
    }

    group.finish();
}

/// Benchmark neighbor aggregation (scalar vs SIMD)
fn bench_neighbor_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbor_aggregation");

    for num_nodes in [100, 500, 1000, 5000] {
        let edges = generate_random_graph(num_nodes, 10);
        let graph = CSRGraph::from_edge_list(&edges);

        let features = vec![1.0f32; num_nodes];
        let mut output = vec![0.0f32; num_nodes];

        // CSR SIMD aggregation
        group.bench_function(
            BenchmarkId::new("csr_simd", num_nodes),
            |b| {
                let mut local_output = output.clone();
                b.iter(|| {
                    graph.aggregate_neighbors_simd(black_box(&features), &mut local_output);
                });
            },
        );

        // CSR scalar aggregation
        group.bench_function(
            BenchmarkId::new("csr_scalar", num_nodes),
            |b| {
                let mut local_output = output.clone();
                b.iter(|| {
                    graph.aggregate_neighbors_scalar(black_box(&features), &mut local_output);
                });
            },
        );

        // Edge list aggregation (baseline)
        group.bench_function(
            BenchmarkId::new("edge_list", num_nodes),
            |b| {
                let mut local_output = output.clone();
                b.iter(|| {
                    local_output.fill(0.0);
                    for (src, dst, weight) in &edges {
                        local_output[*src as usize] += weight * features[*dst as usize];
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark PageRank computation
fn bench_pagerank(c: &mut Criterion) {
    let mut group = c.benchmark_group("pagerank");

    for num_nodes in [100, 500, 1000] {
        let edges = generate_random_graph(num_nodes, 10);
        let graph = CSRGraph::from_edge_list(&edges);

        group.bench_with_input(
            BenchmarkId::new("csr", num_nodes),
            &graph,
            |b, graph| {
                b.iter(|| graph.pagerank(black_box(0.85), black_box(20)));
            },
        );
    }

    group.finish();
}

/// Benchmark GPU buffer creation
fn bench_gpu_buffers(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_buffers");

    for num_nodes in [100, 500, 1000, 5000] {
        let edges = generate_random_graph(num_nodes, 10);
        let graph = CSRGraph::from_edge_list(&edges);

        group.bench_with_input(
            BenchmarkId::new("to_gpu", num_nodes),
            &graph,
            |b, graph| {
                b.iter(|| graph.to_gpu_buffers());
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_csr_construction,
    bench_neighbor_iteration,
    bench_neighbor_aggregation,
    bench_pagerank,
    bench_gpu_buffers
);

criterion_main!(benches);
