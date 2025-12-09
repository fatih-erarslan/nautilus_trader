//! Demonstration of CSR graph format for Phase 2

use tengri_holographic_cortex::csr::CSRGraph;
use tengri_holographic_cortex::topology::{TopologyConfig, Cortex4};
use tengri_holographic_cortex::hyperbolic::LorentzPoint11;
use tengri_holographic_cortex::constants::HYPERBOLIC_DIM;

fn main() {
    println!("=== CSR Graph Format Demo ===\n");

    // Example 1: Build CSR from edge list
    println!("1. Building CSR from edge list:");
    let edges = vec![
        (0, 1, 0.5),
        (0, 2, 0.3),
        (1, 2, 0.8),
        (2, 0, 0.2),
        (2, 3, 0.6),
    ];

    let graph = CSRGraph::from_edge_list(&edges);
    println!("   Nodes: {}, Edges: {}", graph.num_nodes(), graph.num_edges());

    // Example 2: Iterate neighbors
    println!("\n2. Neighbors of node 0:");
    for (neighbor, weight) in graph.neighbors(0) {
        println!("   -> {} (weight: {:.2})", neighbor, weight);
    }

    println!("\n3. Node degrees:");
    for node in 0..graph.num_nodes() {
        println!("   Node {}: degree {}", node, graph.degree(node as u32));
    }

    // Example 4: Build from coupling tensor
    println!("\n4. Building CSR from Cortex4 coupling tensor:");
    let config = TopologyConfig::default();
    let cortex = Cortex4::new(config);
    let tensor_graph = CSRGraph::from_coupling_tensor(cortex.coupling_tensor());

    println!("   Engine graph: {} nodes, {} edges",
             tensor_graph.num_nodes(), tensor_graph.num_edges());

    // Example 5: Neighbor aggregation
    println!("\n5. Testing neighbor aggregation:");
    let features = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0; 4];

    graph.aggregate_neighbors_simd(&features, &mut output);

    println!("   Input features: {:?}", features);
    println!("   Aggregated output:");
    for (node, val) in output.iter().enumerate() {
        println!("     Node {}: {:.3}", node, val);
    }

    // Example 6: PageRank
    println!("\n6. Computing PageRank:");
    let ranks = graph.pagerank(0.85, 50);
    println!("   PageRank scores:");
    for (node, rank) in ranks.iter().enumerate() {
        println!("     Node {}: {:.4}", node, rank);
    }

    // Example 7: Hyperbolic distances
    println!("\n7. Precomputing hyperbolic distances:");
    let mut graph_with_distances = graph.clone();

    let embeddings = vec![
        LorentzPoint11::origin(),
        LorentzPoint11::from_euclidean(&vec![0.1; HYPERBOLIC_DIM]),
        LorentzPoint11::from_euclidean(&vec![0.2; HYPERBOLIC_DIM]),
        LorentzPoint11::from_euclidean(&vec![0.3; HYPERBOLIC_DIM]),
    ];

    graph_with_distances.precompute_hyperbolic_distances(&embeddings);

    if let Some(distances) = &graph_with_distances.hyperbolic_distances {
        println!("   Hyperbolic distances for {} edges:", distances.len());
        for (i, dist) in distances.iter().enumerate().take(5) {
            println!("     Edge {}: {:.4}", i, dist);
        }
    }

    // Example 8: GPU buffer export
    println!("\n8. Exporting to GPU buffers:");
    let (row_offsets, col_indices, edge_weights) = graph.to_gpu_buffers();

    println!("   Row offsets (len {}): {:?}", row_offsets.len(), row_offsets);
    println!("   Col indices (len {}): {:?}", col_indices.len(), col_indices);
    println!("   Edge weights (len {}): {:?}", edge_weights.len(), edge_weights);

    println!("\n=== CSR Demo Complete ===");
}
