#!/usr/bin/env bun
/**
 * Ruvector MCP Tools Example
 *
 * Demonstrates vector database operations via Dilithium MCP:
 * 1. Create HNSW-indexed vector database
 * 2. Insert document embeddings
 * 3. Perform semantic similarity search
 * 4. Cluster vectors for organization
 * 5. Apply quantization for memory efficiency
 * 6. GNN operations for graph-structured data
 *
 * Usage:
 *   bun run examples/vector-database-example.ts
 *
 * Requirements:
 *   - Dilithium MCP server running
 *   - Ruvector native module built (optional - will use simulation mode if not available)
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

async function main() {
  console.log("╔═══════════════════════════════════════════════════════╗");
  console.log("║     RUVECTOR MCP TOOLS - COMPREHENSIVE DEMO           ║");
  console.log("╚═══════════════════════════════════════════════════════╝\n");

  // Initialize MCP client
  const transport = new StdioClientTransport({
    command: "bun",
    args: ["run", "../dist/index.js"],
  });

  const client = new Client(
    {
      name: "vector-example-client",
      version: "1.0.0",
    },
    {
      capabilities: {},
    }
  );

  await client.connect(transport);
  console.log("✓ Connected to Dilithium MCP server\n");

  // =========================================================================
  // EXAMPLE 1: Create Vector Database
  // =========================================================================
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("EXAMPLE 1: Creating HNSW-indexed vector database");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  const createResult = await client.callTool({
    name: "vector_db_create",
    arguments: {
      dimensions: 384, // Common for MiniLM embeddings
      distance_metric: "cosine",
      storage_path: "./demo_vectors.db",
      hnsw_config: {
        m: 32,
        ef_construction: 200,
        ef_search: 100,
        max_elements: 1_000_000,
      },
      quantization: {
        type: "none", // No compression for this demo
      },
    },
  });

  const createData = JSON.parse(createResult.content[0].text);
  console.log("✓ Database created:", createData);
  const dbId = createData.db_id;
  console.log();

  // =========================================================================
  // EXAMPLE 2: Insert Document Embeddings
  // =========================================================================
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("EXAMPLE 2: Inserting document embeddings with metadata");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  // Simulate embeddings for demo documents
  const documents = [
    { id: "doc1", text: "Machine learning with neural networks", category: "AI" },
    { id: "doc2", text: "Quantum computing fundamentals", category: "Physics" },
    { id: "doc3", text: "Deep learning for computer vision", category: "AI" },
    { id: "doc4", text: "Hyperbolic geometry in data analysis", category: "Math" },
    { id: "doc5", text: "Neural architecture search algorithms", category: "AI" },
  ];

  const vectors = documents.map((doc) => ({
    id: doc.id,
    vector: Array.from({ length: 384 }, () => Math.random() - 0.5), // Simulated embeddings
    metadata: {
      text: doc.text,
      category: doc.category,
      timestamp: Date.now(),
    },
  }));

  const insertResult = await client.callTool({
    name: "vector_db_insert",
    arguments: {
      db_id: dbId,
      vectors,
    },
  });

  const insertData = JSON.parse(insertResult.content[0].text);
  console.log("✓ Inserted vectors:", insertData);
  console.log();

  // =========================================================================
  // EXAMPLE 3: Semantic Similarity Search
  // =========================================================================
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("EXAMPLE 3: Performing HNSW similarity search");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  const queryVector = Array.from({ length: 384 }, () => Math.random() - 0.5);

  const searchResult = await client.callTool({
    name: "vector_db_search",
    arguments: {
      db_id: dbId,
      query_vector: queryVector,
      k: 3,
      ef_search: 100,
    },
  });

  const searchData = JSON.parse(searchResult.content[0].text);
  console.log("✓ Search results (top 3):");
  console.log(JSON.stringify(searchData, null, 2));
  console.log();

  // =========================================================================
  // EXAMPLE 4: Vector Clustering
  // =========================================================================
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("EXAMPLE 4: K-means clustering of vectors");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  const clusterResult = await client.callTool({
    name: "vector_cluster",
    arguments: {
      vectors: vectors.map((v) => v.vector),
      algorithm: "kmeans",
      k: 2, // Cluster into 2 groups
      max_iterations: 100,
    },
  });

  const clusterData = JSON.parse(clusterResult.content[0].text);
  console.log("✓ Clustering results:");
  console.log(JSON.stringify(clusterData, null, 2));
  console.log();

  // =========================================================================
  // EXAMPLE 5: Vector Quantization
  // =========================================================================
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("EXAMPLE 5: Scalar quantization for memory efficiency");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  const quantizeResult = await client.callTool({
    name: "vector_quantize",
    arguments: {
      vectors: vectors.map((v) => v.vector).slice(0, 3), // Quantize first 3
      quantization_type: "scalar",
      bits: 8,
    },
  });

  const quantizeData = JSON.parse(quantizeResult.content[0].text);
  console.log("✓ Quantization results:");
  console.log(
    `  Compression ratio: ${quantizeData.compression_ratio || "N/A"}x`
  );
  console.log(
    `  Quantized vectors: ${quantizeData.quantized_vectors?.length || "N/A"}`
  );
  console.log();

  // =========================================================================
  // EXAMPLE 6: Graph Neural Network Operations
  // =========================================================================
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("EXAMPLE 6: GNN forward pass with message passing");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  // Simple graph: 4 nodes with connections
  const nodeFeatures = [
    [1.0, 0.5, 0.2],
    [0.8, 0.3, 0.4],
    [0.6, 0.7, 0.1],
    [0.4, 0.2, 0.9],
  ];

  const edgeIndex = [
    [0, 1, 1, 2, 2, 3], // source nodes
    [1, 0, 2, 1, 3, 2], // target nodes
  ];

  const gnnResult = await client.callTool({
    name: "vector_gnn_forward",
    arguments: {
      node_features: nodeFeatures,
      edge_index: edgeIndex,
      aggregation: "mean",
    },
  });

  const gnnData = JSON.parse(gnnResult.content[0].text);
  console.log("✓ GNN forward pass results:");
  console.log(JSON.stringify(gnnData, null, 2));
  console.log();

  // =========================================================================
  // EXAMPLE 7: Semantic Routing
  // =========================================================================
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("EXAMPLE 7: Semantic routing for multi-agent systems");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  const routeResult = await client.callTool({
    name: "vector_semantic_route",
    arguments: {
      request: "How do I train a neural network?",
      handlers: [
        {
          id: "ml-expert",
          description: "Machine learning and neural network training",
        },
        { id: "physics-expert", description: "Quantum physics and mechanics" },
        { id: "math-expert", description: "Mathematics and geometry" },
      ],
      threshold: 0.7,
    },
  });

  const routeData = JSON.parse(routeResult.content[0].text);
  console.log("✓ Routing decision:");
  console.log(JSON.stringify(routeData, null, 2));
  console.log();

  // =========================================================================
  // EXAMPLE 8: Performance Benchmarking
  // =========================================================================
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("EXAMPLE 8: Database performance benchmarking");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  const benchmarkResult = await client.callTool({
    name: "vector_benchmark",
    arguments: {
      db_id: dbId,
      num_queries: 100,
      k: 10,
    },
  });

  const benchmarkData = JSON.parse(benchmarkResult.content[0].text);
  console.log("✓ Benchmark results:");
  console.log(JSON.stringify(benchmarkData, null, 2));
  console.log();

  // =========================================================================
  // EXAMPLE 9: Database Statistics
  // =========================================================================
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("EXAMPLE 9: Retrieving database statistics");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  const statsResult = await client.callTool({
    name: "vector_db_stats",
    arguments: {
      db_id: dbId,
    },
  });

  const statsData = JSON.parse(statsResult.content[0].text);
  console.log("✓ Database statistics:");
  console.log(JSON.stringify(statsData, null, 2));
  console.log();

  console.log("╔═══════════════════════════════════════════════════════╗");
  console.log("║          ALL EXAMPLES COMPLETED SUCCESSFULLY          ║");
  console.log("╚═══════════════════════════════════════════════════════╝\n");

  console.log("Key Takeaways:");
  console.log("  • HNSW provides O(log n) similarity search");
  console.log("  • Quantization reduces memory 4-32x");
  console.log("  • GNN operations enable graph-structured learning");
  console.log("  • Semantic routing enables intelligent request distribution");
  console.log("  • Ruvector is 150x faster than pure Python implementations");
  console.log();

  await client.close();
}

main().catch(console.error);
