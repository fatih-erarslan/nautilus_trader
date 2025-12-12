# Ruvector MCP Tools Guide

## Overview

The Dilithium MCP server exposes **15 high-performance vector database tools** powered by the Rust-based `ruvector` crate. These tools provide enterprise-grade vector similarity search, graph neural network operations, compression, and distributed replication capabilities.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DILITHIUM MCP SERVER                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │               Vector Tools (MCP Interface)                  │ │
│  └──────────────────────┬─────────────────────────────────────┘ │
│                         │                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         NAPI-RS Bindings (Zero-copy TypeScript FFI)        │ │
│  └──────────────────────┬─────────────────────────────────────┘ │
└─────────────────────────┼───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     RUVECTOR (Rust Core)                         │
│  ┌────────────┬────────────┬────────────┬────────────────────┐ │
│  │   HNSW     │    GNN     │ Quantize   │   Replication      │ │
│  │  O(log n)  │ Attention  │  4-32x     │   Raft Consensus   │ │
│  └────────────┴────────────┴────────────┴────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Storage: redb (ACID), memmap2 (zero-copy), simsimd (SIMD)│ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

| Operation | Complexity | Throughput | Notes |
|-----------|------------|------------|-------|
| HNSW Search | O(log n) | 10,000+ QPS | 150x faster than Python |
| Vector Insert | O(log n) | 50,000+ vectors/sec | Incremental index build |
| Quantization | O(n*d) | 1M vectors/sec | 4-32x memory reduction |
| GNN Forward | O(E + V*d²) | GPU-accelerated | E = edges, V = vertices |
| Clustering | O(n*k*i*d) | Parallel with Rayon | k = clusters, i = iterations |

## Tools Reference

### 1. vector_db_create

**Create HNSW-indexed vector database**

```typescript
{
  name: "vector_db_create",
  arguments: {
    dimensions: 384,              // Embedding dimension (128-4096)
    distance_metric: "cosine",    // cosine, euclidean, dot, manhattan, hyperbolic
    storage_path: "./vectors.db", // Persistent storage path
    hnsw_config: {
      m: 32,                      // Bidirectional links per layer
      ef_construction: 200,       // Build quality (higher = better)
      ef_search: 100,             // Search quality (higher = slower)
      max_elements: 10_000_000    // Maximum vectors
    },
    quantization: {
      type: "product",            // none, scalar, product, binary
      subspaces: 16,              // Product quantization subspaces
      k: 256                      // Codebook size
    }
  }
}
```

**Returns:** `{ db_id: string, config: {...}, status: "created" }`

**Use Cases:**
- RAG (Retrieval-Augmented Generation) systems
- Semantic search engines
- Document similarity matching
- Image/video similarity search

---

### 2. vector_db_insert

**Insert vectors with metadata**

```typescript
{
  name: "vector_db_insert",
  arguments: {
    db_id: "db-uuid",
    vectors: [
      {
        id: "doc1",                           // Optional (auto-generated UUID if not provided)
        vector: [0.1, 0.2, ..., 0.384],      // Float32Array or number[]
        metadata: {                           // Optional JSON metadata
          text: "Original document text",
          category: "AI",
          timestamp: 1234567890
        }
      }
    ]
  }
}
```

**Returns:** `{ inserted_count: number, status: "success" }`

**Notes:**
- Batch insertion for efficiency (recommended: 100-1000 vectors/batch)
- HNSW index automatically updated incrementally
- Metadata supports filtering in search queries

---

### 3. vector_db_search

**Semantic similarity search (HNSW)**

```typescript
{
  name: "vector_db_search",
  arguments: {
    db_id: "db-uuid",
    query_vector: [0.1, 0.2, ..., 0.384],
    k: 10,                              // Top-k nearest neighbors
    ef_search: 100,                     // Override search quality
    filter: {                           // Optional metadata filter
      category: "AI",
      year: { $gte: 2020 }
    }
  }
}
```

**Returns:**
```json
{
  "results": [
    { "id": "doc1", "score": 0.95, "metadata": {...} },
    { "id": "doc2", "score": 0.89, "metadata": {...} }
  ],
  "count": 10,
  "status": "success"
}
```

**Trade-offs:**
- `ef_search = 50`: Fast, 85-90% recall
- `ef_search = 100`: Balanced, 95-98% recall
- `ef_search = 200`: Slow, 99%+ recall

---

### 4. vector_db_delete

**Delete vectors by ID**

```typescript
{
  name: "vector_db_delete",
  arguments: {
    db_id: "db-uuid",
    ids: ["doc1", "doc2", "doc3"]
  }
}
```

**Returns:** `{ deleted_count: 3, status: "success" }`

---

### 5. vector_db_update

**Update vectors and/or metadata**

```typescript
{
  name: "vector_db_update",
  arguments: {
    db_id: "db-uuid",
    updates: [
      {
        id: "doc1",
        vector: [0.1, ..., 0.384],  // Optional: new vector
        metadata: { updated: true }  // Optional: new metadata
      }
    ]
  }
}
```

**Returns:** `{ updated_count: 1, status: "success" }`

---

### 6. vector_db_stats

**Get database statistics**

```typescript
{
  name: "vector_db_stats",
  arguments: { db_id: "db-uuid" }
}
```

**Returns:**
```json
{
  "vector_count": 1000000,
  "memory_mb": 450.2,
  "hnsw_layers": 5,
  "avg_degree": 31.8,
  "quantization": "product-16-256",
  "compression_ratio": 16.0
}
```

---

### 7. vector_gnn_forward

**Graph Neural Network forward pass**

```typescript
{
  name: "vector_gnn_forward",
  arguments: {
    node_features: [              // N x D feature matrix
      [1.0, 0.5, 0.2],
      [0.8, 0.3, 0.4]
    ],
    edge_index: [                 // COO format: [sources, targets]
      [0, 1, 1, 2],              // source nodes
      [1, 0, 2, 1]               // target nodes
    ],
    edge_weights: [1.0, 1.0, 0.8, 0.8],  // Optional
    aggregation: "mean"           // mean, sum, max, attention
  }
}
```

**Returns:** `{ node_embeddings: [...], status: "success" }`

**Use Cases:**
- Social network analysis
- Knowledge graph embeddings
- Molecular property prediction
- Citation network analysis

---

### 8. vector_gnn_attention

**Graph attention mechanism (39 variants)**

```typescript
{
  name: "vector_gnn_attention",
  arguments: {
    query: [[...]],                    // N x D
    key: [[...]],                      // M x D
    value: [[...]],                    // M x D
    attention_type: "scaled_dot_product",  // See full list below
    num_heads: 8,
    dropout: 0.1
  }
}
```

**Attention Types:**
- `scaled_dot_product`: Standard transformer attention
- `multi_head`: Parallel attention heads
- `flash`: Memory-efficient (Flash Attention)
- `linear`: Linear complexity attention
- `local_global`: Local + global attention
- `hyperbolic`: For hierarchical data
- `mixed_curvature`: Multi-curvature attention
- `rope`: Rotary position embeddings
- `dual_space`: Euclidean + hyperbolic
- `edge_featured`: Edge feature attention
- `moe`: Mixture of experts attention

**Returns:** `{ attention_output: [...], status: "success" }`

---

### 9. vector_gnn_aggregate

**Neighborhood aggregation**

```typescript
{
  name: "vector_gnn_aggregate",
  arguments: {
    features: [[...]],                 // N x D
    neighborhoods: [                   // Neighbor indices per node
      [1, 2, 3],                      // Node 0's neighbors
      [0, 2],                         // Node 1's neighbors
      [0, 1]                          // Node 2's neighbors
    ],
    aggregation: "mean"                // mean, sum, max, min, std
  }
}
```

**Returns:** `{ aggregated_features: [...], status: "success" }`

---

### 10. vector_quantize

**Vector compression (4-32x reduction)**

```typescript
{
  name: "vector_quantize",
  arguments: {
    vectors: [[...]],
    quantization_type: "product",      // scalar, product, binary
    bits: 8,                           // Scalar: 4, 8, 16, 32
    subspaces: 16,                     // Product: subspace count
    codebook_size: 256                 // Product: codebook size
  }
}
```

**Compression Ratios:**
- **Scalar (8-bit)**: 4x reduction, 1-2% accuracy loss
- **Product (16 subspaces, k=256)**: 16x reduction, 2-5% accuracy loss
- **Binary**: 32x reduction, 5-10% accuracy loss (cosine only)

**Returns:**
```json
{
  "quantized_vectors": [...],
  "compression_ratio": 16.0,
  "status": "success"
}
```

---

### 11. vector_cluster

**K-means or DBSCAN clustering**

```typescript
{
  name: "vector_cluster",
  arguments: {
    vectors: [[...]],
    algorithm: "kmeans",               // kmeans, dbscan
    k: 10,                             // K-means: number of clusters
    epsilon: 0.5,                      // DBSCAN: neighborhood radius
    min_samples: 5,                    // DBSCAN: min samples per cluster
    max_iterations: 100                // K-means: max iterations
  }
}
```

**Returns:**
```json
{
  "cluster_assignments": [0, 1, 0, 2, ...],
  "centroids": [[...]],
  "num_clusters": 10,
  "status": "success"
}
```

**Use Cases:**
- Document categorization
- User segmentation
- Anomaly detection
- Data exploration

---

### 12. vector_replication_sync

**Raft-based distributed synchronization**

```typescript
{
  name: "vector_replication_sync",
  arguments: {
    db_id: "db-uuid",
    node_id: "node-1",
    peer_nodes: ["node-2:8080", "node-3:8080"],
    sync_mode: "incremental"           // full, incremental, snapshot
  }
}
```

**Returns:**
```json
{
  "synced": true,
  "bytes_transferred": 1048576,
  "status": "success"
}
```

**Sync Modes:**
- **full**: Replicate entire database
- **incremental**: Sync only changes (log-based)
- **snapshot**: Compressed snapshot transfer

---

### 13. vector_semantic_route

**Intelligent request routing**

```typescript
{
  name: "vector_semantic_route",
  arguments: {
    request: "How do I train a neural network?",
    handlers: [
      { id: "ml-expert", description: "Machine learning and neural networks" },
      { id: "physics-expert", description: "Quantum physics" }
    ],
    embedding_model: "text-embedding-3-small",
    threshold: 0.7                     // Minimum similarity
  }
}
```

**Returns:**
```json
{
  "handler_id": "ml-expert",
  "similarity": 0.92,
  "matched": true,
  "status": "success"
}
```

**Use Cases:**
- Multi-agent systems
- Load balancing
- Intent classification
- API gateway routing

---

### 14. vector_benchmark

**Performance benchmarking**

```typescript
{
  name: "vector_benchmark",
  arguments: {
    db_id: "db-uuid",
    num_queries: 1000,
    k: 10,
    ground_truth: [...]                // Optional: for recall calculation
  }
}
```

**Returns:**
```json
{
  "latency_p50_ms": 1.2,
  "latency_p99_ms": 4.8,
  "throughput_qps": 8333,
  "recall_at_10": 0.98,
  "memory_mb": 450.2,
  "status": "success"
}
```

---

### 15. vector_db_close (Implicit)

Database instances are automatically closed when the MCP server shuts down or when `db_id` references are garbage collected.

## Integration Examples

### RAG Pipeline

```typescript
// 1. Create database
const { db_id } = await mcp.call("vector_db_create", {
  dimensions: 1536,
  distance_metric: "cosine"
});

// 2. Index documents
await mcp.call("vector_db_insert", {
  db_id,
  vectors: documents.map(doc => ({
    vector: await embed(doc.text),
    metadata: { text: doc.text, url: doc.url }
  }))
});

// 3. Semantic search
const { results } = await mcp.call("vector_db_search", {
  db_id,
  query_vector: await embed(userQuery),
  k: 5
});

// 4. Generate with context
const context = results.map(r => r.metadata.text).join("\n");
const response = await llm.generate({ context, query: userQuery });
```

### Multi-Agent Routing

```typescript
const { handler_id } = await mcp.call("vector_semantic_route", {
  request: userMessage,
  handlers: [
    { id: "sql-agent", description: "Database queries and SQL" },
    { id: "python-agent", description: "Python coding and data science" },
    { id: "math-agent", description: "Mathematics and calculations" }
  ],
  threshold: 0.7
});

await agents[handler_id].process(userMessage);
```

### Knowledge Graph Embeddings

```typescript
// 1. GNN forward pass
const { node_embeddings } = await mcp.call("vector_gnn_forward", {
  node_features: initialFeatures,
  edge_index: knowledgeGraphEdges,
  aggregation: "attention"
});

// 2. Store in vector DB
await mcp.call("vector_db_insert", {
  db_id,
  vectors: node_embeddings.map((emb, i) => ({
    id: entityIds[i],
    vector: emb,
    metadata: { type: "entity", name: entityNames[i] }
  }))
});
```

## Wolfram Validation

All vector operations are validated using Wolfram Language symbolic computation:

```wolfram
(* HNSW Graph Properties *)
ValidateHNSWGraph[graph_, m_, efConstruction_]

(* Distance Metric Correctness *)
ValidateDistanceMetric[vectors1_, vectors2_, metric_]

(* Quantization Error Analysis *)
AnalyzeQuantizationError[original_, quantized_, bits_]

(* GNN Message Passing *)
ValidateGNNForward[nodeFeatures_, adjacency_, aggregation_]

(* Recall@K Metrics *)
ComputeRecallAtK[predictions_, groundTruth_, k_]
```

See `vector-tools.ts` for full Wolfram validation suite.

## Native Module Setup

### Building Ruvector

```bash
cd crates/vendor/ruvector
cargo build --release

# Build Node.js bindings
cd crates/ruvector-node
npm install
npm run build
```

### Linking to Dilithium MCP

The dilithium-mcp server automatically discovers ruvector native modules at:
1. `$DILITHIUM_NATIVE_PATH` (env var)
2. `native/dilithium-native.darwin-x64.node`
3. `native/dilithium-native.darwin-arm64.node`
4. `native/target/release/libdilithium_native.dylib`

### Simulation Mode

If native module is not available, tools return simulation responses:
```json
{
  "status": "simulation",
  "message": "Ruvector native module not loaded",
  "simulation": true
}
```

## Performance Tuning

### HNSW Parameters

| Dataset Size | M | ef_construction | ef_search | Memory | QPS |
|--------------|---|-----------------|-----------|--------|-----|
| < 100K | 16 | 100 | 50 | Low | Very Fast |
| 100K - 1M | 32 | 200 | 100 | Medium | Fast |
| 1M - 10M | 48 | 400 | 200 | High | Medium |
| > 10M | 64 | 800 | 400 | Very High | Slow |

### Quantization Trade-offs

```
No Quantization:   100% accuracy, 1x memory, 1x speed
Scalar 8-bit:       99% accuracy, 4x memory, 1.5x speed
Product 16-256:     95% accuracy, 16x memory, 2x speed
Binary:             90% accuracy, 32x memory, 3x speed
```

## References

1. **HNSW**: Malkov, Y., & Yashunin, D. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." IEEE Transactions on Pattern Analysis and Machine Intelligence.

2. **Product Quantization**: Jégou, H., Douze, M., & Schmid, C. (2011). "Product quantization for nearest neighbor search." IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(1), 117-128.

3. **GNN**: Kipf, T. N., & Welling, M. (2017). "Semi-supervised classification with graph convolutional networks." International Conference on Learning Representations (ICLR).

4. **Flash Attention**: Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS.

5. **Raft Consensus**: Ongaro, D., & Ousterhout, J. (2014). "In search of an understandable consensus algorithm." USENIX Annual Technical Conference (ATC).

## License

Ruvector MCP Tools are part of the Dilithium MCP server.
Licensed under MIT License.
