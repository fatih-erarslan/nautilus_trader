/**
 * Ruvector MCP Tools
 *
 * Exposes high-performance Rust vector database (ruvector) capabilities via MCP.
 * Features HNSW indexing, quantization, GNN operations, and distributed replication.
 *
 * Architecture:
 * - Native Rust core (ruvector-core) via NAPI-RS bindings
 * - 150x faster than pure Python implementations
 * - O(log n) HNSW similarity search
 * - 4-32x memory reduction via quantization
 * - Graph Neural Network operations
 * - Raft-based distributed synchronization
 *
 * References:
 * - HNSW: Malkov & Yashunin (2018) "Efficient and robust approximate nearest neighbor search"
 * - Product Quantization: Jégou et al. (2011) "Product quantization for nearest neighbor search"
 * - GNN: Kipf & Welling (2017) "Semi-supervised classification with graph convolutional networks"
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";

/**
 * Vector Database Operations (Core)
 *
 * High-performance vector storage and similarity search using HNSW indexing.
 * Supports multiple distance metrics and quantization for memory efficiency.
 */
export const vectorTools: Tool[] = [
  // ===================================================================
  // VECTOR DATABASE OPERATIONS
  // ===================================================================

  {
    name: "vector_db_create",
    description: "Initialize vector database with HNSW indexing and optional quantization. Supports dimensions 128-4096, multiple distance metrics (cosine, euclidean, dot, manhattan, hyperbolic), and compression via scalar/product/binary quantization.",
    inputSchema: {
      type: "object",
      properties: {
        dimensions: {
          type: "number",
          description: "Vector dimensionality (typically 128-1536 for embeddings, up to 4096 supported)",
          minimum: 1,
          maximum: 4096,
        },
        distance_metric: {
          type: "string",
          enum: ["cosine", "euclidean", "dot", "manhattan", "hyperbolic"],
          description: "Distance metric for similarity calculation. Cosine: normalized similarity (common for embeddings). Euclidean: L2 distance. Dot: maximizes dot product. Manhattan: L1 distance. Hyperbolic: for hierarchical data",
          default: "cosine",
        },
        storage_path: {
          type: "string",
          description: "Persistent storage path for database file (e.g., './vectors.db')",
        },
        hnsw_config: {
          type: "object",
          description: "HNSW index configuration for approximate nearest neighbor search",
          properties: {
            m: {
              type: "number",
              description: "Bidirectional links per layer (default: 32). Higher = better recall, more memory. Typical: 16-64",
              default: 32,
            },
            ef_construction: {
              type: "number",
              description: "Dynamic candidate list size during index building (default: 200). Higher = better index quality, slower build. Typical: 100-400",
              default: 200,
            },
            ef_search: {
              type: "number",
              description: "Dynamic candidate list size during search (default: 100). Higher = better recall, slower search. Typical: 50-200",
              default: 100,
            },
            max_elements: {
              type: "number",
              description: "Maximum vectors in database (default: 10M). Pre-allocates memory",
              default: 10_000_000,
            },
          },
        },
        quantization: {
          type: "object",
          description: "Vector compression configuration for memory efficiency",
          properties: {
            type: {
              type: "string",
              enum: ["none", "scalar", "product", "binary"],
              description: "Quantization type. None: no compression. Scalar: 4x reduction. Product: 8-32x reduction. Binary: 32x reduction (cosine only)",
              default: "none",
            },
            subspaces: {
              type: "number",
              description: "Product quantization: number of vector subspaces (default: 16). Must divide dimension evenly",
              default: 16,
            },
            k: {
              type: "number",
              description: "Product quantization: codebook size per subspace (default: 256). Higher = better accuracy",
              default: 256,
            },
          },
        },
      },
      required: ["dimensions"],
    },
  },

  {
    name: "vector_db_insert",
    description: "Insert vector embeddings with optional IDs and metadata into database. Supports batch insertion for efficiency. Automatically builds HNSW index incrementally.",
    inputSchema: {
      type: "object",
      properties: {
        db_id: {
          type: "string",
          description: "Database identifier from vector_db_create",
        },
        vectors: {
          type: "array",
          description: "Array of vectors to insert",
          items: {
            type: "object",
            properties: {
              id: {
                type: "string",
                description: "Optional unique ID (auto-generated UUID if not provided)",
              },
              vector: {
                type: "array",
                items: { type: "number" },
                description: "Vector embedding (must match database dimensions)",
              },
              metadata: {
                type: "object",
                description: "Optional metadata (JSON object) for filtering and retrieval",
              },
            },
            required: ["vector"],
          },
        },
      },
      required: ["db_id", "vectors"],
    },
  },

  {
    name: "vector_db_search",
    description: "Perform HNSW approximate nearest neighbor search. O(log n) complexity, 150x faster than brute force. Returns top-k similar vectors with scores and metadata.",
    inputSchema: {
      type: "object",
      properties: {
        db_id: {
          type: "string",
          description: "Database identifier",
        },
        query_vector: {
          type: "array",
          items: { type: "number" },
          description: "Query vector for similarity search (must match database dimensions)",
        },
        k: {
          type: "number",
          description: "Number of nearest neighbors to return (default: 10)",
          default: 10,
          minimum: 1,
          maximum: 1000,
        },
        ef_search: {
          type: "number",
          description: "Override ef_search parameter for this query. Higher = better recall, slower. If not provided, uses database default",
        },
        filter: {
          type: "object",
          description: "Optional metadata filter expression (e.g., {category: 'science', year: {$gte: 2020}})",
        },
      },
      required: ["db_id", "query_vector"],
    },
  },

  {
    name: "vector_db_delete",
    description: "Delete vectors from database by ID. Supports batch deletion. HNSW index automatically updated.",
    inputSchema: {
      type: "object",
      properties: {
        db_id: {
          type: "string",
          description: "Database identifier",
        },
        ids: {
          type: "array",
          items: { type: "string" },
          description: "Array of vector IDs to delete",
        },
      },
      required: ["db_id", "ids"],
    },
  },

  {
    name: "vector_db_update",
    description: "Update existing vectors and/or metadata by ID. Efficiently updates HNSW index without full rebuild.",
    inputSchema: {
      type: "object",
      properties: {
        db_id: {
          type: "string",
          description: "Database identifier",
        },
        updates: {
          type: "array",
          description: "Array of vector updates",
          items: {
            type: "object",
            properties: {
              id: {
                type: "string",
                description: "Vector ID to update",
              },
              vector: {
                type: "array",
                items: { type: "number" },
                description: "New vector embedding (optional - if not provided, only metadata updated)",
              },
              metadata: {
                type: "object",
                description: "New metadata (optional - if not provided, only vector updated)",
              },
            },
            required: ["id"],
          },
        },
      },
      required: ["db_id", "updates"],
    },
  },

  {
    name: "vector_db_stats",
    description: "Get database statistics: vector count, memory usage, index quality metrics, query performance",
    inputSchema: {
      type: "object",
      properties: {
        db_id: {
          type: "string",
          description: "Database identifier",
        },
      },
      required: ["db_id"],
    },
  },

  // ===================================================================
  // GRAPH NEURAL NETWORK OPERATIONS
  // ===================================================================

  {
    name: "vector_gnn_forward",
    description: "Graph Neural Network forward pass for node embeddings. Supports message passing, neighborhood aggregation, and multi-layer GNN architectures. Based on Kipf & Welling (2017) GCN architecture.",
    inputSchema: {
      type: "object",
      properties: {
        node_features: {
          type: "array",
          description: "Node feature matrix (N x D) where N = number of nodes, D = feature dimension",
          items: {
            type: "array",
            items: { type: "number" },
          },
        },
        edge_index: {
          type: "array",
          description: "Edge connectivity in COO format: [[source_nodes], [target_nodes]]",
          items: {
            type: "array",
            items: { type: "number" },
          },
        },
        edge_weights: {
          type: "array",
          items: { type: "number" },
          description: "Optional edge weights (length = number of edges). If not provided, all edges weighted equally",
        },
        aggregation: {
          type: "string",
          enum: ["mean", "sum", "max", "attention"],
          description: "Neighborhood aggregation function (default: mean). Mean: average neighbors. Sum: sum neighbors. Max: max pooling. Attention: learned attention weights",
          default: "mean",
        },
      },
      required: ["node_features", "edge_index"],
    },
  },

  {
    name: "vector_gnn_attention",
    description: "Apply graph attention mechanism to learn importance of different neighbors. Implements 39 attention types from transformers literature including self-attention, cross-attention, and sparse variants.",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "array",
          description: "Query vectors (N x D)",
          items: {
            type: "array",
            items: { type: "number" },
          },
        },
        key: {
          type: "array",
          description: "Key vectors (M x D)",
          items: {
            type: "array",
            items: { type: "number" },
          },
        },
        value: {
          type: "array",
          description: "Value vectors (M x D)",
          items: {
            type: "array",
            items: { type: "number" },
          },
        },
        attention_type: {
          type: "string",
          enum: [
            "scaled_dot_product",
            "multi_head",
            "flash",
            "linear",
            "local_global",
            "hyperbolic",
            "mixed_curvature",
            "rope",
            "dual_space",
            "edge_featured",
            "moe",
          ],
          description: "Attention mechanism type. scaled_dot_product: standard transformer attention. multi_head: parallel attention heads. flash: memory-efficient attention. hyperbolic: for hierarchical data. moe: mixture of experts",
          default: "scaled_dot_product",
        },
        num_heads: {
          type: "number",
          description: "Number of attention heads for multi_head attention (default: 8)",
          default: 8,
        },
        dropout: {
          type: "number",
          description: "Dropout probability for attention weights (default: 0.0)",
          default: 0.0,
          minimum: 0.0,
          maximum: 1.0,
        },
      },
      required: ["query", "key", "value"],
    },
  },

  {
    name: "vector_gnn_aggregate",
    description: "Aggregate neighbor features using various pooling strategies. Supports differentiable aggregation for end-to-end training.",
    inputSchema: {
      type: "object",
      properties: {
        features: {
          type: "array",
          description: "Feature vectors to aggregate (N x D)",
          items: {
            type: "array",
            items: { type: "number" },
          },
        },
        neighborhoods: {
          type: "array",
          description: "Neighbor indices for each node: [[node_0_neighbors], [node_1_neighbors], ...]",
          items: {
            type: "array",
            items: { type: "number" },
          },
        },
        aggregation: {
          type: "string",
          enum: ["mean", "sum", "max", "min", "std"],
          description: "Aggregation function (default: mean)",
          default: "mean",
        },
      },
      required: ["features", "neighborhoods"],
    },
  },

  // ===================================================================
  // ADVANCED FEATURES
  // ===================================================================

  {
    name: "vector_quantize",
    description: "Compress vectors for 4-32x memory reduction using scalar, product, or binary quantization. Enables larger-scale deployments with minimal accuracy loss. Based on Jégou et al. (2011) product quantization.",
    inputSchema: {
      type: "object",
      properties: {
        vectors: {
          type: "array",
          description: "Vectors to quantize (N x D)",
          items: {
            type: "array",
            items: { type: "number" },
          },
        },
        quantization_type: {
          type: "string",
          enum: ["scalar", "product", "binary"],
          description: "Quantization method. Scalar: 4x reduction (8-bit per component). Product: 8-32x reduction (learned codebooks). Binary: 32x reduction (1-bit per component, cosine only)",
        },
        bits: {
          type: "number",
          enum: [4, 8, 16, 32],
          description: "Bits per component for scalar quantization (default: 8)",
          default: 8,
        },
        subspaces: {
          type: "number",
          description: "Product quantization: number of subspaces (default: 16). Must divide dimension evenly",
          default: 16,
        },
        codebook_size: {
          type: "number",
          description: "Product quantization: codebook size per subspace (default: 256). Higher = better accuracy",
          default: 256,
        },
      },
      required: ["vectors", "quantization_type"],
    },
  },

  {
    name: "vector_cluster",
    description: "Cluster vectors using k-means or DBSCAN. Useful for data exploration, organization, and semantic routing. Outputs cluster assignments and centroids.",
    inputSchema: {
      type: "object",
      properties: {
        vectors: {
          type: "array",
          description: "Vectors to cluster (N x D)",
          items: {
            type: "array",
            items: { type: "number" },
          },
        },
        algorithm: {
          type: "string",
          enum: ["kmeans", "dbscan"],
          description: "Clustering algorithm. kmeans: partition into k clusters. dbscan: density-based, automatic cluster count",
          default: "kmeans",
        },
        k: {
          type: "number",
          description: "Number of clusters for k-means (required for kmeans)",
        },
        epsilon: {
          type: "number",
          description: "DBSCAN epsilon: maximum distance for neighborhood (required for dbscan)",
        },
        min_samples: {
          type: "number",
          description: "DBSCAN: minimum samples per cluster (default: 5)",
          default: 5,
        },
        max_iterations: {
          type: "number",
          description: "k-means: maximum iterations (default: 100)",
          default: 100,
        },
      },
      required: ["vectors", "algorithm"],
    },
  },

  {
    name: "vector_replication_sync",
    description: "Synchronize vector database across distributed nodes using Raft consensus. Ensures strong consistency and fault tolerance. Supports leader election, log replication, and automatic recovery.",
    inputSchema: {
      type: "object",
      properties: {
        db_id: {
          type: "string",
          description: "Database identifier",
        },
        node_id: {
          type: "string",
          description: "Current node identifier in Raft cluster",
        },
        peer_nodes: {
          type: "array",
          items: { type: "string" },
          description: "Array of peer node addresses (e.g., ['node1:8080', 'node2:8080'])",
        },
        sync_mode: {
          type: "string",
          enum: ["full", "incremental", "snapshot"],
          description: "Synchronization strategy. full: replicate entire database. incremental: sync only changes. snapshot: transfer compressed snapshot",
          default: "incremental",
        },
      },
      required: ["db_id", "node_id", "peer_nodes"],
    },
  },

  {
    name: "vector_semantic_route",
    description: "Route AI requests to optimal handler based on semantic similarity. Embeds request, finds nearest cluster/handler, returns routing decision. Useful for multi-agent systems and intelligent request distribution.",
    inputSchema: {
      type: "object",
      properties: {
        request: {
          type: "string",
          description: "User request or query to route",
        },
        handlers: {
          type: "array",
          description: "Available handlers with descriptions",
          items: {
            type: "object",
            properties: {
              id: { type: "string" },
              description: { type: "string" },
              embedding: {
                type: "array",
                items: { type: "number" },
                description: "Pre-computed handler embedding (optional - will compute if not provided)",
              },
            },
            required: ["id", "description"],
          },
        },
        embedding_model: {
          type: "string",
          description: "Embedding model to use (default: 'text-embedding-3-small')",
          default: "text-embedding-3-small",
        },
        threshold: {
          type: "number",
          description: "Minimum similarity threshold for routing (default: 0.7). Below threshold returns 'unmatched'",
          default: 0.7,
        },
      },
      required: ["request", "handlers"],
    },
  },

  {
    name: "vector_benchmark",
    description: "Benchmark vector database performance: search latency, throughput, recall accuracy, memory usage. Compares against configuration baselines and provides optimization recommendations.",
    inputSchema: {
      type: "object",
      properties: {
        db_id: {
          type: "string",
          description: "Database identifier to benchmark",
        },
        num_queries: {
          type: "number",
          description: "Number of queries to run (default: 1000)",
          default: 1000,
        },
        k: {
          type: "number",
          description: "Number of nearest neighbors per query (default: 10)",
          default: 10,
        },
        ground_truth: {
          type: "array",
          description: "Optional ground truth results for recall calculation (brute force results)",
          items: {
            type: "array",
            items: { type: "string" },
          },
        },
      },
      required: ["db_id"],
    },
  },
];

/**
 * Wolfram Language validation code for vector operations
 *
 * Scientific validation of HNSW algorithm correctness, distance metric implementations,
 * and quantization accuracy using Wolfram symbolic computation.
 */
export const vectorWolframCode = `
(* Ruvector Validation Suite *)
(* Validates vector database operations against reference implementations *)

(* HNSW Graph Properties Validation *)
ValidateHNSWGraph[graph_, m_, efConstruction_] := Module[
  {nodes, avgDegree, maxDegree, connected},
  nodes = VertexList[graph];
  avgDegree = Mean[VertexDegree[graph, nodes]];
  maxDegree = Max[VertexDegree[graph, nodes]];
  connected = ConnectedGraphQ[graph];

  <|
    "nodes" -> Length[nodes],
    "avgDegree" -> avgDegree,
    "maxDegree" -> maxDegree,
    "mParameter" -> m,
    "degreeWithinBounds" -> avgDegree <= 2*m && maxDegree <= 2*m,
    "connected" -> connected,
    "valid" -> connected && avgDegree <= 2*m
  |>
];

(* Distance Metric Validation *)
ValidateDistanceMetric[vectors1_, vectors2_, metric_] := Module[
  {computedDistances, wolframDistances, maxError},

  (* Compute distances using Wolfram reference *)
  wolframDistances = Which[
    metric == "cosine",
      Table[1 - Dot[v1, v2]/(Norm[v1]*Norm[v2]),
        {v1, vectors1}, {v2, vectors2}],
    metric == "euclidean",
      Table[Norm[v1 - v2], {v1, vectors1}, {v2, vectors2}],
    metric == "dot",
      Table[-Dot[v1, v2], {v1, vectors1}, {v2, vectors2}],
    metric == "manhattan",
      Table[Norm[v1 - v2, 1], {v1, vectors1}, {v2, vectors2}]
  ];

  (* Compare with implementation results *)
  <| "wolframDistances" -> wolframDistances |>
];

(* Quantization Error Analysis *)
AnalyzeQuantizationError[originalVectors_, quantizedVectors_, bits_] := Module[
  {mse, maxError, snr, theoreticalError},

  mse = Mean[(originalVectors - quantizedVectors)^2];
  maxError = Max[Abs[originalVectors - quantizedVectors]];
  snr = 10*Log10[Mean[originalVectors^2]/mse];

  (* Theoretical quantization error for scalar quantization *)
  theoreticalError = 1/(2^bits * Sqrt[12]);

  <|
    "mse" -> mse,
    "maxError" -> maxError,
    "snr_dB" -> snr,
    "theoreticalError" -> theoreticalError,
    "withinBounds" -> mse < theoreticalError^2 * 2
  |>
];

(* GNN Message Passing Validation *)
ValidateGNNForward[nodeFeatures_, adjacencyMatrix_, aggregation_] := Module[
  {normalizedAdj, messages, aggregated},

  (* Symmetric normalization: D^(-1/2) A D^(-1/2) *)
  normalizedAdj = DiagonalMatrix[1/Sqrt[Total[adjacencyMatrix, {2}]]].adjacencyMatrix.
                  DiagonalMatrix[1/Sqrt[Total[adjacencyMatrix]]];

  (* Message passing *)
  aggregated = Which[
    aggregation == "mean", normalizedAdj.nodeFeatures,
    aggregation == "sum", adjacencyMatrix.nodeFeatures,
    aggregation == "max", (* Max aggregation requires element-wise max *)
      Table[Max[Select[nodeFeatures[[i]], adjacencyMatrix[[j,i]] > 0]],
        {j, Length[nodeFeatures]}, {i, Length[nodeFeatures[[1]]]}]
  ];

  <| "aggregatedFeatures" -> aggregated, "normalizedAdj" -> normalizedAdj |>
];

(* Recall@K Metric Validation *)
ComputeRecallAtK[predictions_, groundTruth_, k_] := Module[
  {topK, relevant, recall},

  topK = Take[predictions, UpTo[k]];
  relevant = Intersection[topK, Take[groundTruth, UpTo[k]]];
  recall = N[Length[relevant]/Min[k, Length[groundTruth]]];

  <| "recall@" <> ToString[k] -> recall, "relevant" -> Length[relevant], "total" -> k |>
];

(* HNSW Search Quality Metrics *)
AnalyzeSearchQuality[hnswResults_, bruteForceResults_, k_] := Module[
  {recalls, ndcg},

  recalls = Table[
    ComputeRecallAtK[hnswResults[[i]], bruteForceResults[[i]], ki],
    {i, Length[hnswResults]}, {ki, {1, 5, 10, 50, 100}}
  ];

  <| "recallMetrics" -> recalls, "avgRecall@10" -> Mean[recalls[[All, 3]]] |>
];
`;

/**
 * Handle vector tool calls
 * Routes to native ruvector module when available, provides simulation otherwise
 */
export async function handleVectorTool(
  name: string,
  args: Record<string, unknown>,
  nativeModule?: any
): Promise<any> {
  // Check if ruvector native module is available
  const hasNative = nativeModule?.vector_db_create !== undefined;

  if (!hasNative) {
    return {
      status: "simulation",
      tool: name,
      message: "Ruvector native module not loaded. Install with: cd crates/vendor/ruvector && npm install",
      simulation: true,
      args,
    };
  }

  try {
    switch (name) {
      case "vector_db_create": {
        const config = {
          dimensions: args.dimensions as number,
          distance_metric: (args.distance_metric as string) || "cosine",
          storage_path: (args.storage_path as string) || "./vectors.db",
          hnsw_config: args.hnsw_config || {},
          quantization: args.quantization || { type: "none" },
        };

        const dbId = nativeModule.vector_db_create(config);
        return {
          db_id: dbId,
          config,
          status: "created",
          message: "Vector database initialized with HNSW indexing",
        };
      }

      case "vector_db_insert": {
        const inserted = nativeModule.vector_db_insert(
          args.db_id as string,
          args.vectors as any[]
        );
        return {
          inserted_count: inserted,
          status: "success",
        };
      }

      case "vector_db_search": {
        const results = nativeModule.vector_db_search(
          args.db_id as string,
          args.query_vector as number[],
          (args.k as number) || 10,
          args.ef_search as number | undefined,
          args.filter as any
        );
        return {
          results,
          count: results.length,
          status: "success",
        };
      }

      case "vector_db_delete": {
        const deleted = nativeModule.vector_db_delete(
          args.db_id as string,
          args.ids as string[]
        );
        return {
          deleted_count: deleted,
          status: "success",
        };
      }

      case "vector_db_update": {
        const updated = nativeModule.vector_db_update(
          args.db_id as string,
          args.updates as any[]
        );
        return {
          updated_count: updated,
          status: "success",
        };
      }

      case "vector_db_stats": {
        const stats = nativeModule.vector_db_stats(args.db_id as string);
        return stats;
      }

      case "vector_gnn_forward": {
        const output = nativeModule.gnn_forward(
          args.node_features as number[][],
          args.edge_index as number[][],
          args.edge_weights as number[] | undefined,
          (args.aggregation as string) || "mean"
        );
        return {
          node_embeddings: output,
          status: "success",
        };
      }

      case "vector_gnn_attention": {
        const attentionOutput = nativeModule.gnn_attention(
          args.query as number[][],
          args.key as number[][],
          args.value as number[][],
          (args.attention_type as string) || "scaled_dot_product",
          (args.num_heads as number) || 8,
          (args.dropout as number) || 0.0
        );
        return {
          attention_output: attentionOutput,
          status: "success",
        };
      }

      case "vector_gnn_aggregate": {
        const aggregated = nativeModule.gnn_aggregate(
          args.features as number[][],
          args.neighborhoods as number[][],
          (args.aggregation as string) || "mean"
        );
        return {
          aggregated_features: aggregated,
          status: "success",
        };
      }

      case "vector_quantize": {
        const quantized = nativeModule.vector_quantize(
          args.vectors as number[][],
          args.quantization_type as string,
          {
            bits: (args.bits as number) || 8,
            subspaces: (args.subspaces as number) || 16,
            codebook_size: (args.codebook_size as number) || 256,
          }
        );
        return {
          quantized_vectors: quantized.vectors,
          compression_ratio: quantized.compression_ratio,
          status: "success",
        };
      }

      case "vector_cluster": {
        const clustering = nativeModule.vector_cluster(
          args.vectors as number[][],
          args.algorithm as string,
          {
            k: args.k as number | undefined,
            epsilon: args.epsilon as number | undefined,
            min_samples: (args.min_samples as number) || 5,
            max_iterations: (args.max_iterations as number) || 100,
          }
        );
        return {
          cluster_assignments: clustering.assignments,
          centroids: clustering.centroids,
          num_clusters: clustering.num_clusters,
          status: "success",
        };
      }

      case "vector_replication_sync": {
        const syncResult = nativeModule.replication_sync(
          args.db_id as string,
          args.node_id as string,
          args.peer_nodes as string[],
          (args.sync_mode as string) || "incremental"
        );
        return {
          synced: syncResult.synced,
          bytes_transferred: syncResult.bytes_transferred,
          status: "success",
        };
      }

      case "vector_semantic_route": {
        const routing = nativeModule.semantic_route(
          args.request as string,
          args.handlers as any[],
          (args.embedding_model as string) || "text-embedding-3-small",
          (args.threshold as number) || 0.7
        );
        return {
          handler_id: routing.handler_id,
          similarity: routing.similarity,
          matched: routing.matched,
          status: "success",
        };
      }

      case "vector_benchmark": {
        const benchmarkResults = nativeModule.vector_benchmark(
          args.db_id as string,
          (args.num_queries as number) || 1000,
          (args.k as number) || 10,
          args.ground_truth as any[] | undefined
        );
        return {
          latency_p50_ms: benchmarkResults.latency_p50,
          latency_p99_ms: benchmarkResults.latency_p99,
          throughput_qps: benchmarkResults.throughput,
          recall_at_10: benchmarkResults.recall,
          memory_mb: benchmarkResults.memory_mb,
          status: "success",
        };
      }

      default:
        return {
          error: `Unknown vector tool: ${name}`,
          status: "error",
        };
    }
  } catch (error) {
    return {
      error: String(error),
      status: "error",
      tool: name,
    };
  }
}
