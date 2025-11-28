# HyperPhysics Similarity Search Architecture Blueprint

## Triangular Constraint Model for Ultra-Low Latency Trading

**Version**: 1.0.0  
**Date**: November 2024  
**Status**: Implementation Ready

---

## Executive Summary

This blueprint defines a dual-crate similarity search architecture combining HNSW (Hierarchical Navigable Small World) for sub-microsecond hot-path queries with LSH (Locality Sensitive Hashing) for specialized use cases where LSH provides irreplaceable value. The architecture follows the **triangular constraint model** from Christiansen & Chater's "Creating Language" framework, ensuring self-consistent system behavior across three timescales.

### Key Decisions

| Decision | Rationale |
|----------|-----------|
| **HNSW for hot path** | 10x faster than Annoy, O(log n) queries vs LSH's O(n^ρ) |
| **LSH for streaming/whale** | O(1) insertion, variable-cardinality set similarity |
| **No pure hybrid** | Complexity doesn't justify marginal gains |
| **Triangular architecture** | Ensures mutual constraints prevent drift |

---

## Part I: Comparative Research Findings

### 1.1 HNSW Performance Analysis

#### Benchmark Data (Multiple Sources)

| Metric | Value | Source |
|--------|-------|--------|
| Query throughput | 15,000 req/s | hnswlib-rs on i9-13900HX |
| Recall@10 | 0.9907 | SIFT1M (1M vectors, 128D) |
| Query complexity | O(log n) | Theoretical |
| Index build | O(n log n) | Theoretical |

#### Key Findings

1. **State-of-the-art recall-speed tradeoffs** - Graph-based algorithms consistently outperform LSH across all benchmarks (ANN-Benchmarks, Big-ANN)

2. **Memory efficiency** - HNSW uses ~1.1x raw vector size with M=16, while achieving 95%+ recall

3. **Scalability** - Linear scaling to billions with DiskANN/Vamana extensions

#### Industry Adoption

- **USearch**: 10x faster than FAISS, used by ClickHouse, DuckDB
- **NVIDIA CAGRA**: 40x speedup for GPU index building
- **Microsoft DiskANN**: Billion-scale on commodity hardware (64GB RAM + SSD)

### 1.2 LSH Performance Analysis

#### Fundamental Limitations

```
Time Complexity: O(n^ρ) where ρ = log(1/p₁)/log(1/p₂)
For typical parameters: ρ ≈ 0.75

For n = 100,000 patterns:
  O(100,000^0.75 × log(100,000)) ≈ O(35,000) distance computations
  
VERDICT: Fundamentally incompatible with sub-microsecond latency
```

#### Accuracy-Speed Tradeoff (Empirical)

| Speedup | Accuracy Loss |
|---------|---------------|
| 2x | 10% |
| 5x | 40% |
| 10x | 60%+ |

#### Expert Assessment

> "I haven't been very impressed by LSH. Graph-based algorithms seem to be the state of the art, in particular HNSW."
> — Erik Bernhardsson (Creator of Annoy, Spotify)

### 1.3 When LSH Provides Irreplaceable Value

Despite inferior query performance, LSH remains essential for:

| Use Case | Why LSH | Why Not HNSW |
|----------|---------|--------------|
| **Streaming Ingestion** | O(1) insertion | O(log n) graph updates per insert |
| **Whale Detection (MinHash)** | Variable-cardinality sets | Requires fixed-dimension vectors |
| **pBit Thermodynamic Sampling** | Hash collision = activation probability | Deterministic, no probability model |
| **FPGA Hardware** | Hash-then-lookup pipelines | Data-dependent graph traversal |

---

## Part II: Triangular Constraint Architecture

### 2.1 Theoretical Foundation

From Christiansen & Chater's "Creating Language" (Figure 1.6):

```
                           ACQUISITION
                         ╱ (Intermediate) ╲
                        ╱   Timescale      ╲
                       ╱                    ╲
                      ╱                      ╲
                     ╱ constrains  constrains ╲
                    ╱   what can    what is    ╲
                   ╱    evolve      useful      ╲
                  ╱                              ╲
   EVOLUTION ◄────────────────────────────────────► PROCESSING
   (Slowest)         constrains each other         (Fastest)
   Timescale                                       Timescale
```

**Key Insight**: Each component constrains AND is constrained by the others. The system achieves stability through mutual constraint, not hierarchy.

### 2.2 Mapping to HyperPhysics Similarity

| Triangle Vertex | HyperPhysics Component | Timescale | Responsibility |
|-----------------|------------------------|-----------|----------------|
| **Processing** | HNSW Hot Index | Microseconds | Sub-μs queries, hot path |
| **Acquisition** | LSH Streaming | Milliseconds | Pattern ingestion, O(1) insert |
| **Evolution** | pBit / Parameter Tuning | Seconds-Hours | Thermodynamic optimization |

### 2.3 Constraint Edges

#### Processing → Acquisition
- Query latency budget constrains what patterns are useful
- Recall requirements affect pattern quality threshold
- Hot index capacity limits what can be promoted

#### Acquisition → Processing  
- Pattern promotion rate constrained by insert latency
- Streaming quality affects hot index relevance
- MinHash whale signals trigger trading actions

#### Acquisition → Evolution
- Available patterns constrain what can be optimized
- Hash collision statistics inform temperature tuning
- Ingestion rate limits evolution feedback frequency

#### Evolution → Acquisition
- Temperature controls pattern sampling distribution
- Hash parameter adjustments affect collision rates
- Fitness signals guide promotion decisions

#### Evolution → Processing
- ef_search tuning based on latency feedback
- Recall optimization adjusts index parameters
- Thermodynamic equilibrium guides capacity planning

#### Processing → Evolution
- Latency measurements drive parameter search
- Recall feedback shapes optimization objective
- Query patterns inform evolution priorities

---

## Part III: Existing LSH Blueprint Bottleneck Analysis

### 3.1 Location in Current Architecture

**File**: `pbit-lsh-cortical-bus-blueprint.md` (28.96 KB, 879 lines)  
**Status**: Blueprint only, no implemented code

**Design Location**: LSH Memory Controller layer between Cortical Bus Network and pBit Memory Arrays

**Configuration**:
- 8 hash tables
- 4 hash functions per table
- 100 bucket split threshold
- Cosine similarity metric

### 3.2 Critical Bottlenecks Identified

#### Bottleneck #1: WTAHash partial_cmp in Hot Loop

```rust
// CURRENT (prevents SIMD vectorization)
.max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())

// Impact: ~50-100ns per hash vs ~5-10ns with SIMD
// Fix: Use total_cmp or SIMD-friendly integer comparisons
```

#### Bottleneck #2: HashSet Allocation in Hot Path

```rust
// CURRENT (heap allocation every query)
let mut visited = HashSet::new();

// Impact: ~200-500ns per query
// Fix: Pre-allocated BloomFilter or fixed-size bitset
```

#### Bottleneck #3: Sequential Hash Table Iteration

```rust
// CURRENT (no parallelism)
for table in &self.hash_tables { ... }

// Impact: 8 tables × 100ns = 800ns minimum
// Fix: Parallel iteration with rayon or SIMD-parallel hash
```

#### Bottleneck #4: Multi-Probe Generation Allocations

```rust
// CURRENT (creates Vec)
let probes = self.generate_probes(hash, threshold);

// Impact: ~100-300ns for probe generation
// Fix: Iterator-based probing with no allocation
```

#### Bottleneck #5: Linear Bucket Scan

```rust
// CURRENT (expensive similarity per item)
for item in &bucket.items { similarity_check(item); }

// Impact: 100 items × 50ns = 5μs per bucket
// Fix: SIMD-accelerated batch similarity using simsimd
```

#### Bottleneck #6: Final Sorting

```rust
// CURRENT (full sort)
results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

// Impact: ~500ns for 100 results
// Fix: Partial sort using select_nth_unstable_by
```

### 3.3 Theoretical Performance Bound

For n = 100,000 patterns with LSH:

```
Distance computations: O(n^0.75 × log(n)) ≈ O(35,000)
At 50ns per computation: 35,000 × 50ns = 1.75ms

CONCLUSION: LSH fundamentally cannot achieve sub-microsecond latency
```

---

## Part IV: Implementation Specifications

### 4.1 Workspace Structure

```
hyperphysics-similarity/
├── Cargo.toml                    # Workspace root
├── README.md
└── crates/
    ├── hyperphysics-hnsw/       # Processing Layer
    │   ├── Cargo.toml
    │   └── src/
    │       ├── lib.rs           # Traits, constants
    │       ├── config.rs        # IndexConfig builder
    │       ├── error.rs         # HnswError types
    │       ├── metric.rs        # HyperbolicMetric
    │       └── index.rs         # HotIndex
    │
    ├── hyperphysics-lsh/        # Acquisition Layer
    │   ├── Cargo.toml
    │   └── src/
    │       ├── lib.rs           # Traits, use cases
    │       ├── config.rs        # LshConfig
    │       ├── error.rs         # LshError types
    │       ├── hash.rs          # SimHash, MinHash, SRP
    │       └── index.rs         # StreamingLshIndex
    │
    └── hyperphysics-similarity/ # Integration Layer
        ├── Cargo.toml
        └── src/
            ├── lib.rs           # Triangular traits
            ├── config.rs        # SearchConfig
            ├── error.rs         # HybridError
            └── router.rs        # HybridIndex
```

### 4.2 Performance Targets

| Operation | Target | Layer | Allocation |
|-----------|--------|-------|------------|
| HNSW query | <1μs | Processing | Minimal |
| LSH hash computation | <100ns | Acquisition | Zero |
| Single insertion | <500ns | Acquisition | Zero |
| Streaming insertion | <200ns | Acquisition | Zero |
| LSH query (when appropriate) | <5μs | Acquisition | Minimal |
| Pattern promotion | <50μs | Acq→Proc | Batch |

### 4.3 Core Dependencies

```toml
[workspace.dependencies]
# HNSW
usearch = "2.12"
simsimd = "4.0"

# Concurrency
parking_lot = "0.12"
crossbeam = "0.8"

# Zero-allocation containers
arrayvec = "0.7"
tinyvec = "1.6"
bitvec = "1.0"

# Hashing
ahash = "0.8"
xxhash-rust = "0.8"

# Persistence
memmap2 = "0.9"
bincode = "1.3"

# Observability
tracing = "0.1"
```

---

## Part V: Key Implementation Details

### 5.1 Hyperbolic Metric (Poincaré Ball Model)

The HyperPhysics system operates in hyperbolic space for consciousness modeling. The Poincaré ball metric:

```
d_H(u, v) = √|K| · arcosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))
```

**Implementation Highlights**:

```rust
pub struct HyperbolicMetric {
    dimensions: usize,
    curvature: f32,      // K, typically -1.0
    sqrt_neg_k: f32,     // Pre-computed √|K|
    boundary_epsilon: f32, // Numerical stability (1e-5)
}

impl HyperbolicMetric {
    /// Fast arcosh using Horner's method for small x
    #[inline]
    fn fast_arcosh(&self, x: f32) -> f32 {
        if x < 1.5 {
            // Taylor series: arcosh(1+y) ≈ √(2y)(1 - y/12 + 3y²/160)
            let y = x - 1.0;
            let sqrt_2y = (2.0 * y).sqrt();
            sqrt_2y * (1.0 - y / 12.0 + 3.0 * y * y / 160.0)
        } else {
            // Standard formula: arcosh(x) = ln(x + √(x²-1))
            (x + (x * x - 1.0).sqrt()).ln()
        }
    }
    
    /// SIMD-optimized squared norm
    #[inline]
    fn squared_norm_simd(&self, v: &[f32]) -> f32 {
        // Process 8 elements at a time with f32x8
        // ... SIMD implementation
    }
}
```

### 5.2 Zero-Allocation LSH Hash Families

#### SimHash (Cosine Similarity)

```rust
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct SimHashSignature {
    bits: [u64; 4],  // 256 bits max, fits in cache line
    len: u8,
}

impl SimHashSignature {
    #[inline]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        (self.bits[0] ^ other.bits[0]).count_ones()
            + (self.bits[1] ^ other.bits[1]).count_ones()
            + (self.bits[2] ^ other.bits[2]).count_ones()
            + (self.bits[3] ^ other.bits[3]).count_ones()
    }
}
```

#### MinHash (Jaccard Similarity for Whale Detection)

```rust
pub struct MinHash {
    seeds: ArrayVec<u64, 256>,  // No heap allocation
    num_hashes: usize,
}

pub struct MinHashSignature {
    values: ArrayVec<u64, 256>,  // Fixed capacity
}

impl MinHashSignature {
    pub fn jaccard_estimate(&self, other: &Self) -> f32 {
        let matches = self.values.iter()
            .zip(other.values.iter())
            .filter(|(&x, &y)| x == y)
            .count();
        matches as f32 / self.values.len() as f32
    }
}
```

### 5.3 Streaming LSH Index

```rust
pub struct StreamingLshIndex {
    config: LshConfig,
    hasher: SimHash,
    tables: Vec<HashTable>,
    items: RwLock<Vec<StoredItem>>,
    next_id: AtomicU64,
    
    // Lock-free streaming buffer
    stream_buffer: Option<ArrayQueue<Vec<f32>>>,
    
    stats: LshStats,
}

impl StreamingLshIndex {
    /// O(1) streaming insert (lock-free, may drop under backpressure)
    pub fn stream_insert(&self, vector: Vec<f32>) -> Result<()> {
        if let Some(buffer) = &self.stream_buffer {
            buffer.push(vector).map_err(|_| LshError::BufferFull { 
                drop_rate: self.calculate_drop_rate() 
            })
        } else {
            self.insert(vector).map(|_| ())
        }
    }
    
    /// Background processing of buffered items
    pub fn process_stream_buffer(&self) -> usize {
        let mut processed = 0;
        while let Some(vector) = self.stream_buffer.as_ref()?.pop() {
            if self.insert(vector).is_ok() {
                processed += 1;
            }
        }
        processed
    }
}
```

### 5.4 Hybrid Index Router

```rust
pub enum SearchMode {
    Hot,       // Direct HNSW, <1μs
    Streaming, // LSH ingestion only
    Whale,     // MinHash set similarity
    Hybrid,    // LSH filter → HNSW refine
    Auto,      // Router decides
}

pub struct HybridIndex {
    config: SearchConfig,
    hnsw: RwLock<Option<HotIndex<HyperbolicMetric>>>,
    lsh: RwLock<Option<StreamingLshIndex>>,
    stats: RouterStats,
}

impl HybridIndex {
    pub fn search(&self, query: &[f32], k: usize, mode: SearchMode) 
        -> Result<Vec<SearchResult>> 
    {
        let effective_mode = match mode {
            SearchMode::Auto => self.auto_select_mode(query),
            other => other,
        };
        
        match effective_mode {
            SearchMode::Hot => self.search_hot(query, k),
            SearchMode::Hybrid => self.search_hybrid(query, k),
            SearchMode::Whale => Err(HybridError::Router { 
                reason: "Use search_whale() for set queries".into() 
            }),
            _ => unreachable!(),
        }
    }
    
    /// Acquisition → Processing pattern promotion
    pub fn promote_patterns(&self, count: usize) -> Result<usize> {
        let lsh = self.lsh.read();
        let mut hnsw = self.hnsw.write();
        
        // Get high-collision patterns from LSH
        // Insert into HNSW hot index
        // Return count of successfully promoted
    }
}
```

### 5.5 Triangular Architecture Traits

```rust
/// Acquisition constrains Processing
pub trait AcquisitionConstraint {
    type Pattern;
    
    fn should_promote(&self, pattern: &Self::Pattern) -> bool;
    fn promotion_threshold(&self) -> f32;
    fn ingestion_rate(&self) -> f64;
}

/// Processing receives from Acquisition
pub trait ProcessingReceiver {
    type Pattern;
    type PatternId;
    
    fn receive_promoted(&mut self, patterns: &[Self::Pattern]) 
        -> Result<Vec<Self::PatternId>>;
    fn useful_patterns(&self) -> &[Self::PatternId];
}

/// Evolution receives from Acquisition
pub trait EvolutionSource {
    type Pattern;
    type Signature;
    
    fn high_collision_patterns(&self, threshold: f32) -> Vec<Self::Pattern>;
    fn boltzmann_sample(&self, temperature: f32, count: usize) -> Vec<Self::Pattern>;
}

/// Processing reports to Evolution
pub trait ProcessingToEvolution {
    fn report_latency(&mut self, latency_ns: u64);
    fn report_recall(&mut self, recall: f32);
    fn apply_evolution_params(&mut self, ef_search: usize) -> Result<()>;
}
```

---

## Part VI: Configuration Presets

### 6.1 Trading Configuration

```rust
impl SearchConfig {
    pub fn trading() -> Self {
        Self {
            hnsw: IndexConfig {
                m: Some(8),
                m0: Some(16),
                ef_construction: Some(100),
                ef_search: Some(32),
                latency_budget_ns: Some(500),
                ..Default::default()
            },
            lsh: LshConfig::simhash(128, 64).with_tables(8),
            router: RouterConfig {
                auto_route: true,
                lsh_threshold: 0.2,
                lsh_candidate_limit: 500,
                query_timeout_us: 5_000,
                ..Default::default()
            },
            evolution: EvolutionConfig {
                target_latency_ns: 500,
                target_recall: 0.9,
                ..Default::default()
            },
        }
    }
}
```

### 6.2 Research Configuration

```rust
impl SearchConfig {
    pub fn research() -> Self {
        Self {
            hnsw: IndexConfig {
                m: Some(32),
                m0: Some(64),
                ef_construction: Some(400),
                ef_search: Some(128),
                latency_budget_ns: Some(5_000),
                ..Default::default()
            },
            lsh: LshConfig::simhash(256, 128).with_tables(16),
            router: RouterConfig {
                lsh_candidate_limit: 5000,
                parallel_query: true,
                query_timeout_us: 100_000,
                ..Default::default()
            },
            evolution: EvolutionConfig {
                target_latency_ns: 10_000,
                target_recall: 0.99,
                ..Default::default()
            },
        }
    }
}
```

### 6.3 Whale Detection Configuration

```rust
impl SearchConfig {
    pub fn whale_detection() -> Self {
        Self {
            hnsw: IndexConfig::default(),
            lsh: LshConfig::minhash(256).with_tables(16),
            router: RouterConfig {
                auto_route: false,  // Always use LSH
                lsh_threshold: 0.0,
                lsh_candidate_limit: 10_000,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}
```

---

## Part VII: Implementation Timeline

### Phase 1: Core HNSW (Week 1-2)

- [ ] USearch integration with HotIndex wrapper
- [ ] HyperbolicMetric with fast arcosh
- [ ] <1μs latency validation harness
- [ ] Basic tests and benchmarks

### Phase 2: LSH Hash Families (Week 3)

- [ ] SimHash with SIMD optimization
- [ ] MinHash for whale detection
- [ ] SRP (Signed Random Projections)
- [ ] Zero-allocation signature types

### Phase 3: StreamingLshIndex (Week 4)

- [ ] Lock-free ArrayQueue buffer
- [ ] O(1) insertion path
- [ ] Background buffer processing
- [ ] Backpressure handling

### Phase 4: pBit Integration (Week 5)

- [ ] Temperature-controlled sampling
- [ ] Hash collision → activation mapping
- [ ] FPGA interface stubs
- [ ] Evolution feedback loop

### Phase 5: Integration (Week 6)

- [ ] HybridIndex router
- [ ] Pattern promotion pipeline
- [ ] End-to-end benchmarks
- [ ] Documentation

---

## Part VIII: Validation Criteria

### 8.1 Performance Validation

```rust
#[test]
fn validate_hot_path_latency() {
    let index = HotIndex::new(config, HyperbolicMetric::new(128))?;
    
    // Insert 100K vectors
    for v in test_vectors(100_000, 128) {
        index.insert(&v)?;
    }
    
    // Measure query latency
    let query = random_vector(128);
    let start = Instant::now();
    let _ = index.search(&query, 10)?;
    let elapsed_ns = start.elapsed().as_nanos();
    
    assert!(elapsed_ns < 1_000, "Query latency {}ns > 1μs", elapsed_ns);
}
```

### 8.2 Triangular Constraint Validation

```rust
#[test]
fn validate_triangular_constraints() {
    let hybrid = HybridIndex::new(SearchConfig::trading())?;
    
    // Acquisition constrains Processing
    let promotion_rate = hybrid.lsh().ingestion_rate();
    assert!(promotion_rate > 0.0, "Acquisition must provide patterns");
    
    // Processing constrains Evolution
    let metrics = hybrid.metrics();
    assert!(metrics.avg_latency_ns < 1_000.0, "Processing must meet latency");
    
    // Evolution constrains Acquisition
    let temperature = hybrid.evolution_temperature();
    assert!(temperature > 0.0, "Evolution must guide sampling");
}
```

### 8.3 Zero-Allocation Validation

```rust
#[test]
fn validate_zero_allocation_hash() {
    let hasher = SimHash::new(128, 64, 42);
    let vector = vec![1.0f32; 128];
    
    // Use allocator tracking
    let allocs_before = GLOBAL_ALLOCATOR.allocations();
    let _ = hasher.hash(&vector);
    let allocs_after = GLOBAL_ALLOCATOR.allocations();
    
    assert_eq!(allocs_before, allocs_after, "Hash must not allocate");
}
```

---

## Part IX: Research References

### Primary Sources

1. Malkov, Y.A. & Yashunin, D.A. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." *IEEE TPAMI*.

2. Christiansen, M.H. & Chater, N. (2016). "Creating Language: Integrating Evolution, Acquisition, and Processing." *MIT Press*.

3. Johnson, W.B. & Lindenstrauss, J. (1984). "Extensions of Lipschitz mappings into a Hilbert space." *Contemporary Mathematics*, 26, 189-206.

4. Indyk, P. & Motwani, R. (1998). "Approximate nearest neighbors: towards removing the curse of dimensionality." *STOC '98*.

5. Broder, A.Z. (1997). "On the resemblance and containment of documents." *SEQUENCES '97*.

### Benchmarks & Tools

6. ANN-Benchmarks: http://ann-benchmarks.com/
7. Big-ANN Benchmarks: https://big-ann-benchmarks.com/
8. USearch: https://github.com/unum-cloud/usearch
9. hnswlib: https://github.com/nmslib/hnswlib
10. NVIDIA CAGRA: https://docs.rapids.ai/api/cugraph/stable/

---

## Appendix A: Error Taxonomy

```rust
pub enum HnswError {
    DimensionMismatch { expected: usize, actual: usize },
    CapacityExceeded { max: usize, attempted: usize },
    InvalidVector { reason: String },
    LatencyExceeded { actual_ns: u64, budget_ns: u64 },
    RebuildRequired { parameter: String },
    // ...
}

pub enum LshError {
    DimensionMismatch { expected: usize, actual: usize },
    BucketOverflow { table_id: usize, bucket_id: u64 },
    BufferFull { drop_rate: f32 },
    Backpressure { current: usize, max: usize },
    // ...
}

pub enum HybridError {
    Hnsw(HnswError),
    Lsh(LshError),
    Router { reason: String },
    Timeout { elapsed_us: u64, limit_us: u64 },
    NotInitialized { component: String },
    PromotionFailed { reason: String },
    // ...
}
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **HNSW** | Hierarchical Navigable Small World - graph-based ANN algorithm |
| **LSH** | Locality Sensitive Hashing - hash-based ANN algorithm |
| **MinHash** | LSH variant for Jaccard similarity on sets |
| **SimHash** | LSH variant for cosine similarity on vectors |
| **SRP** | Signed Random Projections - sparse SimHash variant |
| **pBit** | Probabilistic bit - thermodynamic memory element |
| **Poincaré Ball** | Model of hyperbolic space with conformal boundary |
| **arcosh** | Inverse hyperbolic cosine, used in hyperbolic distance |
| **ef_search** | HNSW expansion factor during search (recall/speed tradeoff) |
| **M** | HNSW maximum connections per node |

---

*Document generated as part of HyperPhysics Ultra-HFT System development*
