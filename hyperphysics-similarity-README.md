# HyperPhysics Similarity Search Architecture
# =============================================
# 
# A dual-crate architecture inspired by the triangular constraint model from
# Christiansen & Chater's "Creating Language" (Figure 1.6), where three processes
# operate at different timescales but mutually constrain each other:
#
#   ┌─────────────────────────────────────────────────────────────────┐
#   │                      ACQUISITION                                 │
#   │                    (LSH Streaming)                               │
#   │              "Pattern Ingestion Layer"                           │
#   │                          │                                       │
#   │    Acquisition           │          Processing constrains        │
#   │    constrains what       │          what can be acquired         │
#   │    can evolve            │                                       │
#   │         │                │                    │                  │
#   │         ▼                ▼                    ▼                  │
#   │    ┌─────────┐    fits what is          ┌──────────┐            │
#   │    │EVOLUTION│◄──learned to processing──►│PROCESSING│            │
#   │    │  (LSH   │     mechanism             │  (HNSW)  │            │
#   │    │ pBit/HW)│                           │          │            │
#   │    └─────────┘                           └──────────┘            │
#   │         │                                      │                 │
#   │         └──────fits language to processing─────┘                 │
#   │                     mechanism                                    │
#   │                          │                                       │
#   │              Processing constrains what can evolve               │
#   └─────────────────────────────────────────────────────────────────┘
#
# This architecture ensures that:
# - Processing (HNSW) provides sub-microsecond queries for the trading hot path
# - Acquisition (LSH) enables O(1) streaming ingestion without blocking queries
# - Evolution (LSH pBit/Hardware) allows thermodynamic optimization over time
#
# Each component constrains and is constrained by the others, creating a
# self-consistent system that adapts across multiple timescales.

[workspace]
resolver = "2"
members = [
    "crates/hyperphysics-hnsw",
    "crates/hyperphysics-lsh",
    "crates/hyperphysics-similarity",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
authors = ["HyperPhysics Team"]
license = "MIT OR Apache-2.0"
repository = "https://github.com/hyperphysics/similarity"

[workspace.dependencies]
# Shared dependencies across crates
thiserror = "1.0"
tracing = "0.1"
parking_lot = "0.12"
crossbeam = "0.8"

# Serialization
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"

# Testing and benchmarking
criterion = "0.5"
proptest = "1.4"

[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true

[profile.bench]
lto = "thin"
debug = true

# HyperPhysics Similarity Search

Unified similarity search system combining HNSW (Processing) and LSH (Acquisition) following the triangular constraint architecture inspired by Christiansen & Chater's "Creating Language" framework.

## Architecture

```
                          ACQUISITION (LSH)
                        ╱ Streaming Ingestion ╲
                       ╱   O(1) Insertion      ╲
                      ╱  Whale Detection        ╲
                     ╱                           ╲
                    ╱ promotes        constrains  ╲
                   ╱  patterns        evolution    ╲
                  ╱                                 ╲
  EVOLUTION ◄──────────────────────────────────────► PROCESSING
  (pBit/FPGA)        tunes parameters              (HNSW Hot)
  Thermodynamic         ◄──────►                   Sub-μs Queries
  Optimization       provides feedback             
```

## Performance Targets

| Operation | Target | Layer |
|-----------|--------|-------|
| HNSW Query | <1μs | Processing |
| LSH Hash | <100ns | Acquisition |
| Streaming Insert | <200ns | Acquisition |
| Pattern Promotion | <50μs | Acquisition→Processing |

## Crates

### `hyperphysics-hnsw`
HNSW implementation wrapping USearch with HyperPhysics optimizations:
- **HyperbolicMetric**: Poincaré ball model for consciousness space
- **HotIndex**: Sub-microsecond query interface
- Triangular trait implementations for Evolution/Acquisition integration

### `hyperphysics-lsh`
Zero-allocation LSH for specialized use cases:
- **SimHash**: Cosine similarity on dense vectors
- **MinHash**: Jaccard similarity for whale detection
- **SRP**: Signed Random Projections
- **StreamingLshIndex**: O(1) insertion, lock-free ingestion

### `hyperphysics-similarity`
Integration crate with unified interface:
- **HybridIndex**: Combined HNSW + LSH management
- **SearchRouter**: Automatic mode selection
- Configuration presets for trading, research, whale detection

## Usage

```rust
use hyperphysics_similarity::{HybridIndex, SearchConfig, SearchMode};

// Create hybrid index for trading
let config = SearchConfig::trading();
let index = HybridIndex::new(config)?;

// Initialize components
index.init_hnsw(128)?;  // 128-dimensional vectors
index.init_lsh()?;

// Stream market data through Acquisition layer
for tick in market_stream {
    index.stream_ingest(tick.to_vector())?;
}

// Promote important patterns to Processing layer
index.promote_patterns(1000)?;

// Hot path query for sub-μs latency
let results = index.search(&query, 10, SearchMode::Hot)?;

// Hybrid query: LSH filter → HNSW refinement
let results = index.search(&query, 10, SearchMode::Hybrid)?;
```

## Build

```bash
# Check all crates
cargo check --workspace

# Run tests
cargo test --workspace

# Build release
cargo build --release --workspace

# Run benchmarks
cargo bench --workspace
```

## Feature Flags

### hyperphysics-hnsw
- `simd` (default): Enable SIMD optimizations
- `avx2`: AVX2 instructions
- `avx512`: AVX-512 instructions  
- `neon`: ARM NEON instructions
- `gpu`: ROCm/CUDA acceleration (future)

### hyperphysics-lsh
- `simd` (default): Enable SIMD optimizations
- `pbit`: pBit thermodynamic memory integration
- `fpga`: FPGA interface stubs

## Key Design Decisions

1. **No pure LSH-HNSW hybrid** - complexity doesn't justify marginal gains
2. **Three-tier architecture** - Hot (USearch HNSW), Warm (CAGRA GPU future), Cold (DiskANN future)
3. **Custom hyperbolic metric** - Poincaré ball model for HyperPhysics consciousness space
4. **Zero-allocation LSH** - ArrayVec, Bitmap, pre-allocated buffers
5. **Triangular constraint model** - ensures self-consistent system across timescales

## Research References

1. Malkov & Yashunin (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
2. Christiansen & Chater (2016). "Creating Language: Integrating Evolution, Acquisition, and Processing"
3. Johnson & Lindenstrauss (1984). "Extensions of Lipschitz mappings into a Hilbert space"
4. Indyk & Motwani (1998). "Approximate nearest neighbors: towards removing the curse of dimensionality"

## License

MIT OR Apache-2.0
