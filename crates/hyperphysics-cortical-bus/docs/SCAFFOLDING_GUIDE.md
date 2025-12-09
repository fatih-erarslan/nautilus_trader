# HyperPhysics Software Cortical Bus
## Enterprise Architecture & Agent Scaffolding Guide

**Version:** 0.1.0  
**Status:** Production Blueprint  
**Classification:** TENGRI-Compliant Implementation

---

## Executive Summary

This document provides complete enterprise-grade instructions for scaffolding, implementing, and extending the HyperPhysics Software Cortical Bus. The system achieves ultra-low-latency neuromorphic computing using pure software (CPU-SIMD + GPU) with a clean Hardware Abstraction Layer (HAL) enabling seamless migration to future hardware (FPGA, Photonic, SFQ).

### Target Performance

| Operation | CPU-SIMD (AVX-512) | GPU (wgpu) | Future: SFQ |
|-----------|-------------------|------------|-------------|
| Single spike inject | **~50 ns** | N/A | ~10 ps |
| Batch inject (1K spikes) | **~5 µs** | ~50 µs | ~1 ns |
| LSH similarity lookup | **~100 ns** | ~500 ns | ~100 ps |
| pBit Metropolis sweep (64K) | **~100 µs** | ~50 µs | ~1 ns |
| SPSC ring buffer push/pop | **~20 ns** | N/A | ~5 ps |

---

## Part I: Architecture Overview

### 1.1 System Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                                │
│  HyperPhysics Trading Engine • Whale Detection • Strategy Execution      │
├─────────────────────────────────────────────────────────────────────────┤
│                    HAL INTERFACE (CorticalBus trait)                     │
│  inject_spike() • inject_batch() • poll_spikes() • similarity_lookup()  │
│  update_pbit_states() • get_latency_stats() • capabilities()            │
├─────────────────────────────────────────────────────────────────────────┤
│                      BACKEND SELECTION (Runtime)                         │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐        │
│  │  CpuSimdBackend  │ │   GpuBackend     │ │  [Future FPGA]   │        │
│  │  AVX-512/AVX2    │ │  wgpu (Metal/    │ │  [Future Photo]  │        │
│  │  NEON            │ │  Vulkan/DX12)    │ │  [Future SFQ]    │        │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘        │
├─────────────────────────────────────────────────────────────────────────┤
│                       COMMUNICATION LAYER                                │
│  SpscRingBuffer (wait-free) • MpscRingBuffer (lock-free CAS)            │
├─────────────────────────────────────────────────────────────────────────┤
│                          MEMORY LAYER                                    │
│  LSH Tables (CAM) • pBit State Arrays • Coupling Matrices (CSR)         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Directory Structure

```
hyperphysics-cortical-bus/
├── Cargo.toml                    # Crate manifest with features
├── src/
│   ├── lib.rs                    # Crate root, re-exports
│   ├── hal/                      # Hardware Abstraction Layer
│   │   ├── mod.rs                # HAL traits (CorticalBus, PBitArray, CAM)
│   │   ├── spike.rs              # Spike struct (8 bytes, cache-aligned)
│   │   ├── latency.rs            # LatencyTracker with RDTSC
│   │   └── error.rs              # BusError enum
│   ├── backend/                  # Backend implementations
│   │   ├── mod.rs                # Backend factory, SIMD detection
│   │   ├── cpu_simd/             # Primary CPU backend
│   │   │   ├── mod.rs            # CpuSimdCorticalBus
│   │   │   └── simd_ops.rs       # AVX-512/AVX2 primitives
│   │   ├── gpu/                  # GPU backend (optional)
│   │   │   └── mod.rs            # wgpu-based implementation
│   │   └── [fpga|photonic|sfq]/  # Future backends (stubs)
│   ├── ringbuf/                  # Lock-free ring buffers
│   │   ├── mod.rs                # Module exports
│   │   ├── spsc.rs               # Single-producer single-consumer
│   │   └── mpsc.rs               # Multi-producer single-consumer
│   ├── lsh/                      # Locality-Sensitive Hashing
│   │   └── mod.rs                # LshTables implementation
│   └── pbit/                     # Probabilistic bits
│       ├── mod.rs                # Module exports
│       ├── array.rs              # PBitArrayImpl
│       ├── coupling.rs           # CouplingMatrix (CSR)
│       └── dynamics.rs           # Metropolis, Gillespie SSA
├── benches/                      # Criterion benchmarks
│   ├── spike_inject.rs
│   ├── lsh_query.rs
│   ├── metropolis.rs
│   └── ringbuf.rs
└── tests/                        # Integration tests
    └── integration.rs
```

---

## Part II: Core Components

### 2.1 Hardware Abstraction Layer (HAL)

The HAL defines traits that ALL backends must implement, enabling runtime backend selection and future hardware migration.

#### CorticalBus Trait

```rust
pub trait CorticalBus: Send + Sync {
    type Error: std::error::Error;
    
    // Core operations
    fn inject_spike(&self, spike: Spike) -> Result<(), Self::Error>;
    fn inject_batch(&self, spikes: &[Spike]) -> Result<(), Self::Error>;
    fn poll_spikes(&self, buffer: &mut [Spike]) -> Result<usize, Self::Error>;
    fn similarity_lookup(&self, query: &[f32], k: usize, results: &mut [(u32, f32)]) 
        -> Result<usize, Self::Error>;
    fn update_pbit_states(&self, couplings: &CouplingMatrix, temperature: f32) 
        -> Result<(), Self::Error>;
    
    // Monitoring
    fn get_latency_stats(&self) -> LatencyStats;
    fn reset_latency_stats(&self);
    fn is_healthy(&self) -> bool;
    fn capabilities(&self) -> BackendCapabilities;
    
    // Identification
    fn backend_name(&self) -> &'static str;
    fn backend_version(&self) -> (u32, u32, u32);
}
```

#### Spike Struct (8 bytes, cache-optimized)

```rust
#[repr(C, align(8))]
pub struct Spike {
    pub source_id: u32,      // Neuron/pBit ID (4B neurons)
    pub timestamp: u16,      // Relative time (wraps at 65535)
    pub strength: i8,        // Excitatory (+) / Inhibitory (-)
    pub routing_hint: u8,    // Destination queue partition
}
```

### 2.2 Ring Buffers

#### SPSC (Wait-Free)

- **Performance:** ~20ns push/pop
- **Use case:** Single producer to single consumer (hot path)
- **Guarantee:** Wait-free for both producer and consumer

```rust
let buf: SpscRingBuffer<Spike, 4096> = SpscRingBuffer::new();

// Producer thread
buf.push(spike);

// Consumer thread
if let Some(spike) = buf.pop() { /* process */ }
```

#### MPSC (Lock-Free)

- **Performance:** ~50ns push (with contention)
- **Use case:** Multiple producers aggregating to single consumer
- **Guarantee:** Lock-free (CAS-based)

```rust
let buf: Arc<MpscRingBuffer<Spike, 4096>> = Arc::new(MpscRingBuffer::new());

// Multiple producer threads
buf.push(spike);

// Single consumer thread
while let Some(spike) = buf.pop() { /* process */ }
```

### 2.3 LSH Tables (Content-Addressable Memory)

- **Algorithm:** Random hyperplane LSH (SimHash) for cosine similarity
- **Complexity:** O(1) approximate lookup, O(k) verification
- **Target:** ~100ns query latency

```rust
let config = LshConfig {
    num_tables: 8,      // More tables = higher recall
    num_hashes: 16,     // More hashes = higher precision
    dim: 128,           // Pattern dimension
};
let lsh = LshTables::new(config);

// Store pattern
lsh.store(pattern_id, &pattern_vec)?;

// Query similar patterns
let mut results = [(0u32, 0.0f32); 10];
let count = lsh.query(&query_vec, 5, &mut results)?;
```

### 2.4 pBit Arrays

Probabilistic bits with Ising-model dynamics for neuromorphic computing.

```rust
let pbits = PBitArrayImpl::new(65536);
let couplings = CouplingMatrix::lattice_2d(256, 256, 1.0);

// Randomize states
pbits.randomize(&rng);

// Metropolis sweep
let sweep = MetropolisSweep::new(temperature);
let flips = sweep.sweep(&pbits, &couplings);

// Get energy
let energy = pbits.compute_energy(&couplings);
```

---

## Part III: Agent Scaffolding Instructions

### 3.1 Phase 1: Repository Setup

```bash
# Clone or initialize repository
git init hyperphysics-cortical-bus
cd hyperphysics-cortical-bus

# Create directory structure
mkdir -p src/{hal,backend/{cpu_simd,gpu},ringbuf,lsh,pbit}
mkdir -p benches tests

# Copy Cargo.toml from blueprint
# Copy all source files from blueprint
```

### 3.2 Phase 2: Dependency Installation

Ensure all dependencies are available:

```toml
[dependencies]
crossbeam = "0.8"
parking_lot = "0.12"
wide = "0.7"
smallvec = { version = "1.13", features = ["const_generics"] }
hashbrown = { version = "0.14", features = ["inline-more"] }
bytemuck = { version = "1.14", features = ["derive", "min_const_generics"] }
tracing = "0.1"
thiserror = "1.0"
fastrand = "2.0"
portable-atomic = { version = "1.6", features = ["float"] }

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1.4"
rand = "0.8"
```

### 3.3 Phase 3: Implementation Order

Agents MUST implement in this order to satisfy dependencies:

1. **hal/error.rs** - BusError enum (no dependencies)
2. **hal/spike.rs** - Spike struct (depends on bytemuck)
3. **hal/latency.rs** - LatencyTracker (depends on portable-atomic)
4. **hal/mod.rs** - HAL traits (depends on above)
5. **ringbuf/spsc.rs** - SPSC buffer (depends on portable-atomic)
6. **ringbuf/mpsc.rs** - MPSC buffer (depends on portable-atomic)
7. **ringbuf/mod.rs** - Module exports
8. **pbit/coupling.rs** - CouplingMatrix (depends on bytemuck, fastrand)
9. **pbit/array.rs** - PBitArrayImpl (depends on hal, coupling)
10. **pbit/dynamics.rs** - Metropolis/Gillespie (depends on array, coupling)
11. **pbit/mod.rs** - Module exports
12. **lsh/mod.rs** - LshTables (depends on hal, hashbrown, smallvec)
13. **backend/cpu_simd/simd_ops.rs** - SIMD primitives
14. **backend/cpu_simd/mod.rs** - CpuSimdCorticalBus (depends on all above)
15. **backend/mod.rs** - Backend factory
16. **lib.rs** - Crate root

### 3.4 Phase 4: Testing Requirements

Each component MUST have:

1. **Unit tests** in the same file (`#[cfg(test)] mod tests`)
2. **Property tests** using proptest for invariants
3. **Benchmarks** using criterion for latency validation

```bash
# Run all tests
cargo test

# Run benchmarks
cargo bench

# Check specific latency targets
cargo bench -- spike_inject
cargo bench -- lsh_query
cargo bench -- ringbuf
```

### 3.5 Phase 5: Validation Checklist

Before marking implementation complete:

- [ ] All tests pass (`cargo test`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Documentation complete (`cargo doc`)
- [ ] Benchmarks meet targets
- [ ] No mock data (TENGRI compliance)
- [ ] Memory safety verified (no unsafe without justification)
- [ ] Thread safety verified (Send + Sync where needed)

---

## Part IV: Extending the System

### 4.1 Adding a New Backend

To add a new backend (e.g., FPGA):

1. Create `src/backend/fpga/mod.rs`
2. Implement `CorticalBus` trait
3. Add feature flag to `Cargo.toml`
4. Register in `backend/mod.rs` factory

```rust
// src/backend/fpga/mod.rs
pub struct FpgaCorticalBus {
    // FPGA-specific state
}

impl CorticalBus for FpgaCorticalBus {
    type Error = BusError;
    
    fn inject_spike(&self, spike: Spike) -> Result<(), BusError> {
        // FPGA-specific implementation
    }
    // ... implement all trait methods
}
```

### 4.2 Adding New LSH Hash Functions

To add new hash functions (e.g., WTAHash):

1. Add hash function to `lsh/mod.rs`
2. Make hash function configurable in `LshConfig`
3. Benchmark against existing SimHash

### 4.3 Adding GPU Kernels

For GPU backend (`gpu` feature):

1. Create WGSL shaders in `src/backend/gpu/shaders/`
2. Implement wgpu pipeline setup
3. Handle CPU-GPU memory transfers efficiently
4. Use double-buffering for async operations

---

## Part V: Performance Optimization Guide

### 5.1 SIMD Optimization

Always check for SIMD at runtime:

```rust
#[cfg(target_arch = "x86_64")]
{
    if std::arch::is_x86_feature_detected!("avx512f") {
        // Use AVX-512 path
    } else if std::arch::is_x86_feature_detected!("avx2") {
        // Use AVX2 path
    }
}
```

### 5.2 Memory Alignment

- All hot-path structures: `#[repr(align(64))]` (cache line)
- Spike arrays: `#[repr(align(8))]` (8 spikes per cache line)
- SIMD vectors: `#[repr(align(32))]` for AVX2, `#[repr(align(64))]` for AVX-512

### 5.3 Lock-Free Patterns

- Use `Ordering::Relaxed` for statistics (non-critical)
- Use `Ordering::Acquire/Release` for data dependencies
- Use `Ordering::SeqCst` sparingly (only for total ordering)

### 5.4 Latency Measurement

Always use RDTSC for sub-microsecond measurement:

```rust
let start = LatencyTracker::rdtsc();
// ... operation ...
let cycles = LatencyTracker::rdtsc() - start;
// Convert: ns ≈ cycles / (CPU_GHz)
```

---

## Part VI: Integration with HyperPhysics

### 6.1 Trading Engine Integration

```rust
use hyperphysics_cortical_bus::prelude::*;

struct TradingEngine {
    cortical_bus: Box<dyn CorticalBus<Error = BusError>>,
    pbits: PBitArrayImpl,
    couplings: CouplingMatrix,
    lsh: LshTables,
}

impl TradingEngine {
    pub fn process_market_data(&self, data: &MarketData) {
        // Convert to spike
        let spike = Spike::new(
            data.symbol_id,
            data.timestamp as u16,
            (data.price_change * 127.0) as i8,
            data.routing_hint(),
        );
        
        // Inject into cortical bus
        self.cortical_bus.inject_spike(spike).unwrap();
        
        // Pattern recognition via LSH
        let pattern = self.encode_pattern(data);
        let mut results = [(0u32, 0.0f32); 10];
        let count = self.cortical_bus.similarity_lookup(&pattern, 5, &mut results).unwrap();
        
        // Update pBit decision layer
        self.cortical_bus.update_pbit_states(&self.couplings, 1.0).unwrap();
    }
}
```

### 6.2 Whale Detection Integration

```rust
impl WhaleDetector {
    pub fn detect(&self, transactions: &[Transaction]) -> Vec<WhaleSignal> {
        // Batch inject for efficiency
        let spikes: Vec<Spike> = transactions
            .iter()
            .filter(|tx| tx.value > self.whale_threshold)
            .map(|tx| tx.to_spike())
            .collect();
        
        self.cortical_bus.inject_batch(&spikes).unwrap();
        
        // Poll and process
        let mut buffer = [Spike::default(); 1000];
        let count = self.cortical_bus.poll_spikes(&mut buffer).unwrap();
        
        self.analyze_spike_patterns(&buffer[..count])
    }
}
```

---

## Part VII: Future Hardware Migration Path

### 7.1 FPGA Backend

When FPGA hardware is available:

1. Implement PCIe communication layer
2. Map spike queues to FPGA memory
3. Implement LSH in FPGA fabric
4. Target: <10ns spike latency

### 7.2 Photonic Interconnect

When photonic hardware is available:

1. Implement optical transceiver interface
2. Map spikes to photonic pulses
3. Implement all-optical LSH
4. Target: <100ps spike latency

### 7.3 Superconducting SFQ

When SFQ hardware is available:

1. Implement cryogenic interface
2. Map pBits to SFQ RSFQ gates
3. Implement superconducting LSH
4. Target: <10ps spike latency

---

## Appendix A: TENGRI Compliance Checklist

- [x] No mock data generation
- [x] No hardcoded values (all configurable)
- [x] Full implementations (no stubs in production paths)
- [x] Real SIMD feature detection
- [x] Real latency measurement (RDTSC)
- [x] Real random number generation (fastrand)
- [x] Agent handoff validation ready
- [x] Constitutional rules embedded in tests

---

## Appendix B: File Checksums

For integrity verification:

```
src/hal/mod.rs          - CorticalBus, ContentAddressableMemory, PBitArray traits
src/hal/spike.rs        - Spike struct (8 bytes)
src/hal/latency.rs      - LatencyTracker with RDTSC
src/hal/error.rs        - BusError enum
src/ringbuf/spsc.rs     - Wait-free SPSC ring buffer
src/ringbuf/mpsc.rs     - Lock-free MPSC ring buffer
src/lsh/mod.rs          - LshTables with SimHash
src/pbit/array.rs       - PBitArrayImpl
src/pbit/coupling.rs    - CouplingMatrix (CSR format)
src/pbit/dynamics.rs    - MetropolisSweep, GillespieSSA
src/backend/cpu_simd/*  - CPU-SIMD backend
```

---

## Appendix C: Contact & Support

For questions about this implementation:

1. Review this document completely
2. Check existing tests for usage examples
3. Run benchmarks to validate performance
4. Consult the HyperPhysics architecture documents

**This is a living document. Update as the system evolves.**

---

*Generated for HyperPhysics Project - TENGRI Framework Compliant*
