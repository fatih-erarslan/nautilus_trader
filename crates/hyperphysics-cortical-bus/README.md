# HyperPhysics Cortical Bus

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

Ultra-low-latency software cortical bus with hardware abstraction layer for neuromorphic trading systems.

## Features

- **~50ns spike injection** via lock-free queues
- **~100ns LSH lookup** for content-addressable memory
- **~20ns ring buffer operations** (wait-free SPSC)
- **Hardware abstraction** for future FPGA/Photonic/SFQ migration
- **SIMD optimized** (AVX-512, AVX2, NEON)
- **Zero-copy design** with cache-line aligned structures

## Quick Start

```rust
use hyperphysics_cortical_bus::prelude::*;

// Create backend
let config = BackendConfig::default();
let bus = create_backend(&config)?;

// Inject spike
let spike = Spike::new(12345, 100, 50, 0xAB);
bus.inject_spike(spike)?;

// Poll spikes
let mut buffer = [Spike::default(); 100];
let count = bus.poll_spikes(&mut buffer)?;
```

## Architecture

```
Application → HAL (CorticalBus trait) → Backend (CPU-SIMD/GPU/Future)
                                     → Ring Buffers (SPSC/MPSC)
                                     → LSH Tables (CAM)
                                     → pBit Arrays (Ising dynamics)
```

## Performance Targets

| Operation | CPU-SIMD | GPU | Future: SFQ |
|-----------|----------|-----|-------------|
| Spike inject | ~50ns | N/A | ~10ps |
| Batch (1K) | ~5µs | ~50µs | ~1ns |
| LSH lookup | ~100ns | ~500ns | ~100ps |
| pBit sweep (64K) | ~100µs | ~50µs | ~1ns |

## Building

```bash
# Default (CPU-SIMD only)
cargo build --release

# With GPU support
cargo build --release --features gpu

# All features
cargo build --release --features full
```

## Testing

```bash
cargo test
cargo bench
```

## Documentation

```bash
cargo doc --open
```

See [SCAFFOLDING_GUIDE.md](SCAFFOLDING_GUIDE.md) for enterprise architecture details.

## License

MIT OR Apache-2.0
