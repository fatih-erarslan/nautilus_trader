//! # HyperPhysics Cortical Bus
//!
//! Ultra-low-latency neuromorphic bus integrating the HyperPhysics ecosystem:
//!
//! - **pBit Dynamics** via `hyperphysics-pbit` (Metropolis, Gillespie, SIMD)
//! - **Similarity Search** via `hyperphysics-similarity` (HNSW + LSH hybrid)
//! - **GPU Acceleration** via `hyperphysics-gpu-unified` (dual-GPU orchestration)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                          CORTICAL BUS                                        │
//! │                                                                              │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
//! │  │   Spike Router  │  │   Pattern CAM   │  │     pBit Fabric             │  │
//! │  │   (Ring Buffers)│  │   (HNSW + LSH)  │  │     (Ising Dynamics)        │  │
//! │  │   ~20ns latency │  │   ~1µs queries  │  │     ~100µs/64K sweep        │  │
//! │  └────────┬────────┘  └────────┬────────┘  └────────┬────────────────────┘  │
//! │           │                    │                    │                        │
//! │           └────────────────────┴────────────────────┘                        │
//! │                                │                                             │
//! │                     ┌──────────▼──────────┐                                  │
//! │                     │   GPU Orchestrator  │                                  │
//! │                     │   (Optional)        │                                  │
//! │                     └─────────────────────┘                                  │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use hyperphysics_cortical_bus::prelude::*;
//!
//! // Create cortical bus with ecosystem integration
//! let bus = CorticalBus::new(CorticalConfig::default())?;
//!
//! // Inject spikes via lock-free ring buffers
//! bus.inject_spike(Spike::excitatory(neuron_id, timestamp, region))?;
//!
//! // Query similar patterns via HNSW (sub-microsecond)
//! let neighbors = bus.similarity_search(&embedding, k)?;
//!
//! // Update pBit lattice via Metropolis sweep
//! bus.update_pbit_fabric(temperature)?;
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

// ============================================================================
// Core Modules (Always Available)
// ============================================================================

pub mod spike;
pub mod ringbuf;
pub mod error;
pub mod scalable_pbit;

/// ChunkProcessor bridge for temporal spike hierarchy (requires `geometry` feature)
#[cfg(feature = "geometry")]
#[cfg_attr(docsrs, doc(cfg(feature = "geometry")))]
pub mod chunk_bridge;

// ============================================================================
// Ecosystem Integration Modules
// ============================================================================

/// pBit dynamics integration (requires `pbit` feature)
#[cfg(feature = "pbit")]
#[cfg_attr(docsrs, doc(cfg(feature = "pbit")))]
pub mod pbit_bridge;

/// Similarity search integration (requires `similarity` feature)
#[cfg(feature = "similarity")]
#[cfg_attr(docsrs, doc(cfg(feature = "similarity")))]
pub mod similarity_bridge;

/// GPU acceleration integration (requires `gpu` feature)
#[cfg(feature = "gpu")]
#[cfg_attr(docsrs, doc(cfg(feature = "gpu")))]
pub mod gpu_bridge;

// ============================================================================
// Re-exports from Ecosystem
// ============================================================================

#[cfg(feature = "pbit")]
pub use hyperphysics_pbit::{
    PBit, PBitLattice, PBitDynamics, Algorithm,
    MetropolisSimulator, GillespieSimulator, CouplingNetwork,
};

#[cfg(feature = "similarity")]
pub use hyperphysics_similarity::{
    HybridIndex, SearchConfig, SearchMode, SearchResult,
};

#[cfg(feature = "gpu")]
pub use hyperphysics_gpu_unified::{
    GpuOrchestrator, OrchestratorConfig, GpuPreference,
};

// ============================================================================
// Core Re-exports
// ============================================================================

pub use spike::{Spike, SpikeVec, SPIKE_SIZE};
pub use ringbuf::{SpscRingBuffer, MpscRingBuffer};
pub use error::{CorticalError, Result};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ============================================================================
// Unified Cortical Bus
// ============================================================================

mod cortical;
pub use cortical::{CorticalBus, CorticalConfig};

/// Prelude for common imports.
pub mod prelude {
    pub use crate::spike::{Spike, SpikeVec};
    pub use crate::ringbuf::{SpscRingBuffer, MpscRingBuffer};
    pub use crate::error::{CorticalError, Result};
    pub use crate::cortical::{CorticalBus, CorticalConfig};
    pub use crate::scalable_pbit::{
        ScalablePBitFabric, ScalablePBitConfig, PackedPBitArray,
        SparseCouplings, MetropolisSweeper, SweepResult,
    };

    #[cfg(feature = "pbit")]
    pub use hyperphysics_pbit::{PBitLattice, PBitDynamics, Algorithm};

    #[cfg(feature = "similarity")]
    pub use hyperphysics_similarity::{HybridIndex, SearchMode};

    #[cfg(feature = "gpu")]
    pub use hyperphysics_gpu_unified::GpuOrchestrator;

    #[cfg(feature = "geometry")]
    pub use crate::chunk_bridge::{
        ChunkProcessorBridge, ChunkBridgeConfig, TimescaleMapper, BridgeStats,
    };
}
