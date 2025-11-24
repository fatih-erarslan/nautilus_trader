//! # HyperPhysics GPU Unified
//!
//! Unified GPU orchestration layer for the HyperPhysics ecosystem.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        GpuOrchestrator                                   │
//! │  - Shared wgpu::Device across all crates                                │
//! │  - Dual-GPU workload distribution (RX 6800 XT + RX 5500 XT)             │
//! │  - Pipeline scheduling & batching                                       │
//! │  - Memory pressure monitoring                                           │
//! └─────────────────────────────────────────────────────────────────────────┘
//!          │              │              │
//!          ▼              ▼              ▼
//!    Physics Pool   Finance Pool   Neural Pool
//!    (pBit, SPH)    (VaR, MC)      (Burn, Attn)
//!          │              │              │
//!          └──────────────┼──────────────┘
//!                         ▼
//!          ┌──────────────────────────────┐
//!          │   DualGpuCoordinator         │
//!          │   RX 6800 XT │ RX 5500 XT   │
//!          │   (Heavy)    │ (Async)       │
//!          └──────────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - `async`: Enable async runtime support via tokio
//! - `burn`: Enable Burn ML framework integration
//! - `full`: Enable all features

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod orchestrator;
pub mod pools;
pub mod kernels;
pub mod interop;
pub mod adapters;

mod error;

pub use error::{GpuError, GpuResult};
pub use orchestrator::{GpuOrchestrator, OrchestratorConfig};
pub use pools::{ComputePool, PoolType};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::orchestrator::{GpuOrchestrator, OrchestratorConfig};
    pub use crate::pools::{ComputePool, PoolType};
    pub use crate::error::{GpuError, GpuResult};
    pub use crate::kernels::KernelRegistry;
    pub use crate::adapters::{UnifiedGpuAccelerator, UnifiedGpuBuffer, UnifiedGpuKernel};

    #[cfg(feature = "burn")]
    pub use crate::interop::burn::{BurnBridge, BurnConfig};
}

/// GPU device preference for dual-GPU systems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuPreference {
    /// High-performance GPU (RX 6800 XT) - for heavy compute
    #[default]
    Primary,
    /// Low-power GPU (RX 5500 XT) - for async/background tasks
    Secondary,
    /// Auto-select based on workload characteristics
    Auto,
}

/// Workload characteristics for intelligent GPU routing
#[derive(Debug, Clone)]
pub enum WorkloadType {
    /// Memory-intensive operation (large buffers)
    MemoryBound {
        /// Required VRAM in bytes
        required_vram: u64,
    },
    /// Compute-intensive operation (many FLOPS)
    ComputeBound {
        /// Estimated FLOPS
        estimated_flops: u64,
    },
    /// Latency-critical operation (real-time requirement)
    LatencyCritical {
        /// Maximum allowed latency in microseconds
        deadline_us: u64,
    },
    /// Background operation (can be deferred)
    Background,
}

impl WorkloadType {
    /// Determine optimal GPU for this workload
    pub fn preferred_gpu(&self) -> GpuPreference {
        match self {
            // Memory-bound: use high-VRAM GPU (16GB)
            WorkloadType::MemoryBound { required_vram } if *required_vram > 4_000_000_000 => {
                GpuPreference::Primary
            }
            // Compute-bound: use high-CU GPU (72 CUs)
            WorkloadType::ComputeBound { estimated_flops } if *estimated_flops > 1_000_000_000 => {
                GpuPreference::Primary
            }
            // Latency-critical: use less-loaded GPU
            WorkloadType::LatencyCritical { .. } => GpuPreference::Auto,
            // Background: use secondary
            WorkloadType::Background => GpuPreference::Secondary,
            // Default: primary
            _ => GpuPreference::Primary,
        }
    }
}

/// Hardware specifications for AMD GPUs
#[derive(Debug, Clone)]
pub struct GpuSpecs {
    /// GPU name
    pub name: String,
    /// Number of compute units
    pub compute_units: u32,
    /// Threads per wavefront
    pub wavefront_size: u32,
    /// VRAM in bytes
    pub vram_bytes: u64,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f32,
    /// Infinity Cache size in bytes (RDNA2+)
    pub infinity_cache_bytes: u64,
}

impl GpuSpecs {
    /// RX 6800 XT specifications
    pub fn rx_6800_xt() -> Self {
        Self {
            name: "AMD Radeon RX 6800 XT".to_string(),
            compute_units: 72,
            wavefront_size: 64,
            vram_bytes: 16 * 1024 * 1024 * 1024, // 16GB
            memory_bandwidth_gbps: 512.0,
            infinity_cache_bytes: 128 * 1024 * 1024, // 128MB
        }
    }

    /// RX 5500 XT specifications
    pub fn rx_5500_xt() -> Self {
        Self {
            name: "AMD Radeon RX 5500 XT".to_string(),
            compute_units: 22,
            wavefront_size: 64,
            vram_bytes: 4 * 1024 * 1024 * 1024, // 4GB
            memory_bandwidth_gbps: 224.0,
            infinity_cache_bytes: 0, // RDNA1, no Infinity Cache
        }
    }

    /// Maximum concurrent threads
    pub fn max_threads(&self) -> u32 {
        self.compute_units * self.wavefront_size
    }

    /// Optimal workgroup size (4 wavefronts per CU)
    pub fn optimal_workgroup_size(&self) -> u32 {
        256 // 4 wavefronts = 256 threads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_specs() {
        let rx6800xt = GpuSpecs::rx_6800_xt();
        assert_eq!(rx6800xt.compute_units, 72);
        assert_eq!(rx6800xt.max_threads(), 72 * 64);
        assert_eq!(rx6800xt.optimal_workgroup_size(), 256);

        let rx5500xt = GpuSpecs::rx_5500_xt();
        assert_eq!(rx5500xt.compute_units, 22);
    }

    #[test]
    fn test_workload_routing() {
        let heavy_memory = WorkloadType::MemoryBound { required_vram: 8_000_000_000 };
        assert_eq!(heavy_memory.preferred_gpu(), GpuPreference::Primary);

        let background = WorkloadType::Background;
        assert_eq!(background.preferred_gpu(), GpuPreference::Secondary);
    }
}
