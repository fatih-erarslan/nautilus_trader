//! # GPU Bridge
//!
//! Integration bridge between the cortical bus and `hyperphysics-gpu-unified`.
//!
//! Provides GPU acceleration for:
//! - pBit Metropolis sweeps (massively parallel)
//! - Batch similarity search
//! - Spike batch processing
//!
//! ## Dual-GPU Architecture
//!
//! ```text
//! Cortical Bus
//!      │
//!      ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │                    GPU Bridge                            │
//! │                                                          │
//! │  ┌─────────────────────┐  ┌──────────────────────────┐  │
//! │  │   Primary GPU       │  │   Secondary GPU           │  │
//! │  │   (RX 6800 XT)      │  │   (RX 5500 XT)            │  │
//! │  │                     │  │                           │  │
//! │  │ - pBit sweeps       │  │ - Background promotion    │  │
//! │  │ - Hot path queries  │  │ - LSH maintenance         │  │
//! │  │ - High throughput   │  │ - Async operations        │  │
//! │  └─────────────────────┘  └──────────────────────────┘  │
//! └─────────────────────────────────────────────────────────┘
//! ```

use hyperphysics_gpu_unified::{
    GpuOrchestrator, OrchestratorConfig, GpuPreference, WorkloadType,
    prelude::*,
};

use crate::spike::Spike;
use crate::error::{CorticalError, Result};

/// GPU bridge configuration.
#[derive(Debug, Clone)]
pub struct GpuBridgeConfig {
    /// Minimum batch size to use GPU (below this, CPU is faster).
    pub min_batch_size: usize,
    /// Maximum pBits to process on GPU per dispatch.
    pub max_pbits_per_dispatch: usize,
    /// Enable async background operations.
    pub enable_async: bool,
    /// Prefer primary (high-power) or secondary (low-power) GPU.
    pub gpu_preference: GpuPreference,
}

impl Default for GpuBridgeConfig {
    fn default() -> Self {
        Self {
            min_batch_size: 1000,
            max_pbits_per_dispatch: 65536,
            enable_async: true,
            gpu_preference: GpuPreference::Auto,
        }
    }
}

/// Bridge between cortical bus and GPU orchestrator.
///
/// Automatically routes workloads to appropriate GPU based on:
/// - Workload size (small → CPU, large → GPU)
/// - Workload type (compute-bound → primary, async → secondary)
/// - Current GPU utilization
pub struct GpuBridge {
    /// GPU orchestrator from hyperphysics-gpu-unified.
    orchestrator: GpuOrchestrator,
    /// Configuration.
    config: GpuBridgeConfig,
    /// Statistics.
    stats: GpuBridgeStats,
}

/// Statistics for GPU operations.
#[derive(Debug, Default, Clone)]
pub struct GpuBridgeStats {
    /// Operations routed to primary GPU.
    pub primary_dispatches: u64,
    /// Operations routed to secondary GPU.
    pub secondary_dispatches: u64,
    /// Operations that fell back to CPU.
    pub cpu_fallbacks: u64,
    /// Total GPU compute time (nanoseconds).
    pub total_gpu_ns: u64,
    /// Total bytes transferred to GPU.
    pub bytes_uploaded: u64,
    /// Total bytes transferred from GPU.
    pub bytes_downloaded: u64,
}

impl GpuBridge {
    /// Create a new GPU bridge.
    pub fn new(config: GpuBridgeConfig) -> Result<Self> {
        let gpu_config = OrchestratorConfig::default();
        let orchestrator = GpuOrchestrator::new(gpu_config)
            .map_err(|e| CorticalError::ConfigError(format!("GPU orchestrator: {}", e)))?;

        Ok(Self {
            orchestrator,
            config,
            stats: GpuBridgeStats::default(),
        })
    }

    /// Check if GPU acceleration is available.
    pub fn is_available(&self) -> bool {
        self.orchestrator.is_available()
    }

    /// Get information about available GPUs.
    pub fn device_info(&self) -> String {
        self.orchestrator.device_info()
    }

    /// Perform GPU-accelerated Metropolis sweep on pBit states.
    ///
    /// Uses checkerboard decomposition for parallel updates.
    pub async fn metropolis_sweep_gpu(
        &mut self,
        states: &mut [bool],
        biases: &[f64],
        couplings: &[(usize, usize, f64)], // (i, j, J_ij)
        temperature: f64,
    ) -> Result<u32> {
        let num_pbits = states.len();

        // Fall back to CPU for small workloads
        if num_pbits < self.config.min_batch_size {
            self.stats.cpu_fallbacks += 1;
            return Err(CorticalError::ConfigError(
                "Workload too small for GPU, use CPU".into(),
            ));
        }

        self.stats.primary_dispatches += 1;
        self.stats.bytes_uploaded += (states.len() + biases.len() * 8) as u64;

        // TODO: Implement actual GPU kernel dispatch
        // For now, this is a placeholder that documents the API
        //
        // The actual implementation would:
        // 1. Upload states, biases, couplings to GPU buffers
        // 2. Dispatch checkerboard Metropolis kernel
        // 3. Download updated states
        // 4. Return flip count

        Err(CorticalError::NotInitialized("GPU Metropolis kernel not yet implemented"))
    }

    /// Perform GPU-accelerated batch similarity search.
    pub async fn batch_similarity_search(
        &mut self,
        queries: &[Vec<f32>],
        index_data: &[f32],
        k: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>> {
        if queries.len() < self.config.min_batch_size {
            self.stats.cpu_fallbacks += 1;
            return Err(CorticalError::ConfigError(
                "Batch too small for GPU".into(),
            ));
        }

        self.stats.primary_dispatches += 1;

        // TODO: Implement GPU similarity search
        // Would dispatch parallel KNN queries across all queries

        Err(CorticalError::NotInitialized("GPU similarity search not yet implemented"))
    }

    /// Process spike batch on GPU.
    ///
    /// Useful for spike-based plasticity computations.
    pub async fn process_spike_batch(
        &mut self,
        spikes: &[Spike],
        weights: &mut [f32],
    ) -> Result<()> {
        if spikes.len() < self.config.min_batch_size {
            self.stats.cpu_fallbacks += 1;
            return Err(CorticalError::ConfigError(
                "Batch too small for GPU".into(),
            ));
        }

        self.stats.secondary_dispatches += 1; // Use secondary GPU for async

        // TODO: Implement spike processing kernel
        // Would compute weight updates from spike timing

        Err(CorticalError::NotInitialized("GPU spike processing not yet implemented"))
    }

    /// Get GPU memory usage.
    pub fn memory_usage(&self) -> (u64, u64) {
        self.orchestrator.memory_usage()
    }

    /// Get statistics.
    pub fn stats(&self) -> &GpuBridgeStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = GpuBridgeStats::default();
    }

    /// Get configuration.
    pub fn config(&self) -> &GpuBridgeConfig {
        &self.config
    }

    /// Synchronize all pending GPU operations.
    pub async fn sync(&self) -> Result<()> {
        self.orchestrator
            .sync()
            .await
            .map_err(CorticalError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_bridge_config() {
        let config = GpuBridgeConfig::default();
        assert_eq!(config.min_batch_size, 1000);
        assert!(config.enable_async);
    }

    // GPU tests require actual GPU hardware
    // #[tokio::test]
    // async fn test_gpu_bridge_creation() {
    //     let config = GpuBridgeConfig::default();
    //     let bridge = GpuBridge::new(config);
    //     // May fail if no GPU available - that's expected
    // }
}
