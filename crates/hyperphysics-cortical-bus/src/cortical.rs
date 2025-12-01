//! # Unified Cortical Bus
//!
//! The main integration point for the HyperPhysics neuromorphic computing stack.
//! Combines spike routing, pattern memory, and pBit dynamics into a unified interface.

use crossbeam::queue::ArrayQueue;
use std::sync::Arc;

use crate::error::{CorticalError, Result};
use crate::spike::Spike;

#[cfg(feature = "pbit")]
use hyperphysics_pbit::{PBitLattice, PBitDynamics, Algorithm};
#[cfg(feature = "pbit")]
use rand::thread_rng;

#[cfg(feature = "similarity")]
use hyperphysics_similarity::{HybridIndex, SearchConfig, SearchResult};

#[cfg(feature = "gpu")]
use hyperphysics_gpu_unified::{GpuOrchestrator, OrchestratorConfig};

/// Cortical bus configuration.
#[derive(Debug, Clone)]
pub struct CorticalConfig {
    // ========================================================================
    // Spike Routing Configuration
    // ========================================================================
    
    /// Number of spike queues (partitioned by routing hint).
    pub num_queues: usize,
    /// Capacity per queue.
    pub queue_capacity: usize,

    // ========================================================================
    // pBit Configuration (hyperbolic tessellation)
    // ========================================================================
    
    /// Polygon sides for {p,q} tessellation.
    #[cfg(feature = "pbit")]
    pub tessellation_p: usize,
    /// Polygons per vertex.
    #[cfg(feature = "pbit")]
    pub tessellation_q: usize,
    /// Tessellation depth.
    #[cfg(feature = "pbit")]
    pub tessellation_depth: usize,
    /// Default temperature for Metropolis dynamics.
    #[cfg(feature = "pbit")]
    pub default_temperature: f64,

    // ========================================================================
    // Similarity Search Configuration
    // ========================================================================
    
    /// Embedding dimension for pattern memory.
    #[cfg(feature = "similarity")]
    pub embedding_dim: usize,
    /// HNSW M parameter (connections per node).
    #[cfg(feature = "similarity")]
    pub hnsw_m: usize,
    /// HNSW ef_construction parameter.
    #[cfg(feature = "similarity")]
    pub hnsw_ef_construction: usize,

    // ========================================================================
    // GPU Configuration
    // ========================================================================
    
    /// Enable GPU acceleration.
    #[cfg(feature = "gpu")]
    pub enable_gpu: bool,
}

impl Default for CorticalConfig {
    fn default() -> Self {
        Self {
            num_queues: 256,
            queue_capacity: 4096,

            // {3,7,2} tessellation gives 48 pBits (ROI standard)
            #[cfg(feature = "pbit")]
            tessellation_p: 3,
            #[cfg(feature = "pbit")]
            tessellation_q: 7,
            #[cfg(feature = "pbit")]
            tessellation_depth: 2,
            #[cfg(feature = "pbit")]
            default_temperature: 1.0,

            #[cfg(feature = "similarity")]
            embedding_dim: 128,
            #[cfg(feature = "similarity")]
            hnsw_m: 16,
            #[cfg(feature = "similarity")]
            hnsw_ef_construction: 100,

            #[cfg(feature = "gpu")]
            enable_gpu: false,
        }
    }
}

/// Unified cortical bus integrating HyperPhysics components.
///
/// This is the main entry point for neuromorphic computing operations,
/// providing:
///
/// - **Spike Routing**: Lock-free ring buffers for ~20ns spike delivery
/// - **Pattern Memory**: HNSW + LSH hybrid for sub-µs similarity search
/// - **pBit Fabric**: Ising dynamics for probabilistic computing
/// - **GPU Acceleration**: Dual-GPU orchestration (optional)
pub struct CorticalBus {
    // ========================================================================
    // Spike Routing Layer
    // ========================================================================
    
    /// Spike queues partitioned by routing hint.
    spike_queues: Vec<Arc<ArrayQueue<Spike>>>,
    /// Number of queues.
    num_queues: usize,

    // ========================================================================
    // pBit Layer
    // ========================================================================
    
    /// pBit dynamics controller (owns lattice).
    #[cfg(feature = "pbit")]
    pbit_dynamics: PBitDynamics,

    // ========================================================================
    // Similarity Search Layer
    // ========================================================================
    
    /// Hybrid HNSW + LSH index for pattern memory.
    #[cfg(feature = "similarity")]
    pattern_memory: HybridIndex,

    // ========================================================================
    // GPU Layer
    // ========================================================================
    
    /// GPU orchestrator for accelerated operations.
    #[cfg(feature = "gpu")]
    gpu: Option<GpuOrchestrator>,

    // ========================================================================
    // Configuration
    // ========================================================================
    
    /// Stored configuration.
    config: CorticalConfig,
}

impl CorticalBus {
    /// Create a new cortical bus with the given configuration.
    pub fn new(config: CorticalConfig) -> Result<Self> {
        // Initialize spike queues
        let spike_queues: Vec<_> = (0..config.num_queues)
            .map(|_| Arc::new(ArrayQueue::new(config.queue_capacity)))
            .collect();

        // Initialize pBit layer with hyperbolic tessellation
        #[cfg(feature = "pbit")]
        let pbit_dynamics = {
            let lattice = PBitLattice::new(
                config.tessellation_p,
                config.tessellation_q,
                config.tessellation_depth,
                config.default_temperature,
            ).map_err(|e| CorticalError::ConfigError(format!("pBit lattice: {}", e)))?;
            PBitDynamics::new_metropolis(lattice, config.default_temperature)
        };

        // Initialize similarity search layer
        #[cfg(feature = "similarity")]
        let pattern_memory = {
            let search_config = SearchConfig::default();
            HybridIndex::new(search_config)
                .map_err(|e| CorticalError::ConfigError(format!("Pattern memory: {}", e)))?
        };

        // Initialize GPU (if enabled and available)
        #[cfg(feature = "gpu")]
        let gpu = if config.enable_gpu {
            let gpu_config = OrchestratorConfig::default();
            match GpuOrchestrator::new(gpu_config) {
                Ok(g) => Some(g),
                Err(e) => {
                    tracing::warn!("GPU initialization failed, falling back to CPU: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            spike_queues,
            num_queues: config.num_queues,

            #[cfg(feature = "pbit")]
            pbit_dynamics,

            #[cfg(feature = "similarity")]
            pattern_memory,

            #[cfg(feature = "gpu")]
            gpu,

            config,
        })
    }

    // ========================================================================
    // Spike Routing Operations
    // ========================================================================

    /// Inject a spike into the cortical bus.
    ///
    /// Routes to appropriate queue based on `routing_hint`.
    /// Target latency: <50ns.
    #[inline]
    pub fn inject_spike(&self, spike: Spike) -> Result<()> {
        let queue_idx = (spike.routing_hint as usize) % self.num_queues;
        let queue = &self.spike_queues[queue_idx];

        queue.push(spike).map_err(|_| CorticalError::QueueFull {
            queue_id: queue_idx,
            pending: queue.len(),
        })
    }

    /// Inject a batch of spikes.
    ///
    /// More efficient than individual injection for bulk operations.
    pub fn inject_batch(&self, spikes: &[Spike]) -> Result<()> {
        for spike in spikes {
            self.inject_spike(*spike)?;
        }
        Ok(())
    }

    /// Poll spikes from all queues.
    ///
    /// Fills `buffer` with available spikes, returns count.
    pub fn poll_spikes(&self, buffer: &mut [Spike]) -> usize {
        let mut count = 0;
        for queue in &self.spike_queues {
            while count < buffer.len() {
                if let Some(spike) = queue.pop() {
                    buffer[count] = spike;
                    count += 1;
                } else {
                    break;
                }
            }
        }
        count
    }

    /// Get total number of pending spikes across all queues.
    pub fn pending_spikes(&self) -> usize {
        self.spike_queues.iter().map(|q| q.len()).sum()
    }

    // ========================================================================
    // pBit Operations
    // ========================================================================

    /// Perform a dynamics step on the pBit fabric.
    ///
    /// Updates pBits according to Metropolis dynamics.
    #[cfg(feature = "pbit")]
    pub fn update_pbit_fabric(&mut self) -> Result<()> {
        let mut rng = thread_rng();
        self.pbit_dynamics.step(&mut rng)
            .map_err(|e| CorticalError::Internal(format!("pBit dynamics: {}", e)))?;
        Ok(())
    }

    /// Run multiple pBit dynamics steps.
    #[cfg(feature = "pbit")]
    pub fn simulate_pbit(&mut self, steps: usize) -> Result<()> {
        let mut rng = thread_rng();
        self.pbit_dynamics.simulate(steps, &mut rng)
            .map_err(|e| CorticalError::Internal(format!("pBit simulation: {}", e)))?;
        Ok(())
    }

    /// Get the current pBit states as a vector.
    #[cfg(feature = "pbit")]
    pub fn pbit_states(&self) -> Vec<bool> {
        self.pbit_dynamics.lattice().states()
    }

    /// Get number of pBits in the fabric.
    #[cfg(feature = "pbit")]
    pub fn pbit_count(&self) -> usize {
        self.pbit_dynamics.lattice().size()
    }

    /// Get current dynamics algorithm.
    #[cfg(feature = "pbit")]
    pub fn pbit_algorithm(&self) -> Algorithm {
        self.pbit_dynamics.algorithm()
    }

    // ========================================================================
    // Similarity Search Operations
    // ========================================================================

    /// Search for similar patterns in the content-addressable memory.
    ///
    /// Uses HNSW for sub-microsecond queries on hot data.
    #[cfg(feature = "similarity")]
    pub fn similarity_search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.pattern_memory
            .search_hot(query, k)
            .map_err(CorticalError::from)
    }

    /// Store a pattern in the content-addressable memory.
    ///
    /// Goes through LSH for streaming ingestion.
    #[cfg(feature = "similarity")]
    pub fn store_pattern(&self, id: u64, embedding: &[f32]) -> Result<()> {
        self.pattern_memory
            .stream_ingest(id, embedding)
            .map_err(CorticalError::from)
    }

    /// Get number of patterns stored.
    #[cfg(feature = "similarity")]
    pub fn pattern_count(&self) -> usize {
        self.pattern_memory.len()
    }

    // ========================================================================
    // GPU Operations
    // ========================================================================

    /// Check if GPU acceleration is available.
    #[cfg(feature = "gpu")]
    pub fn gpu_available(&self) -> bool {
        self.gpu.is_some()
    }

    /// Get GPU device info (if available).
    #[cfg(feature = "gpu")]
    pub fn gpu_info(&self) -> Option<String> {
        self.gpu.as_ref().map(|g| g.device_info())
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /// Get the configuration used to create this bus.
    pub fn config(&self) -> &CorticalConfig {
        &self.config
    }

    /// Check if the bus is healthy (no overflows, systems responding).
    pub fn is_healthy(&self) -> bool {
        // Check queue health (not all full)
        let queues_ok = self.spike_queues.iter().any(|q| !q.is_full());

        // Check pBit health
        #[cfg(feature = "pbit")]
        let pbit_ok = self.pbit_dynamics.lattice().size() > 0;
        #[cfg(not(feature = "pbit"))]
        let pbit_ok = true;

        queues_ok && pbit_ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cortical_bus_creation() {
        let config = CorticalConfig::default();
        let bus = CorticalBus::new(config).unwrap();
        assert!(bus.is_healthy());
    }

    #[test]
    fn test_spike_injection() {
        let config = CorticalConfig::default();
        let bus = CorticalBus::new(config).unwrap();

        let spike = Spike::excitatory(12345, 100, 0xAB);
        bus.inject_spike(spike).unwrap();

        assert_eq!(bus.pending_spikes(), 1);

        let mut buffer = [Spike::default(); 10];
        let count = bus.poll_spikes(&mut buffer);
        assert_eq!(count, 1);
        assert_eq!(buffer[0].source_id, 12345);
    }

    #[test]
    fn test_batch_injection() {
        let config = CorticalConfig::default();
        let bus = CorticalBus::new(config).unwrap();

        let spikes: Vec<Spike> = (0..100)
            .map(|i| Spike::new(i, i as u16, 50, (i % 256) as u8))
            .collect();

        bus.inject_batch(&spikes).unwrap();
        assert_eq!(bus.pending_spikes(), 100);
    }
}
