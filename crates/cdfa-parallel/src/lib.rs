//! Parallel processing backends for CDFA
//! 
//! This crate implements various parallelization strategies:
//! - Lock-free data structures for minimal contention
//! - Tokio-based async processing with backpressure
//! - Rayon-based data parallelism
//! - Thread pool management with NUMA awareness
//! - GPU acceleration via Candle
//! - Distributed processing support
//!
//! # Performance Targets
//! - Throughput: >10M samples/second
//! - Latency: <1μs per operation
//! - Lock-free operations for critical paths
//! - NUMA-aware thread placement

pub mod async_framework;
pub mod communication;
pub mod lock_free;
pub mod parallel_algorithms;
pub mod thread_management;
pub mod ultra_optimization;

// Legacy modules (to be implemented)
pub mod distributed;
pub mod gpu_backend;
pub mod rayon_backend;
pub mod tokio_backend;

// Re-exports for convenience
pub use async_framework::{
    AsyncDiversityAnalyzer, AsyncSignalProcessor, BackpressureController, 
    PipelineConfig, StreamingPipeline,
};
pub use communication::{
    ErrorPropagator, Message, MessageRouter, ResultAggregator, RouterConfig,
};
pub use lock_free::{
    LockFreeFeatureCache, LockFreeResultAggregator, LockFreeSignalBuffer,
    WaitFreeCorrelationMatrix,
};
pub use parallel_algorithms::{
    ConcurrentFusionProcessor, MultiThreadedWaveletTransformer,
    ParallelDiversityCalculator, ParallelStatisticsCalculator,
};
pub use thread_management::{
    CdfaThreadPool, CpuAffinityManager, NumaAwareThreadPool, 
    ThreadAssignmentStrategy, ThreadPoolConfig,
};
pub use ultra_optimization::{UltraFastRingBuffer, prefetch};

#[cfg(target_arch = "x86_64")]
pub use ultra_optimization::correlation_avx2_manual;

#[cfg(feature = "gpu")]
pub use gpu_backend::*;

/// Initialize the parallel processing subsystem
///
/// This function sets up thread pools, configures NUMA awareness,
/// and prepares the system for high-performance parallel processing.
pub fn initialize(num_threads: Option<usize>) -> Result<(), cdfa_core::error::Error> {
    // Set up global thread pool
    if let Some(threads) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("cdfa-global-{}", i))
            .build_global()
            .map_err(|e| cdfa_core::error::Error::Config(e.to_string()))?;
    }
    
    // Initialize NUMA detection
    let topology = thread_management::CpuTopology::detect();
    log::info!(
        "Detected CPU topology: {} NUMA nodes, {} total cores",
        topology.numa_nodes,
        topology.total_cores
    );
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cdfa_core::types::{Signal, SignalId};

    #[test]
    fn test_parallel_initialization() {
        assert!(initialize(Some(4)).is_ok());
    }
    
    #[test]
    fn test_lock_free_signal_processing() {
        let buffer = LockFreeSignalBuffer::new(100);
        
        // Producer
        let signal = Signal::new(SignalId(1), 1000, vec![1.0, 2.0, 3.0]);
        assert!(buffer.push(signal.clone()).is_ok());
        
        // Consumer
        let received = buffer.try_pop().unwrap();
        assert_eq!(received.id, signal.id);
    }
    
    #[test]
    fn test_parallel_diversity_calculation() {
        let calc = ParallelDiversityCalculator::new(None, 64).unwrap();
        
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 1.0, 3.0, 5.0];
        
        let tau = calc.kendall_tau_parallel(&x, &y);
        assert!(tau >= -1.0 && tau <= 1.0);
    }
}