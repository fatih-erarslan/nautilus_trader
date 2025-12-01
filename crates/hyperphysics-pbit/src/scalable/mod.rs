//! # Scalable pBit Backend
//!
//! High-performance pBit implementation for large-scale systems (millions to billions).
//! Designed for minimum latency with measurable, reproducible benchmarks.
//!
//! ## Design Principles
//!
//! 1. **Packed Storage**: 64 pBits per u64 word (64× memory reduction)
//! 2. **No Allocation**: Zero heap allocation in hot path
//! 3. **Cache Efficient**: Sequential memory access patterns
//! 4. **Simple CSR**: Compressed Sparse Row without external dependencies
//!
//! ## Performance Targets (measured, not claimed)
//!
//! | Scale | Target | Metric |
//! |-------|--------|--------|
//! | 1K pBits | <50µs/sweep | 50ns/spin |
//! | 64K pBits | <5ms/sweep | 75ns/spin |
//! | 1M pBits | <100ms/sweep | 100ns/spin |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use hyperphysics_pbit::scalable::{ScalablePBitArray, ScalableCouplings};
//!
//! // Create array
//! let mut array = ScalablePBitArray::new(65536);
//!
//! // Add couplings
//! let mut couplings = ScalableCouplings::new(65536);
//! couplings.add(0, 1, 0.5);
//! couplings.finalize();
//!
//! // Run dynamics
//! let flips = array.metropolis_sweep(&couplings, 1.0);
//! ```

mod array;
mod couplings;
mod sweep;
mod simd_sweep;
mod gpu;

pub use array::ScalablePBitArray;
pub use couplings::{ScalableCouplings, CouplingEntry};
pub use sweep::{MetropolisSweep, SweepStats, AggregateStats};
pub use simd_sweep::{SimdSweep, SimdSweepStats};
pub use gpu::{GpuExecutor, GpuSweepStats, METAL_SHADER};

/// Configuration for scalable pBit systems
#[derive(Debug, Clone)]
pub struct ScalableConfig {
    /// Number of pBits
    pub num_pbits: usize,
    /// Initial temperature
    pub temperature: f64,
    /// RNG seed (for reproducibility)
    pub seed: u64,
}

impl Default for ScalableConfig {
    fn default() -> Self {
        Self {
            num_pbits: 1024,
            temperature: 1.0,
            seed: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration() {
        // Create system
        let mut array = ScalablePBitArray::new(100);
        let mut couplings = ScalableCouplings::new(100);
        
        // Ring coupling
        for i in 0..100 {
            couplings.add(i, (i + 1) % 100, 1.0);
        }
        couplings.finalize();
        
        // Run
        let mut sweep = MetropolisSweep::new(1.0, 42);
        let stats = sweep.execute(&mut array, &couplings, &[0.0; 100]);
        
        assert!(stats.duration_ns > 0);
        assert!(stats.flips <= 100);
    }

    #[test]
    fn test_scaling_1k() {
        let n = 1000;
        let mut array = ScalablePBitArray::random(n, 42);
        let mut couplings = ScalableCouplings::with_capacity(n, n * 6);
        
        // Random sparse couplings
        let mut rng = fastrand::Rng::with_seed(123);
        for _ in 0..(n * 3) {
            let i = rng.usize(0..n);
            let j = rng.usize(0..n);
            if i != j {
                couplings.add(i, j, rng.f32() * 2.0 - 1.0);
            }
        }
        couplings.finalize();
        
        let biases = vec![0.0f32; n];
        let mut sweep = MetropolisSweep::new(1.0, 42);
        
        // Warm up
        for _ in 0..10 {
            sweep.execute(&mut array, &couplings, &biases);
        }
        
        // Benchmark
        let start = std::time::Instant::now();
        let num_sweeps = 100;
        for _ in 0..num_sweeps {
            sweep.execute(&mut array, &couplings, &biases);
        }
        let elapsed = start.elapsed();
        
        let ns_per_sweep = elapsed.as_nanos() / num_sweeps;
        let ns_per_spin = ns_per_sweep / n as u128;
        
        println!("1K pBits: {}ns/sweep, {}ns/spin", ns_per_sweep, ns_per_spin);
        
        // Should be under 200ns/spin in release mode
        assert!(ns_per_spin < 1000, "Too slow: {}ns/spin", ns_per_spin);
    }
}
