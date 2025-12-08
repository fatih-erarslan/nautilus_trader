//! # Optimizers Module
//!
//! High-performance optimization algorithms for neural networks and feature learning.
//!
//! This module provides state-of-the-art optimization algorithms including:
//! - STDP (Spike-Timing Dependent Plasticity) - Sub-microsecond neural weight optimization
//! - Adaptive learning rates and momentum
//! - Memory-efficient implementations with custom allocators
//!
//! ## Performance Features
//!
//! - SIMD vectorized operations for maximum throughput
//! - Lock-free data structures for parallel processing
//! - Cache-aligned memory layouts for optimal CPU performance
//! - Custom memory allocators (mimalloc, bumpalo) for reduced latency
//!

pub mod stdp;

pub use stdp::*;

// Re-export key types
pub use crate::error::{CdfaError, CdfaResult};

/// Common traits for all optimizers
pub trait Optimizer {
    type Config;
    type Output;
    
    /// Create a new optimizer with given configuration
    fn new(config: Self::Config) -> CdfaResult<Self>
    where
        Self: Sized;
        
    /// Reset optimizer state
    fn reset(&mut self) -> CdfaResult<()>;
    
    /// Get optimizer statistics
    fn stats(&self) -> OptimizerStats;
}

/// Common optimizer statistics
#[derive(Debug, Clone)]
pub struct OptimizerStats {
    pub iterations: u64,
    pub convergence_rate: f64,
    pub avg_update_time_ns: u64,
    pub memory_usage_bytes: usize,
}

impl Default for OptimizerStats {
    fn default() -> Self {
        Self {
            iterations: 0,
            convergence_rate: 0.0,
            avg_update_time_ns: 0,
            memory_usage_bytes: 0,
        }
    }
}