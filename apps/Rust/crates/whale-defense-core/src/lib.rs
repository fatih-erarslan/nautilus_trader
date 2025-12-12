//! Ultra-Fast Whale Defense Core
//! 
//! Sub-microsecond whale detection and defense system ported from C++
//! Target performance: <500ns detection, <200ns defense execution
//! 
//! # Architecture
//! - Lock-free data structures for zero-contention processing
//! - SIMD-optimized pattern matching using AVX-512
//! - Cache-aligned memory layouts for optimal performance
//! - Quantum game theory engine for defense strategy selection
//! - Steganographic order management for execution hiding
//! 
//! # Performance Guarantees
//! - Whale detection latency: <500 nanoseconds
//! - Defense strategy execution: <200 nanoseconds
//! - Memory access: Zero-copy, cache-aligned operations
//! - Concurrent throughput: >1M operations/second
//! 
//! # Safety
//! This crate uses extensive unsafe code for maximum performance.
//! All unsafe operations are carefully documented and tested.

//#![no_std] // Temporarily disabled for compilation compatibility
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(feature = "simd", feature(avx512_target_feature))]
#![warn(missing_docs)]
#![warn(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)] // Documented in implementation

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

// Re-export core types and traits
pub use crate::core::{
    WhaleDefenseEngine, ThreatLevel, DefenseResult, MarketOrder, WhaleActivity,
    WhaleType, WhaleSignal, WhaleStatistics, DefenseStrategy
};

pub use crate::quantum::QuantumGameTheoryEngine;
pub use crate::steganography::SteganographicOrderManager;
pub use crate::lockfree::LockFreeRingBuffer;
pub use crate::performance::{PerformanceMonitor, Metrics};
pub use crate::error::WhaleDefenseError;

// Core modules
pub mod core;
pub mod quantum;
pub mod steganography;
pub mod lockfree;
pub mod performance;
pub mod error;
pub mod simd;
pub mod cache;
pub mod timing;

// Configuration and constants
pub mod config {
    //! Configuration constants for ultra-fast whale defense
    
    /// Maximum number of concurrent whale detection threads
    pub const MAX_DETECTION_THREADS: usize = 16;
    
    /// Lock-free ring buffer size (must be power of 2)
    pub const RING_BUFFER_SIZE: usize = 4096;
    
    /// Cache line size for optimal alignment
    pub const CACHE_LINE_SIZE: usize = 64;
    
    /// Maximum whale history to maintain
    pub const MAX_WHALE_HISTORY: usize = 1000;
    
    /// Target detection latency in nanoseconds
    pub const TARGET_DETECTION_LATENCY_NS: u64 = 500;
    
    /// Target defense execution latency in nanoseconds
    pub const TARGET_DEFENSE_EXECUTION_NS: u64 = 200;
    
    /// SIMD vector width for AVX-512
    pub const SIMD_WIDTH: usize = 8;
    
    /// Maximum number of concurrent defense strategies
    pub const MAX_DEFENSE_STRATEGIES: usize = 4;
    
    /// Performance monitoring interval in nanoseconds
    pub const MONITORING_INTERVAL_NS: u64 = 1_000_000; // 1ms
}

// Global allocator for optimal memory management
#[cfg(feature = "std")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Re-export commonly used types
pub use alloc::{vec::Vec, string::String, boxed::Box, collections::VecDeque};
pub use std::{
    time::Duration,
    sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering},
    mem::{align_of, size_of},
    ptr::NonNull,
    slice,
    mem::MaybeUninit,
};

#[cfg(feature = "std")]
pub use std::sync::Arc;

// Conditional re-exports based on features
#[cfg(feature = "simd")]
pub use crate::simd::*;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const GIT_HASH: &str = env!("GIT_HASH");

/// Initialize the whale defense system
/// 
/// # Safety
/// This function must be called before any other whale defense operations.
/// It initializes global state and memory allocators.
pub unsafe fn init() -> Result<(), WhaleDefenseError> {
    // Initialize performance counters
    performance::init_performance_counters()?;
    
    // Warm up CPU caches
    cache::warm_up_caches();
    
    // Initialize quantum random number generators
    quantum::init_quantum_rng()?;
    
    // Verify SIMD capabilities
    #[cfg(feature = "simd")]
    {
        if !simd::check_avx512_support() {
            return Err(WhaleDefenseError::UnsupportedCpuFeature("AVX-512"));
        }
    }
    
    Ok(())
}

/// Shutdown the whale defense system
/// 
/// # Safety
/// This function should be called before program termination to ensure
/// proper cleanup of resources.
pub unsafe fn shutdown() {
    quantum::shutdown_quantum_rng();
    performance::shutdown_performance_counters();
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_initialization() {
        unsafe {
            assert!(init().is_ok());
            shutdown();
        }
    }
    
    #[test]
    fn test_config_constants() {
        assert_eq!(config::CACHE_LINE_SIZE, 64);
        assert_eq!(config::RING_BUFFER_SIZE & (config::RING_BUFFER_SIZE - 1), 0); // Power of 2
        assert!(config::TARGET_DETECTION_LATENCY_NS < 1000); // Sub-microsecond
        assert!(config::TARGET_DEFENSE_EXECUTION_NS < 500); // Sub-microsecond
    }
}