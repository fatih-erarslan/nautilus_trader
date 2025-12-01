//! # Lock-Free Ring Buffers
//!
//! Ultra-low-latency lock-free ring buffers for inter-thread spike communication.
//!
//! ## Design Goals
//!
//! - **SPSC**: Single-Producer Single-Consumer for wait-free operations
//! - **MPSC**: Multi-Producer Single-Consumer for aggregation
//! - **Cache-line aligned**: Prevent false sharing
//! - **Zero allocation**: All memory pre-allocated
//!
//! ## Performance Targets
//!
//! - SPSC push/pop: <20ns
//! - MPSC push: <50ns (with contention)

mod spsc;
mod mpsc;

pub use spsc::SpscRingBuffer;
pub use mpsc::MpscRingBuffer;

/// Default ring buffer capacity (power of 2 for fast modulo).
pub const DEFAULT_CAPACITY: usize = 4096;

/// Cache line size for alignment.
pub const CACHE_LINE_SIZE: usize = 64;
