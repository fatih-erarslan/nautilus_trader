//! Execution layer
//!
//! Low-latency execution implementations for HFT trading
//!
//! This module provides the execution infrastructure:
//! - Order routing and smart order routing (SOR)
//! - Latency-optimized execution paths
//! - Exchange connectivity abstractions
//!
//! # Architecture
//!
//! Execution is organized by latency tiers:
//! - **Direct**: <100μs (co-located, FPGA-assisted)
//! - **Standard**: <1ms (optimized TCP/UDP)
//! - **Aggregated**: <10ms (smart routing across venues)

/// Execution configuration
#[derive(Debug, Clone, Default)]
pub struct ExecutionConfig {
    /// Maximum acceptable latency in microseconds
    pub max_latency_us: u64,
    /// Enable smart order routing
    pub smart_routing: bool,
    /// Number of retry attempts
    pub retry_count: u32,
}

/// Execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Execution latency in microseconds
    pub latency_us: u64,
    /// Fill ratio (0.0-1.0)
    pub fill_ratio: f64,
    /// Average fill price
    pub avg_price: f64,
}

