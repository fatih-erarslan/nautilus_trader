// Hierarchical Attention Cascade Module
// Target: <5ms total system latency
// Architecture: Multi-scale attention with temporal bridging

pub mod cascade_coordinator;
pub mod macro_attention;
pub mod micro_attention;
pub mod milli_attention;
pub mod temporal_bridge;

use crossbeam_channel::{Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub use cascade_coordinator::CascadeCoordinator;
pub use macro_attention::MacroAttention;
pub use micro_attention::MicroAttention;
pub use milli_attention::MilliAttention;
pub use temporal_bridge::TemporalBridge;

/// Attention layer performance targets
#[derive(Debug, Clone)]
pub struct AttentionTargets {
    pub micro_latency_ns: u64,  // <10,000 ns (10μs)
    pub milli_latency_ns: u64,  // <1,000,000 ns (1ms)
    pub macro_latency_ns: u64,  // <10,000,000 ns (10ms)
    pub bridge_latency_ns: u64, // <100,000 ns (100μs)
    pub total_latency_ns: u64,  // <5,000,000 ns (5ms)
}

impl Default for AttentionTargets {
    fn default() -> Self {
        Self {
            micro_latency_ns: 10_000,     // 10μs
            milli_latency_ns: 1_000_000,  // 1ms
            macro_latency_ns: 10_000_000, // 10ms
            bridge_latency_ns: 100_000,   // 100μs
            total_latency_ns: 5_000_000,  // 5ms
        }
    }
}

/// Market data input for attention processing
#[derive(Debug, Clone)]
pub struct MarketInput {
    pub timestamp: u64,
    pub price: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
    pub order_flow: Vec<f64>,
    pub microstructure: Vec<f64>,
}

/// Attention output with confidence scores
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    pub timestamp: u64,
    pub signal_strength: f64,
    pub confidence: f64,
    pub direction: i8, // -1, 0, 1
    pub position_size: f64,
    pub risk_score: f64,
    pub execution_time_ns: u64,
}

/// Performance metrics for attention cascade
#[derive(Debug, Clone)]
pub struct AttentionMetrics {
    pub micro_latency_ns: u64,
    pub milli_latency_ns: u64,
    pub macro_latency_ns: u64,
    pub bridge_latency_ns: u64,
    pub total_latency_ns: u64,
    pub throughput_ops_per_sec: f64,
    pub cache_hit_rate: f64,
    pub memory_usage_bytes: usize,
}

/// Configuration for attention cascade system
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub targets: AttentionTargets,
    pub enable_simd: bool,
    pub enable_gpu: bool,
    pub parallel_cores: usize,
    pub memory_pool_size: usize,
    pub cache_size: usize,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            targets: AttentionTargets::default(),
            enable_simd: true,
            enable_gpu: true,
            parallel_cores: num_cpus::get(),
            memory_pool_size: 1024 * 1024 * 100, // 100MB
            cache_size: 1024 * 1024 * 10,        // 10MB
        }
    }
}

/// Error types for attention system
#[derive(Debug, thiserror::Error)]
pub enum AttentionError {
    #[error("Latency target exceeded: {actual_ns}ns > {target_ns}ns")]
    LatencyExceeded { actual_ns: u64, target_ns: u64 },

    #[error("SIMD not available on this platform")]
    SimdNotAvailable,

    #[error("GPU acceleration failed: {reason}")]
    GpuFailed { reason: String },

    #[error("Memory allocation failed")]
    MemoryAllocation,

    #[error("Attention convergence failed")]
    ConvergenceFailed,
}

pub type AttentionResult<T> = Result<T, AttentionError>;

/// Trait for attention layer implementations
pub trait AttentionLayer: Send + Sync {
    fn process(&mut self, input: &MarketInput) -> AttentionResult<AttentionOutput>;
    fn get_metrics(&self) -> AttentionMetrics;
    fn reset_metrics(&mut self);
    fn validate_performance(&self) -> AttentionResult<()>;
}

/// Channel types for inter-layer communication
pub type MicroChannel = (Sender<MarketInput>, Receiver<AttentionOutput>);
pub type MilliChannel = (Sender<AttentionOutput>, Receiver<AttentionOutput>);
pub type MacroChannel = (Sender<AttentionOutput>, Receiver<AttentionOutput>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_targets() {
        let targets = AttentionTargets::default();
        assert_eq!(targets.micro_latency_ns, 10_000);
        assert_eq!(targets.total_latency_ns, 5_000_000);
    }

    #[test]
    fn test_market_input_creation() {
        let input = MarketInput {
            timestamp: 1640995200000,
            price: 45000.0,
            volume: 1.5,
            bid: 44990.0,
            ask: 45010.0,
            order_flow: vec![0.5, -0.3, 0.8],
            microstructure: vec![0.1, 0.2, -0.1],
        };
        assert!(input.price > 0.0);
        assert!(input.volume > 0.0);
    }
}
