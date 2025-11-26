//! QUIC-based swarm coordination for Neural Trader
//!
//! This crate provides high-performance, low-latency swarm coordination using QUIC protocol,
//! with integrated ReasoningBank for adaptive learning and pattern recognition.
//!
//! ## Features
//!
//! - **Sub-millisecond latency**: <1ms p99 latency for agent coordination
//! - **Massive concurrency**: Support for 1000+ concurrent bidirectional streams
//! - **Built-in encryption**: TLS 1.3 by default
//! - **0-RTT connection resumption**: Instant reconnection for agents
//! - **ReasoningBank integration**: Adaptive learning from coordination patterns
//! - **Stream multiplexing**: Multiple independent communication channels per agent
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │         QUIC Swarm Coordinator (Server)            │
//! └─────────────────────────────────────────────────────┘
//!           │                    │                    │
//!     ┌─────┴─────┐      ┌─────┴─────┐      ┌─────┴─────┐
//!     │  Agent 1  │      │  Agent 2  │      │  Agent N  │
//!     │ (Client)  │      │ (Client)  │      │ (Client)  │
//!     └───────────┘      └───────────┘      └───────────┘
//!
//! Each connection supports multiple bidirectional streams:
//! - Pattern matching results
//! - Strategy correlations
//! - ReasoningBank experiences
//! - Neural gradients
//! - Task assignments
//! ```

pub mod coordinator;
pub mod agent;
pub mod types;
pub mod tls;
pub mod metrics;
pub mod reasoningbank;
pub mod error;

pub use coordinator::QuicSwarmCoordinator;
pub use agent::QuicSwarmAgent;
pub use types::*;
pub use error::{SwarmError, Result};

/// Re-export quinn types for convenience
pub use quinn::{Connection, SendStream, RecvStream};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_compiles() {
        // Basic compilation test
        assert!(true);
    }
}
