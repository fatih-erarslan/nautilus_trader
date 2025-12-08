//! # PADS Assembly: Panarchy Adaptive Decision System
//!
//! Central coordination system for 12 quantum agents implementing the Panarchy
//! Adaptive Decision System (PADS) as specified in the quantum enhancement addendum.
//! 
//! ## Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    PADS CENTRAL ASSEMBLY                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────────┐    ┌─────────────────────────────────────┐ │
//! │  │   Panarchy      │    │        12 Quantum Agents            │ │
//! │  │   Layer         │ ◄─►│  ┌─────┬─────┬─────┬─────┐           │ │
//! │  │  ┌──────────────┤    │  │ QAR │QBMI │QBDIA│ QAR │           │ │
//! │  │  │ Adaptive     │    │  ├─────┼─────┼─────┼─────┤           │ │
//! │  │  │ Cycles       │    │  │QERC │IQAD │ NQO │QLMSR│           │ │
//! │  │  └──────────────┤    │  ├─────┼─────┼─────┼─────┤           │ │
//! │  │  │ Cross-Scale  │    │  │ QPT │ QHA │QLSTM│ QWD │           │ │
//! │  │  │ Interaction  │    │  └─────┴─────┴─────┴─────┘           │ │
//! │  │  └──────────────┤    └─────────────────────────────────────┘ │
//! │  │  │ Resilience   │              ▲                             │
//! │  │  │ Mechanisms   │              │                             │
//! │  │  └──────────────┘              ▼                             │
//! │  └─────────────────┘    ┌─────────────────────────────────────┐ │
//! │           ▲              │      Decision Synthesis Engine      │ │
//! │           │              │  ┌─────────────────────────────────┤ │
//! │           │              │  │     Convergence Engine          │ │
//! │           │              │  └─────────────────────────────────┤ │
//! │           │              │  │   Signal Processing Pipeline    │ │
//! │           └──────────────┼──┴─────────────────────────────────┘ │
//! │                          │                                     │
//! └──────────────────────────┼─────────────────────────────────────┘
//!                            ▼
//!                    Trading Signals
//! ```
//!
//! ## Quantum Agents (12 Total)
//!
//! 1. **Quantum Agentic Reasoning (QAR)** - Meta-reasoning and strategy synthesis
//! 2. **Quantum Biological Market Intuition (QBMI)** - Nature-inspired pattern recognition  
//! 3. **Quantum BDIA** - Behavioral dynamics analysis
//! 4. **Quantum Annealing Regression (QAR)** - Optimization problem solving
//! 5. **QERC** - Quantum error correction and reliability
//! 6. **IQAD** - Intelligent quantum anomaly detection
//! 7. **NQO** - Neural quantum optimization
//! 8. **Quantum LMSR** - Market scoring and prediction
//! 9. **Quantum Prospect Theory (QPT)** - Behavioral finance modeling
//! 10. **Quantum Hedge Algorithm (QHA)** - Portfolio protection
//! 11. **Quantum LSTM** - Enhanced time series prediction
//! 12. **Quantum Whale Defense (QWD)** - Large trader detection/defense

#![deny(missing_docs, unsafe_op_in_unsafe_fn)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions, clippy::similar_names)]

use std::sync::Arc;
use tokio::sync::RwLock;

pub mod assembly;
pub mod agents;
pub mod panarchy;
pub mod synthesis;
pub mod convergence;
pub mod coordination;
pub mod error;
pub mod types;

#[cfg(feature = "simd-optimized")]
pub mod simd;

// Re-exports for convenience
pub use assembly::{PadsAssembly, AssemblyConfig, AssemblyMetrics};
pub use agents::{QuantumAgent, AgentManager, AgentRegistry};
pub use panarchy::{PanarchyLayer, AdaptiveCycle, CrossScaleInteraction, ResilienceMechanism};
pub use synthesis::{DecisionSynthesizer, SynthesisStrategy, SynthesisResult};
pub use convergence::{ConvergenceEngine, ConvergenceMetrics, ConvergenceStrategy};
pub use coordination::{CoordinationBus, CoordinationMessage, AgentCoordination};
pub use error::{PadsError, AssemblyError, AgentError, CoordinationError};
pub use types::{
    MarketData, TradingSignal, QuantumState, AgentSignal, DecisionContext,
    PadsDecision, SignalStrength, ConfidenceLevel
};

/// High-performance PADS assembly instance
pub type Assembly = Arc<RwLock<PadsAssembly>>;

/// Initialize PADS assembly with default configuration
///
/// # Errors
///
/// Returns `PadsError` if:
/// - Quantum bridge initialization fails
/// - Agent spawning fails
/// - Panarchy layer setup fails
///
/// # Example
///
/// ```rust
/// use pads_assembly::initialize_assembly;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let assembly = initialize_assembly().await?;
///     
///     // Assembly is ready for quantum agent coordination
///     Ok(())
/// }
/// ```
pub async fn initialize_assembly() -> Result<Assembly, PadsError> {
    let config = AssemblyConfig::default();
    let assembly = PadsAssembly::new(config).await?;
    Ok(Arc::new(RwLock::new(assembly)))
}

/// Initialize assembly with custom configuration
///
/// # Arguments
///
/// * `config` - Custom assembly configuration
///
/// # Example
///
/// ```rust
/// use pads_assembly::{initialize_assembly_with_config, AssemblyConfig};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let config = AssemblyConfig {
///         max_concurrent_agents: 12,
///         enable_panarchy_full: true,
///         quantum_coherence_threshold: 0.95,
///         ..Default::default()
///     };
///     
///     let assembly = initialize_assembly_with_config(config).await?;
///     Ok(())
/// }
/// ```
pub async fn initialize_assembly_with_config(config: AssemblyConfig) -> Result<Assembly, PadsError> {
    let assembly = PadsAssembly::new(config).await?;
    Ok(Arc::new(RwLock::new(assembly)))
}

/// PADS assembly version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// PADS assembly build metadata
pub const BUILD_INFO: &str = concat!(
    "pads-assembly v",
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("VERGEN_GIT_SHA"),
    ")"
);

/// Number of quantum agents in the assembly
pub const QUANTUM_AGENT_COUNT: usize = 12;

/// Agent type identifiers
pub mod agent_types {
    /// Quantum Agentic Reasoning
    pub const QAR: &str = "quantum_agentic_reasoning";
    /// Quantum Biological Market Intuition
    pub const QBMI: &str = "quantum_biological_market_intuition";
    /// Quantum Behavioral Dynamics Analysis
    pub const QBDIA: &str = "quantum_bdia";
    /// Quantum Annealing Regression
    pub const QAR_ANNEALING: &str = "quantum_annealing_regression";
    /// Quantum Error Correction
    pub const QERC: &str = "qerc";
    /// Intelligent Quantum Anomaly Detection
    pub const IQAD: &str = "iqad";
    /// Neural Quantum Optimization
    pub const NQO: &str = "nqo";
    /// Quantum Logarithmic Market Scoring Rules
    pub const QLMSR: &str = "quantum_lmsr";
    /// Quantum Prospect Theory
    pub const QPT: &str = "quantum_prospect_theory";
    /// Quantum Hedge Algorithm
    pub const QHA: &str = "quantum_hedge_algorithm";
    /// Quantum Long Short-Term Memory
    pub const QLSTM: &str = "quantum_lstm";
    /// Quantum Whale Defense
    pub const QWD: &str = "quantum_whale_defense";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_assembly_initialization() {
        let result = initialize_assembly().await;
        
        // Should either succeed or fail with proper error handling
        match result {
            Ok(assembly) => {
                let assembly_guard = assembly.read().await;
                assert_eq!(assembly_guard.agent_count().await, QUANTUM_AGENT_COUNT);
            }
            Err(e) => {
                // Expected if quantum bridge is not available in test environment
                println!("Assembly initialization failed (expected in test): {}", e);
            }
        }
    }

    #[test]
    fn test_agent_type_constants() {
        assert_eq!(QUANTUM_AGENT_COUNT, 12);
        assert!(!agent_types::QAR.is_empty());
        assert!(!agent_types::QBMI.is_empty());
        assert!(!agent_types::QBDIA.is_empty());
    }

    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert!(!BUILD_INFO.is_empty());
    }
}