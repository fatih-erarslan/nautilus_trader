//! # Quantum Core Framework
//! 
//! High-performance quantum computing primitives and abstractions for trading algorithms.
//! Provides fundamental quantum data types, circuits, gates, and hardware interfaces.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
pub use num_complex::Complex64;

// Core constants
/// Maximum number of qubits supported
pub const MAX_QUBITS: usize = 32;

/// Default precision for quantum computations
pub const DEFAULT_PRECISION: f64 = 1e-10;

// Core modules
pub mod quantum_state;
pub mod quantum_circuits;
pub mod quantum_gates;
pub mod quantum_device;
pub mod hardware;
pub mod error;
// pub mod metrics;  // Temporarily disabled due to metrics API changes
pub mod utils;
pub mod traits;
pub mod core_types;
pub mod quantum_agent_trait;

// Re-export core types
pub use quantum_state::*;
pub use quantum_circuits::*;
pub use quantum_gates::*;
pub use quantum_device::*;
pub use hardware::*;
pub use error::*;
// pub use // metrics::*;  // Temporarily disabled
pub use utils::*;
pub use traits::*;
pub use core_types::*;
pub use quantum_agent_trait::*;

/// Complex amplitude type for quantum computations
pub type ComplexAmplitude = Complex64;

/// Hardware configuration for quantum devices
pub type HardwareConfig = HashMap<String, f64>;

/// Quantum computation result container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumResult {
    /// Quantum state after computation
    pub state: QuantumState,
    /// Measurement probabilities
    pub probabilities: Vec<f64>,
    /// Computation metadata
    pub metadata: ComputationMetadata,
    /// Fidelity of the quantum computation
    pub fidelity: f64,
    /// Execution time in nanoseconds
    pub execution_time_ns: u64,
}

/// Metadata about quantum computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationMetadata {
    /// Number of qubits used
    pub num_qubits: usize,
    /// Number of gates applied
    pub gate_count: usize,
    /// Circuit depth
    pub circuit_depth: usize,
    /// Backend used for computation
    pub backend: String,
    /// Error correction applied
    pub error_correction: bool,
}

impl QuantumResult {
    /// Create a new quantum result
    pub fn new(
        state: QuantumState,
        probabilities: Vec<f64>,
        backend: String,
        execution_time_ns: u64,
    ) -> Self {
        let metadata = ComputationMetadata {
            num_qubits: state.num_qubits(),
            gate_count: 0, // Will be updated by circuit execution
            circuit_depth: 0,
            backend,
            error_correction: false,
        };
        
        Self {
            state,
            probabilities,
            metadata,
            fidelity: 1.0, // Perfect fidelity by default
            execution_time_ns,
        }
    }
    
    /// Get the most likely measurement outcome
    pub fn most_likely_outcome(&self) -> usize {
        self.probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
    
    /// Calculate the entropy of the measurement distribution
    pub fn entropy(&self) -> f64 {
        self.probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum()
    }
}