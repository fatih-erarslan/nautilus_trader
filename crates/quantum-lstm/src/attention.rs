//! Quantum attention mechanism

use crate::{error::Result, types::*};

/// Quantum attention mechanism
pub struct QuantumAttention {
    num_qubits: usize,
    num_heads: usize,
}

impl QuantumAttention {
    /// Create new quantum attention
    pub fn new(num_qubits: usize, num_heads: usize) -> Self {
        Self { num_qubits, num_heads }
    }
}