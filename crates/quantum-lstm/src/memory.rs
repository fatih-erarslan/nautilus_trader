//! Quantum associative memory

use crate::{error::Result, types::*};

/// Quantum memory with error correction
pub struct QuantumMemory {
    num_qubits: usize,
    num_ancilla: usize,
}

impl QuantumMemory {
    /// Create new quantum memory
    pub fn new(num_qubits: usize, num_ancilla: usize) -> Self {
        Self { num_qubits, num_ancilla }
    }
}