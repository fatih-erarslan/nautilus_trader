//! Quantum LSTM gates implementation

use crate::{error::Result, types::*};

/// Quantum LSTM gate operations
pub struct QuantumLSTMGate {
    num_qubits: usize,
}

impl QuantumLSTMGate {
    /// Create new quantum LSTM gate
    pub fn new(num_qubits: usize) -> Self {
        Self { num_qubits }
    }
}