//! Quantum circuit builder utilities

use crate::{
    error::{QBMIAError, Result},
    quantum::{QuantumCircuit, QuantumGate},
};
use ndarray::Array2;

/// Quantum circuit builder for creating common circuit patterns
pub struct CircuitBuilder {
    circuit: QuantumCircuit,
}

impl CircuitBuilder {
    /// Create a new circuit builder
    pub fn new(num_qubits: usize) -> Self {
        Self {
            circuit: QuantumCircuit::new(num_qubits),
        }
    }
    
    /// Add a layer of Hadamard gates to all qubits
    pub fn add_hadamard_layer(mut self) -> Self {
        for i in 0..self.circuit.num_qubits {
            self.circuit.add_gate(QuantumGate::H(i));
        }
        self
    }
    
    /// Add a layer of parameterized rotation gates
    pub fn add_rotation_layer(mut self, param_offset: usize) -> Self {
        for i in 0..self.circuit.num_qubits {
            self.circuit.add_parameterized_gate(
                QuantumGate::RX(i, 0.0),
                param_offset + i * 3,
            );
            self.circuit.add_parameterized_gate(
                QuantumGate::RY(i, 0.0),
                param_offset + i * 3 + 1,
            );
            self.circuit.add_parameterized_gate(
                QuantumGate::RZ(i, 0.0),
                param_offset + i * 3 + 2,
            );
        }
        self
    }
    
    /// Add CNOT entangling layer with ring topology
    pub fn add_cnot_ring_layer(mut self) -> Self {
        for i in 0..self.circuit.num_qubits {
            let target = (i + 1) % self.circuit.num_qubits;
            self.circuit.add_gate(QuantumGate::CNOT(i, target));
        }
        self
    }
    
    /// Add CZ entangling layer for even pairs
    pub fn add_cz_pairs_layer(mut self) -> Self {
        for i in (0..self.circuit.num_qubits - 1).step_by(2) {
            self.circuit.add_gate(QuantumGate::CZ(i, i + 1));
        }
        self
    }
    
    /// Build the final circuit
    pub fn build(self) -> Result<QuantumCircuit> {
        self.circuit.validate()?;
        Ok(self.circuit)
    }
}