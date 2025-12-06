//! Quantum computation module for QBMIA Core
//! 
//! This module provides quantum algorithms for market analysis and game theory,
//! including the quantum Nash equilibrium solver.

pub mod nash_equilibrium;
pub mod pbit_nash;
pub mod state_serializer;
pub mod circuit_builder;

// Re-export main types
pub use nash_equilibrium::{QuantumNashEquilibrium, QuantumNashResult, GameMatrix};
pub use pbit_nash::{PBitNashEquilibrium, PBitNashResult};
pub use state_serializer::QuantumStateSerializer;

use crate::error::{QBMIAError, Result};
use ndarray::{Array1, Array2, Array4};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Common quantum computation utilities
pub mod utils {
    use super::*;
    
    /// Calculate fidelity between two quantum states
    pub fn quantum_fidelity(state1: &Array1<f64>, state2: &Array1<f64>) -> f64 {
        state1.dot(state2).abs()
    }
    
    /// Calculate von Neumann entropy of a quantum state
    pub fn von_neumann_entropy(probabilities: &[f64]) -> f64 {
        probabilities
            .iter()
            .filter(|&&p| p > 1e-12)
            .map(|&p| -p * p.log2())
            .sum()
    }
    
    /// Normalize a quantum state vector
    pub fn normalize_state(state: &mut Array1<f64>) -> Result<()> {
        let norm = state.dot(state).sqrt();
        if norm < 1e-12 {
            return Err(QBMIAError::quantum_simulation("Cannot normalize zero state"));
        }
        *state /= norm;
        Ok(())
    }
    
    /// Generate random quantum state
    pub fn random_quantum_state(dimension: usize) -> Array1<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut state: Array1<f64> = Array1::from_vec(
            (0..dimension).map(|_| rng.random_range(0.0..1.0)).collect()
        );
        normalize_state(&mut state).unwrap();
        state
    }
    
    /// Check if probabilities form a valid probability distribution
    pub fn is_valid_probability_distribution(probs: &[f64], tolerance: f64) -> bool {
        let sum: f64 = probs.iter().sum();
        (sum - 1.0).abs() < tolerance && probs.iter().all(|&p| p >= -tolerance && p <= 1.0 + tolerance)
    }
}

/// Quantum device types for simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantumDevice {
    /// CPU-based simulation
    Cpu,
    /// GPU-accelerated simulation (CUDA)
    Gpu,
    /// Automatic device selection
    Auto,
}

impl Default for QuantumDevice {
    fn default() -> Self {
        Self::Auto
    }
}

/// Quantum gate types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantumGate {
    /// Pauli-X gate
    X(usize),
    /// Pauli-Y gate
    Y(usize),
    /// Pauli-Z gate
    Z(usize),
    /// Hadamard gate
    H(usize),
    /// Rotation around X-axis
    RX(usize, f64),
    /// Rotation around Y-axis
    RY(usize, f64),
    /// Rotation around Z-axis
    RZ(usize, f64),
    /// Controlled-NOT gate
    CNOT(usize, usize),
    /// Controlled-Z gate
    CZ(usize, usize),
    /// Custom unitary gate
    Custom(usize, Array2<f64>),
}

/// Quantum circuit representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Sequence of quantum gates
    pub gates: Vec<QuantumGate>,
    /// Circuit parameters
    pub parameters: Vec<f64>,
}

impl QuantumCircuit {
    /// Create a new quantum circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
            parameters: Vec::new(),
        }
    }
    
    /// Add a quantum gate to the circuit
    pub fn add_gate(&mut self, gate: QuantumGate) {
        self.gates.push(gate);
    }
    
    /// Add a parameterized gate
    pub fn add_parameterized_gate(&mut self, gate: QuantumGate, param_idx: usize) {
        self.gates.push(gate);
        if param_idx >= self.parameters.len() {
            self.parameters.resize(param_idx + 1, 0.0);
        }
    }
    
    /// Get circuit depth
    pub fn depth(&self) -> usize {
        self.gates.len()
    }
    
    /// Validate circuit structure
    pub fn validate(&self) -> Result<()> {
        for gate in &self.gates {
            match gate {
                QuantumGate::X(q) | QuantumGate::Y(q) | QuantumGate::Z(q) | QuantumGate::H(q)
                | QuantumGate::RX(q, _) | QuantumGate::RY(q, _) | QuantumGate::RZ(q, _) => {
                    if *q >= self.num_qubits {
                        return Err(QBMIAError::quantum_simulation(
                            format!("Qubit index {} out of bounds", q)
                        ));
                    }
                }
                QuantumGate::CNOT(control, target) | QuantumGate::CZ(control, target) => {
                    if *control >= self.num_qubits || *target >= self.num_qubits {
                        return Err(QBMIAError::quantum_simulation(
                            format!("Qubit indices {}, {} out of bounds", control, target)
                        ));
                    }
                    if control == target {
                        return Err(QBMIAError::quantum_simulation(
                            "Control and target qubits cannot be the same"
                        ));
                    }
                }
                QuantumGate::Custom(q, matrix) => {
                    if *q >= self.num_qubits {
                        return Err(QBMIAError::quantum_simulation(
                            format!("Qubit index {} out of bounds", q)
                        ));
                    }
                    if matrix.shape() != [2, 2] {
                        return Err(QBMIAError::quantum_simulation(
                            "Custom gate matrix must be 2x2"
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Performance metrics for quantum operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMetrics {
    /// Total execution time in milliseconds
    pub execution_time_ms: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// Number of quantum gates executed
    pub gates_executed: usize,
    /// Quantum state entropy
    pub state_entropy: f64,
    /// Convergence rate
    pub convergence_rate: f64,
}

impl Default for QuantumMetrics {
    fn default() -> Self {
        Self {
            execution_time_ms: 0.0,
            memory_usage_mb: 0.0,
            gates_executed: 0,
            state_entropy: 0.0,
            convergence_rate: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantum_circuit_creation() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.add_gate(QuantumGate::H(0));
        circuit.add_gate(QuantumGate::CNOT(0, 1));
        circuit.add_gate(QuantumGate::RZ(2, std::f64::consts::PI / 4.0));
        
        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.depth(), 3);
        assert!(circuit.validate().is_ok());
    }
    
    #[test]
    fn test_quantum_circuit_validation() {
        let mut circuit = QuantumCircuit::new(2);
        
        // Valid gate
        circuit.add_gate(QuantumGate::H(0));
        assert!(circuit.validate().is_ok());
        
        // Invalid gate - qubit out of bounds
        circuit.add_gate(QuantumGate::X(5));
        assert!(circuit.validate().is_err());
    }
    
    #[test]
    fn test_quantum_utils() {
        use utils::*;
        
        // Test probability distribution validation
        assert!(is_valid_probability_distribution(&[0.3, 0.7], 1e-6));
        assert!(!is_valid_probability_distribution(&[0.3, 0.8], 1e-6));
        
        // Test random state generation
        let state = random_quantum_state(4);
        assert_eq!(state.len(), 4);
        assert!((state.dot(&state) - 1.0).abs() < 1e-10);
        
        // Test entropy calculation
        let uniform_probs = vec![0.25, 0.25, 0.25, 0.25];
        let entropy = von_neumann_entropy(&uniform_probs);
        assert!((entropy - 2.0).abs() < 1e-10); // log2(4) = 2
    }
}