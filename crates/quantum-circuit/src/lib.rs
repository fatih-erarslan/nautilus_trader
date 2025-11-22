//! # Quantum-Enhanced Circuit Architectures
//!
//! A quantum-inspired machine learning library with PennyLane-compatible API design.
//! Focuses on quantum-enhanced classical ML using efficient simulation of small quantum circuits.
//!
//! ## Features
//!
//! - Quantum-inspired neural circuit architectures
//! - Variational quantum circuit (VQC) simulators
//! - Quantum-enhanced optimization algorithms (QAOA-inspired)
//! - Parametric quantum circuits with automatic differentiation
//! - Quantum-inspired feature maps and embeddings
//! - Hybrid classical-quantum neural networks
//! - Quantum-enhanced attention mechanisms
//! - Amplitude encoding for classical data
//!
//! ## Example
//!
//! ```rust
//! use quantum_circuit::{Circuit, gates::*};
//!
//! let mut circuit = Circuit::new(2);
//! circuit.add_gate(RX::new(0, 0.5))?;
//! circuit.add_gate(CNOT::new(0, 1))?;
//! circuit.add_gate(RY::new(1, 0.8))?;
//!
//! let state = circuit.execute()?;
//! println!("Final state: {:?}", state);
//! # Ok::<(), quantum_circuit::QuantumError>(())
//! ```

pub mod gates;
pub mod circuit;
pub mod simulation;
pub mod optimization;
pub mod embeddings;
pub mod neural;
pub mod pennylane_compat;

use num_complex::Complex64;
use thiserror::Error;

/// Complex number type used throughout the library
pub type Complex = Complex64;

/// Quantum state vector type
pub type StateVector = ndarray::Array1<Complex>;

/// Quantum operator matrix type
pub type Operator = ndarray::Array2<Complex>;

/// Parameter type for quantum circuits
pub type Parameter = f64;

/// Quantum circuit error types
#[derive(Error, Debug)]
pub enum QuantumError {
    #[error("Invalid qubit index: {0}")]
    InvalidQubit(usize),
    
    #[error("Invalid gate parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Circuit execution failed: {0}")]
    ExecutionError(String),
    
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Optimization failed: {0}")]
    OptimizationError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Invalid quantum state")]
    InvalidState,
}

/// Result type for quantum operations
pub type Result<T> = std::result::Result<T, QuantumError>;

/// Constants and utilities
pub mod constants {
    use super::*;
    
    /// Identity matrix (2x2)
    pub fn identity() -> Operator {
        ndarray::array![[Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
                       [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)]]
    }
    
    /// Pauli-X gate matrix
    pub fn pauli_x() -> Operator {
        ndarray::array![[Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
                       [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]]
    }
    
    /// Pauli-Y gate matrix
    pub fn pauli_y() -> Operator {
        ndarray::array![[Complex::new(0.0, 0.0), Complex::new(0.0, -1.0)],
                       [Complex::new(0.0, 1.0), Complex::new(0.0, 0.0)]]
    }
    
    /// Pauli-Z gate matrix
    pub fn pauli_z() -> Operator {
        ndarray::array![[Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
                       [Complex::new(0.0, 0.0), Complex::new(-1.0, 0.0)]]
    }
    
    /// Hadamard gate matrix
    pub fn hadamard() -> Operator {
        let sqrt2_inv = 1.0 / (2.0_f64).sqrt();
        ndarray::array![[Complex::new(sqrt2_inv, 0.0), Complex::new(sqrt2_inv, 0.0)],
                       [Complex::new(sqrt2_inv, 0.0), Complex::new(-sqrt2_inv, 0.0)]]
    }
    
    /// |0⟩ state
    pub fn zero_state() -> StateVector {
        ndarray::array![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]
    }
    
    /// |1⟩ state
    pub fn one_state() -> StateVector {
        ndarray::array![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)]
    }
    
    /// |+⟩ state (equal superposition)
    pub fn plus_state() -> StateVector {
        let sqrt2_inv = 1.0 / (2.0_f64).sqrt();
        ndarray::array![Complex::new(sqrt2_inv, 0.0), Complex::new(sqrt2_inv, 0.0)]
    }
    
    /// |-⟩ state
    pub fn minus_state() -> StateVector {
        let sqrt2_inv = 1.0 / (2.0_f64).sqrt();
        ndarray::array![Complex::new(sqrt2_inv, 0.0), Complex::new(-sqrt2_inv, 0.0)]
    }
}

/// Utility functions for quantum operations
pub mod utils {
    use super::*;
    use ndarray::Array1;
    
    /// Calculate the fidelity between two quantum states
    pub fn fidelity(state1: &StateVector, state2: &StateVector) -> Result<f64> {
        if state1.len() != state2.len() {
            return Err(QuantumError::DimensionMismatch {
                expected: state1.len(),
                actual: state2.len(),
            });
        }
        
        let overlap = state1.iter()
            .zip(state2.iter())
            .map(|(a, b)| a.conj() * b)
            .sum::<Complex>();
            
        Ok(overlap.norm_sqr())
    }
    
    /// Normalize a quantum state vector
    pub fn normalize_state(state: &mut StateVector) -> Result<()> {
        let norm = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return Err(QuantumError::InvalidState);
        }
        *state /= Complex::new(norm, 0.0);
        Ok(())
    }
    
    /// Calculate expectation value of an operator
    pub fn expectation_value(state: &StateVector, operator: &Operator) -> Result<Complex> {
        if state.len() != operator.nrows() || operator.nrows() != operator.ncols() {
            return Err(QuantumError::DimensionMismatch {
                expected: state.len(),
                actual: operator.nrows(),
            });
        }
        
        let op_state = operator.dot(state);
        let expectation = state.iter()
            .zip(op_state.iter())
            .map(|(psi, op_psi)| psi.conj() * op_psi)
            .sum();
            
        Ok(expectation)
    }
    
    /// Generate a random quantum state vector
    pub fn random_state(n_qubits: usize) -> StateVector {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let dim = 1 << n_qubits;
        
        let mut state = Array1::zeros(dim);
        for i in 0..dim {
            state[i] = Complex::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
        }
        
        let norm = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        state /= Complex::new(norm, 0.0);
        state
    }
    
    /// Convert classical data to amplitude encoding
    pub fn amplitude_encode(data: &[f64]) -> Result<StateVector> {
        let n = data.len();
        if !n.is_power_of_two() {
            return Err(QuantumError::InvalidParameter(
                "Data length must be a power of 2 for amplitude encoding".to_string()
            ));
        }
        
        let norm = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return Err(QuantumError::InvalidParameter(
                "Data vector has zero norm".to_string()
            ));
        }
        
        let mut state = Array1::zeros(n);
        for (i, &val) in data.iter().enumerate() {
            state[i] = Complex::new(val / norm, 0.0);
        }
        
        Ok(state)
    }
}

// Re-export commonly used items
pub use circuit::{Circuit, CircuitBuilder, VariationalCircuit, EntanglementPattern};
pub use gates::{Gate, QuantumGate, ParametricGate};
pub use simulation::{Simulator, StateEvolution};
pub use optimization::{Optimizer, VariationalOptimizer, OptimizationResult, VQEOptimizer, AdamOptimizer, QAOAOptimizer, OptimizerConfig};
pub use pennylane_compat::{device, qnode, QNode, DefaultQubitDevice, QNodeBuilder};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_constants() {
        let id = constants::identity();
        assert_eq!(id.shape(), &[2, 2]);
        assert_abs_diff_eq!(id[[0, 0]].re, 1.0);
        assert_abs_diff_eq!(id[[1, 1]].re, 1.0);
    }
    
    #[test]
    fn test_utils() {
        let state1 = constants::zero_state();
        let state2 = constants::one_state();
        
        let fidelity = utils::fidelity(&state1, &state1).unwrap();
        assert_abs_diff_eq!(fidelity, 1.0);
        
        let fidelity = utils::fidelity(&state1, &state2).unwrap();
        assert_abs_diff_eq!(fidelity, 0.0);
    }
    
    #[test]
    fn test_amplitude_encoding() {
        let data = vec![1.0, 0.0, 0.0, 0.0];
        let state = utils::amplitude_encode(&data).unwrap();
        assert_abs_diff_eq!(state[0].re, 1.0);
        assert_abs_diff_eq!(state[1].re, 0.0);
    }
}