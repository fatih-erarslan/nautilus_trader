//! Quantum Gate Operations for High-Performance Trading ML
//!
//! Implements basic quantum gate operations optimized for sub-100μs inference
//! with focus on quantum-classical hybrid algorithms for financial prediction.

use crate::TENGRIError;
use nalgebra::{Complex, DMatrix, DVector};
use std::f64::consts::PI;
use num_traits::Float;
use rayon::prelude::*;

/// Complex number type for quantum computations
pub type C64 = Complex<f64>;

/// Quantum state representation
#[derive(Debug, Clone)]
pub struct QuantumState {
    pub amplitudes: DVector<C64>,
    pub n_qubits: usize,
    pub fidelity: f64,
}

impl QuantumState {
    /// Create new quantum state in |0⟩ state
    pub fn new(n_qubits: usize) -> Self {
        let dim = 1usize << n_qubits;
        let mut amplitudes = DVector::zeros(dim);
        amplitudes[0] = C64::new(1.0, 0.0);
        
        Self {
            amplitudes,
            n_qubits,
            fidelity: 1.0,
        }
    }

    /// Create superposition state
    pub fn superposition(n_qubits: usize) -> Self {
        let dim = 1usize << n_qubits;
        let amplitude = C64::new(1.0 / (dim as f64).sqrt(), 0.0);
        let amplitudes = DVector::from_element(dim, amplitude);
        
        Self {
            amplitudes,
            n_qubits,
            fidelity: 1.0,
        }
    }

    /// Calculate quantum state fidelity
    pub fn calculate_fidelity(&self) -> f64 {
        let norm_squared: f64 = self.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .sum();
        norm_squared.sqrt()
    }

    /// Measure quantum state (collapse to classical)
    pub fn measure(&mut self) -> Result<Vec<usize>, TENGRIError> {
        let probabilities: Vec<f64> = self.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .collect();
        
        // Quantum measurement simulation
        let mut rng = rand::thread_rng();
        let random_val: f64 = rand::Rng::gen(&mut rng);
        
        let mut cumulative_prob = 0.0;
        for (i, prob) in probabilities.iter().enumerate() {
            cumulative_prob += prob;
            if random_val < cumulative_prob {
                // Collapse to measured state
                self.amplitudes.fill(C64::new(0.0, 0.0));
                self.amplitudes[i] = C64::new(1.0, 0.0);
                
                // Convert to binary representation
                let binary_result = (0..self.n_qubits)
                    .map(|bit| (i >> bit) & 1)
                    .collect();
                
                return Ok(binary_result);
            }
        }
        
        Err(TENGRIError::MathematicalValidationFailed {
            reason: "Quantum measurement failed".to_string(),
        })
    }
}

/// Quantum gate operations
pub struct QuantumGates;

impl QuantumGates {
    /// Pauli-X gate (quantum NOT)
    pub fn x_gate(state: &mut QuantumState, qubit: usize) -> Result<(), TENGRIError> {
        if qubit >= state.n_qubits {
            return Err(TENGRIError::MathematicalValidationFailed {
                reason: format!("Qubit {} out of range for {}-qubit system", qubit, state.n_qubits),
            });
        }

        let dim = 1usize << state.n_qubits;
        let mut new_amplitudes = state.amplitudes.clone();
        
        // Apply X gate in parallel for performance
        (0..dim).into_par_iter().for_each(|i| {
            let flipped_i = i ^ (1usize << qubit);
            new_amplitudes[i] = state.amplitudes[flipped_i];
        });
        
        state.amplitudes = new_amplitudes;
        state.fidelity = state.calculate_fidelity();
        Ok(())
    }

    /// Pauli-Y gate
    pub fn y_gate(state: &mut QuantumState, qubit: usize) -> Result<(), TENGRIError> {
        if qubit >= state.n_qubits {
            return Err(TENGRIError::MathematicalValidationFailed {
                reason: format!("Qubit {} out of range for {}-qubit system", qubit, state.n_qubits),
            });
        }

        let dim = 1usize << state.n_qubits;
        let mut new_amplitudes = state.amplitudes.clone();
        
        (0..dim).into_par_iter().for_each(|i| {
            let flipped_i = i ^ (1usize << qubit);
            let bit = (i >> qubit) & 1;
            
            if bit == 0 {
                new_amplitudes[i] = C64::new(0.0, 1.0) * state.amplitudes[flipped_i];
            } else {
                new_amplitudes[i] = C64::new(0.0, -1.0) * state.amplitudes[flipped_i];
            }
        });
        
        state.amplitudes = new_amplitudes;
        state.fidelity = state.calculate_fidelity();
        Ok(())
    }

    /// Pauli-Z gate
    pub fn z_gate(state: &mut QuantumState, qubit: usize) -> Result<(), TENGRIError> {
        if qubit >= state.n_qubits {
            return Err(TENGRIError::MathematicalValidationFailed {
                reason: format!("Qubit {} out of range for {}-qubit system", qubit, state.n_qubits),
            });
        }

        let dim = 1usize << state.n_qubits;
        
        (0..dim).into_par_iter().for_each(|i| {
            let bit = (i >> qubit) & 1;
            if bit == 1 {
                state.amplitudes[i] = -state.amplitudes[i];
            }
        });
        
        state.fidelity = state.calculate_fidelity();
        Ok(())
    }

    /// Hadamard gate (creates superposition)
    pub fn h_gate(state: &mut QuantumState, qubit: usize) -> Result<(), TENGRIError> {
        if qubit >= state.n_qubits {
            return Err(TENGRIError::MathematicalValidationFailed {
                reason: format!("Qubit {} out of range for {}-qubit system", qubit, state.n_qubits),
            });
        }

        let dim = 1usize << state.n_qubits;
        let mut new_amplitudes = state.amplitudes.clone();
        let sqrt_2_inv = C64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        
        (0..dim).into_par_iter().for_each(|i| {
            let flipped_i = i ^ (1usize << qubit);
            let bit = (i >> qubit) & 1;
            
            if bit == 0 {
                new_amplitudes[i] = sqrt_2_inv * (state.amplitudes[i] + state.amplitudes[flipped_i]);
            } else {
                new_amplitudes[i] = sqrt_2_inv * (state.amplitudes[i ^ (1usize << qubit)] - state.amplitudes[i]);
            }
        });
        
        state.amplitudes = new_amplitudes;
        state.fidelity = state.calculate_fidelity();
        Ok(())
    }

    /// Controlled-NOT gate (CNOT)
    pub fn cnot_gate(state: &mut QuantumState, control: usize, target: usize) -> Result<(), TENGRIError> {
        if control >= state.n_qubits || target >= state.n_qubits {
            return Err(TENGRIError::MathematicalValidationFailed {
                reason: format!("Qubit indices out of range for {}-qubit system", state.n_qubits),
            });
        }

        let dim = 1usize << state.n_qubits;
        let mut new_amplitudes = state.amplitudes.clone();
        
        (0..dim).into_par_iter().for_each(|i| {
            let control_bit = (i >> control) & 1;
            if control_bit == 1 {
                let flipped_i = i ^ (1usize << target);
                new_amplitudes[i] = state.amplitudes[flipped_i];
            }
        });
        
        state.amplitudes = new_amplitudes;
        state.fidelity = state.calculate_fidelity();
        Ok(())
    }

    /// Rotation gate around X-axis
    pub fn rx_gate(state: &mut QuantumState, qubit: usize, angle: f64) -> Result<(), TENGRIError> {
        if qubit >= state.n_qubits {
            return Err(TENGRIError::MathematicalValidationFailed {
                reason: format!("Qubit {} out of range for {}-qubit system", qubit, state.n_qubits),
            });
        }

        let dim = 1usize << state.n_qubits;
        let mut new_amplitudes = state.amplitudes.clone();
        
        let cos_half = C64::new((angle / 2.0).cos(), 0.0);
        let sin_half = C64::new(0.0, -(angle / 2.0).sin());
        
        (0..dim).into_par_iter().for_each(|i| {
            let flipped_i = i ^ (1usize << qubit);
            let bit = (i >> qubit) & 1;
            
            if bit == 0 {
                new_amplitudes[i] = cos_half * state.amplitudes[i] + sin_half * state.amplitudes[flipped_i];
            } else {
                new_amplitudes[i] = sin_half * state.amplitudes[i ^ (1usize << qubit)] + cos_half * state.amplitudes[i];
            }
        });
        
        state.amplitudes = new_amplitudes;
        state.fidelity = state.calculate_fidelity();
        Ok(())
    }

    /// Rotation gate around Y-axis
    pub fn ry_gate(state: &mut QuantumState, qubit: usize, angle: f64) -> Result<(), TENGRIError> {
        if qubit >= state.n_qubits {
            return Err(TENGRIError::MathematicalValidationFailed {
                reason: format!("Qubit {} out of range for {}-qubit system", qubit, state.n_qubits),
            });
        }

        let dim = 1usize << state.n_qubits;
        let mut new_amplitudes = state.amplitudes.clone();
        
        let cos_half = C64::new((angle / 2.0).cos(), 0.0);
        let sin_half = C64::new((angle / 2.0).sin(), 0.0);
        
        (0..dim).into_par_iter().for_each(|i| {
            let flipped_i = i ^ (1usize << qubit);
            let bit = (i >> qubit) & 1;
            
            if bit == 0 {
                new_amplitudes[i] = cos_half * state.amplitudes[i] - sin_half * state.amplitudes[flipped_i];
            } else {
                new_amplitudes[i] = sin_half * state.amplitudes[i ^ (1usize << qubit)] + cos_half * state.amplitudes[i];
            }
        });
        
        state.amplitudes = new_amplitudes;
        state.fidelity = state.calculate_fidelity();
        Ok(())
    }

    /// Rotation gate around Z-axis
    pub fn rz_gate(state: &mut QuantumState, qubit: usize, angle: f64) -> Result<(), TENGRIError> {
        if qubit >= state.n_qubits {
            return Err(TENGRIError::MathematicalValidationFailed {
                reason: format!("Qubit {} out of range for {}-qubit system", qubit, state.n_qubits),
            });
        }

        let dim = 1usize << state.n_qubits;
        let exp_neg = C64::new(0.0, -angle / 2.0).exp();
        let exp_pos = C64::new(0.0, angle / 2.0).exp();
        
        (0..dim).into_par_iter().for_each(|i| {
            let bit = (i >> qubit) & 1;
            if bit == 0 {
                state.amplitudes[i] = exp_neg * state.amplitudes[i];
            } else {
                state.amplitudes[i] = exp_pos * state.amplitudes[i];
            }
        });
        
        state.fidelity = state.calculate_fidelity();
        Ok(())
    }

    /// Toffoli gate (CCX - controlled-controlled-X)
    pub fn toffoli_gate(state: &mut QuantumState, control1: usize, control2: usize, target: usize) -> Result<(), TENGRIError> {
        if control1 >= state.n_qubits || control2 >= state.n_qubits || target >= state.n_qubits {
            return Err(TENGRIError::MathematicalValidationFailed {
                reason: format!("Qubit indices out of range for {}-qubit system", state.n_qubits),
            });
        }

        let dim = 1usize << state.n_qubits;
        let mut new_amplitudes = state.amplitudes.clone();
        
        (0..dim).into_par_iter().for_each(|i| {
            let control1_bit = (i >> control1) & 1;
            let control2_bit = (i >> control2) & 1;
            
            if control1_bit == 1 && control2_bit == 1 {
                let flipped_i = i ^ (1usize << target);
                new_amplitudes[i] = state.amplitudes[flipped_i];
            }
        });
        
        state.amplitudes = new_amplitudes;
        state.fidelity = state.calculate_fidelity();
        Ok(())
    }
}

/// Quantum circuit for financial ML applications
pub struct QuantumCircuit {
    pub n_qubits: usize,
    pub gates: Vec<QuantumGateOp>,
}

#[derive(Debug, Clone)]
pub enum QuantumGateOp {
    X(usize),
    Y(usize),
    Z(usize),
    H(usize),
    CNOT(usize, usize),
    RX(usize, f64),
    RY(usize, f64),
    RZ(usize, f64),
    Toffoli(usize, usize, usize),
}

impl QuantumCircuit {
    /// Create new quantum circuit
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            gates: Vec::new(),
        }
    }

    /// Add gate to circuit
    pub fn add_gate(&mut self, gate: QuantumGateOp) {
        self.gates.push(gate);
    }

    /// Execute circuit on quantum state
    pub fn execute(&self, state: &mut QuantumState) -> Result<(), TENGRIError> {
        let start_time = std::time::Instant::now();
        
        for gate in &self.gates {
            match gate {
                QuantumGateOp::X(q) => QuantumGates::x_gate(state, *q)?,
                QuantumGateOp::Y(q) => QuantumGates::y_gate(state, *q)?,
                QuantumGateOp::Z(q) => QuantumGates::z_gate(state, *q)?,
                QuantumGateOp::H(q) => QuantumGates::h_gate(state, *q)?,
                QuantumGateOp::CNOT(c, t) => QuantumGates::cnot_gate(state, *c, *t)?,
                QuantumGateOp::RX(q, angle) => QuantumGates::rx_gate(state, *q, *angle)?,
                QuantumGateOp::RY(q, angle) => QuantumGates::ry_gate(state, *q, *angle)?,
                QuantumGateOp::RZ(q, angle) => QuantumGates::rz_gate(state, *q, *angle)?,
                QuantumGateOp::Toffoli(c1, c2, t) => QuantumGates::toffoli_gate(state, *c1, *c2, *t)?,
            }
        }

        let execution_time = start_time.elapsed();
        
        // Log performance for sub-100μs target
        if execution_time.as_micros() > 50 {
            tracing::warn!(
                "Quantum circuit execution time: {}μs (target: <100μs)",
                execution_time.as_micros()
            );
        }

        Ok(())
    }

    /// Create quantum feature encoding circuit for financial data
    pub fn create_feature_encoding_circuit(n_features: usize) -> Self {
        let n_qubits = (n_features as f64).log2().ceil() as usize;
        let mut circuit = Self::new(n_qubits);
        
        // Create superposition for feature encoding
        for i in 0..n_qubits {
            circuit.add_gate(QuantumGateOp::H(i));
        }
        
        // Add entanglement for feature correlation
        for i in 0..n_qubits-1 {
            circuit.add_gate(QuantumGateOp::CNOT(i, i+1));
        }
        
        circuit
    }

    /// Create quantum amplitude encoding for temporal data
    pub fn create_temporal_encoding_circuit(sequence_length: usize) -> Self {
        let n_qubits = (sequence_length as f64).log2().ceil() as usize;
        let mut circuit = Self::new(n_qubits);
        
        // Temporal pattern encoding with rotation gates
        for i in 0..n_qubits {
            let angle = 2.0 * PI * (i as f64) / (n_qubits as f64);
            circuit.add_gate(QuantumGateOp::RY(i, angle));
        }
        
        // Add temporal correlations
        for i in 0..n_qubits-1 {
            circuit.add_gate(QuantumGateOp::CNOT(i, (i+1) % n_qubits));
        }
        
        circuit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::new(2);
        assert_eq!(state.n_qubits, 2);
        assert_eq!(state.amplitudes.len(), 4);
        assert_abs_diff_eq!(state.amplitudes[0].norm(), 1.0);
        assert_abs_diff_eq!(state.fidelity, 1.0);
    }

    #[test]
    fn test_hadamard_gate() {
        let mut state = QuantumState::new(1);
        QuantumGates::h_gate(&mut state, 0).unwrap();
        
        // After H gate, should be in superposition |+⟩ = (|0⟩ + |1⟩)/√2
        assert_abs_diff_eq!(state.amplitudes[0].norm(), 1.0/2.0_f64.sqrt());
        assert_abs_diff_eq!(state.amplitudes[1].norm(), 1.0/2.0_f64.sqrt());
    }

    #[test]
    fn test_cnot_gate() {
        let mut state = QuantumState::new(2);
        QuantumGates::h_gate(&mut state, 0).unwrap();
        QuantumGates::cnot_gate(&mut state, 0, 1).unwrap();
        
        // Should create Bell state (|00⟩ + |11⟩)/√2
        assert_abs_diff_eq!(state.amplitudes[0].norm(), 1.0/2.0_f64.sqrt());
        assert_abs_diff_eq!(state.amplitudes[3].norm(), 1.0/2.0_f64.sqrt());
        assert_abs_diff_eq!(state.amplitudes[1].norm(), 0.0);
        assert_abs_diff_eq!(state.amplitudes[2].norm(), 0.0);
    }

    #[test]
    fn test_quantum_circuit_execution() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGateOp::H(0));
        circuit.add_gate(QuantumGateOp::CNOT(0, 1));
        
        let mut state = QuantumState::new(2);
        circuit.execute(&mut state).unwrap();
        
        // Should create Bell state
        assert_abs_diff_eq!(state.amplitudes[0].norm(), 1.0/2.0_f64.sqrt());
        assert_abs_diff_eq!(state.amplitudes[3].norm(), 1.0/2.0_f64.sqrt());
    }

    #[test]
    fn test_feature_encoding_circuit() {
        let circuit = QuantumCircuit::create_feature_encoding_circuit(4);
        assert_eq!(circuit.n_qubits, 2);
        assert!(circuit.gates.len() > 0);
    }

    #[test]
    fn test_temporal_encoding_circuit() {
        let circuit = QuantumCircuit::create_temporal_encoding_circuit(8);
        assert_eq!(circuit.n_qubits, 3);
        assert!(circuit.gates.len() > 0);
    }
}