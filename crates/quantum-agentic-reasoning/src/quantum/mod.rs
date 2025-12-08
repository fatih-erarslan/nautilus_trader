//! Quantum computing modules for Quantum Agentic Reasoning
//!
//! This module contains all quantum circuit implementations and quantum computing
//! logic for the QAR system.

use crate::core::{QarResult, QarError};

pub mod circuits;
pub mod gates;
pub mod backend;
pub mod optimization;
pub mod fourier;
pub mod pattern;
pub mod decision;
pub mod types;
pub mod traits;

// Re-export commonly used types
pub use circuits::*;
pub use gates::*;
pub use backend::*;
pub use optimization::*;
pub use fourier::*;
pub use pattern::*;
pub use decision::*;
pub use types::*;
pub use traits::*;

// Re-export QuantumCircuit struct for compatibility
pub use crate::core::CoreQuantumCircuit as QuantumCircuit;

/// Quantum state representation
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// Amplitudes for each basis state
    pub amplitudes: Vec<num_complex::Complex64>,
    /// Number of qubits
    pub num_qubits: usize,
}

impl QuantumState {
    /// Create a new quantum state with all qubits in |0⟩
    pub fn new(num_qubits: usize) -> Self {
        let size = 1 << num_qubits;
        let mut amplitudes = vec![num_complex::Complex64::new(0.0, 0.0); size];
        amplitudes[0] = num_complex::Complex64::new(1.0, 0.0); // |00...0⟩
        
        Self {
            amplitudes,
            num_qubits,
        }
    }

    /// Create a superposition state
    pub fn superposition(num_qubits: usize) -> Self {
        let size = 1 << num_qubits;
        let amplitude = num_complex::Complex64::new(1.0 / (size as f64).sqrt(), 0.0);
        let amplitudes = vec![amplitude; size];
        
        Self {
            amplitudes,
            num_qubits,
        }
    }

    /// Get the probability of measuring a specific state
    pub fn probability(&self, state: usize) -> f64 {
        if state >= self.amplitudes.len() {
            0.0
        } else {
            self.amplitudes[state].norm_sqr()
        }
    }

    /// Get all probabilities
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|amp| amp.norm_sqr()).collect()
    }

    /// Normalize the quantum state
    pub fn normalize(&mut self) {
        let norm = self.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .sum::<f64>()
            .sqrt();
        
        if norm > 0.0 {
            for amp in &mut self.amplitudes {
                *amp /= norm;
            }
        }
    }

    /// Check if the state is normalized
    pub fn is_normalized(&self) -> bool {
        let norm = self.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .sum::<f64>();
        (norm - 1.0).abs() < 1e-10
    }

    /// Apply a single-qubit gate
    pub fn apply_single_qubit_gate(&mut self, qubit: usize, gate: &Gate) -> QarResult<()> {
        if qubit >= self.num_qubits {
            return Err(QarError::QuantumError(format!("Qubit {} out of range", qubit)));
        }

        let matrix = gate.matrix();
        let qubit_mask = 1 << qubit;
        
        for i in 0..self.amplitudes.len() {
            if i & qubit_mask == 0 {
                // This is a |0⟩ state for this qubit
                let j = i | qubit_mask; // Corresponding |1⟩ state
                
                let amp_0 = self.amplitudes[i];
                let amp_1 = self.amplitudes[j];
                
                self.amplitudes[i] = matrix[0][0] * amp_0 + matrix[0][1] * amp_1;
                self.amplitudes[j] = matrix[1][0] * amp_0 + matrix[1][1] * amp_1;
            }
        }

        Ok(())
    }

    /// Apply a two-qubit gate
    pub fn apply_two_qubit_gate(&mut self, control: usize, target: usize, gate: &Gate) -> QarResult<()> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(QarError::QuantumError("Qubit out of range".to_string()));
        }
        
        if control == target {
            return Err(QarError::QuantumError("Control and target qubits must be different".to_string()));
        }

        let control_mask = 1 << control;
        let target_mask = 1 << target;
        let matrix = gate.matrix();
        
        for i in 0..self.amplitudes.len() {
            if i & control_mask != 0 && i & target_mask == 0 {
                // Control is |1⟩, target is |0⟩
                let j = i | target_mask; // Corresponding state with target |1⟩
                
                let amp_0 = self.amplitudes[i];
                let amp_1 = self.amplitudes[j];
                
                self.amplitudes[i] = matrix[0][0] * amp_0 + matrix[0][1] * amp_1;
                self.amplitudes[j] = matrix[1][0] * amp_0 + matrix[1][1] * amp_1;
            }
        }

        Ok(())
    }

    /// Measure a single qubit
    pub fn measure_qubit(&mut self, qubit: usize) -> QarResult<bool> {
        if qubit >= self.num_qubits {
            return Err(QarError::QuantumError(format!("Qubit {} out of range", qubit)));
        }

        let qubit_mask = 1 << qubit;
        let mut prob_0 = 0.0;
        let mut prob_1 = 0.0;

        for (i, amp) in self.amplitudes.iter().enumerate() {
            if i & qubit_mask == 0 {
                prob_0 += amp.norm_sqr();
            } else {
                prob_1 += amp.norm_sqr();
            }
        }

        // Simulate measurement
        let random_value: f64 = rand::random();
        let measured_one = random_value > prob_0;

        // Collapse the state
        for (i, amp) in self.amplitudes.iter_mut().enumerate() {
            if (i & qubit_mask != 0) != measured_one {
                *amp = num_complex::Complex64::new(0.0, 0.0);
            }
        }

        // Renormalize
        self.normalize();

        Ok(measured_one)
    }

    /// Calculate expectation value of Pauli-Z on a qubit
    pub fn expectation_z(&self, qubit: usize) -> QarResult<f64> {
        if qubit >= self.num_qubits {
            return Err(QarError::QuantumError(format!("Qubit {} out of range", qubit)));
        }

        let qubit_mask = 1 << qubit;
        let mut expectation = 0.0;

        for (i, amp) in self.amplitudes.iter().enumerate() {
            let prob = amp.norm_sqr();
            if i & qubit_mask == 0 {
                expectation += prob; // |0⟩ state contributes +1
            } else {
                expectation -= prob; // |1⟩ state contributes -1
            }
        }

        Ok(expectation)
    }

    /// Calculate expectation value of Pauli-X on a qubit
    pub fn expectation_x(&self, qubit: usize) -> QarResult<f64> {
        if qubit >= self.num_qubits {
            return Err(QarError::QuantumError(format!("Qubit {} out of range", qubit)));
        }

        let qubit_mask = 1 << qubit;
        let mut expectation = 0.0;

        for (i, amp) in self.amplitudes.iter().enumerate() {
            let j = i ^ qubit_mask; // Flip the qubit
            expectation += 2.0 * (amp.conj() * self.amplitudes[j]).re;
        }

        Ok(expectation)
    }

    /// Calculate expectation value of Pauli-Y on a qubit
    pub fn expectation_y(&self, qubit: usize) -> QarResult<f64> {
        if qubit >= self.num_qubits {
            return Err(QarError::QuantumError(format!("Qubit {} out of range", qubit)));
        }

        let qubit_mask = 1 << qubit;
        let mut expectation = 0.0;

        for (i, amp) in self.amplitudes.iter().enumerate() {
            let j = i ^ qubit_mask; // Flip the qubit
            let sign = if i & qubit_mask == 0 { -1.0 } else { 1.0 };
            expectation += 2.0 * sign * (amp.conj() * self.amplitudes[j]).im;
        }

        Ok(expectation)
    }

    /// Get the fidelity with another quantum state
    pub fn fidelity(&self, other: &QuantumState) -> QarResult<f64> {
        if self.num_qubits != other.num_qubits {
            return Err(QarError::QuantumError("States must have the same number of qubits".to_string()));
        }

        let overlap = self.amplitudes.iter()
            .zip(other.amplitudes.iter())
            .map(|(a, b)| a.conj() * b)
            .sum::<num_complex::Complex64>();

        Ok(overlap.norm_sqr())
    }

    /// Convert to a density matrix representation
    pub fn to_density_matrix(&self) -> Vec<Vec<num_complex::Complex64>> {
        let size = self.amplitudes.len();
        let mut density_matrix = vec![vec![num_complex::Complex64::new(0.0, 0.0); size]; size];

        for i in 0..size {
            for j in 0..size {
                density_matrix[i][j] = self.amplitudes[i] * self.amplitudes[j].conj();
            }
        }

        density_matrix
    }

    /// Calculate the von Neumann entropy
    pub fn von_neumann_entropy(&self) -> f64 {
        let probabilities = self.probabilities();
        let mut entropy = 0.0;

        for &prob in &probabilities {
            if prob > 1e-10 {
                entropy -= prob * prob.ln();
            }
        }

        entropy / 2.0_f64.ln() // Convert to bits
    }

    /// Get the most likely measurement outcome
    pub fn most_likely_outcome(&self) -> (usize, f64) {
        let probabilities = self.probabilities();
        let mut max_prob = 0.0;
        let mut max_index = 0;

        for (i, &prob) in probabilities.iter().enumerate() {
            if prob > max_prob {
                max_prob = prob;
                max_index = i;
            }
        }

        (max_index, max_prob)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::new(2);
        assert_eq!(state.num_qubits, 2);
        assert_eq!(state.amplitudes.len(), 4);
        assert_eq!(state.probability(0), 1.0);
        assert_eq!(state.probability(1), 0.0);
    }

    #[test]
    fn test_superposition_state() {
        let state = QuantumState::superposition(2);
        assert_eq!(state.num_qubits, 2);
        assert_eq!(state.amplitudes.len(), 4);
        
        // All states should have equal probability
        for i in 0..4 {
            assert!((state.probability(i) - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normalization() {
        let mut state = QuantumState::new(2);
        state.amplitudes[0] = num_complex::Complex64::new(0.5, 0.0);
        state.amplitudes[1] = num_complex::Complex64::new(0.5, 0.0);
        state.amplitudes[2] = num_complex::Complex64::new(0.5, 0.0);
        state.amplitudes[3] = num_complex::Complex64::new(0.5, 0.0);
        
        assert!(!state.is_normalized());
        state.normalize();
        assert!(state.is_normalized());
    }

    #[test]
    fn test_expectation_values() {
        let state = QuantumState::new(1);
        
        // |0⟩ state should have Z expectation = +1
        assert!((state.expectation_z(0).unwrap() - 1.0).abs() < 1e-10);
        
        // X expectation should be 0 for |0⟩
        assert!(state.expectation_x(0).unwrap().abs() < 1e-10);
    }

    #[test]
    fn test_fidelity() {
        let state1 = QuantumState::new(2);
        let state2 = QuantumState::new(2);
        
        // Same states should have fidelity 1
        assert!((state1.fidelity(&state2).unwrap() - 1.0).abs() < 1e-10);
        
        let state3 = QuantumState::superposition(2);
        
        // Different states should have fidelity < 1
        assert!(state1.fidelity(&state3).unwrap() < 1.0);
    }

    #[test]
    fn test_von_neumann_entropy() {
        let state = QuantumState::new(2);
        // Pure state should have entropy 0
        assert!(state.von_neumann_entropy().abs() < 1e-10);
        
        let superposition = QuantumState::superposition(2);
        // Superposition should have positive entropy
        assert!(superposition.von_neumann_entropy() > 0.0);
    }

    #[test]
    fn test_most_likely_outcome() {
        let state = QuantumState::new(2);
        let (outcome, prob) = state.most_likely_outcome();
        
        assert_eq!(outcome, 0);
        assert_eq!(prob, 1.0);
    }
}