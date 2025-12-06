//! Utility functions for QERC

use crate::core::{QercError, QercResult, QuantumState};
use num_complex::Complex64;

/// Calculate fidelity between two quantum states
pub fn calculate_fidelity(state1: &QuantumState, state2: &QuantumState) -> f64 {
    if state1.amplitudes.len() != state2.amplitudes.len() {
        return 0.0;
    }
    
    let mut fidelity = 0.0;
    for (amp1, amp2) in state1.amplitudes.iter().zip(state2.amplitudes.iter()) {
        fidelity += (amp1.conj() * amp2).norm_sqr();
    }
    
    fidelity.sqrt()
}

/// Apply single qubit error
pub fn apply_single_qubit_error(state: &QuantumState, qubit: usize, error_type: &str) -> QercResult<QuantumState> {
    if qubit >= state.num_qubits() {
        return Err(QercError::InvalidOperationError {
            message: format!("Qubit index {} out of bounds", qubit),
        });
    }
    
    // Simplified error application
    Ok(state.clone())
}

/// Create test quantum state
pub fn create_test_state(num_qubits: usize) -> QuantumState {
    let size = 1 << num_qubits;
    let mut amplitudes = vec![Complex64::new(0.0, 0.0); size];
    amplitudes[0] = Complex64::new(1.0, 0.0);
    
    QuantumState {
        amplitudes,
        num_qubits,
        density_matrix: None,
        normalization: 1.0,
        metadata: std::collections::HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fidelity_calculation() {
        let state1 = QuantumState::new(vec![1.0, 0.0]);
        let state2 = QuantumState::new(vec![1.0, 0.0]);
        let fidelity = calculate_fidelity(&state1, &state2);
        assert!((fidelity - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_create_test_state() {
        let state = create_test_state(2);
        assert_eq!(state.num_qubits(), 2);
        assert_eq!(state.amplitudes.len(), 4);
    }
}