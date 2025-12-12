//! Quantum Gates Implementation
//! 
//! Native Rust implementation of quantum gates for quantum ML
//! Replaces external quantum libraries with optimized Rust code

use nalgebra::{DMatrix};
use num_complex::Complex64;
use std::f64::consts::PI;
use crate::{QuantumState, QuantumMLError};

/// Quantum gate operations
pub struct QuantumGates;

impl QuantumGates {
    /// Pauli-X gate (NOT gate)
    pub fn pauli_x() -> DMatrix<Complex64> {
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        ])
    }

    /// Pauli-Y gate
    pub fn pauli_y() -> DMatrix<Complex64> {
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0),
        ])
    }

    /// Pauli-Z gate
    pub fn pauli_z() -> DMatrix<Complex64> {
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
        ])
    }

    /// Hadamard gate
    pub fn hadamard() -> DMatrix<Complex64> {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(inv_sqrt2, 0.0), Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0), Complex64::new(-inv_sqrt2, 0.0),
        ])
    }

    /// Phase gate (S gate)
    pub fn phase() -> DMatrix<Complex64> {
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0),
        ])
    }

    /// T gate (π/8 gate)
    pub fn t_gate() -> DMatrix<Complex64> {
        let phase = Complex64::new(0.0, PI / 4.0).exp();
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), phase,
        ])
    }

    /// Rotation X gate
    pub fn rx(theta: f64) -> DMatrix<Complex64> {
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(cos_half, 0.0), Complex64::new(0.0, -sin_half),
            Complex64::new(0.0, -sin_half), Complex64::new(cos_half, 0.0),
        ])
    }

    /// Rotation Y gate
    pub fn ry(theta: f64) -> DMatrix<Complex64> {
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(cos_half, 0.0), Complex64::new(-sin_half, 0.0),
            Complex64::new(sin_half, 0.0), Complex64::new(cos_half, 0.0),
        ])
    }

    /// Rotation Z gate
    pub fn rz(theta: f64) -> DMatrix<Complex64> {
        let exp_neg = Complex64::new(0.0, -theta / 2.0).exp();
        let exp_pos = Complex64::new(0.0, theta / 2.0).exp();
        DMatrix::from_row_slice(2, 2, &[
            exp_neg, Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), exp_pos,
        ])
    }

    /// CNOT gate (2-qubit)
    pub fn cnot() -> DMatrix<Complex64> {
        let mut gate = DMatrix::zeros(4, 4);
        gate[(0, 0)] = Complex64::new(1.0, 0.0);
        gate[(1, 1)] = Complex64::new(1.0, 0.0);
        gate[(2, 3)] = Complex64::new(1.0, 0.0);
        gate[(3, 2)] = Complex64::new(1.0, 0.0);
        gate
    }

    /// CZ gate (2-qubit)
    pub fn cz() -> DMatrix<Complex64> {
        let mut gate = DMatrix::zeros(4, 4);
        gate[(0, 0)] = Complex64::new(1.0, 0.0);
        gate[(1, 1)] = Complex64::new(1.0, 0.0);
        gate[(2, 2)] = Complex64::new(1.0, 0.0);
        gate[(3, 3)] = Complex64::new(-1.0, 0.0);
        gate
    }

    /// Toffoli gate (3-qubit)
    pub fn toffoli() -> DMatrix<Complex64> {
        let mut gate = DMatrix::identity(8, 8);
        gate[(6, 6)] = Complex64::new(0.0, 0.0);
        gate[(6, 7)] = Complex64::new(1.0, 0.0);
        gate[(7, 6)] = Complex64::new(1.0, 0.0);
        gate[(7, 7)] = Complex64::new(0.0, 0.0);
        gate
    }

    /// Apply single-qubit gate to quantum state
    pub fn apply_single_qubit_gate(
        state: &mut QuantumState,
        gate: &DMatrix<Complex64>,
        qubit: usize,
    ) -> Result<(), QuantumMLError> {
        if qubit >= state.n_qubits {
            return Err(QuantumMLError::QuantumGateOperationFailed {
                reason: format!("Qubit index {} out of bounds", qubit),
            });
        }

        let n_states = state.amplitudes.len();
        let mut new_amplitudes = state.amplitudes.clone();

        // Apply gate to each amplitude pair
        for i in 0..n_states {
            if (i >> qubit) & 1 == 0 {
                let j = i | (1 << qubit);
                if j < n_states {
                    let amp0 = state.amplitudes[i];
                    let amp1 = state.amplitudes[j];
                    
                    new_amplitudes[i] = gate[(0, 0)] * amp0 + gate[(0, 1)] * amp1;
                    new_amplitudes[j] = gate[(1, 0)] * amp0 + gate[(1, 1)] * amp1;
                }
            }
        }

        state.amplitudes = new_amplitudes;
        state.normalize();
        Ok(())
    }

    /// Apply two-qubit gate to quantum state
    pub fn apply_two_qubit_gate(
        state: &mut QuantumState,
        gate: &DMatrix<Complex64>,
        control: usize,
        target: usize,
    ) -> Result<(), QuantumMLError> {
        if control >= state.n_qubits || target >= state.n_qubits {
            return Err(QuantumMLError::QuantumGateOperationFailed {
                reason: "Qubit indices out of bounds".to_string(),
            });
        }

        if control == target {
            return Err(QuantumMLError::QuantumGateOperationFailed {
                reason: "Control and target qubits cannot be the same".to_string(),
            });
        }

        let n_states = state.amplitudes.len();
        let mut new_amplitudes = state.amplitudes.clone();

        // Apply gate to each 4-amplitude group
        for i in 0..n_states {
            let control_bit = (i >> control) & 1;
            let target_bit = (i >> target) & 1;
            let basis_state = control_bit * 2 + target_bit;

            if basis_state == 0 {
                let j = i | (1 << target);
                let k = i | (1 << control);
                let l = i | (1 << control) | (1 << target);

                if j < n_states && k < n_states && l < n_states {
                    let amp00 = state.amplitudes[i];
                    let amp01 = state.amplitudes[j];
                    let amp10 = state.amplitudes[k];
                    let amp11 = state.amplitudes[l];

                    new_amplitudes[i] = gate[(0, 0)] * amp00 + gate[(0, 1)] * amp01 +
                                       gate[(0, 2)] * amp10 + gate[(0, 3)] * amp11;
                    new_amplitudes[j] = gate[(1, 0)] * amp00 + gate[(1, 1)] * amp01 +
                                       gate[(1, 2)] * amp10 + gate[(1, 3)] * amp11;
                    new_amplitudes[k] = gate[(2, 0)] * amp00 + gate[(2, 1)] * amp01 +
                                       gate[(2, 2)] * amp10 + gate[(2, 3)] * amp11;
                    new_amplitudes[l] = gate[(3, 0)] * amp00 + gate[(3, 1)] * amp01 +
                                       gate[(3, 2)] * amp10 + gate[(3, 3)] * amp11;
                }
            }
        }

        state.amplitudes = new_amplitudes;
        state.normalize();
        Ok(())
    }

    /// Create parameterized quantum circuit for ML
    pub fn create_variational_circuit(
        state: &mut QuantumState,
        parameters: &[f64],
    ) -> Result<(), QuantumMLError> {
        if parameters.len() < state.n_qubits * 3 {
            return Err(QuantumMLError::QuantumGateOperationFailed {
                reason: "Not enough parameters for variational circuit".to_string(),
            });
        }

        let mut param_idx = 0;

        // Layer 1: RY rotations
        for qubit in 0..state.n_qubits {
            let gate = Self::ry(parameters[param_idx]);
            Self::apply_single_qubit_gate(state, &gate, qubit)?;
            param_idx += 1;
        }

        // Layer 2: Entangling gates
        for qubit in 0..state.n_qubits - 1 {
            let cnot_gate = Self::cnot();
            Self::apply_two_qubit_gate(state, &cnot_gate, qubit, qubit + 1)?;
        }

        // Layer 3: RZ rotations
        for qubit in 0..state.n_qubits {
            let gate = Self::rz(parameters[param_idx]);
            Self::apply_single_qubit_gate(state, &gate, qubit)?;
            param_idx += 1;
        }

        // Layer 4: More entangling gates
        for qubit in 0..state.n_qubits - 1 {
            let cnot_gate = Self::cnot();
            Self::apply_two_qubit_gate(state, &cnot_gate, qubit + 1, qubit)?;
        }

        // Layer 5: Final RY rotations
        for qubit in 0..state.n_qubits {
            let gate = Self::ry(parameters[param_idx]);
            Self::apply_single_qubit_gate(state, &gate, qubit)?;
            param_idx += 1;
        }

        Ok(())
    }

    /// Measure quantum state (partial measurement)
    pub fn measure_qubit(
        state: &mut QuantumState,
        qubit: usize,
        rng: &mut impl rand::Rng,
    ) -> Result<bool, QuantumMLError> {
        if qubit >= state.n_qubits {
            return Err(QuantumMLError::QuantumGateOperationFailed {
                reason: format!("Qubit index {} out of bounds", qubit),
            });
        }

        // Calculate probability of measuring |1⟩
        let mut prob_one = 0.0;
        for i in 0..state.amplitudes.len() {
            if (i >> qubit) & 1 == 1 {
                prob_one += state.amplitudes[i].norm_sqr();
            }
        }

        // Perform measurement
        let measurement_result = rng.gen::<f64>() < prob_one;

        // Collapse state
        let mut new_amplitudes = state.amplitudes.clone();
        let norm_factor = if measurement_result { prob_one.sqrt() } else { (1.0 - prob_one).sqrt() };

        for i in 0..state.amplitudes.len() {
            let bit_value = (i >> qubit) & 1 == 1;
            if bit_value != measurement_result {
                new_amplitudes[i] = Complex64::new(0.0, 0.0);
            } else {
                new_amplitudes[i] /= norm_factor;
            }
        }

        state.amplitudes = new_amplitudes;
        Ok(measurement_result)
    }

    /// Calculate expectation value of Pauli operator
    pub fn expectation_value_pauli(
        state: &QuantumState,
        pauli_string: &str,
    ) -> Result<f64, QuantumMLError> {
        if pauli_string.len() != state.n_qubits {
            return Err(QuantumMLError::QuantumGateOperationFailed {
                reason: "Pauli string length must match number of qubits".to_string(),
            });
        }

        let mut expectation = 0.0;

        for (i, amplitude) in state.amplitudes.iter().enumerate() {
            let mut coefficient = 1.0;
            let mut phase = 0.0;

            for (qubit, pauli_op) in pauli_string.chars().enumerate() {
                let bit = (i >> qubit) & 1;
                
                match pauli_op {
                    'I' => {}, // Identity, no effect
                    'X' => {
                        // X flips the bit, so we need to look at the flipped state
                        let flipped_i = i ^ (1 << qubit);
                        if flipped_i < state.amplitudes.len() {
                            coefficient *= state.amplitudes[flipped_i].norm_sqr();
                        }
                    },
                    'Y' => {
                        // Y flips the bit and adds phase
                        let flipped_i = i ^ (1 << qubit);
                        if flipped_i < state.amplitudes.len() {
                            coefficient *= state.amplitudes[flipped_i].norm_sqr();
                            phase += if bit == 0 { PI / 2.0 } else { -PI / 2.0 };
                        }
                    },
                    'Z' => {
                        // Z adds phase based on bit value
                        coefficient *= if bit == 0 { 1.0 } else { -1.0 };
                    },
                    _ => {
                        return Err(QuantumMLError::QuantumGateOperationFailed {
                            reason: format!("Invalid Pauli operator: {}", pauli_op),
                        });
                    }
                }
            }

            expectation += amplitude.norm_sqr() * coefficient * phase.cos();
        }

        Ok(expectation)
    }

    /// Create quantum feature map for data encoding
    pub fn create_feature_map(
        state: &mut QuantumState,
        features: &[f64],
    ) -> Result<(), QuantumMLError> {
        if features.len() > state.n_qubits {
            return Err(QuantumMLError::QuantumGateOperationFailed {
                reason: "More features than qubits available".to_string(),
            });
        }

        // Reset to |0⟩ state
        state.amplitudes.fill(Complex64::new(0.0, 0.0));
        state.amplitudes[0] = Complex64::new(1.0, 0.0);

        // Apply feature-dependent rotations
        for (i, &feature) in features.iter().enumerate() {
            if i < state.n_qubits {
                // Scale feature to [0, 2π] range
                let angle = feature * 2.0 * PI;
                let ry_gate = Self::ry(angle);
                Self::apply_single_qubit_gate(state, &ry_gate, i)?;
            }
        }

        // Add entanglement for expressivity
        for i in 0..state.n_qubits - 1 {
            let cnot_gate = Self::cnot();
            Self::apply_two_qubit_gate(state, &cnot_gate, i, i + 1)?;
        }

        // Additional feature-dependent phase rotations
        for (i, &feature) in features.iter().enumerate() {
            if i < state.n_qubits {
                let angle = feature * feature * PI; // Nonlinear encoding
                let rz_gate = Self::rz(angle);
                Self::apply_single_qubit_gate(state, &rz_gate, i)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_pauli_gates() {
        let x_gate = QuantumGates::pauli_x();
        let y_gate = QuantumGates::pauli_y();
        let z_gate = QuantumGates::pauli_z();

        assert_eq!(x_gate.nrows(), 2);
        assert_eq!(x_gate.ncols(), 2);
        assert_eq!(y_gate.nrows(), 2);
        assert_eq!(y_gate.ncols(), 2);
        assert_eq!(z_gate.nrows(), 2);
        assert_eq!(z_gate.ncols(), 2);
    }

    #[test]
    fn test_hadamard_gate() {
        let h_gate = QuantumGates::hadamard();
        assert_eq!(h_gate.nrows(), 2);
        assert_eq!(h_gate.ncols(), 2);
        
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        assert_abs_diff_eq!(h_gate[(0, 0)].re, inv_sqrt2, epsilon = 1e-10);
    }

    #[test]
    fn test_single_qubit_gate_application() {
        let mut state = QuantumState::new(2);
        let x_gate = QuantumGates::pauli_x();
        
        let result = QuantumGates::apply_single_qubit_gate(&mut state, &x_gate, 0);
        assert!(result.is_ok());
        
        // After X gate on first qubit, state should be |10⟩
        assert_abs_diff_eq!(state.amplitudes[0].norm_sqr(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state.amplitudes[1].norm_sqr(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cnot_gate() {
        let mut state = QuantumState::new(2);
        let h_gate = QuantumGates::hadamard();
        let cnot_gate = QuantumGates::cnot();
        
        // Create superposition on first qubit
        let _ = QuantumGates::apply_single_qubit_gate(&mut state, &h_gate, 0);
        
        // Apply CNOT
        let result = QuantumGates::apply_two_qubit_gate(&mut state, &cnot_gate, 0, 1);
        assert!(result.is_ok());
        
        // Should create Bell state |00⟩ + |11⟩
        assert_abs_diff_eq!(state.amplitudes[0].norm_sqr(), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(state.amplitudes[1].norm_sqr(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state.amplitudes[2].norm_sqr(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state.amplitudes[3].norm_sqr(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_feature_map() {
        let mut state = QuantumState::new(3);
        let features = vec![0.5, -0.3, 0.8];
        
        let result = QuantumGates::create_feature_map(&mut state, &features);
        assert!(result.is_ok());
        
        // State should be normalized
        let norm_squared: f64 = state.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        assert_abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_variational_circuit() {
        let mut state = QuantumState::new(2);
        let parameters = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        
        let result = QuantumGates::create_variational_circuit(&mut state, &parameters);
        assert!(result.is_ok());
        
        // State should be normalized
        let norm_squared: f64 = state.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        assert_abs_diff_eq!(norm_squared, 1.0, epsilon = 1e-10);
    }
}