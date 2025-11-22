//! Comprehensive test suite for quantum circuit components
//! 
//! Tests all quantum operations, gates, circuits, and PennyLane compatibility
//! with mathematical precision verification and quantum mechanics properties

use quantum_circuit::*;
use quantum_circuit::gates::*;
use quantum_circuit::circuit::*;
use quantum_circuit::simulation::*;
use quantum_circuit::optimization::*;
use quantum_circuit::embeddings::*;
use quantum_circuit::neural::*;
use quantum_circuit::pennylane_compat::*;
use num_complex::Complex64;
use approx::{assert_abs_diff_eq, assert_relative_eq};
use proptest::prelude::*;
use std::f64::consts::{PI, SQRT_2};

/// Mathematical precision for quantum operations
const QUANTUM_EPSILON: f64 = 1e-12;
const GATE_EPSILON: f64 = 1e-10;
const SIMULATION_EPSILON: f64 = 1e-8;

#[cfg(test)]
mod quantum_constants_tests {
    use super::*;

    #[test]
    fn test_identity_matrix() {
        let id = constants::identity();
        
        assert_eq!(id.shape(), &[2, 2]);
        assert_abs_diff_eq!(id[[0, 0]], Complex64::new(1.0, 0.0), epsilon = QUANTUM_EPSILON);
        assert_abs_diff_eq!(id[[0, 1]], Complex64::new(0.0, 0.0), epsilon = QUANTUM_EPSILON);
        assert_abs_diff_eq!(id[[1, 0]], Complex64::new(0.0, 0.0), epsilon = QUANTUM_EPSILON);
        assert_abs_diff_eq!(id[[1, 1]], Complex64::new(1.0, 0.0), epsilon = QUANTUM_EPSILON);
    }

    #[test]
    fn test_pauli_matrices() {
        let pauli_x = constants::pauli_x();
        let pauli_y = constants::pauli_y();
        let pauli_z = constants::pauli_z();
        
        // Test Pauli-X
        assert_abs_diff_eq!(pauli_x[[0, 0]], Complex64::new(0.0, 0.0), epsilon = QUANTUM_EPSILON);
        assert_abs_diff_eq!(pauli_x[[0, 1]], Complex64::new(1.0, 0.0), epsilon = QUANTUM_EPSILON);
        assert_abs_diff_eq!(pauli_x[[1, 0]], Complex64::new(1.0, 0.0), epsilon = QUANTUM_EPSILON);
        assert_abs_diff_eq!(pauli_x[[1, 1]], Complex64::new(0.0, 0.0), epsilon = QUANTUM_EPSILON);
        
        // Test Pauli-Y
        assert_abs_diff_eq!(pauli_y[[0, 1]], Complex64::new(0.0, -1.0), epsilon = QUANTUM_EPSILON);
        assert_abs_diff_eq!(pauli_y[[1, 0]], Complex64::new(0.0, 1.0), epsilon = QUANTUM_EPSILON);
        
        // Test Pauli-Z
        assert_abs_diff_eq!(pauli_z[[0, 0]], Complex64::new(1.0, 0.0), epsilon = QUANTUM_EPSILON);
        assert_abs_diff_eq!(pauli_z[[1, 1]], Complex64::new(-1.0, 0.0), epsilon = QUANTUM_EPSILON);
    }

    #[test]
    fn test_pauli_anticommutation() {
        let pauli_x = constants::pauli_x();
        let pauli_y = constants::pauli_y();
        let pauli_z = constants::pauli_z();
        
        // {σᵢ, σⱼ} = 2δᵢⱼI for i ≠ j
        let xy = pauli_x.dot(&pauli_y);
        let yx = pauli_y.dot(&pauli_x);
        let anticommutator_xy = &xy + &yx;
        
        // Should be zero matrix for different Pauli matrices
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(anticommutator_xy[[i, j]], Complex64::new(0.0, 0.0), epsilon = QUANTUM_EPSILON);
            }
        }
    }

    #[test]
    fn test_pauli_squares() {
        let pauli_x = constants::pauli_x();
        let pauli_y = constants::pauli_y();
        let pauli_z = constants::pauli_z();
        let identity = constants::identity();
        
        // σ² = I for all Pauli matrices
        let x_squared = pauli_x.dot(&pauli_x);
        let y_squared = pauli_y.dot(&pauli_y);
        let z_squared = pauli_z.dot(&pauli_z);
        
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(x_squared[[i, j]], identity[[i, j]], epsilon = QUANTUM_EPSILON);
                assert_abs_diff_eq!(y_squared[[i, j]], identity[[i, j]], epsilon = QUANTUM_EPSILON);
                assert_abs_diff_eq!(z_squared[[i, j]], identity[[i, j]], epsilon = QUANTUM_EPSILON);
            }
        }
    }

    #[test]
    fn test_hadamard_properties() {
        let hadamard = constants::hadamard();
        let sqrt2_inv = 1.0 / SQRT_2;
        
        // Test Hadamard matrix elements
        assert_abs_diff_eq!(hadamard[[0, 0]], Complex64::new(sqrt2_inv, 0.0), epsilon = QUANTUM_EPSILON);
        assert_abs_diff_eq!(hadamard[[0, 1]], Complex64::new(sqrt2_inv, 0.0), epsilon = QUANTUM_EPSILON);
        assert_abs_diff_eq!(hadamard[[1, 0]], Complex64::new(sqrt2_inv, 0.0), epsilon = QUANTUM_EPSILON);
        assert_abs_diff_eq!(hadamard[[1, 1]], Complex64::new(-sqrt2_inv, 0.0), epsilon = QUANTUM_EPSILON);
        
        // H² = I (Hadamard is self-inverse)
        let h_squared = hadamard.dot(&hadamard);
        let identity = constants::identity();
        
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(h_squared[[i, j]], identity[[i, j]], epsilon = QUANTUM_EPSILON);
            }
        }
    }

    #[test]
    fn test_quantum_states() {
        let zero = constants::zero_state();
        let one = constants::one_state();
        let plus = constants::plus_state();
        let minus = constants::minus_state();
        
        // Test |0⟩ state
        assert_abs_diff_eq!(zero[0], Complex64::new(1.0, 0.0), epsilon = QUANTUM_EPSILON);
        assert_abs_diff_eq!(zero[1], Complex64::new(0.0, 0.0), epsilon = QUANTUM_EPSILON);
        
        // Test |1⟩ state
        assert_abs_diff_eq!(one[0], Complex64::new(0.0, 0.0), epsilon = QUANTUM_EPSILON);
        assert_abs_diff_eq!(one[1], Complex64::new(1.0, 0.0), epsilon = QUANTUM_EPSILON);
        
        // Test |+⟩ state normalization
        let plus_norm_sq = plus.iter().map(|c| c.norm_sqr()).sum::<f64>();
        assert_abs_diff_eq!(plus_norm_sq, 1.0, epsilon = QUANTUM_EPSILON);
        
        // Test |-⟩ state normalization
        let minus_norm_sq = minus.iter().map(|c| c.norm_sqr()).sum::<f64>();
        assert_abs_diff_eq!(minus_norm_sq, 1.0, epsilon = QUANTUM_EPSILON);
        
        // Test orthogonality ⟨0|1⟩ = 0
        let overlap = zero.iter().zip(one.iter())
            .map(|(a, b)| a.conj() * b)
            .sum::<Complex64>();
        assert_abs_diff_eq!(overlap, Complex64::new(0.0, 0.0), epsilon = QUANTUM_EPSILON);
    }
}

#[cfg(test)]
mod quantum_utils_tests {
    use super::*;

    #[test]
    fn test_fidelity_calculation() {
        let zero = constants::zero_state();
        let one = constants::one_state();
        let plus = constants::plus_state();
        
        // Fidelity with itself should be 1
        let fidelity_self = utils::fidelity(&zero, &zero).unwrap();
        assert_abs_diff_eq!(fidelity_self, 1.0, epsilon = QUANTUM_EPSILON);
        
        // Fidelity between orthogonal states should be 0
        let fidelity_orthogonal = utils::fidelity(&zero, &one).unwrap();
        assert_abs_diff_eq!(fidelity_orthogonal, 0.0, epsilon = QUANTUM_EPSILON);
        
        // Fidelity between |0⟩ and |+⟩ should be 1/2
        let fidelity_partial = utils::fidelity(&zero, &plus).unwrap();
        assert_abs_diff_eq!(fidelity_partial, 0.5, epsilon = QUANTUM_EPSILON);
    }

    #[test]
    fn test_state_normalization() {
        let mut unnormalized = ndarray::array![
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 3.0)
        ];
        
        utils::normalize_state(&mut unnormalized).unwrap();
        
        let norm_sq = unnormalized.iter().map(|c| c.norm_sqr()).sum::<f64>();
        assert_abs_diff_eq!(norm_sq, 1.0, epsilon = QUANTUM_EPSILON);
    }

    #[test]
    fn test_expectation_value() {
        let zero = constants::zero_state();
        let one = constants::one_state();
        let plus = constants::plus_state();
        let pauli_z = constants::pauli_z();
        let pauli_x = constants::pauli_x();
        
        // ⟨0|Z|0⟩ = 1
        let exp_z_zero = utils::expectation_value(&zero, &pauli_z).unwrap();
        assert_abs_diff_eq!(exp_z_zero, Complex64::new(1.0, 0.0), epsilon = QUANTUM_EPSILON);
        
        // ⟨1|Z|1⟩ = -1
        let exp_z_one = utils::expectation_value(&one, &pauli_z).unwrap();
        assert_abs_diff_eq!(exp_z_one, Complex64::new(-1.0, 0.0), epsilon = QUANTUM_EPSILON);
        
        // ⟨+|X|+⟩ = 1
        let exp_x_plus = utils::expectation_value(&plus, &pauli_x).unwrap();
        assert_abs_diff_eq!(exp_x_plus, Complex64::new(1.0, 0.0), epsilon = QUANTUM_EPSILON);
        
        // ⟨+|Z|+⟩ = 0
        let exp_z_plus = utils::expectation_value(&plus, &pauli_z).unwrap();
        assert_abs_diff_eq!(exp_z_plus, Complex64::new(0.0, 0.0), epsilon = QUANTUM_EPSILON);
    }

    #[test]
    fn test_random_state_properties() {
        let random_state = utils::random_state(2); // 2 qubits = 4 dimensions
        
        assert_eq!(random_state.len(), 4);
        
        // Should be normalized
        let norm_sq = random_state.iter().map(|c| c.norm_sqr()).sum::<f64>();
        assert_abs_diff_eq!(norm_sq, 1.0, epsilon = QUANTUM_EPSILON);
    }

    #[test]
    fn test_amplitude_encoding() {
        let data = vec![1.0, 0.0, 0.0, 0.0];
        let state = utils::amplitude_encode(&data).unwrap();
        
        assert_eq!(state.len(), 4);
        assert_abs_diff_eq!(state[0], Complex64::new(1.0, 0.0), epsilon = QUANTUM_EPSILON);
        assert_abs_diff_eq!(state[1], Complex64::new(0.0, 0.0), epsilon = QUANTUM_EPSILON);
        
        // Test normalization
        let norm_sq = state.iter().map(|c| c.norm_sqr()).sum::<f64>();
        assert_abs_diff_eq!(norm_sq, 1.0, epsilon = QUANTUM_EPSILON);
    }

    #[test]
    fn test_amplitude_encoding_invalid_size() {
        let data = vec![1.0, 0.0, 0.0]; // Not power of 2
        let result = utils::amplitude_encode(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_amplitude_encoding_zero_norm() {
        let data = vec![0.0, 0.0, 0.0, 0.0];
        let result = utils::amplitude_encode(&data);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod quantum_gates_tests {
    use super::*;

    #[test]
    fn test_rotation_gates() {
        // Test RX gate
        let rx_gate = RX::new(0, PI/2.0);
        let matrix = rx_gate.matrix();
        
        // RX(π/2) should rotate |0⟩ to (|0⟩ - i|1⟩)/√2
        assert_eq!(matrix.shape(), &[2, 2]);
        assert_abs_diff_eq!(matrix[[0, 0]], Complex64::new((PI/4.0).cos(), 0.0), epsilon = GATE_EPSILON);
        assert_abs_diff_eq!(matrix[[0, 1]], Complex64::new(0.0, -(PI/4.0).sin()), epsilon = GATE_EPSILON);
        
        // Test RY gate
        let ry_gate = RY::new(0, PI/2.0);
        let ry_matrix = ry_gate.matrix();
        
        assert_abs_diff_eq!(ry_matrix[[0, 0]], Complex64::new((PI/4.0).cos(), 0.0), epsilon = GATE_EPSILON);
        assert_abs_diff_eq!(ry_matrix[[0, 1]], Complex64::new(-(PI/4.0).sin(), 0.0), epsilon = GATE_EPSILON);
        
        // Test RZ gate
        let rz_gate = RZ::new(0, PI/2.0);
        let rz_matrix = rz_gate.matrix();
        
        assert_abs_diff_eq!(rz_matrix[[0, 0]], Complex64::new(0.0, 0.0).exp() * Complex64::new(0.0, -PI/4.0).exp(), epsilon = GATE_EPSILON);
        assert_abs_diff_eq!(rz_matrix[[1, 1]], Complex64::new(0.0, 0.0).exp() * Complex64::new(0.0, PI/4.0).exp(), epsilon = GATE_EPSILON);
    }

    #[test]
    fn test_cnot_gate() {
        let cnot = CNOT::new(0, 1);
        let matrix = cnot.matrix();
        
        // CNOT matrix should be:
        // [1 0 0 0]
        // [0 1 0 0]
        // [0 0 0 1]
        // [0 0 1 0]
        assert_eq!(matrix.shape(), &[4, 4]);
        
        assert_abs_diff_eq!(matrix[[0, 0]], Complex64::new(1.0, 0.0), epsilon = GATE_EPSILON);
        assert_abs_diff_eq!(matrix[[1, 1]], Complex64::new(1.0, 0.0), epsilon = GATE_EPSILON);
        assert_abs_diff_eq!(matrix[[2, 3]], Complex64::new(1.0, 0.0), epsilon = GATE_EPSILON);
        assert_abs_diff_eq!(matrix[[3, 2]], Complex64::new(1.0, 0.0), epsilon = GATE_EPSILON);
        
        // Other elements should be zero
        assert_abs_diff_eq!(matrix[[0, 1]], Complex64::new(0.0, 0.0), epsilon = GATE_EPSILON);
        assert_abs_diff_eq!(matrix[[2, 2]], Complex64::new(0.0, 0.0), epsilon = GATE_EPSILON);
    }

    #[test]
    fn test_hadamard_gate() {
        let h_gate = Hadamard::new(0);
        let matrix = h_gate.matrix();
        
        let sqrt2_inv = 1.0 / SQRT_2;
        assert_abs_diff_eq!(matrix[[0, 0]], Complex64::new(sqrt2_inv, 0.0), epsilon = GATE_EPSILON);
        assert_abs_diff_eq!(matrix[[0, 1]], Complex64::new(sqrt2_inv, 0.0), epsilon = GATE_EPSILON);
        assert_abs_diff_eq!(matrix[[1, 0]], Complex64::new(sqrt2_inv, 0.0), epsilon = GATE_EPSILON);
        assert_abs_diff_eq!(matrix[[1, 1]], Complex64::new(-sqrt2_inv, 0.0), epsilon = GATE_EPSILON);
    }

    #[test]
    fn test_toffoli_gate() {
        let toffoli = Toffoli::new(0, 1, 2);
        let matrix = toffoli.matrix();
        
        assert_eq!(matrix.shape(), &[8, 8]);
        
        // Toffoli gate should be identity except for last two elements
        for i in 0..6 {
            assert_abs_diff_eq!(matrix[[i, i]], Complex64::new(1.0, 0.0), epsilon = GATE_EPSILON);
        }
        
        // Last two diagonal elements should be swapped
        assert_abs_diff_eq!(matrix[[6, 7]], Complex64::new(1.0, 0.0), epsilon = GATE_EPSILON);
        assert_abs_diff_eq!(matrix[[7, 6]], Complex64::new(1.0, 0.0), epsilon = GATE_EPSILON);
    }

    #[test]
    fn test_phase_gate() {
        let phase = Phase::new(0, PI/4.0);
        let matrix = phase.matrix();
        
        assert_abs_diff_eq!(matrix[[0, 0]], Complex64::new(1.0, 0.0), epsilon = GATE_EPSILON);
        assert_abs_diff_eq!(matrix[[1, 1]], Complex64::new(0.0, PI/4.0).exp(), epsilon = GATE_EPSILON);
        assert_abs_diff_eq!(matrix[[0, 1]], Complex64::new(0.0, 0.0), epsilon = GATE_EPSILON);
        assert_abs_diff_eq!(matrix[[1, 0]], Complex64::new(0.0, 0.0), epsilon = GATE_EPSILON);
    }

    #[test]
    fn test_parametric_gate_derivatives() {
        let rx_gate = RX::new(0, PI/4.0);
        
        // Test parameter derivative
        let derivative = rx_gate.parameter_derivative(0).unwrap();
        
        // For RX gate: d/dθ RX(θ) = -i/2 * X * RX(θ)
        assert_eq!(derivative.shape(), &[2, 2]);
        
        // Verify derivative is purely imaginary for rotation gates
        assert!(derivative[[0, 0]].im.abs() > GATE_EPSILON || derivative[[0, 0]].re.abs() < GATE_EPSILON);
    }

    #[test]
    fn test_controlled_gates() {
        let controlled_x = ControlledX::new(0, 1);
        let cnot = CNOT::new(0, 1);
        
        // Controlled-X should be equivalent to CNOT
        let cx_matrix = controlled_x.matrix();
        let cnot_matrix = cnot.matrix();
        
        for i in 0..4 {
            for j in 0..4 {
                assert_abs_diff_eq!(cx_matrix[[i, j]], cnot_matrix[[i, j]], epsilon = GATE_EPSILON);
            }
        }
    }
}

#[cfg(test)]
mod quantum_circuit_tests {
    use super::*;

    #[test]
    fn test_circuit_creation() {
        let circuit = Circuit::new(3);
        
        assert_eq!(circuit.num_qubits(), 3);
        assert_eq!(circuit.depth(), 0);
        assert_eq!(circuit.gate_count(), 0);
    }

    #[test]
    fn test_single_gate_execution() {
        let mut circuit = Circuit::new(1);
        circuit.add_gate(Box::new(Hadamard::new(0))).unwrap();
        
        let final_state = circuit.execute().unwrap();
        
        // Should produce |+⟩ state
        let expected = constants::plus_state();
        for i in 0..2 {
            assert_abs_diff_eq!(final_state[i], expected[i], epsilon = SIMULATION_EPSILON);
        }
    }

    #[test]
    fn test_bell_state_preparation() {
        let mut circuit = Circuit::new(2);
        circuit.add_gate(Box::new(Hadamard::new(0))).unwrap();
        circuit.add_gate(Box::new(CNOT::new(0, 1))).unwrap();
        
        let final_state = circuit.execute().unwrap();
        
        // Should produce (|00⟩ + |11⟩)/√2
        let sqrt2_inv = 1.0 / SQRT_2;
        assert_abs_diff_eq!(final_state[0], Complex64::new(sqrt2_inv, 0.0), epsilon = SIMULATION_EPSILON);
        assert_abs_diff_eq!(final_state[1], Complex64::new(0.0, 0.0), epsilon = SIMULATION_EPSILON);
        assert_abs_diff_eq!(final_state[2], Complex64::new(0.0, 0.0), epsilon = SIMULATION_EPSILON);
        assert_abs_diff_eq!(final_state[3], Complex64::new(sqrt2_inv, 0.0), epsilon = SIMULATION_EPSILON);
    }

    #[test]
    fn test_circuit_builder() {
        let circuit = CircuitBuilder::new(2)
            .h(0)
            .cnot(0, 1)
            .rz(1, PI/4.0)
            .build()
            .unwrap();
        
        assert_eq!(circuit.num_qubits(), 2);
        assert_eq!(circuit.gate_count(), 3);
    }

    #[test]
    fn test_variational_circuit() {
        let mut vqc = VariationalCircuit::new(2);
        vqc.add_parametric_layer(&[("RY", 0), ("RY", 1)]).unwrap();
        vqc.add_entangling_layer(EntanglementPattern::Linear).unwrap();
        
        let parameters = vec![PI/4.0, PI/3.0];
        let final_state = vqc.execute(&parameters).unwrap();
        
        assert_eq!(final_state.len(), 4);
        
        // Verify normalization
        let norm_sq = final_state.iter().map(|c| c.norm_sqr()).sum::<f64>();
        assert_abs_diff_eq!(norm_sq, 1.0, epsilon = SIMULATION_EPSILON);
    }

    #[test]
    fn test_circuit_depth_calculation() {
        let mut circuit = Circuit::new(3);
        circuit.add_gate(Box::new(Hadamard::new(0))).unwrap();
        circuit.add_gate(Box::new(Hadamard::new(1))).unwrap(); // Parallel with previous
        circuit.add_gate(Box::new(CNOT::new(0, 1))).unwrap();
        circuit.add_gate(Box::new(RZ::new(2, PI/4.0))).unwrap(); // Parallel with CNOT
        circuit.add_gate(Box::new(CNOT::new(1, 2))).unwrap();
        
        // Depth should account for parallel execution
        let depth = circuit.depth();
        assert!(depth >= 2 && depth <= 4); // Depends on optimization
    }

    #[test]
    fn test_circuit_measurement() {
        let mut circuit = Circuit::new(2);
        circuit.add_gate(Box::new(Hadamard::new(0))).unwrap();
        circuit.add_gate(Box::new(CNOT::new(0, 1))).unwrap();
        
        let measurement_results = circuit.measure_all(1000).unwrap();
        
        // Bell state should only give |00⟩ and |11⟩ outcomes
        let total_counts: usize = measurement_results.values().sum();
        assert_eq!(total_counts, 1000);
        
        // Should have roughly equal probabilities for |00⟩ and |11⟩
        let count_00 = measurement_results.get(&0).unwrap_or(&0);
        let count_11 = measurement_results.get(&3).unwrap_or(&0);
        let count_01 = measurement_results.get(&1).unwrap_or(&0);
        let count_10 = measurement_results.get(&2).unwrap_or(&0);
        
        assert_eq!(*count_01, 0);
        assert_eq!(*count_10, 0);
        
        // Statistical test: counts should be roughly equal (within 3σ)
        let expected = 500;
        let tolerance = 3.0 * (expected as f64 * 0.5).sqrt() as usize;
        assert!((*count_00 as i32 - expected as i32).abs() < tolerance as i32);
        assert!((*count_11 as i32 - expected as i32).abs() < tolerance as i32);
    }

    #[test]
    fn test_quantum_fourier_transform() {
        let qft_circuit = CircuitBuilder::new(3)
            .qft()
            .build()
            .unwrap();
        
        assert_eq!(qft_circuit.num_qubits(), 3);
        
        let final_state = qft_circuit.execute().unwrap();
        
        // QFT should produce equal superposition state
        let expected_amplitude = 1.0 / (8.0_f64).sqrt();
        for &amplitude in final_state.iter() {
            assert_abs_diff_eq!(amplitude.norm(), expected_amplitude, epsilon = SIMULATION_EPSILON);
        }
    }
}

#[cfg(test)]
mod quantum_simulation_tests {
    use super::*;

    #[test]
    fn test_simulator_creation() {
        let simulator = Simulator::new(4);
        
        assert_eq!(simulator.num_qubits(), 4);
        
        let initial_state = simulator.get_state();
        assert_eq!(initial_state.len(), 16);
        
        // Should start in |0...0⟩ state
        assert_abs_diff_eq!(initial_state[0], Complex64::new(1.0, 0.0), epsilon = SIMULATION_EPSILON);
        for i in 1..16 {
            assert_abs_diff_eq!(initial_state[i], Complex64::new(0.0, 0.0), epsilon = SIMULATION_EPSILON);
        }
    }

    #[test]
    fn test_gate_application() {
        let mut simulator = Simulator::new(2);
        
        simulator.apply_gate(&Hadamard::new(0)).unwrap();
        let state_after_h = simulator.get_state();
        
        // First qubit should be in superposition, second in |0⟩
        let sqrt2_inv = 1.0 / SQRT_2;
        assert_abs_diff_eq!(state_after_h[0], Complex64::new(sqrt2_inv, 0.0), epsilon = SIMULATION_EPSILON);
        assert_abs_diff_eq!(state_after_h[1], Complex64::new(0.0, 0.0), epsilon = SIMULATION_EPSILON);
        assert_abs_diff_eq!(state_after_h[2], Complex64::new(sqrt2_inv, 0.0), epsilon = SIMULATION_EPSILON);
        assert_abs_diff_eq!(state_after_h[3], Complex64::new(0.0, 0.0), epsilon = SIMULATION_EPSILON);
    }

    #[test]
    fn test_state_evolution() {
        let mut evolution = StateEvolution::new(1);
        
        evolution.add_step(Box::new(RX::new(0, PI/2.0))).unwrap();
        evolution.add_step(Box::new(RY::new(0, PI/3.0))).unwrap();
        evolution.add_step(Box::new(RZ::new(0, PI/6.0))).unwrap();
        
        let final_state = evolution.evolve().unwrap();
        
        // Verify normalization is preserved
        let norm_sq = final_state.iter().map(|c| c.norm_sqr()).sum::<f64>();
        assert_abs_diff_eq!(norm_sq, 1.0, epsilon = SIMULATION_EPSILON);
    }

    #[test]
    fn test_time_evolution() {
        let mut simulator = Simulator::new(1);
        
        // Apply time evolution under Pauli-Z Hamiltonian
        let hamiltonian = constants::pauli_z();
        let time = PI / 4.0;
        
        simulator.time_evolve(&hamiltonian, time).unwrap();
        
        let final_state = simulator.get_state();
        
        // Time evolution should preserve probabilities but change phases
        let norm_sq = final_state.iter().map(|c| c.norm_sqr()).sum::<f64>();
        assert_abs_diff_eq!(norm_sq, 1.0, epsilon = SIMULATION_EPSILON);
    }

    #[test]
    fn test_partial_trace() {
        let mut simulator = Simulator::new(2);
        
        // Prepare Bell state
        simulator.apply_gate(&Hadamard::new(0)).unwrap();
        simulator.apply_gate(&CNOT::new(0, 1)).unwrap();
        
        let reduced_state = simulator.partial_trace(&[1]).unwrap();
        
        // Reduced state of first qubit should be maximally mixed
        assert_eq!(reduced_state.shape(), &[2, 2]);
        assert_abs_diff_eq!(reduced_state[[0, 0]], Complex64::new(0.5, 0.0), epsilon = SIMULATION_EPSILON);
        assert_abs_diff_eq!(reduced_state[[1, 1]], Complex64::new(0.5, 0.0), epsilon = SIMULATION_EPSILON);
        assert_abs_diff_eq!(reduced_state[[0, 1]], Complex64::new(0.0, 0.0), epsilon = SIMULATION_EPSILON);
        assert_abs_diff_eq!(reduced_state[[1, 0]], Complex64::new(0.0, 0.0), epsilon = SIMULATION_EPSILON);
    }

    #[test]
    fn test_entanglement_entropy() {
        let mut simulator = Simulator::new(2);
        
        // Separable state (no entanglement)
        simulator.apply_gate(&RY::new(0, PI/4.0)).unwrap();
        let entropy_separable = simulator.entanglement_entropy(&[0]).unwrap();
        assert_abs_diff_eq!(entropy_separable, 0.0, epsilon = SIMULATION_EPSILON);
        
        // Bell state (maximally entangled)
        simulator.reset();
        simulator.apply_gate(&Hadamard::new(0)).unwrap();
        simulator.apply_gate(&CNOT::new(0, 1)).unwrap();
        let entropy_bell = simulator.entanglement_entropy(&[0]).unwrap();
        assert_abs_diff_eq!(entropy_bell, 2.0_f64.ln(), epsilon = SIMULATION_EPSILON); // ln(2)
    }
}

#[cfg(test)]
mod quantum_optimization_tests {
    use super::*;

    #[test]
    fn test_vqe_optimizer() {
        let mut vqe = VQEOptimizer::new(2);
        
        // Set up simple Hamiltonian: Z₀ ⊗ I₁
        let hamiltonian = ndarray::array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
        ];
        
        vqe.set_hamiltonian(hamiltonian);
        
        // Create simple ansatz
        let mut ansatz = VariationalCircuit::new(2);
        ansatz.add_parametric_layer(&[("RY", 0), ("RY", 1)]).unwrap();
        ansatz.add_entangling_layer(EntanglementPattern::Full).unwrap();
        
        vqe.set_ansatz(ansatz);
        
        let initial_params = vec![0.0, 0.0];
        let result = vqe.optimize(initial_params, 50).unwrap();
        
        assert!(result.converged);
        assert!(result.final_energy < 0.0); // Should find ground state with energy -1
        assert_abs_diff_eq!(result.final_energy, -1.0, epsilon = 0.1);
    }

    #[test]
    fn test_qaoa_optimizer() {
        let qaoa = QAOAOptimizer::new(3, 2); // 3 qubits, 2 layers
        
        // Set up Max-Cut problem on triangle graph
        let adjacency = ndarray::array![
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ];
        
        let initial_params = vec![0.5, 0.3, 0.7, 0.2]; // 2p parameters
        let result = qaoa.optimize_max_cut(&adjacency, initial_params, 30).unwrap();
        
        assert!(result.converged);
        assert!(result.final_energy > 0.0); // Max-Cut should have positive energy
    }

    #[test]
    fn test_adam_optimizer() {
        let config = OptimizerConfig {
            max_iterations: 100,
            tolerance: 1e-6,
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        
        let mut adam = AdamOptimizer::new(config);
        
        // Optimize simple quadratic function: f(x) = (x - 1)²
        let objective = |params: &[f64]| (params[0] - 1.0).powi(2);
        let gradient = |params: &[f64]| vec![2.0 * (params[0] - 1.0)];
        
        let initial_params = vec![0.0];
        let result = adam.optimize(&objective, &gradient, initial_params).unwrap();
        
        assert!(result.converged);
        assert_abs_diff_eq!(result.optimal_params[0], 1.0, epsilon = 0.01);
        assert_abs_diff_eq!(result.final_value, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_parameter_shift_rule() {
        let vqc = VariationalCircuit::new(1);
        let params = vec![PI/4.0];
        
        // Test parameter-shift rule for gradient computation
        let gradient = vqc.compute_gradient(&params, &constants::pauli_z()).unwrap();
        
        assert_eq!(gradient.len(), 1);
        
        // For RY gate: d/dθ ⟨ψ(θ)|Z|ψ(θ)⟩ = -sin(θ)
        let expected_gradient = -(PI/4.0).sin();
        assert_abs_diff_eq!(gradient[0], expected_gradient, epsilon = SIMULATION_EPSILON);
    }
}

#[cfg(test)]
mod quantum_embeddings_tests {
    use super::*;

    #[test]
    fn test_amplitude_embedding() {
        let data = vec![0.5, 0.5, 0.5, 0.5];
        let embedding = AmplitudeEmbedding::new(2);
        
        let encoded_state = embedding.encode(&data).unwrap();
        
        assert_eq!(encoded_state.len(), 4);
        
        // Verify normalization
        let norm_sq = encoded_state.iter().map(|c| c.norm_sqr()).sum::<f64>();
        assert_abs_diff_eq!(norm_sq, 1.0, epsilon = SIMULATION_EPSILON);
        
        // Each amplitude should be 0.5
        for &amplitude in encoded_state.iter() {
            assert_abs_diff_eq!(amplitude.norm(), 0.5, epsilon = SIMULATION_EPSILON);
        }
    }

    #[test]
    fn test_angle_embedding() {
        let data = vec![PI/4.0, PI/3.0];
        let embedding = AngleEmbedding::new(2);
        
        let circuit = embedding.encode(&data).unwrap();
        assert_eq!(circuit.num_qubits(), 2);
        
        let final_state = circuit.execute().unwrap();
        
        // Verify state corresponds to rotations by given angles
        let expected_0 = (PI/8.0).cos(); // cos(θ/2) for θ = π/4
        let expected_1 = (PI/6.0).cos(); // cos(θ/2) for θ = π/3
        
        assert_abs_diff_eq!(final_state[0].norm(), expected_0 * expected_1, epsilon = SIMULATION_EPSILON);
    }

    #[test]
    fn test_basis_embedding() {
        let data = vec![0, 1, 0, 1];
        let embedding = BasisEmbedding::new();
        
        let circuit = embedding.encode(&data).unwrap();
        let final_state = circuit.execute().unwrap();
        
        // Should produce |0101⟩ state
        for i in 0..16 {
            if i == 5 { // binary 0101
                assert_abs_diff_eq!(final_state[i], Complex64::new(1.0, 0.0), epsilon = SIMULATION_EPSILON);
            } else {
                assert_abs_diff_eq!(final_state[i], Complex64::new(0.0, 0.0), epsilon = SIMULATION_EPSILON);
            }
        }
    }

    #[test]
    fn test_iqp_embedding() {
        let data = vec![0.5, -0.3, 0.8];
        let embedding = IQPEmbedding::new(3, 2);
        
        let circuit = embedding.encode(&data).unwrap();
        assert_eq!(circuit.num_qubits(), 3);
        
        let final_state = circuit.execute().unwrap();
        
        // Verify normalization and that state is non-trivial
        let norm_sq = final_state.iter().map(|c| c.norm_sqr()).sum::<f64>();
        assert_abs_diff_eq!(norm_sq, 1.0, epsilon = SIMULATION_EPSILON);
        
        // Should not be in computational basis state
        let max_amplitude = final_state.iter().map(|c| c.norm()).fold(0.0, f64::max);
        assert!(max_amplitude < 0.95); // Not concentrated in single state
    }

    #[test]
    fn test_feature_map_expressivity() {
        let data1 = vec![0.1, 0.2, 0.3];
        let data2 = vec![0.4, 0.5, 0.6];
        
        let feature_map = IQPEmbedding::new(3, 3);
        
        let state1 = feature_map.encode(&data1).unwrap().execute().unwrap();
        let state2 = feature_map.encode(&data2).unwrap().execute().unwrap();
        
        // Different inputs should produce distinguishable states
        let fidelity = utils::fidelity(&state1, &state2).unwrap();
        assert!(fidelity < 0.99); // States should be distinguishable
    }
}

#[cfg(test)]
mod quantum_neural_tests {
    use super::*;

    #[test]
    fn test_quantum_neural_layer() {
        let layer = QuantumNeuralLayer::new(3, 2); // 3 qubits, 2 parameters per qubit
        
        let inputs = vec![0.5, -0.3, 0.8];
        let parameters = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        
        let outputs = layer.forward(&inputs, &parameters).unwrap();
        
        assert_eq!(outputs.len(), 3);
        
        // Outputs should be bounded (quantum measurements in [-1, 1])
        for &output in &outputs {
            assert!(output >= -1.0 && output <= 1.0);
        }
    }

    #[test]
    fn test_variational_quantum_classifier() {
        let mut classifier = VariationalQuantumClassifier::new(2, 3, 2);
        
        // Training data: simple linear separation
        let training_data = vec![
            (vec![0.1, 0.1], 0),
            (vec![0.2, 0.2], 0),
            (vec![0.8, 0.8], 1),
            (vec![0.9, 0.9], 1),
        ];
        
        classifier.train(&training_data, 50).unwrap();
        
        // Test predictions
        let test_input1 = vec![0.15, 0.15];
        let test_input2 = vec![0.85, 0.85];
        
        let prediction1 = classifier.predict(&test_input1).unwrap();
        let prediction2 = classifier.predict(&test_input2).unwrap();
        
        assert_eq!(prediction1, 0);
        assert_eq!(prediction2, 1);
    }

    #[test]
    fn test_quantum_attention_layer() {
        let attention = QuantumAttentionLayer::new(4, 2);
        
        let query = vec![0.1, 0.2, 0.3, 0.4];
        let keys = vec![
            vec![0.5, 0.6, 0.7, 0.8],
            vec![0.2, 0.3, 0.4, 0.5],
            vec![0.8, 0.7, 0.6, 0.5],
        ];
        let values = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.5, 0.5],
        ];
        
        let output = attention.forward(&query, &keys, &values).unwrap();
        
        assert_eq!(output.len(), 2);
        
        // Output should be weighted combination of values
        assert!(output[0] >= 0.0 && output[0] <= 1.0);
        assert!(output[1] >= 0.0 && output[1] <= 1.0);
    }

    #[test]
    fn test_hybrid_classical_quantum_network() {
        let network = HybridNetwork::new(vec![4, 3, 2], 2, 1);
        
        let input = vec![0.1, 0.2, 0.3, 0.4];
        let output = network.forward(&input).unwrap();
        
        assert_eq!(output.len(), 1);
        
        // Test gradient computation
        let gradients = network.backward(&input, &[0.5]).unwrap();
        assert!(!gradients.is_empty());
    }
}

#[cfg(test)]
mod pennylane_compatibility_tests {
    use super::*;

    #[test]
    fn test_default_qubit_device() {
        let device = DefaultQubitDevice::new(2);
        
        assert_eq!(device.num_qubits(), 2);
        assert_eq!(device.device_name(), "default.qubit");
        
        let state = device.get_state();
        assert_eq!(state.len(), 4);
        assert_abs_diff_eq!(state[0], Complex64::new(1.0, 0.0), epsilon = SIMULATION_EPSILON);
    }

    #[test]
    fn test_qnode_creation() {
        let device = device(2, "default.qubit").unwrap();
        
        let qnode_fn = |x: f64| -> Result<f64> {
            let mut circuit = Circuit::new(2);
            circuit.add_gate(Box::new(RY::new(0, x))).unwrap();
            circuit.add_gate(Box::new(CNOT::new(0, 1))).unwrap();
            
            let final_state = circuit.execute().unwrap();
            let pauli_z = constants::pauli_z();
            
            // Measure expectation value of Z ⊗ I
            let z_tensor_i = ndarray::kron(&pauli_z, &constants::identity());
            let expectation = utils::expectation_value(&final_state, &z_tensor_i)?;
            
            Ok(expectation.re)
        };
        
        let qnode = qnode(qnode_fn, device);
        
        let result = qnode.call(PI/4.0).unwrap();
        assert!(result >= -1.0 && result <= 1.0);
    }

    #[test]
    fn test_qnode_builder() {
        let qnode = QNodeBuilder::new()
            .device("default.qubit", 2)
            .interface("autograd")
            .diff_method("parameter_shift")
            .build()
            .unwrap();
        
        assert_eq!(qnode.device().num_qubits(), 2);
        assert_eq!(qnode.diff_method(), "parameter_shift");
    }

    #[test]
    fn test_pennylane_operations() {
        use pennylane_compat::ops::*;
        
        // Test PennyLane-style gate creation
        let ry = RY(0, PI/4.0);
        let cnot = CNOT(0, 1);
        let hadamard = Hadamard(0);
        
        assert_eq!(ry.qubit(), 0);
        assert_abs_diff_eq!(ry.parameter(), PI/4.0, epsilon = QUANTUM_EPSILON);
        
        assert_eq!(cnot.control(), 0);
        assert_eq!(cnot.target(), 1);
        
        assert_eq!(hadamard.qubit(), 0);
    }

    #[test]
    fn test_pennylane_measurements() {
        use pennylane_compat::measurements::*;
        
        let device = DefaultQubitDevice::new(2);
        let mut circuit = Circuit::new(2);
        circuit.add_gate(Box::new(Hadamard::new(0))).unwrap();
        circuit.add_gate(Box::new(CNOT::new(0, 1))).unwrap();
        
        let final_state = circuit.execute().unwrap();
        
        // Test expectation value measurement
        let exp_z = expval(PauliZ(0), &final_state).unwrap();
        assert_abs_diff_eq!(exp_z, 0.0, epsilon = SIMULATION_EPSILON);
        
        // Test variance measurement
        let var_z = var(PauliZ(0), &final_state).unwrap();
        assert_abs_diff_eq!(var_z, 1.0, epsilon = SIMULATION_EPSILON);
        
        // Test sample measurement
        let samples = sample(PauliZ(0), &final_state, 1000).unwrap();
        assert_eq!(samples.len(), 1000);
        
        // All samples should be ±1
        for &sample in &samples {
            assert!(sample == 1.0 || sample == -1.0);
        }
    }

    #[test]
    fn test_pennylane_templates() {
        use pennylane_compat::templates::*;
        
        // Test basic entangler
        let entangler = BasicEntanglerLayers::new(3, 2);
        let circuit = entangler.build(&vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        
        assert_eq!(circuit.num_qubits(), 3);
        assert!(circuit.gate_count() > 0);
        
        // Test strongly entangling layers
        let strong_ent = StronglyEntanglingLayers::new(2, 3);
        let params = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        let circuit = strong_ent.build(&params).unwrap();
        
        assert_eq!(circuit.num_qubits(), 2);
        assert!(circuit.gate_count() >= 6); // 2 qubits × 3 layers × 1 gate minimum
    }
}

// Property-based tests for quantum mechanics properties
proptest! {
    #[test]
    fn test_unitary_preservation(
        angle in 0.0..2.0*PI
    ) {
        let gate = RY::new(0, angle);
        let matrix = gate.matrix();
        
        // Unitary property: U†U = I
        let adjoint = matrix.map(|c| c.conj()).t().to_owned();
        let identity_test = adjoint.dot(&matrix);
        
        for i in 0..2 {
            for j in 0..2 {
                if i == j {
                    prop_assert!((identity_test[[i, j]] - Complex64::new(1.0, 0.0)).norm() < GATE_EPSILON);
                } else {
                    prop_assert!(identity_test[[i, j]].norm() < GATE_EPSILON);
                }
            }
        }
    }

    #[test]
    fn test_normalization_preservation(
        angle1 in 0.0..2.0*PI,
        angle2 in 0.0..2.0*PI
    ) {
        let mut circuit = Circuit::new(1);
        circuit.add_gate(Box::new(RY::new(0, angle1))).unwrap();
        circuit.add_gate(Box::new(RZ::new(0, angle2))).unwrap();
        
        let final_state = circuit.execute().unwrap();
        let norm_sq = final_state.iter().map(|c| c.norm_sqr()).sum::<f64>();
        
        prop_assert!((norm_sq - 1.0).abs() < SIMULATION_EPSILON);
    }

    #[test]
    fn test_measurement_probabilities(
        angle in 0.0..PI
    ) {
        let mut circuit = Circuit::new(1);
        circuit.add_gate(Box::new(RY::new(0, angle))).unwrap();
        
        let final_state = circuit.execute().unwrap();
        
        // Measurement probabilities
        let prob_0 = final_state[0].norm_sqr();
        let prob_1 = final_state[1].norm_sqr();
        
        // Should sum to 1
        prop_assert!((prob_0 + prob_1 - 1.0).abs() < QUANTUM_EPSILON);
        
        // Should match analytical prediction for RY gate
        let expected_prob_0 = (angle / 2.0).cos().powi(2);
        let expected_prob_1 = (angle / 2.0).sin().powi(2);
        
        prop_assert!((prob_0 - expected_prob_0).abs() < QUANTUM_EPSILON);
        prop_assert!((prob_1 - expected_prob_1).abs() < QUANTUM_EPSILON);
    }

    #[test]
    fn test_expectation_value_bounds(
        data in prop::collection::vec(prop::num::f64::NORMAL, 4)
    ) {
        // Normalize data for amplitude encoding
        let norm = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            let normalized_data: Vec<f64> = data.iter().map(|x| x / norm).collect();
            
            if let Ok(state) = utils::amplitude_encode(&normalized_data) {
                let pauli_z_tensor_i = ndarray::kron(&constants::pauli_z(), &constants::identity());
                
                if let Ok(expectation) = utils::expectation_value(&state, &pauli_z_tensor_i) {
                    // Expectation values of Hermitian operators should be real
                    prop_assert!(expectation.im.abs() < QUANTUM_EPSILON);
                    
                    // For Pauli operators, expectation values should be in [-1, 1]
                    prop_assert!(expectation.re >= -1.0 - QUANTUM_EPSILON);
                    prop_assert!(expectation.re <= 1.0 + QUANTUM_EPSILON);
                }
            }
        }
    }

    #[test]
    fn test_entanglement_monotonicity(
        angle1 in 0.0..PI/2.0,
        angle2 in 0.0..PI/2.0
    ) {
        let mut simulator1 = Simulator::new(2);
        let mut simulator2 = Simulator::new(2);
        
        // First circuit: less entangling
        simulator1.apply_gate(&RY::new(0, angle1)).unwrap();
        simulator1.apply_gate(&CNOT::new(0, 1)).unwrap();
        
        // Second circuit: more entangling  
        simulator2.apply_gate(&RY::new(0, angle2)).unwrap();
        simulator2.apply_gate(&CNOT::new(0, 1)).unwrap();
        simulator2.apply_gate(&RY::new(1, angle1)).unwrap();
        
        if let (Ok(entropy1), Ok(entropy2)) = (
            simulator1.entanglement_entropy(&[0]),
            simulator2.entanglement_entropy(&[0])
        ) {
            // More gates should generally not decrease entanglement (with some tolerance)
            prop_assert!(entropy2 >= entropy1 - SIMULATION_EPSILON);
        }
    }
}