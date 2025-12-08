//! Quantum-specific tests for QAR crate
//!
//! This module contains tests specifically for quantum computing components
//! including gates, circuits, and quantum algorithms.

use quantum_agentic_reasoning::quantum::*;
use tokio::test;
use approx::assert_relative_eq;
use std::f64::consts::PI;

#[test]
fn test_quantum_gate_operations() {
    // Test basic gate properties
    let pauli_x = StandardGates::pauli_x();
    assert_eq!(pauli_x.name, "X");
    assert_eq!(pauli_x.num_qubits, 1);
    assert!(pauli_x.is_unitary(), "Pauli-X should be unitary");

    let pauli_y = StandardGates::pauli_y();
    assert!(pauli_y.is_unitary(), "Pauli-Y should be unitary");

    let pauli_z = StandardGates::pauli_z();
    assert!(pauli_z.is_unitary(), "Pauli-Z should be unitary");

    let hadamard = StandardGates::hadamard();
    assert!(hadamard.is_unitary(), "Hadamard should be unitary");

    // Test parameterized gates
    let rx_gate = StandardGates::rx(PI / 2.0);
    assert!(rx_gate.is_unitary(), "RX gate should be unitary");
    assert_eq!(rx_gate.parameters[0], PI / 2.0);

    let ry_gate = StandardGates::ry(PI / 3.0);
    assert!(ry_gate.is_unitary(), "RY gate should be unitary");
    assert_eq!(ry_gate.parameters[0], PI / 3.0);

    let rz_gate = StandardGates::rz(PI / 4.0);
    assert!(rz_gate.is_unitary(), "RZ gate should be unitary");
    assert_eq!(rz_gate.parameters[0], PI / 4.0);

    // Test two-qubit gates
    let cnot = StandardGates::cnot();
    assert_eq!(cnot.num_qubits, 2);
    assert!(cnot.is_unitary(), "CNOT should be unitary");

    let cz = StandardGates::cz();
    assert_eq!(cz.num_qubits, 2);
    assert!(cz.is_unitary(), "CZ should be unitary");

    let swap = StandardGates::swap();
    assert_eq!(swap.num_qubits, 2);
    assert!(swap.is_unitary(), "SWAP should be unitary");

    // Test three-qubit gates
    let toffoli = StandardGates::toffoli();
    assert_eq!(toffoli.num_qubits, 3);
    assert!(toffoli.is_unitary(), "Toffoli should be unitary");
}

#[test]
fn test_quantum_state_operations() {
    // Test state creation
    let state = QuantumState::new(2);
    assert_eq!(state.num_qubits, 2);
    assert_eq!(state.amplitudes.len(), 4);
    assert!(state.is_normalized(), "Initial state should be normalized");
    assert_eq!(state.probability(0), 1.0, "Should start in |00⟩ state");

    // Test superposition state
    let superposition = QuantumState::superposition(2);
    assert!(superposition.is_normalized(), "Superposition should be normalized");
    for i in 0..4 {
        assert_relative_eq!(superposition.probability(i), 0.25, epsilon = 1e-10);
    }

    // Test expectation values for |0⟩ state
    let z_expectation = state.expectation_z(0).unwrap();
    assert_relative_eq!(z_expectation, 1.0, epsilon = 1e-10); // |0⟩ has Z expectation +1

    let x_expectation = state.expectation_x(0).unwrap();
    assert_relative_eq!(x_expectation, 0.0, epsilon = 1e-10); // |0⟩ has X expectation 0
}

#[test]
fn test_quantum_gate_application() {
    let mut state = QuantumState::new(1);
    
    // Apply Pauli-X gate (should flip |0⟩ to |1⟩)
    let x_gate = StandardGates::pauli_x();
    state.apply_single_qubit_gate(0, &x_gate).unwrap();
    
    assert_relative_eq!(state.probability(0), 0.0, epsilon = 1e-10); // |0⟩ probability
    assert_relative_eq!(state.probability(1), 1.0, epsilon = 1e-10); // |1⟩ probability
    
    // Z expectation should now be -1
    let z_expectation = state.expectation_z(0).unwrap();
    assert_relative_eq!(z_expectation, -1.0, epsilon = 1e-10);

    // Apply Hadamard to create superposition
    let mut state = QuantumState::new(1);
    let h_gate = StandardGates::hadamard();
    state.apply_single_qubit_gate(0, &h_gate).unwrap();
    
    // Should be in equal superposition
    assert_relative_eq!(state.probability(0), 0.5, epsilon = 1e-10);
    assert_relative_eq!(state.probability(1), 0.5, epsilon = 1e-10);
    
    // X expectation should be 1 for |+⟩ state
    let x_expectation = state.expectation_x(0).unwrap();
    assert_relative_eq!(x_expectation, 1.0, epsilon = 1e-10);
}

#[test]
fn test_two_qubit_gate_application() {
    let mut state = QuantumState::new(2);
    
    // Apply Hadamard to first qubit
    let h_gate = StandardGates::hadamard();
    state.apply_single_qubit_gate(0, &h_gate).unwrap();
    
    // Apply CNOT with first qubit as control, second as target
    let cnot = StandardGates::cnot();
    state.apply_two_qubit_gate(0, 1, &cnot).unwrap();
    
    // Should create Bell state |00⟩ + |11⟩
    assert_relative_eq!(state.probability(0), 0.5, epsilon = 1e-10); // |00⟩
    assert_relative_eq!(state.probability(1), 0.0, epsilon = 1e-10); // |01⟩
    assert_relative_eq!(state.probability(2), 0.0, epsilon = 1e-10); // |10⟩
    assert_relative_eq!(state.probability(3), 0.5, epsilon = 1e-10); // |11⟩
}

#[test]
fn test_quantum_measurement() {
    let mut state = QuantumState::superposition(1);
    
    // Measure the qubit multiple times
    let mut zero_count = 0;
    let mut one_count = 0;
    
    for _ in 0..100 {
        let mut test_state = state.clone();
        let result = test_state.measure_qubit(0).unwrap();
        if result {
            one_count += 1;
        } else {
            zero_count += 1;
        }
    }
    
    // Should get roughly equal counts (within statistical variation)
    assert!(zero_count > 20 && zero_count < 80, "Should get reasonable distribution of 0s");
    assert!(one_count > 20 && one_count < 80, "Should get reasonable distribution of 1s");
    assert_eq!(zero_count + one_count, 100, "Total measurements should be 100");
}

#[test]
fn test_quantum_fidelity() {
    let state1 = QuantumState::new(2);
    let state2 = QuantumState::new(2);
    
    // Identical states should have fidelity 1
    let fidelity = state1.fidelity(&state2).unwrap();
    assert_relative_eq!(fidelity, 1.0, epsilon = 1e-10);
    
    // Orthogonal states should have fidelity 0
    let mut orthogonal_state = QuantumState::new(2);
    let x_gate = StandardGates::pauli_x();
    orthogonal_state.apply_single_qubit_gate(0, &x_gate).unwrap();
    
    let fidelity = state1.fidelity(&orthogonal_state).unwrap();
    assert_relative_eq!(fidelity, 0.0, epsilon = 1e-10);
}

#[test]
fn test_von_neumann_entropy() {
    // Pure state should have zero entropy
    let pure_state = QuantumState::new(2);
    let entropy = pure_state.von_neumann_entropy();
    assert_relative_eq!(entropy, 0.0, epsilon = 1e-10);
    
    // Maximally mixed state should have maximum entropy
    let mixed_state = QuantumState::superposition(2);
    let entropy = mixed_state.von_neumann_entropy();
    assert!(entropy > 1.9); // log2(4) = 2, but numerical precision
}

#[test]
async fn test_qft_circuit() {
    let qft = QftCircuit::new(3);
    
    // Test with encoded parameters
    let params = CircuitParams::new(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3);
    let context = ExecutionContext::default();
    
    let result = qft.execute(&params, &context).await;
    assert!(result.is_ok(), "QFT execution should succeed");
    
    let result = result.unwrap();
    assert!(!result.expectation_values.is_empty(), "Should return spectral information");
    assert!(result.execution_time_ms > 0.0, "Should measure execution time");
    assert!(result.used_quantum || !context.prefer_quantum, "Should use quantum if preferred");
}

#[test]
async fn test_qft_classical_fallback() {
    let qft = QftCircuit::new(3);
    
    // Test classical fallback
    let params = CircuitParams::new(vec![0.5, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05], 3);
    
    let result = qft.classical_fallback(&params).await;
    assert!(result.is_ok(), "Classical fallback should work");
    
    let result = result.unwrap();
    assert!(!result.expectation_values.is_empty(), "Should return FFT result");
    assert!(!result.used_quantum, "Should indicate classical execution");
}

#[test]
async fn test_decision_optimization_circuit() {
    let circuit = DecisionOptimizationCircuit::new(3, 2, 0.5);
    
    let params = CircuitParams::new(vec![0.8, 0.6, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1], 3);
    let context = ExecutionContext::default();
    
    let result = circuit.execute(&params, &context).await;
    assert!(result.is_ok(), "Decision optimization should succeed");
    
    let result = result.unwrap();
    assert_eq!(result.expectation_values.len(), 8, "Should return weights for all states");
    assert!(result.execution_time_ms > 0.0, "Should measure execution time");
}

#[test]
async fn test_pattern_recognition_circuit() {
    let mut circuit = PatternRecognitionCircuit::new(3, 16);
    
    // Add reference patterns
    let bull_pattern = vec![0.8, 0.6, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1];
    let bear_pattern = vec![0.2, 0.4, 0.6, 0.8, 0.1, 0.1, 0.1, 0.1];
    
    circuit.add_reference_pattern(bull_pattern.clone()).unwrap();
    circuit.add_reference_pattern(bear_pattern.clone()).unwrap();
    
    // Test with bull-like pattern
    let test_pattern = vec![0.7, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1];
    let params = CircuitParams::new(test_pattern.clone(), 3);
    let context = ExecutionContext::default();
    
    let result = circuit.execute(&params, &context).await;
    assert!(result.is_ok(), "Pattern recognition should succeed");
    
    let result = result.unwrap();
    assert_eq!(result.expectation_values.len(), 2, "Should return similarity to both patterns");
    
    // Test quantum similarity directly
    let similarity = circuit.quantum_similarity(&test_pattern, &bull_pattern).unwrap();
    assert!(similarity > 0.5, "Should have high similarity to bull pattern");
    
    let similarity = circuit.quantum_similarity(&test_pattern, &bear_pattern).unwrap();
    assert!(similarity < 0.8, "Should have lower similarity to bear pattern");
}

#[test]
async fn test_circuit_parameter_validation() {
    let qft = QftCircuit::new(3);
    
    // Valid parameters
    let valid_params = CircuitParams::new(vec![0.1; 8], 3);
    assert!(qft.validate_parameters(&valid_params).is_ok());
    
    // Invalid qubit count
    let invalid_qubits = CircuitParams::new(vec![0.1; 8], 2);
    assert!(qft.validate_parameters(&invalid_qubits).is_err());
    
    // Invalid parameter count (when parameters are provided)
    let invalid_params = CircuitParams::new(vec![0.1; 4], 3);
    assert!(qft.validate_parameters(&invalid_params).is_err());
    
    // Empty parameters should be valid (creates default state)
    let empty_params = CircuitParams::new(vec![], 3);
    assert!(qft.validate_parameters(&empty_params).is_ok());
}

#[test]
async fn test_quantum_backend_simulator() {
    let config = BackendConfig::new(BackendType::Simulator, 4);
    let backend = SimulatorBackend::new(config);
    
    // Test backend properties
    assert!(backend.is_quantum_available().await);
    
    let backends = backend.get_quantum_backends().await;
    assert_eq!(backends, vec!["simulator"]);
    
    let capabilities = backend.get_capabilities().await;
    assert_eq!(capabilities.max_qubits, 4);
    assert!(!capabilities.supported_gates.is_empty());
    
    // Test circuit execution
    let qft = QftCircuit::new(3);
    let params = CircuitParams::new(vec![0.5; 8], 3);
    
    let result = backend.execute_quantum(&qft, &params).await;
    assert!(result.is_ok(), "Backend should execute circuit successfully");
    
    // Test metrics
    let metrics = backend.get_metrics().await;
    assert!(metrics.quantum_time_ms >= 0.0);
}

#[test]
async fn test_hardware_backend_fallback() {
    let config = BackendConfig::new(BackendType::IbmQuantum, 8);
    let backend = HardwareBackend::new(config);
    
    // Should not be connected initially
    assert!(!backend.is_quantum_available().await);
    
    // Test fallback execution
    let qft = QftCircuit::new(3);
    let params = CircuitParams::new(vec![0.4; 8], 3);
    
    let result = backend.execute_quantum(&qft, &params).await;
    assert!(result.is_ok(), "Should fall back to simulator");
    
    let result = result.unwrap();
    // Should use classical fallback or simulator
    assert!(result.execution_time_ms > 0.0);
}

#[test]
fn test_backend_manager() {
    let mut manager = BackendManager::new();
    
    // Add backends
    let sim_config = BackendConfig::new(BackendType::Simulator, 4);
    let sim_backend = Box::new(SimulatorBackend::new(sim_config));
    manager.add_backend("simulator".to_string(), sim_backend);
    
    let hw_config = BackendConfig::new(BackendType::IbmQuantum, 8);
    let hw_backend = Box::new(HardwareBackend::new(hw_config));
    manager.add_backend("quantum_hardware".to_string(), hw_backend);
    
    // Test selection strategies
    manager.set_selection_strategy(BackendSelectionStrategy::PreferSimulator);
    manager.set_selection_strategy(BackendSelectionStrategy::PreferQuantum);
    manager.set_selection_strategy(BackendSelectionStrategy::OptimalPerformance);
}

#[test]
fn test_gate_builder() {
    // Test custom gate creation
    let custom_gate = GateBuilder::new("CustomRX".to_string(), 1)
        .with_parameter(PI / 3.0)
        .build_rotation_gate("x", PI / 3.0);
    
    assert_eq!(custom_gate.name, "RX");
    assert!(custom_gate.is_unitary());
    assert_eq!(custom_gate.parameters[0], PI / 3.0);
    
    // Test custom matrix gate
    let identity_matrix = vec![
        vec![num_complex::Complex64::new(1.0, 0.0), num_complex::Complex64::new(0.0, 0.0)],
        vec![num_complex::Complex64::new(0.0, 0.0), num_complex::Complex64::new(1.0, 0.0)],
    ];
    
    let custom_identity = GateBuilder::new("CustomI".to_string(), 1)
        .build_from_matrix(identity_matrix);
    
    assert_eq!(custom_identity.name, "CustomI");
    assert!(custom_identity.is_unitary());
}

#[test]
fn test_gate_adjoint() {
    let t_gate = StandardGates::t_gate();
    let t_adjoint = t_gate.adjoint();
    
    assert_eq!(t_adjoint.name, "T_adjoint");
    assert!(t_adjoint.is_unitary());
    
    // For self-adjoint gates like Pauli-X
    let x_gate = StandardGates::pauli_x();
    let x_adjoint = x_gate.adjoint();
    
    // X† = X, so they should be the same (up to numerical precision)
    for i in 0..x_gate.matrix.len() {
        for j in 0..x_gate.matrix[i].len() {
            assert_relative_eq!(
                x_gate.matrix[i][j].re,
                x_adjoint.matrix[i][j].re,
                epsilon = 1e-10
            );
            assert_relative_eq!(
                x_gate.matrix[i][j].im,
                -x_adjoint.matrix[i][j].im, // Conjugate transpose
                epsilon = 1e-10
            );
        }
    }
}

#[test]
async fn test_execution_context() {
    let qft = QftCircuit::new(2);
    let params = CircuitParams::new(vec![0.5, 0.3, 0.2, 0.0], 2);
    
    // Test with different execution contexts
    let quantum_context = ExecutionContext {
        prefer_quantum: true,
        max_execution_time_ms: 5000,
        max_retries: 2,
        cache_enabled: true,
        monitoring_enabled: true,
    };
    
    let classical_context = ExecutionContext {
        prefer_quantum: false,
        max_execution_time_ms: 1000,
        max_retries: 1,
        cache_enabled: false,
        monitoring_enabled: false,
    };
    
    let quantum_result = qft.execute(&params, &quantum_context).await;
    let classical_result = qft.execute(&params, &classical_context).await;
    
    assert!(quantum_result.is_ok());
    assert!(classical_result.is_ok());
    
    // Both should produce valid results
    let q_result = quantum_result.unwrap();
    let c_result = classical_result.unwrap();
    
    assert!(!q_result.expectation_values.is_empty());
    assert!(!c_result.expectation_values.is_empty());
}

#[test]
fn test_quantum_state_advanced_operations() {
    let mut state = QuantumState::new(2);
    
    // Test density matrix
    let density_matrix = state.to_density_matrix();
    assert_eq!(density_matrix.len(), 4);
    assert_eq!(density_matrix[0].len(), 4);
    
    // For |00⟩ state, only (0,0) element should be 1
    assert_relative_eq!(density_matrix[0][0].re, 1.0, epsilon = 1e-10);
    for i in 1..4 {
        assert_relative_eq!(density_matrix[0][i].norm_sqr(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(density_matrix[i][0].norm_sqr(), 0.0, epsilon = 1e-10);
    }
    
    // Test most likely outcome
    let (outcome, probability) = state.most_likely_outcome();
    assert_eq!(outcome, 0); // |00⟩ state
    assert_relative_eq!(probability, 1.0, epsilon = 1e-10);
    
    // Test with superposition
    let superposition = QuantumState::superposition(2);
    let (_, prob) = superposition.most_likely_outcome();
    assert_relative_eq!(prob, 0.25, epsilon = 1e-10); // All states equally likely
}