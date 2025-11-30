//! Integration tests for the quantum-circuit crate

use quantum_circuit::{
    Circuit, CircuitBuilder, VariationalCircuit, EntanglementPattern,
    simulation::{Simulator, BatchSimulator},
    optimization::{QAOAOptimizer, VQEOptimizer, OptimizerConfig, CostHamiltonian, Optimizer, VariationalOptimizer},
    embeddings::{AmplitudeEmbedding, AngleEmbedding, ParametricEmbedding, NormalizationMethod, QuantumEmbedding},
    neural::SimpleHybridNet,
    pennylane_compat::{device, QNodeBuilder, DefaultQubitDevice},
    gates::*,
    constants,
    utils,
};
use approx::assert_abs_diff_eq;
use std::f64::consts::PI;

#[test]
fn test_end_to_end_vqc_workflow() {
    // Create a variational quantum circuit
    let mut vqc = VariationalCircuit::new(3, 2, EntanglementPattern::Circular);
    vqc.build_random().unwrap();
    
    let circuit = vqc.circuit();
    assert_eq!(circuit.n_qubits, 3);
    assert!(circuit.parameter_count() > 0);
    
    // Execute circuit
    let state = circuit.execute().unwrap();
    assert_eq!(state.len(), 8); // 2^3 = 8
    
    // Verify normalization
    let norm_sqr: f64 = state.iter().map(|c| c.norm_sqr()).sum();
    assert_abs_diff_eq!(norm_sqr, 1.0, epsilon = 1e-10);
}

#[test]
fn test_qaoa_max_cut_optimization() {
    // Define a simple graph for Max-Cut
    let edges = vec![(0, 1), (1, 2), (2, 0)]; // Triangle graph
    let cost_hamiltonian = CostHamiltonian::max_cut(&edges);
    
    // Create QAOA optimizer
    let config = OptimizerConfig {
        max_iterations: 20,
        tolerance: 1e-6,
        learning_rate: 0.1,
        verbose: false,
        ..Default::default()
    };
    
    let mut qaoa = QAOAOptimizer::new(config, 2);
    
    // Simple objective function for testing
    let objective = |params: &[f64]| -> f64 {
        params.iter().map(|x| x.abs()).sum::<f64>()
    };
    
    let initial_params = vec![0.5, 0.3, 0.2, 0.1];
    let result = qaoa.optimize(objective, &initial_params).unwrap();
    
    assert!(result.optimal_value >= 0.0);
    assert_eq!(result.optimal_params.len(), initial_params.len());
}

#[test]
fn test_vqe_ground_state_finding() {
    // Simple 1-qubit Hamiltonian: H = -Z (ground state should be |1⟩)
    let hamiltonian = -1.0 * constants::pauli_z();
    
    let config = OptimizerConfig {
        max_iterations: 50,
        tolerance: 1e-6,
        learning_rate: 0.01,
        verbose: false,
        ..Default::default()
    };
    
    let mut vqe = VQEOptimizer::new(config, hamiltonian.clone());
    
    // Create a simple ansatz circuit
    let mut circuit = Circuit::new(1);
    circuit.add_gate(Box::new(RY::new(0, PI))).unwrap(); // Start with |1⟩
    
    let objective = |params: &[f64]| -> f64 {
        let mut test_circuit = Circuit::new(1);
        test_circuit.add_gate(Box::new(RY::new(0, params[0]))).unwrap();
        
        test_circuit.expectation_value(&hamiltonian).unwrap_or(f64::INFINITY)
    };
    
    let initial_params = vec![0.0]; // Start from |0⟩
    let result = vqe.optimize(objective, &initial_params).unwrap();
    
    // Ground state energy should be close to -1
    assert!(result.optimal_value < -0.5);
}

#[test]
fn test_quantum_embeddings_pipeline() {
    // Test different embedding methods
    let data = vec![0.6, 0.8, 0.0, 0.0]; // 4D data for 2-qubit embedding
    
    // Amplitude embedding
    let amp_embedding = AmplitudeEmbedding::new(4, NormalizationMethod::L2);
    let amp_state = amp_embedding.embed(&data).unwrap();
    assert_eq!(amp_state.len(), 4);
    
    // Angle embedding
    let angle_embedding = AngleEmbedding::new(4).with_qubits(2);
    let angle_state = angle_embedding.embed(&data).unwrap();
    assert_eq!(angle_state.len(), 4);
    
    // Parametric embedding
    let param_embedding = ParametricEmbedding::new(4, 2, 1);
    let param_state = param_embedding.embed(&data).unwrap();
    assert_eq!(param_state.len(), 4);
    
    // All states should be normalized
    for state in [&amp_state, &angle_state, &param_state] {
        let norm_sqr: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        assert_abs_diff_eq!(norm_sqr, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn test_hybrid_neural_network_training() {
    // Create a simple hybrid network
    let mut net = SimpleHybridNet::new(3, 4, 2, 2); // 3 input, 4 hidden, 2 output, 2 qubits
    
    // Create dummy training data
    let train_x = ndarray::Array2::from_shape_vec(
        (4, 3),
        vec![
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,
            0.7, 0.8, 0.9,
            0.2, 0.4, 0.6,
        ]
    ).unwrap();
    
    let train_y = ndarray::Array2::from_shape_vec(
        (4, 2),
        vec![
            1.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            0.0, 1.0,
        ]
    ).unwrap();
    
    // Train for a few epochs
    let history = net.train(&train_x, &train_y, 5).unwrap();
    
    assert_eq!(history.losses.len(), 5);
    assert_eq!(history.accuracies.len(), 5);
    
    // Test forward pass
    let predictions = net.forward(&train_x).unwrap();
    assert_eq!(predictions.shape(), &[4, 2]);
}

#[test]
fn test_pennylane_compatibility() {
    // Create a device
    let mut device = device("default.qubit", 2).unwrap();
    
    // Build a quantum circuit using PennyLane-style API
    let mut builder = QNodeBuilder::new(2);
    builder.hadamard(0).unwrap()
           .cnot(0, 1).unwrap()
           .ry(PI / 4.0, 1).unwrap()
           .expectation(constants::pauli_z(), "Z0".to_string()).unwrap();
    
    let qnode = builder.build();
    
    // Execute the QNode
    let result = qnode.execute(device.as_mut(), None).unwrap();
    
    assert!(result.state.is_some());
    assert!(result.expectations.contains_key("Z0"));
    assert!(result.metadata.execution_time_ms > 0.0);
}

#[test]
fn test_circuit_gradient_computation() {
    // Create a parameterized circuit
    let mut circuit = Circuit::new(1);
    circuit.add_gate(Box::new(RY::new(0, PI / 4.0))).unwrap();
    
    // Compute gradients with respect to Pauli-Z observable
    let observable = constants::pauli_z();
    let gradients = circuit.parameter_gradients(&observable).unwrap();
    
    assert_eq!(gradients.len(), 1);
    // Gradient should be non-zero for non-trivial circuit
    assert!(gradients[0].abs() > 1e-10);
}

#[test]
fn test_batch_simulation() {
    // Create multiple circuits
    let mut circuit1 = Circuit::new(2);
    circuit1.add_gate(Box::new(Hadamard::new(0))).unwrap();
    circuit1.add_gate(Box::new(CNOT::new(0, 1))).unwrap();
    
    let mut circuit2 = Circuit::new(2);
    circuit2.add_gate(Box::new(PauliX::new(0))).unwrap();
    circuit2.add_gate(Box::new(Hadamard::new(1))).unwrap();
    
    // Execute in batch
    let mut batch_sim = BatchSimulator::new(2, 2);
    let results = batch_sim.execute_batch(vec![&circuit1, &circuit2]).unwrap();
    
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].len(), 4); // 2^2 = 4
    assert_eq!(results[1].len(), 4);
    
    // Verify normalization
    for state in &results {
        let norm_sqr: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        assert_abs_diff_eq!(norm_sqr, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn test_measurement_statistics() {
    let mut sim = Simulator::new(2);
    
    // Create Bell state: (|00⟩ + |11⟩) / √2
    let mut circuit = Circuit::new(2);
    circuit.add_gate(Box::new(Hadamard::new(0))).unwrap();
    circuit.add_gate(Box::new(CNOT::new(0, 1))).unwrap();
    
    sim.execute_circuit(&circuit).unwrap();
    
    // Sample measurements
    let samples = sim.sample_measurements(1000).unwrap();
    assert_eq!(samples.len(), 1000);
    
    // Count |00⟩ and |11⟩ measurements
    let mut count_00 = 0;
    let mut count_11 = 0;
    
    for sample in samples {
        assert_eq!(sample.len(), 2);
        if sample == vec![0, 0] {
            count_00 += 1;
        } else if sample == vec![1, 1] {
            count_11 += 1;
        }
    }
    
    // Should have roughly equal counts of |00⟩ and |11⟩
    let total_valid = count_00 + count_11;
    assert!(total_valid > 900); // Most measurements should be |00⟩ or |11⟩
    
    let ratio = (count_00 as f64) / (total_valid as f64);
    assert!(ratio > 0.3 && ratio < 0.7); // Should be close to 0.5
}

#[test]
fn test_quantum_fidelity_computation() {
    // Create two similar quantum states
    let state1 = constants::zero_state();
    let state2 = constants::plus_state();
    
    let fidelity = utils::fidelity(&state1, &state2).unwrap();
    
    // Fidelity between |0⟩ and |+⟩ should be 0.5
    assert_abs_diff_eq!(fidelity, 0.5, epsilon = 1e-10);
    
    // Fidelity with itself should be 1
    let self_fidelity = utils::fidelity(&state1, &state1).unwrap();
    assert_abs_diff_eq!(self_fidelity, 1.0, epsilon = 1e-10);
}

#[test]
fn test_expectation_value_computation() {
    // Create |+⟩ state
    let mut sim = Simulator::new(1);
    let mut circuit = Circuit::new(1);
    circuit.add_gate(Box::new(Hadamard::new(0))).unwrap();
    
    sim.execute_circuit(&circuit).unwrap();
    
    // Compute expectation values
    let pauli_x = constants::pauli_x();
    let pauli_z = constants::pauli_z();
    
    let exp_x = sim.expectation_value(&pauli_x).unwrap();
    let exp_z = sim.expectation_value(&pauli_z).unwrap();
    
    // For |+⟩ state: ⟨X⟩ = 1, ⟨Z⟩ = 0
    assert_abs_diff_eq!(exp_x, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(exp_z, 0.0, epsilon = 1e-10);
}

#[test]
fn test_circuit_composition_and_depth() {
    let circuit = CircuitBuilder::new(3)
        .h(0)      // Depth 1
        .h(1)      // Depth 1 (parallel)
        .cnot(0, 1) // Depth 2
        .cnot(1, 2) // Depth 3
        .rx(0, PI / 2.0) // Depth 3 (parallel with cnot(1,2))
        .build();
    
    assert_eq!(circuit.gate_count(), 5);
    assert!(circuit.depth() >= 3);
    assert_eq!(circuit.parameter_count(), 1); // Only RX has a parameter
}

#[test]
fn test_amplitude_encoding_edge_cases() {
    // Test with data that's not a power of 2
    let data = vec![0.3, 0.4, 0.5]; // 3 elements
    let embedding = AmplitudeEmbedding::new(4, NormalizationMethod::L2);
    
    let state = embedding.embed(&data).unwrap();
    assert_eq!(state.len(), 4); // Should be padded to 4
    
    // Test normalization preservation
    let norm_sqr: f64 = state.iter().map(|c| c.norm_sqr()).sum();
    assert_abs_diff_eq!(norm_sqr, 1.0, epsilon = 1e-10);
}

#[test] 
fn test_error_handling() {
    // Test invalid qubit index
    let mut circuit = Circuit::new(2);
    let result = circuit.add_gate(Box::new(Hadamard::new(5))); // Invalid qubit 5
    assert!(result.is_err());
    
    // Test dimension mismatch in expectation value
    let state = constants::zero_state(); // 2-element state
    let bad_observable = ndarray::Array2::eye(4); // 4x4 observable
    let result = utils::expectation_value(&state, &bad_observable);
    assert!(result.is_err());
    
    // Test invalid parameter count
    let mut rx_gate = RX::new(0, 0.0);
    let result = rx_gate.set_parameters(vec![1.0, 2.0]); // Too many parameters
    assert!(result.is_err());
}

#[test]
fn test_performance_with_larger_circuits() {
    // Test with maximum recommended qubits (should still be fast)
    let n_qubits = 10; // 2^10 = 1024 dimensional state space
    let mut circuit = Circuit::new(n_qubits);
    
    // Add gates to all qubits
    for i in 0..n_qubits {
        circuit.add_gate(Box::new(Hadamard::new(i))).unwrap();
    }
    
    // Add some entangling gates
    for i in 0..n_qubits-1 {
        circuit.add_gate(Box::new(CNOT::new(i, i+1))).unwrap();
    }
    
    let start_time = std::time::Instant::now();
    let state = circuit.execute().unwrap();
    let execution_time = start_time.elapsed();
    
    assert_eq!(state.len(), 1 << n_qubits);
    
    // Should complete within reasonable time (< 100ms for this size)
    assert!(execution_time.as_millis() < 100);
    
    // Verify normalization
    let norm_sqr: f64 = state.iter().map(|c| c.norm_sqr()).sum();
    assert_abs_diff_eq!(norm_sqr, 1.0, epsilon = 1e-10);
}