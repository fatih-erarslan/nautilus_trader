//! Integration tests for quantum backend

use quantum_backend::*;
use quantum_core::*;
use ndarray::Array1;

#[tokio::test]
async fn test_complete_quantum_pipeline() {
    // Test the complete quantum backend pipeline
    let backend = QuantumBackend::new().await.unwrap();
    
    // Create a simple quantum circuit
    let mut circuit = QuantumCircuit::new(4);
    circuit.add_gate(QuantumGate::hadamard(0));
    circuit.add_gate(QuantumGate::cnot(0, 1));
    circuit.add_gate(QuantumGate::cnot(1, 2));
    circuit.add_gate(QuantumGate::cnot(2, 3));
    
    // Execute circuit
    let result = backend.execute_circuit(&circuit).await.unwrap();
    
    // Verify result
    assert_eq!(result.probabilities.len(), 16); // 2^4 states
    assert!((result.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    assert!(result.execution_time_ns < 10_000_000); // < 10ms
}

#[tokio::test]
async fn test_nash_equilibrium_solver() {
    let backend = QuantumBackend::new().await.unwrap();
    
    // Prisoner's dilemma
    let payoff = ndarray::Array2::from_shape_vec(
        (2, 2),
        vec![3.0, 0.0, 5.0, 1.0]
    ).unwrap();
    
    let solution = backend.solve_nash_equilibrium(&payoff, 4).await.unwrap();
    
    // Verify solution properties
    assert_eq!(solution.strategy_player1.len(), 2);
    assert_eq!(solution.strategy_player2.len(), 2);
    assert!((solution.strategy_player1.iter().sum::<f64>() - 1.0).abs() < 0.01);
    assert!((solution.strategy_player2.iter().sum::<f64>() - 1.0).abs() < 0.01);
    assert!(solution.quantum_advantage > 0.0);
}

#[tokio::test]
async fn test_vqe_optimization() {
    let backend = QuantumBackend::new().await.unwrap();
    
    // Simple H2 molecule Hamiltonian
    let hamiltonian = QuantumHamiltonian {
        terms: vec![
            (-1.0523, vec![PauliOperator::I, PauliOperator::I]),
            (0.3979, vec![PauliOperator::Z, PauliOperator::I]),
            (-0.3979, vec![PauliOperator::I, PauliOperator::Z]),
            (-0.0112, vec![PauliOperator::Z, PauliOperator::Z]),
            (0.1809, vec![PauliOperator::X, PauliOperator::X]),
        ],
        num_qubits: 2,
    };
    
    let ansatz = QuantumAnsatz {
        ansatz_type: AnsatzType::HardwareEfficient,
        num_qubits: 2,
        depth: 2,
        entanglement: EntanglementType::Linear,
    };
    
    let result = backend.run_vqe(&hamiltonian, &ansatz).await.unwrap();
    
    // Ground state energy should be around -1.85
    assert!((result.energy + 1.85).abs() < 0.2);
    assert!(result.iterations > 0);
}

#[tokio::test]
async fn test_qaoa_optimization() {
    let backend = QuantumBackend::new().await.unwrap();
    
    // Simple MaxCut problem
    let cost_matrix = ndarray::Array2::from_shape_vec(
        (4, 4),
        vec![
            0.0, 1.0, 1.0, 0.0,
            1.0, 0.0, 1.0, 1.0,
            1.0, 1.0, 0.0, 1.0,
            0.0, 1.0, 1.0, 0.0,
        ]
    ).unwrap();
    
    let problem = CombOptProblem {
        problem_type: ProblemType::MaxCut,
        cost_matrix,
        constraints: vec![],
    };
    
    let result = backend.run_qaoa(&problem, 2).await.unwrap();
    
    // Should find a valid solution
    assert!(!result.solution_bitstring.is_empty());
    assert!(result.cost < 0.0); // Negative because we minimize
    assert!(result.probabilities.len() == 16); // 2^4 states
}

#[tokio::test]
async fn test_hybrid_quantum_classical() {
    let backend = Arc::new(QuantumBackend::new().await.unwrap());
    
    let config = HybridStrategyConfig {
        quantum_layers: 2,
        classical_layers: 2,
        feature_dimension: 4,
        quantum_encoding: EncodingType::AngleEncoding,
        optimization_objective: OptimizationObjective::MaximizeSharpe,
    };
    
    let model = HybridTradingModel::new(backend, config).await.unwrap();
    
    // Test prediction
    let features = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.1]);
    let context = MarketContext {
        volatility: 0.2,
        correlation: 0.3,
        market_regime: MarketRegime::Trending,
        volume_profile: VolumeProfile {
            current: 1000000.0,
            average: 800000.0,
            trend: 0.1,
        },
    };
    
    let signal = model.predict(&features, &context).await.unwrap();
    
    // Verify signal properties
    assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
    assert!(signal.risk_score >= 0.0 && signal.risk_score <= 1.0);
    assert!(signal.quantum_contribution.abs() <= 1.0);
    assert!(signal.classical_contribution.abs() <= 1.0);
    assert!(matches!(signal.action, TradeAction::Buy | TradeAction::Sell | TradeAction::Hold | TradeAction::StrongBuy | TradeAction::StrongSell));
}

#[tokio::test]
async fn test_circuit_optimization() {
    let optimizer = CircuitOptimizer::new().await.unwrap();
    
    // Create circuit with redundant gates
    let mut circuit = QuantumCircuit::new(4);
    circuit.add_gate(QuantumGate::hadamard(0));
    circuit.add_gate(QuantumGate::hadamard(0)); // Should cancel
    circuit.add_gate(QuantumGate::rx(1, 0.5));
    circuit.add_gate(QuantumGate::rx(1, 0.3)); // Should merge
    circuit.add_gate(QuantumGate::pauli_x(2));
    circuit.add_gate(QuantumGate::pauli_x(2)); // Should cancel
    
    let original_gates = circuit.gate_count();
    let optimized = optimizer.optimize(&circuit).await.unwrap();
    
    // Should have fewer gates after optimization
    assert!(optimized.gate_count() < original_gates);
    assert!(optimized.depth() <= circuit.depth());
}

#[tokio::test]
async fn test_performance_targets() {
    let backend = QuantumBackend::new().await.unwrap();
    
    // Test that we achieve <10ms execution for reasonable circuits
    let mut circuit = QuantumCircuit::new(8);
    
    // Create realistic trading quantum circuit
    for i in 0..8 {
        circuit.add_gate(QuantumGate::hadamard(i));
    }
    
    for i in 0..7 {
        circuit.add_gate(QuantumGate::cnot(i, i + 1));
    }
    
    for i in 0..8 {
        circuit.add_gate(QuantumGate::rz(i, 0.5));
    }
    
    let start = std::time::Instant::now();
    let result = backend.execute_circuit(&circuit).await.unwrap();
    let elapsed = start.elapsed();
    
    // Verify performance target
    assert!(elapsed.as_millis() < 10); // < 10ms
    assert!(result.execution_time_ns < 10_000_000); // < 10ms in nanoseconds
    
    // Verify correctness
    assert_eq!(result.probabilities.len(), 256); // 2^8 states
    assert!((result.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

use std::sync::Arc;