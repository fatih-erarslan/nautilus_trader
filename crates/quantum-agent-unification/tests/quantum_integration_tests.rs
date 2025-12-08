//! Comprehensive integration tests for quantum agent optimization system

use quantum_agent_unification::*;
use tokio_test;
use approx::assert_relative_eq;
use std::time::Duration;

/// Test all 12 quantum algorithms on benchmark functions
#[tokio::test]
async fn test_all_quantum_algorithms() {
    let mut optimizer = QuantumOptimizer::new();
    
    // Sphere function: f(x) = Σ(xi²)
    let sphere_problem = OptimizationProblem {
        dimensions: 5,
        bounds: vec![(-10.0, 10.0); 5],
        objective_function: |x| x.iter().map(|&xi| xi * xi).sum(),
        constraints: vec![],
        quantum_enhanced: true,
    };
    
    let result = optimizer.optimize_parallel(sphere_problem, 100).await;
    assert!(result.is_ok());
    
    let opt_result = result.unwrap();
    assert_eq!(opt_result.iterations, 100);
    assert!(opt_result.best_fitness >= 0.0);
    assert_eq!(opt_result.best_solution.len(), 5);
    assert_eq!(opt_result.convergence_history.len(), 100);
    
    // Check quantum metrics
    assert!(opt_result.quantum_metrics.coherence_time > 0.0);
    assert!(opt_result.quantum_metrics.quantum_speedup >= 1.0);
}

#[tokio::test]
async fn test_quantum_rastrigin_function() {
    let mut optimizer = QuantumOptimizer::new();
    
    // Rastrigin function: f(x) = A*n + Σ(xi² - A*cos(2π*xi))
    let rastrigin_problem = OptimizationProblem {
        dimensions: 3,
        bounds: vec![(-5.12, 5.12); 3],
        objective_function: |x| {
            let a = 10.0;
            let n = x.len() as f64;
            a * n + x.iter().map(|&xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
        },
        constraints: vec![],
        quantum_enhanced: true,
    };
    
    let result = optimizer.optimize_parallel(rastrigin_problem, 200).await;
    assert!(result.is_ok());
    
    let opt_result = result.unwrap();
    // Rastrigin has global minimum at origin with value 0
    // With quantum enhancement, we should get close to optimal
    assert!(opt_result.best_fitness < 50.0); // Reasonable for quantum optimization
}

#[tokio::test]
async fn test_quantum_rosenbrock_function() {
    let mut optimizer = QuantumOptimizer::new();
    
    // Rosenbrock function: f(x) = Σ(100*(x_{i+1} - xi²)² + (1 - xi)²)
    let rosenbrock_problem = OptimizationProblem {
        dimensions: 2,
        bounds: vec![(-5.0, 5.0); 2],
        objective_function: |x| {
            let mut sum = 0.0;
            for i in 0..x.len()-1 {
                sum += 100.0 * (x[i+1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
            }
            sum
        },
        constraints: vec![],
        quantum_enhanced: true,
    };
    
    let result = optimizer.optimize_parallel(rosenbrock_problem, 300).await;
    assert!(result.is_ok());
    
    let opt_result = result.unwrap();
    // Rosenbrock has global minimum at (1,1) with value 0
    println!("Rosenbrock best: {:?}, fitness: {}", opt_result.best_solution, opt_result.best_fitness);
    assert!(opt_result.best_fitness < 100.0); // Should be reasonable
}

#[test]
fn test_quantum_state_operations() {
    let mut quantum_state = QuantumState::new(3);
    
    // Test initial state
    assert_eq!(quantum_state.qubits.len(), 3);
    
    // Test superposition creation
    quantum_state.create_superposition();
    for qubit in &quantum_state.qubits {
        let p0 = qubit.prob_zero();
        let p1 = qubit.prob_one();
        assert_relative_eq!(p0, 0.5, epsilon = 1e-10);
        assert_relative_eq!(p1, 0.5, epsilon = 1e-10);
    }
    
    // Test entanglement
    let entanglement_result = quantum_state.entangle(0, 1);
    assert!(entanglement_result.is_ok());
    assert_eq!(quantum_state.entangled_pairs.len(), 1);
    assert_eq!(quantum_state.entangled_pairs[0], (0, 1));
    
    // Test quantum tunneling
    let tunnel_prob = quantum_state.quantum_tunnel(10.0, 5.0);
    assert!(tunnel_prob >= 0.0 && tunnel_prob <= 1.0);
    
    // Test coherence measurement
    let coherence = quantum_state.coherence();
    assert!(coherence >= 0.0 && coherence <= 1.0);
    
    // Test decoherence
    let initial_coherence = quantum_state.coherence();
    quantum_state.apply_decoherence(0.1);
    let final_coherence = quantum_state.coherence();
    assert!(final_coherence <= initial_coherence);
}

#[test]
fn test_quantum_entanglement_network() {
    let mut entanglement = QuantumEntanglement::new(4);
    
    // Test Bell state creation
    let bell_state = entanglement.create_bell_state(0, 1, quantum_entanglement::BellStateType::PhiPlus);
    assert!(bell_state.is_ok());
    
    let state = bell_state.unwrap();
    assert_eq!(state.qubit_indices, (0, 1));
    assert!(state.correlation.abs() > 0.0);
    
    // Test quantum channel creation
    let channel_id = entanglement.create_quantum_channel(0, 2, 100.0);
    assert!(channel_id.is_ok());
    assert_eq!(channel_id.unwrap(), 0);
    
    // Test entanglement strength
    let strength = entanglement.get_entanglement_strength(0, 1);
    assert!(strength > 0.0);
    
    // Test network entropy
    let entropy = entanglement.measure_entanglement_entropy();
    assert!(entropy >= 0.0);
    
    // Test Bell violation (quantum behavior)
    let violation = entanglement.check_bell_violation();
    assert!(violation); // Should violate Bell inequality with quantum entanglement
}

#[test]
fn test_quantum_metrics_collection() {
    let mut collector = quantum_metrics::QuantumMetricsCollector::new();
    
    // Test system metrics update
    collector.update_system_metrics(75.0, 2048 * 1024, 85.0);
    
    let metrics = collector.get_metrics();
    assert_eq!(metrics.system_metrics.cpu_utilization, 75.0);
    assert_eq!(metrics.system_metrics.memory_usage_bytes, 2048 * 1024);
    assert_eq!(metrics.system_metrics.simd_utilization, 85.0);
    
    // Test quantum metrics update
    collector.update_quantum_metrics(12, 1000.0, 0.95);
    assert_eq!(metrics.quantum_computing_metrics.total_qubits, 12);
    assert_eq!(metrics.quantum_computing_metrics.gate_operations_per_sec, 1000.0);
    assert_eq!(metrics.quantum_computing_metrics.quantum_fidelity, 0.95);
    
    // Test operation recording
    collector.record_quantum_operation("quantum_evolution", Duration::from_millis(5));
    assert!(metrics.system_metrics.total_operations > 0);
    
    // Test tunneling event recording
    collector.record_tunneling_event(QuantumAlgorithm::QuantumParticleSwarm);
    
    // Test entanglement recording
    collector.record_entanglement(0.8);
    assert!(metrics.quantum_computing_metrics.entangled_pairs > 0);
    
    // Test efficiency score
    let efficiency = collector.quantum_efficiency_score();
    assert!(efficiency >= 0.0 && efficiency <= 1.0);
    
    // Test JSON export
    let json_result = collector.export_json();
    assert!(json_result.is_ok());
    assert!(!json_result.unwrap().is_empty());
    
    // Test report generation
    let report = collector.generate_report();
    assert!(report.contains("Quantum Agent Optimization Report"));
    assert!(report.contains("Total Operations"));
    assert!(report.contains("Quantum Efficiency"));
}

#[test]
fn test_quantum_algorithms_factory() {
    let algorithms = quantum_agents::QuantumAlgorithmFactory::create_all_algorithms();
    
    // Test all 12 algorithms are created
    assert_eq!(algorithms.len(), 12);
    
    // Test each algorithm type
    let expected_types = [
        QuantumAlgorithm::QuantumParticleSwarm,
        QuantumAlgorithm::QuantumGeneticAlgorithm,
        QuantumAlgorithm::QuantumAnnealing,
        QuantumAlgorithm::QuantumDifferentialEvolution,
        QuantumAlgorithm::QuantumFirefly,
        QuantumAlgorithm::QuantumBeeColony,
        QuantumAlgorithm::QuantumGreyWolf,
        QuantumAlgorithm::QuantumCuckooSearch,
        QuantumAlgorithm::QuantumBatAlgorithm,
        QuantumAlgorithm::QuantumWhaleOptimization,
        QuantumAlgorithm::QuantumMothFlame,
        QuantumAlgorithm::QuantumSalpSwarm,
    ];
    
    for &expected_type in &expected_types {
        assert!(algorithms.contains_key(&expected_type));
        let algorithm = algorithms.get(&expected_type).unwrap();
        assert_eq!(algorithm.algorithm_type(), expected_type);
    }
}

#[tokio::test]
async fn test_quantum_algorithm_individual_initialization() {
    // Test individual algorithm initialization and basic operation
    let problem = OptimizationProblem {
        dimensions: 2,
        bounds: vec![(-1.0, 1.0); 2],
        objective_function: |x| x[0] * x[0] + x[1] * x[1],
        constraints: vec![],
        quantum_enhanced: true,
    };
    
    // Test Quantum PSO
    let mut qpso = quantum_agents::QuantumParticleSwarmOptimizer::new();
    qpso.initialize_quantum_population(&problem).unwrap();
    assert_eq!(qpso.particles.len(), 30);
    
    qpso.quantum_evolve_step(&problem, 0).unwrap();
    assert!(qpso.get_best_solution().is_some());
    
    // Test Quantum Annealing
    let mut qa = quantum_agents::QuantumAnnealingAlgorithm::new();
    qa.initialize_quantum_population(&problem).unwrap();
    assert!(qa.current_solution.is_some());
    
    qa.quantum_evolve_step(&problem, 0).unwrap();
    assert!(qa.temperature < qa.annealing_schedule.initial_temperature);
    
    // Test Quantum Firefly
    let mut qfa = quantum_agents::QuantumFireflyAlgorithm::new();
    qfa.initialize_quantum_population(&problem).unwrap();
    assert_eq!(qfa.fireflies.len(), 25);
    
    qfa.quantum_evolve_step(&problem, 0).unwrap();
    assert!(qfa.get_best_solution().is_some());
}

#[test]
fn test_quantum_bit_operations() {
    let mut qubit = QuantumBit::new();
    
    // Test initial state |0⟩
    assert_relative_eq!(qubit.prob_zero(), 1.0, epsilon = 1e-10);
    assert_relative_eq!(qubit.prob_one(), 0.0, epsilon = 1e-10);
    
    // Test superposition creation
    qubit.create_superposition();
    assert_relative_eq!(qubit.prob_zero(), 0.5, epsilon = 1e-10);
    assert_relative_eq!(qubit.prob_one(), 0.5, epsilon = 1e-10);
    
    // Test rotations
    let mut qubit2 = QuantumBit::new();
    qubit2.rotate_x(std::f64::consts::PI);
    assert_relative_eq!(qubit2.prob_zero(), 0.0, epsilon = 1e-10);
    assert_relative_eq!(qubit2.prob_one(), 1.0, epsilon = 1e-10);
    
    // Test Bloch sphere representation
    let bloch = BlochSphere::from_qubit(&qubit);
    assert!(bloch.x.abs() <= 1.0);
    assert!(bloch.y.abs() <= 1.0);
    assert!(bloch.z.abs() <= 1.0);
    
    // Test measurement
    let mut qubit3 = QuantumBit::new();
    qubit3.create_superposition();
    let measurement = qubit3.measure();
    
    // After measurement, should be in definite state
    if measurement {
        assert_relative_eq!(qubit3.prob_one(), 1.0, epsilon = 1e-10);
    } else {
        assert_relative_eq!(qubit3.prob_zero(), 1.0, epsilon = 1e-10);
    }
}

#[tokio::test]
async fn test_quantum_optimization_convergence() {
    let mut optimizer = QuantumOptimizer::new();
    
    // Simple quadratic function with known minimum
    let quadratic_problem = OptimizationProblem {
        dimensions: 1,
        bounds: vec![(-10.0, 10.0)],
        objective_function: |x| (x[0] - 3.0).powi(2) + 5.0, // Minimum at x=3, value=5
        constraints: vec![],
        quantum_enhanced: true,
    };
    
    let result = optimizer.optimize_parallel(quadratic_problem, 150).await.unwrap();
    
    // Check convergence
    assert!(result.convergence_history.len() == 150);
    
    // Check that convergence is improving (generally decreasing)
    let initial_fitness = result.convergence_history[0];
    let final_fitness = result.convergence_history.last().unwrap();
    assert!(final_fitness <= &initial_fitness);
    
    // Check that we're getting close to the known optimum
    println!("Best solution: {:?}, fitness: {}", result.best_solution, result.best_fitness);
    assert!(result.best_fitness < initial_fitness);
    
    // With quantum enhancement, should be closer to optimum
    assert!((result.best_solution[0] - 3.0).abs() < 5.0); // Within reasonable range
    assert!((result.best_fitness - 5.0).abs() < 20.0); // Close to known minimum
}

#[test]
fn test_error_handling() {
    let mut quantum_state = QuantumState::new(2);
    
    // Test out-of-bounds entanglement
    let bad_entanglement = quantum_state.entangle(0, 5);
    assert!(bad_entanglement.is_err());
    
    // Test quantum measurements
    let mut entanglement = QuantumEntanglement::new(2);
    let bad_channel = entanglement.create_quantum_channel(10, 20, 100.0);
    assert!(bad_channel.is_err());
}

/// Performance benchmark test (not run by default)
#[tokio::test]
#[ignore]
async fn benchmark_quantum_optimization_performance() {
    let mut optimizer = QuantumOptimizer::new();
    
    // Large-scale optimization problem
    let large_problem = OptimizationProblem {
        dimensions: 20,
        bounds: vec![(-100.0, 100.0); 20],
        objective_function: |x| x.iter().map(|&xi| xi * xi).sum(),
        constraints: vec![],
        quantum_enhanced: true,
    };
    
    let start_time = std::time::Instant::now();
    let result = optimizer.optimize_parallel(large_problem, 500).await.unwrap();
    let elapsed = start_time.elapsed();
    
    println!("Performance benchmark results:");
    println!("  Dimensions: 20");
    println!("  Iterations: 500");
    println!("  Time elapsed: {:?}", elapsed);
    println!("  Best fitness: {}", result.best_fitness);
    println!("  Quantum speedup: {:.2}x", result.quantum_metrics.quantum_speedup);
    println!("  Final coherence: {:.3}", result.quantum_metrics.coherence_time);
    
    // Performance assertions
    assert!(elapsed < Duration::from_secs(60)); // Should complete within 1 minute
    assert!(result.quantum_metrics.quantum_speedup >= 1.0);
    assert!(result.best_fitness >= 0.0); // Sphere function minimum is 0
}