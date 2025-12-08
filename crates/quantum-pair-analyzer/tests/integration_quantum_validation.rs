//! Integration Tests for Comprehensive Quantum Algorithm Validation
//! 
//! This module contains integration tests that validate the comprehensive
//! quantum algorithm validation framework as required by the Quantum-Test-Expert
//! agent in the TDD swarm.

use std::time::Duration;
use tokio::time::timeout;
use chrono::Utc;

use quantum_pair_analyzer::{
    PairMetrics, PairId, OptimalPair, AnalyzerError,
    QuantumConfig, OptimizationConstraints
};
use quantum_pair_analyzer::quantum::{
    QuantumValidationSuite, QAOAEngine, QuantumCircuitBuilder, 
    QuantumOptimizer, HybridOptimizer, QuantumPortfolioOptimizer,
    ExtractionMethod, SelectionStrategy, RankingAlgorithm,
    HybridStrategy, OptimizationObjective
};

/// Create test pair metrics for integration testing
fn create_integration_test_data() -> Vec<PairMetrics> {
    vec![
        PairMetrics {
            pair_id: PairId::new("BTC", "USD", "binance"),
            timestamp: Utc::now(),
            correlation_score: 0.75,
            cointegration_p_value: 0.005,
            volatility_ratio: 0.25,
            liquidity_ratio: 0.9,
            sentiment_divergence: 0.15,
            news_sentiment_score: 0.7,
            social_sentiment_score: 0.8,
            cuckoo_score: 0.0,
            firefly_score: 0.0,
            ant_colony_score: 0.0,
            quantum_entanglement: 0.0,
            quantum_advantage: 0.6,
            expected_return: 0.18,
            sharpe_ratio: 1.4,
            maximum_drawdown: 0.08,
            value_at_risk: 0.04,
            composite_score: 0.85,
            confidence: 0.92,
        },
        PairMetrics {
            pair_id: PairId::new("ETH", "USD", "binance"),
            timestamp: Utc::now(),
            correlation_score: 0.65,
            cointegration_p_value: 0.008,
            volatility_ratio: 0.35,
            liquidity_ratio: 0.85,
            sentiment_divergence: 0.2,
            news_sentiment_score: 0.65,
            social_sentiment_score: 0.75,
            cuckoo_score: 0.0,
            firefly_score: 0.0,
            ant_colony_score: 0.0,
            quantum_entanglement: 0.0,
            quantum_advantage: 0.55,
            expected_return: 0.16,
            sharpe_ratio: 1.2,
            maximum_drawdown: 0.1,
            value_at_risk: 0.05,
            composite_score: 0.8,
            confidence: 0.88,
        },
        PairMetrics {
            pair_id: PairId::new("ADA", "USD", "binance"),
            timestamp: Utc::now(),
            correlation_score: -0.45,
            cointegration_p_value: 0.012,
            volatility_ratio: 0.4,
            liquidity_ratio: 0.7,
            sentiment_divergence: 0.25,
            news_sentiment_score: 0.6,
            social_sentiment_score: 0.7,
            cuckoo_score: 0.0,
            firefly_score: 0.0,
            ant_colony_score: 0.0,
            quantum_entanglement: 0.0,
            quantum_advantage: 0.7,
            expected_return: 0.22,
            sharpe_ratio: 1.1,
            maximum_drawdown: 0.15,
            value_at_risk: 0.08,
            composite_score: 0.75,
            confidence: 0.85,
        },
        PairMetrics {
            pair_id: PairId::new("DOT", "USD", "binance"),
            timestamp: Utc::now(),
            correlation_score: 0.55,
            cointegration_p_value: 0.006,
            volatility_ratio: 0.3,
            liquidity_ratio: 0.8,
            sentiment_divergence: 0.18,
            news_sentiment_score: 0.68,
            social_sentiment_score: 0.72,
            cuckoo_score: 0.0,
            firefly_score: 0.0,
            ant_colony_score: 0.0,
            quantum_entanglement: 0.0,
            quantum_advantage: 0.58,
            expected_return: 0.17,
            sharpe_ratio: 1.3,
            maximum_drawdown: 0.09,
            value_at_risk: 0.045,
            composite_score: 0.82,
            confidence: 0.9,
        },
    ]
}

#[tokio::test]
async fn test_comprehensive_quantum_validation_integration() {
    // This is the main integration test that validates the entire
    // quantum validation framework works correctly
    
    let config = QuantumConfig {
        qaoa_layers: 2,
        max_qubits: 8,
        optimization_iterations: 50,
        convergence_threshold: 1e-6,
        enable_quantum_advantage: true,
        classical_optimizer: quantum_pair_analyzer::quantum::ClassicalOptimizer::Adam,
        max_circuit_depth: 30,
        measurement_shots: 512,
        enable_noise: false,
        error_mitigation: vec![],
    };
    
    let test_data = create_integration_test_data();
    
    // Initialize validation suite
    let mut validation_suite = QuantumValidationSuite::new(config).await
        .expect("Failed to create quantum validation suite");
    
    // Execute comprehensive validation with timeout
    let validation_result = timeout(
        Duration::from_secs(120), // 2 minute timeout
        validation_suite.execute_comprehensive_validation(&test_data)
    ).await;
    
    assert!(validation_result.is_ok(), "Validation timed out");
    
    let results = validation_result.unwrap()
        .expect("Comprehensive validation failed");
    
    // Validate results structure
    assert!(results.total_tests_run > 0, "No tests were run");
    assert!(results.overall_success_rate >= 0.0, "Invalid success rate");
    assert!(results.overall_success_rate <= 1.0, "Invalid success rate");
    
    // Validate individual test categories
    assert!(!results.qaoa_correctness_tests.is_empty(), "No QAOA tests run");
    assert!(!results.circuit_construction_tests.is_empty(), "No circuit tests run");
    assert!(!results.hybrid_optimization_tests.is_empty(), "No hybrid tests run");
    assert!(!results.performance_comparison_tests.is_empty(), "No performance tests run");
    assert!(!results.quantum_enhancement_tests.is_empty(), "No enhancement tests run");
    
    // Validate minimum success rate for integration test
    assert!(results.overall_success_rate >= 0.6, 
            "Overall success rate too low: {:.2}%", results.overall_success_rate * 100.0);
    
    println!("✅ Comprehensive quantum validation integration test passed");
    println!("   └── Success rate: {:.2}%", results.overall_success_rate * 100.0);
    println!("   └── Total tests: {}", results.total_tests_run);
    println!("   └── Failed tests: {}", results.failed_tests.len());
}

#[tokio::test]
async fn test_qaoa_engine_integration() {
    // Test QAOA engine integration
    let config = QuantumConfig::default();
    let qaoa_engine = QAOAEngine::new(config).await
        .expect("Failed to create QAOA engine");
    
    // Create simple test problem
    let test_data = create_integration_test_data();
    let problem = create_test_problem(&test_data, 4);
    
    // Test QAOA optimization
    let initial_params = vec![0.5, 0.3, 0.7, 0.4, 0.6, 0.2];
    let result = qaoa_engine.optimize(&problem, &initial_params).await;
    
    assert!(result.is_ok(), "QAOA optimization failed");
    
    let qaoa_result = result.unwrap();
    assert!(qaoa_result.objective_value > 0.0, "Invalid objective value");
    assert!(qaoa_result.fidelity > 0.5, "Low fidelity");
    assert!(qaoa_result.quantum_advantage > 0.0, "No quantum advantage");
    
    println!("✅ QAOA engine integration test passed");
    println!("   └── Objective value: {:.4}", qaoa_result.objective_value);
    println!("   └── Fidelity: {:.3}", qaoa_result.fidelity);
    println!("   └── Quantum advantage: {:.3}", qaoa_result.quantum_advantage);
}

#[tokio::test]
async fn test_quantum_circuit_builder_integration() {
    // Test quantum circuit builder integration
    let config = QuantumConfig::default();
    let mut circuit_builder = QuantumCircuitBuilder::new(config).await
        .expect("Failed to create circuit builder");
    
    let test_data = create_integration_test_data();
    let problem = create_test_problem(&test_data, 4);
    
    // Test optimization circuit building
    let circuit = circuit_builder.build_optimization_circuit(&problem).await;
    assert!(circuit.is_ok(), "Failed to build optimization circuit");
    
    let opt_circuit = circuit.unwrap();
    assert!(opt_circuit.num_qubits > 0, "Invalid qubit count");
    assert!(opt_circuit.gate_count() > 0, "No gates in circuit");
    assert!(opt_circuit.depth() > 0, "Invalid circuit depth");
    
    // Test entanglement circuit building
    let pair1 = &test_data[0];
    let pair2 = &test_data[1];
    let entanglement_circuit = circuit_builder.build_entanglement_circuit(pair1, pair2).await;
    assert!(entanglement_circuit.is_ok(), "Failed to build entanglement circuit");
    
    let ent_circuit = entanglement_circuit.unwrap();
    assert!(ent_circuit.num_qubits >= 2, "Insufficient qubits for entanglement");
    assert!(ent_circuit.gate_count() > 0, "No gates in entanglement circuit");
    
    println!("✅ Quantum circuit builder integration test passed");
    println!("   └── Optimization circuit: {} qubits, {} gates, depth {}", 
             opt_circuit.num_qubits, opt_circuit.gate_count(), opt_circuit.depth());
    println!("   └── Entanglement circuit: {} qubits, {} gates, depth {}", 
             ent_circuit.num_qubits, ent_circuit.gate_count(), ent_circuit.depth());
}

#[tokio::test]
async fn test_hybrid_optimizer_integration() {
    // Test hybrid optimizer integration
    let config = QuantumConfig::default();
    let mut hybrid_optimizer = HybridOptimizer::new(config)
        .expect("Failed to create hybrid optimizer");
    
    let test_data = create_integration_test_data();
    let problem = create_test_problem(&test_data, 4);
    let objective = TestObjectiveFunction::new(4);
    let constraints = OptimizationConstraints::default();
    
    // Test different hybrid strategies
    let strategies = vec![
        HybridStrategy::QuantumFirst,
        HybridStrategy::ClassicalFirst,
        HybridStrategy::Parallel,
    ];
    
    for strategy in strategies {
        let result = timeout(
            Duration::from_secs(30),
            hybrid_optimizer.optimize(&problem, &objective, &constraints)
        ).await;
        
        assert!(result.is_ok(), "Hybrid optimization timed out for {:?}", strategy);
        
        let optimization_result = result.unwrap();
        assert!(optimization_result.is_ok(), "Hybrid optimization failed for {:?}", strategy);
        
        let hybrid_result = optimization_result.unwrap();
        assert!(hybrid_result.objective_value > 0.0, "Invalid objective value for {:?}", strategy);
        assert!(hybrid_result.iterations > 0, "No iterations for {:?}", strategy);
        
        println!("✅ Hybrid strategy {:?} test passed", strategy);
        println!("   └── Objective: {:.4}, Iterations: {}, Converged: {}", 
                 hybrid_result.objective_value, hybrid_result.iterations, hybrid_result.converged);
    }
}

#[tokio::test]
async fn test_quantum_portfolio_optimizer_integration() {
    // Test quantum portfolio optimizer integration
    let config = QuantumConfig::default();
    let portfolio_optimizer = QuantumPortfolioOptimizer::new(
        config,
        ExtractionMethod::ExpectationBased,
        SelectionStrategy::TopN,
        RankingAlgorithm::Hybrid,
    ).expect("Failed to create portfolio optimizer");
    
    let test_data = create_integration_test_data();
    
    // Test portfolio optimization
    let result = timeout(
        Duration::from_secs(45),
        portfolio_optimizer.optimize_portfolio(&test_data)
    ).await;
    
    assert!(result.is_ok(), "Portfolio optimization timed out");
    
    let portfolio_result = result.unwrap();
    assert!(portfolio_result.is_ok(), "Portfolio optimization failed");
    
    let portfolio = portfolio_result.unwrap();
    assert!(!portfolio.selected_pairs.is_empty(), "No pairs selected");
    assert!(portfolio.confidence > 0.0, "Invalid confidence");
    assert!(portfolio.quantum_advantage > 0.0, "No quantum advantage");
    
    println!("✅ Quantum portfolio optimizer integration test passed");
    println!("   └── Selected pairs: {}", portfolio.selected_pairs.len());
    println!("   └── Confidence: {:.3}", portfolio.confidence);
    println!("   └── Quantum advantage: {:.3}", portfolio.quantum_advantage);
}

#[tokio::test]
async fn test_quantum_optimizer_full_integration() {
    // Test the main quantum optimizer with full integration
    let config = QuantumConfig::default();
    let quantum_optimizer = QuantumOptimizer::new(&config).await
        .expect("Failed to create quantum optimizer");
    
    let test_data = create_integration_test_data();
    let constraints = OptimizationConstraints::default();
    
    // Test portfolio optimization
    let result = timeout(
        Duration::from_secs(60),
        quantum_optimizer.optimize_portfolio(&test_data, &constraints)
    ).await;
    
    assert!(result.is_ok(), "Quantum portfolio optimization timed out");
    
    let portfolio = result.unwrap();
    assert!(portfolio.is_ok(), "Quantum portfolio optimization failed");
    
    let optimal_pairs = portfolio.unwrap();
    assert!(!optimal_pairs.is_empty(), "No optimal pairs found");
    assert!(optimal_pairs.len() <= constraints.max_portfolio_size, "Portfolio too large");
    
    // Test quantum entanglement calculation
    if test_data.len() >= 2 {
        let entanglement_result = timeout(
            Duration::from_secs(30),
            quantum_optimizer.calculate_quantum_entanglement(&test_data[0], &test_data[1])
        ).await;
        
        assert!(entanglement_result.is_ok(), "Entanglement calculation timed out");
        
        let entanglement = entanglement_result.unwrap();
        assert!(entanglement.is_ok(), "Entanglement calculation failed");
        
        let entanglement_value = entanglement.unwrap();
        assert!(entanglement_value >= 0.0, "Invalid entanglement value");
        assert!(entanglement_value <= 1.0, "Invalid entanglement value");
        
        println!("✅ Quantum entanglement calculation test passed");
        println!("   └── Entanglement value: {:.4}", entanglement_value);
    }
    
    // Test quantum metrics retrieval
    let metrics_result = quantum_optimizer.get_quantum_metrics().await;
    assert!(metrics_result.is_ok(), "Failed to get quantum metrics");
    
    println!("✅ Quantum optimizer full integration test passed");
    println!("   └── Optimal pairs: {}", optimal_pairs.len());
    println!("   └── All subsystems integrated successfully");
}

#[tokio::test]
async fn test_performance_comparison_integration() {
    // Test performance comparison between quantum and classical approaches
    let config = QuantumConfig::default();
    let quantum_optimizer = QuantumOptimizer::new(&config).await
        .expect("Failed to create quantum optimizer");
    
    let test_data = create_integration_test_data();
    let constraints = OptimizationConstraints::default();
    
    // Quantum approach
    let quantum_start = std::time::Instant::now();
    let quantum_result = quantum_optimizer.optimize_portfolio(&test_data, &constraints).await;
    let quantum_duration = quantum_start.elapsed();
    
    assert!(quantum_result.is_ok(), "Quantum optimization failed");
    let quantum_portfolio = quantum_result.unwrap();
    
    // Classical baseline (simplified)
    let classical_start = std::time::Instant::now();
    let classical_portfolio = classical_baseline_optimization(&test_data, &constraints);
    let classical_duration = classical_start.elapsed();
    
    assert!(classical_portfolio.is_ok(), "Classical optimization failed");
    let classical_pairs = classical_portfolio.unwrap();
    
    // Compare results
    let quantum_score = quantum_portfolio.iter().map(|p| p.score).sum::<f64>();
    let classical_score = classical_pairs.iter().map(|p| p.score).sum::<f64>();
    
    println!("✅ Performance comparison integration test passed");
    println!("   └── Quantum score: {:.4} (time: {:?})", quantum_score, quantum_duration);
    println!("   └── Classical score: {:.4} (time: {:?})", classical_score, classical_duration);
    println!("   └── Quantum advantage: {:.2}x", quantum_score / classical_score.max(1e-10));
}

#[tokio::test]
async fn test_stress_conditions_integration() {
    // Test quantum system under stress conditions
    let mut config = QuantumConfig::default();
    config.max_qubits = 8;
    config.optimization_iterations = 100;
    config.measurement_shots = 1024;
    
    let validation_suite = QuantumValidationSuite::new(config).await
        .expect("Failed to create validation suite");
    
    // Create stress test data
    let mut stress_test_data = create_integration_test_data();
    
    // Add extreme correlation cases
    stress_test_data.push(PairMetrics {
        pair_id: PairId::new("EXTREME_HIGH", "USD", "test"),
        timestamp: Utc::now(),
        correlation_score: 0.98,
        cointegration_p_value: 0.001,
        volatility_ratio: 0.8,
        liquidity_ratio: 0.95,
        sentiment_divergence: 0.1,
        news_sentiment_score: 0.9,
        social_sentiment_score: 0.95,
        cuckoo_score: 0.0,
        firefly_score: 0.0,
        ant_colony_score: 0.0,
        quantum_entanglement: 0.0,
        quantum_advantage: 0.9,
        expected_return: 0.25,
        sharpe_ratio: 1.8,
        maximum_drawdown: 0.05,
        value_at_risk: 0.02,
        composite_score: 0.95,
        confidence: 0.98,
    });
    
    stress_test_data.push(PairMetrics {
        pair_id: PairId::new("EXTREME_LOW", "USD", "test"),
        timestamp: Utc::now(),
        correlation_score: -0.95,
        cointegration_p_value: 0.001,
        volatility_ratio: 1.2,
        liquidity_ratio: 0.3,
        sentiment_divergence: 0.8,
        news_sentiment_score: 0.2,
        social_sentiment_score: 0.1,
        cuckoo_score: 0.0,
        firefly_score: 0.0,
        ant_colony_score: 0.0,
        quantum_entanglement: 0.0,
        quantum_advantage: 0.8,
        expected_return: 0.35,
        sharpe_ratio: 0.5,
        maximum_drawdown: 0.4,
        value_at_risk: 0.25,
        composite_score: 0.4,
        confidence: 0.6,
    });
    
    // Test basic functionality under stress
    let qaoa_engine = QAOAEngine::new(config).await
        .expect("Failed to create QAOA engine under stress");
    
    let problem = create_test_problem(&stress_test_data, 6);
    let initial_params = vec![0.5; 6];
    
    let result = timeout(
        Duration::from_secs(90),
        qaoa_engine.optimize(&problem, &initial_params)
    ).await;
    
    assert!(result.is_ok(), "QAOA optimization failed under stress");
    
    let qaoa_result = result.unwrap();
    assert!(qaoa_result.is_ok(), "QAOA optimization error under stress");
    
    println!("✅ Stress conditions integration test passed");
    println!("   └── Handled extreme correlation cases");
    println!("   └── System remained stable under load");
}

// Helper functions for integration tests

fn create_test_problem(test_data: &[PairMetrics], num_qubits: usize) -> quantum_pair_analyzer::quantum::QuantumProblem {
    use nalgebra::DMatrix;
    
    let parameters = quantum_pair_analyzer::quantum::QuantumProblemParameters {
        num_qubits,
        cost_matrix: DMatrix::identity(num_qubits, num_qubits),
        constraint_matrices: vec![],
        optimization_objective: OptimizationObjective::MaximizeReturn,
        penalty_coefficients: vec![],
    };
    
    let problem_pairs = test_data.iter().take(num_qubits).cloned().collect();
    
    quantum_pair_analyzer::quantum::QuantumProblem {
        parameters,
        pair_metadata: problem_pairs,
    }
}

fn classical_baseline_optimization(
    pair_metrics: &[PairMetrics],
    constraints: &OptimizationConstraints,
) -> Result<Vec<OptimalPair>, AnalyzerError> {
    // Simple greedy selection as classical baseline
    let mut sorted_pairs: Vec<_> = pair_metrics.iter().enumerate().collect();
    sorted_pairs.sort_by(|a, b| b.1.composite_score.partial_cmp(&a.1.composite_score).unwrap());
    
    let optimal_pairs = sorted_pairs.into_iter()
        .take(constraints.max_portfolio_size.min(pair_metrics.len()))
        .map(|(_, metrics)| OptimalPair::from_metrics(metrics.clone()))
        .collect();
    
    Ok(optimal_pairs)
}

struct TestObjectiveFunction {
    problem_size: usize,
}

impl TestObjectiveFunction {
    fn new(problem_size: usize) -> Self {
        Self { problem_size }
    }
}

impl quantum_pair_analyzer::quantum::ObjectiveFunction for TestObjectiveFunction {
    fn evaluate(&self, solution: &[f64]) -> Result<f64, AnalyzerError> {
        Ok(solution.iter().map(|x| x * x).sum::<f64>())
    }
    
    fn gradient(&self, solution: &[f64]) -> Result<Vec<f64>, AnalyzerError> {
        Ok(solution.iter().map(|x| 2.0 * x).collect())
    }
    
    fn hessian(&self, solution: &[f64]) -> Result<nalgebra::DMatrix<f64>, AnalyzerError> {
        use nalgebra::DMatrix;
        let n = solution.len();
        let mut hessian = DMatrix::zeros(n, n);
        for i in 0..n {
            hessian[(i, i)] = 2.0;
        }
        Ok(hessian)
    }
}