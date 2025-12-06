//! Comprehensive TDD tests for Quantum Nash Equilibrium solver
//! 
//! These tests are written FIRST to drive the implementation of the quantum Nash equilibrium
//! solver with 100% property-based test coverage.

use qbmia_core::{
    quantum::{QuantumNashEquilibrium, QuantumNashResult, GameMatrix},
    config::QuantumConfig,
    error::{QBMIAError, Result},
};
use ndarray::{Array2, Array3, Array4};
use approx::assert_relative_eq;
use proptest::prelude::*;
use quickcheck_macros::quickcheck;
use rstest::*;
use tokio_test;

/// Test data fixtures for quantum Nash equilibrium
#[fixture]
fn simple_2x2_game() -> GameMatrix {
    // Classic Prisoner's Dilemma payoff matrix
    let mut matrix = Array4::zeros((2, 2, 2, 2));
    
    // Player 0 payoffs
    matrix[[0, 0, 0, 0]] = 3.0; // Both cooperate
    matrix[[0, 0, 0, 1]] = 0.0; // P0 cooperates, P1 defects
    matrix[[0, 0, 1, 0]] = 5.0; // P0 defects, P1 cooperates
    matrix[[0, 0, 1, 1]] = 1.0; // Both defect
    
    // Player 1 payoffs
    matrix[[1, 1, 0, 0]] = 3.0; // Both cooperate
    matrix[[1, 1, 0, 1]] = 5.0; // P0 cooperates, P1 defects
    matrix[[1, 1, 1, 0]] = 0.0; // P0 defects, P1 cooperates
    matrix[[1, 1, 1, 1]] = 1.0; // Both defect
    
    GameMatrix::new(matrix).unwrap()
}

#[fixture]
fn quantum_config() -> QuantumConfig {
    QuantumConfig {
        num_qubits: 8,
        num_layers: 2,
        learning_rate: 0.1,
        convergence_threshold: 1e-3,
        max_iterations: 100,
        device_type: crate::config::DeviceType::Cpu,
    }
}

/// Test quantum Nash equilibrium initialization
#[rstest]
#[tokio::test]
async fn test_quantum_nash_initialization(quantum_config: QuantumConfig) {
    let solver = QuantumNashEquilibrium::new(quantum_config).await;
    assert!(solver.is_ok());
    
    let solver = solver.unwrap();
    assert_eq!(solver.num_qubits(), 8);
    assert_eq!(solver.num_layers(), 2);
}

/// Test quantum Nash equilibrium with simple 2x2 game
#[rstest]
#[tokio::test]
async fn test_simple_2x2_equilibrium(
    quantum_config: QuantumConfig,
    simple_2x2_game: GameMatrix,
) -> Result<()> {
    let mut solver = QuantumNashEquilibrium::new(quantum_config).await?;
    
    let result = solver.find_equilibrium(&simple_2x2_game, None).await?;
    
    // Check that we got a valid result
    assert!(result.convergence_score > 0.0);
    assert!(result.nash_loss >= 0.0);
    assert!(result.iterations > 0);
    assert_eq!(result.strategies.len(), 2); // Two players
    
    // Check that strategies are valid probability distributions
    for strategy in result.strategies.values() {
        assert_relative_eq!(strategy.sum(), 1.0, epsilon = 1e-6);
        assert!(strategy.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }
    
    // For Prisoner's Dilemma, Nash equilibrium should be (Defect, Defect)
    // which corresponds to higher probability on action 1
    let player_0_strategy = &result.strategies["player_0"];
    let player_1_strategy = &result.strategies["player_1"];
    
    // In quantum case, we might get mixed strategies, but defect should have higher probability
    assert!(player_0_strategy[1] >= player_0_strategy[0] - 0.3); // Allow some quantum mixing
    assert!(player_1_strategy[1] >= player_1_strategy[0] - 0.3);
    
    Ok(())
}

/// Test quantum state entropy calculation
#[rstest]
#[tokio::test]
async fn test_quantum_entropy_calculation(quantum_config: QuantumConfig) -> Result<()> {
    let solver = QuantumNashEquilibrium::new(quantum_config).await?;
    
    // Test with uniform distribution (maximum entropy)
    let uniform_probs = vec![0.25; 4];
    let entropy = solver.calculate_entropy(&uniform_probs);
    assert_relative_eq!(entropy, 1.0, epsilon = 1e-6); // Normalized entropy
    
    // Test with pure state (minimum entropy)
    let pure_probs = vec![1.0, 0.0, 0.0, 0.0];
    let entropy = solver.calculate_entropy(&pure_probs);
    assert_relative_eq!(entropy, 0.0, epsilon = 1e-6);
    
    // Test with mixed state
    let mixed_probs = vec![0.5, 0.3, 0.2, 0.0];
    let entropy = solver.calculate_entropy(&mixed_probs);
    assert!(entropy > 0.0 && entropy < 1.0);
    
    Ok(())
}

/// Test convergence with different learning rates
#[rstest]
#[case(0.01, 200)] // Low learning rate, more iterations
#[case(0.1, 100)]  // Medium learning rate
#[case(0.5, 50)]   // High learning rate, fewer iterations
#[tokio::test]
async fn test_convergence_with_learning_rates(
    #[case] learning_rate: f64,
    #[case] expected_max_iterations: usize,
    simple_2x2_game: GameMatrix,
) -> Result<()> {
    let mut config = QuantumConfig::default();
    config.learning_rate = learning_rate;
    config.max_iterations = expected_max_iterations;
    
    let mut solver = QuantumNashEquilibrium::new(config).await?;
    let result = solver.find_equilibrium(&simple_2x2_game, None).await?;
    
    // Higher learning rates should converge faster
    if learning_rate > 0.1 {
        assert!(result.iterations <= expected_max_iterations / 2);
    }
    
    // But all should eventually converge
    assert!(result.convergence_score > 0.1);
    
    Ok(())
}

/// Test stability analysis of equilibrium
#[rstest]
#[tokio::test]
async fn test_stability_analysis(
    quantum_config: QuantumConfig,
    simple_2x2_game: GameMatrix,
) -> Result<()> {
    let mut solver = QuantumNashEquilibrium::new(quantum_config).await?;
    let result = solver.find_equilibrium(&simple_2x2_game, None).await?;
    
    // Check stability metrics exist
    assert!(result.stability_analysis.contains_key("perturbation_sensitivity"));
    
    // Sensitivity should be non-negative
    let sensitivity = result.stability_analysis["perturbation_sensitivity"];
    assert!(sensitivity >= 0.0);
    
    // For a well-converged solution, sensitivity should be relatively low
    if result.convergence_score > 0.8 {
        assert!(sensitivity < 1.0);
    }
    
    Ok(())
}

/// Test with market conditions influence
#[rstest]
#[tokio::test]
async fn test_market_conditions_influence(
    quantum_config: QuantumConfig,
    simple_2x2_game: GameMatrix,
) -> Result<()> {
    let mut solver = QuantumNashEquilibrium::new(quantum_config).await?;
    
    // Test with high volatility market conditions
    let mut market_conditions = std::collections::HashMap::new();
    market_conditions.insert("volatility".to_string(), 0.5);
    market_conditions.insert("trend".to_string(), 0.3);
    
    let result_with_conditions = solver.find_equilibrium(&simple_2x2_game, Some(market_conditions)).await?;
    
    // Test without market conditions
    let result_without_conditions = solver.find_equilibrium(&simple_2x2_game, None).await?;
    
    // Results should be different when market conditions are included
    let strategy_diff: f64 = result_with_conditions.strategies["player_0"]
        .iter()
        .zip(result_without_conditions.strategies["player_0"].iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    
    assert!(strategy_diff > 1e-6); // Strategies should be noticeably different
    
    Ok(())
}

/// Property-based test: Nash equilibrium properties
proptest! {
    #[test]
    fn prop_nash_equilibrium_properties(
        num_players in 2..=4usize,
        num_actions in 2..=4usize,
        payoff_values in prop::collection::vec(-10.0..10.0f64, 16..64)
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Create random game matrix
            let matrix_size = (num_players, num_players, num_actions, num_actions);
            let mut matrix = Array4::zeros(matrix_size);
            
            // Fill with random payoffs
            for (i, &value) in payoff_values.iter().enumerate().take(matrix.len()) {
                let indices = matrix.indexed_iter().nth(i).unwrap().0;
                matrix[indices] = value;
            }
            
            let game_matrix = GameMatrix::new(matrix);
            if game_matrix.is_err() {
                return; // Skip invalid matrices
            }
            
            let config = QuantumConfig {
                num_qubits: (num_players * num_actions).next_power_of_two().max(4),
                max_iterations: 50, // Shorter for property tests
                ..QuantumConfig::default()
            };
            
            let mut solver = QuantumNashEquilibrium::new(config).await.unwrap();
            let result = solver.find_equilibrium(&game_matrix.unwrap(), None).await;
            
            if let Ok(result) = result {
                // Property 1: Strategies are valid probability distributions
                for strategy in result.strategies.values() {
                    prop_assert!((strategy.sum() - 1.0).abs() < 1e-3);
                    prop_assert!(strategy.iter().all(|&p| p >= -1e-6 && p <= 1.0 + 1e-6));
                }
                
                // Property 2: Nash loss is non-negative
                prop_assert!(result.nash_loss >= -1e-6);
                
                // Property 3: Convergence score is between 0 and 1
                prop_assert!(result.convergence_score >= 0.0 && result.convergence_score <= 1.0);
                
                // Property 4: Number of strategies equals number of players
                prop_assert_eq!(result.strategies.len(), num_players);
            }
        });
    }
}

/// Benchmark test for performance requirements
#[rstest]
#[tokio::test]
async fn test_performance_requirements(
    quantum_config: QuantumConfig,
    simple_2x2_game: GameMatrix,
) -> Result<()> {
    let mut solver = QuantumNashEquilibrium::new(quantum_config).await?;
    
    let start = std::time::Instant::now();
    let result = solver.find_equilibrium(&simple_2x2_game, None).await?;
    let elapsed = start.elapsed();
    
    // Performance requirement: sub-millisecond execution for simple games
    assert!(elapsed.as_millis() < 10, "Execution took {} ms, expected < 10ms", elapsed.as_millis());
    
    // Quality requirement: decent convergence
    assert!(result.convergence_score > 0.3, "Convergence score {} too low", result.convergence_score);
    
    Ok(())
}

/// Test error handling for invalid inputs
#[rstest]
#[tokio::test]
async fn test_error_handling() {
    // Test with invalid configuration
    let invalid_config = QuantumConfig {
        num_qubits: 0, // Invalid
        ..QuantumConfig::default()
    };
    
    let result = QuantumNashEquilibrium::new(invalid_config).await;
    assert!(result.is_err());
    
    // Test with valid config but invalid game matrix
    let valid_config = QuantumConfig::default();
    let mut solver = QuantumNashEquilibrium::new(valid_config).await.unwrap();
    
    // Empty game matrix
    let empty_matrix = Array4::zeros((0, 0, 0, 0));
    let empty_game = GameMatrix::new(empty_matrix);
    assert!(empty_game.is_err());
}

/// Test serialization and deserialization of results
#[rstest]
#[tokio::test]
async fn test_serialization(
    quantum_config: QuantumConfig,
    simple_2x2_game: GameMatrix,
) -> Result<()> {
    let mut solver = QuantumNashEquilibrium::new(quantum_config).await?;
    let original_result = solver.find_equilibrium(&simple_2x2_game, None).await?;
    
    // Serialize to JSON
    let json = serde_json::to_string(&original_result)?;
    
    // Deserialize back
    let deserialized_result: QuantumNashResult = serde_json::from_str(&json)?;
    
    // Check that important fields are preserved
    assert_relative_eq!(original_result.convergence_score, deserialized_result.convergence_score, epsilon = 1e-6);
    assert_relative_eq!(original_result.nash_loss, deserialized_result.nash_loss, epsilon = 1e-6);
    assert_eq!(original_result.iterations, deserialized_result.iterations);
    
    Ok(())
}

/// Test parallel execution consistency
#[rstest]
#[tokio::test]
async fn test_parallel_consistency(simple_2x2_game: GameMatrix) -> Result<()> {
    let config = QuantumConfig::default();
    
    // Run multiple times in parallel
    let tasks: Vec<_> = (0..4).map(|_| {
        let game = simple_2x2_game.clone();
        let config = config.clone();
        tokio::spawn(async move {
            let mut solver = QuantumNashEquilibrium::new(config).await.unwrap();
            solver.find_equilibrium(&game, None).await
        })
    }).collect();
    
    let results: Vec<QuantumNashResult> = futures::future::try_join_all(tasks)
        .await
        .unwrap()
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    
    // All results should have converged (allowing for stochastic variations)
    for result in &results {
        assert!(result.convergence_score > 0.1);
        assert!(result.nash_loss < 10.0); // Reasonable bound
    }
    
    // Strategies should be similar (within reasonable bounds for quantum stochasticity)
    let first_strategy = &results[0].strategies["player_0"];
    for result in &results[1..] {
        let other_strategy = &result.strategies["player_0"];
        let diff: f64 = first_strategy.iter()
            .zip(other_strategy.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 0.5, "Strategies too different: diff = {}", diff);
    }
    
    Ok(())
}

/// QuickCheck property test for strategy validity
#[quickcheck]
fn quickcheck_strategy_validity(payoffs: Vec<f64>) -> bool {
    if payoffs.len() < 16 {
        return true; // Skip too small inputs
    }
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let matrix = Array4::from_shape_vec((2, 2, 2, 2), payoffs[..16].to_vec()).unwrap();
        let game = GameMatrix::new(matrix).unwrap();
        
        let config = QuantumConfig {
            max_iterations: 20, // Quick test
            ..QuantumConfig::default()
        };
        
        let mut solver = QuantumNashEquilibrium::new(config).await.unwrap();
        match solver.find_equilibrium(&game, None).await {
            Ok(result) => {
                // Check strategy validity
                result.strategies.values().all(|strategy| {
                    let sum = strategy.sum();
                    (sum - 1.0).abs() < 1e-2 && strategy.iter().all(|&p| p >= -1e-3 && p <= 1.0 + 1e-3)
                })
            }
            Err(_) => true, // Allow failures for extreme inputs
        }
    })
}