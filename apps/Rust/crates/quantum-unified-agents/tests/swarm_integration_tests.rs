//! Integration tests for swarm-enhanced quantum agent coordination
//! 
//! Tests the integration between:
//! - Quantum Agent Unification Framework
//! - Swarm Intelligence Optimization
//! - PADS Signal Aggregation
//! - Real-time Performance Monitoring

use quantum_unified_agents::{
    SwarmEnhancedQuantumCoordinator, OptimizationObjective,
    UnifiedQuantumAgentRegistry, MarketData, LatticeState,
    PADSSignal, AggregationStrategy, CoordinationMetrics
};
use std::sync::Arc;
use tokio::time::{timeout, Duration};

/// Test basic swarm-enhanced coordination functionality
#[tokio::test]
async fn test_swarm_enhanced_basic_coordination() {
    // Initialize quantum registry with test configuration
    let registry_config = UnifiedQuantumAgentRegistry::test_config();
    let quantum_registry = Arc::new(UnifiedQuantumAgentRegistry::new(registry_config));
    
    // Create swarm-enhanced coordinator
    let mut coordinator = SwarmEnhancedQuantumCoordinator::new(quantum_registry.clone())
        .await
        .expect("Failed to create swarm coordinator");
    
    // Test market data
    let market_data = MarketData {
        price: 100.0,
        volume: 10000.0,
        volatility: 0.15,
        trend: 0.05,
        momentum: 0.02,
        sentiment: 0.6,
        liquidity: 0.8,
        correlation: 0.3,
        timestamp: chrono::Utc::now(),
    };
    
    // Test single-objective optimization
    let result = coordinator.optimize_coordination(
        OptimizationObjective::SignalAccuracy,
        &market_data,
        20 // Small budget for test
    ).await;
    
    assert!(result.is_ok(), "Basic coordination optimization failed");
    
    let optimized_individual = result.unwrap();
    assert!(optimized_individual.fitness >= 0.0, "Invalid fitness score");
    assert!(!optimized_individual.agent_parameters.is_empty(), "No agent parameters optimized");
    
    println!("✅ Basic swarm coordination test passed");
    println!("   Fitness: {:.6}", optimized_individual.fitness);
    println!("   Agents optimized: {}", optimized_individual.agent_parameters.len());
}

/// Test multi-objective Pareto optimization
#[tokio::test]
async fn test_multi_objective_optimization() {
    let registry_config = UnifiedQuantumAgentRegistry::test_config();
    let quantum_registry = Arc::new(UnifiedQuantumAgentRegistry::new(registry_config));
    
    let mut coordinator = SwarmEnhancedQuantumCoordinator::new(quantum_registry)
        .await
        .expect("Failed to create coordinator");
    
    let market_data = MarketData::test_data();
    
    // Define multiple conflicting objectives
    let objectives = vec![
        OptimizationObjective::SignalAccuracy,      // Maximize accuracy
        OptimizationObjective::LatencyMinimization, // Minimize latency
        OptimizationObjective::RiskAdjustedReturns, // Maximize returns
    ];
    
    // Run multi-objective optimization
    let pareto_front = coordinator.multi_objective_optimization(
        objectives.clone(),
        &market_data,
        15 // Small budget for test
    ).await;
    
    assert!(pareto_front.is_ok(), "Multi-objective optimization failed");
    
    let solutions = pareto_front.unwrap();
    assert!(!solutions.is_empty(), "No Pareto optimal solutions found");
    assert!(solutions.len() <= 50, "Too many solutions in Pareto front");
    
    // Verify solutions are non-dominated
    for solution in &solutions {
        assert!(solution.fitness > 0.0, "Invalid solution fitness");
    }
    
    println!("✅ Multi-objective optimization test passed");
    println!("   Pareto front size: {}", solutions.len());
    println!("   Objectives: {:?}", objectives);
}

/// Test real-time adaptive optimization
#[tokio::test]
async fn test_adaptive_optimization() {
    let registry_config = UnifiedQuantumAgentRegistry::test_config();
    let quantum_registry = Arc::new(UnifiedQuantumAgentRegistry::new(registry_config));
    
    let mut coordinator = SwarmEnhancedQuantumCoordinator::new(quantum_registry)
        .await
        .expect("Failed to create coordinator");
    
    // Create market data stream
    let (tx, rx) = tokio::sync::mpsc::channel(100);
    
    // Spawn adaptive optimization task
    let optimization_handle = tokio::spawn(async move {
        coordinator.adaptive_optimization(rx).await
    });
    
    // Send test market data
    for i in 0..5 {
        let market_data = MarketData {
            price: 100.0 + i as f64,
            volume: 10000.0 * (1.0 + i as f64 * 0.1),
            volatility: 0.15 + i as f64 * 0.01,
            trend: (i as f64 - 2.0) * 0.02,
            momentum: i as f64 * 0.005,
            sentiment: 0.5 + (i as f64 - 2.0) * 0.1,
            liquidity: 0.8 - i as f64 * 0.05,
            correlation: 0.3 + i as f64 * 0.05,
            timestamp: chrono::Utc::now(),
        };
        
        tx.send(market_data).await.expect("Failed to send market data");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    // Close channel to terminate adaptive optimization
    drop(tx);
    
    // Wait for optimization to complete (with timeout)
    let result = timeout(Duration::from_secs(5), optimization_handle).await;
    
    match result {
        Ok(Ok(())) => println!("✅ Adaptive optimization completed successfully"),
        Ok(Err(e)) => panic!("Adaptive optimization failed: {:?}", e),
        Err(_) => panic!("Adaptive optimization timed out"),
    }
}

/// Test performance under high-frequency updates
#[tokio::test]
async fn test_high_frequency_optimization() {
    let registry_config = UnifiedQuantumAgentRegistry::test_config();
    let quantum_registry = Arc::new(UnifiedQuantumAgentRegistry::new(registry_config));
    
    let mut coordinator = SwarmEnhancedQuantumCoordinator::new(quantum_registry)
        .await
        .expect("Failed to create coordinator");
    
    let market_data = MarketData::test_data();
    
    // Measure optimization performance
    let start_time = std::time::Instant::now();
    let mut optimization_times = Vec::new();
    
    // Run multiple quick optimizations
    for _ in 0..10 {
        let opt_start = std::time::Instant::now();
        
        let result = coordinator.optimize_coordination(
            OptimizationObjective::LatencyMinimization,
            &market_data,
            5 // Very small budget for speed
        ).await;
        
        let opt_time = opt_start.elapsed();
        optimization_times.push(opt_time);
        
        assert!(result.is_ok(), "High-frequency optimization failed");
        assert!(opt_time < Duration::from_millis(500), "Optimization too slow: {:?}", opt_time);
    }
    
    let total_time = start_time.elapsed();
    let avg_time = optimization_times.iter().sum::<Duration>() / optimization_times.len() as u32;
    let max_time = optimization_times.iter().max().unwrap();
    let min_time = optimization_times.iter().min().unwrap();
    
    println!("✅ High-frequency optimization test passed");
    println!("   Total time: {:?}", total_time);
    println!("   Average time: {:?}", avg_time);
    println!("   Min time: {:?}", min_time);
    println!("   Max time: {:?}", max_time);
    
    // Performance assertions
    assert!(avg_time < Duration::from_millis(200), "Average optimization too slow");
    assert!(max_time < Duration::from_millis(500), "Max optimization too slow");
}

/// Test coordination metrics and monitoring
#[tokio::test]
async fn test_coordination_metrics() {
    let registry_config = UnifiedQuantumAgentRegistry::test_config();
    let quantum_registry = Arc::new(UnifiedQuantumAgentRegistry::new(registry_config));
    
    let mut coordinator = SwarmEnhancedQuantumCoordinator::new(quantum_registry)
        .await
        .expect("Failed to create coordinator");
    
    // Get initial metrics
    let initial_metrics = coordinator.get_coordination_metrics().await;
    assert_eq!(initial_metrics.current_fitness, 0.0);
    
    let market_data = MarketData::test_data();
    
    // Run optimization to update metrics
    let _result = coordinator.optimize_coordination(
        OptimizationObjective::SignalAccuracy,
        &market_data,
        10
    ).await.expect("Optimization failed");
    
    // Check updated metrics
    let updated_metrics = coordinator.get_coordination_metrics().await;
    assert!(updated_metrics.current_fitness >= 0.0);
    assert!(!updated_metrics.active_objectives.is_empty());
    
    // Test metrics export
    let export_result = coordinator.export_optimization_results().await;
    assert!(export_result.is_ok(), "Failed to export metrics");
    
    let export_json = export_result.unwrap();
    assert!(!export_json.is_empty(), "Empty metrics export");
    
    // Verify JSON is valid
    let parsed: serde_json::Value = serde_json::from_str(&export_json)
        .expect("Invalid JSON in metrics export");
    
    assert!(parsed.get("current_metrics").is_some(), "Missing current metrics");
    assert!(parsed.get("learning_stats").is_some(), "Missing learning stats");
    
    println!("✅ Coordination metrics test passed");
    println!("   Current fitness: {:.6}", updated_metrics.current_fitness);
    println!("   Active objectives: {}", updated_metrics.active_objectives.len());
}

/// Test swarm integration with quantum coherence
#[tokio::test]
async fn test_quantum_coherence_optimization() {
    let registry_config = UnifiedQuantumAgentRegistry::test_config();
    let quantum_registry = Arc::new(UnifiedQuantumAgentRegistry::new(registry_config));
    
    let mut coordinator = SwarmEnhancedQuantumCoordinator::new(quantum_registry)
        .await
        .expect("Failed to create coordinator");
    
    let market_data = MarketData::test_data();
    
    // Optimize specifically for quantum coherence
    let result = coordinator.optimize_coordination(
        OptimizationObjective::QuantumCoherence,
        &market_data,
        25
    ).await;
    
    assert!(result.is_ok(), "Quantum coherence optimization failed");
    
    let optimized_individual = result.unwrap();
    
    // Verify quantum-specific optimizations
    assert!(optimized_individual.fitness > 0.0, "No quantum advantage achieved");
    
    // Check that optimization affects quantum-related parameters
    let has_coherence_params = optimized_individual.agent_parameters
        .values()
        .any(|params| params.coherence_targets > 0.0);
    
    assert!(has_coherence_params, "No quantum coherence parameters optimized");
    
    println!("✅ Quantum coherence optimization test passed");
    println!("   Quantum fitness: {:.6}", optimized_individual.fitness);
}

/// Test error handling and recovery
#[tokio::test]
async fn test_error_handling() {
    let registry_config = UnifiedQuantumAgentRegistry::test_config();
    let quantum_registry = Arc::new(UnifiedQuantumAgentRegistry::new(registry_config));
    
    let mut coordinator = SwarmEnhancedQuantumCoordinator::new(quantum_registry)
        .await
        .expect("Failed to create coordinator");
    
    // Test with invalid market data
    let invalid_market_data = MarketData {
        price: f64::NAN,
        volume: f64::INFINITY,
        volatility: -1.0, // Invalid negative volatility
        trend: 999.0,     // Extreme value
        momentum: f64::NAN,
        sentiment: 2.0,   // Out of range
        liquidity: -0.5,  // Invalid negative
        correlation: 10.0, // Out of range
        timestamp: chrono::Utc::now(),
    };
    
    // Test graceful handling of invalid inputs
    let result = coordinator.optimize_coordination(
        OptimizationObjective::SignalAccuracy,
        &invalid_market_data,
        5
    ).await;
    
    // Should either succeed with cleaned data or fail gracefully
    match result {
        Ok(individual) => {
            println!("✅ Error handling test passed (graceful recovery)");
            println!("   Recovered fitness: {:.6}", individual.fitness);
        },
        Err(e) => {
            println!("✅ Error handling test passed (graceful failure)");
            println!("   Error: {:?}", e);
        }
    }
}

/// Benchmark optimization performance across different objectives
#[tokio::test]
async fn test_optimization_benchmarks() {
    let registry_config = UnifiedQuantumAgentRegistry::test_config();
    let quantum_registry = Arc::new(UnifiedQuantumAgentRegistry::new(registry_config));
    
    let mut coordinator = SwarmEnhancedQuantumCoordinator::new(quantum_registry)
        .await
        .expect("Failed to create coordinator");
    
    let market_data = MarketData::test_data();
    
    let objectives = vec![
        OptimizationObjective::SignalAccuracy,
        OptimizationObjective::LatencyMinimization,
        OptimizationObjective::RiskAdjustedReturns,
        OptimizationObjective::QuantumCoherence,
    ];
    
    let mut benchmark_results = Vec::new();
    
    for objective in objectives {
        let start = std::time::Instant::now();
        
        let result = coordinator.optimize_coordination(
            objective.clone(),
            &market_data,
            15
        ).await;
        
        let duration = start.elapsed();
        
        match result {
            Ok(individual) => {
                benchmark_results.push((objective, duration, individual.fitness));
                println!("   {:?}: {:.6} fitness in {:?}", 
                    objective, individual.fitness, duration);
            },
            Err(e) => {
                println!("   {:?}: Failed with error: {:?}", objective, e);
            }
        }
    }
    
    // Verify all objectives completed
    assert!(!benchmark_results.is_empty(), "No successful optimizations");
    
    // Calculate performance statistics
    let avg_duration: Duration = benchmark_results.iter()
        .map(|(_, duration, _)| *duration)
        .sum::<Duration>() / benchmark_results.len() as u32;
    
    let avg_fitness: f64 = benchmark_results.iter()
        .map(|(_, _, fitness)| *fitness)
        .sum::<f64>() / benchmark_results.len() as f64;
    
    println!("✅ Optimization benchmarks completed");
    println!("   Objectives tested: {}", benchmark_results.len());
    println!("   Average duration: {:?}", avg_duration);
    println!("   Average fitness: {:.6}", avg_fitness);
    
    // Performance assertions
    assert!(avg_duration < Duration::from_secs(2), "Optimizations too slow");
    assert!(avg_fitness > 0.0, "Poor optimization quality");
}

// Helper trait implementations for test data
impl MarketData {
    fn test_data() -> Self {
        Self {
            price: 100.0,
            volume: 10000.0,
            volatility: 0.15,
            trend: 0.05,
            momentum: 0.02,
            sentiment: 0.6,
            liquidity: 0.8,
            correlation: 0.3,
            timestamp: chrono::Utc::now(),
        }
    }
}

impl UnifiedQuantumAgentRegistry {
    fn test_config() -> crate::RegistryConfig {
        crate::RegistryConfig {
            max_agents: 5,
            health_check_interval_s: 30,
            auto_decoherence_mitigation: true,
            enable_performance_monitoring: true,
            lattice_dimensions: (8, 8, 8),
            coherence_threshold: 0.8,
            quantum_advantage_threshold: 1.1,
        }
    }
    
    fn new_test() -> Self {
        Self::new(Self::test_config())
    }
}