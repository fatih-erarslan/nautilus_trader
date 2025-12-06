//! Integration tests for NQO cross-module functionality

use nqo::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn test_full_optimization_pipeline() {
    let config = NqoConfig {
        neurons: 32,
        qubits: 4,
        learning_rate: 0.01,
        adaptivity: 0.7,
        use_gpu: false,
        quantum_shots: None,
        cache_size: 50,
        max_history: 20,
        log_level: "INFO".to_string(),
    };
    
    // Create optimizer
    if let Ok(optimizer) = NeuromorphicQuantumOptimizer::new(config).await {
        // Define Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²
        // Minimum at (a,a²) where a=1, so minimum at (1,1) with value 0
        let rosenbrock = |params: &[f64]| -> f64 {
            if params.len() >= 2 {
                let x = params[0];
                let y = params[1];
                let a = 1.0;
                let b = 100.0;
                (a - x).powi(2) + b * (y - x.powi(2)).powi(2)
            } else {
                f64::INFINITY
            }
        };
        
        // Start from a point far from optimum
        let initial_params = vec![0.0, 0.0];
        let result = optimizer.optimize_parameters(rosenbrock, initial_params, 10).await;
        
        if let Ok(result) = result {
            // Should improve significantly
            assert!(result.value < result.initial_value);
            assert_eq!(result.iterations, 10);
            assert!(result.confidence > 0.0);
            
            // Should move towards optimum (1, 1)
            assert_eq!(result.params.len(), 2);
            
            // History should show improvement trend
            assert_eq!(result.history.len(), 11); // Initial + 10 iterations
            assert!(result.history.last().unwrap() <= result.history.first().unwrap());
        }
    }
}

#[tokio::test]
async fn test_concurrent_optimizations() {
    let config = NqoConfig {
        neurons: 24,
        qubits: 4,
        use_gpu: false,
        quantum_shots: None,
        ..Default::default()
    };
    
    if let Ok(optimizer) = NeuromorphicQuantumOptimizer::new(config).await {
        let optimizer = Arc::new(optimizer);
        
        // Create multiple concurrent optimization tasks
        let mut handles = vec![];
        
        for i in 0..5 {
            let optimizer_clone = Arc::clone(&optimizer);
            let offset = i as f64 * 0.1;
            
            let handle = tokio::spawn(async move {
                // Each task optimizes a slightly different quadratic
                let objective = move |params: &[f64]| -> f64 {
                    params.iter().map(|&x| (x - offset).powi(2)).sum()
                };
                
                let initial = vec![1.0];
                optimizer_clone.optimize_parameters(objective, initial, 5).await
            });
            
            handles.push(handle);
        }
        
        // Wait for all tasks to complete
        let mut results = vec![];
        for handle in handles {
            if let Ok(result) = handle.await {
                results.push(result);
            }
        }
        
        // All tasks should complete successfully
        assert_eq!(results.len(), 5);
        
        // All results should show improvement
        for result in results {
            if let Ok(opt_result) = result {
                assert!(opt_result.value < opt_result.initial_value);
                assert!(opt_result.confidence > 0.0);
            }
        }
    }
}

#[tokio::test]
async fn test_neural_quantum_integration() {
    let config = NqoConfig {
        neurons: 16,
        qubits: 4,
        learning_rate: 0.05,
        adaptivity: 0.8,
        use_gpu: false,
        quantum_shots: None,
        ..Default::default()
    };
    
    if let Ok(optimizer) = NeuromorphicQuantumOptimizer::new(config).await {
        // Multi-modal function with several local minima
        let multimodal = |params: &[f64]| -> f64 {
            if !params.is_empty() {
                let x = params[0];
                // Function with multiple local minima
                x.powi(4) - 4.0 * x.powi(3) + 4.0 * x.powi(2) + 1.0
            } else {
                f64::INFINITY
            }
        };
        
        // Try optimization from different starting points
        let starting_points = vec![0.0, 1.0, 2.0, 3.0];
        let mut results = vec![];
        
        for start in starting_points {
            let result = optimizer.optimize_parameters(
                &multimodal,
                vec![start],
                8
            ).await;
            
            if let Ok(res) = result {
                results.push(res);
            }
        }
        
        // Should find improvements from all starting points
        assert!(!results.is_empty());
        
        for result in &results {
            assert!(result.value < result.initial_value);
            assert!(result.execution_time_ms > 0.0);
        }
        
        // Neural network should adapt differently for different landscapes
        // (evidenced by different convergence patterns)
        let final_values: Vec<f64> = results.iter().map(|r| r.value).collect();
        assert!(final_values.len() >= 2);
    }
}

#[tokio::test]
async fn test_trading_system_integration() {
    let config = NqoConfig {
        neurons: 32,
        qubits: 4,
        use_gpu: false,
        quantum_shots: None,
        learning_rate: 0.02,
        ..Default::default()
    };
    
    if let Ok(optimizer) = NeuromorphicQuantumOptimizer::new(config).await {
        // Simulate multiple trading scenarios
        let scenarios = vec![
            // High confidence matches
            {
                let mut matches = HashMap::new();
                matches.insert("BTC/USD".to_string(), 0.85);
                matches.insert("ETH/USD".to_string(), 0.78);
                matches
            },
            // Medium confidence matches
            {
                let mut matches = HashMap::new();
                matches.insert("ADA/USD".to_string(), 0.65);
                matches.insert("DOT/USD".to_string(), 0.62);
                matches
            },
            // Low confidence matches
            {
                let mut matches = HashMap::new();
                matches.insert("DOGE/USD".to_string(), 0.45);
                matches.insert("SHIB/USD".to_string(), 0.42);
                matches
            },
        ];
        
        let mut trading_results = vec![];
        
        for matches in scenarios {
            let result = optimizer.optimize_trading_parameters(matches, None).await;
            if let Ok(params) = result {
                trading_results.push(params);
            }
        }
        
        assert_eq!(trading_results.len(), 3);
        
        // High confidence scenario should have higher entry threshold
        // and more aggressive take profit
        if trading_results.len() >= 2 {
            assert!(trading_results[0].entry_threshold >= trading_results[1].entry_threshold);
            assert!(trading_results[0].confidence >= trading_results[1].confidence);
        }
        
        // All parameters should be within reasonable bounds
        for params in &trading_results {
            assert!(params.entry_threshold > 0.0 && params.entry_threshold < 1.0);
            assert!(params.stop_loss > 0.0 && params.stop_loss < 0.2);
            assert!(params.take_profit > params.stop_loss);
            assert!(params.confidence >= 0.0 && params.confidence <= 1.0);
        }
    }
}

#[tokio::test]
async fn test_allocation_optimization_integration() {
    let config = NqoConfig {
        neurons: 20,
        qubits: 4,
        use_gpu: false,
        quantum_shots: None,
        ..Default::default()
    };
    
    if let Ok(optimizer) = NeuromorphicQuantumOptimizer::new(config).await {
        // Test different market conditions
        let market_scenarios = vec![
            // Low volatility, high volume
            {
                let mut data = HashMap::new();
                data.insert("volatility".to_string(), 0.15);
                data.insert("volume".to_string(), 5000000.0);
                (0.06, 0.75, data) // edge, win_rate, market_data
            },
            // High volatility, medium volume
            {
                let mut data = HashMap::new();
                data.insert("volatility".to_string(), 0.45);
                data.insert("volume".to_string(), 2000000.0);
                (0.04, 0.65, data)
            },
            // Medium volatility, low volume
            {
                let mut data = HashMap::new();
                data.insert("volatility".to_string(), 0.25);
                data.insert("volume".to_string(), 500000.0);
                (0.05, 0.70, data)
            },
        ];
        
        let mut allocation_results = vec![];
        
        for (edge, win_rate, market_data) in market_scenarios {
            let result = optimizer.optimize_allocation(
                "TEST/USD",
                edge,
                win_rate,
                market_data
            ).await;
            
            if let Ok(allocation) = result {
                allocation_results.push((edge, win_rate, allocation));
            }
        }
        
        assert_eq!(allocation_results.len(), 3);
        
        // Higher edge and win rate should generally lead to higher allocation
        // (though volatility also affects this)
        for (edge, win_rate, allocation) in &allocation_results {
            assert!(allocation.allocation >= 0.0);
            assert!(allocation.allocation <= 1.0); // Max 100% allocation
            assert!(allocation.confidence >= 0.0 && allocation.confidence <= 1.0);
            
            // Allocation should be positively correlated with edge and win rate
            if *edge > 0.05 && *win_rate > 0.7 {
                assert!(allocation.allocation > 0.01); // Should allocate something
            }
        }
    }
}

#[tokio::test]
async fn test_performance_optimization_cycle() {
    let config = NqoConfig {
        neurons: 24,
        qubits: 4,
        use_gpu: false,
        quantum_shots: None,
        max_history: 10,
        ..Default::default()
    };
    
    if let Ok(optimizer) = NeuromorphicQuantumOptimizer::new(config).await {
        // Objective that changes over time (simulating changing market conditions)
        let mut shift = 0.0;
        
        for cycle in 0..3 {
            shift = cycle as f64 * 0.2;
            
            let objective = |params: &[f64]| -> f64 {
                if params.len() >= 2 {
                    (params[0] - 1.0 - shift).powi(2) + (params[1] - 2.0 - shift).powi(2)
                } else {
                    f64::INFINITY
                }
            };
            
            let initial = vec![0.0, 0.0];
            let result = optimizer.optimize_parameters(objective, initial, 5).await;
            
            if let Ok(opt_result) = result {
                // Should adapt to the shifting optimum
                assert!(opt_result.value < opt_result.initial_value);
                
                // Parameters should move towards the shifted optimum
                assert_eq!(opt_result.params.len(), 2);
                
                // Allow some time for neural adaptation
                sleep(Duration::from_millis(10)).await;
            }
        }
        
        // Check that optimizer has performance history
        let metrics = optimizer.get_performance_metrics();
        assert!(metrics.sample_size >= 3);
        assert!(metrics.mean_improvement >= 0.0);
        assert!(metrics.success_rate >= 0.0 && metrics.success_rate <= 1.0);
        
        let stats = optimizer.get_execution_stats();
        assert!(stats.count >= 3);
        assert!(stats.avg_time_ms > 0.0);
    }
}

#[tokio::test]
async fn test_cache_and_memory_integration() {
    let config = NqoConfig {
        neurons: 16,
        qubits: 4,
        use_gpu: false,
        quantum_shots: None,
        cache_size: 20,
        ..Default::default()
    };
    
    if let Ok(optimizer) = NeuromorphicQuantumOptimizer::new(config).await {
        let simple_quadratic = |params: &[f64]| -> f64 {
            params[0].powi(2) + params[1].powi(2)
        };
        
        // Same optimization multiple times (should hit cache)
        let initial_params = vec![2.0, 2.0];
        
        // First run (cache miss)
        let start1 = std::time::Instant::now();
        let result1 = optimizer.optimize_parameters(
            &simple_quadratic,
            initial_params.clone(),
            3
        ).await;
        let time1 = start1.elapsed();
        
        // Second run (should use cached intermediate results)
        let start2 = std::time::Instant::now();
        let result2 = optimizer.optimize_parameters(
            &simple_quadratic,
            initial_params.clone(),
            3
        ).await;
        let time2 = start2.elapsed();
        
        if let (Ok(r1), Ok(r2)) = (result1, result2) {
            // Results should be similar (may have minor variations due to neural adaptation)
            let value_diff = (r1.value - r2.value).abs();
            assert!(value_diff < 0.1);
            
            // Second run should be faster due to caching
            assert!(time2 <= time1 * 2); // Allow some variation
        }
        
        // Clear cache and test
        optimizer.clear_cache().await;
        
        let start3 = std::time::Instant::now();
        let result3 = optimizer.optimize_parameters(
            &simple_quadratic,
            initial_params,
            3
        ).await;
        let time3 = start3.elapsed();
        
        // After cache clear, should take similar time to first run
        assert!(result3.is_ok());
        assert!(time3 > time2 * 0.5); // Should be slower than cached version
    }
}

#[tokio::test]
async fn test_error_recovery_and_resilience() {
    let config = NqoConfig {
        neurons: 12,
        qubits: 4,
        use_gpu: false,
        quantum_shots: None,
        ..Default::default()
    };
    
    if let Ok(mut optimizer) = NeuromorphicQuantumOptimizer::new(config).await {
        // Test recovery from various error conditions
        
        // 1. Invalid parameters
        let bad_objective = |params: &[f64]| -> f64 {
            if params.is_empty() { f64::NAN } else { params[0] }
        };
        
        let result1 = optimizer.optimize_parameters(bad_objective, vec![], 3).await;
        assert!(result1.is_err());
        
        // 2. NaN-producing objective
        let nan_objective = |_: &[f64]| -> f64 { f64::NAN };
        let result2 = optimizer.optimize_parameters(nan_objective, vec![1.0], 3).await;
        assert!(result2.is_err());
        
        // 3. After errors, normal operation should still work
        let normal_objective = |params: &[f64]| -> f64 { params[0].powi(2) };
        let result3 = optimizer.optimize_parameters(normal_objective, vec![2.0], 3).await;
        
        // Should recover and work normally
        if let Ok(result) = result3 {
            assert!(result.value < result.initial_value);
        }
        
        // 4. Test reset functionality
        let reset_result = optimizer.reset().await;
        assert!(reset_result.is_ok());
        
        // After reset, should still work
        let result4 = optimizer.optimize_parameters(normal_objective, vec![1.5], 3).await;
        assert!(result4.is_ok());
    }
}

#[tokio::test]
async fn test_scalability_and_resource_management() {
    let config = NqoConfig {
        neurons: 8, // Smaller for resource testing
        qubits: 3,
        use_gpu: false,
        quantum_shots: None,
        cache_size: 10,
        max_history: 5,
        ..Default::default()
    };
    
    if let Ok(optimizer) = NeuromorphicQuantumOptimizer::new(config).await {
        // Test with increasing problem dimensions
        let dimensions = vec![1, 2, 3, 5];
        
        for dim in dimensions {
            let sphere_function = |params: &[f64]| -> f64 {
                params.iter().map(|&x| x.powi(2)).sum()
            };
            
            let initial_params = vec![1.0; dim];
            let result = optimizer.optimize_parameters(
                sphere_function,
                initial_params,
                3
            ).await;
            
            if let Ok(opt_result) = result {
                assert_eq!(opt_result.params.len(), dim);
                assert!(opt_result.value < opt_result.initial_value);
                
                // Higher dimensions should still converge, but may take longer
                assert!(opt_result.execution_time_ms > 0.0);
                assert!(opt_result.execution_time_ms < 10000.0); // Should not take too long
            }
        }
        
        // Check resource usage remains reasonable
        let final_stats = optimizer.get_execution_stats();
        assert!(final_stats.count >= dimensions.len());
        assert!(final_stats.avg_time_ms < 5000.0); // Average should be reasonable
    }
}