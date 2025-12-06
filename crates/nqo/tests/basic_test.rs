use nqo::{NeuromorphicQuantumOptimizer, NqoConfig};

#[tokio::test]
async fn test_nqo_creation() {
    let config = NqoConfig::default();
    let optimizer = NeuromorphicQuantumOptimizer::new(config).await;
    assert!(optimizer.is_ok());
}

#[tokio::test]
async fn test_parameter_optimization() {
    let config = NqoConfig {
        neurons: 32,
        qubits: 4,
        learning_rate: 0.01,
        adaptivity: 0.7,
        cache_size: 100,
        use_gpu: false,
        quantum_shots: None,
        max_history: 50,
        log_level: "INFO".to_string(),
    };
    
    let optimizer = NeuromorphicQuantumOptimizer::new(config).await.unwrap();
    
    // Simple quadratic objective function
    let objective = |params: &[f64]| -> f64 {
        params.iter().map(|&x| (x - 1.0).powi(2)).sum()
    };
    
    let initial_params = vec![0.0, 0.5, 2.0];
    let result = optimizer.optimize_parameters(objective, initial_params, 5).await;
    
    assert!(result.is_ok());
    let result = result.unwrap();
    
    // Should improve from initial value
    assert!(result.value < result.initial_value);
    assert_eq!(result.iterations, 5);
    assert!(result.confidence > 0.0);
}

#[tokio::test]
async fn test_trading_optimization() {
    let config = NqoConfig::default();
    let optimizer = NeuromorphicQuantumOptimizer::new(config).await.unwrap();
    
    let mut matches = std::collections::HashMap::new();
    matches.insert("BTC/USD".to_string(), 0.85);
    matches.insert("ETH/USD".to_string(), 0.72);
    
    let result = optimizer.optimize_trading_parameters(matches, None).await;
    
    assert!(result.is_ok());
    let params = result.unwrap();
    
    assert!(params.entry_threshold > 0.0 && params.entry_threshold < 1.0);
    assert!(params.stop_loss > 0.0);
    assert!(params.take_profit > params.stop_loss);
    assert!(params.confidence >= 0.0 && params.confidence <= 1.0);
}

#[tokio::test]
async fn test_performance_metrics() {
    let config = NqoConfig::default();
    let optimizer = NeuromorphicQuantumOptimizer::new(config).await.unwrap();
    
    let metrics = optimizer.get_performance_metrics();
    assert_eq!(metrics.sample_size, 0);
    
    let stats = optimizer.get_execution_stats();
    assert_eq!(stats.count, 0);
}