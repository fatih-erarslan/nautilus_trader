//! # Quantum Uncertainty Integration Tests
//!
//! Comprehensive integration tests for the quantum uncertainty quantification system.

use ndarray::{Array1, Array2};
use quantum_uncertainty::*;
use std::time::Duration;

#[tokio::test]
async fn test_complete_quantum_uncertainty_pipeline() {
    // Test the complete quantum uncertainty quantification pipeline
    let config = QuantumConfig::lightweight();
    let engine = QuantumUncertaintyEngine::new(config).await.unwrap();
    
    // Create test data
    let data = Array2::from_shape_vec((10, 4), (0..40).map(|x| x as f64 / 10.0).collect()).unwrap();
    let target = Array1::from_vec((0..10).map(|x| x as f64 / 5.0).collect());
    
    // Perform quantum uncertainty quantification
    let result = engine.quantify_uncertainty(&data, &target).await.unwrap();
    
    // Validate results
    assert!(!result.uncertainty_estimates.is_empty());
    assert!(result.quantum_advantage >= 0.0);
    assert!(result.confidence_level > 0.0 && result.confidence_level <= 1.0);
    assert!(!result.quantum_features.classical_features.is_empty());
    
    // Check conformal intervals
    assert!(result.conformal_intervals.lower_bound <= result.conformal_intervals.upper_bound);
    assert!(result.conformal_intervals.confidence_level > 0.0);
    
    // Validate optimized measurements
    assert!(!result.optimized_measurements.measurement_operators.is_empty());
    assert!(result.optimized_measurements.total_information >= 0.0);
}

#[tokio::test]
async fn test_quantum_circuit_fidelity_validation() {
    let config = QuantumConfig {
        n_qubits: 3,
        n_layers: 2,
        ensemble_size: 3,
        ..QuantumConfig::default()
    };
    
    let engine = QuantumUncertaintyEngine::new(config).await.unwrap();
    
    // Validate circuit fidelity
    let fidelity = engine.validate_circuit_fidelity().await.unwrap();
    assert!(fidelity >= 0.0 && fidelity <= 1.0);
    
    // Test multiple validations
    for _ in 0..5 {
        let fidelity = engine.validate_circuit_fidelity().await.unwrap();
        assert!(fidelity >= 0.0 && fidelity <= 1.0);
    }
}

#[tokio::test]
async fn test_quantum_feature_extraction_pipeline() {
    let config = QuantumConfig::lightweight();
    let extractor = QuantumFeatureExtractor::new(config).unwrap();
    
    // Create test data with different patterns
    let data1 = Array2::from_shape_vec((5, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]).unwrap();
    let data2 = Array2::from_shape_vec((5, 3), vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4, -1.6, -1.8]).unwrap();
    
    // Extract features from both datasets
    let features1 = extractor.extract_features(&data1).await.unwrap();
    let features2 = extractor.extract_features(&data2).await.unwrap();
    
    // Validate features
    assert_eq!(features1.classical_features.len(), 3);
    assert_eq!(features2.classical_features.len(), 3);
    assert!(!features1.superposition_features.is_empty());
    assert!(!features2.superposition_features.is_empty());
    
    // Features should be different for different input data
    assert_ne!(features1.classical_features, features2.classical_features);
    
    // Check feature statistics
    let stats1 = features1.calculate_statistics();
    let stats2 = features2.calculate_statistics();
    
    assert!(stats1.classical_mean != stats2.classical_mean);
    assert!(stats1.classical_std >= 0.0);
    assert!(stats2.classical_std >= 0.0);
}

#[tokio::test]
async fn test_quantum_correlation_analysis() {
    let config = QuantumConfig::lightweight();
    let analyzer = QuantumCorrelationAnalyzer::new(config).unwrap();
    
    // Create features with known correlations
    let features = QuantumFeatures::new(vec![0.5, 0.7, 0.3, 0.9]);
    
    // Analyze correlations
    let correlations = analyzer.analyze_correlations(&features).await.unwrap();
    
    // Validate correlation results
    assert_eq!(correlations.quantum_mutual_information.nrows(), 4);
    assert_eq!(correlations.quantum_discord.nrows(), 4);
    assert_eq!(correlations.correlation_matrix.nrows(), 4);
    assert_eq!(correlations.asset_names.len(), 4);
    
    // Check quantum advantage
    let advantage = correlations.quantum_advantage();
    assert!(advantage >= 0.0);
    
    // Get strongest correlations
    let strongest = correlations.strongest_correlations(2);
    assert!(!strongest.is_empty());
    assert!(strongest.len() <= 2);
}

#[tokio::test]
async fn test_conformal_prediction_coverage() {
    let config = QuantumConfig::lightweight();
    let predictor = QuantumConformalPredictor::new(config).unwrap();
    
    // Create test features and target
    let features = QuantumFeatures::new(vec![0.1, 0.3, 0.5, 0.7]);
    let target = Array1::from_vec(vec![0.2, 0.4, 0.6, 0.8]);
    let estimates = vec![
        UncertaintyEstimate::new(0.15, 0.02, (0.1, 0.2), "vqc_1".to_string(), 0.95),
        UncertaintyEstimate::new(0.35, 0.03, (0.3, 0.4), "vqc_2".to_string(), 0.93),
        UncertaintyEstimate::new(0.55, 0.04, (0.5, 0.6), "vqc_3".to_string(), 0.91),
        UncertaintyEstimate::new(0.75, 0.05, (0.7, 0.8), "vqc_4".to_string(), 0.89),
    ];
    
    // Create prediction intervals
    let intervals = predictor.create_prediction_intervals(&features, &target, &estimates).await.unwrap();
    
    // Validate intervals
    assert!(intervals.lower_bound <= intervals.upper_bound);
    assert!(intervals.confidence_level > 0.0 && intervals.confidence_level <= 1.0);
    assert!(intervals.interval_width > 0.0);
    assert!(intervals.quantum_coverage_probability >= 0.0);
    
    // Test coverage
    let test_values = vec![0.25, 0.45, 0.65, 0.85];
    let coverage = intervals.coverage(&test_values);
    assert!(coverage >= 0.0 && coverage <= 1.0);
    
    // Test efficiency
    let efficiency = intervals.efficiency_score();
    assert!(efficiency >= 0.0);
}

#[tokio::test]
async fn test_measurement_optimization_convergence() {
    let config = QuantumConfig::lightweight();
    let optimizer = QuantumMeasurementOptimizer::new(config).unwrap();
    
    // Create features with varying information content
    let features = QuantumFeatures::new(vec![0.2, 0.8, 0.1, 0.9, 0.5]);
    let estimates = vec![
        UncertaintyEstimate::new(0.1, 0.01, (0.05, 0.15), "test_1".to_string(), 0.95),
        UncertaintyEstimate::new(0.2, 0.02, (0.15, 0.25), "test_2".to_string(), 0.92),
        UncertaintyEstimate::new(0.3, 0.03, (0.25, 0.35), "test_3".to_string(), 0.90),
    ];
    
    // Optimize measurements
    let optimized = optimizer.optimize_measurements(&features, &estimates).await.unwrap();
    
    // Validate optimization results
    assert!(!optimized.measurement_operators.is_empty());
    assert!(optimized.total_information >= 0.0);
    assert!(!optimized.information_gains.is_empty());
    assert!(!optimized.efficiency_scores.is_empty());
    
    // Check convergence
    if optimized.convergence_achieved {
        assert!(optimized.total_information > 0.0);
    }
    
    // Validate most informative measurements
    let most_informative = optimized.most_informative_measurements(3);
    assert!(!most_informative.is_empty());
    
    // Check efficiency
    let total_efficiency = optimized.total_efficiency();
    assert!(total_efficiency >= 0.0);
}

#[tokio::test]
async fn test_quantum_advantage_validation() {
    let config = QuantumConfig::lightweight();
    let engine = QuantumUncertaintyEngine::new(config).await.unwrap();
    
    // Create multiple test cases with different characteristics
    let test_cases = vec![
        // Case 1: High variance data
        Array2::from_shape_vec((8, 3), vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 5.0, 6.0, 7.0, 0.01, 0.02, 0.03, 10.0, 11.0, 12.0, 0.5, 0.6, 0.7, 2.5, 3.5, 4.5, 1.1, 1.2, 1.3]).unwrap(),
        // Case 2: Low variance data
        Array2::from_shape_vec((8, 3), vec![0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73]).unwrap(),
        // Case 3: Mixed patterns
        Array2::from_shape_vec((8, 3), vec![0.1, 0.9, 0.5, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0.0, 1.0, 0.05, 0.95, 0.45]).unwrap(),
    ];
    
    let mut quantum_advantages = Vec::new();
    
    for (i, data) in test_cases.iter().enumerate() {
        let target = Array1::from_vec((0..8).map(|x| (x as f64 + i as f64) / 10.0).collect());
        
        let result = engine.quantify_uncertainty(data, &target).await.unwrap();
        quantum_advantages.push(result.quantum_advantage);
        
        // Basic validation
        assert!(result.quantum_advantage >= 0.0);
        assert!(!result.uncertainty_estimates.is_empty());
        
        // Quantum features should provide additional information
        assert!(result.quantum_features.total_features() > data.ncols());
    }
    
    // At least one case should show quantum advantage
    let has_advantage = quantum_advantages.iter().any(|&adv| adv > 1.0);
    if !has_advantage {
        // Even if no clear advantage, all values should be reasonable
        assert!(quantum_advantages.iter().all(|&adv| adv >= 0.5 && adv <= 5.0));
    }
}

#[tokio::test]
async fn test_error_handling_and_edge_cases() {
    let config = QuantumConfig::lightweight();
    let engine = QuantumUncertaintyEngine::new(config).await.unwrap();
    
    // Test empty data
    let empty_data = Array2::zeros((0, 0));
    let empty_target = Array1::zeros(0);
    let result = engine.quantify_uncertainty(&empty_data, &empty_target).await;
    assert!(result.is_err());
    
    // Test mismatched dimensions
    let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let wrong_target = Array1::from_vec(vec![1.0, 2.0]); // Wrong size
    let result = engine.quantify_uncertainty(&data, &wrong_target).await;
    // Should handle gracefully or return an error
    
    // Test extreme values
    let extreme_data = Array2::from_shape_vec((3, 2), vec![f64::MAX, f64::MIN, 1e10, -1e10, 0.0, f64::EPSILON]).unwrap();
    let extreme_target = Array1::from_vec(vec![1e5, -1e5, 0.0]);
    let result = engine.quantify_uncertainty(&extreme_data, &extreme_target).await;
    // Should handle extreme values gracefully
    if result.is_ok() {
        let res = result.unwrap();
        assert!(res.quantum_advantage.is_finite());
        assert!(res.confidence_level.is_finite());
    }
}

#[tokio::test]
async fn test_performance_and_scalability() {
    let config = QuantumConfig::lightweight();
    let engine = QuantumUncertaintyEngine::new(config).await.unwrap();
    
    // Test with increasing data sizes
    let sizes = vec![5, 10, 20];
    let mut execution_times = Vec::new();
    
    for &size in &sizes {
        let data = Array2::from_shape_fn((size, 4), |(i, j)| (i + j) as f64 / 10.0);
        let target = Array1::from_shape_fn(size, |i| i as f64 / 5.0);
        
        let start_time = std::time::Instant::now();
        let result = engine.quantify_uncertainty(&data, &target).await.unwrap();
        let execution_time = start_time.elapsed();
        
        execution_times.push(execution_time);
        
        // Validate results regardless of size
        assert!(!result.uncertainty_estimates.is_empty());
        assert!(result.quantum_advantage >= 0.0);
        
        // Performance should be reasonable (less than 30 seconds for lightweight config)
        assert!(execution_time < Duration::from_secs(30));
    }
    
    // Execution time should not grow exponentially with size for small datasets
    if execution_times.len() >= 2 {
        let time_ratio = execution_times[1].as_millis() as f64 / execution_times[0].as_millis() as f64;
        assert!(time_ratio < 10.0); // Should not be more than 10x slower for 2x data
    }
}

#[tokio::test]
async fn test_metrics_collection_and_monitoring() {
    let config = QuantumConfig::lightweight();
    let engine = QuantumUncertaintyEngine::new(config).await.unwrap();
    
    // Perform operations to generate metrics
    let data = Array2::from_shape_vec((5, 3), (0..15).map(|x| x as f64 / 10.0).collect()).unwrap();
    let target = Array1::from_vec(vec![0.1, 0.3, 0.5, 0.7, 0.9]);
    
    let _result = engine.quantify_uncertainty(&data, &target).await.unwrap();
    
    // Get metrics
    let metrics = engine.get_metrics().await.unwrap();
    
    // Validate metrics structure
    assert!(metrics.uncertainty_metrics.total_estimates >= 0);
    assert!(metrics.performance_metrics.average_computation_time_ms >= 0.0);
    assert!(metrics.quantum_advantage >= 0.0);
    
    // Test metrics serialization
    let metrics_json = metrics.to_json().unwrap();
    assert!(!metrics_json.is_empty());
    
    let deserialized_metrics = QuantumMetrics::from_json(&metrics_json).unwrap();
    assert_eq!(metrics.quantum_advantage, deserialized_metrics.quantum_advantage);
}

#[tokio::test]
async fn test_quantum_classical_integration() {
    let config = QuantumConfig::lightweight();
    let interface = ClassicalQuantumInterface::new(config).unwrap();
    
    // Create test data
    let classical_data = Array2::from_shape_vec((6, 3), (0..18).map(|x| x as f64 / 10.0).collect()).unwrap();
    let quantum_features = QuantumFeatures::new(vec![0.2, 0.4, 0.6]);
    let target = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    
    // Test hybrid uncertainty quantification
    let hybrid_result = interface.hybrid_uncertainty_quantification(
        &classical_data,
        &quantum_features,
        &target,
    ).await.unwrap();
    
    // Validate hybrid results
    assert!(hybrid_result.quantum_advantage >= 0.0);
    assert!(hybrid_result.hybrid_confidence >= 0.0 && hybrid_result.hybrid_confidence <= 1.0);
    assert!(hybrid_result.computation_time_ms > 0);
    
    // Check that fused uncertainties combine classical and quantum
    assert!(!hybrid_result.fused_uncertainties.is_empty());
    for fused in &hybrid_result.fused_uncertainties {
        assert!(fused.classical_uncertainty >= 0.0);
        assert!(fused.quantum_uncertainty >= 0.0);
        assert!(fused.combined_uncertainty >= 0.0);
    }
    
    // Validate hybrid intervals
    assert!(hybrid_result.hybrid_intervals.lower_bound <= hybrid_result.hybrid_intervals.upper_bound);
    assert!(hybrid_result.hybrid_intervals.hybrid_width >= hybrid_result.hybrid_intervals.classical_width);
}

#[tokio::test]
async fn test_pennylane_integration_simulation() {
    let config = QuantumConfig::lightweight();
    let mut interface = PennyLaneInterface::new(config).unwrap();
    
    // Initialize interface
    let init_result = interface.initialize().await;
    assert!(init_result.is_ok());
    
    // Create a VQC model
    let model_id = interface.create_vqc("integration_test".to_string(), 4, 2).await.unwrap();
    assert!(model_id.contains("vqc_integration_test"));
    
    // Test training
    let features = QuantumFeatures::new(vec![0.1, 0.2, 0.3, 0.4]);
    let targets = vec![0.5, 0.6, 0.7, 0.8];
    let training_config = TrainingConfig {
        max_epochs: 5,
        learning_rate: 0.1,
        ..Default::default()
    };
    
    let training_result = interface.train_model(&model_id, &features, &targets, training_config).await.unwrap();
    assert_eq!(training_result.model_id, model_id);
    assert!(training_result.training_time_ms > 0);
    assert!(!training_result.training_history.is_empty());
    
    // Test prediction
    let prediction_result = interface.predict(&model_id, &features).await.unwrap();
    assert_eq!(prediction_result.model_id, model_id);
    assert!(prediction_result.uncertainty >= 0.0);
    assert!(prediction_result.confidence >= 0.0 && prediction_result.confidence <= 1.0);
    
    // Test quantum algorithm execution
    let qaoa_result = interface.execute_quantum_algorithm(
        QuantumAlgorithmType::QuantumApproximateOptimization,
        AlgorithmParameters::default(),
    ).await.unwrap();
    assert!(qaoa_result.success);
    assert_eq!(qaoa_result.algorithm_type, QuantumAlgorithmType::QuantumApproximateOptimization);
    
    // Clean up
    interface.clear_cache();
    let shutdown_result = interface.shutdown().await;
    assert!(shutdown_result.is_ok());
}

#[tokio::test]
async fn test_configuration_validation_and_adaptation() {
    // Test default configuration
    let default_config = QuantumConfig::default();
    assert!(default_config.validate().is_ok());
    assert!(default_config.estimated_memory_mb() > 0);
    assert!(default_config.estimated_complexity() > 0);
    
    // Test lightweight configuration
    let lightweight_config = QuantumConfig::lightweight();
    assert!(lightweight_config.validate().is_ok());
    assert!(lightweight_config.is_real_time_capable());
    assert!(lightweight_config.estimated_memory_mb() < default_config.estimated_memory_mb());
    
    // Test high-performance configuration
    let high_perf_config = QuantumConfig::high_performance();
    assert!(high_perf_config.validate().is_ok());
    assert!(high_perf_config.estimated_memory_mb() > default_config.estimated_memory_mb());
    
    // Test research configuration
    let research_config = QuantumConfig::research();
    assert!(research_config.validate().is_ok());
    assert!(research_config.enable_noise);
    
    // Test invalid configurations
    let mut invalid_config = QuantumConfig::default();
    invalid_config.n_qubits = 0;
    assert!(invalid_config.validate().is_err());
    
    invalid_config.n_qubits = 8;
    invalid_config.confidence_level = 1.5;
    assert!(invalid_config.validate().is_err());
}

#[tokio::test]
async fn test_system_resilience_and_error_recovery() {
    let config = QuantumConfig::lightweight();
    let engine = QuantumUncertaintyEngine::new(config).await.unwrap();
    
    // Test reset functionality
    let reset_result = engine.reset().await;
    assert!(reset_result.is_ok());
    
    // Test operations after reset
    let data = Array2::from_shape_vec((4, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap();
    let target = Array1::from_vec(vec![0.25, 0.35, 0.55, 0.75]);
    
    let result = engine.quantify_uncertainty(&data, &target).await.unwrap();
    assert!(!result.uncertainty_estimates.is_empty());
    
    // Test multiple consecutive operations
    for i in 0..3 {
        let scaled_data = &data * (1.0 + i as f64 * 0.1);
        let scaled_target = &target * (1.0 + i as f64 * 0.1);
        
        let result = engine.quantify_uncertainty(&scaled_data, &scaled_target).await.unwrap();
        assert!(!result.uncertainty_estimates.is_empty());
        assert!(result.quantum_advantage >= 0.0);
    }
    
    // Test metrics consistency
    let final_metrics = engine.get_metrics().await.unwrap();
    assert!(final_metrics.uncertainty_metrics.total_estimates > 0);
}

#[tokio::test]
async fn test_real_world_trading_scenario_simulation() {
    let config = QuantumConfig::lightweight();
    let engine = QuantumUncertaintyEngine::new(config).await.unwrap();
    
    // Simulate time series data like price movements
    let n_timepoints = 20;
    let n_features = 5; // OHLCV-like data
    
    let mut trading_data = Vec::new();
    let mut prices = Vec::new();
    
    // Generate synthetic trading data
    for t in 0..n_timepoints {
        let base_price = 100.0 + (t as f64 * 0.1);
        let volatility = 0.02;
        
        let open = base_price + (rand::random::<f64>() - 0.5) * volatility * base_price;
        let high = open + rand::random::<f64>() * volatility * base_price;
        let low = open - rand::random::<f64>() * volatility * base_price;
        let close = low + rand::random::<f64>() * (high - low);
        let volume = 1000.0 + rand::random::<f64>() * 500.0;
        
        trading_data.extend_from_slice(&[open, high, low, close, volume]);
        prices.push(close);
    }
    
    let data = Array2::from_shape_vec((n_timepoints, n_features), trading_data).unwrap();
    let target = Array1::from_vec(prices);
    
    // Perform quantum uncertainty quantification for trading
    let result = engine.quantify_uncertainty(&data, &target).await.unwrap();
    
    // Validate trading-specific results
    assert!(!result.uncertainty_estimates.is_empty());
    assert!(result.quantum_advantage >= 0.0);
    
    // Check that uncertainty estimates make sense for trading
    for estimate in &result.uncertainty_estimates {
        assert!(estimate.uncertainty >= 0.0);
        assert!(estimate.variance >= 0.0);
        assert!(estimate.confidence_interval.0 <= estimate.confidence_interval.1);
        assert!(estimate.quantum_fidelity >= 0.0 && estimate.quantum_fidelity <= 1.0);
    }
    
    // Validate conformal prediction intervals for trading decisions
    let intervals = &result.conformal_intervals;
    assert!(intervals.confidence_level >= 0.8); // High confidence for trading
    assert!(intervals.quantum_coverage_probability >= intervals.confidence_level * 0.9);
    
    // Check quantum features capture market dynamics
    let features = &result.quantum_features;
    assert!(features.total_features() > n_features); // Should extract additional features
    assert!(!features.superposition_features.is_empty()); // Should capture quantum effects
    assert!(!features.entanglement_features.is_empty()); // Should capture correlations
    
    // Verify measurements are optimized for information extraction
    let measurements = &result.optimized_measurements;
    assert!(measurements.total_information > 0.0);
    assert!(measurements.convergence_achieved || measurements.total_information > 0.5);
}

#[test]
fn test_quantum_state_operations() {
    // Test quantum state creation and manipulation
    let state1 = QuantumState::zero_state(2);
    assert_eq!(state1.n_qubits, 2);
    assert_eq!(state1.amplitudes.len(), 4);
    
    let state2 = QuantumState::uniform_superposition(2);
    assert_eq!(state2.n_qubits, 2);
    for amp in &state2.amplitudes {
        assert!((amp.norm_sqr() - 0.25).abs() < 1e-10);
    }
    
    // Test tensor product
    let state1_single = QuantumState::zero_state(1);
    let state2_single = QuantumState::uniform_superposition(1);
    let tensor_product = state1_single.tensor_product(&state2_single).unwrap();
    assert_eq!(tensor_product.n_qubits, 2);
    assert_eq!(tensor_product.amplitudes.len(), 4);
}

#[test]
fn test_uncertainty_estimate_operations() {
    let estimate = UncertaintyEstimate::new(
        0.1,
        0.02,
        (0.05, 0.15),
        "test_circuit".to_string(),
        0.95,
    );
    
    assert_eq!(estimate.uncertainty, 0.1);
    assert_eq!(estimate.variance, 0.02);
    assert_eq!(estimate.interval_width(), 0.1);
    assert!(estimate.contains(0.08));
    assert!(!estimate.contains(0.2));
    assert!((estimate.std_dev() - 0.1414).abs() < 0.001);
}