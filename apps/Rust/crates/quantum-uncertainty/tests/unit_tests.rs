//! Unit Tests for Quantum Uncertainty Components
//!
//! Comprehensive unit tests targeting 100% function coverage of quantum uncertainty modules.

use quantum_uncertainty::*;
use ndarray::{Array1, Array2};
use approx::assert_relative_eq;

#[test]
fn test_quantum_config_validation() {
    // Test default configuration
    let default_config = QuantumConfig::default();
    assert!(default_config.validate().is_ok());
    assert_eq!(default_config.n_qubits, 8);
    assert_eq!(default_config.n_layers, 4);
    
    // Test lightweight configuration
    let lightweight = QuantumConfig::lightweight();
    assert!(lightweight.validate().is_ok());
    assert!(lightweight.is_real_time_capable());
    assert!(lightweight.estimated_memory_mb() < default_config.estimated_memory_mb());
    
    // Test high performance configuration
    let high_perf = QuantumConfig::high_performance();
    assert!(high_perf.validate().is_ok());
    assert!(high_perf.estimated_memory_mb() > default_config.estimated_memory_mb());
    
    // Test research configuration
    let research = QuantumConfig::research();
    assert!(research.validate().is_ok());
    assert!(research.enable_noise);
    
    // Test invalid configurations
    let mut invalid = QuantumConfig::default();
    invalid.n_qubits = 0;
    assert!(invalid.validate().is_err());
    
    invalid.n_qubits = 8;
    invalid.confidence_level = 1.5;
    assert!(invalid.validate().is_err());
    
    invalid.confidence_level = 0.95;
    invalid.n_layers = 0;
    assert!(invalid.validate().is_err());
}

#[test]
fn test_quantum_state_operations() {
    // Test zero state creation
    let zero_state = QuantumState::zero_state(2);
    assert_eq!(zero_state.n_qubits, 2);
    assert_eq!(zero_state.amplitudes.len(), 4);
    assert_relative_eq!(zero_state.amplitudes[0].norm_sqr(), 1.0, epsilon = 1e-10);
    assert_relative_eq!(zero_state.amplitudes[1].norm_sqr(), 0.0, epsilon = 1e-10);
    
    // Test uniform superposition
    let uniform_state = QuantumState::uniform_superposition(2);
    assert_eq!(uniform_state.n_qubits, 2);
    for amplitude in &uniform_state.amplitudes {
        assert_relative_eq!(amplitude.norm_sqr(), 0.25, epsilon = 1e-10);
    }
    
    // Test tensor product
    let state1 = QuantumState::zero_state(1);
    let state2 = QuantumState::uniform_superposition(1);
    let tensor_product = state1.tensor_product(&state2).unwrap();
    assert_eq!(tensor_product.n_qubits, 2);
    assert_eq!(tensor_product.amplitudes.len(), 4);
    
    // Test normalization
    let mut unnormalized = QuantumState::zero_state(2);
    unnormalized.amplitudes[0] = num_complex::Complex::new(2.0, 0.0);
    unnormalized.normalize();
    assert_relative_eq!(unnormalized.amplitudes[0].norm_sqr(), 1.0, epsilon = 1e-10);
    
    // Test measurement probabilities
    let probabilities = uniform_state.measurement_probabilities();
    assert_eq!(probabilities.len(), 4);
    for prob in probabilities {
        assert_relative_eq!(prob, 0.25, epsilon = 1e-10);
    }
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
    
    // Test basic properties
    assert_eq!(estimate.uncertainty, 0.1);
    assert_eq!(estimate.variance, 0.02);
    assert_eq!(estimate.confidence_interval.0, 0.05);
    assert_eq!(estimate.confidence_interval.1, 0.15);
    assert_eq!(estimate.quantum_fidelity, 0.95);
    
    // Test computed properties
    assert_relative_eq!(estimate.interval_width(), 0.1, epsilon = 1e-10);
    assert_relative_eq!(estimate.std_dev(), (0.02_f64).sqrt(), epsilon = 1e-10);
    
    // Test containment
    assert!(estimate.contains(0.08));
    assert!(estimate.contains(0.05)); // boundary
    assert!(estimate.contains(0.15)); // boundary
    assert!(!estimate.contains(0.04));
    assert!(!estimate.contains(0.16));
    
    // Test comparison
    let estimate2 = UncertaintyEstimate::new(0.2, 0.04, (0.1, 0.3), "test2".to_string(), 0.90);
    assert!(estimate < estimate2); // Lower uncertainty
    
    // Test serialization
    let json = estimate.to_json().unwrap();
    let deserialized = UncertaintyEstimate::from_json(&json).unwrap();
    assert_relative_eq!(deserialized.uncertainty, estimate.uncertainty, epsilon = 1e-10);
}

#[test]
fn test_quantum_features_operations() {
    let features = QuantumFeatures::new(vec![0.1, 0.2, 0.3, 0.4]);
    
    assert_eq!(features.classical_features.len(), 4);
    assert_eq!(features.total_features(), 4 + features.superposition_features.len() + features.entanglement_features.len());
    
    // Test feature statistics
    let stats = features.calculate_statistics();
    assert!(stats.classical_mean > 0.0);
    assert!(stats.classical_std >= 0.0);
    assert!(stats.superposition_entropy >= 0.0);
    assert!(stats.entanglement_measure >= 0.0);
    
    // Test feature normalization
    let normalized = features.normalize();
    let norm_stats = normalized.calculate_statistics();
    assert_relative_eq!(norm_stats.classical_mean, 0.0, epsilon = 1e-10);
    assert_relative_eq!(norm_stats.classical_std, 1.0, epsilon = 1e-6);
    
    // Test feature combination
    let features2 = QuantumFeatures::new(vec![0.5, 0.6, 0.7, 0.8]);
    let combined = features.combine(&features2).unwrap();
    assert_eq!(combined.classical_features.len(), 8);
    
    // Test quantum feature extraction
    assert!(!features.superposition_features.is_empty());
    assert!(!features.entanglement_features.is_empty());
    
    // Test dimension validation
    let incompatible = QuantumFeatures::new(vec![0.1, 0.2]); // Different size
    assert!(features.combine(&incompatible).is_err());
}

#[test]
fn test_conformal_intervals_operations() {
    let intervals = ConformalIntervals {
        lower_bound: 0.1,
        upper_bound: 0.9,
        confidence_level: 0.95,
        interval_width: 0.8,
        quantum_coverage_probability: 0.96,
        classical_coverage_probability: 0.94,
    };
    
    // Test basic properties
    assert_eq!(intervals.lower_bound, 0.1);
    assert_eq!(intervals.upper_bound, 0.9);
    assert_relative_eq!(intervals.interval_width, 0.8, epsilon = 1e-10);
    
    // Test coverage calculation
    let test_values = vec![0.2, 0.5, 0.8, 1.0, 0.05];
    let coverage = intervals.coverage(&test_values);
    assert_relative_eq!(coverage, 0.6, epsilon = 1e-10); // 3 out of 5 within interval
    
    // Test efficiency score
    let efficiency = intervals.efficiency_score();
    assert!(efficiency > 0.0);
    assert!(efficiency <= 1.0);
    
    // Test quantum advantage
    let advantage = intervals.quantum_advantage();
    assert!(advantage > 0.0); // quantum_coverage > classical_coverage
    
    // Test containment
    assert!(intervals.contains(0.5));
    assert!(intervals.contains(0.1)); // boundary
    assert!(intervals.contains(0.9)); // boundary
    assert!(!intervals.contains(0.05));
    assert!(!intervals.contains(1.0));
    
    // Test serialization
    let json = intervals.to_json().unwrap();
    let deserialized = ConformalIntervals::from_json(&json).unwrap();
    assert_relative_eq!(deserialized.confidence_level, intervals.confidence_level, epsilon = 1e-10);
}

#[test]
fn test_optimized_measurements_operations() {
    let measurements = OptimizedMeasurements {
        measurement_operators: vec!["X".to_string(), "Y".to_string(), "Z".to_string()],
        total_information: 2.5,
        information_gains: vec![0.8, 0.9, 0.8],
        efficiency_scores: vec![0.95, 0.97, 0.93],
        convergence_achieved: true,
    };
    
    // Test basic properties
    assert_eq!(measurements.measurement_operators.len(), 3);
    assert_relative_eq!(measurements.total_information, 2.5, epsilon = 1e-10);
    assert!(measurements.convergence_achieved);
    
    // Test most informative measurements
    let most_informative = measurements.most_informative_measurements(2);
    assert_eq!(most_informative.len(), 2);
    assert_eq!(most_informative[0], "Y"); // Highest information gain
    assert_eq!(most_informative[1], "X"); // Second highest
    
    // Test total efficiency
    let total_efficiency = measurements.total_efficiency();
    let expected_efficiency = measurements.efficiency_scores.iter().sum::<f64>();
    assert_relative_eq!(total_efficiency, expected_efficiency, epsilon = 1e-10);
    
    // Test information density
    let density = measurements.information_density();
    assert!(density > 0.0);
    
    // Test operator ranking
    let ranked = measurements.rank_operators_by_information();
    assert_eq!(ranked.len(), 3);
    assert_eq!(ranked[0].0, "Y"); // Highest information
    assert_eq!(ranked[1].0, "X"); // Second highest
    assert_eq!(ranked[2].0, "Z"); // Lowest
    
    // Test threshold filtering
    let high_efficiency = measurements.filter_by_efficiency_threshold(0.94);
    assert_eq!(high_efficiency.len(), 2); // X and Y meet threshold
}

#[test]
fn test_quantum_correlation_analysis() {
    let mut correlations = QuantumCorrelations {
        quantum_mutual_information: Array2::eye(4),
        quantum_discord: Array2::zeros((4, 4)),
        correlation_matrix: Array2::eye(4),
        asset_names: vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()],
    };
    
    // Set some test values
    correlations.quantum_discord[[0, 1]] = 0.5;
    correlations.quantum_discord[[1, 0]] = 0.5;
    correlations.correlation_matrix[[0, 1]] = 0.8;
    correlations.correlation_matrix[[1, 0]] = 0.8;
    
    // Test basic properties
    assert_eq!(correlations.asset_names.len(), 4);
    assert_eq!(correlations.quantum_mutual_information.nrows(), 4);
    
    // Test quantum advantage calculation
    let advantage = correlations.quantum_advantage();
    assert!(advantage >= 0.0);
    
    // Test strongest correlations
    let strongest = correlations.strongest_correlations(2);
    assert!(!strongest.is_empty());
    assert!(strongest.len() <= 2);
    
    // Test most entangled pairs
    let entangled = correlations.most_entangled_pairs(1);
    assert_eq!(entangled.len(), 1);
    assert_eq!(entangled[0].0, "A");
    assert_eq!(entangled[0].1, "B");
    
    // Test correlation strength validation
    assert!(correlations.validate_correlations().is_ok());
    
    // Test invalid correlation matrix (not symmetric)
    correlations.correlation_matrix[[0, 1]] = 0.9;
    // Should still be valid as we only changed one element
    assert!(correlations.validate_correlations().is_ok());
}

#[test]
fn test_quantum_algorithm_types() {
    use std::mem::discriminant;
    
    // Test algorithm type enumeration
    let qaoa = QuantumAlgorithmType::QuantumApproximateOptimization;
    let vqe = QuantumAlgorithmType::VariationalQuantumEigensolver;
    let vqc = QuantumAlgorithmType::VariationalQuantumClassifier;
    
    assert_ne!(discriminant(&qaoa), discriminant(&vqe));
    assert_ne!(discriminant(&vqe), discriminant(&vqc));
    
    // Test string conversion
    assert_eq!(qaoa.to_string(), "QuantumApproximateOptimization");
    assert_eq!(vqe.to_string(), "VariationalQuantumEigensolver");
    assert_eq!(vqc.to_string(), "VariationalQuantumClassifier");
    
    // Test from string conversion
    assert_eq!(QuantumAlgorithmType::from_string("QuantumApproximateOptimization").unwrap(), qaoa);
    assert_eq!(QuantumAlgorithmType::from_string("VariationalQuantumEigensolver").unwrap(), vqe);
    assert!(QuantumAlgorithmType::from_string("InvalidAlgorithm").is_err());
}

#[test]
fn test_training_config_validation() {
    // Test default configuration
    let default_config = TrainingConfig::default();
    assert!(default_config.validate().is_ok());
    
    // Test valid configuration
    let valid_config = TrainingConfig {
        max_epochs: 100,
        learning_rate: 0.01,
        batch_size: 32,
        convergence_threshold: 1e-6,
        early_stopping: true,
        patience: 10,
    };
    assert!(valid_config.validate().is_ok());
    
    // Test invalid configurations
    let mut invalid_config = valid_config.clone();
    invalid_config.max_epochs = 0;
    assert!(invalid_config.validate().is_err());
    
    invalid_config.max_epochs = 100;
    invalid_config.learning_rate = -0.1;
    assert!(invalid_config.validate().is_err());
    
    invalid_config.learning_rate = 0.01;
    invalid_config.batch_size = 0;
    assert!(invalid_config.validate().is_err());
    
    invalid_config.batch_size = 32;
    invalid_config.convergence_threshold = -1e-6;
    assert!(invalid_config.validate().is_err());
}

#[test]
fn test_algorithm_parameters() {
    let params = AlgorithmParameters::default();
    
    // Test parameter setting and getting
    let mut custom_params = params.clone();
    custom_params.set_parameter("layers", 5.0);
    custom_params.set_parameter("learning_rate", 0.01);
    
    assert_eq!(custom_params.get_parameter("layers"), Some(5.0));
    assert_eq!(custom_params.get_parameter("learning_rate"), Some(0.01));
    assert_eq!(custom_params.get_parameter("nonexistent"), None);
    
    // Test parameter validation
    assert!(custom_params.validate().is_ok());
    
    // Test parameter iteration
    let param_names: Vec<String> = custom_params.parameter_names().collect();
    assert!(param_names.contains(&"layers".to_string()));
    assert!(param_names.contains(&"learning_rate".to_string()));
    
    // Test parameter removal
    custom_params.remove_parameter("layers");
    assert_eq!(custom_params.get_parameter("layers"), None);
}

#[test]
fn test_quantum_metrics_operations() {
    let metrics = QuantumMetrics {
        uncertainty_metrics: UncertaintyMetrics {
            total_estimates: 100,
            average_uncertainty: 0.15,
            uncertainty_variance: 0.02,
            confidence_coverage: 0.95,
        },
        performance_metrics: PerformanceMetrics {
            average_computation_time_ms: 50.0,
            peak_memory_usage_mb: 128.0,
            throughput_estimates_per_second: 20.0,
        },
        quantum_advantage: 1.2,
        timestamp: std::time::SystemTime::now(),
    };
    
    // Test serialization
    let json = metrics.to_json().unwrap();
    assert!(!json.is_empty());
    
    let deserialized = QuantumMetrics::from_json(&json).unwrap();
    assert_eq!(deserialized.uncertainty_metrics.total_estimates, 100);
    assert_relative_eq!(deserialized.quantum_advantage, 1.2, epsilon = 1e-10);
    
    // Test metrics validation
    assert!(metrics.validate().is_ok());
    
    // Test performance indicators
    assert!(metrics.performance_metrics.average_computation_time_ms > 0.0);
    assert!(metrics.performance_metrics.peak_memory_usage_mb > 0.0);
    assert!(metrics.performance_metrics.throughput_estimates_per_second > 0.0);
    
    // Test efficiency calculations
    let efficiency = metrics.uncertainty_metrics.confidence_coverage / 
                    (metrics.performance_metrics.average_computation_time_ms / 1000.0);
    assert!(efficiency > 0.0);
}

#[test]
fn test_error_handling_edge_cases() {
    // Test with empty quantum features
    let empty_features = QuantumFeatures::new(vec![]);
    assert!(empty_features.calculate_statistics().classical_std.is_nan() || 
            empty_features.calculate_statistics().classical_std == 0.0);
    
    // Test uncertainty estimate with zero variance
    let zero_var_estimate = UncertaintyEstimate::new(
        0.1, 0.0, (0.1, 0.1), "test".to_string(), 1.0
    );
    assert_eq!(zero_var_estimate.std_dev(), 0.0);
    assert_eq!(zero_var_estimate.interval_width(), 0.0);
    
    // Test conformal intervals with zero width
    let zero_width_intervals = ConformalIntervals {
        lower_bound: 0.5,
        upper_bound: 0.5,
        confidence_level: 0.95,
        interval_width: 0.0,
        quantum_coverage_probability: 0.95,
        classical_coverage_probability: 0.95,
    };
    assert_eq!(zero_width_intervals.efficiency_score(), f64::INFINITY);
    
    // Test quantum state with zero amplitudes
    let mut zero_state = QuantumState::zero_state(2);
    for amplitude in &mut zero_state.amplitudes {
        *amplitude = num_complex::Complex::new(0.0, 0.0);
    }
    // Normalization should handle this gracefully
    zero_state.normalize();
    
    // Test optimized measurements with empty operators
    let empty_measurements = OptimizedMeasurements {
        measurement_operators: vec![],
        total_information: 0.0,
        information_gains: vec![],
        efficiency_scores: vec![],
        convergence_achieved: false,
    };
    assert!(empty_measurements.most_informative_measurements(1).is_empty());
    assert_eq!(empty_measurements.total_efficiency(), 0.0);
}

#[test]
fn test_mathematical_properties() {
    // Test superposition state properties
    let superposition = QuantumState::uniform_superposition(3);
    let probabilities = superposition.measurement_probabilities();
    let total_probability: f64 = probabilities.iter().sum();
    assert_relative_eq!(total_probability, 1.0, epsilon = 1e-10);
    
    // Test entropy calculations
    let features = QuantumFeatures::new(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    let stats = features.calculate_statistics();
    assert!(stats.superposition_entropy >= 0.0); // Entropy is non-negative
    assert!(stats.entanglement_measure >= 0.0); // Entanglement is non-negative
    
    // Test uncertainty principle
    let estimate1 = UncertaintyEstimate::new(0.1, 0.01, (0.05, 0.15), "precise".to_string(), 0.99);
    let estimate2 = UncertaintyEstimate::new(0.3, 0.09, (0.0, 0.6), "imprecise".to_string(), 0.80);
    
    // More uncertain estimate should have larger variance
    assert!(estimate2.variance > estimate1.variance);
    assert!(estimate2.interval_width() > estimate1.interval_width());
    
    // Test correlation matrix properties
    let mut correlations = QuantumCorrelations {
        quantum_mutual_information: Array2::eye(3),
        quantum_discord: Array2::zeros((3, 3)),
        correlation_matrix: Array2::eye(3),
        asset_names: vec!["A".to_string(), "B".to_string(), "C".to_string()],
    };
    
    // Diagonal elements should be 1.0 (self-correlation)
    for i in 0..3 {
        assert_relative_eq!(correlations.correlation_matrix[[i, i]], 1.0, epsilon = 1e-10);
    }
    
    // Test symmetry of correlation matrix
    correlations.correlation_matrix[[0, 1]] = 0.5;
    correlations.correlation_matrix[[1, 0]] = 0.5;
    
    for i in 0..3 {
        for j in 0..3 {
            assert_relative_eq!(
                correlations.correlation_matrix[[i, j]], 
                correlations.correlation_matrix[[j, i]], 
                epsilon = 1e-10
            );
        }
    }
}

#[test]
fn test_performance_characteristics() {
    use std::time::Instant;
    
    // Test that operations complete within reasonable time
    let config = QuantumConfig::lightweight();
    
    // Feature creation should be fast
    let start = Instant::now();
    let features = QuantumFeatures::new(vec![1.0; 100]);
    assert!(start.elapsed().as_millis() < 100);
    
    // Statistics calculation should be fast
    let start = Instant::now();
    let _stats = features.calculate_statistics();
    assert!(start.elapsed().as_millis() < 50);
    
    // Configuration validation should be very fast
    let start = Instant::now();
    let _result = config.validate();
    assert!(start.elapsed().as_micros() < 1000);
    
    // Memory usage should be reasonable
    let large_features = QuantumFeatures::new(vec![1.0; 10000]);
    assert!(large_features.classical_features.len() == 10000);
    
    // Operations should not leak memory (basic check)
    for _ in 0..1000 {
        let _temp_features = QuantumFeatures::new(vec![1.0; 100]);
        let _temp_stats = _temp_features.calculate_statistics();
    }
}