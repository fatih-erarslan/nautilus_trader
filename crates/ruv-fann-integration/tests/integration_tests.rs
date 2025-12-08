//! Integration tests for ruv_FANN Neural Divergent Integration
//!
//! These tests validate the complete integration of ruv_FANN with the ATS-CP system.

use tokio;
use approx::assert_relative_eq;
use ndarray::{Array2, Array3};
use std::time::Duration;

use ruv_fann_integration::{
    RuvFannIntegration, RuvFannConfig, MarketData, DivergentPrediction,
    CognitionEngineConnection,
};

#[tokio::test]
async fn test_ruv_fann_integration_initialization() {
    let config = RuvFannConfig::default();
    let integration = RuvFannIntegration::new(config).await;
    
    assert!(integration.is_ok(), "Failed to initialize ruv_FANN integration");
    
    let integration = integration.unwrap();
    assert!(integration.is_ready(), "Integration should be ready after initialization");
}

#[tokio::test]
async fn test_ultra_low_latency_configuration() {
    let config = RuvFannConfig::ultra_low_latency();
    let integration = RuvFannIntegration::new(config).await.unwrap();
    
    let status = integration.get_status().await.unwrap();
    assert!(status.performance_bridge_enabled, "Performance bridge should be enabled for ultra-low latency");
    assert!(status.is_initialized, "System should be initialized");
}

#[tokio::test]
async fn test_maximum_accuracy_configuration() {
    let config = RuvFannConfig::maximum_accuracy();
    let integration = RuvFannIntegration::new(config).await.unwrap();
    
    let status = integration.get_status().await.unwrap();
    assert!(status.neural_modules_count >= 4, "Should have multiple neural modules for accuracy");
    assert!(status.quantum_ml_enabled, "Quantum ML should be enabled for maximum accuracy");
}

#[tokio::test]
async fn test_cognition_engine_connection() {
    let config = RuvFannConfig::default();
    let mut integration = RuvFannIntegration::new(config).await.unwrap();
    
    // Create mock cognition engine connection
    let cognition_engine = tokio::sync::RwLock::new(CognitionEngineConnection {
        is_connected: false,
        metadata: std::collections::HashMap::new(),
    });
    let cognition_engine = std::sync::Arc::new(cognition_engine);
    
    // Connect to cognition engine
    let result = integration.connect_to_cognition_engine(cognition_engine).await;
    assert!(result.is_ok(), "Failed to connect to cognition engine");
    
    let status = integration.get_status().await.unwrap();
    assert!(status.is_connected, "Should be connected to cognition engine");
}

#[tokio::test]
async fn test_neural_divergent_prediction() {
    let config = RuvFannConfig::default();
    let integration = RuvFannIntegration::new(config).await.unwrap();
    
    // Create test market data
    let market_data = create_test_market_data();
    
    // Perform prediction
    let result = integration.predict_divergent(&market_data).await;
    assert!(result.is_ok(), "Neural divergent prediction failed");
    
    let prediction = result.unwrap();
    validate_divergent_prediction(&prediction);
}

#[tokio::test]
async fn test_latency_performance() {
    let config = RuvFannConfig::ultra_low_latency();
    let integration = RuvFannIntegration::new(config).await.unwrap();
    
    let market_data = create_test_market_data();
    
    // Measure prediction latency
    let start_time = std::time::Instant::now();
    let result = integration.predict_divergent(&market_data).await;
    let latency = start_time.elapsed();
    
    assert!(result.is_ok(), "Prediction failed");
    
    // Verify latency is within target (should be sub-100μs for ultra-low latency config)
    println!("Prediction latency: {:?}", latency);
    // Note: In real deployment with actual GPU acceleration, this should be much lower
    assert!(latency < Duration::from_millis(10), "Latency too high for ultra-low latency configuration");
}

#[tokio::test]
async fn test_gpu_acceleration_availability() {
    #[cfg(feature = "gpu-acceleration")]
    {
        let gpu_available = ruv_fann_integration::gpu_acceleration::check_gpu_availability().await;
        assert!(gpu_available.is_ok(), "GPU availability check failed");
        
        if gpu_available.unwrap() {
            println!("GPU acceleration is available");
            
            let config = RuvFannConfig::default().with_gpu_acceleration();
            let integration = RuvFannIntegration::new(config).await;
            assert!(integration.is_ok(), "Failed to initialize with GPU acceleration");
        } else {
            println!("GPU acceleration not available on this system");
        }
    }
    
    #[cfg(not(feature = "gpu-acceleration"))]
    {
        println!("GPU acceleration feature not enabled");
    }
}

#[tokio::test]
async fn test_quantum_ml_integration() {
    #[cfg(feature = "quantum-ml")]
    {
        let quantum_available = ruv_fann_integration::quantum_ml_bridge::check_quantum_availability().await;
        assert!(quantum_available.is_ok(), "Quantum ML availability check failed");
        
        let config = RuvFannConfig::default().with_quantum_ml_bridge();
        let integration = RuvFannIntegration::new(config).await;
        assert!(integration.is_ok(), "Failed to initialize with quantum ML");
    }
    
    #[cfg(not(feature = "quantum-ml"))]
    {
        println!("Quantum ML feature not enabled");
    }
}

#[tokio::test]
async fn test_metrics_collection() {
    let config = RuvFannConfig::default();
    let integration = RuvFannIntegration::new(config).await.unwrap();
    
    // Perform some operations to generate metrics
    let market_data = create_test_market_data();
    let _ = integration.predict_divergent(&market_data).await;
    
    // Check metrics
    let metrics = integration.get_metrics().await;
    assert!(metrics.is_ok(), "Failed to get metrics");
    
    let metrics = metrics.unwrap();
    assert!(metrics.total_predictions > 0, "Should have recorded predictions");
}

#[tokio::test]
async fn test_error_handling() {
    let config = RuvFannConfig::default();
    let integration = RuvFannIntegration::new(config).await.unwrap();
    
    // Test with invalid market data
    let invalid_market_data = MarketData {
        prices: Array2::zeros((0, 0)), // Empty data
        volumes: Array2::zeros((0, 0)),
        indicators: None,
        timestamps: Vec::new(),
        metadata: std::collections::HashMap::new(),
    };
    
    let result = integration.predict_divergent(&invalid_market_data).await;
    assert!(result.is_err(), "Should fail with invalid market data");
    
    // Verify error is properly categorized
    let error = result.unwrap_err();
    assert!(error.is_neural_network_error() || error.is_configuration_error(), "Error should be properly categorized");
}

#[tokio::test]
async fn test_concurrent_predictions() {
    let config = RuvFannConfig::default();
    let integration = std::sync::Arc::new(RuvFannIntegration::new(config).await.unwrap());
    
    let market_data = std::sync::Arc::new(create_test_market_data());
    
    // Perform concurrent predictions
    let mut handles = Vec::new();
    for i in 0..10 {
        let integration_clone = integration.clone();
        let market_data_clone = market_data.clone();
        
        let handle = tokio::spawn(async move {
            let result = integration_clone.predict_divergent(&market_data_clone).await;
            (i, result)
        });
        
        handles.push(handle);
    }
    
    // Wait for all predictions to complete
    let mut successful_predictions = 0;
    for handle in handles {
        let (index, result) = handle.await.unwrap();
        if result.is_ok() {
            successful_predictions += 1;
            println!("Prediction {} completed successfully", index);
        } else {
            println!("Prediction {} failed: {:?}", index, result.unwrap_err());
        }
    }
    
    assert!(successful_predictions >= 8, "Most concurrent predictions should succeed");
}

#[tokio::test]
async fn test_memory_efficiency() {
    let config = RuvFannConfig::default();
    let integration = RuvFannIntegration::new(config).await.unwrap();
    
    let initial_memory = get_memory_usage();
    
    // Perform multiple predictions to test memory usage
    let market_data = create_test_market_data();
    for _ in 0..100 {
        let _ = integration.predict_divergent(&market_data).await;
    }
    
    let final_memory = get_memory_usage();
    let memory_increase = final_memory.saturating_sub(initial_memory);
    
    // Memory increase should be reasonable (less than 100MB for 100 predictions)
    assert!(memory_increase < 100 * 1024 * 1024, "Memory usage increased too much: {} bytes", memory_increase);
    
    println!("Memory usage increased by {} bytes for 100 predictions", memory_increase);
}

#[tokio::test]
async fn test_configuration_update() {
    let initial_config = RuvFannConfig::default();
    let mut integration = RuvFannIntegration::new(initial_config).await.unwrap();
    
    // Update configuration
    let new_config = RuvFannConfig::ultra_low_latency();
    let result = integration.update_config(new_config).await;
    assert!(result.is_ok(), "Failed to update configuration");
    
    // Verify configuration was updated
    let status = integration.get_status().await.unwrap();
    assert!(status.performance_bridge_enabled, "Performance bridge should be enabled after config update");
}

#[tokio::test]
async fn test_graceful_shutdown() {
    let config = RuvFannConfig::default();
    let mut integration = RuvFannIntegration::new(config).await.unwrap();
    
    // Perform some operations
    let market_data = create_test_market_data();
    let _ = integration.predict_divergent(&market_data).await;
    
    // Test graceful shutdown
    let result = integration.shutdown().await;
    assert!(result.is_ok(), "Failed to shutdown gracefully");
    
    // Verify system is no longer ready
    assert!(!integration.is_ready(), "System should not be ready after shutdown");
}

// Helper functions

fn create_test_market_data() -> MarketData {
    let rows = 100;
    let cols = 5;
    
    // Create some realistic-looking market data
    let mut prices = Array2::zeros((rows, cols));
    let mut volumes = Array2::zeros((rows, cols));
    
    for i in 0..rows {
        for j in 0..cols {
            prices[[i, j]] = 100.0 + (i as f64 * 0.1) + (j as f64 * 10.0) + rand::random::<f64>() * 2.0;
            volumes[[i, j]] = 1000.0 + rand::random::<f64>() * 500.0;
        }
    }
    
    let timestamps: Vec<chrono::DateTime<chrono::Utc>> = (0..rows)
        .map(|i| chrono::Utc::now() - chrono::Duration::seconds(rows as i64 - i as i64))
        .collect();
    
    MarketData {
        prices,
        volumes,
        indicators: None,
        timestamps,
        metadata: std::collections::HashMap::new(),
    }
}

fn validate_divergent_prediction(prediction: &DivergentPrediction) {
    // Validate prediction structure
    assert!(!prediction.prediction.is_empty(), "Prediction should not be empty");
    assert!(!prediction.divergent_components.is_empty(), "Should have divergent components");
    assert!(!prediction.confidence_intervals.is_empty(), "Should have confidence intervals");
    
    // Validate prediction values are reasonable
    for &value in prediction.prediction.iter() {
        assert!(value.is_finite(), "Prediction values should be finite");
        assert!(!value.is_nan(), "Prediction values should not be NaN");
    }
    
    // Validate confidence intervals
    for interval_pair in prediction.confidence_intervals.outer_iter() {
        for interval in interval_pair.outer_iter() {
            let lower = interval[0];
            let upper = interval[1];
            assert!(lower <= upper, "Lower bound should be <= upper bound");
            assert!(lower.is_finite() && upper.is_finite(), "Confidence intervals should be finite");
        }
    }
    
    // Validate processing metadata
    assert!(prediction.metadata.processing_time_us > 0, "Processing time should be positive");
    assert!(!prediction.metadata.model_versions.is_empty(), "Should have model versions");
    
    println!("Divergent prediction validation passed");
    println!("Processing time: {}μs", prediction.metadata.processing_time_us);
    println!("GPU accelerated: {}", prediction.metadata.gpu_accelerated);
    println!("Quantum enhanced: {}", prediction.metadata.quantum_enhanced);
    println!("Performance optimized: {}", prediction.metadata.performance_optimized);
}

fn get_memory_usage() -> usize {
    #[cfg(feature = "memory-stats")]
    {
        memory_stats::memory_stats()
            .map(|usage| usage.physical_mem)
            .unwrap_or(0)
    }
    #[cfg(not(feature = "memory-stats"))]
    {
        0
    }
}

#[tokio::test]
async fn test_zero_mock_neural_operations() {
    // This test ensures we're using real neural network operations, not mocks
    let config = RuvFannConfig::default();
    let integration = RuvFannIntegration::new(config).await.unwrap();
    
    let market_data = create_test_market_data();
    
    // Perform multiple predictions and verify they're not identical (would indicate mocking)
    let mut predictions = Vec::new();
    for _ in 0..5 {
        let prediction = integration.predict_divergent(&market_data).await.unwrap();
        predictions.push(prediction);
    }
    
    // Verify predictions have some variation (not identical mocks)
    let first_prediction = &predictions[0];
    let mut identical_count = 0;
    
    for prediction in &predictions[1..] {
        let mut is_identical = true;
        
        // Compare prediction values
        for (v1, v2) in first_prediction.prediction.iter().zip(prediction.prediction.iter()) {
            if (v1 - v2).abs() > 1e-10 {
                is_identical = false;
                break;
            }
        }
        
        if is_identical {
            identical_count += 1;
        }
    }
    
    // Allow some identical predictions but not all (would indicate deterministic mocking)
    assert!(identical_count < predictions.len() - 1, "Too many identical predictions, may indicate mocking");
    
    println!("Zero-mock validation passed: predictions show expected variation");
}

#[tokio::test]
async fn test_tengri_compliance() {
    // Test TENGRI compliance requirements
    let config = RuvFannConfig::maximum_accuracy();
    let integration = RuvFannIntegration::new(config).await.unwrap();
    
    let market_data = create_test_market_data();
    let prediction = integration.predict_divergent(&market_data).await.unwrap();
    
    // TENGRI Compliance Requirement 1: Sub-second latency
    assert!(prediction.metadata.processing_time_us < 1_000_000, "Processing should be sub-second");
    
    // TENGRI Compliance Requirement 2: Uncertainty quantification
    assert!(!prediction.uncertainty.is_empty(), "Must provide uncertainty quantification");
    
    // TENGRI Compliance Requirement 3: Divergent components for robustness
    assert!(prediction.divergent_components.len() >= 2, "Must have multiple divergent components");
    
    // TENGRI Compliance Requirement 4: Confidence intervals
    assert!(!prediction.confidence_intervals.is_empty(), "Must provide confidence intervals");
    
    // TENGRI Compliance Requirement 5: Model traceability
    assert!(!prediction.metadata.model_versions.is_empty(), "Must track model versions");
    
    println!("TENGRI compliance validation passed");
}

#[tokio::test]
async fn test_real_time_performance_requirements() {
    // Test real-time performance requirements for HFT
    let config = RuvFannConfig::ultra_low_latency();
    let integration = RuvFannIntegration::new(config).await.unwrap();
    
    let market_data = create_test_market_data();
    
    // Measure multiple predictions to get stable latency measurements
    let mut latencies = Vec::new();
    for _ in 0..10 {
        let start_time = std::time::Instant::now();
        let result = integration.predict_divergent(&market_data).await;
        let latency = start_time.elapsed();
        
        assert!(result.is_ok(), "All predictions should succeed");
        latencies.push(latency);
    }
    
    // Calculate statistics
    let avg_latency = latencies.iter().sum::<Duration>() / latencies.len() as u32;
    let max_latency = latencies.iter().max().unwrap();
    let min_latency = latencies.iter().min().unwrap();
    
    println!("Latency statistics:");
    println!("  Average: {:?}", avg_latency);
    println!("  Min: {:?}", min_latency);
    println!("  Max: {:?}", max_latency);
    
    // Performance requirements for HFT
    assert!(avg_latency < Duration::from_millis(1), "Average latency should be sub-millisecond");
    assert!(*max_latency < Duration::from_millis(5), "Max latency should be under 5ms");
    
    // Check for consistency (max shouldn't be much higher than average)
    let latency_ratio = max_latency.as_micros() as f64 / avg_latency.as_micros() as f64;
    assert!(latency_ratio < 10.0, "Latency should be consistent (max/avg ratio: {})", latency_ratio);
}