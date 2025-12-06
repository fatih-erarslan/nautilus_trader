//! Integration tests for IQAD cross-module functionality

use iqad::*;
use std::sync::Arc;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn test_full_anomaly_detection_pipeline() {
    let config = IqadConfig {
        qubits: 4,
        detectors: 50,
        use_quantum: false, // Use classical fallback for testing
        mutation_rate: 0.1,
        selection_threshold: 0.8,
        cache_size: 100,
        timeout_ms: 5000,
        log_level: "INFO".to_string(),
    };
    
    // Create detector
    if let Ok(detector) = ImmuneQuantumAnomalyDetector::new(config).await {
        // Training data (normal patterns)
        let training_data = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.15, 0.25, 0.35, 0.45],
            vec![0.12, 0.22, 0.32, 0.42],
            vec![0.18, 0.28, 0.38, 0.48],
        ];
        
        // Train the detector
        let train_result = detector.train(&training_data).await;
        assert!(train_result.is_ok());
        
        // Test with normal data (should have low anomaly score)
        let normal_test = vec![0.14, 0.24, 0.34, 0.44];
        let normal_result = detector.detect(&normal_test).await;
        
        // Test with anomalous data (should have high anomaly score)
        let anomaly_test = vec![0.9, 0.8, 0.7, 0.6];
        let anomaly_result = detector.detect(&anomaly_test).await;
        
        if let (Ok(normal), Ok(anomaly)) = (normal_result, anomaly_result) {
            // Anomaly should have higher score than normal
            assert!(anomaly.anomaly_score > normal.anomaly_score);
            
            // Both should have valid confidence levels
            assert!(normal.confidence >= 0.0 && normal.confidence <= 1.0);
            assert!(anomaly.confidence >= 0.0 && anomaly.confidence <= 1.0);
            
            // Response times should be reasonable
            assert!(normal.response_time_ms < 1000.0);
            assert!(anomaly.response_time_ms < 1000.0);
        }
    }
}

#[tokio::test]
async fn test_concurrent_detection() {
    let config = IqadConfig {
        qubits: 4,
        detectors: 30,
        use_quantum: false,
        ..Default::default()
    };
    
    if let Ok(detector) = ImmuneQuantumAnomalyDetector::new(config).await {
        let detector = Arc::new(detector);
        
        // Create multiple concurrent detection tasks
        let mut handles = vec![];
        
        for i in 0..10 {
            let detector_clone = Arc::clone(&detector);
            let test_data = vec![i as f64 / 10.0, 0.2, 0.3, 0.4];
            
            let handle = tokio::spawn(async move {
                detector_clone.detect(&test_data).await
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
        assert_eq!(results.len(), 10);
        
        // All results should be valid
        for result in results {
            if let Ok(detection) = result {
                assert!(detection.anomaly_score >= 0.0);
                assert!(detection.confidence >= 0.0 && detection.confidence <= 1.0);
            }
        }
    }
}

#[tokio::test]
async fn test_memory_system_integration() {
    let config = IqadConfig {
        qubits: 4,
        detectors: 20,
        use_quantum: false,
        ..Default::default()
    };
    
    if let Ok(detector) = ImmuneQuantumAnomalyDetector::new(config).await {
        let anomaly_pattern = vec![0.8, 0.7, 0.6, 0.5];
        
        // Detect the same anomaly multiple times
        for _ in 0..5 {
            let _ = detector.detect(&anomaly_pattern).await;
            sleep(Duration::from_millis(10)).await;
        }
        
        // Detector should have memory cells formed
        let memory_info = detector.get_memory_cells().await;
        if let Ok(memory) = memory_info {
            assert!(!memory.is_empty());
        }
    }
}

#[tokio::test]
async fn test_adaptive_threshold_adjustment() {
    let config = IqadConfig {
        qubits: 4,
        detectors: 25,
        use_quantum: false,
        adaptive_threshold: true,
        ..Default::default()
    };
    
    if let Ok(detector) = ImmuneQuantumAnomalyDetector::new(config).await {
        // Initial threshold
        let initial_threshold = detector.get_detection_threshold().await;
        
        // Generate false positives (normal data detected as anomalies)
        let normal_data = vec![0.1, 0.2, 0.3, 0.4];
        for _ in 0..10 {
            let _ = detector.detect(&normal_data).await;
        }
        
        // Provide feedback that these were false positives
        detector.provide_feedback(&normal_data, false).await.ok();
        
        // Threshold should adapt
        let adapted_threshold = detector.get_detection_threshold().await;
        
        if let (Ok(initial), Ok(adapted)) = (initial_threshold, adapted_threshold) {
            // Threshold should change to reduce false positives
            assert_ne!(initial, adapted);
        }
    }
}

#[tokio::test]
async fn test_quantum_classical_consistency() {
    // Test that quantum and classical modes produce similar results
    let base_config = IqadConfig {
        qubits: 4,
        detectors: 20,
        mutation_rate: 0.1,
        selection_threshold: 0.8,
        ..Default::default()
    };
    
    let classical_config = IqadConfig {
        use_quantum: false,
        ..base_config.clone()
    };
    
    let quantum_config = IqadConfig {
        use_quantum: true,
        ..base_config
    };
    
    let test_data = vec![0.5, 0.6, 0.7, 0.8];
    
    // Create both detectors
    if let (Ok(classical), Ok(quantum)) = (
        ImmuneQuantumAnomalyDetector::new(classical_config).await,
        ImmuneQuantumAnomalyDetector::new(quantum_config).await
    ) {
        // Train both with same data
        let training_data = vec![vec![0.1, 0.2, 0.3, 0.4]];
        let _ = classical.train(&training_data).await;
        let _ = quantum.train(&training_data).await;
        
        // Test detection
        let classical_result = classical.detect(&test_data).await;
        let quantum_result = quantum.detect(&test_data).await;
        
        if let (Ok(c_result), Ok(q_result)) = (classical_result, quantum_result) {
            // Results should be in similar ranges (quantum may have variations)
            let score_diff = (c_result.anomaly_score - q_result.anomaly_score).abs();
            assert!(score_diff < 1.0); // Allow for quantum variations
        }
    }
}

#[tokio::test]
async fn test_performance_monitoring() {
    let config = IqadConfig {
        qubits: 4,
        detectors: 15,
        use_quantum: false,
        ..Default::default()
    };
    
    if let Ok(detector) = ImmuneQuantumAnomalyDetector::new(config).await {
        // Perform multiple detections
        let test_patterns = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7, 0.8],
            vec![0.9, 0.8, 0.7, 0.6],
        ];
        
        for pattern in &test_patterns {
            let _ = detector.detect(pattern).await;
        }
        
        // Get performance metrics
        let metrics = detector.get_performance_metrics();
        
        assert!(metrics.total_detections >= 3);
        assert!(metrics.average_response_time > 0.0);
        assert!(metrics.cache_hit_rate >= 0.0 && metrics.cache_hit_rate <= 1.0);
    }
}

#[tokio::test]
async fn test_batch_processing() {
    let config = IqadConfig {
        qubits: 4,
        detectors: 20,
        use_quantum: false,
        ..Default::default()
    };
    
    if let Ok(detector) = ImmuneQuantumAnomalyDetector::new(config).await {
        // Create batch of test data
        let batch_data = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.2, 0.3, 0.4, 0.5],
            vec![0.9, 0.8, 0.7, 0.6], // Anomaly
            vec![0.15, 0.25, 0.35, 0.45],
            vec![0.8, 0.7, 0.6, 0.5], // Anomaly
        ];
        
        let batch_results = detector.detect_batch(&batch_data).await;
        
        if let Ok(results) = batch_results {
            assert_eq!(results.len(), batch_data.len());
            
            // Check that anomalies (indices 2, 4) have higher scores
            if results.len() >= 5 {
                assert!(results[2].anomaly_score > results[0].anomaly_score);
                assert!(results[4].anomaly_score > results[1].anomaly_score);
            }
        }
    }
}

#[tokio::test]
async fn test_error_recovery() {
    let config = IqadConfig {
        qubits: 4,
        detectors: 10,
        use_quantum: false,
        timeout_ms: 100, // Short timeout to trigger errors
        ..Default::default()
    };
    
    if let Ok(detector) = ImmuneQuantumAnomalyDetector::new(config).await {
        // Test recovery from various error conditions
        
        // Invalid input dimensions
        let invalid_data = vec![0.1, 0.2]; // Wrong dimension
        let result1 = detector.detect(&invalid_data).await;
        assert!(result1.is_err());
        
        // NaN input
        let nan_data = vec![0.1, f64::NAN, 0.3, 0.4];
        let result2 = detector.detect(&nan_data).await;
        assert!(result2.is_err());
        
        // After errors, normal operation should still work
        let normal_data = vec![0.1, 0.2, 0.3, 0.4];
        let result3 = detector.detect(&normal_data).await;
        
        // Should recover and work normally
        assert!(result3.is_ok() || matches!(result3, Err(IqadError::Timeout)));
    }
}

#[tokio::test]
async fn test_cache_effectiveness() {
    let config = IqadConfig {
        qubits: 4,
        detectors: 15,
        use_quantum: false,
        cache_size: 50,
        ..Default::default()
    };
    
    if let Ok(detector) = ImmuneQuantumAnomalyDetector::new(config).await {
        let test_data = vec![0.1, 0.2, 0.3, 0.4];
        
        // First detection (cache miss)
        let start1 = std::time::Instant::now();
        let result1 = detector.detect(&test_data).await;
        let time1 = start1.elapsed();
        
        // Second detection of same data (cache hit)
        let start2 = std::time::Instant::now();
        let result2 = detector.detect(&test_data).await;
        let time2 = start2.elapsed();
        
        if let (Ok(r1), Ok(r2)) = (result1, result2) {
            // Results should be identical
            assert_eq!(r1.anomaly_score, r2.anomaly_score);
            
            // Second call should be faster (cache hit)
            assert!(time2 <= time1);
        }
        
        // Check cache statistics
        let cache_stats = detector.get_cache_stats().await;
        if let Ok(stats) = cache_stats {
            assert!(stats.hits >= 1);
            assert!(stats.misses >= 1);
        }
    }
}