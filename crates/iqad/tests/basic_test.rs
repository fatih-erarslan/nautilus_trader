use iqad::{ImmuneQuantumAnomalyDetector, IqadConfig};
use std::collections::HashMap;

#[tokio::test]
async fn test_iqad_creation() {
    let config = IqadConfig::default();
    let detector = ImmuneQuantumAnomalyDetector::new(config).await;
    assert!(detector.is_ok());
}

#[tokio::test]
async fn test_anomaly_detection() {
    let config = IqadConfig {
        quantum_dimension: 4,
        num_detectors: 20,
        ..Default::default()
    };
    
    let detector = ImmuneQuantumAnomalyDetector::new(config).await.unwrap();
    
    // Create test features
    let mut features = HashMap::new();
    features.insert("close".to_string(), 100.0);
    features.insert("volume".to_string(), 1000.0);
    features.insert("volatility".to_string(), 0.2);
    features.insert("rsi_14".to_string(), 50.0);
    
    let result = detector.detect_anomalies(features, None, None).await;
    assert!(result.is_ok());
    
    let anomaly_result = result.unwrap();
    assert!(anomaly_result.score >= 0.0 && anomaly_result.score <= 1.0);
}