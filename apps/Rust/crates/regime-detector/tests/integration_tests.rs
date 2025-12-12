//! Integration tests for regime detection

use regime_detector::{RegimeDetector, types::*};
use approx::assert_relative_eq;

#[test]
fn test_sub_100ns_latency_requirement() {
    let detector = RegimeDetector::new();
    
    // Test with various data sizes
    let test_cases = [
        (vec![100.0; 10], vec![1000.0; 10]),
        (vec![100.0; 50], vec![1000.0; 50]),
        (vec![100.0; 100], vec![1000.0; 100]),
    ];
    
    for (prices, volumes) in test_cases {
        let start = std::time::Instant::now();
        let result = detector.detect_regime(&prices, &volumes);
        let elapsed = start.elapsed().as_nanos();
        
        println!("Detection latency: {}ns for {} data points", elapsed, prices.len());
        
        // Critical HFT requirement: must be under 100ns
        assert!(elapsed < 100, "Latency {}ns exceeds 100ns requirement", elapsed);
        assert!(result.latency_ns < 100, "Reported latency {}ns exceeds requirement", result.latency_ns);
    }
}

#[test]
fn test_regime_accuracy_trending_bull() {
    let detector = RegimeDetector::new();
    
    // Generate clear bull trend
    let prices: Vec<f32> = (0..100).map(|i| 100.0 + i as f32 * 0.1).collect();
    let volumes: Vec<f32> = vec![1000.0; 100];
    
    let result = detector.detect_regime(&prices, &volumes);
    
    assert_eq!(result.regime, MarketRegime::TrendingBull);
    assert!(result.confidence > 0.7, "Confidence too low: {}", result.confidence);
    assert!(result.features.trend_strength > 0.0);
}

#[test]
fn test_regime_accuracy_trending_bear() {
    let detector = RegimeDetector::new();
    
    // Generate clear bear trend
    let prices: Vec<f32> = (0..100).map(|i| 200.0 - i as f32 * 0.1).collect();
    let volumes: Vec<f32> = vec![1000.0; 100];
    
    let result = detector.detect_regime(&prices, &volumes);
    
    assert_eq!(result.regime, MarketRegime::TrendingBear);
    assert!(result.confidence > 0.7, "Confidence too low: {}", result.confidence);
    assert!(result.features.trend_strength < 0.0);
}

#[test]
fn test_regime_accuracy_ranging() {
    let detector = RegimeDetector::new();
    
    // Generate ranging market (sine wave)
    let prices: Vec<f32> = (0..100)
        .map(|i| 100.0 + (i as f32 * 0.1).sin() * 0.5)
        .collect();
    let volumes: Vec<f32> = vec![1000.0; 100];
    
    let result = detector.detect_regime(&prices, &volumes);
    
    assert!(matches!(result.regime, MarketRegime::Ranging | MarketRegime::LowVolatility));
    assert!(result.features.trend_strength.abs() < 0.1);
}

#[test]
fn test_regime_accuracy_high_volatility() {
    let detector = RegimeDetector::new();
    
    // Generate high volatility market
    let mut prices = vec![100.0];
    for i in 1..100 {
        let change = if i % 2 == 0 { 5.0 } else { -5.0 };
        prices.push(prices[i-1] + change);
    }
    let volumes: Vec<f32> = vec![1000.0; 100];
    
    let result = detector.detect_regime(&prices, &volumes);
    
    assert!(matches!(result.regime, MarketRegime::HighVolatility | MarketRegime::Transition));
    assert!(result.features.volatility > 0.02);
}

#[test]
fn test_cache_effectiveness() {
    let detector = RegimeDetector::new();
    let prices: Vec<f32> = (0..50).map(|i| 100.0 + i as f32 * 0.1).collect();
    let volumes: Vec<f32> = vec![1000.0; 50];
    
    // First call - should compute
    let start1 = std::time::Instant::now();
    let result1 = detector.detect_regime(&prices, &volumes);
    let latency1 = start1.elapsed().as_nanos();
    
    // Second call - should use cache
    let start2 = std::time::Instant::now();
    let result2 = detector.detect_regime(&prices, &volumes);
    let latency2 = start2.elapsed().as_nanos();
    
    // Results should be identical
    assert_eq!(result1.regime, result2.regime);
    assert_relative_eq!(result1.confidence, result2.confidence, epsilon = 1e-6);
    
    // Second call should be faster (cache hit)
    assert!(latency2 < latency1 / 2, "Cache not effective: {}ns vs {}ns", latency2, latency1);
}

#[test]
fn test_streaming_performance() {
    let detector = RegimeDetector::new();
    
    let price_buffer: Vec<f32> = (0..99).map(|i| 100.0 + i as f32 * 0.01).collect();
    let volume_buffer: Vec<f32> = vec![1000.0; 99];
    
    let start = std::time::Instant::now();
    let result = detector.detect_regime_streaming(&price_buffer, &volume_buffer, 101.0, 1000.0);
    let elapsed = start.elapsed().as_nanos();
    
    assert!(elapsed < 500, "Streaming detection too slow: {}ns", elapsed);
    assert!(result.latency_ns < 500);
}

#[test]
fn test_batch_processing_performance() {
    let detector = RegimeDetector::new();
    
    let windows: Vec<(Vec<f32>, Vec<f32>)> = (0..10).map(|i| {
        let prices: Vec<f32> = (0..50).map(|j| 100.0 + (i + j) as f32 * 0.01).collect();
        let volumes: Vec<f32> = vec![1000.0; 50];
        (prices, volumes)
    }).collect();
    
    let window_refs: Vec<(&[f32], &[f32])> = windows.iter()
        .map(|(p, v)| (p.as_slice(), v.as_slice()))
        .collect();
    
    let start = std::time::Instant::now();
    let results = detector.detect_regime_batch(&window_refs);
    let elapsed = start.elapsed().as_nanos();
    
    assert_eq!(results.len(), 10);
    
    // Batch processing should be efficient
    let avg_latency_per_detection = elapsed / 10;
    assert!(avg_latency_per_detection < 1000, "Batch processing too slow: {}ns per detection", avg_latency_per_detection);
}

#[test]
fn test_regime_persistence() {
    let detector = RegimeDetector::new();
    
    // Create long trending data
    let price_history: Vec<f32> = (0..200).map(|i| 100.0 + i as f32 * 0.1).collect();
    let volume_history: Vec<f32> = vec![1000.0; 200];
    
    let current_regime = MarketRegime::TrendingBull;
    let persistence = detector.get_regime_persistence(&price_history, &volume_history, current_regime);
    
    // Should detect persistence in trending data
    assert!(persistence > 0, "No persistence detected in trending data");
}

#[test]
fn test_feature_calculation_accuracy() {
    use regime_detector::simd_ops::*;
    
    let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
    let volumes = vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0];
    
    let features = calculate_features_simd(&prices, &volumes);
    
    // Trend should be positive
    assert!(features.trend_strength > 0.0);
    
    // VWAP ratio should be reasonable
    assert!(features.vwap_ratio > 0.8 && features.vwap_ratio < 1.2);
    
    // RSI should be in valid range
    assert!(features.rsi >= 0.0 && features.rsi <= 100.0);
    
    // Hurst exponent should be in valid range
    assert!(features.hurst_exponent >= 0.0 && features.hurst_exponent <= 1.0);
}

#[test]
fn test_transition_probabilities() {
    let detector = RegimeDetector::new();
    let features = RegimeFeatures {
        trend_strength: 0.8,
        volatility: 0.01,
        autocorrelation: 0.6,
        vwap_ratio: 1.02,
        hurst_exponent: 0.7,
        rsi: 65.0,
        microstructure_noise: 0.1,
        order_flow_imbalance: 0.3,
    };
    
    // Test transition probabilities for bull regime
    let transitions = detector.scorer.get_transition_probabilities(MarketRegime::TrendingBull, &features);
    
    // Should have probabilities for all regimes
    assert!(!transitions.is_empty());
    
    // Probabilities should sum to 1
    let total_prob: f32 = transitions.iter().map(|(_, p)| p).sum();
    assert_relative_eq!(total_prob, 1.0, epsilon = 1e-6);
    
    // Bull regime should have highest continuation probability
    let bull_prob = transitions.iter()
        .find(|(regime, _)| *regime == MarketRegime::TrendingBull)
        .map(|(_, prob)| *prob)
        .unwrap_or(0.0);
    
    assert!(bull_prob > 0.5, "Bull continuation probability too low: {}", bull_prob);
}

#[test]
fn test_confidence_scoring() {
    let detector = RegimeDetector::new();
    
    // Strong bull signal
    let strong_bull_features = RegimeFeatures {
        trend_strength: 1.0,
        volatility: 0.01,
        autocorrelation: 0.8,
        vwap_ratio: 1.05,
        hurst_exponent: 0.8,
        rsi: 70.0,
        microstructure_noise: 0.05,
        order_flow_imbalance: 0.4,
    };
    
    let scores = detector.scorer.calculate_scores(&strong_bull_features);
    
    // Bull should have highest confidence
    let (best_regime, best_confidence) = scores.iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    
    assert_eq!(*best_regime, MarketRegime::TrendingBull);
    assert!(*best_confidence > 0.8, "Confidence too low for strong signal: {}", best_confidence);
}

#[test]
fn test_memory_efficiency() {
    let detector = RegimeDetector::new();
    
    // Test with large dataset
    let large_prices: Vec<f32> = (0..10000).map(|i| 100.0 + i as f32 * 0.001).collect();
    let large_volumes: Vec<f32> = vec![1000.0; 10000];
    
    let start = std::time::Instant::now();
    let result = detector.detect_regime(&large_prices, &large_volumes);
    let elapsed = start.elapsed();
    
    // Should handle large datasets efficiently
    assert!(elapsed.as_millis() < 10, "Large dataset processing too slow: {}ms", elapsed.as_millis());
    assert!(result.latency_ns < 1_000_000, "Latency too high for large dataset: {}ns", result.latency_ns);
}

#[test]
fn test_numerical_stability() {
    let detector = RegimeDetector::new();
    
    // Test with extreme values
    let extreme_prices = vec![1e-6, 1e6, 1e-6, 1e6, 1e-6];
    let extreme_volumes = vec![1e-3, 1e9, 1e-3, 1e9, 1e-3];
    
    let result = detector.detect_regime(&extreme_prices, &extreme_volumes);
    
    // Should not panic or produce NaN values
    assert!(result.confidence.is_finite());
    assert!(result.features.volatility.is_finite());
    assert!(result.features.trend_strength.is_finite());
    assert!(result.features.autocorrelation.is_finite());
}

#[test]
fn test_edge_cases() {
    let detector = RegimeDetector::new();
    
    // Empty data
    let result = detector.detect_regime(&[], &[]);
    assert!(result.confidence == 0.0 || result.confidence.is_nan());
    
    // Single data point
    let result = detector.detect_regime(&[100.0], &[1000.0]);
    assert!(result.latency_ns < 100);
    
    // Two data points
    let result = detector.detect_regime(&[100.0, 101.0], &[1000.0, 1100.0]);
    assert!(result.latency_ns < 100);
    assert!(result.confidence >= 0.0);
}

#[test]
fn test_deterministic_behavior() {
    let detector = RegimeDetector::new();
    let prices: Vec<f32> = (0..50).map(|i| 100.0 + i as f32 * 0.1).collect();
    let volumes: Vec<f32> = vec![1000.0; 50];
    
    // Multiple calls should produce identical results
    let result1 = detector.detect_regime(&prices, &volumes);
    let result2 = detector.detect_regime(&prices, &volumes);
    
    assert_eq!(result1.regime, result2.regime);
    assert_relative_eq!(result1.confidence, result2.confidence, epsilon = 1e-6);
    assert_relative_eq!(result1.features.trend_strength, result2.features.trend_strength, epsilon = 1e-6);
}