//! Comprehensive tests for Platypus Electroreceptor organism
//! Following TDD methodology with extensive test coverage
//! CQGS Compliance: Zero mocks, real implementations only

use parasitic::organisms::platypus::{
    PlatypusElectroreceptor, SubtleSignalDetector, WeakSignalAmplifier, ElectricalPatternRecognizer,
    ElectroreceptionResult, SignalPattern, ElectricalSignal, SignalType
};
use parasitic::traits::{
    Organism, OrganismMetrics, Adaptive, PerformanceMonitor, MarketData
};
use parasitic::{Result, Error};
use chrono::Utc;
use std::sync::Arc;
use std::thread;

/// Create test market data with specified parameters
fn create_test_market_data(price: f64, volatility: f64, liquidity: f64) -> MarketData {
    MarketData {
        symbol: "BTC_USD".to_string(),
        timestamp: Utc::now(),
        price,
        volume: 1000.0,
        volatility,
        bid: price * 0.999,
        ask: price * 1.001,
        spread_percent: 0.002,
        market_cap: Some(1_000_000_000.0),
        liquidity_score: liquidity,
    }
}

/// Create test electrical signals for pattern testing
fn create_test_electrical_signals(count: usize) -> Vec<ElectricalSignal> {
    let mut signals = Vec::with_capacity(count);
    
    for i in 0..count {
        let frequency = 0.1 + (i as f64) * 0.05; // Varying frequencies
        let amplitude = 0.01 + (i as f64) * 0.002; // Subtle amplitudes
        let phase = (i as f64) * std::f64::consts::PI / 4.0;
        
        signals.push(ElectricalSignal {
            frequency,
            amplitude,
            phase,
            timestamp: chrono::Utc::now().timestamp_millis() as u64 + (i as u64 * 100),
            signal_strength: amplitude * frequency,
            noise_level: amplitude * 0.1, // 10% noise
            signal_type: if amplitude > 0.015 {
                SignalType::Strong
            } else {
                SignalType::Weak
            },
        });
    }
    
    signals
}

#[cfg(test)]
mod platypus_tests {
    use super::*;

    /// Test 1: Basic organism creation and initialization
    #[test]
    fn test_platypus_electroreceptor_creation() {
        let platypus = PlatypusElectroreceptor::new();
        assert!(platypus.is_ok(), "PlatypusElectroreceptor creation should succeed");
        
        let platypus = platypus.unwrap();
        assert_eq!(platypus.name(), "PlatypusElectroreceptor");
        assert_eq!(platypus.organism_type(), "SubtleSignalDetector");
        assert!(platypus.is_active());
        assert_eq!(platypus.get_sensitivity_level(), 0.95); // Should start with high sensitivity
    }

    /// Test 2: Subtle signal detection - core electroreception functionality
    #[test]
    fn test_subtle_signal_detection() {
        let platypus = PlatypusElectroreceptor::new().unwrap();
        let market_data = create_test_market_data(50000.0, 0.05, 0.9); // Calm market
        
        let result = platypus.detect_subtle_signals(&market_data, 0.95);
        assert!(result.is_ok(), "Subtle signal detection should succeed");
        
        let detection_result = result.unwrap();
        
        // Validate detection results
        assert!(!detection_result.detected_signals.is_empty());
        assert!(detection_result.overall_signal_strength > 0.0);
        assert!(detection_result.detection_confidence >= 0.0 && detection_result.detection_confidence <= 1.0);
        assert!(detection_result.processing_time_ns > 0);
        assert!(detection_result.processing_time_ns < 500_000); // Must be sub-millisecond
        
        // Validate individual signals
        for signal in &detection_result.detected_signals {
            assert!(signal.amplitude > 0.0);
            assert!(signal.frequency > 0.0);
            assert!(signal.signal_strength >= 0.0);
        }
    }

    /// Test 3: Weak signal amplification capabilities
    #[test]
    fn test_weak_signal_amplification() {
        let platypus = PlatypusElectroreceptor::new().unwrap();
        
        // Create very weak test signals
        let weak_signals = vec![
            ElectricalSignal {
                frequency: 0.01,
                amplitude: 0.001, // Very weak
                phase: 0.0,
                timestamp: chrono::Utc::now().timestamp_millis() as u64,
                signal_strength: 0.00001,
                noise_level: 0.0001,
                signal_type: SignalType::Weak,
            }
        ];
        
        let amplified_result = platypus.amplify_weak_signals(&weak_signals, 10.0).unwrap();
        
        // Signals should be amplified but not over-amplified
        assert!(!amplified_result.amplified_signals.is_empty());
        
        for amplified in &amplified_result.amplified_signals {
            assert!(amplified.amplitude > weak_signals[0].amplitude);
            assert!(amplified.signal_strength > weak_signals[0].signal_strength);
            // Should maintain frequency
            assert!((amplified.frequency - weak_signals[0].frequency).abs() < 0.001);
        }
        
        assert!(amplified_result.amplification_factor > 1.0);
        assert!(amplified_result.signal_to_noise_ratio > 1.0);
    }

    /// Test 4: Electrical pattern recognition
    #[test]
    fn test_electrical_pattern_recognition() {
        let platypus = PlatypusElectroreceptor::new().unwrap();
        let test_signals = create_test_electrical_signals(20);
        
        let pattern_result = platypus.recognize_electrical_patterns(&test_signals).unwrap();
        
        // Should identify some patterns
        assert!(!pattern_result.identified_patterns.is_empty());
        
        for pattern in &pattern_result.identified_patterns {
            assert!(pattern.confidence_score > 0.0 && pattern.confidence_score <= 1.0);
            assert!(pattern.frequency_range.0 < pattern.frequency_range.1); // Valid range
            assert!(!pattern.signal_indices.is_empty());
            assert!(pattern.pattern_strength > 0.0);
        }
        
        assert!(pattern_result.total_patterns_found > 0);
        assert!(pattern_result.average_pattern_confidence > 0.0);
    }

    /// Test 5: Performance requirements - sub-millisecond detection
    #[test]
    fn test_performance_requirements() {
        let platypus = PlatypusElectroreceptor::new().unwrap();
        let test_data = create_test_market_data(50000.0, 0.02, 0.95);

        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _result = platypus.detect_subtle_signals(&test_data, 0.9).unwrap();
        }
        let duration = start.elapsed();

        // Each detection should be sub-millisecond
        let avg_time_per_detection = duration.as_nanos() as f64 / 1000.0;
        assert!(avg_time_per_detection < 500_000.0, // 0.5ms max
               "Average detection time should be under 0.5ms: actual {:.0}ns", avg_time_per_detection);
        
        println!("Performance: {} detections in {}μs (avg: {:.0}ns per detection)", 
                1000, duration.as_micros(), avg_time_per_detection);
    }

    /// Test 6: Different market conditions and signal types
    #[test]
    fn test_different_market_conditions() {
        let platypus = PlatypusElectroreceptor::new().unwrap();
        
        // Test various market conditions
        let conditions = vec![
            ("Calm", 50000.0, 0.01, 0.95),     // Very calm market
            ("Normal", 50000.0, 0.05, 0.8),   // Normal volatility
            ("Volatile", 50000.0, 0.15, 0.6), // High volatility
            ("Illiquid", 50000.0, 0.03, 0.3), // Low liquidity
            ("Chaotic", 50000.0, 0.3, 0.4),   // High volatility, low liquidity
        ];
        
        for (condition_name, price, volatility, liquidity) in conditions {
            let market_data = create_test_market_data(price, volatility, liquidity);
            let result = platypus.detect_subtle_signals(&market_data, 0.9).unwrap();
            
            println!("{} market: detected {} signals, confidence: {:.3}, strength: {:.6}",
                   condition_name, result.detected_signals.len(), 
                   result.detection_confidence, result.overall_signal_strength);
            
            // Should detect signals in all conditions, but characteristics may vary
            assert!(result.overall_signal_strength >= 0.0);
            assert!(result.detection_confidence >= 0.0);
            
            // Calm markets should show more subtle signals
            if volatility < 0.02 {
                assert!(result.detected_signals.iter().any(|s| s.signal_type == SignalType::Weak),
                       "Calm markets should have weak signals");
            }
        }
    }

    /// Test 7: Sensitivity adjustment and adaptation
    #[test]
    fn test_sensitivity_adjustment() {
        let mut platypus = PlatypusElectroreceptor::new().unwrap();
        let test_data = create_test_market_data(50000.0, 0.05, 0.8);
        
        // Test different sensitivity levels
        let sensitivities = vec![0.99, 0.9, 0.7, 0.5, 0.3];
        
        for sensitivity in sensitivities {
            platypus.set_sensitivity_level(sensitivity).unwrap();
            assert_eq!(platypus.get_sensitivity_level(), sensitivity);
            
            let result = platypus.detect_subtle_signals(&test_data, sensitivity).unwrap();
            
            // Higher sensitivity should generally detect more signals
            println!("Sensitivity {}: detected {} signals", 
                   sensitivity, result.detected_signals.len());
        }
    }

    /// Test 8: Adaptation to market conditions
    #[test]
    fn test_adaptation_to_conditions() {
        let mut platypus = PlatypusElectroreceptor::new().unwrap();
        
        // Adapt to different conditions
        let noisy_market = create_test_market_data(50000.0, 0.2, 0.5); // High volatility
        let quiet_market = create_test_market_data(50000.0, 0.01, 0.9); // Low volatility
        
        let initial_sensitivity = platypus.get_sensitivity_level();
        
        // Adapt to noisy market
        platypus.adapt_to_conditions(&noisy_market, 1000).unwrap();
        let noisy_sensitivity = platypus.get_sensitivity_level();
        
        // Adapt to quiet market
        platypus.adapt_to_conditions(&quiet_market, 2000).unwrap();
        let quiet_sensitivity = platypus.get_sensitivity_level();
        
        println!("Adaptation: initial={:.3}, noisy={:.3}, quiet={:.3}",
                initial_sensitivity, noisy_sensitivity, quiet_sensitivity);
        
        // All sensitivities should be in valid range
        assert!(noisy_sensitivity >= 0.1 && noisy_sensitivity <= 1.0);
        assert!(quiet_sensitivity >= 0.1 && quiet_sensitivity <= 1.0);
        
        // Quiet markets should result in higher sensitivity
        assert!(quiet_sensitivity >= noisy_sensitivity, 
               "Quiet markets should increase sensitivity");
    }

    /// Test 9: Signal pattern classification
    #[test]
    fn test_signal_pattern_classification() {
        let platypus = PlatypusElectroreceptor::new().unwrap();
        
        // Create signals with known patterns
        let periodic_signals = (0..10).map(|i| {
            ElectricalSignal {
                frequency: 0.1, // Same frequency
                amplitude: 0.01 * (1.0 + 0.1 * (i as f64 * 0.5).sin()), // Periodic amplitude
                phase: (i as f64) * std::f64::consts::PI / 5.0,
                timestamp: chrono::Utc::now().timestamp_millis() as u64 + (i as u64 * 200),
                signal_strength: 0.001,
                noise_level: 0.0001,
                signal_type: SignalType::Weak,
            }
        }).collect::<Vec<_>>();
        
        let pattern_result = platypus.recognize_electrical_patterns(&periodic_signals).unwrap();
        
        // Should detect periodic pattern
        assert!(pattern_result.total_patterns_found > 0);
        
        let periodic_pattern = pattern_result.identified_patterns.iter()
            .find(|p| p.pattern_type == "Periodic");
        
        if let Some(pattern) = periodic_pattern {
            assert!(pattern.confidence_score > 0.7, "Should confidently detect periodic pattern");
            assert!(pattern.signal_indices.len() > 3, "Periodic pattern should include multiple signals");
        }
    }

    /// Test 10: Noise filtering and signal clarity
    #[test]
    fn test_noise_filtering() {
        let platypus = PlatypusElectroreceptor::new().unwrap();
        
        // Create signals with varying noise levels
        let noisy_signals = (0..5).map(|i| {
            ElectricalSignal {
                frequency: 0.1,
                amplitude: 0.01,
                phase: 0.0,
                timestamp: chrono::Utc::now().timestamp_millis() as u64 + (i as u64 * 100),
                signal_strength: 0.001,
                noise_level: 0.001 * (i as f64 + 1.0), // Increasing noise
                signal_type: SignalType::Weak,
            }
        }).collect::<Vec<_>>();
        
        let amplified_result = platypus.amplify_weak_signals(&noisy_signals, 5.0).unwrap();
        
        // Signal-to-noise ratio should be improved
        assert!(amplified_result.signal_to_noise_ratio > 1.0);
        assert!(amplified_result.noise_reduction_factor > 1.0);
        
        // Less noisy signals should be amplified more effectively
        for (i, amplified) in amplified_result.amplified_signals.iter().enumerate() {
            if i > 0 {
                // Earlier signals (less noise) should have better amplification
                let prev_signal = &amplified_result.amplified_signals[i - 1];
                if noisy_signals[i - 1].noise_level < noisy_signals[i].noise_level {
                    assert!(prev_signal.signal_strength >= amplified.signal_strength * 0.8);
                }
            }
        }
    }

    /// Test 11: Metrics collection and validation
    #[test]
    fn test_metrics_collection() {
        let platypus = PlatypusElectroreceptor::new().unwrap();
        let test_data = create_test_market_data(50000.0, 0.03, 0.85);

        // Perform operations to generate metrics
        for _ in 0..20 {
            let _result = platypus.detect_subtle_signals(&test_data, 0.9).unwrap();
        }

        let metrics = platypus.get_metrics().unwrap();
        
        assert_eq!(metrics.total_operations, 20);
        assert!(metrics.average_processing_time_ns > 0);
        assert!(metrics.successful_operations <= 20);
        assert!(metrics.accuracy_rate >= 0.0 && metrics.accuracy_rate <= 1.0);
        
        // Validate custom metrics specific to electroreception
        assert!(metrics.custom_metrics.contains_key("sensitivity_level"));
        assert!(metrics.custom_metrics.contains_key("signal_detection_rate"));
        assert!(metrics.custom_metrics.contains_key("pattern_recognition_accuracy"));
        assert!(metrics.custom_metrics.contains_key("average_signal_strength"));
        
        let sensitivity = metrics.custom_metrics["sensitivity_level"];
        assert!(sensitivity >= 0.0 && sensitivity <= 1.0);
        
        println!("Metrics: ops={}, avg_time={}ns, accuracy={:.3}, sensitivity={:.3}",
                metrics.total_operations, metrics.average_processing_time_ns,
                metrics.accuracy_rate, sensitivity);
    }

    /// Test 12: Thread safety and concurrent signal detection
    #[test]
    fn test_thread_safety() {
        let platypus = Arc::new(PlatypusElectroreceptor::new().unwrap());
        let test_data = create_test_market_data(50000.0, 0.05, 0.8);

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let platypus_clone = Arc::clone(&platypus);
                let data_clone = test_data.clone();

                thread::spawn(move || {
                    let sensitivity = 0.8 + (i as f64) * 0.02; // Different sensitivities
                    platypus_clone.detect_subtle_signals(&data_clone, sensitivity)
                })
            })
            .collect();

        let mut successful_detections = 0;
        let mut total_signals_detected = 0;

        for handle in handles {
            match handle.join().unwrap() {
                Ok(result) => {
                    successful_detections += 1;
                    total_signals_detected += result.detected_signals.len();
                }
                Err(e) => {
                    println!("Detection failed: {}", e);
                }
            }
        }

        println!("Thread safety test: {} successful detections, {} total signals",
                successful_detections, total_signals_detected);
        
        assert_eq!(successful_detections, 8, "All concurrent detections should succeed");
        assert!(total_signals_detected > 0, "Should detect some signals across threads");
    }

    /// Test 13: Error handling and edge cases
    #[test]
    fn test_error_handling() {
        let platypus = PlatypusElectroreceptor::new().unwrap();
        
        // Test invalid sensitivity
        let mut platypus_mut = PlatypusElectroreceptor::new().unwrap();
        let result = platypus_mut.set_sensitivity_level(1.5);
        assert!(result.is_err(), "Sensitivity > 1.0 should cause error");
        
        let result = platypus_mut.set_sensitivity_level(-0.1);
        assert!(result.is_err(), "Negative sensitivity should cause error");
        
        // Test invalid market data
        let invalid_market_data = MarketData {
            symbol: "TEST".to_string(),
            timestamp: Utc::now(),
            price: -100.0, // Invalid negative price
            volume: 1000.0,
            volatility: 0.1,
            bid: 50.0,
            ask: 51.0,
            spread_percent: 0.002,
            market_cap: Some(1000000000.0),
            liquidity_score: 0.8,
        };
        
        let result = platypus.detect_subtle_signals(&invalid_market_data, 0.9);
        assert!(result.is_err(), "Negative price should cause error");
        
        // Test empty signals for pattern recognition
        let empty_result = platypus.recognize_electrical_patterns(&[]);
        assert!(empty_result.is_ok(), "Empty signal array should be handled gracefully");
        
        let empty_patterns = empty_result.unwrap();
        assert!(empty_patterns.identified_patterns.is_empty());
        assert_eq!(empty_patterns.total_patterns_found, 0);
    }

    /// Test 14: Signal frequency analysis
    #[test]
    fn test_signal_frequency_analysis() {
        let platypus = PlatypusElectroreceptor::new().unwrap();
        
        // Create signals across different frequency ranges
        let frequency_ranges = vec![
            (0.001, 0.01),  // Ultra-low frequency
            (0.01, 0.1),    // Low frequency  
            (0.1, 1.0),     // Mid frequency
            (1.0, 10.0),    // High frequency
        ];
        
        for (freq_min, freq_max) in frequency_ranges {
            let freq_signals = (0..5).map(|i| {
                let freq = freq_min + (freq_max - freq_min) * (i as f64 / 4.0);
                ElectricalSignal {
                    frequency: freq,
                    amplitude: 0.01,
                    phase: 0.0,
                    timestamp: chrono::Utc::now().timestamp_millis() as u64,
                    signal_strength: 0.01 * freq,
                    noise_level: 0.001,
                    signal_type: if freq > 0.1 { SignalType::Strong } else { SignalType::Weak },
                }
            }).collect::<Vec<_>>();
            
            let pattern_result = platypus.recognize_electrical_patterns(&freq_signals).unwrap();
            
            println!("Frequency range {:.3}-{:.3}: found {} patterns",
                   freq_min, freq_max, pattern_result.total_patterns_found);
            
            // Should be able to analyze any frequency range
            assert!(pattern_result.total_patterns_found >= 0);
        }
    }

    /// Test 15: Performance monitoring compliance
    #[test]
    fn test_performance_monitoring_compliance() {
        let platypus = PlatypusElectroreceptor::new().unwrap();
        let test_data = create_test_market_data(50000.0, 0.03, 0.9);

        // Perform operations to generate performance data
        for _ in 0..100 {
            let _result = platypus.detect_subtle_signals(&test_data, 0.9).unwrap();
        }

        // Check if performance meets requirements
        let meets_requirements = platypus.meets_requirements();
        println!("Performance requirements met: {}", meets_requirements);
        
        let stats = platypus.get_stats().unwrap();
        println!("Performance stats: avg={}ns, max={}ns, throughput={:.2}ops/s",
                stats.avg_processing_time_ns, stats.max_processing_time_ns,
                stats.throughput_ops_per_sec);
        
        // Platypus should meet sub-millisecond performance requirements
        assert!(stats.avg_processing_time_ns < 500_000, 
               "Average processing time should be under 0.5ms");
        assert!(stats.max_processing_time_ns < 1_000_000,
               "Max processing time should be under 1ms");
        assert!(stats.accuracy_rate > 0.8, "Accuracy should be high for subtle signals");
        assert!(stats.throughput_ops_per_sec > 1000.0, "Should process >1000 ops/sec");
    }
}

/// Integration tests for Platypus Electroreceptor
#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Integration test: Full electroreception workflow
    #[test]
    fn test_full_electroreception_workflow() {
        let platypus = PlatypusElectroreceptor::new().unwrap();
        
        // Simulate a sequence of market conditions
        let market_conditions = vec![
            ("Pre-market", 49800.0, 0.005, 0.95),
            ("Market Open", 50000.0, 0.02, 0.85),
            ("Mid-day", 50150.0, 0.01, 0.9),
            ("Volatility Spike", 50300.0, 0.08, 0.7),
            ("Calm Period", 50280.0, 0.003, 0.95),
            ("Market Close", 50250.0, 0.01, 0.8),
        ];
        
        let mut total_signals_detected = 0;
        let mut total_patterns_found = 0;
        let mut detection_history = Vec::new();
        
        for (phase, price, volatility, liquidity) in market_conditions {
            let market_data = create_test_market_data(price, volatility, liquidity);
            
            // Detect signals
            let detection_result = platypus.detect_subtle_signals(&market_data, 0.9).unwrap();
            total_signals_detected += detection_result.detected_signals.len();
            detection_history.push(detection_result.clone());
            
            // Recognize patterns in detected signals
            if !detection_result.detected_signals.is_empty() {
                let pattern_result = platypus.recognize_electrical_patterns(&detection_result.detected_signals).unwrap();
                total_patterns_found += pattern_result.total_patterns_found;
            }
            
            println!("{}: price=${:.0}, vol={:.3}, signals={}, confidence={:.3}",
                   phase, price, volatility, detection_result.detected_signals.len(),
                   detection_result.detection_confidence);
        }
        
        println!("\nIntegration test summary:");
        println!("  Total signals detected: {}", total_signals_detected);
        println!("  Total patterns found: {}", total_patterns_found);
        
        // Should detect signals throughout the day
        assert!(total_signals_detected > 0, "Should detect signals across market phases");
        
        // Final metrics validation
        let final_metrics = platypus.get_metrics().unwrap();
        println!("  Final operations: {}", final_metrics.total_operations);
        println!("  Final accuracy: {:.3}", final_metrics.accuracy_rate);
        
        // Should maintain high performance
        assert!(final_metrics.accuracy_rate > 0.7, "Should maintain high accuracy");
        assert!(final_metrics.average_processing_time_ns < 500_000, "Should stay sub-millisecond");
    }
}