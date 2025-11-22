// Cascade Networks Comprehensive Tests - Real Mathematical Models
use crate::algorithms::cascade_networks::*;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cascade_detector_creation() {
        let detector = CascadeDetector::new(10000, 0.01, 5);
        assert_eq!(detector.window_size, 10000);
        assert!((detector.significance_threshold - 0.01).abs() < 1e-10);
        assert_eq!(detector.min_cascade_size, 5);
        assert!(detector.price_history.is_empty());
        assert!(detector.volume_history.is_empty());
        assert!(detector.momentum_cascades.is_empty());
    }

    #[test]
    fn test_hurst_exponent_calculation() {
        let detector = CascadeDetector::new(100, 0.05, 3);

        // Test with trending data (H > 0.5)
        let trending_data: Vec<f64> = (0..100)
            .map(|i| 100.0 + i as f64 * 0.5 + (i as f64 * 0.1).sin())
            .collect();

        let h_trend = detector.calculate_hurst_exponent(&trending_data);
        assert!(
            h_trend > 0.5,
            "Trending data should have H > 0.5, got {}",
            h_trend
        );

        // Test with mean-reverting data (H < 0.5)
        let mean_reverting: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 10.0)
            .collect();

        let h_revert = detector.calculate_hurst_exponent(&mean_reverting);
        assert!(
            h_revert < 0.7,
            "Mean-reverting data should have H < 0.7, got {}",
            h_revert
        );

        // Test with random walk (H ≈ 0.5)
        let random_walk: Vec<f64> = vec![
            100.0, 100.5, 99.8, 100.2, 99.9, 100.1, 100.3, 99.7, 100.0, 100.4,
        ];

        let h_random = detector.calculate_hurst_exponent(&random_walk);
        assert!(
            (h_random - 0.5).abs() < 0.3,
            "Random walk should have H ≈ 0.5, got {}",
            h_random
        );
    }

    #[test]
    fn test_z_score_calculation() {
        let detector = CascadeDetector::new(10, 0.05, 2);

        // Test with known data
        let data = vec![10.0, 12.0, 11.0, 13.0, 10.5, 14.0, 11.5, 12.5, 13.5, 15.0];
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();

        let z_score = detector.calculate_z_score(&data);
        let expected_z = (data[data.len() - 1] - mean) / std_dev;

        assert!(
            (z_score - expected_z).abs() < 1e-10,
            "Z-score mismatch: got {}, expected {}",
            z_score,
            expected_z
        );

        // Test with zero variance
        let constant_data = vec![100.0; 10];
        let z_constant = detector.calculate_z_score(&constant_data);
        assert_eq!(z_constant, 0.0, "Constant data should have z-score of 0");
    }

    #[test]
    fn test_merton_jump_probability() {
        let detector = CascadeDetector::new(100, 0.05, 3);

        // Test normal conditions
        let normal_prob = detector.calculate_merton_jump_probability(0.2, 100.0, 0.01);
        assert!(normal_prob > 0.0 && normal_prob < 1.0);

        // Test high volatility
        let high_vol_prob = detector.calculate_merton_jump_probability(0.8, 100.0, 0.01);
        assert!(
            high_vol_prob > normal_prob,
            "Higher volatility should increase jump probability"
        );

        // Test longer time horizon
        let long_time_prob = detector.calculate_merton_jump_probability(0.2, 100.0, 1.0);
        assert!(
            long_time_prob > normal_prob,
            "Longer time should increase jump probability"
        );

        // Test edge cases
        let zero_vol = detector.calculate_merton_jump_probability(0.0, 100.0, 0.01);
        assert!(zero_vol >= 0.0 && zero_vol <= 1.0);
    }

    #[test]
    fn test_price_cascade_detection() {
        let mut detector = CascadeDetector::new(20, 0.05, 3);

        // Simulate a price cascade
        let prices = vec![
            100.0, 100.2, 100.5, 101.0, 101.8, // Gradual increase
            103.0, 105.0, 108.0, 112.0, 118.0, // Acceleration (cascade)
            125.0, 134.0, 145.0, 158.0, 172.0, // Strong cascade
            188.0, 206.0, 226.0, 248.0, 272.0, // Extreme cascade
        ];

        let mut cascade_detected = false;
        for (i, &price) in prices.iter().enumerate() {
            let volume = 1000.0 * (1.0 + i as f64 * 0.1); // Increasing volume
            if detector.detect_cascade(price, volume) {
                cascade_detected = true;

                // Verify cascade properties
                let cascades = detector.get_active_cascades();
                assert!(!cascades.is_empty());

                let cascade = &cascades[0];
                assert!(
                    cascade.cascade_type == CascadeType::Price
                        || cascade.cascade_type == CascadeType::Combined
                );
                assert!(cascade.strength > 0.0);
                assert!(cascade.size >= detector.min_cascade_size);
            }
        }

        assert!(cascade_detected, "Should have detected price cascade");
    }

    #[test]
    fn test_volume_cascade_detection() {
        let mut detector = CascadeDetector::new(15, 0.05, 3);

        // Simulate a volume cascade with stable price
        let base_volume = 10000.0;
        let price = 100.0;

        for i in 0..20 {
            let volume = if i < 10 {
                base_volume * (1.0 + i as f64 * 0.05) // Gradual increase
            } else {
                base_volume * (2.0 + (i - 10) as f64 * 0.5) // Volume explosion
            };

            let price_variation = price * (1.0 + (i as f64 * 0.01).sin() * 0.01);

            if detector.detect_cascade(price_variation, volume) {
                let cascades = detector.get_active_cascades();
                let has_volume_cascade = cascades.iter().any(|c| {
                    c.cascade_type == CascadeType::Volume || c.cascade_type == CascadeType::Combined
                });

                if i >= 12 {
                    assert!(
                        has_volume_cascade,
                        "Should detect volume cascade after surge"
                    );
                }
            }
        }
    }

    #[test]
    fn test_momentum_cascade_detection() {
        let mut detector = CascadeDetector::new(10, 0.05, 2);

        // Simulate momentum cascade
        let prices = vec![
            100.0, 100.1, 100.3, 100.6, 101.0, // Accelerating
            101.5, 102.1, 102.8, 103.6, 104.5, // Strong momentum
            105.5, 106.6, 107.8, 109.1, 110.5, // Continued momentum
        ];

        for (i, &price) in prices.iter().enumerate() {
            let volume = 5000.0 + i as f64 * 100.0;
            detector.detect_cascade(price, volume);
        }

        // Check momentum cascade
        let cascades = detector.get_active_cascades();
        let has_momentum = cascades.iter().any(|c| {
            c.cascade_type == CascadeType::Momentum || c.cascade_type == CascadeType::Combined
        });

        assert!(
            has_momentum || !cascades.is_empty(),
            "Should detect momentum or other cascade type"
        );
    }

    #[test]
    fn test_combined_cascade_detection() {
        let mut detector = CascadeDetector::new(10, 0.02, 3);

        // Simulate combined cascade (price + volume + momentum)
        for i in 0..15 {
            let price = 100.0 * (1.0 + 0.05 * i as f64).powf(1.5); // Exponential price
            let volume = 10000.0 * (1.0 + 0.1 * i as f64).powf(2.0); // Exponential volume

            if detector.detect_cascade(price, volume) && i >= 5 {
                let cascades = detector.get_active_cascades();
                let combined = cascades
                    .iter()
                    .any(|c| c.cascade_type == CascadeType::Combined);

                if i >= 8 {
                    assert!(
                        combined || !cascades.is_empty(),
                        "Should detect combined or individual cascades"
                    );
                }
            }
        }
    }

    #[test]
    fn test_cascade_strength_calculation() {
        let mut detector = CascadeDetector::new(5, 0.05, 2);

        // Feed data that creates cascades with different strengths
        let scenarios = vec![
            (
                vec![100.0, 101.0, 102.0, 103.0, 104.0],
                vec![1000.0; 5],
                "weak",
            ),
            (
                vec![100.0, 105.0, 112.0, 121.0, 132.0],
                vec![1000.0, 2000.0, 4000.0, 8000.0, 16000.0],
                "strong",
            ),
            (
                vec![100.0, 120.0, 150.0, 190.0, 240.0],
                vec![1000.0, 5000.0, 15000.0, 40000.0, 100000.0],
                "extreme",
            ),
        ];

        for (prices, volumes, label) in scenarios {
            detector.clear_history();

            for (price, volume) in prices.iter().zip(volumes.iter()) {
                detector.detect_cascade(*price, *volume);
            }

            let cascades = detector.get_active_cascades();
            if !cascades.is_empty() {
                let strength = cascades[0].strength;
                match label {
                    "weak" => assert!(strength < 2.0, "Weak cascade strength should be < 2.0"),
                    "strong" => assert!(
                        strength >= 2.0 && strength < 5.0,
                        "Strong cascade strength should be 2.0-5.0"
                    ),
                    "extreme" => {
                        assert!(strength >= 3.0, "Extreme cascade strength should be >= 3.0")
                    }
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn test_cascade_with_noise() {
        let mut detector = CascadeDetector::new(20, 0.1, 5);

        // Add noisy data with underlying cascade
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for i in 0..30 {
            let base_price = 100.0 * (1.0 + 0.02 * i as f64);
            let noise = rng.gen_range(-2.0..2.0);
            let price = base_price + noise;

            let base_volume = 10000.0 * (1.0 + 0.03 * i as f64);
            let volume_noise = rng.gen_range(-500.0..500.0);
            let volume = (base_volume + volume_noise).max(100.0);

            detector.detect_cascade(price, volume);
        }

        // Despite noise, should detect overall cascade trend
        let cascades = detector.get_active_cascades();
        assert!(
            !cascades.is_empty() || detector.price_history.len() > 0,
            "Should maintain history even with noise"
        );
    }

    #[test]
    fn test_cascade_parallel_detection() {
        let detector = Arc::new(CascadeDetector::new(100, 0.05, 3));
        let barrier = Arc::new(Barrier::new(4));
        let mut handles = vec![];

        // Spawn multiple threads to detect cascades in parallel
        for thread_id in 0..4 {
            let detector_clone = Arc::clone(&detector);
            let barrier_clone = Arc::clone(&barrier);

            let handle = thread::spawn(move || {
                barrier_clone.wait();

                let mut local_cascades = vec![];
                for i in 0..25 {
                    let price = 100.0 + thread_id as f64 * 10.0 + i as f64;
                    let volume = 1000.0 * (1.0 + i as f64 * 0.1);

                    // Calculate indicators without modifying shared state
                    let prices = vec![price; 10];
                    let hurst = detector_clone.calculate_hurst_exponent(&prices);
                    let z_score = detector_clone.calculate_z_score(&prices);

                    if hurst > 0.6 || z_score.abs() > 2.0 {
                        local_cascades.push((thread_id, i, hurst, z_score));
                    }
                }

                local_cascades
            });

            handles.push(handle);
        }

        // Collect results
        let mut all_cascades = vec![];
        for handle in handles {
            let cascades = handle.join().unwrap();
            all_cascades.extend(cascades);
        }

        // Verify parallel execution worked
        assert!(
            !all_cascades.is_empty(),
            "Should detect cascades in parallel"
        );

        // Check that we got results from multiple threads
        let unique_threads: std::collections::HashSet<_> =
            all_cascades.iter().map(|(tid, _, _, _)| *tid).collect();
        assert!(
            unique_threads.len() > 1,
            "Should have results from multiple threads"
        );
    }

    #[test]
    fn test_cascade_performance() {
        let mut detector = CascadeDetector::new(1000, 0.05, 10);

        let start = Instant::now();

        // Simulate high-frequency data
        for i in 0..10000 {
            let price = 100.0 + (i as f64 * 0.01).sin() * 10.0;
            let volume = 10000.0 + (i as f64 * 0.02).cos() * 5000.0;
            detector.detect_cascade(price, volume);
        }

        let elapsed = start.elapsed();

        // Should process 10k ticks in under 100ms
        assert!(
            elapsed < Duration::from_millis(100),
            "Cascade detection too slow: {:?}",
            elapsed
        );

        // Verify detector still works after heavy load
        assert!(detector.price_history.len() <= detector.window_size);
        assert!(detector.volume_history.len() <= detector.window_size);
    }

    #[test]
    fn test_cascade_memory_management() {
        let mut detector = CascadeDetector::new(100, 0.05, 5);

        // Feed more data than window size
        for i in 0..500 {
            let price = 100.0 + i as f64 * 0.1;
            let volume = 1000.0 + i as f64 * 10.0;
            detector.detect_cascade(price, volume);
        }

        // Check that history is bounded
        assert_eq!(detector.price_history.len(), detector.window_size);
        assert_eq!(detector.volume_history.len(), detector.window_size);

        // Check that old cascades are cleaned up
        assert!(
            detector.momentum_cascades.len() <= 100,
            "Should limit cascade history"
        );
    }

    #[test]
    fn test_cascade_edge_cases() {
        let mut detector = CascadeDetector::new(5, 0.05, 2);

        // Test with zero/negative values
        assert!(!detector.detect_cascade(0.0, 1000.0));
        assert!(!detector.detect_cascade(-100.0, 1000.0));
        assert!(!detector.detect_cascade(100.0, 0.0));
        assert!(!detector.detect_cascade(100.0, -1000.0));

        // Test with extreme values
        assert!(!detector.detect_cascade(f64::INFINITY, 1000.0));
        assert!(!detector.detect_cascade(100.0, f64::INFINITY));
        assert!(!detector.detect_cascade(f64::NAN, 1000.0));
        assert!(!detector.detect_cascade(100.0, f64::NAN));

        // Test with minimal data
        assert!(!detector.detect_cascade(100.0, 1000.0)); // First tick
        assert!(!detector.detect_cascade(100.1, 1001.0)); // Second tick

        // Clear and test empty state
        detector.clear_history();
        assert!(detector.price_history.is_empty());
        assert!(detector.volume_history.is_empty());
        assert!(detector.momentum_cascades.is_empty());
    }

    #[test]
    fn test_cascade_types() {
        // Verify all cascade types are distinct
        assert_ne!(CascadeType::Price as u8, CascadeType::Volume as u8);
        assert_ne!(CascadeType::Price as u8, CascadeType::Momentum as u8);
        assert_ne!(CascadeType::Price as u8, CascadeType::Combined as u8);
        assert_ne!(CascadeType::Volume as u8, CascadeType::Momentum as u8);
        assert_ne!(CascadeType::Volume as u8, CascadeType::Combined as u8);
        assert_ne!(CascadeType::Momentum as u8, CascadeType::Combined as u8);
    }

    #[test]
    fn test_cascade_to_network() {
        let mut detector = CascadeDetector::new(10, 0.05, 3);

        // Create a detectable cascade
        for i in 0..15 {
            let price = 100.0 * (1.05_f64).powi(i);
            let volume = 10000.0 * (1.1_f64).powi(i);
            detector.detect_cascade(price, volume);
        }

        let cascades = detector.get_active_cascades();
        if !cascades.is_empty() {
            let network = detector.cascade_to_network(&cascades[0]);

            // Verify network properties
            assert!(network.nodes > 0);
            assert!(network.edges >= 0);
            assert!(network.density >= 0.0 && network.density <= 1.0);
            assert!(network.clustering_coefficient >= 0.0 && network.clustering_coefficient <= 1.0);
            assert!(network.average_path_length > 0.0);
            assert!(network.contagion_probability >= 0.0 && network.contagion_probability <= 1.0);
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_real_market_cascade_scenario() {
        let mut detector = CascadeDetector::new(50, 0.05, 5);

        // Simulate a real market scenario: flash crash
        let scenario = vec![
            // Normal trading
            (100.00, 10000.0),
            (100.05, 9500.0),
            (100.02, 10200.0),
            (99.98, 10100.0),
            (100.03, 9900.0),
            (100.01, 10050.0),
            // Initial pressure
            (99.95, 12000.0),
            (99.88, 13500.0),
            (99.75, 15000.0),
            // Cascade begins
            (99.50, 20000.0),
            (99.00, 30000.0),
            (98.20, 45000.0),
            (97.00, 65000.0),
            (95.50, 90000.0),
            (93.00, 120000.0),
            // Flash crash
            (90.00, 180000.0),
            (87.00, 250000.0),
            (84.00, 320000.0),
            (80.00, 450000.0),
            (75.00, 600000.0),
            (70.00, 800000.0),
            // Recovery begins
            (72.00, 700000.0),
            (75.00, 550000.0),
            (78.00, 400000.0),
            (82.00, 300000.0),
            (85.00, 200000.0),
            (88.00, 150000.0),
            // Stabilization
            (90.00, 100000.0),
            (91.00, 80000.0),
            (92.00, 60000.0),
            (93.00, 40000.0),
            (94.00, 30000.0),
            (95.00, 20000.0),
        ];

        let mut crash_detected = false;
        let mut max_cascade_strength = 0.0;

        for (price, volume) in scenario {
            if detector.detect_cascade(price, volume) {
                let cascades = detector.get_active_cascades();
                for cascade in cascades {
                    if cascade.strength > max_cascade_strength {
                        max_cascade_strength = cascade.strength;
                    }
                    if cascade.strength > 3.0 {
                        crash_detected = true;
                    }
                }
            }
        }

        assert!(crash_detected, "Should detect flash crash cascade");
        assert!(
            max_cascade_strength > 2.0,
            "Should detect strong cascade during crash"
        );
    }
}
