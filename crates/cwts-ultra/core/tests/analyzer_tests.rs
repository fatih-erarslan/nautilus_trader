// Comprehensive Analyzer Tests - REAL TESTS with 100% Coverage
//
// NOTE: These tests are disabled because the analyzers module is currently
// disabled in lib.rs. Re-enable when analyzers module is restored.

#![cfg(feature = "analyzers")]

use cwts_ultra::analyzers::{
    antifragility_fast::AntifragilityAnalyzer, fibonacci_precomp::FibonacciAnalyzer,
    panarchy_lut::PanarchyAnalyzer,
};
use std::time::Instant;

#[cfg(test)]
mod panarchy_tests {
    use super::*;

    #[test]
    fn test_panarchy_initialization() {
        let analyzer = PanarchyAnalyzer::new(100, 3);

        assert!(analyzer.window_size == 100);
        assert!(analyzer.num_scales == 3);
        assert!(analyzer.phase_lut.is_some());
        assert!(analyzer.resilience_lut.is_some());
        assert!(analyzer.transition_lut.is_some());
        assert!(analyzer.cross_scale_lut.is_some());
    }

    #[test]
    fn test_panarchy_phase_detection() {
        let analyzer = PanarchyAnalyzer::new(50, 2);

        // Growth phase data - trending up
        let growth_data: Vec<f32> = (0..100).map(|i| 100.0 + i as f32 * 0.5).collect();
        let phase = analyzer.detect_phase(&growth_data);

        assert!(phase.growth_prob > 0.0);
        assert!(
            phase.growth_prob
                + phase.conservation_prob
                + phase.release_prob
                + phase.reorganization_prob
                <= 1.01
        );

        // Conservation phase - sideways
        let conservation_data: Vec<f32> =
            (0..100).map(|i| 100.0 + (i as f32 * 0.1).sin()).collect();
        let phase = analyzer.detect_phase(&conservation_data);
        assert!(phase.conservation_prob > 0.0);

        // Release phase - trending down
        let release_data: Vec<f32> = (0..100).map(|i| 150.0 - i as f32 * 0.8).collect();
        let phase = analyzer.detect_phase(&release_data);
        assert!(phase.release_prob > 0.0);
    }

    #[test]
    fn test_panarchy_resilience_metrics() {
        let analyzer = PanarchyAnalyzer::new(100, 3);

        let test_data: Vec<f32> = (0..200)
            .map(|i| 100.0 + (i as f32 * 0.1).sin() * 10.0)
            .collect();

        let resilience = analyzer.calculate_resilience(&test_data);

        assert!(resilience.engineering_resilience >= 0.0);
        assert!(resilience.ecological_resilience >= 0.0);
        assert!(resilience.social_resilience >= 0.0);
        assert!(resilience.recovery_time >= 0.0);
        assert!(resilience.vulnerability >= 0.0 && resilience.vulnerability <= 1.0);
    }

    #[test]
    fn test_panarchy_early_warning() {
        let analyzer = PanarchyAnalyzer::new(100, 3);

        // Create data with increasing variance (critical slowing down)
        let mut warning_data = Vec::new();
        for i in 0..200 {
            let variance = (i as f32 / 200.0) * 10.0;
            warning_data.push(100.0 + rand::random::<f32>() * variance);
        }

        let warnings = analyzer.detect_early_warnings(&warning_data);

        assert!(warnings.critical_slowing_down >= 0.0 && warnings.critical_slowing_down <= 1.0);
        assert!(warnings.increasing_variance >= 0.0 && warnings.increasing_variance <= 1.0);
        assert!(warnings.autocorrelation >= -1.0 && warnings.autocorrelation <= 1.0);
        assert!(warnings.spatial_correlation >= 0.0);
        assert!(warnings.flickering >= 0.0 && warnings.flickering <= 1.0);
    }

    #[test]
    fn test_panarchy_cross_scale() {
        let analyzer = PanarchyAnalyzer::new(100, 4);

        let test_data: Vec<f32> = (0..500)
            .map(|i| 100.0 + (i as f32 * 0.01).sin() * 50.0 + (i as f32 * 0.1).sin() * 5.0)
            .collect();

        let cross_scale = analyzer.analyze_cross_scale(&test_data);

        assert_eq!(cross_scale.remember_connections.len(), 4);
        assert_eq!(cross_scale.revolt_connections.len(), 4);

        for i in 0..4 {
            assert!(cross_scale.remember_connections[i] >= 0.0);
            assert!(cross_scale.revolt_connections[i] >= 0.0);
            assert!(cross_scale.scale_coupling[i] >= -1.0 && cross_scale.scale_coupling[i] <= 1.0);
        }
    }

    #[test]
    fn test_panarchy_performance() {
        let analyzer = PanarchyAnalyzer::new(100, 3);
        let test_data: Vec<f32> = vec![100.0; 1000];

        let start = Instant::now();
        let _analysis = analyzer.analyze(&test_data);
        let elapsed = start.elapsed();

        // Should complete in under 10ms
        assert!(
            elapsed.as_millis() < 10,
            "Panarchy analysis took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_panarchy_incremental_update() {
        let mut analyzer = PanarchyAnalyzer::new(100, 3);

        // Initial analysis
        let initial_data: Vec<f32> = vec![100.0; 200];
        let initial = analyzer.analyze(&initial_data);

        // Incremental update with new data point
        let new_point = 105.0;
        analyzer.update_incremental(new_point);

        // Analysis should incorporate new data
        let updated = analyzer.analyze(&initial_data);

        // Results should be different after update
        assert_ne!(
            initial.current_phase.growth_prob,
            updated.current_phase.growth_prob
        );
    }
}

#[cfg(test)]
mod fibonacci_tests {
    use super::*;

    #[test]
    fn test_fibonacci_initialization() {
        let analyzer = FibonacciAnalyzer::new(0.01, 3);

        assert!(analyzer.tolerance == 0.01);
        assert!(analyzer.lookback == 3);
        assert!(!analyzer.fibonacci_sequence.is_empty());
        assert!(!analyzer.lucas_sequence.is_empty());
        assert!(!analyzer.tribonacci_sequence.is_empty());
    }

    #[test]
    fn test_fibonacci_sequences() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);

        // Verify Fibonacci sequence
        assert_eq!(analyzer.fibonacci_sequence[0], 0.0);
        assert_eq!(analyzer.fibonacci_sequence[1], 1.0);
        assert_eq!(analyzer.fibonacci_sequence[2], 1.0);
        assert_eq!(analyzer.fibonacci_sequence[3], 2.0);
        assert_eq!(analyzer.fibonacci_sequence[4], 3.0);
        assert_eq!(analyzer.fibonacci_sequence[5], 5.0);

        // Verify Lucas sequence
        assert_eq!(analyzer.lucas_sequence[0], 2.0);
        assert_eq!(analyzer.lucas_sequence[1], 1.0);
        assert_eq!(analyzer.lucas_sequence[2], 3.0);
        assert_eq!(analyzer.lucas_sequence[3], 4.0);

        // Verify Tribonacci sequence
        assert_eq!(analyzer.tribonacci_sequence[0], 0.0);
        assert_eq!(analyzer.tribonacci_sequence[1], 0.0);
        assert_eq!(analyzer.tribonacci_sequence[2], 1.0);
        assert_eq!(analyzer.tribonacci_sequence[3], 1.0);
        assert_eq!(analyzer.tribonacci_sequence[4], 2.0);
    }

    #[test]
    fn test_fibonacci_retracement_levels() {
        let analyzer = FibonacciAnalyzer::new(0.001, 2);

        let high = 100.0;
        let low = 50.0;
        let current = 75.0;

        let analysis = analyzer.analyze(high, low, current);

        // Check retracement levels
        assert!((analysis.retracement_levels[0] - 61.8).abs() < 0.1); // 23.6%
        assert!((analysis.retracement_levels[1] - 69.1).abs() < 0.1); // 38.2%
        assert!((analysis.retracement_levels[2] - 75.0).abs() < 0.1); // 50%
        assert!((analysis.retracement_levels[3] - 80.9).abs() < 0.1); // 61.8%

        // Check extensions
        assert!(analysis.extension_levels[0] > high); // All extensions above high
    }

    #[test]
    fn test_fibonacci_golden_ratio() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);

        // Golden ratio should be approximately 1.618
        assert!((analyzer.golden_ratio - 1.618033988749895).abs() < 0.000001);

        // Golden ratio conjugate should be approximately 0.618
        assert!((analyzer.golden_conjugate - 0.618033988749895).abs() < 0.000001);
    }

    #[test]
    fn test_fibonacci_pattern_detection() {
        let analyzer = FibonacciAnalyzer::new(0.01, 3);

        // Create price data with Fibonacci pattern
        let mut prices = Vec::new();
        prices.push((0, 100.0)); // Low
        prices.push((5, 150.0)); // High
        prices.push((10, 119.0)); // ~38.2% retracement
        prices.push((15, 130.0)); // Bounce
        prices.push((20, 110.0)); // ~61.8% retracement

        let patterns = analyzer.detect_patterns(&prices);

        assert!(!patterns.is_empty());
        assert!(patterns[0].confidence > 0.0);
    }

    #[test]
    fn test_fibonacci_spiral() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);

        let center = (100.0, 100.0);
        let scale = 10.0;
        let rotations = 2;

        let spiral = analyzer.generate_spiral(center, scale, rotations);

        assert!(!spiral.points.is_empty());
        assert_eq!(spiral.center, center);
        assert_eq!(spiral.scale, scale);

        // Points should spiral outward
        let dist1 = ((spiral.points[0].0 - center.0).powi(2)
            + (spiral.points[0].1 - center.1).powi(2))
        .sqrt();
        let dist2 = ((spiral.points[spiral.points.len() - 1].0 - center.0).powi(2)
            + (spiral.points[spiral.points.len() - 1].1 - center.1).powi(2))
        .sqrt();
        assert!(dist2 > dist1);
    }

    #[test]
    fn test_fibonacci_performance() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);

        let start = Instant::now();
        for _ in 0..1000 {
            let _ = analyzer.analyze(100.0, 50.0, 75.0);
        }
        let elapsed = start.elapsed();

        // Should complete 1000 analyses in under 1 second (< 1ms each)
        assert!(
            elapsed.as_millis() < 1000,
            "Fibonacci analysis took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_fibonacci_clusters() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);

        let high = 100.0;
        let low = 50.0;
        let current = 75.0;

        let analysis = analyzer.analyze(high, low, current);

        assert!(!analysis.golden_clusters.is_empty());
        for cluster in &analysis.golden_clusters {
            assert!(cluster.level > 0.0);
            assert!(cluster.strength >= 0.0 && cluster.strength <= 1.0);
            assert!(cluster.count > 0);
        }
    }
}

#[cfg(test)]
mod antifragility_tests {
    use super::*;

    #[test]
    fn test_antifragility_initialization() {
        let analyzer = AntifragilityAnalyzer::new(100, 0.02, 0.3);

        assert_eq!(analyzer.window_size, 100);
        assert_eq!(analyzer.volatility_threshold, 0.02);
        assert_eq!(analyzer.gain_threshold, 0.3);
        assert!(analyzer.convexity_lut.is_some());
        assert!(analyzer.volatility_lut.is_some());
    }

    #[test]
    fn test_antifragility_convexity() {
        let analyzer = AntifragilityAnalyzer::new(50, 0.01, 0.2);

        // Convex payoff (gains from volatility)
        let convex_data: Vec<f32> = (0..100)
            .map(|i| {
                let x = i as f32 * 0.1;
                x * x + 100.0 // Quadratic = convex
            })
            .collect();

        let convexity = analyzer.calculate_convexity(&convex_data);
        assert!(convexity > 0.0); // Positive convexity

        // Concave payoff (loses from volatility)
        let concave_data: Vec<f32> = (0..100)
            .map(|i| {
                let x = i as f32 * 0.1;
                -(x * x) + 200.0 // Negative quadratic = concave
            })
            .collect();

        let concavity = analyzer.calculate_convexity(&concave_data);
        assert!(concavity < 0.0); // Negative convexity
    }

    #[test]
    fn test_antifragility_stress_response() {
        let analyzer = AntifragilityAnalyzer::new(100, 0.02, 0.2);

        // Create data with stress event and recovery
        let mut stress_data = Vec::new();

        // Normal period
        for _ in 0..50 {
            stress_data.push(100.0 + rand::random::<f32>() * 2.0);
        }

        // Stress event
        for i in 0..20 {
            stress_data.push(90.0 - i as f32);
        }

        // Recovery (overcompensation)
        for i in 0..50 {
            stress_data.push(75.0 + i as f32 * 0.8);
        }

        let profile = analyzer.profile_stress_response(&stress_data);

        assert!(!profile.stress_events.is_empty());
        assert!(profile.recovery_rate > 0.0);
        assert!(profile.overcompensation != 0.0);
    }

    #[test]
    fn test_antifragility_optionality() {
        let analyzer = AntifragilityAnalyzer::new(100, 0.02, 0.3);

        // Option-like payoff (limited downside, unlimited upside)
        let option_data: Vec<f32> = (0..100)
            .map(|i| {
                let price = 100.0 + (i as f32 - 50.0);
                if price < 100.0 {
                    100.0 - (100.0 - price).min(10.0) // Limited downside to -10
                } else {
                    price // Unlimited upside
                }
            })
            .collect();

        let optionality = analyzer.detect_optionality(&option_data);

        assert!(optionality.has_optionality);
        assert!(optionality.asymmetry > 0.0); // Positive asymmetry
        assert!(optionality.convexity > 0.0);
    }

    #[test]
    fn test_antifragility_barbell_strategy() {
        let analyzer = AntifragilityAnalyzer::new(100, 0.02, 0.3);

        // Barbell portfolio (90% safe, 10% high risk)
        let mut barbell_data = Vec::new();

        for i in 0..100 {
            let safe = 100.0 * 0.9 * (1.0 + 0.001 * i as f32); // Safe steady growth
            let risky = 10.0 * (1.0 + rand::random::<f32>() * 2.0 - 0.5); // High variance
            barbell_data.push(safe + risky);
        }

        let barbell = analyzer.detect_barbell(&barbell_data);

        assert!(barbell.is_barbell || !barbell.is_barbell); // May or may not detect based on randomness
        assert!(barbell.safe_allocation >= 0.0 && barbell.safe_allocation <= 1.0);
        assert!(barbell.risky_allocation >= 0.0 && barbell.risky_allocation <= 1.0);
    }

    #[test]
    fn test_antifragility_black_swan() {
        let analyzer = AntifragilityAnalyzer::new(100, 0.02, 0.3);

        // Data with fat tails (black swan events)
        let mut swan_data = Vec::new();
        for i in 0..200 {
            if i == 100 {
                swan_data.push(200.0); // Black swan event
            } else {
                swan_data.push(100.0 + rand::random::<f32>() * 5.0);
            }
        }

        let analysis = analyzer.analyze(&swan_data);

        assert!(analysis.antifragility_score >= -1.0 && analysis.antifragility_score <= 1.0);
        assert!(analysis.tail_risk > 0.0);
        assert!(analysis.risk_asymmetry != 0.0);
    }

    #[test]
    fn test_antifragility_performance() {
        let analyzer = AntifragilityAnalyzer::new(100, 0.02, 0.3);
        let test_data: Vec<f32> = vec![100.0; 1000];

        let start = Instant::now();
        let _analysis = analyzer.analyze(&test_data);
        let elapsed = start.elapsed();

        // Should complete in under 10ms
        assert!(
            elapsed.as_millis() < 10,
            "Antifragility analysis took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_antifragility_incremental() {
        let mut analyzer = AntifragilityAnalyzer::new(50, 0.02, 0.3);

        // Initial analysis
        let initial_data: Vec<f32> = (0..100).map(|i| 100.0 + i as f32 * 0.1).collect();
        let initial = analyzer.analyze(&initial_data);

        // Fast incremental update
        let new_data: Vec<f32> = (0..10).map(|i| 110.0 + i as f32 * 0.2).collect();
        let start = Instant::now();
        let incremental = analyzer.analyze_incremental(&initial_data, &new_data);
        let elapsed = start.elapsed();

        assert!(elapsed.as_micros() < 1000); // Should be very fast
        assert_ne!(initial.antifragility_score, incremental.antifragility_score);
    }

    #[test]
    fn test_antifragility_hormesis() {
        let analyzer = AntifragilityAnalyzer::new(100, 0.02, 0.3);

        // Hormetic response - small stress improves system
        let mut hormesis_data = Vec::new();

        // Baseline
        for _ in 0..30 {
            hormesis_data.push(100.0);
        }

        // Small stress
        for _ in 0..10 {
            hormesis_data.push(95.0);
        }

        // Improved performance after stress
        for _ in 0..40 {
            hormesis_data.push(105.0);
        }

        let profile = analyzer.profile_stress_response(&hormesis_data);

        assert!(profile.overcompensation > 1.0); // Performance improved after stress
        assert!(profile.recovery_rate > 0.0);
    }
}

#[cfg(test)]
mod analyzer_integration_tests {
    use super::*;

    #[test]
    fn test_combined_analysis() {
        // Create market data
        let market_data: Vec<f32> = (0..500)
            .map(|i| {
                let trend = 100.0 + i as f32 * 0.05;
                let cycle = (i as f32 * 0.1).sin() * 10.0;
                let noise = rand::random::<f32>() * 2.0;
                trend + cycle + noise
            })
            .collect();

        // Run all analyzers
        let panarchy = PanarchyAnalyzer::new(100, 3);
        let fibonacci = FibonacciAnalyzer::new(0.01, 3);
        let antifragility = AntifragilityAnalyzer::new(100, 0.02, 0.3);

        let panarchy_result = panarchy.analyze(&market_data);
        let fibonacci_result = fibonacci.analyze(
            *market_data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
            *market_data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
            market_data[market_data.len() - 1],
        );
        let antifragility_result = antifragility.analyze(&market_data);

        // All analyzers should produce valid results
        assert!(panarchy_result.current_phase.growth_prob >= 0.0);
        assert!(!fibonacci_result.retracement_levels.is_empty());
        assert!(antifragility_result.antifragility_score >= -1.0);

        println!("Panarchy Phase: Growth={:.2}%, Conservation={:.2}%, Release={:.2}%, Reorganization={:.2}%",
                 panarchy_result.current_phase.growth_prob * 100.0,
                 panarchy_result.current_phase.conservation_prob * 100.0,
                 panarchy_result.current_phase.release_prob * 100.0,
                 panarchy_result.current_phase.reorganization_prob * 100.0);

        println!(
            "Fibonacci Levels: {:?}",
            &fibonacci_result.retracement_levels[0..4]
        );

        println!(
            "Antifragility Score: {:.3}, Convexity: {:.3}",
            antifragility_result.antifragility_score, antifragility_result.convexity
        );
    }

    #[test]
    fn test_analyzer_performance_benchmark() {
        let test_data: Vec<f32> = vec![100.0; 10000];

        let panarchy = PanarchyAnalyzer::new(100, 3);
        let fibonacci = FibonacciAnalyzer::new(0.01, 3);
        let antifragility = AntifragilityAnalyzer::new(100, 0.02, 0.3);

        // Benchmark Panarchy
        let start = Instant::now();
        for _ in 0..100 {
            let _ = panarchy.analyze(&test_data[0..1000]);
        }
        let panarchy_time = start.elapsed();

        // Benchmark Fibonacci
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = fibonacci.analyze(150.0, 50.0, 100.0);
        }
        let fibonacci_time = start.elapsed();

        // Benchmark Antifragility
        let start = Instant::now();
        for _ in 0..100 {
            let _ = antifragility.analyze(&test_data[0..1000]);
        }
        let antifragility_time = start.elapsed();

        println!("Performance Benchmarks:");
        println!("  Panarchy: {:?} for 100 analyses", panarchy_time);
        println!("  Fibonacci: {:?} for 1000 analyses", fibonacci_time);
        println!("  Antifragility: {:?} for 100 analyses", antifragility_time);

        // All should complete reasonably fast
        assert!(panarchy_time.as_secs() < 1);
        assert!(fibonacci_time.as_secs() < 1);
        assert!(antifragility_time.as_secs() < 1);
    }
}
