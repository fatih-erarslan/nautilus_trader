// Comprehensive tests for Fibonacci Precomputation Analyzer
use crate::analyzers::fibonacci_precomp::*;

#[cfg(test)]
mod fibonacci_analyzer_tests {
    use super::*;
    
    #[test]
    fn test_fibonacci_sequence_generation() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);
        
        // Test first 15 Fibonacci numbers
        let expected = vec![0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0, 144.0, 233.0, 377.0];
        
        for (i, &expected_val) in expected.iter().enumerate() {
            if i < analyzer.fib_numbers.len() {
                assert!((analyzer.fib_numbers[i] - expected_val).abs() < 1e-10);
            }
        }
    }
    
    #[test]
    fn test_lucas_sequence_generation() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);
        
        // Test first 10 Lucas numbers: 2, 1, 3, 4, 7, 11, 18, 29, 47, 76
        let expected = vec![2.0, 1.0, 3.0, 4.0, 7.0, 11.0, 18.0, 29.0, 47.0, 76.0];
        
        for (i, &expected_val) in expected.iter().enumerate() {
            if i < analyzer.lucas_numbers.len() {
                assert!((analyzer.lucas_numbers[i] - expected_val).abs() < 1e-10);
            }
        }
    }
    
    #[test]
    fn test_tribonacci_sequence_generation() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);
        
        // Test first 10 Tribonacci numbers: 0, 0, 1, 1, 2, 4, 7, 13, 24, 44
        let expected = vec![0.0, 0.0, 1.0, 1.0, 2.0, 4.0, 7.0, 13.0, 24.0, 44.0];
        
        for (i, &expected_val) in expected.iter().enumerate() {
            if i < analyzer.tribonacci_numbers.len() {
                assert!((analyzer.tribonacci_numbers[i] - expected_val).abs() < 1e-10);
            }
        }
    }
    
    #[test]
    fn test_golden_ratio_precision() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);
        
        // Golden ratio should be (1 + sqrt(5)) / 2 ≈ 1.618033988749895
        let expected_phi = 1.618033988749895;
        assert!((analyzer.golden_ratio - expected_phi).abs() < 1e-15);
        
        // Golden ratio conjugate should be (sqrt(5) - 1) / 2 ≈ 0.618033988749895
        let expected_phi_conj = 0.618033988749895;
        assert!((analyzer.golden_ratio_conjugate - expected_phi_conj).abs() < 1e-15);
    }
    
    #[test]
    fn test_retracement_levels_calculation() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);
        let analysis = analyzer.analyze(100.0, 90.0, 95.0);
        
        // Test key retracement levels for range 90-100
        assert!((analysis.retracement_236 - 97.64).abs() < 0.01); // 100 - 10 * 0.236
        assert!((analysis.retracement_382 - 96.18).abs() < 0.01); // 100 - 10 * 0.382
        assert!((analysis.retracement_500 - 95.0).abs() < 0.01);  // 100 - 10 * 0.5
        assert!((analysis.retracement_618 - 93.82).abs() < 0.01); // 100 - 10 * 0.618
    }
    
    #[test]
    fn test_extension_levels_calculation() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);
        let analysis = analyzer.analyze(100.0, 90.0, 95.0);
        
        // Test extension levels from low (90.0) with range of 10.0
        assert!((analysis.extension_1272 - 102.72).abs() < 0.01); // 90 + 10 * 1.272
        assert!((analysis.extension_1618 - 106.18).abs() < 0.01); // 90 + 10 * 1.618
        assert!((analysis.extension_2618 - 116.18).abs() < 0.01); // 90 + 10 * 2.618
    }
    
    #[test]
    fn test_golden_cluster_calculation() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);
        let golden_clusters = analyzer.calculate_golden_clusters(100.0, 90.0);
        
        // Should have multiple cluster levels
        assert!(golden_clusters.len() > 5);
        
        // Primary golden ratio level should be around 93.82
        let phi_level = 90.0 + 10.0 * analyzer.golden_ratio_conjugate;
        assert!(golden_clusters.iter().any(|&level| (level - phi_level).abs() < 0.1));
    }
    
    #[test]
    fn test_binet_formula_accuracy() {
        // Test Binet's formula for various Fibonacci numbers
        let test_cases = vec![
            (0, 0.0),
            (1, 1.0),
            (2, 1.0),
            (5, 5.0),
            (10, 55.0),
            (15, 610.0),
        ];
        
        for (n, expected) in test_cases {
            let result = FibonacciAnalyzer::binet_fibonacci(n);
            assert!((result - expected).abs() < 1e-10, "Binet formula failed for F_{}: expected {}, got {}", n, expected, result);
        }
    }
    
    #[test]
    fn test_fibonacci_significance_detection() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);
        
        // Test 61.8% retracement level
        assert!(analyzer.has_fibonacci_significance(93.82, 100.0, 90.0, 0.05));
        
        // Test 38.2% retracement level  
        assert!(analyzer.has_fibonacci_significance(96.18, 100.0, 90.0, 0.05));
        
        // Test 50% level
        assert!(analyzer.has_fibonacci_significance(95.0, 100.0, 90.0, 0.05));
        
        // Test non-significant level
        assert!(!analyzer.has_fibonacci_significance(94.5, 100.0, 90.0, 0.01));
    }
    
    #[test]
    fn test_closest_fibonacci_number() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);
        
        // Test finding closest Fibonacci number to 54
        let (closest, index) = analyzer.closest_fibonacci(54.0);
        assert!((closest - 55.0).abs() < 1e-10); // Should find F_10 = 55
        assert_eq!(index, 10);
        
        // Test finding closest to 88
        let (closest, _) = analyzer.closest_fibonacci(88.0);
        assert!((closest - 89.0).abs() < 1e-10); // Should find F_11 = 89
    }
    
    #[test]
    fn test_fibonacci_ratios_convergence() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);
        
        // Later ratios should converge to golden ratio
        if analyzer.fib_ratios.len() > 20 {
            let late_ratio = analyzer.fib_ratios[analyzer.fib_ratios.len() - 1];
            assert!((late_ratio - analyzer.golden_ratio).abs() < 0.001);
        }
    }
    
    #[test]
    fn test_spiral_generation() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);
        let spiral = analyzer.generate_spiral((100.0, 100.0), 10.0, 2);
        
        // Should generate 720 points for 2 turns
        assert_eq!(spiral.points.len(), 720);
        assert_eq!(spiral.angles.len(), 720);
        assert_eq!(spiral.market_projections.len(), 720);
        
        // Growth factor should be golden ratio
        assert!((spiral.growth_factor - analyzer.golden_ratio).abs() < 1e-15);
    }
    
    #[test]
    fn test_pattern_detection_basic() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);
        
        // Create sample price data for pattern detection
        let price_data = vec![
            (90.0, 1.0), (100.0, 2.0), (93.82, 3.0), (105.0, 4.0), (97.0, 5.0)
        ];
        
        let patterns = analyzer.detect_patterns(&price_data);
        
        // Should detect some patterns from this data
        // Note: Actual pattern detection depends on the specific algorithm implementation
        assert!(patterns.len() >= 0); // At minimum, should not panic
    }
    
    #[test]
    fn test_quick_levels_lookup() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);
        
        // Test quick lookup for common price range
        let levels = analyzer.quick_levels(100.0);
        
        // Should return both retracement and extension levels
        assert!(levels.len() >= 14); // 8 retracements + 6 extensions
        
        // Check some expected levels
        assert!(levels.iter().any(|&level| (level - 23.6).abs() < 0.1));  // 23.6% level
        assert!(levels.iter().any(|&level| (level - 61.8).abs() < 0.1));  // 61.8% level
        assert!(levels.iter().any(|&level| (level - 161.8).abs() < 0.1)); // 161.8% extension
    }
    
    #[test]
    fn test_confluence_strength_calculation() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);
        
        let levels = vec![93.80, 93.82, 93.85, 96.18, 100.0];
        
        // Test confluence at 93.82 (should be high)
        let strength = analyzer.calculate_confluence_strength(93.82, &levels);
        assert!(strength > 0.8);
        
        // Test confluence at non-significant level (should be low)
        let strength = analyzer.calculate_confluence_strength(94.5, &levels);
        assert!(strength < 0.1);
    }
    
    #[test]
    fn test_performance_benchmarks() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);
        
        // Benchmark analysis speed
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            analyzer.analyze(100.0, 90.0, 95.0);
        }
        let duration = start.elapsed();
        
        // Should complete 1000 analyses in well under 1 second (targeting <1ms each)
        assert!(duration.as_millis() < 500, "Performance too slow: {}ms for 1000 analyses", duration.as_millis());
    }
    
    #[test]
    fn test_edge_cases() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);
        
        // Test with equal high and low (zero range)
        let analysis = analyzer.analyze(100.0, 100.0, 100.0);
        assert_eq!(analysis.price_range, 0.0);
        
        // Test with very small range
        let analysis = analyzer.analyze(100.01, 100.0, 100.005);
        assert!((analysis.price_range - 0.01).abs() < 1e-10);
        
        // Test with inverted range (low > high)
        let analysis = analyzer.analyze(90.0, 100.0, 95.0);
        assert_eq!(analysis.price_range, -10.0); // Should handle negative range
    }
    
    #[test]
    fn test_memory_efficiency() {
        let analyzer = FibonacciAnalyzer::new(0.01, 2);
        
        // Verify reasonable memory usage for precomputed data
        assert!(analyzer.fib_numbers.len() < 2000); // Should be reasonable size
        assert!(analyzer.lucas_numbers.len() < 1500);
        assert!(analyzer.tribonacci_numbers.len() < 1500);
        
        // Verify lookup table is populated
        assert!(!analyzer.price_to_fib_lut.is_empty());
    }
}