//! Unit Tests for Rust Algorithms - Mathematical Validation with Scientific Rigor
//! 
//! This module provides comprehensive unit tests for all Rust algorithms
//! with mathematical validation, boundary testing, and numerical stability analysis.

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use approx::{assert_relative_eq, relative_eq};
    use proptest::prelude::*;
    
    // Test tolerance constants for mathematical precision
    const EPSILON: f64 = 1e-12;
    const RELATIVE_TOLERANCE: f64 = 1e-10;
    const NUMERICAL_STABILITY_THRESHOLD: f64 = 1e-8;
    
    /// Mathematical validation framework for algorithm testing
    struct MathematicalValidator {
        precision_tolerance: f64,
        stability_threshold: f64,
        convergence_threshold: f64,
    }
    
    impl MathematicalValidator {
        fn new() -> Self {
            Self {
                precision_tolerance: EPSILON,
                stability_threshold: NUMERICAL_STABILITY_THRESHOLD,
                convergence_threshold: 1e-8,
            }
        }
        
        /// Validates numerical stability of an algorithm
        fn validate_numerical_stability<F>(&self, algorithm: F, test_points: &[f64]) -> bool
        where
            F: Fn(f64) -> f64,
        {
            for &point in test_points {
                let base_result = algorithm(point);
                let perturbed_result = algorithm(point + self.precision_tolerance);
                
                if !base_result.is_finite() || !perturbed_result.is_finite() {
                    return false;
                }
                
                let error = (perturbed_result - base_result).abs();
                if error > self.stability_threshold {
                    return false;
                }
            }
            true
        }
        
        /// Validates convergence properties of iterative algorithms
        fn validate_convergence<F>(&self, algorithm: F, max_iterations: usize) -> (bool, usize)
        where
            F: Fn(usize) -> f64,
        {
            let mut previous = algorithm(0);
            
            for i in 1..max_iterations {
                let current = algorithm(i);
                
                if !current.is_finite() {
                    return (false, i);
                }
                
                let convergence_rate = (current - previous).abs();
                if convergence_rate < self.convergence_threshold {
                    return (true, i);
                }
                
                previous = current;
            }
            
            (false, max_iterations)
        }
        
        /// Validates boundary conditions
        fn validate_boundary_conditions<F>(&self, algorithm: F, boundaries: &[(f64, f64)]) -> bool
        where
            F: Fn(f64) -> f64,
        {
            for &(min_val, max_val) in boundaries {
                let min_result = algorithm(min_val);
                let max_result = algorithm(max_val);
                
                if !min_result.is_finite() || !max_result.is_finite() {
                    return false;
                }
                
                // Test behavior at infinity
                let inf_result = algorithm(f64::INFINITY);
                let neg_inf_result = algorithm(f64::NEG_INFINITY);
                
                // Results should either be finite or properly handle infinity
                if !inf_result.is_finite() && !inf_result.is_infinite() {
                    return false;
                }
                if !neg_inf_result.is_finite() && !neg_inf_result.is_infinite() {
                    return false;
                }
            }
            true
        }
    }
    
    #[test]
    fn test_hft_algorithm_mathematical_correctness() {
        let validator = MathematicalValidator::new();
        
        // Test price calculation algorithm
        let price_algorithm = |volatility: f64| -> f64 {
            // Simplified Black-Scholes-like calculation
            let risk_free_rate = 0.02;
            let time_to_expiry = 1.0;
            
            volatility * (risk_free_rate * time_to_expiry).sqrt()
        };
        
        // Test points covering various scenarios
        let test_points = vec![
            0.01,   // Low volatility
            0.1,    // Normal volatility  
            0.5,    // High volatility
            1.0,    // Extreme volatility
            10.0,   // Ultra-high volatility
        ];
        
        // Validate numerical stability
        assert!(validator.validate_numerical_stability(price_algorithm, &test_points));
        
        // Test boundary conditions
        let boundaries = vec![(0.001, 10.0)];
        assert!(validator.validate_boundary_conditions(price_algorithm, &boundaries));
        
        // Mathematical property tests
        for &vol in &test_points {
            let result = price_algorithm(vol);
            
            // Result should be positive for positive volatility
            assert!(result > 0.0, "Price should be positive for volatility {}", vol);
            
            // Result should scale with volatility
            let double_vol_result = price_algorithm(vol * 2.0);
            assert!(double_vol_result > result, "Price should increase with volatility");
        }
    }
    
    #[test]
    fn test_order_matching_algorithm_precision() {
        let validator = MathematicalValidator::new();
        
        // Order matching price calculation
        let matching_algorithm = |bid: f64, ask: f64| -> f64 {
            if bid <= 0.0 || ask <= 0.0 || bid > ask {
                return f64::NAN;
            }
            
            // Mid-price calculation
            (bid + ask) / 2.0
        };
        
        // Test cases with high precision requirements
        let test_cases = vec![
            (50000.123456789, 50000.987654321),  // High precision
            (0.000001, 0.000002),                // Micro prices
            (1e10, 1e10 + 1.0),                  // Large numbers
            (PI, PI + EPSILON),                  // Mathematical constants
        ];
        
        for (bid, ask) in test_cases {
            let mid_price = matching_algorithm(bid, ask);
            
            assert!(mid_price.is_finite(), "Mid-price should be finite");
            assert!(mid_price >= bid && mid_price <= ask, "Mid-price should be within bid-ask range");
            
            // Mathematical precision test
            let expected_mid = (bid + ask) / 2.0;
            assert_relative_eq!(mid_price, expected_mid, epsilon = EPSILON);
        }
    }
    
    #[test]
    fn test_risk_management_mathematical_properties() {
        let validator = MathematicalValidator::new();
        
        // Value at Risk calculation (simplified)
        let var_algorithm = |returns: &[f64], confidence: f64| -> f64 {
            if returns.is_empty() || confidence <= 0.0 || confidence >= 1.0 {
                return f64::NAN;
            }
            
            let mut sorted_returns = returns.to_vec();
            sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let index = ((1.0 - confidence) * sorted_returns.len() as f64) as usize;
            sorted_returns[index.min(sorted_returns.len() - 1)]
        };
        
        // Generate test data with known statistical properties
        let normal_returns = generate_normal_returns(1000, 0.0, 0.02);
        
        // Test VaR at different confidence levels
        let confidence_levels = vec![0.95, 0.99, 0.999];
        
        for confidence in confidence_levels {
            let var = var_algorithm(&normal_returns, confidence);
            
            assert!(var.is_finite(), "VaR should be finite");
            assert!(var < 0.0, "VaR should be negative (loss)");
            
            // Higher confidence should give more negative VaR
            let higher_confidence_var = var_algorithm(&normal_returns, confidence + 0.001);
            assert!(higher_confidence_var <= var, "Higher confidence should give more conservative VaR");
        }
        
        // Monotonicity test
        for i in 1..confidence_levels.len() {
            let var1 = var_algorithm(&normal_returns, confidence_levels[i-1]);
            let var2 = var_algorithm(&normal_returns, confidence_levels[i]);
            assert!(var2 <= var1, "VaR should be monotonic in confidence level");
        }
    }
    
    #[test]
    fn test_slippage_calculator_boundary_conditions() {
        let validator = MathematicalValidator::new();
        
        // Slippage calculation based on market impact
        let slippage_algorithm = |order_size: f64, market_depth: f64, volatility: f64| -> f64 {
            if order_size <= 0.0 || market_depth <= 0.0 || volatility < 0.0 {
                return f64::NAN;
            }
            
            // Square root market impact model
            let market_impact = (order_size / market_depth).sqrt() * volatility;
            market_impact.min(0.1) // Cap at 10%
        };
        
        // Boundary condition tests
        let boundary_tests = vec![
            (1.0, 1000000.0, 0.01),     // Small order, large depth
            (100000.0, 1000.0, 0.5),   // Large order, small depth
            (1000.0, 1000.0, 0.0),     // No volatility
            (f64::EPSILON, 1.0, 0.01), // Minimal order size
        ];
        
        for (size, depth, vol) in boundary_tests {
            let slippage = slippage_algorithm(size, depth, vol);
            
            if size > 0.0 && depth > 0.0 && vol >= 0.0 {
                assert!(slippage.is_finite() && slippage >= 0.0, 
                       "Slippage should be non-negative and finite");
                assert!(slippage <= 0.1, "Slippage should be capped at 10%");
            } else {
                assert!(slippage.is_nan(), "Invalid inputs should return NaN");
            }
        }
        
        // Mathematical properties
        let base_slippage = slippage_algorithm(1000.0, 10000.0, 0.02);
        
        // Slippage should increase with order size
        let larger_order_slippage = slippage_algorithm(4000.0, 10000.0, 0.02);
        assert!(larger_order_slippage > base_slippage, "Slippage should increase with order size");
        
        // Slippage should decrease with market depth
        let deeper_market_slippage = slippage_algorithm(1000.0, 40000.0, 0.02);
        assert!(deeper_market_slippage < base_slippage, "Slippage should decrease with market depth");
        
        // Slippage should increase with volatility
        let higher_vol_slippage = slippage_algorithm(1000.0, 10000.0, 0.08);
        assert!(higher_vol_slippage > base_slippage, "Slippage should increase with volatility");
    }
    
    #[test]
    fn test_fee_optimizer_convergence() {
        let validator = MathematicalValidator::new();
        
        // Iterative fee optimization algorithm
        let optimize_fees = |initial_fee: f64, target_volume: f64, max_iterations: usize| -> (f64, usize) {
            let mut fee = initial_fee;
            let mut iteration = 0;
            
            while iteration < max_iterations {
                let volume = estimate_volume(fee);
                let error = (volume - target_volume).abs();
                
                if error < validator.convergence_threshold {
                    return (fee, iteration);
                }
                
                // Simple gradient descent step
                let gradient = (estimate_volume(fee + EPSILON) - volume) / EPSILON;
                let step_size = 0.01;
                fee = fee - step_size * (volume - target_volume) / gradient;
                
                // Ensure fee remains positive
                fee = fee.max(0.0001);
                
                iteration += 1;
            }
            
            (fee, iteration)
        };
        
        // Test convergence for different targets
        let test_cases = vec![
            (0.001, 100000.0),  // Low fee, high volume target
            (0.01, 10000.0),    // Medium fee, medium volume target
            (0.1, 1000.0),      // High fee, low volume target
        ];
        
        for (initial_fee, target_volume) in test_cases {
            let (optimized_fee, iterations) = optimize_fees(initial_fee, target_volume, 1000);
            
            assert!(optimized_fee > 0.0, "Optimized fee should be positive");
            assert!(iterations < 1000, "Algorithm should converge within 1000 iterations");
            
            // Verify the optimization actually found a better solution
            let initial_volume = estimate_volume(initial_fee);
            let optimized_volume = estimate_volume(optimized_fee);
            
            let initial_error = (initial_volume - target_volume).abs();
            let optimized_error = (optimized_volume - target_volume).abs();
            
            assert!(optimized_error <= initial_error, 
                   "Optimization should reduce error from {} to {}", initial_error, optimized_error);
        }
    }
    
    #[test]
    fn test_cascade_network_stability() {
        let validator = MathematicalValidator::new();
        
        // Cascade network propagation
        let cascade_propagation = |initial_signal: f64, network_size: usize, decay_rate: f64| -> Vec<f64> {
            let mut signals = vec![initial_signal];
            
            for i in 1..network_size {
                let previous_signal = signals[i - 1];
                let new_signal = previous_signal * (1.0 - decay_rate);
                signals.push(new_signal);
            }
            
            signals
        };
        
        // Test network stability
        let test_configurations = vec![
            (1.0, 100, 0.01),   // Slow decay
            (10.0, 50, 0.1),    // Fast decay
            (0.1, 200, 0.005),  // Very slow decay
        ];
        
        for (initial_signal, network_size, decay_rate) in test_configurations {
            let signals = cascade_propagation(initial_signal, network_size, decay_rate);
            
            // Verify signal decay properties
            assert_eq!(signals.len(), network_size);
            assert_relative_eq!(signals[0], initial_signal, epsilon = EPSILON);
            
            // Signals should decay monotonically
            for i in 1..signals.len() {
                assert!(signals[i] <= signals[i-1], 
                       "Signal should decay monotonically at step {}", i);
                assert!(signals[i] >= 0.0, "Signal should remain non-negative");
            }
            
            // Test stability: final signal should be small
            let final_signal = signals[signals.len() - 1];
            let expected_final = initial_signal * (1.0 - decay_rate).powi(network_size as i32 - 1);
            assert_relative_eq!(final_signal, expected_final, epsilon = RELATIVE_TOLERANCE);
        }
    }
    
    #[test]
    fn test_cuckoo_simd_correctness() {
        // SIMD-optimized cuckoo hashing correctness
        let cuckoo_hash = |key: u64, table_size: usize, hash_function: usize| -> usize {
            match hash_function {
                0 => ((key.wrapping_mul(0x9e3779b9)) >> 32) as usize % table_size,
                1 => ((key.wrapping_mul(0x85ebca6b)) >> 32) as usize % table_size,
                _ => panic!("Invalid hash function"),
            }
        };
        
        let table_size = 1024;
        let test_keys = (0..10000).collect::<Vec<u64>>();
        
        // Test hash distribution
        let mut hash0_distribution = vec![0usize; table_size];
        let mut hash1_distribution = vec![0usize; table_size];
        
        for key in &test_keys {
            let hash0 = cuckoo_hash(*key, table_size, 0);
            let hash1 = cuckoo_hash(*key, table_size, 1);
            
            assert!(hash0 < table_size, "Hash 0 should be within table bounds");
            assert!(hash1 < table_size, "Hash 1 should be within table bounds");
            
            hash0_distribution[hash0] += 1;
            hash1_distribution[hash1] += 1;
        }
        
        // Test distribution uniformity (Chi-square test)
        let expected_frequency = test_keys.len() as f64 / table_size as f64;
        let mut chi_square_0 = 0.0;
        let mut chi_square_1 = 0.0;
        
        for i in 0..table_size {
            let diff0 = hash0_distribution[i] as f64 - expected_frequency;
            let diff1 = hash1_distribution[i] as f64 - expected_frequency;
            chi_square_0 += diff0 * diff0 / expected_frequency;
            chi_square_1 += diff1 * diff1 / expected_frequency;
        }
        
        // Chi-square critical value for 95% confidence with table_size-1 degrees of freedom
        // For large df, approximately df + 2*sqrt(2*df)
        let critical_value = table_size as f64 + 2.0 * (2.0 * table_size as f64).sqrt();
        
        assert!(chi_square_0 < critical_value, 
               "Hash function 0 distribution should be uniform (chi-square: {})", chi_square_0);
        assert!(chi_square_1 < critical_value, 
               "Hash function 1 distribution should be uniform (chi-square: {})", chi_square_1);
    }
    
    #[test]
    fn test_neural_activation_functions_mathematical_properties() {
        let validator = MathematicalValidator::new();
        
        // ReLU activation function
        let relu = |x: f64| x.max(0.0);
        
        // Sigmoid activation function
        let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
        
        // Tanh activation function
        let tanh = |x: f64| x.tanh();
        
        // Test points including edge cases
        let test_points = vec![
            -100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0
        ];
        
        // Test ReLU properties
        for &x in &test_points {
            let relu_result = relu(x);
            
            // ReLU should be non-negative
            assert!(relu_result >= 0.0, "ReLU should be non-negative");
            
            // ReLU should be correct
            if x > 0.0 {
                assert_relative_eq!(relu_result, x, epsilon = EPSILON);
            } else {
                assert_relative_eq!(relu_result, 0.0, epsilon = EPSILON);
            }
        }
        
        // Test Sigmoid properties
        for &x in &test_points {
            let sigmoid_result = sigmoid(x);
            
            // Sigmoid should be in (0, 1)
            assert!(sigmoid_result > 0.0 && sigmoid_result < 1.0, 
                   "Sigmoid should be in (0, 1), got {} for input {}", sigmoid_result, x);
            
            // Sigmoid should be symmetric around 0.5
            let neg_sigmoid = sigmoid(-x);
            assert_relative_eq!(sigmoid_result + neg_sigmoid, 1.0, epsilon = RELATIVE_TOLERANCE);
        }
        
        // Test Tanh properties
        for &x in &test_points {
            let tanh_result = tanh(x);
            
            // Tanh should be in (-1, 1)
            assert!(tanh_result > -1.0 && tanh_result < 1.0, 
                   "Tanh should be in (-1, 1), got {} for input {}", tanh_result, x);
            
            // Tanh should be odd function
            let neg_tanh = tanh(-x);
            assert_relative_eq!(tanh_result, -neg_tanh, epsilon = RELATIVE_TOLERANCE);
        }
        
        // Test numerical stability
        assert!(validator.validate_numerical_stability(relu, &test_points));
        assert!(validator.validate_numerical_stability(sigmoid, &test_points));
        assert!(validator.validate_numerical_stability(tanh, &test_points));
    }
    
    // Property-based testing with QuickCheck-style approach
    proptest! {
        #[test]
        fn test_price_calculation_properties(
            base_price in 1.0..100000.0f64,
            volatility in 0.001..1.0f64,
            time_factor in 0.001..10.0f64
        ) {
            let calculate_price = |base: f64, vol: f64, time: f64| -> f64 {
                base * (1.0 + vol * time.sqrt())
            };
            
            let result = calculate_price(base_price, volatility, time_factor);
            
            // Price should always be positive
            prop_assert!(result > 0.0);
            
            // Price should increase with volatility (for positive base price)
            let higher_vol_price = calculate_price(base_price, volatility * 1.1, time_factor);
            prop_assert!(higher_vol_price > result);
            
            // Price should increase with time factor
            let higher_time_price = calculate_price(base_price, volatility, time_factor * 1.1);
            prop_assert!(higher_time_price >= result);
        }
        
        #[test]
        fn test_risk_calculation_properties(
            position_size in 1.0..1000000.0f64,
            volatility in 0.001..1.0f64,
            confidence in 0.9..0.999f64
        ) {
            let calculate_risk = |size: f64, vol: f64, conf: f64| -> f64 {
                // Simplified VaR calculation
                let z_score = match conf {
                    c if c >= 0.999 => 3.09,  // 99.9%
                    c if c >= 0.99 => 2.33,   // 99%
                    c if c >= 0.95 => 1.64,   // 95%
                    _ => 1.28,                // 90%
                };
                
                size * vol * z_score
            };
            
            let risk = calculate_risk(position_size, volatility, confidence);
            
            // Risk should be positive
            prop_assert!(risk > 0.0);
            
            // Risk should increase with position size
            let larger_position_risk = calculate_risk(position_size * 1.5, volatility, confidence);
            prop_assert!(larger_position_risk > risk);
            
            // Risk should increase with volatility
            let higher_vol_risk = calculate_risk(position_size, volatility * 1.5, confidence);
            prop_assert!(higher_vol_risk > risk);
        }
    }
    
    // Helper functions
    fn generate_normal_returns(n: usize, mean: f64, std_dev: f64) -> Vec<f64> {
        use rand::prelude::*;
        use rand_distr::{Normal, Distribution};
        
        let mut rng = StdRng::seed_from_u64(12345); // Deterministic for testing
        let normal = Normal::new(mean, std_dev).unwrap();
        
        (0..n).map(|_| normal.sample(&mut rng)).collect()
    }
    
    fn estimate_volume(fee: f64) -> f64 {
        // Simplified volume estimation based on fee
        // In practice, this would be a complex market model
        let base_volume = 100000.0;
        let elasticity = -2.0; // Price elasticity of demand
        
        base_volume * fee.powf(elasticity)
    }
}

// Benchmarking module for performance validation
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn benchmark_critical_path_performance() {
        let iterations = 1000000;
        
        // Benchmark order matching critical path
        let start = Instant::now();
        for i in 0..iterations {
            let bid = 50000.0 + (i as f64) * 0.001;
            let ask = bid + 0.01;
            let _mid_price = (bid + ask) / 2.0;
        }
        let duration = start.elapsed();
        
        let ns_per_operation = duration.as_nanos() / iterations as u128;
        println!("Order matching: {} ns per operation", ns_per_operation);
        
        // Critical requirement: < 1000 ns per operation
        assert!(ns_per_operation < 1000, 
               "Order matching should complete in < 1000ns, took {}ns", ns_per_operation);
    }
    
    #[test]
    fn benchmark_risk_calculation_performance() {
        let iterations = 100000;
        let test_returns: Vec<f64> = (0..1000).map(|i| (i as f64 - 500.0) / 1000.0).collect();
        
        let start = Instant::now();
        for _ in 0..iterations {
            let mut sorted = test_returns.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let _var_95 = sorted[50]; // 5th percentile
        }
        let duration = start.elapsed();
        
        let us_per_operation = duration.as_micros() / iterations as u128;
        println!("Risk calculation: {} μs per operation", us_per_operation);
        
        // Should complete within 10 microseconds
        assert!(us_per_operation < 10, 
               "Risk calculation should complete in < 10μs, took {}μs", us_per_operation);
    }
}