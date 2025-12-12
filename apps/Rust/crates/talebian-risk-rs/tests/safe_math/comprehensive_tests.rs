//! Comprehensive test suite for safe mathematical operations

use talebian_risk_rs::safe_math::*;
use talebian_risk_rs::types::MarketData;
use chrono::Utc;
use std::collections::HashMap;

/// Generate test market data with various edge cases
fn generate_test_market_data_suite() -> Vec<MarketData> {
    vec![
        // Valid normal data
        MarketData {
            timestamp: Utc::now(),
            timestamp_unix: 1000000000,
            price: 100.0,
            volume: 1000.0,
            bid: 99.5,
            ask: 100.5,
            bid_volume: 500.0,
            ask_volume: 500.0,
            volatility: 0.2,
            returns: vec![0.01, -0.02, 0.015, -0.01],
            volume_history: vec![900.0, 1100.0, 950.0, 1050.0],
        },
        // Edge case: very small values
        MarketData {
            timestamp: Utc::now(),
            timestamp_unix: 1000000001,
            price: 0.0001,
            volume: 0.01,
            bid: 0.00009,
            ask: 0.00011,
            bid_volume: 0.005,
            ask_volume: 0.005,
            volatility: 0.5,
            returns: vec![0.1, -0.2, 0.15, -0.1],
            volume_history: vec![0.009, 0.011, 0.0095, 0.0105],
        },
        // Edge case: large values
        MarketData {
            timestamp: Utc::now(),
            timestamp_unix: 1000000002,
            price: 1000000.0,
            volume: 100000.0,
            bid: 999900.0,
            ask: 1000100.0,
            bid_volume: 50000.0,
            ask_volume: 50000.0,
            volatility: 0.1,
            returns: vec![0.001, -0.002, 0.0015, -0.001],
            volume_history: vec![90000.0, 110000.0, 95000.0, 105000.0],
        },
    ]
}

/// Generate test returns with various distributions
fn generate_test_returns_suite() -> Vec<Vec<f64>> {
    vec![
        // Normal distribution-like returns
        vec![
            0.01, -0.02, 0.015, -0.01, 0.03, -0.005, 0.02, -0.015,
            0.025, -0.008, 0.012, -0.018, 0.008, -0.022, 0.035,
            -0.012, 0.018, -0.007, 0.028, -0.014, 0.009, -0.025,
            0.016, -0.011, 0.021, -0.006, 0.013, -0.019, 0.024,
            -0.003, 0.017, -0.013, 0.011, -0.026, 0.019, -0.004,
            0.022, -0.009, 0.014, -0.020, 0.027, -0.002, 0.010,
            -0.016, 0.023, -0.005, 0.015, -0.017, 0.020, -0.008
        ],
        // Fat-tailed distribution with extreme values
        vec![
            0.001, -0.002, 0.0015, -0.001, 0.003, -0.0005, 0.002, -0.0015,
            0.15, -0.12, 0.008, -0.006, 0.004, -0.08, 0.02,  // Extreme values
            -0.003, 0.005, -0.002, 0.007, -0.004, 0.003, -0.006,
            0.004, -0.003, 0.005, -0.002, 0.003, -0.004, 0.006,
            -0.001, 0.004, -0.003, 0.002, -0.005, 0.004, -0.001,
            0.005, -0.002, 0.003, -0.004, 0.006, -0.001, 0.002,
            -0.003, 0.005, -0.002, 0.003, -0.004, 0.004, -0.002
        ],
        // Trending returns (mostly positive)
        vec![
            0.02, 0.015, 0.01, 0.025, 0.018, 0.012, 0.022, 0.016,
            0.020, 0.014, 0.018, 0.011, 0.024, 0.017, 0.013,
            0.019, 0.015, 0.021, 0.012, 0.016, 0.023, 0.014,
            0.017, 0.020, 0.015, 0.018, 0.013, 0.021, 0.016,
            0.019, 0.014, 0.022, 0.017, 0.012, 0.020, 0.015,
            0.018, 0.021, 0.014, 0.016, 0.019, 0.013, 0.017,
            0.020, 0.015, 0.018, 0.012, 0.021, 0.016, 0.014
        ],
        // High volatility with zero mean
        vec![
            0.05, -0.05, 0.04, -0.04, 0.06, -0.06, 0.03, -0.03,
            0.07, -0.07, 0.02, -0.02, 0.08, -0.08, 0.01, -0.01,
            0.09, -0.09, 0.05, -0.05, 0.04, -0.04, 0.06, -0.06,
            0.03, -0.03, 0.07, -0.07, 0.02, -0.02, 0.08, -0.08,
            0.01, -0.01, 0.09, -0.09, 0.05, -0.05, 0.04, -0.04,
            0.06, -0.06, 0.03, -0.03, 0.07, -0.07, 0.02, -0.02,
            0.08, -0.08, 0.01, -0.01, 0.09, -0.09, 0.05, -0.05
        ],
    ]
}

#[cfg(test)]
mod safe_arithmetic_tests {
    use super::*;

    #[test]
    fn test_safe_divide_comprehensive() {
        // Test normal cases
        assert!(safe_divide(10.0, 2.0).is_ok());
        assert_eq!(safe_divide(10.0, 2.0).unwrap(), 5.0);
        
        // Test division by zero
        assert!(safe_divide(10.0, 0.0).is_err());
        assert!(safe_divide(10.0, 1e-20).is_err());
        
        // Test invalid inputs
        assert!(safe_divide(f64::NAN, 2.0).is_err());
        assert!(safe_divide(10.0, f64::NAN).is_err());
        assert!(safe_divide(f64::INFINITY, 2.0).is_err());
        assert!(safe_divide(10.0, f64::INFINITY).is_err());
        
        // Test potential overflow
        assert!(safe_divide(1e20, 1e-20).is_err());
        
        // Test edge cases
        assert!(safe_divide(0.0, 1.0).is_ok());
        assert_eq!(safe_divide(0.0, 1.0).unwrap(), 0.0);
    }

    #[test]
    fn test_safe_multiply_comprehensive() {
        // Test normal cases
        assert!(safe_multiply(10.0, 2.0).is_ok());
        assert_eq!(safe_multiply(10.0, 2.0).unwrap(), 20.0);
        
        // Test overflow detection
        assert!(safe_multiply(1e10, 1e10).is_err());
        assert!(safe_multiply(f64::MAX / 2.0, 3.0).is_err());
        
        // Test invalid inputs
        assert!(safe_multiply(f64::NAN, 2.0).is_err());
        assert!(safe_multiply(10.0, f64::INFINITY).is_err());
        
        // Test edge cases
        assert!(safe_multiply(0.0, 1e10).is_ok());
        assert_eq!(safe_multiply(0.0, 1e10).unwrap(), 0.0);
    }

    #[test]
    fn test_safe_pow_comprehensive() {
        // Test normal cases
        assert!(safe_pow(2.0, 3.0).is_ok());
        assert_eq!(safe_pow(2.0, 3.0).unwrap(), 8.0);
        
        // Test special cases
        assert!(safe_pow(5.0, 0.0).is_ok());
        assert_eq!(safe_pow(5.0, 0.0).unwrap(), 1.0);
        
        assert!(safe_pow(0.0, 2.0).is_ok());
        assert_eq!(safe_pow(0.0, 2.0).unwrap(), 0.0);
        
        // Test invalid cases
        assert!(safe_pow(0.0, -1.0).is_err());
        assert!(safe_pow(f64::NAN, 2.0).is_err());
        
        // Test overflow detection
        assert!(safe_pow(10.0, 100.0).is_err());
    }

    #[test]
    fn test_safe_percentage_calculations() {
        // Test normal percentage
        assert!(safe_percentage(50.0, 200.0).is_ok());
        assert_eq!(safe_percentage(50.0, 200.0).unwrap(), 25.0);
        
        // Test percentage change
        assert!(safe_percentage_change(100.0, 110.0).is_ok());
        assert_eq!(safe_percentage_change(100.0, 110.0).unwrap(), 10.0);
        
        // Test negative percentage change
        assert!(safe_percentage_change(100.0, 90.0).is_ok());
        assert_eq!(safe_percentage_change(100.0, 90.0).unwrap(), -10.0);
        
        // Test division by zero
        assert!(safe_percentage(50.0, 0.0).is_err());
        assert!(safe_percentage_change(0.0, 10.0).is_err());
    }

    #[test]
    fn test_approximately_equal() {
        assert!(approximately_equal(1.0, 1.0 + 1e-16));
        assert!(approximately_equal(1.0, 1.0 - 1e-16));
        assert!(!approximately_equal(1.0, 1.1));
        
        // Test with custom tolerance
        assert!(approximately_equal_with_tolerance(1.0, 1.01, 0.02));
        assert!(!approximately_equal_with_tolerance(1.0, 1.01, 0.005));
    }

    #[test]
    fn test_clamp_to_safe_range() {
        assert_eq!(clamp_to_safe_range(500.0), 500.0);
        assert_eq!(clamp_to_safe_range(f64::NAN), 0.0);
        assert_eq!(clamp_to_safe_range(f64::INFINITY), MAX_SAFE_VALUE);
        assert_eq!(clamp_to_safe_range(-f64::INFINITY), MIN_SAFE_VALUE);
        assert_eq!(clamp_to_safe_range(1e-20), MIN_SAFE_VALUE);
        assert_eq!(clamp_to_safe_range(1e15), MAX_SAFE_VALUE);
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn test_market_data_validation_comprehensive() {
        let test_suite = generate_test_market_data_suite();
        
        // Test valid market data
        for (i, data) in test_suite.iter().enumerate() {
            let result = validate_market_data(data);
            if !result.is_valid {
                println!("Test case {} failed: {:?}", i, result.errors);
            }
            assert!(result.is_valid, "Test case {} should be valid", i);
        }
        
        // Test invalid cases
        let mut invalid_data = test_suite[0].clone();
        
        // Invalid price
        invalid_data.price = -10.0;
        let result = validate_market_data(&invalid_data);
        assert!(!result.is_valid);
        
        // Invalid bid/ask order
        invalid_data.price = 100.0;
        invalid_data.bid = 101.0;
        invalid_data.ask = 100.0;
        let result = validate_market_data(&invalid_data);
        assert!(!result.is_valid);
        
        // Invalid volatility
        invalid_data.bid = 99.0;
        invalid_data.ask = 101.0;
        invalid_data.volatility = f64::NAN;
        let result = validate_market_data(&invalid_data);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_returns_validation_comprehensive() {
        let test_suite = generate_test_returns_suite();
        
        for (i, returns) in test_suite.iter().enumerate() {
            let result = validate_returns_array(returns);
            assert!(result.is_valid, "Returns test case {} should be valid", i);
        }
        
        // Test invalid returns
        let invalid_returns = vec![0.01, f64::NAN, 0.02];
        let result = validate_returns_array(&invalid_returns);
        assert!(!result.is_valid);
        
        let extreme_returns = vec![0.01, -1.5, 0.02]; // > 100% loss
        let result = validate_returns_array(&extreme_returns);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_allocation_validation() {
        // Valid allocation
        let mut allocation = HashMap::new();
        allocation.insert("BTC".to_string(), 0.4);
        allocation.insert("ETH".to_string(), 0.3);
        allocation.insert("ADA".to_string(), 0.3);
        assert!(validate_allocation_weights(&allocation).is_ok());
        
        // Invalid sum
        allocation.insert("DOT".to_string(), 0.1);
        assert!(validate_allocation_weights(&allocation).is_err());
        
        // Negative weight
        allocation.clear();
        allocation.insert("BTC".to_string(), -0.1);
        allocation.insert("ETH".to_string(), 1.1);
        assert!(validate_allocation_weights(&allocation).is_err());
    }

    #[test]
    fn test_trading_config_validation() {
        // Valid configuration
        assert!(validate_trading_config(0.2, 2.0, 0.02, 0.06).is_ok());
        
        // Invalid Kelly fraction (too high)
        assert!(validate_trading_config(0.5, 2.0, 0.02, 0.06).is_err());
        
        // Invalid leverage (too high)
        assert!(validate_trading_config(0.2, 20.0, 0.02, 0.06).is_err());
        
        // Poor risk-reward ratio
        assert!(validate_trading_config(0.2, 2.0, 0.06, 0.02).is_err());
        
        // Invalid stop loss (too wide)
        assert!(validate_trading_config(0.2, 2.0, 0.25, 0.50).is_err());
    }
}

#[cfg(test)]
mod position_sizing_tests {
    use super::*;

    #[test]
    fn test_kelly_calculator_comprehensive() {
        let calculator = SafeKellyCalculator::new(0.25);
        
        // Test various win rates and payoffs
        let test_cases = vec![
            (0.6, 150.0, 100.0, 100),  // Good strategy
            (0.4, 200.0, 100.0, 50),   // High payoff
            (0.8, 110.0, 100.0, 200),  // High win rate
            (0.5, 120.0, 100.0, 75),   // Marginal strategy
        ];
        
        for (win_rate, avg_win, avg_loss, sample_size) in test_cases {
            let result = calculator.calculate_kelly_fraction(win_rate, avg_win, avg_loss, sample_size);
            assert!(result.is_ok(), "Kelly calculation failed for case: {}, {}, {}, {}", 
                   win_rate, avg_win, avg_loss, sample_size);
            
            let kelly = result.unwrap();
            assert!(kelly >= 0.0, "Kelly fraction should be non-negative");
            assert!(kelly <= 0.25, "Kelly fraction should not exceed maximum");
        }
        
        // Test insufficient sample size
        let result = calculator.calculate_kelly_fraction(0.6, 150.0, 100.0, 10);
        assert!(result.is_err());
        
        // Test invalid parameters
        assert!(calculator.calculate_kelly_fraction(1.5, 150.0, 100.0, 50).is_err());
        assert!(calculator.calculate_kelly_fraction(0.6, -150.0, 100.0, 50).is_err());
        assert!(calculator.calculate_kelly_fraction(0.6, 150.0, -100.0, 50).is_err());
    }

    #[test]
    fn test_position_sizer_comprehensive() {
        let sizer = SafePositionSizer::new(0.02, 0.1).unwrap();
        
        let test_cases = vec![
            (100000.0, 100.0, 95.0, 0.6, 150.0, 100.0, 50, 0.8),  // Normal case
            (50000.0, 50.0, 47.5, 0.7, 120.0, 80.0, 100, 0.9),    // Different scale
            (200000.0, 200.0, 190.0, 0.5, 200.0, 150.0, 75, 0.7), // Large position
        ];
        
        for (capital, entry, stop, win_rate, avg_win, avg_loss, sample, confidence) in test_cases {
            let result = sizer.calculate_position_size(
                capital, entry, stop, win_rate, avg_win, avg_loss, sample, confidence
            );
            
            assert!(result.is_ok(), "Position sizing failed for case: {:?}", 
                   (capital, entry, stop, win_rate, avg_win, avg_loss, sample, confidence));
            
            let pos_result = result.unwrap();
            assert!(pos_result.position_size > 0.0, "Position size should be positive");
            assert!(pos_result.position_size <= 0.1, "Position size should not exceed maximum");
            assert!(pos_result.actual_risk <= 0.02, "Risk should not exceed maximum");
        }
        
        // Test invalid cases
        assert!(sizer.calculate_position_size(
            100000.0, 95.0, 100.0, 0.6, 150.0, 100.0, 50, 0.8  // Entry < Stop
        ).is_err());
        
        assert!(sizer.calculate_position_size(
            -100000.0, 100.0, 95.0, 0.6, 150.0, 100.0, 50, 0.8  // Negative capital
        ).is_err());
    }

    #[test]
    fn test_portfolio_sizer_comprehensive() {
        let portfolio_sizer = SafePortfolioSizer::new(0.02, 0.1, 0.8, 0.3).unwrap();
        
        let test_cases = vec![
            (0.1, 0.05, 0.2),   // Low exposure, low correlation
            (0.5, 0.1, 0.8),    // Medium exposure, high correlation
            (0.7, 0.25, 0.5),   // High exposure, medium correlation
        ];
        
        for (total_exp, corr_exp, correlation) in test_cases {
            let result = portfolio_sizer.calculate_portfolio_adjusted_size(
                100000.0, 100.0, 95.0, 0.6, 150.0, 100.0, 50, 0.8,
                total_exp, corr_exp, correlation
            );
            
            assert!(result.is_ok(), "Portfolio sizing failed for case: {:?}", 
                   (total_exp, corr_exp, correlation));
            
            let pos_result = result.unwrap();
            assert!(pos_result.risk_adjusted_size >= 0.0, "Adjusted size should be non-negative");
        }
        
        // Test maximum exposure reached
        let result = portfolio_sizer.calculate_portfolio_adjusted_size(
            100000.0, 100.0, 95.0, 0.6, 150.0, 100.0, 50, 0.8,
            0.8, 0.1, 0.2  // At maximum total exposure
        );
        assert!(result.is_ok());
        let pos_result = result.unwrap();
        assert_eq!(pos_result.risk_adjusted_size, 0.0);
    }
}

#[cfg(test)]
mod financial_metrics_tests {
    use super::*;

    #[test]
    fn test_financial_calculator_comprehensive() {
        let calculator = SafeFinancialCalculator::new(0.02).unwrap();
        let test_suite = generate_test_returns_suite();
        
        for (i, returns) in test_suite.iter().enumerate() {
            let result = calculator.calculate_metrics(returns);
            assert!(result.is_ok(), "Metrics calculation failed for test case {}", i);
            
            let metrics = result.unwrap();
            
            // Validate metrics ranges
            assert!(metrics.volatility >= 0.0, "Volatility should be non-negative");
            assert!(metrics.max_drawdown >= 0.0, "Max drawdown should be non-negative");
            assert!(metrics.max_drawdown <= 1.0, "Max drawdown should not exceed 100%");
            assert!(metrics.win_rate >= 0.0 && metrics.win_rate <= 1.0, "Win rate should be between 0 and 1");
            assert!(metrics.sample_size == returns.len(), "Sample size should match input");
            assert!(metrics.confidence_level >= 0.0 && metrics.confidence_level <= 1.0, "Confidence should be between 0 and 1");
            
            // Check for finite values
            assert!(metrics.sharpe_ratio.is_finite() || metrics.sharpe_ratio == f64::INFINITY, "Sharpe ratio should be finite or infinity");
            assert!(metrics.var_95.is_finite(), "VaR should be finite");
            assert!(metrics.var_99.is_finite(), "VaR should be finite");
            
            println!("Test case {}: Sharpe={:.3}, MaxDD={:.3}, Vol={:.3}, WinRate={:.3}", 
                    i, metrics.sharpe_ratio, metrics.max_drawdown, metrics.volatility, metrics.win_rate);
        }
    }

    #[test]
    fn test_rolling_metrics_calculator() {
        let mut rolling_calc = RollingMetricsCalculator::new(0.02, 50).unwrap();
        let returns = generate_test_returns_suite()[0].clone();
        
        for (i, &ret) in returns.iter().enumerate() {
            let result = rolling_calc.add_return(ret);
            assert!(result.is_ok(), "Failed to add return at index {}", i);
            
            let metrics_opt = result.unwrap();
            if i >= 29 {  // Should have metrics once we have enough data
                assert!(metrics_opt.is_some(), "Should have metrics at index {}", i);
                let metrics = metrics_opt.unwrap();
                assert!(metrics.sample_size <= 50, "Sample size should not exceed window");
            }
        }
        
        // Test invalid return
        let result = rolling_calc.add_return(f64::NAN);
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_case_metrics() {
        let calculator = SafeFinancialCalculator::new(0.02).unwrap();
        
        // All zero returns
        let zero_returns = vec![0.0; 50];
        let result = calculator.calculate_metrics(&zero_returns);
        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert_eq!(metrics.volatility, 0.0);
        assert_eq!(metrics.max_drawdown, 0.0);
        
        // All positive returns
        let positive_returns = vec![0.01; 50];
        let result = calculator.calculate_metrics(&positive_returns);
        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert_eq!(metrics.max_drawdown, 0.0);
        assert_eq!(metrics.win_rate, 1.0);
        
        // All negative returns
        let negative_returns = vec![-0.01; 50];
        let result = calculator.calculate_metrics(&negative_returns);
        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert_eq!(metrics.win_rate, 0.0);
        assert!(metrics.max_drawdown > 0.0);
    }
}

#[cfg(test)]
mod stress_tests {
    use super::*;

    #[test]
    fn test_extreme_values_handling() {
        // Test with very small numbers
        assert!(safe_divide(1e-10, 1e-5).is_ok());
        assert!(safe_multiply(1e-10, 1e-5).is_ok());
        
        // Test with very large numbers (should fail safely)
        assert!(safe_divide(1e15, 1e-10).is_err());
        assert!(safe_multiply(1e15, 1e10).is_err());
        
        // Test boundary conditions
        assert!(safe_divide(MIN_SAFE_VALUE, 2.0).is_ok());
        assert!(safe_divide(MAX_SAFE_VALUE, 2.0).is_ok());
    }

    #[test]
    fn test_performance_under_stress() {
        use std::time::Instant;
        
        let calculator = SafeFinancialCalculator::new(0.02).unwrap();
        let large_returns: Vec<f64> = (0..10000)
            .map(|i| (i as f64 * 0.001).sin() * 0.02)
            .collect();
        
        let start = Instant::now();
        let result = calculator.calculate_metrics(&large_returns);
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        println!("Calculated metrics for 10,000 returns in {:?}", duration);
        assert!(duration.as_millis() < 1000, "Calculation should be fast");
    }

    #[test]
    fn test_memory_usage() {
        let mut rolling_calc = RollingMetricsCalculator::new(0.02, 1000).unwrap();
        
        // Add many returns and ensure memory doesn't grow unbounded
        for i in 0..10000 {
            let ret = (i as f64 * 0.001).sin() * 0.02;
            let result = rolling_calc.add_return(ret);
            assert!(result.is_ok());
        }
        
        // Buffer should be limited to window size
        let current_metrics = rolling_calc.get_current_metrics().unwrap();
        assert!(current_metrics.is_some());
        let metrics = current_metrics.unwrap();
        assert_eq!(metrics.sample_size, 1000);
    }

    #[test]
    fn test_concurrent_safety() {
        use std::sync::Arc;
        use std::thread;
        
        let calculator = Arc::new(SafeFinancialCalculator::new(0.02).unwrap());
        let returns = generate_test_returns_suite()[0].clone();
        let returns_arc = Arc::new(returns);
        
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let calc = calculator.clone();
                let rets = returns_arc.clone();
                thread::spawn(move || {
                    calc.calculate_metrics(&rets)
                })
            })
            .collect();
        
        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result.is_ok());
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_trading_workflow() {
        // Simulate a complete trading workflow
        let sizer = SafePositionSizer::new(0.02, 0.1).unwrap();
        let calculator = SafeFinancialCalculator::new(0.02).unwrap();
        
        // Historical performance data
        let returns = generate_test_returns_suite()[0].clone();
        let metrics = calculator.calculate_metrics(&returns).unwrap();
        
        // Calculate position size based on performance
        let position_result = sizer.calculate_position_size(
            100000.0,      // capital
            100.0,         // entry price
            95.0,          // stop loss
            metrics.win_rate,
            metrics.avg_win,
            metrics.avg_loss,
            metrics.sample_size,
            0.8,           // confidence
        ).unwrap();
        
        assert!(position_result.position_size > 0.0);
        assert!(position_result.actual_risk <= 0.02);
        
        // Validate market data before trading
        let market_data = generate_test_market_data_suite()[0].clone();
        let validation = validate_market_data(&market_data);
        assert!(validation.is_valid);
        
        println!("Trading workflow completed successfully:");
        println!("  Position size: {:.2}%", position_result.position_size * 100.0);
        println!("  Actual risk: {:.2}%", position_result.actual_risk * 100.0);
        println!("  Kelly fraction: {:.3}", position_result.kelly_fraction);
        println!("  Sharpe ratio: {:.3}", metrics.sharpe_ratio);
        println!("  Max drawdown: {:.2}%", metrics.max_drawdown * 100.0);
    }

    #[test]
    fn test_error_recovery_workflow() {
        use error_handling::*;
        
        let mut conservative_calc = FailsafeCalculator::new(ConservativeErrorHandler);
        let mut aggressive_calc = FailsafeCalculator::new(AggressiveErrorHandler);
        
        // Test various error scenarios
        let test_cases = vec![
            (10.0, 0.0),     // Division by zero
            (f64::NAN, 2.0), // Invalid input
            (1e20, 1e20),    // Potential overflow
        ];
        
        for (num, den) in test_cases {
            // Conservative should mostly fail with errors
            let conservative_result = conservative_calc.safe_divide_with_recovery(num, den, "test");
            
            // Aggressive should attempt recovery
            let aggressive_result = aggressive_calc.safe_divide_with_recovery(num, den, "test");
            
            println!("Input: {} / {}", num, den);
            println!("  Conservative: {:?}", conservative_result);
            println!("  Aggressive: {:?}", aggressive_result);
        }
    }
}

#[cfg(test)]
mod benchmark_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn benchmark_safe_arithmetic() {
        let iterations = 1_000_000;
        let test_data: Vec<(f64, f64)> = (0..iterations)
            .map(|i| (i as f64 + 1.0, (i % 100) as f64 + 1.0))
            .collect();
        
        // Benchmark safe division
        let start = Instant::now();
        let mut success_count = 0;
        for (a, b) in &test_data {
            if safe_divide(*a, *b).is_ok() {
                success_count += 1;
            }
        }
        let duration = start.elapsed();
        
        println!("Safe division benchmark:");
        println!("  {} operations in {:?}", iterations, duration);
        println!("  {:.2} ops/ms", iterations as f64 / duration.as_millis() as f64);
        println!("  {} successful operations", success_count);
        
        assert!(duration.as_millis() < 1000, "Should complete within 1 second");
    }

    #[test]
    fn benchmark_position_sizing() {
        let sizer = SafePositionSizer::new(0.02, 0.1).unwrap();
        let iterations = 10_000;
        
        let start = Instant::now();
        let mut success_count = 0;
        for i in 0..iterations {
            let capital = 100000.0 + (i as f64 * 1000.0);
            let entry = 100.0 + (i % 50) as f64;
            let stop = entry * 0.95;
            
            if sizer.calculate_position_size(
                capital, entry, stop, 0.6, 150.0, 100.0, 50, 0.8
            ).is_ok() {
                success_count += 1;
            }
        }
        let duration = start.elapsed();
        
        println!("Position sizing benchmark:");
        println!("  {} calculations in {:?}", iterations, duration);
        println!("  {:.2} ops/ms", iterations as f64 / duration.as_millis() as f64);
        println!("  {} successful calculations", success_count);
        
        assert!(duration.as_millis() < 5000, "Should complete within 5 seconds");
    }

    #[test]
    fn benchmark_financial_metrics() {
        let calculator = SafeFinancialCalculator::new(0.02).unwrap();
        let returns = generate_test_returns_suite()[0].clone();
        let iterations = 1000;
        
        let start = Instant::now();
        let mut success_count = 0;
        for _ in 0..iterations {
            if calculator.calculate_metrics(&returns).is_ok() {
                success_count += 1;
            }
        }
        let duration = start.elapsed();
        
        println!("Financial metrics benchmark:");
        println!("  {} calculations in {:?}", iterations, duration);
        println!("  {:.2} ops/ms", iterations as f64 / duration.as_millis() as f64);
        println!("  {} successful calculations", success_count);
        
        assert!(duration.as_millis() < 2000, "Should complete within 2 seconds");
    }
}