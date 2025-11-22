//! Comprehensive Mathematical Validation Test Suite
//!
//! This test suite implements rigorous mathematical validation for the entire
//! CWTS Ultra trading system with:
//! - IEEE 754 floating-point compliance testing
//! - Autopoiesis theory validation  
//! - Peer-reviewed algorithm verification
//! - Regulatory compliance testing
//! - Scientific statistical analysis

use std::collections::HashMap;
use tokio;

// Import the core components for testing
use cwts_ultra_core::{
    ScientificallyRigorousSystem, SystemState, MarketData, TradingSignal,
    MathematicalValidationFramework, FinancialCalculator, AutopoieticSystem,
    ArithmeticError
};

use cwts_ultra_core::neural_models::{
    ScientificActivation, ActivationValidator, ScientificSigmoid, 
    ScientificReLU, QuantumActivation, scientifically_rigorous_sigmoid
};

/// Comprehensive mathematical validation test suite
#[tokio::test]
async fn test_complete_mathematical_validation_framework() {
    let mut framework = MathematicalValidationFramework::new()
        .expect("Failed to initialize mathematical validation framework");
    
    // Run full system validation
    let validation_report = framework.validate_full_system().await
        .expect("Full system validation failed");
    
    // Assert mathematical rigor requirements
    assert!(validation_report.ieee754_compliant, "System must be IEEE 754 compliant");
    assert!(validation_report.autopoiesis_active, "Autopoietic organization must be active");
    assert!(validation_report.mathematical_proofs_valid, "All mathematical proofs must be valid");
    assert!(validation_report.performance_meets_standards, "Performance must meet regulatory standards");
    assert!(validation_report.overall_compliance, "Overall system compliance must be achieved");
    
    // Detailed validations
    assert!(validation_report.detailed_results.ieee754_results.precision_maintained, 
        "IEEE 754 precision must be maintained");
    assert!(validation_report.detailed_results.autopoiesis_results.identity_preserved,
        "Autopoietic identity must be preserved");
    assert!(validation_report.detailed_results.mathematical_results.convergence_proven,
        "Mathematical convergence must be proven");
    
    println!("✅ Complete mathematical validation framework test passed");
}

/// Test IEEE 754 compliant financial calculations
#[tokio::test]
async fn test_ieee754_financial_calculations() {
    let calculator = FinancialCalculator::new()
        .expect("Failed to create financial calculator");
    
    // Test compound interest calculation
    let compound_result = calculator.compound_interest(10000.0, 0.05, 12.0, 5.0)
        .expect("Compound interest calculation failed");
    
    assert!(compound_result > 10000.0, "Compound interest should increase principal");
    assert!(compound_result.is_finite(), "Result must be finite");
    
    // Test Black-Scholes option pricing
    let option_price = calculator.black_scholes_call(100.0, 105.0, 0.25, 0.05, 0.2)
        .expect("Black-Scholes calculation failed");
    
    assert!(option_price > 0.0, "Option price must be positive");
    assert!(option_price < 100.0, "Option price must be reasonable");
    assert!(option_price.is_finite(), "Option price must be finite");
    
    // Test Kahan summation for numerical stability
    let test_values = vec![0.1; 1000000]; // Many small values that might cause precision issues
    let kahan_sum = calculator.kahan_sum(&test_values)
        .expect("Kahan summation failed");
    
    let expected_sum = 0.1 * 1000000.0;
    let difference = (kahan_sum - expected_sum).abs();
    assert!(difference < 1e-10, "Kahan summation must maintain precision: diff = {}", difference);
    
    println!("✅ IEEE 754 financial calculations test passed");
}

/// Test autopoietic system self-organization
#[tokio::test]
async fn test_autopoietic_self_organization() {
    let mut autopoietic_system = AutopoieticSystem::new()
        .expect("Failed to create autopoietic system");
    
    // Initialize the system
    autopoietic_system.initialize().await
        .expect("Failed to initialize autopoietic system");
    
    // Test identity preservation
    let identity_preserved = autopoietic_system.validate_identity_preservation().await
        .expect("Identity validation failed");
    assert!(identity_preserved, "System identity must be preserved");
    
    // Test boundary maintenance
    let boundary_maintained = autopoietic_system.validate_boundary_maintenance().await
        .expect("Boundary validation failed");
    assert!(boundary_maintained, "System boundaries must be maintained");
    
    // Test self-maintenance capabilities
    let self_maintaining = autopoietic_system.validate_self_maintenance().await
        .expect("Self-maintenance validation failed");
    assert!(self_maintaining, "System must be self-maintaining");
    
    // Test structural coupling with environment
    let structurally_coupled = autopoietic_system.validate_structural_coupling().await
        .expect("Structural coupling validation failed");
    assert!(structurally_coupled, "System must be structurally coupled with environment");
    
    // Test adaptation mechanisms
    let adaptation_functioning = autopoietic_system.test_adaptation_mechanisms().await
        .expect("Adaptation testing failed");
    assert!(adaptation_functioning, "Adaptation mechanisms must be functioning");
    
    println!("✅ Autopoietic self-organization test passed");
}

/// Test scientifically rigorous activation functions
#[tokio::test]
async fn test_scientific_activation_functions() {
    // Test Scientific Sigmoid
    let sigmoid = ScientificSigmoid;
    
    // Test normal inputs
    let sigmoid_result = sigmoid.activate(0.0).expect("Sigmoid activation failed");
    assert!((sigmoid_result - 0.5).abs() < 1e-10, "Sigmoid(0) should be 0.5");
    
    let sigmoid_large = sigmoid.activate(100.0).expect("Sigmoid large input failed");
    assert!((sigmoid_large - 1.0).abs() < 1e-10, "Sigmoid(large) should approach 1.0");
    
    let sigmoid_small = sigmoid.activate(-100.0).expect("Sigmoid small input failed");
    assert!(sigmoid_small < 1e-10, "Sigmoid(very negative) should approach 0.0");
    
    // Test error handling for invalid inputs
    assert!(sigmoid.activate(f64::NAN).is_err(), "Sigmoid should reject NaN");
    assert!(sigmoid.activate(f64::INFINITY).is_err(), "Sigmoid should reject infinity");
    
    // Test Scientific ReLU
    let relu = ScientificReLU;
    assert_eq!(relu.activate(5.0).unwrap(), 5.0, "ReLU(positive) = input");
    assert_eq!(relu.activate(-3.0).unwrap(), 0.0, "ReLU(negative) = 0");
    assert_eq!(relu.activate(0.0).unwrap(), 0.0, "ReLU(0) = 0");
    
    // Test Quantum-inspired activation
    let quantum = QuantumActivation::new(1.0, 1.0).expect("Failed to create quantum activation");
    let quantum_result = quantum.activate(0.0).expect("Quantum activation failed");
    assert!(quantum_result.is_finite(), "Quantum activation must produce finite results");
    
    // Validate activation properties
    let sigmoid_properties = sigmoid.validate_properties().expect("Property validation failed");
    assert!(sigmoid_properties.bounded, "Sigmoid should be bounded");
    assert!(sigmoid_properties.monotonic, "Sigmoid should be monotonic");
    assert!(sigmoid_properties.smooth, "Sigmoid should be smooth");
    
    let relu_properties = relu.validate_properties().expect("ReLU property validation failed");
    assert!(!relu_properties.bounded, "ReLU should be unbounded");
    assert!(relu_properties.monotonic, "ReLU should be monotonic");
    assert!(!relu_properties.smooth, "ReLU should not be smooth at x=0");
    
    println!("✅ Scientific activation functions test passed");
}

/// Test activation function validation across ranges
#[tokio::test]
async fn test_activation_function_validation() {
    let sigmoid = ScientificSigmoid;
    
    // Validate sigmoid across a wide range
    let validation_report = ActivationValidator::validate_function(
        &sigmoid, 
        (-1000.0, 1000.0), 
        10000
    ).expect("Activation validation failed");
    
    assert!(validation_report.ieee754_compliant, "Activation function must be IEEE 754 compliant");
    assert!(validation_report.errors.is_empty(), "No errors should occur during validation");
    assert!(validation_report.numerical_issues.is_empty(), "No numerical issues should occur");
    assert_eq!(validation_report.samples_tested, 10000, "All samples should be tested");
    
    println!("✅ Activation function validation test passed");
}

/// Test quantum-inspired neural computation
#[tokio::test] 
async fn test_quantum_neural_computation() {
    // Test different quantum activation configurations
    let quantum_configs = vec![
        (1.0, 1.0),   // Standard
        (2.0, 0.5),   // High frequency, low amplitude  
        (0.5, 2.0),   // Low frequency, high amplitude
    ];
    
    for (frequency, amplitude) in quantum_configs {
        let quantum = QuantumActivation::new(frequency, amplitude)
            .expect(&format!("Failed to create quantum activation ({}, {})", frequency, amplitude));
        
        // Test various inputs
        let test_inputs = vec![-10.0, -1.0, 0.0, 1.0, 10.0];
        
        for input in test_inputs {
            let result = quantum.activate(input)
                .expect(&format!("Quantum activation failed for input {}", input));
            
            assert!(result.is_finite(), "Quantum result must be finite for input {}", input);
            assert!(result.abs() <= amplitude * 1.1, "Result should be bounded by amplitude");
            
            // Test derivative as well
            let derivative = quantum.derivative(input)
                .expect(&format!("Quantum derivative failed for input {}", input));
            
            assert!(derivative.is_finite(), "Quantum derivative must be finite for input {}", input);
        }
        
        // Test properties
        let properties = quantum.validate_properties()
            .expect("Quantum properties validation failed");
        
        assert!(properties.bounded, "Quantum activation should be bounded");
        assert!(!properties.monotonic, "Quantum activation should be oscillatory");
        assert!(properties.smooth, "Quantum activation should be smooth");
        assert!(properties.zero_centered, "Quantum activation should be zero-centered");
    }
    
    println!("✅ Quantum neural computation test passed");
}

/// Test complete scientifically rigorous system integration
#[tokio::test]
async fn test_scientifically_rigorous_system_integration() {
    let mut rigorous_system = ScientificallyRigorousSystem::new().await
        .expect("Failed to create scientifically rigorous system");
    
    // Run full system validation
    let validation_report = rigorous_system.validate_full_system().await
        .expect("Full system validation failed");
    
    assert!(validation_report.overall_compliance, "System must achieve overall compliance");
    
    // Test trading decision with mathematical rigor
    let market_data = MarketData {
        symbol: "BTC/USD".to_string(),
        prices: vec![50000.0, 50100.0, 49900.0, 50200.0, 50050.0],
        volumes: vec![1.0, 1.2, 0.8, 1.5, 1.1],
        timestamp: std::time::SystemTime::now(),
    };
    
    let trading_signal = TradingSignal {
        strength: 0.3,           // Moderate buy signal
        confidence: 0.8,         // High confidence
        expected_return: 0.02,   // 2% expected return
        volatility: 0.15,        // 15% volatility
        win_probability: 0.6,    // 60% win rate
        average_win: 0.05,       // 5% average win
        average_loss: 0.03,      // 3% average loss
    };
    
    let trading_decision = rigorous_system.execute_rigorous_trading_decision(
        &market_data, 
        &trading_signal
    ).await.expect("Trading decision failed");
    
    assert!(trading_decision.confidence > 0.0, "Decision confidence must be positive");
    assert!(trading_decision.risk_score >= 0.0, "Risk score must be non-negative");
    assert!(!trading_decision.mathematical_proof.is_empty(), "Mathematical proof must be provided");
    
    // Test system state monitoring
    let system_state = rigorous_system.get_system_state().await;
    assert!(system_state.ieee754_compliance, "IEEE 754 compliance must be maintained");
    
    println!("✅ Scientifically rigorous system integration test passed");
}

/// Test statistical properties of trading algorithms
#[tokio::test]
async fn test_statistical_properties_validation() {
    let mut validation_framework = MathematicalValidationFramework::new()
        .expect("Failed to create validation framework");
    
    // Test statistical properties
    let stats_results = validation_framework.validate_statistical_properties().await
        .expect("Statistical validation failed");
    
    // Normality test
    assert!(stats_results.normalityTest.pValue > 0.01, 
        "p-value should indicate reasonable normality assumption");
    
    // Stationarity test  
    assert!(stats_results.stationarityTest.passed,
        "Time series should pass stationarity tests");
    
    // Autocorrelation test
    assert!(stats_results.autocorrelationTest.passed,
        "Should pass autocorrelation tests");
    
    // Distribution fit analysis
    assert!(stats_results.distributionFit.goodnessOfFit > 0.8,
        "Distribution fit should be reasonable");
    
    println!("✅ Statistical properties validation test passed");
}

/// Test hypothesis validation with statistical rigor
#[tokio::test]
async fn test_hypothesis_validation() {
    let mut validation_framework = MathematicalValidationFramework::new()
        .expect("Failed to create validation framework");
    
    let hypothesis_results = validation_framework.validate_hypotheses().await
        .expect("Hypothesis validation failed");
    
    // Statistical power should be adequate
    assert!(hypothesis_results.statisticalPower >= 0.8, 
        "Statistical power must be at least 80%");
    
    // Effect size should be meaningful
    assert!(hypothesis_results.effectSize > 0.2,
        "Effect size should be at least small-to-medium");
    
    // All p-values should be significant
    assert!(hypothesis_results.pValues.iter().all(|&p| p < 0.05),
        "All p-values should be statistically significant");
    
    // Confidence intervals should be meaningful
    assert!(hypothesis_results.confidenceIntervals.len() > 0,
        "Confidence intervals should be provided");
    
    for ci in &hypothesis_results.confidenceIntervals {
        assert!(ci.lower < ci.upper, "Confidence interval bounds should be valid");
        assert!(ci.upper - ci.lower > 0.0, "Confidence intervals should have positive width");
    }
    
    println!("✅ Hypothesis validation test passed");
}

/// Test numerical stability across extreme conditions
#[tokio::test] 
async fn test_numerical_stability() {
    let mut validation_framework = MathematicalValidationFramework::new()
        .expect("Failed to create validation framework");
    
    let stability_results = validation_framework.validate_numerical_stability().await
        .expect("Numerical stability validation failed");
    
    assert!(stability_results.stable, "System must be numerically stable");
    assert!(stability_results.convergenceRate > 0.99, 
        "Convergence rate must be at least 99%");
    
    // Check specific stability metrics
    let metrics = &stability_results.stabilityMetrics;
    assert!(metrics.precisionLoss < 1e-12, 
        "Precision loss must be within IEEE 754 double precision limits");
    assert!(metrics.convergenceTime > 0.0,
        "Convergence time must be measurable");
    assert!(!metrics.oscillationDetected,
        "No oscillations should be detected in stable algorithms");
    assert!(metrics.numericalErrors.is_empty(),
        "No numerical errors should occur: {:?}", metrics.numericalErrors);
    
    println!("✅ Numerical stability test passed");
}

/// Performance benchmark for mathematical operations
#[tokio::test]
async fn test_performance_benchmarks() {
    let calculator = FinancialCalculator::new()
        .expect("Failed to create calculator");
    
    let start_time = std::time::Instant::now();
    
    // Perform many calculations to test performance
    for i in 0..10000 {
        let principal = 1000.0 + (i as f64);
        let rate = 0.05 + (i as f64 * 0.000001);
        
        let _result = calculator.compound_interest(principal, rate, 12.0, 1.0)
            .expect("Performance test calculation failed");
    }
    
    let duration = start_time.elapsed();
    
    // Performance should be reasonable (< 1 second for 10k calculations)
    assert!(duration.as_secs() < 1, "Performance test should complete within 1 second");
    
    println!("✅ Performance benchmark test passed in {:?}", duration);
}

/// Test error handling and edge cases
#[tokio::test]
async fn test_error_handling_and_edge_cases() {
    let calculator = FinancialCalculator::new()
        .expect("Failed to create calculator");
    
    // Test invalid inputs
    assert!(calculator.compound_interest(f64::NAN, 0.05, 12.0, 1.0).is_err(),
        "Should reject NaN principal");
    
    assert!(calculator.compound_interest(1000.0, f64::INFINITY, 12.0, 1.0).is_err(),
        "Should reject infinite rate");
    
    assert!(calculator.compound_interest(-1000.0, 0.05, 12.0, 1.0).is_err(),
        "Should reject negative principal");
    
    // Test boundary conditions
    let zero_result = calculator.compound_interest(0.0, 0.05, 12.0, 1.0)
        .expect("Should handle zero principal");
    assert_eq!(zero_result, 0.0, "Zero principal should yield zero result");
    
    let zero_rate = calculator.compound_interest(1000.0, 0.0, 12.0, 1.0)
        .expect("Should handle zero interest rate");
    assert_eq!(zero_rate, 1000.0, "Zero rate should return original principal");
    
    // Test activation function error handling
    let sigmoid = ScientificSigmoid;
    assert!(sigmoid.activate(f64::NAN).is_err(), "Should reject NaN");
    assert!(sigmoid.activate(f64::INFINITY).is_err(), "Should reject infinity");
    assert!(sigmoid.activate(f64::NEG_INFINITY).is_err(), "Should reject negative infinity");
    
    println!("✅ Error handling and edge cases test passed");
}

/// Integration test for complete mathematical validation pipeline
#[tokio::test]
async fn test_complete_validation_pipeline() {
    // This test runs the complete pipeline from raw data to validated trading decision
    
    // Step 1: Initialize all systems
    let mut rigorous_system = ScientificallyRigorousSystem::new().await
        .expect("Failed to initialize rigorous system");
    
    // Step 2: Validate mathematical framework
    let validation_report = rigorous_system.validate_full_system().await
        .expect("Mathematical validation failed");
    
    assert!(validation_report.overall_compliance, "System must be mathematically compliant");
    
    // Step 3: Process market data with scientific rigor
    let market_data = MarketData {
        symbol: "ETH/USD".to_string(),
        prices: generate_realistic_price_series(1000), // 1000 data points
        volumes: generate_realistic_volume_series(1000),
        timestamp: std::time::SystemTime::now(),
    };
    
    // Step 4: Generate scientifically-grounded trading signal
    let trading_signal = generate_scientific_trading_signal(&market_data);
    
    // Step 5: Execute decision with full mathematical rigor
    let decision = rigorous_system.execute_rigorous_trading_decision(
        &market_data,
        &trading_signal
    ).await.expect("Rigorous trading decision failed");
    
    // Step 6: Validate decision properties
    assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0,
        "Decision confidence must be in valid range");
    assert!(decision.risk_score >= 0.0,
        "Risk score must be non-negative");
    assert!(!decision.mathematical_proof.is_empty(),
        "Mathematical proof must be provided");
    
    // Step 7: Verify system state consistency
    let final_state = rigorous_system.get_system_state().await;
    assert!(final_state.ieee754_compliance, "IEEE 754 compliance must be maintained");
    assert!(final_state.autopoietic_organization.identity_preserved,
        "Autopoietic identity must be preserved throughout");
    assert!(final_state.regulatory_compliance.sec_rule_15c3_5_compliant,
        "SEC Rule 15c3-5 compliance must be maintained");
    
    println!("✅ Complete validation pipeline test passed");
}

// Helper functions for test data generation
fn generate_realistic_price_series(length: usize) -> Vec<f64> {
    let mut prices = Vec::with_capacity(length);
    let mut price = 3000.0; // Starting price
    let mut rng_state = 12345u32; // Deterministic PRNG state
    
    for _ in 0..length {
        // Simple linear congruential generator for deterministic randomness
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let random = (rng_state as f64) / (u32::MAX as f64);
        
        // Generate price movement (normal-ish distribution approximation)
        let change = (random - 0.5) * 0.02; // ±1% maximum change
        price *= 1.0 + change;
        
        prices.push(price);
    }
    
    prices
}

fn generate_realistic_volume_series(length: usize) -> Vec<f64> {
    let mut volumes = Vec::with_capacity(length);
    let mut rng_state = 54321u32; // Different seed for volumes
    
    for _ in 0..length {
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let random = (rng_state as f64) / (u32::MAX as f64);
        
        // Generate volume (log-normal-ish distribution)
        let volume = 10.0 + random * 100.0; // 10-110 range
        volumes.push(volume);
    }
    
    volumes
}

fn generate_scientific_trading_signal(market_data: &MarketData) -> TradingSignal {
    // Calculate basic statistics for signal generation
    let prices = &market_data.prices;
    let returns: Vec<f64> = prices.windows(2)
        .map(|w| (w[1] / w[0] - 1.0).ln())
        .collect();
    
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>() / (returns.len() - 1) as f64;
    let volatility = variance.sqrt();
    
    // Generate signal based on scientific metrics
    let momentum = returns.iter().rev().take(10).sum::<f64>() / 10.0; // 10-period momentum
    let signal_strength = momentum.tanh(); // Bound between -1 and 1
    
    TradingSignal {
        strength: signal_strength,
        confidence: 0.7, // Conservative confidence  
        expected_return: mean_return,
        volatility,
        win_probability: 0.55, // Slightly better than random
        average_win: volatility * 2.0,
        average_loss: volatility * 1.5,
    }
}