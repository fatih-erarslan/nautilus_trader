//! Integration tests for the Talebian Risk Management library
//!
//! These tests verify that all components work together correctly
//! in realistic scenarios.

use talebian_risk::prelude::*;
use chrono::Utc;
use std::collections::HashMap;
use ndarray::Array2;

/// Test complete antifragility measurement workflow
#[test]
fn test_antifragility_measurement_workflow() {
    let params = AntifragilityParams {
        volatility_threshold: 0.02,
        convexity_sensitivity: 1.0,
        hormesis_window: 10,
        min_stress_level: 0.05,
        max_stress_level: 0.5,
    };
    
    let mut measurer = AntifragilityMeasurer::new("test_measurer", params);
    
    // Generate sample data with antifragile characteristics
    let returns = generate_antifragile_returns();
    
    // Measure antifragility
    let measurement = measurer.measure_antifragility(&returns).unwrap();
    
    // Verify measurements
    assert!(measurement.overall_score > 0.0, "Should detect antifragility");
    assert!(measurement.convexity > 0.0, "Should have positive convexity");
    assert!(measurement.volatility_benefit > 0.0, "Should benefit from volatility");
    assert!(measurement.is_antifragile(), "Should be classified as antifragile");
    
    // Test trend calculation
    for _ in 0..10 {
        let _ = measurer.measure_antifragility(&returns).unwrap();
    }
    
    let trend = measurer.get_trend(5).unwrap();
    assert!(trend.abs() <= 1.0, "Trend should be bounded");
    
    // Test regime updates
    measurer.update_regime(MarketRegime::Crisis);
    let crisis_measurement = measurer.measure_antifragility(&returns).unwrap();
    assert_eq!(crisis_measurement.market_regime, MarketRegime::Crisis);
}

/// Test complete black swan detection workflow
#[test]
fn test_black_swan_detection_workflow() {
    let params = BlackSwanParams {
        min_std_devs: 2.0,
        probability_threshold: 0.05,
        lookback_period: 100,
        min_impact: 0.03,
        ..Default::default()
    };
    
    let mut detector = BlackSwanDetector::new("test_detector", params);
    
    // Add normal observations
    for i in 0..150 {
        let observation = generate_normal_observation(i);
        detector.add_observation(observation).unwrap();
    }
    
    // Add black swan event
    let black_swan_obs = generate_black_swan_observation();
    detector.add_observation(black_swan_obs).unwrap();
    
    // Verify detection
    let events = detector.get_events();
    assert!(!events.is_empty(), "Should detect black swan events");
    
    let summary = detector.get_summary();
    assert!(summary.total_events > 0, "Should have detected events");
    assert!(summary.current_probability > 0.0, "Should have positive probability");
    
    // Test event filtering
    let crash_events = detector.get_events_by_type(BlackSwanType::MarketCrash);
    assert!(!crash_events.is_empty(), "Should detect market crash events");
    
    // Test alert states
    let alert_state = detector.get_alert_state();
    assert!(matches!(alert_state, AlertState::Elevated | AlertState::High | AlertState::Critical | AlertState::BlackSwanDetected));
}

/// Test complete barbell strategy workflow
#[test]
fn test_barbell_strategy_workflow() {
    let config = StrategyConfig {
        max_position_size: 0.3,
        risk_budget: 0.15,
        rebalancing_frequency: 21,
        min_position_size: 0.01,
        transaction_costs: 0.001,
        risk_aversion: 2.0,
        strategy_params: HashMap::new(),
    };
    
    let barbell_params = BarbellParams {
        safe_target: 0.8,
        risky_target: 0.2,
        max_safe_allocation: 0.95,
        max_risky_allocation: 0.25,
        min_safe_allocation: 0.6,
        min_risky_allocation: 0.05,
        safe_volatility_threshold: 0.05,
        risky_return_threshold: 0.1,
        rebalancing_tolerance: 0.05,
        adjustment_factor: 0.1,
        convexity_bias: 1.5,
    };
    
    let mut strategy = BarbellStrategy::new("test_barbell", config, barbell_params).unwrap();
    
    // Create market data
    let market_data = generate_market_data();
    
    // Test strategy suitability
    let suitable = strategy.is_suitable(&market_data).unwrap();
    assert!(suitable, "Strategy should be suitable for test data");
    
    // Test position size calculation
    let assets = vec!["BONDS".to_string(), "STOCKS".to_string(), "CRYPTO".to_string()];
    let positions = strategy.calculate_position_sizes(&assets, &market_data).unwrap();
    
    assert!(!positions.is_empty(), "Should calculate position sizes");
    let total_weight: f64 = positions.values().sum();
    assert!((total_weight - 1.0).abs() < 0.01, "Positions should sum to 1.0");
    
    // Test strategy updates
    strategy.update_strategy(&market_data).unwrap();
    
    // Test risk metrics
    let risk_metrics = strategy.risk_metrics(&market_data).unwrap();
    assert!(risk_metrics.volatility > 0.0, "Should have positive volatility");
    assert!(risk_metrics.antifragility_score >= 0.0, "Should have valid antifragility score");
    
    // Test expected return
    let expected_return = strategy.expected_return(&market_data).unwrap();
    assert!(expected_return.is_finite(), "Expected return should be finite");
    
    // Test barbell metrics
    let barbell_metrics = strategy.get_barbell_metrics();
    assert!(barbell_metrics.safe_allocation > 0.0, "Should have safe allocation");
    assert!(barbell_metrics.risky_allocation > 0.0, "Should have risky allocation");
    assert!(barbell_metrics.barbell_ratio > 0.0, "Should have positive barbell ratio");
    
    // Test capacity calculation
    let capacity = strategy.calculate_capacity(&market_data).unwrap();
    assert!(capacity > 0.0, "Should have positive capacity");
    
    // Test performance attribution
    let returns = vec![0.01, 0.02, -0.01, 0.03, 0.0];
    let attribution = strategy.performance_attribution(&returns).unwrap();
    assert!(!attribution.asset_contributions.is_empty(), "Should have asset contributions");
    
    // Test robustness assessment
    let scenarios = generate_stress_scenarios();
    let robustness = strategy.robustness_assessment(&scenarios).unwrap();
    assert!(robustness.robustness_score >= 0.0, "Should have valid robustness score");
    assert!(robustness.robustness_score <= 1.0, "Robustness score should be bounded");
}

/// Test integrated risk management workflow
#[test]
fn test_integrated_risk_management() {
    // Create components
    let antifragility_params = AntifragilityParams::default();
    let mut antifragility_measurer = AntifragilityMeasurer::new("portfolio_antifragility", antifragility_params);
    
    let black_swan_params = BlackSwanParams::default();
    let mut black_swan_detector = BlackSwanDetector::new("portfolio_black_swan", black_swan_params);
    
    let strategy_config = StrategyConfig::default();
    let barbell_params = BarbellParams::default();
    let mut barbell_strategy = BarbellStrategy::new("portfolio_barbell", strategy_config, barbell_params).unwrap();
    
    // Simulate market data over time
    let mut portfolio_returns = Vec::new();
    
    for day in 0..100 {
        let market_data = generate_daily_market_data(day);
        
        // Update strategy
        barbell_strategy.update_strategy(&market_data).unwrap();
        
        // Calculate portfolio return
        let portfolio_return = simulate_portfolio_return(&market_data);
        portfolio_returns.push(portfolio_return);
        
        // Add black swan observation
        let observation = create_market_observation(&market_data);
        black_swan_detector.add_observation(observation).unwrap();
        
        // Measure antifragility periodically
        if day % 21 == 0 && portfolio_returns.len() >= 21 {
            let recent_returns = &portfolio_returns[portfolio_returns.len()-21..];
            let measurement = antifragility_measurer.measure_antifragility(recent_returns).unwrap();
            
            // Adjust strategy based on antifragility
            if measurement.overall_score < 0.0 {
                // Strategy is fragile, increase safe allocation
                let mut new_params = barbell_strategy.get_barbell_params().clone();
                new_params.safe_target = (new_params.safe_target + 0.1).min(0.95);
                new_params.risky_target = 1.0 - new_params.safe_target;
                barbell_strategy.update_barbell_params(new_params).unwrap();
            }
        }
        
        // React to black swan events
        if black_swan_detector.get_alert_state() == AlertState::BlackSwanDetected {
            // Emergency risk reduction
            let mut new_params = barbell_strategy.get_barbell_params().clone();
            new_params.safe_target = 0.95;
            new_params.risky_target = 0.05;
            barbell_strategy.update_barbell_params(new_params).unwrap();
        }
    }
    
    // Final assessment
    let final_measurement = antifragility_measurer.measure_antifragility(&portfolio_returns).unwrap();
    let final_summary = black_swan_detector.get_summary();
    let final_metrics = barbell_strategy.get_barbell_metrics();
    
    // Verify integrated system worked
    assert!(final_measurement.overall_score.is_finite(), "Should have valid final antifragility score");
    assert!(final_summary.total_events >= 0, "Should have processed events");
    assert!(final_metrics.safe_allocation > 0.0, "Should maintain safe allocation");
    assert!(final_metrics.risky_allocation >= 0.0, "Should maintain risky allocation");
    
    // Check that system adapted to conditions
    if final_summary.total_events > 0 {
        assert!(final_metrics.safe_allocation > 0.8, "Should have increased safe allocation after events");
    }
}

/// Test extreme market conditions
#[test]
fn test_extreme_market_conditions() {
    let params = BlackSwanParams {
        min_std_devs: 1.0, // Lower threshold for testing
        probability_threshold: 0.1,
        lookback_period: 50,
        min_impact: 0.01,
        ..Default::default()
    };
    
    let mut detector = BlackSwanDetector::new("extreme_test", params);
    
    // Add extreme observations
    for i in 0..60 {
        let observation = if i == 50 {
            generate_extreme_crash_observation()
        } else if i == 55 {
            generate_extreme_volatility_observation()
        } else {
            generate_normal_observation(i)
        };
        
        detector.add_observation(observation).unwrap();
    }
    
    let events = detector.get_events();
    assert!(events.len() >= 2, "Should detect multiple extreme events");
    
    let summary = detector.get_summary();
    assert!(summary.average_severity > 3.0, "Should have high average severity");
    assert!(summary.current_probability > 0.05, "Should have elevated probability");
}

/// Test strategy robustness under stress
#[test]
fn test_strategy_stress_testing() {
    let config = StrategyConfig::default();
    let params = BarbellParams::default();
    let strategy = BarbellStrategy::new("stress_test", config, params).unwrap();
    
    // Generate stress scenarios
    let scenarios = vec![
        create_market_crash_scenario(),
        create_inflation_scenario(),
        create_liquidity_crisis_scenario(),
        create_correlation_breakdown_scenario(),
    ];
    
    let robustness = strategy.robustness_assessment(&scenarios).unwrap();
    
    // Verify robustness assessment
    assert!(robustness.worst_case_performance.is_finite(), "Should have finite worst case");
    assert!(robustness.best_case_performance.is_finite(), "Should have finite best case");
    assert!(robustness.worst_case_performance <= robustness.best_case_performance, "Worst case should be <= best case");
    
    // Check that fragility indicators are provided for poor robustness
    if robustness.robustness_score < 0.5 {
        assert!(!robustness.fragility_indicators.is_empty(), "Should provide fragility indicators");
        assert!(!robustness.recommended_adjustments.is_empty(), "Should provide recommendations");
    }
}

/// Test concurrent access and thread safety
#[test]
fn test_thread_safety() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let detector = Arc::new(Mutex::new(BlackSwanDetector::new("thread_test", BlackSwanParams::default())));
    let measurer = Arc::new(Mutex::new(AntifragilityMeasurer::new("thread_test", AntifragilityParams::default())));
    
    let mut handles = Vec::new();
    
    // Spawn multiple threads
    for i in 0..10 {
        let detector_clone = Arc::clone(&detector);
        let measurer_clone = Arc::clone(&measurer);
        
        let handle = thread::spawn(move || {
            // Add observations
            let observation = generate_normal_observation(i);
            detector_clone.lock().unwrap().add_observation(observation).unwrap();
            
            // Measure antifragility
            let returns = generate_random_returns();
            let _ = measurer_clone.lock().unwrap().measure_antifragility(&returns).unwrap();
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify final state
    let final_detector = detector.lock().unwrap();
    let final_measurer = measurer.lock().unwrap();
    
    assert!(final_detector.get_summary().total_events <= 10, "Should not exceed maximum events");
    assert!(final_measurer.get_history().len() <= 10, "Should not exceed maximum measurements");
}

// Helper functions for generating test data

fn generate_antifragile_returns() -> Vec<f64> {
    // Generate returns with positive convexity (more upside than downside)
    let mut returns = Vec::new();
    for i in 0..100 {
        let base_return = (i as f64 / 100.0 - 0.5) * 0.02; // Small trend
        let volatility = if i % 10 == 0 { 0.05 } else { 0.01 }; // Occasional volatility
        let skew = if base_return < 0.0 { 0.5 } else { 2.0 }; // Positive skew
        returns.push(base_return + volatility * skew);
    }
    returns
}

fn generate_normal_observation(day: usize) -> MarketObservation {
    let mut returns = HashMap::new();
    let mut volatilities = HashMap::new();
    let mut volumes = HashMap::new();
    
    // Generate normal market data
    returns.insert("STOCK1".to_string(), 0.001 * (day as f64 % 10.0 - 5.0));
    returns.insert("STOCK2".to_string(), 0.0005 * (day as f64 % 8.0 - 4.0));
    
    volatilities.insert("STOCK1".to_string(), 0.02);
    volatilities.insert("STOCK2".to_string(), 0.015);
    
    volumes.insert("STOCK1".to_string(), 1000000.0);
    volumes.insert("STOCK2".to_string(), 800000.0);
    
    MarketObservation {
        timestamp: Utc::now(),
        returns,
        volatilities,
        correlations: Array2::from_shape_vec((2, 2), vec![1.0, 0.3, 0.3, 1.0]).unwrap(),
        volumes,
        regime: MarketRegime::Normal,
    }
}

fn generate_black_swan_observation() -> MarketObservation {
    let mut returns = HashMap::new();
    let mut volatilities = HashMap::new();
    let mut volumes = HashMap::new();
    
    // Generate black swan event data
    returns.insert("STOCK1".to_string(), -0.2); // -20% return
    returns.insert("STOCK2".to_string(), -0.15); // -15% return
    
    volatilities.insert("STOCK1".to_string(), 0.1); // High volatility
    volatilities.insert("STOCK2".to_string(), 0.08);
    
    volumes.insert("STOCK1".to_string(), 500000.0); // Reduced volume
    volumes.insert("STOCK2".to_string(), 400000.0);
    
    MarketObservation {
        timestamp: Utc::now(),
        returns,
        volatilities,
        correlations: Array2::from_shape_vec((2, 2), vec![1.0, 0.9, 0.9, 1.0]).unwrap(), // High correlation
        volumes,
        regime: MarketRegime::Crash,
    }
}

fn generate_extreme_crash_observation() -> MarketObservation {
    let mut returns = HashMap::new();
    let mut volatilities = HashMap::new();
    let mut volumes = HashMap::new();
    
    returns.insert("STOCK1".to_string(), -0.5); // -50% crash
    returns.insert("STOCK2".to_string(), -0.4);
    
    volatilities.insert("STOCK1".to_string(), 0.3);
    volatilities.insert("STOCK2".to_string(), 0.25);
    
    volumes.insert("STOCK1".to_string(), 100000.0); // Very low volume
    volumes.insert("STOCK2".to_string(), 80000.0);
    
    MarketObservation {
        timestamp: Utc::now(),
        returns,
        volatilities,
        correlations: Array2::from_shape_vec((2, 2), vec![1.0, 0.95, 0.95, 1.0]).unwrap(),
        volumes,
        regime: MarketRegime::Crash,
    }
}

fn generate_extreme_volatility_observation() -> MarketObservation {
    let mut returns = HashMap::new();
    let mut volatilities = HashMap::new();
    let mut volumes = HashMap::new();
    
    returns.insert("STOCK1".to_string(), 0.3); // +30% spike
    returns.insert("STOCK2".to_string(), -0.25); // -25% drop
    
    volatilities.insert("STOCK1".to_string(), 0.5); // Extreme volatility
    volatilities.insert("STOCK2".to_string(), 0.4);
    
    volumes.insert("STOCK1".to_string(), 5000000.0); // Very high volume
    volumes.insert("STOCK2".to_string(), 4000000.0);
    
    MarketObservation {
        timestamp: Utc::now(),
        returns,
        volatilities,
        correlations: Array2::from_shape_vec((2, 2), vec![1.0, -0.8, -0.8, 1.0]).unwrap(), // Negative correlation
        volumes,
        regime: MarketRegime::HighVolatility,
    }
}

fn generate_market_data() -> MarketData {
    let mut prices = HashMap::new();
    let mut returns = HashMap::new();
    let mut volatilities = HashMap::new();
    let mut correlations = HashMap::new();
    let mut volumes = HashMap::new();
    let mut asset_types = HashMap::new();
    
    // Safe assets
    prices.insert("BONDS".to_string(), 100.0);
    returns.insert("BONDS".to_string(), vec![0.001, 0.002, 0.001, 0.0, 0.001]);
    volatilities.insert("BONDS".to_string(), 0.01);
    volumes.insert("BONDS".to_string(), 1000000.0);
    asset_types.insert("BONDS".to_string(), AssetType::Safe);
    
    // Risky assets
    prices.insert("STOCKS".to_string(), 50.0);
    returns.insert("STOCKS".to_string(), vec![0.02, -0.01, 0.03, 0.01, 0.02]);
    volatilities.insert("STOCKS".to_string(), 0.2);
    volumes.insert("STOCKS".to_string(), 2000000.0);
    asset_types.insert("STOCKS".to_string(), AssetType::Volatile);
    
    prices.insert("CRYPTO".to_string(), 1000.0);
    returns.insert("CRYPTO".to_string(), vec![0.05, -0.03, 0.08, -0.02, 0.1]);
    volatilities.insert("CRYPTO".to_string(), 0.5);
    volumes.insert("CRYPTO".to_string(), 500000.0);
    asset_types.insert("CRYPTO".to_string(), AssetType::Volatile);
    
    // Correlations
    correlations.insert(("BONDS".to_string(), "STOCKS".to_string()), 0.1);
    correlations.insert(("BONDS".to_string(), "CRYPTO".to_string()), 0.0);
    correlations.insert(("STOCKS".to_string(), "CRYPTO".to_string()), 0.6);
    
    MarketData {
        prices,
        returns,
        volatilities,
        correlations,
        volumes,
        asset_types,
        timestamp: Utc::now(),
        regime: MarketRegime::Normal,
    }
}

fn generate_daily_market_data(day: usize) -> MarketData {
    let mut market_data = generate_market_data();
    
    // Add some variation based on day
    let variation = (day as f64 / 100.0).sin() * 0.01;
    for returns in market_data.returns.values_mut() {
        for ret in returns.iter_mut() {
            *ret += variation;
        }
    }
    
    market_data
}

fn simulate_portfolio_return(market_data: &MarketData) -> f64 {
    let mut total_return = 0.0;
    let weights = [0.6, 0.3, 0.1]; // Example weights
    let assets = ["BONDS", "STOCKS", "CRYPTO"];
    
    for (i, asset) in assets.iter().enumerate() {
        if let Some(returns) = market_data.returns.get(*asset) {
            if let Some(&latest_return) = returns.last() {
                total_return += weights[i] * latest_return;
            }
        }
    }
    
    total_return
}

fn create_market_observation(market_data: &MarketData) -> MarketObservation {
    let mut returns = HashMap::new();
    let mut volatilities = HashMap::new();
    let mut volumes = HashMap::new();
    
    // Extract latest returns
    for (asset, asset_returns) in &market_data.returns {
        if let Some(&latest_return) = asset_returns.last() {
            returns.insert(asset.clone(), latest_return);
        }
    }
    
    for (asset, &vol) in &market_data.volatilities {
        volatilities.insert(asset.clone(), vol);
    }
    
    for (asset, &volume) in &market_data.volumes {
        volumes.insert(asset.clone(), volume);
    }
    
    MarketObservation {
        timestamp: market_data.timestamp,
        returns,
        volatilities,
        correlations: Array2::eye(3), // Simplified
        volumes,
        regime: market_data.regime,
    }
}

fn generate_random_returns() -> Vec<f64> {
    (0..50).map(|i| (i as f64 / 50.0 - 0.5) * 0.02).collect()
}

fn generate_stress_scenarios() -> Vec<MarketScenario> {
    vec![
        create_market_crash_scenario(),
        create_inflation_scenario(),
        create_liquidity_crisis_scenario(),
        create_correlation_breakdown_scenario(),
    ]
}

fn create_market_crash_scenario() -> MarketScenario {
    let mut price_shocks = HashMap::new();
    price_shocks.insert("STOCKS".to_string(), -0.3);
    price_shocks.insert("CRYPTO".to_string(), -0.5);
    price_shocks.insert("BONDS".to_string(), -0.1);
    
    MarketScenario {
        name: "Market Crash".to_string(),
        price_shocks,
        volatility_shocks: HashMap::new(),
        correlation_shocks: HashMap::new(),
        probability: 0.05,
        duration_days: 30,
    }
}

fn create_inflation_scenario() -> MarketScenario {
    let mut price_shocks = HashMap::new();
    price_shocks.insert("BONDS".to_string(), -0.2);
    price_shocks.insert("STOCKS".to_string(), -0.1);
    price_shocks.insert("CRYPTO".to_string(), 0.1);
    
    MarketScenario {
        name: "Inflation Shock".to_string(),
        price_shocks,
        volatility_shocks: HashMap::new(),
        correlation_shocks: HashMap::new(),
        probability: 0.1,
        duration_days: 90,
    }
}

fn create_liquidity_crisis_scenario() -> MarketScenario {
    let mut price_shocks = HashMap::new();
    price_shocks.insert("STOCKS".to_string(), -0.15);
    price_shocks.insert("CRYPTO".to_string(), -0.3);
    price_shocks.insert("BONDS".to_string(), 0.05);
    
    MarketScenario {
        name: "Liquidity Crisis".to_string(),
        price_shocks,
        volatility_shocks: HashMap::new(),
        correlation_shocks: HashMap::new(),
        probability: 0.03,
        duration_days: 60,
    }
}

fn create_correlation_breakdown_scenario() -> MarketScenario {
    let mut correlation_shocks = HashMap::new();
    correlation_shocks.insert(("STOCKS".to_string(), "BONDS".to_string()), 0.8);
    correlation_shocks.insert(("STOCKS".to_string(), "CRYPTO".to_string()), 0.9);
    
    MarketScenario {
        name: "Correlation Breakdown".to_string(),
        price_shocks: HashMap::new(),
        volatility_shocks: HashMap::new(),
        correlation_shocks,
        probability: 0.08,
        duration_days: 21,
    }
}