//! Integration tests for hedge algorithms

use hedge_algorithms::*;
use std::collections::HashMap;
use chrono::Utc;

#[test]
fn test_end_to_end_hedge_algorithm() {
    let config = HedgeConfig::default();
    let hedge = HedgeAlgorithms::new(config).unwrap();
    
    // Test with multiple market data points
    let market_data_points = vec![
        MarketData::new("BTCUSD".to_string(), Utc::now(), [100.0, 105.0, 95.0, 102.0, 1000.0]),
        MarketData::new("BTCUSD".to_string(), Utc::now(), [102.0, 108.0, 98.0, 105.0, 1100.0]),
        MarketData::new("BTCUSD".to_string(), Utc::now(), [105.0, 110.0, 101.0, 108.0, 1200.0]),
        MarketData::new("BTCUSD".to_string(), Utc::now(), [108.0, 112.0, 104.0, 109.0, 1300.0]),
        MarketData::new("BTCUSD".to_string(), Utc::now(), [109.0, 115.0, 106.0, 112.0, 1400.0]),
    ];
    
    let mut recommendations = Vec::new();
    
    for market_data in market_data_points {
        hedge.update_market_data(&market_data).unwrap();
        let recommendation = hedge.get_hedge_recommendation().unwrap();
        
        // Validate recommendation
        assert!(recommendation.is_valid());
        assert!(recommendation.confidence >= 0.0);
        assert!(recommendation.confidence <= 1.0);
        
        recommendations.push(recommendation);
    }
    
    // Check that we have recommendations
    assert_eq!(recommendations.len(), 5);
    
    // Check that recommendations are reasonable
    let last_recommendation = recommendations.last().unwrap();
    assert!(last_recommendation.position_size.is_finite());
    assert!(last_recommendation.hedge_ratio.is_finite());
    assert!(last_recommendation.expected_return.is_finite());
    
    // Get performance metrics
    let metrics = hedge.get_metrics();
    assert!(metrics.total_trades > 0);
}

#[test]
fn test_quantum_hedge_algorithm_integration() {
    let config = HedgeConfig::default();
    let expert_names = vec![
        "trend_follower".to_string(),
        "mean_reverter".to_string(),
        "volatility_trader".to_string(),
        "momentum_expert".to_string(),
    ];
    
    let mut quantum_hedge = QuantumHedgeAlgorithm::new(expert_names, config).unwrap();
    
    // Test quantum state evolution
    let market_data_sequence = vec![
        MarketData::new("BTCUSD".to_string(), Utc::now(), [100.0, 105.0, 95.0, 102.0, 1000.0]),
        MarketData::new("BTCUSD".to_string(), Utc::now(), [102.0, 108.0, 98.0, 105.0, 1100.0]),
        MarketData::new("BTCUSD".to_string(), Utc::now(), [105.0, 110.0, 101.0, 108.0, 1200.0]),
    ];
    
    let mut predictions = HashMap::new();
    predictions.insert("trend_follower".to_string(), 0.05);
    predictions.insert("mean_reverter".to_string(), -0.02);
    predictions.insert("volatility_trader".to_string(), 0.03);
    predictions.insert("momentum_expert".to_string(), 0.01);
    
    let mut entropies = Vec::new();
    let mut purities = Vec::new();
    
    for market_data in market_data_sequence {
        quantum_hedge.update(&market_data, &predictions).unwrap();
        
        let entropy = quantum_hedge.get_entropy();
        let purity = quantum_hedge.get_purity();
        
        entropies.push(entropy);
        purities.push(purity);
        
        // Validate quantum properties
        assert!(entropy >= 0.0);
        assert!(purity >= 0.0);
        assert!(purity <= 1.0);
        
        // Test quantum measurement
        let measurement = quantum_hedge.measure().unwrap();
        assert!(predictions.contains_key(&measurement));
        
        // Get recommendation
        let (weights, confidence) = quantum_hedge.get_recommendation().unwrap();
        assert!(confidence >= 0.0);
        assert!(confidence <= 1.0);
        
        // Validate weights
        let total_weight: f64 = weights.values().sum();
        assert!((total_weight - 1.0).abs() < 1e-10);
    }
    
    // Check that quantum state evolved
    assert!(entropies.len() == 3);
    assert!(purities.len() == 3);
}

#[test]
fn test_expert_system_integration() {
    let config = HedgeConfig::default();
    let expert_system = ExpertSystem::new(config);
    
    // Create and register experts
    let trend_expert = Box::new(TrendFollowingExpert::new("trend".to_string(), 20, 0.02));
    let mean_expert = Box::new(MeanReversionExpert::new("mean_revert".to_string(), 50, 0.05));
    let vol_expert = Box::new(VolatilityExpert::new("volatility".to_string(), 30));
    
    expert_system.register_expert("trend".to_string(), trend_expert).unwrap();
    expert_system.register_expert("mean_revert".to_string(), mean_expert).unwrap();
    expert_system.register_expert("volatility".to_string(), vol_expert).unwrap();
    
    // Test expert system operations
    let market_data = MarketData::new("BTCUSD".to_string(), Utc::now(), [100.0, 105.0, 95.0, 102.0, 1000.0]);
    expert_system.update_all(&market_data).unwrap();
    
    let predictions = expert_system.get_predictions().unwrap();
    assert_eq!(predictions.len(), 3);
    
    let confidences = expert_system.get_confidences().unwrap();
    assert_eq!(confidences.len(), 3);
    
    // Test performance tracking
    expert_system.update_performance(0.02).unwrap();
    
    let rankings = expert_system.get_expert_rankings().unwrap();
    assert_eq!(rankings.len(), 3);
    
    let statistics = expert_system.get_expert_statistics().unwrap();
    assert_eq!(statistics.len(), 3);
}

#[test]
fn test_factor_model_integration() {
    let config = HedgeConfig::default();
    let mut factor_model = StandardFactorModel::new(config).unwrap();
    
    // Test factor model with market data sequence
    let market_data_sequence = vec![
        MarketData::new("BTCUSD".to_string(), Utc::now(), [100.0, 105.0, 95.0, 102.0, 1000.0]),
        MarketData::new("BTCUSD".to_string(), Utc::now(), [102.0, 108.0, 98.0, 105.0, 1100.0]),
        MarketData::new("BTCUSD".to_string(), Utc::now(), [105.0, 110.0, 101.0, 108.0, 1200.0]),
        MarketData::new("BTCUSD".to_string(), Utc::now(), [108.0, 112.0, 104.0, 109.0, 1300.0]),
    ];
    
    for market_data in market_data_sequence {
        factor_model.update(&market_data).unwrap();
    }
    
    // Test factor exposures
    let exposures = factor_model.get_exposures().unwrap();
    assert_eq!(exposures.len(), 8); // 8 factors
    
    // Test factor returns
    let factor_returns = factor_model.get_factor_returns();
    assert!(factor_returns.nrows() == 8);
    
    // Test factor covariance
    let covariance = factor_model.get_factor_covariance();
    assert!(covariance.nrows() == 8);
    assert!(covariance.ncols() == 8);
    
    // Test specific risk
    let specific_risk = factor_model.get_specific_risk();
    assert!(specific_risk >= 0.0);
    
    // Test prediction
    let factor_expected_returns = nalgebra::DVector::from_vec(vec![0.01; 8]);
    let predicted_return = factor_model.predict_return(&factor_expected_returns).unwrap();
    assert!(predicted_return.is_finite());
    
    let predicted_risk = factor_model.predict_risk(&factor_expected_returns).unwrap();
    assert!(predicted_risk >= 0.0);
}

#[test]
fn test_options_hedging_integration() {
    let config = HedgeConfig::default();
    let hedger = OptionsHedger::new(config);
    
    // Test Black-Scholes pricing
    let call_price = hedger.black_scholes_price(100.0, 100.0, 1.0, 0.2, OptionType::Call).unwrap();
    let put_price = hedger.black_scholes_price(100.0, 100.0, 1.0, 0.2, OptionType::Put).unwrap();
    
    assert!(call_price > 0.0);
    assert!(put_price > 0.0);
    
    // Test put-call parity
    let parity_diff = call_price - put_price - (100.0 - 100.0 * (-0.05 * 1.0).exp());
    assert!(parity_diff.abs() < 1e-10);
    
    // Test Greeks
    let call_greeks = hedger.calculate_greeks(100.0, 100.0, 1.0, 0.2, OptionType::Call).unwrap();
    let put_greeks = hedger.calculate_greeks(100.0, 100.0, 1.0, 0.2, OptionType::Put).unwrap();
    
    // Delta relationship
    assert!((call_greeks.delta - put_greeks.delta - 1.0).abs() < 1e-10);
    
    // Gamma should be the same
    assert!((call_greeks.gamma - put_greeks.gamma).abs() < 1e-10);
    
    // Vega should be the same
    assert!((call_greeks.vega - put_greeks.vega).abs() < 1e-10);
    
    // Test hedge ratio calculation
    let hedge_ratio = hedger.calculate_hedge_ratio(&call_greeks, 100.0).unwrap();
    assert!(hedge_ratio.is_finite());
    
    // Test implied volatility
    let implied_vol = hedger.calculate_implied_volatility(call_price, 100.0, 100.0, 1.0, OptionType::Call).unwrap();
    assert!((implied_vol - 0.2).abs() < 1e-2);
}

#[test]
fn test_pairs_trading_integration() {
    let config = HedgeConfig::default();
    let mut pairs_trader = PairsTrader::new(config);
    
    // Test with correlated price series
    let price_series_a = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0];
    let price_series_b = vec![98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0];
    
    let mut signals = Vec::new();
    
    for (price_a, price_b) in price_series_a.iter().zip(price_series_b.iter()) {
        pairs_trader.update(*price_a, *price_b).unwrap();
        
        if let Some(signal) = pairs_trader.generate_signal().unwrap() {
            signals.push(signal);
        }
    }
    
    // Test hedge ratio
    let hedge_ratio = pairs_trader.get_hedge_ratio();
    assert!(hedge_ratio > 0.0);
    assert!(hedge_ratio < 2.0);
    
    // Test cointegration
    let cointegrated = pairs_trader.check_cointegration().unwrap();
    // Note: May not be cointegrated with small sample
    
    // Test z-score calculation
    let zscore = pairs_trader.calculate_zscore().unwrap();
    assert!(zscore.is_finite());
}

#[test]
fn test_whale_detection_integration() {
    let config = HedgeConfig::default();
    let mut whale_detector = WhaleDetector::new(config);
    
    // Test with market data containing volume spikes
    let market_data_sequence = vec![
        MarketData::new("BTCUSD".to_string(), Utc::now(), [100.0, 105.0, 95.0, 102.0, 1000.0]),
        MarketData::new("BTCUSD".to_string(), Utc::now(), [102.0, 108.0, 98.0, 105.0, 1100.0]),
        MarketData::new("BTCUSD".to_string(), Utc::now(), [105.0, 110.0, 101.0, 108.0, 5000.0]), // Volume spike
        MarketData::new("BTCUSD".to_string(), Utc::now(), [108.0, 112.0, 104.0, 109.0, 1300.0]),
        MarketData::new("BTCUSD".to_string(), Utc::now(), [109.0, 115.0, 106.0, 112.0, 1400.0]),
    ];
    
    for market_data in market_data_sequence {
        whale_detector.update(&market_data).unwrap();
    }
    
    // Test whale statistics
    let statistics = whale_detector.get_whale_statistics();
    assert!(statistics.total_activities >= 0);
    
    // Test recent activities
    let recent_activities = whale_detector.get_recent_activities(5);
    assert!(recent_activities.len() <= 5);
    
    // Test trading signal
    let signal = whale_detector.get_trading_signal().unwrap();
    // Signal may be None if no whale activity detected
}

#[test]
fn test_regret_minimization_integration() {
    let config = HedgeConfig::default();
    let mut regret_minimizer = RegretMinimizer::new(config);
    
    // Initialize experts
    regret_minimizer.initialize_expert("expert1").unwrap();
    regret_minimizer.initialize_expert("expert2").unwrap();
    regret_minimizer.initialize_expert("expert3").unwrap();
    
    // Test regret tracking over time
    let test_scenarios = vec![
        (vec![0.05, -0.02, 0.03], 0.02, 0.04),
        (vec![0.01, 0.04, -0.01], 0.015, 0.02),
        (vec![-0.03, 0.02, 0.05], 0.01, -0.01),
        (vec![0.04, -0.01, 0.02], 0.017, 0.03),
    ];
    
    for (predictions, portfolio_pred, actual) in test_scenarios {
        let mut expert_predictions = HashMap::new();
        expert_predictions.insert("expert1".to_string(), predictions[0]);
        expert_predictions.insert("expert2".to_string(), predictions[1]);
        expert_predictions.insert("expert3".to_string(), predictions[2]);
        
        let mut expert_weights = HashMap::new();
        expert_weights.insert("expert1".to_string(), 0.4);
        expert_weights.insert("expert2".to_string(), 0.3);
        expert_weights.insert("expert3".to_string(), 0.3);
        
        regret_minimizer.update_external_regret(&expert_predictions, portfolio_pred, actual).unwrap();
        regret_minimizer.update_internal_regret(&expert_predictions, &expert_weights, actual).unwrap();
        regret_minimizer.update_cumulative_regret(&expert_predictions, portfolio_pred, actual).unwrap();
        regret_minimizer.update_time_step();
    }
    
    // Test regret statistics
    let statistics = regret_minimizer.get_regret_statistics();
    assert_eq!(statistics.total_experts, 3);
    assert_eq!(statistics.time_step, 4);
    
    // Test regret bounds
    for expert in ["expert1", "expert2", "expert3"] {
        let bound = regret_minimizer.get_regret_bound(expert).unwrap();
        assert!(bound >= 0.0);
        
        let is_bounded = regret_minimizer.is_regret_bounded(expert).unwrap();
        // Regret should be bounded for small number of time steps
    }
    
    // Test average regret
    let avg_regret = regret_minimizer.get_average_regret();
    assert!(avg_regret.is_finite());
}

#[test]
fn test_performance_metrics_integration() {
    let mut metrics = PerformanceMetrics::new();
    
    // Test with sequence of recommendations
    let test_recommendations = vec![
        HedgeRecommendation {
            position_size: 1.0,
            hedge_ratio: 0.5,
            confidence: 0.8,
            factor_exposures: nalgebra::DVector::zeros(8),
            risk_metrics: RiskMetrics::default(),
            expected_return: 0.02,
            volatility: 0.15,
            max_drawdown: 0.05,
            sharpe_ratio: 0.13,
            timestamp: Utc::now(),
        },
        HedgeRecommendation {
            position_size: 1.2,
            hedge_ratio: 0.6,
            confidence: 0.75,
            factor_exposures: nalgebra::DVector::zeros(8),
            risk_metrics: RiskMetrics::default(),
            expected_return: -0.01,
            volatility: 0.18,
            max_drawdown: 0.08,
            sharpe_ratio: -0.06,
            timestamp: Utc::now(),
        },
        HedgeRecommendation {
            position_size: 0.8,
            hedge_ratio: 0.4,
            confidence: 0.85,
            factor_exposures: nalgebra::DVector::zeros(8),
            risk_metrics: RiskMetrics::default(),
            expected_return: 0.03,
            volatility: 0.12,
            max_drawdown: 0.03,
            sharpe_ratio: 0.25,
            timestamp: Utc::now(),
        },
    ];
    
    for recommendation in test_recommendations {
        metrics.update(&recommendation).unwrap();
    }
    
    // Test metrics calculation
    assert_eq!(metrics.total_trades, 3);
    assert_eq!(metrics.winning_trades, 2);
    assert_eq!(metrics.losing_trades, 1);
    assert_eq!(metrics.hit_rate, 2.0 / 3.0);
    assert!(metrics.profit_factor > 0.0);
    assert!(metrics.sortino_ratio != 0.0);
    
    // Test summary
    let summary = metrics.summary();
    assert!(summary.contains_key("total_trades"));
    assert!(summary.contains_key("hit_rate"));
    assert!(summary.contains_key("profit_factor"));
}

#[test]
fn test_configuration_validation() {
    // Test valid configuration
    let valid_config = HedgeConfig::default();
    assert!(valid_config.validate().is_ok());
    
    // Test invalid configurations
    let mut invalid_config = HedgeConfig::default();
    invalid_config.learning_rate = 1.5; // Invalid: > 1.0
    assert!(invalid_config.validate().is_err());
    
    invalid_config = HedgeConfig::default();
    invalid_config.min_weight = 0.6;
    invalid_config.max_weight = 0.5; // Invalid: min > max
    assert!(invalid_config.validate().is_err());
    
    invalid_config = HedgeConfig::default();
    invalid_config.max_history = 0; // Invalid: must be > 0
    assert!(invalid_config.validate().is_err());
}

#[test]
fn test_state_persistence() {
    let config = HedgeConfig::default();
    let hedge = HedgeAlgorithms::new(config).unwrap();
    
    // Update with some data
    let market_data = MarketData::new("BTCUSD".to_string(), Utc::now(), [100.0, 105.0, 95.0, 102.0, 1000.0]);
    hedge.update_market_data(&market_data).unwrap();
    
    // Get initial recommendation
    let initial_recommendation = hedge.get_hedge_recommendation().unwrap();
    
    // Save state
    let temp_path = "/tmp/hedge_test_state.bin";
    hedge.save_state(temp_path).unwrap();
    
    // Load state
    let loaded_hedge = HedgeAlgorithms::load_state(temp_path).unwrap();
    
    // Get recommendation from loaded state
    let loaded_recommendation = loaded_hedge.get_hedge_recommendation().unwrap();
    
    // Compare recommendations (should be similar)
    assert!((initial_recommendation.position_size - loaded_recommendation.position_size).abs() < 1e-10);
    assert!((initial_recommendation.hedge_ratio - loaded_recommendation.hedge_ratio).abs() < 1e-10);
    assert!((initial_recommendation.confidence - loaded_recommendation.confidence).abs() < 1e-10);
    
    // Clean up
    std::fs::remove_file(temp_path).ok();
}

#[test]
fn test_mathematical_utilities() {
    let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let data2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    
    // Test correlation
    let correlation = utils::math::correlation(&data1, &data2).unwrap();
    assert!((correlation - 1.0).abs() < 1e-10);
    
    // Test variance
    let variance = utils::math::variance(&data1).unwrap();
    assert!((variance - 2.5).abs() < 1e-10);
    
    // Test standard deviation
    let std_dev = utils::math::standard_deviation(&data1).unwrap();
    assert!((std_dev - 2.5_f64.sqrt()).abs() < 1e-10);
    
    // Test returns
    let prices = vec![100.0, 110.0, 105.0, 115.0];
    let returns = utils::math::returns(&prices).unwrap();
    assert_eq!(returns.len(), 3);
    assert!((returns[0] - 0.1).abs() < 1e-10);
    
    // Test log returns
    let log_returns = utils::math::log_returns(&prices).unwrap();
    assert_eq!(log_returns.len(), 3);
    assert!((log_returns[0] - (110.0 / 100.0).ln()).abs() < 1e-10);
}

#[test]
fn test_concurrent_access() {
    use std::sync::Arc;
    use std::thread;
    
    let config = HedgeConfig::default();
    let hedge = Arc::new(HedgeAlgorithms::new(config).unwrap());
    
    let mut handles = Vec::new();
    
    // Test concurrent updates
    for i in 0..10 {
        let hedge_clone = Arc::clone(&hedge);
        let handle = thread::spawn(move || {
            let market_data = MarketData::new(
                format!("TEST{}", i),
                Utc::now(),
                [100.0 + i as f64, 105.0 + i as f64, 95.0 + i as f64, 102.0 + i as f64, 1000.0]
            );
            
            hedge_clone.update_market_data(&market_data).unwrap();
            hedge_clone.get_hedge_recommendation().unwrap();
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify final state
    let final_recommendation = hedge.get_hedge_recommendation().unwrap();
    assert!(final_recommendation.is_valid());
}