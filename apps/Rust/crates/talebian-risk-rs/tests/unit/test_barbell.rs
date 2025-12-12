//! Unit tests for barbell strategy implementation
//! Tests portfolio allocation, convexity optimization, and stress adaptation

use talebian_risk_rs::{
    strategies::{
        barbell::{BarbellStrategy, BarbellParams, BarbellMetrics, PerformanceRecord},
        *,
    },
    MarketData, MacchiavelianConfig,
};
use chrono::Utc;
use std::collections::HashMap;

/// Helper to create test market data for barbell strategy
fn create_test_barbell_market_data() -> MarketData {
    let mut asset_types = HashMap::new();
    asset_types.insert("TREASURY_BONDS".to_string(), AssetType::Safe);
    asset_types.insert("CASH".to_string(), AssetType::Safe);
    asset_types.insert("BITCOIN".to_string(), AssetType::Volatile);
    asset_types.insert("TECH_STOCKS".to_string(), AssetType::Risky);
    asset_types.insert("OPTIONS".to_string(), AssetType::Derivative);
    
    let mut prices = HashMap::new();
    prices.insert("TREASURY_BONDS".to_string(), 100.0);
    prices.insert("CASH".to_string(), 1.0);
    prices.insert("BITCOIN".to_string(), 50000.0);
    prices.insert("TECH_STOCKS".to_string(), 200.0);
    prices.insert("OPTIONS".to_string(), 5.0);
    
    let mut returns = HashMap::new();
    returns.insert("TREASURY_BONDS".to_string(), vec![0.02, 0.025, 0.018, 0.022]);
    returns.insert("CASH".to_string(), vec![0.001, 0.001, 0.001, 0.001]);
    returns.insert("BITCOIN".to_string(), vec![0.15, -0.08, 0.12, 0.25]);
    returns.insert("TECH_STOCKS".to_string(), vec![0.08, 0.12, -0.05, 0.18]);
    returns.insert("OPTIONS".to_string(), vec![0.3, -0.15, 0.2, -0.1]);
    
    let mut volatilities = HashMap::new();
    volatilities.insert("TREASURY_BONDS".to_string(), 0.02);
    volatilities.insert("CASH".to_string(), 0.001);
    volatilities.insert("BITCOIN".to_string(), 0.6);
    volatilities.insert("TECH_STOCKS".to_string(), 0.25);
    volatilities.insert("OPTIONS".to_string(), 0.8);
    
    let mut volumes = HashMap::new();
    volumes.insert("TREASURY_BONDS".to_string(), 10000000.0);
    volumes.insert("CASH".to_string(), 100000000.0);
    volumes.insert("BITCOIN".to_string(), 5000000.0);
    volumes.insert("TECH_STOCKS".to_string(), 8000000.0);
    volumes.insert("OPTIONS".to_string(), 1000000.0);
    
    let mut correlations = HashMap::new();
    correlations.insert(("TREASURY_BONDS".to_string(), "BITCOIN".to_string()), -0.1);
    correlations.insert(("TREASURY_BONDS".to_string(), "TECH_STOCKS".to_string()), 0.05);
    correlations.insert(("BITCOIN".to_string(), "TECH_STOCKS".to_string()), 0.3);
    
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

/// Helper to create stressed market conditions
fn create_stressed_market_data() -> MarketData {
    let mut market_data = create_test_barbell_market_data();
    
    // Increase volatilities during stress
    market_data.volatilities.insert("TREASURY_BONDS".to_string(), 0.05);
    market_data.volatilities.insert("BITCOIN".to_string(), 1.2);
    market_data.volatilities.insert("TECH_STOCKS".to_string(), 0.8);
    
    // Negative returns during crisis
    market_data.returns.insert("BITCOIN".to_string(), vec![-0.3, -0.2, -0.25, -0.15]);
    market_data.returns.insert("TECH_STOCKS".to_string(), vec![-0.15, -0.1, -0.2, -0.08]);
    
    // Higher correlations during stress
    market_data.correlations.insert(("BITCOIN".to_string(), "TECH_STOCKS".to_string()), 0.8);
    
    market_data.regime = MarketRegime::Crisis;
    market_data
}

#[cfg(test)]
mod barbell_strategy_tests {
    use super::*;

    #[test]
    fn test_barbell_strategy_creation() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        
        let strategy = BarbellStrategy::new("test_barbell", config, params);
        assert!(strategy.is_ok());
        
        let strategy = strategy.unwrap();
        assert_eq!(strategy.id(), "test_barbell");
        assert_eq!(strategy.name(), "Barbell Strategy");
        assert_eq!(strategy.strategy_type(), StrategyType::Barbell);
    }

    #[test]
    fn test_barbell_params_validation() {
        let config = StrategyConfig::default();
        let mut params = BarbellParams::default();
        
        // Test invalid allocation sum
        params.safe_target = 0.9;
        params.risky_target = 0.9; // Sum > 1.0
        
        let result = BarbellStrategy::new("invalid_barbell", config.clone(), params);
        assert!(result.is_err());
        
        // Test safe target below minimum
        let mut params = BarbellParams::default();
        params.safe_target = 0.4; // Below min_safe_allocation (0.6)
        
        let result = BarbellStrategy::new("invalid_barbell", config.clone(), params);
        assert!(result.is_err());
        
        // Test risky target below minimum
        let mut params = BarbellParams::default();
        params.risky_target = 0.02; // Below min_risky_allocation (0.05)
        
        let result = BarbellStrategy::new("invalid_barbell", config, params);
        assert!(result.is_err());
    }

    #[test]
    fn test_asset_classification() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        let market_data = create_test_barbell_market_data();
        let (safe_assets, risky_assets) = strategy.classify_assets(&market_data).unwrap();
        
        assert!(!safe_assets.is_empty());
        assert!(!risky_assets.is_empty());
        assert!(safe_assets.contains(&"TREASURY_BONDS".to_string()));
        assert!(safe_assets.contains(&"CASH".to_string()));
        assert!(risky_assets.contains(&"BITCOIN".to_string()));
        assert!(risky_assets.contains(&"TECH_STOCKS".to_string()));
        assert!(risky_assets.contains(&"OPTIONS".to_string()));
    }

    #[test]
    fn test_position_size_calculation() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        let market_data = create_test_barbell_market_data();
        let assets = vec![
            "TREASURY_BONDS".to_string(),
            "CASH".to_string(),
            "BITCOIN".to_string(),
            "TECH_STOCKS".to_string(),
        ];
        
        let positions = strategy.calculate_position_sizes(&assets, &market_data).unwrap();
        
        // Check that positions are calculated
        assert!(!positions.is_empty());
        
        // Check that total allocation is reasonable (should be close to 1.0)
        let total_allocation: f64 = positions.values().sum();
        assert!(total_allocation > 0.9 && total_allocation <= 1.0);
        
        // Check that safe assets get higher allocation than risky assets
        let safe_allocation: f64 = positions.iter()
            .filter(|(asset, _)| asset.contains("TREASURY") || asset.contains("CASH"))
            .map(|(_, weight)| weight)
            .sum();
        
        let risky_allocation: f64 = positions.iter()
            .filter(|(asset, _)| asset.contains("BITCOIN") || asset.contains("TECH"))
            .map(|(_, weight)| weight)
            .sum();
        
        assert!(safe_allocation > risky_allocation);
    }

    #[test]
    fn test_convexity_calculation() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        // Test positive convexity (more upside than downside)
        let returns = vec![-0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25];
        let convexity = strategy.calculate_convexity(&returns).unwrap();
        assert!(convexity > 1.0);
        
        // Test symmetric returns
        let symmetric_returns = vec![-0.1, -0.05, 0.0, 0.05, 0.1];
        let convexity = strategy.calculate_convexity(&symmetric_returns).unwrap();
        assert!(convexity >= 1.0);
        
        // Test insufficient data
        let small_returns = vec![0.1, 0.2];
        let convexity = strategy.calculate_convexity(&small_returns).unwrap();
        assert_eq!(convexity, 0.0);
    }

    #[test]
    fn test_market_stress_calculation() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        // Test normal market conditions
        let normal_market = create_test_barbell_market_data();
        let normal_stress = strategy.calculate_market_stress(&normal_market).unwrap();
        assert!(normal_stress >= 0.0 && normal_stress <= 1.0);
        assert!(normal_stress < 0.5); // Should be low stress
        
        // Test stressed market conditions
        let stressed_market = create_stressed_market_data();
        let high_stress = strategy.calculate_market_stress(&stressed_market).unwrap();
        assert!(high_stress >= 0.0 && high_stress <= 1.0);
        assert!(high_stress > normal_stress); // Should be higher stress
    }

    #[test]
    fn test_allocation_adjustment() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let mut strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        let initial_safe = strategy.safe_allocation;
        let initial_risky = strategy.risky_allocation;
        
        // Test adjustment under stress
        let stressed_market = create_stressed_market_data();
        strategy.adjust_allocations(&stressed_market).unwrap();
        
        // Safe allocation should increase under stress
        assert!(strategy.safe_allocation >= initial_safe);
        // Risky allocation should decrease under stress
        assert!(strategy.risky_allocation <= initial_risky);
        
        // Allocations should still sum to reasonable total
        let total = strategy.safe_allocation + strategy.risky_allocation;
        assert!(total > 0.9 && total <= 1.0);
    }

    #[test]
    fn test_rebalancing_trigger() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        // Test no rebalancing needed
        let mut current_weights = HashMap::new();
        current_weights.insert("TREASURY_BONDS".to_string(), 0.8);
        current_weights.insert("BITCOIN".to_string(), 0.2);
        
        // Need to set up portfolio asset types for proper classification
        let mut strategy = strategy;
        strategy.portfolio.asset_types.insert("TREASURY_BONDS".to_string(), AssetType::Safe);
        strategy.portfolio.asset_types.insert("BITCOIN".to_string(), AssetType::Volatile);
        
        let needs_rebalancing = strategy.needs_rebalancing(&current_weights);
        assert!(!needs_rebalancing); // Should not need rebalancing
        
        // Test rebalancing needed due to drift
        let mut drifted_weights = HashMap::new();
        drifted_weights.insert("TREASURY_BONDS".to_string(), 0.6); // Drifted from 0.8
        drifted_weights.insert("BITCOIN".to_string(), 0.4); // Drifted from 0.2
        
        let needs_rebalancing = strategy.needs_rebalancing(&drifted_weights);
        assert!(needs_rebalancing); // Should need rebalancing
    }

    #[test]
    fn test_barbell_metrics() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let mut strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        // Set up portfolio for testing
        strategy.portfolio.weights.insert("TREASURY_BONDS".to_string(), 0.8);
        strategy.portfolio.weights.insert("BITCOIN".to_string(), 0.2);
        strategy.portfolio.asset_types.insert("TREASURY_BONDS".to_string(), AssetType::Safe);
        strategy.portfolio.asset_types.insert("BITCOIN".to_string(), AssetType::Volatile);
        
        let metrics = strategy.get_barbell_metrics();
        
        assert!(metrics.safe_allocation > 0.0);
        assert!(metrics.risky_allocation > 0.0);
        assert!(metrics.safe_allocation + metrics.risky_allocation <= 1.0);
        assert!(metrics.barbell_ratio > 0.0);
        assert!(metrics.convexity_exposure >= 0.0);
        assert!(metrics.safety_score >= 0.0 && metrics.safety_score <= 1.0);
        assert!(metrics.allocation_drift >= 0.0);
    }

    #[test]
    fn test_performance_tracking() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let mut strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        // Set up portfolio
        strategy.portfolio.weights.insert("TREASURY_BONDS".to_string(), 0.8);
        strategy.portfolio.weights.insert("BITCOIN".to_string(), 0.2);
        strategy.portfolio.asset_types.insert("TREASURY_BONDS".to_string(), AssetType::Safe);
        strategy.portfolio.asset_types.insert("BITCOIN".to_string(), AssetType::Volatile);
        
        let market_data = create_test_barbell_market_data();
        strategy.record_performance(&market_data).unwrap();
        
        let history = strategy.get_performance_history();
        assert_eq!(history.len(), 1);
        
        let record = &history[0];
        assert!(record.total_return.is_finite());
        assert!(record.safe_return.is_finite());
        assert!(record.risky_return.is_finite());
        assert!(record.volatility >= 0.0);
        assert!(record.max_drawdown >= 0.0);
        assert!(record.antifragility_score >= 0.0);
    }

    #[test]
    fn test_strategy_suitability() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        // Test with suitable market (has both safe and risky assets)
        let suitable_market = create_test_barbell_market_data();
        let is_suitable = strategy.is_suitable(&suitable_market).unwrap();
        assert!(is_suitable);
        
        // Test with unsuitable market (only safe assets)
        let mut unsuitable_market = suitable_market.clone();
        unsuitable_market.asset_types.clear();
        unsuitable_market.asset_types.insert("TREASURY_BONDS".to_string(), AssetType::Safe);
        unsuitable_market.asset_types.insert("CASH".to_string(), AssetType::Safe);
        
        let is_suitable = strategy.is_suitable(&unsuitable_market);
        assert!(is_suitable.is_err() || !is_suitable.unwrap());
    }

    #[test]
    fn test_capacity_calculation() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let mut strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        // Set up portfolio
        strategy.portfolio.weights.insert("TREASURY_BONDS".to_string(), 0.8);
        strategy.portfolio.weights.insert("BITCOIN".to_string(), 0.2);
        
        let market_data = create_test_barbell_market_data();
        let capacity = strategy.calculate_capacity(&market_data).unwrap();
        
        assert!(capacity > 0.0);
        assert!(capacity.is_finite());
    }

    #[test]
    fn test_risk_metrics() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let mut strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        // Set up portfolio
        strategy.portfolio.weights.insert("TREASURY_BONDS".to_string(), 0.8);
        strategy.portfolio.weights.insert("BITCOIN".to_string(), 0.2);
        strategy.portfolio.asset_types.insert("TREASURY_BONDS".to_string(), AssetType::Safe);
        strategy.portfolio.asset_types.insert("BITCOIN".to_string(), AssetType::Volatile);
        
        let market_data = create_test_barbell_market_data();
        let risk_metrics = strategy.risk_metrics(&market_data).unwrap();
        
        assert!(risk_metrics.volatility >= 0.0);
        assert!(risk_metrics.max_drawdown >= 0.0);
        assert!(risk_metrics.var_95 <= 0.0); // VaR should be negative
        assert!(risk_metrics.cvar_95 <= risk_metrics.var_95); // CVaR should be worse than VaR
        assert!(risk_metrics.antifragility_score >= 0.0);
        assert!(risk_metrics.black_swan_probability >= 0.0 && risk_metrics.black_swan_probability <= 1.0);
    }

    #[test]
    fn test_expected_return() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let mut strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        // Set up portfolio
        strategy.portfolio.weights.insert("TREASURY_BONDS".to_string(), 0.8);
        strategy.portfolio.weights.insert("BITCOIN".to_string(), 0.2);
        
        let market_data = create_test_barbell_market_data();
        let expected_return = strategy.expected_return(&market_data).unwrap();
        
        assert!(expected_return.is_finite());
        // Expected return should be reasonable (between safe and risky asset returns)
        assert!(expected_return > 0.0 && expected_return < 0.5);
    }

    #[test]
    fn test_robustness_assessment() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let mut strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        // Set up portfolio
        strategy.portfolio.weights.insert("TREASURY_BONDS".to_string(), 0.8);
        strategy.portfolio.weights.insert("BITCOIN".to_string(), 0.2);
        
        // Create test scenarios
        let mut scenarios = Vec::new();
        
        // Scenario 1: Market crash
        let mut crash_scenario = MarketScenario {
            name: "Market Crash".to_string(),
            price_shocks: HashMap::new(),
            duration_days: 30,
        };
        crash_scenario.price_shocks.insert("TREASURY_BONDS".to_string(), 0.05);
        crash_scenario.price_shocks.insert("BITCOIN".to_string(), -0.5);
        scenarios.push(crash_scenario);
        
        // Scenario 2: Recovery
        let mut recovery_scenario = MarketScenario {
            name: "Recovery".to_string(),
            price_shocks: HashMap::new(),
            duration_days: 90,
        };
        recovery_scenario.price_shocks.insert("TREASURY_BONDS".to_string(), 0.02);
        recovery_scenario.price_shocks.insert("BITCOIN".to_string(), 0.3);
        scenarios.push(recovery_scenario);
        
        let robustness = strategy.robustness_assessment(&scenarios).unwrap();
        
        assert!(robustness.robustness_score >= 0.0 && robustness.robustness_score <= 1.0);
        assert!(robustness.worst_case_performance <= robustness.best_case_performance);
        assert_eq!(robustness.stress_performance.len(), scenarios.len());
    }

    #[test]
    fn test_parameter_updates() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let mut strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        let new_params = BarbellParams {
            safe_target: 0.85,
            risky_target: 0.15,
            ..Default::default()
        };
        
        strategy.update_barbell_params(new_params.clone()).unwrap();
        
        assert_eq!(strategy.get_barbell_params().safe_target, 0.85);
        assert_eq!(strategy.get_barbell_params().risky_target, 0.15);
        assert_eq!(strategy.safe_allocation, 0.85);
        assert_eq!(strategy.risky_allocation, 0.15);
    }

    #[test]
    fn test_strategy_reset() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let mut strategy = BarbellStrategy::new("test_barbell", config, params.clone()).unwrap();
        
        // Modify strategy state
        strategy.portfolio.weights.insert("TEST".to_string(), 0.5);
        strategy.safe_allocation = 0.9;
        strategy.risky_allocation = 0.1;
        
        // Reset strategy
        strategy.reset();
        
        assert!(strategy.portfolio.weights.is_empty());
        assert_eq!(strategy.safe_allocation, params.safe_target);
        assert_eq!(strategy.risky_allocation, params.risky_target);
        assert!(strategy.get_performance_history().is_empty());
    }

    #[test]
    fn test_strategy_validation() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        // Valid strategy should pass validation
        assert!(strategy.validate().is_ok());
        
        // Test strategy with invalid allocations
        let mut invalid_strategy = strategy;
        invalid_strategy.safe_allocation = 0.8;
        invalid_strategy.risky_allocation = 0.5; // Sum > 1.0
        
        assert!(invalid_strategy.validate().is_err());
        
        // Test strategy with negative allocations
        invalid_strategy.safe_allocation = -0.1;
        invalid_strategy.risky_allocation = 0.5;
        
        assert!(invalid_strategy.validate().is_err());
    }

    #[test]
    fn test_metadata() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        let metadata = strategy.metadata();
        
        assert_eq!(metadata.get("type").unwrap(), "BarbellStrategy");
        assert_eq!(metadata.get("id").unwrap(), "test_barbell");
        assert!(metadata.contains_key("safe_allocation"));
        assert!(metadata.contains_key("risky_allocation"));
        assert!(metadata.contains_key("num_positions"));
        assert!(metadata.contains_key("performance_records"));
    }

    #[test]
    fn test_stress_response() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let mut strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        // Set up initial portfolio
        strategy.portfolio.weights.insert("TREASURY_BONDS".to_string(), 0.8);
        strategy.portfolio.weights.insert("BITCOIN".to_string(), 0.2);
        strategy.portfolio.asset_types.insert("TREASURY_BONDS".to_string(), AssetType::Safe);
        strategy.portfolio.asset_types.insert("BITCOIN".to_string(), AssetType::Volatile);
        
        let initial_safe = strategy.safe_allocation;
        let initial_risky = strategy.risky_allocation;
        
        // Apply stress conditions
        let stressed_market = create_stressed_market_data();
        strategy.update_strategy(&stressed_market).unwrap();
        
        // Strategy should adapt to stress
        assert!(strategy.safe_allocation >= initial_safe);
        assert!(strategy.risky_allocation <= initial_risky);
        
        // Performance should be recorded
        assert!(!strategy.get_performance_history().is_empty());
    }

    #[test]
    fn test_convexity_exposure_calculation() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let mut strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        // Set up portfolio with different asset types
        strategy.portfolio.weights.insert("TREASURY_BONDS".to_string(), 0.4);
        strategy.portfolio.weights.insert("OPTIONS".to_string(), 0.3);
        strategy.portfolio.weights.insert("ANTIFRAGILE_ASSET".to_string(), 0.2);
        strategy.portfolio.weights.insert("CASH".to_string(), 0.1);
        
        strategy.portfolio.asset_types.insert("TREASURY_BONDS".to_string(), AssetType::Safe);
        strategy.portfolio.asset_types.insert("OPTIONS".to_string(), AssetType::Derivative);
        strategy.portfolio.asset_types.insert("ANTIFRAGILE_ASSET".to_string(), AssetType::Antifragile);
        strategy.portfolio.asset_types.insert("CASH".to_string(), AssetType::Safe);
        
        let convexity_exposure = strategy.calculate_convexity_exposure();
        
        // Should have positive convexity from derivatives and antifragile assets
        assert!(convexity_exposure > 0.0);
        
        // Options should contribute most (weight 0.3 * multiplier 3.0 = 0.9)
        // Antifragile should contribute (weight 0.2 * multiplier 2.0 = 0.4)
        // Expected total ≈ 1.3
        assert!(convexity_exposure > 1.0);
    }

    #[test]
    fn test_safety_score_calculation() {
        let config = StrategyConfig::default();
        let params = BarbellParams::default();
        let mut strategy = BarbellStrategy::new("test_barbell", config, params).unwrap();
        
        // Set up all-safe portfolio
        strategy.portfolio.weights.insert("TREASURY_BONDS".to_string(), 0.8);
        strategy.portfolio.weights.insert("CASH".to_string(), 0.2);
        strategy.portfolio.asset_types.insert("TREASURY_BONDS".to_string(), AssetType::Safe);
        strategy.portfolio.asset_types.insert("CASH".to_string(), AssetType::Safe);
        
        let all_safe_score = strategy.calculate_safety_score();
        assert!((all_safe_score - 1.0).abs() < 0.1); // Should be close to 1.0
        
        // Set up all-risky portfolio
        strategy.portfolio.weights.clear();
        strategy.portfolio.asset_types.clear();
        strategy.portfolio.weights.insert("DERIVATIVES".to_string(), 1.0);
        strategy.portfolio.asset_types.insert("DERIVATIVES".to_string(), AssetType::Derivative);
        
        let all_risky_score = strategy.calculate_safety_score();
        assert_eq!(all_risky_score, 0.0); // Derivatives have 0.0 safety multiplier
        
        // Mixed portfolio should have intermediate score
        strategy.portfolio.weights.clear();
        strategy.portfolio.asset_types.clear();
        strategy.portfolio.weights.insert("TREASURY_BONDS".to_string(), 0.5);
        strategy.portfolio.weights.insert("DERIVATIVES".to_string(), 0.5);
        strategy.portfolio.asset_types.insert("TREASURY_BONDS".to_string(), AssetType::Safe);
        strategy.portfolio.asset_types.insert("DERIVATIVES".to_string(), AssetType::Derivative);
        
        let mixed_score = strategy.calculate_safety_score();
        assert!(mixed_score > 0.0 && mixed_score < 1.0);
        assert!((mixed_score - 0.5).abs() < 0.1); // Should be around 0.5
    }
}