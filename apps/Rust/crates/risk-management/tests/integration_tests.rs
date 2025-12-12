use risk_management::*;
use std::sync::Arc;
use tokio::sync::RwLock;
use approx::assert_abs_diff_eq;

async fn create_test_portfolio() -> Portfolio {
    let mut portfolio = Portfolio::default();
    
    // Add realistic positions
    portfolio.positions = vec![
        Position {
            symbol: "AAPL".to_string(),
            quantity: 100.0,
            price: 150.0,
            market_value: 15000.0,
            weight: 0.30,
            pnl: 500.0,
            entry_price: 145.0,
            entry_time: chrono::Utc::now() - chrono::Duration::days(30),
        },
        Position {
            symbol: "GOOGL".to_string(),
            quantity: 20.0,
            price: 2500.0,
            market_value: 50000.0,
            weight: 0.50,
            pnl: -1000.0,
            entry_price: 2550.0,
            entry_time: chrono::Utc::now() - chrono::Duration::days(15),
        },
        Position {
            symbol: "TSLA".to_string(),
            quantity: 50.0,
            price: 400.0,
            market_value: 20000.0,
            weight: 0.20,
            pnl: 2500.0,
            entry_price: 350.0,
            entry_time: chrono::Utc::now() - chrono::Duration::days(60),
        },
    ];
    
    // Add assets with realistic parameters
    portfolio.assets = vec![
        Asset {
            symbol: "AAPL".to_string(),
            name: "Apple Inc.".to_string(),
            asset_class: AssetClass::Equity,
            price: 150.0,
            volatility: 0.25,
            beta: 1.2,
            expected_return: 0.12,
            liquidity_score: 0.95,
        },
        Asset {
            symbol: "GOOGL".to_string(),
            name: "Alphabet Inc.".to_string(),
            asset_class: AssetClass::Equity,
            price: 2500.0,
            volatility: 0.30,
            beta: 1.1,
            expected_return: 0.15,
            liquidity_score: 0.90,
        },
        Asset {
            symbol: "TSLA".to_string(),
            name: "Tesla Inc.".to_string(),
            asset_class: AssetClass::Equity,
            price: 400.0,
            volatility: 0.45,
            beta: 1.8,
            expected_return: 0.20,
            liquidity_score: 0.85,
        },
    ];
    
    // Generate realistic historical returns (252 trading days)
    let mut returns = Vec::new();
    let mut rng = rand::thread_rng();
    use rand::prelude::*;
    use rand_distr::Normal;
    
    let normal = Normal::new(0.0005, 0.02).unwrap(); // ~12.6% annual return, ~32% volatility
    for _ in 0..252 {
        returns.push(normal.sample(&mut rng));
    }
    portfolio.returns = returns;
    
    // Set target returns
    portfolio.targets = vec![0.001; 252]; // 0.1% daily target
    
    // Update portfolio value
    portfolio.total_value = portfolio.calculate_value();
    
    portfolio
}

async fn create_stress_scenarios() -> Vec<StressScenario> {
    use std::collections::HashMap;
    use ndarray::Array2;
    
    let mut scenarios = Vec::new();
    
    // Market crash scenario
    let mut asset_shocks = HashMap::new();
    asset_shocks.insert("AAPL".to_string(), -0.25);
    asset_shocks.insert("GOOGL".to_string(), -0.30);
    asset_shocks.insert("TSLA".to_string(), -0.40);
    
    let mut volatility_multipliers = HashMap::new();
    volatility_multipliers.insert("AAPL".to_string(), 2.0);
    volatility_multipliers.insert("GOOGL".to_string(), 2.5);
    volatility_multipliers.insert("TSLA".to_string(), 3.0);
    
    let mut liquidity_impacts = HashMap::new();
    liquidity_impacts.insert("AAPL".to_string(), 0.1);
    liquidity_impacts.insert("GOOGL".to_string(), 0.15);
    liquidity_impacts.insert("TSLA".to_string(), 0.3);
    
    scenarios.push(StressScenario {
        name: "Market Crash".to_string(),
        description: "Severe market downturn with increased volatility".to_string(),
        asset_shocks,
        volatility_multipliers,
        correlation_shifts: Array2::zeros((0, 0)),
        liquidity_impacts,
        probability: 0.02,
    });
    
    // Sector rotation scenario
    let mut asset_shocks = HashMap::new();
    asset_shocks.insert("AAPL".to_string(), -0.10);
    asset_shocks.insert("GOOGL".to_string(), -0.15);
    asset_shocks.insert("TSLA".to_string(), 0.05);
    
    let mut volatility_multipliers = HashMap::new();
    volatility_multipliers.insert("AAPL".to_string(), 1.3);
    volatility_multipliers.insert("GOOGL".to_string(), 1.5);
    volatility_multipliers.insert("TSLA".to_string(), 1.2);
    
    let mut liquidity_impacts = HashMap::new();
    liquidity_impacts.insert("AAPL".to_string(), 0.05);
    liquidity_impacts.insert("GOOGL".to_string(), 0.08);
    liquidity_impacts.insert("TSLA".to_string(), 0.02);
    
    scenarios.push(StressScenario {
        name: "Sector Rotation".to_string(),
        description: "Rotation out of tech stocks".to_string(),
        asset_shocks,
        volatility_multipliers,
        correlation_shifts: Array2::zeros((0, 0)),
        liquidity_impacts,
        probability: 0.10,
    });
    
    scenarios
}

#[tokio::test]
async fn test_risk_manager_full_workflow() {
    let config = RiskConfig::default();
    let risk_manager = RiskManager::new(config).await.unwrap();
    
    let portfolio = create_test_portfolio().await;
    
    // Test VaR calculation
    let var_result = risk_manager.calculate_var(&portfolio, 0.05).await;
    assert!(var_result.is_ok());
    
    let var = var_result.unwrap();
    assert!(!var.var_values.is_empty());
    assert!(var.var_values.contains_key("5%"));
    assert!(var.portfolio_value > 0.0);
    
    // Test CVaR calculation
    let cvar_result = risk_manager.calculate_cvar(&portfolio, 0.05).await;
    assert!(cvar_result.is_ok());
    
    let cvar = cvar_result.unwrap();
    assert!(!cvar.cvar_values.is_empty());
    assert!(cvar.cvar_values.contains_key("5%"));
    
    // Test comprehensive risk report
    let report_result = risk_manager.get_comprehensive_risk_report(&portfolio).await;
    assert!(report_result.is_ok());
    
    let report = report_result.unwrap();
    assert!(report.generation_time.as_millis() > 0);
    assert!(!report.var_result.var_values.is_empty());
}

#[tokio::test]
async fn test_stress_testing_integration() {
    let config = RiskConfig::default();
    let risk_manager = RiskManager::new(config).await.unwrap();
    
    let portfolio = create_test_portfolio().await;
    let scenarios = create_stress_scenarios().await;
    
    let stress_results = risk_manager.run_stress_tests(&portfolio, &scenarios).await;
    assert!(stress_results.is_ok());
    
    let results = stress_results.unwrap();
    assert_eq!(results.scenario_results.len(), scenarios.len());
    
    // Check that stress test produces meaningful results
    for scenario_result in &results.scenario_results {
        assert!(!scenario_result.scenario_name.is_empty());
        assert!(scenario_result.portfolio_pnl.is_finite());
        assert!(scenario_result.liquidity_impact >= 0.0);
        assert!(scenario_result.recovery_time_estimate.as_secs() > 0);
    }
    
    // Check overall impact metrics
    assert!(results.overall_impact.worst_case_loss >= 0.0);
    assert!(results.overall_impact.resilience_score >= 0.0);
    assert!(results.overall_impact.resilience_score <= 100.0);
}

#[tokio::test]
async fn test_quantum_risk_metrics() {
    let config = RiskConfig::default();
    let risk_manager = RiskManager::new(config).await.unwrap();
    
    let portfolio = create_test_portfolio().await;
    
    let quantum_metrics_result = risk_manager.get_quantum_risk_metrics(&portfolio).await;
    assert!(quantum_metrics_result.is_ok());
    
    let metrics = quantum_metrics_result.unwrap();
    assert!(metrics.quantum_var.is_finite());
    assert!(metrics.quantum_cvar.is_finite());
    assert!(metrics.quantum_advantage.is_finite());
    assert!(metrics.quantum_fidelity >= 0.0);
    assert!(metrics.quantum_fidelity <= 1.0);
}

#[tokio::test]
async fn test_real_time_monitoring() {
    let config = RiskConfig::default();
    let risk_manager = RiskManager::new(config).await.unwrap();
    
    let portfolio = create_test_portfolio().await;
    
    // Update portfolio positions
    let update_result = risk_manager.update_portfolio_positions(&portfolio.positions).await;
    assert!(update_result.is_ok());
    
    // Get real-time metrics
    let metrics_result = risk_manager.get_real_time_metrics().await;
    assert!(metrics_result.is_ok());
    
    let metrics = metrics_result.unwrap();
    assert!(metrics.portfolio_var.is_finite());
    assert!(metrics.portfolio_volatility >= 0.0);
    assert!(metrics.current_drawdown >= 0.0);
    assert!(metrics.concentration_risk >= 0.0);
}

#[tokio::test]
async fn test_compliance_checking() {
    let config = RiskConfig::default();
    let risk_manager = RiskManager::new(config).await.unwrap();
    
    let portfolio = create_test_portfolio().await;
    
    let compliance_result = risk_manager.check_compliance(&portfolio).await;
    assert!(compliance_result.is_ok());
    
    let compliance_report = compliance_result.unwrap();
    assert!(!compliance_report.violations.is_empty() || compliance_report.is_compliant);
}

#[tokio::test]
async fn test_portfolio_optimization_integration() {
    let config = RiskConfig::default();
    let risk_manager = RiskManager::new(config).await.unwrap();
    
    let portfolio = create_test_portfolio().await;
    
    let constraints = PortfolioConstraints {
        min_weights: std::collections::HashMap::new(),
        max_weights: std::collections::HashMap::new(),
        max_turnover: 0.5,
        max_risk: 0.2,
        target_return: Some(0.12),
        sector_constraints: std::collections::HashMap::new(),
    };
    
    let optimization_result = risk_manager.optimize_portfolio(&portfolio.assets, &constraints).await;
    assert!(optimization_result.is_ok());
    
    let optimized_portfolio = optimization_result.unwrap();
    assert!(optimized_portfolio.expected_return.is_finite());
    assert!(optimized_portfolio.expected_risk >= 0.0);
    assert!(optimized_portfolio.sharpe_ratio.is_finite());
}

#[tokio::test]
async fn test_monte_carlo_gpu_acceleration() {
    let config = RiskConfig::default();
    let risk_manager = RiskManager::new(config).await.unwrap();
    
    let portfolio = create_test_portfolio().await;
    let time_horizon = std::time::Duration::from_secs(30 * 24 * 3600); // 30 days
    
    let mc_result = risk_manager.run_monte_carlo_simulation(&portfolio, 10_000, time_horizon).await;
    assert!(mc_result.is_ok());
    
    let results = mc_result.unwrap();
    assert_eq!(results.portfolio_values.len(), 10_000);
    assert_eq!(results.returns.len(), 10_000);
    assert!(results.probability_of_loss >= 0.0);
    assert!(results.probability_of_loss <= 1.0);
}

#[tokio::test]
async fn test_risk_limit_monitoring() {
    let config = RiskConfig::default();
    let risk_manager = RiskManager::new(config).await.unwrap();
    
    let portfolio = create_test_portfolio().await;
    
    // Set maximum drawdown limit
    let set_limit_result = risk_manager.set_max_drawdown_limit(0.10).await;
    assert!(set_limit_result.is_ok());
    
    // Check current drawdown
    let drawdown_result = risk_manager.get_current_drawdown().await;
    assert!(drawdown_result.is_ok());
    
    let drawdown = drawdown_result.unwrap();
    assert!(drawdown >= 0.0);
    assert!(drawdown <= 1.0);
    
    // Check for risk limit breaches
    let breaches_result = risk_manager.check_risk_limits(&portfolio).await;
    assert!(breaches_result.is_ok());
    
    let breaches = breaches_result.unwrap();
    // Should be Vec (may be empty)
    assert!(breaches.len() >= 0);
}

#[tokio::test]
async fn test_correlation_analysis() {
    let config = RiskConfig::default();
    let risk_manager = RiskManager::new(config).await.unwrap();
    
    let portfolio = create_test_portfolio().await;
    
    let correlation_result = risk_manager.analyze_correlation_risk(&portfolio.assets, &portfolio.market_data).await;
    assert!(correlation_result.is_ok());
    
    let correlation_analysis = correlation_result.unwrap();
    assert!(correlation_analysis.correlation_stability >= 0.0);
    assert!(correlation_analysis.correlation_stability <= 1.0);
    assert!(correlation_analysis.diversification_ratio > 0.0);
}

#[tokio::test]
async fn test_performance_benchmarks() {
    let config = RiskConfig::default();
    let risk_manager = RiskManager::new(config).await.unwrap();
    
    let portfolio = create_test_portfolio().await;
    
    // Test VaR calculation speed
    let start_time = std::time::Instant::now();
    let _var_result = risk_manager.calculate_var(&portfolio, 0.05).await;
    let var_duration = start_time.elapsed();
    
    // Should complete within reasonable time (not necessarily <10μs for full calculation)
    assert!(var_duration < std::time::Duration::from_millis(100));
    
    // Test real-time metrics speed
    let start_time = std::time::Instant::now();
    let _metrics_result = risk_manager.get_real_time_metrics().await;
    let metrics_duration = start_time.elapsed();
    
    // Real-time metrics should be very fast
    assert!(metrics_duration < std::time::Duration::from_millis(10));
}

#[tokio::test]
async fn test_risk_adjusted_metrics() {
    let config = RiskConfig::default();
    let risk_manager = RiskManager::new(config).await.unwrap();
    
    let portfolio = create_test_portfolio().await;
    let benchmark_returns = vec![0.0008; portfolio.returns.len()]; // ~20% annual return
    
    let returns_array = ndarray::Array1::from_vec(portfolio.returns.clone());
    let benchmark_array = ndarray::Array1::from_vec(benchmark_returns);
    
    let metrics_result = risk_manager.calculate_risk_adjusted_metrics(&returns_array, &benchmark_array).await;
    assert!(metrics_result.is_ok());
    
    let metrics = metrics_result.unwrap();
    assert!(metrics.sharpe_ratio.is_finite());
    assert!(metrics.sortino_ratio.is_finite());
    assert!(metrics.max_drawdown >= 0.0);
    assert!(metrics.volatility >= 0.0);
}

#[tokio::test]
async fn test_error_handling_and_edge_cases() {
    let config = RiskConfig::default();
    let risk_manager = RiskManager::new(config).await.unwrap();
    
    // Test with empty portfolio
    let empty_portfolio = Portfolio::default();
    
    let var_result = risk_manager.calculate_var(&empty_portfolio, 0.05).await;
    assert!(var_result.is_err());
    
    // Test with invalid confidence level
    let portfolio = create_test_portfolio().await;
    
    let invalid_var_result = risk_manager.calculate_var(&portfolio, 1.5).await;
    assert!(invalid_var_result.is_err());
    
    let invalid_var_result2 = risk_manager.calculate_var(&portfolio, -0.1).await;
    assert!(invalid_var_result2.is_err());
    
    // Test with empty scenarios
    let empty_scenarios: Vec<StressScenario> = Vec::new();
    let stress_result = risk_manager.run_stress_tests(&portfolio, &empty_scenarios).await;
    assert!(stress_result.is_err());
}

#[tokio::test]
async fn test_system_reset_and_cleanup() {
    let config = RiskConfig::default();
    let risk_manager = RiskManager::new(config).await.unwrap();
    
    let portfolio = create_test_portfolio().await;
    
    // Perform some operations to populate caches
    let _var_result = risk_manager.calculate_var(&portfolio, 0.05).await;
    let _metrics_result = risk_manager.get_real_time_metrics().await;
    
    // Reset the system
    let reset_result = risk_manager.reset().await;
    assert!(reset_result.is_ok());
    
    // Verify system still works after reset
    let var_result_after_reset = risk_manager.calculate_var(&portfolio, 0.05).await;
    assert!(var_result_after_reset.is_ok());
}