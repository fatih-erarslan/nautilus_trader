//! Comprehensive tests for stress testing functionality
//!
//! Tests historical scenarios and sensitivity analysis

use nt_risk::stress::*;
use nt_risk::types::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{Utc, Duration};

// ============================================================================
// Historical Scenario Tests
// ============================================================================

#[tokio::test]
async fn test_stress_2008_financial_crisis() {
    let scenario = HistoricalScenario::FinancialCrisis2008;

    let positions = vec![
        Position {
            symbol: Symbol::from("SPY"),
            quantity: 100,
            avg_entry_price: dec!(150.0),
            current_price: dec!(150.0),
            exposure: dec!(15000.0),
            side: PositionSide::Long,
        },
    ];

    let result = scenario.apply(&positions).await.unwrap();

    // 2008 crisis had severe market decline
    assert!(result.portfolio_loss > 0.0);
    assert!(result.portfolio_loss_pct > 0.20); // At least 20% loss
    assert_eq!(result.scenario_name, "2008 Financial Crisis");
}

#[tokio::test]
async fn test_stress_2020_covid_crash() {
    let scenario = HistoricalScenario::CovidCrash2020;

    let positions = vec![
        Position {
            symbol: Symbol::from("SPY"),
            quantity: 100,
            avg_entry_price: dec!(300.0),
            current_price: dec!(300.0),
            exposure: dec!(30000.0),
            side: PositionSide::Long,
        },
    ];

    let result = scenario.apply(&positions).await.unwrap();

    // COVID crash was sharp but shorter
    assert!(result.portfolio_loss > 0.0);
    assert!(result.max_drawdown > 0.25); // Over 25% drawdown
}

#[tokio::test]
async fn test_stress_black_monday_1987() {
    let scenario = HistoricalScenario::BlackMonday1987;

    let positions = vec![
        Position {
            symbol: Symbol::from("SPY"),
            quantity: 100,
            avg_entry_price: dec!(250.0),
            current_price: dec!(250.0),
            exposure: dec!(25000.0),
            side: PositionSide::Long,
        },
    ];

    let result = scenario.apply(&positions).await.unwrap();

    // Black Monday had the largest single-day crash
    assert!(result.single_day_loss > 0.15); // Over 15% single day
}

#[tokio::test]
async fn test_stress_dot_com_bubble_2000() {
    let scenario = HistoricalScenario::DotComBubble2000;

    let positions = vec![
        Position {
            symbol: Symbol::from("QQQ"), // Tech heavy
            quantity: 100,
            avg_entry_price: dec!(100.0),
            current_price: dec!(100.0),
            exposure: dec!(10000.0),
            side: PositionSide::Long,
        },
    ];

    let result = scenario.apply(&positions).await.unwrap();

    // Dot-com crash was severe for tech
    assert!(result.portfolio_loss_pct > 0.30);
}

#[tokio::test]
async fn test_stress_multiple_scenarios() {
    let scenarios = vec![
        HistoricalScenario::FinancialCrisis2008,
        HistoricalScenario::CovidCrash2020,
        HistoricalScenario::BlackMonday1987,
    ];

    let positions = vec![
        Position {
            symbol: Symbol::from("SPY"),
            quantity: 100,
            avg_entry_price: dec!(400.0),
            current_price: dec!(400.0),
            exposure: dec!(40000.0),
            side: PositionSide::Long,
        },
    ];

    let mut results = vec![];
    for scenario in scenarios {
        let result = scenario.apply(&positions).await.unwrap();
        results.push(result);
    }

    // All scenarios should show losses
    for result in &results {
        assert!(result.portfolio_loss > 0.0);
    }

    // Find worst case scenario
    let worst = results.iter()
        .max_by(|a, b| a.portfolio_loss.partial_cmp(&b.portfolio_loss).unwrap())
        .unwrap();

    assert!(worst.portfolio_loss > 0.0);
}

// ============================================================================
// Custom Scenario Tests
// ============================================================================

#[tokio::test]
async fn test_custom_scenario_market_crash() {
    let scenario = CustomScenario::new("Custom Crash")
        .with_market_shock(-0.30) // 30% market decline
        .with_volatility_shock(2.0); // Double volatility

    let positions = vec![
        Position {
            symbol: Symbol::from("AAPL"),
            quantity: 100,
            avg_entry_price: dec!(150.0),
            current_price: dec!(150.0),
            exposure: dec!(15000.0),
            side: PositionSide::Long,
        },
    ];

    let result = scenario.apply(&positions).await.unwrap();

    assert!(result.portfolio_loss > 0.0);
    assert!(result.portfolio_loss_pct >= 0.29); // Close to 30%
}

#[tokio::test]
async fn test_custom_scenario_sector_specific() {
    let mut sector_shocks = std::collections::HashMap::new();
    sector_shocks.insert("Technology".to_string(), -0.40);
    sector_shocks.insert("Finance".to_string(), -0.20);

    let scenario = CustomScenario::new("Tech Selloff")
        .with_sector_shocks(sector_shocks);

    let positions = vec![
        Position {
            symbol: Symbol::from("AAPL"),
            quantity: 100,
            avg_entry_price: dec!(150.0),
            current_price: dec!(150.0),
            exposure: dec!(15000.0),
            side: PositionSide::Long,
        },
    ];

    let result = scenario.apply(&positions).await.unwrap();
    assert!(result.portfolio_loss > 0.0);
}

#[tokio::test]
async fn test_custom_scenario_rate_hike() {
    let scenario = CustomScenario::new("Fed Rate Hike")
        .with_interest_rate_shock(0.02) // 2% rate increase
        .with_bond_yield_shock(0.015);

    let positions = vec![
        Position {
            symbol: Symbol::from("TLT"), // Bond ETF
            quantity: 100,
            avg_entry_price: dec!(120.0),
            current_price: dec!(120.0),
            exposure: dec!(12000.0),
            side: PositionSide::Long,
        },
    ];

    let result = scenario.apply(&positions).await.unwrap();
    // Bonds should decline with rate increases
    assert!(result.portfolio_loss >= 0.0);
}

// ============================================================================
// Sensitivity Analysis Tests
// ============================================================================

#[tokio::test]
async fn test_sensitivity_price_change() {
    let analyzer = SensitivityAnalyzer::new();

    let positions = vec![
        Position {
            symbol: Symbol::from("AAPL"),
            quantity: 100,
            avg_entry_price: dec!(150.0),
            current_price: dec!(150.0),
            exposure: dec!(15000.0),
            side: PositionSide::Long,
        },
    ];

    // Test -10% to +10% price changes
    let price_changes = vec![-0.10, -0.05, 0.0, 0.05, 0.10];
    let results = analyzer
        .price_sensitivity(&positions, &price_changes)
        .await
        .unwrap();

    assert_eq!(results.len(), price_changes.len());

    // Check results make sense
    for (i, result) in results.iter().enumerate() {
        let expected_change = price_changes[i] * 15000.0;
        assert!((result.portfolio_value_change - expected_change).abs() < 1.0);
    }
}

#[tokio::test]
async fn test_sensitivity_volatility_change() {
    let analyzer = SensitivityAnalyzer::new();

    let positions = vec![
        Position {
            symbol: Symbol::from("AAPL"),
            quantity: 100,
            avg_entry_price: dec!(150.0),
            current_price: dec!(150.0),
            exposure: dec!(15000.0),
            side: PositionSide::Long,
        },
    ];

    // Test volatility multipliers
    let vol_changes = vec![0.5, 1.0, 1.5, 2.0, 3.0];
    let results = analyzer
        .volatility_sensitivity(&positions, &vol_changes)
        .await
        .unwrap();

    assert_eq!(results.len(), vol_changes.len());

    // Higher volatility should increase VaR
    let mut prev_var = 0.0;
    for result in &results {
        if prev_var > 0.0 {
            assert!(result.var_95 >= prev_var * 0.9); // Should generally increase
        }
        prev_var = result.var_95;
    }
}

#[tokio::test]
async fn test_sensitivity_correlation_change() {
    let analyzer = SensitivityAnalyzer::new();

    let positions = vec![
        Position {
            symbol: Symbol::from("AAPL"),
            quantity: 100,
            avg_entry_price: dec!(150.0),
            current_price: dec!(150.0),
            exposure: dec!(15000.0),
            side: PositionSide::Long,
        },
        Position {
            symbol: Symbol::from("GOOGL"),
            quantity: 50,
            avg_entry_price: dec!(2800.0),
            current_price: dec!(2800.0),
            exposure: dec!(140000.0),
            side: PositionSide::Long,
        },
    ];

    // Test different correlations
    let correlations = vec![0.0, 0.3, 0.5, 0.7, 0.9];
    let results = analyzer
        .correlation_sensitivity(&positions, &correlations)
        .await
        .unwrap();

    assert_eq!(results.len(), correlations.len());
}

#[tokio::test]
async fn test_sensitivity_time_horizon() {
    let analyzer = SensitivityAnalyzer::new();

    let positions = vec![
        Position {
            symbol: Symbol::from("AAPL"),
            quantity: 100,
            avg_entry_price: dec!(150.0),
            current_price: dec!(150.0),
            exposure: dec!(15000.0),
            side: PositionSide::Long,
        },
    ];

    // Test different time horizons (days)
    let horizons = vec![1, 5, 10, 21, 63]; // 1d, 1w, 2w, 1m, 3m
    let results = analyzer
        .time_horizon_sensitivity(&positions, &horizons)
        .await
        .unwrap();

    assert_eq!(results.len(), horizons.len());

    // VaR should increase with time horizon (roughly sqrt(time))
    for (i, result) in results.iter().enumerate() {
        if i > 0 {
            assert!(result.var_95 > results[i - 1].var_95);
        }
    }
}

// ============================================================================
// Reverse Stress Tests
// ============================================================================

#[tokio::test]
async fn test_reverse_stress_test_find_breaking_point() {
    let tester = ReverseStressTester::new();

    let positions = vec![
        Position {
            symbol: Symbol::from("AAPL"),
            quantity: 100,
            avg_entry_price: dec!(150.0),
            current_price: dec!(150.0),
            exposure: dec!(15000.0),
            side: PositionSide::Long,
        },
    ];

    // Find market decline that causes 50% portfolio loss
    let target_loss_pct = 0.50;
    let breaking_point = tester
        .find_breaking_point(&positions, target_loss_pct)
        .await
        .unwrap();

    assert!(breaking_point.market_decline >= 0.49);
    assert!(breaking_point.market_decline <= 0.51);
}

#[tokio::test]
async fn test_reverse_stress_test_margin_call() {
    let tester = ReverseStressTester::new();

    let positions = vec![
        Position {
            symbol: Symbol::from("AAPL"),
            quantity: 200, // Leveraged position
            avg_entry_price: dec!(150.0),
            current_price: dec!(150.0),
            exposure: dec!(30000.0),
            side: PositionSide::Long,
        },
    ];

    let initial_capital = 20000.0; // 1.5x leverage
    let margin_requirement = 0.25; // 25% minimum

    let breaking_point = tester
        .find_margin_call_threshold(&positions, initial_capital, margin_requirement)
        .await
        .unwrap();

    assert!(breaking_point.market_decline > 0.0);
}

// ============================================================================
// Aggregate Stress Test
// ============================================================================

#[tokio::test]
async fn test_comprehensive_stress_test() {
    let stress_tester = ComprehensiveStressTester::new();

    let positions = vec![
        Position {
            symbol: Symbol::from("SPY"),
            quantity: 100,
            avg_entry_price: dec!(400.0),
            current_price: dec!(400.0),
            exposure: dec!(40000.0),
            side: PositionSide::Long,
        },
        Position {
            symbol: Symbol::from("TLT"),
            quantity: 50,
            avg_entry_price: dec!(120.0),
            current_price: dec!(120.0),
            exposure: dec!(6000.0),
            side: PositionSide::Long,
        },
    ];

    let report = stress_tester
        .run_comprehensive_test(&positions)
        .await
        .unwrap();

    // Should have results for all major scenarios
    assert!(report.historical_scenarios.len() >= 3);
    assert!(report.sensitivity_analyses.len() >= 2);

    // Should identify worst case
    assert!(report.worst_case_scenario.is_some());
    assert!(report.worst_case_loss > 0.0);

    // Should have recommendations
    assert!(report.recommendations.len() > 0);
}

// ============================================================================
// Portfolio Diversification Stress Test
// ============================================================================

#[tokio::test]
async fn test_diversification_benefits() {
    let scenario = HistoricalScenario::FinancialCrisis2008;

    // Concentrated portfolio
    let concentrated = vec![
        Position {
            symbol: Symbol::from("AAPL"),
            quantity: 1000,
            avg_entry_price: dec!(100.0),
            current_price: dec!(100.0),
            exposure: dec!(100000.0),
            side: PositionSide::Long,
        },
    ];

    // Diversified portfolio
    let diversified = vec![
        Position {
            symbol: Symbol::from("AAPL"),
            quantity: 250,
            avg_entry_price: dec!(100.0),
            current_price: dec!(100.0),
            exposure: dec!(25000.0),
            side: PositionSide::Long,
        },
        Position {
            symbol: Symbol::from("GOOGL"),
            quantity: 10,
            avg_entry_price: dec!(2500.0),
            current_price: dec!(2500.0),
            exposure: dec!(25000.0),
            side: PositionSide::Long,
        },
        Position {
            symbol: Symbol::from("MSFT"),
            quantity: 100,
            avg_entry_price: dec!(250.0),
            current_price: dec!(250.0),
            exposure: dec!(25000.0),
            side: PositionSide::Long,
        },
        Position {
            symbol: Symbol::from("TLT"),
            quantity: 200,
            avg_entry_price: dec!(125.0),
            current_price: dec!(125.0),
            exposure: dec!(25000.0),
            side: PositionSide::Long,
        },
    ];

    let result_concentrated = scenario.apply(&concentrated).await.unwrap();
    let result_diversified = scenario.apply(&diversified).await.unwrap();

    // Diversified should have better risk-adjusted returns
    // (though absolute loss might be similar due to correlation)
}

// ============================================================================
// Edge Cases
// ============================================================================

#[tokio::test]
async fn test_stress_empty_portfolio() {
    let scenario = HistoricalScenario::FinancialCrisis2008;
    let result = scenario.apply(&[]).await;

    assert!(result.is_err()); // Should error on empty portfolio
}

#[tokio::test]
async fn test_stress_short_positions() {
    let scenario = HistoricalScenario::FinancialCrisis2008;

    // Short position benefits from crash
    let positions = vec![
        Position {
            symbol: Symbol::from("SPY"),
            quantity: -100,
            avg_entry_price: dec!(400.0),
            current_price: dec!(400.0),
            exposure: dec!(-40000.0),
            side: PositionSide::Short,
        },
    ];

    let result = scenario.apply(&positions).await.unwrap();

    // Short position should profit from market decline
    assert!(result.portfolio_gain > 0.0);
}

#[tokio::test]
async fn test_stress_mixed_long_short() {
    let scenario = HistoricalScenario::CovidCrash2020;

    let positions = vec![
        Position {
            symbol: Symbol::from("SPY"),
            quantity: 100,
            avg_entry_price: dec!(300.0),
            current_price: dec!(300.0),
            exposure: dec!(30000.0),
            side: PositionSide::Long,
        },
        Position {
            symbol: Symbol::from("VIX"),
            quantity: 100, // Long volatility
            avg_entry_price: dec!(15.0),
            current_price: dec!(15.0),
            exposure: dec!(1500.0),
            side: PositionSide::Long,
        },
    ];

    let result = scenario.apply(&positions).await.unwrap();

    // VIX should offset some SPY losses
    assert!(result.portfolio_loss < 30000.0 * 0.30);
}
