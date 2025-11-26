//! Comprehensive tests for VaR (Value at Risk) calculations
//!
//! Tests all three VaR methods:
//! - Monte Carlo VaR
//! - Historical VaR
//! - Parametric VaR

use nt_risk::var::*;
use nt_risk::types::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

// ============================================================================
// Monte Carlo VaR Tests
// ============================================================================

#[tokio::test]
async fn test_monte_carlo_var_creation() {
    let _config = VaRConfig {
        confidence_level: 0.95,
        time_horizon_days: 1,
        num_simulations: 1000,
        use_gpu: false,
    };

    let calculator = MonteCarloVaR::new(config);
    // Should not panic
}

#[tokio::test]
async fn test_monte_carlo_var_invalid_confidence() {
    let _config = VaRConfig {
        confidence_level: 1.5, // Invalid
        time_horizon_days: 1,
        num_simulations: 1000,
        use_gpu: false,
    };

    let calculator = MonteCarloVaR::new(config);
    // Should handle gracefully with warning
}

#[tokio::test]
async fn test_monte_carlo_var_calculation() {
    let _config = VaRConfig {
        confidence_level: 0.95,
        time_horizon_days: 1,
        num_simulations: 10_000,
        use_gpu: false,
    };

    let calculator = MonteCarloVaR::new(config);

    let positions = vec![
        Position {
            symbol: Symbol::from("AAPL"),
            quantity: 100,
            avg_entry_price: dec!(150.0),
            current_price: dec!(155.0),
            exposure: dec!(15500.0),
            side: PositionSide::Long,
        },
        Position {
            symbol: Symbol::from("GOOGL"),
            quantity: 50,
            avg_entry_price: dec!(2800.0),
            current_price: dec!(2850.0),
            exposure: dec!(142500.0),
            side: PositionSide::Long,
        },
    ];

    let result = calculator.calculate(&positions).await.unwrap();

    // VaR should be positive
    assert!(result.var_95 > 0.0);
    assert!(result.var_99 > 0.0);
    assert!(result.cvar_95 > 0.0);
    assert!(result.cvar_99 > 0.0);

    // CVaR should be greater than VaR
    assert!(result.cvar_95 >= result.var_95);
    assert!(result.cvar_99 >= result.var_99);

    // 99% VaR should be greater than 95% VaR
    assert!(result.var_99 >= result.var_95);
}

#[tokio::test]
async fn test_monte_carlo_var_empty_portfolio() {
    let _config = VaRConfig::default();
    let calculator = MonteCarloVaR::new(config);

    let result = calculator.calculate(&[]).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_monte_carlo_var_single_position() {
    let _config = VaRConfig {
        confidence_level: 0.95,
        time_horizon_days: 1,
        num_simulations: 5_000,
        use_gpu: false,
    };

    let calculator = MonteCarloVaR::new(config);

    let positions = vec![Position {
        symbol: Symbol::from("AAPL"),
        quantity: 100,
        avg_entry_price: dec!(150.0),
        current_price: dec!(150.0),
        exposure: dec!(15000.0),
        side: PositionSide::Long,
    }];

    let result = calculator.calculate(&positions).await.unwrap();
    assert!(result.var_95 > 0.0);
}

#[tokio::test]
async fn test_monte_carlo_var_different_horizons() {
    let positions = vec![Position {
        symbol: Symbol::from("AAPL"),
        quantity: 100,
        avg_entry_price: dec!(150.0),
        current_price: dec!(150.0),
        exposure: dec!(15000.0),
        side: PositionSide::Long,
    }];

    // 1-day VaR
    let config_1d = VaRConfig {
        confidence_level: 0.95,
        time_horizon_days: 1,
        num_simulations: 5_000,
        use_gpu: false,
    };
    let var_1d = MonteCarloVaR::new(config_1d)
        .calculate(&positions)
        .await
        .unwrap();

    // 10-day VaR
    let config_10d = VaRConfig {
        confidence_level: 0.95,
        time_horizon_days: 10,
        num_simulations: 5_000,
        use_gpu: false,
    };
    let var_10d = MonteCarloVaR::new(config_10d)
        .calculate(&positions)
        .await
        .unwrap();

    // 10-day VaR should be higher than 1-day VaR
    assert!(var_10d.var_95 > var_1d.var_95);
}

// ============================================================================
// Historical VaR Tests
// ============================================================================

#[tokio::test]
async fn test_historical_var_creation() {
    let historical_returns = vec![0.01, -0.02, 0.015, -0.01, 0.02];
    let calculator = HistoricalVaR::new(historical_returns, 0.95);
    // Should not panic
}

#[tokio::test]
async fn test_historical_var_calculation() {
    // Create sample returns (sorted for clarity)
    let mut returns = vec![];
    for i in 0..100 {
        let r = (i as f64 - 50.0) / 1000.0; // -0.05 to 0.05
        returns.push(r);
    }

    let calculator = HistoricalVaR::new(returns, 0.95);

    let positions = vec![Position {
        symbol: Symbol::from("AAPL"),
        quantity: 100,
        avg_entry_price: dec!(150.0),
        current_price: dec!(150.0),
        exposure: dec!(15000.0),
        side: PositionSide::Long,
    }];

    let result = calculator.calculate(&positions).await.unwrap();

    assert!(result.var_95 > 0.0);
    assert!(result.cvar_95 >= result.var_95);
}

#[tokio::test]
async fn test_historical_var_insufficient_data() {
    let returns = vec![0.01, -0.02]; // Too few data points
    let calculator = HistoricalVaR::new(returns, 0.95);

    let positions = vec![Position {
        symbol: Symbol::from("AAPL"),
        quantity: 100,
        avg_entry_price: dec!(150.0),
        current_price: dec!(150.0),
        exposure: dec!(15000.0),
        side: PositionSide::Long,
    }];

    let result = calculator.calculate(&positions).await;
    // Should either error or handle gracefully
}

// ============================================================================
// Parametric VaR Tests
// ============================================================================

#[tokio::test]
async fn test_parametric_var_creation() {
    let calculator = ParametricVaR::new(0.95);
    // Should not panic
}

#[tokio::test]
async fn test_parametric_var_calculation() {
    let calculator = ParametricVaR::new(0.95);

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

    let result = calculator.calculate(&positions).await.unwrap();

    assert!(result.var_95 > 0.0);
    assert!(result.cvar_95 >= result.var_95);
}

#[tokio::test]
async fn test_parametric_var_with_volatility() {
    let calculator = ParametricVaR::new(0.95)
        .with_volatility(0.02) // 2% daily volatility
        .with_correlation(0.5); // 50% correlation

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

    let result = calculator.calculate(&positions).await.unwrap();
    assert!(result.var_95 > 0.0);
}

// ============================================================================
// VaR Comparison Tests
// ============================================================================

#[tokio::test]
async fn test_var_methods_comparison() {
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

    // Monte Carlo
    let mc_config = VaRConfig {
        confidence_level: 0.95,
        time_horizon_days: 1,
        num_simulations: 10_000,
        use_gpu: false,
    };
    let mc_result = MonteCarloVaR::new(mc_config)
        .calculate(&positions)
        .await
        .unwrap();

    // Parametric
    let param_result = ParametricVaR::new(0.95)
        .calculate(&positions)
        .await
        .unwrap();

    // Both should give positive VaR
    assert!(mc_result.var_95 > 0.0);
    assert!(param_result.var_95 > 0.0);

    // Results should be in same order of magnitude
    let ratio = mc_result.var_95 / param_result.var_95;
    assert!(ratio > 0.1 && ratio < 10.0);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[tokio::test]
async fn test_var_with_zero_exposure() {
    let _config = VaRConfig::default();
    let calculator = MonteCarloVaR::new(config);

    let positions = vec![Position {
        symbol: Symbol::from("AAPL"),
        quantity: 0,
        avg_entry_price: dec!(150.0),
        current_price: dec!(150.0),
        exposure: dec!(0.0),
        side: PositionSide::Long,
    }];

    let result = calculator.calculate(&positions).await.unwrap();
    assert_eq!(result.var_95, 0.0);
}

#[tokio::test]
async fn test_var_with_negative_position() {
    let _config = VaRConfig::default();
    let calculator = MonteCarloVaR::new(config);

    let positions = vec![Position {
        symbol: Symbol::from("AAPL"),
        quantity: -100, // Short position
        avg_entry_price: dec!(150.0),
        current_price: dec!(150.0),
        exposure: dec!(-15000.0),
        side: PositionSide::Short,
    }];

    let result = calculator.calculate(&positions).await.unwrap();
    assert!(result.var_95 > 0.0); // VaR is always positive
}

// ============================================================================
// Stress Testing VaR
// ============================================================================

#[tokio::test]
async fn test_var_extreme_volatility() {
    let calculator = ParametricVaR::new(0.95)
        .with_volatility(0.50); // 50% daily volatility (extreme)

    let positions = vec![Position {
        symbol: Symbol::from("AAPL"),
        quantity: 100,
        avg_entry_price: dec!(150.0),
        current_price: dec!(150.0),
        exposure: dec!(15000.0),
        side: PositionSide::Long,
    }];

    let result = calculator.calculate(&positions).await.unwrap();
    // VaR should be very high with extreme volatility
    assert!(result.var_95 > 1000.0);
}

#[tokio::test]
async fn test_var_many_simulations() {
    let _config = VaRConfig {
        confidence_level: 0.99,
        time_horizon_days: 1,
        num_simulations: 100_000, // Large number
        use_gpu: false,
    };

    let calculator = MonteCarloVaR::new(config);

    let positions = vec![Position {
        symbol: Symbol::from("AAPL"),
        quantity: 100,
        avg_entry_price: dec!(150.0),
        current_price: dec!(150.0),
        exposure: dec!(15000.0),
        side: PositionSide::Long,
    }];

    let result = calculator.calculate(&positions).await.unwrap();
    assert!(result.var_99 > 0.0);
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[test]
fn test_var_config_default() {
    let _config = VaRConfig::default();
    assert_eq!(config.confidence_level, 0.95);
    assert_eq!(config.time_horizon_days, 1);
    assert_eq!(config.num_simulations, 10_000);
    assert_eq!(config.use_gpu, false);
}

#[test]
fn test_var_config_custom() {
    let _config = VaRConfig {
        confidence_level: 0.99,
        time_horizon_days: 5,
        num_simulations: 50_000,
        use_gpu: true,
    };

    assert_eq!(config.confidence_level, 0.99);
    assert_eq!(config.time_horizon_days, 5);
}

// ============================================================================
// Property-Based Tests
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_var_always_positive(
            quantity in 1..1000i64,
            price in 50.0..500.0f64,
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let _config = VaRConfig {
                    confidence_level: 0.95,
                    time_horizon_days: 1,
                    num_simulations: 1_000,
                    use_gpu: false,
                };

                let calculator = MonteCarloVaR::new(config);

                let positions = vec![Position {
                    symbol: Symbol::from("TEST"),
                    quantity,
                    avg_entry_price: Decimal::from_f64_retain(price).unwrap(),
                    current_price: Decimal::from_f64_retain(price).unwrap(),
                    exposure: Decimal::from_f64_retain(price * quantity as f64).unwrap(),
                    side: PositionSide::Long,
                }];

                let result = calculator.calculate(&positions).await.unwrap();
                prop_assert!(result.var_95 >= 0.0);
                prop_assert!(result.cvar_95 >= result.var_95);
            });
        }

        #[test]
        fn test_var_scales_with_position_size(
            multiplier in 1..10usize,
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let _config = VaRConfig {
                    confidence_level: 0.95,
                    time_horizon_days: 1,
                    num_simulations: 5_000,
                    use_gpu: false,
                };

                let calculator = MonteCarloVaR::new(config);

                let positions1 = vec![Position {
                    symbol: Symbol::from("TEST"),
                    quantity: 100,
                    avg_entry_price: dec!(100.0),
                    current_price: dec!(100.0),
                    exposure: dec!(10000.0),
                    side: PositionSide::Long,
                }];

                let positions2 = vec![Position {
                    symbol: Symbol::from("TEST"),
                    quantity: 100 * multiplier as i64,
                    avg_entry_price: dec!(100.0),
                    current_price: dec!(100.0),
                    exposure: Decimal::from(10000 * multiplier),
                    side: PositionSide::Long,
                }];

                let result1 = calculator.calculate(&positions1).await.unwrap();
                let result2 = calculator.calculate(&positions2).await.unwrap();

                // VaR should scale approximately linearly with position size
                let ratio = result2.var_95 / result1.var_95;
                prop_assert!(ratio > (multiplier as f64 * 0.8));
                prop_assert!(ratio < (multiplier as f64 * 1.2));
            });
        }
    }
}
