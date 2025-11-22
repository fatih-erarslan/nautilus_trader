use crate::algorithms::risk_management::*;
use std::collections::HashMap;

/// Comprehensive risk management tests implementing real quantitative finance models
/// Tests cover Kelly Criterion, ATR, VaR, portfolio optimization, and risk metrics validation

#[cfg(test)]
mod kelly_criterion_tests {
    use super::*;

    #[test]
    fn test_kelly_criterion_optimal_conditions() {
        let risk_manager = RiskManager::new(RiskParameters::default());

        // Test optimal Kelly conditions: 60% win rate, 2:1 risk-reward
        let position_size = risk_manager
            .calculate_kelly_position_size(
                "EURUSD", 0.60,      // 60% win rate
                200.0,     // Average win
                100.0,     // Average loss (b = 2.0)
                100_000.0, // Portfolio value
            )
            .unwrap();

        // Kelly fraction: f = (bp - q) / b = (2*0.6 - 0.4) / 2 = 0.4
        // But capped at max_position_size_pct (2%)
        let expected_max = 100_000.0 * 0.02; // 2% cap
        assert!(position_size > 0.0);
        assert!(position_size <= expected_max);
    }

    #[test]
    fn test_kelly_criterion_negative_expectancy() {
        let risk_manager = RiskManager::new(RiskParameters::default());

        // Negative expectancy: 30% win rate, 1:2 risk-reward
        let position_size = risk_manager
            .calculate_kelly_position_size(
                "GBPJPY", 0.30,  // 30% win rate
                100.0, // Average win
                200.0, // Average loss (b = 0.5)
                100_000.0,
            )
            .unwrap();

        // Kelly fraction: f = (0.5*0.3 - 0.7) / 0.5 = -1.1 (negative)
        // Should return 0 for negative Kelly
        assert_eq!(position_size, 0.0);
    }

    #[test]
    fn test_kelly_criterion_edge_cases() {
        let risk_manager = RiskManager::new(RiskParameters::default());

        // Test zero average loss
        let result = risk_manager
            .calculate_kelly_position_size("BTCUSD", 0.6, 100.0, 0.0, 100_000.0)
            .unwrap();
        assert_eq!(result, 0.0);

        // Test invalid win rates
        assert!(risk_manager
            .calculate_kelly_position_size("ETHUSD", 0.0, 100.0, 50.0, 100_000.0)
            .is_err());

        assert!(risk_manager
            .calculate_kelly_position_size("ETHUSD", 1.0, 100.0, 50.0, 100_000.0)
            .is_err());

        assert!(risk_manager
            .calculate_kelly_position_size("ETHUSD", 1.5, 100.0, 50.0, 100_000.0)
            .is_err());
    }

    #[test]
    fn test_kelly_mathematical_validation() {
        let risk_manager = RiskManager::new(RiskParameters::default());

        // Test various Kelly scenarios with mathematical precision
        let test_cases = vec![
            // (win_rate, avg_win, avg_loss, expected_kelly_fraction)
            (0.55, 100.0, 100.0, 0.10),  // f = (1*0.55 - 0.45) / 1 = 0.10
            (0.45, 150.0, 100.0, 0.175), // f = (1.5*0.45 - 0.55) / 1.5 = 0.175
            (0.70, 80.0, 120.0, 0.025),  // f = (0.667*0.7 - 0.3) / 0.667 = 0.025
        ];

        for (win_rate, avg_win, avg_loss, expected_fraction) in test_cases {
            let position_size = risk_manager
                .calculate_kelly_position_size("TEST", win_rate, avg_win, avg_loss, 100_000.0)
                .unwrap();

            let actual_fraction = position_size / 100_000.0;
            let capped_expected = expected_fraction.min(0.02); // 2% cap

            assert!(
                (actual_fraction - capped_expected).abs() < 0.001,
                "Win rate: {}, Expected: {}, Actual: {}",
                win_rate,
                capped_expected,
                actual_fraction
            );
        }
    }
}

#[cfg(test)]
mod atr_stop_loss_tests {
    use super::*;

    /// Generate realistic ATR values based on historical volatility patterns
    fn generate_realistic_atr(base_price: f64, volatility_pct: f64) -> f64 {
        base_price * (volatility_pct / 100.0) * (14.0_f64).sqrt() // 14-period ATR approximation
    }

    #[test]
    fn test_atr_stop_loss_with_real_volatility() {
        let risk_manager = RiskManager::new(RiskParameters::default());

        // EUR/USD typical volatility scenarios
        let test_cases = vec![
            // (symbol, entry_price, daily_vol_pct, multiplier, is_long)
            ("EURUSD", 1.0500, 0.8, 2.0, true),   // Low volatility
            ("GBPJPY", 150.00, 1.5, 2.5, false),  // Medium volatility
            ("XAUUSD", 2000.0, 2.2, 1.8, true),   // High volatility (Gold)
            ("BTCUSD", 45000.0, 4.5, 1.5, false), // Very high volatility
        ];

        for (symbol, entry_price, vol_pct, multiplier, is_long) in test_cases {
            let atr = generate_realistic_atr(entry_price, vol_pct);
            let stop_loss =
                risk_manager.calculate_atr_stop_loss(symbol, entry_price, atr, multiplier, is_long);

            if is_long {
                assert!(
                    stop_loss < entry_price,
                    "Long stop loss should be below entry for {}",
                    symbol
                );
                let distance = entry_price - stop_loss;
                let expected_distance = atr * multiplier;
                assert!(
                    (distance - expected_distance).abs() < 0.001,
                    "ATR distance mismatch for {}: expected {}, got {}",
                    symbol,
                    expected_distance,
                    distance
                );
            } else {
                assert!(
                    stop_loss > entry_price,
                    "Short stop loss should be above entry for {}",
                    symbol
                );
                let distance = stop_loss - entry_price;
                let expected_distance = atr * multiplier;
                assert!(
                    (distance - expected_distance).abs() < 0.001,
                    "ATR distance mismatch for {}: expected {}, got {}",
                    symbol,
                    expected_distance,
                    distance
                );
            }
        }
    }

    #[test]
    fn test_atr_extreme_volatility_scenarios() {
        let risk_manager = RiskManager::new(RiskParameters::default());

        // Test extreme market conditions
        let btc_price = 50000.0;
        let extreme_atr = 5000.0; // 10% daily ATR (extreme volatility)

        let stop_loss_long =
            risk_manager.calculate_atr_stop_loss("BTCUSD", btc_price, extreme_atr, 2.0, true);

        let stop_loss_short =
            risk_manager.calculate_atr_stop_loss("BTCUSD", btc_price, extreme_atr, 2.0, false);

        // Validate extreme scenarios
        assert_eq!(stop_loss_long, btc_price - (extreme_atr * 2.0));
        assert_eq!(stop_loss_short, btc_price + (extreme_atr * 2.0));

        // Test zero volatility
        let zero_atr_stop = risk_manager.calculate_atr_stop_loss("STABLE", 100.0, 0.0, 2.0, true);
        assert_eq!(zero_atr_stop, 100.0); // No movement with zero ATR
    }

    #[test]
    fn test_trailing_stop_mechanics() {
        let risk_manager = RiskManager::new(RiskParameters::default());

        // Test trailing stop progression for long position
        let entry_price = 100.0;
        let trail_pct = 2.0; // 2% trailing

        // Price moves favorably
        let favorable_prices = vec![102.0, 105.0, 108.0, 106.0, 109.0];
        let mut current_stop = None;

        for price in favorable_prices {
            let new_stop = risk_manager.calculate_trailing_stop(
                entry_price,
                price,
                trail_pct,
                true,
                current_stop,
            );

            // Stop should only move up for long positions
            if let Some(prev_stop) = current_stop {
                if price > prev_stop / (1.0 - trail_pct / 100.0) {
                    // Price moved up significantly
                    assert!(
                        new_stop >= prev_stop,
                        "Trailing stop should not move down: prev {}, new {}",
                        prev_stop,
                        new_stop
                    );
                }
            }

            current_stop = Some(new_stop);
        }

        // Test short position trailing
        let mut short_stop = None;
        let short_prices = vec![98.0, 95.0, 92.0, 94.0, 90.0];

        for price in short_prices {
            let new_stop = risk_manager.calculate_trailing_stop(
                entry_price,
                price,
                trail_pct,
                false,
                short_stop,
            );

            // Stop should only move down for short positions
            if let Some(prev_stop) = short_stop {
                if price < prev_stop / (1.0 + trail_pct / 100.0) {
                    // Price moved down significantly
                    assert!(
                        new_stop <= prev_stop,
                        "Short trailing stop should not move up: prev {}, new {}",
                        prev_stop,
                        new_stop
                    );
                }
            }

            short_stop = Some(new_stop);
        }
    }
}

#[cfg(test)]
mod var_calculation_tests {
    use super::*;

    /// Generate realistic return distribution with fat tails
    fn generate_realistic_returns(n: usize, mean_return: f64, volatility: f64) -> Vec<f64> {
        use std::f64::consts::PI;

        let mut returns = Vec::new();

        // Generate returns using Box-Muller transformation for normal distribution
        // Add fat tails to simulate real market conditions
        for i in 0..n {
            let u1: f64 = ((i + 1) as f64) / (n as f64 + 1.0);
            let u2: f64 = ((i * 7 + 13) % (n + 1)) as f64 / (n as f64 + 1.0);

            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            let return_val = mean_return + volatility * z;

            // Add occasional extreme events (fat tails)
            let extreme_return = if i % 50 == 0 {
                return_val * 3.0 // Simulate 2% probability of 3x extreme events
            } else {
                return_val
            };

            returns.push(extreme_return);
        }

        returns
    }

    #[test]
    fn test_var_95_calculation() {
        let mut risk_manager = RiskManager::new(RiskParameters::default());

        // Generate 252 days of realistic returns (1 year)
        let returns = generate_realistic_returns(252, 0.0008, 0.02); // 0.08% daily, 2% vol
        let portfolio_values: Vec<f64> = returns
            .iter()
            .scan(100_000.0, |acc, &ret| {
                *acc *= 1.0 + ret;
                Some(*acc)
            })
            .collect();

        // Populate portfolio history
        risk_manager.portfolio_history = portfolio_values.clone();

        let current_value = 100_000.0;
        let var_95 = risk_manager.calculate_var(0.95, current_value);

        // VaR should be positive and reasonable (1-10% of portfolio for 95% confidence)
        assert!(var_95 > 0.0);
        assert!(var_95 < current_value * 0.15); // Should not exceed 15%

        // VaR 99% should be higher than VaR 95%
        let var_99 = risk_manager.calculate_var(0.99, current_value);
        assert!(var_99 >= var_95, "VaR 99% should be >= VaR 95%");
    }

    #[test]
    fn test_var_with_extreme_scenarios() {
        let mut risk_manager = RiskManager::new(RiskParameters::default());

        // Create portfolio with known extreme losses
        let extreme_returns = vec![
            -0.05, -0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.05,
            -0.10, // -10% extreme loss
            0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.015, 0.025,
            -0.08, // -8% extreme loss
        ];

        let portfolio_values: Vec<f64> = extreme_returns
            .iter()
            .scan(100_000.0, |acc, &ret| {
                *acc *= 1.0 + ret;
                Some(*acc)
            })
            .collect();

        risk_manager.portfolio_history = portfolio_values;

        let var_95 = risk_manager.calculate_var(0.95, 100_000.0);

        // With extreme losses of 8% and 10%, VaR 95% should capture these
        assert!(
            var_95 >= 7000.0,
            "VaR should capture extreme losses: {}",
            var_95
        ); // At least 7%
        assert!(var_95 <= 12000.0, "VaR should be reasonable: {}", var_95); // Not more than 12%
    }

    #[test]
    fn test_var_insufficient_data() {
        let mut risk_manager = RiskManager::new(RiskParameters::default());

        // Test with insufficient historical data
        risk_manager.portfolio_history = vec![100_000.0, 101_000.0, 99_500.0]; // Only 3 points

        let var_95 = risk_manager.calculate_var(0.95, 100_000.0);
        assert_eq!(var_95, 0.0, "VaR should be 0 with insufficient data");
    }
}

#[cfg(test)]
mod portfolio_optimization_tests {
    use super::*;

    #[test]
    fn test_correlation_matrix_calculation() {
        let mut risk_manager = RiskManager::new(RiskParameters::default());

        // Create correlated asset returns
        let n = 50;
        let mut returns_a = Vec::new();
        let mut returns_b = Vec::new();
        let mut returns_c = Vec::new();

        // Generate correlated returns: A and B are highly correlated, C is independent
        for i in 0..n {
            let base_return = (i as f64 / n as f64 - 0.5) * 0.1; // -5% to +5% trend
            let noise = ((i * 17 + 7) % 100) as f64 / 1000.0 - 0.05; // Random noise

            let return_a = base_return + noise;
            let return_b = base_return * 0.8 + noise * 0.9; // 80% correlation with A
            let return_c = ((i * 23 + 13) % 100) as f64 / 1000.0 - 0.05; // Independent

            returns_a.push(return_a);
            returns_b.push(return_b);
            returns_c.push(return_c);
        }

        risk_manager
            .historical_returns
            .insert("ASSET_A".to_string(), returns_a);
        risk_manager
            .historical_returns
            .insert("ASSET_B".to_string(), returns_b);
        risk_manager
            .historical_returns
            .insert("ASSET_C".to_string(), returns_c);

        let correlations = risk_manager.calculate_correlation_matrix();

        // Verify correlation properties
        assert!(
            correlations.len() > 0,
            "Should have calculated correlations"
        );

        for ((_sym1, _sym2), correlation) in &correlations {
            assert!(
                *correlation >= -1.0 && *correlation <= 1.0,
                "Correlation should be between -1 and 1: {}",
                correlation
            );
        }

        // A and B should be more correlated than A and C
        let corr_ab = correlations
            .get(&("ASSET_A".to_string(), "ASSET_B".to_string()))
            .or_else(|| correlations.get(&("ASSET_B".to_string(), "ASSET_A".to_string())))
            .unwrap();

        let corr_ac = correlations
            .get(&("ASSET_A".to_string(), "ASSET_C".to_string()))
            .or_else(|| correlations.get(&("ASSET_C".to_string(), "ASSET_A".to_string())))
            .unwrap();

        assert!(
            corr_ab.abs() > corr_ac.abs(),
            "A-B correlation ({}) should be stronger than A-C correlation ({})",
            corr_ab,
            corr_ac
        );
    }

    #[test]
    fn test_correlation_risk_validation() {
        let mut risk_manager = RiskManager::new(RiskParameters {
            max_correlation: 0.5,
            ..RiskParameters::default()
        });

        // Create perfectly correlated assets (correlation = 1.0)
        let perfect_correlation = vec![0.01, 0.02, -0.01, 0.03, -0.02, 0.015];
        risk_manager
            .historical_returns
            .insert("HIGH_CORR_A".to_string(), perfect_correlation.clone());
        risk_manager
            .historical_returns
            .insert("HIGH_CORR_B".to_string(), perfect_correlation);

        let result = risk_manager.validate_correlation_risk();
        assert!(result.is_err(), "Should detect correlation limit violation");

        if let Err(RiskError::CorrelationLimitExceeded(corr)) = result {
            assert!(
                corr > 0.5,
                "Detected correlation should exceed limit: {}",
                corr
            );
        } else {
            panic!("Expected CorrelationLimitExceeded error");
        }
    }

    #[test]
    fn test_position_sizing_with_correlations() {
        let risk_manager = RiskManager::new(RiskParameters::default());

        // Test fixed fractional sizing for different risk scenarios
        let test_cases = vec![
            // (portfolio_value, risk_per_trade, entry_price, stop_loss, expected_approx_size)
            (100_000.0, 1.0, 100.0, 98.0, 500.0), // 1% risk, 2% stop = 500 shares
            (50_000.0, 2.0, 50.0, 47.5, 400.0),   // 2% risk, 5% stop = 400 shares
            (200_000.0, 0.5, 1000.0, 990.0, 100.0), // 0.5% risk, 1% stop = 100 shares
        ];

        for (portfolio_value, risk_pct, entry, stop, expected) in test_cases {
            let position_size = risk_manager
                .calculate_fixed_fractional_size(portfolio_value, risk_pct, entry, stop)
                .unwrap();

            let tolerance = expected * 0.01; // 1% tolerance
            assert!(
                (position_size - expected).abs() < tolerance,
                "Position size mismatch: expected {}, got {}",
                expected,
                position_size
            );
        }
    }

    #[test]
    fn test_fixed_fractional_edge_cases() {
        let risk_manager = RiskManager::new(RiskParameters::default());

        // Test invalid stop loss (same as entry)
        let result = risk_manager.calculate_fixed_fractional_size(100_000.0, 1.0, 100.0, 100.0);
        assert!(result.is_err(), "Should error with invalid stop loss");

        // Test stop loss beyond entry (invalid direction)
        let result = risk_manager.calculate_fixed_fractional_size(
            100_000.0, 1.0, 100.0, 102.0, // Stop above entry for implied long
        );
        assert!(result.is_ok(), "Should handle stop above entry"); // Size will be negative, handled by caller
    }
}

#[cfg(test)]
mod sharpe_sortino_tests {
    use super::*;

    fn create_test_portfolio_history(returns: Vec<f64>) -> Vec<f64> {
        returns
            .iter()
            .scan(100_000.0, |acc, &ret| {
                *acc *= 1.0 + ret;
                Some(*acc)
            })
            .collect()
    }

    #[test]
    fn test_sharpe_ratio_calculation() {
        let mut risk_manager = RiskManager::new(RiskParameters {
            risk_free_rate: 0.02, // 2% risk-free rate
            ..RiskParameters::default()
        });

        // Create portfolio with consistent positive returns
        let positive_returns = vec![0.001; 252]; // 0.1% daily return
        risk_manager.portfolio_history = create_test_portfolio_history(positive_returns);

        let sharpe = risk_manager.calculate_sharpe_ratio();

        // With 0.1% daily returns (25.2% annualized) and ~0% volatility
        // Sharpe should be very high
        assert!(
            sharpe > 10.0,
            "Sharpe ratio should be high for consistent returns: {}",
            sharpe
        );

        // Test with realistic mixed returns
        let mixed_returns = vec![
            0.02, -0.01, 0.015, -0.005, 0.01, -0.02, 0.025, -0.01, 0.005, -0.015, 0.01, 0.02,
            -0.01, 0.015, -0.005, 0.01, -0.02, 0.025, -0.01, 0.005,
        ];
        risk_manager.portfolio_history = create_test_portfolio_history(mixed_returns);

        let sharpe_mixed = risk_manager.calculate_sharpe_ratio();
        assert!(
            sharpe_mixed > 0.0,
            "Sharpe should be positive for overall positive returns"
        );
        assert!(
            sharpe_mixed < sharpe,
            "Mixed returns should have lower Sharpe than consistent returns"
        );
    }

    #[test]
    fn test_sortino_ratio_calculation() {
        let mut risk_manager = RiskManager::new(RiskParameters {
            risk_free_rate: 0.02,
            ..RiskParameters::default()
        });

        // Create returns with positive skew (more upside than downside)
        let skewed_returns = vec![
            0.03, 0.02, 0.01, -0.01, 0.025, 0.015, 0.005, -0.005, 0.02, 0.01, 0.035, 0.025, -0.002,
            0.015, 0.008, -0.008, 0.022, 0.012, 0.006, 0.018,
        ];

        risk_manager.portfolio_history = create_test_portfolio_history(skewed_returns);

        let sharpe = risk_manager.calculate_sharpe_ratio();
        let sortino = risk_manager.calculate_sortino_ratio();

        // Sortino should be higher than Sharpe for positively skewed returns
        assert!(
            sortino > sharpe,
            "Sortino ({}) should be higher than Sharpe ({}) for positively skewed returns",
            sortino,
            sharpe
        );

        // Test with all positive returns
        let all_positive = vec![0.01, 0.02, 0.015, 0.008, 0.012, 0.018, 0.005, 0.025];
        risk_manager.portfolio_history = create_test_portfolio_history(all_positive);

        let sortino_positive = risk_manager.calculate_sortino_ratio();
        assert_eq!(
            sortino_positive,
            f64::INFINITY,
            "Sortino should be infinite with no downside"
        );
    }

    #[test]
    fn test_zero_volatility_scenarios() {
        let mut risk_manager = RiskManager::new(RiskParameters::default());

        // Test zero volatility (all same returns)
        let zero_vol_returns = vec![0.005; 100]; // Same 0.5% return every period
        risk_manager.portfolio_history = create_test_portfolio_history(zero_vol_returns);

        let sharpe = risk_manager.calculate_sharpe_ratio();
        let sortino = risk_manager.calculate_sortino_ratio();

        assert_eq!(sharpe, 0.0, "Sharpe should be 0 with zero volatility");
        assert_eq!(
            sortino,
            f64::INFINITY,
            "Sortino should be infinite with no negative returns"
        );
    }

    #[test]
    fn test_insufficient_data_scenarios() {
        let mut risk_manager = RiskManager::new(RiskParameters::default());

        // Test with single data point
        risk_manager.portfolio_history = vec![100_000.0];

        assert_eq!(risk_manager.calculate_sharpe_ratio(), 0.0);
        assert_eq!(risk_manager.calculate_sortino_ratio(), 0.0);

        // Test with empty history
        risk_manager.portfolio_history = vec![];

        assert_eq!(risk_manager.calculate_sharpe_ratio(), 0.0);
        assert_eq!(risk_manager.calculate_sortino_ratio(), 0.0);
    }
}

#[cfg(test)]
mod drawdown_tests {
    use super::*;

    #[test]
    fn test_drawdown_calculation_and_limits() {
        let mut risk_manager = RiskManager::new(RiskParameters {
            max_drawdown_pct: 10.0, // 10% max drawdown
            ..RiskParameters::default()
        });

        // Start with high water mark
        let initial_value = 100_000.0;
        let result = risk_manager.calculate_max_drawdown(initial_value);
        assert!(result.is_ok());
        assert_eq!(risk_manager.high_water_mark, initial_value);

        // Portfolio grows - should update high water mark
        let higher_value = 110_000.0;
        let result = risk_manager.calculate_max_drawdown(higher_value);
        assert!(result.is_ok());
        assert_eq!(risk_manager.high_water_mark, higher_value);

        // Portfolio declines within limit
        let declined_value = 105_000.0; // 4.5% drawdown from 110k
        let drawdown = risk_manager.calculate_max_drawdown(declined_value).unwrap();
        let expected_drawdown = ((110_000.0 - 105_000.0) / 110_000.0) * 100.0;
        assert!((drawdown - expected_drawdown).abs() < 0.001);

        // Portfolio declines beyond limit - should error
        let excessive_decline = 95_000.0; // 13.6% drawdown from 110k
        let result = risk_manager.calculate_max_drawdown(excessive_decline);
        assert!(result.is_err());

        if let Err(RiskError::MaxDrawdownExceeded(dd)) = result {
            assert!(dd > 10.0, "Drawdown should exceed limit: {}%", dd);
        }
    }

    #[test]
    fn test_drawdown_recovery_scenarios() {
        let mut risk_manager = RiskManager::new(RiskParameters {
            max_drawdown_pct: 15.0,
            ..RiskParameters::default()
        });

        // Simulate portfolio journey with recovery
        let portfolio_values = vec![
            100_000.0, // Start
            120_000.0, // +20% (new high water mark)
            115_000.0, // -4.2% drawdown
            110_000.0, // -8.3% drawdown
            105_000.0, // -12.5% drawdown (still within 15% limit)
            115_000.0, // Recovery to -4.2% drawdown
            125_000.0, // New high water mark (+4.2% above previous)
        ];

        for (i, value) in portfolio_values.iter().enumerate() {
            let result = risk_manager.calculate_max_drawdown(*value);
            assert!(
                result.is_ok(),
                "Step {} should succeed with value {}",
                i,
                value
            );

            // Verify high water mark logic
            if *value > risk_manager.high_water_mark {
                assert_eq!(
                    risk_manager.high_water_mark, *value,
                    "High water mark should update at step {}",
                    i
                );
            }
        }

        // Final high water mark should be the highest value
        assert_eq!(risk_manager.high_water_mark, 125_000.0);
    }

    #[test]
    fn test_zero_initial_portfolio() {
        let mut risk_manager = RiskManager::new(RiskParameters::default());

        // Test with zero initial value
        let result = risk_manager.calculate_max_drawdown(0.0);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            0.0,
            "Drawdown should be 0 for zero portfolio"
        );
    }
}

#[cfg(test)]
mod position_management_tests {
    use super::*;

    fn create_test_position(symbol: &str, size: f64, entry: f64, current: f64) -> Position {
        Position {
            symbol: symbol.to_string(),
            size,
            entry_price: entry,
            current_price: current,
            unrealized_pnl: size * (current - entry),
            margin_used: (size * current).abs() / 10.0, // 10x leverage
            leverage: 10.0,
            timestamp: 0,
        }
    }

    #[test]
    fn test_position_update_and_validation() {
        let mut risk_manager = RiskManager::new(RiskParameters {
            max_position_size_pct: 5.0, // 5% max position size
            max_portfolio_risk_pct: 20.0,
            ..RiskParameters::default()
        });

        // Valid position within limits
        let position = create_test_position("EURUSD", 4.0, 1.1000, 1.1050);
        let result = risk_manager.update_position(position.clone());
        assert!(result.is_ok(), "Valid position should be accepted");

        // Verify position was stored
        assert!(risk_manager.positions.contains_key("EURUSD"));
        assert_eq!(risk_manager.positions["EURUSD"].size, 4.0);

        // Invalid position exceeding size limit
        let oversized_position = create_test_position("GBPUSD", 6.0, 1.2500, 1.2600);
        let result = risk_manager.update_position(oversized_position);
        assert!(result.is_err(), "Oversized position should be rejected");

        if let Err(RiskError::InvalidPositionSize(size)) = result {
            assert_eq!(size, 6.0, "Error should report correct size");
        }
    }

    #[test]
    fn test_portfolio_risk_validation() {
        let mut risk_manager = RiskManager::new(RiskParameters {
            max_portfolio_risk_pct: 25.0,
            ..RiskParameters::default()
        });

        // Add multiple positions within risk limits
        let positions = vec![
            create_test_position("EURUSD", 2.0, 1.1000, 1.1100),
            create_test_position("GBPUSD", 1.5, 1.2500, 1.2400),
            create_test_position("USDJPY", -1.8, 150.00, 149.50),
        ];

        for position in positions {
            let result = risk_manager.update_position(position);
            assert!(result.is_ok(), "Position within limits should be accepted");
        }

        assert_eq!(
            risk_manager.positions.len(),
            3,
            "All positions should be stored"
        );
    }

    #[test]
    fn test_risk_metrics_calculation() {
        let mut risk_manager = RiskManager::new(RiskParameters::default());

        // Add positions with known values
        let positions = vec![
            create_test_position("EURUSD", 1000.0, 1.1000, 1.1100), // +$100 P&L
            create_test_position("GBPUSD", -800.0, 1.2500, 1.2400), // +$80 P&L
            create_test_position("USDJPY", 500.0, 150.00, 149.00),  // -$500 P&L
        ];

        for position in positions {
            risk_manager.update_position(position).unwrap();
        }

        let metrics = risk_manager.get_risk_metrics();

        // Verify calculations
        let expected_portfolio_value = 1000.0 * 1.11 + (-800.0) * 1.24 + 500.0 * 149.0;
        assert!((metrics.portfolio_value - expected_portfolio_value).abs() < 1.0);

        let expected_unrealized_pnl = 100.0 + 80.0 - 500.0; // -$320
        assert!((metrics.unrealized_pnl - expected_unrealized_pnl).abs() < 1.0);

        assert!(metrics.total_margin_used > 0.0, "Margin should be used");
        assert_eq!(
            metrics.free_margin,
            metrics.portfolio_value - metrics.total_margin_used
        );
    }
}

#[cfg(test)]
mod risk_reward_tests {
    use super::*;

    #[test]
    fn test_risk_reward_ratio_calculations() {
        let risk_manager = RiskManager::new(RiskParameters::default());

        // Test various risk-reward scenarios
        let test_cases = vec![
            // (entry, stop_loss, take_profit, is_long, expected_ratio)
            (100.0, 95.0, 110.0, true, 2.0), // 5 risk, 10 reward = 2:1
            (100.0, 102.0, 96.0, false, 2.0), // Short: 2 risk, 4 reward = 2:1
            (50.0, 48.0, 56.0, true, 3.0),   // 2 risk, 6 reward = 3:1
            (200.0, 210.0, 185.0, false, 1.5), // Short: 10 risk, 15 reward = 1.5:1
        ];

        for (entry, stop, tp, is_long, expected) in test_cases {
            let ratio = risk_manager
                .calculate_risk_reward_ratio(entry, stop, tp, is_long)
                .unwrap();
            assert!(
                (ratio - expected).abs() < 0.001,
                "Risk-reward mismatch: entry={}, stop={}, tp={}, long={}, expected={}, got={}",
                entry,
                stop,
                tp,
                is_long,
                expected,
                ratio
            );
        }
    }

    #[test]
    fn test_risk_reward_edge_cases() {
        let risk_manager = RiskManager::new(RiskParameters::default());

        // Test zero risk scenario (stop = entry)
        let result = risk_manager.calculate_risk_reward_ratio(100.0, 100.0, 110.0, true);
        assert!(result.is_err(), "Zero risk should cause error");

        if let Err(RiskError::CalculationError(msg)) = result {
            assert!(msg.contains("Risk cannot be zero"));
        }

        // Test negative risk-reward (unfavorable setup)
        let ratio = risk_manager
            .calculate_risk_reward_ratio(
                100.0, // entry
                95.0,  // stop loss
                98.0,  // take profit (closer than stop)
                true,
            )
            .unwrap();

        // Risk = 5, Reward = -2, Ratio = -0.4
        assert!(
            ratio < 0.0,
            "Unfavorable setup should have negative ratio: {}",
            ratio
        );
        assert!((ratio - (-0.4)).abs() < 0.001);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_comprehensive_risk_scenario() {
        let mut risk_manager = RiskManager::new(RiskParameters {
            max_position_size_pct: 3.0,
            max_portfolio_risk_pct: 15.0,
            max_drawdown_pct: 12.0,
            max_leverage: 5.0,
            max_correlation: 0.6,
            kelly_lookback: 50,
            var_confidence: 0.05,
            risk_free_rate: 0.025,
        });

        // Simulate complete trading scenario
        let initial_portfolio = 250_000.0;

        // Step 1: Calculate position sizes using Kelly criterion
        let kelly_size_eur = risk_manager
            .calculate_kelly_position_size("EURUSD", 0.58, 150.0, 100.0, initial_portfolio)
            .unwrap();

        let kelly_size_gbp = risk_manager
            .calculate_kelly_position_size("GBPUSD", 0.52, 120.0, 110.0, initial_portfolio)
            .unwrap();

        assert!(kelly_size_eur > 0.0 && kelly_size_eur <= initial_portfolio * 0.03);
        assert!(kelly_size_gbp > 0.0 && kelly_size_gbp <= initial_portfolio * 0.03);

        // Step 2: Create positions with ATR-based stops
        let eur_entry = 1.0850;
        let eur_atr = 0.0080;
        let eur_stop =
            risk_manager.calculate_atr_stop_loss("EURUSD", eur_entry, eur_atr, 2.5, true);

        let gbp_entry = 1.2750;
        let gbp_atr = 0.0120;
        let gbp_stop =
            risk_manager.calculate_atr_stop_loss("GBPUSD", gbp_entry, gbp_atr, 2.0, true);

        // Step 3: Update positions and validate
        let eur_position = Position {
            symbol: "EURUSD".to_string(),
            size: kelly_size_eur / eur_entry, // Convert to units
            entry_price: eur_entry,
            current_price: 1.0900, // Favorable move
            unrealized_pnl: (kelly_size_eur / eur_entry) * (1.0900 - eur_entry),
            margin_used: kelly_size_eur / 5.0, // 5x leverage
            leverage: 5.0,
            timestamp: 1234567890,
        };

        let gbp_position = Position {
            symbol: "GBPUSD".to_string(),
            size: kelly_size_gbp / gbp_entry,
            entry_price: gbp_entry,
            current_price: 1.2700, // Slight unfavorable move
            unrealized_pnl: (kelly_size_gbp / gbp_entry) * (1.2700 - gbp_entry),
            margin_used: kelly_size_gbp / 5.0,
            leverage: 5.0,
            timestamp: 1234567891,
        };

        assert!(risk_manager.update_position(eur_position).is_ok());
        assert!(risk_manager.update_position(gbp_position).is_ok());

        // Step 4: Check risk metrics
        let metrics = risk_manager.get_risk_metrics();

        assert!(metrics.portfolio_value > 0.0);
        assert!(metrics.total_margin_used <= initial_portfolio);
        assert!(metrics.free_margin >= 0.0);

        // Step 5: Test drawdown management
        let current_portfolio = metrics.portfolio_value + metrics.unrealized_pnl;
        let drawdown_result = risk_manager.calculate_max_drawdown(current_portfolio);

        // Should succeed if within drawdown limits
        if current_portfolio >= initial_portfolio * 0.88 {
            // Within 12% drawdown
            assert!(drawdown_result.is_ok(), "Drawdown should be within limits");
        }

        // Step 6: Calculate risk-reward ratios
        let eur_rr = risk_manager
            .calculate_risk_reward_ratio(
                eur_entry, eur_stop, 1.1000, true, // Target 1.1000
            )
            .unwrap();

        let gbp_rr = risk_manager
            .calculate_risk_reward_ratio(
                gbp_entry, gbp_stop, 1.2950, true, // Target 1.2950
            )
            .unwrap();

        assert!(eur_rr > 0.0, "EUR risk-reward should be positive");
        assert!(gbp_rr > 0.0, "GBP risk-reward should be positive");

        println!("Integration test completed successfully:");
        println!("  Portfolio Value: ${:.2}", metrics.portfolio_value);
        println!("  Unrealized P&L: ${:.2}", metrics.unrealized_pnl);
        println!("  EUR Risk-Reward: {:.2}:1", eur_rr);
        println!("  GBP Risk-Reward: {:.2}:1", gbp_rr);
    }

    #[test]
    fn test_extreme_market_conditions() {
        let mut risk_manager = RiskManager::new(RiskParameters {
            max_drawdown_pct: 25.0, // Higher tolerance for stress test
            ..RiskParameters::default()
        });

        // Simulate market crash scenario
        let crash_returns = vec![
            -0.05, -0.08, -0.12, -0.06, -0.03, // Initial crash
            -0.02, 0.01, -0.04, -0.07, -0.09, // Continued volatility
            0.03, 0.05, 0.02, -0.01, 0.04, // Recovery starts
        ];

        let mut portfolio_value = 100_000.0;

        for (i, &return_rate) in crash_returns.iter().enumerate() {
            portfolio_value *= 1.0 + return_rate;

            let drawdown_result = risk_manager.calculate_max_drawdown(portfolio_value);

            // Even in extreme conditions, system should handle calculations
            match drawdown_result {
                Ok(dd) => {
                    assert!(dd >= 0.0, "Drawdown should be non-negative");
                    println!(
                        "Day {}: Portfolio ${:.2}, Drawdown {:.2}%",
                        i + 1,
                        portfolio_value,
                        dd
                    );
                }
                Err(RiskError::MaxDrawdownExceeded(dd)) => {
                    assert!(dd > 25.0, "Drawdown should exceed limit for error");
                    println!("Day {}: Max drawdown exceeded at {:.2}%", i + 1, dd);
                    // In real system, would trigger emergency procedures
                }
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        // System should survive extreme conditions
        let final_metrics = risk_manager.get_risk_metrics();
        assert!(final_metrics.var_95 >= 0.0, "VaR should be calculated");
        assert!(
            !final_metrics.sharpe_ratio.is_nan(),
            "Sharpe should be valid"
        );
    }
}

/// Performance benchmarks for risk calculations
#[cfg(test)]
mod performance_benchmarks {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore] // Run with: cargo test performance_benchmarks -- --ignored
    fn benchmark_risk_calculations() {
        let mut risk_manager = RiskManager::new(RiskParameters::default());

        // Populate with realistic data
        for i in 0..1000 {
            let return_rate = ((i * 17 + 7) % 200) as f64 / 10000.0 - 0.01; // -1% to +1%
            let portfolio_value = 100000.0 * (1.0 + return_rate);
            risk_manager.portfolio_history.push(portfolio_value);
        }

        let iterations = 10000;

        // Benchmark VaR calculation
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = risk_manager.calculate_var(0.95, 100_000.0);
        }
        let var_duration = start.elapsed();

        // Benchmark Sharpe ratio
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = risk_manager.calculate_sharpe_ratio();
        }
        let sharpe_duration = start.elapsed();

        // Benchmark Kelly criterion
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = risk_manager.calculate_kelly_position_size("TEST", 0.6, 100.0, 50.0, 100_000.0);
        }
        let kelly_duration = start.elapsed();

        println!("Performance Benchmarks ({} iterations):", iterations);
        println!(
            "  VaR Calculation: {:?} ({:.2} μs/op)",
            var_duration,
            var_duration.as_nanos() as f64 / iterations as f64 / 1000.0
        );
        println!(
            "  Sharpe Ratio: {:?} ({:.2} μs/op)",
            sharpe_duration,
            sharpe_duration.as_nanos() as f64 / iterations as f64 / 1000.0
        );
        println!(
            "  Kelly Criterion: {:?} ({:.2} μs/op)",
            kelly_duration,
            kelly_duration.as_nanos() as f64 / iterations as f64 / 1000.0
        );

        // Performance assertions - should complete within reasonable time
        assert!(var_duration.as_millis() < 1000, "VaR calculation too slow");
        assert!(
            sharpe_duration.as_millis() < 500,
            "Sharpe calculation too slow"
        );
        assert!(
            kelly_duration.as_millis() < 100,
            "Kelly calculation too slow"
        );
    }
}
