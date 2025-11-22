/// Integration tests for Python bridge with real finance calculations
///
/// Validates against peer-reviewed references:
/// - Hull, J. (2018). "Options, Futures, and Other Derivatives" (10th ed.)
/// - Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"

#[cfg(test)]
mod python_bridge_tests {
    use hyperphysics_finance::{
        FinanceSystem, FinanceConfig,
        OrderBookState,
        RiskMetrics,
        OptionParams, calculate_black_scholes,
        L2Snapshot, Price, Quantity,
    };
    use ndarray::Array1;
    use approx::assert_relative_eq;

    /// Test order book analytics calculation
    /// Validates that mid-price, spread, and imbalance are calculated correctly
    #[test]
    fn test_orderbook_analytics() {
        // Create test snapshot with known values
        let snapshot = L2Snapshot {
            symbol: "BTC-USD".to_string(),
            timestamp_us: 1000000,
            bids: vec![
                (Price::new(100.0).unwrap(), Quantity::new(5.0).unwrap()),
                (Price::new(99.5).unwrap(), Quantity::new(3.0).unwrap()),
            ],
            asks: vec![
                (Price::new(101.0).unwrap(), Quantity::new(4.0).unwrap()),
                (Price::new(101.5).unwrap(), Quantity::new(2.0).unwrap()),
            ],
        };

        let state = OrderBookState::from_snapshot(snapshot).unwrap();

        // Verify analytics
        assert_eq!(state.analytics.mid_price, 100.5);
        assert_eq!(state.analytics.spread, 1.0);
        assert_relative_eq!(state.analytics.relative_spread, 1.0 / 100.5, epsilon = 1e-10);

        // Bid volume = 5 + 3 = 8, Ask volume = 4 + 2 = 6
        // Imbalance = (8 - 6) / (8 + 6) = 2/14 ≈ 0.142857
        assert_relative_eq!(state.analytics.order_imbalance, 2.0/14.0, epsilon = 1e-6);
    }

    /// Test risk metrics calculation with synthetic data
    /// Verifies VaR, volatility, and other risk measures
    #[test]
    fn test_risk_metrics_calculation() {
        // Create returns with known statistical properties
        let returns: Vec<f64> = vec![
            0.01, -0.02, 0.015, -0.01, 0.005,
            -0.008, 0.012, -0.015, 0.02, -0.01,
            0.008, -0.012, 0.018, -0.005, 0.010,
            -0.015, 0.008, -0.01, 0.012, -0.008,
            0.015, -0.01, 0.005, -0.012, 0.018,
            -0.008, 0.010, -0.015, 0.012, -0.01,
        ];

        let returns_array = Array1::from(returns);
        let metrics = RiskMetrics::from_returns(returns_array.view(), 252.0).unwrap();

        // Verify metrics are reasonable
        assert!(metrics.volatility > 0.0, "Volatility should be positive");
        assert!(metrics.var_95 > 0.0, "VaR 95% should be positive");
        assert!(metrics.var_99 > 0.0, "VaR 99% should be positive");
        assert!(metrics.var_99 >= metrics.var_95, "VaR 99% should be >= VaR 95%");
        assert!(metrics.cvar_95 >= metrics.var_95, "CVaR should be >= VaR 95%");
    }

    /// Test Black-Scholes Greeks calculation
    /// Uses example from Hull (2018), Example 19.1
    #[test]
    fn test_greeks_hull_validation() {
        let params = OptionParams {
            spot: 42.0,
            strike: 40.0,
            rate: 0.10,
            volatility: 0.20,
            time_to_maturity: 0.5,  // 6 months
        };

        let (call_price, greeks) = calculate_black_scholes(&params).unwrap();

        // Hull Example 19.1: Call price should be approximately 4.76
        assert_relative_eq!(call_price, 4.76, epsilon = 0.01);

        // Validate Greeks properties
        assert!(greeks.delta > 0.0 && greeks.delta < 1.0, "Delta should be between 0 and 1 for calls");
        assert!(greeks.gamma > 0.0, "Gamma should be positive");
        assert!(greeks.vega > 0.0, "Vega should be positive");
        assert!(greeks.theta < 0.0, "Theta should be negative (time decay)");
        assert!(greeks.rho > 0.0, "Rho should be positive for calls");
    }

    /// Test at-the-money option Greeks
    /// Validates special case where spot = strike
    #[test]
    fn test_atm_greeks() {
        let params = OptionParams {
            spot: 100.0,
            strike: 100.0,
            rate: 0.05,
            volatility: 0.20,
            time_to_maturity: 1.0,
        };

        let (_call_price, greeks) = calculate_black_scholes(&params).unwrap();

        // ATM call delta should be around 0.5-0.65 (above 0.5 due to positive drift)
        assert!(greeks.delta > 0.5 && greeks.delta < 0.7,
            "ATM delta should be 0.5-0.7, got {}", greeks.delta);

        // Gamma should be highest for ATM options
        assert!(greeks.gamma > 0.01, "ATM gamma should be significant");
    }

    /// Test full system integration
    /// Simulates real market data processing through the entire pipeline
    #[test]
    fn test_finance_system_integration() {
        let mut system = FinanceSystem::new(FinanceConfig::default());

        // Process 100 snapshots with realistic price movements
        for i in 0..100 {
            let volatility = ((i as f64 * 0.1).sin() * 2.0);
            let price = 100.0 + (i as f64 * 0.05) + volatility;

            let snapshot = L2Snapshot {
                symbol: "BTC-USD".to_string(),
                timestamp_us: (1000000 + i * 1000) as u64,
                bids: vec![
                    (Price::new(price - 0.5).unwrap(), Quantity::new(1.0).unwrap()),
                ],
                asks: vec![
                    (Price::new(price + 0.5).unwrap(), Quantity::new(1.0).unwrap()),
                ],
            };

            system.process_snapshot(snapshot).unwrap();
        }

        // Verify system state after processing
        assert!(system.current_price().is_some(), "System should have current price");
        assert!(system.current_spread().is_some(), "System should have current spread");

        // Verify risk metrics can be calculated
        let metrics = system.calculate_risk_metrics().unwrap();
        assert!(metrics.volatility > 0.0, "System should calculate volatility");
        assert!(metrics.var_95 > 0.0, "System should calculate VaR");
        assert!(metrics.var_99 >= metrics.var_95, "VaR 99% >= VaR 95%");
    }

    /// Test error handling for invalid inputs
    #[test]
    fn test_invalid_inputs() {
        // Test invalid prices
        assert!(Price::new(-100.0).is_err());
        assert!(Price::new(f64::NAN).is_err());
        assert!(Price::new(f64::INFINITY).is_err());

        // Test invalid quantities
        assert!(Quantity::new(-1.0).is_err());
        assert!(Quantity::new(f64::NAN).is_err());

        // Test invalid Greeks parameters
        let invalid_params = OptionParams {
            spot: -100.0,  // Invalid
            strike: 100.0,
            rate: 0.05,
            volatility: 0.20,
            time_to_maturity: 1.0,
        };
        assert!(calculate_black_scholes(&invalid_params).is_err());
    }

    /// Performance benchmark: Order book processing
    #[test]
    fn test_orderbook_performance() {
        use std::time::Instant;

        let snapshot = L2Snapshot {
            symbol: "BTC-USD".to_string(),
            timestamp_us: 1000000,
            bids: vec![
                (Price::new(100.0).unwrap(), Quantity::new(1.0).unwrap()),
            ],
            asks: vec![
                (Price::new(101.0).unwrap(), Quantity::new(1.0).unwrap()),
            ],
        };

        let start = Instant::now();
        for _ in 0..10000 {
            let _ = OrderBookState::from_snapshot(snapshot.clone()).unwrap();
        }
        let elapsed = start.elapsed();

        // Should process at least 10,000 order books per second
        let ops_per_sec = 10000.0 / elapsed.as_secs_f64();
        assert!(ops_per_sec > 10000.0,
            "Performance too low: {} ops/sec", ops_per_sec);

        println!("Order book processing: {:.0} ops/sec", ops_per_sec);
    }

    /// Performance benchmark: Greeks calculation
    #[test]
    fn test_greeks_performance() {
        use std::time::Instant;

        let params = OptionParams {
            spot: 100.0,
            strike: 100.0,
            rate: 0.05,
            volatility: 0.20,
            time_to_maturity: 1.0,
        };

        let start = Instant::now();
        for _ in 0..10000 {
            let _ = calculate_black_scholes(&params).unwrap();
        }
        let elapsed = start.elapsed();

        // Should calculate at least 50,000 Greeks per second
        let ops_per_sec = 10000.0 / elapsed.as_secs_f64();
        assert!(ops_per_sec > 50000.0,
            "Performance too low: {} ops/sec", ops_per_sec);

        println!("Greeks calculation: {:.0} ops/sec", ops_per_sec);
    }
}
