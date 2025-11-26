//! Integration tests for multi-market crate

#[cfg(test)]
mod sports_tests {
    use multi_market::sports::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_kelly_optimizer_integration() {
        let optimizer = KellyOptimizer::new(dec!(10000), dec!(0.25));

        let opportunity = BettingOpportunity {
            event_id: "test".to_string(),
            outcome: "Home Win".to_string(),
            odds: dec!(2.5),
            win_probability: dec!(0.5),
            max_stake: None,
        };

        let result = optimizer.calculate(&opportunity).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_syndicate_integration() {
        let mut syndicate = Syndicate::new(
            "test".to_string(),
            "Test Syndicate".to_string(),
            "Test".to_string(),
        );

        syndicate
            .add_member(
                "member1".to_string(),
                "John".to_string(),
                "john@test.com".to_string(),
                MemberRole::Manager,
                dec!(1000),
            )
            .unwrap();

        assert_eq!(syndicate.total_capital, dec!(1000));
    }
}

#[cfg(test)]
mod prediction_tests {
    use multi_market::prediction::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_expected_value_calculator() {
        let calculator = ExpectedValueCalculator::new(dec!(10000));

        // Would need mock Market data
        assert_eq!(calculator.bankroll(), dec!(10000));
    }

    #[test]
    fn test_orderbook_analyzer() {
        let analyzer = OrderbookAnalyzer::new();

        use multi_market::prediction::orderbook::OrderLevel;
        let bids = vec![OrderLevel { price: dec!(0.5), size: dec!(100) }];
        let asks = vec![OrderLevel { price: dec!(0.6), size: dec!(100) }];

        let depth = analyzer.analyze_depth(&bids, &asks).unwrap();
        assert_eq!(depth.spread, dec!(0.1));
    }
}

#[cfg(test)]
mod crypto_tests {
    use multi_market::crypto::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_defi_manager() {
        let manager = DefiManager::new();
        assert!(manager.estimate_gas_cost("ethereum").is_ok());
    }

    #[test]
    fn test_arbitrage_engine() {
        let engine = ArbitrageEngine::new(dec!(1.0));

        let opp = engine.detect_arbitrage(
            "BTC/USD",
            dec!(50000),
            dec!(49500),
            "binance",
            "coinbase",
        );

        assert!(opp.is_some());
    }

    #[test]
    fn test_gas_optimizer() {
        let optimizer = GasOptimizer::new(dec!(3000));
        let estimate = optimizer.estimate_gas("swap");

        assert!(estimate.gas_limit > 0);
        assert!(estimate.total_cost_usd > dec!(0));
    }
}
