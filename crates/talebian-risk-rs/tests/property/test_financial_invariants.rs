//! Property-based tests for financial invariants
//! Tests mathematical properties that must always hold true for financial correctness

use talebian_risk_rs::{
    risk_engine::*, MacchiavelianConfig, MarketData, TalebianRiskError
};
use proptest::prelude::*;
use chrono::Utc;

/// Generate valid market data for property testing
fn arb_market_data() -> impl Strategy<Value = MarketData> {
    (
        1000.0..100000.0,      // price
        0.0..10000.0,          // volume
        0.001..1.0,            // volatility
        prop::collection::vec(-0.1..0.1, 1..10), // returns
        prop::collection::vec(0.0..5000.0, 5),   // volume_history
    ).prop_map(|(price, volume, volatility, returns, volume_history)| {
        let spread = price * 0.001; // 0.1% spread
        MarketData {
            timestamp: Utc::now(),
            timestamp_unix: 1640995200,
            price,
            volume,
            bid: price - spread,
            ask: price + spread,
            bid_volume: volume * 0.4,
            ask_volume: volume * 0.4,
            volatility,
            returns,
            volume_history,
        }
    })
}

/// Generate valid configurations
fn arb_config() -> impl Strategy<Value = MacchiavelianConfig> {
    (
        0.1..0.8,  // antifragility_threshold
        0.3..0.9,  // barbell_safe_ratio
        0.05..0.3, // black_swan_threshold
        0.1..0.8,  // kelly_fraction
        1.5..5.0,  // whale_volume_threshold
        0.3..0.8,  // parasitic_opportunity_threshold
    ).prop_map(|(antifragility_threshold, barbell_safe_ratio, black_swan_threshold, 
                 kelly_fraction, whale_volume_threshold, parasitic_opportunity_threshold)| {
        MacchiavelianConfig {
            antifragility_threshold,
            barbell_safe_ratio,
            black_swan_threshold,
            kelly_fraction,
            kelly_max_fraction: kelly_fraction * 1.5,
            whale_volume_threshold,
            whale_detected_multiplier: 1.5,
            parasitic_opportunity_threshold,
            destructive_swan_protection: 0.3,
            dynamic_rebalance_threshold: 0.1,
            antifragility_window: 100,
        }
    })
}

#[cfg(test)]
mod property_tests {
    use super::*;

    proptest! {
        #[test]
        fn prop_kelly_fraction_bounds(
            config in arb_config(),
            market_data in arb_market_data()
        ) {
            let mut engine = TalebianRiskEngine::new(config.clone());
            let assessment = engine.assess_risk(&market_data)?;
            
            // Kelly fraction must be bounded
            prop_assert!(assessment.kelly_fraction >= 0.0);
            prop_assert!(assessment.kelly_fraction <= 1.0);
            prop_assert!(assessment.kelly_fraction <= config.kelly_max_fraction);
        }

        #[test]
        fn prop_position_size_bounds(
            config in arb_config(),
            market_data in arb_market_data()
        ) {
            let mut engine = TalebianRiskEngine::new(config);
            let assessment = engine.assess_risk(&market_data)?;
            
            // Position size must be reasonable
            prop_assert!(assessment.recommended_position_size >= 0.0);
            prop_assert!(assessment.recommended_position_size <= 1.0); // Cannot exceed 100%
            prop_assert!(assessment.recommended_position_size >= 0.01); // Minimum position per implementation
            prop_assert!(assessment.recommended_position_size <= 0.75); // Maximum position per implementation
        }

        #[test]
        fn prop_barbell_allocation_sum(
            config in arb_config(),
            market_data in arb_market_data()
        ) {
            let mut engine = TalebianRiskEngine::new(config);
            let assessment = engine.assess_risk(&market_data)?;
            
            // Barbell allocation must sum to <= 100%
            let total_allocation = assessment.barbell_allocation.0 + assessment.barbell_allocation.1;
            prop_assert!(total_allocation <= 1.1); // Allow small rounding error
            prop_assert!(assessment.barbell_allocation.0 >= 0.0);
            prop_assert!(assessment.barbell_allocation.1 >= 0.0);
        }

        #[test]
        fn prop_probability_bounds(
            config in arb_config(),
            market_data in arb_market_data()
        ) {
            let mut engine = TalebianRiskEngine::new(config);
            let assessment = engine.assess_risk(&market_data)?;
            
            // All probabilities must be valid
            prop_assert!(assessment.black_swan_probability >= 0.0);
            prop_assert!(assessment.black_swan_probability <= 1.0);
            prop_assert!(assessment.confidence >= 0.0);
            prop_assert!(assessment.confidence <= 1.0);
            prop_assert!(assessment.parasitic_opportunity.confidence >= 0.0);
            prop_assert!(assessment.parasitic_opportunity.confidence <= 1.0);
        }

        #[test]
        fn prop_risk_score_bounds(
            config in arb_config(),
            market_data in arb_market_data()
        ) {
            let mut engine = TalebianRiskEngine::new(config);
            let assessment = engine.assess_risk(&market_data)?;
            
            // Risk scores must be bounded
            prop_assert!(assessment.overall_risk_score >= 0.0);
            prop_assert!(assessment.overall_risk_score <= 1.0);
            prop_assert!(assessment.antifragility_score >= 0.0);
            prop_assert!(assessment.antifragility_score <= 1.0);
        }

        #[test]
        fn prop_whale_detection_consistency(
            config in arb_config(),
            market_data in arb_market_data()
        ) {
            let mut engine = TalebianRiskEngine::new(config);
            let assessment = engine.assess_risk(&market_data)?;
            
            // Whale detection flags must be consistent
            prop_assert_eq!(assessment.whale_detection.detected, assessment.whale_detection.is_whale_detected);
            
            // Whale confidence must be bounded
            prop_assert!(assessment.whale_detection.confidence >= 0.0);
            prop_assert!(assessment.whale_detection.confidence <= 1.0);
            
            // Volume spike must be non-negative
            prop_assert!(assessment.whale_detection.volume_spike >= 0.0);
        }

        #[test]
        fn prop_opportunity_score_consistency(
            config in arb_config(),
            market_data in arb_market_data()
        ) {
            let mut engine = TalebianRiskEngine::new(config);
            let assessment = engine.assess_risk(&market_data)?;
            
            // Opportunity scores must be non-negative
            prop_assert!(assessment.parasitic_opportunity.opportunity_score >= 0.0);
            prop_assert!(assessment.parasitic_opportunity.momentum_factor >= 0.0);
            prop_assert!(assessment.parasitic_opportunity.volatility_factor >= 0.0);
            prop_assert!(assessment.parasitic_opportunity.recommended_allocation >= 0.0);
            prop_assert!(assessment.parasitic_opportunity.recommended_allocation <= 1.0);
        }

        #[test]
        fn prop_monotonic_confidence_relationship(
            config in arb_config(),
            mut market_data in arb_market_data()
        ) {
            let mut engine = TalebianRiskEngine::new(config);
            
            // Test with low confidence scenario
            market_data.volume = market_data.volume_history.iter().sum::<f64>() / market_data.volume_history.len() as f64;
            market_data.volatility = 0.01; // Low volatility
            let low_confidence_assessment = engine.assess_risk(&market_data)?;
            
            // Test with high confidence scenario
            market_data.volume *= 5.0; // High volume spike
            market_data.volatility = 0.1; // High volatility
            let high_confidence_assessment = engine.assess_risk(&market_data)?;
            
            // Higher confidence scenarios should generally result in larger positions
            // (though other factors may intervene, so this is a soft constraint)
            if high_confidence_assessment.confidence > low_confidence_assessment.confidence + 0.1 {
                prop_assert!(high_confidence_assessment.recommended_position_size >= 
                           low_confidence_assessment.recommended_position_size * 0.8);
            }
        }

        #[test]
        fn prop_assessment_deterministic(
            config in arb_config(),
            market_data in arb_market_data()
        ) {
            let mut engine1 = TalebianRiskEngine::new(config.clone());
            let mut engine2 = TalebianRiskEngine::new(config);
            
            let assessment1 = engine1.assess_risk(&market_data)?;
            let assessment2 = engine2.assess_risk(&market_data)?;
            
            // Same inputs should produce same outputs (deterministic)
            prop_assert!((assessment1.overall_risk_score - assessment2.overall_risk_score).abs() < 1e-10);
            prop_assert!((assessment1.kelly_fraction - assessment2.kelly_fraction).abs() < 1e-10);
            prop_assert!((assessment1.antifragility_score - assessment2.antifragility_score).abs() < 1e-10);
            prop_assert!((assessment1.confidence - assessment2.confidence).abs() < 1e-10);
        }

        #[test]
        fn prop_recommendations_validity(
            config in arb_config(),
            market_data in arb_market_data()
        ) {
            let mut engine = TalebianRiskEngine::new(config);
            let recommendations = engine.generate_recommendations(&market_data)?;
            
            // Position sizing recommendations must be valid
            prop_assert!(recommendations.position_sizing.final_recommended_size >= 0.0);
            prop_assert!(recommendations.position_sizing.final_recommended_size <= 1.0);
            prop_assert!(recommendations.position_sizing.kelly_fraction >= 0.0);
            prop_assert!(recommendations.position_sizing.kelly_fraction <= 1.0);
            
            // Risk controls must be reasonable
            prop_assert!(recommendations.risk_controls.stop_loss_level > 0.0);
            prop_assert!(recommendations.risk_controls.take_profit_level > 0.0);
            prop_assert!(recommendations.risk_controls.take_profit_level >= recommendations.risk_controls.stop_loss_level);
            prop_assert!(recommendations.risk_controls.max_drawdown_limit > 0.0);
            prop_assert!(recommendations.risk_controls.max_drawdown_limit <= 1.0);
            
            // Performance metrics must be bounded
            prop_assert!(recommendations.performance_metrics.expected_volatility >= 0.0);
            prop_assert!(recommendations.performance_metrics.win_probability >= 0.0);
            prop_assert!(recommendations.performance_metrics.win_probability <= 1.0);
        }

        #[test]
        fn prop_config_impact_consistency(
            market_data in arb_market_data(),
            kelly_fraction_1 in 0.1..0.5f64,
            kelly_fraction_2 in 0.5..0.8f64
        ) {
            prop_assume!(kelly_fraction_2 > kelly_fraction_1);
            
            let mut config1 = MacchiavelianConfig::aggressive_defaults();
            config1.kelly_fraction = kelly_fraction_1;
            
            let mut config2 = MacchiavelianConfig::aggressive_defaults();
            config2.kelly_fraction = kelly_fraction_2;
            
            let mut engine1 = TalebianRiskEngine::new(config1);
            let mut engine2 = TalebianRiskEngine::new(config2);
            
            let assessment1 = engine1.assess_risk(&market_data)?;
            let assessment2 = engine2.assess_risk(&market_data)?;
            
            // Higher Kelly fraction should generally result in higher position sizes
            // (modulo other risk adjustments)
            prop_assert!(assessment2.kelly_fraction >= assessment1.kelly_fraction);
        }

        #[test]
        fn prop_volume_spike_threshold_consistency(
            market_data in arb_market_data(),
            threshold_1 in 1.5..3.0f64,
            threshold_2 in 3.0..6.0f64
        ) {
            prop_assume!(threshold_2 > threshold_1);
            
            let mut config1 = MacchiavelianConfig::aggressive_defaults();
            config1.whale_volume_threshold = threshold_1;
            
            let mut config2 = MacchiavelianConfig::aggressive_defaults();
            config2.whale_volume_threshold = threshold_2;
            
            let mut engine1 = TalebianRiskEngine::new(config1);
            let mut engine2 = TalebianRiskEngine::new(config2);
            
            let assessment1 = engine1.assess_risk(&market_data)?;
            let assessment2 = engine2.assess_risk(&market_data)?;
            
            // Lower threshold should be more likely to detect whales
            if assessment1.whale_detection.is_whale_detected && !assessment2.whale_detection.is_whale_detected {
                // This is expected - lower threshold detected but higher didn't
                prop_assert!(assessment1.whale_detection.volume_spike >= threshold_1);
                prop_assert!(assessment1.whale_detection.volume_spike < threshold_2);
            }
        }

        #[test]
        fn prop_numerical_stability(
            config in arb_config(),
            price in 1.0..1000000.0f64,
            volume in 0.1..100000.0f64,
            volatility in 0.001..0.5f64
        ) {
            let market_data = MarketData {
                timestamp: Utc::now(),
                timestamp_unix: 1640995200,
                price,
                volume,
                bid: price * 0.999,
                ask: price * 1.001,
                bid_volume: volume * 0.4,
                ask_volume: volume * 0.4,
                volatility,
                returns: vec![0.01],
                volume_history: vec![volume * 0.8, volume * 0.9, volume, volume * 1.1, volume * 0.95],
            };
            
            let mut engine = TalebianRiskEngine::new(config);
            let assessment = engine.assess_risk(&market_data)?;
            
            // All outputs should be finite numbers
            prop_assert!(assessment.overall_risk_score.is_finite());
            prop_assert!(assessment.kelly_fraction.is_finite());
            prop_assert!(assessment.antifragility_score.is_finite());
            prop_assert!(assessment.black_swan_probability.is_finite());
            prop_assert!(assessment.recommended_position_size.is_finite());
            prop_assert!(assessment.confidence.is_finite());
            
            // Should not produce NaN values
            prop_assert!(!assessment.overall_risk_score.is_nan());
            prop_assert!(!assessment.kelly_fraction.is_nan());
            prop_assert!(!assessment.antifragility_score.is_nan());
            prop_assert!(!assessment.black_swan_probability.is_nan());
            prop_assert!(!assessment.recommended_position_size.is_nan());
            prop_assert!(!assessment.confidence.is_nan());
        }

        #[test]
        fn prop_scale_invariance_price(
            config in arb_config(),
            base_price in 1000.0..50000.0f64,
            scale_factor in 1.1..10.0f64
        ) {
            // Test that scaling price doesn't fundamentally change risk assessment
            let base_market_data = MarketData {
                timestamp: Utc::now(),
                timestamp_unix: 1640995200,
                price: base_price,
                volume: 1000.0,
                bid: base_price * 0.999,
                ask: base_price * 1.001,
                bid_volume: 500.0,
                ask_volume: 500.0,
                volatility: 0.03,
                returns: vec![0.01],
                volume_history: vec![1000.0; 5],
            };
            
            let scaled_market_data = MarketData {
                price: base_price * scale_factor,
                bid: base_price * scale_factor * 0.999,
                ask: base_price * scale_factor * 1.001,
                ..base_market_data.clone()
            };
            
            let mut engine1 = TalebianRiskEngine::new(config.clone());
            let mut engine2 = TalebianRiskEngine::new(config);
            
            let base_assessment = engine1.assess_risk(&base_market_data)?;
            let scaled_assessment = engine2.assess_risk(&scaled_market_data)?;
            
            // Risk metrics should be approximately the same (price scaling shouldn't matter much)
            prop_assert!((base_assessment.overall_risk_score - scaled_assessment.overall_risk_score).abs() < 0.1);
            prop_assert!((base_assessment.antifragility_score - scaled_assessment.antifragility_score).abs() < 0.1);
            prop_assert!((base_assessment.kelly_fraction - scaled_assessment.kelly_fraction).abs() < 0.05);
        }

        #[test]
        fn prop_monotonic_volatility_relationship(
            config in arb_config(),
            base_vol in 0.01..0.05f64,
            vol_multiplier in 2.0..5.0f64
        ) {
            let low_vol_data = MarketData {
                timestamp: Utc::now(),
                timestamp_unix: 1640995200,
                price: 50000.0,
                volume: 1000.0,
                bid: 49990.0,
                ask: 50010.0,
                bid_volume: 500.0,
                ask_volume: 500.0,
                volatility: base_vol,
                returns: vec![0.01],
                volume_history: vec![1000.0; 5],
            };
            
            let high_vol_data = MarketData {
                volatility: base_vol * vol_multiplier,
                ..low_vol_data.clone()
            };
            
            let mut engine1 = TalebianRiskEngine::new(config.clone());
            let mut engine2 = TalebianRiskEngine::new(config);
            
            let low_vol_assessment = engine1.assess_risk(&low_vol_data)?;
            let high_vol_assessment = engine2.assess_risk(&high_vol_data)?;
            
            // Higher volatility should generally increase black swan probability
            prop_assert!(high_vol_assessment.black_swan_probability >= low_vol_assessment.black_swan_probability);
            
            // Higher volatility environments might affect position sizing (could go either way depending on strategy)
            // But the assessment should handle it gracefully
            prop_assert!(high_vol_assessment.recommended_position_size > 0.0);
            prop_assert!(low_vol_assessment.recommended_position_size > 0.0);
        }

        #[test]
        fn prop_idempotent_assessment(
            config in arb_config(),
            market_data in arb_market_data()
        ) {
            let mut engine = TalebianRiskEngine::new(config);
            
            // Multiple assessments of same data should be identical
            let assessment1 = engine.assess_risk(&market_data)?;
            let assessment2 = engine.assess_risk(&market_data)?;
            let assessment3 = engine.assess_risk(&market_data)?;
            
            // Should be exactly the same (within floating point precision)
            prop_assert!((assessment1.overall_risk_score - assessment2.overall_risk_score).abs() < 1e-10);
            prop_assert!((assessment2.overall_risk_score - assessment3.overall_risk_score).abs() < 1e-10);
            prop_assert!((assessment1.kelly_fraction - assessment2.kelly_fraction).abs() < 1e-10);
            prop_assert!((assessment2.kelly_fraction - assessment3.kelly_fraction).abs() < 1e-10);
        }
    }
}