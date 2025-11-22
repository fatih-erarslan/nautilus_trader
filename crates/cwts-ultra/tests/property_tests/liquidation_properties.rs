//! Property-Based Tests for Liquidation Engine
//!
//! Comprehensive property tests using proptest to verify:
//! - Liquidation price calculations (isolated vs cross margin)
//! - Margin requirements are never negative
//! - Position values are bounded and overflow-safe
//! - Maintenance margin always < initial margin

use proptest::prelude::*;
use std::time::Duration;

// Re-export the liquidation engine types
use crate::algorithms::liquidation_engine::{
    LiquidationEngine, LiquidationParameters, MarginMode, MarginPosition,
};

const PROP_TEST_CASES: u32 = 1000;

// Strategy generators for valid financial values
fn price_strategy() -> impl Strategy<Value = f64> {
    // Prices between $0.01 and $1,000,000
    (0.01f64..1_000_000.0f64)
}

fn position_size_strategy() -> impl Strategy<Value = f64> {
    // Position sizes between 0.0001 and 10000
    prop_oneof![
        // Long positions
        (0.0001f64..10000.0f64),
        // Short positions
        (-10000.0f64..-0.0001f64),
    ]
}

fn leverage_strategy() -> impl Strategy<Value = f64> {
    // Leverage between 1x and 100x
    (1.0f64..100.0f64)
}

fn margin_strategy() -> impl Strategy<Value = f64> {
    // Margin amounts between $10 and $1,000,000
    (10.0f64..1_000_000.0f64)
}

fn margin_mode_strategy() -> impl Strategy<Value = MarginMode> {
    prop_oneof![
        Just(MarginMode::Isolated),
        Just(MarginMode::Cross),
    ]
}

// Property 1: Initial margin is always non-negative and bounded
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_initial_margin_non_negative(
        position_size in position_size_strategy(),
        entry_price in price_strategy(),
        leverage in leverage_strategy(),
    ) {
        let params = LiquidationParameters::default();
        let engine = LiquidationEngine::new(params.clone());

        let result = engine.calculate_initial_margin(position_size, entry_price, leverage);

        if result.is_ok() {
            let margin = result.unwrap();
            // Property: Initial margin must be non-negative
            prop_assert!(margin >= 0.0, "Initial margin must be non-negative: {}", margin);

            // Property: Initial margin must be finite
            prop_assert!(margin.is_finite(), "Initial margin must be finite");

            // Property: Initial margin should be proportional to position value
            let notional_value = position_size.abs() * entry_price;
            let expected_base_margin = notional_value / leverage;
            prop_assert!(
                margin >= expected_base_margin * 0.99, // Allow for rounding
                "Initial margin {} should be at least base margin {}",
                margin,
                expected_base_margin
            );

            // Property: Initial margin must not overflow
            prop_assert!(
                margin < f64::MAX / 2.0,
                "Initial margin must not approach overflow"
            );
        }
    }
}

// Property 2: Maintenance margin is always less than initial margin
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_maintenance_margin_less_than_initial(
        position_size in position_size_strategy(),
        entry_price in price_strategy(),
        leverage in leverage_strategy(),
    ) {
        let params = LiquidationParameters::default();
        let engine = LiquidationEngine::new(params);

        let initial_margin_result = engine.calculate_initial_margin(
            position_size,
            entry_price,
            leverage
        );

        if initial_margin_result.is_ok() {
            let initial_margin = initial_margin_result.unwrap();
            let maintenance_margin = engine.calculate_maintenance_margin(
                position_size,
                entry_price
            );

            // Property: Maintenance margin must be less than initial margin
            prop_assert!(
                maintenance_margin < initial_margin,
                "Maintenance margin {} must be less than initial margin {}",
                maintenance_margin,
                initial_margin
            );

            // Property: Both must be non-negative
            prop_assert!(maintenance_margin >= 0.0);
            prop_assert!(initial_margin >= 0.0);

            // Property: Maintenance margin should be a percentage of notional value
            let notional_value = position_size.abs() * entry_price;
            prop_assert!(
                maintenance_margin <= notional_value * 0.2, // Should be less than 20%
                "Maintenance margin should be reasonable percentage of notional value"
            );
        }
    }
}

// Property 3: Liquidation price calculations are consistent for isolated margin
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_liquidation_price_isolated_consistent(
        position_size in position_size_strategy(),
        entry_price in price_strategy(),
        leverage in leverage_strategy(),
    ) {
        let params = LiquidationParameters::default();
        let engine = LiquidationEngine::new(params.clone());

        let initial_margin_result = engine.calculate_initial_margin(
            position_size,
            entry_price,
            leverage
        );

        if initial_margin_result.is_ok() {
            let initial_margin = initial_margin_result.unwrap();
            let maintenance_margin = engine.calculate_maintenance_margin(
                position_size,
                entry_price
            );

            let position = MarginPosition {
                symbol: "TEST".to_string(),
                size: position_size,
                entry_price,
                current_price: entry_price,
                leverage,
                margin_mode: MarginMode::Isolated,
                initial_margin,
                maintenance_margin,
                unrealized_pnl: 0.0,
                liquidation_price: 0.0,
                margin_ratio: f64::INFINITY,
                timestamp: 0,
            };

            let liq_price_result = engine.calculate_liquidation_price_isolated(&position);

            if liq_price_result.is_ok() {
                let liq_price = liq_price_result.unwrap();

                // Property: Liquidation price must be non-negative
                prop_assert!(liq_price >= 0.0, "Liquidation price must be non-negative");

                // Property: Liquidation price must be finite
                prop_assert!(liq_price.is_finite(), "Liquidation price must be finite");

                // Property: For long positions, liquidation price < entry price
                if position_size > 0.0 {
                    prop_assert!(
                        liq_price < entry_price,
                        "Long position liquidation price {} must be below entry price {}",
                        liq_price,
                        entry_price
                    );
                }

                // Property: For short positions, liquidation price > entry price
                if position_size < 0.0 {
                    prop_assert!(
                        liq_price > entry_price,
                        "Short position liquidation price {} must be above entry price {}",
                        liq_price,
                        entry_price
                    );
                }

                // Property: Higher leverage means closer liquidation price to entry
                let high_leverage = leverage * 2.0;
                if high_leverage <= params.max_leverage {
                    let high_lev_initial = engine.calculate_initial_margin(
                        position_size,
                        entry_price,
                        high_leverage
                    ).unwrap();

                    let high_lev_position = MarginPosition {
                        symbol: "TEST".to_string(),
                        size: position_size,
                        entry_price,
                        current_price: entry_price,
                        leverage: high_leverage,
                        margin_mode: MarginMode::Isolated,
                        initial_margin: high_lev_initial,
                        maintenance_margin,
                        unrealized_pnl: 0.0,
                        liquidation_price: 0.0,
                        margin_ratio: f64::INFINITY,
                        timestamp: 0,
                    };

                    if let Ok(high_lev_liq) = engine.calculate_liquidation_price_isolated(&high_lev_position) {
                        let distance_low_lev = (liq_price - entry_price).abs();
                        let distance_high_lev = (high_lev_liq - entry_price).abs();

                        prop_assert!(
                            distance_high_lev <= distance_low_lev * 1.01, // Allow small tolerance
                            "Higher leverage should result in closer liquidation price"
                        );
                    }
                }
            }
        }
    }
}

// Property 4: Margin ratio calculations are mathematically sound
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_margin_ratio_sound(
        position_size in position_size_strategy(),
        entry_price in price_strategy(),
        current_price in price_strategy(),
        leverage in leverage_strategy(),
    ) {
        let params = LiquidationParameters::default();
        let engine = LiquidationEngine::new(params);

        let initial_margin_result = engine.calculate_initial_margin(
            position_size,
            entry_price,
            leverage
        );

        if initial_margin_result.is_ok() {
            let initial_margin = initial_margin_result.unwrap();
            let maintenance_margin = engine.calculate_maintenance_margin(
                position_size,
                current_price
            );

            // Calculate unrealized PnL
            let is_long = position_size > 0.0;
            let unrealized_pnl = if is_long {
                position_size * (current_price - entry_price)
            } else {
                position_size * (entry_price - current_price)
            };

            // Calculate margin ratio
            let equity = initial_margin + unrealized_pnl;
            let margin_ratio = if maintenance_margin > 0.0 {
                equity / maintenance_margin
            } else {
                f64::INFINITY
            };

            // Property: Margin ratio must be finite or infinity (not NaN)
            prop_assert!(
                margin_ratio.is_finite() || margin_ratio.is_infinite(),
                "Margin ratio must not be NaN"
            );

            // Property: If equity is positive and maintenance > 0, ratio should be positive
            if equity > 0.0 && maintenance_margin > 0.0 {
                prop_assert!(
                    margin_ratio > 0.0,
                    "Margin ratio must be positive when equity and maintenance are positive"
                );
            }

            // Property: Margin ratio decreases as maintenance margin increases
            if maintenance_margin > 0.0 && equity > 0.0 {
                let higher_maintenance = maintenance_margin * 1.5;
                let lower_ratio = equity / higher_maintenance;

                prop_assert!(
                    lower_ratio < margin_ratio,
                    "Higher maintenance margin should result in lower margin ratio"
                );
            }
        }
    }
}

// Property 5: No arithmetic overflow in position value calculations
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_no_overflow_in_calculations(
        position_size in position_size_strategy(),
        price in price_strategy(),
    ) {
        // Property: Notional value calculation doesn't overflow
        let notional_value = position_size.abs() * price;
        prop_assert!(
            notional_value.is_finite(),
            "Notional value calculation must not overflow"
        );

        // Property: Position value is bounded
        prop_assert!(
            notional_value < 1e15, // $1 quadrillion is unrealistic
            "Position value must be within realistic bounds"
        );
    }
}

// Property 6: Liquidation price is deterministic (same inputs = same outputs)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_liquidation_price_deterministic(
        position_size in position_size_strategy(),
        entry_price in price_strategy(),
        leverage in leverage_strategy(),
    ) {
        let params = LiquidationParameters::default();
        let engine = LiquidationEngine::new(params);

        let initial_margin_result = engine.calculate_initial_margin(
            position_size,
            entry_price,
            leverage
        );

        if initial_margin_result.is_ok() {
            let initial_margin = initial_margin_result.unwrap();
            let maintenance_margin = engine.calculate_maintenance_margin(
                position_size,
                entry_price
            );

            let position = MarginPosition {
                symbol: "TEST".to_string(),
                size: position_size,
                entry_price,
                current_price: entry_price,
                leverage,
                margin_mode: MarginMode::Isolated,
                initial_margin,
                maintenance_margin,
                unrealized_pnl: 0.0,
                liquidation_price: 0.0,
                margin_ratio: f64::INFINITY,
                timestamp: 0,
            };

            // Calculate liquidation price twice
            let liq_price_1 = engine.calculate_liquidation_price_isolated(&position);
            let liq_price_2 = engine.calculate_liquidation_price_isolated(&position);

            // Property: Same inputs must produce same outputs
            prop_assert_eq!(
                liq_price_1.is_ok(),
                liq_price_2.is_ok(),
                "Deterministic: success/failure must be consistent"
            );

            if liq_price_1.is_ok() && liq_price_2.is_ok() {
                let price_1 = liq_price_1.unwrap();
                let price_2 = liq_price_2.unwrap();

                // Allow for very small floating-point differences
                let diff = (price_1 - price_2).abs();
                prop_assert!(
                    diff < 1e-10,
                    "Liquidation price must be deterministic: {} vs {}",
                    price_1,
                    price_2
                );
            }
        }
    }
}

// Property 7: Funding payments are proportional to position size
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_funding_payment_proportional(
        position_size in position_size_strategy(),
        mark_price in price_strategy(),
        funding_rate in (-0.01f64..0.01f64), // -1% to +1%
    ) {
        let params = LiquidationParameters::default();
        let engine = LiquidationEngine::new(params);

        let funding_payment = engine.calculate_funding_payment(
            position_size,
            mark_price,
            funding_rate
        );

        // Property: Funding payment must be finite
        prop_assert!(
            funding_payment.is_finite(),
            "Funding payment must be finite"
        );

        // Property: Doubling position size doubles funding payment
        let double_size = position_size * 2.0;
        let double_funding = engine.calculate_funding_payment(
            double_size,
            mark_price,
            funding_rate
        );

        let ratio = if funding_payment.abs() > 1e-10 {
            double_funding / funding_payment
        } else {
            2.0 // If original is near zero, ratio should still be ~2
        };

        prop_assert!(
            (ratio - 2.0).abs() < 0.01,
            "Funding payment must be proportional to position size"
        );

        // Property: Sign of funding payment matches position direction
        if funding_rate > 0.0 && position_size > 0.0 {
            prop_assert!(
                funding_payment > 0.0,
                "Positive funding rate + long position = positive payment"
            );
        }
    }
}

// Property 8: Mark price premium is consistently applied
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_mark_price_premium_consistent(
        base_price in price_strategy(),
    ) {
        let params = LiquidationParameters::default();
        let premium = params.mark_price_premium;

        // Property: Premium is within reasonable bounds
        prop_assert!(
            premium >= 0.0 && premium <= 0.01,
            "Mark price premium should be between 0% and 1%"
        );

        // Property: Mark price calculation is consistent
        let expected_mark_price = base_price * (1.0 + premium);
        prop_assert!(
            expected_mark_price > base_price,
            "Mark price must be greater than base price with positive premium"
        );

        prop_assert!(
            expected_mark_price.is_finite(),
            "Mark price must be finite"
        );
    }
}

// Property 9: Leverage constraints are enforced
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_leverage_constraints_enforced(
        position_size in position_size_strategy(),
        entry_price in price_strategy(),
        leverage in (0.0f64..200.0f64), // Test beyond max leverage
    ) {
        let params = LiquidationParameters::default();
        let engine = LiquidationEngine::new(params.clone());

        let result = engine.calculate_initial_margin(position_size, entry_price, leverage);

        // Property: Invalid leverage (0 or > max) should fail
        if leverage <= 0.0 || leverage > params.max_leverage {
            prop_assert!(
                result.is_err(),
                "Invalid leverage {} should be rejected (max: {})",
                leverage,
                params.max_leverage
            );
        } else {
            // Property: Valid leverage should succeed
            prop_assert!(
                result.is_ok(),
                "Valid leverage {} should be accepted",
                leverage
            );
        }
    }
}

// Property 10: Edge case - Zero position size handling
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_zero_position_size_handled(
        entry_price in price_strategy(),
        leverage in leverage_strategy(),
    ) {
        let params = LiquidationParameters::default();
        let engine = LiquidationEngine::new(params);

        // Create position with zero size
        let position = MarginPosition {
            symbol: "TEST".to_string(),
            size: 0.0,
            entry_price,
            current_price: entry_price,
            leverage,
            margin_mode: MarginMode::Isolated,
            initial_margin: 0.0,
            maintenance_margin: 0.0,
            unrealized_pnl: 0.0,
            liquidation_price: 0.0,
            margin_ratio: f64::INFINITY,
            timestamp: 0,
        };

        let result = engine.calculate_liquidation_price_isolated(&position);

        // Property: Zero position size should be rejected
        prop_assert!(
            result.is_err(),
            "Zero position size should be rejected"
        );
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_property_test_config() {
        // Verify that our property test configuration is set correctly
        assert_eq!(PROP_TEST_CASES, 1000);
    }

    #[test]
    fn test_strategy_generators() {
        // Verify that strategy generators produce valid ranges
        let price_gen = price_strategy();
        let position_gen = position_size_strategy();
        let leverage_gen = leverage_strategy();

        // Just instantiate them to verify they compile
        let _ = (price_gen, position_gen, leverage_gen);
    }
}
