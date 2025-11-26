//! Integration tests for risk management

use nt_risk::KellyCriterion;
use rust_decimal_macros::dec;

mod utils {
    include!("../utils/assertions.rs");
}

use utils::*;

#[test]
fn test_kelly_criterion_position_sizing() {
    let kelly = KellyCriterion::new(0.5, 0.1, dec!(100))
        .expect("Should create Kelly criterion");

    let win_prob = 0.6;
    let win_loss_ratio = 2.0;
    let portfolio_value = dec!(100000);
    let current_price = dec!(150);

    let position = kelly
        .calculate_position_size(win_prob, win_loss_ratio, portfolio_value, current_price)
        .expect("Should calculate position size");

    // Verify position size is reasonable
    assert_decimal_positive(position.dollar_value);
    assert_decimal_in_range(
        position.dollar_value,
        dec!(0),
        portfolio_value * dec!(0.1), // Should not exceed max_position_fraction
    );
}

#[test]
fn test_kelly_criterion_bounds() {
    let kelly = KellyCriterion::new(1.0, 0.2, dec!(100))
        .expect("Should create Kelly criterion");

    // Test with losing strategy (win_prob < 0.5)
    let position = kelly
        .calculate_position_size(0.3, 1.5, dec!(100000), dec!(100))
        .expect("Should calculate position size");

    // Should recommend very small or zero position for losing strategy
    assert!(position.dollar_value <= dec!(5000));
}

#[test]
fn test_kelly_criterion_max_position_limit() {
    let kelly = KellyCriterion::new(1.0, 0.05, dec!(100))
        .expect("Should create Kelly criterion");

    // Very high Kelly fraction should still be capped by max_position_fraction
    let position = kelly
        .calculate_position_size(0.9, 5.0, dec!(100000), dec!(100))
        .expect("Should calculate position size");

    // Should not exceed 5% of portfolio
    assert_decimal_in_range(
        position.dollar_value,
        dec!(0),
        dec!(100000) * dec!(0.05),
    );
}

#[test]
fn test_kelly_criterion_invalid_inputs() {
    // Invalid kelly_fraction
    let result = KellyCriterion::new(1.5, 0.1, dec!(100));
    assert!(result.is_err(), "Should reject kelly_fraction > 1.0");

    // Invalid max_position_fraction
    let result = KellyCriterion::new(0.5, 1.5, dec!(100));
    assert!(result.is_err(), "Should reject max_position_fraction > 1.0");
}

#[test]
fn test_kelly_criterion_zero_portfolio() {
    let kelly = KellyCriterion::new(0.5, 0.1, dec!(100))
        .expect("Should create Kelly criterion");

    let result = kelly.calculate_position_size(0.6, 2.0, dec!(0), dec!(100));
    assert!(result.is_err(), "Should reject zero portfolio value");
}

#[test]
fn test_kelly_criterion_minimum_position_size() {
    let min_size = dec!(1000);
    let kelly = KellyCriterion::new(0.25, 0.1, min_size)
        .expect("Should create Kelly criterion");

    let position = kelly
        .calculate_position_size(0.55, 1.2, dec!(10000), dec!(50))
        .expect("Should calculate position size");

    // If calculated size is very small, should meet minimum
    if position.shares > dec!(0) {
        assert!(position.dollar_value >= min_size || position.dollar_value == dec!(0));
    }
}
