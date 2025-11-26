//! Property-based tests for position sizing

use proptest::prelude::*;
use nt_risk::KellyCriterion;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;

proptest! {
    #[test]
    fn test_position_size_never_exceeds_portfolio(
        win_prob in 0.5..1.0f64,
        win_loss_ratio in 1.0..5.0f64,
        portfolio_value in 10000.0..1_000_000.0f64,
        price in 1.0..1000.0f64,
    ) {
        let kelly = KellyCriterion::new(0.5, 0.2, Decimal::new(100, 0))
            .expect("Should create Kelly criterion");

        let portfolio_dec = Decimal::from_f64(portfolio_value).unwrap();
        let price_dec = Decimal::from_f64(price).unwrap();

        let position = kelly
            .calculate_position_size(win_prob, win_loss_ratio, portfolio_dec, price_dec)
            .expect("Should calculate position");

        // Invariant: Position size never exceeds portfolio value
        prop_assert!(position.dollar_value <= portfolio_dec);

        // Invariant: Position size never exceeds max_position_fraction
        prop_assert!(position.dollar_value <= portfolio_dec * Decimal::new(20, 2)); // 0.2 = 20%
    }

    #[test]
    fn test_position_size_always_non_negative(
        win_prob in 0.0..1.0f64,
        win_loss_ratio in 0.1..10.0f64,
        portfolio_value in 1000.0..100_000.0f64,
        price in 10.0..500.0f64,
    ) {
        let kelly = KellyCriterion::new(0.5, 0.1, Decimal::new(100, 0))
            .expect("Should create Kelly criterion");

        let portfolio_dec = Decimal::from_f64(portfolio_value).unwrap();
        let price_dec = Decimal::from_f64(price).unwrap();

        let position = kelly
            .calculate_position_size(win_prob, win_loss_ratio, portfolio_dec, price_dec)
            .expect("Should calculate position");

        // Invariant: Position sizes are always non-negative
        prop_assert!(position.dollar_value >= Decimal::ZERO);
        prop_assert!(position.shares >= Decimal::ZERO);
    }

    #[test]
    fn test_kelly_fraction_bounded(
        win_prob in 0.5..0.9f64,
        win_loss_ratio in 1.5..3.0f64,
    ) {
        // Kelly formula: f* = (p * b - q) / b
        let q = 1.0 - win_prob;
        let kelly = (win_prob * win_loss_ratio - q) / win_loss_ratio;

        // Invariant: Kelly fraction should be bounded
        prop_assert!(kelly >= -1.0 && kelly <= 1.0);

        // For winning strategies (p > 0.5, b > 1), Kelly should be positive
        if win_prob > 0.5 && win_loss_ratio > 1.0 {
            prop_assert!(kelly > 0.0);
        }
    }

    #[test]
    fn test_position_shares_consistent_with_value(
        win_prob in 0.55..0.85f64,
        win_loss_ratio in 1.5..4.0f64,
        portfolio_value in 50000.0..200_000.0f64,
        price in 50.0..500.0f64,
    ) {
        let kelly = KellyCriterion::new(0.5, 0.15, Decimal::new(500, 0))
            .expect("Should create Kelly criterion");

        let portfolio_dec = Decimal::from_f64(portfolio_value).unwrap();
        let price_dec = Decimal::from_f64(price).unwrap();

        let position = kelly
            .calculate_position_size(win_prob, win_loss_ratio, portfolio_dec, price_dec)
            .expect("Should calculate position");

        // Invariant: shares * price should approximately equal dollar_value
        let calculated_value = position.shares * price_dec;
        let diff = (calculated_value - position.dollar_value).abs();
        let tolerance = price_dec; // Allow up to 1 share worth of difference

        prop_assert!(
            diff <= tolerance,
            "shares * price ({}) should equal dollar_value ({}), diff: {}",
            calculated_value,
            position.dollar_value,
            diff
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_tests_compile() {
        // This test ensures property tests compile correctly
        assert!(true);
    }
}
