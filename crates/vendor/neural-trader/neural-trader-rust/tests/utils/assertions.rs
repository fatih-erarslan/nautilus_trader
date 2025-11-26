//! Custom test assertions

use rust_decimal::Decimal;

/// Assert that two decimal values are approximately equal
pub fn assert_decimal_approx_eq(left: Decimal, right: Decimal, tolerance: Decimal) {
    let diff = (left - right).abs();
    assert!(
        diff <= tolerance,
        "assertion failed: `(left ≈ right)`\n  left: `{:?}`,\n right: `{:?}`,\n  diff: `{:?}`,\n  tolerance: `{:?}`",
        left, right, diff, tolerance
    );
}

/// Assert that a decimal value is within a range
pub fn assert_decimal_in_range(value: Decimal, min: Decimal, max: Decimal) {
    assert!(
        value >= min && value <= max,
        "assertion failed: `(min <= value <= max)`\n  value: `{:?}`,\n  min: `{:?}`,\n  max: `{:?}`",
        value, min, max
    );
}

/// Assert that a decimal is positive
pub fn assert_decimal_positive(value: Decimal) {
    assert!(
        value > Decimal::ZERO,
        "assertion failed: `(value > 0)`\n  value: `{:?}`",
        value
    );
}

/// Assert that a decimal is non-negative
pub fn assert_decimal_non_negative(value: Decimal) {
    assert!(
        value >= Decimal::ZERO,
        "assertion failed: `(value >= 0)`\n  value: `{:?}`",
        value
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_assert_decimal_approx_eq() {
        assert_decimal_approx_eq(dec!(1.0), dec!(1.001), dec!(0.01));
    }

    #[test]
    #[should_panic]
    fn test_assert_decimal_approx_eq_fail() {
        assert_decimal_approx_eq(dec!(1.0), dec!(1.1), dec!(0.01));
    }

    #[test]
    fn test_assert_decimal_in_range() {
        assert_decimal_in_range(dec!(5), dec!(0), dec!(10));
    }

    #[test]
    fn test_assert_decimal_positive() {
        assert_decimal_positive(dec!(1));
    }
}
