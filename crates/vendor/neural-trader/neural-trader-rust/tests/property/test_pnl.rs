//! Property-based tests for P&L calculations

use proptest::prelude::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;

proptest! {
    #[test]
    fn test_pnl_symmetry(
        entry_price in 10.0..500.0f64,
        exit_price in 10.0..500.0f64,
        quantity in 1.0..1000.0f64,
    ) {
        let entry = Decimal::from_f64(entry_price).unwrap();
        let exit = Decimal::from_f64(exit_price).unwrap();
        let qty = Decimal::from_f64(quantity).unwrap();

        // Long P&L
        let long_pnl = (exit - entry) * qty;

        // Short P&L (inverse)
        let short_pnl = (entry - exit) * qty;

        // Invariant: Long and short P&L should be opposite
        prop_assert_eq!(long_pnl, -short_pnl);
    }

    #[test]
    fn test_pnl_zero_at_breakeven(
        price in 10.0..500.0f64,
        quantity in 1.0..100.0f64,
    ) {
        let price_dec = Decimal::from_f64(price).unwrap();
        let qty = Decimal::from_f64(quantity).unwrap();

        // Entry and exit at same price
        let pnl = (price_dec - price_dec) * qty;

        // Invariant: P&L should be zero at breakeven
        prop_assert_eq!(pnl, Decimal::ZERO);
    }

    #[test]
    fn test_pnl_scales_with_quantity(
        entry_price in 50.0..200.0f64,
        exit_price in 50.0..200.0f64,
        base_quantity in 1.0..10.0f64,
        multiplier in 2.0..10.0f64,
    ) {
        let entry = Decimal::from_f64(entry_price).unwrap();
        let exit = Decimal::from_f64(exit_price).unwrap();
        let base_qty = Decimal::from_f64(base_quantity).unwrap();
        let mult = Decimal::from_f64(multiplier).unwrap();

        let pnl_base = (exit - entry) * base_qty;
        let pnl_scaled = (exit - entry) * (base_qty * mult);

        // Invariant: P&L should scale linearly with quantity
        let expected = pnl_base * mult;
        let diff = (pnl_scaled - expected).abs();

        prop_assert!(
            diff < Decimal::new(1, 2), // Allow 0.01 tolerance for rounding
            "P&L should scale linearly: {} * {} ≈ {}, got {}",
            pnl_base,
            mult,
            expected,
            pnl_scaled
        );
    }

    #[test]
    fn test_pnl_percentage_bounded(
        entry_price in 10.0..500.0f64,
        exit_price in 1.0..1000.0f64,
    ) {
        let entry = Decimal::from_f64(entry_price).unwrap();
        let exit = Decimal::from_f64(exit_price).unwrap();

        if entry > Decimal::ZERO {
            let pnl_pct = (exit - entry) / entry;

            // Invariant: P&L percentage minimum is -100% (total loss)
            prop_assert!(pnl_pct >= Decimal::new(-1, 0));

            // P&L percentage can be unbounded on upside, but should be reasonable
            // in our test range
            prop_assert!(pnl_pct < Decimal::new(100, 0)); // Less than 10000%
        }
    }

    #[test]
    fn test_cumulative_pnl_associativity(
        pnl1 in -1000.0..1000.0f64,
        pnl2 in -1000.0..1000.0f64,
        pnl3 in -1000.0..1000.0f64,
    ) {
        let p1 = Decimal::from_f64(pnl1).unwrap();
        let p2 = Decimal::from_f64(pnl2).unwrap();
        let p3 = Decimal::from_f64(pnl3).unwrap();

        // Invariant: Order of addition shouldn't matter
        let total1 = (p1 + p2) + p3;
        let total2 = p1 + (p2 + p3);
        let total3 = p1 + p2 + p3;

        prop_assert_eq!(total1, total2);
        prop_assert_eq!(total2, total3);
    }

    #[test]
    fn test_realized_vs_unrealized_pnl(
        entry_price in 50.0..500.0f64,
        current_price in 50.0..500.0f64,
        quantity_held in 0.0..100.0f64,
        quantity_sold in 0.0..100.0f64,
    ) {
        let entry = Decimal::from_f64(entry_price).unwrap();
        let current = Decimal::from_f64(current_price).unwrap();
        let held = Decimal::from_f64(quantity_held).unwrap();
        let sold = Decimal::from_f64(quantity_sold).unwrap();

        // Unrealized P&L on held position
        let unrealized = (current - entry) * held;

        // Realized P&L on sold position
        let realized = (current - entry) * sold;

        // Total P&L
        let total = unrealized + realized;

        // Invariant: Total P&L equals P&L on full position
        let full_position_pnl = (current - entry) * (held + sold);

        prop_assert_eq!(total, full_position_pnl);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_pnl_property_tests_compile() {
        assert!(true);
    }
}
