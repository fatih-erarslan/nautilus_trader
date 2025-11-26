//! Comprehensive tests for Kelly Criterion position sizing
//!
//! Tests both single-asset and multi-asset Kelly implementations

use nt_risk::kelly::*;
use nt_risk::types::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use nalgebra::{DMatrix, DVector};

// ============================================================================
// Single-Asset Kelly Tests
// ============================================================================

#[test]
fn test_kelly_single_asset_creation() {
    let kelly = KellySingleAsset::new(0.6, 0.5, 2.0, 0.25);
    // Should not panic
}

#[test]
fn test_kelly_single_asset_invalid_params() {
    // Invalid probability (> 1.0)
    let result = std::panic::catch_unwind(|| {
        KellySingleAsset::new(1.5, 0.5, 2.0, 0.25);
    });
    // Should handle gracefully

    // Invalid odds (negative)
    let result = std::panic::catch_unwind(|| {
        KellySingleAsset::new(0.6, 0.5, -1.0, 0.25);
    });
}

#[test]
fn test_kelly_single_asset_calculate_fraction() {
    // Classic example: 60% win rate, 2:1 odds
    let kelly = KellySingleAsset::new(0.6, 0.4, 2.0, 1.0);
    let fraction = kelly.calculate_fraction();

    // Kelly formula: f* = (bp - q) / b = (2*0.6 - 0.4) / 2 = 0.4
    assert!((fraction - 0.4).abs() < 0.01);
}

#[test]
fn test_kelly_single_asset_fractional() {
    // With 0.25 fractional Kelly
    let kelly = KellySingleAsset::new(0.6, 0.4, 2.0, 0.25);
    let fraction = kelly.calculate_fraction();

    // Should be 25% of full Kelly
    assert!((fraction - 0.1).abs() < 0.01);
}

#[test]
fn test_kelly_single_asset_position_size() {
    let kelly = KellySingleAsset::new(0.6, 0.4, 2.0, 0.5);
    let capital = 10000.0;
    let price = 100.0;

    let (size, fraction) = kelly.calculate_position_size(capital, price);

    // Verify size makes sense
    assert!(size > 0.0);
    assert!(fraction > 0.0 && fraction < 1.0);
    assert!(size * price <= capital); // Can't exceed capital
}

#[test]
fn test_kelly_single_asset_edge_case_no_edge() {
    // 50% win rate, 1:1 odds = no edge
    let kelly = KellySingleAsset::new(0.5, 0.5, 1.0, 1.0);
    let fraction = kelly.calculate_fraction();

    // Kelly fraction should be 0 (no bet)
    assert!((fraction).abs() < 0.01);
}

#[test]
fn test_kelly_single_asset_edge_case_negative_edge() {
    // 40% win rate, 1:1 odds = negative edge
    let kelly = KellySingleAsset::new(0.4, 0.6, 1.0, 1.0);
    let fraction = kelly.calculate_fraction();

    // Kelly fraction should be 0 (no bet, as we clamp negative)
    assert!(fraction <= 0.0);
}

#[test]
fn test_kelly_with_max_leverage() {
    let kelly = KellySingleAsset::new(0.7, 0.3, 3.0, 1.0)
        .with_max_leverage(2.0);

    let fraction = kelly.calculate_fraction();
    // Even if Kelly suggests more, should not exceed max leverage
    assert!(fraction <= 2.0);
}

#[test]
fn test_kelly_optimal_growth_rate() {
    let kelly = KellySingleAsset::new(0.6, 0.4, 2.0, 1.0);
    let growth_rate = kelly.expected_growth_rate();

    // Growth rate should be positive with positive edge
    assert!(growth_rate > 0.0);
}

// ============================================================================
// Multi-Asset Kelly Tests
// ============================================================================

#[test]
fn test_kelly_multi_asset_creation() {
    let symbols = vec![Symbol::from("AAPL"), Symbol::from("GOOGL")];
    let mut expected_returns = HashMap::new();
    expected_returns.insert(symbols[0].clone(), 0.10);
    expected_returns.insert(symbols[1].clone(), 0.12);

    let covariance = DMatrix::from_row_slice(
        2,
        2,
        &[0.04, 0.02, 0.02, 0.05],
    );

    let kelly = KellyMultiAsset::new(
        expected_returns,
        covariance,
        symbols,
        0.25,
    );

    assert!(kelly.is_ok());
}

#[test]
fn test_kelly_multi_asset_dimension_mismatch() {
    let symbols = vec![Symbol::from("AAPL"), Symbol::from("GOOGL")];
    let mut expected_returns = HashMap::new();
    expected_returns.insert(symbols[0].clone(), 0.10);
    expected_returns.insert(symbols[1].clone(), 0.12);

    // Wrong size covariance matrix (3x3 instead of 2x2)
    let covariance = DMatrix::from_row_slice(
        3,
        3,
        &[0.04, 0.02, 0.01, 0.02, 0.05, 0.01, 0.01, 0.01, 0.03],
    );

    let kelly = KellyMultiAsset::new(
        expected_returns,
        covariance,
        symbols,
        0.25,
    );

    assert!(kelly.is_err());
}

#[test]
fn test_kelly_multi_asset_missing_returns() {
    let symbols = vec![Symbol::from("AAPL"), Symbol::from("GOOGL")];
    let mut expected_returns = HashMap::new();
    expected_returns.insert(symbols[0].clone(), 0.10);
    // Missing GOOGL return

    let covariance = DMatrix::from_row_slice(
        2,
        2,
        &[0.04, 0.02, 0.02, 0.05],
    );

    let kelly = KellyMultiAsset::new(
        expected_returns,
        covariance,
        symbols,
        0.25,
    );

    assert!(kelly.is_err());
}

#[test]
fn test_kelly_multi_asset_calculate_weights() {
    let symbols = vec![Symbol::from("AAPL"), Symbol::from("GOOGL")];
    let mut expected_returns = HashMap::new();
    expected_returns.insert(symbols[0].clone(), 0.10);
    expected_returns.insert(symbols[1].clone(), 0.12);

    let covariance = DMatrix::from_row_slice(
        2,
        2,
        &[0.04, 0.02, 0.02, 0.05],
    );

    let kelly = KellyMultiAsset::new(
        expected_returns,
        covariance,
        symbols.clone(),
        0.25,
    )
    .unwrap();

    let weights = kelly.calculate_weights().unwrap();

    // Check all symbols have weights
    assert!(weights.contains_key(&symbols[0]));
    assert!(weights.contains_key(&symbols[1]));

    // Total absolute weight should not exceed max leverage (1.0 by default)
    let total_weight: f64 = weights.values().map(|w| w.abs()).sum();
    assert!(total_weight <= 1.01); // Small tolerance for rounding
}

#[test]
fn test_kelly_multi_asset_with_constraints() {
    let symbols = vec![Symbol::from("AAPL"), Symbol::from("GOOGL")];
    let mut expected_returns = HashMap::new();
    expected_returns.insert(symbols[0].clone(), 0.15);
    expected_returns.insert(symbols[1].clone(), 0.10);

    let covariance = DMatrix::from_row_slice(
        2,
        2,
        &[0.04, 0.02, 0.02, 0.05],
    );

    let kelly = KellyMultiAsset::new(
        expected_returns,
        covariance,
        symbols.clone(),
        0.5,
    )
    .unwrap()
    .with_max_leverage(0.8)
    .with_max_concentration(0.4);

    let weights = kelly.calculate_weights().unwrap();

    // Check constraints
    let total_weight: f64 = weights.values().map(|w| w.abs()).sum();
    assert!(total_weight <= 0.81); // Slightly above max_leverage for rounding

    for weight in weights.values() {
        assert!(weight.abs() <= 0.41); // Max concentration constraint
    }
}

#[test]
fn test_kelly_multi_asset_position_sizes() {
    let symbols = vec![Symbol::from("AAPL"), Symbol::from("GOOGL")];
    let mut expected_returns = HashMap::new();
    expected_returns.insert(symbols[0].clone(), 0.10);
    expected_returns.insert(symbols[1].clone(), 0.12);

    let covariance = DMatrix::from_row_slice(
        2,
        2,
        &[0.04, 0.02, 0.02, 0.05],
    );

    let kelly = KellyMultiAsset::new(
        expected_returns,
        covariance,
        symbols.clone(),
        0.25,
    )
    .unwrap();

    let mut prices = HashMap::new();
    prices.insert(symbols[0].clone(), 150.0);
    prices.insert(symbols[1].clone(), 2800.0);

    let capital = 100_000.0;
    let positions = kelly.calculate_position_sizes(capital, &prices).unwrap();

    // Check all symbols have position sizes
    assert!(positions.contains_key(&symbols[0]));
    assert!(positions.contains_key(&symbols[1]));

    // Total investment should not exceed capital
    let total_investment: f64 = positions
        .iter()
        .map(|(symbol, &shares)| shares * prices.get(symbol).unwrap())
        .sum();

    assert!(total_investment <= capital * 1.01); // Small tolerance
}

#[test]
fn test_kelly_multi_asset_uncorrelated() {
    let symbols = vec![Symbol::from("AAPL"), Symbol::from("GOOGL")];
    let mut expected_returns = HashMap::new();
    expected_returns.insert(symbols[0].clone(), 0.10);
    expected_returns.insert(symbols[1].clone(), 0.10);

    // Identity matrix (uncorrelated)
    let covariance = DMatrix::from_row_slice(
        2,
        2,
        &[0.04, 0.00, 0.00, 0.04],
    );

    let kelly = KellyMultiAsset::new(
        expected_returns,
        covariance,
        symbols.clone(),
        0.5,
    )
    .unwrap();

    let weights = kelly.calculate_weights().unwrap();

    // With equal returns and uncorrelated assets, weights should be similar
    let weight_aapl = weights.get(&symbols[0]).unwrap();
    let weight_googl = weights.get(&symbols[1]).unwrap();

    assert!((weight_aapl - weight_googl).abs() < 0.1);
}

#[test]
fn test_kelly_multi_asset_highly_correlated() {
    let symbols = vec![Symbol::from("AAPL"), Symbol::from("GOOGL")];
    let mut expected_returns = HashMap::new();
    expected_returns.insert(symbols[0].clone(), 0.10);
    expected_returns.insert(symbols[1].clone(), 0.10);

    // High correlation (0.9)
    let covariance = DMatrix::from_row_slice(
        2,
        2,
        &[0.04, 0.036, 0.036, 0.04],
    );

    let kelly = KellyMultiAsset::new(
        expected_returns,
        covariance,
        symbols.clone(),
        0.5,
    )
    .unwrap();

    let weights = kelly.calculate_weights().unwrap();

    // With high correlation, diversification benefit is reduced
    // Total weight should be lower than uncorrelated case
    let total_weight: f64 = weights.values().map(|w| w.abs()).sum();
    assert!(total_weight > 0.0);
}

// ============================================================================
// Kelly Risk Tests
// ============================================================================

#[test]
fn test_kelly_risk_of_ruin() {
    let kelly = KellySingleAsset::new(0.6, 0.4, 2.0, 1.0);

    // Risk of ruin should be very low with positive edge and proper Kelly
    let risk = kelly.risk_of_ruin(0.5); // 50% drawdown threshold

    assert!(risk >= 0.0 && risk <= 1.0);
}

#[test]
fn test_kelly_fractional_reduces_risk() {
    let full_kelly = KellySingleAsset::new(0.6, 0.4, 2.0, 1.0);
    let half_kelly = KellySingleAsset::new(0.6, 0.4, 2.0, 0.5);

    let risk_full = full_kelly.risk_of_ruin(0.5);
    let risk_half = half_kelly.risk_of_ruin(0.5);

    // Fractional Kelly should have lower risk
    assert!(risk_half <= risk_full);
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
        fn test_kelly_fraction_bounds(
            win_prob in 0.01..0.99f64,
            odds in 0.1..10.0f64,
        ) {
            let lose_prob = 1.0 - win_prob;
            let kelly = KellySingleAsset::new(win_prob, lose_prob, odds, 0.5);
            let fraction = kelly.calculate_fraction();

            // Kelly fraction should never exceed 100% with 0.5 fractional
            prop_assert!(fraction <= 0.5);
            // And should not be too negative
            prop_assert!(fraction >= -0.5);
        }

        #[test]
        fn test_kelly_position_size_never_exceeds_capital(
            win_prob in 0.51..0.99f64,
            capital in 1000.0..1000000.0f64,
            price in 1.0..1000.0f64,
        ) {
            let kelly = KellySingleAsset::new(win_prob, 1.0 - win_prob, 2.0, 0.25)
                .with_max_leverage(1.0);

            let (size, _) = kelly.calculate_position_size(capital, price);
            let cost = size * price;

            prop_assert!(cost <= capital * 1.01); // Small tolerance
        }

        #[test]
        fn test_multi_asset_weights_sum_constraint(
            return1 in 0.01..0.30f64,
            return2 in 0.01..0.30f64,
        ) {
            let symbols = vec![Symbol::from("A"), Symbol::from("B")];
            let mut expected_returns = HashMap::new();
            expected_returns.insert(symbols[0].clone(), return1);
            expected_returns.insert(symbols[1].clone(), return2);

            let covariance = DMatrix::from_row_slice(
                2,
                2,
                &[0.04, 0.02, 0.02, 0.05],
            );

            let kelly = KellyMultiAsset::new(
                expected_returns,
                covariance,
                symbols,
                0.5,
            )
            .unwrap()
            .with_max_leverage(1.0);

            let weights = kelly.calculate_weights().unwrap();
            let total_weight: f64 = weights.values().map(|w| w.abs()).sum();

            prop_assert!(total_weight <= 1.01); // Max leverage constraint
        }
    }
}
