/*!
 * Integration tests for FIFO tax calculation algorithm
 *
 * Tests the complete FIFO workflow including:
 * - End-to-end disposal calculations
 * - Multi-asset scenarios
 * - Edge cases and error handling
 * - Performance validation
 */

use agentic_accounting_rust_core::{
    calculate_fifo_disposal,
    Transaction, TaxLot, TransactionType,
};
use chrono::{TimeZone, Utc};
use rust_decimal::Decimal;
use std::str::FromStr;

// Helper function to create test transactions
fn create_sale(
    id: &str,
    asset: &str,
    quantity: &str,
    price: &str,
    date: &str,
) -> Transaction {
    Transaction {
        id: id.to_string(),
        transaction_type: TransactionType::Sell,
        asset: asset.to_string(),
        quantity: Decimal::from_str(quantity).unwrap(),
        price: Decimal::from_str(price).unwrap(),
        timestamp: chrono::DateTime::parse_from_rfc3339(date)
            .unwrap()
            .with_timezone(&Utc),
        source: "test".to_string(),
        fees: Decimal::ZERO,
    }
}

// Helper function to create test tax lots
fn create_lot(
    id: &str,
    asset: &str,
    quantity: &str,
    cost_basis: &str,
    date: &str,
) -> TaxLot {
    let qty = Decimal::from_str(quantity).unwrap();
    TaxLot {
        id: id.to_string(),
        transaction_id: format!("buy-{}", id),
        asset: asset.to_string(),
        quantity: qty,
        remaining_quantity: qty,
        cost_basis: Decimal::from_str(cost_basis).unwrap(),
        acquisition_date: chrono::DateTime::parse_from_rfc3339(date)
            .unwrap()
            .with_timezone(&Utc),
    }
}

#[test]
fn test_fifo_basic_workflow() {
    let sale = create_sale(
        "sale1",
        "BTC",
        "1.0",
        "60000",
        "2024-06-01T00:00:00Z"
    );

    let lot = create_lot(
        "lot1",
        "BTC",
        "1.0",
        "50000",
        "2023-01-01T00:00:00Z"
    );

    let result = calculate_fifo_disposal(&sale, vec![lot]).unwrap();

    assert_eq!(result.disposals.len(), 1);
    assert_eq!(result.disposals[0].gain_loss, Decimal::from_str("10000").unwrap());
    assert!(result.disposals[0].is_long_term);
}

#[test]
fn test_fifo_multiple_lots_partial_usage() {
    // Sell 2.5 BTC from 3 lots of 1.0 each
    let sale = create_sale(
        "sale1",
        "BTC",
        "2.5",
        "60000",
        "2024-06-01T00:00:00Z"
    );

    let lots = vec![
        create_lot("lot1", "BTC", "1.0", "40000", "2023-01-01T00:00:00Z"),
        create_lot("lot2", "BTC", "1.0", "45000", "2023-03-01T00:00:00Z"),
        create_lot("lot3", "BTC", "1.0", "50000", "2023-06-01T00:00:00Z"),
    ];

    let result = calculate_fifo_disposal(&sale, lots).unwrap();

    // Should use lot1 (1.0), lot2 (1.0), and lot3 (0.5)
    assert_eq!(result.disposals.len(), 3);
    assert_eq!(result.disposals[0].quantity, Decimal::from_str("1.0").unwrap());
    assert_eq!(result.disposals[1].quantity, Decimal::from_str("1.0").unwrap());
    assert_eq!(result.disposals[2].quantity, Decimal::from_str("0.5").unwrap());

    // Verify lot3 has remaining quantity
    let lot3_updated = result.updated_lots.iter().find(|l| l.id == "lot3").unwrap();
    assert_eq!(lot3_updated.remaining_quantity, Decimal::from_str("0.5").unwrap());
}

#[test]
fn test_fifo_chronological_ordering() {
    // Test that lots are processed by date, not input order
    let sale = create_sale(
        "sale1",
        "ETH",
        "5.0",
        "3000",
        "2024-06-01T00:00:00Z"
    );

    // Input lots in reverse chronological order
    let lots = vec![
        create_lot("lot3", "ETH", "2.0", "7000", "2023-12-01T00:00:00Z"),
        create_lot("lot1", "ETH", "2.0", "5000", "2023-01-01T00:00:00Z"),
        create_lot("lot2", "ETH", "2.0", "6000", "2023-06-01T00:00:00Z"),
    ];

    let result = calculate_fifo_disposal(&sale, lots).unwrap();

    // Should process in chronological order: lot1, lot2, lot3
    assert_eq!(result.disposals[0].lot_id, "lot1");
    assert_eq!(result.disposals[1].lot_id, "lot2");
    assert_eq!(result.disposals[2].lot_id, "lot3");

    // Verify quantities
    assert_eq!(result.disposals[0].quantity, Decimal::from_str("2.0").unwrap());
    assert_eq!(result.disposals[1].quantity, Decimal::from_str("2.0").unwrap());
    assert_eq!(result.disposals[2].quantity, Decimal::from_str("1.0").unwrap());
}

#[test]
fn test_fifo_short_vs_long_term_mixed() {
    let sale = create_sale(
        "sale1",
        "BTC",
        "3.0",
        "60000",
        "2024-06-01T00:00:00Z"
    );

    let lots = vec![
        create_lot("lot1", "BTC", "1.0", "40000", "2022-01-01T00:00:00Z"), // Long-term
        create_lot("lot2", "BTC", "1.0", "45000", "2023-01-01T00:00:00Z"), // Long-term
        create_lot("lot3", "BTC", "1.0", "50000", "2024-01-01T00:00:00Z"), // Short-term
    ];

    let result = calculate_fifo_disposal(&sale, lots).unwrap();

    assert_eq!(result.disposals.len(), 3);
    assert!(result.disposals[0].is_long_term);
    assert!(result.disposals[1].is_long_term);
    assert!(!result.disposals[2].is_long_term);
}

#[test]
fn test_fifo_exact_boundary_one_year() {
    // Test exactly 365 days holding period
    let sale = create_sale(
        "sale1",
        "BTC",
        "1.0",
        "60000",
        "2024-01-01T00:00:00Z"
    );

    let lot = create_lot(
        "lot1",
        "BTC",
        "1.0",
        "50000",
        "2023-01-01T00:00:00Z" // Exactly 365 days before
    );

    let result = calculate_fifo_disposal(&sale, vec![lot]).unwrap();

    assert!(result.disposals[0].is_long_term);
}

#[test]
fn test_fifo_insufficient_quantity_error() {
    let sale = create_sale(
        "sale1",
        "BTC",
        "10.0",
        "60000",
        "2024-06-01T00:00:00Z"
    );

    let lots = vec![
        create_lot("lot1", "BTC", "2.0", "40000", "2023-01-01T00:00:00Z"),
        create_lot("lot2", "BTC", "3.0", "45000", "2023-03-01T00:00:00Z"),
    ];

    let result = calculate_fifo_disposal(&sale, lots);
    assert!(result.is_err());
}

#[test]
fn test_fifo_zero_quantity_lots_skipped() {
    let sale = create_sale(
        "sale1",
        "BTC",
        "1.0",
        "60000",
        "2024-06-01T00:00:00Z"
    );

    let mut lots = vec![
        create_lot("lot1", "BTC", "1.0", "40000", "2023-01-01T00:00:00Z"),
        create_lot("lot2", "BTC", "1.0", "45000", "2023-03-01T00:00:00Z"),
    ];

    // Set lot1 remaining quantity to zero
    lots[0].remaining_quantity = Decimal::ZERO;

    let result = calculate_fifo_disposal(&sale, lots).unwrap();

    // Should skip lot1 and use lot2
    assert_eq!(result.disposals.len(), 1);
    assert_eq!(result.disposals[0].lot_id, "lot2");
}

#[test]
fn test_fifo_realistic_crypto_scenario() {
    // Realistic scenario: DCA into BTC, then sell some
    let sale = create_sale(
        "sale1",
        "BTC",
        "0.5",
        "65000",
        "2024-06-15T00:00:00Z"
    );

    let lots = vec![
        create_lot("lot1", "BTC", "0.1", "5000", "2023-01-15T00:00:00Z"),  // $50k/BTC
        create_lot("lot2", "BTC", "0.15", "7500", "2023-03-15T00:00:00Z"), // $50k/BTC
        create_lot("lot3", "BTC", "0.2", "12000", "2023-06-15T00:00:00Z"), // $60k/BTC
        create_lot("lot4", "BTC", "0.25", "15000", "2023-09-15T00:00:00Z"), // $60k/BTC
    ];

    let result = calculate_fifo_disposal(&sale, lots).unwrap();

    // Should use lots 1, 2, 3, and part of 4
    assert_eq!(result.disposals.len(), 4);

    // Calculate total gain
    let total_gain: Decimal = result.disposals.iter()
        .map(|d| d.gain_loss)
        .sum();

    // Total proceeds: 0.5 * 65000 = 32500
    // Total cost: 5000 + 7500 + 12000 + (0.05 * 60000) = 27500
    // Gain: 5000
    assert_eq!(total_gain, Decimal::from_str("5000").unwrap());
}

#[test]
fn test_fifo_loss_scenario() {
    let sale = create_sale(
        "sale1",
        "BTC",
        "1.0",
        "40000",
        "2024-06-01T00:00:00Z"
    );

    let lot = create_lot(
        "lot1",
        "BTC",
        "1.0",
        "60000",
        "2023-01-01T00:00:00Z"
    );

    let result = calculate_fifo_disposal(&sale, vec![lot]).unwrap();

    assert_eq!(result.disposals[0].gain_loss, Decimal::from_str("-20000").unwrap());
}

#[test]
fn test_fifo_fractional_crypto_amounts() {
    // Test with very small fractional amounts
    let sale = create_sale(
        "sale1",
        "BTC",
        "0.00123456",
        "60000",
        "2024-06-01T00:00:00Z"
    );

    let lot = create_lot(
        "lot1",
        "BTC",
        "0.01",
        "500",
        "2023-01-01T00:00:00Z"
    );

    let result = calculate_fifo_disposal(&sale, vec![lot]).unwrap();

    assert_eq!(result.disposals.len(), 1);
    assert_eq!(result.disposals[0].quantity, Decimal::from_str("0.00123456").unwrap());

    // Verify calculations
    let unit_cost = Decimal::from_str("500").unwrap() / Decimal::from_str("0.01").unwrap();
    let expected_cost_basis = unit_cost * Decimal::from_str("0.00123456").unwrap();
    assert_eq!(result.disposals[0].cost_basis, expected_cost_basis);
}

#[test]
fn test_fifo_multiple_assets_separate() {
    // FIFO should only consider lots for the same asset
    let sale = create_sale(
        "sale1",
        "BTC",
        "1.0",
        "60000",
        "2024-06-01T00:00:00Z"
    );

    // Mix BTC and ETH lots
    let lots = vec![
        create_lot("lot1", "BTC", "1.0", "50000", "2023-01-01T00:00:00Z"),
        create_lot("lot2", "ETH", "10.0", "30000", "2023-01-01T00:00:00Z"),
    ];

    let result = calculate_fifo_disposal(&sale, lots).unwrap();

    // Should only use BTC lot
    assert_eq!(result.disposals.len(), 1);
    assert_eq!(result.disposals[0].asset, "BTC");
}

#[test]
fn test_fifo_performance_many_lots() {
    // Performance test: process 100 lots
    use std::time::Instant;

    let sale = create_sale(
        "sale1",
        "BTC",
        "50.0",
        "60000",
        "2024-06-01T00:00:00Z"
    );

    // Create 100 lots of 1.0 BTC each
    let lots: Vec<TaxLot> = (0..100)
        .map(|i| {
            create_lot(
                &format!("lot{}", i),
                "BTC",
                "1.0",
                &format!("{}", 40000 + i * 100),
                &format!("2023-01-{}T00:00:00Z", (i % 28) + 1),
            )
        })
        .collect();

    let start = Instant::now();
    let result = calculate_fifo_disposal(&sale, lots).unwrap();
    let duration = start.elapsed();

    // Should use first 50 lots
    assert_eq!(result.disposals.len(), 50);

    // Performance target: < 10ms for 100 lots (we're processing 50)
    assert!(duration.as_millis() < 10, "FIFO took {:?}, expected < 10ms", duration);
}
