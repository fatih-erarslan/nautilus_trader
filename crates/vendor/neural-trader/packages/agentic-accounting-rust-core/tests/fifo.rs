/*!
 * FIFO (First-In, First-Out) Integration Tests
 *
 * Comprehensive tests for FIFO tax calculation method
 */

use agentic_accounting_rust_core::tax::fifo::calculate_fifo_internal;
use agentic_accounting_rust_core::types::{TaxLot, Transaction, TransactionType};
use chrono::{Duration, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

fn create_sale(quantity: Decimal, price: Decimal) -> Transaction {
    Transaction {
        id: "sale1".to_string(),
        transaction_type: TransactionType::Sell,
        asset: "BTC".to_string(),
        quantity,
        price,
        timestamp: Utc::now(),
        source: "test".to_string(),
        fees: Decimal::ZERO,
    }
}

fn create_lot(id: &str, quantity: Decimal, cost_basis: Decimal, days_ago: i64) -> TaxLot {
    TaxLot {
        id: id.to_string(),
        transaction_id: format!("tx_{}", id),
        asset: "BTC".to_string(),
        quantity,
        remaining_quantity: quantity,
        cost_basis,
        acquisition_date: Utc::now() - Duration::days(days_ago),
    }
}

#[test]
fn test_fifo_rising_market() {
    // Rising market: older lots have lower cost basis
    let sale = create_sale(dec!(3.0), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(40000), 500), // $40k
        create_lot("lot2", dec!(1.0), dec!(50000), 300), // $50k
        create_lot("lot3", dec!(1.0), dec!(55000), 100), // $55k
    ];

    let disposals = calculate_fifo_internal(&sale, &mut lots).unwrap();

    // FIFO uses oldest lots first
    assert_eq!(disposals.len(), 3);
    assert_eq!(disposals[0].lot_id, "lot1");
    assert_eq!(disposals[1].lot_id, "lot2");
    assert_eq!(disposals[2].lot_id, "lot3");

    // Total gain: 180k proceeds - 145k cost basis = 35k
    let total_gain: Decimal = disposals.iter().map(|d| d.gain_loss).sum();
    assert_eq!(total_gain, dec!(35000));
}

#[test]
fn test_fifo_falling_market() {
    // Falling market: FIFO may have advantage as older lots have higher cost
    let sale = create_sale(dec!(2.0), dec!(45000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(60000), 500),
        create_lot("lot2", dec!(1.0), dec!(50000), 300),
        create_lot("lot3", dec!(1.0), dec!(40000), 100),
    ];

    let disposals = calculate_fifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 2);

    // Total proceeds: 2 * 45000 = 90k
    // Total cost basis: 60k + 50k = 110k
    // Total loss: -20k
    let total_gain: Decimal = disposals.iter().map(|d| d.gain_loss).sum();
    assert_eq!(total_gain, dec!(-20000));
}

#[test]
fn test_fifo_long_vs_short_term() {
    let sale = create_sale(dec!(2.0), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(50000), 400), // Long-term (>365 days)
        create_lot("lot2", dec!(1.0), dec!(55000), 200), // Short-term (<365 days)
    ];

    let disposals = calculate_fifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 2);
    assert!(disposals[0].is_long_term);
    assert!(!disposals[1].is_long_term);
}

#[test]
fn test_fifo_exact_quantity_match() {
    let sale = create_sale(dec!(2.5), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(50000), 400),
        create_lot("lot2", dec!(1.5), dec!(75000), 200),
    ];

    let disposals = calculate_fifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 2);

    let total_quantity: Decimal = disposals.iter().map(|d| d.quantity).sum();
    assert_eq!(total_quantity, dec!(2.5));

    // Check remaining quantities
    assert_eq!(lots[0].remaining_quantity, dec!(0.0));
    assert_eq!(lots[1].remaining_quantity, dec!(0.0));
}

#[test]
fn test_fifo_partial_lot_depletion() {
    let sale = create_sale(dec!(1.5), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(50000), 400),
        create_lot("lot2", dec!(2.0), dec!(100000), 200),
    ];

    let disposals = calculate_fifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 2);

    // First lot fully depleted
    assert_eq!(disposals[0].quantity, dec!(1.0));
    assert_eq!(lots[0].remaining_quantity, dec!(0.0));

    // Second lot partially used
    assert_eq!(disposals[1].quantity, dec!(0.5));
    assert_eq!(lots[1].remaining_quantity, dec!(1.5));
}

#[test]
fn test_fifo_with_fees() {
    let mut sale = create_sale(dec!(1.0), dec!(60000));
    sale.fees = dec!(100);

    let mut lots = vec![create_lot("lot1", dec!(1.0), dec!(50000), 400)];

    let disposals = calculate_fifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 1);

    // Proceeds should be reduced by fees: 60000 - 100 = 59900
    assert_eq!(disposals[0].proceeds, dec!(59900));
    // Gain: 59900 - 50000 = 9900
    assert_eq!(disposals[0].gain_loss, dec!(9900));
}

#[test]
fn test_fifo_zero_cost_basis() {
    // Test airdrop or gift with zero cost basis
    let sale = create_sale(dec!(1.0), dec!(60000));
    let mut lots = vec![create_lot("lot1", dec!(1.0), dec!(0), 400)];

    let disposals = calculate_fifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 1);
    assert_eq!(disposals[0].cost_basis, dec!(0));
    assert_eq!(disposals[0].gain_loss, dec!(60000)); // All proceeds are gain
}

#[test]
fn test_fifo_micro_quantities() {
    // Test with very small (satoshi-level) quantities
    let sale = create_sale(dec!(0.00000001), dec!(60000));
    let mut lots = vec![create_lot("lot1", dec!(0.00000001), dec!(0.0005), 400)];

    let disposals = calculate_fifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 1);
    assert_eq!(disposals[0].quantity, dec!(0.00000001));
}

#[test]
fn test_fifo_large_quantities() {
    // Test with large quantities
    let sale = create_sale(dec!(1000000), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(500000), dec!(25000000), 400),
        create_lot("lot2", dec!(500000), dec!(25000000), 200),
    ];

    let disposals = calculate_fifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 2);

    let total_quantity: Decimal = disposals.iter().map(|d| d.quantity).sum();
    assert_eq!(total_quantity, dec!(1000000));
}

#[test]
fn test_fifo_multiple_assets_filtered() {
    // Ensure FIFO only uses lots for the correct asset
    let sale = create_sale(dec!(1.0), dec!(60000));
    let mut lots = vec![
        TaxLot {
            id: "lot1".to_string(),
            transaction_id: "tx_lot1".to_string(),
            asset: "ETH".to_string(), // Different asset
            quantity: dec!(10.0),
            remaining_quantity: dec!(10.0),
            cost_basis: dec!(20000),
            acquisition_date: Utc::now() - Duration::days(500),
        },
        create_lot("lot2", dec!(1.0), dec!(50000), 400),
    ];

    let disposals = calculate_fifo_internal(&sale, &mut lots).unwrap();

    // Should only use BTC lot
    assert_eq!(disposals.len(), 1);
    assert_eq!(disposals[0].lot_id, "lot2");
}

#[test]
fn test_fifo_performance_1000_lots() {
    use std::time::Instant;

    // Create 1000 lots
    let mut lots: Vec<TaxLot> = (0..1000)
        .map(|i| create_lot(&format!("lot{}", i), dec!(0.01), dec!(500), 1000 - i))
        .collect();

    let sale = create_sale(dec!(5.0), dec!(60000));

    let start = Instant::now();
    let disposals = calculate_fifo_internal(&sale, &mut lots).unwrap();
    let elapsed = start.elapsed();

    // Should complete in <10ms
    assert!(elapsed.as_millis() < 10, "FIFO took {}ms, should be <10ms", elapsed.as_millis());

    // Verify correctness
    let total_quantity: Decimal = disposals.iter().map(|d| d.quantity).sum();
    assert_eq!(total_quantity, dec!(5.0));
}
