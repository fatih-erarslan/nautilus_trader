/*!
 * LIFO (Last-In, First-Out) Integration Tests
 *
 * Comprehensive tests for LIFO tax calculation method
 */

use agentic_accounting_rust_core::types::{TaxLot, Transaction, TransactionType};
use agentic_accounting_rust_core::tax::lifo::calculate_lifo_internal;
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
fn test_lifo_rising_market() {
    // Rising market: LIFO advantage (uses higher cost basis)
    let sale = create_sale(dec!(3.0), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(40000), 500), // $40k
        create_lot("lot2", dec!(1.0), dec!(50000), 300), // $50k
        create_lot("lot3", dec!(1.0), dec!(55000), 100), // $55k (newest)
    ];

    let disposals = calculate_lifo_internal(&sale, &mut lots).unwrap();

    // LIFO uses newest lots first (reverse order)
    assert_eq!(disposals.len(), 3);
    assert_eq!(disposals[0].lot_id, "lot3");
    assert_eq!(disposals[1].lot_id, "lot2");
    assert_eq!(disposals[2].lot_id, "lot1");

    // Total proceeds: 180k
    // Total cost basis: 55k + 50k + 40k = 145k
    // Total gain: 35k (same as FIFO, but different lot order)
    let total_gain: Decimal = disposals.iter().map(|d| d.gain_loss).sum();
    assert_eq!(total_gain, dec!(35000));
}

#[test]
fn test_lifo_reduces_short_term_in_rising_market() {
    // In rising markets with recent purchases at higher prices,
    // LIFO results in lower gains than FIFO
    let sale = create_sale(dec!(1.0), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(40000), 500), // Old, low cost
        create_lot("lot2", dec!(1.0), dec!(58000), 100), // New, high cost
    ];

    let disposals = calculate_lifo_internal(&sale, &mut lots).unwrap();

    // LIFO uses lot2 (higher cost = smaller gain)
    assert_eq!(disposals[0].lot_id, "lot2");
    assert_eq!(disposals[0].cost_basis, dec!(58000));
    assert_eq!(disposals[0].gain_loss, dec!(2000));

    // This gain is much smaller than if we used lot1 (would be 20000 gain)
}

#[test]
fn test_lifo_falling_market() {
    // Falling market: older lots have higher cost
    let sale = create_sale(dec!(2.0), dec!(45000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(60000), 500),
        create_lot("lot2", dec!(1.0), dec!(50000), 300),
        create_lot("lot3", dec!(1.0), dec!(40000), 100), // Newest, lowest cost
    ];

    let disposals = calculate_lifo_internal(&sale, &mut lots).unwrap();

    // LIFO uses newest lots first
    assert_eq!(disposals.len(), 2);
    assert_eq!(disposals[0].lot_id, "lot3");
    assert_eq!(disposals[1].lot_id, "lot2");

    // Total proceeds: 2 * 45000 = 90k
    // Total cost basis: 40k + 50k = 90k
    // Total gain/loss: 0
    let total_gain: Decimal = disposals.iter().map(|d| d.gain_loss).sum();
    assert_eq!(total_gain, dec!(0));
}

#[test]
fn test_lifo_short_term_tendency() {
    // LIFO tends to create more short-term disposals
    let sale = create_sale(dec!(2.0), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(50000), 400), // Long-term
        create_lot("lot2", dec!(1.0), dec!(55000), 200), // Short-term
    ];

    let disposals = calculate_lifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 2);

    // First disposal should be from lot2 (newest, short-term)
    assert_eq!(disposals[0].lot_id, "lot2");
    assert!(!disposals[0].is_long_term);

    // Second disposal from lot1 (older, long-term)
    assert_eq!(disposals[1].lot_id, "lot1");
    assert!(disposals[1].is_long_term);
}

#[test]
fn test_lifo_partial_lot_usage() {
    let sale = create_sale(dec!(1.5), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(50000), 400),
        create_lot("lot2", dec!(2.0), dec!(100000), 100), // Newest
    ];

    let disposals = calculate_lifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 1);

    // Should use lot2 (newest) but only partially
    assert_eq!(disposals[0].lot_id, "lot2");
    assert_eq!(disposals[0].quantity, dec!(1.5));
    assert_eq!(lots[1].remaining_quantity, dec!(0.5));

    // lot1 should remain untouched
    assert_eq!(lots[0].remaining_quantity, dec!(1.0));
}

#[test]
fn test_lifo_same_date_lots() {
    // When lots have same acquisition date, order is stable
    let base_date = Utc::now() - Duration::days(400);

    let sale = Transaction {
        id: "sale1".to_string(),
        transaction_type: TransactionType::Sell,
        asset: "BTC".to_string(),
        quantity: dec!(2.0),
        price: dec!(60000),
        timestamp: Utc::now(),
        source: "test".to_string(),
        fees: Decimal::ZERO,
    };

    let mut lots = vec![
        TaxLot {
            id: "lot1".to_string(),
            transaction_id: "tx_lot1".to_string(),
            asset: "BTC".to_string(),
            quantity: dec!(1.0),
            remaining_quantity: dec!(1.0),
            cost_basis: dec!(50000),
            acquisition_date: base_date,
        },
        TaxLot {
            id: "lot2".to_string(),
            transaction_id: "tx_lot2".to_string(),
            asset: "BTC".to_string(),
            quantity: dec!(1.0),
            remaining_quantity: dec!(1.0),
            cost_basis: dec!(55000),
            acquisition_date: base_date,
        },
    ];

    let disposals = calculate_lifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 2);

    // Both lots should be used (order doesn't matter when dates are same)
    let total_quantity: Decimal = disposals.iter().map(|d| d.quantity).sum();
    assert_eq!(total_quantity, dec!(2.0));
}

#[test]
fn test_lifo_with_fees() {
    let mut sale = create_sale(dec!(1.0), dec!(60000));
    sale.fees = dec!(100);

    let mut lots = vec![create_lot("lot1", dec!(1.0), dec!(50000), 100)];

    let disposals = calculate_lifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 1);

    // Proceeds should be reduced by fees
    assert_eq!(disposals[0].proceeds, dec!(59900));
    assert_eq!(disposals[0].gain_loss, dec!(9900));
}

#[test]
fn test_lifo_insufficient_quantity() {
    let sale = create_sale(dec!(5.0), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(50000), 400),
        create_lot("lot2", dec!(1.0), dec!(55000), 200),
    ];

    let result = calculate_lifo_internal(&sale, &mut lots);
    assert!(result.is_err());

    if let Err(e) = result {
        assert!(e.to_string().contains("Insufficient quantity"));
    }
}

#[test]
fn test_lifo_zero_remaining_quantity_lots() {
    // Lots with zero remaining quantity should be skipped
    let sale = create_sale(dec!(1.0), dec!(60000));
    let mut lots = vec![
        TaxLot {
            id: "lot1".to_string(),
            transaction_id: "tx_lot1".to_string(),
            asset: "BTC".to_string(),
            quantity: dec!(1.0),
            remaining_quantity: dec!(0.0), // Already used
            cost_basis: dec!(60000),
            acquisition_date: Utc::now() - Duration::days(100),
        },
        create_lot("lot2", dec!(1.0), dec!(50000), 400),
    ];

    let disposals = calculate_lifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 1);
    assert_eq!(disposals[0].lot_id, "lot2");
}

#[test]
fn test_lifo_multiple_assets_filtered() {
    // Ensure LIFO only uses lots for the correct asset
    let sale = create_sale(dec!(1.0), dec!(60000));
    let mut lots = vec![
        TaxLot {
            id: "lot1".to_string(),
            transaction_id: "tx_lot1".to_string(),
            asset: "ETH".to_string(), // Different asset (newer)
            quantity: dec!(10.0),
            remaining_quantity: dec!(10.0),
            cost_basis: dec!(20000),
            acquisition_date: Utc::now() - Duration::days(100),
        },
        create_lot("lot2", dec!(1.0), dec!(50000), 400),
    ];

    let disposals = calculate_lifo_internal(&sale, &mut lots).unwrap();

    // Should only use BTC lot
    assert_eq!(disposals.len(), 1);
    assert_eq!(disposals[0].lot_id, "lot2");
    assert_eq!(disposals[0].asset, "BTC");
}

#[test]
fn test_lifo_fractional_quantities() {
    let sale = create_sale(dec!(0.75), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(0.5), dec!(25000), 400),
        create_lot("lot2", dec!(0.3), dec!(18000), 100), // Newest
    ];

    let disposals = calculate_lifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 2);

    // Use lot2 first (newest)
    assert_eq!(disposals[0].lot_id, "lot2");
    assert_eq!(disposals[0].quantity, dec!(0.3));

    // Then use 0.45 from lot1
    assert_eq!(disposals[1].lot_id, "lot1");
    assert_eq!(disposals[1].quantity, dec!(0.45));
}

#[test]
fn test_lifo_performance_1000_lots() {
    use std::time::Instant;

    // Create 1000 lots (reverse chronological order for LIFO)
    let mut lots: Vec<TaxLot> = (0..1000)
        .map(|i| create_lot(&format!("lot{}", i), dec!(0.01), dec!(500), i))
        .collect();

    let sale = create_sale(dec!(5.0), dec!(60000));

    let start = Instant::now();
    let disposals = calculate_lifo_internal(&sale, &mut lots).unwrap();
    let elapsed = start.elapsed();

    // Should complete in <10ms
    assert!(elapsed.as_millis() < 10, "LIFO took {}ms, should be <10ms", elapsed.as_millis());

    // Verify correctness
    let total_quantity: Decimal = disposals.iter().map(|d| d.quantity).sum();
    assert_eq!(total_quantity, dec!(5.0));

    // Verify LIFO order - should use most recent lots (smallest days_ago values)
    assert!(disposals[0].acquisition_date > disposals[disposals.len() - 1].acquisition_date);
}

#[test]
fn test_lifo_exact_quantity_match() {
    let sale = create_sale(dec!(2.5), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(50000), 400),
        create_lot("lot2", dec!(1.5), dec!(75000), 200), // Newest
    ];

    let disposals = calculate_lifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 2);

    let total_quantity: Decimal = disposals.iter().map(|d| d.quantity).sum();
    assert_eq!(total_quantity, dec!(2.5));

    // Both lots fully depleted
    assert_eq!(lots[0].remaining_quantity, dec!(0.0));
    assert_eq!(lots[1].remaining_quantity, dec!(0.0));
}
