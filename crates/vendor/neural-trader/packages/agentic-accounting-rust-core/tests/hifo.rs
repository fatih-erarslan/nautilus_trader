/*!
 * HIFO (Highest-In, First-Out) Integration Tests
 *
 * Comprehensive tests for HIFO tax calculation method
 */

use agentic_accounting_rust_core::types::{TaxLot, Transaction, TransactionType};
use agentic_accounting_rust_core::tax::hifo::calculate_hifo_internal;
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
fn test_hifo_single_lot() {
    let sale = create_sale(dec!(1.0), dec!(60000));
    let mut lots = vec![create_lot("lot1", dec!(1.0), dec!(50000), 400)];

    let disposals = calculate_hifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 1);
    assert_eq!(disposals[0].quantity, dec!(1.0));
    assert_eq!(disposals[0].gain_loss, dec!(10000));
}

#[test]
fn test_hifo_highest_cost_first() {
    // HIFO should always use lots with highest unit cost basis first
    let sale = create_sale(dec!(3.0), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(40000), 500), // Unit: $40,000
        create_lot("lot2", dec!(1.0), dec!(58000), 300), // Unit: $58,000
        create_lot("lot3", dec!(1.0), dec!(50000), 100), // Unit: $50,000
    ];

    let disposals = calculate_hifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 3);

    // Order should be: lot2 ($58k), lot3 ($50k), lot1 ($40k)
    assert_eq!(disposals[0].lot_id, "lot2");
    assert_eq!(disposals[1].lot_id, "lot3");
    assert_eq!(disposals[2].lot_id, "lot1");
}

#[test]
fn test_hifo_minimizes_gains() {
    // HIFO minimizes taxable gains by using highest cost basis
    let sale = create_sale(dec!(1.0), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(40000), 400),  // Gain: $20k
        create_lot("lot2", dec!(1.0), dec!(58000), 200),  // Gain: $2k
    ];

    let disposals = calculate_hifo_internal(&sale, &mut lots).unwrap();

    // Should use lot2 (highest cost = lowest gain)
    assert_eq!(disposals[0].lot_id, "lot2");
    assert_eq!(disposals[0].cost_basis, dec!(58000));
    assert_eq!(disposals[0].gain_loss, dec!(2000));
}

#[test]
fn test_hifo_volatile_market() {
    // HIFO excels in volatile markets with varying purchase prices
    let sale = create_sale(dec!(4.0), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(40000), 500),  // Unit: $40k
        create_lot("lot2", dec!(1.0), dec!(59000), 400),  // Unit: $59k (highest)
        create_lot("lot3", dec!(1.0), dec!(45000), 300),  // Unit: $45k
        create_lot("lot4", dec!(1.0), dec!(58000), 200),  // Unit: $58k
    ];

    let disposals = calculate_hifo_internal(&sale, &mut lots).unwrap();

    // Order: lot2 ($59k), lot4 ($58k), lot3 ($45k), lot1 ($40k)
    assert_eq!(disposals[0].lot_id, "lot2");
    assert_eq!(disposals[1].lot_id, "lot4");
    assert_eq!(disposals[2].lot_id, "lot3");
    assert_eq!(disposals[3].lot_id, "lot1");

    // Total cost basis: 59k + 58k + 45k + 40k = 202k
    // Total proceeds: 60k * 4 = 240k
    // Total gain: 38k (minimized compared to other methods)
    let total_gain: Decimal = disposals.iter().map(|d| d.gain_loss).sum();
    assert_eq!(total_gain, dec!(38000));
}

#[test]
fn test_hifo_fractional_quantities() {
    // Test unit cost basis calculation with fractional quantities
    let sale = create_sale(dec!(2.0), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(0.5), dec!(20000), 400),   // Unit: $40k
        create_lot("lot2", dec!(1.0), dec!(58000), 300),   // Unit: $58k
        create_lot("lot3", dec!(0.75), dec!(45000), 200),  // Unit: $60k (highest)
    ];

    let disposals = calculate_hifo_internal(&sale, &mut lots).unwrap();

    // Should use lot3 (unit: $60k) first, then lot2 ($58k)
    assert_eq!(disposals[0].lot_id, "lot3");
    assert_eq!(disposals[0].quantity, dec!(0.75));

    assert_eq!(disposals[1].lot_id, "lot2");
    assert_eq!(disposals[1].quantity, dec!(1.0));

    assert_eq!(disposals[2].lot_id, "lot1");
    assert_eq!(disposals[2].quantity, dec!(0.25));
}

#[test]
fn test_hifo_with_losses() {
    // HIFO with losses - uses highest cost (biggest loss) first
    let sale = create_sale(dec!(2.0), dec!(50000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(55000), 400),  // Loss: $5k
        create_lot("lot2", dec!(1.0), dec!(60000), 300),  // Loss: $10k (highest cost)
    ];

    let disposals = calculate_hifo_internal(&sale, &mut lots).unwrap();

    // Should use lot2 first (highest cost = biggest loss)
    assert_eq!(disposals[0].lot_id, "lot2");
    assert_eq!(disposals[0].gain_loss, dec!(-10000));

    assert_eq!(disposals[1].lot_id, "lot1");
    assert_eq!(disposals[1].gain_loss, dec!(-5000));
}

#[test]
fn test_hifo_same_cost_basis_tie() {
    // When lots have same unit cost basis, order is stable
    let sale = create_sale(dec!(2.0), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(55000), 400),
        create_lot("lot2", dec!(1.0), dec!(55000), 200),
    ];

    let disposals = calculate_hifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 2);

    // Both lots have same unit cost, so both used
    let total_cost: Decimal = disposals.iter().map(|d| d.cost_basis).sum();
    assert_eq!(total_cost, dec!(110000));
}

#[test]
fn test_hifo_partial_lot_usage() {
    let sale = create_sale(dec!(0.5), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(50000), 400),  // Unit: $50k
        create_lot("lot2", dec!(1.0), dec!(58000), 200),  // Unit: $58k (highest)
    ];

    let disposals = calculate_hifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 1);

    // Should use lot2 partially
    assert_eq!(disposals[0].lot_id, "lot2");
    assert_eq!(disposals[0].quantity, dec!(0.5));
    assert_eq!(lots[1].remaining_quantity, dec!(0.5));

    // lot1 untouched
    assert_eq!(lots[0].remaining_quantity, dec!(1.0));
}

#[test]
fn test_hifo_zero_cost_basis() {
    // Airdrop or gift with zero cost basis should be used last
    let sale = create_sale(dec!(2.0), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(0), 400),      // Unit: $0 (lowest)
        create_lot("lot2", dec!(1.0), dec!(58000), 200),  // Unit: $58k (highest)
    ];

    let disposals = calculate_hifo_internal(&sale, &mut lots).unwrap();

    // Use lot2 first (highest cost), then lot1
    assert_eq!(disposals[0].lot_id, "lot2");
    assert_eq!(disposals[1].lot_id, "lot1");
}

#[test]
fn test_hifo_with_fees() {
    let mut sale = create_sale(dec!(1.0), dec!(60000));
    sale.fees = dec!(100);

    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(50000), 400),
        create_lot("lot2", dec!(1.0), dec!(55000), 200),
    ];

    let disposals = calculate_hifo_internal(&sale, &mut lots).unwrap();

    // Should use lot2 (higher cost)
    assert_eq!(disposals[0].lot_id, "lot2");

    // Proceeds reduced by fees
    assert_eq!(disposals[0].proceeds, dec!(59900));
    assert_eq!(disposals[0].gain_loss, dec!(4900));
}

#[test]
fn test_hifo_insufficient_quantity() {
    let sale = create_sale(dec!(5.0), dec!(60000));
    let mut lots = vec![create_lot("lot1", dec!(1.0), dec!(50000), 400)];

    let result = calculate_hifo_internal(&sale, &mut lots);
    assert!(result.is_err());

    if let Err(e) = result {
        assert!(e.to_string().contains("Insufficient quantity"));
    }
}

#[test]
fn test_hifo_multiple_assets_filtered() {
    let sale = create_sale(dec!(1.0), dec!(60000));
    let mut lots = vec![
        TaxLot {
            id: "lot1".to_string(),
            transaction_id: "tx_lot1".to_string(),
            asset: "ETH".to_string(), // Different asset, higher cost
            quantity: dec!(10.0),
            remaining_quantity: dec!(10.0),
            cost_basis: dec!(100000),
            acquisition_date: Utc::now() - Duration::days(200),
        },
        create_lot("lot2", dec!(1.0), dec!(50000), 400),
    ];

    let disposals = calculate_hifo_internal(&sale, &mut lots).unwrap();

    // Should only use BTC lot
    assert_eq!(disposals.len(), 1);
    assert_eq!(disposals[0].lot_id, "lot2");
    assert_eq!(disposals[0].asset, "BTC");
}

#[test]
fn test_hifo_different_holding_periods() {
    // HIFO ignores holding period, only considers unit cost
    let sale = create_sale(dec!(2.0), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(55000), 400),  // Long-term, unit: $55k
        create_lot("lot2", dec!(1.0), dec!(58000), 100),  // Short-term, unit: $58k (higher)
    ];

    let disposals = calculate_hifo_internal(&sale, &mut lots).unwrap();

    // Should use lot2 first despite being short-term
    assert_eq!(disposals[0].lot_id, "lot2");
    assert!(!disposals[0].is_long_term);

    assert_eq!(disposals[1].lot_id, "lot1");
    assert!(disposals[1].is_long_term);
}

#[test]
fn test_hifo_micro_quantities() {
    // Test with satoshi-level quantities
    let sale = create_sale(dec!(0.00000002), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(0.00000001), dec!(0.0005), 400),  // Unit: $50k
        create_lot("lot2", dec!(0.00000001), dec!(0.0006), 200),  // Unit: $60k (higher)
    ];

    let disposals = calculate_hifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 2);
    assert_eq!(disposals[0].lot_id, "lot2");
}

#[test]
fn test_hifo_performance_1000_lots() {
    use std::time::Instant;

    // Create 1000 lots with varying cost bases
    let mut lots: Vec<TaxLot> = (0..1000)
        .map(|i| {
            let cost = 40000 + (i * 10); // Varying costs
            create_lot(&format!("lot{}", i), dec!(0.01), Decimal::from(cost), 500 - i)
        })
        .collect();

    let sale = create_sale(dec!(5.0), dec!(60000));

    let start = Instant::now();
    let disposals = calculate_hifo_internal(&sale, &mut lots).unwrap();
    let elapsed = start.elapsed();

    // Should complete in <10ms
    assert!(elapsed.as_millis() < 10, "HIFO took {}ms, should be <10ms", elapsed.as_millis());

    // Verify correctness
    let total_quantity: Decimal = disposals.iter().map(|d| d.quantity).sum();
    assert_eq!(total_quantity, dec!(5.0));

    // Verify highest cost basis lots used first
    if disposals.len() > 1 {
        for i in 0..disposals.len() - 1 {
            let unit_cost_1 = disposals[i].cost_basis / disposals[i].quantity;
            let unit_cost_2 = disposals[i + 1].cost_basis / disposals[i + 1].quantity;
            assert!(unit_cost_1 >= unit_cost_2, "HIFO order violation");
        }
    }
}

#[test]
fn test_hifo_exact_quantity_match() {
    let sale = create_sale(dec!(2.5), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(50000), 400),  // Unit: $50k
        create_lot("lot2", dec!(1.5), dec!(82500), 200),  // Unit: $55k (higher)
    ];

    let disposals = calculate_hifo_internal(&sale, &mut lots).unwrap();

    assert_eq!(disposals.len(), 2);

    let total_quantity: Decimal = disposals.iter().map(|d| d.quantity).sum();
    assert_eq!(total_quantity, dec!(2.5));

    // Both lots fully depleted
    assert_eq!(lots[0].remaining_quantity, dec!(0.0));
    assert_eq!(lots[1].remaining_quantity, dec!(0.0));
}

#[test]
fn test_hifo_complex_sorting() {
    // Complex scenario with many lots needing proper sorting
    let sale = create_sale(dec!(5.0), dec!(60000));
    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(42000), 600),
        create_lot("lot2", dec!(1.5), dec!(87000), 500),  // Unit: $58k
        create_lot("lot3", dec!(0.8), dec!(48000), 400),  // Unit: $60k
        create_lot("lot4", dec!(1.2), dec!(50400), 300),  // Unit: $42k
        create_lot("lot5", dec!(1.0), dec!(59000), 200),  // Unit: $59k
        create_lot("lot6", dec!(0.5), dec!(20000), 100),  // Unit: $40k
    ];

    let disposals = calculate_hifo_internal(&sale, &mut lots).unwrap();

    // Verify descending unit cost order
    let mut prev_unit_cost = Decimal::MAX;
    for disposal in &disposals {
        let unit_cost = disposal.cost_basis / disposal.quantity;
        assert!(unit_cost <= prev_unit_cost, "HIFO sort order violated");
        prev_unit_cost = unit_cost;
    }

    let total_quantity: Decimal = disposals.iter().map(|d| d.quantity).sum();
    assert_eq!(total_quantity, dec!(5.0));
}
