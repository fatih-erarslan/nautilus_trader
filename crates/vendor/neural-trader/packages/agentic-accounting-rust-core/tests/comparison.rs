/*!
 * Comparison Tests for FIFO, LIFO, and HIFO
 *
 * Tests that compare all three methods to validate correctness
 * and demonstrate differences in various market scenarios
 */

use agentic_accounting_rust_core::types::{TaxLot, Transaction, TransactionType};
use agentic_accounting_rust_core::tax::fifo::calculate_fifo_disposal;
use agentic_accounting_rust_core::tax::lifo::calculate_lifo_internal;
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
fn test_all_methods_same_quantity() {
    // All methods should dispose the exact quantity requested
    let sale = create_sale(dec!(2.5), dec!(60000));

    let mut lots_fifo = vec![
        create_lot("lot1", dec!(1.0), dec!(50000), 400),
        create_lot("lot2", dec!(2.0), dec!(100000), 200),
    ];

    let mut lots_lifo = lots_fifo.clone();
    let mut lots_hifo = lots_fifo.clone();

    let disposals_fifo = calculate_fifo_disposal(&sale, &mut lots_fifo).unwrap();
    let disposals_lifo = calculate_lifo_internal(&sale, &mut lots_lifo).unwrap();
    let disposals_hifo = calculate_hifo_internal(&sale, &mut lots_hifo).unwrap();

    let qty_fifo: Decimal = disposals_fifo.iter().map(|d| d.quantity).sum();
    let qty_lifo: Decimal = disposals_lifo.iter().map(|d| d.quantity).sum();
    let qty_hifo: Decimal = disposals_hifo.iter().map(|d| d.quantity).sum();

    assert_eq!(qty_fifo, dec!(2.5));
    assert_eq!(qty_lifo, dec!(2.5));
    assert_eq!(qty_hifo, dec!(2.5));
}

#[test]
fn test_all_methods_same_proceeds() {
    // All methods should calculate the same total proceeds
    let sale = create_sale(dec!(2.0), dec!(60000));

    let mut lots_fifo = vec![
        create_lot("lot1", dec!(1.0), dec!(40000), 400),
        create_lot("lot2", dec!(1.0), dec!(50000), 200),
    ];

    let mut lots_lifo = lots_fifo.clone();
    let mut lots_hifo = lots_fifo.clone();

    let disposals_fifo = calculate_fifo_disposal(&sale, &mut lots_fifo).unwrap();
    let disposals_lifo = calculate_lifo_internal(&sale, &mut lots_lifo).unwrap();
    let disposals_hifo = calculate_hifo_internal(&sale, &mut lots_hifo).unwrap();

    let proceeds_fifo: Decimal = disposals_fifo.iter().map(|d| d.proceeds).sum();
    let proceeds_lifo: Decimal = disposals_lifo.iter().map(|d| d.proceeds).sum();
    let proceeds_hifo: Decimal = disposals_hifo.iter().map(|d| d.proceeds).sum();

    let expected_proceeds = dec!(120000); // 2.0 * 60000
    assert_eq!(proceeds_fifo, expected_proceeds);
    assert_eq!(proceeds_lifo, expected_proceeds);
    assert_eq!(proceeds_hifo, expected_proceeds);
}

#[test]
fn test_rising_market_comparison() {
    // Rising market: LIFO and HIFO should have advantage over FIFO
    let sale = create_sale(dec!(2.0), dec!(60000));

    let mut lots_fifo = vec![
        create_lot("lot1", dec!(1.0), dec!(40000), 400),  // Old, cheap
        create_lot("lot2", dec!(1.0), dec!(58000), 100),  // New, expensive
    ];

    let mut lots_lifo = lots_fifo.clone();
    let mut lots_hifo = lots_fifo.clone();

    let disposals_fifo = calculate_fifo_disposal(&sale, &mut lots_fifo).unwrap();
    let disposals_lifo = calculate_lifo_internal(&sale, &mut lots_lifo).unwrap();
    let disposals_hifo = calculate_hifo_internal(&sale, &mut lots_hifo).unwrap();

    let gain_fifo: Decimal = disposals_fifo.iter().map(|d| d.gain_loss).sum();
    let gain_lifo: Decimal = disposals_lifo.iter().map(|d| d.gain_loss).sum();
    let gain_hifo: Decimal = disposals_hifo.iter().map(|d| d.gain_loss).sum();

    // FIFO gain: 120k - 98k = 22k
    // LIFO gain: 120k - 98k = 22k (same total, but uses newer lots first)
    // HIFO gain: 120k - 98k = 22k (same total, but uses highest cost first)

    // In this case, gains are same because we use all lots
    assert_eq!(gain_fifo, dec!(22000));
    assert_eq!(gain_lifo, dec!(22000));
    assert_eq!(gain_hifo, dec!(22000));
}

#[test]
fn test_partial_sale_rising_market() {
    // Partial sale in rising market shows clear difference
    let sale = create_sale(dec!(1.0), dec!(60000));

    let mut lots_fifo = vec![
        create_lot("lot1", dec!(1.0), dec!(40000), 400),
        create_lot("lot2", dec!(1.0), dec!(58000), 100),
    ];

    let mut lots_lifo = lots_fifo.clone();
    let mut lots_hifo = lots_fifo.clone();

    let disposals_fifo = calculate_fifo_disposal(&sale, &mut lots_fifo).unwrap();
    let disposals_lifo = calculate_lifo_internal(&sale, &mut lots_lifo).unwrap();
    let disposals_hifo = calculate_hifo_internal(&sale, &mut lots_hifo).unwrap();

    let gain_fifo: Decimal = disposals_fifo.iter().map(|d| d.gain_loss).sum();
    let gain_lifo: Decimal = disposals_lifo.iter().map(|d| d.gain_loss).sum();
    let gain_hifo: Decimal = disposals_hifo.iter().map(|d| d.gain_loss).sum();

    // FIFO uses lot1: 60k - 40k = 20k gain
    // LIFO uses lot2: 60k - 58k = 2k gain
    // HIFO uses lot2: 60k - 58k = 2k gain

    assert_eq!(gain_fifo, dec!(20000));
    assert_eq!(gain_lifo, dec!(2000));
    assert_eq!(gain_hifo, dec!(2000));

    // LIFO and HIFO both minimize the gain
    assert!(gain_lifo < gain_fifo);
    assert!(gain_hifo < gain_fifo);
}

#[test]
fn test_volatile_market_hifo_advantage() {
    // Volatile market with varying prices - HIFO should excel
    let sale = create_sale(dec!(3.0), dec!(60000));

    let mut lots_fifo = vec![
        create_lot("lot1", dec!(1.0), dec!(40000), 500),  // Old, low
        create_lot("lot2", dec!(1.0), dec!(59000), 400),  // High cost
        create_lot("lot3", dec!(1.0), dec!(45000), 300),  // Medium cost
        create_lot("lot4", dec!(1.0), dec!(58000), 200),  // High cost
    ];

    let mut lots_lifo = lots_fifo.clone();
    let mut lots_hifo = lots_fifo.clone();

    let disposals_fifo = calculate_fifo_disposal(&sale, &mut lots_fifo).unwrap();
    let disposals_lifo = calculate_lifo_internal(&sale, &mut lots_lifo).unwrap();
    let disposals_hifo = calculate_hifo_internal(&sale, &mut lots_hifo).unwrap();

    let gain_fifo: Decimal = disposals_fifo.iter().map(|d| d.gain_loss).sum();
    let gain_lifo: Decimal = disposals_lifo.iter().map(|d| d.gain_loss).sum();
    let gain_hifo: Decimal = disposals_hifo.iter().map(|d| d.gain_loss).sum();

    // FIFO uses: lot1 (40k), lot2 (59k), lot3 (45k) = 144k cost basis
    // LIFO uses: lot4 (58k), lot3 (45k), lot2 (59k) = 162k cost basis
    // HIFO uses: lot2 (59k), lot4 (58k), lot3 (45k) = 162k cost basis

    assert_eq!(gain_fifo, dec!(36000));  // 180k - 144k
    assert_eq!(gain_lifo, dec!(18000));  // 180k - 162k
    assert_eq!(gain_hifo, dec!(18000));  // 180k - 162k

    // LIFO and HIFO both minimize the gain
    assert!(gain_lifo < gain_fifo);
    assert!(gain_hifo < gain_fifo);
}

#[test]
fn test_falling_market_fifo_advantage() {
    // Falling market: FIFO may have advantage
    let sale = create_sale(dec!(2.0), dec!(45000));

    let mut lots_fifo = vec![
        create_lot("lot1", dec!(1.0), dec!(60000), 400),  // Old, high cost
        create_lot("lot2", dec!(1.0), dec!(50000), 200),  // Medium cost
        create_lot("lot3", dec!(1.0), dec!(40000), 100),  // New, low cost
    ];

    let mut lots_lifo = lots_fifo.clone();
    let mut lots_hifo = lots_fifo.clone();

    let disposals_fifo = calculate_fifo_disposal(&sale, &mut lots_fifo).unwrap();
    let disposals_lifo = calculate_lifo_internal(&sale, &mut lots_lifo).unwrap();
    let disposals_hifo = calculate_hifo_internal(&sale, &mut lots_hifo).unwrap();

    let gain_fifo: Decimal = disposals_fifo.iter().map(|d| d.gain_loss).sum();
    let gain_lifo: Decimal = disposals_lifo.iter().map(|d| d.gain_loss).sum();
    let gain_hifo: Decimal = disposals_hifo.iter().map(|d| d.gain_loss).sum();

    // FIFO uses: lot1 (60k), lot2 (50k) = 110k cost basis = -20k loss
    // LIFO uses: lot3 (40k), lot2 (50k) = 90k cost basis = 0 gain/loss
    // HIFO uses: lot1 (60k), lot2 (50k) = 110k cost basis = -20k loss

    assert_eq!(gain_fifo, dec!(-20000));
    assert_eq!(gain_lifo, dec!(0));
    assert_eq!(gain_hifo, dec!(-20000));

    // FIFO and HIFO both get the loss (tax advantage)
    // LIFO breaks even
}

#[test]
fn test_holding_period_differences() {
    // Test that methods can result in different short-term/long-term classifications
    let sale = create_sale(dec!(1.0), dec!(60000));

    let mut lots_fifo = vec![
        create_lot("lot1", dec!(1.0), dec!(50000), 400),  // Long-term
        create_lot("lot2", dec!(1.0), dec!(55000), 200),  // Short-term
    ];

    let mut lots_lifo = lots_fifo.clone();

    let disposals_fifo = calculate_fifo_disposal(&sale, &mut lots_fifo).unwrap();
    let disposals_lifo = calculate_lifo_internal(&sale, &mut lots_lifo).unwrap();

    // FIFO uses lot1 (long-term)
    assert!(disposals_fifo[0].is_long_term);

    // LIFO uses lot2 (short-term)
    assert!(!disposals_lifo[0].is_long_term);
}

#[test]
fn test_lot_selection_order() {
    // Verify that each method selects lots in correct order
    let sale = create_sale(dec!(4.0), dec!(60000));

    let mut lots_fifo = vec![
        create_lot("lot1", dec!(1.0), dec!(40000), 500),  // Oldest
        create_lot("lot2", dec!(1.0), dec!(50000), 400),
        create_lot("lot3", dec!(1.0), dec!(58000), 300),
        create_lot("lot4", dec!(1.0), dec!(55000), 200),  // Newest
    ];

    let mut lots_lifo = lots_fifo.clone();
    let mut lots_hifo = lots_fifo.clone();

    let disposals_fifo = calculate_fifo_disposal(&sale, &mut lots_fifo).unwrap();
    let disposals_lifo = calculate_lifo_internal(&sale, &mut lots_lifo).unwrap();
    let disposals_hifo = calculate_hifo_internal(&sale, &mut lots_hifo).unwrap();

    // FIFO: oldest to newest
    assert_eq!(disposals_fifo[0].lot_id, "lot1");
    assert_eq!(disposals_fifo[1].lot_id, "lot2");
    assert_eq!(disposals_fifo[2].lot_id, "lot3");
    assert_eq!(disposals_fifo[3].lot_id, "lot4");

    // LIFO: newest to oldest
    assert_eq!(disposals_lifo[0].lot_id, "lot4");
    assert_eq!(disposals_lifo[1].lot_id, "lot3");
    assert_eq!(disposals_lifo[2].lot_id, "lot2");
    assert_eq!(disposals_lifo[3].lot_id, "lot1");

    // HIFO: highest unit cost to lowest
    assert_eq!(disposals_hifo[0].lot_id, "lot3"); // $58k
    assert_eq!(disposals_hifo[1].lot_id, "lot4"); // $55k
    assert_eq!(disposals_hifo[2].lot_id, "lot2"); // $50k
    assert_eq!(disposals_hifo[3].lot_id, "lot1"); // $40k
}

#[test]
fn test_performance_comparison() {
    use std::time::Instant;

    // Create 1000 lots
    let mut lots: Vec<TaxLot> = (0..1000)
        .map(|i| {
            let cost = 40000 + (i * 10);
            create_lot(&format!("lot{}", i), dec!(0.01), Decimal::from(cost), 1000 - i)
        })
        .collect();

    let sale = create_sale(dec!(5.0), dec!(60000));

    // Test FIFO performance
    let mut lots_fifo = lots.clone();
    let start_fifo = Instant::now();
    calculate_fifo_disposal(&sale, &mut lots_fifo).unwrap();
    let elapsed_fifo = start_fifo.elapsed();

    // Test LIFO performance
    let mut lots_lifo = lots.clone();
    let start_lifo = Instant::now();
    calculate_lifo_internal(&sale, &mut lots_lifo).unwrap();
    let elapsed_lifo = start_lifo.elapsed();

    // Test HIFO performance
    let mut lots_hifo = lots.clone();
    let start_hifo = Instant::now();
    calculate_hifo_internal(&sale, &mut lots_hifo).unwrap();
    let elapsed_hifo = start_hifo.elapsed();

    // All should be <10ms
    assert!(elapsed_fifo.as_millis() < 10, "FIFO took {}ms", elapsed_fifo.as_millis());
    assert!(elapsed_lifo.as_millis() < 10, "LIFO took {}ms", elapsed_lifo.as_millis());
    assert!(elapsed_hifo.as_millis() < 10, "HIFO took {}ms", elapsed_hifo.as_millis());

    println!("Performance for 1000 lots processing 5.0 BTC:");
    println!("  FIFO: {}ms", elapsed_fifo.as_millis());
    println!("  LIFO: {}ms", elapsed_lifo.as_millis());
    println!("  HIFO: {}ms", elapsed_hifo.as_millis());
}

#[test]
fn test_tax_strategy_scenario() {
    // Real-world scenario: investor wants to minimize current year taxes
    let sale = create_sale(dec!(2.0), dec!(60000));

    let mut lots = vec![
        create_lot("lot1", dec!(1.0), dec!(30000), 500),  // Old, low cost, huge gain
        create_lot("lot2", dec!(1.0), dec!(59500), 100),  // Recent, high cost, tiny gain
        create_lot("lot3", dec!(1.0), dec!(50000), 300),  // Medium
    ];

    let mut lots_fifo = lots.clone();
    let mut lots_lifo = lots.clone();
    let mut lots_hifo = lots.clone();

    let disposals_fifo = calculate_fifo_disposal(&sale, &mut lots_fifo).unwrap();
    let disposals_lifo = calculate_lifo_internal(&sale, &mut lots_lifo).unwrap();
    let disposals_hifo = calculate_hifo_internal(&sale, &mut lots_hifo).unwrap();

    let gain_fifo: Decimal = disposals_fifo.iter().map(|d| d.gain_loss).sum();
    let gain_lifo: Decimal = disposals_lifo.iter().map(|d| d.gain_loss).sum();
    let gain_hifo: Decimal = disposals_hifo.iter().map(|d| d.gain_loss).sum();

    // FIFO: lot1 ($30k) + lot3 ($50k) = $80k cost = $40k gain
    // LIFO: lot2 ($59.5k) + lot3 ($50k) = $109.5k cost = $10.5k gain
    // HIFO: lot2 ($59.5k) + lot3 ($50k) = $109.5k cost = $10.5k gain

    println!("Tax Strategy Comparison:");
    println!("  FIFO gain: ${}", gain_fifo);
    println!("  LIFO gain: ${}", gain_lifo);
    println!("  HIFO gain: ${}", gain_hifo);
    println!("  Tax savings (LIFO vs FIFO): ${}", gain_fifo - gain_lifo);
    println!("  Tax savings (HIFO vs FIFO): ${}", gain_fifo - gain_hifo);

    // LIFO and HIFO provide significant tax savings
    assert!(gain_lifo < gain_fifo);
    assert!(gain_hifo < gain_fifo);
}
