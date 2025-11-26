/*!
 * Integration tests for Average Cost tax method
 */

use agentic_accounting_rust_core::*;
use chrono::Utc;
use rust_decimal::Decimal;
use std::str::FromStr;

fn create_lot(id: &str, asset: &str, qty: &str, cost: &str, days_old: i64) -> TaxLot {
    let timestamp = Utc::now() - chrono::Duration::days(days_old);
    TaxLot {
        id: id.to_string(),
        transaction_id: format!("tx_{}", id),
        asset: asset.to_string(),
        quantity: Decimal::from_str(qty).unwrap(),
        remaining_quantity: Decimal::from_str(qty).unwrap(),
        cost_basis: Decimal::from_str(cost).unwrap(),
        acquisition_date: timestamp,
    }
}

fn create_sale(asset: &str, qty: &str, price: &str) -> Transaction {
    Transaction {
        id: "sale_1".to_string(),
        transaction_type: TransactionType::Sell,
        asset: asset.to_string(),
        quantity: Decimal::from_str(qty).unwrap(),
        price: Decimal::from_str(price).unwrap(),
        timestamp: Utc::now(),
        source: "test".to_string(),
        fees: Decimal::ZERO,
    }
}

#[test]
fn test_average_cost_single_lot() {
    let lots = vec![create_lot("lot1", "BTC", "2.0", "100000", 400)];
    let sale = create_sale("BTC", "1.0", "60000");

    let result = calculate_average_cost(&sale, &lots);
    assert!(result.is_ok());

    let (disposals, updated_lots) = result.unwrap();

    // Single disposal
    assert_eq!(disposals.len(), 1);
    assert_eq!(disposals[0].quantity, Decimal::from_str("1.0").unwrap());

    // Average cost = 100000 / 2.0 = 50000 per BTC
    // Cost basis for 1.0 BTC = 50000
    assert_eq!(disposals[0].cost_basis, Decimal::from_str("50000").unwrap());

    // Proceeds = 1.0 * 60000 = 60000
    assert_eq!(disposals[0].proceeds, Decimal::from_str("60000").unwrap());

    // Gain = 60000 - 50000 = 10000
    assert_eq!(disposals[0].gain_loss, Decimal::from_str("10000").unwrap());

    // Should be long-term (400 days > 365)
    assert!(disposals[0].is_long_term);

    // Lot ID should be AVERAGE_
    assert!(disposals[0].lot_id.starts_with("AVERAGE_"));

    // Remaining quantity = 2.0 - 1.0 = 1.0
    assert_eq!(updated_lots[0].remaining_quantity, Decimal::from_str("1.0").unwrap());
}

#[test]
fn test_average_cost_multiple_lots_equal_weight() {
    let lots = vec![
        create_lot("lot1", "BTC", "1.0", "40000", 400),
        create_lot("lot2", "BTC", "1.0", "60000", 200),
    ];

    let sale = create_sale("BTC", "1.0", "70000");

    let result = calculate_average_cost(&sale, &lots);
    assert!(result.is_ok());

    let (disposals, updated_lots) = result.unwrap();

    // Single disposal with average cost
    assert_eq!(disposals.len(), 1);

    // Average cost = (40000 + 60000) / 2.0 = 50000 per BTC
    assert_eq!(disposals[0].cost_basis, Decimal::from_str("50000").unwrap());

    // Gain = 70000 - 50000 = 20000
    assert_eq!(disposals[0].gain_loss, Decimal::from_str("20000").unwrap());

    // Both lots reduced by 50%
    for lot in updated_lots.iter() {
        assert_eq!(lot.remaining_quantity, Decimal::from_str("0.5").unwrap());
    }
}

#[test]
fn test_average_cost_multiple_lots_different_weights() {
    let lots = vec![
        create_lot("lot1", "ETH", "10.0", "20000", 400),  // $2000 per ETH
        create_lot("lot2", "ETH", "5.0", "15000", 200),   // $3000 per ETH
    ];

    // Total: 15 ETH, Cost: 35000
    // Average: 35000 / 15 = 2333.333... per ETH

    let sale = create_sale("ETH", "6.0", "4000");

    let result = calculate_average_cost(&sale, &lots);
    assert!(result.is_ok());

    let (disposals, updated_lots) = result.unwrap();

    assert_eq!(disposals.len(), 1);

    // Average cost for 6 ETH = 6 * (35000/15) = 14000
    let expected_cost = Decimal::from_str("35000").unwrap() / Decimal::from_str("15").unwrap()
        * Decimal::from_str("6").unwrap();
    assert_eq!(disposals[0].cost_basis, expected_cost);

    // Verify proportional reduction (6/15 = 40% sold)
    let lot1 = updated_lots.iter().find(|l| l.id == "lot1").unwrap();
    let lot2 = updated_lots.iter().find(|l| l.id == "lot2").unwrap();

    // lot1: 10 * 0.6 = 6
    assert_eq!(lot1.remaining_quantity, Decimal::from_str("6").unwrap());

    // lot2: 5 * 0.6 = 3
    assert_eq!(lot2.remaining_quantity, Decimal::from_str("3").unwrap());
}

#[test]
fn test_average_cost_complete_disposal() {
    let lots = vec![
        create_lot("lot1", "BTC", "0.5", "25000", 400),
        create_lot("lot2", "BTC", "0.5", "30000", 200),
    ];

    // Sell entire position
    let sale = create_sale("BTC", "1.0", "60000");

    let result = calculate_average_cost(&sale, &lots);
    assert!(result.is_ok());

    let (disposals, updated_lots) = result.unwrap();

    // Average cost = (25000 + 30000) / 1.0 = 55000
    assert_eq!(disposals[0].cost_basis, Decimal::from_str("55000").unwrap());

    // All lots should be depleted
    for lot in updated_lots.iter() {
        assert!(lot.remaining_quantity < Decimal::from_str("0.0001").unwrap());
    }
}

#[test]
fn test_average_cost_holding_period_uses_oldest() {
    let lots = vec![
        create_lot("recent", "BTC", "1.0", "50000", 100),  // Short-term
        create_lot("old", "BTC", "1.0", "60000", 500),     // Long-term (oldest)
        create_lot("medium", "BTC", "1.0", "55000", 200),  // Medium
    ];

    let sale = create_sale("BTC", "1.5", "70000");

    let result = calculate_average_cost(&sale, &lots);
    assert!(result.is_ok());

    let (disposals, _) = result.unwrap();

    // Should use oldest lot's date (500 days) for holding period
    assert!(disposals[0].is_long_term);
    assert_eq!(disposals[0].acquisition_date, lots.iter().find(|l| l.id == "old").unwrap().acquisition_date);
}

#[test]
fn test_average_cost_with_partially_disposed_lots() {
    // Simulate lots where some quantity was previously sold
    let mut lot1 = create_lot("lot1", "BTC", "2.0", "100000", 400);
    lot1.remaining_quantity = Decimal::from_str("1.0").unwrap(); // Half disposed

    let mut lot2 = create_lot("lot2", "BTC", "3.0", "180000", 200);
    lot2.remaining_quantity = Decimal::from_str("2.0").unwrap(); // 1/3 disposed

    let lots = vec![lot1, lot2];

    // Total available: 1.0 + 2.0 = 3.0 BTC
    // Unit costs: 100000/2.0=50000, 180000/3.0=60000
    // Total cost of available: 50000*1.0 + 60000*2.0 = 170000
    // Average: 170000 / 3.0 = 56666.666... per BTC

    let sale = create_sale("BTC", "1.5", "70000");

    let result = calculate_average_cost(&sale, &lots);
    assert!(result.is_ok());

    let (disposals, _) = result.unwrap();

    let expected_avg_cost = Decimal::from_str("170000").unwrap() / Decimal::from_str("3").unwrap();
    let expected_cost_basis = expected_avg_cost * Decimal::from_str("1.5").unwrap();

    assert_eq!(disposals[0].cost_basis, expected_cost_basis);
}

#[test]
fn test_average_cost_error_insufficient_quantity() {
    let lots = vec![create_lot("lot1", "BTC", "0.5", "25000", 400)];
    let sale = create_sale("BTC", "2.0", "60000");

    let result = calculate_average_cost(&sale, &lots);
    assert!(result.is_err());
}

#[test]
fn test_average_cost_error_no_lots() {
    let lots: Vec<TaxLot> = vec![];
    let sale = create_sale("BTC", "1.0", "60000");

    let result = calculate_average_cost(&sale, &lots);
    assert!(result.is_err());
}

#[test]
fn test_average_cost_error_wrong_asset() {
    let lots = vec![create_lot("lot1", "ETH", "10.0", "20000", 400)];
    let sale = create_sale("BTC", "1.0", "60000");

    let result = calculate_average_cost(&sale, &lots);
    assert!(result.is_err());
}

#[test]
fn test_average_cost_with_fees() {
    let lots = vec![
        create_lot("lot1", "BTC", "1.0", "40000", 400),
        create_lot("lot2", "BTC", "1.0", "60000", 200),
    ];

    let mut sale = create_sale("BTC", "1.0", "70000");
    sale.fees = Decimal::from_str("1000").unwrap(); // $1000 fee

    let result = calculate_average_cost(&sale, &lots);
    assert!(result.is_ok());

    let (disposals, _) = result.unwrap();

    // Average cost = 50000
    assert_eq!(disposals[0].cost_basis, Decimal::from_str("50000").unwrap());

    // Proceeds = 70000 - 1000 = 69000
    assert_eq!(disposals[0].proceeds, Decimal::from_str("69000").unwrap());

    // Gain = 69000 - 50000 = 19000
    assert_eq!(disposals[0].gain_loss, Decimal::from_str("19000").unwrap());
}

#[test]
fn test_average_cost_loss_scenario() {
    let lots = vec![
        create_lot("lot1", "BTC", "1.0", "60000", 400),
        create_lot("lot2", "BTC", "1.0", "70000", 200),
    ];

    // Average cost = 65000
    // Selling below average cost
    let sale = create_sale("BTC", "1.0", "50000");

    let result = calculate_average_cost(&sale, &lots);
    assert!(result.is_ok());

    let (disposals, _) = result.unwrap();

    // Cost basis = 65000
    assert_eq!(disposals[0].cost_basis, Decimal::from_str("65000").unwrap());

    // Loss = 50000 - 65000 = -15000
    assert_eq!(disposals[0].gain_loss, Decimal::from_str("-15000").unwrap());
    assert!(disposals[0].gain_loss < Decimal::ZERO);
}

#[test]
fn test_average_cost_many_lots() {
    let lots = vec![
        create_lot("lot1", "BTC", "0.1", "5000", 500),
        create_lot("lot2", "BTC", "0.2", "12000", 400),
        create_lot("lot3", "BTC", "0.15", "9000", 300),
        create_lot("lot4", "BTC", "0.25", "15000", 200),
        create_lot("lot5", "BTC", "0.3", "18000", 100),
    ];

    // Total: 1.0 BTC, Cost: 59000
    // Average: 59000 per BTC

    let sale = create_sale("BTC", "0.5", "65000");

    let result = calculate_average_cost(&sale, &lots);
    assert!(result.is_ok());

    let (disposals, updated_lots) = result.unwrap();

    // Average cost for 0.5 BTC = 0.5 * 59000 = 29500
    assert_eq!(disposals[0].cost_basis, Decimal::from_str("29500").unwrap());

    // All lots reduced by 50%
    assert_eq!(updated_lots[0].remaining_quantity, Decimal::from_str("0.05").unwrap());
    assert_eq!(updated_lots[1].remaining_quantity, Decimal::from_str("0.1").unwrap());
    assert_eq!(updated_lots[2].remaining_quantity, Decimal::from_str("0.075").unwrap());
    assert_eq!(updated_lots[3].remaining_quantity, Decimal::from_str("0.125").unwrap());
    assert_eq!(updated_lots[4].remaining_quantity, Decimal::from_str("0.15").unwrap());
}

#[test]
fn test_average_cost_rounding_precision() {
    // Test with numbers that cause rounding issues
    let lots = vec![
        create_lot("lot1", "BTC", "0.33333333", "16666.67", 400),
        create_lot("lot2", "BTC", "0.66666667", "33333.33", 200),
    ];

    let sale = create_sale("BTC", "0.5", "60000");

    let result = calculate_average_cost(&sale, &lots);
    assert!(result.is_ok());

    let (disposals, updated_lots) = result.unwrap();

    // Should handle rounding correctly
    assert!(disposals[0].cost_basis > Decimal::ZERO);
    assert!(disposals[0].proceeds > Decimal::ZERO);

    // All remaining quantities should be valid
    for lot in updated_lots.iter() {
        assert!(lot.remaining_quantity >= Decimal::ZERO);
    }
}

#[test]
fn test_average_cost_crypto_mutual_fund_simulation() {
    // Simulate dollar-cost averaging into crypto (like a mutual fund)
    let lots = vec![
        create_lot("jan", "BTC", "0.020", "1000", 365),   // $50k/BTC
        create_lot("feb", "BTC", "0.025", "1000", 335),   // $40k/BTC
        create_lot("mar", "BTC", "0.016", "1000", 305),   // $62.5k/BTC
        create_lot("apr", "BTC", "0.022", "1000", 275),   // $45k/BTC
        create_lot("may", "BTC", "0.018", "1000", 245),   // $55.5k/BTC
    ];

    // Total: 0.101 BTC, Cost: 5000
    // Average: ~49504.95 per BTC

    let sale = create_sale("BTC", "0.05", "60000");

    let result = calculate_average_cost(&sale, &lots);
    assert!(result.is_ok());

    let (disposals, _) = result.unwrap();

    // Should be long-term (oldest is 365+ days)
    assert!(disposals[0].is_long_term);

    // Should have positive gain
    assert!(disposals[0].gain_loss > Decimal::ZERO);
}
