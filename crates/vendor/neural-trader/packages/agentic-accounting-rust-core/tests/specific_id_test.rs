/*!
 * Integration tests for Specific Identification tax method
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
fn test_specific_id_single_complete_lot() {
    let lots = vec![create_lot("lot1", "BTC", "1.0", "50000", 400)];
    let sale = create_sale("BTC", "1.0", "60000");
    let selected = vec!["lot1".to_string()];

    let result = calculate_specific_id(&sale, &selected, &lots);
    assert!(result.is_ok());

    let (disposals, updated_lots) = result.unwrap();
    assert_eq!(disposals.len(), 1);
    assert_eq!(disposals[0].quantity, Decimal::from_str("1.0").unwrap());
    assert_eq!(disposals[0].cost_basis, Decimal::from_str("50000").unwrap());
    assert_eq!(disposals[0].proceeds, Decimal::from_str("60000").unwrap());
    assert_eq!(disposals[0].gain_loss, Decimal::from_str("10000").unwrap());
    assert!(disposals[0].is_long_term);

    assert_eq!(updated_lots[0].remaining_quantity, Decimal::ZERO);
}

#[test]
fn test_specific_id_partial_lot() {
    let lots = vec![create_lot("lot1", "BTC", "2.0", "100000", 400)];
    let sale = create_sale("BTC", "0.75", "60000");
    let selected = vec!["lot1".to_string()];

    let result = calculate_specific_id(&sale, &selected, &lots);
    assert!(result.is_ok());

    let (disposals, updated_lots) = result.unwrap();
    assert_eq!(disposals.len(), 1);
    assert_eq!(disposals[0].quantity, Decimal::from_str("0.75").unwrap());

    // Unit cost = 100000 / 2.0 = 50000 per BTC
    // Cost basis = 0.75 * 50000 = 37500
    assert_eq!(disposals[0].cost_basis, Decimal::from_str("37500").unwrap());

    // Remaining = 2.0 - 0.75 = 1.25
    assert_eq!(updated_lots[0].remaining_quantity, Decimal::from_str("1.25").unwrap());
}

#[test]
fn test_specific_id_multiple_lots_ordered() {
    let lots = vec![
        create_lot("lot1", "ETH", "5.0", "10000", 400),
        create_lot("lot2", "ETH", "3.0", "9000", 200),
        create_lot("lot3", "ETH", "2.0", "8000", 100),
    ];

    let sale = create_sale("ETH", "7.0", "4000");
    // User chooses lot3, then lot1
    let selected = vec!["lot3".to_string(), "lot1".to_string()];

    let result = calculate_specific_id(&sale, &selected, &lots);
    assert!(result.is_ok());

    let (disposals, updated_lots) = result.unwrap();
    assert_eq!(disposals.len(), 2);

    // First disposal from lot3 (2.0 ETH)
    assert_eq!(disposals[0].lot_id, "lot3");
    assert_eq!(disposals[0].quantity, Decimal::from_str("2.0").unwrap());
    assert!(!disposals[0].is_long_term); // 100 days < 365

    // Second disposal from lot1 (5.0 ETH, but only need 5.0)
    assert_eq!(disposals[1].lot_id, "lot1");
    assert_eq!(disposals[1].quantity, Decimal::from_str("5.0").unwrap());
    assert!(disposals[1].is_long_term); // 400 days > 365

    // Check lot3 is depleted
    let lot3 = updated_lots.iter().find(|l| l.id == "lot3").unwrap();
    assert_eq!(lot3.remaining_quantity, Decimal::ZERO);

    // Check lot1 is depleted
    let lot1 = updated_lots.iter().find(|l| l.id == "lot1").unwrap();
    assert_eq!(lot1.remaining_quantity, Decimal::ZERO);

    // Check lot2 is untouched
    let lot2 = updated_lots.iter().find(|l| l.id == "lot2").unwrap();
    assert_eq!(lot2.remaining_quantity, Decimal::from_str("3.0").unwrap());
}

#[test]
fn test_specific_id_tax_optimization_hifo_like() {
    // User manually selects highest cost lots first (similar to HIFO)
    let lots = vec![
        create_lot("cheap", "BTC", "1.0", "30000", 400),
        create_lot("expensive", "BTC", "1.0", "70000", 200),
        create_lot("medium", "BTC", "1.0", "50000", 100),
    ];

    let sale = create_sale("BTC", "1.5", "60000");
    // Select expensive lot first to maximize cost basis
    let selected = vec!["expensive".to_string(), "medium".to_string()];

    let result = calculate_specific_id(&sale, &selected, &lots);
    assert!(result.is_ok());

    let (disposals, updated_lots) = result.unwrap();
    assert_eq!(disposals.len(), 2);

    // Total cost basis = 70000 + (0.5 * 50000) = 95000
    let total_cost_basis: Decimal = disposals.iter().map(|d| d.cost_basis).sum();
    assert_eq!(total_cost_basis, Decimal::from_str("95000").unwrap());

    // Total gain = 90000 - 95000 = -5000 (loss)
    let total_gain: Decimal = disposals.iter().map(|d| d.gain_loss).sum();
    assert!(total_gain < Decimal::ZERO);
}

#[test]
fn test_specific_id_error_invalid_lot() {
    let lots = vec![create_lot("lot1", "BTC", "1.0", "50000", 400)];
    let sale = create_sale("BTC", "0.5", "60000");
    let selected = vec!["nonexistent".to_string()];

    let result = calculate_specific_id(&sale, &selected, &lots);
    assert!(result.is_err());
}

#[test]
fn test_specific_id_error_insufficient_quantity() {
    let lots = vec![create_lot("lot1", "BTC", "0.5", "25000", 400)];
    let sale = create_sale("BTC", "2.0", "60000");
    let selected = vec!["lot1".to_string()];

    let result = calculate_specific_id(&sale, &selected, &lots);
    assert!(result.is_err());
}

#[test]
fn test_specific_id_error_duplicate_lot_ids() {
    let lots = vec![create_lot("lot1", "BTC", "2.0", "100000", 400)];
    let sale = create_sale("BTC", "1.0", "60000");
    let selected = vec!["lot1".to_string(), "lot1".to_string()];

    let result = calculate_specific_id(&sale, &selected, &lots);
    assert!(result.is_err());
}

#[test]
fn test_specific_id_error_wrong_asset() {
    let lots = vec![create_lot("lot1", "ETH", "10.0", "20000", 400)];
    let sale = create_sale("BTC", "1.0", "60000");
    let selected = vec!["lot1".to_string()];

    let result = calculate_specific_id(&sale, &selected, &lots);
    assert!(result.is_err());
}

#[test]
fn test_specific_id_mixed_term_gains() {
    let lots = vec![
        create_lot("short", "BTC", "1.0", "55000", 100),  // Short-term
        create_lot("long", "BTC", "1.0", "40000", 400),   // Long-term
    ];

    let sale = create_sale("BTC", "2.0", "60000");
    let selected = vec!["short".to_string(), "long".to_string()];

    let result = calculate_specific_id(&sale, &selected, &lots);
    assert!(result.is_ok());

    let (disposals, _) = result.unwrap();
    assert_eq!(disposals.len(), 2);

    // First disposal is short-term
    assert!(!disposals[0].is_long_term);
    assert_eq!(disposals[0].lot_id, "short");

    // Second disposal is long-term
    assert!(disposals[1].is_long_term);
    assert_eq!(disposals[1].lot_id, "long");
}

#[test]
fn test_specific_id_with_fees() {
    let lots = vec![create_lot("lot1", "BTC", "1.0", "50000", 400)];

    let mut sale = create_sale("BTC", "1.0", "60000");
    sale.fees = Decimal::from_str("500").unwrap(); // $500 fee

    let selected = vec!["lot1".to_string()];

    let result = calculate_specific_id(&sale, &selected, &lots);
    assert!(result.is_ok());

    let (disposals, _) = result.unwrap();
    assert_eq!(disposals.len(), 1);

    // Proceeds should account for fees: 60000 - 500 = 59500
    assert_eq!(disposals[0].proceeds, Decimal::from_str("59500").unwrap());

    // Gain = 59500 - 50000 = 9500
    assert_eq!(disposals[0].gain_loss, Decimal::from_str("9500").unwrap());
}

#[test]
fn test_specific_id_precise_decimal_handling() {
    let lots = vec![create_lot("lot1", "BTC", "0.12345678", "6172.84", 400)];
    let sale = create_sale("BTC", "0.12345678", "60000");
    let selected = vec!["lot1".to_string()];

    let result = calculate_specific_id(&sale, &selected, &lots);
    assert!(result.is_ok());

    let (disposals, updated_lots) = result.unwrap();
    assert_eq!(disposals.len(), 1);
    assert_eq!(disposals[0].quantity, Decimal::from_str("0.12345678").unwrap());

    // Should handle decimal precision correctly
    assert!(updated_lots[0].remaining_quantity < Decimal::from_str("0.00000001").unwrap());
}
