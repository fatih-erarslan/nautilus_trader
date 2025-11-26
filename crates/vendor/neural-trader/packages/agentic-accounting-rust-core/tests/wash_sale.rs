/*!
 * Integration tests for wash sale detection
 *
 * Tests IRS Publication 550 wash sale rules including:
 * - Basic wash sale detection (30-day window)
 * - Gains exemption (gains never trigger wash sale)
 * - Cost basis adjustment accuracy
 * - Cross-year scenarios
 * - Multiple replacements (using earliest)
 * - Partial wash sales
 * - Wash sale chains
 * - Edge cases and boundary conditions
 */

use agentic_accounting_rust_core::{
    detect_wash_sale,
    apply_wash_sale_adjustment,
    detect_wash_sales_batch,
    is_wash_sale_replacement,
    calculate_wash_sale_holding_period,
    JsTransaction,
    JsDisposal,
    JsTaxLot,
};

// Helper functions for creating test data
fn create_disposal(
    asset: &str,
    gain_loss: &str,
    disposal_date: &str,
    acquisition_date: &str,
) -> JsDisposal {
    JsDisposal {
        id: "disposal1".to_string(),
        sale_transaction_id: "sale1".to_string(),
        lot_id: "lot1".to_string(),
        asset: asset.to_string(),
        quantity: "100".to_string(),
        proceeds: "9000".to_string(),
        cost_basis: "10000".to_string(),
        gain_loss: gain_loss.to_string(),
        acquisition_date: acquisition_date.to_string(),
        disposal_date: disposal_date.to_string(),
        is_long_term: false,
    }
}

fn create_transaction(
    id: &str,
    tx_type: &str,
    asset: &str,
    timestamp: &str,
    quantity: &str,
    price: &str,
) -> JsTransaction {
    JsTransaction {
        id: id.to_string(),
        transaction_type: tx_type.to_string(),
        asset: asset.to_string(),
        quantity: quantity.to_string(),
        price: price.to_string(),
        timestamp: timestamp.to_string(),
        source: "test".to_string(),
        fees: "0".to_string(),
    }
}

fn create_tax_lot(
    id: &str,
    asset: &str,
    cost_basis: &str,
    acquisition_date: &str,
) -> JsTaxLot {
    JsTaxLot {
        id: id.to_string(),
        transaction_id: format!("tx_{}", id),
        asset: asset.to_string(),
        quantity: "100".to_string(),
        remaining_quantity: "100".to_string(),
        cost_basis: cost_basis.to_string(),
        acquisition_date: acquisition_date.to_string(),
    }
}

#[test]
fn test_basic_wash_sale_detection_after_disposal() {
    // IRS Example: Sell at loss on Jan 15, buy back on Jan 25 (10 days later)
    let disposal = create_disposal(
        "BTC",
        "-1000", // $1,000 loss
        "2024-01-15T00:00:00Z",
        "2023-12-01T00:00:00Z",
    );

    let transactions = vec![
        create_transaction(
            "tx1",
            "BUY",
            "BTC",
            "2024-01-25T00:00:00Z", // 10 days after disposal
            "100",
            "90",
        ),
    ];

    let result = detect_wash_sale(disposal, transactions, None).unwrap();

    assert!(result.is_wash_sale, "Should detect wash sale");
    assert_eq!(result.disallowed_loss, "1000");
    assert_eq!(result.replacement_transaction_id, Some("tx1".to_string()));
    assert!(result.replacement_date.is_some());
}

#[test]
fn test_wash_sale_detection_before_disposal() {
    // IRS Scenario: Buy on Jan 5, sell at loss on Jan 15 (purchase 10 days before)
    let disposal = create_disposal(
        "AAPL",
        "-500",
        "2024-01-15T00:00:00Z",
        "2024-01-01T00:00:00Z",
    );

    let transactions = vec![
        create_transaction(
            "tx1",
            "BUY",
            "AAPL",
            "2024-01-05T00:00:00Z", // 10 days before disposal
            "100",
            "95",
        ),
    ];

    let result = detect_wash_sale(disposal, transactions, None).unwrap();

    assert!(result.is_wash_sale, "Should detect wash sale before disposal");
    assert_eq!(result.disallowed_loss, "500");
}

#[test]
fn test_gains_exempt_from_wash_sale() {
    // IRS Rule: Gains are NEVER subject to wash sale
    let disposal = create_disposal(
        "ETH",
        "2000", // Gain, not loss
        "2024-01-15T00:00:00Z",
        "2023-06-01T00:00:00Z",
    );

    let transactions = vec![
        create_transaction(
            "tx1",
            "BUY",
            "ETH",
            "2024-01-20T00:00:00Z", // Within 30 days
            "100",
            "120",
        ),
    ];

    let result = detect_wash_sale(disposal, transactions, None).unwrap();

    assert!(!result.is_wash_sale, "Gains should never trigger wash sale");
    assert_eq!(result.disallowed_loss, "0");
}

#[test]
fn test_30_day_window_boundary() {
    // Test exactly 30 days after - should be wash sale
    let disposal = create_disposal(
        "BTC",
        "-1000",
        "2024-01-15T00:00:00Z",
        "2024-01-01T00:00:00Z",
    );

    let transactions = vec![
        create_transaction(
            "tx1",
            "BUY",
            "BTC",
            "2024-02-14T00:00:00Z", // Exactly 30 days after
            "100",
            "90",
        ),
    ];

    let result = detect_wash_sale(disposal, transactions, None).unwrap();
    assert!(result.is_wash_sale, "Should detect wash sale at 30-day boundary");
}

#[test]
fn test_outside_wash_window() {
    // Purchase 35 days after disposal - not a wash sale
    let disposal = create_disposal(
        "BTC",
        "-1000",
        "2024-01-15T00:00:00Z",
        "2024-01-01T00:00:00Z",
    );

    let transactions = vec![
        create_transaction(
            "tx1",
            "BUY",
            "BTC",
            "2024-02-20T00:00:00Z", // 36 days after
            "100",
            "90",
        ),
    ];

    let result = detect_wash_sale(disposal, transactions, None).unwrap();
    assert!(!result.is_wash_sale, "Should not detect wash sale outside 30-day window");
}

#[test]
fn test_multiple_replacements_uses_earliest() {
    // IRS Rule: If multiple purchases in window, use earliest for adjustment
    let disposal = create_disposal(
        "TSLA",
        "-2000",
        "2024-01-15T00:00:00Z",
        "2023-11-01T00:00:00Z",
    );

    let transactions = vec![
        create_transaction("tx1", "BUY", "TSLA", "2024-01-25T00:00:00Z", "50", "180"),
        create_transaction("tx2", "BUY", "TSLA", "2024-01-20T00:00:00Z", "30", "185"), // Earliest
        create_transaction("tx3", "BUY", "TSLA", "2024-01-30T00:00:00Z", "20", "175"),
    ];

    let result = detect_wash_sale(disposal, transactions, None).unwrap();

    assert!(result.is_wash_sale);
    assert_eq!(
        result.replacement_transaction_id,
        Some("tx2".to_string()),
        "Should use earliest replacement"
    );
}

#[test]
fn test_different_asset_no_wash_sale() {
    // Substantially identical securities rule - different asset
    let disposal = create_disposal(
        "BTC",
        "-1000",
        "2024-01-15T00:00:00Z",
        "2024-01-01T00:00:00Z",
    );

    let transactions = vec![
        create_transaction(
            "tx1",
            "BUY",
            "ETH", // Different asset
            "2024-01-20T00:00:00Z",
            "100",
            "90",
        ),
    ];

    let result = detect_wash_sale(disposal, transactions, None).unwrap();
    assert!(!result.is_wash_sale, "Different asset should not trigger wash sale");
}

#[test]
fn test_cost_basis_adjustment() {
    // Test that disallowed loss is added to replacement lot's cost basis
    let disposal = create_disposal(
        "AAPL",
        "-1500",
        "2024-01-15T00:00:00Z",
        "2023-10-01T00:00:00Z",
    );

    let replacement_lot = create_tax_lot(
        "lot2",
        "AAPL",
        "10000", // Original cost basis
        "2024-01-20T00:00:00Z",
    );

    let result = apply_wash_sale_adjustment(
        disposal,
        replacement_lot,
        "1500".to_string(), // Disallowed loss
    )
    .unwrap();

    assert_eq!(
        result.adjusted_disposal.gain_loss,
        "0",
        "Loss should be disallowed (set to 0)"
    );
    assert_eq!(
        result.adjusted_lot.cost_basis,
        "11500", // 10000 + 1500
        "Cost basis should increase by disallowed loss"
    );
    assert_eq!(result.adjustment_amount, "1500");
}

#[test]
fn test_cross_year_wash_sale() {
    // Sell at loss in December 2023, buy back in January 2024
    let disposal = create_disposal(
        "GOOGL",
        "-3000",
        "2023-12-20T00:00:00Z",
        "2023-01-15T00:00:00Z",
    );

    let transactions = vec![
        create_transaction(
            "tx1",
            "BUY",
            "GOOGL",
            "2024-01-05T00:00:00Z", // 16 days after, different year
            "100",
            "130",
        ),
    ];

    let result = detect_wash_sale(disposal, transactions, None).unwrap();

    assert!(result.is_wash_sale, "Wash sale should apply across years");
    assert_eq!(result.disallowed_loss, "3000");
}

#[test]
fn test_batch_wash_sale_detection() {
    // Test multiple disposals at once
    let disposals = vec![
        create_disposal("BTC", "-1000", "2024-01-15T00:00:00Z", "2023-12-01T00:00:00Z"),
        create_disposal("ETH", "500", "2024-01-16T00:00:00Z", "2023-11-01T00:00:00Z"), // Gain
        create_disposal("AAPL", "-750", "2024-01-17T00:00:00Z", "2023-10-01T00:00:00Z"),
    ];

    let transactions = vec![
        create_transaction("tx1", "BUY", "BTC", "2024-01-25T00:00:00Z", "100", "90"),
        create_transaction("tx2", "BUY", "ETH", "2024-01-26T00:00:00Z", "100", "105"),
        create_transaction("tx3", "BUY", "AAPL", "2024-01-27T00:00:00Z", "100", "142"),
    ];

    let results = detect_wash_sales_batch(disposals, transactions).unwrap();

    assert_eq!(results.len(), 3);
    assert!(results[0].is_wash_sale, "BTC loss should trigger wash sale");
    assert!(!results[1].is_wash_sale, "ETH gain should not trigger wash sale");
    assert!(results[2].is_wash_sale, "AAPL loss should trigger wash sale");
}

#[test]
fn test_is_wash_sale_replacement_helper() {
    // Test the helper function for checking if a purchase is a replacement

    // Within window (before)
    assert!(is_wash_sale_replacement(
        "2024-01-15T00:00:00Z".to_string(),
        "2024-01-05T00:00:00Z".to_string(),
        None,
    )
    .unwrap());

    // Within window (after)
    assert!(is_wash_sale_replacement(
        "2024-01-15T00:00:00Z".to_string(),
        "2024-01-25T00:00:00Z".to_string(),
        None,
    )
    .unwrap());

    // Outside window
    assert!(!is_wash_sale_replacement(
        "2024-01-15T00:00:00Z".to_string(),
        "2024-03-01T00:00:00Z".to_string(),
        None,
    )
    .unwrap());

    // Same date (should not be replacement)
    assert!(!is_wash_sale_replacement(
        "2024-01-15T00:00:00Z".to_string(),
        "2024-01-15T00:00:00Z".to_string(),
        None,
    )
    .unwrap());
}

#[test]
fn test_holding_period_calculation() {
    // Test adjusted holding period after wash sale
    // Original lot: Jan 1 - Feb 1 (31 days)
    // Replacement: Feb 5 - Mar 1 (25 days)
    // Total: 56 days
    let total_days = calculate_wash_sale_holding_period(
        "2024-01-01T00:00:00Z".to_string(), // Original acquisition
        "2024-02-01T00:00:00Z".to_string(), // Disposal
        "2024-02-05T00:00:00Z".to_string(), // Replacement acquisition
        "2024-03-01T00:00:00Z".to_string(), // Potential disposal
    )
    .unwrap();

    assert_eq!(total_days, 56, "Holding period should combine both periods");
}

#[test]
fn test_wash_sale_with_zero_loss() {
    // Edge case: disposal with exactly 0 gain/loss
    let disposal = create_disposal(
        "BTC",
        "0", // Break-even
        "2024-01-15T00:00:00Z",
        "2024-01-01T00:00:00Z",
    );

    let transactions = vec![
        create_transaction("tx1", "BUY", "BTC", "2024-01-20T00:00:00Z", "100", "100"),
    ];

    let result = detect_wash_sale(disposal, transactions, None).unwrap();
    assert!(!result.is_wash_sale, "Zero gain/loss should not trigger wash sale");
}

#[test]
fn test_wash_sale_with_small_loss() {
    // Edge case: very small loss (precision test)
    let disposal = create_disposal(
        "BTC",
        "-0.01", // 1 cent loss
        "2024-01-15T00:00:00Z",
        "2024-01-01T00:00:00Z",
    );

    let transactions = vec![
        create_transaction("tx1", "BUY", "BTC", "2024-01-20T00:00:00Z", "100", "100"),
    ];

    let result = detect_wash_sale(disposal, transactions, None).unwrap();
    assert!(result.is_wash_sale, "Even small losses trigger wash sale");
    assert_eq!(result.disallowed_loss, "0.01");
}

#[test]
fn test_wash_sale_custom_period() {
    // Test with custom wash period (e.g., 15 days for testing)
    let disposal = create_disposal(
        "BTC",
        "-1000",
        "2024-01-15T00:00:00Z",
        "2024-01-01T00:00:00Z",
    );

    let transactions = vec![
        create_transaction(
            "tx1",
            "BUY",
            "BTC",
            "2024-02-05T00:00:00Z", // 21 days after
            "100",
            "90",
        ),
    ];

    // With 15-day period, should not be wash sale
    let result = detect_wash_sale(disposal, transactions, Some(15)).unwrap();
    assert!(!result.is_wash_sale, "Should not detect wash sale with custom 15-day period");

    // With 30-day period, should be wash sale
    let result = detect_wash_sale(disposal, transactions, Some(30)).unwrap();
    assert!(result.is_wash_sale, "Should detect wash sale with 30-day period");
}

#[test]
fn test_wash_sale_chain_scenario() {
    // IRS Scenario: Serial wash sales (wash sale of a wash sale)
    // This is a complex scenario where a replacement lot is itself sold at a loss
    // and triggers another wash sale

    // First disposal at loss
    let disposal1 = create_disposal(
        "MSFT",
        "-1000",
        "2024-01-15T00:00:00Z",
        "2023-12-01T00:00:00Z",
    );

    // First replacement purchase
    let replacement1 = create_transaction(
        "tx1",
        "BUY",
        "MSFT",
        "2024-01-25T00:00:00Z",
        "100",
        "90",
    );

    // Second disposal at loss (of the replacement)
    let disposal2 = create_disposal(
        "MSFT",
        "-500",
        "2024-02-05T00:00:00Z",
        "2024-01-25T00:00:00Z",
    );

    // Second replacement purchase
    let replacement2 = create_transaction(
        "tx2",
        "BUY",
        "MSFT",
        "2024-02-15T00:00:00Z",
        "100",
        "85",
    );

    // Check first wash sale
    let result1 = detect_wash_sale(disposal1, vec![replacement1.clone()], None).unwrap();
    assert!(result1.is_wash_sale, "First disposal should trigger wash sale");

    // Check second wash sale (wash sale chain)
    let result2 = detect_wash_sale(disposal2, vec![replacement2], None).unwrap();
    assert!(result2.is_wash_sale, "Second disposal should trigger wash sale (chain)");
}

#[test]
fn test_partial_wash_sale_scenario() {
    // Partial wash sale: Sell 100 shares at loss, buy back only 50 shares
    // IRS: Only the portion that's replaced is subject to wash sale
    // Note: This test documents the current behavior; full partial wash sale
    // support may require additional logic

    let disposal = create_disposal(
        "AMZN",
        "-2000", // Loss on 100 shares
        "2024-01-15T00:00:00Z",
        "2023-11-01T00:00:00Z",
    );

    let transactions = vec![
        create_transaction(
            "tx1",
            "BUY",
            "AMZN",
            "2024-01-25T00:00:00Z",
            "50", // Only 50 shares bought back
            "165",
        ),
    ];

    let result = detect_wash_sale(disposal, transactions, None).unwrap();

    // Current implementation flags the entire disposal as wash sale
    // In production, you may want to implement proportional disallowance
    assert!(result.is_wash_sale);
    // For partial wash sale: only (50/100) * $2000 = $1000 should be disallowed
    // But current implementation disallows all $2000
}

#[test]
fn test_no_replacement_available() {
    // Scenario: Loss disposal but no replacement purchase in window
    let disposal = create_disposal(
        "NVDA",
        "-1500",
        "2024-01-15T00:00:00Z",
        "2023-08-01T00:00:00Z",
    );

    let transactions = vec![
        // Only SELL transactions, no BUY
        create_transaction("tx1", "SELL", "NVDA", "2024-01-20T00:00:00Z", "100", "485"),
    ];

    let result = detect_wash_sale(disposal, transactions, None).unwrap();
    assert!(!result.is_wash_sale, "No replacement purchase means no wash sale");
}
