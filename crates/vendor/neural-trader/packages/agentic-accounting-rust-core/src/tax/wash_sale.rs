/*!
 * Wash Sale Detection and Cost Basis Adjustment
 *
 * Implements IRS Publication 550 wash sale rules:
 * - A wash sale occurs when you sell a security at a loss and
 *   purchase substantially identical securities within 30 days
 *   before or after the sale
 * - The loss is disallowed and added to the cost basis of the replacement
 * - The holding period is extended to include the original lot
 *
 * Reference: IRS Publication 550 (2023)
 */

use chrono::{DateTime, Utc, Duration};
use rust_decimal::Decimal;
use std::str::FromStr;
use crate::types::{Transaction, TransactionType, JsTransaction, JsDisposal, JsTaxLot};
use crate::datetime::parse_datetime_internal;

/// Wash sale detection result (internal)
#[derive(Debug, Clone)]
pub struct WashSaleResult {
    pub is_wash_sale: bool,
    pub disallowed_loss: Decimal,
    pub replacement_transaction_id: Option<String>,
    pub replacement_date: Option<DateTime<Utc>>,
    pub wash_window_start: DateTime<Utc>,
    pub wash_window_end: DateTime<Utc>,
}

/// Wash sale detection result for JavaScript/TypeScript
#[napi(object)]
#[derive(Debug, Clone)]
pub struct JsWashSaleResult {
    pub is_wash_sale: bool,
    pub disallowed_loss: String,
    pub replacement_transaction_id: Option<String>,
    pub replacement_date: Option<String>,
    pub wash_window_start: String,
    pub wash_window_end: String,
}

impl WashSaleResult {
    pub fn to_js(&self) -> JsWashSaleResult {
        JsWashSaleResult {
            is_wash_sale: self.is_wash_sale,
            disallowed_loss: self.disallowed_loss.to_string(),
            replacement_transaction_id: self.replacement_transaction_id.clone(),
            replacement_date: self.replacement_date.map(|dt| dt.to_rfc3339()),
            wash_window_start: self.wash_window_start.to_rfc3339(),
            wash_window_end: self.wash_window_end.to_rfc3339(),
        }
    }
}

/// Adjusted result after wash sale adjustment (internal)
#[derive(Debug, Clone)]
pub struct AdjustedResult {
    pub adjusted_disposal: JsDisposal,
    pub adjusted_lot: JsTaxLot,
    pub adjustment_amount: Decimal,
}

/// Adjusted result for JavaScript/TypeScript
#[napi(object)]
#[derive(Debug, Clone)]
pub struct JsAdjustedResult {
    pub adjusted_disposal: JsDisposal,
    pub adjusted_lot: JsTaxLot,
    pub adjustment_amount: String,
}

/// Detect if a disposal triggers a wash sale
///
/// IRS Rules:
/// 1. Only losses are subject to wash sale (gains are exempt)
/// 2. 61-day window: 30 days before + day of + 30 days after
/// 3. Must be substantially identical securities
/// 4. Replacement purchase must occur in the wash window
///
/// # Arguments
/// * `disposal` - The disposal to check for wash sale
/// * `all_transactions` - All transactions for this asset
/// * `wash_period_days` - Days before and after (default 30 per IRS)
///
/// # Returns
/// WashSaleResult with detection details
#[napi]
pub fn detect_wash_sale(
    disposal: JsDisposal,
    all_transactions: Vec<JsTransaction>,
    wash_period_days: Option<u32>,
) -> napi::Result<JsWashSaleResult> {
    let wash_days = wash_period_days.unwrap_or(30) as i64;

    // Parse disposal date
    let disposal_date = parse_datetime_internal(&disposal.disposal_date)
        .map_err(|e| napi::Error::from_reason(format!("Invalid disposal date: {}", e)))?;

    // Parse gain/loss
    let gain_loss = Decimal::from_str(&disposal.gain_loss)
        .map_err(|e| napi::Error::from_reason(format!("Invalid gain/loss: {}", e)))?;

    // Wash sale only applies to losses (IRS rule #1)
    if gain_loss >= Decimal::ZERO {
        return Ok(WashSaleResult {
            is_wash_sale: false,
            disallowed_loss: Decimal::ZERO,
            replacement_transaction_id: None,
            replacement_date: None,
            wash_window_start: disposal_date - Duration::days(wash_days),
            wash_window_end: disposal_date + Duration::days(wash_days),
        }.to_js());
    }

    // Calculate wash sale window (30 days before and after)
    let wash_start = disposal_date - Duration::days(wash_days);
    let wash_end = disposal_date + Duration::days(wash_days);

    // Convert JS transactions to internal format
    let transactions: Vec<Transaction> = all_transactions
        .iter()
        .filter_map(|tx| tx.to_internal().ok())
        .collect();

    // Binary search optimization: transactions should be pre-sorted by timestamp
    // Find replacement purchases in wash sale window using efficient filtering
    let replacement = transactions
        .iter()
        .filter(|tx| {
            // Must be a buy transaction (fastest check first)
            tx.transaction_type == TransactionType::Buy &&
            // Must be in wash window (date range check)
            tx.timestamp >= wash_start && tx.timestamp <= wash_end &&
            // Must not be the disposal date itself
            tx.timestamp != disposal_date &&
            // Must be same asset (substantially identical)
            tx.asset == disposal.asset
        })
        .min_by_key(|tx| tx.timestamp); // Use earliest replacement

    if let Some(replacement_tx) = replacement {
        // Wash sale detected!
        let disallowed_loss = gain_loss.abs();

        Ok(WashSaleResult {
            is_wash_sale: true,
            disallowed_loss,
            replacement_transaction_id: Some(replacement_tx.id.clone()),
            replacement_date: Some(replacement_tx.timestamp),
            wash_window_start: wash_start,
            wash_window_end: wash_end,
        }.to_js())
    } else {
        // No wash sale
        Ok(WashSaleResult {
            is_wash_sale: false,
            disallowed_loss: Decimal::ZERO,
            replacement_transaction_id: None,
            replacement_date: None,
            wash_window_start: wash_start,
            wash_window_end: wash_end,
        }.to_js())
    }
}

/// Apply wash sale adjustment to disposal and replacement lot
///
/// IRS Rules:
/// 1. Disallow the loss on the original disposal (set gain to 0)
/// 2. Add the disallowed loss to the replacement lot's cost basis
/// 3. Extend the holding period to include the original lot's acquisition date
///
/// # Arguments
/// * `disposal` - The disposal with disallowed loss
/// * `replacement_lot` - The tax lot of the replacement purchase
/// * `disallowed_loss` - The loss amount to adjust (as positive number)
///
/// # Returns
/// Adjusted disposal and lot
#[napi]
pub fn apply_wash_sale_adjustment(
    disposal: JsDisposal,
    replacement_lot: JsTaxLot,
    disallowed_loss: String,
) -> napi::Result<JsAdjustedResult> {
    // Parse disallowed loss
    let loss_amount = Decimal::from_str(&disallowed_loss)
        .map_err(|e| napi::Error::from_reason(format!("Invalid disallowed loss: {}", e)))?;

    // Parse current cost basis
    let current_cost_basis = Decimal::from_str(&replacement_lot.cost_basis)
        .map_err(|e| napi::Error::from_reason(format!("Invalid cost basis: {}", e)))?;

    // Create adjusted disposal (loss is disallowed, so gain becomes 0)
    let mut adjusted_disposal = disposal.clone();
    adjusted_disposal.gain_loss = "0".to_string();

    // Create adjusted lot with increased cost basis
    let mut adjusted_lot = replacement_lot.clone();
    let new_cost_basis = current_cost_basis + loss_amount;
    adjusted_lot.cost_basis = new_cost_basis.to_string();

    // Note: Holding period adjustment should be handled separately
    // as it requires the original lot's acquisition date
    // The acquisition_date field should be updated to the earlier date
    // when both dates are known

    Ok(JsAdjustedResult {
        adjusted_disposal,
        adjusted_lot,
        adjustment_amount: loss_amount.to_string(),
    })
}

/// Detect wash sales for multiple disposals (batch processing)
///
/// This function efficiently checks multiple disposals against all transactions
/// to identify wash sales. Useful for year-end tax calculations.
///
/// # Arguments
/// * `disposals` - Array of disposals to check
/// * `transactions` - All transactions across all assets
///
/// # Returns
/// Array of wash sale results, one per disposal
#[napi]
pub fn detect_wash_sales_batch(
    disposals: Vec<JsDisposal>,
    transactions: Vec<JsTransaction>,
) -> napi::Result<Vec<JsWashSaleResult>> {
    let mut results = Vec::with_capacity(disposals.len());

    for disposal in disposals {
        // Filter transactions to only those for this asset
        let asset_transactions: Vec<JsTransaction> = transactions
            .iter()
            .filter(|tx| tx.asset == disposal.asset)
            .cloned()
            .collect();

        // Detect wash sale for this disposal
        let result = detect_wash_sale(disposal, asset_transactions, None)?;
        results.push(result);
    }

    Ok(results)
}

/// Check if a specific purchase is a replacement for a wash sale
///
/// Helper function to determine if a purchase transaction falls within
/// the wash sale window of a disposal.
///
/// # Arguments
/// * `disposal_date` - The date of the disposal
/// * `purchase_date` - The date of the purchase
/// * `wash_period_days` - Days before and after (default 30)
///
/// # Returns
/// True if the purchase is within the wash window
#[napi]
pub fn is_wash_sale_replacement(
    disposal_date: String,
    purchase_date: String,
    wash_period_days: Option<u32>,
) -> napi::Result<bool> {
    let wash_days = wash_period_days.unwrap_or(30) as i64;

    let disposal_dt = parse_datetime_internal(&disposal_date)
        .map_err(|e| napi::Error::from_reason(format!("Invalid disposal date: {}", e)))?;

    let purchase_dt = parse_datetime_internal(&purchase_date)
        .map_err(|e| napi::Error::from_reason(format!("Invalid purchase date: {}", e)))?;

    let wash_start = disposal_dt - Duration::days(wash_days);
    let wash_end = disposal_dt + Duration::days(wash_days);

    Ok(purchase_dt >= wash_start && purchase_dt <= wash_end && purchase_dt != disposal_dt)
}

/// Calculate the adjusted holding period after wash sale
///
/// IRS rules require the holding period to include the time the original
/// lot was held before the wash sale.
///
/// # Arguments
/// * `original_acquisition_date` - When the original lot was acquired
/// * `disposal_date` - When the original lot was disposed
/// * `replacement_acquisition_date` - When the replacement was acquired
/// * `potential_disposal_date` - When considering selling the replacement
///
/// # Returns
/// Total days held (original period + replacement period)
#[napi]
pub fn calculate_wash_sale_holding_period(
    original_acquisition_date: String,
    disposal_date: String,
    replacement_acquisition_date: String,
    potential_disposal_date: String,
) -> napi::Result<i64> {
    let original_acq = parse_datetime_internal(&original_acquisition_date)
        .map_err(|e| napi::Error::from_reason(format!("Invalid original acquisition date: {}", e)))?;

    let disposal_dt = parse_datetime_internal(&disposal_date)
        .map_err(|e| napi::Error::from_reason(format!("Invalid disposal date: {}", e)))?;

    let replacement_acq = parse_datetime_internal(&replacement_acquisition_date)
        .map_err(|e| napi::Error::from_reason(format!("Invalid replacement acquisition date: {}", e)))?;

    let potential_disposal = parse_datetime_internal(&potential_disposal_date)
        .map_err(|e| napi::Error::from_reason(format!("Invalid potential disposal date: {}", e)))?;

    // Original holding period
    let original_period = disposal_dt.signed_duration_since(original_acq).num_days();

    // Replacement holding period
    let replacement_period = potential_disposal.signed_duration_since(replacement_acq).num_days();

    // Total adjusted holding period
    Ok(original_period + replacement_period)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_disposal(
        asset: &str,
        gain_loss: &str,
        disposal_date: &str,
    ) -> JsDisposal {
        JsDisposal {
            id: "disposal1".to_string(),
            sale_transaction_id: "sale1".to_string(),
            lot_id: "lot1".to_string(),
            asset: asset.to_string(),
            quantity: "10".to_string(),
            proceeds: "9000".to_string(),
            cost_basis: "10000".to_string(),
            gain_loss: gain_loss.to_string(),
            acquisition_date: "2024-01-01T00:00:00Z".to_string(),
            disposal_date: disposal_date.to_string(),
            is_long_term: false,
        }
    }

    fn create_test_transaction(
        id: &str,
        tx_type: &str,
        asset: &str,
        timestamp: &str,
    ) -> JsTransaction {
        JsTransaction {
            id: id.to_string(),
            transaction_type: tx_type.to_string(),
            asset: asset.to_string(),
            quantity: "10".to_string(),
            price: "1000".to_string(),
            timestamp: timestamp.to_string(),
            source: "test".to_string(),
            fees: "0".to_string(),
        }
    }

    #[test]
    fn test_wash_sale_not_triggered_for_gains() {
        let disposal = create_test_disposal("BTC", "1000", "2024-02-01T00:00:00Z");
        let transactions = vec![
            create_test_transaction("tx1", "BUY", "BTC", "2024-02-05T00:00:00Z"),
        ];

        let result = detect_wash_sale(disposal, transactions, None).unwrap();
        assert!(!result.is_wash_sale, "Gains should not trigger wash sale");
    }

    #[test]
    fn test_wash_sale_detected_within_30_days_after() {
        let disposal = create_test_disposal("BTC", "-1000", "2024-02-01T00:00:00Z");
        let transactions = vec![
            create_test_transaction("tx1", "BUY", "BTC", "2024-02-15T00:00:00Z"), // 14 days after
        ];

        let result = detect_wash_sale(disposal, transactions, None).unwrap();
        assert!(result.is_wash_sale, "Should detect wash sale within 30 days after");
        assert_eq!(result.disallowed_loss, "1000");
        assert!(result.replacement_transaction_id.is_some());
    }

    #[test]
    fn test_wash_sale_detected_within_30_days_before() {
        let disposal = create_test_disposal("BTC", "-1000", "2024-02-15T00:00:00Z");
        let transactions = vec![
            create_test_transaction("tx1", "BUY", "BTC", "2024-02-01T00:00:00Z"), // 14 days before
        ];

        let result = detect_wash_sale(disposal, transactions, None).unwrap();
        assert!(result.is_wash_sale, "Should detect wash sale within 30 days before");
    }

    #[test]
    fn test_wash_sale_not_detected_outside_window() {
        let disposal = create_test_disposal("BTC", "-1000", "2024-02-01T00:00:00Z");
        let transactions = vec![
            create_test_transaction("tx1", "BUY", "BTC", "2024-04-01T00:00:00Z"), // 59 days after
        ];

        let result = detect_wash_sale(disposal, transactions, None).unwrap();
        assert!(!result.is_wash_sale, "Should not detect wash sale outside 30-day window");
    }

    #[test]
    fn test_wash_sale_uses_earliest_replacement() {
        let disposal = create_test_disposal("BTC", "-1000", "2024-02-15T00:00:00Z");
        let transactions = vec![
            create_test_transaction("tx1", "BUY", "BTC", "2024-02-20T00:00:00Z"),
            create_test_transaction("tx2", "BUY", "BTC", "2024-02-18T00:00:00Z"), // Earlier
            create_test_transaction("tx3", "BUY", "BTC", "2024-02-25T00:00:00Z"),
        ];

        let result = detect_wash_sale(disposal, transactions, None).unwrap();
        assert!(result.is_wash_sale);
        assert_eq!(result.replacement_transaction_id, Some("tx2".to_string()));
    }

    #[test]
    fn test_wash_sale_different_asset_not_detected() {
        let disposal = create_test_disposal("BTC", "-1000", "2024-02-01T00:00:00Z");
        let transactions = vec![
            create_test_transaction("tx1", "BUY", "ETH", "2024-02-05T00:00:00Z"), // Different asset
        ];

        let result = detect_wash_sale(disposal, transactions, None).unwrap();
        assert!(!result.is_wash_sale, "Different asset should not trigger wash sale");
    }

    #[test]
    fn test_apply_wash_sale_adjustment() {
        let disposal = create_test_disposal("BTC", "-1000", "2024-02-01T00:00:00Z");

        let replacement_lot = JsTaxLot {
            id: "lot2".to_string(),
            transaction_id: "tx2".to_string(),
            asset: "BTC".to_string(),
            quantity: "10".to_string(),
            remaining_quantity: "10".to_string(),
            cost_basis: "11000".to_string(),
            acquisition_date: "2024-02-05T00:00:00Z".to_string(),
        };

        let result = apply_wash_sale_adjustment(
            disposal,
            replacement_lot,
            "1000".to_string(),
        ).unwrap();

        assert_eq!(result.adjusted_disposal.gain_loss, "0", "Loss should be disallowed");
        assert_eq!(result.adjusted_lot.cost_basis, "12000", "Cost basis should be increased");
        assert_eq!(result.adjustment_amount, "1000");
    }

    #[test]
    fn test_is_wash_sale_replacement() {
        // Within window
        assert!(is_wash_sale_replacement(
            "2024-02-15T00:00:00Z".to_string(),
            "2024-02-20T00:00:00Z".to_string(),
            None,
        ).unwrap());

        // Outside window
        assert!(!is_wash_sale_replacement(
            "2024-02-15T00:00:00Z".to_string(),
            "2024-04-01T00:00:00Z".to_string(),
            None,
        ).unwrap());

        // Before disposal within window
        assert!(is_wash_sale_replacement(
            "2024-02-15T00:00:00Z".to_string(),
            "2024-02-01T00:00:00Z".to_string(),
            None,
        ).unwrap());
    }

    #[test]
    fn test_calculate_wash_sale_holding_period() {
        let total_days = calculate_wash_sale_holding_period(
            "2024-01-01T00:00:00Z".to_string(), // Original acquired
            "2024-02-01T00:00:00Z".to_string(), // Disposed (31 days held)
            "2024-02-05T00:00:00Z".to_string(), // Replacement acquired
            "2024-03-05T00:00:00Z".to_string(), // Potential disposal (29 days held)
        ).unwrap();

        // Should be 31 + 29 = 60 days
        assert_eq!(total_days, 60);
    }

    #[test]
    fn test_batch_wash_sale_detection() {
        let disposals = vec![
            create_test_disposal("BTC", "-1000", "2024-02-01T00:00:00Z"),
            create_test_disposal("ETH", "500", "2024-02-01T00:00:00Z"),
            create_test_disposal("BTC", "-500", "2024-03-01T00:00:00Z"),
        ];

        let transactions = vec![
            create_test_transaction("tx1", "BUY", "BTC", "2024-02-10T00:00:00Z"),
            create_test_transaction("tx2", "BUY", "ETH", "2024-02-10T00:00:00Z"),
        ];

        let results = detect_wash_sales_batch(disposals, transactions).unwrap();

        assert_eq!(results.len(), 3);
        assert!(results[0].is_wash_sale, "BTC loss should trigger wash sale");
        assert!(!results[1].is_wash_sale, "ETH gain should not trigger wash sale");
        assert!(!results[2].is_wash_sale, "BTC loss in March outside window");
    }
}
