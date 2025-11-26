/*!
 * FIFO (First-In, First-Out) Tax Calculation Algorithm
 *
 * Implements the FIFO method for calculating capital gains/losses.
 * Processes tax lots in chronological order (oldest first) to determine
 * cost basis and holding periods for asset disposals.
 *
 * Performance target: <10ms for 1000 lots
 */

use crate::error::{RustCoreError, Result};
use crate::types::{Transaction, TaxLot, Disposal};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::Duration;
use uuid::Uuid;

/// Result of a FIFO disposal calculation
#[derive(Debug, Clone)]
pub struct FifoDisposalResult {
    /// List of disposal records created
    pub disposals: Vec<Disposal>,
    /// Updated tax lots after disposal
    pub updated_lots: Vec<TaxLot>,
}

/// Calculate disposals using the FIFO (First-In, First-Out) method
///
/// # Arguments
/// * `sale_transaction` - The sale transaction to process
/// * `available_lots` - Tax lots available for disposal (will be sorted by acquisition date)
///
/// # Returns
/// * `FifoDisposalResult` containing disposal records and updated lots
///
/// # Errors
/// * `InsufficientQuantity` - Not enough lots to cover the sale quantity
/// * `CalculationError` - Invalid calculation parameters
///
/// # Example
/// ```rust
/// let sale = Transaction { ... };
/// let lots = vec![lot1, lot2, lot3];
/// let result = calculate_fifo_disposal(&sale, lots)?;
/// ```
pub fn calculate_fifo_disposal(
    sale_transaction: &Transaction,
    mut available_lots: Vec<TaxLot>,
) -> Result<FifoDisposalResult> {
    // Validate inputs
    if sale_transaction.quantity <= Decimal::ZERO {
        return Err(RustCoreError::CalculationError(
            "Sale quantity must be greater than zero".to_string()
        ));
    }

    if available_lots.is_empty() {
        return Err(RustCoreError::CalculationError(
            "No available tax lots for disposal".to_string()
        ));
    }

    // Sort lots by acquisition date (oldest first - FIFO)
    // Use unstable sort for better performance (50% faster for large datasets)
    available_lots.sort_unstable_by(|a, b| a.acquisition_date.cmp(&b.acquisition_date));

    // Pre-allocate with estimated capacity to reduce reallocations
    let mut disposals = Vec::with_capacity(available_lots.len().min(10));
    let mut quantity_to_dispose = sale_transaction.quantity;
    let total_proceeds = sale_transaction.price * sale_transaction.quantity;

    // Process each lot until we've disposed of the required quantity
    for lot in available_lots.iter_mut() {
        if quantity_to_dispose <= Decimal::ZERO {
            break;
        }

        // Skip lots with no remaining quantity
        if lot.remaining_quantity <= Decimal::ZERO {
            continue;
        }

        // Determine how much to take from this lot
        let disposal_quantity = quantity_to_dispose.min(lot.remaining_quantity);

        // Calculate unit cost basis for this lot
        let unit_cost_basis = if lot.quantity > Decimal::ZERO {
            lot.cost_basis / lot.quantity
        } else {
            return Err(RustCoreError::CalculationError(
                format!("Invalid lot quantity for lot {}", lot.id)
            ));
        };

        // Calculate cost basis for this disposal
        let disposal_cost_basis = disposal_quantity * unit_cost_basis;

        // Calculate proportional proceeds
        let disposal_proceeds = if sale_transaction.quantity > Decimal::ZERO {
            (disposal_quantity / sale_transaction.quantity) * total_proceeds
        } else {
            return Err(RustCoreError::CalculationError(
                "Sale transaction quantity is zero".to_string()
            ));
        };

        // Calculate gain/loss
        let gain_loss = disposal_proceeds - disposal_cost_basis;

        // Determine holding period (short-term < 1 year, long-term >= 1 year)
        let holding_period = sale_transaction.timestamp
            .signed_duration_since(lot.acquisition_date);
        let is_long_term = holding_period >= Duration::days(365);

        // Create disposal record
        let disposal = Disposal {
            id: Uuid::new_v4().to_string(),
            sale_transaction_id: sale_transaction.id.clone(),
            lot_id: lot.id.clone(),
            asset: sale_transaction.asset.clone(),
            quantity: disposal_quantity,
            proceeds: disposal_proceeds,
            cost_basis: disposal_cost_basis,
            gain_loss,
            acquisition_date: lot.acquisition_date,
            disposal_date: sale_transaction.timestamp,
            is_long_term,
        };

        disposals.push(disposal);

        // Update lot remaining quantity
        lot.remaining_quantity -= disposal_quantity;
        quantity_to_dispose -= disposal_quantity;
    }

    // Check if we have enough lots to cover the entire disposal
    if quantity_to_dispose > dec!(0.00000001) { // Use small epsilon for floating point comparison
        return Err(RustCoreError::CalculationError(
            format!(
                "Insufficient quantity in tax lots. Needed {}, but only {} available",
                sale_transaction.quantity,
                sale_transaction.quantity - quantity_to_dispose
            )
        ));
    }

    Ok(FifoDisposalResult {
        disposals,
        updated_lots: available_lots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use crate::types::TransactionType;

    fn create_test_transaction(
        id: &str,
        asset: &str,
        quantity: &str,
        price: &str,
        timestamp: &str,
    ) -> Transaction {
        Transaction {
            id: id.to_string(),
            transaction_type: TransactionType::Sell,
            asset: asset.to_string(),
            quantity: Decimal::from_str_exact(quantity).unwrap(),
            price: Decimal::from_str_exact(price).unwrap(),
            timestamp: chrono::DateTime::parse_from_rfc3339(timestamp)
                .unwrap()
                .with_timezone(&Utc),
            source: "test".to_string(),
            fees: Decimal::ZERO,
        }
    }

    fn create_test_lot(
        id: &str,
        asset: &str,
        quantity: &str,
        cost_basis: &str,
        acquisition_date: &str,
    ) -> TaxLot {
        let qty = Decimal::from_str_exact(quantity).unwrap();
        TaxLot {
            id: id.to_string(),
            transaction_id: format!("buy-{}", id),
            asset: asset.to_string(),
            quantity: qty,
            remaining_quantity: qty,
            cost_basis: Decimal::from_str_exact(cost_basis).unwrap(),
            acquisition_date: chrono::DateTime::parse_from_rfc3339(acquisition_date)
                .unwrap()
                .with_timezone(&Utc),
        }
    }

    #[test]
    fn test_simple_fifo_disposal() {
        // Single lot, full disposal
        let sale = create_test_transaction(
            "sale1",
            "BTC",
            "1.0",
            "50000",
            "2024-06-01T00:00:00Z"
        );

        let lot = create_test_lot(
            "lot1",
            "BTC",
            "1.0",
            "40000",
            "2023-01-01T00:00:00Z"
        );

        let result = calculate_fifo_disposal(&sale, vec![lot]).unwrap();

        assert_eq!(result.disposals.len(), 1);
        let disposal = &result.disposals[0];

        assert_eq!(disposal.quantity, Decimal::from_str_exact("1.0").unwrap());
        assert_eq!(disposal.proceeds, Decimal::from_str_exact("50000").unwrap());
        assert_eq!(disposal.cost_basis, Decimal::from_str_exact("40000").unwrap());
        assert_eq!(disposal.gain_loss, Decimal::from_str_exact("10000").unwrap());
        assert!(disposal.is_long_term);
    }

    #[test]
    fn test_multi_lot_fifo_disposal() {
        // Multiple lots, should use oldest first
        let sale = create_test_transaction(
            "sale1",
            "BTC",
            "2.5",
            "50000",
            "2024-06-01T00:00:00Z"
        );

        let lots = vec![
            create_test_lot("lot1", "BTC", "1.0", "30000", "2023-01-01T00:00:00Z"),
            create_test_lot("lot2", "BTC", "1.0", "35000", "2023-06-01T00:00:00Z"),
            create_test_lot("lot3", "BTC", "1.0", "40000", "2024-01-01T00:00:00Z"),
        ];

        let result = calculate_fifo_disposal(&sale, lots).unwrap();

        assert_eq!(result.disposals.len(), 3);

        // Check that lots are processed in order (oldest first)
        assert_eq!(result.disposals[0].lot_id, "lot1");
        assert_eq!(result.disposals[1].lot_id, "lot2");
        assert_eq!(result.disposals[2].lot_id, "lot3");

        // Verify quantities
        assert_eq!(result.disposals[0].quantity, Decimal::from_str_exact("1.0").unwrap());
        assert_eq!(result.disposals[1].quantity, Decimal::from_str_exact("1.0").unwrap());
        assert_eq!(result.disposals[2].quantity, Decimal::from_str_exact("0.5").unwrap());

        // Verify remaining quantities
        assert_eq!(result.updated_lots[0].remaining_quantity, Decimal::ZERO);
        assert_eq!(result.updated_lots[1].remaining_quantity, Decimal::ZERO);
        assert_eq!(result.updated_lots[2].remaining_quantity, Decimal::from_str_exact("0.5").unwrap());
    }

    #[test]
    fn test_partial_lot_disposal() {
        // Use part of a lot
        let sale = create_test_transaction(
            "sale1",
            "BTC",
            "0.5",
            "50000",
            "2024-06-01T00:00:00Z"
        );

        let lot = create_test_lot(
            "lot1",
            "BTC",
            "1.0",
            "40000",
            "2023-01-01T00:00:00Z"
        );

        let result = calculate_fifo_disposal(&sale, vec![lot]).unwrap();

        assert_eq!(result.disposals.len(), 1);

        let disposal = &result.disposals[0];
        assert_eq!(disposal.quantity, Decimal::from_str_exact("0.5").unwrap());
        assert_eq!(disposal.cost_basis, Decimal::from_str_exact("20000").unwrap());

        // Lot should have remaining quantity
        assert_eq!(result.updated_lots[0].remaining_quantity, Decimal::from_str_exact("0.5").unwrap());
    }

    #[test]
    fn test_short_term_vs_long_term() {
        // Test holding period determination

        // Short-term (< 1 year)
        let sale_short = create_test_transaction(
            "sale1",
            "BTC",
            "1.0",
            "50000",
            "2024-06-01T00:00:00Z"
        );

        let lot_short = create_test_lot(
            "lot1",
            "BTC",
            "1.0",
            "40000",
            "2024-01-01T00:00:00Z" // 5 months before sale
        );

        let result_short = calculate_fifo_disposal(&sale_short, vec![lot_short]).unwrap();
        assert!(!result_short.disposals[0].is_long_term);

        // Long-term (>= 1 year)
        let sale_long = create_test_transaction(
            "sale2",
            "BTC",
            "1.0",
            "50000",
            "2024-06-01T00:00:00Z"
        );

        let lot_long = create_test_lot(
            "lot2",
            "BTC",
            "1.0",
            "40000",
            "2023-01-01T00:00:00Z" // 17 months before sale
        );

        let result_long = calculate_fifo_disposal(&sale_long, vec![lot_long]).unwrap();
        assert!(result_long.disposals[0].is_long_term);
    }

    #[test]
    fn test_insufficient_lots_error() {
        // Try to sell more than available
        let sale = create_test_transaction(
            "sale1",
            "BTC",
            "5.0",
            "50000",
            "2024-06-01T00:00:00Z"
        );

        let lots = vec![
            create_test_lot("lot1", "BTC", "1.0", "30000", "2023-01-01T00:00:00Z"),
            create_test_lot("lot2", "BTC", "1.0", "35000", "2023-06-01T00:00:00Z"),
        ];

        let result = calculate_fifo_disposal(&sale, lots);
        assert!(result.is_err());

        if let Err(RustCoreError::CalculationError(msg)) = result {
            assert!(msg.contains("Insufficient quantity"));
        } else {
            panic!("Expected InsufficientQuantity error");
        }
    }

    #[test]
    fn test_zero_quantity_sale_error() {
        let sale = create_test_transaction(
            "sale1",
            "BTC",
            "0",
            "50000",
            "2024-06-01T00:00:00Z"
        );

        let lot = create_test_lot(
            "lot1",
            "BTC",
            "1.0",
            "40000",
            "2023-01-01T00:00:00Z"
        );

        let result = calculate_fifo_disposal(&sale, vec![lot]);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_lots_error() {
        let sale = create_test_transaction(
            "sale1",
            "BTC",
            "1.0",
            "50000",
            "2024-06-01T00:00:00Z"
        );

        let result = calculate_fifo_disposal(&sale, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_chronological_ordering() {
        // Verify lots are processed by acquisition date, not input order
        let sale = create_test_transaction(
            "sale1",
            "BTC",
            "3.0",
            "50000",
            "2024-06-01T00:00:00Z"
        );

        // Insert lots in reverse chronological order
        let lots = vec![
            create_test_lot("lot3", "BTC", "1.0", "40000", "2024-01-01T00:00:00Z"),
            create_test_lot("lot1", "BTC", "1.0", "30000", "2023-01-01T00:00:00Z"),
            create_test_lot("lot2", "BTC", "1.0", "35000", "2023-06-01T00:00:00Z"),
        ];

        let result = calculate_fifo_disposal(&sale, lots).unwrap();

        // Should be processed as lot1, lot2, lot3 (oldest first)
        assert_eq!(result.disposals[0].lot_id, "lot1");
        assert_eq!(result.disposals[1].lot_id, "lot2");
        assert_eq!(result.disposals[2].lot_id, "lot3");
    }
}
