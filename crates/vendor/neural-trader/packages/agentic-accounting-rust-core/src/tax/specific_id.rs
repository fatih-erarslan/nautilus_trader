/*!
 * Specific Identification Tax Method
 *
 * Allows user to manually select which lots to dispose
 * Most flexible but requires detailed record-keeping
 */

use crate::types::{Transaction, TaxLot, Disposal};
use crate::error::{RustCoreError, Result};
use rust_decimal::Decimal;
use uuid::Uuid;
use std::str::FromStr;

/// Calculate disposals using specific identification method
///
/// # Arguments
/// * `sale` - The sale transaction
/// * `selected_lot_ids` - User-selected lot IDs in the order to process
/// * `all_lots` - All available tax lots for the asset
///
/// # Returns
/// * `Ok((disposals, updated_lots))` - Disposal records and updated lots
/// * `Err(RustCoreError)` - If validation fails or insufficient quantity
pub fn calculate_specific_id(
    sale: &Transaction,
    selected_lot_ids: &[String],
    all_lots: &[TaxLot],
) -> Result<(Vec<Disposal>, Vec<TaxLot>)> {
    // Validate sale transaction
    if sale.quantity <= Decimal::ZERO {
        return Err(RustCoreError::CalculationError(
            "Sale quantity must be positive".to_string()
        ));
    }

    // Check for duplicate lot IDs
    let mut unique_ids = std::collections::HashSet::new();
    for id in selected_lot_ids {
        if !unique_ids.insert(id) {
            return Err(RustCoreError::CalculationError(
                format!("Duplicate lot ID specified: {}", id)
            ));
        }
    }

    // Filter to selected lots and validate existence
    let mut selected_lots: Vec<TaxLot> = Vec::new();
    for lot_id in selected_lot_ids {
        let lot = all_lots.iter()
            .find(|l| l.id == *lot_id && l.asset == sale.asset)
            .ok_or_else(|| RustCoreError::InvalidTaxLot(
                format!("Lot ID not found or asset mismatch: {}", lot_id)
            ))?;

        // Only include lots with available quantity
        if lot.remaining_quantity > Decimal::ZERO {
            selected_lots.push(lot.clone());
        } else {
            return Err(RustCoreError::CalculationError(
                format!("Selected lot {} has no available quantity", lot_id)
            ));
        }
    }

    if selected_lots.is_empty() {
        return Err(RustCoreError::CalculationError(
            "No valid lots selected".to_string()
        ));
    }

    // Calculate total available quantity in selected lots
    let total_available: Decimal = selected_lots.iter()
        .map(|lot| lot.remaining_quantity)
        .sum();

    if total_available < sale.quantity {
        return Err(RustCoreError::CalculationError(
            format!(
                "Insufficient quantity in selected lots. Available: {}, Required: {}",
                total_available, sale.quantity
            )
        ));
    }

    // Process disposals in user-specified order
    let mut disposals = Vec::new();
    let mut updated_lots = all_lots.to_vec();
    let mut remaining_quantity = sale.quantity;

    for lot in selected_lots.iter() {
        if remaining_quantity <= Decimal::ZERO {
            break;
        }

        // Find the lot in updated_lots
        let lot_index = updated_lots.iter()
            .position(|l| l.id == lot.id)
            .expect("Lot should exist in updated_lots");

        let current_lot = &updated_lots[lot_index];

        // Determine disposal quantity for this lot
        let disposal_quantity = remaining_quantity.min(current_lot.remaining_quantity);

        // Calculate unit cost basis
        let unit_cost_basis = if current_lot.quantity > Decimal::ZERO {
            current_lot.cost_basis / current_lot.quantity
        } else {
            return Err(RustCoreError::CalculationError(
                format!("Lot {} has zero original quantity", lot.id)
            ));
        };

        // Calculate cost basis for this disposal
        let disposal_cost_basis = disposal_quantity * unit_cost_basis;

        // Calculate proportional proceeds
        let total_proceeds = sale.price * sale.quantity - sale.fees;
        let disposal_proceeds = (disposal_quantity / sale.quantity) * total_proceeds;

        // Calculate gain/loss
        let gain_loss = disposal_proceeds - disposal_cost_basis;

        // Determine if long-term (> 1 year holding period)
        let holding_days = (sale.timestamp - current_lot.acquisition_date).num_days();
        let is_long_term = holding_days > 365;

        // Create disposal record
        let disposal = Disposal {
            id: Uuid::new_v4().to_string(),
            sale_transaction_id: sale.id.clone(),
            lot_id: current_lot.id.clone(),
            asset: sale.asset.clone(),
            quantity: disposal_quantity,
            proceeds: disposal_proceeds,
            cost_basis: disposal_cost_basis,
            gain_loss,
            acquisition_date: current_lot.acquisition_date,
            disposal_date: sale.timestamp,
            is_long_term,
        };

        disposals.push(disposal);

        // Update the lot's remaining quantity
        updated_lots[lot_index].remaining_quantity -= disposal_quantity;
        remaining_quantity -= disposal_quantity;
    }

    // Final validation - should never happen but safety check
    if remaining_quantity > Decimal::from_str("0.00000001").unwrap() {
        return Err(RustCoreError::CalculationError(
            format!("Failed to dispose complete quantity. Remaining: {}", remaining_quantity)
        ));
    }

    Ok((disposals, updated_lots))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use crate::types::TransactionType;

    fn create_test_lot(id: &str, asset: &str, quantity: &str, cost_basis: &str, days_old: i64) -> TaxLot {
        let timestamp = Utc::now() - chrono::Duration::days(days_old);
        TaxLot {
            id: id.to_string(),
            transaction_id: format!("tx_{}", id),
            asset: asset.to_string(),
            quantity: Decimal::from_str(quantity).unwrap(),
            remaining_quantity: Decimal::from_str(quantity).unwrap(),
            cost_basis: Decimal::from_str(cost_basis).unwrap(),
            acquisition_date: timestamp,
        }
    }

    fn create_test_sale(asset: &str, quantity: &str, price: &str) -> Transaction {
        Transaction {
            id: "sale_1".to_string(),
            transaction_type: TransactionType::Sell,
            asset: asset.to_string(),
            quantity: Decimal::from_str(quantity).unwrap(),
            price: Decimal::from_str(price).unwrap(),
            timestamp: Utc::now(),
            source: "test".to_string(),
            fees: Decimal::ZERO,
        }
    }

    #[test]
    fn test_specific_id_single_lot() {
        let lots = vec![
            create_test_lot("lot1", "BTC", "1.0", "50000", 400),
        ];

        let sale = create_test_sale("BTC", "0.5", "60000");
        let selected = vec!["lot1".to_string()];

        let result = calculate_specific_id(&sale, &selected, &lots);
        assert!(result.is_ok());

        let (disposals, updated_lots) = result.unwrap();
        assert_eq!(disposals.len(), 1);
        assert_eq!(disposals[0].quantity, Decimal::from_str("0.5").unwrap());
        assert_eq!(disposals[0].cost_basis, Decimal::from_str("25000").unwrap());
        assert!(disposals[0].is_long_term);

        // Check remaining quantity
        assert_eq!(updated_lots[0].remaining_quantity, Decimal::from_str("0.5").unwrap());
    }

    #[test]
    fn test_specific_id_multiple_lots() {
        let lots = vec![
            create_test_lot("lot1", "BTC", "0.5", "25000", 400),
            create_test_lot("lot2", "BTC", "0.8", "48000", 200),
            create_test_lot("lot3", "BTC", "1.0", "55000", 100),
        ];

        let sale = create_test_sale("BTC", "1.0", "60000");
        // User specifically selects lot2 then lot1
        let selected = vec!["lot2".to_string(), "lot1".to_string()];

        let result = calculate_specific_id(&sale, &selected, &lots);
        assert!(result.is_ok());

        let (disposals, updated_lots) = result.unwrap();
        assert_eq!(disposals.len(), 2);

        // First disposal from lot2 (0.8 BTC)
        assert_eq!(disposals[0].lot_id, "lot2");
        assert_eq!(disposals[0].quantity, Decimal::from_str("0.8").unwrap());

        // Second disposal from lot1 (0.2 BTC to complete the sale)
        assert_eq!(disposals[1].lot_id, "lot1");
        assert_eq!(disposals[1].quantity, Decimal::from_str("0.2").unwrap());

        // Check lot2 is fully depleted
        let lot2 = updated_lots.iter().find(|l| l.id == "lot2").unwrap();
        assert_eq!(lot2.remaining_quantity, Decimal::ZERO);

        // Check lot1 has 0.3 remaining
        let lot1 = updated_lots.iter().find(|l| l.id == "lot1").unwrap();
        assert_eq!(lot1.remaining_quantity, Decimal::from_str("0.3").unwrap());
    }

    #[test]
    fn test_specific_id_invalid_lot() {
        let lots = vec![
            create_test_lot("lot1", "BTC", "1.0", "50000", 400),
        ];

        let sale = create_test_sale("BTC", "0.5", "60000");
        let selected = vec!["lot999".to_string()]; // Non-existent lot

        let result = calculate_specific_id(&sale, &selected, &lots);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), RustCoreError::InvalidTaxLot(_)));
    }

    #[test]
    fn test_specific_id_insufficient_quantity() {
        let lots = vec![
            create_test_lot("lot1", "BTC", "0.3", "15000", 400),
        ];

        let sale = create_test_sale("BTC", "1.0", "60000");
        let selected = vec!["lot1".to_string()];

        let result = calculate_specific_id(&sale, &selected, &lots);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), RustCoreError::CalculationError(_)));
    }

    #[test]
    fn test_specific_id_duplicate_lot_ids() {
        let lots = vec![
            create_test_lot("lot1", "BTC", "1.0", "50000", 400),
        ];

        let sale = create_test_sale("BTC", "0.5", "60000");
        let selected = vec!["lot1".to_string(), "lot1".to_string()]; // Duplicate

        let result = calculate_specific_id(&sale, &selected, &lots);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), RustCoreError::CalculationError(_)));
    }

    #[test]
    fn test_specific_id_zero_quantity_lot() {
        let mut lot = create_test_lot("lot1", "BTC", "1.0", "50000", 400);
        lot.remaining_quantity = Decimal::ZERO; // Lot already fully disposed
        let lots = vec![lot];

        let sale = create_test_sale("BTC", "0.5", "60000");
        let selected = vec!["lot1".to_string()];

        let result = calculate_specific_id(&sale, &selected, &lots);
        assert!(result.is_err());
    }

    #[test]
    fn test_specific_id_holding_period() {
        let lots = vec![
            create_test_lot("lot_short", "BTC", "1.0", "50000", 100), // Short-term
            create_test_lot("lot_long", "BTC", "1.0", "45000", 400),  // Long-term
        ];

        let sale = create_test_sale("BTC", "1.5", "60000");
        let selected = vec!["lot_short".to_string(), "lot_long".to_string()];

        let result = calculate_specific_id(&sale, &selected, &lots);
        assert!(result.is_ok());

        let (disposals, _) = result.unwrap();

        // First disposal from short-term lot
        assert_eq!(disposals[0].lot_id, "lot_short");
        assert!(!disposals[0].is_long_term);

        // Second disposal from long-term lot
        assert_eq!(disposals[1].lot_id, "lot_long");
        assert!(disposals[1].is_long_term);
    }
}
