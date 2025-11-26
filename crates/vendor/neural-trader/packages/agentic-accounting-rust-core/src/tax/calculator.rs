/*!
 * Tax calculator functions
 *
 * This module provides high-level interfaces to all tax calculation algorithms:
 * - FIFO (First-In, First-Out)
 * - LIFO (Last-In, First-Out) - TODO
 * - HIFO (Highest-In, First-Out) - TODO
 * - Specific Identification
 * - Average Cost
 */

use crate::types::{TaxLot, Disposal, Transaction};
use crate::error::{RustCoreError, Result};
use crate::tax::fifo;

/// Calculate disposals using FIFO method
pub fn calculate_fifo(sale_transaction: &Transaction, lots: Vec<TaxLot>) -> Result<Vec<Disposal>> {
    let result = fifo::calculate_fifo_disposal(sale_transaction, lots)?;
    Ok(result.disposals)
}

/// Placeholder for LIFO calculation
/// Will be implemented by tax algorithm agents
pub fn calculate_lifo(_sale: &Transaction, _lots: &[TaxLot]) -> Result<Vec<Disposal>> {
    Err(RustCoreError::CalculationError(
        "LIFO calculation not yet implemented".to_string()
    ))
}

/// Placeholder for HIFO calculation
/// Will be implemented by tax algorithm agents
pub fn calculate_hifo(_sale: &Transaction, _lots: &[TaxLot]) -> Result<Vec<Disposal>> {
    Err(RustCoreError::CalculationError(
        "HIFO calculation not yet implemented".to_string()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TransactionType;
    use chrono::Utc;
    use rust_decimal::Decimal;
    use std::str::FromStr;

    fn create_test_transaction() -> Transaction {
        Transaction {
            id: "test_tx".to_string(),
            transaction_type: TransactionType::Sell,
            asset: "BTC".to_string(),
            quantity: Decimal::from_str("1.0").unwrap(),
            price: Decimal::from_str("50000").unwrap(),
            timestamp: Utc::now(),
            source: "test".to_string(),
            fees: Decimal::ZERO,
        }
    }

    #[test]
    fn test_lifo_not_implemented() {
        let tx = create_test_transaction();
        let result = calculate_lifo(&tx, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_hifo_not_implemented() {
        let tx = create_test_transaction();
        let result = calculate_hifo(&tx, &[]);
        assert!(result.is_err());
    }
}
