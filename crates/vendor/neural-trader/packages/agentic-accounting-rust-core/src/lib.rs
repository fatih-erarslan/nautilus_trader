/*!
 * High-Performance Rust Core for Agentic Accounting System
 *
 * This module provides NAPI bindings for performance-critical operations:
 * - Precise decimal arithmetic for financial calculations
 * - Tax lot tracking and disposal calculations
 * - FIFO, LIFO, HIFO, and specific identification methods
 * - Wash sale detection
 * - Date/time operations
 */

#![deny(clippy::all)]

#[macro_use]
extern crate napi_derive;

mod error;
mod math;
mod datetime;
pub mod types;
pub mod tax;

pub use error::{RustCoreError, Result};
pub use math::{DecimalMath, add_decimals, subtract_decimals, multiply_decimals, divide_decimals};
pub use datetime::{parse_datetime, format_datetime, days_between, is_within_wash_sale_period};
pub use types::{Transaction, TaxLot, Disposal, TransactionType, JsTransaction, JsTaxLot, JsDisposal};
pub use tax::fifo::{calculate_fifo_disposal, FifoDisposalResult};
pub use tax::wash_sale::{
    detect_wash_sale,
    apply_wash_sale_adjustment,
    detect_wash_sales_batch,
    is_wash_sale_replacement,
    calculate_wash_sale_holding_period,
    JsWashSaleResult,
    JsAdjustedResult,
};

use napi::{Error as NapiError, Status};

/// Convert our custom error type to NAPI error
impl From<RustCoreError> for NapiError {
    fn from(err: RustCoreError) -> Self {
        NapiError::new(Status::GenericFailure, format!("{}", err))
    }
}

/// Get version information
#[napi]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Health check function
#[napi]
pub fn health_check() -> bool {
    true
}

/// NAPI result structure for FIFO disposal calculation
#[napi(object)]
#[derive(Debug, Clone)]
pub struct JsDisposalResult {
    pub disposals: Vec<JsDisposal>,
    pub updated_lots: Vec<JsTaxLot>,
}

/// Calculate FIFO disposals (NAPI export)
///
/// # Arguments
/// * `sale` - The sale transaction
/// * `available_lots` - Available tax lots
///
/// # Returns
/// Result containing disposals and updated lots
#[napi]
pub fn calculate_fifo(
    sale: JsTransaction,
    available_lots: Vec<JsTaxLot>,
) -> napi::Result<JsDisposalResult> {
    // Convert JS types to internal types
    let sale_internal = sale.to_internal()
        .map_err(|e| napi::Error::from_reason(format!("Invalid sale transaction: {}", e)))?;

    let lots_internal: Result<Vec<TaxLot>> = available_lots
        .iter()
        .map(|lot| lot.to_internal())
        .collect();

    let lots_internal = lots_internal
        .map_err(|e| napi::Error::from_reason(format!("Invalid tax lot: {}", e)))?;

    // Calculate FIFO disposal
    let result = calculate_fifo_disposal(&sale_internal, lots_internal)
        .map_err(|e| napi::Error::from_reason(format!("FIFO calculation failed: {}", e)))?;

    // Convert back to JS types
    let js_disposals = result.disposals
        .iter()
        .map(|d| d.to_js())
        .collect();

    let js_lots = result.updated_lots
        .iter()
        .map(|l| l.to_js())
        .collect();

    Ok(JsDisposalResult {
        disposals: js_disposals,
        updated_lots: js_lots,
    })
}

// TODO: Implement Specific Identification method
// Currently disabled due to incompatible function signature
// #[napi]
// pub fn calculate_specific_identification(...) { ... }

// TODO: Implement Average Cost method
// Currently disabled due to incompatible function signature
// #[napi]
// pub fn calculate_average_cost_method(...) { ... }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let version = get_version();
        assert!(!version.is_empty());
    }

    #[test]
    fn test_health_check() {
        assert!(health_check());
    }
}
