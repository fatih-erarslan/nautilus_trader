/*!
 * Core data types for the accounting system
 *
 * These types must match the TypeScript interfaces in @neural-trader/agentic-accounting-types
 */

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use crate::error::{RustCoreError, Result};
use crate::datetime::parse_datetime_internal;

/// Transaction type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionType {
    Buy,
    Sell,
    Trade,
    Income,
    Transfer,
    Fee,
}

impl FromStr for TransactionType {
    type Err = RustCoreError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "BUY" => Ok(TransactionType::Buy),
            "SELL" => Ok(TransactionType::Sell),
            "TRADE" => Ok(TransactionType::Trade),
            "INCOME" => Ok(TransactionType::Income),
            "TRANSFER" => Ok(TransactionType::Transfer),
            "FEE" => Ok(TransactionType::Fee),
            _ => Err(RustCoreError::InvalidTransactionType(s.to_string())),
        }
    }
}

impl ToString for TransactionType {
    fn to_string(&self) -> String {
        match self {
            TransactionType::Buy => "BUY".to_string(),
            TransactionType::Sell => "SELL".to_string(),
            TransactionType::Trade => "TRADE".to_string(),
            TransactionType::Income => "INCOME".to_string(),
            TransactionType::Transfer => "TRANSFER".to_string(),
            TransactionType::Fee => "FEE".to_string(),
        }
    }
}

/// Transaction struct (internal Rust representation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub id: String,
    pub transaction_type: TransactionType,
    pub asset: String,
    pub quantity: Decimal,
    pub price: Decimal,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub fees: Decimal,
}

/// Transaction struct for JavaScript/TypeScript (NAPI export)
#[napi(object)]
#[derive(Debug, Clone)]
pub struct JsTransaction {
    pub id: String,
    pub transaction_type: String,
    pub asset: String,
    pub quantity: String,
    pub price: String,
    pub timestamp: String,
    pub source: String,
    pub fees: String,
}

impl JsTransaction {
    /// Convert to internal Transaction
    pub fn to_internal(&self) -> Result<Transaction> {
        Ok(Transaction {
            id: self.id.clone(),
            transaction_type: TransactionType::from_str(&self.transaction_type)?,
            asset: self.asset.clone(),
            quantity: Decimal::from_str(&self.quantity)
                .map_err(|e| RustCoreError::DecimalError(format!("Invalid quantity: {}", e)))?,
            price: Decimal::from_str(&self.price)
                .map_err(|e| RustCoreError::DecimalError(format!("Invalid price: {}", e)))?,
            timestamp: parse_datetime_internal(&self.timestamp)?,
            source: self.source.clone(),
            fees: Decimal::from_str(&self.fees)
                .map_err(|e| RustCoreError::DecimalError(format!("Invalid fees: {}", e)))?,
        })
    }
}

impl Transaction {
    /// Convert to JavaScript/TypeScript representation
    pub fn to_js(&self) -> JsTransaction {
        JsTransaction {
            id: self.id.clone(),
            transaction_type: self.transaction_type.to_string(),
            asset: self.asset.clone(),
            quantity: self.quantity.to_string(),
            price: self.price.to_string(),
            timestamp: self.timestamp.to_rfc3339(),
            source: self.source.clone(),
            fees: self.fees.to_string(),
        }
    }
}

/// Tax lot struct (internal Rust representation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxLot {
    pub id: String,
    pub transaction_id: String,
    pub asset: String,
    pub quantity: Decimal,
    pub remaining_quantity: Decimal,
    pub cost_basis: Decimal,
    pub acquisition_date: DateTime<Utc>,
}

/// Tax lot struct for JavaScript/TypeScript (NAPI export)
#[napi(object)]
#[derive(Debug, Clone)]
pub struct JsTaxLot {
    pub id: String,
    pub transaction_id: String,
    pub asset: String,
    pub quantity: String,
    pub remaining_quantity: String,
    pub cost_basis: String,
    pub acquisition_date: String,
}

impl JsTaxLot {
    /// Convert to internal TaxLot
    pub fn to_internal(&self) -> Result<TaxLot> {
        Ok(TaxLot {
            id: self.id.clone(),
            transaction_id: self.transaction_id.clone(),
            asset: self.asset.clone(),
            quantity: Decimal::from_str(&self.quantity)
                .map_err(|e| RustCoreError::DecimalError(format!("Invalid quantity: {}", e)))?,
            remaining_quantity: Decimal::from_str(&self.remaining_quantity)
                .map_err(|e| RustCoreError::DecimalError(format!("Invalid remaining quantity: {}", e)))?,
            cost_basis: Decimal::from_str(&self.cost_basis)
                .map_err(|e| RustCoreError::DecimalError(format!("Invalid cost basis: {}", e)))?,
            acquisition_date: parse_datetime_internal(&self.acquisition_date)?,
        })
    }
}

impl TaxLot {
    /// Convert to JavaScript/TypeScript representation
    pub fn to_js(&self) -> JsTaxLot {
        JsTaxLot {
            id: self.id.clone(),
            transaction_id: self.transaction_id.clone(),
            asset: self.asset.clone(),
            quantity: self.quantity.to_string(),
            remaining_quantity: self.remaining_quantity.to_string(),
            cost_basis: self.cost_basis.to_string(),
            acquisition_date: self.acquisition_date.to_rfc3339(),
        }
    }
}

/// Disposal struct (internal Rust representation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Disposal {
    pub id: String,
    pub sale_transaction_id: String,
    pub lot_id: String,
    pub asset: String,
    pub quantity: Decimal,
    pub proceeds: Decimal,
    pub cost_basis: Decimal,
    pub gain_loss: Decimal,
    pub acquisition_date: DateTime<Utc>,
    pub disposal_date: DateTime<Utc>,
    pub is_long_term: bool,
}

/// Disposal struct for JavaScript/TypeScript (NAPI export)
#[napi(object)]
#[derive(Debug, Clone)]
pub struct JsDisposal {
    pub id: String,
    pub sale_transaction_id: String,
    pub lot_id: String,
    pub asset: String,
    pub quantity: String,
    pub proceeds: String,
    pub cost_basis: String,
    pub gain_loss: String,
    pub acquisition_date: String,
    pub disposal_date: String,
    pub is_long_term: bool,
}

impl JsDisposal {
    /// Convert to internal Disposal
    pub fn to_internal(&self) -> Result<Disposal> {
        Ok(Disposal {
            id: self.id.clone(),
            sale_transaction_id: self.sale_transaction_id.clone(),
            lot_id: self.lot_id.clone(),
            asset: self.asset.clone(),
            quantity: Decimal::from_str(&self.quantity)
                .map_err(|e| RustCoreError::DecimalError(format!("Invalid quantity: {}", e)))?,
            proceeds: Decimal::from_str(&self.proceeds)
                .map_err(|e| RustCoreError::DecimalError(format!("Invalid proceeds: {}", e)))?,
            cost_basis: Decimal::from_str(&self.cost_basis)
                .map_err(|e| RustCoreError::DecimalError(format!("Invalid cost basis: {}", e)))?,
            gain_loss: Decimal::from_str(&self.gain_loss)
                .map_err(|e| RustCoreError::DecimalError(format!("Invalid gain/loss: {}", e)))?,
            acquisition_date: parse_datetime_internal(&self.acquisition_date)?,
            disposal_date: parse_datetime_internal(&self.disposal_date)?,
            is_long_term: self.is_long_term,
        })
    }
}

impl Disposal {
    /// Convert to JavaScript/TypeScript representation
    pub fn to_js(&self) -> JsDisposal {
        JsDisposal {
            id: self.id.clone(),
            sale_transaction_id: self.sale_transaction_id.clone(),
            lot_id: self.lot_id.clone(),
            asset: self.asset.clone(),
            quantity: self.quantity.to_string(),
            proceeds: self.proceeds.to_string(),
            cost_basis: self.cost_basis.to_string(),
            gain_loss: self.gain_loss.to_string(),
            acquisition_date: self.acquisition_date.to_rfc3339(),
            disposal_date: self.disposal_date.to_rfc3339(),
            is_long_term: self.is_long_term,
        }
    }
}

/// Create a new transaction (NAPI export for testing)
#[napi]
pub fn create_transaction(tx: JsTransaction) -> napi::Result<JsTransaction> {
    let internal_tx = tx.to_internal()
        .map_err(|e| napi::Error::from_reason(format!("Invalid transaction: {}", e)))?;
    Ok(internal_tx.to_js())
}

/// Create a new tax lot (NAPI export for testing)
#[napi]
pub fn create_tax_lot(lot: JsTaxLot) -> napi::Result<JsTaxLot> {
    let internal_lot = lot.to_internal()
        .map_err(|e| napi::Error::from_reason(format!("Invalid tax lot: {}", e)))?;
    Ok(internal_lot.to_js())
}

/// Create a new disposal (NAPI export for testing)
#[napi]
pub fn create_disposal(disposal: JsDisposal) -> napi::Result<JsDisposal> {
    let internal_disposal = disposal.to_internal()
        .map_err(|e| napi::Error::from_reason(format!("Invalid disposal: {}", e)))?;
    Ok(internal_disposal.to_js())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_type_parsing() {
        assert_eq!(TransactionType::from_str("BUY").unwrap(), TransactionType::Buy);
        assert_eq!(TransactionType::from_str("sell").unwrap(), TransactionType::Sell);
        assert!(TransactionType::from_str("INVALID").is_err());
    }

    #[test]
    fn test_transaction_conversion() {
        let js_tx = JsTransaction {
            id: "tx1".to_string(),
            transaction_type: "BUY".to_string(),
            asset: "BTC".to_string(),
            quantity: "1.5".to_string(),
            price: "50000".to_string(),
            timestamp: "2024-01-15T10:30:00Z".to_string(),
            source: "Coinbase".to_string(),
            fees: "10.50".to_string(),
        };

        let internal = js_tx.to_internal().unwrap();
        assert_eq!(internal.id, "tx1");
        assert_eq!(internal.transaction_type, TransactionType::Buy);
        assert_eq!(internal.asset, "BTC");

        let back_to_js = internal.to_js();
        assert_eq!(back_to_js.id, "tx1");
        assert_eq!(back_to_js.transaction_type, "BUY");
    }
}
