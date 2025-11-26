/*!
 * HIFO (Highest-In, First-Out) Tax Calculation
 *
 * Sorts lots by unit cost basis (highest first)
 * Minimizes taxable gains by using highest cost lots
 */

use crate::error::{RustCoreError, Result};
use crate::types::{Disposal, JsDisposal, JsTransaction, JsTaxLot, TaxLot, Transaction};
use chrono::Duration;
use rust_decimal::Decimal;
use std::str::FromStr;
use uuid::Uuid;

/// Calculate disposals using HIFO method
pub fn calculate_hifo_internal(sale: &Transaction, lots: &mut [TaxLot]) -> Result<Vec<Disposal>> {
    let mut disposals = Vec::new();
    let mut quantity_to_dispose = sale.quantity;

    let total_available: Decimal = lots.iter().map(|lot| lot.remaining_quantity).sum();
    if total_available < quantity_to_dispose {
        return Err(RustCoreError::CalculationError(format!(
            "Insufficient quantity: need {}, have {}", quantity_to_dispose, total_available
        )));
    }

    // Use unstable sort with pre-computed keys for better performance
    lots.sort_unstable_by(|a, b| {
        let unit_cost_a = if a.quantity > Decimal::ZERO {
            a.cost_basis / a.quantity
        } else { Decimal::ZERO };
        let unit_cost_b = if b.quantity > Decimal::ZERO {
            b.cost_basis / b.quantity
        } else { Decimal::ZERO };
        unit_cost_b.cmp(&unit_cost_a)
    });

    for lot in lots.iter_mut() {
        if quantity_to_dispose <= Decimal::ZERO { break; }
        if lot.remaining_quantity <= Decimal::ZERO { continue; }
        if lot.asset != sale.asset { continue; }

        let disposal_quantity = lot.remaining_quantity.min(quantity_to_dispose);
        let unit_cost_basis = if lot.quantity > Decimal::ZERO {
            lot.cost_basis / lot.quantity
        } else { Decimal::ZERO };
        let disposal_cost_basis = disposal_quantity * unit_cost_basis;
        let total_proceeds = sale.price * sale.quantity - sale.fees;
        let disposal_proceeds = if sale.quantity > Decimal::ZERO {
            (disposal_quantity / sale.quantity) * total_proceeds
        } else { Decimal::ZERO };
        let gain_loss = disposal_proceeds - disposal_cost_basis;
        let holding_period = sale.timestamp - lot.acquisition_date;
        let is_long_term = holding_period > Duration::days(365);

        let disposal = Disposal {
            id: Uuid::new_v4().to_string(),
            sale_transaction_id: sale.id.clone(),
            lot_id: lot.id.clone(),
            asset: sale.asset.clone(),
            quantity: disposal_quantity,
            proceeds: disposal_proceeds,
            cost_basis: disposal_cost_basis,
            gain_loss,
            acquisition_date: lot.acquisition_date,
            disposal_date: sale.timestamp,
            is_long_term,
        };

        disposals.push(disposal);
        lot.remaining_quantity -= disposal_quantity;
        quantity_to_dispose -= disposal_quantity;
    }

    if quantity_to_dispose > Decimal::from_str("0.00000001").unwrap() {
        return Err(RustCoreError::CalculationError(format!(
            "Could not dispose full quantity, {} remaining", quantity_to_dispose
        )));
    }

    Ok(disposals)
}

#[napi]
pub fn calculate_hifo(sale: JsTransaction, lots: Vec<JsTaxLot>) -> napi::Result<Vec<JsDisposal>> {
    let sale_internal = sale.to_internal()
        .map_err(|e| napi::Error::from_reason(format!("Invalid sale transaction: {}", e)))?;
    let mut lots_internal: Vec<TaxLot> = lots.iter().map(|lot| lot.to_internal())
        .collect::<Result<Vec<_>>>()
        .map_err(|e| napi::Error::from_reason(format!("Invalid tax lot: {}", e)))?;
    let disposals = calculate_hifo_internal(&sale_internal, &mut lots_internal)
        .map_err(|e| napi::Error::from_reason(format!("HIFO calculation failed: {}", e)))?;
    Ok(disposals.into_iter().map(|d| d.to_js()).collect())
}
