/*!
 * Average Cost Tax Calculation
 */

use crate::types::{TaxLot, Disposal};
use crate::error::{RustCoreError, Result};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{DateTime, Utc, Duration};

pub fn calculate_average_cost(
    lots: &[TaxLot],
    sale_quantity: Decimal,
    sale_price: Decimal,
    sale_date: DateTime<Utc>,
    sale_id: &str,
    asset: &str,
) -> Result<Vec<Disposal>> {
    if sale_quantity <= dec!(0) {
        return Err(RustCoreError::CalculationError(
            "Sale quantity must be positive".to_string()
        ));
    }

    // Calculate weighted average cost per unit
    let mut total_quantity = dec!(0);
    let mut total_cost = dec!(0);

    for lot in lots.iter() {
        if lot.asset == asset && lot.remaining_quantity > dec!(0) {
            total_quantity += lot.remaining_quantity;
            total_cost += lot.cost_basis * (lot.remaining_quantity / lot.quantity);
        }
    }

    if total_quantity < sale_quantity {
        return Err(RustCoreError::CalculationError(
            format!("Insufficient lots: needed {}, available {}", 
                sale_quantity, total_quantity)
        ));
    }

    let avg_cost_per_unit = if total_quantity > dec!(0) {
        total_cost / total_quantity
    } else {
        dec!(0)
    };

    let mut disposals = Vec::with_capacity(8);
    let mut remaining_quantity = sale_quantity;

    for lot in lots.iter() {
        if remaining_quantity <= dec!(0) {
            break;
        }

        if lot.remaining_quantity <= dec!(0) || lot.asset != asset {
            continue;
        }

        let disposal_quantity = remaining_quantity.min(lot.remaining_quantity);
        let disposal_cost_basis = avg_cost_per_unit * disposal_quantity;
        let disposal_proceeds = sale_price * disposal_quantity;
        let gain_loss = disposal_proceeds - disposal_cost_basis;

        let holding_period = sale_date.signed_duration_since(lot.acquisition_date);
        let is_long_term = holding_period > Duration::days(365);

        let disposal = Disposal {
            id: format!("{}_{}", sale_id, disposals.len()),
            sale_transaction_id: sale_id.to_string(),
            lot_id: lot.id.clone(),
            asset: asset.to_string(),
            quantity: disposal_quantity,
            proceeds: disposal_proceeds,
            cost_basis: disposal_cost_basis,
            gain_loss,
            acquisition_date: lot.acquisition_date,
            disposal_date: sale_date,
            is_long_term,
        };

        disposals.push(disposal);
        remaining_quantity -= disposal_quantity;
    }

    Ok(disposals)
}
