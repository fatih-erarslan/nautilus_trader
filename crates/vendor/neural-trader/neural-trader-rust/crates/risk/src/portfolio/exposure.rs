//! Portfolio exposure analysis by asset class, sector, geography

use crate::{Result};
use crate::types::{Portfolio, Symbol};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Exposure breakdown by different dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExposureBreakdown {
    /// Total portfolio value
    pub total_value: f64,
    /// Exposure by symbol
    pub by_symbol: HashMap<String, f64>,
    /// Exposure by asset class
    pub by_asset_class: HashMap<String, f64>,
    /// Long vs short exposure
    pub long_exposure: f64,
    pub short_exposure: f64,
    /// Net exposure (long - short)
    pub net_exposure: f64,
    /// Gross exposure (long + short)
    pub gross_exposure: f64,
}

/// Exposure analyzer
pub struct ExposureAnalyzer;

impl ExposureAnalyzer {
    /// Analyze portfolio exposure
    pub fn analyze(portfolio: &Portfolio) -> Result<ExposureBreakdown> {
        let total_value = portfolio.total_value();

        let mut by_symbol = HashMap::new();
        let mut by_asset_class = HashMap::new();
        let mut long_exposure = 0.0;
        let mut short_exposure = 0.0;

        for (symbol, position) in &portfolio.positions {
            let exposure = position.exposure();
            by_symbol.insert(symbol.as_str().to_string(), exposure);

            // Classify asset class (simplified)
            let asset_class = Self::classify_asset_class(symbol);
            *by_asset_class.entry(asset_class).or_insert(0.0) += exposure;

            // Long/short exposure
            match position.side {
                crate::types::PositionSide::Long => long_exposure += exposure,
                crate::types::PositionSide::Short => short_exposure += exposure.abs(),
            }
        }

        let net_exposure = long_exposure - short_exposure;
        let gross_exposure = long_exposure + short_exposure;

        Ok(ExposureBreakdown {
            total_value,
            by_symbol,
            by_asset_class,
            long_exposure,
            short_exposure,
            net_exposure,
            gross_exposure,
        })
    }

    /// Classify asset into asset class (simplified)
    fn classify_asset_class(symbol: &Symbol) -> String {
        // In production, use proper asset classification
        // This is a simplified heuristic
        let s = symbol.as_str();
        if s.ends_with("USD") || s.ends_with("USDT") {
            "Crypto".to_string()
        } else if s.len() <= 5 {
            "Equity".to_string()
        } else {
            "Other".to_string()
        }
    }

    /// Calculate concentration risk (Herfindahl index)
    pub fn calculate_concentration(exposure: &ExposureBreakdown) -> f64 {
        if exposure.total_value == 0.0 {
            return 0.0;
        }

        exposure
            .by_symbol
            .values()
            .map(|&exp| {
                let weight = exp / exposure.total_value;
                weight * weight
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PositionSide};
    use rust_decimal_macros::dec;

    #[test]
    fn test_exposure_analysis() {
        let mut portfolio = Portfolio::new(dec!(100000));

        portfolio.update_position(Position {
            symbol: Symbol::new("AAPL"),
            quantity: dec!(100),
            avg_entry_price: dec!(150),
            current_price: dec!(150),
            market_value: dec!(15000),
            unrealized_pnl: dec!(0),
            unrealized_pnl_percent: dec!(0),
            side: PositionSide::Long,
            opened_at: chrono::Utc::now(),
        });

        let exposure = ExposureAnalyzer::analyze(&portfolio).unwrap();
        assert!(exposure.total_value > 0.0);
        assert_eq!(exposure.by_symbol.len(), 1);
        assert!(exposure.long_exposure > 0.0);
    }
}
