//! Cross-Exchange Consistency Validation
//!
//! Validates price consistency across multiple exchanges to detect:
//! - Data feed errors
//! - Stale data
//! - Exchange-specific issues
//! - Arbitrage opportunities (which may indicate data problems)
//!
//! # Scientific Foundation
//!
//! Based on:
//! - Law of one price (arbitrage-free pricing)
//! - Market efficiency hypothesis (Fama, 1970)
//! - Cross-exchange price discovery (Hasbrouck, 1995)
//!
//! # References
//!
//! - Fama, E. F. (1970). Efficient Capital Markets
//! - Hasbrouck, J. (1995). One Security, Many Markets

use std::collections::HashMap;

use crate::data::Bar;
use crate::error::{MarketError, MarketResult};

/// Cross-exchange consistency validator
#[derive(Debug)]
pub struct CrossExchangeValidator {
    /// Price tolerance percentage across exchanges
    tolerance_pct: f64,
}

impl CrossExchangeValidator {
    /// Create new cross-exchange validator
    pub fn new(tolerance_pct: f64) -> Self {
        Self { tolerance_pct }
    }

    /// Validate price consistency across exchanges
    pub fn validate(&self, bars: &HashMap<String, Bar>) -> MarketResult<()> {
        if bars.len() < 2 {
            return Ok(());  // Need at least 2 exchanges to compare
        }

        // Extract all close prices
        let prices: Vec<f64> = bars.values().map(|bar| bar.close).collect();

        // Calculate price range
        let min_price = prices.iter().copied().fold(f64::INFINITY, f64::min);
        let max_price = prices.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let avg_price = prices.iter().sum::<f64>() / prices.len() as f64;

        // Check if any price deviates beyond tolerance
        let deviation_pct = ((max_price - min_price) / avg_price).abs();

        if deviation_pct > self.tolerance_pct {
            return Err(MarketError::ValidationError(format!(
                "Cross-exchange price deviation {:.2}% exceeds tolerance {:.2}%",
                deviation_pct * 100.0,
                self.tolerance_pct * 100.0
            )));
        }

        Ok(())
    }

    /// Find price outliers across exchanges
    pub fn find_outliers(&self, bars: &HashMap<String, Bar>) -> Vec<String> {
        if bars.len() < 3 {
            return Vec::new();
        }

        // Calculate median price
        let mut prices: Vec<(String, f64)> = bars.iter()
            .map(|(exchange, bar)| (exchange.clone(), bar.close))
            .collect();
        prices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let median_price = prices[prices.len() / 2].1;

        // Find exchanges with prices far from median
        bars.iter()
            .filter(|(_, bar)| {
                let deviation = ((bar.close - median_price) / median_price).abs();
                deviation > self.tolerance_pct
            })
            .map(|(exchange, _)| exchange.clone())
            .collect()
    }

    /// Calculate cross-exchange spread
    pub fn calculate_spread(&self, bars: &HashMap<String, Bar>) -> f64 {
        if bars.is_empty() {
            return 0.0;
        }

        let prices: Vec<f64> = bars.values().map(|bar| bar.close).collect();
        let min_price = prices.iter().copied().fold(f64::INFINITY, f64::min);
        let max_price = prices.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let avg_price = prices.iter().sum::<f64>() / prices.len() as f64;

        ((max_price - min_price) / avg_price).abs()
    }

    /// Validate timestamp synchronization across exchanges
    pub fn validate_timestamps(&self, bars: &HashMap<String, Bar>) -> MarketResult<()> {
        if bars.len() < 2 {
            return Ok(());
        }

        let timestamps: Vec<_> = bars.values().map(|bar| bar.timestamp).collect();

        // Check if all timestamps are within 1 second of each other
        let min_ts = timestamps.iter().min().unwrap();
        let max_ts = timestamps.iter().max().unwrap();

        let duration = max_ts.signed_duration_since(*min_ts);
        if duration > chrono::Duration::seconds(1) {
            return Err(MarketError::ValidationError(format!(
                "Cross-exchange timestamp drift {} seconds exceeds 1 second",
                duration.num_seconds()
            )));
        }

        Ok(())
    }

    /// Detect potential arbitrage opportunities
    /// (May indicate data feed issues)
    pub fn detect_arbitrage(&self, bars: &HashMap<String, Bar>) -> Option<(String, String, f64)> {
        if bars.len() < 2 {
            return None;
        }

        let exchanges: Vec<_> = bars.keys().cloned().collect();

        for i in 0..exchanges.len() {
            for j in (i+1)..exchanges.len() {
                let ex1 = &exchanges[i];
                let ex2 = &exchanges[j];

                let price1 = bars[ex1].close;
                let price2 = bars[ex2].close;

                let spread_pct = ((price2 - price1) / price1).abs();

                // If spread > tolerance, potential arbitrage (or data issue)
                if spread_pct > self.tolerance_pct {
                    return Some((ex1.clone(), ex2.clone(), spread_pct));
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_bar(symbol: &str, price: f64) -> Bar {
        Bar {
            symbol: symbol.to_string(),
            timestamp: Utc::now(),
            open: price,
            high: price * 1.01,
            low: price * 0.99,
            close: price,
            volume: 1000,
            vwap: Some(price),
            trade_count: None,
        }
    }

    #[test]
    fn test_consistent_prices() {
        let validator = CrossExchangeValidator::new(0.01);  // 1% tolerance

        let mut bars = HashMap::new();
        bars.insert("binance".to_string(), create_test_bar("BTC", 50000.0));
        bars.insert("kraken".to_string(), create_test_bar("BTC", 50100.0));
        bars.insert("coinbase".to_string(), create_test_bar("BTC", 49950.0));

        // Prices within 1% tolerance
        assert!(validator.validate(&bars).is_ok());
    }

    #[test]
    fn test_inconsistent_prices() {
        let validator = CrossExchangeValidator::new(0.01);  // 1% tolerance

        let mut bars = HashMap::new();
        bars.insert("binance".to_string(), create_test_bar("BTC", 50000.0));
        bars.insert("kraken".to_string(), create_test_bar("BTC", 51000.0));  // 2% deviation

        // Prices exceed 1% tolerance
        assert!(validator.validate(&bars).is_err());
    }

    #[test]
    fn test_outlier_detection() {
        let validator = CrossExchangeValidator::new(0.01);

        let mut bars = HashMap::new();
        bars.insert("binance".to_string(), create_test_bar("BTC", 50000.0));
        bars.insert("kraken".to_string(), create_test_bar("BTC", 50100.0));
        bars.insert("bad_exchange".to_string(), create_test_bar("BTC", 55000.0));  // Outlier

        let outliers = validator.find_outliers(&bars);
        assert!(outliers.contains(&"bad_exchange".to_string()));
    }

    #[test]
    fn test_arbitrage_detection() {
        let validator = CrossExchangeValidator::new(0.01);

        let mut bars = HashMap::new();
        bars.insert("cheap".to_string(), create_test_bar("BTC", 50000.0));
        bars.insert("expensive".to_string(), create_test_bar("BTC", 51000.0));

        let arbitrage = validator.detect_arbitrage(&bars);
        assert!(arbitrage.is_some());

        if let Some((ex1, ex2, spread)) = arbitrage {
            assert!(spread > 0.01);
            assert!(ex1 == "cheap" || ex2 == "cheap");
        }
    }
}
