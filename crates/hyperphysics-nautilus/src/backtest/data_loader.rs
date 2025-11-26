//! Data loader for backtest market data.

use crate::backtest::MarketDataEvent;
use crate::error::{IntegrationError, Result};
use crate::types::{NautilusBar, NautilusQuoteTick, NautilusTradeTick};
use std::path::Path;
use tracing::info;

/// Data loader for loading market data from various sources.
pub struct DataLoader;

impl DataLoader {
    /// Load quote ticks from CSV file
    ///
    /// Expected CSV format:
    /// instrument_id,bid_price,ask_price,bid_size,ask_size,price_precision,size_precision,ts_event
    pub fn load_quotes_csv(path: &Path) -> Result<Vec<MarketDataEvent>> {
        let mut events = Vec::new();
        let mut rdr = csv::Reader::from_path(path)
            .map_err(|e| IntegrationError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            )))?;

        for result in rdr.records() {
            let record = result.map_err(|e| IntegrationError::Serialization(e.to_string()))?;

            let quote = NautilusQuoteTick {
                instrument_id: record[0].parse().unwrap_or(0),
                bid_price: record[1].parse().unwrap_or(0),
                ask_price: record[2].parse().unwrap_or(0),
                bid_size: record[3].parse().unwrap_or(0),
                ask_size: record[4].parse().unwrap_or(0),
                price_precision: record[5].parse().unwrap_or(2),
                size_precision: record[6].parse().unwrap_or(0),
                ts_event: record[7].parse().unwrap_or(0),
                ts_init: record[7].parse().unwrap_or(0),
            };

            events.push(MarketDataEvent::Quote(quote));
        }

        info!(count = events.len(), path = %path.display(), "Loaded quotes from CSV");
        Ok(events)
    }

    /// Load bars from CSV file
    ///
    /// Expected CSV format:
    /// instrument_id,open,high,low,close,volume,price_precision,size_precision,ts_event
    pub fn load_bars_csv(path: &Path) -> Result<Vec<MarketDataEvent>> {
        let mut events = Vec::new();
        let mut rdr = csv::Reader::from_path(path)
            .map_err(|e| IntegrationError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            )))?;

        for result in rdr.records() {
            let record = result.map_err(|e| IntegrationError::Serialization(e.to_string()))?;

            let bar = NautilusBar {
                instrument_id: record[0].parse().unwrap_or(0),
                open: record[1].parse().unwrap_or(0),
                high: record[2].parse().unwrap_or(0),
                low: record[3].parse().unwrap_or(0),
                close: record[4].parse().unwrap_or(0),
                volume: record[5].parse().unwrap_or(0),
                price_precision: record[6].parse().unwrap_or(2),
                size_precision: record[7].parse().unwrap_or(0),
                ts_event: record[8].parse().unwrap_or(0),
                ts_init: record[8].parse().unwrap_or(0),
            };

            events.push(MarketDataEvent::Bar(bar));
        }

        info!(count = events.len(), path = %path.display(), "Loaded bars from CSV");
        Ok(events)
    }

    /// Generate synthetic test data
    ///
    /// Creates a series of quote ticks with random walk prices.
    pub fn generate_synthetic_quotes(
        instrument_id: u64,
        num_ticks: usize,
        start_price: f64,
        volatility: f64,
        start_time_ns: u64,
        interval_ns: u64,
    ) -> Vec<MarketDataEvent> {
        use std::f64::consts::PI;

        let mut events = Vec::with_capacity(num_ticks);
        let mut price = start_price;
        let precision = 2u8;
        let scale = 10_f64.powi(precision as i32);

        for i in 0..num_ticks {
            // Simple random walk with trend
            let noise = ((i as f64 * 0.1).sin() + (i as f64 * 0.03).cos()) * volatility;
            price = price * (1.0 + noise);
            price = price.max(start_price * 0.5).min(start_price * 2.0);

            let spread = price * 0.0001; // 1 bps spread
            let bid = (price * scale) as i64;
            let ask = ((price + spread) * scale) as i64;

            let quote = NautilusQuoteTick {
                instrument_id,
                bid_price: bid,
                ask_price: ask,
                bid_size: 100 + (i % 50) as u64,
                ask_size: 100 + ((i + 25) % 50) as u64,
                price_precision: precision,
                size_precision: 0,
                ts_event: start_time_ns + (i as u64) * interval_ns,
                ts_init: start_time_ns + (i as u64) * interval_ns,
            };

            events.push(MarketDataEvent::Quote(quote));
        }

        info!(
            count = events.len(),
            instrument_id = instrument_id,
            start_price = start_price,
            "Generated synthetic quotes"
        );

        events
    }

    /// Sort events by timestamp
    pub fn sort_by_time(events: &mut [MarketDataEvent]) {
        events.sort_by_key(|e| e.timestamp());
    }

    /// Filter events by time range
    pub fn filter_time_range(
        events: Vec<MarketDataEvent>,
        start_ns: u64,
        end_ns: u64,
    ) -> Vec<MarketDataEvent> {
        events
            .into_iter()
            .filter(|e| {
                let ts = e.timestamp();
                ts >= start_ns && ts <= end_ns
            })
            .collect()
    }

    /// Filter events by instrument
    pub fn filter_instrument(
        events: Vec<MarketDataEvent>,
        instrument_id: u64,
    ) -> Vec<MarketDataEvent> {
        events
            .into_iter()
            .filter(|e| e.instrument_id() == instrument_id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data_generation() {
        let events = DataLoader::generate_synthetic_quotes(
            1,
            1000,
            100.0,
            0.001,
            1000000000,
            1000000,
        );

        assert_eq!(events.len(), 1000);

        // Check ordering
        for window in events.windows(2) {
            assert!(window[0].timestamp() < window[1].timestamp());
        }

        // Check first event
        if let MarketDataEvent::Quote(q) = &events[0] {
            assert_eq!(q.instrument_id, 1);
            assert!(q.bid_price > 0);
            assert!(q.ask_price > q.bid_price);
        } else {
            panic!("Expected quote event");
        }
    }

    #[test]
    fn test_filter_time_range() {
        let events = DataLoader::generate_synthetic_quotes(
            1,
            100,
            100.0,
            0.001,
            1000000000,
            1000000,
        );

        let start_ns = 1000000000 + 20 * 1000000;
        let end_ns = 1000000000 + 50 * 1000000;

        let filtered = DataLoader::filter_time_range(events, start_ns, end_ns);

        assert!(filtered.len() < 100);
        for e in &filtered {
            let ts = e.timestamp();
            assert!(ts >= start_ns && ts <= end_ns);
        }
    }
}
