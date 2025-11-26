//! Data adapter for converting Nautilus market data to HyperPhysics format.

use crate::config::IntegrationConfig;
use crate::error::{IntegrationError, Result};
use crate::types::{
    MarketSnapshot, NautilusBar, NautilusOrderBookDelta, NautilusQuoteTick,
    NautilusTradeTick,
};
use hyperphysics_hft_ecosystem::core::unified_pipeline::MarketFeed;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, trace};

/// State maintained per instrument
#[derive(Debug, Default)]
struct InstrumentState {
    /// Current market snapshot
    snapshot: MarketSnapshot,
    /// Last quote tick
    last_quote: Option<NautilusQuoteTick>,
    /// Last trade price for return calculation
    last_trade_price: i64,
    /// Accumulated bars for volatility calculation
    recent_bars: Vec<NautilusBar>,
    /// Order book state (price -> size)
    bids: HashMap<i64, u64>,
    asks: HashMap<i64, u64>,
}

/// Adapter for converting Nautilus data to HyperPhysics format.
///
/// This adapter maintains state per instrument and provides conversion
/// methods for all Nautilus data types to HyperPhysics MarketFeed.
pub struct NautilusDataAdapter {
    /// Configuration
    config: IntegrationConfig,
    /// State per instrument (keyed by instrument_id hash)
    state: Arc<RwLock<HashMap<u64, InstrumentState>>>,
    /// Maximum bars to keep for volatility calculation
    max_bars: usize,
}

impl NautilusDataAdapter {
    /// Create a new data adapter
    pub fn new(config: IntegrationConfig) -> Self {
        let max_bars = config.volatility_window.max(config.return_lookback);
        Self {
            config,
            state: Arc::new(RwLock::new(HashMap::new())),
            max_bars,
        }
    }

    /// Process a quote tick and return updated MarketFeed
    pub async fn on_quote(&self, quote: &NautilusQuoteTick) -> Result<MarketFeed> {
        let mut state_map = self.state.write().await;
        let state = state_map.entry(quote.instrument_id).or_default();

        // Update snapshot from quote
        let mut new_snapshot = quote.to_market_snapshot();

        // Carry forward historical data from previous snapshot
        new_snapshot.returns = state.snapshot.returns.clone();
        new_snapshot.volatility = state.snapshot.volatility;
        new_snapshot.volume = state.snapshot.volume;

        // Calculate return if we have a previous quote
        if let Some(prev_quote) = &state.last_quote {
            let prev_mid = (prev_quote.bid_price + prev_quote.ask_price) as f64 / 2.0;
            let curr_mid = (quote.bid_price + quote.ask_price) as f64 / 2.0;
            if prev_mid > 0.0 {
                let ret = (curr_mid - prev_mid) / prev_mid;
                new_snapshot.push_return(ret, self.config.return_lookback);
                new_snapshot.update_volatility();
            }
        }

        state.snapshot = new_snapshot.clone();
        state.last_quote = Some(*quote);

        trace!(
            instrument_id = quote.instrument_id,
            bid = new_snapshot.bid_price,
            ask = new_snapshot.ask_price,
            "Processed quote tick"
        );

        self.snapshot_to_feed(&new_snapshot)
    }

    /// Process a trade tick
    pub async fn on_trade(&self, trade: &NautilusTradeTick) -> Result<MarketFeed> {
        let mut state_map = self.state.write().await;
        let state = state_map.entry(trade.instrument_id).or_default();

        // Calculate return from trade
        if state.last_trade_price > 0 {
            let ret = trade.calculate_return(state.last_trade_price);
            state.snapshot.push_return(ret, self.config.return_lookback);
            state.snapshot.update_volatility();
        }

        state.last_trade_price = trade.price;

        // Update volume
        let trade_size = crate::types::conversions::fixed_u64_to_f64(
            trade.size,
            trade.size_precision,
        );
        state.snapshot.volume += trade_size;
        state.snapshot.timestamp = crate::types::conversions::nanos_to_seconds(trade.ts_event);

        trace!(
            instrument_id = trade.instrument_id,
            price = trade.price,
            size = trade.size,
            "Processed trade tick"
        );

        self.snapshot_to_feed(&state.snapshot)
    }

    /// Process a bar update
    pub async fn on_bar(&self, bar: &NautilusBar) -> Result<MarketFeed> {
        let mut state_map = self.state.write().await;
        let state = state_map.entry(bar.instrument_id).or_default();

        // Add bar to history
        state.recent_bars.push(*bar);
        if state.recent_bars.len() > self.max_bars {
            state.recent_bars.remove(0);
        }

        // Update snapshot from bar
        bar.update_snapshot(&mut state.snapshot);

        debug!(
            instrument_id = bar.instrument_id,
            close = bar.close,
            volume = bar.volume,
            returns_len = state.snapshot.returns.len(),
            "Processed bar"
        );

        self.snapshot_to_feed(&state.snapshot)
    }

    /// Process an order book delta
    pub async fn on_book_delta(&self, delta: &NautilusOrderBookDelta) -> Result<()> {
        let mut state_map = self.state.write().await;
        let state = state_map.entry(delta.instrument_id).or_default();

        let book = if delta.side == 1 { &mut state.bids } else { &mut state.asks };

        match delta.action {
            1 | 2 => { // Add or Update
                book.insert(delta.price, delta.size);
            }
            3 => { // Delete
                book.remove(&delta.price);
            }
            4 => { // Clear
                book.clear();
            }
            _ => {}
        }

        // Recalculate book imbalance
        let total_bid: u64 = state.bids.values().sum();
        let total_ask: u64 = state.asks.values().sum();
        let total = total_bid + total_ask;
        if total > 0 {
            state.snapshot.book_imbalance =
                (total_bid as f64 - total_ask as f64) / total as f64;
        }

        trace!(
            instrument_id = delta.instrument_id,
            action = delta.action_str(),
            imbalance = state.snapshot.book_imbalance,
            "Processed book delta"
        );

        Ok(())
    }

    /// Convert MarketSnapshot to HyperPhysics MarketFeed
    fn snapshot_to_feed(&self, snapshot: &MarketSnapshot) -> Result<MarketFeed> {
        Ok(MarketFeed {
            price: snapshot.mid_price,
            returns: snapshot.returns.clone(),
            volatility: snapshot.volatility,
            vwap: snapshot.vwap,
            spread: snapshot.spread,
            timestamp: snapshot.timestamp,
        })
    }

    /// Get current snapshot for an instrument
    pub async fn get_snapshot(&self, instrument_id: u64) -> Option<MarketSnapshot> {
        let state_map = self.state.read().await;
        state_map.get(&instrument_id).map(|s| s.snapshot.clone())
    }

    /// Get current feed for an instrument
    pub async fn get_feed(&self, instrument_id: u64) -> Result<MarketFeed> {
        let snapshot = self.get_snapshot(instrument_id).await
            .ok_or_else(|| IntegrationError::Validation(
                format!("No data for instrument {}", instrument_id)
            ))?;
        self.snapshot_to_feed(&snapshot)
    }

    /// Clear state for an instrument
    pub async fn clear(&self, instrument_id: u64) {
        let mut state_map = self.state.write().await;
        state_map.remove(&instrument_id);
    }

    /// Clear all state
    pub async fn clear_all(&self) {
        let mut state_map = self.state.write().await;
        state_map.clear();
    }

    /// Get number of tracked instruments
    pub async fn instrument_count(&self) -> usize {
        let state_map = self.state.read().await;
        state_map.len()
    }
}

impl Clone for NautilusDataAdapter {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            state: Arc::clone(&self.state),
            max_bars: self.max_bars,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quote_processing() {
        let config = IntegrationConfig::default();
        let adapter = NautilusDataAdapter::new(config);

        let quote = NautilusQuoteTick {
            instrument_id: 1,
            bid_price: 10000,
            ask_price: 10010,
            bid_size: 100,
            ask_size: 100,
            price_precision: 2,
            size_precision: 0,
            ts_event: 1000000000,
            ts_init: 1000000000,
        };

        let feed = adapter.on_quote(&quote).await.unwrap();
        assert!((feed.price - 100.05).abs() < 0.01);
        assert!((feed.spread - 0.10).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_return_calculation() {
        let config = IntegrationConfig::default();
        let adapter = NautilusDataAdapter::new(config);

        // First quote
        let quote1 = NautilusQuoteTick {
            instrument_id: 1,
            bid_price: 10000,
            ask_price: 10000,
            bid_size: 100,
            ask_size: 100,
            price_precision: 2,
            size_precision: 0,
            ts_event: 1000000000,
            ts_init: 1000000000,
        };
        let _ = adapter.on_quote(&quote1).await.unwrap();

        // Second quote (1% higher)
        let quote2 = NautilusQuoteTick {
            instrument_id: 1,
            bid_price: 10100,
            ask_price: 10100,
            bid_size: 100,
            ask_size: 100,
            price_precision: 2,
            size_precision: 0,
            ts_event: 1000000001,
            ts_init: 1000000001,
        };
        let feed = adapter.on_quote(&quote2).await.unwrap();

        assert!(!feed.returns.is_empty());
        assert!((feed.returns[0] - 0.01).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_book_delta_imbalance() {
        let config = IntegrationConfig::default();
        let adapter = NautilusDataAdapter::new(config);

        // Initialize with a quote first
        let quote = NautilusQuoteTick {
            instrument_id: 1,
            bid_price: 10000,
            ask_price: 10010,
            bid_size: 100,
            ask_size: 100,
            price_precision: 2,
            size_precision: 0,
            ts_event: 1000000000,
            ts_init: 1000000000,
        };
        let _ = adapter.on_quote(&quote).await.unwrap();

        // Add bid side
        let bid_delta = NautilusOrderBookDelta {
            instrument_id: 1,
            action: 1, // Add
            side: 1,   // Bid
            price: 10000,
            size: 200,
            order_id: 1,
            sequence: 1,
            flags: 0,
            price_precision: 2,
            size_precision: 0,
            ts_event: 1000000001,
            ts_init: 1000000001,
        };
        adapter.on_book_delta(&bid_delta).await.unwrap();

        // Add ask side (smaller)
        let ask_delta = NautilusOrderBookDelta {
            instrument_id: 1,
            action: 1,
            side: 2, // Ask
            price: 10010,
            size: 100,
            order_id: 2,
            sequence: 2,
            flags: 0,
            price_precision: 2,
            size_precision: 0,
            ts_event: 1000000002,
            ts_init: 1000000002,
        };
        adapter.on_book_delta(&ask_delta).await.unwrap();

        let snapshot = adapter.get_snapshot(1).await.unwrap();
        // Bid heavy = positive imbalance
        assert!(snapshot.book_imbalance > 0.0);
    }
}
