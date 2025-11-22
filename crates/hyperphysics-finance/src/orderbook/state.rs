/// Order book state and analytics
///
/// References:
/// - Hasbrouck, J. (2007). "Empirical Market Microstructure"
///   Oxford University Press, ISBN: 978-0195301649
/// - Gould, M. D., et al. (2013). "Limit order books"
///   Quantitative Finance, 13(11), 1709-1742.

use crate::types::{L2Snapshot, FinanceError};

/// Order book analytics state
#[derive(Debug, Clone)]
pub struct OrderBookState {
    pub snapshot: L2Snapshot,
    pub analytics: OrderBookAnalytics,
}

/// Computed analytics from order book state
#[derive(Debug, Clone)]
pub struct OrderBookAnalytics {
    /// Mid-price: (best_bid + best_ask) / 2
    pub mid_price: f64,

    /// Spread: best_ask - best_bid
    pub spread: f64,

    /// Relative spread: spread / mid_price
    pub relative_spread: f64,

    /// Volume-weighted mid-price (VWMP) using top 5 levels
    pub vwmp: f64,

    /// Order imbalance: (bid_volume - ask_volume) / (bid_volume + ask_volume)
    pub order_imbalance: f64,

    /// Total bid volume (top 5 levels)
    pub total_bid_volume: f64,

    /// Total ask volume (top 5 levels)
    pub total_ask_volume: f64,

    /// Depth at best bid
    pub bid_depth: f64,

    /// Depth at best ask
    pub ask_depth: f64,
}

impl OrderBookState {
    /// Create order book state from snapshot and calculate analytics
    pub fn from_snapshot(snapshot: L2Snapshot) -> Result<Self, FinanceError> {
        snapshot.validate()?;

        let analytics = calculate_analytics(&snapshot)?;

        Ok(Self {
            snapshot,
            analytics,
        })
    }

    /// Update with new snapshot
    pub fn update(&mut self, snapshot: L2Snapshot) -> Result<(), FinanceError> {
        snapshot.validate()?;
        self.analytics = calculate_analytics(&snapshot)?;
        self.snapshot = snapshot;
        Ok(())
    }

    /// Get current mid-price
    pub fn mid_price(&self) -> f64 {
        self.analytics.mid_price
    }

    /// Get current spread
    pub fn spread(&self) -> f64 {
        self.analytics.spread
    }

    /// Get order imbalance (positive = more buy pressure)
    pub fn order_imbalance(&self) -> f64 {
        self.analytics.order_imbalance
    }
}

/// Calculate order book analytics
fn calculate_analytics(snapshot: &L2Snapshot) -> Result<OrderBookAnalytics, FinanceError> {
    // Get best bid and ask
    let best_bid = snapshot.best_bid()
        .ok_or(FinanceError::EmptyOrderBook)?
        .value();
    let best_ask = snapshot.best_ask()
        .ok_or(FinanceError::EmptyOrderBook)?
        .value();

    // Mid-price and spread
    let mid_price = (best_bid + best_ask) / 2.0;
    let spread = best_ask - best_bid;
    let relative_spread = spread / mid_price;

    // Calculate volume-weighted mid-price (top 5 levels)
    let (vwmp, total_bid_volume, total_ask_volume) = calculate_vwmp(snapshot);

    // Order imbalance
    let order_imbalance = if total_bid_volume + total_ask_volume > 0.0 {
        (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
    } else {
        0.0
    };

    // Depth at best levels
    let bid_depth = snapshot.bids.first()
        .map(|(_, q)| q.value())
        .unwrap_or(0.0);
    let ask_depth = snapshot.asks.first()
        .map(|(_, q)| q.value())
        .unwrap_or(0.0);

    Ok(OrderBookAnalytics {
        mid_price,
        spread,
        relative_spread,
        vwmp,
        order_imbalance,
        total_bid_volume,
        total_ask_volume,
        bid_depth,
        ask_depth,
    })
}

/// Calculate volume-weighted mid-price using top N levels
///
/// VWMP = Σ(price_i × volume_i) / Σ(volume_i)
fn calculate_vwmp(snapshot: &L2Snapshot) -> (f64, f64, f64) {
    const MAX_LEVELS: usize = 5;

    let mut bid_value = 0.0;
    let mut bid_volume = 0.0;

    for (price, qty) in snapshot.bids.iter().take(MAX_LEVELS) {
        let vol = qty.value();
        bid_value += price.value() * vol;
        bid_volume += vol;
    }

    let mut ask_value = 0.0;
    let mut ask_volume = 0.0;

    for (price, qty) in snapshot.asks.iter().take(MAX_LEVELS) {
        let vol = qty.value();
        ask_value += price.value() * vol;
        ask_volume += vol;
    }

    let total_value = bid_value + ask_value;
    let total_volume = bid_volume + ask_volume;

    let vwmp = if total_volume > 0.0 {
        total_value / total_volume
    } else {
        // Fallback to simple mid-price
        (snapshot.best_bid().unwrap().value() + snapshot.best_ask().unwrap().value()) / 2.0
    };

    (vwmp, bid_volume, ask_volume)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Price, Quantity};

    fn create_test_snapshot() -> L2Snapshot {
        L2Snapshot {
            symbol: "BTC-USD".to_string(),
            timestamp_us: 1000000,
            bids: vec![
                (Price::new(100.0).unwrap(), Quantity::new(5.0).unwrap()),
                (Price::new(99.5).unwrap(), Quantity::new(3.0).unwrap()),
                (Price::new(99.0).unwrap(), Quantity::new(2.0).unwrap()),
            ],
            asks: vec![
                (Price::new(101.0).unwrap(), Quantity::new(4.0).unwrap()),
                (Price::new(101.5).unwrap(), Quantity::new(2.0).unwrap()),
                (Price::new(102.0).unwrap(), Quantity::new(1.0).unwrap()),
            ],
        }
    }

    #[test]
    fn test_order_book_state_creation() {
        let snapshot = create_test_snapshot();
        let state = OrderBookState::from_snapshot(snapshot).unwrap();

        assert_eq!(state.analytics.mid_price, 100.5);
        assert_eq!(state.analytics.spread, 1.0);
        assert!(state.analytics.relative_spread > 0.0);
    }

    #[test]
    fn test_order_imbalance() {
        let snapshot = create_test_snapshot();
        let state = OrderBookState::from_snapshot(snapshot).unwrap();

        // Bid volume = 5 + 3 + 2 = 10
        // Ask volume = 4 + 2 + 1 = 7
        // Imbalance = (10 - 7) / (10 + 7) = 3/17 ≈ 0.176
        assert!(state.analytics.order_imbalance > 0.15);
        assert!(state.analytics.order_imbalance < 0.20);
    }

    #[test]
    fn test_vwmp_calculation() {
        let snapshot = create_test_snapshot();
        let (vwmp, bid_vol, ask_vol) = calculate_vwmp(&snapshot);

        // VWMP should be weighted toward the side with more volume
        assert!(vwmp > 100.0);  // Should be between best bid and best ask
        assert!(vwmp < 101.0);

        assert_eq!(bid_vol, 10.0);  // 5 + 3 + 2
        assert_eq!(ask_vol, 7.0);   // 4 + 2 + 1
    }

    #[test]
    fn test_depth_at_best() {
        let snapshot = create_test_snapshot();
        let state = OrderBookState::from_snapshot(snapshot).unwrap();

        assert_eq!(state.analytics.bid_depth, 5.0);
        assert_eq!(state.analytics.ask_depth, 4.0);
    }

    #[test]
    fn test_update_state() {
        let snapshot1 = create_test_snapshot();
        let mut state = OrderBookState::from_snapshot(snapshot1).unwrap();

        let old_mid = state.mid_price();

        // Create new snapshot with different prices
        let snapshot2 = L2Snapshot {
            symbol: "BTC-USD".to_string(),
            timestamp_us: 2000000,
            bids: vec![
                (Price::new(102.0).unwrap(), Quantity::new(3.0).unwrap()),
            ],
            asks: vec![
                (Price::new(103.0).unwrap(), Quantity::new(2.0).unwrap()),
            ],
        };

        state.update(snapshot2).unwrap();

        let new_mid = state.mid_price();
        assert!(new_mid > old_mid);
        assert_eq!(new_mid, 102.5);
    }
}
