//! Type conversion implementations between Nautilus and HyperPhysics types.

use crate::error::{IntegrationError, Result};
use crate::types::{
    MarketSnapshot, NautilusBar, NautilusOrderBookDelta, NautilusQuoteTick,
    NautilusTradeTick, OrderSide, OrderType, TimeInForce, HyperPhysicsOrderCommand,
};
use hyperphysics_hft_ecosystem::core::{Action, MarketTick, TradingDecision};

/// Precision scaling factor for fixed-point conversion
const PRECISION_SCALE: [f64; 17] = [
    1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0,
    100000000.0, 1000000000.0, 10000000000.0, 100000000000.0, 1000000000000.0,
    10000000000000.0, 100000000000000.0, 1000000000000000.0, 10000000000000000.0,
];

/// Convert fixed-point integer to f64 with precision
#[inline]
pub fn fixed_to_f64(value: i64, precision: u8) -> f64 {
    value as f64 / PRECISION_SCALE[precision as usize]
}

/// Convert fixed-point unsigned to f64 with precision
#[inline]
pub fn fixed_u64_to_f64(value: u64, precision: u8) -> f64 {
    value as f64 / PRECISION_SCALE[precision as usize]
}

/// Convert f64 to fixed-point integer with precision
#[inline]
pub fn f64_to_fixed(value: f64, precision: u8) -> i64 {
    (value * PRECISION_SCALE[precision as usize]).round() as i64
}

/// Convert nanoseconds to seconds
#[inline]
pub fn nanos_to_seconds(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000_000.0
}

/// Convert seconds to nanoseconds
#[inline]
pub fn seconds_to_nanos(seconds: f64) -> u64 {
    (seconds * 1_000_000_000.0) as u64
}

impl NautilusQuoteTick {
    /// Convert to MarketSnapshot
    pub fn to_market_snapshot(&self) -> MarketSnapshot {
        let bid = fixed_to_f64(self.bid_price, self.price_precision);
        let ask = fixed_to_f64(self.ask_price, self.price_precision);
        let bid_size = fixed_u64_to_f64(self.bid_size, self.size_precision);
        let ask_size = fixed_u64_to_f64(self.ask_size, self.size_precision);

        let mid = (bid + ask) / 2.0;
        let spread = ask - bid;
        let spread_bps = if mid > 0.0 { (spread / mid) * 10000.0 } else { 0.0 };

        let mut snapshot = MarketSnapshot {
            mid_price: mid,
            bid_price: bid,
            ask_price: ask,
            bid_size,
            ask_size,
            spread,
            spread_bps,
            vwap: mid, // Initialize with mid
            timestamp: nanos_to_seconds(self.ts_event),
            instrument_id: format!("INST-{}", self.instrument_id),
            ..Default::default()
        };

        snapshot.calculate_imbalance();
        snapshot
    }

    /// Convert to HyperPhysics MarketTick
    pub fn to_market_tick(&self) -> Result<MarketTick> {
        let snapshot = self.to_market_snapshot();

        // Serialize orderbook data
        let orderbook = bincode::serialize(&(snapshot.bid_price, snapshot.ask_price, snapshot.bid_size, snapshot.ask_size))
            .map_err(|e| IntegrationError::Serialization(e.to_string()))?;

        Ok(MarketTick {
            timestamp: chrono::DateTime::from_timestamp_nanos(self.ts_event as i64),
            orderbook,
            trades: vec![],
        })
    }
}

impl NautilusTradeTick {
    /// Convert to trade record for inclusion in MarketTick
    pub fn to_trade_record(&self) -> (f64, f64, bool) {
        let price = fixed_to_f64(self.price, self.price_precision);
        let size = fixed_u64_to_f64(self.size, self.size_precision);
        let is_buyer_maker = self.aggressor_side == 2; // Seller aggressor = buyer maker

        (price, size, is_buyer_maker)
    }

    /// Calculate return from previous price
    pub fn calculate_return(&self, prev_price: i64) -> f64 {
        if prev_price == 0 {
            return 0.0;
        }
        let current = fixed_to_f64(self.price, self.price_precision);
        let previous = fixed_to_f64(prev_price, self.price_precision);
        (current - previous) / previous
    }
}

impl NautilusBar {
    /// Convert to return value
    pub fn to_return(&self) -> f64 {
        let open = fixed_to_f64(self.open, self.price_precision);
        let close = fixed_to_f64(self.close, self.price_precision);
        if open == 0.0 {
            return 0.0;
        }
        (close - open) / open
    }

    /// Calculate bar range (high - low)
    pub fn range(&self) -> f64 {
        let high = fixed_to_f64(self.high, self.price_precision);
        let low = fixed_to_f64(self.low, self.price_precision);
        high - low
    }

    /// Calculate typical price (HLC/3)
    pub fn typical_price(&self) -> f64 {
        let high = fixed_to_f64(self.high, self.price_precision);
        let low = fixed_to_f64(self.low, self.price_precision);
        let close = fixed_to_f64(self.close, self.price_precision);
        (high + low + close) / 3.0
    }

    /// Update market snapshot with bar data
    pub fn update_snapshot(&self, snapshot: &mut MarketSnapshot) {
        let close = fixed_to_f64(self.close, self.price_precision);
        let volume = fixed_u64_to_f64(self.volume, self.size_precision);

        // Calculate return from previous mid price
        if snapshot.mid_price > 0.0 {
            let ret = (close - snapshot.mid_price) / snapshot.mid_price;
            snapshot.push_return(ret, 50);
        }

        snapshot.mid_price = close;
        snapshot.volume += volume;
        snapshot.timestamp = nanos_to_seconds(self.ts_event);
        snapshot.update_volatility();
    }
}

impl NautilusOrderBookDelta {
    /// Get book action as string
    pub fn action_str(&self) -> &'static str {
        match self.action {
            1 => "ADD",
            2 => "UPDATE",
            3 => "DELETE",
            4 => "CLEAR",
            _ => "UNKNOWN",
        }
    }

    /// Get side as OrderSide
    pub fn order_side(&self) -> OrderSide {
        match self.side {
            1 => OrderSide::Buy,
            2 => OrderSide::Sell,
            _ => OrderSide::NoSide,
        }
    }

    /// Get price as f64
    pub fn price_f64(&self) -> f64 {
        fixed_to_f64(self.price, self.price_precision)
    }

    /// Get size as f64
    pub fn size_f64(&self) -> f64 {
        fixed_u64_to_f64(self.size, self.size_precision)
    }
}

/// Convert HyperPhysics TradingDecision to order command
pub fn decision_to_order_command(
    decision: &TradingDecision,
    instrument_id: &str,
    order_id_prefix: &str,
    algorithm: &str,
    latency_us: u64,
    consensus_term: u64,
) -> Result<Option<HyperPhysicsOrderCommand>> {
    // Skip hold actions
    if matches!(decision.action, Action::Hold) {
        return Ok(None);
    }

    let side = match decision.action {
        Action::Buy => OrderSide::Buy,
        Action::Sell => OrderSide::Sell,
        Action::Hold => return Ok(None),
    };

    let client_order_id = format!(
        "{}-{}-{}",
        order_id_prefix,
        chrono::Utc::now().timestamp_micros(),
        uuid::Uuid::new_v4().to_string().split('-').next().unwrap_or("0000")
    );

    Ok(Some(HyperPhysicsOrderCommand {
        client_order_id,
        instrument_id: instrument_id.to_string(),
        side,
        order_type: OrderType::Market, // Default to market orders
        quantity: decision.size.abs(),
        price: None,
        time_in_force: TimeInForce::IOC,
        reduce_only: false,
        post_only: false,
        hp_confidence: decision.confidence,
        hp_algorithm: algorithm.to_string(),
        hp_latency_us: latency_us,
        hp_consensus_term: consensus_term,
    }))
}

/// Convert multiple bars to return series
pub fn bars_to_returns(bars: &[NautilusBar]) -> Vec<f64> {
    if bars.len() < 2 {
        return vec![];
    }

    bars.windows(2)
        .map(|window| {
            let prev_close = fixed_to_f64(window[0].close, window[0].price_precision);
            let curr_close = fixed_to_f64(window[1].close, window[1].price_precision);
            if prev_close == 0.0 {
                0.0
            } else {
                (curr_close - prev_close) / prev_close
            }
        })
        .collect()
}

/// Calculate realized volatility from bars
pub fn calculate_volatility(bars: &[NautilusBar]) -> f64 {
    let returns = bars_to_returns(bars);
    if returns.len() < 2 {
        return 0.0;
    }

    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance: f64 = returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / (returns.len() - 1) as f64;

    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_conversion() {
        assert!((fixed_to_f64(12345, 2) - 123.45).abs() < 1e-10);
        assert!((fixed_to_f64(1000000, 4) - 100.0).abs() < 1e-10);
        assert_eq!(f64_to_fixed(123.45, 2), 12345);
    }

    #[test]
    fn test_quote_to_snapshot() {
        let quote = NautilusQuoteTick {
            instrument_id: 12345,
            bid_price: 10000,
            ask_price: 10010,
            bid_size: 100,
            ask_size: 150,
            price_precision: 2,
            size_precision: 0,
            ts_event: 1000000000,
            ts_init: 1000000100,
        };

        let snapshot = quote.to_market_snapshot();
        assert!((snapshot.bid_price - 100.0).abs() < 1e-10);
        assert!((snapshot.ask_price - 100.10).abs() < 1e-10);
        assert!((snapshot.spread - 0.10).abs() < 1e-10);
        assert!(snapshot.book_imbalance < 0.0); // More ask size = negative imbalance
    }

    #[test]
    fn test_bar_return() {
        let bar = NautilusBar {
            instrument_id: 1,
            open: 10000,
            high: 10100,
            low: 9900,
            close: 10050,
            volume: 1000,
            price_precision: 2,
            size_precision: 0,
            ts_event: 1000000000,
            ts_init: 1000000100,
        };

        let ret = bar.to_return();
        assert!((ret - 0.005).abs() < 1e-10); // 0.5% return
    }

    #[test]
    fn test_nanos_conversion() {
        let nanos: u64 = 1_500_000_000;
        let seconds = nanos_to_seconds(nanos);
        assert!((seconds - 1.5).abs() < 1e-10);

        let back = seconds_to_nanos(seconds);
        assert_eq!(back, nanos);
    }
}
