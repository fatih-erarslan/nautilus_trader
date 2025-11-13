//! OHLCV bar data structures

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Represents a single OHLCV (Open, High, Low, Close, Volume) bar
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Bar {
    /// Trading symbol (e.g., "AAPL", "BTCUSD")
    pub symbol: String,

    /// Timestamp of the bar (start of the period)
    pub timestamp: DateTime<Utc>,

    /// Opening price
    pub open: f64,

    /// Highest price during the period
    pub high: f64,

    /// Lowest price during the period
    pub low: f64,

    /// Closing price
    pub close: f64,

    /// Trading volume
    pub volume: u64,

    /// Volume-weighted average price (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vwap: Option<f64>,

    /// Number of trades (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trade_count: Option<u64>,
}

impl Bar {
    /// Create a new Bar instance
    pub fn new(
        symbol: String,
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: u64,
    ) -> Self {
        Self {
            symbol,
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            vwap: None,
            trade_count: None,
        }
    }

    /// Calculate the bar's typical price: (high + low + close) / 3
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate the bar's range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if this is a bullish bar (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if this is a bearish bar (close < open)
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }
}

/// Timeframe enumeration for bar data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Timeframe {
    /// 1-minute bars
    Minute1,

    /// 5-minute bars
    Minute5,

    /// 15-minute bars
    Minute15,

    /// 30-minute bars
    Minute30,

    /// 1-hour bars
    Hour1,

    /// 4-hour bars
    Hour4,

    /// Daily bars
    Day1,

    /// Weekly bars
    Week1,

    /// Monthly bars
    Month1,
}

impl Timeframe {
    /// Convert timeframe to string representation for API calls
    pub fn as_str(&self) -> &'static str {
        match self {
            Timeframe::Minute1 => "1Min",
            Timeframe::Minute5 => "5Min",
            Timeframe::Minute15 => "15Min",
            Timeframe::Minute30 => "30Min",
            Timeframe::Hour1 => "1Hour",
            Timeframe::Hour4 => "4Hour",
            Timeframe::Day1 => "1Day",
            Timeframe::Week1 => "1Week",
            Timeframe::Month1 => "1Month",
        }
    }

    /// Get duration in seconds
    pub fn duration_secs(&self) -> i64 {
        match self {
            Timeframe::Minute1 => 60,
            Timeframe::Minute5 => 300,
            Timeframe::Minute15 => 900,
            Timeframe::Minute30 => 1800,
            Timeframe::Hour1 => 3600,
            Timeframe::Hour4 => 14400,
            Timeframe::Day1 => 86400,
            Timeframe::Week1 => 604800,
            Timeframe::Month1 => 2592000, // Approximate (30 days)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use approx::assert_relative_eq;

    #[test]
    fn test_bar_creation() {
        let bar = Bar::new(
            "AAPL".to_string(),
            Utc::now(),
            150.0,
            155.0,
            149.0,
            154.0,
            1000000,
        );

        assert_eq!(bar.symbol, "AAPL");
        assert_eq!(bar.open, 150.0);
        assert!(bar.is_bullish());
    }

    #[test]
    fn test_bar_metrics() {
        let bar = Bar::new(
            "TEST".to_string(),
            Utc::now(),
            100.0,
            110.0,
            95.0,
            105.0,
            5000,
        );

        // Typical price = (high + low + close) / 3 = (110 + 95 + 105) / 3 = 103.333...
        assert_relative_eq!(bar.typical_price(), 103.33333333333333, epsilon = 1e-10);
        assert_relative_eq!(bar.range(), 15.0, epsilon = 1e-10);
        assert!(bar.is_bullish());
        assert!(!bar.is_bearish());
    }
}
