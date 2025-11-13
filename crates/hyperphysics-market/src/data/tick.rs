//! Tick data structures for trade and quote updates

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Represents a single trade tick
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tick {
    /// Trading symbol
    pub symbol: String,

    /// Timestamp of the trade
    pub timestamp: DateTime<Utc>,

    /// Trade price
    pub price: f64,

    /// Trade size
    pub size: f64,

    /// Exchange where trade occurred (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exchange: Option<String>,

    /// Conditions/flags for the trade (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conditions: Option<Vec<String>>,
}

/// Represents a quote (bid/ask) update
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Quote {
    /// Trading symbol
    pub symbol: String,

    /// Timestamp of the quote
    pub timestamp: DateTime<Utc>,

    /// Bid price
    pub bid_price: f64,

    /// Bid size
    pub bid_size: f64,

    /// Ask price
    pub ask_price: f64,

    /// Ask size
    pub ask_size: f64,

    /// Exchange (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exchange: Option<String>,
}

impl Quote {
    /// Calculate the bid-ask spread
    pub fn spread(&self) -> f64 {
        self.ask_price - self.bid_price
    }

    /// Calculate the mid price
    pub fn mid_price(&self) -> f64 {
        (self.bid_price + self.ask_price) / 2.0
    }

    /// Calculate spread as percentage of mid price
    pub fn spread_percentage(&self) -> f64 {
        (self.spread() / self.mid_price()) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use approx::assert_relative_eq;

    #[test]
    fn test_quote_calculations() {
        let quote = Quote {
            symbol: "AAPL".to_string(),
            timestamp: Utc::now(),
            bid_price: 150.00,
            bid_size: 100.0,
            ask_price: 150.10,
            ask_size: 100.0,
            exchange: None,
        };

        assert_relative_eq!(quote.spread(), 0.10, epsilon = 1e-10);
        assert_relative_eq!(quote.mid_price(), 150.05, epsilon = 1e-10);
        assert_relative_eq!(quote.spread_percentage(), 0.06663, epsilon = 0.0001);
    }
}
