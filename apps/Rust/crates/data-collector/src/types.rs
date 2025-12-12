//! Type definitions for cryptocurrency data

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Market data types that can be collected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DataType {
    /// OHLCV candlestick data
    Klines,
    /// Individual trades
    Trades,
    /// Order book snapshots
    OrderBook,
    /// Funding rates (for futures)
    FundingRates,
    /// 24hr ticker statistics
    Ticker24hr,
    /// Mark price (for derivatives)
    MarkPrice,
}

/// Time intervals for candlestick data
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Interval {
    #[serde(rename = "1m")]
    OneMinute,
    #[serde(rename = "3m")]
    ThreeMinutes,
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "15m")]
    FifteenMinutes,
    #[serde(rename = "30m")]
    ThirtyMinutes,
    #[serde(rename = "1h")]
    OneHour,
    #[serde(rename = "2h")]
    TwoHours,
    #[serde(rename = "4h")]
    FourHours,
    #[serde(rename = "6h")]
    SixHours,
    #[serde(rename = "8h")]
    EightHours,
    #[serde(rename = "12h")]
    TwelveHours,
    #[serde(rename = "1d")]
    OneDay,
    #[serde(rename = "3d")]
    ThreeDays,
    #[serde(rename = "1w")]
    OneWeek,
    #[serde(rename = "1M")]
    OneMonth,
}

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub symbol: String,
    pub open_time: DateTime<Utc>,
    pub close_time: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub quote_volume: f64,
    pub trades_count: u64,
    pub taker_buy_base_volume: f64,
    pub taker_buy_quote_volume: f64,
    pub interval: Interval,
    pub exchange: String,
}

/// Individual trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub symbol: String,
    pub trade_id: u64,
    pub price: f64,
    pub quantity: f64,
    pub quote_quantity: f64,
    pub timestamp: DateTime<Utc>,
    pub is_buyer_maker: bool,
    pub exchange: String,
}

/// Order book data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub exchange: String,
}

/// Order book price level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// Funding rate data (for futures)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    pub symbol: String,
    pub funding_rate: f64,
    pub funding_time: DateTime<Utc>,
    pub mark_price: f64,
    pub exchange: String,
}

/// 24hr ticker statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker24hr {
    pub symbol: String,
    pub price_change: f64,
    pub price_change_percent: f64,
    pub weighted_avg_price: f64,
    pub prev_close_price: f64,
    pub last_price: f64,
    pub last_qty: f64,
    pub bid_price: f64,
    pub bid_qty: f64,
    pub ask_price: f64,
    pub ask_qty: f64,
    pub open_price: f64,
    pub high_price: f64,
    pub low_price: f64,
    pub volume: f64,
    pub quote_volume: f64,
    pub open_time: DateTime<Utc>,
    pub close_time: DateTime<Utc>,
    pub first_id: u64,
    pub last_id: u64,
    pub count: u64,
    pub exchange: String,
}

/// Collection parameters for data requests
#[derive(Debug, Clone)]
pub struct CollectionParams {
    pub symbols: Vec<String>,
    pub data_types: Vec<DataType>,
    pub interval: Option<Interval>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub limit: Option<u32>,
}

/// Collection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStats {
    pub total_records: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub data_quality_score: f64,
    pub collection_duration_ms: u64,
    pub average_latency_ms: f64,
    pub rate_limit_hits: u64,
}

impl std::fmt::Display for Interval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Interval::OneMinute => "1m",
            Interval::ThreeMinutes => "3m",
            Interval::FiveMinutes => "5m",
            Interval::FifteenMinutes => "15m",
            Interval::ThirtyMinutes => "30m",
            Interval::OneHour => "1h",
            Interval::TwoHours => "2h",
            Interval::FourHours => "4h",
            Interval::SixHours => "6h",
            Interval::EightHours => "8h",
            Interval::TwelveHours => "12h",
            Interval::OneDay => "1d",
            Interval::ThreeDays => "3d",
            Interval::OneWeek => "1w",
            Interval::OneMonth => "1M",
        };
        write!(f, "{}", s)
    }
}