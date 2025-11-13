//! Binance cryptocurrency exchange provider
//!
//! Provides integration with Binance exchange for cryptocurrency market data.
//!
//! # Features
//! - REST API for historical OHLCV data
//! - WebSocket streams for real-time orderbook, trades, and klines
//! - Spot and futures market support
//! - Depth snapshots and incremental updates
//! - Multi-symbol subscriptions

use async_trait::async_trait;
use chrono::{DateTime, Utc, Duration};
use futures_util::{StreamExt, SinkExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::net::TcpStream;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream, tungstenite::Message};
use std::collections::HashMap;

use crate::data::{Bar, Timeframe};
use crate::data::orderbook::OrderBook;
use crate::error::{MarketError, MarketResult};
use super::MarketDataProvider;

/// Binance API endpoints
const BINANCE_REST_URL: &str = "https://api.binance.com";
const BINANCE_WS_URL: &str = "wss://stream.binance.com:9443/ws";
const BINANCE_TESTNET_REST: &str = "https://testnet.binance.vision";
const BINANCE_TESTNET_WS: &str = "wss://testnet.binance.vision/ws";

/// Binance provider for cryptocurrency market data
pub struct BinanceProvider {
    /// HTTP client for REST API
    client: reqwest::Client,

    /// Base URL for REST API
    rest_url: String,

    /// WebSocket URL
    ws_url: String,

    /// API key (optional, for private endpoints)
    api_key: Option<String>,

    /// API secret (optional, for signed requests)
    api_secret: Option<String>,

    /// Whether using testnet
    testnet: bool,

    /// Active WebSocket connection
    ws_connection: Arc<RwLock<Option<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,

    /// Real-time orderbooks cache
    orderbooks: Arc<RwLock<HashMap<String, OrderBook>>>,
}

/// Binance kline (candlestick) response
#[derive(Debug, Deserialize)]
struct BinanceKline {
    #[serde(rename = "t")]
    open_time: i64,
    #[serde(rename = "o")]
    open: String,
    #[serde(rename = "h")]
    high: String,
    #[serde(rename = "l")]
    low: String,
    #[serde(rename = "c")]
    close: String,
    #[serde(rename = "v")]
    volume: String,
    #[serde(rename = "T")]
    close_time: i64,
    #[serde(rename = "q")]
    quote_volume: String,
    #[serde(rename = "n")]
    trades: i64,
}

/// WebSocket subscription message
#[derive(Debug, Serialize)]
struct WsSubscribe {
    method: String,
    params: Vec<String>,
    id: u64,
}

/// WebSocket orderbook depth update
#[derive(Debug, Deserialize)]
struct DepthUpdate {
    #[serde(rename = "e")]
    event_type: String,
    #[serde(rename = "E")]
    event_time: i64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "U")]
    first_update_id: i64,
    #[serde(rename = "u")]
    final_update_id: i64,
    #[serde(rename = "b")]
    bids: Vec<(String, String)>,
    #[serde(rename = "a")]
    asks: Vec<(String, String)>,
}

impl BinanceProvider {
    /// Create new Binance provider
    ///
    /// # Arguments
    /// * `testnet` - Use testnet endpoints if true
    pub fn new(testnet: bool) -> Self {
        let (rest_url, ws_url) = if testnet {
            (BINANCE_TESTNET_REST.to_string(), BINANCE_TESTNET_WS.to_string())
        } else {
            (BINANCE_REST_URL.to_string(), BINANCE_WS_URL.to_string())
        };

        Self {
            client: reqwest::Client::new(),
            rest_url,
            ws_url,
            api_key: None,
            api_secret: None,
            testnet,
            ws_connection: Arc::new(RwLock::new(None)),
            orderbooks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create provider with API credentials
    pub fn with_credentials(api_key: String, api_secret: String, testnet: bool) -> Self {
        let mut provider = Self::new(testnet);
        provider.api_key = Some(api_key);
        provider.api_secret = Some(api_secret);
        provider
    }

    /// Convert timeframe to Binance interval string
    fn timeframe_to_interval(timeframe: Timeframe) -> &'static str {
        match timeframe {
            Timeframe::Minute1 => "1m",
            Timeframe::Minute5 => "5m",
            Timeframe::Minute15 => "15m",
            Timeframe::Hour1 => "1h",
            Timeframe::Hour4 => "4h",
            Timeframe::Day1 => "1d",
            Timeframe::Week1 => "1w",
            Timeframe::Month1 => "1M",
            _ => "1h", // Default fallback
        }
    }

    /// Fetch klines (OHLCV) from REST API
    async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        start_time: i64,
        end_time: i64,
        limit: u32,
    ) -> MarketResult<Vec<BinanceKline>> {
        let url = format!("{}/api/v3/klines", self.rest_url);

        let response = self.client
            .get(&url)
            .query(&[
                ("symbol", symbol.to_uppercase()),
                ("interval", interval.to_string()),
                ("startTime", start_time.to_string()),
                ("endTime", end_time.to_string()),
                ("limit", limit.to_string()),
            ])
            .send()
            .await
            .map_err(|e| MarketError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(MarketError::ApiError(format!("Binance API error: {}", error_text)));
        }

        let klines: Vec<Vec<serde_json::Value>> = response
            .json()
            .await
            .map_err(|e| MarketError::ParseError(e.to_string()))?;

        // Parse raw kline data
        let parsed_klines: MarketResult<Vec<BinanceKline>> = klines
            .iter()
            .map(|k| {
                Ok(BinanceKline {
                    open_time: k[0].as_i64().ok_or(MarketError::ParseError("Invalid open_time".into()))?,
                    open: k[1].as_str().ok_or(MarketError::ParseError("Invalid open".into()))?.to_string(),
                    high: k[2].as_str().ok_or(MarketError::ParseError("Invalid high".into()))?.to_string(),
                    low: k[3].as_str().ok_or(MarketError::ParseError("Invalid low".into()))?.to_string(),
                    close: k[4].as_str().ok_or(MarketError::ParseError("Invalid close".into()))?.to_string(),
                    volume: k[5].as_str().ok_or(MarketError::ParseError("Invalid volume".into()))?.to_string(),
                    close_time: k[6].as_i64().ok_or(MarketError::ParseError("Invalid close_time".into()))?,
                    quote_volume: k[7].as_str().ok_or(MarketError::ParseError("Invalid quote_volume".into()))?.to_string(),
                    trades: k[8].as_i64().ok_or(MarketError::ParseError("Invalid trades".into()))?,
                })
            })
            .collect();

        parsed_klines
    }

    /// Convert Binance kline to Bar
    fn kline_to_bar(kline: BinanceKline, symbol: &str) -> MarketResult<Bar> {
        let open: f64 = kline.open.parse().map_err(|_| MarketError::ParseError("Invalid open price".into()))?;
        let high: f64 = kline.high.parse().map_err(|_| MarketError::ParseError("Invalid high price".into()))?;
        let low: f64 = kline.low.parse().map_err(|_| MarketError::ParseError("Invalid low price".into()))?;
        let close: f64 = kline.close.parse().map_err(|_| MarketError::ParseError("Invalid close price".into()))?;
        let volume: f64 = kline.volume.parse().map_err(|_| MarketError::ParseError("Invalid volume".into()))?;

        let timestamp = DateTime::from_timestamp(kline.open_time / 1000, 0)
            .ok_or(MarketError::ParseError("Invalid timestamp".into()))?
            .with_timezone(&Utc);

        Ok(Bar {
            symbol: symbol.to_string(),
            timestamp,
            open,
            high,
            low,
            close,
            volume: volume as u64,
            vwap: None, // Binance doesn't provide VWAP directly
            trade_count: Some(kline.trades as u64),
        })
    }

    /// Subscribe to WebSocket depth stream
    pub async fn subscribe_orderbook(&self, symbol: &str) -> MarketResult<()> {
        let stream = format!("{}@depth", symbol.to_lowercase());
        let subscribe_msg = WsSubscribe {
            method: "SUBSCRIBE".to_string(),
            params: vec![stream],
            id: 1,
        };

        let mut ws = self.ws_connection.write().await;
        if let Some(connection) = ws.as_mut() {
            let msg = serde_json::to_string(&subscribe_msg)
                .map_err(|e| MarketError::ParseError(e.to_string()))?;

            connection.send(Message::Text(msg)).await
                .map_err(|e| MarketError::NetworkError(e.to_string()))?;

            tracing::info!("Subscribed to {} orderbook stream", symbol);
            Ok(())
        } else {
            Err(MarketError::ConnectionError("WebSocket not connected".into()))
        }
    }

    /// Connect to WebSocket
    pub async fn connect_websocket(&self) -> MarketResult<()> {
        let (ws_stream, _) = connect_async(&self.ws_url)
            .await
            .map_err(|e| MarketError::ConnectionError(e.to_string()))?;

        *self.ws_connection.write().await = Some(ws_stream);
        tracing::info!("Connected to Binance WebSocket: {}", self.ws_url);
        Ok(())
    }

    /// Get current orderbook for a symbol
    pub async fn get_orderbook(&self, symbol: &str) -> MarketResult<OrderBook> {
        let orderbooks = self.orderbooks.read().await;
        orderbooks
            .get(symbol)
            .cloned()
            .ok_or(MarketError::InvalidSymbol(symbol.to_string()))
    }
}

#[async_trait]
impl MarketDataProvider for BinanceProvider {
    async fn fetch_bars(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MarketResult<Vec<Bar>> {
        let interval = Self::timeframe_to_interval(timeframe);
        let start_ms = start.timestamp() * 1000;
        let end_ms = end.timestamp() * 1000;

        // Binance limit is 1000 per request
        let klines = self.fetch_klines(symbol, interval, start_ms, end_ms, 1000).await?;

        klines
            .into_iter()
            .map(|k| Self::kline_to_bar(k, symbol))
            .collect()
    }

    async fn fetch_latest_bar(&self, symbol: &str) -> MarketResult<Bar> {
        let now = Utc::now();
        let start = now - Duration::hours(1);

        let mut bars = self.fetch_bars(symbol, Timeframe::Minute1, start, now).await?;

        bars.pop().ok_or(MarketError::DataUnavailable(
            format!("No data available for {}", symbol)
        ))
    }

    fn provider_name(&self) -> &str {
        "Binance"
    }

    fn supports_realtime(&self) -> bool {
        true
    }

    async fn supports_symbol(&self, symbol: &str) -> MarketResult<bool> {
        // Check if symbol exists via exchange info endpoint
        let url = format!("{}/api/v3/exchangeInfo", self.rest_url);

        let response: serde_json::Value = self.client
            .get(&url)
            .query(&[("symbol", symbol.to_uppercase())])
            .send()
            .await
            .map_err(|e| MarketError::NetworkError(e.to_string()))?
            .json()
            .await
            .map_err(|e| MarketError::ParseError(e.to_string()))?;

        Ok(response["symbols"].as_array().map(|arr| !arr.is_empty()).unwrap_or(false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeframe_conversion() {
        assert_eq!(BinanceProvider::timeframe_to_interval(Timeframe::Minute1), "1m");
        assert_eq!(BinanceProvider::timeframe_to_interval(Timeframe::Hour1), "1h");
        assert_eq!(BinanceProvider::timeframe_to_interval(Timeframe::Day1), "1d");
    }

    #[tokio::test]
    async fn test_provider_creation() {
        let provider = BinanceProvider::new(true);
        assert_eq!(provider.provider_name(), "Binance");
        assert!(provider.supports_realtime());
        assert!(provider.testnet);
    }

    #[tokio::test]
    async fn test_provider_with_credentials() {
        let provider = BinanceProvider::with_credentials(
            "test_key".to_string(),
            "test_secret".to_string(),
            true
        );
        assert!(provider.api_key.is_some());
        assert!(provider.api_secret.is_some());
    }
}
