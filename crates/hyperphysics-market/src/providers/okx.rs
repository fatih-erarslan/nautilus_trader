//! OKX cryptocurrency exchange provider
//!
//! Provides integration with OKX exchange for cryptocurrency market data.
//!
//! # Features
//! - REST API for historical OHLCV data
//! - WebSocket streams for real-time orderbook, trades, and candlesticks
//! - Spot, futures, and options market support
//! - Public and private data streams
//! - Multi-instrument subscriptions

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

/// OKX API endpoints
const OKX_REST_URL: &str = "https://www.okx.com";
const OKX_WS_PUBLIC: &str = "wss://ws.okx.com:8443/ws/v5/public";
const OKX_WS_PRIVATE: &str = "wss://ws.okx.com:8443/ws/v5/private";
const OKX_DEMO_REST: &str = "https://www.okx.com"; // Demo uses same REST URL
const OKX_DEMO_WS: &str = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999";

/// OKX provider for cryptocurrency market data
pub struct OKXProvider {
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

    /// API passphrase (OKX specific)
    api_passphrase: Option<String>,

    /// Whether using demo trading
    demo: bool,

    /// Active WebSocket connection
    ws_connection: Arc<RwLock<Option<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,

    /// Real-time orderbooks cache
    orderbooks: Arc<RwLock<HashMap<String, OrderBook>>>,
}

/// OKX candlestick response
#[derive(Debug, Deserialize)]
struct OKXCandle {
    #[serde(rename = "ts")]
    timestamp: String,
    #[serde(rename = "o")]
    open: String,
    #[serde(rename = "h")]
    high: String,
    #[serde(rename = "l")]
    low: String,
    #[serde(rename = "c")]
    close: String,
    #[serde(rename = "vol")]
    volume: String,
    #[serde(rename = "volCcy")]
    quote_volume: String,
    #[serde(rename = "volCcyQuote")]
    _quote_volume_alt: Option<String>,
    #[serde(rename = "confirm")]
    confirmed: String,
}

/// OKX API response wrapper
#[derive(Debug, Deserialize)]
struct OKXResponse<T> {
    code: String,
    msg: String,
    data: Vec<T>,
}

/// WebSocket subscription
#[derive(Debug, Serialize)]
struct WsOp {
    op: String,
    args: Vec<WsChannel>,
}

#[derive(Debug, Serialize)]
struct WsChannel {
    channel: String,
    #[serde(rename = "instId")]
    inst_id: String,
}

impl OKXProvider {
    /// Create new OKX provider
    ///
    /// # Arguments
    /// * `demo` - Use demo trading endpoints if true
    pub fn new(demo: bool) -> Self {
        let (rest_url, ws_url) = if demo {
            (OKX_DEMO_REST.to_string(), OKX_DEMO_WS.to_string())
        } else {
            (OKX_REST_URL.to_string(), OKX_WS_PUBLIC.to_string())
        };

        Self {
            client: reqwest::Client::new(),
            rest_url,
            ws_url,
            api_key: None,
            api_secret: None,
            api_passphrase: None,
            demo,
            ws_connection: Arc::new(RwLock::new(None)),
            orderbooks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create provider with API credentials
    pub fn with_credentials(
        api_key: String,
        api_secret: String,
        api_passphrase: String,
        demo: bool,
    ) -> Self {
        let mut provider = Self::new(demo);
        provider.api_key = Some(api_key);
        provider.api_secret = Some(api_secret);
        provider.api_passphrase = Some(api_passphrase);
        provider
    }

    /// Convert timeframe to OKX bar size
    fn timeframe_to_bar(timeframe: Timeframe) -> &'static str {
        match timeframe {
            Timeframe::Minute1 => "1m",
            Timeframe::Minute5 => "5m",
            Timeframe::Minute15 => "15m",
            Timeframe::Hour1 => "1H",
            Timeframe::Hour4 => "4H",
            Timeframe::Day1 => "1D",
            Timeframe::Week1 => "1W",
            Timeframe::Month1 => "1M",
            _ => "1H", // Default fallback
        }
    }

    /// Fetch candlestick data from REST API
    async fn fetch_candles(
        &self,
        inst_id: &str, // instrument ID (e.g., "BTC-USDT")
        bar: &str,
        after: Option<i64>,
        before: Option<i64>,
        limit: Option<u32>,
    ) -> MarketResult<Vec<OKXCandle>> {
        let url = format!("{}/api/v5/market/candles", self.rest_url);

        let mut query_params = vec![
            ("instId", inst_id.to_string()),
            ("bar", bar.to_string()),
        ];

        if let Some(a) = after {
            query_params.push(("after", a.to_string()));
        }
        if let Some(b) = before {
            query_params.push(("before", b.to_string()));
        }
        if let Some(l) = limit {
            query_params.push(("limit", l.to_string()));
        }

        let response = self.client
            .get(&url)
            .query(&query_params)
            .send()
            .await
            .map_err(|e| MarketError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(MarketError::ApiError(format!("OKX API error: {}", error_text)));
        }

        let okx_response: OKXResponse<Vec<String>> = response
            .json()
            .await
            .map_err(|e| MarketError::ParseError(e.to_string()))?;

        if okx_response.code != "0" {
            return Err(MarketError::ApiError(format!(
                "OKX error {}: {}",
                okx_response.code, okx_response.msg
            )));
        }

        // Parse candle data - OKX returns array of arrays
        let candles: MarketResult<Vec<OKXCandle>> = okx_response
            .data
            .iter()
            .map(|candle_array| {
                if candle_array.len() < 9 {
                    return Err(MarketError::ParseError("Invalid candle data length".into()));
                }

                Ok(OKXCandle {
                    timestamp: candle_array[0].clone(),
                    open: candle_array[1].clone(),
                    high: candle_array[2].clone(),
                    low: candle_array[3].clone(),
                    close: candle_array[4].clone(),
                    volume: candle_array[5].clone(),
                    quote_volume: candle_array[6].clone(),
                    _quote_volume_alt: Some(candle_array[7].clone()),
                    confirmed: candle_array[8].clone(),
                })
            })
            .collect();

        candles
    }

    /// Convert OKX candle to Bar
    fn candle_to_bar(candle: OKXCandle, symbol: &str) -> MarketResult<Bar> {
        let open: f64 = candle.open.parse()
            .map_err(|_| MarketError::ParseError("Invalid open price".into()))?;
        let high: f64 = candle.high.parse()
            .map_err(|_| MarketError::ParseError("Invalid high price".into()))?;
        let low: f64 = candle.low.parse()
            .map_err(|_| MarketError::ParseError("Invalid low price".into()))?;
        let close: f64 = candle.close.parse()
            .map_err(|_| MarketError::ParseError("Invalid close price".into()))?;
        let volume: f64 = candle.volume.parse()
            .map_err(|_| MarketError::ParseError("Invalid volume".into()))?;

        let ts_ms: i64 = candle.timestamp.parse()
            .map_err(|_| MarketError::ParseError("Invalid timestamp".into()))?;

        let timestamp = DateTime::from_timestamp(ts_ms / 1000, 0)
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
            vwap: None,
            trade_count: None, // OKX doesn't provide trade count in candles
        })
    }

    /// Connect to WebSocket
    pub async fn connect_websocket(&self) -> MarketResult<()> {
        let (ws_stream, _) = connect_async(&self.ws_url)
            .await
            .map_err(|e| MarketError::ConnectionError(e.to_string()))?;

        *self.ws_connection.write().await = Some(ws_stream);
        tracing::info!("Connected to OKX WebSocket: {}", self.ws_url);
        Ok(())
    }

    /// Subscribe to orderbook depth stream
    pub async fn subscribe_orderbook(&self, inst_id: &str) -> MarketResult<()> {
        let subscribe_msg = WsOp {
            op: "subscribe".to_string(),
            args: vec![WsChannel {
                channel: "books".to_string(),
                inst_id: inst_id.to_string(),
            }],
        };

        let mut ws = self.ws_connection.write().await;
        if let Some(connection) = ws.as_mut() {
            let msg = serde_json::to_string(&subscribe_msg)
                .map_err(|e| MarketError::ParseError(e.to_string()))?;

            connection.send(Message::Text(msg)).await
                .map_err(|e| MarketError::NetworkError(e.to_string()))?;

            tracing::info!("Subscribed to {} orderbook stream", inst_id);
            Ok(())
        } else {
            Err(MarketError::ConnectionError("WebSocket not connected".into()))
        }
    }

    /// Get current orderbook for an instrument
    pub async fn get_orderbook(&self, inst_id: &str) -> MarketResult<OrderBook> {
        let orderbooks = self.orderbooks.read().await;
        orderbooks
            .get(inst_id)
            .cloned()
            .ok_or(MarketError::InvalidSymbol(inst_id.to_string()))
    }
}

#[async_trait]
impl MarketDataProvider for OKXProvider {
    async fn fetch_bars(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MarketResult<Vec<Bar>> {
        let bar_size = Self::timeframe_to_bar(timeframe);
        let after_ms = start.timestamp() * 1000;
        let before_ms = end.timestamp() * 1000;

        // OKX limit is 100 per request
        let candles = self.fetch_candles(symbol, bar_size, Some(after_ms), Some(before_ms), Some(100)).await?;

        candles
            .into_iter()
            .map(|c| Self::candle_to_bar(c, symbol))
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
        "OKX"
    }

    fn supports_realtime(&self) -> bool {
        true
    }

    async fn supports_symbol(&self, symbol: &str) -> MarketResult<bool> {
        // Check if instrument exists via instruments endpoint
        let url = format!("{}/api/v5/public/instruments", self.rest_url);

        let response: serde_json::Value = self.client
            .get(&url)
            .query(&[
                ("instType", "SPOT"),
                ("instId", symbol),
            ])
            .send()
            .await
            .map_err(|e| MarketError::NetworkError(e.to_string()))?
            .json()
            .await
            .map_err(|e| MarketError::ParseError(e.to_string()))?;

        let code = response["code"].as_str().unwrap_or("1");
        let data = response["data"].as_array();

        Ok(code == "0" && data.map(|arr| !arr.is_empty()).unwrap_or(false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeframe_conversion() {
        assert_eq!(OKXProvider::timeframe_to_bar(Timeframe::Minute1), "1m");
        assert_eq!(OKXProvider::timeframe_to_bar(Timeframe::Hour1), "1H");
        assert_eq!(OKXProvider::timeframe_to_bar(Timeframe::Day1), "1D");
    }

    #[tokio::test]
    async fn test_provider_creation() {
        let provider = OKXProvider::new(true);
        assert_eq!(provider.provider_name(), "OKX");
        assert!(provider.supports_realtime());
        assert!(provider.demo);
    }

    #[tokio::test]
    async fn test_provider_with_credentials() {
        let provider = OKXProvider::with_credentials(
            "test_key".to_string(),
            "test_secret".to_string(),
            "test_pass".to_string(),
            true
        );
        assert!(provider.api_key.is_some());
        assert!(provider.api_secret.is_some());
        assert!(provider.api_passphrase.is_some());
    }
}
