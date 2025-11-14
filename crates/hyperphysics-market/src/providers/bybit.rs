//! Bybit cryptocurrency exchange provider
//!
//! Provides integration with Bybit exchange for cryptocurrency market data.
//!
//! # Features
//! - REST API for historical kline data
//! - WebSocket streams for real-time orderbook and trades
//! - Spot and derivatives market support
//! - Multi-symbol subscriptions

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose};
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

/// Bybit API endpoints
const BYBIT_REST_URL: &str = "https://api.bybit.com";
const BYBIT_WS_URL: &str = "wss://stream.bybit.com/v5/public/linear";

/// Bybit provider for cryptocurrency market data
pub struct BybitProvider {
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

    /// Active WebSocket connection
    ws_connection: Arc<RwLock<Option<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,

    /// Real-time orderbooks cache
    orderbooks: Arc<RwLock<HashMap<String, OrderBook>>>,
}

/// Bybit kline response
#[derive(Debug, Deserialize)]
struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: Option<BybitKlineResult>,
}

#[derive(Debug, Deserialize)]
struct BybitKlineResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// WebSocket subscription message
#[derive(Debug, Serialize)]
struct WsSubscribe {
    op: String,
    args: Vec<String>,
}

impl BybitProvider {
    /// Create new Bybit provider
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            rest_url: BYBIT_REST_URL.to_string(),
            ws_url: BYBIT_WS_URL.to_string(),
            api_key: None,
            api_secret: None,
            ws_connection: Arc::new(RwLock::new(None)),
            orderbooks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create provider with API credentials
    pub fn with_credentials(api_key: String, api_secret: String) -> Self {
        let mut provider = Self::new();
        provider.api_key = Some(api_key);
        provider.api_secret = Some(api_secret);
        provider
    }

    /// Convert timeframe to Bybit interval string
    fn timeframe_to_interval(timeframe: Timeframe) -> &'static str {
        match timeframe {
            Timeframe::Minute1 => "1",
            Timeframe::Minute5 => "5",
            Timeframe::Minute15 => "15",
            Timeframe::Minute30 => "30",
            Timeframe::Hour1 => "60",
            Timeframe::Hour4 => "240",
            Timeframe::Day1 => "D",
            Timeframe::Week1 => "W",
            Timeframe::Month1 => "M",
        }
    }

    /// Fetch klines from REST API
    async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        limit: u32,
    ) -> MarketResult<BybitKlineResult> {
        let url = format!("{}/v5/market/kline", self.rest_url);

        let response = self.client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", interval),
                ("start", &start.timestamp_millis().to_string()),
                ("end", &end.timestamp_millis().to_string()),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await
            .map_err(|e| MarketError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(MarketError::ApiError(format!("Bybit API error: {}", error_text)));
        }

        let kline_response: BybitKlineResponse = response
            .json()
            .await
            .map_err(|e| MarketError::ParseError(e.to_string()))?;

        if kline_response.ret_code != 0 {
            return Err(MarketError::ApiError(format!("Bybit error: {}", kline_response.ret_msg)));
        }

        kline_response.result.ok_or(MarketError::ParseError("No result data".into()))
    }

    /// Convert Bybit kline entry to Bar
    /// Format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
    fn kline_to_bar(kline: &[String], symbol: &str) -> MarketResult<Bar> {
        if kline.len() < 6 {
            return Err(MarketError::ParseError("Invalid kline format".into()));
        }

        let timestamp_ms: i64 = kline[0].parse().map_err(|_| MarketError::ParseError("Invalid timestamp".into()))?;
        let open: f64 = kline[1].parse().map_err(|_| MarketError::ParseError("Invalid open price".into()))?;
        let high: f64 = kline[2].parse().map_err(|_| MarketError::ParseError("Invalid high price".into()))?;
        let low: f64 = kline[3].parse().map_err(|_| MarketError::ParseError("Invalid low price".into()))?;
        let close: f64 = kline[4].parse().map_err(|_| MarketError::ParseError("Invalid close price".into()))?;
        let volume: f64 = kline[5].parse().map_err(|_| MarketError::ParseError("Invalid volume".into()))?;

        let timestamp = DateTime::from_timestamp(timestamp_ms / 1000, 0)
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
            trade_count: None,
        })
    }

    /// Generate authentication signature for API requests
    fn generate_signature(&self, timestamp: &str, params: &str) -> Option<String> {
        let api_secret = self.api_secret.as_ref()?;

        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        let message = format!("{}{}", timestamp, params);
        let mut mac = Hmac::<Sha256>::new_from_slice(api_secret.as_bytes()).ok()?;
        mac.update(message.as_bytes());
        let result = mac.finalize();

        Some(hex::encode(result.into_bytes()))
    }

    /// Subscribe to WebSocket orderbook stream
    pub async fn subscribe_orderbook(&self, symbol: &str) -> MarketResult<()> {
        let subscribe_msg = WsSubscribe {
            op: "subscribe".to_string(),
            args: vec![format!("orderbook.50.{}", symbol)],
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
        tracing::info!("Connected to Bybit WebSocket: {}", self.ws_url);
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
impl MarketDataProvider for BybitProvider {
    async fn fetch_bars(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MarketResult<Vec<Bar>> {
        let interval = Self::timeframe_to_interval(timeframe);
        let result = self.fetch_klines(symbol, interval, start, end, 200).await?;

        result.list
            .iter()
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
        "Bybit"
    }

    fn supports_realtime(&self) -> bool {
        true
    }

    async fn supports_symbol(&self, symbol: &str) -> MarketResult<bool> {
        let url = format!("{}/v5/market/instruments-info", self.rest_url);

        let response: serde_json::Value = self.client
            .get(&url)
            .query(&[("category", "linear"), ("symbol", symbol)])
            .send()
            .await
            .map_err(|e| MarketError::NetworkError(e.to_string()))?
            .json()
            .await
            .map_err(|e| MarketError::ParseError(e.to_string()))?;

        let ret_code = response["retCode"].as_i64().unwrap_or(-1);
        Ok(ret_code == 0 && response["result"]["list"].as_array().map(|arr| !arr.is_empty()).unwrap_or(false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeframe_conversion() {
        assert_eq!(BybitProvider::timeframe_to_interval(Timeframe::Minute1), "1");
        assert_eq!(BybitProvider::timeframe_to_interval(Timeframe::Hour1), "60");
        assert_eq!(BybitProvider::timeframe_to_interval(Timeframe::Day1), "D");
    }

    #[tokio::test]
    async fn test_provider_creation() {
        let provider = BybitProvider::new();
        assert_eq!(provider.provider_name(), "Bybit");
        assert!(provider.supports_realtime());
    }

    #[tokio::test]
    async fn test_provider_with_credentials() {
        let provider = BybitProvider::with_credentials(
            "test_key".to_string(),
            "test_secret".to_string()
        );
        assert!(provider.api_key.is_some());
        assert!(provider.api_secret.is_some());
    }
}
