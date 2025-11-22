//! Coinbase Pro cryptocurrency exchange provider
//!
//! Provides integration with Coinbase Pro exchange for cryptocurrency market data.
//!
//! # Features
//! - REST API for historical OHLCV data
//! - WebSocket streams for real-time orderbook, trades, and candles
//! - Professional trading features
//! - Multi-symbol subscriptions

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose};
use chrono::{DateTime, Utc, Duration};
use futures_util::SinkExt;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::net::TcpStream;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream, tungstenite::Message};
use std::collections::HashMap;

use crate::data::{Bar, Timeframe};
use crate::data::orderbook::OrderBook;
use crate::error::{MarketError, MarketResult};
use super::MarketDataProvider;

/// Coinbase Pro API endpoints
const COINBASE_REST_URL: &str = "https://api.exchange.coinbase.com";
const COINBASE_WS_URL: &str = "wss://ws-feed.exchange.coinbase.com";

/// Coinbase Pro provider for cryptocurrency market data
pub struct CoinbaseProvider {
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

    /// API passphrase (required for Coinbase Pro)
    api_passphrase: Option<String>,

    /// Active WebSocket connection
    ws_connection: Arc<RwLock<Option<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,

    /// Real-time orderbooks cache
    orderbooks: Arc<RwLock<HashMap<String, OrderBook>>>,
}

/// Coinbase Pro candle response format: [timestamp, low, high, open, close, volume]
type CoinbaseCandle = (i64, f64, f64, f64, f64, f64);

/// WebSocket subscription message
#[derive(Debug, Serialize)]
struct WsSubscribe {
    #[serde(rename = "type")]
    msg_type: String,
    product_ids: Vec<String>,
    channels: Vec<String>,
}

impl CoinbaseProvider {
    /// Create new Coinbase Pro provider
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            rest_url: COINBASE_REST_URL.to_string(),
            ws_url: COINBASE_WS_URL.to_string(),
            api_key: None,
            api_secret: None,
            api_passphrase: None,
            ws_connection: Arc::new(RwLock::new(None)),
            orderbooks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create provider with API credentials
    pub fn with_credentials(api_key: String, api_secret: String, api_passphrase: String) -> Self {
        let mut provider = Self::new();
        provider.api_key = Some(api_key);
        provider.api_secret = Some(api_secret);
        provider.api_passphrase = Some(api_passphrase);
        provider
    }

    /// Convert timeframe to Coinbase granularity (in seconds)
    fn timeframe_to_granularity(timeframe: Timeframe) -> u32 {
        match timeframe {
            Timeframe::Minute1 => 60,
            Timeframe::Minute5 => 300,
            Timeframe::Minute15 => 900,
            Timeframe::Minute30 => 1800,
            Timeframe::Hour1 => 3600,
            Timeframe::Hour4 => 14400,
            Timeframe::Day1 => 86400,
            Timeframe::Week1 => 604800,
            Timeframe::Month1 => 2592000,
        }
    }

    /// Fetch candles from REST API
    async fn fetch_candles(
        &self,
        symbol: &str,
        granularity: u32,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MarketResult<Vec<CoinbaseCandle>> {
        let url = format!("{}/products/{}/candles", self.rest_url, symbol);

        let response = self.client
            .get(&url)
            .query(&[
                ("granularity", granularity.to_string()),
                ("start", start.to_rfc3339()),
                ("end", end.to_rfc3339()),
            ])
            .send()
            .await
            .map_err(|e| MarketError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(MarketError::ApiError(format!("Coinbase API error: {}", error_text)));
        }

        let candles: Vec<CoinbaseCandle> = response
            .json()
            .await
            .map_err(|e| MarketError::ParseError(e.to_string()))?;

        Ok(candles)
    }

    /// Convert Coinbase candle to Bar
    fn candle_to_bar(candle: CoinbaseCandle, symbol: &str) -> MarketResult<Bar> {
        let (timestamp_secs, low, high, open, close, volume) = candle;

        let timestamp = DateTime::from_timestamp(timestamp_secs, 0)
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
    #[allow(dead_code)]
    fn generate_signature(&self, timestamp: &str, method: &str, path: &str, body: &str) -> Option<String> {
        let api_secret = self.api_secret.as_ref()?;
        let message = format!("{}{}{}{}", timestamp, method, path, body);

        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        let key = general_purpose::STANDARD.decode(api_secret).ok()?;
        let mut mac = Hmac::<Sha256>::new_from_slice(&key).ok()?;
        mac.update(message.as_bytes());
        let result = mac.finalize();

        Some(general_purpose::STANDARD.encode(result.into_bytes()))
    }

    /// Subscribe to WebSocket level2 (orderbook) stream
    pub async fn subscribe_orderbook(&self, symbol: &str) -> MarketResult<()> {
        let subscribe_msg = WsSubscribe {
            msg_type: "subscribe".to_string(),
            product_ids: vec![symbol.to_string()],
            channels: vec!["level2".to_string()],
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
        tracing::info!("Connected to Coinbase Pro WebSocket: {}", self.ws_url);
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
impl MarketDataProvider for CoinbaseProvider {
    async fn fetch_bars(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MarketResult<Vec<Bar>> {
        let granularity = Self::timeframe_to_granularity(timeframe);
        let candles = self.fetch_candles(symbol, granularity, start, end).await?;

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
        "Coinbase Pro"
    }

    fn supports_realtime(&self) -> bool {
        true
    }

    async fn supports_symbol(&self, symbol: &str) -> MarketResult<bool> {
        let url = format!("{}/products/{}", self.rest_url, symbol);

        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| MarketError::NetworkError(e.to_string()))?;

        Ok(response.status().is_success())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeframe_conversion() {
        assert_eq!(CoinbaseProvider::timeframe_to_granularity(Timeframe::Minute1), 60);
        assert_eq!(CoinbaseProvider::timeframe_to_granularity(Timeframe::Hour1), 3600);
        assert_eq!(CoinbaseProvider::timeframe_to_granularity(Timeframe::Day1), 86400);
    }

    #[tokio::test]
    async fn test_provider_creation() {
        let provider = CoinbaseProvider::new();
        assert_eq!(provider.provider_name(), "Coinbase Pro");
        assert!(provider.supports_realtime());
    }

    #[tokio::test]
    async fn test_provider_with_credentials() {
        let provider = CoinbaseProvider::with_credentials(
            "test_key".to_string(),
            "test_secret".to_string(),
            "test_passphrase".to_string()
        );
        assert!(provider.api_key.is_some());
        assert!(provider.api_secret.is_some());
        assert!(provider.api_passphrase.is_some());
    }
}
