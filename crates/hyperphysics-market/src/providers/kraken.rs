//! Kraken cryptocurrency exchange provider
//!
//! Provides integration with Kraken exchange for cryptocurrency market data.
//!
//! # Features
//! - REST API for historical OHLC data
//! - WebSocket streams for real-time orderbook and trades
//! - Spot trading support
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

/// Kraken API endpoints
const KRAKEN_REST_URL: &str = "https://api.kraken.com";
const KRAKEN_WS_URL: &str = "wss://ws.kraken.com";

/// Kraken provider for cryptocurrency market data
pub struct KrakenProvider {
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

/// Kraken OHLC response
#[derive(Debug, Deserialize)]
struct KrakenOhlcResponse {
    error: Vec<String>,
    result: Option<HashMap<String, KrakenOhlcData>>,
}

/// Kraken OHLC data wrapper
#[derive(Debug, Deserialize)]
struct KrakenOhlcData {
    #[serde(flatten)]
    data: HashMap<String, serde_json::Value>,
}

/// Kraken OHLC entry: [time, open, high, low, close, vwap, volume, count]
type KrakenOhlc = (i64, String, String, String, String, String, String, i64);

/// WebSocket subscription message
#[derive(Debug, Serialize)]
struct WsSubscribe {
    event: String,
    pair: Vec<String>,
    subscription: WsSubscription,
}

#[derive(Debug, Serialize)]
struct WsSubscription {
    name: String,
}

impl KrakenProvider {
    /// Create new Kraken provider
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            rest_url: KRAKEN_REST_URL.to_string(),
            ws_url: KRAKEN_WS_URL.to_string(),
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

    /// Convert timeframe to Kraken interval (in minutes)
    fn timeframe_to_interval(timeframe: Timeframe) -> u32 {
        match timeframe {
            Timeframe::Minute1 => 1,
            Timeframe::Minute5 => 5,
            Timeframe::Minute15 => 15,
            Timeframe::Minute30 => 30,
            Timeframe::Hour1 => 60,
            Timeframe::Hour4 => 240,
            Timeframe::Day1 => 1440,
            Timeframe::Week1 => 10080,
            Timeframe::Month1 => 21600,
        }
    }

    /// Fetch OHLC data from REST API
    async fn fetch_ohlc(
        &self,
        symbol: &str,
        interval: u32,
        since: Option<i64>,
    ) -> MarketResult<Vec<KrakenOhlc>> {
        let url = format!("{}/0/public/OHLC", self.rest_url);

        let mut query = vec![
            ("pair", symbol.to_string()),
            ("interval", interval.to_string()),
        ];

        if let Some(since_ts) = since {
            query.push(("since", since_ts.to_string()));
        }

        let response = self.client
            .get(&url)
            .query(&query)
            .send()
            .await
            .map_err(|e| MarketError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(MarketError::ApiError(format!("Kraken API error: {}", error_text)));
        }

        let ohlc_response: KrakenOhlcResponse = response
            .json()
            .await
            .map_err(|e| MarketError::ParseError(e.to_string()))?;

        if !ohlc_response.error.is_empty() {
            return Err(MarketError::ApiError(format!("Kraken error: {:?}", ohlc_response.error)));
        }

        let result = ohlc_response.result.ok_or(MarketError::ParseError("No result data".into()))?;

        // Extract OHLC array from the result
        let ohlc_data = result.values()
            .next()
            .ok_or(MarketError::ParseError("No OHLC data".into()))?;

        let ohlc_array = ohlc_data.data.values()
            .next()
            .ok_or(MarketError::ParseError("No OHLC array".into()))?;

        let ohlc_entries: Vec<KrakenOhlc> = serde_json::from_value(ohlc_array.clone())
            .map_err(|e| MarketError::ParseError(e.to_string()))?;

        Ok(ohlc_entries)
    }

    /// Convert Kraken OHLC to Bar
    fn ohlc_to_bar(ohlc: KrakenOhlc, symbol: &str) -> MarketResult<Bar> {
        let (timestamp_secs, open_str, high_str, low_str, close_str, _vwap_str, volume_str, _count) = ohlc;

        let open: f64 = open_str.parse().map_err(|_| MarketError::ParseError("Invalid open price".into()))?;
        let high: f64 = high_str.parse().map_err(|_| MarketError::ParseError("Invalid high price".into()))?;
        let low: f64 = low_str.parse().map_err(|_| MarketError::ParseError("Invalid low price".into()))?;
        let close: f64 = close_str.parse().map_err(|_| MarketError::ParseError("Invalid close price".into()))?;
        let volume: f64 = volume_str.parse().map_err(|_| MarketError::ParseError("Invalid volume".into()))?;

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
    fn generate_signature(&self, path: &str, nonce: &str, postdata: &str) -> Option<String> {
        let api_secret = self.api_secret.as_ref()?;

        use hmac::{Hmac, Mac};
        use sha2::{Sha256, Sha512, Digest};

        let mut hasher = Sha256::new();
        hasher.update(format!("{}{}", nonce, postdata));
        let hash = hasher.finalize();

        let mut message = path.as_bytes().to_vec();
        message.extend_from_slice(&hash);

        let key = general_purpose::STANDARD.decode(api_secret).ok()?;
        let mut mac = Hmac::<Sha512>::new_from_slice(&key).ok()?;
        mac.update(&message);
        let result = mac.finalize();

        Some(general_purpose::STANDARD.encode(result.into_bytes()))
    }

    /// Subscribe to WebSocket orderbook stream
    pub async fn subscribe_orderbook(&self, symbol: &str) -> MarketResult<()> {
        let subscribe_msg = WsSubscribe {
            event: "subscribe".to_string(),
            pair: vec![symbol.to_string()],
            subscription: WsSubscription {
                name: "book".to_string(),
            },
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
        tracing::info!("Connected to Kraken WebSocket: {}", self.ws_url);
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
impl MarketDataProvider for KrakenProvider {
    async fn fetch_bars(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MarketResult<Vec<Bar>> {
        let interval = Self::timeframe_to_interval(timeframe);
        let since = Some(start.timestamp());

        let ohlc_data = self.fetch_ohlc(symbol, interval, since).await?;

        // Filter by end time
        let end_ts = end.timestamp();
        ohlc_data
            .into_iter()
            .filter(|(ts, _, _, _, _, _, _, _)| *ts <= end_ts)
            .map(|o| Self::ohlc_to_bar(o, symbol))
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
        "Kraken"
    }

    fn supports_realtime(&self) -> bool {
        true
    }

    async fn supports_symbol(&self, symbol: &str) -> MarketResult<bool> {
        let url = format!("{}/0/public/AssetPairs", self.rest_url);

        let response: serde_json::Value = self.client
            .get(&url)
            .query(&[("pair", symbol)])
            .send()
            .await
            .map_err(|e| MarketError::NetworkError(e.to_string()))?
            .json()
            .await
            .map_err(|e| MarketError::ParseError(e.to_string()))?;

        Ok(response["result"].as_object().map(|obj| !obj.is_empty()).unwrap_or(false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeframe_conversion() {
        assert_eq!(KrakenProvider::timeframe_to_interval(Timeframe::Minute1), 1);
        assert_eq!(KrakenProvider::timeframe_to_interval(Timeframe::Hour1), 60);
        assert_eq!(KrakenProvider::timeframe_to_interval(Timeframe::Day1), 1440);
    }

    #[tokio::test]
    async fn test_provider_creation() {
        let provider = KrakenProvider::new();
        assert_eq!(provider.provider_name(), "Kraken");
        assert!(provider.supports_realtime());
    }

    #[tokio::test]
    async fn test_provider_with_credentials() {
        let provider = KrakenProvider::with_credentials(
            "test_key".to_string(),
            "test_secret".to_string()
        );
        assert!(provider.api_key.is_some());
        assert!(provider.api_secret.is_some());
    }
}
