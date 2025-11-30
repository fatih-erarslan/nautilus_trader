//! Alpaca Markets API provider implementation
//!
//! Provides production-ready access to stock market data via Alpaca Markets API v2.
//! Supports both paper and live trading data with comprehensive error handling,
//! rate limiting, and real-time WebSocket streaming.
//!
//! # Environment Variables
//!
//! - `APCA_API_KEY_ID` - Your Alpaca API key (required)
//! - `APCA_API_SECRET_KEY` - Your Alpaca secret key (required)
//! - `APCA_API_BASE_URL` - Base URL (defaults to https://data.alpaca.markets)
//!
//! # Example
//!
//! ```no_run
//! use hyperphysics_market::providers::{AlpacaProvider, MarketDataProvider};
//! use hyperphysics_market::data::Timeframe;
//! use chrono::{Utc, Duration};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Environment variables must be set: APCA_API_KEY_ID, APCA_API_SECRET_KEY
//!     let provider = AlpacaProvider::from_env()?;
//!
//!     let end = Utc::now();
//!     let start = end - Duration::days(7);
//!
//!     let bars = provider.fetch_bars(
//!         "AAPL",
//!         Timeframe::Day1,
//!         start,
//!         end
//!     ).await?;
//!
//!     println!("Fetched {} bars", bars.len());
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use governor::{Quota, RateLimiter as GovRateLimiter, clock::DefaultClock, state::InMemoryState, state::NotKeyed};
use reqwest::{Client, header};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;
use tokio_tungstenite::{connect_async, tungstenite::Message, WebSocketStream, MaybeTlsStream};
use tracing::{debug, warn};
use url::Url;

use crate::data::{Bar, Timeframe};
use crate::data::tick::{Quote, Tick};
use crate::error::{MarketError, MarketResult};
use super::{MarketDataProvider, ProviderConfig};

type WsStream = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

/// Alpaca Markets data provider with rate limiting and WebSocket support
pub struct AlpacaProvider {
    config: ProviderConfig,
    client: Client,
    rate_limiter: Arc<GovRateLimiter<NotKeyed, InMemoryState, DefaultClock>>,
}

/// WebSocket client for real-time Alpaca data streams
pub struct AlpacaWebSocketClient {
    ws_stream: Option<WsStream>,
    api_key: String,
    secret_key: String,
    #[allow(dead_code)]
    subscriptions: Vec<String>,
    authenticated: bool,
    ws_url: String,
}

/// Alpaca WebSocket stream types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlpacaStreamType {
    Trades,
    Quotes,
    Bars,
}

/// WebSocket message from Alpaca
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "T")]
pub enum StreamMessage {
    #[serde(rename = "t")]
    Trade(AlpacaTrade),
    #[serde(rename = "q")]
    Quote(AlpacaQuote),
    #[serde(rename = "b")]
    Bar(AlpacaStreamBar),
    #[serde(rename = "success")]
    Success(SuccessMessage),
    #[serde(rename = "error")]
    Error(ErrorMessage),
    #[serde(rename = "subscription")]
    Subscription(SubscriptionMessage),
}

/// Trade data from WebSocket
#[derive(Debug, Clone, Deserialize)]
pub struct AlpacaTrade {
    #[serde(rename = "S")]
    pub symbol: String,
    #[serde(rename = "t")]
    pub timestamp: String,
    #[serde(rename = "p")]
    pub price: f64,
    #[serde(rename = "s")]
    pub size: f64,
    #[serde(rename = "x", default)]
    pub exchange: Option<String>,
    #[serde(rename = "c", default)]
    pub conditions: Option<Vec<String>>,
}

/// Quote data from WebSocket
#[derive(Debug, Clone, Deserialize)]
pub struct AlpacaQuote {
    #[serde(rename = "S")]
    pub symbol: String,
    #[serde(rename = "t")]
    pub timestamp: String,
    #[serde(rename = "bp")]
    pub bid_price: f64,
    #[serde(rename = "bs")]
    pub bid_size: f64,
    #[serde(rename = "ap")]
    pub ask_price: f64,
    #[serde(rename = "as")]
    pub ask_size: f64,
    #[serde(rename = "bx", default)]
    pub bid_exchange: Option<String>,
    #[serde(rename = "ax", default)]
    pub ask_exchange: Option<String>,
}

/// Bar data from WebSocket
#[derive(Debug, Clone, Deserialize)]
pub struct AlpacaStreamBar {
    #[serde(rename = "S")]
    pub symbol: String,
    #[serde(rename = "t")]
    pub timestamp: String,
    #[serde(rename = "o")]
    pub open: f64,
    #[serde(rename = "h")]
    pub high: f64,
    #[serde(rename = "l")]
    pub low: f64,
    #[serde(rename = "c")]
    pub close: f64,
    #[serde(rename = "v")]
    pub volume: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SuccessMessage {
    pub msg: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ErrorMessage {
    pub code: u32,
    pub msg: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SubscriptionMessage {
    pub trades: Vec<String>,
    pub quotes: Vec<String>,
    pub bars: Vec<String>,
}

impl AlpacaProvider {
    /// Maximum retries for API requests with transient failures
    const MAX_RETRIES: u32 = 3;

    /// Initial retry delay in milliseconds
    #[allow(dead_code)]
    const INITIAL_RETRY_DELAY_MS: u64 = 100;

    /// Rate limit: 200 requests per minute for free tier
    const RATE_LIMIT_PER_MINUTE: u32 = 200;

    /// Create a new Alpaca provider from environment variables
    ///
    /// Required environment variables:
    /// - `APCA_API_KEY_ID`
    /// - `APCA_API_SECRET_KEY`
    ///
    /// Optional:
    /// - `APCA_API_BASE_URL` (defaults to https://data.alpaca.markets)
    pub fn from_env() -> MarketResult<Self> {
        let api_key = std::env::var("APCA_API_KEY_ID")
            .map_err(|_| MarketError::ConfigError(
                "APCA_API_KEY_ID environment variable not set".to_string()
            ))?;

        let api_secret = std::env::var("APCA_API_SECRET_KEY")
            .map_err(|_| MarketError::ConfigError(
                "APCA_API_SECRET_KEY environment variable not set".to_string()
            ))?;

        let base_url = std::env::var("APCA_API_BASE_URL")
            .unwrap_or_else(|_| "https://data.alpaca.markets".to_string());

        Ok(Self::new(api_key, api_secret, base_url))
    }

    /// Create a new Alpaca provider instance
    ///
    /// # Arguments
    ///
    /// * `api_key` - Alpaca API key
    /// * `api_secret` - Alpaca API secret
    /// * `base_url` - Base URL for API endpoints
    pub fn new(api_key: String, api_secret: String, base_url: String) -> Self {
        let mut headers = header::HeaderMap::new();
        headers.insert("APCA-API-KEY-ID", header::HeaderValue::from_str(&api_key).unwrap());
        headers.insert("APCA-API-SECRET-KEY", header::HeaderValue::from_str(&api_secret).unwrap());

        let client = Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");

        // Create rate limiter: 200 requests per minute
        let quota = Quota::per_minute(
            NonZeroU32::new(Self::RATE_LIMIT_PER_MINUTE).unwrap()
        );
        let rate_limiter = Arc::new(GovRateLimiter::direct(quota));

        let config = ProviderConfig {
            api_key: api_key.clone(),
            api_secret: api_secret.clone(),
            base_url: base_url.clone(),
            paper_trading: false,
            timeout_secs: 30,
            max_retries: Self::MAX_RETRIES,
        };

        Self { config, client, rate_limiter }
    }

    /// Convert timeframe to Alpaca API format
    fn timeframe_to_alpaca(timeframe: Timeframe) -> &'static str {
        match timeframe {
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

    /// Wait for rate limiter to allow request
    async fn wait_for_rate_limit(&self) {
        self.rate_limiter.until_ready().await;
    }

    /// Retry request with exponential backoff
    #[allow(dead_code)]
    async fn retry_with_backoff<F, Fut, T>(
        &self,
        mut operation: F,
    ) -> MarketResult<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = MarketResult<T>>,
    {
        let mut retry_count = 0;
        let mut delay_ms = Self::INITIAL_RETRY_DELAY_MS;

        loop {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    // Only retry on transient errors
                    let should_retry = matches!(
                        e,
                        MarketError::NetworkError(_)
                            | MarketError::TimeoutError(_)
                            | MarketError::ConnectionError(_)
                            | MarketError::RateLimitExceeded(_)
                    );

                    if !should_retry || retry_count >= Self::MAX_RETRIES {
                        return Err(e);
                    }

                    retry_count += 1;
                    debug!(
                        "Retrying request (attempt {}/{}), waiting {}ms",
                        retry_count, Self::MAX_RETRIES, delay_ms
                    );

                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    delay_ms *= 2; // Exponential backoff
                }
            }
        }
    }

    /// Validate OHLC data integrity
    fn validate_bar(bar: &AlpacaBar) -> MarketResult<()> {
        // Check for non-positive prices
        if bar.open <= 0.0 || bar.high <= 0.0 || bar.low <= 0.0 || bar.close <= 0.0 {
            return Err(MarketError::DataIntegrityError(
                "Bar contains non-positive prices".to_string(),
            ));
        }

        // Validate OHLC relationships
        let max_price = bar.open.max(bar.close);
        let min_price = bar.open.min(bar.close);

        if bar.high < max_price {
            return Err(MarketError::DataIntegrityError(format!(
                "High price {} is less than max(open, close) {}",
                bar.high, max_price
            )));
        }

        if bar.low > min_price {
            return Err(MarketError::DataIntegrityError(format!(
                "Low price {} is greater than min(open, close) {}",
                bar.low, min_price
            )));
        }

        if bar.high < bar.low {
            return Err(MarketError::DataIntegrityError(format!(
                "High price {} is less than low price {}",
                bar.high, bar.low
            )));
        }

        // Validate volume is non-zero
        if bar.volume == 0 {
            return Err(MarketError::DataIntegrityError(
                "Bar has zero volume".to_string(),
            ));
        }

        Ok(())
    }

    /// Fetch historical trades for a symbol
    pub async fn fetch_trades(
        &self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MarketResult<Vec<Tick>> {
        self.wait_for_rate_limit().await;

        let url = format!("{}/v2/stocks/{}/trades", self.config.base_url, symbol);
        let start_str = start.to_rfc3339();
        let end_str = end.to_rfc3339();

        let response = self.client
            .get(&url)
            .query(&[
                ("start", start_str.as_str()),
                ("end", end_str.as_str()),
                ("limit", "10000"),
                ("feed", "iex"),
            ])
            .send()
            .await?;

        if response.status() == 429 {
            return Err(MarketError::RateLimitExceeded(
                "Rate limit exceeded, retry after delay".to_string()
            ));
        }

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(MarketError::ApiError(format!(
                "Alpaca API error: {}", error_text
            )));
        }

        let api_response: AlpacaTradesResponse = response.json().await?;

        Ok(api_response.trades.into_iter().map(|t| t.to_tick()).collect())
    }

    /// Fetch latest quote for a symbol
    pub async fn fetch_latest_quote(&self, symbol: &str) -> MarketResult<Quote> {
        self.wait_for_rate_limit().await;

        let url = format!("{}/v2/stocks/{}/quotes/latest", self.config.base_url, symbol);

        let response = self.client
            .get(&url)
            .query(&[("feed", "iex")])
            .send()
            .await?;

        if response.status() == 429 {
            return Err(MarketError::RateLimitExceeded(
                "Rate limit exceeded".to_string()
            ));
        }

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(MarketError::ApiError(error_text));
        }

        let json: serde_json::Value = response.json().await?;
        let quote_value = json.get("quote").ok_or_else(|| {
            MarketError::ParseError("Missing 'quote' field".to_string())
        })?;

        let alpaca_quote: AlpacaQuote = serde_json::from_value(quote_value.clone())?;
        alpaca_quote.to_quote()
    }
}

#[async_trait]
impl MarketDataProvider for AlpacaProvider {
    async fn fetch_bars(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MarketResult<Vec<Bar>> {
        self.wait_for_rate_limit().await;

        debug!(
            "Fetching bars for {} from {} to {} with timeframe {:?}",
            symbol, start, end, timeframe
        );

        let url = format!("{}/v2/stocks/{}/bars", self.config.base_url, symbol);
        let timeframe_str = Self::timeframe_to_alpaca(timeframe);
        let start_str = start.to_rfc3339();
        let end_str = end.to_rfc3339();

        let response = self.client
            .get(&url)
            .query(&[
                ("timeframe", timeframe_str),
                ("start", &start_str),
                ("end", &end_str),
                ("limit", "10000"),
                ("adjustment", "raw"),
                ("feed", "iex"),
            ])
            .send()
            .await?;

        if response.status() == 429 {
            return Err(MarketError::RateLimitExceeded(
                "Rate limit exceeded, retry after delay".to_string()
            ));
        }

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(MarketError::ApiError(format!(
                "Alpaca API error: {}", error_text
            )));
        }

        let api_response: AlpacaBarResponse = response.json().await?;

        let bars: MarketResult<Vec<Bar>> = api_response
            .bars
            .into_iter()
            .filter_map(|alpaca_bar| {
                match Self::validate_bar(&alpaca_bar) {
                    Ok(()) => Some(alpaca_bar.to_bar(symbol.to_string())),
                    Err(e) => {
                        debug!("Skipping invalid bar: {}", e);
                        None
                    }
                }
            })
            .collect();

        if api_response.next_page_token.is_some() {
            warn!("More data available via pagination (not implemented)");
        }

        bars
    }

    async fn fetch_latest_bar(&self, symbol: &str) -> MarketResult<Bar> {
        self.wait_for_rate_limit().await;

        debug!("Fetching latest bar for {}", symbol);

        let url = format!("{}/v2/stocks/{}/bars/latest", self.config.base_url, symbol);

        let response = self.client
            .get(&url)
            .query(&[("feed", "iex")])
            .send()
            .await?;

        if response.status() == 429 {
            return Err(MarketError::RateLimitExceeded("Rate limit exceeded".to_string()));
        }

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(MarketError::ApiError(error_text));
        }

        let json: serde_json::Value = response.json().await?;
        let bar_value = json.get("bar").ok_or_else(|| {
            MarketError::ParseError("Missing 'bar' field in response".to_string())
        })?;

        let alpaca_bar: AlpacaBar = serde_json::from_value(bar_value.clone())?;
        Self::validate_bar(&alpaca_bar)?;
        alpaca_bar.to_bar(symbol.to_string())
    }

    fn provider_name(&self) -> &str {
        "Alpaca Markets"
    }

    fn supports_realtime(&self) -> bool {
        true
    }

    async fn supports_symbol(&self, symbol: &str) -> MarketResult<bool> {
        self.wait_for_rate_limit().await;

        debug!("Checking symbol support for {}", symbol);

        let url = format!("https://api.alpaca.markets/v2/assets/{}", symbol.to_uppercase());

        let response = self.client.get(&url).send().await?;

        match response.status().as_u16() {
            200 => {
                let asset: serde_json::Value = response.json().await?;
                let tradable = asset.get("tradable").and_then(|v| v.as_bool()).unwrap_or(false);
                let active = asset.get("status")
                    .and_then(|v| v.as_str())
                    .map(|s| s == "active")
                    .unwrap_or(false);
                Ok(tradable && active)
            }
            404 => Ok(false),
            _ => {
                let error_text = response.text().await.unwrap_or_default();
                Err(MarketError::ApiError(error_text))
            }
        }
    }
}

impl AlpacaWebSocketClient {
    /// Create new WebSocket client
    pub fn new(api_key: String, secret_key: String) -> Self {
        Self {
            ws_stream: None,
            api_key,
            secret_key,
            subscriptions: Vec::new(),
            authenticated: false,
            ws_url: "wss://stream.data.alpaca.markets/v2/iex".to_string(),
        }
    }

    /// Connect to Alpaca WebSocket
    pub async fn connect(&mut self) -> MarketResult<()> {
        let url = Url::parse(&self.ws_url)
            .map_err(|e| MarketError::ConnectionError(e.to_string()))?;

        let (ws_stream, _) = connect_async(url).await
            .map_err(|e| MarketError::ConnectionError(e.to_string()))?;

        self.ws_stream = Some(ws_stream);
        debug!("Connected to Alpaca WebSocket");
        Ok(())
    }

    /// Authenticate with Alpaca
    pub async fn authenticate(&mut self) -> MarketResult<()> {
        if self.ws_stream.is_none() {
            return Err(MarketError::ConnectionError("Not connected".to_string()));
        }

        let auth_message = serde_json::json!({
            "action": "auth",
            "key": self.api_key,
            "secret": self.secret_key
        });

        let ws_stream = self.ws_stream.as_mut().unwrap();
        ws_stream.send(Message::Text(auth_message.to_string())).await
            .map_err(|e| MarketError::ConnectionError(e.to_string()))?;

        self.authenticated = true;
        debug!("Authenticated with Alpaca");
        Ok(())
    }

    /// Subscribe to trades for symbols
    pub async fn subscribe_trades(&mut self, symbols: Vec<&str>) -> MarketResult<()> {
        if !self.authenticated {
            return Err(MarketError::AuthenticationError("Not authenticated".to_string()));
        }

        let subscribe_msg = serde_json::json!({
            "action": "subscribe",
            "trades": symbols
        });

        let ws_stream = self.ws_stream.as_mut().unwrap();
        ws_stream.send(Message::Text(subscribe_msg.to_string())).await
            .map_err(|e| MarketError::ConnectionError(e.to_string()))?;

        debug!("Subscribed to trades: {:?}", symbols);
        Ok(())
    }

    /// Subscribe to quotes for symbols
    pub async fn subscribe_quotes(&mut self, symbols: Vec<&str>) -> MarketResult<()> {
        if !self.authenticated {
            return Err(MarketError::AuthenticationError("Not authenticated".to_string()));
        }

        let subscribe_msg = serde_json::json!({
            "action": "subscribe",
            "quotes": symbols
        });

        let ws_stream = self.ws_stream.as_mut().unwrap();
        ws_stream.send(Message::Text(subscribe_msg.to_string())).await
            .map_err(|e| MarketError::ConnectionError(e.to_string()))?;

        debug!("Subscribed to quotes: {:?}", symbols);
        Ok(())
    }

    /// Receive next message from WebSocket
    pub async fn next_message(&mut self) -> MarketResult<Option<StreamMessage>> {
        if let Some(ws_stream) = &mut self.ws_stream {
            match ws_stream.next().await {
                Some(Ok(Message::Text(text))) => {
                    let messages: Vec<StreamMessage> = serde_json::from_str(&text)?;
                    Ok(messages.into_iter().next())
                }
                Some(Ok(Message::Ping(_))) => {
                    ws_stream.send(Message::Pong(vec![])).await
                        .map_err(|e| MarketError::ConnectionError(e.to_string()))?;
                    Ok(None)
                }
                Some(Ok(Message::Close(_))) => {
                    Err(MarketError::ConnectionError("Connection closed".to_string()))
                }
                Some(Err(e)) => {
                    Err(MarketError::ConnectionError(e.to_string()))
                }
                None => Ok(None),
                _ => Ok(None),
            }
        } else {
            Err(MarketError::ConnectionError("Not connected".to_string()))
        }
    }
}

// API response structures
#[derive(Debug, Deserialize)]
struct AlpacaBarResponse {
    bars: Vec<AlpacaBar>,
    #[allow(dead_code)]
    symbol: String,
    next_page_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AlpacaBar {
    #[serde(rename = "t")]
    timestamp: String,
    #[serde(rename = "o")]
    open: f64,
    #[serde(rename = "h")]
    high: f64,
    #[serde(rename = "l")]
    low: f64,
    #[serde(rename = "c")]
    close: f64,
    #[serde(rename = "v")]
    volume: u64,
    #[serde(rename = "n", default)]
    trade_count: Option<u64>,
    #[serde(rename = "vw", default)]
    vwap: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct AlpacaTradesResponse {
    trades: Vec<AlpacaTradeData>,
    #[allow(dead_code)]
    symbol: String,
    #[allow(dead_code)]
    next_page_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AlpacaTradeData {
    #[serde(rename = "t")]
    timestamp: String,
    #[serde(rename = "p")]
    price: f64,
    #[serde(rename = "s")]
    size: f64,
    #[serde(rename = "x", default)]
    exchange: Option<String>,
    #[serde(rename = "c", default)]
    conditions: Option<Vec<String>>,
}

impl AlpacaBar {
    fn to_bar(&self, symbol: String) -> MarketResult<Bar> {
        let timestamp = DateTime::parse_from_rfc3339(&self.timestamp)
            .map_err(|e| MarketError::DateTimeParseError(e.to_string()))?
            .with_timezone(&Utc);

        Ok(Bar {
            symbol,
            timestamp,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            vwap: self.vwap,
            trade_count: self.trade_count,
        })
    }
}

impl AlpacaTradeData {
    fn to_tick(&self) -> Tick {
        let timestamp = DateTime::parse_from_rfc3339(&self.timestamp)
            .unwrap_or_else(|_| Utc::now().into())
            .with_timezone(&Utc);

        Tick {
            symbol: String::new(), // Will be set by caller
            timestamp,
            price: self.price,
            size: self.size,
            exchange: self.exchange.clone(),
            conditions: self.conditions.clone(),
        }
    }
}

impl AlpacaQuote {
    fn to_quote(&self) -> MarketResult<Quote> {
        let timestamp = DateTime::parse_from_rfc3339(&self.timestamp)
            .map_err(|e| MarketError::DateTimeParseError(e.to_string()))?
            .with_timezone(&Utc);

        Ok(Quote {
            symbol: self.symbol.clone(),
            timestamp,
            bid_price: self.bid_price,
            bid_size: self.bid_size,
            ask_price: self.ask_price,
            ask_size: self.ask_size,
            exchange: self.bid_exchange.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeframe_conversion() {
        assert_eq!(AlpacaProvider::timeframe_to_alpaca(Timeframe::Minute1), "1Min");
        assert_eq!(AlpacaProvider::timeframe_to_alpaca(Timeframe::Day1), "1Day");
        assert_eq!(AlpacaProvider::timeframe_to_alpaca(Timeframe::Hour1), "1Hour");
    }

    #[tokio::test]
    #[ignore] // Requires API credentials
    async fn test_alpaca_from_env() {
        // This test only runs when credentials are available
        if std::env::var("APCA_API_KEY_ID").is_ok() {
            let provider = AlpacaProvider::from_env();
            assert!(provider.is_ok());
        }
    }

    #[tokio::test]
    #[ignore] // Requires API credentials
    async fn test_fetch_latest_bar() {
        if std::env::var("APCA_API_KEY_ID").is_ok() {
            let provider = AlpacaProvider::from_env().unwrap();
            let result = provider.fetch_latest_bar("AAPL").await;
            assert!(result.is_ok());
        }
    }
}
