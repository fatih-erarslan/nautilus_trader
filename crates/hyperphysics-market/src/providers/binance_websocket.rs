//! Production-grade Binance WebSocket client
//!
//! This module provides a robust, production-ready WebSocket client for Binance
//! with automatic reconnection, rate limiting, and comprehensive error handling.
//!
//! # Features
//! - Real-time trade, kline, and depth streams
//! - Automatic reconnection with exponential backoff
//! - Rate limit handling (10 requests/second)
//! - Circuit breaker pattern for network failures
//! - Thread-safe concurrent message processing
//! - Comprehensive error recovery
//!
//! # Example
//! ```no_run
//! use hyperphysics_market::providers::BinanceWebSocketClient;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut client = BinanceWebSocketClient::new(false)?;
//!     client.connect().await?;
//!     client.subscribe_trades("btcusdt").await?;
//!
//!     // Process messages in real-time
//!     while let Some(msg) = client.next_message().await? {
//!         println!("Received: {:?}", msg);
//!     }
//!
//!     Ok(())
//! }
//! ```

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration as StdDuration, Instant};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, RwLock, Mutex};
use tokio::time::sleep;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream, tungstenite::Message};

use crate::error::{MarketError, MarketResult};

/// Binance WebSocket endpoints
const BINANCE_WS_URL: &str = "wss://stream.binance.com:9443/ws";
const BINANCE_TESTNET_WS_URL: &str = "wss://testnet.binance.vision/ws";

/// Rate limit configuration (Binance allows 10 requests/second)
const RATE_LIMIT_PER_SECOND: u32 = 10;
const RATE_LIMIT_WINDOW: StdDuration = StdDuration::from_secs(1);

/// Reconnection configuration
const MAX_RECONNECT_ATTEMPTS: u32 = 10;
const INITIAL_BACKOFF_MS: u64 = 1000;
const MAX_BACKOFF_MS: u64 = 60000;
const BACKOFF_MULTIPLIER: f64 = 2.0;

/// Connection timeout
const CONNECTION_TIMEOUT: StdDuration = StdDuration::from_secs(10);
const MESSAGE_TIMEOUT: StdDuration = StdDuration::from_secs(30);

/// Circuit breaker configuration
const CIRCUIT_BREAKER_THRESHOLD: u32 = 5;
const CIRCUIT_BREAKER_TIMEOUT: StdDuration = StdDuration::from_secs(60);

/// WebSocket message types from Binance
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "e", rename_all = "camelCase")]
pub enum BinanceStreamMessage {
    /// Trade event
    #[serde(rename = "trade")]
    Trade(TradeEvent),

    /// Kline/candlestick event
    #[serde(rename = "kline")]
    Kline(KlineEvent),

    /// Depth update event
    #[serde(rename = "depthUpdate")]
    DepthUpdate(DepthUpdateEvent),

    /// 24hr ticker
    #[serde(rename = "24hrTicker")]
    Ticker24hr(Ticker24hrEvent),
}

/// Trade event from WebSocket stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeEvent {
    #[serde(rename = "E")]
    pub event_time: i64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "t")]
    pub trade_id: i64,
    #[serde(rename = "p")]
    pub price: String,
    #[serde(rename = "q")]
    pub quantity: String,
    #[serde(rename = "T")]
    pub trade_time: i64,
    #[serde(rename = "m")]
    pub is_buyer_maker: bool,
}

/// Kline event from WebSocket stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlineEvent {
    #[serde(rename = "E")]
    pub event_time: i64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "k")]
    pub kline: KlineData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlineData {
    #[serde(rename = "t")]
    pub start_time: i64,
    #[serde(rename = "T")]
    pub close_time: i64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "i")]
    pub interval: String,
    #[serde(rename = "o")]
    pub open: String,
    #[serde(rename = "h")]
    pub high: String,
    #[serde(rename = "l")]
    pub low: String,
    #[serde(rename = "c")]
    pub close: String,
    #[serde(rename = "v")]
    pub volume: String,
    #[serde(rename = "n")]
    pub num_trades: i64,
    #[serde(rename = "x")]
    pub is_closed: bool,
}

/// Depth update event from WebSocket stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthUpdateEvent {
    #[serde(rename = "E")]
    pub event_time: i64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "U")]
    pub first_update_id: i64,
    #[serde(rename = "u")]
    pub final_update_id: i64,
    #[serde(rename = "b")]
    pub bids: Vec<(String, String)>,
    #[serde(rename = "a")]
    pub asks: Vec<(String, String)>,
}

/// 24hr ticker event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker24hrEvent {
    #[serde(rename = "E")]
    pub event_time: i64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "p")]
    pub price_change: String,
    #[serde(rename = "P")]
    pub price_change_percent: String,
    #[serde(rename = "c")]
    pub last_price: String,
    #[serde(rename = "v")]
    pub volume: String,
}

/// WebSocket subscription request
#[derive(Debug, Serialize)]
struct SubscribeRequest {
    method: String,
    params: Vec<String>,
    id: u64,
}

/// Circuit breaker state for connection management
#[derive(Debug, Clone, Copy, PartialEq)]
enum CircuitState {
    Closed,  // Normal operation
    Open,    // Failures detected, blocking requests
    HalfOpen, // Testing if service recovered
}

/// Circuit breaker for connection failures
#[derive(Debug)]
struct CircuitBreaker {
    state: CircuitState,
    failure_count: u32,
    last_failure_time: Option<Instant>,
}

impl CircuitBreaker {
    fn new() -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            last_failure_time: None,
        }
    }

    fn record_success(&mut self) {
        self.state = CircuitState::Closed;
        self.failure_count = 0;
        self.last_failure_time = None;
    }

    fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Instant::now());

        if self.failure_count >= CIRCUIT_BREAKER_THRESHOLD {
            self.state = CircuitState::Open;
        }
    }

    fn can_attempt(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => true,
            CircuitState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() > CIRCUIT_BREAKER_TIMEOUT {
                        self.state = CircuitState::HalfOpen;
                        true
                    } else {
                        false
                    }
                } else {
                    self.state = CircuitState::Closed;
                    true
                }
            }
        }
    }
}

/// Rate limiter for WebSocket requests
#[derive(Debug)]
struct RateLimiter {
    requests: Vec<Instant>,
    max_requests: u32,
    window: StdDuration,
}

impl RateLimiter {
    fn new(max_requests: u32, window: StdDuration) -> Self {
        Self {
            requests: Vec::new(),
            max_requests,
            window,
        }
    }

    fn cleanup_old_requests(&mut self) {
        let cutoff = Instant::now() - self.window;
        self.requests.retain(|&time| time > cutoff);
    }

    async fn wait_if_needed(&mut self) {
        self.cleanup_old_requests();

        if self.requests.len() >= self.max_requests as usize {
            if let Some(&oldest) = self.requests.first() {
                let wait_time = self.window.saturating_sub(oldest.elapsed());
                if !wait_time.is_zero() {
                    tracing::debug!("Rate limit reached, waiting {:?}", wait_time);
                    sleep(wait_time).await;
                    self.cleanup_old_requests();
                }
            }
        }

        self.requests.push(Instant::now());
    }
}

/// Production-grade Binance WebSocket client
pub struct BinanceWebSocketClient {
    /// WebSocket connection
    ws_stream: Arc<RwLock<Option<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,

    /// Active subscriptions
    subscriptions: Arc<RwLock<Vec<String>>>,

    /// Reconnection attempt counter
    reconnect_attempts: Arc<Mutex<u32>>,

    /// Last received message timestamp
    last_message_time: Arc<Mutex<Instant>>,

    /// WebSocket URL
    ws_url: String,

    /// Whether using testnet
    testnet: bool,

    /// Circuit breaker for connection failures
    circuit_breaker: Arc<Mutex<CircuitBreaker>>,

    /// Rate limiter
    rate_limiter: Arc<Mutex<RateLimiter>>,

    /// Message channel receiver
    message_rx: Arc<Mutex<Option<mpsc::UnboundedReceiver<BinanceStreamMessage>>>>,

    /// Shutdown signal sender
    shutdown_tx: Arc<Mutex<Option<mpsc::Sender<()>>>>,
}

impl BinanceWebSocketClient {
    /// Create a new Binance WebSocket client
    ///
    /// # Arguments
    /// * `testnet` - Whether to use testnet endpoints
    ///
    /// # Returns
    /// New WebSocket client instance
    pub fn new(testnet: bool) -> MarketResult<Self> {
        let ws_url = if testnet {
            BINANCE_TESTNET_WS_URL.to_string()
        } else {
            BINANCE_WS_URL.to_string()
        };

        Ok(Self {
            ws_stream: Arc::new(RwLock::new(None)),
            subscriptions: Arc::new(RwLock::new(Vec::new())),
            reconnect_attempts: Arc::new(Mutex::new(0)),
            last_message_time: Arc::new(Mutex::new(Instant::now())),
            ws_url,
            testnet,
            circuit_breaker: Arc::new(Mutex::new(CircuitBreaker::new())),
            rate_limiter: Arc::new(Mutex::new(RateLimiter::new(
                RATE_LIMIT_PER_SECOND,
                RATE_LIMIT_WINDOW,
            ))),
            message_rx: Arc::new(Mutex::new(None)),
            shutdown_tx: Arc::new(Mutex::new(None)),
        })
    }

    /// Connect to Binance WebSocket
    ///
    /// Establishes WebSocket connection with timeout and circuit breaker protection.
    pub async fn connect(&mut self) -> MarketResult<()> {
        let mut circuit = self.circuit_breaker.lock().await;

        if !circuit.can_attempt() {
            return Err(MarketError::ConnectionError(
                "Circuit breaker open - too many failures".into()
            ));
        }
        drop(circuit);

        tracing::info!("Connecting to Binance WebSocket: {}", self.ws_url);

        let connect_result = tokio::time::timeout(
            CONNECTION_TIMEOUT,
            connect_async(&self.ws_url)
        ).await;

        match connect_result {
            Ok(Ok((ws_stream, _))) => {
                *self.ws_stream.write().await = Some(ws_stream);
                *self.last_message_time.lock().await = Instant::now();
                *self.reconnect_attempts.lock().await = 0;

                {
                    let mut circuit = self.circuit_breaker.lock().await;
                    circuit.record_success();
                }

                tracing::info!("Successfully connected to Binance WebSocket");

                // Start message processing task
                self.start_message_processor().await;

                Ok(())
            }
            Ok(Err(e)) => {
                let mut circuit = self.circuit_breaker.lock().await;
                circuit.record_failure();
                Err(MarketError::ConnectionError(format!("WebSocket connection failed: {}", e)))
            }
            Err(_) => {
                let mut circuit = self.circuit_breaker.lock().await;
                circuit.record_failure();
                Err(MarketError::TimeoutError("Connection timeout".into()))
            }
        }
    }

    /// Start background message processing task
    async fn start_message_processor(&mut self) {
        let (tx, rx) = mpsc::unbounded_channel();
        *self.message_rx.lock().await = Some(rx);

        let (shutdown_tx, mut shutdown_rx) = mpsc::channel(1);
        *self.shutdown_tx.lock().await = Some(shutdown_tx);

        let ws_stream = Arc::clone(&self.ws_stream);
        let last_message_time = Arc::clone(&self.last_message_time);

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        tracing::info!("Shutting down message processor");
                        break;
                    }
                    _ = sleep(StdDuration::from_millis(100)) => {
                        let mut stream = ws_stream.write().await;
                        if let Some(ws) = stream.as_mut() {
                            match ws.next().await {
                                Some(Ok(Message::Text(text))) => {
                                    *last_message_time.lock().await = Instant::now();

                                    match serde_json::from_str::<BinanceStreamMessage>(&text) {
                                        Ok(msg) => {
                                            if let Err(e) = tx.send(msg) {
                                                tracing::error!("Failed to send message: {}", e);
                                                break;
                                            }
                                        }
                                        Err(e) => {
                                            tracing::warn!("Failed to parse message: {} - {}", e, text);
                                        }
                                    }
                                }
                                Some(Ok(Message::Ping(data))) => {
                                    if let Err(e) = ws.send(Message::Pong(data)).await {
                                        tracing::error!("Failed to send pong: {}", e);
                                        break;
                                    }
                                }
                                Some(Ok(Message::Close(_))) => {
                                    tracing::warn!("WebSocket closed by server");
                                    break;
                                }
                                Some(Err(e)) => {
                                    tracing::error!("WebSocket error: {}", e);
                                    break;
                                }
                                None => {
                                    tracing::warn!("WebSocket stream ended");
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        });
    }

    /// Reconnect with exponential backoff
    ///
    /// Implements exponential backoff strategy for reconnection attempts.
    pub async fn reconnect(&mut self) -> MarketResult<()> {
        let mut attempts = self.reconnect_attempts.lock().await;

        if *attempts >= MAX_RECONNECT_ATTEMPTS {
            *attempts = 0;
            return Err(MarketError::ConnectionError(
                "Max reconnection attempts exceeded".into()
            ));
        }

        *attempts += 1;
        let backoff_ms = (INITIAL_BACKOFF_MS as f64
            * BACKOFF_MULTIPLIER.powi((*attempts - 1) as i32)) as u64;
        let backoff_ms = backoff_ms.min(MAX_BACKOFF_MS);

        tracing::info!("Reconnection attempt {} after {}ms", *attempts, backoff_ms);
        sleep(StdDuration::from_millis(backoff_ms)).await;

        drop(attempts);

        // Attempt reconnection
        self.connect().await?;

        // Resubscribe to all streams
        let subscriptions = self.subscriptions.read().await.clone();
        for stream in subscriptions {
            self.send_subscribe(&[stream]).await?;
        }

        Ok(())
    }

    /// Send subscription request
    async fn send_subscribe(&self, streams: &[String]) -> MarketResult<()> {
        let mut rate_limiter = self.rate_limiter.lock().await;
        rate_limiter.wait_if_needed().await;
        drop(rate_limiter);

        let subscribe = SubscribeRequest {
            method: "SUBSCRIBE".to_string(),
            params: streams.to_vec(),
            id: chrono::Utc::now().timestamp_millis() as u64,
        };

        let msg = serde_json::to_string(&subscribe)
            .map_err(|e| MarketError::ParseError(e.to_string()))?;

        let mut ws = self.ws_stream.write().await;
        if let Some(stream) = ws.as_mut() {
            stream.send(Message::Text(msg)).await
                .map_err(|e| MarketError::NetworkError(e.to_string()))?;
            Ok(())
        } else {
            Err(MarketError::ConnectionError("Not connected".into()))
        }
    }

    /// Subscribe to trade stream
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "btcusdt")
    pub async fn subscribe_trades(&self, symbol: &str) -> MarketResult<()> {
        let stream = format!("{}@trade", symbol.to_lowercase());
        self.send_subscribe(&[stream.clone()]).await?;

        let mut subs = self.subscriptions.write().await;
        subs.push(stream);

        tracing::info!("Subscribed to trades for {}", symbol);
        Ok(())
    }

    /// Subscribe to kline/candlestick stream
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol
    /// * `interval` - Kline interval (e.g., "1m", "5m", "1h")
    pub async fn subscribe_klines(&self, symbol: &str, interval: &str) -> MarketResult<()> {
        let stream = format!("{}@kline_{}", symbol.to_lowercase(), interval);
        self.send_subscribe(&[stream.clone()]).await?;

        let mut subs = self.subscriptions.write().await;
        subs.push(stream);

        tracing::info!("Subscribed to klines for {} at {}", symbol, interval);
        Ok(())
    }

    /// Subscribe to depth/orderbook stream
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol
    pub async fn subscribe_depth(&self, symbol: &str) -> MarketResult<()> {
        let stream = format!("{}@depth", symbol.to_lowercase());
        self.send_subscribe(&[stream.clone()]).await?;

        let mut subs = self.subscriptions.write().await;
        subs.push(stream);

        tracing::info!("Subscribed to depth for {}", symbol);
        Ok(())
    }

    /// Get next message from the stream
    ///
    /// # Returns
    /// Next message or None if stream closed
    pub async fn next_message(&self) -> MarketResult<Option<BinanceStreamMessage>> {
        let mut rx = self.message_rx.lock().await;
        if let Some(receiver) = rx.as_mut() {
            match tokio::time::timeout(MESSAGE_TIMEOUT, receiver.recv()).await {
                Ok(Some(msg)) => Ok(Some(msg)),
                Ok(None) => Ok(None),
                Err(_) => {
                    // Check if connection is still alive
                    let last_msg = *self.last_message_time.lock().await;
                    if last_msg.elapsed() > MESSAGE_TIMEOUT {
                        Err(MarketError::TimeoutError("No messages received".into()))
                    } else {
                        Ok(None)
                    }
                }
            }
        } else {
            Err(MarketError::ConnectionError("Message receiver not initialized".into()))
        }
    }

    /// Check if connected
    pub async fn is_connected(&self) -> bool {
        self.ws_stream.read().await.is_some()
    }

    /// Disconnect from WebSocket
    pub async fn disconnect(&self) -> MarketResult<()> {
        if let Some(sender) = self.shutdown_tx.lock().await.as_ref() {
            let _ = sender.send(()).await;
        }

        let mut ws = self.ws_stream.write().await;
        if let Some(mut stream) = ws.take() {
            stream.close(None).await
                .map_err(|e| MarketError::NetworkError(e.to_string()))?;
        }

        tracing::info!("Disconnected from Binance WebSocket");
        Ok(())
    }
}

impl Drop for BinanceWebSocketClient {
    fn drop(&mut self) {
        // Spawn task to properly close WebSocket
        let ws_stream = Arc::clone(&self.ws_stream);
        let shutdown_tx = Arc::clone(&self.shutdown_tx);

        tokio::spawn(async move {
            if let Some(sender) = shutdown_tx.lock().await.as_ref() {
                let _ = sender.send(()).await;
            }

            let mut ws = ws_stream.write().await;
            if let Some(mut stream) = ws.take() {
                let _ = stream.close(None).await;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker() {
        let mut breaker = CircuitBreaker::new();

        assert_eq!(breaker.state, CircuitState::Closed);
        assert!(breaker.can_attempt());

        // Record failures
        for _ in 0..CIRCUIT_BREAKER_THRESHOLD {
            breaker.record_failure();
        }

        assert_eq!(breaker.state, CircuitState::Open);
        assert!(!breaker.can_attempt());

        // Record success
        breaker.record_success();
        assert_eq!(breaker.state, CircuitState::Closed);
        assert!(breaker.can_attempt());
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(2, StdDuration::from_secs(1));

        limiter.wait_if_needed().await;
        limiter.wait_if_needed().await;

        assert_eq!(limiter.requests.len(), 2);

        // Third request should wait
        let start = Instant::now();
        limiter.wait_if_needed().await;
        let elapsed = start.elapsed();

        assert!(elapsed >= StdDuration::from_millis(900));
    }

    #[tokio::test]
    async fn test_client_creation() {
        let client = BinanceWebSocketClient::new(true).unwrap();
        assert!(client.ws_url.contains("testnet"));
        assert!(!client.is_connected().await);
    }

    #[test]
    fn test_trade_event_parsing() {
        let json = r#"{
            "e": "trade",
            "E": 1672515782136,
            "s": "BTCUSDT",
            "t": 12345,
            "p": "50000.00",
            "q": "0.001",
            "T": 1672515782136,
            "m": true
        }"#;

        let event: TradeEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.symbol, "BTCUSDT");
        assert_eq!(event.price, "50000.00");
        assert!(event.is_buyer_maker);
    }

    #[test]
    fn test_kline_event_parsing() {
        let json = r#"{
            "e": "kline",
            "E": 1672515782136,
            "s": "BTCUSDT",
            "k": {
                "t": 1672515780000,
                "T": 1672515839999,
                "s": "BTCUSDT",
                "i": "1m",
                "o": "50000.00",
                "h": "50100.00",
                "l": "49900.00",
                "c": "50050.00",
                "v": "100.5",
                "n": 150,
                "x": false
            }
        }"#;

        let event: KlineEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.symbol, "BTCUSDT");
        assert_eq!(event.kline.interval, "1m");
        assert_eq!(event.kline.open, "50000.00");
    }
}
