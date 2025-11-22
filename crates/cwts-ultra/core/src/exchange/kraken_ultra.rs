// Kraken Ultra - REAL IMPLEMENTATION with WebSocket v2 and REST API
use tokio::net::TcpStream;
use tokio_tungstenite::{connect_async, WebSocketStream, MaybeTlsStream};
use tokio_tungstenite::tungstenite::Message;
use futures_util::{StreamExt, SinkExt};
use serde::{Deserialize, Serialize};
use serde_json;
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use hmac::{Hmac, Mac};
use sha2::{Sha256, Sha512};
use base64;
use tokio::time::{sleep, timeout, interval};
use reqwest::Client;
use url::Url;
use tracing::{info, warn, error, debug};

// Kraken WebSocket v2 URLs
const KRAKEN_WS_PUBLIC: &str = "wss://ws.kraken.com/v2";
const KRAKEN_WS_AUTH: &str = "wss://ws-auth.kraken.com/v2";
const KRAKEN_REST_URL: &str = "https://api.kraken.com";
const KRAKEN_REST_FUTURES_URL: &str = "https://futures.kraken.com";

/// Kraken WebSocket v2 message structure
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KrakenMessage {
    pub channel: Option<String>,
    pub r#type: Option<String>,
    pub data: Option<serde_json::Value>,
    pub sequence: Option<u64>,
    pub timestamp: Option<String>,
    pub symbol: Option<String>,
}

/// WebSocket subscription request
#[derive(Debug, Clone, Serialize)]
pub struct KrakenSubscription {
    pub method: String,
    pub params: KrakenSubscriptionParams,
    pub req_id: Option<u32>,
}

/// Subscription parameters
#[derive(Debug, Clone, Serialize)]
pub struct KrakenSubscriptionParams {
    pub channel: String,
    pub symbol: Option<Vec<String>>,
    pub snapshot: Option<bool>,
    pub event_trigger: Option<String>,
    pub ratecounter: Option<bool>,
}

/// Kraken order book data (Level 2)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KrakenOrderBook {
    pub symbol: String,
    pub bids: Vec<KrakenOrderBookLevel>,
    pub asks: Vec<KrakenOrderBookLevel>,
    pub checksum: Option<u32>,
    pub timestamp: String,
}

/// Order book level
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KrakenOrderBookLevel {
    pub price: String,
    pub qty: String,
    pub timestamp: Option<String>,
}

/// Kraken trade data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KrakenTrade {
    pub symbol: String,
    pub side: String,
    pub price: String,
    pub qty: String,
    pub ord_type: String,
    pub trade_id: u64,
    pub timestamp: String,
}

/// Kraken ticker data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KrakenTicker {
    pub symbol: String,
    pub bid: String,
    pub bid_qty: String,
    pub ask: String,
    pub ask_qty: String,
    pub last: String,
    pub volume: String,
    pub vwap: String,
    pub low: String,
    pub high: String,
    pub change: String,
    pub change_pct: String,
}

/// Kraken OHLC data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KrakenOHLC {
    pub symbol: String,
    pub open: String,
    pub high: String,
    pub low: String,
    pub close: String,
    pub volume: String,
    pub vwap: String,
    pub trades: u32,
    pub interval: u32,
    pub timestamp: String,
}

/// Order request structure
#[derive(Debug, Clone, Serialize)]
pub struct KrakenOrderRequest {
    pub ordertype: String,
    pub r#type: String,
    pub volume: String,
    pub pair: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price2: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leverage: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oflags: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub starttm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expiretm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub userref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub close: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trading_agreement: Option<String>,
}

/// Order response structure
#[derive(Debug, Clone, Deserialize)]
pub struct KrakenOrderResponse {
    pub descr: KrakenOrderDescription,
    pub txid: Vec<String>,
}

/// Order description
#[derive(Debug, Clone, Deserialize)]
pub struct KrakenOrderDescription {
    pub order: String,
    pub close: Option<String>,
}

/// Account balance
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KrakenBalance {
    pub asset: String,
    pub balance: String,
    pub hold: Option<String>,
}

/// Open position data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KrakenPosition {
    pub ordertxid: String,
    pub pair: String,
    pub time: f64,
    pub r#type: String,
    pub ordertype: String,
    pub cost: String,
    pub fee: String,
    pub vol: String,
    pub vol_closed: String,
    pub margin: String,
    pub value: Option<String>,
    pub net: String,
    pub misc: String,
    pub oflags: String,
}

/// Open orders data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KrakenOpenOrder {
    pub refid: Option<String>,
    pub userref: Option<String>,
    pub status: String,
    pub opentm: f64,
    pub starttm: f64,
    pub expiretm: f64,
    pub descr: KrakenOrderDescription,
    pub vol: String,
    pub vol_exec: String,
    pub cost: String,
    pub fee: String,
    pub price: String,
    pub stopprice: Option<String>,
    pub limitprice: Option<String>,
    pub misc: String,
    pub oflags: String,
}

/// API credentials
#[derive(Debug, Clone)]
pub struct KrakenCredentials {
    pub api_key: String,
    pub api_secret: String,
    pub api_passphrase: Option<String>, // For futures
}

/// Market type
#[derive(Debug, Clone, PartialEq)]
pub enum KrakenMarketType {
    Spot,
    Futures,
}

/// Rate limiter for API calls
#[derive(Debug)]
pub struct RateLimiter {
    counter: Arc<RwLock<u32>>,
    window_start: Arc<RwLock<SystemTime>>,
    limit: u32,
    window_duration: Duration,
}

impl RateLimiter {
    pub fn new(limit: u32, window_duration: Duration) -> Self {
        Self {
            counter: Arc::new(RwLock::new(0)),
            window_start: Arc::new(RwLock::new(SystemTime::now())),
            limit,
            window_duration,
        }
    }

    pub async fn acquire(&self) -> Result<(), Box<dyn std::error::Error>> {
        let now = SystemTime::now();
        let mut window_start = self.window_start.write();
        let mut counter = self.counter.write();

        // Reset counter if window has passed
        if now.duration_since(*window_start).unwrap_or(Duration::ZERO) > self.window_duration {
            *window_start = now;
            *counter = 0;
        }

        if *counter >= self.limit {
            let sleep_duration = self.window_duration - now.duration_since(*window_start).unwrap_or(Duration::ZERO);
            drop(counter);
            drop(window_start);
            
            sleep(sleep_duration).await;
            return self.acquire().await;
        }

        *counter += 1;
        Ok(())
    }
}

/// Kraken Ultra exchange connector
pub struct KrakenUltra {
    pub credentials: Option<KrakenCredentials>,
    pub market_type: KrakenMarketType,
    pub order_books: Arc<RwLock<HashMap<String, KrakenOrderBook>>>,
    pub trades: Arc<RwLock<Vec<KrakenTrade>>>,
    pub tickers: Arc<RwLock<HashMap<String, KrakenTicker>>>,
    pub ohlc_data: Arc<RwLock<HashMap<String, Vec<KrakenOHLC>>>>,
    pub balances: Arc<RwLock<HashMap<String, KrakenBalance>>>,
    pub positions: Arc<RwLock<HashMap<String, KrakenPosition>>>,
    pub open_orders: Arc<RwLock<HashMap<String, KrakenOpenOrder>>>,
    pub ws_stream: Arc<RwLock<Option<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,
    pub http_client: Client,
    pub rate_limiter: RateLimiter,
    pub connection_id: Arc<RwLock<Option<String>>>,
    pub subscriptions: Arc<RwLock<Vec<String>>>,
    pub ping_interval: Duration,
    pub reconnect_delay: Duration,
    pub max_reconnect_attempts: u32,
    pub nonce_counter: Arc<RwLock<u64>>,
}

impl KrakenUltra {
    /// Create new Kraken Ultra instance
    pub fn new(market_type: KrakenMarketType) -> Self {
        Self {
            credentials: None,
            market_type,
            order_books: Arc::new(RwLock::new(HashMap::new())),
            trades: Arc::new(RwLock::new(Vec::new())),
            tickers: Arc::new(RwLock::new(HashMap::new())),
            ohlc_data: Arc::new(RwLock::new(HashMap::new())),
            balances: Arc::new(RwLock::new(HashMap::new())),
            positions: Arc::new(RwLock::new(HashMap::new())),
            open_orders: Arc::new(RwLock::new(HashMap::new())),
            ws_stream: Arc::new(RwLock::new(None)),
            http_client: Client::new(),
            rate_limiter: RateLimiter::new(15, Duration::from_secs(1)), // Kraken API limits
            connection_id: Arc::new(RwLock::new(None)),
            subscriptions: Arc::new(RwLock::new(Vec::new())),
            ping_interval: Duration::from_secs(30),
            reconnect_delay: Duration::from_secs(5),
            max_reconnect_attempts: 10,
            nonce_counter: Arc::new(RwLock::new(0)),
        }
    }

    /// Set credentials for authenticated operations
    pub fn set_credentials(&mut self, api_key: String, api_secret: String, api_passphrase: Option<String>) {
        self.credentials = Some(KrakenCredentials {
            api_key,
            api_secret,
            api_passphrase,
        });
    }

    /// Get WebSocket URL based on authentication need
    fn get_ws_url(&self, is_private: bool) -> &'static str {
        if is_private {
            KRAKEN_WS_AUTH
        } else {
            KRAKEN_WS_PUBLIC
        }
    }

    /// Generate nonce for API requests
    fn generate_nonce(&self) -> u64 {
        let mut nonce = self.nonce_counter.write();
        *nonce += 1;
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64 + *nonce
    }

    /// Generate HMAC signature for Kraken API
    fn generate_signature(&self, endpoint: &str, nonce: u64, post_data: &str) -> Result<String, Box<dyn std::error::Error>> {
        let credentials = self.credentials.as_ref().ok_or("No credentials set")?;
        
        // Create SHA256 hash of nonce + post_data
        let mut hasher = sha2::Sha256::new();
        use sha2::Digest;
        hasher.update(format!("{}{}", nonce, post_data));
        let hash = hasher.finalize();

        // Create HMAC-SHA512 of endpoint + hash with decoded secret
        let decoded_secret = base64::decode(&credentials.api_secret)?;
        let mut mac = Hmac::<Sha512>::new_from_slice(&decoded_secret)?;
        mac.update(endpoint.as_bytes());
        mac.update(&hash);
        
        let result = mac.finalize();
        Ok(base64::encode(result.into_bytes()))
    }

    /// Connect to WebSocket
    pub async fn connect_websocket(&mut self, is_private: bool) -> Result<(), Box<dyn std::error::Error>> {
        let url = self.get_ws_url(is_private);
        info!("Connecting to Kraken WebSocket: {}", url);
        
        let (ws_stream, _) = timeout(Duration::from_secs(10), connect_async(url.as_str())).await??;
        
        let mut stream_guard = self.ws_stream.write();
        *stream_guard = Some(ws_stream);
        drop(stream_guard);
        
        info!("Connected to Kraken WebSocket successfully");
        
        // If private connection, get authentication token
        if is_private {
            self.authenticate_websocket().await?;
        }
        
        Ok(())
    }

    /// Authenticate WebSocket connection using REST API token
    async fn authenticate_websocket(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Get WebSocket token from REST API
        let token = self.get_websocket_token().await?;
        
        let auth_message = serde_json::json!({
            "method": "subscribe",
            "params": {
                "channel": "balances",
                "token": token
            },
            "req_id": 1001
        });
        
        if let Some(ref mut ws) = *self.ws_stream.write() {
            ws.send(Message::Text(auth_message.to_string())).await?;
        }
        
        info!("Sent WebSocket authentication message");
        Ok(())
    }

    /// Get WebSocket authentication token
    async fn get_websocket_token(&self) -> Result<String, Box<dyn std::error::Error>> {
        let endpoint = "/0/private/GetWebSocketsToken";
        let nonce = self.generate_nonce();
        let post_data = format!("nonce={}", nonce);
        
        let response = self.make_authenticated_request("POST", endpoint, &post_data).await?;
        
        if let Some(result) = response.get("result") {
            if let Some(token) = result.get("token").and_then(|t| t.as_str()) {
                return Ok(token.to_string());
            }
        }
        
        Err("Failed to get WebSocket token".into())
    }

    /// Subscribe to order book updates
    pub async fn subscribe_order_book(&mut self, symbols: Vec<String>, depth: Option<u32>) -> Result<(), Box<dyn std::error::Error>> {
        let subscription = KrakenSubscription {
            method: "subscribe".to_string(),
            params: KrakenSubscriptionParams {
                channel: format!("book{}", depth.unwrap_or(10)),
                symbol: Some(symbols.clone()),
                snapshot: Some(true),
                event_trigger: None,
                ratecounter: None,
            },
            req_id: Some(1000),
        };

        if let Some(ref mut ws) = *self.ws_stream.write() {
            let message = serde_json::to_string(&subscription)?;
            ws.send(Message::Text(message)).await?;
            
            // Add to subscriptions list
            self.subscriptions.write().extend(symbols.into_iter().map(|s| format!("book-{}", s)));
            
            info!("Subscribed to order book updates");
        }
        
        Ok(())
    }

    /// Subscribe to trade data
    pub async fn subscribe_trades(&mut self, symbols: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
        let subscription = KrakenSubscription {
            method: "subscribe".to_string(),
            params: KrakenSubscriptionParams {
                channel: "trade".to_string(),
                symbol: Some(symbols.clone()),
                snapshot: Some(false),
                event_trigger: None,
                ratecounter: None,
            },
            req_id: Some(1001),
        };

        if let Some(ref mut ws) = *self.ws_stream.write() {
            let message = serde_json::to_string(&subscription)?;
            ws.send(Message::Text(message)).await?;
            
            // Add to subscriptions list
            self.subscriptions.write().extend(symbols.into_iter().map(|s| format!("trade-{}", s)));
            
            info!("Subscribed to trade updates");
        }
        
        Ok(())
    }

    /// Subscribe to ticker data
    pub async fn subscribe_ticker(&mut self, symbols: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
        let subscription = KrakenSubscription {
            method: "subscribe".to_string(),
            params: KrakenSubscriptionParams {
                channel: "ticker".to_string(),
                symbol: Some(symbols.clone()),
                snapshot: Some(true),
                event_trigger: None,
                ratecounter: None,
            },
            req_id: Some(1002),
        };

        if let Some(ref mut ws) = *self.ws_stream.write() {
            let message = serde_json::to_string(&subscription)?;
            ws.send(Message::Text(message)).await?;
            
            // Add to subscriptions list
            self.subscriptions.write().extend(symbols.into_iter().map(|s| format!("ticker-{}", s)));
            
            info!("Subscribed to ticker updates");
        }
        
        Ok(())
    }

    /// Subscribe to OHLC data
    pub async fn subscribe_ohlc(&mut self, symbols: Vec<String>, interval: u32) -> Result<(), Box<dyn std::error::Error>> {
        let subscription = KrakenSubscription {
            method: "subscribe".to_string(),
            params: KrakenSubscriptionParams {
                channel: format!("ohlc{}", interval),
                symbol: Some(symbols.clone()),
                snapshot: Some(false),
                event_trigger: None,
                ratecounter: None,
            },
            req_id: Some(1003),
        };

        if let Some(ref mut ws) = *self.ws_stream.write() {
            let message = serde_json::to_string(&subscription)?;
            ws.send(Message::Text(message)).await?;
            
            // Add to subscriptions list
            self.subscriptions.write().extend(symbols.into_iter().map(|s| format!("ohlc{}-{}", interval, s)));
            
            info!("Subscribed to OHLC updates with interval {}", interval);
        }
        
        Ok(())
    }

    /// Subscribe to own trades (private)
    pub async fn subscribe_own_trades(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let token = self.get_websocket_token().await?;
        
        let subscription = serde_json::json!({
            "method": "subscribe",
            "params": {
                "channel": "ownTrades",
                "token": token,
                "snapshot": true
            },
            "req_id": 2001
        });

        if let Some(ref mut ws) = *self.ws_stream.write() {
            ws.send(Message::Text(subscription.to_string())).await?;
            
            // Add to subscriptions list
            self.subscriptions.write().push("ownTrades".to_string());
            
            info!("Subscribed to own trades");
        }
        
        Ok(())
    }

    /// Subscribe to open orders (private)
    pub async fn subscribe_open_orders(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let token = self.get_websocket_token().await?;
        
        let subscription = serde_json::json!({
            "method": "subscribe",
            "params": {
                "channel": "openOrders",
                "token": token,
                "ratecounter": true
            },
            "req_id": 2002
        });

        if let Some(ref mut ws) = *self.ws_stream.write() {
            ws.send(Message::Text(subscription.to_string())).await?;
            
            // Add to subscriptions list
            self.subscriptions.write().push("openOrders".to_string());
            
            info!("Subscribed to open orders");
        }
        
        Ok(())
    }

    /// Start message processing loop
    pub async fn start_message_loop(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let ws_stream = Arc::clone(&self.ws_stream);
        let order_books = Arc::clone(&self.order_books);
        let trades = Arc::clone(&self.trades);
        let tickers = Arc::clone(&self.tickers);
        let ohlc_data = Arc::clone(&self.ohlc_data);
        let balances = Arc::clone(&self.balances);
        let positions = Arc::clone(&self.positions);
        let open_orders = Arc::clone(&self.open_orders);
        
        tokio::spawn(async move {
            let mut ping_interval = interval(Duration::from_secs(30));
            
            loop {
                tokio::select! {
                    _ = ping_interval.tick() => {
                        // Send periodic ping
                        if let Some(ref mut ws) = *ws_stream.write() {
                            if let Err(e) = ws.send(Message::Ping(vec![])).await {
                                error!("Failed to send ping: {}", e);
                                break;
                            }
                        }
                    }
                    message = async {
                        let mut stream_guard = ws_stream.write();
                        if let Some(ref mut ws) = *stream_guard {
                            match timeout(Duration::from_secs(35), ws.next()).await {
                                Ok(Some(Ok(msg))) => Some(msg),
                                Ok(Some(Err(e))) => {
                                    error!("WebSocket error: {}", e);
                                    None
                                }
                                Ok(None) => {
                                    warn!("WebSocket stream ended");
                                    None
                                }
                                Err(_) => {
                                    warn!("WebSocket timeout");
                                    None
                                }
                            }
                        } else {
                            sleep(Duration::from_millis(100)).await;
                            None
                        }
                    } => {
                        if let Some(msg) = message {
                            match msg {
                                Message::Text(text) => {
                                    if let Err(e) = Self::process_message(
                                        &text,
                                        &order_books,
                                        &trades,
                                        &tickers,
                                        &ohlc_data,
                                        &balances,
                                        &positions,
                                        &open_orders,
                                    ).await {
                                        error!("Failed to process message: {}", e);
                                    }
                                }
                                Message::Pong(_) => {
                                    debug!("Received pong");
                                }
                                Message::Close(_) => {
                                    warn!("WebSocket closed by server");
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Process incoming WebSocket message
    async fn process_message(
        text: &str,
        order_books: &Arc<RwLock<HashMap<String, KrakenOrderBook>>>,
        trades: &Arc<RwLock<Vec<KrakenTrade>>>,
        tickers: &Arc<RwLock<HashMap<String, KrakenTicker>>>,
        ohlc_data: &Arc<RwLock<HashMap<String, Vec<KrakenOHLC>>>>,
        balances: &Arc<RwLock<HashMap<String, KrakenBalance>>>,
        positions: &Arc<RwLock<HashMap<String, KrakenPosition>>>,
        open_orders: &Arc<RwLock<HashMap<String, KrakenOpenOrder>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let message: KrakenMessage = serde_json::from_str(text)?;
        
        if let (Some(channel), Some(data)) = (&message.channel, &message.data) {
            match channel.as_str() {
                c if c.starts_with("book") => {
                    Self::process_order_book_update(c, data, order_books).await?;
                }
                "trade" => {
                    Self::process_trade_update(data, trades).await?;
                }
                "ticker" => {
                    Self::process_ticker_update(data, tickers).await?;
                }
                c if c.starts_with("ohlc") => {
                    Self::process_ohlc_update(c, data, ohlc_data).await?;
                }
                "balances" => {
                    Self::process_balance_update(data, balances).await?;
                }
                "ownTrades" => {
                    Self::process_own_trades_update(data, positions).await?;
                }
                "openOrders" => {
                    Self::process_open_orders_update(data, open_orders).await?;
                }
                _ => {
                    debug!("Unhandled channel: {}", channel);
                }
            }
        }

        Ok(())
    }

    /// Process order book updates
    async fn process_order_book_update(
        channel: &str,
        data: &serde_json::Value,
        order_books: &Arc<RwLock<HashMap<String, KrakenOrderBook>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(updates) = serde_json::from_value::<Vec<KrakenOrderBook>>(data.clone()) {
            for update in updates {
                let symbol = update.symbol.clone();
                order_books.write().insert(symbol.clone(), update);
                debug!("Updated order book for {} via {}", symbol, channel);
            }
        }
        Ok(())
    }

    /// Process trade updates
    async fn process_trade_update(
        data: &serde_json::Value,
        trades: &Arc<RwLock<Vec<KrakenTrade>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(new_trades) = serde_json::from_value::<Vec<KrakenTrade>>(data.clone()) {
            let mut trades_guard = trades.write();
            for trade in new_trades {
                debug!("Trade update: {} {} @ {} ({})", 
                    trade.symbol, trade.qty, trade.price, trade.side);
                trades_guard.push(trade);
                
                // Keep only last 10000 trades
                if trades_guard.len() > 10000 {
                    trades_guard.drain(..1000);
                }
            }
        }
        Ok(())
    }

    /// Process ticker updates
    async fn process_ticker_update(
        data: &serde_json::Value,
        tickers: &Arc<RwLock<HashMap<String, KrakenTicker>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(ticker_updates) = serde_json::from_value::<Vec<KrakenTicker>>(data.clone()) {
            let mut tickers_guard = tickers.write();
            for ticker in ticker_updates {
                debug!("Ticker update: {} bid:{} ask:{} last:{}", 
                    ticker.symbol, ticker.bid, ticker.ask, ticker.last);
                tickers_guard.insert(ticker.symbol.clone(), ticker);
            }
        }
        Ok(())
    }

    /// Process OHLC updates
    async fn process_ohlc_update(
        channel: &str,
        data: &serde_json::Value,
        ohlc_data: &Arc<RwLock<HashMap<String, Vec<KrakenOHLC>>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(ohlc_updates) = serde_json::from_value::<Vec<KrakenOHLC>>(data.clone()) {
            let mut ohlc_guard = ohlc_data.write();
            for ohlc in ohlc_updates {
                debug!("OHLC update for {} via {}: O:{} H:{} L:{} C:{}", 
                    ohlc.symbol, channel, ohlc.open, ohlc.high, ohlc.low, ohlc.close);
                
                let key = format!("{}-{}", ohlc.symbol, ohlc.interval);
                let ohlc_vec = ohlc_guard.entry(key).or_insert_with(Vec::new);
                ohlc_vec.push(ohlc);
                
                // Keep only last 1000 candles per symbol/interval
                if ohlc_vec.len() > 1000 {
                    ohlc_vec.drain(..100);
                }
            }
        }
        Ok(())
    }

    /// Process balance updates
    async fn process_balance_update(
        data: &serde_json::Value,
        balances: &Arc<RwLock<HashMap<String, KrakenBalance>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(balance_updates) = serde_json::from_value::<Vec<KrakenBalance>>(data.clone()) {
            let mut balances_guard = balances.write();
            for balance in balance_updates {
                debug!("Balance update: {} = {}", balance.asset, balance.balance);
                balances_guard.insert(balance.asset.clone(), balance);
            }
        }
        Ok(())
    }

    /// Process own trades updates
    async fn process_own_trades_update(
        data: &serde_json::Value,
        positions: &Arc<RwLock<HashMap<String, KrakenPosition>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(position_updates) = serde_json::from_value::<Vec<KrakenPosition>>(data.clone()) {
            let mut positions_guard = positions.write();
            for position in position_updates {
                debug!("Position update: {} {} {}", position.ordertxid, position.pair, position.vol);
                positions_guard.insert(position.ordertxid.clone(), position);
            }
        }
        Ok(())
    }

    /// Process open orders updates
    async fn process_open_orders_update(
        data: &serde_json::Value,
        open_orders: &Arc<RwLock<HashMap<String, KrakenOpenOrder>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(order_updates) = serde_json::from_value::<HashMap<String, KrakenOpenOrder>>(data.clone()) {
            let mut orders_guard = open_orders.write();
            for (order_id, order) in order_updates {
                debug!("Open order update: {} {} {}", order_id, order.descr.order, order.status);
                orders_guard.insert(order_id, order);
            }
        }
        Ok(())
    }

    /// Place market order
    pub async fn place_market_order(
        &self,
        pair: &str,
        order_type: &str, // "buy" or "sell"
        volume: &str,
        leverage: Option<&str>,
    ) -> Result<KrakenOrderResponse, Box<dyn std::error::Error>> {
        self.rate_limiter.acquire().await?;
        
        let order = KrakenOrderRequest {
            ordertype: "market".to_string(),
            r#type: order_type.to_string(),
            volume: volume.to_string(),
            pair: pair.to_string(),
            price: None,
            price2: None,
            leverage: leverage.map(|l| l.to_string()),
            oflags: None,
            starttm: None,
            expiretm: None,
            userref: None,
            validate: None,
            close: None,
            trading_agreement: Some("agree".to_string()),
        };
        
        self.send_order(order).await
    }

    /// Place limit order
    pub async fn place_limit_order(
        &self,
        pair: &str,
        order_type: &str, // "buy" or "sell"
        volume: &str,
        price: &str,
        leverage: Option<&str>,
        time_in_force: Option<&str>,
    ) -> Result<KrakenOrderResponse, Box<dyn std::error::Error>> {
        self.rate_limiter.acquire().await?;
        
        let mut oflags = Vec::new();
        if let Some(tif) = time_in_force {
            match tif {
                "IOC" => oflags.push("fciq"),
                "FOK" => oflags.push("fcib"),
                _ => {}
            }
        }
        
        let order = KrakenOrderRequest {
            ordertype: "limit".to_string(),
            r#type: order_type.to_string(),
            volume: volume.to_string(),
            pair: pair.to_string(),
            price: Some(price.to_string()),
            price2: None,
            leverage: leverage.map(|l| l.to_string()),
            oflags: if oflags.is_empty() { None } else { Some(oflags.join(",")) },
            starttm: None,
            expiretm: None,
            userref: None,
            validate: None,
            close: None,
            trading_agreement: Some("agree".to_string()),
        };
        
        self.send_order(order).await
    }

    /// Place stop loss order
    pub async fn place_stop_order(
        &self,
        pair: &str,
        order_type: &str, // "buy" or "sell"
        volume: &str,
        price: &str, // stop price
        price2: Option<&str>, // limit price for stop-limit
        leverage: Option<&str>,
    ) -> Result<KrakenOrderResponse, Box<dyn std::error::Error>> {
        self.rate_limiter.acquire().await?;
        
        let ordertype = if price2.is_some() { "stop-loss-limit" } else { "stop-loss" };
        
        let order = KrakenOrderRequest {
            ordertype: ordertype.to_string(),
            r#type: order_type.to_string(),
            volume: volume.to_string(),
            pair: pair.to_string(),
            price: Some(price.to_string()),
            price2: price2.map(|p| p.to_string()),
            leverage: leverage.map(|l| l.to_string()),
            oflags: None,
            starttm: None,
            expiretm: None,
            userref: None,
            validate: None,
            close: None,
            trading_agreement: Some("agree".to_string()),
        };
        
        self.send_order(order).await
    }

    /// Send order to Kraken
    async fn send_order(&self, order: KrakenOrderRequest) -> Result<KrakenOrderResponse, Box<dyn std::error::Error>> {
        let endpoint = "/0/private/AddOrder";
        let post_data = serde_urlencoded::to_string(&order)?;
        
        let response = self.make_authenticated_request("POST", endpoint, &post_data).await?;
        
        if let Some(error) = response.get("error") {
            if let Some(error_arr) = error.as_array() {
                if !error_arr.is_empty() {
                    return Err(format!("Kraken API error: {:?}", error_arr).into());
                }
            }
        }
        
        if let Some(result) = response.get("result") {
            let order_response: KrakenOrderResponse = serde_json::from_value(result.clone())?;
            return Ok(order_response);
        }
        
        Err("Invalid response format".into())
    }

    /// Cancel order
    pub async fn cancel_order(&self, txid: &str) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        self.rate_limiter.acquire().await?;
        
        let endpoint = "/0/private/CancelOrder";
        let post_data = format!("nonce={}&txid={}", self.generate_nonce(), txid);
        
        self.make_authenticated_request("POST", endpoint, &post_data).await
    }

    /// Cancel all orders
    pub async fn cancel_all_orders(&self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        self.rate_limiter.acquire().await?;
        
        let endpoint = "/0/private/CancelAll";
        let post_data = format!("nonce={}", self.generate_nonce());
        
        self.make_authenticated_request("POST", endpoint, &post_data).await
    }

    /// Get account balance
    pub async fn get_balance(&self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        self.rate_limiter.acquire().await?;
        
        let endpoint = "/0/private/Balance";
        let post_data = format!("nonce={}", self.generate_nonce());
        
        self.make_authenticated_request("POST", endpoint, &post_data).await
    }

    /// Get extended balance with holds
    pub async fn get_balance_ex(&self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        self.rate_limiter.acquire().await?;
        
        let endpoint = "/0/private/BalanceEx";
        let post_data = format!("nonce={}", self.generate_nonce());
        
        self.make_authenticated_request("POST", endpoint, &post_data).await
    }

    /// Get trade balance
    pub async fn get_trade_balance(&self, asset: Option<&str>) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        self.rate_limiter.acquire().await?;
        
        let endpoint = "/0/private/TradeBalance";
        let mut post_data = format!("nonce={}", self.generate_nonce());
        
        if let Some(a) = asset {
            post_data.push_str(&format!("&asset={}", a));
        }
        
        self.make_authenticated_request("POST", endpoint, &post_data).await
    }

    /// Get open orders
    pub async fn get_open_orders(&self, trades: Option<bool>, userref: Option<&str>) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        self.rate_limiter.acquire().await?;
        
        let endpoint = "/0/private/OpenOrders";
        let mut post_data = format!("nonce={}", self.generate_nonce());
        
        if let Some(t) = trades {
            post_data.push_str(&format!("&trades={}", t));
        }
        if let Some(ur) = userref {
            post_data.push_str(&format!("&userref={}", ur));
        }
        
        self.make_authenticated_request("POST", endpoint, &post_data).await
    }

    /// Get closed orders
    pub async fn get_closed_orders(
        &self,
        trades: Option<bool>,
        userref: Option<&str>,
        start: Option<u64>,
        end: Option<u64>,
        ofs: Option<u32>,
        closetime: Option<&str>,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        self.rate_limiter.acquire().await?;
        
        let endpoint = "/0/private/ClosedOrders";
        let mut post_data = format!("nonce={}", self.generate_nonce());
        
        if let Some(t) = trades {
            post_data.push_str(&format!("&trades={}", t));
        }
        if let Some(ur) = userref {
            post_data.push_str(&format!("&userref={}", ur));
        }
        if let Some(s) = start {
            post_data.push_str(&format!("&start={}", s));
        }
        if let Some(e) = end {
            post_data.push_str(&format!("&end={}", e));
        }
        if let Some(o) = ofs {
            post_data.push_str(&format!("&ofs={}", o));
        }
        if let Some(ct) = closetime {
            post_data.push_str(&format!("&closetime={}", ct));
        }
        
        self.make_authenticated_request("POST", endpoint, &post_data).await
    }

    /// Get trades history
    pub async fn get_trades_history(
        &self,
        r#type: Option<&str>,
        trades: Option<bool>,
        start: Option<u64>,
        end: Option<u64>,
        ofs: Option<u32>,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        self.rate_limiter.acquire().await?;
        
        let endpoint = "/0/private/TradesHistory";
        let mut post_data = format!("nonce={}", self.generate_nonce());
        
        if let Some(t) = r#type {
            post_data.push_str(&format!("&type={}", t));
        }
        if let Some(tr) = trades {
            post_data.push_str(&format!("&trades={}", tr));
        }
        if let Some(s) = start {
            post_data.push_str(&format!("&start={}", s));
        }
        if let Some(e) = end {
            post_data.push_str(&format!("&end={}", e));
        }
        if let Some(o) = ofs {
            post_data.push_str(&format!("&ofs={}", o));
        }
        
        self.make_authenticated_request("POST", endpoint, &post_data).await
    }

    /// Get open positions
    pub async fn get_open_positions(&self, txid: Option<&str>, docalcs: Option<bool>) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        self.rate_limiter.acquire().await?;
        
        let endpoint = "/0/private/OpenPositions";
        let mut post_data = format!("nonce={}", self.generate_nonce());
        
        if let Some(t) = txid {
            post_data.push_str(&format!("&txid={}", t));
        }
        if let Some(dc) = docalcs {
            post_data.push_str(&format!("&docalcs={}", dc));
        }
        
        self.make_authenticated_request("POST", endpoint, &post_data).await
    }

    /// Get ledgers
    pub async fn get_ledgers(
        &self,
        asset: Option<&str>,
        aclass: Option<&str>,
        r#type: Option<&str>,
        start: Option<u64>,
        end: Option<u64>,
        ofs: Option<u32>,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        self.rate_limiter.acquire().await?;
        
        let endpoint = "/0/private/Ledgers";
        let mut post_data = format!("nonce={}", self.generate_nonce());
        
        if let Some(a) = asset {
            post_data.push_str(&format!("&asset={}", a));
        }
        if let Some(ac) = aclass {
            post_data.push_str(&format!("&aclass={}", ac));
        }
        if let Some(t) = r#type {
            post_data.push_str(&format!("&type={}", t));
        }
        if let Some(s) = start {
            post_data.push_str(&format!("&start={}", s));
        }
        if let Some(e) = end {
            post_data.push_str(&format!("&end={}", e));
        }
        if let Some(o) = ofs {
            post_data.push_str(&format!("&ofs={}", o));
        }
        
        self.make_authenticated_request("POST", endpoint, &post_data).await
    }

    /// Make authenticated REST API request
    async fn make_authenticated_request(
        &self,
        method: &str,
        endpoint: &str,
        post_data: &str,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let credentials = self.credentials.as_ref().ok_or("No credentials set")?;
        
        let nonce = if post_data.contains("nonce=") {
            post_data.to_string()
        } else {
            format!("nonce={}&{}", self.generate_nonce(), post_data)
        };
        
        let signature = self.generate_signature(endpoint, 
            nonce.split('=').nth(1).and_then(|s| s.split('&').next()).unwrap_or("0").parse()?, 
            &nonce)?;
        
        let url = match self.market_type {
            KrakenMarketType::Spot => format!("{}{}", KRAKEN_REST_URL, endpoint),
            KrakenMarketType::Futures => format!("{}{}", KRAKEN_REST_FUTURES_URL, endpoint),
        };

        let mut request = self.http_client.request(
            method.parse().unwrap_or(reqwest::Method::POST),
            &url,
        );

        request = request
            .header("API-Key", &credentials.api_key)
            .header("API-Sign", signature)
            .header("Content-Type", "application/x-www-form-urlencoded");

        if method != "GET" {
            request = request.body(nonce);
        }

        let response = request.send().await?;
        let response_text = response.text().await?;
        
        debug!("Kraken API Response: {}", response_text);
        
        let json_response: serde_json::Value = serde_json::from_str(&response_text)?;
        
        Ok(json_response)
    }

    /// Reconnect WebSocket with exponential backoff
    pub async fn reconnect_websocket(&mut self, is_private: bool, attempt: u32) -> Result<(), Box<dyn std::error::Error>> {
        if attempt > self.max_reconnect_attempts {
            return Err("Max reconnection attempts reached".into());
        }

        let delay = self.reconnect_delay * 2_u32.pow(attempt.min(5));
        warn!("Reconnecting to Kraken WebSocket in {:?} (attempt {})", delay, attempt);
        sleep(delay).await;

        match self.connect_websocket(is_private).await {
            Ok(_) => {
                info!("Reconnected to Kraken WebSocket successfully");
                
                // Re-subscribe to all topics
                let subscriptions = self.subscriptions.read().clone();
                for topic in subscriptions {
                    if let Err(e) = self.resubscribe_topic(&topic).await {
                        error!("Failed to resubscribe to {}: {}", topic, e);
                    }
                }
                
                Ok(())
            }
            Err(e) => {
                error!("Kraken WebSocket reconnection attempt {} failed: {}", attempt, e);
                self.reconnect_websocket(is_private, attempt + 1).await
            }
        }
    }

    /// Re-subscribe to a topic after reconnection
    async fn resubscribe_topic(&mut self, topic: &str) -> Result<(), Box<dyn std::error::Error>> {
        let parts: Vec<&str> = topic.split('-').collect();
        
        match parts[0] {
            "book" => {
                if parts.len() > 1 {
                    self.subscribe_order_book(vec![parts[1].to_string()], Some(10)).await?;
                }
            }
            "trade" => {
                if parts.len() > 1 {
                    self.subscribe_trades(vec![parts[1].to_string()]).await?;
                }
            }
            "ticker" => {
                if parts.len() > 1 {
                    self.subscribe_ticker(vec![parts[1].to_string()]).await?;
                }
            }
            "ohlc1" => {
                if parts.len() > 1 {
                    self.subscribe_ohlc(vec![parts[1].to_string()], 1).await?;
                }
            }
            "ownTrades" => {
                self.subscribe_own_trades().await?;
            }
            "openOrders" => {
                self.subscribe_open_orders().await?;
            }
            _ => {
                warn!("Unknown topic for resubscription: {}", topic);
            }
        }
        
        Ok(())
    }

    /// Get current order book for symbol
    pub fn get_order_book(&self, symbol: &str) -> Option<KrakenOrderBook> {
        self.order_books.read().get(symbol).cloned()
    }

    /// Get recent trades for symbol
    pub fn get_recent_trades(&self, symbol: &str, limit: usize) -> Vec<KrakenTrade> {
        self.trades.read()
            .iter()
            .filter(|t| t.symbol == symbol)
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get current ticker for symbol
    pub fn get_ticker(&self, symbol: &str) -> Option<KrakenTicker> {
        self.tickers.read().get(symbol).cloned()
    }

    /// Get OHLC data for symbol and interval
    pub fn get_ohlc_data(&self, symbol: &str, interval: u32, limit: usize) -> Vec<KrakenOHLC> {
        let key = format!("{}-{}", symbol, interval);
        self.ohlc_data.read()
            .get(&key)
            .map(|data| data.iter().rev().take(limit).cloned().collect())
            .unwrap_or_else(Vec::new)
    }

    /// Get current balances
    pub fn get_current_balances(&self) -> HashMap<String, KrakenBalance> {
        self.balances.read().clone()
    }

    /// Get current positions
    pub fn get_current_positions(&self) -> HashMap<String, KrakenPosition> {
        self.positions.read().clone()
    }

    /// Get current open orders
    pub fn get_current_open_orders(&self) -> HashMap<String, KrakenOpenOrder> {
        self.open_orders.read().clone()
    }

    /// Get connection status
    pub fn is_connected(&self) -> bool {
        self.ws_stream.read().is_some()
    }

    /// Get connection ID
    pub fn get_connection_id(&self) -> Option<String> {
        self.connection_id.read().clone()
    }

    /// Close WebSocket connection
    pub async fn disconnect(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut ws) = self.ws_stream.write().take() {
            ws.close(None).await?;
            info!("Kraken WebSocket connection closed");
        }
        Ok(())
    }

    /// Get system status
    pub async fn get_system_status(&self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let url = format!("{}/0/public/SystemStatus", KRAKEN_REST_URL);
        let response = self.http_client.get(&url).send().await?;
        let json_response: serde_json::Value = response.json().await?;
        Ok(json_response)
    }

    /// Get server time
    pub async fn get_server_time(&self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let url = format!("{}/0/public/Time", KRAKEN_REST_URL);
        let response = self.http_client.get(&url).send().await?;
        let json_response: serde_json::Value = response.json().await?;
        Ok(json_response)
    }

    /// Get asset info
    pub async fn get_asset_info(&self, asset: Option<&str>, aclass: Option<&str>) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut url = format!("{}/0/public/Assets", KRAKEN_REST_URL);
        let mut params = Vec::new();
        
        if let Some(a) = asset {
            params.push(format!("asset={}", a));
        }
        if let Some(ac) = aclass {
            params.push(format!("aclass={}", ac));
        }
        
        if !params.is_empty() {
            url.push('?');
            url.push_str(&params.join("&"));
        }
        
        let response = self.http_client.get(&url).send().await?;
        let json_response: serde_json::Value = response.json().await?;
        Ok(json_response)
    }

    /// Get tradable asset pairs
    pub async fn get_asset_pairs(&self, pair: Option<&str>, info: Option<&str>) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut url = format!("{}/0/public/AssetPairs", KRAKEN_REST_URL);
        let mut params = Vec::new();
        
        if let Some(p) = pair {
            params.push(format!("pair={}", p));
        }
        if let Some(i) = info {
            params.push(format!("info={}", i));
        }
        
        if !params.is_empty() {
            url.push('?');
            url.push_str(&params.join("&"));
        }
        
        let response = self.http_client.get(&url).send().await?;
        let json_response: serde_json::Value = response.json().await?;
        Ok(json_response)
    }

    /// Get ticker information
    pub async fn get_ticker_info(&self, pair: Option<&str>) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut url = format!("{}/0/public/Ticker", KRAKEN_REST_URL);
        
        if let Some(p) = pair {
            url.push_str(&format!("?pair={}", p));
        }
        
        let response = self.http_client.get(&url).send().await?;
        let json_response: serde_json::Value = response.json().await?;
        Ok(json_response)
    }

    /// Get OHLC data via REST
    pub async fn get_ohlc_rest(
        &self,
        pair: &str,
        interval: Option<u32>,
        since: Option<u64>,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut url = format!("{}/0/public/OHLC?pair={}", KRAKEN_REST_URL, pair);
        
        if let Some(i) = interval {
            url.push_str(&format!("&interval={}", i));
        }
        if let Some(s) = since {
            url.push_str(&format!("&since={}", s));
        }
        
        let response = self.http_client.get(&url).send().await?;
        let json_response: serde_json::Value = response.json().await?;
        Ok(json_response)
    }

    /// Get order book via REST
    pub async fn get_order_book_rest(&self, pair: &str, count: Option<u32>) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut url = format!("{}/0/public/Depth?pair={}", KRAKEN_REST_URL, pair);
        
        if let Some(c) = count {
            url.push_str(&format!("&count={}", c));
        }
        
        let response = self.http_client.get(&url).send().await?;
        let json_response: serde_json::Value = response.json().await?;
        Ok(json_response)
    }

    /// Get recent trades via REST
    pub async fn get_trades_rest(&self, pair: &str, since: Option<u64>) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut url = format!("{}/0/public/Trades?pair={}", KRAKEN_REST_URL, pair);
        
        if let Some(s) = since {
            url.push_str(&format!("&since={}", s));
        }
        
        let response = self.http_client.get(&url).send().await?;
        let json_response: serde_json::Value = response.json().await?;
        Ok(json_response)
    }

    /// Get spread data
    pub async fn get_spread(&self, pair: &str, since: Option<u64>) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut url = format!("{}/0/public/Spread?pair={}", KRAKEN_REST_URL, pair);
        
        if let Some(s) = since {
            url.push_str(&format!("&since={}", s));
        }
        
        let response = self.http_client.get(&url).send().await?;
        let json_response: serde_json::Value = response.json().await?;
        Ok(json_response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_kraken_ultra_creation() {
        let kraken = KrakenUltra::new(KrakenMarketType::Spot);
        assert_eq!(kraken.market_type, KrakenMarketType::Spot);
        assert!(!kraken.is_connected());
    }

    #[tokio::test]
    async fn test_credential_setting() {
        let mut kraken = KrakenUltra::new(KrakenMarketType::Spot);
        kraken.set_credentials("test_key".to_string(), "test_secret".to_string(), None);
        
        assert!(kraken.credentials.is_some());
        let creds = kraken.credentials.unwrap();
        assert_eq!(creds.api_key, "test_key");
        assert_eq!(creds.api_secret, "test_secret");
    }

    #[tokio::test]
    async fn test_nonce_generation() {
        let kraken = KrakenUltra::new(KrakenMarketType::Spot);
        let nonce1 = kraken.generate_nonce();
        let nonce2 = kraken.generate_nonce();
        
        assert!(nonce2 > nonce1);
    }

    #[tokio::test]
    async fn test_websocket_url_selection() {
        let kraken = KrakenUltra::new(KrakenMarketType::Spot);
        assert_eq!(kraken.get_ws_url(false), KRAKEN_WS_PUBLIC);
        assert_eq!(kraken.get_ws_url(true), KRAKEN_WS_AUTH);
    }

    #[tokio::test]
    async fn test_signature_generation() {
        let mut kraken = KrakenUltra::new(KrakenMarketType::Spot);
        kraken.set_credentials("test_key".to_string(), "a2VybmVsL3NlY3JldA==".to_string(), None);
        
        let endpoint = "/0/private/Balance";
        let nonce = 1234567890u64;
        let post_data = format!("nonce={}", nonce);
        
        let signature = kraken.generate_signature(endpoint, nonce, &post_data);
        assert!(signature.is_ok());
        
        let sig = signature.unwrap();
        assert!(!sig.is_empty());
        assert!(base64::decode(&sig).is_ok());
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = RateLimiter::new(2, Duration::from_secs(1));
        
        // First two requests should be immediate
        let start = SystemTime::now();
        limiter.acquire().await.unwrap();
        limiter.acquire().await.unwrap();
        
        let elapsed = start.elapsed().unwrap();
        assert!(elapsed < Duration::from_millis(100));
        
        // Third request should be delayed
        let start = SystemTime::now();
        limiter.acquire().await.unwrap();
        let elapsed = start.elapsed().unwrap();
        assert!(elapsed >= Duration::from_millis(900));
    }

    #[tokio::test]
    async fn test_message_parsing() {
        let json_msg = r#"{
            "channel": "book10",
            "type": "snapshot",
            "data": [
                {
                    "symbol": "BTC/USD",
                    "bids": [{"price": "50000.0", "qty": "1.0"}],
                    "asks": [{"price": "50100.0", "qty": "0.5"}],
                    "timestamp": "2024-01-01T00:00:00.000Z"
                }
            ],
            "sequence": 12345,
            "timestamp": "2024-01-01T00:00:00.000Z"
        }"#;
        
        let msg: Result<KrakenMessage, _> = serde_json::from_str(json_msg);
        assert!(msg.is_ok());
        
        let parsed = msg.unwrap();
        assert_eq!(parsed.channel, Some("book10".to_string()));
        assert_eq!(parsed.r#type, Some("snapshot".to_string()));
        assert_eq!(parsed.sequence, Some(12345));
        assert!(parsed.data.is_some());
    }

    #[tokio::test]
    #[ignore] // Requires real API connection
    async fn test_websocket_connection() {
        let mut kraken = KrakenUltra::new(KrakenMarketType::Spot);
        
        match kraken.connect_websocket(false).await {
            Ok(_) => println!("Kraken WebSocket connected successfully"),
            Err(e) => println!("Kraken WebSocket connection failed: {}", e),
        }
    }

    #[tokio::test]
    #[ignore] // Requires real API connection
    async fn test_public_api() {
        let kraken = KrakenUltra::new(KrakenMarketType::Spot);
        
        match kraken.get_system_status().await {
            Ok(status) => println!("System status: {:?}", status),
            Err(e) => println!("Failed to get system status: {}", e),
        }
    }

    #[tokio::test]
    async fn test_order_request_serialization() {
        let order = KrakenOrderRequest {
            ordertype: "limit".to_string(),
            r#type: "buy".to_string(),
            volume: "1.0".to_string(),
            pair: "BTCUSD".to_string(),
            price: Some("50000.0".to_string()),
            price2: None,
            leverage: None,
            oflags: None,
            starttm: None,
            expiretm: None,
            userref: None,
            validate: None,
            close: None,
            trading_agreement: Some("agree".to_string()),
        };
        
        let serialized = serde_urlencoded::to_string(&order);
        assert!(serialized.is_ok());
        let data = serialized.unwrap();
        assert!(data.contains("ordertype=limit"));
        assert!(data.contains("type=buy"));
        assert!(data.contains("volume=1.0"));
        assert!(data.contains("pair=BTCUSD"));
        assert!(data.contains("price=50000.0"));
    }
}