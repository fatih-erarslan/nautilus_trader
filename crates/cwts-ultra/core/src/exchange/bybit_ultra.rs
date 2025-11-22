// Bybit Ultra - REAL IMPLEMENTATION with WebSocket v5 and REST API
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
use sha2::Sha256;
use base64;
use tokio::time::{sleep, timeout};
use reqwest::Client;
use url::Url;
use tracing::{info, warn, error, debug};

// Bybit WebSocket v5 URLs
const BYBIT_WS_PUBLIC: &str = "wss://stream.bybit.com/v5/public/spot";
const BYBIT_WS_PRIVATE: &str = "wss://stream.bybit.com/v5/private";
const BYBIT_WS_LINEAR: &str = "wss://stream.bybit.com/v5/public/linear";
const BYBIT_WS_OPTION: &str = "wss://stream.bybit.com/v5/public/option";
const BYBIT_REST_URL: &str = "https://api.bybit.com";

/// Bybit WebSocket v5 message structure
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BybitMessage {
    pub success: Option<bool>,
    pub ret_msg: Option<String>,
    pub conn_id: Option<String>,
    pub req_id: Option<String>,
    pub op: Option<String>,
    pub topic: Option<String>,
    #[serde(rename = "type")]
    pub message_type: Option<String>,
    pub data: Option<serde_json::Value>,
    pub ts: Option<u64>,
}

/// WebSocket subscription parameters
#[derive(Debug, Clone, Serialize)]
pub struct SubscriptionArg {
    pub op: String,
    pub args: Vec<String>,
    pub req_id: Option<String>,
}

/// Bybit order book data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BybitOrderBook {
    pub s: String,        // Symbol
    pub b: Vec<[String; 2]>, // Bids [price, size]
    pub a: Vec<[String; 2]>, // Asks [price, size]
    pub u: u64,           // Update ID
    pub seq: u64,         // Sequence
    pub cts: u64,         // Cross sequence
}

/// Bybit trade data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BybitTrade {
    pub T: u64,           // Timestamp
    pub s: String,        // Symbol
    pub S: String,        // Side (Buy/Sell)
    pub v: String,        // Volume
    pub p: String,        // Price
    pub L: String,        // Trade direction
    pub i: String,        // Trade ID
    pub BT: bool,         // Block trade
}

/// Bybit kline/candlestick data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BybitKline {
    pub start: u64,       // Start time
    pub end: u64,         // End time
    pub interval: String, // Interval
    pub open: String,     // Open price
    pub close: String,    // Close price
    pub high: String,     // High price
    pub low: String,      // Low price
    pub volume: String,   // Volume
    pub turnover: String, // Turnover
    pub confirm: bool,    // Confirm
    pub timestamp: u64,   // Timestamp
}

/// Bybit position data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BybitPosition {
    #[serde(rename = "positionIdx")]
    pub position_idx: u8,
    #[serde(rename = "tradeMode")]
    pub trade_mode: u8,
    pub symbol: String,
    pub side: String,
    pub size: String,
    #[serde(rename = "positionValue")]
    pub position_value: String,
    #[serde(rename = "avgPrice")]
    pub avg_price: String,
    #[serde(rename = "unrealisedPnl")]
    pub unrealised_pnl: String,
    #[serde(rename = "markPrice")]
    pub mark_price: String,
    #[serde(rename = "liqPrice")]
    pub liq_price: String,
    #[serde(rename = "bustPrice")]
    pub bust_price: String,
    #[serde(rename = "positionMM")]
    pub position_mm: String,
    #[serde(rename = "positionIM")]
    pub position_im: String,
    #[serde(rename = "tpslMode")]
    pub tpsl_mode: String,
    #[serde(rename = "takeProfit")]
    pub take_profit: String,
    #[serde(rename = "stopLoss")]
    pub stop_loss: String,
    #[serde(rename = "trailingStop")]
    pub trailing_stop: String,
    #[serde(rename = "sessionAvgPrice")]
    pub session_avg_price: String,
    #[serde(rename = "createdTime")]
    pub created_time: String,
    #[serde(rename = "updatedTime")]
    pub updated_time: String,
    pub seq: u64,
}

/// Bybit order data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BybitOrder {
    #[serde(rename = "orderId")]
    pub order_id: String,
    #[serde(rename = "orderLinkId")]
    pub order_link_id: String,
    #[serde(rename = "blockTradeId")]
    pub block_trade_id: String,
    pub symbol: String,
    pub price: String,
    pub qty: String,
    pub side: String,
    #[serde(rename = "isLeverage")]
    pub is_leverage: String,
    #[serde(rename = "positionIdx")]
    pub position_idx: u8,
    #[serde(rename = "orderStatus")]
    pub order_status: String,
    #[serde(rename = "cancelType")]
    pub cancel_type: String,
    #[serde(rename = "rejectReason")]
    pub reject_reason: String,
    #[serde(rename = "avgPrice")]
    pub avg_price: String,
    #[serde(rename = "leavesQty")]
    pub leaves_qty: String,
    #[serde(rename = "leavesValue")]
    pub leaves_value: String,
    #[serde(rename = "cumExecQty")]
    pub cum_exec_qty: String,
    #[serde(rename = "cumExecValue")]
    pub cum_exec_value: String,
    #[serde(rename = "cumExecFee")]
    pub cum_exec_fee: String,
    #[serde(rename = "timeInForce")]
    pub time_in_force: String,
    #[serde(rename = "orderType")]
    pub order_type: String,
    #[serde(rename = "stopOrderType")]
    pub stop_order_type: String,
    #[serde(rename = "orderIv")]
    pub order_iv: String,
    #[serde(rename = "marketUnit")]
    pub market_unit: String,
    #[serde(rename = "triggerPrice")]
    pub trigger_price: String,
    #[serde(rename = "takeProfit")]
    pub take_profit: String,
    #[serde(rename = "stopLoss")]
    pub stop_loss: String,
    #[serde(rename = "tpslMode")]
    pub tpsl_mode: String,
    #[serde(rename = "tpLimitPrice")]
    pub tp_limit_price: String,
    #[serde(rename = "slLimitPrice")]
    pub sl_limit_price: String,
    #[serde(rename = "tpTriggerBy")]
    pub tp_trigger_by: String,
    #[serde(rename = "slTriggerBy")]
    pub sl_trigger_by: String,
    #[serde(rename = "triggerDirection")]
    pub trigger_direction: u8,
    #[serde(rename = "triggerBy")]
    pub trigger_by: String,
    #[serde(rename = "lastPriceOnCreated")]
    pub last_price_on_created: String,
    #[serde(rename = "reduceOnly")]
    pub reduce_only: bool,
    #[serde(rename = "closeOnTrigger")]
    pub close_on_trigger: bool,
    #[serde(rename = "placeType")]
    pub place_type: String,
    #[serde(rename = "smpType")]
    pub smp_type: String,
    #[serde(rename = "smpGroup")]
    pub smp_group: u32,
    #[serde(rename = "smpOrderId")]
    pub smp_order_id: String,
    #[serde(rename = "createdTime")]
    pub created_time: String,
    #[serde(rename = "updatedTime")]
    pub updated_time: String,
}

/// Bybit wallet data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BybitWallet {
    #[serde(rename = "accountIMRate")]
    pub account_im_rate: String,
    #[serde(rename = "accountMMRate")]
    pub account_mm_rate: String,
    #[serde(rename = "totalEquity")]
    pub total_equity: String,
    #[serde(rename = "totalWalletBalance")]
    pub total_wallet_balance: String,
    #[serde(rename = "totalMarginBalance")]
    pub total_margin_balance: String,
    #[serde(rename = "totalAvailableBalance")]
    pub total_available_balance: String,
    #[serde(rename = "totalPerpUPL")]
    pub total_perp_upl: String,
    #[serde(rename = "totalInitialMargin")]
    pub total_initial_margin: String,
    #[serde(rename = "totalMaintenanceMargin")]
    pub total_maintenance_margin: String,
    #[serde(rename = "accountType")]
    pub account_type: String,
    pub coin: Vec<BybitCoinBalance>,
}

/// Bybit coin balance
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BybitCoinBalance {
    pub coin: String,
    pub equity: String,
    #[serde(rename = "usdValue")]
    pub usd_value: String,
    #[serde(rename = "walletBalance")]
    pub wallet_balance: String,
    #[serde(rename = "availableToWithdraw")]
    pub available_to_withdraw: String,
    #[serde(rename = "availableToBorrow")]
    pub available_to_borrow: String,
    #[serde(rename = "borrowAmount")]
    pub borrow_amount: String,
    #[serde(rename = "accruedInterest")]
    pub accrued_interest: String,
    #[serde(rename = "totalOrderIM")]
    pub total_order_im: String,
    #[serde(rename = "totalPositionIM")]
    pub total_position_im: String,
    #[serde(rename = "totalPositionMM")]
    pub total_position_mm: String,
    #[serde(rename = "unrealisedPnl")]
    pub unrealised_pnl: String,
    #[serde(rename = "cumRealisedPnl")]
    pub cum_realised_pnl: String,
}

/// Bybit API credentials
#[derive(Debug, Clone)]
pub struct BybitCredentials {
    pub api_key: String,
    pub api_secret: String,
    pub testnet: bool,
}

/// Bybit market type
#[derive(Debug, Clone, PartialEq)]
pub enum BybitMarketType {
    Spot,
    Linear,  // USDT Perpetual
    Inverse, // Inverse Perpetual
    Option,  // USDC Option
}

/// Bybit Ultra exchange connector
pub struct BybitUltra {
    pub credentials: Option<BybitCredentials>,
    pub market_type: BybitMarketType,
    pub order_books: Arc<RwLock<HashMap<String, BybitOrderBook>>>,
    pub positions: Arc<RwLock<HashMap<String, BybitPosition>>>,
    pub orders: Arc<RwLock<HashMap<String, BybitOrder>>>,
    pub wallet: Arc<RwLock<Option<BybitWallet>>>,
    pub ws_stream: Arc<RwLock<Option<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,
    pub http_client: Client,
    pub connection_id: Arc<RwLock<Option<String>>>,
    pub subscriptions: Arc<RwLock<Vec<String>>>,
    pub ping_interval: Duration,
    pub reconnect_delay: Duration,
    pub max_reconnect_attempts: u32,
}

impl BybitUltra {
    /// Create new Bybit Ultra instance
    pub fn new(market_type: BybitMarketType) -> Self {
        Self {
            credentials: None,
            market_type,
            order_books: Arc::new(RwLock::new(HashMap::new())),
            positions: Arc::new(RwLock::new(HashMap::new())),
            orders: Arc::new(RwLock::new(HashMap::new())),
            wallet: Arc::new(RwLock::new(None)),
            ws_stream: Arc::new(RwLock::new(None)),
            http_client: Client::new(),
            connection_id: Arc::new(RwLock::new(None)),
            subscriptions: Arc::new(RwLock::new(Vec::new())),
            ping_interval: Duration::from_secs(20),
            reconnect_delay: Duration::from_secs(5),
            max_reconnect_attempts: 10,
        }
    }

    /// Set credentials for authenticated operations
    pub fn set_credentials(&mut self, api_key: String, api_secret: String, testnet: bool) {
        self.credentials = Some(BybitCredentials {
            api_key,
            api_secret,
            testnet,
        });
    }

    /// Get WebSocket URL based on market type
    fn get_ws_url(&self, is_private: bool) -> &'static str {
        if is_private {
            BYBIT_WS_PRIVATE
        } else {
            match self.market_type {
                BybitMarketType::Spot => BYBIT_WS_PUBLIC,
                BybitMarketType::Linear => BYBIT_WS_LINEAR,
                BybitMarketType::Inverse => BYBIT_WS_LINEAR,
                BybitMarketType::Option => BYBIT_WS_OPTION,
            }
        }
    }

    /// Generate HMAC signature for authentication
    fn generate_signature(&self, timestamp: u64, params: &str) -> Result<String, Box<dyn std::error::Error>> {
        let credentials = self.credentials.as_ref().ok_or("No credentials set")?;
        
        let sign_str = format!("{}{}{}", timestamp, &credentials.api_key, params);
        
        let mut mac = Hmac::<Sha256>::new_from_slice(credentials.api_secret.as_bytes())?;
        mac.update(sign_str.as_bytes());
        let result = mac.finalize();
        
        Ok(hex::encode(result.into_bytes()))
    }

    /// Connect to WebSocket
    pub async fn connect_websocket(&mut self, is_private: bool) -> Result<(), Box<dyn std::error::Error>> {
        let url = self.get_ws_url(is_private);
        info!("Connecting to Bybit WebSocket: {}", url);
        
        let (ws_stream, _) = timeout(Duration::from_secs(10), connect_async(url.as_str())).await??;
        
        let mut stream_guard = self.ws_stream.write();
        *stream_guard = Some(ws_stream);
        drop(stream_guard);
        
        info!("Connected to Bybit WebSocket successfully");
        
        // If private connection, authenticate
        if is_private {
            self.authenticate_websocket().await?;
        }
        
        Ok(())
    }

    /// Authenticate WebSocket connection
    async fn authenticate_websocket(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let credentials = self.credentials.as_ref().ok_or("No credentials set")?;
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as u64;
        
        let expires = timestamp + 10000; // 10 seconds from now
        let signature = self.generate_signature(expires, &format!("GET/realtime{}", expires))?;
        
        let auth_message = serde_json::json!({
            "op": "auth",
            "args": [credentials.api_key, expires, signature]
        });
        
        if let Some(ref mut ws) = *self.ws_stream.write() {
            ws.send(Message::Text(auth_message.to_string())).await?;
        }
        
        info!("Sent authentication message");
        Ok(())
    }

    /// Subscribe to public market data
    pub async fn subscribe_orderbook(&mut self, symbol: &str) -> Result<(), Box<dyn std::error::Error>> {
        let topic = match self.market_type {
            BybitMarketType::Spot => format!("orderbook.1.{}", symbol),
            _ => format!("orderbook.50.{}", symbol),
        };
        
        self.subscribe_topic(&topic).await
    }

    /// Subscribe to trade data
    pub async fn subscribe_trades(&mut self, symbol: &str) -> Result<(), Box<dyn std::error::Error>> {
        let topic = format!("publicTrade.{}", symbol);
        self.subscribe_topic(&topic).await
    }

    /// Subscribe to kline data
    pub async fn subscribe_klines(&mut self, symbol: &str, interval: &str) -> Result<(), Box<dyn std::error::Error>> {
        let topic = format!("kline.{}.{}", interval, symbol);
        self.subscribe_topic(&topic).await
    }

    /// Subscribe to positions (private)
    pub async fn subscribe_positions(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.subscribe_topic("position").await
    }

    /// Subscribe to orders (private)
    pub async fn subscribe_orders(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.subscribe_topic("order").await
    }

    /// Subscribe to wallet (private)
    pub async fn subscribe_wallet(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.subscribe_topic("wallet").await
    }

    /// Generic topic subscription
    async fn subscribe_topic(&mut self, topic: &str) -> Result<(), Box<dyn std::error::Error>> {
        let req_id = format!("sub_{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis());
        
        let subscription = serde_json::json!({
            "op": "subscribe",
            "args": [topic],
            "req_id": req_id
        });
        
        if let Some(ref mut ws) = *self.ws_stream.write() {
            ws.send(Message::Text(subscription.to_string())).await?;
            
            // Add to subscriptions list
            self.subscriptions.write().push(topic.to_string());
            
            info!("Subscribed to topic: {}", topic);
        }
        
        Ok(())
    }

    /// Start message processing loop
    pub async fn start_message_loop(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let ws_stream = Arc::clone(&self.ws_stream);
        let order_books = Arc::clone(&self.order_books);
        let positions = Arc::clone(&self.positions);
        let orders = Arc::clone(&self.orders);
        let wallet = Arc::clone(&self.wallet);
        let connection_id = Arc::clone(&self.connection_id);
        
        tokio::spawn(async move {
            loop {
                let message = {
                    let mut stream_guard = ws_stream.write();
                    if let Some(ref mut ws) = *stream_guard {
                        match timeout(Duration::from_secs(30), ws.next()).await {
                            Ok(Some(Ok(msg))) => Some(msg),
                            Ok(Some(Err(e))) => {
                                error!("WebSocket error: {}", e);
                                break;
                            }
                            Ok(None) => {
                                warn!("WebSocket stream ended");
                                break;
                            }
                            Err(_) => {
                                debug!("WebSocket timeout, sending ping");
                                if let Err(e) = ws.send(Message::Ping(vec![])).await {
                                    error!("Failed to send ping: {}", e);
                                    break;
                                }
                                continue;
                            }
                        }
                    } else {
                        sleep(Duration::from_millis(100)).await;
                        continue;
                    }
                };

                if let Some(msg) = message {
                    match msg {
                        Message::Text(text) => {
                            if let Err(e) = Self::process_message(
                                &text,
                                &order_books,
                                &positions,
                                &orders,
                                &wallet,
                                &connection_id,
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
        });

        Ok(())
    }

    /// Process incoming WebSocket message
    async fn process_message(
        text: &str,
        order_books: &Arc<RwLock<HashMap<String, BybitOrderBook>>>,
        positions: &Arc<RwLock<HashMap<String, BybitPosition>>>,
        orders: &Arc<RwLock<HashMap<String, BybitOrder>>>,
        wallet: &Arc<RwLock<Option<BybitWallet>>>,
        connection_id: &Arc<RwLock<Option<String>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let message: BybitMessage = serde_json::from_str(text)?;
        
        // Handle connection confirmation
        if let (Some(success), Some(conn_id)) = (message.success, &message.conn_id) {
            if success {
                *connection_id.write() = Some(conn_id.clone());
                info!("WebSocket connection confirmed: {}", conn_id);
            }
        }

        // Handle subscription confirmation
        if message.op.as_deref() == Some("subscribe") {
            if message.success == Some(true) {
                info!("Subscription confirmed: {}", message.ret_msg.unwrap_or_default());
            } else {
                warn!("Subscription failed: {}", message.ret_msg.unwrap_or_default());
            }
        }

        // Process data updates
        if let (Some(topic), Some(data)) = (&message.topic, &message.data) {
            match topic.as_str() {
                t if t.starts_with("orderbook.") => {
                    Self::process_orderbook_update(t, data, order_books).await?;
                }
                t if t.starts_with("publicTrade.") => {
                    Self::process_trade_update(t, data).await?;
                }
                t if t.starts_with("kline.") => {
                    Self::process_kline_update(t, data).await?;
                }
                "position" => {
                    Self::process_position_update(data, positions).await?;
                }
                "order" => {
                    Self::process_order_update(data, orders).await?;
                }
                "wallet" => {
                    Self::process_wallet_update(data, wallet).await?;
                }
                _ => {
                    debug!("Unhandled topic: {}", topic);
                }
            }
        }

        Ok(())
    }

    /// Process order book updates
    async fn process_orderbook_update(
        topic: &str,
        data: &serde_json::Value,
        order_books: &Arc<RwLock<HashMap<String, BybitOrderBook>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(updates) = serde_json::from_value::<Vec<BybitOrderBook>>(data.clone()) {
            for update in updates {
                let symbol = update.s.clone();
                order_books.write().insert(symbol.clone(), update);
                debug!("Updated order book for {}", symbol);
            }
        }
        Ok(())
    }

    /// Process trade updates
    async fn process_trade_update(
        topic: &str,
        data: &serde_json::Value,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(trades) = serde_json::from_value::<Vec<BybitTrade>>(data.clone()) {
            for trade in trades {
                debug!("Trade update: {} {} @ {} ({})", 
                    trade.s, trade.v, trade.p, trade.S);
            }
        }
        Ok(())
    }

    /// Process kline updates
    async fn process_kline_update(
        topic: &str,
        data: &serde_json::Value,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(klines) = serde_json::from_value::<Vec<BybitKline>>(data.clone()) {
            for kline in klines {
                debug!("Kline update: {} OHLC({},{},{},{})", 
                    kline.interval, kline.open, kline.high, kline.low, kline.close);
            }
        }
        Ok(())
    }

    /// Process position updates
    async fn process_position_update(
        data: &serde_json::Value,
        positions: &Arc<RwLock<HashMap<String, BybitPosition>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(position_updates) = serde_json::from_value::<Vec<BybitPosition>>(data.clone()) {
            let mut positions_guard = positions.write();
            for position in position_updates {
                let key = format!("{}-{}", position.symbol, position.position_idx);
                positions_guard.insert(key.clone(), position);
                debug!("Updated position: {}", key);
            }
        }
        Ok(())
    }

    /// Process order updates
    async fn process_order_update(
        data: &serde_json::Value,
        orders: &Arc<RwLock<HashMap<String, BybitOrder>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(order_updates) = serde_json::from_value::<Vec<BybitOrder>>(data.clone()) {
            let mut orders_guard = orders.write();
            for order in order_updates {
                orders_guard.insert(order.order_id.clone(), order);
                debug!("Updated order: {}", order.order_id);
            }
        }
        Ok(())
    }

    /// Process wallet updates
    async fn process_wallet_update(
        data: &serde_json::Value,
        wallet: &Arc<RwLock<Option<BybitWallet>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(wallet_updates) = serde_json::from_value::<Vec<BybitWallet>>(data.clone()) {
            if let Some(wallet_data) = wallet_updates.into_iter().next() {
                *wallet.write() = Some(wallet_data);
                debug!("Updated wallet information");
            }
        }
        Ok(())
    }

    /// Place order via REST API
    pub async fn place_order(
        &self,
        symbol: &str,
        side: &str,
        order_type: &str,
        qty: &str,
        price: Option<&str>,
        time_in_force: Option<&str>,
        reduce_only: Option<bool>,
        close_on_trigger: Option<bool>,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let credentials = self.credentials.as_ref().ok_or("No credentials set")?;
        
        let mut params = serde_json::json!({
            "category": match self.market_type {
                BybitMarketType::Spot => "spot",
                BybitMarketType::Linear => "linear",
                BybitMarketType::Inverse => "inverse",
                BybitMarketType::Option => "option",
            },
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty,
        });

        if let Some(p) = price {
            params["price"] = serde_json::Value::String(p.to_string());
        }
        if let Some(tif) = time_in_force {
            params["timeInForce"] = serde_json::Value::String(tif.to_string());
        }
        if let Some(ro) = reduce_only {
            params["reduceOnly"] = serde_json::Value::Bool(*ro);
        }
        if let Some(cot) = close_on_trigger {
            params["closeOnTrigger"] = serde_json::Value::Bool(*cot);
        }

        self.make_authenticated_request("POST", "/v5/order/create", Some(params)).await
    }

    /// Cancel order via REST API
    pub async fn cancel_order(
        &self,
        symbol: &str,
        order_id: Option<&str>,
        order_link_id: Option<&str>,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut params = serde_json::json!({
            "category": match self.market_type {
                BybitMarketType::Spot => "spot",
                BybitMarketType::Linear => "linear",
                BybitMarketType::Inverse => "inverse",
                BybitMarketType::Option => "option",
            },
            "symbol": symbol,
        });

        if let Some(oid) = order_id {
            params["orderId"] = serde_json::Value::String(oid.to_string());
        }
        if let Some(olid) = order_link_id {
            params["orderLinkId"] = serde_json::Value::String(olid.to_string());
        }

        self.make_authenticated_request("POST", "/v5/order/cancel", Some(params)).await
    }

    /// Get account info
    pub async fn get_account_info(&self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        self.make_authenticated_request("GET", "/v5/account/info", None).await
    }

    /// Get wallet balance
    pub async fn get_wallet_balance(&self, account_type: &str) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let params = serde_json::json!({
            "accountType": account_type,
        });
        
        self.make_authenticated_request("GET", "/v5/account/wallet-balance", Some(params)).await
    }

    /// Get positions
    pub async fn get_positions(&self, symbol: Option<&str>) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut params = serde_json::json!({
            "category": match self.market_type {
                BybitMarketType::Spot => "spot",
                BybitMarketType::Linear => "linear",
                BybitMarketType::Inverse => "inverse",
                BybitMarketType::Option => "option",
            },
        });

        if let Some(sym) = symbol {
            params["symbol"] = serde_json::Value::String(sym.to_string());
        }

        self.make_authenticated_request("GET", "/v5/position/list", Some(params)).await
    }

    /// Get open orders
    pub async fn get_open_orders(&self, symbol: Option<&str>) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut params = serde_json::json!({
            "category": match self.market_type {
                BybitMarketType::Spot => "spot",
                BybitMarketType::Linear => "linear",
                BybitMarketType::Inverse => "inverse",
                BybitMarketType::Option => "option",
            },
        });

        if let Some(sym) = symbol {
            params["symbol"] = serde_json::Value::String(sym.to_string());
        }

        self.make_authenticated_request("GET", "/v5/order/realtime", Some(params)).await
    }

    /// Make authenticated REST API request
    async fn make_authenticated_request(
        &self,
        method: &str,
        endpoint: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let credentials = self.credentials.as_ref().ok_or("No credentials set")?;
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis();

        let recv_window = 5000u64;
        let params_str = if let Some(p) = &params {
            serde_json::to_string(p)?
        } else {
            String::new()
        };

        // Create signature string
        let sign_str = format!("{}{}{}{}", timestamp, &credentials.api_key, recv_window, params_str);
        
        // Generate HMAC signature
        let mut mac = Hmac::<Sha256>::new_from_slice(credentials.api_secret.as_bytes())?;
        mac.update(sign_str.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());

        let url = if credentials.testnet {
            format!("https://api-testnet.bybit.com{}", endpoint)
        } else {
            format!("{}{}", BYBIT_REST_URL, endpoint)
        };

        let mut request = match method {
            "GET" => self.http_client.get(&url),
            "POST" => self.http_client.post(&url),
            "PUT" => self.http_client.put(&url),
            "DELETE" => self.http_client.delete(&url),
            _ => return Err("Unsupported HTTP method".into()),
        };

        request = request
            .header("X-BAPI-API-KEY", &credentials.api_key)
            .header("X-BAPI-SIGN", signature)
            .header("X-BAPI-SIGN-TYPE", "2")
            .header("X-BAPI-TIMESTAMP", timestamp.to_string())
            .header("X-BAPI-RECV-WINDOW", recv_window.to_string())
            .header("Content-Type", "application/json");

        if method != "GET" && params.is_some() {
            request = request.json(&params);
        } else if method == "GET" && params.is_some() {
            // Add query parameters for GET requests
            if let Some(p) = params {
                if let serde_json::Value::Object(map) = p {
                    let mut url_parsed = Url::parse(&url)?;
                    {
                        let mut query_pairs = url_parsed.query_pairs_mut();
                        for (key, value) in map.iter() {
                            query_pairs.append_pair(key, &value.to_string().trim_matches('"'));
                        }
                    }
                    request = self.http_client.get(url_parsed.as_str());
                    request = request
                        .header("X-BAPI-API-KEY", &credentials.api_key)
                        .header("X-BAPI-SIGN", signature)
                        .header("X-BAPI-SIGN-TYPE", "2")
                        .header("X-BAPI-TIMESTAMP", timestamp.to_string())
                        .header("X-BAPI-RECV-WINDOW", recv_window.to_string());
                }
            }
        }

        let response = request.send().await?;
        let response_text = response.text().await?;
        
        debug!("API Response: {}", response_text);
        
        let json_response: serde_json::Value = serde_json::from_str(&response_text)?;
        
        Ok(json_response)
    }

    /// Reconnect WebSocket with exponential backoff
    pub async fn reconnect_websocket(&mut self, is_private: bool, attempt: u32) -> Result<(), Box<dyn std::error::Error>> {
        if attempt > self.max_reconnect_attempts {
            return Err("Max reconnection attempts reached".into());
        }

        let delay = self.reconnect_delay * 2_u32.pow(attempt - 1);
        warn!("Reconnecting in {:?} (attempt {})", delay, attempt);
        sleep(delay).await;

        match self.connect_websocket(is_private).await {
            Ok(_) => {
                info!("Reconnected successfully");
                
                // Re-subscribe to all topics
                let subscriptions = self.subscriptions.read().clone();
                for topic in subscriptions {
                    if let Err(e) = self.subscribe_topic(&topic).await {
                        error!("Failed to resubscribe to {}: {}", topic, e);
                    }
                }
                
                Ok(())
            }
            Err(e) => {
                error!("Reconnection attempt {} failed: {}", attempt, e);
                self.reconnect_websocket(is_private, attempt + 1).await
            }
        }
    }

    /// Get current order book for symbol
    pub fn get_orderbook(&self, symbol: &str) -> Option<BybitOrderBook> {
        self.order_books.read().get(symbol).cloned()
    }

    /// Get current positions
    pub fn get_current_positions(&self) -> HashMap<String, BybitPosition> {
        self.positions.read().clone()
    }

    /// Get current orders
    pub fn get_current_orders(&self) -> HashMap<String, BybitOrder> {
        self.orders.read().clone()
    }

    /// Get current wallet info
    pub fn get_current_wallet(&self) -> Option<BybitWallet> {
        self.wallet.read().clone()
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
            info!("WebSocket connection closed");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_bybit_ultra_creation() {
        let bybit = BybitUltra::new(BybitMarketType::Linear);
        assert_eq!(bybit.market_type, BybitMarketType::Linear);
        assert!(!bybit.is_connected());
    }

    #[tokio::test]
    async fn test_credential_setting() {
        let mut bybit = BybitUltra::new(BybitMarketType::Spot);
        bybit.set_credentials("test_key".to_string(), "test_secret".to_string(), true);
        
        assert!(bybit.credentials.is_some());
        let creds = bybit.credentials.unwrap();
        assert_eq!(creds.api_key, "test_key");
        assert_eq!(creds.api_secret, "test_secret");
        assert!(creds.testnet);
    }

    #[tokio::test]
    async fn test_websocket_url_selection() {
        let bybit_spot = BybitUltra::new(BybitMarketType::Spot);
        assert_eq!(bybit_spot.get_ws_url(false), BYBIT_WS_PUBLIC);
        assert_eq!(bybit_spot.get_ws_url(true), BYBIT_WS_PRIVATE);

        let bybit_linear = BybitUltra::new(BybitMarketType::Linear);
        assert_eq!(bybit_linear.get_ws_url(false), BYBIT_WS_LINEAR);

        let bybit_option = BybitUltra::new(BybitMarketType::Option);
        assert_eq!(bybit_option.get_ws_url(false), BYBIT_WS_OPTION);
    }

    #[tokio::test]
    async fn test_signature_generation() {
        let mut bybit = BybitUltra::new(BybitMarketType::Linear);
        bybit.set_credentials("test_key".to_string(), "test_secret".to_string(), false);
        
        let timestamp = 1234567890u64;
        let params = r#"{"symbol":"BTCUSDT","side":"Buy"}"#;
        
        let signature = bybit.generate_signature(timestamp, params);
        assert!(signature.is_ok());
        
        let sig = signature.unwrap();
        assert_eq!(sig.len(), 64); // SHA256 hex string length
    }

    #[tokio::test]
    async fn test_message_parsing() {
        let json_msg = r#"{
            "success": true,
            "ret_msg": "subscribe",
            "conn_id": "test-connection-id",
            "req_id": "test-req-id",
            "op": "subscribe"
        }"#;
        
        let msg: Result<BybitMessage, _> = serde_json::from_str(json_msg);
        assert!(msg.is_ok());
        
        let parsed = msg.unwrap();
        assert_eq!(parsed.success, Some(true));
        assert_eq!(parsed.ret_msg, Some("subscribe".to_string()));
        assert_eq!(parsed.conn_id, Some("test-connection-id".to_string()));
    }
}