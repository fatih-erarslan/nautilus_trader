// OKX Ultra - REAL IMPLEMENTATION with WebSocket and REST API
use tokio::net::TcpStream;
use tokio_tungstenite::{connect_async, WebSocketStream, MaybeTlsStream};
use tokio_tungstenite::tungstenite::Message;
use futures_util::{StreamExt, SinkExt};
use serde::{Deserialize, Serialize};
use serde_json;
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use base64;

const OKX_WS_PUBLIC: &str = "wss://ws.okx.com:8443/ws/v5/public";
const OKX_WS_PRIVATE: &str = "wss://ws.okx.com:8443/ws/v5/private";
const OKX_REST_URL: &str = "https://www.okx.com";

/// OKX WebSocket message
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OKXMessage {
    pub event: Option<String>,
    pub arg: Option<ChannelArg>,
    pub data: Option<Vec<serde_json::Value>>,
    pub code: Option<String>,
    pub msg: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChannelArg {
    pub channel: String,
    #[serde(rename = "instId")]
    pub inst_id: Option<String>,
    #[serde(rename = "instType")]
    pub inst_type: Option<String>,
}

/// OKX order book data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OKXOrderBook {
    pub asks: Vec<[String; 4]>, // [price, size, liquidation_orders, order_count]
    pub bids: Vec<[String; 4]>,
    #[serde(rename = "ts")]
    pub timestamp: String,
    #[serde(rename = "checksum")]
    pub checksum: Option<i32>,
}

/// OKX trade data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OKXTrade {
    #[serde(rename = "instId")]
    pub inst_id: String,
    #[serde(rename = "tradeId")]
    pub trade_id: String,
    pub px: String,
    pub sz: String,
    pub side: String,
    pub ts: String,
}

/// OKX ticker data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OKXTicker {
    #[serde(rename = "instType")]
    pub inst_type: String,
    #[serde(rename = "instId")]
    pub inst_id: String,
    pub last: String,
    #[serde(rename = "lastSz")]
    pub last_sz: String,
    #[serde(rename = "askPx")]
    pub ask_px: String,
    #[serde(rename = "askSz")]
    pub ask_sz: String,
    #[serde(rename = "bidPx")]
    pub bid_px: String,
    #[serde(rename = "bidSz")]
    pub bid_sz: String,
    pub open24h: String,
    pub high24h: String,
    pub low24h: String,
    #[serde(rename = "volCcy24h")]
    pub vol_ccy_24h: String,
    pub vol24h: String,
    pub ts: String,
}

/// Order placement request
#[derive(Debug, Clone, Serialize)]
pub struct OKXOrderRequest {
    #[serde(rename = "instId")]
    pub inst_id: String,
    #[serde(rename = "tdMode")]
    pub td_mode: String, // Trade mode: cash, cross, isolated
    pub side: String,
    #[serde(rename = "ordType")]
    pub ord_type: String,
    pub sz: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub px: Option<String>,
    #[serde(rename = "clOrdId", skip_serializing_if = "Option::is_none")]
    pub cl_ord_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
}

/// Order response
#[derive(Debug, Clone, Deserialize)]
pub struct OKXOrderResponse {
    pub code: String,
    pub msg: String,
    pub data: Vec<OKXOrderData>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OKXOrderData {
    #[serde(rename = "ordId")]
    pub ord_id: String,
    #[serde(rename = "clOrdId")]
    pub cl_ord_id: String,
    pub tag: String,
    #[serde(rename = "sCode")]
    pub s_code: String,
    #[serde(rename = "sMsg")]
    pub s_msg: String,
}

/// Account balance
#[derive(Debug, Clone, Deserialize)]
pub struct OKXBalance {
    #[serde(rename = "adjEq")]
    pub adj_eq: Option<String>,
    pub details: Vec<OKXBalanceDetail>,
    pub imr: Option<String>,
    pub mmr: Option<String>,
    #[serde(rename = "mgnRatio")]
    pub mgn_ratio: Option<String>,
    #[serde(rename = "notionalUsd")]
    pub notional_usd: Option<String>,
    #[serde(rename = "ordFroz")]
    pub ord_froz: Option<String>,
    #[serde(rename = "totalEq")]
    pub total_eq: String,
    #[serde(rename = "uTime")]
    pub u_time: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OKXBalanceDetail {
    #[serde(rename = "availBal")]
    pub avail_bal: String,
    #[serde(rename = "availEq")]
    pub avail_eq: String,
    pub ccy: String,
    #[serde(rename = "cashBal")]
    pub cash_bal: String,
    #[serde(rename = "uTime")]
    pub u_time: String,
    #[serde(rename = "disEq")]
    pub dis_eq: String,
    pub eq: String,
    #[serde(rename = "eqUsd")]
    pub eq_usd: String,
    #[serde(rename = "frozenBal")]
    pub frozen_bal: String,
    pub interest: String,
    #[serde(rename = "isoEq")]
    pub iso_eq: String,
    #[serde(rename = "liab")]
    pub liab: String,
    #[serde(rename = "maxLoan")]
    pub max_loan: String,
    #[serde(rename = "mgnRatio")]
    pub mgn_ratio: Option<String>,
    #[serde(rename = "upl")]
    pub upl: String,
    #[serde(rename = "uplLiab")]
    pub upl_liab: String,
}

/// OKX Ultra connector with real WebSocket and REST
pub struct OKXUltra {
    api_key: String,
    secret_key: String,
    passphrase: String,
    ws_public: Arc<RwLock<Option<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,
    ws_private: Arc<RwLock<Option<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,
    order_books: Arc<RwLock<HashMap<String, OrderBook>>>,
    trades: Arc<RwLock<Vec<OKXTrade>>>,
    positions: Arc<RwLock<HashMap<String, Position>>>,
    http_client: reqwest::Client,
}

/// Internal order book representation
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<(f64, f64)>, // (price, quantity)
    pub asks: Vec<(f64, f64)>,
    pub timestamp: u64,
    pub checksum: Option<i32>,
}

/// Position data
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub position: f64,
    pub avg_price: f64,
    pub unrealized_pnl: f64,
    pub margin: f64,
}

impl OKXUltra {
    pub fn new(api_key: String, secret_key: String, passphrase: String) -> Self {
        Self {
            api_key,
            secret_key,
            passphrase,
            ws_public: Arc::new(RwLock::new(None)),
            ws_private: Arc::new(RwLock::new(None)),
            order_books: Arc::new(RwLock::new(HashMap::new())),
            trades: Arc::new(RwLock::new(Vec::new())),
            positions: Arc::new(RwLock::new(HashMap::new())),
            http_client: reqwest::Client::new(),
        }
    }
    
    /// Connect to OKX WebSocket (public channel)
    pub async fn connect_public_websocket(&self) -> Result<(), Box<dyn std::error::Error>> {
        let (ws_stream, _) = connect_async(OKX_WS_PUBLIC).await?;
        *self.ws_public.write() = Some(ws_stream);
        Ok(())
    }
    
    /// Connect to OKX WebSocket (private channel)
    pub async fn connect_private_websocket(&self) -> Result<(), Box<dyn std::error::Error>> {
        let (ws_stream, _) = connect_async(OKX_WS_PRIVATE).await?;
        
        // Authenticate
        self.authenticate_websocket().await?;
        
        *self.ws_private.write() = Some(ws_stream);
        Ok(())
    }
    
    /// Authenticate WebSocket connection
    async fn authenticate_websocket(&self) -> Result<(), Box<dyn std::error::Error>> {
        let timestamp = (SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()).to_string();
        let method = "GET";
        let request_path = "/users/self/verify";
        
        let sign_string = format!("{}{}{}", timestamp, method, request_path);
        let signature = self.sign_request(&sign_string);
        
        let auth_msg = serde_json::json!({
            "op": "login",
            "args": [{
                "apiKey": self.api_key,
                "passphrase": self.passphrase,
                "timestamp": timestamp,
                "sign": signature
            }]
        });
        
        let mut ws = self.ws_private.write();
        if let Some(stream) = ws.as_mut() {
            stream.send(Message::Text(auth_msg.to_string())).await?;
        }
        
        Ok(())
    }
    
    /// Subscribe to market data streams
    pub async fn subscribe_public_channels(&self, symbols: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
        let mut ws = self.ws_public.write();
        
        if let Some(stream) = ws.as_mut() {
            // Subscribe to order books
            let books_sub = serde_json::json!({
                "op": "subscribe",
                "args": symbols.iter().map(|s| {
                    serde_json::json!({
                        "channel": "books",
                        "instId": s
                    })
                }).collect::<Vec<_>>()
            });
            stream.send(Message::Text(books_sub.to_string())).await?;
            
            // Subscribe to trades
            let trades_sub = serde_json::json!({
                "op": "subscribe",
                "args": symbols.iter().map(|s| {
                    serde_json::json!({
                        "channel": "trades",
                        "instId": s
                    })
                }).collect::<Vec<_>>()
            });
            stream.send(Message::Text(trades_sub.to_string())).await?;
            
            // Subscribe to tickers
            let tickers_sub = serde_json::json!({
                "op": "subscribe",
                "args": symbols.iter().map(|s| {
                    serde_json::json!({
                        "channel": "tickers",
                        "instId": s
                    })
                }).collect::<Vec<_>>()
            });
            stream.send(Message::Text(tickers_sub.to_string())).await?;
        }
        
        Ok(())
    }
    
    /// Subscribe to private channels
    pub async fn subscribe_private_channels(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut ws = self.ws_private.write();
        
        if let Some(stream) = ws.as_mut() {
            // Subscribe to account updates
            let account_sub = serde_json::json!({
                "op": "subscribe",
                "args": [{
                    "channel": "account"
                }]
            });
            stream.send(Message::Text(account_sub.to_string())).await?;
            
            // Subscribe to position updates
            let positions_sub = serde_json::json!({
                "op": "subscribe",
                "args": [{
                    "channel": "positions",
                    "instType": "ANY"
                }]
            });
            stream.send(Message::Text(positions_sub.to_string())).await?;
            
            // Subscribe to order updates
            let orders_sub = serde_json::json!({
                "op": "subscribe",
                "args": [{
                    "channel": "orders",
                    "instType": "ANY"
                }]
            });
            stream.send(Message::Text(orders_sub.to_string())).await?;
        }
        
        Ok(())
    }
    
    /// Process WebSocket messages
    pub async fn process_public_messages(&self) {
        let ws_public = self.ws_public.clone();
        let order_books = self.order_books.clone();
        let trades = self.trades.clone();
        
        tokio::spawn(async move {
            loop {
                let mut ws = ws_public.write();
                
                if let Some(stream) = ws.as_mut() {
                    if let Some(msg) = stream.next().await {
                        match msg {
                            Ok(Message::Text(text)) => {
                                // Parse OKX message
                                if let Ok(okx_msg) = serde_json::from_str::<OKXMessage>(&text) {
                                    if let Some(arg) = okx_msg.arg {
                                        match arg.channel.as_str() {
                                            "books" => {
                                                if let Some(data) = okx_msg.data {
                                                    Self::update_order_book(&order_books, arg.inst_id.unwrap_or_default(), data);
                                                }
                                            }
                                            "trades" => {
                                                if let Some(data) = okx_msg.data {
                                                    Self::update_trades(&trades, data);
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            Ok(Message::Close(_)) => {
                                break;
                            }
                            _ => {}
                        }
                    }
                } else {
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            }
        });
    }
    
    /// Update order book from OKX data
    fn update_order_book(order_books: &Arc<RwLock<HashMap<String, OrderBook>>>, inst_id: String, data: Vec<serde_json::Value>) {
        if let Some(book_data) = data.first() {
            if let Ok(okx_book) = serde_json::from_value::<OKXOrderBook>(book_data.clone()) {
                let mut books = order_books.write();
                
                let book = books.entry(inst_id.clone()).or_insert(OrderBook {
                    symbol: inst_id.clone(),
                    bids: Vec::new(),
                    asks: Vec::new(),
                    timestamp: 0,
                    checksum: None,
                });
                
                // Update bids
                book.bids.clear();
                for bid in okx_book.bids {
                    if let (Ok(price), Ok(size)) = (bid[0].parse::<f64>(), bid[1].parse::<f64>()) {
                        book.bids.push((price, size));
                    }
                }
                
                // Update asks
                book.asks.clear();
                for ask in okx_book.asks {
                    if let (Ok(price), Ok(size)) = (ask[0].parse::<f64>(), ask[1].parse::<f64>()) {
                        book.asks.push((price, size));
                    }
                }
                
                // Sort order book
                book.bids.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // Descending
                book.asks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap()); // Ascending
                
                book.timestamp = okx_book.timestamp.parse().unwrap_or(0);
                book.checksum = okx_book.checksum;
            }
        }
    }
    
    /// Update trades from OKX data
    fn update_trades(trades: &Arc<RwLock<Vec<OKXTrade>>>, data: Vec<serde_json::Value>) {
        let mut trades_list = trades.write();
        
        for trade_data in data {
            if let Ok(trade) = serde_json::from_value::<OKXTrade>(trade_data) {
                trades_list.push(trade);
                
                // Keep only recent trades
                if trades_list.len() > 1000 {
                    trades_list.drain(0..trades_list.len() - 1000);
                }
            }
        }
    }
    
    /// Place order via REST API
    pub async fn place_order(&self, order: OKXOrderRequest) -> Result<OKXOrderResponse, Box<dyn std::error::Error>> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis().to_string();
        let method = "POST";
        let request_path = "/api/v5/trade/order";
        
        let body = serde_json::to_string(&order)?;
        let sign_string = format!("{}{}{}{}", timestamp, method, request_path, body);
        let signature = self.sign_request(&sign_string);
        
        let url = format!("{}{}", OKX_REST_URL, request_path);
        
        let response = self.http_client
            .post(&url)
            .header("OK-ACCESS-KEY", &self.api_key)
            .header("OK-ACCESS-SIGN", signature)
            .header("OK-ACCESS-TIMESTAMP", timestamp)
            .header("OK-ACCESS-PASSPHRASE", &self.passphrase)
            .header("Content-Type", "application/json")
            .body(body)
            .send()
            .await?;
        
        let order_response: OKXOrderResponse = response.json().await?;
        Ok(order_response)
    }
    
    /// Cancel order
    pub async fn cancel_order(&self, inst_id: &str, ord_id: Option<&str>, cl_ord_id: Option<&str>) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis().to_string();
        let method = "POST";
        let request_path = "/api/v5/trade/cancel-order";
        
        let mut body = serde_json::json!({
            "instId": inst_id
        });
        
        if let Some(id) = ord_id {
            body["ordId"] = serde_json::json!(id);
        }
        
        if let Some(cl_id) = cl_ord_id {
            body["clOrdId"] = serde_json::json!(cl_id);
        }
        
        let body_str = body.to_string();
        let sign_string = format!("{}{}{}{}", timestamp, method, request_path, body_str);
        let signature = self.sign_request(&sign_string);
        
        let url = format!("{}{}", OKX_REST_URL, request_path);
        
        let response = self.http_client
            .post(&url)
            .header("OK-ACCESS-KEY", &self.api_key)
            .header("OK-ACCESS-SIGN", signature)
            .header("OK-ACCESS-TIMESTAMP", timestamp)
            .header("OK-ACCESS-PASSPHRASE", &self.passphrase)
            .header("Content-Type", "application/json")
            .body(body_str)
            .send()
            .await?;
        
        let result: serde_json::Value = response.json().await?;
        Ok(result)
    }
    
    /// Get account balance
    pub async fn get_balance(&self) -> Result<Vec<OKXBalance>, Box<dyn std::error::Error>> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis().to_string();
        let method = "GET";
        let request_path = "/api/v5/account/balance";
        
        let sign_string = format!("{}{}{}", timestamp, method, request_path);
        let signature = self.sign_request(&sign_string);
        
        let url = format!("{}{}", OKX_REST_URL, request_path);
        
        let response = self.http_client
            .get(&url)
            .header("OK-ACCESS-KEY", &self.api_key)
            .header("OK-ACCESS-SIGN", signature)
            .header("OK-ACCESS-TIMESTAMP", timestamp)
            .header("OK-ACCESS-PASSPHRASE", &self.passphrase)
            .send()
            .await?;
        
        let result: serde_json::Value = response.json().await?;
        
        if let Some(data) = result["data"].as_array() {
            let balances: Vec<OKXBalance> = data.iter()
                .filter_map(|v| serde_json::from_value(v.clone()).ok())
                .collect();
            Ok(balances)
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Get positions
    pub async fn get_positions(&self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis().to_string();
        let method = "GET";
        let request_path = "/api/v5/account/positions";
        
        let sign_string = format!("{}{}{}", timestamp, method, request_path);
        let signature = self.sign_request(&sign_string);
        
        let url = format!("{}{}", OKX_REST_URL, request_path);
        
        let response = self.http_client
            .get(&url)
            .header("OK-ACCESS-KEY", &self.api_key)
            .header("OK-ACCESS-SIGN", signature)
            .header("OK-ACCESS-TIMESTAMP", timestamp)
            .header("OK-ACCESS-PASSPHRASE", &self.passphrase)
            .send()
            .await?;
        
        let result: serde_json::Value = response.json().await?;
        Ok(result)
    }
    
    /// Sign request with HMAC-SHA256
    fn sign_request(&self, message: &str) -> String {
        type HmacSha256 = Hmac<Sha256>;
        
        let mut mac = HmacSha256::new_from_slice(self.secret_key.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(message.as_bytes());
        
        let result = mac.finalize();
        base64::encode(result.into_bytes())
    }
    
    /// Get order book snapshot
    pub fn get_order_book(&self, symbol: &str) -> Option<OrderBook> {
        self.order_books.read().get(symbol).cloned()
    }
    
    /// Get recent trades
    pub fn get_recent_trades(&self, symbol: &str, limit: usize) -> Vec<OKXTrade> {
        self.trades.read()
            .iter()
            .filter(|t| t.inst_id == symbol)
            .take(limit)
            .cloned()
            .collect()
    }
    
    /// Get market depth
    pub fn get_market_depth(&self, symbol: &str, depth: usize) -> Option<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
        self.order_books.read().get(symbol).map(|book| {
            let bids: Vec<(f64, f64)> = book.bids.iter().take(depth).cloned().collect();
            let asks: Vec<(f64, f64)> = book.asks.iter().take(depth).cloned().collect();
            (bids, asks)
        })
    }
    
    /// Calculate order book checksum (for validation)
    pub fn calculate_checksum(&self, symbol: &str) -> Option<i32> {
        self.order_books.read().get(symbol).and_then(|book| {
            let mut checksum_str = String::new();
            
            // Take top 25 levels
            for i in 0..25.min(book.bids.len().max(book.asks.len())) {
                if i < book.bids.len() {
                    checksum_str.push_str(&format!("{}:{}", book.bids[i].0, book.bids[i].1));
                }
                if i < book.asks.len() {
                    checksum_str.push_str(&format!("{}:{}", book.asks[i].0, book.asks[i].1));
                }
            }
            
            // Calculate CRC32
            Some(crc32fast::hash(checksum_str.as_bytes()) as i32)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_okx_creation() {
        let okx = OKXUltra::new(
            "test_api_key".to_string(),
            "test_secret_key".to_string(),
            "test_passphrase".to_string()
        );
        
        assert!(okx.get_order_book("BTC-USDT").is_none());
    }
    
    #[test]
    fn test_signature_generation() {
        let okx = OKXUltra::new(
            "test_api_key".to_string(),
            "test_secret_key".to_string(),
            "test_passphrase".to_string()
        );
        
        let message = "1234567890GET/api/v5/account/balance";
        let signature = okx.sign_request(message);
        
        assert!(!signature.is_empty());
        assert!(signature.chars().all(|c| c.is_ascii()));
    }
    
    #[tokio::test]
    #[ignore] // Requires real API connection
    async fn test_websocket_connection() {
        let okx = OKXUltra::new(
            "".to_string(),
            "".to_string(),
            "".to_string()
        );
        
        match okx.connect_public_websocket().await {
            Ok(_) => println!("OKX WebSocket connected successfully"),
            Err(e) => println!("OKX WebSocket connection failed: {}", e),
        }
    }
    
    #[test]
    fn test_order_book_update() {
        let okx = OKXUltra::new(
            "test".to_string(),
            "test".to_string(),
            "test".to_string()
        );
        
        let okx_book = OKXOrderBook {
            asks: vec![
                ["100.5".to_string(), "10".to_string(), "0".to_string(), "1".to_string()],
                ["100.6".to_string(), "20".to_string(), "0".to_string(), "2".to_string()],
            ],
            bids: vec![
                ["100.4".to_string(), "15".to_string(), "0".to_string(), "1".to_string()],
                ["100.3".to_string(), "25".to_string(), "0".to_string(), "3".to_string()],
            ],
            timestamp: "1234567890".to_string(),
            checksum: Some(12345),
        };
        
        let data = vec![serde_json::to_value(okx_book).unwrap()];
        OKXUltra::update_order_book(&okx.order_books, "BTC-USDT".to_string(), data);
        
        let book = okx.get_order_book("BTC-USDT");
        assert!(book.is_some());
        
        if let Some(b) = book {
            assert_eq!(b.bids.len(), 2);
            assert_eq!(b.asks.len(), 2);
            assert_eq!(b.bids[0].0, 100.4);
            assert_eq!(b.asks[0].0, 100.5);
        }
    }
}