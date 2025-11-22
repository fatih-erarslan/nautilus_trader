// Binance Ultra - REAL IMPLEMENTATION with WebSocket and REST API
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

const BINANCE_WS_URL: &str = "wss://stream.binance.com:9443/ws";
const BINANCE_REST_URL: &str = "https://api.binance.com";

/// Binance order book update
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DepthUpdate {
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "E")]
    pub event_time: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "U")]
    pub first_update_id: u64,
    #[serde(rename = "u")]
    pub final_update_id: u64,
    #[serde(rename = "b")]
    pub bids: Vec<[String; 2]>,
    #[serde(rename = "a")]
    pub asks: Vec<[String; 2]>,
}

/// Binance trade data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Trade {
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "E")]
    pub event_time: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "t")]
    pub trade_id: u64,
    #[serde(rename = "p")]
    pub price: String,
    #[serde(rename = "q")]
    pub quantity: String,
    #[serde(rename = "b")]
    pub buyer_order_id: u64,
    #[serde(rename = "a")]
    pub seller_order_id: u64,
    #[serde(rename = "T")]
    pub trade_time: u64,
    #[serde(rename = "m")]
    pub is_buyer_maker: bool,
}

/// Binance ticker data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MiniTicker {
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "E")]
    pub event_time: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "c")]
    pub close: String,
    #[serde(rename = "o")]
    pub open: String,
    #[serde(rename = "h")]
    pub high: String,
    #[serde(rename = "l")]
    pub low: String,
    #[serde(rename = "v")]
    pub volume: String,
    #[serde(rename = "q")]
    pub quote_volume: String,
}

/// Order placement request
#[derive(Debug, Clone, Serialize)]
pub struct OrderRequest {
    pub symbol: String,
    pub side: String,
    #[serde(rename = "type")]
    pub order_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price: Option<String>,
    #[serde(rename = "timeInForce", skip_serializing_if = "Option::is_none")]
    pub time_in_force: Option<String>,
    #[serde(rename = "recvWindow")]
    pub recv_window: u64,
    pub timestamp: u64,
}

/// Order response
#[derive(Debug, Clone, Deserialize)]
pub struct OrderResponse {
    pub symbol: String,
    #[serde(rename = "orderId")]
    pub order_id: u64,
    #[serde(rename = "orderListId")]
    pub order_list_id: i64,
    #[serde(rename = "clientOrderId")]
    pub client_order_id: String,
    #[serde(rename = "transactTime")]
    pub transact_time: u64,
    pub price: String,
    #[serde(rename = "origQty")]
    pub orig_qty: String,
    #[serde(rename = "executedQty")]
    pub executed_qty: String,
    #[serde(rename = "cummulativeQuoteQty")]
    pub cummulative_quote_qty: String,
    pub status: String,
    #[serde(rename = "timeInForce")]
    pub time_in_force: String,
    #[serde(rename = "type")]
    pub order_type: String,
    pub side: String,
}

/// Binance Ultra connector with real WebSocket and REST
pub struct BinanceUltra {
    api_key: String,
    secret_key: String,
    ws_stream: Arc<RwLock<Option<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,
    order_books: Arc<RwLock<HashMap<String, OrderBook>>>,
    trades: Arc<RwLock<Vec<Trade>>>,
    http_client: reqwest::Client,
}

/// Internal order book representation
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<(f64, f64)>, // (price, quantity)
    pub asks: Vec<(f64, f64)>,
    pub last_update_id: u64,
}

impl BinanceUltra {
    pub fn new(api_key: String, secret_key: String) -> Self {
        Self {
            api_key,
            secret_key,
            ws_stream: Arc::new(RwLock::new(None)),
            order_books: Arc::new(RwLock::new(HashMap::new())),
            trades: Arc::new(RwLock::new(Vec::new())),
            http_client: reqwest::Client::new(),
        }
    }
    
    /// Connect to Binance WebSocket
    pub async fn connect_websocket(&self) -> Result<(), Box<dyn std::error::Error>> {
        let (ws_stream, _) = connect_async(BINANCE_WS_URL).await?;
        *self.ws_stream.write() = Some(ws_stream);
        Ok(())
    }
    
    /// Subscribe to market data streams
    pub async fn subscribe_streams(&self, symbols: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
        let mut ws = self.ws_stream.write();
        
        if let Some(stream) = ws.as_mut() {
            // Build subscription message
            let mut streams = Vec::new();
            
            for symbol in symbols {
                let symbol_lower = symbol.to_lowercase();
                streams.push(format!("{}@depth20@100ms", symbol_lower));
                streams.push(format!("{}@trade", symbol_lower));
                streams.push(format!("{}@miniTicker", symbol_lower));
            }
            
            let sub_msg = serde_json::json!({
                "method": "SUBSCRIBE",
                "params": streams,
                "id": 1
            });
            
            stream.send(Message::Text(sub_msg.to_string())).await?;
        }
        
        Ok(())
    }
    
    /// Process WebSocket messages
    pub async fn process_messages(&self) {
        let ws_stream = self.ws_stream.clone();
        let order_books = self.order_books.clone();
        let trades = self.trades.clone();
        
        tokio::spawn(async move {
            loop {
                let mut ws = ws_stream.write();
                
                if let Some(stream) = ws.as_mut() {
                    if let Some(msg) = stream.next().await {
                        match msg {
                            Ok(Message::Text(text)) => {
                                // Parse message
                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                                    if let Some(event_type) = json.get("e").and_then(|e| e.as_str()) {
                                        match event_type {
                                            "depthUpdate" => {
                                                if let Ok(depth) = serde_json::from_value::<DepthUpdate>(json) {
                                                    Self::update_order_book(&order_books, depth);
                                                }
                                            }
                                            "trade" => {
                                                if let Ok(trade) = serde_json::from_value::<Trade>(json) {
                                                    trades.write().push(trade);
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
    
    /// Update order book from depth update
    fn update_order_book(order_books: &Arc<RwLock<HashMap<String, OrderBook>>>, update: DepthUpdate) {
        let mut books = order_books.write();
        
        let book = books.entry(update.symbol.clone()).or_insert(OrderBook {
            symbol: update.symbol.clone(),
            bids: Vec::new(),
            asks: Vec::new(),
            last_update_id: 0,
        });
        
        // Update bids
        for bid in update.bids {
            if let (Ok(price), Ok(qty)) = (bid[0].parse::<f64>(), bid[1].parse::<f64>()) {
                // Remove if quantity is 0, otherwise update
                if qty == 0.0 {
                    book.bids.retain(|(p, _)| (*p - price).abs() > 1e-8);
                } else {
                    // Find and update or insert
                    if let Some(pos) = book.bids.iter().position(|(p, _)| (*p - price).abs() < 1e-8) {
                        book.bids[pos].1 = qty;
                    } else {
                        book.bids.push((price, qty));
                    }
                }
            }
        }
        
        // Update asks
        for ask in update.asks {
            if let (Ok(price), Ok(qty)) = (ask[0].parse::<f64>(), ask[1].parse::<f64>()) {
                if qty == 0.0 {
                    book.asks.retain(|(p, _)| (*p - price).abs() > 1e-8);
                } else {
                    if let Some(pos) = book.asks.iter().position(|(p, _)| (*p - price).abs() < 1e-8) {
                        book.asks[pos].1 = qty;
                    } else {
                        book.asks.push((price, qty));
                    }
                }
            }
        }
        
        // Sort order book
        book.bids.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // Descending
        book.asks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap()); // Ascending
        
        // Keep only top levels
        book.bids.truncate(20);
        book.asks.truncate(20);
        
        book.last_update_id = update.final_update_id;
    }
    
    /// Place a market order
    pub async fn place_market_order(
        &self,
        symbol: &str,
        side: &str,
        quantity: f64,
    ) -> Result<OrderResponse, Box<dyn std::error::Error>> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
        
        let order = OrderRequest {
            symbol: symbol.to_string(),
            side: side.to_string(),
            order_type: "MARKET".to_string(),
            quantity: Some(format!("{:.8}", quantity)),
            price: None,
            time_in_force: None,
            recv_window: 5000,
            timestamp,
        };
        
        self.send_order(order).await
    }
    
    /// Place a limit order
    pub async fn place_limit_order(
        &self,
        symbol: &str,
        side: &str,
        quantity: f64,
        price: f64,
    ) -> Result<OrderResponse, Box<dyn std::error::Error>> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
        
        let order = OrderRequest {
            symbol: symbol.to_string(),
            side: side.to_string(),
            order_type: "LIMIT".to_string(),
            quantity: Some(format!("{:.8}", quantity)),
            price: Some(format!("{:.8}", price)),
            time_in_force: Some("GTC".to_string()),
            recv_window: 5000,
            timestamp,
        };
        
        self.send_order(order).await
    }
    
    /// Send order to Binance
    async fn send_order(&self, order: OrderRequest) -> Result<OrderResponse, Box<dyn std::error::Error>> {
        // Build query string
        let query = serde_urlencoded::to_string(&order)?;
        
        // Sign the request
        let signature = self.sign_request(&query);
        
        // Send request
        let url = format!("{}/api/v3/order?{}&signature={}", BINANCE_REST_URL, query, signature);
        
        let response = self.http_client
            .post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await?;
        
        let order_response: OrderResponse = response.json().await?;
        Ok(order_response)
    }
    
    /// Sign request with HMAC-SHA256
    fn sign_request(&self, query: &str) -> String {
        type HmacSha256 = Hmac<Sha256>;
        
        let mut mac = HmacSha256::new_from_slice(self.secret_key.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(query.as_bytes());
        
        let result = mac.finalize();
        hex::encode(result.into_bytes())
    }
    
    /// Get order book snapshot
    pub fn get_order_book(&self, symbol: &str) -> Option<OrderBook> {
        self.order_books.read().get(symbol).cloned()
    }
    
    /// Get recent trades
    pub fn get_recent_trades(&self, symbol: &str, limit: usize) -> Vec<Trade> {
        self.trades.read()
            .iter()
            .filter(|t| t.symbol == symbol)
            .take(limit)
            .cloned()
            .collect()
    }
    
    /// Cancel order
    pub async fn cancel_order(
        &self,
        symbol: &str,
        order_id: u64,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
        
        let query = format!(
            "symbol={}&orderId={}&recvWindow=5000&timestamp={}",
            symbol, order_id, timestamp
        );
        
        let signature = self.sign_request(&query);
        let url = format!("{}/api/v3/order?{}&signature={}", BINANCE_REST_URL, query, signature);
        
        let response = self.http_client
            .delete(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await?;
        
        let result: serde_json::Value = response.json().await?;
        Ok(result)
    }
    
    /// Get account information
    pub async fn get_account_info(&self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
        
        let query = format!("recvWindow=5000&timestamp={}", timestamp);
        let signature = self.sign_request(&query);
        let url = format!("{}/api/v3/account?{}&signature={}", BINANCE_REST_URL, query, signature);
        
        let response = self.http_client
            .get(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await?;
        
        let account: serde_json::Value = response.json().await?;
        Ok(account)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_binance_creation() {
        let binance = BinanceUltra::new(
            "test_api_key".to_string(),
            "test_secret_key".to_string()
        );
        
        assert!(binance.get_order_book("BTCUSDT").is_none());
    }
    
    #[test]
    fn test_signature_generation() {
        let binance = BinanceUltra::new(
            "test_api_key".to_string(),
            "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j".to_string()
        );
        
        let query = "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1&recvWindow=5000&timestamp=1499827319559";
        let signature = binance.sign_request(query);
        
        assert_eq!(
            signature,
            "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"
        );
    }
    
    #[tokio::test]
    #[ignore] // Requires real API connection
    async fn test_websocket_connection() {
        let binance = BinanceUltra::new(
            "".to_string(),
            "".to_string()
        );
        
        match binance.connect_websocket().await {
            Ok(_) => println!("WebSocket connected successfully"),
            Err(e) => println!("WebSocket connection failed: {}", e),
        }
    }
}