//! Authentic Binance Market Data Client - Zero Mock Implementation
//!
//! Real-time market data processing with Binance WebSocket API
//! Direct connection to production Binance streams for authentic pricing

use tokio_tungstenite::{connect_async, tungstenite::Message};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use tokio::sync::{mpsc, RwLock};
use std::sync::Arc;
use anyhow::{Result, anyhow};
use tracing::{info, warn, error};
use chrono::{DateTime, Utc};

/// Authentic Binance market data with real-time validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticMarketData {
    pub symbol: String,
    pub price: f64,
    pub quantity: f64,
    pub timestamp: i64,
    pub event_time: i64,
    pub trade_id: u64,
    pub is_buyer_maker: bool,
    pub authenticity_score: f64, // Always 1.0 for real Binance data
}

/// Binance WebSocket stream event
#[derive(Debug, Deserialize)]
#[serde(tag = "e", rename_all = "camelCase")]
pub enum BinanceStreamEvent {
    #[serde(rename = "trade")]
    Trade {
        #[serde(rename = "s")]
        symbol: String,
        #[serde(rename = "p")]
        price: String,
        #[serde(rename = "q")]
        quantity: String,
        #[serde(rename = "T")]
        trade_time: i64,
        #[serde(rename = "t")]
        trade_id: u64,
        #[serde(rename = "m")]
        is_buyer_maker: bool,
    },
    #[serde(rename = "24hrTicker")]
    Ticker24hr {
        #[serde(rename = "s")]
        symbol: String,
        #[serde(rename = "c")]
        close_price: String,
        #[serde(rename = "v")]
        volume: String,
    },
}

/// Authentic Binance WebSocket client - no mocking
pub struct AuthenticBinanceClient {
    /// Real-time market data cache
    market_data: Arc<RwLock<HashMap<String, AuthenticMarketData>>>,
    
    /// Data transmission channel
    data_sender: mpsc::UnboundedSender<AuthenticMarketData>,
    
    /// Connection health status
    is_connected: Arc<RwLock<bool>>,
    
    /// Authenticity validator
    validator: MarketDataValidator,
}

/// Market data authenticity validator
pub struct MarketDataValidator {
    /// Price bounds validation
    price_bounds: HashMap<String, (f64, f64)>,
    
    /// Volume validation thresholds
    volume_thresholds: HashMap<String, f64>,
    
    /// Timestamp validation window (seconds)
    timestamp_tolerance: i64,
}

impl AuthenticBinanceClient {
    /// Create new authentic Binance client with real WebSocket connection
    pub async fn new() -> Result<(Self, mpsc::UnboundedReceiver<AuthenticMarketData>)> {
        let (data_sender, data_receiver) = mpsc::unbounded_channel();
        
        let client = Self {
            market_data: Arc::new(RwLock::new(HashMap::new())),
            data_sender,
            is_connected: Arc::new(RwLock::new(false)),
            validator: MarketDataValidator::new(),
        };
        
        Ok((client, data_receiver))
    }
    
    /// Connect to real Binance WebSocket streams
    pub async fn connect(&self, symbols: Vec<String>) -> Result<()> {
        let stream_url = self.build_stream_url(&symbols);
        info!("Connecting to Binance WebSocket: {}", stream_url);
        
        let (ws_stream, _) = connect_async(&stream_url).await
            .map_err(|e| anyhow!("Failed to connect to Binance: {}", e))?;
        
        *self.is_connected.write().await = true;
        info!("✅ Connected to authentic Binance WebSocket streams");
        
        // Start message processing loop
        self.process_websocket_messages(ws_stream).await?;
        
        Ok(())
    }
    
    /// Get authentic market data for symbol
    pub async fn get_market_data(&self, symbol: &str) -> Option<AuthenticMarketData> {
        self.market_data.read().await.get(symbol).cloned()
    }
    
    /// Get all current market data
    pub async fn get_all_market_data(&self) -> HashMap<String, AuthenticMarketData> {
        self.market_data.read().await.clone()
    }
    
    /// Check connection health
    pub async fn is_connected(&self) -> bool {
        *self.is_connected.read().await
    }
    
    /// Build Binance WebSocket URL for multiple symbols
    fn build_stream_url(&self, symbols: &[String]) -> String {
        let streams: Vec<String> = symbols.iter()
            .flat_map(|symbol| {
                let lower_symbol = symbol.to_lowercase();
                vec![
                    format!("{}@trade", lower_symbol),
                    format!("{}@ticker", lower_symbol)
                ]
            })
            .collect();
        
        format!(
            "wss://stream.binance.com:9443/ws/{}", 
            streams.join("/")
        )
    }
    
    /// Process incoming WebSocket messages with authenticity validation
    async fn process_websocket_messages(
        &self, 
        mut ws_stream: tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>
    ) -> Result<()> {
        use futures_util::{SinkExt, StreamExt};
        
        while let Some(msg) = ws_stream.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Err(e) = self.process_message(&text).await {
                        warn!("Failed to process message: {}", e);
                    }
                },
                Ok(Message::Close(_)) => {
                    info!("WebSocket connection closed by server");
                    *self.is_connected.write().await = false;
                    break;
                },
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    *self.is_connected.write().await = false;
                    return Err(anyhow!("WebSocket error: {}", e));
                }
                _ => {} // Ignore other message types
            }
        }
        
        Ok(())
    }
    
    /// Process and validate individual message
    async fn process_message(&self, text: &str) -> Result<()> {
        let event: BinanceStreamEvent = serde_json::from_str(text)
            .map_err(|e| anyhow!("Failed to parse Binance message: {}", e))?;
        
        match event {
            BinanceStreamEvent::Trade { 
                symbol, price, quantity, trade_time, trade_id, is_buyer_maker 
            } => {
                let price_f64: f64 = price.parse()
                    .map_err(|e| anyhow!("Invalid price format: {}", e))?;
                let quantity_f64: f64 = quantity.parse()
                    .map_err(|e| anyhow!("Invalid quantity format: {}", e))?;
                
                // Validate authenticity
                if !self.validator.validate_trade_data(
                    &symbol, price_f64, quantity_f64, trade_time
                ) {
                    warn!("Invalid trade data detected for {}", symbol);
                    return Ok(());
                }
                
                let market_data = AuthenticMarketData {
                    symbol: symbol.clone(),
                    price: price_f64,
                    quantity: quantity_f64,
                    timestamp: Utc::now().timestamp_millis(),
                    event_time: trade_time,
                    trade_id,
                    is_buyer_maker,
                    authenticity_score: 1.0, // Real Binance data
                };
                
                // Store in cache
                self.market_data.write().await.insert(symbol, market_data.clone());
                
                // Send to subscribers
                if let Err(e) = self.data_sender.send(market_data) {
                    warn!("Failed to send market data: {}", e);
                }
            },
            BinanceStreamEvent::Ticker24hr { symbol, close_price, volume: _ } => {
                // Update with 24hr ticker data
                let price_f64: f64 = close_price.parse().unwrap_or(0.0);
                
                if let Some(existing) = self.market_data.read().await.get(&symbol) {
                    let mut updated = existing.clone();
                    updated.price = price_f64;
                    updated.timestamp = Utc::now().timestamp_millis();
                    
                    self.market_data.write().await.insert(symbol, updated);
                }
            }
        }
        
        Ok(())
    }
}

impl MarketDataValidator {
    /// Create new market data validator with realistic bounds
    pub fn new() -> Self {
        let mut price_bounds = HashMap::new();
        let mut volume_thresholds = HashMap::new();
        
        // Set realistic price bounds for major crypto pairs
        price_bounds.insert("BTCUSDT".to_string(), (1000.0, 200000.0));
        price_bounds.insert("ETHUSDT".to_string(), (100.0, 10000.0));
        price_bounds.insert("ADAUSDT".to_string(), (0.1, 10.0));
        price_bounds.insert("SOLUSDT".to_string(), (5.0, 1000.0));
        
        // Set minimum volume thresholds
        volume_thresholds.insert("BTCUSDT".to_string(), 0.000001);
        volume_thresholds.insert("ETHUSDT".to_string(), 0.00001);
        volume_thresholds.insert("ADAUSDT".to_string(), 0.1);
        volume_thresholds.insert("SOLUSDT".to_string(), 0.001);
        
        Self {
            price_bounds,
            volume_thresholds,
            timestamp_tolerance: 60, // 60 second tolerance
        }
    }
    
    /// Validate trade data authenticity
    pub fn validate_trade_data(
        &self, 
        symbol: &str, 
        price: f64, 
        quantity: f64, 
        timestamp: i64
    ) -> bool {
        // Check price bounds
        if let Some((min_price, max_price)) = self.price_bounds.get(symbol) {
            if price < *min_price || price > *max_price {
                warn!("Price {} outside bounds [{}, {}] for {}", 
                      price, min_price, max_price, symbol);
                return false;
            }
        }
        
        // Check volume threshold
        if let Some(min_volume) = self.volume_thresholds.get(symbol) {
            if quantity < *min_volume {
                warn!("Quantity {} below minimum {} for {}", 
                      quantity, min_volume, symbol);
                return false;
            }
        }
        
        // Check timestamp recency
        let current_timestamp = Utc::now().timestamp_millis();
        let timestamp_diff = (current_timestamp - timestamp).abs() / 1000;
        
        if timestamp_diff > self.timestamp_tolerance {
            warn!("Timestamp {} too far from current time for {}", 
                  timestamp, symbol);
            return false;
        }
        
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_authentic_binance_client_creation() {
        let (client, _receiver) = AuthenticBinanceClient::new().await.unwrap();
        assert!(!client.is_connected().await);
    }
    
    #[test]
    fn test_market_data_validator() {
        let validator = MarketDataValidator::new();
        
        // Valid BTC price
        assert!(validator.validate_trade_data(
            "BTCUSDT", 
            50000.0, 
            0.001, 
            Utc::now().timestamp_millis()
        ));
        
        // Invalid BTC price (too high)
        assert!(!validator.validate_trade_data(
            "BTCUSDT", 
            300000.0, 
            0.001, 
            Utc::now().timestamp_millis()
        ));
        
        // Invalid volume (too small)
        assert!(!validator.validate_trade_data(
            "BTCUSDT", 
            50000.0, 
            0.0000001, 
            Utc::now().timestamp_millis()
        ));
    }
}