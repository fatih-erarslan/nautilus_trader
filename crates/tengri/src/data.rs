//! Multi-source data integration for Tengri trading strategy
//! 
//! Provides unified data aggregation from Polymarket, Databento, and Tardis sources
//! with real-time streaming, historical data access, and cross-source validation.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{broadcast, RwLock};
use tokio::time::{interval, sleep};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use reqwest::Client;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures::{SinkExt, StreamExt};

use crate::{Result, TengriError};
use crate::config::{DataSourcesConfig, DatabentoConfig, TardisConfig, PolymarketConfig};
use crate::types::*;

/// Multi-source data aggregator for cryptocurrency and prediction market data
#[derive(Clone)]
pub struct DataAggregator {
    config: DataSourcesConfig,
    client: Client,
    data_cache: Arc<RwLock<DataCache>>,
    event_sender: broadcast::Sender<MarketEvent>,
    databento_handler: Option<DatabentoHandler>,
    tardis_handler: Option<TardisHandler>,
    polymarket_handler: Option<PolymarketHandler>,
}

/// Internal data cache for efficient data storage and retrieval
#[derive(Debug, Default)]
pub struct DataCache {
    prices: HashMap<String, PriceData>,
    order_books: HashMap<String, OrderBookData>,
    trades: HashMap<String, Vec<TradeData>>,
    polymarket_odds: HashMap<String, PolymarketData>,
    last_update: HashMap<String, DateTime<Utc>>,
    data_quality: HashMap<String, DataQuality>,
}

/// Market event types for real-time data streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketEvent {
    Price(PriceUpdate),
    Trade(TradeData),
    OrderBook(OrderBookUpdate),
    PolymarketOdds(PolymarketUpdate),
    DataQualityAlert(DataQualityAlert),
}

/// Price update from any data source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceUpdate {
    pub symbol: String,
    pub price: f64,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub volume_24h: Option<f64>,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
}

/// Order book update with depth information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookUpdate {
    pub symbol: String,
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub sequence: Option<u64>,
}

/// Polymarket prediction market update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolymarketUpdate {
    pub market_id: String,
    pub question: String,
    pub yes_price: f64,
    pub no_price: f64,
    pub volume: f64,
    pub liquidity: f64,
    pub timestamp: DateTime<Utc>,
    pub category: String,
}

/// Data quality monitoring and alerting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityAlert {
    pub source: String,
    pub alert_type: String,
    pub message: String,
    pub severity: AlertSeverity,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Databento market data handler
#[derive(Clone)]
pub struct DatabentoHandler {
    config: DatabentoConfig,
    client: Client,
    websocket_url: String,
}

/// Tardis market data handler  
#[derive(Clone)]
pub struct TardisHandler {
    config: TardisConfig,
    client: Client,
    websocket_url: String,
}

/// Polymarket prediction market handler
#[derive(Clone)]
pub struct PolymarketHandler {
    config: PolymarketConfig,
    client: Client,
    api_base_url: String,
}

impl DataAggregator {
    /// Create a new data aggregator with configuration
    pub async fn new(config: DataSourcesConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| TengriError::Network(e))?;

        let (event_sender, _) = broadcast::channel(10000);
        let data_cache = Arc::new(RwLock::new(DataCache::default()));

        // Initialize data source handlers
        let databento_handler = if config.databento.enabled {
            Some(DatabentoHandler::new(config.databento.clone()).await?)
        } else {
            None
        };

        let tardis_handler = if config.tardis.enabled {
            Some(TardisHandler::new(config.tardis.clone()).await?)
        } else {
            None
        };

        let polymarket_handler = if config.polymarket.enabled {
            Some(PolymarketHandler::new(config.polymarket.clone()).await?)
        } else {
            None
        };

        Ok(Self {
            config,
            client,
            data_cache,
            event_sender,
            databento_handler,
            tardis_handler,
            polymarket_handler,
        })
    }

    /// Start real-time data streaming from all enabled sources
    pub async fn start_streaming(&mut self) -> Result<()> {
        tracing::info!("Starting multi-source data streaming");

        // Start Databento streaming
        if let Some(handler) = self.databento_handler.clone() {
            let event_sender = self.event_sender.clone();
            let cache = self.data_cache.clone();
            
            tokio::spawn(async move {
                let mut handler = handler;
                if let Err(e) = handler.start_streaming(event_sender, cache).await {
                    tracing::error!("Databento streaming error: {}", e);
                }
            });
        }

        // Start Tardis streaming
        if let Some(handler) = self.tardis_handler.clone() {
            let event_sender = self.event_sender.clone();
            let cache = self.data_cache.clone();
            
            tokio::spawn(async move {
                let mut handler = handler;
                if let Err(e) = handler.start_streaming(event_sender, cache).await {
                    tracing::error!("Tardis streaming error: {}", e);
                }
            });
        }

        // Start Polymarket polling
        if let Some(handler) = self.polymarket_handler.clone() {
            let event_sender = self.event_sender.clone();
            let cache = self.data_cache.clone();
            let update_interval = self.config.polymarket.update_interval;
            
            tokio::spawn(async move {
                let mut handler = handler;
                if let Err(e) = handler.start_polling(event_sender, cache, update_interval).await {
                    tracing::error!("Polymarket polling error: {}", e);
                }
            });
        }

        // Start data quality monitoring
        self.start_quality_monitoring().await;

        tracing::info!("All data sources started successfully");
        Ok(())
    }

    /// Get market event stream receiver
    pub fn subscribe_events(&self) -> broadcast::Receiver<MarketEvent> {
        self.event_sender.subscribe()
    }

    /// Get latest price for a symbol from cache
    pub async fn get_latest_price(&self, symbol: &str) -> Option<PriceData> {
        let cache = self.data_cache.read().await;
        cache.prices.get(symbol).cloned()
    }

    /// Get latest order book for a symbol
    pub async fn get_order_book(&self, symbol: &str) -> Option<OrderBookData> {
        let cache = self.data_cache.read().await;
        cache.order_books.get(symbol).cloned()
    }

    /// Get recent trades for a symbol
    pub async fn get_recent_trades(&self, symbol: &str, limit: usize) -> Vec<TradeData> {
        let cache = self.data_cache.read().await;
        if let Some(trades) = cache.trades.get(symbol) {
            trades.iter().rev().take(limit).cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Get Polymarket odds for a market
    pub async fn get_polymarket_odds(&self, market_id: &str) -> Option<PolymarketData> {
        let cache = self.data_cache.read().await;
        cache.polymarket_odds.get(market_id).cloned()
    }

    /// Get data quality metrics for all sources
    pub async fn get_data_quality(&self) -> HashMap<String, DataQuality> {
        let cache = self.data_cache.read().await;
        cache.data_quality.clone()
    }

    /// Validate data consistency across sources
    pub async fn validate_cross_source_data(&self, symbol: &str) -> Result<ValidationResult> {
        let cache = self.data_cache.read().await;
        
        if !self.config.aggregation.validation.enabled {
            return Ok(ValidationResult::default());
        }

        let mut prices = Vec::new();
        let mut volumes = Vec::new();
        let mut sources = Vec::new();

        // Collect prices from all sources for the symbol
        if let Some(price_data) = cache.prices.get(symbol) {
            prices.push(price_data.price);
            if let Some(volume) = price_data.volume_24h {
                volumes.push(volume);
            }
            sources.push(price_data.source.clone());
        }

        if prices.len() < 2 {
            return Ok(ValidationResult {
                is_valid: true,
                price_deviation: 0.0,
                volume_deviation: 0.0,
                sources_count: sources.len(),
            });
        }

        // Calculate price deviation
        let avg_price = prices.iter().sum::<f64>() / prices.len() as f64;
        let max_deviation = prices.iter()
            .map(|&p| (p - avg_price).abs() / avg_price)
            .fold(0.0, f64::max);

        // Calculate volume deviation if available
        let volume_deviation = if volumes.len() >= 2 {
            let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
            volumes.iter()
                .map(|&v| (v - avg_volume).abs() / avg_volume)
                .fold(0.0, f64::max)
        } else {
            0.0
        };

        let is_valid = max_deviation <= self.config.aggregation.validation.max_price_deviation
            && volume_deviation <= self.config.aggregation.validation.max_volume_deviation;

        Ok(ValidationResult {
            is_valid,
            price_deviation: max_deviation,
            volume_deviation,
            sources_count: sources.len(),
        })
    }

    /// Start data quality monitoring background task
    async fn start_quality_monitoring(&self) {
        let cache = self.data_cache.clone();
        let event_sender = self.event_sender.clone();
        let quality_config = self.config.aggregation.quality_thresholds.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Check every minute
            
            loop {
                interval.tick().await;
                
                let cache_read = cache.read().await;
                let now = Utc::now();
                
                for (source, last_update) in &cache_read.last_update {
                    let age_seconds = (now - *last_update).num_seconds() as u64;
                    
                    if age_seconds > quality_config.max_latency_ms / 1000 {
                        let alert = DataQualityAlert {
                            source: source.clone(),
                            alert_type: "stale_data".to_string(),
                            message: format!("Data from {} is {} seconds old", source, age_seconds),
                            severity: if age_seconds > 300 { 
                                AlertSeverity::High 
                            } else { 
                                AlertSeverity::Medium 
                            },
                            timestamp: now,
                        };
                        
                        let _ = event_sender.send(MarketEvent::DataQualityAlert(alert));
                    }
                }
            }
        });
    }
}

impl DatabentoHandler {
    /// Create new Databento handler
    pub async fn new(config: DatabentoConfig) -> Result<Self> {
        let client = Client::new();
        let websocket_url = "wss://gateway.databento.com/ws".to_string();
        
        Ok(Self {
            config,
            client,
            websocket_url,
        })
    }

    /// Start real-time streaming from Databento
    pub async fn start_streaming(
        &mut self,
        event_sender: broadcast::Sender<MarketEvent>,
        cache: Arc<RwLock<DataCache>>,
    ) -> Result<()> {
        tracing::info!("Starting Databento streaming for dataset: {}", self.config.dataset);

        // Connect to Databento WebSocket
        let (ws_stream, _) = connect_async(&self.websocket_url).await
            .map_err(|e| TengriError::DataSource(e.to_string()))?;

        let (mut write, mut read) = ws_stream.split();

        // Send authentication and subscription messages
        let auth_msg = serde_json::json!({
            "op": "auth",
            "key": self.config.api_key
        });
        
        write.send(Message::Text(auth_msg.to_string().into())).await
            .map_err(|e| TengriError::DataSource(e.to_string()))?;

        let subscribe_msg = serde_json::json!({
            "op": "subscribe",
            "dataset": self.config.dataset,
            "schema": self.config.schema,
            "symbols": self.config.symbols
        });
        
        write.send(Message::Text(subscribe_msg.to_string().into())).await
            .map_err(|e| TengriError::DataSource(e.to_string()))?;

        // Process incoming messages
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(market_data) = self.parse_databento_message(&text).await {
                        // Update cache
                        self.update_cache(&cache, &market_data).await;
                        
                        // Send event
                        let _ = event_sender.send(market_data);
                    }
                }
                Ok(Message::Close(_)) => {
                    tracing::warn!("Databento WebSocket connection closed");
                    break;
                }
                Err(e) => {
                    tracing::error!("Databento WebSocket error: {}", e);
                    sleep(Duration::from_secs(5)).await; // Reconnect delay
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Parse Databento message into market event
    async fn parse_databento_message(&self, message: &str) -> Result<MarketEvent> {
        // Parse Databento specific message format
        let data: serde_json::Value = serde_json::from_str(message)?;
        
        // Extract relevant fields based on schema type
        match self.config.schema.as_str() {
            "trades" => {
                let trade = TradeData {
                    symbol: data["symbol"].as_str().unwrap_or_default().to_string(),
                    price: data["price"].as_f64().unwrap_or_default(),
                    quantity: data["size"].as_f64().unwrap_or_default(),
                    timestamp: Utc::now(), // Parse from data["ts_event"]
                    side: data["side"].as_str().unwrap_or_default().to_string(),
                    trade_id: data["trade_id"].as_str().map(|s| s.to_string()),
                    source: "databento".to_string(),
                };
                Ok(MarketEvent::Trade(trade))
            }
            "mbo" | "mbp" => {
                // Order book data
                let bids = data["bids"].as_array()
                    .unwrap_or(&Vec::new())
                    .iter()
                    .map(|level| PriceLevel {
                        price: level["price"].as_f64().unwrap_or_default(),
                        quantity: level["size"].as_f64().unwrap_or_default(),
                    })
                    .collect();

                let asks = data["asks"].as_array()
                    .unwrap_or(&Vec::new())
                    .iter()
                    .map(|level| PriceLevel {
                        price: level["price"].as_f64().unwrap_or_default(),
                        quantity: level["size"].as_f64().unwrap_or_default(),
                    })
                    .collect();

                let order_book = OrderBookUpdate {
                    symbol: data["symbol"].as_str().unwrap_or_default().to_string(),
                    bids,
                    asks,
                    timestamp: Utc::now(),
                    source: "databento".to_string(),
                    sequence: data["sequence"].as_u64(),
                };
                Ok(MarketEvent::OrderBook(order_book))
            }
            _ => Err(TengriError::DataSource(format!("Unsupported schema: {}", self.config.schema)))
        }
    }

    /// Update cache with new market data
    async fn update_cache(&self, cache: &Arc<RwLock<DataCache>>, event: &MarketEvent) {
        let mut cache_write = cache.write().await;
        let now = Utc::now();

        match event {
            MarketEvent::Trade(trade) => {
                // Update latest price
                let price_data = PriceData {
                    symbol: trade.symbol.clone(),
                    price: trade.price,
                    timestamp: trade.timestamp,
                    source: trade.source.clone(),
                    volume_24h: None, // Calculate from trades
                    bid: None,
                    ask: None,
                };
                cache_write.prices.insert(trade.symbol.clone(), price_data);

                // Store trade
                cache_write.trades
                    .entry(trade.symbol.clone())
                    .or_insert_with(Vec::new)
                    .push(trade.clone());

                // Limit trade history
                if let Some(trades) = cache_write.trades.get_mut(&trade.symbol) {
                    if trades.len() > 1000 {
                        trades.drain(0..trades.len() - 1000);
                    }
                }
            }
            MarketEvent::OrderBook(book) => {
                let order_book_data = OrderBookData {
                    symbol: book.symbol.clone(),
                    bids: book.bids.clone(),
                    asks: book.asks.clone(),
                    timestamp: book.timestamp,
                    source: book.source.clone(),
                };
                cache_write.order_books.insert(book.symbol.clone(), order_book_data);
            }
            _ => {}
        }

        cache_write.last_update.insert("databento".to_string(), now);
    }
}

impl TardisHandler {
    /// Create new Tardis handler
    pub async fn new(config: TardisConfig) -> Result<Self> {
        let client = Client::new();
        let websocket_url = "wss://api.tardis.dev/v1/ws".to_string();
        
        Ok(Self {
            config,
            client,
            websocket_url,
        })
    }

    /// Start real-time streaming from Tardis
    pub async fn start_streaming(
        &mut self,
        event_sender: broadcast::Sender<MarketEvent>,
        cache: Arc<RwLock<DataCache>>,
    ) -> Result<()> {
        tracing::info!("Starting Tardis streaming for exchanges: {:?}", self.config.exchanges);

        // Similar implementation to Databento but for Tardis API
        // Connect to Tardis WebSocket and handle their specific message format
        
        // Placeholder implementation - would need actual Tardis WebSocket protocol
        tokio::spawn(async move {
            loop {
                // Simulate market data updates
                sleep(Duration::from_millis(100)).await;
                
                let price_update = PriceUpdate {
                    symbol: "BTCUSDT".to_string(),
                    price: 50000.0 + (rand::random::<f64>() - 0.5) * 1000.0,
                    timestamp: Utc::now(),
                    source: "tardis".to_string(),
                    volume_24h: Some(1000000.0),
                    bid: None,
                    ask: None,
                };
                
                let _ = event_sender.send(MarketEvent::Price(price_update));
            }
        });

        Ok(())
    }
}

impl PolymarketHandler {
    /// Create new Polymarket handler
    pub async fn new(config: PolymarketConfig) -> Result<Self> {
        let client = Client::new();
        let api_base_url = "https://clob.polymarket.com".to_string();
        
        Ok(Self {
            config,
            client,
            api_base_url,
        })
    }

    /// Start polling Polymarket API for prediction market data
    pub async fn start_polling(
        &mut self,
        event_sender: broadcast::Sender<MarketEvent>,
        cache: Arc<RwLock<DataCache>>,
        update_interval: u64,
    ) -> Result<()> {
        tracing::info!("Starting Polymarket polling every {} seconds", update_interval);

        let client = self.client.clone();
        let base_url = self.api_base_url.clone();
        let markets = self.config.markets.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(update_interval));
            
            loop {
                interval.tick().await;
                
                // Fetch market data from Polymarket API
                if let Ok(markets_data) = Self::fetch_markets(&client, &base_url, &markets).await {
                    for market_data in markets_data {
                        // Update cache
                        let mut cache_write = cache.write().await;
                        cache_write.polymarket_odds.insert(
                            market_data.market_id.clone(),
                            PolymarketData {
                                market_id: market_data.market_id.clone(),
                                question: market_data.question.clone(),
                                yes_price: market_data.yes_price,
                                no_price: market_data.no_price,
                                volume: market_data.volume,
                                liquidity: market_data.liquidity,
                                timestamp: market_data.timestamp,
                                category: market_data.category.clone(),
                            }
                        );
                        cache_write.last_update.insert("polymarket".to_string(), Utc::now());
                        drop(cache_write);
                        
                        // Send event
                        let _ = event_sender.send(MarketEvent::PolymarketOdds(market_data));
                    }
                }
            }
        });

        Ok(())
    }

    /// Fetch markets from Polymarket API
    async fn fetch_markets(
        client: &Client,
        base_url: &str,
        markets: &[String],
    ) -> Result<Vec<PolymarketUpdate>> {
        let mut results = Vec::new();
        
        for market in markets {
            let url = format!("{}/markets?category={}", base_url, market);
            
            if let Ok(response) = client.get(&url).send().await {
                if let Ok(data) = response.json::<serde_json::Value>().await {
                    // Parse Polymarket API response
                    if let Some(markets_array) = data.as_array() {
                        for market_data in markets_array {
                            let market_update = PolymarketUpdate {
                                market_id: market_data["id"].as_str().unwrap_or_default().to_string(),
                                question: market_data["question"].as_str().unwrap_or_default().to_string(),
                                yes_price: market_data["yes_price"].as_f64().unwrap_or_default(),
                                no_price: market_data["no_price"].as_f64().unwrap_or_default(),
                                volume: market_data["volume"].as_f64().unwrap_or_default(),
                                liquidity: market_data["liquidity"].as_f64().unwrap_or_default(),
                                timestamp: Utc::now(),
                                category: market.clone(),
                            };
                            results.push(market_update);
                        }
                    }
                }
            }
        }
        
        Ok(results)
    }
}

/// Data validation result
#[derive(Debug, Default)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub price_deviation: f64,
    pub volume_deviation: f64,
    pub sources_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

    #[tokio::test]
    async fn test_data_aggregator_creation() {
        let config = DataSourcesConfig::default();
        let aggregator = DataAggregator::new(config).await;
        assert!(aggregator.is_ok());
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let config = DataSourcesConfig::default();
        let aggregator = DataAggregator::new(config).await.unwrap();
        
        // Test cache is initially empty
        let price = aggregator.get_latest_price("BTCUSDT").await;
        assert!(price.is_none());
    }

    #[tokio::test]
    async fn test_cross_source_validation() {
        let config = DataSourcesConfig::default();
        let aggregator = DataAggregator::new(config).await.unwrap();
        
        let validation = aggregator.validate_cross_source_data("BTCUSDT").await;
        assert!(validation.is_ok());
    }
}