// Real-Time Data Pipeline - Production Data Sources Only
// Copyright (c) 2025 TENGRI Trading Swarm

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio_stream::Stream;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug};

use crate::anti_mock::{DataSource, ValidationError, enforce_real_data};
use crate::{TradingPair, PairId, AnalyzerError};

pub mod pipeline;
pub mod exchanges;
pub mod aggregator;
pub mod validator;

pub use pipeline::*;
pub use exchanges::*;
pub use aggregator::*;
pub use validator::*;

/// Market data update from real sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketUpdate {
    pub pair_id: PairId,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub update_type: UpdateType,
    pub data: MarketData,
    pub source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpdateType {
    Ticker,
    OrderBook,
    Trade,
    Kline,
    BookTicker,
}

/// Real market data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub price: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub change_24h: f64,
    pub volume_24h: f64,
    pub quote_volume: f64,
    pub volatility: Option<f64>,
    pub liquidity_score: Option<f64>,
}

/// Exchange configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfig {
    pub name: String,
    pub endpoint: String,
    pub websocket_endpoint: String,
    pub api_key: String,
    pub api_secret: String,
    pub passphrase: Option<String>,
    pub rate_limit: u32,
    pub enabled: bool,
    pub testnet: bool,
}

impl ExchangeConfig {
    /// Validate configuration contains real endpoints only
    pub fn validate_real_endpoints(&self) -> Result<(), ValidationError> {
        // Check for mock/synthetic patterns
        let forbidden_patterns = ["mock", "fake", "test", "localhost", "127.0.0.1"];
        
        for pattern in &forbidden_patterns {
            if self.endpoint.to_lowercase().contains(pattern) && !self.testnet {
                return Err(ValidationError::InvalidEndpoint(
                    format!("Forbidden pattern '{}' in endpoint: {}", pattern, self.endpoint)
                ));
            }
        }
        
        // Validate known real endpoints
        let real_endpoints = [
            "api.binance.com",
            "api.coinbase.com",
            "api.kraken.com",
            "api.bitfinex.com",
            "api.huobi.pro",
            "testnet.binance.vision", // Testnet allowed
            "api-public.sandbox.pro.coinbase.com", // Sandbox allowed
        ];
        
        let is_real = real_endpoints.iter().any(|real| self.endpoint.contains(real));
        
        if !is_real && !self.testnet {
            return Err(ValidationError::InvalidEndpoint(
                format!("Unknown endpoint (not in whitelist): {}", self.endpoint)
            ));
        }
        
        Ok(())
    }
}

/// News source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsConfig {
    pub name: String,
    pub endpoint: String,
    pub api_key: String,
    pub rate_limit: u32,
    pub enabled: bool,
}

/// Data pipeline metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetrics {
    pub messages_received: u64,
    pub messages_processed: u64,
    pub messages_rejected: u64,
    pub average_latency: Duration,
    pub last_update: chrono::DateTime<chrono::Utc>,
    pub error_count: u64,
    pub connection_status: HashMap<String, ConnectionStatus>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionStatus {
    Connected,
    Connecting,
    Disconnected,
    Error,
}

/// Real-time data pipeline with zero-mock enforcement
#[derive(Debug)]
pub struct RealTimeDataPipeline {
    // Real exchange connections only
    exchanges: Arc<RwLock<HashMap<String, Box<dyn ExchangeConnector>>>>,
    
    // Data processing
    aggregator: Arc<RwLock<MarketDataAggregator>>,
    validator: Arc<RwLock<DataValidator>>,
    
    // Metrics and monitoring
    metrics: Arc<RwLock<PipelineMetrics>>,
    
    // Configuration
    config: HashMap<String, ExchangeConfig>,
}

impl RealTimeDataPipeline {
    /// Connect to real data sources only
    pub async fn connect_real_sources(config: &crate::AnalyzerConfig) -> Result<Self> {
        info!("Connecting to real data sources with zero-mock enforcement");
        
        let mut exchanges: HashMap<String, Box<dyn ExchangeConnector>> = HashMap::new();
        let mut validated_configs = HashMap::new();
        
        // Validate and connect to each exchange
        for (name, exchange_config) in &config.exchange_configs {
            // Validate endpoint is real
            exchange_config.validate_real_endpoints()
                .context(format!("Invalid endpoint for exchange: {}", name))?;
            
            // Create connector based on exchange type
            let connector: Box<dyn ExchangeConnector> = match name.as_str() {
                "binance" => Box::new(BinanceConnector::new(exchange_config.clone()).await?),
                "coinbase" => Box::new(CoinbaseConnector::new(exchange_config.clone()).await?),
                "kraken" => Box::new(KrakenConnector::new(exchange_config.clone()).await?),
                _ => {
                    warn!("Unknown exchange type: {}, skipping", name);
                    continue;
                }
            };
            
            // Test connection
            if connector.test_connection().await? {
                info!("✅ Connected to real exchange: {}", name);
                exchanges.insert(name.clone(), connector);
                validated_configs.insert(name.clone(), exchange_config.clone());
            } else {
                error!("❌ Failed to connect to exchange: {}", name);
                return Err(AnalyzerError::DataSourceError(
                    format!("Failed to connect to exchange: {}", name).into()
                ).into());
            }
        }
        
        if exchanges.is_empty() {
            return Err(AnalyzerError::DataSourceError(
                "No valid exchange connections established".into()
            ).into());
        }
        
        // Initialize components
        let aggregator = MarketDataAggregator::new();
        let validator = DataValidator::new();
        let metrics = PipelineMetrics {
            messages_received: 0,
            messages_processed: 0,
            messages_rejected: 0,
            average_latency: Duration::from_millis(0),
            last_update: chrono::Utc::now(),
            error_count: 0,
            connection_status: exchanges.keys()
                .map(|k| (k.clone(), ConnectionStatus::Connected))
                .collect(),
        };
        
        info!("Real-time data pipeline initialized with {} exchanges", exchanges.len());
        
        Ok(Self {
            exchanges: Arc::new(RwLock::new(exchanges)),
            aggregator: Arc::new(RwLock::new(aggregator)),
            validator: Arc::new(RwLock::new(validator)),
            metrics: Arc::new(RwLock::new(metrics)),
            config: validated_configs,
        })
    }
    
    /// Stream real market data from all connected exchanges
    pub fn stream_market_data(&self) -> impl Stream<Item = MarketUpdate> {
        let exchanges = self.exchanges.clone();
        let validator = self.validator.clone();
        let metrics = self.metrics.clone();
        
        async_stream::stream! {
            // Create streams from all exchanges
            let exchange_streams = {
                let exchanges_guard = exchanges.read().await;
                let mut streams = Vec::new();
                
                for (name, connector) in exchanges_guard.iter() {
                    let stream = connector.subscribe_market_data().await;
                    streams.push(stream);
                }
                streams
            };
            
            // Merge all streams
            let mut combined_stream = tokio_stream::StreamExt::merge_multiple(exchange_streams);
            
            while let Some(update) = tokio_stream::StreamExt::next(&mut combined_stream).await {
                // Validate data is real
                let validator_guard = validator.read().await;
                match validator_guard.validate_market_update(&update).await {
                    Ok(validated_update) => {
                        // Update metrics
                        {
                            let mut metrics_guard = metrics.write().await;
                            metrics_guard.messages_received += 1;
                            metrics_guard.messages_processed += 1;
                            metrics_guard.last_update = chrono::Utc::now();
                        }
                        
                        yield validated_update;
                    },
                    Err(e) => {
                        warn!("Invalid market update rejected: {}", e);
                        {
                            let mut metrics_guard = metrics.write().await;
                            metrics_guard.messages_received += 1;
                            metrics_guard.messages_rejected += 1;
                            metrics_guard.error_count += 1;
                        }
                    }
                }
            }
        }
    }
    
    /// Validate all data sources are real
    pub async fn validate_all_sources(&self) -> Result<()> {
        debug!("Validating all data sources");
        
        let exchanges = self.exchanges.read().await;
        
        for (name, connector) in exchanges.iter() {
            // Validate connection is to real exchange
            if !connector.is_real_connection().await? {
                return Err(AnalyzerError::MockDataDetected(
                    format!("Mock connection detected for exchange: {}", name)
                ).into());
            }
            
            // Validate data freshness
            let last_update = connector.last_data_update().await?;
            let age = chrono::Utc::now() - last_update;
            
            if age > chrono::Duration::seconds(300) {
                return Err(AnalyzerError::DataSourceError(
                    format!("Stale data from exchange {}: {} seconds old", name, age.num_seconds()).into()
                ).into());
            }
        }
        
        debug!("✅ All data sources validated");
        Ok(())
    }
    
    /// Validate specific pair data source
    pub async fn validate_pair_data_source(&self, pair: &PairId) -> Result<()> {
        let exchanges = self.exchanges.read().await;
        
        if let Some(connector) = exchanges.get(&pair.exchange) {
            // Check if exchange supports this pair
            if !connector.supports_pair(pair).await? {
                return Err(AnalyzerError::DataSourceError(
                    format!("Exchange {} does not support pair: {}", pair.exchange, pair.symbol()).into()
                ).into());
            }
            
            // Validate pair data is real and fresh
            if let Some(data) = connector.get_pair_data(pair).await? {
                if !self.validator.read().await.validate_pair_data(&data).await? {
                    return Err(AnalyzerError::MockDataDetected(
                        format!("Invalid data for pair: {}", pair.symbol())
                    ).into());
                }
            }
        } else {
            return Err(AnalyzerError::DataSourceError(
                format!("No connector found for exchange: {}", pair.exchange).into()
            ).into());
        }
        
        Ok(())
    }
    
    /// Fetch all available trading pairs from real exchanges
    pub async fn fetch_all_pairs(&self) -> Result<Vec<TradingPair>> {
        let exchanges = self.exchanges.read().await;
        let mut all_pairs = Vec::new();
        
        for (name, connector) in exchanges.iter() {
            match connector.fetch_trading_pairs().await {
                Ok(pairs) => {
                    info!("Fetched {} pairs from {}", pairs.len(), name);
                    all_pairs.extend(pairs);
                },
                Err(e) => {
                    error!("Failed to fetch pairs from {}: {}", name, e);
                }
            }
        }
        
        // Remove duplicates and validate
        let mut unique_pairs = HashMap::new();
        for pair in all_pairs {
            let key = (pair.id.base.clone(), pair.id.quote.clone());
            if let Some(existing) = unique_pairs.get(&key) {
                // Keep pair with higher liquidity
                let existing_pair: &TradingPair = existing;
                if pair.liquidity_score > existing_pair.liquidity_score {
                    unique_pairs.insert(key, pair);
                }
            } else {
                unique_pairs.insert(key, pair);
            }
        }
        
        let result: Vec<TradingPair> = unique_pairs.into_values().collect();
        info!("Total unique pairs after deduplication: {}", result.len());
        
        Ok(result)
    }
    
    /// Get pipeline metrics
    pub async fn get_metrics(&self) -> PipelineMetrics {
        self.metrics.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_exchange_config_validation() {
        // Test valid configuration
        let valid_config = ExchangeConfig {
            name: "binance".to_string(),
            endpoint: "https://api.binance.com".to_string(),
            websocket_endpoint: "wss://stream.binance.com".to_string(),
            api_key: "real_api_key".to_string(),
            api_secret: "real_secret".to_string(),
            passphrase: None,
            rate_limit: 1200,
            enabled: true,
            testnet: false,
        };
        
        assert!(valid_config.validate_real_endpoints().is_ok());
        
        // Test invalid configuration
        let invalid_config = ExchangeConfig {
            name: "mock_exchange".to_string(),
            endpoint: "http://localhost:8080/mock".to_string(),
            websocket_endpoint: "ws://localhost:8080/ws".to_string(),
            api_key: "test_api_key".to_string(),
            api_secret: "test_secret".to_string(),
            passphrase: None,
            rate_limit: 1000,
            enabled: true,
            testnet: false,
        };
        
        assert!(invalid_config.validate_real_endpoints().is_err());
        
        // Test testnet configuration (should be allowed)
        let testnet_config = ExchangeConfig {
            name: "binance_testnet".to_string(),
            endpoint: "https://testnet.binance.vision".to_string(),
            websocket_endpoint: "wss://testnet.binance.vision/ws".to_string(),
            api_key: "testnet_key".to_string(),
            api_secret: "testnet_secret".to_string(),
            passphrase: None,
            rate_limit: 1200,
            enabled: true,
            testnet: true,
        };
        
        assert!(testnet_config.validate_real_endpoints().is_ok());
    }
    
    #[test]
    fn test_market_data_structure() {
        let market_data = MarketData {
            symbol: "BTCUSDT".to_string(),
            timestamp: chrono::Utc::now(),
            price: 50000.0,
            volume: 1.5,
            bid: 49995.0,
            ask: 50005.0,
            high_24h: 51000.0,
            low_24h: 49000.0,
            change_24h: 0.025,
            volume_24h: 15000.0,
            quote_volume: 750000000.0,
            volatility: Some(0.25),
            liquidity_score: Some(0.85),
        };
        
        // Verify structure
        assert_eq!(market_data.symbol, "BTCUSDT");
        assert!(market_data.price > 0.0);
        assert!(market_data.volume > 0.0);
        assert!(market_data.bid < market_data.ask);
        assert!(market_data.volatility.unwrap() > 0.0);
        assert!(market_data.liquidity_score.unwrap() > 0.0);
    }
}