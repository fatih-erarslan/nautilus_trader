//! # Data Ingestion Agent
//!
//! Ultra-fast market data ingestion agent for high-frequency trading systems.
//! Supports multiple exchanges with sub-100μs latency targets.

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::sync::{RwLock, mpsc, Mutex};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug};
use async_trait::async_trait;

use crate::agents::base::{
    DataAgent, DataAgentId, DataAgentType, DataAgentState, DataAgentInfo,
    DataMessage, DataMessageType, MessageMetadata, MessagePriority,
    CoordinationMessage, HealthStatus, HealthLevel, HealthMetrics,
    AgentMetrics, BaseDataAgent, MetricsUpdate
};

/// Data ingestion agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataIngestionConfig {
    /// Target latency in microseconds
    pub target_latency_us: u64,
    /// Maximum connections per exchange
    pub max_connections_per_exchange: usize,
    /// Buffer size for incoming data
    pub buffer_size: usize,
    /// Exchanges to connect to
    pub exchanges: Vec<ExchangeConfig>,
    /// Enable high-frequency mode
    pub high_frequency_mode: bool,
    /// Compression settings
    pub compression: CompressionConfig,
    /// Failover settings
    pub failover: FailoverConfig,
}

impl Default for DataIngestionConfig {
    fn default() -> Self {
        Self {
            target_latency_us: 50,
            max_connections_per_exchange: 10,
            buffer_size: 1000000,
            exchanges: vec![
                ExchangeConfig {
                    name: "binance".to_string(),
                    enabled: true,
                    priority: 1,
                    endpoints: vec!["wss://stream.binance.com:9443".to_string()],
                    symbols: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
                    rate_limit: 1000,
                },
                ExchangeConfig {
                    name: "coinbase".to_string(),
                    enabled: true,
                    priority: 2,
                    endpoints: vec!["wss://ws-feed.exchange.coinbase.com".to_string()],
                    symbols: vec!["BTC-USD".to_string(), "ETH-USD".to_string()],
                    rate_limit: 1000,
                },
            ],
            high_frequency_mode: true,
            compression: CompressionConfig::default(),
            failover: FailoverConfig::default(),
        }
    }
}

/// Exchange configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfig {
    pub name: String,
    pub enabled: bool,
    pub priority: u8,
    pub endpoints: Vec<String>,
    pub symbols: Vec<String>,
    pub rate_limit: u32,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub enabled: bool,
    pub algorithm: CompressionAlgorithm,
    pub level: u8,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: CompressionAlgorithm::Lz4,
            level: 1,
        }
    }
}

/// Compression algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Lz4,
    Zstd,
    Gzip,
}

/// Failover configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverConfig {
    pub enabled: bool,
    pub max_retries: u32,
    pub retry_delay_ms: u64,
    pub circuit_breaker_threshold: u32,
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_retries: 3,
            retry_delay_ms: 100,
            circuit_breaker_threshold: 10,
        }
    }
}

/// Market data source
#[derive(Debug, Clone)]
pub struct MarketDataSource {
    pub exchange: String,
    pub symbol: String,
    pub connection: Arc<Mutex<Option<WebSocketConnection>>>,
    pub last_update: Instant,
    pub message_count: u64,
    pub error_count: u32,
}

/// WebSocket connection wrapper
#[derive(Debug)]
pub struct WebSocketConnection {
    pub url: String,
    pub connection: tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    pub last_ping: Instant,
}

/// Market data message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataMessage {
    pub exchange: String,
    pub symbol: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub data_type: MarketDataType,
    pub data: serde_json::Value,
    pub sequence: u64,
}

/// Market data types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MarketDataType {
    Tick,
    OrderBook,
    Trade,
    Candle,
    Volume,
    Spread,
}

/// Data ingestion agent
pub struct DataIngestionAgent {
    base: BaseDataAgent,
    config: Arc<DataIngestionConfig>,
    sources: Arc<RwLock<HashMap<String, MarketDataSource>>>,
    ingestion_metrics: Arc<RwLock<IngestionMetrics>>,
    buffer: Arc<RwLock<Vec<MarketDataMessage>>>,
    state: Arc<RwLock<IngestionState>>,
}

/// Ingestion metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionMetrics {
    pub messages_ingested: u64,
    pub bytes_ingested: u64,
    pub ingestion_rate_per_sec: f64,
    pub average_latency_us: f64,
    pub max_latency_us: f64,
    pub connection_count: usize,
    pub error_rate: f64,
    pub last_update: chrono::DateTime<chrono::Utc>,
}

impl Default for IngestionMetrics {
    fn default() -> Self {
        Self {
            messages_ingested: 0,
            bytes_ingested: 0,
            ingestion_rate_per_sec: 0.0,
            average_latency_us: 0.0,
            max_latency_us: 0.0,
            connection_count: 0,
            error_rate: 0.0,
            last_update: chrono::Utc::now(),
        }
    }
}

/// Ingestion state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionState {
    pub connections_active: usize,
    pub total_connections: usize,
    pub buffer_usage: f64,
    pub is_healthy: bool,
    pub last_health_check: chrono::DateTime<chrono::Utc>,
}

impl Default for IngestionState {
    fn default() -> Self {
        Self {
            connections_active: 0,
            total_connections: 0,
            buffer_usage: 0.0,
            is_healthy: true,
            last_health_check: chrono::Utc::now(),
        }
    }
}

impl DataIngestionAgent {
    /// Create a new data ingestion agent
    pub async fn new(config: DataIngestionConfig) -> Result<Self> {
        let base = BaseDataAgent::new(DataAgentType::DataIngestion);
        let config = Arc::new(config);
        let sources = Arc::new(RwLock::new(HashMap::new()));
        let ingestion_metrics = Arc::new(RwLock::new(IngestionMetrics::default()));
        let buffer = Arc::new(RwLock::new(Vec::with_capacity(config.buffer_size)));
        let state = Arc::new(RwLock::new(IngestionState::default()));
        
        // Initialize sources
        {
            let mut sources_map = sources.write().await;
            for exchange_config in &config.exchanges {
                if exchange_config.enabled {
                    for symbol in &exchange_config.symbols {
                        let key = format!("{}:{}", exchange_config.name, symbol);
                        sources_map.insert(key.clone(), MarketDataSource {
                            exchange: exchange_config.name.clone(),
                            symbol: symbol.clone(),
                            connection: Arc::new(Mutex::new(None)),
                            last_update: Instant::now(),
                            message_count: 0,
                            error_count: 0,
                        });
                    }
                }
            }
        }
        
        Ok(Self {
            base,
            config,
            sources,
            ingestion_metrics,
            buffer,
            state,
        })
    }
    
    /// Start ingestion from all configured sources
    pub async fn start_ingestion(&self) -> Result<()> {
        info!("Starting data ingestion from {} sources", self.sources.read().await.len());
        
        let sources = self.sources.read().await;
        for (key, source) in sources.iter() {
            let source_clone = source.clone();
            let config = self.config.clone();
            let ingestion_metrics = self.ingestion_metrics.clone();
            let buffer = self.buffer.clone();
            let state = self.state.clone();
            
            // Spawn ingestion task for each source
            tokio::spawn(async move {
                Self::ingest_from_source(
                    source_clone,
                    config,
                    ingestion_metrics,
                    buffer,
                    state,
                ).await;
            });
        }
        
        Ok(())
    }
    
    /// Ingest data from a single source
    async fn ingest_from_source(
        source: MarketDataSource,
        config: Arc<DataIngestionConfig>,
        metrics: Arc<RwLock<IngestionMetrics>>,
        buffer: Arc<RwLock<Vec<MarketDataMessage>>>,
        state: Arc<RwLock<IngestionState>>,
    ) {
        let exchange_config = config.exchanges.iter()
            .find(|e| e.name == source.exchange)
            .unwrap();
        
        for endpoint in &exchange_config.endpoints {
            match Self::connect_to_exchange(endpoint, &source.symbol).await {
                Ok(mut connection) => {
                    info!("Connected to {} for {}", source.exchange, source.symbol);
                    
                    // Update state
                    {
                        let mut state_guard = state.write().await;
                        state_guard.connections_active += 1;
                        state_guard.total_connections += 1;
                    }
                    
                    // Message ingestion loop
                    loop {
                        match Self::read_message(&mut connection).await {
                            Ok(message) => {
                                let start_time = Instant::now();
                                
                                // Process the message
                                let market_data = MarketDataMessage {
                                    exchange: source.exchange.clone(),
                                    symbol: source.symbol.clone(),
                                    timestamp: chrono::Utc::now(),
                                    data_type: MarketDataType::Tick,
                                    data: message,
                                    sequence: 0, // Would be set from exchange
                                };
                                
                                // Add to buffer
                                {
                                    let mut buffer_guard = buffer.write().await;
                                    buffer_guard.push(market_data);
                                    
                                    // Check buffer size
                                    if buffer_guard.len() > config.buffer_size {
                                        buffer_guard.remove(0);
                                    }
                                }
                                
                                // Update metrics
                                {
                                    let mut metrics_guard = metrics.write().await;
                                    metrics_guard.messages_ingested += 1;
                                    
                                    let latency = start_time.elapsed().as_micros() as f64;
                                    metrics_guard.average_latency_us = 
                                        (metrics_guard.average_latency_us + latency) / 2.0;
                                    
                                    if latency > metrics_guard.max_latency_us {
                                        metrics_guard.max_latency_us = latency;
                                    }
                                    
                                    metrics_guard.last_update = chrono::Utc::now();
                                    
                                    // Check latency target
                                    if latency > config.target_latency_us as f64 {
                                        warn!(
                                            "Latency target exceeded: {}μs > {}μs",
                                            latency,
                                            config.target_latency_us
                                        );
                                    }
                                }
                                
                                // Update buffer usage
                                {
                                    let mut state_guard = state.write().await;
                                    state_guard.buffer_usage = 
                                        buffer.read().await.len() as f64 / config.buffer_size as f64;
                                }
                            }
                            Err(e) => {
                                error!("Error reading message from {}: {}", source.exchange, e);
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to connect to {} for {}: {}", source.exchange, source.symbol, e);
                }
            }
        }
    }
    
    /// Connect to exchange WebSocket
    async fn connect_to_exchange(
        endpoint: &str,
        symbol: &str,
    ) -> Result<WebSocketConnection> {
        let url = format!("{}?symbol={}", endpoint, symbol);
        
        let (ws_stream, _) = tokio_tungstenite::connect_async(&url).await?;
        
        Ok(WebSocketConnection {
            url,
            connection: ws_stream,
            last_ping: Instant::now(),
        })
    }
    
    /// Read message from WebSocket connection
    async fn read_message(
        connection: &mut WebSocketConnection,
    ) -> Result<serde_json::Value> {
        use tokio_tungstenite::tungstenite::protocol::Message;
        use futures_util::StreamExt;
        
        if let Some(msg) = connection.connection.next().await {
            match msg? {
                Message::Text(text) => {
                    Ok(serde_json::from_str(&text)?)
                }
                Message::Binary(data) => {
                    // Handle binary data if needed
                    Ok(serde_json::json!({"binary": data}))
                }
                Message::Ping(_) => {
                    // Handle ping
                    Ok(serde_json::json!({"ping": true}))
                }
                Message::Pong(_) => {
                    // Handle pong
                    Ok(serde_json::json!({"pong": true}))
                }
                Message::Close(_) => {
                    Err(anyhow::anyhow!("Connection closed"))
                }
                Message::Frame(_) => {
                    Ok(serde_json::json!({"frame": true}))
                }
            }
        } else {
            Err(anyhow::anyhow!("No message received"))
        }
    }
    
    /// Get buffered data
    pub async fn get_buffered_data(&self) -> Vec<MarketDataMessage> {
        self.buffer.read().await.clone()
    }
    
    /// Clear buffer
    pub async fn clear_buffer(&self) -> Result<()> {
        self.buffer.write().await.clear();
        Ok(())
    }
    
    /// Get ingestion metrics
    pub async fn get_ingestion_metrics(&self) -> IngestionMetrics {
        self.ingestion_metrics.read().await.clone()
    }
    
    /// Get ingestion state
    pub async fn get_ingestion_state(&self) -> IngestionState {
        self.state.read().await.clone()
    }
}

#[async_trait]
impl DataAgent for DataIngestionAgent {
    fn get_id(&self) -> DataAgentId {
        self.base.id
    }
    
    fn get_type(&self) -> DataAgentType {
        DataAgentType::DataIngestion
    }
    
    async fn get_state(&self) -> DataAgentState {
        self.base.state.read().await.clone()
    }
    
    async fn get_info(&self) -> DataAgentInfo {
        self.base.info.read().await.clone()
    }
    
    async fn start(&self) -> Result<()> {
        info!("Starting data ingestion agent");
        
        self.base.update_state(DataAgentState::Running).await?;
        self.start_ingestion().await?;
        
        info!("Data ingestion agent started successfully");
        Ok(())
    }
    
    async fn stop(&self) -> Result<()> {
        info!("Stopping data ingestion agent");
        
        self.base.update_state(DataAgentState::Stopping).await?;
        
        // Close all connections
        let sources = self.sources.read().await;
        for (_, source) in sources.iter() {
            let connection = source.connection.lock().await;
            if let Some(_conn) = connection.as_ref() {
                // Close connection
                drop(connection);
            }
        }
        
        self.base.update_state(DataAgentState::Stopped).await?;
        
        info!("Data ingestion agent stopped successfully");
        Ok(())
    }
    
    async fn process(&self, message: DataMessage) -> Result<DataMessage> {
        let start_time = Instant::now();
        
        // Process the incoming message
        let processed_data = match message.message_type {
            DataMessageType::MarketData => {
                // Handle market data processing
                message.payload
            }
            _ => {
                // Handle other message types
                message.payload
            }
        };
        
        // Update metrics
        let latency = start_time.elapsed().as_micros() as f64;
        self.base.update_metrics(MetricsUpdate::MessageProcessed(latency)).await?;
        
        // Create response message
        let response = DataMessage {
            id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            source: self.get_id(),
            destination: message.destination,
            message_type: DataMessageType::ProcessedData,
            payload: processed_data,
            metadata: MessageMetadata {
                priority: MessagePriority::High,
                expires_at: Some(chrono::Utc::now() + chrono::Duration::seconds(30)),
                retry_count: 0,
                trace_id: format!("data_ingestion_{}", uuid::Uuid::new_v4()),
                span_id: format!("span_{}", uuid::Uuid::new_v4()),
            },
        };
        
        Ok(response)
    }
    
    async fn health_check(&self) -> Result<HealthStatus> {
        let state = self.get_ingestion_state().await;
        let metrics = self.get_ingestion_metrics().await;
        
        let health_level = if state.is_healthy && state.connections_active > 0 {
            HealthLevel::Healthy
        } else if state.connections_active > 0 {
            HealthLevel::Warning
        } else {
            HealthLevel::Critical
        };
        
        Ok(HealthStatus {
            status: health_level,
            last_check: chrono::Utc::now(),
            uptime: self.base.start_time.elapsed(),
            issues: Vec::new(),
            metrics: HealthMetrics {
                cpu_usage_percent: 0.0, // Would be measured
                memory_usage_mb: 0.0,   // Would be measured
                network_usage_mbps: 0.0, // Would be measured
                disk_usage_mb: 0.0,     // Would be measured
                error_rate: metrics.error_rate,
                response_time_ms: metrics.average_latency_us / 1000.0,
            },
        })
    }
    
    async fn get_metrics(&self) -> Result<AgentMetrics> {
        self.base.metrics.read().await.clone().into()
    }
    
    async fn reset(&self) -> Result<()> {
        info!("Resetting data ingestion agent");
        
        self.clear_buffer().await?;
        
        // Reset metrics
        {
            let mut metrics = self.ingestion_metrics.write().await;
            *metrics = IngestionMetrics::default();
        }
        
        // Reset state
        {
            let mut state = self.state.write().await;
            *state = IngestionState::default();
        }
        
        info!("Data ingestion agent reset successfully");
        Ok(())
    }
    
    async fn handle_coordination(&self, message: CoordinationMessage) -> Result<()> {
        debug!("Handling coordination message: {:?}", message.coordination_type);
        
        match message.coordination_type {
            crate::agents::base::CoordinationType::LoadBalancing => {
                // Handle load balancing
                info!("Received load balancing coordination");
            }
            crate::agents::base::CoordinationType::StateSync => {
                // Handle state synchronization
                info!("Received state sync coordination");
            }
            _ => {
                debug!("Unhandled coordination type: {:?}", message.coordination_type);
            }
        }
        
        Ok(())
    }
}

impl From<AgentMetrics> for Result<AgentMetrics> {
    fn from(metrics: AgentMetrics) -> Self {
        Ok(metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    
    #[test]
    async fn test_data_ingestion_agent_creation() {
        let config = DataIngestionConfig::default();
        let agent = DataIngestionAgent::new(config).await;
        assert!(agent.is_ok());
    }
    
    #[test]
    async fn test_agent_lifecycle() {
        let config = DataIngestionConfig::default();
        let agent = DataIngestionAgent::new(config).await.unwrap();
        
        let start_result = agent.start().await;
        assert!(start_result.is_ok());
        
        let stop_result = agent.stop().await;
        assert!(stop_result.is_ok());
    }
    
    #[test]
    async fn test_health_check() {
        let config = DataIngestionConfig::default();
        let agent = DataIngestionAgent::new(config).await.unwrap();
        
        let health = agent.health_check().await;
        assert!(health.is_ok());
    }
}