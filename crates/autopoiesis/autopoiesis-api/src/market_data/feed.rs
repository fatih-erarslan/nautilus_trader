//! Market data feed implementation

use crate::prelude::*;
use crate::models::MarketData;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::collections::HashMap;
use num_traits::ToPrimitive;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tokio::time::{interval, Duration};

/// Market data feed that ingests and distributes real-time market data
#[derive(Debug)]
pub struct MarketDataFeed {
    /// Feed configuration
    config: MarketDataFeedConfig,
    
    /// Active subscriptions by symbol
    subscriptions: Arc<RwLock<HashMap<String, SubscriptionInfo>>>,
    
    /// Data publishers for each symbol
    publishers: Arc<RwLock<HashMap<String, broadcast::Sender<MarketData>>>>,
    
    /// Connection pool for data sources
    connections: Arc<RwLock<Vec<DataSourceConnection>>>,
    
    /// Feed state
    state: Arc<RwLock<FeedState>>,
    
    /// Data quality metrics
    metrics: Arc<RwLock<DataQualityMetrics>>,
}

#[derive(Debug, Clone)]
pub struct MarketDataFeedConfig {
    /// Data sources to connect to
    pub data_sources: Vec<DataSourceConfig>,
    
    /// Symbols to subscribe to
    pub symbols: Vec<String>,
    
    /// Update frequency in milliseconds
    pub update_frequency_ms: u64,
    
    /// Data validation rules
    pub validation_rules: ValidationRules,
    
    /// Reconnection settings
    pub reconnect_settings: ReconnectSettings,
    
    /// Buffer sizes
    pub buffer_sizes: BufferSizes,
}

#[derive(Debug, Clone)]
pub struct DataSourceConfig {
    pub name: String,
    pub endpoint: String,
    pub api_key: Option<String>,
    pub rate_limit: RateLimit,
    pub priority: u8,
}

#[derive(Debug, Clone)]
pub struct ValidationRules {
    pub max_price_deviation_pct: f64,
    pub max_spread_pct: f64,
    pub min_volume_threshold: Decimal,
    pub max_age_seconds: u64,
}

#[derive(Debug, Clone)]
pub struct ReconnectSettings {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
}

#[derive(Debug, Clone)]
pub struct BufferSizes {
    pub channel_buffer: usize,
    pub history_buffer: usize,
    pub metrics_buffer: usize,
}

#[derive(Debug, Clone)]
pub struct RateLimit {
    pub requests_per_second: u32,
    pub burst_capacity: u32,
}

#[derive(Debug, Clone)]
struct SubscriptionInfo {
    symbol: String,
    subscriber_count: u32,
    last_update: DateTime<Utc>,
    data_quality_score: f64,
}

#[derive(Debug)]
struct DataSourceConnection {
    source: DataSourceConfig,
    state: ConnectionState,
    last_heartbeat: DateTime<Utc>,
    error_count: u32,
}

#[derive(Debug, Clone)]
enum ConnectionState {
    Connected,
    Connecting,
    Disconnected,
    Error(String),
}

#[derive(Debug, Clone)]
enum FeedState {
    Starting,
    Running,
    Degraded,
    Stopped,
    Error(String),
}

#[derive(Debug, Clone, Default)]
struct DataQualityMetrics {
    total_updates: u64,
    valid_updates: u64,
    invalid_updates: u64,
    latency_ms: f64,
    uptime_pct: f64,
    error_rate: f64,
}

impl Default for MarketDataFeedConfig {
    fn default() -> Self {
        Self {
            data_sources: vec![
                DataSourceConfig {
                    name: "primary".to_string(),
                    endpoint: "wss://api.exchange.com/ws".to_string(),
                    api_key: None,
                    rate_limit: RateLimit {
                        requests_per_second: 100,
                        burst_capacity: 200,
                    },
                    priority: 1,
                }
            ],
            symbols: vec!["BTC/USD".to_string(), "ETH/USD".to_string()],
            update_frequency_ms: 100,
            validation_rules: ValidationRules {
                max_price_deviation_pct: 0.05,
                max_spread_pct: 0.01,
                min_volume_threshold: Decimal::from(1000),
                max_age_seconds: 10,
            },
            reconnect_settings: ReconnectSettings {
                max_retries: 5,
                initial_delay_ms: 1000,
                max_delay_ms: 30000,
                backoff_multiplier: 2.0,
            },
            buffer_sizes: BufferSizes {
                channel_buffer: 1000,
                history_buffer: 10000,
                metrics_buffer: 1000,
            },
        }
    }
}

impl MarketDataFeed {
    /// Create a new market data feed
    pub fn new(config: MarketDataFeedConfig) -> Self {
        Self {
            config,
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            publishers: Arc::new(RwLock::new(HashMap::new())),
            connections: Arc::new(RwLock::new(Vec::new())),
            state: Arc::new(RwLock::new(FeedState::Starting)),
            metrics: Arc::new(RwLock::new(DataQualityMetrics::default())),
        }
    }

    /// Start the market data feed
    pub async fn start(&self) -> Result<()> {
        *self.state.write().await = FeedState::Running;
        
        // Initialize connections to data sources
        self.initialize_connections().await?;
        
        // Start subscription management
        self.start_subscription_manager().await?;
        
        // Start health monitoring
        self.start_health_monitor().await?;
        
        info!("Market data feed started successfully");
        Ok(())
    }

    /// Stop the market data feed
    pub async fn stop(&self) -> Result<()> {
        *self.state.write().await = FeedState::Stopped;
        
        // Clean up connections
        self.cleanup_connections().await?;
        
        info!("Market data feed stopped");
        Ok(())
    }

    /// Subscribe to market data for a symbol
    pub async fn subscribe(&self, symbol: &str) -> Result<broadcast::Receiver<MarketData>> {
        let mut subscriptions = self.subscriptions.write().await;
        let mut publishers = self.publishers.write().await;

        // Create publisher if it doesn't exist
        if !publishers.contains_key(symbol) {
            let (tx, _) = broadcast::channel(self.config.buffer_sizes.channel_buffer);
            publishers.insert(symbol.to_string(), tx);
            
            // Initialize subscription info
            subscriptions.insert(symbol.to_string(), SubscriptionInfo {
                symbol: symbol.to_string(),
                subscriber_count: 0,
                last_update: Utc::now(),
                data_quality_score: 1.0,
            });
        }

        // Update subscriber count
        if let Some(sub_info) = subscriptions.get_mut(symbol) {
            sub_info.subscriber_count += 1;
        }

        // Return receiver
        let publisher = publishers.get(symbol).unwrap();
        Ok(publisher.subscribe())
    }

    /// Unsubscribe from market data for a symbol
    pub async fn unsubscribe(&self, symbol: &str) -> Result<()> {
        let mut subscriptions = self.subscriptions.write().await;
        
        if let Some(sub_info) = subscriptions.get_mut(symbol) {
            sub_info.subscriber_count = sub_info.subscriber_count.saturating_sub(1);
            
            // Remove subscription if no subscribers
            if sub_info.subscriber_count == 0 {
                subscriptions.remove(symbol);
                self.publishers.write().await.remove(symbol);
            }
        }

        Ok(())
    }

    /// Publish market data to subscribers
    pub async fn publish_data(&self, market_data: MarketData) -> Result<()> {
        // Validate data quality
        if !self.validate_market_data(&market_data).await? {
            self.update_metrics(false).await;
            return Ok(()); // Skip invalid data
        }

        // Update metrics
        self.update_metrics(true).await;

        // Publish to subscribers
        let publishers = self.publishers.read().await;
        if let Some(publisher) = publishers.get(&market_data.symbol) {
            if let Err(e) = publisher.send(market_data.clone()) {
                warn!("Failed to publish market data for {}: {}", market_data.symbol, e);
            }
        }

        // Update subscription info
        let mut subscriptions = self.subscriptions.write().await;
        if let Some(sub_info) = subscriptions.get_mut(&market_data.symbol) {
            sub_info.last_update = market_data.timestamp;
        }

        Ok(())
    }

    /// Get current feed state
    pub async fn state(&self) -> FeedState {
        self.state.read().await.clone()
    }

    /// Get data quality metrics
    pub async fn metrics(&self) -> DataQualityMetrics {
        self.metrics.read().await.clone()
    }

    /// Get active subscriptions
    pub async fn subscriptions(&self) -> HashMap<String, SubscriptionInfo> {
        self.subscriptions.read().await.clone()
    }

    async fn initialize_connections(&self) -> Result<()> {
        let mut connections = self.connections.write().await;
        
        for source_config in &self.config.data_sources {
            let connection = DataSourceConnection {
                source: source_config.clone(),
                state: ConnectionState::Connecting,
                last_heartbeat: Utc::now(),
                error_count: 0,
            };
            connections.push(connection);
        }

        // Simulate connection establishment
        for connection in connections.iter_mut() {
            connection.state = ConnectionState::Connected;
            connection.last_heartbeat = Utc::now();
        }

        Ok(())
    }

    async fn cleanup_connections(&self) -> Result<()> {
        let mut connections = self.connections.write().await;
        connections.clear();
        Ok(())
    }

    async fn start_subscription_manager(&self) -> Result<()> {
        let subscriptions = Arc::clone(&self.subscriptions);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(config.update_frequency_ms));
            
            loop {
                interval.tick().await;
                
                // Check for subscription updates
                let subs = subscriptions.read().await;
                for (symbol, _sub_info) in subs.iter() {
                    // In a real implementation, this would fetch data from sources
                    // For now, we'll simulate data generation
                    Self::simulate_market_data(symbol).await;
                }
            }
        });

        Ok(())
    }

    async fn start_health_monitor(&self) -> Result<()> {
        let state = Arc::clone(&self.state);
        let connections = Arc::clone(&self.connections);
        let metrics = Arc::clone(&self.metrics);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // Check connection health
                let conns = connections.read().await;
                let healthy_connections = conns.iter()
                    .filter(|c| matches!(c.state, ConnectionState::Connected))
                    .count();

                if healthy_connections == 0 {
                    *state.write().await = FeedState::Error("No healthy connections".to_string());
                } else if healthy_connections < conns.len() {
                    *state.write().await = FeedState::Degraded;
                } else {
                    *state.write().await = FeedState::Running;
                }

                // Update uptime metrics
                let mut m = metrics.write().await;
                m.uptime_pct = if healthy_connections > 0 { 
                    (healthy_connections as f64 / conns.len() as f64) * 100.0 
                } else { 
                    0.0 
                };
            }
        });

        Ok(())
    }

    async fn validate_market_data(&self, data: &MarketData) -> Result<bool> {
        let rules = &self.config.validation_rules;

        // Check age
        let age = (Utc::now() - data.timestamp).num_seconds() as u64;
        if age > rules.max_age_seconds {
            return Ok(false);
        }

        // Check spread
        let spread_pct = ((data.ask - data.bid) / data.mid).to_f64().unwrap_or(0.0);
        if spread_pct > rules.max_spread_pct {
            return Ok(false);
        }

        // Check volume
        if data.volume_24h < rules.min_volume_threshold {
            return Ok(false);
        }

        // Additional validation rules would go here
        Ok(true)
    }

    async fn update_metrics(&self, valid: bool) {
        let mut metrics = self.metrics.write().await;
        metrics.total_updates += 1;
        
        if valid {
            metrics.valid_updates += 1;
        } else {
            metrics.invalid_updates += 1;
        }

        metrics.error_rate = (metrics.invalid_updates as f64 / metrics.total_updates as f64) * 100.0;
    }

    async fn simulate_market_data(symbol: &str) {
        // This is a placeholder for actual data fetching
        // In a real implementation, this would connect to actual exchanges
        trace!("Simulating market data for {}", symbol);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_feed_creation() {
        let config = MarketDataFeedConfig::default();
        let feed = MarketDataFeed::new(config);
        
        let state = feed.state().await;
        assert!(matches!(state, FeedState::Starting));
    }

    #[tokio::test]
    async fn test_subscription() {
        let config = MarketDataFeedConfig::default();
        let feed = MarketDataFeed::new(config);
        
        let receiver = feed.subscribe("BTC/USD").await;
        assert!(receiver.is_ok());
        
        let subscriptions = feed.subscriptions().await;
        assert!(subscriptions.contains_key("BTC/USD"));
    }
}